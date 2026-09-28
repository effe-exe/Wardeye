# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Train and run the amodal card detector: RF-DETR keypoint (Apache-2.0) with the four card corners.

`rfdetr` is imported lazily, so the rest of the package works without it (`pip install -e '.[detect]'`).
The weights are RF-DETR's keypoint preview, Apache-2.0 like the code ([03 §3.2](../../../docs/research/03-models-and-licensing.md)).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from .export import CLASSES, TILE
from .geometry import canonical_quad, quad_iou, tile_origins


def train(dataset: Path, out: Path, epochs: int = 12, batch_size: int = 8, grad_accum: int = 2, lr: float = 1e-4,
          device: str | None = None, num_workers: int = 4, seed: int = 0, resume: str | None = None) -> None:
    """Fine-tune RF-DETR keypoint. `last.ckpt` (about 650 MB with the optimiser) is rewritten every epoch
    and is what `resume` takes; the best weights go to `checkpoint_best_total.pth` (about 160 MB)."""
    from rfdetr import RFDETRKeypointPreview

    kw = dict(dataset_dir=str(dataset), output_dir=str(out), epochs=epochs, batch_size=batch_size,
              grad_accum_steps=grad_accum, lr=lr, num_workers=num_workers, seed=seed, resolution=TILE,
              checkpoint_interval=10_000,  # no numbered snapshots; they are the size of last.ckpt each
              # RF-DETR's default steps the rate down only after 100 epochs; a short fine-tune wants it to decay
              lr_scheduler="cosine", lr_scheduler_kwargs={"min_factor": 0.05}, warmup_epochs=0.3,
              tensorboard=importlib.util.find_spec("tensorboard") is not None)
    if device:
        kw["device"] = device
    if resume:
        kw["resume"] = resume
    # The head keeps the pretrained checkpoint's size, num_classes + 1 = 2 logits, whatever num_classes is
    # asked for. RF-DETR's default loss (IA-BCE) trains every logit as a class of its own, so labels 0
    # (card) and 1 (card_back) both learn; only its naming calls logit 1 "__background__".
    RFDETRKeypointPreview().train(**kw)


def pack(checkpoint: str | Path, out: str | Path) -> int:
    """Keep only what inference needs, in float16: about half the size of `checkpoint_best_total.pth`.
    `Detector` loads the result like the original (the weights go back to float32 on load)."""
    import torch

    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    small = {k: ck[k] for k in ("args", "model_name", "rfdetr_version") if k in ck}
    small["model"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
    torch.save(small, str(out))
    return Path(out).stat().st_size


class Detector:
    """A trained checkpoint, run over square tiles of a frame's camera window."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch
        from rfdetr.detr import RFDETR

        kw = {"device": device} if device else {}
        self.rf = RFDETR.from_checkpoint(str(checkpoint), **kw)
        # from_checkpoint leaves the weights on the CPU even when asked for CUDA (rfdetr 1.11.0), so move
        # them, and take the device from the weights so the inputs always go where the model is.
        want = torch.device(device) if device else self.rf.model.device
        self.net = self.rf.model.model.to(want).eval()
        self.post = self.rf.model.postprocess
        self.device = next(self.net.parameters()).device
        self.names = [str(n) for n in (self.rf.class_names or [])]
        self.torch = torch

    def _class_name(self, label: int) -> str | None:
        """`card`, `card_back`, or None for the head's no-object slot."""
        if not 0 <= label < len(CLASSES):
            return None
        name = self.names[label] if label < len(self.names) else CLASSES[label]
        return name if name in CLASSES else CLASSES[label]

    def detect_tiles(self, tiles: list[Image.Image], threshold: float = 0.3) -> list[list[dict]]:
        """Detections per tile, in tile pixels: class, score, box, corners, and per corner the chance it
        was findable (inside the tile) and visible (not covered)."""
        torch = self.torch
        import torchvision.transforms.functional as F

        batch = torch.stack([F.to_tensor(t.convert("RGB")) for t in tiles])
        batch = F.normalize(batch, self.rf.means, self.rf.stds).to(self.device)
        sizes = torch.tensor([[t.height, t.width] for t in tiles], device=self.device)
        with torch.inference_mode():
            pred = self.net(batch)
            found = self.post(pred, target_sizes=sizes, score_threshold=threshold)
            kp = pred["pred_keypoints"].clone()
            kp[..., 2] = kp[..., 3]  # the same selection again, reading the visible logit instead
            shown = self.post({**pred, "pred_keypoints": kp}, target_sizes=sizes, score_threshold=threshold)
        out = []
        for f, s in zip(found, shown):
            keep = (f["scores"] > threshold).nonzero(as_tuple=True)[0].cpu().numpy()
            sc, lab = f["scores"].float().cpu().numpy(), f["labels"].cpu().numpy()
            box, k, v = f["boxes"].float().cpu().numpy(), f["keypoints"].float().cpu().numpy(), s["keypoints"].float().cpu().numpy()
            out.append([{"cls": self._class_name(int(lab[i])), "score": float(sc[i]), "box": box[i].tolist(),
                         "quad": k[i, :4, :2], "found": k[i, :4, 2].tolist(), "visible": v[i, :4, 2].tolist()}
                        for i in keep if self._class_name(int(lab[i]))])
        return out

    def detect(self, frame: Image.Image, window, card_px: float, target: float = 70.0, overlap: float = 0.2,
               threshold: float = 0.3, batch: int = 8) -> list[dict]:
        """Cards in one frame, in frame pixels. `window` is the camera window (the layout preset's ROI) and
        `card_px` the long side of a card there, so the window is scaled to put cards near `target` px."""
        wx0, wy0, wx1, wy1 = [int(round(v)) for v in window]
        scale = target / card_px
        win = frame.convert("RGB").crop((wx0, wy0, wx1, wy1))
        sw, sh = max(1, round((wx1 - wx0) * scale)), max(1, round((wy1 - wy0) * scale))
        win = win.resize((sw, sh), Image.BICUBIC)
        origins = [(tx, ty) for ty in tile_origins(sh, TILE, overlap) for tx in tile_origins(sw, TILE, overlap)]
        tiles = [win.crop((tx, ty, tx + TILE, ty + TILE)) for tx, ty in origins]
        per_tile = []
        for i in range(0, len(tiles), batch):
            per_tile += self.detect_tiles(tiles[i:i + batch], threshold)
        return merge_tiles(per_tile, origins, (sw, sh), scale, (wx0, wy0))


def merge_tiles(per_tile: list[list[dict]], origins, size, scale: float, offset, margin: float = 2.0,
                same: float = 0.8) -> list[dict]:
    """One list of detections for the window.

    Tiles overlap by more than a card, so every card that is not cut by the window lies whole in some
    tile. A detection touching an edge shared with another tile is that card's cut-off duplicate and is
    dropped. Whole detections of the same card in two tiles are merged. Stacked cards overlap a lot
    by design (IoU 0.5-0.7 for a rune column), so nothing below `same` is suppressed.
    """
    sw, sh = size
    kept: list[dict] = []
    for dets, (tx, ty) in zip(per_tile, origins):
        inner = (tx > 0, ty > 0, tx + TILE < sw, ty + TILE < sh)  # left, top, right, bottom are shared edges
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            touches = (x0 <= margin, y0 <= margin, x1 >= TILE - margin, y1 >= TILE - margin)
            if any(t and s for t, s in zip(touches, inner)):
                continue
            q = canonical_quad((np.asarray(d["quad"]) + [tx, ty]) / scale + offset)
            kept.append({"cls": d["cls"], "score": round(d["score"], 4), "quad": q,
                         "found": [round(v, 3) for v in d["found"]], "visible": [round(v, 3) for v in d["visible"]],
                         "truncated": any(touches), "tile": (tx, ty)})
    kept.sort(key=lambda d: -d["score"])
    out: list[dict] = []
    for d in kept:
        if not any(o["tile"] != d["tile"] and o["cls"] == d["cls"] and quad_iou(o["quad"], d["quad"]) >= same for o in out):
            out.append(d)
    for d in out:
        d["quad"] = [round(float(v), 1) for v in d["quad"].ravel()]
        del d["tile"]
    return out
