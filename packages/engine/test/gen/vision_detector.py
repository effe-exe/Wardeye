# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The Python reference for vision-detector.test.ts: Detector.detect_tiles (RF-DETR's keypoint PostProcess, twice),
merge_tiles and live/pipeline's detector_boxes on seeded synthetic head outputs and detections.

    python packages/engine/test/gen/vision_detector.py    # writes packages/engine/test/vectors/vision-detector.json

No pictures, no weights: the head's outputs are made up from a seed in the ranges the trained head gives (a few
confident queries among many unlikely ones, corners near the boxes, the precision numbers the real tiles show),
with exact ties and scores next to the threshold. The postprocess is RF-DETR's own, as the checkpoint builds it
(num_select 100, four keypoints a class, trace_alpha 0.2), run by Detector.detect_tiles' own code on a stand-in
net. Raw outputs are stored as base64 float32 (little-endian), the rest as Python's repr.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rfdetr.models.postprocess import PostProcess

from rifteye_ml.detect.model import TILE, Detector, merge_tiles
from rifteye_ml.live.pipeline import detector_boxes, drop_nested, drop_straddlers
from rifteye_ml.matcrops import CardBox

OUT = Path(__file__).resolve().parents[1] / "vectors" / "vision-detector.json"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, "<f4").tobytes()).decode()


def stand_in(outputs: dict) -> Detector:
    """Detector.detect_tiles' own code, with a net that returns `outputs` and the checkpoint's postprocess."""
    det = Detector.__new__(Detector)
    det.torch = torch
    det.device = torch.device("cpu")
    det.names = ["card", "card_back"]
    det.post = PostProcess(num_select=100, num_keypoints_per_class=[4, 4], trace_alpha=0.2)

    class Rf:
        means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    det.rf = Rf()
    det.net = lambda batch: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in outputs.items()}
    return det


def head(rng: np.random.Generator, n: int) -> dict:
    q = 100
    logits = rng.normal(-7, 1.5, (n, q, 2)).astype(np.float32)
    for b in range(n):
        hot = rng.choice(q, 45, replace=False)
        logits[b, hot, rng.integers(0, 2, 45)] = rng.uniform(-2.5, 9, 45).astype(np.float32)
        logits[b, hot[:6], 0] = logits[b, hot[6], 0]  # exact ties: the lower index goes first
        logits[b, hot[7:9], 1] = np.float32(-0.8472979)  # sigmoid near 0.3, the threshold
    cxcy = rng.uniform(0, 1, (n, q, 2))
    wh = rng.uniform(0.05, 0.3, (n, q, 2))
    wh[:, :3] = -0.01  # a negative size is clamped to 0
    boxes = np.concatenate([cxcy, wh], axis=-1).astype(np.float32)
    boxes[:, 3:6, :2] = [[-0.02, 1.03]]  # boxes past the tile are clamped to it
    kp = np.zeros((n, q, 8, 8), np.float32)
    corner = cxcy[:, :, None, :] + rng.normal(0, 0.06, (n, q, 8, 2))
    kp[..., 0:2] = corner
    kp[..., 2] = rng.normal(3, 3, (n, q, 8))    # found logit
    kp[..., 3] = rng.normal(2, 4, (n, q, 8))    # visible logit
    kp[..., 4] = rng.normal(5, 1, (n, q, 8))    # log_l11
    kp[..., 5] = rng.normal(0, 30, (n, q, 8))   # l21
    kp[..., 5][:, :2] = 0.0                     # |l21| clamped to 1e-12
    kp[..., 6] = rng.normal(5, 1, (n, q, 8))    # log_l22
    kp[..., 7] = rng.normal(0, 2, (n, q, 8))
    return {"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": kp}


def tile_json(d: dict) -> dict:
    return {"cls": d["cls"], "score": d["score"], "box": d["box"], "quad": np.asarray(d["quad"]).tolist(), "found": d["found"], "visible": d["visible"]}


def cardbox(cx, cy, long, short, deg, score, vis, back=False) -> CardBox:
    b = CardBox((np.float64(cx), np.float64(cy)), np.float64(long), np.float64(short), float(deg), 1.0)
    b.back, b.score, b.vis = back, score, vis
    return b


def main() -> None:
    rng = np.random.default_rng(576)
    out: dict = {"note": "made by test/gen/vision_detector.py; synthetic, seeded"}

    # the postprocess: a batch of two tiles, at detect_tiles' threshold and a lower one
    outputs = head(rng, 2)
    det = stand_in(outputs)
    black = Image.new("RGB", (TILE, TILE))
    out["head"] = {k: b64(v) for k, v in outputs.items()}
    # at a low threshold: the test filters it at 0.3 for detect_tiles' own
    out["decoded"] = {"threshold": 0.05, "tiles": [[tile_json(d) for d in ds] for ds in det.detect_tiles([black, black], 0.05)]}

    # merge_tiles: a 2 x 2 tiling of a 1000 x 800 window; each card seen by the tiles that hold it, with the jitter
    # of two different crops, cut-off copies at the shared edges, and scores that tie once rounded
    origins, size, scale, offset = [(0, 0), (424, 0), (0, 224), (424, 224)], (1000, 800), 70 / 155, (365, 65)
    per_tile = [[] for _ in origins]
    for c in range(30):
        cx, cy = rng.uniform(40, 960), rng.uniform(40, 760)
        long, short, ang = rng.uniform(60, 75), rng.uniform(42, 52), rng.uniform(0, np.pi)
        u, v = np.array([np.cos(ang), np.sin(ang)]) * long / 2, np.array([-np.sin(ang), np.cos(ang)]) * short / 2
        quad = np.array([[cx, cy] + u + v, [cx, cy] + u - v, [cx, cy] - u - v, [cx, cy] - u + v])
        score = float(np.float32(rng.choice([0.91234, 0.91236, rng.uniform(0.3, 1)])))
        cls = "card_back" if c % 7 == 0 else "card"
        for i, (tx, ty) in enumerate(origins):
            q = (quad - [tx, ty] + rng.normal(0, 0.4, (4, 2))).astype(np.float32)
            lo, hi = q.min(axis=0), q.max(axis=0)
            if hi[0] < 0 or hi[1] < 0 or lo[0] > TILE or lo[1] > TILE:
                continue
            box = np.clip(np.concatenate([lo, hi]), 0, TILE).astype(np.float32)
            per_tile[i].append({"cls": cls, "score": float(np.float32(score + rng.normal(0, 0.002))), "box": box.tolist(), "quad": q[rng.permutation(4)],
                                "found": [float(np.float32(v)) for v in rng.uniform(0, 1, 4)],
                                "visible": [float(np.float32(v)) for v in rng.uniform(0, 1, 4)]})
    merged = merge_tiles(per_tile, origins, size, scale, offset)
    out["merge"] = {"per_tile": [[tile_json(d) for d in ds] for ds in per_tile], "origins": origins, "size": size, "scale": scale,
                    "offset": offset, "out": merged}

    # detector_boxes on that, and on made-up neighbours: nested outlines (a toploader), a slip across two cards
    extra = []
    base = [(300.0, 300.0, 70.0, 50.0, 30.0)]
    for cx, cy, long, short, deg in base:
        for grow in (1.0, 1.1, 1.3):
            extra.append((cx + grow, cy, long * grow, short * grow, deg + grow, 0.9, 0.9))
    extra += [(500.0, 400.0, 70.0, 50.0, 0.0, 0.95, 0.9), (570.0, 400.0, 70.0, 50.0, 0.0, 0.94, 0.9), (535.0, 400.0, 70.0, 50.0, 0.0, 0.6, 0.8),
              (700.0, 400.0, 70.0, 50.0, 90.0, 0.95, 0.9), (700.0, 450.0, 70.0, 50.0, 90.0, 0.94, 0.9), (700.0, 425.0, 70.0, 50.0, 90.0, 0.6, 0.3)]
    boxes = [cardbox(*e) for e in extra]
    as_json = lambda bs: [{"centre": [float(b.centre[0]), float(b.centre[1])], "long_px": float(b.long_px), "short_px": float(b.short_px),  # noqa: E731
                           "angle_deg": float(b.angle_deg), "fill": float(b.fill), "back": bool(b.back), "score": float(b.score), "vis": float(b.vis)} for b in bs]
    out["boxes"] = {"min_score": 0.4, "out": as_json(detector_boxes(merged, 0.4)), "extra": as_json(boxes),
                    "nested": as_json(drop_nested(boxes)), "straddlers": as_json(drop_straddlers(drop_nested(boxes)))}
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB; decoded {[len(t) for t in out['decoded']['tiles']]}, merged {len(merged)}, "
          f"boxes {len(out['boxes']['out'])}, nested {len(out['boxes']['nested'])}, straddlers {len(out['boxes']['straddlers'])}")


if __name__ == "__main__":
    main()
