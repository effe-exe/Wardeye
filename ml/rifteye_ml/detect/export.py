# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Synthetic boards -> a COCO keypoint dataset of square tiles for the amodal card detector.

Each frame's camera window (the overhead ROI) is scaled so its cards are a random 45-110 px on the long
side, as the detector will see them (ARCHITECTURE §3.2), and cut into overlapping square tiles.
Every card that shows at least a little of itself in a tile is a target:

* the box is the card's *full* quad, clipped to the tile, even where other cards cover it;
* the four keypoints are its corners in image order (`geometry.CORNERS`): 2 when the corner shows,
  1 when a card, hand or die covers it, 0 when it lies outside the tile;
* the class is `card` (face up) or `card_back`. Nothing about a face-down card's identity is written.

The layout is Roboflow's (`train/`, `valid/`, each with `_annotations.coco.json`), which RF-DETR reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import CORNERS, canonical_quad, polygon_area, tile_origins

CLASSES = ("card", "card_back")
TILE = 576  # RF-DETR keypoint's input side; divisible by its patch size times windows (12 x 2)


def categories() -> list[dict]:
    skeleton = [[1, 2], [2, 3], [3, 4], [4, 1]]
    return [{"id": i + 1, "name": n, "supercategory": "card", "keypoints": list(CORNERS), "skeleton": skeleton}
            for i, n in enumerate(CLASSES)]


def tile_targets(cards: list[dict], ids: np.ndarray, window, scale: float, origin, tile: int = TILE,
                 min_visible: float = 0.08, min_px: float = 100.0) -> list[dict]:
    """COCO annotations (without ids) for one tile of a frame.

    `window` is the camera window in frame pixels, `scale` maps it to tile pixels and `origin` is the
    tile's top-left in the scaled window. `ids` is the frame's id map (card id + 1 where uppermost).
    """
    wx0, wy0, wx1, wy1 = window
    tx, ty = origin
    fx0, fy0 = wx0 + tx / scale, wy0 + ty / scale  # the tile, in frame pixels
    fx1, fy1 = min(wx1, fx0 + tile / scale), min(wy1, fy0 + tile / scale)
    sub = ids[int(fy0):int(np.ceil(fy1)), int(fx0):int(np.ceil(fx1))]
    counts = np.bincount(sub.ravel(), minlength=65536)
    wide, high = (wx1 - wx0) * scale - tx, (wy1 - wy0) * scale - ty  # where the window ends inside the tile
    out = []
    for c in cards:
        if c["kind"] not in CLASSES:
            continue
        seen = counts[c["id"] + 1] * scale * scale
        qf = canonical_quad(c["quad"])
        qt = (qf - [wx0, wy0]) * scale - [tx, ty]
        area = polygon_area(qt)
        if area <= 0 or seen < min_px or seen / area < min_visible:
            continue
        x0, y0 = np.clip(qt.min(axis=0), 0, [min(tile, wide), min(tile, high)])
        x1, y1 = np.clip(qt.max(axis=0), 0, [min(tile, wide), min(tile, high)])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        centre = qf.mean(axis=0)
        kps = []
        for (px, py), corner in zip(qt, qf):
            if not (0 <= px < min(tile, wide) and 0 <= py < min(tile, high)):
                kps += [0.0, 0.0, 0]
                continue
            ix, iy = np.round(corner + 0.1 * (centre - corner)).astype(int)  # just inside the rounded corner
            shows = 0 <= iy < ids.shape[0] and 0 <= ix < ids.shape[1] and ids[iy, ix] == c["id"] + 1
            kps += [round(float(px), 2), round(float(py), 2), 2 if shows else 1]
        out.append({"category_id": CLASSES.index(c["kind"]) + 1,
                    "bbox": [round(float(x0), 2), round(float(y0), 2), round(float(x1 - x0), 2), round(float(y1 - y0), 2)],
                    "area": round(area, 1), "iscrowd": 0, "keypoints": kps, "num_keypoints": sum(1 for v in kps[2::3] if v)})
    return out


def export_run(run: Path | list[Path], out: Path, tile: int = TILE, target=(45.0, 110.0), overlap: float = 0.2,
               val_every: int = 10, quality: int = 95, seed: int = 0, min_visible: float = 0.08, scales: int = 1) -> dict:
    """Write the tiles of one or more synthetic runs; boards with `board % val_every == val_every - 1` go to
    `valid/`. Each frame is cut at `scales` random scales, so one board teaches several card sizes."""
    runs = [Path(r) for r in (run if isinstance(run, (list, tuple)) else [run])]
    out = Path(out)
    data = {s: {"images": [], "annotations": [], "categories": categories()} for s in ("train", "valid")}
    for s in data:
        (out / s).mkdir(parents=True, exist_ok=True)
    for ri, rdir in enumerate(runs):
        with open(rdir / "annotations.jsonl", encoding="utf-8") as f:
            anns = [json.loads(line) for line in f]
        for fi, a in enumerate(anns):
            split = "valid" if a["board"] % val_every == val_every - 1 else "train"
            rng = np.random.default_rng([seed, ri, fi])
            wx0, wy0, wx1, wy1 = a["window"]
            with Image.open(rdir / a["image"]) as im:
                full = im.convert("RGB").crop((wx0, wy0, wx1, wy1))
            with Image.open(rdir / a["ids"]) as im:
                ids = np.asarray(im).astype(np.int32)
            for k in range(scales):
                scale = float(np.exp(rng.uniform(np.log(target[0]), np.log(target[1])))) / a["card_px"]
                sw, sh = max(1, round((wx1 - wx0) * scale)), max(1, round((wy1 - wy0) * scale))
                win = full.resize((sw, sh), Image.BICUBIC)
                for ty in tile_origins(sh, tile, overlap):
                    for tx in tile_origins(sw, tile, overlap):
                        d = data[split]
                        image_id = len(d["images"]) + 1
                        name = f"r{ri}_{Path(a['image']).stem}_s{k}_{tx}_{ty}.jpg"
                        win.crop((tx, ty, tx + tile, ty + tile)).save(out / split / name, quality=quality)  # black past the window
                        d["images"].append({"id": image_id, "file_name": name, "width": tile, "height": tile,
                                            "frame": str(rdir / a["image"]), "offset": [tx, ty], "scale": round(scale, 5)})
                        for t in tile_targets(a["cards"], ids, a["window"], scale, (tx, ty), tile, min_visible):
                            d["annotations"].append({"id": len(d["annotations"]) + 1, "image_id": image_id, **t})
    stats = {}
    for s, d in data.items():
        with open(out / s / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(d, f)
        kp = np.array([v for t in d["annotations"] for v in t["keypoints"][2::3]], int)
        stats[s] = {"tiles": len(d["images"]), "cards": len(d["annotations"]),
                    "backs": sum(t["category_id"] == 2 for t in d["annotations"]),
                    "corners_covered": round(float((kp == 1).mean()), 3) if kp.size else 0.0}
    return stats
