# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""A classical bootstrap detector: isolated cards on a known playmat, for M0 labeling.

This is not the product detector (that is a trained, amodal model: ARCHITECTURE §3.2 and
§3.8). It finds cards whose dark border or sleeve stands out from the mat, fits a rotated
rectangle, rejects blobs that are not a single card (stacks, hands, merged neighbours), and
writes an upright crop per card. Stacked and light-bordered cards are missed by design;
the crops seed the real test set, and a person labels them (`rifteye_ml.label`).

The default border rule, dark and not red, fits the red mat of the M0 reference broadcast:
R < 90, R − G < 45 and R − B < 45. For other mats, `--mask notmat` takes everything far from
the mat's colour instead (estimated from the table area, or given with `--mat R,G,B`).

    python -m rifteye_ml.matcrops --frames frames/seg-*/ --table 0.17,0.09,0.86,0.884 \\
        --long 131 --every 24 --out real-crops/
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass
class CardBox:
    centre: tuple[float, float]  # frame px
    long_px: float
    short_px: float
    angle_deg: float             # direction of the long side, degrees from the x axis
    fill: float                  # blob area / rectangle area

    @property
    def aspect(self) -> float:
        return self.long_px / max(1e-6, self.short_px)


def border_mask(rgb: np.ndarray, r_max: int = 90, rg: int = 45, rb: int = 45) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r < r_max) & (r - g < rg) & (r - b < rb)


def mat_colour(rgb: np.ndarray) -> np.ndarray:
    """The playmat's colour: the most common colour of the table area (coarse 3-D histogram peak)."""
    q = (rgb.reshape(-1, 3) // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    peak = np.bincount(keys).argmax()
    sel = rgb.reshape(-1, 3)[keys == peak]
    return np.median(sel, axis=0).astype(np.int16)


def notmat_mask(rgb: np.ndarray, mat: np.ndarray, tol: int = 60) -> np.ndarray:
    """Pixels far from the mat colour (max channel difference over `tol`): cards, whatever their border."""
    return np.abs(rgb.astype(np.int16) - mat.astype(np.int16)).max(axis=2) > tol


def min_area_rect(points: np.ndarray) -> tuple[tuple[float, float], float, float, float]:
    """Minimum-area rectangle of 2-D points (x, y): centre, long side, short side, long-side angle."""
    from scipy.spatial import ConvexHull

    hull = points[ConvexHull(points).vertices]
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        theta = math.atan2(e[1], e[0])
        c, s = math.cos(theta), math.sin(theta)
        rot = hull @ np.array([[c, -s], [s, c]])  # into the edge's frame
        lo, hi = rot.min(axis=0), rot.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, theta, lo, hi)
    _, theta, lo, hi = best
    c, s = math.cos(theta), math.sin(theta)
    mid = (lo + hi) / 2
    centre = (mid[0] * c - mid[1] * s, mid[0] * s + mid[1] * c)
    w, h = hi - lo
    angle = math.degrees(theta) if w >= h else math.degrees(theta) + 90
    return (float(centre[0]), float(centre[1])), float(max(w, h)), float(min(w, h)), (angle + 180) % 180


def find_cards(rgb: np.ndarray, long_px: float, tol: float = 0.12, aspect: tuple[float, float] = (1.33, 1.46),
               min_fill: float = 0.88, mask: np.ndarray | None = None) -> list[CardBox]:
    """Isolated single cards whose long side is within `tol` of `long_px`. `mask` replaces the
    default dark-border rule (see `notmat_mask`)."""
    from scipy import ndimage

    mask = ndimage.binary_closing(border_mask(rgb) if mask is None else mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    labels, n = ndimage.label(mask)
    boxes: list[CardBox] = []
    lo_area, hi_area = (long_px * (1 - tol)) ** 2 / 1.46 * 0.8, (long_px * (1 + tol)) ** 2 / 1.33 * 1.1
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        blob = labels[sl] == i
        area = int(blob.sum())
        if not lo_area <= area <= hi_area:
            continue
        ys, xs = np.nonzero(blob)
        pts = np.stack([xs + sl[1].start, ys + sl[0].start], axis=1).astype(np.float64)
        centre, lng, sht, ang = min_area_rect(pts)
        box = CardBox(centre, lng, sht, ang, area / max(1.0, lng * sht))
        if abs(lng / long_px - 1) <= tol and aspect[0] <= box.aspect <= aspect[1] and box.fill >= min_fill:
            boxes.append(box)
    return boxes


def upright_crop(frame: Image.Image, box: CardBox, pad: float = 0.0) -> Image.Image:
    """Rotate the frame about the card's centre so the long side is vertical, and crop it."""
    cx, cy = box.centre
    # PIL rotates counter-clockwise for positive angles, with y pointing down.
    rot = frame.rotate(box.angle_deg - 90, resample=Image.BICUBIC, center=(cx, cy))
    w, h = box.short_px * (1 + pad), box.long_px * (1 + pad)
    return rot.crop((round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)))


def frames_in(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        q = Path(p)
        out += sorted(q.glob("*.jpg")) if q.is_dir() else [q]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.matcrops", description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", nargs="+", required=True, help="frame files or folders of .jpg")
    ap.add_argument("--only", help="optional text file of frame names to keep (e.g. overhead frames)")
    ap.add_argument("--table", default="0,0,1,1", help="x0,y0,x1,y1 fractions: the mat area free of overlays")
    ap.add_argument("--long", type=float, required=True, help="expected long side of a card in px")
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--mask", choices=["border", "notmat"], default="border",
                    help="border: dark, not-red card borders (the M0 red mat); notmat: anything unlike the mat")
    ap.add_argument("--mat", help="R,G,B of the bare mat for --mask notmat (default: estimated per frame)")
    ap.add_argument("--mat-tol", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    keep = set(Path(a.only).read_text().split()) if a.only else None
    files = [f for f in frames_in(a.frames) if keep is None or f.name in keep][:: max(1, a.every)]
    fx0, fy0, fx1, fy1 = (float(v) for v in a.table.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for f in files:
        frame = Image.open(f).convert("RGB")
        W, H = frame.size
        x0, y0 = round(fx0 * W), round(fy0 * H)
        roi = np.asarray(frame)[y0:round(fy1 * H), x0:round(fx1 * W)]
        mask = None
        if a.mask == "notmat":
            mat = np.array([int(v) for v in a.mat.split(",")], np.int16) if a.mat else mat_colour(roi)
            mask = notmat_mask(roi, mat, a.mat_tol)
        for k, b in enumerate(find_cards(roi, a.long, mask=mask)):
            b.centre = (b.centre[0] + x0, b.centre[1] + y0)
            name = f"{f.stem}_{k:02d}.png"
            upright_crop(frame, b, a.pad).save(out / name)
            meta.append({"file": name, "frame": str(f), **asdict(b)})
    (out / "crops.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(files)} frames, {len(meta)} crops -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
