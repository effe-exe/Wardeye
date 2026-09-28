# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""A layout found from the footage itself, for a broadcast no preset describes (`--layout auto`).

A layout (`layouts.py`) is where the table camera's picture sits in the frame and how long a card is
there. `table_window` finds the picture's borders first: a broadcast that puts panels beside the table
draws them as long straight edges in the same place in every frame (18.1% and 81.5% of the width at Los
Angeles and Barcelona), and the panels, player cams and hand lists outside them are never looked at.
Inside the borders the mat decides: the colour that fills the middle, grown over the cards lying on
it, and its bounding box (Shenyang's red mat, full screen, has no side borders). `card_size` runs the
detector over that window at a few candidate card sizes and keeps the one its confident boxes agree on.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

from ..matcrops import mat_colour
from .layouts import Layout

W, H = 480, 270                                  # the thumbnail the borders are found on
SIZES = (80, 100, 125, 155, 190, 235)            # candidate card long sides at 1080p


def borders(frames: Sequence[np.ndarray], edge: int = 18, keep: float = 0.9) -> tuple[float, float, float, float]:
    """The camera picture's edges as fractions (x0, y0, x1, y1): the long straight edges nearest the middle
    that every frame shares, or the frame's own edges where there are none."""
    g = np.stack([np.asarray(Image.fromarray(f).convert("L").resize((W, H), Image.BOX), np.float32) for f in frames])
    cols = (np.abs(np.diff(g, axis=2)) > edge).mean(axis=1).min(axis=0)  # per column, in the frame where it is weakest
    rows = (np.abs(np.diff(g, axis=1)) > edge).mean(axis=2).min(axis=0)
    left = [x for x in range(len(cols)) if cols[x] >= keep and x < 0.45 * W]
    right = [x for x in range(len(cols)) if cols[x] >= keep and x > 0.55 * W]
    top = [y for y in range(len(rows)) if rows[y] >= keep and y < 0.45 * H]
    bottom = [y for y in range(len(rows)) if rows[y] >= keep and y > 0.55 * H]
    return ((max(left) + 1) / W if left else 0.0, (max(top) + 1) / H if top else 0.0,
            min(right) / W if right else 1.0, min(bottom) / H if bottom else 1.0)


def table_window(frames: Sequence[np.ndarray], tol: int = 45) -> tuple[tuple[float, float, float, float], tuple[int, int, int], float] | None:
    """The table window as fractions of the frame, the mat's colour and the share of the window it fills;
    None when no one colour fills the middle of the picture (not a table shot)."""
    from scipy import ndimage

    bx0, by0, bx1, by1 = borders(frames)
    img = np.median(np.stack([np.asarray(Image.fromarray(f).resize((W, H), Image.BOX), np.int16) for f in frames]), axis=0)
    X0, Y0, X1, Y1 = round(bx0 * W), round(by0 * H), round(bx1 * W), round(by1 * H)
    pic = img[Y0:Y1, X0:X1]
    ph, pw = pic.shape[:2]
    mid = pic[ph // 5: ph * 4 // 5, pw // 5: pw * 4 // 5]
    mat = mat_colour(mid.astype(np.uint8))
    near = np.abs(pic - mat).max(axis=2) < tol
    if near[ph // 5: ph * 4 // 5, pw // 5: pw * 4 // 5].mean() < 0.3:
        return None
    padded = np.pad(near, 8, mode="edge")  # so the closing does not eat the picture's own edges
    grown = ndimage.binary_fill_holes(ndimage.binary_closing(padded, structure=np.ones((15, 15))))[8:-8, 8:-8]  # cards join the mat
    lab, n = ndimage.label(grown)
    counts = np.bincount(lab[ph // 5: ph * 4 // 5, pw // 5: pw * 4 // 5].ravel(), minlength=n + 1)
    counts[0] = 0
    if n == 0 or counts.max() == 0:
        return None
    table = lab == int(counts.argmax())  # the mat and the cards lying on it
    ys, xs = np.nonzero(table)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # Trim the rows the table only touches (a HUD band runs the whole width), and the columns it hardly
    # reaches: cards lie along the mat's edges and partly off it, so a side column is kept at a quarter.
    while r1 - r0 > 10 and table[r0, c0:c1].mean() < 0.5:
        r0 += 1
    while r1 - r0 > 10 and table[r1 - 1, c0:c1].mean() < 0.5:
        r1 -= 1
    while c1 - c0 > 10 and table[r0:r1, c0].mean() < 0.25:
        c0 += 1
    while c1 - c0 > 10 and table[r0:r1, c1 - 1].mean() < 0.25:
        c1 -= 1
    margin = round(0.015 * W)  # a card on the mat's edge, inside the picture's borders
    c0, c1 = max(0, c0 - margin), min(pw, c1 + margin)
    x0, y0, x1, y1 = (X0 + c0) / W, (Y0 + r0) / H, (X0 + c1) / W, (Y0 + r1) / H
    if (x1 - x0) * (y1 - y0) < 0.2:
        return None
    share = float(near[r0:r1, c0:c1].mean())
    return tuple(round(float(v), 3) for v in (x0, y0, x1, y1)), tuple(int(v) for v in mat), share


def card_size(detect, frames: Sequence[np.ndarray], window: tuple[float, float, float, float]) -> float | None:
    """A card's long side at 1080p: of the candidate sizes, the one at which the detector finds the most
    confident cards whose own size agrees with it; then the median size of those cards."""
    best, best_n, best_longs = None, 0, []
    for px in SIZES:
        longs = []
        for f in frames:
            h, w = f.shape[:2]
            box = (window[0] * w, window[1] * h, window[2] * w, window[3] * h)
            for d in detect(f, box, px * h / 1080):
                if d["score"] < 0.6:
                    continue
                q = np.asarray(d["quad"], np.float64).reshape(4, 2)
                side = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])) * 1080 / h
                if abs(side / px - 1) < 0.35:
                    longs.append(side)
        if len(longs) > best_n:
            best, best_n, best_longs = px, len(longs), longs
    return float(np.median(best_longs)) if best is not None and best_n >= 5 else None


def card_size_finder(frames: Sequence[np.ndarray], window: tuple[float, float, float, float],
                     mat: tuple[int, int, int]) -> float | None:
    """Without the detector: the candidate size at which the bootstrap finder sees the most isolated cards,
    then their median size."""
    from ..matcrops import find_cards, notmat_mask

    best: list[float] = []
    for px in SIZES:
        longs = []
        for f in frames:
            h, w = f.shape[:2]
            roi = f[round(window[1] * h):round(window[3] * h), round(window[0] * w):round(window[2] * w)]
            longs += [b.long_px * 1080 / h for b in find_cards(roi, px * h / 1080, tol=0.15,
                                                                  mask=notmat_mask(roi, np.asarray(mat, np.int16), 45))]
        if len(longs) > len(best):
            best = longs
    return float(np.median(best)) if len(best) >= 3 else None


def auto_layout(frames: Sequence[np.ndarray], detect=None) -> Layout | None:
    """A layout for these frames of the table camera, or None when they do not show a table (or no card
    on it yet)."""
    found = table_window(frames)
    if found is None:
        return None
    window, mat, share = found
    px = card_size(detect, frames, window) if detect is not None else card_size_finder(frames, window, mat)
    if px is None:
        return None
    return Layout("auto", "this broadcast", window, round(px, 1), mat=mat, mat_share=round(0.8 * share, 2))
