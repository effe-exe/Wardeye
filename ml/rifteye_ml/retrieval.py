# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Gallery search and the metrics the evaluation suite reports.

Search is brute force: a few thousand printings times a few hundred dimensions is
under a million multiply-adds per query, so no approximate index is needed or wanted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence, Union

import numpy as np
from PIL import Image

from .encoders import Encoder

ROTATIONS = (0, 90, 180, 270)


def at_long_side(im: Image.Image, side: int) -> Image.Image:
    """A clean card as it lands on screen with its long side at `side` px (area-averaged, like optics)."""
    scale = side / max(im.size)
    return im.convert("RGB").resize((max(2, round(im.width * scale)), max(2, round(im.height * scale))), Image.BOX)


EDGES = ("top", "bottom", "left", "right")


def band(im: Image.Image, view: str) -> Image.Image:
    """The part of an upright card a stack leaves visible: 'top:0.25' is the top quarter.
    'full' (or '') is the whole card."""
    if view in ("", "full"):
        return im
    edge, _, frac = view.partition(":")
    f = float(frac)
    if edge not in EDGES or not 0 < f <= 1:
        raise ValueError(f"bad view {view!r}; expected <top|bottom|left|right>:<fraction>")
    w, h = im.size
    bw, bh = max(2, round(w * f)), max(2, round(h * f))
    box = {"top": (0, 0, w, bh), "bottom": (0, h - bh, w, h), "left": (0, 0, bw, h), "right": (w - bw, 0, w, h)}[edge]
    return im.crop(box)


class Pyramid:
    """Gallery embeddings of the clean art at a few on-screen sizes.

    A pretrained backbone sees a sharp 744 px card and a blurry 40 px crop as different
    images. Embedding the gallery at the size a card appears on screen closes most of that
    gap, and the detector already knows each card's size. Each query is searched against
    the level nearest its own long side, in log space."""

    def __init__(self, levels: dict[int, np.ndarray]):
        if not levels:
            raise ValueError("a pyramid needs at least one scale")
        self.levels = {int(x): v for x, v in levels.items()}
        self.scales = sorted(self.levels)
        self.rows = next(iter(self.levels.values())).shape[0]

    @classmethod
    def build(cls, encoder: Encoder, images: Sequence[Image.Image], scales: Sequence[int]) -> "Pyramid":
        return cls({int(x): encoder.embed([at_long_side(im, int(x)) for im in images]) for x in set(scales)})

    def level_for(self, long_side: int) -> int:
        return min(self.scales, key=lambda x: abs(np.log(x / max(1, long_side))))


Gallery = Union[np.ndarray, Pyramid]


def topk(queries: np.ndarray, gallery: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the k nearest gallery rows (inputs L2-normalised)."""
    sims = queries @ gallery.T
    k = min(k, gallery.shape[0])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


def search(encoder: Encoder, gallery: Gallery, queries: Sequence[Image.Image], k: int = 5,
           rotation_invariant: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k over the gallery. With `rotation_invariant`, each query is embedded at all four
    90° rotations (one batch) and each gallery row keeps its best score, which also
    returns the rotation that matched. That is how exhausted and opponent-side cards are read.
    A `Pyramid` gallery routes each query to the level nearest its size."""
    if isinstance(gallery, Pyramid):
        k = min(k, gallery.rows)
        idx = np.zeros((len(queries), k), np.int64)
        scores = np.zeros((len(queries), k), np.float32)
        rot = np.zeros((len(queries), k), np.int64)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, q in enumerate(queries):
            groups[gallery.level_for(max(q.size))].append(i)
        for level, members in groups.items():
            a, b, c = search(encoder, gallery.levels[level], [queries[i] for i in members], k, rotation_invariant)
            idx[members], scores[members], rot[members] = a, b, c
        return idx, scores, rot
    rots = ROTATIONS if rotation_invariant else (0,)
    batch = [q.rotate(r, expand=True) if r else q for q in queries for r in rots]
    emb = encoder.embed(batch).reshape(len(queries), len(rots), -1)
    sims = np.einsum("nrd,gd->nrg", emb, gallery)
    best_rot = sims.argmax(axis=1)
    best = sims.max(axis=1)
    k = min(k, gallery.shape[0])
    idx = np.argsort(-best, axis=1)[:, :k]
    scores = np.take_along_axis(best, idx, axis=1)
    rot = np.array(rots)[np.take_along_axis(best_rot, idx, axis=1)]
    return idx, scores, rot


def ranked_labels(idx: np.ndarray, labels: Sequence[str]) -> list[list[str]]:
    """Map gallery indices to labels, dropping duplicates while keeping rank order.
    With card-level labels this rolls printings up into cards (docs/ARCHITECTURE.md §4)."""
    out = []
    for row in idx:
        seen: list[str] = []
        for i in row:
            lab = labels[int(i)]
            if lab not in seen:
                seen.append(lab)
        out.append(seen)
    return out


def accuracy(ranked: list[list[str]], truth: Sequence[str]) -> dict[str, float]:
    n = len(truth)
    if n == 0:
        return {"top1": float("nan"), "top5": float("nan"), "n": 0}
    top1 = sum(1 for r, t in zip(ranked, truth) if r[:1] == [t])
    top5 = sum(1 for r, t in zip(ranked, truth) if t in r[:5])
    return {"top1": top1 / n, "top5": top5 / n, "n": n}
