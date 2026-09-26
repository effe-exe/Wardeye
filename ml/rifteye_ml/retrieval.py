# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Gallery search and the metrics the evaluation suite reports.

Search is brute force: a few thousand printings times a few hundred dimensions is
under a million multiply-adds per query, so no approximate index is needed or wanted.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from .encoders import Encoder

ROTATIONS = (0, 90, 180, 270)


def topk(queries: np.ndarray, gallery: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the k nearest gallery rows (inputs L2-normalised)."""
    sims = queries @ gallery.T
    k = min(k, gallery.shape[0])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


def search(encoder: Encoder, gallery: np.ndarray, queries: Sequence[Image.Image], k: int = 5,
           rotation_invariant: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k over the gallery. With `rotation_invariant`, each query is embedded at all four
    90° rotations (one batch) and each gallery row keeps its best score, which also
    returns the rotation that matched. That is how exhausted and opponent-side cards are read."""
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
