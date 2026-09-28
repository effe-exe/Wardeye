# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Quads and tiles for the amodal card detector.

The detector predicts the four corners of every card, in *image* order: the corner up and to the left
of the card's centre first, then clockwise on screen. It does not say which way the card is printed;
the matcher already tries all four turns (ARCHITECTURE §3.3), and a sleeve back has no printed top.
"""
from __future__ import annotations

import numpy as np

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def canonical_quad(q) -> np.ndarray:
    """The four corners of a card, the one up and left of its centre first, then clockwise on screen."""
    q = np.asarray(q, np.float64).reshape(4, 2)
    d = q - q.mean(axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])  # y points down, so clockwise on screen is increasing angle
    first = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang + 3 * np.pi / 4))))))
    return q[np.argsort((ang - ang[first]) % (2 * np.pi), kind="stable")]


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _signed(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject: list, clip: np.ndarray) -> list:
    """Sutherland-Hodgman: the part of a polygon inside a convex polygon of positive orientation."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def overlap_area(a, b) -> float:
    """The area two convex quads share (any corner order that goes round the quad)."""
    a = np.asarray(a, np.float64).reshape(-1, 2)
    b = np.asarray(b, np.float64).reshape(-1, 2)
    if _signed(a) < 0:
        a = a[::-1]
    if _signed(b) < 0:
        b = b[::-1]
    if (a[:, 0].max() <= b[:, 0].min() or b[:, 0].max() <= a[:, 0].min()
            or a[:, 1].max() <= b[:, 1].min() or b[:, 1].max() <= a[:, 1].min()):
        return 0.0
    inter = _clip([tuple(p) for p in a], b)
    return polygon_area(np.array(inter)) if len(inter) >= 3 else 0.0


def quad_iou(a, b) -> float:
    """Intersection over union of two convex quads (any corner order that goes round the quad)."""
    ia = overlap_area(a, b)
    if ia <= 0:
        return 0.0
    union = polygon_area(np.asarray(a, np.float64).reshape(-1, 2)) + polygon_area(np.asarray(b, np.float64).reshape(-1, 2)) - ia
    return ia / union if union > 0 else 0.0


def tile_origins(size: float, tile: int, overlap: float) -> list[int]:
    """Left (or top) edges of tiles of side `tile` that cover `size` pixels with at least `overlap` shared."""
    if size <= tile:
        return [0]
    n = int(np.ceil((size - tile) / (tile * (1 - overlap)) - 1e-9)) + 1
    return [int(round(v)) for v in np.linspace(0, size - tile, n)]
