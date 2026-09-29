# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The Python reference for vision-geometry.test.ts: detect/geometry.py and the float rules the detector's port
needs (np.linalg.norm, math.dist, round() on a numpy scalar, a fused multiply-add), on seeded synthetic inputs.

    python packages/engine/test/gen/vision_geometry.py    # writes packages/engine/test/vectors/vision-geometry.json

No pictures and nothing private: quads and numbers made up from a seed. The values are written as Python's
repr, which JSON.parse reads back as the same doubles, so the test can ask for the same bits.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from rifteye_ml.detect.geometry import canonical_quad, overlap_area, polygon_area, quad_iou, tile_origins

OUT = Path(__file__).resolve().parents[1] / "vectors" / "vision-geometry.json"


def rect(cx: float, cy: float, long: float, short: float, deg: float) -> np.ndarray:
    a = math.radians(deg)
    ux, uy = math.cos(a) * long / 2, math.sin(a) * long / 2
    vx, vy = -math.sin(a) * short / 2, math.cos(a) * short / 2
    return np.array([[cx + ux + vx, cy + uy + vy], [cx + ux - vx, cy + uy - vy], [cx - ux - vx, cy - uy - vy], [cx - ux + vx, cy - uy + vy]])


def main() -> None:
    rng = np.random.default_rng(20260929)
    out: dict = {"note": "made by test/gen/vision_geometry.py; synthetic, seeded"}

    # canonical_quad: rotated rectangles in a random corner order, float32 corners as the detector gives them,
    # a diamond (two corners as far from up-left as each other) and a square
    quads = []
    for _ in range(60):
        q = rect(*rng.uniform(0, 1000, 2), rng.uniform(40, 160), rng.uniform(30, 110), rng.uniform(0, 360))
        quads.append(q[rng.permutation(4)])
    for _ in range(20):
        quads.append(rng.uniform(0, 576, (4, 2)).astype(np.float32).astype(np.float64))
    quads += [np.array([[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]]), np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
              np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])]
    out["canonical_quad"] = [{"q": q.tolist(), "out": canonical_quad(q).tolist()} for q in quads]

    # polygon_area: 3 to 8 corners (a clipped quad has up to 8)
    polys = [rng.normal(500, 300, (int(rng.integers(3, 9)), 2)) for _ in range(80)]
    out["polygon_area"] = [{"p": p.tolist(), "out": polygon_area(p)} for p in polys]

    # overlap_area and quad_iou: neighbours of all kinds, both orientations, flat and nested lists
    pairs = []
    for _ in range(120):
        a = rect(*rng.uniform(200, 800, 2), rng.uniform(60, 160), rng.uniform(40, 110), rng.uniform(0, 180))
        c = a.mean(axis=0) + rng.normal(0, 60, 2)
        b = rect(*c, rng.uniform(60, 160), rng.uniform(40, 110), rng.uniform(0, 180))
        if rng.random() < 0.3:
            b = b[::-1]
        if rng.random() < 0.3:
            a = np.round(a, 1)
            b = np.round(b, 1)
        pairs.append((a, b))
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    pairs += [(sq, sq), (sq, sq + [10.0, 0.0]), (sq, sq + [5.0, 5.0]), (sq, sq * 0.5 + 2.5), (sq, sq + [20.0, 0.0]), (sq, sq[::-1] + [3.0, 0.0])]
    # as lists, as the callers pass them (a reversed numpy view would make np.dot skip BLAS: another summation order)
    out["overlap"] = [{"a": a.tolist(), "b": b.ravel().tolist(), "area": overlap_area(a.tolist(), b.ravel().tolist()),
                       "iou": quad_iou(a.tolist(), b.ravel().tolist())} for a, b in pairs]

    # tile_origins: every size up to four tiles, the ones whose origins end in .5 (rounded to even), float sizes
    sizes = list(range(560, 2400, 29)) + [576, 577, 1037, 1498, 1036.5, 700.25, 2881, 3000]
    out["tile_origins"] = [{"size": s, "tile": 576, "overlap": o, "out": tile_origins(s, 576, o)} for s in sizes for o in (0.2, 0.35)]

    # np.linalg.norm of two numbers (one fused multiply-add here), math.dist (CPython's own), numpy's round
    vec = [rng.normal(0, 80, 2) for _ in range(150)] + [np.round(rng.normal(0, 80, 2), 1) for _ in range(50)]
    out["norm2"] = [{"v": v.tolist(), "out": float(np.linalg.norm(v))} for v in vec]
    pts = [(rng.normal(900, 400, 2), rng.normal(900, 400, 2)) for _ in range(150)]
    out["dist"] = [{"p": p.tolist(), "q": q.tolist(), "out": math.dist(tuple(p), tuple(q))} for p, q in pts]
    xs = list(rng.uniform(-2000, 2000, 100)) + [4.35, 0.25, 0.75, -2.45, 1.15, 2.675, 1e-9]
    out["np_round1"] = [{"x": float(x), "out": float(round(np.float64(x), 1))} for x in xs]
    out["py_round1"] = [{"x": float(x), "out": round(float(x), 1)} for x in xs]

    # fused multiply-add, exact (Fraction), on the patterns the port meets: squares plus a square, cancellation
    fma = []
    for _ in range(100):
        a, b, c = (float(v) for v in rng.uniform(-2000, 2000, 3))
        for x, y, z in ((a, b, c * 1000), (a, a, b * b), (a, b, -(a * b)), (a, b, -(a * b) + math.ulp(a * b) / 2)):
            fma.append({"a": x, "b": y, "c": z, "out": float(Fraction(x) * Fraction(y) + Fraction(z))})
    out["fma"] = fma

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
