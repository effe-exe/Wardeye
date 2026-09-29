# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The Python reference for table.test.ts: what the table modules of ml/rifteye_ml compute on seeded synthetic inputs.

    . ml/.venv/bin/activate
    RIFTEYE_DATA=~/rifteye-data python packages/engine/test/gen/table.py    # writes packages/engine/test/vectors/table.json

The modules: retrieval.py (the gallery pyramid, top-k, the four-turn search), SciPy's linear_sum_assignment,
matcrops.py, changegate.py, live/layouts.py and live/autolayout.py, plus the scipy.ndimage operations the last three use.

No pictures and nothing from a broadcast or a card: every picture is described by a spec (rectangles, noise, cells)
that both sides render from one seeded 32-bit xorshift, and the vectors keep the spec and the SHA-256 of what Python
made of it. Numbers are written as Python's repr, which JSON.parse reads back as the same doubles.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from rifteye_ml import changegate as cg
from rifteye_ml import matcrops as mc
from rifteye_ml import retrieval as rt
from rifteye_ml.live import autolayout as al
from rifteye_ml.live.layouts import LAYOUTS, Layout

OUT = Path(__file__).resolve().parents[1] / "vectors" / "table.json"


# --- the generator and the pictures (test/table-helpers.ts has the same) ------------------------------------------

class XorShift32:
    """Marsaglia's 32-bit xorshift (13, 17, 5)."""

    def __init__(self, seed: int):
        self.x = (seed & 0xFFFFFFFF) or 1

    def next(self) -> int:
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x
        return x

    def below(self, n: int) -> int:
        return self.next() % n

    def unit(self) -> float:
        return self.next() / 4294967296.0


def render(spec: dict) -> np.ndarray:
    """The picture a spec describes (see Layer in table-helpers.ts)."""
    w, h = spec["w"], spec["h"]
    img = np.zeros((h, w, 3), np.uint8)
    main_rng = XorShift32(spec["seed"])
    for layer in spec["layers"]:
        x0, y0, rw, rh = layer["rect"]
        ya, yb, xa, xb = max(0, y0), min(h, y0 + rh), max(0, x0), min(w, x0 + rw)
        n = max(0, yb - ya) * max(0, xb - xa) * 3
        rng = XorShift32(layer["seed"]) if "seed" in layer else main_rng  # a layer with its own seed looks the same in every picture
        if layer.get("cell"):
            cell = layer["cell"]
            for cy in range(-(-rh // cell)):
                for cx in range(-(-rw // cell)):
                    c = [rng.next() >> 24 for _ in range(3)]
                    img[max(ya, y0 + cy * cell):min(yb, y0 + (cy + 1) * cell), max(xa, x0 + cx * cell):min(xb, x0 + (cx + 1) * cell)] = c
        elif layer.get("noise"):
            if n:
                img[ya:yb, xa:xb] = np.array([rng.next() >> 24 for _ in range(n)], np.uint8).reshape(yb - ya, xb - xa, 3)
        else:
            colour, j = layer["colour"], layer.get("jitter", 0)
            if j and n:
                arr = np.array([rng.below(2 * j + 1) - j for _ in range(n)]).reshape(yb - ya, xb - xa, 3) + np.array(colour)
                img[ya:yb, xa:xb] = np.clip(arr, 0, 255)
            elif n:
                img[ya:yb, xa:xb] = colour
    return img


def sha(a: np.ndarray) -> str:
    """The first 16 hex digits of the SHA-256 of an array's bytes: 64 bits, plenty to tell two results apart."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a)


def flat(colour, rect, jitter=0) -> dict:
    return {"rect": list(rect), "colour": list(colour), "jitter": jitter}


# --- numbers: round, %, //, median ---------------------------------------------------------------------------------

def sec_pynum(rng: XorShift32) -> dict:
    xs = [k / 8 for k in range(-24, 25)] + [k / 16 for k in range(-12, 13)]
    xs += [2.675, 0.285, 1.005, 4.35, 2.5, 3.5, -2.5, 0.5, 1.5, 0.125, 0.375, 0.0625, 1e-7, 123456.789, 8.4999999,
           0.49999999999999994, 4503599627370495.5, -0.4, -0.04, -0.0004, 0.05, 0.15, 0.25, 0.35, 155.05, 91.65, 66.65]
    xs += [(rng.unit() - 0.5) * 2000 for _ in range(60)]
    xs += [round((rng.unit() - 0.5) * 200, 4) for _ in range(30)]
    rounds = [[x, nd, round(x) if nd == 0 else round(x, nd)] for x in xs for nd in (0, 1, 2, 3)]
    avs = [-7.5, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 7.5, 10.0, 179.9, 180.0, 359.99, -90.0, -180.0, 2.5e9]
    bvs = [-3.0, -2.5, -1.0, 1.0, 2.5, 3.0, 180.0, 0.3, 2 * math.pi]
    mods = [[a, b, a % b, a // b] for a in avs for b in bvs]
    medians = []
    for n in list(range(1, 10)) * 3:
        v = [rng.below(256) if n % 2 else round((rng.unit() - 0.5) * 100, 3) for _ in range(n)]
        medians.append([v, float(np.median(v))])
    return {"round": rounds, "mod": mods, "median": medians}


# --- layouts -------------------------------------------------------------------------------------------------------

def layout_json(l: Layout) -> dict:
    return {"name": l.name, "title": l.title, "table": list(l.table), "card_long_1080": l.card_long_1080, "split": l.split,
            "mask": l.mask, "mat_tol": l.mat_tol, "mat": list(l.mat) if l.mat else None, "mat_share": l.mat_share}


def sec_layouts() -> dict:
    custom = [Layout("half", "ties", (0.25, 0.125, 0.75, 0.875), 100.0),
              Layout("wide", "horizontal", (0.1, 0.2, 0.9, 0.8), 77.5, split="horizontal", mat=(1, 2, 3), mat_share=0.5)]
    every = list(LAYOUTS.values()) + custom
    cases = []
    for li, l in enumerate(every):
        for w, h in [(1920, 1080), (1280, 720), (1921, 1081), (640, 360), (3840, 2160), (1001, 563), (6, 4), (10, 7), (14, 9), (18, 10)]:
            bx = l.box(w, h)
            pts = [[x, y, l.side(x, y, w, h)] for x, y in [(0, 0), (w / 2, h / 2), (w / 2 - 0.5, h / 2 - 0.5), (w * 0.4, h * 0.6), (w * 0.9, h * 0.1),
                                                          ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)]]
            cases.append({"layout": li, "w": w, "h": h, "box": list(bx), "px": l.card_px(h), "points": pts})
    return {"presets": {k: layout_json(v) for k, v in LAYOUTS.items()}, "layouts": [layout_json(l) for l in every],
            "sides": [list(l.sides()) for l in every], "cases": cases}


# --- linear_sum_assignment -----------------------------------------------------------------------------------------

def sec_lsap(rng: XorShift32) -> dict:
    cases = []
    for i in range(360):
        nr, nc = 1 + rng.below(7), 1 + rng.below(7)
        kind = ["int", "wide", "float", "const", "masked", "signed"][i % 6]
        vals = []
        for _ in range(nr * nc):
            if kind == "int":
                vals.append(rng.below(5))
            elif kind == "wide":
                vals.append(rng.below(100))
            elif kind == "float":
                vals.append(rng.unit() * 10 if i % 12 == 2 else rng.below(10240) / 1024)
            elif kind == "const":
                vals.append(3)
            elif kind == "masked":  # the tracker's costs: a distance below 0.6, or the mask value 1e6
                vals.append(1e6 if rng.below(3) else rng.below(600) / 1000)
            else:
                vals.append(rng.below(41) - 20)
        maximize = i % 3 == 0
        a, b = linear_sum_assignment(np.array(vals, np.float64).reshape(nr, nc), maximize=maximize)
        cases.append({"rows": nr, "cols": nc, "cost": vals, "maximize": maximize, "a": a.tolist(), "b": b.tolist()})
    special = []

    def add(name: str, rows: list, maximize: bool = False) -> None:
        conv = [[float(v) for v in r] for r in rows]
        shown = [[v if isinstance(v, str) else v for v in r] for r in rows]
        try:
            a, b = linear_sum_assignment(np.array(conv, np.float64).reshape(len(rows), -1), maximize=maximize)
            out = {"a": a.tolist(), "b": b.tolist()}
        except ValueError as e:
            out = {"error": str(e)}
        special.append({"name": name, "cost": shown, "maximize": maximize, **out})

    add("infeasible", [[1, "inf"], [2, "inf"]])
    add("inf entry that is not needed", [[1, "inf"], ["inf", 2]])
    add("nan", [[1, "nan"], [2, 3]])
    add("minus inf", [[1, "-inf"], [2, 3]])
    add("plus inf, maximized, becomes minus inf", [[1, "inf"], [2, 3]], True)
    add("minus inf, maximized, becomes plus inf", [[1, "-inf"], [2, 3]], True)
    add("wide with inf", [[1, "inf", 3], [2, 5, "inf"]])
    add("tall with inf", [[1, "inf"], [2, 5], ["inf", 7]])
    add("infeasible tall", [["inf", "inf"], [2, 5], [3, 1]])
    add("one by one", [[7]])
    add("all inf", [["inf", "inf"], ["inf", "inf"]])
    add("large costs", [[1e300, 1e300], [1e300, 2e300]])
    empty = [{"rows": 0, "cols": 3, "a": [], "b": []}, {"rows": 3, "cols": 0, "a": [], "b": []}]
    # big ones, drawn from a seed on both sides
    big = []
    for seed, nr, nc, kind in [(1, 100, 100, "unit"), (2, 120, 150, "unit"), (3, 150, 80, "unit"), (4, 60, 60, "int5"), (5, 90, 90, "masked"), (6, 40, 200, "int5")]:
        r = XorShift32(seed)
        vals = [r.unit() * 100 if kind == "unit" else r.below(5) if kind == "int5" else (1e6 if r.below(4) else r.unit()) for _ in range(nr * nc)]
        a, b = linear_sum_assignment(np.array(vals, np.float64).reshape(nr, nc))
        big.append({"seed": seed, "rows": nr, "cols": nc, "kind": kind, "a": a.tolist(), "b": b.tolist()})
    return {"cases": cases, "special": special, "empty": empty, "big": big}


# --- scipy.ndimage -------------------------------------------------------------------------------------------------

def mask_from(spec: dict) -> np.ndarray:
    rng = XorShift32(spec["seed"])
    w, h = spec["w"], spec["h"]
    m = np.zeros((h, w), np.uint8)
    if spec["kind"] == "rects":
        for _ in range(spec["n"]):
            x, y = rng.below(w), rng.below(h)
            rw, rh = 1 + rng.below(max(1, w // 3)), 1 + rng.below(max(1, h // 3))
            m[y:y + rh, x:x + rw] = 1
    else:
        for y in range(h):
            for x in range(w):
                m[y, x] = 1 if rng.below(1000) < spec["permille"] else 0
    return m


def sec_ndimage(rng: XorShift32) -> dict:
    cases = []
    for i in range(48):
        w, h = 1 + rng.below(44), 1 + rng.below(44)
        spec = {"kind": "rects", "n": 1 + rng.below(9), "seed": 100 + i, "w": w, "h": h} if i % 2 == 0 else \
            {"kind": "speckle", "permille": [80, 300, 500, 800][i % 4], "seed": 100 + i, "w": w, "h": h}
        m = mask_from(spec)
        lab, n = ndimage.label(m)
        res = {"dil1": ndimage.binary_dilation(m, iterations=1), "dil2": ndimage.binary_dilation(m, iterations=2),
               "dil4": ndimage.binary_dilation(m, iterations=4), "dil6": ndimage.binary_dilation(m, iterations=6),
               "open3": ndimage.binary_opening(m, structure=np.ones((3, 3))), "close3": ndimage.binary_closing(m, structure=np.ones((3, 3))),
               "close15": ndimage.binary_closing(m, structure=np.ones((15, 15))), "fill": ndimage.binary_fill_holes(m)}
        objs = [[o[0].start, o[0].stop, o[1].start, o[1].stop] for o in ndimage.find_objects(lab)]
        cases.append({"spec": spec, "mask": sha(m), "sha": {k: sha(v.astype(np.uint8)) for k, v in res.items()},
                      "labels": sha(lab.astype(np.int32)), "count": int(n), "objects": objs})
    return {"cases": cases}


# --- matcrops ------------------------------------------------------------------------------------------------------

MAT = [34, 44, 55]


def picture_specs() -> list[dict]:
    """A few named synthetic pictures: a mat with cards on it, ties in the histogram, noise, cells."""
    specs = [
        {"name": "mat and cards", "w": 120, "h": 80, "seed": 1,
         "layers": [flat(MAT, (0, 0, 120, 80), 3), flat((200, 180, 60), (20, 10, 30, 42), 30), {"rect": [70, 30, 30, 42], "noise": True}]},
        {"name": "noise", "w": 40, "h": 30, "seed": 2, "layers": [{"rect": [0, 0, 40, 30], "noise": True}]},
        {"name": "two colours, equal counts", "w": 32, "h": 16, "seed": 3,
         "layers": [flat((16, 16, 16), (0, 0, 16, 16)), flat((240, 240, 240), (16, 0, 16, 16))]},
        {"name": "two values in one bin, equal counts", "w": 8, "h": 8, "seed": 4,
         "layers": [flat((32, 40, 48), (0, 0, 8, 4)), flat((33, 41, 49), (0, 4, 8, 4))]},
        {"name": "three values in one bin", "w": 9, "h": 3, "seed": 5,
         "layers": [flat((130, 130, 130), (0, 0, 3, 3)), flat((131, 133, 129), (3, 0, 3, 3)), flat((143, 140, 139), (6, 0, 3, 3))]},
        {"name": "cells", "w": 61, "h": 47, "seed": 6, "layers": [{"rect": [0, 0, 61, 47], "cell": 5}]},
        {"name": "red mat, dark border cards", "w": 90, "h": 60, "seed": 7,
         "layers": [flat((190, 25, 45), (0, 0, 90, 60), 4), flat((20, 20, 24), (10, 8, 24, 34), 5), flat((60, 30, 30), (50, 20, 20, 28), 6),
                    flat((230, 230, 225), (30, 40, 12, 12), 3)]},
        {"name": "odd size", "w": 37, "h": 23, "seed": 8, "layers": [flat((10, 200, 90), (0, 0, 37, 23), 60)]},
    ]
    return specs


def sec_matcrops() -> dict:
    specs = picture_specs()
    mat_colour, notmat, border = [], [], []
    for s in specs:
        img = render(s)
        for step in (1, 2, 3, 4):
            mat_colour.append({"spec": s["name"], "step": step, "mat": [int(v) for v in mc.mat_colour(img[::step, ::step])]})
        mat = mc.mat_colour(img)
        for tol, m in [(60, mat), (45, mat), (10, mat), (60, np.array([120, 120, 120]))]:
            mask = mc.notmat_mask(img, m, tol)
            notmat.append({"spec": s["name"], "mat": [int(v) for v in m], "tol": tol, "count": int(mask.sum()), "sha": sha(mask.astype(np.uint8))})
        for args in [(), (90, 45, 45), (200, 100, 100), (50, 10, 10)]:
            mask = mc.border_mask(img, *args)
            border.append({"spec": s["name"], "args": list(args), "count": int(mask.sum()), "sha": sha(mask.astype(np.uint8))})
    # detail: a card face (coloured squares, noise), a plain sleeve (flat, or with a little noise), stripes; small and odd sizes too
    details = []
    every = [(110, 155), (47, 66), (300, 420), (20, 28), (5, 7), (2, 2), (1, 1), (63, 88), (155, 110), (48, 64), (33, 100)]
    plan = [("cells", 5, every), ("noise", 0, every), ("flat", 3, every), ("flat", 0, every), ("stripes", 0, [(110, 155), (47, 66), (63, 88), (20, 28)])]
    seed = 500
    for kind, param, sizes in plan:
        for w, h in sizes:
            seed += 1
            if kind == "cells":
                layers = [{"rect": [0, 0, w, h], "cell": param}]
            elif kind == "noise":
                layers = [{"rect": [0, 0, w, h], "noise": True}]
            elif kind == "flat":
                layers = [flat((200, 60, 120), (0, 0, w, h), param)]
            else:
                layers = [flat((30, 30, 30), (0, 0, w, h))] + [flat((220, 220, 220), (0, y, w, 2)) for y in range(0, h, 6)]
            details.append({"w": w, "h": h, "seed": seed, "layers": layers, "detail": mc.detail(pil(render({"w": w, "h": h, "seed": seed, "layers": layers})))})
    return {"specs": specs, "mat_colour": mat_colour, "notmat": notmat, "border": border, "detail": details, "face_down": mc.FACE_DOWN_DETAIL}


# --- retrieval -----------------------------------------------------------------------------------------------------

def f32_exact(x: Fraction) -> float:
    """The float32 nearest to an exact value (ties to even), as a double."""
    if x == 0:
        return 0.0
    sign, x = (-1, -x) if x < 0 else (1, x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    scale = e - 23
    m = x / Fraction(2) ** scale  # in [2^23, 2^24)
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return sign * float(Fraction(fl) * Fraction(2) ** scale)


def sec_fma32(rng: XorShift32) -> list:
    """Fused multiply-adds in float32, with the correct single rounding, including sums a double rounds onto a float32 midpoint."""
    out = []

    def add(a: float, b: float, c: float) -> None:
        a, b, c = (float(np.float32(v)) for v in (a, b, c))
        out.append([a, b, c, f32_exact(Fraction(a) * Fraction(b) + Fraction(c))])

    for _ in range(150):
        a, b = (rng.unit() - 0.5) * 4, (rng.unit() - 0.5) * 4
        a, b = float(np.float32(a)), float(np.float32(b))
        c = [(rng.unit() - 0.5) * 4, -a * b, -a * b * (1 + 2.0 ** -20), 0.0, rng.unit() * 2.0 ** -30][rng.below(5)]
        add(a, b, c)
    # exact sum 1 + 2^-24 + 2^-54: a double rounds it to the float32 midpoint 1 + 2^-24, the true value is above it
    for j in range(-3, 4):
        for s in (1, -1):
            add(s * 13325 * 2.0 ** (-27 + j), 80581 * 2.0 ** -27, s * 2.0 ** j)
            add(s * 13325 * 2.0 ** (-27 + j), -80581 * 2.0 ** -27, -s * 2.0 ** j)
    return out


def emulate_chain(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A @ B.T as a chain of float32 fused multiply-adds over k: what OpenBLAS's kernel does for these shapes."""
    acc = np.zeros((A.shape[0], B.shape[0]), np.float32)
    a64, b64 = A.astype(np.float64), B.astype(np.float64)
    for k in range(A.shape[1]):
        acc = (a64[:, k:k + 1] * b64[:, k][None, :] + acc.astype(np.float64)).astype(np.float32)
    return acc


def f32_matrix(rng: XorShift32, rows: int, dim: int) -> np.ndarray:
    return np.array([rng.unit() - 0.5 for _ in range(rows * dim)], np.float32).reshape(rows, dim)


def sec_similarities() -> list:
    out = []
    for seed, m, k, n in [(11, 4, 256, 1194), (12, 8, 64, 800), (13, 4, 384, 1000), (14, 3, 256, 450), (15, 5, 100, 600), (16, 2, 256, 700)]:
        rng = XorShift32(seed)
        a, b = f32_matrix(rng, m, k), f32_matrix(rng, n, k)
        c = a @ b.T
        # the shapes are ones where numpy's float32 matmul is that chain; say so if a numpy or OpenBLAS ever changes it
        assert np.array_equal(c, emulate_chain(a, b)), (m, k, n)
        out.append({"seed": seed, "queries": m, "dim": k, "gallery": n, "sha": sha(c.astype("<f4")), "best": sha(c.max(axis=0).astype("<f4"))})
    return out


def sec_pyramid() -> dict:
    cases = []
    longs = list(range(0, 401)) + [1000, 5000]
    for scales in ([40, 80, 160], [120, 130, 150], [60], [40, 45, 50, 60, 80, 100, 120, 160], [140, 150, 160]):
        p = rt.Pyramid({s: np.zeros((3, 4), np.float32) for s in scales})
        cases.append({"scales": scales, "level": [p.level_for(x) for x in longs]})
    return {"long": longs, "cases": cases}


class QuadMean:
    """A stand-in encoder: the mean colour of each quarter of a picture, 12 numbers, float32."""

    name, dim = "quadmean", 12

    def embed(self, images):
        out = np.zeros((len(images), 12), np.float32)
        for i, im in enumerate(images):
            a = np.asarray(im.convert("RGB")).astype(np.float64)
            h, w = a.shape[:2]
            hh, ww = h // 2, w // 2
            for q, region in enumerate([a[:hh, :ww], a[:hh, ww:], a[hh:, :ww], a[hh:, ww:]]):
                n = region.shape[0] * region.shape[1]
                out[i, q * 3:q * 3 + 3] = region.reshape(-1, 3).sum(axis=0) / n if n else 0
        return out


def sec_retrieval(rng: XorShift32) -> dict:
    # at_long_side and band on named pictures
    sizes = [(60, 84), (300, 420), (84, 60), (7, 5), (3, 400)]
    at_long, bands = [], []
    for i, (w, h) in enumerate(sizes):
        spec = {"w": w, "h": h, "seed": 900 + i, "layers": [{"rect": [0, 0, w, h], "cell": 3}]}
        im = pil(render(spec))
        for side in (40, 60, 100, 155, 1, 700):
            r = rt.at_long_side(im, side)
            at_long.append({"spec": spec, "side": side, "size": list(r.size), "sha": sha(np.asarray(r))})
        for view in ("full", "", "top:0.25", "bottom:0.5", "left:0.3", "right:1", "top:0.001"):
            r = rt.band(im, view)
            bands.append({"spec": spec, "view": view, "size": list(r.size), "sha": sha(np.asarray(r))})
    bad = []
    for view in ("top", "top:", "top:0", "top:1.5", "middle:0.5", "top:abc", "top:-1"):
        try:
            rt.band(pil(np.zeros((10, 10, 3), np.uint8)), view)
            bad.append({"view": view, "error": None})
        except ValueError:
            bad.append({"view": view, "error": "ValueError"})
    # topk on random float32 embeddings
    topks = []
    for seed, nq, ng, dim, k in [(31, 5, 200, 24, 5), (32, 1, 50, 8, 3), (33, 7, 30, 16, 40), (34, 3, 1194, 256, 5)]:
        r = XorShift32(seed)
        q, g = f32_matrix(r, nq, dim), f32_matrix(r, ng, dim)
        idx, sc = rt.topk(q, g, k)
        topks.append({"seed": seed, "queries": nq, "gallery": ng, "dim": dim, "k": k, "idx": idx.tolist(), "scores": [[float(v) for v in row] for row in sc]})
    # search with the stand-in encoder: a pyramid gallery and a plain one, cards drawn as four coloured quarters
    enc = QuadMean()
    art_specs = []
    for j in range(14):
        layers = []
        for qx, qy in [(0, 0), (30, 0), (0, 42), (30, 42)]:
            layers.append(flat((rng.below(256), rng.below(256), rng.below(256)), (qx, qy, 30, 42), 12))
        art_specs.append({"w": 60, "h": 84, "seed": 700 + j, "layers": layers})
    art = [pil(render(s)) for s in art_specs]
    scales = [30, 60]
    pyr = rt.Pyramid.build(enc, art, scales)
    plain = enc.embed([rt.at_long_side(a, 60) for a in art])
    queries_spec = []
    queries = []
    for n in range(10):
        j, side, angle = rng.below(14), [20, 30, 45, 60, 80][rng.below(5)], [0, 90, 180, 270][rng.below(4)]
        q = rt.at_long_side(art[j], side)
        q = q.rotate(angle, expand=True) if angle else q
        queries_spec.append({"art": j, "side": side, "angle": angle})
        queries.append(q)

    def res(gallery, k, rot_inv):
        idx, sc, rot = rt.search(enc, gallery, queries, k, rot_inv)
        return {"k": k, "rotation_invariant": rot_inv, "idx": idx.tolist(), "scores": [[float(v) for v in row] for row in sc], "rot": rot.tolist()}

    searches = {"art": art_specs, "queries": queries_spec, "scales": scales,
                "pyramid": [res(pyr, 3, True), res(pyr, 5, False), res(pyr, 40, True)],
                "plain": [res(plain, 4, True), res(plain, 3, False)]}
    # ranked_labels and accuracy
    labels = ["a", "b", "a", "c", "d", "b", "e"]
    idx = np.array([[0, 2, 1, 3, 4], [3, 4, 5, 0, 6], [6, 6, 5, 1, 0], [2, 0, 1, 5, 3]])
    ranked = rt.ranked_labels(idx, labels)
    truth = ["a", "d", "c", "e"]
    return {"at_long_side": at_long, "band": bands, "bad_views": bad, "topk": topks, "search": searches,
            "ranked": {"labels": labels, "idx": idx.tolist(), "ranked": ranked, "truth": truth, "accuracy": rt.accuracy(ranked, truth)}}


# --- the change gate -----------------------------------------------------------------------------------------------

BARE_MAT = [190, 25, 45]
SKIN = [225, 180, 150]
CARD1, CARD2, CARD3 = [30, 30, 200], [240, 200, 40], [20, 160, 90]


def gate_frames(seq: dict) -> list[np.ndarray]:
    """The frames of a sequence: each step is n frames of a background colour with rectangles (colour, or "noise") on it."""
    w, h, noise = seq["w"], seq["h"], seq["noise"]
    frames = []
    for step in seq["steps"]:
        for _ in range(step["n"]):
            layers = [flat(step.get("bg", BARE_MAT), (0, 0, w, h), noise)]
            for x, y, rw, rh, c in step.get("rects", []):
                layers.append({"rect": [x, y, rw, rh], "noise": True, "seed": c[1]} if c[0] == "noise" else flat(c, (x, y, rw, rh), noise))
            frames.append(render({"w": w, "h": h, "seed": seq["seed"] + len(frames), "layers": layers}))
    return frames


def card(x, y, c=CARD1, w=8, h=11):
    return [x, y, w, h, c]


def hand(x, y):
    return [x, y, 20, 14, SKIN]


def sequences() -> list[dict]:
    base = {"w": 160, "h": 90, "noise": 2, "fps": 5.0, "settings": {"width": 160, "mat_rgb": BARE_MAT}}
    s = []
    # a card played with the hand still on it, moved; a hand sweeping across first: no event for the hand
    s.append({**base, "name": "played, moved, a hand passing", "seed": 1000, "steps": [{"n": 3}] + [{"n": 1, "rects": [hand(20 + 6 * i, 30)]} for i in range(5)]
              + [{"n": 5}, {"n": 2, "rects": [card(50, 30), hand(47, 25)]}, {"n": 6, "rects": [card(50, 30)]}, {"n": 6, "rects": [card(100, 45)]}]})
    # most of the picture changes at once: one cut event
    s.append({**base, "name": "a camera cut", "seed": 2000, "steps": [{"n": 3}, {"n": 5, "rects": [[0, 0, 100, 90, ["noise", 77]]]}, {"n": 4, "rects": [[0, 0, 100, 90, ["noise", 77]], card(120, 40)]}]})
    # a player cam for 10 frames, a card played meanwhile, found on return
    s.append({**base, "name": "a cutaway", "seed": 3000, "steps": [{"n": 3}, {"n": 10, "bg": [90, 90, 90]}, {"n": 6, "rects": [card(75, 40, CARD2)]}]})
    # a hand in the first frames leaving is no change; a real card afterwards
    s.append({**base, "name": "a hand in the first frame", "seed": 4000, "steps": [{"n": 2, "rects": [hand(50, 30)]}, {"n": 8}, {"n": 8, "rects": [card(100, 45)]}]})
    # a hand resting on a settled card, then leaving: nothing happens
    s.append({**base, "name": "a hand rests on a card", "seed": 5000, "steps": [{"n": 4}, {"n": 8, "rects": [card(50, 30)]}, {"n": 6, "rects": [card(50, 30), hand(45, 25)]},
                                                                             {"n": 6, "rects": [card(50, 30)]}]})
    # light drifting down, a small die, a weak patch, a card replaced, one on the edge, one in an ignored box, one removed
    drift = [{"n": 2, "bg": [190 - i // 2 * 1, 25, 45]} for i in range(12)]
    dark = [184, 25, 45]
    s.append({**base, "name": "drift, die, patch, replace, edge, ignore, remove", "seed": 6000,
              "settings": {"width": 160, "mat_rgb": None, "ignore": [[0.75, 0.0, 1.0, 0.3]]},
              "steps": [{"n": 4}] + drift + [{"n": 6, "bg": dark, "rects": [card(30, 40)]},
                                            {"n": 6, "bg": dark, "rects": [card(30, 40), [60, 60, 3, 3, [10, 10, 10]]]},
                                            {"n": 6, "bg": dark, "rects": [card(30, 40), [60, 60, 3, 3, [10, 10, 10]], [120, 50, 12, 12, [224, 65, 85]]]},
                                            {"n": 6, "bg": dark, "rects": [card(30, 40, CARD3), [60, 60, 3, 3, [10, 10, 10]], [120, 50, 12, 12, [224, 65, 85]]]},
                                            {"n": 6, "bg": dark, "rects": [card(30, 40, CARD3), card(0, 60, CARD2)]},
                                            {"n": 6, "bg": dark, "rects": [card(30, 40, CARD3), card(0, 60, CARD2), card(130, 10, CARD1)]},
                                            {"n": 6, "bg": dark, "rects": [card(0, 60, CARD2), card(130, 10, CARD1)]}]})
    # slower frames (2 a second, settle 1 s), the mat colour worked out from the first frame
    s.append({**base, "name": "two frames a second", "seed": 7000, "fps": 2.0, "settings": {"width": 160, "mat_rgb": None, "settle_s": 1.0},
              "steps": [{"n": 2}, {"n": 3, "rects": [card(40, 20, CARD3)]}, {"n": 3, "rects": [card(40, 20, CARD3), card(90, 50)]}, {"n": 3, "rects": [card(90, 50)]}]})
    # the size the runner uses, default settings: two cards, a hand between them, one card off again
    big = {"w": 320, "h": 180, "noise": 3, "fps": 5.0, "settings": {"mat_rgb": None}}
    s.append({**big, "name": "the table view at 320 px", "seed": 8000,
              "steps": [{"n": 3}, {"n": 2, "rects": [hand(100, 60), card(110, 70, CARD1, 16, 22)]}, {"n": 6, "rects": [card(110, 70, CARD1, 16, 22)]},
                        {"n": 3, "rects": [card(110, 70, CARD1, 16, 22), hand(200, 90)]}, {"n": 6, "rects": [card(110, 70, CARD1, 16, 22), card(210, 100, CARD2, 16, 22)]},
                        {"n": 6, "rects": [card(210, 100, CARD2, 16, 22)]}]})
    # the table camera is not on for the first frames; the mat colour is given (else the cutaway would be taken for the mat)
    s.append({**base, "name": "starts on a cutaway", "seed": 9000, "steps": [{"n": 3, "bg": [90, 90, 90]}, {"n": 4}, {"n": 5, "rects": [card(60, 40, CARD2)]}, {"n": 5}]})
    s += [fuzz_sequence(k) for k in (4, 7, 8, 9, 22, 34)]
    return s


def fuzz_sequence(k: int) -> dict:
    """A random scene: cards and hands coming and going, a cut, a cutaway, settings drawn at random (and the same seeds every run)."""
    import random

    r = random.Random(1000 + k)
    w, h = r.choice([(96, 54), (128, 72), (160, 90)])
    fps = r.choice([2.0, 5.0])
    mat = [r.randrange(20, 235) for _ in range(3)] if r.random() < 0.7 else BARE_MAT
    settings = {"width": r.choice([96, 160, 320]), "mat_rgb": mat if r.random() < 0.6 else None, "settle_s": r.choice([0.4, 0.6, 1.0]),
                "diff": r.choice([16.0, 28.0, 40.0]), "motion": r.choice([12.0, 18.0, 30.0]), "min_area": r.choice([0.15, 0.35, 0.6]),
                "hand_share": r.choice([0.02, 0.05, 0.2]), "strong": r.choice([0.2, 0.3, 0.5]), "adapt": r.choice([0.05, 0.05, 0.2]),
                "global_cut": r.choice([0.35, 0.5]), "min_mat": r.choice([0.1, 0.25, 0.4])}
    if r.random() < 0.3:
        settings["ignore"] = [[r.random() * 0.5, r.random() * 0.5, 0.5 + r.random() * 0.5, 0.5 + r.random() * 0.5]]
    objs = []
    n = r.randrange(35, 60)
    for _ in range(r.randrange(2, 6)):
        cw, ch = r.randrange(5, w // 5), r.randrange(6, h // 4)
        objs.append(("card", r.randrange(0, w - cw), r.randrange(0, h - ch), cw, ch, [r.randrange(256) for _ in range(3)], r.randrange(0, n - 5), r.randrange(0, n + 10)))
    for _ in range(r.randrange(0, 3)):
        hw, hh = r.randrange(8, w // 4), r.randrange(6, h // 4)
        objs.append(("hand", r.randrange(0, w - hw), r.randrange(0, h - hh), hw, hh, SKIN if r.random() < 0.8 else [r.randrange(256) for _ in range(3)], r.randrange(0, n - 5), r.randrange(0, n + 10)))
    cut_at = r.randrange(5, n) if r.random() < 0.25 else None
    away = (r.randrange(3, n - 5), r.randrange(2, 8)) if r.random() < 0.3 else None
    steps: list[dict] = []
    for i in range(n):
        rects = []
        for kind, x, y, cw, ch, colour, t0, t1 in objs:
            if t0 <= i < t1:
                rects.append([(x + 3 * (i - t0)) % max(1, w - cw) if kind == "hand" else x, y, cw, ch, colour])
        step: dict = {"n": 1, "rects": rects}
        if away and away[0] <= i < away[0] + away[1]:
            step["bg"], step["rects"] = [90, 90, 90], []
        if cut_at is not None and i >= cut_at:
            step["rects"] = rects + [[0, 0, w // 2, h, ["noise", 5 + k]]]
        if steps and {kk: v for kk, v in steps[-1].items() if kk != "n"} == {kk: v for kk, v in step.items() if kk != "n"}:
            steps[-1]["n"] += 1  # the same picture again: one step of several frames
        else:
            steps.append(step)
    return {"name": f"random scene {k}", "w": w, "h": h, "noise": r.choice([0, 1, 2, 4]), "fps": fps, "seed": 5000 * (k + 1), "settings": settings, "steps": steps}


def run_gate(seq: dict) -> dict:
    frames = gate_frames(seq)
    kw = dict(seq["settings"])
    if kw.get("mat_rgb") is not None:
        kw["mat_rgb"] = tuple(kw["mat_rgb"])
    if kw.get("ignore"):
        kw["ignore"] = tuple(tuple(b) for b in kw["ignore"])
    s = cg.GateSettings(fps=seq["fps"], **kw)
    gate = cg.ChangeGate(s)
    per_frame = []
    for i, f in enumerate(frames):
        new = gate.feed(i / s.fps, f)
        per_frame.append({"events": [{"t": e.t, "box": list(e.box), "kind": e.kind, "area": e.area, "before_mat": e.before_mat,
                                      "after_mat": e.after_mat, "extra": e.extra} for e in new],
                          "background": sha(gate.background) if gate.background is not None else None})
    return {"frames": len(frames), "per_frame": per_frame, "still": sha(gate.still.astype(np.int32)) if gate.still is not None else None,
            "last_same": sha(gate.last_same) if gate.last_same is not None else None,
            "startup_hand": sha(gate.startup_hand.astype(np.uint8)) if gate.startup_hand is not None else None,
            "off_table": gate.off_table, "mat": [int(v) for v in gate.mat]}


def sec_changegate() -> dict:
    out = []
    for seq in sequences():
        res = run_gate(seq)
        kinds = [e["kind"] for f in res["per_frame"] for e in f["events"]]
        print(f"  gate: {seq['name']}: {res['frames']} frames, events {kinds}", file=sys.stderr)
        out.append({"seq": seq, "expected": res})
    # skin on noise and on the colour cube's corners
    skin = []
    for seed, w, h in [(21, 40, 30), (22, 17, 9)]:
        spec = {"w": w, "h": h, "seed": seed, "layers": [{"rect": [0, 0, w, h], "noise": True}]}
        m = cg.skin(render(spec))
        skin.append({"spec": spec, "count": int(m.sum()), "sha": sha(m.astype(np.uint8))})
    cube = np.array([[r, g, b] for r in range(0, 256, 5) for g in range(0, 256, 5) for b in range(0, 256, 5)], np.uint8).reshape(-1, 1, 3)
    skin.append({"cube": True, "count": int(cg.skin(cube).sum()), "sha": sha(cg.skin(cube).astype(np.uint8))})
    return {"sequences": out, "skin": skin, "defaults": {k: getattr(cg.GateSettings(), k) for k in cg.GateSettings.__dataclass_fields__}}


# --- the layout finder ---------------------------------------------------------------------------------------------

def broadcast_spec(seed: int, cards: list, mat=(30, 40, 55), w=640, h=360, panel=(128, 512)) -> dict:
    """A frame with busy panels either side of a mat, gold rules at the picture's borders, cards lying on the mat."""
    x0, x1 = panel
    layers = [{"rect": [0, 0, w, h], "cell": 4}, flat(mat, (x0, 0, x1 - x0, h), 3), flat((212, 175, 55), (x0 - 1, 0, 3, h)), flat((212, 175, 55), (x1, 0, 3, h))]
    for (cx, cy, cw, ch) in cards:
        layers.append({"rect": [cx, cy, cw, ch], "cell": 5})
    return {"w": w, "h": h, "seed": seed, "layers": layers}


class Recorder:
    """A stand-in detector that says what it is asked and answers from a seeded generator; every call is kept."""

    def __init__(self, frames, seed: int, true_side: float):
        self.frames, self.rng, self.true, self.calls = frames, XorShift32(seed), true_side, []

    def __call__(self, f, box, px):
        out = []
        for _ in range(2 + self.rng.below(4)):
            score = [0.95, 0.9, 0.7, 0.55, 0.62][self.rng.below(5)]
            side = self.true * (0.94 + 0.12 * self.rng.unit()) if self.rng.below(5) else self.true * (0.4 + 1.6 * self.rng.unit())
            short = side / 1.4
            cx, cy = 200 + self.rng.below(200), 40 + self.rng.below(260)
            ang = [0, 0.05, -0.05, 1.5707963267948966][self.rng.below(4)]
            c, s = math.cos(ang), math.sin(ang)
            corners = [(cx - short / 2, cy - side / 2), (cx + short / 2, cy - side / 2), (cx + short / 2, cy + side / 2), (cx - short / 2, cy + side / 2)]
            quad = [[round(cx + (x - cx) * c - (y - cy) * s, 1), round(cy + (x - cx) * s + (y - cy) * c, 1)] for x, y in corners]
            out.append({"score": score, "quad": [v for p in quad for v in p], "cls": "card"})
        self.calls.append({"frame": next(i for i, g in enumerate(self.frames) if g is f), "box": list(box), "px": px, "out": out})
        return out


def autolayout_case(name: str, specs: list, true_side: float | None) -> dict:
    """What the layout finder makes of some frames: with a stand-in detector (its every call kept) when `true_side` is given, and without."""
    frames = [render(s) for s in specs]
    b = al.borders(frames)
    tw = al.table_window(frames)
    case = {"name": name, "specs": specs, "borders": list(b),
            "table_window": None if tw is None else {"window": list(tw[0]), "mat": list(tw[1]), "share": tw[2]},
            "detector_calls": [], "card_size": None, "auto_layout": None}
    if true_side is not None:
        rec = Recorder(frames, 77, true_side)
        lay = al.auto_layout(frames, rec)
        case["detector_calls"] = rec.calls
        case["auto_layout"] = None if lay is None else layout_json(lay)
        if tw is not None:
            case["card_size"] = al.card_size(Recorder(frames, 77, true_side), frames, tw[0])
    lay_finder = al.auto_layout(frames)
    case["auto_layout_finder"] = None if lay_finder is None else layout_json(lay_finder)
    case["card_size_finder"] = al.card_size_finder(frames, tw[0], tw[1]) if tw is not None else None
    print(f"  autolayout: {name}: borders {b}, window {None if tw is None else tw[0]}, layout {case['auto_layout'] and (case['auto_layout']['table'], case['auto_layout']['card_long_1080'])}, "
          f"finder {case['auto_layout_finder'] and case['auto_layout_finder']['card_long_1080']}", file=sys.stderr)
    return case


def sec_autolayout() -> dict:
    spots = [(173, 53), (280, 133), (400, 60), (220, 253), (333, 200), (187, 167)]
    cases = []
    for name, cards, nframes in [("six cards", [(x, y, 37, 52) for x, y in spots], 5), ("few cards", [(x, y, 37, 52) for x, y in spots[:2]], 5),
                                 ("four frames, no cards", [], 4)]:
        cases.append(autolayout_case(name, [broadcast_spec(40 + k, cards) for k in range(nframes)], 52.0))
    # player cams: noise, no table
    cases.append(autolayout_case("player cams: noise", [{"w": 640, "h": 360, "seed": 300 + k, "layers": [{"rect": [0, 0, 640, 360], "cell": 4}]} for k in range(5)], 52.0))
    # two frames (an even count: the median takes the mean of two values), the mat a little different in each
    cases.append(autolayout_case("two frames, two mats", [broadcast_spec(90 + k, [(200, 70, 37, 52)], mat=(30, 40, 56 + 5 * k)) for k in range(2)], None))
    return {"cases": cases}


def sec_findcards() -> dict:
    """The bootstrap finder on axis-aligned cards: on a blue mat by colour (some found, a stack, a die and a card of another size
    rejected), and on a red mat by the dark border rule."""
    specs = []
    for seed, cards in [(61, [(20, 20, 60, 84), (120, 30, 60, 84), (220, 100, 84, 60)]),
                        (62, [(20, 20, 60, 84), (60, 60, 60, 84), (200, 30, 60, 84), (250, 150, 10, 10)]),
                        (63, [(30, 40, 60, 90), (140, 40, 55, 78), (230, 40, 70, 100)])]:
        layers = [flat((30, 60, 140), (0, 0, 320, 200), 3)] + [{"rect": [x, y, w, h], "cell": 7} for x, y, w, h in cards]
        specs.append({"w": 320, "h": 200, "seed": seed, "layers": layers})
    red = [flat((190, 25, 45), (0, 0, 320, 200), 4)]
    for x, y, w, h in [(20, 20, 60, 84), (120, 30, 60, 84), (220, 100, 84, 60), (60, 60, 60, 84)]:
        red += [flat((20, 20, 24), (x, y, w, h), 3), flat((150, 180, 90), (x + 5, y + 5, w - 10, h - 10), 20)]
    specs.append({"w": 320, "h": 200, "seed": 64, "layers": red})
    runs = [("notmat 60", 84, 60, {}), ("notmat 45 loose", 84, 45, {"tol": 0.3}), ("notmat 60 at 60 px", 60, 60, {}), ("border", 84, None, {}), ("border loose", 84, None, {"tol": 0.3, "min_fill": 0.7})]
    out = []
    for s in specs:
        img = render(s)
        mat = mc.mat_colour(img)
        row = {"spec": s, "mat": [int(v) for v in mat], "runs": []}
        for name, long_px, mask_tol, opts in runs:
            mask = mc.notmat_mask(img, mat, mask_tol) if mask_tol is not None else None
            boxes = mc.find_cards(img, long_px, mask=mask, **opts)
            row["runs"].append({"name": name, "long_px": long_px, "mask_tol": mask_tol, "opts": opts,
                                "found": [{"centre": list(b.centre), "long_px": b.long_px, "short_px": b.short_px, "angle_deg": b.angle_deg, "fill": b.fill} for b in boxes]})
        out.append(row)
    rects = []
    rng = XorShift32(5)
    for _ in range(24):
        pts = np.array([[rng.below(60), rng.below(60)] for _ in range(6 + rng.below(20))], np.float64)
        try:
            c, lg, sh, ang = mc.min_area_rect(pts)
        except Exception:  # noqa: BLE001 - a flat set of points has no hull
            continue
        # how many hull edges give the least area: on a grid of whole numbers some sets tie (a right isosceles triangle fits a
        # rectangle of the same area two ways), and then which rectangle comes first depends on where Qhull starts the hull
        from scipy.spatial import ConvexHull

        hull = pts[ConvexHull(pts).vertices]
        areas = []
        for i in range(len(hull)):
            e = hull[(i + 1) % len(hull)] - hull[i]
            t = math.atan2(e[1], e[0])
            rot = hull @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
            areas.append(float(np.prod(rot.max(axis=0) - rot.min(axis=0))))
        ties = sum(1 for a in areas if a <= min(areas) * (1 + 1e-9))
        rects.append({"points": pts.tolist(), "centre": list(c), "long_px": lg, "short_px": sh, "angle_deg": ang, "ties": ties})
    return {"cases": out, "min_area_rect": rects}


# --- the gate's view of the table ----------------------------------------------------------------------------------

def sec_views() -> list:
    from rifteye_ml.reviewpack import view_to_frame

    out = []
    tables = [LAYOUTS['la-rq'].table, LAYOUTS['shenyang'].table, (0.25, 0.125, 0.75, 0.875), (0.183, 0.0, 0.815, 1.0)]
    for table in tables:
        for vw in (320, 97):
            tx0, ty0, tx1, ty1 = table
            vh = round(vw * (ty1 - ty0) * 1080 / ((tx1 - tx0) * 1920) / 2) * 2
            for (fw, fh) in [(1920, 1080), (1001, 563)]:
                for box in [(0, 0, vw, vh), (12, 30, 45, 71)]:
                    out.append({"table": list(table), "vw": vw, "vh": vh, "frame": [fw, fh], "box": list(box), "frame_box": list(view_to_frame(box, table, vw, fw, fh))})
    return out


# --- all of it -----------------------------------------------------------------------------------------------------

def main() -> None:
    rng = XorShift32(20260929)
    out = {"note": "made by test/gen/table.py; synthetic and seeded, no pictures",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "pillow": Image.__version__},
           "xorshift": {"seed": 12345, "first": [XorShift32(12345).next() for _ in range(1)][:0] + (lambda r: [r.next() for _ in range(8)])(XorShift32(12345))}}
    print("pynum", file=sys.stderr)
    out["pynum"] = sec_pynum(rng)
    print("layouts", file=sys.stderr)
    out["layouts"] = sec_layouts()
    print("lsap", file=sys.stderr)
    out["lsap"] = sec_lsap(rng)
    print("ndimage", file=sys.stderr)
    out["ndimage"] = sec_ndimage(rng)
    print("matcrops", file=sys.stderr)
    out["matcrops"] = sec_matcrops()
    print("fma32, similarities, pyramid", file=sys.stderr)
    out["fma32"] = sec_fma32(rng)
    out["similarities"] = sec_similarities()
    out["pyramid"] = sec_pyramid()
    print("retrieval", file=sys.stderr)
    out["retrieval"] = sec_retrieval(rng)
    print("changegate", file=sys.stderr)
    out["changegate"] = sec_changegate()
    print("autolayout", file=sys.stderr)
    out["autolayout"] = sec_autolayout()
    print("find_cards", file=sys.stderr)
    out["findcards"] = sec_findcards()
    out["views"] = sec_views()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, separators=(",", ":"), allow_nan=False)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"{OUT}: {len(text) / 1000:.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
