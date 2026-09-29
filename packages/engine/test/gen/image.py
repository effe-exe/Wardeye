# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""What Pillow makes of the picture operations src/image.ts ports, for its tests.

    . ml/.venv/bin/activate
    python packages/engine/test/gen/image.py              # test/vectors/image.json, synthetic pictures
    RIFTEYE_DATA=~/rifteye-data python packages/engine/test/gen/image.py --real ~/rifteye-data/m3
                                                          # fixtures/image/real.json, the LA final's frames

The vectors hold no pixels. Each case names a synthetic picture that both sides build from the same seeded
32-bit xorshift, the operations done to it, and the SHA-256 of Pillow's output with a few of its pixels.

The real-frame fixtures do the live runner's own calls (Scene.small, table_like, the detector's window and
tiles, card_crop and the identify turns, watch, matcrops.detail, encoders.letterbox) on the raw RGB frames in
frames/la-final-rgb, each written as the same steps and checked here against the ml function it stands for.
They come from a broadcast and stay private (D-006).

Rotations: Pillow's matrix takes cos and sin from the C library, image.ts from a correctly rounded double-double.
Where they differ (glibc is not correctly rounded for about 1 value in 700), the expected output is Pillow's
transform with the correctly rounded matrix, and the case says what Pillow itself gives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import PIL
from PIL import Image

PILLOW = "12.3.0"  # the version src/image.ts ports
HERE = Path(__file__).resolve().parent
VECTORS = HERE.parent / "vectors" / "image.json"
FILTERS = {"nearest": Image.Resampling.NEAREST, "box": Image.Resampling.BOX, "bilinear": Image.Resampling.BILINEAR,
           "hamming": Image.Resampling.HAMMING, "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
CHANNELS = {"RGB": 3, "L": 1, "RGBA": 4}
NOTES: list[str] = []


# --- the synthetic pictures (test/image.test.ts builds the same) ------------------------------------------------

def xorshift_bytes(seed: int, n: int) -> bytes:
    """n bytes of Marsaglia's 32-bit xorshift (13, 17, 5), four a step, low byte first."""
    x = (seed & 0xFFFFFFFF) or 1
    out = bytearray(n)
    for i in range(0, n, 4):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i:i + 4] = x.to_bytes(4, "little")[:n - i]
    return bytes(out)


def synth(spec: dict) -> Image.Image:
    """noise: the bytes themselves; smooth: diagonal ramps plus a little noise (20..226, never clipped);
    blocks: 8 x 8 squares of 0 or 255 a channel (hard edges, where bicubic and lanczos overshoot); black."""
    kind, w, h, mode = spec["kind"], spec["w"], spec["h"], spec.get("mode", "RGB")
    ch = CHANNELS[mode]
    n = w * h * ch
    r = xorshift_bytes(spec.get("seed", 1), n) if kind != "black" else bytes(n)
    if kind in ("noise", "black"):
        data = r
    elif kind == "smooth":
        data = bytes(20 + (x * 5 + y * 3 + c * 50) % 200 + (r[(y * w + x) * ch + c] & 7)
                     for y in range(h) for x in range(w) for c in range(ch))
    elif kind == "blocks":
        bw = (w + 7) // 8
        data = bytes(255 if r[((y // 8) * bw + x // 8) * ch + c] & 1 else 0
                     for y in range(h) for x in range(w) for c in range(ch))
    else:
        raise ValueError(kind)
    return Image.frombytes(mode, (w, h), data)


def pic(kind: str, w: int, h: int, seed: int = 1, mode: str = "RGB") -> dict:
    return {"kind": kind, "w": w, "h": h, "seed": seed, "mode": mode}


# --- correctly rounded sin and cos, and Pillow's rotation matrix --------------------------------------------------

def cr_sin_cos(t: float) -> tuple[float, float]:
    """sin(t) and cos(t) rounded correctly: Taylor series in 90-digit decimals (float(Decimal) rounds correctly)."""
    with localcontext() as ctx:
        ctx.prec = 90
        x = Decimal(t)
        x2 = x * x
        s = term = x
        n = 1
        while abs(term) > Decimal("1e-88"):
            term = -term * x2 / ((2 * n) * (2 * n + 1))
            s += term
            n += 1
        c = term = Decimal(1)
        n = 1
        while abs(term) > Decimal("1e-88"):
            term = -term * x2 / ((2 * n - 1) * (2 * n))
            c += term
            n += 1
        return float(s), float(c)


def libm_sin_cos(t: float) -> tuple[float, float]:
    return math.sin(t), math.cos(t)


def rotate_matrix(size, angle, center, expand, sin_cos=cr_sin_cos):
    """Image.rotate's matrix and output size (Pillow 12.3.0, src/PIL/Image.py), with cos and sin from `sin_cos`."""
    w, h = size
    angle = angle % 360.0
    post_trans = (0, 0)
    if center is None:
        center = (w / 2, h / 2)
    angle = -math.radians(angle)
    s, c = sin_cos(angle)
    matrix = [round(c, 15), round(s, 15), 0.0, round(-s, 15), round(c, 15), 0.0]

    def transform(x, y, matrix):
        a, b, c, d, e, f = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(-center[0] - post_trans[0], -center[1] - post_trans[1], matrix)
    matrix[2] += center[0]
    matrix[5] += center[1]
    if expand:
        xx, yy = [], []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            tx, ty = transform(x, y, matrix)
            xx.append(tx)
            yy.append(ty)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
        w, h = nw, nh
    return matrix, (w, h)


def spied_rotate(im: Image.Image, angle: float, **kw) -> tuple[Image.Image, dict | None]:
    """im.rotate(angle, **kw), and the size and matrix it hands Image.transform (None on a fast path)."""
    seen: dict = {}
    orig = Image.Image.transform

    def spy(self, size, method, data=None, resample=Image.Resampling.NEAREST, fill=1, fillcolor=None):
        seen.update(size=tuple(size), matrix=list(data))
        return orig(self, size, method, data, resample, fill, fillcolor)

    Image.Image.transform = spy
    try:
        out = im.rotate(angle, **kw)
    finally:
        Image.Image.transform = orig
    return out, (seen or None)


def rotated(im: Image.Image, step: dict, name: str) -> Image.Image:
    resample = FILTERS[step.get("resample", "nearest")]
    center = tuple(step["center"]) if step.get("center") is not None else None
    fill = tuple(step["fill"]) if step.get("fill") is not None else None
    kw: dict = {"resample": resample, "expand": step.get("expand", False)}
    if center is not None:
        kw["center"] = center
    if fill is not None:
        kw["fillcolor"] = fill
    out, seen = spied_rotate(im, step["angle"], **kw)
    if seen is None:
        return out
    theirs = rotate_matrix(im.size, step["angle"], center, step.get("expand", False), libm_sin_cos)
    assert (theirs[0], theirs[1]) == (seen["matrix"], seen["size"]), f"{name}: the matrix replica is not Pillow's"
    ours, size = rotate_matrix(im.size, step["angle"], center, step.get("expand", False))
    if ours != seen["matrix"]:
        fixed = im.transform(size, Image.Transform.AFFINE, ours, resample, fillcolor=fill)
        same = fixed.tobytes() == out.tobytes()
        NOTES.append(f"{name}: the C library's cos/sin round differently at {step['angle']} degrees; "
                     f"the pixels are {'the same' if same else 'different'}")
        return fixed
    return out


# --- running the steps -----------------------------------------------------------------------------------------

def apply(im: Image.Image, steps: list[dict], name: str) -> Image.Image:
    for step in steps:
        op = step["op"]
        if op == "crop":
            im = im.crop(tuple(step["box"]))
        elif op == "resize":
            im = im.resize(tuple(step["size"]), FILTERS[step["resample"]])
        elif op == "rotate":
            im = rotated(im, step, name)
        elif op == "gray":
            im = im.convert("L")
        elif op == "rgb":
            im = im.convert("RGB")
        elif op == "paste":
            dst = synth(step["onto"])
            dst.paste(im, tuple(step["at"]))
            im = dst
        else:
            raise ValueError(op)
    return im


def samples(im: Image.Image) -> list:
    w, h = im.size
    if not w or not h:
        return []
    out = []
    for x, y in ((0, 0), (w - 1, h - 1), (w // 2, h // 2), (w // 3, 2 * h // 3)):
        v = im.getpixel((x, y))
        out.append([x, y, list(v) if isinstance(v, tuple) else [v]])
    return out


def result(im: Image.Image) -> dict:
    return {"mode": im.mode, "size": list(im.size), "sha256": hashlib.sha256(im.tobytes()).hexdigest(), "px": samples(im)}


# --- the synthetic cases -----------------------------------------------------------------------------------------

def crop_cases():
    src = pic("noise", 31, 17, 11)
    boxes = [(0, 0, 31, 17), (5, 3, 20, 11), (-4, -3, 10, 8), (25, 10, 40, 30), (-10, -10, 50, 40), (40, 40, 50, 50),
             (-20, -20, -5, -5), (3, 3, 3, 10), (3, 3, 10, 3), (2.5, 3.5, 10.5, 11.5), (1.4999, 0.5001, 7.5, 9.5),
             (-0.5, -1.5, 5.5, 6.5), (30.5, 16.5, 31.5, 17.5), (0.49, 0.51, 30.5, 16.5), (-3.5, 2.5, -2.5, 4.5)]
    for b in boxes:
        yield f"crop {b}", src, [{"op": "crop", "box": list(b)}]
    for b in [(4, 2, 19, 13), (-6, -2, 12, 9), (29.5, 5.5, 40.5, 20.5)]:
        yield f"crop grey {b}", pic("noise", 31, 17, 12, "L"), [{"op": "crop", "box": list(b)}]


RESIZE_SIZES = [(32, 24), (21, 16), (25, 19), (9, 7), (3, 2), (1, 1), (64, 1), (1, 48), (100, 70), (128, 96), (237, 178),
                (100, 20), (13, 150), (64, 30), (40, 48), (64, 48), (65, 48), (64, 49), (5, 200)]


def resize_cases():
    for f in FILTERS:
        for size in RESIZE_SIZES:
            yield f"resize {f} 64x48 noise -> {size}", pic("noise", 64, 48, 21), [{"op": "resize", "size": list(size), "resample": f}]
        yield f"resize {f} 64x48 noise -> (640, 480)", pic("noise", 64, 48, 22), [{"op": "resize", "size": [640, 480], "resample": f}]
        for size in [(11, 7), (74, 46), (5, 23), (37, 5), (1, 1), (36, 22), (38, 24)]:
            yield f"resize {f} 37x23 smooth -> {size}", pic("smooth", 37, 23, 23), [{"op": "resize", "size": list(size), "resample": f}]
        for size in [(17, 23), (90, 90), (7, 7), (41, 39)]:
            yield f"resize {f} 40x40 blocks -> {size}", pic("blocks", 40, 40, 24), [{"op": "resize", "size": list(size), "resample": f}]
        for (w, h), size in [((1, 1), (5, 5)), ((1, 1), (1, 7)), ((1, 9), (3, 4)), ((9, 1), (2, 3)), ((2, 2), (1, 1)),
                             ((300, 3), (7, 20)), ((300, 3), (3, 2)), ((2, 250), (1, 20)), ((2, 250), (3, 11)),
                             ((2, 250), (2, 100)), ((3, 400), (5, 30)), ((1, 150), (1, 7)), ((1, 150), (4, 3)),
                             ((2, 150), (1, 9)), ((1, 101), (1, 100)), ((1, 101), (2, 101)), ((500, 2), (9, 1))]:
            yield f"resize {f} {w}x{h} noise -> {size}", pic("noise", w, h, 25), [{"op": "resize", "size": list(size), "resample": f}]
        for size in [(32, 24), (13, 150), (48, 64), (1, 1), (100, 70), (7, 48)]:
            yield f"resize {f} grey 64x48 noise -> {size}", pic("noise", 64, 48, 26, "L"), [{"op": "resize", "size": list(size), "resample": f}]
        yield f"resize {f} grey 2x250 -> (1, 20)", pic("noise", 2, 250, 27, "L"), [{"op": "resize", "size": [1, 20], "resample": f}]


ANGLES = [0, 0.3, 17, 45, 89.9, 90, 90.0000001, 135.5, 179.5, 180, 200, 270, 300, -33, -90, -179.99, 359.7, 360, 720.5,
          1e-7, 123.456789, -1e-12]


def rotate_cases():
    src = pic("noise", 40, 30, 31)
    for f in ("nearest", "bilinear", "bicubic"):
        for a in ANGLES:
            yield f"rotate {f} {a}", src, [{"op": "rotate", "angle": a, "resample": f}]
            yield f"rotate {f} {a} expand", src, [{"op": "rotate", "angle": a, "resample": f, "expand": True}]
            yield f"rotate {f} {a} about (13.37, 9.81)", src, [{"op": "rotate", "angle": a, "resample": f, "center": [13.37, 9.81]}]
        for a in (17, 45, 90, -33, 179.5):
            for expand in (False, True):
                yield (f"rotate {f} {a} fill{' expand' if expand else ''}", src,
                       [{"op": "rotate", "angle": a, "resample": f, "expand": expand, "fill": [200, 30, 90]}])
        sq = pic("smooth", 33, 33, 32)
        for a in (90, 270, 180, -90, 30):
            yield f"rotate {f} square {a}", sq, [{"op": "rotate", "angle": a, "resample": f}]
            yield f"rotate {f} square {a} about the middle", sq, [{"op": "rotate", "angle": a, "resample": f, "center": [16.5, 16.5]}]
        for c in ([20, 15], [-5.5, 40.25], [0, 0], [39.999, 29.5]):
            yield f"rotate {f} 37 about {c}", src, [{"op": "rotate", "angle": 37, "resample": f, "center": c}]
            yield f"rotate {f} 37 about {c} expand", src, [{"op": "rotate", "angle": 37, "resample": f, "center": c, "expand": True}]
        blocks = pic("blocks", 25, 25, 33)
        for a in (12.5, 77, -140):
            yield f"rotate {f} blocks {a}", blocks, [{"op": "rotate", "angle": a, "resample": f, "center": [12.1, 11.7]}]
        for (w, h) in ((1, 1), (1, 9), (7, 1)):
            for a in (90, 33.3):
                yield f"rotate {f} {w}x{h} {a} expand", pic("noise", w, h, 34), [{"op": "rotate", "angle": a, "resample": f, "expand": True}]
    big = pic("noise", 3, 3, 35)
    yield "rotate nearest 3x3 far centre (float path)", big, [{"op": "rotate", "angle": 10, "center": [40000.5, -3.25]}]
    yield "rotate bicubic 3x3 far centre", big, [{"op": "rotate", "angle": 10, "resample": "bicubic", "center": [40000.5, -3.25]}]


def card_steps(cx: float, cy: float, long_px: float, short_px: float, angle_deg: float) -> list[dict]:
    """live/pipeline.card_crop as steps: the card's neighbourhood, turned upright about the card's centre, and
    the card's box. Checked against card_crop itself on the real frames."""
    r = math.ceil(math.hypot(long_px, short_px) / 2) + 2
    x0, y0 = int(cx) - r, int(cy) - r
    lx, ly = cx - x0, cy - y0
    w, h = short_px, long_px
    return [{"op": "crop", "box": [x0, y0, x0 + 2 * r, y0 + 2 * r]},
            {"op": "rotate", "angle": angle_deg - 90, "resample": "bicubic", "center": [lx, ly]},
            {"op": "crop", "box": [round(lx - w / 2), round(ly - h / 2), round(lx + w / 2), round(ly + h / 2)]}]


def detail_steps(w: int, h: int) -> list[dict]:
    """matcrops.detail's grey middle, 48 x 64."""
    return [{"op": "gray"}, {"op": "crop", "box": [round(w * 0.12), round(h * 0.12), round(w * 0.88), round(h * 0.88)]},
            {"op": "resize", "size": [48, 64], "resample": "box"}]


def letterbox_steps(w: int, h: int, size: int = 224) -> list[dict]:
    """encoders.letterbox: onto a black square, then a bicubic resize."""
    side = max(w, h)
    return [{"op": "paste", "onto": pic("black", side, side, 0), "at": [(side - w) // 2, (side - h) // 2]},
            {"op": "resize", "size": [size, size], "resample": "bicubic"}]


CARDS = [(101.37, 77.62, 88.0, 63.0, 3.7), (60.5, 90.25, 70.3, 50.33, 91.2), (140.9, 60.1, 60.0, 42.95, 178.9),
         (99.99, 75.01, 110.0, 78.75, 45.0), (30.2, 30.7, 40.0, 28.6, 0.0), (170.4, 130.6, 65.5, 46.9, 120.25)]


def chain_cases():
    src = pic("smooth", 200, 150, 41)
    for i, (cx, cy, lng, sht, a) in enumerate(CARDS):
        steps = card_steps(cx, cy, lng, sht, a)
        yield f"card_crop {i}", src, steps
        crop_box = steps[2]["box"]  # the crop's size: its box is rounded edge by edge
        w, h = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
        for turn in (90, 180, 270):
            yield f"card_crop {i} turned {turn}", src, steps + [{"op": "rotate", "angle": turn, "expand": True}]
        yield f"card_crop {i} detail", src, steps + detail_steps(w, h)
        yield f"card_crop {i} letterbox", src, steps + letterbox_steps(w, h)
    noise = pic("noise", 20, 10, 42)
    for at in ((5, 10), (-5, -3), (25, 25), (-30, 0), (0, 0), (10, 20), (-19, -9), (29, 29)):
        yield f"paste 20x10 at {at}", noise, [{"op": "paste", "onto": pic("smooth", 30, 30, 43), "at": list(at)}]
    yield "paste 20x10 onto 20x10", noise, [{"op": "paste", "onto": pic("black", 20, 10, 0), "at": [0, 0]}]
    yield "grey 97x31 noise", pic("noise", 97, 31, 44), [{"op": "gray"}]
    yield "grey 16x16 blocks", pic("blocks", 16, 16, 45), [{"op": "gray"}]
    yield "rgb of rgba 13x7", pic("noise", 13, 7, 46, "RGBA"), [{"op": "rgb"}]
    yield "detail of 87x121", pic("smooth", 87, 121, 47), detail_steps(87, 121)
    yield "letterbox 43x60", pic("noise", 43, 60, 48), letterbox_steps(43, 60)
    yield "letterbox 60x43", pic("noise", 60, 43, 49), letterbox_steps(60, 43)
    yield "window resize then tile", pic("smooth", 119, 101, 50), [
        {"op": "resize", "size": [54, 46], "resample": "bicubic"}, {"op": "crop", "box": [0, 0, 57, 57]}]


def matrix_sweep(n: int = 20000) -> dict:
    """rotateMatrix over many angles, sizes, centres and expands (test/image.test.ts builds the same list): the
    SHA-256 of every correctly rounded matrix and size as little-endian doubles, and the cases where the C
    library's cos and sin give Pillow another matrix."""
    x = 0x9E3779B9
    packed = bytearray()
    differ = []
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        angle = x / 4294967296 * 720 - 360 if i % 2 else i * 0.0137 - 137
        w, h = 40 + i % 13, 30 + i % 11
        center = None if i % 4 < 2 else (17.3 + (i % 7) * 0.61, 9.1 + (i % 5) * 1.37)
        expand = i % 3 == 0
        m, size = rotate_matrix((w, h), angle, center, expand)
        packed += struct.pack("<8d", *m, *size)
        if (m, size) != rotate_matrix((w, h), angle, center, expand, libm_sin_cos):
            differ.append(i)
    return {"n": n, "sha256": hashlib.sha256(bytes(packed)).hexdigest(), "libm_differs": differ}


def matrix_cases() -> list[dict]:
    out = []
    for (w, h), a, c, e in [((40, 30), 17, None, False), ((40, 30), 17, None, True), ((40, 30), -33, (13.37, 9.81), False),
                            ((250, 250), 91.2 - 90, (125.4, 124.9), False), ((40, 30), 90, None, False), ((33, 33), 0.3, None, True),
                            ((40, 30), 0, (1, 2), False), ((10, 7), 180, (5, 3.5), False)]:
        m, size = rotate_matrix((w, h), a, c, e)
        out.append({"size": [w, h], "angle": a, "center": list(c) if c else None, "expand": e, "matrix": m, "out": list(size)})
    return out


def vectors() -> dict:
    cases = []
    for gen in (crop_cases, resize_cases, rotate_cases, chain_cases):
        for name, spec, steps in gen():
            out = apply(synth(spec), steps, name)
            cases.append({"name": name, "input": spec, "steps": steps, **result(out)})
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "case names must be unique"
    return {"pillow": PIL.__version__, "note": "Pillow's output for each case's steps on its synthetic input "
            "(test/gen/image.py); no pixels are stored", "matrices": matrix_cases(), "sweep": matrix_sweep(),
            "notes": NOTES, "cases": cases}


# --- the real frames (private) -----------------------------------------------------------------------------------

def real(m3: Path) -> dict:
    """The live runner's picture calls on three frames of the LA final, as steps with Pillow's hashes."""
    import numpy as np

    from rifteye_ml.detect.geometry import tile_origins
    from rifteye_ml.encoders import letterbox
    from rifteye_ml.live.layouts import LAYOUTS
    from rifteye_ml.live.pipeline import card_crop
    from rifteye_ml.matcrops import CardBox, detail

    meta = json.loads((m3 / "frames/la-final-rgb/frames.json").read_text())
    layout = LAYOUTS["la-rq"]
    cases = []

    def add(name: str, frame: str, steps: list[dict], im: Image.Image, expect: Image.Image | None = None, **extra):
        out = apply(im, steps, name)
        if expect is not None:
            assert out.tobytes() == expect.tobytes() and out.size == expect.size, f"{name}: the steps are not the ml call"
        cases.append({"name": name, "frame": frame, "steps": steps, **result(out), **extra})
        return out

    for f in meta["frames"]:
        w, h = f["width"], f["height"]
        raw = (m3 / "frames/la-final-rgb" / f["file"]).read_bytes()
        image = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        frame = Image.fromarray(image)
        name = f["file"]
        x0, y0, x1, y1 = layout.box(w, h)
        add(f"{name} Scene.small", name, [{"op": "resize", "size": [96, 54], "resample": "box"}], frame,
            Image.fromarray(image).resize((96, 54), Image.Resampling.BOX))
        add(f"{name} table_like", name, [{"op": "crop", "box": [x0, y0, x1, y1]}, {"op": "resize", "size": [160, 120], "resample": "box"}],
            frame, Image.fromarray(image[y0:y1, x0:x1]).resize((160, 120), Image.Resampling.BOX))
        add(f"{name} grey", name, [{"op": "gray"}], frame)
        # the detector's window at the layout's card size, and at two others as autolayout.card_size tries them
        for card_px in (layout.card_px(h), 131.0, 200.0):
            scale = 70.0 / card_px
            sw, sh = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
            win = [{"op": "crop", "box": [x0, y0, x1, y1]}, {"op": "resize", "size": [sw, sh], "resample": "bicubic"}]
            add(f"{name} detector window at {card_px:g} px", name, win, frame)
            for ty in tile_origins(sh, 576, 0.2):
                for tx in tile_origins(sw, 576, 0.2):
                    add(f"{name} detector tile ({tx}, {ty}) at {card_px:g} px", name, win + [{"op": "crop", "box": [tx, ty, tx + 576, ty + 576]}], frame)
        # the change gate's view of the table
        tx0, ty0, tx1, ty1 = layout.table
        vw = 320
        vh = round(vw * (ty1 - ty0) * 1080 / ((tx1 - tx0) * 1920) / 2) * 2
        add(f"{name} watch view", name, [{"op": "crop", "box": [x0, y0, x1, y1]}, {"op": "resize", "size": [vw, vh], "resample": "bilinear"}],
            frame, Image.fromarray(image[y0:y1, x0:x1]).resize((vw, vh), Image.Resampling.BILINEAR))
        # card crops over the table: a grid of fractional centres, each at a few turns and sizes
        k = 0
        for gy in range(3):
            for gx in range(4):
                cx = x0 + (gx + 0.5) * (x1 - x0) / 4 + 0.37 * gx
                cy = y0 + (gy + 0.5) * (y1 - y0) / 3 + 0.61 * gy
                lng = 155.0 * (0.8 + 0.1 * ((gx + gy) % 4)) + 0.25
                sht = lng * 63 / 88
                angle = [0.0, 2.3, 45.0, 87.6, 90.0, 91.2, 135.0, 178.9, 12.25, 60.5, 100.75, 169.3][k % 12]
                k += 1
                box = CardBox((cx, cy), lng, sht, angle, 1.0)
                steps = card_steps(cx, cy, lng, sht, angle)
                crop_ = add(f"{name} card_crop {k}", name, steps, frame, card_crop(frame, box))
                for turn in (90, 180, 270):
                    add(f"{name} card_crop {k} turned {turn}", name, steps + [{"op": "rotate", "angle": turn, "expand": True}], frame,
                        crop_.rotate(turn, expand=True))
                g = add(f"{name} card_crop {k} detail", name, steps + detail_steps(*crop_.size), frame)
                a = np.asarray(g, np.float32)
                value = float((np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2)
                assert value == detail(crop_), f"{name} card_crop {k}: the detail steps are not matcrops.detail"
                cases[-1]["detail"] = value
                add(f"{name} card_crop {k} letterbox", name, steps + letterbox_steps(*crop_.size), frame, letterbox(crop_, 224))
        # a region the gate saw change, turned portrait (Recognizer.watch)
        region = [{"op": "crop", "box": [700, 400, 911, 551]}, {"op": "rotate", "angle": 90, "expand": True}]
        add(f"{name} watch region", name, region, frame, frame.crop((700, 400, 911, 551)).rotate(90, expand=True))
    return {"pillow": PIL.__version__, "note": "Pillow's output of the live runner's picture calls on "
            "frames/la-final-rgb (test/gen/image.py --real); private, D-006", "notes": NOTES, "cases": cases}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", metavar="M3", help="write M3/fixtures/image/real.json from M3/frames/la-final-rgb instead")
    args = ap.parse_args(argv)
    if PIL.__version__ != PILLOW:
        sys.exit(f"Pillow {PIL.__version__}: image.ts ports {PILLOW}, whose bytes these must be")
    if args.real:
        m3 = Path(args.real).expanduser()
        out = m3 / "fixtures/image/real.json"
        data = real(m3)
    else:
        out = VECTORS
        data = vectors()
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = data.pop("cases")
    body = json.dumps(data, separators=(",", ":"))[:-1]
    text = body + ',"cases":[\n' + ",\n".join(json.dumps(c, separators=(",", ":")) for c in cases) + "\n]}\n"
    out.write_text(text, encoding="utf-8")
    print(f"{len(cases)} cases -> {out} ({len(text) // 1024} KB)")
    for n in NOTES:
        print("note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
