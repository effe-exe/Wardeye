# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Compose a sampled board into a broadcast frame, with exact annotations.

1. **Table plane.** Mats on a table, then every card in board order: art or a sleeve back,
   sleeve rim, foil and glare (shared with the M0 stream simulator), rounded corners, and a soft
   shadow on whatever lies below. Dice, counters, markers and hands go on top. An id map records
   which card is uppermost at each pixel.
2. **Camera.** One mesh warp maps the frame's window back to the table plane: a quarter turn and
   yaw, keystone tilt and radial lens distortion. Colour and id map go through the same warp,
   then white balance, exposure, gamma, uneven light and defocus.
3. **Broadcast.** The window fills the frame, sits between side panels (the RQ package seen in
   M0) or is a picture-in-picture over a player camera.

Every card's full quad (TL, TR, BR, BL of the upright card, so it carries the orientation), its
visible fraction and visible box come from the same geometry as the pixels, with no labelling.
Face-down cards carry no identity. The codec pass happens afterwards, per clip (`__main__`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..degrade import StreamSettings, _add_glare, _camera_photometry, _rainbow
from .layout import CARD_H_MM, CARD_W_MM, MAT_H_MM, MAT_W_MM, Board, Instance, Occluder

OCCLUDER_ID = 65535   # id map value under a hand, die, counter or marker
GRAPHIC_ID = 65534    # a broadcast graphic showing a card (never a table card)


@dataclass(frozen=True)
class Shot:
    """How one board is filmed and broadcast."""
    frame_w: int = 1920
    frame_h: int = 1080
    layout: str = "rq"               # full | rq | pip
    window: tuple[int, int, int, int] = (365, 0, 1555, 1080)  # where the table camera sits in the frame
    card_px: float = 130.0           # a card's long side at the centre of the view, in frame px
    turn: int = 0                    # quarter turns of the table in view (players left/right or top/bottom)
    yaw_deg: float = 0.0
    keystone: float = 0.0            # the far edge of the view covers this much more table
    k1: float = 0.0                  # radial lens distortion (negative: barrel)
    centre_mm: tuple[float, float] = (0.0, 0.0)
    defocus_px: float = 0.4
    saturation: float = 1.0          # the camera's colour saturation (below 1: muted)
    black_lift: float = 0.0          # blacks lifted to this level (0-1)
    knee: float = 1.0                # highlights roll off above this level
    wb_jitter: float = 0.05
    exposure_jitter: float = 0.15
    gamma_jitter: float = 0.1
    light_jitter: float = 0.12
    featured: bool = True            # the RQ package's featured-card graphic


def layout_window(layout: str, frame_w: int, frame_h: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """The table camera's rectangle in the frame: all of it, between the RQ side panels, or a PiP."""
    if layout == "full":
        return 0, 0, frame_w, frame_h
    if layout == "rq":
        return round(0.19 * frame_w), 0, round(0.81 * frame_w), frame_h
    ww = round(float(rng.uniform(0.35, 0.6)) * frame_w); wh = round(ww * 9 / 16)
    x0 = int(rng.integers(0, frame_w - ww + 1)); y0 = int(rng.integers(0, frame_h - wh + 1))
    return x0, y0, x0 + ww, y0 + wh


def sample_shot(rng: np.random.Generator, frame_w: int, frame_h: int, layout: str,
                view_mm: tuple[float, float] = (550.0, 950.0)) -> Shot:
    """A camera and broadcast for one board. The camera frames `view_mm` of table vertically
    (log-uniform), so card size follows from the window: M0's broadcasts showed 130-166 px cards
    full screen at 1080p, and a picture-in-picture shows far smaller ones."""
    window = layout_window(layout, frame_w, frame_h, rng)
    view_h = float(np.exp(rng.uniform(np.log(view_mm[0]), np.log(view_mm[1]))))
    card_px = (window[3] - window[1]) * CARD_H_MM / view_h
    return Shot(frame_w=frame_w, frame_h=frame_h, layout=layout, window=window, card_px=card_px, turn=int(rng.integers(4)),
                yaw_deg=float(rng.uniform(-3, 3)), keystone=float(rng.uniform(0, 0.12)), k1=float(rng.uniform(-0.06, 0.01)),
                centre_mm=(float(rng.normal(0, 25)), float(rng.normal(0, 25))),
                defocus_px=float(rng.uniform(0.5, 1.4)) * max(0.5, card_px / 130), saturation=float(rng.uniform(0.6, 0.95)),
                black_lift=float(rng.uniform(0.02, 0.1)), knee=float(rng.uniform(0.75, 0.95)), featured=bool(rng.random() < 0.6))


# ------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------

def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 H with dst ~ H @ src for four point pairs."""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


class Camera:
    """Table mm -> window px: a homography, then radial distortion about the window centre."""

    def __init__(self, shot: Shot, win_w: int, win_h: int):
        self.w, self.h, self.k1 = win_w, win_h, shot.k1
        ppm = shot.card_px / CARD_H_MM
        vw, vh = win_w / ppm, win_h / ppm
        k = shot.keystone
        # The view on the table: a trapezoid, wider and deeper on the far side (the top of the window).
        near = np.array([[-vw / 2, vh / 2], [vw / 2, vh / 2]])
        far = np.array([[-vw * (1 + k) / 2, -vh * (1 + k / 2) / 2], [vw * (1 + k) / 2, -vh * (1 + k / 2) / 2]])
        quad = np.array([far[0], far[1], near[1], near[0]])  # window TL, TR, BR, BL
        t = np.deg2rad(shot.turn * 90 + shot.yaw_deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        quad = quad @ rot.T + np.asarray(shot.centre_mm)
        self.view_mm = quad
        self.H = homography(quad, np.array([[0, 0], [win_w, 0], [win_w, win_h], [0, win_h]], np.float64))
        self.Hinv = np.linalg.inv(self.H)
        self.c = np.array([win_w / 2, win_h / 2]); self.r0 = float(np.hypot(win_w, win_h) / 2)

    def distort(self, p: np.ndarray) -> np.ndarray:
        d = (p - self.c) / self.r0
        r2 = (d ** 2).sum(axis=1, keepdims=True)
        return self.c + d * (1 + self.k1 * r2) * self.r0

    def undistort(self, q: np.ndarray) -> np.ndarray:
        d = (q - self.c) / self.r0
        u = d.copy()
        for _ in range(8):  # fixed point: u (1 + k1 |u|^2) = d
            u = d / (1 + self.k1 * (u ** 2).sum(axis=1, keepdims=True))
        return self.c + u * self.r0

    def to_window(self, mm: np.ndarray) -> np.ndarray:
        return self.distort(apply_h(self.H, mm))

    def to_table(self, win: np.ndarray) -> np.ndarray:
        return apply_h(self.Hinv, self.undistort(win))


def camera_tone(img: np.ndarray, saturation: float, black_lift: float, knee: float) -> np.ndarray:
    """A broadcast camera's look: colours pulled towards grey, blacks lifted, highlights rolled off."""
    a = img.astype(np.float32) / 255
    lum = (a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    a = lum + (a - lum) * saturation
    a = black_lift + (1 - black_lift) * a
    if knee < 1:
        over = np.maximum(a - knee, 0)
        a = np.where(over > 0, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), a)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def rotate_pts(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Counter-clockwise on screen (y down), like PIL's Image.rotate."""
    t = np.deg2rad(angle_deg)
    return np.c_[pts[:, 0] * np.cos(t) + pts[:, 1] * np.sin(t), -pts[:, 0] * np.sin(t) + pts[:, 1] * np.cos(t)]


def card_quad_mm(inst: Instance) -> np.ndarray:
    """TL, TR, BR, BL of the card as printed, on the table."""
    w, h = (CARD_H_MM, CARD_W_MM) if inst.landscape else (CARD_W_MM, CARD_H_MM)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return rotate_pts(corners, inst.angle) + np.asarray(inst.centre)


def polygon_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


# ------------------------------------------------------------------------------------------
# The table plane
# ------------------------------------------------------------------------------------------

def _smooth_noise(rng: np.random.Generator, w: int, h: int, cells: int, sigma: float) -> np.ndarray:
    lo = rng.normal(0, sigma, size=(max(2, h // cells), max(2, w // cells), 3)) + 128
    return np.asarray(Image.fromarray(np.clip(lo, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) - 128


# Dark, muted playmat colours like those on the M0 broadcasts (navy, red, black, green, purple, grey).
MAT_COLOURS = np.array([(22, 32, 58), (30, 44, 80), (120, 26, 30), (20, 20, 24), (26, 52, 38), (52, 30, 70), (56, 58, 62), (70, 44, 30)],
                       np.float32)


def mat_image(rng: np.random.Generator, w: int, h: int, mats: Sequence[Image.Image] = ()) -> np.ndarray:
    """A playmat: a local image when one is supplied (official art is Riot IP and never bundled),
    else a dark base with blotches and a large faint printed emblem."""
    if mats:
        m = mats[int(rng.integers(len(mats)))].convert("RGB").resize((w, h), Image.BICUBIC)
        return np.asarray(m, np.float32)
    base = MAT_COLOURS[int(rng.integers(len(MAT_COLOURS)))] * rng.uniform(0.75, 1.2) + rng.normal(0, 6, size=3)
    arr = base + _smooth_noise(rng, w, h, max(8, w // 10), 7) + rng.normal(0, 2.5, size=(h, w, 1))
    emb = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(emb)
    cx, cy, r = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.3, 0.7), min(w, h) * rng.uniform(0.25, 0.45)
    for k in range(int(rng.integers(1, 4))):
        rr = r * (1 - 0.25 * k)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], float(rng.uniform(0, 360)), float(rng.uniform(0, 360)) + 200,
              fill=255, width=max(2, int(r * rng.uniform(0.04, 0.12))))
    emb = np.asarray(emb.filter(ImageFilter.GaussianBlur(max(1, r * 0.02))), np.float32)[..., None] / 255
    tint = rng.uniform(-35, 35, size=3)
    return np.clip(arr + emb * tint, 0, 255)


def card_back(sleeve: tuple[int, int, int] | None, style: str, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """A face-down card: a sleeve back (nearly one colour, glossy) or a printed pattern."""
    w, h = size
    colour = np.asarray(sleeve if sleeve is not None else (40, 40, 60), np.float32)
    arr = colour + rng.normal(0, 1.5, size=(h, w, 1)) + _smooth_noise(rng, w, h, max(4, w // 3), 3)[..., :1]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if style == "pattern":  # an unsleeved printed back: a centred emblem and a border
        d = ImageDraw.Draw(img)
        light = tuple(int(min(255, c + 90)) for c in colour)
        d.rectangle([w * 0.06, h * 0.05, w * 0.94, h * 0.95], outline=light, width=max(1, w // 40))
        d.ellipse([w * 0.25, h * 0.35, w * 0.75, h * 0.65], outline=light, width=max(1, w // 30))
        d.line([w * 0.5, h * 0.2, w * 0.5, h * 0.8], fill=light, width=max(1, w // 50))
    return img


def printed(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A physical print is not the digital art: its own contrast, saturation, brightness and cast."""
    a = img.astype(np.float32) / 255
    lum = (a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    a = lum + (a - lum) * rng.uniform(0.8, 1.15)
    a = np.clip((a - 0.5) * rng.uniform(0.85, 1.1) + 0.5 + rng.uniform(-0.06, 0.06), 0, 1) ** np.exp(rng.uniform(-0.2, 0.2))
    return np.clip(a * rng.uniform(0.94, 1.06, size=3) * 255, 0, 255).astype(np.uint8)


def foil(card: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Holographic foil under stage light, as the M0 broadcasts show it: a rainbow sheen drifting
    across the card and a bright band where the light catches it, stronger than the M0 simulator's."""
    h, w = card.shape[:2]
    th = rng.uniform(0, np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs * np.cos(th) + ys * np.sin(th)) / max(w, h)
    sheen = _rainbow((u * rng.uniform(0.8, 3.0) + rng.uniform(0, 1)) % 1.0)
    band = np.exp(-(((u - rng.uniform(-0.2, 1.2)) / rng.uniform(0.08, 0.3)) ** 2))
    a = (rng.uniform(0.25, 0.5) * (0.5 + band))[..., None]
    out = card.astype(np.float32) * (1 - a) + sheen * 255 * a + 70 * band[..., None] * a
    # the whole card's hue drifts with the viewing angle
    hsv = np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(rng.uniform(-30, 30))) % 256
    return np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))


def sleeve_streak(card: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A glossy sleeve reflecting a stage light: a long bright streak across the card."""
    h, w = card.shape[:2]
    th = rng.uniform(0, np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xs - rng.uniform(0, w)) * np.cos(th) + (ys - rng.uniform(0, h)) * np.sin(th)) / max(w, h)
    m = np.exp(-(d / rng.uniform(0.04, 0.15)) ** 2)[..., None] * rng.uniform(0.3, 0.7)
    a = card.astype(np.float32)
    return np.clip(a + (255 - a) * m, 0, 255).astype(np.uint8)


def _with_sleeve(img: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    """The rim of a sleeve's coloured back, about 1.5 mm around the card."""
    w, h = img.size
    pad = max(1, round(h * 1.5 / 88))
    out = Image.new("RGB", (w, h), colour)
    out.paste(img.resize((w - 2 * pad, h - 2 * pad), Image.BILINEAR), (pad, pad))
    return out


def _rounded_alpha(w: int, h: int, r: float) -> Image.Image:
    a = Image.new("L", (w, h), 0)
    ImageDraw.Draw(a).rounded_rectangle([0, 0, w - 1, h - 1], radius=max(1, r), fill=255)
    return a


class Canvas:
    """The table plane over a region of the table, at `ppm` px per mm."""

    def __init__(self, x0: float, y0: float, x1: float, y1: float, ppm: float):
        self.x0, self.y0, self.ppm = x0, y0, ppm
        self.w, self.h = int(np.ceil((x1 - x0) * ppm)), int(np.ceil((y1 - y0) * ppm))
        self.rgb = np.zeros((self.h, self.w, 3), np.float32)
        self.ids = np.zeros((self.h, self.w), np.int32)

    def px(self, mm: Sequence[float]) -> tuple[float, float]:
        return (mm[0] - self.x0) * self.ppm, (mm[1] - self.y0) * self.ppm

    def paste(self, patch: np.ndarray, alpha: np.ndarray, cx: float, cy: float, ident: int | None,
              shadow: float = 0.0, rng: np.random.Generator | None = None) -> None:
        """Alpha-blend `patch` (h×w×3) centred at canvas px (cx, cy); record `ident` where it is opaque."""
        ph, pw = alpha.shape
        x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
        if shadow > 0:
            off = max(1, int(round(1.5 * self.ppm)))
            blur = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(1.0, 1.4 * self.ppm)))
            self._blend(np.zeros_like(patch), np.asarray(blur, np.float32) / 255 * shadow, x0 + off // 2, y0 + off, None)
        self._blend(patch, alpha, x0, y0, ident)

    def _blend(self, patch, alpha, x0, y0, ident):
        ph, pw = alpha.shape
        sx0, sy0, sx1, sy1 = max(0, -x0), max(0, -y0), min(pw, self.w - x0), min(ph, self.h - y0)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        a = alpha[sy0:sy1, sx0:sx1, None]
        region = self.rgb[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
        region[:] = a * patch[sy0:sy1, sx0:sx1] + (1 - a) * region
        if ident is not None:
            ids = self.ids[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
            ids[a[..., 0] > 0.5] = ident


def _card_image(inst: Instance, rows: Sequence[dict], load: Callable[[dict], Image.Image], ppm: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w_mm, h_mm = (CARD_H_MM, CARD_W_MM) if inst.landscape else (CARD_W_MM, CARD_H_MM)
    size = (max(4, round(w_mm * ppm)), max(4, round(h_mm * ppm)))
    if inst.face_up and inst.row is not None:
        img = load(rows[inst.row]).convert("RGB")
        if (img.width > img.height) != inst.landscape:
            img = img.rotate(90, expand=True)
        img = img.resize(size, Image.BOX if img.width > size[0] else Image.BICUBIC)
        if inst.sleeve is not None:
            img = _with_sleeve(img, inst.sleeve)
    else:
        img = card_back(inst.sleeve, inst.back, size, rng)
    arr = np.asarray(img)
    if inst.face_up:
        arr = printed(arr, rng)
    if inst.foil:
        arr = foil(arr, rng)
    if inst.glare:
        arr = sleeve_streak(arr, rng) if inst.sleeve is not None and rng.random() < 0.7 else _add_glare(arr, rng)
    alpha = _rounded_alpha(size[0], size[1], 3.0 * ppm)
    rgba = Image.fromarray(arr).convert("RGBA")
    rgba.putalpha(alpha)
    rgba = rgba.rotate(inst.angle, resample=Image.BICUBIC, expand=True)
    a = np.asarray(rgba, np.float32)
    return a[..., :3], a[..., 3] / 255


def _occluder_image(o: Occluder, ppm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    s = max(6, round(o.size * ppm))
    if o.kind == "hand":
        # fingertips at the top centre of the patch; palm and forearm below, towards the player
        W, H = s, int(s * 3.2)
        img = Image.new("RGB", (W, H), o.colour); a = Image.new("L", (W, H), 0); d = ImageDraw.Draw(a)
        d.rounded_rectangle([W * 0.14, H * 0.12, W * 0.86, H * 0.36], radius=W * 0.22, fill=255)   # palm
        for fx, top in [(0.24, 0.05), (0.42, 0.01), (0.6, 0.02), (0.77, 0.07)]:                  # fingers, touching
            d.rounded_rectangle([W * (fx - 0.095), H * top, W * (fx + 0.095), H * 0.22], radius=W * 0.09, fill=255)
        d.rounded_rectangle([W * 0.0, H * 0.2, W * 0.24, H * 0.3], radius=W * 0.1, fill=255)      # thumb
        d.rounded_rectangle([W * 0.24, H * 0.3, W * 0.78, H * 1.0], radius=W * 0.2, fill=255)     # forearm
        shade = np.asarray(img, np.float32) * (0.85 + 0.25 * np.linspace(1, 0, W)[None, :, None])
        shade += rng.normal(0, 4, size=shade.shape)
        rgba = Image.fromarray(np.clip(shade, 0, 255).astype(np.uint8)).convert("RGBA")
        rgba.putalpha(a.filter(ImageFilter.GaussianBlur(max(1, s * 0.01))))
        rgba = rgba.rotate(o.angle, resample=Image.BICUBIC, expand=True)
    else:
        W, H = (s, s) if o.kind in ("die", "counter") else (s, max(4, s // 3))
        a = _rounded_alpha(W, H, min(W, H) * (0.18 if o.kind == "die" else 0.25))
        img = Image.new("RGB", (W, H), o.colour); d = ImageDraw.Draw(img)
        ink = (20, 20, 20) if sum(o.colour) > 380 else (235, 235, 235)
        if o.kind == "die":
            side = max(1, round(W * 0.14))  # the visible sides of the cube
            face = Image.new("RGB", (W, H), tuple(int(c * 0.62) for c in o.colour))
            face.paste(img.resize((W - side, H - side)), (0, 0))
            img = face; d = ImageDraw.Draw(img)
            W0 = W - side
            pip = W0 * 0.09
            spots = {1: [(0.5, 0.5)], 2: [(0.28, 0.28), (0.72, 0.72)], 3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
                     4: [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)],
                     5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
                     6: [(0.28, 0.22), (0.72, 0.22), (0.28, 0.5), (0.72, 0.5), (0.28, 0.78), (0.72, 0.78)]}[o.value]
            for fx, fy in spots:
                d.ellipse([W0 * fx - pip, W0 * fy - pip, W0 * fx + pip, W0 * fy + pip], fill=ink)
            shade = np.asarray(img, np.float32) * (0.8 + 0.2 * np.linspace(1, 0, H)[:, None, None])
            img = Image.fromarray(np.clip(shade, 0, 255).astype(np.uint8))
        else:
            text = f"+{o.value}" if o.kind == "counter" else ["EMPOWERED", "HIDDEN", "STUNNED", "BUFF"][o.value % 4]
            font = ImageFont.load_default(size=max(6, int(H * (0.6 if o.kind == "counter" else 0.55))))
            box = d.textbbox((0, 0), text, font=font)
            d.text(((W - (box[2] - box[0])) / 2, (H - (box[3] - box[1])) / 2 - box[1]), text, fill=ink, font=font)
        rgba = img.convert("RGBA"); rgba.putalpha(a)
        rgba = rgba.rotate(o.angle, resample=Image.BICUBIC, expand=True)
    arr = np.asarray(rgba, np.float32)
    return arr[..., :3], arr[..., 3] / 255


def render_table(board: Board, rows: Sequence[dict], load: Callable[[dict], Image.Image], region: tuple[float, float, float, float],
                 ppm: float, rng: np.random.Generator, mats: Sequence[Image.Image] = ()) -> Canvas:
    cv = Canvas(*region, ppm)
    table = np.asarray([(18, 18, 20), (40, 30, 24), (60, 60, 64), (28, 34, 48)][int(rng.integers(4))], np.float32)
    cv.rgb[:] = np.clip(table * rng.uniform(0.7, 1.4) + _smooth_noise(rng, cv.w, cv.h, max(8, cv.w // 6), 6), 0, 255)
    gap = float(rng.uniform(0, 20))
    for sgn in (1, -1):  # the two playmats meet near the centre line
        cx, cy = rng.normal(0, 6), sgn * (MAT_H_MM / 2 + gap / 2)
        x0, y0 = cv.px((cx - MAT_W_MM / 2, cy - MAT_H_MM / 2))
        x1, y1 = cv.px((cx + MAT_W_MM / 2, cy + MAT_H_MM / 2))
        mw, mh = int(round(x1 - x0)), int(round(y1 - y0))
        if mw > 2 and mh > 2:
            m = mat_image(rng, mw, mh, mats)
            cv.paste(m, np.asarray(_rounded_alpha(mw, mh, 8 * ppm), np.float32) / 255, (x0 + x1) / 2, (y0 + y1) / 2, None)
    for inst in board.instances:
        patch, alpha = _card_image(inst, rows, load, ppm, rng)
        cv.paste(patch, alpha, *cv.px(inst.centre), inst.id + 1, shadow=0.35, rng=rng)
    for o in board.occluders:
        patch, alpha = _occluder_image(o, ppm, rng)
        if o.kind == "hand":
            # the patch was turned about the fingertips: place the fingertips at `centre`
            tx, ty = cv.px(o.centre)
            ph, pw = alpha.shape
            t = np.deg2rad(o.angle)
            H = o.size * ppm * 3.2
            # the patch centre sits half a patch behind the fingertips, along the hand
            cx, cy = tx + np.sin(t) * H / 2, ty + np.cos(t) * H / 2
            cv.paste(patch, alpha, cx, cy, OCCLUDER_ID, shadow=0.3, rng=rng)
        else:
            cv.paste(patch, alpha, *cv.px(o.centre), OCCLUDER_ID, shadow=0.35, rng=rng)
    return cv


# ------------------------------------------------------------------------------------------
# Camera, broadcast and annotations
# ------------------------------------------------------------------------------------------

def _broadcast_background(shot: Shot, rng: np.random.Generator) -> np.ndarray:
    """Side panels or a player camera: gradients, a face-like blob, bars. Nothing from a real broadcast."""
    W, H = shot.frame_w, shot.frame_h
    bg = rng.uniform(10, 60, size=3) + _smooth_noise(rng, W, H, max(8, W // 8), 18)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 6))):  # boxes: player cams, name bars, score chips
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        bw, bh = rng.uniform(0.05, 0.25) * W, rng.uniform(0.02, 0.25) * H
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
        if bh > 0.1 * H:  # a face in a player cam
            fx, fy, fr = x0 + bw / 2, y0 + bh * 0.45, min(bw, bh) * 0.22
            d.ellipse([fx - fr, fy - fr * 1.2, fx + fr, fy + fr * 1.2], fill=(int(rng.integers(90, 235)), int(rng.integers(60, 190)), int(rng.integers(40, 160))))
    return np.asarray(img, np.float32)


def render(board: Board, rows: Sequence[dict], load: Callable[[dict], Image.Image], shot: Shot, rng: np.random.Generator,
           mats: Sequence[Image.Image] = ()) -> tuple[np.ndarray, np.ndarray, dict]:
    """(frame RGB uint8, id map uint16, annotation) for one board, before the codec pass."""
    wx0, wy0, wx1, wy1 = shot.window
    ww, wh = wx1 - wx0, wy1 - wy0
    cam = Camera(shot, ww, wh)
    # Render the table plane over what the window sees, at twice the view's scale (at most 4 px/mm).
    grid = np.array([[x, y] for x in np.linspace(0, ww, 9) for y in np.linspace(0, wh, 9)])
    seen = cam.to_table(grid)
    pad = 20.0  # the table surface continues as far as the camera sees
    region = (seen[:, 0].min() - pad, seen[:, 1].min() - pad, seen[:, 0].max() + pad, seen[:, 1].max() + pad)
    ppm = float(min(4.0, 2.0 * shot.card_px / CARD_H_MM))
    cv = render_table(board, rows, load, region, ppm, rng, mats)

    # One mesh warp, window <- table plane, for colour and ids alike.
    nx, ny = 24, 14
    xs, ys = np.linspace(0, ww, nx + 1), np.linspace(0, wh, ny + 1)
    pts = np.array([[x, y] for y in ys for x in xs])
    src = cam.to_table(pts)
    src = np.c_[(src[:, 0] - cv.x0) * cv.ppm, (src[:, 1] - cv.y0) * cv.ppm].reshape(ny + 1, nx + 1, 2)
    mesh = []
    for j in range(ny):
        for i in range(nx):
            box = (int(round(xs[i])), int(round(ys[j])), int(round(xs[i + 1])), int(round(ys[j + 1])))
            ul, ll, lr, ur = src[j, i], src[j + 1, i], src[j + 1, i + 1], src[j, i + 1]
            mesh.append((box, (*ul, *ll, *lr, *ur)))
    table_img = Image.fromarray(np.clip(cv.rgb, 0, 255).astype(np.uint8))
    win = np.asarray(table_img.transform((ww, wh), Image.MESH, mesh, Image.BICUBIC), np.float32)
    ids = np.asarray(Image.fromarray(cv.ids).transform((ww, wh), Image.MESH, mesh, Image.NEAREST), np.int32)

    look = StreamSettings(wb_jitter=shot.wb_jitter, exposure_jitter=shot.exposure_jitter, gamma_jitter=shot.gamma_jitter,
                          light_jitter=shot.light_jitter)
    win8 = _camera_photometry(win.astype(np.uint8), look, rng)
    win8 = camera_tone(win8, shot.saturation, shot.black_lift, shot.knee)
    win8 = np.asarray(Image.fromarray(win8).filter(ImageFilter.GaussianBlur(shot.defocus_px)))

    frame = _broadcast_background(shot, rng) if shot.layout != "full" else np.zeros((shot.frame_h, shot.frame_w, 3), np.float32)
    frame = frame.astype(np.uint8)
    frame[wy0:wy1, wx0:wx1] = win8
    idmap = np.zeros((shot.frame_h, shot.frame_w), np.uint16)
    idmap[wy0:wy1, wx0:wx1] = np.clip(ids, 0, 65535).astype(np.uint16)

    graphics = []
    if shot.layout == "rq" and shot.featured:  # the featured-card graphic in the right panel
        face_up = [i for i in board.instances if i.face_up and i.row is not None and not i.landscape]
        if face_up:
            inst = face_up[int(rng.integers(len(face_up)))]
            gw = round(0.16 * shot.frame_w); gh = round(gw * 88 / 63)
            gx0, gy0 = shot.frame_w - gw - round(0.015 * shot.frame_w), shot.frame_h - gh - round(0.02 * shot.frame_h)
            art = load(rows[inst.row]).convert("RGB").resize((gw, gh), Image.BICUBIC)
            frame[gy0:gy0 + gh, gx0:gx0 + gw] = np.asarray(art)
            idmap[gy0:gy0 + gh, gx0:gx0 + gw] = GRAPHIC_ID
            graphics.append({"kind": "featured_card", "printing_id": rows[inst.row]["printing_id"], "box": [gx0, gy0, gx0 + gw, gy0 + gh]})

    cards = []
    counts = np.bincount(idmap.ravel(), minlength=len(board.instances) + 2)
    off = np.array([wx0, wy0], np.float64)
    for inst in board.instances:
        q = cam.to_window(card_quad_mm(inst)) + off
        area = polygon_area(q)
        vis = int(counts[inst.id + 1]) if inst.id + 1 < len(counts) else 0
        inside = (q[:, 0] >= wx0) & (q[:, 0] <= wx1) & (q[:, 1] >= wy0) & (q[:, 1] <= wy1)
        if vis == 0 and not inside.any():
            continue  # out of view
        ys_, xs_ = np.nonzero(idmap == inst.id + 1) if vis else (np.array([]), np.array([]))
        rec = {"id": inst.id, "kind": "card" if inst.face_up else "card_back", "zone": inst.zone, "controller": inst.controller,
               "exhausted": inst.exhausted, "landscape": inst.landscape,
               "quad": [round(float(v), 2) for v in q.ravel()], "visible": round(min(1.0, vis / max(area, 1.0)), 4),
               "visible_box": [int(xs_.min()), int(ys_.min()), int(xs_.max()) + 1, int(ys_.max()) + 1] if vis else None,
               "truncated": bool(not inside.all()), "pile": inst.pile, "pile_index": inst.pile_index,
               "foil": inst.foil, "sleeved": inst.sleeve is not None}
        if inst.face_up and inst.row is not None:  # face-down cards carry no identity, ever
            rec["printing_id"] = rows[inst.row]["printing_id"]
            rec["card_id"] = rows[inst.row].get("card_id", "")
        cards.append(rec)
    occluders = []
    for o in board.occluders:
        c = cam.to_window(np.array([o.centre])) + off
        occluders.append({"kind": o.kind, "centre": [round(float(v), 1) for v in c[0]], "on": o.on})
    ann = {"width": shot.frame_w, "height": shot.frame_h, "layout": shot.layout, "window": [wx0, wy0, wx1, wy1],
           "card_px": round(shot.card_px, 1), "camera": {"turn": shot.turn, "yaw_deg": round(shot.yaw_deg, 2),
                                                          "keystone": round(shot.keystone, 3), "k1": round(shot.k1, 3)},
           "turn": board.turn, "piles": {str(k): v for k, v in board.piles.items()},
           "cards": cards, "occluders": occluders, "graphics": graphics}
    return frame, idmap, ann
