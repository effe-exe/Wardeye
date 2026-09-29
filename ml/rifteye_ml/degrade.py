# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""What a broadcast does to a card: the stream-domain degradation behind the M0 spike.

This mirrors docs/research/04 §4.3.

1. Place cards on a table at a target on-screen height, at random 90° rotations
   (opponent side, exhausted), some in sleeves, some with glare.
2. Hold each board for a few frames, adding sensor noise, the way a static camera
   feeds an encoder.
3. Push every frame through a **real H.264 encode** (libx264, 4:2:0, VBV-limited
   bitrate, 2 s GOP) and decode it back. JPEG noise is the wrong noise: blocking,
   ringing, chroma bleed and inter-frame smear only come from a video codec.
4. Crop each card from the last decoded frame of its board, where the encoder has
   reached its steady state.

Two realism levels (`REALISM`):

* **codec**: steps 1–4 only, with perfect crops. It measures how much of a card
  survives downscaling and the codec, an upper bound.
* **camera**: adds what a real table camera and detector do. Tilt (foreshortening and
  a little keystone), residual rotation, white balance, exposure and uneven stage light,
  defocus, occluders (fingers, counters, overlapping cards) and detector box error. The
  magnitudes are assumptions until real footage calibrates them.

Frames are streamed through ffmpeg rather than held in memory, so a 1,200-card
catalogue at 1080p runs in minutes on a laptop.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CARD_ASPECT = 63 / 88  # width / height of a portrait card


@dataclass(frozen=True)
class StreamSettings:
    frame_w: int = 1920
    frame_h: int = 1080
    card_h: int = 100               # the card's long side on screen, in px
    bitrate_kbps: int = 6000
    fps: int = 30
    frames_per_board: int = 12      # the encoder settles; the last frame is cropped
    max_cards_per_frame: int = 36   # a real table, not a contact sheet: bits per card stay realistic
    noise_sigma: float = 2.0        # sensor noise keeps P-frames spending bits
    rotate: bool = True             # random 0/90/180/270 like a real table
    sleeve_prob: float = 0.7
    glare_prob: float = 0.2
    preset: str = "veryfast"        # what most streamers run in OBS
    seed: int = 0
    # Camera and detector realism. All zero is the "codec" level.
    tilt_deg: float = 0.0           # camera tilt: foreshortening along the frame's vertical axis
    keystone: float = 0.0           # far edge shorter by up to this fraction
    angle_jitter_deg: float = 0.0   # residual rotation a rectifier leaves behind
    wb_jitter: float = 0.0          # per-channel white-balance gain drift, per board
    exposure_jitter: float = 0.0    # log exposure drift, per board
    gamma_jitter: float = 0.0       # log gamma drift, per board
    light_jitter: float = 0.0       # uneven stage light: a smooth gain field across the frame
    defocus_px: float = 0.0         # extra blur radius at the card's on-screen scale
    occlusion_prob: float = 0.0     # a finger, counter or overlapping card covers part of the card
    box_jitter: float = 0.0         # detector error: crop offset and scale, as a fraction of the box
    foil_prob: float = 0.0          # holographic foil under stage light: a rainbow sheen that shifts colours


REALISM: dict[str, dict[str, float]] = {
    "codec": {},
    "camera": {"tilt_deg": 15.0, "keystone": 0.03, "angle_jitter_deg": 4.0, "wb_jitter": 0.08,
               "exposure_jitter": 0.25, "gamma_jitter": 0.15, "light_jitter": 0.15, "defocus_px": 0.6,
               "occlusion_prob": 0.3, "box_jitter": 0.06},
}
# Foil printings were the colour grid's main misses on real footage (M0 report §5.2). A separate
# level, so runs at the other levels stay bit-exact.
REALISM["foil"] = {**REALISM["camera"], "foil_prob": 0.2}


def with_realism(s: StreamSettings, level: str) -> StreamSettings:
    if level not in REALISM:
        raise ValueError(f"unknown realism level {level!r}; choose from {', '.join(REALISM)}")
    return replace(s, **REALISM[level])


@dataclass
class Placement:
    card_index: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the frame
    rotation: int                   # degrees counter-clockwise applied to the upright card


@dataclass
class DegradedCrop:
    card_index: int
    image: Image.Image              # as it appears on stream (rotated by `rotation`)
    rotation: int
    settings: StreamSettings = field(repr=False)

    def upright(self) -> Image.Image:
        """Undo the known rotation: an oracle rectifier, for isolating the codec's effect."""
        return self.image.rotate(-self.rotation, expand=True) if self.rotation else self.image


def ffmpeg_exe() -> str:
    import imageio_ffmpeg  # dev-time dependency; the binary is never shipped

    return imageio_ffmpeg.get_ffmpeg_exe()


# ------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------

def table_texture(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible playmat: low-frequency colour blotches on a dark base, plus fine grain."""
    base = rng.uniform(25, 70, size=3)
    lo = rng.normal(0, 18, size=(max(2, h // 120), max(2, w // 120), 3))
    blot = np.asarray(Image.fromarray(np.clip(lo + 128, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) - 128
    grain = rng.normal(0, 3, size=(h, w, 1))
    return np.clip(base + blot + grain, 0, 255).astype(np.uint8)


def _add_sleeve(card: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A sleeve is ~2 mm of coloured border around a 63×88 mm card."""
    pad = max(1, round(card.height * 2 / 88))
    colour = tuple(int(c) for c in rng.integers(0, 90, 3))
    sleeved = Image.new("RGB", (card.width + 2 * pad, card.height + 2 * pad), colour)
    sleeved.paste(card, (pad, pad))
    return sleeved.resize(card.size, Image.BOX)


def _add_glare(card: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = card.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    rad = rng.uniform(0.25, 0.6) * max(w, h)
    ys, xs = np.ogrid[:h, :w]
    mask = np.clip(1.0 - np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / rad, 0, 1) ** 2 * rng.uniform(0.15, 0.45)
    a = card.astype(np.float32)
    return np.clip(a + (255 - a) * mask[..., None], 0, 255).astype(np.uint8)


def _rainbow(hue: np.ndarray) -> np.ndarray:
    """Fully saturated RGB in [0, 1] for hues in [0, 1)."""
    h6 = hue[..., None] * 6.0
    return np.clip(np.concatenate([np.abs(h6 - 3) - 1, 2 - np.abs(h6 - 2), 2 - np.abs(h6 - 4)], axis=-1), 0, 1)


def _add_foil(card: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A holographic foil lit from above: a rainbow whose hue drifts across the card along a random
    direction, strongest in a bright band where the light catches it."""
    h, w = card.shape[:2]
    theta = rng.uniform(0, np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs * np.cos(theta) + ys * np.sin(theta)) / max(w, h)
    sheen = _rainbow((u * rng.uniform(0.8, 2.5) + rng.uniform(0, 1)) % 1.0)
    band = np.exp(-(((u - rng.uniform(-0.2, 1.2)) / rng.uniform(0.1, 0.35)) ** 2))
    a = (rng.uniform(0.12, 0.3) * (0.5 + band))[..., None]
    out = card.astype(np.float32) * (1 - a) + sheen * 255 * a + 40 * band[..., None] * a
    return np.clip(out, 0, 255).astype(np.uint8)


SKIN = [(236, 188, 160), (224, 172, 138), (198, 140, 106), (160, 110, 80), (112, 76, 56), (74, 52, 40)]


def _add_occluder(card: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Cover part of the card with a finger, a counter or die, or the edge of another card."""
    img = card.copy()
    w, h = img.size
    d = ImageDraw.Draw(img)
    kind = int(rng.integers(3))
    if kind == 0:  # a finger or hand entering from one edge
        colour = SKIN[int(rng.integers(len(SKIN)))]
        depth = rng.uniform(0.12, 0.35)
        fw = rng.uniform(0.18, 0.4) * w
        cx = rng.uniform(0.2, 0.8) * w
        if rng.random() < 0.5:
            d.ellipse([cx - fw / 2, -h * 0.3, cx + fw / 2, depth * h], fill=colour)
        else:
            d.ellipse([cx - fw / 2, h * (1 - depth), cx + fw / 2, h * 1.3], fill=colour)
    elif kind == 1:  # a counter, token or die on the card
        size = rng.uniform(0.18, 0.32) * w
        x, y = rng.uniform(0, w - size), rng.uniform(0, h - size)
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        d.rounded_rectangle([x, y, x + size, y + size], radius=size * 0.2, fill=colour)
        pip = size * 0.18
        d.ellipse([x + size / 2 - pip, y + size / 2 - pip, x + size / 2 + pip, y + size / 2 + pip],
                  fill=tuple(255 - c for c in colour))
    else:  # another card overlapping a long end (stacked units, attached gear)
        depth = int(rng.uniform(0.1, 0.3) * h)
        lo = rng.normal(0, 40, size=(4, 3, 3)) + rng.uniform(40, 200, size=3)
        patch = Image.fromarray(np.clip(lo, 0, 255).astype(np.uint8)).resize((w, depth), Image.BICUBIC)
        img.paste(patch, (0, 0 if rng.random() < 0.5 else h - depth))
    return img


def _perspective_coeffs(dst: np.ndarray, src: np.ndarray) -> list[float]:
    """PIL PERSPECTIVE coefficients mapping output points `dst` to input points `src`."""
    a, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64)).tolist()


def _camera_geometry(arr: np.ndarray, s: StreamSettings, rng: np.random.Generator) -> np.ndarray:
    """Tilt, keystone and residual rotation of a placed card, as an RGBA patch."""
    h, w = arr.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    c = src.mean(axis=0)
    t = np.deg2rad(rng.uniform(-s.angle_jitter_deg, s.angle_jitter_deg))
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = (src - c) @ rot.T
    pts[:, 1] *= np.cos(np.deg2rad(rng.uniform(0, s.tilt_deg)))  # foreshortening
    k = rng.uniform(0, s.keystone)
    far = pts[:, 1] < 0 if rng.random() < 0.5 else pts[:, 1] > 0
    pts[far, 0] *= 1 - k
    pts -= pts.min(axis=0)
    ow, oh = int(np.ceil(pts[:, 0].max())) + 1, int(np.ceil(pts[:, 1].max())) + 1
    rgba = Image.fromarray(arr).convert("RGBA")
    out = rgba.transform((ow, oh), Image.PERSPECTIVE, _perspective_coeffs(pts, src), Image.BICUBIC)
    return np.asarray(out)


def card_on_screen(card: Image.Image, s: StreamSettings, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Resize a clean card to its on-screen size, apply sleeve/glare/lens blur and rotation.

    Returns RGB, or RGBA when camera geometry leaves table showing around the card."""
    landscape = card.width > card.height
    long_side = s.card_h
    short_side = max(2, round(long_side * CARD_ASPECT))
    size = (long_side, short_side) if landscape else (short_side, long_side)
    img = card.convert("RGB")
    if rng.random() < s.sleeve_prob:
        img = _add_sleeve(img, rng)
    if s.occlusion_prob > 0 and rng.random() < s.occlusion_prob:
        img = _add_occluder(img, rng)
    img = img.resize(size, Image.BOX)  # optics average light over each pixel
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7 + s.defocus_px))))
    arr = np.asarray(img)
    if s.foil_prob > 0 and rng.random() < s.foil_prob:  # no draw at all when off: other levels stay bit-exact
        arr = _add_foil(arr, rng)
    if rng.random() < s.glare_prob:
        arr = _add_glare(arr, rng)
    rotation = int(rng.choice([0, 90, 180, 270])) if s.rotate else 0
    if rotation:
        arr = np.asarray(Image.fromarray(arr).rotate(rotation, expand=True))
    if s.tilt_deg > 0 or s.keystone > 0 or s.angle_jitter_deg > 0:
        arr = _camera_geometry(arr, s, rng)
    return arr, rotation


def _camera_photometry(frame: np.ndarray, s: StreamSettings, rng: np.random.Generator) -> np.ndarray:
    """White balance, exposure, gamma and uneven stage light: one camera per board."""
    h, w = frame.shape[:2]
    gains = 1 + rng.uniform(-s.wb_jitter, s.wb_jitter, size=3)
    gains *= np.exp(rng.uniform(-s.exposure_jitter, s.exposure_jitter))
    gamma = np.exp(rng.uniform(-s.gamma_jitter, s.gamma_jitter))
    lo = 1 + rng.uniform(-s.light_jitter, s.light_jitter, size=(3, 4)).astype(np.float32)
    light = np.asarray(Image.fromarray(lo).resize((w, h), Image.BICUBIC), np.float32)
    a = np.clip(frame.astype(np.float32) / 255 * gains.astype(np.float32) * light[..., None], 0, 1)
    return (a ** gamma * 255).astype(np.uint8)


def render_boards(cards: list[Image.Image], s: StreamSettings) -> Iterator[tuple[np.ndarray, list[Placement]]]:
    """Yield (frame, placements) boards until every card has been placed exactly once."""
    rng = np.random.default_rng(s.seed)
    cell = int(round(s.card_h * 1.25))  # room for a rotated card plus a gap
    cols, rows = max(1, s.frame_w // cell), max(1, s.frame_h // cell)
    per_frame = max(1, min(s.max_cards_per_frame, cols * rows))
    order = list(range(len(cards)))
    for start in range(0, len(order), per_frame):
        chunk = order[start : start + per_frame]
        frame = table_texture(s.frame_w, s.frame_h, rng)
        cells = rng.choice(cols * rows, size=len(chunk), replace=False)
        placements: list[Placement] = []
        for idx, c in zip(chunk, cells):
            arr, rotation = card_on_screen(cards[idx], s, rng)
            ch, cw = arr.shape[:2]
            gx, gy = int(c) % cols, int(c) // cols
            x0 = gx * cell + (cell - cw) // 2 + int(rng.integers(-cell // 10, cell // 10 + 1))
            y0 = gy * cell + (cell - ch) // 2 + int(rng.integers(-cell // 10, cell // 10 + 1))
            x0 = int(np.clip(x0, 0, s.frame_w - cw))
            y0 = int(np.clip(y0, 0, s.frame_h - ch))
            if arr.shape[2] == 4:  # table shows through around a tilted or rotated card
                alpha = arr[..., 3:].astype(np.float32) / 255
                region = frame[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
                frame[y0 : y0 + ch, x0 : x0 + cw] = (alpha * arr[..., :3] + (1 - alpha) * region).astype(np.uint8)
            else:
                frame[y0 : y0 + ch, x0 : x0 + cw] = arr
            placements.append(Placement(idx, (x0, y0, x0 + cw, y0 + ch), rotation))
        if s.wb_jitter > 0 or s.exposure_jitter > 0 or s.gamma_jitter > 0 or s.light_jitter > 0:
            frame = _camera_photometry(frame, s, rng)
        yield frame, placements


# ------------------------------------------------------------------------------------
# The codec pass
# ------------------------------------------------------------------------------------

def h264_roundtrip(frames: Iterator[np.ndarray], w: int, h: int, s: StreamSettings, keep: set[int]) -> dict[int, np.ndarray]:
    """Encode `frames` with libx264 at the stream settings, decode, and return the frames
    whose indices are in `keep`. Frames are streamed; only kept frames are held in memory."""
    exe = ffmpeg_exe()
    br = int(s.bitrate_kbps)
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mkv")
        enc = subprocess.Popen(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(s.fps), "-i", "-",
             # One thread: x264's frame-threaded rate control is not bit-exact between runs,
             # and spike results must be reproducible.
             "-c:v", "libx264", "-threads", "1", "-preset", s.preset, "-bf", "2",
             "-b:v", f"{br}k", "-maxrate", f"{br}k", "-bufsize", f"{2 * br}k",
             "-g", str(2 * s.fps), "-pix_fmt", "yuv420p", "-y", clip],
            stdin=subprocess.PIPE,
        )
        assert enc.stdin is not None
        try:
            for f in frames:
                enc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        finally:
            enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")

        dec = subprocess.Popen(
            [exe, "-hide_banner", "-loglevel", "error", "-i", clip, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        assert dec.stdout is not None
        size = w * h * 3
        out: dict[int, np.ndarray] = {}
        i = 0
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            if i in keep:
                out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            i += 1
        dec.stdout.close()
        if dec.wait() != 0:
            raise RuntimeError("ffmpeg decode failed")
    return out


def simulate(cards: list[Image.Image], s: StreamSettings) -> list[DegradedCrop]:
    """Every card, as it would look on a stream with settings `s`."""
    rng = np.random.default_rng(s.seed + 1)
    boards = list(render_boards(cards, s))
    last = {(b + 1) * s.frames_per_board - 1: b for b in range(len(boards))}
    # One bank of sensor-noise fields, reused for every board: generating fresh 1080p
    # noise per frame dominated the runtime, and the codec cannot tell the difference.
    shape = (s.frame_h, s.frame_w, 3)
    bank = [rng.standard_normal(shape, dtype=np.float32) * s.noise_sigma for _ in range(s.frames_per_board)] if s.noise_sigma > 0 else None

    def frames() -> Iterator[np.ndarray]:
        for frame, _ in boards:
            base = frame.astype(np.float32)
            for i in range(s.frames_per_board):
                yield np.clip(base + bank[i], 0, 255).astype(np.uint8) if bank is not None else frame

    decoded = h264_roundtrip(frames(), s.frame_w, s.frame_h, s, keep=set(last))
    box_rng = np.random.default_rng(s.seed + 2)
    crops: list[DegradedCrop] = []
    for fi, b in sorted(last.items()):
        frame = decoded[fi]
        for p in boards[b][1]:
            x0, y0, x1, y1 = _detector_box(p.box, s, box_rng) if s.box_jitter > 0 else p.box
            crops.append(DegradedCrop(p.card_index, Image.fromarray(frame[y0:y1, x0:x1]), p.rotation, s))
    crops.sort(key=lambda c: c.card_index)
    return crops


def _detector_box(box: tuple[int, int, int, int], s: StreamSettings, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """The true box, shifted and rescaled the way a detector's box is off."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    j = s.box_jitter
    cx = (x0 + x1) / 2 + rng.uniform(-j, j) * w
    cy = (y0 + y1) / 2 + rng.uniform(-j, j) * h
    scale = 1 + rng.uniform(-j, j)
    hw, hh = max(1.0, w * scale / 2), max(1.0, h * scale / 2)
    nx0, ny0 = int(np.clip(round(cx - hw), 0, s.frame_w - 2)), int(np.clip(round(cy - hh), 0, s.frame_h - 2))
    nx1, ny1 = int(np.clip(round(cx + hw), nx0 + 2, s.frame_w)), int(np.clip(round(cy + hh), ny0 + 2, s.frame_h))
    return nx0, ny0, nx1, ny1


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
