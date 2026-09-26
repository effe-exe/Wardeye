# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
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

Frames are streamed through ffmpeg rather than held in memory, so a 1,200-card
catalogue at 1080p runs in minutes on a laptop.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from PIL import Image, ImageFilter

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


def card_on_screen(card: Image.Image, s: StreamSettings, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Resize a clean card to its on-screen size, apply sleeve/glare/lens blur and rotation."""
    landscape = card.width > card.height
    long_side = s.card_h
    short_side = max(2, round(long_side * CARD_ASPECT))
    size = (long_side, short_side) if landscape else (short_side, long_side)
    img = card.convert("RGB")
    if rng.random() < s.sleeve_prob:
        img = _add_sleeve(img, rng)
    img = img.resize(size, Image.BOX)  # optics average light over each pixel
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7))))
    arr = np.asarray(img)
    if rng.random() < s.glare_prob:
        arr = _add_glare(arr, rng)
    rotation = int(rng.choice([0, 90, 180, 270])) if s.rotate else 0
    if rotation:
        arr = np.asarray(Image.fromarray(arr).rotate(rotation, expand=True))
    return arr, rotation


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
            frame[y0 : y0 + ch, x0 : x0 + cw] = arr
            placements.append(Placement(idx, (x0, y0, x0 + cw, y0 + ch), rotation))
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
             "-c:v", "libx264", "-preset", s.preset, "-bf", "2",
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
    crops: list[DegradedCrop] = []
    for fi, b in sorted(last.items()):
        frame = decoded[fi]
        for p in boards[b][1]:
            x0, y0, x1, y1 = p.box
            crops.append(DegradedCrop(p.card_index, Image.fromarray(frame[y0:y1, x0:x1]), p.rotation, s))
    crops.sort(key=lambda c: c.card_index)
    return crops


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
