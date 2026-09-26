# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Layer 1: a cheap change gate that says when and where the table changed.

The detector and embedder are the expensive stages. Most of the time nothing on the table
changes, and when something does, it changes in one place. This gate watches a small,
downscaled view of the table and fires only when a region differs from the still table
*and has settled* (no motion for `settle_s`). Hands passing over the table move, so they
are ignored until they leave. Each event carries a box and a guess at its kind:

* `appeared`: the region was bare playmat and now is not (a card was played or moved here);
* `disappeared`: the region was covered and is now bare playmat (a card left or moved away);
* `changed`: covered before and after (a card turned, was replaced, or got a counter).

The event engine pairs `disappeared` + `appeared` with the same art into `card_moved`, and
the pipeline behind the gate runs detection and identification on the event's box only.

    python -m rifteye_ml.changegate --video seg.mp4 --start 60 --duration 600 --table 0.15,0.10,0.88,0.884 \\
        --out events.json --sheets events/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image


@dataclass
class GateSettings:
    fps: float = 5.0
    width: int = 320                 # the table view is downscaled to this width
    diff: float = 28.0               # per-pixel colour distance that counts as a change
    motion: float = 18.0             # per-pixel distance between consecutive frames that counts as motion
    settle_s: float = 0.6            # a changed region must be still this long
    min_area: float = 0.35           # of one card's area at this scale
    card_long_frac: float = 131 / 1080  # a card's long side as a fraction of the frame height
    frame_h: int = 1080
    global_cut: float = 0.5          # more than this share of the table changed at once: a camera cut
    adapt: float = 0.05              # still, unchanged pixels drift toward the frame (slow light changes)
    min_mat: float = 0.25            # below this share of bare playmat the frame is not the table view (a cutaway)
    hand_share: float = 0.05         # a region this much covered by skin (dilated) waits for the hand to leave
    strong: float = 0.3              # share of a region that must differ by 2× `diff`: cards do, light and codec do not
    ignore: tuple[tuple[float, float, float, float], ...] = ()  # boxes (fractions of the table view) never watched
    mat_rgb: tuple[int, int, int] | None = None  # bare playmat colour; estimated from the first frame if None


@dataclass
class ChangeEvent:
    t: float                          # seconds from the start of the window
    box: tuple[int, int, int, int]    # x0, y0, x1, y1 in the gate's downscaled table view
    kind: str                         # appeared | disappeared | changed | cut
    area: int
    before_mat: float                 # share of the box that was bare mat before
    after_mat: float
    extra: dict = field(default_factory=dict)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2).astype(np.float32) / 3


def skin(frame: np.ndarray) -> np.ndarray:
    """Skin-coloured pixels (YCrCb box). On a red mat skin separates cleanly: the mat's Cr is ~200."""
    f = frame.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    cr = 128 + 0.713 * (f[..., 0] - y)
    cb = 128 + 0.564 * (f[..., 2] - y)
    return (cr > 135) & (cr < 180) & (cb > 80) & (cb < 130) & (y > 60)


class ChangeGate:
    """Feed frames of the table view (H × W × 3 uint8) in order; collect `events`."""

    def __init__(self, s: GateSettings):
        from scipy import ndimage  # dev-time dependency, like the rest of the M0 tooling

        self.nd = ndimage
        self.s = s
        self.background: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.still = None                # frames each pixel has been still
        self.events: list[ChangeEvent] = []
        self.snapshots: list[tuple[np.ndarray, np.ndarray]] = []  # (still table before, frame after) per event
        self.mat: np.ndarray | None = None
        self.off_table = 0               # consecutive frames that were not the table view

    def _is_mat(self, img: np.ndarray) -> np.ndarray:
        assert self.mat is not None
        return np.abs(img.astype(np.int16) - self.mat.astype(np.int16)).max(axis=2) < 45

    def feed(self, t: float, frame: np.ndarray) -> list[ChangeEvent]:
        s = self.s
        if self.mat is None:
            self.mat = np.array(s.mat_rgb, np.uint8) if s.mat_rgb else np.median(frame.reshape(-1, 3), axis=0).astype(np.uint8)
        if self._is_mat(frame).mean() < s.min_mat:  # a cutaway (player cam, graphic): wait for the table
            self.off_table += 1
            return []
        if self.background is None:
            self.background, self.prev = frame.copy(), frame.copy()
            self.still = np.zeros(frame.shape[:2], np.int32)
            return []
        if self.off_table:  # back on the table: the old still table stays, so changes made meanwhile are found
            self.off_table = 0
            self.prev = frame.copy()
            self.still[:] = 0
            return []
        moving = _dist(frame, self.prev) > s.motion
        self.still = np.where(moving, 0, self.still + 1)
        self.prev = frame
        dist = _dist(frame, self.background)
        changed = dist > s.diff
        h, w = changed.shape
        for fx0, fy0, fx1, fy1 in s.ignore:
            changed[round(fy0 * h):round(fy1 * h), round(fx0 * w):round(fx1 * w)] = False
        hand = self.nd.binary_dilation(skin(frame), iterations=4)
        new: list[ChangeEvent] = []
        if changed.mean() > s.global_cut:  # camera cut or layout change: reset, report once
            new.append(ChangeEvent(t, (0, 0, frame.shape[1], frame.shape[0]), "cut", int(changed.sum()), 0.0, 0.0))
            self.snapshots.append((self.background.copy(), frame.copy()))
            self.background = frame.copy()
            self.still[:] = 0
            self.events += new
            return new
        settle = max(1, round(s.settle_s * s.fps))
        settled = changed & (self.still >= settle)
        settled = self.nd.binary_opening(settled, structure=np.ones((3, 3)))
        labels, n = self.nd.label(settled)
        scale = s.width / 1920
        card_long = s.card_long_frac * s.frame_h * scale
        min_area = s.min_area * card_long * card_long / 1.4
        for i, sl in enumerate(self.nd.find_objects(labels), 1):
            if sl is None:
                continue
            region = labels[sl] == i
            area = int(region.sum())
            # Only act once the whole blob is still: part of it still moving means a hand is there.
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            pad = 2
            ys, xs = slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad)
            if (changed[ys, xs] & (self.still[ys, xs] < settle)).any():
                continue
            # A resting hand is still but is not the table: wait until it leaves, and never absorb it.
            # Arms enter from the sides, so a region touching the side edges waits too.
            if hand[sl][region].mean() > s.hand_share or x0 == 0 or x1 >= w:
                continue
            before = self._is_mat(self.background[sl])[region].mean()
            after = self._is_mat(frame[sl])[region].mean()
            weak = (dist[sl][region] > 2 * s.diff).mean() < s.strong
            kind = ("noise" if weak or (before > 0.6 and after > 0.6) else
                    "appeared" if before > 0.6 and after < 0.4 else
                    "disappeared" if before < 0.4 and after > 0.6 else "changed")
            if area >= min_area and kind != "noise":  # bare mat before and after: light or codec, absorb it
                new.append(ChangeEvent(t, (int(x0), int(y0), int(x1), int(y1)), kind, area, float(before), float(after)))
                cy, cx = slice(max(0, y0 - 12), y1 + 12), slice(max(0, x0 - 12), x1 + 12)
                self.snapshots.append((self.background[cy, cx].copy(), frame[cy, cx].copy()))
            self.background[sl][region] = frame[sl][region]  # absorb the change (small ones silently)
        # Slow light changes: still, unchanged pixels drift toward the current frame.
        calm = (self.still >= settle) & ~changed & ~hand
        bg = self.background.astype(np.float32)
        bg[calm] += s.adapt * (frame[calm].astype(np.float32) - bg[calm])
        self.background = np.clip(bg, 0, 255).astype(np.uint8)
        self.events += new
        return new


def video_frames(video: str, start: float, duration: float, table: Sequence[float], s: GateSettings,
                 ffmpeg: str | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Decode a window of `video` at `s.fps`, cropped to the table and scaled to `s.width`."""
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    x0, y0, x1, y1 = table
    w = s.width
    probe_h = round(w * (y1 - y0) * 1080 / ((x1 - x0) * 1920) / 2) * 2
    vf = f"fps={s.fps},crop=iw*{x1 - x0}:ih*{y1 - y0}:iw*{x0}:ih*{y0},scale={w}:{probe_h}"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(duration), "-i", video,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    size = w * probe_h * 3
    i = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield i / s.fps, np.frombuffer(buf, np.uint8).reshape(probe_h, w, 3)
        i += 1
    proc.wait()


def run(frames: Iterable[tuple[float, np.ndarray]], s: GateSettings) -> ChangeGate:
    gate = ChangeGate(s)
    for t, f in frames:
        gate.feed(t, f)
    return gate


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.changegate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--table", default="0,0,1,1")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--mat", help="bare playmat colour R,G,B (estimated from the first frame if omitted)")
    ap.add_argument("--ignore", action="append", default=[],
                    help="repeatable x0,y0,x1,y1 box in fractions of the *frame* to never watch (overlays)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", help="folder for before/after images of each event (private)")
    a = ap.parse_args(argv)
    table = [float(v) for v in a.table.split(",")]
    tx0, ty0, tx1, ty1 = table
    ignore = []
    for box in a.ignore:  # frame fractions -> table-view fractions
        bx0, by0, bx1, by1 = (float(v) for v in box.split(","))
        ignore.append(((bx0 - tx0) / (tx1 - tx0), (by0 - ty0) / (ty1 - ty0), (bx1 - tx0) / (tx1 - tx0), (by1 - ty0) / (ty1 - ty0)))
    s = GateSettings(fps=a.fps, mat_rgb=tuple(int(v) for v in a.mat.split(",")) if a.mat else None,
                     ignore=tuple(tuple(min(1.0, max(0.0, v)) for v in b) for b in ignore))
    gate = run(video_frames(a.video, a.start, a.duration, table, s), s)
    events = gate.events
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"settings": asdict(s), "start": a.start, "table": table,
                                       "events": [asdict(e) for e in events]}, indent=1), encoding="utf-8")
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"{len(events)} events in {a.duration:.0f} s: {kinds} -> {a.out}")
    if a.sheets:
        out = Path(a.sheets)
        out.mkdir(parents=True, exist_ok=True)
        for k, (e, (b, f)) in enumerate(zip(events, gate.snapshots)):
            if e.kind == "cut":
                continue
            before, after = Image.fromarray(b), Image.fromarray(f)
            pair = Image.new("RGB", (before.width * 2 + 6, before.height), (0, 0, 0))
            pair.paste(before, (0, 0))
            pair.paste(after, (before.width + 6, 0))
            pair.save(out / f"{k:03d}_{a.start + e.t:08.1f}s_{e.kind}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
