# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The frame source: a file, an http(s) URL, a Twitch channel or VOD, or a YouTube video or live
stream, decoded by ffmpeg into a steady sequence of RGB frames.

A daemon thread runs ffmpeg and holds one slot. In real time (`realtime=True`, the default: a
broadcast waits for no one) it always overwrites the slot with the newest decoded frame, so a slow
consumer never falls behind; `FrameSource.dropped` counts what never got delivered, and the gaps in
`Frame.index` show where. With `realtime=False` (tests, batch runs) the slot is instead a strict
one-frame handoff: the reader blocks until the previous frame is taken, so nothing is skipped.

    python -m rifteye_ml.live.source match.mp4 --fps 5 --frames 10 --out /tmp/preview
    python -m rifteye_ml.live.source https://www.twitch.tv/videos/2885620401 --start 53640 --frames 3

Twitch needs `pip install streamlink`, YouTube needs `pip install yt-dlp`; both are optional and
only imported once that kind of source is actually opened.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

_MAX_RESTARTS = 3                  # live reconnect attempts per outage
_BACKOFF = (1.0, 2.0, 4.0)         # seconds to wait before each reconnect attempt
_FRESH_AFTER = 60.0                # seconds of frames after which the next outage gets a fresh budget
_PROBE_TIMEOUT = 30.0              # ffmpeg must report the output frame size within this long
_TAIL_LINES = 40                   # stderr lines kept around, for error messages
_SIZE_RE = re.compile(r",\s*(\d+)x(\d+)\s*(?:,|\[|$)")  # WxH in ffmpeg's own "Stream ... Video:" line


@dataclass
class Frame:
    t: float            # media seconds of this frame: start + index / fps
    image: np.ndarray   # H x W x 3, uint8, RGB
    index: int           # sequence number in the decoded (fps-filtered) stream; gaps mean frames were dropped
    wall: float           # time.monotonic() when the frame was decoded (for latency)


# ------------------------------------------------------------------------------------
# Spec classification and resolution
# ------------------------------------------------------------------------------------

_TWITCH_VOD = re.compile(r"^(?:https?://)?(?:www\.)?twitch\.tv/videos?/(\d+)", re.I)
_TWITCH_LIVE = re.compile(r"^(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)/?(?:[?#].*)?$", re.I)
_YOUTUBE = re.compile(r"^(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?|live/)|youtu\.be/)", re.I)


def classify(spec: str) -> str:
    """"file" | "url" | "twitch-live" | "twitch-vod" | "youtube", from the shape of `spec` alone.
    No network: a Twitch channel that happens to be offline, or a YouTube URL that turns out to be a
    live stream rather than a VOD, still classifies the same way -- that only shows once
    `open_source` resolves it. A path that exists on disk is always "file", even if it also happens
    to look like a URL."""
    if os.path.exists(spec):
        return "file"
    s = spec.strip()
    if _TWITCH_VOD.match(s):
        return "twitch-vod"
    if _TWITCH_LIVE.match(s):
        return "twitch-live"
    if _YOUTUBE.match(s):
        return "youtube"
    if s.lower().startswith(("http://", "https://")):
        return "url"
    return "file"  # not a URL shape we know, and not on disk yet -- still the most likely reading


def _with_scheme(spec: str) -> str:
    return spec if re.match(r"^https?://", spec, re.I) else f"https://{spec}"


def _twitch_url(spec: str, quality: str) -> str:
    """A playable HLS URL for a Twitch channel or VOD, picked at `quality` (e.g. "best")."""
    try:
        import streamlink
    except ImportError as e:
        raise RuntimeError("Twitch sources need streamlink: pip install streamlink") from e
    session = streamlink.Streamlink()
    try:
        session.set_option("twitch-disable-ads", True)  # skip embedded ads, where this streamlink version supports it
    except Exception:
        pass
    streams = session.streams(_with_scheme(spec))
    if quality not in streams:
        raise RuntimeError(f"no {quality!r} stream for {spec}; available: {sorted(streams)}")
    return streams[quality].url


def _youtube_url(spec: str, height: int | None) -> tuple[str, bool]:
    """A playable URL for a YouTube video or live stream, and whether it is live."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("YouTube sources need yt-dlp: pip install yt-dlp") from e
    h = f"[height<={height}]" if height else ""
    # video only, since the audio is dropped: most VODs' combined audio+video files stop at 360p
    fmt = f"bestvideo*{h}[vcodec^=avc1]/bestvideo*{h}/best{h}"
    with yt_dlp.YoutubeDL({"format": fmt, "quiet": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(_with_scheme(spec), download=False)
    return info["url"], bool(info.get("is_live"))


def _resolve(spec: str, kind: str, quality: str, height: int | None) -> tuple[str, bool]:
    """The URL or path ffmpeg reads from, and whether the source is live (no seeking, no end)."""
    if kind in ("file", "url"):
        return spec, False
    if kind == "twitch-live":
        return _twitch_url(spec, quality), True
    if kind == "twitch-vod":
        return _twitch_url(spec, quality), False
    if kind == "youtube":
        return _youtube_url(spec, height)
    raise ValueError(f"unknown kind {kind!r}")  # pragma: no cover - classify() only returns the above


# ------------------------------------------------------------------------------------
# ffmpeg process management
# ------------------------------------------------------------------------------------

def _ffmpeg_exe() -> str:
    import imageio_ffmpeg  # dev-time dependency: the ffmpeg binary, never our own bundled build

    return imageio_ffmpeg.get_ffmpeg_exe()


class _Tail(threading.Thread):
    """Drains ffmpeg's stderr continuously, in small chunks, so its pipe never fills up and blocks
    ffmpeg (its progress line updates in place with carriage returns, not newlines, so a plain
    readline() can stall for a long time). Keeps the last few lines for error messages, and the
    output stream's WxH once ffmpeg has printed it."""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._remainder = b""
        self._lines: deque[str] = deque(maxlen=_TAIL_LINES)
        self._in_output = False
        self.size: tuple[int, int] | None = None
        self.start()

    def run(self) -> None:
        while True:
            try:
                chunk = self._stream.read(64)
            except (ValueError, OSError):
                break
            if not chunk:
                break
            parts = (self._remainder + chunk).replace(b"\r", b"\n").split(b"\n")
            self._remainder = parts.pop()
            for raw in parts:
                self._line(raw.decode("utf-8", "replace"))
        if self._remainder:
            self._line(self._remainder.decode("utf-8", "replace"))
        try:
            self._stream.close()
        except OSError:
            pass

    def _line(self, line: str) -> None:
        if not line:
            return
        self._lines.append(line)
        s = line.lstrip()
        if s.startswith("Output "):
            self._in_output = True
        elif self._in_output and self.size is None and s.startswith("Stream ") and " Video: " in s:
            m = _SIZE_RE.search(s)
            if m:
                self.size = (int(m.group(1)), int(m.group(2)))

    def text(self) -> str:
        return "\n".join(self._lines)


def _terminate(proc: subprocess.Popen, timeout: float = 2.0) -> None:
    """Stop ffmpeg: ask nicely, then kill after `timeout` seconds."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def _ffmpeg_env() -> dict[str, str] | None:
    """ffmpeg's own HTTP(S) client only honours the lowercase `http_proxy` variable, for http and
    https alike; some environments set only `https_proxy` / `HTTPS_PROXY` (the modern convention),
    which ffmpeg then never sees, and it does not fail cleanly when a direct connection is blocked
    by whatever that proxy was fronting. Fill it in for ffmpeg alone, without touching this process's
    own environment. None means "inherit os.environ unchanged", same as passing no env at all."""
    if os.environ.get("http_proxy"):
        return None
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not proxy:
        return None
    env = os.environ.copy()
    env["http_proxy"] = proxy
    return env


def _spawn(url: str, fps: float, height: int | None, start: float, realtime: bool, live: bool) -> tuple[subprocess.Popen, _Tail]:
    """Start ffmpeg decoding `url` to raw rgb24 frames on stdout, fps-filtered and scaled."""
    vf = f"fps={fps}" + (f",scale=-2:{height}" if height else "")
    args = [_ffmpeg_exe(), "-hide_banner"]
    if not live and start:
        args += ["-ss", str(start)]  # input seek: file, url, VOD (not meaningful on a live stream)
    if realtime and not live:
        args += ["-re"]              # play a recording at 1x, like a stream
    args += ["-i", url, "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=_ffmpeg_env())
    return proc, _Tail(proc.stderr)


def _probe(proc: subprocess.Popen, tail: _Tail) -> tuple[int, int]:
    """Block until ffmpeg has printed the output frame size, or raise with its stderr."""
    deadline = time.monotonic() + _PROBE_TIMEOUT
    while tail.size is None and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if tail.size is not None:
        return tail.size
    if proc.poll() is None:
        _terminate(proc)  # never printed a header: stuck (e.g. a slow or unreachable network source)
    else:
        tail.join(timeout=1.0)  # let the tail thread finish draining the last of stderr
    raise RuntimeError("ffmpeg failed before the first frame:\n" + tail.text())


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None  # EOF mid-frame counts as end of stream, same as a clean EOF between frames
        buf += chunk
    return bytes(buf)


# ------------------------------------------------------------------------------------
# FrameSource
# ------------------------------------------------------------------------------------

class FrameSource:
    """Opens `spec` and decodes it to RGB frames at `fps`. Construction (like `open_source`) blocks
    until the source's frame size is known or it fails to open. A context manager: `close()` on exit."""

    def __init__(self, spec: str, fps: float = 5.0, realtime: bool = True, start: float = 0.0,
                 height: int | None = 1080, quality: str = "best") -> None:
        self.kind = classify(spec)
        self.fps = float(fps)
        self.dropped = 0
        self._spec, self._quality, self._start, self._realtime = spec, quality, start, realtime

        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._has_frame = False
        self._done = False
        self._exc: BaseException | None = None
        self._closing = False
        self._close_event = threading.Event()
        self._next_index = 0
        self._restarts = 0
        self._proc_lock = threading.Lock()

        url, live = _resolve(spec, self.kind, quality, height)
        self.live = live
        proc, tail = _spawn(url, self.fps, height, start, realtime, live)
        try:
            self.width, self.height = _probe(proc, tail)
        except BaseException:
            _terminate(proc)
            raise
        self._proc = proc
        self._thread = threading.Thread(target=self._run, args=(proc, tail), daemon=True, name="frame-source")
        self._thread.start()

    # --- decoding (the reader thread) -------------------------------------------

    def _run(self, proc: subprocess.Popen, tail: _Tail) -> None:
        try:
            while True:
                first = self._next_index
                self._pump(proc, tail)
                if self._closing:
                    break
                code = proc.wait()
                if code == 0 and not self.live:
                    break  # a file or VOD reached its end
                if self._closing:
                    break
                if not self.live:
                    raise RuntimeError(f"ffmpeg exited ({code}) before the end of {self._spec!r}:\n{tail.text()}")
                # a live stream ended unexpectedly (network hiccup, or ffmpeg lost the edge): reconnect
                if self._next_index - first >= _FRESH_AFTER * self.fps:
                    self._restarts = 0
                nxt = self._reconnect(f"ffmpeg exit {code}:\n{tail.text()}")
                if nxt is None:
                    break  # closed while waiting to reconnect
                proc, tail = nxt
                with self._proc_lock:
                    self._proc = proc
        except BaseException as e:
            with self._cond:
                self._exc = e if isinstance(e, RuntimeError) else RuntimeError(f"{type(e).__name__}: {e}")
                self._done = True
                self._cond.notify_all()
            return
        finally:
            _terminate(proc)  # the one this thread holds, also when close() raced a reconnect
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def _reconnect(self, why: str) -> tuple[subprocess.Popen, _Tail] | None:
        """Re-resolve (a live URL can rotate or expire) and restart at the live edge, up to
        `_MAX_RESTARTS` tries with a backoff; None if the source is closed meanwhile. `index` keeps
        counting up across the gap and `t = start + index / fps` keeps moving forward at the same
        rate, so it stays continuous-ish even though real seconds of broadcast were missed; it does
        not jump to reflect how long the reconnect took. The output size is pinned to what the first
        connection reported, so every frame this source ever hands out has the same shape."""
        while self._restarts < _MAX_RESTARTS:
            self._restarts += 1
            if self._close_event.wait(_BACKOFF[min(self._restarts - 1, len(_BACKOFF) - 1)]):
                return None
            try:
                url, _ = _resolve(self._spec, self.kind, self._quality, self.height)
                proc, tail = _spawn(url, self.fps, self.height, 0.0, self._realtime, True)
                size = _probe(proc, tail)
            except Exception as e:  # noqa: BLE001 - offline for now, or the network still down: try again
                why = str(e)
                continue
            if size != (self.width, self.height):
                _terminate(proc)
                raise RuntimeError(f"the stream's frame size changed on reconnect ({size[0]}x{size[1]}, "
                                    f"was {self.width}x{self.height})")
            return proc, tail
        raise RuntimeError(f"lost the live stream and gave up after {_MAX_RESTARTS} reconnect attempts ({why})")

    def _pump(self, proc: subprocess.Popen, tail: _Tail) -> None:
        """Read raw frames from this one ffmpeg session until it ends, delivering each one."""
        frame_bytes = self.width * self.height * 3
        while True:
            buf = _read_exact(proc.stdout, frame_bytes)
            if buf is None:
                return
            index = self._next_index
            self._next_index += 1
            image = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
            self._deliver(Frame(t=self._start + index / self.fps, image=image, index=index, wall=time.monotonic()))
            if self._closing:
                return

    def _deliver(self, frame: Frame) -> None:
        with self._cond:
            if self._realtime:
                if self._has_frame:
                    self.dropped += 1  # the previous frame was never taken; it is gone now
            else:
                while self._has_frame and not self._closing:
                    self._cond.wait()  # a strict handoff: wait for the consumer, drop nothing
            self._frame = frame
            self._has_frame = True
            self._cond.notify_all()

    # --- consuming ---------------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        while True:
            with self._cond:
                while not self._has_frame and not self._done:
                    self._cond.wait()
                if not self._has_frame:  # done, and nothing left to deliver
                    if self._exc is not None:
                        raise self._exc
                    return
                frame, self._frame, self._has_frame = self._frame, None, False
                self._cond.notify_all()
            yield frame

    def close(self) -> None:
        with self._cond:
            if self._closing:
                return
            self._closing = True
            self._done = True  # wake anyone blocked in __iter__
            self._cond.notify_all()
        self._close_event.set()  # wake a reconnect backoff, if one is in progress
        with self._proc_lock:
            proc = self._proc
        _terminate(proc)
        self._thread.join(timeout=3.0)

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_source(spec: str, fps: float = 5.0, realtime: bool = True, start: float = 0.0,
                 height: int | None = 1080, quality: str = "best") -> FrameSource:
    """Open `spec` -- a file, an http(s) URL, a Twitch channel or VOD, or a YouTube video or live
    stream -- and start decoding at `fps` frames/s, scaled to `height` (keeping aspect; None keeps
    the source's own size). `realtime` paces a non-live source at 1x, like a stream (tests and batch
    runs want `realtime=False`, which also guarantees no dropped frames). `start` seeks a seekable
    source. `quality` picks a Twitch stream's rendition (e.g. "best", "480p"); YouTube instead honours
    `height`. Blocks until the frame size is known or the source fails to open."""
    return FrameSource(spec, fps=fps, realtime=realtime, start=start, height=height, quality=quality)


# ------------------------------------------------------------------------------------
# CLI (debugging aid)
# ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.live.source", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--quality", default="best")
    ap.add_argument("--no-realtime", action="store_true", help="decode as fast as possible instead of at 1x")
    ap.add_argument("--out", type=Path, help="write the frames here as JPEG")
    a = ap.parse_args(argv)

    print(f"kind: {classify(a.spec)}")
    t0 = time.monotonic()
    with open_source(a.spec, fps=a.fps, realtime=not a.no_realtime, start=a.start, height=a.height,
                      quality=a.quality) as src:
        print(f"{src.kind}: {src.width}x{src.height} @ {src.fps:g} fps, live={src.live} "
              f"(opened in {time.monotonic() - t0:.2f}s)")
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
        n = 0
        for frame in src:
            print(f"  frame {frame.index}: t={frame.t:.2f}s shape={frame.image.shape} dropped={src.dropped}")
            if a.out:
                Image.fromarray(frame.image).save(a.out / f"{frame.index:06d}.jpg", quality=90)
            n += 1
            if n >= a.frames:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
