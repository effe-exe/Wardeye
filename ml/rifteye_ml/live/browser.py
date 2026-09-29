# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Frames from the browser extension: what the viewer is watching on Twitch, in its own player.

    python -m rifteye_ml.live --source browser --detector detector-v0.pth

The extension (`apps/extension`) grabs the player's current frame a few times a second and posts it
to `POST /frame` on this machine's live server, with the video's own time; the runner reads it like
any other source and the extension draws the board it answers with on top of the player. Frames
never leave this machine, and only the table window is ever looked at (D-005), as with a stream.

Like `FrameSource`, only the newest frame waits: a frame posted while the previous one is still being
read replaces it, so the board follows the video instead of falling behind it.
"""
from __future__ import annotations

import threading
import time
from io import BytesIO
from typing import Iterator

import numpy as np
from PIL import Image

from .source import Frame


class BrowserSource:
    kind = "browser"

    def __init__(self, fps: float = 5.0) -> None:
        self.fps = fps
        self.width = self.height = 0
        self._cond = threading.Condition()
        self._latest: tuple[bytes, float, str, float] | None = None
        self._closed = False
        self._index = 0

    def post(self, jpeg: bytes, t: float, video: str = "") -> None:
        """A frame from the extension, at time `t` (s) of the video `video` (its page, e.g. /videos/123).
        Replaces one not yet read."""
        with self._cond:
            self._latest = (jpeg, t, video, time.monotonic())
            self._cond.notify_all()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            with self._cond:
                while self._latest is None and not self._closed:
                    self._cond.wait(timeout=1.0)
                if self._closed:
                    return
                jpeg, t, video, wall = self._latest  # type: ignore[misc]  # the loop above guarantees this
                self._latest = None
            try:
                image = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"))
            except Exception:
                continue  # a broken upload is skipped, not fatal
            self.height, self.width = image.shape[:2]
            frame = Frame(t=t, image=image, index=self._index, wall=wall)
            frame.video = video  # type: ignore[attr-defined]  # which video: another one is another board
            yield frame
            self._index += 1

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __enter__(self) -> "BrowserSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
