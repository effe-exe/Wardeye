# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The first frames of the LA final as raw RGB, gzipped, for the recognizer's Node replay test
(recognizer.replay.test.ts), which has no JPEG decoder: Pillow decodes them, as Chromium does, to the bytes whose
SHA-256 frames.json lists.

    python packages/engine/test/gen/recognizer_frames.py ~/rifteye-data/m3 --frames 12

Writes fixtures/recognizer/frames-rgb/fNNNN.rgb.gz (width x height x 3 bytes, rows top to bottom, no header; the
size is frames.json's). Private, like the frames (D-006).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("m3", type=Path, help="the private m3 folder")
    ap.add_argument("--frames", type=int, default=12, help="how many, from the first")
    a = ap.parse_args(argv)
    src = a.m3 / "frames" / "la-final"
    out = a.m3 / "fixtures" / "recognizer" / "frames-rgb"
    out.mkdir(parents=True, exist_ok=True)
    listing = json.loads((src / "frames.json").read_text(encoding="utf-8"))
    total = 0
    for fr in listing["frames"][: a.frames]:
        rgb = np.asarray(Image.open(src / fr["file"]).convert("RGB")).tobytes()
        assert hashlib.sha256(rgb).hexdigest() == fr["rgb_sha256"], fr["file"]
        path = out / (Path(fr["file"]).stem + ".rgb.gz")
        path.write_bytes(gzip.compress(rgb, compresslevel=6, mtime=0))
        total += path.stat().st_size
    print(f"{a.frames} frames, {total / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
