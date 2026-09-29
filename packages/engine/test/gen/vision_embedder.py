# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The Python reference for vision-embedder.test.ts: encoders.letterbox and l2n, embed/onnx.batch_of and the Onnx
encoder's name, on seeded synthetic pictures and rows.

    python packages/engine/test/gen/vision_embedder.py    # writes packages/engine/test/vectors/vision-embedder.json

No real crops: small random pictures (a colour grid scaled up, as embed/onnx.random_images makes them) of every
aspect, letterboxed to a small side so the vectors stay small. Bytes and float32 rows are base64, little-endian.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rifteye_ml.embed.onnx import batch_of
from rifteye_ml.encoders import l2n, letterbox

OUT = Path(__file__).resolve().parents[1] / "vectors" / "vision-embedder.json"


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def main() -> None:
    rng = np.random.default_rng(224)
    out: dict = {"note": "made by test/gen/vision_embedder.py; synthetic, seeded"}
    pics = []
    for w, h in ((13, 29), (29, 13), (20, 20), (1, 7), (31, 30), (8, 3)):
        grid = Image.fromarray(rng.integers(0, 256, (4, 4, 3), dtype=np.uint8))
        im = grid.resize((w, h), Image.BICUBIC)
        lb = letterbox(im, 24)
        pics.append({"size": [w, h], "rgb": b64(np.asarray(im).tobytes()), "letterbox": b64(np.asarray(lb).tobytes())})
    out["letterbox"] = {"side": 24, "pictures": pics}
    ims = [Image.frombytes("RGB", tuple(p["size"]), base64.b64decode(p["rgb"])) for p in pics[:3]]
    out["batch_of"] = {"side": 24, "n": 3, "sha256": hashlib.sha256(batch_of(ims, 24).tobytes()).hexdigest()}
    # rows of the graph's kind (near unit length) and of any length, for every branch of numpy's pairwise sum
    rows = {}
    for dim in (256, 7, 100, 136, 300):
        x = rng.normal(0, 1, (4, dim)).astype(np.float32)
        x[0] /= np.linalg.norm(x[0])
        x[1] *= 1e-3
        x[2] = 0.0
        rows[str(dim)] = {"x": b64(x.astype("<f4").tobytes()), "l2n": b64(l2n(x).astype("<f4").tobytes())}
    out["l2n"] = rows
    names = ["embedder-v1.onnx", "embedder-v1.fp16.onnx", "models/onnx/embedder-v1.onnx", "noext", ".hidden", "a.", "x.tar.gz"]
    data = b"not a model, just bytes to hash"
    out["name"] = {"bytes": b64(data), "names": {n: f"onnx:{Path(n).stem}-{hashlib.sha256(data).hexdigest()[:8]}" for n in names}}
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
