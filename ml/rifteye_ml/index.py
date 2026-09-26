# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The shipped embedding index: a row-major float16 matrix plus a manifest.

The manifest follows `EmbeddingIndexManifest` in packages/schema. It names the encoder
that built the vectors. `load_index` refuses to return vectors for any other encoder,
because two encoders' vectors in one search return confident nonsense, silently
(docs/ARCHITECTURE.md §8). The index holds only vectors keyed by collector code: no
card images and no card text (decision D-015).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from .encoders import Encoder


class IndexModelMismatch(RuntimeError):
    pass


def build_index(rows: Sequence[dict], load_image: Callable[[dict], Image.Image], encoder: Encoder,
                out_dir: str | Path, catalog_version: str, batch: int = 64) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vecs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        vecs.append(encoder.embed([load_image(r) for r in chunk]))
    mat = np.concatenate(vecs).astype("<f2") if vecs else np.zeros((0, encoder.dim), "<f2")
    (out / "index.f16").write_bytes(mat.tobytes(order="C"))
    manifest = {
        "schema": "rifteye.index",
        "version": 1,
        "catalogVersion": catalog_version,
        "model": encoder.name,
        "modelSha256": encoder.fingerprint(),
        "dim": int(mat.shape[1]),
        "dtype": "float16",
        "rows": [r["printing_id"] for r in rows],
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_index(index_dir: str | Path, encoder: Encoder) -> tuple[dict, np.ndarray]:
    d = Path(index_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != encoder.name or manifest.get("modelSha256") != encoder.fingerprint():
        raise IndexModelMismatch(
            f"index was built by {manifest.get('model')!r} ({str(manifest.get('modelSha256'))[:12]}), "
            f"but the running encoder is {encoder.name!r} ({encoder.fingerprint()[:12]}). Rebuild the index."
        )
    raw = np.frombuffer((d / "index.f16").read_bytes(), dtype="<f2")
    mat = raw.reshape(len(manifest["rows"]), manifest["dim"]).astype(np.float32)
    return manifest, mat
