# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The standalone extension's data (M2): what the live runner works out when it starts, written once for the browser.

    python -m rifteye_ml.web_assets OUT_DIR --models $RIFTEYE_DATA/models/onnx

    OUT_DIR/gallery/index.json    the rows in order, the embedding size, the levels, the encoder and the model's hash
    OUT_DIR/gallery/L<px>.bin     one level: every row's embedding, float16, little-endian, row after row
    OUT_DIR/catalog.json          the rows as the tracker and the overlay read them
    OUT_DIR/thumbs/<id>.jpg       the hover card's picture, as the runner's /art/<id>.jpg serves it
    OUT_DIR/embed-cache/          what `live.gallery()` keeps (float32 .npy), so a second run only converts

The gallery is `live/__main__.gallery()` itself, run with the ONNX embedder (the float32 file) at every level a
broadcast may need. A table whose cards are `px` long at 1080p is read against the levels
`round(px * f / 10) * 10` for f in 0.8, 0.9 and 1.0 (the runner's formula), so cards from 100 to 200 px take 80 to
200 in steps of 10. The pictures come from the art cache; nothing is fetched. Like the models, all of it is made
from Riot's card art: it goes into the private build's zip and never into the repository (D-006).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import catalog as cat
from .live.__main__ import gallery
from .live.server import LiveServer

FORMAT = 1
FRACTIONS = (0.8, 0.9, 1.0)  # live/__main__: the levels a table's card size reads against
CATALOG_FIELDS = ("printing_id", "card_id", "name", "type")  # what the tracker and the overlay read (live/pipeline.py)
RULE_FIELDS = ("domains", "variant")  # what the legend rule reads (priors.py), when a row has them; a Legend keeps its tags
_SAFE = re.compile(r"[A-Za-z0-9.-]")


def levels_for(px: float) -> list[int]:
    """The gallery levels the live runner builds for cards `px` long at 1080p (its formula, `round` and all)."""
    return sorted({int(round(px * f / 10) * 10) for f in FRACTIONS})


def levels_between(lo: int, hi: int) -> list[int]:
    """Every level a broadcast whose cards are `lo` to `hi` px long at 1080p may need."""
    return sorted({lv for px in range(lo, hi + 1) for lv in levels_for(px)})


def thumb_name(printing_id: str) -> str:
    """The thumbnail's file name, without its extension: the id, with what a file name may not hold (the `*` of
    OGN-299*) as `_` and the byte's two hex digits. The extension's script does the same (thumbs.ts)."""
    return "".join(c if _SAFE.fullmatch(c) else "".join(f"_{b:02x}" for b in c.encode()) for c in printing_id)


def data_dir() -> Path | None:
    """The private data folder, when $RIFTEYE_DATA names it. There is no default: nothing is read from a place the
    caller did not name."""
    v = os.environ.get("RIFTEYE_DATA")
    return Path(v) if v else None


def _under_data(*parts: str, what: str) -> Path:
    data = data_dir()
    if data is None:
        raise SystemExit(f"{what}: name it, or set RIFTEYE_DATA to the private data folder")
    return data.joinpath(*parts)


def find_catalogue(catalog: Path | None) -> Path:
    """The catalogue to use: the given one, else the one with the supplement, else the official one, in $RIFTEYE_DATA."""
    if catalog is not None:
        return catalog
    for name in ("catalog-plus.jsonl", "catalog.jsonl"):
        found = _under_data("catalog", name, what="--catalog")
        if found.exists():
            return found
    raise SystemExit(f"no catalogue in {_under_data('catalog', what='--catalog')}: pass one with --catalog")


def gallery_rows(catalog: Path, art_cache: Path) -> list[dict]:
    """The rows the gallery holds, in order: the catalogue's rows whose art is cached, as the runner reads them."""
    rows = cat.read_catalog(catalog)
    have = [r for r in rows if cat.cache_path(art_cache, r["image_url"]).exists()]
    if not have:
        raise SystemExit(f"no card art cached in {art_cache} for {catalog}")
    if len(have) < len(rows):
        print(f"{len(rows) - len(have)} of {len(rows)} printings have no art in {art_cache} and are left out", flush=True)
    return have


def _write(path: Path, data: bytes) -> None:
    """A file all at once: a run that stops midway leaves no half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_catalog(rows: Sequence[dict], out_dir: Path) -> Path:
    """catalog.json: the rows' fields the tracker and the overlay read, and those the legend rule reads, in the
    gallery's order."""
    slim = [{**{k: r.get(k, "") for k in CATALOG_FIELDS}, **{k: r[k] for k in RULE_FIELDS if k in r},
             **({"tags": r["tags"]} if r.get("type") == "Legend" and r.get("tags") else {})} for r in rows]
    path = out_dir / "catalog.json"
    _write(path, json.dumps(slim, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return path


def write_thumbs(rows: Sequence[dict], art_cache: Path, out_dir: Path) -> int:
    """thumbs/<id>.jpg for every row, made by the live server's own /art code (360 px on the long side, JPEG 85).
    Returns how many were made; the ones already there are kept."""
    art_paths = {r["printing_id"]: cat.cache_path(art_cache, r["image_url"]) for r in rows}
    server = LiveServer(port=0, art=lambda pid: art_paths[pid])  # never started: only its art_jpeg is used
    made = 0
    for r in rows:
        pid = r["printing_id"]
        dest = out_dir / "thumbs" / f"{thumb_name(pid)}.jpg"
        if dest.exists() and dest.stat().st_size:
            continue
        jpeg = server.art_jpeg(pid)
        if jpeg is None:
            raise SystemExit(f"the art of {pid} could not be read: {art_paths[pid]}")
        _write(dest, jpeg)
        made += 1
    return made


class _Timed:
    """An encoder that says how far the gallery is: `live.gallery()` embeds one level at a time."""

    def __init__(self, enc) -> None:
        self.enc, self.done, self.t0 = enc, 0, time.monotonic()
        self.name, self.dim = enc.name, enc.dim

    def embed(self, images):
        out = self.enc.embed(images)
        self.done += 1
        print(f"  {self.done} level(s) embedded, {(time.monotonic() - self.t0) / 60:.1f} min", flush=True)
        return out

    def fingerprint(self) -> str:
        return self.enc.fingerprint()


def write_gallery(enc, rows: Sequence[dict], art_cache: Path, out_dir: Path, levels: Sequence[int],
                  model: dict | None = None) -> dict:
    """gallery/L<px>.bin for every level and gallery/index.json, from `live.gallery()` with its embedding cache in
    out_dir/embed-cache. Returns the index."""
    pyr = gallery(_Timed(enc), list(rows), art_cache, out_dir / "embed-cache", list(levels))
    for lv in levels:
        x = np.ascontiguousarray(pyr.levels[lv], dtype="<f2")
        if x.shape != (len(rows), enc.dim):
            raise SystemExit(f"level {lv} has shape {x.shape}, expected {(len(rows), enc.dim)}")
        _write(out_dir / "gallery" / f"L{lv}.bin", x.tobytes())
    index = {"format": FORMAT, "encoder": enc.name, **(model or {}), "dim": int(enc.dim), "dtype": "float16",
             "levels": list(levels), "rows": [r["printing_id"] for r in rows]}
    _write(out_dir / "gallery" / "index.json", json.dumps(index, separators=(",", ":")).encode("utf-8"))
    return index


def build(out_dir: Path, models: Path, embedder: str = "embedder-v1", catalog: Path | None = None,
          art_cache: Path | None = None, px: tuple[int, int] = (100, 200), threads: int = 2,
          thumbs: bool = True) -> dict:
    """Writes every asset into out_dir and returns the gallery's index."""
    from .embed.onnx import Onnx

    art_cache = art_cache or _under_data("art", what="--cache")
    fp32, fp16 = models / f"{embedder}.onnx", models / f"{embedder}.fp16.onnx"
    if not fp32.exists():
        raise SystemExit(f"{fp32} does not exist: the gallery is embedded with the float32 model")
    rows = gallery_rows(find_catalogue(catalog), art_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_catalog(rows, out_dir)
    print(f"catalog.json: {len(rows)} printings", flush=True)
    if thumbs:
        n = write_thumbs(rows, art_cache, out_dir)
        print(f"thumbs/: {n} made, {len(rows) - n} kept", flush=True)
    enc = Onnx(fp32, threads=threads)
    model = {"model": embedder, "sha256": enc.fingerprint(), "fp16_sha256": _sha256(fp16) if fp16.exists() else None}
    levels = levels_between(*px)
    print(f"gallery: {len(rows)} printings x {len(levels)} levels ({levels[0]}..{levels[-1]}) with {enc.name}", flush=True)
    return write_gallery(enc, rows, art_cache, out_dir, levels, model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.web_assets", description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path, help="where the assets go")
    ap.add_argument("--models", type=Path,
                    help="the folder with the embedder's ONNX files (default: $RIFTEYE_DATA/models/onnx)")
    ap.add_argument("--embedder", default="embedder-v1", help="the model's name: <name>.onnx is embedded, <name>.fp16.onnx hashed")
    ap.add_argument("--catalog", type=Path, help="default: $RIFTEYE_DATA/catalog/catalog-plus.jsonl or catalog.jsonl")
    ap.add_argument("--cache", type=Path, help="the card art cache (default: $RIFTEYE_DATA/art; nothing is fetched)")
    ap.add_argument("--px-min", type=int, default=100, help="the smallest card long side at 1080p a broadcast may have")
    ap.add_argument("--px-max", type=int, default=200, help="the largest")
    ap.add_argument("--threads", type=int, default=2, help="threads for onnxruntime (0: its own choice)")
    ap.add_argument("--no-thumbs", action="store_true", help="leave out thumbs/")
    a = ap.parse_args(argv)
    if not 0 < a.px_min <= a.px_max:
        ap.error("--px-min and --px-max must be positive, the first not above the second")
    models = a.models or _under_data("models", "onnx", what="--models")
    t0 = time.monotonic()
    index = build(a.out_dir, models, a.embedder, a.catalog, a.cache, (a.px_min, a.px_max), a.threads, not a.no_thumbs)
    print(f"done in {(time.monotonic() - t0) / 60:.1f} min: {len(index['rows'])} rows, levels {index['levels']}, {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
