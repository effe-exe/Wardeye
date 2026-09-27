# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Realism check for a synthetic run: name its fully visible cards the way real ones are named.

Each face-up card that is fully visible and inside the frame is cut out through its quad, with the
box a few per cent off as a detector's would be, and searched against the gallery with the M0
baselines. If synthetic cards are much easier to name than the reviewed real crops of the M0
broadcasts (card-level top-1: colour grid 70-95%, colour grid + dHash 81-97%), the run is too
clean to teach an embedder what real footage does to cards.

    python -m rifteye_ml.synth.check --run ~/rifteye-data/synth/v0 \
        --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art --out m1-synth-check.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import catalog as cat
from ..degrade import _perspective_coeffs
from ..encoders import get_encoder
from ..fixtures import load_fixture_image, synthetic_catalog
from ..retrieval import search
from ..spike import _cached_loader, _gallery, _ints, catalog_key

BUCKETS = [(0, 60), (60, 100), (100, 140), (140, 10_000)]


def visible_crops(run: Path, by_pid: dict[str, int], jitter: tuple[float, float] = (0.02, 0.015), seed: int = 0):
    """(crop, gallery row, long side, zone, foil) for every fully visible face-up card in the run."""
    rng = np.random.default_rng(seed)
    out = []
    with open(run / "annotations.jsonl", encoding="utf-8") as f:
        anns = [json.loads(line) for line in f]
    for a in anns:
        frame = None
        for c in a["cards"]:
            if c["kind"] != "card" or c["visible"] < 0.95 or c["truncated"] or c.get("printing_id") not in by_pid:
                continue
            if frame is None:
                with Image.open(run / a["image"]) as im:
                    frame = im.convert("RGB")
            q = np.array(c["quad"], np.float64).reshape(4, 2)
            long_px = max(np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[0] - q[1]))
            w, h = (round(long_px), round(long_px * 63 / 88)) if c["landscape"] else (round(long_px * 63 / 88), round(long_px))
            cen = q.mean(axis=0)
            q = cen + (q - cen) * (1 + rng.uniform(-jitter[0], jitter[0])) + rng.uniform(-jitter[1], jitter[1], size=2) * long_px
            dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
            crop = frame.transform((max(2, w), max(2, h)), Image.PERSPECTIVE, _perspective_coeffs(dst, q), Image.BICUBIC)
            out.append((crop, by_pid[c["printing_id"]], round(long_px), c["zone"], c["foil"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.synth.check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a folder written by python -m rifteye_ml.synth")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--catalog")
    src.add_argument("--fixtures", type=int, help="the run was made with --fixtures N")
    ap.add_argument("--cache")
    ap.add_argument("--embed-cache")
    ap.add_argument("--encoder", action="append", help="default: the colour grid, dHash, and the two together")
    ap.add_argument("--gallery-scales", default="40,60,80,120,140,160")
    ap.add_argument("--jitter", default="0.02,0.015", help="detector box error: scale and shift, as fractions")
    ap.add_argument("--out", help="CSV of the numbers (safe to share)")
    a = ap.parse_args(argv)
    if a.catalog:
        rows = [r for r in cat.read_catalog(a.catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
        load, key = _cached_loader(a.cache, 512), catalog_key(rows, 512)
    else:
        rows, load, key = synthetic_catalog(a.fixtures), load_fixture_image, f"fixtures:{a.fixtures}"
    by_pid = {r["printing_id"]: i for i, r in enumerate(rows)}
    cards = [r["card_id"] for r in rows]
    images = [load(r) for r in rows]
    jit = tuple(float(v) for v in a.jitter.split(","))
    items = visible_crops(Path(a.run), by_pid, jit)  # type: ignore[arg-type]
    if not items:
        ap.error("no fully visible face-up cards in the run")
    print(f"{len(items)} fully visible face-up cards, long side median {int(np.median([i[2] for i in items]))} px")
    results = []
    for spec in a.encoder or ["colorgrid/trim0.03", "dhash/trim0.03", "colorgrid/trim0.03+dhash/trim0.03"]:
        enc = get_encoder(spec)
        gal = _gallery(enc, images, _ints(a.gallery_scales), Path(a.embed_cache) if a.embed_cache else None, key)
        idx, _, _ = search(enc, gal, [i[0] for i in items], k=5, rotation_invariant=False)  # upright via the quad
        ok = np.array([cards[r[0]] == cards[t[1]] for r, t in zip(idx, items)])
        groups = {"all": np.ones(len(items), bool), "legends": np.array([i[3] == "legend" for i in items]),
                  "foil": np.array([i[4] for i in items])}
        for lo, hi in BUCKETS:
            groups[f"{lo}-{hi if hi < 10_000 else ''}px"] = np.array([lo <= i[2] < hi for i in items])
        for g, sel in groups.items():
            if sel.any():
                results.append({"encoder": enc.name, "group": g, "n": int(sel.sum()), "top1_card": round(float(ok[sel].mean()), 4)})
        print(f"  {enc.name:<40} " + "  ".join(f"{r['group']} {r['top1_card']:.3f}" for r in results if r["encoder"] == enc.name))
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["encoder", "group", "n", "top1_card"])
            w.writeheader(); w.writerows(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
