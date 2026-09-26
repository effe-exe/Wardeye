# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The M0 feasibility spike: identification accuracy vs on-screen card size.

See docs/research/04-data-and-evaluation.md §4.8. Three modes:

* **synthetic**: every catalogue image, degraded through the H.264 stream simulator at
  each card height × bitrate, identified against the clean gallery.
* **real**: hand-labeled crops from real streams (`labels.csv`: file,printing_id),
  bucketed by crop height. This is the curve that decides.
* **demo**: synthetic mode on procedural fake cards. Needs no data; used by CI.

Results are appended to a CSV and printed as a table. Accuracy is reported at printing
level and at card level (printings rolled up to gameplay cards).

    python -m rifteye_ml.spike synthetic --catalog catalog.jsonl --cache ~/.cache/rifteye/art \
        --encoder colorgrid --encoder timm:vit_small_patch14_dinov2.lvd142m \
        --heights 40,60,80,120,160 --bitrates 2000,4000,6000 --out reports/m0-synthetic.csv
    python -m rifteye_ml.spike real --catalog catalog.jsonl --cache ~/.cache/rifteye/art \
        --crops crops/ --labels crops/labels.csv --encoder colorgrid --out reports/m0-real.csv
    python -m rifteye_ml.spike demo
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from . import catalog as cat
from .degrade import StreamSettings, simulate
from .encoders import Encoder, get_encoder
from .fixtures import load_fixture_image, synthetic_catalog
from .retrieval import accuracy, ranked_labels, search

FIELDS = ["mode", "encoder", "frame", "card_h", "bitrate_kbps", "rotation", "n",
          "top1_printing", "top5_printing", "top1_card", "top5_card", "seconds"]
HEIGHT_BUCKETS = [(0, 50), (50, 80), (80, 120), (120, 10_000)]


def _score(encoder: Encoder, gallery: np.ndarray, rows: Sequence[dict], queries: Sequence[Image.Image],
           truth_idx: Sequence[int], rotation_invariant: bool) -> dict[str, float]:
    idx, _, _ = search(encoder, gallery, queries, k=min(50, len(rows)), rotation_invariant=rotation_invariant)
    printings = [r["printing_id"] for r in rows]
    cards = [r["card_id"] for r in rows]
    p = accuracy(ranked_labels(idx, printings), [printings[i] for i in truth_idx])
    c = accuracy(ranked_labels(idx, cards), [cards[i] for i in truth_idx])
    return {"n": p["n"], "top1_printing": p["top1"], "top5_printing": p["top5"], "top1_card": c["top1"], "top5_card": c["top5"]}


def run_synthetic(rows: Sequence[dict], load_image: Callable[[dict], Image.Image], encoders: Sequence[Encoder],
                  heights: Sequence[int], bitrates: Sequence[int], base: StreamSettings,
                  mode: str = "synthetic") -> list[dict]:
    images = [load_image(r) for r in rows]
    galleries = [enc.embed(images) for enc in encoders]
    results = []
    for h in heights:
        for br in bitrates:
            s = replace(base, card_h=h, bitrate_kbps=br)
            t0 = time.time()
            crops = simulate(images, s)  # once per setting, shared by every encoder
            sim_seconds = time.time() - t0
            truth = [c.card_index for c in crops]
            for enc, gallery in zip(encoders, galleries):
                for rot_mode in ("search", "oracle"):
                    t1 = time.time()
                    queries = [c.image if rot_mode == "search" else c.upright() for c in crops]
                    m = _score(enc, gallery, rows, queries, truth, rot_mode == "search")
                    results.append({"mode": mode, "encoder": enc.name, "frame": f"{s.frame_w}x{s.frame_h}", "card_h": h,
                                    "bitrate_kbps": br, "rotation": rot_mode, **m,
                                    "seconds": round(time.time() - t1 + sim_seconds, 1)})
                    _print_row(results[-1])
    return results


def run_real(rows: Sequence[dict], load_image: Callable[[dict], Image.Image], encoders: Sequence[Encoder],
             crops_dir: Path, labels_csv: Path) -> list[dict]:
    by_pid = {r["printing_id"]: i for i, r in enumerate(rows)}
    labeled: list[tuple[Image.Image, int]] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            pid = (rec.get("printing_id") or "").strip()
            if pid not in by_pid:
                print(f"  skipping {rec.get('file')}: {pid!r} not in catalogue", file=sys.stderr)
                continue
            labeled.append((Image.open(crops_dir / rec["file"]).convert("RGB"), by_pid[pid]))
    images = [load_image(r) for r in rows]
    results = []
    for enc in encoders:
        gallery = enc.embed(images)
        for lo, hi in HEIGHT_BUCKETS:
            sel = [(im, i) for im, i in labeled if lo <= max(im.size) < hi]
            if not sel:
                continue
            t0 = time.time()
            m = _score(enc, gallery, rows, [im for im, _ in sel], [i for _, i in sel], True)
            results.append({"mode": "real", "encoder": enc.name, "frame": "", "card_h": f"{lo}-{hi if hi < 10_000 else ''}",
                            "bitrate_kbps": "", "rotation": "search", **m, "seconds": round(time.time() - t0, 1)})
            _print_row(results[-1])
    return results


def _print_row(r: dict) -> None:
    print(f"  {r['encoder']:<28} h={str(r['card_h']):>7} br={str(r['bitrate_kbps']):>5} {r['rotation']:<6} "
          f"n={r['n']:<5} printing top1={r['top1_printing']:.3f} top5={r['top5_printing']:.3f}  "
          f"card top1={r['top1_card']:.3f} top5={r['top5_card']:.3f}", flush=True)


def write_csv(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _cached_loader(cache: str) -> Callable[[dict], Image.Image]:
    def load(row: dict) -> Image.Image:
        return Image.open(cat.cache_path(cache, row["image_url"])).convert("RGB")
    return load


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.spike", description="M0 feasibility spike")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("synthetic", "real", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--encoder", action="append", help="repeatable: colorgrid, dhash, timm:<model>")
        p.add_argument("--out", default=f"reports/m0-{name}.csv")
        if name != "real":
            p.add_argument("--heights", default="40,60,80,120,160" if name == "synthetic" else "40,120")
            p.add_argument("--bitrates", default="2000,4000,6000" if name == "synthetic" else "3000")
            p.add_argument("--frame", default="1920x1080" if name == "synthetic" else "1280x720")
            p.add_argument("--seed", type=int, default=0)
        if name != "demo":
            p.add_argument("--catalog", required=True)
            p.add_argument("--cache", required=True, help="image cache written by `catalog download`")
            p.add_argument("--limit", type=int, default=0, help="use only the first N printings (0 = all)")
        else:
            p.add_argument("--cards", type=int, default=60)
        if name == "real":
            p.add_argument("--crops", required=True)
            p.add_argument("--labels", required=True)
    a = ap.parse_args(argv)
    encoders = [get_encoder(e) for e in (a.encoder or ["colorgrid"])]

    if a.mode == "demo":
        rows, loader = synthetic_catalog(a.cards, seed=a.seed), load_fixture_image
    else:
        rows = cat.read_catalog(a.catalog)
        loader = _cached_loader(a.cache)
        rows = [r for r in rows if cat.cache_path(a.cache, r["image_url"]).exists()]
        if a.limit:
            rows = rows[: a.limit]
        if not rows:
            ap.error("no cached images found; run `python -m rifteye_ml.catalog download` first")
    print(f"{len(rows)} printings, {len({r['card_id'] for r in rows})} cards; encoders: {', '.join(e.name for e in encoders)}")

    if a.mode == "real":
        results = run_real(rows, loader, encoders, Path(a.crops), Path(a.labels))
    else:
        fw, fh = (int(x) for x in a.frame.lower().split("x"))
        base = StreamSettings(frame_w=fw, frame_h=fh, seed=a.seed)
        print("stream settings:", {k: v for k, v in asdict(base).items() if k not in ("card_h", "bitrate_kbps")})
        results = run_synthetic(rows, loader, encoders, _ints(a.heights), _ints(a.bitrates), base,
                                mode="demo" if a.mode == "demo" else "synthetic")
    write_csv(results, Path(a.out))
    print(f"{len(results)} rows appended to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
