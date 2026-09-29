# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
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
        --encoder colorgrid --encoder timm:vit_small_patch14_dinov2.lvd142m --realism camera \
        --heights 40,60,80,120,160 --bitrates 2000,4000,6000 --queries 300 --out reports/m0-synthetic.csv
    python -m rifteye_ml.spike real --catalog catalog.jsonl --cache ~/.cache/rifteye/art \
        --crops crops/ --labels crops/labels.csv --encoder colorgrid --out reports/m0-real.csv
    python -m rifteye_ml.spike demo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from . import catalog as cat
from .degrade import REALISM, StreamSettings, simulate, with_realism
from .encoders import Encoder, get_encoder
from .fixtures import load_fixture_image, synthetic_catalog
from .retrieval import ROTATIONS, Gallery, Pyramid, accuracy, at_long_side, band, ranked_labels, search

FIELDS = ["mode", "realism", "queries", "gallery", "view", "encoder", "frame", "card_h", "bitrate_kbps", "rotation", "n",
          "top1_printing", "top5_printing", "top1_card", "top5_card", "seconds"]
HEIGHT_BUCKETS = [(0, 50), (50, 80), (80, 120), (120, 10_000)]


def catalog_key(rows: Sequence[dict], max_side: int) -> str:
    """Identifies an image set for the embedding cache."""
    return hashlib.sha1(json.dumps([[r["printing_id"], r["image_url"]] for r in rows] + [max_side]).encode()).hexdigest()


def _gallery(encoder: Encoder, images: Sequence[Image.Image], scales: Sequence[int],
             cache: Path | None = None, key: str = "", view: str = "full") -> Gallery:
    """The gallery once from the sharp art, or a `Pyramid` at `scales`. `view` embeds only a band
    of each card (see `band`). With `cache`, each level is stored as .npy under a key of encoder,
    scale, view and image set, and reused by later runs."""
    def level(scale: int | None) -> np.ndarray:
        views = [band(im if scale is None else at_long_side(im, scale), view) for im in images]
        if cache is None:
            return encoder.embed(views)
        vtag = "" if view in ("", "full") else f"|{view}"
        tag = hashlib.sha1(f"{encoder.name}|{scale or 'sharp'}{vtag}|{key}".encode()).hexdigest()[:24]
        f = cache / f"{tag}.npy"
        if f.exists():
            return np.load(f)
        x = encoder.embed(views)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, x)
        return x
    return Pyramid({x: level(x) for x in set(scales)}) if scales else level(None)


def _gallery_label(scales: Sequence[int]) -> str:
    return "px:" + ",".join(str(x) for x in sorted(set(scales))) if scales else "sharp"


def _score(encoder: Encoder, gallery: Gallery, rows: Sequence[dict], queries: Sequence[Image.Image],
           truth_idx: Sequence[int], rotation_invariant: bool) -> dict[str, float]:
    idx, _, _ = search(encoder, gallery, queries, k=min(50, len(rows)), rotation_invariant=rotation_invariant)
    printings = [r["printing_id"] for r in rows]
    cards = [r["card_id"] for r in rows]
    p = accuracy(ranked_labels(idx, printings), [printings[i] for i in truth_idx])
    c = accuracy(ranked_labels(idx, cards), [cards[i] for i in truth_idx])
    return {"n": p["n"], "top1_printing": p["top1"], "top5_printing": p["top5"], "top1_card": c["top1"], "top5_card": c["top5"]}


def run_synthetic(rows: Sequence[dict], load_image: Callable[[dict], Image.Image], encoders: Sequence[Encoder],
                  heights: Sequence[int], bitrates: Sequence[int], base: StreamSettings,
                  mode: str = "synthetic", queries: int = 0, realism: str = "codec",
                  query_set: tuple[str, list[Image.Image], list[int]] | None = None,
                  gallery_scales: Sequence[int] = (), cache: Path | None = None, cache_key: str = "",
                  strips: Sequence[str] = ()) -> list[dict]:
    """Degrade the catalogue (or a seeded sample of `queries` printings) and search the full gallery.

    `query_set` = (label, images, gallery index of each image) degrades other printings of the
    gallery's cards instead, such as another language's printings. `gallery_scales` embeds the
    gallery at those on-screen sizes (a `Pyramid`) instead of once from the sharp art.

    `strips` (e.g. "top:0.25") also scores each card from only that band of its upright crop,
    against the same band of every gallery card at the crop's height: what a stack leaves visible."""
    images = [load_image(r) for r in rows]
    galleries = [_gallery(enc, images, gallery_scales, cache, cache_key) for enc in encoders]
    label, q_images, q_truth = query_set or ("same", images, list(range(len(rows))))
    picked = list(range(len(q_images)))
    if 0 < queries < len(q_images):
        picked = sorted(np.random.default_rng(base.seed).choice(len(q_images), size=queries, replace=False).tolist())
    results = []
    for h in heights:
        for br in bitrates:
            s = replace(base, card_h=h, bitrate_kbps=br)
            t0 = time.time()
            crops = simulate([q_images[i] for i in picked], s)  # once per setting, shared by every encoder
            sim_seconds = time.time() - t0
            truth = [q_truth[picked[c.card_index]] for c in crops]
            for enc, gallery in zip(encoders, galleries):
                for rot_mode in ("search", "oracle"):
                    t1 = time.time()
                    batch = [c.image if rot_mode == "search" else c.upright() for c in crops]
                    m = _score(enc, gallery, rows, batch, truth, rot_mode == "search")
                    results.append({"mode": mode, "realism": realism, "queries": label,
                                    "gallery": _gallery_label(gallery_scales), "view": "full", "encoder": enc.name,
                                    "frame": f"{s.frame_w}x{s.frame_h}", "card_h": h,
                                    "bitrate_kbps": br, "rotation": rot_mode, **m,
                                    "seconds": round(time.time() - t1 + sim_seconds, 1)})
                    _print_row(results[-1])
                for view in strips:
                    t1 = time.time()
                    g = _gallery(enc, images, [h], cache, cache_key, view=view)
                    batch = [band(c.upright(), view) for c in crops]
                    m = _score(enc, g, rows, batch, truth, False)
                    results.append({"mode": mode, "realism": realism, "queries": label, "gallery": f"px:{h}",
                                    "view": view, "encoder": enc.name, "frame": f"{s.frame_w}x{s.frame_h}",
                                    "card_h": h, "bitrate_kbps": br, "rotation": "oracle", **m,
                                    "seconds": round(time.time() - t1, 1)})
                    _print_row(results[-1])
    return results


def _level(gallery: Gallery, long_side: int) -> np.ndarray:
    return gallery.levels[gallery.level_for(long_side)] if isinstance(gallery, Pyramid) else gallery


def upright(crops: Sequence[Image.Image], truth_idx: Sequence[int], images: Sequence[Image.Image],
            gallery_scales: Sequence[int] = (), cache: Path | None = None, cache_key: str = "") -> list[Image.Image]:
    """Each labelled crop turned upright: of its four 90° turns, the one the trimmed colour grid
    matches best to the crop's own labelled printing. The label, not a search, picks the turn,
    so a strip is always cut from the card's real top."""
    enc = get_encoder("colorgrid/trim0.03")
    gallery = _gallery(enc, images, gallery_scales, cache, cache_key)
    views = [im.rotate(r, expand=True) if r else im for im in crops for r in ROTATIONS]
    emb = enc.embed(views).reshape(len(crops), len(ROTATIONS), -1)
    out = []
    for n, (im, i) in enumerate(zip(crops, truth_idx)):
        out.append(views[n * len(ROTATIONS) + int(np.argmax(emb[n] @ _level(gallery, max(im.size))[i]))])
    return out


def _score_strips(encoder: Encoder, images: Sequence[Image.Image], rows: Sequence[dict], crops: Sequence[Image.Image],
                  truth_idx: Sequence[int], view: str, gallery_scales: Sequence[int],
                  cache: Path | None, cache_key: str) -> dict[str, float]:
    """Top-1/top-5 from one band of each upright crop, against the same band of every gallery card.
    Each crop meets the pyramid level nearest its whole card's size, not the strip's."""
    gallery = _gallery(encoder, images, gallery_scales, cache, cache_key, view=view)
    k = min(50, len(rows))
    idx = np.zeros((len(crops), k), np.int64)
    groups: dict[int, list[int]] = {}
    for n, im in enumerate(crops):
        groups.setdefault(gallery.level_for(max(im.size)) if isinstance(gallery, Pyramid) else 0, []).append(n)
    for level, members in groups.items():
        g = gallery.levels[level] if isinstance(gallery, Pyramid) else gallery
        idx[members] = search(encoder, g, [band(crops[n], view) for n in members], k=k, rotation_invariant=False)[0]
    printings = [r["printing_id"] for r in rows]
    cards = [r["card_id"] for r in rows]
    p = accuracy(ranked_labels(idx, printings), [printings[i] for i in truth_idx])
    c = accuracy(ranked_labels(idx, cards), [cards[i] for i in truth_idx])
    return {"n": p["n"], "top1_printing": p["top1"], "top5_printing": p["top5"], "top1_card": c["top1"], "top5_card": c["top5"]}


def run_real(rows: Sequence[dict], load_image: Callable[[dict], Image.Image], encoders: Sequence[Encoder],
             crops_dir: Path, labels_csv: Path, gallery_scales: Sequence[int] = (),
             cache: Path | None = None, cache_key: str = "", strips: Sequence[str] = (),
             skip_types: Sequence[str] = (), only_types: Sequence[str] = ()) -> list[dict]:
    """Score labelled real crops, bucketed by size, orientation unknown. `strips` also scores each
    portrait card from one band of its upright crop (see `upright`): what a stack leaves visible
    ('full' scores the whole upright card, the baseline for the bands). Landscape cards
    (battlefields) are left out of the strips; they are not stacked. `skip_types` leaves out
    cards of those types, e.g. legends, which never go in a stack; `only_types` keeps only those."""
    by_pid = {r["printing_id"]: i for i, r in enumerate(rows)}
    labeled: list[tuple[Image.Image, int]] = []
    skipped = 0
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            pid = (rec.get("printing_id") or "").strip()
            if pid not in by_pid:
                print(f"  skipping {rec.get('file')}: {pid!r} not in catalogue", file=sys.stderr)
                continue
            kind = rows[by_pid[pid]].get("type")
            if kind in skip_types or (only_types and kind not in only_types):
                skipped += 1
                continue
            labeled.append((Image.open(crops_dir / rec["file"]).convert("RGB"), by_pid[pid]))
    if skipped:
        print(f"  left out {skipped} crops by card type")
    queries = "real" + (f", not {'/'.join(skip_types)}" if skip_types else "") + (f", {'/'.join(only_types)} only" if only_types else "")
    images = [load_image(r) for r in rows]
    results = []
    for enc in encoders:
        gallery = _gallery(enc, images, gallery_scales, cache, cache_key)
        for lo, hi in HEIGHT_BUCKETS:
            sel = [(im, i) for im, i in labeled if lo <= max(im.size) < hi]
            if not sel:
                continue
            t0 = time.time()
            m = _score(enc, gallery, rows, [im for im, _ in sel], [i for _, i in sel], True)
            results.append({"mode": "real", "realism": "", "queries": queries, "gallery": _gallery_label(gallery_scales),
                            "view": "full",
                            "encoder": enc.name, "frame": "",
                            "card_h": f"{lo}-{hi if hi < 10_000 else ''}",
                            "bitrate_kbps": "", "rotation": "search", **m, "seconds": round(time.time() - t0, 1)})
            _print_row(results[-1])
    if not strips:
        return results
    portrait = [(im, i) for im, i in labeled if rows[i].get("orientation") != "landscape"]
    turned = upright([im for im, _ in portrait], [i for _, i in portrait], images, gallery_scales, cache, cache_key)
    truth = [i for _, i in portrait]
    for enc in encoders:
        for view in strips:
            for lo, hi in HEIGHT_BUCKETS:
                sel = [n for n, im in enumerate(turned) if lo <= max(im.size) < hi]
                if not sel:
                    continue
                t0 = time.time()
                m = _score_strips(enc, images, rows, [turned[n] for n in sel], [truth[n] for n in sel], view,
                                  gallery_scales, cache, cache_key)
                results.append({"mode": "real", "realism": "", "queries": queries, "gallery": _gallery_label(gallery_scales),
                                "view": view, "encoder": enc.name, "frame": "",
                                "card_h": f"{lo}-{hi if hi < 10_000 else ''}",
                                "bitrate_kbps": "", "rotation": "oracle", **m, "seconds": round(time.time() - t0, 1)})
                _print_row(results[-1])
    return results


def _print_row(r: dict) -> None:
    view = "" if r.get("view", "full") == "full" else f" [{r['view']}]"
    print(f"  {r['encoder'] + view:<28} h={str(r['card_h']):>7} br={str(r['bitrate_kbps']):>5} {r['rotation']:<6} "
          f"n={r['n']:<5} printing top1={r['top1_printing']:.3f} top5={r['top5_printing']:.3f}  "
          f"card top1={r['top1_card']:.3f} top5={r['top5_card']:.3f}", flush=True)


def write_csv(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    if not new:
        with open(out, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            raise SystemExit(f"{out} has different columns ({','.join(header)}); write to a new file")
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _cached_loader(cache: str, max_side: int = 0) -> Callable[[dict], Image.Image]:
    """Load cached card images, optionally shrunk so the long side is at most `max_side`.

    Shrinking changes nothing measurable: cards are rendered at most 160 px tall and
    encoders read 224 px inputs, but a full-size gallery would hold gigabytes in memory."""
    def load(row: dict) -> Image.Image:
        im = Image.open(cat.cache_path(cache, row["image_url"])).convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        return im
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
        p.add_argument("--gallery-scales", default="",
                       help="embed the gallery at these on-screen long sides in px, e.g. 48,64,96,128 "
                            "(empty = once, from the sharp art)")
        if name != "real":
            p.add_argument("--heights", default="40,60,80,120,160" if name == "synthetic" else "40,120")
            p.add_argument("--bitrates", default="2000,4000,6000" if name == "synthetic" else "3000")
            p.add_argument("--frame", default="1920x1080" if name == "synthetic" else "1280x720")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--queries", type=int, default=0,
                           help="degrade a seeded sample of N printings; the gallery stays complete (0 = all)")
            p.add_argument("--realism", default="codec", choices=list(REALISM),
                           help="codec: perfect crops, codec only; camera: adds camera, occluders and detector error")
        p.add_argument("--strips", default="",
                       help="also score cards from one band only, e.g. top:0.25,top:0.4,left:0.3 (stacked cards); "
                            "real crops are turned upright by their label first")
        if name == "synthetic":
            p.add_argument("--query-catalog",
                           help="degrade this catalogue's printings (e.g. another language) instead of the gallery's; "
                                "rows are matched to the gallery by printing_id and must have a different image")
        if name != "demo":
            p.add_argument("--catalog", required=True)
            p.add_argument("--cache", required=True, help="image cache written by `catalog download`")
            p.add_argument("--limit", type=int, default=0, help="use only the first N printings (0 = all)")
            p.add_argument("--max-side", type=int, default=512, help="shrink cached images to this long side (0 = off)")
            p.add_argument("--embed-cache", help="directory to keep gallery embeddings in between runs (.npy, private)")
        else:
            p.add_argument("--cards", type=int, default=60)
        if name == "real":
            p.add_argument("--crops", required=True)
            p.add_argument("--labels", required=True)
            p.add_argument("--skip-types", default="", help="leave out cards of these types, e.g. Legend,Battlefield")
            p.add_argument("--only-types", default="", help="keep only cards of these types, e.g. Rune")
    a = ap.parse_args(argv)
    encoders = [get_encoder(e) for e in (a.encoder or ["colorgrid"])]

    if a.mode == "demo":
        rows, loader = synthetic_catalog(a.cards, seed=a.seed), load_fixture_image
    else:
        rows = cat.read_catalog(a.catalog)
        loader = _cached_loader(a.cache, a.max_side)
        rows = [r for r in rows if cat.cache_path(a.cache, r["image_url"]).exists()]
        if a.limit:
            rows = rows[: a.limit]
        if not rows:
            ap.error("no cached images found; run `python -m rifteye_ml.catalog download` first")
    print(f"{len(rows)} printings, {len({r['card_id'] for r in rows})} cards; encoders: {', '.join(e.name for e in encoders)}")

    cache, cache_key = None, ""
    if a.mode != "demo" and a.embed_cache:
        cache, cache_key = Path(a.embed_cache), catalog_key(rows, a.max_side)

    query_set = None
    if a.mode == "synthetic" and a.query_catalog:
        by_pid = {r["printing_id"]: i for i, r in enumerate(rows)}
        qrows = [q for q in cat.read_catalog(a.query_catalog)
                 if q["printing_id"] in by_pid and q["image_url"] != rows[by_pid[q["printing_id"]]]["image_url"]
                 and cat.cache_path(a.cache, q["image_url"]).exists()]
        if not qrows:
            ap.error("no query printings with their own cached image; run `catalog download` on the query catalogue")
        languages = sorted({q["language"] for q in qrows})
        query_set = ("+".join(languages), [loader(q) for q in qrows], [by_pid[q["printing_id"]] for q in qrows])
        print(f"queries: {len(qrows)} {query_set[0]} printings from {a.query_catalog}")

    if a.mode == "real":
        results = run_real(rows, loader, encoders, Path(a.crops), Path(a.labels), _ints(a.gallery_scales),
                           cache, cache_key, strips=[v for v in a.strips.split(",") if v.strip()],
                           skip_types=[t for t in a.skip_types.split(",") if t.strip()],
                           only_types=[t for t in a.only_types.split(",") if t.strip()])
    else:
        fw, fh = (int(x) for x in a.frame.lower().split("x"))
        base = with_realism(StreamSettings(frame_w=fw, frame_h=fh, seed=a.seed), a.realism)
        print("stream settings:", {k: v for k, v in asdict(base).items() if k not in ("card_h", "bitrate_kbps")})
        results = run_synthetic(rows, loader, encoders, _ints(a.heights), _ints(a.bitrates), base,
                                mode="demo" if a.mode == "demo" else "synthetic", queries=a.queries,
                                realism=a.realism, query_set=query_set, gallery_scales=_ints(a.gallery_scales),
                                cache=cache, cache_key=cache_key, strips=[v for v in a.strips.split(",") if v.strip()])
    write_csv(results, Path(a.out))
    print(f"{len(results)} rows appended to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
