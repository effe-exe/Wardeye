# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The card embedder (M1): crop banks, fine-tuning, packing and scoring.

    python -m rifteye_ml.embed crops --catalog catalog.jsonl --cache ~/rifteye-data/art --out bank --seeds 1-16
    python -m rifteye_ml.embed crops --catalog ... --cache ... --out bank-eval --eval --heights 40,60,80,120,160
    python -m rifteye_ml.embed real-bank --catalog ... --cache ... --crops la-v1/crops --labels la-v1/labels.csv --out real-la
    python -m rifteye_ml.embed train --catalog ... --cache ... --bank bank --out runs/heldout --train-sets OGN,OGS,SFD
    python -m rifteye_ml.embed pack --checkpoint runs/heldout/final.pt --out embedder-v0-heldout.pth
    python -m rifteye_ml.embed evaluate --catalog ... --cache ... --bank bank-eval --train-sets OGN,OGS,SFD \\
        --encoder timm:vit_small_patch14_dinov2.lvd142m --encoder embedder:embedder-v0-heldout.pth --out scores.csv

The packed weights work wherever an encoder spec is taken, e.g. `spike real --encoder embedder:<file>`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .. import catalog as cat
from ..spike import _cached_loader, _ints

HEIGHTS = "36,43,51,61,73,87,104,124,148,176"  # log-spaced over the sizes a card has on a stream
EVAL_HEIGHTS = "40,60,80,120,160"                # the M0 report's


def seeds(s: str) -> list[int]:
    """'1-16' or '1,2,5'."""
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        if a:
            out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def catalogue(path: str, cache: str, limit: int = 0) -> list[dict]:
    """The printings whose art is cached; with `limit`, that many spread evenly over the catalogue."""
    rows = [r for r in cat.read_catalog(path) if cat.cache_path(cache, r["image_url"]).exists()]
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def in_bank_order(rows: list[dict], printings: list[str]) -> list[dict]:
    by_id = {r["printing_id"]: r for r in rows}
    missing = [p for p in printings if p not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} printings of the bank are not in the catalogue (or their art is not cached), "
                         f"e.g. {missing[0]}")
    return [by_id[p] for p in printings]


def sets_of(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def _crops(a) -> int:
    from .bank import generate, plan, plan_eval

    rows = catalogue(a.catalog, a.cache, a.limit)
    frame = tuple(int(x) for x in a.frame.lower().split("x")) if a.frame else None
    tasks = plan_eval(_ints(a.heights or EVAL_HEIGHTS), seed=a.eval_seed, frame=frame or (1920, 1080)) if a.eval else \
        plan(seeds(a.seeds), _ints(a.heights or HEIGHTS), text_prob=a.text_prob, realism=a.realism, frame=frame)
    workers = a.workers or min(os.cpu_count() or 1, 8)
    t0 = time.time()
    print(f"{len(rows)} printings, {len(tasks)} tasks ({len(rows) * len(tasks)} crops) on {workers} processes", flush=True)
    ran = generate(rows, a.cache, tasks, a.out, workers=workers, max_side=a.max_side, log=lambda m: print(m, flush=True))
    print(f"{ran} tasks in {time.time() - t0:.0f} s -> {a.out}")
    return 0


def _real_bank(a) -> int:
    """Reviewed real crops (labels.csv: file, printing_id) as a bank. A crop whose label is not a printing
    of the catalogue is left out: a sleeve (never identified, D-005), a token, an unsure or wrong answer."""
    import csv

    from PIL import Image

    from ..spike import upright
    from .bank import write_real

    rows = catalogue(a.catalog, a.cache, a.limit)
    by_pid = {r["printing_id"]: i for i, r in enumerate(rows)}
    crops, truth, left_out = [], [], 0
    with open(a.labels, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            i = by_pid.get((rec.get("printing_id") or "").strip())
            path = Path(a.crops) / rec["file"]
            if i is None or not path.exists():
                left_out += 1
                continue
            crops.append(Image.open(path).convert("RGB"))
            truth.append(i)
    if crops:  # turned upright by their label, like the synthetic crops
        uniq = sorted(set(truth))
        pos = {r: k for k, r in enumerate(uniq)}
        load = _cached_loader(a.cache, 256)
        crops = upright(crops, [pos[i] for i in truth], [load(rows[r]) for r in uniq])
    n = write_real(crops, truth, [r["printing_id"] for r in rows], a.out, repeat=a.repeat)
    print(f"{n} real crops of {len(set(truth))} printings ({left_out} left out), each {a.repeat}x an epoch -> {a.out}")
    return 0


def _train(a) -> int:
    from .bank import Bank
    from .model import train

    bank = Bank(a.bank)
    rows = in_bank_order(catalogue(a.catalog, a.cache), bank.printings)
    wanted = sets_of(a.train_sets)
    keep = [i for i, r in enumerate(rows) if not wanted or r["set_code"] in wanted]
    if not keep:
        raise SystemExit(f"no printings in the training sets {sorted(wanted)}")
    t0 = time.time()
    images = [_cached_loader(a.cache, 256)(r) for r in rows]  # clean renders are at most 176 px
    print(f"{len(rows)} printings loaded in {time.time() - t0:.0f} s; training on {len(keep)} "
          f"({'all sets' if not wanted else ', '.join(sorted(wanted))})", flush=True)
    amp = {"auto": None, "on": True, "off": False}[a.amp]
    train(bank, rows, images, a.out, train_rows=keep, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          lr_head=a.lr_head, clean_per_printing=a.clean, device=a.device, workers=a.workers, seed=a.seed, amp=amp,
          backbone=a.backbone, img_size=a.img_size, dim=a.dim, pretrained=not a.no_pretrained,
          log=lambda m: print(m, flush=True), meta={"trained_on": ",".join(sorted(wanted)) or "all"})
    print(f"trained in {(time.time() - t0) / 60:.0f} min -> {Path(a.out) / 'final.pt'}")
    return 0


def _pack(a) -> int:
    from .model import pack

    size = pack(a.checkpoint, a.out)
    print(f"{a.out}: {size / 1e6:.1f} MB")
    return 0


def _evaluate(a) -> int:
    from ..encoders import get_encoder
    from .bank import Bank
    from .evaluate import evaluate, write

    bank = Bank(a.bank)
    rows = in_bank_order(catalogue(a.catalog, a.cache), bank.printings)
    wanted = sets_of(a.train_sets)
    splits = {"held-out sets": [i for i, r in enumerate(rows) if r["set_code"] not in wanted],
              "training sets": [i for i, r in enumerate(rows) if r["set_code"] in wanted]}
    splits = {k: v for k, v in splits.items() if v}
    images = [_cached_loader(a.cache, a.max_side)(r) for r in rows]
    encoders = {}
    for spec in a.encoder:
        enc = get_encoder(spec)
        meta = getattr(enc, "meta", None)
        label = Path(spec.split(":", 1)[1]).stem if spec.startswith("embedder:") else \
            "frozen" if spec.startswith("timm:") else spec
        encoders[label] = (enc, meta.get("trained_on", "?") if meta is not None else "none")
    t0 = time.time()
    results = evaluate(encoders, rows, images, bank, splits, views=[v.strip() for v in a.views.split(",") if v.strip()],
                       log=lambda m: print(m, flush=True))
    write(results, a.out)
    print(f"{len(results)} rows in {time.time() - t0:.0f} s -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.embed", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data(p: argparse.ArgumentParser) -> None:
        p.add_argument("--catalog", required=True)
        p.add_argument("--cache", required=True, help="the art cache")

    p = sub.add_parser("crops", help="degrade the catalogue into a crop bank (every CPU core)")
    data(p)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="1-16", help="one pass over every printing and height per seed")
    p.add_argument("--heights", help=f"card heights in px (default {HEIGHTS}, or {EVAL_HEIGHTS} with --eval)")
    p.add_argument("--eval", action="store_true", help="fresh crops for scoring, as the M0 report made them")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--realism", default="foil")
    p.add_argument("--text-prob", type=float, default=0.2, help="share of standard frames whose rules text is swapped")
    p.add_argument("--workers", type=int, default=0, help="processes (default: the CPU cores, at most 8)")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--frame", help="WxH for every task (default: 1080p or 720p per task; 1080p with --eval)")
    p.add_argument("--limit", type=int, default=0, help="only this many printings, spread over the catalogue (tests)")
    p.set_defaults(fn=_crops)

    p = sub.add_parser("real-bank", help="reviewed crops from real broadcasts as a bank to train on")
    data(p)
    p.add_argument("--crops", required=True, help="the folder the labels' file names are in")
    p.add_argument("--labels", required=True, help="labels.csv with file and printing_id")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=1, help="times an epoch each crop is seen")
    p.add_argument("--limit", type=int, default=0, help="as for crops: must match the synthetic bank's")
    p.set_defaults(fn=_real_bank)

    p = sub.add_parser("train", help="fine-tune on a crop bank")
    data(p)
    p.add_argument("--bank", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-sets", default="", help="comma-separated set codes; default: every set")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5, help="the backbone's last block; earlier blocks get less")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--clean", type=int, default=20, help="clean renders of each printing per epoch")
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="bfloat16 autocast (auto: on CUDA)")
    p.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--no-pretrained", action="store_true", help="random weights (tests)")
    p.set_defaults(fn=_train)

    p = sub.add_parser("pack", help="float16 weights for retrieval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_pack)

    p = sub.add_parser("evaluate", help="score encoders on an eval bank")
    data(p)
    p.add_argument("--bank", nargs="+", required=True)
    p.add_argument("--train-sets", default="OGN,OGS,SFD", help="the sets the held-out model trained on")
    p.add_argument("--encoder", action="append", required=True, help="repeatable: timm:<model> or embedder:<file>")
    p.add_argument("--views", default="full,top:0.4,top:0.25")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_evaluate)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
