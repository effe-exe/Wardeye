# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The M0 "quick fine-tune": a linear head on a frozen backbone, trained on synthetic crops.

Frozen features match clean crops well and camera-level crops badly. This asks how much
of that gap a cheap trained head closes, and whether it holds for cards the head never
saw. It is a proxy for the real fine-tune in M1, sized to run on a laptop CPU.

1. Degrade the catalogue at every height, `--train-seeds` times, at the camera realism
   level, and embed the crops upright with the frozen backbone (the rectifier's job).
2. Train `W` (dim × dim, initialised to the identity) so that `normalize(W q)` is nearest
   `normalize(W g)` of its own printing, with a full softmax over the gallery pyramid
   level for the crop's height (InfoNCE, temperature τ). Only printings from the
   `--train-sets` take part in training, as crops and as negatives.
3. Evaluate on fresh crops (the spike's seed) of held-out sets and of training sets,
   before and after the head, against the whole gallery.
4. With `--real-crops` and `--real-labels`, also on real stream crops: orientation unknown
   (four rotations, best kept), each crop at the pyramid level nearest its size. This is
   the question that matters: does a head trained only on synthetic crops help on footage?

    python -m rifteye_ml.adapter --catalog catalog.jsonl --cache ~/rifteye-data/art \\
        --embed-cache ~/rifteye-data/embed-cache --encoder timm:vit_small_patch14_dinov2.lvd142m \\
        --train-sets OGN,OGS,SFD --heights 40,60,80,120,160 --out reports/m0-adapter.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import catalog as cat
from .degrade import StreamSettings, simulate, with_realism
from .encoders import Encoder, get_encoder, l2n
from .retrieval import Pyramid
from .spike import _cached_loader, _gallery, _ints, catalog_key

FIELDS = ["encoder", "head", "split", "card_h", "n", "top1_card", "top5_card", "top1_printing"]


def crops_at(images: Sequence, heights: Sequence[int], seeds: Sequence[int], base: StreamSettings,
             pick: Sequence[int]) -> list[tuple[int, int, object]]:
    """(gallery index, height, upright crop) for every picked printing, height and seed."""
    out = []
    for seed in seeds:
        for h in heights:
            s = replace(base, card_h=h, seed=seed)
            for c in simulate([images[i] for i in pick], s):
                out.append((pick[c.card_index], h, c.upright()))
    return out


def train_head(q: np.ndarray, truth: np.ndarray, levels: np.ndarray, gallery: dict[int, np.ndarray],
               epochs: int = 30, tau: float = 0.05, lr: float = 1e-3, weight_decay: float = 1e-4,
               seed: int = 0) -> np.ndarray:
    """Learn W (dim × dim) with InfoNCE over each crop's gallery level. Returns W as float32."""
    import torch

    torch.manual_seed(seed)
    dim = q.shape[1]
    w = torch.nn.Parameter(torch.eye(dim))
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=weight_decay)
    qt, tt = torch.from_numpy(q), torch.from_numpy(truth.astype(np.int64))
    gal = {h: torch.from_numpy(g) for h, g in gallery.items()}
    by_level = {h: torch.from_numpy(np.flatnonzero(levels == h)) for h in gallery}
    for _ in range(epochs):
        for h, members in by_level.items():
            if len(members) == 0:
                continue
            for batch in torch.randperm(len(members)).split(256):
                idx = members[batch]
                zq = torch.nn.functional.normalize(qt[idx] @ w.T, dim=1)
                zg = torch.nn.functional.normalize(gal[h] @ w.T, dim=1)
                loss = torch.nn.functional.cross_entropy(zq @ zg.T / tau, tt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
    return w.detach().numpy().astype(np.float32)


def evaluate(q: np.ndarray, truth: np.ndarray, levels: np.ndarray, gallery: dict[int, np.ndarray],
             rows: Sequence[dict], w: np.ndarray | None) -> dict[int, dict[str, float]]:
    """Card- and printing-level accuracy per height, optionally through the head `w`."""
    cards = np.array([r["card_id"] for r in rows])
    printings = np.array([r["printing_id"] for r in rows])
    out = {}
    for h, g in gallery.items():
        sel = np.flatnonzero(levels == h)
        if len(sel) == 0:
            continue
        zq, zg = (q[sel], g) if w is None else (l2n(q[sel] @ w.T), l2n(g @ w.T))
        order = np.argsort(-(zq @ zg.T), axis=1)[:, :50]
        top1_c = top5_c = top1_p = 0
        for row, t in zip(order, truth[sel]):
            ranked_cards = list(dict.fromkeys(cards[row]))
            top1_c += ranked_cards[0] == cards[t]
            top5_c += cards[t] in ranked_cards[:5]
            top1_p += printings[row[0]] == printings[t]
        n = len(sel)
        out[h] = {"n": n, "top1_card": top1_c / n, "top5_card": top5_c / n, "top1_printing": top1_p / n}
    return out


def evaluate_real(q4: np.ndarray, truth: np.ndarray, levels: np.ndarray, gallery: dict[int, np.ndarray],
                  rows: Sequence[dict], w: np.ndarray | None) -> dict[str, float]:
    """Card-level accuracy on real crops with unknown orientation. `q4` is crops × 4 rotations × dim;
    each gallery row keeps its best rotation, as in `retrieval.search`."""
    cards = np.array([r["card_id"] for r in rows])
    top1 = top5 = 0
    for h, g in gallery.items():
        sel = np.flatnonzero(levels == h)
        if len(sel) == 0:
            continue
        zg = g if w is None else l2n(g @ w.T)
        zq = q4[sel] if w is None else l2n(q4[sel] @ w.T)
        sims = np.einsum("nrd,gd->nrg", zq, zg).max(axis=1)
        for row, t in zip(np.argsort(-sims, axis=1)[:, :50], truth[sel]):
            ranked = list(dict.fromkeys(cards[row]))
            top1 += ranked[0] == cards[t]
            top5 += cards[t] in ranked[:5]
    n = len(truth)
    return {"n": n, "top1_card": top1 / max(1, n), "top5_card": top5 / max(1, n), "top1_printing": float("nan")}


def real_crops(folder: str, labels: str, rows: Sequence[dict]) -> list[tuple[int, str, object]]:
    """(gallery row, track, crop) for every labelled real crop whose printing is in the gallery."""
    from PIL import Image

    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r["printing_id"], i)
    out = []
    for r in csv.DictReader(open(labels, encoding="utf-8")):
        i = first.get(r.get("printing_id", ""))
        if i is not None:
            out.append((i, r.get("track") or r["file"], Image.open(Path(folder) / r["file"]).convert("RGB")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.adapter", description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--embed-cache")
    ap.add_argument("--encoder", default="timm:vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--train-sets", default="OGN,OGS,SFD", help="sets whose printings give training crops")
    ap.add_argument("--heights", default="40,60,80,120,160")
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument("--train-seeds", default="101,102", help="one full pass of crops per seed")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--eval-per-split", type=int, default=150, help="printings sampled per split for evaluation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--frame", default="1920x1080")
    ap.add_argument("--real-crops", help="folder of real stream crops (private)")
    ap.add_argument("--real-labels", help="labels.csv for them: file, printing_id and optionally track")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = [r for r in cat.read_catalog(a.catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
    load = _cached_loader(a.cache, a.max_side)
    images = [load(r) for r in rows]
    heights = _ints(a.heights)
    enc: Encoder = get_encoder(a.encoder)
    pyramid = _gallery(enc, images, heights, Path(a.embed_cache) if a.embed_cache else None, catalog_key(rows, a.max_side))
    assert isinstance(pyramid, Pyramid)
    fw, fh = (int(x) for x in a.frame.lower().split("x"))
    base = with_realism(StreamSettings(frame_w=fw, frame_h=fh, bitrate_kbps=a.bitrate), "camera")

    train_sets = {x.strip() for x in a.train_sets.split(",") if x.strip()}
    seen = [i for i, r in enumerate(rows) if r["set_code"] in train_sets]
    unseen = [i for i, r in enumerate(rows) if r["set_code"] not in train_sets]
    print(f"{len(rows)} printings: {len(seen)} in training sets {sorted(train_sets)}, {len(unseen)} held out")

    def embed(items: list[tuple[int, int, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (enc.embed([im for _, _, im in items]), np.array([i for i, _, _ in items]),
                np.array([h for _, h, _ in items]))

    t0 = time.time()
    q_train, t_train, l_train = embed(crops_at(images, heights, _ints(a.train_seeds), base, seen))
    print(f"{len(t_train)} training crops embedded in {time.time() - t0:.0f} s")
    # Train against the training sets' printings only, so held-out cards stay unseen by the head.
    local = {g: k for k, g in enumerate(seen)}
    w = train_head(q_train, np.array([local[int(i)] for i in t_train]), l_train,
                   {h: g[np.array(seen)] for h, g in pyramid.levels.items()}, epochs=a.epochs)

    rng = np.random.default_rng(a.eval_seed)
    results = []
    for split, pool in (("held-out sets", unseen), ("training sets", seen)):
        if not pool:
            continue
        pick = sorted(rng.choice(pool, size=min(a.eval_per_split, len(pool)), replace=False).tolist())
        q, t, lv = embed(crops_at(images, heights, [a.eval_seed], base, pick))
        for head, mat in (("frozen", None), ("linear", w)):
            for h, m in sorted(evaluate(q, t, lv, pyramid.levels, rows, mat).items()):
                results.append({"encoder": enc.name, "head": head, "split": split, "card_h": h, **m})
                print(f"  {head:<6} {split:<13} h={h:>3} n={m['n']:<4} card top1={m['top1_card']:.3f} "
                      f"top5={m['top5_card']:.3f} printing top1={m['top1_printing']:.3f}", flush=True)

    if a.real_crops and a.real_labels:
        items = real_crops(a.real_crops, a.real_labels, rows)
        t0 = time.time()
        rots = (0, 90, 180, 270)
        q4 = enc.embed([im.rotate(r, expand=True) if r else im for _, _, im in items for r in rots])
        q4 = q4.reshape(len(items), len(rots), -1)
        print(f"{len(items)} real crops embedded in {time.time() - t0:.0f} s")
        truth = np.array([i for i, _, _ in items])
        levels = np.array([pyramid.level_for(max(im.size)) for _, _, im in items])
        tracks = np.array([t for _, t, _ in items])
        in_train = np.array([rows[i]["set_code"] in train_sets for i in truth])
        one_per_track = np.array(sorted({t: k for k, t in reversed(list(enumerate(tracks)))}.values()))
        subsets = [("real", np.arange(len(items))), ("real, one per track", one_per_track),
                   ("real, training sets", np.flatnonzero(in_train)), ("real, held-out sets", np.flatnonzero(~in_train))]
        for name, sel in subsets:
            if len(sel) == 0:
                continue
            for head, mat in (("frozen", None), ("linear", w)):
                m = evaluate_real(q4[sel], truth[sel], levels[sel], pyramid.levels, rows, mat)
                results.append({"encoder": enc.name, "head": head, "split": name, "card_h": "real", **m})
                print(f"  {head:<6} {name:<20} n={m['n']:<5} card top1={m['top1_card']:.3f} top5={m['top5_card']:.3f}",
                      flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for r in results:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
