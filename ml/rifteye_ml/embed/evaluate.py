# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Score encoders on fresh synthetic crops (an eval bank), split into held-out and training sets.

The protocol is the M0 quick fine-tune's (`adapter`, M0 report §8): crops at the camera level, 1080p at
4 Mbit/s, seed 0, upright; each searched against the whole catalogue at the gallery pyramid level of
its height. Strip views ('top:0.4', ...) score the band a stack leaves visible, portrait cards only,
against the same band of every gallery card (M0 §7).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..encoders import Encoder
from ..retrieval import band
from ..spike import _gallery
from .bank import Bank

FIELDS = ["encoder", "trained_on", "split", "view", "card_h", "n", "top1_card", "top5_card", "top1_printing"]


def score(q: np.ndarray, truth: np.ndarray, g: np.ndarray, cards: np.ndarray, printings: np.ndarray) -> dict[str, float]:
    order = np.argsort(-(q @ g.T), axis=1)[:, :50]
    top1_c = top5_c = top1_p = 0
    for row, t in zip(order, truth):
        ranked = list(dict.fromkeys(cards[row]))
        top1_c += ranked[0] == cards[t]
        top5_c += cards[t] in ranked[:5]
        top1_p += printings[row[0]] == printings[t]
    n = len(truth)
    return {"n": n, "top1_card": top1_c / n, "top5_card": top5_c / n, "top1_printing": top1_p / n}


def evaluate(encoders: dict[str, tuple[Encoder, str]], rows: Sequence[dict], images: Sequence, bank: Bank,
             splits: dict[str, Sequence[int]], views: Sequence[str] = ("full", "top:0.4", "top:0.25"),
             log: Callable[[str], None] = print) -> list[dict]:
    """One row per encoder, split, view and height. `encoders` maps a label to (encoder, trained_on)."""
    cards = np.array([r["card_id"] for r in rows])
    printings = np.array([r["printing_id"] for r in rows])
    portrait = np.array([r.get("orientation") != "landscape" for r in rows])
    heights = sorted({int(h) for h in bank.heights})
    split_of = np.full(len(rows), "", dtype=object)
    for name, members in splits.items():
        split_of[list(members)] = name
    crops = [bank.image(i) for i in range(len(bank))]
    truth_all = bank.rows.astype(np.int64)
    out = []
    for label, (enc, trained_on) in encoders.items():
        for view in views:
            gallery = _gallery(enc, images, heights, view=view)
            sel = np.arange(len(bank)) if view == "full" else np.flatnonzero(portrait[truth_all])
            q = enc.embed([band(crops[i], view) for i in sel])
            for split in splits:
                for h in heights:
                    at = np.flatnonzero((split_of[truth_all[sel]] == split) & (bank.heights[sel] == h))
                    if len(at) == 0:
                        continue
                    m = score(q[at], truth_all[sel][at], gallery.levels[h], cards, printings)
                    out.append({"encoder": label, "trained_on": trained_on, "split": split, "view": view, "card_h": h, **m})
                    log(f"  {label:<18} {split:<13} {view:<9} h={h:>3} n={m['n']:<5} card top1={m['top1_card']:.3f} "
                        f"top5={m['top5_card']:.3f} printing top1={m['top1_printing']:.3f}")
    return out


def write(results: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
