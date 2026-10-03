# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The expected values for test/recognizer.test.ts, computed by live/pipeline.py itself on seeded synthetic inputs:

* quad, aabb, smooth and similarity on random boxes and points (the boxes hold numpy floats, as detector_boxes
  makes them, so quad rounds the numpy way), math.hypot, and numpy's float32 and float64 sums;
* the scene's float32 arithmetic (Scene.score and Scene.learn) on thumbnails drawn by a formula, and Scene.on_table
  on frames drawn by a formula (learning the table camera, losing it when the camera moves, learning it again);
* the tracker's rules (label, due, covered, stacked_on, twin, vanished, stacks, state, announce, left, recently_played,
  reanchor) on random boards of tracks, set up by hand on a Recognizer with a made-up catalogue;
* the bootstrap finder (Recognizer.find with no finder) on a mat with cards, drawn by formula;
* the overlay laid over every shot (Scene.see and overlay_patches) on thumbnails drawn by formula, with cuts between
  shots and a corner that stays; the still table (StillTable) on frames drawn by formula; the hand rule against it
  (hand_share with `still` and `now`) on a wooden table; and printed zones (plain_zone).

No pictures, no card art, nothing from a broadcast: the catalogue's names are made up and the boards are random.

    RIFTEYE_DATA=/tmp python packages/engine/test/gen/recognizer.py   # -> packages/engine/test/vectors/recognizer.json
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

from rifteye_ml.live import pipeline as pl
from rifteye_ml.live.layouts import LAYOUTS
from rifteye_ml.matcrops import CardBox
from rifteye_ml.retrieval import Pyramid

OUT = Path(__file__).resolve().parents[1] / "vectors" / "recognizer.json"
LAYOUT = LAYOUTS["la-rq"]
W, H = 1920, 1080


def f(x) -> float:
    return float(x)


def box_json(b: CardBox) -> dict:
    d = {"centre": [f(b.centre[0]), f(b.centre[1])], "long_px": f(b.long_px), "short_px": f(b.short_px),
         "angle_deg": f(b.angle_deg), "fill": f(b.fill)}
    for k in ("back", "score", "vis"):
        if hasattr(b, k):
            v = getattr(b, k)
            d[k] = bool(v) if k == "back" else f(v)
    return d


def det_box(cx, cy, long, short, angle, back=None) -> CardBox:
    """A box as detector_boxes makes it: the centre and sides numpy floats, the angle a Python float."""
    b = CardBox((np.float64(cx), np.float64(cy)), np.float64(long), np.float64(short), float(angle), 1.0)
    if back is not None:
        b.back = back
    return b


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


# --- pure helpers ----------------------------------------------------------------------------------------------

def series64(n: int) -> np.ndarray:
    """n float64 values from an integer formula (the test computes the same)."""
    i = np.arange(n)
    return ((i * 7919 + n * 104729) % 20011 - 10005) / 3.0


def series32(n: int) -> np.ndarray:
    """n float32 values, exact in float32 (multiples of 1/64 below 2^8)."""
    i = np.arange(n)
    return (((i * 7919 + n * 104729) % 20011 - 10005) / 64.0).astype(np.float32)


def geometry(rng) -> dict:
    boxes = []
    for _ in range(40):
        b = det_box(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(40, 200), rng.uniform(30, 150), rng.uniform(0, 180))
        boxes.append(b)
    # ties at the rounding digit: numpy's round (rint of x * 10) and Python's round differ on these
    for cx in (95.35, 100.25, 1003.45, 12.05):
        boxes.append(det_box(cx, cx / 2, 10.0, 6.0, 0.0))
        boxes.append(det_box(cx, cx / 2, 10.0, 6.0, 90.0))
    quads = [{"box": box_json(b), "quad": plain(pl.quad(b)), "aabb": plain(pl.aabb(b))} for b in boxes]

    smooths = []
    for _ in range(60):
        old = det_box(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(80, 180), rng.uniform(60, 130), rng.uniform(0, 180))
        mode = rng.integers(0, 4)
        if mode == 0:  # jitter: eased
            new = det_box(old.centre[0] + rng.normal(0, 3), old.centre[1] + rng.normal(0, 3), old.long_px + rng.normal(0, 2),
                          old.short_px + rng.normal(0, 2), (old.angle_deg + rng.normal(0, 3)) % 180, back=bool(rng.integers(0, 2)))
        elif mode == 1:  # across the 0/180 seam
            old = det_box(old.centre[0], old.centre[1], old.long_px, old.short_px, rng.uniform(0, 4))
            new = det_box(old.centre[0] + 1, old.centre[1] - 1, old.long_px, old.short_px, rng.uniform(176, 180))
        elif mode == 2:  # moved: followed at once
            new = det_box(old.centre[0] + rng.uniform(20, 60), old.centre[1], old.long_px, old.short_px, old.angle_deg)
        else:  # turned: followed at once
            new = det_box(old.centre[0], old.centre[1], old.long_px, old.short_px, (old.angle_deg + rng.uniform(15, 90)) % 180)
        smooths.append({"old": box_json(old), "new": box_json(new), "out": box_json(pl.smooth(old, new))})

    sims = []
    for n in (2, 2, 3, 4, 5, 7, 9, 12, 16):
        src = rng.uniform(0, 1500, (n, 2))
        th = rng.uniform(-math.pi, math.pi)
        s = rng.uniform(0.6, 1.6)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        dst = s * src @ rot.T + rng.uniform(-300, 300, 2) + rng.normal(0, 2, (n, 2))
        if n == 16:
            dst[:, 0] = -dst[:, 0]  # a mirror: the best proper rotation
        scale, r, shift = pl.similarity(src, dst)
        sims.append({"src": src.tolist(), "dst": dst.tolist(), "scale": scale, "rot": r.tolist(), "shift": shift.tolist()})
    same = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    scale, r, shift = pl.similarity(same, same + 5)
    sims.append({"src": same.tolist(), "dst": (same + 5).tolist(), "scale": scale, "rot": r.tolist(), "shift": shift.tolist()})

    hyp = [[x, y, math.hypot(x, y)] for x, y in rng.uniform(0, 300, (30, 2)).tolist()]

    # the values come from a formula the test repeats: only the sums are stored
    sums64 = [{"n": n, "sum": float(series64(n).sum())} for n in (1, 3, 7, 8, 9, 15, 16, 17, 60, 127, 128, 129, 200, 300, 1000, 5000)]
    sums32 = [{"n": n, "sum": float(series32(n).sum())} for n in (1, 5, 8, 9, 24, 100, 128, 129, 257, 1000, 3001, 15552)]
    return {"quads": quads, "smooth": smooths, "similarity": sims, "hypot": hyp, "sums64": sums64, "sums32": sums32}


# --- the scene's float32 arithmetic ------------------------------------------------------------------------------

def thumb(k: int, flat: bool = False, cut: bool = False) -> np.ndarray:
    """A 96 x 54 thumbnail drawn by integer formulas (the test draws the same): a still pattern (the overlay, the mat's
    print), a region that changes (the board) and a little jitter; a cut is another pattern altogether."""
    y, x, c = np.meshgrid(np.arange(54), np.arange(96), np.arange(3), indexing="ij")
    if cut:
        v = (x * 3 + y * 29 + c * 71 + k * 13) % 256
    else:
        base = np.full_like(x, 60) if flat else (x * 7 + y * 13 + c * 50) % 200 + 20
        noise = (x * 31 + y * 17 + c * 5 + k * 11) % 7 - 3
        board = (x >= 30) & (x < 60) & (y >= 10) & (y < 40)
        v = np.where(board, (k * 37 + x * 3 + y * 5) % 256, base + noise)
    return v.astype(np.float32)


def scene_vectors() -> list[dict]:
    """Per frame: the score the scene gives it (null before five learns, or when the still parts have no pattern),
    and whether it was learnt; the frames come every 0.5 s and are learnt at most every 2 s."""
    out = []
    for name, frames in (("patterned", [(k, False, k in (40, 41, 42)) for k in range(60)]),
                         ("flat", [(k, True, False) for k in range(30)])):
        sc = pl.Scene(LAYOUT)
        steps = []
        for k, flat, cut in frames:
            t = 0.5 * k
            x = thumb(k, flat, cut)
            s = sc.score(x) if sc.n >= 5 else None
            ok = s is None or s >= sc.corr
            if ok:
                sc.learn(t, x)
            steps.append({"k": k, "flat": flat, "cut": cut, "t": t, "score": s, "learnt": bool(ok), "n": sc.n})
        out.append({"name": name, "steps": steps,
                    "mean_sum": float(np.float64(sc.mean.astype(np.float64).sum())), "var_sum": float(sc.var.astype(np.float64).sum()),
                    "mean_head": [float(v) for v in sc.mean.ravel()[:64]], "var_head": [float(v) for v in sc.var.ravel()[:64]]})
    return out


def scene_frame(shift: int) -> np.ndarray:
    """A 1920 x 1080 table-camera frame drawn by formula (the test draws the same): around the table window the
    broadcast's overlay, a print of 40 px blocks; inside it la-rq's mat with a print of its own in 40 px blocks, moved
    `shift` px when the camera moves (the overlay does not move with it)."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.empty((H, W, 3), np.uint8)
    for c in range(3):
        bx, by = x // 40, y // 40
        img[..., c] = (bx * bx * 37 + by * by * 91 + bx * by * 13 + c * 50) % 256
    x0, y0, x1, y1 = LAYOUT.box(W, H)
    for c in range(3):
        bx, by = (x[y0:y1, x0:x1] + shift) // 40, y[y0:y1, x0:x1] // 40
        img[y0:y1, x0:x1, c] = LAYOUT.mat[c] + (bx * bx * 37 + by * by * 91 + bx * by * 13 + c * 50) % 40
    return img


def scene_on_table(cards: int, shown: int = 0) -> list[dict]:
    """Scene.on_table over 81 frames at 2 fps: the table camera for 10 s (it learns it), then the same table seen by a
    camera that moved (the mat's print is 40 px off): unrecognised, away, until after 20 s three table-like looks with
    `cards` card-sized boxes in the view (RELEARN_CARDS, and half the `shown` the table camera showed) learn the view
    again; without them it stays away."""
    sc = pl.Scene(LAYOUT)
    sc.saw_cards(0.0, shown)
    frames = {0: scene_frame(0), 40: scene_frame(40)}
    out = []
    for k in range(81):
        t = 0.5 * k
        shift = 0 if t <= 10.0 else 40
        ok = sc.on_table(t, frames[shift], lambda: cards)
        out.append({"t": t, "shift": shift, "ok": bool(ok), "n": sc.n, "away_since": sc.away_since, "looks": sc.looks,
                    "last_look": sc.last_look, "last_learn": sc.last_learn})
    return out


# --- the tracker's rules on random boards --------------------------------------------------------------------------

TYPES = ["Unit", "Unit", "Unit", "Gear", "Spell", "Legend", "Battlefield", "Rune"]


def catalogue() -> list[dict]:
    """Made-up printings: 16 cards, some with two printings, of every type the rules look at."""
    rows = []
    for c in range(16):
        kind = TYPES[c % len(TYPES)]
        for p in range(1 + (c % 3 == 0)):
            rows.append({"printing_id": f"TST-{c:02d}{'ab'[p]}", "card_id": f"card-{c:02d}", "name": f"Card {c:02d}",
                         "type": kind})
    return rows


def recognizer(rows) -> pl.Recognizer:
    gallery = Pyramid({120: np.zeros((len(rows), 4), np.float32)})
    return pl.Recognizer(LAYOUT, rows, None, gallery, fps=2.0)


def track_json(tr: pl.Track) -> dict:
    return plain({"id": tr.id, "box": box_json(tr.box), "first": tr.first, "last": tr.last, "hits": tr.hits, "reads": tr.reads,
                  "down": tr.down, "prob": [[c, p] for c, p in tr.prob.items()],
                  "best_row": [[c, s, i] for c, (s, i) in tr.best_row.items()], "last_read": tr.last_read, "named": tr.named,
                  "side": tr.side, "kind": tr.kind, "pinned": tr.pinned, "free_since": tr.free_since, "free_at": tr.free_at, "placed": tr.placed})


def board(rng, rows, rec: pl.Recognizer, t: float, cut: bool) -> None:
    """Random tracks on the table: clusters that overlap (stacks, twins), cards in and out of sight, named and not."""
    cards = sorted({r["card_id"] for r in rows})
    kind_of = {r["card_id"]: r["type"] for r in rows}
    n = int(rng.integers(6, 14))
    anchors = [(rng.uniform(420, 1500), rng.uniform(120, 960)) for _ in range(max(2, n // 3))]
    for k in range(n):
        ax, ay = anchors[int(rng.integers(0, len(anchors)))]
        jitter = rng.choice([0.0, 8.0, 40.0, 90.0])
        long = rng.uniform(140, 165)
        b = det_box(ax + rng.normal(0, jitter + 1), ay + rng.normal(0, jitter + 1), long, long * rng.uniform(0.68, 0.74),
                    rng.choice([rng.uniform(85, 95), rng.uniform(0, 5), rng.uniform(0, 180)]),
                    back=bool(rng.random() < 0.1) if rng.random() < 0.8 else None)
        first = round(float(rng.uniform(0, t)) * 2) / 2
        last = float(rng.choice([t, t, t, t - 0.5, t - 1.5, t - 5.0, t - 30.0, t - 70.0]))
        last = max(first, last)
        tr = pl.Track(f"t{k}", b, first, last, side=LAYOUT.side(*b.centre, W, H))
        tr.placed = True  # on the table: seen out of a hand
        tr.hits = int(rng.choice([1, 2, 3, 8, 20]))
        tr.reads = int(rng.choice([0, 1, 2, 3, 4, 6, 12]))
        tr.down = int(rng.choice([0, 0, 0, 1, 2, 3]))
        tr.last_read = float(rng.choice([-1e9, t, t - 4.0, t - 9.0]))
        if tr.reads:
            picks = list(rng.choice(cards, size=int(rng.integers(1, 6)), replace=False))
            weights = rng.dirichlet(np.ones(len(picks)) * rng.choice([0.2, 1.0, 5.0]))
            for c, w in zip(picks, weights):
                tr.prob[str(c)] = float(w) * tr.reads
                printings = [j for j, r in enumerate(rows) if r["card_id"] == c]
                i = printings[int(rng.integers(0, len(printings)))]
                if rng.random() < 0.8:
                    tr.best_row[str(c)] = (float(rng.uniform(0.3, 0.9)), i)
            if rng.random() < 0.6:
                top = max(tr.prob, key=tr.prob.get)
                tr.named = str(top) if rng.random() < 0.85 else str(picks[-1])
                tr.kind = kind_of[tr.named]
                tr.pinned = tr.kind in pl.STATIC and rng.random() < 0.7
        rec.tracks[tr.id] = tr
    rec.next_id = n
    rec.t0 = 0.0
    for tr in rec.tracks.values():
        if tr.pinned and tr.kind == "Legend" and tr.side not in rec.legends and rng.random() < 0.7:
            r = rows[rec.first_row[tr.named]]
            rec.legends[tr.side] = {"printing_id": r["printing_id"], "name": r["name"]}
    for _ in range(int(rng.integers(0, 4))):
        c = str(rng.choice(cards))
        r = rows[rec.first_row[c]]
        rec.ghosts.append({"t": float(t - rng.uniform(1, 70)), "card": c, "x": float(rng.uniform(400, 1500)),
                           "y": float(rng.uniform(100, 1000)), "name": r["name"], "printing_id": r["printing_id"],
                           "side": "left" if rng.random() < 0.5 else "right", "id": f"t{100 + len(rec.ghosts)}"})
    for _ in range(int(rng.integers(0, 3))):
        c = str(rng.choice(cards))
        rec.plays.append((float(t - rng.uniform(0, 20)), c, float(rng.uniform(400, 1500)), float(rng.uniform(100, 1000))))
    if cut:
        rec.cut_at = float(t - rng.choice([0.5, 1.0, 3.0]))
        rec.anchor_base = {k: det_box(tr.box.centre[0] - 40, tr.box.centre[1] + 25, tr.box.long_px, tr.box.short_px, tr.box.angle_deg)
                           for k, tr in rec.tracks.items() if tr.first < rec.cut_at}
    rec.boxes_now = {k: pl.aabb(tr.box) for k, tr in rec.tracks.items()}


def setup_json(rec: pl.Recognizer) -> dict:
    return plain({"tracks": [track_json(tr) for tr in rec.tracks.values()], "next_id": rec.next_id, "t0": rec.t0,
                  "legends": rec.legends, "ghosts": rec.ghosts, "plays": [list(p) for p in rec.plays], "cut_at": rec.cut_at,
                  "anchor_base": {k: box_json(b) for k, b in rec.anchor_base.items()},
                  "boxes_now": {k: list(v) for k, v in rec.boxes_now.items()}})


def after_json(rec: pl.Recognizer) -> dict:
    """What announce and reanchor change: the tracks (their probabilities as a count and a sum), the legends, the
    ghosts, the plays, the anchors."""
    tracks = []
    for tr in rec.tracks.values():
        d = track_json(tr)
        prob = d.pop("prob")
        d.pop("best_row")
        d["prob_n"], d["prob_sum"] = len(prob), sum(p for _, p in prob)
        tracks.append(d)
    return plain({"tracks": tracks, "legends": rec.legends, "ghosts": rec.ghosts, "plays": [list(p) for p in rec.plays],
                  "anchor_pairs": [[list(a), list(b)] for a, b in rec.anchor_pairs]})


def moved(rows, rec: pl.Recognizer, t: float) -> None:
    """A card that vanished and is read again a card's width away (moved), one read again nearby (the same track
    again), and one that went out of sight long ago (a ghost now)."""
    def add(k, cx, cy, first, last, named=None, reads=0, prob=None):
        tr = pl.Track(f"t{k}", det_box(cx, cy, 152.0, 108.0, 90.0), first, last, side=LAYOUT.side(cx, cy, W, H))
        tr.placed = True  # on the table: seen out of a hand
        tr.hits, tr.reads, tr.named = 12, reads, named
        tr.kind = rows[rec.first_row[named]]["type"] if named else ""
        for c, p in (prob or {}).items():
            tr.prob[c] = p
            tr.best_row[c] = (0.8, rec.first_row[c])
        rec.tracks[tr.id] = tr
    add(0, 700.0, 400.0, 2.0, t - 2.0, "card-01", 3, {"card-01": 2.9})
    add(1, 900.0, 400.0, t - 2.2, t, None, 2, {"card-01": 1.9, "card-02": 0.1})
    add(2, 1200.0, 700.0, 2.0, t - 1.5, "card-02", 3, {"card-02": 2.8})
    add(3, 1230.0, 690.0, t - 1.4, t, None, 2, {"card-02": 1.95})
    add(4, 500.0, 900.0, 1.0, t - 61.0, "card-04", 2, {"card-04": 1.9})
    rec.next_id, rec.t0 = 5, 0.0
    rec.boxes_now = {k: pl.aabb(tr.box) for k, tr in rec.tracks.items()}


def pins(rows, rec: pl.Recognizer, t: float) -> None:
    """Pinned battlefields: one misread once (a rune turned sideways) and read as the rune since; one out of sight,
    read again elsewhere (moved); one in sight, outlined again over itself; and a card held past the window's edge."""
    def add(k, cx, cy, first, last, named=None, reads=0, prob=None, pinned=False):
        tr = pl.Track(f"t{k}", det_box(cx, cy, 152.0, 108.0, 90.0), first, last, side=LAYOUT.side(cx, cy, W, H))
        tr.placed = True  # on the table: seen out of a hand
        tr.hits, tr.reads, tr.named, tr.pinned = 12, reads, named, pinned
        tr.kind = rows[rec.first_row[named]]["type"] if named else ""
        for c, p in (prob or {}).items():
            tr.prob[c] = p
            tr.best_row[c] = (0.8, rec.first_row[c])
        rec.tracks[tr.id] = tr
    add(0, 700.0, 400.0, 2.0, t, "card-06", 6, {"card-06": 0.6, "card-07": 5.2}, pinned=True)
    add(1, 800.0, 800.0, 2.0, t - 3.0, "card-14", 8, {"card-14": 7.8}, pinned=True)
    add(2, 1100.0, 600.0, t - 2.0, t, None, 3, {"card-14": 2.9})
    add(3, 1400.0, 300.0, 2.0, t, "card-06", 8, {"card-06": 7.9}, pinned=True)
    add(4, 1430.0, 320.0, t - 2.0, t, None, 3, {"card-06": 2.95})
    add(5, 300.0, 500.0, t - 1.0, t, None, 2, {"card-01": 1.9})
    rec.next_id, rec.t0 = 6, 0.0
    rec.boxes_now = {k: pl.aabb(tr.box) for k, tr in rec.tracks.items()}


def boards(rng) -> list[dict]:
    rows = catalogue()
    out = []
    for n in range(12):
        t = float(rng.choice([20.0, 45.5, 90.0]))
        rec = recognizer(rows)
        if n == 10:
            t = 80.0
            moved(rows, rec, t)
        elif n == 11:
            t = 80.0
            pins(rows, rec, t)
        else:
            board(rng, rows, rec, t, cut=n % 4 == 3)
        setup = setup_json(rec)
        per = []
        for tr in list(rec.tracks.values()):
            so = rec.stacked_on(t, tr)
            va = rec.vanished(t, tr) if tr.named else None
            per.append(plain({"id": tr.id, "label": list(rec.label(tr)), "due": rec.due(tr, t), "covered": rec.covered(t, tr),
                              "stacked_on": so.id if so else None, "twin": rec.twin(t, tr), "vanished": va.id if va else None,
                              "face_down": rec.face_down(tr), "on_legend": rec.on_legend(tr.box),
                              "side_legend": (lambda o: o.id if o else None)(rec.side_legend(tr)),
                              "misread_battlefield": rec.misread_battlefield(tr) if tr.pinned and tr.kind == "Battlefield" else False,
                              "pinned_twin": (lambda o: o.id if o else None)(rec.pinned_twin(t, tr)) if tr.named else None,
                              "held": rec.held(tr.box, W, H), "in_strip": rec.in_strip(*tr.box.centre)}))
        stacks = {k: [u.id for u in v] for k, v in rec.stacks(t).items()}
        state = plain(rec.state(t, W, H))
        asks = [(str(rng.choice(sorted({r["card_id"] for r in rows}))), float(rng.uniform(400, 1500)), float(rng.uniform(100, 1000)))
                for _ in range(3)]
        played = [[c, x, y, rec.recently_played(t, c, x, y)] for c, x, y in asks]
        events = plain(rec.announce(t))
        after = after_json(rec)
        state2 = plain(rec.state(t, W, H))
        x0, y0 = float(rng.uniform(400, 1300)), float(rng.uniform(100, 800))
        left_box = (x0, y0, x0 + float(rng.uniform(100, 400)), y0 + float(rng.uniform(100, 300)))
        left = plain(rec.left(t + 6.0, left_box))
        out.append({"t": t, "setup": setup, "per_track": per, "stacks": stacks, "state": state,
                    "recently_played": played,
                    "announce": events, "after": after, "state_after": state2, "left_box": list(left_box), "left": left,
                    "after_left": {"ids": list(rec.tracks), "ghosts": plain(rec.ghosts)}})
    return out


def reanchor_vectors(rng) -> list[dict]:
    """A cut: tracks from before it, found again by name after it in a view moved by a known similarity."""
    rows = catalogue()
    out = []
    for n in range(4):
        rec = recognizer(rows)
        rec.cut_at = 50.0
        th = rng.uniform(-0.1, 0.1)
        s = rng.uniform(0.9, 1.1)
        shift = rng.uniform(-60, 60, 2)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        cards = sorted({r["card_id"] for r in rows if r["type"] in ("Unit", "Gear", "Spell")})
        for k in range(8):
            b = det_box(rng.uniform(500, 1400), rng.uniform(150, 950), 150.0, 107.0, rng.uniform(85, 95))
            tr = pl.Track(f"t{k}", b, 10.0, 49.5, side=LAYOUT.side(*b.centre, W, H))
            tr.placed = True  # on the table: seen out of a hand
            tr.hits, tr.reads = 20, 3
            if k < 6:
                tr.named = cards[k]
                tr.kind = rows[rec.first_row[tr.named]]["type"]
                tr.prob[tr.named] = 2.7
            rec.tracks[tr.id] = tr
            rec.anchor_base[tr.id] = b
        for k in range(8, 12):  # new tracks after the cut, some where old ones moved to
            o = rec.tracks[f"t{k - 8 + 2}"]
            c = s * rot @ np.asarray(o.box.centre) + shift
            b = det_box(c[0] + rng.normal(0, 3), c[1] + rng.normal(0, 3), 150.0 * s, 107.0 * s, o.box.angle_deg)
            tr = pl.Track(f"t{k}", b, 50.5, 51.0, side=LAYOUT.side(*b.centre, W, H))
            tr.placed = True  # on the table: seen out of a hand
            tr.hits = 3
            if k == 11:
                tr.reads, tr.prob[cards[7]] = 1, 0.9
            rec.tracks[tr.id] = tr
        rec.next_id = 12
        for k in (0, 1):  # two cards found again by name: the anchors
            o = rec.tracks[f"t{k}"]
            c = s * rot @ np.asarray(o.box.centre) + shift
            rec.anchor_pairs.append((o.box.centre, (float(c[0] + rng.normal(0, 1)), float(c[1] + rng.normal(0, 1)))))
        setup = setup_json(rec)
        setup["anchor_pairs"] = plain([[list(a), list(b)] for a, b in rec.anchor_pairs])
        rec.reanchor()
        out.append({"setup": setup, "after": after_json(rec)})
    return out


# --- the bootstrap finder ------------------------------------------------------------------------------------------

def mat_frame(cards: list[dict]) -> np.ndarray:
    """A 1920 x 1080 frame drawn by formula (the test draws the same): la-rq's mat with a faint print, and cards as
    rotated rectangles of one colour. A pixel is a card's when its centre lies inside it."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.empty((H, W, 3), np.uint8)
    for c in range(3):
        img[..., c] = LAYOUT.mat[c] + (x * 3 + y * 7 + c) % 5
    px, py = x + 0.5, y + 0.5
    for cd in cards:
        (cx, cy), co, si = cd["centre"], cd["cos"], cd["sin"]
        u = (px - cx) * co + (py - cy) * si
        v = -(px - cx) * si + (py - cy) * co
        img[(np.abs(u) <= cd["long"] / 2) & (np.abs(v) <= cd["short"] / 2)] = cd["rgb"]
    return img


def bootstrap_vectors() -> list[dict]:
    """Recognizer.find with no finder (matcrops.find_cards on the table window, the mat told apart by colour): two
    frames, the second 40 s on, when the mat's colour is measured again."""
    rec = recognizer(catalogue())
    out = []
    for t, placed in ((0.0, [(600, 300, 0), (900, 330, 90), (1200, 600, 30), (500, 800, 75), (1400, 250, 10)]),
                      (40.0, [(640, 320, 5), (1000, 700, 120), (1300, 400, 88), (800, 950, 45)])):
        cards = []
        for k, (cx, cy, a) in enumerate(placed):
            long = 150.0 + 3 * k
            cards.append({"centre": [float(cx), float(cy)], "long": long, "short": round(long / 1.4, 2),
                          "cos": math.cos(math.radians(a)), "sin": math.sin(math.radians(a)),
                          "rgb": [(200, 170, 120), (180, 60, 60), (230, 230, 230), (90, 160, 200), (220, 200, 60)][k]})
        boxes = rec.find(t, mat_frame(cards))
        out.append({"t": t, "cards": cards, "mat": [int(v) for v in rec.mat], "boxes": [box_json(b) for b in boxes]})
    return out


def shot_thumb(k: int, shot: str) -> np.ndarray:
    """A 96 x 54 thumbnail of one shot, drawn by integer formulas (the test draws the same): the table camera (thumb's
    pattern), or another shot (a pattern of its own per shot); in every shot the same corner, bottom right, as a
    co-streamer's webcam stays put through the cuts. The co-streamer in it moves at the cut at k = 30, and their
    face reaches the frame's bottom edge."""
    y, x, c = np.meshgrid(np.arange(54), np.arange(96), np.arange(3), indexing="ij")
    if shot == "table":
        v = (x * 7 + y * 13 + c * 50) % 200 + 20 + (x * 31 + y * 17 + c * 5 + k * 11) % 7 - 3
    else:
        s = 1 if shot == "cam" else 2
        v = (x * (3 + s) + y * (29 - s) + c * 71 + k * 13) % 256
    v = np.where((x >= 80) & (y >= 40), 100 + c * 40, v)
    v = np.where((x >= 85) & (x <= 90) & (y >= 44), 100 + c * 40 + (60 if k >= 30 else 0), v)
    return v.astype(np.float32)


OVERLAY_SHOTS = (["table"] * 10 + ["cam"] * 4 + ["table"] * 6 + ["wide"] * 4 + ["table"] * 6 + ["cam"] * 4 + ["table"] * 6)


def overlay_vectors() -> dict:
    """Scene.see over 40 thumbnails, the table camera with cuts to two other shots: per step the cuts seen, how often
    the overlay was worked out, and its size and whether it holds the co-streamer's face (which moves at the fifth cut:
    the overlay keeps it); then the overlay, and where it is in frame points."""
    sc = pl.Scene(LAYOUT)
    steps = []
    for k, shot in enumerate(OVERLAY_SHOTS):
        sc.see(shot_thumb(k, shot))
        steps.append({"k": k, "shot": shot, "cuts": sc.cuts, "overlay_n": sc.overlay_n,
                      "suspect": None if sc.suspect is None else int(sc.suspect.sum()), "corner": sc.in_suspect(1800, 950, W, H),
                      "overlay": None if sc.overlay is None else int(sc.overlay.sum()), "face": sc.in_overlay(1760, 980, W, H)})
    ov = sc.overlay
    points = [(1800, 950), (1700, 820), (960, 540), (100, 100), (1919, 1079), (1590, 790)]
    return {"steps": steps, "same_sum": int(sc.same.sum()), "overlay": [int(i) for i in np.flatnonzero(ov)] if ov is not None else None,
            "points": [{"x": px, "y": py, "in": sc.in_overlay(px, py, W, H)} for px, py in points]}


def still_frame(k: int) -> np.ndarray:
    """A 960 x 540 frame drawn by integer formulas (the test draws the same): a table that slowly changes with k, and a
    hand that crosses it, there in frames 2 and 6 only."""
    y, x, c = np.mgrid[0:540, 0:960, 0:3]
    v = (x * 5 + y * 3 + c * 40 + (k * 7) * (x // 120)) % 256
    if k in (2, 6):
        v = np.where((x >= 300) & (x < 420) & (y >= 200) & (y < 300), 230 * (c == 0) + 185 * (c == 1) + 160 * (c == 2), v)
    return v.astype(np.uint8)


def still_vectors() -> dict:
    """StillTable over 16 frames at 0.4 s: the frames HAND_BG_EVERY apart are taken (every other one); the first five
    make the median, without the hand in two of them; the rest step it nearer."""
    st = pl.StillTable()
    out = []
    for k in range(16):
        t = 0.4 * k
        st.feed(t, pl.StillTable.small(still_frame(k)))
        out.append({"t": t, "last": st.last, "first": len(st.first), "bg": st.bg is not None,
                    "bg_sum": None if st.bg is None else int(st.bg.astype(np.int64).sum()),
                    "bg_head": None if st.bg is None else [int(v) for v in st.bg.ravel()[:48]]})
    return {"steps": out}


def wood_frames() -> tuple[np.ndarray, np.ndarray]:
    """A wooden table, skin-coloured (170, 120, 100), 960 x 540, and the same with fingers (230, 185, 160) along the left
    edge of the card at (628, 239)."""
    wood = np.empty((540, 960, 3), np.uint8)
    wood[...] = (170, 120, 100)
    held = wood.copy()
    held[200:278, 576:600] = (230, 185, 160)
    return wood, held


def hand_vectors() -> list[dict]:
    """hand_share on a wooden table: the colour test alone, then against the still table (`still`, `now`)."""
    wood, held = wood_frames()
    sw, sh = pl.StillTable.small(wood), pl.StillTable.small(held)
    table = (0, 0, 960, 540)
    out = []
    for name, b in (("upright", det_box(628, 239, 78, 56, 90.0)), ("tilted", det_box(628, 239, 78, 56, 63.0)),
                    ("across", det_box(610, 250, 78, 56, 0.0))):
        out.append({"box": box_json(b),
                    "alone": pl.hand_share(wood, b, table), "held_alone": pl.hand_share(held, b, table),
                    "still": pl.hand_share(wood, b, table, still=sw, now=sw), "held": pl.hand_share(held, b, table, still=sw, now=sh)})
    return out


def plain_frame() -> np.ndarray:
    """The mat (30, 40, 55), 960 x 540, with a card-sized zone printed on it (a 2 px outline, (150, 130, 70), around
    (600, 200) to (656, 278)), a card's face (a busy patch) at (350, 200) and a face-down card in a pink sleeve at
    (200, 200), each 56 x 78."""
    y, x, c = np.mgrid[0:540, 0:960, 0:3]
    im = np.empty((540, 960, 3), np.uint8)
    im[...] = (30, 40, 55)
    im[200:279, 600:657] = (150, 130, 70)
    im[202:277, 602:655] = (30, 40, 55)
    face = ((x * 37 + y * 11 + c * 90) % 256).astype(np.uint8)
    im[200:278, 350:406] = face[200:278, 350:406]
    im[200:278, 200:256] = (220, 90, 150)
    return im


def plain_vectors() -> list[dict]:
    im = plain_frame()
    out = []
    for name, b in (("zone", det_box(628.0, 239.0, 78, 56, 90.0)), ("face", det_box(378.0, 239.0, 78, 56, 90.0)),
                    ("sleeve", det_box(228.0, 239.0, 78, 56, 90.0)), ("zone turned", det_box(628.0, 239.0, 78, 56, 87.0)),
                    ("mat", det_box(800.0, 400.0, 78, 56, 0.0)), ("edge", det_box(20.0, 20.0, 78, 56, 0.0))):
        out.append({"name": name, "box": box_json(b), "plain": pl.plain_zone(im, b)})
    return out


def main() -> int:
    rng = np.random.default_rng(20260929)
    vectors = {
        "note": "computed by packages/engine/test/gen/recognizer.py from live/pipeline.py; synthetic inputs only",
        "layout": "la-rq", "frame": [W, H], "rows": catalogue(),
        "geometry": geometry(rng),
        "scene": scene_vectors(),
        "on_table": scene_on_table(6), "on_table_nocards": scene_on_table(0), "on_table_few": scene_on_table(6, 20),
    }
    brng = np.random.default_rng(7)
    vectors["boards"] = boards(brng)
    vectors["reanchor"] = reanchor_vectors(np.random.default_rng(11))
    vectors["bootstrap"] = bootstrap_vectors()
    vectors["overlay"] = overlay_vectors()
    vectors["still"] = still_vectors()
    vectors["hands"] = hand_vectors()
    vectors["plain"] = plain_vectors()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(vectors, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
