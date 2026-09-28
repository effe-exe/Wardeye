# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The live recogniser: cards found, tracked and named frame by frame, as a stream plays.

Each frame: the bootstrap finder (`matcrops.find_cards`, isolated cards on the mat) or, once it is
trained, the detector gives card boxes inside the layout's table window. Boxes are matched to the
tracks of cards already on the table by position and size, so a card is identified once, not once
per frame: new tracks and uncertain ones get a crop identified (colour + structure against the
gallery pyramid, the M0 identifier, D-019), and named tracks are re-checked now and then. A track's
name is its readings' mean probability (softmax over each crop's card scores, M0's temperature).
Plain crops are face-down cards: they are never identified (D-005). Only the table window is ever
looked at; the broadcast's panels with the players' hands are hidden information and never read.

A card named on the table becomes a "played" event; a named card gone for a while, "left". The finder
only sees cards lying on their own, so the change gate watches the whole table too: when a region
settles after a change (a card put on a stack, a rune channelled), the region is read like a crop, and
a confident read is a play even though no track holds the card (as in `rifteye_ml.demo`).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from ..changegate import ChangeGate, GateSettings
from ..matcrops import FACE_DOWN_DETAIL, CardBox, detail, find_cards, mat_colour, notmat_mask
from ..retrieval import Pyramid, ROTATIONS
from .layouts import Layout

TEMPERATURE = 0.0212  # fitted on the M0 real labels (reviewpack identity)


@dataclass
class Track:
    id: str
    box: CardBox
    first: float
    last: float
    hits: int = 1
    reads: int = 0                                   # identifications of this card so far
    down: int = 0                                    # face-down looks in a row
    prob: dict[str, float] = field(default_factory=dict)       # card_id -> summed probability
    best_row: dict[str, tuple[float, int]] = field(default_factory=dict)  # card_id -> (score, gallery row)
    last_read: float = -1e9
    named: str | None = None                         # card_id once the track has been announced
    side: str = ""

    def top(self) -> list[tuple[str, float]]:
        if not self.reads:
            return []
        return sorted(((c, p / self.reads) for c, p in self.prob.items()), key=lambda kv: -kv[1])


def quad(box: CardBox) -> list[list[float]]:
    """The rotated rectangle's corners in frame px."""
    cx, cy = box.centre
    a = math.radians(box.angle_deg)
    ux, uy = math.cos(a) * box.long_px / 2, math.sin(a) * box.long_px / 2
    vx, vy = -math.sin(a) * box.short_px / 2, math.cos(a) * box.short_px / 2
    return [[round(cx + ux + vx, 1), round(cy + uy + vy, 1)], [round(cx + ux - vx, 1), round(cy + uy - vy, 1)],
            [round(cx - ux - vx, 1), round(cy - uy - vy, 1)], [round(cx - ux + vx, 1), round(cy - uy + vy, 1)]]


def card_crop(frame: Image.Image, box: CardBox) -> Image.Image:
    """The card upright, long side vertical. Only the card's neighbourhood is rotated, not the frame."""
    cx, cy = box.centre
    r = math.ceil(math.hypot(box.long_px, box.short_px) / 2) + 2
    x0, y0 = int(cx) - r, int(cy) - r
    local = frame.crop((x0, y0, x0 + 2 * r, y0 + 2 * r))
    lx, ly = cx - x0, cy - y0
    rot = local.rotate(box.angle_deg - 90, resample=Image.BICUBIC, center=(lx, ly))
    w, h = box.short_px, box.long_px
    return rot.crop((round(lx - w / 2), round(ly - h / 2), round(lx + w / 2), round(ly + h / 2)))


def detector_boxes(dets: Sequence[dict], min_score: float = 0.4) -> list[CardBox]:
    """The trained detector's cards (`detect.model.Detector.detect`: corners in frame px) as boxes for the
    tracker. A `card_back` is marked `back`, so it is never identified."""
    out = []
    for d in dets:
        if d["score"] < min_score:
            continue
        q = np.asarray(d["quad"], np.float64).reshape(4, 2)
        e = [q[(k + 1) % 4] - q[k] for k in range(4)]
        a, b = (np.linalg.norm(e[0]) + np.linalg.norm(e[2])) / 2, (np.linalg.norm(e[1]) + np.linalg.norm(e[3])) / 2
        long_e = e[0] if a >= b else e[1]
        box = CardBox(tuple(q.mean(axis=0)), max(a, b), min(a, b), math.degrees(math.atan2(long_e[1], long_e[0])) % 180, 1.0)
        box.back = d["cls"] == "card_back"  # type: ignore[attr-defined]
        out.append(box)
    return out


class Recognizer:
    """Holds the gallery and the table's tracks; `step` takes one frame and returns the state and events.
    `finder(t, image)` replaces the bootstrap finder, e.g. with the trained detector (`detector_boxes`)."""

    def __init__(self, layout: Layout, rows: Sequence[dict], encoder, gallery: Pyramid, title: str = "",
                 min_p: float = 0.5, sure_p: float = 0.85, recheck_s: float = 8.0, forget_s: float = 4.0,
                 max_reads: int = 12, settle_s: float = 3.0, gate: bool = True, fps: float = 5.0, gate_p: float = 0.7,
                 finder=None):
        self.layout, self.rows, self.enc, self.gallery = layout, list(rows), encoder, gallery
        self.cards = np.array([r["card_id"] for r in self.rows])
        self.first_row: dict[str, int] = {}
        for i, r in enumerate(self.rows):
            self.first_row.setdefault(r["card_id"], i)
        self.row_of = {r["printing_id"]: r for r in self.rows}
        self.title = title or layout.title
        self.min_p, self.sure_p, self.recheck_s, self.forget_s = min_p, sure_p, recheck_s, forget_s
        self.max_reads, self.settle_s = max_reads, settle_s
        self.tracks: dict[str, Track] = {}
        self.next_id = 0
        self.t0: float | None = None
        self.mat: np.ndarray | None = None
        self.mat_t = -1e9
        self.timing: dict[str, float] = {}
        self.gate = ChangeGate(GateSettings(fps=fps, card_long_frac=layout.card_long_1080 / 1080)) if gate else None
        self.gate_p = gate_p
        self.finder = finder
        self.pending: list[tuple[float, tuple[float, float, float, float]]] = []  # (read at, frame box)
        self.flashes: list[dict] = []   # plays the gate found, drawn for a few seconds
        self.plays: list[tuple[float, str, float, float]] = []  # (t, card, x, y) of recent plays, for de-duplication
        # Named cards the finder lost (a hand over them, a card touching them): remembered for a while, so
        # the same card found again at the same spot is not played twice. A card leaves the table when
        # the gate sees its spot change, not when the finder loses it.
        self.ghosts: list[dict] = []
        # A legend never changes during a game (M0 §5.4): once one is named on a side, that player keeps it.
        # ponytail: the first confident legend wins for the whole run; reset per game once games are detected
        self.legends: dict[str, dict] = {}

    # --- finding ---------------------------------------------------------------

    def find(self, t: float, rgb: np.ndarray) -> list[CardBox]:
        if self.finder is not None:
            return self.finder(t, rgb)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = self.layout.box(w, h)
        roi = rgb[y0:y1, x0:x1]
        mask = None
        if self.layout.mask == "notmat":
            if self.mat is None or t - self.mat_t > 30:  # the light drifts over a match
                self.mat, self.mat_t = mat_colour(roi[::4, ::4]), t
            mask = notmat_mask(roi, self.mat, self.layout.mat_tol)
        boxes = find_cards(roi, self.layout.card_px(h), mask=mask)
        for b in boxes:
            b.centre = (b.centre[0] + x0, b.centre[1] + y0)
        return boxes

    # --- tracking --------------------------------------------------------------

    def match(self, t: float, boxes: Sequence[CardBox], w: int, h: int) -> list[Track]:
        """Each box continues the nearest live track of about its size, or starts a new one."""
        # ponytail: greedy nearest-box matching; Hungarian assignment if tracks swap in the detector's dense stacks
        free = {k: tr for k, tr in self.tracks.items()}
        seen = []
        for b in sorted(boxes, key=lambda b: -b.long_px):
            best, best_d = None, 0.3 * b.long_px
            for k, tr in free.items():
                d = math.dist(tr.box.centre, b.centre)
                if d < best_d and abs(tr.box.long_px / b.long_px - 1) < 0.2:
                    best, best_d = k, d
            if best is None:
                tr = Track(f"t{self.next_id}", b, t, t, side=self.layout.side(*b.centre, w, h))
                self.next_id += 1
                self.tracks[tr.id] = tr
            else:
                tr = free.pop(best)
                tr.box, tr.last, tr.hits = b, t, tr.hits + 1
                tr.side = self.layout.side(*b.centre, w, h)
            seen.append(tr)
        return seen

    # --- naming ----------------------------------------------------------------

    def face_down(self, tr: Track) -> bool:
        """Two face-down looks in a row on a card never named. A hand resting on a named card or a
        blurred frame does not hide it, and a card played under a hand is looked at again later."""
        return tr.named is None and tr.down >= 2

    def due(self, tr: Track, t: float) -> bool:
        if self.face_down(tr):
            return t - tr.last_read > self.recheck_s  # looked at now and then, never identified
        top = tr.top()
        if not top or top[0][1] < self.sure_p and tr.reads < self.max_reads:
            return True
        return t - tr.last_read > self.recheck_s

    def identify(self, crops: Sequence[Image.Image]) -> list[list[tuple[str, float, float, int]]]:
        """Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first.
        All four turns go in one batch: which way up a card lies is unknown (exhausted, opponent side)."""
        if not crops:
            return []
        views = [c.rotate(r, expand=True) if r else c for c in crops for r in ROTATIONS]
        emb = self.enc.embed(views).reshape(len(crops), len(ROTATIONS), -1)
        out = []
        for n, c in enumerate(crops):
            level = self.gallery.levels[self.gallery.level_for(max(c.size))]
            sims = (emb[n] @ level.T).max(axis=0)
            scores: dict[str, tuple[float, int]] = {}
            for i in np.argsort(-sims)[:60]:
                card = self.cards[i]
                if card not in scores:
                    scores[card] = (float(sims[i]), int(i))
            vals = np.array([v[0] for v in scores.values()])
            p = np.exp((vals - vals.max()) / TEMPERATURE)
            p /= p.sum()
            out.append([(card, float(pc), sc, i) for (card, (sc, i)), pc in zip(scores.items(), p)])
        return out

    def read(self, t: float, frame: Image.Image, todo: Sequence[Track]) -> None:
        crops, owners = [], []
        for tr in todo:
            tr.last_read = t
            if getattr(tr.box, "back", False):
                tr.down += 1  # the detector saw a card back
                continue
            c = card_crop(frame, tr.box)
            if detail(c) < FACE_DOWN_DETAIL:
                tr.down += 1
                continue
            tr.down = 0
            crops.append(c)
            owners.append(tr)
        for tr, cands in zip(owners, self.identify(crops)):
            tr.reads += 1
            for card, pc, sc, i in cands:
                tr.prob[card] = tr.prob.get(card, 0.0) + pc
                if sc > tr.best_row.get(card, (-1.0, 0))[0]:
                    tr.best_row[card] = (sc, i)

    def watch(self, t: float, image: np.ndarray, frame: Image.Image) -> list[dict]:
        """The change gate on the whole table; a settled change is read a moment later, like a crop."""
        from ..reviewpack import view_to_frame

        if self.gate is None:
            return []
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.layout.box(w, h)
        tx0, ty0, tx1, ty1 = self.layout.table
        vw = self.gate.s.width
        vh = round(vw * (ty1 - ty0) * 1080 / ((tx1 - tx0) * 1920) / 2) * 2
        view = np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize((vw, vh), Image.BILINEAR))
        events = []
        for ev in self.gate.feed(t, view):
            fb = view_to_frame(ev.box, self.layout.table, vw, w, h)
            if ev.kind in ("appeared", "changed"):
                self.pending.append((t + 0.4, fb))
            elif ev.kind == "disappeared":
                events += self.left(t, fb)
        due = [b for when, b in self.pending if when <= t]
        self.pending = [(when, b) for when, b in self.pending if when > t]
        card_long = self.layout.card_px(h)
        for bx0, by0, bx1, by1 in due:
            gx, gy = (bx1 - bx0) * 0.08, (by1 - by0) * 0.08
            box = (max(0, bx0 - gx), max(0, by0 - gy), min(w, bx1 + gx), min(h, by1 + gy))
            region = frame.crop(tuple(round(v) for v in box))
            if min(region.size) < 0.45 * card_long * 63 / 88 or max(region.size) > 2.2 * card_long:
                continue  # a die or counter, or a whole area at once: not one card
            if region.width > region.height * 1.15:
                region = region.rotate(90, expand=True)  # cards stand portrait
            if detail(region) < FACE_DOWN_DETAIL:
                continue  # a face-down card: never identified
            # ponytail: the whole changed region is read as one card; the visible-band matcher (stacks.py) for cards put on stacks
            cands = self.identify([region])[0]
            card, p, sc, i = cands[0]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if p < self.gate_p or self.recently_played(t, card, cx, cy):
                continue
            r = self.rows[i]
            side = self.layout.side(cx, cy, w, h)
            self.plays.append((t, card, cx, cy))
            q = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            guesses = [{"printing_id": self.rows[j]["printing_id"], "card_id": c, "name": self.rows[j]["name"], "p": round(pc, 3)}
                       for c, pc, _, j in cands[:3]]
            self.flashes.append({"id": f"g{len(self.plays)}", "quad": [[round(x, 1), round(y, 1)] for x, y in q],
                                 "side": side, "state": "named", "printing_id": r["printing_id"], "name": r["name"],
                                 "confidence": round(p, 3), "guesses": guesses, "since": round(t, 2), "until": t + 5})
            events.append({"t": round(t, 2), "kind": "played", "text": f"{r['name']} played", "printing_id": r["printing_id"],
                           "track": self.flashes[-1]["id"], "side": side})
        return events

    def left(self, t: float, box: tuple[float, float, float, float]) -> list[dict]:
        """A named card whose spot the gate saw cleared (a ghost, or a track the finder still holds)."""
        x0, y0, x1, y1 = box
        gx, gy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        inside = lambda x, y: x0 - gx <= x <= x1 + gx and y0 - gy <= y <= y1 + gy  # noqa: E731
        for gh in sorted(self.ghosts, key=lambda gh: -gh["t"]):
            if inside(gh["x"], gh["y"]):
                self.ghosts.remove(gh)
                return [{"t": round(t, 2), "kind": "left", "text": f"{gh['name']} left the table", "printing_id": gh["printing_id"],
                         "track": gh["id"], "side": gh["side"]}]
        return []

    def recently_played(self, t: float, card: str, x: float, y: float, window: float = 12.0, skip: str = "") -> bool:
        """The same card announced nearby a moment ago, by the gate or by another track."""
        reach = 1.5 * self.layout.card_long_1080
        self.plays = [pl for pl in self.plays if t - pl[0] < window]
        if any(c == card and math.dist((x, y), (px, py)) < reach for _, c, px, py in self.plays):
            return True
        return any(tr.id != skip and tr.named == card and t - tr.first < window and math.dist((x, y), tr.box.centre) < reach
                   for tr in self.tracks.values())

    # --- one frame -------------------------------------------------------------

    def step(self, t: float, image: np.ndarray, budget: int = 10) -> tuple[dict, list[dict]]:
        tic = time.perf_counter()
        if self.t0 is None:
            self.t0 = t
        h, w = image.shape[:2]
        boxes = self.find(t, image)
        tf = time.perf_counter()
        seen = self.match(t, boxes, w, h)
        frame = Image.fromarray(image)
        # New and uncertain cards first, then the oldest re-checks; a budget keeps each frame in time.
        todo = sorted((tr for tr in seen if self.due(tr, t)), key=lambda tr: (tr.reads > 0, tr.last_read))[:budget]
        self.read(t, frame, todo)
        tr_ = time.perf_counter()
        events = self.announce(t) + self.watch(t, image, frame)
        self.flashes = [f for f in self.flashes if f["until"] > t]
        self.timing = {"find_ms": 1000 * (tf - tic), "read_ms": 1000 * (tr_ - tf), "gate_ms": 1000 * (time.perf_counter() - tr_),
                       "reads": len(todo), "boxes": len(boxes)}
        return self.state(t, w, h), events

    def label(self, tr: Track) -> tuple[str, float, list[dict]]:
        """The track's state, confidence and up to three guesses (each card's best-matching printing)."""
        if self.face_down(tr):
            return "facedown", 0.0, []
        top = tr.top()
        if not top:
            return "new", 0.0, []
        guesses = []
        for c, p in top[:3]:
            r = self.rows[tr.best_row.get(c, (0.0, self.first_row[c]))[1]]
            guesses.append({"printing_id": r["printing_id"], "card_id": c, "name": r["name"], "p": round(p, 3)})
        p0 = top[0][1]
        named = p0 >= self.sure_p or (p0 >= self.min_p and tr.reads >= 2)
        return ("named" if named else "unsure"), p0, guesses

    def announce(self, t: float) -> list[dict]:
        """'played' when a card is first named; a lost named card becomes a ghost (see `ghosts`)."""
        events = []
        self.ghosts = [gh for gh in self.ghosts if t - gh["t"] < 60]
        for tr in list(self.tracks.values()):
            gone = t - tr.last
            if gone > self.forget_s:
                del self.tracks[tr.id]
                if tr.named:
                    g = self.label(tr)[2]
                    self.ghosts.append({"t": tr.last, "card": tr.named, "x": tr.box.centre[0], "y": tr.box.centre[1],
                                        "name": g[0]["name"] if g else tr.named, "printing_id": g[0]["printing_id"] if g else None,
                                        "side": tr.side, "id": tr.id})
                continue
            state, p, g = self.label(tr)
            if state == "named" and tr.named != g[0]["card_id"]:
                changed = tr.named is not None
                tr.named = g[0]["card_id"]
                reach = 1.5 * self.layout.card_long_1080
                back = [gh for gh in self.ghosts if gh["card"] == tr.named and math.dist((gh["x"], gh["y"]), tr.box.centre) < reach]
                if back and not changed:
                    self.ghosts.remove(back[0])  # the same card, found again
                    continue
                if tr.first - (self.t0 or 0.0) < self.settle_s and not changed:
                    continue  # on the table when we tuned in, not played now
                if changed:
                    events.append(self.event(t, "changed", f"{g[0]['name']} (read again)", tr, g[0]["printing_id"]))
                elif not self.recently_played(t, tr.named, *tr.box.centre, skip=tr.id):
                    self.plays.append((t, tr.named, *tr.box.centre))
                    events.append(self.event(t, "played", f"{g[0]['name']} played", tr, g[0]["printing_id"]))
                else:
                    self.plays.append((t, tr.named, *tr.box.centre))
        return events

    def event(self, t: float, kind: str, text: str, tr: Track, pid: str | None) -> dict:
        return {"t": round(t, 2), "kind": kind, "text": text, "printing_id": pid, "track": tr.id, "side": tr.side}

    def state(self, t: float, w: int, h: int) -> dict:
        tracks = []
        for tr in self.tracks.values():
            if t - tr.last > 1.0:
                continue  # not in view right now (a hand over it, or lost); kept, but not drawn
            state, p, g = self.label(tr)
            top = g[0] if g and state == "named" else None
            tracks.append({"id": tr.id, "quad": quad(tr.box), "side": tr.side, "state": state,
                           "printing_id": top["printing_id"] if top else None, "name": top["name"] if top else "",
                           "confidence": round(p, 3), "guesses": g if state != "facedown" else [],
                           "since": round(tr.first, 2)})
        tracks += [{k: v for k, v in f.items() if k != "until"} for f in self.flashes]
        for tr in tracks:
            pid = tr["printing_id"]
            if tr["state"] == "named" and pid and tr["side"] not in self.legends and tr["confidence"] >= self.sure_p:
                r = self.row_of.get(pid)
                if r is not None and r.get("type") == "Legend":
                    self.legends[tr["side"]] = {"printing_id": pid, "name": r["name"]}
        return {"t": round(t, 2), "status": "live", "message": "", "title": self.title,
                "frame": {"width": w, "height": h},
                "players": [{"side": s, "label": f"Player {k + 1}", "legend": self.legends.get(s)}
                            for k, s in enumerate(self.layout.sides())],
                "layout": {"name": self.layout.name, "table": list(self.layout.table)},
                "tracks": tracks}
