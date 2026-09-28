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
from itertools import combinations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from ..changegate import ChangeGate, GateSettings
from ..detect.geometry import overlap_area
from ..matcrops import FACE_DOWN_DETAIL, CardBox, detail, find_cards, mat_colour, notmat_mask
from ..retrieval import Pyramid, ROTATIONS
from .layouts import Layout

TEMPERATURE = 0.0212  # fitted on the M0 real labels (reviewpack identity)
KEEP_S = 60.0   # a named card out of sight this long has gone: not a hand over it, dice, a card on top
MOVE_S = 10.0   # a named card that vanished this recently and is named again elsewhere has moved
STATIC = ("Legend", "Battlefield")  # set up before the game and never moved: pinned where they are named
QUIET = ("Rune",)                   # tracked, but never labelled, listed or announced: not worth watching
KINDS = {"Legend": "legend", "Battlefield": "battlefield", "Rune": "rune"}


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
    kind: str = ""                                   # the named card's type (Unit, Rune, Legend, ...)
    pinned: bool = False                             # a legend or battlefield: kept where it is all game

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


def aabb(box: CardBox) -> tuple[float, float, float, float]:
    q = np.asarray(quad(box))
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def smooth(old: CardBox, new: CardBox, k: float = 0.35) -> CardBox:
    """The new box eased from the old one while the card barely moves: the detector's boxes jitter by a
    few pixels from frame to frame. A card that moves or turns (exhausted) is followed at once."""
    da = (new.angle_deg - old.angle_deg + 90) % 180 - 90
    if math.dist(old.centre, new.centre) > 0.1 * new.long_px or abs(da) > 10:
        return new
    b = CardBox(((1 - k) * old.centre[0] + k * new.centre[0], (1 - k) * old.centre[1] + k * new.centre[1]),
                (1 - k) * old.long_px + k * new.long_px, (1 - k) * old.short_px + k * new.short_px,
                (old.angle_deg + k * da) % 180, new.fill)
    b.back = getattr(new, "back", False)  # type: ignore[attr-defined]
    return b


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
        box.score = d["score"]  # type: ignore[attr-defined]
        box.vis = float(np.min(d.get("visible") or [1.0]))  # type: ignore[attr-defined]  # its least visible corner
        out.append(box)
    return drop_straddlers(drop_nested(out))


def drop_nested(boxes: list[CardBox]) -> list[CardBox]:
    """One card, one box. The detector can outline a card in a magnetic case or toploader two or three
    times (the card, the case's inner and outer edge): boxes on nearly the same centre, turned the same
    way and of nearly the same size are one card, and the smallest, the card itself, stays."""
    keep: list[CardBox] = []
    for b in sorted(boxes, key=lambda b: b.long_px * b.short_px):
        if not any(math.dist(b.centre, k.centre) <= 0.15 * b.long_px and abs((b.angle_deg - k.angle_deg + 90) % 180 - 90) <= 12
                   and b.long_px / k.long_px <= 1.43 for k in keep):
            keep.append(b)
    return keep


def drop_straddlers(boxes: list[CardBox]) -> list[CardBox]:
    """The detector's slips across two neighbouring cards. Such a box lies almost wholly on two cards that
    lie side by side, scores below both, and still claims its four corners visible. A card under a stack
    has covered corners, and the cards of a column overlap each other, so neither is dropped."""
    quads = [np.array(quad(b)) for b in boxes]
    areas = [b.long_px * b.short_px for b in boxes]
    keep = []
    for i, b in enumerate(boxes):
        score = getattr(b, "score", 1.0)
        if getattr(b, "vis", 0.0) >= 0.5:
            near = [j for j in range(len(boxes)) if j != i and getattr(boxes[j], "score", 1.0) > score
                    and math.dist(boxes[j].centre, b.centre) < b.long_px]
            share = {j: overlap_area(quads[i], quads[j]) / areas[i] for j in near}
            on = [j for j in near if share[j] >= 0.25]
            if any(share[j] + share[k] >= 0.75 and overlap_area(quads[j], quads[k]) <= 0.1 * min(areas[j], areas[k])
                   for j, k in combinations(on, 2)):
                continue
        keep.append(b)
    return keep


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The scale, rotation and shift that take the points `src` onto `dst`, least squares (Umeyama)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rot = u @ d @ vt
    scale = float((sv * np.diag(d)).sum() / max(1e-9, (a ** 2).sum() / len(src)))
    return scale, rot, md - scale * rot @ ms


class Scene:
    """Whether a frame shows the table camera, on any broadcast. The other shots (player cams, a wide shot
    of the stage, a title card) show players' hands and the cards they hold: hidden information, never
    processed (D-005), and nothing on the board changes while they are on.

    It learns the table camera from the footage itself: the parts of a 96 x 54 thumbnail that stay put
    while the board changes (the broadcast's overlay, the mat's edges and print) and their colours. A
    frame is the table camera when those parts match; a cut replaces them, play does not (the M0 final:
    table frames score 0.5 to 0.8, other shots about 0). To start, a frame is taken as the table camera
    when the mat fills the table window as it does there (a layout that knows its mat colour) or when
    at least five cards lie in it; if the view stays unrecognised but looks like a table again for a few
    seconds (the camera itself moved), it learns again. The thumbnail is too coarse to show any card."""

    def __init__(self, layout: Layout, corr: float = 0.45, learn_every: float = 2.0, relearn_after: float = 20.0):
        self.layout, self.corr, self.learn_every, self.relearn_after = layout, corr, learn_every, relearn_after
        self.n = 0
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.last_learn = -1e9
        self.away_since: float | None = None
        self.looks = 0  # table-like frames in a row while away (checked every learn_every)
        self.last_look = -1e9

    @staticmethod
    def small(image: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(image).resize((96, 54), Image.BOX), np.float32)

    def table_like(self, image: np.ndarray, count) -> bool:
        if self.layout.mat is not None:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = self.layout.box(w, h)
            a = np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize((160, 120), Image.BOX), np.float32)
            mat = np.asarray(self.layout.mat, np.float32)
            return float((np.abs(a - mat).max(axis=2) < self.layout.mat_tol).mean()) >= self.layout.mat_share
        return count() >= 5

    def score(self, x: np.ndarray) -> float | None:
        """How well the frame's still parts match the table camera's, or None when those parts have no
        pattern to match (a plain mat and no overlay): then the mat share or the cards decide."""
        std = np.sqrt(self.var).max(axis=2)
        still = std < 12 if (std < 12).mean() >= 0.1 else std <= np.quantile(std, 0.3)
        a, b = x[still].ravel(), self.mean[still].ravel()
        if b.std() < 8:
            return None
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def learn(self, t: float, x: np.ndarray) -> None:
        if t - self.last_learn < self.learn_every:
            return
        self.last_learn, self.n = t, self.n + 1
        if self.mean is None:
            self.mean, self.var = x.copy(), np.full_like(x, 400.0)  # unsure at first: nothing counts as still
            return
        k = max(1.0 / self.n, 0.02)  # the mean of the first fifty, then a slow drift (light, overlay updates)
        d = x - self.mean
        self.mean += k * d
        self.var = (1 - k) * (self.var + k * d * d)

    def on_table(self, t: float, image: np.ndarray, count=lambda: 0) -> bool:
        x = self.small(image)
        sc = self.score(x) if self.n >= 5 else None
        if sc is None:
            ok = self.table_like(image, count)
        else:
            ok = sc >= self.corr
            if not ok and self.away_since is not None and t - self.away_since > self.relearn_after \
                    and t - self.last_look >= self.learn_every:
                self.last_look = t
                self.looks = self.looks + 1 if self.table_like(image, count) else 0
                if self.looks >= 3:  # the table again, but not as it was learnt: the camera moved
                    self.n, self.mean, self.var, self.looks, ok = 0, None, None, 0, True
        if ok:
            self.learn(t, x)
            self.away_since = None
        elif self.away_since is None:
            self.away_since = t
        return ok


class Recognizer:
    """Holds the gallery and the table's tracks; `step` takes one frame and returns the state and events.
    `finder(t, image)` replaces the bootstrap finder, e.g. with the trained detector (`detector_boxes`)."""

    def __init__(self, layout: Layout, rows: Sequence[dict], encoder, gallery: Pyramid, title: str = "",
                 min_p: float = 0.5, sure_p: float = 0.85, recheck_s: float = 8.0, forget_s: float = 4.0,
                 max_reads: int = 12, settle_s: float = 3.0, gate: bool = True, fps: float = 5.0, gate_p: float = 0.7,
                 finder=None, temperature: float = TEMPERATURE):
        self.layout, self.rows, self.enc, self.gallery = layout, list(rows), encoder, gallery
        self.temperature = temperature  # turns an encoder's scores into how sure a read is; fitted per encoder
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
        self.boxes_now: dict[str, tuple[float, float, float, float]] = {}  # track id -> its box's extent, this frame
        # Camera cuts: frames off the table camera are skipped and the board's clocks stop (`pause`). After a
        # cut the view may be framed differently, so tracks found again by name re-anchor the rest (`cut`).
        self.scene = Scene(layout)
        self.gate_settings = GateSettings(fps=fps, card_long_frac=layout.card_long_1080 / 1080) if gate else None
        self.last_t: float | None = None
        self.away = False
        self.cut_at: float | None = None
        self.anchor_base: dict[str, CardBox] = {}   # track id -> its box before the cut
        self.anchor_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []  # (before, after) centres
        self.prev_seen: set[str] = set()     # confirmed tracks the last frame matched
        self.before_away: set[str] = set()   # ... the last frame before a cut away

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
        """Boxes to tracks one to one at the least total distance (Hungarian assignment). A box continues a
        track of about its size within a third of a card, in view or out of sight for a while, so a card
        found again where it was keeps its id and name; any other box starts a new track."""
        from scipy.optimize import linear_sum_assignment

        tracks = list(self.tracks.values())
        pairs: dict[int, Track] = {}
        if tracks and boxes:
            cost = np.full((len(boxes), len(tracks)), 1e6)
            for i, b in enumerate(boxes):
                for j, tr in enumerate(tracks):
                    d = math.dist(tr.box.centre, b.centre) / b.long_px
                    if d < 0.35 and abs(tr.box.long_px / b.long_px - 1) < 0.25:
                        cost[i, j] = d + (0.25 if t - tr.last > 1.0 else 0.0)  # the ones in view first
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    pairs[int(i)] = tracks[int(j)]
        seen = []
        for i, b in enumerate(boxes):
            tr = pairs.get(i)
            if tr is None and self.on_legend(b):
                continue  # the detector's second outline of a legend's case, or of the die on it: not a card
            if tr is None:
                tr = Track(f"t{self.next_id}", b, t, t, side=self.layout.side(*b.centre, w, h))
                self.next_id += 1
                self.tracks[tr.id] = tr
            else:
                tr.box = smooth(tr.box, b) if t - tr.last <= 1.0 else b
                tr.last, tr.hits = t, tr.hits + 1
                if not tr.pinned:
                    tr.side = self.layout.side(*tr.box.centre, w, h)
            seen.append(tr)
        return seen

    def on_legend(self, box: CardBox) -> bool:
        """The box's centre lies well inside a named legend. Only dice and counters go on a legend; the champion
        and the cards beside it lie about a card width away."""
        return any(o.pinned and o.kind == "Legend" and math.dist(o.box.centre, box.centre) < 0.35 * o.box.long_px
                   for o in self.tracks.values())

    def side_legend(self, tr: Track) -> Track | None:
        """The pinned legend of the track's side, when that is another track."""
        return next((o for o in self.tracks.values()
                     if o.pinned and o.kind == "Legend" and o.side == tr.side and o is not tr), None)

    def other_legend(self, side: str, card: str) -> bool:
        """The side already has its legend, and it is another card: one player, one legend (a rune column or a
        champion read as a legend is not a second one)."""
        lg = self.legends.get(side)
        return lg is not None and self.row_of[lg["printing_id"]]["card_id"] != card

    def covered(self, t: float, tr: Track) -> bool:
        """Something newer lies on this card: a card put on it or overlapping it. A covered card keeps its
        name (no re-reads of a half-hidden face) and stays on the board until its spot clears."""
        ax0, ay0, ax1, ay1 = self.boxes_now.get(tr.id) or aabb(tr.box)
        area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
        for oid, (bx0, by0, bx1, by1) in self.boxes_now.items():
            o = self.tracks.get(oid)
            if o is None or o is tr or o.first <= tr.first or t - o.last > 1.0:
                continue
            ix, iy = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if ix > 0 and iy > 0 and ix * iy >= 0.25 * area:
                return True
        return False

    def share(self, a: Track, b: Track) -> float:
        """The part of card `a` that card `b` overlaps, on their outlines."""
        return overlap_area(quad(a.box), quad(b.box)) / max(1.0, a.box.long_px * a.box.short_px)

    def stacked_on(self, t: float, tr: Track) -> Track | None:
        """Another named card in view, not this card outlined twice, that overlaps a quarter of this one or
        more: gear tucked under a unit, a card put on another. Legends, battlefields and runes are not stacks.
        (The detector's corner visibility would say which lies on top, but it is not reliable enough yet.)"""
        if tr.kind in STATIC + QUIET or t - tr.last > 1.0:
            return None
        best, top = 0.25, None
        for o in self.tracks.values():
            if o is tr or not o.named or o.named == tr.named or o.kind in STATIC + QUIET or t - o.last > 1.0:
                continue
            if (sh := self.share(tr, o)) >= best:
                best, top = sh, o
        return top

    def stacks(self, t: float) -> dict[str, list[Track]]:
        """The named cards under each card: a gear that overlaps a unit goes with the unit, and a card out of
        sight under a newer one (`covered`) lies under it. Units side by side at a battlefield are no stack."""
        out: dict[str, list[Track]] = {}
        for u in self.tracks.values():
            if not u.named or u.kind in STATIC + QUIET:
                continue
            host = self.stacked_on(t, u) if u.kind == "Gear" else None
            if host is not None and host.kind == "Gear":
                host = None
            if host is None and t - u.last > 1.0 and self.covered(t, u):
                host = max((o for o in self.tracks.values() if o is not u and o.named and o.named != u.named
                            and o.kind not in STATIC + QUIET and o.first > u.first and t - o.last <= 1.0
                            and self.share(u, o) >= 0.25),
                           key=lambda o: o.first, default=None)
            if host is not None:
                out.setdefault(host.id, []).append(u)
        return out

    def vanished(self, t: float, tr: Track) -> Track | None:
        """Another track of `tr`'s card that went out of sight around when `tr` appeared: the card moved."""
        gone = [o for o in self.tracks.values() if o is not tr and o.named == tr.named and not o.pinned
                and t - o.last > 0.5 and o.last < tr.first + 0.5 and t - o.last < MOVE_S and not self.covered(t, o)]
        return max(gone, key=lambda o: o.last) if gone else None

    # --- naming ----------------------------------------------------------------

    def face_down(self, tr: Track) -> bool:
        """Two face-down looks in a row on a card never named. A hand resting on a named card or a
        blurred frame does not hide it, and a card played under a hand is looked at again later."""
        return tr.named is None and tr.down >= 2

    def due(self, tr: Track, t: float) -> bool:
        if self.face_down(tr):
            return t - tr.last_read > self.recheck_s  # looked at now and then, never identified
        if tr.named and (self.covered(t, tr) or self.stacked_on(t, tr) is not None):
            return False  # its name is locked while something lies on it, or it lies on something
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
            p = np.exp((vals - vals.max()) / self.temperature)
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
            r = self.rows[i]
            if p < self.gate_p or r.get("type") in QUIET + STATIC or self.recently_played(t, card, cx, cy):
                continue
            side = self.layout.side(cx, cy, w, h)
            self.plays.append((t, card, cx, cy))
            q = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            guesses = [{"printing_id": self.rows[j]["printing_id"], "card_id": c, "name": self.rows[j]["name"], "p": round(pc, 3)}
                       for c, pc, _, j in cands[:3]]
            self.flashes.append({"id": f"g{len(self.plays)}", "quad": [[round(x, 1), round(y, 1)] for x, y in q],
                                 "side": side, "state": "named", "printing_id": r["printing_id"], "name": r["name"],
                                 "confidence": round(p, 3), "guesses": guesses, "since": round(t, 2), "until": t + 5,
                                 "kind": "card", "hidden": False})
            events.append({"t": round(t, 2), "kind": "played", "text": f"{r['name']} played", "printing_id": r["printing_id"],
                           "track": self.flashes[-1]["id"], "side": side})
        return events

    def left(self, t: float, box: tuple[float, float, float, float]) -> list[dict]:
        """A named card whose spot the gate saw cleared (a ghost, or a track the finder still holds)."""
        x0, y0, x1, y1 = box
        gx, gy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        inside = lambda x, y: x0 - gx <= x <= x1 + gx and y0 - gy <= y <= y1 + gy  # noqa: E731
        for tr in sorted(self.tracks.values(), key=lambda tr: -tr.last):
            if tr.named and not tr.pinned and t - tr.last > self.forget_s and inside(*tr.box.centre):
                del self.tracks[tr.id]
                g = self.label(tr)[2]
                return [self.event(t, "left", f"{g[0]['name'] if g else tr.named} left the table", tr,
                                   g[0]["printing_id"] if g else None)]
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

    def pause(self, dt: float) -> None:
        """Off the table camera: the board's clocks stop, so nothing is forgotten or re-read for the time away."""
        for tr in self.tracks.values():
            tr.last += dt
            tr.last_read += dt
        for gh in self.ghosts:
            gh["t"] += dt
        self.plays = [(pt + dt, c, x, y) for pt, c, x, y in self.plays]
        self.pending, self.flashes = [], []

    def cut(self, t: float) -> None:
        """The view is framed anew: cards found again by name re-anchor the rest (`reanchor`), the gate starts
        over, and the cards first seen now were on the table already, not played."""
        self.cut_at = t
        self.anchor_base = {k: tr.box for k, tr in self.tracks.items()}
        self.anchor_pairs = []
        if self.gate_settings is not None:
            self.gate = ChangeGate(self.gate_settings)

    def reanchor(self) -> None:
        """Move the tracks not seen since the cut as the view moved (scale, turn and shift fitted to the cards
        found again by name); a new track lying where a moved one now is, is that card and takes its id."""
        if len(self.anchor_pairs) < 2 or self.cut_at is None:
            return
        scale, rot, shift = similarity(np.array([a for a, _ in self.anchor_pairs]), np.array([b for _, b in self.anchor_pairs]))
        if not 0.5 <= scale <= 2.0:
            return
        turn = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        old = [o for o in self.tracks.values() if o.first < self.cut_at and o.last < self.cut_at and o.id in self.anchor_base]
        for o in old:
            b = self.anchor_base[o.id]
            c = scale * rot @ np.asarray(b.centre) + shift
            o.box = CardBox((float(c[0]), float(c[1])), b.long_px * scale, b.short_px * scale, (b.angle_deg + turn) % 180, b.fill)
        for n in [n for n in self.tracks.values() if n.first >= self.cut_at]:
            near = [o for o in old if math.dist(o.box.centre, n.box.centre) < 0.35 * n.box.long_px
                    and abs(o.box.long_px / n.box.long_px - 1) < 0.25 and (o.named is None or n.named in (None, o.named))]
            if not near:
                continue
            o = min(near, key=lambda o: math.dist(o.box.centre, n.box.centre))
            o.box, o.last, o.hits, o.side = n.box, n.last, o.hits + n.hits, n.side
            if o.named is None and n.reads:
                o.reads, o.prob, o.best_row, o.down, o.named, o.kind = n.reads, n.prob, n.best_row, n.down, n.named, n.kind
            del self.tracks[n.id]
            old.remove(o)

    def step(self, t: float, image: np.ndarray, budget: int = 10) -> tuple[dict, list[dict]]:
        tic = time.perf_counter()
        if self.t0 is None:
            self.t0 = t
        h, w = image.shape[:2]
        dt = t - self.last_t if self.last_t is not None else 0.0
        self.last_t = t
        boxes: list[CardBox] | None = None

        def count() -> int:  # the scene asks only while it learns a broadcast with no known mat colour
            nonlocal boxes
            boxes = self.find(t, image)
            return len(boxes)

        if not self.scene.on_table(t, image, count):
            if not self.away:
                self.away, self.before_away = True, set(self.prev_seen)
            self.pause(dt)
            state = self.state(t, w, h)
            for tr in state["tracks"]:
                tr["hidden"] = True  # the video is not the table: list the board, draw nothing on it
            state["status"], state["message"] = "away", "the table camera is off; nothing is looked at until it is back"
            return state, []
        back, self.away = self.away, False
        if boxes is None:
            boxes = self.find(t, image)
        tf = time.perf_counter()
        seen = self.match(t, boxes, w, h)
        before = self.before_away if back else self.prev_seen
        if len(before) >= 6 and len(before - {tr.id for tr in seen}) >= 0.7 * len(before) and len(boxes) >= 3:
            self.cut(t)  # most of the board moved at once: the view is framed anew
        elif back and self.gate_settings is not None:
            self.gate = ChangeGate(self.gate_settings)  # the same view again: only the gate starts over
        self.prev_seen = {tr.id for tr in seen if tr.hits >= 2}
        self.boxes_now = {k: aabb(tr.box) for k, tr in self.tracks.items()}
        frame = Image.fromarray(image)
        # New and uncertain cards first, then the oldest re-checks; a budget keeps each frame in time.
        # A box seen once may be the detector's slip (between two cards): only tracks seen twice are read.
        todo = sorted((tr for tr in seen if tr.hits >= 2 and self.due(tr, t)),
                      key=lambda tr: (tr.reads > 0, tr.last_read))[:budget]
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
        if tr.pinned:  # a legend or battlefield keeps its name all game: re-reads move only its confidence
            top = sorted(top, key=lambda kv: kv[0] != tr.named)
        guesses = []
        for c, p in top[:3]:
            r = self.rows[tr.best_row.get(c, (0.0, self.first_row[c]))[1]]
            guesses.append({"printing_id": r["printing_id"], "card_id": c, "name": r["name"], "p": round(p, 3)})
        p0 = top[0][1]
        if not tr.pinned and self.rows[self.first_row[top[0][0]]].get("type") == "Legend" and self.side_legend(tr):
            return "unsure", p0, guesses  # one player, one legend: another outline of it, or a card misread as one
        named = tr.pinned or p0 >= self.sure_p or (p0 >= self.min_p and tr.reads >= 2)
        if not named and tr.reads >= 4 and p0 >= 0.3 and self.rows[self.first_row[top[0][0]]].get("type") == "Legend" \
                and not self.other_legend(tr.side, top[0][0]):
            named = True  # a legend, read the same way four times: one a player, and its frame is like no other card's
        return ("named" if named else "unsure"), p0, guesses

    def announce(self, t: float) -> list[dict]:
        """'played' when a card is first named, 'moved' when a named card that just vanished is named again
        elsewhere (it keeps its first id). A card out of sight keeps its track: an unnamed one `forget_s`,
        a named one `KEEP_S` or as long as something lies on it, a legend or battlefield all game. Then a
        named card becomes a ghost (see `ghosts`). Runes, legends and battlefields are never announced."""
        events = []
        self.ghosts = [gh for gh in self.ghosts if t - gh["t"] < 60]
        for tr in list(self.tracks.values()):
            if tr.id not in self.tracks:
                continue  # merged into the track it moved from
            gone = t - tr.last
            limit = KEEP_S if tr.named else self.forget_s if tr.hits >= 2 else 1.0
            if not tr.pinned and gone > limit and not (tr.named and self.covered(t, tr)):
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
                tr.kind = (self.row_of.get(g[0]["printing_id"]) or {}).get("type", "")
                after_cut = self.cut_at is not None and not changed and 0 <= tr.first - self.cut_at < 30
                if after_cut:  # a card on the table before the cut, found again by name in the new view
                    olds = [o for o in self.tracks.values() if o is not tr and o.named == tr.named
                            and o.first < self.cut_at and o.last < self.cut_at]
                    if len(olds) == 1:
                        o = olds[0]
                        self.anchor_pairs.append((self.anchor_base.get(o.id, o.box).centre, tr.box.centre))
                        o.box, o.last, o.hits, o.side = tr.box, tr.last, o.hits + tr.hits, tr.side
                        del self.tracks[tr.id]  # it keeps its first id
                        self.reanchor()
                        continue
                if tr.kind in STATIC and not (tr.kind == "Legend" and (self.other_legend(tr.side, tr.named)
                                                                      or self.side_legend(tr) is not None)):
                    tr.pinned = True  # set up before the game: nothing to announce, and it stays put
                    if tr.kind == "Legend":  # the side's legend, however sure its reads are under the dice
                        self.legends.setdefault(tr.side, {"printing_id": g[0]["printing_id"], "name": g[0]["name"]})
                    for o in [o for o in self.tracks.values() if not o.pinned and self.on_legend(o.box)]:
                        del self.tracks[o.id]  # a second outline of a legend, read as a card of its own
                if tr.kind in QUIET + STATIC:
                    continue
                if not changed and not after_cut and (was := self.vanished(t, tr)) is not None:
                    far = math.dist(was.box.centre, tr.box.centre) > 0.6 * tr.box.long_px
                    was.box, was.last, was.hits, was.side = tr.box, tr.last, was.hits + tr.hits, tr.side
                    del self.tracks[tr.id]  # the same card: it keeps its first id and what was read of it
                    if far:
                        events.append(self.event(t, "moved", f"{g[0]['name']} moved", was, g[0]["printing_id"]))
                    continue
                reach = 1.5 * self.layout.card_long_1080
                back = [gh for gh in self.ghosts if gh["card"] == tr.named and math.dist((gh["x"], gh["y"]), tr.box.centre) < reach]
                if back and not changed:
                    self.ghosts.remove(back[0])  # the same card, found again
                    continue
                if tr.first - (self.t0 or 0.0) < self.settle_s and not changed:
                    continue  # on the table when we tuned in, not played now
                if after_cut and tr.first - self.cut_at < self.settle_s + 2:
                    continue  # on the table when the view changed
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
        under = self.stacks(t)
        for tr in self.tracks.values():
            if tr.hits < 2:
                continue  # seen once: maybe the detector's slip (a box between two cards), not shown yet
            hidden = t - tr.last > 1.0 and not tr.pinned
            if hidden and not tr.named:
                continue
            state, p, g = self.label(tr)
            if (lg := self.side_legend(tr)) is not None and not tr.pinned and g and g[0]["card_id"] == lg.named:
                continue  # another outline of the side's legend (its case, the die on it): not a card
            top = g[0] if g and state == "named" else None
            # hidden: out of sight (under a hand or another card) but still on the board, so listed, not drawn
            tracks.append({"id": tr.id, "quad": quad(tr.box), "side": tr.side, "state": state,
                           "printing_id": top["printing_id"] if top else None, "name": top["name"] if top else "",
                           "confidence": round(p, 3), "guesses": g if state != "facedown" else [],
                           "since": round(tr.first, 2), "kind": KINDS.get(tr.kind, "card"), "hidden": hidden,
                           "under": [{"id": u.id, "name": gu[0]["name"], "printing_id": gu[0]["printing_id"]}
                                     for u in under.get(tr.id, []) if (gu := self.label(u)[2])]})
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
