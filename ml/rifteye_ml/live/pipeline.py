# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
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

The legend rule (D-026, `priors.legend_mask`): once a side's legend is pinned, every crop read on that
side competes only with the printings that fit that legend's domains, runes included, and with every
battlefield and token. Until then, and with `legend_rule=False`, a crop competes with the whole gallery.
"""
from __future__ import annotations

import math
import time
from itertools import combinations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from .. import priors
from ..changegate import ChangeGate, GateSettings, skin
from ..detect.geometry import overlap_area
from ..matcrops import FACE_DOWN_DETAIL, CardBox, detail, find_cards, mat_colour, notmat_mask
from ..retrieval import Pyramid, ROTATIONS
from .layouts import Layout

TEMPERATURE = 0.0212  # fitted on the M0 real labels (reviewpack identity)
KEEP_S = 60.0   # a named card out of sight this long has gone: not a hand over it, dice, a card on top
MOVE_S = 10.0   # a named card that vanished this recently and is named again elsewhere has moved
STATIC = ("Legend", "Battlefield")  # set up before the game: pinned where they are named (a moved battlefield's pin follows it)
QUIET = ("Rune",)                   # tracked, but never labelled, listed or announced: not worth watching
UNPIN_READS = 4    # a pinned battlefield read this often, and as itself on fewer than one read in UNPIN_P, was misread
UNPIN_P = 0.2      # once: a rune or a unit turned sideways (exhausted) looks like a battlefield's landscape art
PIN_HIDE_S = 20.0  # a legend or battlefield out of sight this long is not drawn (a hand resting on it is shorter)
HELD_OUT = 0.15    # a card past the table window's edge on a player's side by this much of its width is in a hand
STRIP = 0.12       # the battlefield strip: the band this far either side of the midline between the players, as a share
                   # of the table's width across them (the official mat's is 0.11; battlefields measured within 0.09)
STRIP_READS = 3    # off the strip, a battlefield is named only after this many reads, and never pinned: a rune or a unit
                   # turned sideways looks like a battlefield's landscape art, and players lay out their tables differently
RUNE_SEEN_S = 1.0       # a rune counts in a frame while seen this recently (a hand passing over it is shorter)
RUNE_APART = 0.15       # two rune boxes closer than this share of a card's length are one rune (two tracks on one card)
RUNE_ASPECT = 1.7       # a rune box longer than this for its width, or than RUNE_LONG cards, spans two runes: a slip
RUNE_LONG = 1.3
RUNE_WINDOW_S = 12.0    # a player's rune count is the upper quartile of the frames' counts over this long, and the exhausted
RUNE_EXHAUSTED_S = 3.0  # ones the median over this long: hands over the runes and boxes between two come and go, runes stay
RUNE_SHARE = 0.3        # a box whose reads put this share on runes is one, whatever it is named: runes are counted, never
                        # named, and a foil rune or a stacked one's strip is often read as another card
RUNE_JOIN = 0.6         # an unread or unnamed card-sized box this close to a rune (in cards), turned its way, is one too
RUNE_TURN = 20.0        # (within this many degrees, and its length within RUNE_SIZE of a card's): a stack's covered strips
RUNE_SIZE = 0.2
RUNE_LINK = 0.6         # runes this close (in cards) are one stack: a column or a fan
RUNE_STEP = (0.18, 0.35)  # a stack's step from strip to strip, as shares of a card's length: the gaps of that size
RUNE_GAP = 1.6          # a gap this many steps wide hides runes the detector missed: about gap / step - 1 of them
HAND_SKIN = 0.10       # a card with this share of skin in the band around it is in a hand: held, or being put down
HAND_FREE_S = 0.5      # out of a hand and still this long before a card is read: a card held over the table is never named
HAND_STILL = 0.04      # still: moved less than this share of a card's length since (a hand that holds a card moves it)
HAND_RING = (0.1, 0.2, 0.3)  # the band looked at, as distances out from the card's edges in shares of its width
HAND_POINTS = 8        # points along each side of the band, at each distance
HAND_MIN_POINTS = 8    # fewer of them off the other cards and in the window: no hand to see
HAND_DIFF = 40         # a skin-coloured point this unlike the still table there (`StillTable`) is a hand; one like it is the
                       # table itself: a wooden table is skin-coloured, and the cards lying beside it are not in a hand
HAND_BG = (480, 270)   # the still table: the frame at a quarter of 1080p, ...
HAND_BG_FRAMES = 5     # ... first the median of this many table frames HAND_BG_EVERY apart (hands move; the table does not),
HAND_BG_EVERY = 0.5
HAND_BG_STEP = 2       # ... then every HAND_BG_EVERY this much nearer the frame, so a hand passing over it stays a hand
SIZE_MAX = 1.45        # a box longer than this many cards is not one: the co-stream's chat, two cards or a card and the
                       # printed zone beside it outlined as one (named cards: 99.5% within 1.24 on two finals)
PLAIN_TOL = 18         # a box the mat's own colour inside as around it (medians, this close), and plain inside (the middle
PLAIN_SPREAD = 24      # half of its points within this), is a zone printed on the mat, not a card
BURST = 4              # this many cards named for the first time on the table within BURST_S are not that many plays: a
BURST_S = 1.0          # graphic of cards (a sideboard, a decklist) or a view framed anew; nobody plays four cards a second
OVERLAY_TOL = 16       # a thumbnail pixel within this of the frame before, through a cut, stayed: the broadcast's overlay
CUT_SHARE = 0.5        # a frame whose thumbnail changed this much from the one before is a cut (play changes a quarter at most)
OVERLAY_SHARE = 0.9    # the overlay: what stayed through this share of the cuts, once there are OVERLAY_CUTS, in patches
OVERLAY_CUTS = 4       # reaching within OVERLAY_EDGE pixels of the frame's edge (a co-streamer's webcam and chat, a
OVERLAY_EDGE = 2       # scoreboard, a sponsor banner), holes filled
OVERLAY_KEEP = 0.6     # a pixel once overlay stays so while it stayed through this share of the cuts: a face moving in a
                       # webcam changes it at some cuts, and opens the webcam's frame to the table
SUSPECT_S = 60.0       # before then, what stayed through every cut so far is suspect: nothing there is read, until the cuts
                       # make it overlay, or for this long after the last cut (one cut alone can be the camera reframed)
SCENE_GRID = (6, 3)    # the scene's score is the mean of its scores in a 6 x 3 grid of blocks of the thumbnail: an arm or a
SCENE_BLOCK = 20       # banner over the table spoils a block or two, another shot all of them; a block counts with this
                       # many still pixels of the table window in it
RELEARN_CARDS = 5      # the scene learns a moved camera again only when this many card-sized boxes lie in the view, and
CARDS_S = 60.0         # half the most the table camera showed in this long of its last time on screen
KINDS ={"Legend": "legend", "Battlefield": "battlefield", "Rune": "rune"}


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
    free_since: float | None = None                  # seen out of a hand and still since (`hand_share`), from free_at
    free_at: tuple[float, float] | None = None
    placed: bool = False                             # out of a hand and still HAND_FREE_S once: on the table

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


def on_box(box: CardBox, x: float, y: float) -> bool:
    """The point lies on the card."""
    a = math.radians(box.angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    dx, dy = x - box.centre[0], y - box.centre[1]
    return abs(dx * ux + dy * uy) <= box.long_px / 2 and abs(dy * ux - dx * uy) <= box.short_px / 2


def hand_share(image: np.ndarray, box: CardBox, table: tuple[int, int, int, int], others: Sequence[CardBox] = (),
               still: np.ndarray | None = None, now: np.ndarray | None = None) -> float:
    """The share of skin-coloured points in a band around the box, inside the table window and off the other cards
    (`others`): the fingers holding a card in a hand over the table, or putting it down. No card's art is looked at:
    gold, faces and fire are skin-coloured too. Hemmed in by other cards, it sees no hand. With `still`, the still
    table (`StillTable.bg`), and `now`, this frame at its size (HAND_BG), a point counts only where the frame there is
    HAND_DIFF unlike the table: on a wooden table the wood is skin-coloured too, and a hand is what is not the table
    (compared at the same size, so a mat's thin printed lines are the table too)."""
    x0, y0, x1, y1 = table
    h, w = image.shape[:2]
    a = math.radians(box.angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    cx, cy = box.centre
    near = [o for o in others if math.dist(o.centre, box.centre) < o.long_px + box.long_px]
    px, at = [], []
    for f in HAND_RING:
        d = f * box.short_px
        hl, hs = box.long_px / 2 + d, box.short_px / 2 + d
        for k in range(HAND_POINTS):
            along_l = (k + 0.5) / HAND_POINTS * 2 * hl - hl
            along_s = (k + 0.5) / HAND_POINTS * 2 * hs - hs
            for su, sv in ((along_l, hs), (along_l, -hs), (hl, along_s), (-hl, along_s)):
                x, y = cx + su * ux - sv * uy, cy + su * uy + sv * ux
                xi, yi = math.floor(x), math.floor(y)
                if max(x0, 0) <= xi < min(x1, w) and max(y0, 0) <= yi < min(y1, h) and not any(on_box(o, x, y) for o in near):
                    px.append(image[yi, xi])
                    at.append((yi, xi))
    if len(px) < HAND_MIN_POINTS:
        return 0.0
    pts = np.asarray(px, np.uint8).reshape(1, -1, 3)
    hand = skin(pts)[0]
    if still is not None and now is not None:
        sh, sw = still.shape[:2]
        at_bg = [(min(sh - 1, yi * sh // h), min(sw - 1, xi * sw // w)) for yi, xi in at]
        bg = np.array([still[by, bx] for by, bx in at_bg], np.int16)
        fg = np.array([now[by, bx] for by, bx in at_bg], np.int16)
        hand &= np.abs(fg - bg).max(axis=1) >= HAND_DIFF
    return float(hand.mean())


class StillTable:
    """The table camera's picture without the hands over it, for the hand rule (`hand_share`): HAND_BG_FRAMES table
    frames HAND_BG_EVERY apart, their median per pixel (hands move, the table does not), then every HAND_BG_EVERY a
    step of HAND_BG_STEP nearer the frame: a hand passing over the table for a few seconds stays unlike it, a card put
    down becomes part of it within half a minute. Whole numbers throughout, as the engine's are. Fed the frames at its
    size (`small`)."""

    def __init__(self) -> None:
        self.first: list[np.ndarray] = []
        self.bg: np.ndarray | None = None
        self.last = -1e9

    @staticmethod
    def small(image: np.ndarray) -> np.ndarray:
        """A frame at the still table's size."""
        return np.asarray(Image.fromarray(image).resize(HAND_BG, Image.BOX), np.int16)

    def feed(self, t: float, x: np.ndarray) -> None:
        if t - self.last < HAND_BG_EVERY:
            return
        self.last = t
        if self.bg is None:
            self.first.append(x)
            if len(self.first) == HAND_BG_FRAMES:
                self.bg = np.sort(np.stack(self.first), axis=0)[HAND_BG_FRAMES // 2]
                self.first = []
            return
        self.bg = self.bg + np.clip(x - self.bg, -HAND_BG_STEP, HAND_BG_STEP)


def plain_zone(image: np.ndarray, box: CardBox) -> bool:
    """A box of the mat's own colour inside as around it, and plain inside: a zone printed on the mat (outlined, a
    card's size) or the mat's logo, which the detector outlines like a card. A card's face is never plain, and a face-down
    card is plain in its sleeve's colour, not the mat's (one the mat's colour is not told apart, and is never read either).
    Inside: 5 x 5 points over the middle 60% of the box; around: the band HAND_RING[1] out from its edges. Their medians,
    and the inside's quartiles, of whole numbers."""
    h, w = image.shape[:2]
    a = math.radians(box.angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    cx, cy = box.centre
    inside, ring = [], []
    for i in range(5):
        for j in range(5):
            su, sv = (i - 2) * 0.15 * box.long_px, (j - 2) * 0.15 * box.short_px
            xi, yi = math.floor(cx + su * ux - sv * uy), math.floor(cy + su * uy + sv * ux)
            if 0 <= xi < w and 0 <= yi < h:
                inside.append(image[yi, xi])
    d = HAND_RING[1] * box.short_px
    hl, hs = box.long_px / 2 + d, box.short_px / 2 + d
    for k in range(HAND_POINTS):
        along_l = (k + 0.5) / HAND_POINTS * 2 * hl - hl
        along_s = (k + 0.5) / HAND_POINTS * 2 * hs - hs
        for su, sv in ((along_l, hs), (along_l, -hs), (hl, along_s), (-hl, along_s)):
            xi, yi = math.floor(cx + su * ux - sv * uy), math.floor(cy + su * uy + sv * ux)
            if 0 <= xi < w and 0 <= yi < h:
                ring.append(image[yi, xi])
    if len(inside) < 25 or len(ring) < 2 * HAND_POINTS:
        return False
    si, sr = np.sort(np.asarray(inside, np.int16), axis=0), np.sort(np.asarray(ring, np.int16), axis=0)
    n = len(ring)
    mid_in, mid_ring = si[12], (sr[(n - 1) // 2] + sr[n // 2]) / 2
    return bool(np.abs(mid_in - mid_ring).max() < PLAIN_TOL and (si[18] - si[6]).max() <= PLAIN_SPREAD)


def turn_apart(a: float, b: float) -> float:
    """Degrees between two boxes' long sides (a box's angle is modulo 180)."""
    return abs((a - b + 90) % 180 - 90)


def stacks_of(boxes: Sequence[CardBox], reach: float) -> list[list[CardBox]]:
    """The boxes in groups, each linked box closer than `reach` to another of its group."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if math.dist(boxes[i].centre, boxes[j].centre) <= reach:
                parent[find(i)] = find(j)
    groups: dict[int, list[CardBox]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    return list(groups.values())


def hidden_runes(stack: Sequence[CardBox], card: float) -> int:
    """Runes a column or fan hides from the detector: along the stack, its usual step from strip to strip (the gaps of
    RUNE_STEP cards), and each gap of RUNE_GAP steps or more holding about gap / step - 1 more. Cards are one size."""
    if len(stack) < 3:
        return 0
    a, b = max(((x, y) for x in stack for y in stack), key=lambda p: math.dist(p[0].centre, p[1].centre))
    span = math.dist(a.centre, b.centre)
    if span < 1e-6:
        return 0
    ux, uy = (b.centre[0] - a.centre[0]) / span, (b.centre[1] - a.centre[1]) / span
    at = sorted((o.centre[0] - a.centre[0]) * ux + (o.centre[1] - a.centre[1]) * uy for o in stack)
    gaps = [y - x for x, y in zip(at, at[1:])]
    steps = sorted(g for g in gaps if RUNE_STEP[0] * card <= g <= RUNE_STEP[1] * card)
    if not steps:
        return 0
    step = steps[len(steps) // 2]
    return sum(max(0, math.floor(g / step + 0.5) - 1) for g in gaps if g >= RUNE_GAP * step)


def hidden_now(t: float, tr: "Track") -> bool:
    """Out of sight: not seen for more than a second."""
    return t - tr.last > 1.0


def upright(box: CardBox) -> bool:
    """The card's long side runs up the picture rather than across it."""
    return 45 <= box.angle_deg % 180 < 135


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


MIN_ASPECT = 0.5  # a card's short side over its long side is 0.72 (63 x 88 mm); a box under half is a strip


def detector_boxes(dets: Sequence[dict], min_score: float = 0.4, min_aspect: float = MIN_ASPECT) -> list[CardBox]:
    """The trained detector's cards (`detect.model.Detector.detect`: corners in frame px) as boxes for the
    tracker. A `card_back` is marked `back`, so it is never identified. A box less than `min_aspect` as wide
    as it is long is not a card: the detector outlines the art of Riot's showdown banner, laid over the
    bottom of the table, as strips 2.5 times as long as wide (the detector's cards, even under others, keep
    a card's shape)."""
    out = []
    for d in dets:
        if d["score"] < min_score:
            continue
        q = np.asarray(d["quad"], np.float64).reshape(4, 2)
        e = [q[(k + 1) % 4] - q[k] for k in range(4)]
        a, b = (np.linalg.norm(e[0]) + np.linalg.norm(e[2])) / 2, (np.linalg.norm(e[1]) + np.linalg.norm(e[3])) / 2
        if min(a, b) < min_aspect * max(a, b):
            continue
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
    seconds, with five cards of the table's size in it (the camera itself moved), it learns again. A mat's
    colour alone is no proof: on a co-stream every shot, a player in a maroon shirt included, had it. The
    thumbnail is too coarse to show any card.

    Only the table window is scored: the panels beside it are laid over every shot, a close-up of a hand
    included. It is scored in blocks (SCENE_GRID), and the score is their mean: an arm or a banner over the
    table spoils a block or two, where a cut to another shot spoils them all.

    What stays put through the cuts is not the table camera's at all: a co-streamer's webcam and chat, a scoreboard,
    a sponsor banner, laid over every shot (`overlay`). Kept in the score, it makes every shot look like the table, so
    it is left out once known; and a frame is learnt only when the one before was the table camera too, so the first
    frame after a cut, another shot that happens to look alike, never teaches the scene what the table is."""

    def __init__(self, layout: Layout, corr: float = 0.45, learn_every: float = 2.0, relearn_after: float = 20.0):
        self.layout, self.corr, self.learn_every, self.relearn_after = layout, corr, learn_every, relearn_after
        self.n = 0
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.last_learn = -1e9
        self.away_since: float | None = None
        self.looks = 0  # table-like frames in a row while away (checked every learn_every)
        self.last_look = -1e9
        self.was_on = False                   # the frame before was the table camera
        self.prev: np.ndarray | None = None   # the frame before's thumbnail
        self.same = np.zeros((54, 96), np.int32)  # cuts each thumbnail pixel stayed through, the board's own
        self.cuts = 0
        self.same_all = np.zeros((54, 96), np.int32)  # ... and with those seen before the board (`Recognizer.prime`)
        self.cuts_all = 0
        self.overlay: np.ndarray | None = None    # (54, 96) bool, once OVERLAY_CUTS cuts are seen
        self.overlay_n = 0                    # how often it was worked out: the board drops what lies in it then
        self.suspect: np.ndarray | None = None    # (54, 96) bool: what stayed through every cut so far, before then
        self.cut_t: float | None = None       # when the last cut was (the cuts seen before the board, when it began)
        self.cards_seen: list[tuple[float, int]] = []  # (t, card-sized boxes) on the table camera, its last minute
        tx0, ty0, tx1, ty1 = layout.table
        self.window = np.zeros((54, 96), bool)   # the thumbnail pixels in the table window: the only ones scored
        self.window[math.floor(ty0 * 54):math.ceil(ty1 * 54), math.floor(tx0 * 96):math.ceil(tx1 * 96)] = True

    @staticmethod
    def small(image: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(image).resize((96, 54), Image.BOX), np.float32)

    def see(self, x: np.ndarray, before: bool = False) -> None:
        """A cut, when half the thumbnail changed from the frame before (play changes a quarter at most): every pixel
        that stayed counts once more as overlay, and the overlay is worked out again. A cut seen `before` the board (the
        frames the table was looked for in) makes what stayed suspect, never overlay: a scoreboard that comes with the
        table camera did not stay through the cut from a player cam to it, and is overlay all the same."""
        if self.prev is not None:
            moved = np.abs(x - self.prev).max(axis=2) >= OVERLAY_TOL
            if int(moved.sum()) * 2 >= moved.size:
                self.same_all += ~moved
                self.cuts_all += 1
                if not before:
                    self.same += ~moved
                    self.cuts += 1
                self.cut_t = None  # stamped by on_table, with the time
                if self.cuts >= OVERLAY_CUTS:
                    self.overlay = overlay_patches(self.same, self.cuts, self.overlay)
                    self.overlay_n += 1
                    self.suspect = None
                else:
                    self.suspect = overlay_patches(self.same_all, self.cuts_all)
        self.prev = x

    def in_suspect(self, x: float, y: float, w: int, h: int) -> bool:
        """The frame point (x, y) lies where the overlay may be, before the cuts have shown it."""
        if self.suspect is None:
            return False
        return bool(self.suspect[min(53, max(0, math.floor(y * 54 / h))), min(95, max(0, math.floor(x * 96 / w)))])

    def in_overlay(self, x: float, y: float, w: int, h: int) -> bool:
        """The frame point (x, y) lies in the overlay."""
        if self.overlay is None:
            return False
        return bool(self.overlay[min(53, max(0, math.floor(y * 54 / h))), min(95, max(0, math.floor(x * 96 / w)))])

    def table_like(self, image: np.ndarray, count) -> bool:
        if self.layout.mat is not None:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = self.layout.box(w, h)
            a = np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize((160, 120), Image.BOX), np.float32)
            mat = np.asarray(self.layout.mat, np.float32)
            return float((np.abs(a - mat).max(axis=2) < self.layout.mat_tol).mean()) >= self.layout.mat_share
        return count() >= 5

    def saw_cards(self, t: float, n: int) -> None:
        """The board found `n` card-sized boxes on a frame of the table camera at `t`."""
        self.cards_seen = [(tt, c) for tt, c in self.cards_seen if t - tt < CARDS_S] + [(t, n)]

    def table_again(self, image: np.ndarray, count) -> bool:
        """A view to learn as the table camera again: table-like, with as many cards of the table's size in it as the
        table camera showed in its last minute on screen (half the most, and at least RELEARN_CARDS). A close-up of one
        side of the table, or of a hand over it, shows a few of its cards."""
        if not self.table_like(image, count):
            return False
        n = count()
        return n >= RELEARN_CARDS and n * 2 >= max((c for _, c in self.cards_seen), default=0)

    def score(self, x: np.ndarray) -> float | None:
        """How well the frame's still parts in the table window match the table camera's: the mean of their
        correlations block by block (SCENE_GRID), or None when no block has a pattern to match (a plain mat):
        then the mat share or the cards decide. The overlay laid over every shot is not the table camera's: it
        is left out."""
        std = np.sqrt(self.var).max(axis=2)
        still = std < 12 if (std < 12).mean() >= 0.1 else std <= np.quantile(std, 0.3)
        if self.overlay is not None:
            still = still & ~self.overlay
        still = still & self.window
        gx, gy = SCENE_GRID
        bw, bh = 96 // gx, 54 // gy
        total, n = 0.0, 0
        for by in range(gy):
            for bx in range(gx):
                sl = (slice(by * bh, (by + 1) * bh), slice(bx * bw, (bx + 1) * bw))
                m = still[sl]
                if int(m.sum()) < SCENE_BLOCK:
                    continue
                a, b = x[sl][m].ravel(), self.mean[sl][m].ravel()
                if b.std() < 8:
                    continue
                a, b = a - a.mean(), b - b.mean()
                total += float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
                n += 1
        return total / n if n else None

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
        self.see(x)
        if self.cuts_all and self.cut_t is None:
            self.cut_t = t  # a cut now, or the ones seen before the board began
        if self.suspect is not None and t - self.cut_t > SUSPECT_S:
            self.suspect = None  # no cut for a minute: nothing there is held back any longer
        sc = self.score(x) if self.n >= 5 else None
        if sc is None:
            ok = self.table_like(image, count)
        else:
            ok = sc >= self.corr
            if not ok and self.away_since is not None and t - self.away_since > self.relearn_after \
                    and t - self.last_look >= self.learn_every:
                self.last_look = t
                self.looks = self.looks + 1 if self.table_again(image, count) else 0
                if self.looks >= 3:  # the table again, but not as it was learnt: the camera moved
                    self.n, self.mean, self.var, self.looks, ok = 0, None, None, 0, True
        if ok:
            if self.was_on:  # not the first frame after a cut: another shot may look like the table for a frame
                self.learn(t, x)
            self.away_since = None
        elif self.away_since is None:
            self.away_since = t
        self.was_on = ok
        return ok


def overlay_patches(same: np.ndarray, cuts: int, was: np.ndarray | None = None) -> np.ndarray:
    """The overlay: the thumbnail pixels that stayed through OVERLAY_SHARE of the cuts, in patches reaching within
    OVERLAY_EDGE pixels of the frame's edge (a broadcast lays its graphics along the edges; the mat lies inside),
    each patch's holes filled (a webcam's frame stays, the face in it moves). The overlay worked out before, `was`,
    stays where it stayed through OVERLAY_KEEP of the cuts: the face that moved at a cut also crossed the webcam's
    frame, and a hole open to the table is not filled."""
    from scipy import ndimage

    ov = same * 10 >= cuts * round(OVERLAY_SHARE * 10)
    if was is not None:
        ov |= was & (same * 10 >= cuts * round(OVERLAY_KEEP * 10))
    lab, _ = ndimage.label(ov)
    keep = np.zeros_like(ov)
    h, w = ov.shape
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None and (sl[0].start <= OVERLAY_EDGE or sl[1].start <= OVERLAY_EDGE
                               or sl[0].stop >= h - OVERLAY_EDGE or sl[1].stop >= w - OVERLAY_EDGE):
            keep |= lab == i
    return ndimage.binary_fill_holes(keep)


class Recognizer:
    """Holds the gallery and the table's tracks; `step` takes one frame and returns the state and events.
    `finder(t, image)` replaces the bootstrap finder, e.g. with the trained detector (`detector_boxes`)."""

    def __init__(self, layout: Layout, rows: Sequence[dict], encoder, gallery: Pyramid, title: str = "",
                 min_p: float = 0.5, sure_p: float = 0.85, recheck_s: float = 8.0, forget_s: float = 4.0,
                 max_reads: int = 12, settle_s: float = 3.0, gate: bool = True, fps: float = 5.0, gate_p: float = 0.7,
                 finder=None, temperature: float = TEMPERATURE, legend_rule: bool = True):
        self.layout, self.rows, self.enc, self.gallery = layout, list(rows), encoder, gallery
        self.temperature = temperature  # turns an encoder's scores into how sure a read is; fitted per encoder
        self.cards = np.array([r["card_id"] for r in self.rows])
        self.first_row: dict[str, int] = {}
        for i, r in enumerate(self.rows):
            self.first_row.setdefault(r["card_id"], i)
        self.row_of = {r["printing_id"]: r for r in self.rows}
        self.type_of = {r["card_id"]: r.get("type", "") for r in self.rows}
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
        # The legend rule: a side's crops compete only with the rows its pinned legend allows (`allowed`).
        self.legend_rule = legend_rule
        self.masks: dict[str, np.ndarray] = {}   # legend card_id -> the gallery rows its side's crops compete with
        self.decks: list = []                    # the decklists given (set_lists): a list holds the side whose legend it names
        self._cat = None                         # the rows indexed for decklists, once a list is read
        self.tokens: np.ndarray | None = None    # priors.token_rows, once a legend needs it
        self.boxes_now: dict[str, tuple[float, float, float, float]] = {}  # track id -> its box's extent, this frame
        self.frame_wh: tuple[int, int] = (1920, 1080)  # the frame's size, for where a card lies on the table
        self.card_rows: dict[str, np.ndarray] = {}    # card_id -> its gallery rows, for a named card read again
        self.rune_counts: dict[str, list[tuple[float, int, int]]] = {}  # side -> (t, runes, exhausted) of recent frames
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
        self.still = StillTable()            # the table without the hands over it, for the hand rule
        self.overlay_n = 0                   # the scene's overlay as last swept off the board (`drop_overlaid`)
        self.first_named: list[float] = []   # when the cards first named lately were, for BURST
        # The plays announced, by track (or flash) id: where they were, so a play read off the overlay before the cuts
        # showed it can be withdrawn (`drop_overlaid`).
        self.announced: dict[str, tuple[float, float, str, str | None, str]] = {}

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

    def prime(self, frames: Sequence[np.ndarray]) -> None:
        """The frames seen before the board began, while the table was looked for (`autolayout`), in order: a cut among
        them (from a player cam to the table, most often) shows the scene some of the overlay from the start."""
        for f in frames:
            self.scene.see(Scene.small(f), before=True)

    def keep(self, boxes: Sequence[CardBox], image: np.ndarray, w: int, h: int) -> list[CardBox]:
        """The finder's boxes that can be cards: no longer than SIZE_MAX cards, off the overlay laid over every shot
        (`Scene.overlay`), and not a zone printed on the mat (`plain_zone`)."""
        px = self.layout.card_px(h)
        return [b for b in boxes if b.long_px <= SIZE_MAX * px and not self.scene.in_overlay(*b.centre, w, h)
                and not plain_zone(image, b)]

    def drop_overlaid(self, t: float, w: int, h: int) -> list[dict]:
        """What lies in the overlay, now that the cuts have shown it: the cards read off a webcam or a banner before
        then go, a legend one of them gave its side with it, and the plays they made are withdrawn."""
        self.overlay_n = self.scene.overlay_n
        events = []
        for key, (x, y, name, pid, side) in list(self.announced.items()):
            if self.scene.in_overlay(x, y, w, h):
                del self.announced[key]
                events.append({"t": round(t, 2), "kind": "withdrawn", "text": f"{name} withdrawn: it was the stream's overlay",
                               "printing_id": pid, "track": key, "side": side})
        for tr in [tr for tr in self.tracks.values() if self.scene.in_overlay(*tr.box.centre, w, h)]:
            del self.tracks[tr.id]
            lg = self.legends.get(tr.side)
            if lg is not None and tr.named is not None and (self.row_of.get(lg["printing_id"]) or {}).get("card_id") == tr.named:
                del self.legends[tr.side]
        self.ghosts = [gh for gh in self.ghosts if not self.scene.in_overlay(gh["x"], gh["y"], w, h)]
        return events

    # --- tracking --------------------------------------------------------------

    def match(self, t: float, boxes: Sequence[CardBox], w: int, h: int) -> list[Track]:
        """Boxes to tracks one to one at the least total distance (Hungarian assignment). A box continues a
        track of about its size within a third of a card, in view or out of sight for a while, so a card
        found again where it was keeps its id and name; any other box starts a new track."""
        from scipy.optimize import linear_sum_assignment

        boxes = [b for b in boxes if not self.held(b, w, h)]
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

    def held(self, box: CardBox, w: int, h: int) -> bool:
        """A card across the table window's edge on a player's side: in a hand held over the table, or on its way
        there, not lying on it. Never tracked, so a player's hand is never read (D-005). (A card held just inside
        the edge looks like the cards lying there, on some broadcasts a quarter of a card from it: not caught.)"""
        x0, y0, x1, y1 = self.layout.box(w, h)
        bx0, by0, bx1, by1 = aabb(box)
        m = HELD_OUT * box.short_px
        if self.layout.split == "horizontal":
            return by0 < y0 - m or by1 > y1 + m
        return bx0 < x0 - m or bx1 > x1 + m

    def in_strip(self, x: float, y: float) -> bool:
        """The point lies in the battlefield strip: the band along the table's midline between the players, where the
        battlefields lie and where either player's units go to fight over them."""
        w, h = self.frame_wh
        x0, y0, x1, y1 = self.layout.box(w, h)
        u = (y - y0) / max(1, y1 - y0) if self.layout.split == "horizontal" else (x - x0) / max(1, x1 - x0)
        return abs(u - 0.5) <= STRIP

    def other_side(self, side: str) -> str:
        a, b = self.layout.sides()
        return b if side == a else a

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

    def twin(self, t: float, tr: Track) -> bool:
        """An older track in view with the same name that covers half of this one: the same card outlined twice."""
        return bool(tr.named) and not tr.pinned and not hidden_now(t, tr) and any(
            o is not tr and o.named == tr.named and (o.first, int(o.id[1:])) < (tr.first, int(tr.id[1:]))
            and not hidden_now(t, o) and self.share(tr, o) >= 0.5 for o in self.tracks.values())

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

    def misread_battlefield(self, tr: Track) -> bool:
        """A pinned battlefield whose reads since say it is another card: a rune or a unit turned sideways
        (exhausted), read once as a battlefield, which the pin would otherwise keep all game."""
        top = tr.top()
        return tr.reads >= UNPIN_READS and bool(top) and top[0][0] != tr.named and dict(top).get(tr.named, 0.0) < UNPIN_P

    def pinned_twin(self, t: float, tr: Track) -> Track | None:
        """The pinned track of the battlefield `tr` is read as, when it is that card again: out of sight (the
        battlefield was moved, and its pin follows it), or in sight and overlapping `tr` (outlined twice)."""
        return next((o for o in self.tracks.values() if o is not tr and o.pinned and o.kind == "Battlefield"
                     and o.named == tr.named and (hidden_now(t, o) or max(self.share(tr, o), self.share(o, tr)) >= 0.25)),
                    None)

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
        if tr.pinned and tr.kind == "Battlefield" and top[0][0] != tr.named and tr.reads < self.max_reads:
            return True  # read as another card than its pin: read again at once, until the reads can undo it
        return t - tr.last_read > self.recheck_s

    def allowed(self, side: str) -> np.ndarray | None:
        """The gallery rows a crop on `side` competes with: once the side's legend is pinned, the cards of the lists
        that name it (`decklist.list_mask`: every printing of them, both lists' battlefields and the tokens) or, when no
        list does, those that fit the legend (`priors.legend_mask`, runes held to its domains; every battlefield and
        token), one mask per legend. None, the whole gallery, before that, on a side with no legend, or with the rule
        off."""
        from .. import decklist

        lg = self.legends.get(side) if self.legend_rule and side else None
        row = self.row_of.get(lg["printing_id"]) if lg is not None else None
        card = row["card_id"] if row is not None else self.legend_by_elimination(side) if self.legend_rule and side else None
        if card is None:
            return None
        if card not in self.masks:
            if self.tokens is None:
                self.tokens = priors.token_rows(self.rows)[0]
            mask = decklist.list_mask(self.catalogue(), self.decks, card) if self.decks else None
            self.masks[card] = mask if mask is not None else priors.legend_mask(self.rows, [card], self.tokens, runes=True)[0]
        return self.masks[card]

    def legend_by_elimination(self, side: str) -> str | None:
        """With two lists given, once the other player's legend is one list's, this side's legend is the other list's:
        its cards are read against that list, its own legend under dice or not read yet. None otherwise: lists for
        another match name neither legend on the table, and then neither is used."""
        if len(self.decks) != 2:
            return None
        lg = self.legends.get(self.other_side(side))
        row = self.row_of.get(lg["printing_id"]) if lg is not None else None
        if row is None:
            return None
        named = [set(d.legends()) for d in self.decks]
        theirs = [k for k in (0, 1) if row["card_id"] in named[k]]
        if len(theirs) != 1 or len(named[1 - theirs[0]]) != 1:
            return None
        return next(iter(named[1 - theirs[0]]))

    def with_card(self, allowed: np.ndarray, card: str) -> np.ndarray:
        """A side's rows (`allowed`) and the rows of `card`, the card a track is named: a card named on one side stays
        itself wherever it goes, a unit moved to a battlefield across the midline or taken by the other player. (Both
        players' cards on the battlefield strip would name the cards of a hand held over it: the side's rows stay.)"""
        if card not in self.card_rows:
            self.card_rows[card] = self.cards == card
        return allowed | self.card_rows[card]

    def catalogue(self):
        """The gallery's rows, indexed for decklists (`decklist.Catalogue`, made once, when a list is read)."""
        from ..decklist import Catalogue

        if self._cat is None:
            self._cat = Catalogue(self.rows)
        return self._cat

    def set_lists(self, decks) -> None:
        """The decklists given, read through `catalogue()`: from the next read on, a side whose pinned legend a list
        names competes with that list's cards; the other sides keep the legend rule. None: the rule everywhere."""
        self.decks = list(decks)
        self.masks.clear()  # the legend masks, and the other side's list by elimination, follow the lists

    def identify(self, crops: Sequence[Image.Image], sides: Sequence[str] | None = None,
                 keeps: Sequence[str | None] | None = None) -> list[list[tuple[str, float, float, int]]]:
        """Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first.
        All four turns go in one batch: which way up a card lies is unknown (exhausted, opponent side).
        `sides[n]` is where crop n lies: under the legend rule, the rows its side's legend rules out score -inf
        before the best 60 are taken, and the softmax runs over the cards left. `keeps[n]`, when given, is the card crop
        n's track is named: it competes too, wherever the crop lies (`with_card`)."""
        if not crops:
            return []
        views = [c.rotate(r, expand=True) if r else c for c in crops for r in ROTATIONS]
        emb = self.enc.embed(views).reshape(len(crops), len(ROTATIONS), -1)
        out = []
        for n, c in enumerate(crops):
            level = self.gallery.levels[self.gallery.level_for(max(c.size))]
            sims = (emb[n] @ level.T).max(axis=0)
            allowed = self.allowed(sides[n]) if sides is not None else None
            if allowed is not None and keeps is not None and keeps[n] is not None:
                allowed = self.with_card(allowed, keeps[n])
            if allowed is not None:
                sims = np.where(allowed, sims, -np.inf)
            scores: dict[str, tuple[float, int]] = {}
            # stable: equal scores in gallery order, as the engine's argsortDescending takes them
            for i in np.argsort(-sims, kind="stable")[:60]:
                if not np.isfinite(sims[i]):
                    continue  # a row the legend rules out
                card = self.cards[i]
                if card not in scores:
                    scores[card] = (float(sims[i]), int(i))
            if not scores:
                out.append([])
                continue
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
        keeps = [tr.named if tr.named and tr.kind != "Battlefield" else None for tr in owners]
        for tr, cands in zip(owners, self.identify(crops, [tr.side for tr in owners], keeps)):
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
            if self.scene.in_overlay((bx0 + bx1) / 2, (by0 + by1) / 2, w, h) or self.scene.in_suspect((bx0 + bx1) / 2, (by0 + by1) / 2, w, h):
                continue  # a webcam or a banner changing, laid over the table: not a card
            gx, gy = (bx1 - bx0) * 0.08, (by1 - by0) * 0.08
            box = (max(0, bx0 - gx), max(0, by0 - gy), min(w, bx1 + gx), min(h, by1 + gy))
            region = frame.crop(tuple(round(v) for v in box))
            if min(region.size) < 0.45 * card_long * 63 / 88 or max(region.size) > 2.2 * card_long:
                continue  # a die or counter, or a whole area at once: not one card
            if region.width > region.height * 1.15:
                region = region.rotate(90, expand=True)  # cards stand portrait
            if detail(region) < FACE_DOWN_DETAIL:
                continue  # a face-down card: never identified
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            side = self.layout.side(cx, cy, w, h)  # before the read: the side's legend rules its candidates
            # ponytail: the whole changed region is read as one card; the visible-band matcher (stacks.py) for cards put on stacks
            cands = self.identify([region], [side])[0]
            if not cands:
                continue
            card, p, sc, i = cands[0]
            r = self.rows[i]
            if p < self.gate_p or r.get("type") in QUIET + STATIC or self.recently_played(t, card, cx, cy):
                continue
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
            self.announced[self.flashes[-1]["id"]] = (cx, cy, r["name"], r["printing_id"], side)
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
        for tr in self.tracks.values():
            if tr.free_since is not None:
                tr.free_since += dt
        self.rune_counts = {side: [(pt + dt, n, ex) for pt, n, ex in h] for side, h in self.rune_counts.items()}
        self.pending, self.flashes = [], []

    def cut(self, t: float) -> None:
        """The view is framed anew: cards found again by name re-anchor the rest (`reanchor`), the gate starts
        over, and the cards first seen now were on the table already, not played."""
        self.cut_at = t
        self.anchor_base = {k: tr.box for k, tr in self.tracks.items()}
        self.anchor_pairs = []
        self.still = StillTable()  # another view of the table: its still picture is taken again
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
        self.frame_wh = (w, h)
        dt = t - self.last_t if self.last_t is not None else 0.0
        self.last_t = t
        boxes: list[CardBox] | None = None

        def count() -> int:  # the scene asks while it learns a broadcast with no known mat colour, or learns it again:
            nonlocal boxes   # the card-sized boxes (a close-up's cards are bigger)
            if boxes is None:
                boxes = self.find(t, image)
            px = self.layout.card_px(h)
            return sum(1 for b in boxes if b.long_px <= SIZE_MAX * px)

        on = self.scene.on_table(t, image, count)
        withdrawn = self.drop_overlaid(t, w, h) if self.scene.overlay_n != self.overlay_n else []
        if not on:
            if not self.away:
                self.away, self.before_away = True, set(self.prev_seen)
            self.pause(dt)
            state = self.state(t, w, h)
            for tr in state["tracks"]:
                tr["hidden"] = True  # the video is not the table: list the board, draw nothing on it
            state["status"], state["message"] = "away", "the table camera is off; nothing is looked at until it is back"
            return state, withdrawn
        back, self.away = self.away, False
        small = StillTable.small(image)
        self.still.feed(t, small)
        if boxes is None:
            boxes = self.find(t, image)
        self.scene.saw_cards(t, count())  # what a view must show to be learnt as the table camera again
        boxes = self.keep(boxes, image, w, h)
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
        table, still = self.layout.box(w, h), HAND_STILL * self.layout.card_px(h)
        for tr in seen:  # a card in a hand is not read, and not shown until it has been put down (D-005)
            if self.scene.in_suspect(*tr.box.centre, w, h) or \
                    hand_share(image, tr.box, table, [o.box for o in seen if o is not tr], self.still.bg, small) >= HAND_SKIN:
                tr.free_since = tr.free_at = None
            elif tr.free_since is None or math.dist(tr.box.centre, tr.free_at) > still:
                tr.free_since, tr.free_at = t, tr.box.centre  # out of the hand, or moved since: still from now
            tr.placed = tr.placed or self.put_down(t, tr)
        todo = sorted((tr for tr in seen if tr.hits >= 2 and self.put_down(t, tr) and self.due(tr, t)),
                      key=lambda tr: (tr.reads > 0, tr.last_read))[:budget]
        self.read(t, frame, todo)
        tr_ = time.perf_counter()
        events = withdrawn + self.announce(t) + self.watch(t, image, frame)
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
        kind = self.rows[self.first_row[top[0][0]]].get("type")
        if not tr.pinned and kind == "Legend" and self.side_legend(tr):
            return "unsure", p0, guesses  # one player, one legend: another outline of it, or a card misread as one
        if not tr.pinned and kind == "Battlefield" and not self.in_strip(*tr.box.centre):
            # off the battlefield strip, a battlefield only once the reads agree: a card turned sideways is not one
            return ("named" if tr.reads >= STRIP_READS and p0 >= self.min_p else "unsure"), p0, guesses
        named = tr.pinned or p0 >= self.sure_p or (p0 >= self.min_p and tr.reads >= 2)
        if not named and tr.reads >= 4 and p0 >= 0.3 and kind == "Legend" and not self.other_legend(tr.side, top[0][0]) \
                and self.listed_legend(top[0][0]):
            named = True  # a legend, read the same way four times: one a player, and its frame is like no other card's
        return ("named" if named else "unsure"), p0, guesses

    def listed_legend(self, card: str) -> bool:
        """No list given, or a given list names this legend: with lists, another legend needs a sure read to be named."""
        return not self.decks or any(card in d.legends() for d in self.decks)

    def announce(self, t: float) -> list[dict]:
        """'played' when a card is first named, 'moved' when a named card that just vanished is named again
        elsewhere (it keeps its first id). A card out of sight keeps its track: an unnamed one `forget_s`,
        a named one `KEEP_S` or as long as something lies on it, a legend or battlefield all game. Then a
        named card becomes a ghost (see `ghosts`). Runes, legends and battlefields are never announced, and nor are the
        cards of a burst: BURST of them first named within BURST_S (a graphic of a deck, a view framed anew)."""
        events = []
        played: list[tuple[Track, dict]] = []  # this step's plays, kept back until every card first named now is counted
        self.ghosts = [gh for gh in self.ghosts if t - gh["t"] < 60]
        self.first_named = [ft for ft in self.first_named if t - ft < BURST_S]
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
            if tr.pinned and tr.kind == "Battlefield" and self.misread_battlefield(tr):
                tr.pinned = False  # read once as a battlefield, as another card since: named again as that card
            if not tr.pinned and tr.kind == "Battlefield" and tr.named and not hidden_now(t, tr) and self.in_strip(*tr.box.centre) \
                    and (o := self.pinned_twin(t, tr)) is not None and hidden_now(t, o):
                o.box, o.last, o.hits = tr.box, tr.last, o.hits + tr.hits
                del self.tracks[tr.id]  # a pinned battlefield's second outline, in sight where it is not: the pin goes there
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
                strip = self.in_strip(*tr.box.centre)
                if tr.kind == "Battlefield" and strip and (o := self.pinned_twin(t, tr)) is not None:
                    if hidden_now(t, o):  # the battlefield was moved: its pin follows it, under its first id
                        o.box, o.last, o.hits = tr.box, tr.last, o.hits + tr.hits
                        del self.tracks[tr.id]
                    continue  # or the same battlefield outlined again: not a second one, and not drawn (state)
                if tr.kind in STATIC and (tr.kind != "Battlefield" or strip) \
                        and not (tr.kind == "Legend" and (self.other_legend(tr.side, tr.named) or self.side_legend(tr) is not None)):
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
                if not changed:
                    self.first_named.append(t)  # a card new to the board
                if tr.first - (self.t0 or 0.0) < self.settle_s and not changed:
                    continue  # on the table when we tuned in, not played now
                if after_cut and tr.first - self.cut_at < self.settle_s + 2:
                    continue  # on the table when the view changed
                if changed:
                    events.append(self.event(t, "changed", f"{g[0]['name']} (read again)", tr, g[0]["printing_id"]))
                elif not self.recently_played(t, tr.named, *tr.box.centre, skip=tr.id):
                    self.plays.append((t, tr.named, *tr.box.centre))
                    played.append((tr, self.event(t, "played", f"{g[0]['name']} played", tr, g[0]["printing_id"])))
                else:
                    self.plays.append((t, tr.named, *tr.box.centre))
        if len(self.first_named) < BURST:
            for tr, ev in played:
                self.announced[tr.id] = (*tr.box.centre, ev["text"][:-len(" played")], ev["printing_id"], tr.side)
                events.append(ev)
        return events

    def event(self, t: float, kind: str, text: str, tr: Track, pid: str | None) -> dict:
        return {"t": round(t, 2), "kind": kind, "text": text, "printing_id": pid, "track": tr.id, "side": tr.side}

    def ready_upright(self) -> bool:
        """Whether a ready card stands upright in the picture. From the player's seat, a ready card points straight at
        them and an exhausted (used) one lies across, along their edge. Battlefields, printed landscape and never
        exhausted, lie along the edges too, so a ready card stands at right angles to them; before one is named, the
        layout says where the players sit (left and right: a ready card lies across the picture)."""
        fields = [tr for tr in self.tracks.values() if tr.pinned and tr.kind == "Battlefield"]
        if fields:
            return sum(1 for tr in fields if upright(tr.box)) * 2 < len(fields)
        return self.layout.split == "horizontal"

    @staticmethod
    def put_down(t: float, tr: Track) -> bool:
        """Out of a hand and still for HAND_FREE_S: a card on the table, not one held over it."""
        return tr.free_since is not None and t - tr.free_since >= HAND_FREE_S

    def runes_seen(self, t: float, side: str, ready: bool) -> tuple[int, int]:
        """A player's runes in this frame, one per card (the newest of two tracks on one card), and how many are exhausted."""
        px = self.layout.card_px(self.frame_wh[1])
        kept: list[Track] = []
        for tr in sorted((tr for tr in self.tracks.values() if tr.kind == "Rune" and tr.side == side and tr.hits >= 2
                          and t - tr.last <= RUNE_SEEN_S and tr.box.aspect <= RUNE_ASPECT and tr.box.long_px <= RUNE_LONG * px),
                         key=lambda tr: (-tr.last, -tr.hits)):
            if all(math.dist(tr.box.centre, k.box.centre) > RUNE_APART * px for k in kept):
                kept.append(tr)
        return len(kept), sum(1 for tr in kept if upright(tr.box) != ready)

    def rune_like(self, tr: Track) -> bool:
        """Read as a rune: named one, or a rune on RUNE_SHARE of its reads."""
        if tr.kind == "Rune":
            return True
        return tr.reads >= 2 and sum(p for c, p in tr.prob.items() if self.type_of.get(c) == "Rune") >= RUNE_SHARE * tr.reads

    def runes_now(self, t: float, side: str) -> int:
        """A player's runes in this frame: the boxes read as runes, the unnamed card-sized boxes beside them turned their
        way (a stack's covered strips), and the runes a stack's wider gaps hide (`hidden_runes`)."""
        px = self.layout.card_px(self.frame_wh[1])
        now = [tr for tr in self.tracks.values() if tr.side == side and tr.hits >= 2 and tr.last == t
               and tr.box.aspect <= RUNE_ASPECT and tr.box.long_px <= RUNE_LONG * px]
        runes = [tr for tr in now if self.rune_like(tr)]
        while True:
            more = [tr for tr in now if tr.kind == "" and tr not in runes and abs(tr.box.long_px / px - 1) <= RUNE_SIZE
                    and any(math.dist(tr.box.centre, r.box.centre) < RUNE_JOIN * px
                            and turn_apart(tr.box.angle_deg, r.box.angle_deg) <= RUNE_TURN for r in runes)]
            if not more:
                break
            runes += more
        return len(runes) + sum(hidden_runes(g, px) for g in stacks_of([tr.box for tr in runes], RUNE_LINK * px))

    def runes(self, t: float, side: str, ready: bool) -> dict:
        """A player's runes on the table: counted, never named, and how many are exhausted (used this turn), over the
        last frames (`RUNE_WINDOW_S`); off the table camera it holds. The count is this frame's (`runes_now`), the
        exhausted ones those named runes seen in the last second that lie across (`runes_seen`)."""
        recent = self.rune_counts.setdefault(side, [])
        if not self.away:
            n, ex = self.runes_now(t, side), self.runes_seen(t, side, ready)[1]
            if recent and recent[-1][0] >= t:
                recent.pop()  # the state asked again for this frame
            recent.append((t, n, ex))
            while recent[0][0] <= t - RUNE_WINDOW_S:
                recent.pop(0)
        if not recent:
            return {"count": 0, "exhausted": 0}
        counts = sorted(n for _, n, _ in recent)
        count = counts[min(len(counts) - 1, 3 * len(counts) // 4)]
        newest = recent[-1][0]
        used = sorted(ex for pt, _, ex in recent if pt > newest - RUNE_EXHAUSTED_S)
        return {"count": count, "exhausted": min(count, used[len(used) // 2])}

    def state(self, t: float, w: int, h: int) -> dict:
        tracks = []
        under = self.stacks(t)
        ready = self.ready_upright()
        for tr in self.tracks.values():
            if tr.hits < 2 or not tr.placed:
                continue  # seen once (maybe the detector's slip, a box between two cards), or only ever in a hand
            hidden = t - tr.last > (PIN_HIDE_S if tr.pinned else 1.0)
            if hidden and not tr.named:
                continue
            state, p, g = self.label(tr)
            if (lg := self.side_legend(tr)) is not None and not tr.pinned and g and g[0]["card_id"] == lg.named:
                continue  # another outline of the side's legend (its case, the die on it): not a card
            if self.twin(t, tr):
                continue  # the same card outlined again (a sleeve's or a toploader's edge): drawn once
            if not tr.pinned and tr.kind == "Battlefield" and (o := self.pinned_twin(t, tr)) is not None and not hidden_now(t, o):
                continue  # a pinned battlefield's second outline: drawn once
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
                "players": [{"side": s, "label": f"Player {k + 1}", "legend": self.legends.get(s), "runes": self.runes(t, s, ready)}
                            for k, s in enumerate(self.layout.sides())],
                "layout": {"name": self.layout.name, "table": list(self.layout.table)},
                "tracks": tracks}
