# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The layout sampler: plausible 1v1 Riftbound boards on a table plane, in millimetres.

The table follows Tournament Rules 508 (docs/research/01 §1.3). Runes sit closest to each player
and everything else further in. The legend and the chosen champion sit on one side, the main deck
and trash on one side, and the rune deck on the other. Ready cards face their controller, and
exhausted cards are turned 90°, all the same way. The two battlefields share the centre line.

Coordinates: x to the right, y towards the near player, origin at the table centre. Everything is
drawn in list order, so a later card lies on an earlier one. `Pile` records which cards overlap
by design (rune columns, fanned rows, tucked gear, decks), so the stacks the M1 detector must
learn are dense, not left to chance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

CARD_W_MM, CARD_H_MM = 63.0, 88.0
TABLE_W_MM, TABLE_H_MM = 1000.0, 900.0   # the surface the camera can see, mats and table around them
MAT_W_MM, MAT_H_MM = 610.0, 355.0         # one player's playmat (24 × 14 in)


@dataclass
class Instance:
    """One physical card on the board."""
    id: int
    row: int | None                   # catalogue row, or None for a card back
    face_up: bool
    zone: str                         # legend, champion, base, runes, battlefield, facedown, trash, deck, rune_deck, chain
    controller: str                   # near | far
    centre: tuple[float, float]       # mm
    angle: float                      # degrees counter-clockwise from upright as seen by the near player
    landscape: bool = False
    exhausted: bool = False
    pile: int | None = None
    pile_index: int = 0
    sleeve: tuple[int, int, int] | None = None
    foil: bool = False
    glare: bool = False
    back: str = "sleeve"              # face-down look: a plain sleeve back or a printed pattern


@dataclass
class Occluder:
    kind: str                         # die, counter, marker, hand
    centre: tuple[float, float]
    size: float                       # mm
    angle: float
    on: int | None = None             # the card it sits on, when it sits on one
    value: int = 0                    # die face or counter number
    colour: tuple[int, int, int] = (240, 240, 236)


@dataclass
class Board:
    instances: list[Instance] = field(default_factory=list)
    occluders: list[Occluder] = field(default_factory=list)
    piles: dict[int, str] = field(default_factory=dict)     # pile id -> kind
    turn: int = 1


def _by(rows: Sequence[dict], rng: np.random.Generator, types: Sequence[str], domains: Sequence[str] = (),
        variant: str | None = None) -> int | None:
    """A random row of one of `types`, of the legend's domains when any match. Any row when the
    catalogue has none of `types` (the procedural test catalogue); None for an empty catalogue."""
    pool = [i for i, r in enumerate(rows) if r.get("type") in types and (variant is None or r.get("variant") == variant)]
    if not pool:
        pool = list(range(len(rows)))
    if domains:
        own = [i for i in pool if set(rows[i].get("domains") or []) & set(domains)]
        pool = own or pool
    return int(pool[rng.integers(len(pool))]) if pool else None


class _Builder:
    def __init__(self, rows: Sequence[dict], rng: np.random.Generator):
        self.rows, self.rng, self.board = rows, rng, Board()
        self.next_pile = 0

    def pile(self, kind: str) -> int:
        self.next_pile += 1
        self.board.piles[self.next_pile] = kind
        return self.next_pile

    def add(self, row: int | None, face_up: bool, zone: str, who: str, x: float, y: float, *, exhausted: bool = False,
            pile: int | None = None, index: int = 0, sleeve=None, jitter: float = 3.0, back: str = "sleeve") -> Instance:
        rng = self.rng
        land = row is not None and self.rows[row].get("orientation") == "landscape"
        facing = 0.0 if who == "near" else 180.0       # ready cards face their controller (508.10)
        turn = -90.0 if exhausted else 0.0            # exhausted: a quarter turn, the same way for everyone
        inst = Instance(id=len(self.board.instances), row=row if face_up else None, face_up=face_up, zone=zone, controller=who,
                        centre=(float(x), float(y)), angle=facing + turn + float(rng.uniform(-jitter, jitter)),
                        landscape=land, exhausted=exhausted, pile=pile, pile_index=index, sleeve=sleeve,
                        foil=face_up and rng.random() < 0.12, glare=rng.random() < 0.2, back=back)
        self.board.instances.append(inst)
        return inst


def sample_board(rows: Sequence[dict], rng: np.random.Generator) -> Board:
    """A plausible mid-game board for two players. `rows` are catalogue rows (see catalog.py)."""
    b = _Builder(rows, rng)
    board = b.board
    board.turn = int(rng.integers(1, 11))
    for who in ("near", "far"):
        s = 1.0 if who == "near" else -1.0            # y points at the near player
        side = 1.0 if rng.random() < 0.5 else -1.0    # which side the legend sits on (508.6)
        legend = _by(rows, rng, ["Legend"])
        domains = (rows[legend].get("domains") or []) if legend is not None else []
        main_sleeve = tuple(int(c) for c in rng.integers(0, 256, 3))
        rune_sleeve = tuple(int(c) for c in rng.integers(0, 256, 3)) if rng.random() < 0.6 else main_sleeve

        # Main deck and trash together on one side, the rune deck on the other (508.4-508.5).
        _deck(b, "deck", who, -side * 300.0, s * 205.0, int(rng.integers(10, 30)), main_sleeve)
        _trash(b, who, -side * 300.0, s * 315.0, min(20, int(rng.poisson(1.5 * board.turn))), domains, main_sleeve)
        _deck(b, "rune_deck", who, side * 300.0, s * 395.0, max(0, 12 - 2 * board.turn), rune_sleeve)

        # The legend, then the chosen champion outermost (508.6); players often keep a die on the legend.
        if legend is not None:
            leg = b.add(legend, True, "legend", who, side * 215.0, s * 205.0, exhausted=rng.random() < 0.15, sleeve=main_sleeve)
            if rng.random() < 0.6:
                _die(b, leg)
        if rng.random() < 0.5:
            champ = _by(rows, rng, ["Unit"], domains)
            if champ is not None:
                b.add(champ, True, "champion", who, side * 290.0, s * 205.0, sleeve=main_sleeve)

        # Runes, closest to the player: spread out, fanned, or in short columns.
        n_runes = int(min(12, 2 * board.turn + (1 if who == "far" and rng.random() < 0.5 else 0)))
        _runes(b, who, s, n_runes, domains, rune_sleeve)

        # Units and gear in the base, between the legend and the decks.
        _base(b, who, s, -side * 40.0, int(rng.integers(0, 7)), domains, main_sleeve)

    # Two battlefields on the centre line, units of both players at each, and a facedown slot.
    _battlefields(b)
    if rng.random() < 0.1:  # a spell on the chain, for a moment
        sp = _by(rows, rng, ["Spell"])
        if sp is not None:
            b.add(sp, True, "chain", "near" if rng.random() < 0.5 else "far", float(rng.uniform(-60, 60)), float(rng.uniform(-40, 40)), jitter=12.0)
    _hands(b)
    return board


def _deck(b: _Builder, zone: str, who: str, x: float, y: float, n: int, sleeve) -> None:
    if n <= 0:
        return
    p = b.pile(zone)
    back = "pattern" if b.rng.random() < 0.1 else "sleeve"
    for i in range(min(n, 8)):  # the top few: their edges make the pile's thickness
        b.add(None, False, zone, who, x + b.rng.normal(0, 0.8), y + b.rng.normal(0, 0.8), pile=p, index=i, sleeve=sleeve,
              jitter=1.5, back=back)


def _trash(b: _Builder, who: str, x: float, y: float, n: int, domains, sleeve) -> None:
    if n <= 0:
        return
    p = b.pile("trash")
    for i in range(min(n, 6)):
        r = _by(b.rows, b.rng, ["Spell", "Unit", "Gear"], domains)
        if r is not None:
            b.add(r, True, "trash", who, x + b.rng.normal(0, 2.0), y + b.rng.normal(0, 2.0), pile=p, index=i, sleeve=sleeve, jitter=5.0)


def _die(b: _Builder, on: Instance) -> None:
    rng = b.rng
    white = rng.random() < 0.75
    colour = (240, 240, 236) if white else tuple(int(c) for c in rng.integers(20, 230, 3))
    cx, cy = on.centre
    b.board.occluders.append(Occluder("die", (cx + rng.normal(0, 8), cy + rng.normal(0, 12)), float(rng.uniform(14, 17)),
                                      float(rng.uniform(0, 90)), on=on.id, value=int(rng.integers(1, 7)), colour=colour))


def _runes(b: _Builder, who: str, s: float, n: int, domains, sleeve) -> None:
    rng = b.rng
    if n <= 0:
        return
    runes = [r for r in (_by(b.rows, rng, ["Rune"], [d]) for d in (domains or [None])) if r is not None] or \
        [r for r in [_by(b.rows, rng, ["Rune"])] if r is not None]
    style = rng.choice(["spread", "fan", "columns"], p=[0.3, 0.3, 0.4])
    y0 = s * 395.0
    spent = int(rng.integers(0, n + 1))  # exhausted to pay this turn
    if style == "columns":  # the M0 broadcasts: short overlapping columns, the top of each rune showing
        per = int(rng.integers(2, 5)); step = float(rng.uniform(16, 32)); gap = float(rng.uniform(68, 80))
        k, ncol = 0, (n + per - 1) // per
        for c in range(ncol):
            p = b.pile("rune_column")
            x = (c - (ncol - 1) / 2) * gap
            for j in range(min(per, n - k)):
                # nearer the player is on top: each rune covers the lower part of the one before
                b.add(runes[int(rng.integers(len(runes)))], True, "runes", who, x + rng.normal(0, 1.5), y0 - s * (40 - j * step),
                      exhausted=k < spent and rng.random() < 0.5, pile=p, index=j, sleeve=sleeve, jitter=2.5)
                k += 1
    else:
        step = float(rng.uniform(22, 42)) if style == "fan" else float(rng.uniform(68, 78))
        p = b.pile("rune_fan") if style == "fan" else None
        for j in range(n):
            x = (j - (n - 1) / 2) * step * (1 if who == "near" else -1)
            b.add(runes[int(rng.integers(len(runes)))], True, "runes", who, x + rng.normal(0, 1.5), y0 + rng.normal(0, 2),
                  exhausted=j < spent, pile=p, index=j, sleeve=sleeve, jitter=3.0)


def _base(b: _Builder, who: str, s: float, x0: float, n: int, domains, sleeve) -> None:
    rng = b.rng
    if n <= 0:
        return
    fanned = rng.random() < 0.4
    step = float(rng.uniform(30, 50)) if fanned else float(rng.uniform(68, 80))
    p = b.pile("base_fan") if fanned else None
    y = s * 300.0
    for j in range(n):
        x = x0 + (j - (n - 1) / 2) * step
        r = _by(b.rows, rng, ["Unit"], domains) if rng.random() < 0.8 else _by(b.rows, rng, ["Unit"], variant="token")
        if r is None:
            continue
        if rng.random() < 0.15:  # attached equipment, tucked under its unit so its bottom strip shows (01 §1.2)
            g = _by(b.rows, rng, ["Gear"], domains)
            if g is not None:
                gp = b.pile("gear")
                b.add(g, True, "base", who, x, y + s * float(rng.uniform(18, 30)), pile=gp, index=0, sleeve=sleeve)
                u = b.add(r, True, "base", who, x, y, exhausted=False, pile=gp, index=1, sleeve=sleeve)
                continue
        u = b.add(r, True, "base", who, x + rng.normal(0, 2), y + rng.normal(0, 4), exhausted=rng.random() < 0.4,
                  pile=p, index=j, sleeve=sleeve)
        if rng.random() < 0.08:
            _counter(b, u)


def _counter(b: _Builder, on: Instance) -> None:
    rng = b.rng
    cx, cy = on.centre
    kind = "counter" if rng.random() < 0.7 else "marker"
    b.board.occluders.append(Occluder(kind, (cx + rng.normal(0, 10), cy + rng.normal(0, 15)),
                                      float(rng.uniform(18, 24)) if kind == "counter" else float(rng.uniform(36, 46)),
                                      float(rng.uniform(-20, 20)), on=on.id, value=int(rng.integers(1, 5)),
                                      colour=tuple(int(c) for c in rng.integers(20, 200, 3))))


def _battlefields(b: _Builder) -> None:
    rng = b.rng
    for k, x in enumerate((-130.0, 130.0)):
        owner = "near" if k == 0 else "far"
        bf = _by(b.rows, rng, ["Battlefield"])
        if bf is not None:
            b.add(bf, True, "battlefield", owner, x + rng.normal(0, 3), rng.normal(0, 3), jitter=2.0)
        for who, s in (("near", 1.0), ("far", -1.0)):
            n = int(rng.choice([0, 0, 1, 1, 2, 3]))
            if n:
                p = b.pile("battlefield_units") if n > 1 else None
                step = float(rng.uniform(28, 55))
                for j in range(n):
                    r = _by(b.rows, rng, ["Unit"])
                    if r is not None:
                        b.add(r, True, "battlefield", who, x + (j - (n - 1) / 2) * step, s * float(rng.uniform(95, 115)),
                              exhausted=rng.random() < 0.5, pile=p, index=j)
            if rng.random() < 0.2:  # a hidden card in the battlefield's facedown slot: never identified (01 §1.5)
                b.add(None, False, "facedown", who, x + s * 85.0, s * 20.0, jitter=6.0)


SKIN = [(236, 188, 160), (224, 172, 138), (198, 140, 106), (160, 110, 80), (112, 76, 56), (74, 52, 40)]


def _hands(b: _Builder) -> None:
    """Hands reaching in from their player's edge; `centre` is the fingertips, `angle` where they point."""
    rng = b.rng
    for _ in range(int(rng.choice([0, 0, 1, 1, 2]))):
        s = 1.0 if rng.random() < 0.5 else -1.0
        tip = (float(rng.uniform(-320, 320)), s * float(rng.uniform(120, 400)))
        facing = 0.0 if s > 0 else 180.0   # fingers point away from their player
        b.board.occluders.append(Occluder("hand", tip, float(rng.uniform(85, 110)), facing + float(rng.normal(0, 25)),
                                          colour=SKIN[int(rng.integers(len(SKIN)))]))
