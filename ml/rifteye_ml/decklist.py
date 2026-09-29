# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Decklists as a prior on card identification: which printings a player can put on the table.

Each card on stream is named by its art against the whole gallery (`retrieval`). When the players'
decklists are public, they narrow that search. Four export formats are read (`parse`): deckbuilder
JSON, text with collector codes, the tourney sheet (names only, mapped through the catalogue, tolerant
of case and punctuation, each card keeping its section) and the deck code (`decode_code`, with
`encode_code` for round trips). Ids and names the catalogue cannot map are reported, never guessed.

Players often play another printing of a listed card: an alternate art, an overnumbered showcase
print, a signature print or a reprint in a later set. `expand` therefore takes every listed card to
all printings of its card_id in the gallery (in every language the gallery holds), adds any printing of
the same normalised name filed under another card_id (and reports it), and always allows tokens, which
no list names. `legend_domains` is the prior that needs no list: the cards whose domains fit the
legends on the table, and, with `runes`, runes only of those domains, as a rune deck follows its legend. `adjust` puts a prior on retrieval scores: hard (only these printings compete)
or soft (a bonus on their cosine scores).

A prior only helps name cards that lie face up on the table. It is never shown, and never used to
guess or reveal hidden cards: hands and face-down cards stay unprocessed (D-005). Nothing is counted
across matches (D-012). Decklists, crops and the numbers written here stay private (D-006).

    python -m rifteye_ml.decklist show --catalog catalog.jsonl deck.json
    python -m rifteye_ml.decklist check --catalog catalog.jsonl deck.json deck.txt deck-tourney.txt deck-code.txt
    python -m rifteye_ml.decklist evaluate --catalog catalog.jsonl --cache art --embed-cache embed-cache \\
        --crops real-crops/bcn-v1/crops --labels real-crops/bcn-v1/labels.csv \\
        --match final=12:18:30-12:44:00 --deck final=a.json --deck final=b.json --match other=03:29:00-03:57:30
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

BOARDS = ("main", "side")
KEEP_TYPES = ("Rune", "Battlefield")   # kept by the legend prior whatever their domains
COLORLESS = "Colorless"


# ---------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------

def normalise(name: str) -> str:
    """A name for matching: accents folded, case and apostrophes dropped, other punctuation a space.
    'Ornn - Blacksmith' and 'Ornn, Blacksmith' meet at 'ornn blacksmith'; letters of other scripts stay."""
    s = "".join(ch for ch in unicodedata.normalize("NFKD", name or "") if not unicodedata.combining(ch))
    s = re.sub(r"['’`]", "", s.casefold())
    return re.sub(r"[\W_]+", " ", s).strip()


def base_name(name: str) -> str:
    """The name without a trailing parenthetical: 'Recruit (ZN)' -> 'Recruit'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", name or "").strip()


class Catalogue:
    """Gallery rows indexed for decklists: by printing, by card_id and by normalised name.

    A legend's catalogue name is its title only ('Title', the champion in its tags), while lists write
    'Champion - Title' or 'Champion, Title', so each legend is also known as '<tag> <title>'."""

    def __init__(self, rows: Sequence[dict]):
        self.rows = list(rows)
        self.pid: dict[str, int] = {}
        self.card: dict[str, list[int]] = defaultdict(list)
        self.names: dict[str, set[str]] = defaultdict(set)
        self.bases: dict[str, set[str]] = defaultdict(set)
        self.kind: dict[str, str] = {}
        for i, r in enumerate(self.rows):
            self.pid.setdefault(r["printing_id"], i)
            self.card[r["card_id"]].append(i)
            self.kind.setdefault(r["card_id"], r.get("type") or "")
            name = r.get("name") or ""
            for key, index in ((normalise(name), self.names), (normalise(base_name(name)), self.bases)):
                if key:
                    index[key].add(r["card_id"])
            if r.get("type") == "Legend":
                for tag in r.get("tags") or []:
                    self.names[normalise(f"{tag} {name}")].add(r["card_id"])
        self.cards = np.array([r["card_id"] for r in self.rows])
        self.printings = np.array([r["printing_id"] for r in self.rows])
        self.tokens, self.token_extra = self._tokens()

    def _tokens(self) -> tuple[np.ndarray, list[str]]:
        """Tokens: printings marked token, every printing of a token's card, and printings whose name without
        its parenthetical is a token's (Origins printed some tokens as numbered cards, such as 'Recruit (ZN)')."""
        marked = [i for i, r in enumerate(self.rows) if r.get("variant") == "token" or (r.get("type") or "").lower() == "token"]
        cards = {self.rows[i]["card_id"] for i in marked}
        names = {normalise(base_name(self.rows[i].get("name") or "")) for i in marked} - {""}
        mask = np.zeros(len(self.rows), bool)
        mask[marked] = True
        for i, r in enumerate(self.rows):
            if r["card_id"] in cards or normalise(base_name(r.get("name") or "")) in names:
                mask[i] = True
        extra = sorted(self.rows[i]["printing_id"] for i in np.flatnonzero(mask) if i not in set(marked))
        return mask, extra

    def resolve_id(self, pid: str) -> str | None:
        """The printing a listed id names: as written, or with its set code in capitals."""
        pid = (pid or "").strip()
        if pid in self.pid:
            return pid
        m = re.fullmatch(r"([A-Za-z]+)-(.+)", pid)
        if m and f"{m[1].upper()}-{m[2]}" in self.pid:
            return f"{m[1].upper()}-{m[2]}"
        return None

    def resolve_name(self, name: str, types: Sequence[str] = ()) -> tuple[str | None, list[str]]:
        """The card a written name means, and every card it could mean. A name that means more than one card
        is settled by the section's card types (`types`), or left unmapped."""
        for key, index in ((normalise(name), self.names), (normalise(base_name(name)), self.bases)):
            found = sorted(index.get(key, ()))
            if types and len(found) > 1:
                found = [c for c in found if self.kind.get(c) in types] or found
            if len(found) == 1:
                return found[0], found
            if found:
                return None, found
        return None, []


# ---------------------------------------------------------------------------------------
# Decklists in four formats
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    card_id: str
    count: int
    board: str               # 'main' (the legend, champion, main deck, battlefields and runes) or 'side'
    section: str = ""        # the tourney sheet's section: legend, champion, maindeck, battlefields, runes, sideboard
    listed: str = ""         # the printing the list names, when its format gives one
    kind: str = ""           # the card's type in the catalogue


@dataclass
class Deck:
    entries: list[Entry]
    fmt: str
    unmapped: list[str] = field(default_factory=list)   # ids or names the catalogue does not know
    notes: list[str] = field(default_factory=list)

    def counts(self) -> Counter:
        """(board, card_id) -> copies."""
        out: Counter = Counter()
        for e in self.entries:
            out[(e.board, e.card_id)] += e.count
        return out

    def listed(self) -> Counter:
        """(board, printing_id) -> copies, for the formats that name printings."""
        out: Counter = Counter()
        for e in self.entries:
            if e.listed:
                out[(e.board, e.listed)] += e.count
        return out

    def card_ids(self, boards: Sequence[str] = BOARDS) -> set[str]:
        return {e.card_id for e in self.entries if e.board in boards}

    def legends(self) -> list[str]:
        return sorted({e.card_id for e in self.entries if e.kind == "Legend"})

    def battlefields(self) -> set[str]:
        return {e.card_id for e in self.entries if e.kind == "Battlefield"}


def _entry(cat: Catalogue, card_id: str, count: int, board: str, section: str = "", listed: str = "") -> Entry:
    return Entry(card_id, count, board, section, listed, cat.kind.get(card_id, ""))


def _count(v) -> int:
    n = int(v)
    if n < 1:
        raise ValueError(f"a count of {v!r}")
    return n


def parse_json(doc: dict, cat: Catalogue) -> Deck:
    """Deckbuilder JSON: deck['Main Board'] (with the legend, battlefields and runes) and deck['Side Board'],
    entries {id, count}."""
    deck = doc.get("deck", doc) if isinstance(doc, dict) else {}
    out = Deck([], "json")
    for key, board in (("Main Board", "main"), ("Side Board", "side")):
        for item in deck.get(key) or []:
            pid = cat.resolve_id(str(item.get("id", "")))
            if pid is None:
                out.unmapped.append(str(item.get("id", "")))
                continue
            out.entries.append(_entry(cat, cat.rows[cat.pid[pid]]["card_id"], _count(item.get("count", 1)), board, listed=pid))
    return out


_LINE_ID = re.compile(r"^\s*(\d+)\s*[xX]?\s+(.*?)\s*\(\s*([A-Za-z]{2,5}-[A-Za-z0-9]+\*?)\s*\)\s*$")
_LINE = re.compile(r"^\s*(\d+)\s*[xX]?\s+(.+?)\s*$")
_SIDE_HEADER = re.compile(r"^\s*side\s*board\s*:?\s*$", re.I)
_SECTIONS = {"legend": "legend", "legends": "legend", "champion": "champion", "champions": "champion",
             "maindeck": "maindeck", "main deck": "maindeck", "battlefield": "battlefields", "battlefields": "battlefields",
             "rune": "runes", "runes": "runes", "sideboard": "sideboard", "side board": "sideboard"}
_SECTION_HEADER = re.compile(r"^\s*(" + "|".join(sorted(_SECTIONS, key=len, reverse=True)) + r")\s*:\s*$", re.I)
SECTION_TYPES = {"legend": ("Legend",), "champion": ("Unit",), "battlefields": ("Battlefield",), "runes": ("Rune",),
                 "maindeck": ("Unit", "Spell", "Gear")}


def parse_text(text: str, cat: Catalogue) -> Deck:
    """'N Name (SET-NNN)' lines, then a 'Side Board:' header. The id decides; a name that disagrees is noted.
    A line without an id is mapped by its name."""
    out = Deck([], "text")
    board = "main"
    for line in text.splitlines():
        if not line.strip():
            continue
        if _SIDE_HEADER.match(line):
            board = "side"
            continue
        m = _LINE_ID.match(line)
        if m:
            pid = cat.resolve_id(m[3])
            if pid is not None:
                card = cat.rows[cat.pid[pid]]["card_id"]
                named, _ = cat.resolve_name(m[2])
                if named is not None and named != card:
                    out.notes.append(f"{m[3]} is written as {m[2]!r}, which names {named}; the id decides")
                out.entries.append(_entry(cat, card, _count(m[1]), board, listed=pid))
                continue
            named, _ = cat.resolve_name(m[2])
            if named is None:
                out.unmapped.append(line.strip())
                continue
            out.notes.append(f"{m[3]} is not in the catalogue; mapped by its name to {named}")
            out.entries.append(_entry(cat, named, _count(m[1]), board))
            continue
        m = _LINE.match(line)
        named, found = cat.resolve_name(m[2]) if m else (None, [])
        if named is None:
            out.unmapped.append(line.strip() + (f" (could be {', '.join(found)})" if found else ""))
            continue
        out.entries.append(_entry(cat, named, _count(m[1]), board))
    return out


def parse_tourney(text: str, cat: Catalogue) -> Deck:
    """The tourney sheet: sections 'Legend:', 'Champion:', 'MainDeck:', 'Battlefields:', 'Runes:', 'Sideboard:',
    lines 'N Name' without ids. Names go through the catalogue; a name that could mean two cards is settled
    by its section's card types. Every entry keeps its section; all but the sideboard are the main board."""
    out = Deck([], "tourney")
    section = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        h = _SECTION_HEADER.match(line)
        if h:
            section = _SECTIONS[h[1].casefold()]
            continue
        m = _LINE.match(line)
        if not m:
            out.unmapped.append(line.strip())
            continue
        named, found = cat.resolve_name(m[2], SECTION_TYPES.get(section, ()))
        if named is None:
            out.unmapped.append(line.strip() + (f" (could be {', '.join(found)})" if found else ""))
            continue
        out.entries.append(_entry(cat, named, _count(m[1]), "side" if section == "sideboard" else "main", section))
    if not section:
        out.notes.append("no section headers: every line was read as the main board")
    return out


def parse_code(code: str, cat: Catalogue) -> Deck:
    main, side = decode_code(code)
    out = Deck([], "code")
    for board, cards in (("main", main), ("side", side)):
        for pid, n in cards.items():
            got = cat.resolve_id(pid)
            if got is None:
                out.unmapped.append(pid)
                continue
            out.entries.append(_entry(cat, cat.rows[cat.pid[got]]["card_id"], n, board, listed=got))
    return out


def detect_format(text: str) -> str:
    s = text.strip()
    if s.startswith("{"):
        return "json"
    if re.fullmatch(r"[A-Za-z2-7]{8,}", s):
        return "code"
    if any(_SECTION_HEADER.match(line) and not _SIDE_HEADER.match(line) and "side" not in line.casefold()
           for line in s.splitlines()):
        return "tourney"
    return "text"


def parse(text: str, cat: Catalogue, fmt: str | None = None) -> Deck:
    """A decklist in any of the four formats (detected unless `fmt` names it)."""
    fmt = fmt or detect_format(text)
    if fmt == "json":
        return parse_json(json.loads(text), cat)
    if fmt == "code":
        return parse_code(text, cat)
    if fmt == "tourney":
        return parse_tourney(text, cat)
    if fmt == "text":
        return parse_text(text, cat)
    raise ValueError(f"unknown decklist format {fmt!r}")


def read_deck(path: str | Path, cat: Catalogue) -> Deck:
    return parse(Path(path).read_text(encoding="utf-8"), cat)


# ---------------------------------------------------------------------------------------
# The deck code
# ---------------------------------------------------------------------------------------

CODE_HEADER = 0x13                                          # format 1, version 3
CODE_SETS = {0: "OGN", 1: "OGS", 3: "SFD", 4: "UNL", 5: "VEN"}  # verified on two exported lists; 2 is unknown
MAIN_COUNTS = tuple(range(12, 0, -1))
SIDE_COUNTS = tuple(range(3, 0, -1))
_B32 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


class DeckCodeError(ValueError):
    """A deck code this reader does not understand. It refuses rather than guesses."""


def _varints(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= len(data):
            raise DeckCodeError("the code ends inside a number")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, pos


def _varint(value: int) -> bytes:
    if value < 0:
        raise DeckCodeError(f"cannot write {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def decode_code(code: str) -> tuple[dict[str, int], dict[str, int]]:
    """A deck code -> (main board, side board), printing id -> copies.

    Base32 (RFC 4648, no padding); byte 0 is 0x13 (format 1, version 3); then LEB128 numbers. For each count
    from 12 down to 1 (main board), then 3 down to 1 (side board): the number of groups with that count, and
    per group its size n, set index, variant, then n card numbers. One trailing 0 is accepted. Card id:
    '{SET}-{number:03d}'. An unknown set index (2 among them) or a non-zero variant is refused."""
    text = "".join((code or "").split()).upper()
    if not text or set(text) - _B32:
        raise DeckCodeError("not a deck code: base32 letters A-Z and 2-7 only")
    try:
        data = base64.b32decode(text + "=" * (-len(text) % 8))
    except (binascii.Error, ValueError) as e:
        raise DeckCodeError(f"not valid base32 ({e})") from None
    if not data or data[0] != CODE_HEADER:
        raise DeckCodeError(f"header byte {data[0] if data else None!r}: only format 1 version 3 (0x13) is known")
    pos = 1
    boards: list[dict[str, int]] = []
    for counts in (MAIN_COUNTS, SIDE_COUNTS):
        cards: dict[str, int] = {}
        for count in counts:
            groups, pos = _varints(data, pos)
            for _ in range(groups):
                n, pos = _varints(data, pos)
                set_index, pos = _varints(data, pos)
                variant, pos = _varints(data, pos)
                if set_index not in CODE_SETS:
                    known = ", ".join(f"{k} {v}" for k, v in CODE_SETS.items())
                    raise DeckCodeError(f"set index {set_index} is not known (known: {known}); refusing to guess")
                if variant != 0:
                    raise DeckCodeError(f"variant {variant} in {CODE_SETS[set_index]}: only 0 is known; refusing to guess")
                for _ in range(n):
                    number, pos = _varints(data, pos)
                    pid = f"{CODE_SETS[set_index]}-{number:03d}"
                    if pid in cards:
                        raise DeckCodeError(f"{pid} is listed twice in one board")
                    cards[pid] = count
        boards.append(cards)
    if pos < len(data):
        tail, pos = _varints(data, pos)
        if tail != 0 or pos != len(data):
            raise DeckCodeError(f"{len(data) - pos + 1} unexpected bytes after the side board")
    return boards[0], boards[1]


def encode_code(main: dict[str, int], side: dict[str, int] | None = None) -> str:
    """The deck code of a main and a side board (printing id -> copies), groups in set order and numbers
    ascending, as the exports write them. Only 'SET-NNN' printings of known sets fit in a code."""
    index = {v: k for k, v in CODE_SETS.items()}
    out = bytearray([CODE_HEADER])
    for cards, counts in ((main, MAIN_COUNTS), (side or {}, SIDE_COUNTS)):
        groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for pid, n in cards.items():
            m = re.fullmatch(r"([A-Z]+)-(\d+)", pid)
            if not m or m[1] not in index:
                raise DeckCodeError(f"{pid} has no place in a deck code (an unknown set, or a printing suffix)")
            if n not in counts:
                raise DeckCodeError(f"{pid}: {n} copies; a board holds 1 to {max(counts)}")
            groups[n][index[m[1]]].append(int(m[2]))
        for n in counts:
            sets = groups.get(n, {})
            out += _varint(len(sets))
            for s in sorted(sets):
                numbers = sorted(sets[s])
                out += _varint(len(numbers)) + _varint(s) + _varint(0) + b"".join(_varint(x) for x in numbers)
    out += _varint(0)
    return base64.b32encode(bytes(out)).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------

@dataclass
class Prior:
    """The printings a prior allows (one flag per gallery row) and how it was built."""
    mask: np.ndarray
    report: dict


def expand(cat: Catalogue, card_ids: Iterable[str], tokens: bool = True) -> Prior:
    """Every printing a player of these cards can put on the table: all printings of each card_id (any
    variant, set or language the gallery holds), any printing of the same normalised name filed under
    another card_id (reported in 'same_name'), and, with `tokens`, every token."""
    card_ids = sorted(set(card_ids))
    mask = np.zeros(len(cat.rows), bool)
    unknown, same_name = [], []
    for cid in card_ids:
        rows = cat.card.get(cid)
        if not rows:
            unknown.append(cid)
            continue
        mask[rows] = True
        for key in {normalise(base_name(cat.rows[i].get("name") or "")) for i in rows} - {""}:
            for other in sorted(cat.bases.get(key, set()) - {cid}):
                if other in card_ids:
                    continue
                mask[cat.card[other]] = True
                same_name.append({"listed": cid, "card_id": other, "printings": [cat.rows[i]["printing_id"] for i in cat.card[other]]})
    listed = mask.copy()
    if tokens:
        mask |= cat.tokens
    variants = Counter(cat.rows[i].get("variant", "") for i in np.flatnonzero(listed))
    languages = Counter(cat.rows[i].get("language", "") for i in np.flatnonzero(listed))
    return Prior(mask, {"cards": len(card_ids), "unknown": unknown, "printings": int(mask.sum()),
                        "printings_of_listed_cards": int(listed.sum()), "variants": dict(variants),
                        "languages": dict(languages), "same_name": same_name,
                        "tokens": int((mask & ~listed).sum()) if tokens else 0})


def legend_domains(cat: Catalogue, legends: Iterable[str], runes: bool = False) -> Prior:
    """The prior that needs no list: printings whose domains fit one of the legends' (colourless fits any),
    plus every battlefield and token whatever its domains, and every rune; with `runes`, only the runes of the
    legends' domains, since a player's rune deck follows their legend's domains."""
    legends = sorted(set(legends))
    keep = tuple(t for t in KEEP_TYPES if not (runes and t == "Rune"))
    fits: list[set[str]] = []
    for lg in legends:
        if lg not in cat.card:
            raise ValueError(f"legend {lg} is not in the catalogue")
        fits.append(set(cat.rows[cat.card[lg][0]].get("domains") or []) - {COLORLESS})
    mask = np.zeros(len(cat.rows), bool)
    for i, r in enumerate(cat.rows):
        doms = set(r.get("domains") or []) - {COLORLESS}
        mask[i] = r.get("type") in keep or cat.tokens[i] or any(doms <= f for f in fits)
    kept = Counter(cat.rows[i].get("type") or "" for i in np.flatnonzero(mask))
    total = Counter(r.get("type") or "" for r in cat.rows)
    return Prior(mask, {"legends": legends, "domains": [sorted(f) for f in fits], "printings": int(mask.sum()),
                        "kept_by_type": {t: [kept.get(t, 0), n] for t, n in sorted(total.items())},
                        "rule": "domains within one legend's (Colorless fits any); every Battlefield and token kept, and "
                                + ("Runes only of the legends' domains" if runes else "every Rune")})


def adjust(sims: np.ndarray, allowed: np.ndarray | None = None, bonus: float | None = None) -> np.ndarray:
    """Retrieval scores under a prior. `allowed` flags gallery rows (one row of flags, or one per query): with
    no `bonus` the prior is hard (only allowed printings compete), with one it is soft (+bonus on allowed)."""
    if allowed is None:
        return sims
    if bonus is None:
        return np.where(allowed, sims, -np.inf)
    return sims + bonus * allowed


def read(scores: np.ndarray, cards: np.ndarray, temperature: float, top: int = 60) -> tuple[int, float]:
    """One crop read as the live runner reads it (`live.pipeline.Recognizer.identify`): the best gallery rows
    rolled up into cards, each at its best score, then a softmax at `temperature`. Returns the best row and
    its card's probability."""
    order = np.argsort(-scores)[:top]
    best: dict[str, float] = {}
    for i in order:
        s = float(scores[i])
        if not np.isfinite(s):
            break
        best.setdefault(str(cards[i]), s)
    if not best:
        return int(order[0]), 0.0
    vals = np.array(list(best.values()))
    p = np.exp((vals - vals.max()) / temperature)
    return int(order[0]), float(p[0] / p.sum())


# ---------------------------------------------------------------------------------------
# Scoring a condition
# ---------------------------------------------------------------------------------------

def wilson(k: int, n: int) -> list[float]:
    from .reviewpack import wilson as w

    lo, hi = w(k, n)
    return [round(lo, 4), round(hi, 4)]


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "ci95": wilson(int(k), int(n)) if n else None}


def score(cat: Catalogue, sims: np.ndarray, truth: np.ndarray, tracks: Sequence[str], allowed: np.ndarray | None = None,
          bonus: float | None = None, temperature: float = 0.0347, sure: float = 0.85,
          reference: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Top-1 card and printing accuracy of `sims` (queries x gallery rows) under a prior, with Wilson 95%
    intervals, and the share of crops read surely (p >= `sure` at `temperature`) with their accuracy.
    `truth` holds each crop's gallery row. With `reference` (another condition's predicted rows) it also
    counts the crops this one fixes and breaks. Returns the numbers and the predicted rows."""
    adj = adjust(sims, allowed, bonus)
    reads = [read(row, cat.cards, temperature) for row in adj]
    pred = np.array([r[0] for r in reads], np.int64)
    p = np.array([r[1] for r in reads])
    card_ok = cat.cards[pred] == cat.cards[truth]
    print_ok = cat.printings[pred] == cat.printings[truth]
    named = p >= sure
    out = {"crops": int(len(truth)), "tracks": len(set(tracks)),
           "card": _rate(card_ok.sum(), len(truth)), "printing": _rate(print_ok.sum(), len(truth)),
           "sure": {**_rate(named.sum(), len(truth)), "right": _rate((named & card_ok).sum(), named.sum())}}
    errors: dict[tuple[str, str], list[str]] = defaultdict(list)
    for n in np.flatnonzero(~card_ok):
        errors[(cat.printings[truth[n]], cat.printings[pred[n]])].append(tracks[n])
    out["errors"] = [{"truth": t, "predicted": q, "crops": len(v), "tracks": sorted(set(v))}
                     for (t, q), v in sorted(errors.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    wrong_print: dict[tuple[str, str], int] = Counter()
    for n in np.flatnonzero(card_ok & ~print_ok):
        wrong_print[(cat.printings[truth[n]], cat.printings[pred[n]])] += 1
    out["printing_only_errors"] = [{"truth": t, "predicted": q, "crops": c} for (t, q), c in wrong_print.most_common()]
    if allowed is not None:
        inside = allowed[np.arange(len(truth)), truth] if allowed.ndim == 2 else allowed[truth]
        out["outside"] = {"crops": int((~inside).sum()), "cards": sorted({str(cat.cards[truth[n]]) for n in np.flatnonzero(~inside)})}
        if reference is not None:
            ref_ok = cat.cards[reference] == cat.cards[truth]
            lost = ~inside & ref_ok & ~card_ok
            out["outside"]["lost"] = {"crops": int(lost.sum()), "cards": sorted({str(cat.cards[truth[n]]) for n in np.flatnonzero(lost)})}
    if reference is not None:
        ref_ok = cat.cards[reference] == cat.cards[truth]
        out["fixed"] = int((~ref_ok & card_ok).sum())
        out["broken"] = int((ref_ok & ~card_ok).sum())
    return out, pred


def classify_truths(cat: Catalogue, truth: np.ndarray, decks: Sequence[Deck], sides: Sequence[str] | None = None,
                    deck_sides: Sequence[str] | None = None) -> list[dict]:
    """Every true printing among the crops against the lists: listed as printed, another printing of a listed
    card (and which variant), a same-name printing, a token, or outside every list. With `sides`, also whether
    it is in the list of the crop's own side."""
    listed = {e.listed for d in decks for e in d.entries if e.listed}
    cards = {c for d in decks for c in d.card_ids()}
    same = {s["card_id"] for d in decks for s in expand(cat, d.card_ids(), tokens=False).report["same_name"]}
    out: dict[tuple, dict] = {}
    for n, t in enumerate(truth):
        r = cat.rows[t]
        pid, cid = r["printing_id"], r["card_id"]
        if pid in listed:
            how = "listed"
        elif cid in cards:
            how = f"another printing of a listed card ({r.get('variant', '')}, {r.get('set_code', '')})"
        elif cid in same:
            how = "same name as a listed card, another card_id"
        elif cat.tokens[t]:
            how = "token"
        else:
            how = "outside the lists"
        own = ""
        if sides is not None and deck_sides is not None:
            mine = [d for d, s in zip(decks, deck_sides) if s == sides[n]]
            own = "own side's list" if any(cid in d.card_ids() for d in mine) else \
                "the other side's list" if cid in cards else ("token" if cat.tokens[t] else "no list")
        key = (pid, how, own)
        if key not in out:
            out[key] = {"printing": pid, "card_id": cid, "type": r.get("type", ""), "how": how, "crops": 0}
            if own:
                out[key]["per_side"] = own
        out[key]["crops"] += 1
    return sorted(out.values(), key=lambda x: (x["how"] == "listed", -x["crops"], x["printing"]))


# ---------------------------------------------------------------------------------------
# Crops, sides and the experiment
# ---------------------------------------------------------------------------------------

def hms_seconds(s: str) -> float:
    parts = [float(p) for p in s.strip().split(":")]
    return sum(v * 60 ** k for k, v in enumerate(reversed(parts)))


def load_crops(labels_csv: Path, meta_json: Path | None, cat: Catalogue, layout,
               frame_size: tuple[int, int] = (1920, 1080)) -> tuple[list[dict], list[tuple[float, str]]]:
    """The labelled crops whose printing the gallery holds, each with its VOD time, track and table side
    (`layout.side` on its centre), and the rows left out: (VOD time, label) for sleeves ('back'), cards the
    gallery lacks ('none', some tokens), unsure answers ('?') and anything else."""
    from PIL import Image

    from .reviewpack import frame_time

    meta = {}
    if meta_json is not None and meta_json.exists():
        meta = {m["file"]: m for m in json.loads(meta_json.read_text(encoding="utf-8"))}
    sizes: dict[str, tuple[int, int]] = {}
    crops, left_out = [], []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            pid = (rec.get("printing_id") or "").strip()
            if pid not in cat.pid:
                left_out.append((frame_time(rec["file"]), pid if pid in ("back", "none", "?") else "other"))
                continue
            c = {"file": rec["file"], "track": rec.get("track") or rec["file"], "t": frame_time(rec["file"]),
                 "truth": cat.pid[pid], "verdict": rec.get("verdict", ""), "side": "", "offset": None}
            m = meta.get(rec["file"])
            if m is not None and layout is not None:
                frame = m.get("frame") or ""
                if frame not in sizes:
                    try:
                        with Image.open(frame) as im:
                            sizes[frame] = im.size
                    except OSError:
                        sizes[frame] = frame_size
                w, h = sizes[frame]
                x, y = m["centre"]
                c["side"] = layout.side(x, y, w, h)
                x0, y0, x1, y1 = layout.box(w, h)
                mid = (x0 + x1) / 2 if layout.split != "horizontal" else (y0 + y1) / 2
                c["offset"] = round(((x if layout.split != "horizontal" else y) - mid) / max(1.0, float(m.get("long_px") or 1)), 3)
            crops.append(c)
    return crops, left_out


def gallery(enc, rows: Sequence[dict], art: str, cache: Path | None, scales: Sequence[int]):
    """The gallery pyramid, read from the embedding cache when it holds every level (then no art is loaded);
    the same files as `spike` and the live runner write."""
    from .retrieval import Pyramid
    from .spike import _cached_loader, _gallery, catalog_key

    key = catalog_key(rows, 512)
    if cache is not None:
        files = {s: cache / f"{hashlib.sha1(f'{enc.name}|{s}|{key}'.encode()).hexdigest()[:24]}.npy" for s in scales}
        if all(f.exists() for f in files.values()):
            return Pyramid({s: np.load(f) for s, f in files.items()})
    load = _cached_loader(art, 512)
    return _gallery(enc, [load(r) for r in rows], scales, cache, key)


def query_sims(enc, pyramid, crops_dir: Path, files: Sequence[str], cache: Path | None) -> np.ndarray:
    """Each crop's best score against every gallery row over its four turns, at the pyramid level nearest its
    size (the protocol of `spike real`). Query embeddings are kept in `cache` (private .npz) between runs."""
    from PIL import Image

    from .retrieval import ROTATIONS

    key = hashlib.sha1("|".join([enc.name] + [f"{f}:{(crops_dir / f).stat().st_size}" for f in files]).encode()).hexdigest()[:24]
    path = cache / f"queries-{key}.npz" if cache is not None else None
    if path is not None and path.exists():
        z = np.load(path)
        emb, longs = z["emb"], z["long"]
    else:
        ims = [Image.open(crops_dir / f).convert("RGB") for f in files]
        longs = np.array([max(im.size) for im in ims])
        views = [im.rotate(r, expand=True) if r else im for im in ims for r in ROTATIONS]
        emb = enc.embed(views).reshape(len(ims), len(ROTATIONS), -1)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, emb=emb, long=longs)
    sims = np.zeros((len(files), pyramid.rows), np.float32)
    groups: dict[int, list[int]] = defaultdict(list)
    for n, s in enumerate(longs):
        groups[pyramid.level_for(int(s))].append(n)
    for level, members in groups.items():
        sims[members] = np.einsum("nrd,gd->nrg", emb[members], pyramid.levels[level]).max(axis=1)
    return sims


def assign_sides(cat: Catalogue, decks: Sequence[Deck], truth: np.ndarray, sides: Sequence[str],
                 choices: Sequence[str]) -> tuple[list[str], dict]:
    """Which side of the table each of two lists plays on: the assignment most crops agree with, counting only
    cards that are in one list and not the other. Refuses when the crops do not tell."""
    if len(decks) != 2:
        raise ValueError("sides are found for two lists")
    only = [decks[0].card_ids() - decks[1].card_ids(), decks[1].card_ids() - decks[0].card_ids()]
    votes = [[sum(1 for t, s in zip(truth, sides) if s == side and cat.rows[t]["card_id"] in only[k]) for side in choices]
             for k in range(2)]
    straight = votes[0][0] + votes[1][1]
    swapped = votes[0][1] + votes[1][0]
    if straight == swapped:
        raise ValueError(f"the crops do not tell which side each list plays on (votes {votes})")
    order = [choices[0], choices[1]] if straight > swapped else [choices[1], choices[0]]
    return order, {"votes": {choices[0]: [votes[0][0], votes[1][0]], choices[1]: [votes[0][1], votes[1][1]]},
                   "agree": max(straight, swapped), "disagree": min(straight, swapped)}


@dataclass
class Match:
    name: str
    start: float
    end: float
    decks: list[tuple[str, Path, str | None]] = field(default_factory=list)   # label, path, forced side


def evaluate_match(cat: Catalogue, sims: np.ndarray, crops: list[dict], match: Match, choices: Sequence[str],
                   temperature: float, sure: float, bonuses: Sequence[float], foreign: Sequence[Deck] = (),
                   left_out: Sequence[tuple[float, str]] = ()) -> dict:
    """Every condition on one match's crops (a VOD time range): (a) the whole catalogue; (b) the legends'
    domains, both and per side, with every rune or with runes held to the legends' domains; with lists, (c) both lists, (d) each side its own list (strict, and with both
    lists' battlefields, which lie on the midline), (e) soft versions over `bonuses` and (f) each list only on the
    side whose legend it names; the oracle (each side's "list" is the cards it showed, an upper bound); and,
    without lists, another match's lists (`foreign`), hard, soft and by legend."""
    sel = [n for n, c in enumerate(crops) if match.start <= c["t"] <= match.end]
    if not sel:
        return {"crops": 0}
    S = sims[sel]
    truth = np.array([crops[n]["truth"] for n in sel], np.int64)
    tracks = [crops[n]["track"] for n in sel]
    sides = [crops[n]["side"] for n in sel]
    out: dict = {"range": [match.start, match.end], "crops": len(sel), "tracks": len(set(tracks)),
                 "crops_per_side": dict(Counter(sides)),
                 "left_out": dict(Counter(k for t, k in left_out if match.start <= t <= match.end)), "conditions": {}}
    kw = dict(temperature=temperature, sure=sure)
    base, ref = score(cat, S, truth, tracks, **kw)
    out["conditions"]["a_catalogue"] = base

    def per_side(masks: dict[str, np.ndarray]) -> np.ndarray:
        """One row of flags per crop: its side's; a crop of unknown side gets every side's."""
        anywhere = np.logical_or.reduce(list(masks.values()))
        return np.stack([masks.get(s, anywhere) for s in sides])

    # the legends on the table, per side, as the labels show them
    seen: dict[str, Counter] = defaultdict(Counter)
    for t, s in zip(truth, sides):
        if cat.rows[t].get("type") == "Legend":
            seen[s][cat.rows[t]["card_id"]] += 1
    legends = {s: c.most_common(1)[0][0] for s, c in seen.items()}
    out["legends_seen"] = {s: dict(c) for s, c in seen.items()}

    decks, deck_sides = [], []
    if match.decks:
        decks = [read_deck(p, cat) for _, p, _ in match.decks]
        forced = [s for _, _, s in match.decks]
        if all(forced):
            deck_sides, how = list(forced), {"forced": True}
        else:
            deck_sides, how = assign_sides(cat, decks, truth, sides, choices)
        out["decks"] = {label: {"side": s, "legend": d.legends(), "cards": len(d.card_ids()),
                                "unmapped": d.unmapped, "expansion": expand(cat, d.card_ids()).report}
                        for (label, _, _), d, s in zip(match.decks, decks, deck_sides)}
        out["sides_found_by"] = how
        for d, s in zip(decks, deck_sides):
            if s not in legends and d.legends():
                legends[s] = d.legends()[0]
                out.setdefault("legends_from_lists", {})[s] = d.legends()[0]

    # (b) the legend-domain prior: both legends', and each side its own legend's
    if len(legends) == len(choices):
        lp = legend_domains(cat, legends.values())
        out["legend_prior"] = lp.report
        out["conditions"]["b_legend_domains"], _ = score(cat, S, truth, tracks, lp.mask, reference=ref, **kw)
        own = {s: legend_domains(cat, [legends[s]]).mask for s in choices}
        out["conditions"]["b_legend_domains_per_side"], _ = score(cat, S, truth, tracks, per_side(own), reference=ref, **kw)
        # the same, with each rune held to the legends' domains: both, and each side its own legend's
        lr = legend_domains(cat, legends.values(), runes=True).mask
        out["conditions"]["b_legend_domains_runes"], _ = score(cat, S, truth, tracks, lr, reference=ref, **kw)
        own_r = {s: legend_domains(cat, [legends[s]], runes=True).mask for s in choices}
        out["conditions"]["b_legend_domains_per_side_runes"], _ = score(cat, S, truth, tracks, per_side(own_r),
                                                                         reference=ref, **kw)

    if decks:
        union = expand(cat, set().union(*(d.card_ids() for d in decks)))
        fields = set().union(*(d.battlefields() for d in decks))
        own_list = {s: expand(cat, set().union(*(d.card_ids() for d, ds in zip(decks, deck_sides) if ds == s))).mask
                    for s in choices}
        own_bf = {s: own_list[s] | expand(cat, fields, tokens=False).mask for s in choices}
        c = out["conditions"]
        c["c_both_lists"], _ = score(cat, S, truth, tracks, union.mask, reference=ref, **kw)
        c["d_own_list_strict"], _ = score(cat, S, truth, tracks, per_side(own_list), reference=ref, **kw)
        c["d_own_list"], _ = score(cat, S, truth, tracks, per_side(own_bf), reference=ref, **kw)
        for b in bonuses:
            c[f"e_soft_both_lists+{b:g}"], _ = score(cat, S, truth, tracks, union.mask, bonus=b, reference=ref, **kw)
            c[f"e_soft_own_list+{b:g}"], _ = score(cat, S, truth, tracks, per_side(own_bf), bonus=b, reference=ref, **kw)
        out["truths_vs_lists"] = classify_truths(cat, truth, decks, sides, deck_sides)
        out["battlefields_shared"] = sorted(fields)

    def by_legend(lists: Sequence[Deck]) -> tuple[np.ndarray, dict[str, str]]:
        """A list only on the side whose legend, as seen on the table, it names (with every list's battlefields
        and the tokens); a side no list names gets its own legend's prior, runes held to it; a side whose
        legend was not seen gets the whole catalogue. The guard against a list that is not the players'."""
        fields = set().union(*(d.battlefields() for d in lists))
        masks, used = {}, {}
        for s in choices:
            lg = seen[s].most_common(1)[0][0] if seen.get(s) else None
            mine = [d for d in lists if lg is not None and lg in d.legends()]
            if mine:
                masks[s], used[s] = expand(cat, set().union(*(d.card_ids() for d in mine)) | fields).mask, "list"
            elif lg is not None:
                masks[s], used[s] = legend_domains(cat, [lg], runes=True).mask, "legend"
            else:
                masks[s], used[s] = np.ones(len(cat.rows), bool), "catalogue"
        return per_side(masks), used

    if decks:
        mask, used = by_legend(decks)
        out["conditions"]["f_lists_by_legend"], _ = score(cat, S, truth, tracks, mask, reference=ref, **kw)
        out["lists_by_legend"] = used

    # the oracle: each side's "list" is the cards that side showed in the labels (an upper bound)
    shown = {s: {cat.rows[t]["card_id"] for t, ss in zip(truth, sides) if ss == s} for s in choices}
    c = out["conditions"]
    c["oracle_union"], _ = score(cat, S, truth, tracks, expand(cat, set().union(*shown.values())).mask, reference=ref, **kw)
    c["oracle_per_side"], _ = score(cat, S, truth, tracks, per_side({s: expand(cat, shown[s]).mask for s in choices}),
                                    reference=ref, **kw)
    out["oracle_cards_per_side"] = {s: len(v) for s, v in shown.items()}

    # another match's lists, to measure what a wrong list costs
    if foreign and not decks:
        wrong = expand(cat, set().union(*(d.card_ids() for d in foreign)))
        c["wrong_lists_hard"], _ = score(cat, S, truth, tracks, wrong.mask, reference=ref, **kw)
        for b in bonuses:
            c[f"wrong_lists_soft+{b:g}"], _ = score(cat, S, truth, tracks, wrong.mask, bonus=b, reference=ref, **kw)
        mask, used = by_legend(foreign)
        c["wrong_lists_by_legend"], _ = score(cat, S, truth, tracks, mask, reference=ref, **kw)
        out["wrong_lists_by_legend"] = used
    return out


def _guard_out(path: Path) -> None:
    """Results name crops and lists: they stay out of the repository (D-006)."""
    repo = Path(__file__).resolve().parents[2]
    if repo in path.resolve().parents:
        raise SystemExit(f"{path} is inside the repository; write the results under the private data folder")


def _evaluate(a: argparse.Namespace) -> int:
    from . import catalog as catmod
    from .encoders import get_encoder
    from .live.layouts import LAYOUTS

    out_path = Path(a.out)
    _guard_out(out_path)
    rows = [r for r in merged_rows(a.catalog) if catmod.cache_path(a.cache, r["image_url"]).exists()]
    cat = Catalogue(rows)
    layout = LAYOUTS[a.layout]
    crops_dir = Path(a.crops)
    crops, left_out = load_crops(Path(a.labels), Path(a.meta) if a.meta else crops_dir / "crops.json", cat, layout)
    temperature, sure = a.temperature, a.sure
    if temperature is None or sure is None:
        t_live, s_live = live_defaults()
        temperature = t_live if temperature is None else temperature
        sure = s_live if sure is None else sure
    t0 = time.time()
    enc = get_encoder(a.encoder)
    scales = [int(x) for x in a.scales.split(",") if x.strip()]
    embed_cache = Path(a.embed_cache) if a.embed_cache else None
    pyramid = gallery(enc, rows, a.cache, embed_cache, scales)
    sims = query_sims(enc, pyramid, crops_dir, [c["file"] for c in crops], embed_cache)
    print(f"{len(rows)} printings, {len(crops)} labelled crops scored in {time.time() - t0:.0f} s", flush=True)

    truth = np.array([c["truth"] for c in crops], np.int64)
    everything, _ = score(cat, sims, truth, [c["track"] for c in crops], temperature=temperature, sure=sure)
    everything.pop("errors")
    everything.pop("printing_only_errors")
    matches = []
    for spec in a.match:
        name, _, rng = spec.partition("=")
        lo, _, hi = rng.partition("-")
        matches.append(Match(name, hms_seconds(lo), hms_seconds(hi)))
    by_name = {m.name: m for m in matches}
    for spec in a.deck:
        key, _, path = spec.partition("=")
        name, _, side = key.partition(":")
        if name not in by_name:
            raise SystemExit(f"--deck {spec}: no --match named {name}")
        by_name[name].decks.append((Path(path).name.split(".")[0], Path(path), side or None))
    foreign = [read_deck(p, cat) for m in matches for _, p, _ in m.decks]
    bonuses = [float(x) for x in a.bonuses.split(",") if x.strip()]
    result = {
        "question": "how much knowing the players' decklists improves card identification on real broadcast crops",
        "protocol": {"encoder": enc.name, "gallery_scales": scales, "rotation": "search (four turns)",
                     "catalogue": [str(c) for c in a.catalog], "printings": len(rows), "cards": len(cat.card),
                     "temperature": temperature, "sure_p": sure, "bonuses": bonuses, "layout": a.layout,
                     "sides": "each crop's centre against the table window's midline (layout.side)"},
        "labels": {"file": str(a.labels), "scored": len(crops), "left_out": dict(Counter(k for _, k in left_out)),
                   "verdicts_scored": dict(Counter(c["verdict"] for c in crops))},
        "all_crops": everything,
        "matches": {m.name: evaluate_match(cat, sims, crops, m, layout.sides(), temperature, sure, bonuses,
                                           foreign=[d for d in foreign if not m.decks], left_out=left_out)
                    for m in matches},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    c = everything["card"]
    print(f"all {c['n']} crops: card {c['rate']:.2%}, printing {everything['printing']['rate']:.2%}")
    for name, m in result["matches"].items():
        for cond, r in m.get("conditions", {}).items():
            print(f"  {name:<8} {cond:<28} n={r['crops']:<4} card {r['card']['rate']:.2%} printing {r['printing']['rate']:.2%} "
                  f"sure {r['sure']['rate']:.2%} right {r['sure']['right']['rate'] or 0:.2%}")
    print(f"-> {out_path}")
    return 0


def live_defaults() -> tuple[float, float]:
    """The live runner's temperature for a fine-tuned embedder and its one-read naming threshold."""
    import inspect

    from .live.__main__ import EMBEDDER_T
    from .live.pipeline import Recognizer

    return EMBEDDER_T, float(inspect.signature(Recognizer.__init__).parameters["sure_p"].default)


def merged_rows(paths: Sequence[str | Path]) -> list[dict]:
    """The rows of several catalogues (e.g. English and a localised one), each printing once per language:
    localised printings share the English printing id."""
    from . import catalog as catmod

    rows, seen = [], set()
    for p in paths:
        for r in catmod.read_catalog(p):
            key = (r["printing_id"], r.get("language", ""))
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return rows


def _load_catalogue(paths: Sequence[str]) -> Catalogue:
    return Catalogue(merged_rows(paths))


def _show(a: argparse.Namespace) -> int:
    cat = _load_catalogue(a.catalog)
    for path in a.decks:
        d = read_deck(path, cat)
        prior = expand(cat, d.card_ids())
        print(f"{path} ({d.fmt}): {sum(e.count for e in d.entries if e.board == 'main')} main, "
              f"{sum(e.count for e in d.entries if e.board == 'side')} side, {len(d.card_ids())} cards, legend {d.legends()}")
        for e in d.entries:
            print(f"  {e.board:<4} {e.count:>2} {e.card_id:<32} {e.listed or '':<10} {e.section} {e.kind}")
        print(f"  unmapped: {d.unmapped or 'none'}" + (f"; notes: {d.notes}" if d.notes else ""))
        print(f"  expansion: {json.dumps(prior.report)}")
    return 0


def _check(a: argparse.Namespace) -> int:
    """The same list in several formats: identical cards and copies per board?"""
    cat = _load_catalogue(a.catalog)
    decks = [(p, read_deck(p, cat)) for p in a.decks]
    ref = decks[0][1].counts()
    same = True
    for p, d in decks:
        diff = (d.counts() - ref) + (ref - d.counts())
        same &= not diff and not d.unmapped
        print(f"{p} ({d.fmt}): {len(d.card_ids())} cards, {sum(d.counts().values())} copies; unmapped {d.unmapped or 'none'}"
              + (f"; differs from {decks[0][0]}: {dict(diff)}" if diff else ""))
    listed = [d.listed() for _, d in decks if d.listed()]
    ids_same = all(x == listed[0] for x in listed)
    print(f"identical cards and copies in every format: {'yes' if same else 'NO'}; "
          f"identical printings where the format names them: {'yes' if ids_same else 'NO'}")
    return 0 if same and ids_same else 1


def main(argv: list[str] | None = None) -> int:
    data = Path(os.environ.get("RIFTEYE_DATA", Path.home() / "rifteye-data"))
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.decklist", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show", help="a decklist as the catalogue reads it, and the printings it allows")
    p.add_argument("--catalog", action="append", required=True)
    p.add_argument("decks", nargs="+")
    p.set_defaults(fn=_show)
    p = sub.add_parser("check", help="one list in several formats: identical cards and copies?")
    p.add_argument("--catalog", action="append", required=True)
    p.add_argument("decks", nargs="+")
    p.set_defaults(fn=_check)
    p = sub.add_parser("evaluate", help="card identification on labelled crops with and without the lists")
    p.add_argument("--catalog", action="append", default=None, help=f"repeatable (default {data}/catalog/catalog.jsonl)")
    p.add_argument("--cache", default=str(data / "art"))
    p.add_argument("--embed-cache", default=str(data / "embed-cache"))
    p.add_argument("--encoder", default=f"embedder:{data / 'models' / 'embedder-v1.pth'}")
    p.add_argument("--scales", default="120,140,160", help="the gallery levels (spike real's for the held-out sets)")
    p.add_argument("--crops", default=str(data / "real-crops" / "bcn-v1" / "crops"))
    p.add_argument("--labels", default=str(data / "real-crops" / "bcn-v1" / "labels.csv"))
    p.add_argument("--meta", help="crop positions (default: crops.json in the crops folder)")
    p.add_argument("--layout", default="plusrb", help="the broadcast layout that splits the table into sides")
    p.add_argument("--match", action="append", default=[], help="repeatable: NAME=HH:MM:SS-HH:MM:SS (VOD time)")
    p.add_argument("--deck", action="append", default=[], help="repeatable: MATCH=PATH or MATCH:SIDE=PATH")
    p.add_argument("--temperature", type=float, help="default: the live runner's for a fine-tuned embedder")
    p.add_argument("--sure", type=float, help="default: the live runner's one-read naming threshold")
    p.add_argument("--bonuses", default="0.01,0.02,0.03,0.05,0.08,0.12", help="soft prior bonuses on cosine scores")
    p.add_argument("--out", default=str(data / "decklist" / "m2-decklist-prior.json"))
    p.set_defaults(fn=_evaluate)
    a = ap.parse_args(argv)
    if a.cmd == "evaluate" and not a.catalog:
        a.catalog = [str(data / "catalog" / "catalog.jsonl")]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
