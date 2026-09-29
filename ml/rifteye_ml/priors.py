# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Priors on card identification: which gallery rows may compete for a crop (D-026).

Shared by the measurement (`decklist.py`) and the live runner (`live/pipeline.py`), and ported to the engine as
packages/engine/src/priors.ts, which must flag the same rows. numpy only.

- `token_rows`: the tokens, which no list names and every player may put on the table.
- `legend_mask`: the legend rule. A deck's cards fit its legend's domains, runes included (Core Rules 103), so once
  a side's legend is known, a crop on that side competes only with the printings whose domains, less Colorless, lie
  within the legend's; every battlefield (both players' lie on the midline) and every token stays in. A row with no
  domains at all fits any legend, so an older catalogue without them keeps the whole gallery.

A prior only helps name cards face up on the table. It is never shown, and never used to guess or reveal a hand, a
face-down card or the rest of a list (D-005).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np

KEEP_TYPES = ("Rune", "Battlefield")   # kept by the legend prior whatever their domains
COLORLESS = "Colorless"


def normalise(name: str) -> str:
    """A name for matching: accents folded, case and apostrophes dropped, other punctuation a space.
    'Ornn - Blacksmith' and 'Ornn, Blacksmith' meet at 'ornn blacksmith'; letters of other scripts stay."""
    s = "".join(ch for ch in unicodedata.normalize("NFKD", name or "") if not unicodedata.combining(ch))
    s = re.sub(r"['’`]", "", s.casefold())
    return re.sub(r"[\W_]+", " ", s).strip()


def base_name(name: str) -> str:
    """The name without a trailing parenthetical: 'Recruit (ZN)' -> 'Recruit'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", name or "").strip()


def domains(row: dict) -> set[str]:
    """A printing's domains, less Colorless: empty for a colourless row or one with no domains, which fits any legend."""
    return set(row.get("domains") or []) - {COLORLESS}


def token_rows(rows: Sequence[dict]) -> tuple[np.ndarray, list[int]]:
    """Tokens: printings marked token, every printing of a token's card, and printings whose name without its
    parenthetical is a token's (Origins printed some tokens as numbered cards, such as 'Recruit (ZN)'). One flag per
    row, and the rows marked token."""
    marked = [i for i, r in enumerate(rows) if r.get("variant") == "token" or (r.get("type") or "").lower() == "token"]
    cards = {rows[i]["card_id"] for i in marked}
    names = {normalise(base_name(rows[i].get("name") or "")) for i in marked} - {""}
    mask = np.zeros(len(rows), bool)
    mask[marked] = True
    for i, r in enumerate(rows):
        if r["card_id"] in cards or normalise(base_name(r.get("name") or "")) in names:
            mask[i] = True
    return mask, marked


def legend_mask(rows: Sequence[dict], legends: Iterable[str], tokens: np.ndarray | None = None,
                runes: bool = True) -> tuple[np.ndarray, list[set[str]]]:
    """The rows a crop may be read as, given the legends (card ids) of its side: printings whose domains fit one
    legend's (the domains of the legend card's first row), plus every battlefield and token whatever their domains;
    and every rune too unless `runes` holds the runes to the legends' domains, as a rune deck follows its legend.
    `tokens` is `token_rows(rows)[0]`, when the caller has it. One flag per row, and each legend's domains."""
    first: dict[str, int] = {}
    for i, r in enumerate(rows):
        first.setdefault(r["card_id"], i)
    fits: list[set[str]] = []
    for lg in legends:
        if lg not in first:
            raise ValueError(f"legend {lg} is not in the catalogue")
        fits.append(domains(rows[first[lg]]))
    if tokens is None:
        tokens = token_rows(rows)[0]
    keep = tuple(t for t in KEEP_TYPES if not (runes and t == "Rune"))
    mask = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        doms = domains(r)
        mask[i] = r.get("type") in keep or bool(tokens[i]) or any(doms <= f for f in fits)
    return mask, fits
