"""Priors on identification (rifteye_ml.priors), which decklist.py and the live runner share.

The synthetic rows are test_decklist.py's. packages/engine/test/priors.test.ts holds the same rows and the same
expected values, so the two ports are pinned to the same flags and the same names."""
import numpy as np
import pytest

from rifteye_ml import priors
from test_decklist import ROWS


def kept(mask):
    return sorted({f"{ROWS[i]['printing_id']}/{ROWS[i]['language']}" for i in np.flatnonzero(mask)})


def out(mask):
    return sorted({ROWS[i]["printing_id"] for i in np.flatnonzero(~mask)})


def test_tokens_are_marked_printings_their_cards_and_their_names():
    mask, marked = priors.token_rows(ROWS)
    assert marked == [24, 26]
    assert [ROWS[i]["printing_id"] for i in np.flatnonzero(mask)] == ["UNL-T91", "OGN-910", "VEN-T92", "OGN-911"]


# (legends, runes) -> the printings left out; everything else is kept
OUT = [
    (("gleaming-anvil", "thunder-crown"), False, ["OGN-914", "OGN-915", "OGN-916"]),
    (("gleaming-anvil", "thunder-crown"), True, ["OGN-914", "OGN-915", "OGN-916", "OGN-920"]),
    (("gleaming-anvil",), True, ["OGN-914", "OGN-915", "OGN-916", "OGN-920", "VEN-906", "VEN-912"]),
    (("thunder-crown",), True, ["OGN-904", "OGN-905", "OGN-913", "OGN-914", "OGN-915", "OGN-916", "OGN-918", "OGN-918a",
                                "OGN-919", "OGN-920", "OGN-921", "SFD-901", "SFD-902", "SFD-902a", "SFD-951", "SFD-952",
                                "SFD-952*", "SFD-P01", "VEN-902"]),
    (("silent-loom",), False, ["OGN-914", "OGN-915", "OGN-916", "VEN-906", "VEN-912"]),
]


@pytest.mark.parametrize("legends,runes,left_out", OUT)
def test_legend_mask_flags_the_rows_the_engine_flags(legends, runes, left_out):
    mask, fits = priors.legend_mask(ROWS, legends, runes=runes)
    assert out(mask) == left_out
    assert mask.dtype == bool and len(mask) == len(ROWS)
    assert fits == [priors.domains(next(r for r in ROWS if r["card_id"] == lg)) for lg in legends]


def test_the_legend_rule_keeps_the_legends_own_card_battlefields_and_tokens():
    mask, _ = priors.legend_mask(ROWS, ["thunder-crown"])   # runes held to the legend by default
    assert kept(mask) == ["OGN-907/en", "OGN-910/en", "OGN-911/en", "OGN-917/en", "OGN-922/en", "UNL-909/en",
                          "UNL-T91/en", "VEN-906/en", "VEN-912/en", "VEN-T92/en"]
    tokens, _ = priors.token_rows(ROWS)
    assert np.array_equal(priors.legend_mask(ROWS, ["thunder-crown"], tokens)[0], mask)


def test_a_row_without_domains_fits_any_legend():
    rows = [dict(r) for r in ROWS]
    del rows[14]["domains"]            # Blaze Fist, Fury: ruled out while it has its domains
    rows[15]["domains"] = None         # Iron Wall, Body
    mask, _ = priors.legend_mask(rows, ["thunder-crown"])
    assert mask[14] and mask[15] and not mask[16]
    bare = [{k: v for k, v in r.items() if k != "domains"} for r in ROWS]   # a catalogue without domains
    assert priors.legend_mask(bare, ["thunder-crown"])[0].all()


def test_an_unknown_legend_is_refused():
    with pytest.raises(ValueError, match="no-such-legend"):
        priors.legend_mask(ROWS, ["no-such-legend"])


# name -> (normalise(name), base_name(name)): accents, case folding, other scripts, Python's whitespace
NAMES = [
    ("Fakesmith - Hammerer", "fakesmith hammerer", "Fakesmith - Hammerer"),
    ("Zed'Ka", "zedka", "Zed'Ka"),
    ("Kha’Zix", "khazix", "Kha’Zix"),
    ("Rek`Sai", "reksai", "Rek`Sai"),
    ("Éclat–Noir", "eclat noir", "Éclat–Noir"),
    ("Crème Brûlée", "creme brulee", "Crème Brûlée"),
    ("Squire (QX)", "squire qx", "Squire"),
    ("Recruit(ZN)", "recruit zn", "Recruit"),
    ("(Only)", "only", ""),
    ("A (b) (c)", "a b c", "A (b)"),
    ("A (b\n)", "a b", "A"),
    (" A (x) ", "a x", "A"),
    ("\x1cA (B)\x85", "a b", "A"),
    ("﻿X (Y)", "x y", "﻿X"),
    ("a_b__c", "a b c", "a_b__c"),
    ("Straße", "strasse", "Straße"),
    ("GROẞE", "grosse", "GROẞE"),
    ("ΣΊΣΥΦΟΣ", "σισυφοσ", "ΣΊΣΥΦΟΣ"),
    ("ﬁnal ﬂight", "final flight", "ﬁnal ﬂight"),
    ("Ｆｕｌｌ　Ｗｉｄｔｈ", "full width", "Ｆｕｌｌ　Ｗｉｄｔｈ"),
    ("x² ½", "x2 1 2", "x² ½"),
    ("Ⅻ", "xii", "Ⅻ"),
    ("İstanbul", "istanbul", "İstanbul"),
    ("ǅemal", "dzemal", "ǅemal"),
    ("ŉ", "ʼn", "ŉ"),
    ("ᲀᲁᲂ", "вдо", "ᲀᲁᲂ"),
    ("Ꭰꭰᏸ", "ᎠᎠᏰ", "Ꭰꭰᏸ"),
    ("假铁匠", "假铁匠", "假铁匠"),
    ("がぎぐ", "かきく", "がぎぐ"),
    ("한국어", "\u1112\u1161\u11ab\u1100\u116e\u11a8\u110b\u1165", "한국어"),   # NFKD: Hangul as its jamo
    ("नमस्ते", "नमसत", "नमस्ते"),
    ("مَرْحَبًا", "مرحبا", "مَرْحَبًا"),
    ("A️B", "a b", "A️B"),
    ("🔥 Fire", "fire", "🔥 Fire"),
    ("𝐁𝐨𝐥𝐝", "bold", "𝐁𝐨𝐥𝐝"),
    ("", "", ""),
]


@pytest.mark.parametrize("name,norm,base", NAMES)
def test_names_normalise_as_the_engine_normalises_them(name, norm, base):
    assert priors.normalise(name) == norm
    assert priors.base_name(name) == base
