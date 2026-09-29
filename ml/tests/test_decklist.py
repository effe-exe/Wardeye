"""Decklists as a prior on identification (rifteye_ml.decklist).

Synthetic catalogues and lists only: every name and collector number here is invented. The set codes are the
deck code's, because the code stores sets by index."""
import json

import numpy as np
import pytest

from rifteye_ml import decklist as dl


def row(pid, card_id, name, kind, domains, variant="standard", language="en", tags=()):
    return {"printing_id": pid, "card_id": card_id, "name": name, "type": kind, "domains": list(domains),
            "variant": variant, "language": language, "tags": list(tags), "set_code": pid.split("-")[0],
            "collector_number": pid.split("-")[1], "orientation": "landscape" if kind == "Battlefield" else "portrait",
            "image_url": f"https://example.invalid/{pid}-{language}.png"}


ROWS = [
    # Two legends: a legend's catalogue name is its title; its champion is a tag.
    row("SFD-901", "gleaming-anvil", "Gleaming Anvil", "Legend", ["Calm", "Mind"], tags=["Fakesmith"]),
    row("SFD-951", "gleaming-anvil", "Gleaming Anvil", "Legend", ["Calm", "Mind"], variant="overnumbered", tags=["Fakesmith"]),
    row("VEN-912", "thunder-crown", "Thunder Crown", "Legend", ["Order", "Chaos"], tags=["Fakezap"]),
    row("OGN-913", "silent-loom", "Silent Loom", "Legend", ["Calm", "Mind"], tags=["Fakeweaver"]),
    # A champion with every kind of other printing: alt art, overnumbered, signature, a reprint, a localised one.
    row("SFD-902", "fakesmith-hammerer", "Fakesmith, Hammerer", "Unit", ["Calm"]),
    row("SFD-902a", "fakesmith-hammerer", "Fakesmith, Hammerer", "Unit", ["Calm"], variant="alt_art"),
    row("SFD-952", "fakesmith-hammerer", "Fakesmith, Hammerer", "Unit", ["Calm"], variant="overnumbered"),
    row("SFD-952*", "fakesmith-hammerer", "Fakesmith, Hammerer", "Unit", ["Calm"], variant="signature"),
    row("VEN-902", "fakesmith-hammerer", "Fakesmith, Hammerer", "Unit", ["Calm"]),
    row("SFD-902", "fakesmith-hammerer", "假铁匠", "Unit", ["Calm"], language="zh-Hans"),
    # A special printing of it filed under another card_id.
    row("SFD-P01", "fakesmith-hammerer-promo", "Fakesmith, Hammerer (Promo)", "Unit", ["Calm"], variant="promo"),
    row("OGN-904", "pocket-gadget", "Pocket Gadget", "Gear", ["Mind"]),
    row("OGN-905", "quick-trick", "Quick-Trick", "Spell", ["Mind"]),
    row("VEN-906", "spark-bolt", "Spark Bolt", "Spell", ["Order", "Chaos"]),
    row("OGN-914", "blaze-fist", "Blaze Fist", "Unit", ["Fury"]),
    row("OGN-915", "iron-wall", "Iron Wall", "Gear", ["Body"]),
    row("OGN-916", "tidal-edict", "Tidal Edict", "Spell", ["Calm", "Order"]),
    row("OGN-917", "plain-lantern", "Plain Lantern", "Gear", ["Colorless"]),
    row("OGN-907", "quiet-glade", "Quiet Glade", "Battlefield", ["Colorless"]),
    row("UNL-909", "far-tower", "Far Tower", "Battlefield", ["Colorless"]),
    row("OGN-918", "hush-rune", "Hush Rune", "Rune", ["Calm"]),
    row("OGN-918a", "hush-rune", "Hush Rune", "Rune", ["Calm"], variant="alt_art"),
    row("OGN-919", "muse-rune", "Muse Rune", "Rune", ["Mind"]),
    row("OGN-920", "ember-rune", "Ember Rune", "Rune", ["Fury"]),
    # Tokens: one marked token, one of its card printed as a numbered card, and a numbered token whose name
    # differs from the marked one only by a parenthetical.
    row("UNL-T91", "wisp", "Wisp", "Unit", ["Colorless"], variant="token"),
    row("OGN-910", "wisp", "Wisp", "Unit", ["Colorless"]),
    row("VEN-T92", "squire", "Squire", "Unit", ["Colorless"], variant="token"),
    row("OGN-911", "squire-qx", "Squire (QX)", "Unit", ["Colorless"]),
    # Two cards whose names normalise alike: a section settles which is meant.
    row("OGN-921", "zed-ka", "Zed'Ka", "Unit", ["Mind"]),
    row("OGN-922", "zedka", "Zedka", "Battlefield", ["Colorless"]),
]


@pytest.fixture()
def cat():
    return dl.Catalogue(ROWS)


MAIN = {"SFD-901": 1, "SFD-902": 3, "OGN-904": 2, "OGN-905": 3, "OGN-907": 1, "OGN-918": 7, "OGN-919": 5}
SIDE = {"OGN-904": 1, "OGN-914": 2}

JSON_LIST = json.dumps({"metadata": {"name": "Test list"}, "deck": {
    "Main Board": [{"id": k, "count": v} for k, v in MAIN.items()],
    "Side Board": [{"id": k, "count": v} for k, v in SIDE.items()]}})
TEXT_LIST = """1 Fakesmith - Gleaming Anvil (SFD-901)
3 Fakesmith - Hammerer (SFD-902)
2 Pocket Gadget (OGN-904)
3 Quick-Trick (OGN-905)
1 Quiet Glade (OGN-907)
7 Hush Rune (OGN-918)
5 Muse Rune (OGN-919)

Side Board:
1 Pocket Gadget (OGN-904)
2 Blaze Fist (OGN-914)"""
TOURNEY_LIST = """Legend:
1 Fakesmith, Gleaming Anvil

Champion:
3 FAKESMITH, HAMMERER

MainDeck:
2 pocket gadget
3 Quick Trick

Battlefields:
1 Quiet Glade

Runes:
7 Hush Rune
5 Muse Rune

Sideboard:
1 Pocket Gadget
2 Blaze Fist"""


def test_the_four_formats_give_the_same_cards_and_copies(cat):
    decks = [dl.parse(JSON_LIST, cat), dl.parse(TEXT_LIST, cat), dl.parse(TOURNEY_LIST, cat),
             dl.parse(dl.encode_code(MAIN, SIDE), cat)]
    assert [d.fmt for d in decks] == ["json", "text", "tourney", "code"]
    for d in decks:
        assert not d.unmapped, d.fmt
        assert d.counts() == decks[0].counts(), d.fmt
        assert d.legends() == ["gleaming-anvil"]
    assert decks[0].counts()[("main", "fakesmith-hammerer")] == 3
    assert decks[0].counts()[("side", "pocket-gadget")] == 1 and decks[0].counts()[("main", "pocket-gadget")] == 2
    # the formats that name printings name the same ones; the tourney sheet names none
    assert decks[0].listed() == decks[1].listed() == decks[3].listed()
    assert not decks[2].listed()
    assert decks[0].battlefields() == {"quiet-glade"}


def test_tourney_sections_are_kept_and_unknown_names_reported(cat):
    d = dl.parse(TOURNEY_LIST + "\n2 No Such Card", cat)
    sections = {e.card_id: e.section for e in d.entries if e.board == "main"}
    assert sections["gleaming-anvil"] == "legend" and sections["fakesmith-hammerer"] == "champion"
    assert sections["quiet-glade"] == "battlefields" and sections["hush-rune"] == "runes"
    assert {e.card_id for e in d.entries if e.board == "side"} == {"pocket-gadget", "blaze-fist"}
    assert d.unmapped == ["2 No Such Card"]


def test_a_section_settles_a_name_two_cards_share(cat):
    d = dl.parse("Battlefields:\n1 Zedka\n\nMainDeck:\n2 Zed'ka", cat)
    assert {e.card_id: e.section for e in d.entries} == {"zedka": "battlefields", "zed-ka": "maindeck"}
    loose = dl.parse("1 zedka", cat)  # no section, no id: it could be either card, so it is not guessed
    assert not loose.entries and "zed-ka" in loose.unmapped[0] and "zedka" in loose.unmapped[0]


def test_text_ids_decide_and_unknown_ids_are_reported(cat):
    d = dl.parse("2 Pocket Gadget (OGN-904)\n1 Blaze Fist (OGN-905)\n1 Missing Card (OGN-999)", cat)
    assert [e.card_id for e in d.entries] == ["pocket-gadget", "quick-trick"]
    assert any("OGN-905" in n and "blaze-fist" in n for n in d.notes)
    assert d.unmapped == ["1 Missing Card (OGN-999)"]
    j = dl.parse(json.dumps({"deck": {"Main Board": [{"id": "OGN-999", "count": 1}, {"id": "ogn-904", "count": 2}]}}), cat)
    assert j.unmapped == ["OGN-999"] and [e.listed for e in j.entries] == ["OGN-904"]


def test_names_normalise_across_formats():
    assert dl.normalise("Fakesmith - Hammerer") == dl.normalise("fakesmith, HAMMERER") == "fakesmith hammerer"
    assert dl.normalise("Zed'Ka") == dl.normalise("Zedka") == "zedka"
    assert dl.normalise("Éclat–Noir") == "eclat noir"
    assert dl.base_name("Squire (QX)") == "Squire"


# ---- the deck code -----------------------------------------------------------------------

BIG_MAIN = {"OGN-012": 12, "OGN-311": 9, "SFD-005": 9, "OGS-001": 4, "UNL-200": 3, "VEN-150": 3, "VEN-151": 3,
            "SFD-129": 2, "OGN-900": 1, "OGS-002": 1, "UNL-201": 1, "VEN-152": 1}
BIG_SIDE = {"OGN-900": 3, "SFD-130": 2, "UNL-202": 1, "VEN-999": 1}


def test_deck_code_round_trips():
    for main, side in ((MAIN, SIDE), (BIG_MAIN, BIG_SIDE), ({"VEN-001": 1}, {})):
        code = dl.encode_code(main, side)
        assert set(code) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") and "=" not in code
        assert dl.decode_code(code) == (main, side)
        assert dl.encode_code(*dl.decode_code(code)) == code
        assert dl.decode_code(code.lower()) == (main, side)


def _raw(numbers, header=dl.CODE_HEADER, trailer=(0,)):
    """A code from raw numbers (after the header byte), for codes the encoder would never write."""
    import base64

    data = bytes([header]) + b"".join(dl._varint(x) for x in list(numbers) + list(trailer))
    return base64.b32encode(data).decode().rstrip("=")


def _one_card(set_index, variant=0):
    # main board: one group with 1 copy (count 1 is the 12th count); side board empty
    return [0] * 11 + [1, 1, set_index, variant, 7] + [0, 0, 0]


def test_deck_code_refuses_what_it_does_not_know():
    assert dl.decode_code(_raw(_one_card(4))) == ({"UNL-007": 1}, {})
    assert dl.decode_code(_raw(_one_card(4), trailer=())) == ({"UNL-007": 1}, {})  # the trailing 0 is optional
    with pytest.raises(dl.DeckCodeError, match="set index 2"):
        dl.decode_code(_raw(_one_card(2)))
    with pytest.raises(dl.DeckCodeError, match="set index 9"):
        dl.decode_code(_raw(_one_card(9)))
    with pytest.raises(dl.DeckCodeError, match="variant 1"):
        dl.decode_code(_raw(_one_card(0, variant=1)))
    with pytest.raises(dl.DeckCodeError, match="header"):
        dl.decode_code(_raw(_one_card(0), header=0x12))
    with pytest.raises(dl.DeckCodeError, match="unexpected"):
        dl.decode_code(_raw(_one_card(0), trailer=(0, 5)))
    with pytest.raises(dl.DeckCodeError, match="inside a number"):
        dl.decode_code(_raw([0] * 5 + [0x80], trailer=()))  # 0x80 as a varint's first byte: more follows
    with pytest.raises(dl.DeckCodeError, match="base32"):
        dl.decode_code("not a code!")
    for bad in ({"OGN-901a": 1}, {"XYZ-001": 1}, {"OGN-001": 13}):
        with pytest.raises(dl.DeckCodeError):
            dl.encode_code(bad)
    with pytest.raises(dl.DeckCodeError):
        dl.encode_code({"OGN-001": 1}, {"OGN-002": 4})


def test_detect_format():
    assert dl.detect_format(JSON_LIST) == "json"
    assert dl.detect_format(TEXT_LIST) == "text"
    assert dl.detect_format(TOURNEY_LIST) == "tourney"
    assert dl.detect_format(dl.encode_code(MAIN, SIDE)) == "code"


# ---- the priors ----------------------------------------------------------------------------

def _pids(cat, mask):
    return sorted({(cat.rows[i]["printing_id"], cat.rows[i]["language"]) for i in np.flatnonzero(mask)})


def test_expand_takes_a_card_to_every_printing_and_tokens(cat):
    p = dl.expand(cat, {"fakesmith-hammerer"})
    got = _pids(cat, p.mask)
    for pid in ("SFD-902", "SFD-902a", "SFD-952", "SFD-952*", "VEN-902"):
        assert (pid, "en") in got
    assert ("SFD-902", "zh-Hans") in got                      # every language the gallery holds
    assert ("SFD-P01", "en") in got                            # same name, filed under another card_id
    assert p.report["same_name"] == [{"listed": "fakesmith-hammerer", "card_id": "fakesmith-hammerer-promo",
                                      "printings": ["SFD-P01"]}]
    for token in ("UNL-T91", "OGN-910", "VEN-T92", "OGN-911"):  # tokens, never listed, always allowed
        assert (token, "en") in got
    assert cat.token_extra == ["OGN-910", "OGN-911"]
    assert ("OGN-904", "en") not in got
    assert p.report["variants"] == {"standard": 3, "alt_art": 1, "overnumbered": 1, "signature": 1, "promo": 1}
    bare = dl.expand(cat, {"fakesmith-hammerer", "no-such-card"}, tokens=False)
    assert bare.report["unknown"] == ["no-such-card"] and ("UNL-T91", "en") not in _pids(cat, bare.mask)


def test_legend_domains_keeps_fitting_cards_runes_battlefields_and_tokens(cat):
    p = dl.legend_domains(cat, ["gleaming-anvil", "thunder-crown"])
    kept = {pid for pid, _ in _pids(cat, p.mask)}
    assert {"SFD-902", "OGN-904", "VEN-906", "OGN-917", "OGN-913", "SFD-951"} <= kept   # fits one legend, or colourless
    assert {"OGN-918", "OGN-920", "OGN-907", "UNL-909", "UNL-T91", "OGN-911"} <= kept   # every rune, battlefield, token
    assert not {"OGN-914", "OGN-915", "OGN-916"} & kept   # off-domain, and Calm+Order fits neither legend alone
    assert p.report["kept_by_type"]["Rune"] == [4, 4]
    with pytest.raises(ValueError):
        dl.legend_domains(cat, ["no-such-legend"])


def test_legend_domains_can_hold_runes_to_the_legends(cat):
    """A rune deck follows its legend: with `runes`, a Fury rune does not fit Calm+Mind or Order+Chaos."""
    both = {pid for pid, _ in _pids(cat, dl.legend_domains(cat, ["gleaming-anvil", "thunder-crown"], runes=True).mask)}
    assert {"OGN-918", "OGN-918a", "OGN-919"} <= both and "OGN-920" not in both
    assert {"OGN-907", "UNL-909", "UNL-T91"} <= both   # battlefields and tokens are still kept
    own = {pid for pid, _ in _pids(cat, dl.legend_domains(cat, ["thunder-crown"], runes=True).mask)}
    assert not {"OGN-918", "OGN-919", "OGN-920"} & own


def test_adjust_hard_and_soft():
    sims = np.array([[0.5, 0.9, 0.7]])
    allowed = np.array([True, False, True])
    assert np.argmax(dl.adjust(sims, allowed)) == 2
    assert np.isneginf(dl.adjust(sims, allowed)[0, 1])
    assert np.allclose(dl.adjust(sims, allowed, bonus=0.1), [[0.6, 0.9, 0.8]])
    assert dl.adjust(sims) is sims


def test_read_rolls_printings_up_into_cards_as_the_live_runner_does():
    cards = np.array(["a", "a", "b"])
    row, p = dl.read(np.array([0.9, 0.8, 0.85]), cards, temperature=0.05)
    assert row == 0 and p == pytest.approx(1 / (1 + np.exp(-1)))
    row, p = dl.read(np.array([-np.inf, 0.2, -np.inf]), cards, temperature=0.05)
    assert row == 1 and p == pytest.approx(1.0)


def test_score_counts_fixes_breaks_and_what_lies_outside_the_prior(cat):
    rows = [cat.pid[p] for p in ("SFD-902", "OGN-904", "OGN-914")]
    truth = np.array(rows)
    sims = np.full((3, len(ROWS)), 0.1, np.float32)
    sims[0, rows[0]], sims[0, cat.pid["OGN-916"]] = 0.80, 0.85   # wrong on the whole catalogue, fixed by the list
    sims[1, rows[1]] = 0.9                                         # right either way
    sims[2, rows[2]] = 0.9                                         # right, but its card is not listed
    base, ref = dl.score(cat, sims, truth, ["t1", "t2", "t3"])
    assert base["card"]["k"] == 2 and base["errors"][0] == {"truth": "SFD-902", "predicted": "OGN-916", "crops": 1,
                                                             "tracks": ["t1"]}
    listed = dl.expand(cat, {"fakesmith-hammerer", "pocket-gadget"}).mask
    hard, _ = dl.score(cat, sims, truth, ["t1", "t2", "t3"], listed, reference=ref)
    assert hard["card"]["k"] == 2 and hard["fixed"] == 1 and hard["broken"] == 1
    assert hard["outside"] == {"crops": 1, "cards": ["blaze-fist"], "lost": {"crops": 1, "cards": ["blaze-fist"]}}
    soft, _ = dl.score(cat, sims, truth, ["t1", "t2", "t3"], listed, bonus=0.1, reference=ref)
    assert soft["card"]["k"] == 3 and soft["outside"]["lost"]["crops"] == 0
    assert soft["card"]["ci95"][0] < 1.0 == soft["card"]["rate"]
    per_crop = np.stack([listed, listed, dl.expand(cat, {"blaze-fist"}).mask])  # one row of flags per crop
    side, _ = dl.score(cat, sims, truth, ["t1", "t2", "t3"], per_crop, reference=ref)
    assert side["card"]["k"] == 3 and side["outside"]["crops"] == 0


def test_sides_follow_the_cards_only_one_list_has(cat):
    a = dl.parse(JSON_LIST, cat)
    b = dl.parse(json.dumps({"deck": {"Main Board": [{"id": "VEN-912", "count": 1}, {"id": "VEN-906", "count": 3},
                                                     {"id": "OGN-918", "count": 5}]}}), cat)
    truth = np.array([cat.pid[p] for p in ("OGN-904", "SFD-902", "VEN-906", "OGN-918")])
    order, how = dl.assign_sides(cat, [a, b], truth, ["right", "right", "left", "left"], ("left", "right"))
    assert order == ["right", "left"] and how["agree"] == 3 and how["disagree"] == 0   # the shared rune does not vote
    with pytest.raises(ValueError):
        dl.assign_sides(cat, [a, b], truth[3:], ["left"], ("left", "right"))


def test_evaluate_match_runs_every_condition(cat, tmp_path):
    (tmp_path / "a.json").write_text(JSON_LIST)
    (tmp_path / "b.json").write_text(json.dumps({"deck": {"Main Board": [
        {"id": "VEN-912", "count": 1}, {"id": "VEN-906", "count": 3}, {"id": "UNL-909", "count": 1}]}}))
    truths = [("SFD-902a", "left"), ("OGN-904", "left"), ("VEN-906", "right"), ("UNL-909", "left"), ("UNL-T91", "right"),
              ("SFD-901", "left")]
    crops = [{"file": f"t00h00m{n:02d}s_00.png", "track": f"t{n}", "t": float(n), "truth": cat.pid[p], "side": s,
              "verdict": "correct"} for n, (p, s) in enumerate(truths)]
    rng = np.random.default_rng(0)
    sims = rng.uniform(0.0, 0.5, (len(crops), len(ROWS))).astype(np.float32)
    for n, c in enumerate(crops):
        sims[n, c["truth"]] = 0.9
    sims[0, cat.pid["OGN-916"]] = 0.95   # the alt-art champion looks like an off-list card
    m = dl.Match("m", 0, 100, [("a", tmp_path / "a.json", None), ("b", tmp_path / "b.json", None)])
    out = dl.evaluate_match(cat, sims, crops, m, ("left", "right"), temperature=0.03, sure=0.85, bonuses=[0.02, 0.1],
                            left_out=[(3.0, "back"), (500.0, "none")])
    assert out["crops"] == 6 and out["left_out"] == {"back": 1}
    assert out["decks"]["a"]["side"] == "left" and out["decks"]["b"]["side"] == "right"
    assert out["legends_from_lists"] == {"right": "thunder-crown"}
    c = out["conditions"]
    assert c["a_catalogue"]["card"]["k"] == 5
    assert c["c_both_lists"]["card"]["k"] == 6 and c["c_both_lists"]["fixed"] == 1
    # the other side's battlefield, on the left: wrong under the strict own list, right with shared battlefields
    assert c["d_own_list_strict"]["outside"]["cards"] == ["far-tower"] and c["d_own_list"]["card"]["k"] == 6
    assert c["e_soft_both_lists+0.1"]["card"]["k"] == 6 and "b_legend_domains" in c and "oracle_per_side" in c
    assert "b_legend_domains_runes" in c and "b_legend_domains_per_side_runes" in c
    # only the left legend is on the table: its list is used there, and the right side keeps the catalogue
    assert out["lists_by_legend"] == {"left": "list", "right": "catalogue"} and c["f_lists_by_legend"]["card"]["k"] == 6
    how = {t["printing"]: t["how"] for t in out["truths_vs_lists"]}
    assert how["SFD-902a"].startswith("another printing of a listed card (alt_art") and how["UNL-T91"] == "token"
    assert how["OGN-904"] == "listed"


def test_a_list_whose_legend_is_not_on_the_table_is_not_used(cat, tmp_path):
    """Another match's list: hard, it wrecks the reads; by legend, the sides fall back to their legends' priors."""
    (tmp_path / "a.json").write_text(JSON_LIST)   # a Gleaming Anvil list
    truths = [("OGN-914", "left"), ("OGN-913", "left"), ("VEN-906", "right"), ("VEN-912", "right"), ("OGN-920", "left")]
    crops = [{"file": f"t00h00m{n:02d}s_00.png", "track": f"t{n}", "t": float(n), "truth": cat.pid[p], "side": s,
              "verdict": "correct"} for n, (p, s) in enumerate(truths)]
    sims = np.full((len(crops), len(ROWS)), 0.2, np.float32)
    for n, c in enumerate(crops):
        sims[n, c["truth"]] = 0.9
    foreign = [dl.read_deck(tmp_path / "a.json", cat)]
    out = dl.evaluate_match(cat, sims, crops, dl.Match("m", 0, 100), ("left", "right"), temperature=0.03, sure=0.85,
                            bonuses=[0.02], foreign=foreign)
    c = out["conditions"]
    assert c["wrong_lists_hard"]["card"]["k"] < 5
    assert out["wrong_lists_by_legend"] == {"left": "legend", "right": "legend"}
