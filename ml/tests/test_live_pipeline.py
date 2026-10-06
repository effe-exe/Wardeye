import math

import numpy as np
import pytest
from PIL import Image

from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog
from rifteye_ml.live.layouts import LAYOUTS, Layout
from rifteye_ml.live.pipeline import Recognizer, Track, card_crop, hidden_runes, quad
from rifteye_ml.matcrops import CardBox

MAT = (30, 40, 55)
LAYOUT = Layout("test", "a test table", (0.0, 0.0, 1.0, 1.0), card_long_1080=156,  # 78 px cards in a 540 px frame
                mat=MAT, mat_share=0.5)


def _setup(n=6, **kw):
    pytest.importorskip("scipy")
    from rifteye_ml.encoders import get_encoder
    from rifteye_ml.retrieval import Pyramid, at_long_side

    rows = synthetic_catalog(n, seed=11)
    art = [load_fixture_image(r) for r in rows]
    enc = get_encoder("colorgrid/trim0.03+dhash/trim0.03")
    pyr = Pyramid({s: enc.embed([at_long_side(im, s) for im in art]) for s in (70, 80)})
    return rows, art, Recognizer(LAYOUT, rows, enc, pyr, fps=5.0, **kw)


def _frame(cards):
    """A 960 x 540 mat with each (image, x, y) pasted as a 56 x 78 card at (x, y)."""
    im = Image.new("RGB", (960, 540), MAT)
    for card, x, y in cards:
        im.paste(card.resize((56, 78), Image.BOX), (x, y))
    return np.asarray(im)


def test_quad_and_crop_follow_the_card():
    box = CardBox((100.0, 80.0), 78.0, 56.0, 90.0, 1.0)  # long side vertical
    q = np.array(quad(box))
    assert np.allclose(q.mean(axis=0), (100, 80)) and np.isclose(np.ptp(q[:, 1]), 78) and np.isclose(np.ptp(q[:, 0]), 56)
    frame = Image.new("RGB", (300, 200), MAT)
    frame.paste((200, 10, 10), (72, 41, 128, 119))
    crop = card_crop(frame, box)
    assert abs(crop.size[0] - 56) <= 1 and abs(crop.size[1] - 78) <= 1
    assert np.asarray(crop)[39, 28].tolist() == [200, 10, 10]


def test_cards_are_named_played_once_and_face_down_never(tmp_path):
    rows, art, rec = _setup()
    sleeve = Image.new("RGB", (56, 78), (220, 90, 150))
    events, state = [], {}
    for k in range(70):  # 14 s at 5 fps
        t = k / 5
        cards = [(sleeve, 600, 300)]
        if not 8 <= t < 11:            # card 0 on the table from the start; a hand hides it for 3 s
            cards.append((art[0], 100, 100))
        if t >= 5:                     # card 3 is played at 5 s
            cards.append((art[3], 350, 200))
        state, ev = rec.step(t, _frame(cards))
        events += ev
    by_state = {tr["state"]: tr for tr in state["tracks"]}
    named = {tr["name"] for tr in state["tracks"] if tr["state"] == "named"}
    assert named == {rows[0]["name"], rows[3]["name"]}
    assert "facedown" in by_state and by_state["facedown"]["guesses"] == [] and by_state["facedown"]["name"] == ""
    played = [e for e in events if e["kind"] == "played"]
    assert [e["text"] for e in played] == [f"{rows[3]['name']} played"]   # card 0 was there when we tuned in
    assert 5 <= played[0]["t"] < 7 and played[0]["side"] == "left"
    assert not [e for e in events if e["kind"] == "left"]                  # hidden for 3 s is not leaving
    assert state["frame"] == {"width": 960, "height": 540} and [p["side"] for p in state["players"]] == ["left", "right"]


def test_a_card_held_in_a_hand_over_the_table_is_not_read_until_it_is_put_down():
    # D-005: a card in a player's hand is hidden information, even where the table camera sees it
    from rifteye_ml.live.pipeline import hand_share

    rows, art, rec = _setup(gate=False)
    box = CardBox((378.0, 239.0), 78.0, 56.0, 90.0, 1.0)  # the card pasted at (350, 200)
    rec.finder = lambda t, image: [box] if t >= 4 else []
    events, shown = [], {}
    for k in range(60):  # 12 s at 5 fps: a hand brings the card over the table at 4 s, holds it there, puts it down at 8 s
        t = k / 5
        im = Image.fromarray(_frame([(art[3], 350, 200)] if t >= 4 else []))
        if 4 <= t < 8:
            im.paste((200, 140, 110), (326, 200, 350, 278))  # the fingers along its left edge
        state, ev = rec.step(t, np.asarray(im))
        events += ev
        shown[t] = [tr["name"] for tr in state["tracks"]]
        if 4 <= t < 8:
            assert [tr.reads for tr in rec.tracks.values()] == [0]  # never read in the hand
    assert not any(v for t, v in shown.items() if t < 8)  # nor drawn
    played = [e for e in events if e["kind"] == "played"]
    assert [e["text"] for e in played] == [f"{rows[3]['name']} played"] and 8 <= played[0]["t"] < 9  # put down: played
    im = _frame([(art[3], 350, 200)])
    fingers = np.array(im)
    fingers[200:278, 326:350] = (200, 140, 110)
    table = LAYOUT.box(960, 540)
    assert hand_share(im, box, table) == 0.0 and hand_share(fingers, box, table) >= 0.1
    other = CardBox((338.0, 239.0), 78.0, 30.0, 90.0, 1.0)  # skin-coloured art of a card beside it is no hand
    assert hand_share(fingers, box, table, [other]) == 0.0


def test_layouts_split_the_table_between_the_players():
    la = LAYOUTS["la-rq"]
    assert la.card_px(1080) == 155 and la.card_px(720) == pytest.approx(103.3, abs=0.1)
    assert la.side(500, 500, 1920, 1080) == "left" and la.side(1500, 500, 1920, 1080) == "right"
    top = Layout("x", "", (0, 0, 1, 1), 100, split="horizontal")
    assert top.side(10, 10, 100, 100) == "top" and top.sides() == ("top", "bottom")


def test_detector_quads_become_boxes_and_backs_stay_unnamed():
    from rifteye_ml.live.pipeline import detector_boxes

    q = [100, 50, 156, 50, 156, 128, 100, 128]  # a 56 x 78 card, long side vertical
    boxes = detector_boxes([{"cls": "card", "score": 0.9, "quad": q},
                            {"cls": "card_back", "score": 0.8, "quad": [x + 300 for x in q]},
                            {"cls": "card", "score": 0.1, "quad": q}])
    assert len(boxes) == 2 and [b.back for b in boxes] == [False, True]
    b = boxes[0]
    assert b.centre == pytest.approx((128, 89)) and b.long_px == pytest.approx(78) and b.short_px == pytest.approx(56)
    assert b.angle_deg == pytest.approx(90)


def test_a_strip_is_not_a_card():
    # Riot's showdown banner lies over the bottom of the table, and the detector outlines its art as strips 2.5
    # times as long as wide; a card, even one under others, keeps its 63 x 88 shape
    from rifteye_ml.live.pipeline import MIN_ASPECT, detector_boxes

    card = [100, 50, 156, 50, 156, 128, 100, 128]    # 56 x 78
    strip = [300, 1016, 460, 1016, 460, 1080, 300, 1080]  # 160 x 64
    dets = [{"cls": "card", "score": 0.8, "quad": card}, {"cls": "card", "score": 0.8, "quad": strip}]
    assert MIN_ASPECT == 0.5
    assert [b.long_px for b in detector_boxes(dets)] == pytest.approx([78])
    assert len(detector_boxes(dets, min_aspect=0.0)) == 2


def test_a_legend_named_on_a_side_is_that_players_legend():
    rows, art, rec = _setup()
    rows[2]["type"] = "Legend"
    rec.row_of = {r["printing_id"]: r for r in rows}
    for k in range(15):
        state, _ = rec.step(k / 5, _frame([(art[2], 700, 100), (art[0], 100, 100)]))
    legends = {p["side"]: p["legend"] for p in state["players"]}
    assert legends == {"left": None, "right": {"printing_id": rows[2]["printing_id"], "name": rows[2]["name"]}}


def test_the_run_stops_when_most_card_pictures_cannot_be_fetched(tmp_path, monkeypatch):
    from rifteye_ml import catalog as cat
    from rifteye_ml.live.__main__ import ensure_catalogue

    monkeypatch.setattr(cat, "download_images", lambda rows, cache, workers=4: {})  # every fetch fails
    few, many = tmp_path / "few.jsonl", tmp_path / "many.jsonl"
    cat.write_catalog(synthetic_catalog(3, seed=1), few)
    cat.write_catalog(synthetic_catalog(12, seed=1), many)
    assert ensure_catalogue(few, tmp_path / "art") == few  # a few broken links: carry on
    with pytest.raises(SystemExit, match="12 card pictures could not be fetched"):
        ensure_catalogue(many, tmp_path / "art")


def test_face_down_takes_two_looks_in_a_row_and_never_unnames_a_card():
    rows, art, rec = _setup()
    box = CardBox((100.0, 100.0), 78.0, 56.0, 90.0, 1.0)
    card = rows[0]["card_id"]
    held = Track("a", box, 0.0, 0.0, reads=2, prob={card: 2.0}, named=card, down=5)  # a hand rests on it
    assert rec.label(held)[0] == "named"
    new = Track("b", box, 0.0, 0.0, down=2, last_read=1.0)  # two face-down looks, never named
    assert rec.label(new)[0] == "facedown" and not rec.due(new, 5.0) and rec.due(new, 9.5)
    new.down = 1
    assert rec.label(new)[0] == "new"  # one look is not enough


def _boxes(cards):
    return [CardBox((x + 28.0, y + 39.0), 78.0, 56.0, 90.0, 1.0) for _, x, y in cards]


def _run(rec, plan, seconds, fps=5):
    """plan(t) -> (cards on the table, the ones the detector reports); the last state and every event."""
    events, state = [], {}
    for k in range(int(seconds * fps)):
        t = k / fps
        shown, found = plan(t)
        rec.finder = lambda t_, im, found=found: _boxes(found)
        state, ev = rec.step(t, _frame(shown))
        events += ev
    return state, events


def test_a_card_out_of_sight_keeps_its_id_and_name_where_it_was():
    rows, art, rec = _setup(gate=False)
    a = (art[0], 100, 100)
    state, events = _run(rec, lambda t: ([a], [] if 3 <= t < 23 else [a]), 26)  # 20 s under a hand
    named = [tr for tr in state["tracks"] if tr["state"] == "named"]
    assert [tr["id"] for tr in named] == ["t0"] and named[0]["name"] == rows[0]["name"] and not named[0]["hidden"]
    assert not [e for e in events if e["kind"] in ("played", "left", "moved")]  # on the table when we tuned in


def test_a_card_picked_up_and_put_elsewhere_moves_with_its_id():
    rows, art, rec = _setup(gate=False)

    def plan(t):
        c = (art[1], 100, 100) if t < 6 else (art[1], 500, 300)
        return [c], ([] if 6 <= t < 6.6 else [c])  # in the player's hand for a moment

    state, events = _run(rec, plan, 10)
    moved = [e for e in events if e["kind"] == "moved"]
    assert [e["text"] for e in moved] == [f"{rows[1]['name']} moved"] and moved[0]["track"] == "t0"
    assert [tr["id"] for tr in state["tracks"] if tr["state"] == "named"] == ["t0"]
    assert not [e for e in events if e["kind"] == "played"]


def test_a_card_under_another_keeps_its_name_while_covered(monkeypatch):
    from rifteye_ml.live import pipeline

    monkeypatch.setattr(pipeline, "KEEP_S", 3.0)
    rows, art, rec = _setup(gate=False)
    under, top = (art[2], 100, 100), (art[4], 118, 125)  # the new card lies across most of it
    state, events = _run(rec, lambda t: ([under], [under]) if t < 5 else ([under, top], [top]), 15)
    by_id = {tr["id"]: tr for tr in state["tracks"]}
    assert by_id["t0"]["name"] == rows[2]["name"] and by_id["t0"]["hidden"]  # still on the board, 10 s later
    assert [e["text"] for e in events if e["kind"] == "played"] == [f"{rows[4]['name']} played"]


def test_legends_and_battlefields_stay_pinned_and_runes_are_quiet():
    rows, art, rec = _setup(gate=False)
    rows[0]["type"], rows[1]["type"], rows[2]["type"] = "Battlefield", "Legend", "Rune"
    bf, lg, rune = (art[0], 440, 100), (art[1], 300, 100), (art[2], 600, 100)  # the battlefield on the strip

    def plan(t):
        if t < 4:
            return [], []
        return [bf, lg, rune], ([bf, lg, rune] if t < 8 else [rune])  # then a hand over the legend and battlefield

    state, events = _run(rec, plan, 14)
    by_kind = {tr["kind"]: tr for tr in state["tracks"]}
    assert by_kind["battlefield"]["name"] == rows[0]["name"] and not by_kind["battlefield"]["hidden"]
    assert by_kind["legend"]["name"] == rows[1]["name"] and by_kind["rune"]["name"] == rows[2]["name"]
    assert events == []  # set-up cards and runes are never announced


def test_a_battlefield_misread_once_is_named_again_as_the_card_it_is():
    # a rune turned sideways (exhausted) looks like a battlefield's landscape art: read once as one, it was pinned
    # under that name all game; the reads since say what it is
    rows, art, rec = _setup(gate=False)
    rows[0]["type"], rows[1]["type"] = "Battlefield", "Rune"
    rec.row_of = {r["printing_id"]: r for r in rows}
    bf, rune = rows[0]["card_id"], rows[1]["card_id"]
    box = CardBox((100.0, 100.0), 78.0, 56.0, 0.0, 1.0)

    def pinned(prob):
        return Track("t0", box, 0.0, 10.0, hits=20, reads=5, prob=prob, best_row={c: (0.9, i) for i, c in enumerate(prob)},
                     named=bf, kind="Battlefield", pinned=True, side="left")

    misread = pinned({bf: 0.5, rune: 4.4})
    rec.tracks = {"t0": misread}
    assert rec.announce(10.0) == [] and not misread.pinned and misread.named == rune  # a rune: never announced
    kept = pinned({bf: 3.0, rune: 2.0})  # read as itself three times in five: still the battlefield, under dice or a hand
    rec.tracks = {"t0": kept}
    assert rec.announce(10.0) == [] and kept.pinned and kept.named == bf
    doubt, sure = pinned({bf: 1.0, rune: 1.9}), pinned({bf: 4.9, rune: 0.1})  # read half a second ago
    doubt.last_read = sure.last_read = 9.5
    assert rec.due(doubt, 10.0) and not rec.due(sure, 10.0)  # leaning to another card, it is read again at once


def test_a_battlefield_moved_keeps_its_pin_and_is_not_drawn_twice():
    rows, art, rec = _setup(gate=False)
    rows[0]["type"] = "Battlefield"
    rec.row_of = {r["printing_id"]: r for r in rows}

    def plan(t):
        c = (art[0], 400, 100) if t < 6 else (art[0], 470, 350)
        return [c], ([] if 6 <= t < 7.5 else [c])  # picked up and put down elsewhere on the strip

    state, events = _run(rec, plan, 12)
    fields = [tr for tr in state["tracks"] if tr["kind"] == "battlefield"]
    assert [tr["id"] for tr in fields] == ["t0"] and not fields[0]["hidden"] and events == []
    cx, cy = np.mean(fields[0]["quad"], axis=0)
    assert abs(cx - 498) < 3 and abs(cy - 389) < 3  # where it lies now


def test_a_pinned_card_long_out_of_sight_is_listed_but_not_drawn():
    rows, art, rec = _setup(gate=False)
    rows[0]["type"] = "Battlefield"
    rec.row_of = {r["printing_id"]: r for r in rows}
    bf = (art[0], 440, 100)
    state, _ = _run(rec, lambda t: ([bf], [bf] if t < 6 else []), 28)  # gone from the detector's view for 22 s
    fields = [tr for tr in state["tracks"] if tr["kind"] == "battlefield"]
    assert len(fields) == 1 and fields[0]["name"] == rows[0]["name"] and fields[0]["hidden"]


def test_a_card_held_across_the_table_edge_is_never_read():
    # a player's hand, held over the table at its edge: the camera sees the cards' faces, but they are not on it
    edge = Layout("edge", "a table inside the frame", (0.2, 0.0, 0.8, 1.0), card_long_1080=156, mat=MAT, mat_share=0.5)
    rows, art, _ = _setup(gate=False)
    from rifteye_ml.encoders import get_encoder
    from rifteye_ml.retrieval import Pyramid, at_long_side

    enc = get_encoder("colorgrid/trim0.03+dhash/trim0.03")
    rec = Recognizer(edge, rows, enc, Pyramid({s: enc.embed([at_long_side(im, s) for im in art]) for s in (70, 80)}), fps=5.0, gate=False)
    # the window starts at x = 192: one card held 20 px past its edge, one lying just inside it, one a card's width in
    past, edge, lying = (art[0], 172, 200), (art[2], 196, 330), (art[1], 400, 200)
    state, events = _run(rec, lambda t: ([past, edge, lying], [past, edge, lying]), 4)
    assert sorted(tr["name"] for tr in state["tracks"]) == sorted([rows[2]["name"], rows[1]["name"]])
    assert rec.held(_boxes([past])[0], 960, 540) and not rec.held(_boxes([edge])[0], 960, 540)


def test_a_battlefield_off_the_strip_needs_agreeing_reads_and_is_never_pinned():
    # the battlefields lie on the strip along the midline; off it, a card read as one is most likely turned sideways
    rows, art, rec = _setup(gate=False)
    rows[0]["type"] = "Battlefield"
    rec.row_of = {r["printing_id"]: r for r in rows}
    bf = rows[0]["card_id"]
    side = CardBox((150.0, 300.0), 78.0, 56.0, 0.0, 1.0)  # x 150 of 960: a player's side of the table
    rec.frame_wh = (960, 540)
    assert not rec.in_strip(150.0, 300.0) and rec.in_strip(470.0, 300.0)
    tr = Track("t0", side, 0.0, 10.0, hits=20, reads=2, prob={bf: 1.96}, best_row={bf: (0.9, 0)})
    assert rec.label(tr)[0] == "unsure"  # read twice, surely: on the strip that would name it
    tr.reads, tr.prob = 3, {bf: 2.9}
    rec.tracks = {"t0": tr}
    assert rec.label(tr)[0] == "named" and rec.announce(10.0) == [] and tr.named == bf and not tr.pinned


def test_a_named_card_competes_as_itself_wherever_it_goes():
    rows, art, rec = _setup(gate=False)
    left = np.array([1, 1, 0, 0, 0, 1], bool)
    taken = rows[4]["card_id"]  # a unit moved across the midline to a battlefield, or taken by the other player
    assert (rec.with_card(left, taken) == (left | (rec.cards == taken))).all()


def test_two_lists_give_the_other_player_theirs_once_one_legend_is_read():
    class Deck:
        def __init__(self, legend):
            self.legend = legend

        def legends(self):
            return [self.legend]

    rows, art, rec = _setup(gate=False)
    a, b = rows[1], rows[2]
    rec.decks = [Deck(a["card_id"]), Deck(b["card_id"])]
    assert rec.legend_by_elimination("right") is None  # no legend read yet
    rec.legends["left"] = {"printing_id": a["printing_id"], "name": a["name"]}
    assert rec.legend_by_elimination("right") == b["card_id"] and rec.legend_by_elimination("left") is None
    rec.legends["left"] = {"printing_id": rows[3]["printing_id"], "name": rows[3]["name"]}
    assert rec.legend_by_elimination("right") is None  # lists for another match: neither is used


def test_with_lists_a_legend_on_neither_needs_a_sure_read():
    class Deck:
        def legends(self):
            return ["no-such-legend"]

    rows, art, rec = _setup(gate=False)
    rows[2]["type"] = "Legend"
    rec.row_of = {r["printing_id"]: r for r in rows}
    lg = rows[2]["card_id"]
    tr = Track("t0", CardBox((300.0, 100.0), 78.0, 56.0, 90.0, 1.0), 0.0, 10.0, hits=20, reads=4, prob={lg: 1.6},
               best_row={lg: (0.9, 2)})
    assert rec.label(tr)[0] == "named"  # four reads of a legend at 0.4: named, with no list
    rec.decks = [Deck()]
    assert rec.label(tr)[0] == "unsure"


def test_runes_are_counted_never_named_and_the_exhausted_ones_told():
    # from the player's seat a ready rune points straight at them and an exhausted (used) one lies across, along their
    # edge; players sit left and right of this table, so a ready rune lies across the picture
    rows, art, rec = _setup(gate=False)
    rows[0]["type"] = "Battlefield"
    rune, bf = rows[1]["card_id"], rows[0]["card_id"]

    def track(k, angle, kind, named, side="left", hits=5, pinned=False):
        return Track(f"t{k}", CardBox((100.0 + 60 * k, 300.0), 78.0, 56.0, angle, 1.0), 0.0, 10.0, hits=hits, reads=3,
                     prob={named: 2.9}, named=named, kind=kind, side=side, pinned=pinned, placed=True)

    rec.tracks = {tr.id: tr for tr in [track(0, 0.0, "Rune", rune), track(1, 178.0, "Rune", rune), track(2, 90.0, "Rune", rune),
                                       track(4, 91.0, "Rune", rune, "right"), track(5, 0.0, "Rune", rune, hits=1)]}
    assert not rec.ready_upright()  # LAYOUT puts the players left and right
    assert rec.runes(10.0, "left", False) == {"count": 3, "exhausted": 1}  # a box seen once is no rune
    assert rec.runes(10.0, "right", False) == {"count": 1, "exhausted": 1}
    assert rec.runes(30.0, "left", False)["count"] == 0  # not seen for 20 s: recycled, or the camera is elsewhere
    state = rec.state(10.0, 960, 540)
    assert {p["side"]: p["runes"] for p in state["players"]} == {"left": {"count": 3, "exhausted": 1}, "right": {"count": 1, "exhausted": 1}}
    assert "exhausted" not in state["tracks"][0]  # said of runes only
    rec.tracks["t9"] = track(9, 2.0, "Battlefield", bf, pinned=True)  # a battlefield lying across: the players sit above and below
    assert rec.ready_upright() and rec.runes(10.0, "left", True) == {"count": 3, "exhausted": 2}


def test_a_players_rune_count_holds_through_a_hand_a_box_between_runes_and_a_camera_cut():
    rows, art, rec = _setup(gate=False)
    rune = rows[1]["card_id"]

    def track(k, x, y, last, hits=5):
        return Track(f"t{k}", CardBox((x, y), 78.0, 56.0, 0.0, 1.0), 0.0, last, hits=hits, reads=3, prob={rune: 2.9},
                     named=rune, kind="Rune", side="left")

    counts, seen = [], []
    for k in range(20):
        t = 0.5 * k
        covered = 1.5 if 4 <= k < 7 else t  # a hand rests on two of the three runes for 1.5 s
        trs = [track(0, 100.0, 300.0, t), track(1, 160.0, 300.0, covered), track(2, 220.0, 300.0, covered),
               track(3, 104.0, 302.0, t - 0.5)]  # a second track on the first rune's card: one rune
        if k == 10:
            trs.append(track(4, 130.0, 340.0, t, hits=2))  # a box between two runes, for a frame
        rec.tracks = {tr.id: tr for tr in trs}
        seen.append(rec.runes_seen(t, "left", False)[0])
        counts.append(rec.runes(t, "left", False)["count"])
    assert seen[6] == 1 and seen[10] == 4  # frame by frame, the hand and the box move it
    assert counts == [3] * 20
    rec.away, rec.tracks = True, {}  # off the table camera: the count holds, and the board's clocks stop
    rec.pause(30.0)
    assert rec.runes(40.0, "left", False)["count"] == 3
    rec.away = False
    assert rec.runes(40.5, "left", False)["count"] == 3  # back: still the runes seen before the cut, until frames say otherwise
    for k in range(22):
        rec.runes(41.0 + 0.5 * k, "left", False)
    assert rec.runes(52.0, "left", False) == {"count": 0, "exhausted": 0}  # gone: recycled, or the camera moved


def test_a_stacks_covered_runes_are_counted_from_the_card_size_and_its_step_and_a_foil_rune_from_its_reads():
    rows, art, rec = _setup(gate=False)
    rec.frame_wh = (960, 540)  # 78 px cards
    rune, other = rows[1]["card_id"], rows[2]["card_id"]
    rec.type_of[rune] = "Rune"

    def track(k, x, kind="Rune", prob=None, side="left"):
        return Track(f"t{k}", CardBox((x, 300.0), 78.0, 56.0, 90.0, 1.0), 0.0, 10.0, hits=5, reads=4 if prob else 0,
                     prob=prob or {}, kind=kind, side=side, placed=True)

    # a fan of six runes a quarter of a card apart: the detector boxed five, the strip between the third and the fifth is missing
    fan = [track(0, 100.0), track(1, 120.0), track(2, 140.0), track(3, 180.0), track(4, 200.0)]
    rec.tracks = {tr.id: tr for tr in fan}
    assert rec.runes_now(10.0, "left") == 6
    assert hidden_runes([tr.box for tr in fan], 78.0) == 1 and hidden_runes([tr.box for tr in fan[:2]], 78.0) == 0  # two: no step
    # a covered strip boxed but not read joins the runes beside it; a card two cards away does not
    rec.tracks.update({tr.id: tr for tr in [track(2, 140.0, kind=""), track(5, 300.0, kind="")]})
    assert rec.runes_now(10.0, "left") == 6
    # a foil rune read as a spell counts on the share of its reads that say rune; a spell that only looks like one does not
    rec.tracks.update({tr.id: tr for tr in [track(6, 600.0, "Spell", {rune: 1.6, other: 2.4}, "right"),
                                            track(7, 800.0, "Spell", {rune: 0.8, other: 3.2}, "right")]})
    assert rec.runes_now(10.0, "right") == 1


def test_a_box_seen_once_is_neither_shown_nor_read():
    rows, art, rec = _setup(gate=False)
    a = (art[0], 100, 100)
    state, _ = _run(rec, lambda t: ([a], [a] if t == 1.0 else []), 3)
    assert state["tracks"] == [] and all(tr.reads == 0 for tr in rec.tracks.values())


def test_boxes_are_eased_while_a_card_barely_moves():
    from rifteye_ml.live.pipeline import smooth

    old = CardBox((100.0, 100.0), 78.0, 56.0, 90.0, 1.0)
    assert smooth(old, CardBox((104.0, 100.0), 78.0, 56.0, 92.0, 1.0)).centre[0] == pytest.approx(101.4)
    assert smooth(old, CardBox((150.0, 100.0), 78.0, 56.0, 90.0, 1.0)).centre == (150.0, 100.0)  # moved: followed


def test_off_the_table_camera_nothing_is_looked_at_and_the_board_waits():
    rows, art, rec = _setup(gate=False)
    a = (art[0], 100, 100)
    looked = []
    player_cam = np.asarray(Image.new("RGB", (960, 540), (200, 30, 120)))
    events, away = [], []
    for k in range(int(100 * 5)):
        t = k / 5
        rec.finder = lambda t_, im: looked.append(t_) or _boxes([a])
        state, ev = rec.step(t, player_cam if 5 <= t < 95 else _frame([a]))  # 90 s on the players
        events += ev
        if 5 <= t < 95:
            away.append(state["status"] == "away" and all(tr["hidden"] for tr in state["tracks"]))
    assert all(away) and not [lt for lt in looked if 5 <= lt < 95]  # no card looked at while away (D-005)
    named = [tr for tr in state["tracks"] if tr["state"] == "named"]
    assert [tr["id"] for tr in named] == ["t0"] and state["status"] == "live" and events == []


def test_a_view_framed_anew_keeps_every_id_by_name():
    rows, art, rec = _setup(gate=False)
    spots = [(100, 100), (250, 100), (400, 100), (100, 300), (250, 300), (400, 300)]

    def plan(t):
        dx, dy = (120, 60) if t >= 8 else (0, 0)  # the table camera pans at 8 s
        cards = [(art[i], x + dx, y + dy) for i, (x, y) in enumerate(spots)]
        return cards, cards

    state, events = _run(rec, plan, 16)
    named = {tr["name"]: tr["id"] for tr in state["tracks"] if tr["state"] == "named"}
    assert named == {rows[i]["name"]: f"t{i}" for i in range(6)}  # every card keeps the id it had before
    assert all(not tr["hidden"] for tr in state["tracks"]) and events == []


def test_a_box_across_two_cards_side_by_side_is_dropped():
    from rifteye_ml.live.pipeline import detector_boxes

    def card(x, score, visible):
        return {"cls": "card", "score": score, "visible": visible, "quad": [x, 100, x + 56, 100, x + 56, 178, x, 178]}

    left, right = card(100, 0.95, [1, 1, 1, 1]), card(158, 0.93, [1, 1, 1, 1])
    across = card(129, 0.6, [0.9, 0.9, 0.9, 0.9])  # half on each, claiming to be whole
    under = card(129, 0.6, [0.1, 0.9, 0.9, 0.1])   # the same box with covered corners: a card under the two
    assert len(detector_boxes([left, right, across])) == 2
    assert len(detector_boxes([left, right, under])) == 3


def test_a_card_in_a_case_outlined_three_times_is_one_card():
    from rifteye_ml.live.pipeline import detector_boxes

    def outline(pad, score):  # the card, and a magnetic case's inner and outer edge around it
        x0, y0, x1, y1 = 100 - pad, 100 - pad, 156 + pad, 178 + pad
        return {"cls": "card", "score": score, "visible": [1, 1, 1, 1], "quad": [x0, y0, x1, y0, x1, y1, x0, y1]}

    boxes = detector_boxes([outline(8, 0.9), outline(0, 0.7), outline(4, 0.8)])
    assert len(boxes) == 1 and boxes[0].long_px == pytest.approx(78)  # the card itself stays


def test_a_legend_read_the_same_way_four_times_is_named_with_less_certainty():
    rows, art, rec = _setup()
    rows[1]["type"] = "Legend"
    box = CardBox((100.0, 100.0), 78.0, 56.0, 90.0, 1.0)
    legend = Track("a", box, 0.0, 0.0, reads=4, prob={rows[1]["card_id"]: 1.4, rows[2]["card_id"]: 1.2})  # 0.35 vs 0.30
    unit = Track("b", box, 0.0, 0.0, reads=4, prob={rows[2]["card_id"]: 1.4, rows[3]["card_id"]: 1.2})
    assert rec.label(legend)[0] == "named" and rec.label(unit)[0] == "unsure"


def test_a_pinned_card_keeps_its_name_and_a_second_outline_on_a_legend_is_no_card():
    rows, art, rec = _setup(gate=False, recheck_s=1.0)
    rows[1]["type"], rows[2]["type"] = "Legend", "Battlefield"
    rec.row_of = {r["printing_id"]: r for r in rows}
    legend, bf, unit = CardBox((328.0, 139.0), 78.0, 56.0, 90.0, 1.0), (468.0, 139.0), (474.0, 143.0)

    def boxes(k, t):
        found = [CardBox(legend.centre, 78.0, 56.0, 90.0, 1.0), CardBox(bf, 78.0, 56.0, 90.0, 1.0)]
        if k % 2:  # the detector's other outline of the legend's case, off-centre around a die
            found.append(CardBox((340.0, 146.0), 66.0, 58.0, 90.0, 1.0))
        if t >= 10:  # a unit put on the battlefield
            found.append(CardBox(unit, 78.0, 56.0, 90.0, 1.0))
        return found

    for k in range(100):  # 20 s at 5 fps; from 5 s on, every re-read sees other pictures there (dice, a hand)
        t = k / 5
        faces = [(art[1] if t < 5 else art[4], 300, 100), (art[2] if t < 5 else art[5], 440, 100)]
        rec.finder = lambda t_, im, k=k, t=t: boxes(k, t)
        state, _ = rec.step(t, _frame(faces + ([(art[3], 446, 104)] if t >= 10 else [])))
        on_legend = [tr for tr in state["tracks"] if math.dist(np.mean(tr["quad"], axis=0), legend.centre) < 30]
        if t >= 1:
            assert [(tr["kind"], tr["state"], tr["name"]) for tr in on_legend] == [("legend", "named", rows[1]["name"])]
    by_name = {tr["name"]: tr for tr in state["tracks"] if tr["state"] == "named"}
    assert by_name[rows[2]["name"]]["kind"] == "battlefield" and rows[3]["name"] in by_name  # the unit on it is a card
    assert rows[4]["name"] not in by_name and rows[5]["name"] not in by_name
    rows[0]["type"] = "Legend"  # something else on the legend's side read as another legend with less certainty
    other = Track("x", CardBox((100.0, 400.0), 78.0, 56.0, 90.0, 1.0), 0.0, 0.0, reads=4, side="left",
                  prob={rows[0]["card_id"]: 1.4, rows[3]["card_id"]: 1.2})
    assert rec.legends["left"]["name"] == rows[1]["name"] and rec.label(other)[0] == "unsure"  # one player, one legend
    twin = Track("y", CardBox((330.0, 200.0), 78.0, 56.0, 90.0, 1.0), t, t, hits=5, reads=3, side="left",
                 prob={rows[1]["card_id"]: 3.0})  # a sure read of the legend itself, somewhere else on its side
    assert rec.label(twin)[0] == "unsure"
    rec.tracks["y"] = twin
    assert "y" not in [tr["id"] for tr in rec.state(t, 640, 360)["tracks"]]  # another outline of it: not drawn


def test_a_card_tucked_under_another_keeps_its_name_and_shows_under_it():
    rows, art, rec = _setup(gate=False, recheck_s=0.5)
    rows[3]["type"], rows[4]["type"] = "Gear", "Unit"
    rec.row_of = {r["printing_id"]: r for r in rows}
    gear, unit = (328.0, 139.0), (328.0, 169.0)

    def boxes(t):  # the gear, and from 4 s the unit laid over most of it
        return [CardBox(gear, 78.0, 56.0, 90.0, 1.0)] + ([CardBox(unit, 78.0, 56.0, 90.0, 1.0)] if t >= 4 else [])

    reads = None
    for k in range(60):  # 12 s at 5 fps; the unit goes on the gear at 4 s
        t = k / 5
        rec.finder = lambda t_, im, t=t: boxes(t)
        state, _ = rec.step(t, _frame([(art[3], 300, 100)] + ([(art[4], 300, 130)] if t >= 4 else [])))
        if k == 25:
            reads = next(tr.reads for tr in rec.tracks.values() if tr.named == rows[3]["card_id"])
    by_name = {tr["name"]: tr for tr in state["tracks"] if tr["state"] == "named"}
    assert rows[3]["name"] in by_name and rows[4]["name"] in by_name  # the gear keeps its name under the unit
    assert [u["name"] for u in by_name[rows[4]["name"]]["under"]] == [rows[3]["name"]]
    assert by_name[rows[3]["name"]]["under"] == []
    assert next(tr.reads for tr in rec.tracks.values() if tr.named == rows[3]["card_id"]) == reads  # not read again


def test_a_card_outlined_twice_is_drawn_once():
    rows, art, rec = _setup(gate=False, recheck_s=0.5)
    rows[3]["type"] = "Unit"
    rec.row_of = {r["printing_id"]: r for r in rows}
    for k in range(30):  # the detector outlines one card twice, a few pixels apart (a toploader's edge)
        t = k / 5
        rec.finder = lambda t_, im: [CardBox((328.0, 139.0), 78.0, 56.0, 90.0, 1.0), CardBox((331.0, 142.0), 82.0, 60.0, 90.0, 1.0)]
        state, _ = rec.step(t, _frame([(art[3], 300, 100)]))
    assert sum(1 for tr in rec.tracks.values() if tr.named == rows[3]["card_id"]) == 2  # both tracks named the card
    assert [tr["name"] for tr in state["tracks"] if tr["state"] == "named"] == [rows[3]["name"]]  # shown once


def _split(t, card, copy_from, copy_at, outline=(15, 99)):
    """A card from the start; from `outline[0]` to `outline[1]` s a hand passing over it splits it into a second outline
    a few pixels off (the detector outlines it twice); from `copy_from` s a second copy of it at `copy_at`."""
    shown, found = [(card, 100, 100)], [CardBox((128.0, 139.0), 78.0, 56.0, 90.0, 1.0)]
    if outline[0] <= t < outline[1]:
        found.append(CardBox((133.0, 144.0), 82.0, 60.0, 90.0, 1.0))
    if t >= copy_from:
        shown.append((card, *copy_at))
        found.append(CardBox((copy_at[0] + 28.0, copy_at[1] + 39.0), 78.0, 56.0, 90.0, 1.0))
    return shown, found


def _outline(rec):
    return next(tr for tr in rec.tracks.values() if math.dist(tr.box.centre, (133.0, 144.0)) < 3)


def test_a_second_outline_of_a_card_is_no_play_and_a_copy_put_beside_it_still_is():
    rows, art, rec = _setup(gate=False)
    events = []
    for k in range(125):  # 25 s at 5 fps: the card from the start, its second outline from 15 s, a copy beside it from 20 s
        t = k / 5
        shown, found = _split(t, art[2], 20, (180, 100))
        rec.finder = lambda t_, im, found=found: found
        events += rec.step(t, _frame(shown))[1]
    assert _outline(rec).named == rows[2]["card_id"]  # the outline is named as the card, ...
    played = [e for e in events if e["kind"] == "played"]
    assert [e["text"] for e in played] == [f"{rows[2]['name']} played"]  # ... but only the copy is a play
    assert played[0]["t"] >= 20 and played[0]["track"] != _outline(rec).id


def test_a_second_outline_gone_again_is_not_the_card_moved():
    rows, art, rec = _setup(gate=False)
    events = []
    for k in range(110):  # 22 s: the card outlined twice from 15 to 17 s, then a copy of it put far off from 18 s
        t = k / 5
        shown, found = _split(t, art[2], 18, (600, 300), outline=(15, 17))
        rec.finder = lambda t_, im, found=found: found
        events += rec.step(t, _frame(shown))[1]
    outline = _outline(rec)
    assert outline.named == rows[2]["card_id"] and 16 < outline.last < 17.5  # named, then out of sight where the card lies
    assert [(e["kind"], e["text"]) for e in events if e["kind"] in ("played", "moved")] == [
        ("played", f"{rows[2]['name']} played")]  # a new card, not the outline moved


def test_a_card_named_over_half_of_one_like_it_is_that_card_however_far_off_its_centre():
    # a hand straightening a card left it outlined again a quarter of a card off on the Los Angeles final: one card, no play
    rows, art, rec = _setup(gate=False)
    events = []
    for k in range(100):  # 20 s: from 15 s the card shows again over two thirds of itself, a third of a card down
        t = k / 5
        shown, found = _split(t, art[2], 15, (100, 126), outline=(0, 0))
        rec.finder = lambda t_, im, found=found: found
        events += rec.step(t, _frame(shown))[1]
    assert sum(1 for tr in rec.tracks.values() if tr.named == rows[2]["card_id"]) == 2  # outlined twice, named twice
    assert not [e for e in events if e["kind"] == "played"]


# ---- the legend rule (D-026) --------------------------------------------------------------------------------
# test_decklist.py's rows; gallery row i is the unit vector i, and the stub encoder gives every crop the same scores,
# so a read's candidates are these scores less the rows the side's legend rules out. The engine's recognizer.test.ts
# replays the same reads.

RULE_SCORES = {"OGN-914": 0.9, "OGN-920": 0.85, "OGN-918": 0.8, "SFD-902a": 0.75, "VEN-906": 0.7}  # the rest 0.1


def _rule_setup(**kw):
    from rifteye_ml.retrieval import Pyramid
    from test_decklist import ROWS

    rows = [dict(r) for r in ROWS]
    scores = np.full(len(rows), 0.1, np.float32)
    for i, r in enumerate(rows):
        scores[i] = RULE_SCORES.get(r["printing_id"], 0.1) if r["language"] == "en" else 0.1

    class Stub:
        name, dim = "stub", len(rows)

        def embed(self, images):
            return np.tile(scores, (len(images), 1))

    return rows, Recognizer(LAYOUT, rows, Stub(), Pyramid({80: np.eye(len(rows), dtype=np.float32)}), fps=5.0, gate=False, **kw)


def test_the_legend_rule_holds_a_side_to_its_legend():
    from rifteye_ml import priors

    rows, rec = _rule_setup()
    crop = Image.new("RGB", (56, 78), (200, 100, 50))
    before = rec.identify([crop], ["left"])[0]
    assert [c for c, *_ in before[:3]] == ["blaze-fist", "ember-rune", "hush-rune"] and rec.masks == {}
    rec.legends["left"] = {"printing_id": "SFD-901", "name": "Gleaming Anvil"}   # Calm and Mind, pinned on the left
    left, right, nowhere = rec.identify([crop] * 3, ["left", "right", ""])
    assert [c for c, *_ in left[:2]] == ["hush-rune", "fakesmith-hammerer"]    # Fury and Order+Chaos ruled out
    assert left[1][3] == 5                                                      # SFD-902a, its best printing
    allowed = priors.legend_mask(rows, ["gleaming-anvil"])[0]
    assert {c for c, *_ in left} == {r["card_id"] for r, ok in zip(rows, allowed) if ok}
    assert [c for c, *_ in right] == [c for c, *_ in before] == [c for c, *_ in nowhere]
    # the softmax runs over the cards left: hush-rune against the other allowed cards only
    vals = np.array([sc for _, _, sc, _ in left])
    p = np.exp((vals - vals.max()) / rec.temperature)
    assert np.allclose([pc for _, pc, _, _ in left], p / p.sum()) and left[0][1] > before[2][1]
    assert set(rec.masks) == {"gleaming-anvil"} and rec.allowed("left") is rec.masks["gleaming-anvil"]
    assert rec.allowed("right") is None and rec.allowed("") is None and rec.allowed("top") is None
    assert rec.identify([crop])[0] == before                                     # no sides: the whole gallery


def test_a_pasted_list_holds_the_side_whose_legend_it_names():
    from rifteye_ml import decklist

    crop = Image.new("RGB", (56, 78), (200, 100, 50))
    _, plain = _rule_setup()
    plain.legends["right"] = {"printing_id": "VEN-912", "name": "Thunder Crown"}
    rule_right = plain.identify([crop], ["right"])[0]
    _, rec = _rule_setup()
    rec.legends["left"] = {"printing_id": "SFD-901", "name": "Gleaming Anvil"}
    rec.legends["right"] = {"printing_id": "VEN-912", "name": "Thunder Crown"}
    # the Gleaming Anvil list, blaze-fist on its side board; the other player's list is not given
    lst = decklist.parse("1 Fakesmith - Gleaming Anvil (SFD-901)\n3 Fakesmith - Hammerer (SFD-902)\n7 Hush Rune (OGN-918)\n"
                         "1 Quiet Glade (OGN-907)\nSide Board:\n2 Blaze Fist (OGN-914)", rec.catalogue())
    rec.set_lists([lst])
    left, right = rec.identify([crop, crop], ["left", "right"])
    cards = [c for c, *_ in left]
    assert cards[:3] == ["blaze-fist", "hush-rune", "fakesmith-hammerer"]   # listed, in every printing
    assert "ember-rune" not in cards and "spark-bolt" not in cards          # what the legend alone allowed, off the list
    assert "quiet-glade" in cards and "wisp" in cards                       # its battlefield, and the tokens
    assert [c for c, *_ in right] == [c for c, *_ in rule_right]            # no list names Thunder Crown: the legend rule
    rec.set_lists([])
    assert [c for c, *_ in rec.identify([crop], ["left"])[0][:2]] == ["hush-rune", "fakesmith-hammerer"]


def test_a_tracks_crops_follow_its_side_and_the_rule_can_be_turned_off():
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 256, (540, 960, 3), dtype=np.uint8))
    for rule in (True, False):
        rows, rec = _rule_setup(legend_rule=rule)
        rec.legends["left"] = {"printing_id": "SFD-901", "name": "Gleaming Anvil"}
        a = Track("a", CardBox((200.0, 200.0), 78.0, 56.0, 90.0, 1.0), 0.0, 0.0, hits=2, side="left")
        b = Track("b", CardBox((700.0, 200.0), 78.0, 56.0, 90.0, 1.0), 0.0, 0.0, hits=2, side="right")
        rec.read(0.0, noise, [a, b])
        assert a.reads == b.reads == 1
        assert ("blaze-fist" not in a.prob) is rule and "blaze-fist" in b.prob
        assert max(a.prob, key=a.prob.get) == ("hush-rune" if rule else "blaze-fist")


def _on(background, cards, size=(960, 540)):
    """A frame of one colour with each (image, x, y) pasted as a 56 x 78 card."""
    im = Image.new("RGB", size, background)
    for card, x, y in cards:
        im.paste(card.resize((56, 78), Image.BOX), (x, y))
    return im


def test_on_a_wooden_table_a_card_beside_the_wood_is_read_and_one_in_a_hand_still_is_not():
    # a co-stream: the table is wood, and wood is skin-coloured; to the hand rule every card beside it was in a
    # hand, so legends, champions and battlefields were never named. A hand is what is not the still table.
    from rifteye_ml.changegate import skin
    from rifteye_ml.live.pipeline import hand_share

    wood, fingers = (170, 120, 100), (230, 185, 160)
    assert skin(np.array([[wood, fingers]], np.uint8)).all()  # both skin to the colour test alone
    rows, art, base = _setup(gate=False)
    layout = Layout("wood", "a wooden table", (0.0, 0.0, 1.0, 1.0), card_long_1080=156, mat=wood, mat_share=0.5)
    rec = Recognizer(layout, rows, base.enc, base.gallery, fps=5.0, gate=False)
    lying, held = CardBox((378.0, 239.0), 78.0, 56.0, 90.0, 1.0), CardBox((628.0, 239.0), 78.0, 56.0, 90.0, 1.0)
    rec.finder = lambda t, image: [lying] + ([held] if t >= 6 else [])
    named_at, held_reads = {}, []
    for k in range(75):  # 15 s at 5 fps: card 0 lies on the wood throughout; a hand brings card 1 at 6 s, puts it down at 10 s
        t = k / 5
        im = _on(wood, [(art[0], 350, 200)] + ([(art[1], 600, 200)] if t >= 6 else []))
        if 6 <= t < 10:
            im.paste(fingers, (576, 200, 600, 278))
        state, _ = rec.step(t, np.asarray(im))
        for tr in state["tracks"]:
            if tr["state"] == "named":
                named_at.setdefault(tr["name"], t)
        if 6 <= t < 10:
            held_reads += [tr.reads for tr in rec.tracks.values() if tr.box.centre[0] > 500]
    assert named_at[rows[0]["name"]] < 4  # named once the still table is known (2.4 s) and it has lain still 0.5 s
    assert held_reads and not any(held_reads)  # never read while held
    assert 10 <= named_at[rows[1]["name"]] < 12
    plain = np.asarray(_on(wood, [(art[0], 350, 200)]))
    assert hand_share(plain, lying, layout.box(960, 540)) == 1.0  # the colour test alone: all wood, all "skin"
    from rifteye_ml.live.pipeline import StillTable
    small = StillTable.small(plain)
    assert hand_share(plain, lying, layout.box(960, 540), still=small, now=small) == 0.0
    held = np.asarray(_on(wood, [(art[0], 350, 200)]))
    held = held.copy()
    held[200:278, 326:350] = fingers
    assert hand_share(held, lying, layout.box(960, 540), still=small, now=StillTable.small(held)) >= 0.1


def test_what_stays_put_through_the_cuts_is_the_overlay_and_nothing_on_it_is_read():
    # a co-streamer's webcam in the corner of every shot: read as a legend, a battlefield and a play; and kept in the
    # scene's score, it made close-ups look like the table
    from rifteye_ml.live.pipeline import OVERLAY_CUTS

    rows, art, rec = _setup(gate=False)
    rng = np.random.default_rng(5)
    webcam = Image.fromarray(rng.integers(40, 200, (120, 140, 3), dtype=np.uint8))
    card = CardBox((378.0, 239.0), 78.0, 56.0, 90.0, 1.0)
    on_cam = CardBox((888.0, 479.0), 78.0, 56.0, 90.0, 1.0)  # the card the co-streamer holds up to the webcam from 4 s
    rec.finder = lambda t, image: [card] + ([on_cam] if t >= 4 else [])
    table = [(0, 5), (6, 8), (9, 11), (12, 15)]  # the table camera; between, other shots (four cuts by 9 s)
    status, cam_tracks, events = {}, {}, []
    for k in range(75):
        t = k / 5
        if any(a <= t < b for a, b in table):
            im = _on(MAT, [(art[0], 350, 200)])
        else:
            im = Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))  # a player cam
        cam = webcam.copy()
        if t >= 4:
            cam.paste(art[1].resize((56, 78), Image.BOX), (40, 20))
        im.paste(cam, (820, 420))
        state, ev = rec.step(t, np.asarray(im))
        events += ev
        status[t] = state["status"]
        cam_tracks[t] = [tr for tr in rec.tracks.values() if tr.box.centre[0] > 800]
    assert rec.scene.cuts >= OVERLAY_CUTS and rec.scene.overlay is not None
    assert rec.scene.in_overlay(888, 479, 960, 540) and not rec.scene.in_overlay(378, 239, 960, 540)
    assert cam_tracks[4.8] and not any(cam_tracks[t] for t in cam_tracks if t >= 9.2)  # read off the webcam, then never
    assert all(status[t] == "away" for t in status if 11.2 <= t < 12) and status[13.0] == "live"
    played = [e for e in events if e["kind"] == "played"]
    withdrawn = [e for e in events if e["kind"] == "withdrawn"]
    assert [e["text"] for e in played] == [f"{rows[1]['name']} played"]  # read off the webcam before the cuts showed it
    assert [e["track"] for e in withdrawn] == [played[0]["track"]] and 9 <= withdrawn[0]["t"] < 10


def test_a_zone_printed_on_the_mat_is_not_a_card_and_a_box_longer_than_a_card_is_none():
    from rifteye_ml.live.pipeline import plain_zone

    rows, art, rec = _setup(gate=False)
    im = _on(MAT, [(art[0], 350, 200)])
    from PIL import ImageDraw
    ImageDraw.Draw(im).rectangle((600, 200, 656, 278), outline=(150, 130, 70), width=2)  # a card-sized zone, printed
    im.paste((220, 90, 150), (200, 200, 256, 278))  # a face-down card in a pink sleeve
    frame = np.asarray(im)
    zone, card, sleeve = (CardBox((628.0, 239.0), 78.0, 56.0, 90.0, 1.0), CardBox((378.0, 239.0), 78.0, 56.0, 90.0, 1.0),
                          CardBox((228.0, 239.0), 78.0, 56.0, 90.0, 1.0))
    assert plain_zone(frame, zone) and not plain_zone(frame, card) and not plain_zone(frame, sleeve)
    two = CardBox((450.0, 239.0), 160.0, 78.0, 0.0, 1.0)  # two cards side by side outlined as one: 2 cards long
    rec.finder = lambda t, image: [zone, card, sleeve, two]
    for k in range(10):
        rec.step(k / 5, frame)
    assert sorted(round(tr.box.centre[0]) for tr in rec.tracks.values()) == [228, 378]


def test_a_burst_of_cards_named_at_once_is_not_that_many_plays():
    # a sideboard graphic's eight cards, all "played" at once: nobody plays four cards in a second
    rows, art, rec = _setup(n=7, gate=False)
    events = []
    for k in range(70):  # 14 s at 5 fps: card 0 from the start; cards 1 to 5 all at once at 5 s; card 6 alone at 9 s
        t = k / 5
        cards = [(art[0], 100, 100)]
        if t >= 5:
            cards += [(art[i], 100 + 150 * i, 300) for i in range(1, 6)]
        if t >= 9:
            cards.append((art[6], 400, 100))
        _, ev = rec.step(t, _frame(cards))
        events += ev
    played = [e["text"] for e in events if e["kind"] == "played"]
    assert played == [f"{rows[6]['name']} played"]


def test_a_face_moving_in_a_corner_webcam_keeps_the_webcam_overlay():
    # a co-stream: the co-streamer's webcam in the bottom right corner, and their face reaching the frame's bottom
    # edge, so a hole open to it is never filled. The face moved at the fifth cut: on its own, that cut opened the
    # webcam to the table, and the card on its wall was read and announced again 15 s later
    from rifteye_ml.live.pipeline import overlay_patches

    same = np.zeros((54, 96), np.int32)
    same[40:, 80:] = 4  # the webcam stayed through the first four cuts
    first = overlay_patches(same, 4)
    assert first[40:, 80:].all() and int(first.sum()) == 14 * 16
    same[40:, 80:] = 5
    same[44:, 85:91] = 4  # the fifth cut: the face moved
    assert not overlay_patches(same, 5)[48, 88]
    assert overlay_patches(same, 5, first)[40:, 80:].all()  # what was overlay stays, while it stays through 60%
    same[44:, 85:91] = 2  # what changed at three cuts of five is not
    assert not overlay_patches(same, 5, first)[48, 88]


def test_the_frames_the_table_was_looked_for_in_hold_the_overlay_back_from_the_start():
    # a co-stream: Wardeye found the table at the cut from the player cams to it, then read the webcam for 25 s
    # before enough cuts showed it; the cut among the frames it looked in shows the webcam at once
    from rifteye_ml.live.pipeline import SUSPECT_S

    rows, art, rec = _setup(gate=False)
    rng = np.random.default_rng(5)
    webcam = Image.fromarray(rng.integers(40, 200, (120, 140, 3), dtype=np.uint8))
    cam = Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))
    cam.paste(webcam, (820, 420))
    table = _on(MAT, [(art[0], 350, 200)])
    table.paste(webcam, (820, 420))
    rec.prime([np.asarray(cam)] * 4 + [np.asarray(table)])  # four looks at a player cam, then the table
    assert (rec.scene.cuts, rec.scene.cuts_all) == (0, 1)  # suspect from the looks' cut, never overlay from it alone
    assert rec.scene.in_suspect(888, 479, 960, 540) and not rec.scene.in_suspect(378, 239, 960, 540)
    card = CardBox((378.0, 239.0), 78.0, 56.0, 90.0, 1.0)
    on_cam = CardBox((888.0, 479.0), 78.0, 56.0, 90.0, 1.0)
    rec.finder = lambda t, image: [card, on_cam]
    events, shown = [], set()
    for k in range(50):  # 10 s of the table camera, the webcam showing a card from the start
        t = k / 5
        im = table.copy()
        shot = webcam.copy()
        shot.paste(art[1].resize((56, 78), Image.BOX), (40, 20))
        im.paste(shot, (820, 420))
        state, ev = rec.step(t, np.asarray(im))
        events += ev
        shown |= {tr["name"] for tr in state["tracks"] if tr["state"] == "named"}
    assert rows[0]["name"] in shown and rows[1]["name"] not in shown  # the table's card named, the webcam's held back
    assert not [e for e in events if e["kind"] in ("played", "withdrawn")]
    assert rec.scene.suspect is not None  # no cut since: held back for SUSPECT_S
    rec.scene.on_table(10 + SUSPECT_S + 1, np.asarray(table))
    assert rec.scene.suspect is None  # and then let go: one cut alone can be the camera reframed


def _mat_frame(shift=0, arm=False):
    """A 1920 x 1080 frame of la-rq's table window: the mat with a print of 40 px blocks (moved `shift` px when the
    camera moves), plain but for a strong band along its top, like the score track printed on the Riftbound mat;
    `arm`, a dark sleeve over that band."""
    from rifteye_ml.live.layouts import LAYOUTS
    lay = LAYOUTS["la-rq"]
    y, x = np.mgrid[0:1080, 0:1920]
    img = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, x1, y1 = lay.box(1920, 1080)
    bx, by = (x + shift) // 40, y // 40
    for c in range(3):
        v = bx * bx * 37 + by * by * 91 + bx * by * 13 + c * 50
        img[..., c] = np.where((y < 340) & (x >= 1160), v % 200, lay.mat[c] + v % 32)
    img[:y0], img[:, :x0], img[:, x1:] = 90, 90, 90  # the panels beside the window
    if arm:
        img[y0:340, 1160:x1] = 12
    return lay, img


def test_an_arm_over_the_table_leaves_it_the_table_and_a_close_up_of_one_side_is_never_learnt_as_it():
    # the Barcelona final: with the panels left out of the score, an arm over the mat's printed score track took the
    # table for another shot for 9 s. And a co-stream: away from the table camera for 20 s, the scene learnt a player
    # cam as the table, its maroon shirt the mat's colour; then a close-up of one side of the table, with a few of its
    # cards, that the detector boxed
    from rifteye_ml.live.pipeline import Scene

    lay, table = _mat_frame()
    _, arm = _mat_frame(arm=True)
    _, moved = _mat_frame(shift=40)
    for side_cards, learnt in ((8, False), (12, True)):
        sc = Scene(lay)
        on = [sc.on_table(0.5 * k, table, lambda: 20) for k in range(21)]
        for k in range(21):
            sc.saw_cards(0.5 * k, 20)  # the board saw twenty cards on the table camera
        on += [sc.on_table(10.5 + 0.5 * k, arm, lambda: 20) for k in range(10)]
        assert all(on), "an arm over a block of the table is still the table"
        away = [sc.on_table(15.5 + 0.5 * k, moved, lambda: side_cards) for k in range(80)]
        assert not away[0]
        assert away[-1] is learnt, f"{side_cards} cards where the table showed 20"
