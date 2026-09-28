import math

import numpy as np
import pytest
from PIL import Image

from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog
from rifteye_ml.live.layouts import LAYOUTS, Layout
from rifteye_ml.live.pipeline import Recognizer, Track, card_crop, quad
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
    bf, lg, rune = (art[0], 100, 100), (art[1], 300, 100), (art[2], 500, 100)

    def plan(t):
        if t < 4:
            return [], []
        return [bf, lg, rune], ([bf, lg, rune] if t < 8 else [rune])  # then a hand over the legend and battlefield

    state, events = _run(rec, plan, 14)
    by_kind = {tr["kind"]: tr for tr in state["tracks"]}
    assert by_kind["battlefield"]["name"] == rows[0]["name"] and not by_kind["battlefield"]["hidden"]
    assert by_kind["legend"]["name"] == rows[1]["name"] and by_kind["rune"]["name"] == rows[2]["name"]
    assert events == []  # set-up cards and runes are never announced


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
    legend, bf, unit = CardBox((328.0, 139.0), 78.0, 56.0, 90.0, 1.0), (628.0, 139.0), (634.0, 143.0)

    def boxes(k, t):
        found = [CardBox(legend.centre, 78.0, 56.0, 90.0, 1.0), CardBox(bf, 78.0, 56.0, 90.0, 1.0)]
        if k % 2:  # the detector's other outline of the legend's case, off-centre around a die
            found.append(CardBox((340.0, 146.0), 66.0, 58.0, 90.0, 1.0))
        if t >= 10:  # a unit put on the battlefield
            found.append(CardBox(unit, 78.0, 56.0, 90.0, 1.0))
        return found

    for k in range(100):  # 20 s at 5 fps; from 4 s on, every re-read sees other pictures there (dice, a hand)
        t = k / 5
        faces = [(art[1] if t < 4 else art[4], 300, 100), (art[2] if t < 4 else art[5], 600, 100)]
        rec.finder = lambda t_, im, k=k, t=t: boxes(k, t)
        state, _ = rec.step(t, _frame(faces + ([(art[3], 606, 104)] if t >= 10 else [])))
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
