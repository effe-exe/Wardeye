import numpy as np
import pytest
from PIL import Image

from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog
from rifteye_ml.live.layouts import LAYOUTS, Layout
from rifteye_ml.live.pipeline import Recognizer, Track, card_crop, quad
from rifteye_ml.matcrops import CardBox

MAT = (30, 40, 55)
LAYOUT = Layout("test", "a test table", (0.0, 0.0, 1.0, 1.0), card_long_1080=156)  # 78 px cards in a 540 px frame


def _setup(n=6):
    pytest.importorskip("scipy")
    from rifteye_ml.encoders import get_encoder
    from rifteye_ml.retrieval import Pyramid, at_long_side

    rows = synthetic_catalog(n, seed=11)
    art = [load_fixture_image(r) for r in rows]
    enc = get_encoder("colorgrid/trim0.03+dhash/trim0.03")
    pyr = Pyramid({s: enc.embed([at_long_side(im, s) for im in art]) for s in (70, 80)})
    return rows, art, Recognizer(LAYOUT, rows, enc, pyr, fps=5.0)


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
