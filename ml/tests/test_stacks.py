import json

import numpy as np
from PIL import Image

from rifteye_ml.stacks import best_band, covers, inside, main, on_top, views_for, visible_mask
from rifteye_ml.synth.__main__ import main as synth


def _card(cx, cy, w=63.0, h=88.0):
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def _det(q, vis):
    return {"cls": "card", "score": 0.9, "quad": np.asarray(q).ravel().tolist(), "visible": vis}


def test_the_card_whose_corners_show_inside_the_other_is_on_top():
    lower, upper = _card(100, 100), _card(100, 130)  # a rune column: upper hides lower's bottom
    a = _det(lower, [1, 1, 0, 0])   # its bottom corners, inside the upper card, are covered
    b = _det(upper, [1, 1, 1, 1])   # its top corners, inside the lower card, show
    assert inside(upper[0] + [1, 1], lower) and not inside(upper[3], lower)
    assert on_top(b, a) is True and on_top(a, b) is False
    assert covers([a, b]) == [[1], []]


def test_the_visible_part_and_its_band():
    lower, upper = _card(100, 100), _card(100, 130)
    m = visible_mask(lower, [upper])
    assert abs(m.mean() - 30 / 88) < 0.03                 # the top 30 mm show
    assert best_band(m) == ("top", 0.25)                   # 34% visible: the top quarter is clear
    assert best_band(visible_mask(lower, [])) == ("full", 1.0)
    cut = visible_mask(lower, [], window=(0, 0, 1000, 100))  # the camera window ends halfway down
    assert abs(cut.mean() - 0.5) < 0.03 and best_band(cut) == ("top", 0.4)


def test_both_ways_up_are_tried_and_the_band_follows_the_turn():
    crop = Image.new("RGB", (63, 88))
    views = views_for(crop, ("top", 0.4))
    assert [(t, v) for t, _, v in views] == [(0, "top:0.4"), (180, "bottom:0.4")]
    side = views_for(Image.new("RGB", (88, 63)), ("left", 0.25))  # an exhausted card lies sideways
    assert [(t, v) for t, _, v in side] == [(90, "bottom:0.25"), (270, "top:0.25")]
    assert all(p.height < 88 * 0.3 for _, p, _ in side)


def test_exact_synthetic_detections_carry_the_truth(tmp_path):
    run = tmp_path / "run"
    assert synth(["--fixtures", "20", "--boards", "2", "--clip", "2", "--frames-per-board", "2", "--sizes", "960x540",
                  "--layouts", "full", "--out", str(run)]) == 0
    assert main(["synth-dets", "--run", str(run), "--out", str(tmp_path / "d.jsonl")]) == 0
    recs = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text().splitlines()]
    cards = [c for r in recs for c in r["cards"]]
    assert len(recs) == 2 and cards and all(len(c["visible"]) == 4 for c in cards)
    assert all("truth_card" not in c for c in cards if c["cls"] == "card_back")  # never an identity for a back
    assert any(sum(c["visible"]) < 4 for c in cards)                              # some corners are covered
