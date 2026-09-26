import json

import numpy as np
import pytest
from PIL import Image

from rifteye_ml.reviewpack import (apply_answers, card_scores, data_uri, event_view, fit_temperature, frame_time,
                                   hms, link_tracks, option_label, review_order, softmax, view_to_frame, wilson)


def test_frame_time_and_hms():
    assert frame_time("t03h20m40s_02.png") == 3 * 3600 + 20 * 60 + 40
    assert frame_time("seg-03h19m00s-04h19m30s-1080p60.mp4") == 3 * 3600 + 19 * 60
    assert hms(12040.4) == "3:20:40"
    with pytest.raises(ValueError):
        frame_time("crop.png")


def test_softmax_and_temperature_fit():
    p = softmax(np.array([0.9, 0.8, 0.1]), 0.05)
    assert p.sum() == pytest.approx(1) and p[0] > 0.8
    # Labelled crops whose true card is usually first but not always: the fit is neither
    # the coldest (overconfident) nor the hottest (uninformative) temperature on the grid.
    lists = [np.array([0.95, 0.90, 0.5])] * 8 + [np.array([0.95, 0.90, 0.5])] * 2
    truth = [0] * 8 + [1] * 2
    t = fit_temperature(lists, truth)
    assert 0.005 < t < 0.2
    assert softmax(lists[0], t)[0] == pytest.approx(0.8, abs=0.05)


def test_card_scores_keep_the_best_printing_per_card():
    rows = [{"card_id": "a"}, {"card_id": "a"}, {"card_id": "b"}]
    assert card_scores([1, 0, 2], [0.9, 0.8, 0.7], rows) == {"a": (0.9, 1), "b": (0.7, 2)}


def _crop(seg, t, x, y=100.0, long_px=131.0):
    return {"segment": seg, "t": t, "centre": (x, y), "long_px": long_px}


def test_link_tracks_follows_a_card_and_splits_new_ones():
    meta = [
        _crop("s1", 0, 100), _crop("s1", 10, 102), _crop("s1", 30, 101),  # one card, one frame missed
        _crop("s1", 70, 100),                                               # same spot, 40 s later: too late
        _crop("s1", 10, 400),                                               # another card, same frame
        _crop("s2", 10, 100),                                               # another segment
        _crop("s1", 40, 100),                                               # same spot, but a different card
    ]
    n = len(meta)
    sim = np.full((n, n), 0.99)
    sim[6, :] = sim[:, 6] = 0.5
    tracks = link_tracks(meta, sim, max_gap=30)
    assert tracks == [[0, 1, 2], [4], [5], [6], [3]]


def test_review_order_puts_uncertain_first_and_spreads_the_audit():
    conf = [0.9, 0.1, 0.99, 0.5, 0.999, 0.3, 0.95, 0.97, 0.98, 0.999, 0.2, 0.6]
    order, audit = review_order(conf, n=6, audit=0.34, seed=1)
    assert len(order) == 6 and len(set(order)) == 6 and len(audit) == 2
    uncertain = [i for i in order if i not in audit]
    assert uncertain == [1, 10, 5, 3]  # the four least confident, in order
    assert all(conf[i] >= 0.6 for i in audit)
    assert order[0] not in audit and order[-1] not in audit  # spread through, not bunched at an end
    everything, none = review_order(conf, n=50, audit=0.1)
    assert sorted(everything) == list(range(len(conf))) and none == set()


def test_wilson_interval():
    lo, hi = wilson(38, 40)
    assert 0.83 < lo < 0.84 and 0.98 < hi < 0.99
    assert wilson(0, 0) != wilson(0, 0)  # nan


def test_view_to_frame_maps_the_gate_view_to_pixels():
    table = [0.15, 0.10, 0.88, 0.884]
    x0, y0, x1, y1 = view_to_frame([0, 0, 320, 194], table, 320, 1920, 1080)
    assert (x0, y0) == pytest.approx((288, 108))
    assert (x1, y1) == pytest.approx((1689.6, 954.72), abs=0.5)


def test_option_label_and_data_uri():
    row = {"printing_id": "OGN-066a", "name": "阿狸", "variant": "alt_art", "language": "zh-Hans"}
    assert option_label(row, {"OGN-066a": "Ahri, Alluring"}) == "Ahri, Alluring · OGN-066a · alt art · zh-Hans"
    assert data_uri(Image.new("RGB", (8, 8), (200, 10, 10))).startswith("data:image/jpeg;base64,")


def test_apply_answers_labels_every_crop_of_a_track():
    side = {"pack": "p", "kind": "identity", "card_ids": {"OGN-002": "two"},
            "items": {
                "t1": {"files": ["a.png", "b.png"], "audit": False,
                       "proposal": {"printing_id": "OGN-001", "card_id": "one"}, "alternatives": []},
                "t2": {"files": ["c.png"], "audit": True,
                       "proposal": {"printing_id": "OGN-001", "card_id": "one"}, "alternatives": []},
                "t3": {"files": ["d.png"], "audit": False,
                       "proposal": {"printing_id": "OGN-003", "card_id": "three"}, "alternatives": []},
                "t4": {"files": ["e.png"], "audit": False,
                       "proposal": {"printing_id": "OGN-003", "card_id": "three"}, "alternatives": []},
                "t5": {"files": ["f.png"], "audit": False,
                       "proposal": {"printing_id": "OGN-003", "card_id": "three"}, "alternatives": []}}}
    answers = {"packId": "p", "reviewer": "fede", "answers": [
        {"itemId": "t1", "verdict": "correct"}, {"itemId": "t2", "verdict": "wrong", "value": "OGN-002"},
        {"itemId": "t3", "verdict": "wrong"}, {"itemId": "t4", "verdict": "unsure"},
        {"itemId": "t5", "verdict": "wrong", "text": "mech token"}, {"itemId": "gone", "verdict": "correct"}]}
    rows, stats = apply_answers(side, answers)
    got = [(r["file"], r["printing_id"], r["card_id"], r["labeled_by"]) for r in rows]
    assert got == [("a.png", "OGN-001", "one", "review:fede"), ("b.png", "OGN-001", "one", "review:fede"),
                   ("c.png", "OGN-002", "two", "review:fede"), ("d.png", "!OGN-003", "!three", "review:fede"),
                   ("e.png", "?", "?", "review:fede"), ("f.png", "none", "none", "review:fede")]
    assert rows[-1]["name"] == "mech token" and rows[0]["name"] == ""
    assert stats["review_n"] == 3 and stats["review_correct"] == pytest.approx(1 / 3) and stats["review_unsure"] == 1
    assert stats["audit_n"] == 1 and stats["audit_correct"] == 0.0
    with pytest.raises(ValueError):
        apply_answers(side, {**answers, "packId": "other"})


def test_apply_answers_for_events():
    side = {"pack": "e", "kind": "event", "items": {
        "e000": {"event": 0, "t": 7.8, "t_before": 3.4, "kind": "appeared", "box": [1, 2, 3, 4]},
        "e001": {"event": 1, "t": 9.0, "t_before": 8.0, "kind": "changed", "box": [1, 2, 3, 4]}}}
    rows, _ = apply_answers(side, {"packId": "e", "answers": [
        {"itemId": "e000", "verdict": "correct"}, {"itemId": "e001", "verdict": "wrong", "value": "none"}]})
    assert [(r["proposed"], r["verdict"], r["kind"]) for r in rows] == [("appeared", "correct", "appeared"),
                                                                        ("changed", "wrong", "none")]
    json.dumps(rows)  # plain data


def test_event_view_stays_inside_the_frame():
    frame = Image.new("RGB", (1920, 1080), (200, 30, 60))
    view = event_view(frame, (900, 5, 960, 60), min_side=400, long_side=10_000)
    px = np.asarray(view)
    assert (px.reshape(-1, 3).min(axis=0) > 0).all()  # no black padding anywhere
    assert view.height == 400 and view.width == 480
