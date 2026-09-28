import json

import numpy as np
import pytest

from rifteye_ml.detect import canonical_quad, export_run, quad_iou, tile_origins, tile_targets
from rifteye_ml.detect.evaluate import box_quad, match, score_frames, score_real
from rifteye_ml.detect.geometry import _signed
from rifteye_ml.detect.model import merge_tiles
from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog
from rifteye_ml.synth import Board, Instance, Occluder, Shot, render
from rifteye_ml.synth.__main__ import main as synth


def _card(cx, cy, angle_deg, w=63.0, h=88.0):
    """A card's corners as printed (TL, TR, BR, BL), turned by angle_deg on screen."""
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ rot.T + [cx, cy]


@pytest.mark.parametrize("angle", [0, 8, -11, 90, 97, 180, 173, 270, 262])
def test_corners_come_in_image_order_whatever_the_print_orientation(angle):
    q = canonical_quad(_card(100, 200, angle))
    assert q[0, 0] < 100 and q[0, 1] < 200            # up and left of the centre first
    assert _signed(q) > 0                              # then clockwise on screen
    np.testing.assert_allclose(canonical_quad(q[[2, 3, 0, 1]]), q)  # where the list starts does not matter


def test_quad_overlap():
    a = _card(0, 0, 0)
    assert quad_iou(a, a) == pytest.approx(1.0)
    assert quad_iou(a, a[::-1]) == pytest.approx(1.0)
    assert quad_iou(a, _card(500, 0, 0)) == 0.0
    assert quad_iou(a, _card(0, 44, 0)) == pytest.approx(44 / 132, abs=1e-6)  # half the height overlaps
    assert quad_iou(_card(0, 0, 45), _card(0, 0, 45)) == pytest.approx(1.0)
    assert 0.6 < quad_iou(a, _card(0, 20, 0)) < 0.7    # a rune column: stacked cards overlap a lot


def test_tiles_cover_the_window_with_overlap():
    assert tile_origins(400, 576, 0.2) == [0]
    o = tile_origins(1000, 576, 0.2)
    assert o[0] == 0 and o[-1] == 1000 - 576 and all(b - a <= 576 * 0.8 for a, b in zip(o, o[1:]))


def _clean_shot(**kw) -> Shot:
    base = dict(frame_w=640, frame_h=480, layout="full", window=(0, 0, 640, 480), card_px=150.0, turn=0, yaw_deg=0.0,
                keystone=0.0, k1=0.0, centre_mm=(0.0, 0.0), defocus_px=0.0, wb_jitter=0.0, exposure_jitter=0.0,
                gamma_jitter=0.0, light_jitter=0.0, featured=False)
    base.update(kw)
    return Shot(**base)


def test_a_covered_card_keeps_its_full_quad_and_says_which_corners_are_hidden():
    rows = synthetic_catalog(4)
    lower = Instance(0, 0, True, "runes", "near", (0.0, 0.0), 0.0, pile=1, pile_index=0)
    upper = Instance(1, 1, True, "runes", "near", (0.0, 30.0), 0.0, pile=1, pile_index=1)
    down = Instance(2, None, False, "facedown", "near", (-150.0, 0.0), 0.0)
    board = Board(instances=[lower, upper, down], occluders=[Occluder("die", (0.0, 45.0), 16.0, 0.0, on=1, value=5)],
                  piles={1: "rune_column"})
    _, ids, ann = render(board, rows, load_fixture_image, _clean_shot(), np.random.default_rng(0))
    t = tile_targets(ann["cards"], ids.astype(np.int32), ann["window"], 1.0, (0, 0))
    by_box = sorted(t, key=lambda x: (x["category_id"], x["bbox"][1]))
    low, up, back = by_box
    assert [low["category_id"], up["category_id"], back["category_id"]] == [1, 1, 2]
    assert low["keypoints"][2::3] == [2, 2, 1, 1]      # the top shows, the card above hides the bottom corners
    assert up["keypoints"][2::3] == [2, 2, 2, 2]       # the die sits in the middle, not on a corner
    assert back["keypoints"][2::3] == [2, 2, 2, 2]
    assert low["bbox"][3] == pytest.approx(150, abs=2)  # the box is the whole card, not the visible strip
    assert "printing_id" not in json.dumps(t)


def test_export_writes_roboflow_coco_keypoints(tmp_path):
    run = tmp_path / "run"
    assert synth(["--fixtures", "20", "--boards", "2", "--clip", "2", "--frames-per-board", "2", "--sizes", "960x540",
                  "--layouts", "full", "--out", str(run)]) == 0
    stats = export_run(run, tmp_path / "tiles", val_every=2, target=(60.0, 60.0))
    assert stats["train"]["tiles"] > 0 and stats["valid"]["tiles"] > 0 and stats["train"]["cards"] > 10
    d = json.loads((tmp_path / "tiles" / "train" / "_annotations.coco.json").read_text())
    assert [c["name"] for c in d["categories"]] == ["card", "card_back"]
    assert d["categories"][0]["keypoints"] == ["top_left", "top_right", "bottom_right", "bottom_left"]
    for im in d["images"]:
        assert (tmp_path / "tiles" / "train" / im["file_name"]).exists()
    for t in d["annotations"]:
        k = np.array(t["keypoints"]).reshape(4, 3)
        on = k[:, 2] > 0
        assert ((k[on, :2] >= 0) & (k[on, :2] <= 576)).all() and t["num_keypoints"] == on.sum()
        x, y, w, h = t["bbox"]
        assert 0 <= x and 0 <= y and x + w <= 576.01 and y + h <= 576.01
    assert "printing_id" not in json.dumps(d)          # identities never reach the detector's data


def _det(quad, score=0.9, cls="card"):
    q = np.asarray(quad, np.float64)
    return {"cls": cls, "score": score, "box": [*q.min(axis=0), *q.max(axis=0)], "quad": q,
            "found": [0.9] * 4, "visible": [0.9] * 4}


def test_tiles_merge_into_one_list_without_losing_stacked_cards():
    a = _card(518, 100, 0)          # whole in both tiles: they overlap from x=460 to x=576
    lower, upper = _card(100, 100, 0), _card(100, 120, 0)  # a rune column in the left tile
    cut = _card(560, 300, 0)
    cut[:, 0] = np.minimum(cut[:, 0], 576)  # cut off by the left tile's right edge
    left = [_det(a), _det(lower), _det(upper, 0.8), _det(cut, 0.7)]
    right = [_det(a - [460, 0], 0.85)]  # the same card, seen by the right tile
    out = merge_tiles([left, right], [(0, 0), (460, 0)], (1036, 576), 1.0, (0, 0))
    assert len(out) == 3                                  # one copy of `a`, both stacked cards, no cut-off copy
    assert sorted(round(float(np.mean(np.reshape(o["quad"], (4, 2))[:, 1]))) for o in out) == [100, 100, 120]


def test_matching_is_one_to_one_and_recall_is_split_by_visibility():
    lower, upper = _card(100, 100, 0), _card(100, 120, 0)
    gts = [{"cls": "card", "quad": canonical_quad(lower), "visible": 0.23, "truncated": False, "zone": "runes", "long": 88.0},
           {"cls": "card", "quad": canonical_quad(upper), "visible": 1.0, "truncated": False, "zone": "runes", "long": 88.0}]
    dets = [_det(canonical_quad(upper) + 1, 0.9), _det(canonical_quad(upper), 0.4)]
    assert quad_iou(lower, upper) > 0.5                   # at IoU 0.5 the spare copy would "find" the card below
    assert [j for _, j, _ in match(dets, gts)] == [1]     # at 0.75 it does not, and one card takes one detection
    rows = {(r["threshold"], r["group"]): r for r in score_frames([(dets, gts)], thresholds=(0.5,))}
    assert rows[(0.5, "whole")]["recall"] == 1.0 and rows[(0.5, "strip")]["recall"] == 0.0
    assert rows[(0.5, "all")]["precision"] == 1.0


def test_real_recall_uses_the_mat_detector_boxes():
    crops = [{"file": "a.png", "frame": "/v/f1.jpg", "centre": [100, 100], "long_px": 88, "short_px": 63, "angle_deg": 90.0},
             {"file": "b.png", "frame": "/v/f1.jpg", "centre": [300, 100], "long_px": 88, "short_px": 63, "angle_deg": 0.0},
             {"file": "c.png", "frame": "/v/f2.jpg", "centre": [100, 100], "long_px": 88, "short_px": 63, "angle_deg": 90.0}]
    assert quad_iou(box_quad([100, 100], 88, 63, 90.0), _card(100, 100, 0)) == pytest.approx(1.0)   # long side vertical
    assert quad_iou(box_quad([300, 100], 88, 63, 0.0), _card(300, 100, 90)) == pytest.approx(1.0)   # lying sideways
    dets = {"f1.jpg": [{"cls": "card", "quad": _card(100, 102, 0).ravel().tolist()},
                       {"cls": "card", "quad": _card(300, 100, 90).ravel().tolist()}]}  # f2 was not run
    rows = {r["group"]: r for r in score_real(dets, crops, {"a.png": "OGN-001", "b.png": "back", "c.png": "OGN-002"})}
    assert rows["all"]["n"] == 2 and rows["all"]["recall"] == 1.0 and rows["all"]["class_right"] == 0.5
    assert rows["card_back"]["n"] == 1


def test_detector_weights_are_found_from_the_working_directory(tmp_path, monkeypatch):
    from rifteye_ml.detect.model import weights_path

    (tmp_path / "w.pth").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert weights_path("w.pth") == (tmp_path / "w.pth").resolve()  # not RF-DETR's own model cache
    with pytest.raises(FileNotFoundError):
        weights_path("missing.pth")
