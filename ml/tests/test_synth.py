import collections
import json

import numpy as np
import pytest

from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog
from rifteye_ml.synth import Board, Instance, Occluder, Shot, render, sample_board
from rifteye_ml.synth.__main__ import main
from rifteye_ml.synth.compose import OCCLUDER_ID, Camera, card_quad_mm
from rifteye_ml.synth.layout import CARD_H_MM


def _clean_shot(**kw) -> Shot:
    base = dict(frame_w=640, frame_h=480, layout="full", window=(0, 0, 640, 480), card_px=150.0, turn=0, yaw_deg=0.0,
                keystone=0.0, k1=0.0, centre_mm=(0.0, 0.0), defocus_px=0.0, wb_jitter=0.0, exposure_jitter=0.0,
                gamma_jitter=0.0, light_jitter=0.0, featured=False)
    base.update(kw)
    return Shot(**base)


def test_boards_follow_the_tournament_table():
    rows = synthetic_catalog(30)
    for seed in range(12):
        b = sample_board(rows, np.random.default_rng(seed))
        zones = collections.Counter((i.controller, i.zone) for i in b.instances)
        assert zones[("near", "legend")] == 1 and zones[("far", "legend")] == 1
        assert zones[("near", "runes")] <= 12 and zones[("far", "runes")] <= 12
        for i in b.instances:
            assert abs(i.centre[0]) < 500 and abs(i.centre[1]) < 450
            assert i.pile is None or i.pile in b.piles
            if not i.face_up:
                assert i.row is None  # a face-down card has no identity to leak
            # ready cards face their controller; exhausted ones are a quarter turn off that
            facing = 0.0 if i.controller == "near" else 180.0
            off = (i.angle - facing - (-90.0 if i.exhausted else 0.0) + 180) % 360 - 180
            assert abs(off) <= 12.0


def test_camera_maps_there_and_back():
    cam = Camera(_clean_shot(turn=1, yaw_deg=2.0, keystone=0.1, k1=-0.05, centre_mm=(10.0, -20.0)), 800, 600)
    pts = np.random.default_rng(0).uniform([-200, -200], [200, 200], size=(50, 2))
    np.testing.assert_allclose(cam.to_table(cam.to_window(pts)), pts, atol=1e-3)


def test_a_covered_card_shows_what_the_geometry_says():
    rows = synthetic_catalog(4)
    lower = Instance(0, 0, True, "runes", "near", (0.0, 0.0), 0.0, pile=1, pile_index=0)
    upper = Instance(1, 1, True, "runes", "near", (0.0, 30.0), 0.0, pile=1, pile_index=1)
    down = Instance(2, None, False, "facedown", "near", (-150.0, 0.0), 0.0)
    board = Board(instances=[lower, upper, down], occluders=[Occluder("die", (0.0, 45.0), 16.0, 0.0, on=1, value=5)],
                  piles={1: "rune_column"})
    frame, ids, ann = render(board, rows, load_fixture_image, _clean_shot(), np.random.default_rng(0))
    cards = {c["id"]: c for c in ann["cards"]}
    assert cards[0]["visible"] == pytest.approx(30 / CARD_H_MM, abs=0.04)   # the top 30 mm show
    assert cards[1]["visible"] == pytest.approx(1 - (16 * 16) / (63 * 88), abs=0.03)  # all but the die
    assert cards[2]["kind"] == "card_back" and "printing_id" not in cards[2] and "printing_id" in cards[0]
    assert (ids == OCCLUDER_ID).sum() > 0
    # every pixel of a card lies inside its quad, which starts at the card's printed top-left
    q = np.array(cards[0]["quad"]).reshape(4, 2)
    ys, xs = np.nonzero(ids == 1)
    assert q[:, 0].min() - 2 <= xs.min() and xs.max() <= q[:, 0].max() + 2 and q[:, 1].min() - 2 <= ys.min()
    assert q[0, 0] < q[1, 0] and q[0, 1] < q[3, 1]  # upright: TL left of TR, above BL
    ppm = 150.0 / CARD_H_MM
    np.testing.assert_allclose(q, (card_quad_mm(lower) * ppm) + [320, 240], atol=0.5)


def test_the_generator_is_reproducible(tmp_path):
    args = ["--fixtures", "30", "--boards", "3", "--clip", "2", "--frames-per-board", "4", "--sizes", "640x360",
            "--previews", "1", "--seed", "5"]
    assert main(args + ["--out", str(tmp_path / "a")]) == 0
    assert main(args + ["--out", str(tmp_path / "b")]) == 0
    la = (tmp_path / "a" / "annotations.jsonl").read_text().splitlines()
    assert len(la) == 3 and la == (tmp_path / "b" / "annotations.jsonl").read_text().splitlines()
    for name in ("000000.png", "000002.png"):  # the codec pass is single-threaded, so bit-exact
        assert (tmp_path / "a" / "frames" / name).read_bytes() == (tmp_path / "b" / "frames" / name).read_bytes()
    first = json.loads(la[0])
    assert first["image"] == "frames/000000.png" and first["cards"] and all(0 <= c["visible"] <= 1 for c in first["cards"])
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["boards"] == 3 and manifest["codec"] is True
    assert (tmp_path / "a" / "previews" / "000000.jpg").exists()


def test_the_realism_check_names_fake_cards(tmp_path):
    from rifteye_ml.synth.check import main as check

    run = tmp_path / "run"
    assert main(["--fixtures", "20", "--boards", "2", "--clip", "2", "--frames-per-board", "3", "--sizes", "960x540",
                 "--layouts", "full", "--view-mm", "600,600", "--out", str(run)]) == 0
    out = tmp_path / "check.csv"
    assert check(["--run", str(run), "--fixtures", "20", "--encoder", "colorgrid/trim0.03", "--gallery-scales", "60,80",
                  "--out", str(out)]) == 0
    rows = {r["group"]: r for r in csv_rows(out)}
    assert int(rows["all"]["n"]) > 5 and float(rows["all"]["top1_card"]) >= 0.8  # clean fake cards are easy


def csv_rows(path):
    import csv

    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
