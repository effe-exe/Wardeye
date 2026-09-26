import csv

from rifteye_ml import spike


def test_demo_spike_end_to_end(tmp_path):
    out = tmp_path / "m0-demo.csv"
    assert spike.main(["demo", "--cards", "24", "--heights", "40,160", "--bitrates", "1500", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 4  # 2 heights × 1 bitrate × 2 rotation modes × 1 encoder
    assert set(rows[0]) == set(spike.FIELDS)
    top1 = {(r["card_h"], r["rotation"]): float(r["top1_card"]) for r in rows}
    assert top1[("160", "oracle")] >= top1[("40", "oracle")]
    assert top1[("160", "oracle")] >= 0.9  # large, upright, fake cards are easy


def test_query_sample_keeps_the_full_gallery(tmp_path):
    out = tmp_path / "m0-demo.csv"
    args = ["demo", "--cards", "24", "--queries", "10", "--heights", "160", "--bitrates", "3000", "--out", str(out)]
    assert spike.main(args) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [int(r["n"]) for r in rows] == [10, 10]  # 10 queries, searched against all 24 cards
    assert float(rows[1]["top1_card"]) >= 0.9


def test_camera_realism_is_labeled_and_harder(tmp_path):
    out = tmp_path / "m0-demo.csv"
    for level in ("codec", "camera"):
        args = ["demo", "--cards", "24", "--heights", "60", "--bitrates", "3000", "--realism", level, "--out", str(out)]
        assert spike.main(args) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["realism"] for r in rows] == ["codec", "codec", "camera", "camera"]
    top1 = {(r["realism"], r["rotation"]): float(r["top1_printing"]) for r in rows}
    assert top1[("camera", "oracle")] <= top1[("codec", "oracle")]


def test_write_csv_refuses_a_file_with_other_columns(tmp_path):
    import pytest

    out = tmp_path / "old.csv"
    out.write_text("mode,encoder\nsynthetic,colorgrid16\n")
    with pytest.raises(SystemExit):
        spike.write_csv([], out)


def test_query_set_searches_other_printings_against_the_gallery():
    from rifteye_ml.degrade import StreamSettings
    from rifteye_ml.encoders import get_encoder
    from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog

    rows = synthetic_catalog(12, seed=0)
    images = [load_fixture_image(r) for r in rows]
    # "Other printings": the same cards, slightly recoloured, pointing at gallery rows 0..5.
    other = [im.point(lambda v: min(255, v + 12)) for im in images[:6]]
    base = StreamSettings(frame_w=640, frame_h=360, frames_per_board=4, seed=1)
    results = spike.run_synthetic(rows, load_fixture_image, [get_encoder("colorgrid")], [120], [3000], base,
                                  query_set=("xx", other, list(range(6))))
    assert {r["queries"] for r in results} == {"xx"} and {r["n"] for r in results} == {6}
    assert max(r["top1_card"] for r in results) >= 0.8


def test_gallery_scales_are_labeled(tmp_path):
    out = tmp_path / "m0-demo.csv"
    args = ["demo", "--cards", "12", "--heights", "60", "--bitrates", "3000", "--gallery-scales", "48,96",
            "--out", str(out)]
    assert spike.main(args) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert {r["gallery"] for r in rows} == {"px:48,96"}


def test_gallery_cache_reuses_levels(tmp_path):
    from rifteye_ml.encoders import get_encoder
    from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog

    rows = synthetic_catalog(6, seed=0)
    images = [load_fixture_image(r) for r in rows]
    enc = get_encoder("colorgrid")
    first = spike._gallery(enc, images, [40, 80], cache=tmp_path, key="k")
    assert len(list(tmp_path.glob("*.npy"))) == 2
    again = spike._gallery(enc, images, [40, 80], cache=tmp_path, key="k")
    assert all((first.levels[x] == again.levels[x]).all() for x in (40, 80))
    spike._gallery(enc, images, [], cache=tmp_path, key="k")
    assert len(list(tmp_path.glob("*.npy"))) == 3  # plus the sharp gallery


def test_strip_views_are_scored_and_labeled(tmp_path):
    out = tmp_path / "m0-demo.csv"
    args = ["demo", "--cards", "12", "--heights", "120", "--bitrates", "3000", "--strips", "top:0.4,left:0.5",
            "--out", str(out)]
    assert spike.main(args) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["view"] for r in rows] == ["full", "full", "top:0.4", "left:0.5"]
    assert {r["gallery"] for r in rows[2:]} == {"px:120"} and {r["rotation"] for r in rows[2:]} == {"oracle"}
