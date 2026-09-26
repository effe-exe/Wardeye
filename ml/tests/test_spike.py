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
