import csv

import pytest

from rifteye_ml import adapter
from rifteye_ml import catalog as cat
from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog


def test_adapter_trains_and_reports_both_splits(tmp_path):
    pytest.importorskip("torch", reason="the optional torch extra; CI installs the CPU build")
    rows = synthetic_catalog(16, seed=0)
    for i, r in enumerate(rows):
        r["set_code"] = "FAK" if i < 10 else "NEW"
    cat.write_catalog(rows, tmp_path / "catalog.jsonl")
    for r in rows:
        dest = cat.cache_path(tmp_path / "art", r["image_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        load_fixture_image(r).save(dest)
    out = tmp_path / "adapter.csv"
    args = ["--catalog", str(tmp_path / "catalog.jsonl"), "--cache", str(tmp_path / "art"), "--encoder", "colorgrid",
            "--train-sets", "FAK", "--heights", "60,120", "--train-seeds", "101", "--eval-per-split", "5",
            "--epochs", "3", "--frame", "640x360", "--out", str(out)]
    assert adapter.main(args) == 0
    got = list(csv.DictReader(open(out, encoding="utf-8")))
    assert {(r["head"], r["split"]) for r in got} == {(h, s) for h in ("frozen", "linear")
                                                      for s in ("held-out sets", "training sets")}
    assert {r["card_h"] for r in got} == {"60", "120"}
    assert all(0.0 <= float(r["top1_card"]) <= 1.0 and r["n"] == "5" for r in got)
