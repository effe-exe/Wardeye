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


def test_adapter_scores_real_crops_with_unknown_orientation(tmp_path):
    pytest.importorskip("torch", reason="the optional torch extra; CI installs the CPU build")
    rows = synthetic_catalog(12, seed=1)
    for i, r in enumerate(rows):
        r["set_code"] = "FAK" if i < 8 else "NEW"
    cat.write_catalog(rows, tmp_path / "catalog.jsonl")
    crops = tmp_path / "crops"
    crops.mkdir()
    with open(tmp_path / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "track", "printing_id"])
        for i, r in enumerate(rows):
            dest = cat.cache_path(tmp_path / "art", r["image_url"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            im = load_fixture_image(r)
            im.save(dest)
            # A "real" crop: the card small and turned, as a stream shows it.
            small = im.resize((im.width // 6, im.height // 6)).rotate(90 * (i % 4), expand=True)
            small.save(crops / f"c{i}.png")
            w.writerow([f"c{i}.png", f"t{i // 2}", r["printing_id"]])
        w.writerow(["back.png", "t99", "back"])  # not a card in the gallery: skipped
    out = tmp_path / "adapter.csv"
    args = ["--catalog", str(tmp_path / "catalog.jsonl"), "--cache", str(tmp_path / "art"), "--encoder", "colorgrid",
            "--train-sets", "FAK", "--heights", "60,120", "--train-seeds", "101", "--eval-per-split", "4",
            "--epochs", "2", "--frame", "640x360", "--real-crops", str(crops), "--real-labels", str(tmp_path / "labels.csv"),
            "--out", str(out)]
    assert adapter.main(args) == 0
    real = {(r["head"], r["split"]): r for r in csv.DictReader(open(out, encoding="utf-8")) if r["card_h"] == "real"}
    assert set(real) == {(h, s) for h in ("frozen", "linear")
                         for s in ("real", "real, one per track", "real, training sets", "real, held-out sets")}
    assert real[("frozen", "real")]["n"] == "12" and real[("frozen", "real, one per track")]["n"] == "6"
    assert real[("frozen", "real, training sets")]["n"] == "8"
    # Clean, only shrunk and turned: the frozen colour grid finds nearly all of them in any orientation.
    assert float(real[("frozen", "real")]["top1_card"]) >= 0.8
