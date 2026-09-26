import csv

from rifteye_ml import catalog as cat
from rifteye_ml import label
from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog


def test_propose_ranks_candidates_and_renders_sheets(tmp_path):
    rows = synthetic_catalog(10, seed=0)
    cat.write_catalog(rows, tmp_path / "catalog.jsonl")
    for r in rows:
        dest = cat.cache_path(tmp_path / "art", r["image_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        load_fixture_image(r).save(dest)
    crops = tmp_path / "crops"
    crops.mkdir()
    for i in (1, 4, 7):
        im = load_fixture_image(rows[i]).resize((72, 100))
        (im.rotate(90, expand=True) if i == 4 else im).save(crops / f"c{i}.png")
    (crops / "c1_mask.png").write_bytes((crops / "c1.png").read_bytes())  # masks are skipped

    out = tmp_path / "proposals.csv"
    args = ["propose", "--catalog", str(tmp_path / "catalog.jsonl"), "--cache", str(tmp_path / "art"),
            "--crops", str(crops), "--k", "3", "--out", str(out), "--sheets", str(tmp_path / "sheets")]
    assert label.main(args) == 0
    props = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(props) == 3 * 3
    top1 = {p["file"]: p["printing_id"] for p in props if p["rank"] == "1"}
    assert top1 == {"c1.png": "FAKE-001", "c4.png": "FAKE-004", "c7.png": "FAKE-007"}
    assert sorted(p.name for p in (tmp_path / "sheets").iterdir()) == ["c1.jpg", "c4.jpg", "c7.jpg"]
