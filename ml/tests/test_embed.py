import csv
import json

import numpy as np
import pytest
from PIL import Image

from rifteye_ml import catalog as cat
from rifteye_ml.embed.bank import TEXT_BOX, Bank, donor_for, generate, plan, plan_eval, swap_text, text_groups
from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog


def _catalogue(tmp_path, n=12):
    rows = synthetic_catalog(n, seed=3)
    for i, r in enumerate(rows):
        r["set_code"] = "FAK" if i < n * 2 // 3 else "NEW"
        r["card_id"] = f"card-{i // 2 if i < 4 else i}"  # the first four are two cards with two printings each
        r["type"] = "Spell" if i % 3 == 0 else "Unit"
    rows[-1].update(orientation="landscape", type="Battlefield")
    cat.write_catalog(rows, tmp_path / "catalog.jsonl")
    for r in rows:
        dest = cat.cache_path(tmp_path / "art", r["image_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        load_fixture_image(r).save(dest)
    return rows


def test_text_swap_takes_the_donors_box_and_only_between_standard_frames():
    a, b = Image.new("RGB", (63, 88), (200, 0, 0)), Image.new("RGB", (126, 176), (0, 0, 200))
    out = np.asarray(swap_text(a, b))
    x0, y0, x1, y1 = (round(f * s) for f, s in zip(TEXT_BOX, (63, 88, 63, 88)))
    assert (out[y0 + 1:y1 - 1, x0 + 1:x1 - 1] == (0, 0, 200)).all()   # the rules text is the donor's
    assert (out[:y0 - 1] == (200, 0, 0)).all()                          # the art is untouched
    rows = [{"type": "Unit", "set_code": "A"}, {"type": "Unit", "set_code": "A"}, {"type": "Legend", "set_code": "A"},
            {"type": "Unit", "set_code": "B"}, {"type": "Unit", "set_code": "A", "variant": "alt_art"},
            {"type": "Unit", "set_code": "A", "orientation": "landscape"}]
    assert text_groups(rows) == {("Unit", "A"): [0, 1], ("Unit", "B"): [3]}
    rng = np.random.default_rng(0)
    assert {donor_for(1, [0, 1, 2, 3], rng) for _ in range(200)} == {0, 2, 3}


def test_training_tasks_vary_the_stream_and_eval_tasks_follow_m0():
    tasks = plan(range(1, 9), [40, 120])
    assert len(tasks) == 16 and len({t.name for t in tasks}) == 16 and len({t.seed for t in tasks}) == 16
    assert {t.frame for t in tasks} == {(1920, 1080), (1280, 720)} and len({t.bitrate for t in tasks}) > 4
    weak = next(t for t in tasks if t.strength == 0.6).settings()
    assert weak.tilt_deg == pytest.approx(9.0) and weak.foil_prob == pytest.approx(0.12)
    ev = plan_eval([40, 60])
    assert [(t.seed, t.bitrate, t.realism, t.text_prob) for t in ev] == [(0, 4000, "camera", 0.0)] * 2


def test_the_bank_holds_every_printing_per_task_and_resumes(tmp_path):
    rows = _catalogue(tmp_path)
    tasks = plan([1], [40, 64], frame=(640, 360))
    assert generate(rows, str(tmp_path / "art"), tasks, tmp_path / "bank", workers=2, log=lambda m: None) == 2
    assert generate(rows, str(tmp_path / "art"), tasks, tmp_path / "bank", log=lambda m: None) == 0  # already there
    bank = Bank([tmp_path / "bank"])
    assert bank.printings == [r["printing_id"] for r in rows]
    assert len(bank) == 2 * len(rows) and sorted(set(bank.rows.tolist())) == list(range(len(rows)))
    for i in range(len(bank)):
        h, w, _ = bank.crop(i).shape
        assert abs(max(h, w) - bank.heights[i]) <= 0.3 * bank.heights[i]
        if rows[int(bank.rows[i])]["orientation"] == "portrait":
            assert h > w  # upright
    with pytest.raises(SystemExit):  # a bank belongs to one catalogue
        generate(rows[:5], str(tmp_path / "art"), tasks, tmp_path / "bank", log=lambda m: None)


def test_arcface_prefers_the_nearest_subcentre_and_the_margin_costs():
    torch = pytest.importorskip("torch")
    from rifteye_ml.embed.model import SubCenterArcFace

    head = SubCenterArcFace(4, classes=2, k=2, s=10.0, m=0.3)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]))
    emb = torch.tensor([[0.0, 1.0, 0, 0]])  # class 0's second centre
    loss, cos = head(emb, torch.tensor([0]))
    assert cos.tolist()[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    easy, _ = head(emb, torch.tensor([0]), margin=0.0)
    wrong, _ = head(emb, torch.tensor([1]))
    assert easy < loss < wrong


def test_train_resume_pack_and_encode(tmp_path):
    pytest.importorskip("timm")
    import torch

    from rifteye_ml.embed.model import FineTuned, Samples, pack, train
    from rifteye_ml.encoders import get_encoder

    rows = _catalogue(tmp_path, n=8)
    generate(rows, str(tmp_path / "art"), plan([1], [48], frame=(480, 270)), tmp_path / "bank", log=lambda m: None)
    bank = Bank([tmp_path / "bank"])
    images = [load_fixture_image(r).resize((92, 128)) for r in rows]
    kw = dict(train_rows=range(6), batch_size=4, clean_per_printing=2, workers=0, backbone="test_vit", img_size=32,
              dim=16, pretrained=False, log=lambda m: None)
    final = train(bank, rows, images, tmp_path / "run", epochs=1, **kw)
    assert (tmp_path / "run" / "last.pt").exists()
    final = train(bank, rows, images, tmp_path / "run", epochs=2, amp=True, **kw)  # resumes at epoch 2, bf16 on CPU
    lines = (tmp_path / "run" / "metrics.csv").read_text().splitlines()
    assert [ln.split(",")[0] for ln in lines] == ["epoch", "1", "2"]
    ck = torch.load(final, weights_only=True)
    assert ck["meta"]["printings"] == 6 and ck["cards"] == sorted({r["card_id"] for r in rows[:6]})
    size = pack(final, tmp_path / "e.pth")
    assert size < final.stat().st_size
    enc = get_encoder(f"embedder:{tmp_path / 'e.pth'}")
    assert isinstance(enc, FineTuned) and enc.dim == 16 and enc.name.startswith("embedder:e-")
    q = enc.embed([images[0], images[1].rotate(90, expand=True)])
    assert q.shape == (2, 16) and np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
    # the same sample twice is the same image; the next epoch draws it again
    data = Samples(bank, np.arange(len(bank)), [0, 1], {r: r for r in range(6)}, images, rows, 32, seed=1)
    a, b, c = data.sample(3), data.sample(3), data.sample(3 + len(data))
    assert a[1] == b[1] == c[1] and np.array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_the_cli_runs_end_to_end(tmp_path):
    pytest.importorskip("timm")
    from rifteye_ml.embed.__main__ import main

    _catalogue(tmp_path, n=9)
    base = ["--catalog", str(tmp_path / "catalog.jsonl"), "--cache", str(tmp_path / "art")]
    assert main(["crops", *base, "--out", str(tmp_path / "bank"), "--seeds", "1-2", "--heights", "40,56",
                 "--frame", "480x270", "--workers", "2"]) == 0
    assert main(["crops", *base, "--out", str(tmp_path / "eval"), "--eval", "--heights", "40,60", "--frame", "480x270"]) == 0
    assert main(["train", *base, "--bank", str(tmp_path / "bank"), "--out", str(tmp_path / "run"), "--train-sets", "FAK",
                 "--epochs", "2", "--batch-size", "8", "--clean", "2", "--workers", "2", "--backbone", "test_vit",
                 "--img-size", "32", "--dim", "16", "--no-pretrained"]) == 0
    assert main(["pack", "--checkpoint", str(tmp_path / "run" / "final.pt"), "--out", str(tmp_path / "heldout.pth")]) == 0
    out = tmp_path / "scores.csv"
    assert main(["evaluate", *base, "--bank", str(tmp_path / "eval"), "--train-sets", "FAK", "--encoder", "colorgrid",
                 "--encoder", f"embedder:{tmp_path / 'heldout.pth'}", "--out", str(out)]) == 0
    got = list(csv.DictReader(open(out, encoding="utf-8")))
    keys = {(r["encoder"], r["trained_on"], r["split"], r["view"], r["card_h"]) for r in got}
    assert ("heldout", "FAK", "held-out sets", "full", "40") in keys and ("colorgrid", "none", "training sets", "top:0.4", "60") in keys
    assert all(0 <= float(r["top1_card"]) <= 1 for r in got)
    full = [r for r in got if r["view"] == "full" and r["encoder"] == "colorgrid"]
    assert sum(int(r["n"]) for r in full) == 9 * 2  # every printing at both heights
    assert json.loads((tmp_path / "bank" / "printings.json").read_text())[0] == "FAKE-000"
