import hashlib
import json
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rifteye_ml import catalog as cat
from rifteye_ml import web_assets as wa

COLOURS = [(200, 40, 40), (40, 200, 60), (50, 60, 210), (220, 200, 30)]
IDS = ["TST-001", "TST-002*", "TST-003", "TST-004a"]


def _rows(n: int = 4) -> list[dict]:
    return [{"printing_id": IDS[i], "card_id": f"card-{i}", "name": f"Card {i}", "type": ["Unit", "Spell", "Legend", "Rune"][i],
             "rarity": "common", "image_url": f"https://example.test/art/{i}.png"} for i in range(n)]


def _tiny_model(path: Path, size: int = 8, dim: int = 4) -> None:
    """crops -> mean colour -> a fixed 3 x dim map: an embedder that tells the four colours apart."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    w = np.random.default_rng(0).normal(size=(3, dim)).astype(np.float32)
    nodes = [helper.make_node("ReduceMean", ["crops"], ["m"], axes=[2, 3], keepdims=0),
             helper.make_node("MatMul", ["m", "w"], ["embedding"])]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("crops", TensorProto.FLOAT, ["batch", 3, size, size])],
                              [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", dim])],
                              [numpy_helper.from_array(w, "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture()
def world(tmp_path):
    """A catalogue of four printings with their art cached, and a tiny embedder in a models folder."""
    rows = _rows()
    art = tmp_path / "art"
    for r, colour in zip(rows, COLOURS):
        dest = cat.cache_path(art, r["image_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 84), colour).save(dest)
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    _tiny_model(models / "tiny.onnx")
    (models / "tiny.fp16.onnx").write_bytes(b"not a real fp16 copy, only hashed")
    return {"rows": rows, "art": art, "catalog": catalog, "models": models, "out": tmp_path / "out"}


def _build(w, **kw):
    kw.setdefault("px", (100, 106))
    return wa.build(w["out"], w["models"], embedder="tiny", catalog=w["catalog"], art_cache=w["art"], threads=1, **kw)


def test_the_levels_are_the_live_runners_for_every_card_size_a_broadcast_may_have():
    assert wa.levels_for(155) == [120, 140, 160]  # la-rq
    assert wa.levels_for(140) == [110, 130, 140]  # plusrb
    assert wa.levels_for(131) == [100, 120, 130]  # shenyang
    assert wa.levels_for(105) == [80, 90, 100]  # 10.5 is a half: to the even 10, as Python rounds
    assert wa.levels_between(100, 200) == list(range(80, 210, 10))
    assert wa.levels_between(155, 155) == [120, 140, 160]


def test_the_formula_is_the_one_in_the_live_runner():
    import inspect

    from rifteye_ml.live import __main__ as live

    assert "int(round(px * f / 10) * 10) for f in (0.8, 0.9, 1.0)" in inspect.getsource(live.main)
    assert wa.FRACTIONS == (0.8, 0.9, 1.0)


def test_a_thumbnail_is_named_by_its_printing_id_where_a_file_name_can_hold_it():
    # the same cases as apps/extension/test/thumbs.test.ts
    assert wa.thumb_name("SFD-195a") == "SFD-195a"
    assert wa.thumb_name("OGN-299*") == "OGN-299_2a"
    assert wa.thumb_name("a_b") == "a_5fb"
    assert wa.thumb_name("é/") == "_c3_a9_2f"
    assert len({wa.thumb_name(p) for p in ("a*", "a_2a", "a")}) == 3


def test_build_writes_the_gallery_the_catalogue_and_the_thumbnails(world):
    from rifteye_ml.embed.onnx import Onnx
    from rifteye_ml.live.server import LiveServer
    from rifteye_ml.retrieval import at_long_side
    from rifteye_ml.spike import _cached_loader

    index = _build(world)
    out = world["out"]
    enc = Onnx(world["models"] / "tiny.onnx")
    raw = (world["models"] / "tiny.onnx").read_bytes()

    # the index says what the gallery is
    assert json.loads((out / "gallery" / "index.json").read_text()) == index
    assert index["format"] == 1 and index["dtype"] == "float16" and index["dim"] == 4
    assert index["encoder"] == enc.name and index["model"] == "tiny"
    assert index["sha256"] == hashlib.sha256(raw).hexdigest()
    assert index["fp16_sha256"] == hashlib.sha256((world["models"] / "tiny.fp16.onnx").read_bytes()).hexdigest()
    assert index["levels"] == [80, 90, 100, 110]
    assert index["rows"] == IDS  # the catalogue's order

    # each level is the runner's own gallery at that size, as float16
    load = _cached_loader(str(world["art"]), 512)
    art = [load(r) for r in world["rows"]]
    for lv in index["levels"]:
        blob = (out / "gallery" / f"L{lv}.bin").read_bytes()
        assert len(blob) == 4 * 4 * 2
        want = enc.embed([at_long_side(im, lv) for im in art])
        assert np.array_equal(np.frombuffer(blob, "<f2").reshape(4, 4), want.astype("<f2"))
        assert np.abs(np.frombuffer(blob, "<f2").astype(np.float32).reshape(4, 4) - want).max() < 1e-3
    assert len(list((out / "embed-cache").glob("*.npy"))) == 4  # gallery()'s own cache, in the output folder

    # the catalogue: what the tracker and the overlay read, in the gallery's order
    slim = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    assert [set(r) for r in slim] == [{"printing_id", "card_id", "name", "type"}] * 4
    assert [r["printing_id"] for r in slim] == IDS
    assert slim[2] == {"printing_id": "TST-003", "card_id": "card-2", "name": "Card 2", "type": "Legend"}

    # the thumbnails: what the runner's /art/<id>.jpg serves
    server = LiveServer(port=0, art=lambda pid: cat.cache_path(world["art"], next(r for r in world["rows"] if r["printing_id"] == pid)["image_url"]))
    for pid in IDS:
        jpeg = (out / "thumbs" / f"{wa.thumb_name(pid)}.jpg").read_bytes()
        assert jpeg == server.art_jpeg(pid)
        with Image.open(BytesIO(jpeg)) as im:
            assert im.format == "JPEG" and max(im.size) <= 360
    assert (out / "thumbs" / "TST-002_2a.jpg").exists()


def test_a_second_run_only_converts(world, monkeypatch, capsys):
    from rifteye_ml.embed.onnx import Onnx

    first = _build(world)
    capsys.readouterr()

    def never(self, images):
        raise AssertionError("the encoder ran again")

    monkeypatch.setattr(Onnx, "embed", never)
    thumb = world["out"] / "thumbs" / "TST-001.jpg"
    mtime = thumb.stat().st_mtime_ns
    assert _build(world) == first
    assert "thumbs/: 0 made, 4 kept" in capsys.readouterr().out
    assert thumb.stat().st_mtime_ns == mtime


def test_a_printing_whose_art_is_not_cached_is_left_out_and_nothing_is_fetched(world, monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError("something tried to fetch")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(cat, "download_images", no_network)
    cat.cache_path(world["art"], world["rows"][1]["image_url"]).unlink()
    index = _build(world)
    assert index["rows"] == ["TST-001", "TST-003", "TST-004a"]
    assert "1 of 4 printings have no art" in capsys.readouterr().out
    assert [r["printing_id"] for r in json.loads((world["out"] / "catalog.json").read_text())] == index["rows"]


def test_without_art_or_without_the_float32_model_it_says_so(world):
    for r in world["rows"]:
        cat.cache_path(world["art"], r["image_url"]).unlink()
    with pytest.raises(SystemExit, match="no card art cached"):
        _build(world)
    (world["models"] / "tiny.onnx").unlink()
    with pytest.raises(SystemExit, match="float32 model"):
        _build(world)


def test_main_takes_its_options_from_the_command_line(world, capsys):
    args = [str(world["out"]), "--models", str(world["models"]), "--embedder", "tiny", "--catalog", str(world["catalog"]),
            "--cache", str(world["art"]), "--px-min", "100", "--px-max", "100", "--threads", "1", "--no-thumbs"]
    assert wa.main(args) == 0
    assert "done in" in capsys.readouterr().out
    assert json.loads((world["out"] / "gallery" / "index.json").read_text())["levels"] == [80, 90, 100]
    assert not (world["out"] / "thumbs").exists()
    with pytest.raises(SystemExit):
        wa.main([str(world["out"]), "--px-min", "200", "--px-max", "100"])


def test_there_is_no_default_place_to_read_from(world, monkeypatch, tmp_path):
    monkeypatch.delenv("RIFTEYE_DATA", raising=False)
    assert wa.data_dir() is None
    with pytest.raises(SystemExit, match="--models: name it, or set RIFTEYE_DATA"):
        wa.main([str(world["out"]), "--catalog", str(world["catalog"]), "--cache", str(world["art"])])
    with pytest.raises(SystemExit, match="--cache: name it"):
        wa.build(world["out"], world["models"], embedder="tiny", catalog=world["catalog"])
    with pytest.raises(SystemExit, match="--catalog: name it"):
        wa.find_catalogue(None)
    # named through $RIFTEYE_DATA, the defaults are found under it
    data = tmp_path / "data"
    (data / "catalog").mkdir(parents=True)
    (data / "catalog" / "catalog.jsonl").write_text("")
    monkeypatch.setenv("RIFTEYE_DATA", str(data))
    assert wa.data_dir() == data
    assert wa.find_catalogue(None) == data / "catalog" / "catalog.jsonl"
    (data / "catalog" / "catalog-plus.jsonl").write_text("")
    assert wa.find_catalogue(None) == data / "catalog" / "catalog-plus.jsonl"
