import contextlib
import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A random-init tiny ViT, packed like the real weights and exported by the CLI (softmax kept in float32 in the
    half copy): (net, weights, files, its output)."""
    for mod in ("torch", "timm", "onnx", "onnxruntime", "onnxscript", "onnxconverter_common"):
        pytest.importorskip(mod)
    import torch

    from rifteye_ml.embed.__main__ import main
    from rifteye_ml.embed.model import Net, pack

    tmp = tmp_path_factory.mktemp("onnx")
    torch.manual_seed(0)
    net = Net("test_vit", img_size=32, dim=16, pretrained=False, drop_path=0.0).eval()
    torch.save({"net": net.state_dict(), "config": net.config, "meta": {}}, tmp / "final.pt")
    pack(tmp / "final.pt", tmp / "tiny.pth")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["onnx", str(tmp / "tiny.pth"), str(tmp / "out"), "--keep-fp32", "Softmax"]) == 0
    return net, tmp / "tiny.pth", (tmp / "out" / "tiny.onnx", tmp / "out" / "tiny.fp16.onnx"), out.getvalue()


def _cos(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_the_graph_takes_crops_and_gives_embeddings_for_any_batch(exported):
    import onnx
    import torch

    from rifteye_ml.embed.onnx import Onnx, random_images

    net, _, (fp32, fp16), _ = exported
    for path, dtype in ((fp32, onnx.TensorProto.FLOAT), (fp16, onnx.TensorProto.FLOAT16)):
        m = onnx.load(path)
        (i,), (o,) = m.graph.input, m.graph.output
        assert (i.name, o.name) == ("crops", "embedding")
        assert [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] == ["batch", 3, 32, 32]
        assert [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim] == ["batch", 16]
        assert i.type.tensor_type.elem_type == o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT  # both files
        assert {t.data_type for t in m.graph.initializer if t.dims and t.data_type in (1, 10)} == {dtype}  # the weights
        assert max(d.version for d in m.opset_import if d.domain in ("", "ai.onnx")) <= 20
    assert fp16.stat().st_size < 0.7 * fp32.stat().st_size  # about half at full size: the graph is a share of the tiny one
    x = random_images(9, 32)
    with torch.inference_mode():
        ref = net(torch.from_numpy(x)).numpy()
    for path, floor in ((fp32, 0.99999), (fp16, 0.999)):
        enc = Onnx(path)
        for n in (1, 2, 9):  # the batch axis is dynamic, including a single crop
            got = enc.run(x[:n])
            assert got.shape == (n, 16) and got.dtype == np.float32
            assert _cos(got, ref[:n]).min() >= floor


def _softmax_fed_by_cast_to_float32(path):
    import onnx

    nodes = onnx.load(path).graph.node
    made = {out: n for n in nodes for out in n.output}
    feeds = [made.get(n.input[0]) for n in nodes if n.op_type == "Softmax"]
    assert feeds
    return all(c is not None and c.op_type == "Cast" and any(a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in c.attribute)
               for c in feeds)


def test_operators_named_to_keep_stay_float32_in_the_half_copy(exported):
    _, _, (fp32, fp16), _ = exported
    assert _softmax_fed_by_cast_to_float32(fp16) and not _softmax_fed_by_cast_to_float32(fp32)


def test_the_cli_prints_how_close_each_file_is_to_torch(exported):
    *_, printed = exported
    lines = [ln for ln in printed.splitlines() if "against torch" in ln]
    assert [ln.split(":")[0].rsplit("/", 1)[-1] for ln in lines] == ["tiny.onnx", "tiny.fp16.onnx"]
    assert all("lowest cosine" in ln for ln in lines)


def test_the_onnx_encoder_reads_crops_as_the_embedder_does(exported):
    from rifteye_ml.embed.model import FineTuned
    from rifteye_ml.embed.onnx import Onnx
    from rifteye_ml.encoders import get_encoder

    _, weights, (fp32, fp16), _ = exported
    rng = np.random.default_rng(3)
    crops = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((70, 50), (48, 66), (31, 20))]
    crops.append(crops[0].rotate(90, expand=True))  # a turned crop, as the live search makes them
    ref = get_encoder(f"embedder:{weights}")
    assert isinstance(ref, FineTuned)
    names = set()
    for path, floor in ((fp32, 0.99999), (fp16, 0.999)):
        enc = get_encoder(f"onnx:{path}")
        assert isinstance(enc, Onnx) and enc.dim == 16 and enc.img_size == 32
        rows = enc.embed(crops)
        assert rows.shape == (4, 16) and rows.dtype == np.float32 and np.allclose(np.linalg.norm(rows, axis=1), 1, atol=1e-5)
        assert _cos(rows, ref.embed(crops)).min() >= floor
        assert enc.embed([]).shape == (0, 16)
        enc.batch = 3  # a batch of the crops in two runs gives the same rows
        assert np.allclose(enc.embed(crops), rows, atol=1e-3)
        assert len(enc.fingerprint()) == 64
        names.add(enc.name)
    assert len(names) == 2 and all(n.startswith("onnx:tiny") for n in names) and ref.name not in names  # caches key on it
    with pytest.raises(ValueError):
        get_encoder("onnx:")


def test_the_live_runner_calibrates_the_export_like_the_embedder():
    from rifteye_ml.live.__main__ import EMBEDDER_T, temperature_for

    assert temperature_for("onnx:/x/embedder-v1.fp16.onnx", object(), None) == EMBEDDER_T
    assert temperature_for("onnx:/x/embedder-v1.onnx", object(), 0.02) == 0.02
    assert temperature_for("colorgrid", object(), None) is None
