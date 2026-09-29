import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIDE = 48


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random RF-DETR keypoint net of the detector's kind (card and card_back, four corners each), small
    enough to export in seconds: 48 px tiles, two DINOv2 layers, a narrow decoder. Nothing is downloaded."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    import rfdetr.models.backbone.dinov2 as dinov2
    from rfdetr.config import RFDETRKeypointPreviewConfig
    from rfdetr.models import MODEL_DEFAULTS, PostProcess, build_model_from_config

    from rifteye_ml.detect import onnx as dx
    from rifteye_ml.detect.model import Detector

    full = dinov2.get_config
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dinov2, "get_config", lambda size, registers: {**full(size, registers), "num_hidden_layers": 2})
        torch.manual_seed(0)
        mc = RFDETRKeypointPreviewConfig(num_classes=1, num_keypoints_per_class=[4, 4], resolution=SIDE,
                                         positional_encoding_size=SIDE // 12, out_feature_indexes=[1, 2], dec_layers=1,
                                         hidden_dim=32, sa_nheads=2, ca_nheads=2, num_queries=6, num_select=6,
                                         pretrain_weights=None, device="cpu")
        net = build_model_from_config(mc, defaults=dataclasses.replace(MODEL_DEFAULTS, dim_feedforward=32)).eval()
    det = Detector.__new__(Detector)  # what Detector.__init__ sets up from a checkpoint
    det.torch, det.device, det.names, det.net = torch, torch.device("cpu"), ["card", "card_back"], net
    det.rf = SimpleNamespace(model_config=mc, means=MEANS, stds=STDS)
    det.post = PostProcess(num_select=6, num_keypoints_per_class=[4, 4])
    out = tmp_path_factory.mktemp("onnx")
    fp32 = dx.export(det, out / "tiny.onnx", size=SIDE)
    fp16 = dx.to_fp16(fp32, out / "tiny.fp16.onnx")
    yield det, fp32, fp16
    fp32.unlink()  # 30 MB together; pytest keeps its last few temporary folders
    fp16.unlink()


def _pixels(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, SIDE, SIDE, 3), dtype=np.uint8)


def _tiles(pixels: np.ndarray) -> np.ndarray:
    """What a browser sends: RGB / 255, NCHW."""
    return (np.moveaxis(pixels, -1, 1) / 255).astype(np.float32)


def test_the_graph_takes_tiles_in_0_1_normalises_them_itself_and_any_batch(tiny):
    import torch
    import torchvision.transforms.functional as F

    from rifteye_ml.detect import onnx as dx

    det, fp32, _ = tiny
    s = dx.session(fp32)
    assert [(i.name, i.shape) for i in s.get_inputs()] == [("tiles", ["batch", 3, SIDE, SIDE])]
    assert [o.name for o in s.get_outputs()] == ["pred_logits", "pred_boxes", "pred_keypoints"]
    for batch in (1, 3):
        tiles = _tiles(_pixels(batch))
        with torch.no_grad():
            want = det.net(F.normalize(torch.from_numpy(tiles), MEANS, STDS))
        for name, got in zip(dx.OUTPUTS, s.run(None, {"tiles": tiles})):
            assert got.shape == tuple(want[name].shape) and got.shape[0] == batch
            np.testing.assert_allclose(got, want[name].numpy(), atol=1e-4)


def test_an_onnx_file_runs_through_the_detectors_own_postprocess(tiny):
    from rifteye_ml.detect import onnx as dx

    det, fp32, _ = tiny
    tiles = [Image.fromarray(p) for p in _pixels(2, seed=1)]
    want = det.detect_tiles(tiles, threshold=0.0)
    net = det.net
    try:
        det.net = dx.OnnxNet(fp32, MEANS, STDS)
        got = det.detect_tiles(tiles, threshold=0.0)
    finally:
        det.net = net
    assert [len(t) for t in got] == [len(t) for t in want] and sum(map(len, want)) > 0
    for a, b in zip(sum(want, []), sum(got, [])):
        assert a["cls"] == b["cls"] and a["score"] == pytest.approx(b["score"], abs=1e-4)
        np.testing.assert_allclose(a["quad"], b["quad"], atol=1e-2)
        np.testing.assert_allclose(a["visible"], b["visible"], atol=1e-4)


def test_the_float16_copy_has_float16_weights_and_float32_inputs_and_outputs(tiny):
    import onnx

    from rifteye_ml.detect import onnx as dx

    _, fp32, fp16 = tiny
    weights = onnx.load(str(fp16)).graph.initializer
    assert any(w.data_type == onnx.TensorProto.FLOAT16 for w in weights)
    assert not any(w.data_type == onnx.TensorProto.FLOAT and len(w.dims) > 1 for w in weights)
    assert fp16.stat().st_size < 0.6 * fp32.stat().st_size
    s = dx.session(fp16)
    assert [i.type for i in s.get_inputs()] == ["tensor(float)"]
    assert [o.type for o in s.get_outputs()] == ["tensor(float)"] * 3
    tiles = _tiles(_pixels(2))
    for half, full in zip(s.run(None, {"tiles": tiles}), dx.session(fp32).run(None, {"tiles": tiles})):
        assert half.shape == full.shape and np.isfinite(half).all()
