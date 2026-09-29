# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The embedder as ONNX, to run in a browser (M2): `export` writes the float32 graph and a float16 copy, `Onnx`
runs a file as the encoder `onnx:<file>` with onnxruntime (no torch needed to run it).

    python -m rifteye_ml.embed onnx embedder-v1.pth out/     # out/embedder-v1.onnx and out/embedder-v1.fp16.onnx
    python -m rifteye_ml.spike real --encoder onnx:out/embedder-v1.fp16.onnx ...

The graph is `Net` as it is: "crops", float32 [batch, 3, 224, 224] holding RGB values 0..255 (the crop letterboxed
as `FineTuned` does it), to "embedding", float32 [batch, 256], L2-normalised. Only the batch axis is dynamic. The
float16 copy computes in float16 but keeps float32 inputs and outputs, so a page feeds both files the same way.
Like the weights, the files are trained on Riot's card art and stay private.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..encoders import l2n, letterbox

OPSET = 20  # the first opset with a Gelu operator (one node, not the erf written out); 18 for an older runtime


def batch_of(images: Sequence[Image.Image], size: int) -> np.ndarray:
    """What `model.to_batch` makes, as the float32 NCHW array the graph takes (values 0..255)."""
    x = np.stack([np.asarray(letterbox(im, size), np.uint8) for im in images])
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def random_images(n: int, size: int, seed: int = 0) -> np.ndarray:
    """Smooth random pictures (a random 8 x 8 colour grid scaled up) as a graph input: not plain noise."""
    rng = np.random.default_rng(seed)
    grids = [Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)) for _ in range(n)]
    return batch_of([g.resize((size, size), Image.BICUBIC) for g in grids], size)


def export(weights: str | Path, out_dir: str | Path, opset: int = OPSET,
           keep_fp32: Sequence[str] = ()) -> tuple[Path, Path]:
    """Write `<name>.onnx` (float32) and `<name>.fp16.onnx` for packed weights `<name>.pth`. `keep_fp32` names
    operator types the float16 copy still computes in float32 (each with casts around it)."""
    import onnx
    import torch
    from onnxconverter_common import float16

    from .model import FineTuned

    weights, out_dir = Path(weights), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = FineTuned(weights, device="cpu").net  # the packed float16 weights, upcast
    size = int(net.config["img_size"])
    with torch.inference_mode():
        # torch.export would read an example batch of 1 as a constant, so the example holds two
        program = torch.onnx.export(net, (torch.zeros(2, 3, size, size),), dynamo=True, opset_version=opset,
                                    input_names=["crops"], output_names=["embedding"], external_data=False,
                                    dynamic_shapes=({0: torch.export.Dim("batch", min=1)},))
    model = program.model_proto
    onnx.checker.check_model(model)
    fp32, fp16 = out_dir / f"{weights.stem}.onnx", out_dir / f"{weights.stem}.fp16.onnx"
    onnx.save(model, fp32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # one warning per weight of 6e-8, the smallest float16 step
        half = float16.convert_float_to_float16(model, keep_io_types=True,
                                                op_block_list=[*float16.DEFAULT_OP_BLOCK_LIST, *keep_fp32])
    onnx.save(half, fp16)
    return fp32, fp16


def parity(net, paths: Sequence[str | Path], n: int = 8, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Per file, the lowest cosine and the largest |difference| between its rows and `net`'s on `n` random pictures."""
    import torch

    x = random_images(n, int(net.config["img_size"]), seed)
    with torch.inference_mode():
        ref = net(torch.from_numpy(x)).numpy()
    out = {}
    for p in paths:
        got = Onnx(p).run(x)
        cos = (got * ref).sum(axis=1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
        out[Path(p).name] = (float(cos.min()), float(np.abs(got - ref).max()))
    return out


class Onnx:
    """An exported embedder as an `Encoder` (spec `onnx:<file>`): crops go in as `FineTuned` takes them."""

    def __init__(self, path: str | Path, batch: int = 64, threads: int = 0):
        import onnxruntime as ort

        path = Path(path).expanduser()
        raw = path.read_bytes()
        self._fp = hashlib.sha256(raw).hexdigest()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # 0: onnxruntime's own choice
        self.session = ort.InferenceSession(raw, opts, providers=["CPUExecutionProvider"])
        (inp,), (out,) = self.session.get_inputs(), self.session.get_outputs()
        self.input = inp.name
        self.img_size, self.batch, self.dim = int(inp.shape[-1]), batch, int(out.shape[-1])
        self.name = f"onnx:{path.stem}-{self._fp[:8]}"  # a file per name: embedding caches key on it

    def run(self, x: np.ndarray) -> np.ndarray:
        """The graph's rows for a float32 NCHW batch (values 0..255)."""
        return self.session.run(None, {self.input: x})[0]

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = [self.run(batch_of(images[i: i + self.batch], self.img_size)) for i in range(0, len(images), self.batch)]
        return l2n(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def fingerprint(self) -> str:
        return self._fp
