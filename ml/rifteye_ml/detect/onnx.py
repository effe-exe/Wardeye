# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The card detector as ONNX, for the browser (M2): the same net, run by onnxruntime instead of PyTorch.

The graph takes what a browser has to hand, RGB tiles as float32 in 0..1 (pixels / 255, NCHW, 576 x 576),
normalises them itself and returns the raw head outputs that `Detector.detect_tiles` postprocesses:

* `pred_logits` [batch, 100, 2]: per query, the logits of card and card_back;
* `pred_boxes` [batch, 100, 4]: the box as centre and size, fractions of the tile;
* `pred_keypoints` [batch, 100, 8, 8]: per class, four corners of x, y (fractions of the tile), the found
  and visible logits, the corner's precision (three numbers) and its class logit.

`OnnxNet` stands in for `Detector.net`, so an .onnx file runs through the same postprocess and tile merge.
Needs `pip install -e '.[detector,onnx]'`.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .export import TILE

INPUT = "tiles"
OUTPUTS = ("pred_logits", "pred_boxes", "pred_keypoints")
# RF-DETR's own export opset. Not above 19: onnxruntime-web's WebGPU GridSample (the deformable attention)
# stops at opset 19, and a later one would run on the CPU.
OPSET = 17


class TileNet(nn.Module):
    """An RF-DETR in export mode that takes tiles in 0..1 and returns the outputs named in `OUTPUTS`."""

    def __init__(self, net: nn.Module, means, stds):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(means, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(stds, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, tiles: torch.Tensor):
        boxes, logits, keypoints = self.net((tiles - self.mean) / self.std)
        return logits, boxes, keypoints


def export(det, out: str | Path, dynamic: bool = True, opset: int = OPSET, size: int = TILE) -> Path:
    """Write a `Detector`'s net as float32 ONNX for tiles of `size` px, with a dynamic batch axis unless
    `dynamic` is False. The detector itself is left as it was: a copy is switched to RF-DETR's export forward."""
    from rfdetr.export.prepare import prepare_export_graph

    net = copy.deepcopy(det.net).cpu().float().eval()
    prepare_export_graph(net, det.rf.model_config, shape=(size, size), device="cpu")  # fixes DINOv2's position grid
    net.export()
    model = TileNet(net, det.rf.means, det.rf.stds).eval()
    names = (INPUT, *OUTPUTS)
    # Traced at batch 2, so nothing that should follow the batch is fixed at 1. The tracer warns about RF-DETR's
    # checks on the tile's shape, which the graph fixes anyway.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(model, (torch.rand(2 if dynamic else 1, 3, size, size),), str(out), input_names=[INPUT],
                          output_names=list(OUTPUTS), opset_version=opset, do_constant_folding=True, dynamo=False,
                          dynamic_axes={n: {0: "batch"} for n in names} if dynamic else None)
    return Path(out)


def to_fp16(src: str | Path, out: str | Path) -> Path:
    """A float16 copy of an .onnx file whose inputs and outputs stay float32. The ops on onnxconverter-common's
    own list stay float32 too; here that is the query selection (TopK) and a Range."""
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    # Two things the converter leaves inconsistent. It retypes tensors but not the graph's own casts to float32
    # (DINOv2 casts its input to the weights' type, the attention its scale): those now cast to float16, as
    # their outputs are typed. And an output the graph also reads itself (the class logits are summed from
    # pred_keypoints) reaches those readers as the float32 copy made for the output: they read the float16 one.
    g = model.graph
    half = {v.name for v in g.value_info if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16}
    outs = {o.name for o in g.output}
    halved = {n.output[0]: n.input[0] for n in g.node if n.op_type == "Cast" and n.output[0] in outs}
    for node in g.node:
        if node.op_type == "Cast" and node.output[0] in half:
            for a in node.attribute:
                if a.name == "to" and a.i == onnx.TensorProto.FLOAT:
                    a.i = onnx.TensorProto.FLOAT16
        node.input[:] = [halved.get(i, i) for i in node.input]
    onnx.save(model, str(out))
    return Path(out)


def session(path: str | Path, threads: int | None = None):
    """An onnxruntime CPU session."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class OnnxNet:
    """An .onnx detector in `Detector.net`'s place: `det.net = OnnxNet(path, det.rf.means, det.rf.stds)`.

    It takes the normalised batch `detect_tiles` makes, undoes the normalisation (the graph does its own) and
    returns the head outputs as torch tensors, so the same postprocess and tile merge run. The tiles are 8-bit
    pictures, so rounding to whole levels / 255 gives the graph exactly what a browser would send."""

    def __init__(self, path: str | Path, means, stds, threads: int | None = None):
        self.sess = session(path, threads)
        self.mean = torch.tensor(means, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(stds, dtype=torch.float32).view(1, 3, 1, 1)

    def run(self, tiles: np.ndarray) -> dict[str, np.ndarray]:
        """The outputs for tiles in 0..1, [batch, 3, 576, 576] float32."""
        return dict(zip(OUTPUTS, self.sess.run(list(OUTPUTS), {INPUT: np.ascontiguousarray(tiles, np.float32)})))

    def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        tiles = (torch.round((batch.detach().float().cpu() * self.std + self.mean) * 255) / 255).numpy()
        return {k: torch.from_numpy(v).to(batch.device) for k, v in self.run(tiles).items()}
