# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Image encoders for card retrieval. Every encoder returns L2-normalised float32 rows.

* `colorgrid[:N][/trimF]`: an N×N colour grid of the whole card, mean-centred. It is the
  fingerprint used by the open-source riftbound-scanner, and the baseline to beat.
* `dhash[:N][/trimF]`: a difference hash (gradient signs) as a ±1 vector; the classic
  perceptual-hash baseline.

`/trimF` crops a fraction F off every edge first, e.g. `colorgrid:16/trim0.03`. On real
stream crops that drops the sleeve edge and the mat around the card (M0: 93.7% → 96.6%).
* `timm:<model>[@<size>][/<pool>]`: any timm backbone, e.g.
  `timm:vit_small_patch14_dinov2.lvd142m` (DINOv2 ViT-S/14, Apache-2.0 weights) at 224 px.
  `/avg` pools the patch tokens instead of using the model's default head (the class token
  for DINOv2). Needs the optional `torch` extra. Cards are letterboxed to a square, so no
  centre crop cuts off the name or the cost.
* `embedder:<file>`: the fine-tuned M1 embedder (`rifteye_ml.embed`), 256-d.
* `onnx:<file>`: an ONNX export of it (`embed onnx`, float32 or float16), run with onnxruntime: the same
  crops in, the same 256-d rows out, and no torch needed.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class Encoder(Protocol):
    name: str
    dim: int

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def fingerprint(self) -> str: ...


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def trim(im: Image.Image, frac: float) -> Image.Image:
    """Crop `frac` of the width and height off every edge."""
    if frac <= 0:
        return im
    w, h = im.size
    dx, dy = round(w * frac), round(h * frac)
    return im.crop((dx, dy, max(dx + 1, w - dx), max(dy + 1, h - dy)))


class ColorGrid:
    def __init__(self, grid: int = 16, trim: float = 0.0):
        self.grid = grid
        self.trim = trim
        self.name = f"colorgrid{grid}" + (f"-trim{trim:g}" if trim else "")
        self.dim = grid * grid * 3

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        g = self.grid
        rows = [np.asarray(trim(im.convert("RGB"), self.trim).resize((g, g), Image.BOX), np.float32).reshape(-1) / 255.0
                for im in images]
        x = np.stack(rows) if rows else np.zeros((0, self.dim), np.float32)
        return l2n(x - x.mean(axis=1, keepdims=True))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()


class DHash:
    def __init__(self, size: int = 16, trim: float = 0.0):
        self.size = size
        self.trim = trim
        self.name = f"dhash{size}" + (f"-trim{trim:g}" if trim else "")
        self.dim = size * size

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        n = self.size
        rows = []
        for im in images:
            a = np.asarray(trim(im.convert("L"), self.trim).resize((n + 1, n), Image.BOX), np.float32)
            rows.append(np.where(a[:, 1:] > a[:, :-1], 1.0, -1.0).reshape(-1))
        return l2n(np.stack(rows)) if rows else np.zeros((0, self.dim), np.float32)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()


def letterbox(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


class TimmEncoder:
    """A pretrained timm backbone used as a frozen feature extractor."""

    def __init__(self, model: str, img_size: int = 224, batch: int = 64, device: str | None = None, pool: str = ""):
        import timm  # optional extra: pip install -e 'ml[torch]'
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        kwargs: dict = {"pretrained": True, "num_classes": 0}
        if pool:
            # Keep the pretrained final norm: for "avg" timm ViTs otherwise add a fresh, untrained fc_norm.
            kwargs.update(global_pool=pool, fc_norm=False)
        attempts = [{"img_size": img_size, **kwargs}, {"img_size": img_size, **{k: v for k, v in kwargs.items() if k != "fc_norm"}},
                    {k: v for k, v in kwargs.items() if k != "fc_norm"}]  # CNNs take neither img_size nor fc_norm
        for i, attempt in enumerate(attempts):
            try:
                self.model = timm.create_model(model, **attempt)
                break
            except TypeError:
                if i == len(attempts) - 1:
                    raise
        self.model.eval().to(self.device)
        # Normalisation belongs to the pretrained weights, not to the architecture.
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(self.device)
        self.img_size = img_size
        self.batch = batch
        self.name = f"timm:{model}@{img_size}" + (f"/{pool}" if pool else "")
        with torch.no_grad():
            self.dim = int(self.model(torch.zeros(1, 3, img_size, img_size, device=self.device)).shape[-1])
        self._fp: str | None = None

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch):
                arr = np.stack([np.asarray(letterbox(im, self.img_size), np.float32) / 255.0 for im in images[i : i + self.batch]])
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
                out.append(self.model((x - self.mean) / self.std).float().cpu().numpy())
        return l2n(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256(self.name.encode())
            for k, v in sorted(self.model.state_dict().items()):
                h.update(k.encode())
                h.update(v.detach().cpu().numpy().tobytes())
            self._fp = h.hexdigest()
        return self._fp


def parse_timm_spec(arg: str) -> tuple[str, int, str]:
    """'<model>[@<size>][/<pool>]' -> (model, size, pool)."""
    rest, _, pool = arg.partition("/")
    model, _, size = rest.partition("@")
    if not model:
        raise ValueError(f"no model in timm encoder spec {arg!r}")
    return model, int(size) if size else 224, pool


class Fused:
    """Several encoders as one: their rows side by side, each scaled by the square root of its
    weight, so a dot product is the weighted mean of the parts' cosines and rows stay unit length.
    Every part sees the same turn of a query, so a search keeps one orientation for all of them.

    Colour and structure fail differently: a die on a card or a foil sheen moves the colour grid,
    a washed-out camera moves dHash less (M0 §5.4)."""

    def __init__(self, parts: Sequence[Encoder], weights: Sequence[float] | None = None):
        if len(parts) < 2:
            raise ValueError("a fused encoder needs at least two parts")
        w = np.asarray(weights if weights is not None else [1.0] * len(parts), np.float64)
        if len(w) != len(parts) or (w <= 0).any():
            raise ValueError(f"need one positive weight per part, got {list(w)}")
        self.parts = list(parts)
        self.weights = w / w.sum()
        equal = np.allclose(self.weights, self.weights[0])
        self.name = "+".join(p.name + ("" if equal else f"*{x:g}") for p, x in zip(self.parts, self.weights))
        self.dim = sum(p.dim for p in self.parts)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.hstack([np.float32(np.sqrt(w)) * p.embed(images) for p, w in zip(self.parts, self.weights)]).astype(np.float32)

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.name.encode())
        for p in self.parts:
            h.update(p.fingerprint().encode())
        return h.hexdigest()


def get_encoder(spec: str) -> Encoder:
    """'colorgrid', 'colorgrid:8', 'dhash', 'dhash:8', 'timm:<model>[@<size>][/<pool>]',
    'embedder:<file>' (fine-tuned weights from `rifteye_ml.embed pack`), or 'onnx:<file>' (from `embed onnx`).
    Parts joined by '+' make a `Fused` encoder, equally weighted unless a part ends in '*<weight>',
    e.g. 'colorgrid/trim0.03+dhash/trim0.03' or 'colorgrid*1+dhash*3'."""
    if "+" in spec:
        parts = [p.strip().partition("*") for p in spec.split("+")]
        return Fused([get_encoder(p) for p, _, _ in parts], [float(w) if w else 1.0 for _, _, w in parts])
    kind = spec.split(":")[0].split("/")[0]
    rest = spec[len(kind):]
    arg = rest[1:] if rest.startswith(":") else rest
    if kind in ("colorgrid", "dhash"):
        size, _, opt = arg.partition("/")
        frac = float(opt.removeprefix("trim")) if opt else 0.0
        if opt and not opt.startswith("trim"):
            raise ValueError(f"unknown option {opt!r} in encoder spec {spec!r}")
        cls = ColorGrid if kind == "colorgrid" else DHash
        return cls(int(size) if size else 16, trim=frac)
    if kind == "timm":
        model, size, pool = parse_timm_spec(arg)
        return TimmEncoder(model, img_size=size, pool=pool)
    if kind == "embedder":
        from .embed.model import FineTuned  # needs torch and timm

        if not arg:
            raise ValueError("embedder needs a weights file: 'embedder:<file>'")
        return FineTuned(arg)
    if kind == "onnx":
        from .embed.onnx import Onnx  # needs onnxruntime

        if not arg:
            raise ValueError("onnx needs a model file: 'onnx:<file>'")
        return Onnx(arg)
    raise ValueError(f"unknown encoder {spec!r}")
