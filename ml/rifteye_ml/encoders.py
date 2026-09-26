# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Image encoders for card retrieval. Every encoder returns L2-normalised float32 rows.

* `colorgrid[:N]`: an N×N colour grid of the whole card, mean-centred. It is the
  fingerprint used by the open-source riftbound-scanner, and the baseline to beat.
* `dhash[:N]`: a difference hash (gradient signs) as a ±1 vector; the classic
  perceptual-hash baseline.
* `timm:<model>`: any timm backbone, e.g. `timm:vit_small_patch14_dinov2.lvd142m`
  (DINOv2 ViT-S/14, Apache-2.0 weights). Needs the optional `torch` extra. Cards are
  letterboxed to a square, so no centre crop cuts off the name or the cost.
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


class ColorGrid:
    def __init__(self, grid: int = 16):
        self.grid = grid
        self.name = f"colorgrid{grid}"
        self.dim = grid * grid * 3

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        g = self.grid
        rows = [np.asarray(im.convert("RGB").resize((g, g), Image.BOX), np.float32).reshape(-1) / 255.0 for im in images]
        x = np.stack(rows) if rows else np.zeros((0, self.dim), np.float32)
        return l2n(x - x.mean(axis=1, keepdims=True))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()


class DHash:
    def __init__(self, size: int = 16):
        self.size = size
        self.name = f"dhash{size}"
        self.dim = size * size

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        n = self.size
        rows = []
        for im in images:
            a = np.asarray(im.convert("L").resize((n + 1, n), Image.BOX), np.float32)
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

    def __init__(self, model: str, img_size: int = 224, batch: int = 64, device: str | None = None):
        import timm  # optional extra: pip install -e 'ml[torch]'
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        kwargs = {"pretrained": True, "num_classes": 0}
        try:
            self.model = timm.create_model(model, img_size=img_size, **kwargs)
        except TypeError:  # CNNs take no img_size
            self.model = timm.create_model(model, **kwargs)
        self.model.eval().to(self.device)
        # Normalisation belongs to the pretrained weights, not to the architecture.
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(self.device)
        self.img_size = img_size
        self.batch = batch
        self.name = f"timm:{model}@{img_size}"
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


def get_encoder(spec: str) -> Encoder:
    """'colorgrid', 'colorgrid:8', 'dhash', 'dhash:8', 'timm:<model>' or 'timm:<model>@<size>'."""
    kind, _, arg = spec.partition(":")
    if kind == "colorgrid":
        return ColorGrid(int(arg) if arg else 16)
    if kind == "dhash":
        return DHash(int(arg) if arg else 16)
    if kind == "timm":
        model, _, size = arg.partition("@")
        return TimmEncoder(model, img_size=int(size) if size else 224)
    raise ValueError(f"unknown encoder {spec!r}")
