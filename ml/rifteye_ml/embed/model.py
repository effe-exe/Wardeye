# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The embedder network, its loss and its training loop.

`Net` is a timm backbone (DINOv2 ViT-S/14, Apache-2.0 weights) read at its class token, a linear
neck to 256 dimensions, and L2 normalisation. It is trained with Sub-center ArcFace over gameplay
cards: three centres per card, so alt arts and signature printings of one card need not share one.

A batch mixes the crop bank (degraded, upright) with clean art made the way the gallery is (the
card at a random on-screen size, `retrieval.at_long_side`), so both sides of a search learn the same
space. Random covering: a card may be cut to the band a stack leaves visible, or have another card
over one end. Crops are letterboxed exactly as `encoders.TimmEncoder` does. Nothing is flipped:
cards are not symmetric.

The packed weights (`pack`) load as the encoder `embedder:<file>` (`FineTuned`), for every tool that
takes an encoder spec. They are trained on Riot's card art, so they stay private like it.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ..encoders import l2n, letterbox
from ..retrieval import at_long_side, band
from .bank import ROW, Bank, donor_for, swap_text, text_groups

BACKBONE = "vit_small_patch14_dinov2.lvd142m"


class Net(nn.Module):
    """uint8 NCHW images to L2-normalised float32 embeddings."""

    def __init__(self, backbone: str = BACKBONE, img_size: int = 224, dim: int = 256, pretrained: bool = True,
                 drop_path: float = 0.1, mean: Sequence[float] | None = None, std: Sequence[float] | None = None):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size,
                                          drop_path_rate=drop_path)
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        mean = list(mean or cfg["mean"])
        std = list(std or cfg["std"])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.neck = nn.Linear(self.backbone.num_features, dim, bias=False)
        nn.init.orthogonal_(self.neck.weight)  # starts as a projection that keeps angles
        self.config = {"backbone": backbone, "img_size": img_size, "dim": dim, "mean": mean, "std": std}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() / 255.0 - self.mean) / self.std
        return F.normalize(self.neck(self.backbone(x)).float(), dim=-1)


class SubCenterArcFace(nn.Module):
    """ArcFace with `k` centres per class; a sample is scored against its class's nearest centre."""

    def __init__(self, dim: int, classes: int, k: int = 3, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.classes, self.k, self.s, self.m = classes, k, s, m
        self.weight = nn.Parameter(torch.empty(classes * k, dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        """Batch × classes."""
        return (emb @ F.normalize(self.weight, dim=1).T).view(-1, self.classes, self.k).amax(dim=2)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, margin: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.m if margin is None else margin
        cos = self.cosine(emb.float()).clamp(-1 + 1e-7, 1 - 1e-7)
        target = cos.gather(1, labels[:, None])
        sine = torch.sqrt((1.0 - target * target).clamp(min=1e-6))
        phi = target * math.cos(m) - sine * math.sin(m)  # cos(theta + m)
        phi = torch.where(target > math.cos(math.pi - m), phi, target - math.sin(math.pi - m) * m)
        logits = cos.scatter(1, labels[:, None], phi) * self.s
        return F.cross_entropy(logits, labels), cos.detach()


# ------------------------------------------------------------------------------------
# Samples
# ------------------------------------------------------------------------------------

def random_view(rng: np.random.Generator) -> str:
    """A band a stack leaves visible: mostly the top, a quarter to 70% of the card."""
    edge = ("top", "top", "bottom", "left", "right")[int(rng.integers(5))]
    return f"{edge}:{rng.uniform(0.25, 0.7):.2f}"


def cover(im: Image.Image, other: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Another card over one end of `im`, showing its own top."""
    w, h = im.size
    depth = max(1, round(h * rng.uniform(0.15, 0.5)))
    top = other.resize((w, max(2, round(other.height * w / max(1, other.width)))), Image.BILINEAR)
    part = top.crop((0, 0, w, min(depth, top.height)))
    out = im.copy()
    out.paste(part, (0, 0) if rng.random() < 0.5 else (0, h - part.height))
    return out


class Samples(torch.utils.data.Dataset):
    """Bank crops, then `clean_per_printing` clean renders of each training printing. An index is
    epoch × len + item, so every sample's augmentation is fixed by (seed, epoch, item), however the
    data loader spreads the work."""

    def __init__(self, bank: Bank, items: np.ndarray, clean_rows: Sequence[int], labels: dict[int, int],
                 images: Sequence[Image.Image], rows: Sequence[dict], img_size: int, seed: int = 0,
                 p_band: float = 0.3, p_cover: float = 0.15, p_text: float = 0.2, sizes: tuple[int, int] = (36, 176)):
        self.bank, self.items, self.clean_rows, self.labels = bank, np.asarray(items), list(clean_rows), labels
        self.images, self.img_size, self.seed = images, img_size, seed
        self.portrait = [r.get("orientation") != "landscape" for r in rows]
        allowed = set(labels)  # text donors come from the training printings only
        self.groups = {}
        for key, members in text_groups(rows).items():
            members = [i for i in members if i in allowed]
            if len(members) > 1:
                self.groups.update({i: members for i in members})
        self.p_band, self.p_cover, self.p_text = p_band, p_cover, p_text
        self.log_sizes = (math.log(sizes[0]), math.log(sizes[1]))

    def __len__(self) -> int:
        return len(self.items) + len(self.clean_rows)

    def sample(self, idx: int) -> tuple[Image.Image, int]:
        epoch, i = divmod(int(idx), len(self))
        rng = np.random.default_rng([self.seed, epoch, i])
        if i < len(self.items):
            b = int(self.items[i])
            row = int(self.bank.index[b, 1 + ROW])
            im = self.bank.image(b)
            if rng.random() < self.p_cover:
                im = cover(im, self.bank.image(int(self.items[int(rng.integers(len(self.items)))])), rng)
        else:
            row = self.clean_rows[i - len(self.items)]
            im = self.images[row]
            members = self.groups.get(row)
            if members and rng.random() < self.p_text:
                im = swap_text(im, self.images[donor_for(row, members, rng)])
            im = at_long_side(im, round(math.exp(rng.uniform(*self.log_sizes))))
        if self.portrait[row] and rng.random() < self.p_band:
            im = band(im, random_view(rng))
        return im, self.labels[row]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        im, label = self.sample(idx)
        x = np.asarray(letterbox(im, self.img_size), np.uint8).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), label


class EpochSampler(torch.utils.data.Sampler):
    """A fresh shuffle each epoch, as indices epoch × n + item (see `Samples`)."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.seed, self.epoch = n, seed, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter((torch.randperm(self.n, generator=g) + self.epoch * self.n).tolist())

    def __len__(self) -> int:
        return self.n


def to_batch(images: Sequence[Image.Image], size: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.asarray(letterbox(im, size), np.uint8) for im in images])).permute(0, 3, 1, 2).contiguous()


# ------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------

def param_groups(net: Net, head: nn.Module, lr: float, lr_head: float, decay: float = 0.8, wd: float = 0.05) -> list[dict]:
    """Layer-wise decayed rates for the backbone (the stem slowest), full rate for the neck and centres."""
    blocks = len(getattr(net.backbone, "blocks", []))
    groups: dict[tuple[int, bool], dict] = {}
    for name, p in net.backbone.named_parameters():
        if name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
        elif name.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed", "mask_token")):
            depth = 0
        else:
            depth = blocks + 1
        flat = p.ndim <= 1 or name.endswith("_token") or name == "pos_embed"
        g = groups.setdefault((depth, flat), {"params": [], "lr": lr * decay ** (blocks + 1 - depth),
                                              "weight_decay": 0.0 if flat else wd})
        g["params"].append(p)
    return [*groups.values(), {"params": list(net.neck.parameters()), "lr": lr_head, "weight_decay": 1e-4},
            {"params": list(head.parameters()), "lr": lr_head, "weight_decay": 0.0}]


def schedule(total: int, warmup: int, floor: float = 0.01) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def init_centres(net: Net, head: SubCenterArcFace, images: Sequence[Image.Image], rows_of_class: list[list[int]],
                 device: torch.device, size: int = 120, batch: int = 64) -> None:
    """Start each card's centres at its printings' clean art: centre j at printing j (cycling), so alt
    arts begin apart. A little noise keeps repeated centres distinct."""
    net.eval()
    order = sorted({r for rs in rows_of_class for r in rs})
    emb = {}
    for s in range(0, len(order), batch):
        part = order[s: s + batch]
        x = to_batch([at_long_side(images[r], size) for r in part], net.config["img_size"]).to(device)
        for r, e in zip(part, net(x).float().cpu()):
            emb[r] = e
    g = torch.Generator().manual_seed(0)
    w = torch.stack([emb[rs[j % len(rs)]] for rs in rows_of_class for j in range(head.k)])
    w = F.normalize(w + 0.01 * torch.randn(w.shape, generator=g), dim=1)
    head.weight.data.copy_(w.to(head.weight.device))


def train(bank: Bank, rows: Sequence[dict], images: Sequence[Image.Image], out: str | Path,
          train_rows: Sequence[int] | None = None, epochs: int = 8, batch_size: int = 192, lr: float = 3e-5,
          lr_head: float = 1e-3, clean_per_printing: int = 20, device: str | None = None, workers: int = 4,
          seed: int = 0, amp: bool | None = None, backbone: str = BACKBONE, img_size: int = 224, dim: int = 256,
          pretrained: bool = True, k: int = 3, margin: float = 0.3, scale: float = 30.0,
          log: Callable[[str], None] = print, meta: dict | None = None) -> Path:
    """Fine-tune and write `final.pt`. `last.pt` is rewritten every epoch and resumed from."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    amp = (dev.type == "cuda") if amp is None else amp
    torch.manual_seed(seed)
    keep = sorted(set(range(len(rows))) if train_rows is None else set(int(r) for r in train_rows))
    cards = sorted({rows[r]["card_id"] for r in keep})
    cls = {c: n for n, c in enumerate(cards)}
    labels = {r: cls[rows[r]["card_id"]] for r in keep}
    rows_of_class: list[list[int]] = [[] for _ in cards]
    for r in keep:
        rows_of_class[labels[r]].append(r)
    items = np.flatnonzero(np.isin(bank.rows, keep))
    data = Samples(bank, items, [r for r in keep for _ in range(clean_per_printing)], labels, images, rows, img_size, seed=seed)
    sampler = EpochSampler(len(data), seed)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=workers,
                                         pin_memory=dev.type == "cuda", drop_last=True, persistent_workers=workers > 0,
                                         prefetch_factor=4 if workers > 0 else None)
    steps = len(loader)
    if steps == 0:
        raise ValueError(f"{len(data)} samples make no batch of {batch_size}")
    total = steps * epochs

    net = Net(backbone, img_size, dim, pretrained=pretrained).to(dev)
    head = SubCenterArcFace(dim, len(cards), k=k, s=scale, m=margin).to(dev)
    opt = torch.optim.AdamW(param_groups(net, head, lr, lr_head), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(total, warmup=min(500, max(1, total // 10))))
    start, step = 0, 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["net"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, step = int(ck["epoch"]) + 1, int(ck["step"])
        log(f"resuming after epoch {start} of {epochs}")
    else:
        init_centres(net, head, images, rows_of_class, dev)
    log(f"{len(keep)} printings of {len(cards)} cards; {len(items)} bank crops + {len(data) - len(items)} clean renders "
        f"per epoch; {steps} steps of {batch_size} on {dev} (bf16: {amp})")
    params = [p for g in opt.param_groups for p in g["params"]]
    warm = max(1, steps)  # the margin grows over the first epoch
    metrics = out / "metrics.csv"
    if start == 0 and metrics.exists():
        metrics.unlink()
    for epoch in range(start, epochs):
        sampler.epoch = epoch
        net.train()
        head.train()
        t0, seen, loss_sum, hits = time.time(), 0, 0.0, 0
        for n, (x, y) in enumerate(loader, 1):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                emb = net(x)
            loss, cos = head(emb, y, margin=margin * min(1.0, step / warm))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(y)
            loss_sum += float(loss.detach()) * len(y)
            hits += int((cos.argmax(dim=1) == y).sum())
            if n % 100 == 0:
                log(f"  epoch {epoch + 1} step {n}/{steps}: loss {loss_sum / seen:.3f}, accuracy {hits / seen:.3f}, "
                    f"{seen / (time.time() - t0):.0f} img/s")
        secs = time.time() - t0
        ck = {"net": net.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "step": step}
        torch.save(ck, out / "last.tmp")
        os.replace(out / "last.tmp", last)
        new = not metrics.exists()
        with open(metrics, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "step", "loss", "train_accuracy", "lr", "seconds"])
            w.writerow([epoch + 1, step, f"{loss_sum / seen:.4f}", f"{hits / seen:.4f}", f"{sched.get_last_lr()[0]:.3g}", f"{secs:.0f}"])
        log(f"== {time.strftime('%H:%M:%S', time.gmtime())} epoch {epoch + 1}/{epochs}: loss {loss_sum / seen:.3f}, "
            f"accuracy {hits / seen:.3f}, {seen / secs:.0f} img/s")
    final = out / "final.pt"
    torch.save({"net": net.state_dict(), "config": net.config, "cards": cards,
                "meta": {**(meta or {}), "printings": len(keep), "cards": len(cards), "epochs": epochs}},
               out / "final.tmp")
    os.replace(out / "final.tmp", final)
    return final


def pack(final: str | Path, out: str | Path) -> int:
    """What retrieval needs, in float16: the backbone and neck with their config (about 45 MB)."""
    ck = torch.load(str(final), map_location="cpu", weights_only=True)
    small = {"config": ck["config"], "meta": ck.get("meta", {}),
             "net": {k: v.half() if v.is_floating_point() else v for k, v in ck["net"].items()}}
    torch.save(small, str(out))
    return Path(out).stat().st_size


class FineTuned:
    """Packed embedder weights as an `Encoder` (spec `embedder:<file>`)."""

    def __init__(self, path: str | Path, batch: int = 64, device: str | None = None):
        path = Path(path).expanduser()
        raw = path.read_bytes()
        self._fp = hashlib.sha256(raw).hexdigest()
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        cfg = ck["config"]
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.net = Net(cfg["backbone"], cfg["img_size"], cfg["dim"], pretrained=False, drop_path=0.0,
                       mean=cfg.get("mean"), std=cfg.get("std"))
        self.net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["net"].items()})
        self.net.eval().to(self.device)
        self.meta = ck.get("meta", {})
        self.img_size, self.batch, self.dim = int(cfg["img_size"]), batch, int(cfg["dim"])
        self.name = f"embedder:{path.stem}-{self._fp[:8]}"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch):
                x = to_batch(images[i: i + self.batch], self.img_size).to(self.device)
                out.append(self.net(x).float().cpu().numpy())
        return l2n(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def fingerprint(self) -> str:
        return self._fp
