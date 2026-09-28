# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The crop bank: every printing as a stream shows it, many times over, cropped upright.

One task is one seed at one card height. Every printing is placed on boards, degraded at the camera
realism level with foil sheen, pushed through H.264 and cropped back out (`degrade.simulate`, the
M0 pipeline). Each task draws its own frame size (1080p or 720p), bitrate (2-8 Mbit/s) and
strength (0.6, 1 or 1.4 times the camera level's magnitudes), so the bank spans more than one
broadcast.

Text scrambling: before degrading, some standard Unit, Spell and Gear printings get the rules text
box of another printing of the same type and set. Text layout differs by card and by language (the
Shenyang broadcast played Chinese cards), and the identity should come from the art.

A task writes one shard: `<name>.u8`, its crops' RGB pixels back to back, then `<name>.npy`, one row
per crop (byte offset, height, width, catalogue row, card height). The `.npy` comes last, so a shard
that has one is complete and a rerun skips it. `Bank` maps the shards read-only, so the trainer's
data loader workers share one copy in memory.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from ..degrade import REALISM, StreamSettings, simulate, with_realism

TEXT_BOX = (0.05, 0.635, 0.95, 0.915)  # x0, y0, x1, y1 as fractions of a portrait card: the rules text
TEXT_TYPES = ("Unit", "Spell", "Gear")  # the standard frame; legends, runes and battlefields differ
OFFSET, HEIGHT, WIDTH, ROW, CARD_H = range(5)


@dataclass(frozen=True)
class Task:
    name: str
    seed: int
    card_h: int
    frame: tuple[int, int] = (1920, 1080)
    bitrate: int = 4000
    realism: str = "foil"
    strength: float = 1.0
    text_prob: float = 0.0

    def settings(self) -> StreamSettings:
        s = with_realism(StreamSettings(frame_w=self.frame[0], frame_h=self.frame[1], card_h=self.card_h,
                                        bitrate_kbps=self.bitrate, seed=self.seed), self.realism)
        if self.strength != 1.0:
            scaled = {k: getattr(s, k) * self.strength for k in REALISM[self.realism]}
            s = replace(s, **{k: min(v, 1.0) if k.endswith("_prob") else v for k, v in scaled.items()})
        return s


def plan(seeds: Sequence[int], heights: Sequence[int], text_prob: float = 0.2, realism: str = "foil",
         frame: tuple[int, int] | None = None) -> list[Task]:
    """Training tasks: every seed at every height, each with its own frame (unless `frame` fixes it),
    bitrate and strength."""
    out = []
    for k in seeds:
        for h in heights:
            r = np.random.default_rng([k, h])
            size = (1280, 720) if r.random() < 0.3 else (1920, 1080)
            bitrate = int(round(float(np.exp(r.uniform(np.log(2000), np.log(8000)))), -2))
            strength = float(r.choice([0.6, 1.0, 1.4]))
            out.append(Task(f"s{k:03d}_h{h:03d}", seed=100_000 + 1000 * k + h, card_h=h, frame=frame or size, bitrate=bitrate,
                            realism=realism, strength=strength, text_prob=text_prob))
    return out


def plan_eval(heights: Sequence[int], seed: int = 0, bitrate: int = 4000, frame: tuple[int, int] = (1920, 1080)) -> list[Task]:
    """Fresh crops for scoring, as the M0 quick fine-tune made them (`adapter`): camera realism, 1080p
    at 4 Mbit/s, seed 0, no text scrambling."""
    return [Task(f"eval_h{h:03d}", seed=seed, card_h=h, frame=frame, bitrate=bitrate, realism="camera") for h in heights]


def text_groups(rows: Sequence[dict]) -> dict[tuple[str, str], list[int]]:
    """Printings whose rules text can trade places, by type and set: standard portrait frames only.
    Within one set, a model trained on some sets never sees the text of the others."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("type") in TEXT_TYPES and r.get("variant", "standard") == "standard" and r.get("orientation") != "landscape":
            groups.setdefault((r["type"], r.get("set_code", "")), []).append(i)
    return groups


def _box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = (round(f * s) for f, s in zip(TEXT_BOX, (w, h, w, h)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def swap_text(im: Image.Image, donor: Image.Image) -> Image.Image:
    """`im` with the rules text box of `donor`."""
    box = _box(im.size)
    part = donor.convert("RGB").crop(_box(donor.size)).resize((box[2] - box[0], box[3] - box[1]), Image.BILINEAR)
    out = im.convert("RGB").copy()
    out.paste(part, box[:2])
    return out


def donor_for(i: int, members: Sequence[int], rng: np.random.Generator) -> int:
    """Another member than `i`, uniformly."""
    j = members[int(rng.integers(len(members) - 1))]
    return members[-1] if j == i else j


def scrambled(images: Sequence[Image.Image], groups: dict, prob: float, rng: np.random.Generator) -> list[Image.Image]:
    """A task's copy of the art: each eligible printing's text swapped with probability `prob`."""
    out = list(images)
    if prob <= 0:
        return out
    for members in groups.values():
        if len(members) < 2:
            continue
        for i in members:
            if rng.random() < prob:
                out[i] = swap_text(images[i], images[donor_for(i, members, rng)])
    return out


# ------------------------------------------------------------------------------------
# Generation, one process per task
# ------------------------------------------------------------------------------------

_STATE: dict = {}


def _load(rows: Sequence[dict], cache: str, max_side: int) -> None:
    from ..spike import _cached_loader

    load = _cached_loader(cache, max_side)
    _STATE.update(images=[load(r) for r in rows], groups=text_groups(rows))


def _run(task: Task, out: str) -> tuple[str, int, float]:
    t0 = time.time()
    rng = np.random.default_rng(task.seed + 7)
    cards = scrambled(_STATE["images"], _STATE["groups"], task.text_prob, rng)
    crops = simulate(cards, task.settings())
    arrays = [np.ascontiguousarray(np.asarray(c.upright().convert("RGB"), np.uint8)) for c in crops]
    index = np.zeros((len(arrays), 5), np.int64)
    off = 0
    for n, (c, a) in enumerate(zip(crops, arrays)):
        index[n] = (off, a.shape[0], a.shape[1], c.card_index, task.card_h)
        off += a.size
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    tmp_u8, tmp_idx = folder / f".{task.name}.u8", folder / f".{task.name}.npy"
    with open(tmp_u8, "wb") as f:
        for a in arrays:
            f.write(a.tobytes())
    np.save(tmp_idx, index)
    os.replace(tmp_u8, folder / f"{task.name}.u8")
    os.replace(tmp_idx, folder / f"{task.name}.npy")  # last: its presence marks the shard complete
    return task.name, len(arrays), time.time() - t0


def generate(rows: Sequence[dict], cache: str, tasks: Sequence[Task], out: str | Path, workers: int = 1,
             max_side: int = 512, log: Callable[[str], None] = print) -> int:
    """Run the tasks whose shard is missing, `workers` at a time. Returns how many ran."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    ids = [r["printing_id"] for r in rows]
    manifest = folder / "printings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != ids:
        raise SystemExit(f"{folder} holds crops of another catalogue; write to a new folder")
    manifest.write_text(json.dumps(ids))
    todo = [t for t in tasks if not (folder / f"{t.name}.npy").exists()]
    if len(todo) < len(tasks):
        log(f"  {len(tasks) - len(todo)} of {len(tasks)} shards already there")
    if not todo:
        return 0
    t0 = time.time()
    if workers <= 1:
        _load(rows, cache, max_side)
        for n, t in enumerate(todo, 1):
            name, count, secs = _run(t, str(folder))
            log(f"  {n}/{len(todo)} {name}: {count} crops in {secs:.0f} s")
        return len(todo)
    # Linux forks: the art is loaded once and the workers share its pages. Elsewhere each worker loads it.
    fork = sys.platform.startswith("linux")
    ctx = multiprocessing.get_context("fork" if fork else "spawn")
    kw: dict = {}
    if fork:
        _load(rows, cache, max_side)
    else:
        kw = {"initializer": _load, "initargs": (list(rows), cache, max_side)}
    pool = ProcessPoolExecutor(workers, mp_context=ctx, **kw)
    try:
        futures = [pool.submit(_run, t, str(folder)) for t in todo]
        for n, f in enumerate(as_completed(futures), 1):
            name, count, secs = f.result()
            if n == 1 or n == len(todo) or n % max(1, len(todo) // 20) == 0:
                log(f"  {n}/{len(todo)} {name}: {count} crops in {secs:.0f} s ({time.time() - t0:.0f} s so far)")
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return len(todo)


# ------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------

class Bank:
    """The crops of one or more bank folders, mapped read-only. `printings` lists the catalogue the
    crops' rows refer to, in order."""

    def __init__(self, folders: Sequence[str | Path]):
        self.shards: list[Path] = []
        self.printings: list[str] = []
        parts = []
        for folder in folders:
            manifest = Path(folder) / "printings.json"
            if not manifest.exists():
                raise FileNotFoundError(f"{manifest} is missing; is {folder} a crop bank?")
            ids = json.loads(manifest.read_text())
            if self.printings and ids != self.printings:
                raise ValueError(f"{folder} was made from another catalogue than {folders[0]}")
            self.printings = ids
            for p in sorted(Path(folder).glob("*.npy")):
                if p.name.startswith("."):  # a shard still being written
                    continue
                x = np.load(p)
                if len(x) == 0:
                    continue
                self.shards.append(p.with_suffix(".u8"))
                parts.append(np.column_stack([np.full(len(x), len(self.shards) - 1, np.int64), x]))
        if not parts:
            raise FileNotFoundError(f"no crop shards in {', '.join(str(f) for f in folders)}")
        self.index = np.concatenate(parts)  # shard, then OFFSET, HEIGHT, WIDTH, ROW, CARD_H
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> dict:  # data loader workers started by spawn map the shards themselves
        return {**self.__dict__, "_maps": {}}

    @property
    def rows(self) -> np.ndarray:
        return self.index[:, 1 + ROW]

    @property
    def heights(self) -> np.ndarray:
        return self.index[:, 1 + CARD_H]

    def crop(self, i: int) -> np.ndarray:
        k, off, h, w = (int(v) for v in self.index[i, :4])
        m = self._maps.get(k)
        if m is None:
            m = self._maps[k] = np.memmap(self.shards[k], np.uint8, mode="r")
        return np.array(m[off: off + h * w * 3]).reshape(h, w, 3)

    def image(self, i: int) -> Image.Image:
        return Image.fromarray(self.crop(i))
