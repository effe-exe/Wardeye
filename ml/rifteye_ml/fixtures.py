# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Procedural stand-in cards for tests and demos.

They share a real card's layout: frame colour, an art window, a name bar, a text box
and a cost gem. They contain no Riot Games content, so they may live in tests, CI and
screenshots. Real card art never enters the repository (decision D-006).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

CARD_W, CARD_H = 744, 1039  # the official image size, 63:88
FRAME_COLOURS = [(196, 58, 49), (58, 124, 196), (70, 160, 90), (200, 160, 60), (140, 80, 170), (90, 90, 100)]


def _span(rng: np.random.Generator, lo: int, hi: int, min_len: int) -> tuple[int, int]:
    """A random [a, b) inside [lo, hi) of length >= min_len."""
    a = int(rng.integers(lo, max(lo + 1, hi - min_len)))
    b = int(rng.integers(min(hi, a + min_len), hi + 1))
    return a, max(b, a + 1)


def synthetic_card(seed: int, landscape: bool = False) -> Image.Image:
    """A deterministic fake card: same seed, same image."""
    rng = np.random.default_rng(seed)
    w, h = (CARD_H, CARD_W) if landscape else (CARD_W, CARD_H)
    img = Image.new("RGB", (w, h), FRAME_COLOURS[int(rng.integers(len(FRAME_COLOURS)))])
    d = ImageDraw.Draw(img)

    ax0, ay0, ax1, ay1 = int(w * 0.07), int(h * 0.09), int(w * 0.93), int(h * 0.56)
    d.rectangle([ax0, ay0, ax1, ay1], fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    for _ in range(int(rng.integers(6, 14))):
        x0, x1 = _span(rng, ax0, ax1, 24)
        y0, y1 = _span(rng, ay0, ay1, 24)
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        kind = int(rng.integers(3))
        if kind == 0:
            d.ellipse([x0, y0, x1, y1], fill=colour)
        elif kind == 1:
            d.rectangle([x0, y0, x1, y1], fill=colour)
        else:
            d.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=colour)

    d.rectangle([int(w * 0.07), int(h * 0.58), int(w * 0.93), int(h * 0.64)], fill=(235, 230, 220))
    d.rectangle([int(w * 0.07), int(h * 0.66), int(w * 0.93), int(h * 0.92)], fill=(245, 242, 235))
    for i in range(int(rng.integers(2, 6))):
        y = int(h * 0.69) + i * int(h * 0.04)
        d.line([int(w * 0.10), y, int(w * (0.4 + rng.random() * 0.5)), y], fill=(60, 60, 60), width=max(2, h // 200))
    d.ellipse([int(w * 0.02), int(h * 0.015), int(w * 0.14), int(h * 0.10)], fill=(20, 20, 20))
    return img


def synthetic_catalog(n: int, seed: int = 0) -> list[dict]:
    """Catalogue rows (same shape as catalog.normalise) for n fake printings."""
    return [
        {
            "printing_id": f"FAKE-{i:03d}",
            "card_id": f"fake-card-{i:03d}",
            "name": f"Fake Card {i}",
            "type": "Unit",
            "domains": [],
            "energy": None,
            "might": None,
            "set_code": "FAKE",
            "collector_number": f"{i:03d}",
            "variant": "standard",
            "language": "en",
            "orientation": "portrait",
            "image_url": f"fixture://{seed + i}",
        }
        for i in range(n)
    ]


def load_fixture_image(row: dict) -> Image.Image:
    """Image loader for rows produced by synthetic_catalog."""
    url = row["image_url"]
    assert url.startswith("fixture://"), url
    return synthetic_card(int(url.removeprefix("fixture://")), landscape=row.get("orientation") == "landscape")
