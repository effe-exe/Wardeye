# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Synthetic broadcast frames of Riftbound boards, with exact annotations (docs/research/04 §4.3).

`layout` samples a plausible board on the table plane: zones, piles and stacks, orientation,
face-down cards, dice, counters and hands. `compose` films it: the table plane, one camera mesh
warp, a broadcast layout, and every card's full quad and visible fraction. `__main__` groups
boards into clips for the real H.264 round trip and writes frames, id maps and annotations.
"""
from .compose import Shot, render, sample_shot
from .layout import Board, Instance, Occluder, sample_board

__all__ = ["Board", "Instance", "Occluder", "Shot", "render", "sample_board", "sample_shot"]
