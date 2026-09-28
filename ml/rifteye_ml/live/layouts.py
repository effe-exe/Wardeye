# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Broadcast layouts: where the table is on screen and how big a card is there.

Every production puts the table camera in its own box with overlays around it. A layout says
which part of the frame is table (fractions of the frame), how long a card's long side is at 1080p,
how the table splits between the two players, and how the bootstrap finder separates cards from
the mat. Measured on the M0 broadcasts (docs/reports/m0-spike.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    name: str
    title: str
    table: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the frame
    card_long_1080: float                     # a card's long side in px at 1080p
    split: str = "vertical"                   # vertical: players left and right; horizontal: top and bottom
    mask: str = "notmat"                      # notmat: anything unlike the mat; border: dark card borders (red mat)
    mat_tol: int = 45
    # The mat's colour, and the least share of the table window it fills on the table camera (measured on
    # the M0 frames): how a run tells the table camera from the other shots before it knows the overlay.
    mat: tuple[int, int, int] | None = None
    mat_share: float = 0.6

    def card_px(self, frame_h: int) -> float:
        return self.card_long_1080 * frame_h / 1080

    def box(self, w: int, h: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.table
        return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)

    def side(self, x: float, y: float, w: int, h: int) -> str:
        # ponytail: the table's midline splits the players; per-layout zones once shared battlefields need an owner
        x0, y0, x1, y1 = self.box(w, h)
        if self.split == "horizontal":
            return "top" if y < (y0 + y1) / 2 else "bottom"
        return "left" if x < (x0 + x1) / 2 else "right"

    def sides(self) -> tuple[str, str]:
        return ("top", "bottom") if self.split == "horizontal" else ("left", "right")


LAYOUTS = {
    # Riot's official English stream of the US Regional Qualifiers (Atomic): the table camera in the
    # middle 62%, a player panel on each side, the navy mat; each player plays on their panel's side.
    "la-rq": Layout("la-rq", "Riftbound Regional Qualifier, official stream", (0.19, 0.06, 0.81, 1.0), 155,
                    mat=(34, 44, 55), mat_share=0.62),
    # PlusRB's restream of the Barcelona Regional, the same broadcast package.
    "plusrb": Layout("plusrb", "PlusRB restream", (0.19, 0.06, 0.81, 1.0), 140, mat=(43, 54, 61), mat_share=0.55),
    # The Shenyang broadcast (M0 reference): full-screen overhead camera, red mat, HUD bands.
    "shenyang": Layout("shenyang", "Shenyang Regional", (0.17, 0.09, 0.86, 0.884), 131, mask="border",
                       mat=(151, 0, 54), mat_share=0.35),
}
