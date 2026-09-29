# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The amodal card detector (M1): every card's full quad, even where other cards cover it.

`export` turns synthetic boards into training tiles, `model` trains and runs RF-DETR keypoint (needs
`pip install -e '.[detect]'`), `evaluate` scores detections against exact synthetic annotations.
"""
from .export import CLASSES, export_run, tile_targets
from .geometry import CORNERS, canonical_quad, quad_iou, tile_origins

__all__ = ["CLASSES", "CORNERS", "canonical_quad", "export_run", "quad_iou", "tile_origins", "tile_targets"]
