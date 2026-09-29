// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The presets as a fallback: when the layout cannot be found from the footage, a broadcast the presets know shows
// itself by its mat: the share of its table window within the layout's tolerance of the mat's colour (as the
// recogniser's own scene test, `Scene.table_like`, asks it). Presets whose mats are alike (the Regional
// Qualifier's and PlusRB's restream) are told apart by how near the window's colour is to each. Only the table
// window is looked at (D-005).

import { layouts, type Layout, type RgbImage } from '@rifteye/engine';

const GRID = [160, 120] as const;

/** How much of the layout's table window, sampled on a 160 x 120 grid, is near its mat's colour (within `mat_tol`),
 * and how far, on average, those pixels are from it. A layout with no mat colour matches nothing. */
export function matStats(frame: RgbImage, layout: Layout): { share: number; dist: number } {
  if (!layout.mat) return { share: 0, dist: Infinity };
  const [x0, y0, x1, y1] = layouts.box(layout, frame.width, frame.height);
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const [mr, mg, mb] = layout.mat;
  let near = 0;
  let far = 0;
  for (let j = 0; j < GRID[1]; j++) {
    const y = Math.min(frame.height - 1, y0 + Math.floor(((j + 0.5) * h) / GRID[1]));
    for (let i = 0; i < GRID[0]; i++) {
      const x = Math.min(frame.width - 1, x0 + Math.floor(((i + 0.5) * w) / GRID[0]));
      const p = (y * frame.width + x) * 3;
      const d = Math.max(Math.abs(frame.data[p]! - mr), Math.abs(frame.data[p + 1]! - mg), Math.abs(frame.data[p + 2]! - mb));
      if (d < layout.mat_tol) {
        near++;
        far += d;
      }
    }
  }
  return { share: near / (GRID[0] * GRID[1]), dist: near ? far / near : Infinity };
}

/** The preset whose mat fills its table window as much as it needs to (its `mat_share`) and whose colour is nearest
 * the window's; null when none does. */
export function matchPreset(frame: RgbImage, presets: readonly Layout[]): Layout | null {
  let best: Layout | null = null;
  let bestDist = Infinity;
  for (const p of presets) {
    const { share, dist } = matStats(frame, p);
    if (share >= p.mat_share && dist < bestDist) {
      best = p;
      bestDist = dist;
    }
  }
  return best;
}
