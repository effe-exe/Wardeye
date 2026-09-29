// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Which gallery levels a table reads against. The live runner (ml/rifteye_ml/live/__main__.py) builds, for cards
// `px` long at 1080p, the levels round(px * f / 10) * 10 for f in 0.8, 0.9 and 1.0; the package holds every level
// a broadcast may need (web_assets.py), and the engine picks the same ones.

const FRACTIONS = [0.8, 0.9, 1.0];

/** Python's round(): a half goes to the even neighbour. */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** The runner's levels for cards `px` long at 1080p, ascending. */
export function levelsFor(px: number): number[] {
  return [...new Set(FRACTIONS.map((f) => pyRound((px * f) / 10) * 10))].sort((a, b) => a - b);
}

/** The level of `available` nearest `want`, in log space like the search's `level_for`. */
function nearest(available: readonly number[], want: number): number {
  let best = available[0]!;
  for (const a of available) if (Math.abs(Math.log(a / want)) < Math.abs(Math.log(best / want))) best = a;
  return best;
}

/** The runner's levels for `px`, each replaced by the nearest one the package holds when it has none of that size
 * (a card size outside the range the package was made for). */
export function pickLevels(px: number, available: readonly number[]): number[] {
  if (available.length === 0) throw new Error('the gallery holds no levels');
  const want = levelsFor(px);
  return [...new Set(want.map((w) => (available.includes(w) ? w : nearest(available, w))))].sort((a, b) => a - b);
}
