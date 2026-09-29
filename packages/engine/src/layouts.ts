// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Broadcast layouts, ported from ml/rifteye_ml/live/layouts.py: where the table is on screen and how big a card
// is there. The Layout type is in types.ts; a layout is plain data, so its methods here take it first.

import { pyRound } from './pynum';
import type { Layout } from './types';

/** Which half of the table a point is on: left or right when the players sit at the sides, top or bottom when
 * they sit at the ends. */
export type Side = 'left' | 'right' | 'top' | 'bottom';

/** A layout with the dataclass's defaults filled in: players left and right, cards told from the mat by colour
 * (tolerance 45), no known mat colour, and the mat filling at least 0.6 of the table window. */
export function makeLayout(fields: Pick<Layout, 'name' | 'title' | 'table' | 'card_long_1080'> & Partial<Layout>): Layout {
  return { split: 'vertical', mask: 'notmat', mat_tol: 45, mat: null, mat_share: 0.6, ...fields };
}

/** The presets are frozen, like the dataclass. */
function frozen(l: Layout): Layout {
  Object.freeze(l.table);
  if (l.mat) Object.freeze(l.mat);
  return Object.freeze(l);
}

/** The presets' names. */
export type LayoutName = 'la-rq' | 'plusrb' | 'shenyang';

export const LAYOUTS: Readonly<Record<LayoutName, Layout>> = {
  // Riot's official English stream of the US Regional Qualifiers (Atomic): the table camera in the
  // middle 62%, a player panel on each side, the navy mat; each player plays on their panel's side.
  'la-rq': frozen(
    makeLayout({
      name: 'la-rq',
      title: 'Riftbound Regional Qualifier, official stream',
      table: [0.19, 0.06, 0.81, 1.0],
      card_long_1080: 155,
      mat: [34, 44, 55],
      mat_share: 0.62,
    }),
  ),
  // PlusRB's restream of the Barcelona Regional, the same broadcast package.
  plusrb: frozen(
    makeLayout({ name: 'plusrb', title: 'PlusRB restream', table: [0.19, 0.06, 0.81, 1.0], card_long_1080: 140, mat: [43, 54, 61], mat_share: 0.55 }),
  ),
  // The Shenyang broadcast (M0 reference): full-screen overhead camera, red mat, HUD bands.
  shenyang: frozen(
    makeLayout({
      name: 'shenyang',
      title: 'Shenyang Regional',
      table: [0.17, 0.09, 0.86, 0.884],
      card_long_1080: 131,
      mask: 'border',
      mat: [151, 0, 54],
      mat_share: 0.35,
    }),
  ),
};

/** A card's long side in px in a frame `frameH` px high. */
export function cardPx(layout: Layout, frameH: number): number {
  return (layout.card_long_1080 * frameH) / 1080;
}

/** The table window in px of a w x h frame: x0, y0, x1, y1, each rounded as Python's round() does (ties to even). */
export function box(layout: Layout, w: number, h: number): [number, number, number, number] {
  const [x0, y0, x1, y1] = layout.table;
  return [pyRound(x0 * w), pyRound(y0 * h), pyRound(x1 * w), pyRound(y1 * h)];
}

/** Which player's half of the table (x, y) is on: the table's midline splits the players. */
export function side(layout: Layout, x: number, y: number, w: number, h: number): Side {
  // ponytail: the table's midline splits the players; per-layout zones once shared battlefields need an owner
  const [x0, y0, x1, y1] = box(layout, w, h);
  if (layout.split === 'horizontal') return y < (y0 + y1) / 2 ? 'top' : 'bottom';
  return x < (x0 + x1) / 2 ? 'left' : 'right';
}

/** The two sides, in the order the players are listed. */
export function sides(layout: Layout): [Side, Side] {
  return layout.split === 'horizontal' ? ['top', 'bottom'] : ['left', 'right'];
}
