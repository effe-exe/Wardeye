// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The brand mark, a stake topped with a vision orb, for the badge on the player: the shapes of assets/brand/logo/mark.svg
// (test/mark.test.ts holds them to that file). Only geometry is here. The paint comes from the overlay's tokens
// (.rifteye-mark-* in overlay.css), so no colour is written in the script.

export const MARK_VIEWBOX = '0 0 53.688 96.836';

export interface MarkShape {
  tag: 'path' | 'circle';
  /** The class the stylesheet paints it by. */
  cls: string;
  /** The shape's geometry, as the SVG attributes give it. */
  attrs: Readonly<Record<string, string>>;
}

export const MARK_SHAPES: readonly MarkShape[] = [
  // the stake and its base
  { tag: 'path', cls: 'rifteye-mark-ward', attrs: { d: 'M 17.250 48.898 L 13.414 84.375 L 21.086 90.125 L 32.594 90.125 L 40.266 84.375 L 36.430 48.898Z M 9.578 90.125 L 44.102 90.125 L 38.344 96.836 L 15.336 96.836Z' } },
  // the orb: its body, its ring, its pupil and the glint on it
  { tag: 'circle', cls: 'rifteye-mark-ward', attrs: { cx: '26.844', cy: '26.844', r: '26.844' } },
  { tag: 'circle', cls: 'rifteye-mark-ring', attrs: { cx: '26.844', cy: '26.844', r: '17.258', 'stroke-width': '5.753' } },
  { tag: 'circle', cls: 'rifteye-mark-pupil', attrs: { cx: '26.844', cy: '26.844', r: '8.625' } },
  { tag: 'circle', cls: 'rifteye-mark-glint', attrs: { cx: '32.594', cy: '21.086', r: '3.836' } },
];
