// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The mat and card tests, ported from ml/rifteye_ml/matcrops.py: the mat's colour, which pixels are not mat, which
// look like a dark card border, and how much detail a crop has (a face has art and text, a card back is one
// colour). The bootstrap finder (find_cards) is here too, for the layout finder's fallback without a detector.

import * as image from './image';
import { binaryClosing, binaryFillHoles, binaryOpening, findObjects, label, type Mask } from './ndimage';
import { pyMod, pyRound } from './pynum';
import type { CardBox, RgbImage } from './types';

export type { Mask };

/** Below this, a crop is a plain sleeve or card back. */
export const FACE_DOWN_DETAIL = 4.0;

/** Mean grey-level step between neighbouring pixels inside the card (its middle 76%, at 48 x 64). A face has art,
 * frame and text; a sleeve back is one colour. Float32 as in numpy, so the same value as Python's. */
export function detail(im: RgbImage): number {
  const g = image.toGray(im);
  const w = g.width;
  const h = g.height;
  const inner = image.cropGray(g, [pyRound(w * 0.12), pyRound(h * 0.12), pyRound(w * 0.88), pyRound(h * 0.88)]);
  const a = image.resizeGray(inner, [48, 64], 'box');
  return stepMean(a.data, a.width, a.height);
}

/** (np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2 for a picture of 8-bit values, in
 * float32: the sums are whole numbers below 2^24, so exact, and each mean, their sum and the halving are rounded to
 * float32 once, as numpy's float32 scalars do. */
export function stepMean(a: ArrayLike<number>, w: number, h: number): number {
  let down = 0; // |a[y + 1][x] - a[y][x]|
  for (let y = 0; y + 1 < h; y++) for (let x = 0; x < w; x++) down += Math.abs(a[(y + 1) * w + x]! - a[y * w + x]!);
  let across = 0; // |a[y][x + 1] - a[y][x]|
  for (let y = 0; y < h; y++) for (let x = 0; x + 1 < w; x++) across += Math.abs(a[y * w + x + 1]! - a[y * w + x]!);
  const f = Math.fround;
  return f(f(f(down / ((h - 1) * w)) + f(across / (h * (w - 1)))) / 2);
}

/** The playmat's colour: the most common colour of the table area (coarse 3-D histogram peak). `step` takes every
 * step-th row and column, like rgb[::step, ::step]. */
export function matColour(rgb: RgbImage, step = 1): [number, number, number] {
  const { width: w, height: h, data } = rgb;
  const key = (o: number): number => ((data[o]! >> 4) << 8) | ((data[o + 1]! >> 4) << 4) | (data[o + 2]! >> 4);
  const bins = new Int32Array(4096);
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const k = key((y * w + x) * 3);
      bins[k] = bins[k]! + 1;
    }
  }
  let peak = 0; // np.bincount(keys).argmax(): the first of the most common
  for (let k = 1; k < 4096; k++) if (bins[k]! > bins[peak]!) peak = k;
  const hist = [new Int32Array(256), new Int32Array(256), new Int32Array(256)];
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const o = (y * w + x) * 3;
      if (key(o) !== peak) continue;
      for (let c = 0; c < 3; c++) {
        const hc = hist[c]!;
        hc[data[o + c]!] = hc[data[o + c]!]! + 1;
      }
    }
  }
  const n = bins[peak]!;
  return [medianOfCounts(hist[0]!, n), medianOfCounts(hist[1]!, n), medianOfCounts(hist[2]!, n)];
}

/** np.median(values).astype(np.int16 or np.uint8) for `n` 8-bit values given as counts (counts[v] of each v): the
 * middle value, or the mean of the two middle ones cut to a whole number. */
export function medianOfCounts(counts: ArrayLike<number>, n: number): number {
  const nth = (p: number): number => {
    let seen = 0;
    for (let v = 0; v < 256; v++) {
      seen += counts[v]!;
      if (seen > p) return v;
    }
    return 255;
  };
  return Math.floor((nth((n - 1) >> 1) + nth(n >> 1)) / 2);
}

/** np.median(rgb.reshape(-1, 3), axis=0).astype(np.uint8): the colour of the median of each channel. */
export function channelMedians(rgb: RgbImage): [number, number, number] {
  const { data } = rgb;
  const hist = [new Int32Array(256), new Int32Array(256), new Int32Array(256)];
  for (let o = 0; o < data.length; o += 3) {
    for (let c = 0; c < 3; c++) {
      const hc = hist[c]!;
      hc[data[o + c]!] = hc[data[o + c]!]! + 1;
    }
  }
  const n = rgb.width * rgb.height;
  return [medianOfCounts(hist[0]!, n), medianOfCounts(hist[1]!, n), medianOfCounts(hist[2]!, n)];
}

/** Pixels far from the mat colour (max channel difference over `tol`): cards, whatever their border. */
export function notmatMask(rgb: RgbImage, mat: readonly number[], tol = 60): Mask {
  const { width, height, data } = rgb;
  const m0 = Math.trunc(mat[0]!);
  const m1 = Math.trunc(mat[1]!);
  const m2 = Math.trunc(mat[2]!);
  const out = new Uint8Array(width * height);
  for (let i = 0, o = 0; i < out.length; i++, o += 3) {
    const d = Math.max(Math.abs(data[o]! - m0), Math.abs(data[o + 1]! - m1), Math.abs(data[o + 2]! - m2));
    out[i] = d > tol ? 1 : 0;
  }
  return { width, height, data: out };
}

/** Dark, not red: R < rMax, R - G < rg and R - B < rb (the M0 red mat's card borders). */
export function borderMask(rgb: RgbImage, rMax = 90, rg = 45, rb = 45): Mask {
  const { width, height, data } = rgb;
  const out = new Uint8Array(width * height);
  for (let i = 0, o = 0; i < out.length; i++, o += 3) {
    const r = data[o]!;
    out[i] = r < rMax && r - data[o + 1]! < rg && r - data[o + 2]! < rb ? 1 : 0;
  }
  return { width, height, data: out };
}

/** CardBox.aspect: the long side over the short side. */
export function aspect(box: CardBox): number {
  return box.long_px / Math.max(1e-6, box.short_px);
}

/** The corners of the convex hull of 2-D points, one turn round (Andrew's monotone chain; points on an edge are not
 * corners). scipy.spatial.ConvexHull(points).vertices lists the same corners, from a start of Qhull's choosing. */
function convexHull(points: readonly (readonly [number, number])[]): [number, number][] {
  const pts = points.map((p): [number, number] => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const turn = (o: readonly number[], a: readonly number[], b: readonly number[]): number => (a[0]! - o[0]!) * (b[1]! - o[1]!) - (a[1]! - o[1]!) * (b[0]! - o[0]!);
  const half = (order: readonly [number, number][]): [number, number][] => {
    const chain: [number, number][] = [];
    for (const p of order) {
      while (chain.length >= 2 && turn(chain[chain.length - 2]!, chain[chain.length - 1]!, p) <= 0) chain.pop();
      chain.push(p);
    }
    chain.pop();
    return chain;
  };
  return [...half(pts), ...half([...pts].reverse())];
}

/** Minimum-area rectangle of 2-D points (x, y): centre, long side, short side, long-side angle in [0, 180). It tries
 * each edge of the convex hull as a side of the rectangle and keeps the first of the least area. */
export function minAreaRect(points: readonly (readonly [number, number])[]): {
  centre: [number, number];
  long_px: number;
  short_px: number;
  angle_deg: number;
} {
  const hull = convexHull(points);
  if (hull.length < 3) throw new Error('minAreaRect needs points that do not lie on a line');
  let best: { area: number; theta: number; lo: [number, number]; hi: [number, number] } | null = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i]!;
    const b = hull[(i + 1) % hull.length]!;
    const theta = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    let lo0 = Infinity;
    let lo1 = Infinity;
    let hi0 = -Infinity;
    let hi1 = -Infinity;
    for (const [x, y] of hull) {
      const u = x * c + y * s; // hull @ [[c, -s], [s, c]]: into the edge's frame
      const v = x * -s + y * c;
      if (u < lo0) lo0 = u;
      if (u > hi0) hi0 = u;
      if (v < lo1) lo1 = v;
      if (v > hi1) hi1 = v;
    }
    const area = (hi0 - lo0) * (hi1 - lo1);
    if (best === null || area < best.area) best = { area, theta, lo: [lo0, lo1], hi: [hi0, hi1] };
  }
  const { theta, lo, hi } = best!;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2] as const;
  const centre: [number, number] = [mid[0] * c - mid[1] * s, mid[0] * s + mid[1] * c];
  const w = hi[0] - lo[0];
  const h = hi[1] - lo[1];
  const deg = theta * (180 / Math.PI); // math.degrees
  const angle = w >= h ? deg : deg + 90;
  return { centre, long_px: Math.max(w, h), short_px: Math.min(w, h), angle_deg: pyMod(angle + 180, 180) };
}

/** The bootstrap finder: isolated single cards whose long side is within `tol` of `longPx`. `mask` replaces the
 * default dark-border rule (see notmatMask). */
export function findCards(rgb: RgbImage, longPx: number, opts: { tol?: number; aspect?: [number, number]; minFill?: number; mask?: Mask } = {}): CardBox[] {
  const tol = opts.tol ?? 0.12;
  const [aspectLo, aspectHi] = opts.aspect ?? [1.33, 1.46];
  const minFill = opts.minFill ?? 0.88;
  const closed = binaryClosing(opts.mask ?? borderMask(rgb), 3);
  const mask = binaryOpening(binaryFillHoles(closed), 3);
  const { labels, count } = label(mask);
  const boxes: CardBox[] = [];
  const loArea = ((longPx * (1 - tol)) ** 2 / 1.46) * 0.8;
  const hiArea = ((longPx * (1 + tol)) ** 2 / 1.33) * 1.1;
  const w = mask.width;
  findObjects(labels, w, mask.height, count).forEach(([y0, y1, x0, x1], k) => {
    const points: [number, number][] = [];
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (labels[y * w + x] === k + 1) points.push([x, y]);
    const area = points.length;
    if (!(loArea <= area && area <= hiArea)) return;
    const { centre, long_px, short_px, angle_deg } = minAreaRect(points);
    const box: CardBox = { centre, long_px, short_px, angle_deg, fill: area / Math.max(1.0, long_px * short_px) };
    const asp = aspect(box);
    if (Math.abs(long_px / longPx - 1) <= tol && aspectLo <= asp && asp <= aspectHi && box.fill >= minFill) boxes.push(box);
  });
  return boxes;
}
