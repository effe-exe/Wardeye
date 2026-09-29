// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A layout found from the footage itself, ported from ml/rifteye_ml/live/autolayout.py, for a broadcast no preset
// describes. A layout is where the table camera's picture sits in the frame and how long a card is there.
// `tableWindow` finds the picture's borders first: a broadcast that puts panels beside the table draws them as long
// straight edges in the same place in every frame, and the panels, player cams and hand lists outside them are never
// looked at. Inside the borders the mat decides: the colour that fills the middle, grown over the cards lying on it,
// and its bounding box. `cardSize` runs the detector over that window at a few candidate card sizes and keeps the
// one its confident boxes agree on. Frames are the full picture (RgbImage), a handful from the first seconds.

import { norm2 } from './geometry';
import * as image from './image';
import { makeLayout } from './layouts';
import { findCards, matColour, notmatMask } from './matcrops';
import { binaryClosing, binaryFillHoles, label, type Mask } from './ndimage';
import { median, pyRound } from './pynum';
import type { Detection, Layout, RgbImage } from './types';

/** The thumbnail the borders are found on. */
export const W = 480;
export const H = 270;
/** Candidate card long sides at 1080p. */
export const SIZES = [80, 100, 125, 155, 190, 235] as const;

export type Window = [number, number, number, number];

/** The detector, as cardSize calls it: the frame, the window in px (x0, y0, x1, y1) and a card's long side in px there.
 * It may be asynchronous. */
export type Detect = (frame: RgbImage, box: Window, cardPx: number) => Promise<readonly Detection[]> | readonly Detection[];

/** The camera picture's edges as fractions (x0, y0, x1, y1): the long straight edges nearest the middle that every
 * frame shares, or the frame's own edges where there are none. */
export function borders(frames: readonly RgbImage[], edge = 18, keep = 0.9): Window {
  const grey = frames.map((f) => image.resizeGray(image.toGray(f), [W, H], 'box').data);
  // per column (the step from x to x + 1), the fewest rows in any frame where the step is over `edge`; per row likewise
  const colRows = new Int32Array(W - 1).fill(H);
  const rowCols = new Int32Array(H - 1).fill(W);
  for (const g of grey) {
    const cols = new Int32Array(W - 1);
    const rows = new Int32Array(H - 1);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W - 1; x++) if (Math.abs(g[y * W + x + 1]! - g[y * W + x]!) > edge) cols[x] = cols[x]! + 1;
    }
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W; x++) if (Math.abs(g[(y + 1) * W + x]! - g[y * W + x]!) > edge) rows[y] = rows[y]! + 1;
    }
    for (let x = 0; x < W - 1; x++) colRows[x] = Math.min(colRows[x]!, cols[x]!);
    for (let y = 0; y < H - 1; y++) rowCols[y] = Math.min(rowCols[y]!, rows[y]!);
  }
  const share = (n: number, of: number): number => n / of; // the mean of a column's or a row's yes/no
  const left: number[] = [];
  const right: number[] = [];
  const top: number[] = [];
  const bottom: number[] = [];
  for (let x = 0; x < W - 1; x++) {
    if (share(colRows[x]!, H) >= keep && x < 0.45 * W) left.push(x);
    if (share(colRows[x]!, H) >= keep && x > 0.55 * W) right.push(x);
  }
  for (let y = 0; y < H - 1; y++) {
    if (share(rowCols[y]!, W) >= keep && y < 0.45 * H) top.push(y);
    if (share(rowCols[y]!, W) >= keep && y > 0.55 * H) bottom.push(y);
  }
  return [
    left.length ? (Math.max(...left) + 1) / W : 0.0,
    top.length ? (Math.max(...top) + 1) / H : 0.0,
    right.length ? Math.min(...right) / W : 1.0,
    bottom.length ? Math.min(...bottom) / H : 1.0,
  ];
}

/** np.pad(m, pad, mode="edge"). */
function padEdge(m: Mask, pad: number): Mask {
  const w = m.width + 2 * pad;
  const h = m.height + 2 * pad;
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(m.height - 1, Math.max(0, y - pad));
    for (let x = 0; x < w; x++) data[y * w + x] = m.data[sy * m.width + Math.min(m.width - 1, Math.max(0, x - pad))]!;
  }
  return { width: w, height: h, data };
}

/** The median over frames of one value, as np.median: the middle one, or the mean of the two middle ones. */
function medianOf(values: Float64Array, n: number): number {
  for (let i = 1; i < n; i++) {
    const v = values[i]!;
    let j = i - 1;
    for (; j >= 0 && values[j]! > v; j--) values[j + 1] = values[j]!;
    values[j + 1] = v;
  }
  return n % 2 ? values[n >> 1]! : (values[(n >> 1) - 1]! + values[n >> 1]!) / 2;
}

/** The table window as fractions of the frame, the mat's colour and the share of the window it fills; null when no one
 * colour fills the middle of the picture (not a table shot). */
export function tableWindow(frames: readonly RgbImage[], tol = 45): { window: Window; mat: [number, number, number]; share: number } | null {
  const [bx0, by0, bx1, by1] = borders(frames);
  const thumbs = frames.map((f) => image.resize(f, [W, H], 'box').data);
  const img = new Float64Array(W * H * 3); // np.median over the frames of the thumbnails
  const values = new Float64Array(thumbs.length);
  for (let i = 0; i < img.length; i++) {
    for (let k = 0; k < thumbs.length; k++) values[k] = thumbs[k]![i]!;
    img[i] = medianOf(values, thumbs.length);
  }
  const X0 = pyRound(bx0 * W);
  const Y0 = pyRound(by0 * H);
  const X1 = pyRound(bx1 * W);
  const Y1 = pyRound(by1 * H);
  const pw = X1 - X0;
  const ph = Y1 - Y0;
  const my0 = Math.floor(ph / 5);
  const my1 = Math.floor((ph * 4) / 5);
  const mx0 = Math.floor(pw / 5);
  const mx1 = Math.floor((pw * 4) / 5);
  if (pw <= 0 || ph <= 0 || my1 <= my0 || mx1 <= mx0) return null; // no picture to look at
  const at = (y: number, x: number, c: number): number => img[((Y0 + y) * W + X0 + x) * 3 + c]!;
  // the mat: the colour that fills the middle of the picture (mid.astype(np.uint8) cuts the halves)
  const mid = new Uint8Array((my1 - my0) * (mx1 - mx0) * 3);
  for (let y = my0; y < my1; y++) {
    for (let x = mx0; x < mx1; x++) for (let c = 0; c < 3; c++) mid[((y - my0) * (mx1 - mx0) + (x - mx0)) * 3 + c] = Math.floor(at(y, x, c));
  }
  const mat = matColour({ width: mx1 - mx0, height: my1 - my0, data: mid });
  const near = new Uint8Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const d = Math.max(Math.abs(at(y, x, 0) - mat[0]), Math.abs(at(y, x, 1) - mat[1]), Math.abs(at(y, x, 2) - mat[2]));
      near[y * pw + x] = d < tol ? 1 : 0;
    }
  }
  let nearMid = 0;
  for (let y = my0; y < my1; y++) for (let x = mx0; x < mx1; x++) nearMid += near[y * pw + x]!;
  if (nearMid / ((my1 - my0) * (mx1 - mx0)) < 0.3) return null;
  // so the closing does not eat the picture's own edges, pad it; the cards join the mat
  const closed = binaryFillHoles(binaryClosing(padEdge({ width: pw, height: ph, data: near }, 8), 15));
  const grown = new Uint8Array(pw * ph);
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) grown[y * pw + x] = closed.data[(y + 8) * closed.width + x + 8]!;
  const { labels, count: n } = label({ width: pw, height: ph, data: grown });
  const counts = new Int32Array(n + 1);
  for (let y = my0; y < my1; y++) for (let x = mx0; x < mx1; x++) counts[labels[y * pw + x]!] = counts[labels[y * pw + x]!]! + 1;
  counts[0] = 0;
  let biggest = 0; // counts.argmax(): the first of the largest
  for (let k = 1; k <= n; k++) if (counts[k]! > counts[biggest]!) biggest = k;
  if (n === 0 || counts[biggest] === 0) return null;
  const table = (y: number, x: number): number => (labels[y * pw + x] === biggest ? 1 : 0); // the mat and the cards lying on it
  let r0 = ph;
  let r1 = 0;
  let c0 = pw;
  let c1 = 0;
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      if (!table(y, x)) continue;
      if (y < r0) r0 = y;
      if (y + 1 > r1) r1 = y + 1;
      if (x < c0) c0 = x;
      if (x + 1 > c1) c1 = x + 1;
    }
  }
  // Trim the rows the table only touches (a HUD band runs the whole width), and the columns it hardly
  // reaches: cards lie along the mat's edges and partly off it, so a side column is kept at a quarter.
  const rowShare = (r: number): number => {
    let s = 0;
    for (let x = c0; x < c1; x++) s += table(r, x);
    return s / (c1 - c0);
  };
  const colShare = (c: number): number => {
    let s = 0;
    for (let y = r0; y < r1; y++) s += table(y, c);
    return s / (r1 - r0);
  };
  while (r1 - r0 > 10 && rowShare(r0) < 0.5) r0++;
  while (r1 - r0 > 10 && rowShare(r1 - 1) < 0.5) r1--;
  while (c1 - c0 > 10 && colShare(c0) < 0.25) c0++;
  while (c1 - c0 > 10 && colShare(c1 - 1) < 0.25) c1--;
  const margin = pyRound(0.015 * W); // a card on the mat's edge, inside the picture's borders
  c0 = Math.max(0, c0 - margin);
  c1 = Math.min(pw, c1 + margin);
  const x0 = (X0 + c0) / W;
  const y0 = (Y0 + r0) / H;
  const x1 = (X0 + c1) / W;
  const y1 = (Y0 + r1) / H;
  if ((x1 - x0) * (y1 - y0) < 0.2) return null;
  let nearIn = 0;
  for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) nearIn += near[y * pw + x]!;
  const share = nearIn / ((r1 - r0) * (c1 - c0));
  return { window: [pyRound(x0, 3), pyRound(y0, 3), pyRound(x1, 3), pyRound(y1, 3)], mat, share };
}

/** A card's long side at 1080p: of the candidate sizes, the one at which the detector finds the most confident cards
 * whose own size agrees with it; then the median size of those cards. Asynchronous: the detector is. */
export async function cardSize(detect: Detect, frames: readonly RgbImage[], window: Window): Promise<number | null> {
  let best: number | null = null;
  let bestN = 0;
  let bestLongs: number[] = [];
  for (const px of SIZES) {
    const longs: number[] = [];
    for (const f of frames) {
      const { width: w, height: h } = f;
      const box: Window = [window[0] * w, window[1] * h, window[2] * w, window[3] * h];
      for (const d of await detect(f, box, (px * h) / 1080)) {
        if (d.score < 0.6) continue;
        const q = d.quad;
        const side = (Math.max(norm2(q[1]![0] - q[0]![0], q[1]![1] - q[0]![1]), norm2(q[2]![0] - q[1]![0], q[2]![1] - q[1]![1])) * 1080) / h;
        if (Math.abs(side / px - 1) < 0.35) longs.push(side);
      }
    }
    if (longs.length > bestN) {
      best = px;
      bestN = longs.length;
      bestLongs = longs;
    }
  }
  return best !== null && bestN >= 5 ? median(bestLongs) : null;
}

/** f[y0:y1, x0:x1] as numpy slices it: clipped to the picture. */
function slice(f: RgbImage, y0: number, y1: number, x0: number, x1: number): RgbImage {
  const ya = Math.min(f.height, Math.max(0, y0));
  const yb = Math.min(f.height, Math.max(ya, y1));
  const xa = Math.min(f.width, Math.max(0, x0));
  const xb = Math.min(f.width, Math.max(xa, x1));
  const w = xb - xa;
  const data = new Uint8Array(w * (yb - ya) * 3);
  for (let y = ya; y < yb; y++) data.set(f.data.subarray((y * f.width + xa) * 3, (y * f.width + xb) * 3), (y - ya) * w * 3);
  return { width: w, height: yb - ya, data };
}

/** Without the detector: the candidate size at which the bootstrap finder sees the most isolated cards, then their
 * median size. */
export function cardSizeFinder(frames: readonly RgbImage[], window: Window, mat: readonly [number, number, number]): number | null {
  let best: number[] = [];
  for (const px of SIZES) {
    const longs: number[] = [];
    for (const f of frames) {
      const { width: w, height: h } = f;
      const roi = slice(f, pyRound(window[1] * h), pyRound(window[3] * h), pyRound(window[0] * w), pyRound(window[2] * w));
      for (const b of findCards(roi, (px * h) / 1080, { tol: 0.15, mask: notmatMask(roi, mat, 45) })) longs.push((b.long_px * 1080) / h);
    }
    if (longs.length > best.length) best = longs;
  }
  return best.length >= 3 ? median(best) : null;
}

/** A layout for these frames of the table camera, or null when they do not show a table (or no card on it yet). With a
 * detector it is asked for the card size (so this is asynchronous); without, the bootstrap finder is. */
export async function autoLayout(frames: readonly RgbImage[], detect?: Detect): Promise<Layout | null> {
  const found = tableWindow(frames);
  if (found === null) return null;
  const { window, mat, share } = found;
  const px = detect !== undefined ? await cardSize(detect, frames, window) : cardSizeFinder(frames, window, mat);
  if (px === null) return null;
  return makeLayout({ name: 'auto', title: 'this broadcast', table: window, card_long_1080: pyRound(px, 1), mat, mat_share: pyRound(0.8 * share, 2) });
}
