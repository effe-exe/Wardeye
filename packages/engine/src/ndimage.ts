// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The scipy.ndimage operations the change gate and the layout finder use, on boolean pictures (one byte a pixel,
// 0 or 1, rows top to bottom). They give the results of SciPy 1.17.1's scipy/ndimage/_morphology.py (binary_dilation,
// binary_erosion, binary_opening, binary_closing and binary_fill_holes, with the default border value 0) and
// scipy/ndimage/_measurements.py (label, find_objects) on the same input, by other means: the structures used
// here are the default cross and all-ones squares, which dilate and erode in separable passes, and labels are
// numbered as SciPy numbers them, by the first pixel of each component in a scan by rows (checked against SciPy
// in test/table.test.ts). The code is our own; SciPy is BSD-3-Clause.

/** A boolean picture: data[y * width + x] is 0 or 1. */
export interface Mask {
  width: number;
  height: number;
  data: Uint8Array;
}

/** binary_dilation with the default cross structure (4 neighbours), `iterations` times, the outside taken as 0. A few
 * dilations by the cross are one dilation by the diamond of that radius: every pixel within that many steps (rows plus
 * columns) of a set one, which two sweeps over the picture give (the city-block distance transform). */
export function binaryDilation(m: Mask, iterations = 1): Mask {
  const { width: w, height: h, data } = m;
  const out = new Uint8Array(w * h);
  if (iterations < 2 || iterations > 250) {
    let src = data;
    for (let it = 0; it < iterations; it++) {
      const next = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          const i = row + x;
          next[i] = src[i]! | (x > 0 ? src[i - 1]! : 0) | (x < w - 1 ? src[i + 1]! : 0) | (y > 0 ? src[i - w]! : 0) | (y < h - 1 ? src[i + w]! : 0);
        }
      }
      src = next;
    }
    out.set(src);
    return { width: w, height: h, data: out };
  }
  const cap = iterations + 1; // further than the radius: no need to know how far
  const d = out;
  for (let i = 0; i < d.length; i++) d[i] = data[i] ? 0 : cap;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = d[i]!;
      if (v === 0) continue;
      if (x > 0 && d[i - 1]! + 1 < v) v = d[i - 1]! + 1;
      if (y > 0 && d[i - w]! + 1 < v) v = d[i - w]! + 1;
      d[i] = v;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = d[i]!;
      if (v === 0) continue;
      if (x < w - 1 && d[i + 1]! + 1 < v) v = d[i + 1]! + 1;
      if (y < h - 1 && d[i + w]! + 1 < v) v = d[i + w]! + 1;
      d[i] = v;
    }
  }
  for (let i = 0; i < d.length; i++) d[i] = d[i]! <= iterations ? 1 : 0;
  return { width: w, height: h, data: d };
}

/** One separable pass of an all-ones structure of odd `size`: dilation sets a pixel when any pixel of its window is
 * set; erosion when every pixel of the window is set and the window lies inside the picture (the outside is 0). */
function boxPass(src: Uint8Array, w: number, h: number, size: number, erode: boolean, vertical: boolean): Uint8Array {
  const r = (size - 1) >> 1;
  const out = new Uint8Array(w * h);
  const n = vertical ? h : w; // the line's length
  const lines = vertical ? w : h;
  const step = vertical ? w : 1; // between pixels along a line
  const prefix = new Int32Array(n + 1);
  for (let line = 0; line < lines; line++) {
    const base = vertical ? line : line * w;
    for (let k = 0; k < n; k++) prefix[k + 1] = prefix[k]! + src[base + k * step]!;
    for (let k = 0; k < n; k++) {
      const lo = k - r;
      const hi = k + r; // inclusive
      let v: number;
      if (erode) v = lo >= 0 && hi <= n - 1 && prefix[hi + 1]! - prefix[lo]! === size ? 1 : 0;
      else v = prefix[Math.min(n - 1, hi) + 1]! - prefix[Math.max(0, lo)]! > 0 ? 1 : 0;
      out[base + k * step] = v;
    }
  }
  return out;
}

function boxMorph(m: Mask, size: number, erode: boolean): Mask {
  if (size < 1 || size % 2 === 0) throw new RangeError('ndimage: the structure size must be odd');
  const { width: w, height: h } = m;
  return { width: w, height: h, data: boxPass(boxPass(m.data, w, h, size, erode, false), w, h, size, erode, true) };
}

/** binary_dilation / binary_erosion with structure=np.ones((size, size)) (size odd), the outside taken as 0. */
export const boxDilation = (m: Mask, size = 3): Mask => boxMorph(m, size, false);
export const boxErosion = (m: Mask, size = 3): Mask => boxMorph(m, size, true);

/** binary_opening with structure=np.ones((size, size)): an erosion, then a dilation. */
export function binaryOpening(m: Mask, size = 3): Mask {
  return boxDilation(boxErosion(m, size), size);
}

/** binary_closing with structure=np.ones((size, size)): a dilation, then an erosion (the outside is 0 for both, so
 * the closing eats the picture's own edges: pad it first). */
export function binaryClosing(m: Mask, size = 3): Mask {
  return boxErosion(boxDilation(m, size), size);
}

/** binary_fill_holes with the default cross structure: 1 for everything but the background that reaches the
 * picture's edge through 4-connected background. */
export function binaryFillHoles(m: Mask): Mask {
  const { width: w, height: h, data } = m;
  const outside = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const seed = (i: number): void => {
    if (data[i] === 0 && outside[i] === 0) {
      outside[i] = 1;
      stack[sp++] = i;
    }
  };
  for (let x = 0; x < w; x++) {
    seed(x);
    seed((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    seed(y * w);
    seed(y * w + w - 1);
  }
  while (sp > 0) {
    const i = stack[--sp]!;
    const x = i % w;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (i >= w) seed(i - w);
    if (i < (h - 1) * w) seed(i + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = outside[i] ? 0 : 1;
  return { width: w, height: h, data: out };
}

/** label with the default cross structure (4 neighbours): labels 1..count, 0 for the background, numbered in the
 * order the components' first pixels come in a scan by rows, as SciPy numbers them. */
export function label(m: Mask): { labels: Int32Array; count: number } {
  const { width: w, height: h, data } = m;
  const labels = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  let count = 0;
  for (let start = 0; start < w * h; start++) {
    if (data[start] === 0 || labels[start] !== 0) continue;
    count++;
    labels[start] = count;
    let sp = 0;
    stack[sp++] = start;
    while (sp > 0) {
      const i = stack[--sp]!;
      const x = i % w;
      if (x > 0 && data[i - 1] !== 0 && labels[i - 1] === 0) {
        labels[i - 1] = count;
        stack[sp++] = i - 1;
      }
      if (x < w - 1 && data[i + 1] !== 0 && labels[i + 1] === 0) {
        labels[i + 1] = count;
        stack[sp++] = i + 1;
      }
      if (i >= w && data[i - w] !== 0 && labels[i - w] === 0) {
        labels[i - w] = count;
        stack[sp++] = i - w;
      }
      if (i < (h - 1) * w && data[i + w] !== 0 && labels[i + w] === 0) {
        labels[i + w] = count;
        stack[sp++] = i + w;
      }
    }
  }
  return { labels, count };
}

/** find_objects: for each label 1..count (in that order), its bounding box as [y0, y1, x0, x1] with the stops
 * exclusive, like the slices SciPy returns. */
export function findObjects(labels: Int32Array, width: number, height: number, count: number): [number, number, number, number][] {
  const boxes: [number, number, number, number][] = Array.from({ length: count }, () => [height, 0, width, 0]);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const l = labels[y * width + x]!;
      if (l === 0) continue;
      const b = boxes[l - 1]!;
      if (y < b[0]) b[0] = y;
      if (y + 1 > b[1]) b[1] = y + 1;
      if (x < b[2]) b[2] = x;
      if (x + 1 > b[3]) b[3] = x + 1;
    }
  }
  return boxes;
}
