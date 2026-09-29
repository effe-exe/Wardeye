// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Quads and tiles for the card detector: a port of ml/rifteye_ml/detect/geometry.py, function for function and
// in the same order, plus the float rules of numpy and CPython that the detector's ports need beyond pynum.ts.
//
// The results are Python's to the last bit wherever the arithmetic allows. numpy's np.dot is OpenBLAS's ddot
// here: a strided vector (a column of an n x 2 array) is summed in two interleaved halves, four products at a
// time, and its tail with fused multiply-adds; np.linalg.norm of two numbers is one fused multiply-add. Both are
// reproduced, the fused multiply-add emulated exactly. math.dist is CPython's own algorithm. Only atan2, sin
// and cos are the browser's, which can differ from the C library's in the last bit; canonicalQuad uses them
// only to put the corners in order.

import { pyMod } from './pynum';

/** A point: x, y. */
export type Point = [number, number];

/** Corners as [x, y] pairs, or flat as x0, y0, x1, y1, ...: what numpy's reshape(-1, 2) takes. */
export type Corners = readonly (readonly number[])[] | ArrayLike<number>;

/** np.asarray(q, np.float64).reshape(-1, 2). */
export function points(q: Corners): Point[] {
  const flat: number[] = [];
  for (let i = 0; i < q.length; i++) {
    const v = q[i]!;
    if (typeof v === 'number') flat.push(v);
    else for (const x of v) flat.push(x);
  }
  if (flat.length % 2) throw new Error(`${flat.length} numbers are not x, y pairs`);
  const out: Point[] = [];
  for (let i = 0; i < flat.length; i += 2) out.push([flat[i]!, flat[i + 1]!]);
  return out;
}

export const CORNERS = ['top_left', 'top_right', 'bottom_right', 'bottom_left'] as const;

/** The four corners of a card, the one up and left of its centre first, then clockwise on screen. */
export function canonicalQuad(quad: Corners): Point[] {
  const q = points(quad);
  if (q.length !== 4) throw new Error(`a quad has 4 corners, not ${q.length}`);
  const [mx, my] = mean(q);
  const ang = q.map(([x, y]) => Math.atan2(y - my, x - mx)); // y points down, so clockwise on screen is increasing angle
  // np.angle(np.exp(1j * t)): t wrapped into (-pi, pi]
  const off = ang.map((a) => {
    const t = a + (3 * Math.PI) / 4;
    return Math.abs(Math.atan2(Math.sin(t), Math.cos(t)));
  });
  const first = argmin(off);
  const key = ang.map((a) => pyMod(a - ang[first]!, 2 * Math.PI));
  return argsortStable(key).map((i) => q[i]!);
}

export function polygonArea(p: Corners): number {
  return Math.abs(signed(points(p))) / 2;
}

function signed(p: readonly Point[]): number {
  const x = p.map((v) => v[0]);
  const y = p.map((v) => v[1]);
  return blasDot(x, roll(y)) - blasDot(y, roll(x));
}

/** Sutherland-Hodgman: the part of a polygon inside a convex polygon of positive orientation. */
function clip(subject: readonly Point[], clipper: readonly Point[]): Point[] {
  let out: Point[] = [...subject];
  for (let i = 0; i < clipper.length; i++) {
    const a = clipper[i]!;
    const b = clipper[(i + 1) % clipper.length]!;
    const inp = out;
    out = [];
    if (!inp.length) break;
    const side = (p: Point) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    for (let j = 0; j < inp.length; j++) {
      const p = inp[j]!;
      const q = inp[(j + 1) % inp.length]!;
      const sp = side(p);
      const sq = side(q);
      if (sp >= 0) out.push(p);
      if (sp >= 0 !== sq >= 0) {
        const t = sp / (sp - sq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

/** The area two convex quads share (any corner order that goes round the quad). */
export function overlapArea(quadA: Corners, quadB: Corners): number {
  let a = points(quadA);
  let b = points(quadB);
  if (signed(a) < 0) a = a.reverse();
  if (signed(b) < 0) b = b.reverse();
  const xa = a.map((p) => p[0]);
  const ya = a.map((p) => p[1]);
  const xb = b.map((p) => p[0]);
  const yb = b.map((p) => p[1]);
  if (Math.max(...xa) <= Math.min(...xb) || Math.max(...xb) <= Math.min(...xa) || Math.max(...ya) <= Math.min(...yb) || Math.max(...yb) <= Math.min(...ya)) {
    return 0;
  }
  const inter = clip(a, b);
  return inter.length >= 3 ? polygonArea(inter) : 0;
}

/** Intersection over union of two convex quads (any corner order that goes round the quad). */
export function quadIou(a: Corners, b: Corners): number {
  const ia = overlapArea(a, b);
  if (ia <= 0) return 0;
  const union = polygonArea(a) + polygonArea(b) - ia;
  return union > 0 ? ia / union : 0;
}

/** Left (or top) edges of tiles of side `tile` that cover `size` pixels with at least `overlap` shared. */
export function tileOrigins(size: number, tile: number, overlap: number): number[] {
  if (size <= tile) return [0];
  const n = Math.ceil((size - tile) / (tile * (1 - overlap)) - 1e-9) + 1;
  return linspace(0, size - tile, n).map((v) => roundHalfEven(v));
}

// ---- numpy's and CPython's float arithmetic, as the ports above and in detector.ts need it

/** np.linspace(start, stop, n): start + i * step, and stop itself last. */
export function linspace(start: number, stop: number, n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [start];
  const step = (stop - start) / (n - 1);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(i * step + start);
  out[n - 1] = stop;
  return out;
}

/** Python's round(x) and numpy's rint: the nearest whole number, and on a tie the even one. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x); // halves go up
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** numpy's round(x, decimals) of a float64 (np.round, and Python's round() on a numpy scalar): rint(x * 10**d) / 10**d.
 * Not CPython's round() of a float, which rounds the exact decimal value: round(np.float64(4.35), 1) is 4.4,
 * round(4.35, 1) is 4.3. */
export function npRound(x: number, decimals: number): number {
  const f = 10 ** decimals;
  return roundHalfEven(x * f) / f;
}

/** The arithmetic mean of points, coordinate by coordinate, summed in order as numpy's mean(axis=0) does. */
export function mean(p: readonly Point[]): Point {
  let sx = 0;
  let sy = 0;
  for (const [x, y] of p) {
    sx += x;
    sy += y;
  }
  return [sx / p.length, sy / p.length];
}

/** np.dot of a strided float64 vector (a column of an n x 2 array) with another: OpenBLAS's ddot sums four
 * products at a time into two partial sums, (m0 + m2) and (m1 + m3), and the tail into the first with fused
 * multiply-adds. */
export function blasDot(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = x.length;
  let t1 = 0;
  let t2 = 0;
  let i = 0;
  for (; i + 4 <= n; i += 4) {
    t1 += x[i]! * y[i]! + x[i + 2]! * y[i + 2]!;
    t2 += x[i + 1]! * y[i + 1]! + x[i + 3]! * y[i + 3]!;
  }
  for (; i < n; i++) t1 = fma(x[i]!, y[i]!, t1);
  return t1 + t2;
}

/** np.linalg.norm of a vector of two: the square root of one fused multiply-add. */
export function norm2(x: number, y: number): number {
  return Math.sqrt(fma(y, y, x * x));
}

function roll<T>(v: readonly T[]): T[] {
  return [...v.slice(1), ...v.slice(0, 1)];
}

/** The index of the first smallest value, as np.argmin. */
export function argmin(v: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! < v[best]!) best = i;
  return best;
}

/** np.argsort(v, kind="stable"). */
export function argsortStable(v: readonly number[]): number[] {
  return v.map((_, i) => i).sort((i, j) => (v[i]! < v[j]! ? -1 : v[i]! > v[j]! ? 1 : i - j));
}

// fused multiply-add, emulated exactly: Boldo and Melquiond, "Emulation of a FMA and correctly-rounded sums:
// proved algorithms using rounding to odd" (IEEE Trans. Computers, 2008), algorithm 5.4
const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
const LO = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1 ? 0 : 1;
const HI = 1 - LO;
const SPLIT = 134217729; // 2 ** 27 + 1, Veltkamp's constant

/** a * b + c rounded once, as a hardware fused multiply-add (and C's fma) gives it. */
export function fma(a: number, b: number, c: number): number {
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a === 0 || b === 0) return a * b + c;
  // a * b = uh + ul exactly (Dekker), c + uh = th + tl exactly (Knuth)
  const uh = a * b;
  let t = SPLIT * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = SPLIT * b;
  const bh = t - (t - b);
  const bl = b - bh;
  const ul = ah * bh - uh + ah * bl + al * bh + al * bl;
  const th = c + uh;
  const bb = th - c;
  const tl = c - (th - bb) + (uh - bb);
  return th + roundToOdd(tl, ul);
}

/** a + b rounded to odd: exact when it can be, else the neighbour of the exact sum whose last bit is 1. */
function roundToOdd(a: number, b: number): number {
  const s = a + b;
  const bb = s - a;
  const e = a - (s - bb) + (b - bb);
  if (e === 0) return s;
  f64[0] = s;
  if (u32[LO]! & 1) return s;
  // one step away from s towards the exact sum: its magnitude up or down one unit in the last place
  const up = e > 0 === s > 0;
  let lo = u32[LO]!;
  let hi = u32[HI]!;
  if (up) {
    lo = (lo + 1) >>> 0;
    if (lo === 0) hi = (hi + 1) >>> 0;
  } else {
    if (lo === 0) hi = (hi - 1) >>> 0;
    lo = (lo - 1) >>> 0;
  }
  u32[LO] = lo;
  u32[HI] = hi;
  return f64[0]!;
}

/** math.dist(p, q) for two points, as CPython computes it (vector_norm in Modules/mathmodule.c: lossless
 * scaling, Dekker squaring, compensated sums and a correction of the square root), so the same bits come out. */
export function dist(p: readonly [number, number] | readonly number[], q: readonly [number, number] | readonly number[]): number {
  const dx = Math.abs(p[0]! - q[0]!);
  const dy = Math.abs(p[1]! - q[1]!);
  if (Number.isNaN(dx) || Number.isNaN(dy)) return dx === Infinity || dy === Infinity ? Infinity : NaN;
  return vectorNorm([dx, dy], Math.max(dx, dy));
}

const DBL_MIN = 2.2250738585072014e-308;

function vectorNorm(vec: number[], max: number): number {
  if (max === Infinity) return max;
  if (max === 0 || vec.length <= 1) return max;
  const maxE = frexpExponent(max);
  if (maxE < -1023) return DBL_MIN * vectorNorm(vec.map((v) => v / DBL_MIN), max / DBL_MIN);
  const scale = pow2(-maxE);
  let csum = 1;
  let frac1 = 0;
  let frac2 = 0;
  for (let x of vec) {
    x *= scale; // lossless
    const [ph, pl] = dlMul(x, x);
    const sh = csum + ph; // lossless: |csum| >= |ph|
    frac2 += csum - sh + ph;
    csum = sh;
    frac1 += pl;
  }
  let h = Math.sqrt(csum - 1 + (frac1 + frac2));
  const [ph, pl] = dlMul(-h, h);
  const sh = csum + ph;
  frac2 += csum - sh + ph;
  csum = sh;
  frac1 += pl;
  const x = csum - 1 + (frac1 + frac2);
  h += x / (2 * h); // the differential correction
  return h / scale;
}

/** x * y as hi + lo exactly, by Dekker's splitting (no fused multiply-add, as CPython 3.11 is built). */
function dlMul(x: number, y: number): [number, number] {
  let t = x * SPLIT;
  const xh = t - (t - x);
  const xl = x - xh;
  t = y * SPLIT;
  const yh = t - (t - y);
  const yl = y - yh;
  const p = xh * yh;
  const q = xh * yl + xl * yh;
  const z = p + q;
  const zz = p - z + q + xl * yl;
  return [z, zz];
}

/** The exponent frexp gives: x = m * 2 ** e with 0.5 <= |m| < 1. */
function frexpExponent(x: number): number {
  f64[0] = x;
  const biased = (u32[HI]! >>> 20) & 0x7ff;
  if (biased === 0) {
    // subnormal: scale into the normal range first
    f64[0] = x * 2 ** 64;
    return ((u32[HI]! >>> 20) & 0x7ff) - 1022 - 64;
  }
  return biased - 1022;
}

/** 2 ** k, exactly, for a whole k (built from its bits in the normal range). */
function pow2(k: number): number {
  if (k < -1022 || k > 1023) return 2 ** k;
  u32[LO] = 0;
  u32[HI] = (k + 1023) << 20;
  return f64[0]!;
}
