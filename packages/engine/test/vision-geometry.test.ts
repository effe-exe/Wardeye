// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// geometry.ts against detect/geometry.py and the float rules it reproduces, on the seeded synthetic vectors of
// test/gen/vision_geometry.py. The areas, overlaps, tiles and rounding are asked for bit for bit.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  blasDot,
  canonicalQuad,
  dist,
  fma,
  linspace,
  norm2,
  npRound,
  overlapArea,
  points,
  polygonArea,
  quadIou,
  roundHalfEven,
  tileOrigins,
} from '../src/geometry';
import { pyRound } from '../src/pynum';

interface Vectors {
  canonical_quad: { q: number[][]; out: number[][] }[];
  polygon_area: { p: number[][]; out: number[] }[] & { p: number[][]; out: number }[];
  overlap: { a: number[][]; b: number[]; area: number; iou: number }[];
  tile_origins: { size: number; tile: number; overlap: number; out: number[] }[];
  norm2: { v: [number, number]; out: number }[];
  dist: { p: [number, number]; q: [number, number]; out: number }[];
  np_round1: { x: number; out: number }[];
  py_round1: { x: number; out: number }[];
  fma: { a: number; b: number; c: number; out: number }[];
}

const V = JSON.parse(readFileSync(new URL('./vectors/vision-geometry.json', import.meta.url), 'utf8')) as Vectors;

describe('canonicalQuad', () => {
  it('orders the corners as Python does: up-left first, then clockwise on screen', () => {
    for (const c of V.canonical_quad) expect(canonicalQuad(c.q)).toEqual(c.out);
  });

  it('takes flat corners too', () => {
    const c = V.canonical_quad[0]!;
    expect(canonicalQuad(c.q.flat())).toEqual(c.out);
  });

  it('needs four corners', () => {
    expect(() => canonicalQuad([[0, 0], [1, 0], [1, 1]])).toThrow('4 corners');
  });
});

describe('polygonArea', () => {
  it('is Python\'s to the bit, for 3 to 8 corners', () => {
    for (const c of V.polygon_area as unknown as { p: number[][]; out: number }[]) expect(polygonArea(c.p)).toBe(c.out);
  });
});

describe('overlapArea and quadIou', () => {
  it('are Python\'s to the bit, in either orientation, flat or as pairs', () => {
    for (const c of V.overlap) {
      expect(overlapArea(c.a, c.b)).toBe(c.area);
      expect(quadIou(c.a, c.b)).toBe(c.iou);
    }
  });

  it('are symmetric enough to be used either way round', () => {
    const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];
    expect(overlapArea(sq, sq)).toBe(100);
    expect(quadIou(sq, sq)).toBe(1);
    expect(overlapArea(sq, sq.map(([x, y]) => [x! + 10, y!]))).toBe(0); // touching edges share nothing
    expect(quadIou(sq, sq.map(([x, y]) => [x! + 5, y!]))).toBeCloseTo(50 / 150, 15);
  });
});

describe('tileOrigins', () => {
  it('is Python\'s, halves rounded to even', () => {
    for (const c of V.tile_origins) expect(tileOrigins(c.size, c.tile, c.overlap)).toEqual(c.out);
  });

  it('is one tile for a window that fits', () => {
    expect(tileOrigins(576, 576, 0.2)).toEqual([0]);
    expect(tileOrigins(100, 576, 0.2)).toEqual([0]);
  });

  it('rounds 230.5 to 230', () => {
    // 1037 - 576 = 461 over two steps
    expect(tileOrigins(1037, 576, 0.2)).toEqual([0, 230, 461]);
  });
});

describe('numpy\'s and CPython\'s float rules', () => {
  it('np.linalg.norm of two numbers: one fused multiply-add', () => {
    for (const c of V.norm2) expect(norm2(c.v[0], c.v[1])).toBe(c.out);
  });

  it('math.dist: CPython\'s algorithm, bit for bit', () => {
    for (const c of V.dist) expect(dist(c.p, c.q)).toBe(c.out);
    expect(dist([0, 0], [3, 4])).toBe(5);
    expect(dist([1, 1], [1, 1])).toBe(0);
    expect(dist([0, 0], [NaN, 1])).toBeNaN();
    expect(dist([0, 0], [Infinity, NaN])).toBe(Infinity);
  });

  it('round() on a numpy scalar and on a float', () => {
    for (const c of V.np_round1) expect(npRound(c.x, 1)).toBe(c.out);
    for (const c of V.py_round1) expect(pyRound(c.x, 1)).toBe(c.out);
    expect(npRound(4.35, 1)).toBe(4.4);
    expect(pyRound(4.35, 1)).toBe(4.3);
  });

  it('round half to even', () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4999999999999996, 3.5000000000000004].map(roundHalfEven)).toEqual([0, 2, 2, -0, -2, -2, 2, 4]);
  });

  it('a fused multiply-add rounds once', () => {
    for (const c of V.fma) expect(fma(c.a, c.b, c.c)).toBe(c.out);
    // (1 + 2^-30)^2 - 1 needs the exact square: 2^-29 + 2^-60
    const a = 1 + 2 ** -30;
    expect(fma(a, a, -1)).toBe(2 ** -29 + 2 ** -60);
    expect(a * a - 1).toBe(2 ** -29); // the unfused product loses the 2^-60
    expect(fma(0, 5, 3)).toBe(3);
    expect(fma(2, 3, Infinity)).toBe(Infinity);
  });

  it('OpenBLAS\'s dot order', () => {
    // 1e16 + 1 - 1e16 + 1: in pairs (m0 + m2) + (m1 + m3) = (1e16 - 1e16) + (1 + 1) = 2
    expect(blasDot([1e16, 1, -1e16, 1], [1, 1, 1, 1])).toBe(2);
    expect(blasDot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it('linspace puts the stop itself last', () => {
    expect(linspace(0, 1, 4)).toEqual([0, 1 / 3, 2 / 3, 1]);
    expect(linspace(0, 5, 1)).toEqual([0]);
  });

  it('points reshapes as numpy does', () => {
    expect(points([1, 2, 3, 4])).toEqual([[1, 2], [3, 4]]);
    expect(points([[1, 2], [3, 4]])).toEqual([[1, 2], [3, 4]]);
    expect(points(new Float32Array([0.5, 1.5]))).toEqual([[0.5, 1.5]]);
    expect(() => points([1, 2, 3])).toThrow('pairs');
  });
});
