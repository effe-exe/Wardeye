// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The table modules against their Python originals on the seeded synthetic vectors of test/gen/table.py: SciPy's
// linear_sum_assignment, the scipy.ndimage operations, matcrops, the change gate, the gallery pyramid and search, the
// layouts and the layout finder. Pictures are never stored: the vectors hold specs that test/table-helpers.ts renders
// from the same xorshift as the generator, and the hashes of what Python made of them. Integers, booleans and
// pictures are asked for bit for bit; so are the floats, except where a test says otherwise.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { autoLayout, borders, cardSize, cardSizeFinder, tableWindow } from '../src/autolayout';
import { ChangeGate, gateSettings, skin, viewHeight, viewToFrame, type GateSettings } from '../src/changegate';
import * as image from '../src/image';
import { box, cardPx, LAYOUTS, makeLayout, side, sides } from '../src/layouts';
import { linearSumAssignment } from '../src/lsap';
import {
  borderMask,
  detail,
  FACE_DOWN_DETAIL,
  findCards,
  matColour,
  minAreaRect,
  notmatMask,
  stepMean,
} from '../src/matcrops';
import { binaryClosing, binaryDilation, binaryFillHoles, binaryOpening, findObjects, label, type Mask } from '../src/ndimage';
import { median, pyFloorDiv, pyMod, pyRound } from '../src/pynum';
import {
  accuracy,
  argsortDescending,
  atLongSide,
  band,
  bestSimilarities,
  fmaF32,
  Pyramid,
  rankedLabels,
  search,
  similarities,
  topk,
  type Embeddings,
} from '../src/retrieval';
import type { Encoder, Layout, RgbImage } from '../src/types';
import { render, sha256, XorShift32, type PictureSpec } from './table-helpers';
import { replayDetector } from './table-replay';

// --- the vectors -------------------------------------------------------------------------------------------------

type Spec = PictureSpec;
interface Sha {
  sha: string;
  count?: number;
}
interface GateSequence {
  seq: {
    name: string;
    w: number;
    h: number;
    noise: number;
    fps: number;
    seed: number;
    settings: { width?: number; mat_rgb?: [number, number, number] | null; ignore?: [number, number, number, number][]; settle_s?: number };
    steps: { n: number; bg?: [number, number, number]; rects?: (number | number[] | string)[][] }[];
  };
  expected: {
    frames: number;
    per_frame: { events: unknown[]; background: string | null }[];
    still: string | null;
    last_same: string | null;
    startup_hand: string | null;
    off_table: number;
    mat: number[];
  };
}
interface AutoCase {
  name: string;
  specs: Spec[];
  borders: number[];
  table_window: { window: number[]; mat: number[]; share: number } | null;
  detector_calls: { frame: number; box: number[]; px: number; out: { score: number; quad: number[]; cls: string }[] }[];
  card_size: number | null;
  auto_layout: Layout | null;
  auto_layout_finder: Layout | null;
  card_size_finder: number | null;
}
interface Vectors {
  versions: Record<string, string>;
  xorshift: { seed: number; first: number[] };
  pynum: { round: [number, number, number][]; mod: [number, number, number, number][]; median: [number[], number][] };
  layouts: {
    presets: Record<string, Layout>;
    layouts: Layout[];
    sides: [string, string][];
    cases: { layout: number; w: number; h: number; box: number[]; px: number; points: [number, number, string][] }[];
  };
  lsap: {
    cases: { rows: number; cols: number; cost: number[]; maximize: boolean; a: number[]; b: number[] }[];
    special: { name: string; cost: (number | string)[][]; maximize: boolean; a?: number[]; b?: number[]; error?: string }[];
    empty: { rows: number; cols: number; a: number[]; b: number[] }[];
    big: { seed: number; rows: number; cols: number; kind: 'unit' | 'int5' | 'masked'; a: number[]; b: number[] }[];
  };
  ndimage: {
    cases: {
      spec: { kind: 'rects' | 'speckle'; n?: number; permille?: number; seed: number; w: number; h: number };
      mask: string;
      sha: Record<string, string>;
      labels: string;
      count: number;
      objects: number[][];
    }[];
  };
  matcrops: {
    specs: (Spec & { name: string })[];
    mat_colour: { spec: string; step: number; mat: number[] }[];
    notmat: (Sha & { spec: string; mat: number[]; tol: number })[];
    border: (Sha & { spec: string; args: number[] })[];
    detail: { w: number; h: number; seed: number; layers: Spec['layers']; detail: number }[];
    face_down: number;
  };
  fma32: [number, number, number, number][];
  similarities: { seed: number; queries: number; dim: number; gallery: number; sha: string; best: string }[];
  pyramid: { long: number[]; cases: { scales: number[]; level: number[] }[] };
  retrieval: {
    at_long_side: (Sha & { spec: Spec; side: number; size: number[] })[];
    band: (Sha & { spec: Spec; view: string; size: number[] })[];
    bad_views: { view: string; error: string | null }[];
    topk: { seed: number; queries: number; gallery: number; dim: number; k: number; idx: number[][]; scores: number[][] }[];
    search: {
      art: Spec[];
      queries: { art: number; side: number; angle: number }[];
      scales: number[];
      pyramid: SearchResult[];
      plain: SearchResult[];
    };
    ranked: { labels: string[]; idx: number[][]; ranked: string[][]; truth: string[]; accuracy: { top1: number; top5: number; n: number } };
  };
  changegate: { sequences: GateSequence[]; skin: (Sha & { spec?: Spec; cube?: boolean })[]; defaults: Record<string, unknown> };
  autolayout: { cases: AutoCase[] };
  views: { table: number[]; vw: number; vh: number; frame: number[]; box: number[]; frame_box: number[] }[];
  findcards: {
    cases: {
      spec: Spec;
      mat: number[];
      runs: {
        name: string;
        long_px: number;
        mask_tol: number | null;
        opts: { tol?: number; min_fill?: number };
        found: { centre: number[]; long_px: number; short_px: number; angle_deg: number; fill: number }[];
      }[];
    }[];
    min_area_rect: { points: [number, number][]; centre: number[]; long_px: number; short_px: number; angle_deg: number; ties: number }[];
  };
}
interface SearchResult {
  k: number;
  rotation_invariant: boolean;
  idx: number[][];
  scores: number[][];
  rot: number[][];
}

const V = JSON.parse(readFileSync(new URL('./vectors/table.json', import.meta.url), 'utf8')) as Vectors;

/** The first 16 hex digits of the SHA-256, as the vectors keep them. */
const sha16 = (a: Uint8Array | Int32Array | Float32Array | Float64Array): string => sha256(a).slice(0, 16);

/** The values of a numeric array as a plain array, for a readable failure. */
const arr = (a: ArrayLike<number>): number[] => Array.from(a);

// --- numbers -----------------------------------------------------------------------------------------------------

describe('the generator', () => {
  it('is the same xorshift on both sides', () => {
    const r = new XorShift32(V.xorshift.seed);
    expect(V.xorshift.first.map(() => r.next())).toEqual(V.xorshift.first);
  });
});

describe('Python round(), %, // and np.median', () => {
  it('rounds ties to even, and decimals from the double\'s exact value', () => {
    for (const [x, nd, want] of V.pynum.round) expect(pyRound(x, nd), `round(${x}, ${nd})`).toBe(want);
  });

  it('takes the sign of the divisor', () => {
    for (const [a, b, mod, div] of V.pynum.mod) {
      expect(pyMod(a, b), `${a} % ${b}`).toBe(mod);
      expect(pyFloorDiv(a, b), `${a} // ${b}`).toBe(div);
    }
  });

  it('takes the mean of the two middle values when the count is even', () => {
    for (const [v, want] of V.pynum.median) expect(median(v)).toBe(want);
    expect(median([])).toBeNaN();
  });
});

// --- layouts -----------------------------------------------------------------------------------------------------

describe('layouts', () => {
  it('has the presets of live/layouts.py', () => {
    expect(LAYOUTS).toEqual(V.layouts.presets);
  });

  it('fills the dataclass defaults', () => {
    expect(makeLayout({ name: 'x', title: 'y', table: [0, 0, 1, 1], card_long_1080: 100 })).toEqual({
      name: 'x',
      title: 'y',
      table: [0, 0, 1, 1],
      card_long_1080: 100,
      split: 'vertical',
      mask: 'notmat',
      mat_tol: 45,
      mat: null,
      mat_share: 0.6,
    });
  });

  it('gives the table window, a card\'s size and the side of a point as Python does', () => {
    for (const c of V.layouts.cases) {
      const l = V.layouts.layouts[c.layout]!;
      expect(box(l, c.w, c.h)).toEqual(c.box);
      expect(cardPx(l, c.h)).toBe(c.px);
      for (const [x, y, s] of c.points) expect(side(l, x, y, c.w, c.h), `${l.name} ${c.w}x${c.h} (${x}, ${y})`).toBe(s);
    }
    V.layouts.layouts.forEach((l, i) => expect(sides(l)).toEqual(V.layouts.sides[i]));
  });

  it('freezes the presets', () => {
    expect(Object.isFrozen(LAYOUTS['la-rq'])).toBe(true);
    expect(Object.isFrozen(LAYOUTS['la-rq'].table)).toBe(true);
  });
});

// --- linear_sum_assignment ---------------------------------------------------------------------------------------

describe('linearSumAssignment', () => {
  it('is SciPy\'s on random cost matrices: square and rectangular, whole and fractional, ties, the tracker\'s masked costs', () => {
    expect(V.lsap.cases.length).toBeGreaterThanOrEqual(300);
    for (const c of V.lsap.cases) {
      const [a, b] = linearSumAssignment({ rows: c.rows, cols: c.cols, data: c.cost }, c.maximize);
      expect([a, b], JSON.stringify(c)).toEqual([c.a, c.b]);
    }
  });

  it('takes a list of rows as well', () => {
    const c = V.lsap.cases.find((x) => x.rows > 1 && x.cols > 1)!;
    const rows = Array.from({ length: c.rows }, (_, i) => c.cost.slice(i * c.cols, (i + 1) * c.cols));
    expect(linearSumAssignment(rows, c.maximize)).toEqual([c.a, c.b]);
  });

  it('says what SciPy says of infinite, undefined and empty matrices', () => {
    const num = (v: number | string): number => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v === 'nan' ? NaN : (v as number));
    for (const s of V.lsap.special) {
      const cost = s.cost.map((r) => r.map(num));
      if (s.error) expect(() => linearSumAssignment(cost, s.maximize), s.name).toThrow(s.error);
      else expect(linearSumAssignment(cost, s.maximize), s.name).toEqual([s.a, s.b]);
    }
    for (const e of V.lsap.empty) expect(linearSumAssignment({ rows: e.rows, cols: e.cols, data: [] })).toEqual([e.a, e.b]);
  });

  it('is SciPy\'s on large matrices too (100 x 100, 120 x 150, 150 x 80, a masked one), and quick', () => {
    for (const c of V.lsap.big) {
      const rng = new XorShift32(c.seed);
      const data = Array.from({ length: c.rows * c.cols }, () => (c.kind === 'unit' ? rng.unit() * 100 : c.kind === 'int5' ? rng.below(5) : rng.below(4) ? 1e6 : rng.unit()));
      const t0 = performance.now();
      const [a, b] = linearSumAssignment({ rows: c.rows, cols: c.cols, data });
      expect(performance.now() - t0).toBeLessThan(2000);
      expect([a, b], `${c.rows} x ${c.cols} ${c.kind}`).toEqual([c.a, c.b]);
    }
  });

  it('gives the identity for a constant matrix, and does not change its input', () => {
    const cost = [
      [5, 5, 5],
      [5, 5, 5],
      [5, 5, 5],
    ];
    expect(linearSumAssignment(cost)).toEqual([
      [0, 1, 2],
      [0, 1, 2],
    ]);
    expect(cost[0]).toEqual([5, 5, 5]);
  });
});

// --- scipy.ndimage -----------------------------------------------------------------------------------------------

/** The masks test/gen/table.py draws from a seed. */
function maskFrom(spec: Vectors['ndimage']['cases'][number]['spec']): Mask {
  const rng = new XorShift32(spec.seed);
  const { w, h } = spec;
  const data = new Uint8Array(w * h);
  if (spec.kind === 'rects') {
    for (let i = 0; i < spec.n!; i++) {
      const x = rng.below(w);
      const y = rng.below(h);
      const rw = 1 + rng.below(Math.max(1, Math.floor(w / 3)));
      const rh = 1 + rng.below(Math.max(1, Math.floor(h / 3)));
      for (let yy = y; yy < Math.min(h, y + rh); yy++) for (let xx = x; xx < Math.min(w, x + rw); xx++) data[yy * w + xx] = 1;
    }
  } else {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = rng.below(1000) < spec.permille! ? 1 : 0;
  }
  return { width: w, height: h, data };
}

describe('the scipy.ndimage operations', () => {
  it('give SciPy\'s dilations, opening, closings, filled holes, labels and boxes', () => {
    for (const c of V.ndimage.cases) {
      const m = maskFrom(c.spec);
      const where = JSON.stringify(c.spec);
      expect(sha16(m.data), where).toBe(c.mask);
      const got: Record<string, Uint8Array> = {
        dil1: binaryDilation(m, 1).data,
        dil2: binaryDilation(m, 2).data,
        dil4: binaryDilation(m, 4).data,
        dil6: binaryDilation(m, 6).data,
        open3: binaryOpening(m, 3).data,
        close3: binaryClosing(m, 3).data,
        close15: binaryClosing(m, 15).data,
        fill: binaryFillHoles(m).data,
      };
      for (const [k, want] of Object.entries(c.sha)) expect(sha16(got[k]!), `${k} of ${where}`).toBe(want);
      const { labels, count } = label(m);
      expect(sha16(labels), `labels of ${where}`).toBe(c.labels);
      expect(count).toBe(c.count);
      expect(findObjects(labels, m.width, m.height, count)).toEqual(c.objects);
    }
  });

  it('does not change its input', () => {
    const m = maskFrom(V.ndimage.cases[0]!.spec);
    const before = m.data.slice();
    binaryDilation(m, 3);
    binaryClosing(m, 3);
    binaryFillHoles(m);
    expect(m.data).toEqual(before);
  });
});

// --- matcrops ----------------------------------------------------------------------------------------------------

const picture = (name: string): RgbImage => render(V.matcrops.specs.find((s) => s.name === name)!);

describe('matcrops', () => {
  it('finds the mat\'s colour as the middle of the peak of the coarse histogram, on every strided view', () => {
    for (const c of V.matcrops.mat_colour) expect(matColour(picture(c.spec), c.step), `${c.spec}, every ${c.step}`).toEqual(c.mat);
  });

  it('marks the pixels far from the mat', () => {
    for (const c of V.matcrops.notmat) {
      const m = notmatMask(picture(c.spec), c.mat, c.tol);
      expect(m.data.reduce((a, b) => a + b, 0)).toBe(c.count);
      expect(sha16(m.data), `${c.spec}, tol ${c.tol}`).toBe(c.sha);
    }
  });

  it('marks dark, not red pixels', () => {
    for (const c of V.matcrops.border) {
      const m = borderMask(picture(c.spec), ...(c.args as [number?, number?, number?]));
      expect(m.data.reduce((a, b) => a + b, 0)).toBe(c.count);
      expect(sha16(m.data), `${c.spec}, ${c.args}`).toBe(c.sha);
    }
  });

  it('measures detail as Python does, bit for bit (float32)', () => {
    expect(FACE_DOWN_DETAIL).toBe(V.matcrops.face_down);
    for (const c of V.matcrops.detail) {
      expect(detail(render({ w: c.w, h: c.h, seed: c.seed, layers: c.layers })), `${c.w} x ${c.h}, seed ${c.seed}`).toBe(c.detail);
    }
  });

  it('tells a face from a plain sleeve', () => {
    const sleeve = render({ w: 110, h: 155, seed: 1, layers: [{ rect: [0, 0, 110, 155], colour: [200, 60, 120], jitter: 1 }] });
    const face = render({ w: 110, h: 155, seed: 2, layers: [{ rect: [0, 0, 110, 155], cell: 5 }] });
    expect(detail(sleeve)).toBeLessThan(FACE_DOWN_DETAIL);
    expect(detail(face)).toBeGreaterThan(FACE_DOWN_DETAIL);
  });

  it('takes the mean step in float32 like numpy', () => {
    // rows differ by 1 everywhere, columns by 0: (1 + 0) / 2, exactly
    const a = Uint8Array.from({ length: 48 * 64 }, (_, i) => Math.floor(i / 48) % 256);
    expect(stepMean(a, 48, 64)).toBe(0.5);
    // a mean that is not a float32 (1/3 of the steps are 1): rounded to float32 at each of numpy's three steps
    const b = Uint8Array.from({ length: 48 * 64 }, (_, i) => (Math.floor(i / 48) % 3 === 0 ? 1 : 0));
    const f = Math.fround;
    expect(stepMean(b, 48, 64)).toBe(f(f(f(42 * 48) / 3024 + f(0 / 3008)) / 2));
  });
});

describe('the bootstrap finder', () => {
  it('finds the isolated cards Python finds, on a mat by colour and on a red mat by dark border', () => {
    for (const c of V.findcards.cases) {
      const img = render(c.spec);
      expect(matColour(img)).toEqual(c.mat);
      for (const r of c.runs) {
        const opts: Parameters<typeof findCards>[2] = { ...r.opts };
        if (r.mask_tol !== null) opts.mask = notmatMask(img, c.mat, r.mask_tol);
        const got = findCards(img, r.long_px, opts);
        expect(got.length, `${r.name}: ${got.length} found`).toBe(r.found.length);
        got.forEach((b, i) => {
          const w = r.found[i]!;
          // the hull's corners come in another order than Qhull's and cos and sin are the browser's: equal to ~1e-12
          expect(b.centre[0]).toBeCloseTo(w.centre[0]!, 9);
          expect(b.centre[1]).toBeCloseTo(w.centre[1]!, 9);
          expect(b.long_px).toBeCloseTo(w.long_px, 9);
          expect(b.short_px).toBeCloseTo(w.short_px, 9);
          expect(b.angle_deg).toBeCloseTo(w.angle_deg, 8);
          expect(b.fill).toBeCloseTo(w.fill, 9);
        });
      }
    }
  });

  it('fits the least-area rectangle of random points', () => {
    let unique = 0;
    for (const c of V.findcards.min_area_rect) {
      const r = minAreaRect(c.points);
      // the least area is Python's whatever the order of the hull's corners
      expect(r.long_px * r.short_px).toBeCloseTo(c.long_px * c.short_px, 7);
      if (c.ties > 1) continue; // some sets tie, and then the rectangle is the one Qhull's hull order reaches first
      unique++;
      expect(r.centre[0]).toBeCloseTo(c.centre[0]!, 8);
      expect(r.centre[1]).toBeCloseTo(c.centre[1]!, 8);
      expect(r.long_px).toBeCloseTo(c.long_px, 8);
      expect(r.short_px).toBeCloseTo(c.short_px, 8);
      // the angle is modulo 180: a rectangle nearly upright can come out as 0.0000001 or 179.9999999
      const d = Math.abs(r.angle_deg - c.angle_deg) % 180;
      expect(Math.min(d, 180 - d)).toBeLessThan(1e-7);
    }
    expect(unique).toBeGreaterThan(5);
  });

  it('needs points that do not lie on a line', () => {
    expect(() => minAreaRect([[0, 0], [1, 1], [2, 2]])).toThrow('do not lie on a line');
  });
});

// --- the gallery -------------------------------------------------------------------------------------------------

/** rows x dim float32 values as test/gen/table.py draws them: a unit draw less a half, rounded to float32. */
function f32Matrix(rng: XorShift32, rows: number, dim: number): Float32Array {
  return Float32Array.from({ length: rows * dim }, () => rng.unit() - 0.5);
}

describe('float32 scores', () => {
  it('rounds a fused multiply-add once, also where a double would round it onto a float32 midpoint', () => {
    expect(V.fma32.length).toBeGreaterThan(150);
    for (const [a, b, c, want] of V.fma32) expect(fmaF32(a, b, c), `${a} * ${b} + ${c}`).toBe(want);
    // the case the double gets wrong: 1 + 2^-24 + 2^-54 is above the midpoint, so it rounds up
    const a = 13325 * 2 ** -27;
    const b = 80581 * 2 ** -27;
    expect(Math.fround(a * b + 1)).toBe(1);
    expect(fmaF32(a, b, 1)).toBe(1 + 2 ** -23);
  });

  it('are numpy\'s float32 matmul, bit for bit, in the shapes the tracker uses (the OpenBLAS chain of fused multiply-adds)', () => {
    for (const c of V.similarities) {
      const rng = new XorShift32(c.seed);
      const q = f32Matrix(rng, c.queries, c.dim);
      const g = f32Matrix(rng, c.gallery, c.dim);
      const views: Embeddings = { data: q, rows: c.queries, dim: c.dim };
      const gallery: Embeddings = { data: g, rows: c.gallery, dim: c.dim };
      expect(sha16(similarities(views, gallery)), `${c.queries} x ${c.dim} @ ${c.dim} x ${c.gallery}`).toBe(c.sha);
      expect(sha16(bestSimilarities(views, gallery)), `the best of ${c.queries} views over ${c.gallery} rows`).toBe(c.best);
    }
  });

  it('takes each row\'s best over the views (the four turns of a crop)', () => {
    const views: Embeddings = { data: Float32Array.from([1, 0, 0, 1, -1, 0]), rows: 3, dim: 2 };
    const gallery: Embeddings = { data: Float32Array.from([1, 0, 0, 1, -1, 0, 0, -1]), rows: 4, dim: 2 };
    expect(arr(bestSimilarities(views, gallery))).toEqual([1, 1, 1, 0]);
  });

  it('orders from the largest, equal values in index order', () => {
    expect(arr(argsortDescending([0.5, 0.9, 0.5, 0.9, 0.1, 0.5]))).toEqual([1, 3, 0, 2, 5, 4]);
    expect(arr(argsortDescending(new Float32Array([3, 3, 3])))).toEqual([0, 1, 2]);
    expect(arr(argsortDescending([]))).toEqual([]);
  });
});

describe('the gallery pyramid', () => {
  it('searches the level nearest a crop\'s long side, in log space', () => {
    for (const c of V.pyramid.cases) {
      const p = new Pyramid(new Map(c.scales.map((s) => [s, new Float32Array(12)])), 4);
      expect(p.scales).toEqual([...c.scales].sort((a, b) => a - b));
      expect(p.rows).toBe(3);
      expect(V.pyramid.long.map((l) => p.levelFor(l))).toEqual(c.level);
    }
  });

  it('takes an object of levels, and needs at least one', () => {
    const p = new Pyramid({ 80: new Float32Array(8), 40: new Float32Array(8) }, 4);
    expect(p.scales).toEqual([40, 80]);
    expect(p.level(70)).toBe(p.levels.get(80));
    expect(() => new Pyramid(new Map(), 4)).toThrow('at least one scale');
  });
});

/** A stand-in encoder as test/gen/table.py has it: the mean colour of each quarter of a picture. */
const quadMean: Encoder = {
  name: 'quadmean',
  dim: 12,
  async embed(images) {
    const out = new Float32Array(images.length * 12);
    images.forEach((im, i) => {
      const hh = im.height >> 1;
      const ww = im.width >> 1;
      const regions = [
        [0, hh, 0, ww],
        [0, hh, ww, im.width],
        [hh, im.height, 0, ww],
        [hh, im.height, ww, im.width],
      ] as const;
      regions.forEach(([y0, y1, x0, x1], q) => {
        const n = (y1 - y0) * (x1 - x0);
        for (let c = 0; c < 3; c++) {
          let sum = 0;
          for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) sum += im.data[(y * im.width + x) * 3 + c]!;
          out[i * 12 + q * 3 + c] = n ? sum / n : 0;
        }
      });
    });
    return out;
  },
};

describe('retrieval', () => {
  it('draws a card at a long side like Pillow\'s BOX resize', () => {
    for (const c of V.retrieval.at_long_side) {
      const r = atLongSide(render(c.spec), c.side);
      expect([r.width, r.height]).toEqual(c.size);
      expect(sha16(r.data), `${c.spec.w}x${c.spec.h} at ${c.side}`).toBe(c.sha);
    }
  });

  it('cuts the visible band of a card', () => {
    for (const c of V.retrieval.band) {
      const r = band(render(c.spec), c.view);
      expect([r.width, r.height]).toEqual(c.size);
      expect(sha16(r.data), `${c.spec.w}x${c.spec.h} ${c.view}`).toBe(c.sha);
    }
    for (const b of V.retrieval.bad_views) expect(() => band(render({ w: 10, h: 10, seed: 1, layers: [] }), b.view), b.view).toThrow('bad view');
  });

  it('finds the k nearest rows: the same rows, in the same order, with the same scores', () => {
    for (const c of V.retrieval.topk) {
      const rng = new XorShift32(c.seed);
      const q = f32Matrix(rng, c.queries, c.dim);
      const g = f32Matrix(rng, c.gallery, c.dim);
      const r = topk({ data: q, rows: c.queries, dim: c.dim }, { data: g, rows: c.gallery, dim: c.dim }, c.k);
      expect(r.k).toBe(c.idx[0]!.length);
      expect(arr(r.idx)).toEqual(c.idx.flat());
      // numpy's routine differs with the shape (a lone query, a small gallery): the scores agree to float32 rounding
      const want = c.scores.flat();
      arr(r.scores).forEach((s, i) => expect(Math.abs(s - want[i]!)).toBeLessThanOrEqual(2e-7 * Math.max(1, Math.abs(want[i]!))));
    }
  });

  it('breaks ties by the lower index', () => {
    const g: Embeddings = { data: Float32Array.from([1, 0, 1, 0, 0, 1, 1, 0]), rows: 4, dim: 2 };
    const r = topk({ data: Float32Array.from([1, 0]), rows: 1, dim: 2 }, g, 3);
    expect(arr(r.idx)).toEqual([0, 1, 3]);
  });

  it('searches a pyramid and a plain gallery at four turns, or one, and says which turn matched', async () => {
    const s = V.retrieval.search;
    const art = s.art.map((a) => render(a));
    const queries = s.queries.map((q) => {
      const im = atLongSide(art[q.art]!, q.side);
      return q.angle ? image.rotate(im, q.angle, { expand: true }) : im;
    });
    const pyr = await Pyramid.build(quadMean, art, s.scales);
    const plain: Embeddings = { data: await quadMean.embed(art.map((a) => atLongSide(a, 60))), rows: art.length, dim: 12 };
    const check = async (gallery: Pyramid | Embeddings, want: SearchResult): Promise<void> => {
      const r = await search(quadMean, gallery, queries, want.k, want.rotation_invariant);
      expect(r.k).toBe(want.idx[0]!.length);
      expect(arr(r.idx)).toEqual(want.idx.flat());
      expect(arr(r.rot)).toEqual(want.rot.flat());
      const w = want.scores.flat();
      arr(r.scores).forEach((v, i) => expect(Math.abs(v - w[i]!)).toBeLessThanOrEqual(1e-6 * Math.abs(w[i]!)));
    };
    for (const w of s.pyramid) await check(pyr, w);
    for (const w of s.plain) await check(plain, w);
  });

  it('rolls printings up into cards and counts the hits', () => {
    const r = V.retrieval.ranked;
    const ranked = rankedLabels(r.idx.flat(), 5, r.labels);
    expect(ranked).toEqual(r.ranked);
    expect(accuracy(ranked, r.truth)).toEqual(r.accuracy);
    expect(accuracy([], [])).toEqual({ top1: NaN, top5: NaN, n: 0 });
  });
});

// --- the change gate ---------------------------------------------------------------------------------------------

/** The frames of a sequence, as test/gen/table.py builds them. */
function gateFrames(seq: GateSequence['seq']): RgbImage[] {
  const frames: RgbImage[] = [];
  for (const step of seq.steps) {
    for (let i = 0; i < step.n; i++) {
      const layers: Spec['layers'] = [{ rect: [0, 0, seq.w, seq.h], colour: step.bg ?? [190, 25, 45], jitter: seq.noise }];
      for (const r of step.rects ?? []) {
        const [x, y, w, h, c] = r as [number, number, number, number, number[] | string[]];
        if (typeof c[0] === 'string') layers.push({ rect: [x, y, w, h], noise: true, seed: c[1] as unknown as number });
        else layers.push({ rect: [x, y, w, h], colour: c as [number, number, number], jitter: seq.noise });
      }
      frames.push(render({ w: seq.w, h: seq.h, seed: seq.seed + frames.length, layers }));
    }
  }
  return frames;
}

describe('the change gate', () => {
  it('has the settings\' defaults', () => {
    expect(gateSettings()).toEqual(V.changegate.defaults);
  });

  it('tells skin on noise and over the colour cube as Python does', () => {
    for (const c of V.changegate.skin) {
      let img: RgbImage;
      if (c.cube) {
        const px: number[] = [];
        for (let r = 0; r < 256; r += 5) for (let g = 0; g < 256; g += 5) for (let b = 0; b < 256; b += 5) px.push(r, g, b);
        img = { width: 1, height: px.length / 3, data: Uint8Array.from(px) };
      } else {
        img = render(c.spec!);
      }
      const m = skin(img);
      expect(m.data.reduce((a, b) => a + b, 0)).toBe(c.count);
      expect(sha16(m.data)).toBe(c.sha);
    }
  });

  for (const { seq, expected } of V.changegate.sequences) {
    it(`fires the same events at the same times, and keeps the same still table: ${seq.name}`, () => {
      const frames = gateFrames(seq);
      expect(frames.length).toBe(expected.frames);
      const s: GateSettings = gateSettings({ fps: seq.fps, ...seq.settings } as Partial<GateSettings>);
      const gate = new ChangeGate(s);
      frames.forEach((f, i) => {
        const events = gate.feed(i / s.fps, f);
        const want = expected.per_frame[i]!;
        expect(events, `frame ${i}`).toEqual(want.events);
        const bg = gate.state().background;
        expect(bg === null ? null : sha16(bg), `still table after frame ${i}`).toBe(want.background);
      });
      const st = gate.state();
      expect(sha16(st.still)).toBe(expected.still);
      expect(sha16(st.lastSame)).toBe(expected.last_same);
      expect(sha16(st.startupHand)).toBe(expected.startup_hand);
      expect(st.offTable).toBe(expected.off_table);
      expect(st.mat).toEqual(expected.mat);
      expect(gate.events).toEqual(expected.per_frame.flatMap((f) => f.events));
    });
  }

  it('sees the played and moved card, ignores the hand', () => {
    const { seq } = V.changegate.sequences[0]!;
    const gate = new ChangeGate(gateSettings({ fps: seq.fps, ...seq.settings } as Partial<GateSettings>));
    gateFrames(seq).forEach((f, i) => gate.feed(i / seq.fps, f));
    expect(gate.events.map((e) => e.kind)).toEqual(['appeared', 'disappeared', 'appeared']);
    const first = gate.events[0]!;
    expect(first.box[0]).toBeGreaterThanOrEqual(45);
    expect(first.box[2] - first.box[0]).toBeLessThan(14);
    expect(first.extra.t_before).toBeCloseTo(12 / seq.fps, 9); // the last empty frame before the hand came back
  });

  it('draws its view of the table as the tracker does, and maps its boxes back to the frame', () => {
    for (const c of V.views) {
      const table = c.table as [number, number, number, number];
      expect(viewHeight(table, c.vw)).toBe(c.vh);
      expect(viewToFrame(c.box as [number, number, number, number], table, c.vw, c.frame[0]!, c.frame[1]!), JSON.stringify(c)).toEqual(c.frame_box);
    }
    expect(viewHeight(LAYOUTS['la-rq'].table, 320)).toBe(272);
  });

  it('throws when the view changes size', () => {
    const gate = new ChangeGate(gateSettings({ mat_rgb: [0, 0, 0] }));
    const black = (w: number, h: number): RgbImage => ({ width: w, height: h, data: new Uint8Array(w * h * 3) });
    gate.feed(0, black(20, 10));
    expect(() => gate.feed(0.2, black(30, 10))).toThrow('began at 20 x 10');
  });
});

// --- the layout finder -------------------------------------------------------------------------------------------

describe('the layout finder', () => {
  for (const c of V.autolayout.cases) {
    it(`finds the borders, the table window and the card size like Python: ${c.name}`, async () => {
      const frames = c.specs.map((s) => render(s));
      expect(borders(frames)).toEqual(c.borders);
      const tw = tableWindow(frames);
      expect(tw).toEqual(c.table_window);
      if (c.detector_calls.length > 0) {
        // the detector, replaced by what it said in Python
        const r = replayDetector(c.detector_calls, frames);
        expect(await autoLayout(frames, r.detect)).toEqual(c.auto_layout);
        expect(r.used()).toBe(c.detector_calls.length);
        if (tw !== null) expect(await cardSize(replayDetector(c.detector_calls, frames).detect, frames, tw.window)).toBe(c.card_size);
      }
      // without a detector: the bootstrap finder
      expect(await autoLayout(frames)).toEqual(c.auto_layout_finder);
      if (tw !== null) expect(cardSizeFinder(frames, tw.window, tw.mat)).toBe(c.card_size_finder);
    }, 60_000); // each renders its frames and runs the finder, about a second and a half on a laptop
  }
});
