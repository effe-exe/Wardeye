// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// src/image.ts against Pillow 12.3.0, byte for byte. The vectors (test/vectors/image.json, from test/gen/image.py)
// name synthetic pictures both sides build from one seeded xorshift, the steps done to them, and the SHA-256 of
// Pillow's result. The real-frame parity runs the live runner's own picture calls on frames of the LA final; it
// needs the private data (RIFTEYE_M3=~/rifteye-data/m3) and skips without it.

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { crop, cropGray, fromRgba, paste, resize, resizeGray, rgbImage, rotate, rotateCrop, rotateMatrix, toGray, type RotateOptions } from '../src/image';
import type { Resample, RgbImage } from '../src/types';

// ---- the synthetic pictures, as test/gen/image.py builds them ----

type Mode = 'RGB' | 'L' | 'RGBA';
const CHANNELS: Record<Mode, number> = { RGB: 3, L: 1, RGBA: 4 };

interface Spec {
  kind: 'noise' | 'smooth' | 'blocks' | 'black';
  w: number;
  h: number;
  seed: number;
  mode: Mode;
}

/** A picture and its Pillow mode (RgbImage and GrayImage have the same shape). */
interface Pic {
  mode: Mode;
  im: RgbImage;
}

/** n bytes of Marsaglia's 32-bit xorshift (13, 17, 5), four a step, low byte first. */
function xorshiftBytes(seed: number, n: number): Uint8Array {
  let x = seed >>> 0 || 1;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 4) {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    for (let b = 0; b < 4 && i + b < n; b++) out[i + b] = (x >>> (8 * b)) & 255;
  }
  return out;
}

function synth(spec: Spec): Pic {
  const { kind, w, h, mode } = spec;
  const ch = CHANNELS[mode];
  const n = w * h * ch;
  const r = kind === 'black' ? new Uint8Array(n) : xorshiftBytes(spec.seed, n);
  let data = r;
  if (kind === 'smooth' || kind === 'blocks') {
    data = new Uint8Array(n);
    const bw = (w + 7) >> 3;
    for (let y = 0, i = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < ch; c++, i++) {
          data[i] = kind === 'smooth' ? 20 + ((x * 5 + y * 3 + c * 50) % 200) + (r[i]! & 7) : r[((y >> 3) * bw + (x >> 3)) * ch + c]! & 1 ? 255 : 0;
        }
      }
    }
  }
  return { mode, im: { width: w, height: h, data } };
}

// ---- the steps ----

type Step =
  | { op: 'crop'; box: [number, number, number, number] }
  | { op: 'resize'; size: [number, number]; resample: Resample }
  | {
      op: 'rotate';
      angle: number;
      resample?: 'nearest' | 'bilinear' | 'bicubic';
      expand?: boolean;
      center?: [number, number] | null;
      fill?: [number, number, number] | null;
    }
  | { op: 'gray' }
  | { op: 'rgb' }
  | { op: 'paste'; onto: Spec; at: [number, number] }
  | { op: 'rotateCrop'; rotate: Extract<Step, { op: 'rotate' }>; box: [number, number, number, number] };

type RotateStep = Extract<Step, { op: 'rotate' }>;

function rotateOptions(s: RotateStep): RotateOptions {
  return {
    ...(s.resample ? { resample: s.resample } : {}),
    ...(s.expand ? { expand: true } : {}),
    ...(s.center ? { center: s.center } : {}),
    ...(s.fill ? { fill: s.fill } : {}),
  };
}

/** The steps with each rotate followed by a crop done as one rotateCrop. */
function fused(steps: Step[]): Step[] {
  const out: Step[] = [];
  for (const s of steps) {
    const last = out[out.length - 1];
    if (s.op === 'crop' && last?.op === 'rotate') out[out.length - 1] = { op: 'rotateCrop', rotate: last, box: s.box };
    else out.push(s);
  }
  return out;
}

function step(p: Pic, s: Step): Pic {
  switch (s.op) {
    case 'crop':
      return { mode: p.mode, im: p.mode === 'L' ? cropGray(p.im, s.box) : crop(p.im, s.box) };
    case 'resize':
      return { mode: p.mode, im: p.mode === 'L' ? resizeGray(p.im, s.size, s.resample) : resize(p.im, s.size, s.resample) };
    case 'rotate':
      return { mode: 'RGB', im: rotate(p.im, s.angle, rotateOptions(s)) };
    case 'rotateCrop':
      return { mode: 'RGB', im: rotateCrop(p.im, s.rotate.angle, rotateOptions(s.rotate), s.box) };
    case 'gray':
      return { mode: 'L', im: toGray(p.im) };
    case 'rgb':
      return { mode: 'RGB', im: fromRgba(p.im.data, p.im.width, p.im.height) };
    case 'paste': {
      const dst = synth(s.onto);
      paste(dst.im, p.im, s.at);
      return dst;
    }
  }
}

interface Expected {
  name: string;
  steps: Step[];
  mode: Mode;
  size: [number, number];
  sha256: string;
  px: [number, number, number[]][];
}

function sha256(data: Uint8Array): string {
  return createHash('sha256').update(data).digest('hex');
}

/** The case's result as the vectors describe one: mode, size, a few pixels, and the hash. */
function described(p: Pic): Omit<Expected, 'name' | 'steps'> {
  const { width: w, height: h, data } = p.im;
  const ch = CHANNELS[p.mode];
  const at = (x: number, y: number): [number, number, number[]] => [x, y, Array.from(data.subarray((y * w + x) * ch, (y * w + x + 1) * ch))];
  const px = w && h ? [at(0, 0), at(w - 1, h - 1), at(w >> 1, h >> 1), at(Math.floor(w / 3), Math.floor((2 * h) / 3))] : [];
  return { mode: p.mode, size: [w, h], sha256: sha256(data), px };
}

function check(input: Pic, c: Expected, steps: Step[] = c.steps): void {
  const out = steps.reduce(step, input);
  const got = described(out);
  expect({ mode: got.mode, size: got.size, px: got.px }).toEqual({ mode: c.mode, size: c.size, px: c.px });
  expect(got.sha256).toBe(c.sha256);
}

/** The case as it is, and with rotateCrop for each rotate and crop that follow each other. */
function checkBoth(input: Pic, c: Expected): void {
  check(input, c);
  const f = fused(c.steps);
  if (f.length < c.steps.length) check(input, c, f);
}

// ---- the vectors ----

interface Vectors {
  pillow: string;
  matrices: { size: [number, number]; angle: number; center: [number, number] | null; expand: boolean; matrix: number[]; out: [number, number] }[];
  sweep: { n: number; sha256: string; libm_differs: number[] };
  cases: (Expected & { input: Spec })[];
}

const vectors = JSON.parse(readFileSync(new URL('./vectors/image.json', import.meta.url), 'utf8')) as Vectors;
const groups = new Map<string, Vectors['cases']>();
for (const c of vectors.cases) {
  const op = c.steps.length === 1 ? c.steps[0]!.op : 'chain';
  groups.set(op, [...(groups.get(op) ?? []), c]);
}

describe(`Pillow ${vectors.pillow} on synthetic pictures`, () => {
  for (const [op, cases] of groups) {
    describe(op, () => {
      it.each(cases)('$name', (c) => checkBoth(synth(c.input), c));
    });
  }
});

describe('rotateCrop', () => {
  const rotations = vectors.cases.filter((c) => c.steps.length === 1 && c.steps[0]!.op === 'rotate');
  it.each(rotations)('is crop(rotate) for $name', (c) => {
    const s = c.steps[0] as RotateStep;
    const im = synth(c.input).im;
    const whole = rotate(im, s.angle, rotateOptions(s));
    const [w, h] = [whole.width, whole.height];
    const boxes: [number, number, number, number][] = [
      [0, 0, w, h], [1, 2, Math.max(1, w - 3), Math.max(2, h - 1)], [-4.5, -2, w / 2, h / 2 + 0.5], [w / 3, h / 4, w + 7, h + 3],
      [w + 1, 0, w + 5, 4], [2, 2, 2, 9],
    ];
    for (const box of boxes) expect(rotateCrop(im, s.angle, rotateOptions(s), box)).toEqual(crop(whole, box));
  });
});

describe('the rotation matrix', () => {
  it.each(vectors.matrices)('is Image.rotate\'s for $size at $angle degrees', (m) => {
    const got = rotateMatrix(m.size[0], m.size[1], m.angle, { expand: m.expand, ...(m.center ? { center: m.center } : {}) });
    expect(got).toEqual({ matrix: m.matrix, size: m.out });
  });

  it(`is Pillow's with correctly rounded cos and sin for ${vectors.sweep.n} angles, centres and sizes`, () => {
    // the list test/gen/image.py matrix_sweep builds
    const bytes = new DataView(new ArrayBuffer(vectors.sweep.n * 64));
    let x = 0x9e3779b9;
    for (let i = 0; i < vectors.sweep.n; i++) {
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      x >>>= 0;
      const angle = i % 2 ? (x / 4294967296) * 720 - 360 : i * 0.0137 - 137;
      const center: [number, number] | undefined = i % 4 < 2 ? undefined : [17.3 + (i % 7) * 0.61, 9.1 + (i % 5) * 1.37];
      const { matrix, size } = rotateMatrix(40 + (i % 13), 30 + (i % 11), angle, { expand: i % 3 === 0, ...(center ? { center } : {}) });
      [...matrix, ...size].forEach((v, k) => bytes.setFloat64(i * 64 + k * 8, v, true));
    }
    expect(sha256(new Uint8Array(bytes.buffer))).toBe(vectors.sweep.sha256);
    // glibc is not always correctly rounded, so a few of these are not Linux Pillow's own
    expect(vectors.sweep.libm_differs.length).toBeLessThan(vectors.sweep.n / 1000);
  });
});

describe('the picture operations', () => {
  it('wrap and check their bytes', () => {
    expect(rgbImage(2, 1).data).toEqual(new Uint8Array(6));
    expect(() => rgbImage(2, 2, new Uint8Array(11))).toThrow('needs 12 bytes, not 11');
    expect(() => resize({ width: 2, height: 2, data: new Uint8Array(4) }, [1, 1], 'box')).toThrow('needs 12 bytes');
    expect(() => resizeGray({ width: 2, height: 2, data: new Uint8Array(12) }, [1, 1], 'box')).toThrow('needs 4 bytes');
  });

  it('drop the alpha of canvas pixels', () => {
    const rgba = new Uint8ClampedArray([1, 2, 3, 255, 4, 5, 6, 0]);
    expect(fromRgba(rgba, 2, 1)).toEqual({ width: 2, height: 1, data: new Uint8Array([1, 2, 3, 4, 5, 6]) });
    expect(() => fromRgba(rgba, 3, 1)).toThrow('needs 12 bytes, not 8');
  });

  it('give the same bytes from pixels that do not start on a word', () => {
    // toGray and fromRgba read four pixels at a time when the bytes are word-aligned, and byte by byte otherwise
    const rgba = xorshiftBytes(7, 1 + 11 * 5 * 4);
    const rgb = xorshiftBytes(8, 1 + 11 * 5 * 3);
    const [a, b] = [rgba.slice(1), rgba.subarray(1)];
    expect(fromRgba(b, 11, 5)).toEqual(fromRgba(a, 11, 5));
    const [c, d] = [{ width: 11, height: 5, data: rgb.slice(1) }, { width: 11, height: 5, data: rgb.subarray(1) }];
    expect(toGray(d)).toEqual(toGray(c));
    const px = (x: number): number => (c.data[x * 3]! * 19595 + c.data[x * 3 + 1]! * 38470 + c.data[x * 3 + 2]! * 7471 + 0x8000) >> 16;
    expect(Array.from(toGray(c).data)).toEqual(Array.from({ length: 55 }, (_, x) => px(x)));
    expect(Array.from(fromRgba(a, 11, 5).data)).toEqual(Array.from(a).filter((_, i) => i % 4 !== 3));
  });

  it('refuse what Pillow refuses', () => {
    const im = synth({ kind: 'noise', w: 4, h: 3, seed: 1, mode: 'RGB' }).im;
    expect(() => crop(im, [3, 0, 2, 1])).toThrow("'right' is less than 'left'");
    expect(() => crop(im, [0, 2, 1, 1])).toThrow("'lower' is less than 'upper'");
    expect(() => crop(im, [0, 0, NaN, 1])).toThrow('must be finite');
    expect(() => resize(im, [0, 3], 'bilinear')).toThrow('height and width must be > 0');
    expect(() => resize(im, [2.5, 3], 'bilinear')).toThrow('whole pixels');
    expect(() => rotate(im, 10, { resample: 'box' as 'nearest' })).toThrow('cannot be used');
    expect(() => rotate(im, 10, { fill: [1.5, 0, 0] })).toThrow('three whole numbers');
    expect(() => paste(im, im, [0.5, 0])).toThrow('whole pixels');
  });

  it('crop as Python rounds, halves to even', () => {
    const im = synth({ kind: 'noise', w: 8, h: 8, seed: 3, mode: 'RGB' }).im;
    expect(crop(im, [0.5, 1.5, 2.5, 3.5])).toEqual(crop(im, [0, 2, 2, 4]));
    expect(crop(im, [-0.5, -1.5, 5.5, 6.5])).toEqual(crop(im, [0, -2, 6, 6]));
  });

  it('paste a picture onto itself as it was', () => {
    const im = synth({ kind: 'noise', w: 6, h: 5, seed: 4, mode: 'RGB' }).im;
    const before = { ...im, data: im.data.slice() };
    const want = { ...im, data: im.data.slice() };
    paste(want, before, [2, 1]);
    paste(im, im, [2, 1]);
    expect(im).toEqual(want);
  });

  it('keep a fill colour clipped to 0..255', () => {
    const im = synth({ kind: 'noise', w: 5, h: 5, seed: 5, mode: 'RGB' }).im;
    const a = rotate(im, 45, { fill: [300, -4, 7] });
    const b = rotate(im, 45, { fill: [255, 0, 7] });
    expect(a).toEqual(b);
    expect(Array.from(a.data.subarray(0, 3))).toEqual([255, 0, 7]);
  });
});

// ---- the real frames (private) ----

const M3 = process.env.RIFTEYE_M3;
const REAL = M3 ? join(M3, 'fixtures/image/real.json') : '';
const FRAMES = M3 ? join(M3, 'frames/la-final-rgb') : '';

interface Real {
  pillow: string;
  /** detail: matcrops.detail's number on that grey picture, for the table's parity. */
  cases: (Expected & { frame: string; detail?: number })[];
}

const HAVE_REAL = !!M3 && existsSync(REAL) && existsSync(join(FRAMES, 'frames.json'));

describe.skipIf(!HAVE_REAL)('Pillow on the LA final (private: RIFTEYE_M3)', () => {
  const real = (HAVE_REAL ? JSON.parse(readFileSync(REAL, 'utf8')) : { pillow: '', cases: [] }) as Real;
  const meta = (HAVE_REAL ? JSON.parse(readFileSync(join(FRAMES, 'frames.json'), 'utf8')) : { frames: [] }) as {
    frames: { file: string; width: number; height: number }[];
  };
  const frames = new Map<string, Pic>();
  /** A raw RGB frame (frames/la-final-rgb: no header, rows top to bottom), read once. */
  const frame = (file: string): Pic => {
    let p = frames.get(file);
    if (!p) {
      const f = meta.frames.find((m) => m.file === file);
      if (!f) throw new Error(`${file} is not in frames.json`);
      p = { mode: 'RGB', im: rgbImage(f.width, f.height, new Uint8Array(readFileSync(join(FRAMES, file)))) };
      frames.set(file, p);
    }
    return p;
  };

  it('was written by the Pillow image.ts ports', () => expect(real.pillow).toBe(vectors.pillow));
  it.each(real.cases)('$name', (c) => checkBoth(frame(c.frame), c));
});
