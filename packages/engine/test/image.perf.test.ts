// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// How fast the hot picture operations run in Node, against the budgets the live frame rate leaves them. Timings
// mean little on a shared machine, so this is opt-in:
//   RIFTEYE_PERF=1 npx vitest run packages/engine/test/image.perf.test.ts
// Each line gives the fastest and the median of 25 runs after 20 warm-ups (so the JIT has optimised the loops), on
// synthetic pictures of the real sizes. The budget is checked against the fastest: what the code costs when it has
// a core to itself.

import { describe, expect, it } from 'vitest';
import { crop, fromRgba, resize, resizeGray, rotate, rotateCrop, toGray } from '../src/image';
import type { RgbImage } from '../src/types';

/** A picture of noise from a 32-bit xorshift: the loops do not depend on the content. */
function noise(width: number, height: number, seed: number): RgbImage {
  const data = new Uint8Array(width * height * 3);
  let x = seed;
  for (let i = 0; i < data.length; i++) {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    data[i] = x & 255;
  }
  return { width, height, data };
}

function timings(f: () => unknown): { fastest: number; median: number } {
  for (let i = 0; i < 20; i++) f();
  const t: number[] = [];
  for (let i = 0; i < 25; i++) {
    const start = performance.now();
    f();
    t.push(performance.now() - start);
  }
  t.sort((a, b) => a - b);
  return { fastest: t[0]!, median: t[12]! };
}

const frame = noise(1920, 1080, 1);
const rgba = new Uint8ClampedArray(1920 * 1080 * 4);
const table = crop(frame, [365, 65, 1555, 1080]); // la-rq's table window at 1080p
const narrower = crop(frame, [371, 65, 1549, 1080]);
const patch = crop(frame, [800, 500, 1050, 750]);
const local = crop(frame, [862, 502, 1058, 698]); // card_crop's neighbourhood of a 155 px card
const grey = toGray(frame);

/** [what, budget in ms or null for none, the call]. */
const CASES: [string, number | null, () => unknown][] = [
  ['1920 x 1080 -> 96 x 54 box (Scene.small)', 25, () => resize(frame, [96, 54], 'box')],
  ['1178 x 1015 -> 537 x 463 bicubic', 40, () => resize(narrower, [537, 463], 'bicubic')],
  ['1190 x 1015 -> 537 x 458 bicubic (the detector window, la-rq)', 40, () => resize(table, [537, 458], 'bicubic')],
  ['250 x 250 bicubic rotation about a fractional centre', 5, () => rotate(patch, 33.3, { resample: 'bicubic', center: [125.4, 124.6] })],
  ['toGray of 1920 x 1080', 10, () => toGray(frame)],
  ['fromRgba of 1920 x 1080', null, () => fromRgba(rgba, 1920, 1080)],
  ['crop of the table window', null, () => crop(frame, [365, 65, 1555, 1080])],
  ['table window -> 160 x 120 box (table_like)', null, () => resize(table, [160, 120], 'box')],
  ['table window -> 320 x 272 bilinear (watch)', null, () => resize(table, [320, 272], 'bilinear')],
  ['card_crop: 196 x 196 turned bicubic, 111 x 155 kept (rotateCrop)', null, () =>
    rotateCrop(local, -52.5, { resample: 'bicubic', center: [98.4, 98.7] }, [42.9, 21.2, 153.9, 176.2])],
  ['card_crop turned 90 (rotate expand)', null, () => rotate(crop(local, [42, 21, 153, 176]), 90, { expand: true })],
  ['letterbox 155 x 155 -> 224 x 224 bicubic', null, () => resize(crop(local, [20, 20, 175, 175]), [224, 224], 'bicubic')],
  ['grey 1920 x 1080 -> 96 x 54 box (autolayout)', null, () => resizeGray(grey, [96, 54], 'box')],
];

describe.skipIf(process.env.RIFTEYE_PERF !== '1')('picture operations in Node (RIFTEYE_PERF=1)', () => {
  it.each(CASES)('%s', (what, budget, f) => {
    const { fastest, median } = timings(f);
    console.log(`${what}: ${fastest.toFixed(2)} ms fastest, ${median.toFixed(2)} ms median${budget === null ? '' : ` (budget ${budget} ms)`}`);
    if (budget !== null) expect(fastest).toBeLessThan(budget);
  });
});
