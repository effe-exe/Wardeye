// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What test/table.test.ts shares with test/gen/table.py: one seeded 32-bit xorshift and one picture renderer, so both
// sides build the same synthetic pictures without any pixels being stored.

import { createHash } from 'node:crypto';
import type { RgbImage } from '../src/types';

/** Marsaglia's 32-bit xorshift (13, 17, 5). */
export class XorShift32 {
  private x: number;

  constructor(seed: number) {
    this.x = seed >>> 0 || 1;
  }

  /** The next 32-bit value. */
  next(): number {
    let x = this.x;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.x = x >>> 0;
    return this.x;
  }

  /** A whole number in [0, n). */
  below(n: number): number {
    return this.next() % n;
  }

  /** A double in [0, 1): a 32-bit value over 2^32, exact. */
  unit(): number {
    return this.next() / 4294967296;
  }
}

/** One layer of a synthetic picture, drawn over the ones before it, clipped to the picture:
 *  - `colour` (with `jitter`): the colour, each channel of each pixel moved by a whole number in [-jitter, jitter];
 *  - `noise`: each channel the top byte of the next random number;
 *  - `cell`: squares of that many pixels, each a random colour (three draws a square, clipped or not).
 * A layer with its own `seed` draws from a generator of its own, so it looks the same in every picture. */
export interface Layer {
  rect: [number, number, number, number];
  colour?: [number, number, number];
  jitter?: number;
  noise?: boolean;
  cell?: number;
  seed?: number;
}

export interface PictureSpec {
  w: number;
  h: number;
  seed: number;
  layers: Layer[];
}

/** The picture a spec describes: rows top to bottom, pixels in raster order, channels r, g, b, one draw of the
 * generator for each jittered channel (no draw when the jitter is 0). */
export function render(spec: PictureSpec): RgbImage {
  const { w, h } = spec;
  const data = new Uint8Array(w * h * 3);
  const mainRng = new XorShift32(spec.seed);
  for (const layer of spec.layers) {
    const rng = layer.seed !== undefined ? new XorShift32(layer.seed) : mainRng;
    const [x0, y0, rw, rh] = layer.rect;
    const ya = Math.max(0, y0);
    const yb = Math.min(h, y0 + rh);
    const xa = Math.max(0, x0);
    const xb = Math.min(w, x0 + rw);
    if (layer.cell) {
      const cell = layer.cell;
      for (let cy = 0; cy < Math.ceil(rh / cell); cy++) {
        for (let cx = 0; cx < Math.ceil(rw / cell); cx++) {
          const c = [rng.next() >>> 24, rng.next() >>> 24, rng.next() >>> 24];
          for (let y = Math.max(ya, y0 + cy * cell); y < Math.min(yb, y0 + (cy + 1) * cell); y++) {
            for (let x = Math.max(xa, x0 + cx * cell); x < Math.min(xb, x0 + (cx + 1) * cell); x++) {
              for (let k = 0; k < 3; k++) data[(y * w + x) * 3 + k] = c[k]!;
            }
          }
        }
      }
    } else if (layer.noise) {
      for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) for (let k = 0; k < 3; k++) data[(y * w + x) * 3 + k] = rng.next() >>> 24;
    } else {
      const colour = layer.colour!;
      const j = layer.jitter ?? 0;
      for (let y = ya; y < yb; y++) {
        for (let x = xa; x < xb; x++) {
          for (let k = 0; k < 3; k++) {
            let v = colour[k]!;
            if (j) v += rng.below(2 * j + 1) - j;
            data[(y * w + x) * 3 + k] = Math.min(255, Math.max(0, v));
          }
        }
      }
    }
  }
  return { width: w, height: h, data };
}

/** The SHA-256 of the bytes of an array, as hex. */
export function sha256(a: Uint8Array | Int32Array | Float32Array | Float64Array): string {
  return createHash('sha256').update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest('hex');
}
