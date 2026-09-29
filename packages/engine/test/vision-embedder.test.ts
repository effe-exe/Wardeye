// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// embedder.ts against encoders.letterbox and l2n and embed/onnx.py (batch_of, the encoder's name), on the seeded
// synthetic vectors of test/gen/vision_embedder.py, and the encoder's fixed batches on a stand-in net.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BATCH, DIM, IMG_SIZE, OnnxEncoder, cropBatch, encoderName, l2n, letterbox, stem } from '../src/embedder';
import { rgbImage } from '../src/image';
import type { RgbImage } from '../src/types';

interface Vectors {
  letterbox: { side: number; pictures: { size: [number, number]; rgb: string; letterbox: string }[] };
  batch_of: { side: number; n: number; sha256: string };
  l2n: Record<string, { x: string; l2n: string }>;
  name: { bytes: string; names: Record<string, string> };
}

const V = JSON.parse(readFileSync(new URL('./vectors/vision-embedder.json', import.meta.url), 'utf8')) as Vectors;
const bytesOf = (b64: string) => new Uint8Array(Buffer.from(b64, 'base64'));
const floatsOf = (b64: string) => {
  const b = bytesOf(b64);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const pictures: RgbImage[] = V.letterbox.pictures.map((p) => rgbImage(p.size[0], p.size[1], bytesOf(p.rgb)));

describe('letterbox and the batch', () => {
  it('letterbox is encoders.letterbox, byte for byte, for every aspect', () => {
    V.letterbox.pictures.forEach((p, i) => {
      const lb = letterbox(pictures[i]!, V.letterbox.side);
      expect([lb.width, lb.height]).toEqual([V.letterbox.side, V.letterbox.side]);
      expect(Buffer.from(lb.data).equals(Buffer.from(bytesOf(p.letterbox)))).toBe(true);
    });
  });

  it('cropBatch is batch_of: float32 NCHW, 0..255', () => {
    const x = cropBatch(pictures.slice(0, V.batch_of.n), V.batch_of.side);
    expect(createHash('sha256').update(new Uint8Array(x.buffer)).digest('hex')).toBe(V.batch_of.sha256);
  });

  it('pads with black crops', () => {
    const x = cropBatch(pictures.slice(0, 1), 24, 3);
    expect(x.length).toBe(3 * 3 * 24 * 24);
    expect(x.subarray(3 * 24 * 24).every((v) => v === 0)).toBe(true);
    expect(() => cropBatch(pictures, 24, 2)).toThrow('do not fit');
  });
});

describe('l2n', () => {
  it('is numpy\'s in float32, to the bit, whatever the row length', () => {
    for (const [dim, v] of Object.entries(V.l2n)) {
      const got = l2n(floatsOf(v.x), Number(dim));
      expect(Array.from(got)).toEqual(Array.from(floatsOf(v.l2n)));
    }
  });
});

describe('the encoder\'s name', () => {
  it('is embed/onnx.Onnx\'s: "onnx:<stem>-<sha8>"', async () => {
    for (const [file, want] of Object.entries(V.name.names)) expect(await encoderName(file, bytesOf(V.name.bytes))).toBe(want);
    expect(stem('C:\\models\\embedder-v1.fp16.onnx')).toBe('embedder-v1.fp16');
  });
});

describe('OnnxEncoder', () => {
  /** A stand-in graph: each row is the crop's mean red, green and blue, then zeros; each run's size is logged. */
  const standIn = (runs: number[]) => async (x: Float32Array, n: number) => {
    runs.push(n);
    const plane = IMG_SIZE * IMG_SIZE;
    const out = new Float32Array(n * DIM);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let p = 0; p < plane; p++) s += x[(i * 3 + c) * plane + p]!;
        out[i * DIM + c] = s / plane + 1;
      }
    }
    return out;
  };
  const crop = (v: number, w = 30, h = 50) => {
    const im = rgbImage(w, h);
    im.data.fill(v);
    return im;
  };

  it('runs fixed batches of 8, the last padded, and gives one unit row a crop', async () => {
    const runs: number[] = [];
    const enc = new OnnxEncoder('onnx:standin-00000000', standIn(runs));
    expect([enc.batch, enc.dim, enc.imgSize, BATCH]).toEqual([8, 256, 224, 8]);
    const crops = Array.from({ length: 12 }, (_, i) => crop(10 * i));
    const rows = await enc.embed(crops);
    expect(runs).toEqual([8, 8]);
    expect(rows.length).toBe(12 * DIM);
    for (let i = 0; i < 12; i++) {
      const row = rows.subarray(i * DIM, (i + 1) * DIM);
      expect(Math.hypot(...row)).toBeCloseTo(1, 6);
    }
    // a crop's row does not depend on what shares its batch
    const alone = await enc.embed([crops[9]!]);
    expect(Array.from(alone)).toEqual(Array.from(rows.subarray(9 * DIM, 10 * DIM)));
    expect(await enc.embed([])).toEqual(new Float32Array(0));
  });

  it('refuses a net that gives the wrong number of values', async () => {
    const enc = new OnnxEncoder('x', async () => new Float32Array(5), { batch: 2 });
    await expect(enc.embed([crop(1)])).rejects.toThrow('gave 5 values');
  });
});
