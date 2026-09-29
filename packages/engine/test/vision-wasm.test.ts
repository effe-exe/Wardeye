// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The real models on onnxruntime-web's WASM build in Node, through ort.ts, against Python's ONNX Runtime CPU on the
// same pixels: the detector on the three LA frames decoded to raw RGB, the embedder on the 40 crops of
// test/gen/vision_parity.py. It needs the private fixtures and models (D-006), and takes minutes of CPU, so it runs
// only when both are named (e2e/vision.spec.ts checks the same in Chromium):
//
//   RIFTEYE_M3=~/rifteye-data/m3 RIFTEYE_MODELS=~/rifteye-data/models/onnx npx vitest run packages/engine/test/vision-wasm.test.ts
//
// Two WASM threads at most.

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ortWeb from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { rgbImage } from '../src/image';
import { chooseRuntime, openDetector, openEncoder, type OrtModule } from '../src/ort';
import type { Detection } from '../src/types';

const M3 = process.env.RIFTEYE_M3;
const VIS = M3 ? join(M3, 'fixtures', 'vision') : '';
const MODELS = process.env.RIFTEYE_MODELS ?? '';
const file = (name: string) => ({ name, bytes: () => new Uint8Array(readFileSync(join(MODELS, name))) });
const has = (...p: string[]) => Boolean(M3 && MODELS) && existsSync(join(...p));
const RGB = ['f0000', 'f0120', 'f0239'];

interface PyDet {
  cls: Detection['cls'];
  score: number;
  quad: number[];
}

/** Each expected card matched to a found one of its class with the closest corners; the worst differences. */
function match(got: readonly Detection[], want: readonly PyDet[]): { missing: number; extra: number; corner: number; score: number } {
  const pairs: [number, number, number][] = [];
  want.forEach((w, i) =>
    got.forEach((g, j) => {
      if (g.cls !== w.cls) return;
      const c = Math.max(...g.quad.map(([x, y], k) => Math.hypot(x - w.quad[2 * k]!, y - w.quad[2 * k + 1]!)));
      pairs.push([c, i, j]);
    }),
  );
  pairs.sort((a, b) => a[0] - b[0]);
  const wi = new Set<number>();
  const gj = new Set<number>();
  let corner = 0;
  let score = 0;
  for (const [c, i, j] of pairs) {
    if (wi.has(i) || gj.has(j) || c > 5) continue;
    wi.add(i);
    gj.add(j);
    corner = Math.max(corner, c);
    score = Math.max(score, Math.abs(got[j]!.score - want[i]!.score));
  }
  return { missing: want.length - wi.size, extra: got.length - gj.size, corner, score };
}

const threads = 2;

describe.skipIf(!has(MODELS, 'detector-v0.onnx') || !has(VIS, 'detector', 'index.json'))('the detector on WASM in Node', () => {
  it('finds Python\'s cards on the LA frames: scores within 1e-4, corners within 0.1 px', async () => {
    const rt = await chooseRuntime({ wasm: ortWeb as OrtModule, threads });
    const { detector, model } = await openDetector(rt, { fp32: file('detector-v0.onnx') });
    const times: number[] = [];
    for (const n of RGB) {
      const f = JSON.parse(readFileSync(join(VIS, 'detector', `la-${n}.json`), 'utf8')) as { window: number[]; card_px: number; detect: PyDet[] };
      const frame = rgbImage(1920, 1080, new Uint8Array(readFileSync(join(M3!, 'frames', 'la-final-rgb', `${n}.rgb`))));
      const t0 = performance.now();
      const got = await detector.detect(frame, f.window, f.card_px);
      times.push(performance.now() - t0);
      const m = match(got, f.detect);
      expect(m).toMatchObject({ missing: 0, extra: 0 });
      expect(m.corner).toBeLessThanOrEqual(0.1 + 1e-9);
      expect(m.score).toBeLessThanOrEqual(1e-4 + 1e-9);
    }
    console.log(`detector fp32 on WASM (Node, ${threads} threads): ${times.map((t) => t.toFixed(0)).join(', ')} ms a frame (one tile each, the first includes warm-up)`);
    await model.session.release();
  }, 600_000);
});

describe.skipIf(!has(MODELS, 'embedder-v1.onnx') || !has(VIS, 'crops', 'crops.json'))('the embedder on WASM in Node', () => {
  it('gives Python\'s rows for the 40 crops (cosine >= 0.99999) under Python\'s name', async () => {
    const c = JSON.parse(readFileSync(join(VIS, 'crops', 'crops.json'), 'utf8')) as { encoder: string; crops: { id: string; size: [number, number] }[] };
    const rt = await chooseRuntime({ wasm: ortWeb as OrtModule, threads });
    const { encoder, model } = await openEncoder(rt, { fp32: file('embedder-v1.onnx') });
    expect(encoder.name).toBe(c.encoder);
    const crops = c.crops.map((m) => rgbImage(m.size[0], m.size[1], new Uint8Array(readFileSync(join(VIS, 'crops', `${m.id}.rgb`)))));
    await encoder.embed(crops.slice(0, 8)); // warm-up
    const t0 = performance.now();
    const rows = await encoder.embed(crops);
    const ms = performance.now() - t0;
    const b = readFileSync(join(VIS, 'crops', 'embed.bin'));
    const want = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    let worst = 1;
    for (let i = 0; i < crops.length; i++) {
      let dot = 0;
      for (let k = 0; k < 256; k++) dot += rows[i * 256 + k]! * want[i * 256 + k]!;
      worst = Math.min(worst, dot);
    }
    console.log(`embedder fp32 on WASM (Node, ${threads} threads): ${(ms / 5).toFixed(0)} ms a batch of 8; lowest cosine ${worst.toFixed(8)}`);
    expect(worst).toBeGreaterThanOrEqual(0.99999);
    await model.session.release();
  }, 600_000);
});
