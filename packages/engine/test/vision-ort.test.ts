// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// ort.ts: which runtime a browser gets (fake modules and GPUs), and sessions made with the real onnxruntime-web
// (its Node build, WASM, one thread) on stand-in graphs written with the bench's tiny ONNX writer: the fixed batch,
// the runners, and a Detector and an encoder on top.

import * as ort from 'onnxruntime-web';
import { afterEach, describe, expect, it } from 'vitest';
import { encodeModel, type TinyGraph } from '../../../apps/bench/src/tiny-onnx';
import { TILE } from '../src/detector';
import { DIM, encoderName, IMG_SIZE } from '../src/embedder';
import { rgbImage } from '../src/image';
import {
  chooseRuntime,
  embedRunner,
  openDetector,
  openEncoder,
  openModel,
  sessionOptions,
  tileRunner,
  type GpuLike,
  type ModelFiles,
  type OrtModule,
  type OrtSession,
  type Runtime,
} from '../src/ort';

ort.env.wasm.numThreads = 1;
const real: OrtModule = ort; // onnxruntime-web itself fits the engine's interface

/** input -> GlobalAveragePool -> Flatten -> MatMul by a [3, k] constant, for each output: small, fixed, checkable. */
function standIn(input: string, side: number, outputs: Record<string, number>): Uint8Array {
  const g: TinyGraph = {
    inputs: [{ name: input, type: 'float32', shape: ['batch', 3, side, side] }],
    outputs: Object.entries(outputs).map(([name, k]) => ({ name, type: 'float32' as const, shape: ['batch', k] })),
    constants: Object.entries(outputs).map(([name, k]) => ({ name: `w_${name}`, type: 'float32' as const, dims: [3, k], values: weights(k) })),
    nodes: [
      { op: 'GlobalAveragePool', inputs: [input], outputs: ['pooled'] },
      { op: 'Flatten', inputs: ['pooled'], outputs: ['flat'], ints: { axis: 1 } },
      ...Object.keys(outputs).map((name) => ({ op: 'MatMul', inputs: ['flat', `w_${name}`], outputs: [name] })),
    ],
  };
  return encodeModel(g);
}

function weights(k: number): number[] {
  return Array.from({ length: 3 * k }, (_, i) => ((i * 7919) % 101) / 101 - 0.3);
}

const fakeOrt = (create: (bytes: Uint8Array, o: unknown) => Promise<OrtSession>): OrtModule => ({
  env: { wasm: {} },
  InferenceSession: { create },
  Tensor: ort.Tensor as unknown as OrtModule['Tensor'],
});
const gpu = (features: string[] | null): GpuLike => ({ requestAdapter: async () => (features ? { features: new Set(features) } : null) });
const files = (fp16 = true): ModelFiles => ({
  fp32: { name: 'm.onnx', bytes: () => new Uint8Array([32]) },
  ...(fp16 ? { fp16: { name: 'm.fp16.onnx', bytes: () => new Uint8Array([16]) } } : {}),
});

const W = WebAssembly as unknown as { Suspending?: unknown };
const hadJspi = 'Suspending' in W;
const withJspi = () => {
  if (!W.Suspending) W.Suspending = function Suspending() {};
};
afterEach(() => {
  if (!hadJspi) delete W.Suspending;
});

describe('chooseRuntime', () => {
  const wasm = fakeOrt(async () => ({}) as OrtSession);
  const webgpu = fakeOrt(async () => ({}) as OrtSession);

  it('takes WebGPU and float16 on a GPU with shader-f16, float32 on one without', async () => {
    withJspi();
    expect(await chooseRuntime({ webgpu, wasm, gpu: gpu(['shader-f16']) })).toMatchObject({ ep: 'webgpu', precision: 'fp16', ort: webgpu });
    expect(await chooseRuntime({ webgpu, wasm, gpu: gpu([]) })).toMatchObject({ ep: 'webgpu', precision: 'fp32', ort: webgpu });
  });

  it('falls back to WASM and float32, and says why', async () => {
    withJspi();
    expect(await chooseRuntime({ wasm })).toMatchObject({ ep: 'wasm', precision: 'fp32', reason: 'no WebGPU build given' });
    expect(await chooseRuntime({ webgpu, wasm, gpu: null })).toMatchObject({ ep: 'wasm', reason: 'no WebGPU in this browser' });
    expect(await chooseRuntime({ webgpu, wasm, gpu: gpu(null) })).toMatchObject({ ep: 'wasm', reason: 'no WebGPU adapter' });
    expect(await chooseRuntime({ webgpu, wasm, gpu: gpu(['shader-f16']), wasmOnly: true })).toMatchObject({ ep: 'wasm' });
    delete W.Suspending;
    expect(await chooseRuntime({ webgpu, wasm, gpu: gpu(['shader-f16']) })).toMatchObject({ ep: 'wasm', reason: 'no WebAssembly JSPI in this browser' });
  });

  it('points both builds at the runtime files and sets the threads', async () => {
    withJspi();
    const a = fakeOrt(async () => ({}) as OrtSession);
    const b = fakeOrt(async () => ({}) as OrtSession);
    await chooseRuntime({ webgpu: a, wasm: b, wasmPaths: 'chrome-extension://x/ort/', threads: 2, gpu: gpu([]) });
    for (const m of [a, b]) expect(m.env.wasm).toEqual({ wasmPaths: 'chrome-extension://x/ort/', numThreads: 2, proxy: false });
  });
});

describe('openModel', () => {
  it('fixes the batch axis, and opens the float16 file only on a float16 runtime', async () => {
    const seen: [number, unknown][] = [];
    const m = fakeOrt(async (bytes, o) => {
      seen.push([bytes[0]!, o]);
      return {} as OrtSession;
    });
    const rt: Runtime = { ep: 'webgpu', precision: 'fp16', ort: m, wasm: m, reason: '' };
    expect(await openModel(rt, files(), 1)).toMatchObject({ ep: 'webgpu', precision: 'fp16', file: 'm.fp16.onnx', batch: 1, fallback: null, sha256: null });
    expect(await openModel({ ...rt, precision: 'fp32' }, files(), 8)).toMatchObject({ precision: 'fp32', file: 'm.onnx' });
    expect(await openModel(rt, files(false), null)).toMatchObject({ precision: 'fp32', file: 'm.onnx', batch: null });
    expect(seen.map(([b]) => b)).toEqual([16, 32, 32]);
    expect(seen[0]![1]).toEqual({ executionProviders: ['webgpu'], graphOptimizationLevel: 'all', freeDimensionOverrides: { batch: 1 } });
    expect(sessionOptions('wasm', null)).toEqual({ executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  });

  it('falls back to WASM and the float32 file when the WebGPU session fails', async () => {
    const gpuOrt = fakeOrt(async () => {
      throw new Error('no GridSample');
    });
    const wasmOrt = fakeOrt(async () => ({}) as OrtSession);
    const rt: Runtime = { ep: 'webgpu', precision: 'fp16', ort: gpuOrt, wasm: wasmOrt, reason: '' };
    const m = await openModel(rt, files(), 1, true);
    expect(m).toMatchObject({ ep: 'wasm', precision: 'fp32', ort: wasmOrt, file: 'm.onnx', fallback: 'WebGPU failed: no GridSample' });
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('the runners', () => {
  it('run one session\'s calls one after another, even when they overlap', async () => {
    let busy = 0;
    let most = 0;
    const session = {
      inputNames: ['crops'],
      outputNames: ['embedding'],
      run: async () => {
        busy++;
        most = Math.max(most, busy);
        await new Promise((r) => setTimeout(r, 5));
        busy--;
        return { embedding: { data: new Float32Array(2 * DIM), dims: [2, DIM], type: 'float32' } };
      },
      release: async () => {},
    } as OrtSession;
    const run = embedRunner(real, session, 8);
    const rows = await Promise.all([run(new Float32Array(2 * 3 * 64), 2), run(new Float32Array(2 * 3 * 64), 2), run(new Float32Array(2 * 3 * 64), 2)]);
    expect(most).toBe(1);
    expect(rows.map((r) => r.length)).toEqual([2 * DIM, 2 * DIM, 2 * DIM]);
  });

  it('refuse an output that is missing or not float32', async () => {
    const session = { inputNames: [], outputNames: [], run: async () => ({ embedding: { data: new Uint8Array(4), dims: [4], type: 'uint8' } }), release: async () => {} } as OrtSession;
    await expect(embedRunner(real, session, 8)(new Float32Array(3 * 64), 1)).rejects.toThrow('output "embedding" is uint8, not float32');
    await expect(tileRunner(real, session)(new Float32Array(3 * TILE * TILE), 1)).rejects.toThrow('the model has no output "pred_logits"');
  });
});

describe('sessions on the real onnxruntime-web (WASM)', () => {
  it('an encoder: fixed batch 8, named after its file, rows as the graph gives them, normalised', async () => {
    const bytes = standIn('crops', IMG_SIZE, { embedding: DIM });
    const rt = await chooseRuntime({ wasm: real });
    const { encoder, model } = await openEncoder(rt, { fp32: { name: 'standin-embedder.onnx', bytes: () => bytes } });
    expect(model).toMatchObject({ ep: 'wasm', precision: 'fp32', batch: 8 });
    expect(encoder.name).toBe(await encoderName('standin-embedder.onnx', bytes));
    const crops = [1, 2, 3].map((v) => {
      const im = rgbImage(40, 60);
      for (let i = 0; i < im.data.length; i++) im.data[i] = (v * 50 + i) % 256;
      return im;
    });
    const rows = await encoder.embed(crops);
    expect(rows.length).toBe(3 * DIM);
    for (let i = 0; i < 3; i++) expect(Math.hypot(...rows.subarray(i * DIM, (i + 1) * DIM))).toBeCloseTo(1, 5);
    // the session holds batch 8 only: another shape is refused
    const x = new ort.Tensor('float32', new Float32Array(3 * 3 * IMG_SIZE * IMG_SIZE), [3, 3, IMG_SIZE, IMG_SIZE]);
    await expect(model.session.run({ crops: x })).rejects.toThrow();
    await model.session.release();
  });

  it('a detector: one tile a run, padded, the three outputs decoded', async () => {
    const bytes = standIn('tiles', TILE, { pred_logits: 200, pred_boxes: 400, pred_keypoints: 6400 });
    const rt = await chooseRuntime({ wasm: real });
    const { detector, model } = await openDetector(rt, { fp32: { name: 'standin-detector.onnx', bytes: () => bytes } });
    expect(model).toMatchObject({ ep: 'wasm', batch: 1 });
    expect([detector.batch, detector.pad]).toEqual([1, true]);
    const tile = rgbImage(TILE, TILE);
    tile.data.fill(128);
    const per = await detector.detectTiles([tile, tile], 0);
    expect(per).toHaveLength(2);
    expect(per[0]).toEqual(per[1]);
    expect(per[0]!.length).toBeGreaterThan(0);
    await model.session.release();
  });
});
