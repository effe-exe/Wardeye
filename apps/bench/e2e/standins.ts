// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Stand-in models for the bench's browser test, made at test time (no ONNX or .bin file is committed): tiny
// graphs written with src/tiny-onnx.ts, their check files, and the manifests that describe them in the bench
// contract. "Expected" values are computed here in plain JavaScript, standing in for PyTorch.

import { linkSync, copyFileSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { encodeModel, type TinyGraph } from '../src/tiny-onnx';

const ITEM = 3 * 8 * 8;

/** n items of made-up pixel values (0..255, not whole numbers), the same every time. */
export function pixels(n: number): Float32Array {
  const out = new Float32Array(n * ITEM);
  let seed = 7;
  for (let i = 0; i < out.length; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    out[i] = (seed / 2 ** 32) * 255;
  }
  return out;
}

const bytes = (f: Float32Array): Uint8Array => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
const scaled = (x: Float32Array, k: number, add = 0): Float32Array => x.map((v) => Math.fround(Math.fround(v * k) + add));

const shape = ['batch', 3, 8, 8];
const scalar = (name: string, value: number, type: 'float32' | 'float16' = 'float32') => ({ name, type, dims: [], values: [value] });

/** y = x * 2 (batch x 3 x 8 x 8) and z = flatten(x + 1) (batch x 192): two outputs of different shapes. */
function detectorGraph(): TinyGraph {
  return {
    inputs: [{ name: 'x', type: 'float32', shape }],
    outputs: [
      { name: 'y', type: 'float32', shape },
      { name: 'z', type: 'float32', shape: ['batch', ITEM] },
    ],
    constants: [scalar('two', 2), scalar('one', 1)],
    nodes: [
      { op: 'Mul', inputs: ['x', 'two'], outputs: ['y'] },
      { op: 'Add', inputs: ['x', 'one'], outputs: ['t'] },
      { op: 'Flatten', inputs: ['t'], outputs: ['z'], ints: { axis: 1 } },
    ],
  };
}

/** embedding = flatten(crops * 3): batch x 192; the fp16 build computes in half precision and casts back. */
function embedderGraph(half: boolean): TinyGraph {
  const io = {
    inputs: [{ name: 'crops', type: 'float32' as const, shape }],
    outputs: [{ name: 'embedding', type: 'float32' as const, shape: ['batch', ITEM] }],
  };
  if (!half) {
    return {
      ...io,
      constants: [scalar('three', 3)],
      nodes: [
        { op: 'Mul', inputs: ['crops', 'three'], outputs: ['m'] },
        { op: 'Flatten', inputs: ['m'], outputs: ['embedding'], ints: { axis: 1 } },
      ],
    };
  }
  return {
    ...io,
    constants: [scalar('three', 3, 'float16')],
    nodes: [
      { op: 'Cast', inputs: ['crops'], outputs: ['h'], ints: { to: 10 } },
      { op: 'Mul', inputs: ['h', 'three'], outputs: ['m'] },
      { op: 'Flatten', inputs: ['m'], outputs: ['f'], ints: { axis: 1 } },
      { op: 'Cast', inputs: ['f'], outputs: ['embedding'], ints: { to: 1 } },
    ],
  };
}

type Files = Record<string, Uint8Array | string>;

const json = (v: unknown): string => JSON.stringify(v, null, 2);

/** Everything that goes into a models/ folder: name -> content. */
export function standInFiles(): Files {
  const files: Files = {};
  const x2 = pixels(2);
  const x4 = pixels(4);

  // standin-detector: maxabs over two outputs; its fp16 file is left out ("not included")
  files['standin-detector.onnx'] = encodeModel(detectorGraph());
  files['standin-detector.check.input.bin'] = bytes(x2);
  files['standin-detector.check.y.bin'] = bytes(scaled(x2, 2));
  files['standin-detector.check.z.bin'] = bytes(scaled(x2, 1, 1));
  files['standin-detector.bench.json'] = json({
    id: 'standin-detector',
    title: 'Stand-in detector (x2, x+1)',
    variants: [
      { precision: 'fp32', file: 'standin-detector.onnx' },
      { precision: 'fp16', file: 'standin-detector.fp16.onnx' },
    ],
    input: { name: 'x', dtype: 'float32', shape, note: 'made-up pixels' },
    outputs: [
      { name: 'y', shape },
      { name: 'z', shape: ['batch', ITEM] },
    ],
    batches: [1, 2, 4],
    per_frame: { items: 3, note: 'three tiles a frame' },
    check: {
      batch: 2,
      input: 'standin-detector.check.input.bin',
      expected: { y: 'standin-detector.check.y.bin', z: 'standin-detector.check.z.bin' },
      metric: 'maxabs',
      tolerance: { fp32: 1e-6, fp16: { y: 1, z: 1 } },
      note: 'exact in float32',
    },
  });

  // standin-embedder: cosine, both precisions present
  files['standin-embedder.onnx'] = encodeModel(embedderGraph(false));
  files['standin-embedder.fp16.onnx'] = encodeModel(embedderGraph(true));
  files['standin-embedder.check.input.bin'] = bytes(x4);
  files['standin-embedder.check.embedding.bin'] = bytes(scaled(x4, 3));
  files['standin-embedder.bench.json'] = json({
    id: 'standin-embedder',
    title: 'Stand-in embedder (x3)',
    variants: [
      { precision: 'fp32', file: 'standin-embedder.onnx' },
      { precision: 'fp16', file: 'standin-embedder.fp16.onnx' },
    ],
    input: { name: 'crops', dtype: 'float32', shape },
    outputs: [{ name: 'embedding', shape: ['batch', ITEM] }],
    batches: [1, 4],
    per_frame: { items: 6, note: 'six crops a frame' },
    check: {
      batch: 4,
      input: 'standin-embedder.check.input.bin',
      expected: { embedding: 'standin-embedder.check.embedding.bin' },
      metric: 'cosine',
      tolerance: { fp32: 0.9999, fp16: 0.999 },
    },
  });

  // standin-wrong: a good model whose expected values are not what it computes: its check must FAIL
  files['standin-wrong.onnx'] = encodeModel(embedderGraph(false));
  files['standin-wrong.check.input.bin'] = bytes(x2);
  files['standin-wrong.check.embedding.bin'] = bytes(scaled(x2, -3));
  files['standin-wrong.bench.json'] = json({
    id: 'standin-wrong',
    title: 'Stand-in with the wrong expected values',
    variants: [{ precision: 'fp32', file: 'standin-wrong.onnx' }],
    input: { name: 'crops', dtype: 'float32', shape },
    outputs: [{ name: 'embedding', shape: ['batch', ITEM] }],
    batches: [2],
    check: {
      batch: 2,
      input: 'standin-wrong.check.input.bin',
      expected: { embedding: 'standin-wrong.check.embedding.bin' },
      metric: 'cosine',
      tolerance: { fp32: 0.9999 },
    },
  });

  // standin-broken: the model file is not an ONNX model: its rows must say so, and the bench must go on
  files['standin-broken.onnx'] = new TextEncoder().encode('this is not an ONNX model');
  files['standin-broken.check.input.bin'] = bytes(x2);
  files['standin-broken.check.embedding.bin'] = bytes(scaled(x2, 3));
  files['standin-broken.bench.json'] = json({
    id: 'standin-broken',
    title: 'Stand-in whose model file is garbage',
    variants: [{ precision: 'fp32', file: 'standin-broken.onnx' }],
    input: { name: 'crops', dtype: 'float32', shape },
    outputs: [{ name: 'embedding', shape: ['batch', ITEM] }],
    batches: [1],
    check: {
      batch: 2,
      input: 'standin-broken.check.input.bin',
      expected: { embedding: 'standin-broken.check.embedding.bin' },
      metric: 'cosine',
      tolerance: { fp32: 0.9999 },
    },
  });

  // standin-invalid: a manifest that says too little: its row must say why, and the others must run
  files['standin-invalid.bench.json'] = json({ id: 'standin-invalid', variants: [] });

  files['index.json'] = json([
    'standin-detector.bench.json',
    'standin-embedder.bench.json',
    'standin-wrong.bench.json',
    'standin-broken.bench.json',
    'standin-invalid.bench.json',
  ]);
  return files;
}

export function writeFiles(dir: string, files: Files): void {
  mkdirSync(dir, { recursive: true });
  for (const [name, content] of Object.entries(files)) writeFileSync(join(dir, name), content);
}

/** A copy of a folder tree that costs no disk: files are hard-linked (copied where that is not possible). */
export function mirror(from: string, to: string): void {
  mkdirSync(to, { recursive: true });
  for (const name of readdirSync(from)) {
    const a = join(from, name);
    const b = join(to, name);
    if (statSync(a).isDirectory()) {
      mirror(a, b);
    } else {
      try {
        linkSync(a, b);
      } catch {
        copyFileSync(a, b);
      }
    }
  }
}
