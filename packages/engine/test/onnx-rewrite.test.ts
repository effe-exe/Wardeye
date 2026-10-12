// onnx-rewrite.ts: a lower-bound Clip written as Max, checked with the real onnxruntime-web (WASM) on a tiny graph, and on
// the real embedder files when they are on this machine (RIFTEYE_DATA): the bytes must be the ones ml's clip_to_max writes.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { encodeModel, type TinyGraph } from '../../../apps/bench/src/tiny-onnx';
import { clipToMax } from '../src/onnx-rewrite';

ort.env.wasm.numThreads = 1;

/** As in the embedder: each row's norm, clamped from below (F.normalize), and the rows divided by it. `upper` adds a bound above. */
function normalise(upper = false): Uint8Array {
  const g: TinyGraph = {
    inputs: [{ name: 'x', type: 'float32', shape: ['n', 4] }],
    outputs: [{ name: 'y', type: 'float32', shape: ['n', 4] }],
    constants: [
      { name: 'eps', type: 'float32', dims: [], values: [1e-6] },
      { name: 'big', type: 'float32', dims: [], values: [1e6] },
    ],
    nodes: [
      { op: 'Mul', inputs: ['x', 'x'], outputs: ['sq'] },
      { op: 'ReduceSum', inputs: ['sq'], outputs: ['s'], ints: { keepdims: 1 } },
      { op: 'Sqrt', inputs: ['s'], outputs: ['norm'] },
      { op: 'Clip', inputs: upper ? ['norm', 'eps', 'big'] : ['norm', 'eps'], outputs: ['clamped'] },
      { op: 'Div', inputs: ['x', 'clamped'], outputs: ['y'] },
    ],
  };
  return encodeModel(g);
}

async function run(model: Uint8Array, x: Float32Array): Promise<Float32Array> {
  const s = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
  const out = await s.run({ x: new ort.Tensor('float32', x, [x.length / 4, 4]) });
  await s.release();
  return out['y']!.data as Float32Array;
}

const ops = (model: Uint8Array): string[] => ['Clip', 'Max'].filter((op) => Buffer.from(model).includes(Buffer.from(op)));

describe('a Clip with a lower bound alone, on WebGPU', () => {
  it('is a Max: the same values, and nothing else in the file changes but its name and the lengths around it', async () => {
    const before = normalise();
    const after = clipToMax(before);
    expect(ops(before)).toEqual(['Clip']);
    expect(ops(after)).toEqual(['Max']);
    expect(after.length).toBe(before.length - 1); // "Clip" to "Max"
    // a zero row too: its norm is clamped to eps, and the row stays zero
    const x = Float32Array.from([0, 0, 0, 0, 1, 2, 3, 4, -1, 0.5, 0, 2]);
    expect(Array.from(await run(after, x))).toEqual(Array.from(await run(before, x)));
  });

  it('leaves a Clip with an upper bound as it is, and a file with none the same array', () => {
    const both = normalise(true);
    expect(clipToMax(both)).toBe(both);
  });

  it('hands bytes that are not ONNX to onnxruntime as they are, for it to say what is wrong', () => {
    const junk = Uint8Array.from([0x0a, 0xff, 0xff, 0xff]); // a length that runs past the end
    expect(clipToMax(junk)).toBe(junk);
  });
});

const DATA = process.env['RIFTEYE_DATA'] ?? '';
const ONNX = join(DATA, 'models', 'onnx');
const sha = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');

describe.skipIf(!DATA || !existsSync(join(ONNX, 'embedder-v1.onnx')))('the real embedder files', () => {
  it('become, byte for byte, the files ml/rifteye_ml/embed/onnx.py clip_to_max writes', () => {
    // the files as published (Hugging Face, D-038), and their rewrite as onnx.save wrote it (each with its one Clip as Max)
    for (const [file, published, rewritten] of [
      ['embedder-v1.onnx', '3f0a857c8cbbfc7b49c81fb8b44ba4290f492b488f25b11c5cf2f7e827b2ba28', 'cf25d3586dd4c76a7fc036718b777f2b0c735471a2ffaf927915f06129902268'],
      ['embedder-v1.fp16.onnx', '40a9c263c8339a101ebce32c85f3a9580867e7481c4f8a61250a82094c184a1a', 'fcad31f0c6beb22d255cd2d46f3c4764e39f05dac6779d60fa0d985a957d91b5'],
    ] as const) {
      const bytes = new Uint8Array(readFileSync(join(ONNX, file)));
      expect(sha(bytes), file).toBe(published);
      expect(sha(clipToMax(bytes)), file).toBe(rewritten);
    }
  });
});
