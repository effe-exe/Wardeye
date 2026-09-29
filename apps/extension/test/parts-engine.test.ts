import { describe, expect, it } from 'vitest';
import { layouts } from '@rifteye/engine';
import { sha256Hex, type Gallery } from '../src/assets';
import { loadParts } from '../src/parts-engine';
import type { LoadContext, Trace } from '../src/parts';
import { Timer } from '../src/timer';
import { picture } from './fakes';

/** onnxruntime-web as far as the engine uses it, with sessions that answer as the two graphs do: no card in any
 * tile, and every crop the same row. */
function fakeOrt(failWebgpu = false) {
  const opened: { ep: string; batch: number | undefined; size: number }[] = [];
  const ort = {
    env: { wasm: {} as Record<string, unknown> },
    Tensor: class {
      constructor(readonly type: string, readonly data: Float32Array, readonly dims: readonly number[]) {}
    },
    InferenceSession: {
      async create(bytes: Uint8Array, options: { executionProviders?: string[]; freeDimensionOverrides?: { batch: number } } = {}) {
        const ep = options.executionProviders?.[0] ?? '';
        if (failWebgpu && ep === 'webgpu') throw new Error('GridSample is not supported');
        opened.push({ ep, batch: options.freeDimensionOverrides?.batch, size: bytes.length });
        return {
          inputNames: [],
          outputNames: [],
          async run(feeds: Record<string, { dims: readonly number[] }>) {
            if (feeds.tiles) {
              const n = feeds.tiles.dims[0]!;
              return {
                pred_logits: { data: new Float32Array(n * 100 * 2).fill(-20), dims: [n, 100, 2], type: 'float32' },
                pred_boxes: { data: new Float32Array(n * 100 * 4).fill(0.5), dims: [n, 100, 4], type: 'float32' },
                pred_keypoints: { data: new Float32Array(n * 100 * 64), dims: [n, 100, 8, 8], type: 'float32' },
              };
            }
            const n = feeds.crops!.dims[0]!;
            const rows = new Float32Array(n * 256);
            for (let i = 0; i < n; i++) rows[i * 256] = 1;
            return { embedding: { data: rows, dims: [n, 256], type: 'float32' } };
          },
          async release() {},
        };
      },
    },
  };
  return { ort, opened };
}

const EMBEDDER = Uint8Array.from([1, 2, 3, 4, 5]);

async function setup(over: { sha?: string; failWebgpu?: boolean; detector?: 'fp16' | 'fp32'; embedder?: 'fp16' | 'fp32'; runtime?: 'webgpu' | 'wasm'; trace?: Trace } = {}) {
  const { ort, opened } = fakeOrt(over.failWebgpu);
  const reads: string[] = [];
  const dim = 256;
  const gallery: Gallery = {
    index: {
      format: 1, encoder: 'onnx:embedder-v1-x', model: 'embedder-v1', sha256: over.sha ?? (await sha256Hex(EMBEDDER)), fp16_sha256: over.sha ?? (await sha256Hex(EMBEDDER)), dim,
      dtype: 'float16', levels: [100, 120, 130, 140, 160], rows: ['A-1', 'B-2'],
    },
    rows: [{ printing_id: 'A-1', card_id: 'a', name: 'A', type: 'Unit' }, { printing_id: 'B-2', card_id: 'b', name: 'B', type: 'Spell' }],
    levels: new Map([100, 120, 130, 140, 160].map((l) => [l, new Float32Array(2 * dim)])),
  };
  const timer = new Timer(() => performance.now());
  const ctx: LoadContext = {
    ort: ort as unknown as LoadContext['ort'],
    pkg: {
      format: 1, runtime: 'auto', data: 'data/',
      detector: { id: 'detector-v0', fp16: 'models/detector-v0.fp16.onnx', fp32: 'models/detector-v0.onnx' },
      embedder: { id: 'embedder-v1', fp16: 'models/embedder-v1.fp16.onnx', fp32: 'models/embedder-v1.onnx' },
    },
    attempt: { runtime: over.runtime ?? 'wasm', detector: over.detector ?? 'fp32', embedder: over.embedder ?? 'fp32' },
    read: async (path) => {
      reads.push(path);
      return path.includes('embedder') ? EMBEDDER : Uint8Array.from([9]);
    },
    decode: async () => picture(),
    gallery,
    timer,
    fps: 2,
    progress: () => {},
    ...(over.trace ? { trace: over.trace } : {}),
  };
  return { ctx, reads, opened, timer };
}

describe('the real parts, on an onnxruntime-web that answers like the models', () => {
  it('load the model files of the way chosen, one session each, fixed to the batch the engine asks', async () => {
    const { ctx, reads, opened } = await setup();
    const parts = await loadParts(ctx);
    expect(reads).toEqual(['models/detector-v0.onnx', 'models/embedder-v1.onnx']);
    expect(opened).toEqual([{ ep: 'wasm', batch: 1, size: 1 }, { ep: 'wasm', batch: 8, size: EMBEDDER.length }]);
    expect(parts.presets().map((l) => l.name)).toEqual(['la-rq', 'plusrb', 'shenyang']);
    await parts.dispose();
  });

  it('take each model in its own precision: on WebGPU the detector in float32, whatever the GPU, and the embedder in float16', async () => {
    const { ctx, reads, opened } = await setup({ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' });
    await loadParts(ctx);
    expect(reads).toEqual(['models/detector-v0.onnx', 'models/embedder-v1.fp16.onnx']);
    expect(opened.map((o) => o.ep)).toEqual(['webgpu', 'webgpu']);
  });

  it('run a float16 file on WASM too, when that is all the package holds', async () => {
    const { ctx, reads, opened } = await setup({ runtime: 'wasm', detector: 'fp16', embedder: 'fp16' });
    await loadParts(ctx);
    expect(reads).toEqual(['models/detector-v0.fp16.onnx', 'models/embedder-v1.fp16.onnx']);
    expect(opened.map((o) => o.ep)).toEqual(['wasm', 'wasm']);
  });

  it('check the float16 embedder against the float16 file\'s hash in the gallery, and the float32 one against the float32 file\'s', async () => {
    const good = await sha256Hex(EMBEDDER);
    const { ctx } = await setup({ embedder: 'fp16' });
    ctx.gallery.index.sha256 = 'a'.repeat(64); // the float32 file's: another file
    await expect(loadParts(ctx)).resolves.toBeDefined(); // the float16 file is the one the gallery's index names for it
    const other = await setup({ embedder: 'fp32' });
    other.ctx.gallery.index.sha256 = 'a'.repeat(64);
    other.ctx.gallery.index.fp16_sha256 = good;
    await expect(loadParts(other.ctx)).rejects.toThrow('is not the one the gallery was made with');
  });

  it('refuse an embedder that is not the one the gallery was made with', async () => {
    const { ctx } = await setup({ sha: 'f'.repeat(64) });
    await expect(loadParts(ctx)).rejects.toThrow('is not the one the gallery was made with');
  });

  it('fail, rather than run on the CPU, when the WebGPU build cannot make a session: the document tries the WASM build', async () => {
    const { ctx } = await setup({ runtime: 'webgpu', failWebgpu: true });
    await expect(loadParts(ctx)).rejects.toThrow('WebGPU failed: GridSample is not supported');
  });

  it('make a board for a table, of the layout and the gallery levels the live runner uses, and read a frame of it', async () => {
    const boxes: number[] = [];
    const embeds: number[] = [];
    const { ctx, timer } = await setup({ trace: { boxes: (_t, b) => void boxes.push(b.length), embed: (n) => void embeds.push(n) } });
    const parts = await loadParts(ctx);
    expect(embeds).toEqual([]); // the warm-up runs are not traced
    const la = layouts.LAYOUTS['la-rq'];
    const board = parts.board(la);
    timer.reset();
    const { state, events } = await board.step(0, picture(1920, 1080, la.mat!)); // the table camera: its mat fills the window
    expect(state).toMatchObject({ status: 'live', title: la.title, layout: { name: 'la-rq' }, frame: { width: 1920, height: 1080 }, tracks: [] });
    expect(events).toEqual([]);
    expect(boxes).toEqual([0]); // the finder was asked once, and found no card
    expect(timer.get('detect')).toBeGreaterThan(0); // and its time was counted
  });

  it('look for a layout with the detector, and find none in frames that show no table', async () => {
    const { ctx } = await setup();
    const parts = await loadParts(ctx);
    expect(await parts.findLayout(Array.from({ length: 5 }, () => picture(320, 180, [200, 200, 200])))).toBeNull();
  });
});
