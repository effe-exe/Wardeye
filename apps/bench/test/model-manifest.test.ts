import { describe, expect, it } from 'vitest';
import { batchAxis, batchesOf, elementCount, fixedBatch, itemSize, outputCount, parseIndex, parseManifest, resolveOutShape, resolveShape } from '../src/model-manifest';

// the embedder manifest of the bench contract
const embedder = () => ({
  id: 'embedder-v1',
  title: 'Card embedder (DINOv2-S, 256-d)',
  variants: [
    { precision: 'fp32', file: 'embedder-v1.onnx' },
    { precision: 'fp16', file: 'embedder-v1.fp16.onnx' },
  ],
  input: { name: 'crops', dtype: 'float32', shape: ['batch', 3, 224, 224], note: 'RGB 0..255, NCHW' },
  outputs: [{ name: 'embedding', shape: ['batch', 256] }],
  batches: [1, 8, 16],
  per_frame: { items: 12, note: 'crops embedded per processed frame (median, LA final replay)' },
  check: {
    batch: 8,
    input: 'embedder-v1.check.input.bin',
    expected: { embedding: 'embedder-v1.check.embedding.bin' },
    metric: 'cosine',
    tolerance: { fp32: 0.9999, fp16: 0.999 },
    note: 'how the tolerance was chosen',
  },
});

describe('a manifest', () => {
  it('is read as the contract writes it', () => {
    const m = parseManifest(embedder());
    expect(m.id).toBe('embedder-v1');
    expect(m.title).toBe('Card embedder (DINOv2-S, 256-d)');
    expect(m.variants).toEqual([
      { precision: 'fp32', file: 'embedder-v1.onnx' },
      { precision: 'fp16', file: 'embedder-v1.fp16.onnx' },
    ]);
    expect(m.input).toEqual({ name: 'crops', dtype: 'float32', shape: ['batch', 3, 224, 224] });
    expect(m.outputs).toEqual([{ name: 'embedding', shape: ['batch', 256] }]);
    expect(m.batches).toEqual([1, 8, 16]);
    expect(m.perFrame).toBe(12);
    expect(m.check).toEqual({
      batch: 8,
      input: 'embedder-v1.check.input.bin',
      expected: { embedding: 'embedder-v1.check.embedding.bin' },
      metric: 'cosine',
      tolerance: { fp32: 0.9999, fp16: 0.999 },
    });
  });

  it('takes per-frame items that are not whole (a median over frames)', () => {
    expect(parseManifest({ ...embedder(), per_frame: { items: 11.5 } }).perFrame).toBe(11.5);
  });

  it('may give a tolerance per output, for a detector with several outputs', () => {
    const j = {
      ...embedder(),
      id: 'detector-v0',
      outputs: [
        { name: 'pred_logits', shape: ['batch', 300, 2] },
        { name: 'pred_boxes', shape: ['batch', 300, 4] },
      ],
      check: {
        batch: 2,
        input: 'd.in.bin',
        expected: { pred_logits: 'd.logits.bin', pred_boxes: 'd.boxes.bin' },
        metric: 'maxabs',
        tolerance: { fp32: 0.001, fp16: { pred_logits: 0.5, pred_boxes: 0.05 } },
      },
    };
    const m = parseManifest(j);
    expect(m.check?.metric).toBe('maxabs');
    expect(m.check?.tolerance).toEqual({ fp32: 0.001, fp16: { pred_logits: 0.5, pred_boxes: 0.05 } });
  });

  it('lets an output leave an axis open (the real detector\'s keypoints have a size the exporter left symbolic), but not an input', () => {
    const j = {
      ...embedder(),
      outputs: [{ name: 'pred_keypoints', shape: ['batch', 'Concatpred_keypoints_dim_1', 8, 8] }],
      check: { ...embedder().check, metric: 'maxabs', expected: { pred_keypoints: 'k.bin' }, tolerance: { fp32: 0.01, fp16: 0.1 } },
    };
    expect(parseManifest(j).outputs[0]!.shape).toEqual(['batch', 'Concatpred_keypoints_dim_1', 8, 8]);
    expect(() => parseManifest({ ...j, input: { name: 'crops', shape: ['batch', 'h', 4] } })).toThrow('input.shape[1]: expected a positive integer or "batch", got "h"');
    expect(() => parseManifest({ ...j, outputs: [{ name: 'pred_keypoints', shape: ['batch', 0] }] })).toThrow('outputs[0].shape[1]: expected a positive integer or "batch" or an axis name, got 0');
  });

  it('takes uint8 input, and float32 when the dtype is not said', () => {
    const j = embedder();
    expect(parseManifest({ ...j, input: { name: 'crops', shape: ['batch', 3, 4, 4] } }).input.dtype).toBe('float32');
    expect(parseManifest({ ...j, input: { name: 'crops', dtype: 'uint8', shape: ['batch', 3, 4, 4] } }).input.dtype).toBe('uint8');
    expect(() => parseManifest({ ...j, input: { name: 'crops', dtype: 'float64', shape: [1] } })).toThrow('embedder-v1: input.dtype: expected "float32" or "uint8", got "float64"');
  });

  it('may leave out the title, the per-frame items and the check', () => {
    const { title: _t, per_frame: _p, check: _c, ...rest } = embedder();
    const m = parseManifest(rest);
    expect(m.title).toBe('embedder-v1');
    expect(m.perFrame).toBeNull();
    expect(m.check).toBeNull();
  });

  it('names the field at fault', () => {
    const bad = (patch: Record<string, unknown>) => () => parseManifest({ ...embedder(), ...patch });
    expect(() => parseManifest(null)).toThrow('manifest: expected an object');
    expect(() => parseManifest({})).toThrow('id: expected a non-empty string');
    expect(bad({ variants: [] })).toThrow('embedder-v1: variants: expected a non-empty list');
    expect(bad({ variants: [{ precision: 'fp32', file: '../x.onnx' }] })).toThrow('variants[0].file: expected a plain file name, got "../x.onnx"');
    expect(bad({ variants: [{ precision: 'fp32', file: 'models/x.onnx' }] })).toThrow('plain file name');
    expect(bad({ input: { name: 'crops', shape: ['batch', 0, 4] } })).toThrow('input.shape[1]: expected a positive integer or "batch", got 0');
    expect(bad({ input: { name: 'crops', shape: ['n', 3] } })).toThrow('got "n"');
    expect(bad({ outputs: [] })).toThrow('outputs: expected a non-empty list');
    expect(bad({ batches: [] })).toThrow('batches: expected a non-empty list of positive integers');
    expect(bad({ batches: [1, 2.5] })).toThrow('batches');
    expect(bad({ per_frame: { items: 0 } })).toThrow('per_frame.items: expected a positive number');
    expect(bad({ per_frame: {} })).toThrow('per_frame.items: expected a positive number');
    expect(bad({ check: { ...embedder().check, metric: 'l2' } })).toThrow('check.metric: expected "maxabs" or "cosine", got "l2"');
    expect(bad({ check: { ...embedder().check, expected: {} } })).toThrow('check.expected');
    expect(bad({ check: { ...embedder().check, expected: { other: 'x.bin' } } })).toThrow('check.expected: no file for output "embedding"');
    expect(bad({ check: { ...embedder().check, tolerance: { fp32: 'tight' } } })).toThrow('check.tolerance.fp32: expected a number, or an object of numbers');
    expect(bad({ check: { ...embedder().check, batch: 0 } })).toThrow('check.batch: expected a positive integer');
  });
});

describe('models/index.json', () => {
  it('is a list of manifest file names', () => {
    expect(parseIndex(['detector-v0.bench.json', 'embedder-v1.bench.json'])).toEqual(['detector-v0.bench.json', 'embedder-v1.bench.json']);
    expect(parseIndex([])).toEqual([]);
  });

  it('is refused when it is anything else', () => {
    expect(() => parseIndex({})).toThrow('expected a list');
    expect(() => parseIndex(['detector-v0.json'])).toThrow('ending in .bench.json');
    expect(() => parseIndex(['../x.bench.json'])).toThrow('plain file name');
    expect(() => parseIndex([3])).toThrow('non-empty string');
  });
});

describe('shapes and batches', () => {
  it('sets the batch axis', () => {
    expect(resolveShape(['batch', 3, 224, 224], 8)).toEqual([8, 3, 224, 224]);
    expect(resolveShape([1, 3, 640, 640], 8)).toEqual([1, 3, 640, 640]);
    expect(elementCount([8, 256])).toBe(2048);
    expect(itemSize(['batch', 3, 224, 224])).toBe(3 * 224 * 224);
    expect(batchAxis(['batch', 3])).toBe(0);
    expect(batchAxis([3, 'batch'])).toBe(1);
  });

  it('counts the values of an output, exactly or (with an open axis) as a multiple', () => {
    expect(resolveOutShape(['batch', 100, 4], 2)).toEqual([2, 100, 4]);
    expect(resolveOutShape(['batch', 'n', 8], 2)).toEqual([2, 'n', 8]);
    expect(outputCount([2, 100, 4])).toEqual({ count: 800, open: false });
    expect(outputCount([2, 'n', 8])).toEqual({ count: 16, open: true });
  });

  it('times a model with a fixed batch at that batch only', () => {
    expect(fixedBatch(['batch', 3, 4])).toBeNull();
    expect(fixedBatch([2, 3, 4])).toBe(2);
    expect(batchesOf({ input: { shape: ['batch', 3] }, batches: [1, 8] })).toEqual([1, 8]);
    expect(batchesOf({ input: { shape: [4, 3] }, batches: [1, 8] })).toEqual([4]);
    expect(itemSize([4, 3, 5])).toBe(15);
  });
});
