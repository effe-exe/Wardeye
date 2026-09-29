import { describe, expect, it } from 'vitest';
import { cosineRows, decodeBin, driftOf, evaluate, halfToFloat, maxAbs, rowCosines, toFloat32, toleranceOf } from '../src/compare';

describe('maxabs', () => {
  it('is the largest difference', () => {
    expect(maxAbs([1, 2, 3], [1, 2.5, 2])).toBe(1);
    expect(maxAbs(new Float32Array([0.5, -0.5]), new Float32Array([0.5, -0.5]))).toBe(0);
  });

  it('is NaN when a value is NaN, and 0 for equal infinities', () => {
    expect(maxAbs([1, NaN], [1, 1])).toBeNaN();
    expect(maxAbs([Infinity, 1], [Infinity, 1])).toBe(0);
    expect(maxAbs([Infinity], [-Infinity])).toBe(Infinity);
  });

  it('refuses outputs of different lengths', () => {
    expect(() => maxAbs([1, 2], [1])).toThrow('length 2 differs from expected 1');
  });
});

describe('cosine', () => {
  it('is 1 for the same rows, 0 for orthogonal ones, -1 for opposite ones', () => {
    expect(cosineRows([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 3)).toBeCloseTo(1, 12);
    expect(cosineRows([1, 0], [0, 1], 2)).toBe(0);
    expect(cosineRows([1, 2], [-1, -2], 2)).toBeCloseTo(-1, 12);
    expect(cosineRows([2, 2], [5, 5], 2)).toBeCloseTo(1, 12); // length does not matter
  });

  it('is the smallest over rows', () => {
    // row 0 identical, row 1 is (1, 0) against (0.6, 0.8): 0.6
    expect(cosineRows([1, 0, 1, 0], [1, 0, 0.6, 0.8], 2)).toBeCloseTo(0.6, 12);
  });

  it('is NaN for an empty row or a NaN value', () => {
    expect(cosineRows([0, 0], [1, 1], 2)).toBeNaN();
    expect(cosineRows([NaN, 1], [1, 1], 2)).toBeNaN();
  });

  it('needs whole rows', () => {
    expect(() => cosineRows([1, 2, 3], [1, 2, 3], 2)).toThrow('do not split into rows of 2');
    expect(() => cosineRows([1, 2], [1, 2, 3, 4], 2)).toThrow('length');
    expect(() => cosineRows([1, 2], [1, 2], 0)).toThrow('rows');
  });
});

describe('drift', () => {
  it('counts the values past the tolerance and takes the median difference', () => {
    // three big differences (a swapped query) over tiny ones
    const want = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const got = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 5, 6, 7];
    expect(driftOf(got, want, 0.01)).toEqual({ over: 3, median: 5.5e-5 });
    expect(driftOf(got, want, null)).toEqual({ over: null, median: 5.5e-5 });
    expect(driftOf([], [], 1)).toEqual({ over: 0, median: null });
  });

  it('counts a NaN as over and leaves it out of the median', () => {
    expect(driftOf([NaN, 1, 3], [0, 1, 1], 0.5)).toEqual({ over: 2, median: 1 });
  });

  it('per row, for cosine', () => {
    expect(rowCosines([1, 0, 1, 0], [1, 0, 0, 1], 2)).toEqual([1, 0]);
  });

  it('comes with each output of a check: values over, of how many, and the median', () => {
    const r = evaluate('maxabs', [{ name: 'y', got: [0.1, 0.2, 3], want: [0, 0, 0], rowLength: 3 }], 0.5);
    expect(r.outputs[0]).toMatchObject({ over: 1, total: 3, median: 0.2, pass: false });
    const c = evaluate('cosine', [{ name: 'e', got: [1, 0, 1, 1], want: [1, 0, 0, 1], rowLength: 2 }], 0.9);
    expect(c.outputs[0]).toMatchObject({ over: 1, total: 2, median: null, pass: false });
    expect(evaluate('maxabs', [{ name: 'y', got: [1], want: [0], rowLength: 1 }], undefined).outputs[0]).toMatchObject({ over: null, tolerance: null });
  });
});

describe('tolerance', () => {
  it('is one number for every output, or one per output name', () => {
    expect(toleranceOf(0.01, 'anything')).toBe(0.01);
    expect(toleranceOf({ pred_boxes: 0.5 }, 'pred_boxes')).toBe(0.5);
    expect(toleranceOf({ pred_boxes: 0.5 }, 'pred_logits')).toBeUndefined();
    expect(toleranceOf(undefined, 'x')).toBeUndefined();
  });
});

describe('judging a check', () => {
  const same = { name: 'y', got: [1, 2], want: [1, 2], rowLength: 2 };

  it('passes maxabs when every output is at or under its tolerance', () => {
    const r = evaluate('maxabs', [{ name: 'boxes', got: [1, 1.04], want: [1, 1], rowLength: 2 }, { name: 'logits', got: [0], want: [0.5], rowLength: 1 }], { boxes: 0.05, logits: 0.5 });
    expect(r.pass).toBe(true);
    expect(r.value).toBe(0.5); // the worst output
    expect(r.outputs.map((o) => [o.name, o.tolerance, o.pass])).toEqual([['boxes', 0.05, true], ['logits', 0.5, true]]);
    expect(r.note).toBe('');
  });

  it('fails when one output is over its own tolerance, though it would pass another output\'s', () => {
    const r = evaluate('maxabs', [{ name: 'boxes', got: [1.2], want: [1], rowLength: 1 }, { name: 'logits', got: [0], want: [0.5], rowLength: 1 }], { boxes: 0.05, logits: 0.5 });
    expect(r.pass).toBe(false);
    expect(r.outputs.map((o) => o.pass)).toEqual([false, true]);
  });

  it('passes cosine when the smallest cosine is at or above the tolerance', () => {
    expect(evaluate('cosine', [{ name: 'e', got: [1, 0], want: [1, 0.01], rowLength: 2 }], 0.999).pass).toBe(true);
    const bad = evaluate('cosine', [{ name: 'e', got: [1, 0], want: [1, 1], rowLength: 2 }], 0.999);
    expect(bad.pass).toBe(false);
    expect(bad.value).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it('fails on NaN', () => {
    const r = evaluate('maxabs', [{ name: 'y', got: [NaN], want: [0], rowLength: 1 }], 1);
    expect(r.pass).toBe(false);
    expect(r.value).toBeNaN();
  });

  it('has no verdict without a tolerance, and says which output lacks one', () => {
    const r = evaluate('maxabs', [same, { name: 'z', got: [1], want: [1], rowLength: 1 }], { y: 1 });
    expect(r.pass).toBe(false);
    expect(r.note).toBe('no tolerance for z');
    expect(evaluate('maxabs', [same], undefined).note).toBe('no tolerance for y');
  });
});

describe('check files', () => {
  it('are raw little-endian float32, or uint8', () => {
    const f = new Float32Array([1.5, -2, 0.25]);
    const bytes = new Uint8Array(f.buffer);
    expect(Array.from(decodeBin(bytes, 'float32'))).toEqual([1.5, -2, 0.25]);
    expect(decodeBin(new Uint8Array([1, 2, 255]), 'uint8')).toEqual(new Uint8Array([1, 2, 255]));
  });

  it('are read from wherever they sit in a buffer', () => {
    const f = new Float32Array([1, 2, 3]);
    const padded = new Uint8Array(1 + 12);
    padded.set(new Uint8Array(f.buffer), 1); // starts at an odd offset
    expect(Array.from(decodeBin(padded.subarray(1), 'float32'))).toEqual([1, 2, 3]);
  });

  it('must hold whole float32 values', () => {
    expect(() => decodeBin(new Uint8Array(6), 'float32')).toThrow('not a whole number of float32');
  });
});

describe('what a session returns', () => {
  it('reads half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x7bff)).toBe(65504);
    expect(halfToFloat(0x0001)).toBe(2 ** -24);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(halfToFloat(0xfc00)).toBe(-Infinity);
    expect(halfToFloat(0x7e00)).toBeNaN();
    expect(halfToFloat(0)).toBe(0);
  });

  it('as float32 whatever the type', () => {
    const f = new Float32Array([1, 2]);
    expect(toFloat32(f, 'float32')).toBe(f);
    expect(Array.from(toFloat32(new Uint16Array([0x3c00, 0xc000]), 'float16'))).toEqual([1, -2]);
    expect(Array.from(toFloat32(new Float64Array([0.5]), 'float64'))).toEqual([0.5]);
    expect(Array.from(toFloat32(new BigInt64Array([3n]), 'int64'))).toEqual([3]);
    expect(Array.from(toFloat32(new Uint8Array([7]), 'uint8'))).toEqual([7]);
    expect(() => toFloat32(['a'], 'string')).toThrow('cannot read an output of type string');
  });
});
