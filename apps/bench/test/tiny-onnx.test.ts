import { describe, expect, it } from 'vitest';
import { halfToFloat } from '../src/compare';
import { encodeModel, floatToHalf, startupModel } from '../src/tiny-onnx';

describe('half floats', () => {
  it('are rounded to the nearest, ties to even', () => {
    expect(floatToHalf(0)).toBe(0);
    expect(floatToHalf(1)).toBe(0x3c00);
    expect(floatToHalf(-2)).toBe(0xc000);
    expect(floatToHalf(3)).toBe(0x4200);
    expect(floatToHalf(0.1)).toBe(0x2e66);
    expect(floatToHalf(65504)).toBe(0x7bff);
    expect(floatToHalf(65520)).toBe(0x7c00); // halfway to 65536: rounds to even, which is infinity
    expect(floatToHalf(1e9)).toBe(0x7c00);
    expect(floatToHalf(-Infinity)).toBe(0xfc00);
    expect(floatToHalf(NaN) & 0x7c00).toBe(0x7c00);
    expect(floatToHalf(2 ** -24)).toBe(0x0001); // the smallest subnormal
    expect(floatToHalf(2 ** -25)).toBe(0); // half of it: a tie, to even
    expect(floatToHalf(3 * 2 ** -25)).toBe(0x0002);
    expect(floatToHalf(1e-9)).toBe(0);
    expect(floatToHalf(1 + 2 ** -11)).toBe(0x3c00); // a tie between 1 and 1 + 2^-10: to even
    expect(floatToHalf(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it('come back exactly through halfToFloat, for every pattern that is a number', () => {
    for (let h = 0; h < 0x10000; h++) {
      const f = halfToFloat(h);
      if (Number.isNaN(f)) continue;
      expect(floatToHalf(f)).toBe(h);
    }
  });
});

describe('the model writer', () => {
  it('writes a ModelProto: IR version 8 first, and the node it was given', () => {
    const bytes = startupModel();
    expect(Array.from(bytes.slice(0, 2))).toEqual([0x08, 0x08]); // field 1 (ir_version) = 8
    expect(new TextDecoder().decode(bytes)).toContain('Mul');
  });

  it('carries dynamic axes by name and fixed ones by size', () => {
    const bytes = encodeModel({
      inputs: [{ name: 'x', type: 'float32', shape: ['batch', 3] }],
      outputs: [{ name: 'y', type: 'uint8', shape: ['batch', 3] }],
      nodes: [{ op: 'Cast', inputs: ['x'], outputs: ['y'], ints: { to: 2 } }],
    });
    const text = new TextDecoder('latin1').decode(bytes);
    expect(text).toContain('batch');
    expect(text).toContain('Cast');
    expect(text).toContain('to');
  });
});
