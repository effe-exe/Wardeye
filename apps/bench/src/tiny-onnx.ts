// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A minimal ONNX writer: a ModelProto with one dynamic input, some nodes, constants and outputs, encoded by hand
// (protobuf wire format, no dependency). The bench starts each runtime with a one-node model so that loading the
// wasm module and the GPU is not counted as loading a real model, and the tests build their stand-in models
// with it. Integer attributes only; that is all Cast and Flatten need.

export type ElemType = 'float32' | 'float16' | 'uint8';

export interface TinyValue {
  name: string;
  type: ElemType;
  /** A size, or a name for a dynamic axis. */
  shape: (number | string)[];
}

export interface TinyNode {
  op: string;
  inputs: string[];
  outputs: string[];
  ints?: Record<string, number>;
}

export interface TinyConstant {
  name: string;
  type: 'float32' | 'float16';
  dims: number[];
  values: number[];
}

export interface TinyGraph {
  inputs: TinyValue[];
  outputs: TinyValue[];
  nodes: TinyNode[];
  constants?: TinyConstant[];
}

const ELEM = { float32: 1, uint8: 2, float16: 10 } as const;

class Writer {
  private bytes: number[] = [];

  private varint(v: number): void {
    while (v > 0x7f) {
      this.bytes.push((v % 128) | 0x80);
      v = Math.floor(v / 128);
    }
    this.bytes.push(v);
  }

  int(field: number, v: number): this {
    this.varint(field * 8);
    this.varint(v);
    return this;
  }

  raw(field: number, data: ArrayLike<number>): this {
    this.varint(field * 8 + 2);
    this.varint(data.length);
    for (let i = 0; i < data.length; i++) this.bytes.push(data[i]!);
    return this;
  }

  str(field: number, s: string): this {
    return this.raw(field, new TextEncoder().encode(s));
  }

  msg(field: number, w: Writer): this {
    return this.raw(field, w.done());
  }

  done(): Uint8Array {
    return Uint8Array.from(this.bytes);
  }
}

/** The nearest IEEE half-precision bit pattern of a number (ties to even). */
export function floatToHalf(v: number): number {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0]!;
  const sign = (x >>> 16) & 0x8000;
  const e32 = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (e32 === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const exp = e32 - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >> shift;
    const rest = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rest > mid || (rest === mid && half & 1)) half++;
    return sign | half;
  }
  let half = (exp << 10) | (mant >> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++;
  return sign | half;
}

function valueInfo(v: TinyValue): Writer {
  const shape = new Writer();
  for (const d of v.shape) shape.msg(1, typeof d === 'number' ? new Writer().int(1, d) : new Writer().str(2, d));
  const tensor = new Writer().int(1, ELEM[v.type]).msg(2, shape);
  return new Writer().str(1, v.name).msg(2, new Writer().msg(1, tensor));
}

function constant(c: TinyConstant): Writer {
  const w = new Writer();
  for (const d of c.dims) w.int(1, d);
  w.int(2, ELEM[c.type]).str(8, c.name);
  const raw: number[] = [];
  for (const x of c.values) {
    if (c.type === 'float16') {
      const h = floatToHalf(x);
      raw.push(h & 0xff, h >> 8);
    } else {
      raw.push(...new Uint8Array(new Float32Array([x]).buffer));
    }
  }
  return w.raw(9, raw);
}

function node(n: TinyNode): Writer {
  const w = new Writer();
  for (const i of n.inputs) w.str(1, i);
  for (const o of n.outputs) w.str(2, o);
  w.str(4, n.op);
  for (const [name, value] of Object.entries(n.ints ?? {})) w.msg(5, new Writer().str(1, name).int(3, value).int(20, 2));
  return w;
}

/** The bytes of an ONNX ModelProto (IR version 8, opset 17) for the graph. */
export function encodeModel(g: TinyGraph): Uint8Array {
  const graph = new Writer();
  for (const n of g.nodes) graph.msg(1, node(n));
  graph.str(2, 'bench');
  for (const c of g.constants ?? []) graph.msg(5, constant(c));
  for (const i of g.inputs) graph.msg(11, valueInfo(i));
  for (const o of g.outputs) graph.msg(12, valueInfo(o));
  return new Writer().int(1, 8).str(2, 'rifteye-bench').msg(7, graph).msg(8, new Writer().int(2, 17)).done();
}

/** y = x * 1 on a 1 x 4 input: the model each runtime starts with. */
export function startupModel(): Uint8Array {
  return encodeModel({
    inputs: [{ name: 'x', type: 'float32', shape: [1, 4] }],
    outputs: [{ name: 'y', type: 'float32', shape: [1, 4] }],
    constants: [{ name: 'one', type: 'float32', dims: [], values: [1] }],
    nodes: [{ op: 'Mul', inputs: ['x', 'one'], outputs: ['y'] }],
  });
}
