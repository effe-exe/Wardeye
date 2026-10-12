// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// One rewrite of an ONNX file's bytes, made as the model is opened on WebGPU: a Clip with a lower bound alone (the
// embedder's `F.normalize`, `clamp_min(eps)`) becomes a Max, which gives the same values bit for bit (ml: embed/onnx.py
// clip_to_max does the same at export). Firefox's WebGPU shader compiler (naga) rejects onnxruntime's Clip shader
// ("Invalid access into expression"), and the embedder could not run there; Max compiles everywhere. The file is
// changed in place of the node's operator name alone, and the lengths that hold it: the rest is copied as it is. The
// model's sha256 is taken before, on the file as it was published.

// The protobuf fields walked (onnx.proto): ModelProto.graph = 7, GraphProto.node = 1, NodeProto.input = 1,
// NodeProto.op_type = 4, NodeProto.attribute = 5, NodeProto.domain = 7.
const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const NODE_INPUT = 1;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const NODE_DOMAIN = 7;
const LEN = 2; // wire type of a length-delimited field

interface Field {
  field: number;
  wire: number;
  /** Where the field starts (its tag), where its data starts and ends. */
  start: number;
  data: number;
  end: number;
}

function varint(b: Uint8Array, pos: number): [number, number] {
  let value = 0;
  let scale = 1;
  for (;;) {
    if (pos >= b.length) throw new Error('the ONNX file ends inside a number');
    const byte = b[pos++]!;
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, pos];
    scale *= 128;
  }
}

function encodeVarint(n: number): number[] {
  const out: number[] = [];
  while (n >= 0x80) {
    out.push((n % 128) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}

/** The fields of a message held in b[from, to). */
function fields(b: Uint8Array, from: number, to: number): Field[] {
  const out: Field[] = [];
  let pos = from;
  while (pos < to) {
    const start = pos;
    const [tag, afterTag] = varint(b, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    let data = afterTag;
    let end: number;
    if (wire === 0) end = varint(b, afterTag)[1];
    else if (wire === 1) end = afterTag + 8;
    else if (wire === 5) end = afterTag + 4;
    else if (wire === LEN) {
      const [n, at] = varint(b, afterTag);
      data = at;
      end = at + n;
    } else throw new Error(`the ONNX file has a field of wire type ${wire}, which it should not`);
    if (end > to) throw new Error('the ONNX file is cut short');
    out.push({ field, wire, start, data, end });
    pos = end;
  }
  return out;
}

const text = new TextDecoder();
const str = (b: Uint8Array, f: Field): string => text.decode(b.subarray(f.data, f.end));

/** A length-delimited field `field` holding `body`. */
function lenField(field: number, body: Uint8Array): Uint8Array {
  const head = [...encodeVarint(field * 8 + LEN), ...encodeVarint(body.length)];
  const out = new Uint8Array(head.length + body.length);
  out.set(head);
  out.set(body, head.length);
  return out;
}

function concat(parts: readonly Uint8Array[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A message with some of its length-delimited fields' bodies replaced (by index in `fs`); null when none is. */
function rebuild(b: Uint8Array, from: number, to: number, fs: readonly Field[], replaced: Map<number, Uint8Array>): Uint8Array | null {
  if (replaced.size === 0) return null;
  const parts: Uint8Array[] = [];
  let copied = from;
  fs.forEach((f, i) => {
    const body = replaced.get(i);
    if (!body) return;
    parts.push(b.subarray(copied, f.start), lenField(f.field, body));
    copied = f.end;
  });
  parts.push(b.subarray(copied, to));
  return concat(parts);
}

/** A node as Max when it is a Clip of the default domain with a lower bound alone and no attributes; else null. */
function clipNode(b: Uint8Array, from: number, to: number): Uint8Array | null {
  const fs = fields(b, from, to);
  const inputs = fs.filter((f) => f.field === NODE_INPUT && f.wire === LEN);
  const op = fs.findIndex((f) => f.field === NODE_OP_TYPE && f.wire === LEN);
  const domain = fs.find((f) => f.field === NODE_DOMAIN && f.wire === LEN);
  if (op < 0 || str(b, fs[op]!) !== 'Clip') return null;
  if (inputs.length !== 2 || str(b, inputs[1]!) === '') return null; // an upper bound too, or none at all
  if (fs.some((f) => f.field === NODE_ATTRIBUTE) || (domain && str(b, domain) !== '')) return null;
  return rebuild(b, from, to, fs, new Map([[op, new TextEncoder().encode('Max')]]));
}

/** The model's bytes with each lower-bound-only Clip of its main graph written as Max: the same array when there is none,
 * or when the bytes cannot be read as ONNX (onnxruntime then says what is wrong with them). */
export function clipToMax(model: Uint8Array): Uint8Array {
  try {
    return rewrite(model);
  } catch {
    return model;
  }
}

function rewrite(model: Uint8Array): Uint8Array {
  const top = fields(model, 0, model.length);
  const g = top.findIndex((f) => f.field === MODEL_GRAPH && f.wire === LEN);
  if (g < 0) return model;
  const graph = top[g]!;
  const nodes = fields(model, graph.data, graph.end);
  const changed = new Map<number, Uint8Array>();
  nodes.forEach((f, i) => {
    if (f.field !== GRAPH_NODE || f.wire !== LEN) return;
    const node = clipNode(model, f.data, f.end);
    if (node) changed.set(i, node);
  });
  const body = rebuild(model, graph.data, graph.end, nodes, changed);
  return body ? (rebuild(model, 0, model.length, top, new Map([[g, body]])) ?? model) : model;
}
