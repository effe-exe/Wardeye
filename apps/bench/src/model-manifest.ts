// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The model manifests the bench reads (<id>.bench.json, listed by models/index.json), written by whoever exports
// the ONNX files. This is not the extension's manifest.json. Parsing is strict: a manifest that does not say
// what the bench needs fails with a sentence that names the field, and that model's rows say so.

import type { Metric, Tolerance } from './compare';
import type { DetectionTolerance } from './decoded';

/** An input shape entry: a size, or the batch axis. */
export type Dim = number | 'batch';
/** An output shape entry: a size, the batch axis, or the name of an axis whose size the manifest leaves open. */
export type OutDim = number | string;
export type Dtype = 'float32' | 'uint8';

export interface Variant {
  precision: string;
  file: string;
}

export interface TensorSpec {
  name: string;
  dtype: Dtype;
  shape: Dim[];
}

export interface CheckSpec {
  batch: number;
  /** Check file names, next to the manifest. */
  input: string;
  expected: Record<string, string>;
  metric: Metric;
  /** By precision: one number, or one per output. */
  tolerance: Record<string, Tolerance>;
  /** The decoded detector check (decoded.ts), when the manifest asks for it. */
  detections?: DetectionsCheck;
}

/** check.detections: the cards expected on the check tile, and how closely they must be found. */
export interface DetectionsCheck {
  /** The expected detections file, next to the manifest. */
  file: string;
  /** The score cards are judged at (the live runner's --det-score). */
  threshold: number;
  /** By precision: how far a found card's corners (px) and score may be from the expected card's. */
  tolerance: Record<string, DetectionTolerance>;
}

export interface BenchManifest {
  id: string;
  title: string;
  variants: Variant[];
  input: TensorSpec;
  outputs: { name: string; shape: OutDim[] }[];
  batches: number[];
  /** Items one processed frame costs; null when the manifest does not say. */
  perFrame: number | null;
  /** null when the manifest has no check. */
  check: CheckSpec | null;
}

type Json = Record<string, unknown>;

const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const isPositiveInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v > 0;
// File names in a manifest sit next to it: no folders, so a manifest cannot point outside its own folder.
const PLAIN_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function fail(where: string, what: string): never {
  throw new Error(`${where}: ${what}`);
}

function text(v: unknown, where: string): string {
  if (typeof v !== 'string' || v === '') fail(where, 'expected a non-empty string');
  return v;
}

function fileName(v: unknown, where: string): string {
  const s = text(v, where);
  if (!PLAIN_NAME.test(s)) fail(where, `expected a plain file name, got "${s}"`);
  return s;
}

function shapeOf(v: unknown, where: string): Dim[];
function shapeOf(v: unknown, where: string, open: true): OutDim[];
function shapeOf(v: unknown, where: string, open = false): OutDim[] {
  if (!Array.isArray(v) || v.length === 0) fail(where, 'expected a non-empty list of sizes');
  return v.map((d, i) => {
    if (isPositiveInt(d) || d === 'batch' || (open && typeof d === 'string' && d !== '')) return d as OutDim;
    return fail(`${where}[${i}]`, `expected a positive integer or "batch"${open ? ' or an axis name' : ''}, got ${JSON.stringify(d)}`);
  });
}

function tolerances(v: unknown, where: string): Record<string, Tolerance> {
  if (!isObject(v)) fail(where, 'expected an object keyed by precision');
  const out: Record<string, Tolerance> = {};
  for (const [precision, t] of Object.entries(v)) {
    if (typeof t === 'number' && Number.isFinite(t) && t >= 0) {
      out[precision] = t;
    } else if (isObject(t) && Object.values(t).every((x) => typeof x === 'number' && Number.isFinite(x) && x >= 0)) {
      out[precision] = t as Record<string, number>;
    } else {
      fail(`${where}.${precision}`, 'expected a number, or an object of numbers keyed by output name');
    }
  }
  return out;
}

const nonNegative = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0;

function detectionsOf(v: unknown, where: string): DetectionsCheck {
  if (!isObject(v)) fail(where, 'expected an object');
  if (!nonNegative(v.threshold) || v.threshold > 1) fail(`${where}.threshold`, 'expected a number from 0 to 1');
  if (!isObject(v.tolerance)) fail(`${where}.tolerance`, 'expected an object keyed by precision');
  const tolerance: Record<string, DetectionTolerance> = {};
  for (const [precision, t] of Object.entries(v.tolerance)) {
    if (!isObject(t) || !nonNegative(t.corner_px) || !nonNegative(t.score)) {
      fail(`${where}.tolerance.${precision}`, 'expected { "corner_px": px, "score": difference }, numbers of 0 or more');
    }
    tolerance[precision] = { corner_px: t.corner_px, score: t.score };
  }
  return { file: fileName(v.file, `${where}.file`), threshold: v.threshold, tolerance };
}

function checkOf(v: unknown, where: string): CheckSpec {
  if (!isObject(v)) fail(where, 'expected an object');
  if (!isPositiveInt(v.batch)) fail(`${where}.batch`, 'expected a positive integer');
  if (v.metric !== 'maxabs' && v.metric !== 'cosine') fail(`${where}.metric`, `expected "maxabs" or "cosine", got ${JSON.stringify(v.metric)}`);
  if (!isObject(v.expected) || Object.keys(v.expected).length === 0) fail(`${where}.expected`, 'expected an object of output name to file');
  const expected: Record<string, string> = {};
  for (const [name, file] of Object.entries(v.expected)) expected[name] = fileName(file, `${where}.expected.${name}`);
  return {
    batch: v.batch,
    input: fileName(v.input, `${where}.input`),
    expected,
    metric: v.metric,
    tolerance: tolerances(v.tolerance, `${where}.tolerance`),
    ...(v.detections !== undefined ? { detections: detectionsOf(v.detections, `${where}.detections`) } : {}),
  };
}

/** Parses and checks one <id>.bench.json; throws an Error whose message names the field at fault. */
export function parseManifest(json: unknown): BenchManifest {
  if (!isObject(json)) fail('manifest', 'expected an object');
  const id = text(json.id, 'id');
  const at = (field: string) => `${id}: ${field}`;

  if (!Array.isArray(json.variants) || json.variants.length === 0) fail(at('variants'), 'expected a non-empty list');
  const variants = json.variants.map((raw, i): Variant => {
    if (!isObject(raw)) return fail(at(`variants[${i}]`), 'expected an object');
    return { precision: text(raw.precision, at(`variants[${i}].precision`)), file: fileName(raw.file, at(`variants[${i}].file`)) };
  });

  if (!isObject(json.input)) fail(at('input'), 'expected an object');
  const dtype = json.input.dtype ?? 'float32';
  if (dtype !== 'float32' && dtype !== 'uint8') fail(at('input.dtype'), `expected "float32" or "uint8", got ${JSON.stringify(dtype)}`);
  const input: TensorSpec = { name: text(json.input.name, at('input.name')), dtype, shape: shapeOf(json.input.shape, at('input.shape')) };

  if (!Array.isArray(json.outputs) || json.outputs.length === 0) fail(at('outputs'), 'expected a non-empty list');
  const outputs = json.outputs.map((raw, i) => {
    if (!isObject(raw)) return fail(at(`outputs[${i}]`), 'expected an object');
    return { name: text(raw.name, at(`outputs[${i}].name`)), shape: shapeOf(raw.shape, at(`outputs[${i}].shape`), true) };
  });

  if (!Array.isArray(json.batches) || json.batches.length === 0 || !json.batches.every(isPositiveInt)) {
    fail(at('batches'), 'expected a non-empty list of positive integers');
  }

  // items a frame costs: usually a whole number, but a median over frames may be fractional
  let perFrame: number | null = null;
  if (json.per_frame !== undefined) {
    const items = isObject(json.per_frame) ? json.per_frame.items : undefined;
    if (typeof items !== 'number' || !Number.isFinite(items) || items <= 0) fail(at('per_frame.items'), 'expected a positive number');
    perFrame = items;
  }

  const check = json.check === undefined ? null : checkOf(json.check, at('check'));
  if (check) {
    for (const o of outputs) if (!(o.name in check.expected)) fail(at('check.expected'), `no file for output "${o.name}"`);
  }

  return {
    id,
    title: typeof json.title === 'string' && json.title ? json.title : id,
    variants,
    input,
    outputs,
    batches: json.batches as number[],
    perFrame,
    check,
  };
}

/** The names in models/index.json: a list of "<id>.bench.json". */
export function parseIndex(json: unknown): string[] {
  if (!Array.isArray(json)) fail('models/index.json', 'expected a list of "<id>.bench.json" names');
  return json.map((n, i) => {
    const name = fileName(n, `models/index.json[${i}]`);
    if (!name.endsWith('.bench.json')) fail(`models/index.json[${i}]`, `expected a name ending in .bench.json, got "${name}"`);
    return name;
  });
}

/** The batch a model was exported with when it has no batch axis; null when the batch is free. */
export function fixedBatch(shape: readonly Dim[]): number | null {
  if (shape.includes('batch')) return null;
  return typeof shape[0] === 'number' ? shape[0] : null;
}

/** The batches to time: those listed, and for a fixed-batch model only its own. */
export function batchesOf(m: { input: { shape: readonly Dim[] }; batches: readonly number[] }): number[] {
  const fixed = fixedBatch(m.input.shape);
  if (fixed === null) return [...m.batches];
  return [fixed];
}

/** Which axis is the batch: the one named "batch", or the first when the batch is fixed. */
export function batchAxis(shape: readonly Dim[]): number {
  const i = shape.indexOf('batch');
  return i < 0 ? 0 : i;
}

/** How many values one item of the batch takes: all the dimensions but the batch axis. */
export function itemSize(shape: readonly Dim[]): number {
  const axis = batchAxis(shape);
  return shape.reduce<number>((n, d, i) => (i === axis ? n : n * (d as number)), 1);
}

/** The input shape with the batch axis set to `batch`. */
export function resolveShape(shape: readonly Dim[], batch: number): number[] {
  return shape.map((d) => (d === 'batch' ? batch : d));
}

/** An output shape with the batch axis set to `batch`; axes the manifest leaves open stay names. */
export function resolveOutShape(shape: readonly OutDim[], batch: number): OutDim[] {
  return shape.map((d) => (d === 'batch' ? batch : d));
}

export function elementCount(shape: readonly number[]): number {
  return shape.reduce((n, d) => n * d, 1);
}

/** The values an output of this shape holds: exact, or (with open axes) a multiple of `count`. */
export function outputCount(shape: readonly OutDim[]): { count: number; open: boolean } {
  let count = 1;
  let open = false;
  for (const d of shape) {
    if (typeof d === 'number') count *= d;
    else open = true;
  }
  return { count, open };
}
