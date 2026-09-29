// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The correctness check: what the browser computed against what PyTorch computed (the manifest's check files).
// "maxabs" is the largest |browser - expected| per output; "cosine" is the smallest cosine between a browser row
// and its expected row. Pure functions, no DOM.

export type Metric = 'maxabs' | 'cosine';
/** One tolerance for every output, or one per output name. */
export type Tolerance = number | Record<string, number>;

export interface OutputCheck {
  name: string;
  value: number;
  /** null when the manifest gives no tolerance for this output. */
  tolerance: number | null;
  pass: boolean;
  /** How many values are past the tolerance (maxabs), or rows under it (cosine); null without a tolerance. */
  over: number | null;
  /** Of how many values (maxabs) or rows (cosine). */
  total: number;
  /** The median |difference| (maxabs), the typical drift; null for cosine. A few large differences over a tiny median is a swapped query, not a broken kernel. */
  median: number | null;
}

export interface CheckResult {
  metric: Metric;
  /** The worst output: the largest maxabs, the smallest cosine (NaN if any is NaN). */
  value: number;
  pass: boolean;
  outputs: OutputCheck[];
  /** Why there is no verdict, when there is not one (no tolerance for this precision). */
  note: string;
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** The largest |got - want|; NaN as soon as one difference is NaN. Equal infinities count as equal. */
export function maxAbs(got: ArrayLike<number>, want: ArrayLike<number>): number {
  if (got.length !== want.length) throw new Error(`length ${got.length} differs from expected ${want.length}`);
  let worst = 0;
  for (let i = 0; i < got.length; i++) {
    const a = got[i]!;
    const b = want[i]!;
    if (a === b) continue;
    const d = Math.abs(a - b);
    if (Number.isNaN(d)) return NaN;
    if (d > worst) worst = d;
  }
  return worst;
}

/** The cosine of each row of `got` with the same row of `want`, rows of `rowLength` values (NaN for a row with no length). */
export function rowCosines(got: ArrayLike<number>, want: ArrayLike<number>, rowLength: number): number[] {
  if (got.length !== want.length) throw new Error(`length ${got.length} differs from expected ${want.length}`);
  if (!Number.isInteger(rowLength) || rowLength <= 0 || got.length % rowLength !== 0) {
    throw new Error(`${got.length} values do not split into rows of ${rowLength}`);
  }
  const out: number[] = [];
  for (let start = 0; start < got.length; start += rowLength) {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = start; i < start + rowLength; i++) {
      const a = got[i]!;
      const b = want[i]!;
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    out.push(dot / (Math.sqrt(na) * Math.sqrt(nb)));
  }
  return out;
}

/** The smallest cosine between a row of `got` and the same row of `want`; NaN if one is. */
export function cosineRows(got: ArrayLike<number>, want: ArrayLike<number>, rowLength: number): number {
  let worst = Infinity;
  for (const c of rowCosines(got, want, rowLength)) {
    if (Number.isNaN(c)) return NaN;
    if (c < worst) worst = c;
  }
  return worst;
}

/** Of the differences |got - want|: how many are past `tolerance` (NaN counts), and their median (NaN ones left out). */
export function driftOf(got: ArrayLike<number>, want: ArrayLike<number>, tolerance: number | null): { over: number | null; median: number | null } {
  if (got.length !== want.length) throw new Error(`length ${got.length} differs from expected ${want.length}`);
  const diffs: number[] = [];
  let over = 0;
  for (let i = 0; i < got.length; i++) {
    const a = got[i]!;
    const b = want[i]!;
    const d = a === b ? 0 : Math.abs(a - b);
    if (Number.isNaN(d)) {
      over++;
    } else {
      diffs.push(d);
      if (tolerance !== null && d > tolerance) over++;
    }
  }
  diffs.sort((x, y) => x - y);
  const mid = diffs.length >> 1;
  const median = diffs.length === 0 ? null : diffs.length % 2 ? diffs[mid]! : (diffs[mid - 1]! + diffs[mid]!) / 2;
  return { over: tolerance === null ? null : over, median };
}

/** The tolerance of one output: the number, or the one named for it; undefined when there is none. */
export function toleranceOf(tolerance: Tolerance | undefined, output: string): number | undefined {
  if (tolerance === undefined) return undefined;
  if (typeof tolerance === 'number') return tolerance;
  const t = tolerance[output];
  return typeof t === 'number' ? t : undefined;
}

export interface OutputPair {
  name: string;
  got: ArrayLike<number>;
  want: ArrayLike<number>;
  /** The length of one row: the last dimension of the output's shape (cosine only). */
  rowLength: number;
}

/** Scores every output by the metric and judges it against the tolerance: maxabs must be <=, cosine must be >=. */
export function evaluate(metric: Metric, pairs: readonly OutputPair[], tolerance: Tolerance | undefined): CheckResult {
  const outputs: OutputCheck[] = pairs.map((p) => {
    const tol = toleranceOf(tolerance, p.name) ?? null;
    if (metric === 'maxabs') {
      const value = maxAbs(p.got, p.want);
      const drift = driftOf(p.got, p.want, tol);
      return { name: p.name, value, tolerance: tol, pass: tol !== null && value <= tol, over: drift.over, total: p.got.length, median: drift.median };
    }
    const cosines = rowCosines(p.got, p.want, p.rowLength);
    const value = cosines.some(Number.isNaN) ? NaN : Math.min(...cosines);
    const under = tol === null ? null : cosines.filter((c) => !(c >= tol)).length;
    return { name: p.name, value, tolerance: tol, pass: tol !== null && value >= tol, over: under, total: cosines.length, median: null };
  });
  const values = outputs.map((o) => o.value);
  const value = values.some(Number.isNaN) ? NaN : metric === 'maxabs' ? Math.max(...values) : Math.min(...values);
  const missing = outputs.filter((o) => o.tolerance === null).map((o) => o.name);
  return {
    metric,
    value,
    pass: outputs.length > 0 && outputs.every((o) => o.pass),
    outputs,
    note: missing.length ? `no tolerance for ${missing.join(', ')}` : '',
  };
}

/** A check file's bytes as numbers: raw little-endian, C order, no header. */
export function decodeBin(bytes: Uint8Array, dtype: 'float32' | 'uint8'): Float32Array | Uint8Array {
  if (dtype === 'uint8') return bytes;
  if (bytes.byteLength % 4 !== 0) throw new Error(`${bytes.byteLength} bytes are not a whole number of float32`);
  const n = bytes.byteLength / 4;
  if (LITTLE_ENDIAN && bytes.byteOffset % 4 === 0) return new Float32Array(bytes.buffer, bytes.byteOffset, n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** An IEEE half-precision bit pattern as a number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** What a session output holds, as float32: float32 as it is, half floats decoded, the rest converted. */
export function toFloat32(data: unknown, type: string): Float32Array {
  if (data instanceof Float32Array) return data;
  if (type === 'float16' && data instanceof Uint16Array) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = halfToFloat(data[i]!);
    return out;
  }
  if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
    return Float32Array.from(data as unknown as ArrayLike<number | bigint>, (v) => Number(v));
  }
  throw new Error(`cannot read an output of type ${type}`);
}
