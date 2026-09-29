// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The private build's files, read and checked: standalone.json (what the package holds), the gallery (an index and
// one float16 .bin a level) and the catalogue, all made by ml/rifteye_ml/web_assets.py. The store build has no catalogue
// file: its rows are the index's, named by Riot's card list. Pure: the bytes come in through a reader, so the tests need
// no files and no browser.

import type { CatalogRow } from '@rifteye/engine';

/** How the engine is asked to run: 'auto' chooses from the browser; the others force one way (a package's
 * standalone.json can say, for a machine that needs it). */
export type RuntimeSetting = 'auto' | 'webgpu' | 'wasm' | 'companion';

export interface ModelFiles {
  id: string;
  /** Paths inside the package, of the precisions it holds. */
  fp16?: string;
  fp32?: string;
}

/** standalone.json: what the private build holds. It is not there in the public build. */
export interface StandalonePackage {
  format: 1;
  runtime: RuntimeSetting;
  detector: ModelFiles;
  embedder: ModelFiles;
  /** Where the gallery, the catalogue and the thumbnails are (a folder inside the package). */
  data: string;
  /** The name of a preset layout to use for every video instead of finding one. */
  layout?: string;
  /** The frames a second the boards are built for, when not the default (a replay of frames recorded at another pace). */
  fps?: number;
  /** WASM threads, when not as many as the cores allow (up to 4). */
  threads?: number;
  /** The engine worker says what the finder and the embedder gave, for a check against a reference (the browser replay). */
  trace?: boolean;
}

export interface GalleryIndex {
  format: 1;
  encoder: string;
  model: string;
  sha256: string;
  fp16_sha256: string | null;
  dim: number;
  /** float16 in the package; float32 for a check that must not lose a bit to the rounding. */
  dtype: 'float16' | 'float32';
  levels: number[];
  rows: string[];
}

export type Reader = (path: string) => Promise<Uint8Array>;

const isObject = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);

function field<T>(o: Record<string, unknown>, key: string, ok: (v: unknown) => v is T, what: string, where: string): T {
  const v = o[key];
  if (!ok(v)) throw new Error(`${where}: ${key} must be ${what}`);
  return v;
}

const isString = (v: unknown): v is string => typeof v === 'string' && v.length > 0;
const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);
const isStrings = (v: unknown): v is string[] => Array.isArray(v) && v.every((s) => typeof s === 'string');
const isInts = (v: unknown): v is number[] => Array.isArray(v) && v.length > 0 && v.every(isInt);

function modelFiles(x: unknown, where: string): ModelFiles {
  if (!isObject(x)) throw new Error(`${where} must be an object`);
  const out: ModelFiles = { id: field(x, 'id', isString, 'a name', where) };
  for (const p of ['fp16', 'fp32'] as const) {
    if (x[p] !== undefined) out[p] = field(x, p, isString, 'a path', where);
  }
  if (!out.fp16 && !out.fp32) throw new Error(`${where} names no model file (fp16 or fp32)`);
  return out;
}

/** standalone.json, checked. */
export function parsePackage(x: unknown): StandalonePackage {
  if (!isObject(x)) throw new Error('standalone.json is not an object');
  if (x.format !== 1) throw new Error(`standalone.json: format ${String(x.format)} is not one this extension reads (1)`);
  const runtime = x.runtime ?? 'auto';
  if (runtime !== 'auto' && runtime !== 'webgpu' && runtime !== 'wasm' && runtime !== 'companion') {
    throw new Error(`standalone.json: runtime must be auto, webgpu, wasm or companion, not ${String(runtime)}`);
  }
  const data = x.data === undefined ? 'data/' : field(x, 'data', isString, 'a folder', 'standalone.json');
  const out: StandalonePackage = {
    format: 1,
    runtime,
    detector: modelFiles(x.detector, 'standalone.json: detector'),
    embedder: modelFiles(x.embedder, 'standalone.json: embedder'),
    data: data.endsWith('/') ? data : `${data}/`,
  };
  if (x.layout !== undefined) out.layout = field(x, 'layout', isString, 'the name of a preset layout', 'standalone.json');
  if (x.fps !== undefined) {
    const fps = x.fps;
    if (typeof fps !== 'number' || !(fps > 0 && fps <= 60)) throw new Error('standalone.json: fps must be a number from 0 to 60');
    out.fps = fps;
  }
  if (x.trace !== undefined) {
    if (typeof x.trace !== 'boolean') throw new Error('standalone.json: trace must be true or false');
    out.trace = x.trace;
  }
  if (x.threads !== undefined) {
    const threads = x.threads;
    if (typeof threads !== 'number' || !Number.isInteger(threads) || threads < 1 || threads > 8) throw new Error('standalone.json: threads must be a whole number from 1 to 8');
    out.threads = threads;
  }
  return out;
}

/** gallery/index.json, checked. */
export function parseGalleryIndex(x: unknown): GalleryIndex {
  const where = 'gallery/index.json';
  if (!isObject(x)) throw new Error(`${where} is not an object`);
  if (x.format !== 1) throw new Error(`${where}: format ${String(x.format)} is not one this extension reads (1)`);
  if (x.dtype !== 'float16' && x.dtype !== 'float32') throw new Error(`${where}: dtype must be float16 or float32`);
  const dim = field(x, 'dim', isInt, 'a whole number', where);
  if (dim < 1) throw new Error(`${where}: dim must be positive`);
  const fp16 = x.fp16_sha256;
  return {
    format: 1,
    encoder: field(x, 'encoder', isString, 'a name', where),
    model: field(x, 'model', isString, 'a name', where),
    sha256: field(x, 'sha256', isString, 'a hash', where),
    fp16_sha256: typeof fp16 === 'string' ? fp16 : null,
    dim,
    dtype: x.dtype,
    levels: field(x, 'levels', isInts, 'a list of whole numbers', where),
    rows: field(x, 'rows', isStrings, 'a list of printing ids', where),
  };
}

/** catalog.json, checked: the rows the tracker and the overlay read. */
export function parseCatalog(x: unknown): CatalogRow[] {
  if (!Array.isArray(x) || x.length === 0) throw new Error('catalog.json is not a list of printings');
  return x.map((r, i) => {
    if (!isObject(r)) throw new Error(`catalog.json: row ${i} is not an object`);
    for (const k of ['printing_id', 'card_id', 'name'] as const) {
      if (typeof r[k] !== 'string' || !r[k]) throw new Error(`catalog.json: row ${i} has no ${k}`);
    }
    return { ...r, type: typeof r.type === 'string' ? r.type : '' } as CatalogRow;
  });
}

/** The gallery's rows are the catalogue's rows, in the same order. */
export function checkPairing(index: GalleryIndex, catalog: readonly CatalogRow[]): void {
  if (index.rows.length !== catalog.length) {
    throw new Error(`the gallery has ${index.rows.length} rows and the catalogue ${catalog.length}`);
  }
  const at = index.rows.findIndex((id, i) => catalog[i]!.printing_id !== id);
  if (at >= 0) throw new Error(`the gallery's row ${at} is ${index.rows[at]}, the catalogue's is ${catalog[at]!.printing_id}`);
}

// every float16 bit pattern as the float32 it is
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 0x1f ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

/** Little-endian float16 values as float32. */
export function halfToFloat(bytes: Uint8Array): Float32Array {
  if (bytes.length % 2) throw new Error('a float16 file has an odd number of bytes');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = HALF[view.getUint16(i * 2, true)]!;
  return out;
}

/** Little-endian float32 values (a copy: the bytes may not be aligned). */
export function floatsOf(bytes: Uint8Array): Float32Array {
  if (bytes.length % 4) throw new Error('a float32 file has a length that is not a multiple of 4');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** Every level of the gallery as rows x dim float32 values, from `<folder>gallery/L<px>.bin`. */
export async function loadLevels(read: Reader, folder: string, index: GalleryIndex): Promise<Map<number, Float32Array>> {
  const levels = new Map<number, Float32Array>();
  for (const px of index.levels) {
    const name = `${folder}gallery/L${px}.bin`;
    const bytes = await read(name);
    const size = index.dtype === 'float16' ? 2 : 4;
    const want = index.rows.length * index.dim * size;
    if (bytes.length !== want) throw new Error(`${name} is ${bytes.length} bytes, ${index.rows.length} rows of ${index.dim} ${index.dtype} are ${want}`);
    levels.set(px, index.dtype === 'float16' ? halfToFloat(bytes) : floatsOf(bytes));
  }
  return levels;
}

export interface Gallery {
  index: GalleryIndex;
  rows: CatalogRow[];
  levels: Map<number, Float32Array>;
}

/** The store build's catalogue (it has no catalog.json, and no card name, type or image): one row for each row of the gallery's
 * index, in its order, from Riot's card list (feed.ts). The gallery also holds a few printings seen on stream that the list does
 * not name (ml/rifteye_ml/catalog.py's supplement):
 * - another art of a listed printing ('SFD-195a' of 'SFD-195') is that card, under its own id: it takes the listed printing's
 *   row (name, type, domains), as variant alt_art. It has no picture, since the list gives no address for it.
 * - a token by its code ('SFD-T01') is a unit named 'Token';
 * - any other is named by its own id, with no type: the engine still reads it, as a card of no known kind.
 * Nothing is named but from the list or the printing's own id. */
export function catalogFromFeed(ids: readonly string[], cards: readonly CatalogRow[]): CatalogRow[] {
  const listed = new Map(cards.map((c) => [c.printing_id, c]));
  return ids.map((id) => {
    const card = listed.get(id);
    if (card) return { ...card };
    const base = /^(.+\d)[a-z]$/.exec(id);
    const art = base ? listed.get(base[1]!) : undefined;
    if (art) return { ...art, printing_id: id, variant: 'alt_art' };
    if (/^[A-Z]{2,4}-T\d+$/.test(id)) return { printing_id: id, card_id: id, name: 'Token', type: 'Unit', variant: 'token' };
    return { printing_id: id, card_id: id, name: id, type: '' };
  });
}

/** The catalogue and the gallery, read and checked against each other. The catalogue is catalog.json (the developer build); given
 * `cards` (the store build), it is the index's rows named by them. */
export async function loadGallery(read: Reader, folder: string, cards?: readonly CatalogRow[]): Promise<Gallery> {
  const json = async (path: string): Promise<unknown> => JSON.parse(new TextDecoder().decode(await read(path)));
  const index = parseGalleryIndex(await json(`${folder}gallery/index.json`));
  const rows = cards ? catalogFromFeed(index.rows, cards) : parseCatalog(await json(`${folder}catalog.json`));
  checkPairing(index, rows);
  return { index, rows, levels: await loadLevels(read, folder, index) };
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
