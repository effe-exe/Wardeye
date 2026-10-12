// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Model files the package names but does not hold (decision D-038). standalone.json gives each one's address, size and
// sha256. The first time the engine needs a file, it is downloaded from that address, checked against both, and kept in the
// browser's Cache Storage under its hash; from then on it is read from there, with no network. A kept file whose hash the
// package no longer names (an update brought new models) is deleted. Pure: the network and the storage come in from the
// caller, so the tests need neither.

/** A file to download: where from, and what it must be. */
export interface Download {
  url: string;
  sha256: string;
  bytes: number;
}

/** Where downloaded files are kept, by their sha256: Cache Storage in the extension (cacheKeep), a Map in the tests. */
export interface Keep {
  get(sha256: string): Promise<Uint8Array | null>;
  put(sha256: string, bytes: Uint8Array<ArrayBuffer>): Promise<void>;
  keys(): Promise<string[]>;
  delete(sha256: string): Promise<void>;
}

export interface Fetching {
  fetch(url: string): Promise<Response>;
  sha256(bytes: Uint8Array): Promise<string>;
  /** What the badge says while a file downloads ("downloading detector-v0.onnx, the first time only: 42 of 145 MB"). */
  progress(message: string): void;
}

const MB = 1_000_000;
/** A progress message at most every this many bytes, so that the badge is not rewritten for every chunk. */
const PROGRESS_EVERY = 5 * MB;

/** The file at `name` in the package: the kept copy, or else a download, checked, then kept. */
export async function fetchFile(name: string, d: Download, keep: Keep, f: Fetching): Promise<Uint8Array> {
  const kept = await keep.get(d.sha256);
  if (kept && kept.length === d.bytes) return kept; // a kept file was checked when it came, and Cache Storage writes it whole
  const file = name.split('/').pop() ?? name;
  const said = (got: number): void => f.progress(`downloading ${file}, the first time only: ${Math.floor(got / MB)} of ${Math.round(d.bytes / MB)} MB`);
  said(0);
  const r = await f.fetch(d.url);
  if (!r.ok || !r.body) throw new Error(`${file} could not be downloaded (HTTP ${r.status}): the engine starts again later`);
  // read into one buffer of the size it must be, so that a 145 MB model is never in memory twice
  const out = new Uint8Array(d.bytes);
  let got = 0;
  let next = PROGRESS_EVERY;
  const reader = r.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > d.bytes) {
      await reader.cancel();
      throw new Error(`${file} is bigger than this version of Wardeye expects: it was not kept`);
    }
    out.set(value, got);
    got += value.length;
    if (got >= next) {
      said(got);
      next = got + PROGRESS_EVERY;
    }
  }
  if (got !== d.bytes) throw new Error(`${file} came incomplete (${got} of ${d.bytes} bytes): it was not kept`);
  if ((await f.sha256(out)) !== d.sha256) throw new Error(`${file} is not the file this version of Wardeye was made with: it was not kept`);
  await keep.put(d.sha256, out);
  return out;
}

/** Deletes every kept file but the ones the package names now. */
export async function prune(keep: Keep, wanted: readonly string[]): Promise<void> {
  for (const k of await keep.keys()) if (!wanted.includes(k)) await keep.delete(k);
}

/** Cache Storage keys requests by URL: each kept file is under this made-up address, its hash at the end. */
const KEY = 'https://models.wardeye.invalid/';

/** A Keep in a Cache Storage cache (the engine worker opens "wardeye-models"). */
export function cacheKeep(cache: Cache): Keep {
  return {
    async get(sha256) {
      const r = await cache.match(KEY + sha256);
      return r ? new Uint8Array(await r.arrayBuffer()) : null;
    },
    put: (sha256, bytes) => cache.put(KEY + sha256, new Response(bytes)),
    keys: async () => (await cache.keys()).map((r) => r.url).filter((u) => u.startsWith(KEY)).map((u) => u.slice(KEY.length)),
    delete: async (sha256) => void (await cache.delete(KEY + sha256)),
  };
}
