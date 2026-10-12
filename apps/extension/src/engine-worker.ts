// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The engine worker's loop, for one onnxruntime-web build (engine-webgpu.ts, engine-wasm.ts): the document sends
// `init` (which models, which way, and in the store build the card list's rows), then a frame at a time; the worker
// answers with the board. It runs in a worker of the engine document so that a hung or lost GPU costs only this worker,
// which the document starts afresh.

import { loadGallery, sha256Hex, type Reader, type StandalonePackage } from './assets';
import { cacheKeep, fetchFile, prune } from './downloads';
import { bytesOfBase64, decodeJpeg } from './decode';
import { EngineHost, FPS } from './engine-host';
import type { FromEngine, ToEngine } from './protocol';
import type { Trace } from './parts';
import { loadParts } from './parts-engine';
import { Timer } from './timer';

export type Ort = typeof import('onnxruntime-web');

function describe(e: unknown): string {
  if (e instanceof Error) return e.name && e.name !== 'Error' ? `${e.name}: ${e.message}` : e.message;
  if (typeof e === 'number') return `error code ${e} from the runtime`; // the wasm side throws numbers
  return String(e);
}

/** What the GPU's shader compiler said of the shaders it would not compile, with the line it stopped at: the reason a WebGPU
 * engine fails is otherwise only "ShaderModule with 'Clip' label is invalid" (Firefox's compiler is stricter than Chrome's). */
const shaderErrors: string[] = [];
const SHADER_ERRORS = 3;

interface ShaderMessage {
  type: string;
  message: string;
  lineNum: number;
}
interface ShaderModuleLike {
  getCompilationInfo?: () => Promise<{ messages: readonly ShaderMessage[] }>;
}
type CreateShaderModule = (this: unknown, desc: { code: string; label?: string }) => ShaderModuleLike;

/** Every shader module made in this worker is asked what its compiler said; errors are kept (a few, the first ones). */
function watchShaders(): void {
  const proto = (globalThis as { GPUDevice?: { prototype: { createShaderModule: CreateShaderModule; __wardeye?: boolean } } }).GPUDevice?.prototype;
  if (!proto || proto.__wardeye) return;
  proto.__wardeye = true;
  const create = proto.createShaderModule;
  proto.createShaderModule = function (desc) {
    const module = create.call(this, desc);
    void module.getCompilationInfo?.().then(
      (info) => {
        for (const m of info.messages) {
          if (m.type !== 'error' || shaderErrors.length >= SHADER_ERRORS) continue;
          const line = (desc.code.split('\n')[m.lineNum - 1] ?? '').trim().slice(0, 160);
          shaderErrors.push(`${desc.label ?? 'a shader'}: ${m.message}${line ? ` (line ${m.lineNum}: ${line})` : ''}`);
        }
      },
      () => {},
    );
    return module;
  };
}

/** An error, with what the shader compiler said when it refused a shader. */
function withShaders(e: unknown): string {
  return shaderErrors.length ? `${describe(e)}; the shader compiler said: ${shaderErrors.join('; ')}` : describe(e);
}

/** WASM threads to ask for: up to 4 and never more than the cores, but 1 without cross-origin isolation. */
export function wasmThreads(isolated: boolean, cores: number): number {
  return isolated ? Math.max(1, Math.min(4, cores || 1)) : 1;
}

/** The package's files: from the package itself, or, for a file it names but does not hold, downloaded once and kept in the
 * browser's Cache Storage (downloads.ts). The kept files the package no longer names are deleted first. */
async function packageReader(pkg: StandalonePackage, base: URL, progress: (message: string) => void): Promise<Reader> {
  const fromPackage: Reader = async (path) => {
    const r = await fetch(new URL(path, base));
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const listed = pkg.downloads ?? {};
  if (!Object.keys(listed).length) return fromPackage;
  const keep = cacheKeep(await caches.open('wardeye-models'));
  await prune(keep, Object.values(listed).map((d) => d.sha256));
  // the address is all a download sends: no cookie, no referrer
  const net = { fetch: (url: string) => fetch(url, { credentials: 'omit', referrerPolicy: 'no-referrer' }), sha256: sha256Hex, progress };
  return async (path) => {
    const d = listed[path];
    return d ? fetchFile(path, d, keep, net) : fromPackage(path);
  };
}

export function serve(ort: Ort): void {
  const post = (m: FromEngine): void => self.postMessage(m);
  const base = new URL('./', self.location.href);
  let host: EngineHost | null = null;
  let frameId = 0; // the frame being read: what the trace tells is of it

  async function init(m: Extract<ToEngine, { kind: 'init' }>): Promise<EngineHost> {
    if (m.attempt.runtime === 'webgpu') watchShaders();
    ort.env.wasm.wasmPaths = new URL('ort/', base).href;
    ort.env.wasm.numThreads = m.pkg.threads ?? wasmThreads(self.crossOriginIsolated, navigator.hardwareConcurrency);
    ort.env.logLevel = 'warning';
    const progress = (message: string): void => post({ kind: 'progress', message });
    const read = await packageReader(m.pkg, base, progress);
    progress('reading the gallery');
    const gallery = await loadGallery(read, m.pkg.data, m.cards);
    if (m.cards) {
      // the store build: how many of the gallery's printings Riot's card list named (the rest are named by their ids)
      const listed = new Set(m.cards.map((c) => c.printing_id));
      console.info(`Wardeye: the card list names ${gallery.index.rows.filter((id) => listed.has(id)).length} of the gallery's ${gallery.index.rows.length} printings`);
    }
    const timer = new Timer(() => performance.now());
    const fps = m.pkg.fps ?? FPS;
    const trace: Trace | undefined = m.pkg.trace
      ? {
          boxes: (t, boxes) => post({ kind: 'trace', id: frameId, what: 'boxes', t, boxes: [...boxes] }),
          embed: (count, rows) => post({ kind: 'trace', id: frameId, what: 'embed', count, rows: rows.slice() }),
        }
      : undefined;
    const parts = await loadParts({ ort, pkg: m.pkg, attempt: m.attempt, read, gallery, timer, fps, progress, decode: decodeJpeg, ...(trace ? { trace } : {}) });
    const layout = m.pkg.layout ? (parts.presets().find((l) => l.name === m.pkg.layout) ?? null) : null;
    if (m.pkg.layout && !layout) throw new Error(`standalone.json: no layout named ${m.pkg.layout}`);
    return new EngineHost({ parts, timer, attempt: m.attempt, layout, fps });
  }

  self.onmessage = async (e: MessageEvent<ToEngine>) => {
    const m = e.data;
    if (m.kind === 'init') {
      try {
        host = await init(m);
        post({ kind: 'ready' });
      } catch (err) {
        post({ kind: 'failed', error: withShaders(err) });
      }
    } else if (m.kind === 'frame') {
      try {
        if (!host) throw new Error('the engine is not loaded');
        frameId = m.id;
        const out = await host.frame({ tab: m.tab, t: m.t, video: m.video, jpeg: bytesOfBase64(m.jpeg), ...(m.lists ? { lists: m.lists } : {}) });
        post({ kind: 'state', id: m.id, state: out.state, events: out.events });
      } catch (err) {
        post({ kind: 'error', id: m.id, error: withShaders(err) });
      }
    } else if (m.kind === 'forget') {
      host?.forget(m.tab);
    }
  };
}
