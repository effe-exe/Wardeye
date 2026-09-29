// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The engine worker's loop, for one onnxruntime-web build (engine-webgpu.ts, engine-wasm.ts): the document sends
// `init` (which models, which way, and in the store build the card list's rows), then a frame at a time; the worker
// answers with the board. It runs in a worker of the engine document so that a hung or lost GPU costs only this worker,
// which the document starts afresh.

import { loadGallery, type Reader } from './assets';
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

/** WASM threads to ask for: up to 4 and never more than the cores, but 1 without cross-origin isolation. */
export function wasmThreads(isolated: boolean, cores: number): number {
  return isolated ? Math.max(1, Math.min(4, cores || 1)) : 1;
}

export function serve(ort: Ort): void {
  const post = (m: FromEngine): void => self.postMessage(m);
  const base = new URL('./', self.location.href);
  const read: Reader = async (path) => {
    const r = await fetch(new URL(path, base));
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  let host: EngineHost | null = null;
  let frameId = 0; // the frame being read: what the trace tells is of it

  async function init(m: Extract<ToEngine, { kind: 'init' }>): Promise<EngineHost> {
    ort.env.wasm.wasmPaths = new URL('ort/', base).href;
    ort.env.wasm.numThreads = m.pkg.threads ?? wasmThreads(self.crossOriginIsolated, navigator.hardwareConcurrency);
    ort.env.logLevel = 'warning';
    const progress = (message: string): void => post({ kind: 'progress', message });
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
        post({ kind: 'failed', error: describe(err) });
      }
    } else if (m.kind === 'frame') {
      try {
        if (!host) throw new Error('the engine is not loaded');
        frameId = m.id;
        const out = await host.frame({ tab: m.tab, t: m.t, video: m.video, jpeg: bytesOfBase64(m.jpeg), ...(m.lists ? { lists: m.lists } : {}) });
        post({ kind: 'state', id: m.id, state: out.state, events: out.events });
      } catch (err) {
        post({ kind: 'error', id: m.id, error: describe(err) });
      }
    } else if (m.kind === 'forget') {
      host?.forget(m.tab);
    }
  };
}
