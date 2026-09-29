// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The page of the browser replay (replay.spec.ts): it plays the engine document's part. It starts the engine worker
// as the extension does, sends it the frames of replay.json one after another as base64 JPEGs, and keeps what
// comes back in window.__replay for the test to read. It also decodes each frame the parity way and checks the
// RGB against the SHA-256 Pillow's decode of it has.

import { parsePackage, sha256Hex } from '../src/assets';
import { b64Of } from '../src/companion';
import { decodeJpeg } from '../src/decode';
import { WorkerEngine } from '../src/engine-client';
import type { Attempt } from '../src/mode';
import type { CardBox } from '@rifteye/engine';
import { compareBoxes, compareRows, type Box, type GotStep, type StepDiag } from './replay-compare';

interface Info {
  frames: { file: string; t: number; rgb_sha256: string }[];
  attempt: Attempt;
  worker: string;
  /** The reference's steps.jsonl, embeds.json and embeds.bin are served under reference/. */
  reference: boolean;
}

export interface Replay {
  total: number;
  progress: number;
  log: string[];
  parity: { same: number; different: number; first: string | null };
  results: (GotStep & { timing: unknown; ms: number })[];
  /** How near the finder's boxes and the embedder's rows are to Python's, for each step (when the reference is served). */
  diag: StepDiag[];
  done: boolean;
  error: string | null;
  isolated: boolean;
}

declare global {
  interface Window {
    __replay: Replay;
  }
}

const state: Replay = { total: 0, progress: 0, log: [], parity: { same: 0, different: 0, first: null }, results: [], diag: [], done: false, error: null, isolated: self.crossOriginIsolated };
window.__replay = state;
const say = (m: string): void => void state.log.push(`${(performance.now() / 1000).toFixed(1)} s: ${m}`);

interface Reference {
  steps: { boxes: Box[] | null; embeds: number[] }[];
  calls: { count: number; offset: number }[];
  rows: Float32Array;
  dim: number;
}

async function loadReference(): Promise<Reference> {
  const steps = (await (await fetch('reference/steps.jsonl')).text()).split('\n').filter((l) => l.trim() !== '').map((l) => JSON.parse(l) as { boxes: Box[] | null; embeds: number[] });
  const embeds = (await (await fetch('reference/embeds.json')).json()) as { dim: number; calls: { count: number; offset: number }[] };
  return { steps, calls: embeds.calls, dim: embeds.dim, rows: new Float32Array(await (await fetch('reference/embeds.bin')).arrayBuffer()) };
}

/** The step's diagnostics: the finder's boxes and the embed calls this engine made, against Python's. */
function diagnose(i: number, traces: { what: string; boxes?: CardBox[]; count?: number; rows?: Float32Array }[], ref: Reference): StepDiag {
  const want = ref.steps[i]!;
  const boxes = traces.find((t) => t.what === 'boxes')?.boxes ?? [];
  const b = want.boxes ? compareBoxes(boxes as Box[], want.boxes) : { same: boxes.length === 0, maxDelta: 0, back: true, inOrder: true };
  const calls = traces.filter((t) => t.what === 'embed');
  const wantCalls = want.embeds.map((k) => ref.calls[k]!);
  let rows = 0;
  let minCos = 1;
  let maxAbs = 0;
  calls.forEach((c, k) => {
    const w = wantCalls[k];
    if (!w || w.count !== c.count) return;
    const r = compareRows(c.rows!, ref.rows.subarray(w.offset * ref.dim, (w.offset + w.count) * ref.dim), ref.dim);
    rows += w.count;
    minCos = Math.min(minCos, r.minCos);
    maxAbs = Math.max(maxAbs, r.maxAbs);
  });
  return {
    i,
    boxes: { got: boxes.length, want: want.boxes?.length ?? 0, maxDelta: b.maxDelta, back: b.back, inOrder: b.inOrder },
    embeds: { got: calls.map((c) => c.count!), want: wantCalls.map((c) => c.count), rows, minCos, maxAbs },
  };
}

async function main(): Promise<void> {
  const info = (await (await fetch('replay.json')).json()) as Info;
  const pkg = parsePackage(await (await fetch('standalone.json')).json());
  state.total = info.frames.length;
  const reference = info.reference ? await loadReference() : null;
  const traces = new Map<number, { what: string; boxes?: CardBox[]; count?: number; rows?: Float32Array }[]>();
  const engine = new WorkerEngine(info.worker, 10 * 60_000, 10 * 60_000); // WASM reads slowly; the test has its own limit
  engine.onTrace = (m) => {
    const list = traces.get(m.id) ?? [];
    traces.set(m.id, list);
    list.push(m.what === 'boxes' ? { what: 'boxes', boxes: m.boxes } : { what: 'embed', count: m.count, rows: m.rows });
  };
  say(`starting ${info.attempt.runtime}: detector ${info.attempt.detector}, embedder ${info.attempt.embedder}`);
  await engine.init(pkg, info.attempt, say);
  say('engine ready');
  for (const [i, f] of info.frames.entries()) {
    const bytes = new Uint8Array(await (await fetch(`frames/${f.file}`)).arrayBuffer());
    const rgb = await decodeJpeg(bytes);
    if ((await sha256Hex(rgb.data)) === f.rgb_sha256) {
      state.parity.same++;
    } else {
      state.parity.different++;
      state.parity.first ??= `${f.file}: the RGB is not Pillow's`;
    }
    const tic = performance.now();
    const out = await engine.frame({ tab: 1, t: f.t, video: 'la-final', jpeg: b64Of(bytes) });
    const ms = performance.now() - tic;
    state.results.push({ i, file: f.file, t: f.t, state: out.state as unknown as Record<string, unknown>, events: out.events, timing: (out.state as { engine?: { timing?: unknown } }).engine?.timing, ms });
    if (reference) state.diag.push(diagnose(i, traces.get(i + 1) ?? [], reference));
    traces.delete(i + 1);
    state.progress = i + 1;
    if (i < 5 || (i + 1) % 20 === 0) say(`frame ${i} (t=${f.t}): ${ms.toFixed(0)} ms, ${(out.state as { tracks?: unknown[] }).tracks?.length ?? 0} tracks, ${(out.state as { status?: string }).status}`);
  }
  engine.terminate();
  say('done');
  state.done = true;
}

main().catch((e) => {
  state.error = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
  state.done = true;
});
