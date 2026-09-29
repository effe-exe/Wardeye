// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The per-frame estimate: what one processed frame costs when every model runs on the items a frame gives it
// (the detector's tiles, the embedder's crops) at the batch size that suits it best, and how many reads a second
// that allows. Only rows whose check passed count.

import type { RuntimeId } from './runtimes';
import type { RowResult } from './types';

export interface TimedRow {
  model: string;
  precision: string;
  runtime: RuntimeId;
  passed: boolean;
  medians: { batch: number; medianMs: number }[];
}

/** The rows that were timed on a runtime, as the estimate reads them; a row counts as passed when its check did. */
export function timedRows(rows: readonly RowResult[]): TimedRow[] {
  const out: TimedRow[] = [];
  for (const r of rows) {
    if (r.runtime === 'all' || r.status === 'skipped') continue;
    const medians = r.batches.flatMap((b) => (b.medianMs === null ? [] : [{ batch: b.batch, medianMs: b.medianMs }]));
    out.push({ model: r.model, precision: r.precision, runtime: r.runtime, passed: r.check?.pass === true, medians });
  }
  return out;
}

export interface FrameItems {
  model: string;
  items: number;
}

export interface BestBatch {
  batch: number;
  calls: number;
  callMs: number;
  ms: number;
}

/** The cheapest batch size for `items` items: the b that gives the least ceil(items / b) x median(b). */
export function bestBatch(items: number, medians: readonly { batch: number; medianMs: number }[]): BestBatch | null {
  let best: BestBatch | null = null;
  for (const { batch, medianMs } of medians) {
    if (!(batch > 0) || !Number.isFinite(medianMs)) continue;
    const calls = Math.ceil(items / batch);
    const ms = calls * medianMs;
    // on a tie, the bigger batch: fewer calls
    if (!best || ms < best.ms || (ms === best.ms && batch > best.batch)) best = { batch, calls, callMs: medianMs, ms };
  }
  return best;
}

export interface Part extends BestBatch {
  model: string;
  precision: string;
  items: number;
}

export interface FrameEstimate {
  runtime: RuntimeId;
  parts: Part[];
  frameMs: number;
  readsPerSec: number;
}

export type EstimateResult = { ok: true; estimate: FrameEstimate } | { ok: false; reason: string };

/**
 * One runtime's estimate: for each model, its best passed precision at its best batch, added up. Models may use
 * different precisions (they are separate sessions). Not available when a model has no passed row on this runtime.
 */
export function estimateFrame(runtime: RuntimeId, rows: readonly TimedRow[], frame: readonly FrameItems[]): EstimateResult {
  if (frame.length === 0) return { ok: false, reason: 'no model states its per-frame items' };
  const parts: Part[] = [];
  for (const { model, items } of frame) {
    let best: Part | null = null;
    for (const r of rows) {
      if (r.model !== model || r.runtime !== runtime || !r.passed) continue;
      const b = bestBatch(items, r.medians);
      if (b && (!best || b.ms < best.ms)) best = { ...b, model, precision: r.precision, items };
    }
    if (!best) return { ok: false, reason: `no row of ${model} on ${runtime} has a passed check and timings` };
    parts.push(best);
  }
  const frameMs = parts.reduce((s, p) => s + p.ms, 0);
  return { ok: true, estimate: { runtime, parts, frameMs, readsPerSec: 1000 / frameMs } };
}

export interface Estimates {
  /** The runtime with the least ms per frame. */
  fastest: EstimateResult;
  /** The WASM runtime alone. */
  wasm: EstimateResult;
  all: EstimateResult[];
}

export function estimates(rows: readonly TimedRow[], frame: readonly FrameItems[], runtimes: readonly RuntimeId[]): Estimates {
  const all = runtimes.map((r) => estimateFrame(r, rows, frame));
  let fastest: EstimateResult | null = null;
  for (const e of all) if (e.ok && (!fastest || !fastest.ok || e.estimate.frameMs < fastest.estimate.frameMs)) fastest = e;
  return {
    fastest: fastest ?? { ok: false, reason: [...new Set(all.map((e) => (e.ok ? '' : e.reason)).filter(Boolean))].join('; ') || 'no runtime was measured' },
    wasm: estimateFrame('wasm', rows, frame),
    all,
  };
}
