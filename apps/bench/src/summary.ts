// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The plain-text summary the page shows in its text box, for the person running the bench to paste into a chat:
// the environment, one line per row, and the per-frame estimate. Pure text, no DOM.

import type { OutputCheck } from './compare';
import { decodedText } from './decoded';
import type { CheckSummary, RowResult } from './types';
import type { Estimates, EstimateResult, FrameItems } from './estimate';
import { RUNTIMES } from './runtimes';

export interface GpuInfo {
  available: boolean;
  /** Why there is no WebGPU, when there is none. */
  note: string;
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  shaderF16: boolean;
  features: string[];
  maxBufferSize: number | null;
  maxStorageBufferBindingSize: number | null;
}

export interface EnvInfo {
  date: string;
  benchVersion: string;
  ortVersion: string;
  userAgent: string;
  /** From userAgentData (platform, version, architecture); empty when the browser gives none. */
  platform: string;
  cores: number;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  /** WASM threads the bench asks for. */
  threads: number;
  jspi: boolean;
  battery: string;
  gpu: GpuInfo;
}

/** ms with the digits that mean something: 1234, 123, 12.3, 1.23. */
export function fmtMs(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return '-';
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

/** A check value: cosine to 6 decimals (1 - 1e-5 must not read as 1), maxabs to 3 digits. */
export function fmtValue(metric: CheckSummary['metric'], v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (metric === 'cosine') return v.toFixed(6);
  if (v === 0) return '0';
  return v >= 0.001 && v < 1000 ? String(Number(v.toPrecision(3))) : v.toExponential(2);
}

/** "maxabs 0.0123 <= 0.05 PASS", or with several outputs each one's value against its own tolerance. */
export function checkText(c: CheckSummary): string {
  if (c.error) return `not checked (${c.error})`;
  const ok = c.metric === 'maxabs' ? '<=' : '>=';
  const bad = c.metric === 'maxabs' ? '>' : '<';
  // an output that fails says how far it is off: a few values far off over a tiny median (a swapped query), or all of it
  const one = (o: OutputCheck, named: boolean) => {
    const verdict = o.tolerance === null ? ' (no tolerance)' : ` ${o.pass ? ok : bad} ${o.tolerance}`;
    const drift =
      o.pass || o.over === null
        ? ''
        : c.metric === 'maxabs'
          ? ` [${o.over} of ${o.total} values over${o.median === null ? '' : `, median diff ${fmtValue('maxabs', o.median)}`}]`
          : ` [${o.over} of ${o.total} rows under]`;
    return `${named ? `${o.name} ` : ''}${fmtValue(c.metric, o.value)}${verdict}${drift}`;
  };
  const parts = c.outputs.map((o) => one(o, c.outputs.length > 1));
  const decoded = c.decoded ? `; ${decodedText(c.decoded)}` : '';
  return `${c.metric} ${parts.join('; ')} ${c.pass ? 'PASS' : 'FAIL'}${decoded}`;
}

/** The runtime's name as shown: "-" for a variant that is not in the folder. */
export const runtimeName = (r: RowResult): string => (r.runtime === 'all' ? '-' : r.runtime);

/** One row on one line: model | precision | runtime | init, load, first | batches | check. */
export function rowLine(r: RowResult): string {
  const head = `${r.model} | ${r.precision} | ${runtimeName(r)}`;
  if (r.status === 'skipped') return `${head} | ${r.note}`;
  const file = r.modelBytes === null ? '' : ` (${(r.modelBytes / 1e6).toFixed(1)} MB${r.modelSha ? `, sha256 ${r.modelSha}` : ''})`;
  const times = `init ${fmtMs(r.initMs)} | load ${fmtMs(r.loadMs)}${file} | first ${fmtMs(r.firstMs)}`;
  const batches = r.batches
    .map((b) =>
      b.error
        ? `b${b.batch} ERROR`
        : `b${b.batch} run ${fmtMs(b.medianMs)}/${fmtMs(b.p90Ms)} item ${fmtMs(b.perItemMs)}/${fmtMs(b.p90PerItemMs)} first ${fmtMs(b.firstMs)}`,
    )
    .join(' ');
  const check = r.check ? `check ${checkText(r.check)}` : 'no check';
  return [head, times, batches || 'no timings', check].join(' | ');
}

function estimateLines(label: string, result: EstimateResult): string[] {
  if (!result.ok) return [`${label}: not available (${result.reason})`];
  const e = result.estimate;
  const parts = e.parts.map(
    (p) => `${p.model} ${p.precision} ${p.items} item${p.items === 1 ? '' : 's'} = ${p.calls} x b${p.batch} @ ${fmtMs(p.callMs)} = ${fmtMs(p.ms)} ms`,
  );
  return [`${label}: ${e.runtime}: ${parts.join(' + ')} = ${fmtMs(e.frameMs)} ms/frame -> ${e.readsPerSec.toFixed(e.readsPerSec < 1 ? 2 : 1)} reads/s`];
}

export function frameLines(frame: readonly FrameItems[], est: Estimates, leftOut: readonly string[] = []): string[] {
  const items = frame.map((f) => `${f.model} ${f.items} item${f.items === 1 ? '' : 's'}/frame`).join(', ');
  return [
    `per frame (${items || 'no model states its items'}; each model at its best batch; only rows whose check passed):`,
    ...(leftOut.length ? [`  NOT A WHOLE FRAME: ${leftOut.join('; ')}`] : []),
    ...estimateLines('  fastest', est.fastest),
    ...estimateLines('  wasm only', est.wasm),
  ];
}

const PRECISION_ORDER = ['fp32', 'fp16'];

/** Rows by model (in the order the models were listed), then precision (fp32 first), then runtime (GPU first). */
export function sortRows(rows: readonly RowResult[], modelOrder: readonly string[]): RowResult[] {
  const rank = (list: readonly string[], v: string) => {
    const i = list.indexOf(v);
    return i < 0 ? list.length : i;
  };
  const runtimes = RUNTIMES.map((r) => r.id as string);
  return [...rows].sort(
    (a, b) =>
      rank(modelOrder, a.model) - rank(modelOrder, b.model) ||
      rank(PRECISION_ORDER, a.precision) - rank(PRECISION_ORDER, b.precision) ||
      rank(runtimes, a.runtime) - rank(runtimes, b.runtime),
  );
}

const big = (n: number | null) => (n === null ? '?' : String(n));

export function envLines(env: EnvInfo): string[] {
  const g = env.gpu;
  const lines = [
    `browser: ${env.userAgent}`,
    env.platform ? `platform: ${env.platform}` : '',
    `cores ${env.cores} | crossOriginIsolated ${env.crossOriginIsolated ? 'yes' : 'NO'} | SharedArrayBuffer ${env.sharedArrayBuffer ? 'yes' : 'no'} | wasm threads asked ${env.threads} | JSPI ${env.jspi ? 'yes' : 'no'} | ${env.battery}`,
    g.available
      ? `WebGPU: yes | vendor ${g.vendor || '-'} | architecture ${g.architecture || '-'} | device ${g.device || '-'} | description ${g.description || '-'}`
      : `WebGPU: no (${g.note})`,
  ];
  if (g.available) {
    lines.push(
      `WebGPU: shader-f16 ${g.shaderF16 ? 'yes' : 'NO'} | maxBufferSize ${big(g.maxBufferSize)} | maxStorageBufferBindingSize ${big(g.maxStorageBufferBindingSize)}`,
      `WebGPU features: ${g.features.join(', ') || '-'}`,
    );
  }
  return lines.filter(Boolean);
}

/** What the workers saw: whether they were cross-origin isolated and how many wasm threads the runtime took. */
export function workerLine(rows: readonly RowResult[]): string {
  const seen = rows.filter((r) => r.isolated !== null);
  if (seen.length === 0) return 'workers: no row has started a worker yet';
  const isolated = seen.filter((r) => r.isolated).length;
  const threads = [...new Set(seen.map((r) => r.threads))].sort((a, b) => (a ?? 0) - (b ?? 0));
  return `workers: crossOriginIsolated ${isolated === seen.length ? 'yes' : `NO in ${seen.length - isolated} of ${seen.length} rows`} | wasm threads used ${threads.join('/')}`;
}

export interface SummaryInput {
  env: EnvInfo;
  rows: readonly RowResult[];
  modelOrder: readonly string[];
  frame: readonly FrameItems[];
  est: Estimates;
  state: 'ready' | 'running' | 'done' | 'stopped';
  elapsedMs: number | null;
  hiddenDuringRun: boolean;
  /** Something wrong before any row could run: no index, an unreadable manifest. */
  problems: readonly string[];
  /** Models the per-frame estimate has to leave out: a manifest that could not be read, one without per_frame. */
  leftOut?: readonly string[];
  /** The page was opened with ?quick=1. */
  quick?: boolean;
}

/** The whole summary as text. */
export function renderSummary(s: SummaryInput): string {
  const state =
    s.state === 'done' ? `done in ${((s.elapsedMs ?? 0) / 1000).toFixed(0)} s` : s.state === 'stopped' ? 'STOPPED, results are partial' : s.state === 'running' ? 'RUNNING, results so far' : 'not run yet';
  const head = [
    `RiftEye bench ${s.env.benchVersion} | onnxruntime-web ${s.env.ortVersion} | ${s.env.date} | ${state}`,
    ...envLines(s.env),
  ];
  if (s.state === 'ready') return [...head, '', 'Press Run.'].join('\n');
  const lines = [
    ...head,
    ...(s.rows.length ? [workerLine(s.rows)] : []),
    `tab hidden during the run: ${s.hiddenDuringRun ? 'YES (timings may be off)' : 'no'}`,
    ...(s.quick ? ['QUICK MODE (?quick=1): one warm-up and one timed run per batch, so the medians are single runs.'] : []),
    '',
    'ms, median/p90. bN = batch N: run = per run, item = per item (run / N), first = the first run at that batch (compiles shaders).',
    'init = starting the runtime (wasm, GPU device), load = creating the session, first = the very first run.',
    'runtimes: webgpu = onnxruntime-web native WebGPU EP (JSPI build), webgpu-jsep = its JSEP WebGPU EP, wasm = its plain WASM (CPU) build. Each row runs in a fresh worker.',
    '! = an error in that row, ~ = a warning the runtime logged while it ran.',
    ...s.problems.map((p) => `PROBLEM: ${p}`),
  ];
  for (const r of sortRows(s.rows, s.modelOrder)) {
    lines.push(rowLine(r));
    for (const e of r.errors) lines.push(`  ! ${e}`);
    for (const w of r.warnings) lines.push(`  ~ ${w}`);
  }
  lines.push('', ...frameLines(s.frame, s.est, s.leftOut ?? []));
  return lines.join('\n');
}
