// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// What one row of the bench (one model, one precision, one runtime) reports, and what the page and its
// workers say to each other. Plain data: it crosses postMessage.

import type { CheckResult } from './compare';
import type { DecodedCheck } from './decoded';
import type { BenchManifest } from './model-manifest';
import type { RuntimeId } from './runtimes';

export interface BatchResult {
  batch: number;
  /** The first warm-up run at this batch: for a new batch size it includes the shader compile. */
  firstMs: number | null;
  runs: number;
  medianMs: number | null;
  p90Ms: number | null;
  perItemMs: number | null;
  p90PerItemMs: number | null;
  error: string | null;
}

/** The check's outcome; `error` is set (and value is NaN) when it could not be made. */
export interface CheckSummary extends CheckResult {
  batch: number;
  error: string | null;
  /** The decoded detector check, when the manifest has check.detections: its own verdict (`pass` is the raw check's). */
  decoded?: DecodedCheck;
}

export interface RowResult {
  model: string;
  title: string;
  precision: string;
  /** 'all' for a variant that is not in the folder: one row for every runtime. */
  runtime: RuntimeId | 'all';
  status: 'ok' | 'error' | 'skipped';
  /** Why it was skipped, or another remark. */
  note: string;
  /** Every stage that failed, in order. */
  errors: string[];
  /** What the runtime logged as a warning or error while the row ran (nodes that fell back to the CPU, for one). */
  warnings: string[];
  ortVersion: string | null;
  threads: number | null;
  isolated: boolean | null;
  /** Runtime start: the wasm module, and the GPU device for WebGPU (measured on a tiny model). */
  initMs: number | null;
  fetchMs: number | null;
  /** The model file: its size and the start of its SHA-256, so that results can be tied to the exact export. */
  modelBytes: number | null;
  modelSha: string | null;
  /** Creating the session for the real model. */
  loadMs: number | null;
  /** The very first run, at the first batch. */
  firstMs: number | null;
  batches: BatchResult[];
  check: CheckSummary | null;
}

export interface RowJob {
  model: string;
  title: string;
  precision: string;
  runtime: RuntimeId;
  ep: 'webgpu' | 'wasm';
  manifest: BenchManifest;
  /** The variant's file name, next to the manifest. */
  file: string;
  /** Where the models folder is, ending in "/". */
  modelsUrl: string;
  /** Where the runtime files are, ending in "/". */
  ortBase: string;
  threads: number;
  /** One warm-up and one timed run per batch: to see that a model runs and checks, not for timings. */
  quick: boolean;
}

export type ToWorker = { kind: 'run'; job: RowJob };
export type FromWorker = { kind: 'progress'; text: string } | { kind: 'done'; row: RowResult };

export function blankRow(job: Pick<RowJob, 'model' | 'title' | 'precision'> & { runtime: RuntimeId | 'all' }): RowResult {
  return {
    model: job.model,
    title: job.title,
    precision: job.precision,
    runtime: job.runtime,
    status: 'ok',
    note: '',
    errors: [],
    warnings: [],
    ortVersion: null,
    threads: null,
    isolated: null,
    initMs: null,
    fetchMs: null,
    modelBytes: null,
    modelSha: null,
    loadMs: null,
    firstMs: null,
    batches: [],
    check: null,
  };
}
