// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// What the bench will do: one row per model x variant x runtime, and for each row whether it runs or why it is
// skipped (the variant is not in the folder, no WebGPU, no fp16 on this GPU). Pure: the page finds out what is
// present and what the browser can do, and passes it in.

import type { BenchManifest } from './model-manifest';
import { RUNTIMES, type RuntimeId } from './runtimes';

/** One line of models/index.json: its manifest, or why it could not be read. */
export interface ModelEntry {
  /** The manifest's file name, "<id>.bench.json". */
  name: string;
  manifest: BenchManifest | null;
  error: string | null;
}

export interface Support {
  /** An adapter came back from navigator.gpu.requestAdapter(). */
  webgpu: boolean;
  /** Why not, when not. */
  webgpuNote: string;
  shaderF16: boolean;
  /** WebAssembly JSPI, which the native WebGPU build needs (Chrome 137). */
  jspi: boolean;
}

export interface RowPlan {
  model: string;
  title: string;
  precision: string;
  runtime: RuntimeId | 'all';
  file: string;
  action: 'run' | 'skip' | 'error';
  note: string;
}

/** The rows in the order they run: by runtime (GPU first), then model, then variant. */
export function planRows(
  entries: readonly ModelEntry[],
  present: (file: string) => boolean,
  support: Support,
  runtimes: readonly RuntimeId[] = RUNTIMES.map((r) => r.id),
): RowPlan[] {
  const rows: RowPlan[] = [];
  const seen = new Set<string>();
  for (const e of entries) {
    if (!e.manifest) {
      rows.push({ model: e.name.replace(/\.bench\.json$/, ''), title: e.name, precision: '-', runtime: 'all', file: '', action: 'error', note: e.error ?? 'manifest could not be read' });
    }
  }
  for (const id of runtimes) {
    const runtime = RUNTIMES.find((r) => r.id === id);
    if (!runtime) continue;
    for (const { manifest: m } of entries) {
      if (!m) continue;
      for (const v of m.variants) {
        const base = { model: m.id, title: m.title, precision: v.precision, file: v.file };
        if (!present(v.file)) {
          // one row for the variant, not one per runtime
          const key = `${m.id}/${v.precision}`;
          if (!seen.has(key)) {
            seen.add(key);
            rows.push({ ...base, runtime: 'all', action: 'skip', note: 'not included' });
          }
        } else if (runtime.ep === 'webgpu' && !support.webgpu) {
          rows.push({ ...base, runtime: id, action: 'skip', note: support.webgpuNote || 'WebGPU is not available' });
        } else if (id === 'webgpu' && !support.jspi) {
          rows.push({ ...base, runtime: id, action: 'skip', note: 'this Chrome has no WebAssembly JSPI (Chrome 137 or later has it)' });
        } else if (runtime.ep === 'webgpu' && v.precision === 'fp16' && !support.shaderF16) {
          rows.push({ ...base, runtime: id, action: 'skip', note: 'fp16 not run: this WebGPU adapter has no shader-f16' });
        } else {
          rows.push({ ...base, runtime: id, action: 'run', note: '' });
        }
      }
    }
  }
  return rows;
}
