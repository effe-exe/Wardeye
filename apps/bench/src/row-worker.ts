// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The worker loop: the page sends one row to run, the worker runs it in the given onnxruntime-web build and
// sends progress lines and then the finished row back. A worker runs one row and is then thrown away, so every
// row starts with a fresh wasm module and GPU device, and a row that hangs costs only its own worker.

import { runRow, describe, type Ort } from './row-runner';
import { blankRow, type FromWorker, type RowJob, type ToWorker } from './types';

export function serve(ort: Ort): void {
  const post = (m: FromWorker) => self.postMessage(m);
  self.onmessage = async (e: MessageEvent<ToWorker>) => {
    if (e.data.kind !== 'run') return;
    const job: RowJob = e.data.job;
    try {
      const row = await runRow(ort, job, {
        read: async (name) => {
          const res = await fetch(new URL(name, job.modelsUrl));
          if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
          return new Uint8Array(await res.arrayBuffer());
        },
        now: () => performance.now(),
        progress: (text) => post({ kind: 'progress', text }),
        isolated: self.crossOriginIsolated,
      });
      post({ kind: 'done', row });
    } catch (e2) {
      const row = blankRow(job);
      row.status = 'error';
      row.errors.push(`worker: ${describe(e2)}`);
      post({ kind: 'done', row });
    }
  };
}
