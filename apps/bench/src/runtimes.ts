// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The ways the bench runs a model. onnxruntime-web 1.30 ships its WebGPU support twice: the native WebGPU
// execution provider (the recommended one; the JSPI build) and JSEP (the older one, kept for comparison), and a
// plain WASM build with no GPU code. Each is a separate bundle in a worker of its own (bench-*.ts), and the
// runtime files it loads are copied into ort/ by build.mjs.

export type RuntimeId = 'webgpu' | 'webgpu-jsep' | 'wasm';
export type Ep = 'webgpu' | 'wasm';

export interface Runtime {
  id: RuntimeId;
  /** The execution provider asked of onnxruntime-web. */
  ep: Ep;
  label: string;
  /** The worker script this runtime runs in (in dist/). */
  worker: string;
  /** The runtime build it loads from ort/. */
  files: string;
}

// Run order: the GPU first, the slow CPU last, so a bench that is stopped early has the GPU numbers.
export const RUNTIMES: readonly Runtime[] = [
  { id: 'webgpu', ep: 'webgpu', label: 'WebGPU (native EP, JSPI)', worker: 'bench-webgpu.js', files: 'ort-wasm-simd-threaded.jspi' },
  { id: 'webgpu-jsep', ep: 'webgpu', label: 'WebGPU (JSEP)', worker: 'bench-jsep.js', files: 'ort-wasm-simd-threaded.jsep' },
  { id: 'wasm', ep: 'wasm', label: 'WASM (CPU)', worker: 'bench-wasm.js', files: 'ort-wasm-simd-threaded' },
];

export function runtimeOf(id: RuntimeId): Runtime {
  const r = RUNTIMES.find((x) => x.id === id);
  if (!r) throw new Error(`unknown runtime ${id}`);
  return r;
}
