// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Standalone or companion: what the browser and the package allow. The engine runs on WebGPU (native, JSPI) or on
// plain WASM, each model in the precision that works: the detector in float32 (its float16 file fails on native
// WebGPU: measured on an M-series Mac, 34 of 35 cards, one score 0.71 to 0.46 and three extra cards, and JSEP breaks
// it altogether), the embedder in float16 where the GPU has shader-f16 (cosine 0.9998 to the float32 file). A browser
// without WebGPU, or a package without models, is served by the live runner on this machine (companion mode). The
// Chrome Web Store build has no live runner to fall back on: a browser without WebGPU runs plain WASM (slowly), and when
// there is nothing to try the engine is unavailable, which the overlay says.

import type { ModelFiles, RuntimeSetting, StandalonePackage } from './assets';

export type Runtime = 'webgpu' | 'wasm';
export type Precision = 'fp16' | 'fp32';

/** What the browser can do, as the engine's own document finds it. */
export interface Capabilities {
  /** An adapter came back from navigator.gpu.requestAdapter(). */
  webgpu: boolean;
  shaderF16: boolean;
  /** WebAssembly JSPI: what the native WebGPU build of onnxruntime-web needs. */
  jspi: boolean;
  /** What the probe found of the GPU, in words: the adapter's vendor and architecture, or why there is none. */
  gpu?: string;
}

/** One way to run the engine: the runtime, and the precision of each model's file. */
export interface Attempt {
  runtime: Runtime;
  detector: Precision;
  embedder: Precision;
}

export interface Plan {
  /** What to try, in order: the first that loads is used. Empty: companion mode; in the store build, unavailable. */
  attempts: Attempt[];
  /** Why there is nothing to try (or why only some things are). */
  reason: string;
}

const has = (m: ModelFiles, p: Precision): boolean => Boolean(m[p]);

/** The ways to run the engine, best first. WebGPU where the browser has it and can run the native build; WASM after
 * it, for a browser that has WebGPU but cannot run the native build (or whose GPU cannot run the embedder in the
 * precision the package holds); WASM alone when the package says so. With `store` (the Chrome Web Store build, which has no
 * live runner) WASM is also the way for a browser with no WebGPU at all, and a package that says to use the live runner
 * has nothing to try. */
export function plan(pkg: StandalonePackage, caps: Capabilities, setting: RuntimeSetting = pkg.runtime, store = false): Plan {
  if (setting === 'companion') {
    return { attempts: [], reason: store ? 'standalone.json says to use the live runner, which this build does not have' : 'standalone.json says to use the live runner' };
  }
  const { detector, embedder } = pkg;
  const attempts: Attempt[] = [];
  const why: string[] = [];
  if (setting === 'auto' || setting === 'webgpu') {
    if (!caps.webgpu) why.push(`this browser has no WebGPU adapter${caps.gpu ? ` (${caps.gpu})` : ''}`);
    else if (!caps.jspi) why.push('this browser has no WebAssembly JSPI (the native WebGPU runtime needs it)');
    else if (!has(detector, 'fp32')) why.push('the package holds no float32 detector (its float16 file fails on WebGPU)');
    else if (caps.shaderF16 && has(embedder, 'fp16')) attempts.push({ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' });
    else if (has(embedder, 'fp32')) attempts.push({ runtime: 'webgpu', detector: 'fp32', embedder: 'fp32' });
    else why.push('the GPU has no shader-f16 and the package holds no float32 embedder');
  }
  // plain WASM: asked for, or the way on for a browser that has WebGPU but could not use it (or, with no live runner to
  // leave the frames to, one that has none); float32 files first (a float16 file is computed in float32 inside, and slower)
  if (setting === 'wasm' || (setting === 'auto' && (caps.webgpu || store))) {
    const det = has(detector, 'fp32') ? 'fp32' : has(detector, 'fp16') ? 'fp16' : null;
    const emb = has(embedder, 'fp32') ? 'fp32' : has(embedder, 'fp16') ? 'fp16' : null;
    if (det && emb) attempts.push({ runtime: 'wasm', detector: det, embedder: emb });
    else why.push('the package holds no complete set of models');
  }
  return { attempts, reason: why.join('; ') };
}

/** How long a "not available" answer holds before the worker asks again (ms). */
export const RETRY_AFTER_MS = 5 * 60_000;
