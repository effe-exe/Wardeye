// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// ONNX Runtime Web sessions for the detector and the embedder, made the way the extension makes them. The GPU
// path is onnxruntime-web 1.30's native WebGPU execution provider (its JSPI build, "onnxruntime-web/jspi", the
// bench's "webgpu" runtime), with a model's float16 file when the adapter has "shader-f16" and its float32 file
// otherwise; the older JSEP WebGPU build is not used (it breaks GridSample in float16). The fallback is the plain
// WASM build ("onnxruntime-web/wasm") with the float32 file. A session is made for one batch size
// (freeDimensionOverrides on the graphs' "batch" axis), so it never meets a new shape: nothing is recompiled after
// the first run.
//
// The onnxruntime-web modules are passed in, never imported here, so the engine has no dependency on them: the
// caller imports the builds it ships (and bundles their .mjs and .wasm files, which `wasmPaths` points at).

import { Detector, type DetectorOptions, type HeadOutputs, type TileRunner } from './detector';
import { OnnxEncoder, sha256Hex, stem, type EmbedRunner, type OnnxEncoderOptions } from './embedder';

// ---- the parts of onnxruntime-web the engine uses (typeof import('onnxruntime-web') fits them)

export interface OrtTensor {
  readonly data: unknown;
  readonly dims: readonly number[];
  readonly type: string;
}

export interface OrtSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
}

export interface OrtSessionOptions {
  executionProviders?: readonly string[];
  freeDimensionOverrides?: Record<string, number>;
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
}

export interface OrtModule {
  env: { wasm: { wasmPaths?: unknown; numThreads?: number; proxy?: boolean }; versions?: { web?: string } };
  InferenceSession: { create(model: Uint8Array, options?: OrtSessionOptions): Promise<OrtSession> };
  Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => OrtTensor;
}

/** What the engine needs of navigator.gpu. */
export interface GpuLike {
  requestAdapter(): Promise<{ features: { has(feature: string): boolean } } | null>;
}

// ---- choosing the runtime

export type Ep = 'webgpu' | 'wasm';
export type Precision = 'fp32' | 'fp16';

export interface RuntimeOptions {
  /** onnxruntime-web/jspi: the native WebGPU build. Leave it out for WASM only. */
  webgpu?: OrtModule;
  /** onnxruntime-web/wasm: the plain WASM build, the fallback. */
  wasm: OrtModule;
  /** Where the builds' ort-wasm-simd-threaded*.mjs and .wasm files are served from, ending in "/". */
  wasmPaths?: string;
  /** WASM threads (more than one needs a cross-origin isolated page); onnxruntime-web's own default otherwise. */
  threads?: number;
  /** navigator.gpu, by default the global one. */
  gpu?: GpuLike | null;
  /** Skip WebGPU even where it works (tests, a user setting). */
  wasmOnly?: boolean;
}

/** The runtime a page uses: WebGPU when it can, float16 when the GPU can, and the WASM build to fall back on. */
export interface Runtime {
  ep: Ep;
  precision: Precision;
  /** The build for `ep`. */
  ort: OrtModule;
  /** The WASM build, for a model that fails on WebGPU. */
  wasm: OrtModule;
  /** Why this runtime: for a status line and bug reports. */
  reason: string;
}

function configure(ort: OrtModule, o: RuntimeOptions): void {
  if (o.wasmPaths !== undefined) ort.env.wasm.wasmPaths = o.wasmPaths;
  if (o.threads !== undefined) ort.env.wasm.numThreads = o.threads;
  ort.env.wasm.proxy = false; // the engine runs in the extension's own worker already
}

/** WebGPU with the JSPI build when the browser has WebGPU, JSPI and an adapter; float16 when the adapter has
 * "shader-f16". Otherwise the WASM build and float32. */
export async function chooseRuntime(o: RuntimeOptions): Promise<Runtime> {
  configure(o.wasm, o);
  const wasm = (reason: string): Runtime => ({ ep: 'wasm', precision: 'fp32', ort: o.wasm, wasm: o.wasm, reason });
  if (!o.webgpu) return wasm('no WebGPU build given');
  if (o.wasmOnly) return wasm('WASM asked for');
  configure(o.webgpu, o);
  const gpu = o.gpu !== undefined ? o.gpu : ((globalThis as { navigator?: { gpu?: GpuLike } }).navigator?.gpu ?? null);
  if (!gpu) return wasm('no WebGPU in this browser');
  if (typeof (WebAssembly as unknown as { Suspending?: unknown }).Suspending !== 'function') return wasm('no WebAssembly JSPI in this browser');
  let adapter: Awaited<ReturnType<GpuLike['requestAdapter']>> = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    adapter = null;
  }
  if (!adapter) return wasm('no WebGPU adapter');
  const f16 = adapter.features.has('shader-f16');
  return { ep: 'webgpu', precision: f16 ? 'fp16' : 'fp32', ort: o.webgpu, wasm: o.wasm, reason: f16 ? 'WebGPU with shader-f16' : 'WebGPU without shader-f16' };
}

// ---- sessions

/** A model file: its name (as exported, e.g. "embedder-v1.fp16.onnx") and its bytes, read only when needed. */
export interface ModelFile {
  name: string;
  bytes: () => Promise<Uint8Array> | Uint8Array;
}

/** A model's two files; without fp16 the float32 one runs everywhere. */
export interface ModelFiles {
  fp32: ModelFile;
  fp16?: ModelFile;
}

/** A session and what it runs on. */
export interface OpenModel {
  session: OrtSession;
  ort: OrtModule;
  ep: Ep;
  precision: Precision;
  file: string;
  /** The file's SHA-256 (hex), when asked for: the model bytes themselves are not kept. */
  sha256: string | null;
  /** The batch the session is fixed to (null: free). */
  batch: number | null;
  /** Set when WebGPU failed and the model fell back to WASM. */
  fallback: string | null;
}

/** Session options for one execution provider and, when `batch` is given, the graph's "batch" axis fixed to it. */
export function sessionOptions(ep: Ep, batch: number | null): OrtSessionOptions {
  return {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    ...(batch !== null ? { freeDimensionOverrides: { batch } } : {}),
  };
}

/** A session for a model on the runtime: its float16 file on a GPU with shader-f16, else its float32 file; on
 * WASM (float32) if the WebGPU session cannot be made. `hash` also takes the file's SHA-256. */
export async function openModel(rt: Runtime, files: ModelFiles, batch: number | null, hash = false): Promise<OpenModel> {
  const tryOpen = async (ort: OrtModule, ep: Ep, precision: Precision): Promise<OpenModel> => {
    const f = precision === 'fp16' && files.fp16 ? files.fp16 : files.fp32;
    const bytes = await f.bytes();
    const sha256 = hash ? await sha256Hex(bytes) : null;
    const session = await ort.InferenceSession.create(bytes, sessionOptions(ep, batch));
    return { session, ort, ep, precision: f === files.fp32 ? 'fp32' : 'fp16', file: f.name, sha256, batch, fallback: null };
  };
  if (rt.ep === 'wasm') return tryOpen(rt.wasm, 'wasm', 'fp32');
  try {
    return await tryOpen(rt.ort, 'webgpu', rt.precision);
  } catch (e) {
    const why = e instanceof Error ? e.message : String(e);
    return { ...(await tryOpen(rt.wasm, 'wasm', 'fp32')), fallback: `WebGPU failed: ${why}` };
  }
}

function float32(t: OrtTensor | undefined, name: string): Float32Array {
  if (!t) throw new Error(`the model has no output "${name}"`);
  if (!(t.data instanceof Float32Array)) throw new Error(`output "${name}" is ${t.type}, not float32`);
  return t.data;
}

/** Runs of one session one after another: a session is not re-entrant, and two frames' calls may overlap. */
function queued<A extends unknown[], R>(run: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
  let last: Promise<unknown> = Promise.resolve();
  return (...args) => {
    const next = last.then(() => run(...args));
    last = next.catch(() => undefined);
    return next;
  };
}

/** The detector's graph as a Detector's runner: "tiles" in, the three head outputs out. */
export function tileRunner(ort: OrtModule, session: OrtSession): TileRunner {
  return queued(async (tiles: Float32Array, n: number): Promise<HeadOutputs> => {
    const out = await session.run({ tiles: new ort.Tensor('float32', tiles, [n, 3, 576, 576]) });
    return {
      pred_logits: float32(out.pred_logits, 'pred_logits'),
      pred_boxes: float32(out.pred_boxes, 'pred_boxes'),
      pred_keypoints: float32(out.pred_keypoints, 'pred_keypoints'),
    };
  });
}

/** The embedder's graph as an encoder's runner: "crops" in, "embedding" out. */
export function embedRunner(ort: OrtModule, session: OrtSession, imgSize = 224): EmbedRunner {
  return queued(async (crops: Float32Array, n: number) => {
    const out = await session.run({ crops: new ort.Tensor('float32', crops, [n, 3, imgSize, imgSize]) });
    return float32(out.embedding, 'embedding');
  });
}

/** The detector on a runtime. One tile a run by default: every layout's frames then run the same shape, and a frame
 * of k tiles is k runs. */
export async function openDetector(rt: Runtime, files: ModelFiles, options: DetectorOptions = {}): Promise<{ detector: Detector; model: OpenModel }> {
  const batch = options.batch ?? 1;
  const model = await openModel(rt, files, batch);
  return { detector: new Detector(tileRunner(model.ort, model.session), { batch, pad: true }), model };
}

/** The embedder on a runtime, named as embed/onnx.py names the file it runs ("onnx:<stem>-<sha8>"). */
export async function openEncoder(rt: Runtime, files: ModelFiles, options: OnnxEncoderOptions = {}): Promise<{ encoder: OnnxEncoder; model: OpenModel }> {
  const batch = options.batch ?? 8;
  const model = await openModel(rt, files, batch, true);
  const name = `onnx:${stem(model.file)}-${model.sha256!.slice(0, 8)}`; // as embedder.encoderName
  return { encoder: new OnnxEncoder(name, embedRunner(model.ort, model.session, options.imgSize), { ...options, batch }), model };
}
