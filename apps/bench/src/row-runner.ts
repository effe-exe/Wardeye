// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// One row of the bench: one model, one precision, one runtime. Starts the runtime on a one-node model (so that
// loading wasm and the GPU is not counted as loading the model), loads the real model, times it at every batch
// size, checks its outputs against the expected ones, and releases the session. Every failure goes into the
// row's errors and the row still comes back. It takes the onnxruntime-web module and its file access as
// arguments, so it runs in a worker of the page, and in Node under vitest.

import type { InferenceSession } from 'onnxruntime-web';
import { decodeBin, evaluate, toFloat32, type OutputPair } from './compare';
import { batchesOf, elementCount, itemSize, outputCount, resolveOutShape, resolveShape } from './model-manifest';
import { measureRuns, summarize } from './stats';
import { startupModel } from './tiny-onnx';
import { blankRow, type BatchResult, type CheckSummary, type RowJob, type RowResult } from './types';

export type Ort = typeof import('onnxruntime-web');

export interface RowIO {
  /** The bytes of a file in the models folder. */
  read(name: string): Promise<Uint8Array>;
  /** A clock in ms. */
  now(): number;
  progress(text: string): void;
  isolated: boolean;
}

/** An error as one line: what was thrown may be an Error, a string, or a number from the wasm side. */
export function describe(e: unknown): string {
  if (e instanceof Error) return e.name && e.name !== 'Error' ? `${e.name}: ${e.message}` : e.message;
  if (typeof e === 'number') return `error code ${e} from the runtime`;
  return String(e);
}

const NOISE = /Removing initializer|CleanUnusedInitializers|Rerunning with verbose output/;
const ORT_LOG = /\[[WEF]:onnxruntime/;
// onnxruntime colours its lines and starts them with a time stamp: neither helps whoever reads the summary
const COLOUR = /\u001b\[[0-9;]*m/g;
const STAMP = /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(\.\d+)? /;

/**
 * Collects what the runtime prints while a row runs: console warnings and errors, and onnxruntime's own [W:...]
 * lines (they come through console.log or console.error). Up to `limit` different lines go into `sink`, and how
 * many more there were. Call it before the runtime starts (the wasm module binds console.* when it loads), and
 * call what it returns when the row is done.
 */
export function collectLogs(sink: string[], limit = 6): () => void {
  const methods = ['log', 'info', 'warn', 'error'] as const;
  const saved = methods.map((m) => console[m]);
  let more = 0;
  methods.forEach((m, i) => {
    console[m] = (...args: unknown[]) => {
      const line = args.map(String).join(' ').replace(COLOUR, '').replace(/\s+/g, ' ').trim().replace(STAMP, '');
      if (line && !NOISE.test(line) && (m === 'warn' || m === 'error' || ORT_LOG.test(line))) {
        if (sink.length < limit) {
          const text = line.length > 300 ? `${line.slice(0, 300)}...` : line;
          if (!sink.includes(text)) sink.push(text);
        } else {
          more++;
        }
      }
      saved[i]!.apply(console, args);
    };
  });
  return () => {
    methods.forEach((m, i) => (console[m] = saved[i]!));
    if (more) sink.push(`(${more} more)`);
  };
}

/** The first 12 hex digits of the SHA-256 of some bytes; null where crypto.subtle is not there. */
export async function shortSha(bytes: Uint8Array): Promise<string | null> {
  try {
    const digest = await crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
    return Array.from(new Uint8Array(digest).subarray(0, 6), (b) => b.toString(16).padStart(2, '0')).join('');
  } catch {
    return null;
  }
}

/** Made-up but fixed input, 0..255 like a picture, for a model whose check input cannot be read. */
function fillSynthetic(data: Float32Array | Uint8Array): void {
  let seed = 12345;
  for (let i = 0; i < data.length; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    data[i] = Math.floor((seed / 2 ** 32) * 256);
  }
}

export async function runRow(ort: Ort, job: RowJob, io: RowIO): Promise<RowResult> {
  const row = blankRow(job);
  if (job.ortBase) ort.env.wasm.wasmPaths = job.ortBase; // empty: the runtime's own default
  ort.env.wasm.numThreads = job.threads;
  ort.env.logLevel = 'warning';
  row.ortVersion = ort.env.versions.web ?? null;
  row.threads = job.threads;
  row.isolated = io.isolated;
  const stopLogs = collectLogs(row.warnings);
  try {
    await measure(ort, job, io, row);
  } finally {
    stopLogs();
  }
  return row;
}

/** The stages of a row, each recording its own failure in `row` and going on where it can. */
async function measure(ort: Ort, job: RowJob, io: RowIO, row: RowResult): Promise<void> {
  const m = job.manifest;
  const say = (text: string) => io.progress(`${job.model} ${job.precision} on ${job.runtime}: ${text}`);
  const fail = (stage: string, e: unknown) => {
    row.errors.push(`${stage}: ${describe(e)}`);
    row.status = 'error';
  };

  // 1. the runtime, on a model of one node
  say('starting the runtime');
  try {
    const t0 = io.now();
    const startup = await ort.InferenceSession.create(startupModel(), { executionProviders: [job.ep] });
    await startup.run({ x: new ort.Tensor('float32', new Float32Array(4), [1, 4]) });
    await startup.release();
    row.initMs = io.now() - t0;
    // the runtime falls back to one thread, and says so only in the console, when it cannot share memory
    row.threads = ort.env.wasm.numThreads ?? job.threads;
  } catch (e) {
    fail('runtime start', e);
    return;
  }

  // 2. the check files (a failure here costs the check, not the timings)
  const check = m.check;
  let checkInput: Float32Array | Uint8Array | null = null;
  const expected: Record<string, Float32Array> = {};
  let checkFilesError: string | null = null;
  if (check) {
    try {
      say('reading the check files');
      const shape = resolveShape(m.input.shape, check.batch);
      const data = decodeBin(await io.read(check.input), m.input.dtype);
      if (data.length !== elementCount(shape)) {
        throw new Error(`${check.input} holds ${data.length} values, the shape ${shape.join('x')} needs ${elementCount(shape)}`);
      }
      checkInput = data;
      for (const o of m.outputs) {
        const file = check.expected[o.name]!;
        const want = decodeBin(await io.read(file), 'float32') as Float32Array;
        const oshape = resolveOutShape(o.shape, check.batch);
        const { count, open } = outputCount(oshape);
        if (open ? want.length % count !== 0 : want.length !== count) {
          throw new Error(`${file} holds ${want.length} values, the shape ${oshape.join('x')} needs ${open ? `a multiple of ${count}` : count}`);
        }
        expected[o.name] = want;
      }
    } catch (e) {
      checkInput = null;
      checkFilesError = describe(e);
    }
  }

  // 3. the model
  let session: InferenceSession;
  try {
    say('reading the model');
    const t0 = io.now();
    const bytes = await io.read(job.file);
    row.fetchMs = io.now() - t0;
    row.modelBytes = bytes.byteLength;
    row.modelSha = await shortSha(bytes);
    say(`creating the session (${(bytes.byteLength / 1e6).toFixed(1)} MB)`);
    const t1 = io.now();
    session = await ort.InferenceSession.create(bytes, { executionProviders: [job.ep] });
    row.loadMs = io.now() - t1;
  } catch (e) {
    fail('load', e);
    if (check) row.check = failedCheck(check.metric, check.batch, 'not run: the model did not load');
    return;
  }

  // Input for a batch: the check input's items repeated to fill it, or made-up pixels when it cannot be read.
  const feedsFor = (batch: number, source: Float32Array | Uint8Array | null, sourceBatch: number) => {
    const shape = resolveShape(m.input.shape, batch);
    const size = elementCount(shape);
    const per = itemSize(m.input.shape);
    const data = m.input.dtype === 'uint8' ? new Uint8Array(size) : new Float32Array(size);
    if (source) {
      for (let i = 0; i < batch; i++) {
        const from = (i % sourceBatch) * per;
        (data as Float32Array).set((source as Float32Array).subarray(from, from + per), i * per);
      }
    } else {
      fillSynthetic(data);
    }
    const tensor = data instanceof Uint8Array ? new ort.Tensor('uint8', data, shape) : new ort.Tensor('float32', data, shape);
    return { [m.input.name]: tensor };
  };

  const timeBatch = async (batch: number) => {
    const result: BatchResult = { batch, firstMs: null, runs: 0, medianMs: null, p90Ms: null, perItemMs: null, p90PerItemMs: null, error: null };
    row.batches.push(result);
    try {
      const feeds = feedsFor(batch, checkInput, check?.batch ?? 1);
      const measured = await measureRuns(
        () => session.run(feeds),
        io.now,
        (phase, done, total) => say(`batch ${batch}, ${phase} ${done}/${total}`),
        job.quick ? { warmups: 1, repeats: 1 } : {},
      );
      const t = summarize(measured.samples, batch);
      Object.assign(result, {
        firstMs: measured.warmupMs[0] ?? null,
        runs: t.runs,
        medianMs: t.medianMs,
        p90Ms: t.p90Ms,
        perItemMs: t.perItemMs,
        p90PerItemMs: t.p90ItemMs,
      });
    } catch (e) {
      result.error = describe(e);
      fail(`batch ${batch}`, e);
    }
  };

  const runCheck = async () => {
    if (!check) return;
    const summary = failedCheck(check.metric, check.batch, null);
    row.check = summary;
    try {
      if (checkFilesError || !checkInput) throw new Error(checkFilesError ?? 'check input not read');
      say(`checking the outputs (batch ${check.batch})`);
      const out = await session.run(feedsFor(check.batch, checkInput, check.batch));
      const pairs: OutputPair[] = m.outputs.map((o) => {
        const t = out[o.name];
        if (!t) throw new Error(`the model has no output "${o.name}" (it has ${session.outputNames.join(', ')})`);
        const got = toFloat32(t.data, t.type);
        const want = expected[o.name]!;
        if (got.length !== want.length) {
          throw new Error(`output "${o.name}" has ${got.length} values (shape ${t.dims.join('x')}), expected ${want.length}`);
        }
        const last = resolveOutShape(o.shape, check.batch).at(-1)!;
        if (check.metric === 'cosine' && typeof last !== 'number') {
          throw new Error(`output "${o.name}": cosine needs a fixed last dimension, the manifest gives "${last}"`);
        }
        return { name: o.name, got, want, rowLength: typeof last === 'number' ? last : 1 };
      });
      Object.assign(summary, evaluate(check.metric, pairs, check.tolerance[job.precision]), { error: null });
    } catch (e) {
      summary.error = describe(e);
      fail('check', e);
    }
  };

  try {
    // the check comes right after the first batch: it matters more than the timings of the bigger batches
    for (const [i, batch] of batchesOf(m).entries()) {
      await timeBatch(batch);
      if (i === 0) {
        row.firstMs = row.batches[0]?.firstMs ?? null;
        await runCheck();
      }
    }
  } finally {
    try {
      await session.release();
    } catch (e) {
      fail('release', e);
    }
  }
}

function failedCheck(metric: CheckSummary['metric'], batch: number, error: string | null): CheckSummary {
  return { metric, batch, value: NaN, pass: false, outputs: [], note: '', error };
}
