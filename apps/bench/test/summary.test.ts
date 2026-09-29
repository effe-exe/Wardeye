import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import type { OutputCheck } from '../src/compare';
import { estimates } from '../src/estimate';
import { checkText, fmtMs, fmtValue, frameLines, renderSummary, rowLine, sortRows, workerLine, type EnvInfo } from '../src/summary';
import { blankRow, type BatchResult, type CheckSummary, type RowResult } from '../src/types';

// The bench's own version comes from its manifest (build.mjs hands it to the page), so the summary's first line is checked
// against that and no version is written out here.
const VERSION = (JSON.parse(readFileSync(fileURLToPath(new URL('../src/manifest.json', import.meta.url)), 'utf8')) as { version: string }).version;

const env: EnvInfo = {
  date: '2026-09-28T12:00:00.000Z',
  benchVersion: VERSION,
  ortVersion: '1.30.0',
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/141.0.0.0',
  platform: 'macOS 15.1 arm 64-bit',
  cores: 10,
  crossOriginIsolated: true,
  sharedArrayBuffer: true,
  threads: 4,
  jspi: true,
  battery: 'battery charging 100%',
  gpu: {
    available: true,
    note: '',
    vendor: 'apple',
    architecture: 'metal-3',
    device: '',
    description: '',
    shaderF16: true,
    features: ['shader-f16', 'subgroups'],
    maxBufferSize: 4294967296,
    maxStorageBufferBindingSize: 4294967292,
  },
};

const batch = (b: number, median: number, p90: number, first: number): BatchResult => ({
  batch: b, firstMs: first, runs: 20, medianMs: median, p90Ms: p90, perItemMs: median / b, p90PerItemMs: p90 / b, error: null,
});
const out = (name: string, value: number, tolerance: number | null, pass: boolean, extra: Partial<OutputCheck> = {}): OutputCheck => ({
  name, value, tolerance, pass, over: tolerance === null ? null : pass ? 0 : 1, total: 8, median: null, ...extra,
});
const check = (over: Partial<CheckSummary> = {}): CheckSummary => ({
  metric: 'cosine', batch: 8, value: 0.999988, pass: true, outputs: [out('embedding', 0.999988, 0.999, true)], note: '', error: null, ...over,
});
const ran = (model: string, precision: string, runtime: RowResult['runtime'], over: Partial<RowResult> = {}): RowResult => ({
  ...blankRow({ model, title: model, precision, runtime }),
  initMs: 210, loadMs: 1450.4, firstMs: 3120, isolated: true, threads: 4, batches: [batch(1, 12.34, 14.5, 3120), batch(8, 40, 45, 210)], check: check(), ...over,
});

describe('numbers', () => {
  it('keep the digits that mean something', () => {
    expect(fmtMs(1234.56)).toBe('1235');
    expect(fmtMs(123.4)).toBe('123');
    expect(fmtMs(12.34)).toBe('12.3');
    expect(fmtMs(1.234)).toBe('1.23');
    expect(fmtMs(null)).toBe('-');
    expect(fmtMs(NaN)).toBe('-');
  });

  it('show a cosine to 6 decimals, and a maxabs to 3 digits', () => {
    expect(fmtValue('cosine', 0.9999876)).toBe('0.999988');
    expect(fmtValue('maxabs', 0.012345)).toBe('0.0123');
    expect(fmtValue('maxabs', 0)).toBe('0');
    expect(fmtValue('maxabs', 0.00001234)).toBe('1.23e-5');
    expect(fmtValue('maxabs', NaN)).toBe('NaN');
  });
});

describe('a check', () => {
  it('reads as its value against the tolerance, and the verdict', () => {
    expect(checkText(check())).toBe('cosine 0.999988 >= 0.999 PASS');
    expect(checkText(check({ pass: false, outputs: [out('embedding', 0.9, 0.999, false, { over: 2, total: 8 })] }))).toBe('cosine 0.900000 < 0.999 [2 of 8 rows under] FAIL');
  });

  it('names each output when there are several', () => {
    const c = check({
      metric: 'maxabs',
      pass: false,
      outputs: [
        out('pred_logits', 0.01, 0.5, true),
        out('pred_boxes', 0.2, 0.05, false, { over: 3, total: 400, median: 0.00002 }),
      ],
    });
    // a failing output says how far off it is: 3 values far off over a tiny median is a swapped query
    expect(checkText(c)).toBe('maxabs pred_logits 0.01 <= 0.5; pred_boxes 0.2 > 0.05 [3 of 400 values over, median diff 2.00e-5] FAIL');
  });

  it('says when it could not be made, or has no tolerance', () => {
    expect(checkText(check({ error: 'check.bin holds 3 values' }))).toBe('not checked (check.bin holds 3 values)');
    expect(checkText(check({ pass: false, outputs: [out('e', 0.99, null, false)] }))).toBe('cosine 0.990000 (no tolerance) FAIL');
  });
});

describe('a row', () => {
  it('is one line: model, precision, runtime, times, batches, check', () => {
    expect(rowLine(ran('detector-v0', 'fp16', 'webgpu'))).toBe(
      'detector-v0 | fp16 | webgpu | init 210 | load 1450 | first 3120 | b1 run 12.3/14.5 item 12.3/14.5 first 3120 b8 run 40.0/45.0 item 5.00/5.63 first 210 | check cosine 0.999988 >= 0.999 PASS',
    );
  });

  it('names the model file it loaded: size and the start of its sha256', () => {
    expect(rowLine(ran('det', 'fp16', 'wasm', { modelBytes: 73558108, modelSha: '3fa9c2d1e07b' }))).toContain('load 1450 (73.6 MB, sha256 3fa9c2d1e07b) | first 3120');
    expect(rowLine(ran('det', 'fp16', 'wasm', { modelBytes: 73558108 }))).toContain('load 1450 (73.6 MB) | first');
  });

  it('says a skipped variant is skipped, and why', () => {
    expect(rowLine({ ...blankRow({ model: 'emb', title: 'e', precision: 'fp32', runtime: 'all' }), status: 'skipped', note: 'not included' })).toBe('emb | fp32 | - | not included');
  });

  it('shows what a failed batch and a failed load left', () => {
    const r = ran('emb', 'fp32', 'wasm', { batches: [batch(1, 5, 6, 7), { ...batch(8, 0, 0, 0), medianMs: null, p90Ms: null, error: 'out of memory' }] });
    expect(rowLine(r)).toContain('b8 ERROR');
    const none = ran('emb', 'fp32', 'wasm', { loadMs: null, firstMs: null, batches: [], check: null });
    expect(rowLine(none)).toBe('emb | fp32 | wasm | init 210 | load - | first - | no timings | no check');
  });
});

describe('the order of rows', () => {
  it('is by model as listed, then fp32 before fp16, then GPU before CPU', () => {
    const rows = [ran('emb', 'fp16', 'wasm'), ran('det', 'fp16', 'wasm'), ran('det', 'fp32', 'wasm'), ran('det', 'fp16', 'webgpu'), ran('emb', 'fp32', 'webgpu-jsep')];
    expect(sortRows(rows, ['det', 'emb']).map((r) => `${r.model} ${r.precision} ${r.runtime}`)).toEqual([
      'det fp32 wasm', 'det fp16 webgpu', 'det fp16 wasm', 'emb fp32 webgpu-jsep', 'emb fp16 wasm',
    ]);
  });
});

describe('workers', () => {
  it('are reported as cross-origin isolated, with the threads the runtime took', () => {
    expect(workerLine([ran('a', 'fp32', 'wasm'), ran('b', 'fp32', 'wasm')])).toBe('workers: crossOriginIsolated yes | wasm threads used 4');
    expect(workerLine([ran('a', 'fp32', 'wasm'), ran('b', 'fp32', 'wasm', { isolated: false, threads: 1 })])).toBe('workers: crossOriginIsolated NO in 1 of 2 rows | wasm threads used 1/4');
  });
});

describe('the summary', () => {
  const frame = [{ model: 'det', items: 1 }, { model: 'emb', items: 12 }];
  const rows = [
    ran('det', 'fp16', 'webgpu', { batches: [batch(1, 12, 14, 30)] }),
    ran('emb', 'fp16', 'webgpu', { batches: [batch(8, 20, 22, 30), batch(16, 30, 33, 40)] }),
    ran('det', 'fp16', 'wasm', { batches: [batch(1, 100, 110, 150)] }),
    ran('emb', 'fp16', 'wasm', { batches: [batch(1, 50, 55, 60)] }),
  ];
  const timed = rows.map((r) => ({ model: r.model, precision: r.precision, runtime: r.runtime as 'webgpu' | 'wasm', passed: true, medians: r.batches.map((b) => ({ batch: b.batch, medianMs: b.medianMs! })) }));
  const est = estimates(timed, frame, ['webgpu', 'wasm']);

  it('has the environment, a line per row, and the per-frame estimate', () => {
    const text = renderSummary({ env, rows, modelOrder: ['det', 'emb'], frame, est, state: 'done', elapsedMs: 143_400, hiddenDuringRun: false, problems: [] });
    const lines = text.split('\n');
    expect(lines[0]).toBe(`Wardeye bench ${VERSION} | onnxruntime-web 1.30.0 | 2026-09-28T12:00:00.000Z | done in 143 s`);
    expect(text).toContain('browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/141.0.0.0');
    expect(text).toContain('cores 10 | crossOriginIsolated yes | SharedArrayBuffer yes | wasm threads asked 4 | JSPI yes | battery charging 100%');
    expect(text).toContain('WebGPU: yes | vendor apple | architecture metal-3 | device - | description -');
    expect(text).toContain('WebGPU: shader-f16 yes | maxBufferSize 4294967296 | maxStorageBufferBindingSize 4294967292');
    expect(text).toContain('tab hidden during the run: no');
    expect(lines.filter((l) => /^(det|emb) \| /.test(l))).toHaveLength(4);
    expect(text).toContain('per frame (det 1 item/frame, emb 12 items/frame;');
    // webgpu: det 12 ms + emb 12 items in 2 x b8 @ 20 = 40 (b16 @ 30 = 30 is cheaper: 1 x 30)
    expect(text).toContain('  fastest: webgpu: det fp16 1 item = 1 x b1 @ 12.0 = 12.0 ms + emb fp16 12 items = 1 x b16 @ 30.0 = 30.0 ms = 42.0 ms/frame -> 23.8 reads/s');
    expect(text).toContain('  wasm only: wasm: det fp16 1 item = 1 x b1 @ 100 = 100 ms + emb fp16 12 items = 12 x b1 @ 50.0 = 600 ms = 700 ms/frame -> 1.4 reads/s');
  });

  it('says when it was a quick run', () => {
    const base = { env, rows, modelOrder: ['det', 'emb'], frame, est, state: 'done' as const, elapsedMs: 1000, hiddenDuringRun: false, problems: [] };
    expect(renderSummary({ ...base, quick: true })).toContain('QUICK MODE (?quick=1): one warm-up and one timed run per batch');
    expect(renderSummary(base)).not.toContain('QUICK');
  });

  it('says when it is partial, when the tab was hidden, when there is no WebGPU, and lists problems and errors', () => {
    const failed = ran('det', 'fp16', 'wasm', { errors: ['load: protobuf parsing failed'], warnings: ['Some nodes were not assigned to the preferred execution providers.'], status: 'error' });
    const text = renderSummary({
      env: { ...env, gpu: { ...env.gpu, available: false, note: 'navigator.gpu is missing' } },
      rows: [failed],
      modelOrder: ['det'],
      frame,
      est: estimates([], frame, ['wasm']),
      state: 'stopped',
      elapsedMs: 1000,
      hiddenDuringRun: true,
      problems: ['models/index.json lists no model.'],
    });
    expect(text.split('\n')[0]).toContain('STOPPED, results are partial');
    expect(text).toContain('WebGPU: no (navigator.gpu is missing)');
    expect(text).not.toContain('shader-f16');
    expect(text).toContain('tab hidden during the run: YES (timings may be off)');
    expect(text).toContain('PROBLEM: models/index.json lists no model.');
    expect(text).toContain('  ! load: protobuf parsing failed');
    expect(text).toContain('  ~ Some nodes were not assigned to the preferred execution providers.');
    expect(text).toContain('  fastest: not available (');
  });

  it('is short before a run: the browser, and Press Run', () => {
    const text = renderSummary({ env, rows: [], modelOrder: [], frame: [], est: estimates([], [], []), state: 'ready', elapsedMs: null, hiddenDuringRun: false, problems: [] });
    expect(text.split('\n')[0]).toContain('not run yet');
    expect(text).toContain('WebGPU: yes');
    expect(text.endsWith('Press Run.')).toBe(true);
    expect(text).not.toContain('per frame');
  });

  it('gives a slow machine\'s reads a second to two decimals', () => {
    const slow = estimates([{ model: 'det', precision: 'fp32', runtime: 'wasm', passed: true, medians: [{ batch: 1, medianMs: 20000 }] }], [{ model: 'det', items: 1 }], ['wasm']);
    expect(frameLines([{ model: 'det', items: 1 }], slow)).toContain('  wasm only: wasm: det fp32 1 item = 1 x b1 @ 20000 = 20000 ms = 20000 ms/frame -> 0.05 reads/s');
  });

  it('describes the estimate lines alone', () => {
    expect(frameLines(frame, est)[0]).toContain('each model at its best batch; only rows whose check passed');
  });

  it('warns that a model the estimate had to leave out makes it less than a whole frame', () => {
    const lines = frameLines(frame, est, ['emb.bench.json could not be read', 'cls has no per_frame items']);
    expect(lines[1]).toBe('  NOT A WHOLE FRAME: emb.bench.json could not be read; cls has no per_frame items');
    expect(frameLines(frame, est)).toHaveLength(3);
  });
});
