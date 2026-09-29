import { describe, expect, it } from 'vitest';
import { bestBatch, estimateFrame, estimates, timedRows, type TimedRow } from '../src/estimate';
import { blankRow } from '../src/types';

const row = (model: string, precision: string, runtime: TimedRow['runtime'], medians: [number, number][], passed = true): TimedRow => ({
  model,
  precision,
  runtime,
  passed,
  medians: medians.map(([batch, medianMs]) => ({ batch, medianMs })),
});
const frame = [
  { model: 'det', items: 1 },
  { model: 'emb', items: 12 },
];

describe('the best batch', () => {
  it('is the batch that gives the least ceil(items / b) x median(b)', () => {
    // 12 crops: 12 x 10 = 120 at b1, 2 x 30 = 60 at b8, 1 x 45 = 45 at b16
    const best = bestBatch(12, [{ batch: 1, medianMs: 10 }, { batch: 8, medianMs: 30 }, { batch: 16, medianMs: 45 }]);
    expect(best).toEqual({ batch: 16, calls: 1, callMs: 45, ms: 45 });
  });

  it('may be a small batch: 12 crops in 2 calls of b8 (2 x 20) beat 1 call of b16 (50)', () => {
    expect(bestBatch(12, [{ batch: 8, medianMs: 20 }, { batch: 16, medianMs: 50 }])).toEqual({ batch: 8, calls: 2, callMs: 20, ms: 40 });
  });

  it('takes the bigger batch on a tie, and ignores batches that were not timed', () => {
    expect(bestBatch(4, [{ batch: 2, medianMs: 10 }, { batch: 4, medianMs: 20 }])?.batch).toBe(4);
    expect(bestBatch(4, [{ batch: 1, medianMs: NaN }, { batch: 4, medianMs: 9 }])?.batch).toBe(4);
    expect(bestBatch(4, [])).toBeNull();
  });
});

describe('a frame', () => {
  const rows = [
    row('det', 'fp32', 'webgpu', [[1, 30]]),
    row('det', 'fp16', 'webgpu', [[1, 12]]),
    row('emb', 'fp16', 'webgpu', [[1, 6], [8, 20], [16, 30]]),
    row('det', 'fp32', 'wasm', [[1, 200]]),
    row('emb', 'fp32', 'wasm', [[1, 90], [8, 500]]),
  ];

  it('costs each model at its best batch and best passed precision, added up', () => {
    const r = estimateFrame('webgpu', rows, frame);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.estimate.parts).toMatchObject([
      { model: 'det', precision: 'fp16', items: 1, batch: 1, calls: 1, ms: 12 },
      { model: 'emb', precision: 'fp16', items: 12, batch: 16, calls: 1, ms: 30 },
    ]);
    expect(r.estimate.frameMs).toBe(42);
    expect(r.estimate.readsPerSec).toBeCloseTo(1000 / 42, 10);
  });

  it('leaves out a row whose check did not pass', () => {
    const r = estimateFrame('webgpu', [row('det', 'fp16', 'webgpu', [[1, 1]], false), ...rows.filter((x) => x.model !== 'det' || x.precision === 'fp32')], frame);
    expect(r.ok && r.estimate.parts[0]).toMatchObject({ precision: 'fp32', ms: 30 });
  });

  it('is not available when a model has no passed row on the runtime', () => {
    const r = estimateFrame('webgpu-jsep', rows, frame);
    expect(r).toEqual({ ok: false, reason: 'no row of det on webgpu-jsep has a passed check and timings' });
    expect(estimateFrame('webgpu', [], [])).toEqual({ ok: false, reason: 'no model states its per-frame items' });
  });

  it('is given for the fastest runtime, and for WASM alone', () => {
    const e = estimates(rows, frame, ['webgpu', 'webgpu-jsep', 'wasm']);
    expect(e.fastest.ok && e.fastest.estimate.runtime).toBe('webgpu');
    expect(e.wasm.ok && e.wasm.estimate.frameMs).toBe(200 + 2 * 500); // 2 calls of b8 (1000) beat 12 calls of b1 (1080)
    expect(e.all).toHaveLength(3);
  });

  it('says why when no runtime has an estimate', () => {
    const e = estimates([], frame, ['webgpu', 'wasm']);
    expect(e.fastest.ok).toBe(false);
    expect(!e.fastest.ok && e.fastest.reason).toContain('no row of det on webgpu');
    expect(estimates([], [], ['webgpu', 'wasm']).fastest).toEqual({ ok: false, reason: 'no model states its per-frame items' }); // said once, not once per runtime
    expect(e.wasm.ok).toBe(false);
  });
});

describe('rows to estimate from', () => {
  it('are the rows that ran on a runtime, with the batches that were timed', () => {
    const ran = blankRow({ model: 'det', title: 'Det', precision: 'fp16', runtime: 'wasm' });
    ran.batches = [
      { batch: 1, firstMs: 5, runs: 20, medianMs: 12, p90Ms: 14, perItemMs: 12, p90PerItemMs: 14, error: null },
      { batch: 8, firstMs: null, runs: 0, medianMs: null, p90Ms: null, perItemMs: null, p90PerItemMs: null, error: 'out of memory' },
    ];
    ran.check = { metric: 'cosine', batch: 8, value: 1, pass: true, outputs: [], note: '', error: null };
    const skipped = { ...blankRow({ model: 'det', title: 'Det', precision: 'fp32', runtime: 'all' }), status: 'skipped' as const };
    const noCheck = blankRow({ model: 'emb', title: 'Emb', precision: 'fp16', runtime: 'wasm' });
    expect(timedRows([ran, skipped, noCheck])).toEqual([
      { model: 'det', precision: 'fp16', runtime: 'wasm', passed: true, medians: [{ batch: 1, medianMs: 12 }] },
      { model: 'emb', precision: 'fp16', runtime: 'wasm', passed: false, medians: [] },
    ]);
  });
});
