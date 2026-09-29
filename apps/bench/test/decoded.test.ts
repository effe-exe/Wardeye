import * as ort from 'onnxruntime-web';
import { detector, type Detection } from '@rifteye/engine';
import { describe, expect, it } from 'vitest';
import { checkText } from '../src/summary';
import { cornerDistance, decodedText, judgeDetections, parseDetections } from '../src/decoded';
import { parseManifest } from '../src/model-manifest';
import { runRow, type RowIO } from '../src/row-runner';
import { encodeModel } from '../src/tiny-onnx';
import type { CheckSummary, RowJob } from '../src/types';

// The decoded detector check: the rule on made-up cards, the manifest's check.detections, and a whole row with the
// real onnxruntime-web (Node, WASM) on a stand-in detector whose outputs are made-up head outputs.

const card = (score: number, x: number, y: number, cls: Detection['cls'] = 'card'): Detection => ({
  cls,
  score,
  box: [x, y, x + 50, y + 70],
  quad: [
    [x, y],
    [x + 50, y],
    [x + 50, y + 70],
    [x, y + 70],
  ],
  found: [1, 1, 1, 1],
  visible: [1, 1, 1, 1],
});
const moved = (d: Detection, dx: number, score = d.score): Detection => ({ ...d, score, quad: d.quad.map(([x, y]) => [x + dx, y] as [number, number]) });
const tol = { corner_px: 2, score: 0.05 };

describe('the rule', () => {
  const want = [card(0.9, 0, 0), card(0.8, 100, 0), card(0.6, 200, 0, 'card_back'), card(0.43, 300, 0), card(0.35, 400, 0)];

  it('passes the same cards, reporting the share and the worst differences', () => {
    const got = [moved(want[0]!, 1.5, 0.87), moved(want[1]!, -0.5), want[2]!, want[3]!, want[4]!];
    const d = judgeDetections(got, want, 0.4, tol);
    expect(d).toMatchObject({ pass: true, required: 3, matched: 3, share: 1, missing: [], extra: [], near: 0 });
    expect(d.worstCornerPx).toBeCloseTo(1.5, 12);
    expect(d.worstScore).toBeCloseTo(0.03, 12);
    expect(decodedText(d)).toBe('decoded 3/3 cards above 0.45 (100.0%), worst corner 1.50 px, score 0.030 PASS');
  });

  it('fails a card that must be found and is not, or is found too far off', () => {
    expect(judgeDetections(want.slice(1), want, 0.4, tol)).toMatchObject({ pass: false, missing: ['card 0.900'], matched: 2 });
    expect(judgeDetections([moved(want[0]!, 3), ...want.slice(1)], want, 0.4, tol)).toMatchObject({ pass: false, missing: ['card 0.900'], extra: ['card 0.900'] });
    expect(judgeDetections([moved(want[0]!, 0, 0.7), ...want.slice(1)], want, 0.4, tol)).toMatchObject({ pass: false, missing: ['card 0.900 (found at 0.700)'] });
  });

  it('fails a card of the other class, and one nobody expected', () => {
    const other = { ...want[2]!, cls: 'card' as const };
    expect(judgeDetections([want[0]!, want[1]!, other, want[3]!], want, 0.4, tol)).toMatchObject({ pass: false, missing: ['card_back 0.600'], extra: ['card 0.600'] });
    expect(judgeDetections([...want, card(0.7, 600, 0)], want, 0.4, tol)).toMatchObject({ pass: false, extra: ['card 0.700'] });
  });

  it('lets cards near the threshold go missing on either side', () => {
    // 0.43 and 0.35 score below 0.4 + 0.05: either may be gone, and a stray at 0.44 may appear
    const d = judgeDetections([...want.slice(0, 3), card(0.44, 700, 0)], want, 0.4, tol);
    expect(d).toMatchObject({ pass: true, required: 3, matched: 3, near: 3 });
    expect(decodedText(d)).toContain('3 near the threshold unpaired PASS');
  });

  it('pairs one to one, closest corners first', () => {
    const a = card(0.9, 0, 0);
    const b = card(0.9, 1, 0);
    const d = judgeDetections([moved(a, 0.2), moved(b, 0.2)], [a, b], 0.4, tol);
    expect(d).toMatchObject({ pass: true, matched: 2 });
    expect(d.worstCornerPx).toBeCloseTo(0.2, 12);
    expect(cornerDistance(a, moved(a, 3))).toBe(3);
  });

  it('has no verdict without a tolerance for the precision', () => {
    const d = judgeDetections(want, want, 0.4, undefined);
    expect(d).toMatchObject({ pass: false, note: 'no tolerance for this precision' });
    expect(decodedText(d)).toBe('decoded: no tolerance for this precision');
  });
});

describe('the files', () => {
  it('reads the expected detections Python wrote', () => {
    const e = parseDetections({ tile: 576, threshold: 0.3, detections: [card(0.9, 0, 0)], note: 'more fields are fine' });
    expect(e).toEqual({ tile: 576, threshold: 0.3, detections: [card(0.9, 0, 0)] });
    expect(() => parseDetections({})).toThrow('no "detections" list');
    expect(() => parseDetections({ detections: [{ ...card(0.9, 0, 0), cls: 'dog' }] })).toThrow('detections[0].cls');
    expect(() => parseDetections({ detections: [{ ...card(0.9, 0, 0), quad: [[0, 0]] }] })).toThrow('detections[0].quad');
  });

  const manifest = (detections: unknown) => ({
    id: 'det',
    variants: [{ precision: 'fp16', file: 'det.fp16.onnx' }],
    input: { name: 'tiles', shape: ['batch', 3, 576, 576] },
    outputs: [{ name: 'pred_logits', shape: ['batch', 100, 2] }],
    batches: [1],
    check: { batch: 1, input: 'det.check.input.bin', expected: { pred_logits: 'det.check.pred_logits.bin' }, metric: 'maxabs', tolerance: { fp16: 12 }, detections },
  });

  it('reads check.detections from a manifest, and leaves it out when there is none', () => {
    const spec = { file: 'det.check.detections.json', threshold: 0.4, tolerance: { fp32: { corner_px: 1, score: 0.01 }, fp16: { corner_px: 8, score: 0.2 } }, note: 'why' };
    expect(parseManifest(manifest(spec)).check!.detections).toEqual({ file: spec.file, threshold: 0.4, tolerance: spec.tolerance });
    const { detections: _gone, ...plain } = manifest(undefined).check;
    expect(parseManifest({ ...manifest(undefined), check: plain }).check).not.toHaveProperty('detections');
  });

  it('refuses a check.detections it cannot use, naming the field', () => {
    const bad = (d: unknown) => () => parseManifest(manifest(d));
    expect(bad({ file: '../x.json', threshold: 0.4, tolerance: {} })).toThrow('det: check.detections.file: expected a plain file name');
    expect(bad({ file: 'x.json', threshold: 2, tolerance: {} })).toThrow('det: check.detections.threshold');
    expect(bad({ file: 'x.json', threshold: 0.4, tolerance: { fp16: { corner_px: -1, score: 0.1 } } })).toThrow('det: check.detections.tolerance.fp16');
  });
});

// ---- a whole row: a stand-in detector graph whose outputs are made-up head outputs, decoded and judged

/** Made-up head outputs for one tile, the same every time: a few likely cards among many unlikely queries. */
function head(): { pred_logits: Float32Array; pred_boxes: Float32Array; pred_keypoints: Float32Array } {
  let seed = 99;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  const logits = Float32Array.from({ length: 200 }, (_, i) => (i % 9 === 0 ? 1 + 6 * rnd() : -6 - rnd()));
  const boxes = Float32Array.from({ length: 400 }, (_, i) => (i % 4 < 2 ? 0.1 + 0.8 * rnd() : 0.08 + 0.05 * rnd()));
  const keypoints = Float32Array.from({ length: 6400 }, (_, i) => {
    const d = i % 8;
    return d < 2 ? 0.1 + 0.8 * rnd() : d === 4 || d === 6 ? 4 + rnd() : 3 * rnd() - 1;
  });
  return { pred_logits: logits, pred_boxes: boxes, pred_keypoints: keypoints };
}

describe('a row with the decoded check', () => {
  const h = head();
  const names = ['pred_logits', 'pred_boxes', 'pred_keypoints'] as const;
  // tiles [batch, 3] times a [3, k] matrix whose first row is the output: x = [1, 0, 0] gives it back exactly
  const model = encodeModel({
    inputs: [{ name: 'tiles', type: 'float32', shape: ['batch', 3] }],
    outputs: names.map((n) => ({ name: n, type: 'float32' as const, shape: ['batch', h[n].length] })),
    constants: names.map((n) => ({ name: `w_${n}`, type: 'float32' as const, dims: [3, h[n].length], values: [...h[n], ...new Array<number>(2 * h[n].length).fill(0)] })),
    nodes: names.map((n) => ({ op: 'MatMul', inputs: ['tiles', `w_${n}`], outputs: [n] })),
  });
  const cards = detector.decodeTile(h.pred_logits, h.pred_boxes, h.pred_keypoints);
  const bytes = (f: Float32Array) => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  const manifest = parseManifest({
    id: 'standin-cards',
    variants: [{ precision: 'fp32', file: 'standin-cards.onnx' }],
    input: { name: 'tiles', shape: ['batch', 3] },
    outputs: [
      { name: 'pred_logits', shape: ['batch', 100, 2] },
      { name: 'pred_boxes', shape: ['batch', 100, 4] },
      { name: 'pred_keypoints', shape: ['batch', 100, 8, 8] },
    ],
    batches: [1],
    check: {
      batch: 1,
      input: 'c.input.bin',
      expected: { pred_logits: 'c.pred_logits.bin', pred_boxes: 'c.pred_boxes.bin', pred_keypoints: 'c.pred_keypoints.bin' },
      metric: 'maxabs',
      tolerance: { fp32: 1e-6 },
      detections: { file: 'c.detections.json', threshold: 0.4, tolerance: { fp32: { corner_px: 1, score: 0.01 } } },
    },
  });
  const files = (expected: Detection[]): Record<string, Uint8Array> => ({
    'standin-cards.onnx': model,
    'c.input.bin': bytes(new Float32Array([1, 0, 0])),
    'c.pred_logits.bin': bytes(h.pred_logits),
    'c.pred_boxes.bin': bytes(h.pred_boxes),
    'c.pred_keypoints.bin': bytes(h.pred_keypoints),
    'c.detections.json': new TextEncoder().encode(JSON.stringify({ tile: 576, threshold: 0.3, detections: expected })),
  });
  const io = (f: Record<string, Uint8Array>): RowIO => ({
    read: async (name) => {
      const b = f[name];
      if (!b) throw new Error(`${name}: HTTP 404`);
      return b;
    },
    now: () => performance.now(),
    progress: () => {},
    isolated: false,
  });
  const job: RowJob = { model: 'standin-cards', title: 'x', precision: 'fp32', runtime: 'wasm', ep: 'wasm', manifest, file: 'standin-cards.onnx', modelsUrl: '', ortBase: '', threads: 1, quick: true };

  it('decodes the check tile\'s outputs and finds the expected cards', async () => {
    expect(cards.filter((c) => c.score >= 0.41).length).toBeGreaterThan(3);
    const row = await runRow(ort, job, io(files(cards)));
    expect(row.errors).toEqual([]);
    const check = row.check as CheckSummary;
    expect(check.pass).toBe(true);
    expect(check.decoded).toMatchObject({ pass: true, missing: [], extra: [], share: 1 });
    expect(check.decoded!.required).toBe(cards.filter((c) => c.score >= 0.41).length);
    expect(checkText(check)).toMatch(/^maxabs .* PASS; decoded \d+\/\d+ cards above 0\.41 \(100\.0%\), worst corner 0\.00 px, score 0\.000 PASS$/);
  });

  it('says which card was expected and not found', async () => {
    const best = [...cards].sort((a, b) => b.score - a.score)[0]!;
    const row = await runRow(ort, job, io(files([...cards, { ...best, quad: best.quad.map(([x, y]) => [x + 100, y] as [number, number]) }])));
    expect(row.check!.pass).toBe(true); // the raw check is its own verdict
    expect(row.check!.decoded).toMatchObject({ pass: false, missing: [`${best.cls} ${best.score.toFixed(3)}`] });
  });

  it('reports a missing cards file on the decoded check only', async () => {
    const f = files(cards);
    delete f['c.detections.json'];
    const row = await runRow(ort, job, io(f));
    expect(row.check!.pass).toBe(true);
    expect(row.check!.decoded).toMatchObject({ pass: false, note: 'c.detections.json: c.detections.json: HTTP 404' });
  });
});
