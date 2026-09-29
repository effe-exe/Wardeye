import * as ort from 'onnxruntime-web';
import { describe, expect, it, vi } from 'vitest';
import { standInFiles } from '../e2e/standins';
import { parseManifest } from '../src/model-manifest';
import { collectLogs, runRow, shortSha, type Ort, type RowIO } from '../src/row-runner';
import { encodeModel } from '../src/tiny-onnx';
import type { RowJob } from '../src/types';

// The row runner with the real onnxruntime-web (its Node build, WASM, one thread) on the stand-in models.
const files = standInFiles();

const fileIo = (extra: Record<string, Uint8Array> = {}, missing: string[] = []): RowIO & { lines: string[] } => {
  const lines: string[] = [];
  return {
    lines,
    read: async (name) => {
      if (missing.includes(name)) throw new Error(`${name}: HTTP 404`);
      const f = extra[name] ?? files[name];
      if (f === undefined) throw new Error(`${name}: HTTP 404`);
      return typeof f === 'string' ? new TextEncoder().encode(f) : f;
    },
    now: () => performance.now(),
    progress: (t) => lines.push(t),
    isolated: false,
  };
};

const jobOf = (id: string, precision: string, file: string, manifest = parseManifest(JSON.parse(files[`${id}.bench.json`] as string))): RowJob => ({
  model: id,
  title: manifest.title,
  precision,
  runtime: 'wasm',
  ep: 'wasm',
  manifest,
  file,
  modelsUrl: '',
  ortBase: '',
  threads: 1,
  quick: false,
});

describe('the model file\'s sha256', () => {
  it('is the first 12 hex digits of the digest', async () => {
    // sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    expect(await shortSha(new TextEncoder().encode('abc'))).toBe('ba7816bf8f01');
  });
});

describe('a row', () => {
  it('times every batch, checks two outputs by maxabs, and passes', async () => {
    const io = fileIo();
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx'), io);
    expect(row.errors).toEqual([]);
    expect(row.status).toBe('ok');
    expect(row.runtime).toBe('wasm');
    expect(row.ortVersion).toMatch(/^1\./);
    expect(row.threads).toBe(1);
    expect(row.initMs).toBeGreaterThan(0);
    expect(row.loadMs).toBeGreaterThan(0);
    expect(row.batches.map((b) => b.batch)).toEqual([1, 2, 4]);
    for (const b of row.batches) {
      expect(b.runs).toBe(20);
      expect(b.medianMs).toBeGreaterThan(0);
      expect(b.p90Ms!).toBeGreaterThanOrEqual(b.medianMs!);
      expect(b.perItemMs).toBeCloseTo(b.medianMs! / b.batch, 12);
      expect(b.firstMs).toBeGreaterThan(0);
    }
    expect(row.firstMs).toBe(row.batches[0]!.firstMs);
    expect(row.modelBytes).toBe((files['standin-detector.onnx'] as Uint8Array).byteLength);
    expect(row.modelSha).toMatch(/^[0-9a-f]{12}$/);
    expect(row.check).toMatchObject({ metric: 'maxabs', batch: 2, pass: true, error: null, value: 0 });
    expect(row.check!.outputs.map((o) => o.name)).toEqual(['y', 'z']);
    expect(io.lines.some((l) => l.includes('batch 4, run 20/20'))).toBe(true);
    expect(io.lines[0]).toBe('standin-detector fp32 on wasm: starting the runtime');
  });

  it('in quick mode warms up once and times one run per batch, and still checks', async () => {
    const quick = { ...jobOf('standin-detector', 'fp32', 'standin-detector.onnx'), quick: true };
    const row = await runRow(ort, quick, fileIo());
    expect(row.errors).toEqual([]);
    expect(row.batches.map((b) => b.runs)).toEqual([1, 1, 1]);
    expect(row.check?.pass).toBe(true);
  });

  it('checks the embedder by cosine, in fp32 and in fp16', async () => {
    for (const [precision, file] of [['fp32', 'standin-embedder.onnx'], ['fp16', 'standin-embedder.fp16.onnx']] as const) {
      const row = await runRow(ort, jobOf('standin-embedder', precision, file), fileIo());
      expect(row.errors, precision).toEqual([]);
      expect(row.check, precision).toMatchObject({ metric: 'cosine', batch: 4, pass: true });
      expect(row.check!.value).toBeGreaterThan(0.99999);
    }
  });

  it('fails the check of a model that computes something else, and that is not an error', async () => {
    const row = await runRow(ort, jobOf('standin-wrong', 'fp32', 'standin-wrong.onnx'), fileIo());
    expect(row.status).toBe('ok');
    expect(row.errors).toEqual([]);
    expect(row.check?.pass).toBe(false);
    expect(row.check?.value).toBeCloseTo(-1, 6);
  });

  it('reports a model file that is not a model, and does not throw', async () => {
    const row = await runRow(ort, jobOf('standin-broken', 'fp32', 'standin-broken.onnx'), fileIo());
    expect(row.status).toBe('error');
    expect(row.errors).toHaveLength(1);
    expect(row.errors[0]).toMatch(/^load: .*(protobuf|parsing)/);
    expect(row.batches).toEqual([]);
    expect(row.check).toMatchObject({ pass: false, error: 'not run: the model did not load' });
    expect(row.initMs).toBeGreaterThan(0); // the runtime itself started
  });

  it('reports a model file that is missing', async () => {
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx'), fileIo({}, ['standin-detector.onnx']));
    expect(row.errors).toEqual(['load: standin-detector.onnx: HTTP 404']);
  });

  it('still times the batches when the check files cannot be read, and says the check was not made', async () => {
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx'), fileIo({}, ['standin-detector.check.z.bin']));
    expect(row.batches.map((b) => b.runs)).toEqual([20, 20, 20]);
    expect(row.check?.error).toBe('standin-detector.check.z.bin: HTTP 404');
    expect(row.check?.pass).toBe(false);
    expect(row.errors).toEqual(['check: standin-detector.check.z.bin: HTTP 404']);
    expect(row.status).toBe('error');
  });

  it('says so when a check file has the wrong size', async () => {
    const short = new Uint8Array(files['standin-detector.check.input.bin'] as Uint8Array).slice(0, 400);
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx'), fileIo({ 'standin-detector.check.input.bin': short }));
    expect(row.check?.error).toContain('standin-detector.check.input.bin holds 100 values, the shape 2x3x8x8 needs 384');
    expect(row.batches[0]!.runs).toBe(20); // made-up pixels stand in for the check input
  });

  it('puts a failing batch into its own entry and goes on', async () => {
    const m = parseManifest({ ...JSON.parse(files['standin-embedder.bench.json'] as string), input: { name: 'wrong_name', shape: ['batch', 3, 8, 8] } });
    const row = await runRow(ort, jobOf('standin-embedder', 'fp32', 'standin-embedder.onnx', m), fileIo());
    expect(row.batches).toHaveLength(2);
    expect(row.batches.every((b) => b.error !== null)).toBe(true);
    expect(row.errors.some((e) => e.startsWith('batch 1:'))).toBe(true);
    expect(row.errors.some((e) => e.startsWith('batch 4:'))).toBe(true);
    expect(row.errors.some((e) => e.startsWith('check:'))).toBe(true);
  });

  it('times a fixed-batch model at its own batch, whatever the manifest lists', async () => {
    const fixed = encodeModel({
      inputs: [{ name: 'x', type: 'float32', shape: [2, 3, 8, 8] }],
      outputs: [{ name: 'y', type: 'float32', shape: [2, 3, 8, 8] }],
      constants: [{ name: 'two', type: 'float32', dims: [], values: [2] }],
      nodes: [{ op: 'Mul', inputs: ['x', 'two'], outputs: ['y'] }],
    });
    const m = parseManifest({
      ...JSON.parse(files['standin-detector.bench.json'] as string),
      input: { name: 'x', shape: [2, 3, 8, 8] },
      outputs: [{ name: 'y', shape: [2, 3, 8, 8] }],
      batches: [1, 2],
      check: { ...JSON.parse(files['standin-detector.bench.json'] as string).check, expected: { y: 'standin-detector.check.y.bin' } },
    });
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'fixed.onnx', m), fileIo({ 'fixed.onnx': fixed }));
    expect(row.errors).toEqual([]);
    expect(row.batches.map((b) => b.batch)).toEqual([2]);
    expect(row.check?.pass).toBe(true);
  });

  it('checks an output whose axis the manifest leaves open, by maxabs, and refuses to by cosine', async () => {
    const base = JSON.parse(files['standin-detector.bench.json'] as string);
    const open = parseManifest({ ...base, outputs: [{ name: 'y', shape: ['batch', 3, 'rows', 8] }, base.outputs[1]] });
    const row = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx', open), fileIo());
    expect(row.errors).toEqual([]);
    expect(row.check?.pass).toBe(true);

    const cosine = parseManifest({ ...base, check: { ...base.check, metric: 'cosine', tolerance: { fp32: 0.99 } }, outputs: [{ name: 'y', shape: ['batch', 3, 8, 'cols'] }, base.outputs[1]] });
    const bad = await runRow(ort, jobOf('standin-detector', 'fp32', 'standin-detector.onnx', cosine), fileIo());
    expect(bad.check?.error).toBe('output "y": cosine needs a fixed last dimension, the manifest gives "cols"');
  });

  it('releases every session it created', async () => {
    let created = 0;
    let released = 0;
    const counting = {
      ...ort,
      InferenceSession: {
        create: async (...args: Parameters<typeof ort.InferenceSession.create>) => {
          const s = await (ort.InferenceSession.create as (...a: unknown[]) => Promise<ort.InferenceSession>)(...args);
          created++;
          const release = s.release.bind(s);
          s.release = async () => {
            released++;
            return release();
          };
          return s;
        },
      },
    } as unknown as Ort;
    await runRow(counting, jobOf('standin-embedder', 'fp32', 'standin-embedder.onnx'), fileIo());
    expect(created).toBe(2); // the startup model and the real one
    expect(released).toBe(2);
  });

  it('when the runtime cannot start, says so and stops', async () => {
    const dead = { ...ort, InferenceSession: { create: async () => Promise.reject(new Error('WebGPU is not supported in current environment')) } } as unknown as Ort;
    const row = await runRow(dead, jobOf('standin-embedder', 'fp32', 'standin-embedder.onnx'), fileIo());
    expect(row.status).toBe('error');
    expect(row.errors).toEqual(['runtime start: WebGPU is not supported in current environment']);
    expect(row.batches).toEqual([]);
  });
});

describe('the runtime\'s log', () => {
  it('keeps warnings, errors and onnxruntime\'s own [W:] lines, not the noise, and gives the console back', () => {
    const originals = (['log', 'info', 'warn', 'error'] as const).map((m) => vi.spyOn(console, m).mockImplementation(() => undefined));
    try {
      const warn = console.warn;
      const sink: string[] = [];
      const stop = collectLogs(sink, 4);
      console.warn('Some nodes were not assigned', 'to the preferred   execution providers');
      console.log('a plain log line');
      console.log('2026 [W:onnxruntime:, graph.cc:3] Removing initializer \'w\'. It is not used by any node');
      console.error('2026 [E:onnxruntime:x] it failed');
      console.log('2026 [W:onnxruntime:y] on the other hand');
      console.log('\u001b[0;93m2026-09-28 23:57:51.900645 [W:onnxruntime:, session_state.cc:1397 X] coloured and stamped\u001b[m');
      console.log('2026 [W:onnxruntime:z] Rerunning with verbose output on a non-minimal build will show node assignments.');
      console.warn('w1');
      console.warn('w1'); // (the same line twice, once it is over the limit, counts twice)
      console.warn('w2');
      console.warn('w3');
      stop();
      expect(sink).toEqual([
        'Some nodes were not assigned to the preferred execution providers',
        '2026 [E:onnxruntime:x] it failed',
        '2026 [W:onnxruntime:y] on the other hand',
        '[W:onnxruntime:, session_state.cc:1397 X] coloured and stamped', // no colour codes, no time stamp
        '(4 more)', // w1, w1, w2 and w3 came after the limit of 4 different lines
      ]);
      expect(console.warn).toBe(warn);
    } finally {
      originals.forEach((o) => o.mockRestore());
    }
  });

  it('ends up in the row', async () => {
    const chatty = {
      ...ort,
      InferenceSession: {
        create: async (...args: Parameters<typeof ort.InferenceSession.create>) => {
          console.warn('Some nodes were not assigned to the preferred execution providers.');
          return (ort.InferenceSession.create as (...a: unknown[]) => Promise<ort.InferenceSession>)(...args);
        },
      },
    } as unknown as Ort;
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const row = await runRow(chatty, jobOf('standin-embedder', 'fp32', 'standin-embedder.onnx'), fileIo());
      expect(row.warnings).toEqual(['Some nodes were not assigned to the preferred execution providers.']);
    } finally {
      spy.mockRestore();
    }
  });
});
