import { describe, expect, it } from 'vitest';
import { parseManifest } from '../src/model-manifest';
import { planRows, type ModelEntry, type Support } from '../src/plan';

const manifest = (id: string, precisions: string[] = ['fp32', 'fp16']) =>
  parseManifest({
    id,
    variants: precisions.map((p) => ({ precision: p, file: p === 'fp32' ? `${id}.onnx` : `${id}.${p}.onnx` })),
    input: { name: 'x', shape: ['batch', 3, 4, 4] },
    outputs: [{ name: 'y', shape: ['batch', 4] }],
    batches: [1],
  });
const entry = (id: string, precisions?: string[]): ModelEntry => ({ name: `${id}.bench.json`, manifest: manifest(id, precisions), error: null });
const everything: Support = { webgpu: true, webgpuNote: '', shaderF16: true, jspi: true };
const all = () => true;
const summary = (rows: ReturnType<typeof planRows>) => rows.map((r) => `${r.model} ${r.precision} ${r.runtime} ${r.action}${r.note ? ` (${r.note})` : ''}`);

describe('the plan', () => {
  it('has a row per model, variant and runtime, the GPU first and the CPU last', () => {
    const rows = planRows([entry('det'), entry('emb', ['fp32'])], all, everything);
    expect(summary(rows)).toEqual([
      'det fp32 webgpu run', 'det fp16 webgpu run', 'emb fp32 webgpu run',
      'det fp32 webgpu-jsep run', 'det fp16 webgpu-jsep run', 'emb fp32 webgpu-jsep run',
      'det fp32 wasm run', 'det fp16 wasm run', 'emb fp32 wasm run',
    ]);
  });

  it('skips a variant that is not in the folder, in one row, whatever the runtimes', () => {
    const rows = planRows([entry('det')], (f) => f !== 'det.fp16.onnx', everything);
    expect(summary(rows.filter((r) => r.precision === 'fp16'))).toEqual(['det fp16 all skip (not included)']);
    expect(rows.filter((r) => r.precision === 'fp32')).toHaveLength(3);
  });

  it('runs only WASM when there is no WebGPU adapter, and says why', () => {
    const rows = planRows([entry('det', ['fp32'])], all, { ...everything, webgpu: false, webgpuNote: 'requestAdapter() gave no adapter' });
    expect(summary(rows)).toEqual([
      'det fp32 webgpu skip (requestAdapter() gave no adapter)',
      'det fp32 webgpu-jsep skip (requestAdapter() gave no adapter)',
      'det fp32 wasm run',
    ]);
  });

  it('skips fp16 on WebGPU without shader-f16, but not on WASM', () => {
    const rows = planRows([entry('det')], all, { ...everything, shaderF16: false });
    expect(summary(rows)).toEqual([
      'det fp32 webgpu run',
      'det fp16 webgpu skip (fp16 not run: this WebGPU adapter has no shader-f16)',
      'det fp32 webgpu-jsep run',
      'det fp16 webgpu-jsep skip (fp16 not run: this WebGPU adapter has no shader-f16)',
      'det fp32 wasm run',
      'det fp16 wasm run',
    ]);
  });

  it('skips the native WebGPU build where WebAssembly has no JSPI, and keeps JSEP', () => {
    const rows = planRows([entry('det', ['fp32'])], all, { ...everything, jspi: false });
    expect(summary(rows)).toEqual(['det fp32 webgpu skip (this Chrome has no WebAssembly JSPI (Chrome 137 or later has it))', 'det fp32 webgpu-jsep run', 'det fp32 wasm run']);
  });

  it('reports a manifest that could not be read as an error row, and goes on with the others', () => {
    const rows = planRows([{ name: 'bad.bench.json', manifest: null, error: 'bad.bench.json could not be read: id: expected a non-empty string' }, entry('det', ['fp32'])], all, everything, ['wasm']);
    expect(rows[0]).toMatchObject({ model: 'bad', action: 'error', runtime: 'all', note: 'bad.bench.json could not be read: id: expected a non-empty string' });
    expect(summary(rows.slice(1))).toEqual(['det fp32 wasm run']);
  });

  it('can be limited to some runtimes', () => {
    expect(summary(planRows([entry('det', ['fp32'])], all, everything, ['wasm']))).toEqual(['det fp32 wasm run']);
  });
});
