import { describe, expect, it } from 'vitest';
import type { StandalonePackage } from '../src/assets';
import { plan, type Capabilities } from '../src/mode';

/** The private build's models: the detector in float32, the embedder in float16. */
const pkg = (over: Partial<StandalonePackage> = {}): StandalonePackage => ({
  format: 1,
  runtime: 'auto',
  detector: { id: 'detector-v0', fp32: 'models/d32.onnx' },
  embedder: { id: 'embedder-v1', fp16: 'models/e16.onnx' },
  data: 'data/',
  ...over,
});
const everything = (): Partial<StandalonePackage> => ({
  detector: { id: 'detector-v0', fp16: 'd16', fp32: 'd32' },
  embedder: { id: 'embedder-v1', fp16: 'e16', fp32: 'e32' },
});
const gpu = (over: Partial<Capabilities> = {}): Capabilities => ({ webgpu: true, shaderF16: true, jspi: true, ...over });

describe('how the engine runs', () => {
  it('is WebGPU with the detector in float32 and the embedder in float16, and WASM after it', () => {
    expect(plan(pkg(), gpu()).attempts).toEqual([
      { runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' },
      { runtime: 'wasm', detector: 'fp32', embedder: 'fp16' }, // for a browser whose native runtime cannot start
    ]);
  });

  it('never puts the detector on WebGPU in float16, whatever the package holds', () => {
    expect(plan(pkg(everything()), gpu()).attempts[0]).toEqual({ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' });
    const half = pkg({ detector: { id: 'detector-v0', fp16: 'd16' } });
    const p = plan(half, gpu());
    expect(p.attempts).toEqual([{ runtime: 'wasm', detector: 'fp16', embedder: 'fp16' }]); // WASM computes it in float32 inside
    expect(p.reason).toContain('no float32 detector');
  });

  it('runs the embedder in float32 on a GPU without shader-f16, when the package has that file', () => {
    expect(plan(pkg(everything()), gpu({ shaderF16: false })).attempts).toEqual([
      { runtime: 'webgpu', detector: 'fp32', embedder: 'fp32' },
      { runtime: 'wasm', detector: 'fp32', embedder: 'fp32' },
    ]);
  });

  it('cannot use a GPU without shader-f16 with the float16 embedder alone: WASM does', () => {
    const p = plan(pkg(), gpu({ shaderF16: false }));
    expect(p.attempts).toEqual([{ runtime: 'wasm', detector: 'fp32', embedder: 'fp16' }]);
    expect(p.reason).toContain('no shader-f16');
  });

  it('goes to the live runner in a browser with no WebGPU adapter', () => {
    const p = plan(pkg(), gpu({ webgpu: false, shaderF16: false }));
    expect(p.attempts).toEqual([]);
    expect(p.reason).toContain('no WebGPU adapter');
  });

  it('runs plain WASM when the browser has WebGPU but not JSPI', () => {
    const p = plan(pkg(everything()), gpu({ jspi: false }));
    expect(p.attempts).toEqual([{ runtime: 'wasm', detector: 'fp32', embedder: 'fp32' }]);
    expect(p.reason).toContain('JSPI');
  });

  it('is what standalone.json says when it says', () => {
    expect(plan(pkg({ runtime: 'companion' }), gpu()).attempts).toEqual([]);
    expect(plan(pkg({ runtime: 'wasm', ...everything() }), gpu({ webgpu: false, jspi: false })).attempts).toEqual([{ runtime: 'wasm', detector: 'fp32', embedder: 'fp32' }]);
    expect(plan(pkg({ runtime: 'webgpu' }), gpu()).attempts).toEqual([{ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' }]);
    expect(plan(pkg({ runtime: 'webgpu' }), gpu({ webgpu: false })).attempts).toEqual([]);
  });

  it('needs both models', () => {
    expect(plan(pkg({ detector: { id: 'd' } as never }), gpu({ webgpu: false })).attempts).toEqual([]);
    const noEmbedder = pkg({ runtime: 'wasm', embedder: { id: 'e' } as never });
    expect(plan(noEmbedder, gpu()).attempts).toEqual([]);
    expect(plan(noEmbedder, gpu()).reason).toContain('no complete set of models');
  });
});
