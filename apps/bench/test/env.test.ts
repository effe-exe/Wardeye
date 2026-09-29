import { describe, expect, it } from 'vitest';
import { supportOf, wasmThreads } from '../src/env';
import type { EnvInfo } from '../src/summary';

describe('wasm threads', () => {
  it('are up to 4, and never more than the cores', () => {
    expect(wasmThreads(true, 10)).toBe(4);
    expect(wasmThreads(true, 4)).toBe(4);
    expect(wasmThreads(true, 2)).toBe(2);
    expect(wasmThreads(true, 1)).toBe(1);
  });

  it('are one without cross-origin isolation, and at least one whatever the browser says of its cores', () => {
    expect(wasmThreads(false, 10)).toBe(1);
    expect(wasmThreads(true, 0)).toBe(1);
    expect(wasmThreads(true, NaN)).toBe(1);
  });
});

describe('what the plan needs to know', () => {
  it('is read from the environment', () => {
    const env = {
      jspi: true,
      gpu: { available: true, note: '', shaderF16: false },
    } as EnvInfo;
    expect(supportOf(env)).toEqual({ webgpu: true, webgpuNote: '', shaderF16: false, jspi: true });
    expect(supportOf({ jspi: false, gpu: { available: false, note: 'navigator.gpu is missing', shaderF16: false } } as EnvInfo)).toEqual({
      webgpu: false,
      webgpuNote: 'navigator.gpu is missing',
      shaderF16: false,
      jspi: false,
    });
  });
});
