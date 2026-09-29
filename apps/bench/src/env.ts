// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// What this browser is: user agent, cores, isolation, WebGPU adapter. Read in the page, once per run.

import type { EnvInfo, GpuInfo } from './summary';
import type { Support } from './plan';

/** WASM threads to ask for: up to 4 and never more than the cores, but 1 without cross-origin isolation. */
export function wasmThreads(isolated: boolean, cores: number): number {
  return isolated ? Math.max(1, Math.min(4, cores || 1)) : 1;
}

const noGpu = (note: string): GpuInfo => ({
  available: false,
  note,
  vendor: '',
  architecture: '',
  device: '',
  description: '',
  shaderF16: false,
  features: [],
  maxBufferSize: null,
  maxStorageBufferBindingSize: null,
});

async function gpuInfo(): Promise<GpuInfo> {
  if (!('gpu' in navigator) || !navigator.gpu) return noGpu('navigator.gpu is missing');
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return noGpu('requestAdapter() gave no adapter');
    const info = adapter.info;
    return {
      available: true,
      note: '',
      vendor: info?.vendor ?? '',
      architecture: info?.architecture ?? '',
      device: info?.device ?? '',
      description: info?.description ?? '',
      shaderF16: adapter.features.has('shader-f16'),
      features: [...adapter.features].sort(),
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    };
  } catch (e) {
    return noGpu(`requestAdapter() threw: ${e instanceof Error ? e.message : String(e)}`);
  }
}

interface UserAgentData {
  platform?: string;
  getHighEntropyValues?(hints: string[]): Promise<Record<string, unknown>>;
}

async function platformInfo(): Promise<string> {
  const uad = (navigator as unknown as { userAgentData?: UserAgentData }).userAgentData;
  if (!uad) return '';
  try {
    const v = (await uad.getHighEntropyValues?.(['architecture', 'bitness', 'platformVersion', 'uaFullVersion'])) ?? {};
    const bits = v.bitness ? `${v.bitness}-bit` : '';
    return [uad.platform, v.platformVersion, v.architecture, bits, v.uaFullVersion ? `Chrome ${v.uaFullVersion}` : ''].filter(Boolean).join(' ');
  } catch {
    return uad.platform ?? '';
  }
}

async function batteryInfo(): Promise<string> {
  try {
    const b = await (navigator as unknown as { getBattery?: () => Promise<{ charging: boolean; level: number }> }).getBattery?.();
    if (b) return `battery ${b.charging ? 'charging' : 'on battery'} ${Math.round(b.level * 100)}%`;
  } catch {
    // no battery information
  }
  return 'battery unknown';
}

export async function collectEnv(ortVersion: string, benchVersion: string): Promise<EnvInfo> {
  const isolated = self.crossOriginIsolated;
  const cores = navigator.hardwareConcurrency || 1;
  return {
    date: new Date().toISOString(),
    benchVersion,
    ortVersion,
    userAgent: navigator.userAgent,
    platform: await platformInfo(),
    cores,
    crossOriginIsolated: isolated,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    threads: wasmThreads(isolated, cores),
    jspi: 'Suspending' in WebAssembly,
    battery: await batteryInfo(),
    gpu: await gpuInfo(),
  };
}

/** What the plan needs to know about the browser. */
export function supportOf(env: EnvInfo): Support {
  return { webgpu: env.gpu.available, webgpuNote: env.gpu.note, shaderF16: env.gpu.shaderF16, jspi: env.jspi };
}
