// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The engine document (chrome.offscreen, reason WORKERS): the page the extension's worker keeps for the engine,
// since a service worker cannot hold WebGPU models for long. It answers the worker's requests (controller.ts) and
// runs the engine in a worker of its own (engine-client.ts). With no frame for a while it closes itself; the
// worker makes it again when the next one comes.

import { parsePackage, type StandalonePackage } from './assets';
import { Controller, type ControllerEnv } from './controller';
import { FRAME_MS, WorkerEngine } from './engine-client';
import type { Capabilities } from './mode';
import type { EngineReply, EngineRequest } from './protocol';
import { idleLongEnough } from './router';

const IDLE_MS = 5 * 60_000;
const CHECK_MS = 30_000;

async function readPackage(): Promise<StandalonePackage | null> {
  try {
    const r = await fetch(new URL('standalone.json', location.href));
    return r.ok ? parsePackage(await r.json()) : null;
  } catch {
    return null;
  }
}

/** What this browser can do, asked of the same document the engine runs from. */
async function probe(): Promise<Capabilities> {
  const jspi = 'Suspending' in WebAssembly;
  try {
    const adapter = 'gpu' in navigator && navigator.gpu ? await navigator.gpu.requestAdapter() : null;
    return { webgpu: adapter !== null, shaderF16: adapter?.features.has('shader-f16') ?? false, jspi };
  } catch {
    return { webgpu: false, shaderF16: false, jspi };
  }
}

const env: ControllerEnv = {
  now: () => Date.now(),
  readPackage,
  probe,
  // plain WASM reads a frame in seconds, and finds a layout in minutes: it is given the time
  spawn: (attempt) => (attempt.runtime === 'webgpu' ? new WorkerEngine('engine-webgpu.js') : new WorkerEngine('engine-wasm.js', 10 * FRAME_MS)),
};

const controller = new Controller(env);

async function answer(req: EngineRequest): Promise<EngineReply | { kind: 'forgotten' }> {
  if (req.kind === 'hello') return controller.hello();
  if (req.kind === 'frame') return controller.frame(req);
  controller.forget(req.tab);
  return { kind: 'forgotten' };
}

chrome.runtime.onMessage.addListener((msg: EngineRequest, sender, sendResponse) => {
  if (msg?.target !== 'engine' || sender.tab) return; // the worker's requests only: a content script (it has a tab) is a page's
  answer(msg).then(sendResponse, (e) => sendResponse({ kind: 'unavailable', reason: e instanceof Error ? e.message : String(e) }));
  return true; // the answer comes later
});

setInterval(() => {
  if (idleLongEnough(Date.now(), controller.lastFrame, IDLE_MS)) {
    controller.dispose();
    window.close();
  }
}, CHECK_MS);
