// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The engine document (chrome.offscreen, reason WORKERS): the page the extension's worker keeps for the engine,
// since a service worker cannot hold WebGPU models for long. It answers the worker's requests (controller.ts) and
// runs the engine in a worker of its own (engine-client.ts). With no frame for a while it closes itself; the
// worker makes it again when the next one comes.
//
// In the Chrome Web Store build (`__STORE__`, store.d.ts) it also holds Riot's public card list (feed.ts): read once, in
// this document, which lives as long as the engine does; it names the gallery's rows for the engine worker, and the pictures
// the worker asks for are fetched from Riot's image server here (the last 64 kept in memory). Nothing is stored. These are
// the only requests the extension makes, and the two hosts are all its manifest gives access to (docs/PRIVACY.md).

import { parsePackage, type StandalonePackage } from './assets';
import { Controller, type ControllerEnv } from './controller';
import { FRAME_MS, WorkerEngine } from './engine-client';
import { Feed, Pictures } from './feed';
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

/** What this browser can do, asked of the same document the engine runs from, and what it found of the GPU, in words (the
 * badge's second line gives them when the engine runs on WASM). The default adapter is asked for first, then the high-performance
 * one: a laptop's default may be a chip that WebGPU cannot use while its discrete GPU can. */
async function probe(): Promise<Capabilities> {
  const jspi = 'Suspending' in WebAssembly;
  if (!('gpu' in navigator) || !navigator.gpu) return { webgpu: false, shaderF16: false, jspi, gpu: 'navigator.gpu is not there' };
  try {
    const adapter = (await navigator.gpu.requestAdapter()) ?? (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));
    if (!adapter) return { webgpu: false, shaderF16: false, jspi, gpu: 'navigator.gpu gave no adapter' };
    const info = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ');
    return { webgpu: true, shaderF16: adapter.features.has('shader-f16'), jspi, gpu: info || 'an adapter' };
  } catch (e) {
    return { webgpu: false, shaderF16: false, jspi, gpu: `requestAdapter failed: ${e instanceof Error ? e.message : String(e)}` };
  }
}

/** How long the engine's start, and a card's picture, wait for the first try at Riot's card list before going on without it (ms). */
const LIST_WAIT_MS = 15_000;
/** One request to Riot's gallery: the list's page, a picture (ms). */
const LIST_REQUEST_MS = 15_000;
const PICTURE_REQUEST_MS = 10_000;

/** A GET from Riot's public gallery, with nothing of the user's: no cookies, no referrer. */
const fromRiot = (url: string, ms: number): Promise<Response> => fetch(url, { credentials: 'omit', referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(ms) });

/** Riot's public card gallery, for the store build: the card list, the pictures, and a wait for the list's first try. It is a function
 * so that the developer build, which never calls it, has none of this in it. */
function riotGallery() {
  const feed = new Feed({
    getJson: async (url) => {
      const r = await fromRiot(url, LIST_REQUEST_MS);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    },
    later: (run, ms) => void setTimeout(run, ms),
  });
  /** The first try at the card list is over, or a moment has passed. */
  const listInHand = async (): Promise<void> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([feed.settled(), new Promise<void>((done) => void (timer = setTimeout(done, LIST_WAIT_MS)))]);
    clearTimeout(timer);
  };
  const pictures = new Pictures({
    imageUrl: async (printingId) => {
      await listInHand();
      return feed.imageUrl(printingId);
    },
    getBytes: async (url) => {
      const r = await fromRiot(url, PICTURE_REQUEST_MS);
      return r.ok ? new Uint8Array(await r.arrayBuffer()) : null;
    },
  });
  return { feed, pictures, listInHand };
}

const riot = __STORE__ ? riotGallery() : null;

const env: ControllerEnv = {
  now: () => Date.now(),
  readPackage,
  probe,
  // plain WASM reads a frame in seconds, and finds a layout in minutes: it is given the time
  spawn: (attempt) => (attempt.runtime === 'webgpu' ? new WorkerEngine('engine-webgpu.js') : new WorkerEngine('engine-wasm.js', 10 * FRAME_MS)),
  store: __STORE__,
  ...(riot
    ? {
        cards: async () => {
          await riot.listInHand();
          return riot.feed.rows();
        },
      }
    : {}),
};

const controller = new Controller(env);
// a list that comes after the engine started without it (its first try failed): the engine starts afresh with it
riot?.feed.onLoaded(() => controller.cardsArrived());

async function answer(req: EngineRequest): Promise<EngineReply | { kind: 'forgotten' }> {
  if (req.kind === 'hello') return controller.hello();
  if (req.kind === 'frame') return controller.frame(req);
  if (req.kind === 'art') return { kind: 'art', jpeg: riot ? await riot.pictures.get(req.printing_id) : null };
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
