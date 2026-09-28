// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The extension's background worker: the only part that talks to the live runner on this machine
// (python -m rifteye_ml.live --source browser). A page script may not reach 127.0.0.1 itself, so the
// overlay hands it each frame over a port; it posts the frame to the runner and hands back the newest
// state, and fetches card pictures for the hover card. Nothing is sent anywhere else.

const PORTS = Array.from({ length: 10 }, (_, k) => 8765 + k); // the runner takes the next free one of these
let runner: string | null = null;
let failures = 0; // answers in a row that did not come: one slow answer is not a runner gone

async function findRunner(): Promise<string | null> {
  for (const p of PORTS) {
    const base = `http://127.0.0.1:${p}`;
    try {
      const r = await fetch(`${base}/hello`, { signal: AbortSignal.timeout(1500) });
      const hello = (await r.json()) as { rifteye?: string; frames?: boolean };
      if (hello.rifteye === 'live' && hello.frames) return base;
    } catch {
      // not there, or another program on that port
    }
  }
  return null;
}

function bytesOf(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64Of(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** The runner's newest state; undefined when this answer was slow or lost (keep the board); null when there is no
 * runner at all. */
async function postFrame(msg: { t: number; video: string; jpeg: string }): Promise<unknown | null | undefined> {
  runner ??= await findRunner();
  if (!runner) return null;
  try {
    const r = await fetch(`${runner}/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg', 'X-Media-Time': String(msg.t), 'X-Video': msg.video },
      body: bytesOf(msg.jpeg),
      signal: AbortSignal.timeout(10_000),
    });
    if (r.ok) {
      failures = 0;
      return await r.json();
    }
  } catch {
    // slow, or the runner stopped
  }
  if (++failures >= 3) {
    runner = null; // look for it again, maybe on another port
    failures = 0;
  }
  return undefined;
}

async function art(printingId: string): Promise<string | null> {
  if (!runner) return null;
  try {
    const r = await fetch(`${runner}/art/${encodeURIComponent(printingId)}.jpg`, { signal: AbortSignal.timeout(5000) });
    return r.ok ? b64Of(await r.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'rifteye') return;
  port.onMessage.addListener(async (msg: { kind: string; t: number; video: string; jpeg: string; printing_id: string }) => {
    if (msg.kind === 'frame') {
      const state = await postFrame(msg);
      port.postMessage({ kind: 'state', online: state !== null, state: state ?? null });
    } else if (msg.kind === 'art') {
      port.postMessage({ kind: 'art', printing_id: msg.printing_id, jpeg: await art(msg.printing_id) });
    }
  });
});
