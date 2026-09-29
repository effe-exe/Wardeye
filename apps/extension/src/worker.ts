// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The extension's background worker. The overlay hands it each frame over a port. In the private build, with a
// browser that can run the engine, the frame goes to the engine document (an offscreen page that reads it with
// ONNX Runtime Web) and the board it answers with is handed back; otherwise, as in the public build, the frame is
// posted to the live runner on this machine (companion mode), the only place a page script cannot reach itself.
// Nothing is sent anywhere else. The worker keeps nothing the engine needs: a tab's board lives in the engine
// document, so the worker can be put to sleep and woken at any time.

import * as companion from './companion';
import { Standalone, type Env } from './standalone';
import type { State } from './geometry';
import type { FromContent, ToContent } from './protocol';

const OFFSCREEN = 'offscreen.html';

let creating: Promise<void> | null = null;

/** The engine document, made when it is not there (only one can exist, and none survives the browser). */
async function ensureDocument(): Promise<void> {
  if (creating) return creating;
  const here = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (here.length > 0) return;
  creating = chrome.offscreen
    .createDocument({ url: OFFSCREEN, reasons: ['WORKERS'], justification: "Reads the Twitch player's frames with ONNX Runtime Web, in a worker, on this computer." })
    .finally(() => {
      creating = null;
    });
  return creating;
}

const env: Env = {
  now: () => Date.now(),
  read: async (path) => {
    try {
      const r = await fetch(chrome.runtime.getURL(path));
      return r.ok ? new Uint8Array(await r.arrayBuffer()) : null;
    } catch {
      return null; // not in this build
    }
  },
  ensureDocument,
  send: (request) => chrome.runtime.sendMessage(request),
};

const standalone = new Standalone(env);
let anonymous = 0; // a port that names no tab still gets a board of its own

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'rifteye') return;
  const tab = port.sender?.tab?.id ?? --anonymous;
  const send = (m: ToContent): void => {
    try {
      port.postMessage(m);
    } catch {
      // the tab went away while the frame was read
    }
  };
  void standalone.warm();
  port.onDisconnect.addListener(() => void standalone.forget(tab));
  port.onMessage.addListener(async (msg: FromContent) => {
    try {
      if (msg.kind === 'frame') {
        const served = await standalone.frame(tab, msg);
        if (served) {
          send({ kind: 'state', online: true, state: served.state });
          return;
        }
        const state = await companion.postFrame(msg);
        send({ kind: 'state', online: state !== null, state: (state ?? null) as State | null });
      } else if (msg.kind === 'art') {
        const jpeg = (await standalone.usable()) ? await standalone.art(msg.printing_id) : await companion.art(msg.printing_id);
        send({ kind: 'art', printing_id: msg.printing_id, jpeg });
      }
    } catch (e) {
      // whatever went wrong, the overlay is answered: it waits for an answer before it sends another frame
      console.warn(`Wardeye: ${e instanceof Error ? e.message : String(e)}`);
      send(msg.kind === 'frame' ? { kind: 'state', online: true, state: null } : { kind: 'art', printing_id: msg.printing_id, jpeg: null });
    }
  });
});
