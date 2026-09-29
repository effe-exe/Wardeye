// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The extension's background worker. The overlay hands it each frame over a port. In the private build, with a
// browser that can run the engine, the frame goes to the engine document (an offscreen page that reads it with
// ONNX Runtime Web) and the board it answers with is handed back; otherwise, as in the public build, the frame is
// posted to the live runner on this machine (companion mode), the only place a page script cannot reach itself.
// The Chrome Web Store build (`__STORE__`, store.d.ts) has no companion mode, and no code for it: when the engine
// cannot run, the overlay is told so, and its badge says why; a card's picture is asked of the engine document, which
// holds Riot's card list. Nothing is sent anywhere else. The worker keeps nothing the engine needs: a tab's board lives
// in the engine document, so the worker can be put to sleep and woken at any time.

import * as companion from './companion';
import { RETRY_AFTER_MS } from './mode';
import { Standalone, cannotReadState, type Env } from './standalone';
import type { State } from './geometry';
import type { FromContent, FromPanel, ToContent, ToPanel } from './protocol';

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

const standalone = new Standalone(env, RETRY_AFTER_MS, __STORE__);

// The toolbar button: a click turns Wardeye off or on in that tab (content.ts does it), and while it is off there the button says
// OFF. Alt+R and the badge's own button do the same, and tell the worker, so the toolbar button always says how the tab is.
const ON_TITLE = 'Wardeye is on. Click to turn it off (Alt+R)';
const OFF_TITLE = 'Wardeye is off. Click to turn it on (Alt+R)';
void chrome.action.setBadgeBackgroundColor({ color: '#71717A' }).catch(() => {});
chrome.action.onClicked.addListener((tab) => {
  if (tab.id === undefined) return;
  chrome.tabs.sendMessage(tab.id, { kind: 'toggle' }).catch(() => {}); // not a twitch.tv page: there is nothing to turn off
});
chrome.runtime.onMessage.addListener((msg: { kind?: unknown; on?: unknown }, sender) => {
  const tabId = sender.tab?.id;
  if (tabId === undefined) return;
  if (msg?.kind === 'switched' && typeof msg.on === 'boolean') {
    void chrome.action.setBadgeText({ tabId, text: msg.on ? '' : 'OFF' }).catch(() => {});
    void chrome.action.setTitle({ tabId, title: msg.on ? ON_TITLE : OFF_TITLE }).catch(() => {});
  } else if (msg?.kind === 'open-panel') {
    // the badge's plays button, clicked: the browser's side panel opens on the plays panel (a click in a content script may)
    void chrome.sidePanel?.open({ tabId }).catch(() => {});
  }
});

/** A card's hover picture, base64, for the overlay and the plays panel alike; null when there is none. */
async function art(printingId: string): Promise<string | null> {
  if (__STORE__) return standalone.art(printingId);
  return (await standalone.usable()) ? standalone.art(printingId) : companion.art(printingId);
}

let anonymous = 0; // a port that names no tab still gets a board of its own

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'panel') {
    // the plays panel: only pictures, and no board of its own
    port.onMessage.addListener(async (msg: FromPanel) => {
      if (msg?.kind !== 'art' || typeof msg.printing_id !== 'string') return;
      const jpeg = await art(msg.printing_id).catch(() => null);
      try {
        port.postMessage({ kind: 'art', printing_id: msg.printing_id, jpeg } satisfies ToPanel);
      } catch {
        // the panel was closed
      }
    });
    return;
  }
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
          send({ kind: 'state', online: true, state: served.state, events: served.events });
        } else if (__STORE__) {
          send({ kind: 'state', online: true, state: cannotReadState(msg.t, (await standalone.package()) === null) });
        } else {
          const state = await companion.postFrame(msg);
          send({ kind: 'state', online: state !== null, state: (state ?? null) as State | null });
        }
      } else if (msg.kind === 'art') {
        send({ kind: 'art', printing_id: msg.printing_id, jpeg: await art(msg.printing_id) });
      }
    } catch (e) {
      // whatever went wrong, the overlay is answered: it waits for an answer before it sends another frame
      console.warn(`Wardeye: ${e instanceof Error ? e.message : String(e)}`);
      send(msg.kind === 'frame' ? { kind: 'state', online: true, state: null } : { kind: 'art', printing_id: msg.printing_id, jpeg: null });
    }
  });
});
