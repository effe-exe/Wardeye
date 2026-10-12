// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Chrome's engine document (chrome.offscreen, reason WORKERS): the engine's page (engine-document.ts), reached by the worker's
// runtime messages. With no frame for a while it closes itself; the worker makes it again when the next one comes.

import { answer, watchIdle } from './engine-document';
import type { EngineRequest } from './protocol';

chrome.runtime.onMessage.addListener((msg: EngineRequest, sender, sendResponse) => {
  if (msg?.target !== 'engine' || sender.tab) return; // the worker's requests only: a content script (it has a tab) is a page's
  answer(msg).then(sendResponse, (e) => sendResponse({ kind: 'unavailable', reason: e instanceof Error ? e.message : String(e) }));
  return true; // the answer comes later
});

watchIdle(() => window.close());
