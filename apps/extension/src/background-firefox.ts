// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Firefox's background page (build.mjs --firefox bundles it as worker.js): Chrome's worker (worker.ts) and the engine's page
// (engine-document.ts) in one, since Firefox has no offscreen documents and its background page is a page. The engine is let
// go after five minutes without a frame, as in Chrome; the page itself is Firefox's to unload.

import { answer, watchIdle } from './engine-document';
import { serveHere } from './engine-link';
import './worker';

serveHere(answer);
watchIdle(() => {});
