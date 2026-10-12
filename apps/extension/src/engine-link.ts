// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Where the worker's requests to the engine go. In Chrome the engine lives in its own document, reached by runtime messages
// (worker.ts makes the document, offscreen.ts answers). In Firefox it lives in the background page with the worker's code
// (background-firefox.ts), which registers it here, and the requests are plain calls.

import type { EngineRequest } from './protocol';

let local: ((request: EngineRequest) => Promise<unknown>) | null = null;

/** The engine answers in this same page (Firefox). */
export function serveHere(answer: (request: EngineRequest) => Promise<unknown>): void {
  local = answer;
}

/** The engine in this page, if it is here. */
export const here = (): ((request: EngineRequest) => Promise<unknown>) | null => local;
