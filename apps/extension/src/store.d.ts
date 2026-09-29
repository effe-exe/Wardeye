// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The build's own constant, set by build.mjs (esbuild's `define`): true in the Chrome Web Store build (node build.mjs
// --store), false in the developer build. The store build has no companion mode: its worker never posts a frame to the
// live runner, its manifest has no access to 127.0.0.1, and the names, types and pictures of the cards come from Riot's
// public card gallery (feed.ts). Only worker.ts and offscreen.ts read it; every other file takes what it needs as an
// argument, so that the unit tests need no build.
//
// Read it as `if (__STORE__) { ... } else { ... }` or `__STORE__ ? a : b`: the build folds the constant and drops the branch that
// is not taken, and with it the code only that branch used (the companion client, in the store build). A constant copied
// into a variable is not folded.

declare const __STORE__: boolean;
