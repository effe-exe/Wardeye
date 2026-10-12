// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// `__FIREFOX__` is true in the Firefox build (build.mjs --firefox) and false in Chrome's, as `__STORE__` is for the store build
// (store.d.ts): read it as a constant, and the build drops the branch that is not taken.

declare const __FIREFOX__: boolean;
