// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The worker for the native WebGPU execution provider (onnxruntime-web/jspi: WebGPU compiled into the wasm,
// suspended with WebAssembly JSPI). Its runtime files are ort/ort-wasm-simd-threaded.jspi.*.

import * as ort from 'onnxruntime-web/jspi';
import { serve } from './row-worker';

serve(ort);
