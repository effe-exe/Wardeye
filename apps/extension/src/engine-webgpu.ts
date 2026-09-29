// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The engine worker on the native WebGPU build of onnxruntime-web (WebGPU compiled into the wasm, suspended with
// WebAssembly JSPI). Its runtime files are ort/ort-wasm-simd-threaded.jspi.*.

import * as ort from 'onnxruntime-web/jspi';
import { serve } from './engine-worker';

serve(ort);
