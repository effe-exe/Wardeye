// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The worker for the JSEP WebGPU execution provider (the default build of onnxruntime-web 1.30: WebGPU kernels
// written in JavaScript; the older path). Its runtime files are ort/ort-wasm-simd-threaded.jsep.*.

import * as ort from 'onnxruntime-web';
import { serve } from './row-worker';

serve(ort);
