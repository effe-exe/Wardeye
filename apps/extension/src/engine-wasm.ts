// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The engine worker on the plain WASM (CPU) build of onnxruntime-web, with no GPU code in it. Its runtime files are
// ort/ort-wasm-simd-threaded.*.

import * as ort from 'onnxruntime-web/wasm';
import { serve } from './engine-worker';

serve(ort);
