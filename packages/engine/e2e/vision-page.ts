// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The page vision.spec.ts runs in Chromium: the detector and the embedder as the extension runs them, on
// onnxruntime-web's WASM build (float32). Frames and crops are decoded the extension's way (createImageBitmap with
// no colour conversion and no premultiplying, an OffscreenCanvas, getImageData). The spec bundles this with esbuild
// and serves it, the runtime files, the models and the private fixtures from one made-up origin.

import * as ortWasm from 'onnxruntime-web/wasm';
import { detector, embedder, image, ort, type RgbImage } from '../src/index';

async function bytesOf(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** A JPEG or PNG as RGB, as the extension decodes a frame. */
async function decode(url: string): Promise<RgbImage> {
  const bitmap = await createImageBitmap(await (await fetch(url)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    return image.fromRgba(ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, bitmap.width, bitmap.height);
  } finally {
    bitmap.close();
  }
}

const runtime = (threads: number) => ort.chooseRuntime({ wasm: ortWasm as unknown as ort.OrtModule, wasmPaths: '/ort/', threads });

export interface FrameJob {
  url: string;
  window: number[];
  cardPx: number;
}

async function runDetector(model: string, frames: FrameJob[], threads: number) {
  const rt = await runtime(threads);
  const t0 = performance.now();
  const { detector: det, model: m } = await ort.openDetector(rt, { fp32: { name: model, bytes: () => bytesOf(`/models/${model}`) } });
  const loadMs = performance.now() - t0;
  const out = [];
  for (const f of frames) {
    const frame = await decode(f.url);
    const tiling = detector.cutTiles(frame, f.window, f.cardPx);
    const t1 = performance.now();
    const cards = await det.detect(frame, f.window, f.cardPx);
    out.push({
      rgbSha: await embedder.sha256Hex(frame.data),
      tileShas: await Promise.all(tiling.tiles.map((t) => embedder.sha256Hex(t.data))),
      inputShas: await Promise.all(tiling.tiles.map((t) => embedder.sha256Hex(new Uint8Array(detector.tileBatch([t]).buffer)))),
      cards,
      ms: performance.now() - t1,
    });
  }
  await m.session.release();
  return { loadMs, isolated: crossOriginIsolated, threads: ortWasm.env.wasm.numThreads, frames: out };
}

async function runEmbedder(model: string, crops: string[], threads: number) {
  const rt = await runtime(threads);
  const { encoder, model: m } = await ort.openEncoder(rt, { fp32: { name: model, bytes: () => bytesOf(`/models/${model}`) } });
  const images = await Promise.all(crops.map(decode));
  const letterboxShas = await Promise.all(images.map((im) => embedder.sha256Hex(embedder.letterbox(im).data)));
  await encoder.embed(images.slice(0, encoder.batch)); // warm-up
  const t0 = performance.now();
  const rows = await encoder.embed(images);
  const ms = performance.now() - t0;
  await m.session.release();
  return { name: encoder.name, letterboxShas, rows: Array.from(rows), ms, batches: Math.ceil(images.length / encoder.batch) };
}

Object.assign(globalThis, { rifteye: { runDetector, runEmbedder } });
document.title = 'ready';
