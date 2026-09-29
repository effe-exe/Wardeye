// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The page e2e/recognizer.spec.ts runs in Chromium. The LA final's JPEG frames are decoded as the extension decodes
// them (createImageBitmap with no colour conversion, a 2D canvas, getImageData), each checked against the RGB
// Pillow decodes, and the recognizer is replayed against its Python reference (test/recognizer-replay.ts). The spec
// serves the frames (/frames/) and the fixtures (/fx/) from the private folder.

import { fromRgba } from '../src/image';
import type { RgbImage } from '../src/types';
import { parseFixture, picture, replay, rgbSha256, summary, type ReplayOptions, type ReplayReport } from '../test/recognizer-replay';

export interface PageResult {
  report: ReplayReport;
  text: string;
  /** Per frame: JPEG to RGB, ms. */
  decodeMs: number[];
  total: number;
}

async function get(path: string): Promise<Response> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r;
}

const text = async (path: string): Promise<string> => (await get(path)).text();
const bytes = async (path: string): Promise<ArrayBuffer> => (await get(path)).arrayBuffer();
const gunzipped = async (path: string): Promise<string> =>
  new Response((await get(path)).body!.pipeThrough(new DecompressionStream('gzip'))).text();

/** A frame as the extension reads one: no colour management, no premultiplied alpha, the canvas's RGBA to RGB. */
async function decode(blob: Blob): Promise<RgbImage> {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  bmp.close();
  return fromRgba(rgba, canvas.width, canvas.height);
}

/** Replays the fixture under `base` (/fx for the LA final, /fx/cut for the cut scenario). */
async function runReplay(base: string, opts: ReplayOptions): Promise<PageResult> {
  const meta = await text(`${base}/meta.json`);
  const levels = (JSON.parse(meta) as { levels: number[] }).levels;
  const fx = parseFixture({
    meta,
    rows: await text(`${base}/rows.json`),
    steps: await text(`${base}/steps.jsonl`),
    embeds: await text(`${base}/embeds.json`),
    embedsBin: await bytes(`${base}/embeds.bin`),
    levels: new Map(await Promise.all(levels.map(async (s): Promise<[number, ArrayBuffer]> => [s, await bytes(`${base}/levels/${s}.bin`)]))),
    internals: await gunzipped(`${base}/internals.jsonl.gz`),
  });
  const listing = JSON.parse(await text('/frames/frames.json')) as { frames: { file: string; rgb_sha256: string }[] };
  const pillow = new Map(listing.frames.map((f) => [f.file, f.rgb_sha256]));
  const decodeMs: number[] = [];
  const load = async (file: string): Promise<RgbImage> => {
    const blob = await (await get(`/frames/${file}`)).blob();
    const tic = performance.now();
    const im = await decode(blob);
    decodeMs.push(performance.now() - tic);
    const h = await rgbSha256(im);
    if (h !== pillow.get(file)) throw new Error(`${file}: Chromium decodes it to other RGB than Pillow (sha256 ${h})`);
    return im;
  };
  const report = await replay(fx, (s) => picture(s, load), opts);
  const total = Math.min(opts.steps ?? fx.steps.length, fx.steps.length);
  return { report, text: summary(report, total), decodeMs, total };
}

(globalThis as unknown as { runReplay: typeof runReplay }).runReplay = runReplay;
