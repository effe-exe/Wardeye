// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The page of table.spec.ts: it decodes the LA final's JPEGs as the extension does, makes the change gate's view of each
// (the table window, 320 px wide, bilinear: live/pipeline.py Recognizer.watch), runs the gate over them and the layout
// finder over the first seconds, and hands what came out to the test to compare with Python's.

import { autoLayout, borders, cardSize, cardSizeFinder, tableWindow } from '../src/autolayout';
import { ChangeGate, gateSettings, viewHeight, type GateSettings } from '../src/changegate';
import * as image from '../src/image';
import { box, LAYOUTS } from '../src/layouts';
import type { Layout, RgbImage } from '../src/types';
import { replayDetector, type DetectorCall } from '../test/table-replay';

/** The SHA-256 of some bytes, as hex. */
async function sha256(a: Uint8Array | Int32Array | Float64Array): Promise<string> {
  const bytes = new Uint8Array(a.byteLength); // a copy of our own, so that it is an ArrayBuffer's and not maybe a shared one's
  bytes.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** A JPEG as the extension decodes a frame: no colour conversion, no premultiplying, drawn on a 2D canvas, read back. */
async function decode(url: string): Promise<RgbImage> {
  const blob = await (await fetch(url)).blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return image.fromRgba(ctx.getImageData(0, 0, width, height).data, width, height);
}

const layout = LAYOUTS['la-rq'];

/** The gate's view of a frame. */
function viewOf(frame: RgbImage, vw = 320): RgbImage {
  return image.resize(image.crop(frame, box(layout, frame.width, frame.height)), [vw, viewHeight(layout.table, vw)], 'bilinear');
}

export interface GateRunParams {
  settings: GateSettings;
  /** The frames after which the gate's memory is hashed. */
  checkpoints: number[];
}

export interface GateResult {
  /** SHA-256 of each decoded frame's RGB, and the first 16 hex digits of each view's. */
  rgbSha: string[];
  viewSha: string[];
  runs: {
    events: { i: number; events: unknown[] }[];
    checkpoints: { i: number; background: string; still: string }[];
    last_same: string;
    startup_hand: string;
    off_table: number;
    mat: number[];
    kinds: string[];
    ms: number;
  }[];
  decodeMs: number;
}

export interface LayoutAttempt {
  frames: number[];
  calls: DetectorCall[];
}

export interface LayoutResult {
  borders: number[];
  table_window: { window: number[]; mat: number[]; share: number } | null;
  auto_layout: Layout | null;
  card_size: number | null;
  /** Without the detector: the bootstrap finder's card size and the layout it gives. */
  card_size_finder: number | null;
  auto_layout_finder: Layout | null;
  used: number;
  error: string | null;
}

const short = async (a: Uint8Array | Int32Array | Float64Array): Promise<string> => (await sha256(a)).slice(0, 16);

const api = {
  /** The gate over every frame, once for each set of settings. */
  async gate(files: string[], times: number[], runs: GateRunParams[]): Promise<GateResult> {
    const t0 = performance.now();
    const views: RgbImage[] = [];
    const rgbSha: string[] = [];
    const viewSha: string[] = [];
    for (const f of files) {
      const im = await decode(`/frames/${f}`);
      rgbSha.push(await sha256(im.data));
      const v = viewOf(im);
      views.push(v);
      viewSha.push(await short(v.data));
    }
    const decodeMs = performance.now() - t0;
    const out: GateResult['runs'] = [];
    for (const run of runs) {
      const t1 = performance.now();
      const gate = new ChangeGate(gateSettings(run.settings));
      const events: { i: number; events: unknown[] }[] = [];
      const checkpoints: { i: number; background: string; still: string }[] = [];
      for (const [i, v] of views.entries()) {
        const fired = gate.feed(times[i]!, v);
        if (fired.length > 0) events.push({ i, events: fired });
        if (run.checkpoints.includes(i)) {
          const st = gate.state();
          checkpoints.push({ i, background: await short(st.background!), still: await short(st.still) });
        }
      }
      const st = gate.state();
      out.push({
        events,
        checkpoints,
        last_same: await short(st.lastSame),
        startup_hand: await short(st.startupHand),
        off_table: st.offTable,
        mat: st.mat!,
        kinds: gate.events.map((e) => e.kind),
        ms: performance.now() - t1,
      });
    }
    return { rgbSha, viewSha, runs: out, decodeMs };
  },

  /** The layout finder on some of the frames, the detector replaced by what Python's said. */
  async layout(files: string[], attempts: LayoutAttempt[]): Promise<LayoutResult[]> {
    const decoded = new Map<number, RgbImage>();
    const results: LayoutResult[] = [];
    for (const a of attempts) {
      for (const i of a.frames) if (!decoded.has(i)) decoded.set(i, await decode(`/frames/${files[i]!}`));
      const frames = a.frames.map((i) => decoded.get(i)!);
      const tw = tableWindow(frames);
      const r = replayDetector(a.calls, frames);
      let found: Layout | null = null;
      let size: number | null = null;
      let error: string | null = null;
      try {
        found = await autoLayout(frames, r.detect);
        if (tw !== null) size = await cardSize(replayDetector(a.calls, frames).detect, frames, tw.window);
      } catch (e) {
        error = String(e);
      }
      results.push({
        borders: borders(frames),
        table_window: tw,
        auto_layout: found,
        card_size: size,
        card_size_finder: tw === null ? null : cardSizeFinder(frames, tw.window, tw.mat),
        auto_layout_finder: await autoLayout(frames),
        used: r.used(),
        error,
      });
    }
    return results;
  },
};

export type TableApi = typeof api;

declare global {
  interface Window {
    rifteyeTable: TableApi;
  }
}

window.rifteyeTable = api;
