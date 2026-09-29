// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The table modules against Python on real footage: the mat and card tests on the table window of three raw frames of the
// LA final, on real card crops, and the change gate and the layout finder on frames held for a few steps each. The
// expected values are what ml/rifteye_ml computed (test/gen/table_fixtures.py); the frames and fixtures come from a
// broadcast and are private (D-006), so these tests need RIFTEYE_M3=~/rifteye-data/m3 and skip without it. The gate over
// all 240 frames and the layout finder on the first seconds decode the JPEGs and so run in Chromium (e2e/table.spec.ts).

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { autoLayout, borders, cardSize, cardSizeFinder, tableWindow } from '../src/autolayout';
import { ChangeGate, gateSettings, skin, viewHeight, type GateSettings } from '../src/changegate';
import * as image from '../src/image';
import { box, LAYOUTS } from '../src/layouts';
import { borderMask, detail, FACE_DOWN_DETAIL, matColour, notmatMask } from '../src/matcrops';
import type { Layout, RgbImage } from '../src/types';
import { replayDetector, type DetectorCall } from './table-replay';

const M3 = process.env['RIFTEYE_M3'];
const FX = M3 ? join(M3, 'fixtures/table') : '';
const ready = (...names: string[]): boolean => Boolean(M3) && names.every((n) => existsSync(join(FX, n)));

const sha16 = (a: Uint8Array | Int32Array | Float64Array): string => createHash('sha256').update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest('hex').slice(0, 16);
const json = <T>(name: string): T => JSON.parse(readFileSync(join(FX, name), 'utf8')) as T;

/** A raw RGB frame (no header) of the LA final. */
const rawFrame = (name: string): RgbImage => {
  const b = readFileSync(join(M3!, 'frames/la-final-rgb', `${name}.rgb`));
  return { width: 1920, height: 1080, data: new Uint8Array(b.buffer, b.byteOffset, b.byteLength) };
};

const layout = LAYOUTS['la-rq'];

/** The gate's view of a frame as live/pipeline.py Recognizer.watch makes it: the table window, 320 px wide, bilinear. */
function viewOf(frame: RgbImage, vw = 320): RgbImage {
  const roi = image.crop(frame, box(layout, frame.width, frame.height));
  return image.resize(roi, [vw, viewHeight(layout.table, vw)], 'bilinear');
}

interface Ev {
  t: number;
  box: number[];
  kind: string;
  area: number;
  before_mat: number;
  after_mat: number;
  extra: Record<string, number>;
}
interface GateRun {
  settings: GateSettings;
  events: { i: number; events: Ev[] }[];
  checkpoints: { i: number; background: string; still: string }[];
  last_same: string;
  startup_hand: string;
  off_table: number;
  mat: number[];
  kinds: string[];
}

/** Feeds views to a gate as Python did and compares what it fires and keeps. */
function checkGate(views: RgbImage[], times: number[], run: GateRun): void {
  const gate = new ChangeGate(gateSettings(run.settings));
  const fired = new Map<number, Ev[]>();
  const checkpoints = new Map(run.checkpoints.map((c) => [c.i, c]));
  views.forEach((v, i) => {
    const events = gate.feed(times[i]!, v);
    if (events.length > 0) fired.set(i, events as unknown as Ev[]);
    const c = checkpoints.get(i);
    if (c) {
      const st = gate.state();
      expect(sha16(st.background!), `still table after frame ${i}`).toBe(c.background);
      expect(sha16(st.still), `still counters after frame ${i}`).toBe(c.still);
    }
  });
  expect([...fired].map(([i, events]) => ({ i, events }))).toEqual(run.events);
  const st = gate.state();
  expect(sha16(st.lastSame)).toBe(run.last_same);
  expect(sha16(st.startupHand)).toBe(run.startup_hand);
  expect(st.offTable).toBe(run.off_table);
  expect(st.mat).toEqual(run.mat);
  expect(gate.events.map((e) => e.kind)).toEqual(run.kinds);
}

interface FramesFixture {
  frames: {
    index: number;
    file: string;
    roi: number[];
    mat_colour: number[];
    mat_colour_step4: number[];
    mat_colour_frame: number[];
    masks: Record<string, { mat?: number[]; tol?: number; count: number; sha: string }>;
    view: { shape: number[]; sha: string; skin_count: number; skin_sha: string };
  }[];
  gate_2: GateRun & { order: number[]; fps: number };
  gate_5: GateRun & { order: number[]; fps: number };
}

describe.skipIf(!ready('frames.json'))('the mat tests and the gate on raw frames of the LA final', () => {
  const fx = ready('frames.json') ? json<FramesFixture>('frames.json') : null;

  it('find the mat\'s colour, the pixels far from it and the dark borders in the table window as Python does', () => {
    for (const f of fx!.frames) {
      const frame = rawFrame(f.file);
      const roi = image.crop(frame, f.roi as [number, number, number, number]);
      expect(matColour(roi), `${f.file} mat`).toEqual(f.mat_colour);
      expect(matColour(roi, 4), `${f.file} every 4th`).toEqual(f.mat_colour_step4);
      expect(matColour(frame), `${f.file} whole frame`).toEqual(f.mat_colour_frame);
      for (const [name, want] of Object.entries(f.masks)) {
        const m = name === 'border' ? borderMask(roi) : notmatMask(roi, want.mat!, want.tol!);
        expect(m.data.reduce((a, b) => a + b, 0), `${f.file} ${name} count`).toBe(want.count);
        expect(sha16(m.data), `${f.file} ${name}`).toBe(want.sha);
      }
    }
  });

  it('see skin in the gate\'s view of the table', () => {
    for (const f of fx!.frames) {
      const view = viewOf(rawFrame(f.file));
      expect([view.height, view.width, 3]).toEqual(f.view.shape);
      expect(sha16(view.data), `${f.file} view`).toBe(f.view.sha);
      const s = skin(view);
      expect(s.data.reduce((a, b) => a + b, 0)).toBe(f.view.skin_count);
      expect(sha16(s.data), `${f.file} skin`).toBe(f.view.skin_sha);
    }
  });

  for (const key of ['gate_2', 'gate_5'] as const) {
    it(`fire the gate's events and keep its still table as Python does, real frames held a few steps (${key === 'gate_2' ? '2' : '5'} fps)`, () => {
      const run = fx![key];
      const views = new Map<number, RgbImage>();
      const names: Record<number, string> = { 0: 'f0000', 120: 'f0120', 239: 'f0239' };
      for (const i of new Set(run.order)) views.set(i, viewOf(rawFrame(names[i]!)));
      checkGate(
        run.order.map((i) => views.get(i)!),
        run.order.map((_, k) => k / run.fps),
        run,
      );
    });
  }
});

interface CropsFixture {
  threshold: number;
  crops: { set: string; file: string; kind: 'face' | 'back'; width: number; height: number; offset: number; detail: number; mat_colour: number[] }[];
}

describe.skipIf(!ready('crops.json', 'crops.rgb'))('detail on real card crops', () => {
  it('is Python\'s bit for bit, and tells faces from card backs', () => {
    const fx = json<CropsFixture>('crops.json');
    const blob = readFileSync(join(FX, 'crops.rgb'));
    expect(fx.threshold).toBe(FACE_DOWN_DETAIL);
    let faces = 0;
    let backs = 0;
    for (const c of fx.crops) {
      const im: RgbImage = { width: c.width, height: c.height, data: new Uint8Array(blob.buffer, blob.byteOffset + c.offset, c.width * c.height * 3) };
      expect(detail(im), `${c.set}/${c.file}`).toBe(c.detail);
      expect(matColour(im), `${c.set}/${c.file} mat colour`).toEqual(c.mat_colour);
      if (c.kind === 'face') faces += detail(im) >= FACE_DOWN_DETAIL ? 1 : 0;
      else backs += detail(im) < FACE_DOWN_DETAIL ? 1 : 0;
    }
    // the classification is Python's too; this says the fixture has both kinds in it
    expect(faces).toBeGreaterThan(20);
    expect(backs).toBeGreaterThan(5);
  });
});

interface LayoutRecord {
  frames: number[];
  borders: number[];
  table_window: { window: number[]; mat: number[]; share: number } | null;
  calls: DetectorCall[];
  auto_layout: Layout | null;
  card_size?: number;
  card_size_finder?: number | null;
  auto_layout_finder?: Layout | null;
}

describe.skipIf(!ready('autolayout.json'))('the layout finder on real frames', () => {
  it('finds the borders, the table window and the card size of frames a minute apart, with the detector replaced by what it said', async () => {
    const rec = json<{ raw3: LayoutRecord }>('autolayout.json').raw3;
    const names = ['f0000', 'f0120', 'f0239'];
    const frames = rec.frames.map((i) => rawFrame(names[[0, 120, 239].indexOf(i)]!));
    expect(borders(frames)).toEqual(rec.borders);
    const tw = tableWindow(frames);
    expect(tw).toEqual(rec.table_window);
    const r = replayDetector(rec.calls, frames);
    expect(await autoLayout(frames, r.detect)).toEqual(rec.auto_layout);
    expect(r.used()).toBe(rec.calls.length);
    if (tw !== null) expect(await cardSize(replayDetector(rec.calls, frames).detect, frames, tw.window)).toBe(rec.card_size);
  });

  it('finds the card size without the detector, by the bootstrap finder, as Python does (to 1e-9: the rectangles are fitted with the browser\'s sin and cos)', async () => {
    const rec = json<{ raw3: LayoutRecord }>('autolayout.json').raw3;
    const names = ['f0000', 'f0120', 'f0239'];
    const frames = rec.frames.map((i) => rawFrame(names[[0, 120, 239].indexOf(i)]!));
    const tw = tableWindow(frames)!;
    expect(cardSizeFinder(frames, tw.window, tw.mat)).toBeCloseTo(rec.card_size_finder!, 9);
    const layout = await autoLayout(frames);
    expect(layout).toEqual(rec.auto_layout_finder);
  }, 120_000); // 36 passes of the finder over a 1190 x 1015 window
});
