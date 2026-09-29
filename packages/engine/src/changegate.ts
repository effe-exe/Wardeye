// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Layer 1, the change gate, ported from ml/rifteye_ml/changegate.py: it watches a small, downscaled view of the table
// and fires when a region differs from the still table and has settled (no motion for `settle_s`). Hands passing over
// the table move, so they are ignored until they leave. Each event carries a box and a guess at its kind:
//
//  - appeared: the region was bare playmat and now is not (a card was played or moved here);
//  - disappeared: the region was covered and is now bare playmat (a card left or moved away);
//  - changed: covered before and after (a card turned, was replaced, or got a counter).
//
// The gate takes frames of the table view already downscaled (live/pipeline.py makes them with a bilinear resize).
// The arithmetic is numpy's: integers where numpy has them, float32 (Math.fround at each step) where numpy has
// float32, so the events and the still table it keeps are the same as Python's.

import { channelMedians } from './matcrops';
import { binaryDilation, binaryOpening, findObjects, label, type Mask } from './ndimage';
import { median, pyRound } from './pynum';
import type { RgbImage } from './types';

/** changegate.GateSettings: the field names are Python's, so a settings file written by Python loads as it is. */
export interface GateSettings {
  fps: number;
  /** The table view is downscaled to this width. */
  width: number;
  /** Per-pixel colour distance that counts as a change. */
  diff: number;
  /** Per-pixel distance between consecutive frames that counts as motion. */
  motion: number;
  /** A changed region must be still this long. */
  settle_s: number;
  /** Of one card's area at this scale. */
  min_area: number;
  /** A card's long side as a fraction of the frame height. */
  card_long_frac: number;
  frame_h: number;
  /** More than this share of the table changed at once: a camera cut. */
  global_cut: number;
  /** Still, unchanged pixels drift toward the frame (slow light changes). */
  adapt: number;
  /** Below this share of bare playmat the frame is not the table view (a cutaway). */
  min_mat: number;
  /** A region this much covered by skin (dilated) waits for the hand to leave. */
  hand_share: number;
  /** Share of a region that must differ by 2x `diff`: cards do, light and codec do not. */
  strong: number;
  /** Boxes (fractions of the table view) never watched. */
  ignore: [number, number, number, number][];
  /** Bare playmat colour; estimated from the first frame if null. */
  mat_rgb: [number, number, number] | null;
}

/** GateSettings with Python's defaults, `over` applied on top. */
export function gateSettings(over: Partial<GateSettings> = {}): GateSettings {
  return {
    fps: 5.0,
    width: 320,
    diff: 28.0,
    motion: 18.0,
    settle_s: 0.6,
    min_area: 0.35,
    card_long_frac: 131 / 1080,
    frame_h: 1080,
    global_cut: 0.5,
    adapt: 0.05,
    min_mat: 0.25,
    hand_share: 0.05,
    strong: 0.3,
    ignore: [],
    mat_rgb: null,
    ...over,
  };
}

export type ChangeKind = 'appeared' | 'disappeared' | 'changed' | 'cut';

/** changegate.ChangeEvent. */
export interface ChangeEvent {
  /** Seconds from the start of the window. */
  t: number;
  /** x0, y0, x1, y1 in the gate's downscaled table view. */
  box: [number, number, number, number];
  kind: ChangeKind;
  area: number;
  /** Share of the box that was bare mat before. */
  before_mat: number;
  after_mat: number;
  /** t_before: when the region last looked like the still table, about when the hand arrived. */
  extra: { t_before?: number };
}

const f32 = Math.fround;

// skin()'s constants and their products with each 8-bit value, in float32 (numpy turns the Python floats into
// float32 before it multiplies)
const TIMES_299 = Float32Array.from({ length: 256 }, (_, v) => f32(f32(0.299) * v));
const TIMES_587 = Float32Array.from({ length: 256 }, (_, v) => f32(f32(0.587) * v));
const TIMES_114 = Float32Array.from({ length: 256 }, (_, v) => f32(f32(0.114) * v));
const K_CR = f32(0.713);
const K_CB = f32(0.564);

/** Skin-coloured pixels (YCrCb box), in float32 as numpy does it. On a red mat skin separates cleanly: the mat's Cr
 * is about 200. */
export function skin(frame: RgbImage): Mask {
  const { width, height, data } = frame;
  const out = new Uint8Array(width * height);
  for (let i = 0, o = 0; i < out.length; i++, o += 3) {
    const r = data[o]!;
    const g = data[o + 1]!;
    const b = data[o + 2]!;
    // Two quick outs that the full test below would give too. Cr is 128 plus 0.713 times red less the luma: with red the
    // least of the three the luma is at least red, so Cr is not over 135; and with 0.587 (r - g) + 0.114 (r - b), the
    // red less the luma, at 75 or more, Cr is over 180.
    if (r <= g && r <= b) continue;
    if (587 * (r - g) + 114 * (r - b) > 75000) continue;
    const y = f32(f32(TIMES_299[r]! + TIMES_587[g]!) + TIMES_114[b]!);
    const cr = f32(128 + f32(K_CR * f32(r - y)));
    const cb = f32(128 + f32(K_CB * f32(b - y)));
    out[i] = cr > 135 && cr < 180 && cb > 80 && cb < 130 && y > 60 ? 1 : 0;
  }
  return { width, height, data: out };
}

/** How many of a picture's pixels are within 45 of `mat` in every channel: the bare playmat. */
function matCount(data: Uint8Array, mat: readonly number[]): number {
  const m0 = mat[0]!;
  const m1 = mat[1]!;
  const m2 = mat[2]!;
  let n = 0;
  for (let o = 0; o < data.length; o += 3) {
    if (Math.max(Math.abs(data[o]! - m0), Math.abs(data[o + 1]! - m1), Math.abs(data[o + 2]! - m2)) < 45) n++;
  }
  return n;
}

/** Feed frames of the table view (h x w x 3, rows top to bottom) in order; collect `events`. */
export class ChangeGate {
  readonly s: GateSettings;
  readonly events: ChangeEvent[] = [];
  private background: Uint8Array | null = null;
  private prev: Uint8Array | null = null;
  /** Frames each pixel has been still. */
  private still: Int32Array = new Int32Array(0);
  /** Last time each pixel matched the still table. */
  private lastSame: Float64Array = new Float64Array(0);
  /** Hands in the frame the still table was first taken from. */
  private startupHand: Uint8Array = new Uint8Array(0);
  private mat: [number, number, number] | null = null;
  /** Consecutive frames that were not the table view. */
  private offTable = 0;
  private size = 0;
  private w = 0;
  private h = 0;
  // Scratch space kept between frames, and lookup tables for the settings that decide the per-pixel tests. A pixel's
  // distance is the sum of its three channels' steps over 3, in float32 (numpy's _dist): the sum is a whole number up
  // to 765, so "is it over `motion`", "over `diff`" and "over twice `diff`" are tables indexed by the sum.
  private sums = new Uint16Array(0);
  private changed = new Uint8Array(0);
  private settledRaw = new Uint8Array(0);
  private readonly movingLut = new Uint8Array(766);
  private readonly changedLut = new Uint8Array(766);
  private readonly strongLut = new Uint8Array(766);
  // what a still, unchanged pixel's channel does when the frame differs from the still table by d (the index is
  // d + 255): float32(adapt * d), and whether the still table's value stays (0), goes down by one (1) or is worked out (2)
  private readonly driftT = new Float32Array(511);
  private readonly driftMode = new Uint8Array(511);
  private lutFor: [number, number, number] | null = null;

  constructor(s: GateSettings) {
    this.s = s;
  }

  /** The lookup tables, made again when the settings they come from have changed. */
  private tables(): void {
    const { motion, diff, adapt } = this.s;
    if (this.lutFor && this.lutFor[0] === motion && this.lutFor[1] === diff && this.lutFor[2] === adapt) return;
    const motionF = f32(motion);
    const diffF = f32(diff);
    const strongF = f32(2 * diff);
    for (let sum = 0; sum <= 765; sum++) {
      const d = f32(sum / 3);
      this.movingLut[sum] = d > motionF ? 1 : 0;
      this.changedLut[sum] = d > diffF ? 1 : 0;
      this.strongLut[sum] = d > strongF ? 1 : 0;
    }
    const adaptF = f32(adapt);
    for (let k = 0; k < 511; k++) {
      const d = k - 255;
      const t = f32(adaptF * d);
      this.driftT[k] = t;
      // b + t, cut to a whole number, is b while 0 <= t < 0.5; and b - 1 while -0.5 < t <= -2^-10 (the step to b is
      // more than the float32's rounding at 255), where b is at least 1 because the frame's value, d less, is not below 0
      this.driftMode[k] = t >= 0 && t < 0.5 ? 0 : d < 0 && t <= -1 / 1024 && t > -0.5 ? 1 : 2;
    }
    this.lutFor = [motion, diff, adapt];
  }

  /** What the gate remembers, for tests: the still table (RGB bytes, null before the first table frame), how many
   * frames each pixel has been still, when each last matched the still table, the hands of the first frame, the
   * mat colour and the count of frames off the table. Not copies: do not change them. */
  state(): {
    background: Uint8Array | null;
    still: Int32Array;
    lastSame: Float64Array;
    startupHand: Uint8Array;
    mat: [number, number, number] | null;
    offTable: number;
  } {
    return { background: this.background, still: this.still, lastSame: this.lastSame, startupHand: this.startupHand, mat: this.mat, offTable: this.offTable };
  }

  /** Whether the pixel at byte offset `o` of `data` is bare playmat. */
  private isMat(data: Uint8Array, o: number): boolean {
    const m = this.mat!;
    return Math.max(Math.abs(data[o]! - m[0]), Math.abs(data[o + 1]! - m[1]), Math.abs(data[o + 2]! - m[2])) < 45;
  }

  /** One frame at time `t`: the events it fires (also appended to `events`). */
  feed(t: number, frame: RgbImage): ChangeEvent[] {
    const s = this.s;
    const { width: w, height: h, data } = frame;
    if (this.mat === null) this.mat = s.mat_rgb ? [s.mat_rgb[0], s.mat_rgb[1], s.mat_rgb[2]] : channelMedians(frame);
    if (matCount(data, this.mat) / (w * h) < s.min_mat) {
      // a cutaway (player cam, graphic): wait for the table
      this.offTable += 1;
      return [];
    }
    if (this.background === null) {
      this.w = w;
      this.h = h;
      this.size = w * h;
      this.background = data.slice();
      this.prev = data.slice();
      this.still = new Int32Array(this.size);
      this.lastSame = new Float64Array(this.size).fill(t);
      this.sums = new Uint16Array(this.size);
      this.changed = new Uint8Array(this.size);
      this.settledRaw = new Uint8Array(this.size);
      // A hand in this first frame is not part of the table; when it leaves, that is not a change.
      this.startupHand = binaryDilation(skin(frame), 6).data;
      return [];
    }
    if (w !== this.w || h !== this.h) throw new Error(`the table view is ${w} x ${h}, and the gate began at ${this.w} x ${this.h}`);
    if (this.offTable) {
      // back on the table: the old still table stays, so changes made meanwhile are found
      this.offTable = 0;
      this.prev!.set(data);
      this.still.fill(0);
      return [];
    }
    const size = this.size;
    const background = this.background;
    const prev = this.prev!;
    const still = this.still;
    const lastSame = this.lastSame;

    this.tables();
    const { movingLut, changedLut, strongLut, sums, changed } = this;
    changed.fill(0);
    let changedCount = 0;
    for (let i = 0, o = 0; i < size; i++, o += 3) {
      const r = data[o]!;
      const g = data[o + 1]!;
      const b = data[o + 2]!;
      still[i] = movingLut[Math.abs(r - prev[o]!) + Math.abs(g - prev[o + 1]!) + Math.abs(b - prev[o + 2]!)]! ? 0 : still[i]! + 1;
      const sum = Math.abs(r - background[o]!) + Math.abs(g - background[o + 1]!) + Math.abs(b - background[o + 2]!);
      sums[i] = sum;
      if (changedLut[sum]) {
        changed[i] = 1;
        changedCount++;
      } else {
        lastSame[i] = t;
      }
    }
    prev.set(data);
    for (const [fx0, fy0, fx1, fy1] of s.ignore) {
      const y0 = Math.min(h, Math.max(0, pyRound(fy0 * h)));
      const y1 = Math.min(h, Math.max(0, pyRound(fy1 * h)));
      const x0 = Math.min(w, Math.max(0, pyRound(fx0 * w)));
      const x1 = Math.min(w, Math.max(0, pyRound(fx1 * w)));
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          if (changed[y * w + x]) {
            changed[y * w + x] = 0;
            changedCount--;
          }
        }
      }
    }
    const hand = binaryDilation(skin(frame), 4).data;
    const fresh: ChangeEvent[] = [];
    if (changedCount / size > s.global_cut) {
      // camera cut or layout change: reset, report once
      fresh.push({ t, box: [0, 0, w, h], kind: 'cut', area: changedCount, before_mat: 0.0, after_mat: 0.0, extra: {} });
      background.set(data);
      still.fill(0);
      lastSame.fill(t);
      this.events.push(...fresh);
      return fresh;
    }
    const settle = Math.max(1, pyRound(s.settle_s * s.fps));
    // the changed pixels that have been still long enough; the opening of none is none, so most frames stop here
    const settledRaw = this.settledRaw;
    settledRaw.fill(0);
    let settledCount = 0;
    if (changedCount > 0) {
      for (let i = 0; i < size; i++) {
        if (changed[i] && still[i]! >= settle) {
          settledRaw[i] = 1;
          settledCount++;
        }
      }
    }
    let labels: Int32Array = new Int32Array(0);
    let count = 0;
    let objects: [number, number, number, number][] = [];
    if (settledCount > 0) {
      ({ labels, count } = label(binaryOpening({ width: w, height: h, data: settledRaw }, 3)));
      objects = findObjects(labels, w, h, count);
    }
    const scale = s.width / 1920;
    const cardLong = s.card_long_frac * s.frame_h * scale;
    const minArea = (s.min_area * cardLong * cardLong) / 1.4;
    for (let n = 1; n <= count; n++) {
      const [y0, y1, x0, x1] = objects[n - 1]!;
      // the blob's own pixels, as offsets into the picture
      const region: number[] = [];
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (labels[y * w + x] === n) region.push(y * w + x);
      const area = region.length;
      // Only act once the whole blob is still: part of it still moving means a hand is there.
      const pad = 2;
      let moving = false;
      for (let y = Math.max(0, y0 - pad); y < Math.min(h, y1 + pad) && !moving; y++) {
        for (let x = Math.max(0, x0 - pad); x < Math.min(w, x1 + pad); x++) {
          if (changed[y * w + x] && still[y * w + x]! < settle) {
            moving = true;
            break;
          }
        }
      }
      if (moving) continue;
      // A resting hand is still but is not the table: wait until it leaves, and never absorb it.
      // Arms enter from the sides, so a region touching the side edges waits too.
      let handPx = 0;
      for (const i of region) handPx += hand[i]!;
      if (handPx / area > s.hand_share || x0 === 0 || x1 >= w) continue;
      let beforeMat = 0;
      let afterMat = 0;
      let strongPx = 0;
      let startupPx = 0;
      for (const i of region) {
        if (this.isMat(background, i * 3)) beforeMat++;
        if (this.isMat(data, i * 3)) afterMat++;
        if (strongLut[sums[i]!]) strongPx++;
        startupPx += this.startupHand[i]!;
      }
      const before = beforeMat / area;
      const after = afterMat / area;
      const weak = strongPx / area < s.strong;
      const kind =
        weak || (before > 0.6 && after > 0.6)
          ? 'noise'
          : before > 0.6 && after < 0.4
            ? 'appeared'
            : before < 0.4 && after > 0.6
              ? 'disappeared'
              : 'changed';
      const wasHand = startupPx / area > 0.3;
      for (const i of region) this.startupHand[i] = 0;
      if (area >= minArea && kind !== 'noise' && !wasHand) {
        // otherwise absorb it silently. When the region last looked like the still table: about when the hand arrived.
        const tBefore = median(region.map((i) => lastSame[i]!));
        fresh.push({ t, box: [x0, y0, x1, y1], kind, area, before_mat: before, after_mat: after, extra: { t_before: tBefore } });
      }
      // absorb the change (small ones silently)
      for (const i of region) {
        background[i * 3] = data[i * 3]!;
        background[i * 3 + 1] = data[i * 3 + 1]!;
        background[i * 3 + 2] = data[i * 3 + 2]!;
      }
    }
    // Slow light changes: still, unchanged pixels drift toward the current frame (float32, then cut to 8 bits).
    const { driftMode, driftT } = this;
    for (let i = 0; i < size; i++) {
      if (still[i]! < settle || changed[i] || hand[i]) continue;
      for (let c = i * 3, e = c + 3; c < e; c++) {
        const b = background[c]!;
        const k = data[c]! - b + 255;
        const mode = driftMode[k]!;
        if (mode === 0) continue;
        if (mode === 1) {
          background[c] = b - 1;
          continue;
        }
        const v = f32(b + driftT[k]!);
        background[c] = v < 0 ? 0 : v > 255 ? 255 : v; // a Uint8Array cuts the fraction, as astype(np.uint8) does
      }
    }
    this.events.push(...fresh);
    return fresh;
  }
}

/** Run the gate over (t, frame) pairs. */
export function run(frames: Iterable<readonly [number, RgbImage]>, s: GateSettings): ChangeGate {
  const gate = new ChangeGate(s);
  for (const [t, f] of frames) gate.feed(t, f);
  return gate;
}

/** The height of the gate's view of a layout's table window at width `viewW`: the table's aspect on a 16:9 frame,
 * rounded to an even number (live/pipeline.py, Recognizer.watch). */
export function viewHeight(table: readonly [number, number, number, number], viewW: number): number {
  const [tx0, ty0, tx1, ty1] = table;
  return pyRound(((viewW * (ty1 - ty0) * 1080) / ((tx1 - tx0) * 1920)) / 2) * 2;
}

/** A box in the gate's downscaled table view -> frame pixels (the view keeps the table's aspect): reviewpack.view_to_frame. */
export function viewToFrame(
  box: readonly [number, number, number, number],
  table: readonly [number, number, number, number],
  viewW: number,
  frameW: number,
  frameH: number,
): [number, number, number, number] {
  const [tx0, ty0, tx1, ty1] = table;
  const viewH = viewHeight(table, viewW);
  const sx = ((tx1 - tx0) * frameW) / viewW;
  const sy = ((ty1 - ty0) * frameH) / viewH;
  const [x0, y0, x1, y1] = box;
  return [tx0 * frameW + x0 * sx, ty0 * frameH + y0 * sy, tx0 * frameW + x1 * sx, ty0 * frameH + y1 * sy];
}
