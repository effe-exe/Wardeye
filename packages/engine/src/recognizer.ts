// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The live recogniser, ported from ml/rifteye_ml/live/pipeline.py: cards found, tracked and named frame by frame, as
// a stream plays.
//
// Each frame the finder gives card boxes inside the layout's table window: the trained detector's (detector_boxes),
// or without one the bootstrap finder's (matcrops.find_cards, isolated cards on the mat). Boxes are matched to the
// tracks of cards already on the table by position and size, so a card is identified
// once, not once per frame: new tracks and uncertain ones get a crop identified against the gallery pyramid, and
// named tracks are re-checked now and then. A track's name is its readings' mean probability (softmax over each
// crop's card scores). Plain crops are face-down cards: they are never identified (D-005). Only the table window is
// ever looked at; the broadcast's panels with the players' hands are hidden information and never read.
//
// A card named on the table becomes a "played" event; a named card gone for a while, "left". The finder only sees
// cards lying on their own, so the change gate watches the whole table too: when a region settles after a change
// (a card put on a stack, a rune channelled), the region is read like a crop, and a confident read is a play even
// though no track holds the card.
//
// The port keeps pipeline.py's names (camelCase), its order and its arithmetic, so the two read side by side and,
// fed the same frames, boxes and embeddings, give the same state and events. Where numpy computes in float32 (the
// scene's thumbnails), so does this (Math.fround at each step). A few things can differ in their last bits: the
// scene's two vector norms (numpy's BLAS sums them in its own order), exp, cos and sin (the platform's), and the
// similarity fit (a closed form for the SVD); none changes a step of the LA final replay. `step` is asynchronous
// because the finder and the encoder are.

import { ChangeGate, gateSettings, viewHeight, viewToFrame, type GateSettings } from './changegate';
import { dist, mean as meanOf, npRound, overlapArea } from './geometry';
import * as image from './image';
import { box as layoutBox, cardPx, side as layoutSide, sides as layoutSides } from './layouts';
import { linearSumAssignment } from './lsap';
import { FACE_DOWN_DETAIL, detail, findCards, matColour, notmatMask, type Mask } from './matcrops';
import { pyMod, pyRound } from './pynum';
import { ROTATIONS, argsortDescending, bestSimilarities, type Pyramid } from './retrieval';
import type { CardBox, CatalogRow, Encoder, Finder, Layout, RgbImage } from './types';

/** Fitted on the M0 real labels (reviewpack identity). */
export const TEMPERATURE = 0.0212;
/** A named card out of sight this long has gone: not a hand over it, dice, a card on top. */
export const KEEP_S = 60.0;
/** A named card that vanished this recently and is named again elsewhere has moved. */
export const MOVE_S = 10.0;
/** Set up before the game and never moved: pinned where they are named. */
export const STATIC: readonly string[] = ['Legend', 'Battlefield'];
/** Tracked, but never labelled, listed or announced: not worth watching. */
export const QUIET: readonly string[] = ['Rune'];
export const KINDS: ReadonlyMap<string, string> = new Map([
  ['Legend', 'legend'],
  ['Battlefield', 'battlefield'],
  ['Rune', 'rune'],
]);

/** kind in STATIC + QUIET */
const settled = (kind: string): boolean => STATIC.includes(kind) || QUIET.includes(kind);

const DEG_TO_RAD = Math.PI / 180; // math.radians
const RAD_TO_DEG = 180 / Math.PI; // math.degrees
const f32 = Math.fround;

// --- what the overlay draws ---------------------------------------------------------------------------------------

/** One of a track's candidate cards: that card's best-matching printing, and its probability. */
export interface Guess {
  printing_id: string;
  card_id: string;
  name: string;
  p: number;
}

/** A card that lies under another. */
export interface Under {
  id: string;
  name: string;
  printing_id: string;
}

/** A card on the table as the overlay draws it (apps/extension/src/geometry.ts's Track). The gate's plays are
 * listed the same way for a few seconds, without `under`. */
export interface StateTrack {
  id: string;
  /** The corners in frame px, rounded to 0.1. */
  quad: [number, number][];
  side: string;
  /** new, named, unsure or facedown. */
  state: string;
  printing_id: string | null;
  name: string;
  confidence: number;
  guesses: Guess[];
  /** When it was first seen, in s. */
  since: number;
  /** card, legend, battlefield or rune. */
  kind: string;
  /** Out of sight (under a hand or another card) but still on the board: listed, not drawn. */
  hidden: boolean;
  under?: Under[];
}

export interface Legend {
  printing_id: string;
  name: string;
}

export interface Player {
  side: string;
  label: string;
  legend: Legend | null;
}

/** What `step` returns: the state the overlay draws (apps/extension/src/geometry.ts's State). */
export interface RecognizerState {
  t: number;
  /** live, or away while the video is not the table camera. */
  status: string;
  message: string;
  title: string;
  frame: { width: number; height: number };
  players: Player[];
  layout: { name: string; table: number[] };
  tracks: StateTrack[];
}

/** played, moved, changed or left. */
export interface RecognizerEvent {
  t: number;
  kind: string;
  text: string;
  printing_id: string | null;
  track: string;
  side: string;
}

/** A play the gate found, drawn until `until`. */
export interface Flash extends StateTrack {
  until: number;
}

/** A named card the finder lost, remembered for a while. */
export interface Ghost {
  t: number;
  card: string;
  x: number;
  y: number;
  name: string;
  printing_id: string | null;
  side: string;
  id: string;
}

/** Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first. */
export type Candidate = [card: string, p: number, score: number, row: number];

type Box4 = [number, number, number, number];

// --- tracks and boxes ---------------------------------------------------------------------------------------------

export class Track {
  id: string;
  box: CardBox;
  first: number;
  last: number;
  hits = 1;
  /** Identifications of this card so far. */
  reads = 0;
  /** Face-down looks in a row. */
  down = 0;
  /** card_id -> summed probability. */
  prob = new Map<string, number>();
  /** card_id -> (score, gallery row). */
  bestRow = new Map<string, [number, number]>();
  lastRead = -1e9;
  /** card_id once the track has been announced. */
  named: string | null = null;
  side = '';
  /** The named card's type (Unit, Rune, Legend, ...). */
  kind = '';
  /** A legend or battlefield: kept where it is all game. */
  pinned = false;

  constructor(id: string, box: CardBox, first: number, last: number, side = '') {
    this.id = id;
    this.box = box;
    this.first = first;
    this.last = last;
    this.side = side;
  }

  top(): [string, number][] {
    if (!this.reads) return [];
    // sorted by -p: stable, so equal probabilities keep the order the cards were first read in
    return [...this.prob].map(([c, p]): [string, number] => [c, p / this.reads]).sort((a, b) => b[1] - a[1]);
  }
}

/** The rotated rectangle's corners in frame px. The detector's boxes hold numpy floats in Python (detector_boxes
 * averages a numpy quad), so round() here is numpy's, rint(x * 10) / 10, not Python's. */
export function quad(box: CardBox): [number, number][] {
  const [cx, cy] = box.centre;
  const a = box.angle_deg * DEG_TO_RAD;
  const ux = (Math.cos(a) * box.long_px) / 2;
  const uy = (Math.sin(a) * box.long_px) / 2;
  const vx = (-Math.sin(a) * box.short_px) / 2;
  const vy = (Math.cos(a) * box.short_px) / 2;
  return [
    [npRound(cx + ux + vx, 1), npRound(cy + uy + vy, 1)],
    [npRound(cx + ux - vx, 1), npRound(cy + uy - vy, 1)],
    [npRound(cx - ux - vx, 1), npRound(cy - uy - vy, 1)],
    [npRound(cx - ux + vx, 1), npRound(cy - uy + vy, 1)],
  ];
}

export function aabb(box: CardBox): Box4 {
  const q = quad(box);
  const xs = q.map((p) => p[0]);
  const ys = q.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

/** Out of sight: not seen for more than a second. */
export function hiddenNow(t: number, tr: Track): boolean {
  return t - tr.last > 1.0;
}

/** The new box eased from the old one while the card barely moves: the detector's boxes jitter by a few pixels
 * from frame to frame. A card that moves or turns (exhausted) is followed at once. */
export function smooth(old: CardBox, newBox: CardBox, k = 0.35): CardBox {
  const da = pyMod(newBox.angle_deg - old.angle_deg + 90, 180) - 90;
  if (dist(old.centre, newBox.centre) > 0.1 * newBox.long_px || Math.abs(da) > 10) return newBox;
  return {
    centre: [(1 - k) * old.centre[0] + k * newBox.centre[0], (1 - k) * old.centre[1] + k * newBox.centre[1]],
    long_px: (1 - k) * old.long_px + k * newBox.long_px,
    short_px: (1 - k) * old.short_px + k * newBox.short_px,
    angle_deg: pyMod(old.angle_deg + k * da, 180),
    fill: newBox.fill,
    back: newBox.back ?? false,
  };
}

/** math.hypot(x, y), as CPython computes it: the same vector_norm as math.dist. */
export function hypot(x: number, y: number): number {
  return dist([x, y], [0, 0]);
}

/** The card upright, long side vertical. Only the card's neighbourhood is rotated, not the frame. */
export function cardCrop(frame: RgbImage, box: CardBox): RgbImage {
  const [cx, cy] = box.centre;
  const r = Math.ceil(hypot(box.long_px, box.short_px) / 2) + 2;
  const x0 = Math.trunc(cx) - r;
  const y0 = Math.trunc(cy) - r;
  const local = image.crop(frame, [x0, y0, x0 + 2 * r, y0 + 2 * r]);
  const lx = cx - x0;
  const ly = cy - y0;
  const w = box.short_px;
  const h = box.long_px;
  // local.rotate(...).crop(...) in one: rotateCrop computes only the pixels the crop keeps, the same bytes
  return image.rotateCrop(local, box.angle_deg - 90, { resample: 'bicubic', center: [lx, ly] }, [
    pyRound(lx - w / 2),
    pyRound(ly - h / 2),
    pyRound(lx + w / 2),
    pyRound(ly + h / 2),
  ]);
}

/** The scale, rotation and shift that take the points `src` onto `dst`, least squares (Umeyama). In two dimensions
 * the rotation of U diag(1, sign det(U Vt)) Vt is the turn by atan2(m10 - m01, m00 + m11) of the cross-covariance m,
 * and the singular values summed with those signs are the length of (m00 + m11, m10 - m01), so no SVD is needed. */
export function similarity(
  src: readonly (readonly [number, number])[],
  dst: readonly (readonly [number, number])[],
): [number, [[number, number], [number, number]], [number, number]] {
  const n = src.length;
  const ms = meanOf(src.map((p): [number, number] => [p[0], p[1]]));
  const md = meanOf(dst.map((p): [number, number] => [p[0], p[1]]));
  const a = src.map((p) => [p[0] - ms[0], p[1] - ms[1]] as const);
  const b = dst.map((p) => [p[0] - md[0], p[1] - md[1]] as const);
  let m00 = 0; // (b.T @ a)[0, 0]
  let m01 = 0;
  let m10 = 0;
  let m11 = 0;
  for (let i = 0; i < n; i++) {
    m00 += b[i]![0] * a[i]![0];
    m01 += b[i]![0] * a[i]![1];
    m10 += b[i]![1] * a[i]![0];
    m11 += b[i]![1] * a[i]![1];
  }
  m00 /= n;
  m01 /= n;
  m10 /= n;
  m11 /= n;
  const turn = Math.atan2(m10 - m01, m00 + m11);
  const c = Math.cos(turn);
  const s = Math.sin(turn);
  const rot: [[number, number], [number, number]] = [
    [c, -s],
    [s, c],
  ];
  const squares: number[] = [];
  for (const [x, y] of a) squares.push(x * x, y * y);
  const scale = hypot(m00 + m11, m10 - m01) / Math.max(1e-9, pairwiseSum(squares) / n);
  const shift: [number, number] = [
    md[0] - (scale * rot[0][0] * ms[0] + scale * rot[0][1] * ms[1]),
    md[1] - (scale * rot[1][0] * ms[0] + scale * rot[1][1] * ms[1]),
  ];
  return [scale, rot, shift];
}

/** numpy's sum of a float64 vector: pairwise, eight running sums in blocks of up to 128 (0 + the pairwise sum). */
export function pairwiseSum(a: ArrayLike<number>, lo = 0, n = a.length - lo): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[lo + i]!;
    return res;
  }
  if (n <= 128) {
    const r = [a[lo]!, a[lo + 1]!, a[lo + 2]!, a[lo + 3]!, a[lo + 4]!, a[lo + 5]!, a[lo + 6]!, a[lo + 7]!];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = r[j]! + a[lo + i + j]!;
    let res = r[0]! + r[1]! + (r[2]! + r[3]!) + (r[4]! + r[5]! + (r[6]! + r[7]!));
    for (; i < n; i++) res += a[lo + i]!;
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}

/** The same for a float32 vector, in float32. */
export function pairwiseSum32(a: ArrayLike<number>, lo = 0, n = a.length - lo): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f32(res + a[lo + i]!);
    return res;
  }
  if (n <= 128) {
    const r = new Float32Array(8);
    for (let j = 0; j < 8; j++) r[j] = a[lo + j]!;
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = r[j]! + a[lo + i + j]!;
    let res = f32(f32(f32(r[0]! + r[1]!) + f32(r[2]! + r[3]!)) + f32(f32(r[4]! + r[5]!) + f32(r[6]! + r[7]!)));
    for (; i < n; i++) res = f32(res + a[lo + i]!);
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return f32(pairwiseSum32(a, lo, n2) + pairwiseSum32(a, lo + n2, n - n2));
}

/** x.mean() of a float32 vector. */
function mean32(x: Float32Array): number {
  return f32(f32(0 + pairwiseSum32(x)) / x.length);
}

/** x.std() of a float32 vector (numpy's _var: the mean, the squared deviations, their mean, its root). */
function std32(x: Float32Array): number {
  const m = mean32(x);
  const d = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = f32(x[i]! - m);
    d[i] = v * v;
  }
  return f32(Math.sqrt(f32(f32(0 + pairwiseSum32(d)) / x.length)));
}

/** np.quantile(x, q) of float32 values (the default, linear): q, the index and the interpolation are float32, as
 * numpy keeps a float32 array's quantile float32. */
function quantile32(x: Float32Array, q: number): number {
  const s = Float32Array.from(x).sort();
  const n = s.length;
  const virtual = f32((n - 1) * f32(q));
  let prev = Math.floor(virtual);
  let next = prev + 1;
  if (virtual >= n - 1) prev = next = n - 1;
  if (virtual < 0) prev = next = 0;
  const gamma = f32(virtual - prev);
  const a = s[prev]!;
  const b = s[next]!;
  const diff = f32(b - a);
  return gamma >= 0.5 ? f32(b - f32(diff * f32(1 - gamma))) : f32(a + f32(diff * gamma));
}

/** The dot product of two float32 vectors, rounded to float32 (np.linalg.norm's BLAS sdot sums in its own order; this
 * is within a float32 step or two of it). */
function dot32(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return f32(s);
}

// --- the scene ----------------------------------------------------------------------------------------------------

const SMALL_W = 96;
const SMALL_H = 54;

/** Whether a frame shows the table camera, on any broadcast. The other shots (player cams, a wide shot of the stage,
 * a title card) show players' hands and the cards they hold: hidden information, never processed (D-005), and
 * nothing on the board changes while they are on.
 *
 * It learns the table camera from the footage itself: the parts of a 96 x 54 thumbnail that stay put while the board
 * changes (the broadcast's overlay, the mat's edges and print) and their colours. A frame is the table camera when
 * those parts match; a cut replaces them, play does not (the M0 final: table frames score 0.5 to 0.8, other shots
 * about 0). To start, a frame is taken as the table camera when the mat fills the table window as it does there (a
 * layout that knows its mat colour) or when at least five cards lie in it; if the view stays unrecognised but looks
 * like a table again for a few seconds (the camera itself moved), it learns again. The thumbnail is too coarse to
 * show any card. Its numbers are float32, as numpy's are. */
export class Scene {
  layout: Layout;
  corr: number;
  learnEvery: number;
  relearnAfter: number;
  n = 0;
  mean: Float32Array | null = null;
  var: Float32Array | null = null;
  lastLearn = -1e9;
  awaySince: number | null = null;
  /** Table-like frames in a row while away (checked every learnEvery). */
  looks = 0;
  lastLook = -1e9;

  constructor(layout: Layout, corr = 0.45, learnEvery = 2.0, relearnAfter = 20.0) {
    this.layout = layout;
    this.corr = corr;
    this.learnEvery = learnEvery;
    this.relearnAfter = relearnAfter;
  }

  static small(frame: RgbImage): Float32Array {
    return Float32Array.from(image.resize(frame, [SMALL_W, SMALL_H], 'box').data);
  }

  async tableLike(frame: RgbImage, count: () => Promise<number>): Promise<boolean> {
    const mat = this.layout.mat;
    if (mat !== null) {
      const [x0, y0, x1, y1] = layoutBox(this.layout, frame.width, frame.height);
      const a = image.resize(image.crop(frame, [x0, y0, x1, y1]), [160, 120], 'box').data;
      let n = 0;
      for (let o = 0; o < a.length; o += 3) {
        const d = Math.max(Math.abs(a[o]! - mat[0]), Math.abs(a[o + 1]! - mat[1]), Math.abs(a[o + 2]! - mat[2]));
        if (d < this.layout.mat_tol) n++;
      }
      return n / (a.length / 3) >= this.layout.mat_share;
    }
    return (await count()) >= 5;
  }

  /** How well the frame's still parts match the table camera's, or null when those parts have no pattern to match
   * (a plain mat and no overlay): then the mat share or the cards decide. */
  score(x: Float32Array): number | null {
    const mean = this.mean!;
    const v = this.var!;
    const px = x.length / 3;
    const std = new Float32Array(px);
    let steady = 0;
    for (let p = 0; p < px; p++) {
      std[p] = Math.max(f32(Math.sqrt(v[3 * p]!)), f32(Math.sqrt(v[3 * p + 1]!)), f32(Math.sqrt(v[3 * p + 2]!)));
      if (std[p]! < 12) steady++;
    }
    const cut = steady / px >= 0.1 ? null : quantile32(std, 0.3);
    const idx: number[] = [];
    for (let p = 0; p < px; p++) if (cut === null ? std[p]! < 12 : std[p]! <= cut) idx.push(p);
    const a = new Float32Array(idx.length * 3);
    const b = new Float32Array(idx.length * 3);
    idx.forEach((p, i) => {
      for (let c = 0; c < 3; c++) {
        a[3 * i + c] = x[3 * p + c]!;
        b[3 * i + c] = mean[3 * p + c]!;
      }
    });
    if (std32(b) < 8) return null;
    const am = mean32(a);
    const bm = mean32(b);
    const ab = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      a[i] = a[i]! - am;
      b[i] = b[i]! - bm;
      ab[i] = a[i]! * b[i]!;
    }
    const norms = f32(f32(Math.sqrt(dot32(a, a))) * f32(Math.sqrt(dot32(b, b))));
    return f32(f32(0 + pairwiseSum32(ab)) / f32(norms + f32(1e-6)));
  }

  learn(t: number, x: Float32Array): void {
    if (t - this.lastLearn < this.learnEvery) return;
    this.lastLearn = t;
    this.n += 1;
    if (this.mean === null || this.var === null) {
      this.mean = Float32Array.from(x);
      this.var = new Float32Array(x.length).fill(400.0); // unsure at first: nothing counts as still
      return;
    }
    const k = Math.max(1.0 / this.n, 0.02); // the mean of the first fifty, then a slow drift (light, overlay updates)
    const kf = f32(k); // numpy takes the Python float as a float32
    const keep = f32(1 - k);
    for (let i = 0; i < x.length; i++) {
      const d = f32(x[i]! - this.mean[i]!);
      this.mean[i] = this.mean[i]! + f32(kf * d);
      this.var[i] = keep * f32(this.var[i]! + f32(f32(kf * d) * d));
    }
  }

  async onTable(t: number, frame: RgbImage, count: () => Promise<number> = async () => 0): Promise<boolean> {
    const x = Scene.small(frame);
    const sc = this.n >= 5 ? this.score(x) : null;
    let ok: boolean;
    if (sc === null) {
      ok = await this.tableLike(frame, count);
    } else {
      ok = sc >= this.corr;
      if (!ok && this.awaySince !== null && t - this.awaySince > this.relearnAfter && t - this.lastLook >= this.learnEvery) {
        this.lastLook = t;
        this.looks = (await this.tableLike(frame, count)) ? this.looks + 1 : 0;
        if (this.looks >= 3) {
          // the table again, but not as it was learnt: the camera moved
          this.n = 0;
          this.mean = null;
          this.var = null;
          this.looks = 0;
          ok = true;
        }
      }
    }
    if (ok) {
      this.learn(t, x);
      this.awaySince = null;
    } else if (this.awaySince === null) {
      this.awaySince = t;
    }
    return ok;
  }
}

// --- the recogniser -----------------------------------------------------------------------------------------------

export interface RecognizerOptions {
  title?: string;
  minP?: number;
  sureP?: number;
  recheckS?: number;
  forgetS?: number;
  maxReads?: number;
  settleS?: number;
  /** Watch the whole table with the change gate. */
  gate?: boolean;
  fps?: number;
  gateP?: number;
  /** What finds the cards; without one, the bootstrap finder (isolated cards on the mat). */
  finder?: Finder | null;
  /** Turns an encoder's scores into how sure a read is; fitted per encoder. */
  temperature?: number;
}

/** Holds the gallery and the table's tracks; `step` takes one frame and returns the state and events. `finder(t,
 * image)` replaces the bootstrap finder, e.g. with the trained detector (detector_boxes). */
export class Recognizer {
  layout: Layout;
  rows: CatalogRow[];
  enc: Encoder;
  gallery: Pyramid;
  temperature: number;
  cards: string[];
  firstRow = new Map<string, number>();
  rowOf = new Map<string, CatalogRow>();
  title: string;
  minP: number;
  sureP: number;
  recheckS: number;
  forgetS: number;
  maxReads: number;
  settleS: number;
  tracks = new Map<string, Track>();
  nextId = 0;
  t0: number | null = null;
  mat: [number, number, number] | null = null;
  matT = -1e9;
  timing: Record<string, number> = {};
  gate: ChangeGate | null;
  gateP: number;
  finder: Finder | null;
  /** (read at, frame box) */
  pending: [number, Box4][] = [];
  /** Plays the gate found, drawn for a few seconds. */
  flashes: Flash[] = [];
  /** (t, card, x, y) of recent plays, for de-duplication. */
  plays: [number, string, number, number][] = [];
  // Named cards the finder lost (a hand over them, a card touching them): remembered for a while, so the same card
  // found again at the same spot is not played twice. A card leaves the table when the gate sees its spot change,
  // not when the finder loses it.
  ghosts: Ghost[] = [];
  // A legend never changes during a game (M0 section 5.4): once one is named on a side, that player keeps it.
  // ponytail: the first confident legend wins for the whole run; reset per game once games are detected
  legends = new Map<string, Legend>();
  /** Track id -> its box's extent, this frame. */
  boxesNow = new Map<string, Box4>();
  // Camera cuts: frames off the table camera are skipped and the board's clocks stop (`pause`). After a cut the view
  // may be framed differently, so tracks found again by name re-anchor the rest (`cut`).
  scene: Scene;
  gateSettings: GateSettings | null;
  lastT: number | null = null;
  away = false;
  cutAt: number | null = null;
  /** Track id -> its box before the cut. */
  anchorBase = new Map<string, CardBox>();
  /** (before, after) centres. */
  anchorPairs: [[number, number], [number, number]][] = [];
  /** Confirmed tracks the last frame matched. */
  prevSeen = new Set<string>();
  /** ... the last frame before a cut away. */
  beforeAway = new Set<string>();

  constructor(layout: Layout, rows: readonly CatalogRow[], encoder: Encoder, gallery: Pyramid, opts: RecognizerOptions = {}) {
    const { title = '', minP = 0.5, sureP = 0.85, recheckS = 8.0, forgetS = 4.0, maxReads = 12, settleS = 3.0 } = opts;
    const { gate = true, fps = 5.0, gateP = 0.7, finder = null, temperature = TEMPERATURE } = opts;
    this.layout = layout;
    this.rows = [...rows];
    this.enc = encoder;
    this.gallery = gallery;
    this.temperature = temperature; // turns an encoder's scores into how sure a read is; fitted per encoder
    this.cards = this.rows.map((r) => r.card_id);
    this.rows.forEach((r, i) => {
      if (!this.firstRow.has(r.card_id)) this.firstRow.set(r.card_id, i);
    });
    for (const r of this.rows) this.rowOf.set(r.printing_id, r);
    this.title = title || layout.title;
    this.minP = minP;
    this.sureP = sureP;
    this.recheckS = recheckS;
    this.forgetS = forgetS;
    this.maxReads = maxReads;
    this.settleS = settleS;
    this.gate = gate ? new ChangeGate(gateSettings({ fps, card_long_frac: layout.card_long_1080 / 1080 })) : null;
    this.gateP = gateP;
    this.finder = finder;
    this.scene = new Scene(layout);
    this.gateSettings = gate ? gateSettings({ fps, card_long_frac: layout.card_long_1080 / 1080 }) : null;
  }

  // --- finding ------------------------------------------------------------------------------------------------------

  async find(t: number, rgb: RgbImage): Promise<CardBox[]> {
    if (this.finder !== null) return this.finder(t, rgb);
    const { width: w, height: h } = rgb;
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const roi = image.crop(rgb, [x0, y0, x1, y1]);
    let mask: Mask | undefined;
    if (this.layout.mask === 'notmat') {
      if (this.mat === null || t - this.matT > 30) {
        // the light drifts over a match
        this.mat = matColour(roi, 4);
        this.matT = t;
      }
      mask = notmatMask(roi, this.mat, this.layout.mat_tol);
    }
    const boxes = findCards(roi, cardPx(this.layout, h), mask === undefined ? {} : { mask });
    return boxes.map((b) => ({ ...b, centre: [b.centre[0] + x0, b.centre[1] + y0] }));
  }

  // --- tracking -----------------------------------------------------------------------------------------------------

  /** Boxes to tracks one to one at the least total distance (Hungarian assignment). A box continues a track of about
   * its size within a third of a card, in view or out of sight for a while, so a card found again where it was keeps
   * its id and name; any other box starts a new track. */
  match(t: number, boxes: readonly CardBox[], w: number, h: number): Track[] {
    const tracks = [...this.tracks.values()];
    const pairs = new Map<number, Track>();
    if (tracks.length && boxes.length) {
      const cost = new Float64Array(boxes.length * tracks.length).fill(1e6);
      boxes.forEach((b, i) => {
        tracks.forEach((tr, j) => {
          const d = dist(tr.box.centre, b.centre) / b.long_px;
          if (d < 0.35 && Math.abs(tr.box.long_px / b.long_px - 1) < 0.25) {
            cost[i * tracks.length + j] = d + (t - tr.last > 1.0 ? 0.25 : 0.0); // the ones in view first
          }
        });
      });
      const [rows, cols] = linearSumAssignment({ rows: boxes.length, cols: tracks.length, data: cost });
      rows.forEach((i, n) => {
        const j = cols[n]!;
        if (cost[i * tracks.length + j]! < 1e6) pairs.set(i, tracks[j]!);
      });
    }
    const seen: Track[] = [];
    boxes.forEach((b, i) => {
      let tr = pairs.get(i);
      if (tr === undefined && this.onLegend(b)) return; // the detector's second outline of a legend's case, or of the die on it: not a card
      if (tr === undefined) {
        tr = new Track(`t${this.nextId}`, b, t, t, layoutSide(this.layout, b.centre[0], b.centre[1], w, h));
        this.nextId += 1;
        this.tracks.set(tr.id, tr);
      } else {
        tr.box = t - tr.last <= 1.0 ? smooth(tr.box, b) : b;
        tr.last = t;
        tr.hits = tr.hits + 1;
        if (!tr.pinned) tr.side = layoutSide(this.layout, tr.box.centre[0], tr.box.centre[1], w, h);
      }
      seen.push(tr);
    });
    return seen;
  }

  /** The box's centre lies well inside a named legend. Only dice and counters go on a legend; the champion and the
   * cards beside it lie about a card width away. */
  onLegend(box: CardBox): boolean {
    for (const o of this.tracks.values()) {
      if (o.pinned && o.kind === 'Legend' && dist(o.box.centre, box.centre) < 0.35 * o.box.long_px) return true;
    }
    return false;
  }

  /** The pinned legend of the track's side, when that is another track. */
  sideLegend(tr: Track): Track | null {
    for (const o of this.tracks.values()) if (o.pinned && o.kind === 'Legend' && o.side === tr.side && o !== tr) return o;
    return null;
  }

  /** The side already has its legend, and it is another card: one player, one legend (a rune column or a champion
   * read as a legend is not a second one). */
  otherLegend(side: string, card: string): boolean {
    const lg = this.legends.get(side);
    return lg !== undefined && this.rowOf.get(lg.printing_id)!.card_id !== card;
  }

  /** Something newer lies on this card: a card put on it or overlapping it. A covered card keeps its name (no re-reads
   * of a half-hidden face) and stays on the board until its spot clears. */
  covered(t: number, tr: Track): boolean {
    const [ax0, ay0, ax1, ay1] = this.boxesNow.get(tr.id) ?? aabb(tr.box);
    const area = Math.max(1.0, (ax1 - ax0) * (ay1 - ay0));
    for (const [oid, [bx0, by0, bx1, by1]] of this.boxesNow) {
      const o = this.tracks.get(oid);
      if (o === undefined || o === tr || o.first <= tr.first || t - o.last > 1.0) continue;
      const ix = Math.min(ax1, bx1) - Math.max(ax0, bx0);
      const iy = Math.min(ay1, by1) - Math.max(ay0, by0);
      if (ix > 0 && iy > 0 && ix * iy >= 0.25 * area) return true;
    }
    return false;
  }

  /** The part of card `a` that card `b` overlaps, on their outlines. */
  share(a: Track, b: Track): number {
    return overlapArea(quad(a.box), quad(b.box)) / Math.max(1.0, a.box.long_px * a.box.short_px);
  }

  /** Another named card in view, not this card outlined twice, that overlaps a quarter of this one or more: gear
   * tucked under a unit, a card put on another. Legends, battlefields and runes are not stacks. (The detector's
   * corner visibility would say which lies on top, but it is not reliable enough yet.) */
  stackedOn(t: number, tr: Track): Track | null {
    if (settled(tr.kind) || t - tr.last > 1.0) return null;
    let best = 0.25;
    let top: Track | null = null;
    for (const o of this.tracks.values()) {
      if (o === tr || !o.named || o.named === tr.named || settled(o.kind) || t - o.last > 1.0) continue;
      const sh = this.share(tr, o);
      if (sh >= best) {
        best = sh;
        top = o;
      }
    }
    return top;
  }

  /** An older track in view with the same name that covers half of this one: the same card outlined twice. */
  twin(t: number, tr: Track): boolean {
    if (!tr.named || tr.pinned || hiddenNow(t, tr)) return false;
    const n = Number(tr.id.slice(1));
    for (const o of this.tracks.values()) {
      if (o === tr || o.named !== tr.named) continue;
      const older = o.first < tr.first || (o.first === tr.first && Number(o.id.slice(1)) < n); // (o.first, int(o.id[1:])) < ...
      if (older && !hiddenNow(t, o) && this.share(tr, o) >= 0.5) return true;
    }
    return false;
  }

  /** The named cards under each card: a gear that overlaps a unit goes with the unit, and a card out of sight under a
   * newer one (`covered`) lies under it. Units side by side at a battlefield are no stack. */
  stacks(t: number): Map<string, Track[]> {
    const out = new Map<string, Track[]>();
    for (const u of this.tracks.values()) {
      if (!u.named || settled(u.kind)) continue;
      let host = u.kind === 'Gear' ? this.stackedOn(t, u) : null;
      if (host !== null && host.kind === 'Gear') host = null;
      if (host === null && t - u.last > 1.0 && this.covered(t, u)) {
        for (const o of this.tracks.values()) {
          // max(..., key=first): the first of the newest
          if (o !== u && o.named && o.named !== u.named && !settled(o.kind) && o.first > u.first && t - o.last <= 1.0 && this.share(u, o) >= 0.25) {
            if (host === null || o.first > host.first) host = o;
          }
        }
      }
      if (host !== null) {
        const list = out.get(host.id);
        if (list) list.push(u);
        else out.set(host.id, [u]);
      }
    }
    return out;
  }

  /** Another track of `tr`'s card that went out of sight around when `tr` appeared: the card moved. */
  vanished(t: number, tr: Track): Track | null {
    let best: Track | null = null;
    for (const o of this.tracks.values()) {
      if (o !== tr && o.named === tr.named && !o.pinned && t - o.last > 0.5 && o.last < tr.first + 0.5 && t - o.last < MOVE_S && !this.covered(t, o)) {
        if (best === null || o.last > best.last) best = o; // max(gone, key=last): the first of the latest
      }
    }
    return best;
  }

  // --- naming -------------------------------------------------------------------------------------------------------

  /** Two face-down looks in a row on a card never named. A hand resting on a named card or a blurred frame does not
   * hide it, and a card played under a hand is looked at again later. */
  faceDown(tr: Track): boolean {
    return tr.named === null && tr.down >= 2;
  }

  due(tr: Track, t: number): boolean {
    if (this.faceDown(tr)) return t - tr.lastRead > this.recheckS; // looked at now and then, never identified
    if (tr.named && (this.covered(t, tr) || this.stackedOn(t, tr) !== null)) return false; // its name is locked while something lies on it, or it lies on something
    const top = tr.top();
    if (!top.length || (top[0]![1] < this.sureP && tr.reads < this.maxReads)) return true;
    return t - tr.lastRead > this.recheckS;
  }

  /** Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first. All four
   * turns go in one batch: which way up a card lies is unknown (exhausted, opponent side). */
  async identify(crops: readonly RgbImage[]): Promise<Candidate[][]> {
    if (!crops.length) return [];
    const views: RgbImage[] = [];
    for (const c of crops) for (const r of ROTATIONS) views.push(r ? image.rotate(c, r, { expand: true }) : c);
    const emb = await this.enc.embed(views);
    const dim = emb.length / views.length;
    const turns = ROTATIONS.length;
    const out: Candidate[][] = [];
    crops.forEach((c, n) => {
      const level = { data: this.gallery.level(Math.max(c.width, c.height)), rows: this.gallery.rows, dim: this.gallery.dim };
      const sims = bestSimilarities({ data: emb.subarray(n * turns * dim, (n + 1) * turns * dim), rows: turns, dim }, level);
      const scores = new Map<string, [number, number]>();
      for (const i of argsortDescending(sims).subarray(0, 60)) {
        const card = this.cards[i]!;
        if (!scores.has(card)) scores.set(card, [sims[i]!, i]);
      }
      const vals = [...scores.values()].map((v) => v[0]);
      const top = Math.max(...vals);
      const p = vals.map((v) => Math.exp((v - top) / this.temperature));
      const sum = 0 + pairwiseSum(p);
      out.push([...scores].map(([card, [sc, i]], j): Candidate => [card, p[j]! / sum, sc, i]));
    });
    return out;
  }

  async read(t: number, frame: RgbImage, todo: readonly Track[]): Promise<void> {
    const crops: RgbImage[] = [];
    const owners: Track[] = [];
    for (const tr of todo) {
      tr.lastRead = t;
      if (tr.box.back) {
        tr.down += 1; // the detector saw a card back
        continue;
      }
      const c = cardCrop(frame, tr.box);
      if (detail(c) < FACE_DOWN_DETAIL) {
        tr.down += 1;
        continue;
      }
      tr.down = 0;
      crops.push(c);
      owners.push(tr);
    }
    const found = await this.identify(crops);
    owners.forEach((tr, n) => {
      tr.reads += 1;
      for (const [card, pc, sc, i] of found[n]!) {
        tr.prob.set(card, (tr.prob.get(card) ?? 0.0) + pc);
        if (sc > (tr.bestRow.get(card) ?? [-1.0, 0])[0]) tr.bestRow.set(card, [sc, i]);
      }
    });
  }

  /** The change gate on the whole table; a settled change is read a moment later, like a crop. */
  async watch(t: number, frame: RgbImage): Promise<RecognizerEvent[]> {
    if (this.gate === null) return [];
    const { width: w, height: h } = frame;
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const vw = this.gate.s.width;
    const vh = viewHeight(this.layout.table, vw);
    const view = image.resize(image.crop(frame, [x0, y0, x1, y1]), [vw, vh], 'bilinear');
    const events: RecognizerEvent[] = [];
    for (const ev of this.gate.feed(t, view)) {
      const fb = viewToFrame(ev.box, this.layout.table, vw, w, h);
      if (ev.kind === 'appeared' || ev.kind === 'changed') this.pending.push([t + 0.4, fb]);
      else if (ev.kind === 'disappeared') events.push(...this.left(t, fb));
    }
    const due = this.pending.filter(([when]) => when <= t).map(([, b]) => b);
    this.pending = this.pending.filter(([when]) => when > t);
    const cardLong = cardPx(this.layout, h);
    for (const [bx0, by0, bx1, by1] of due) {
      const gx = (bx1 - bx0) * 0.08;
      const gy = (by1 - by0) * 0.08;
      const box: Box4 = [Math.max(0, bx0 - gx), Math.max(0, by0 - gy), Math.min(w, bx1 + gx), Math.min(h, by1 + gy)];
      let region = image.crop(frame, [pyRound(box[0]), pyRound(box[1]), pyRound(box[2]), pyRound(box[3])]);
      if (Math.min(region.width, region.height) < (0.45 * cardLong * 63) / 88 || Math.max(region.width, region.height) > 2.2 * cardLong) {
        continue; // a die or counter, or a whole area at once: not one card
      }
      if (region.width > region.height * 1.15) region = image.rotate(region, 90, { expand: true }); // cards stand portrait
      if (detail(region) < FACE_DOWN_DETAIL) continue; // a face-down card: never identified
      // ponytail: the whole changed region is read as one card; the visible-band matcher (stacks.py) for cards put on stacks
      const cands = (await this.identify([region]))[0]!;
      const [card, p, , i] = cands[0]!;
      const cx = (box[0] + box[2]) / 2;
      const cy = (box[1] + box[3]) / 2;
      const r = this.rows[i]!;
      if (p < this.gateP || settled(r.type ?? '') || this.recentlyPlayed(t, card, cx, cy)) continue;
      const side = layoutSide(this.layout, cx, cy, w, h);
      this.plays.push([t, card, cx, cy]);
      const q: [number, number][] = [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
      ];
      const guesses = cands.slice(0, 3).map(([c, pc, , j]): Guess => ({ printing_id: this.rows[j]!.printing_id, card_id: c, name: this.rows[j]!.name, p: pyRound(pc, 3) }));
      const flash: Flash = {
        id: `g${this.plays.length}`,
        quad: q.map(([x, y]): [number, number] => [pyRound(x, 1), pyRound(y, 1)]),
        side,
        state: 'named',
        printing_id: r.printing_id,
        name: r.name,
        confidence: pyRound(p, 3),
        guesses,
        since: pyRound(t, 2),
        until: t + 5,
        kind: 'card',
        hidden: false,
      };
      this.flashes.push(flash);
      events.push({ t: pyRound(t, 2), kind: 'played', text: `${r.name} played`, printing_id: r.printing_id, track: flash.id, side });
    }
    return events;
  }

  /** A named card whose spot the gate saw cleared (a ghost, or a track the finder still holds). */
  left(t: number, box: Box4): RecognizerEvent[] {
    const [x0, y0, x1, y1] = box;
    const gx = (x1 - x0) * 0.1;
    const gy = (y1 - y0) * 0.1;
    const inside = (x: number, y: number): boolean => x0 - gx <= x && x <= x1 + gx && y0 - gy <= y && y <= y1 + gy;
    for (const tr of [...this.tracks.values()].sort((a, b) => b.last - a.last)) {
      if (tr.named && !tr.pinned && t - tr.last > this.forgetS && inside(tr.box.centre[0], tr.box.centre[1])) {
        this.tracks.delete(tr.id);
        const g = this.label(tr)[2];
        return [this.event(t, 'left', `${g.length ? g[0]!.name : tr.named} left the table`, tr, g.length ? g[0]!.printing_id : null)];
      }
    }
    for (const gh of [...this.ghosts].sort((a, b) => b.t - a.t)) {
      if (inside(gh.x, gh.y)) {
        this.ghosts.splice(this.ghosts.indexOf(gh), 1);
        return [{ t: pyRound(t, 2), kind: 'left', text: `${gh.name} left the table`, printing_id: gh.printing_id, track: gh.id, side: gh.side }];
      }
    }
    return [];
  }

  /** The same card announced nearby a moment ago, by the gate or by another track. */
  recentlyPlayed(t: number, card: string, x: number, y: number, window = 12.0, skip = ''): boolean {
    const reach = 1.5 * this.layout.card_long_1080;
    this.plays = this.plays.filter((pl) => t - pl[0] < window);
    if (this.plays.some(([, c, px, py]) => c === card && dist([x, y], [px, py]) < reach)) return true;
    for (const tr of this.tracks.values()) {
      if (tr.id !== skip && tr.named === card && t - tr.first < window && dist([x, y], tr.box.centre) < reach) return true;
    }
    return false;
  }

  // --- one frame ----------------------------------------------------------------------------------------------------

  /** Off the table camera: the board's clocks stop, so nothing is forgotten or re-read for the time away. */
  pause(dt: number): void {
    for (const tr of this.tracks.values()) {
      tr.last += dt;
      tr.lastRead += dt;
    }
    for (const gh of this.ghosts) gh.t += dt;
    this.plays = this.plays.map(([pt, c, x, y]): [number, string, number, number] => [pt + dt, c, x, y]);
    this.pending = [];
    this.flashes = [];
  }

  /** The view is framed anew: cards found again by name re-anchor the rest (`reanchor`), the gate starts over, and
   * the cards first seen now were on the table already, not played. */
  cut(t: number): void {
    this.cutAt = t;
    this.anchorBase = new Map([...this.tracks].map(([k, tr]) => [k, tr.box]));
    this.anchorPairs = [];
    if (this.gateSettings !== null) this.gate = new ChangeGate(this.gateSettings);
  }

  /** Move the tracks not seen since the cut as the view moved (scale, turn and shift fitted to the cards found again
   * by name); a new track lying where a moved one now is, is that card and takes its id. */
  reanchor(): void {
    if (this.anchorPairs.length < 2 || this.cutAt === null) return;
    const cutAt = this.cutAt;
    const [scale, rot, shift] = similarity(this.anchorPairs.map(([a]) => a), this.anchorPairs.map(([, b]) => b));
    if (!(scale >= 0.5 && scale <= 2.0)) return;
    const turn = Math.atan2(rot[1][0], rot[0][0]) * RAD_TO_DEG;
    const old = [...this.tracks.values()].filter((o) => o.first < cutAt && o.last < cutAt && this.anchorBase.has(o.id));
    for (const o of old) {
      const b = this.anchorBase.get(o.id)!;
      const [bx, by] = b.centre;
      const c: [number, number] = [scale * rot[0][0] * bx + scale * rot[0][1] * by + shift[0], scale * rot[1][0] * bx + scale * rot[1][1] * by + shift[1]];
      o.box = { centre: c, long_px: b.long_px * scale, short_px: b.short_px * scale, angle_deg: pyMod(b.angle_deg + turn, 180), fill: b.fill };
    }
    for (const n of [...this.tracks.values()].filter((n) => n.first >= cutAt)) {
      const near = old.filter(
        (o) =>
          dist(o.box.centre, n.box.centre) < 0.35 * n.box.long_px &&
          Math.abs(o.box.long_px / n.box.long_px - 1) < 0.25 &&
          (o.named === null || n.named === null || n.named === o.named),
      );
      if (!near.length) continue;
      let o = near[0]!; // min(near, key=distance): the first of the nearest
      for (const m of near) if (dist(m.box.centre, n.box.centre) < dist(o.box.centre, n.box.centre)) o = m;
      o.box = n.box;
      o.last = n.last;
      o.hits = o.hits + n.hits;
      o.side = n.side;
      if (o.named === null && n.reads) {
        o.reads = n.reads;
        o.prob = n.prob;
        o.bestRow = n.bestRow;
        o.down = n.down;
        o.named = n.named;
        o.kind = n.kind;
      }
      this.tracks.delete(n.id);
      old.splice(old.indexOf(o), 1);
    }
  }

  /** One frame: the state the overlay draws and the events it saw. `budget` caps the crops read this frame. Steps
   * must not overlap: await one before starting the next. */
  async step(t: number, frame: RgbImage, budget = 10): Promise<[RecognizerState, RecognizerEvent[]]> {
    const tic = performance.now();
    if (this.t0 === null) this.t0 = t;
    const { width: w, height: h } = frame;
    const dt = this.lastT !== null ? t - this.lastT : 0.0;
    this.lastT = t;
    let boxes: CardBox[] | null = null;
    // the scene asks only while it learns a broadcast with no known mat colour
    const count = async (): Promise<number> => {
      boxes = await this.find(t, frame);
      return boxes.length;
    };
    if (!(await this.scene.onTable(t, frame, count))) {
      if (!this.away) {
        this.away = true;
        this.beforeAway = new Set(this.prevSeen);
      }
      this.pause(dt);
      const state = this.state(t, w, h);
      for (const tr of state.tracks) tr.hidden = true; // the video is not the table: list the board, draw nothing on it
      state.status = 'away';
      state.message = 'the table camera is off; nothing is looked at until it is back';
      return [state, []];
    }
    const back = this.away;
    this.away = false;
    const found: CardBox[] = boxes ?? (await this.find(t, frame));
    const tf = performance.now();
    const seen = this.match(t, found, w, h);
    const before = back ? this.beforeAway : this.prevSeen;
    const ids = new Set(seen.map((tr) => tr.id));
    const gone = [...before].filter((id) => !ids.has(id)).length;
    if (before.size >= 6 && gone >= 0.7 * before.size && found.length >= 3) {
      this.cut(t); // most of the board moved at once: the view is framed anew
    } else if (back && this.gateSettings !== null) {
      this.gate = new ChangeGate(this.gateSettings); // the same view again: only the gate starts over
    }
    this.prevSeen = new Set(seen.filter((tr) => tr.hits >= 2).map((tr) => tr.id));
    this.boxesNow = new Map([...this.tracks].map(([k, tr]) => [k, aabb(tr.box)]));
    // New and uncertain cards first, then the oldest re-checks; a budget keeps each frame in time.
    // A box seen once may be the detector's slip (between two cards): only tracks seen twice are read.
    const todo = seen
      .filter((tr) => tr.hits >= 2 && this.due(tr, t))
      .sort((a, b) => Number(a.reads > 0) - Number(b.reads > 0) || a.lastRead - b.lastRead)
      .slice(0, budget);
    await this.read(t, frame, todo);
    const tr_ = performance.now();
    const events = [...this.announce(t), ...(await this.watch(t, frame))];
    this.flashes = this.flashes.filter((f) => f.until > t);
    this.timing = { find_ms: tf - tic, read_ms: tr_ - tf, gate_ms: performance.now() - tr_, reads: todo.length, boxes: found.length };
    return [this.state(t, w, h), events];
  }

  /** The track's state, confidence and up to three guesses (each card's best-matching printing). */
  label(tr: Track): [string, number, Guess[]] {
    if (this.faceDown(tr)) return ['facedown', 0.0, []];
    let top = tr.top();
    if (!top.length) return ['new', 0.0, []];
    if (tr.pinned) top = [...top].sort((a, b) => Number(a[0] !== tr.named) - Number(b[0] !== tr.named)); // a legend or battlefield keeps its name all game: re-reads move only its confidence
    const guesses: Guess[] = [];
    for (const [c, p] of top.slice(0, 3)) {
      const r = this.rows[(tr.bestRow.get(c) ?? [0.0, this.firstRow.get(c)!])[1]]!;
      guesses.push({ printing_id: r.printing_id, card_id: c, name: r.name, p: pyRound(p, 3) });
    }
    const p0 = top[0]![1];
    const legend = this.rows[this.firstRow.get(top[0]![0])!]!.type === 'Legend';
    if (!tr.pinned && legend && this.sideLegend(tr) !== null) return ['unsure', p0, guesses]; // one player, one legend: another outline of it, or a card misread as one
    let named = tr.pinned || p0 >= this.sureP || (p0 >= this.minP && tr.reads >= 2);
    if (!named && tr.reads >= 4 && p0 >= 0.3 && legend && !this.otherLegend(tr.side, top[0]![0])) {
      named = true; // a legend, read the same way four times: one a player, and its frame is like no other card's
    }
    return [named ? 'named' : 'unsure', p0, guesses];
  }

  /** 'played' when a card is first named, 'moved' when a named card that just vanished is named again elsewhere (it
   * keeps its first id). A card out of sight keeps its track: an unnamed one `forgetS`, a named one `KEEP_S` or as
   * long as something lies on it, a legend or battlefield all game. Then a named card becomes a ghost (see `ghosts`).
   * Runes, legends and battlefields are never announced. */
  announce(t: number): RecognizerEvent[] {
    const events: RecognizerEvent[] = [];
    this.ghosts = this.ghosts.filter((gh) => t - gh.t < 60);
    for (const tr of [...this.tracks.values()]) {
      if (!this.tracks.has(tr.id)) continue; // merged into the track it moved from
      const gone = t - tr.last;
      const limit = tr.named ? KEEP_S : tr.hits >= 2 ? this.forgetS : 1.0;
      if (!tr.pinned && gone > limit && !(tr.named && this.covered(t, tr))) {
        this.tracks.delete(tr.id);
        if (tr.named) {
          const g = this.label(tr)[2];
          this.ghosts.push({
            t: tr.last,
            card: tr.named,
            x: tr.box.centre[0],
            y: tr.box.centre[1],
            name: g.length ? g[0]!.name : tr.named,
            printing_id: g.length ? g[0]!.printing_id : null,
            side: tr.side,
            id: tr.id,
          });
        }
        continue;
      }
      const [state, , g] = this.label(tr);
      if (state === 'named' && tr.named !== g[0]!.card_id) {
        const changed = tr.named !== null;
        const named = g[0]!.card_id;
        tr.named = named;
        tr.kind = this.rowOf.get(g[0]!.printing_id)?.type ?? '';
        const cutAt = this.cutAt;
        const afterCut = cutAt !== null && !changed && tr.first - cutAt >= 0 && tr.first - cutAt < 30;
        if (afterCut) {
          // a card on the table before the cut, found again by name in the new view
          const olds = [...this.tracks.values()].filter((o) => o !== tr && o.named === named && o.first < cutAt && o.last < cutAt);
          if (olds.length === 1) {
            const o = olds[0]!;
            this.anchorPairs.push([(this.anchorBase.get(o.id) ?? o.box).centre, tr.box.centre]);
            o.box = tr.box;
            o.last = tr.last;
            o.hits = o.hits + tr.hits;
            o.side = tr.side;
            this.tracks.delete(tr.id); // it keeps its first id
            this.reanchor();
            continue;
          }
        }
        if (STATIC.includes(tr.kind) && !(tr.kind === 'Legend' && (this.otherLegend(tr.side, named) || this.sideLegend(tr) !== null))) {
          tr.pinned = true; // set up before the game: nothing to announce, and it stays put
          if (tr.kind === 'Legend' && !this.legends.has(tr.side)) {
            // the side's legend, however sure its reads are under the dice
            this.legends.set(tr.side, { printing_id: g[0]!.printing_id, name: g[0]!.name });
          }
          for (const o of [...this.tracks.values()].filter((o) => !o.pinned && this.onLegend(o.box))) {
            this.tracks.delete(o.id); // a second outline of a legend, read as a card of its own
          }
        }
        if (settled(tr.kind)) continue;
        const was = !changed && !afterCut ? this.vanished(t, tr) : null;
        if (was !== null) {
          const far = dist(was.box.centre, tr.box.centre) > 0.6 * tr.box.long_px;
          was.box = tr.box;
          was.last = tr.last;
          was.hits = was.hits + tr.hits;
          was.side = tr.side;
          this.tracks.delete(tr.id); // the same card: it keeps its first id and what was read of it
          if (far) events.push(this.event(t, 'moved', `${g[0]!.name} moved`, was, g[0]!.printing_id));
          continue;
        }
        const reach = 1.5 * this.layout.card_long_1080;
        const back = this.ghosts.filter((gh) => gh.card === named && dist([gh.x, gh.y], tr.box.centre) < reach);
        if (back.length && !changed) {
          this.ghosts.splice(this.ghosts.indexOf(back[0]!), 1); // the same card, found again
          continue;
        }
        if (tr.first - (this.t0 ?? 0.0) < this.settleS && !changed) continue; // on the table when we tuned in, not played now
        if (afterCut && tr.first - cutAt < this.settleS + 2) continue; // on the table when the view changed
        if (changed) {
          events.push(this.event(t, 'changed', `${g[0]!.name} (read again)`, tr, g[0]!.printing_id));
        } else if (!this.recentlyPlayed(t, named, tr.box.centre[0], tr.box.centre[1], 12.0, tr.id)) {
          this.plays.push([t, named, tr.box.centre[0], tr.box.centre[1]]);
          events.push(this.event(t, 'played', `${g[0]!.name} played`, tr, g[0]!.printing_id));
        } else {
          this.plays.push([t, named, tr.box.centre[0], tr.box.centre[1]]);
        }
      }
    }
    return events;
  }

  event(t: number, kind: string, text: string, tr: Track, pid: string | null): RecognizerEvent {
    return { t: pyRound(t, 2), kind, text, printing_id: pid, track: tr.id, side: tr.side };
  }

  state(t: number, w: number, h: number): RecognizerState {
    const tracks: StateTrack[] = [];
    const under = this.stacks(t);
    for (const tr of this.tracks.values()) {
      if (tr.hits < 2) continue; // seen once: maybe the detector's slip (a box between two cards), not shown yet
      const hidden = t - tr.last > 1.0 && !tr.pinned;
      if (hidden && !tr.named) continue;
      const [state, p, g] = this.label(tr);
      const lg = this.sideLegend(tr);
      if (lg !== null && !tr.pinned && g.length && g[0]!.card_id === lg.named) continue; // another outline of the side's legend (its case, the die on it): not a card
      if (this.twin(t, tr)) continue; // the same card outlined again (a sleeve's or a toploader's edge): drawn once
      const top = g.length && state === 'named' ? g[0]! : null;
      // hidden: out of sight (under a hand or another card) but still on the board, so listed, not drawn
      const unders: Under[] = [];
      for (const u of under.get(tr.id) ?? []) {
        const gu = this.label(u)[2];
        if (gu.length) unders.push({ id: u.id, name: gu[0]!.name, printing_id: gu[0]!.printing_id });
      }
      tracks.push({
        id: tr.id,
        quad: quad(tr.box),
        side: tr.side,
        state,
        printing_id: top ? top.printing_id : null,
        name: top ? top.name : '',
        confidence: pyRound(p, 3),
        guesses: state !== 'facedown' ? g : [],
        since: pyRound(tr.first, 2),
        kind: KINDS.get(tr.kind) ?? 'card',
        hidden,
        under: unders,
      });
    }
    for (const f of this.flashes) {
      const { until: _until, ...shown } = f;
      tracks.push(shown);
    }
    for (const tr of tracks) {
      const pid = tr.printing_id;
      if (tr.state === 'named' && pid && !this.legends.has(tr.side) && tr.confidence >= this.sureP) {
        const r = this.rowOf.get(pid);
        if (r !== undefined && r.type === 'Legend') this.legends.set(tr.side, { printing_id: pid, name: r.name });
      }
    }
    return {
      t: pyRound(t, 2),
      status: 'live',
      message: '',
      title: this.title,
      frame: { width: w, height: h },
      players: layoutSides(this.layout).map((s, k) => ({ side: s, label: `Player ${k + 1}`, legend: this.legends.get(s) ?? null })),
      layout: { name: this.layout.name, table: [...this.layout.table] },
      tracks,
    };
  }
}

/** The gallery levels live/__main__.py embeds for a layout: a card's long side at 1080p times 0.8, 0.9 and 1, each
 * rounded to 10 px (the Pyramid the Recognizer is given should hold these). */
export function galleryScales(layout: Layout): number[] {
  const px = cardPx(layout, 1080);
  return [...new Set([0.8, 0.9, 1.0].map((f) => Math.trunc(pyRound((px * f) / 10) * 10)))].sort((a, b) => a - b);
}
