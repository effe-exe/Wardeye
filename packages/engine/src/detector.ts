// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The card detector, from a frame to the tracker's boxes: a port of ml/rifteye_ml/detect/model.py (Detector.detect,
// detect_tiles, merge_tiles), RF-DETR's keypoint PostProcess (rfdetr 1.11.0, models/postprocess.py, Apache-2.0,
// Copyright (c) 2025 Roboflow) and live/pipeline.py's detector_boxes, drop_nested, drop_straddlers and quad.
//
// The net itself is not here: a Detector is given a runner that takes tiles (RGB / 255, NCHW, float32, 576 x 576,
// as the ONNX graph takes them) and returns the graph's three outputs, so any ONNX Runtime build (ort.ts), or a
// test, can run it. Only the camera window is ever cut out of a frame and looked at (D-005), and a card_back box
// is marked as one, so it is never identified.
//
// The head's outputs are float32, and PyTorch post-processes them in float32: so does this, rounding to float32
// wherever a PyTorch tensor would be. Its exp and log are the C library's or SLEEF's; these are the browser's,
// rounded to float32, so a score or corner can be a float32 unit in the last place away from Python's. The tile
// merge and the tracker's boxes are float64 in Python and are the same bits here.

import { canonicalQuad, dist, mean, norm2, npRound, overlapArea, points, quadIou, tileOrigins, type Point } from './geometry';
import { crop, resize } from './image';
import { box as windowBox, cardPx as layoutCardPx } from './layouts';
import { pyMod, pyRound } from './pynum';
import type { CardBox, Detection, Finder, Layout, RgbImage } from './types';

/** detect/export.py: RF-DETR keypoint's input side. */
export const TILE = 576;
/** detect/export.py: the head's classes, by logit. */
export const CLASSES = ['card', 'card_back'] as const;
/** live/__main__.py --det-score: the detector's confidence below which the tracker drops a box. */
export const DET_SCORE = 0.4;

/** The graph's three outputs for a batch of tiles, each float32 in C order: pred_logits [n, 100, 2], pred_boxes
 * [n, 100, 4] (centre and size, fractions of the tile), pred_keypoints [n, 100, 8, 8] (per class four corners of
 * x, y, found and visible logits, three precision numbers and a class logit). */
export interface HeadOutputs {
  pred_logits: Float32Array;
  pred_boxes: Float32Array;
  pred_keypoints: Float32Array;
}

/** Runs the detector's graph on `n` tiles: RGB / 255, NCHW, float32, n x 3 x 576 x 576. */
export type TileRunner = (tiles: Float32Array, n: number) => Promise<HeadOutputs>;

/** RF-DETR's PostProcess as the detector's checkpoint builds it: the best 100 of the 200 query-class pairs, four
 * keypoints a class, and the scores fused with the corners' uncertainty (trace_alpha 0.2). */
export interface HeadConfig {
  numSelect: number;
  keypointsPerClass: readonly number[];
  traceAlpha: number;
}

export const HEAD: Readonly<HeadConfig> = Object.freeze({ numSelect: 100, keypointsPerClass: Object.freeze([4, 4]), traceAlpha: 0.2 });

/** A card of the merged frame: types.Detection, plus merge_tiles' `truncated` (it touched a tile's edge). Python's
 * detect() gives no box; `box` is its quad's bounding box. */
export interface FrameDetection extends Detection {
  truncated: boolean;
}

// ---- the tiles of a frame (Detector.detect)

/** A frame's camera window, scaled so its cards are near `target` px and cut into 576 px tiles. */
export interface Tiling {
  tiles: RgbImage[];
  /** Each tile's top-left in the scaled window. */
  origins: [number, number][];
  /** The scaled window's size, sw x sh. */
  size: [number, number];
  scale: number;
  /** The window's top-left in the frame. */
  offset: [number, number];
}

/** The tiles Detector.detect cuts: the window (rounded to whole px) cropped, resized with Pillow's bicubic filter so
 * cards come out near `target` px long, and cut into overlapping tiles, black past the window's edge. */
export function cutTiles(frame: RgbImage, window: readonly number[], cardPx: number, target = 70, overlap = 0.2): Tiling {
  const [wx0, wy0, wx1, wy1] = window.slice(0, 4).map((v) => pyRound(v)) as [number, number, number, number];
  const scale = target / cardPx;
  const win = crop(frame, [wx0, wy0, wx1, wy1]);
  const sw = Math.max(1, pyRound((wx1 - wx0) * scale));
  const sh = Math.max(1, pyRound((wy1 - wy0) * scale));
  const small = resize(win, [sw, sh], 'bicubic');
  const origins: [number, number][] = [];
  for (const ty of tileOrigins(sh, TILE, overlap)) for (const tx of tileOrigins(sw, TILE, overlap)) origins.push([tx, ty]);
  const tiles = origins.map(([tx, ty]) => crop(small, [tx, ty, tx + TILE, ty + TILE]));
  return { tiles, origins, size: [sw, sh], scale, offset: [wx0, wy0] };
}

/** A byte as the graph's input: float32(p) / 255 in float32, as torchvision's to_tensor gives it. */
const LEVELS = Float32Array.from({ length: 256 }, (_, p) => p / 255);

/** Tiles as the graph's input, RGB / 255, NCHW, float32; `n` slots, those past the tiles left black (zero). */
export function tileBatch(tiles: readonly RgbImage[], n = tiles.length): Float32Array {
  if (n < tiles.length) throw new Error(`${tiles.length} tiles do not fit a batch of ${n}`);
  const plane = TILE * TILE;
  const out = new Float32Array(n * 3 * plane);
  tiles.forEach((t, i) => {
    if (t.width !== TILE || t.height !== TILE) throw new Error(`a tile is ${TILE} x ${TILE} px, not ${t.width} x ${t.height}`);
    const d = t.data;
    const r = i * 3 * plane;
    const g = r + plane;
    const b = g + plane;
    for (let p = 0, s = 0; p < plane; p++, s += 3) {
      out[r + p] = LEVELS[d[s]!]!;
      out[g + p] = LEVELS[d[s + 1]!]!;
      out[b + p] = LEVELS[d[s + 2]!]!;
    }
  });
  return out;
}

// ---- RF-DETR's keypoint PostProcess, in float32 as PyTorch computes it

const f32 = Math.fround;
const CLAMP = f32(1e-12);
const BELOW_ONE = f32(1 - 2 ** -24); // torch.nextafter(1, 0) in float32

/** torch.sigmoid on float32: 1 / (1 + exp(-x)), each step rounded to float32. */
function sigmoid32(x: number): number {
  return f32(1 / f32(1 + f32(Math.exp(-x))));
}

function log32(x: number): number {
  return f32(Math.log(x));
}

/** torch.logsumexp of float32 values: max + log(sum(exp(x - max))). */
function logsumexp32(xs: readonly number[]): number {
  let m = -Infinity;
  for (const x of xs) if (x > m || Number.isNaN(x)) m = x;
  let s = 0;
  for (const x of xs) s = f32(s + f32(Math.exp(f32(x - m))));
  return f32(log32(s) + (Math.abs(m) === Infinity ? 0 : m));
}

/** PostProcess._keypoint_log_mean_trace for one detection: the log of the findability-weighted mean trace of its
 * corners' covariance. `kp` holds the class's corners, `d` numbers each. */
function logMeanTrace(kp: Float32Array, at: number, corners: number, d: number): number {
  const trace: number[] = [];
  const wFind: number[] = [];
  for (let j = 0; j < corners; j++) {
    const o = at + j * d;
    const logT1 = f32(-2 * kp[o + 4]!);
    const logT2 = f32(-2 * kp[o + 6]!);
    const logT3 = f32(f32(f32(2 * log32(Math.max(Math.abs(kp[o + 5]!), CLAMP))) + logT1) + logT2);
    trace.push(logsumexp32([logT1, logT2, logT3]));
    wFind.push(log32(Math.max(sigmoid32(kp[o + 2]!), CLAMP)));
  }
  return f32(logsumexp32(trace.map((t, j) => f32(t + wFind[j]!))) - logsumexp32(wFind));
}

/** One tile's detections, in tile px, as Detector.detect_tiles makes them of the head's outputs: PostProcess run
 * twice (the second time with the visible logit in the found logit's place), then the scores above `threshold`.
 * `logits`, `boxes` and `keypoints` are this tile's parts of the outputs; `size` is the tile's width and height. */
export function decodeTile(
  logits: Float32Array,
  boxes: Float32Array,
  keypoints: Float32Array,
  threshold = 0.3,
  size: readonly [number, number] = [TILE, TILE],
  head: HeadConfig = HEAD,
): Detection[] {
  const nClasses = head.keypointsPerClass.length;
  const nQueries = logits.length / nClasses;
  const maxKp = Math.max(0, ...head.keypointsPerClass);
  const d = keypoints.length / (nQueries * nClasses * maxKp);
  if (!Number.isInteger(nQueries) || boxes.length !== nQueries * 4 || !Number.isInteger(d) || d < 7) {
    throw new Error(`outputs of ${logits.length}, ${boxes.length} and ${keypoints.length} values do not fit ${nClasses} classes`);
  }
  const [w, h] = size;

  // _select_topk: sigmoid, sorted descending, ties to the lower flattened index
  const prob = Array.from(logits, sigmoid32);
  const order = prob.map((_, i) => i).sort((i, j) => (prob[i]! > prob[j]! ? -1 : prob[i]! < prob[j]! ? 1 : i - j));
  const picked = order.slice(0, Math.min(head.numSelect, order.length));
  const min = f32(threshold); // torch compares a float32 tensor with the threshold in float32

  const out: Detection[] = [];
  for (const k of picked) {
    const q = Math.floor(k / nClasses);
    const label = k % nClasses;
    let score = prob[k]!;
    const corners = head.keypointsPerClass[label]!;
    const at = (q * nClasses * maxKp + label * maxKp) * d;
    if (corners > 0 && head.traceAlpha > 0) {
      const fused = f32(log32(score) - f32(f32(head.traceAlpha) * logMeanTrace(keypoints, at, corners, d)));
      score = Math.min(sigmoid32(fused), BELOW_ONE);
    }
    if (!(score > min) || label >= CLASSES.length) continue;
    // box_cxcywh_to_xyxy, scaled to the tile and clamped to it
    const [cx, cy, bw, bh] = [0, 1, 2, 3].map((i) => boxes[q * 4 + i]!) as [number, number, number, number];
    const hw = f32(0.5 * Math.max(bw, 0));
    const hh = f32(0.5 * Math.max(bh, 0));
    const px = (v: number, s: number) => Math.min(Math.max(f32(v * s), 0), s);
    const box: [number, number, number, number] = [px(f32(cx - hw), w), px(f32(cy - hh), h), px(f32(cx + hw), w), px(f32(cy + hh), h)];
    const quad: [number, number][] = [];
    const found: number[] = [];
    const visible: number[] = [];
    for (let j = 0; j < corners; j++) {
      const o = at + j * d;
      quad.push([f32(keypoints[o]! * w), f32(keypoints[o + 1]! * h)]);
      found.push(sigmoid32(keypoints[o + 2]!));
      visible.push(sigmoid32(keypoints[o + 3]!));
    }
    out.push({ cls: CLASSES[label]!, score, box, quad, found, visible });
  }
  return out;
}

/** The b-th tile's parts of a batch's outputs. */
export function tileOutputs(out: HeadOutputs, b: number, n: number): HeadOutputs {
  const part = (a: Float32Array) => {
    const per = a.length / n;
    return a.subarray(b * per, (b + 1) * per);
  };
  return { pred_logits: part(out.pred_logits), pred_boxes: part(out.pred_boxes), pred_keypoints: part(out.pred_keypoints) };
}

// ---- merge_tiles

interface Kept {
  cls: Detection['cls'];
  score: number;
  quad: Point[];
  found: number[];
  visible: number[];
  truncated: boolean;
  tile: [number, number];
}

/** One list of detections for the window, in frame px (merge_tiles).
 *
 * Tiles overlap by more than a card, so every card that is not cut by the window lies whole in some tile. A
 * detection touching an edge shared with another tile is that card's cut-off duplicate and is dropped. Whole
 * detections of the same card in two tiles are merged. Stacked cards overlap a lot by design (IoU 0.5-0.7 for a
 * rune column), so nothing below `same` is suppressed. */
export function mergeTiles(
  perTile: readonly (readonly Detection[])[],
  origins: readonly (readonly [number, number])[],
  size: readonly [number, number],
  scale: number,
  offset: readonly [number, number],
  margin = 2,
  same = 0.8,
): FrameDetection[] {
  const [sw, sh] = size;
  const kept: Kept[] = [];
  perTile.forEach((dets, i) => {
    const [tx, ty] = origins[i]!;
    const inner = [tx > 0, ty > 0, tx + TILE < sw, ty + TILE < sh]; // left, top, right, bottom are shared edges
    for (const d of dets) {
      const [x0, y0, x1, y1] = d.box;
      const touches = [x0 <= margin, y0 <= margin, x1 >= TILE - margin, y1 >= TILE - margin];
      if (touches.some((t, k) => t && inner[k])) continue;
      const q = canonicalQuad(points(d.quad).map(([x, y]) => [(x + tx) / scale + offset[0], (y + ty) / scale + offset[1]]));
      kept.push({
        cls: d.cls,
        score: pyRound(d.score, 4),
        quad: q,
        found: d.found.map((v) => pyRound(v, 3)),
        visible: d.visible.map((v) => pyRound(v, 3)),
        truncated: touches.some(Boolean),
        tile: [tx, ty],
      });
    }
  });
  kept.sort((a, b) => b.score - a.score); // stable, as Python's sort
  const out: Kept[] = [];
  for (const d of kept) {
    const dup = out.some((o) => (o.tile[0] !== d.tile[0] || o.tile[1] !== d.tile[1]) && o.cls === d.cls && quadIou(o.quad, d.quad) >= same);
    if (!dup) out.push(d);
  }
  return out.map((d) => {
    const quad = d.quad.map(([x, y]) => [pyRound(x, 1), pyRound(y, 1)] as [number, number]);
    const xs = quad.map((p) => p[0]);
    const ys = quad.map((p) => p[1]);
    return {
      cls: d.cls,
      score: d.score,
      box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      quad,
      found: d.found,
      visible: d.visible,
      truncated: d.truncated,
    };
  });
}

// ---- the detector

export interface DetectOptions {
  /** The card length, in px, the window is scaled to. */
  target?: number;
  /** The least share of a tile its neighbours overlap. */
  overlap?: number;
  /** detect_tiles' score threshold. */
  threshold?: number;
}

export interface DetectorOptions {
  /** Tiles run at once (Detector.detect's batch). */
  batch?: number;
  /** Every run gets exactly `batch` tiles, the last padded with black ones, so a session never sees a new shape. */
  pad?: boolean;
}

/** The trained detector over a runner for its graph: Detector.detect and detect_tiles. */
export class Detector {
  readonly batch: number;
  readonly pad: boolean;

  constructor(
    private readonly run: TileRunner,
    options: DetectorOptions = {},
  ) {
    this.batch = options.batch ?? 8;
    this.pad = options.pad ?? false;
    if (!Number.isInteger(this.batch) || this.batch < 1) throw new Error(`a batch of ${this.batch} tiles`);
  }

  /** Detections per tile, in tile pixels: class, score, box, corners, and per corner the chance it was findable
   * (inside the tile) and visible (not covered). */
  async detectTiles(tiles: readonly RgbImage[], threshold = 0.3): Promise<Detection[][]> {
    const out: Detection[][] = [];
    for (let i = 0; i < tiles.length; i += this.batch) {
      const chunk = tiles.slice(i, i + this.batch);
      const n = this.pad ? this.batch : chunk.length;
      const raw = await this.run(tileBatch(chunk, n), n);
      for (let b = 0; b < chunk.length; b++) {
        const t = tileOutputs(raw, b, n);
        out.push(decodeTile(t.pred_logits, t.pred_boxes, t.pred_keypoints, threshold));
      }
    }
    return out;
  }

  /** Cards in one frame, in frame pixels. `window` is the camera window (the layout preset's ROI) and `cardPx` the
   * long side of a card there, so the window is scaled to put cards near `target` px. */
  async detect(frame: RgbImage, window: readonly number[], cardPx: number, options: DetectOptions = {}): Promise<FrameDetection[]> {
    const t = cutTiles(frame, window, cardPx, options.target ?? 70, options.overlap ?? 0.2);
    const perTile = await this.detectTiles(t.tiles, options.threshold ?? 0.3);
    return mergeTiles(perTile, t.origins, t.size, t.scale, t.offset);
  }
}

// ---- the tracker's boxes (live/pipeline.py)

/** The rotated rectangle's corners in frame px (pipeline.quad). In Python a detector box's centre and sides are
 * numpy float64, so round() there is numpy's (npRound), not CPython's. */
export function boxQuad(box: CardBox): [number, number][] {
  const [cx, cy] = box.centre;
  const a = box.angle_deg * (Math.PI / 180); // math.radians
  const ux = (Math.cos(a) * box.long_px) / 2;
  const uy = (Math.sin(a) * box.long_px) / 2;
  const vx = (-Math.sin(a) * box.short_px) / 2;
  const vy = (Math.cos(a) * box.short_px) / 2;
  const r = (v: number) => npRound(v, 1);
  return [
    [r(cx + ux + vx), r(cy + uy + vy)],
    [r(cx + ux - vx), r(cy + uy - vy)],
    [r(cx - ux - vx), r(cy - uy - vy)],
    [r(cx - ux + vx), r(cy - uy + vy)],
  ];
}

/** A card's short side over its long side is 0.72 (63 x 88 mm); a box under half is a strip. */
export const MIN_ASPECT = 0.5;

/** The trained detector's cards (Detector.detect: corners in frame px) as boxes for the tracker. A card_back is
 * marked `back`, so it is never identified. A box less than `minAspect` as wide as it is long is not a card: the
 * detector outlines the art of Riot's showdown banner, laid over the bottom of the table, as strips 2.5 times as long
 * as wide (the detector's cards, even under others, keep a card's shape). */
export function detectorBoxes(dets: readonly Detection[], minScore = DET_SCORE, minAspect = MIN_ASPECT): CardBox[] {
  const out: CardBox[] = [];
  for (const d of dets) {
    if (d.score < minScore) continue;
    const q = points(d.quad);
    if (q.length !== 4) throw new Error(`a detection has 4 corners, not ${q.length}`);
    const e = [0, 1, 2, 3].map((k) => [q[(k + 1) % 4]![0] - q[k]![0], q[(k + 1) % 4]![1] - q[k]![1]] as Point);
    const len = (v: Point) => norm2(v[0], v[1]);
    const a = (len(e[0]!) + len(e[2]!)) / 2;
    const b = (len(e[1]!) + len(e[3]!)) / 2;
    if (Math.min(a, b) < minAspect * Math.max(a, b)) continue;
    const longE = a >= b ? e[0]! : e[1]!;
    out.push({
      centre: mean(q),
      long_px: Math.max(a, b),
      short_px: Math.min(a, b),
      angle_deg: pyMod(Math.atan2(longE[1], longE[0]) * (180 / Math.PI), 180), // math.degrees
      fill: 1,
      back: d.cls === 'card_back',
      score: d.score,
      vis: Math.min(...(d.visible.length ? d.visible : [1])), // its least visible corner
    });
  }
  return dropStraddlers(dropNested(out));
}

/** One card, one box. The detector can outline a card in a magnetic case or toploader two or three times (the card,
 * the case's inner and outer edge): boxes on nearly the same centre, turned the same way and of nearly the same size
 * are one card, and the smallest, the card itself, stays. The boxes come back smallest first. */
export function dropNested(boxes: readonly CardBox[]): CardBox[] {
  const keep: CardBox[] = [];
  const area = (b: CardBox) => b.long_px * b.short_px;
  const sorted = boxes.map((b) => ({ b, key: area(b) })).sort((x, y) => (x.key < y.key ? -1 : x.key > y.key ? 1 : 0));
  for (const { b } of sorted) {
    const nested = keep.some(
      (k) => dist(b.centre, k.centre) <= 0.15 * b.long_px && Math.abs(pyMod(b.angle_deg - k.angle_deg + 90, 180) - 90) <= 12 && b.long_px / k.long_px <= 1.43,
    );
    if (!nested) keep.push(b);
  }
  return keep;
}

/** The detector's slips across two neighbouring cards. Such a box lies almost wholly on two cards that lie side by
 * side, scores below both, and still claims its four corners visible. A card under a stack has covered corners, and
 * the cards of a column overlap each other, so neither is dropped. */
export function dropStraddlers(boxes: readonly CardBox[]): CardBox[] {
  const quads = boxes.map(boxQuad);
  const areas = boxes.map((b) => b.long_px * b.short_px);
  const keep: CardBox[] = [];
  boxes.forEach((b, i) => {
    const score = b.score ?? 1;
    if ((b.vis ?? 0) >= 0.5) {
      const near = boxes.flatMap((o, j) => (j !== i && (o.score ?? 1) > score && dist(o.centre, b.centre) < b.long_px ? [j] : []));
      const share = new Map(near.map((j) => [j, overlapArea(quads[i]!, quads[j]!) / areas[i]!]));
      const on = near.filter((j) => share.get(j)! >= 0.25);
      for (let x = 0; x < on.length; x++) {
        for (let y = x + 1; y < on.length; y++) {
          const j = on[x]!;
          const k = on[y]!;
          if (share.get(j)! + share.get(k)! >= 0.75 && overlapArea(quads[j]!, quads[k]!) <= 0.1 * Math.min(areas[j]!, areas[k]!)) return;
        }
      }
    }
    keep.push(b);
  });
  return keep;
}

// ---- the finder

export interface FinderOptions extends DetectOptions {
  /** The detector's confidence below which a box is dropped (live/__main__.py --det-score). */
  detScore?: number;
}

/** The live runner's finder (live/__main__.py): detector_boxes over Detector.detect on the layout's camera window,
 * with a card's size there from the layout. */
export function makeFinder(detector: Detector, layout: Layout, options: FinderOptions = {}): Finder {
  const minScore = options.detScore ?? DET_SCORE;
  return async (_t, frame) => {
    const { width: w, height: h } = frame;
    return detectorBoxes(await detector.detect(frame, windowBox(layout, w, h), layoutCardPx(layout, h), options), minScore);
  };
}
