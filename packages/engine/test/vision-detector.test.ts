// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// detector.ts against detect/model.py, RF-DETR's PostProcess and live/pipeline.py on the seeded synthetic vectors
// of test/gen/vision_detector.py, and the detector's plumbing (tiles, batches, the finder) on made-up pictures.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  CLASSES,
  DET_SCORE,
  MIN_ASPECT,
  Detector,
  TILE,
  boxQuad,
  cutTiles,
  decodeTile,
  detectorBoxes,
  dropNested,
  dropStraddlers,
  makeFinder,
  mergeTiles,
  tileBatch,
  tileOutputs,
  type FrameDetection,
  type HeadOutputs,
} from '../src/detector';
import { rgbImage } from '../src/image';
import { LAYOUTS } from '../src/layouts';
import type { CardBox, Detection } from '../src/types';

interface PyBox {
  centre: [number, number];
  long_px: number;
  short_px: number;
  angle_deg: number;
  fill: number;
  back: boolean;
  score: number;
  vis: number;
}

interface Vectors {
  head: Record<keyof HeadOutputs, string>;
  decoded: { threshold: number; tiles: Detection[][] };
  merge: { per_tile: Detection[][]; origins: [number, number][]; size: [number, number]; scale: number; offset: [number, number]; out: PyMerged[] };
  boxes: { min_score: number; out: PyBox[]; extra: PyBox[]; nested: PyBox[]; straddlers: PyBox[] };
}

/** merge_tiles' output: the quad flat, no box. */
interface PyMerged {
  cls: Detection['cls'];
  score: number;
  quad: number[];
  found: number[];
  visible: number[];
  truncated: boolean;
}

const V = JSON.parse(readFileSync(new URL('./vectors/vision-detector.json', import.meta.url), 'utf8')) as Vectors;

const f32 = (b64: string) => {
  const bytes = Buffer.from(b64, 'base64');
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const HEAD: HeadOutputs = { pred_logits: f32(V.head.pred_logits), pred_boxes: f32(V.head.pred_boxes), pred_keypoints: f32(V.head.pred_keypoints) };

/** Detections compared field by field: class, box and corners to the bit, the float32 scores within `tol`. */
function expectSame(got: readonly Detection[], want: readonly Detection[], tol: number): void {
  expect(got.map((d) => d.cls)).toEqual(want.map((d) => d.cls));
  got.forEach((g, i) => {
    const w = want[i]!;
    expect(g.box).toEqual(w.box);
    expect(g.quad).toEqual(w.quad);
    expect(Math.abs(g.score - w.score)).toBeLessThanOrEqual(tol);
    g.found.forEach((v, j) => expect(Math.abs(v - w.found[j]!)).toBeLessThanOrEqual(tol));
    g.visible.forEach((v, j) => expect(Math.abs(v - w.visible[j]!)).toBeLessThanOrEqual(tol));
  });
}

const asBox = (b: PyBox): CardBox => ({ ...b });

describe('decodeTile (RF-DETR\'s keypoint PostProcess, as detect_tiles runs it)', () => {
  it('gives Python\'s detections: the same queries in the same order, corners to the bit, scores within a float32 step', () => {
    V.decoded.tiles.forEach((want, b) => {
      const t = tileOutputs(HEAD, b, 2);
      expectSame(decodeTile(t.pred_logits, t.pred_boxes, t.pred_keypoints, V.decoded.threshold), want, 2.5e-7);
    });
  });

  it('keeps what scores above detect_tiles\' threshold, compared in float32', () => {
    V.decoded.tiles.forEach((all, b) => {
      const t = tileOutputs(HEAD, b, 2);
      const want = all.filter((d) => d.score > Math.fround(0.3));
      expectSame(decodeTile(t.pred_logits, t.pred_boxes, t.pred_keypoints), want, 2.5e-7);
    });
  });

  it('refuses outputs of the wrong size', () => {
    expect(() => decodeTile(new Float32Array(200), new Float32Array(10), new Float32Array(6400))).toThrow('do not fit');
  });
});

describe('mergeTiles', () => {
  it('is merge_tiles to the bit: the cut-off copies dropped, the duplicates merged, the order by rounded score', () => {
    const m = V.merge;
    const got = mergeTiles(m.per_tile, m.origins, m.size, m.scale, m.offset);
    expect(got.map(({ box: _box, quad, ...rest }) => ({ ...rest, quad: quad.flat() }))).toEqual(m.out);
  });

  it('gives each card the bounding box of its corners', () => {
    const m = V.merge;
    for (const d of mergeTiles(m.per_tile, m.origins, m.size, m.scale, m.offset)) {
      const xs = d.quad.map((p) => p[0]);
      const ys = d.quad.map((p) => p[1]);
      expect(d.box).toEqual([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
    }
  });
});

describe('detectorBoxes, dropNested, dropStraddlers', () => {
  it('are live/pipeline\'s: the same boxes in the same order', () => {
    const m = V.merge;
    const merged = mergeTiles(m.per_tile, m.origins, m.size, m.scale, m.offset);
    const got = detectorBoxes(merged, V.boxes.min_score);
    expect(got.length).toBe(V.boxes.out.length);
    got.forEach((b, i) => {
      const w = V.boxes.out[i]!;
      expect(b.centre).toEqual(w.centre);
      expect(b.long_px).toBe(w.long_px);
      expect(b.short_px).toBe(w.short_px);
      expect(b.angle_deg).toBeCloseTo(w.angle_deg, 11); // math.atan2 is the C library's
      expect({ fill: b.fill, back: b.back, score: b.score, vis: b.vis }).toEqual({ fill: w.fill, back: w.back, score: w.score, vis: w.vis });
    });
    expect(got.filter((b) => b.back).length).toBeGreaterThan(0);
  });

  it('keep the card itself of a toploader\'s outlines, and drop a slip across two cards', () => {
    const extra = V.boxes.extra.map(asBox);
    expect(dropNested(extra)).toEqual(V.boxes.nested);
    expect(dropStraddlers(dropNested(extra))).toEqual(V.boxes.straddlers);
    expect(V.boxes.straddlers.length).toBeLessThan(V.boxes.nested.length);
    expect(V.boxes.nested.length).toBeLessThan(V.boxes.extra.length);
  });

  it('skip strips: a box less than half as wide as it is long is a banner\'s art, not a card', () => {
    const d = (w: number, h: number): Detection => ({ cls: 'card', score: 0.8, box: [0, 0, w, h], quad: [[0, 0], [w, 0], [w, h], [0, h]], found: [1, 1, 1, 1], visible: [1, 1, 1, 1] });
    expect(MIN_ASPECT).toBe(0.5);
    expect(detectorBoxes([d(56, 78), d(160, 64)]).map((b) => b.long_px)).toEqual([78]);
    expect(detectorBoxes([d(56, 78), d(160, 64)], DET_SCORE, 0)).toHaveLength(2);
  });

  it('skip scores below the tracker\'s --det-score', () => {
    const d = (score: number): Detection => ({ cls: 'card', score, box: [0, 0, 10, 10], quad: [[0, 0], [70, 0], [70, 50], [0, 50]], found: [1, 1, 1, 1], visible: [1, 1, 1, 1] });
    expect(DET_SCORE).toBe(0.4);
    expect(detectorBoxes([d(0.39), d(0.4)]).map((b) => b.score)).toEqual([0.4]);
  });

  it('boxQuad rounds as numpy does on a numpy scalar', () => {
    // cx + ux + vx = 4.35 exactly: numpy's round() makes it 4.4 (rint(43.5)), CPython's would make it 4.3
    const b: CardBox = { centre: [4.35, 10], long_px: 0, short_px: 0, angle_deg: 0, fill: 1 };
    expect(boxQuad(b)[0]).toEqual([4.4, 10]);
  });
});

describe('the tiles of a frame', () => {
  it('cuts the window as Detector.detect does: rounded, scaled, and one 576 px tile when it fits', () => {
    const frame = rgbImage(1920, 1080);
    frame.data.fill(200);
    const t = cutTiles(frame, [364.8, 64.8, 1555.2, 1080], 155); // la-rq's window, rounded to 365, 65, 1555, 1080
    expect(t.offset).toEqual([365, 65]);
    expect(t.scale).toBe(70 / 155);
    expect(t.size).toEqual([537, 458]); // round(1190 * 70 / 155), round(1015 * 70 / 155)
    expect(t.origins).toEqual([[0, 0]]);
    expect(t.tiles).toHaveLength(1);
    const tile = t.tiles[0]!;
    expect([tile.width, tile.height]).toEqual([TILE, TILE]);
    // the window inside, black past it
    expect(tile.data[(100 * TILE + 100) * 3]).toBe(200);
    expect(tile.data[(500 * TILE + 560) * 3]).toBe(0);
  });

  it('cuts two overlapping tiles for PlusRB\'s window, rounding 507.5 to 508', () => {
    const t = cutTiles(rgbImage(1920, 1080), [365, 65, 1555, 1080], 140);
    expect(t.size).toEqual([595, 508]);
    expect(t.origins).toEqual([[0, 0], [19, 0]]);
  });

  it('makes the graph\'s input: RGB / 255 as float32, NCHW, black past the tiles', () => {
    const tile = rgbImage(TILE, TILE);
    tile.data.set([255, 51, 1], 0);
    const x = tileBatch([tile], 2);
    const plane = TILE * TILE;
    expect(x.length).toBe(2 * 3 * plane);
    expect([x[0], x[plane], x[2 * plane]]).toEqual([1, Math.fround(0.2), Math.fround(1 / 255)]);
    expect(x.subarray(3 * plane).every((v) => v === 0)).toBe(true);
    expect(() => tileBatch([rgbImage(10, 10)])).toThrow('576 x 576');
  });
});

describe('Detector', () => {
  /** A stand-in net: every tile gets the synthetic head's first tile, and each run is logged. */
  const standIn = (runs: number[]) => async (_x: Float32Array, n: number): Promise<HeadOutputs> => {
    runs.push(n);
    const one = tileOutputs(HEAD, 0, 2);
    const rep = (a: Float32Array) => {
      const out = new Float32Array(a.length * n);
      for (let i = 0; i < n; i++) out.set(a, i * a.length);
      return out;
    };
    return { pred_logits: rep(one.pred_logits), pred_boxes: rep(one.pred_boxes), pred_keypoints: rep(one.pred_keypoints) };
  };

  it('runs the tiles in batches, and pads the last one when asked', async () => {
    const tiles = Array.from({ length: 5 }, () => rgbImage(TILE, TILE));
    const runs: number[] = [];
    const per = await new Detector(standIn(runs), { batch: 2 }).detectTiles(tiles);
    expect(runs).toEqual([2, 2, 1]);
    expect(per).toHaveLength(5);
    const padded: number[] = [];
    await new Detector(standIn(padded), { batch: 2, pad: true }).detectTiles(tiles);
    expect(padded).toEqual([2, 2, 2]);
    const want = V.decoded.tiles[0]!.filter((d) => d.score > Math.fround(0.3));
    for (const d of per) expectSame(d, want, 2.5e-7);
  });

  it('detects a frame and makes the finder the live runner builds', async () => {
    const frame = rgbImage(1920, 1080);
    const det = new Detector(standIn([]), { batch: 1 });
    const cards: FrameDetection[] = await det.detect(frame, [365, 65, 1555, 1080], 155);
    expect(cards.length).toBeGreaterThan(10);
    for (const c of cards) {
      expect(CLASSES).toContain(c.cls);
      expect(c.quad).toHaveLength(4);
      expect(typeof c.truncated).toBe('boolean');
    }
    const finder = makeFinder(det, LAYOUTS['la-rq']);
    const boxes = await finder(0, frame);
    expect(boxes).toEqual(detectorBoxes(cards, DET_SCORE));
  });
});
