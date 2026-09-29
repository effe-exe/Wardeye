// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The detector's and the embedder's port against Python on real frames: what test/gen/vision_parity.py recorded
// of Detector.detect (21 LA frames, 10 Barcelona ones), detector_boxes, and 40 crops of those boxes. It needs the
// private fixtures (D-006) and skips without them: RIFTEYE_M3=~/rifteye-data/m3 npx vitest run packages/engine.

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Detector, cutTiles, decodeTile, detectorBoxes, makeFinder, mergeTiles, tileBatch, type HeadOutputs } from '../src/detector';
import { cropBatch, l2n, letterbox } from '../src/embedder';
import { points } from '../src/geometry';
import { rgbImage } from '../src/image';
import { LAYOUTS } from '../src/layouts';
import type { CardBox, Detection, RgbImage } from '../src/types';

const M3 = process.env.RIFTEYE_M3;
const VIS = M3 ? join(M3, 'fixtures', 'vision') : '';
const ready = Boolean(M3 && existsSync(join(VIS, 'detector', 'index.json')));

interface PyMerged {
  cls: Detection['cls'];
  score: number;
  quad: number[];
  found: number[];
  visible: number[];
  truncated: boolean;
}

interface Frame {
  id: string;
  file: string;
  layout: 'la-rq' | 'plusrb';
  size: [number, number];
  window: [number, number, number, number];
  card_px: number;
  scale: number;
  win_size: [number, number];
  tiles: { origin: [number, number]; rgb_sha256: string; input_sha256: string }[];
  per_tile: Detection[][];
  detect: PyMerged[];
  min_score: number;
  boxes: Required<CardBox>[];
}

const json = <T>(...p: string[]): T => JSON.parse(readFileSync(join(VIS, ...p), 'utf8')) as T;
const floats = (...p: string[]) => {
  const b = readFileSync(join(VIS, ...p));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const sha = (b: Uint8Array | Float32Array) => createHash('sha256').update(new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).digest('hex');

const frames: Frame[] = ready ? json<{ frames: string[] }>('detector', 'index.json').frames.map((id) => json<Frame>('detector', `${id}.json`)) : [];
const raw = (f: Frame): HeadOutputs => ({
  pred_logits: floats('detector', `${f.id}.pred_logits.bin`),
  pred_boxes: floats('detector', `${f.id}.pred_boxes.bin`),
  pred_keypoints: floats('detector', `${f.id}.pred_keypoints.bin`),
});
const part = (a: Float32Array, b: number, n: number) => a.subarray((b * a.length) / n, ((b + 1) * a.length) / n);

/** The largest difference between two detections' floats, and whether the rest is equal. */
function worst(got: Detection, want: Detection): number {
  expect(got.cls).toBe(want.cls);
  expect(got.box).toEqual(want.box);
  expect(got.quad).toEqual(want.quad);
  return Math.max(Math.abs(got.score - want.score), ...got.found.map((v, j) => Math.abs(v - want.found[j]!)), ...got.visible.map((v, j) => Math.abs(v - want.visible[j]!)));
}

/** merge_tiles' output as detector.ts gives it: pairs for the flat quad, and a box. */
const flat = (ds: readonly Detection[]) => ds.map(({ box: _b, quad, ...rest }) => ({ ...rest, quad: quad.flat() }));

/** Merged detections equal up to one rounding step: the float32 postprocess can move a score or corner across a
 * rounding boundary (4, 3 and 1 decimals). Returns how many values sat on one. */
function nearlySame(got: readonly Detection[], want: readonly PyMerged[]): number {
  expect(got.map((d) => d.cls)).toEqual(want.map((d) => d.cls));
  let steps = 0;
  const close = (a: number, b: number, step: number) => {
    const d = Math.abs(a - b);
    expect(d).toBeLessThanOrEqual(step * 1.0001);
    if (d > 0) steps++;
  };
  got.forEach((g, i) => {
    const w = want[i]!;
    close(g.score, w.score, 1e-4);
    g.quad.flat().forEach((v, j) => close(v, w.quad[j]!, 0.1));
    g.found.forEach((v, j) => close(v, w.found[j]!, 1e-3));
    g.visible.forEach((v, j) => close(v, w.visible[j]!, 1e-3));
  });
  return steps;
}

describe.skipIf(!ready)('the detector on real frames, against Python', () => {
  it('decodeTile gives detect_tiles\' detections of the same outputs: corners and boxes to the bit, scores within float32 steps', () => {
    let tiles = 0;
    let exact = 0;
    let dets = 0;
    let max = 0;
    for (const f of frames) {
      const r = raw(f);
      const n = f.tiles.length;
      f.per_tile.forEach((want, b) => {
        const got = decodeTile(part(r.pred_logits, b, n), part(r.pred_boxes, b, n), part(r.pred_keypoints, b, n));
        expect(got.length).toBe(want.length);
        got.forEach((g, i) => {
          const d = worst(g, want[i]!);
          max = Math.max(max, d);
          if (d === 0) exact++;
          dets++;
        });
        tiles++;
      });
    }
    console.log(`decodeTile: ${tiles} tiles, ${dets} detections, ${exact} bit for bit; largest float difference ${max.toExponential(2)}`);
    expect(tiles).toBe(41);
    expect(max).toBeLessThan(1e-6);
  });

  it('mergeTiles is merge_tiles, to the bit, on the same per-tile detections', () => {
    for (const f of frames) {
      const t = f.tiles.map((x) => x.origin);
      expect(flat(mergeTiles(f.per_tile, t, f.win_size, f.scale, [f.window[0], f.window[1]]))).toEqual(f.detect);
    }
  });

  it('detectorBoxes is detector_boxes: the same boxes in the same order', () => {
    let n = 0;
    let angle = 0;
    for (const f of frames) {
      const got = detectorBoxes(f.detect.map((d): Detection => ({ ...d, box: [0, 0, 0, 0], quad: points(d.quad) })), f.min_score);
      expect(got.length).toBe(f.boxes.length);
      got.forEach((b, i) => {
        const w = f.boxes[i]!;
        expect([b.centre, b.long_px, b.short_px, b.fill, b.back, b.score, b.vis]).toEqual([w.centre, w.long_px, w.short_px, w.fill, w.back, w.score, w.vis]);
        angle = Math.max(angle, Math.abs(b.angle_deg - w.angle_deg));
        n++;
      });
    }
    console.log(`detectorBoxes: ${n} boxes; largest angle difference ${angle.toExponential(2)} deg (math.atan2)`);
    expect(angle).toBeLessThan(1e-9);
  });

  it('the whole chain from the raw outputs gives Python\'s cards and boxes', () => {
    let steps = 0;
    let boxes = 0;
    for (const f of frames) {
      const r = raw(f);
      const n = f.tiles.length;
      const per = f.tiles.map((_, b) => decodeTile(part(r.pred_logits, b, n), part(r.pred_boxes, b, n), part(r.pred_keypoints, b, n)));
      const merged = mergeTiles(per, f.tiles.map((x) => x.origin), f.win_size, f.scale, [f.window[0], f.window[1]]);
      steps += nearlySame(merged, f.detect);
      const b = detectorBoxes(merged, f.min_score);
      expect(b.length).toBe(f.boxes.length);
      boxes += b.length;
    }
    console.log(`the chain: ${boxes} boxes over ${frames.length} frames; ${steps} values one rounding step from Python's`);
  });
});

/** The three frames decoded to raw RGB for Node (Pillow's decode of f0000, f0120, f0239). */
const RGB = ['f0000', 'f0120', 'f0239'];
const rgbReady = ready && RGB.every((n) => existsSync(join(M3!, 'frames', 'la-final-rgb', `${n}.rgb`)));
const frameRgb = (n: string): RgbImage => rgbImage(1920, 1080, new Uint8Array(readFileSync(join(M3!, 'frames', 'la-final-rgb', `${n}.rgb`))));

describe.skipIf(!rgbReady)('the detector from a frame\'s pixels, against Python', () => {
  it('cuts the same tiles and gives the graph the same input, byte for byte', () => {
    for (const n of RGB) {
      const f = frames.find((x) => x.id === `la-${n}`)!;
      const t = cutTiles(frameRgb(n), f.window, f.card_px);
      expect(t.size).toEqual(f.win_size);
      expect(t.origins).toEqual(f.tiles.map((x) => x.origin));
      t.tiles.forEach((tile, i) => {
        expect(sha(tile.data)).toBe(f.tiles[i]!.rgb_sha256);
        expect(sha(tileBatch([tile]))).toBe(f.tiles[i]!.input_sha256);
      });
    }
  });

  it('Detector.detect and the finder give Python\'s cards, with the net\'s outputs replayed', async () => {
    for (const n of RGB) {
      const f = frames.find((x) => x.id === `la-${n}`)!;
      const r = raw(f);
      const replay = async (x: Float32Array) => {
        expect(sha(x)).toBe(f.tiles[0]!.input_sha256);
        return r;
      };
      const det = new Detector(replay, { batch: 1 });
      const image = frameRgb(n);
      nearlySame(await det.detect(image, f.window, f.card_px), f.detect);
      const boxes = await makeFinder(det, LAYOUTS['la-rq'])(0, image);
      expect(boxes.map((b) => b.centre)).toEqual(f.boxes.map((b) => b.centre));
    }
  });
});

interface Crops {
  encoder: string;
  img_size: number;
  dim: number;
  crops: { id: string; size: [number, number]; rgb_sha256: string; letterbox_sha256: string; input_sha256: string }[];
}

const cropsReady = ready && existsSync(join(VIS, 'crops', 'crops.json'));

describe.skipIf(!cropsReady)('the embedder\'s input on real crops, against Python', () => {
  it('letterboxes and batches the crops as Python does, byte for byte, and normalises as l2n', () => {
    const c = json<Crops>('crops', 'crops.json');
    for (const m of c.crops) {
      const im = rgbImage(m.size[0], m.size[1], new Uint8Array(readFileSync(join(VIS, 'crops', `${m.id}.rgb`))));
      expect(sha(im.data)).toBe(m.rgb_sha256);
      expect(sha(letterbox(im, c.img_size).data)).toBe(m.letterbox_sha256);
      expect(sha(cropBatch([im], c.img_size))).toBe(m.input_sha256);
    }
    expect(Array.from(l2n(floats('crops', 'embed_raw.bin'), c.dim))).toEqual(Array.from(floats('crops', 'embed.bin')));
  });
});
