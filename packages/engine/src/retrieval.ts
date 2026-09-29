// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Gallery search, ported from ml/rifteye_ml/retrieval.py: the gallery pyramid (embeddings of the clean art at a few
// on-screen sizes, and which level a crop of a given size is searched against), the top-k search and the
// four-turn search. Embeddings are flat Float32Arrays, row by row, as Encoder.embed returns them.
//
// Scores are float32 as in numpy. numpy's float32 matmul (OpenBLAS 0.3.31, SkylakeX kernels, as on the machine the
// fixtures come from) adds the products of a dot product one after another, each add fused with its multiply and
// rounded to float32 once: acc = fl32(a[k] * b[k] + acc). Doing the same here (fmaF32) gives numpy's scores bit for
// bit for a gallery of 400 rows or more and two or more queries, up to 384 values a row; on other machines, and for
// a single query (numpy takes another routine then) or a long row (OpenBLAS adds block sums), a score can differ
// in its last bit or two (about 1e-7). Ties in a ranking go to the lower index (numpy's default argsort leaves them
// in a machine-dependent order).

import * as image from './image';
import { pyRound } from './pynum';
import type { Encoder, RgbImage } from './types';

export const ROTATIONS = [0, 90, 180, 270] as const;

/** Embeddings, one row for each of `rows` pictures, `dim` values a row, row by row: what Encoder.embed gives. */
export interface Embeddings {
  readonly data: Float32Array;
  readonly rows: number;
  readonly dim: number;
}

/** A clean card as it lands on screen with its long side at `side` px (area-averaged, like optics). */
export function atLongSide(im: RgbImage, side: number): RgbImage {
  const scale = side / Math.max(im.width, im.height);
  return image.resize(im, [Math.max(2, pyRound(im.width * scale)), Math.max(2, pyRound(im.height * scale))], 'box');
}

export const EDGES = ['top', 'bottom', 'left', 'right'] as const;

/** The part of an upright card a stack leaves visible: 'top:0.25' is the top quarter. 'full' (or '') is the
 * whole card. */
export function band(im: RgbImage, view: string): RgbImage {
  if (view === '' || view === 'full') return im;
  const colon = view.indexOf(':');
  const edge = colon < 0 ? view : view.slice(0, colon);
  const text = colon < 0 ? '' : view.slice(colon + 1).trim();
  const f = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(text) ? Number(text) : NaN;
  if (!(EDGES as readonly string[]).includes(edge) || !(f > 0 && f <= 1)) {
    throw new Error(`bad view ${JSON.stringify(view)}; expected <top|bottom|left|right>:<fraction>`);
  }
  const w = im.width;
  const h = im.height;
  const bw = Math.max(2, pyRound(w * f));
  const bh = Math.max(2, pyRound(h * f));
  const box: Record<string, [number, number, number, number]> = {
    top: [0, 0, w, bh],
    bottom: [0, h - bh, w, h],
    left: [0, 0, bw, h],
    right: [w - bw, 0, w, h],
  };
  return image.crop(im, box[edge]!);
}

/** Gallery embeddings of the clean art at a few on-screen sizes.
 *
 * A pretrained backbone sees a sharp 744 px card and a blurry 40 px crop as different images. Embedding the
 * gallery at the size a card appears on screen closes most of that gap, and the detector already knows each
 * card's size. Each query is searched against the level nearest its own long side, in log space. */
export class Pyramid {
  /** Each level's embeddings (rows x dim, row by row), by the long side in px the art was drawn at. */
  readonly levels: ReadonlyMap<number, Float32Array>;
  /** The levels' sizes, ascending. */
  readonly scales: readonly number[];
  readonly dim: number;
  /** Gallery rows: the printings, the same in every level. */
  readonly rows: number;

  constructor(levels: ReadonlyMap<number, Float32Array> | Readonly<Record<number, Float32Array>>, dim: number) {
    const entries: [number, Float32Array][] =
      levels instanceof Map ? [...levels] : Object.entries(levels).map(([x, v]) => [Number(x), v as Float32Array]);
    if (entries.length === 0) throw new Error('a pyramid needs at least one scale');
    this.levels = new Map(entries.map(([x, v]) => [Math.trunc(x), v]));
    this.scales = [...this.levels.keys()].sort((a, b) => a - b);
    this.dim = dim;
    this.rows = entries[0]![1].length / dim;
  }

  /** The gallery embedded at each of `scales` (a long side in px each): `encoder` reads every picture once per level. */
  static async build(encoder: Encoder, images: readonly RgbImage[], scales: Iterable<number>): Promise<Pyramid> {
    const levels = new Map<number, Float32Array>();
    for (const x of new Set([...scales].map(Math.trunc))) {
      levels.set(x, await encoder.embed(images.map((im) => atLongSide(im, x))));
    }
    return new Pyramid(levels, encoder.dim);
  }

  /** The level nearest a crop's long side, in log space (the lowest level when two are as near). */
  levelFor(longSide: number): number {
    let best = this.scales[0]!;
    let bestKey = Infinity;
    for (const x of this.scales) {
      const key = Math.abs(Math.log(x / Math.max(1, longSide)));
      if (key < bestKey) {
        best = x;
        bestKey = key;
      }
    }
    return best;
  }

  /** The embeddings a crop of this long side is searched against: `levels[levelFor(longSide)]`. */
  level(longSide: number): Float32Array {
    return this.levels.get(this.levelFor(longSide))!;
  }
}

export type Gallery = Embeddings | Pyramid;

/** fl32(a * b + c) for float32 values, rounded once: a fused multiply-add in float32. The product of two float32
 * values is exact in a double, so the sum is one double rounding; rounding that to float32 again can go wrong only
 * when the double lands exactly halfway between two float32 values, and then the sign of the sum's error says
 * which way the true value lies. */
export function fmaF32(a: number, b: number, c: number): number {
  const s = a * b + c;
  const r = Math.fround(s);
  if (r !== s) {
    const other = 2 * s - r; // the float32 on the other side of s, if s is halfway between two
    if (Math.fround(other) === other) return fmaMidpoint(a, b, c, s, r);
  }
  return r;
}

/** fmaF32 where the double sum s is exactly halfway between two float32 values (r is the one it rounded to). */
function fmaMidpoint(a: number, b: number, c: number, s: number, r: number): number {
  const p = a * b;
  const bb = s - p;
  const err = p - (s - bb) + (c - bb); // p + c = s + err exactly (TwoSum)
  return err === 0 ? r : Math.fround(s + (err > 0 ? 1 : -1) * Math.abs(s) * 2 ** -40);
}

/** queries @ gallery.T: nq x ng cosine scores (inputs L2-normalised), row by row, float32, summed the way numpy's
 * float32 matmul does (see the top of this file). Four queries' chains run together against each gallery row: each is
 * summed in the same order as alone, so the same bits come out, only faster. */
export function similarities(queries: Embeddings, gallery: Embeddings): Float32Array {
  const { dim } = queries;
  if (gallery.dim !== dim) throw new Error(`the queries have ${dim} values a row and the gallery ${gallery.dim}`);
  const nq = queries.rows;
  const ng = gallery.rows;
  const q = queries.data;
  const g = gallery.data;
  const out = new Float32Array(nq * ng);
  for (let row = 0; row < ng; row++) {
    const go = row * dim;
    let i = 0;
    for (; i + 4 <= nq; i += 4) {
      const o0 = i * dim;
      const o1 = o0 + dim;
      const o2 = o1 + dim;
      const o3 = o2 + dim;
      let a0 = 0;
      let a1 = 0;
      let a2 = 0;
      let a3 = 0;
      for (let d = 0; d < dim; d++) {
        const x = g[go + d]!;
        a0 = fmaF32(q[o0 + d]!, x, a0);
        a1 = fmaF32(q[o1 + d]!, x, a1);
        a2 = fmaF32(q[o2 + d]!, x, a2);
        a3 = fmaF32(q[o3 + d]!, x, a3);
      }
      out[i * ng + row] = a0;
      out[(i + 1) * ng + row] = a1;
      out[(i + 2) * ng + row] = a2;
      out[(i + 3) * ng + row] = a3;
    }
    for (; i < nq; i++) {
      const o = i * dim;
      let acc = 0;
      for (let d = 0; d < dim; d++) acc = fmaF32(q[o + d]!, g[go + d]!, acc);
      out[i * ng + row] = acc;
    }
  }
  return out;
}

/** The order of `x` from its largest value down (np.argsort(-x)), equal values in index order, NaNs last. */
export function argsortDescending(x: ArrayLike<number>): Int32Array {
  const order = Int32Array.from({ length: x.length }, (_, i) => i);
  return order.sort((i, j) => {
    const a = x[i]!;
    const b = x[j]!;
    if (a !== a) return b !== b ? i - j : 1;
    if (b !== b) return -1;
    return b - a || i - j;
  });
}

/** (views @ gallery.T).max(axis=0): for each gallery row, its best score over the rows of `views`. The identifier
 * embeds a crop at all four turns and keeps a card's best. */
export function bestSimilarities(views: Embeddings, gallery: Embeddings): Float32Array {
  const s = similarities(views, gallery);
  const out = new Float32Array(gallery.rows);
  for (let g = 0; g < gallery.rows; g++) {
    let best = s[g]!;
    for (let r = 1; r < views.rows; r++) best = Math.max(best, s[r * gallery.rows + g]!);
    out[g] = best;
  }
  return out;
}

/** Indices and cosine scores of the k nearest gallery rows for each query (inputs L2-normalised): nq x k each, row by
 * row, best first. `k` is at most the gallery's rows. */
export function topk(queries: Embeddings, gallery: Embeddings, k = 5): { idx: Int32Array; scores: Float32Array; k: number } {
  const sims = similarities(queries, gallery);
  const kk = Math.min(k, gallery.rows);
  const idx = new Int32Array(queries.rows * kk);
  const scores = new Float32Array(queries.rows * kk);
  for (let q = 0; q < queries.rows; q++) {
    const row = sims.subarray(q * gallery.rows, (q + 1) * gallery.rows);
    const order = argsortDescending(row);
    for (let j = 0; j < kk; j++) {
      idx[q * kk + j] = order[j]!;
      scores[q * kk + j] = row[order[j]!]!;
    }
  }
  return { idx, scores, k: kk };
}

export interface SearchResult {
  /** Gallery rows: nq x k, best first. */
  idx: Int32Array;
  /** Their cosine scores, float32. */
  scores: Float32Array;
  /** The turn (0, 90, 180 or 270 degrees) that matched each. */
  rot: Int32Array;
  k: number;
}

/** Top-k over the gallery. With `rotationInvariant`, each query is embedded at all four 90 degree turns (one batch) and
 * each gallery row keeps its best score, which also returns the turn that matched. That is how exhausted and
 * opponent-side cards are read. A `Pyramid` gallery routes each query to the level nearest its size. */
export async function search(
  encoder: Encoder,
  gallery: Gallery,
  queries: readonly RgbImage[],
  k = 5,
  rotationInvariant = true,
): Promise<SearchResult> {
  if (gallery instanceof Pyramid) {
    const kk = Math.min(k, gallery.rows);
    const idx = new Int32Array(queries.length * kk);
    const scores = new Float32Array(queries.length * kk);
    const rot = new Int32Array(queries.length * kk);
    const groups = new Map<number, number[]>();
    queries.forEach((q, i) => {
      const level = gallery.levelFor(Math.max(q.width, q.height));
      groups.set(level, [...(groups.get(level) ?? []), i]);
    });
    for (const [level, members] of groups) {
      const part = await search(encoder, { data: gallery.levels.get(level)!, rows: gallery.rows, dim: gallery.dim }, members.map((i) => queries[i]!), kk, rotationInvariant);
      members.forEach((m, n) => {
        idx.set(part.idx.subarray(n * kk, (n + 1) * kk), m * kk);
        scores.set(part.scores.subarray(n * kk, (n + 1) * kk), m * kk);
        rot.set(part.rot.subarray(n * kk, (n + 1) * kk), m * kk);
      });
    }
    return { idx, scores, rot, k: kk };
  }
  const rots = rotationInvariant ? ROTATIONS : ([0] as const);
  const batch: RgbImage[] = [];
  for (const q of queries) for (const r of rots) batch.push(r ? image.rotate(q, r, { expand: true }) : q);
  const emb = await encoder.embed(batch);
  const kk = Math.min(k, gallery.rows);
  const idx = new Int32Array(queries.length * kk);
  const scores = new Float32Array(queries.length * kk);
  const rot = new Int32Array(queries.length * kk);
  const { dim } = gallery;
  for (let n = 0; n < queries.length; n++) {
    const views: Embeddings = { data: emb.subarray(n * rots.length * dim, (n + 1) * rots.length * dim), rows: rots.length, dim };
    const sims = similarities(views, gallery); // rots x gallery rows
    const best = new Float32Array(gallery.rows);
    const bestRot = new Int32Array(gallery.rows);
    for (let g = 0; g < gallery.rows; g++) {
      let b = sims[g]!;
      let br = 0;
      for (let r = 1; r < rots.length; r++) {
        const v = sims[r * gallery.rows + g]!;
        if (v > b) {
          b = v;
          br = r; // the first turn with the top score, as argmax takes it
        }
      }
      best[g] = b;
      bestRot[g] = br;
    }
    const order = argsortDescending(best);
    for (let j = 0; j < kk; j++) {
      const g = order[j]!;
      idx[n * kk + j] = g;
      scores[n * kk + j] = best[g]!;
      rot[n * kk + j] = rots[bestRot[g]!]!;
    }
  }
  return { idx, scores, rot, k: kk };
}

/** Map gallery indices to labels, dropping duplicates while keeping rank order. With card-level labels this rolls
 * printings up into cards (docs/ARCHITECTURE.md section 4). `idx` is nq x k, row by row. */
export function rankedLabels(idx: ArrayLike<number>, k: number, labels: readonly string[]): string[][] {
  const out: string[][] = [];
  for (let q = 0; q * k < idx.length; q++) {
    const seen: string[] = [];
    for (let j = 0; j < k; j++) {
      const lab = labels[idx[q * k + j]!]!;
      if (!seen.includes(lab)) seen.push(lab);
    }
    out.push(seen);
  }
  return out;
}

/** Top-1 and top-5 card accuracy of ranked labels against the truth; NaN for both when there is none. */
export function accuracy(ranked: readonly (readonly string[])[], truth: readonly string[]): { top1: number; top5: number; n: number } {
  const n = truth.length;
  if (n === 0) return { top1: NaN, top5: NaN, n: 0 };
  let top1 = 0;
  let top5 = 0;
  ranked.forEach((r, i) => {
    if (i >= n) return;
    if (r[0] === truth[i]) top1++;
    if (r.slice(0, 5).includes(truth[i]!)) top5++;
  });
  return { top1: top1 / n, top5: top5 / n, n };
}
