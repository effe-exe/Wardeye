// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The card embedder: crops to rows. A port of ml/rifteye_ml/encoders.py (letterbox, l2n) and embed/onnx.py
// (batch_of, and the Onnx encoder with its name "onnx:<stem>-<sha8>"), so a row and the name an embedding cache
// is keyed on come out as Python's for the same crop and the same file.
//
// Like the detector, the encoder is given a runner for the graph ("crops", float32 NCHW, RGB 0..255, 224 x 224, to
// "embedding", float32, 256 a row), so any ONNX Runtime build (ort.ts) can run it. Unlike Python's, it runs fixed
// batches: the last one is padded with black crops, so a WebGPU session never meets a new shape and compiles once.
// Eight a batch: the tracker embeds a median of 8 pictures a frame (each crop in its four turns, so always a
// multiple of 4; p90 16, at most 40), so the typical frame is one run with nothing padded, a frame with an odd
// number of crops pads 4, and a busy one takes a few full runs. 16 or 32 would double or quadruple the typical
// frame's work; 4 would double its runs, each with its own dispatch and read-back.

import { paste, resize, rgbImage } from './image';
import type { Encoder, RgbImage } from './types';

/** embedder-v1's input side and row length. */
export const IMG_SIZE = 224;
export const DIM = 256;
/** Crops per run of the graph (see above). */
export const BATCH = 8;

/** Runs the embedder's graph on `n` crops (float32 NCHW, RGB 0..255): n rows of `dim` values. */
export type EmbedRunner = (crops: Float32Array, n: number) => Promise<Float32Array>;

/** encoders.letterbox: the crop centred on a black square of its long side, resized with Pillow's bicubic filter. */
export function letterbox(im: RgbImage, size = IMG_SIZE): RgbImage {
  const side = Math.max(im.width, im.height);
  const canvas = rgbImage(side, side);
  paste(canvas, im, [Math.floor((side - im.width) / 2), Math.floor((side - im.height) / 2)]);
  return resize(canvas, [size, size], 'bicubic');
}

/** embed/onnx.batch_of: the crops letterboxed, as the float32 NCHW array the graph takes (values 0..255); `n`
 * slots, those past the crops black (zero). */
export function cropBatch(images: readonly RgbImage[], size = IMG_SIZE, n = images.length): Float32Array {
  if (n < images.length) throw new Error(`${images.length} crops do not fit a batch of ${n}`);
  const plane = size * size;
  const out = new Float32Array(n * 3 * plane);
  images.forEach((im, i) => {
    const d = letterbox(im, size).data;
    const r = i * 3 * plane;
    for (let p = 0, s = 0; p < plane; p++, s += 3) {
      out[r + p] = d[s]!;
      out[r + plane + p] = d[s + 1]!;
      out[r + 2 * plane + p] = d[s + 2]!;
    }
  });
  return out;
}

const f32 = Math.fround;
const TINY = f32(1e-12);

/** numpy's add.reduce of float32 values, in its pairwise order, accumulating in float32. */
function pairwiseSum32(a: Float32Array, at: number, n: number): number {
  if (n < 8) {
    let s = 0;
    for (let i = 0; i < n; i++) s = f32(s + a[at + i]!);
    return s;
  }
  if (n <= 128) {
    const r = Array.from(a.subarray(at, at + 8));
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] = f32(r[k]! + a[at + i + k]!);
    let s = f32(f32(f32(r[0]! + r[1]!) + f32(r[2]! + r[3]!)) + f32(f32(r[4]! + r[5]!) + f32(r[6]! + r[7]!)));
    for (; i < n; i++) s = f32(s + a[at + i]!);
    return s;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f32(pairwiseSum32(a, at, n2) + pairwiseSum32(a, at + n2, n - n2));
}

/** encoders.l2n on float32 rows of `dim`: each divided by its norm (at least 1e-12), in float32 as numpy does it. */
export function l2n(rows: Float32Array, dim: number): Float32Array {
  const out = new Float32Array(rows.length);
  const sq = new Float32Array(dim);
  for (let at = 0; at < rows.length; at += dim) {
    for (let i = 0; i < dim; i++) sq[i] = rows[at + i]! * rows[at + i]!;
    const norm = Math.max(f32(Math.sqrt(pairwiseSum32(sq, 0, dim))), TINY);
    for (let i = 0; i < dim; i++) out[at + i] = rows[at + i]! / norm;
  }
  return out;
}

/** pathlib's Path(name).stem: the file name without its last suffix. */
export function stem(file: string): string {
  const name = file.split(/[\\/]/).pop() ?? '';
  const i = name.lastIndexOf('.');
  return i > 0 && i < name.length - 1 ? name.slice(0, i) : name;
}

/** SHA-256 as hex, by the Web Crypto every browser, worker and Node has. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** embed/onnx.Onnx's name for a model file: "onnx:<stem>-<first 8 hex of its SHA-256>". One name per file, so
 * embedding caches key on it; the fp32 and fp16 files have different names. */
export async function encoderName(file: string, bytes: Uint8Array): Promise<string> {
  return `onnx:${stem(file)}-${(await sha256Hex(bytes)).slice(0, 8)}`;
}

export interface OnnxEncoderOptions {
  /** Crops per run; the last run is padded to it. */
  batch?: number;
  imgSize?: number;
  dim?: number;
}

/** An exported embedder as an Encoder (embed/onnx.Onnx): crops in, L2-normalised rows out, run in fixed batches. */
export class OnnxEncoder implements Encoder {
  readonly batch: number;
  readonly imgSize: number;
  readonly dim: number;

  constructor(
    readonly name: string,
    private readonly run: EmbedRunner,
    options: OnnxEncoderOptions = {},
  ) {
    this.batch = options.batch ?? BATCH;
    this.imgSize = options.imgSize ?? IMG_SIZE;
    this.dim = options.dim ?? DIM;
    if (!Number.isInteger(this.batch) || this.batch < 1) throw new Error(`a batch of ${this.batch} crops`);
  }

  async embed(images: readonly RgbImage[]): Promise<Float32Array> {
    const raw = new Float32Array(images.length * this.dim);
    for (let i = 0; i < images.length; i += this.batch) {
      const chunk = images.slice(i, i + this.batch);
      const rows = await this.run(cropBatch(chunk, this.imgSize, this.batch), this.batch);
      if (rows.length !== this.batch * this.dim) throw new Error(`the embedder gave ${rows.length} values for ${this.batch} crops of ${this.dim}`);
      raw.set(rows.subarray(0, chunk.length * this.dim), i * this.dim);
    }
    return l2n(raw, this.dim);
  }
}
