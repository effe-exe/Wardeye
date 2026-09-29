// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The picture operations the engine shares, bit for bit as Pillow does them, so a port of ml/ sees exactly the
// pixels its Python original sees. A port of Pillow 12.3.0:
//   src/PIL/Image.py              crop, resize, rotate, transform, paste, convert: the rules at the Python level
//   src/_imaging.c                _resize (its box checks, the crop shortcut, "nearest" as an affine scale)
//   src/libImaging/Resample.c     the two-pass convolution resize and its fixed-point coefficients
//   src/libImaging/Geometry.c     the transposes; affine transforms with nearest, bilinear and bicubic
//   src/libImaging/Crop.c, Paste.c    crop (black outside the picture) and paste (clipped)
//   src/libImaging/Convert.c      rgb2l
// Pillow is Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and
// contributors, Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors, under the MIT-CMU licence (HPND).
//
// Each function does the C's arithmetic in the C's order: doubles where it uses doubles, float32 where it uses
// floats (the resize box), integers where it uses ints, so the bytes come out the same. One number is not
// Pillow's to give: rotate takes cos and sin from Python's math module, that is from the C library, and rounds
// them to 15 decimals. V8's Math.cos and Math.sin are one unit in the last place away from the true value often
// enough to change that rounding for about 1 angle in 175, so the cos and sin here are correctly rounded
// (double-double arithmetic). glibc is itself correctly rounded for all but about 1 value in 700, so the matrix
// can still differ from Linux Pillow's, for about 1 angle in 5000 and then by 1e-15: the pixels almost never.
//
// Pictures are packed rows, top to bottom (RgbImage, GrayImage in types.ts). Pillow keeps RGB as four bytes a
// pixel; every operation here treats the channels one by one as it does, so the fourth byte never matters.

import type { GrayImage, Resample, RgbImage } from './types';

type Box = readonly [number, number, number, number];
type Channels = 1 | 3;
type Filter = Exclude<Resample, 'nearest'>;
type Matrix = readonly [number, number, number, number, number, number];

/** The byte order 32-bit words are read in: little-endian on every machine the extension runs on. The word-wise
 * loops below assume it and fall back to byte loops if not. */
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** What RgbImage and GrayImage both are; the channel count travels beside it. */
interface Pic {
  width: number;
  height: number;
  data: Uint8Array;
}

// ---- pictures ----

/** A black picture of this size, or one that wraps `data` (width * height * 3 bytes). */
export function rgbImage(width: number, height: number, data?: Uint8Array): RgbImage {
  const size = width * height * 3;
  if (data && data.length !== size) throw new Error(`an RGB picture of ${width} x ${height} needs ${size} bytes, not ${data.length}`);
  return { width, height, data: data ?? new Uint8Array(size) };
}

/** Canvas pixels (RGBA, as getImageData gives them) as an RGB picture: the alpha is dropped, as Pillow's
 * convert("RGB") of an RGBA picture drops it. */
export function fromRgba(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): RgbImage {
  const n = width * height;
  if (rgba.length !== n * 4) throw new Error(`an RGBA picture of ${width} x ${height} needs ${n * 4} bytes, not ${rgba.length}`);
  const out = new Uint8Array(n * 3);
  let done = 0;
  if (LITTLE_ENDIAN && rgba.byteOffset % 4 === 0) {
    // four pixels at a time: four RGBA words in, three RGB words out
    const src = new Uint32Array(rgba.buffer, rgba.byteOffset, n);
    const dst = new Uint32Array(out.buffer, 0, (n >> 2) * 3);
    for (let i = 0, j = 0; j + 3 < n; i += 3, j += 4) {
      const p1 = src[j + 1]!;
      const p2 = src[j + 2]!;
      dst[i] = (src[j]! & 0xffffff) | (p1 << 24);
      dst[i + 1] = ((p1 >>> 8) & 0xffff) | (p2 << 16);
      dst[i + 2] = ((p2 >>> 16) & 0xff) | (src[j + 3]! << 8);
    }
    done = n & ~3;
  }
  for (let i = done * 3, j = done * 4; i < out.length; i += 3, j += 4) {
    out[i] = rgba[j]!;
    out[i + 1] = rgba[j + 1]!;
    out[i + 2] = rgba[j + 2]!;
  }
  return { width, height, data: out };
}

function check(im: Pic, ch: Channels, what: string): void {
  const { width: w, height: h } = im;
  if (!Number.isInteger(w) || !Number.isInteger(h) || w < 0 || h < 0 || im.data.length !== w * h * ch) {
    const kind = ch === 3 ? 'an RGB' : 'a grey';
    throw new Error(`${what}: ${kind} picture of ${w} x ${h} needs ${w * h * ch} bytes, not ${im.data.length}`);
  }
}

function copyOf(im: Pic): Pic {
  return { width: im.width, height: im.height, data: im.data.slice() };
}

/** Python's round(x) of a float: the nearest integer, halves to the even one. */
function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  const r = d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
  return r + 0; // never -0
}

// ---- crop, paste, grey ----

/** Pillow Image.crop: the box's pixels, where parts outside the picture are black. Each edge is rounded as
 * Python rounds (halves to even), so (10.5, 0, 20.5, 5) is the box (10, 0, 20, 5). */
export function crop(im: RgbImage, box: Box): RgbImage {
  check(im, 3, 'crop');
  return cropBox(im, 3, box);
}

/** The same for a one-channel picture (matcrops.detail crops its grey copy). */
export function cropGray(im: GrayImage, box: Box): GrayImage {
  check(im, 1, 'cropGray');
  return cropBox(im, 1, box);
}

function cropBox(im: Pic, ch: Channels, box: Box): Pic {
  const [x0, y0, x1, y1] = edges(box);
  return cropInts(im, ch, x0, y0, x1, y1);
}

/** Image.crop and Image._crop: the checks, then map(int, map(round, box)). */
function edges(box: Box): [number, number, number, number] {
  if (!box.every(Number.isFinite)) throw new Error(`crop: the box (${box.join(', ')}) must be finite`);
  const [left, upper, right, lower] = box;
  if (right < left) throw new Error("crop: coordinate 'right' is less than 'left'");
  if (lower < upper) throw new Error("crop: coordinate 'lower' is less than 'upper'");
  return [pyRound(left), pyRound(upper), pyRound(right), pyRound(lower)];
}

/** libImaging/Crop.c ImagingCrop: a new picture of the box's size, black where the box leaves the picture. */
function cropInts(im: Pic, ch: Channels, x0: number, y0: number, x1: number, y1: number): Pic {
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const out = new Uint8Array(w * h * ch);
  const ix0 = Math.max(x0, 0);
  const ix1 = Math.min(x1, im.width);
  const iy1 = Math.min(y1, im.height);
  if (ix1 > ix0) {
    for (let y = Math.max(y0, 0); y < iy1; y++) {
      const from = (y * im.width + ix0) * ch;
      out.set(im.data.subarray(from, from + (ix1 - ix0) * ch), ((y - y0) * w + ix0 - x0) * ch);
    }
  }
  return { width: w, height: h, data: out };
}

/** Pillow Image.paste(src, (x, y)) of a whole RGB picture onto another, clipped to `dst`, in place
 * (libImaging/Paste.c ImagingPaste with no mask). The corner must be whole pixels, as Pillow requires. */
export function paste(dst: RgbImage, src: RgbImage, at: readonly [number, number]): void {
  check(dst, 3, 'paste');
  check(src, 3, 'paste');
  const [dx, dy] = at;
  if (!Number.isInteger(dx) || !Number.isInteger(dy)) throw new Error(`paste: the corner (${dx}, ${dy}) must be whole pixels`);
  // A picture pasted onto itself is read as it was before (Pillow copies bottom-up when the rows overlap).
  const from = src.data.buffer === dst.data.buffer ? src.data.slice() : src.data;
  const sx = Math.max(0, -dx);
  const sy = Math.max(0, -dy);
  const w = Math.min(src.width, dst.width - dx) - sx;
  const h = Math.min(src.height, dst.height - dy) - sy;
  if (w <= 0 || h <= 0) return;
  for (let y = 0; y < h; y++) {
    const s = ((sy + y) * src.width + sx) * 3;
    dst.data.set(from.subarray(s, s + w * 3), ((dy + sy + y) * dst.width + dx + sx) * 3);
  }
}

/** Pillow Image.convert("L"): ITU-R 601-2 luma in 16-bit fixed point, as Convert.c rgb2l rounds it. */
export function toGray(im: RgbImage): GrayImage {
  check(im, 3, 'toGray');
  const src = im.data;
  const n = im.width * im.height;
  const out = new Uint8Array(n);
  let done = 0;
  if (LITTLE_ENDIAN && src.byteOffset % 4 === 0) {
    // four pixels at a time: three words of RGB in, one word of grey out
    const words = new Uint32Array(src.buffer, src.byteOffset, (n >> 2) * 3);
    const grey = new Uint32Array(out.buffer, 0, n >> 2);
    for (let i = 0, j = 0; i < grey.length; i++, j += 3) {
      const w0 = words[j]!;
      const w1 = words[j + 1]!;
      const w2 = words[j + 2]!;
      const g0 = luma(w0 & 255, (w0 >>> 8) & 255, (w0 >>> 16) & 255);
      const g1 = luma(w0 >>> 24, w1 & 255, (w1 >>> 8) & 255);
      const g2 = luma((w1 >>> 16) & 255, w1 >>> 24, w2 & 255);
      const g3 = luma((w2 >>> 8) & 255, (w2 >>> 16) & 255, w2 >>> 24);
      grey[i] = g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
    }
    done = n & ~3;
  }
  for (let i = done, j = done * 3; i < n; i++, j += 3) out[i] = luma(src[j]!, src[j + 1]!, src[j + 2]!);
  return { width: im.width, height: im.height, data: out };
}

/** Convert.c L24: (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16. */
function luma(r: number, g: number, b: number): number {
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

// ---- resize ----

/** Pillow Image.resize(size, resample) with no reducing_gap: the same bytes, filter for filter. */
export function resize(im: RgbImage, size: readonly [number, number], resample: Resample): RgbImage {
  check(im, 3, 'resize');
  return resizePic(im, 3, size, resample);
}

/** The same for a one-channel picture (Pillow "L"). */
export function resizeGray(im: GrayImage, size: readonly [number, number], resample: Resample): GrayImage {
  check(im, 1, 'resizeGray');
  return resizePic(im, 1, size, resample);
}

const RESAMPLES: readonly Resample[] = ['nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'];

/** Image.resize: the same size is a copy; a picture over 100 times taller than wide that gets shorter is
 * resized in two steps, the height first (new in Pillow 12); anything else is one core resize. */
function resizePic(im: Pic, ch: Channels, size: readonly [number, number], resample: Resample): Pic {
  const [w, h] = size;
  if (!Number.isInteger(w) || !Number.isInteger(h)) throw new Error(`resize: the size (${w}, ${h}) must be whole pixels`);
  if (!RESAMPLES.includes(resample)) throw new Error(`resize: unknown resampling filter ${String(resample)}`);
  if (w === im.width && h === im.height) return copyOf(im);
  if (im.height > im.width * 100 && h < im.height) {
    const tall = coreResize(im, ch, im.width, h, resample, [0, 0, im.width, im.height]);
    return coreResize(tall, ch, w, h, resample, [0, 0, im.width, h]);
  }
  return coreResize(im, ch, w, h, resample, [0, 0, im.width, im.height]);
}

/** _imaging.c _resize. The box arrives as C floats. A whole-pixel box of the output's size is a crop,
 * "nearest" is an affine scale (Geometry.c), and the other filters convolve (Resample.c). */
function coreResize(im: Pic, ch: Channels, xsize: number, ysize: number, resample: Resample, box: Box): Pic {
  const b0 = Math.fround(box[0]);
  const b1 = Math.fround(box[1]);
  const b2 = Math.fround(box[2]);
  const b3 = Math.fround(box[3]);
  if (xsize < 1 || ysize < 1) throw new Error('resize: height and width must be > 0');
  if (b0 < 0 || b1 < 0) throw new Error("resize: box offset can't be negative");
  if (b2 > im.width || b3 > im.height) throw new Error("resize: box can't exceed original image size");
  const bw = Math.fround(b2 - b0);
  const bh = Math.fround(b3 - b1);
  if (bw < 0 || bh < 0) throw new Error("resize: box can't be empty");
  if (b0 - Math.trunc(b0) === 0 && bw === xsize && b1 - Math.trunc(b1) === 0 && bh === ysize) {
    return cropInts(im, ch, Math.trunc(b0), Math.trunc(b1), Math.trunc(b2), Math.trunc(b3));
  }
  if (resample === 'nearest') {
    const out: Pic = { width: xsize, height: ysize, data: new Uint8Array(xsize * ysize * ch) };
    scaleAffine(out, im, ch, bw / xsize, b0, bh / ysize, b1);
    return out;
  }
  return resample2Pass(im, ch, xsize, ysize, FILTERS[resample], [b0, b1, b2, b3]);
}

// Resample.c's filters: each a weight function and its support (half-width) at scale 1.

interface FilterSpec {
  weight: (x: number) => number;
  support: number;
}

/** The C's 0.54f and 0.46f, float constants widened to double. */
const HAMMING_A = Math.fround(0.54);
const HAMMING_B = Math.fround(0.46);

function boxWeight(x: number): number {
  return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
}

function bilinearWeight(x: number): number {
  if (x < 0.0) x = -x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

function hammingWeight(x: number): number {
  if (x < 0.0) x = -x;
  if (x === 0.0) return 1.0;
  if (x >= 1.0) return 0.0;
  x = x * Math.PI;
  return (Math.sin(x) / x) * (HAMMING_A + HAMMING_B * Math.cos(x));
}

/** Keys' cubic with a = -0.5, written as the C expands its macro. */
function bicubicWeight(x: number): number {
  if (x < 0.0) x = -x;
  if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * -0.5;
  return 0.0;
}

function sinc(x: number): number {
  if (x === 0.0) return 1.0;
  x = x * Math.PI;
  return Math.sin(x) / x;
}

function lanczosWeight(x: number): number {
  return -3.0 <= x && x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0;
}

const FILTERS: Record<Filter, FilterSpec> = {
  box: { weight: boxWeight, support: 0.5 },
  bilinear: { weight: bilinearWeight, support: 1.0 },
  hamming: { weight: hammingWeight, support: 1.0 },
  bicubic: { weight: bicubicWeight, support: 2.0 },
  lanczos: { weight: lanczosWeight, support: 3.0 },
};

/** 8 bits of result, 22 of fraction, and two spare for the sums a filter's negative lobes let overshoot. */
const PRECISION_BITS = 32 - 8 - 2;
const HALF = 1 << (PRECISION_BITS - 1);

/** One axis's weights: output pixel i reads `count` input pixels from `first`, weighted by
 * k[i * ksize + j] / 2^22. */
interface Coeffs {
  ksize: number;
  bounds: Int32Array; // first, count for each output pixel
  k: Int32Array;
}

/** Resample.c precompute_coeffs, then normalize_coeffs_8bpc. `in0` and `in1` are float32 values. */
function coefficients(inSize: number, in0: number, in1: number, outSize: number, filter: FilterSpec): Coeffs {
  const scale = Math.fround(in1 - in0) / outSize;
  const filterscale = scale < 1.0 ? 1.0 : scale;
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Float64Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const invFilterscale = 1.0 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = in0 + (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const at = xx * ksize;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.weight((x + xmin - center + 0.5) * invFilterscale);
      kk[at + x] = w;
      ww += w;
    }
    if (ww !== 0.0) for (let x = 0; x < xmax; x++) kk[at + x] = kk[at + x]! / ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  const k = new Int32Array(kk.length);
  const one = 1 << PRECISION_BITS;
  for (let i = 0; i < kk.length; i++) {
    const v = kk[i]!;
    k[i] = v < 0 ? Math.trunc(-0.5 + v * one) : Math.trunc(0.5 + v * one);
  }
  return { ksize, bounds, k };
}

/** Resample.c ImagingResampleInner: the horizontal pass over the rows the vertical pass will read, rounded to
 * 8 bits, then the vertical pass; a pass whose axis keeps its size and box is skipped. */
function resample2Pass(im: Pic, ch: Channels, xsize: number, ysize: number, filter: FilterSpec, box: Box): Pic {
  const [b0, b1, b2, b3] = box;
  const needH = xsize !== im.width || b0 !== 0 || b2 !== xsize;
  const needV = ysize !== im.height || b1 !== 0 || b3 !== ysize;
  const vert = coefficients(im.height, b1, b3, ysize, filter);
  const firstRow = vert.bounds[0]!;
  const lastRow = vert.bounds[ysize * 2 - 2]! + vert.bounds[ysize * 2 - 1]!;
  let cur = im;
  if (needH) {
    const hor = coefficients(im.width, b0, b2, xsize, filter);
    for (let i = 0; i < ysize; i++) vert.bounds[i * 2] = vert.bounds[i * 2]! - firstRow;
    const rows = lastRow - firstRow;
    const tmp = new Uint8Array(xsize * rows * ch);
    (ch === 3 ? horizontal3 : horizontal1)(im.data, im.width, firstRow, tmp, xsize, rows, hor);
    cur = { width: xsize, height: rows, data: tmp };
  }
  if (needV) {
    const out = new Uint8Array(cur.width * ysize * ch);
    vertical(cur.data, cur.width * ch, out, ysize, vert);
    return { width: cur.width, height: ysize, data: out };
  }
  return cur === im ? copyOf(im) : cur;
}

// The passes below sum in 32-bit integers (Math.imul, | 0) as the C's ints do: the sums are the C's exactly, in
// whatever order they are added, and a store into a Uint8ClampedArray is clip8. For speed the horizontal passes
// do two rows at a time and the vertical pass four input rows at a time.

/** _ImagingResampleHorizontal_8bpc for RGB: each output pixel a fixed-point sum along its input row, rounded by
 * the half added up front, shifted down and clipped to 0..255 (clip8). */
function horizontal3(src: Uint8Array, srcW: number, firstRow: number, out: Uint8Array, xsize: number, rows: number, c: Coeffs): void {
  const { ksize, bounds, k } = c;
  const dst = new Uint8ClampedArray(out.buffer, out.byteOffset, out.length);
  const stride = srcW * 3;
  const outStride = xsize * 3;
  let yy = 0;
  for (; yy + 1 < rows; yy += 2) {
    const row = (yy + firstRow) * stride;
    for (let xx = 0, o = yy * outStride; xx < xsize; xx++, o += 3) {
      const kb = xx * ksize;
      const end = kb + bounds[xx * 2 + 1]!;
      let p = row + bounds[xx * 2]! * 3;
      let a0 = HALF;
      let a1 = HALF;
      let a2 = HALF;
      let b0 = HALF;
      let b1 = HALF;
      let b2 = HALF;
      for (let j = kb; j < end; j++, p += 3) {
        const w = k[j]!;
        const q = p + stride;
        a0 = (a0 + Math.imul(src[p]!, w)) | 0;
        a1 = (a1 + Math.imul(src[p + 1]!, w)) | 0;
        a2 = (a2 + Math.imul(src[p + 2]!, w)) | 0;
        b0 = (b0 + Math.imul(src[q]!, w)) | 0;
        b1 = (b1 + Math.imul(src[q + 1]!, w)) | 0;
        b2 = (b2 + Math.imul(src[q + 2]!, w)) | 0;
      }
      dst[o] = a0 >> PRECISION_BITS;
      dst[o + 1] = a1 >> PRECISION_BITS;
      dst[o + 2] = a2 >> PRECISION_BITS;
      dst[o + outStride] = b0 >> PRECISION_BITS;
      dst[o + outStride + 1] = b1 >> PRECISION_BITS;
      dst[o + outStride + 2] = b2 >> PRECISION_BITS;
    }
  }
  if (yy < rows) {
    const row = (yy + firstRow) * stride;
    for (let xx = 0, o = yy * outStride; xx < xsize; xx++, o += 3) {
      const kb = xx * ksize;
      const end = kb + bounds[xx * 2 + 1]!;
      let p = row + bounds[xx * 2]! * 3;
      let a0 = HALF;
      let a1 = HALF;
      let a2 = HALF;
      for (let j = kb; j < end; j++, p += 3) {
        const w = k[j]!;
        a0 = (a0 + Math.imul(src[p]!, w)) | 0;
        a1 = (a1 + Math.imul(src[p + 1]!, w)) | 0;
        a2 = (a2 + Math.imul(src[p + 2]!, w)) | 0;
      }
      dst[o] = a0 >> PRECISION_BITS;
      dst[o + 1] = a1 >> PRECISION_BITS;
      dst[o + 2] = a2 >> PRECISION_BITS;
    }
  }
}

/** The same for one channel. */
function horizontal1(src: Uint8Array, srcW: number, firstRow: number, out: Uint8Array, xsize: number, rows: number, c: Coeffs): void {
  const { ksize, bounds, k } = c;
  const dst = new Uint8ClampedArray(out.buffer, out.byteOffset, out.length);
  let yy = 0;
  for (; yy + 1 < rows; yy += 2) {
    const row = (yy + firstRow) * srcW;
    for (let xx = 0, o = yy * xsize; xx < xsize; xx++, o++) {
      const kb = xx * ksize;
      const end = kb + bounds[xx * 2 + 1]!;
      let p = row + bounds[xx * 2]!;
      let a = HALF;
      let b = HALF;
      for (let j = kb; j < end; j++, p++) {
        const w = k[j]!;
        a = (a + Math.imul(src[p]!, w)) | 0;
        b = (b + Math.imul(src[p + srcW]!, w)) | 0;
      }
      dst[o] = a >> PRECISION_BITS;
      dst[o + xsize] = b >> PRECISION_BITS;
    }
  }
  if (yy < rows) {
    const row = (yy + firstRow) * srcW;
    for (let xx = 0, o = yy * xsize; xx < xsize; xx++, o++) {
      const kb = xx * ksize;
      const end = kb + bounds[xx * 2 + 1]!;
      let a = HALF;
      for (let j = kb, p = row + bounds[xx * 2]!; j < end; j++, p++) a = (a + Math.imul(src[p]!, k[j]!)) | 0;
      dst[o] = a >> PRECISION_BITS;
    }
  }
}

/** _ImagingResampleVertical_8bpc: each output row a fixed-point sum of input rows. It works byte by byte, so one
 * loop serves any channel count, and adds up whole rows into an accumulator row, four input rows at a time. */
function vertical(src: Uint8Array, rowLen: number, out: Uint8Array, ysize: number, c: Coeffs): void {
  const { ksize, bounds, k } = c;
  const dst = new Uint8ClampedArray(out.buffer, out.byteOffset, out.length);
  const acc = new Int32Array(rowLen);
  for (let yy = 0; yy < ysize; yy++) {
    const kb = yy * ksize;
    const n = bounds[yy * 2 + 1]!;
    let r = bounds[yy * 2]! * rowLen;
    acc.fill(HALF);
    let j = 0;
    for (; j + 3 < n; j += 4, r += 4 * rowLen) {
      const w0 = k[kb + j]!;
      const w1 = k[kb + j + 1]!;
      const w2 = k[kb + j + 2]!;
      const w3 = k[kb + j + 3]!;
      const r1 = r + rowLen;
      const r2 = r1 + rowLen;
      const r3 = r2 + rowLen;
      for (let i = 0; i < rowLen; i++) {
        acc[i] = (acc[i]! + Math.imul(src[r + i]!, w0) + Math.imul(src[r1 + i]!, w1) + Math.imul(src[r2 + i]!, w2) + Math.imul(src[r3 + i]!, w3)) | 0;
      }
    }
    for (; j < n; j++, r += rowLen) {
      const w = k[kb + j]!;
      for (let i = 0; i < rowLen; i++) acc[i] = (acc[i]! + Math.imul(src[r + i]!, w)) | 0;
    }
    const o = yy * rowLen;
    for (let i = 0; i < rowLen; i++) dst[o + i] = acc[i]! >> PRECISION_BITS;
  }
}

// ---- rotate ----

export interface RotateOptions {
  resample?: 'nearest' | 'bilinear' | 'bicubic';
  expand?: boolean;
  center?: readonly [number, number];
  fill?: readonly [number, number, number];
}

/** Pillow Image.rotate(angle, resample, expand, center, fillcolor): counter-clockwise degrees, the same bytes.
 * With no centre, 0 is a copy, 180 a flip, and 90 or 270 a plain transpose when expanding or square, as in
 * Pillow; anything else is an affine transform of the rounded matrix (rotateMatrix). */
export function rotate(im: RgbImage, angle: number, opts: RotateOptions = {}): RgbImage {
  const { resample, fill } = rotation(im, angle, opts, 'rotate');
  const fast = transposed(im, angle, opts);
  if (fast) return fast;
  const { matrix, size } = rotateMatrix(im.width, im.height, angle, opts);
  const [w, h] = size;
  const out: Pic = { width: w, height: h, data: new Uint8Array(w * h * 3) };
  transformAffine({ out, ox: 0, oy: 0, x0: 0, y0: 0, x1: w, y1: h }, im, matrix, resample, fill);
  return out;
}

/** crop(rotate(im, angle, opts), box), the same bytes, computing only the box's pixels when the filter is
 * bilinear or bicubic: live/pipeline.card_crop turns a card's neighbourhood and keeps the card, about half. */
export function rotateCrop(im: RgbImage, angle: number, opts: RotateOptions, box: Box): RgbImage {
  const { resample, fill } = rotation(im, angle, opts, 'rotateCrop');
  const [bx0, by0, bx1, by1] = edges(box);
  const fast = transposed(im, angle, opts);
  if (fast || resample === 'nearest') return cropInts(fast ?? rotate(im, angle, opts), 3, bx0, by0, bx1, by1);
  const { matrix, size } = rotateMatrix(im.width, im.height, angle, opts);
  const w = Math.max(0, bx1 - bx0);
  const h = Math.max(0, by1 - by0);
  const out: Pic = { width: w, height: h, data: new Uint8Array(w * h * 3) };
  const win = { out, ox: bx0, oy: by0, x0: Math.max(bx0, 0), y0: Math.max(by0, 0), x1: Math.min(bx1, size[0]), y1: Math.min(by1, size[1]) };
  if (win.x1 > win.x0 && win.y1 > win.y0) transformAffine(win, im, matrix, resample, fill);
  return out;
}

function rotation(im: Pic, angle: number, opts: RotateOptions, what: string): { resample: 'nearest' | 'bilinear' | 'bicubic'; fill?: Uint8Array } {
  check(im, 3, what);
  const resample = opts.resample ?? 'nearest';
  if (resample !== 'nearest' && resample !== 'bilinear' && resample !== 'bicubic') {
    throw new Error(`${what}: resampling filter ${String(resample)} cannot be used; use nearest, bilinear or bicubic`);
  }
  if (!Number.isFinite(angle)) throw new Error(`${what}: the angle ${angle} must be finite`);
  return opts.fill === undefined ? { resample } : { resample, fill: inkOf(opts.fill) };
}

/** Image.rotate's fast paths, taken only without a centre: 0 is a copy, 180 a flip, 90 and 270 a transpose when
 * expanding or square. */
function transposed(im: Pic, angle: number, opts: RotateOptions): Pic | null {
  if (opts.center !== undefined) return null;
  const a = mod360(angle);
  if (a === 0) return copyOf(im);
  if (a === 180) return rotate180(im);
  if ((a === 90 || a === 270) && (opts.expand || im.width === im.height)) return a === 90 ? rotate90(im) : rotate270(im);
  return null;
}

/** The matrix Image.rotate hands Image.transform (from an output pixel's centre to the input point it
 * samples: x_in = m0 x + m1 y + m2, y_in = m3 x + m4 y + m5) and the output size. For mapping a point of a
 * rotated crop back to the picture it came from. */
export function rotateMatrix(width: number, height: number, angle: number, opts: Pick<RotateOptions, 'expand' | 'center'> = {}): {
  matrix: [number, number, number, number, number, number];
  size: [number, number];
} {
  let w = width;
  let h = height;
  const [cx, cy] = opts.center ?? [w / 2, h / 2];
  const t = -(mod360(angle) * DEG_TO_RAD);
  const [sin, cos] = sinCos(t);
  const c = round15(cos);
  const m: [number, number, number, number, number, number] = [c, round15(sin), 0.0, round15(-sin), c, 0.0];
  // matrix[2], matrix[5] = transform(-center[0] - 0, -center[1] - 0, matrix); then += center
  const px = -cx - 0;
  const py = -cy - 0;
  const m2 = m[0] * px + m[1] * py + m[2];
  const m5 = m[3] * px + m[4] * py + m[5];
  m[2] = m2 + cx;
  m[5] = m5 + cy;
  if (opts.expand) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const [x, y] of [[0, 0], [w, 0], [w, h], [0, h]] as const) {
      xs.push(m[0] * x + m[1] * y + m[2]);
      ys.push(m[3] * x + m[4] * y + m[5]);
    }
    const nw = Math.ceil(Math.max(...xs)) - Math.floor(Math.min(...xs));
    const nh = Math.ceil(Math.max(...ys)) - Math.floor(Math.min(...ys));
    // -(nw - w) / 2.0 with Python's integers, which have no -0
    const tx = (w - nw) / 2 + 0;
    const ty = (h - nh) / 2 + 0;
    const e2 = m[0] * tx + m[1] * ty + m[2];
    const e5 = m[3] * tx + m[4] * ty + m[5];
    m[2] = e2;
    m[5] = e5;
    w = nw;
    h = nh;
  }
  return { matrix: m, size: [w, h] };
}

/** math.radians's constant, Py_MATH_PI / 180.0. */
const DEG_TO_RAD = Math.PI / 180;

/** Python's angle % 360.0: fmod, then moved to the divisor's sign; a zero is +0. */
function mod360(angle: number): number {
  const m = angle % 360;
  if (m === 0) return 0;
  return m < 0 ? m + 360 : m;
}

/** Python's round(x, 15): the nearest multiple of 1e-15 to x's exact value, halves to even (dtoa mode 3), read
 * back correctly rounded. toFixed rounds the exact value too but sends halves up; a half is exactly a double
 * with 16 decimals ending in 5, that is x * 2^16 odd, and is settled with BigInt. */
function round15(x: number): number {
  if (x === 0 || !Number.isFinite(x)) return x;
  const ax = Math.abs(x);
  if (ax >= 2 ** 52) return x;
  const m = ax * 65536;
  let r: number;
  if (Number.isInteger(m) && m % 2 === 1) {
    const low = (BigInt(m) * 5n ** 15n - 1n) / 2n; // the two multiples of 1e-15 either side, in units of 1e-15
    r = parseFloat(`${low % 2n === 0n ? low : low + 1n}e-15`);
  } else {
    r = parseFloat(ax.toFixed(15));
  }
  return x < 0 ? -r : r;
}

/** Where a transform writes: output pixels x0 <= x < x1, y0 <= y < y1 of the whole output, into `out`, whose
 * first pixel is output pixel (ox, oy). A whole output is ox = oy = x0 = y0 = 0, x1 and y1 its size. */
interface Window {
  out: Pic;
  ox: number;
  oy: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Image.transform(size, AFFINE, matrix, resample, fillcolor), after Image.new has made the output (the fill
 * colour, black without one), then Geometry.c ImagingTransformAffine: bilinear and bicubic go through the
 * generic per-pixel transform; nearest is a scale when the matrix has no shear, else 16.16 fixed point while
 * every corner maps inside +-32768, else floating point. What maps outside the input keeps the fill. Nearest
 * steps from the output's first pixel, so it takes a whole output only. */
function transformAffine(win: Window, im: Pic, m: Matrix, resample: 'nearest' | 'bilinear' | 'bicubic', fill?: Uint8Array): void {
  const { out, x1: w, y1: h } = win;
  if (fill && (fill[0] || fill[1] || fill[2])) {
    for (let y = win.y0; y < win.y1; y++) {
      for (let x = win.x0, o = ((y - win.oy) * out.width + x - win.ox) * 3; x < win.x1; x++, o += 3) out.data.set(fill, o);
    }
  }
  if (resample === 'bicubic') bicubicAffine(win, im, m);
  else if (resample === 'bilinear') bilinearAffine(win, im, m);
  else if (m[1] === 0 && m[3] === 0) scaleAffine(out, im, 3, m[0], m[2], m[4], m[5]);
  else if (fitsFixed(m, 0, 0) && fitsFixed(m, w, h) && fitsFixed(m, 0, h) && fitsFixed(m, w, 0)) fixedAffine(out, im, m);
  else floatAffine(out, im, m);
}

/** _imaging.c getink for an RGB tuple: whole numbers, clipped to 0..255. */
function inkOf(fill: readonly [number, number, number]): Uint8Array {
  if (fill.length !== 3 || !fill.every(Number.isInteger)) throw new Error(`rotate: the fill (${fill.join(', ')}) must be three whole numbers`);
  return Uint8Array.from(fill, (v) => Math.min(255, Math.max(0, v)));
}

/** Geometry.c ImagingScaleAffine: nearest neighbour with no shear, the source column of each output column
 * tabulated by adding a0 again and again (as the C adds it), rows likewise with a4. */
function scaleAffine(out: Pic, im: Pic, ch: Channels, a0: number, a2: number, a4: number, a5: number): void {
  const { width: xsize, height: ysize, data: dst } = out;
  const src = im.data;
  const xintab = new Int32Array(xsize);
  let xo = a2 + a0 * 0.5;
  let yo = a5 + a4 * 0.5;
  let xmin = xsize;
  let xmax = 0;
  for (let x = 0; x < xsize; x++) {
    const xin = xo < 0.0 ? -1 : Math.trunc(xo);
    if (xin >= 0 && xin < im.width) {
      xmax = x + 1;
      if (x < xmin) xmin = x;
      xintab[x] = xin;
    }
    xo += a0;
  }
  for (let y = 0; y < ysize; y++) {
    const yi = yo < 0.0 ? -1 : Math.trunc(yo);
    if (yi >= 0 && yi < im.height) {
      const row = yi * im.width * ch;
      const o = y * xsize * ch;
      if (ch === 1) {
        for (let x = xmin; x < xmax; x++) dst[o + x] = src[row + xintab[x]!]!;
      } else {
        for (let x = xmin; x < xmax; x++) {
          const s = row + xintab[x]! * 3;
          const d = o + x * 3;
          dst[d] = src[s]!;
          dst[d + 1] = src[s + 1]!;
          dst[d + 2] = src[s + 2]!;
        }
      }
    }
    yo += a4;
  }
}

/** Geometry.c check_fixed. */
function fitsFixed(m: Matrix, x: number, y: number): boolean {
  return Math.abs(x * m[0] + y * m[1] + m[2]) < 32768.0 && Math.abs(x * m[3] + y * m[4] + m[5]) < 32768.0;
}

/** Geometry.c FIX: to 16.16 fixed point, FLOOR(v * 65536 + 0.5). */
function fix(v: number): number {
  return Math.floor(v * 65536.0 + 0.5) | 0;
}

/** Geometry.c affine_fixed: nearest neighbour stepping 16.16 fixed-point coordinates (32-bit ints). */
function fixedAffine(out: Pic, im: Pic, m: Matrix): void {
  const { width: w, height: h, data: dst } = out;
  const { width: W, height: H, data: src } = im;
  const a0 = fix(m[0]);
  const a1 = fix(m[1]);
  const a3 = fix(m[3]);
  const a4 = fix(m[4]);
  let a2 = fix(m[2] + m[0] * 0.5 + m[1] * 0.5);
  let a5 = fix(m[5] + m[3] * 0.5 + m[4] * 0.5);
  for (let y = 0; y < h; y++) {
    let xx = a2;
    let yy = a5;
    for (let x = 0, o = y * w * 3; x < w; x++, o += 3) {
      const xin = xx >> 16;
      if (xin >= 0 && xin < W) {
        const yin = yy >> 16;
        if (yin >= 0 && yin < H) {
          const s = (yin * W + xin) * 3;
          dst[o] = src[s]!;
          dst[o + 1] = src[s + 1]!;
          dst[o + 2] = src[s + 2]!;
        }
      }
      xx = (xx + a0) | 0;
      yy = (yy + a3) | 0;
    }
    a2 = (a2 + a1) | 0;
    a5 = (a5 + a4) | 0;
  }
}

/** The floating-point nearest neighbour of ImagingTransformAffine, for matrices too large for 16.16. */
function floatAffine(out: Pic, im: Pic, m: Matrix): void {
  const { width: w, height: h, data: dst } = out;
  const { width: W, height: H, data: src } = im;
  let xo = m[2] + m[1] * 0.5 + m[0] * 0.5;
  let yo = m[5] + m[4] * 0.5 + m[3] * 0.5;
  for (let y = 0; y < h; y++) {
    let xx = xo;
    let yy = yo;
    for (let x = 0, o = y * w * 3; x < w; x++, o += 3) {
      const xin = xx < 0.0 ? -1 : Math.trunc(xx);
      if (xin >= 0 && xin < W) {
        const yin = yy < 0.0 ? -1 : Math.trunc(yy);
        if (yin >= 0 && yin < H) {
          const s = (yin * W + xin) * 3;
          dst[o] = src[s]!;
          dst[o + 1] = src[s + 1]!;
          dst[o + 2] = src[s + 2]!;
        }
      }
      xx += m[0];
      yy += m[3];
    }
    xo += m[1];
    yo += m[4];
  }
}

/** Where each output pixel's centre lands in the input, as Geometry.c affine_transform computes it:
 * a0 (x + 0.5) + a1 (y + 0.5) + a2. The products by x are the same doubles whichever row, so they are made
 * once, for x0 <= x < x1; the sums keep the C's order. */
function columnTerms(x0: number, x1: number, a: number): Float64Array {
  const t = new Float64Array(x1 - x0);
  for (let x = x0; x < x1; x++) t[x - x0] = a * (x + 0.5);
  return t;
}

/** Geometry.c bilinear_filter32RGB in ImagingGenericTransform. */
function bilinearAffine(win: Window, im: Pic, m: Matrix): void {
  const { out, ox, oy, x0: wx0, y0: wy0, x1: wx1, y1: wy1 } = win;
  const dst = out.data;
  const { width: W, height: H, data: src } = im;
  const [a0, a1, a2, a3, a4, a5] = m;
  const tx = columnTerms(wx0, wx1, a0);
  const ty = columnTerms(wx0, wx1, a3);
  for (let y = wy0; y < wy1; y++) {
    const by = a1 * (y + 0.5);
    const ey = a4 * (y + 0.5);
    for (let i = 0, o = ((y - oy) * out.width + wx0 - ox) * 3; i < tx.length; i++, o += 3) {
      let xin = tx[i]! + by + a2;
      let yin = ty[i]! + ey + a5;
      if (xin < 0.0 || xin >= W || yin < 0.0 || yin >= H) continue;
      xin -= 0.5;
      yin -= 0.5;
      const ix = Math.floor(xin);
      const iy = Math.floor(yin);
      const dx = xin - ix;
      const dy = yin - iy;
      const x0 = clip(ix, W) * 3;
      const x1 = clip(ix + 1, W) * 3;
      const r0 = clip(iy, H) * W * 3;
      const r1 = iy + 1 >= 0 && iy + 1 < H ? (iy + 1) * W * 3 : -1;
      for (let b = 0; b < 3; b++) {
        const p0 = src[r0 + x0 + b]!;
        let v1 = p0 + (src[r0 + x1 + b]! - p0) * dx;
        let v2 = v1;
        if (r1 >= 0) {
          const q0 = src[r1 + x0 + b]!;
          v2 = q0 + (src[r1 + x1 + b]! - q0) * dx;
        }
        v1 = v1 + (v2 - v1) * dy;
        dst[o + b] = v1; // (UINT8)v1: v1 is in 0..255, and the store truncates
      }
    }
  }
}

/** Geometry.c bicubic_filter32RGB in ImagingGenericTransform: the 4 x 4 neighbourhood, columns clipped to the
 * picture, the first row clipped and each further row outside it repeating the one before; then clipped to
 * 0..255 and truncated. */
function bicubicAffine(win: Window, im: Pic, m: Matrix): void {
  const { out, ox, oy, x0: wx0, y0: wy0, x1: wx1, y1: wy1 } = win;
  const dst = out.data;
  const { width: W, height: H, data: src } = im;
  const [a0, a1, a2, a3, a4, a5] = m;
  const tx = columnTerms(wx0, wx1, a0);
  const ty = columnTerms(wx0, wx1, a3);
  const stride = W * 3;
  for (let y = wy0; y < wy1; y++) {
    const by = a1 * (y + 0.5);
    const ey = a4 * (y + 0.5);
    for (let i = 0, o = ((y - oy) * out.width + wx0 - ox) * 3; i < tx.length; i++, o += 3) {
      let xin = tx[i]! + by + a2;
      let yin = ty[i]! + ey + a5;
      if (xin < 0.0 || xin >= W || yin < 0.0 || yin >= H) continue;
      xin -= 0.5;
      yin -= 0.5;
      let ix = Math.floor(xin);
      let iy = Math.floor(yin);
      const dx = xin - ix;
      const dy = yin - iy;
      ix--;
      iy--;
      const x0 = clip(ix, W) * 3;
      const x1 = clip(ix + 1, W) * 3;
      const x2 = clip(ix + 2, W) * 3;
      const x3 = clip(ix + 3, W) * 3;
      const r0 = clip(iy, H) * stride;
      const r1 = iy + 1 >= 0 && iy + 1 < H ? (iy + 1) * stride : -1;
      const r2 = iy + 2 >= 0 && iy + 2 < H ? (iy + 2) * stride : -1;
      const r3 = iy + 3 >= 0 && iy + 3 < H ? (iy + 3) * stride : -1;
      for (let b = 0; b < 3; b++) {
        const v1 = cubicRow(src, r0 + b, x0, x1, x2, x3, dx);
        const v2 = r1 >= 0 ? cubicRow(src, r1 + b, x0, x1, x2, x3, dx) : v1;
        const v3 = r2 >= 0 ? cubicRow(src, r2 + b, x0, x1, x2, x3, dx) : v2;
        const v4 = r3 >= 0 ? cubicRow(src, r3 + b, x0, x1, x2, x3, dx) : v3;
        const v = cubic(v1, v2, v3, v4, dy);
        dst[o + b] = v <= 0.0 ? 0 : v >= 255.0 ? 255 : v;
      }
    }
  }
}

function clip(v: number, size: number): number {
  return v < 0 ? 0 : v < size ? v : size - 1;
}

/** One row of the bicubic: the C's BICUBIC macro on four bytes, whose sums are integers. */
function cubicRow(src: Uint8Array, row: number, x0: number, x1: number, x2: number, x3: number, d: number): number {
  const v1 = src[row + x0]!;
  const v2 = src[row + x1]!;
  const v3 = src[row + x2]!;
  const v4 = src[row + x3]!;
  const p2 = -v1 + v3;
  const p3 = 2 * (v1 - v2) + v3 - v4;
  const p4 = -v1 + v2 - v3 + v4;
  return v2 + d * (p2 + d * (p3 + d * p4));
}

/** Geometry.c's BICUBIC macro: p1 + d (p2 + d (p3 + d p4)), each p summed in the C's order. */
function cubic(v1: number, v2: number, v3: number, v4: number, d: number): number {
  const p2 = -v1 + v3;
  const p3 = 2 * (v1 - v2) + v3 - v4;
  const p4 = -v1 + v2 - v3 + v4;
  return v2 + d * (p2 + d * (p3 + d * p4));
}

/** Geometry.c ImagingRotate90: counter-clockwise, (x, y) -> (y, W - 1 - x). */
function rotate90(im: Pic): Pic {
  const { width: W, height: H, data: src } = im;
  const dst = new Uint8Array(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0, s = y * W * 3; x < W; x++, s += 3) {
      const d = ((W - 1 - x) * H + y) * 3;
      dst[d] = src[s]!;
      dst[d + 1] = src[s + 1]!;
      dst[d + 2] = src[s + 2]!;
    }
  }
  return { width: H, height: W, data: dst };
}

/** Geometry.c ImagingRotate270: clockwise, (x, y) -> (H - 1 - y, x). */
function rotate270(im: Pic): Pic {
  const { width: W, height: H, data: src } = im;
  const dst = new Uint8Array(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0, s = y * W * 3; x < W; x++, s += 3) {
      const d = (x * H + H - 1 - y) * 3;
      dst[d] = src[s]!;
      dst[d + 1] = src[s + 1]!;
      dst[d + 2] = src[s + 2]!;
    }
  }
  return { width: H, height: W, data: dst };
}

/** Geometry.c ImagingRotate180: (x, y) -> (W - 1 - x, H - 1 - y). */
function rotate180(im: Pic): Pic {
  const src = im.data;
  const dst = new Uint8Array(src.length);
  for (let s = 0, d = src.length - 3; s < src.length; s += 3, d -= 3) {
    dst[d] = src[s]!;
    dst[d + 1] = src[s + 1]!;
    dst[d + 2] = src[s + 2]!;
  }
  return { width: im.width, height: im.height, data: dst };
}

// ---- correctly rounded sin and cos ----
//
// Double-double arithmetic (Dekker, Knuth; the QD library's formulas): a number as hi + lo with |lo| at most half
// a unit in hi's last place, about 106 bits. Only rotateMatrix calls it, twice a rotation.

type DD = readonly [number, number];

function twoSum(a: number, b: number): DD {
  const s = a + b;
  const bb = s - a;
  return [s, a - (s - bb) + (b - bb)];
}

function quickTwoSum(a: number, b: number): DD {
  const s = a + b;
  return [s, b - (s - a)];
}

/** Veltkamp's split of a double into two 26-bit halves. */
function split(a: number): DD {
  const t = 134217729 * a; // 2^27 + 1
  const hi = t - (t - a);
  return [hi, a - hi];
}

/** a * b exactly, as the rounded product and its error (Dekker's product; JavaScript has no fma). */
function twoProd(a: number, b: number): DD {
  const p = a * b;
  const [ah, al] = split(a);
  const [bh, bl] = split(b);
  return [p, ah * bh - p + ah * bl + al * bh + al * bl];
}

function ddAdd(a: DD, b: DD): DD {
  const [s, e] = twoSum(a[0], b[0]);
  const [t, f] = twoSum(a[1], b[1]);
  const [s2, e2] = quickTwoSum(s, e + t);
  return quickTwoSum(s2, e2 + f);
}

function ddMul(a: DD, b: DD): DD {
  const [p, e] = twoProd(a[0], b[0]);
  return quickTwoSum(p, e + (a[0] * b[1] + a[1] * b[0]));
}

function ddDiv(a: DD, n: number): DD {
  const q1 = a[0] / n;
  const [p, pe] = twoProd(q1, n);
  const [s, se] = twoSum(a[0], -p);
  return quickTwoSum(q1, (s + (se + a[1] - pe)) / n);
}

/** pi / 2 as three doubles, 159 bits. */
const PIO2: readonly [number, number, number] = [1.5707963267948966, 6.123233995736766e-17, -1.4973849048591698e-33];

/** sin(t) and cos(t), each the double nearest the true value (for |t| up to a few turns): t less the nearest
 * multiple of pi / 2 in double-double, then the Taylor series of that remainder. */
function sinCos(t: number): [number, number] {
  if (t === 0) return [t, 1]; // sin keeps the zero's sign, as the C library's does
  const k = Math.round(t / PIO2[0]);
  const [ph, pl] = twoProd(k, PIO2[0]);
  const [qh, ql] = twoProd(k, PIO2[1]);
  let r = twoSum(t, -ph);
  r = ddAdd(r, [-pl, 0]);
  r = ddAdd(r, [-qh, 0]);
  r = ddAdd(r, [-ql - k * PIO2[2], 0]);
  const r2 = ddMul(r, r);
  let s: DD = r;
  let sTerm: DD = r;
  let c: DD = [1, 0];
  let cTerm: DD = [1, 0];
  for (let n = 1; n < 40; n++) {
    sTerm = ddDiv(ddMul(sTerm, r2), -(2 * n) * (2 * n + 1));
    cTerm = ddDiv(ddMul(cTerm, r2), -(2 * n - 1) * (2 * n));
    s = ddAdd(s, sTerm);
    c = ddAdd(c, cTerm);
    if (Math.abs(cTerm[0]) < 1e-40 && Math.abs(sTerm[0]) < 1e-40) break;
  }
  const q = ((k % 4) + 4) % 4;
  const sr = s[0] + s[1];
  const cr = c[0] + c[1];
  return q === 0 ? [sr, cr] : q === 1 ? [cr, -sr] : q === 2 ? [-sr, -cr] : [-cr, sr];
}
