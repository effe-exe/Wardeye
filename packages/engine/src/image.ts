// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The picture operations the engine shares, bit for bit as Pillow does them, so a port of ml/ sees exactly the
// pixels its Python original sees. PLACEHOLDER: the signatures are the contract; the image agent replaces these
// bodies with ports of Pillow's own code and tests them against Pillow.

import type { GrayImage, Resample, RgbImage } from './types';

const todo = (name: string): never => {
  throw new Error(`image.${name} is not written yet`);
};

/** A black picture of this size, or one that wraps `data` (width * height * 3 bytes). */
export function rgbImage(width: number, height: number, data?: Uint8Array): RgbImage {
  const size = width * height * 3;
  if (data && data.length !== size) throw new Error(`an RGB picture of ${width} x ${height} needs ${size} bytes, not ${data.length}`);
  return { width, height, data: data ?? new Uint8Array(size) };
}

/** Canvas pixels (RGBA, as getImageData gives them) as an RGB picture: the alpha is dropped. */
export function fromRgba(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): RgbImage {
  return todo('fromRgba');
}

/** Pillow Image.crop: the box's pixels, where parts outside the picture are black. */
export function crop(im: RgbImage, box: readonly [number, number, number, number]): RgbImage {
  return todo('crop');
}

/** Pillow Image.resize(size, resample) with no reducing_gap: the same bytes, filter for filter. */
export function resize(im: RgbImage, size: readonly [number, number], resample: Resample): RgbImage {
  return todo('resize');
}

/** The same for a one-channel picture (Pillow "L"). */
export function resizeGray(im: GrayImage, size: readonly [number, number], resample: Resample): GrayImage {
  return todo('resizeGray');
}

/** Pillow Image.rotate(angle, resample, expand, center, fillcolor): counter-clockwise degrees, the same bytes.
 * A multiple of 90 with expand is a plain transpose, as in Pillow. */
export function rotate(im: RgbImage, angle: number, opts?: {
  resample?: 'nearest' | 'bilinear' | 'bicubic';
  expand?: boolean;
  center?: readonly [number, number];
  fill?: readonly [number, number, number];
}): RgbImage {
  return todo('rotate');
}

/** Pillow Image.convert("L"): ITU-R 601-2 luma, as Pillow rounds it. */
export function toGray(im: RgbImage): GrayImage {
  return todo('toGray');
}

/** Pillow Image.paste(src, (x, y)) of a whole RGB picture onto another, clipped to `dst`, in place. */
export function paste(dst: RgbImage, src: RgbImage, at: readonly [number, number]): void {
  todo('paste');
}
