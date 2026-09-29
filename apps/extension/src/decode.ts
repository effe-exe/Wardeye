// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A page's JPEG as RGB, decoded the way that gives Pillow's bytes: createImageBitmap with no colour conversion and
// no premultiplying, drawn on an OffscreenCanvas, read back with getImageData (checked on the LA final's frames,
// which Pillow and Chromium decode to the same RGB).

import { image, type RgbImage } from '@rifteye/engine';

let canvas: OffscreenCanvas | null = null;

export async function decodeJpeg(bytes: Uint8Array): Promise<RgbImage> {
  const bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: 'image/jpeg' }), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  try {
    const { width, height } = bitmap;
    if (!canvas || canvas.width !== width || canvas.height !== height) canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('no 2D canvas to read the frame with');
    ctx.drawImage(bitmap, 0, 0);
    return image.fromRgba(ctx.getImageData(0, 0, width, height).data, width, height);
  } finally {
    bitmap.close();
  }
}

/** A base64 string as bytes. */
export function bytesOfBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
