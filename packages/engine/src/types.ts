// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The types the engine's parts share. Each mirrors the Python it is ported from (ml/rifteye_ml), so a port reads
// like its original and a fixture written by Python loads as it is.

/** An RGB picture: 8 bits a channel, rows top to bottom, no padding, so data.length === width * height * 3.
 * What a numpy uint8 [h, w, 3] array or a Pillow "RGB" image holds. */
export interface RgbImage {
  width: number;
  height: number;
  data: Uint8Array;
}

/** A one-channel 8-bit picture: Pillow "L". data.length === width * height. */
export interface GrayImage {
  width: number;
  height: number;
  data: Uint8Array;
}

/** Pillow's resampling filters, by their Python names. */
export type Resample = 'nearest' | 'box' | 'bilinear' | 'hamming' | 'bicubic' | 'lanczos';

/** matcrops.CardBox, plus what live/pipeline.detector_boxes adds: a card as a rotated rectangle in frame px. */
export interface CardBox {
  centre: [number, number];
  long_px: number;
  short_px: number;
  /** The direction of the long side, in degrees from the x axis, in [0, 180). */
  angle_deg: number;
  /** Blob area / rectangle area for the bootstrap finder; 1 for the detector's boxes. */
  fill: number;
  /** The detector's card_back: never identified (D-005). */
  back?: boolean;
  score?: number;
  /** Its least visible corner (the detector's per-corner visible probability). */
  vis?: number;
}

/** One card of detect.model.Detector.detect, in frame px. */
export interface Detection {
  cls: 'card' | 'card_back';
  score: number;
  /** x0, y0, x1, y1. */
  box: [number, number, number, number];
  /** The four corners, x and y. */
  quad: [number, number][];
  /** Per corner, the chance it was findable (inside the tile). */
  found: number[];
  /** Per corner, the chance it is visible (not covered). */
  visible: number[];
}

/** live/layouts.Layout: where a broadcast's table camera is and how big its cards are. */
export interface Layout {
  name: string;
  title: string;
  /** x0, y0, x1, y1 as fractions of the frame. */
  table: [number, number, number, number];
  /** A card's long side in px at 1080p. */
  card_long_1080: number;
  split: 'vertical' | 'horizontal';
  mask: 'notmat' | 'border';
  mat_tol: number;
  mat: [number, number, number] | null;
  mat_share: number;
}

/** One printing of the catalogue (catalog-plus.jsonl): the fields the engine reads, and whatever else a row holds. */
export interface CatalogRow {
  printing_id: string;
  card_id: string;
  name: string;
  /** Unit, Spell, Gear, Rune, Legend, Battlefield, Token, ... */
  type: string;
  [key: string]: unknown;
}

/** encoders.Encoder: pictures in, one L2-normalised row each out. Asynchronous, because the models run on
 * ONNX Runtime Web. */
export interface Encoder {
  /** What an embedding cache is keyed on: one name per model file. */
  readonly name: string;
  readonly dim: number;
  /** images.length x dim values, row by row. */
  embed(images: readonly RgbImage[]): Promise<Float32Array>;
}

/** What finds the cards in a frame: the trained detector, as live/pipeline.detector_boxes over Detector.detect. */
export type Finder = (t: number, frame: RgbImage) => Promise<CardBox[]>;
