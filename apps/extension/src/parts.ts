// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What the engine host needs from the engine, behind a few small interfaces: reading a JPEG, finding a table's
// layout, and a board (a Recognizer) for that table. The real parts are in parts-engine.ts (@rifteye/engine on
// onnxruntime-web); tests and the browser test plug in others.

import type { CardBox, Layout, RgbImage } from '@rifteye/engine';
import type { State } from './geometry';
import type { Attempt } from './mode';
import type { StandalonePackage, Gallery, Reader } from './assets';
import type { Timer } from './timer';

/** What the recogniser says of a frame: the live runner's state (ml/rifteye_ml/live/pipeline.py). */
export type BoardState = State & Record<string, unknown>;

/** Something that happened on the table: a card played, moved, left. */
export interface BoardEvent {
  t: number;
  kind: string;
  text: string;
  printing_id: string | null;
  track: string;
  side: string;
}

export interface BoardResult {
  state: BoardState;
  events: BoardEvent[];
}

/** What the engine made of a decklist the viewer pasted: the legends it names, how many cards it holds, and what it could not
 * read. `error` when it could not be read at all (a deck code or JSON it does not understand). */
export interface ListSummary {
  legends: string[];
  cards: number;
  unmapped: string[];
  error: string | null;
}

/** Decklists, read: what the boards are given (`Board.setLists`) and what the panel shows of each. */
export interface ReadLists {
  lists: unknown;
  summaries: ListSummary[];
}

/** A table's board: the Recognizer with its tracks. */
export interface Board {
  step(t: number, frame: RgbImage): Promise<BoardResult>;
  /** The decklists to hold its sides to (Parts.readLists): a side whose pinned legend a list names reads only that list's cards. */
  setLists?(lists: unknown): void;
  /** The frames the table was looked for in, before the board began (Recognizer.prime): a cut among them shows the overlay laid
   * over every shot (a co-streamer's webcam) from the start. */
  prime?(frames: readonly RgbImage[]): void;
}

export interface Parts {
  /** The JPEG a page sent, as RGB. */
  decode(jpeg: Uint8Array): Promise<RgbImage>;
  /** A layout from frames of the table camera (five, a second apart), or null while they show none. */
  findLayout(frames: readonly RgbImage[]): Promise<Layout | null>;
  /** The presets to fall back on. */
  presets(): readonly Layout[];
  /** A new board for a table, on the same models and gallery as every other. */
  board(layout: Layout): Board;
  /** Decklists the viewer pasted, read through the gallery's rows (none, when the parts cannot read lists). */
  readLists?(texts: readonly string[]): ReadLists;
  /** Frees the models. */
  dispose(): Promise<void>;
}

/** What loading the parts is given. `ort` is the onnxruntime-web module of the build this worker was made for. */
export interface LoadContext {
  ort: typeof import('onnxruntime-web');
  pkg: StandalonePackage;
  attempt: Attempt;
  read: Reader;
  /** A page's JPEG as RGB (the parity decode). */
  decode(jpeg: Uint8Array): Promise<RgbImage>;
  gallery: Gallery;
  /** Adds up the frame's time in the spans the engine wraps ('detect', 'embed'). */
  timer: Timer;
  /** The frames a second the boards are built for (the change gate's clock). */
  fps: number;
  /** Set for a check against a reference: told what the finder and the embedder give. */
  trace?: Trace;
  progress(message: string): void;
}

/** What the engine's finder and embedder gave, told as they give it (a browser replay checks it against Python's). */
export interface Trace {
  /** The finder's boxes for a frame, before the tracker sees them. */
  boxes(t: number, boxes: readonly CardBox[]): void;
  /** One embed() call: how many pictures it was given, and the rows it gave. */
  embed(count: number, rows: Float32Array): void;
}

/** The state of a table not read yet: the live runner's own "starting" payload. */
export function startingState(t: number, size: { width: number; height: number }, message: string, fps: number, title = 'Wardeye live'): BoardState {
  return {
    t,
    status: 'starting',
    message,
    title,
    frame: { width: size.width, height: size.height },
    fps: { source: fps, processed: 0 },
    latency_s: 0,
    players: [],
    tracks: [],
  };
}
