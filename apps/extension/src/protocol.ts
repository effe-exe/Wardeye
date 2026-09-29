// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// What the parts of the extension say to each other.
//
//   content script <-port 'rifteye'-> worker: today's protocol, unchanged. 'frame' (the video's time, its page and a
//     base64 JPEG) is answered by 'state' (the board, or null to keep the one on screen); 'art' by the card's picture.
//   worker <-runtime message-> engine document: one request, one reply, each tagged `target: 'engine'`. The worker
//     holds nothing the engine needs, so it may be put to sleep and woken at any time; a tab's board lives in the
//     document, under the tab's id.
//   engine document <-postMessage-> engine worker: the same requests, with an id, and the answers.

import type { CardBox } from '@rifteye/engine';
import type { BoardEvent } from './parts';
import type { State } from './geometry';
import type { Attempt } from './mode';
import type { StandalonePackage } from './assets';

export interface FrameMessage {
  kind: 'frame';
  t: number;
  video: string;
  jpeg: string;
}

export interface ArtMessage {
  kind: 'art';
  printing_id: string;
}

export type FromContent = FrameMessage | ArtMessage;

export type ToContent =
  | { kind: 'state'; online: boolean; state: State | null }
  | { kind: 'art'; printing_id: string; jpeg: string | null };

/** Worker -> engine document. */
export type EngineRequest =
  | { target: 'engine'; kind: 'hello' }
  | { target: 'engine'; kind: 'frame'; tab: number; t: number; video: string; jpeg: string }
  | { target: 'engine'; kind: 'forget'; tab: number };

/** Engine document -> worker. */
export type EngineReply =
  /** The engine can run here; it is loading, or ready. */
  | { kind: 'hello' }
  /** The state to draw, or null when this frame was passed over (a newer one of the tab came): keep the board. */
  | { kind: 'state'; state: State | null }
  /** The engine cannot run here (no WebGPU, no models, a failed start): the live runner takes over. */
  | { kind: 'unavailable'; reason: string };

/** Engine document -> engine worker. */
export type ToEngine =
  | { kind: 'init'; pkg: StandalonePackage; attempt: Attempt }
  | { kind: 'frame'; id: number; tab: number; t: number; video: string; jpeg: string }
  | { kind: 'forget'; tab: number };

/** Engine worker -> engine document. */
export type FromEngine =
  | { kind: 'ready' }
  | { kind: 'failed'; error: string }
  | { kind: 'progress'; message: string }
  | { kind: 'state'; id: number; state: State; events: BoardEvent[] }
  | { kind: 'error'; id: number; error: string }
  /** For a check against a reference (standalone.json's `trace`): what the finder and the embedder gave for frame `id`. */
  | { kind: 'trace'; id: number; what: 'boxes'; t: number; boxes: CardBox[] }
  | { kind: 'trace'; id: number; what: 'embed'; count: number; rows: Float32Array };

/** A reply's kind, checked: what came over the wire is not trusted to have the shape. */
export function isEngineReply(x: unknown): x is EngineReply {
  if (typeof x !== 'object' || x === null) return false;
  const k = (x as { kind?: unknown }).kind;
  return k === 'hello' || k === 'state' || k === 'unavailable';
}
