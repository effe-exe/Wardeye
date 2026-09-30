// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The overlay's pure logic, no DOM: where the video's picture sits inside its element, and what each
// of the live runner's tracks draws and says. The state is the live runner's (ml/rifteye_ml/live/pipeline.py),
// whether it comes from the runner on this machine or from the engine inside the extension.

import type { Precision, Runtime } from './mode';
import type { Timing } from './timer';

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Guess {
  printing_id: string;
  name: string;
  p: number;
}

export interface Track {
  id: string;
  quad: [number, number][];
  side: string;
  state: 'new' | 'named' | 'unsure' | 'facedown' | string;
  printing_id: string | null;
  name: string;
  confidence: number;
  guesses: Guess[];
  kind: string;
  hidden: boolean;
  under?: { id: string; name: string; printing_id: string }[];
}

/** What only the engine inside the extension adds to the state. */
export interface EngineInfo {
  runtime: Runtime;
  /** The precision each model runs in. */
  detector: Precision;
  embedder: Precision;
  /** How often the overlay should send a frame (ms): the pace the boards are built for. */
  every_ms: number;
  reads_per_s: number;
  /** What the last frame cost. */
  timing: Timing;
  /** The layout in use: a preset's name, "auto" while it is being found. */
  layout: string;
  /** Why it does not run on WebGPU, when it does not (what the engine document found of the GPU, or why WebGPU would not load). */
  note?: string;
}

export interface State {
  t: number;
  status: string;
  message: string;
  frame: { width: number; height: number };
  tracks: Track[];
  fps?: { source: number; processed: number };
  latency_s?: number;
  /** Set when the board was read by the engine inside the extension (standalone mode). */
  engine?: EngineInfo;
  /** How long until the next frame is wanted (ms), when the engine has none to read now (it is loading). */
  retry_ms?: number;
}

/** The picture inside a <video> element of `box`: `object-fit: contain`, letterboxed on the short side. */
export function contentRect(box: Rect, videoWidth: number, videoHeight: number): Rect {
  if (!videoWidth || !videoHeight || !box.width || !box.height) return box;
  const scale = Math.min(box.width / videoWidth, box.height / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height };
}

/** The size to grab a frame at: the video's own, at most `maxWidth` wide (the runner reads 1080p best). */
export function captureSize(videoWidth: number, videoHeight: number, maxWidth = 1920): [number, number] {
  if (videoWidth <= maxWidth) return [videoWidth, videoHeight];
  return [maxWidth, Math.round((videoHeight * maxWidth) / videoWidth)];
}

/** Drawn at all: in sight, and not the detector's first sight of it. Runes are drawn but never labelled. */
export function drawn(track: Track): boolean {
  return !track.hidden;
}

function stateOf(track: Track): 'named' | 'unsure' | 'facedown' | 'new' {
  return track.state === 'named' || track.state === 'unsure' || track.state === 'facedown' ? track.state : 'new';
}

/** The card's own shape: what the pointer finds, the outline of the card under it, and the flash of one just named. */
export function boxClass(track: Track): string {
  return `rifteye-box rifteye-${stateOf(track)}${track.kind === 'rune' ? ' rifteye-rune' : ''}`;
}

/** The card's corner marks, in the colour of how sure the read is; face-down cards and runes have none. */
export function ticksClass(track: Track): string {
  return `rifteye-ticks rifteye-ticks-${track.kind === 'rune' ? 'rune' : stateOf(track)}`;
}

const at1 = (v: number): string => String(Math.round(v * 10) / 10);

/** A quad's corner marks, as one SVG path: at each corner, a short stroke along each of its two edges, a fifth of the edge
 * but between `min` and `max` frame pixels (and never past the edge's middle). They mark a card and leave its art in sight. */
export function corners(quad: [number, number][], share = 0.2, min = 6, max = 26): string {
  const n = quad.length;
  if (n < 3) return '';
  let d = '';
  for (let i = 0; i < n; i++) {
    const [x, y] = quad[i]!;
    const along = (j: number): string => {
      const [qx, qy] = quad[(j + n) % n]!;
      const len = Math.hypot(qx - x, qy - y);
      const k = len > 0 ? Math.min(Math.max(len * share, min), max, len / 2) / len : 0;
      return `${at1(x + (qx - x) * k)},${at1(y + (qy - y) * k)}`;
    };
    d += `M${along(i - 1)}L${at1(x)},${at1(y)}L${along(i + 1)}`;
  }
  return d;
}

/** A card that has just been named: named now, and not before (or not drawn before). It gets one pulse of its outline; runes never do. */
export function becameNamed(before: Track | undefined, now: Track): boolean {
  return now.state === 'named' && now.kind !== 'rune' && before?.state !== 'named';
}

/** The label over a box: the card's name when Wardeye is sure of it, else nothing. */
export function label(track: Track): string {
  return track.state === 'named' && track.kind !== 'rune' ? track.name : '';
}

/** The top edge's middle of a quad, where its label goes. */
export function labelAnchor(quad: [number, number][]): [number, number] {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, Math.min(...ys)];
}

/** The bottom edge's middle of a quad, where its label goes when another is in the way above. */
export function belowAnchor(quad: [number, number][]): [number, number] {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, Math.max(...ys)];
}

/** A card lying under another: each printing once, with how many of it there are. */
export interface Under {
  name: string;
  printing_id: string | null;
  count: number;
}

export type HoverCard =
  | { kind: 'named'; printing_id: string | null; name: string; meta: string; sure: string; confidence: number; under: Under[] }
  | { kind: 'unsure'; guesses: Guess[]; under: Under[] }
  | { kind: 'text'; text: string };

/** What lies under a card, one row a printing: three copies of the same gear are one row, counted. */
export function underOf(track: Track): Under[] {
  const rows: Under[] = [];
  for (const u of track.under ?? []) {
    const same = rows.find((r) => (u.printing_id ? r.printing_id === u.printing_id : r.name === u.name));
    if (same) same.count++;
    else rows.push({ name: u.name, printing_id: u.printing_id ?? null, count: 1 });
  }
  return rows;
}

/** How sure a read is, as a viewer reads it: "90% sure". */
export function sureText(confidence: number): string {
  return `${Math.round(Math.min(1, Math.max(0, confidence)) * 100)}% sure`;
}

/** The card types the state carries: it says a track's kind only for these (the runner's and the engine's KINDS). Any other
 * card is a plain "card", and its type is not in the data the overlay has. */
const TYPES: Readonly<Record<string, string>> = { legend: 'Legend', battlefield: 'Battlefield' };

/** What pointing at a card shows. Runes show nothing; a face-down card is never identified (D-005). The meta line is the card's
 * type, when the state has it; the state has no domains, so none are shown. */
export function hoverCard(track: Track): HoverCard | null {
  if (track.kind === 'rune') return null;
  const under = underOf(track);
  if (track.state === 'named')
    return { kind: 'named', printing_id: track.printing_id, name: track.name, meta: TYPES[track.kind] ?? '', sure: sureText(track.confidence), confidence: track.confidence, under };
  if (track.state === 'unsure') return { kind: 'unsure', guesses: track.guesses.slice(0, 3), under };
  if (track.state === 'facedown') return { kind: 'text', text: 'Face-down card: never identified' };
  return { kind: 'text', text: 'New card, not identified yet' };
}

/** The name the badge starts with. */
export const NAME = 'Wardeye';

/** The status badge on the player. */
export function badge(online: boolean, state: State | null): string {
  if (!online) return `${NAME}: start the live runner on this computer`;
  if (!state || state.status === 'starting') return state?.message ? `${NAME}: ${state.message}` : `${NAME}: starting`;
  if (state.status === 'away') return `${NAME}: waiting for the table camera`;
  if (state.status === 'error') return `${NAME}: ${state.message}`;
  const named = state.tracks.filter((t) => t.state === 'named' && t.kind !== 'rune' && !t.hidden).length;
  const cards = `${NAME} · ${named} card${named === 1 ? '' : 's'} named`;
  return state.engine?.runtime === 'wasm' ? `${cards} · on the processor` : cards; // slower: the second line says why
}

/** The badge's text in two parts, for its two typefaces: the name (Space Grotesk) and the status after it (Inter). Put back
 * together they are the text again, punctuation and all. */
export function badgeParts(text: string): { name: string; status: string } {
  return text.startsWith(NAME) ? { name: NAME, status: text.slice(NAME.length) } : { name: '', status: text };
}

/** The badge's second line, of the engine, shown when the badge is pointed at: how fast it reads, how it runs and where a
 * frame's time goes (ms). */
export function badgeDetail(state: State | null): string {
  const e = state?.engine;
  if (!e) return '';
  const t = e.timing;
  const way = `${e.reads_per_s.toFixed(1)} reads/s · ${e.runtime === 'webgpu' ? 'WebGPU' : 'WASM'} · detector ${e.detector} · embedder ${e.embedder}`;
  const line = `${way} · decode ${t.decode} · detect ${t.detect} · embed ${t.embed} · track ${t.track} · total ${t.total} ms · layout ${e.layout}`;
  return e.note ? `${line} · not WebGPU: ${e.note}` : line;
}

/** How often the overlay sends a frame (ms): what the engine asks (it says more when it is loading), else its own
 * pace when it reads them, else the live runner's. */
export function frameInterval(state: State | null, fallback = 250): number {
  const ms = state?.retry_ms ?? state?.engine?.every_ms;
  return typeof ms === 'number' && ms >= 50 ? ms : fallback;
}
