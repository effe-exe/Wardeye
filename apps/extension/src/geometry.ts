// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The overlay's pure logic, no DOM: where the video's picture sits inside its element, and what each
// of the live runner's tracks draws and says. The state is the live runner's (ml/rifteye_ml/live/pipeline.py).

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

export interface State {
  t: number;
  status: string;
  message: string;
  frame: { width: number; height: number };
  tracks: Track[];
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

export function boxClass(track: Track): string {
  const state = ['named', 'unsure', 'facedown'].includes(track.state) ? track.state : 'new';
  return `rifteye-box rifteye-${state}${track.kind === 'rune' ? ' rifteye-rune' : ''}`;
}

/** The label over a box: the card's name when RiftEye is sure of it, else nothing. */
export function label(track: Track): string {
  return track.state === 'named' && track.kind !== 'rune' ? track.name : '';
}

/** The top edge's middle of a quad, where its label goes. */
export function labelAnchor(quad: [number, number][]): [number, number] {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, Math.min(...ys)];
}

export type HoverCard =
  | { kind: 'named'; printing_id: string | null; name: string; sure: string; under: string }
  | { kind: 'unsure'; guesses: Guess[]; under: string }
  | { kind: 'text'; text: string };

/** What pointing at a card shows. Runes show nothing; a face-down card is never identified (D-005). */
export function hoverCard(track: Track): HoverCard | null {
  if (track.kind === 'rune') return null;
  const under = track.under?.length ? `Under it: ${track.under.map((u) => u.name).join(', ')}` : '';
  if (track.state === 'named')
    return { kind: 'named', printing_id: track.printing_id, name: track.name, sure: `RiftEye is ${Math.round(track.confidence * 100)}% sure`, under };
  if (track.state === 'unsure') return { kind: 'unsure', guesses: track.guesses.slice(0, 3), under };
  if (track.state === 'facedown') return { kind: 'text', text: 'Face-down card: never identified' };
  return { kind: 'text', text: 'New card, not identified yet' };
}

/** The status badge on the player. */
export function badge(online: boolean, state: State | null): string {
  if (!online) return 'RiftEye: start the runner on this computer (rifteye-overlay)';
  if (!state || state.status === 'starting') return state?.message ? `RiftEye: ${state.message}` : 'RiftEye: starting';
  if (state.status === 'away') return 'RiftEye: waiting for the table camera';
  if (state.status === 'error') return `RiftEye: ${state.message}`;
  const named = state.tracks.filter((t) => t.state === 'named' && t.kind !== 'rune' && !t.hidden).length;
  return `RiftEye · ${named} card${named === 1 ? '' : 's'} named`;
}
