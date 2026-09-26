// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The demo bundle (ml/rifteye_ml/demo.py) and the pure logic that reads it: which cards are on
// screen at a moment, where their boxes are, and what the hover card says. No DOM here.

/** [t (s), centre x, centre y (fractions of the frame), long side, short side (fractions of its height), long-side angle (deg)] */
export type Sample = [number, number, number, number, number, number];

export interface Guess {
  card: string;
  p: number;
}

export interface Track {
  id: string;
  samples: Sample[];
  guesses?: Guess[];
  faceDown?: boolean;
}

export interface DemoEvent {
  t: number;
  tBefore: number;
  kind: 'played' | 'left' | 'changed';
  track?: string;
  guesses?: Guess[];
  box: [number, number, number, number];
}

export interface CardInfo {
  name: string;
  type: string;
  art: string;
  printing: string;
}

export interface DemoBundle {
  schema: 'rifteye.demo';
  version: 1;
  title: string;
  video: string;
  frame: [number, number];
  detectFps: number;
  cards: Record<string, CardInfo>;
  tracks: Track[];
  events: DemoEvent[];
}

export function checkBundle(value: unknown): string[] {
  const b = value as Partial<DemoBundle> | null;
  if (!b || typeof b !== 'object') return ['no demo data (data.js missing?)'];
  const issues: string[] = [];
  if (b.schema !== 'rifteye.demo') issues.push("schema must be 'rifteye.demo'");
  if (!Array.isArray(b.frame) || b.frame.length !== 2) issues.push('frame must be [width, height]');
  if (!Array.isArray(b.tracks)) issues.push('tracks must be an array');
  if (!Array.isArray(b.events)) issues.push('events must be an array');
  if (typeof b.cards !== 'object' || b.cards === null) issues.push('cards must be an object');
  if (typeof b.video !== 'string' || b.video === '') issues.push('video must be a file name');
  return issues;
}

/** Where a track's card is at `t`: interpolated between samples up to `hold` s apart, else absent. */
export function sampleAt(track: Track, t: number, hold = 0.6): Sample | undefined {
  const s = track.samples;
  if (s.length === 0 || t < s[0]![0] - hold || t > s[s.length - 1]![0] + hold) return undefined;
  let lo = 0;
  let hi = s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (s[mid]![0] <= t) lo = mid;
    else hi = mid - 1;
  }
  const a = s[lo]!;
  const b = s[lo + 1];
  if (a[0] > t) return t >= a[0] - hold ? a : undefined; // before the first sample
  if (!b) return t - a[0] <= hold ? a : undefined;
  const gap = b[0] - a[0];
  if (gap > 2 * hold) {
    // a gap (a hand over the card): show it only near either end
    if (t - a[0] <= hold) return a;
    if (b[0] - t <= hold) return b;
    return undefined;
  }
  const f = gap > 0 ? (t - a[0]) / gap : 0;
  const mix = (i: number) => a[i]! + (b[i]! - a[i]!) * f;
  // Angles near 0/180 wrap; take the nearer turn.
  let da = b[5] - a[5];
  if (da > 90) da -= 180;
  if (da < -90) da += 180;
  return [t, mix(1), mix(2), mix(3), mix(4), a[5] + da * f];
}

/** The rotated box's corners in frame pixels. */
export function corners(s: Sample, frame: [number, number]): [number, number][] {
  const [w, h] = frame;
  const cx = s[1] * w;
  const cy = s[2] * h;
  const a = (s[5] * Math.PI) / 180;
  const ux = (Math.cos(a) * s[3] * h) / 2;
  const uy = (Math.sin(a) * s[3] * h) / 2;
  const vx = (-Math.sin(a) * s[4] * h) / 2;
  const vy = (Math.cos(a) * s[4] * h) / 2;
  return [
    [cx + ux + vx, cy + uy + vy],
    [cx + ux - vx, cy + uy - vy],
    [cx - ux - vx, cy - uy - vy],
    [cx - ux + vx, cy - uy + vy],
  ];
}

/** Sure enough to name one card, or show the best three (ARCHITECTURE §3.5). */
export const SURE = 0.75;

export type HoverState =
  | { kind: 'sure'; card: CardInfo; p: number }
  | { kind: 'unsure'; options: { card: CardInfo; p: number }[] }
  | { kind: 'face-down' }
  | { kind: 'unknown' };

export function hoverState(bundle: DemoBundle, guesses: Guess[] | undefined, faceDown = false): HoverState {
  if (faceDown) return { kind: 'face-down' };
  const known = (guesses ?? []).flatMap((g) => (bundle.cards[g.card] ? [{ card: bundle.cards[g.card]!, p: g.p }] : []));
  if (known.length === 0) return { kind: 'unknown' };
  if (known[0]!.p >= SURE) return { kind: 'sure', card: known[0]!.card, p: known[0]!.p };
  return { kind: 'unsure', options: known.slice(0, 3) };
}

export function eventGuesses(bundle: DemoBundle, e: DemoEvent): Guess[] | undefined {
  return e.guesses ?? bundle.tracks.find((t) => t.id === e.track)?.guesses;
}

export const EVENT_WORDS: Record<DemoEvent['kind'], string> = { played: 'played', left: 'left the table', changed: 'turned or changed' };

/** The last event at or before `t`, for highlighting the timeline. */
export function currentEvent(events: readonly DemoEvent[], t: number): number {
  let k = -1;
  events.forEach((e, i) => {
    if (e.t <= t + 0.05) k = i;
  });
  return k;
}

export function formatTime(t: number): string {
  const s = Math.max(0, Math.floor(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
