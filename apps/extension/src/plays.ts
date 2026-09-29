// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What the plays panel shows, kept as the live runner's page keeps it (ml/rifteye_ml/live/static/app.js): each player's legend
// and what lies face up on their side of the table, and the plays in the order they happened on the video. It lives in the tab,
// in memory: nothing is stored, and it goes when the tab closes or another video starts. It says nothing the overlay does not:
// the cards face up on the table, never a hand, a face-down card or a list (D-005).

import type { State } from './geometry';
import type { BoardEvent, ListSummary } from './parts';

/** A player as the recogniser names them: their half of the table, a label, and their legend once it is pinned. */
export interface Player {
  side: string;
  label: string;
  legend: { printing_id: string; name: string } | null;
}

/** Named cards of one printing on a player's side: how many, and the cards lying under them. */
export interface Group {
  printing_id: string | null;
  name: string;
  count: number;
  under: string[];
}

/** A player's side of the table. */
export interface Side extends Player {
  cards: Group[];
  runes: number;
  unsure: number;
  facedown: number;
}

/** The panel's view of one tab: the board and the plays, sent by the content script. */
export interface Snapshot {
  kind: 'board';
  /** The page the plays are of (its path), and its title. */
  video: string;
  title: string;
  /** Wardeye is on in the tab (off: nothing is read, the list stays). */
  on: boolean;
  /** A live stream: its plays cannot be jumped to. */
  live: boolean;
  status: string;
  message: string;
  sides: Side[];
  plays: BoardEvent[];
  /** The decklists pasted for this video, and what the engine made of each (null until a frame has been read with it). */
  lists: { text: string; read: ListSummary | null }[];
}

function players(state: State): Player[] {
  const raw = (state as { players?: unknown }).players;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((p: unknown): Player[] => {
    if (typeof p !== 'object' || p === null) return [];
    const { side, label, legend } = p as { side?: unknown; label?: unknown; legend?: unknown };
    if (typeof side !== 'string') return [];
    const l = legend as { printing_id?: unknown; name?: unknown } | null | undefined;
    return [{
      side,
      label: typeof label === 'string' ? label : side,
      legend: l && typeof l.printing_id === 'string' && typeof l.name === 'string' ? { printing_id: l.printing_id, name: l.name } : null,
    }];
  });
}

/** Each player's legend and the cards on their side: named cards grouped by printing (a card lying under another is listed with
 * it, not on its own), runes, unsure reads and face-down cards counted, legends left out (shown above as the player's own). A card
 * out of sight for a moment (a hand over it, the camera away) is still on the table, and listed. */
export function sidesOf(state: State | null): Side[] {
  if (!state) return [];
  const tracks = state.tracks;
  const under = new Set(tracks.flatMap((t) => (t.under ?? []).map((u) => u.id)));
  return players(state).map((p) => {
    const side: Side = { ...p, cards: [], runes: 0, unsure: 0, facedown: 0 };
    const groups = new Map<string, Group>();
    for (const t of tracks) {
      if (t.side !== p.side || under.has(t.id) || t.kind === 'legend') continue;
      if (t.kind === 'rune') side.runes++;
      else if (t.state === 'unsure') side.unsure++;
      else if (t.state === 'facedown') side.facedown++;
      else if (t.state === 'named') {
        const key = t.printing_id ?? t.name ?? t.id;
        let g = groups.get(key);
        if (!g) {
          g = { printing_id: t.printing_id, name: t.name || '(unnamed)', count: 0, under: [] };
          groups.set(key, g);
          side.cards.push(g);
        }
        g.count++;
        for (const u of t.under ?? []) if (!g.under.includes(u.name)) g.under.push(u.name);
      }
    }
    return side;
  });
}

/** The same play said twice: a card named again after its track was read afresh, or after a seek back replayed the moment. */
const AGAIN_S = 3;
/** The most plays one video keeps; the oldest go first. */
export const MAX_PLAYS = 500;

/** The plays of one video, in the order they happened on it. Another video starts a new list. */
export class PlayLog {
  private video: string | null = null;
  private list: BoardEvent[] = [];

  get plays(): readonly BoardEvent[] {
    return this.list;
  }

  /** The events of a frame of `video`; the ones new to the list are returned. */
  add(video: string, events: readonly BoardEvent[]): BoardEvent[] {
    if (video !== this.video) {
      this.video = video;
      this.list = [];
    }
    const fresh: BoardEvent[] = [];
    for (const e of events) {
      if (this.list.some((p) => p.kind === e.kind && p.text === e.text && p.side === e.side && Math.abs(p.t - e.t) <= AGAIN_S)) continue;
      let i = this.list.length;
      while (i > 0 && this.list[i - 1]!.t > e.t) i--; // after a seek back, a play goes where it happened
      this.list.splice(i, 0, e);
      fresh.push(e);
    }
    if (this.list.length > MAX_PLAYS) this.list.splice(0, this.list.length - MAX_PLAYS);
    return fresh;
  }
}

/** A time on the video's clock: 3:05, or 14:59:55 an hour or more in. */
export function clock(t: number): string {
  const s = Math.max(0, Math.floor(Number.isFinite(t) ? t : 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${r}` : `${m}:${r}`;
}

/** A snapshot's shape, checked: the panel does not trust what came over the wire. */
export function isSnapshot(x: unknown): x is Snapshot {
  if (typeof x !== 'object' || x === null) return false;
  const s = x as Partial<Snapshot>;
  return s.kind === 'board' && typeof s.video === 'string' && Array.isArray(s.sides) && Array.isArray(s.plays);
}
