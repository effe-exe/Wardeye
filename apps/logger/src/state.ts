// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Pure timeline-logging logic: no DOM, so it is unit-tested directly.

import {
  newId,
  validateTimelineDocument,
  type EventType,
  type MatchSource,
  type Player,
  type TimelineDocument,
  type TimelineEvent,
  type Zone,
} from '@rifteye/schema';

export const ENGINE = 'rifteye-logger/0.1.0';

/** Single-key shortcuts for the events a human logs while watching. */
export const HOTKEYS: Readonly<Record<string, EventType>> = {
  g: 'game_start',
  e: 'game_end',
  t: 'turn_start',
  u: 'runes_channeled',
  p: 'card_played',
  c: 'spell_cast',
  m: 'card_moved',
  h: 'card_hidden',
  r: 'card_revealed',
  x: 'card_left_play',
  s: 'score_changed',
};

/** Event types that name a card (the entry form asks for one). */
export const CARD_EVENTS: ReadonlySet<EventType> = new Set<EventType>([
  'card_played',
  'spell_cast',
  'card_moved',
  'card_left_play',
  'card_revealed',
  'card_exhausted',
  'card_readied',
]);

/** Where an event of each type usually happens; the logger pre-selects it. */
export function defaultZone(type: EventType): Zone | undefined {
  switch (type) {
    case 'card_played':
      return 'base';
    case 'spell_cast':
      return 'chain';
    case 'card_hidden':
    case 'card_revealed':
      return 'facedown_1';
    case 'runes_channeled':
      return 'runes';
    case 'card_left_play':
      return 'trash';
    default:
      return undefined;
  }
}

export interface LoggerState {
  doc: TimelineDocument;
}

export interface EventInput {
  t: number;
  type: EventType;
  player: Player | null;
  zone?: Zone;
  /** undefined = no card for this event type; null = a card was seen but not identified. */
  cardId?: string | null;
  score?: { A: number; B: number };
  note?: string;
}

export function createState(source: MatchSource, now: Date = new Date()): LoggerState {
  const iso = now.toISOString();
  return {
    doc: {
      schema: 'rifteye.timeline',
      version: 1,
      match: { matchId: newId(now.getTime()), source },
      events: [],
      createdAt: iso,
      updatedAt: iso,
    },
  };
}

function toEvent(matchId: string, id: string, input: EventInput): TimelineEvent {
  const ev: TimelineEvent = {
    id,
    matchId,
    t: Math.max(0, Math.round(input.t * 1000) / 1000),
    clock: 'media',
    type: input.type,
    player: input.player,
    evidence: ['manual'],
    engine: ENGINE,
  };
  if (input.zone !== undefined) ev.zone = input.zone;
  if (input.cardId !== undefined) ev.card = { cardId: input.cardId, confidence: input.cardId === null ? 0 : 1 };
  if (input.score !== undefined) ev.score = input.score;
  if (input.note !== undefined && input.note.trim() !== '') ev.note = input.note.trim();
  return ev;
}

const touch = (s: LoggerState, events: TimelineEvent[], now: Date): LoggerState => ({
  doc: { ...s.doc, events, updatedAt: now.toISOString() },
});

export function addEvent(s: LoggerState, input: EventInput, now: Date = new Date()): { state: LoggerState; id: string } {
  const id = newId(now.getTime());
  return { state: touch(s, [...s.doc.events, toEvent(s.doc.match.matchId, id, input)], now), id };
}

export function updateEvent(s: LoggerState, id: string, input: EventInput, now: Date = new Date()): LoggerState {
  return touch(s, s.doc.events.map((e) => (e.id === id ? toEvent(s.doc.match.matchId, id, input) : e)), now);
}

export function removeEvent(s: LoggerState, id: string, now: Date = new Date()): LoggerState {
  return touch(s, s.doc.events.filter((e) => e.id !== id), now);
}

/** Removes the most recently *created* event (ids are ULIDs, so they sort by creation). */
export function undoLast(s: LoggerState, now: Date = new Date()): LoggerState {
  if (s.doc.events.length === 0) return s;
  const last = s.doc.events.reduce((a, b) => (b.id > a.id ? b : a));
  return removeEvent(s, last.id, now);
}

export function sortedEvents(s: LoggerState): TimelineEvent[] {
  return [...s.doc.events].sort((a, b) => a.t - b.t || (a.id < b.id ? -1 : 1));
}

export function setMatchInfo(
  s: LoggerState,
  info: { title?: string; players?: { A?: string; B?: string }; legends?: { A?: string; B?: string } },
  now: Date = new Date(),
): LoggerState {
  const match = { ...s.doc.match };
  if (info.title !== undefined) match.title = info.title;
  if (info.players !== undefined) match.players = info.players;
  if (info.legends !== undefined) match.legends = info.legends;
  return { doc: { ...s.doc, match, updatedAt: now.toISOString() } };
}

/** The exported document, events in time order. */
export function toDocument(s: LoggerState): TimelineDocument {
  return { ...s.doc, events: sortedEvents(s) };
}

export function fromDocument(value: unknown): { state?: LoggerState; issues: string[] } {
  const issues = validateTimelineDocument(value).map((i) => `${i.path}: ${i.message}`);
  if (issues.length > 0) return { issues };
  return { state: { doc: value as TimelineDocument }, issues };
}

export function formatTime(t: number): string {
  const sign = t < 0 ? '-' : '';
  const a = Math.abs(t);
  const h = Math.floor(a / 3600);
  const m = Math.floor((a % 3600) / 60);
  const sec = (a % 60).toFixed(1).padStart(4, '0');
  return h > 0 ? `${sign}${h}:${String(m).padStart(2, '0')}:${sec}` : `${sign}${m}:${sec}`;
}
