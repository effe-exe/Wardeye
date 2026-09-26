// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// RiftEye data formats. Types mirror docs/ARCHITECTURE.md §6; every document carries a
// `schema` tag and a `version` so old files stay readable as the formats evolve.

export const SCHEMA_VERSION = 1 as const;

// ---------------------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------------------

export type Orientation = 'portrait' | 'landscape';

export const VARIANTS = ['standard', 'alt_art', 'overnumbered', 'showcase', 'signature', 'promo', 'token'] as const;
export type Variant = (typeof VARIANTS)[number];

/** One printed card face. Many printings share one gameplay card. */
export interface Printing {
  /** Stable collector code, e.g. "OGN-066" or "OGN-007a". */
  printingId: string;
  /** Gameplay identity shared by reprints, alternate arts and languages. */
  cardId: string;
  setCode: string;
  collectorNumber: string;
  variant: Variant;
  /** BCP-47 tag, e.g. "en", "zh-Hans". */
  language: string;
  /** The public source URL. RiftEye never stores or re-hosts card images (decision D-015). */
  imageUrl: string;
  orientation: Orientation;
}

export interface Card {
  cardId: string;
  name: string;
  /** "Unit", "Spell", "Gear", "Legend", "Battlefield", "Rune", ... */
  type: string;
  domains: string[];
  energy?: number;
  might?: number;
  printings: string[];
}

/** Shipped as manifest.json next to a row-major float16 matrix of `rows.length × dim`. */
export interface EmbeddingIndexManifest {
  schema: 'rifteye.index';
  version: typeof SCHEMA_VERSION;
  /** Bumped on any catalogue change. */
  catalogVersion: string;
  /** Encoder tag. Matching must refuse to run when it differs from the running encoder. */
  model: string;
  modelSha256: string;
  dim: number;
  dtype: 'float16';
  /** printingId for each matrix row, in order. */
  rows: string[];
  createdAt: string;
}

// ---------------------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------------------

export const EVENT_TYPES = [
  'game_start',
  'game_end',
  'turn_start',
  'runes_channeled',
  'score_changed',
  'card_played',
  'spell_cast',
  'card_moved',
  'card_exhausted',
  'card_readied',
  'card_left_play',
  'card_hidden',
  'card_revealed',
  'retracted',
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export const ZONES = [
  'legend',
  'champion',
  'base',
  'runes',
  'battlefield_1',
  'battlefield_2',
  'facedown_1',
  'facedown_2',
  'chain',
  'trash',
  'banishment',
  'unknown',
] as const;
export type Zone = (typeof ZONES)[number];

export const EVIDENCE = ['vision', 'graphic', 'audio', 'decklist', 'manual'] as const;
export type Evidence = (typeof EVIDENCE)[number];

export type Player = 'A' | 'B';

export interface CardRef {
  /** null = a card was seen but not identified. */
  cardId: string | null;
  printingId?: string;
  /** Calibrated probability in [0, 1]. Manual entries use 1. */
  confidence: number;
  alternatives?: { cardId: string; confidence: number }[];
}

export interface TimelineEvent {
  /** ULID: sortable by creation time. */
  id: string;
  matchId: string;
  /** Seconds on `clock`. */
  t: number;
  /** media = offset into the video; wall = Unix time in milliseconds (live). */
  clock: 'media' | 'wall';
  type: EventType;
  /** Controller. A = bottom of the overhead frame unless the layout says otherwise. */
  player: Player | null;
  zone?: Zone;
  card?: CardRef;
  /** Four corners (x, y) in normalised video coordinates. */
  quad?: [number, number, number, number, number, number, number, number];
  /** For 'retracted': the id of the event being withdrawn (a take-back). */
  retracts?: string;
  /** For 'score_changed'. */
  score?: { A: number; B: number };
  evidence: Evidence[];
  /** Producer, e.g. "rifteye-logger/0.1.0" or "rifteye/0.3.0+det-v4+emb-v7". */
  engine: string;
  note?: string;
}

export interface MatchSource {
  kind: 'file' | 'youtube' | 'twitch' | 'other';
  /** File name, video id or URL. */
  ref: string;
}

export interface MatchInfo {
  matchId: string;
  title?: string;
  source: MatchSource;
  players?: { A?: string; B?: string };
  /** cardId of each player's legend. */
  legends?: { A?: string; B?: string };
}

export interface TimelineDocument {
  schema: 'rifteye.timeline';
  version: typeof SCHEMA_VERSION;
  match: MatchInfo;
  events: TimelineEvent[];
  createdAt: string;
  updatedAt: string;
}

// ---------------------------------------------------------------------------------------
// Layout presets (layouts/<broadcaster>.json, CC0)
// ---------------------------------------------------------------------------------------

export const LAYOUT_REGIONS = ['overhead', 'closeup', 'graphic', 'handcam', 'scoreboard'] as const;
export type LayoutRegion = (typeof LAYOUT_REGIONS)[number];
/** [x0, y0, x1, y1] as fractions of the frame. */
export type Rect = [number, number, number, number];

export interface LayoutPreset {
  schema: 'rifteye.layout';
  version: typeof SCHEMA_VERSION;
  id: string;
  match: { channels?: string[]; youtubeChannelIds?: string[] };
  regions: Partial<Record<LayoutRegion, Rect>>;
  playerA: 'bottom' | 'top' | 'left' | 'right';
  /** ISO date; broadcasts change layouts between events. */
  validFrom?: string;
}

// ---------------------------------------------------------------------------------------
// Validation (dependency-free; returns every problem found, never throws)
// ---------------------------------------------------------------------------------------

export interface Issue {
  path: string;
  message: string;
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const isStr = (v: unknown): v is string => typeof v === 'string' && v.length > 0;
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const oneOf = <T extends readonly string[]>(list: T, v: unknown): v is T[number] =>
  typeof v === 'string' && (list as readonly string[]).includes(v);

export function validateTimelineEvent(value: unknown, path = 'event'): Issue[] {
  const issues: Issue[] = [];
  const bad = (field: string, message: string) => issues.push({ path: `${path}.${field}`, message });
  if (!isObj(value)) return [{ path, message: 'must be an object' }];
  const e = value;

  if (!isStr(e.id)) bad('id', 'must be a non-empty string');
  if (!isStr(e.matchId)) bad('matchId', 'must be a non-empty string');
  if (!isNum(e.t) || e.t < 0) bad('t', 'must be a finite number ≥ 0');
  if (!oneOf(['media', 'wall'] as const, e.clock)) bad('clock', "must be 'media' or 'wall'");
  if (!oneOf(EVENT_TYPES, e.type)) bad('type', `must be one of ${EVENT_TYPES.join(', ')}`);
  if (!(e.player === null || e.player === 'A' || e.player === 'B')) bad('player', "must be 'A', 'B' or null");
  if (e.zone !== undefined && !oneOf(ZONES, e.zone)) bad('zone', `must be one of ${ZONES.join(', ')}`);
  if (!isStr(e.engine)) bad('engine', 'must be a non-empty string');
  if (!Array.isArray(e.evidence) || e.evidence.length === 0 || !e.evidence.every((x) => oneOf(EVIDENCE, x))) {
    bad('evidence', `must be a non-empty array of ${EVIDENCE.join(', ')}`);
  }

  if (e.card !== undefined) {
    if (!isObj(e.card)) bad('card', 'must be an object');
    else {
      const c = e.card;
      if (!(c.cardId === null || isStr(c.cardId))) bad('card.cardId', 'must be a non-empty string or null');
      if (!isNum(c.confidence) || c.confidence < 0 || c.confidence > 1) bad('card.confidence', 'must be in [0, 1]');
      if (c.printingId !== undefined && !isStr(c.printingId)) bad('card.printingId', 'must be a non-empty string');
      if (c.alternatives !== undefined) {
        const ok =
          Array.isArray(c.alternatives) &&
          c.alternatives.every((a) => isObj(a) && isStr(a.cardId) && isNum(a.confidence) && a.confidence >= 0 && a.confidence <= 1);
        if (!ok) bad('card.alternatives', 'must be an array of { cardId, confidence }');
      }
    }
  }
  if (e.quad !== undefined) {
    const ok = Array.isArray(e.quad) && e.quad.length === 8 && e.quad.every((n) => isNum(n) && n >= 0 && n <= 1);
    if (!ok) bad('quad', 'must be 8 numbers in [0, 1]');
  }
  if (e.type === 'retracted' && !isStr(e.retracts)) bad('retracts', "is required for 'retracted' events");
  if (e.type === 'score_changed') {
    const s = e.score;
    if (!isObj(s) || !isNum(s.A) || !isNum(s.B)) bad('score', "is required for 'score_changed' events as { A, B }");
  }
  if (e.note !== undefined && typeof e.note !== 'string') bad('note', 'must be a string');
  return issues;
}

export function validateTimelineDocument(value: unknown): Issue[] {
  if (!isObj(value)) return [{ path: 'document', message: 'must be an object' }];
  const d = value;
  const issues: Issue[] = [];
  if (d.schema !== 'rifteye.timeline') issues.push({ path: 'schema', message: "must be 'rifteye.timeline'" });
  if (d.version !== SCHEMA_VERSION) issues.push({ path: 'version', message: `must be ${SCHEMA_VERSION}` });
  if (!isStr(d.createdAt)) issues.push({ path: 'createdAt', message: 'must be an ISO date string' });
  if (!isStr(d.updatedAt)) issues.push({ path: 'updatedAt', message: 'must be an ISO date string' });

  const m = d.match;
  if (!isObj(m)) issues.push({ path: 'match', message: 'must be an object' });
  else {
    if (!isStr(m.matchId)) issues.push({ path: 'match.matchId', message: 'must be a non-empty string' });
    const s = m.source;
    if (!isObj(s) || !oneOf(['file', 'youtube', 'twitch', 'other'] as const, s.kind) || !isStr(s.ref)) {
      issues.push({ path: 'match.source', message: "must be { kind: 'file' | 'youtube' | 'twitch' | 'other', ref }" });
    }
  }

  if (!Array.isArray(d.events)) issues.push({ path: 'events', message: 'must be an array' });
  else {
    const ids = new Set<string>();
    d.events.forEach((ev, i) => {
      issues.push(...validateTimelineEvent(ev, `events[${i}]`));
      if (isObj(ev) && isStr(ev.id)) {
        if (ids.has(ev.id)) issues.push({ path: `events[${i}].id`, message: 'duplicate id' });
        ids.add(ev.id);
      }
      if (isObj(ev) && isObj(m) && isStr(m.matchId) && ev.matchId !== m.matchId) {
        issues.push({ path: `events[${i}].matchId`, message: 'must equal match.matchId' });
      }
    });
  }
  return issues;
}

export function validateLayoutPreset(value: unknown): Issue[] {
  if (!isObj(value)) return [{ path: 'layout', message: 'must be an object' }];
  const l = value;
  const issues: Issue[] = [];
  if (l.schema !== 'rifteye.layout') issues.push({ path: 'schema', message: "must be 'rifteye.layout'" });
  if (l.version !== SCHEMA_VERSION) issues.push({ path: 'version', message: `must be ${SCHEMA_VERSION}` });
  if (!isStr(l.id)) issues.push({ path: 'id', message: 'must be a non-empty string' });
  if (!oneOf(['bottom', 'top', 'left', 'right'] as const, l.playerA)) {
    issues.push({ path: 'playerA', message: "must be 'bottom', 'top', 'left' or 'right'" });
  }
  if (!isObj(l.regions)) issues.push({ path: 'regions', message: 'must be an object' });
  else {
    for (const [name, rect] of Object.entries(l.regions)) {
      if (!oneOf(LAYOUT_REGIONS, name)) {
        issues.push({ path: `regions.${name}`, message: `unknown region (expected ${LAYOUT_REGIONS.join(', ')})` });
        continue;
      }
      const ok =
        Array.isArray(rect) &&
        rect.length === 4 &&
        rect.every((n) => isNum(n) && n >= 0 && n <= 1) &&
        (rect[0] as number) < (rect[2] as number) &&
        (rect[1] as number) < (rect[3] as number);
      if (!ok) issues.push({ path: `regions.${name}`, message: 'must be [x0, y0, x1, y1] in [0, 1] with x0 < x1 and y0 < y1' });
    }
    if (!('overhead' in l.regions)) issues.push({ path: 'regions.overhead', message: 'is required' });
  }
  return issues;
}

export function validateIndexManifest(value: unknown): Issue[] {
  if (!isObj(value)) return [{ path: 'manifest', message: 'must be an object' }];
  const m = value;
  const issues: Issue[] = [];
  if (m.schema !== 'rifteye.index') issues.push({ path: 'schema', message: "must be 'rifteye.index'" });
  if (m.version !== SCHEMA_VERSION) issues.push({ path: 'version', message: `must be ${SCHEMA_VERSION}` });
  for (const f of ['catalogVersion', 'model', 'modelSha256', 'createdAt'] as const) {
    if (!isStr(m[f])) issues.push({ path: f, message: 'must be a non-empty string' });
  }
  if (!isNum(m.dim) || !Number.isInteger(m.dim) || m.dim <= 0) issues.push({ path: 'dim', message: 'must be a positive integer' });
  if (m.dtype !== 'float16') issues.push({ path: 'dtype', message: "must be 'float16'" });
  if (!Array.isArray(m.rows) || m.rows.length === 0 || !m.rows.every(isStr)) {
    issues.push({ path: 'rows', message: 'must be a non-empty array of printing ids' });
  } else if (new Set(m.rows).size !== m.rows.length) {
    issues.push({ path: 'rows', message: 'must not contain duplicates' });
  }
  return issues;
}

// ---------------------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------------------

const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

/** A ULID: 48-bit millisecond timestamp + 80 random bits, Crockford base32, sortable. */
export function newId(now: number = Date.now()): string {
  let time = '';
  let t = Math.floor(now);
  for (let i = 0; i < 10; i++) {
    time = CROCKFORD[t % 32] + time;
    t = Math.floor(t / 32);
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  let rand = '';
  for (let i = 0; i < 16; i++) rand += CROCKFORD[(bytes[i] as number) % 32];
  return time + rand;
}
