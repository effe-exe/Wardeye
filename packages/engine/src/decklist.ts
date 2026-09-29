// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Decklists, ported from ml/rifteye_ml/decklist.py: a list a viewer pastes, in any of four export formats (deckbuilder JSON,
// text with collector codes, the tourney sheet, the deck code), read through the gallery's rows into cards and copies; and
// the printings a player of those cards can put on the table (`expand`). Ids and names the rows do not know are reported,
// never guessed. `listMask` is the guard that makes a wrong list harmless: a list counts only on the side whose legend,
// as read on the table, it names. A list only helps name the cards face up on the table: it is never shown, and never used
// to guess a hand, a face-down card or the rest of the list (D-005, D-026).

import { baseName, normalise, tokenRows } from './priors';
import type { CatalogRow } from './types';

export type Board = 'main' | 'side';
export const BOARDS: readonly Board[] = ['main', 'side'];

const text = (x: unknown): string => (typeof x === 'string' ? x : '');

/** Python's str.splitlines(): every line break it knows, and no empty last line for a trailing one. */
export function splitLines(s: string): string[] {
  const lines = s.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
  if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
  return s === '' ? [] : lines;
}

/** Gallery rows indexed for decklists: by printing, by card_id and by normalised name. A legend's catalogue name is its title
 * only (the champion is in its tags), while lists write 'Champion - Title' or 'Champion, Title', so each legend is also known
 * as '<tag> <title>'. */
export class Catalogue {
  readonly rows: readonly CatalogRow[];
  readonly pid = new Map<string, number>();
  readonly card = new Map<string, number[]>();
  readonly names = new Map<string, Set<string>>();
  readonly bases = new Map<string, Set<string>>();
  readonly kind = new Map<string, string>();
  /** Tokens, which no list names and every player may make. */
  readonly tokens: Uint8Array;

  constructor(rows: readonly CatalogRow[]) {
    this.rows = rows;
    const add = (index: Map<string, Set<string>>, key: string, cardId: string): void => {
      if (!key) return;
      let set = index.get(key);
      if (!set) index.set(key, (set = new Set()));
      set.add(cardId);
    };
    rows.forEach((r, i) => {
      if (!this.pid.has(r.printing_id)) this.pid.set(r.printing_id, i);
      const same = this.card.get(r.card_id);
      if (same) same.push(i);
      else this.card.set(r.card_id, [i]);
      if (!this.kind.has(r.card_id)) this.kind.set(r.card_id, text(r.type));
      const name = text(r.name);
      add(this.names, normalise(name), r.card_id);
      add(this.bases, normalise(baseName(name)), r.card_id);
      if (r.type === 'Legend' && Array.isArray(r.tags)) for (const tag of r.tags) add(this.names, normalise(`${String(tag)} ${name}`), r.card_id);
    });
    this.tokens = tokenRows(rows).mask;
  }

  /** The printing a listed id names: as written, or with its set code in capitals. */
  resolveId(pid: string): string | null {
    const p = (pid ?? '').trim();
    if (this.pid.has(p)) return p;
    const m = /^([A-Za-z]+)-([\s\S]+)$/.exec(p);
    const upper = m ? `${m[1]!.toUpperCase()}-${m[2]}` : null;
    return upper !== null && this.pid.has(upper) ? upper : null;
  }

  /** The card a written name means, and every card it could mean. A name that means more than one card is settled by the
   * section's card types (`types`), or left unmapped. */
  resolveName(name: string, types: readonly string[] = []): [string | null, string[]] {
    for (const [key, index] of [[normalise(name), this.names], [normalise(baseName(name)), this.bases]] as const) {
      let found = [...(index.get(key) ?? [])].sort();
      if (types.length && found.length > 1) {
        const typed = found.filter((c) => types.includes(this.kind.get(c) ?? ''));
        if (typed.length) found = typed;
      }
      if (found.length === 1) return [found[0]!, found];
      if (found.length) return [null, found];
    }
    return [null, []];
  }

  printingCard(pid: string): string {
    return this.rows[this.pid.get(pid)!]!.card_id;
  }
}

export interface Entry {
  card_id: string;
  count: number;
  /** 'main' (the legend, champion, main deck, battlefields and runes) or 'side'. */
  board: Board;
  /** The tourney sheet's section: legend, champion, maindeck, battlefields, runes, sideboard. */
  section: string;
  /** The printing the list names, when its format gives one. */
  listed: string;
  /** The card's type in the catalogue. */
  kind: string;
}

/** A list, read: its cards and copies, what it names that the rows do not know, and notes on what was settled. */
export class Deck {
  constructor(
    readonly fmt: string,
    readonly entries: Entry[] = [],
    readonly unmapped: string[] = [],
    readonly notes: string[] = [],
  ) {}

  /** 'board card_id' -> copies. */
  counts(): Map<string, number> {
    const out = new Map<string, number>();
    for (const e of this.entries) out.set(`${e.board} ${e.card_id}`, (out.get(`${e.board} ${e.card_id}`) ?? 0) + e.count);
    return out;
  }

  /** 'board printing_id' -> copies, for the formats that name printings. */
  listed(): Map<string, number> {
    const out = new Map<string, number>();
    for (const e of this.entries) if (e.listed) out.set(`${e.board} ${e.listed}`, (out.get(`${e.board} ${e.listed}`) ?? 0) + e.count);
    return out;
  }

  cardIds(boards: readonly Board[] = BOARDS): Set<string> {
    return new Set(this.entries.filter((e) => boards.includes(e.board)).map((e) => e.card_id));
  }

  legends(): string[] {
    return [...new Set(this.entries.filter((e) => e.kind === 'Legend').map((e) => e.card_id))].sort();
  }

  battlefields(): Set<string> {
    return new Set(this.entries.filter((e) => e.kind === 'Battlefield').map((e) => e.card_id));
  }
}

const entry = (cat: Catalogue, cardId: string, count: number, board: Board, section = '', listed = ''): Entry => ({
  card_id: cardId,
  count,
  board,
  section,
  listed,
  kind: cat.kind.get(cardId) ?? '',
});

/** Python's int() on a list's count, then at least one. */
function count(v: unknown): number {
  let n: number;
  if (typeof v === 'number' && Number.isFinite(v)) n = Math.trunc(v);
  else if (typeof v === 'string' && /^\s*[+-]?\d+\s*$/.test(v)) n = Number.parseInt(v, 10);
  else if (typeof v === 'boolean') n = v ? 1 : 0;
  else throw new Error(`a count of ${JSON.stringify(v)}`);
  if (n < 1) throw new Error(`a count of ${JSON.stringify(v)}`);
  return n;
}

/** Deckbuilder JSON: deck['Main Board'] (with the legend, battlefields and runes) and deck['Side Board'], entries {id, count}. */
export function parseJson(doc: unknown, cat: Catalogue): Deck {
  const top = typeof doc === 'object' && doc !== null && !Array.isArray(doc) ? (doc as Record<string, unknown>) : {};
  const deck = (('deck' in top ? top['deck'] : top) ?? {}) as Record<string, unknown>;
  const out = new Deck('json');
  for (const [key, board] of [['Main Board', 'main'], ['Side Board', 'side']] as const) {
    const items = deck[key];
    for (const item of Array.isArray(items) ? items : []) {
      const it = (item ?? {}) as Record<string, unknown>;
      const id = String(it['id'] ?? '');
      const pid = cat.resolveId(id);
      if (pid === null) {
        out.unmapped.push(id);
        continue;
      }
      out.entries.push(entry(cat, cat.printingCard(pid), count(it['count'] ?? 1), board, '', pid));
    }
  }
  return out;
}

const LINE_ID = /^\s*(\d+)\s*[xX]?\s+(.*?)\s*\(\s*([A-Za-z]{2,5}-[A-Za-z0-9]+\*?)\s*\)\s*$/;
const LINE = /^\s*(\d+)\s*[xX]?\s+(.+?)\s*$/;
const SIDE_HEADER = /^\s*side\s*board\s*:?\s*$/i;
const SECTIONS: Readonly<Record<string, string>> = {
  legend: 'legend', legends: 'legend', champion: 'champion', champions: 'champion', maindeck: 'maindeck', 'main deck': 'maindeck',
  battlefield: 'battlefields', battlefields: 'battlefields', rune: 'runes', runes: 'runes', sideboard: 'sideboard', 'side board': 'sideboard',
};
const SECTION_HEADER = new RegExp(`^\\s*(${Object.keys(SECTIONS).sort((a, b) => b.length - a.length).join('|')})\\s*:\\s*$`, 'i');
export const SECTION_TYPES: Readonly<Record<string, readonly string[]>> = {
  legend: ['Legend'], champion: ['Unit'], battlefields: ['Battlefield'], runes: ['Rune'], maindeck: ['Unit', 'Spell', 'Gear'],
};

const couldBe = (found: readonly string[]): string => (found.length ? ` (could be ${found.join(', ')})` : '');

/** 'N Name (SET-NNN)' lines, then a 'Side Board:' header. The id decides; a name that disagrees is noted. A line without an
 * id is mapped by its name. */
export function parseText(s: string, cat: Catalogue): Deck {
  const out = new Deck('text');
  let board: Board = 'main';
  for (const line of splitLines(s)) {
    if (!line.trim()) continue;
    if (SIDE_HEADER.test(line)) {
      board = 'side';
      continue;
    }
    const m = LINE_ID.exec(line);
    if (m) {
      const pid = cat.resolveId(m[3]!);
      if (pid !== null) {
        const card = cat.printingCard(pid);
        const [named] = cat.resolveName(m[2]!);
        if (named !== null && named !== card) out.notes.push(`${m[3]} is written as '${m[2]}', which names ${named}; the id decides`);
        out.entries.push(entry(cat, card, count(m[1]), board, '', pid));
        continue;
      }
      const [named] = cat.resolveName(m[2]!);
      if (named === null) {
        out.unmapped.push(line.trim());
        continue;
      }
      out.notes.push(`${m[3]} is not in the catalogue; mapped by its name to ${named}`);
      out.entries.push(entry(cat, named, count(m[1]), board));
      continue;
    }
    const l = LINE.exec(line);
    const [named, found] = l ? cat.resolveName(l[2]!) : [null, [] as string[]];
    if (named === null) {
      out.unmapped.push(line.trim() + couldBe(found));
      continue;
    }
    out.entries.push(entry(cat, named, count(l![1]), board));
  }
  return out;
}

/** The tourney sheet: sections 'Legend:', 'Champion:', 'MainDeck:', 'Battlefields:', 'Runes:', 'Sideboard:', lines 'N Name'
 * without ids. Names go through the catalogue; a name that could mean two cards is settled by its section's card types. Every
 * entry keeps its section; all but the sideboard are the main board. */
export function parseTourney(s: string, cat: Catalogue): Deck {
  const out = new Deck('tourney');
  let section = '';
  for (const line of splitLines(s)) {
    if (!line.trim()) continue;
    const h = SECTION_HEADER.exec(line);
    if (h) {
      section = SECTIONS[h[1]!.toLowerCase()]!;
      continue;
    }
    const m = LINE.exec(line);
    if (!m) {
      out.unmapped.push(line.trim());
      continue;
    }
    const [named, found] = cat.resolveName(m[2]!, SECTION_TYPES[section] ?? []);
    if (named === null) {
      out.unmapped.push(line.trim() + couldBe(found));
      continue;
    }
    out.entries.push(entry(cat, named, count(m[1]), section === 'sideboard' ? 'side' : 'main', section));
  }
  if (!section) out.notes.push('no section headers: every line was read as the main board');
  return out;
}

export function parseCode(code: string, cat: Catalogue): Deck {
  const [main, side] = decodeCode(code);
  const out = new Deck('code');
  for (const [board, cards] of [['main', main], ['side', side]] as const) {
    for (const [pid, n] of cards) {
      const got = cat.resolveId(pid);
      if (got === null) {
        out.unmapped.push(pid);
        continue;
      }
      out.entries.push(entry(cat, cat.printingCard(got), n, board, '', got));
    }
  }
  return out;
}

export type Format = 'json' | 'code' | 'tourney' | 'text';

export function detectFormat(s: string): Format {
  const t = s.trim();
  if (t.startsWith('{')) return 'json';
  if (/^[A-Za-z2-7]{8,}$/.test(t)) return 'code';
  if (splitLines(t).some((line) => SECTION_HEADER.test(line) && !SIDE_HEADER.test(line) && !line.toLowerCase().includes('side'))) return 'tourney';
  return 'text';
}

/** A decklist in any of the four formats (detected unless `fmt` names it). Throws for a code or JSON it cannot read. */
export function parse(s: string, cat: Catalogue, fmt: Format = detectFormat(s)): Deck {
  if (fmt === 'json') return parseJson(JSON.parse(s) as unknown, cat);
  if (fmt === 'code') return parseCode(s, cat);
  if (fmt === 'tourney') return parseTourney(s, cat);
  return parseText(s, cat);
}

// --- the deck code ----------------------------------------------------------------------------------------------------

export const CODE_HEADER = 0x13; // format 1, version 3
export const CODE_SETS: Readonly<Record<number, string>> = { 0: 'OGN', 1: 'OGS', 3: 'SFD', 4: 'UNL', 5: 'VEN' }; // 2 is unknown
const MAIN_COUNTS = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
const SIDE_COUNTS = [3, 2, 1];
const B32 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';

/** A deck code this reader does not understand. It refuses rather than guesses. */
export class DeckCodeError extends Error {
  override name = 'DeckCodeError';
}

/** RFC 4648 base32 without padding, as Python's b32decode reads it once padded: a length that no bytes give is refused. */
function base32(s: string): Uint8Array {
  if ([1, 3, 6].includes(s.length % 8)) throw new DeckCodeError('not valid base32 (incorrect padding)');
  const out: number[] = [];
  let bits = 0;
  let value = 0;
  for (const ch of s) {
    value = (value << 5) | B32.indexOf(ch);
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      out.push((value >> bits) & 0xff);
    }
  }
  return Uint8Array.from(out);
}

function varint(data: Uint8Array, pos: number): [number, number] {
  let value = 0;
  let scale = 1;
  for (;;) {
    if (pos >= data.length) throw new DeckCodeError('the code ends inside a number');
    const byte = data[pos++]!;
    value += (byte & 0x7f) * scale;
    scale *= 128;
    if (!(byte & 0x80)) return [value, pos];
  }
}

/** A deck code -> [main board, side board], printing id -> copies. Base32 (no padding); byte 0 is 0x13 (format 1, version 3);
 * then LEB128 numbers. For each count from 12 down to 1 (main board), then 3 down to 1 (side board): the number of groups with
 * that count, and per group its size n, set index, variant, then n card numbers. One trailing 0 is accepted. Card id:
 * '{SET}-{number:03d}'. An unknown set index (2 among them) or a non-zero variant is refused. */
export function decodeCode(code: string): [Map<string, number>, Map<string, number>] {
  const s = (code ?? '').replace(/\s+/g, '').toUpperCase();
  if (!s || [...s].some((ch) => !B32.includes(ch))) throw new DeckCodeError('not a deck code: base32 letters A-Z and 2-7 only');
  const data = base32(s);
  if (!data.length || data[0] !== CODE_HEADER) throw new DeckCodeError(`header byte ${data.length ? data[0] : 'none'}: only format 1 version 3 (0x13) is known`);
  let pos = 1;
  const boards: Map<string, number>[] = [];
  for (const counts of [MAIN_COUNTS, SIDE_COUNTS]) {
    const cards = new Map<string, number>();
    for (const n of counts) {
      let groups: number;
      [groups, pos] = varint(data, pos);
      for (let g = 0; g < groups; g++) {
        let size: number;
        let set: number;
        let variant: number;
        [size, pos] = varint(data, pos);
        [set, pos] = varint(data, pos);
        [variant, pos] = varint(data, pos);
        const code = CODE_SETS[set];
        if (code === undefined) {
          const known = Object.entries(CODE_SETS).map(([k, v]) => `${k} ${v}`).join(', ');
          throw new DeckCodeError(`set index ${set} is not known (known: ${known}); refusing to guess`);
        }
        if (variant !== 0) throw new DeckCodeError(`variant ${variant} in ${code}: only 0 is known; refusing to guess`);
        for (let k = 0; k < size; k++) {
          let number: number;
          [number, pos] = varint(data, pos);
          const pid = `${code}-${String(number).padStart(3, '0')}`;
          if (cards.has(pid)) throw new DeckCodeError(`${pid} is listed twice in one board`);
          cards.set(pid, n);
        }
      }
    }
    boards.push(cards);
  }
  if (pos < data.length) {
    const [tail, end] = varint(data, pos);
    if (tail !== 0 || end !== data.length) throw new DeckCodeError(`${data.length - end + 1} unexpected bytes after the side board`);
  }
  return [boards[0]!, boards[1]!];
}

function putVarint(out: number[], value: number): void {
  if (value < 0) throw new DeckCodeError(`cannot write ${value}`);
  let v = value;
  for (;;) {
    const byte = v % 128;
    v = Math.floor(v / 128);
    out.push(byte | (v ? 0x80 : 0));
    if (!v) return;
  }
}

/** The deck code of a main and a side board (printing id -> copies), groups in set order and numbers ascending, as the
 * exports write them. Only 'SET-NNN' printings of known sets fit in a code. */
export function encodeCode(main: ReadonlyMap<string, number>, side: ReadonlyMap<string, number> = new Map()): string {
  const index = new Map(Object.entries(CODE_SETS).map(([k, v]) => [v, Number(k)]));
  const out: number[] = [CODE_HEADER];
  for (const [cards, counts] of [[main, MAIN_COUNTS], [side, SIDE_COUNTS]] as const) {
    const groups = new Map<number, Map<number, number[]>>();
    for (const [pid, n] of cards) {
      const m = /^([A-Z]+)-(\d+)$/.exec(pid);
      if (!m || !index.has(m[1]!)) throw new DeckCodeError(`${pid} has no place in a deck code (an unknown set, or a printing suffix)`);
      if (!counts.includes(n)) throw new DeckCodeError(`${pid}: ${n} copies; a board holds 1 to ${Math.max(...counts)}`);
      let bySet = groups.get(n);
      if (!bySet) groups.set(n, (bySet = new Map()));
      const set = index.get(m[1]!)!;
      bySet.set(set, [...(bySet.get(set) ?? []), Number(m[2])]);
    }
    for (const n of counts) {
      const sets = groups.get(n) ?? new Map<number, number[]>();
      putVarint(out, sets.size);
      for (const s of [...sets.keys()].sort((a, b) => a - b)) {
        const numbers = [...sets.get(s)!].sort((a, b) => a - b);
        putVarint(out, numbers.length);
        putVarint(out, s);
        putVarint(out, 0);
        for (const x of numbers) putVarint(out, x);
      }
    }
  }
  putVarint(out, 0);
  let bits = 0;
  let value = 0;
  let s = '';
  for (const byte of out) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      s += B32[(value >> bits) & 31];
    }
    value &= (1 << bits) - 1;
  }
  if (bits > 0) s += B32[(value << (5 - bits)) & 31];
  return s;
}

// --- the prior --------------------------------------------------------------------------------------------------------

export interface Expansion {
  /** One flag per row: the printings a player of these cards can put on the table. */
  mask: Uint8Array;
  /** Card ids the rows do not hold. */
  unknown: string[];
  /** Printings of the same name filed under another card_id, taken in too. */
  sameName: { listed: string; card_id: string; printings: string[] }[];
}

/** Every printing a player of these cards can put on the table: all printings of each card_id (any variant, set or language
 * the gallery holds), any printing of the same normalised name filed under another card_id, and, with `tokens`, every token. */
export function expand(cat: Catalogue, cardIds: Iterable<string>, tokens = true): Expansion {
  const ids = [...new Set(cardIds)].sort();
  const want = new Set(ids);
  const mask = new Uint8Array(cat.rows.length);
  const unknown: string[] = [];
  const sameName: Expansion['sameName'] = [];
  for (const cid of ids) {
    const rows = cat.card.get(cid);
    if (!rows) {
      unknown.push(cid);
      continue;
    }
    for (const i of rows) mask[i] = 1;
    const keys = new Set(rows.map((i) => normalise(baseName(text(cat.rows[i]!.name)))));
    keys.delete('');
    for (const key of keys) {
      for (const other of [...(cat.bases.get(key) ?? [])].filter((c) => c !== cid).sort()) {
        if (want.has(other)) continue;
        for (const i of cat.card.get(other)!) mask[i] = 1;
        sameName.push({ listed: cid, card_id: other, printings: cat.card.get(other)!.map((i) => cat.rows[i]!.printing_id) });
      }
    }
  }
  if (tokens) for (let i = 0; i < mask.length; i++) if (cat.tokens[i]) mask[i] = 1;
  return { mask, unknown, sameName };
}

/** The printings a side may be read as when a list names its legend: every card of the lists that name it (both boards: a
 * sideboard card can come in between games), every battlefield of all the lists (both players' battlefields lie on the
 * midline, so one list's turns up on the other half), and the tokens. null when no list names the legend: the legend rule
 * holds there, and a list that is not the players' costs nothing. */
export function listMask(cat: Catalogue, decks: readonly Deck[], legend: string): Uint8Array | null {
  const mine = decks.filter((d) => d.legends().includes(legend));
  if (!mine.length) return null;
  const cards = new Set<string>();
  for (const d of mine) for (const c of d.cardIds()) cards.add(c);
  for (const d of decks) for (const c of d.battlefields()) cards.add(c);
  return expand(cat, cards).mask;
}
