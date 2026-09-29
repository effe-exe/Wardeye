// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Priors on card identification, ported from ml/rifteye_ml/priors.py: which gallery rows may compete for a crop
// (D-026). The recogniser applies the legend rule with them: once a side's legend is pinned, a crop on that side
// competes only with the printings whose domains, less Colorless, lie within the legend's, runes included, and with
// every battlefield and token. A row with no domains at all fits any legend, so a catalogue without them keeps the
// whole gallery. Both ports flag the same rows (test/priors.test.ts, and the recogniser's replay against Python).
//
// A prior only helps name cards face up on the table. It is never shown, and never used to guess or reveal a hand, a
// face-down card or the rest of a list (D-005).

import type { CatalogRow } from './types';

/** Kept by the legend prior whatever their domains (runes only while they are not held to the legend). */
export const KEEP_TYPES: readonly string[] = ['Rune', 'Battlefield'];
export const COLORLESS = 'Colorless';

// --- names, as Python's str methods and re module treat them ------------------------------------------------------

/** Python's str.isspace (and re's \s): JavaScript's \s also takes U+FEFF, and leaves out U+001C to U+001F and U+0085. */
const PY_SPACE = '[\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const TRAILING_PARENTHETICAL = new RegExp(`${PY_SPACE}*\\([^()]*\\)${PY_SPACE}*$`);
const EDGE_SPACE = new RegExp(`^${PY_SPACE}+|${PY_SPACE}+$`, 'g');
/** Python's [\W_]+: runs of what str.isalnum rejects (letters are L*, numbers N*: the same code points). */
const NOT_ALNUM = /[^\p{L}\p{N}]+/gu;

/** unicodedata.combining(ch) != 0 for a character NFKD leaves as it is: canonical ordering moves a mark of class c
 * before U+0334 (class 1) when c > 1, and after U+0301 (class 230) when 0 < c < 230. Nothing below U+0300 has one. */
function combining(ch: string): boolean {
  if (ch.codePointAt(0)! < 0x300) return false;
  if ((ch + '̴').normalize('NFD').startsWith('̴')) return true;
  return ('́' + ch).normalize('NFD').startsWith(ch);
}

/** The Old Cyrillic letter variants U+1C80 to U+1C88, folded to the letters they are variants of. */
const CYRILLIC_VARIANTS = ['в', 'д', 'о', 'с', 'т', 'т', 'ъ', 'ѣ', 'ꙋ'];

/** str.casefold for one character NFKD leaves as it is, with no combining class: its lowercase, except where full
 * case folding differs (ß and ẞ, final sigma, the Cyrillic variants, and Cherokee, which folds to its capitals). */
function casefold(ch: string): string {
  const c = ch.codePointAt(0)!;
  if (c === 0xdf || c === 0x1e9e) return 'ss';
  if (c === 0x3c2) return 'σ';
  if (c >= 0x1c80 && c <= 0x1c88) return CYRILLIC_VARIANTS[c - 0x1c80]!;
  if (c >= 0x13a0 && c <= 0x13f5) return ch;
  if (c >= 0x13f8 && c <= 0x13fd) return String.fromCodePoint(c - 8);
  if (c >= 0xab70 && c <= 0xabbf) return String.fromCodePoint(c - 0xab70 + 0x13a0);
  return ch.toLowerCase(); // one character alone: no final sigma
}

/** A name for matching: accents folded, case and apostrophes dropped, other punctuation a space. 'Ornn - Blacksmith'
 * and 'Ornn, Blacksmith' meet at 'ornn blacksmith'; letters of other scripts stay. */
export function normalise(name: string | null | undefined): string {
  let s = '';
  for (const ch of (name ?? '').normalize('NFKD')) if (!combining(ch)) s += casefold(ch);
  return s.replace(/['’`]/g, '').replace(NOT_ALNUM, ' ').trim(); // only spaces are left at the edges
}

/** The name without a trailing parenthetical: 'Recruit (ZN)' -> 'Recruit'. */
export function baseName(name: string | null | undefined): string {
  return (name ?? '').replace(TRAILING_PARENTHETICAL, '').replace(EDGE_SPACE, '');
}

const text = (x: unknown): string => (typeof x === 'string' ? x : '');

/** A printing's domains, less Colorless: empty for a colourless row or one with no domains, which fits any legend. */
export function domains(row: CatalogRow): Set<string> {
  const d = new Set(Array.isArray(row.domains) ? (row.domains as string[]) : []);
  d.delete(COLORLESS);
  return d;
}

// --- the priors ---------------------------------------------------------------------------------------------------

/** Tokens: printings marked token, every printing of a token's card, and printings whose name without its
 * parenthetical is a token's (Origins printed some tokens as numbered cards, such as 'Recruit (ZN)'). One flag per
 * row, and the rows marked token. */
export function tokenRows(rows: readonly CatalogRow[]): { mask: Uint8Array; marked: number[] } {
  const marked: number[] = [];
  rows.forEach((r, i) => {
    if (r.variant === 'token' || text(r.type).toLowerCase() === 'token') marked.push(i);
  });
  const cards = new Set(marked.map((i) => rows[i]!.card_id));
  const names = new Set(marked.map((i) => normalise(baseName(text(rows[i]!.name)))));
  names.delete('');
  const mask = new Uint8Array(rows.length);
  for (const i of marked) mask[i] = 1;
  rows.forEach((r, i) => {
    if (cards.has(r.card_id) || names.has(normalise(baseName(text(r.name))))) mask[i] = 1;
  });
  return { mask, marked };
}

export interface LegendMaskOptions {
  /** tokenRows(rows).mask, when the caller has it. */
  tokens?: Uint8Array;
  /** Hold the runes to the legends' domains too, as a rune deck follows its legend (default); false keeps every rune. */
  runes?: boolean;
}

const subset = (a: ReadonlySet<string>, b: ReadonlySet<string>): boolean => {
  for (const x of a) if (!b.has(x)) return false;
  return true;
};

/** The rows a crop may be read as, given the legends (card ids) of its side: printings whose domains fit one legend's
 * (the domains of the legend card's first row), plus every battlefield and token whatever their domains; and every
 * rune too unless `runes` holds them to the legends' domains. One flag per row, and each legend's domains. Throws
 * for a legend the rows do not hold. */
export function legendMask(rows: readonly CatalogRow[], legends: Iterable<string>, opts: LegendMaskOptions = {}): { mask: Uint8Array; fits: Set<string>[] } {
  const first = new Map<string, number>();
  rows.forEach((r, i) => {
    if (!first.has(r.card_id)) first.set(r.card_id, i);
  });
  const fits: Set<string>[] = [];
  for (const lg of legends) {
    const i = first.get(lg);
    if (i === undefined) throw new Error(`legend ${lg} is not in the catalogue`);
    fits.push(domains(rows[i]!));
  }
  const tokens = opts.tokens ?? tokenRows(rows).mask;
  const runes = opts.runes ?? true;
  const keep = KEEP_TYPES.filter((t) => !(runes && t === 'Rune'));
  const mask = new Uint8Array(rows.length);
  rows.forEach((r, i) => {
    const doms = domains(r);
    mask[i] = keep.includes(text(r.type)) || tokens[i] === 1 || fits.some((f) => subset(doms, f)) ? 1 : 0;
  });
  return { mask, fits };
}
