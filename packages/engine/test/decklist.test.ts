// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// ml/tests/test_decklist.py's reading of lists and codes, and its expansion, on the same synthetic catalogue: the port reads
// every list as Python does.

import { describe, expect, it } from 'vitest';
import { Catalogue, DeckCodeError, decodeCode, detectFormat, encodeCode, expand, listMask, parse, splitLines } from '../src/decklist';
import { ROWS } from './priors-rows';

const cat = new Catalogue(ROWS);
const map = (o: Record<string, number>): Map<string, number> => new Map(Object.entries(o));

const MAIN = { 'SFD-901': 1, 'SFD-902': 3, 'OGN-904': 2, 'OGN-905': 3, 'OGN-907': 1, 'OGN-918': 7, 'OGN-919': 5 };
const SIDE = { 'OGN-904': 1, 'OGN-914': 2 };
const JSON_LIST = JSON.stringify({
  metadata: { name: 'Test list' },
  deck: { 'Main Board': Object.entries(MAIN).map(([id, count]) => ({ id, count })), 'Side Board': Object.entries(SIDE).map(([id, count]) => ({ id, count })) },
});
const TEXT_LIST = `1 Fakesmith - Gleaming Anvil (SFD-901)
3 Fakesmith - Hammerer (SFD-902)
2 Pocket Gadget (OGN-904)
3 Quick-Trick (OGN-905)
1 Quiet Glade (OGN-907)
7 Hush Rune (OGN-918)
5 Muse Rune (OGN-919)

Side Board:
1 Pocket Gadget (OGN-904)
2 Blaze Fist (OGN-914)`;
const TOURNEY_LIST = `Legend:
1 Fakesmith, Gleaming Anvil

Champion:
3 FAKESMITH, HAMMERER

MainDeck:
2 pocket gadget
3 Quick Trick

Battlefields:
1 Quiet Glade

Runes:
7 Hush Rune
5 Muse Rune

Sideboard:
1 Pocket Gadget
2 Blaze Fist`;

describe('decklists in four formats', () => {
  it('give the same cards and copies', () => {
    const decks = [parse(JSON_LIST, cat), parse(TEXT_LIST, cat), parse(TOURNEY_LIST, cat), parse(encodeCode(map(MAIN), map(SIDE)), cat)];
    expect(decks.map((d) => d.fmt)).toEqual(['json', 'text', 'tourney', 'code']);
    for (const d of decks) {
      expect(d.unmapped, d.fmt).toEqual([]);
      expect(d.counts(), d.fmt).toEqual(decks[0]!.counts());
      expect(d.legends()).toEqual(['gleaming-anvil']);
    }
    expect(decks[0]!.counts().get('main fakesmith-hammerer')).toBe(3);
    expect([decks[0]!.counts().get('side pocket-gadget'), decks[0]!.counts().get('main pocket-gadget')]).toEqual([1, 2]);
    // the formats that name printings name the same ones; the tourney sheet names none
    expect(decks[1]!.listed()).toEqual(decks[0]!.listed());
    expect(decks[3]!.listed()).toEqual(decks[0]!.listed());
    expect(decks[2]!.listed().size).toBe(0);
    expect(decks[0]!.battlefields()).toEqual(new Set(['quiet-glade']));
  });

  it("keep the tourney sheet's sections, and report the names they cannot map", () => {
    const d = parse(`${TOURNEY_LIST}\n2 No Such Card`, cat);
    const sections = new Map(d.entries.filter((e) => e.board === 'main').map((e) => [e.card_id, e.section]));
    expect([sections.get('gleaming-anvil'), sections.get('fakesmith-hammerer'), sections.get('quiet-glade'), sections.get('hush-rune')]).toEqual(['legend', 'champion', 'battlefields', 'runes']);
    expect(new Set(d.entries.filter((e) => e.board === 'side').map((e) => e.card_id))).toEqual(new Set(['pocket-gadget', 'blaze-fist']));
    expect(d.unmapped).toEqual(['2 No Such Card']);
  });

  it('settle a name two cards share by the section, and leave it unmapped without one', () => {
    const d = parse("Battlefields:\n1 Zedka\n\nMainDeck:\n2 Zed'ka", cat);
    expect(Object.fromEntries(d.entries.map((e) => [e.card_id, e.section]))).toEqual({ zedka: 'battlefields', 'zed-ka': 'maindeck' });
    const loose = parse('1 zedka', cat); // no section, no id: it could be either card, so it is not guessed
    expect(loose.entries).toEqual([]);
    expect(loose.unmapped[0]).toContain('zed-ka');
    expect(loose.unmapped[0]).toContain('zedka');
  });

  it("let a text list's ids decide, and report ids they cannot map", () => {
    const d = parse('2 Pocket Gadget (OGN-904)\n1 Blaze Fist (OGN-905)\n1 Missing Card (OGN-999)', cat);
    expect(d.entries.map((e) => e.card_id)).toEqual(['pocket-gadget', 'quick-trick']);
    expect(d.notes.some((n) => n.includes('OGN-905') && n.includes('blaze-fist'))).toBe(true);
    expect(d.unmapped).toEqual(['1 Missing Card (OGN-999)']);
    const j = parse(JSON.stringify({ deck: { 'Main Board': [{ id: 'OGN-999', count: 1 }, { id: 'ogn-904', count: 2 }] } }), cat);
    expect(j.unmapped).toEqual(['OGN-999']);
    expect(j.entries.map((e) => e.listed)).toEqual(['OGN-904']);
  });

  it('are told apart by their look', () => {
    expect(detectFormat(JSON_LIST)).toBe('json');
    expect(detectFormat(TEXT_LIST)).toBe('text');
    expect(detectFormat(TOURNEY_LIST)).toBe('tourney');
    expect(detectFormat(encodeCode(map(MAIN), map(SIDE)))).toBe('code');
  });

  it('split lines as Python does', () => {
    expect(splitLines('a\r\nb\rc\nd\n')).toEqual(['a', 'b', 'c', 'd']);
    expect(splitLines('')).toEqual([]);
    expect(splitLines('a\u2028b')).toEqual(['a', 'b']);
  });
});

describe('the deck code', () => {
  const BIG_MAIN = { 'OGN-012': 12, 'OGN-311': 9, 'SFD-005': 9, 'OGS-001': 4, 'UNL-200': 3, 'VEN-150': 3, 'VEN-151': 3, 'SFD-129': 2, 'OGN-900': 1, 'OGS-002': 1, 'UNL-201': 1, 'VEN-152': 1 };
  const BIG_SIDE = { 'OGN-900': 3, 'SFD-130': 2, 'UNL-202': 1, 'VEN-999': 1 };
  const sorted = (m: Map<string, number>) => Object.fromEntries([...m].sort(([a], [b]) => (a < b ? -1 : 1)));

  it('round-trips', () => {
    for (const [main, side] of [[MAIN, SIDE], [BIG_MAIN, BIG_SIDE], [{ 'VEN-001': 1 }, {}]] as const) {
      const code = encodeCode(map(main), map(side));
      expect(code).toMatch(/^[A-Z2-7]+$/);
      const [m, s] = decodeCode(code);
      expect([sorted(m), sorted(s)]).toEqual([sorted(map(main)), sorted(map(side))]);
      expect(encodeCode(m, s)).toBe(code);
      expect(sorted(decodeCode(code.toLowerCase())[0])).toEqual(sorted(map(main)));
    }
  });

  /** A code from raw numbers after the header byte, for codes the encoder would never write. */
  const raw = (numbers: number[], header = 0x13, trailer = [0]): string => {
    const bytes = [header];
    for (const x of [...numbers, ...trailer]) {
      let v = x;
      do {
        const b = v % 128;
        v = Math.floor(v / 128);
        bytes.push(b | (v ? 0x80 : 0));
      } while (v);
    }
    const B = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
    let bits = 0;
    let value = 0;
    let out = '';
    for (const byte of bytes) {
      value = (value << 8) | byte;
      bits += 8;
      while (bits >= 5) out += B[(value >> (bits -= 5)) & 31];
      value &= (1 << bits) - 1;
    }
    return bits ? out + B[(value << (5 - bits)) & 31] : out;
  };
  const oneCard = (set: number, variant = 0): number[] => [...Array(11).fill(0), 1, 1, set, variant, 7, 0, 0, 0];

  it('refuses what it does not know', () => {
    expect(sorted(decodeCode(raw(oneCard(4)))[0])).toEqual({ 'UNL-007': 1 });
    expect(sorted(decodeCode(raw(oneCard(4), 0x13, []))[0])).toEqual({ 'UNL-007': 1 }); // the trailing 0 is optional
    expect(() => decodeCode(raw(oneCard(2)))).toThrow(/set index 2/);
    expect(() => decodeCode(raw(oneCard(9)))).toThrow(/set index 9/);
    expect(() => decodeCode(raw(oneCard(0, 1)))).toThrow(/variant 1/);
    expect(() => decodeCode(raw(oneCard(0), 0x12))).toThrow(/header/);
    expect(() => decodeCode(raw(oneCard(0), 0x13, [0, 5]))).toThrow(/unexpected/);
    expect(() => decodeCode(raw([0, 0, 0, 0, 0, 0x80], 0x13, []))).toThrow(/inside a number/);
    expect(() => decodeCode('not a code!')).toThrow(/base32/);
    expect(() => decodeCode('not a code!')).toThrow(DeckCodeError);
    for (const bad of [{ 'OGN-901a': 1 }, { 'XYZ-001': 1 }, { 'OGN-001': 13 }]) expect(() => encodeCode(map(bad))).toThrow(DeckCodeError);
    expect(() => encodeCode(map({ 'OGN-001': 1 }), map({ 'OGN-002': 4 }))).toThrow(DeckCodeError);
  });
});

describe('a list as a prior', () => {
  const pids = (mask: Uint8Array): Set<string> => new Set(ROWS.filter((_, i) => mask[i]).map((r) => `${r.printing_id} ${String(r.language)}`));

  it('takes a card to every printing, every language, the same name under another card, and the tokens', () => {
    const p = expand(cat, ['fakesmith-hammerer']);
    const got = pids(p.mask);
    for (const pid of ['SFD-902', 'SFD-902a', 'SFD-952', 'SFD-952*', 'VEN-902']) expect(got.has(`${pid} en`), pid).toBe(true);
    expect(got.has('SFD-902 zh-Hans')).toBe(true);
    expect(got.has('SFD-P01 en')).toBe(true);
    expect(p.sameName).toEqual([{ listed: 'fakesmith-hammerer', card_id: 'fakesmith-hammerer-promo', printings: ['SFD-P01'] }]);
    for (const token of ['UNL-T91', 'OGN-910', 'VEN-T92', 'OGN-911']) expect(got.has(`${token} en`), token).toBe(true);
    expect(got.has('OGN-904 en')).toBe(false);
    const bare = expand(cat, ['fakesmith-hammerer', 'no-such-card'], false);
    expect(bare.unknown).toEqual(['no-such-card']);
    expect(pids(bare.mask).has('UNL-T91 en')).toBe(false);
  });

  it("counts only on the side whose legend it names, with every list's battlefields", () => {
    const mine = parse(TEXT_LIST, cat);
    const other = parse('1 Fakezap, Thunder Crown (VEN-912)\n1 Zedka (OGN-922)', cat);
    expect(listMask(cat, [mine, other], 'silent-loom')).toBeNull(); // no list names it: the legend rule holds
    const m = pids(listMask(cat, [mine, other], 'gleaming-anvil')!);
    expect(m.has('SFD-902 en') && m.has('OGN-914 en')).toBe(true); // its cards, the side board too
    expect(m.has('OGN-922 en')).toBe(true); // the other list's battlefield
    expect(m.has('VEN-912 en')).toBe(false); // but not its cards
    expect(m.has('UNL-T91 en')).toBe(true); // and the tokens
  });
});
