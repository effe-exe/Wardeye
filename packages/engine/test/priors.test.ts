// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The priors against ml/rifteye_ml/priors.py: ml/tests/test_decklist.py's legend tests on its synthetic rows, and
// ml/tests/test_priors.py's expected values (the same rows, the same flags, the same names). No private data.

import { describe, expect, it } from 'vitest';
import { baseName, domains, legendMask, normalise, tokenRows } from '../src/priors';
import type { CatalogRow } from '../src/types';
import { ROWS } from './priors-rows';

/** The (printing, language) pairs a mask keeps, as test_decklist's _pids gives them. */
function kept(rows: readonly CatalogRow[], mask: Uint8Array): Set<string> {
  const out = new Set<string>();
  mask.forEach((m, i) => {
    if (m) out.add(`${rows[i]!.printing_id}/${String(rows[i]!.language)}`);
  });
  return out;
}

const pids = (mask: Uint8Array): Set<string> => new Set([...kept(ROWS, mask)].map((k) => k.split('/')[0]!));
const leftOut = (mask: Uint8Array): string[] => [...new Set(ROWS.filter((_, i) => !mask[i]).map((r) => r.printing_id))].sort();
const hasAll = (set: Set<string>, xs: string[]): void => {
  for (const x of xs) expect(set.has(x), x).toBe(true);
};
const hasNone = (set: Set<string>, xs: string[]): void => {
  for (const x of xs) expect(set.has(x), x).toBe(false);
};

describe('tokens (test_decklist: test_expand_takes_a_card_to_every_printing_and_tokens)', () => {
  it('are the marked printings, every printing of their cards, and printings named like them', () => {
    const { mask, marked } = tokenRows(ROWS);
    expect(marked).toEqual([24, 26]);
    expect(ROWS.filter((_, i) => mask[i]).map((r) => r.printing_id)).toEqual(['UNL-T91', 'OGN-910', 'VEN-T92', 'OGN-911']);
    const extra = ROWS.filter((_, i) => mask[i] && !marked.includes(i)).map((r) => r.printing_id).sort();
    expect(extra).toEqual(['OGN-910', 'OGN-911']); // cat.token_extra
  });
});

describe('the legend prior (test_decklist)', () => {
  it('keeps fitting cards, runes, battlefields and tokens', () => {
    const { mask } = legendMask(ROWS, ['gleaming-anvil', 'thunder-crown'], { runes: false });
    const k = pids(mask);
    hasAll(k, ['SFD-902', 'OGN-904', 'VEN-906', 'OGN-917', 'OGN-913', 'SFD-951']); // fits one legend, or colourless
    hasAll(k, ['OGN-918', 'OGN-920', 'OGN-907', 'UNL-909', 'UNL-T91', 'OGN-911']); // every rune, battlefield, token
    hasNone(k, ['OGN-914', 'OGN-915', 'OGN-916']); // off-domain, and Calm+Order fits neither legend alone
    const runes = ROWS.map((r, i) => [r.type, mask[i]] as const).filter(([t]) => t === 'Rune');
    expect([runes.filter(([, m]) => m).length, runes.length]).toEqual([4, 4]); // kept_by_type Rune
    expect(() => legendMask(ROWS, ['no-such-legend'])).toThrow(/no-such-legend/);
  });

  it('can hold runes to the legends', () => {
    // a rune deck follows its legend: with runes held, a Fury rune fits neither Calm+Mind nor Order+Chaos
    const both = pids(legendMask(ROWS, ['gleaming-anvil', 'thunder-crown'], { runes: true }).mask);
    hasAll(both, ['OGN-918', 'OGN-918a', 'OGN-919']);
    hasNone(both, ['OGN-920']);
    hasAll(both, ['OGN-907', 'UNL-909', 'UNL-T91']); // battlefields and tokens are still kept
    const own = pids(legendMask(ROWS, ['thunder-crown'], { runes: true }).mask);
    hasNone(own, ['OGN-918', 'OGN-919', 'OGN-920']);
  });
});

describe('the legend prior (test_priors)', () => {
  const OUT: [string[], boolean, string[]][] = [
    [['gleaming-anvil', 'thunder-crown'], false, ['OGN-914', 'OGN-915', 'OGN-916']],
    [['gleaming-anvil', 'thunder-crown'], true, ['OGN-914', 'OGN-915', 'OGN-916', 'OGN-920']],
    [['gleaming-anvil'], true, ['OGN-914', 'OGN-915', 'OGN-916', 'OGN-920', 'VEN-906', 'VEN-912']],
    [
      ['thunder-crown'],
      true,
      ['OGN-904', 'OGN-905', 'OGN-913', 'OGN-914', 'OGN-915', 'OGN-916', 'OGN-918', 'OGN-918a', 'OGN-919', 'OGN-920', 'OGN-921', 'SFD-901', 'SFD-902', 'SFD-902a', 'SFD-951', 'SFD-952', 'SFD-952*', 'SFD-P01', 'VEN-902'],
    ],
    [['silent-loom'], false, ['OGN-914', 'OGN-915', 'OGN-916', 'VEN-906', 'VEN-912']],
  ];

  it.each(OUT)('flags the rows Python flags: %j, runes held %s', (legends, runes, want) => {
    const { mask, fits } = legendMask(ROWS, legends, { runes });
    expect(leftOut(mask)).toEqual(want);
    expect(mask.length).toBe(ROWS.length);
    expect(fits).toEqual(legends.map((lg) => domains(ROWS.find((r) => r.card_id === lg)!)));
  });

  it("keeps the legend's own card, battlefields and tokens", () => {
    const { mask } = legendMask(ROWS, ['thunder-crown']); // runes held to the legend by default
    expect([...kept(ROWS, mask)].sort()).toEqual(['OGN-907/en', 'OGN-910/en', 'OGN-911/en', 'OGN-917/en', 'OGN-922/en', 'UNL-909/en', 'UNL-T91/en', 'VEN-906/en', 'VEN-912/en', 'VEN-T92/en']);
    expect(legendMask(ROWS, ['thunder-crown'], { tokens: tokenRows(ROWS).mask }).mask).toEqual(mask);
  });

  it('lets a row without domains fit any legend', () => {
    const rows: CatalogRow[] = ROWS.map((r) => ({ ...r }));
    delete rows[14]!.domains; // Blaze Fist, Fury: ruled out while it has its domains
    rows[15]!.domains = null; // Iron Wall, Body
    const { mask } = legendMask(rows, ['thunder-crown']);
    expect([mask[14], mask[15], mask[16]]).toEqual([1, 1, 0]);
    const bare = ROWS.map(({ domains: _d, ...r }) => r as CatalogRow); // a catalogue without domains
    expect(legendMask(bare, ['thunder-crown']).mask.every((m) => m === 1)).toBe(true);
  });
});

describe('names', () => {
  it('normalise across formats (test_decklist: test_names_normalise_across_formats)', () => {
    expect(normalise('Fakesmith - Hammerer')).toBe('fakesmith hammerer');
    expect(normalise('fakesmith, HAMMERER')).toBe('fakesmith hammerer');
    expect(normalise("Zed'Ka")).toBe('zedka');
    expect(normalise('Zedka')).toBe('zedka');
    expect(normalise('Éclat–Noir')).toBe('eclat noir');
    expect(baseName('Squire (QX)')).toBe('Squire');
  });

  // test_priors.NAMES: name, normalise(name), base_name(name), as Python gives them
  const NAMES: [string, string, string][] = [
    ['Fakesmith - Hammerer', 'fakesmith hammerer', 'Fakesmith - Hammerer'],
    ["Zed'Ka", 'zedka', "Zed'Ka"],
    ['Kha’Zix', 'khazix', 'Kha’Zix'],
    ['Rek`Sai', 'reksai', 'Rek`Sai'],
    ['Éclat–Noir', 'eclat noir', 'Éclat–Noir'],
    ['Crème Brûlée', 'creme brulee', 'Crème Brûlée'],
    ['Squire (QX)', 'squire qx', 'Squire'],
    ['Recruit(ZN)', 'recruit zn', 'Recruit'],
    ['(Only)', 'only', ''],
    ['A (b) (c)', 'a b c', 'A (b)'],
    ['A (b\n)', 'a b', 'A'],
    [' A (x) ', 'a x', 'A'],
    ['\x1cA (B)\x85', 'a b', 'A'],
    ['﻿X (Y)', 'x y', '﻿X'],
    ['a_b__c', 'a b c', 'a_b__c'],
    ['Straße', 'strasse', 'Straße'],
    ['GROẞE', 'grosse', 'GROẞE'],
    ['ΣΊΣΥΦΟΣ', 'σισυφοσ', 'ΣΊΣΥΦΟΣ'],
    ['ﬁnal ﬂight', 'final flight', 'ﬁnal ﬂight'],
    ['Ｆｕｌｌ　Ｗｉｄｔｈ', 'full width', 'Ｆｕｌｌ　Ｗｉｄｔｈ'],
    ['x² ½', 'x2 1 2', 'x² ½'],
    ['Ⅻ', 'xii', 'Ⅻ'],
    ['İstanbul', 'istanbul', 'İstanbul'],
    ['ǅemal', 'dzemal', 'ǅemal'],
    ['ŉ', 'ʼn', 'ŉ'],
    ['ᲀᲁᲂ', 'вдо', 'ᲀᲁᲂ'],
    ['Ꭰꭰᏸ', 'ᎠᎠᏰ', 'Ꭰꭰᏸ'],
    ['假铁匠', '假铁匠', '假铁匠'],
    ['がぎぐ', 'かきく', 'がぎぐ'],
    ['한국어', '한국어', '한국어'], // NFKD: Hangul as its jamo
    ['नमस्ते', 'नमसत', 'नमस्ते'],
    ['مَرْحَبًا', 'مرحبا', 'مَرْحَبًا'],
    ['A️B', 'a b', 'A️B'],
    ['🔥 Fire', 'fire', '🔥 Fire'],
    ['𝐁𝐨𝐥𝐝', 'bold', '𝐁𝐨𝐥𝐝'],
    ['', '', ''],
  ];

  it.each(NAMES)('normalise(%j) as Python does', (name, norm, base) => {
    expect(normalise(name)).toBe(norm);
    expect(baseName(name)).toBe(base);
  });
});
