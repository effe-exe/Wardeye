// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// ml/tests/test_decklist.py's synthetic catalogue, row for row, for the priors' tests and the legend rule's in the
// recognizer. Every name and collector number here is invented.

import type { CatalogRow } from '../src/types';

function row(pid: string, cardId: string, name: string, kind: string, domains: string[], variant = 'standard', language = 'en', tags: string[] = []): CatalogRow {
  const [set, number] = pid.split('-') as [string, string];
  return {
    printing_id: pid,
    card_id: cardId,
    name,
    type: kind,
    domains,
    variant,
    language,
    tags,
    set_code: set,
    collector_number: number,
    orientation: kind === 'Battlefield' ? 'landscape' : 'portrait',
    image_url: `https://example.invalid/${pid}-${language}.png`,
  };
}

export const ROWS: readonly CatalogRow[] = [
  // Two legends: a legend's catalogue name is its title; its champion is a tag.
  row('SFD-901', 'gleaming-anvil', 'Gleaming Anvil', 'Legend', ['Calm', 'Mind'], 'standard', 'en', ['Fakesmith']),
  row('SFD-951', 'gleaming-anvil', 'Gleaming Anvil', 'Legend', ['Calm', 'Mind'], 'overnumbered', 'en', ['Fakesmith']),
  row('VEN-912', 'thunder-crown', 'Thunder Crown', 'Legend', ['Order', 'Chaos'], 'standard', 'en', ['Fakezap']),
  row('OGN-913', 'silent-loom', 'Silent Loom', 'Legend', ['Calm', 'Mind'], 'standard', 'en', ['Fakeweaver']),
  // A champion with every kind of other printing: alt art, overnumbered, signature, a reprint, a localised one.
  row('SFD-902', 'fakesmith-hammerer', 'Fakesmith, Hammerer', 'Unit', ['Calm']),
  row('SFD-902a', 'fakesmith-hammerer', 'Fakesmith, Hammerer', 'Unit', ['Calm'], 'alt_art'),
  row('SFD-952', 'fakesmith-hammerer', 'Fakesmith, Hammerer', 'Unit', ['Calm'], 'overnumbered'),
  row('SFD-952*', 'fakesmith-hammerer', 'Fakesmith, Hammerer', 'Unit', ['Calm'], 'signature'),
  row('VEN-902', 'fakesmith-hammerer', 'Fakesmith, Hammerer', 'Unit', ['Calm']),
  row('SFD-902', 'fakesmith-hammerer', '假铁匠', 'Unit', ['Calm'], 'standard', 'zh-Hans'),
  // A special printing of it filed under another card_id.
  row('SFD-P01', 'fakesmith-hammerer-promo', 'Fakesmith, Hammerer (Promo)', 'Unit', ['Calm'], 'promo'),
  row('OGN-904', 'pocket-gadget', 'Pocket Gadget', 'Gear', ['Mind']),
  row('OGN-905', 'quick-trick', 'Quick-Trick', 'Spell', ['Mind']),
  row('VEN-906', 'spark-bolt', 'Spark Bolt', 'Spell', ['Order', 'Chaos']),
  row('OGN-914', 'blaze-fist', 'Blaze Fist', 'Unit', ['Fury']),
  row('OGN-915', 'iron-wall', 'Iron Wall', 'Gear', ['Body']),
  row('OGN-916', 'tidal-edict', 'Tidal Edict', 'Spell', ['Calm', 'Order']),
  row('OGN-917', 'plain-lantern', 'Plain Lantern', 'Gear', ['Colorless']),
  row('OGN-907', 'quiet-glade', 'Quiet Glade', 'Battlefield', ['Colorless']),
  row('UNL-909', 'far-tower', 'Far Tower', 'Battlefield', ['Colorless']),
  row('OGN-918', 'hush-rune', 'Hush Rune', 'Rune', ['Calm']),
  row('OGN-918a', 'hush-rune', 'Hush Rune', 'Rune', ['Calm'], 'alt_art'),
  row('OGN-919', 'muse-rune', 'Muse Rune', 'Rune', ['Mind']),
  row('OGN-920', 'ember-rune', 'Ember Rune', 'Rune', ['Fury']),
  // Tokens: one marked token, one of its card printed as a numbered card, and a numbered token whose name differs
  // from the marked one only by a parenthetical.
  row('UNL-T91', 'wisp', 'Wisp', 'Unit', ['Colorless'], 'token'),
  row('OGN-910', 'wisp', 'Wisp', 'Unit', ['Colorless']),
  row('VEN-T92', 'squire', 'Squire', 'Unit', ['Colorless'], 'token'),
  row('OGN-911', 'squire-qx', 'Squire (QX)', 'Unit', ['Colorless']),
  // Two cards whose names normalise alike: a section settles which is meant.
  row('OGN-921', 'zed-ka', "Zed'Ka", 'Unit', ['Mind']),
  row('OGN-922', 'zedka', 'Zedka', 'Battlefield', ['Colorless']),
];
