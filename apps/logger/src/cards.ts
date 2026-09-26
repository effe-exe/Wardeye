// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Card names for the logger's autocomplete. The catalogue is a local file on the
// logger's machine (ml/ `catalog build` output); nothing is fetched or re-hosted.

export interface CardEntry {
  cardId: string;
  name: string;
  type: string;
  printingIds: string[];
}

interface Row {
  card_id?: unknown;
  name?: unknown;
  type?: unknown;
  printing_id?: unknown;
}

/** Accepts catalog.jsonl (one printing per line) or a JSON array of the same rows. */
export function parseCatalog(text: string): CardEntry[] {
  const trimmed = text.trim();
  if (trimmed === '') return [];
  let rows: Row[];
  if (trimmed.startsWith('[')) {
    const parsed: unknown = JSON.parse(trimmed);
    rows = Array.isArray(parsed) ? (parsed as Row[]) : [];
  } else {
    rows = trimmed
      .split(/\r?\n/)
      .filter((l) => l.trim() !== '')
      .map((l) => JSON.parse(l) as Row);
  }
  const byCard = new Map<string, CardEntry>();
  for (const r of rows) {
    if (typeof r.card_id !== 'string' || typeof r.name !== 'string') continue;
    const entry = byCard.get(r.card_id) ?? { cardId: r.card_id, name: r.name, type: typeof r.type === 'string' ? r.type : '', printingIds: [] };
    if (typeof r.printing_id === 'string' && !entry.printingIds.includes(r.printing_id)) entry.printingIds.push(r.printing_id);
    byCard.set(r.card_id, entry);
  }
  return [...byCard.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export const fold = (s: string): string =>
  s
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

/** Ranks exact name, then prefix, then word prefix, then substring; shorter names first. */
export function searchCards(cards: readonly CardEntry[], query: string, limit = 8): CardEntry[] {
  const q = fold(query);
  if (q === '') return [];
  const scored: { c: CardEntry; rank: number }[] = [];
  for (const c of cards) {
    const n = fold(c.name);
    let rank = -1;
    if (n === q) rank = 0;
    else if (n.startsWith(q)) rank = 1;
    else if (n.split(' ').some((w) => w.startsWith(q))) rank = 2;
    else if (n.includes(q)) rank = 3;
    else if (c.printingIds.some((p) => p.toLowerCase() === query.trim().toLowerCase())) rank = 0;
    if (rank >= 0) scored.push({ c, rank });
  }
  scored.sort((a, b) => a.rank - b.rank || a.c.name.length - b.c.name.length || a.c.name.localeCompare(b.c.name));
  return scored.slice(0, limit).map((s) => s.c);
}
