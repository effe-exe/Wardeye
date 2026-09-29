// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Riot's public card gallery, for the Chrome Web Store build (decisions D-015 and D-025): that build carries the models and
// an embedding gallery keyed by printing id, and no card name, type, text or image. The engine document reads the gallery's
// card list once (one request, about 3 MB of JSON) and keeps the names and types in memory; a card's picture is fetched from Riot's
// image server when the overlay shows it, and the last 64 are kept in memory. Nothing is stored anywhere by the extension
// (the browser's own HTTP cache may keep the responses), and nothing is sent but these requests (docs/PRIVACY.md).
//
// The list's reading is ml/rifteye_ml/catalog.py's (`_CODE`, `slugify`, `parse_code`, `variant_of`, `from_feed`), ported
// line for line: a printing has the same card id here as in the gallery's own catalogue. What is kept of an item is what the
// engine reads (printing id, card id, name, type) and what the legend rule will (domains, variant, a Legend's tags), and the
// picture's address; never the card's text.
//
// If the list cannot be read, nothing else stops: the engine starts with each printing named by its id, and the list is
// asked for again a minute later, at most five times (`Feed`).

import type { CatalogRow } from '@rifteye/engine';
import { b64Of } from './base64';

/** The gallery's list of cards, in English; `from` counts items, and `limit` takes the whole list (2000 answers with all of it). */
export const FEED_URL = 'https://content.publishing.riotgames.com/publishing-content/v2.0/public/channel/riftbound_website/list/riftbound_gallery_cards';
/** Where the pictures are: the only place a card's picture is fetched from. */
export const IMAGE_ORIGIN = 'https://cmsassets.rgpub.io';
/** What is added to a picture's address: 400 px wide, as a JPEG (about 42 KB; the original is a 778 KB PNG). The address has
 * a query already (`?accountingTag=RB`), which stays. */
export const PICTURE_QUERY = 'w=400&fm=jpg&q=80';
/** How many items one request asks for: the whole list, which holds about 1,200. Riot's pages are not stable from one request to the
 * next: measured on 29 September 2026, the same seven pages of 200, asked for twice, came back in another order, one printing
 * doubled and another missing. So the list is read in one request, and page by page only when it outgrows one (`Feed.readAll`). */
export const PAGE_SIZE = 2000;
/** A list is asked for again this long after a try that failed, up to `MAX_RETRIES` times. */
export const RETRY_MS = 60_000;
export const MAX_RETRIES = 5;
/** A stop for a feed that never says where its pages end (the gallery has seven). */
const MAX_PAGES = 30;
/** Pictures kept in memory. */
export const PICTURES_KEPT = 64;
/** A picture larger than this is not one of the gallery's. */
const MAX_PICTURE_BYTES = 2_000_000;

export type Variant = 'token' | 'signature' | 'alt_art' | 'overnumbered' | 'standard';

/** One printing of the gallery. */
export interface FeedCard {
  printing_id: string;
  /** The card's identity across printings: its slugified name (a champion unit's name includes its subtitle). */
  card_id: string;
  name: string;
  /** The type labels joined by a space: Unit, Spell, Gear, Legend, Battlefield, Rune, Token, ... */
  type: string;
  /** The domains' labels, as from_feed writes them. */
  domains: string[];
  variant: Variant;
  /** A Legend's tags (its champion among them); no other card's are kept. */
  tags?: string[];
  /** The card's picture on Riot's image server. */
  image_url: string;
}

/** What the engine's catalogue holds of a printing: a card less its picture's address. */
export type CardRow = Omit<FeedCard, 'image_url'>;

// --- the reading of the list: ml/rifteye_ml/catalog.py ------------------------------------------------------------------

// catalog.py's `_CODE`. (Its `\d` matches the digits of every script, this one only ASCII digits: the gallery's codes have no others.)
const CODE = /^(?<set>[A-Z]{2,4})-(?<num>[A-Z]*\d+[a-z]?\*?)(?:\/(?<total>\d+))?$/;

export interface Code {
  set: string;
  num: string;
  total: number | null;
  printing_id: string;
}

/** 'OGN-007a/298' -> { set: 'OGN', num: '007a', total: 298, printing_id: 'OGN-007a' }; null when it is no collector code. */
export function parseCode(code: string): Code | null {
  const m = CODE.exec(code.trim());
  if (!m) return null;
  const { set, num, total } = m.groups as { set: string; num: string; total?: string };
  return { set, num, total: total ? Number(total) : null, printing_id: `${set}-${num}` };
}

export function variantOf(code: Code, cardType: string, isAltArt: boolean): Variant {
  const num = code.num;
  if (cardType.toLowerCase() === 'token' || num.startsWith('T')) return 'token';
  if (num.endsWith('*')) return 'signature';
  if (isAltArt || num.endsWith('a') || num.endsWith('b')) return 'alt_art';
  const digits = num.replace(/\D/g, '');
  if (code.total && digits && Number(digits) > code.total) return 'overnumbered';
  return 'standard';
}

const ANNOTATION = /\s*\((?:alternate art|alt art|showcase|signature|overnumbered|promo|foil)[^)]*\)\s*/giu;

/** Gameplay identity from a card name: the same name is the same card. Accents fold to ASCII; letters of other scripts are
 * kept. (catalog.py drops the marks with a combining class; this drops every mark, which is the same for Latin letters.) */
export function slugify(name: string): string {
  const s = name.replace(ANNOTATION, ' ').normalize('NFKD').replace(/\p{M}/gu, '');
  return s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-+|-+$/g, '') || 'unknown';
}

const object = (v: unknown): Record<string, unknown> => (typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const text = (v: unknown): string => (typeof v === 'string' ? v.trim() : '');
/** A label as it is, for the ones that are strings (catalog.py joins `label or ""`). */
const label = (v: unknown): string => (typeof v === 'string' ? v : '');
const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((s): s is string => typeof s === 'string' && s !== '') : []);

/** The gallery's items as printings: those with a collector code, a picture and a name, the first of each printing id. */
export function parseFeed(items: readonly unknown[]): FeedCard[] {
  const out = new Map<string, FeedCard>();
  for (const raw of items) {
    const it = object(raw);
    const code = parseCode(typeof it.publicCode === 'string' ? it.publicCode : '');
    const url = object(it.cardImage).url;
    const name = text(it.name);
    if (!code || typeof url !== 'string' || url === '' || !name || out.has(code.printing_id)) continue;
    const labels = object(it.cardType).type;
    const type = (Array.isArray(labels) ? labels : []).map((t) => label(object(t).label)).join(' ').trim();
    // champion units share a name and differ by subtitle ("Ahri, Alluring" and "Ahri, Inquisitive" are two cards); other
    // subtitles are annotations, such as the champion on a signature spell or "Starter" on a starter-deck legend
    const subtitle = text(it.subtitle);
    const full = subtitle && type === 'Unit' ? `${name}, ${subtitle}` : name;
    const domainValues = object(it.domain).values;
    out.set(code.printing_id, {
      printing_id: code.printing_id,
      card_id: slugify(full),
      name: full,
      type,
      domains: strings(Array.isArray(domainValues) ? domainValues.map((v) => object(v).label) : []),
      variant: variantOf(code, type, false),
      ...(type === 'Legend' ? { tags: strings(object(it.tags).tags) } : {}),
      image_url: url,
    });
  }
  return [...out.values()];
}

// --- the list, once, and again when it failed ---------------------------------------------------------------------------

/** What the list needs of the world, so that tests need no network and no clock. */
export interface FeedEnv {
  /** The JSON body of a GET; throws when the request fails, is not answered 2xx, or is not JSON. */
  getJson(url: string): Promise<unknown>;
  /** Runs `run` once, `ms` later. */
  later(run: () => void, ms: number): void;
}

const messageOf = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export class Feed {
  private cards = new Map<string, FeedCard>();
  private first: Promise<void> | null = null;
  private retries = 0;
  private readonly listeners: (() => void)[] = [];

  constructor(
    private readonly env: FeedEnv,
    private readonly retryMs = RETRY_MS,
    private readonly maxRetries = MAX_RETRIES,
    private readonly pageSize = PAGE_SIZE,
  ) {}

  /** Starts reading the list (once) and says when the first try is over, whether it worked or not; it does not fail. */
  settled(): Promise<void> {
    this.first ??= this.attempt();
    return this.first;
  }

  /** The list was read: for the first time, or after a try that failed. Called once for each try that works. */
  onLoaded(listener: () => void): void {
    this.listeners.push(listener);
  }

  get loaded(): boolean {
    return this.cards.size > 0;
  }

  /** The printings the list names, by printing id, less their pictures; empty until the list is read. */
  rows(): CatalogRow[] {
    return [...this.cards.values()].map(({ image_url: _picture, ...row }) => row);
  }

  imageUrl(printingId: string): string | undefined {
    return this.cards.get(printingId)?.image_url;
  }

  private async attempt(): Promise<void> {
    try {
      const cards = await this.readAll();
      this.cards = new Map(cards.map((c) => [c.printing_id, c]));
      console.info(`Wardeye: the card list names ${cards.length} printings`);
      for (const listener of this.listeners) listener();
    } catch (e) {
      const more = this.retries < this.maxRetries;
      console.warn(`Wardeye: the card list could not be read (${messageOf(e)}); ${more ? `trying again in ${Math.round(this.retryMs / 1000)} s` : 'not trying again'}`);
      if (more) {
        this.retries++;
        this.env.later(() => void this.attempt(), this.retryMs);
      }
    }
  }

  /** The whole list, all of it or nothing: one request while it fits in one. A longer list is read page by page twice, keeping every
   * printing either pass saw, since a printing can move across a page's edge between two requests (`PAGE_SIZE`). */
  private async readAll(): Promise<FeedCard[]> {
    const all = new Map<string, FeedCard>();
    if ((await this.pass(all)) > 1) await this.pass(all);
    if (all.size === 0) throw new Error('the list names no card');
    return [...all.values()];
  }

  /** One reading of the pages, from = 0, pageSize, ... until the pages the feed says it has, into `all`; how many pages it read. */
  private async pass(all: Map<string, FeedCard>): Promise<number> {
    const size = this.pageSize;
    let pages = 1;
    let read = 0;
    for (let page = 0; page < pages; page++) {
      const body = object(await this.env.getJson(`${FEED_URL}?locale=en_US&from=${page * size}&limit=${size}`));
      if (!Array.isArray(body.data)) throw new Error('the list has no data');
      read++;
      for (const c of parseFeed(body.data)) if (!all.has(c.printing_id)) all.set(c.printing_id, c);
      const meta = object(body.metadata);
      const total =
        typeof meta.totalPages === 'number' ? meta.totalPages : typeof meta.totalItems === 'number' ? Math.ceil(meta.totalItems / size) : body.data.length >= size ? page + 2 : page + 1;
      pages = Math.min(total, MAX_PAGES);
      if (body.data.length === 0) break;
    }
    return read;
  }
}

// --- the pictures ------------------------------------------------------------------------------------------------------

/** A card's picture at 400 px as a JPEG, from its address in the list; null for an address that is not on Riot's image server. */
export function pictureUrl(imageUrl: string): string | null {
  let url: URL;
  try {
    url = new URL(imageUrl);
  } catch {
    return null;
  }
  if (url.origin !== IMAGE_ORIGIN) return null;
  url.hash = '';
  return `${url.href}${url.search ? '&' : '?'}${PICTURE_QUERY}`;
}

/** Whether the bytes are a JPEG, a PNG or a WebP, of a size a card's picture can be. */
export function looksLikePicture(bytes: Uint8Array): boolean {
  if (bytes.length < 12 || bytes.length > MAX_PICTURE_BYTES) return false;
  const jpeg = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
  const webp = String.fromCharCode(...bytes.subarray(0, 4)) === 'RIFF' && String.fromCharCode(...bytes.subarray(8, 12)) === 'WEBP';
  return jpeg || png || webp;
}

export interface PicturesEnv {
  /** A card's picture address in the list (waiting for the list when it is on its way), or undefined. */
  imageUrl(printingId: string): Promise<string | undefined>;
  /** The bytes of a GET; null when the answer is not a success. Throws when the request fails. */
  getBytes(url: string): Promise<Uint8Array | null>;
}

/** The pictures the overlay asks for, base64, with the last `limit` kept in memory (and never on disk). A card whose picture
 * could not be had is asked for again next time. */
export class Pictures {
  private readonly kept = new Map<string, string>();
  private readonly pending = new Map<string, Promise<string | null>>();

  constructor(
    private readonly env: PicturesEnv,
    private readonly limit = PICTURES_KEPT,
  ) {}

  get size(): number {
    return this.kept.size;
  }

  get(printingId: string): Promise<string | null> {
    const hit = this.kept.get(printingId);
    if (hit !== undefined) {
      this.kept.delete(printingId); // the one used last is the last to go
      this.kept.set(printingId, hit);
      return Promise.resolve(hit);
    }
    let p = this.pending.get(printingId);
    if (!p) {
      p = this.fetch(printingId).finally(() => this.pending.delete(printingId));
      this.pending.set(printingId, p);
    }
    return p;
  }

  private async fetch(printingId: string): Promise<string | null> {
    try {
      const listed = await this.env.imageUrl(printingId);
      const url = listed === undefined ? null : pictureUrl(listed);
      if (!url) return null;
      const bytes = await this.env.getBytes(url);
      if (!bytes || !looksLikePicture(bytes)) return null;
      const jpeg = b64Of(bytes);
      this.kept.set(printingId, jpeg);
      while (this.kept.size > this.limit) this.kept.delete(this.kept.keys().next().value as string);
      return jpeg;
    } catch {
      return null; // no picture this time
    }
  }
}
