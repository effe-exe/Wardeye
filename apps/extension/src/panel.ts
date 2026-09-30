// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The plays panel, in the browser's side panel: it follows the active tab (or, opened as panel.html?tab=<id>, that one tab: a
// window of its own, and the tests), connects to its content script while it is open
// (a port named 'plays'), and shows what that tab's overlay knows: each player's legend and what is face up on their side of
// the table, and the plays as they happened, newest first, as the live runner's page does. A play clicked jumps the replay to
// just before it. Card pictures are asked of the worker, over a port named 'panel', as the overlay asks them. Nothing is kept
// here: the list lives in the tab (plays.ts). How it looks is panel.css; this file makes the markup and never sets a colour.

import { MARK_SHAPES, MARK_VIEWBOX } from './mark';
import type { BoardEvent } from './parts';
import { clock, isSnapshot, type Side, type Snapshot } from './plays';

const SVG = 'http://www.w3.org/2000/svg';
const RETRY_MS = 2000; // a tab whose page is loading has no content script yet: it is asked again, a few times
const RETRIES = 5;
const FIXED = Number(new URLSearchParams(location.search).get('tab')) || null; // panel.html?tab=<id>: that tab, not the active one

const statusEl = document.getElementById('status')!;
const playersEl = document.getElementById('players')!;
const playsEl = document.getElementById('plays')!;
const noteEl = document.getElementById('plays-note')!;
const listsEl = document.getElementById('lists')!;
const SLOTS = 2; // a box for each player, as the content script keeps them

let tabId: number | null = null;
let line: chrome.runtime.Port | null = null;
let retry: ReturnType<typeof setTimeout> | null = null;
let retries = 0;
let shownPlays = ''; // the plays on screen, as a key: they are drawn again only when they change
let texts: string[] = []; // the tab's decklists, as pasted
let shownLists = '';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, text = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function mark(): SVGSVGElement {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'wd-mark');
  svg.setAttribute('viewBox', MARK_VIEWBOX);
  svg.setAttribute('aria-hidden', 'true');
  for (const shape of MARK_SHAPES) {
    const e = document.createElementNS(SVG, shape.tag);
    e.setAttribute('class', shape.cls.replace('rifteye-', 'wd-'));
    for (const [name, value] of Object.entries(shape.attrs)) e.setAttribute(name, value);
    svg.append(e);
  }
  return svg;
}
document.getElementById('mark')!.append(mark());

// --- pictures ---------------------------------------------------------------------------------------------------------

let art: chrome.runtime.Port | null = null;
const pictures = new Map<string, Promise<string | null>>();
const waiting = new Map<string, (jpeg: string | null) => void>();

function artPort(): chrome.runtime.Port {
  if (art) return art;
  const p = chrome.runtime.connect({ name: 'panel' });
  p.onMessage.addListener((m: { kind?: unknown; printing_id?: unknown; jpeg?: unknown }) => {
    if (m?.kind !== 'art' || typeof m.printing_id !== 'string') return;
    waiting.get(m.printing_id)?.(typeof m.jpeg === 'string' ? m.jpeg : null);
    waiting.delete(m.printing_id);
  });
  p.onDisconnect.addListener(() => {
    art = null; // the worker was put to sleep: the next picture wakes it
    for (const [id, resolve] of waiting) {
      pictures.delete(id);
      resolve(null);
    }
    waiting.clear();
  });
  art = p;
  return p;
}

function picture(printingId: string): Promise<string | null> {
  let p = pictures.get(printingId);
  if (!p) {
    p = new Promise<string | null>((resolve) => {
      waiting.set(printingId, resolve);
      artPort().postMessage({ kind: 'art', printing_id: printingId });
    }).then((jpeg) => {
      if (!jpeg) pictures.delete(printingId); // not there this time: asked again next time
      return jpeg;
    });
    pictures.set(printingId, p);
  }
  return p;
}

/** A card's picture, small: empty until it comes, and left empty when there is none. */
function thumb(printingId: string | null, cls: string): HTMLImageElement {
  const img = el('img', cls);
  img.alt = '';
  img.decoding = 'async';
  if (printingId) {
    void picture(printingId).then((jpeg) => {
      if (jpeg) img.src = `data:image/jpeg;base64,${jpeg}`;
    });
  }
  return img;
}

// --- drawing ----------------------------------------------------------------------------------------------------------

function sideSection(s: Side): HTMLElement {
  const section = el('section', 'wd-player');
  section.append(el('h2', 'wd-eyebrow', s.label));
  const legend = el('div', 'wd-legend');
  if (s.legend) legend.append(thumb(s.legend.printing_id, 'wd-thumb'), el('span', '', s.legend.name));
  else legend.append(el('span', 'wd-muted', 'Legend not seen yet'));
  section.append(legend);
  const list = el('ul', 'wd-cards');
  for (const g of s.cards) {
    const li = el('li', '');
    const text = el('span', 'wd-card-text', g.count > 1 ? `${g.name} ×${g.count}` : g.name); // what lies under it follows in the same run
    if (g.under.length) text.append(el('span', 'wd-with', ` + ${g.under.join(', ')}`));
    li.append(thumb(g.printing_id, 'wd-thumb-sm'), text);
    list.append(li);
  }
  const bits = [
    s.runes ? `${s.runes} ${s.runes > 1 ? 'runes' : 'rune'}` : '',
    s.unsure ? `${s.unsure} unsure` : '',
    s.facedown ? `${s.facedown} face down` : '',
  ].filter(Boolean);
  if (!s.cards.length && !bits.length) list.append(el('li', 'wd-muted', 'Nothing on the table yet'));
  if (bits.length) list.append(el('li', 'wd-muted', bits.join(', ')));
  section.append(list);
  return section;
}

function playItem(e: BoardEvent, live: boolean): HTMLLIElement {
  const li = el('li', `wd-play wd-play-${e.kind.replace(/[^\w-]/g, '')}`);
  const row = live ? el('div', 'wd-play-row') : el('button', 'wd-play-row');
  if (row instanceof HTMLButtonElement) {
    row.type = 'button';
    row.title = `Watch from ${clock(Math.max(0, e.t - 2))}`;
    row.addEventListener('click', () => line?.postMessage({ kind: 'seek', t: e.t }));
  }
  row.append(el('span', 'wd-time', clock(e.t)), thumb(e.printing_id, 'wd-thumb-sm'), el('span', 'wd-play-text', e.text));
  li.append(row);
  return li;
}

/** What the engine made of a box's list: the legend it names and how many cards, or what went wrong. */
function listRead(read: Snapshot['lists'][number]['read']): HTMLElement[] {
  const what = el('span', 'wd-list-what');
  if (!read) what.textContent = 'Read with the next frame';
  else if (read.error) {
    what.textContent = `Not read: ${read.error}`;
    what.classList.add('wd-list-bad');
  } else {
    const legend = read.legends.length ? read.legends.join(' / ') : 'no legend named';
    what.textContent = `${legend} · ${read.cards} ${read.cards === 1 ? 'card' : 'cards'}`;
    if (!read.legends.length) what.classList.add('wd-list-bad'); // a list counts only for the legend it names: this one, for none
  }
  const out: HTMLElement[] = [what];
  if (read?.unmapped.length) out.push(el('span', 'wd-list-meta', `${read.unmapped.length} ${read.unmapped.length === 1 ? 'line' : 'lines'} not read: ${read.unmapped.slice(0, 3).join('; ')}`));
  return out;
}

/** One player's box: its list as read, with a way to take it out, or a place to paste one. */
function listSlot(i: number, label: string, text: string, read: Snapshot['lists'][number]['read'], draft: string): HTMLElement {
  const box = el('div', 'wd-list');
  box.dataset['slot'] = String(i);
  box.append(el('span', 'wd-list-label', `${label}'s list`));
  if (text) {
    box.append(...listRead(read));
    const remove = el('button', 'wd-list-remove', 'Remove');
    remove.type = 'button';
    remove.setAttribute('aria-label', `Remove ${label}'s list`);
    remove.addEventListener('click', () => sendSlot(i, ''));
    box.append(remove);
    box.title = text.slice(0, 400);
    return box;
  }
  const paste = el('textarea', 'wd-paste');
  paste.rows = 3;
  paste.spellcheck = false;
  paste.placeholder = 'Paste a deck code or a deckbuilder\'s export';
  paste.setAttribute('aria-label', `${label}'s decklist`);
  paste.value = draft;
  const use = el('button', 'wd-button', 'Use this list');
  use.type = 'button';
  use.disabled = !line;
  use.addEventListener('click', () => {
    if (paste.value.trim()) sendSlot(i, paste.value.trim());
  });
  box.append(paste, use);
  return box;
}

/** A box's list, sent to the tab: it goes with its next frame. */
function sendSlot(i: number, text: string): void {
  const next = Array.from({ length: SLOTS }, (_, k) => texts[k] ?? '');
  next[i] = text;
  texts = next;
  line?.postMessage({ kind: 'lists', texts });
}

/** The boxes, drawn again: what is typed in an empty one stays. */
function drawSlots(s: Snapshot | null): void {
  const drafts = new Map<string, string>();
  for (const t of listsEl.querySelectorAll<HTMLTextAreaElement>('[data-slot] textarea')) drafts.set(t.closest<HTMLElement>('[data-slot]')!.dataset['slot']!, t.value);
  const lists = s?.lists ?? [];
  listsEl.replaceChildren(
    ...Array.from({ length: SLOTS }, (_, i) =>
      listSlot(i, s?.sides[i]?.label ?? `Player ${i + 1}`, lists[i]?.text ?? '', lists[i]?.read ?? null, drafts.get(String(i)) ?? ''),
    ),
  );
}

function render(s: Snapshot | null): void {
  if (!s) {
    if (shownLists !== 'none') drawSlots(null);
    shownLists = 'none';
    statusEl.textContent = 'Open a Riftbound video on Twitch: its plays are listed here as Wardeye reads the table.';
    playersEl.replaceChildren();
    playsEl.replaceChildren();
    noteEl.hidden = true;
    shownPlays = '';
    return;
  }
  statusEl.textContent = !s.on ? 'Off in this tab: nothing is read. Turn it on from the badge, the toolbar button or Alt+R.' : s.message || 'Reading the table';
  playersEl.replaceChildren(...s.sides.map(sideSection));
  const lists = s.lists ?? [];
  texts = Array.from({ length: SLOTS }, (_, i) => lists[i]?.text ?? '');
  const listKey = JSON.stringify([lists, s.sides.map((x) => x.label)]);
  if (listKey !== shownLists) {
    shownLists = listKey;
    drawSlots(s);
  }
  const key = `${s.video}|${s.live}|${s.plays.length}|${s.plays.map((p) => `${p.t}:${p.track}`).join(',')}`;
  if (key === shownPlays) return; // the same plays: the list, and whatever has the focus in it, stays
  shownPlays = key;
  noteEl.hidden = s.plays.length > 0 && !s.live;
  noteEl.textContent = s.plays.length === 0 ? 'No play yet. Cards played on the table come here as they are named.' : 'Live: the plays come as they happen.';
  playsEl.replaceChildren(...[...s.plays].reverse().map((e) => playItem(e, s.live)));
}

// --- the tab ----------------------------------------------------------------------------------------------------------

/** The active tab's content script, connected (again, after `force`: its page reloaded). */
async function follow(force = false): Promise<void> {
  const [tab] = FIXED === null ? await chrome.tabs.query({ active: true, currentWindow: true }).catch(() => []) : [{ id: FIXED }];
  const id = tab?.id ?? null;
  if (id === tabId && line && !force) return;
  if (retry) clearTimeout(retry);
  retry = null;
  if (id !== tabId) retries = 0;
  line?.disconnect();
  line = null;
  tabId = id;
  if (id === null) return render(null);
  const p = chrome.tabs.connect(id, { name: 'plays' });
  p.onMessage.addListener((m: unknown) => {
    retries = 0;
    if (isSnapshot(m)) render(m);
  });
  p.onDisconnect.addListener(() => {
    if (line !== p) return;
    line = null;
    render(null); // not a Twitch page, or its page is loading: asked again in a moment
    if (++retries <= RETRIES) retry = setTimeout(() => void follow(true), RETRY_MS);
  });
  line = p;
}

chrome.tabs.onActivated.addListener(() => void follow());
chrome.tabs.onUpdated.addListener((id, change) => {
  if (id === tabId && change.status === 'complete') {
    retries = 0;
    void follow(true); // the page loaded again: its content script is new
  }
});
void follow();
