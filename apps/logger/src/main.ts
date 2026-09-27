// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// DOM wiring for the timeline logger. All timeline logic lives in state.ts (unit-tested).

import { ZONES, type EventType, type Player, type TimelineEvent, type Zone } from '@rifteye/schema';
import { parseCatalog, searchCards, type CardEntry } from './cards';
import {
  CARD_EVENTS,
  HOTKEYS,
  addEvent,
  createState,
  defaultZone,
  formatTime,
  fromDocument,
  removeEvent,
  setMatchInfo,
  sortedEvents,
  toDocument,
  undoLast,
  updateEvent,
  type EventInput,
  type LoggerState,
} from './state';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const video = $<HTMLVideoElement>('video');
const entry = $<HTMLFormElement>('entry');
const entryCard = $<HTMLInputElement>('entry-card');
const suggestions = $<HTMLUListElement>('suggestions');
const eventsList = $<HTMLOListElement>('events');

let state: LoggerState | null = null;
let cards: CardEntry[] = [];
let activePlayer: Player = 'A';
let pending: { type: EventType; t: number; editingId?: string; wasPlaying: boolean } | null = null;
const selectedCard = new WeakMap<HTMLInputElement, string>(); // input -> cardId
let suggestFor: HTMLInputElement | null = null;
let suggestIndex = 0;

// --------------------------------------------------------------------------------------
// Persistence (per-viewer convenience only; the exported file is the real record)
// --------------------------------------------------------------------------------------

const storageKey = (ref: string) => `rifteye-logger:${ref}`;

function save(): void {
  if (!state) return;
  try {
    localStorage.setItem(storageKey(state.doc.match.source.ref), JSON.stringify(state.doc));
  } catch {
    /* storage may be unavailable (private window); the in-memory state still works */
  }
}

function restore(ref: string): LoggerState | null {
  try {
    const raw = localStorage.getItem(storageKey(ref));
    if (!raw) return null;
    return fromDocument(JSON.parse(raw)).state ?? null;
  } catch {
    return null;
  }
}

function commit(next: LoggerState): void {
  state = next;
  save();
  render();
}

// --------------------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------------------

const cardName = (cardId: string | null | undefined): string => {
  if (cardId === null) return 'unidentified card';
  if (cardId === undefined) return '';
  return cards.find((c) => c.cardId === cardId)?.name ?? cardId;
};

const label = (type: EventType): string => type.replace(/_/g, ' ');

function render(): void {
  $<HTMLButtonElement>('export').disabled = !state;
  $('player-chip').textContent = `Player ${activePlayer}`;
  $('player-chip').classList.toggle('b', activePlayer === 'B');
  $('card-count').textContent = cards.length ? `${cards.length} cards loaded` : 'no cards loaded';
  eventsList.replaceChildren(
    ...(state ? sortedEvents(state) : []).map((ev) => {
      const li = document.createElement('li');
      li.dataset.id = ev.id;
      const time = document.createElement('span');
      time.className = 'time';
      time.textContent = formatTime(ev.t);
      time.title = 'Jump to this moment';
      time.addEventListener('click', () => {
        video.currentTime = ev.t;
      });
      const who = document.createElement('span');
      who.className = 'who';
      who.textContent = ev.player ?? '·';
      const what = document.createElement('span');
      what.className = 'what';
      what.textContent = describe(ev);
      if (ev.zone || ev.note) {
        const small = document.createElement('small');
        small.textContent = [ev.zone, ev.note].filter(Boolean).join(' · ');
        what.append(small);
      }
      what.addEventListener('dblclick', () => openEntry(ev.type, ev.t, ev));
      const del = document.createElement('button');
      del.type = 'button';
      del.textContent = '×';
      del.title = 'Delete';
      del.addEventListener('click', () => state && commit(removeEvent(state, ev.id)));
      li.append(time, who, what, del);
      return li;
    }),
  );
}

function describe(ev: TimelineEvent): string {
  if (ev.type === 'score_changed' && ev.score) return `score ${ev.score.A}–${ev.score.B}`;
  const name = ev.card ? cardName(ev.card.cardId) : '';
  return name ? `${label(ev.type)}: ${name}` : label(ev.type);
}

// --------------------------------------------------------------------------------------
// Card autocomplete (entry form and legend fields)
// --------------------------------------------------------------------------------------

function showSuggestions(input: HTMLInputElement): void {
  const hits = searchCards(cards, input.value, 8);
  suggestFor = input;
  suggestIndex = 0;
  if (hits.length === 0) {
    suggestions.hidden = true;
    return;
  }
  suggestions.replaceChildren(
    ...hits.map((c, i) => {
      const li = document.createElement('li');
      li.dataset.cardId = c.cardId;
      li.className = i === 0 ? 'active' : '';
      li.textContent = c.name;
      const small = document.createElement('small');
      small.textContent = c.type;
      li.append(small);
      li.addEventListener('mousedown', (e) => {
        e.preventDefault();
        pick(input, c);
      });
      return li;
    }),
  );
  const box = input.getBoundingClientRect();
  const side = $('side').getBoundingClientRect();
  suggestions.style.top = `${box.bottom - side.top + $('side').scrollTop + 4}px`;
  suggestions.style.left = `${box.left - side.left}px`;
  suggestions.style.width = `${box.width}px`;
  suggestions.hidden = false;
}

function pick(input: HTMLInputElement, c: CardEntry): void {
  input.value = c.name;
  selectedCard.set(input, c.cardId);
  suggestions.hidden = true;
  if (input.dataset.role === 'legend') syncMatch();
}

function onCardKey(e: KeyboardEvent): void {
  const input = e.target as HTMLInputElement;
  const items = [...suggestions.children] as HTMLElement[];
  if (suggestions.hidden || suggestFor !== input || items.length === 0) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    suggestIndex = (suggestIndex + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
    items.forEach((li, i) => li.classList.toggle('active', i === suggestIndex));
  } else if (e.key === 'Enter') {
    e.preventDefault(); // pick, don't submit: the next Enter saves
    const id = items[suggestIndex]?.dataset.cardId;
    const c = cards.find((x) => x.cardId === id);
    if (c) pick(input, c);
  } else if (e.key === 'Escape') {
    e.stopPropagation();
    suggestions.hidden = true;
  }
}

for (const input of document.querySelectorAll<HTMLInputElement>('.card-input')) {
  input.addEventListener('input', () => {
    selectedCard.delete(input);
    showSuggestions(input);
  });
  input.addEventListener('keydown', onCardKey);
  input.addEventListener('blur', () => setTimeout(() => (suggestions.hidden = true), 100));
}

// --------------------------------------------------------------------------------------
// Match info
// --------------------------------------------------------------------------------------

function syncMatch(): void {
  if (!state) return;
  const players: { A?: string; B?: string } = {};
  const a = $<HTMLInputElement>('player-a').value.trim();
  const b = $<HTMLInputElement>('player-b').value.trim();
  if (a) players.A = a;
  if (b) players.B = b;
  const legends: { A?: string; B?: string } = {};
  const la = selectedCard.get($<HTMLInputElement>('legend-a'));
  const lb = selectedCard.get($<HTMLInputElement>('legend-b'));
  if (la) legends.A = la;
  if (lb) legends.B = lb;
  commit(setMatchInfo(state, { title: $<HTMLInputElement>('match-title').value.trim(), players, legends }));
}

for (const id of ['match-title', 'player-a', 'player-b']) $(id).addEventListener('change', syncMatch);

function fillMatch(): void {
  if (!state) return;
  const m = state.doc.match;
  $<HTMLInputElement>('match-title').value = m.title ?? '';
  $<HTMLInputElement>('player-a').value = m.players?.A ?? '';
  $<HTMLInputElement>('player-b').value = m.players?.B ?? '';
  for (const [side, id] of [['A', 'legend-a'], ['B', 'legend-b']] as const) {
    const input = $<HTMLInputElement>(id);
    const cardId = m.legends?.[side];
    input.value = cardId ? cardName(cardId) : '';
    if (cardId) selectedCard.set(input, cardId);
  }
}

// --------------------------------------------------------------------------------------
// Event entry
// --------------------------------------------------------------------------------------

const zoneSelect = $<HTMLSelectElement>('entry-zone');
zoneSelect.replaceChildren(
  new Option('—', ''),
  ...ZONES.map((z) => new Option(z.replace(/_/g, ' '), z)),
);

function openEntry(type: EventType, t: number, existing?: TimelineEvent): void {
  if (!state) return;
  pending = { type, t, wasPlaying: !video.paused };
  if (existing) pending.editingId = existing.id;
  video.pause();
  $('entry-type').textContent = label(type);
  $('entry-time').textContent = formatTime(t);
  const gameLevel = type === 'game_start' || type === 'game_end';
  $<HTMLSelectElement>('entry-player').value = existing ? (existing.player ?? '') : gameLevel ? '' : activePlayer;
  zoneSelect.value = (existing ? existing.zone : defaultZone(type)) ?? '';
  const wantsCard = CARD_EVENTS.has(type);
  $('entry-card-row').hidden = !wantsCard;
  $('entry-unknown-row').hidden = !wantsCard;
  $('entry-score-row').hidden = type !== 'score_changed';
  entryCard.value = existing?.card ? cardName(existing.card.cardId) : '';
  selectedCard.delete(entryCard);
  if (existing?.card?.cardId) selectedCard.set(entryCard, existing.card.cardId);
  $<HTMLInputElement>('entry-unknown').checked = existing?.card?.cardId === null;
  $<HTMLInputElement>('entry-note').value = existing?.note ?? '';
  if (existing?.score) {
    $<HTMLInputElement>('score-a').value = String(existing.score.A);
    $<HTMLInputElement>('score-b').value = String(existing.score.B);
  }
  entry.hidden = false;
  // Card events start in the card field; everything else on Save, so Enter logs it at once.
  (wantsCard ? entryCard : entry.querySelector<HTMLButtonElement>('button[type="submit"]'))?.focus();
}

function closeEntry(resume: boolean): void {
  // Release focus before hiding the form. Chrome moves focus off a hidden element a frame later,
  // and if the next hotkey has reopened the form by then, that late move would steal its focus.
  if (entry.contains(document.activeElement)) (document.activeElement as HTMLElement).blur();
  entry.hidden = true;
  suggestions.hidden = true;
  if (resume && pending?.wasPlaying) void video.play();
  pending = null;
}

entry.addEventListener('submit', (e) => {
  e.preventDefault();
  if (!state || !pending) return;
  const player = $<HTMLSelectElement>('entry-player').value;
  const input: EventInput = { t: pending.t, type: pending.type, player: player === 'A' || player === 'B' ? player : null };
  const zone = zoneSelect.value;
  if (zone) input.zone = zone as Zone;
  if (CARD_EVENTS.has(pending.type)) {
    const picked = selectedCard.get(entryCard);
    if ($<HTMLInputElement>('entry-unknown').checked) input.cardId = null;
    else if (picked) input.cardId = picked;
    else if (entryCard.value.trim()) {
      const best = searchCards(cards, entryCard.value, 1)[0];
      if (best) input.cardId = best.cardId;
    }
  }
  if (pending.type === 'score_changed') {
    input.score = { A: Number($<HTMLInputElement>('score-a').value) || 0, B: Number($<HTMLInputElement>('score-b').value) || 0 };
  }
  const note = $<HTMLInputElement>('entry-note').value;
  if (note.trim()) input.note = note;
  commit(pending.editingId ? updateEvent(state, pending.editingId, input) : addEvent(state, input).state);
  closeEntry(true);
});

$('entry-cancel').addEventListener('click', () => closeEntry(true));
entry.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    e.preventDefault();
    closeEntry(true);
  }
});

// --------------------------------------------------------------------------------------
// Global keyboard shortcuts
// --------------------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement;
  // Enter saves an open entry even if focus has wandered out of the form.
  if (!entry.hidden && e.key === 'Enter' && !entry.contains(target) && !target.closest('input, select, textarea')) {
    e.preventDefault();
    entry.requestSubmit();
    return;
  }
  if (target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (!entry.hidden || !state) return;
  const key = e.key.toLowerCase();
  const step = e.shiftKey ? 10 : 2;
  if (e.key === ' ') {
    e.preventDefault();
    if (video.paused) void video.play();
    else video.pause();
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    video.currentTime = Math.max(0, video.currentTime + (e.key === 'ArrowLeft' ? -step : step));
  } else if (e.key === ',' || e.key === '.') {
    video.pause();
    video.currentTime = Math.max(0, video.currentTime + (e.key === ',' ? -1 : 1) / 30);
  } else if (key === 'a' || key === 'b') {
    activePlayer = key === 'a' ? 'A' : 'B';
    render();
  } else if (key === 'z') {
    commit(undoLast(state));
  } else if (key === '?') {
    $<HTMLDialogElement>('help').showModal();
  } else if (HOTKEYS[key]) {
    e.preventDefault();
    openEntry(HOTKEYS[key] as EventType, video.currentTime);
  }
});

const tick = () => {
  $('clock').textContent = formatTime(video.currentTime || 0);
  requestAnimationFrame(tick);
};
requestAnimationFrame(tick);

// --------------------------------------------------------------------------------------
// Files: video, catalogue, import / export
// --------------------------------------------------------------------------------------

$<HTMLInputElement>('open-video').addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  video.src = URL.createObjectURL(file);
  video.hidden = false;
  $('empty').hidden = true;
  const ref = file.name;
  const saved = restore(ref);
  state = saved && confirm(`Continue the ${saved.doc.events.length} events saved for ${ref}?`) ? saved : createState({ kind: 'file', ref });
  fillMatch();
  save();
  render();
});

$<HTMLInputElement>('open-cards').addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    cards = parseCatalog(await file.text());
  } catch (err) {
    alert(`Could not read ${file.name}: ${String(err)}`);
  }
  fillMatch();
  render();
});

$<HTMLInputElement>('open-timeline').addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const { state: imported, issues } = fromDocument(JSON.parse(await file.text()));
    if (!imported) {
      alert(`Not a valid RiftEye timeline:\n${issues.slice(0, 10).join('\n')}`);
      return;
    }
    state = imported;
    fillMatch();
    save();
    render();
  } catch (err) {
    alert(`Could not read ${file.name}: ${String(err)}`);
  }
});

$('export').addEventListener('click', () => {
  if (!state) return;
  const doc = toDocument(state);
  const blob = new Blob([JSON.stringify(doc, null, 2) + '\n'], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const base = state.doc.match.source.ref.replace(/\.[^.]+$/, '') || state.doc.match.matchId;
  a.download = `${base}.timeline.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$('show-help').addEventListener('click', () => $<HTMLDialogElement>('help').showModal());
render();
