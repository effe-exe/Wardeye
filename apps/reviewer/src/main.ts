// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// DOM wiring for the reviewer. All review logic lives in state.ts (unit-tested).

import type { ReviewOption, ReviewPack } from '@rifteye/schema';
import {
  actionForKey,
  answer,
  createSession,
  current,
  move,
  openPack,
  optionFor,
  progress,
  restore,
  searchOptions,
  toAnswers,
  undo,
  type Action,
  type Session,
} from './state';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const ask = $<HTMLFormElement>('ask');
const askInput = $<HTMLInputElement>('ask-input');
const suggestions = $<HTMLUListElement>('suggestions');
const reviewer = $<HTMLInputElement>('reviewer');

let session: Session | null = null;
let shownAt = performance.now();
let shownIndex = -1;
let hits: ReviewOption[] = [];
let hitIndex = 0;

// --------------------------------------------------------------------------------------
// Persistence (per-viewer convenience only; the exported file is the real record)
// --------------------------------------------------------------------------------------

const storageKey = (packId: string) => `rifteye-reviewer:${packId}`;

function save(): void {
  if (!session) return;
  try {
    localStorage.setItem(storageKey(session.pack.id), JSON.stringify(toAnswers(session, reviewer.value)));
  } catch {
    /* storage may be unavailable (private window); the in-memory session still works */
  }
}

function saved(pack: ReviewPack): Session | null {
  try {
    const raw = localStorage.getItem(storageKey(pack.id));
    if (!raw) return null;
    const s = restore(pack, JSON.parse(raw)).session;
    return s && Object.keys(s.answers).length > 0 ? s : null;
  } catch {
    return null;
  }
}

try {
  reviewer.value = localStorage.getItem('rifteye-reviewer:name') ?? '';
} catch {
  /* ignore */
}
reviewer.addEventListener('change', () => {
  try {
    localStorage.setItem('rifteye-reviewer:name', reviewer.value.trim());
  } catch {
    /* ignore */
  }
  save();
});

function commit(next: Session): void {
  session = next;
  save();
  render();
}

// --------------------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------------------

/** Embedded pictures come from the pack itself; anything else resolves next to the page. */
const resolve = (pack: ReviewPack, path: string | undefined): string => (path ? (pack.files?.[path] ?? path) : '');

function optionTile(pack: ReviewPack, o: ReviewOption, n: number | null): HTMLElement {
  const fig = document.createElement('figure');
  fig.className = 'option';
  fig.dataset.value = o.value;
  const img = document.createElement('img');
  img.alt = o.label;
  const src = resolve(pack, o.image);
  if (src) img.src = src;
  const cap = document.createElement('figcaption');
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = n === null ? o.label : `${n}. ${o.label}`;
  cap.append(label);
  fig.append(img, cap);
  return fig;
}

function render(): void {
  $<HTMLButtonElement>('export').disabled = !session;
  $('empty').hidden = !!session;
  if (!session) {
    $('review').hidden = true;
    $('done').hidden = true;
    return;
  }
  const { pack } = session;
  const p = progress(session);
  const item = current(session);
  $('review').hidden = !item;
  $('done').hidden = !!item;
  const counts = `${p.correct} correct · ${p.wrong} wrong · ${p.unsure} can't tell`;
  if (!item) {
    $('done-counts').textContent = `${p.answered} of ${p.total} answered: ${counts}.`;
    closeAsk();
    return;
  }
  if (session.cursor !== shownIndex) {
    shownIndex = session.cursor;
    shownAt = performance.now();
    closeAsk();
  }
  $('question').textContent = pack.question;
  $('position').textContent = `${session.cursor + 1} / ${p.total}`;
  $('bar-fill').style.width = `${(100 * p.answered) / Math.max(1, p.total)}%`;
  $('counts').textContent = `${p.answered} answered · ${counts}`;

  $('evidence').replaceChildren(
    ...item.images.map((path, i) => {
      const img = document.createElement('img');
      img.src = resolve(pack, path);
      // Identity items: the crop, then its surroundings (smaller). Event items: before and after, alike.
      const context = pack.kind === 'identity' && i > 0;
      img.alt = context ? 'context' : pack.kind === 'event' ? (i === 0 ? 'before' : 'after') : 'what the model saw';
      img.className = context ? 'context' : 'crop';
      return img;
    }),
  );
  const prop = $('proposal');
  const propImg = prop.querySelector('img')!;
  const src = resolve(pack, item.proposal.image);
  if (src) propImg.src = src;
  else propImg.removeAttribute('src');
  propImg.alt = item.proposal.label;
  prop.querySelector('.label')!.textContent = item.proposal.label;
  prop.querySelector('.confidence')!.textContent =
    item.confidence === undefined ? 'model guess' : `model guess · ${Math.round(item.confidence * 100)}% sure`;
  $('note').textContent = item.note ?? '';

  const a = session.answers[item.id];
  const picked = optionFor(pack, item, a);
  const said = $('answered');
  said.className = a ? `muted ${a.verdict}` : 'muted';
  said.textContent = !a
    ? ''
    : a.verdict === 'correct'
      ? 'You said: correct.'
      : a.verdict === 'unsure'
        ? "You said: can't tell."
        : picked
          ? `You said: wrong, it is ${picked.label}.`
          : "You said: wrong (you didn't know what it is).";
  $('item').className = a?.verdict === 'correct' ? 'correct' : a?.verdict === 'wrong' ? 'wrong' : '';

  $('alternatives').replaceChildren(
    ...item.alternatives.slice(0, 9).map((o, i) => {
      const tile = optionTile(pack, o, i + 1);
      if (a?.verdict === 'wrong' && a.value === o.value) tile.classList.add('picked');
      tile.addEventListener('click', () => act({ kind: 'answer', verdict: 'wrong', value: o.value }));
      return tile;
    }),
  );
}

// --------------------------------------------------------------------------------------
// Answering
// --------------------------------------------------------------------------------------

function act(action: Action): void {
  if (!session) return;
  switch (action.kind) {
    case 'answer':
      commit(answer(session, action.verdict, action.value, performance.now() - shownAt, action.text));
      break;
    case 'ask-value':
      openAsk();
      break;
    case 'undo':
      commit(undo(session));
      break;
    case 'move':
      commit(move(session, action.by));
      break;
  }
}

function openAsk(): void {
  ask.hidden = false;
  askInput.value = '';
  showHits();
  askInput.focus();
}

function closeAsk(): void {
  ask.hidden = true;
  hits = [];
  suggestions.replaceChildren();
  if (document.activeElement === askInput) askInput.blur();
}

/** The last suggestion is always the typed name itself, for cards the list does not have. */
const TYPED = '';

function showHits(): void {
  const typed = askInput.value.trim();
  hits = searchOptions(session?.pack.vocabulary ?? [], typed, 8);
  if (typed.length >= 2) hits.push({ value: TYPED, label: typed });
  hitIndex = 0;
  suggestions.replaceChildren(
    ...hits.map((o, i) => {
      const li = document.createElement('li');
      li.className = i === 0 ? 'active' : '';
      const free = o.value === TYPED;
      li.textContent = free ? `Use “${o.label}”` : o.label;
      const small = document.createElement('small');
      small.textContent = free ? 'not in the list' : o.value;
      li.append(small);
      li.addEventListener('mousedown', (e) => {
        e.preventDefault();
        pickHit(o);
      });
      return li;
    }),
  );
}

function pickHit(o: ReviewOption): void {
  if (o.value === TYPED) act({ kind: 'answer', verdict: 'wrong', text: o.label });
  else act({ kind: 'answer', verdict: 'wrong', value: o.value });
}

askInput.addEventListener('input', showHits);
askInput.addEventListener('keydown', (e) => {
  const items = [...suggestions.children] as HTMLElement[];
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (items.length === 0) return;
    hitIndex = (hitIndex + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
    items.forEach((li, i) => li.classList.toggle('active', i === hitIndex));
  } else if (e.key === 'Escape') {
    e.preventDefault();
    closeAsk();
  }
});
ask.addEventListener('submit', (e) => {
  e.preventDefault();
  // Enter picks the highlighted name; Enter on an empty box means "wrong, I don't know what it is".
  const hit = hits[hitIndex];
  if (hit) pickHit(hit);
  else if (askInput.value.trim() === '') act({ kind: 'answer', verdict: 'wrong' });
});

document.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement;
  if (target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
  if ($<HTMLDialogElement>('help').open) return;
  if (e.key === '?') {
    $<HTMLDialogElement>('help').showModal();
    return;
  }
  if (!session) return;
  const action = actionForKey(e.key, current(session), (session.pack.vocabulary?.length ?? 0) > 0);
  if (!action) return;
  e.preventDefault();
  act(action);
});

for (const b of document.querySelectorAll<HTMLButtonElement>('#keys button')) {
  b.addEventListener('click', () => {
    if (!session) return;
    const action = actionForKey(b.dataset.key ?? '', current(session), (session.pack.vocabulary?.length ?? 0) > 0);
    if (action) act(action);
  });
}

// --------------------------------------------------------------------------------------
// Files: pack, answers
// --------------------------------------------------------------------------------------

$<HTMLInputElement>('open-pack').addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  try {
    const { pack, issues } = openPack(JSON.parse(await file.text()));
    if (!pack) {
      alert(`Not a valid RiftEye review pack:\n${issues.slice(0, 10).join('\n')}`);
      return;
    }
    const before = saved(pack);
    const n = before ? Object.keys(before.answers).length : 0;
    session = before && confirm(`Continue the ${n} answers saved for this pack?`) ? before : createSession(pack);
    shownIndex = -1;
    save();
    render();
  } catch (err) {
    alert(`Could not read ${file.name}: ${String(err)}`);
  }
});

$<HTMLInputElement>('open-answers').addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  if (!session) {
    alert('Open the pack first, then import its answers.');
    return;
  }
  try {
    const { session: s, issues } = restore(session.pack, JSON.parse(await file.text()));
    if (!s) {
      alert(`Could not use these answers:\n${issues.slice(0, 10).join('\n')}`);
      return;
    }
    shownIndex = -1;
    commit(s);
  } catch (err) {
    alert(`Could not read ${file.name}: ${String(err)}`);
  }
});

$('export').addEventListener('click', () => {
  if (!session) return;
  const doc = toAnswers(session, reviewer.value);
  const blob = new Blob([JSON.stringify(doc, null, 2) + '\n'], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${session.pack.id}.answers.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$('show-help').addEventListener('click', () => $<HTMLDialogElement>('help').showModal());
render();
