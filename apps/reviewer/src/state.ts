// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Pure review logic: no DOM, so it is unit-tested directly. A model proposes an answer for
// every item in a pack; the reviewer only says correct, wrong (and, if they know it, what is
// right) or can't tell. Those answers become labels (docs/research/04 §4.4).

import {
  validateReviewAnswers,
  validateReviewPack,
  type ReviewAnswer,
  type ReviewAnswers,
  type ReviewItem,
  type ReviewOption,
  type ReviewPack,
  type Verdict,
} from '@rifteye/schema';

export interface Session {
  pack: ReviewPack;
  /** Answers by item id. */
  answers: Readonly<Record<string, ReviewAnswer>>;
  /** Index of the item on screen; equal to items.length once everything is answered. */
  cursor: number;
  /** What each answer replaced, newest last, so Z can take it back. */
  history: readonly { itemId: string; previous?: ReviewAnswer }[];
}

export type Action =
  | { kind: 'answer'; verdict: Verdict; value?: string }
  | { kind: 'ask-value' }
  | { kind: 'undo' }
  | { kind: 'move'; by: -1 | 1 };

/** Keys a reviewer uses. `n` asks for the right value when the pack has a vocabulary. */
export function actionForKey(key: string, item: ReviewItem | undefined, hasVocabulary: boolean): Action | undefined {
  const k = key.length === 1 ? key.toLowerCase() : key;
  if (k === 'z') return { kind: 'undo' };
  if (k === 'ArrowLeft') return { kind: 'move', by: -1 };
  if (k === 'ArrowRight') return { kind: 'move', by: 1 };
  if (!item) return undefined;
  if (k === 'y') return { kind: 'answer', verdict: 'correct' };
  if (k === 's') return { kind: 'answer', verdict: 'unsure' };
  if (k === 'n') return hasVocabulary ? { kind: 'ask-value' } : { kind: 'answer', verdict: 'wrong' };
  if (/^[1-9]$/.test(k)) {
    const alt = item.alternatives[Number(k) - 1];
    if (alt) return { kind: 'answer', verdict: 'wrong', value: alt.value };
  }
  return undefined;
}

export function openPack(value: unknown): { pack?: ReviewPack; issues: string[] } {
  const issues = validateReviewPack(value).map((i) => `${i.path}: ${i.message}`);
  return issues.length > 0 ? { issues } : { pack: value as ReviewPack, issues };
}

export function createSession(pack: ReviewPack): Session {
  return { pack, answers: {}, cursor: 0, history: [] };
}

export const current = (s: Session): ReviewItem | undefined => s.pack.items[s.cursor];

/** The first unanswered item after `from`, wrapping around; items.length when none is left. */
export function nextOpen(s: Session, from: number): number {
  const n = s.pack.items.length;
  for (let k = 1; k <= n; k++) {
    const i = (from + k) % n;
    if (!s.answers[s.pack.items[i]!.id]) return i;
  }
  return n;
}

/** Answers the item on screen and moves to the next unanswered one. */
export function answer(s: Session, verdict: Verdict, value?: string, ms?: number): Session {
  const item = current(s);
  if (!item) return s;
  const a: ReviewAnswer = { itemId: item.id, verdict };
  // A value only makes sense for 'wrong', and naming the proposal itself means it was right.
  if (verdict === 'wrong' && value !== undefined && value !== '') {
    if (value === item.proposal.value) a.verdict = 'correct';
    else a.value = value;
  }
  if (ms !== undefined && Number.isFinite(ms)) a.ms = Math.max(0, Math.round(ms));
  const previous = s.answers[item.id];
  const next: Session = {
    ...s,
    answers: { ...s.answers, [item.id]: a },
    history: [...s.history, previous ? { itemId: item.id, previous } : { itemId: item.id }],
  };
  return { ...next, cursor: nextOpen(next, s.cursor) };
}

/** Takes back the most recent answer and shows that item again. */
export function undo(s: Session): Session {
  const last = s.history[s.history.length - 1];
  if (!last) return s;
  const answers = { ...s.answers };
  if (last.previous) answers[last.itemId] = last.previous;
  else delete answers[last.itemId];
  const cursor = s.pack.items.findIndex((it) => it.id === last.itemId);
  return { ...s, answers, history: s.history.slice(0, -1), cursor: cursor < 0 ? s.cursor : cursor };
}

export function move(s: Session, by: number): Session {
  if (s.cursor >= s.pack.items.length && by > 0) return s;
  const cursor = Math.min(Math.max(0, s.cursor + by), s.pack.items.length - 1);
  return { ...s, cursor };
}

export function progress(s: Session): { total: number; answered: number; correct: number; wrong: number; unsure: number } {
  const all = Object.values(s.answers);
  return {
    total: s.pack.items.length,
    answered: all.length,
    correct: all.filter((a) => a.verdict === 'correct').length,
    wrong: all.filter((a) => a.verdict === 'wrong').length,
    unsure: all.filter((a) => a.verdict === 'unsure').length,
  };
}

/** The exported file: answers in pack order. */
export function toAnswers(s: Session, reviewer?: string, now: Date = new Date()): ReviewAnswers {
  const doc: ReviewAnswers = {
    schema: 'rifteye.reviewanswers',
    version: 1,
    packId: s.pack.id,
    answers: s.pack.items.flatMap((it) => (s.answers[it.id] ? [s.answers[it.id]!] : [])),
    exportedAt: now.toISOString(),
  };
  if (reviewer !== undefined && reviewer.trim() !== '') doc.reviewer = reviewer.trim();
  return doc;
}

/** Continues from saved or exported answers; answers for items not in the pack are dropped. */
export function restore(pack: ReviewPack, value: unknown): { session?: Session; issues: string[] } {
  const issues = validateReviewAnswers(value).map((i) => `${i.path}: ${i.message}`);
  if (issues.length > 0) return { issues };
  const doc = value as ReviewAnswers;
  if (doc.packId !== pack.id) return { issues: [`these answers are for pack ${doc.packId}, not ${pack.id}`] };
  const ids = new Set(pack.items.map((it) => it.id));
  const answers: Record<string, ReviewAnswer> = {};
  for (const a of doc.answers) if (ids.has(a.itemId)) answers[a.itemId] = a;
  const s: Session = { pack, answers, cursor: 0, history: [] };
  return { session: { ...s, cursor: answers[pack.items[0]!.id] ? nextOpen(s, 0) : 0 }, issues };
}

/** The option an answer points at, for display: the proposal, an alternative or a vocabulary entry. */
export function optionFor(pack: ReviewPack, item: ReviewItem, a: ReviewAnswer | undefined): ReviewOption | undefined {
  if (!a) return undefined;
  if (a.verdict === 'correct') return item.proposal;
  if (a.value === undefined) return undefined;
  return (
    item.alternatives.find((o) => o.value === a.value) ??
    pack.vocabulary?.find((o) => o.value === a.value) ?? { value: a.value, label: a.value }
  );
}

export const fold = (s: string): string =>
  s
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

/** Ranks exact label or value, then prefix, then word prefix, then substring; shorter labels first. */
export function searchOptions(options: readonly ReviewOption[], query: string, limit = 8): ReviewOption[] {
  const q = fold(query);
  if (q === '') return [];
  const scored: { o: ReviewOption; rank: number }[] = [];
  for (const o of options) {
    const n = fold(o.label);
    let rank = -1;
    if (n === q || fold(o.value) === q) rank = 0;
    else if (n.startsWith(q)) rank = 1;
    else if (n.split(' ').some((w) => w.startsWith(q))) rank = 2;
    else if (n.includes(q) || fold(o.value).includes(q)) rank = 3;
    if (rank >= 0) scored.push({ o, rank });
  }
  scored.sort((a, b) => a.rank - b.rank || a.o.label.length - b.o.label.length || a.o.label.localeCompare(b.o.label));
  return scored.slice(0, limit).map((x) => x.o);
}
