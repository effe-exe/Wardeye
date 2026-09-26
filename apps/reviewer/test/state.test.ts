import { describe, expect, it } from 'vitest';
import { validateReviewAnswers, type ReviewPack } from '@rifteye/schema';
import {
  actionForKey,
  answer,
  createSession,
  current,
  move,
  nextOpen,
  openPack,
  optionFor,
  progress,
  restore,
  searchOptions,
  toAnswers,
  undo,
} from '../src/state';

const opt = (value: string, label = value) => ({ value, label });
const pack = (n = 4): ReviewPack => ({
  schema: 'rifteye.reviewpack',
  version: 1,
  id: 'test-pack',
  kind: 'identity',
  question: 'Is this the card?',
  items: Array.from({ length: n }, (_, i) => ({
    id: `i${i}`,
    images: [`crop-${i}.jpg`],
    proposal: opt(`FAK-00${i}`, `Card ${i}`),
    confidence: 0.5,
    alternatives: [opt('FAK-101', 'Heroic Strike'), opt('FAK-102', 'Fake Legend')],
  })),
  vocabulary: [opt('FAK-101', 'Heroic Strike'), opt('FAK-102', 'Fake Legend'), opt('FAK-103', 'Fake Hero'), opt('FAK-104', 'Hero of Légend')],
  createdAt: '2026-09-26T00:00:00Z',
});
const at = new Date(Date.UTC(2026, 8, 26, 12));

describe('review session', () => {
  it('answers items in order and exports valid answers in pack order', () => {
    let s = createSession(pack());
    s = answer(s, 'correct', undefined, 1234.4);
    s = answer(s, 'wrong', 'FAK-102');
    s = answer(s, 'wrong');
    s = answer(s, 'unsure');
    expect(s.cursor).toBe(4);
    expect(current(s)).toBeUndefined();
    const doc = toAnswers(s, '  Fede ', at);
    expect(validateReviewAnswers(doc)).toEqual([]);
    expect(doc.reviewer).toBe('Fede');
    expect(doc.answers).toEqual([
      { itemId: 'i0', verdict: 'correct', ms: 1234 },
      { itemId: 'i1', verdict: 'wrong', value: 'FAK-102' },
      { itemId: 'i2', verdict: 'wrong' },
      { itemId: 'i3', verdict: 'unsure' },
    ]);
    expect(progress(s)).toEqual({ total: 4, answered: 4, correct: 1, wrong: 2, unsure: 1 });
  });

  it('keeps a typed name for a card the list does not have', () => {
    const s = answer(createSession(pack()), 'wrong', undefined, 900, '  mech token ');
    expect(s.answers.i0).toEqual({ itemId: 'i0', verdict: 'wrong', text: 'mech token', ms: 900 });
    expect(validateReviewAnswers(toAnswers(s, undefined, at))).toEqual([]);
    expect(optionFor(s.pack, s.pack.items[0]!, s.answers.i0)?.label).toBe('“mech token” (not in the list)');
    // A catalogue value wins over text; text never rides on other verdicts.
    expect(answer(createSession(pack()), 'wrong', 'FAK-101', undefined, 'x').answers.i0).toEqual({ itemId: 'i0', verdict: 'wrong', value: 'FAK-101' });
    expect(answer(createSession(pack()), 'unsure', undefined, undefined, 'x').answers.i0).toEqual({ itemId: 'i0', verdict: 'unsure' });
  });

  it('treats naming the proposal as correct', () => {
    const s = answer(createSession(pack()), 'wrong', 'FAK-000');
    expect(s.answers.i0).toEqual({ itemId: 'i0', verdict: 'correct' });
  });

  it('undoes answers one by one, restoring what they replaced', () => {
    let s = createSession(pack());
    s = answer(s, 'correct'); // i0
    s = move(s, -1); // back to i0
    s = answer(s, 'wrong', 'FAK-101'); // replaces i0, then moves to the next open item
    expect(s.cursor).toBe(1);
    s = undo(s);
    expect(s.cursor).toBe(0);
    expect(s.answers.i0?.verdict).toBe('correct');
    s = undo(s);
    expect(s.answers.i0).toBeUndefined();
    expect(undo(s)).toBe(s);
  });

  it('skips answered items and wraps around to ones left open', () => {
    let s = createSession(pack());
    s = move(s, 1); // leave i0 open
    s = answer(s, 'correct'); // i1
    s = answer(s, 'correct'); // i2
    s = answer(s, 'correct'); // i3 -> wraps to i0
    expect(s.cursor).toBe(0);
    expect(nextOpen(answer(s, 'correct'), 0)).toBe(4);
    expect(move(move(answer(s, 'correct'), 1), 1).cursor).toBe(4);
  });

  it('restores saved answers, drops unknown items and refuses another pack', () => {
    const p = pack();
    let s = answer(answer(createSession(p), 'correct'), 'unsure');
    const doc = toAnswers(s, undefined, at);
    doc.answers.push({ itemId: 'gone', verdict: 'correct' });
    const back = restore(p, doc).session!;
    expect(Object.keys(back.answers).sort()).toEqual(['i0', 'i1']);
    expect(back.cursor).toBe(2);
    expect(restore({ ...p, id: 'other' }, doc).issues[0]).toMatch(/not other/);
    expect(restore(p, { nope: 1 }).session).toBeUndefined();
    s = createSession(p);
    expect(restore(p, toAnswers(answer(move(s, 2), 'correct'), undefined, at)).session!.cursor).toBe(0);
  });

  it('maps keys to actions', () => {
    const p = pack();
    const item = p.items[0];
    expect(actionForKey('y', item, true)).toEqual({ kind: 'answer', verdict: 'correct' });
    expect(actionForKey('Y', item, true)).toEqual({ kind: 'answer', verdict: 'correct' });
    expect(actionForKey('2', item, true)).toEqual({ kind: 'answer', verdict: 'wrong', value: 'FAK-102' });
    expect(actionForKey('3', item, true)).toBeUndefined();
    expect(actionForKey('n', item, true)).toEqual({ kind: 'ask-value' });
    expect(actionForKey('n', item, false)).toEqual({ kind: 'answer', verdict: 'wrong' });
    expect(actionForKey('s', item, true)).toEqual({ kind: 'answer', verdict: 'unsure' });
    expect(actionForKey('z', undefined, true)).toEqual({ kind: 'undo' });
    expect(actionForKey('ArrowLeft', undefined, true)).toEqual({ kind: 'move', by: -1 });
    expect(actionForKey('y', undefined, true)).toBeUndefined();
  });

  it('opens valid packs only', () => {
    expect(openPack(pack()).pack?.id).toBe('test-pack');
    expect(openPack({ ...pack(), items: [] }).issues).toEqual(['items: must be a non-empty array']);
  });

  it('names the option an answer points at', () => {
    const p = pack();
    const item = p.items[0]!;
    expect(optionFor(p, item, { itemId: 'i0', verdict: 'correct' })?.label).toBe('Card 0');
    expect(optionFor(p, item, { itemId: 'i0', verdict: 'wrong', value: 'FAK-103' })?.label).toBe('Fake Hero');
    expect(optionFor(p, item, { itemId: 'i0', verdict: 'wrong', value: 'XYZ' })?.label).toBe('XYZ');
    expect(optionFor(p, item, { itemId: 'i0', verdict: 'wrong' })).toBeUndefined();
  });

  it('searches the vocabulary by label or value, accent-insensitively', () => {
    const v = pack().vocabulary!;
    expect(searchOptions(v, 'hero').map((o) => o.value)).toEqual(['FAK-101', 'FAK-104', 'FAK-103']); // prefix before word prefix
    expect(searchOptions(v, 'legend').map((o) => o.value)).toEqual(['FAK-102', 'FAK-104']);
    expect(searchOptions(v, 'fak-104').map((o) => o.value)).toEqual(['FAK-104']);
    expect(searchOptions(v, '  ')).toEqual([]);
  });
});
