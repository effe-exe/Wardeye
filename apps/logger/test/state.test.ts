import { describe, expect, it } from 'vitest';
import { validateTimelineDocument } from '@rifteye/schema';
import { parseCatalog, searchCards } from '../src/cards';
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
} from '../src/state';

const src = { kind: 'file' as const, ref: 'final.mp4' };
const at = (ms: number) => new Date(Date.UTC(2026, 8, 26, 12, 0, 0, ms));

describe('timeline state', () => {
  it('adds events that validate and exports them in time order', () => {
    let s = createState(src, at(0));
    s = addEvent(s, { t: 30, type: 'card_played', player: 'A', zone: 'base', cardId: 'fake-hero' }, at(1)).state;
    s = addEvent(s, { t: 5, type: 'game_start', player: null }, at(2)).state;
    s = addEvent(s, { t: 12.34567, type: 'score_changed', player: 'B', score: { A: 0, B: 1 } }, at(3)).state;
    const doc = toDocument(s);
    expect(validateTimelineDocument(doc)).toEqual([]);
    expect(doc.events.map((e) => e.type)).toEqual(['game_start', 'score_changed', 'card_played']);
    expect(doc.events[1]?.t).toBe(12.346);
    expect(doc.events.every((e) => e.evidence[0] === 'manual' && e.clock === 'media')).toBe(true);
  });

  it('records an unidentifiable card with confidence 0 and a hidden card without any identity', () => {
    let s = createState(src, at(0));
    s = addEvent(s, { t: 1, type: 'card_played', player: 'A', cardId: null }).state;
    s = addEvent(s, { t: 2, type: 'card_hidden', player: 'B', zone: 'facedown_1' }).state;
    const [unknown, hidden] = sortedEvents(s);
    expect(unknown?.card).toEqual({ cardId: null, confidence: 0 });
    expect(hidden?.card).toBeUndefined();
    expect(validateTimelineDocument(toDocument(s))).toEqual([]);
  });

  it('updates, removes and undoes the most recently created event', () => {
    let s = createState(src, at(0));
    const first = addEvent(s, { t: 50, type: 'turn_start', player: 'A' }, at(1));
    s = first.state;
    const second = addEvent(s, { t: 10, type: 'turn_start', player: 'B' }, at(2));
    s = second.state;
    s = updateEvent(s, first.id, { t: 51, type: 'turn_start', player: 'A', note: ' fixed ' }, at(3));
    expect(sortedEvents(s).find((e) => e.id === first.id)).toMatchObject({ t: 51, note: 'fixed' });
    s = undoLast(s, at(4)); // removes `second` (created last), even though it is earlier in video time
    expect(s.doc.events.map((e) => e.id)).toEqual([first.id]);
    s = removeEvent(s, first.id, at(5));
    expect(s.doc.events).toEqual([]);
    expect(undoLast(s)).toBe(s);
  });

  it('round-trips through fromDocument and rejects invalid files', () => {
    let s = createState(src, at(0));
    s = setMatchInfo(s, { title: 'Top 8', players: { A: 'Ann', B: 'Bo' }, legends: { A: 'fake-legend' } }, at(1));
    s = addEvent(s, { t: 3, type: 'card_played', player: 'A', cardId: 'fake-hero' }, at(2)).state;
    const back = fromDocument(JSON.parse(JSON.stringify(toDocument(s))));
    expect(back.issues).toEqual([]);
    expect(back.state?.doc.match.players).toEqual({ A: 'Ann', B: 'Bo' });
    expect(fromDocument({ schema: 'nope' }).state).toBeUndefined();
  });

  it('maps hotkeys and default zones sensibly', () => {
    expect(HOTKEYS.p).toBe('card_played');
    expect(HOTKEYS.h).toBe('card_hidden');
    expect(CARD_EVENTS.has('card_hidden')).toBe(false); // hidden cards are never named
    expect(defaultZone('spell_cast')).toBe('chain');
    expect(defaultZone('turn_start')).toBeUndefined();
  });

  it('formats video time', () => {
    expect(formatTime(0)).toBe('0:00.0');
    expect(formatTime(75.25)).toBe('1:15.3');
    expect(formatTime(3725)).toBe('1:02:05.0');
  });
});

describe('card catalogue', () => {
  const jsonl = [
    { printing_id: 'FAK-001', card_id: 'fake-hero', name: 'Fake Hero', type: 'Unit' },
    { printing_id: 'FAK-001a', card_id: 'fake-hero', name: 'Fake Hero', type: 'Unit' },
    { printing_id: 'FAK-002', card_id: 'heroic-strike', name: 'Heroic Strike', type: 'Spell' },
    { printing_id: 'FAK-003', card_id: 'zelda-s-hero', name: "Zélda's Hero", type: 'Unit' },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  it('groups printings under cards', () => {
    const cards = parseCatalog(jsonl);
    expect(cards.map((c) => c.cardId)).toEqual(['fake-hero', 'heroic-strike', 'zelda-s-hero']);
    expect(cards[0]?.printingIds).toEqual(['FAK-001', 'FAK-001a']);
    expect(parseCatalog(`[${jsonl.split('\n').join(',')}]`)).toHaveLength(3);
  });

  it('ranks prefix above word prefix above substring, ignoring case and accents', () => {
    const cards = parseCatalog(jsonl);
    expect(searchCards(cards, 'her').map((c) => c.cardId)).toEqual(['heroic-strike', 'fake-hero', 'zelda-s-hero']);
    expect(searchCards(cards, 'zelda').map((c) => c.cardId)).toEqual(['zelda-s-hero']);
    expect(searchCards(cards, 'FAK-002').map((c) => c.cardId)).toEqual(['heroic-strike']);
    expect(searchCards(cards, '   ')).toEqual([]);
  });
});
