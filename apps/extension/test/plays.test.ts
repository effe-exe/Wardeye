// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors

import { describe, expect, it } from 'vitest';
import type { State, Track } from '../src/geometry';
import type { BoardEvent } from '../src/parts';
import { MAX_PLAYS, PlayLog, clock, isSnapshot, sidesOf } from '../src/plays';

const track = (id: string, side: string, fields: Partial<Track> = {}): Track => ({
  id,
  quad: [[0, 0], [1, 0], [1, 1], [0, 1]],
  side,
  state: 'named',
  printing_id: null,
  name: '',
  confidence: 0.9,
  guesses: [],
  kind: 'card',
  hidden: false,
  ...fields,
});

const play = (t: number, text: string, side = 'left', kind = 'played', trackId = `t${t}`): BoardEvent => ({
  t,
  kind,
  text,
  printing_id: 'OGN-001',
  track: trackId,
  side,
});

describe('the board, per player', () => {
  const state = {
    t: 10,
    status: 'live',
    message: '',
    frame: { width: 1920, height: 1080 },
    players: [
      { side: 'left', label: 'Player 1', legend: { printing_id: 'SFD-195a', name: 'Blade Dancer' } },
      { side: 'right', label: 'Player 2', legend: null },
    ],
    tracks: [
      track('t1', 'left', { printing_id: 'OGN-010', name: 'Zhonya', under: [{ id: 't9', name: 'Charm', printing_id: 'OGN-020' }] }),
      track('t2', 'left', { printing_id: 'OGN-010', name: 'Zhonya', hidden: true }), // a hand over it: still on the table
      track('t9', 'left', { printing_id: 'OGN-020', name: 'Charm' }), // under t1: listed with it, not on its own
      track('t3', 'left', { kind: 'rune', printing_id: 'OGN-042', name: 'Calm Rune' }),
      track('t4', 'left', { kind: 'legend', printing_id: 'SFD-195a', name: 'Blade Dancer' }),
      track('t5', 'right', { state: 'unsure' }),
      track('t6', 'right', { state: 'facedown' }),
      track('t7', 'right', { state: 'new' }),
    ],
  } as unknown as State;

  it("lists each player's legend, their named cards grouped by printing with what lies under them, and counts the rest", () => {
    const [left, right] = sidesOf(state);
    expect(left).toEqual({
      side: 'left',
      label: 'Player 1',
      legend: { printing_id: 'SFD-195a', name: 'Blade Dancer' },
      cards: [{ printing_id: 'OGN-010', name: 'Zhonya', count: 2, under: ['Charm'] }],
      runes: 1,
      unsure: 0,
      facedown: 0,
    });
    expect(right).toEqual({ side: 'right', label: 'Player 2', legend: null, cards: [], runes: 0, unsure: 1, facedown: 1 });
  });

  it('says nothing without a board, or of players the state does not name', () => {
    expect(sidesOf(null)).toEqual([]);
    expect(sidesOf({ ...state, players: 'nobody' } as unknown as State)).toEqual([]);
    expect(sidesOf({ ...state, players: [{ side: 3 }, null, { side: 'top' }] } as unknown as State).map((s) => s.label)).toEqual(['top']);
  });
});

describe('the plays of a video', () => {
  it('keeps them in the order they happened, each once, and returns the new ones', () => {
    const log = new PlayLog();
    expect(log.add('/videos/1', [play(5, 'Sivir played'), play(9, 'Charm played', 'right')])).toHaveLength(2);
    // the same play again, a frame later or after its track was read afresh: kept once
    expect(log.add('/videos/1', [play(6, 'Sivir played', 'left', 'played', 't12')])).toEqual([]);
    // after a seek back, an earlier play goes where it happened
    expect(log.add('/videos/1', [play(2, 'Mel played')]).map((e) => e.text)).toEqual(['Mel played']);
    expect(log.plays.map((e) => e.t)).toEqual([2, 5, 9]);
    // the same card played again later is another play
    expect(log.add('/videos/1', [play(30, 'Sivir played')])).toHaveLength(1);
  });

  it('starts over for another video, and keeps the newest plays of a long one', () => {
    const log = new PlayLog();
    log.add('/videos/1', [play(5, 'Sivir played')]);
    expect(log.add('/videos/2', []).length).toBe(0);
    expect(log.plays).toEqual([]);
    for (let k = 0; k < MAX_PLAYS + 20; k++) log.add('/videos/2', [play(10 * k, `card ${k} played`)]);
    expect(log.plays).toHaveLength(MAX_PLAYS);
    expect(log.plays[0]!.text).toBe('card 20 played');
  });
});

describe('the video clock', () => {
  it('reads as the replay does: minutes, or hours an hour in', () => {
    expect([clock(0), clock(65.9), clock(3599), clock(3600), clock(14 * 3600 + 59 * 60 + 55)]).toEqual(['0:00', '1:05', '59:59', '1:00:00', '14:59:55']);
    expect([clock(-3), clock(Number.NaN)]).toEqual(['0:00', '0:00']);
  });
});

describe('a snapshot over the wire', () => {
  it('is checked before the panel draws it', () => {
    expect(isSnapshot({ kind: 'board', video: '/videos/1', sides: [], plays: [] })).toBe(true);
    expect(isSnapshot({ kind: 'board', video: '/videos/1', sides: [] })).toBe(false);
    expect(isSnapshot({ kind: 'state' })).toBe(false);
    expect(isSnapshot(null)).toBe(false);
  });
});
