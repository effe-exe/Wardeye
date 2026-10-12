import { describe, expect, it } from 'vitest';
import type { State, Track } from '../src/geometry';
import { CLEARED_FOR_S, MatchWatch, SET_UP, onTable } from '../src/match';

/** Frames from `from` to `to` seconds, one a second, with `cards` on the table; what the watcher says after the last. */
function run(w: MatchWatch, from: number, to: number, cards: number): boolean {
  let asking = false;
  for (let t = from; t <= to; t++) asking = w.see(t, cards);
  return asking;
}

describe('a new match in the same video', () => {
  it('is asked about once a table that held a game has stayed cleared for a while', () => {
    const w = new MatchWatch();
    expect(run(w, 0, 300, 14)).toBe(false); // a game
    expect(run(w, 301, 301 + CLEARED_FOR_S - 2, 0)).toBe(false); // the cards picked up: not yet
    expect(run(w, 301 + CLEARED_FOR_S - 1, 301 + CLEARED_FOR_S + 5, 1)).toBe(true);
    // the next game is set up before the viewer answers: the question stays
    expect(run(w, 330, 340, 8)).toBe(true);
  });

  it('is not asked about before a game, nor while the table only thins out', () => {
    const w = new MatchWatch();
    expect(run(w, 0, 120, 0)).toBe(false); // the casters' desk, then an empty mat: no game yet
    expect(run(w, 121, 200, SET_UP)).toBe(false);
    expect(run(w, 201, 400, 3)).toBe(false); // a few cards left: still the game
  });

  it('asks again only after another game on the table, whatever the answer', () => {
    const w = new MatchWatch();
    run(w, 0, 100, 10);
    expect(run(w, 101, 140, 0)).toBe(true);
    w.answered(); // the same match (a long pause between turns, say)
    expect(run(w, 141, 300, 0)).toBe(false);
    run(w, 301, 400, 10);
    expect(run(w, 401, 440, 0)).toBe(true);
  });

  it('starts the cleared time again after a seek', () => {
    const w = new MatchWatch();
    run(w, 0, 100, 10);
    run(w, 101, 101 + CLEARED_FOR_S - 5, 0);
    expect(run(w, 50, 50 + CLEARED_FOR_S - 5, 0)).toBe(false); // back in the video: the empty table is timed afresh
    expect(run(w, 50 + CLEARED_FOR_S - 4, 50 + CLEARED_FOR_S + 1, 0)).toBe(true);
  });
});

describe('the cards on the table', () => {
  it('are every track but the legends', () => {
    const track = (kind: string): Track => ({ id: kind, quad: [], side: 'left', state: 'named', printing_id: null, name: '', confidence: 1, guesses: [], kind, hidden: false });
    const state = { tracks: ['legend', 'legend', 'card', 'rune', 'battlefield'].map(track) } as unknown as State;
    expect(onTable(state)).toBe(3);
  });
});
