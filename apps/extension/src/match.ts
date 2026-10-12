// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A new match in the same video: a VOD of a whole event, or a live stream, plays one game after another. Between two games
// the players take their cards off the table; a card stays on the board while it is only out of sight (a hand over it, the
// camera away), so a table that held a game and has held next to nothing for a while was cleared. Wardeye does not start the
// next match by itself: the plays panel asks, since a mistake would lose the plays. Pure, on the video's clock, so that a
// pause or a slow computer changes nothing.

import type { State } from './geometry';

/** Cards on the table (legends aside: they stay pinned) from which it held a game. */
export const SET_UP = 6;
/** Cards on the table up to which it counts as cleared. */
export const CLEARED = 1;
/** How long, on the video, a table that held a game stays cleared before the panel asks. */
export const CLEARED_FOR_S = 20;
/** A jump of the video's clock further than this (or back) is a seek: the time the table was seen cleared starts again. */
const SEEK_S = 30;

/** The cards on a board's table, as the watcher counts them: every track but the legends. */
export function onTable(state: State): number {
  return state.tracks.filter((t) => t.kind !== 'legend').length;
}

/** Watches one match's table, frame by frame, for the moment it was cleared. */
export class MatchWatch {
  /** The panel asks whether a new match started; until it is answered. */
  asking = false;
  private heldGame = false;
  private clearedSince: number | null = null;
  private lastT: number | null = null;

  /** A board read at `t` seconds on the video, with `cards` on its table. True while the panel should ask. */
  see(t: number, cards: number): boolean {
    if (this.lastT !== null && (t < this.lastT || t - this.lastT > SEEK_S)) this.clearedSince = null;
    this.lastT = t;
    if (cards >= SET_UP) {
      this.heldGame = true;
      this.clearedSince = null;
    } else if (cards <= CLEARED && this.heldGame) {
      this.clearedSince ??= t;
      if (t - this.clearedSince >= CLEARED_FOR_S) {
        this.asking = true;
        this.heldGame = false; // asked once for this clearing
      }
    } else {
      this.clearedSince = null;
    }
    return this.asking;
  }

  /** The viewer answered (a new match, or the same one): the watcher waits for a game on the table again. */
  answered(): void {
    this.asking = false;
    this.heldGame = false;
    this.clearedSince = null;
  }
}
