// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// One tab's board, run as ml/rifteye_ml/live/__main__.py runs its loop: the layout is found from the first frames of
// the table camera (a look a second, over the last five; the presets take over when the footage will not give one),
// a jump in the video starts a new board on the same table, and another video (the page the content script names)
// finds its own layout and starts over.

import type { Layout, RgbImage } from '@rifteye/engine';
import { matchPreset } from './presets';
import { startingState, type BoardEvent, type BoardState, type Parts, type Board } from './parts';

/** A step in the video's time within this range continues the board; beyond it the board starts over (the runner's). */
const JUMP_BACK_S = -1;
const JUMP_FORWARD_S = 15;

export interface SessionConfig {
  parts: Parts;
  /** The frames a second the boards are built for. */
  fps: number;
  /** A layout to use for every video instead of finding one. */
  layout?: Layout | null;
  /** Seconds of video between the looks the layout is found from. */
  lookEvery?: number;
  /** How many looks a layout is found from. */
  looks?: number;
  /** Failed attempts (with all the looks) before the presets are tried. */
  presetAfter?: number;
}

export interface StepResult {
  state: BoardState;
  events: BoardEvent[];
}

export class Session {
  private video: string | null = null;
  private layout: Layout | null;
  private board: Board | null = null;
  private lastT: number | null = null;
  private seen: RgbImage[] = [];
  private lastLook: number | null = null;
  private tries = 0;
  private complained = false;
  private readonly lookEvery: number;
  private readonly lookCount: number;
  private readonly presetAfter: number;

  constructor(private readonly cfg: SessionConfig) {
    this.layout = cfg.layout ?? null;
    this.lookEvery = cfg.lookEvery ?? 1;
    this.lookCount = cfg.looks ?? 5;
    this.presetAfter = cfg.presetAfter ?? 12;
  }

  /** The layout in use: its name, or "auto" while none is found. */
  get layoutName(): string {
    return this.layout?.name ?? 'auto';
  }

  private reset(video: string): void {
    this.video = video;
    this.board = null;
    this.lastT = null;
    this.seen = [];
    this.lastLook = null;
    this.tries = 0;
    if (!this.cfg.layout) this.layout = null; // another video: its own table
  }

  /** A frame of `video` at `t` s of its time. */
  async step(t: number, video: string, image: RgbImage): Promise<StepResult> {
    if (video !== this.video) this.reset(video);
    if (!this.layout) return this.find(t, image);
    const { parts } = this.cfg;
    const jump = this.lastT !== null && !(t - this.lastT >= JUMP_BACK_S && t - this.lastT <= JUMP_FORWARD_S);
    if (!this.board || jump) this.board = parts.board(this.layout); // a jump: a new board on the same table
    this.lastT = t;
    return this.board.step(t, image);
  }

  /** The layout from the first frames of the table camera. The frames it is found from are not read as a board. */
  private async find(t: number, image: RgbImage): Promise<StepResult> {
    const { parts, fps } = this.cfg;
    const gone = this.lastLook !== null && (t < this.lastLook || t - this.lastLook > JUMP_FORWARD_S);
    if (gone) {
      this.seen = []; // a seek: the looks were another scene
      this.lastLook = null;
    }
    if (this.lastLook === null || t - this.lastLook >= this.lookEvery) {
      this.lastLook = t;
      this.seen = [...this.seen, image].slice(-this.lookCount);
      if (this.seen.length === this.lookCount) {
        let found = await parts.findLayout(this.seen).catch((e: unknown) => {
          if (!this.complained) console.warn(`RiftEye: looking for the table failed (${e instanceof Error ? e.message : String(e)}); the presets are tried in a while`);
          this.complained = true;
          return null; // a look that failed is a look that found nothing
        });
        if (!found && ++this.tries >= this.presetAfter) found = matchPreset(image, parts.presets());
        if (found) {
          this.layout = found;
          this.seen = [];
          this.lastT = null;
        }
      }
    }
    return { state: startingState(t, image, 'finding the table and the size of a card', fps), events: [] };
  }
}
