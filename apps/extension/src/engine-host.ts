// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The engine host: a frame in (a tab's JPEG and the video's time), the overlay's state out. Each tab has its own
// board (its layout, its tracks) on the one set of models; the host adds what the live runner adds to its state
// (fps, latency) and what only the browser has: the frame's timings and the reads a second.

import type { Layout } from '@rifteye/engine';
import type { Attempt } from './mode';
import type { BoardEvent, BoardState, Parts } from './parts';
import { Session } from './session';
import { Rate, Timer } from './timer';

/** The frames a second the boards are built for, and so the overlay's pace (the runner's default is 5). */
export const FPS = 5;

export interface FrameRequest {
  tab: number;
  t: number;
  video: string;
  jpeg: Uint8Array;
  /** The decklists the viewer pasted for the tab (the plays panel's), sent with every frame. */
  lists?: readonly string[];
}

export interface HostConfig {
  parts: Parts;
  /** The timer the parts wrap the detector and the embedder with. */
  timer: Timer;
  /** How it runs: the runtime and each model's precision. */
  attempt: Attempt;
  fps?: number;
  /** Boards kept: the ones least recently used are let go. */
  maxSessions?: number;
  /** A layout for every video, instead of finding one. */
  layout?: Layout | null;
  /** A clock in ms. */
  now?: () => number;
}

const round = (v: number, d: number): number => Math.round(v * 10 ** d) / 10 ** d;

export class EngineHost {
  private readonly sessions = new Map<number, Session>();
  private readonly rate = new Rate();
  private readonly fps: number;
  private readonly now: () => number;

  constructor(private readonly cfg: HostConfig) {
    this.fps = cfg.fps ?? FPS;
    this.now = cfg.now ?? (() => performance.now());
  }

  private session(tab: number): Session {
    let s = this.sessions.get(tab);
    if (s) {
      this.sessions.delete(tab); // most recently used last
    } else {
      s = new Session({ parts: this.cfg.parts, fps: this.fps, layout: this.cfg.layout ?? null });
      for (const old of [...this.sessions.keys()].slice(0, Math.max(0, this.sessions.size + 1 - (this.cfg.maxSessions ?? 3)))) {
        this.sessions.delete(old);
      }
    }
    this.sessions.set(tab, s);
    return s;
  }

  /** One frame of a tab's video, read. */
  async frame(req: FrameRequest): Promise<{ state: BoardState; events: BoardEvent[] }> {
    const { parts, timer } = this.cfg;
    const t0 = this.now();
    timer.reset();
    const image = await timer.span('decode', () => parts.decode(req.jpeg));
    const session = this.session(req.tab);
    session.setLists(req.lists ?? []);
    const t1 = this.now();
    const out = await session.step(req.t, req.video, image);
    out.state.lists = session.listSummaries; // what the engine made of each list, for the plays panel
    const t2 = this.now();
    const reads = round(this.rate.tick(t2 / 1000), 1);
    const timing = timer.timing(t2 - t1);
    out.state.fps = { source: this.fps, processed: reads };
    out.state.latency_s = round((t2 - t0) / 1000, 2);
    out.state.engine = {
      runtime: this.cfg.attempt.runtime,
      detector: this.cfg.attempt.detector,
      embedder: this.cfg.attempt.embedder,
      every_ms: Math.round(1000 / this.fps),
      reads_per_s: reads,
      timing,
      layout: session.layoutName,
    };
    return out;
  }

  /** A tab went away: its board goes with it. */
  forget(tab: number): void {
    this.sessions.delete(tab);
  }

  get boards(): number {
    return this.sessions.size;
  }

  async dispose(): Promise<void> {
    this.sessions.clear();
    await this.cfg.parts.dispose();
  }
}
