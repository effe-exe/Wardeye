// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The engine document's brain: what can run here (the package's models, the browser's WebGPU), starting the engine
// worker (the best way first, the next when one will not start), answering the extension worker's requests, and
// starting it again when it crashes. The document is a thin shell around this; the worker and the clock come in
// through `ControllerEnv`, so it is tested without a browser.

import type { StandalonePackage } from './assets';
import type { State } from './geometry';
import { plan, RETRY_AFTER_MS, type Attempt, type Capabilities } from './mode';

/** An attempt in a few words: how it runs, and each model's precision. */
export const describeAttempt = (a: Attempt): string => `${a.runtime}: detector ${a.detector}, embedder ${a.embedder}`;
import { FPS } from './engine-host';
import { startingState, type BoardEvent } from './parts';
import type { EngineReply } from './protocol';
import { DROPPED, FrameQueue } from './router';

export interface FrameReq {
  tab: number;
  t: number;
  video: string;
  jpeg: string;
}

export interface FrameOut {
  state: State;
  events: BoardEvent[];
}

/** An engine worker, as the document runs it. */
export interface EngineWorker {
  /** Loads the models and the gallery; rejects with why it could not. */
  init(pkg: StandalonePackage, attempt: Attempt, progress: (message: string) => void): Promise<void>;
  frame(req: FrameReq): Promise<FrameOut>;
  forget(tab: number): void;
  /** Called once when the worker dies on its own. */
  onCrash(cb: (reason: string) => void): void;
  terminate(): void;
}

export interface ControllerEnv {
  now(): number;
  readPackage(): Promise<StandalonePackage | null>;
  probe(): Promise<Capabilities>;
  spawn(attempt: Attempt): EngineWorker;
}

type Decision = { pkg: StandalonePackage; attempts: Attempt[] } | { reason: string };

type Status =
  | { kind: 'idle' }
  | { kind: 'loading'; message: string }
  | { kind: 'ready'; worker: EngineWorker; attempt: Attempt }
  | { kind: 'failed'; reason: string; until: number };

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** Restarts allowed in a row within `CRASH_WINDOW_MS` before the engine is written off for a while. */
const MAX_RESTARTS = 3;
const CRASH_WINDOW_MS = 2 * 60_000;
/** How long the overlay waits before sending another frame while the engine loads (ms). */
export const LOADING_RETRY_MS = 1000;
/** Frames in a row that failed before the engine worker is started afresh (a lost GPU device does not come back). */
const MAX_BAD_FRAMES = 3;

export class Controller {
  private status: Status = { kind: 'idle' };
  private readonly queue = new FrameQueue<FrameOut>();
  private crashes: number[] = [];
  private bad = 0;
  private decision: Promise<Decision> | null = null;
  /** When a frame last came (ms, the env's clock): the document closes itself when none has for a while. */
  lastFrame: number;
  /** The way the engine runs, once it does. */
  running: Attempt | null = null;

  constructor(
    private readonly env: ControllerEnv,
    private readonly retryMs = RETRY_AFTER_MS,
  ) {
    this.lastFrame = env.now();
  }

  private fail(reason: string): void {
    this.status = { kind: 'failed', reason, until: this.env.now() + this.retryMs };
    this.running = null;
    this.decision = null; // asked again when the time is up: the browser may have changed
  }

  /** What can run here: the package's models and what the browser has. Decided once, and quickly. */
  private decide(): Promise<Decision> {
    this.decision ??= (async (): Promise<Decision> => {
      const pkg = await this.env.readPackage();
      if (!pkg) return { reason: 'this build has no models' };
      const { attempts, reason } = plan(pkg, await this.env.probe());
      return attempts.length > 0 ? { pkg, attempts } : { reason: reason || 'nothing to run' };
    })().catch((e): Decision => ({ reason: message(e) }));
    return this.decision;
  }

  /** Starts the engine unless it is going or written off; never waits for it. */
  private start(): void {
    const s = this.status;
    if (s.kind === 'loading' || s.kind === 'ready') return;
    if (s.kind === 'failed' && this.env.now() < s.until) return;
    this.status = { kind: 'loading', message: 'starting the engine' };
    void this.load();
  }

  private async load(): Promise<void> {
    try {
      const decided = await this.decide();
      if ('reason' in decided) return this.fail(decided.reason);
      const { pkg, attempts } = decided;
      const errors: string[] = [];
      for (const attempt of attempts) {
        const worker = this.env.spawn(attempt);
        this.status = { kind: 'loading', message: `loading the models (${describeAttempt(attempt)})` };
        try {
          await worker.init(pkg, attempt, (m) => {
            if (this.status.kind === 'loading') this.status = { kind: 'loading', message: m };
          });
        } catch (e) {
          worker.terminate();
          errors.push(`${describeAttempt(attempt)}: ${message(e)}`);
          continue;
        }
        worker.onCrash((why) => this.crashed(worker, why));
        this.status = { kind: 'ready', worker, attempt };
        this.running = attempt;
        this.bad = 0;
        return;
      }
      this.fail(errors.join('; '));
    } catch (e) {
      this.fail(message(e));
    }
  }

  /** The engine worker died (or kept failing): its boards are lost; it starts afresh with the next frame, unless it
   * keeps happening. */
  private crashed(worker: EngineWorker, why: string): void {
    if (this.status.kind !== 'ready' || this.status.worker !== worker) return;
    worker.terminate();
    const now = this.env.now();
    this.crashes = [...this.crashes.filter((t) => now - t < CRASH_WINDOW_MS), now];
    this.running = null;
    if (this.crashes.length >= MAX_RESTARTS) this.fail(`the engine keeps stopping (${why})`);
    else this.status = { kind: 'idle' };
  }

  /** Whether the engine can run here: it is decided at once (what the package and the browser allow), not when the
   * models are loaded. */
  async hello(): Promise<EngineReply> {
    this.start();
    const s = this.status;
    if (s.kind === 'failed') return { kind: 'unavailable', reason: s.reason };
    const decided = await this.decide(); // the decision is waited for, not the models
    return 'reason' in decided ? { kind: 'unavailable', reason: decided.reason } : { kind: 'hello' };
  }

  async frame(req: FrameReq): Promise<EngineReply> {
    this.lastFrame = this.env.now();
    this.start();
    const s = this.status;
    if (s.kind === 'failed') return { kind: 'unavailable', reason: s.reason };
    const blank = { width: 0, height: 0 };
    if (s.kind !== 'ready') {
      const state = startingState(req.t, blank, s.kind === 'loading' ? s.message : 'starting the engine', FPS);
      return { kind: 'state', state: { ...state, retry_ms: LOADING_RETRY_MS } }; // no need to send frames as fast while it loads
    }
    const { worker } = s;
    try {
      const out = await this.queue.submit({ tab: req.tab, run: () => worker.frame(req) });
      this.bad = 0;
      return { kind: 'state', state: out === DROPPED ? null : out.state };
    } catch (e) {
      if (++this.bad >= MAX_BAD_FRAMES) this.crashed(worker, message(e));
      return { kind: 'state', state: { t: req.t, status: 'error', message: message(e), frame: blank, tracks: [] } };
    }
  }

  forget(tab: number): void {
    this.queue.forget(tab);
    if (this.status.kind === 'ready') this.status.worker.forget(tab);
  }

  /** Lets the engine go (the document is closing). */
  dispose(): void {
    if (this.status.kind === 'ready') this.status.worker.terminate();
    this.status = { kind: 'idle' };
    this.running = null;
  }

  get state(): Status['kind'] {
    return this.status.kind;
  }
}
