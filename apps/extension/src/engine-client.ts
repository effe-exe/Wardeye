// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// An engine worker as the engine document runs it: requests out with an id, answers matched back, and a worker
// that dies or does not answer reported, not waited for.

import type { StandalonePackage } from './assets';
import type { EngineWorker, FrameOut, FrameReq } from './controller';
import type { Attempt } from './mode';
import type { FromEngine, ToEngine } from './protocol';

/** A frame that takes longer than this has hung the worker (the models load before the first frame is sent). */
export const FRAME_MS = 30_000;
/** Loading the models and the gallery, and the first runs, take a few seconds on a GPU and about a minute on WASM. */
export const INIT_MS = 180_000;

export class WorkerEngine implements EngineWorker {
  private readonly worker: Worker;
  private next = 0;
  private readonly pending = new Map<number, { resolve: (o: FrameOut) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private starting: { resolve: () => void; reject: (e: Error) => void; progress: (m: string) => void; timer: ReturnType<typeof setTimeout> } | null = null;
  private crash: ((reason: string) => void) | null = null;
  private dead = false;
  /** Told what the finder and the embedder gave, when the package asks for a trace. */
  onTrace: ((m: Extract<FromEngine, { kind: 'trace' }>) => void) | null = null;

  /** `script` is the worker's file next to the document (engine-webgpu.js or engine-wasm.js); `frameMs` is how long a
   * frame may take before the worker is taken to have hung, `initMs` how long it may take to load. */
  constructor(
    script: string,
    private readonly frameMs = FRAME_MS,
    private readonly initMs = INIT_MS,
  ) {
    this.worker = new Worker(new URL(script, location.href), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromEngine>) => this.on(e.data);
    this.worker.onerror = (e) => this.die(e.message || 'the engine worker stopped');
    this.worker.onmessageerror = () => this.die('the engine worker sent something unreadable');
  }

  private on(m: FromEngine): void {
    if (m.kind === 'trace') this.onTrace?.(m);
    else if (m.kind === 'progress') this.starting?.progress(m.message);
    else if (m.kind === 'ready') this.settle(null);
    else if (m.kind === 'failed') this.settle(new Error(m.error));
    else {
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      clearTimeout(p.timer);
      if (m.kind === 'state') p.resolve({ state: m.state, events: m.events });
      else p.reject(new Error(m.error));
    }
  }

  private settle(error: Error | null): void {
    const s = this.starting;
    this.starting = null;
    if (s) clearTimeout(s.timer);
    if (error) s?.reject(error);
    else s?.resolve();
  }

  /** The worker is gone: everything waiting for it fails, and the document is told once (unless it is loading, which
   * fails its own way). */
  private die(reason: string): void {
    if (this.dead) return;
    this.dead = true;
    this.worker.terminate();
    const error = new Error(reason);
    if (this.starting) this.settle(error);
    else this.crash?.(reason);
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(error);
    }
    this.pending.clear();
  }

  init(pkg: StandalonePackage, attempt: Attempt, progress: (message: string) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.die(`the engine did not start in ${this.initMs / 1000} s`), this.initMs);
      this.starting = { resolve, reject, progress, timer };
      this.post({ kind: 'init', pkg, attempt });
    });
  }

  frame(req: FrameReq): Promise<FrameOut> {
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => this.die(`no answer to a frame in ${this.frameMs / 1000} s`), this.frameMs);
      this.pending.set(id, { resolve, reject, timer });
      this.post({ kind: 'frame', id, ...req });
    });
  }

  forget(tab: number): void {
    this.post({ kind: 'forget', tab });
  }

  private post(m: ToEngine): void {
    if (!this.dead) this.worker.postMessage(m);
  }

  onCrash(cb: (reason: string) => void): void {
    this.crash = cb;
  }

  terminate(): void {
    this.dead = true;
    this.worker.terminate();
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new Error('the engine worker was stopped'));
    }
    this.pending.clear();
  }
}
