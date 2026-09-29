// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// One engine, several Twitch tabs. Frames wait their turn and run one at a time; a tab's frame that has not begun
// when its next one arrives is dropped (only the newest is worth reading), so the tab that sends frames is the one
// that is answered, and a tab never queues behind itself.

export interface Job<T> {
  /** Which tab sent it. */
  tab: number;
  run: () => Promise<T>;
}

/** What a dropped job resolves to. */
export const DROPPED = Symbol('dropped');

export class FrameQueue<T> {
  private waiting: { tab: number; run: () => Promise<T>; done: (v: T | typeof DROPPED) => void; fail: (e: unknown) => void }[] = [];
  private busy = false;

  /** Resolves with the job's result, or DROPPED when a newer job of the same tab replaced it, or when the tab was forgotten. */
  submit(job: Job<T>): Promise<T | typeof DROPPED> {
    return new Promise((done, fail) => {
      const old = this.waiting.findIndex((w) => w.tab === job.tab);
      if (old >= 0) this.waiting.splice(old, 1)[0]!.done(DROPPED);
      this.waiting.push({ ...job, done, fail });
      void this.pump();
    });
  }

  /** A tab went away: its waiting job is dropped. */
  forget(tab: number): void {
    const i = this.waiting.findIndex((w) => w.tab === tab);
    if (i >= 0) this.waiting.splice(i, 1)[0]!.done(DROPPED);
  }

  get pending(): number {
    return this.waiting.length;
  }

  private async pump(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      for (let next = this.waiting.shift(); next; next = this.waiting.shift()) {
        try {
          next.done(await next.run());
        } catch (e) {
          next.fail(e);
        }
      }
    } finally {
      this.busy = false;
    }
  }
}

/** Whether the engine's document has been idle long enough to close itself (ms). */
export function idleLongEnough(now: number, lastFrame: number, idleMs: number): boolean {
  return now - lastFrame >= idleMs;
}
