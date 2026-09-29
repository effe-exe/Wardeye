// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Where a frame's time goes, and how many frames a second the engine reads.

/** What one frame cost, in ms: the JPEG's decode, the detector, the embedder, the tracker (everything else the
 * recogniser does) and the whole. */
export interface Timing {
  decode: number;
  detect: number;
  embed: number;
  track: number;
  total: number;
}

/** Adds up the time of named spans for one frame. Spans do not nest: `track` is what the whole took beyond them. */
export class Timer {
  private spans = new Map<string, number>();

  constructor(private readonly now: () => number) {}

  reset(): void {
    this.spans.clear();
  }

  add(name: string, ms: number): void {
    this.spans.set(name, (this.spans.get(name) ?? 0) + ms);
  }

  get(name: string): number {
    return this.spans.get(name) ?? 0;
  }

  /** Runs `fn`, adding the time it takes (also when it throws) to `name`. */
  async span<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const t0 = this.now();
    try {
      return await fn();
    } finally {
      this.add(name, this.now() - t0);
    }
  }

  /** `fn` that adds its own time to `name` each time it is called. */
  wrap<A extends unknown[], R>(name: string, fn: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
    return (...args) => this.span(name, () => fn(...args));
  }

  /** The frame's timing: `decode` and `step` (the recogniser's whole call) as measured, `detect` and `embed` from the
   * spans, `track` the rest of the step. */
  timing(step: number): Timing {
    const decode = this.get('decode');
    const detect = this.get('detect');
    const embed = this.get('embed');
    const r = (v: number) => Math.round(v * 10) / 10;
    return { decode: r(decode), detect: r(detect), embed: r(embed), track: r(Math.max(0, step - detect - embed)), total: r(decode + step) };
  }
}

/** Frames read a second: an average that follows the last few (as the live runner's `fps.processed` does). A pause
 * longer than `gap` s starts it afresh. */
export class Rate {
  private last: number | null = null;
  private value = 0;

  constructor(private readonly gap = 2) {}

  /** A frame was read at `now` (s); returns the rate. */
  tick(now: number): number {
    const dt = this.last === null ? Infinity : now - this.last;
    this.last = now;
    if (dt > this.gap) return this.value;
    const sample = 1 / Math.max(1e-3, dt);
    this.value = this.value ? 0.8 * this.value + 0.2 * sample : sample;
    return this.value;
  }

  /** Forgets the average: nothing was read for a while. */
  reset(): void {
    this.last = null;
    this.value = 0;
  }
}
