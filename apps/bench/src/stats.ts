// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Timing for the bench: median and p90 of a batch's timed runs, and how many warm-up and timed runs a
// batch gets (fewer when a run is slow, so the whole bench stays within a few minutes on a laptop).

export const WARMUPS = 3;
export const REPEATS = 20;
export const MIN_REPEATS = 3;
/** A run longer than this is slow. */
export const SLOW_MS = 2000;
/** What one batch's timed runs may add up to: the 20 runs of a 2 s model. */
export const BUDGET_MS = REPEATS * SLOW_MS;

/** The median (the mean of the two middle values when the count is even); NaN when there are none. */
export function median(xs: readonly number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** The p-th percentile (0 < p <= 100) by nearest rank, so always a value that was measured; NaN when there are none. */
export function percentile(xs: readonly number[], p: number): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const rank = Math.min(s.length, Math.max(1, Math.ceil((p / 100) * s.length)));
  return s[rank - 1]!;
}

export const p90 = (xs: readonly number[]): number => percentile(xs, 90);

/** How many timed runs to do when the last warm-up run took `lastMs`: 20, fewer for slow runs, never under 3. */
export function repeatsFor(lastMs: number): number {
  if (!(lastMs > SLOW_MS)) return REPEATS;
  return Math.min(REPEATS, Math.max(MIN_REPEATS, Math.floor(BUDGET_MS / lastMs)));
}

export interface Timing {
  runs: number;
  medianMs: number;
  p90Ms: number;
  perItemMs: number;
  p90ItemMs: number;
}

/** The numbers reported for one batch size: per run and per item (a run's time over the batch). */
export function summarize(samples: readonly number[], batch: number): Timing {
  const medianMs = median(samples);
  const p90Ms = p90(samples);
  return { runs: samples.length, medianMs, p90Ms, perItemMs: medianMs / batch, p90ItemMs: p90Ms / batch };
}

export interface Measured {
  /** Every warm-up run's time; the first is the run that compiles shaders. */
  warmupMs: number[];
  samples: number[];
}

export interface MeasureOptions {
  /** Warm-up runs (default 3). */
  warmups?: number;
  /** Timed runs (default: by how slow the model is). */
  repeats?: number;
}

/**
 * Warms `run` up (3 runs; two are enough when the second is still slow, because then the compile is not
 * what makes it slow), then times it. `now` is a clock in ms; `onStep` reports the progress.
 */
export async function measureRuns(
  run: () => Promise<unknown>,
  now: () => number,
  onStep?: (phase: 'warm-up' | 'run', done: number, total: number) => void,
  options: MeasureOptions = {},
): Promise<Measured> {
  const timeOne = async (): Promise<number> => {
    const t0 = now();
    await run();
    return now() - t0;
  };
  const warmups = options.warmups ?? WARMUPS;
  const warmupMs: number[] = [];
  for (let i = 0; i < warmups; i++) {
    onStep?.('warm-up', i + 1, warmups);
    warmupMs.push(await timeOne());
    if (i >= 1 && warmupMs[i]! > SLOW_MS) break;
  }
  const total = options.repeats ?? repeatsFor(warmupMs[warmupMs.length - 1] ?? 0);
  const samples: number[] = [];
  for (let i = 0; i < total; i++) {
    onStep?.('run', i + 1, total);
    samples.push(await timeOne());
  }
  return { warmupMs, samples };
}
