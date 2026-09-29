import { describe, expect, it } from 'vitest';
import { BUDGET_MS, MIN_REPEATS, REPEATS, SLOW_MS, WARMUPS, measureRuns, median, p90, percentile, repeatsFor, summarize } from '../src/stats';

describe('median and p90', () => {
  it('takes the middle value, or the mean of the two middle ones', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([7])).toBe(7);
    expect(median([])).toBeNaN();
  });

  it('does not reorder what it is given', () => {
    const xs = [3, 1, 2];
    median(xs);
    p90(xs);
    expect(xs).toEqual([3, 1, 2]);
  });

  it('gives the 90th percentile by nearest rank: a value that was measured', () => {
    const twenty = Array.from({ length: 20 }, (_, i) => i + 1); // 1..20
    expect(p90(twenty)).toBe(18); // the 18th of 20
    expect(p90([5, 1, 3])).toBe(5); // 3 runs: the slowest
    expect(p90([9])).toBe(9);
    expect(p90([])).toBeNaN();
    expect(percentile(twenty, 100)).toBe(20);
    expect(percentile(twenty, 50)).toBe(10);
  });
});

describe('how many runs', () => {
  it('is 3 warm-ups and 20 timed runs for a run of 2 s or less', () => {
    expect([WARMUPS, REPEATS, MIN_REPEATS, SLOW_MS]).toEqual([3, 20, 3, 2000]);
    expect(repeatsFor(5)).toBe(20);
    expect(repeatsFor(SLOW_MS)).toBe(20);
  });

  it('is fewer when a run takes over 2 s, and never fewer than 3', () => {
    expect(repeatsFor(2500)).toBe(16);
    expect(repeatsFor(5000)).toBe(8);
    expect(repeatsFor(10_000)).toBe(4);
    expect(repeatsFor(BUDGET_MS / 3)).toBe(3);
    expect(repeatsFor(60_000)).toBe(3);
  });
});

describe('a batch summary', () => {
  it('reports per run and per item', () => {
    const t = summarize([10, 20, 30, 40], 8);
    expect(t).toEqual({ runs: 4, medianMs: 25, p90Ms: 40, perItemMs: 25 / 8, p90ItemMs: 5 });
  });
});

/** A clock that only moves when a run takes its scripted time. */
function scripted(times: number[]) {
  let clock = 0;
  let calls = 0;
  const run = async () => {
    clock += times[Math.min(calls, times.length - 1)]!;
    calls++;
  };
  return { run, now: () => clock, calls: () => calls };
}

describe('measuring a batch', () => {
  it('warms up 3 times and times 20 runs of a fast model', async () => {
    const s = scripted([50, 10]);
    const m = await measureRuns(s.run, s.now);
    expect(m.warmupMs).toEqual([50, 10, 10]); // the first is the compile
    expect(m.samples).toHaveLength(20);
    expect(s.calls()).toBe(23);
    expect(median(m.samples)).toBe(10);
  });

  it('does not take a slow first run (a shader compile) for a slow model', async () => {
    const s = scripted([9000, 30]);
    const m = await measureRuns(s.run, s.now);
    expect(m.warmupMs).toHaveLength(3);
    expect(m.samples).toHaveLength(20);
  });

  it('times fewer runs of a model that is slow once it is warm, and warms up twice', async () => {
    const s = scripted([5000]);
    const m = await measureRuns(s.run, s.now);
    expect(m.warmupMs).toEqual([5000, 5000]);
    expect(m.samples).toHaveLength(8);
    expect(s.calls()).toBe(10);
  });

  it('times at least 3 runs however slow', async () => {
    const s = scripted([30_000]);
    const m = await measureRuns(s.run, s.now);
    expect(m.samples).toHaveLength(3);
  });

  it('can be told how many runs to do', async () => {
    const s = scripted([50, 10]);
    const m = await measureRuns(s.run, s.now, undefined, { warmups: 1, repeats: 1 });
    expect(m.warmupMs).toEqual([50]);
    expect(m.samples).toEqual([10]);
    expect(s.calls()).toBe(2);
    const t = scripted([7]);
    expect((await measureRuns(t.run, t.now, undefined, { repeats: 4 })).samples).toHaveLength(4);
  });

  it('reports its progress', async () => {
    const steps: string[] = [];
    const s = scripted([1]);
    await measureRuns(s.run, s.now, (phase, done, total) => steps.push(`${phase} ${done}/${total}`));
    expect(steps.slice(0, 4)).toEqual(['warm-up 1/3', 'warm-up 2/3', 'warm-up 3/3', 'run 1/20']);
    expect(steps.at(-1)).toBe('run 20/20');
  });

  it('lets a failing run through', async () => {
    await expect(measureRuns(async () => Promise.reject(new Error('device lost')), () => 0)).rejects.toThrow('device lost');
  });
});
