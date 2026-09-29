import { describe, expect, it } from 'vitest';
import { DROPPED, FrameQueue, idleLongEnough } from '../src/router';

/** A job that finishes when its `release` is called. */
function job(tab: number, log: string[], name: string) {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  return {
    release,
    job: {
      tab,
      run: async () => {
        log.push(`start ${name}`);
        await gate;
        log.push(`end ${name}`);
        return name;
      },
    },
  };
}

describe('frames of several tabs on one engine', () => {
  it('run one at a time, in the order they came', async () => {
    const q = new FrameQueue<string>();
    const log: string[] = [];
    const a = job(1, log, 'a');
    const b = job(2, log, 'b');
    const pa = q.submit(a.job);
    const pb = q.submit(b.job);
    await Promise.resolve();
    expect(log).toEqual(['start a']); // b waits for a
    a.release();
    expect(await pa).toBe('a');
    b.release();
    expect(await pb).toBe('b');
    expect(log).toEqual(['start a', 'end a', 'start b', 'end b']);
  });

  it("drop a tab's waiting frame when its next one comes, and answer the newest", async () => {
    const q = new FrameQueue<string>();
    const log: string[] = [];
    const a = job(1, log, 'a');
    const old = job(2, log, 'old');
    const fresh = job(2, log, 'fresh');
    const pa = q.submit(a.job);
    const pOld = q.submit(old.job);
    const pFresh = q.submit(fresh.job);
    expect(await pOld).toBe(DROPPED);
    expect(q.pending).toBe(1);
    a.release();
    fresh.release();
    expect(await pa).toBe('a');
    expect(await pFresh).toBe('fresh');
    expect(log).not.toContain('start old');
  });

  it('never drops a frame that has begun', async () => {
    const q = new FrameQueue<string>();
    const log: string[] = [];
    const first = job(1, log, 'first');
    const next = job(1, log, 'next');
    const p1 = q.submit(first.job);
    await Promise.resolve();
    const p2 = q.submit(next.job); // first is running: next waits, it does not replace it
    first.release();
    next.release();
    expect(await p1).toBe('first');
    expect(await p2).toBe('next');
  });

  it('forget a tab: its waiting frame is dropped, and the others go on', async () => {
    const q = new FrameQueue<string>();
    const log: string[] = [];
    const a = job(1, log, 'a');
    const b = job(2, log, 'b');
    const c = job(3, log, 'c');
    const pa = q.submit(a.job);
    const pb = q.submit(b.job);
    const pc = q.submit(c.job);
    q.forget(2);
    expect(await pb).toBe(DROPPED);
    a.release();
    c.release();
    expect([await pa, await pc]).toEqual(['a', 'c']);
  });

  it('pass a failure to the frame that had it, and go on', async () => {
    const q = new FrameQueue<string>();
    const bad = q.submit({ tab: 1, run: async () => { throw new Error('gpu lost'); } });
    const good = q.submit({ tab: 2, run: async () => 'ok' });
    await expect(bad).rejects.toThrow('gpu lost');
    expect(await good).toBe('ok');
  });
});

describe('an idle engine document', () => {
  it('closes itself when no frame has come for long enough', () => {
    expect(idleLongEnough(299_999, 0, 300_000)).toBe(false);
    expect(idleLongEnough(300_000, 0, 300_000)).toBe(true);
  });
});
