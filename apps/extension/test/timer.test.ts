import { describe, expect, it } from 'vitest';
import { Rate, Timer } from '../src/timer';

describe('the timer', () => {
  it('adds up the spans of a frame, and the track is what the step took beyond them', async () => {
    let clock = 0;
    const timer = new Timer(() => clock);
    await timer.span('decode', async () => void (clock += 6));
    const detect = timer.wrap('detect', async (n: number) => {
      clock += n;
      return n;
    });
    expect(await detect(20)).toBe(20);
    await detect(24); // twice in a frame: added
    await timer.span('embed', async () => void (clock += 22));
    expect(timer.timing(100)).toEqual({ decode: 6, detect: 44, embed: 22, track: 34, total: 106 });
    timer.reset();
    expect(timer.timing(10)).toEqual({ decode: 0, detect: 0, embed: 0, track: 10, total: 10 });
  });

  it('counts a span that throws, and lets the error through', async () => {
    let clock = 0;
    const timer = new Timer(() => clock);
    await expect(timer.span('detect', async () => { clock += 5; throw new Error('boom'); })).rejects.toThrow('boom');
    expect(timer.get('detect')).toBe(5);
  });
});

describe('the reads a second', () => {
  it('follows the last few frames, and starts afresh after a pause', () => {
    const rate = new Rate();
    expect(rate.tick(0)).toBe(0); // the first frame has no gap to measure
    expect(rate.tick(0.2)).toBeCloseTo(5, 5);
    expect(rate.tick(0.4)).toBeCloseTo(5, 5);
    expect(rate.tick(0.5)).toBeCloseTo(0.8 * 5 + 0.2 * 10, 5);
    expect(rate.tick(60)).toBeCloseTo(0.8 * 5 + 0.2 * 10, 5); // a pause: the frame after it is not a sample
    expect(rate.tick(60.1)).toBeCloseTo(0.8 * 6 + 0.2 * 10, 5);
    rate.reset();
    expect(rate.tick(61)).toBe(0);
  });
});
