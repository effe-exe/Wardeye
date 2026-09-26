import { describe, expect, it } from 'vitest';
import {
  checkBundle,
  corners,
  currentEvent,
  eventGuesses,
  formatTime,
  hoverState,
  sampleAt,
  type DemoBundle,
  type Track,
} from '../src/demo';

const bundle = (over: Partial<DemoBundle> = {}): DemoBundle => ({
  schema: 'rifteye.demo',
  version: 1,
  title: 'test',
  video: 'clip.webm',
  frame: [1920, 1080],
  detectFps: 3,
  cards: {
    'FAK-001': { name: 'Fake Hero', type: 'Unit', art: 'art/FAK-001.jpg', printing: 'FAK-001' },
    'FAK-002': { name: 'Fake Legend', type: 'Legend', art: 'art/FAK-002.jpg', printing: 'FAK-002' },
  },
  tracks: [],
  events: [],
  ...over,
});

const track: Track = {
  id: 'k0',
  samples: [
    [10, 0.5, 0.5, 0.12, 0.09, 90],
    [10.4, 0.52, 0.5, 0.12, 0.09, 90],
    [20, 0.6, 0.5, 0.12, 0.09, 178], // after a gap (a hand over the card), turned
    [20.4, 0.6, 0.5, 0.12, 0.09, 2], // angle wraps through 0/180
  ],
  guesses: [{ card: 'FAK-001', p: 0.9 }],
};

describe('sampleAt', () => {
  it('interpolates between close samples and holds briefly at the ends', () => {
    expect(sampleAt(track, 10.2)?.[1]).toBeCloseTo(0.51);
    expect(sampleAt(track, 9.6)?.[0]).toBe(10);
    expect(sampleAt(track, 9.0)).toBeUndefined();
    expect(sampleAt(track, 21.2)).toBeUndefined();
  });

  it('shows nothing in the middle of a long gap, and wraps angles the short way', () => {
    expect(sampleAt(track, 10.8)?.[0]).toBe(10.4);
    expect(sampleAt(track, 15)).toBeUndefined();
    expect(sampleAt(track, 19.6)?.[0]).toBe(20);
    expect(sampleAt(track, 20.2)?.[5]).toBeCloseTo(180); // 178 -> 182, not 178 -> 2
  });
});

describe('corners', () => {
  it('puts an upright card (long side vertical) in a tall box', () => {
    const c = corners([0, 0.5, 0.5, 0.12, 0.09, 90], [1920, 1080]);
    const xs = c.map((p) => p[0]);
    const ys = c.map((p) => p[1]);
    expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(0.09 * 1080);
    expect(Math.max(...ys) - Math.min(...ys)).toBeCloseTo(0.12 * 1080);
    expect((Math.max(...xs) + Math.min(...xs)) / 2).toBeCloseTo(960);
  });
});

describe('hover card', () => {
  it('names a card when sure, shows three guesses when not, and never names a face-down card', () => {
    const b = bundle();
    expect(hoverState(b, [{ card: 'FAK-001', p: 0.9 }])).toMatchObject({ kind: 'sure', p: 0.9 });
    const unsure = hoverState(b, [{ card: 'FAK-001', p: 0.5 }, { card: 'FAK-002', p: 0.2 }, { card: 'gone', p: 0.1 }]);
    expect(unsure.kind).toBe('unsure');
    expect(unsure.kind === 'unsure' && unsure.options.map((o) => o.card.name)).toEqual(['Fake Hero', 'Fake Legend']);
    expect(hoverState(b, [{ card: 'FAK-001', p: 0.99 }], true)).toEqual({ kind: 'face-down' });
    expect(hoverState(b, undefined)).toEqual({ kind: 'unknown' });
  });
});

describe('timeline', () => {
  it('finds an event\'s card through its track or its own guesses, and the current event', () => {
    const b = bundle({ tracks: [track] });
    const events = [
      { t: 5, tBefore: 3, kind: 'played' as const, track: 'k0', box: [0, 0, 1, 1] as [number, number, number, number] },
      { t: 9, tBefore: 8, kind: 'changed' as const, guesses: [{ card: 'FAK-002', p: 0.8 }], box: [0, 0, 1, 1] as [number, number, number, number] },
    ];
    expect(eventGuesses(b, events[0]!)?.[0]?.card).toBe('FAK-001');
    expect(eventGuesses(b, events[1]!)?.[0]?.card).toBe('FAK-002');
    expect([currentEvent(events, 1), currentEvent(events, 5), currentEvent(events, 100)]).toEqual([-1, 0, 1]);
    expect(formatTime(83.9)).toBe('1:23');
  });

  it('explains what is missing when the bundle is not there', () => {
    expect(checkBundle(undefined)).toEqual(['no demo data (data.js missing?)']);
    expect(checkBundle(bundle())).toEqual([]);
    expect(checkBundle({ ...bundle(), schema: 'x' })).toEqual(["schema must be 'rifteye.demo'"]);
  });
});
