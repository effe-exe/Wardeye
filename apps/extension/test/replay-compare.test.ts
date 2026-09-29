import { describe, expect, it } from 'vitest';
import { compareBoxes, compareRows, compareRuns, firstDifference, jsonLines, recogniserState, summarize, summarizeDiag, type Box, type GotStep, type StepDiag, type WantStep } from '../e2e/replay-compare';

const track = (id: string, x: number, over: Record<string, unknown> = {}) => ({ id, quad: [[x, 1], [x + 5, 1]], state: 'named', name: 'Card', confidence: 0.912, ...over });
const step = (i: number, tracks: unknown[], events: unknown[] = [], extra: Record<string, unknown> = {}): GotStep => ({
  i, file: `f${i}.jpg`, t: i / 2, state: { t: i / 2, status: 'live', tracks, ...extra }, events,
});
const want = (s: GotStep): WantStep => ({ i: s.i, file: s.file, t: s.t, state: { ...s.state }, events: s.events });

describe('the first difference between two values', () => {
  it('is where a number, a string or a length differs, with the path', () => {
    expect(firstDifference({ a: [1, { b: 'x' }] }, { a: [1, { b: 'x' }] }, 'state')).toBeNull();
    expect(firstDifference({ a: [1, { b: 'y' }] }, { a: [1, { b: 'x' }] }, 'state')).toBe('state.a[1].b: "y" here, "x" in Python');
    expect(firstDifference([track('t1', 1)], [track('t1', 1), track('t2', 2)], 'tracks')).toBe('tracks: 1 items here, 2 in Python (extra here: none; missing here: t2)');
    expect(firstDifference({ a: 1 }, { a: 1, b: 2 }, 's')).toBe('s.b: missing here (2 in Python)');
    expect(firstDifference({ a: 1, b: 2 }, { a: 1 }, 's')).toBe('s.b: 2 here, not in Python');
  });

  it('allows numbers to differ by a tolerance and not more', () => {
    expect(firstDifference({ x: 1.0000004 }, { x: 1 }, 's', 1e-6)).toBeNull();
    expect(firstDifference({ x: 1.0000004 }, { x: 1 }, 's', 0)).toBe('s.x: 1.0000004 here, 1 in Python');
    expect(firstDifference({ x: 0.15 }, { x: 0 }, 's', 0.11)).toBe('s.x: 0.15 here, 0 in Python');
  });
});

describe('the run against the reference', () => {
  it('ignores what the host adds to a state', () => {
    const s = step(0, [track('t1', 1)], [], { fps: { source: 5, processed: 3 }, latency_s: 0.1, engine: { runtime: 'wasm' } });
    expect(recogniserState(s.state)).toEqual({ t: 0, status: 'live', tracks: [track('t1', 1)] });
    expect(compareRuns([s], [want(step(0, [track('t1', 1)]))], 0).identical).toBe(1);
  });

  it('counts the identical steps, the events that agree, and says where the first step went its own way', () => {
    const a = [step(0, [track('t1', 1)]), step(1, [track('t1', 2)], [{ kind: 'played' }]), step(2, [track('t1', 3, { confidence: 0.9121 })])];
    const b = [step(0, [track('t1', 1)]), step(1, [track('t1', 2)], [{ kind: 'played' }]), step(2, [track('t1', 3)])];
    const exact = compareRuns(a, b.map(want), 0);
    expect(exact).toMatchObject({ steps: 3, identical: 2, eventsEqual: 3, divergent: [2] });
    expect(exact.firstDivergence).toBe('step 2 (f2.jpg, t=1): state.tracks[0].confidence: 0.9121 here, 0.912 in Python');
    expect(compareRuns(a, b.map(want), 1e-3).identical).toBe(3);
    const events = compareRuns([step(0, [], [{ kind: 'left' }])], [want(step(0, [], [{ kind: 'played' }]))], 0);
    expect(events).toMatchObject({ identical: 0, eventsEqual: 0 });
  });

  it('compares the steps both runs have', () => {
    const a = [step(0, []), step(1, [])];
    expect(compareRuns(a, [want(a[0]!)], 0).steps).toBe(1);
  });

  it('is summarised in a line for each tolerance', () => {
    const a = [step(0, [track('t1', 1)])];
    const text = summarize([compareRuns(a, [want(step(0, [track('t1', 2)]))], 0), compareRuns(a, [want(a[0]!)], 1e-6)], 1, 1);
    expect(text).toContain('steps: 1 run, 1 in the reference');
    expect(text).toMatch(/exact: 0 of 1 steps identical, events equal on 1; first divergence: step 0 \(f0\.jpg, t=0\): state\.tracks\[0\]\.quad\[0\]\[0\]/);
    expect(text).toContain('numbers within 0.000001: 1 of 1 steps identical');
  });

  it('reads a jsonl file', () => {
    expect(jsonLines<{ a: number }>('{"a":1}\n\n{"a":2}\n')).toEqual([{ a: 1 }, { a: 2 }]);
  });
});

const box = (x: number, over: Partial<Box> = {}): Box => ({ centre: [x, 10], long_px: 150, short_px: 100, angle_deg: 90, fill: 1, back: false, score: 0.9, vis: 0.99, ...over });

describe('the finder and the embedder against Python\'s', () => {
  it('finds how far apart two lists of boxes are', () => {
    expect(compareBoxes([box(1), box(2)], [box(1), box(2)])).toEqual({ same: true, maxDelta: 0, back: true, inOrder: true });
    expect(compareBoxes([box(1), box(2.0004, { score: 0.9001 })], [box(1), box(2)])).toEqual({ same: true, maxDelta: expect.closeTo(0.0004, 9), back: true, inOrder: true });
    expect(compareBoxes([box(1)], [box(1), box(2)])).toMatchObject({ same: false });
    expect(compareBoxes([box(1, { back: true })], [box(1)])).toMatchObject({ same: true, back: false });
    // the same boxes in another order are matched by their centres, not by their places in the list
    expect(compareBoxes([box(500), box(1.0001)], [box(1), box(500)])).toEqual({ same: true, maxDelta: expect.closeTo(0.0001, 9), back: true, inOrder: false });
  });

  it('finds the lowest cosine and the largest difference of two sets of rows', () => {
    const a = Float32Array.from([1, 0, 0, 0.6, 0.8, 0]);
    expect(compareRows(a, a, 3)).toEqual({ minCos: expect.closeTo(1, 6), maxAbs: 0 });
    const b = Float32Array.from([1, 0, 0, 0.8, 0.6, 0]);
    const r = compareRows(a, b, 3);
    expect(r.minCos).toBeCloseTo(0.96, 6);
    expect(r.maxAbs).toBeCloseTo(0.2, 6);
  });

  it('says in two lines where they first parted from Python\'s, and how far', () => {
    const d = (i: number, over: Partial<StepDiag> = {}): StepDiag => ({
      i, boxes: { got: 5, want: 5, maxDelta: 1e-5, back: true, inOrder: true }, embeds: { got: [8], want: [8], rows: 8, minCos: 0.9999999, maxAbs: 2e-6 }, ...over,
    });
    expect(summarizeDiag([d(0), d(1)])).toBe(
      'finder: 2 of 2 steps gave as many boxes as Python\'s, the largest difference in a number of a matched box 1.00e-5, 0 steps with the boxes in another order\n' +
        'embedder: 16 rows compared with Python\'s, lowest cosine 0.999999900, largest difference in a value 2.00e-6; every step made the same embed calls',
    );
    const text = summarizeDiag([d(0), d(1, { boxes: { got: 4, want: 5, maxDelta: NaN, back: true, inOrder: false }, embeds: { got: [8, 4], want: [8], rows: 8, minCos: 1, maxAbs: 0 } })]);
    expect(text).toContain('first step with other boxes: 1');
    expect(text).toContain("first step whose embed calls differ from Python's: 1");
  });
});
