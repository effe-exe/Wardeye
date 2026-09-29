import { describe, expect, it } from 'vitest';
import { badge, badgeDetail, boxClass, captureSize, contentRect, frameInterval, hoverCard, label, labelAnchor, type State, type Track } from '../src/geometry';

const track = (over: Partial<Track> = {}): Track => ({
  id: 't1',
  quad: [
    [100, 200],
    [160, 200],
    [160, 284],
    [100, 284],
  ],
  side: 'left',
  state: 'named',
  printing_id: 'SFD-195a',
  name: 'Blade Dancer',
  confidence: 0.914,
  guesses: [],
  kind: 'legend',
  hidden: false,
  ...over,
});

describe('the picture inside the player', () => {
  it('is letterboxed on the short side, like object-fit: contain', () => {
    expect(contentRect({ left: 0, top: 0, width: 1600, height: 1000 }, 1920, 1080)).toEqual({ left: 0, top: 50, width: 1600, height: 900 });
    expect(contentRect({ left: 10, top: 20, width: 1000, height: 900 }, 1600, 900)).toEqual({ left: 10, top: 188.75, width: 1000, height: 562.5 });
    const pillar = contentRect({ left: 0, top: 0, width: 2000, height: 900 }, 1600, 900);
    expect(pillar.left).toBe(200);
    expect(pillar.width).toBe(1600);
  });

  it('grabs frames at the video size, at most 1080p wide', () => {
    expect(captureSize(1920, 1080)).toEqual([1920, 1080]);
    expect(captureSize(1280, 720)).toEqual([1280, 720]);
    expect(captureSize(3840, 2160)).toEqual([1920, 1080]);
  });
});

describe('what a card shows', () => {
  it('is named and boxed solid when RiftEye is sure, and says how sure', () => {
    expect(boxClass(track())).toBe('rifteye-box rifteye-named');
    expect(label(track())).toBe('Blade Dancer');
    expect(labelAnchor(track().quad)).toEqual([130, 200]);
    expect(hoverCard(track())).toEqual({ kind: 'named', printing_id: 'SFD-195a', name: 'Blade Dancer', sure: 'RiftEye is 91% sure', under: '' });
  });

  it('shows its best guesses while unsure, and what lies under it', () => {
    const g = [
      { printing_id: 'a', name: 'A', p: 0.4 },
      { printing_id: 'b', name: 'B', p: 0.3 },
      { printing_id: 'c', name: 'C', p: 0.2 },
      { printing_id: 'd', name: 'D', p: 0.1 },
    ];
    const hc = hoverCard(track({ state: 'unsure', guesses: g, under: [{ id: 't2', name: 'Tactical Gear', printing_id: 'x' }] }));
    expect(hc).toEqual({ kind: 'unsure', guesses: g.slice(0, 3), under: 'Under it: Tactical Gear' });
    expect(label(track({ state: 'unsure' }))).toBe('');
  });

  it('never names a face-down card, and runes are boxed faintly with no label or hover', () => {
    expect(hoverCard(track({ state: 'facedown' }))).toEqual({ kind: 'text', text: 'Face-down card: never identified' });
    const rune = track({ kind: 'rune', name: 'Fury Rune' });
    expect(boxClass(rune)).toBe('rifteye-box rifteye-named rifteye-rune');
    expect(label(rune)).toBe('');
    expect(hoverCard(rune)).toBeNull();
  });
});

describe('the badge', () => {
  const state = (over: Partial<State> = {}): State => ({ t: 1, status: 'live', message: '', frame: { width: 1920, height: 1080 }, tracks: [], ...over });
  it('tells what the runner is doing', () => {
    expect(badge(false, null)).toContain('start the runner');
    expect(badge(true, state({ status: 'starting', message: 'finding the table and the size of a card' }))).toBe(
      'RiftEye: finding the table and the size of a card',
    );
    expect(badge(true, state({ status: 'away' }))).toBe('RiftEye: waiting for the table camera');
    expect(badge(true, state({ tracks: [track(), track({ id: 't2', kind: 'rune' }), track({ id: 't3', hidden: true })] }))).toBe('RiftEye · 1 card named');
  });
});

describe('the badge of the engine inside the extension', () => {
  const engine = { runtime: 'webgpu' as const, detector: 'fp32' as const, embedder: 'fp16' as const, every_ms: 200, reads_per_s: 11.94, timing: { decode: 6, detect: 44.5, embed: 22.5, track: 8.2, total: 81.2 }, layout: 'la-rq' };
  const state = (over: Partial<State> = {}): State => ({ t: 1, status: 'live', message: '', frame: { width: 1920, height: 1080 }, tracks: [], engine, ...over });

  it('says the reads a second beside the cards named', () => {
    expect(badge(true, state({ tracks: [track()] }))).toBe('RiftEye · 1 card named · 11.9 reads/s');
    expect(badge(true, state({ status: 'starting', message: 'loading the models (webgpu, fp16)' }))).toBe('RiftEye: loading the models (webgpu, fp16)');
  });

  it('says how it runs and where a frame\'s time goes on a second line, and nothing without the engine', () => {
    expect(badgeDetail(state())).toBe('WebGPU · detector fp32 · embedder fp16 · decode 6 · detect 44.5 · embed 22.5 · track 8.2 · total 81.2 ms · layout la-rq');
    expect(badgeDetail(state({ engine: { ...engine, runtime: 'wasm', embedder: 'fp32' } }))).toContain('WASM · detector fp32 · embedder fp32');
    expect(badgeDetail(state({ engine: undefined as never }))).toBe('');
    expect(badgeDetail(null)).toBe('');
  });

  it('sends frames at the pace the engine asks, and at the live runner\'s otherwise', () => {
    expect(frameInterval(state())).toBe(200);
    expect(frameInterval(state({ engine: undefined as never }))).toBe(250);
    expect(frameInterval(null)).toBe(250);
    expect(frameInterval(state({ engine: { ...engine, every_ms: 5 } }))).toBe(250); // not a pace a page can keep
    expect(frameInterval(state({ engine: undefined as never, retry_ms: 1000 }))).toBe(1000); // it is loading: not so fast
  });
});
