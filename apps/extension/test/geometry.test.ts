import { describe, expect, it } from 'vitest';
import { badge, boxClass, captureSize, contentRect, hoverCard, label, labelAnchor, type State, type Track } from '../src/geometry';

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
