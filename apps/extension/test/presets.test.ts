import { describe, expect, it } from 'vitest';
import { layouts } from '@rifteye/engine';
import { matchPreset, matStats } from '../src/presets';
import { picture } from './fakes';

const { LAYOUTS } = layouts;
const all = Object.values(LAYOUTS);

describe('finding a preset by its mat', () => {
  it('sees the whole window as mat on a frame of the mat colour', () => {
    const la = picture(384, 216, LAYOUTS['la-rq'].mat!);
    expect(matStats(la, LAYOUTS['la-rq'])).toEqual({ share: 1, dist: 0 });
    expect(matStats(la, LAYOUTS.shenyang).share).toBe(0); // a red mat is nowhere on a navy one
  });

  it('tells the Regional Qualifier from its restream by the colour, not the threshold', () => {
    expect(matchPreset(picture(384, 216, LAYOUTS['la-rq'].mat!), all)?.name).toBe('la-rq');
    expect(matchPreset(picture(384, 216, LAYOUTS.plusrb.mat!), all)?.name).toBe('plusrb');
    expect(matchPreset(picture(384, 216, LAYOUTS.shenyang.mat!), all)?.name).toBe('shenyang');
  });

  it('needs the mat to fill its window as much as the layout says', () => {
    const la = LAYOUTS['la-rq'];
    const frame = picture(384, 216, [200, 200, 200]);
    expect(matchPreset(frame, all)).toBeNull(); // not a table
    // paint the mat over the left half of the window only: 0.5 < 0.62
    const [x0, y0, x1, y1] = layouts.box(la, frame.width, frame.height);
    for (let y = y0; y < y1; y++) for (let x = x0; x < (x0 + x1) / 2; x++) frame.data.set(la.mat!, (y * frame.width + x) * 3);
    expect(matchPreset(frame, [la])).toBeNull();
    expect(matStats(frame, la).share).toBeGreaterThan(0.45);
  });

  it('allows for the light drifting a little', () => {
    const [r, g, b] = LAYOUTS['la-rq'].mat!;
    expect(matchPreset(picture(384, 216, [r + 12, g - 10, b + 8]), [LAYOUTS['la-rq']])?.name).toBe('la-rq');
  });
});
