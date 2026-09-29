import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { layouts, recognizer } from '@rifteye/engine';
import { levelsFor, pickLevels, pyRound } from '../src/levels';

describe("Python's round", () => {
  it('sends a half to the even neighbour', () => {
    expect([0.5, 1.5, 2.5, 3.5, 4.5].map(pyRound)).toEqual([0, 2, 2, 4, 4]);
    expect([12.4, 12.6, 9.45, 10.5].map(pyRound)).toEqual([12, 13, 9, 10]);
  });
});

describe('the levels a table reads against', () => {
  it('are the live runner\'s: round(px * f / 10) * 10 for f in 0.8, 0.9, 1.0', () => {
    expect(levelsFor(155)).toEqual([120, 140, 160]); // la-rq
    expect(levelsFor(140)).toEqual([110, 130, 140]); // plusrb
    expect(levelsFor(131)).toEqual([100, 120, 130]); // shenyang
    expect(levelsFor(100)).toEqual([80, 90, 100]);
    expect(levelsFor(105)).toEqual([80, 90, 100]); // 10.5 is a half: to the even 10
  });

  it('are the formula the runner has in __main__.py (a drift alarm)', () => {
    const main = readFileSync(fileURLToPath(new URL('../../../ml/rifteye_ml/live/__main__.py', import.meta.url)), 'utf8');
    expect(main).toContain('int(round(px * f / 10) * 10) for f in (0.8, 0.9, 1.0)');
  });

  it('are the ones the engine\'s Recognizer is given a pyramid of', () => {
    const held = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
    for (const layout of Object.values(layouts.LAYOUTS)) {
      expect(pickLevels(layout.card_long_1080, held)).toEqual(recognizer.galleryScales(layout));
    }
    for (let px = 100; px <= 200; px += 0.5) expect(levelsFor(px)).toEqual(recognizer.galleryScales(layouts.makeLayout({ name: 'x', title: 'x', table: [0, 0, 1, 1], card_long_1080: px })));
  });

  it('come from the package as they are, or as the nearest it holds', () => {
    const held = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
    expect(pickLevels(155, held)).toEqual([120, 140, 160]);
    expect(pickLevels(230, held)).toEqual([180, 200]); // bigger cards than the package was made for: 210 and 230 give way to its largest
    expect(pickLevels(60, held)).toEqual([80]); // 50 and 60 give way to its smallest
    expect(pickLevels(155, [100, 150])).toEqual([100, 150]); // 120 -> 100, 140 -> 150, 160 -> 150
    expect(() => pickLevels(155, [])).toThrow('no levels');
  });
});
