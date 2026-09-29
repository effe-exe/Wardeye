import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MARK_SHAPES, MARK_VIEWBOX } from '../src/mark';

const MARK_SVG = readFileSync(new URL('../../../assets/brand/logo/mark.svg', import.meta.url), 'utf8');
const attrsOf = (tag: string): Record<string, string> => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map((m) => [m[1]!, m[2]!]));

describe("the badge's mark", () => {
  // the brand's file is the source: the badge draws the same shapes, painted from the tokens instead of its colours
  const brand = [...MARK_SVG.matchAll(/<(path|circle)\b[^>]*>/g)].map((m) => ({ tag: m[1], attrs: attrsOf(m[0]) }));

  it("has the shapes of the brand's mark.svg, in the same order, and its viewBox", () => {
    expect(attrsOf(MARK_SVG.match(/<svg\b[^>]*>/)![0]).viewBox).toBe(MARK_VIEWBOX);
    expect(brand).toHaveLength(MARK_SHAPES.length);
    const geometry = (attrs: Record<string, string>) => Object.fromEntries(Object.entries(attrs).filter(([k]) => k !== 'fill' && k !== 'stroke'));
    expect(MARK_SHAPES.map((s) => ({ tag: s.tag, attrs: { ...s.attrs } }))).toEqual(brand.map((s) => ({ tag: s.tag, attrs: geometry(s.attrs) })));
  });

  it('writes no colour: every shape is painted by a class of the stylesheet', () => {
    for (const shape of MARK_SHAPES) {
      expect(shape.cls).toMatch(/^rifteye-mark-[a-z]+$/);
      expect(Object.keys(shape.attrs).filter((k) => ['fill', 'stroke', 'style', 'opacity'].includes(k))).toEqual([]);
      expect(Object.values(shape.attrs).some((v) => /#[0-9a-f]{3,8}\b/i.test(v))).toBe(false);
    }
  });
});
