import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { MARK_SHAPES } from '../src/mark';

const EXT = fileURLToPath(new URL('../', import.meta.url));
const BRAND = fileURLToPath(new URL('../../../assets/brand/', import.meta.url));
const stripComments = (css: string): string => css.replace(/\/\*[\s\S]*?\*\//g, '');

interface BrandHelper {
  FONTS: { family: string; file: string; weight: string }[];
  tokensCss(selector?: string): string;
  fontFaceCss(url?: (file: string) => string): string;
}
interface Manifest {
  name: string;
  description: string;
  icons: Record<string, string>;
  content_scripts: { matches: string[]; js: string[]; css: string[] }[];
  web_accessible_resources: { resources: string[]; matches: string[] }[];
  [key: string]: unknown;
}

/** The css without its @keyframes (balanced braces), and the names of those. */
function withoutKeyframes(css: string): { names: string[]; rest: string } {
  const names: string[] = [];
  let rest = '';
  let from = 0;
  const re = /@keyframes\s+([\w-]+)\s*\{/g;
  for (let m = re.exec(css); m; m = re.exec(css)) {
    rest += css.slice(from, m.index);
    names.push(m[1]!);
    let depth = 1;
    let j = re.lastIndex;
    while (depth > 0 && j < css.length) {
      const c = css[j++];
      if (c === '{') depth++;
      else if (c === '}') depth--;
    }
    from = j;
    re.lastIndex = j;
  }
  return { names, rest: rest + css.slice(from) };
}

describe('the build of the extension', () => {
  let dist: string;
  let brand: BrandHelper;
  let manifest: Manifest;
  const built = (path: string): string => readFileSync(join(dist, path), 'utf8');
  const listFiles = (dir: string, prefix = ''): string[] =>
    readdirSync(join(dir, prefix), { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? listFiles(dir, join(prefix, e.name)) : [join(prefix, e.name)]));
  const rules = stripComments(readFileSync(join(EXT, 'src', 'overlay.css'), 'utf8')); // the overlay's own rules
  const tokens = readFileSync(join(BRAND, 'tokens.css'), 'utf8');

  beforeAll(async () => {
    dist = mkdtempSync(join(tmpdir(), 'wardeye-build-test-'));
    const run = spawnSync('node', [join(EXT, 'build.mjs'), '--out', dist], { encoding: 'utf8' });
    expect(run.status, run.stderr).toBe(0);
    brand = (await import(pathToFileURL(join(BRAND, 'brand.mjs')).href)) as BrandHelper; // a variable specifier: the helper is plain JS
    manifest = JSON.parse(built('manifest.json')) as Manifest;
  }, 60_000);

  afterAll(() => rmSync(dist, { recursive: true, force: true }));

  it('names the extension Wardeye, with a description that fits the store (132 characters) and says nothing of RiftEye', () => {
    expect(manifest.name).toBe('Wardeye');
    expect(manifest.description.length).toBeLessThanOrEqual(132);
    expect(manifest.description).toContain('Wardeye');
    expect(JSON.stringify(manifest)).not.toMatch(/rifteye/i);
  });

  it('declares its four icons, and ships them: PNGs of the sizes they are named for', () => {
    expect(Object.keys(manifest.icons).sort((a, b) => Number(a) - Number(b))).toEqual(['16', '32', '48', '128']);
    for (const [size, path] of Object.entries(manifest.icons)) {
      const png = readFileSync(join(dist, path));
      expect(png.subarray(1, 4).toString(), path).toBe('PNG');
      expect([png.readUInt32BE(16), png.readUInt32BE(20)], path).toEqual([Number(size), Number(size)]);
    }
  });

  it('copies the three typefaces, and each one licence, into dist/fonts, and makes those typefaces web accessible to twitch.tv and nothing else', () => {
    for (const { file } of brand.FONTS) expect(readFileSync(join(dist, file)).equals(readFileSync(join(BRAND, file))), file).toBe(true);
    expect(readdirSync(join(dist, 'fonts')).filter((n) => n.startsWith('OFL-')).sort()).toEqual(['OFL-Inter.txt', 'OFL-JetBrainsMono.txt', 'OFL-SpaceGrotesk.txt']);
    expect(manifest.web_accessible_resources).toEqual([{ resources: brand.FONTS.map((f) => f.file), matches: ['https://www.twitch.tv/*'] }]);
    for (const r of manifest.web_accessible_resources.flatMap((w) => w.resources)) expect(existsSync(join(dist, r)), r).toBe(true);
  });

  it("writes the overlay's stylesheet as the brand's tokens, its typefaces from the extension's own files, then the overlay's rules", () => {
    const css = built('overlay.css');
    const at = (part: string): number => css.indexOf(part);
    const faces = brand.fontFaceCss((file) => `chrome-extension://__MSG_@@extension_id__/${file}`);
    expect(at(brand.tokensCss('.rifteye-root'))).toBeGreaterThan(-1);
    expect(at(faces)).toBeGreaterThan(at(brand.tokensCss('.rifteye-root')));
    expect(at(readFileSync(join(EXT, 'src', 'overlay.css'), 'utf8'))).toBeGreaterThan(at(faces));
    // the extension's own files, never a data: URL (the fonts would be in the stylesheet twice) or the network
    expect(faces.match(/url\(([^)]+)\)/g)).toHaveLength(3);
    expect([...faces.matchAll(/url\(([^)]+)\)/g)].every((m) => /^chrome-extension:\/\/__MSG_@@extension_id__\/fonts\/[\w-]+\.woff2$/.test(m[1]!))).toBe(true);
    expect(css).not.toMatch(/https?:\/\/|url\(\s*['"]?data:/);
  });

  it("puts the tokens on the overlay's root and nowhere else, so nothing reaches Twitch's own styles", () => {
    const css = stripComments(built('overlay.css'));
    expect(css).not.toMatch(/:root/);
    expect(css).toMatch(/\.rifteye-root \{\s*--wd-bg:/);
    const defined = [...css.matchAll(/(--wd-[\w-]+)\s*:/g)].map((m) => m[1]);
    const brandTokens = [...stripComments(tokens).matchAll(/(--wd-[\w-]+)\s*:/g)].map((m) => m[1]);
    expect(defined).toEqual(brandTokens); // each token once, from the brand's file: the rules define none of their own
  });

  it('is scoped to the overlay: every rule is on a rifteye- class, the keyframes are named rifteye-, and the only properties it defines are its own', () => {
    const { names, rest } = withoutKeyframes(rules);
    expect(names.length).toBeGreaterThan(0);
    expect(names.every((n) => n.startsWith('rifteye-'))).toBe(true);
    const selectors = [...rest.matchAll(/([^{}]+)\{/g)].map((m) => m[1]!.trim()).filter((s) => !s.startsWith('@media'));
    expect(selectors.length).toBeGreaterThan(20);
    for (const s of selectors) for (const one of s.split(',')) expect(one.trim(), one).toMatch(/\.rifteye-[\w-]+/);
    expect([...rest.matchAll(/(--[\w-]+)\s*:/g)].map((m) => m[1]!).filter((p) => !p.startsWith('--rifteye-'))).toEqual([]);
  });

  it('uses the tokens for every colour: no colour of its own, and a token colour only in a transparency mix', () => {
    const { rest } = withoutKeyframes(rules);
    const builtRest = stripComments(built('overlay.css').replace(brand.tokensCss('.rifteye-root'), '')); // the stylesheet but for the brand's own tokens
    for (const css of [rules, builtRest]) {
      expect(css).not.toMatch(/#[0-9a-f]{3,8}\b/i);
      expect(css).not.toMatch(/\b(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch|color)\(/i);
      expect(css).not.toMatch(/:[^;{}]*\b(?:white|black|red|green|blue|yellow|orange|purple|gray|grey|silver|gold|violet)\b/i);
    }
    const mixes = rest.match(/color-mix\(/g) ?? [];
    expect(mixes.length).toBeGreaterThan(0);
    expect(rest.match(/color-mix\(in srgb, var\(--wd-[\w-]+\) \d+%, transparent\)/g) ?? []).toHaveLength(mixes.length);
    // and every variable it reads is one that exists (a token, or a property of its own)
    const brandTokens = new Set([...stripComments(tokens).matchAll(/(--wd-[\w-]+)\s*:/g)].map((m) => m[1]!));
    const own = new Set([...rest.matchAll(/(--rifteye-[\w-]+)\s*:/g)].map((m) => m[1]!));
    for (const m of rules.matchAll(/var\((--[\w-]+)/g)) expect(brandTokens.has(m[1]!) || own.has(m[1]!), m[1]).toBe(true);
  });

  it('moves as little as the brand book says: a 150 ms fade for the hover card, one 500 ms pulse for a named card, nothing looping, and both off for reduced motion', () => {
    const ms = (token: string): number => Number(new RegExp(`${token}:\\s*(\\d+)ms`).exec(tokens)![1]);
    expect(ms('--wd-fast')).toBe(150);
    expect(2 * ms('--wd-slow')).toBeLessThanOrEqual(600);
    expect(rules).toMatch(/\.rifteye-card\s*\{[^}]*animation:\s*rifteye-rise\s+var\(--wd-fast\)\s+var\(--wd-ease\)/);
    expect(rules).toMatch(/\.rifteye-pulse\s*\{\s*animation:\s*rifteye-pulse\s+calc\(var\(--wd-slow\)\s*\*\s*2\)\s+var\(--wd-ease\)\s+1\s*;\s*\}/);
    expect(rules).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{\s*\.rifteye-card,\s*\.rifteye-pulse\s*\{\s*animation:\s*none\s*;?\s*\}\s*\}/);
    expect(rules).not.toMatch(/infinite|transition/);
    expect(rules.match(/animation:/g)).toHaveLength(3); // the fade, the pulse, and reduced motion's off
  });

  it('styles every class the overlay makes', () => {
    const made = new Set<string>([
      ...readFileSync(join(EXT, 'src', 'content.ts'), 'utf8').matchAll(/'(rifteye-[\w-]+)'/g),
    ].map((m) => m[1]!));
    for (const c of ['rifteye-box', 'rifteye-named', 'rifteye-unsure', 'rifteye-facedown', 'rifteye-new', 'rifteye-rune', 'rifteye-pulse']) made.add(c); // geometry.ts's boxClass, and the pulse
    for (const shape of MARK_SHAPES) made.add(shape.cls);
    expect(made.size).toBeGreaterThan(25);
    for (const c of made) expect(rules, c).toMatch(new RegExp(`\\.${c}(?![\\w-])`));
  });

  it('is what pack.mjs packs: the private build takes every file of dist, so the icons and the typefaces the manifest names go into the zip', () => {
    const list = /const DIST_FILES = \[([^\]]*)\]/.exec(readFileSync(join(EXT, 'pack.mjs'), 'utf8'))![1]!;
    const packed = [...stripComments(list.replace(/\/\/.*$/gm, '')).matchAll(/'([^']+)'/g)].map((m) => m[1]!).sort();
    expect(packed).toEqual(listFiles(dist).sort());
    for (const path of [...Object.values(manifest.icons), ...manifest.web_accessible_resources.flatMap((w) => w.resources), ...manifest.content_scripts.flatMap((c) => [...c.js, ...c.css])]) {
      expect(packed, path).toContain(path);
    }
  });
});
