import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// The extension's own manifest.json (not a model manifest): what makes it load, run WASM threads and stay local.
const BENCH = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(readFileSync(`${BENCH}src/manifest.json`, 'utf8'));

describe('the extension manifest', () => {
  it('is a Manifest V3 extension named Wardeye bench, with a version Chrome takes', () => {
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.name).toBe('Wardeye bench');
    // one to four dot-separated integers, each up to 65535: which version it is changes with every release, so it is not written out here
    expect(manifest.version).toMatch(/^\d+(\.\d+){0,3}$/);
    expect((manifest.version as string).split('.').every((n) => Number(n) <= 65535)).toBe(true);
  });

  it('says Wardeye wherever a person reads it', () => {
    expect([manifest.name, manifest.description, manifest.action.default_title].join('\n')).not.toMatch(/rifteye/i);
    expect(manifest.description.length).toBeLessThanOrEqual(132); // what Chrome takes of a description
  });

  it("declares the brand's icons at Chrome's four sizes, and each file it names is a PNG of that size", () => {
    expect(manifest.icons).toEqual({
      '16': 'icons/icon-16.png',
      '32': 'icons/icon-32.png',
      '48': 'icons/icon-48.png',
      '128': 'icons/icon-128.png',
    });
    for (const [size, file] of Object.entries(manifest.icons as Record<string, string>)) {
      const png = readFileSync(`${BENCH}${file}`); // build.mjs copies each of them into dist/ at this same path
      expect([...png.subarray(0, 8)], file).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]); // the PNG signature
      expect([png.readUInt32BE(16), png.readUInt32BE(20)], file).toEqual([Number(size), Number(size)]); // the width and height in the IHDR chunk
    }
  });

  it('allows wasm (and nothing remote) in its pages', () => {
    expect(manifest.content_security_policy).toEqual({ extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'" });
  });

  it('isolates its pages from other origins, so that shared memory (WASM threads) works', () => {
    expect(manifest.cross_origin_embedder_policy).toEqual({ value: 'require-corp' });
    expect(manifest.cross_origin_opener_policy).toEqual({ value: 'same-origin' });
  });

  it('asks for no host permission and no permission at all', () => {
    expect(manifest.host_permissions).toBeUndefined();
    expect(manifest.optional_host_permissions).toBeUndefined();
    expect(manifest.permissions).toBeUndefined();
    expect(manifest.content_scripts).toBeUndefined();
  });

  it('has a toolbar button and a background worker that opens the page', () => {
    expect(manifest.action.default_title).toBeTruthy();
    expect(manifest.action.default_popup).toBeUndefined(); // no popup: a click reaches the worker
    expect(manifest.background).toEqual({ service_worker: 'worker.js', type: 'module' });
  });
});
