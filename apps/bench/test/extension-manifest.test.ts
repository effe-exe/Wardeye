import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// The extension's own manifest.json (not a model manifest): what makes it load, run WASM threads and stay local.
const manifest = JSON.parse(readFileSync(fileURLToPath(new URL('../src/manifest.json', import.meta.url)), 'utf8'));

describe('the extension manifest', () => {
  it('is a Manifest V3 extension named RiftEye bench, version 0.1.1', () => {
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.name).toBe('RiftEye bench');
    expect(manifest.version).toBe('0.1.1');
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
