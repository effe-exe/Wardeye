// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What the browser tests of the standalone mode share: the extension loaded with a stand-in package (or none, the public
// build), and the stand-in Twitch.

import { spawnSync } from 'node:child_process';
import { linkSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, type Plugin } from 'esbuild';
import { chromium, type BrowserContext, type Page } from '@playwright/test';
import { mirror } from '../../bench/e2e/standins';
import { standInFiles, type StandIn } from './standins';
import { makeVideo, routeTwitch } from './twitch';

const require = createRequire(import.meta.url);
const EXT = fileURLToPath(new URL('../', import.meta.url));
const DIST = join(EXT, 'dist');
const BUILD = join(EXT, 'build.mjs');
const ORT_FILES = ['ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];

/** Swaps the real parts (the engine) for the stand-ins in the engine workers' bundles. */
const standIn: Plugin = {
  name: 'stand-in-parts',
  setup(b) {
    b.onResolve({ filter: /\/parts-engine$/ }, () => ({ path: join(EXT, 'e2e', 'parts-standin.ts') }));
  },
};

export interface Loaded {
  root: string;
  ext: string;
  context: BrowserContext;
}

export interface LoadOptions {
  package?: StandIn | null;
  gpu?: boolean;
  /** The Chrome Web Store build (node build.mjs --store, made here) with the store's package: no catalogue, no thumbnails. */
  store?: boolean;
  /** More arguments for Chromium (the fake Riot's: fake-riot.ts). */
  args?: readonly string[];
}

/** A copy of the public build with the engine's stand-in bundles, onnxruntime-web's runtime files and (unless it is
 * asked to be the public build) the stand-in package; loaded in Chromium. With `store` it is the store build instead,
 * built for the test into the folder it is copied from. WebGPU is on with a software adapter (SwiftShader) unless
 * `gpu` is false. */
export async function load(opts: LoadOptions = {}): Promise<Loaded> {
  const root = mkdtempSync(join(tmpdir(), 'rifteye-standalone-'));
  const ext = join(root, 'ext');
  let dist = DIST;
  if (opts.store) {
    dist = join(root, 'dist-store');
    const built = spawnSync('node', [BUILD, '--store', '--out', dist], { encoding: 'utf8' });
    if (built.status !== 0) throw new Error(`the store build failed: ${built.stderr}`);
  }
  mirror(dist, ext); // hard links: the engine's bundles are replaced below, never written through
  if (opts.package !== null) {
    for (const name of ['engine-webgpu', 'engine-wasm']) {
      rmSync(join(ext, `${name}.js`));
      await build({
        entryPoints: [join(EXT, 'src', `${name}.ts`)],
        bundle: true,
        format: 'esm',
        target: 'es2022',
        outfile: join(ext, `${name}.js`),
        conditions: ['onnxruntime-web-use-extern-wasm'],
        plugins: [standIn],
        logLevel: 'warning',
      });
    }
    mkdirSync(join(ext, 'ort'), { recursive: true });
    for (const f of ORT_FILES) linkSync(require.resolve(`onnxruntime-web/${f}`), join(ext, 'ort', f));
    for (const [path, content] of Object.entries(standInFiles({ ...(opts.package ?? {}), ...(opts.store ? { store: true } : {}) }))) {
      mkdirSync(dirname(join(ext, path)), { recursive: true });
      writeFileSync(join(ext, path), content);
    }
  }
  const context = await chromium.launchPersistentContext(join(root, 'profile'), {
    headless: true,
    args: [
      `--disable-extensions-except=${ext}`,
      `--load-extension=${ext}`,
      '--autoplay-policy=no-user-gesture-required',
      ...(opts.gpu === false ? [] : ['--enable-unsafe-webgpu']),
      ...(opts.args ?? []),
    ],
    // Playwright's default headless shell cannot load extensions; its full Chromium (channel) can
    ...(process.env.RIFTEYE_CHROMIUM ? { executablePath: process.env.RIFTEYE_CHROMIUM } : { channel: 'chromium' }),
  });
  return { root, ext, context };
}

export async function unload(l: Loaded | null): Promise<void> {
  await l?.context.close();
  if (l) rmSync(l.root, { recursive: true, force: true });
}

/** The ids of the engine documents there are (chrome.runtime.getContexts). */
export const documentIds = async (context: BrowserContext): Promise<string[]> => {
  const sw = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
  return sw.evaluate(async () => (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })).map((c) => String((c as { documentId?: string }).documentId)));
};
export const offscreenDocuments = async (context: BrowserContext): Promise<number> => (await documentIds(context)).length;

/** The stand-in Twitch, with its video (`seconds` long) recorded first. */
export async function twitch(context: BrowserContext, seconds = 16): Promise<Page> {
  const page = await context.newPage();
  await page.goto('about:blank');
  await routeTwitch(context, await makeVideo(page, seconds, ['#c0a040', '#40a0c0']));
  return page;
}

