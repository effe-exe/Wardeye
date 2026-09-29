// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The detector and the embedder in Chromium, as the extension runs them, against Python on the same pictures:
// frames decoded the extension's way must be Pillow's bytes, the tiles and letterboxed crops the same bytes as
// Python's, and onnxruntime-web's WASM build (float32) must find Python's cards (ONNX Runtime CPU, float32: scores
// within 1e-4, corners within 0.1 px) and give its crop rows (cosine >= 0.99999). A handful of frames, as WASM is
// slow: three LA frames (one tile each) and two Barcelona ones (two tiles each).
//
// Needs the private fixtures and models (D-006) and skips without them:
//   RIFTEYE_M3=~/rifteye-data/m3 RIFTEYE_CHROMIUM=/path/to/chrome npx playwright test packages/engine/e2e/vision.spec.ts
// The models are read from RIFTEYE_MODELS, by default the m3 folder's ../models/onnx.

import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { createRequire } from 'node:module';
import type { AddressInfo } from 'node:net';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { build } from 'esbuild';

const M3 = process.env.RIFTEYE_M3 ?? '';
const VIS = join(M3, 'fixtures', 'vision');
const MODELS = process.env.RIFTEYE_MODELS ?? join(M3, '..', 'models', 'onnx');
const ORT_DIST = dirname(createRequire(import.meta.url).resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'));
const THREADS = 2;
const ready = Boolean(M3) && existsSync(join(VIS, 'detector', 'index.json')) && existsSync(join(MODELS, 'detector-v0.onnx'));

const TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
};

interface Card {
  cls: string;
  score: number;
  quad: [number, number][];
}
interface PyCard {
  cls: string;
  score: number;
  quad: number[];
}

let pageJs = '';
let server: Server | null = null;
let origin = '';

/** A file under `root`, never outside it. */
function under(root: string, rest: string): string {
  const p = normalize(join(root, decodeURIComponent(rest)));
  if (!p.startsWith(normalize(root))) throw new Error(`${rest} is outside ${root}`);
  return p;
}

/** The page, the runtime files, the models and the fixtures from localhost (a secure context), streamed, and cross-origin
 * isolated so that WASM can have threads. */
function serve(): Promise<void> {
  const headers = (type: string) => ({
    'content-type': type,
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cross-origin-resource-policy': 'same-origin',
  });
  server = createServer((req, res) => {
    const path = new URL(req.url ?? '/', 'http://localhost').pathname;
    const body = (type: string, text: string) => res.writeHead(200, headers(type)).end(text);
    try {
      if (path === '/index.html') return body('text/html', '<!doctype html><meta charset="utf-8"><script type="module" src="/page.js"></script>');
      if (path === '/page.js') return body('text/javascript', pageJs);
      if (path === '/favicon.ico') return res.writeHead(204).end();
      const file = path.startsWith('/ort/') ? under(ORT_DIST, path.slice(5)) : path.startsWith('/models/') ? under(MODELS, path.slice(8)) : path.startsWith('/m3/') ? under(M3, path.slice(4)) : null;
      if (!file || !existsSync(file)) return res.writeHead(404).end('not here');
      res.writeHead(200, { ...headers(TYPES[extname(file)] ?? 'application/octet-stream'), 'content-length': String(statSync(file).size) });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(400).end('bad path');
    }
  });
  return new Promise((done) =>
    server!.listen(0, '127.0.0.1', () => {
      origin = `http://localhost:${(server!.address() as AddressInfo).port}`;
      done();
    }),
  );
}

test.afterAll(async () => {
  await new Promise((done) => (server ? server.close(done) : done(null)));
});

test.beforeAll(async () => {
  if (!ready) return;
  await serve();
  const out = await build({
    entryPoints: [fileURLToPath(new URL('./vision-page.ts', import.meta.url))],
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    conditions: ['onnxruntime-web-use-extern-wasm'],
    logLevel: 'warning',
  });
  pageJs = out.outputFiles[0]!.text;
});

/** Opens the page; its errors and a crash go to the test's output. */
async function open(page: Page): Promise<void> {
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') console.log(`page ${m.type()}: ${m.text().slice(0, 300)}`);
  });
  page.on('pageerror', (e) => console.log(`page error: ${e.message}`));
  page.on('crash', () => console.log('the page crashed'));
  await page.goto(`${origin}/index.html`);
  await expect(page).toHaveTitle('ready', { timeout: 60_000 });
}

/** Each expected card matched to a found one of its class with the closest corners; the worst differences. */
function match(got: readonly Card[], want: readonly PyCard[]) {
  const pairs: [number, number, number][] = [];
  want.forEach((w, i) =>
    got.forEach((g, j) => {
      if (g.cls === w.cls) pairs.push([Math.max(...g.quad.map(([x, y], k) => Math.hypot(x - w.quad[2 * k]!, y - w.quad[2 * k + 1]!))), i, j]);
    }),
  );
  pairs.sort((a, b) => a[0] - b[0]);
  const wi = new Set<number>();
  const gj = new Set<number>();
  let corner = 0;
  let score = 0;
  for (const [c, i, j] of pairs) {
    if (wi.has(i) || gj.has(j) || c > 5) continue;
    wi.add(i);
    gj.add(j);
    corner = Math.max(corner, c);
    score = Math.max(score, Math.abs(got[j]!.score - want[i]!.score));
  }
  return { missing: want.length - wi.size, extra: got.length - gj.size, corner, score };
}

test.describe('vision in Chromium', () => {
  test.skip(!ready, 'needs the private fixtures and models: RIFTEYE_M3 (and RIFTEYE_MODELS)');

  test('the detector on WASM finds Python\'s cards in frames decoded as the extension decodes them', async ({ page }) => {
    test.setTimeout(30 * 60_000);
    await open(page);
    const la = JSON.parse(readFileSync(join(M3, 'frames', 'la-final', 'frames.json'), 'utf8')) as { frames: { file: string; rgb_sha256: string }[] };
    const bcn = JSON.parse(readFileSync(join(VIS, 'barcelona', 'frames.json'), 'utf8')) as { frames: { file: string; rgb_sha256: string }[] };
    const picks = [
      { id: 'la-f0000', url: '/m3/frames/la-final/f0000.jpg', sha: la.frames.find((f) => f.file === 'f0000.jpg')!.rgb_sha256 },
      { id: 'la-f0120', url: '/m3/frames/la-final/f0120.jpg', sha: la.frames.find((f) => f.file === 'f0120.jpg')!.rgb_sha256 },
      { id: 'la-f0239', url: '/m3/frames/la-final/f0239.jpg', sha: la.frames.find((f) => f.file === 'f0239.jpg')!.rgb_sha256 },
      { id: 'bcn-b0170', url: '/m3/fixtures/vision/barcelona/b0170.jpg', sha: bcn.frames.find((f) => f.file === 'b0170.jpg')!.rgb_sha256 },
      { id: 'bcn-b0980', url: '/m3/fixtures/vision/barcelona/b0980.jpg', sha: bcn.frames.find((f) => f.file === 'b0980.jpg')!.rgb_sha256 },
    ];
    type Fixture = { window: number[]; card_px: number; tiles: { rgb_sha256: string; input_sha256: string }[]; detect: PyCard[] };
    const fx = picks.map((p) => JSON.parse(readFileSync(join(VIS, 'detector', `${p.id}.json`), 'utf8')) as Fixture);
    const jobs = picks.map((p, i) => ({ url: p.url, window: fx[i]!.window, cardPx: fx[i]!.card_px }));
    type Result = { loadMs: number; isolated: boolean; threads: number; frames: { rgbSha: string; tileShas: string[]; inputShas: string[]; cards: Card[]; ms: number }[] };
    const r = await page.evaluate(
      ([jobs, threads]) => (globalThis as unknown as { rifteye: { runDetector(m: string, f: unknown, t: number): Promise<unknown> } }).rifteye.runDetector('detector-v0.onnx', jobs, threads),
      [jobs, THREADS] as const,
    ) as Result;
    expect(r.isolated).toBe(true);
    const lines: string[] = [];
    r.frames.forEach((f, i) => {
      expect(f.rgbSha, `${picks[i]!.id}: Chromium decodes the JPEG to Pillow's RGB`).toBe(picks[i]!.sha);
      expect(f.tileShas).toEqual(fx[i]!.tiles.map((t) => t.rgb_sha256));
      expect(f.inputShas).toEqual(fx[i]!.tiles.map((t) => t.input_sha256));
      const m = match(f.cards, fx[i]!.detect);
      expect(m, picks[i]!.id).toMatchObject({ missing: 0, extra: 0 });
      expect(m.corner).toBeLessThanOrEqual(0.1 + 1e-9);
      expect(m.score).toBeLessThanOrEqual(1e-4 + 1e-9);
      lines.push(`${picks[i]!.id}: ${f.cards.length} cards, ${f.tileShas.length} tile(s), ${f.ms.toFixed(0)} ms, worst corner ${m.corner.toFixed(2)} px, score ${m.score.toExponential(1)}`);
    });
    console.log(`detector fp32 on WASM in Chromium (${r.threads} threads, isolated ${r.isolated}; session ${r.loadMs.toFixed(0)} ms):\n  ${lines.join('\n  ')}`);
  });

  test('the embedder on WASM gives Python\'s rows for the 40 crops, under Python\'s name', async ({ page }) => {
    test.setTimeout(30 * 60_000);
    test.skip(!existsSync(join(MODELS, 'embedder-v1.onnx')) || !existsSync(join(VIS, 'crops', 'crops.json')), 'no embedder or crops');
    await open(page);
    const c = JSON.parse(readFileSync(join(VIS, 'crops', 'crops.json'), 'utf8')) as { encoder: string; dim: number; crops: { id: string; letterbox_sha256: string }[] };
    type Result = { name: string; letterboxShas: string[]; rows: number[]; ms: number; batches: number };
    const r = await page.evaluate(
      ([crops, threads]) => (globalThis as unknown as { rifteye: { runEmbedder(m: string, c: string[], t: number): Promise<unknown> } }).rifteye.runEmbedder('embedder-v1.onnx', crops, threads),
      [c.crops.map((m) => `/m3/fixtures/vision/crops/${m.id}.png`), THREADS] as const,
    ) as Result;
    expect(r.name).toBe(c.encoder);
    expect(r.letterboxShas).toEqual(c.crops.map((m) => m.letterbox_sha256));
    const b = readFileSync(join(VIS, 'crops', 'embed.bin'));
    const want = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    let worst = 1;
    for (let i = 0; i < c.crops.length; i++) {
      let dot = 0;
      for (let k = 0; k < c.dim; k++) dot += r.rows[i * c.dim + k]! * want[i * c.dim + k]!;
      worst = Math.min(worst, dot);
    }
    console.log(`embedder fp32 on WASM in Chromium (${THREADS} threads): ${(r.ms / r.batches).toFixed(0)} ms a batch of 8; lowest cosine ${worst.toFixed(8)}`);
    expect(worst).toBeGreaterThanOrEqual(0.99999);
  });
});
