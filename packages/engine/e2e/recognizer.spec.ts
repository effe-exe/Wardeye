// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The recognizer in Chromium against its Python reference, all 240 steps of the LA final (2 min at 2 fps): the JPEG
// frames decoded as the extension decodes them, the finder's boxes and the encoder's rows as
// test/gen/recognizer_replay.py recorded them (each crop the encoder is handed checked against Python's first), and
// every step's state and events compared with Python's, numbers to 1e-6. Needs the private data (RIFTEYE_M3, e.g.
// ~/rifteye-data/m3) and skips without it. RIFTEYE_REPLAY_LENIENT=1 goes on past crops unlike Python's, counting
// them; RIFTEYE_REPLAY_STEPS=n replays only the first n steps; RIFTEYE_REPLAY_TOLERANCE=0 asks for the same bits.

import { existsSync } from 'node:fs';
import { join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { build } from 'esbuild';
import type { PageResult } from './recognizer-page';

const M3 = process.env.RIFTEYE_M3;
const FX = M3 ? join(M3, 'fixtures', 'recognizer') : '';
const FRAMES = M3 ? join(M3, 'frames', 'la-final') : '';
const HAVE = !!M3 && existsSync(join(FX, 'meta.json')) && existsSync(join(FRAMES, 'frames.json'));
// localhost is a secure context, so the page has crypto.subtle; nothing listens there: the route answers
const ORIGIN = 'http://localhost:8765';

async function run(page: Page, base: string): Promise<PageResult> {
  const bundle = await build({
    entryPoints: [fileURLToPath(new URL('./recognizer-page.ts', import.meta.url))],
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    logLevel: 'warning',
  });
  const js = bundle.outputFiles[0]!.text;
  const serve = (root: string, rel: string): { path: string } | null => {
    const file = normalize(join(root, rel));
    return file.startsWith(root + sep) && existsSync(file) ? { path: file } : null;
  };
  await page.route(`${ORIGIN}/**`, async (route) => {
    const path = decodeURIComponent(new URL(route.request().url()).pathname);
    if (path === '/') {
      return route.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><title>recognizer replay</title><script type="module" src="/page.js"></script>' });
    }
    if (path === '/page.js') return route.fulfill({ contentType: 'text/javascript', body: js });
    const file = path.startsWith('/fx/') ? serve(FX, path.slice(4)) : path.startsWith('/frames/') ? serve(FRAMES, path.slice(8)) : null;
    return file ? route.fulfill(file) : route.fulfill({ status: 404, body: 'not found' });
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${ORIGIN}/`);
  await page.waitForFunction(() => typeof (globalThis as { runReplay?: unknown }).runReplay === 'function');
  const opts = {
    lenient: process.env.RIFTEYE_REPLAY_LENIENT === '1',
    ...(process.env.RIFTEYE_REPLAY_STEPS ? { steps: Number(process.env.RIFTEYE_REPLAY_STEPS) } : {}),
    ...(process.env.RIFTEYE_REPLAY_TOLERANCE ? { tolerance: Number(process.env.RIFTEYE_REPLAY_TOLERANCE) } : {}),
  };
  const out = await page.evaluate(
    ([b, o]) => (globalThis as unknown as { runReplay(b: string, o: object): Promise<PageResult> }).runReplay(b, o),
    [base, opts] as const,
  );
  const d = [...out.decodeMs].sort((a, b) => a - b);
  console.log(`${base}:\n${out.text}\nJPEG decode ms a frame: median ${d[d.length >> 1]?.toFixed(1)}, max ${d[d.length - 1]?.toFixed(1)}`);
  expect(errors).toEqual([]);
  return out;
}

test('the recognizer gives Python\'s state and events on the LA final, in Chromium', async ({ page }) => {
  test.skip(!HAVE, 'needs the private data: RIFTEYE_M3 with fixtures/recognizer and frames/la-final');
  test.setTimeout(1_800_000);
  const out = await run(page, '/fx');
  expect(out.report.firstDivergence).toBeNull();
  expect(out.report.steps).toBe(out.total);
  expect(out.report.identical).toBe(out.total);
});

// recognizer_replay.py --scenario cut: the table, another shot (grey), then the table re-framed: the scene goes away
// and back, the board is cut, found again by name and re-anchored (the LA final never leaves the table camera)
test('the recognizer goes away, comes back re-framed and re-anchors the board as Python does, in Chromium', async ({ page }) => {
  test.skip(!HAVE || !existsSync(join(FX, 'cut', 'meta.json')), 'needs the private data: RIFTEYE_M3 with fixtures/recognizer/cut');
  test.setTimeout(600_000);
  const out = await run(page, '/fx/cut');
  expect(out.report.firstDivergence).toBeNull();
  expect(out.report.identical).toBe(out.total);
});
