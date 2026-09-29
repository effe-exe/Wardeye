// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The change gate over the LA final's 240 frames and the layout finder on its first seconds, in Chromium: the JPEGs are
// decoded as the extension decodes them (createImageBitmap, a 2D canvas, getImageData), turned into the gate's view with
// image.ts, and the events, times and still tables are compared with what ml/rifteye_ml computed on the same JPEGs
// (test/gen/table_fixtures.py). It needs the private frames and fixtures, so it skips without RIFTEYE_M3:
//
//   RIFTEYE_M3=~/rifteye-data/m3 RIFTEYE_CHROMIUM=/opt/pw-browsers/chromium npx playwright test packages/engine/e2e/table.spec.ts

import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';
import { build } from 'esbuild';
import type { GateSettings } from '../src/changegate';
import type { GateResult, GateRunParams, LayoutAttempt, LayoutResult, TableApi } from './table-page';

const HERE = fileURLToPath(new URL('./', import.meta.url));
const M3 = process.env['RIFTEYE_M3'];
const FX = M3 ? join(M3, 'fixtures', 'table') : '';
const FRAMES = M3 ? join(M3, 'frames', 'la-final') : '';

test.skip(!M3, 'needs the private frames and fixtures: set RIFTEYE_M3 to the folder with frames/ and fixtures/ (~/rifteye-data/m3)');
test.skip(Boolean(M3) && !existsSync(join(FX, 'gate.json')), 'run test/gen/table_fixtures.py first (it writes fixtures/table/)');

const json = <T>(path: string): T => JSON.parse(readFileSync(path, 'utf8')) as T;

/** The page and the frames, served from 127.0.0.1 (a secure context: crypto.subtle is there). */
function serve(bundle: string): Promise<{ server: Server; url: string }> {
  const server = createServer((req, res) => {
    const path = decodeURIComponent((req.url ?? '/').split('?')[0]!);
    if (path === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html' }).end('<!doctype html><meta charset="utf-8"><title>table</title><script src="/table-page.js"></script>');
      return;
    }
    if (path === '/table-page.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript' }).end(bundle);
      return;
    }
    const m = /^\/frames\/(f\d{4}\.jpg)$/.exec(path);
    if (m && existsSync(join(FRAMES, m[1]!))) {
      res.writeHead(200, { 'Content-Type': 'image/jpeg' });
      createReadStream(join(FRAMES, m[1]!)).pipe(res);
      return;
    }
    res.writeHead(404).end();
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${(server.address() as { port: number }).port}/` })));
}

interface GateFixture {
  views: string[];
  times: number[];
  runs: ({ name: string; settings: GateSettings } & Omit<GateResult['runs'][number], 'ms'>)[];
}

interface LayoutFixture {
  first_seconds: (LayoutAttempt & Pick<LayoutResult, 'borders' | 'table_window' | 'auto_layout' | 'card_size' | 'card_size_finder' | 'auto_layout_finder'>)[];
}

test('the gate over 240 frames and the layout finder on the first seconds agree with Python', async ({ page }) => {
  test.setTimeout(10 * 60_000);
  const manifest = json<{ frames: { file: string; t: number; rgb_sha256: string }[] }>(join(FRAMES, 'frames.json'));
  const gate = json<GateFixture>(join(FX, 'gate.json'));
  const bundle = (await build({ entryPoints: [join(HERE, 'table-page.ts')], bundle: true, format: 'iife', target: 'es2022', write: false, logLevel: 'warning' })).outputFiles[0]!.text;
  const { server, url } = await serve(bundle);
  try {
    await page.goto(url);
    await page.waitForFunction(() => 'rifteyeTable' in window);
    const files = manifest.frames.map((f) => f.file);

    // the gate: every frame decoded and viewed once, then run with each set of settings
    const runs: GateRunParams[] = gate.runs.map((r) => ({ settings: r.settings, checkpoints: r.checkpoints.map((c) => c.i) }));
    const got = await page.evaluate(([f, t, r]) => (window as unknown as { rifteyeTable: TableApi }).rifteyeTable.gate(f, t, r), [files, gate.times, runs] as const);
    console.log(`table: decoded and viewed ${files.length} frames in ${(got.decodeMs / 1000).toFixed(1)} s; gate runs ${got.runs.map((r) => `${(r.ms / 1000).toFixed(1)} s`).join(', ')}`);
    expect(got.rgbSha, 'Chromium decodes the JPEGs to the RGB Pillow does').toEqual(manifest.frames.map((f) => f.rgb_sha256));
    expect(got.viewSha, 'image.ts makes the gate\'s view as Pillow does').toEqual(gate.views);
    gate.runs.forEach((want, k) => {
      const run = got.runs[k]!;
      expect(run.events, `${want.name}: events and their times`).toEqual(want.events);
      expect(run.checkpoints, `${want.name}: the still table`).toEqual(want.checkpoints);
      expect(run.last_same, `${want.name}: when each pixel last matched`).toBe(want.last_same);
      expect(run.startup_hand).toBe(want.startup_hand);
      expect(run.off_table).toBe(want.off_table);
      expect(run.mat).toEqual(want.mat);
      expect(run.kinds).toEqual(want.kinds);
    });

    // the layout finder: find_layout's attempts on the first seconds, the detector replaced by what it said
    if (existsSync(join(FX, 'autolayout.json'))) {
      const fx = json<LayoutFixture>(join(FX, 'autolayout.json'));
      const attempts: LayoutAttempt[] = fx.first_seconds.map((a) => ({ frames: a.frames, calls: a.calls }));
      const found = await page.evaluate(([f, a]) => (window as unknown as { rifteyeTable: TableApi }).rifteyeTable.layout(f, a), [files, attempts] as const);
      fx.first_seconds.forEach((want, k) => {
        const r = found[k]!;
        expect(r.error, `attempt ${k}`).toBeNull();
        expect(r.borders).toEqual(want.borders);
        expect(r.table_window).toEqual(want.table_window);
        expect(r.used, 'detector calls').toBe(want.calls.length);
        expect(r.auto_layout).toEqual(want.auto_layout);
        expect(r.card_size).toBe(want.card_size ?? null);
        // without the detector: the bootstrap finder (its rectangles are fitted with the browser's sin and cos: 1e-9)
        expect(r.card_size_finder).toBeCloseTo(want.card_size_finder!, 9);
        expect(r.auto_layout_finder).toEqual(want.auto_layout_finder);
      });
    }
  } finally {
    server.close();
  }
});
