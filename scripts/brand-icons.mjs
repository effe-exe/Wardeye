// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
// Renders an extension's icons from the brand mark (assets/brand/logo/mark.svg): square PNGs with the mark centred on
// a transparent background, at Chrome's sizes. Run it again if the mark changes:
//   node scripts/brand-icons.mjs apps/extension/icons
// Chromium comes from Playwright; set RIFTEYE_CHROMIUM to use another build.
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from '@playwright/test';

const out = resolve(process.argv[2] ?? 'icons');
const sizes = (process.argv[3] ?? '16,32,48,128').split(',').map(Number);
const mark = readFileSync(new URL('../assets/brand/logo/mark.svg', import.meta.url), 'utf8');
mkdirSync(out, { recursive: true });

const exe = process.env.RIFTEYE_CHROMIUM;
const browser = await chromium.launch(exe ? { executablePath: exe } : {});
try {
  const page = await browser.newPage();
  for (const size of sizes) {
    // the mark is tall (about 5:9): it fills the icon's height, less a small margin at the larger sizes
    const pad = size >= 48 ? Math.round(size * 0.06) : 0;
    await page.setViewportSize({ width: size, height: size });
    await page.setContent(
      `<!doctype html><html><body style="margin:0;background:transparent;width:${size}px;height:${size}px;` +
        `display:flex;align-items:center;justify-content:center">` +
        mark.replace(/ width="[^"]+" height="[^"]+"/, ` height="${size - 2 * pad}"`) +
        '</body></html>',
    );
    await page.screenshot({ path: `${out}/icon-${size}.png`, omitBackground: true });
    console.log(`${out}/icon-${size}.png`);
  }
} finally {
  await browser.close();
}
