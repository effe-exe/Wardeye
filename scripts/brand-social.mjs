// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
// Renders the brand book's social images into a folder (they are uploaded, not committed):
//   social-preview.png  1280 x 640, GitHub's social preview: the lockup and the one-liner on the dark grid
//   avatar-dark.png     512 x 512, the mark on the background
//   avatar-primary.png  512 x 512, the white mark on the primary
//   node scripts/brand-social.mjs OUT_DIR
// Chromium comes from Playwright; set RIFTEYE_CHROMIUM to use another build.
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from '@playwright/test';
import { fontFaceCss, tokensCss } from '../assets/brand/brand.mjs';

const out = resolve(process.argv[2] ?? 'social');
mkdirSync(out, { recursive: true });
const svg = (name) => readFileSync(new URL(`../assets/brand/logo/${name}.svg`, import.meta.url), 'utf8');
const sized = (s, h) => s.replace(/ width="[^"]+" height="[^"]+"/, ` height="${h}"`);
const page = (body, w, h, bg = 'var(--wd-bg)') =>
  `<!doctype html><html><head><style>${fontFaceCss()}\n${tokensCss()}
  html, body { margin: 0; width: ${w}px; height: ${h}px; overflow: hidden; }
  body { background: ${bg}; color: var(--wd-text); font-family: var(--wd-font-ui); display: flex; flex-direction: column;
         align-items: center; justify-content: center; }
  .grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px); background-size: 40px 40px; }
  .stack { position: relative; display: flex; flex-direction: column; align-items: center; }
  </style></head><body>${body}</body></html>`;

const images = [
  ['social-preview', 1280, 640, page(`<div class="grid"></div><div class="stack">
    ${sized(svg('lockup'), 132)}
    <div style="margin-top: 36px; font: 500 38px/1.2 var(--wd-font-ui); letter-spacing: -0.01em">Place the ward. See the table.</div>
    <div style="margin-top: 14px; font: 400 23px/1.3 var(--wd-font-ui); color: var(--wd-muted)">Open-source computer vision for Riftbound streams</div>
    <div style="margin-top: 34px; width: 128px; height: 3px; border-radius: 2px; background: var(--wd-primary)"></div>
    <div style="margin-top: 30px; font: 500 16px/1 var(--wd-font-ui); color: var(--wd-dim); letter-spacing: 0.02em">
      Alpha · A community project by Federico Vietti</div></div>`, 1280, 640)],
  ['avatar-dark', 512, 512, page(sized(svg('mark'), 360), 512, 512)],
  ['avatar-primary', 512, 512, page(sized(svg('mark-white'), 360), 512, 512, 'var(--wd-primary)')],
];

const exe = process.env.RIFTEYE_CHROMIUM;
const browser = await chromium.launch(exe ? { executablePath: exe } : {});
try {
  const tab = await browser.newPage();
  for (const [name, w, h, html] of images) {
    await tab.setViewportSize({ width: w, height: h });
    await tab.setContent(html);
    await tab.evaluate(() => document.fonts.ready);
    await tab.screenshot({ path: `${out}/${name}.png` });
    console.log(`${out}/${name}.png`);
  }
} finally {
  await browser.close();
}
