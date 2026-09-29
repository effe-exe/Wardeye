// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
// Gradeon's typeface for Wardeye's pages (D-020). A build puts these @font-face rules in front of a page's
// stylesheet with the fonts inlined, so the page works from file:// too (Chrome loads no font from a file:// URL).
// The fonts are Space Mono, unmodified, under the SIL Open Font License 1.1 (OFL.txt, next to this file).
import { readFileSync } from 'node:fs';

const FONTS = [
  ['SpaceMono-Regular.ttf', 400],
  ['SpaceMono-Bold.ttf', 700],
];

/** @font-face rules for Space Mono, regular (400) and bold (700), as data: URLs. */
export function spaceMono() {
  return FONTS.map(([file, weight]) => {
    const b64 = readFileSync(new URL(file, import.meta.url)).toString('base64');
    return `@font-face { font-family: 'Space Mono'; font-style: normal; font-weight: ${weight}; font-display: swap; ` +
      `src: url(data:font/ttf;base64,${b64}) format('truetype'); }`;
  }).join('\n');
}

/** A page's stylesheet with Space Mono in front: what a build writes to dist/. */
export function withSpaceMono(css) {
  return '/* Space Mono: Copyright 2016 The Space Mono Project Authors, SIL Open Font License 1.1 (assets/brand/OFL.txt) */\n' +
    `${spaceMono()}\n${css}`;
}
