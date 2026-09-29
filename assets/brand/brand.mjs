// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
// Wardeye's brand for the builds: the design tokens (tokens.css) and the three typefaces, so every page and the
// extension's overlay look the same (README.md here). The fonts are under the SIL Open Font License 1.1
// (fonts/OFL-*.txt), not the AGPL.
import { readFileSync } from 'node:fs';

const here = (p) => new URL(p, import.meta.url);

/** The typefaces: Latin subsets of the variable fonts, as WOFF2, with their weight ranges. */
export const FONTS = [
  { family: 'Space Grotesk', file: 'fonts/SpaceGrotesk-latin.woff2', weight: '300 700' },
  { family: 'Inter', file: 'fonts/Inter-latin.woff2', weight: '100 900' },
  { family: 'JetBrains Mono', file: 'fonts/JetBrainsMono-latin.woff2', weight: '100 800' },
];

/** The design tokens on `selector`: ':root' for a page, the overlay's root element for the extension. */
export function tokensCss(selector = ':root') {
  const css = readFileSync(here('tokens.css'), 'utf8');
  return selector === ':root' ? css : css.replace(/^:root \{/m, `${selector} {`);
}

/**
 * @font-face rules for the three typefaces. By default the fonts are inlined as data: URLs, so a page works from
 * file:// too (Chrome loads no font from a file:// URL). Pass `url` to point at the font files instead, as the
 * extension does with its own copies.
 */
export function fontFaceCss(url) {
  return FONTS.map(({ family, file, weight }) => {
    const src = url ? url(file) : `data:font/woff2;base64,${readFileSync(here(file)).toString('base64')}`;
    return `@font-face { font-family: '${family}'; font-style: normal; font-weight: ${weight}; font-display: swap; ` +
      `src: url(${src}) format('woff2'); }`;
  }).join('\n');
}

/** A page's stylesheet with the brand in front (the fonts inlined, the tokens on :root): what a build writes. */
export function withBrand(css) {
  return '/* Wardeye brand (assets/brand). Space Grotesk, Inter and JetBrains Mono: SIL Open Font License 1.1 */\n' +
    `${fontFaceCss()}\n${tokensCss()}\n${css}`;
}
