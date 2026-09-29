// Bundles the logger into dist/: a static page that works from file:// or any static host.
import { build } from 'esbuild';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { withBrand } from '../../assets/brand/brand.mjs';

const here = (p) => new URL(p, import.meta.url).pathname;
mkdirSync(here('./dist'), { recursive: true });
await build({
  entryPoints: [here('./src/main.ts')],
  bundle: true,
  format: 'iife',
  target: 'es2022',
  outfile: here('./dist/app.js'),
  sourcemap: true,
  legalComments: 'inline',
});
copyFileSync(here('./src/index.html'), here('./dist/index.html'));
writeFileSync(here('./dist/style.css'), withBrand(readFileSync(here('./src/style.css'), 'utf8'))); // Wardeye's brand
copyFileSync(here('../../assets/brand/logo/mark.svg'), here('./dist/mark.svg')); // the mark in the page header
console.log('logger built -> apps/logger/dist/');
