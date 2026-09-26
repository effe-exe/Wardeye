// Bundles the reviewer into dist/: a static page that works from file:// or any static host.
import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'node:fs';

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
for (const f of ['index.html', 'style.css']) copyFileSync(here(`./src/${f}`), here(`./dist/${f}`));
console.log('reviewer built -> apps/reviewer/dist/');
