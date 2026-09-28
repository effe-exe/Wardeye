// Bundles the extension into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'node:fs';

const here = (p) => new URL(p, import.meta.url).pathname;
mkdirSync(here('./dist'), { recursive: true });
for (const [entry, out, format] of [['content.ts', 'content.js', 'iife'], ['worker.ts', 'worker.js', 'esm']]) {
  await build({
    entryPoints: [here(`./src/${entry}`)],
    bundle: true,
    format,
    target: 'es2022',
    outfile: here(`./dist/${out}`),
    legalComments: 'inline',
  });
}
for (const f of ['manifest.json', 'overlay.css']) copyFileSync(here(`./src/${f}`), here(`./dist/${f}`));
console.log('extension built -> apps/extension/dist/');
