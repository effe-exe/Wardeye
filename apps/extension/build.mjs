// Bundles the extension into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
// This is the public build, which reads with the live runner on this machine. The private build adds what the
// engine inside the extension needs (the models, the gallery, onnxruntime-web's runtime files): pack.mjs.
import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'node:fs';

const here = (p) => new URL(p, import.meta.url).pathname;
mkdirSync(here('./dist'), { recursive: true });
for (const [entry, out, format] of [['content.ts', 'content.js', 'iife'], ['worker.ts', 'worker.js', 'esm'], ['offscreen.ts', 'offscreen.js', 'esm']]) {
  await build({
    entryPoints: [here(`./src/${entry}`)],
    bundle: true,
    format,
    target: 'es2022',
    outfile: here(`./dist/${out}`),
    legalComments: 'inline',
  });
}
// the engine's workers, one for each onnxruntime-web build; "extern wasm" leaves the build's .mjs and .wasm to be
// loaded from ort/ (pack.mjs puts them there): an extension may not load remote code
await build({
  entryPoints: ['engine-webgpu', 'engine-wasm'].map((n) => here(`./src/${n}.ts`)),
  bundle: true,
  format: 'esm',
  target: 'es2022',
  outdir: here('./dist'),
  conditions: ['onnxruntime-web-use-extern-wasm'],
  legalComments: 'inline',
  logLevel: 'warning',
});
for (const f of ['manifest.json', 'overlay.css', 'offscreen.html']) copyFileSync(here(`./src/${f}`), here(`./dist/${f}`));
console.log('extension built -> apps/extension/dist/');
