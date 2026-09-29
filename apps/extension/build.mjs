// Bundles the extension into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
// This is the public build, which reads with the live runner on this machine. The private build adds what the
// engine inside the extension needs (the models, the gallery, onnxruntime-web's runtime files): pack.mjs.
//
//   node build.mjs [--out DIR]      (default: dist/, next to this file)
import { build } from 'esbuild';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FONTS, fontFaceCss, tokensCss } from '../../assets/brand/brand.mjs';

const here = (p) => new URL(p, import.meta.url).pathname;
const brand = (p) => fileURLToPath(new URL(`../../assets/brand/${p}`, import.meta.url));
const at = process.argv.indexOf('--out');
if (at >= 0 && !process.argv[at + 1]) throw new Error('usage: node build.mjs [--out DIR]');
const dist = at >= 0 ? resolve(process.argv[at + 1]) : here('./dist');
mkdirSync(dist, { recursive: true });
for (const [entry, out, format] of [['content.ts', 'content.js', 'iife'], ['worker.ts', 'worker.js', 'esm'], ['offscreen.ts', 'offscreen.js', 'esm']]) {
  await build({
    entryPoints: [here(`./src/${entry}`)],
    bundle: true,
    format,
    target: 'es2022',
    outfile: join(dist, out),
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
  outdir: dist,
  conditions: ['onnxruntime-web-use-extern-wasm'],
  legalComments: 'inline',
  logLevel: 'warning',
});
for (const f of ['manifest.json', 'offscreen.html']) copyFileSync(here(`./src/${f}`), join(dist, f));

// The overlay's stylesheet: the brand's tokens on the overlay's own root element (so nothing reaches Twitch's styles), then its
// typefaces, then the overlay's rules. The typefaces are the extension's own copies (dist/fonts/, below); Chrome fills in the
// extension's id in a content script's CSS, and the manifest makes them web accessible to twitch.tv and nothing else.
const fontUrl = (file) => `chrome-extension://__MSG_@@extension_id__/${file}`;
writeFileSync(
  join(dist, 'overlay.css'),
  '/* Wardeye brand (assets/brand): the tokens, and Space Grotesk, Inter and JetBrains Mono under the SIL Open Font License 1.1 */\n' +
    `${tokensCss('.rifteye-root')}\n${fontFaceCss(fontUrl)}\n${readFileSync(here('./src/overlay.css'), 'utf8')}`,
);
// the three typefaces, each with the licence that comes with it (the OFL asks for it wherever the font goes)
mkdirSync(join(dist, 'fonts'), { recursive: true });
for (const { file } of FONTS) copyFileSync(brand(file), join(dist, file));
for (const name of readdirSync(brand('fonts')).filter((n) => /^OFL-.+\.txt$/.test(n))) copyFileSync(brand(`fonts/${name}`), join(dist, 'fonts', name));
// the toolbar icons the manifest names (rendered from the brand mark: scripts/brand-icons.mjs)
const manifest = JSON.parse(readFileSync(here('./src/manifest.json'), 'utf8'));
for (const path of Object.values(manifest.icons)) {
  mkdirSync(dirname(join(dist, path)), { recursive: true });
  copyFileSync(here(`./${path}`), join(dist, path));
}
console.log(`extension built -> ${at >= 0 ? dist : 'apps/extension/dist/'}`);
