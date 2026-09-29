// Bundles the extension into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
// This is the public build, which reads with the live runner on this machine. The private build adds what the
// engine inside the extension needs (the models, the gallery, onnxruntime-web's runtime files): pack.mjs.
//
//   node build.mjs [--store] [--out DIR]      (default: dist/, next to this file; with --store, dist-store/)
//
// --store   the Chrome Web Store build (decision D-025), standalone only: the constant `__STORE__` (src/store.d.ts) is true, which
//           takes the companion mode out of the bundles (the worker has no code to reach 127.0.0.1), and the manifest is the
//           store's: no access to 127.0.0.1, and access to the two hosts of Riot's public card gallery, from which the names,
//           types and pictures of the cards are loaded as the viewer watches (src/feed.ts, D-015). Pack it: node pack.mjs --store.
import { build } from 'esbuild';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FONTS, fontFaceCss, tokensCss } from '../../assets/brand/brand.mjs';

const here = (p) => new URL(p, import.meta.url).pathname;
const brand = (p) => fileURLToPath(new URL(`../../assets/brand/${p}`, import.meta.url));
const USAGE = 'usage: node build.mjs [--store] [--out DIR]';
// Riot's public card gallery: the card list (src/feed.ts FEED_URL) and the pictures (IMAGE_ORIGIN); a test holds these to those
const RIOT_HOSTS = ['https://content.publishing.riotgames.com/*', 'https://cmsassets.rgpub.io/*'];

let store = false;
let out = null;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--store') store = true;
  else if (args[i] === '--out' && args[i + 1]) out = args[++i];
  else throw new Error(USAGE);
}
const dist = out ? resolve(out) : here(store ? './dist-store' : './dist');
mkdirSync(dist, { recursive: true });
// `define` sets __STORE__. The worker and the engine document are the two files that read it, and are built with `minifySyntax`, which
// folds the constant and drops the branch that is not taken (the companion client, in the store build); it renames nothing and
// squeezes no whitespace.
const define = { __STORE__: String(store) };
for (const [entry, file, format, reads] of [['content.ts', 'content.js', 'iife', false], ['worker.ts', 'worker.js', 'esm', true], ['offscreen.ts', 'offscreen.js', 'esm', true]]) {
  await build({
    entryPoints: [here(`./src/${entry}`)],
    bundle: true,
    format,
    target: 'es2022',
    outfile: join(dist, file),
    legalComments: 'inline',
    ...(reads ? { define, minifySyntax: true } : {}),
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
copyFileSync(here('./src/offscreen.html'), join(dist, 'offscreen.html'));

// The manifest. The store's is the developer's less its access to 127.0.0.1 (companion mode), plus access to Riot's card gallery.
const manifest = JSON.parse(readFileSync(here('./src/manifest.json'), 'utf8'));
if (store) {
  manifest.host_permissions = [...(manifest.host_permissions ?? []).filter((h) => h !== 'http://127.0.0.1/*'), ...RIOT_HOSTS];
  if (JSON.stringify(manifest).includes('127.0.0.1')) throw new Error('the store manifest must not mention 127.0.0.1');
  writeFileSync(join(dist, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
} else {
  copyFileSync(here('./src/manifest.json'), join(dist, 'manifest.json'));
}

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
for (const path of Object.values(manifest.icons)) {
  mkdirSync(dirname(join(dist, path)), { recursive: true });
  copyFileSync(here(`./${path}`), join(dist, path));
}
console.log(`${store ? 'Chrome Web Store build' : 'extension built'} -> ${out ? dist : `apps/extension/${store ? 'dist-store' : 'dist'}/`}`);
