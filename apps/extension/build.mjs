// Bundles the extension into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
// This is the public build, which reads with the live runner on this machine. The private build adds what the
// engine inside the extension needs (the models, the gallery, onnxruntime-web's runtime files): pack.mjs.
//
//   node build.mjs [--store | --firefox] [--out DIR]      (default: dist/, next to this file; dist-store/, dist-firefox/)
//
// --store   the Chrome Web Store build (decision D-025), standalone only: the constant `__STORE__` (src/store.d.ts) is true, which
//           takes the companion mode out of the bundles (the worker has no code to reach 127.0.0.1), and the manifest is the
//           store's: no access to 127.0.0.1, and access to the two hosts of Riot's public card gallery, from which the names,
//           types and pictures of the cards are loaded as the viewer watches (src/feed.ts, D-015). Pack it: node pack.mjs --store.
// --firefox the store build for Firefox (decision D-039): the same, with Firefox's manifest (a background page, not a service
//           worker; its sidebar, not Chrome's side panel; no offscreen document, so the engine runs in the background page:
//           src/background-firefox.ts), the bundles calling Firefox's `browser` namespace, the overlay's typefaces at
//           moz-extension:// addresses, and onnxruntime-web's Asyncify WebGPU runtime, which needs no JSPI. Pack it: node
//           pack.mjs --firefox.
import { build } from 'esbuild';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FONTS, fontFaceCss, tokensCss } from '../../assets/brand/brand.mjs';

const here = (p) => new URL(p, import.meta.url).pathname;
const brand = (p) => fileURLToPath(new URL(`../../assets/brand/${p}`, import.meta.url));
const USAGE = 'usage: node build.mjs [--store | --firefox] [--out DIR]';
// Riot's public card gallery: the card list (src/feed.ts FEED_URL) and the pictures (IMAGE_ORIGIN); a test holds these to those
const RIOT_HOSTS = ['https://content.publishing.riotgames.com/*', 'https://cmsassets.rgpub.io/*'];
// Firefox: the add-on's id (it never changes once published), and the oldest Firefox it is built for: 141, the first with WebGPU
const FIREFOX_ID = 'wardeye@effe-exe.github.io';
const FIREFOX_MIN = '141.0';

let store = false;
let firefox = false;
let out = null;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--store') store = true;
  else if (args[i] === '--firefox') firefox = store = true; // Firefox's is a store build: no companion mode
  else if (args[i] === '--out' && args[i + 1]) out = args[++i];
  else throw new Error(USAGE);
}
const folder = firefox ? 'dist-firefox' : store ? 'dist-store' : 'dist';
const dist = out ? resolve(out) : here(`./${folder}`);
mkdirSync(dist, { recursive: true });
// `define` sets __STORE__. The worker and the engine document are the two files that read it, and are built with `minifySyntax`, which
// folds the constant and drops the branch that is not taken (the companion client, in the store build); it renames nothing and
// squeezes no whitespace.
const define = { __STORE__: String(store), __FIREFOX__: String(firefox) };
// Firefox's `browser` namespace has every API the extension uses, each returning a promise: in its build, `chrome` is `browser`
// (`globalThis.browser`, so that no local name in the code can stand in for it)
const namespace = firefox ? { define: { chrome: 'globalThis.browser' } } : {};
// content.js runs once in a page. The toolbar button can inject it into a YouTube tab (src/access.ts), and a second copy in a
// page that already has one would draw a second overlay: the bundle runs only where no copy has run before.
const once = { banner: { js: 'if (!globalThis.__wardeye) { globalThis.__wardeye = true;' }, footer: { js: '}' } };
const entries = firefox
  ? [['content.ts', 'content.js', 'iife', false], ['background-firefox.ts', 'worker.js', 'esm', true], ['panel.ts', 'panel.js', 'esm', false]]
  : [['content.ts', 'content.js', 'iife', false], ['worker.ts', 'worker.js', 'esm', true], ['offscreen.ts', 'offscreen.js', 'esm', true], ['panel.ts', 'panel.js', 'esm', false]];
for (const [entry, file, format, reads] of entries) {
  await build({
    entryPoints: [here(`./src/${entry}`)],
    bundle: true,
    format,
    target: 'es2022',
    outfile: join(dist, file),
    legalComments: 'inline',
    ...(reads ? { minifySyntax: true } : {}),
    define: { ...(reads ? define : {}), ...namespace.define },
    ...(entry === 'content.ts' ? once : {}),
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
  // Firefox has no WebAssembly JSPI (or only lately): its WebGPU worker takes onnxruntime-web's Asyncify build of the same runtime
  ...(firefox ? { alias: { 'onnxruntime-web/jspi': 'onnxruntime-web/webgpu' } } : {}),
});
if (!firefox) copyFileSync(here('./src/offscreen.html'), join(dist, 'offscreen.html'));
copyFileSync(here('./src/panel.html'), join(dist, 'panel.html'));

// The manifest. The store's is the developer's less its access to 127.0.0.1 (companion mode), plus access to Riot's card gallery.
// Firefox's is the store's in Firefox's terms (firefoxManifest).
const manifest = JSON.parse(readFileSync(here('./src/manifest.json'), 'utf8'));
if (store) {
  manifest.host_permissions = [...(manifest.host_permissions ?? []).filter((h) => h !== 'http://127.0.0.1/*'), ...RIOT_HOSTS];
  if (JSON.stringify(manifest).includes('127.0.0.1')) throw new Error('the store manifest must not mention 127.0.0.1');
  writeFileSync(join(dist, 'manifest.json'), `${JSON.stringify(firefox ? firefoxManifest(manifest) : manifest, null, 2)}\n`);
} else {
  copyFileSync(here('./src/manifest.json'), join(dist, 'manifest.json'));
}

// The overlay's stylesheet: the brand's tokens on the overlay's own root element (so nothing reaches Twitch's styles), then its
// typefaces, then the overlay's rules. The typefaces are the extension's own copies (dist/fonts/, below); Chrome fills in the
// extension's id in a content script's CSS, and the manifest makes them web accessible to twitch.tv and nothing else.
const fontUrl = (file) => `${firefox ? 'moz-extension' : 'chrome-extension'}://__MSG_@@extension_id__/${file}`;
writeFileSync(
  join(dist, 'overlay.css'),
  '/* Wardeye brand (assets/brand): the tokens, and Space Grotesk, Inter and JetBrains Mono under the SIL Open Font License 1.1 */\n' +
    `${tokensCss('.rifteye-root')}\n${fontFaceCss(fontUrl)}\n${readFileSync(here('./src/overlay.css'), 'utf8')}`,
);
// The plays panel's stylesheet: it is the extension's own page, so the tokens go on :root and the typefaces are the files beside it.
writeFileSync(
  join(dist, 'panel.css'),
  '/* Wardeye brand (assets/brand): the tokens, and Space Grotesk, Inter and JetBrains Mono under the SIL Open Font License 1.1 */\n' +
    `${tokensCss(':root')}\n${fontFaceCss((file) => file)}\n${readFileSync(here('./src/panel.css'), 'utf8')}`,
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
console.log(`${firefox ? 'Firefox build' : store ? 'Chrome Web Store build' : 'extension built'} -> ${out ? dist : `apps/extension/${folder}/`}`);

/** The store's manifest, for Firefox: a background page (Firefox runs `scripts`, not a service worker), its sidebar for the
 * plays panel, no offscreen document or side panel, no cross-origin isolation keys (Firefox does not isolate extension pages),
 * and its own settings: the add-on's id, the oldest Firefox it runs on, and that it collects no data (D-039). */
function firefoxManifest(m) {
  const { side_panel: _panel, cross_origin_embedder_policy: _coep, cross_origin_opener_policy: _coop, ...rest } = m;
  return {
    ...rest,
    background: { scripts: ['worker.js'], type: 'module' },
    permissions: m.permissions.filter((p) => p !== 'offscreen' && p !== 'sidePanel'),
    sidebar_action: { default_panel: 'panel.html', default_title: 'Wardeye', default_icon: m.action.default_icon, open_at_install: false },
    browser_specific_settings: { gecko: { id: FIREFOX_ID, strict_min_version: FIREFOX_MIN, data_collection_permissions: { required: ['none'] } } },
  };
}
