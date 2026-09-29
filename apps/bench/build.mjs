// Bundles the bench into dist/: load it in Chrome with "Load unpacked" (chrome://extensions, Developer mode).
// dist/ort/ gets the onnxruntime-web runtime files the three builds load (the JSPI build for the native WebGPU
// provider, the JSEP build, the plain WASM build). The models are data, not part of dist: pack.mjs adds them.
import { build } from 'esbuild';
import { copyFileSync, linkSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withBrand } from '../../assets/brand/brand.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const require = createRequire(import.meta.url);
const dist = here('./dist');

// the .mjs (worker and loader) and .wasm of each build; src/runtimes.ts says which runtime loads which
const RUNTIME_FILES = ['ort-wasm-simd-threaded.jspi', 'ort-wasm-simd-threaded.jsep', 'ort-wasm-simd-threaded'].flatMap((b) => [
  `${b}.mjs`,
  `${b}.wasm`,
]);
const ortFile = (name) => require.resolve(`onnxruntime-web/${name}`);
const ortVersion = JSON.parse(readFileSync(join(dirname(ortFile('ort-wasm-simd-threaded.wasm')), '..', 'package.json'), 'utf8')).version;
const manifest = JSON.parse(readFileSync(here('./src/manifest.json'), 'utf8'));
const benchVersion = manifest.version;

// The runtime files are big (58 MB): hard-linked into dist when the disk allows, so there is no second copy.
function place(from, to) {
  rmSync(to, { force: true });
  try {
    linkSync(from, to);
  } catch {
    copyFileSync(from, to);
  }
}

mkdirSync(join(dist, 'ort'), { recursive: true });
const common = { bundle: true, format: 'esm', target: 'es2022', outdir: dist, legalComments: 'inline', logLevel: 'warning' };
await build({
  ...common,
  entryPoints: [here('./src/bench.ts'), here('./src/worker.ts')],
  define: { __ORT_VERSION__: JSON.stringify(ortVersion), __BENCH_VERSION__: JSON.stringify(benchVersion) },
});
// each onnxruntime-web build is a bundle of its own; "extern wasm" leaves its .mjs and .wasm to be loaded from ort/
await build({
  ...common,
  entryPoints: ['bench-webgpu', 'bench-jsep', 'bench-wasm'].map((n) => here(`./src/${n}.ts`)),
  conditions: ['onnxruntime-web-use-extern-wasm'],
});
for (const f of ['manifest.json', 'bench.html']) copyFileSync(here(`./src/${f}`), join(dist, f));
// the stylesheet takes Wardeye's brand in front of it: the tokens and the three fonts, inlined (assets/brand/brand.mjs)
writeFileSync(join(dist, 'bench.css'), withBrand(readFileSync(here('./src/bench.css'), 'utf8')));
copyFileSync(here('../../assets/brand/logo/mark.svg'), join(dist, 'mark.svg')); // the mark in the page header
// the icons the manifest names (icons/, made by scripts/brand-icons.mjs), at the same paths in dist/
for (const f of Object.values(manifest.icons ?? {})) {
  mkdirSync(dirname(join(dist, f)), { recursive: true });
  copyFileSync(here(`./${f}`), join(dist, f));
}
const sizes = RUNTIME_FILES.map((f) => {
  place(ortFile(f), join(dist, 'ort', f));
  return `${f} ${(statSync(join(dist, 'ort', f)).size / 1e6).toFixed(2)} MB`;
});
console.log(`bench built (onnxruntime-web ${ortVersion}) -> apps/bench/dist/\n  ${sizes.join('\n  ')}`);
