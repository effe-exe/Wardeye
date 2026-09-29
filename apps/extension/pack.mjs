// Packs the private standalone build of the extension: the folder rifteye-standalone/, with dist/ (the extension),
// ort/ (onnxruntime-web's runtime files: the native WebGPU build with JSPI, and the plain WASM build), models/ (the
// detector and the embedder), data/ (the gallery, the catalogue and the hover pictures, from ml/rifteye_ml/web_assets.py)
// and standalone.json, which tells the extension what it holds. It is streamed into parts, so the whole zip is never on
// disk:
//
//   zip -r - rifteye-standalone | split -b 30800000 - OUT.zip.part-
//
// Whoever gets the parts joins them (`cat OUT.zip.part-* > OUT.zip`), unzips, opens chrome://extensions, turns on
// Developer mode, clicks "Load unpacked" and picks the rifteye-standalone folder.
//
//   node pack.mjs --out OUT.zip [--models DIR] [--assets DIR] [--precisions detector:fp32,embedder:fp16] [--size-only]
//                 [--part-bytes N] [--runtime auto|webgpu|wasm|companion] [--layout NAME]
//                 [--dist DIR] [--ort DIR] [--stage DIR] [--no-verify]
//   node pack.mjs --store --out OUT.zip [--models DIR] [--assets DIR] [--precisions ...] [--dist DIR] [--ort DIR] [--stage DIR]
//
// --out         the parts are OUT.part-aa, OUT.part-ab, ...; OUT ends in .zip
// --models      where the ONNX files are (default $RIFTEYE_DATA/models/onnx)
// --assets      where web_assets.py wrote the gallery, the catalogue and the thumbnails (default
//               $RIFTEYE_DATA/m3/web-assets)
// --precisions  which file of each model goes in: model:precision, joined by + for several (default
//               detector:fp32,embedder:fp16). The detector runs in float32 (its float16 file fails on native WebGPU) and the
//               embedder in float16 (cosine 0.9998 to its float32 file); a GPU without shader-f16 needs embedder:fp32+fp16.
//               A bare list (fp32,fp16) applies to both models.
// --part-bytes  the size of a part (default 30800000, a little under 30 MiB)
// --size-only   builds nothing: streams the zip through a byte counter and prints what it would be
// --no-verify   skips the check of the parts (their sha256 and every file of the joined zip)
// --stage       where the staging folder of symlinks goes (default: a folder of its own under the temp folder)
// --store       the Chrome Web Store zip (decision D-025) instead of the private build: ONE zip whose root is the extension
//               (manifest.json at the root), from the store build (npm run build:store -w @rifteye/extension: --dist defaults to
//               dist-store/, and is refused when it is not the store's). It holds the extension, ort/, models/ (the detector in
//               float32, the embedder in float16), data/gallery/ (the embedding gallery, keyed by printing id), standalone.json,
//               LICENSE, NOTICE, and the onnxruntime notices; and NO data/catalog.json, no data/thumbs, no card picture, name or
//               text: those load from Riot's public card gallery as the viewer watches (D-015). The zip is checked (unzip -t, and
//               that its files are what was staged and nothing that must not be there) and its size printed.
//
// Nothing big is copied: the staging folder holds symlinks, which `zip` follows. The models, the gallery and the
// thumbnails are made from Riot's card art and stay private (decision D-006): they go only into this zip (the store's
// has the models and the gallery, whose vectors are keyed by printing id, and never a thumbnail).
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, renameSync, rmdirSync, rmSync, statSync, symlinkSync,
  unlinkSync, writeFileSync,
} from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Transform, Writable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const TOP = 'rifteye-standalone';
const DIST_FILES = [
  'manifest.json', 'content.js', 'overlay.css', 'worker.js', 'offscreen.html', 'offscreen.js', 'engine-webgpu.js', 'engine-wasm.js',
  'panel.html', 'panel.js', 'panel.css', // the plays panel, in the browser's side panel
  // what the manifest names besides: the toolbar icons, and the overlay's typefaces with their licences (SIL OFL 1.1)
  'icons/icon-16.png', 'icons/icon-32.png', 'icons/icon-48.png', 'icons/icon-128.png',
  'fonts/SpaceGrotesk-latin.woff2', 'fonts/Inter-latin.woff2', 'fonts/JetBrainsMono-latin.woff2',
  'fonts/OFL-SpaceGrotesk.txt', 'fonts/OFL-Inter.txt', 'fonts/OFL-JetBrainsMono.txt',
];
// onnxruntime-web's runtime: the native WebGPU build (JSPI) and the plain WASM build; not JSEP (broken for GridSample in fp16)
const ORT_FILES = ['ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const PORTABLE = /^[A-Za-z0-9._-]+$/; // a file name that every system unzips
const MODELS = ['detector-v0', 'embedder-v1'];
// the store's manifest: access to Riot's public card gallery and nothing else (build.mjs --store writes it)
const RIOT_HOSTS = ['https://content.publishing.riotgames.com/*', 'https://cmsassets.rgpub.io/*'];
const USAGE =
  'usage: node pack.mjs --out FILE.zip [--models DIR] [--assets DIR] [--precisions detector:fp32,embedder:fp16] [--size-only] [--part-bytes N] ' +
  '[--runtime auto|webgpu|wasm|companion] [--layout NAME] [--dist DIR] [--ort DIR] [--stage DIR] [--no-verify]\n' +
  '   or: node pack.mjs --store --out FILE.zip [--models DIR] [--assets DIR] [--precisions ...] [--dist DIR] [--ort DIR] [--stage DIR]';

class PackError extends Error {}
const fail = (message) => {
  throw new PackError(message);
};

/** --precisions: "detector:fp32,embedder:fp16" or "embedder:fp32+fp16", or a bare list ("fp32,fp16") for both models. */
function parsePrecisions(text) {
  const bad = () => fail('--precisions is model:precision pairs (detector:fp32,embedder:fp16, several with +), or a list for both (fp32,fp16)');
  const known = (p) => (p === 'fp16' || p === 'fp32' ? p : bad());
  const tokens = text.split(',').map((t) => t.trim().toLowerCase()).filter(Boolean);
  if (tokens.length === 0) bad();
  const out = {};
  if (tokens.every((t) => !t.includes(':'))) {
    for (const id of MODELS) out[id] = [...new Set(tokens.map(known))];
    return out;
  }
  for (const t of tokens) {
    const [name, list] = t.split(':');
    const id = MODELS.find((m) => m === name || m.split('-')[0] === name) ?? bad();
    out[id] = [...new Set((list ?? '').split('+').map(known))];
  }
  if (MODELS.some((id) => !out[id])) bad();
  return out;
}

function parseArgs(argv) {
  // no default place: the private folder is named (RIFTEYE_DATA) or its parts are
  const data = process.env.RIFTEYE_DATA || null;
  const o = {
    out: null, models: data && join(data, 'models', 'onnx'), assets: data && join(data, 'm3', 'web-assets'), precisions: parsePrecisions('detector:fp32,embedder:fp16'), sizeOnly: false, partBytes: 30_800_000,
    runtime: 'auto', layout: null, dist: null, ort: null, stage: null, verify: true, store: false,
  };
  const given = new Set();
  for (let i = 0; i < argv.length; i++) {
    const [flag, inline] = argv[i].startsWith('--') ? argv[i].split(/=(.*)/s, 2) : [argv[i]];
    given.add(flag);
    const value = () => inline ?? argv[++i] ?? fail(USAGE);
    if (flag === '--store') o.store = true;
    else if (flag === '--out') o.out = value();
    else if (flag === '--models') o.models = value();
    else if (flag === '--assets') o.assets = value();
    else if (flag === '--precisions') o.precisions = parsePrecisions(value());
    else if (flag === '--part-bytes') o.partBytes = Number(value());
    else if (flag === '--runtime') o.runtime = value();
    else if (flag === '--layout') o.layout = value();
    else if (flag === '--dist') o.dist = value();
    else if (flag === '--ort') o.ort = value();
    else if (flag === '--stage') o.stage = value();
    else if (flag === '--size-only') o.sizeOnly = true;
    else if (flag === '--no-verify') o.verify = false;
    else fail(USAGE);
  }
  if (!['auto', 'webgpu', 'wasm', 'companion'].includes(o.runtime)) fail('--runtime is auto, webgpu, wasm or companion');
  if (!Number.isInteger(o.partBytes) || o.partBytes < 1000) fail('--part-bytes must be a whole number of at least 1000');
  if (o.store) {
    // one zip, made the way the store takes it: nothing of the private build's parts, and its runtime is the browser's to choose
    for (const flag of ['--part-bytes', '--size-only', '--runtime', '--layout', '--no-verify']) if (given.has(flag)) fail(`${flag} does not apply to --store (one zip, always checked, its runtime is auto)`);
    if (!o.out) fail(USAGE);
  } else if (!o.sizeOnly && !o.out) fail(USAGE);
  o.dist ??= here(o.store ? './dist-store' : './dist');
  if (!o.models) fail('--models: name the folder with the ONNX files, or set RIFTEYE_DATA to the private data folder');
  if (!o.assets) fail('--assets: name the folder web_assets.py wrote, or set RIFTEYE_DATA to the private data folder');
  return o;
}

const mb = (n) => `${(n / 1e6).toFixed(1)} MB`;
const mib = (n) => `${(n / 1048576).toFixed(2)} MiB`;

/** Every file under `dir`, as paths relative to it. */
function listFiles(dir, prefix = '') {
  return readdirSync(join(dir, prefix), { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? listFiles(dir, join(prefix, e.name)) : [join(prefix, e.name)],
  );
}

function sha256File(file) {
  return new Promise((done, reject) => {
    const h = createHash('sha256');
    createReadStream(file).on('data', (c) => h.update(c)).on('end', () => done(h.digest('hex'))).on('error', reject);
  });
}

/** Where onnxruntime-web's runtime files are: --ort, else the installed package's dist/. */
function ortDir(given) {
  if (given) return resolve(given);
  try {
    return dirname(createRequire(import.meta.url).resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'));
  } catch {
    return fail('onnxruntime-web is not installed (npm install), and --ort names no folder');
  }
}

/** What the package holds of the models: each one's files for the chosen precisions, checked to be there. */
function chooseModels(modelsDir, precisions) {
  if (!existsSync(modelsDir)) fail(`the models folder ${modelsDir} does not exist`);
  const out = {};
  for (const id of MODELS) {
    const files = {};
    for (const p of precisions[id]) {
      const name = p === 'fp16' ? `${id}.fp16.onnx` : `${id}.onnx`;
      if (!existsSync(join(modelsDir, name))) fail(`${name} (${p}) is not in ${modelsDir}`);
      files[p] = name;
    }
    out[id] = files;
  }
  return out;
}

/** The gallery, the catalogue and the thumbnails, read and checked against each other and against the models. The store build has
 * neither catalogue nor thumbnails (Riot's card gallery gives names, types and pictures as the viewer watches): with `store`,
 * only the gallery, and the models it was made with, are read. */
async function checkAssets(assetsDir, modelsDir, models, store = false) {
  const at = (...p) => join(assetsDir, ...p);
  for (const f of store ? ['gallery/index.json'] : ['gallery/index.json', 'catalog.json', 'thumbs']) if (!existsSync(at(f))) fail(`${at(f)} is missing: run python -m rifteye_ml.web_assets first`);
  let index;
  let catalog = [];
  try {
    index = JSON.parse(readFileSync(at('gallery', 'index.json'), 'utf8'));
    if (!store) catalog = JSON.parse(readFileSync(at('catalog.json'), 'utf8'));
  } catch (e) {
    return fail(`the gallery's index or the catalogue cannot be read (${e.message})`);
  }
  if (index.dtype !== 'float16') fail(`the gallery is ${index.dtype}: the package ships float16 (python -m rifteye_ml.web_assets writes it)`);
  if (index.format !== 1 || !Array.isArray(index.levels) || !Array.isArray(index.rows) || !Array.isArray(catalog)) fail('the gallery index is not what this extension reads (format 1)');
  if (!store && index.rows.length !== catalog.length) fail(`the gallery has ${index.rows.length} rows and the catalogue ${catalog.length}`);
  const missing = index.levels.filter((l) => !existsSync(at('gallery', `L${l}.bin`)));
  if (missing.length) fail(`the gallery's levels ${missing.join(', ')} have no .bin file`);
  const want = index.rows.length * index.dim * 2;
  const wrong = index.levels.filter((l) => statSync(at('gallery', `L${l}.bin`)).size !== want);
  if (wrong.length) fail(`the gallery's levels ${wrong.join(', ')} are not ${want} bytes (${index.rows.length} rows of ${index.dim} float16)`);
  const thumbs = store ? [] : readdirSync(at('thumbs')).filter((n) => n.endsWith('.jpg'));
  if (!store && thumbs.length < catalog.length) fail(`there are ${thumbs.length} thumbnails for ${catalog.length} printings`);
  // the embedder the gallery was made with is the one that goes in: a gallery for another model names cards wrongly
  const embedder = models['embedder-v1'];
  const checks = [['fp32', index.sha256], ['fp16', index.fp16_sha256]];
  for (const [p, sha] of checks) {
    if (!embedder[p]) continue;
    if (!sha) fail(`the gallery's index has no hash of the ${p} embedder: run python -m rifteye_ml.web_assets with ${embedder[p]} in the models folder`);
    const got = await sha256File(join(modelsDir, embedder[p]));
    if (got !== sha) fail(`${embedder[p]} (sha256 ${got.slice(0, 12)}) is not the ${p} embedder the gallery was made with (${sha.slice(0, 12)}): run web_assets.py again`);
  }
  return { index, catalog, thumbs };
}

/** standalone.json: what the extension is told about the package. */
function standaloneJson(o, models) {
  const files = (id) => ({ id, ...Object.fromEntries(Object.entries(models[id]).map(([p, name]) => [p, `models/${name}`])) });
  return `${JSON.stringify({
    format: 1, runtime: o.runtime, detector: files('detector-v0'), embedder: files('embedder-v1'), data: 'data/', ...(o.layout ? { layout: o.layout } : {}),
  }, null, 2)}\n`;
}

const INSTALL = `Wardeye, standalone build (private)

1. Unzip this file. You get one folder, rifteye-standalone.
2. In Chrome (137 or newer), open chrome://extensions and turn on Developer mode.
3. Click "Load unpacked" and pick the rifteye-standalone folder.
4. Play a Riftbound replay or stream on twitch.tv. The badge on the player says what Wardeye is doing.

The models and data in this folder are made from Riot's card art: keep them to yourself (decision D-006).
Nothing leaves your computer. If your browser has no WebGPU, Wardeye uses the live runner on this machine instead.
`;

/** Runs zip in `cwd` and streams its output through `tap` into `sink` (a Writable). */
async function streamZip(cwd, tap, sink) {
  const zip = spawn('zip', ['-X', '-r', '-q', '-', TOP], { cwd, stdio: ['ignore', 'pipe', 'inherit'] });
  const done = new Promise((resolveExit, reject) => {
    zip.on('error', (e) => reject(e.code === 'ENOENT' ? new PackError('the zip command is not installed') : e));
    zip.on('close', (code) => (code === 0 ? resolveExit() : reject(new PackError(`zip exited with code ${code}`))));
  });
  await Promise.all([pipeline(zip.stdout, tap, sink), done]);
}

/** `split` writing the parts: a Writable that feeds it, and a promise for its exit. */
function splitter(prefix, partBytes) {
  const child = spawn('split', ['-b', String(partBytes), '-', prefix], { stdio: ['pipe', 'ignore', 'inherit'] });
  const exited = new Promise((resolveExit, reject) => {
    child.on('error', (e) => reject(e.code === 'ENOENT' ? new PackError('the split command is not installed') : e));
    child.on('close', (code) => (code === 0 ? resolveExit() : reject(new PackError(`split exited with code ${code}`))));
  });
  return { sink: child.stdin, exited };
}

/** The joined parts read by Python's zipfile through a seekable view of them: every file's CRC checked, the whole never on disk. */
const VERIFY = `
import sys, zipfile
class Joined:
    def __init__(self, paths):
        self.fs = [open(p, "rb") for p in paths]
        self.sizes = [f.seek(0, 2) for f in self.fs]
        self.total = sum(self.sizes)
        self.pos = 0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos
    def read(self, n=-1):
        n = self.total - self.pos if n is None or n < 0 else max(0, min(n, self.total - self.pos))
        out, base = [], 0
        for f, size in zip(self.fs, self.sizes):
            lo, hi = max(self.pos, base), min(self.pos + n, base + size)
            if lo < hi:
                f.seek(lo - base)
                out.append(f.read(hi - lo))
            base += size
        data = b"".join(out)
        self.pos += len(data)
        return data
zf = zipfile.ZipFile(Joined(sys.argv[1:]))
bad = zf.testzip()
if bad: sys.exit("bad file in the zip: " + bad)
print(len(zf.namelist()))
`;

async function verifyParts(parts, expected) {
  const h = createHash('sha256');
  let bytes = 0;
  for (const p of parts) {
    await pipeline(createReadStream(p), new Writable({ write(c, _e, cb) { h.update(c); bytes += c.length; cb(); } }));
  }
  const sha = h.digest('hex');
  if (bytes !== expected.bytes || sha !== expected.sha) fail(`the parts hold ${bytes} bytes (sha256 ${sha.slice(0, 12)}), the zip streamed ${expected.bytes} (${expected.sha.slice(0, 12)})`);
  const py = spawnSync('python3', ['-c', VERIFY, ...parts], { encoding: 'utf8' });
  if (py.error) return { sha, files: null, note: 'python3 is not installed: the zip structure was not checked' };
  if (py.status !== 0) fail(`the joined parts are not a good zip: ${py.stderr.trim().split('\n').pop()}`);
  return { sha, files: Number(py.stdout.trim()), note: '' };
}

/** A staging folder under `stageRoot`: symlinks to what goes in, and files written; all of it noted in `made`, for the clean-up. */
function stager(stageRoot, made) {
  const st = { unpacked: 0 };
  const mkdir = (d) => {
    mkdirSync(d, { recursive: true });
    for (let p = d; p.length > stageRoot.length && !made.dirs.includes(p); p = dirname(p)) made.dirs.push(p);
  };
  st.link = (from, to) => {
    mkdir(dirname(to));
    symlinkSync(from, to);
    made.files.push(to);
    st.unpacked += statSync(from).size;
  };
  st.write = (to, text) => {
    mkdir(dirname(to));
    writeFileSync(to, text);
    made.files.push(to);
    st.unpacked += Buffer.byteLength(text);
  };
  return st;
}

/** The store's manifest, checked: access to Riot's public card gallery (its two hosts) and to nothing else, never to 127.0.0.1. */
function checkStoreManifest(manifest, where) {
  const hosts = JSON.stringify(manifest.host_permissions ?? []);
  if (JSON.stringify(manifest).includes('127.0.0.1')) fail(`${where} mentions 127.0.0.1: it is not the store build (npm run build:store -w @rifteye/extension)`);
  if (hosts !== JSON.stringify(RIOT_HOSTS)) fail(`${where}: host_permissions are ${hosts}, the store's are ${JSON.stringify(RIOT_HOSTS)} (npm run build:store -w @rifteye/extension)`);
}

/** `unzip` on the zip; its output is text. */
function unzip(args) {
  const r = spawnSync('unzip', args, { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
  if (r.error) fail(r.error.code === 'ENOENT' ? 'the unzip command is not installed (it checks the zip)' : r.error.message);
  return r;
}

/** What the store zip must hold: by a name that says what it is. */
const STORE_REQUIRED = ['manifest.json', 'standalone.json', 'LICENSE', 'NOTICE', 'models/detector-v0.onnx', 'models/embedder-v1.fp16.onnx', 'data/gallery/index.json'];
/** What it must not: the catalogue and the thumbnails of the private build, and any picture (the toolbar icons are the extension's own). */
const storeForbidden = (n) => n === 'data/catalog.json' || n.startsWith('data/thumbs/') || /\.(jpe?g|webp|gif|bmp|tiff?|avif|heic)$/i.test(n) || (/\.png$/i.test(n) && !n.startsWith('icons/')) || n === 'INSTALL.txt';

/** The store zip, checked: every entry passes its CRC, its entries are what was staged and no more, manifest.json is at its root,
 * nothing that must not be there is, and its manifest is the store's. Returns what it holds. */
function verifyStoreZip(file, staged) {
  const t = unzip(['-tq', file]);
  if (t.status !== 0) fail(`the zip does not pass unzip -t: ${`${t.stdout}${t.stderr}`.trim().split('\n').pop()}`);
  const names = unzip(['-Z1', file]).stdout.split('\n').filter(Boolean);
  const want = new Set(staged);
  const extra = names.filter((n) => !want.has(n));
  const lost = staged.filter((n) => !names.includes(n));
  if (extra.length || lost.length) fail(`the zip's entries are not what was staged (extra: ${extra.slice(0, 5).join(', ') || 'none'}; missing: ${lost.slice(0, 5).join(', ') || 'none'})`);
  if (!names.includes('manifest.json')) fail('manifest.json is not at the root of the zip');
  const bad = names.filter(storeForbidden);
  if (bad.length) fail(`the store zip holds what it must not (no card picture, name or text; D-015, D-025): ${bad.slice(0, 5).join(', ')}`);
  const absent = STORE_REQUIRED.filter((n) => !names.includes(n));
  if (!names.some((n) => /^licenses\/onnxruntime-.+-ThirdPartyNotices\.txt$/.test(n))) absent.push('licenses/onnxruntime-*-ThirdPartyNotices.txt');
  if (names.filter((n) => /^fonts\/OFL-.+\.txt$/.test(n)).length < 3) absent.push('fonts/OFL-*.txt');
  if (absent.length) fail(`the store zip lacks ${absent.join(', ')}`);
  checkStoreManifest(JSON.parse(unzip(['-p', file, 'manifest.json']).stdout), 'the zip\'s manifest.json');
  return { names, top: [...new Set(names.map((n) => (n.includes('/') ? `${n.split('/')[0]}/` : n)))] };
}

/** The Chrome Web Store zip: one zip whose root is the extension (see --store above). */
async function mainStore(o, made) {
  const outFile = resolve(o.out);
  if (!outFile.endsWith('.zip')) fail(`${outFile}: the output must end in .zip`);
  const dist = resolve(o.dist);
  for (const f of DIST_FILES) if (!existsSync(join(dist, f))) fail(`${join(dist, f)} is missing: build the store version first (npm run build:store -w @rifteye/extension)`);
  const extra = listFiles(dist).filter((f) => !DIST_FILES.includes(f.split(sep).join('/')));
  if (extra.length) fail(`${dist} holds files the store zip does not: ${extra.slice(0, 5).join(', ')}`);
  checkStoreManifest(JSON.parse(readFileSync(join(dist, 'manifest.json'), 'utf8')), `${join(dist, 'manifest.json')}`);
  const ort = ortDir(o.ort);
  for (const f of ORT_FILES) if (!existsSync(join(ort, f))) fail(`${join(ort, f)} is missing (onnxruntime-web's runtime file)`);
  const modelsDir = resolve(o.models);
  const assetsDir = resolve(o.assets);
  const models = chooseModels(modelsDir, o.precisions);
  const { index } = await checkAssets(assetsDir, modelsDir, models, true);
  const root = resolve(join(here('./'), '..', '..'));
  for (const f of ['LICENSE', 'NOTICE']) if (!existsSync(join(root, f))) fail(`${join(root, f)} is missing: the store zip carries it`);
  const notices = existsSync(join(root, 'licenses')) ? readdirSync(join(root, 'licenses')).filter((n) => /^onnxruntime-.+-ThirdPartyNotices\.txt$/.test(n)) : [];
  if (notices.length === 0) fail(`${join(root, 'licenses')} has no onnxruntime-*-ThirdPartyNotices.txt: the store zip carries the runtime's notices`);

  // the staging folder: symlinks to everything, and standalone.json. Its root is the zip's root: the extension.
  const stageRoot = o.stage ? resolve(o.stage) : mkdtempSync(join(tmpdir(), 'wardeye-store-pack-'));
  made.root = o.stage ? null : stageRoot;
  const top = join(stageRoot, 'wardeye');
  if (existsSync(top)) fail(`${top} exists already; remove it or pass another --stage`);
  const st = stager(stageRoot, made);
  for (const f of DIST_FILES) st.link(join(dist, f), join(top, f));
  for (const f of ORT_FILES) st.link(join(ort, f), join(top, 'ort', f));
  for (const id of Object.keys(models)) for (const name of Object.values(models[id])) st.link(join(modelsDir, name), join(top, 'models', name));
  st.link(join(assetsDir, 'gallery', 'index.json'), join(top, 'data', 'gallery', 'index.json'));
  for (const l of index.levels) st.link(join(assetsDir, 'gallery', `L${l}.bin`), join(top, 'data', 'gallery', `L${l}.bin`));
  for (const f of ['LICENSE', 'NOTICE']) st.link(join(root, f), join(top, f));
  for (const n of notices) st.link(join(root, 'licenses', n), join(top, 'licenses', n));
  st.write(join(top, 'standalone.json'), standaloneJson(o, models));
  const staged = made.files.map((f) => relative(top, f).split(sep).join('/'));

  // one zip, written beside its final name and moved there once it checks out (an earlier zip of the name is replaced)
  mkdirSync(dirname(outFile), { recursive: true });
  const partial = `${outFile}.partial`;
  rmSync(partial, { force: true });
  try {
    const zip = spawn('zip', ['-X', '-r', '-q', '-D', partial, '.'], { cwd: top, stdio: ['ignore', 'inherit', 'inherit'] });
    await new Promise((done, reject) => {
      zip.on('error', (e) => reject(e.code === 'ENOENT' ? new PackError('the zip command is not installed') : e));
      zip.on('close', (code) => (code === 0 ? done() : reject(new PackError(`zip exited with code ${code}`))));
    });
    const held = verifyStoreZip(partial, staged);
    renameSync(partial, outFile);
    const bytes = statSync(outFile).size;
    console.log(`  ${held.names.length} files: the extension, ${ORT_FILES.length} onnxruntime-web files, models (${MODELS.map((id) => `${id.split('-')[0]} ${o.precisions[id].join('+')}`).join(', ')}), ` +
      `${index.levels.length} gallery levels of ${index.rows.length} printings, notices; ${mb(st.unpacked)} unpacked; no catalogue, no thumbnails`);
    console.log(`${outFile}: ${bytes} bytes (${mb(bytes)}), sha256 ${await sha256File(outFile)}`);
    console.log(`top level: ${held.top.join(' ')}`);
    console.log('verified: unzip -t passes for every entry; the entries are the ones staged; manifest.json is at the root and is the store\'s (no 127.0.0.1); no card picture, name or text');
  } finally {
    rmSync(partial, { force: true });
  }
}

async function main(o, made) {
  if (o.store) return mainStore(o, made);
  const dist = resolve(o.dist);
  for (const f of DIST_FILES) if (!existsSync(join(dist, f))) fail(`${join(dist, f)} is missing: build first (npm run build -w @rifteye/extension)`);
  const ort = ortDir(o.ort);
  for (const f of ORT_FILES) if (!existsSync(join(ort, f))) fail(`${join(ort, f)} is missing (onnxruntime-web's runtime file)`);
  const modelsDir = resolve(o.models);
  const assetsDir = resolve(o.assets);
  const models = chooseModels(modelsDir, o.precisions);
  const { catalog, thumbs, index } = await checkAssets(assetsDir, modelsDir, models);
  const root = resolve(join(here('./'), '..', '..'));
  const outFile = o.sizeOnly ? null : resolve(o.out);
  if (outFile && !outFile.endsWith('.zip')) fail(`${outFile}: the output must end in .zip`);

  // the staging folder: symlinks to everything, and the two generated files
  const stageRoot = o.stage ? resolve(o.stage) : mkdtempSync(join(tmpdir(), 'rifteye-standalone-pack-'));
  made.root = o.stage ? null : stageRoot;
  const top = join(stageRoot, TOP);
  if (existsSync(top)) fail(`${top} exists already; remove it or pass another --stage`);
  const st = stager(stageRoot, made);
  const { link, write } = st;
  for (const f of DIST_FILES) link(join(dist, f), join(top, f));
  for (const f of ORT_FILES) link(join(ort, f), join(top, 'ort', f));
  for (const id of Object.keys(models)) for (const name of Object.values(models[id])) link(join(modelsDir, name), join(top, 'models', name));
  link(join(assetsDir, 'catalog.json'), join(top, 'data', 'catalog.json'));
  link(join(assetsDir, 'gallery', 'index.json'), join(top, 'data', 'gallery', 'index.json'));
  for (const l of index.levels) link(join(assetsDir, 'gallery', `L${l}.bin`), join(top, 'data', 'gallery', `L${l}.bin`));
  for (const name of thumbs) {
    if (!PORTABLE.test(name)) fail(`the thumbnail ${name} has a name that is not portable`);
    link(join(assetsDir, 'thumbs', name), join(top, 'data', 'thumbs', name));
  }
  for (const f of ['LICENSE', 'NOTICE']) if (existsSync(join(root, f))) link(join(root, f), join(top, f));
  write(join(top, 'standalone.json'), standaloneJson(o, models));
  write(join(top, 'INSTALL.txt'), INSTALL);

  const fileCount = made.files.length;
  const hash = createHash('sha256');
  let bytes = 0;
  const tap = new Transform({
    transform(chunk, _enc, cb) {
      hash.update(chunk);
      bytes += chunk.length;
      cb(null, chunk);
    },
  });
  const summary = () =>
    console.log(
      `  ${fileCount} files: the extension, ${ORT_FILES.length} onnxruntime-web files, models (${MODELS.map((id) => `${id.split('-')[0]} ${o.precisions[id].join('+')}`).join(', ')}), ` +
        `${index.levels.length} gallery levels of ${catalog.length} printings, ${thumbs.length} thumbnails; ${mb(st.unpacked)} unpacked`,
    );

  if (o.sizeOnly) {
    await streamZip(stageRoot, tap, new Writable({ write: (_c, _e, cb) => cb() }));
    summary();
    console.log(`the zip would be ${mb(bytes)} (${bytes} bytes), ${Math.ceil(bytes / o.partBytes)} parts of at most ${mib(o.partBytes)}; nothing was written`);
    return;
  }

  // the parts: OUT.part-aa, OUT.part-ab, ...; those of an earlier pack of the same name are replaced
  const dir = dirname(outFile);
  mkdirSync(dir, { recursive: true });
  const prefix = `${outFile}.part-`;
  const mine = new RegExp(`^${basename(outFile).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\.part-[a-z]{2}$`);
  for (const n of readdirSync(dir).filter((n) => mine.test(n))) unlinkSync(join(dir, n));
  const { sink, exited } = splitter(prefix, o.partBytes);
  await Promise.all([streamZip(stageRoot, tap, sink), exited]);
  const parts = readdirSync(dir).filter((n) => mine.test(n)).sort().map((n) => join(dir, n));
  const sha = hash.digest('hex');
  summary();
  for (const p of parts) console.log(`  ${basename(p)}  ${statSync(p).size} bytes`);
  console.log(`${parts.length} parts, ${bytes} bytes (${mb(bytes)}), none over ${mib(o.partBytes)}; joined, sha256 ${sha}`);
  if (parts.some((p) => statSync(p).size > o.partBytes)) fail('a part is over the size asked for');
  if (o.verify) {
    const v = await verifyParts(parts, { bytes, sha });
    console.log(`verified: the parts join to the zip that was streamed (sha256 ${v.sha.slice(0, 12)}...)${v.files === null ? '' : `, and every one of its ${v.files} entries passes its CRC`}${v.note ? `; ${v.note}` : ''}`);
  }
  console.log(`to rebuild the zip: cat ${basename(outFile)}.part-* > ${basename(outFile)}`);
}

// what a run makes in the staging folder is removed at the end: the links and the two generated files (never what the
// links point to), then the folders
const made = { files: [], dirs: [], root: null };
let code = 0;
try {
  await main(parseArgs(process.argv.slice(2)), made);
} catch (e) {
  console.error(e instanceof PackError ? `pack: ${e.message}` : e);
  code = 1;
} finally {
  for (const f of made.files) if (lstatSync(f, { throwIfNoEntry: false })) unlinkSync(f);
  for (const d of [...made.dirs].sort((a, b) => b.length - a.length)) {
    try {
      rmdirSync(d);
    } catch {
      // not empty: something else is in there, leave it
    }
  }
  if (made.root) {
    try {
      rmdirSync(made.root);
    } catch {
      // not empty: leave it
    }
  }
}
process.exit(code);
