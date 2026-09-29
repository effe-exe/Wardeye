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
//
// Nothing big is copied: the staging folder holds symlinks, which `zip` follows. The models, the gallery and the
// thumbnails are made from Riot's card art and stay private (decision D-006): they go only into this zip.
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmdirSync, statSync, symlinkSync, unlinkSync,
  writeFileSync,
} from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Transform, Writable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const TOP = 'rifteye-standalone';
const DIST_FILES = ['manifest.json', 'content.js', 'overlay.css', 'worker.js', 'offscreen.html', 'offscreen.js', 'engine-webgpu.js', 'engine-wasm.js'];
// onnxruntime-web's runtime: the native WebGPU build (JSPI) and the plain WASM build; not JSEP (broken for GridSample in fp16)
const ORT_FILES = ['ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const PORTABLE = /^[A-Za-z0-9._-]+$/; // a file name that every system unzips
const MODELS = ['detector-v0', 'embedder-v1'];
const USAGE =
  'usage: node pack.mjs --out FILE.zip [--models DIR] [--assets DIR] [--precisions detector:fp32,embedder:fp16] [--size-only] [--part-bytes N] ' +
  '[--runtime auto|webgpu|wasm|companion] [--layout NAME] [--dist DIR] [--ort DIR] [--stage DIR] [--no-verify]';

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
    runtime: 'auto', layout: null, dist: here('./dist'), ort: null, stage: null, verify: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const [flag, inline] = argv[i].startsWith('--') ? argv[i].split(/=(.*)/s, 2) : [argv[i]];
    const value = () => inline ?? argv[++i] ?? fail(USAGE);
    if (flag === '--out') o.out = value();
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
  if (!o.sizeOnly && !o.out) fail(USAGE);
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

/** The gallery, the catalogue and the thumbnails, read and checked against each other and against the models. */
async function checkAssets(assetsDir, modelsDir, models) {
  const at = (...p) => join(assetsDir, ...p);
  for (const f of ['gallery/index.json', 'catalog.json', 'thumbs']) if (!existsSync(at(f))) fail(`${at(f)} is missing: run python -m rifteye_ml.web_assets first`);
  let index;
  let catalog;
  try {
    index = JSON.parse(readFileSync(at('gallery', 'index.json'), 'utf8'));
    catalog = JSON.parse(readFileSync(at('catalog.json'), 'utf8'));
  } catch (e) {
    return fail(`the gallery's index or the catalogue cannot be read (${e.message})`);
  }
  if (index.dtype !== 'float16') fail(`the gallery is ${index.dtype}: the package ships float16 (python -m rifteye_ml.web_assets writes it)`);
  if (index.format !== 1 || !Array.isArray(index.levels) || !Array.isArray(index.rows) || !Array.isArray(catalog)) fail('the gallery index is not what this extension reads (format 1)');
  if (index.rows.length !== catalog.length) fail(`the gallery has ${index.rows.length} rows and the catalogue ${catalog.length}`);
  const missing = index.levels.filter((l) => !existsSync(at('gallery', `L${l}.bin`)));
  if (missing.length) fail(`the gallery's levels ${missing.join(', ')} have no .bin file`);
  const want = index.rows.length * index.dim * 2;
  const wrong = index.levels.filter((l) => statSync(at('gallery', `L${l}.bin`)).size !== want);
  if (wrong.length) fail(`the gallery's levels ${wrong.join(', ')} are not ${want} bytes (${index.rows.length} rows of ${index.dim} float16)`);
  const thumbs = readdirSync(at('thumbs')).filter((n) => n.endsWith('.jpg'));
  if (thumbs.length < catalog.length) fail(`there are ${thumbs.length} thumbnails for ${catalog.length} printings`);
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

const INSTALL = `RiftEye, standalone build (private)

1. Unzip this file. You get one folder, rifteye-standalone.
2. In Chrome (137 or newer), open chrome://extensions and turn on Developer mode.
3. Click "Load unpacked" and pick the rifteye-standalone folder.
4. Play a Riftbound replay or stream on twitch.tv. The badge on the player says what RiftEye is doing.

The models and data in this folder are made from Riot's card art: keep them to yourself (decision D-006).
Nothing leaves your computer. If your browser has no WebGPU, RiftEye uses the live runner on this machine instead.
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

async function main(o, made) {
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
  const mkdir = (d) => {
    mkdirSync(d, { recursive: true });
    for (let p = d; p.length > stageRoot.length && !made.dirs.includes(p); p = dirname(p)) made.dirs.push(p);
  };
  let unpacked = 0;
  const link = (from, to) => {
    mkdir(dirname(to));
    symlinkSync(from, to);
    made.files.push(to);
    unpacked += statSync(from).size;
  };
  const write = (to, text) => {
    mkdir(dirname(to));
    writeFileSync(to, text);
    made.files.push(to);
    unpacked += Buffer.byteLength(text);
  };
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
        `${index.levels.length} gallery levels of ${catalog.length} printings, ${thumbs.length} thumbnails; ${mb(unpacked)} unpacked`,
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
