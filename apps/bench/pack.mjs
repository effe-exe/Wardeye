// Packs the bench for someone who is to run it: a zip with one folder, rifteye-bench/, holding dist/ (the
// extension) and models/ (index.json, the manifests, the chosen precisions' .onnx files and the check files).
// They unzip it, load the folder in chrome://extensions ("Load unpacked", Developer mode) and press Run.
//
//   node pack.mjs --out rifteye-bench.zip [--models DIR] [--precisions fp16|fp32,fp16] [--size-only]
//                 [--dist DIR] [--stage DIR]
//
// --models      where the exported models are (default $RIFTEYE_DATA/models/onnx, else ~/rifteye-data/models/onnx,
//               like the Python tools)
// --precisions  which variants go in (default fp16 only, the smaller download); the others are left out and the
//               bench lists them as "not included"
// --size-only   builds nothing: streams the zip through a byte counter and prints what its size would be
// --stage       where the staging folder of symlinks goes (default: a folder of its own under the temp folder)
//
// The model files are never copied: the staging folder holds symlinks, which `zip` follows, so the only new
// file on disk is the zip. The models are private (decision D-006) and stay outside the repository.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  closeSync, createReadStream, existsSync, lstatSync, mkdirSync, mkdtempSync, openSync, readdirSync, readFileSync, readSync,
  rmdirSync, statSync, symlinkSync, unlinkSync, writeFileSync,
} from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const TOP = 'rifteye-bench';
const PLAIN_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const USAGE = 'usage: node pack.mjs --out FILE.zip [--models DIR] [--precisions fp16|fp32,fp16] [--size-only] [--dist DIR] [--stage DIR]';

class PackError extends Error {}
const fail = (message) => {
  throw new PackError(message);
};

function parseArgs(argv) {
  const data = process.env.RIFTEYE_DATA || join(homedir(), 'rifteye-data');
  const o = { models: join(data, 'models', 'onnx'), out: null, precisions: ['fp16'], sizeOnly: false, dist: here('./dist'), stage: null };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const [flag, inline] = argv[i].startsWith('--') ? argv[i].split(/=(.*)/s, 2) : [argv[i]];
    const value = () => inline ?? argv[++i] ?? fail(USAGE);
    if (flag === '--models') o.models = value();
    else if (flag === '--out') o.out = value();
    else if (flag === '--precisions') o.precisions = value().split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
    else if (flag === '--dist') o.dist = value();
    else if (flag === '--stage') o.stage = value();
    else if (flag === '--size-only') o.sizeOnly = true;
    else if (flag.startsWith('-')) fail(USAGE);
    else positional.push(flag);
  }
  // a plain argument is the zip when it ends in .zip, and the models folder otherwise
  for (const p of positional) (p.endsWith('.zip') ? (o.out = p) : (o.models = p));
  if (o.precisions.length === 0 || (!o.sizeOnly && !o.out)) fail(USAGE);
  return o;
}

const mb = (n) => `${(n / 1e6).toFixed(1)} MB`;

/** Every file under `dir`, as paths relative to it. */
function listFiles(dir, prefix = '') {
  return readdirSync(join(dir, prefix), { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? listFiles(dir, join(prefix, e.name)) : [join(prefix, e.name)],
  );
}

/** What goes into models/: read from each manifest which files it names, and choose the variants. */
function chooseModels(modelsDir, precisions) {
  if (!existsSync(modelsDir)) fail(`the models folder ${modelsDir} does not exist`);
  const manifests = readdirSync(modelsDir).filter((n) => n.endsWith('.bench.json')).sort();
  if (manifests.length === 0) fail(`no *.bench.json in ${modelsDir}`);
  const files = new Set();
  const models = [];
  for (const name of manifests) {
    let m;
    try {
      m = JSON.parse(readFileSync(join(modelsDir, name), 'utf8'));
    } catch (e) {
      fail(`${name} is not valid JSON (${e.message})`);
    }
    const id = String(m.id ?? name.replace(/\.bench\.json$/, ''));
    const named = (f) => (typeof f === 'string' && PLAIN_NAME.test(f) ? f : fail(`${name}: "${f}" is not a plain file name`));
    const variants = (m.variants ?? []).map((v) => ({ precision: String(v.precision), file: named(v.file) }));
    const chosen = variants.filter((v) => precisions.includes(v.precision.toLowerCase()));
    const missing = chosen.filter((v) => !existsSync(join(modelsDir, v.file)));
    if (missing.length) fail(`${id}: ${missing.map((v) => `${v.file} (${v.precision})`).join(', ')} is not in ${modelsDir}`);
    files.add(name);
    if (chosen.length === 0) {
      console.warn(`pack: ${id} has no ${precisions.join('/')} variant; only its manifest goes in`);
    } else {
      for (const v of chosen) files.add(v.file);
      // the check's input and expected outputs, and the decoded check's expected cards when it has one
      const checkFiles = m.check
        ? [m.check.input, ...Object.values(m.check.expected ?? {}), ...(m.check.detections ? [m.check.detections.file] : [])].map(named)
        : [];
      const gone = checkFiles.filter((f) => !existsSync(join(modelsDir, f)));
      if (gone.length) fail(`${id}: check files ${gone.join(', ')} are not in ${modelsDir}`);
      for (const f of checkFiles) files.add(f);
    }
    models.push({ id, chosen: chosen.map((v) => v.precision), left: variants.filter((v) => !chosen.includes(v)).map((v) => v.precision) });
  }
  return { manifests, files: [...files], models };
}

/** Runs `zip` in `cwd`; when `out` is "-" the zip goes to stdout and the bytes are counted (and returned). */
function zip(cwd, out) {
  return new Promise((done, reject) => {
    const child = spawn('zip', ['-X', '-r', '-q', out, TOP], { cwd, stdio: ['ignore', out === '-' ? 'pipe' : 'ignore', 'inherit'] });
    let bytes = 0;
    child.stdout?.on('data', (c) => (bytes += c.length));
    child.on('error', (e) => reject(e.code === 'ENOENT' ? new PackError('the zip command is not installed') : e));
    child.on('close', (code) => (code === 0 ? done(bytes) : reject(new PackError(`zip exited with code ${code}`))));
  });
}

function sha256(file) {
  return new Promise((done, reject) => {
    const h = createHash('sha256');
    createReadStream(file).on('data', (c) => h.update(c)).on('end', () => done(h.digest('hex'))).on('error', reject);
  });
}

function isZip(file) {
  const fd = openSync(file, 'r');
  try {
    const head = Buffer.alloc(2);
    readSync(fd, head, 0, 2, 0);
    return head.toString('latin1') === 'PK';
  } finally {
    closeSync(fd);
  }
}

async function main(opts, made) {
  const dist = resolve(opts.dist);
  for (const f of ['manifest.json', 'bench.html', 'bench.js', 'ort']) {
    if (!existsSync(join(dist, f))) fail(`${join(dist, f)} is missing: build first (npm run build -w @rifteye/bench)`);
  }
  const modelsDir = resolve(opts.models);
  const { manifests, files, models } = chooseModels(modelsDir, opts.precisions);
  const outFile = opts.sizeOnly ? null : resolve(opts.out);
  if (outFile && !outFile.endsWith('.zip')) fail(`${outFile}: the output must be a .zip`);
  if (outFile && existsSync(outFile) && !isZip(outFile)) fail(`${outFile} exists and is not a zip; not overwriting it`);

  // the staging folder: symlinks to dist's files and the models' files, and the generated models/index.json
  const stageRoot = opts.stage ? resolve(opts.stage) : mkdtempSync(join(tmpdir(), 'rifteye-bench-pack-'));
  made.root = opts.stage ? null : stageRoot;
  const top = join(stageRoot, TOP);
  if (existsSync(top)) fail(`${top} exists already; remove it or pass another --stage`);
  const mkdir = (d) => {
    mkdirSync(d, { recursive: true });
    for (let p = d; p.length > stageRoot.length && !made.dirs.includes(p); p = dirname(p)) made.dirs.push(p);
  };
  const link = (from, to) => {
    mkdir(dirname(to));
    symlinkSync(from, to);
    made.files.push(to);
  };
  let unpacked = 0;
  for (const f of listFiles(dist)) {
    link(join(dist, f), join(top, f));
    unpacked += statSync(join(dist, f)).size;
  }
  for (const f of files) {
    link(join(modelsDir, f), join(top, 'models', f));
    unpacked += statSync(join(modelsDir, f)).size;
  }
  mkdir(join(top, 'models'));
  const index = join(top, 'models', 'index.json');
  writeFileSync(index, `${JSON.stringify(manifests, null, 2)}\n`);
  made.files.push(index);

  let size;
  if (outFile) {
    if (existsSync(outFile)) unlinkSync(outFile); // an earlier pack, replaced
    mkdirSync(dirname(outFile), { recursive: true });
    await zip(stageRoot, outFile);
    size = statSync(outFile).size;
  } else {
    size = await zip(stageRoot, '-');
  }
  for (const m of models) {
    console.log(`  ${m.id}: ${m.chosen.length ? m.chosen.join(', ') : 'manifest only'}${m.left.length ? `; ${m.left.join(', ')} left out (shown as "not included")` : ''}`);
  }
  console.log(`  ${files.length} model files; the extension and the models are ${mb(unpacked)} unpacked`);
  if (outFile) console.log(`${outFile}: ${mb(size)} (${size} bytes), sha256 ${await sha256(outFile)}`);
  else console.log(`the zip with ${opts.precisions.join(',')} would be ${mb(size)} (${size} bytes); nothing was written`);
}

// what a run makes in the staging folder is removed at the end: the symlinks (never what they point to), then the folders
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
