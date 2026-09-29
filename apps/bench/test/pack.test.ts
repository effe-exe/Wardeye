import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateRawSync } from 'node:zlib';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const PACK = fileURLToPath(new URL('../pack.mjs', import.meta.url));
const haveZip = spawnSync('zip', ['-v']).status === 0;

/** The entries of a zip, read from its central directory (no unzip needed): name -> the bytes of the file. */
function readZip(file: string): Map<string, Buffer> {
  const buf = readFileSync(file);
  let end = buf.length - 22;
  while (end >= 0 && buf.readUInt32LE(end) !== 0x06054b50) end--;
  expect(end).toBeGreaterThanOrEqual(0);
  const count = buf.readUInt16LE(end + 10);
  let p = buf.readUInt32LE(end + 16);
  const entries = new Map<string, Buffer>();
  for (let i = 0; i < count; i++) {
    const method = buf.readUInt16LE(p + 10);
    const packed = buf.readUInt32LE(p + 20);
    const n = buf.readUInt16LE(p + 28);
    const extra = buf.readUInt16LE(p + 30);
    const comment = buf.readUInt16LE(p + 32);
    const local = buf.readUInt32LE(p + 42);
    const name = buf.toString('utf8', p + 46, p + 46 + n);
    const start = local + 30 + buf.readUInt16LE(local + 26) + buf.readUInt16LE(local + 28);
    const data = buf.subarray(start, start + packed);
    entries.set(name, method === 0 ? data : inflateRawSync(data));
    p += 46 + n + extra + comment;
  }
  return entries;
}

const zipNames = (file: string): string[] => [...readZip(file).keys()];

const run = (args: string[]) => spawnSync('node', [PACK, ...args], { encoding: 'utf8' });

describe.skipIf(!haveZip)('pack.mjs', () => {
  let root: string;
  let dist: string;
  let models: string;
  let stage: string;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'rifteye-pack-test-'));
    dist = join(root, 'dist');
    models = join(root, 'models');
    stage = join(root, 'stage');
    for (const d of [join(dist, 'ort'), models, stage]) mkdirSync(d, { recursive: true });
    for (const f of ['manifest.json', 'bench.html', 'bench.js', 'ort/ort-wasm-simd-threaded.wasm']) writeFileSync(join(dist, f), `dist ${f}`);
    const manifest = (id: string) =>
      JSON.stringify({
        id,
        variants: [
          { precision: 'fp32', file: `${id}.onnx` },
          { precision: 'fp16', file: `${id}.fp16.onnx` },
        ],
        check: { batch: 1, input: `${id}.check.input.bin`, expected: { out: `${id}.check.out.bin` } },
      });
    for (const id of ['det', 'emb']) {
      writeFileSync(join(models, `${id}.bench.json`), manifest(id));
      for (const f of [`${id}.onnx`, `${id}.fp16.onnx`, `${id}.check.input.bin`, `${id}.check.out.bin`]) writeFileSync(join(models, f), `${'x'.repeat(2000)} ${f}`);
    }
    writeFileSync(join(models, 'unrelated.txt'), 'not part of the bench');
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it('packs dist and, by default, the fp16 models with their manifests and check files', () => {
    const out = join(root, 'a.zip');
    const r = run(['--models', models, '--dist', dist, '--stage', stage, '--out', out]);
    expect(r.status, r.stderr).toBe(0);
    expect(zipNames(out).filter((n) => !n.endsWith('/')).sort()).toEqual([
      'rifteye-bench/bench.html',
      'rifteye-bench/bench.js',
      'rifteye-bench/manifest.json',
      'rifteye-bench/models/det.bench.json',
      'rifteye-bench/models/det.check.input.bin',
      'rifteye-bench/models/det.check.out.bin',
      'rifteye-bench/models/det.fp16.onnx',
      'rifteye-bench/models/emb.bench.json',
      'rifteye-bench/models/emb.check.input.bin',
      'rifteye-bench/models/emb.check.out.bin',
      'rifteye-bench/models/emb.fp16.onnx',
      'rifteye-bench/models/index.json',
      'rifteye-bench/ort/ort-wasm-simd-threaded.wasm',
    ]);
    expect(r.stdout).toContain('det: fp16; fp32 left out (shown as "not included")');
    expect(r.stdout).toMatch(/a\.zip: [\d.]+ MB \(\d+ bytes\), sha256 [0-9a-f]{64}/);
  });

  it('puts both precisions in when asked, and lists the manifests in models/index.json', () => {
    const out = join(root, 'b.zip');
    expect(run(['--models', models, '--dist', dist, '--stage', stage, '--precisions', 'fp32,fp16', '--out', out]).status).toBe(0);
    const names = zipNames(out);
    expect(names).toContain('rifteye-bench/models/det.onnx');
    expect(names).toContain('rifteye-bench/models/det.fp16.onnx');
    expect(names).toContain('rifteye-bench/models/emb.onnx');
    expect(names).not.toContain('rifteye-bench/models/unrelated.txt');
    // models/index.json is generated: the manifests, sorted
    expect(JSON.parse(readZip(out).get('rifteye-bench/models/index.json')!.toString('utf8'))).toEqual(['det.bench.json', 'emb.bench.json']);
    // and the model files are the real ones, followed through their symlinks
    expect(readZip(out).get('rifteye-bench/models/emb.fp16.onnx')!.toString('utf8')).toContain('emb.fp16.onnx');
  });

  it('follows the symlinks: the models are in the zip, and the models folder is left as it was', () => {
    const out = join(root, 'c.zip');
    const before = readdirSync(models).sort();
    expect(run(['--models', models, '--dist', dist, '--stage', stage, '--out', out]).status).toBe(0);
    expect(statSync(out).size).toBeGreaterThan(0);
    expect(readdirSync(models).sort()).toEqual(before);
    expect(readFileSync(join(models, 'det.fp16.onnx'), 'utf8')).toContain('det.fp16.onnx');
    expect(readdirSync(stage)).toEqual([]); // the staging symlinks are removed
  });

  it('says what a zip would weigh without writing one', () => {
    const real = join(root, 'd.zip');
    run(['--models', models, '--dist', dist, '--stage', stage, '--precisions', 'fp32,fp16', '--out', real]);
    const r = run(['--models', models, '--dist', dist, '--stage', stage, '--precisions', 'fp32,fp16', '--size-only']);
    expect(r.status, r.stderr).toBe(0);
    const bytes = Number(/would be [\d.]+ MB \((\d+) bytes\); nothing was written/.exec(r.stdout)?.[1]);
    expect(bytes).toBeGreaterThan(0);
    expect(Math.abs(bytes - statSync(real).size)).toBeLessThan(statSync(real).size * 0.25);
    expect(existsSync(join(root, 'e.zip'))).toBe(false);
    expect(readdirSync(stage)).toEqual([]);
  });

  it('fails, and leaves nothing behind, when a variant it was asked for is not there', () => {
    const lacking = join(root, 'lacking');
    mkdirSync(lacking);
    writeFileSync(join(lacking, 'det.bench.json'), readFileSync(join(models, 'det.bench.json')));
    const out = join(root, 'f.zip');
    const r = run(['--models', lacking, '--dist', dist, '--stage', stage, '--out', out]);
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('det: det.fp16.onnx (fp16) is not in');
    expect(existsSync(out)).toBe(false);
    expect(readdirSync(stage)).toEqual([]);
  });

  it('will not overwrite a file that is not a zip, and asks for a build when dist is missing', () => {
    const notZip = join(root, 'notes.zip');
    writeFileSync(notZip, 'my notes');
    const r = run(['--models', models, '--dist', dist, '--stage', stage, '--out', notZip]);
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('is not a zip; not overwriting it');
    expect(readFileSync(notZip, 'utf8')).toBe('my notes');
    const nodist = run(['--models', models, '--dist', join(root, 'nope'), '--out', join(root, 'g.zip')]);
    expect(nodist.status).toBe(1);
    expect(nodist.stderr).toContain('build first');
  });

  it('takes the models folder and the zip as plain arguments too', () => {
    const out = join(root, 'h.zip');
    const r = run([models, out, '--dist', dist, '--stage', stage]);
    expect(r.status, r.stderr).toBe(0);
    expect(zipNames(out)).toContain('rifteye-bench/models/index.json');
  });
});
