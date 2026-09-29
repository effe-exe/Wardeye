import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateRawSync } from 'node:zlib';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const PACK = fileURLToPath(new URL('../pack.mjs', import.meta.url));
const haveTools = ['zip', 'split', 'python3'].every((c) => spawnSync(c, ['--version']).status !== null && !spawnSync(c, ['--version']).error);

/** The entries of a zip, read from its central directory: name -> the bytes of the file. */
function readZip(buf: Buffer): Map<string, Buffer> {
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

const sha = (b: Buffer | string): string => createHash('sha256').update(b).digest('hex');
const partsOf = (dir: string, base: string): string[] => readdirSync(dir).filter((n) => n.startsWith(`${base}.part-`)).sort().map((n) => join(dir, n));
const joined = (parts: string[]): Buffer => Buffer.concat(parts.map((p) => readFileSync(p)));
const run = (args: string[]) => spawnSync('node', [PACK, ...args], { encoding: 'utf8' });

describe.skipIf(!haveTools)('pack.mjs', () => {
  let root: string;
  let dist: string;
  let ort: string;
  let models: string;
  let assets: string;
  let stage: string;
  let out: string;
  const args = (...more: string[]) => ['--dist', dist, '--ort', ort, '--models', models, '--assets', assets, '--stage', stage, ...more];

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'rifteye-standalone-pack-test-'));
    [dist, ort, models, assets, stage, out] = ['dist', 'ort', 'models', 'assets', 'stage', 'out'].map((d) => join(root, d)) as [string, string, string, string, string, string];
    for (const d of [dist, ort, models, join(assets, 'gallery'), join(assets, 'thumbs'), stage, out]) mkdirSync(d, { recursive: true });
    for (const f of ['manifest.json', 'content.js', 'overlay.css', 'worker.js', 'offscreen.html', 'offscreen.js', 'engine-webgpu.js', 'engine-wasm.js']) writeFileSync(join(dist, f), `dist ${f}`);
    // what the manifest names besides: the toolbar icons, and the overlay's typefaces with their licences
    for (const d of ['icons', 'fonts']) mkdirSync(join(dist, d), { recursive: true });
    for (const f of [
      'icons/icon-16.png', 'icons/icon-32.png', 'icons/icon-48.png', 'icons/icon-128.png', 'fonts/SpaceGrotesk-latin.woff2', 'fonts/Inter-latin.woff2',
      'fonts/JetBrainsMono-latin.woff2', 'fonts/OFL-SpaceGrotesk.txt', 'fonts/OFL-Inter.txt', 'fonts/OFL-JetBrainsMono.txt',
    ]) writeFileSync(join(dist, f), `dist ${f}`);
    for (const f of ['ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.jsep.wasm']) writeFileSync(join(ort, f), `ort ${f}`);
    // models: made-up bytes, and the gallery says which embedder it was made with
    const body = (f: string) => `${f} ${'x'.repeat(3000)}`;
    for (const f of ['detector-v0.onnx', 'detector-v0.fp16.onnx', 'embedder-v1.onnx', 'embedder-v1.fp16.onnx']) writeFileSync(join(models, f), body(f));
    writeFileSync(join(models, 'detector-v0.bench.json'), '{}');
    const rows = ['TST-001', 'TST-002*', 'TST-003'];
    writeFileSync(join(assets, 'gallery', 'index.json'), JSON.stringify({
      format: 1, encoder: 'onnx:embedder-v1-abc', model: 'embedder-v1', sha256: sha(body('embedder-v1.onnx')), fp16_sha256: sha(body('embedder-v1.fp16.onnx')),
      dim: 4, dtype: 'float16', levels: [80, 90], rows,
    }));
    for (const l of [80, 90]) writeFileSync(join(assets, 'gallery', `L${l}.bin`), Buffer.alloc(rows.length * 4 * 2, l));
    writeFileSync(join(assets, 'catalog.json'), JSON.stringify(rows.map((r) => ({ printing_id: r, card_id: r, name: r, type: 'Unit' }))));
    for (const r of ['TST-001', 'TST-002_2a', 'TST-003']) writeFileSync(join(assets, 'thumbs', `${r}.jpg`), `jpeg ${r}`);
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it('streams the private build into parts, none over the size asked, that join to a good zip (the detector in float32, the embedder in float16)', () => {
    const zip = join(out, 'a.zip');
    const r = run(args('--out', zip, '--part-bytes', '3000'));
    expect(r.status, r.stderr).toBe(0);
    const parts = partsOf(out, 'a.zip');
    expect(parts.length).toBeGreaterThan(3);
    expect(parts.every((p) => statSync(p).size <= 3000)).toBe(true);
    expect(existsSync(zip)).toBe(false); // the whole zip is never on disk
    const zipBytes = joined(parts);
    expect(r.stdout).toContain(`${parts.length} parts, ${zipBytes.length} bytes`);
    expect(r.stdout).toContain(`sha256 ${sha(zipBytes)}`);
    expect(r.stdout).toMatch(/verified: the parts join to the zip that was streamed .*every one of its \d+ entries passes its CRC/);

    const files = readZip(zipBytes);
    expect([...files.keys()].filter((n) => !n.endsWith('/')).sort()).toEqual(
      [
        'INSTALL.txt', 'LICENSE', 'NOTICE', 'content.js', 'data/catalog.json', 'data/gallery/L80.bin', 'data/gallery/L90.bin', 'data/gallery/index.json',
        'data/thumbs/TST-001.jpg', 'data/thumbs/TST-002_2a.jpg', 'data/thumbs/TST-003.jpg', 'engine-wasm.js', 'engine-webgpu.js',
        'fonts/Inter-latin.woff2', 'fonts/JetBrainsMono-latin.woff2', 'fonts/OFL-Inter.txt', 'fonts/OFL-JetBrainsMono.txt', 'fonts/OFL-SpaceGrotesk.txt',
        'fonts/SpaceGrotesk-latin.woff2', 'icons/icon-128.png', 'icons/icon-16.png', 'icons/icon-32.png', 'icons/icon-48.png', 'manifest.json',
        'models/detector-v0.onnx', 'models/embedder-v1.fp16.onnx', 'offscreen.html', 'offscreen.js', 'ort/ort-wasm-simd-threaded.jspi.mjs',
        'ort/ort-wasm-simd-threaded.jspi.wasm', 'ort/ort-wasm-simd-threaded.mjs', 'ort/ort-wasm-simd-threaded.wasm', 'overlay.css', 'standalone.json', 'worker.js',
      ].map((n) => `rifteye-standalone/${n}`),
    );
    // the files are the real ones, followed through their symlinks; JSEP is not in; the other precisions are left out
    expect(files.get('rifteye-standalone/models/embedder-v1.fp16.onnx')!.toString()).toContain('embedder-v1.fp16.onnx');
    expect(files.get('rifteye-standalone/ort/ort-wasm-simd-threaded.jspi.wasm')!.toString()).toBe('ort ort-wasm-simd-threaded.jspi.wasm');
    expect(files.get('rifteye-standalone/icons/icon-16.png')!.toString()).toBe('dist icons/icon-16.png'); // the manifest's icons and typefaces are in, as themselves
    expect(files.get('rifteye-standalone/fonts/Inter-latin.woff2')!.toString()).toBe('dist fonts/Inter-latin.woff2');
    // what the person who unzips it reads says Wardeye; the folder's name is the one thing that still says rifteye
    const install = files.get('rifteye-standalone/INSTALL.txt')!.toString();
    expect(install).toContain('Wardeye, standalone build');
    expect(install.replaceAll('rifteye-standalone', '')).not.toMatch(/rifteye/i);
    expect(JSON.parse(files.get('rifteye-standalone/standalone.json')!.toString())).toEqual({
      format: 1,
      runtime: 'auto',
      detector: { id: 'detector-v0', fp32: 'models/detector-v0.onnx' },
      embedder: { id: 'embedder-v1', fp16: 'models/embedder-v1.fp16.onnx' },
      data: 'data/',
    });
    expect(files.has('rifteye-standalone/models/detector-v0.fp16.onnx')).toBe(false); // it fails on native WebGPU
    expect(files.has('rifteye-standalone/models/embedder-v1.onnx')).toBe(false);
    expect(readdirSync(stage)).toEqual([]); // the staging symlinks are removed
    expect(readFileSync(join(models, 'embedder-v1.fp16.onnx'), 'utf8')).toContain('embedder-v1.fp16.onnx'); // and what they pointed to is not
  });

  it('puts the precisions asked for in, for each model or for both, and says the runtime and the layout it is told', () => {
    const zip = join(out, 'b.zip');
    expect(run(args('--out', zip, '--precisions', 'detector:fp32,embedder:fp32+fp16', '--runtime', 'wasm', '--layout', 'la-rq')).status).toBe(0);
    const files = readZip(joined(partsOf(out, 'b.zip')));
    expect(files.has('rifteye-standalone/models/detector-v0.onnx')).toBe(true);
    expect(files.has('rifteye-standalone/models/detector-v0.fp16.onnx')).toBe(false);
    expect(files.has('rifteye-standalone/models/embedder-v1.onnx')).toBe(true);
    expect(JSON.parse(files.get('rifteye-standalone/standalone.json')!.toString())).toMatchObject({
      runtime: 'wasm', layout: 'la-rq', detector: { id: 'detector-v0', fp32: 'models/detector-v0.onnx' },
      embedder: { id: 'embedder-v1', fp16: 'models/embedder-v1.fp16.onnx', fp32: 'models/embedder-v1.onnx' },
    });
    // a bare list is for both models
    const c = join(out, 'c2.zip');
    expect(run(args('--out', c, '--precisions', 'fp32,fp16')).status).toBe(0);
    const both = [...readZip(joined(partsOf(out, 'c2.zip'))).keys()].filter((n) => n.includes('/models/') && !n.endsWith('/'));
    expect(both.sort()).toEqual(['detector-v0.fp16.onnx', 'detector-v0.onnx', 'embedder-v1.fp16.onnx', 'embedder-v1.onnx'].map((n) => `rifteye-standalone/models/${n}`));
  });

  it('says what the zip would weigh without writing anything', () => {
    const r = run(args('--size-only', '--part-bytes', '3000'));
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toMatch(/the zip would be [\d.]+ MB \((\d+) bytes\), \d+ parts of at most [\d.]+ MiB; nothing was written/);
    const real = join(out, 'c.zip');
    expect(run(args('--out', real, '--part-bytes', '3000')).status).toBe(0);
    const bytes = joined(partsOf(out, 'c.zip')).length;
    expect(r.stdout).toContain(`(${bytes} bytes)`); // the dry run is exact
    expect(readdirSync(stage)).toEqual([]);
  });

  it('replaces the parts of an earlier pack of the same name, and leaves the other files of the folder', () => {
    writeFileSync(join(out, 'd.zip.part-zz'), 'an old part');
    writeFileSync(join(out, 'notes.txt'), 'mine');
    expect(run(args('--out', join(out, 'd.zip'), '--part-bytes', '3000')).status).toBe(0);
    expect(existsSync(join(out, 'd.zip.part-zz'))).toBe(false);
    expect(readFileSync(join(out, 'notes.txt'), 'utf8')).toBe('mine');
  });

  it('refuses a build that is not built, models that are missing, and a gallery made for another embedder', () => {
    expect(run(args('--out', join(out, 'e.zip'), '--dist', join(root, 'nowhere'))).stderr).toContain('build first');
    expect(run(args('--out', join(out, 'e.zip'), '--models', join(root, 'nowhere'))).stderr).toContain('does not exist');
    writeFileSync(join(models, 'embedder-v1.fp16.onnx'), 'another export');
    const r = run(args('--out', join(out, 'e.zip')));
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("is not the fp16 embedder the gallery was made with");
    writeFileSync(join(models, 'embedder-v1.fp16.onnx'), `embedder-v1.fp16.onnx ${'x'.repeat(3000)}`);
    expect(partsOf(out, 'e.zip')).toEqual([]);
  });

  it('refuses a gallery that does not hold together, and asks to run web_assets first', () => {
    const bin = join(assets, 'gallery', 'L90.bin');
    const good = readFileSync(bin);
    writeFileSync(bin, 'short');
    expect(run(args('--out', join(out, 'f.zip'))).stderr).toContain('are not 24 bytes');
    writeFileSync(bin, good);
    expect(run(args('--out', join(out, 'f.zip'), '--assets', join(root, 'nowhere'))).stderr).toContain('run python -m rifteye_ml.web_assets first');
    rmSync(join(assets, 'thumbs', 'TST-003.jpg'));
    expect(run(args('--out', join(out, 'f.zip'))).stderr).toContain('2 thumbnails for 3 printings');
    writeFileSync(join(assets, 'thumbs', 'TST-003.jpg'), 'jpeg TST-003');
  });

  it('refuses a float16 embedder that its gallery has no hash for: the pairing could not be checked', () => {
    const index = join(assets, 'gallery', 'index.json');
    const good = readFileSync(index, 'utf8');
    writeFileSync(index, JSON.stringify({ ...JSON.parse(good), fp16_sha256: null }));
    expect(run(args('--out', join(out, 'g.zip'))).stderr).toContain('has no hash of the fp16 embedder');
    writeFileSync(index, good);
  });

  it('refuses options that make no sense', () => {
    expect(run(['--precisions', 'int8', '--size-only']).stderr).toContain('--precisions');
    expect(run(['--precisions', 'detector:fp32', '--size-only']).stderr).toContain('--precisions'); // no embedder
    expect(run(['--precisions', 'detector:fp32,embedder:int8', '--size-only']).stderr).toContain('--precisions');
    expect(run(['--precisions', 'tracker:fp32,embedder:fp16', '--size-only']).stderr).toContain('--precisions');
    expect(run(['--runtime', 'gpu', '--size-only']).stderr).toContain('--runtime');
    expect(run(['--part-bytes', '10', '--size-only']).stderr).toContain('--part-bytes');
    expect(run([]).stderr).toContain('usage');
  });
});
