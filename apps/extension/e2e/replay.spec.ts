import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, type Plugin } from 'esbuild';
import { chromium, expect, test } from '@playwright/test';
import { standInFiles } from './standins';
import { compareRuns, jsonLines, summarize, summarizeDiag, type WantStep } from './replay-compare';
import type { Replay } from './replay-page';

// The whole engine on the LA final's 240 frames, in Chromium: the real detector and embedder on onnxruntime-web's
// WASM build (float32), the recogniser, layout la-rq, through the engine worker as the extension runs it; every
// step's state and events are compared with the Python reference. It needs the private frames, models and reference,
// so it skips unless RIFTEYE_M3 names the private folder (~/rifteye-data/m3), and it takes minutes:
//
//   RIFTEYE_M3=~/rifteye-data/m3 RIFTEYE_CHROMIUM=/opt/pw-browsers/chromium npx playwright test apps/extension/e2e/replay.spec.ts
//
// RIFTEYE_REPLAY_GALLERY=reference (default) reads the gallery Python ran with, as float32; shipped reads the private
// build's own (web_assets.py, float16). RIFTEYE_REPLAY_FRAMES=n runs the first n frames only. RIFTEYE_REPLAY_STANDIN=1
// runs the stand-in parts and models instead of the engine (to try the harness itself). RIFTEYE_REPLAY_LAYOUT=auto
// leaves the layout to be found from the footage (the frames it is found from are not read as a board, so the steps
// are not compared with the reference; the layout found is printed).

const require = createRequire(import.meta.url);
const EXT = fileURLToPath(new URL('../', import.meta.url));
// no default location: the private folder is named, and everything else (the models, the outputs) is found beside it
const M3 = process.env.RIFTEYE_M3;
const DATA = process.env.RIFTEYE_DATA ?? (M3 ? dirname(M3) : '');
const GALLERY = process.env.RIFTEYE_REPLAY_GALLERY ?? 'reference';
const STANDIN = process.env.RIFTEYE_REPLAY_STANDIN === '1';
const LIMIT = Number(process.env.RIFTEYE_REPLAY_FRAMES ?? 0);
const AUTO = process.env.RIFTEYE_REPLAY_LAYOUT === 'auto';
const ORT_WASM = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const TYPES: Record<string, string> = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.wasm': 'application/wasm', '.jpg': 'image/jpeg' };

type Served = { path: string } | { body: string | Buffer };

const standInPlugin: Plugin = {
  name: 'stand-in-parts',
  setup(b) {
    b.onResolve({ filter: /\/parts-engine$/ }, () => ({ path: join(EXT, 'e2e', 'parts-standin.ts') }));
  },
};

const sha256 = (path: string): string => createHash('sha256').update(readFileSync(path)).digest('hex');

/** A page's files served at 127.0.0.1 with the headers that make it cross-origin isolated (WASM threads). */
function serve(files: Map<string, Served>): Promise<{ server: Server; url: string }> {
  const server = createServer((req, res) => {
    const name = decodeURIComponent((req.url ?? '/').split('?')[0]!);
    const f = files.get(name);
    const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Resource-Policy': 'same-origin' };
    if (!f) return void res.writeHead(404, headers).end();
    const type = TYPES[extname(name)] ?? 'application/octet-stream';
    if ('body' in f) return void res.writeHead(200, { ...headers, 'Content-Type': type, 'Content-Length': Buffer.byteLength(f.body) }).end(f.body);
    res.writeHead(200, { ...headers, 'Content-Type': type, 'Content-Length': statSync(f.path).size });
    createReadStream(f.path).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${(server.address() as { port: number }).port}/` })));
}

test.skip(!M3, 'needs the private frames, models and reference: set RIFTEYE_M3 (the folder with frames/ and fixtures/)');

test('the whole engine on the LA final agrees with Python, step for step', async () => {
  test.setTimeout(3 * 60 * 60_000);
  const m3 = M3!;
  const framesDir = join(m3, 'frames', 'la-final');
  const manifest = JSON.parse(readFileSync(join(framesDir, 'frames.json'), 'utf8')) as { fps: number; layout: string; frames: { file: string; t: number; rgb_sha256: string }[] };
  const frames = LIMIT > 0 ? manifest.frames.slice(0, LIMIT) : manifest.frames;
  const root = mkdtempSync(join(tmpdir(), 'rifteye-replay-'));
  let server: Server | null = null;
  try {
    // the site: the real engine worker (or the stand-ins), the page, onnxruntime-web's WASM build, the models and the gallery
    await build({
      entryPoints: [join(EXT, 'src', 'engine-wasm.ts')], bundle: true, format: 'esm', target: 'es2022', outfile: join(root, 'engine-wasm.js'),
      conditions: ['onnxruntime-web-use-extern-wasm'], plugins: STANDIN ? [standInPlugin] : [], logLevel: 'warning',
    });
    await build({ entryPoints: [join(EXT, 'e2e', 'replay-page.ts')], bundle: true, format: 'esm', target: 'es2022', outfile: join(root, 'replay.js'), logLevel: 'warning' });
    const files = new Map<string, Served>([
      ['/replay.html', { body: '<!doctype html><meta charset="utf-8"><title>replay</title><script type="module" src="replay.js"></script>' }],
      ['/replay.js', { path: join(root, 'replay.js') }],
      ['/engine-wasm.js', { path: join(root, 'engine-wasm.js') }],
    ]);
    for (const f of ORT_WASM) files.set(`/ort/${f}`, { path: require.resolve(`onnxruntime-web/${f}`) });
    for (const f of frames) files.set(`/frames/${f.file}`, { path: join(framesDir, f.file) });
    const modelsDir = join(DATA, 'models', 'onnx');
    const reference = join(m3, 'fixtures', 'recognizer');
    if (STANDIN) {
      for (const [path, content] of Object.entries(standInFiles({ runtime: 'wasm' }))) files.set(`/${path}`, { body: typeof content === 'string' ? content : Buffer.from(content) });
    } else {
      const embedder = join(modelsDir, 'embedder-v1.onnx');
      files.set('/models/detector-v0.onnx', { path: join(modelsDir, 'detector-v0.onnx') });
      files.set('/models/embedder-v1.onnx', { path: embedder });
      files.set('/standalone.json', { body: JSON.stringify({ format: 1, runtime: 'wasm', detector: { id: 'detector-v0', fp32: 'models/detector-v0.onnx' }, embedder: { id: 'embedder-v1', fp32: 'models/embedder-v1.onnx' }, data: 'data/', ...(AUTO ? {} : { layout: manifest.layout }), fps: manifest.fps, threads: 2, trace: true }) });
      if (GALLERY === 'shipped') {
        const assets = join(m3, 'web-assets');
        for (const f of ['catalog.json', 'gallery/index.json']) files.set(`/data/${f}`, { path: join(assets, f) });
        const index = JSON.parse(readFileSync(join(assets, 'gallery', 'index.json'), 'utf8')) as { levels: number[] };
        for (const l of index.levels) files.set(`/data/gallery/L${l}.bin`, { path: join(assets, 'gallery', `L${l}.bin`) });
      } else {
        // the gallery Python ran with: float32 levels, the catalogue it read, and an index that says which embedder made them
        const rows = JSON.parse(readFileSync(join(reference, 'rows.json'), 'utf8')) as { printing_id: string }[];
        const levels = [120, 140, 160].filter((l) => existsSync(join(reference, 'levels', `${l}.bin`)));
        const index = { format: 1, encoder: 'onnx:embedder-v1', model: 'embedder-v1', sha256: sha256(embedder), fp16_sha256: null, dim: 256, dtype: 'float32', levels, rows: rows.map((r) => r.printing_id) };
        files.set('/data/catalog.json', { path: join(reference, 'rows.json') });
        files.set('/data/gallery/index.json', { body: JSON.stringify(index) });
        for (const l of levels) files.set(`/data/gallery/L${l}.bin`, { path: join(reference, 'levels', `${l}.bin`) });
      }
    }
    const withReference = !STANDIN && !AUTO && existsSync(join(reference, 'steps.jsonl')) && existsSync(join(reference, 'embeds.bin'));
    if (withReference) for (const f of ['steps.jsonl', 'embeds.json', 'embeds.bin']) files.set(`/reference/${f}`, { path: join(reference, f) });
    files.set('/replay.json', { body: JSON.stringify({ frames, attempt: { runtime: 'wasm', detector: 'fp32', embedder: 'fp32' }, worker: 'engine-wasm.js', reference: withReference }) });
    const served = await serve(files);
    server = served.server;

    const browser = await chromium.launch({ headless: true, ...(process.env.RIFTEYE_CHROMIUM ? { executablePath: process.env.RIFTEYE_CHROMIUM } : {}) });
    try {
      const page = await browser.newPage();
      page.on('console', (m) => {
        if (m.type() === 'error' || m.type() === 'warning') console.log(`[page ${m.type()}] ${m.text().slice(0, 300)}`);
      });
      await page.goto(`${served.url}replay.html`);
      const read = () => page.evaluate(() => ({ progress: window.__replay.progress, total: window.__replay.total, done: window.__replay.done, error: window.__replay.error, log: window.__replay.log }));
      let last = -1;
      let logged = 0;
      for (;;) {
        const s = await read();
        if (s.error) throw new Error(`the replay page failed: ${s.error}`);
        for (const line of s.log.slice(logged)) console.log(`replay page: ${line}`);
        logged = s.log.length;
        if (s.progress !== last && (s.progress % 10 === 0 || s.done)) console.log(`replay: ${s.progress} of ${s.total} frames`);
        last = s.progress;
        if (s.done) break;
        await page.waitForTimeout(5000);
      }
      const replay = (await page.evaluate(() => window.__replay)) as Replay;
      expect(replay.parity.different, replay.parity.first ?? '').toBe(0);
      expect(replay.results).toHaveLength(frames.length);
      console.log(`decode: ${replay.parity.same} of ${frames.length} frames decode to Pillow's RGB (cross-origin isolated: ${replay.isolated})`);

      const out = join(m3, 'fixtures', 'extension');
      mkdirSync(out, { recursive: true });
      const outFile = join(out, `replay-${STANDIN ? 'standin' : GALLERY}.json`);
      writeFileSync(outFile, JSON.stringify(replay.results));
      const ms = replay.results.map((r) => r.ms).sort((a, b) => a - b);
      console.log(`ms a frame (whole, through the worker): median ${ms[Math.floor(ms.length / 2)]!.toFixed(0)}, p90 ${ms[Math.floor(ms.length * 0.9)]!.toFixed(0)}, max ${ms.at(-1)!.toFixed(0)}; results in ${outFile}`);
      const parts = ['decode', 'detect', 'embed', 'track', 'total'] as const;
      const timings = replay.results.map((r) => r.timing as Record<(typeof parts)[number], number> | undefined).filter((t): t is Record<(typeof parts)[number], number> => t !== undefined);
      const mean = (k: (typeof parts)[number]): string => (timings.reduce((a, t) => a + t[k], 0) / Math.max(1, timings.length)).toFixed(0);
      console.log(`the frame's ms, mean over ${timings.length} frames: ${parts.map((k) => `${k} ${mean(k)}`).join(', ')}`);

      const stepsFile = join(reference, 'steps.jsonl');
      if (AUTO) {
        const found = replay.results.map((r) => r.state as { status?: string; layout?: { name: string; table: number[] } }).find((st) => st.layout);
        console.log(`layout found from the footage: ${found ? JSON.stringify(found.layout) : 'none in these frames'} (the preset la-rq is ${JSON.stringify(manifest.layout)}, table 0.19 0.06 0.81 1)`);
      } else if (!STANDIN && existsSync(stepsFile)) {
        const want = jsonLines<WantStep>(readFileSync(stepsFile, 'utf8')).slice(0, frames.length);
        const runs = [0, 1e-6, 0.11].map((tol) => compareRuns(replay.results, want, tol));
        const text = summarize(runs, replay.results.length, want.length);
        console.log(text);
        const diag = summarizeDiag(replay.diag);
        console.log(diag);
        writeFileSync(join(out, `replay-${GALLERY}.summary.txt`), `${text}\n${diag}\n`);
        if (process.env.RIFTEYE_REPLAY_STRICT === '1') expect(runs[0]!.identical).toBe(frames.length);
      }
    } finally {
      await browser.close();
    }
  } finally {
    server?.close();
    rmSync(root, { recursive: true, force: true });
  }
});
