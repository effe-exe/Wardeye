import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, expect, test, type BrowserContext, type Page } from '@playwright/test';
import { mirror, standInFiles, writeFiles } from './standins';

const BENCH = fileURLToPath(new URL('../', import.meta.url));
const DIST = join(BENCH, 'dist');
// the extension's own manifest: the version the summary's first line names and the icons the build has to ship are read
// from it, never written out here (a release changes them, a test must not have to)
const MANIFEST = JSON.parse(readFileSync(join(BENCH, 'src', 'manifest.json'), 'utf8')) as { version: string; icons: Record<string, string> };
const VERSION = MANIFEST.version;

test.beforeAll(() => {
  // the tests load the built extension: build it when nobody has (npm run build does)
  if (!existsSync(join(DIST, 'bench.html'))) execFileSync('node', ['build.mjs'], { cwd: BENCH, stdio: 'inherit' });
});

/** The summary's line for a row: "<model> | <precision> | <runtime> | ...". */
const rowOf = (summary: string, model: string, precision: string, runtime: string): string =>
  summary.split('\n').find((l) => l.startsWith(`${model} | ${precision} | ${runtime} |`)) ?? '';

interface Loaded {
  root: string;
  context: BrowserContext;
  page: Page;
}

/** Loads the built extension, with stand-in models in its models/ folder (or none), and finds the page its worker opens. */
async function load(withModels: boolean): Promise<Loaded> {
  const root = mkdtempSync(join(tmpdir(), 'rifteye-bench-'));
  // a throwaway copy of dist (the bench reads models/ from its own package)
  const ext = join(root, 'ext');
  mirror(DIST, ext);
  if (withModels) writeFiles(join(ext, 'models'), standInFiles());
  const context = await chromium.launchPersistentContext(join(root, 'profile'), {
    headless: true,
    args: [
      `--disable-extensions-except=${ext}`,
      `--load-extension=${ext}`,
      // a software WebGPU adapter (SwiftShader), so that the WebGPU rows can run without a GPU
      '--enable-unsafe-webgpu',
      '--use-webgpu-adapter=swiftshader',
    ],
    // Playwright's default headless shell cannot load extensions; its full Chromium (channel) can
    ...(process.env.RIFTEYE_CHROMIUM ? { executablePath: process.env.RIFTEYE_CHROMIUM } : { channel: 'chromium' }),
  });
  const find = () => context.pages().find((p) => p.url().endsWith('/bench.html'));
  // the extension's worker opens the page when it is installed
  await expect.poll(() => find() !== undefined, { timeout: 30_000 }).toBe(true);
  return { root, context, page: find()! };
}

/** "chrome-extension://<id>": the extension's origin (URL.origin is "null" for this scheme). */
const originOf = (page: Page): string => /^chrome-extension:\/\/[a-p]{32}/.exec(page.url())![0];

async function unload(l: Loaded | null): Promise<void> {
  await l?.context.close();
  if (l) rmSync(l.root, { recursive: true, force: true });
}

test('the bench extension runs stand-in models on every runtime this browser has, and reports them', async () => {
  test.setTimeout(420_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load(true);
    const { context, page } = loaded;
    await expect(page).toHaveTitle('Wardeye bench');
    await expect(page.locator('h1')).toHaveText('Wardeye bench');
    expect(page.url()).toMatch(/^chrome-extension:\/\/[a-p]{32}\/bench\.html$/);
    // the build puts the mark and every icon the manifest names next to the page, the header shows the mark, and the
    // brand's three fonts (inlined in bench.css) load under the page's content security policy
    for (const file of ['mark.svg', ...Object.values(MANIFEST.icons)]) expect(await page.evaluate(async (f) => (await fetch(f)).ok, file), file).toBe(true);
    expect(await page.locator('h1 img').evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
    await expect
      .poll(() => page.evaluate(() => [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family).sort()))
      .toEqual(['Inter', 'JetBrains Mono', 'Space Grotesk']);
    // the manifest's COOP and COEP make the page cross-origin isolated: WASM threads can work
    expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
    expect(await page.evaluate(() => typeof SharedArrayBuffer)).toBe('function');

    // the toolbar button opens the page too (its click, sent to the worker)
    const benchPages = () => context.pages().filter((p) => p.url().endsWith('/bench.html'));
    const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
    await worker.evaluate(() => (chrome.action.onClicked as unknown as { dispatch(tab: object): void }).dispatch({}));
    await expect.poll(() => benchPages().length).toBe(2);
    await benchPages().find((p) => p !== page)!.close();

    await page.click('#run');
    await expect(page.locator('#status')).toContainText('Done in', { timeout: 400_000 });
    const summary = await page.inputValue('#summary');
    console.log(summary);

    // environment: the page and, from the rows, the workers (threads need isolated workers too)
    const first = new RegExp(`^Wardeye bench ${VERSION.replaceAll('.', '\\.')} \\| onnxruntime-web 1\\.\\d+\\.\\d+ \\| \\d{4}-\\d\\d-\\d\\dT.* \\| done in \\d+ s$`, 'm');
    expect(summary).toMatch(first);
    expect(summary).toContain('crossOriginIsolated yes');
    expect(summary).toMatch(/wasm threads asked [1-4] /);
    expect(summary).toMatch(/^workers: crossOriginIsolated yes \| wasm threads used [1-4]$/m);
    expect(summary).toMatch(/^WebGPU: (yes|no \()/m);

    // WASM: timings for every batch, and a passing check
    const detector = rowOf(summary, 'standin-detector', 'fp32', 'wasm');
    expect(detector).toMatch(/init [\d.]+ \| load [\d.]+ \([\d.]+ MB, sha256 [0-9a-f]{12}\) \| first [\d.]+ \| b1 run [\d.]+\/[\d.]+ item [\d.]+\/[\d.]+ first [\d.]+ b2 run .* b4 run .*\| check maxabs y .* z .* PASS$/);
    expect(rowOf(summary, 'standin-embedder', 'fp32', 'wasm')).toMatch(/\| b1 run .* b4 run .*\| check cosine 1\.000000 >= 0\.9999 PASS$/);
    expect(rowOf(summary, 'standin-embedder', 'fp16', 'wasm')).toMatch(/\| check cosine (1\.000000|0\.9999\d+) >= 0\.999 PASS$/);

    // a variant that is not in the folder is said so, once; a wrong check fails; a broken file is an error row
    expect(summary).toMatch(/^standin-detector \| fp16 \| - \| not included$/m);
    expect(rowOf(summary, 'standin-wrong', 'fp32', 'wasm')).toMatch(/\| check cosine -1\.000000 < 0\.9999 \[2 of 2 rows under\] FAIL$/);
    expect(rowOf(summary, 'standin-broken', 'fp32', 'wasm')).toContain('not checked');
    expect(summary).toMatch(/^ {2}! load: .+/m);
    // a manifest that cannot be read is one row of error, said with the field at fault, and does not stop the others
    expect(summary).toMatch(/^standin-invalid \| - \| - \| .*\n {2}! standin-invalid\.bench\.json could not be read: standin-invalid: variants: expected a non-empty list$/m);
    expect(summary).toMatch(/NOT A WHOLE FRAME: .*standin-invalid\.bench\.json could not be read/); // the estimate says it leaves it out

    // the per-frame estimate: 3 detector tiles and 6 embedder crops at their best batches, WASM alone
    expect(summary).toMatch(
      /^ {2}wasm only: wasm: standin-detector fp32 3 items = \d+ x b\d+ @ [\d.]+ = [\d.]+ ms \+ standin-embedder (fp32|fp16) 6 items = \d+ x b\d+ @ [\d.]+ = [\d.]+ ms = [\d.]+ ms\/frame -> [\d.]+ reads\/s$/m,
    );

    // the table shows the rows too
    await expect(page.locator('#rows tr', { hasText: 'standin-detector' }).first()).toBeVisible();
    await expect(page.locator('#rows tr.error')).toHaveCount(4); // the broken model, on each runtime, and the unreadable manifest
    await expect(page.locator('#rows tr.detail', { hasText: '! load:' })).toHaveCount(3);

    // WebGPU: with an adapter the rows run and pass; fp16 is skipped unless the adapter has shader-f16
    if (/^WebGPU: yes/m.test(summary)) {
      const f16 = /shader-f16 yes/.test(summary);
      for (const runtime of ['webgpu', 'webgpu-jsep']) {
        expect(rowOf(summary, 'standin-detector', 'fp32', runtime), runtime).toMatch(/\| check maxabs .* PASS$/);
        expect(rowOf(summary, 'standin-embedder', 'fp32', runtime), runtime).toMatch(/\| check cosine 1\.000000 >= 0\.9999 PASS$/);
        const half = rowOf(summary, 'standin-embedder', 'fp16', runtime);
        if (f16) expect(half, runtime).toMatch(/PASS$/);
        else expect(half, runtime).toContain('no shader-f16');
      }
    } else {
      console.log('no WebGPU adapter in this browser: the WebGPU rows were skipped');
      expect(rowOf(summary, 'standin-detector', 'fp32', 'webgpu')).toMatch(/\| WebGPU|\| requestAdapter|\| navigator/);
    }

    // Copy results puts the summary on the clipboard; it is read back on an ordinary page, which can be granted clipboard-read
    await page.click('#copy');
    await expect(page.locator('#copied')).toContainText('Copied');
    await context.route('https://clip.test/**', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>clip</title>' }));
    await context.grantPermissions(['clipboard-read'], { origin: 'https://clip.test' });
    const other = await context.newPage();
    await other.goto('https://clip.test/');
    expect(await other.evaluate(() => navigator.clipboard.readText())).toBe(summary);
  } finally {
    await unload(loaded);
  }
});

test('without a models folder the page says so instead of doing nothing', async () => {
  test.setTimeout(90_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load(false);
    const { page } = loaded;
    await page.click('#run');
    await expect(page.locator('#status')).toContainText('Done in', { timeout: 60_000 });
    expect(await page.inputValue('#summary')).toContain('PROBLEM: models/index.json could not be read');
    await expect(page.locator('#frame')).toContainText('PROBLEM: models/index.json could not be read');
    await expect(page.locator('#run')).toBeEnabled();
  } finally {
    await unload(loaded);
  }
});

test('Stop ends the run at once, the summary says it is partial, and Run can start again', async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load(true);
    const { page } = loaded;
    // WASM only, and the run has to be under way before it is stopped
    await page.goto(`${originOf(page)}/bench.html?runtimes=wasm`);
    await page.click('#run');
    await expect(page.locator('#status')).toContainText('Row 1 of');
    await expect(page.locator('#stop')).toBeEnabled();
    await page.click('#stop');
    await expect(page.locator('#status')).toContainText('Stopped in', { timeout: 30_000 });
    expect(await page.inputValue('#summary')).toContain('STOPPED, results are partial');
    await expect(page.locator('#run')).toBeEnabled();
    await expect(page.locator('#stop')).toBeDisabled();

    await page.click('#run');
    await expect(page.locator('#status')).toContainText('Done in', { timeout: 120_000 });
    const summary = await page.inputValue('#summary');
    expect(summary).not.toContain('STOPPED');
    expect(rowOf(summary, 'standin-detector', 'fp32', 'wasm')).toMatch(/PASS$/);
    expect(summary).not.toMatch(/\| webgpu/); // ?runtimes=wasm
  } finally {
    await unload(loaded);
  }
});

test('?quick=1 times one run per batch, says so, and still checks', async () => {
  test.setTimeout(120_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load(true);
    const { page } = loaded;
    await page.goto(`${originOf(page)}/bench.html?runtimes=wasm&quick=1`);
    await page.click('#run');
    await expect(page.locator('#status')).toContainText('Done in', { timeout: 100_000 });
    const summary = await page.inputValue('#summary');
    expect(summary).toContain('QUICK MODE (?quick=1): one warm-up and one timed run per batch');
    expect(rowOf(summary, 'standin-detector', 'fp32', 'wasm')).toMatch(/\| check maxabs .* PASS$/);
    await expect(page.locator('#rows tr', { hasText: 'standin-detector' }).first()).toContainText('only 1 runs');
  } finally {
    await unload(loaded);
  }
});
