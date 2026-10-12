import { expect, test, type BrowserContext } from '@playwright/test';
import { MODELS_URL, fakeModels, type FakeModels } from './fake-models';
import { fakeRiot, type FakeRiot } from './fake-riot';
import { documentIds, load, offscreenDocuments, twitch, unload, type Loaded } from './harness';
import { standInModels } from './standins';

/** Closes the engine document, as the browser does after five minutes without a frame: the next frame starts the engine anew. */
async function closeEngine(context: BrowserContext): Promise<void> {
  const sw = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
  await sw.evaluate(() => chrome.offscreen.closeDocument());
}

const RULES = '--host-resolver-rules=';
/** Chromium's arguments for both fakes at once: one list of host mappings (a second flag would replace the first). */
function bothFakes(models: FakeModels, riot: FakeRiot): string[] {
  const rules = [...models.args, ...riot.args].filter((a) => a.startsWith(RULES)).map((a) => a.slice(RULES.length));
  const rest = new Set([...models.args, ...riot.args].filter((a) => !a.startsWith(RULES)));
  return [`${RULES}${rules.join(', ')}`, ...rest];
}

/** The store build, with the models downloaded from the fake host. It is the build that downloads them, and it has no local
 * runner to fall back on (the dev build would draw another test's fake runner's boxes). Riot's gallery is a fake, empty. */
async function loadStore(host: FakeModels, riot: FakeRiot): Promise<Loaded> {
  return load({ store: true, package: { remote: MODELS_URL }, args: bothFakes(host, riot) });
}

test('the models are downloaded the first time, with no permission, then read from the browser\'s cache', async () => {
  test.setTimeout(240_000);
  let host: FakeModels | null = null;
  let riot: FakeRiot | null = null;
  let loaded: Loaded | null = null;
  try {
    host = await fakeModels(standInModels());
    riot = await fakeRiot([]);
    loaded = await loadStore(host, riot);
    const { context } = loaded;
    const page = await twitch(context, 30);
    await page.goto('https://www.twitch.tv/videos/12345');
    // the package holds no model: the engine downloads both, checks them, and reads the table
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    expect(host.seen.sort()).toEqual(['/v1/standin-detector.onnx', '/v1/standin-embedder.onnx']);
    expect(host.cdn.sort()).toEqual(['/standin-detector.onnx', '/standin-embedder.onnx']); // through the redirect, as on Hugging Face

    // the engine starts again (its document closed): the models come from the cache, and nothing is downloaded
    const before = await documentIds(context);
    await closeEngine(context);
    await expect.poll(() => offscreenDocuments(context)).toBe(0);
    await page.goto('https://www.twitch.tv/videos/777');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    const after = await documentIds(context);
    expect(after).toHaveLength(1);
    expect(after[0]).not.toBe(before[0]); // a new engine document, with a new engine in it
    expect(host.seen).toHaveLength(2);
    expect(host.cdn).toHaveLength(2);
  } finally {
    await unload(loaded);
    await host?.close();
    await riot?.close();
  }
});

test('a downloaded model that is not the file the package names is refused, kept nowhere, and the reason is said', async () => {
  test.setTimeout(120_000);
  let host: FakeModels | null = null;
  let riot: FakeRiot | null = null;
  let loaded: Loaded | null = null;
  try {
    host = await fakeModels(standInModels());
    const detector = host.files.get('standin-detector.onnx')!;
    host.files.set('standin-detector.onnx', detector.map((b) => b ^ 1)); // the same size, other bytes
    riot = await fakeRiot([]);
    loaded = await loadStore(host, riot);
    const { context } = loaded;
    const said: string[] = [];
    context.on('console', (m) => {
      if (m.text().includes('Wardeye: no engine')) said.push(m.text());
    });
    const page = await twitch(context);
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect.poll(() => said.length, { timeout: 60_000 }).toBeGreaterThan(0);
    expect(said[0]).toContain('standin-detector.onnx is not the file this version of Wardeye was made with');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(0);
  } finally {
    await unload(loaded);
    await host?.close();
    await riot?.close();
  }
});
