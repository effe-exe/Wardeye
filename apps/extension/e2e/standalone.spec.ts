import { expect, test } from '@playwright/test';
import { loadedFonts, rgb, styleOf } from './brand';
import { documentIds, load, offscreenDocuments, twitch, unload, type Loaded } from './harness';
import { routeTwitch } from './twitch';

test('the engine in the extension reads the player: an offscreen document, WebGPU, and its board drawn', async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load();
    const { context } = loaded;
    const page = await twitch(context);
    await page.goto('https://www.twitch.tv/videos/12345');

    // a tab connecting makes the engine's document, and the first frames say what it is doing
    await expect.poll(() => offscreenDocuments(context), { timeout: 20_000 }).toBe(1);
    const badge = page.locator('.rifteye-badge-main');
    await expect(badge).toContainText(/Wardeye: (starting the engine|loading the models|reading the gallery|finding the table)/, { timeout: 30_000 });

    // ... and then the board: the layout was found from the first seconds of the video, and both blocks are boxed
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']); // the other is unsure: no label
    await expect(page.locator('.rifteye-badge-main')).toHaveText(/^Wardeye · 1 card named · \d+\.\d reads\/s$/);
    // it ran on WebGPU (SwiftShader has no shader-f16: both models in float32), and the badge's second line says where a frame's time goes
    await expect(page.locator('.rifteye-badge-detail')).toHaveText(/^WebGPU · detector fp32 · embedder fp32 · decode [\d.]+ · detect [\d.]+ · embed [\d.]+ · track [\d.]+ · total [\d.]+ ms · layout standin$/);
    // the timings line is in JetBrains Mono at the micro size, dim; the name and status above it in Space Grotesk and Inter
    const detail = await styleOf(page, '.rifteye-badge-detail', ['font-family', 'font-size', 'color']);
    expect(detail).toMatchObject({ 'font-size': '10px', color: rgb('dim') });
    expect(detail['font-family']).toMatch(/^"?JetBrains Mono"?,/);
    await expect.poll(() => loadedFonts(page)).toEqual(expect.arrayContaining(['Inter', 'JetBrains Mono', 'Space Grotesk']));

    // the overlay sits on the picture: the gold block, centred at (130, 142) of 640 x 360, is at (195, 213) of the 960 x 540 player
    const b = (await boxes.first().boundingBox())!;
    expect(Math.abs(b.x + b.width / 2 - 195)).toBeLessThan(8);
    expect(Math.abs(b.y + b.height / 2 - 213)).toBeLessThan(8);

    // the hover card: the card's picture from the package (no runner, no network), and the guesses of the unsure block
    await boxes.first().hover();
    const card = page.locator('.rifteye-card');
    await expect(card).toContainText('Test Unit');
    await expect(card).toContainText(/Confidence [01]\.\d\d/);
    const painted = () =>
      page.evaluate(() => {
        const c = document.querySelector('.rifteye-card canvas') as HTMLCanvasElement | null;
        return c ? c.getContext('2d')!.getImageData(c.width / 2, c.height / 2, 1, 1).data[3] : -1;
      });
    await expect.poll(painted, { timeout: 5000 }).toBe(255);
    await boxes.nth(1).hover();
    await expect(card).toContainText('Not sure yet. Best guesses:');
    await expect(card).toContainText(/Guess (Two|Three) · \d+%/);

    // a second tab, another video: its own layout and board on the same engine (one document, the tab that sends frames is answered)
    const second = await context.newPage();
    await second.goto('https://www.twitch.tv/videos/999');
    await expect(second.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    expect(await offscreenDocuments(context)).toBe(1);
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2);
  } finally {
    await unload(loaded);
  }
});

test('a worker put to sleep and woken again goes on with the same engine document', async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load();
    const { context } = loaded;
    const page = await twitch(context, 30);
    await page.goto('https://www.twitch.tv/videos/12345');
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2, { timeout: 90_000 });
    const before = await documentIds(context);
    expect(before).toHaveLength(1);

    // the browser stops the extension's worker when it has been idle; here it is told to (a session on an extension page
    // may stop the extension's workers), and its states are watched
    const id = /^chrome-extension:\/\/([a-p]{32})\//.exec(context.serviceWorkers()[0]!.url())![1]!;
    const admin = await context.newPage();
    await admin.goto(`chrome-extension://${id}/manifest.json`);
    const cdp = await context.newCDPSession(admin);
    const seen: string[] = [];
    cdp.on('ServiceWorker.workerVersionUpdated', (e: { versions: { runningStatus: string }[] }) => seen.push(...e.versions.map((v) => v.runningStatus)));
    await cdp.send('ServiceWorker.enable');
    await cdp.send('ServiceWorker.stopAllWorkers');
    await expect.poll(() => seen.includes('stopped'), { timeout: 20_000 }).toBe(true);

    // the overlay's port closed, so it connects again after a second, which wakes the worker; the frames go on, read by
    // the same engine: the board is drawn, its second block is followed as it drifts, and no other document was made
    await expect.poll(() => seen.slice(seen.lastIndexOf('stopped')).includes('running'), { timeout: 30_000 }).toBe(true);
    const at = await boxes.nth(1).getAttribute('points');
    await expect.poll(async () => boxes.nth(1).getAttribute('points'), { timeout: 20_000 }).not.toBe(at);
    await expect(boxes).toHaveCount(2);
    expect(await documentIds(context)).toEqual(before);
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']);
  } finally {
    await unload(loaded);
  }
});

test('a package that says wasm runs the engine on plain WASM', async () => {
  test.setTimeout(150_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ package: { runtime: 'wasm' } });
    const page = await twitch(loaded.context);
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('.rifteye-badge-detail')).toHaveText(/^WASM · detector fp32 · embedder fp32 · /);
  } finally {
    await unload(loaded);
  }
});

test('a float16 embedder on a GPU without shader-f16 runs on WASM, the detector in float32', async () => {
  test.setTimeout(150_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ package: { embedderFp16: true } }); // the way the private build ships its models; the software adapter has no shader-f16
    const page = await twitch(loaded.context);
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('.rifteye-badge-detail')).toHaveText(/^WASM · detector fp32 · embedder fp16 · /);
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']); // the float16 graph names the block as the float32 one does
  } finally {
    await unload(loaded);
  }
});

test('a browser with no WebGPU adapter leaves the frames to the live runner, and says why', async () => {
  test.setTimeout(90_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ gpu: false });
    const { context } = loaded;
    const said: string[] = [];
    context.on('console', (m) => {
      if (m.text().includes('Wardeye: no engine')) said.push(m.text());
    });
    const page = await context.newPage();
    await routeTwitch(context, null); // a Twitch page with no video: no frame is sent anywhere
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect.poll(() => offscreenDocuments(context), { timeout: 20_000 }).toBe(1);
    await expect.poll(() => said.length, { timeout: 20_000 }).toBeGreaterThan(0);
    expect(said[0]).toContain('this browser has no WebGPU adapter');
  } finally {
    await unload(loaded);
  }
});

test('the public build (no models) makes no offscreen document and asks nothing of one', async () => {
  test.setTimeout(60_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ package: null });
    const { context } = loaded;
    const page = await context.newPage();
    await routeTwitch(context, null);
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect(page.locator('.rifteye-badge')).toBeHidden(); // no video: nothing on the page yet
    await page.waitForTimeout(2500);
    expect(await offscreenDocuments(context)).toBe(0);
  } finally {
    await unload(loaded);
  }
});
