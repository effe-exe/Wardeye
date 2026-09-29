import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { fakeRiot, item, makeJpeg, pictureOf, type FakeRiot } from './fake-riot';
import { load, offscreenDocuments, twitch, unload, type Loaded } from './harness';

// The Chrome Web Store build (node build.mjs --store): standalone only, with no card name, type or picture in its package.
// Riot's card list and its pictures are a fake (fake-riot.ts): made-up cards, pictures made in the browser, no real network.

// The stand-in gallery (standins.ts) holds TST-001 (the gold block), TST-002 and TST-003 (which the blue block cannot tell apart).
// The fake list names the first two, differently from anything the package could say, and does not name TST-003; it also holds a
// printing the gallery does not, one with no picture, and enough filler for a second page.
const LIST = [
  item('TST-001/100', 'Fed Unit', { subtitle: 'The Golden' }), // a champion unit: its subtitle is in its name
  item('TST-002/100', 'Fed Guess Two'),
  item('TST-050/100', 'Not In The Gallery'),
  item('TST-051/100', 'No Picture', { cardImage: {} }),
  ...Array.from({ length: 230 }, (_, i) => item(`ZZZ-${String(i + 1).padStart(3, '0')}/230`, `Filler ${i + 1}`)),
];
const FED = 'Fed Unit, The Golden';
const MAGENTA = [224, 32, 224];
const CYAN = [32, 224, 224];

/** The colour at the middle of a canvas of the hover card: [r, g, b, a], or null when there is none. */
async function middle(canvas: Locator): Promise<number[] | null> {
  return canvas.evaluate((c) => {
    const cv = c as HTMLCanvasElement;
    return Array.from(cv.getContext('2d')!.getImageData(cv.width / 2, cv.height / 2, 1, 1).data);
  });
}
const near = (got: number[] | null, want: number[]): boolean => got !== null && got[3] === 255 && want.every((w, i) => Math.abs(got[i]! - w) < 40);

async function open(loaded: Loaded, riot: FakeRiot): Promise<Page> {
  const page = await twitch(loaded.context);
  riot.pictures.set(pictureOf('TST-001/100'), await makeJpeg(page, `rgb(${MAGENTA.join(',')})`));
  riot.pictures.set(pictureOf('TST-002/100'), await makeJpeg(page, `rgb(${CYAN.join(',')})`));
  await page.goto('https://www.twitch.tv/videos/12345');
  // the recorded video is short and a test that waits a minute for a retry is not: the player starts it again when it ends
  await page.evaluate(() => {
    setInterval(() => {
      const v = document.querySelector('video');
      if (v?.ended) {
        v.currentTime = 0;
        void v.play();
      }
    }, 300);
  });
  return page;
}

const feedRequests = (riot: FakeRiot) => riot.seen.filter((s) => s.host === 'content.publishing.riotgames.com').map((s) => s.url);
const pictureRequests = (riot: FakeRiot) => riot.seen.filter((s) => s.host === 'cmsassets.rgpub.io').map((s) => s.url);

test("the store build names the cards from Riot's gallery, and draws the picture it gives; its package holds no card name, type or picture", async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  let riot: FakeRiot | null = null;
  try {
    riot = await fakeRiot(LIST);
    loaded = await load({ store: true, args: riot.args });
    const { context, ext } = loaded;
    // what is in the package: the models and the gallery's vectors; and no catalogue and no thumbnails
    expect(existsSync(join(ext, 'data', 'gallery', 'index.json'))).toBe(true);
    for (const gone of ['data/catalog.json', 'data/thumbs']) expect(existsSync(join(ext, gone)), gone).toBe(false);
    const manifest = JSON.parse(readFileSync(join(ext, 'manifest.json'), 'utf8')) as { host_permissions: string[] };
    expect(manifest.host_permissions).toEqual(['https://content.publishing.riotgames.com/*', 'https://cmsassets.rgpub.io/*']);

    const page = await open(loaded, riot);
    await expect.poll(() => offscreenDocuments(context), { timeout: 20_000 }).toBe(1);

    // the engine's rows are the gallery's, named by the list: the gold block is TST-001, "Fed Unit, The Golden"
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText([FED]); // the other block is unsure: no label
    await expect(page.locator('.rifteye-badge-main')).toHaveText(/^Wardeye · 1 card named · \d+\.\d reads\/s$/);
    await expect(page.locator('.rifteye-badge-detail')).toHaveText(/^WebGPU · detector fp32 · embedder fp32 · /);

    // the list was read a page at a time, 200 items from each `from`
    expect(feedRequests(riot)).toEqual([0, 200].map((from) => `/publishing-content/v2.0/public/channel/riftbound_website/list/riftbound_gallery_cards?locale=en_US&from=${from}&limit=200`));

    // the hover card: the name, and the picture the gallery gives (400 x 559: the card is drawn 200 x 280, not the 16 x 16 of a package's)
    await boxes.first().hover();
    const card = page.locator('.rifteye-card');
    await expect(card).toContainText(FED);
    await expect(card).toContainText(/Confidence [01]\.\d\d/);
    const art = card.locator('canvas.rifteye-art').first();
    await expect.poll(async () => near(await middle(art), MAGENTA), { timeout: 10_000 }).toBe(true);
    expect(await art.evaluate((c) => [(c as HTMLCanvasElement).width, (c as HTMLCanvasElement).height])).toEqual([200, 280]);

    // the guesses of the unsure block: TST-002 by the list's name, with its picture, and TST-003, which the list does not name, by its id
    await boxes.nth(1).hover();
    await expect(card).toContainText('Not sure yet. Best guesses:');
    await expect(card).toContainText(/Fed Guess Two · \d+%/);
    await expect(card).toContainText(/TST-003 · \d+%/);
    const two = card.locator('figure', { hasText: 'Fed Guess Two' }).locator('canvas');
    await expect.poll(async () => near(await middle(two), CYAN), { timeout: 10_000 }).toBe(true);

    // what was asked of Riot's servers, and what was sent with it: pictures at 400 px as a JPEG, the address's own query kept, nothing of the user's
    const pictures = pictureRequests(riot);
    expect(pictures).toContain('/fake/tst-001-100.png?accountingTag=RB&w=400&fm=jpg&q=80');
    expect(pictures).toContain('/fake/tst-002-100.png?accountingTag=RB&w=400&fm=jpg&q=80');
    expect(pictures.every((u) => u.endsWith('&w=400&fm=jpg&q=80'))).toBe(true);
    expect(pictures.some((u) => u.includes('tst-003'))).toBe(false); // the list does not name it: there is no picture to ask for
    expect(riot.seen.length).toBeGreaterThan(2);
    for (const s of riot.seen) {
      expect(s.headers.cookie, `${s.host}${s.url}`).toBeUndefined();
      expect(s.headers.referer, `${s.host}${s.url}`).toBeUndefined();
    }
    // a picture is asked for once, and kept in memory for the next hover
    const before = pictureRequests(riot).length;
    await boxes.first().hover();
    await expect(card).toContainText(FED);
    await boxes.nth(1).hover();
    await expect(card).toContainText('Not sure yet');
    expect(pictureRequests(riot).length).toBe(before);
  } finally {
    await unload(loaded);
    await riot?.close();
  }
});

test('with the card list down, recognition still runs and the overlay shows printing ids; the list is asked for again a minute later, and names the cards', async () => {
  test.setTimeout(300_000);
  let loaded: Loaded | null = null;
  let riot: FakeRiot | null = null;
  try {
    riot = await fakeRiot(LIST);
    riot.down = true;
    loaded = await load({ store: true, args: riot.args });
    const page = await open(loaded, riot);

    // no names: each printing by its id, and the engine reads the table all the same
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['TST-001']);
    await expect(page.locator('.rifteye-badge-main')).toHaveText(/^Wardeye · 1 card named · /);
    await boxes.first().hover();
    const card = page.locator('.rifteye-card');
    await expect(card).toContainText('TST-001');
    await page.waitForTimeout(1500); // a picture would have come by now
    expect(await middle(card.locator('canvas.rifteye-art').first())).toEqual([0, 0, 0, 0]); // nothing drawn
    expect(pictureRequests(riot)).toEqual([]); // no list, no address to ask for a picture at
    expect(feedRequests(riot)).toHaveLength(1); // one try, which failed on its first page

    // the gallery comes back; a minute after the first try it is asked again, and the engine starts afresh with the names
    riot.down = false;
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText([FED], { timeout: 200_000 });
    const asked = feedRequests(riot);
    expect(asked.filter((u) => u.includes('from=0')).length).toBeGreaterThanOrEqual(2); // the first try, and the one that worked
    expect(asked.filter((u) => u.includes('from=200'))).toHaveLength(1); // the second page was asked for by the one that worked, and by no other
    await boxes.first().hover();
    await expect(card).toContainText(FED);
    await expect.poll(async () => near(await middle(card.locator('canvas.rifteye-art').first()), MAGENTA), { timeout: 10_000 }).toBe(true);
    expect(await offscreenDocuments(loaded.context)).toBe(1);
  } finally {
    await unload(loaded);
    await riot?.close();
  }
});

test('a browser with no WebGPU adapter runs the store build on plain WASM, where the developer build hands the frames to the live runner', async () => {
  test.setTimeout(150_000);
  let loaded: Loaded | null = null;
  let riot: FakeRiot | null = null;
  try {
    riot = await fakeRiot(LIST);
    loaded = await load({ store: true, gpu: false, args: riot.args });
    const page = await open(loaded, riot);
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 120_000 });
    await expect(page.locator('.rifteye-badge-detail')).toHaveText(/^WASM · detector fp32 · embedder fp32 · /);
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText([FED]);
  } finally {
    await unload(loaded);
    await riot?.close();
  }
});

test('an engine that cannot start is said plainly in the badge: the store build has no live runner to hand the frames to', async () => {
  test.setTimeout(90_000);
  let loaded: Loaded | null = null;
  let riot: FakeRiot | null = null;
  try {
    riot = await fakeRiot(LIST);
    loaded = await load({ store: true, package: { noModels: true }, args: riot.args }); // a package whose model files are missing
    const said: string[] = [];
    loaded.context.on('console', (m) => {
      if (m.text().includes('Wardeye: no engine')) said.push(m.text());
    });
    const page = await open(loaded, riot);
    await expect(page.locator('.rifteye-badge-main')).toHaveText('Wardeye: this browser cannot run the engine (WebGPU or WebAssembly needed)', { timeout: 60_000 });
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(0);
    expect(said.join('\n')).toContain('the overlay says so'); // and the console says why, and not that a live runner is used
    expect(said.join('\n')).not.toContain('live runner is used');
  } finally {
    await unload(loaded);
    await riot?.close();
  }
});
