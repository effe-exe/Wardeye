import { expect, test, type BrowserContext } from '@playwright/test';
import { SCRIPT_ID, YOUTUBE_ORIGINS } from '../src/access';
import { loadedFonts } from './brand';
import { load, offscreenDocuments, unload, type Loaded } from './harness';
import { youtube } from './youtube';

/** The extension's service worker (not any other worker the browser runs). */
const worker = async (context: BrowserContext) =>
  context.serviceWorkers().find((w) => w.url().startsWith('chrome-extension://')) ??
  (await context.waitForEvent('serviceworker', { predicate: (w) => w.url().startsWith('chrome-extension://') }));
/** The overlay scripts the worker registered; none yet while the worker is still starting (its APIs not bound). */
const registered = async (context: BrowserContext): Promise<string[]> =>
  (await worker(context)).evaluate(async () => ((await chrome.scripting?.getRegisteredContentScripts()) ?? []).map((s) => s.id));

/** What the plays panel would be told of the YouTube tab: its video's key and title, and its plays. */
async function snapshot(context: BrowserContext): Promise<{ video: string; title: string; on: boolean; plays: unknown[] }> {
  return (await worker(context)).evaluate(async () => {
    const [tab] = await chrome.tabs.query({ url: 'https://www.youtube.com/*' });
    return new Promise<{ video: string; title: string; on: boolean; plays: unknown[] }>((resolve) => {
      const p = chrome.tabs.connect(tab!.id!, { name: 'plays' });
      p.onMessage.addListener((m: { video: string; title: string; on: boolean; plays: unknown[] }) => {
        p.disconnect();
        resolve(m);
      });
    });
  });
}

test('on YouTube, once allowed, Wardeye reads the player of a video\'s page and nothing else', async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ grant: YOUTUBE_ORIGINS });
    const { context } = loaded;
    // allowed: the worker registers the overlay's script for YouTube's pages
    await expect.poll(() => registered(context), { timeout: 20_000 }).toEqual([SCRIPT_ID]);
    const page = await youtube(context);
    const root = page.locator('.rifteye-root');

    // the home page has a player too (its previews): Wardeye is there, but reads nothing and shows nothing
    await page.goto('https://www.youtube.com/');
    await expect(root).toHaveCount(1);
    await page.waitForTimeout(3000);
    await expect(root).toBeHidden();
    expect(await offscreenDocuments(context)).toBe(0); // nothing to read: the engine is not even started

    // a video's page: the main player is read and boxed, not the bigger preview beside it
    await page.goto('https://www.youtube.com/watch?v=AAAAAAAAAAA&list=PL1');
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('.rifteye-badge-main')).toHaveText('Wardeye · 1 card named');
    const player = (await page.locator('#movie_player video').boundingBox())!;
    const overlay = (await root.boundingBox())!;
    expect(overlay).toEqual(player);
    // the registered stylesheet finds the extension's own typefaces, as the manifest's does on Twitch
    await expect.poll(() => loadedFonts(page)).toEqual(expect.arrayContaining(['Inter', 'Space Grotesk']));
    // the panel hears of the video by its id, under its own title
    expect(await snapshot(context)).toMatchObject({ video: '/watch?v=AAAAAAAAAAA', title: 'Riftbound test' });

    // an ad in the player: the overlay goes, and comes back with the video
    await page.evaluate(() => document.getElementById('movie_player')!.classList.add('ad-showing'));
    await expect(root).toBeHidden();
    await page.evaluate(() => document.getElementById('movie_player')!.classList.remove('ad-showing'));
    await expect(root).toBeVisible();

    // injected again (the toolbar button on a tab that already has it): still one overlay
    await (await worker(context)).evaluate(async () => {
      const [tab] = await chrome.tabs.query({ url: 'https://www.youtube.com/*' });
      await chrome.scripting.executeScript({ target: { tabId: tab!.id! }, files: ['content.js'] });
    });
    await page.waitForTimeout(1000);
    await expect(root).toHaveCount(1);
    await expect(boxes).toHaveCount(2);

    // YouTube moves to another video without loading a page: its plays start over, under its own key
    await page.evaluate(() => history.pushState({}, '', '/watch?v=BBBBBBBBBBB'));
    expect(await snapshot(context)).toMatchObject({ video: '/watch?v=BBBBBBBBBBB', plays: [] });

    // ... and a Short is not read
    await page.evaluate(() => history.pushState({}, '', '/shorts/BBBBBBBBBBB'));
    await expect(root).toBeHidden();
  } finally {
    await unload(loaded);
  }
});

test('on YouTube, a video whose title does not name Riftbound is read only once Wardeye is turned on for it', async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ grant: YOUTUBE_ORIGINS });
    const { context } = loaded;
    await expect.poll(() => registered(context), { timeout: 20_000 }).toEqual([SCRIPT_ID]);
    const page = await youtube(context, 16, 'Pasta carbonara in 10 minutes');
    await page.goto('https://www.youtube.com/watch?v=CCCCCCCCCCC');
    const root = page.locator('.rifteye-root');
    await page.waitForTimeout(4000);
    await expect(root).toBeHidden();
    expect(await offscreenDocuments(context)).toBe(0);
    expect(await snapshot(context)).toMatchObject({ on: false });

    // the toolbar button (here, its message): read and drawn from now on, for this video
    const toggle = () =>
      worker(context).then((w) =>
        w.evaluate(async () => {
          const [tab] = await chrome.tabs.query({ url: 'https://www.youtube.com/*' });
          await chrome.tabs.sendMessage(tab!.id!, { kind: 'toggle' });
        }),
      );
    await toggle();
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    expect(await snapshot(context)).toMatchObject({ on: true });
    // ... and off again
    await toggle();
    await expect(root).toBeHidden();
    expect(await snapshot(context)).toMatchObject({ on: false });
  } finally {
    await unload(loaded);
  }
});

test("on YouTube, a video in the player whose picture cannot be read (another site's, like an ad) is passed over", async () => {
  test.setTimeout(180_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ grant: YOUTUBE_ORIGINS });
    const { context } = loaded;
    await expect.poll(() => registered(context), { timeout: 20_000 }).toEqual([SCRIPT_ID]);
    const page = await youtube(context, 16, 'Riftbound test', true);
    await page.goto('https://www.youtube.com/watch?v=AAAAAAAAAAA');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 90_000 });
    await expect(page.locator('.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']);
  } finally {
    await unload(loaded);
  }
});

test('on YouTube, not allowed, Wardeye is not there at all', async () => {
  test.setTimeout(90_000);
  let loaded: Loaded | null = null;
  try {
    loaded = await load();
    const { context } = loaded;
    const page = await youtube(context, 4);
    await page.goto('https://www.youtube.com/watch?v=AAAAAAAAAAA');
    await page.waitForTimeout(3000);
    await expect(page.locator('.rifteye-root')).toHaveCount(0);
    expect(await (await worker(context)).evaluate(async () => (await chrome.scripting.getRegisteredContentScripts()).length)).toBe(0);
  } finally {
    await unload(loaded);
  }
});
