import { mkdtempSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, expect, test, type BrowserContext, type Page } from '@playwright/test';
import { animationsSeen, hex, loadedFonts, logAnimations, rgb, styleOf } from './brand';
import { load, offscreenDocuments, twitch, unload, type Loaded } from './harness';

const DIST = fileURLToPath(new URL('../dist/', import.meta.url));
// A 16 x 16 grey JPEG: the fake runner's "card art" (a made-up picture, no card art in git).
const JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/wAALCAAQABABAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q==',
  'base64',
);

interface Posted {
  t: number;
  video: string;
  bytes: number;
  jpeg: boolean;
}

/** Stands in for `python -m rifteye_ml.live --source browser`: answers every frame with a fixed board, in which the second card is unsure
 * (or, after `namesSecondAfter` frames, named), and then the `more` tracks, if any. */
function fakeRunner(posted: Posted[], artAsked: string[], namesSecondAfter = Infinity, more: object[] = []): Promise<Server> {
  const second = (named: boolean) =>
    named
      ? { id: 't2', quad: [[300, 120], [360, 120], [360, 204], [300, 204]], side: 'right', state: 'named', printing_id: 'TST-002',
          name: 'Guess Two', confidence: 0.8, guesses: [], kind: 'card', hidden: false }
      : { id: 't2', quad: [[300, 120], [360, 120], [360, 204], [300, 204]], side: 'right', state: 'unsure', printing_id: null,
          name: '', confidence: 0.4, guesses: [{ printing_id: 'TST-002', name: 'Guess Two', p: 0.4 }], kind: 'card', hidden: false };
  const board = () => ({
    t: 1, status: 'live', message: '', title: 'test', frame: { width: 640, height: 360 },
    players: [], tracks: [
      { id: 't1', quad: [[100, 100], [160, 100], [160, 184], [100, 184]], side: 'left', state: 'named', printing_id: 'TST-001',
        name: 'Test Unit', confidence: 0.9, guesses: [], kind: 'card', hidden: false,
        under: [{ id: 't3', name: 'Test Gear', printing_id: 'TST-003' }] },
      second(posted.length > namesSecondAfter),
      ...more,
    ],
  });
  const server = createServer((req: IncomingMessage, res) => {
    if (req.method === 'GET' && req.url === '/hello') {
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ rifteye: 'live', frames: true }));
    } else if (req.method === 'POST' && req.url === '/frame') {
      const chunks: Buffer[] = [];
      req.on('data', (c: Buffer) => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        posted.push({ t: Number(req.headers['x-media-time']), video: String(req.headers['x-video']), bytes: body.length,
          jpeg: body[0] === 0xff && body[1] === 0xd8 });
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(board()));
      });
    } else if (req.method === 'GET' && req.url?.startsWith('/art/')) {
      artAsked.push(req.url);
      res.writeHead(200, { 'Content-Type': 'image/jpeg' }).end(JPEG);
    } else {
      res.writeHead(404).end();
    }
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(8765, '127.0.0.1', () => resolve(server));
  });
}

/** Records a short WebM in the browser itself, so the test needs no media in git. */
async function makeVideo(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d')!;
    const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: 'video/webm;codecs=vp8' });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    const stopped = new Promise((r) => (rec.onstop = r));
    rec.start(100);
    const t0 = performance.now();
    await new Promise<void>((resolve) => {
      const draw = () => {
        ctx.fillStyle = '#10303e';
        ctx.fillRect(0, 0, 640, 360);
        ctx.fillStyle = '#c0a040';
        ctx.fillRect(100, 100, 60, 84);
        ctx.fillRect(300 + ((performance.now() - t0) / 200) % 20, 120, 60, 84);
        if (performance.now() - t0 < 8000) requestAnimationFrame(draw);
        else resolve();
      };
      draw();
    });
    rec.stop();
    await stopped;
    const buf = await new Blob(chunks, { type: 'video/webm' }).arrayBuffer();
    let bin = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  });
  return Buffer.from(b64, 'base64');
}

test('frames from the Twitch player reach the runner and its board is drawn on the player', async () => {
  test.setTimeout(90_000);
  const posted: Posted[] = [];
  const artAsked: string[] = [];
  const runner = await fakeRunner(posted, artAsked);
  let context: BrowserContext | null = null;
  try {
    context = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'rifteye-ext-')), {
      headless: true,
      args: [`--disable-extensions-except=${DIST}`, `--load-extension=${DIST}`, '--autoplay-policy=no-user-gesture-required'],
      // Playwright's default headless shell cannot load extensions; its full Chromium (channel) can
      ...(process.env.RIFTEYE_CHROMIUM ? { executablePath: process.env.RIFTEYE_CHROMIUM } : { channel: 'chromium' }),
    });
    const page = await context.newPage();
    await page.goto('about:blank');
    const webm = await makeVideo(page);
    await context.route('https://www.twitch.tv/**', (route) => {
      const url = route.request().url();
      if (url.endsWith('/test.webm')) return route.fulfill({ status: 200, contentType: 'video/webm', body: webm });
      return route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<!doctype html><html><body style="margin:0;background:#000">' +
          '<video id="v" src="/test.webm" muted autoplay playsinline style="width:960px;height:540px"></video></body></html>',
      });
    });
    await logAnimations(page); // every CSS animation that starts on the page is recorded, to check that nothing else moves
    await page.goto('https://www.twitch.tv/videos/12345');

    await expect.poll(() => posted.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(3);
    expect(posted.every((p) => p.jpeg && p.bytes > 1000 && p.video === '/videos/12345')).toBe(true);
    expect(posted.at(-1)!.t).toBeGreaterThan(posted[0]!.t); // the video's own clock

    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2);
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']); // unsure: no label
    await expect(page.locator('.rifteye-badge')).toHaveText('Wardeye · 1 card named');

    // the overlay sits on the picture: the named box, centred at (130, 142) of 640 x 360, is at (195, 213) of the
    // 960 x 540 player (its measured outline also holds the stroke, so the centre is what is compared)
    const b = (await boxes.first().boundingBox())!;
    expect(Math.abs(b.x + b.width / 2 - 195)).toBeLessThan(2);
    expect(Math.abs(b.y + b.height / 2 - 213)).toBeLessThan(2);

    // the look (assets/brand), before the pointer is on anything. The brand's tokens are on the overlay's root and nowhere else, so nothing
    // reaches Twitch's own styles; its typefaces come from the extension's own files, which the manifest makes web accessible.
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--wd-primary'))).toBe('');
    expect(await page.evaluate(() => getComputedStyle(document.body).getPropertyValue('--wd-primary'))).toBe('');
    expect(await styleOf(page, '.rifteye-root', ['--wd-primary'])).toEqual({ '--wd-primary': hex('primary') });
    await expect.poll(() => loadedFonts(page)).toEqual(expect.arrayContaining(['Inter', 'Space Grotesk']));
    // boxes: a named card is outlined in the primary (once its pulse is over), an unsure read dashed in the warning colour
    await expect.poll(() => boxes.first().evaluate((e) => e.getAnimations().length)).toBe(0);
    expect(await styleOf(page, 'polygon.rifteye-named', ['stroke', 'stroke-width', 'stroke-dasharray'])).toEqual({ stroke: rgb('primary'), 'stroke-width': '2px', 'stroke-dasharray': 'none' });
    const unsure = await styleOf(page, 'polygon.rifteye-unsure', ['stroke', 'stroke-dasharray']);
    expect(unsure.stroke).toBe(rgb('warning'));
    expect(unsure['stroke-dasharray']).not.toBe('none');
    // a label: Inter SemiBold, with a dark halo so it reads on video
    const named = await styleOf(page, 'text.rifteye-label', ['font-family', 'font-weight', 'fill', 'stroke', 'paint-order']);
    expect(named).toMatchObject({ 'font-weight': '600', fill: rgb('text'), stroke: rgb('bg'), 'paint-order': 'stroke' });
    expect(named['font-family']).toMatch(/^"?Inter"?,/);
    // the badge: a translucent surface (the surface colour at 85%), a hairline border, 8 px corners, a blur behind it, the mark (14 px tall)
    // and the name in Space Grotesk, then the status in Inter
    expect(await styleOf(page, '.rifteye-badge', ['border-top-width', 'border-top-color', 'border-top-left-radius'])).toEqual({
      'border-top-width': '1px', 'border-top-color': rgb('border'), 'border-top-left-radius': '8px',
    });
    expect((await styleOf(page, '.rifteye-badge', ['backdrop-filter']))['backdrop-filter']).toMatch(/^blur\(/);
    const surface = await page.evaluate(() => {
      const c = document.createElement('canvas');
      c.width = c.height = 1;
      const g = c.getContext('2d')!;
      g.fillStyle = getComputedStyle(document.querySelector('.rifteye-badge')!).backgroundColor;
      g.fillRect(0, 0, 1, 1);
      return [...g.getImageData(0, 0, 1, 1).data];
    });
    const [sr, sg, sb] = rgb('surface').match(/\d+/g)!.map(Number);
    expect([Math.abs(surface[0]! - sr!), Math.abs(surface[1]! - sg!), Math.abs(surface[2]! - sb!)].every((d) => d <= 2)).toBe(true);
    expect(Math.abs(surface[3]! - 0.85 * 255)).toBeLessThanOrEqual(2);
    const mark = (await page.locator('.rifteye-badge svg.rifteye-mark').boundingBox())!;
    expect(mark.height).toBeGreaterThan(13);
    expect(mark.height).toBeLessThan(15);
    expect(await styleOf(page, '.rifteye-mark-ward', ['fill'])).toEqual({ fill: rgb('primary') });
    expect((await styleOf(page, '.rifteye-badge-name', ['font-family']))['font-family']).toMatch(/^"?Space Grotesk"?,/);
    expect((await styleOf(page, '.rifteye-badge-status', ['font-family']))['font-family']).toMatch(/^"?Inter"?,/);
    expect(await page.locator('.rifteye-badge-name').textContent()).toBe('Wardeye');

    await boxes.first().hover();
    const card = page.locator('.rifteye-card');
    await expect(card).toBeVisible();
    await expect(card).toContainText('Test Unit');
    await expect(card).toContainText('Confidence 0.90');
    await expect(card).toContainText('Under it: Test Gear');
    await expect.poll(() => artAsked).toContain('/art/TST-001.jpg');
    const painted = () =>
      page.evaluate(() => {
        const c = document.querySelector('.rifteye-card canvas') as HTMLCanvasElement | null;
        return c ? c.getContext('2d')!.getImageData(c.width / 2, c.height / 2, 1, 1).data[3] : -1;
      });
    await expect.poll(painted, { timeout: 5000 }).toBe(255); // the card's picture is drawn

    // the card preview: a surface panel with a primary border and 8 px corners, the picture on top, then the name (Inter SemiBold), the
    // confidence (primary, at the caption size) and what lies under it (muted). This card has no type to show, so no meta line. The box
    // under the pointer takes the lighter primary, with a faint fill of it.
    expect(await card.evaluate((c) => [...c.children].map((e) => e.className))).toEqual(['rifteye-art', 'rifteye-name', 'rifteye-sure', 'rifteye-under']);
    expect(await styleOf(page, '.rifteye-card', ['background-color', 'border-top-color', 'border-top-left-radius'])).toEqual({
      'background-color': rgb('surface'), 'border-top-color': rgb('primary'), 'border-top-left-radius': '8px',
    });
    expect(await styleOf(page, '.rifteye-name', ['font-weight', 'color'])).toEqual({ 'font-weight': '600', color: rgb('text') });
    expect((await styleOf(page, '.rifteye-name', ['font-family']))['font-family']).toMatch(/^"?Inter"?,/);
    expect(await styleOf(page, '.rifteye-sure', ['color', 'font-size'])).toEqual({ color: rgb('primary'), 'font-size': '11px' });
    expect(await styleOf(page, '.rifteye-under', ['color'])).toEqual({ color: rgb('muted') });
    expect(await styleOf(page, 'polygon.rifteye-named', ['stroke', 'fill', 'fill-opacity'])).toEqual({ stroke: rgb('primary-light'), fill: rgb('primary-light'), 'fill-opacity': '0.14' });

    await boxes.nth(1).hover();
    await expect(card).toContainText('Not sure yet. Best guesses:');
    await expect(card).toContainText('Guess Two · 40%');
    // the box under the pointer is the lighter primary, still dashed while it is unsure; the first box is back to its own colour; the
    // percentages are in JetBrains Mono, the note in the warning colour
    const hovered = await styleOf(page, 'polygon.rifteye-unsure', ['stroke', 'stroke-dasharray']);
    expect(hovered.stroke).toBe(rgb('primary-light'));
    expect(hovered['stroke-dasharray']).not.toBe('none');
    expect(await styleOf(page, 'polygon.rifteye-named', ['stroke', 'fill-opacity'])).toEqual({ stroke: rgb('primary'), 'fill-opacity': '1' });
    expect(await page.locator('.rifteye-pct').allTextContents()).toEqual(['40%']);
    expect((await styleOf(page, '.rifteye-pct', ['font-family']))['font-family']).toMatch(/^"?JetBrains Mono"?,/);
    expect(await styleOf(page, '.rifteye-note', ['color'])).toEqual({ color: rgb('warning') });
    await expect.poll(() => loadedFonts(page)).toEqual(expect.arrayContaining(['Inter', 'JetBrains Mono', 'Space Grotesk']));

    await page.evaluate(() => (document.getElementById('v') as HTMLVideoElement).pause());
    await page.waitForTimeout(600);
    const n = posted.length;
    await page.waitForTimeout(1200);
    expect(posted.length).toBe(n); // paused: nothing is sent, the board stays
    await expect(boxes).toHaveCount(2);

    // motion: the hover card fades in (150 ms), the card that became named pulsed once (under 600 ms), and nothing else moved
    const seen = await animationsSeen(page);
    expect(seen.every((a) => a.name === 'rifteye-pulse' || a.name === 'rifteye-rise')).toBe(true);
    const pulses = seen.filter((a) => a.name === 'rifteye-pulse');
    expect(pulses).toHaveLength(1); // one card became named, and the boards that followed did not pulse it again
    expect(pulses[0]!.on).toContain('rifteye-named');
    expect(pulses[0]!.iterations).toBe(1);
    expect(pulses[0]!.ms).toBeLessThanOrEqual(600);
    const rises = seen.filter((a) => a.name === 'rifteye-rise');
    expect(rises.length).toBeGreaterThan(0);
    expect(rises.every((a) => a.ms === 150 && a.iterations === 1)).toBe(true);
    // ... and reduced motion turns both off: no animation on the card, none on a pulsing box
    const animationNames = () =>
      page.evaluate(() => {
        const box = document.querySelector('polygon.rifteye-named')!;
        box.classList.add('rifteye-pulse');
        const names = [getComputedStyle(document.querySelector('.rifteye-card')!).animationName, getComputedStyle(box).animationName];
        box.classList.remove('rifteye-pulse');
        return names;
      });
    expect(await animationNames()).toEqual(['rifteye-rise', 'rifteye-pulse']);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    expect(await animationNames()).toEqual(['none', 'none']);
    await page.emulateMedia({ reducedMotion: 'no-preference' });

    // Alt+R turns it off: hidden, and not a frame read while the video plays; again, and it reads and draws the board once more
    await page.keyboard.press('Alt+KeyR');
    await expect(page.locator('.rifteye-root')).toBeHidden();
    await page.evaluate(() => (document.getElementById('v') as HTMLVideoElement).play());
    await page.waitForTimeout(600);
    const off = posted.length;
    await page.waitForTimeout(1500);
    expect(posted.length).toBe(off); // off: nothing is read or sent
    await page.keyboard.press('Alt+KeyR');
    await expect(page.locator('.rifteye-root')).toBeVisible();
    await expect.poll(() => posted.length, { timeout: 10_000 }).toBeGreaterThan(off);
    await expect(boxes).toHaveCount(2);
  } finally {
    await context?.close();
    runner.close();
  }
});

test('a card that is named later gets a pulse of its own, and only one', async () => {
  test.setTimeout(120_000);
  const posted: Posted[] = [];
  const runner = await fakeRunner(posted, [], 10); // the second card is unsure for ten frames (at least two seconds), then named
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ package: null }); // the public build: the runner reads
    const page = await twitch(loaded.context);
    await logAnimations(page);
    await page.goto('https://www.twitch.tv/videos/12345');
    const labels = page.locator('text.rifteye-label', { hasText: /\S/ });
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 30_000 });
    await expect(labels).toHaveText(['Test Unit'], { timeout: 30_000 }); // the second is unsure: no label yet
    await expect(labels).toHaveText(['Test Unit', 'Guess Two'], { timeout: 30_000 }); // ... and now it is named
    const pulses = async () => (await animationsSeen(page)).filter((a) => a.name === 'rifteye-pulse');
    await expect.poll(async () => (await pulses()).length).toBe(2); // the first card at first sight, the second when it was named
    await page.waitForTimeout(2000); // more boards come; a card that stays named is not pulsed again
    const seen = await pulses();
    expect(seen).toHaveLength(2);
    expect(seen.every((a) => a.on.includes('rifteye-named') && a.iterations === 1 && a.ms <= 600)).toBe(true);
    expect((await animationsSeen(page)).every((a) => a.name === 'rifteye-pulse')).toBe(true); // nothing else moved: the pointer was never on a box
  } finally {
    await unload(loaded);
    runner.close();
  }
});

test('a legend says its type on its card preview, and a face-down card is dim, dotted and never identified', async () => {
  test.setTimeout(120_000);
  const runner = await fakeRunner([], [], Infinity, [
    { id: 't4', quad: [[430, 50], [490, 50], [490, 134], [430, 134]], side: 'right', state: 'named', printing_id: 'TST-004',
      name: 'Test Legend', confidence: 0.97, guesses: [], kind: 'legend', hidden: false },
    { id: 't5', quad: [[200, 230], [260, 230], [260, 314], [200, 314]], side: 'left', state: 'facedown', printing_id: null,
      name: '', confidence: 0, guesses: [], kind: 'card', hidden: false },
  ]);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ package: null }); // the public build: the runner reads
    const page = await twitch(loaded.context, 10);
    await page.goto('https://www.twitch.tv/videos/12345');
    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(4, { timeout: 30_000 });
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit', 'Test Legend']); // the face-down card has no name
    // a face-down card: the dim token, dotted (zero-length dashes with round caps)
    const facedown = await styleOf(page, 'polygon.rifteye-facedown', ['stroke', 'stroke-dasharray', 'stroke-linecap']);
    expect(facedown.stroke).toBe(rgb('dim'));
    expect(facedown['stroke-dasharray']).toMatch(/^0(px)?[ ,]/);
    expect(facedown['stroke-linecap']).toBe('round');
    // pointing at it says only what is true (D-005), in the muted colour, with no picture
    const card = page.locator('.rifteye-card');
    await page.locator('polygon.rifteye-facedown').hover();
    await expect(card).toHaveText('Face-down card: never identified');
    expect(await card.evaluate((c) => [...c.children].map((e) => e.className))).toEqual(['rifteye-plain']);
    expect(await styleOf(page, '.rifteye-plain', ['color'])).toEqual({ color: rgb('muted') });
    // a legend is the one card the state gives a type for: a muted meta line between its name and its confidence
    await page.locator('polygon.rifteye-named').nth(1).hover();
    await expect(card).toContainText('Test Legend');
    expect(await card.evaluate((c) => [...c.children].map((e) => e.className))).toEqual(['rifteye-art', 'rifteye-name', 'rifteye-meta', 'rifteye-sure']);
    await expect(page.locator('.rifteye-meta')).toHaveText('Legend');
    expect(await styleOf(page, '.rifteye-meta', ['color', 'font-size'])).toEqual({ color: rgb('muted'), 'font-size': '11px' });
    await expect(page.locator('.rifteye-sure')).toHaveText('Confidence 0.97');
  } finally {
    await unload(loaded);
    runner.close();
  }
});

test('a private build in a browser with no WebGPU adapter hands the frames to the runner, and draws its board', async () => {
  test.setTimeout(120_000);
  const posted: Posted[] = [];
  const runner = await fakeRunner(posted, []);
  let loaded: Loaded | null = null;
  try {
    loaded = await load({ gpu: false }); // the package is there; the browser cannot run the engine
    const { context } = loaded;
    const page = await twitch(context, 10);
    await page.goto('https://www.twitch.tv/videos/12345');
    await expect(page.locator('polygon.rifteye-box')).toHaveCount(2, { timeout: 30_000 });
    await expect(page.locator('.rifteye-badge')).toHaveText('Wardeye · 1 card named'); // the runner's board: no reads a second
    await expect.poll(() => posted.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(3);
    expect(posted.every((p) => p.jpeg && p.video === '/videos/12345')).toBe(true);
    expect(await offscreenDocuments(context)).toBe(1); // made when the tab connected, which said it cannot run the engine
  } finally {
    await unload(loaded);
    runner.close();
  }
});
