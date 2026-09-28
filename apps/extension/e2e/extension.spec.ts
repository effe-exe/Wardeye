import { mkdtempSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, expect, test, type BrowserContext, type Page } from '@playwright/test';

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

/** Stands in for `python -m rifteye_ml.live --source browser`: answers every frame with a fixed board. */
function fakeRunner(posted: Posted[], artAsked: string[]): Promise<Server> {
  const state = {
    t: 1, status: 'live', message: '', title: 'test', frame: { width: 640, height: 360 },
    players: [], tracks: [
      { id: 't1', quad: [[100, 100], [160, 100], [160, 184], [100, 184]], side: 'left', state: 'named', printing_id: 'TST-001',
        name: 'Test Unit', confidence: 0.9, guesses: [], kind: 'card', hidden: false,
        under: [{ id: 't3', name: 'Test Gear', printing_id: 'TST-003' }] },
      { id: 't2', quad: [[300, 120], [360, 120], [360, 204], [300, 204]], side: 'right', state: 'unsure', printing_id: null,
        name: '', confidence: 0.4, guesses: [{ printing_id: 'TST-002', name: 'Guess Two', p: 0.4 }], kind: 'card', hidden: false },
    ],
  };
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
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(state));
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
    await page.goto('https://www.twitch.tv/videos/12345');

    await expect.poll(() => posted.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(3);
    expect(posted.every((p) => p.jpeg && p.bytes > 1000 && p.video === '/videos/12345')).toBe(true);
    expect(posted.at(-1)!.t).toBeGreaterThan(posted[0]!.t); // the video's own clock

    const boxes = page.locator('polygon.rifteye-box');
    await expect(boxes).toHaveCount(2);
    await expect(page.locator('text.rifteye-label', { hasText: /\S/ })).toHaveText(['Test Unit']); // unsure: no label
    await expect(page.locator('.rifteye-badge')).toHaveText('RiftEye · 1 card named');

    // the overlay sits on the picture: the named box, centred at (130, 142) of 640 x 360, is at (195, 213) of the
    // 960 x 540 player (its measured outline also holds the stroke, so the centre is what is compared)
    const b = (await boxes.first().boundingBox())!;
    expect(Math.abs(b.x + b.width / 2 - 195)).toBeLessThan(2);
    expect(Math.abs(b.y + b.height / 2 - 213)).toBeLessThan(2);

    await boxes.first().hover();
    const card = page.locator('.rifteye-card');
    await expect(card).toBeVisible();
    await expect(card).toContainText('Test Unit');
    await expect(card).toContainText('RiftEye is 90% sure');
    await expect(card).toContainText('Under it: Test Gear');
    await expect.poll(() => artAsked).toContain('/art/TST-001.jpg');
    const painted = () =>
      page.evaluate(() => {
        const c = document.querySelector('.rifteye-card canvas') as HTMLCanvasElement | null;
        return c ? c.getContext('2d')!.getImageData(c.width / 2, c.height / 2, 1, 1).data[3] : -1;
      });
    await expect.poll(painted, { timeout: 5000 }).toBe(255); // the card's picture is drawn

    await boxes.nth(1).hover();
    await expect(card).toContainText('Not sure yet. Best guesses:');
    await expect(card).toContainText('Guess Two · 40%');

    await page.evaluate(() => (document.getElementById('v') as HTMLVideoElement).pause());
    await page.waitForTimeout(600);
    const n = posted.length;
    await page.waitForTimeout(1200);
    expect(posted.length).toBe(n); // paused: nothing is sent, the board stays
    await expect(boxes).toHaveCount(2);

    await page.keyboard.press('Alt+KeyR');
    await expect(page.locator('.rifteye-root')).toBeHidden();
  } finally {
    await context?.close();
    runner.close();
  }
});
