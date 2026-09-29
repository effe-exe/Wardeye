import { cpSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

const DIST = fileURLToPath(new URL('../dist/', import.meta.url));

const art = (text: string, hue: number) =>
  'data:image/svg+xml;base64,' +
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="120" height="168"><rect width="120" height="168" fill="hsl(${hue} 50% 40%)"/>` +
      `<text x="10" y="90" fill="#fff" font-size="16">${text}</text></svg>`,
  ).toString('base64');

/** Records a short WebM in the browser itself, so the test needs no media fixtures in git. */
async function makeVideo(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 180;
    const ctx = canvas.getContext('2d')!;
    const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: 'video/webm;codecs=vp8' });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    const stopped = new Promise((r) => (rec.onstop = r));
    rec.start(100);
    const t0 = performance.now();
    await new Promise<void>((resolve) => {
      const draw = () => {
        ctx.fillStyle = '#a0103e';
        ctx.fillRect(0, 0, 320, 180);
        ctx.fillStyle = '#222';
        ctx.fillRect(150, 70, 30, 42);
        if (performance.now() - t0 < 3000) requestAnimationFrame(draw);
        else resolve();
      };
      draw();
    });
    rec.stop();
    await stopped;
    const bytes = new Uint8Array(await new Blob(chunks, { type: 'video/webm' }).arrayBuffer());
    let s = '';
    for (const b of bytes) s += String.fromCharCode(b);
    return btoa(s);
  });
  return Buffer.from(b64, 'base64');
}

test('hover a recognised card, see three guesses for an unsure one, and jump from the timeline', async ({ page }) => {
  const dir = mkdtempSync(join(tmpdir(), 'rifteye-viewer-'));
  cpSync(DIST, dir, { recursive: true });
  await page.goto('about:blank');
  writeFileSync(join(dir, 'clip.webm'), await makeVideo(page));
  const bundle = {
    schema: 'rifteye.demo',
    version: 1,
    title: 'e2e match',
    video: 'clip.webm',
    frame: [320, 180],
    detectFps: 3,
    cards: {
      'FAK-001': { name: 'Fake Hero', type: 'Unit', art: art('Fake Hero', 10), printing: 'FAK-001' },
      'FAK-002': { name: 'Fake Legend', type: 'Legend', art: art('Fake Legend', 120), printing: 'FAK-002' },
      'FAK-003': { name: 'Heroic Strike', type: 'Spell', art: art('Heroic Strike', 220), printing: 'FAK-003' },
    },
    tracks: [
      { id: 'k0', samples: [[0, 0.5, 0.5, 0.25, 0.18, 90], [2.5, 0.5, 0.5, 0.25, 0.18, 90]], guesses: [{ card: 'FAK-001', p: 0.93 }] },
      { id: 'k1', samples: [[0, 0.2, 0.4, 0.25, 0.18, 90], [2.5, 0.2, 0.4, 0.25, 0.18, 90]],
        guesses: [{ card: 'FAK-002', p: 0.5 }, { card: 'FAK-003', p: 0.2 }, { card: 'FAK-001', p: 0.1 }] },
      { id: 'k2', samples: [[0, 0.8, 0.4, 0.25, 0.18, 90], [2.5, 0.8, 0.4, 0.25, 0.18, 90]], faceDown: true },
    ],
    events: [{ t: 2.0, tBefore: 1.6, kind: 'played', track: 'k0', box: [0.4, 0.3, 0.6, 0.7] }],
  };
  writeFileSync(join(dir, 'data.js'), `window.RIFTEYE_DEMO = ${JSON.stringify(bundle)};\n`);

  await page.goto(`file://${join(dir, 'index.html')}`);
  // the page wears Wardeye's brand: the build ships the mark next to it and puts the tokens in front of its stylesheet
  expect(await page.locator('h1 img').evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--wd-primary').trim())).not.toBe('');
  await expect(page.locator('#title')).toHaveText('e2e match');
  await expect(page.locator('#events li')).toHaveCount(1);
  await expect(page.locator('#events li .what')).toContainText('Fake Hero');
  await expect(page.locator('#overlay polygon')).toHaveCount(3);
  await expect(page.locator('#now-list')).toHaveText('Fake Hero, Fake Legend?, a face-down card');

  const hoverOn = async (track: string) => {
    const box = (await page.locator(`#overlay polygon[data-track="${track}"]`).boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  };
  await hoverOn('k0');
  await expect(page.locator('#hover')).toBeVisible();
  await expect(page.locator('#hover .name')).toHaveText('Fake Hero');
  await expect(page.locator('#hover .sure')).toHaveText('Wardeye is 93% sure');
  await hoverOn('k1');
  await expect(page.locator('#hover .three figure')).toHaveCount(3);
  await expect(page.locator('#hover .note')).toHaveText('Not sure yet. Best guesses:');
  await hoverOn('k2');
  await expect(page.locator('#hover')).toContainText('never tries to identify it');
  await page.mouse.move(5, 5);
  await expect(page.locator('#hover')).toBeHidden();

  await page.locator('#events li').click(); // jumps to a second before the hand arrived
  await expect.poll(() => page.evaluate(() => document.querySelector('video')!.currentTime)).toBeGreaterThan(0.5);
  await page.locator('#show-boxes').uncheck();
  await expect(page.locator('#overlay')).toHaveClass(/hidden-boxes/);
});
