import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { validateTimelineDocument, type TimelineDocument } from '../../../packages/schema/src/index';

const APP = new URL('../dist/index.html', import.meta.url).href;

const CATALOG = [
  { printing_id: 'FAK-001', card_id: 'fake-hero', name: 'Fake Hero', type: 'Unit' },
  { printing_id: 'FAK-002', card_id: 'fake-legend', name: 'Fake Legend', type: 'Legend' },
  { printing_id: 'FAK-003', card_id: 'heroic-strike', name: 'Heroic Strike', type: 'Spell' },
]
  .map((r) => JSON.stringify(r))
  .join('\n');

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
        const t = performance.now() - t0;
        ctx.fillStyle = `hsl(${(t / 10) % 360} 60% 40%)`;
        ctx.fillRect(0, 0, 320, 180);
        ctx.fillStyle = '#fff';
        ctx.fillRect((t / 10) % 300, 80, 20, 20);
        if (t < 4000) requestAnimationFrame(draw);
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

test('log a match from the keyboard, export a valid timeline, and restore it', async ({ page }) => {
  await page.goto(APP);
  const video = await makeVideo(page);
  page.on('dialog', (d) => void d.accept());

  await page.setInputFiles('#open-cards', { name: 'catalog.jsonl', mimeType: 'application/json', buffer: Buffer.from(CATALOG) });
  await expect(page.locator('#card-count')).toHaveText('3 cards loaded');
  await page.setInputFiles('#open-video', { name: 'match.webm', mimeType: 'video/webm', buffer: video });
  await expect(page.locator('#video')).toBeVisible();

  await page.fill('#player-a', 'Ann');
  await page.locator('#player-a').press('Tab');
  await page.fill('#legend-a', 'fake leg');
  await page.locator('#legend-a').press('Enter');
  await expect(page.locator('#legend-a')).toHaveValue('Fake Legend');
  await page.locator('body').click({ position: { x: 5, y: 5 } }); // leave the form

  // Let the video run for a moment so events get real timestamps.
  await page.evaluate(() => (document.getElementById('video') as HTMLVideoElement).play());
  await page.waitForFunction(() => (document.getElementById('video') as HTMLVideoElement).currentTime > 0.8);

  await page.keyboard.press('b');
  await expect(page.locator('#player-chip')).toHaveText('Player B');
  await page.keyboard.press('p');
  await expect(page.locator('#entry')).toBeVisible();
  await expect(page.locator('#entry-type')).toHaveText('card played');
  expect(await page.evaluate(() => (document.getElementById('video') as HTMLVideoElement).paused)).toBe(true);
  await page.keyboard.type('fake h');
  await expect(page.locator('#suggestions li').first()).toContainText('Fake Hero');
  await page.keyboard.press('Enter'); // pick the suggestion
  await expect(page.locator('#entry-card')).toHaveValue('Fake Hero');
  await page.keyboard.press('Enter'); // save
  await expect(page.locator('#entry')).toBeHidden();
  await expect(page.locator('#events li')).toHaveCount(1);
  await expect(page.locator('#events li').first()).toContainText('card played: Fake Hero');

  await page.keyboard.press('t');
  await page.keyboard.press('Enter');
  await page.keyboard.press('h');
  await page.keyboard.press('Enter');
  await expect(page.locator('#events li')).toHaveCount(3);
  await page.keyboard.press('z'); // undo the hidden card
  await expect(page.locator('#events li')).toHaveCount(2);

  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#export')]);
  expect(download.suggestedFilename()).toBe('match.timeline.json');
  const doc = JSON.parse(readFileSync(fileURLToPath(new URL(`file://${await download.path()}`)), 'utf8')) as TimelineDocument;
  expect(validateTimelineDocument(doc)).toEqual([]);
  expect(doc.match.players).toEqual({ A: 'Ann' });
  expect(doc.match.legends).toEqual({ A: 'fake-legend' });
  expect(doc.events.map((e) => e.type).sort()).toEqual(['card_played', 'turn_start']);
  const played = doc.events.find((e) => e.type === 'card_played')!;
  expect(played).toMatchObject({ player: 'B', zone: 'base', card: { cardId: 'fake-hero', confidence: 1 } });
  expect(played.t).toBeGreaterThan(0.5);

  // Reopening the same recording offers to continue from the browser's autosave.
  await page.reload();
  await page.setInputFiles('#open-video', { name: 'match.webm', mimeType: 'video/webm', buffer: video });
  await expect(page.locator('#events li')).toHaveCount(2);
});
