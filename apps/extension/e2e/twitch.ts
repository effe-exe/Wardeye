// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A stand-in Twitch for the browser tests: a page at https://www.twitch.tv/... (routed, never fetched) with one
// video, recorded in the browser itself so that no media is in git.

import type { BrowserContext, Page } from '@playwright/test';

/** Records a WebM of two coloured blocks on a dark floor (one drifts a little), `seconds` long, in the browser. */
export async function makeVideo(page: Page, seconds: number, colours: readonly [string, string]): Promise<Buffer> {
  const b64 = await page.evaluate(
    async ({ seconds, colours }) => {
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
          ctx.fillStyle = colours[0];
          ctx.fillRect(100, 100, 60, 84);
          ctx.fillStyle = colours[1];
          ctx.fillRect(300 + ((performance.now() - t0) / 200) % 20, 120, 60, 84);
          if (performance.now() - t0 < seconds * 1000) requestAnimationFrame(draw);
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
    },
    { seconds, colours },
  );
  return Buffer.from(b64, 'base64');
}

/** Every page of twitch.tv is the stand-in: /test.webm is the video, the rest a page that plays it (or, with
 * `video` false, a page with none). */
export async function routeTwitch(context: BrowserContext, webm: Buffer | null): Promise<void> {
  await context.route('https://www.twitch.tv/**', (route) => {
    const url = route.request().url();
    if (webm && url.endsWith('/test.webm')) return route.fulfill({ status: 200, contentType: 'video/webm', body: webm });
    const video = webm ? '<video id="v" src="/test.webm" muted autoplay playsinline style="width:960px;height:540px"></video>' : '<p>a Twitch page with no video</p>';
    return route.fulfill({
      status: 200,
      contentType: 'text/html',
      body: `<!doctype html><html><body style="margin:0;background:#000">${video}</body></html>`,
    });
  });
}
