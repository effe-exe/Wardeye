// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A stand-in YouTube for the browser tests: pages at https://www.youtube.com/... (routed, never fetched). Each holds the
// player as sites.ts finds it (#movie_player, its video, its clock) and, outside it, a bigger preview video, which Wardeye
// must leave alone. The video is recorded in the browser (twitch.ts), so that no media is in git.

import type { BrowserContext, Page } from '@playwright/test';
import { makeVideo } from './twitch';

/** Every page of youtube.com is the stand-in: /test.webm is the video, the rest a page with the player and a preview, under
 * the video's `title`. */
export async function routeYouTube(context: BrowserContext, webm: Buffer, title: string): Promise<void> {
  await context.route('https://www.youtube.com/**', (route) => {
    if (route.request().url().endsWith('/test.webm')) return route.fulfill({ status: 200, contentType: 'video/webm', body: webm });
    return route.fulfill({
      status: 200,
      contentType: 'text/html',
      body:
        `<!doctype html><html><head><title>(3) ${title} - YouTube</title></head>` +
        '<body style="margin:0;background:#000">' +
        '<div id="movie_player" class="html5-video-player">' +
        '<video class="html5-main-video video-stream" src="/test.webm" muted autoplay playsinline style="width:960px;height:540px"></video>' +
        '<div class="ytp-time-display"></div></div>' +
        '<video id="preview" src="/test.webm" muted autoplay playsinline style="width:1200px;height:675px"></video>' +
        '</body></html>',
    });
  });
}

/** The stand-in YouTube, with its video (`seconds` long) recorded first. */
export async function youtube(context: BrowserContext, seconds = 16, title = 'Riftbound test'): Promise<Page> {
  const page = await context.newPage();
  await page.goto('about:blank');
  await routeYouTube(context, await makeVideo(page, seconds, ['#c0a040', '#40a0c0']), title);
  return page;
}
