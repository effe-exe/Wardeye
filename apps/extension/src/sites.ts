// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The sites Wardeye reads, one adapter each. An adapter says which pages hold a video to read, what names that video (each
// video gets its own board, plays and decklists), which <video> is the player, when the picture is not the stream (an ad),
// and whether a play can be jumped back to. Everything that knows a site's addresses or its page structure lives here,
// because that is what breaks when a site changes (docs/research/09-risks-and-open-questions.md, R8). The tests in
// test/sites.test.ts say what each adapter expects of the page.

/** The little of a page an adapter looks at: a Document in the browser, a fake in the tests. */
export interface PageLike<V = unknown> {
  querySelector(selectors: string): unknown;
  querySelectorAll(selectors: string): ArrayLike<V>;
}

export interface Site {
  readonly name: 'Twitch' | 'YouTube';
  /** What names the video on this page, or null when the page holds no video to read. */
  key(url: URL): string | null;
  /** The <video> elements that may be the player; the overlay takes the largest that has a picture. */
  videos<V>(page: PageLike<V>): V[];
  /** The player shows something other than the stream (an ad): nothing is read, and the overlay is hidden. */
  interrupted(page: PageLike): boolean;
  /** A live stream: its plays cannot be jumped back to. */
  live(url: URL, page: PageLike, video: { duration: number } | null): boolean;
  /** The page's title, without what the site adds to every page. */
  title(raw: string): string;
  /** Wardeye starts by itself on this video (its page's title says what it is); on any other the viewer turns it on. */
  startsOn(title: string): boolean;
  /** The engine is made ready as soon as a page of the site opens, so that the first frame is read sooner; otherwise only once
   * there is a video to read. */
  readonly early: boolean;
}

export const TWITCH: Site = {
  name: 'Twitch',
  // every page is read as it always was: its path names the video (/videos/123 a replay, /name a channel)
  key: (url) => url.pathname,
  videos: (page) => Array.from(page.querySelectorAll('video')),
  interrupted: () => false,
  live: (url) => !/^\/videos\/\d+/.test(url.pathname),
  title: (raw) => raw,
  startsOn: () => true, // as it always has: the streams of Twitch one opens
  early: true,
};

/** A YouTube video id: 11 letters, digits, '-' or '_'. */
const VIDEO_ID = /^[\w-]{11}$/;

export const YOUTUBE: Site = {
  name: 'YouTube',
  // Only a video's own page: /watch?v=ID, or /live/ID. Not the home page and search results (their hover previews are
  // videos too), not Shorts, not a channel's page. The other query parameters (a playlist, a start time) name no other video.
  key(url) {
    if (url.pathname === '/watch') {
      const v = url.searchParams.get('v');
      return v !== null && VIDEO_ID.test(v) ? `/watch?v=${v}` : null;
    }
    const live = /^\/live\/([\w-]{11})\/?$/.exec(url.pathname);
    return live ? `/watch?v=${live[1]}` : null;
  },
  // the main player is #movie_player; the previews of other videos play elsewhere on the page
  videos: (page) => Array.from(page.querySelectorAll('#movie_player video')),
  // the player gets these classes while an ad plays in it (checked against YouTube's player stylesheet, October 2026)
  interrupted: (page) => page.querySelector('#movie_player.ad-showing, #movie_player.ad-interrupting') !== null,
  // the player's clock says LIVE on a live stream; a stream that has no end yet is live too
  live: (_url, page, video) => page.querySelector('#movie_player .ytp-time-display.ytp-live') !== null || !Number.isFinite(video?.duration ?? Infinity),
  // "(3) Riftbound Regional Qualifier - YouTube": the count of notifications and the site's name go
  title: (raw) => raw.replace(/^\(\d+\+?\)\s*/, '').replace(/\s+-\s+YouTube$/, ''),
  // YouTube is mostly other videos, and reading one costs the computer for nothing: Wardeye starts by itself only on a video
  // whose title names the game (in English, or its Chinese name)
  startsOn: (title) => /riftbound|符文战场/i.test(title),
  early: false, // most pages of YouTube are not a Riftbound video: the engine, and Riot's card list, wait for one
};

/** The site of a page, or null for a page of no site Wardeye reads. */
export function siteOf(url: URL): Site | null {
  if (url.protocol !== 'https:') return null;
  if (url.hostname === 'www.twitch.tv') return TWITCH;
  if (url.hostname === 'www.youtube.com') return YOUTUBE;
  return null;
}
