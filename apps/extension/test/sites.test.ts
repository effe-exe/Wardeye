import { describe, expect, it } from 'vitest';
import { TWITCH, YOUTUBE, siteOf, type PageLike } from '../src/sites';

/** A page that holds only the elements named here: what each selector finds. */
function page(found: Record<string, unknown[]> = {}): PageLike<unknown> {
  return {
    querySelector: (s) => found[s]?.[0] ?? null,
    querySelectorAll: (s) => found[s] ?? [],
  };
}
const url = (path: string, host = 'www.youtube.com') => new URL(`https://${host}${path}`);

describe('which site a page is', () => {
  it('knows twitch.tv and youtube.com, over https, and nothing else', () => {
    expect(siteOf(new URL('https://www.twitch.tv/videos/1'))).toBe(TWITCH);
    expect(siteOf(new URL('https://www.youtube.com/watch?v=dQw4w9WgXcQ'))).toBe(YOUTUBE);
    expect(siteOf(new URL('http://www.youtube.com/watch?v=dQw4w9WgXcQ'))).toBeNull();
    expect(siteOf(new URL('https://m.youtube.com/watch?v=dQw4w9WgXcQ'))).toBeNull();
    expect(siteOf(new URL('https://example.com/'))).toBeNull();
  });
});

describe('Twitch, as before', () => {
  it('names a video by its path, and only a replay can be jumped back in', () => {
    expect(TWITCH.key(url('/videos/12345', 'www.twitch.tv'))).toBe('/videos/12345');
    expect(TWITCH.key(url('/somechannel', 'www.twitch.tv'))).toBe('/somechannel');
    expect(TWITCH.live(url('/videos/12345', 'www.twitch.tv'), page(), null)).toBe(false);
    expect(TWITCH.live(url('/somechannel', 'www.twitch.tv'), page(), null)).toBe(true);
  });

  it('takes any video on the page, and never sees an ad', () => {
    expect(TWITCH.videos(page({ video: ['a', 'b'] }))).toEqual(['a', 'b']);
    expect(TWITCH.interrupted(page({ '#movie_player.ad-showing, #movie_player.ad-interrupting': ['x'] }))).toBe(false);
    expect(TWITCH.title('Riftbound Regional - Twitch')).toBe('Riftbound Regional - Twitch');
  });

  it('starts by itself on every stream, as it always has', () => {
    expect(TWITCH.startsOn('Just Chatting')).toBe(true);
  });
});

describe('YouTube', () => {
  it('names a video by its id, whatever else the address carries', () => {
    expect(YOUTUBE.key(url('/watch?v=dQw4w9WgXcQ'))).toBe('/watch?v=dQw4w9WgXcQ');
    expect(YOUTUBE.key(url('/watch?v=dQw4w9WgXcQ&list=PL123&t=95s'))).toBe('/watch?v=dQw4w9WgXcQ');
    expect(YOUTUBE.key(url('/watch?t=95s&v=dQw4w9WgXcQ'))).toBe('/watch?v=dQw4w9WgXcQ');
    expect(YOUTUBE.key(url('/live/dQw4w9WgXcQ'))).toBe('/watch?v=dQw4w9WgXcQ');
    expect(YOUTUBE.key(url('/live/dQw4w9WgXcQ/'))).toBe('/watch?v=dQw4w9WgXcQ');
  });

  it('reads no page but a video\'s own: not the home page, search, Shorts, a channel or a bad id', () => {
    for (const path of ['/', '/results?search_query=riftbound', '/shorts/dQw4w9WgXcQ', '/@riftbound', '/@riftbound/live', '/watch', '/watch?v=short', '/watch?v=dQw4w9WgXcQ0', '/embed/dQw4w9WgXcQ']) {
      expect(YOUTUBE.key(url(path)), path).toBeNull();
    }
  });

  it('takes only the main player\'s video, not the previews elsewhere on the page', () => {
    expect(YOUTUBE.videos(page({ '#movie_player video': ['main'], video: ['main', 'preview'] }))).toEqual(['main']);
    expect(YOUTUBE.videos(page({ video: ['preview'] }))).toEqual([]);
  });

  it('knows an ad by the player\'s classes', () => {
    expect(YOUTUBE.interrupted(page())).toBe(false);
    expect(YOUTUBE.interrupted(page({ '#movie_player.ad-showing, #movie_player.ad-interrupting': ['player'] }))).toBe(true);
  });

  it('is live when the player\'s clock says so, or the stream has no end yet', () => {
    const live = page({ '#movie_player .ytp-time-display.ytp-live': ['clock'] });
    expect(YOUTUBE.live(url('/watch?v=dQw4w9WgXcQ'), live, { duration: 3600 })).toBe(true);
    expect(YOUTUBE.live(url('/watch?v=dQw4w9WgXcQ'), page(), { duration: Infinity })).toBe(true);
    expect(YOUTUBE.live(url('/watch?v=dQw4w9WgXcQ'), page(), { duration: 3600 })).toBe(false);
    expect(YOUTUBE.live(url('/watch?v=dQw4w9WgXcQ'), page(), null)).toBe(true); // no player yet: nothing to jump in
  });

  it('starts by itself only on a video whose title names Riftbound', () => {
    expect(YOUTUBE.startsOn('(3) Riftbound National Chengdu - Day 2 Top Cut - YouTube')).toBe(true);
    expect(YOUTUBE.startsOn("Is this the best ZED deck in the game? | Miracle Returns for Riftbound's Newest Set, Radiance - YouTube")).toBe(true);
    expect(YOUTUBE.startsOn('RIFTBOUND gameplay')).toBe(true);
    expect(YOUTUBE.startsOn('符文战场 成都站')).toBe(true);
    expect(YOUTUBE.startsOn('Pasta carbonara in 10 minutes - YouTube')).toBe(false);
    expect(YOUTUBE.startsOn('YouTube')).toBe(false); // a page whose title has not come yet
  });

  it('drops what YouTube adds to every title', () => {
    expect(YOUTUBE.title('(3) Riftbound Regional Qualifier - YouTube')).toBe('Riftbound Regional Qualifier');
    expect(YOUTUBE.title('(99+) Finals - Day 2 - YouTube')).toBe('Finals - Day 2');
    expect(YOUTUBE.title('YouTube')).toBe('YouTube');
  });
});
