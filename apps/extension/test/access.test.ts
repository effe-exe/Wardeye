import { describe, expect, it } from 'vitest';
import { SCRIPT_ID, YOUTUBE_ORIGINS, clicked, onYouTube, sync, type Browser } from '../src/access';

/** A browser where YouTube is allowed or not, the prompt says `answer`, and what happens is written down. */
function fake(opts: { allowed?: boolean; answer?: boolean; registered?: boolean; present?: boolean } = {}) {
  let allowed = opts.allowed ?? false;
  const scripts = new Set<string>(opts.registered ? [SCRIPT_ID] : []);
  const did: string[] = [];
  const b: Browser = {
    allowed: async () => allowed,
    ask: (origins) => {
      did.push(`ask ${origins.join(' ')}`);
      if (!allowed && opts.answer) allowed = true;
      return Promise.resolve(allowed);
    },
    registered: async () => [...scripts],
    register: async (id, origins) => {
      did.push(`register ${id} ${origins.join(' ')}`);
      scripts.add(id);
    },
    unregister: async (id) => {
      did.push(`unregister ${id}`);
      scripts.delete(id);
    },
    present: async () => opts.present ?? false,
    inject: async (tabId) => void did.push(`inject ${tabId}`),
    toggle: async (tabId) => void did.push(`toggle ${tabId}`),
  };
  return { b, did };
}

describe('the registered script follows the permission', () => {
  it('is registered once YouTube is allowed, and only once', async () => {
    const { b, did } = fake({ allowed: true });
    await sync(b);
    await sync(b);
    expect(did).toEqual([`register ${SCRIPT_ID} ${YOUTUBE_ORIGINS.join(' ')}`]);
  });

  it('goes when the permission is taken back, and nothing happens when it was never given', async () => {
    const taken = fake({ allowed: false, registered: true });
    await sync(taken.b);
    expect(taken.did).toEqual([`unregister ${SCRIPT_ID}`]);
    const never = fake({ allowed: false });
    await sync(never.b);
    expect(never.did).toEqual([]);
  });
});

describe('the toolbar button', () => {
  it('on Twitch, or any page but YouTube, turns Wardeye off or on and asks for nothing', async () => {
    for (const url of ['https://www.twitch.tv/videos/1', 'https://example.com/', undefined]) {
      const { b, did } = fake();
      await clicked(b, { id: 7, ...(url ? { url } : {}) });
      expect(did, String(url)).toEqual(['toggle 7']);
    }
  });

  it('on YouTube the first time asks, registers the script and starts Wardeye in the tab', async () => {
    const { b, did } = fake({ answer: true });
    await clicked(b, { id: 7, url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' });
    expect(did).toEqual([`ask ${YOUTUBE_ORIGINS.join(' ')}`, `register ${SCRIPT_ID} ${YOUTUBE_ORIGINS.join(' ')}`, 'inject 7']);
  });

  it('asks before anything is awaited: the browser allows the question only while it answers the click', () => {
    const { b, did } = fake({ answer: true });
    void clicked(b, { id: 7, url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' });
    expect(did).toEqual([`ask ${YOUTUBE_ORIGINS.join(' ')}`]); // synchronously, in the click
  });

  it('on YouTube, refused, does nothing more', async () => {
    const { b, did } = fake({ answer: false });
    await clicked(b, { id: 7, url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' });
    expect(did).toEqual([`ask ${YOUTUBE_ORIGINS.join(' ')}`]);
  });

  it('on YouTube once Wardeye runs there, turns it off or on, as on Twitch', async () => {
    const { b, did } = fake({ allowed: true, registered: true, present: true });
    await clicked(b, { id: 7, url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' });
    expect(did).toEqual([`ask ${YOUTUBE_ORIGINS.join(' ')}`, 'toggle 7']);
  });

  it('on a YouTube tab opened before access was given, starts Wardeye there without asking again', async () => {
    const { b, did } = fake({ allowed: true, registered: true, present: false });
    await clicked(b, { id: 7, url: 'https://www.youtube.com/@channel' });
    expect(did).toEqual([`ask ${YOUTUBE_ORIGINS.join(' ')}`, 'inject 7']);
  });

  it('does nothing for a tab with no id', async () => {
    const { b, did } = fake();
    await clicked(b, { url: 'https://www.youtube.com/' });
    expect(did).toEqual([]);
  });
});

describe('a YouTube page', () => {
  it('is any page of www.youtube.com, and only that', () => {
    expect(onYouTube('https://www.youtube.com/')).toBe(true);
    expect(onYouTube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')).toBe(true);
    expect(onYouTube('https://www.youtube.com.evil.example/')).toBe(false);
    expect(onYouTube('https://www.twitch.tv/')).toBe(false);
    expect(onYouTube('not a url')).toBe(false);
    expect(onYouTube(undefined)).toBe(false);
  });
});
