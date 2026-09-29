import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';
import {
  FEED_URL, Feed, IMAGE_ORIGIN, MAX_RETRIES, PAGE_SIZE, PICTURES_KEPT, RETRY_MS, Pictures, looksLikePicture, parseCode, parseFeed, pictureUrl, slugify, variantOf,
  type FeedCard, type FeedEnv,
} from '../src/feed';

// Made-up cards in the shape of Riot's gallery items: no real name, text or picture is in the repository.
const PICTURE = (code: string): string => `https://cmsassets.rgpub.io/made-up/${code.replace(/[^A-Za-z0-9]/g, '-')}.png?accountingTag=RB`;
const item = (publicCode: string, name: string, over: Record<string, unknown> = {}): Record<string, unknown> => ({
  publicCode,
  name,
  cardType: { type: [{ id: 'unit', label: 'Unit' }] },
  cardImage: { url: PICTURE(publicCode) },
  set: { value: { id: 'TST', label: 'Test Set' } },
  rarity: { value: { id: 'common', label: 'Common' } },
  domain: { values: [{ id: 'fury', label: 'Fury' }, { id: 'calm', label: 'Calm' }] },
  energy: { value: { id: 3, label: '3' } },
  text: { richText: { body: 'Made-up rules text that must never be kept.' } },
  ...over,
});
const types = (...labels: string[]): Record<string, unknown> => ({ cardType: { type: labels.map((label) => ({ label })) } });

const byId = (cards: FeedCard[]): Record<string, FeedCard> => Object.fromEntries(cards.map((c) => [c.printing_id, c]));

/** A small synthetic gallery: one of each kind of printing the reading has a rule for. */
const gallery = (): unknown[] => [
  item('TST-001/100', 'Zephyr Knight', { subtitle: 'the Bold' }), // a champion unit: its subtitle is part of its name
  item('TST-002/100', 'Ember Bolt', { subtitle: 'Zephyr Knight', ...types('Spell') }), // a signature spell: its subtitle is an annotation
  item('TST-001a/100', 'Zephyr Knight', { subtitle: 'the Bold' }), // its alternate art
  item('TST-003*/100', 'Ember Bolt (Signature)', types('Spell')), // a signature printing
  item('TST-T01', 'Wisp', types('Token')), // a token: no total
  item('TST-105/100', 'Grand Wisp', types('Unit')), // numbered past the set
  item('TST-L01/100', 'Old Lantern', { ...types('Legend'), subtitle: 'Starter', tags: { tags: ['Lantern Keeper'] }, domain: { values: [{ label: 'Calm' }, { label: 'Order' }, { label: '' }] } }),
  item('TST-B01/100', 'Quiet Field', types('Battlefield')),
  item('TST-R01', 'Fury Rune', types('Rune')),
  item('TST-C01/100', 'Twin Blade', types('Champion', 'Unit')), // two labels: a type of its own, not "Unit"
  item('TST-004/100', 'Duplicate', { cardImage: { url: 'https://cmsassets.rgpub.io/made-up/first.png?accountingTag=RB' } }),
  item('TST-004/100', 'Duplicate Later', { cardImage: { url: 'https://cmsassets.rgpub.io/made-up/second.png' } }), // the same printing: the first wins
  item('TST-005/100', 'No Picture', { cardImage: {} }),
  item('TST-006/100', 'No Picture Either', { cardImage: undefined }),
  item('TST-007/100', '   '), // no name
  item('TST-008/100', undefined as unknown as string), // no name at all
  item('', 'No Code'),
  item('not a code', 'Bad Code'),
  item('tst-009/100', 'Lower Case Set'), // sets are upper case
  null,
  'nonsense',
];

describe("the gallery's collector codes", () => {
  it('give the printing id, and the set, number and total that made it', () => {
    expect(parseCode('OGN-007a/298')).toEqual({ set: 'OGN', num: '007a', total: 298, printing_id: 'OGN-007a' });
    expect(parseCode('UNL-131/219')).toEqual({ set: 'UNL', num: '131', total: 219, printing_id: 'UNL-131' });
    expect(parseCode('OGN-299*/298')).toEqual({ set: 'OGN', num: '299*', total: 298, printing_id: 'OGN-299*' });
    expect(parseCode('VEN-T04')).toEqual({ set: 'VEN', num: 'T04', total: null, printing_id: 'VEN-T04' });
    expect(parseCode('  SFD-020/221  ')?.printing_id).toBe('SFD-020');
  });

  it('are refused when they are not collector codes', () => {
    for (const code of ['', 'nonsense', 'T-1/2', 'ABCDE-001', 'ogn-001', 'OGN-', 'OGN-001/', 'OGN-001b2', 'OGN 001', 'OGN-001/298/2']) expect(parseCode(code), code).toBeNull();
  });

  it('tell a printing from the card it prints: token, signature, alternate art, numbered past the set', () => {
    const v = (code: string, type = 'Unit', alt = false) => variantOf(parseCode(code)!, type, alt);
    expect(v('VEN-T04')).toBe('token');
    expect(v('OGN-010/298', 'Token')).toBe('token');
    expect(v('OGN-299*/298')).toBe('signature');
    expect(v('OGN-007a/298')).toBe('alt_art');
    expect(v('OGN-007b/298')).toBe('alt_art');
    expect(v('OGN-007/298', 'Unit', true)).toBe('alt_art');
    expect(v('OGN-298/298')).toBe('standard');
    expect(v('OGN-299/298')).toBe('overnumbered');
    expect(v('OGN-050')).toBe('standard'); // no total, nothing to be past
  });
});

describe("a card's identity", () => {
  it('is its name folded: same name, same card, whatever the annotation, accent or case', () => {
    expect(slugify('Zephyr Knight')).toBe('zephyr-knight');
    expect(slugify('Zephyr Knight (Alternate Art)')).toBe('zephyr-knight');
    expect(slugify('Zephyr Knight (alt art)')).toBe('zephyr-knight');
    expect(slugify('Ember Bolt (Signature)')).toBe('ember-bolt');
    expect(slugify('Ember Bolt (OVERNUMBERED 2)')).toBe('ember-bolt');
    expect(slugify('Ember (Bolt) (Promo)')).toBe('ember-bolt'); // only the listed annotations go; the parenthesis after it
    expect(slugify('Crème Brûlée!')).toBe('creme-brulee');
    expect(slugify("Kai'Sa_Two")).toBe('kai-sa-two');
    expect(slugify('  --Hello,   World--  ')).toBe('hello-world');
  });

  it('keeps letters of other scripts, and is "unknown" for a name with none', () => {
    expect(slugify('示例卡牌')).toBe('示例卡牌');
    expect(slugify('Ольга Кнехт')).toBe('ольга-кнехт');
    expect(slugify('!!!')).toBe('unknown');
    expect(slugify('')).toBe('unknown');
  });
});

describe("the gallery's items as printings", () => {
  const cards = byId(parseFeed(gallery()));

  it('are the items with a code, a picture and a name: the rest are skipped, and the first of a printing wins', () => {
    expect(Object.keys(cards)).toEqual(['TST-001', 'TST-002', 'TST-001a', 'TST-003*', 'TST-T01', 'TST-105', 'TST-L01', 'TST-B01', 'TST-R01', 'TST-C01', 'TST-004']);
    expect(cards['TST-004']).toMatchObject({ name: 'Duplicate', image_url: 'https://cmsassets.rgpub.io/made-up/first.png?accountingTag=RB' });
  });

  it("name a champion unit with its subtitle, and give any other card its name (a subtitle there is an annotation)", () => {
    expect(cards['TST-001']).toMatchObject({ name: 'Zephyr Knight, the Bold', card_id: 'zephyr-knight-the-bold', type: 'Unit' });
    expect(cards['TST-002']).toMatchObject({ name: 'Ember Bolt', card_id: 'ember-bolt', type: 'Spell' });
    expect(cards['TST-L01']).toMatchObject({ name: 'Old Lantern', type: 'Legend' });
    expect(cards['TST-C01']).toMatchObject({ name: 'Twin Blade', type: 'Champion Unit' }); // the labels joined by a space
  });

  it('give an alternate art and a signature printing the card id of the card, and their own printing ids', () => {
    expect(cards['TST-001a']).toMatchObject({ printing_id: 'TST-001a', card_id: 'zephyr-knight-the-bold', variant: 'alt_art' });
    expect(cards['TST-001']!.variant).toBe('standard');
    expect(cards['TST-003*']).toMatchObject({ printing_id: 'TST-003*', card_id: 'ember-bolt', variant: 'signature' });
    expect(cards['TST-002']!.card_id).toBe(cards['TST-003*']!.card_id); // the signature printing of the spell is the spell
  });

  it('read a token, and a card numbered past its set', () => {
    expect(cards['TST-T01']).toMatchObject({ type: 'Token', variant: 'token', card_id: 'wisp' });
    expect(cards['TST-105']).toMatchObject({ variant: 'overnumbered' });
  });

  it('keep each printing\'s domains, labels only, as catalog.py writes them (and drop what has no label)', () => {
    expect(cards['TST-001']!.domains).toEqual(['Fury', 'Calm']);
    expect(cards['TST-L01']!.domains).toEqual(['Calm', 'Order']);
    expect(cards['TST-B01']!.domains).toEqual(['Fury', 'Calm']);
    expect(parseFeed([item('TST-050/100', 'Blank', { domain: undefined })])[0]!.domains).toEqual([]);
    expect(parseFeed([item('TST-051/100', 'Blank', { domain: { values: 'Fury' } })])[0]!.domains).toEqual([]);
  });

  it("keep a Legend's tags, and no other card's", () => {
    expect(cards['TST-L01']!.tags).toEqual(['Lantern Keeper']);
    expect(parseFeed([item('TST-052/100', 'A Legend', { ...types('Legend') })])[0]!.tags).toEqual([]); // a Legend with none says so
    expect(cards['TST-001']).not.toHaveProperty('tags');
    expect(parseFeed([item('TST-053/100', 'Tagged', { tags: { tags: ['A', 'B'] } })])[0]).not.toHaveProperty('tags');
  });

  it("keep the picture's address, and never the card's text or anything else the gallery says", () => {
    expect(cards['TST-001']!.image_url).toBe(PICTURE('TST-001/100'));
    expect(Object.keys(cards['TST-001']!).sort()).toEqual(['card_id', 'domains', 'image_url', 'name', 'printing_id', 'type', 'variant']);
    expect(JSON.stringify(parseFeed(gallery()))).not.toContain('rules text');
  });
});

/** A fake gallery of `n` printings, served by the page the URL asks for: what the real one answers, less its weight. */
function server(n: number, over: (page: number, body: Record<string, unknown>) => Record<string, unknown> = (_p, b) => b) {
  const all = Array.from({ length: n }, (_, i) => item(`TST-${String(i + 1).padStart(3, '0')}/${n}`, `Card ${i + 1}`));
  const asked: string[] = [];
  const getJson = async (url: string): Promise<unknown> => {
    asked.push(url);
    const q = new URL(url).searchParams;
    const from = Number(q.get('from'));
    const limit = Number(q.get('limit'));
    return over(from / limit, { metadata: { totalItems: n, totalPages: Math.ceil(n / limit) }, data: all.slice(from, from + limit) });
  };
  return { asked, getJson };
}

function feedOf(getJson: FeedEnv['getJson']) {
  const timers: { run: () => void; ms: number }[] = [];
  const feed = new Feed({ getJson, later: (run, ms) => void timers.push({ run, ms }) });
  return { feed, timers };
}

describe('the card list', () => {
  let warn: MockInstance;
  let info: MockInstance;
  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    info = vi.spyOn(console, 'info').mockImplementation(() => {});
  });
  afterEach(() => {
    warn.mockRestore();
    info.mockRestore();
  });

  it('is read a page at a time, 200 items from each `from`, until the pages the feed says it has', async () => {
    const s = server(450);
    const { feed, timers } = feedOf(s.getJson);
    expect(feed.loaded).toBe(false);
    expect(feed.rows()).toEqual([]);
    await feed.settled();
    expect(s.asked).toEqual([0, 200, 400].map((from) => `${FEED_URL}?locale=en_US&from=${from}&limit=${PAGE_SIZE}`));
    expect(feed.loaded).toBe(true);
    expect(feed.rows()).toHaveLength(450);
    expect(feed.rows()[0]).toEqual({ printing_id: 'TST-001', card_id: 'card-1', name: 'Card 1', type: 'Unit', domains: ['Fury', 'Calm'], variant: 'standard' }); // no picture address
    expect(feed.imageUrl('TST-450')).toBe(PICTURE('TST-450/450'));
    expect(feed.imageUrl('TST-999')).toBeUndefined();
    expect(timers).toEqual([]); // nothing to try again
    expect(info).toHaveBeenCalledWith(expect.stringContaining('450 printings'));
  });

  it('is read once, whoever asks and whenever', async () => {
    const s = server(10);
    const { feed } = feedOf(s.getJson);
    await Promise.all([feed.settled(), feed.settled()]);
    await feed.settled();
    expect(s.asked).toHaveLength(1);
  });

  it('stops where the pages end even when the feed gives no count of them, and never runs on', async () => {
    const noPages = server(300, (_p, b) => ({ data: b.data }));
    await feedOf(noPages.getJson).feed.settled();
    expect(noPages.asked).toHaveLength(2); // a full page, then a short one
    const totalOnly = server(450, (_p, b) => ({ metadata: { totalItems: 450 }, data: b.data }));
    await feedOf(totalOnly.getJson).feed.settled();
    expect(totalOnly.asked).toHaveLength(3);
    const endless = server(7000, (_p, b) => ({ metadata: { totalPages: 1e9 }, data: b.data }));
    await feedOf(endless.getJson).feed.settled();
    expect(endless.asked).toHaveLength(30); // a stop for a feed that never says where it ends
  });

  it('fails as a whole when a page fails, and keeps nothing of the pages before it', async () => {
    const s = server(450);
    let fail = true;
    const { feed } = feedOf(async (url) => {
      if (fail && url.includes('from=200')) throw new Error('HTTP 503');
      return s.getJson(url);
    });
    await feed.settled();
    expect(feed.loaded).toBe(false);
    expect(feed.rows()).toEqual([]);
    expect(feed.imageUrl('TST-001')).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/could not be read \(HTTP 503\); trying again in 60 s/));
    fail = false;
  });

  it('is asked for again a minute after a try that failed, and the first try settles at once, whatever it says', async () => {
    const s = server(450);
    let up = false;
    const { feed, timers } = feedOf(async (url) => {
      if (!up) throw new Error('offline');
      return s.getJson(url);
    });
    const loaded = vi.fn();
    feed.onLoaded(loaded);
    await feed.settled(); // does not reject
    expect(timers.map((t) => t.ms)).toEqual([RETRY_MS]);
    expect(RETRY_MS).toBe(60_000);
    expect(loaded).not.toHaveBeenCalled();
    timers.shift()!.run(); // a minute later: still offline
    await vi.waitFor(() => expect(timers).toHaveLength(1));
    up = true;
    timers.shift()!.run();
    await vi.waitFor(() => expect(feed.loaded).toBe(true));
    expect(feed.rows()).toHaveLength(450);
    expect(loaded).toHaveBeenCalledTimes(1);
    expect(timers).toEqual([]); // and it is not asked again
  });

  it('is asked for at most five more times, and then left', async () => {
    const { feed, timers } = feedOf(async () => {
      throw new Error('offline');
    });
    await feed.settled();
    for (let retry = 1; retry <= MAX_RETRIES; retry++) {
      expect(timers, `before retry ${retry}`).toHaveLength(1);
      timers.shift()!.run();
      await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(retry + 1));
    }
    expect(MAX_RETRIES).toBe(5);
    expect(timers).toEqual([]); // the fifth try was the last
    expect(feed.loaded).toBe(false);
    expect(warn).toHaveBeenLastCalledWith(expect.stringContaining('not trying again'));
  });

  it('counts a body that is not a list, or that names no card, as a failure', async () => {
    for (const body of [null, 'text', { data: 'nope' }, { metadata: {}, data: [] }, { metadata: { totalPages: 1 }, data: [item('junk', 'X')] }]) {
      const { feed, timers } = feedOf(async () => body);
      await feed.settled();
      expect(feed.loaded, JSON.stringify(body)).toBe(false);
      expect(timers).toHaveLength(1);
    }
  });
});

describe("a card's picture", () => {
  it("is Riot's own, 400 px wide as a JPEG, with the address's own query kept", () => {
    expect(pictureUrl(`${IMAGE_ORIGIN}/made-up/a.png?accountingTag=RB`)).toBe(`${IMAGE_ORIGIN}/made-up/a.png?accountingTag=RB&w=400&fm=jpg&q=80`);
    expect(pictureUrl(`${IMAGE_ORIGIN}/made-up/a.png`)).toBe(`${IMAGE_ORIGIN}/made-up/a.png?w=400&fm=jpg&q=80`);
    expect(pictureUrl(`${IMAGE_ORIGIN}/made-up/a.png?accountingTag=RB#frag`)).toBe(`${IMAGE_ORIGIN}/made-up/a.png?accountingTag=RB&w=400&fm=jpg&q=80`);
  });

  it("is asked of no other place: the extension's manifest gives access to two hosts", () => {
    for (const url of ['http://cmsassets.rgpub.io/a.png', 'https://cmsassets.rgpub.io.example.com/a.png', 'https://example.com/cmsassets.rgpub.io/a.png', 'https://user@example.com/a.png', 'a.png', '']) {
      expect(pictureUrl(url), url).toBeNull();
    }
  });

  it('is told from anything else by its first bytes', () => {
    const bytes = (...b: number[]) => Uint8Array.from([...b, ...new Array(20).fill(0)]);
    expect(looksLikePicture(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe(true);
    expect(looksLikePicture(bytes(0x89, 0x50, 0x4e, 0x47))).toBe(true);
    expect(looksLikePicture(Uint8Array.from([...new TextEncoder().encode('RIFF'), 0, 0, 0, 0, ...new TextEncoder().encode('WEBP'), 0]))).toBe(true);
    expect(looksLikePicture(new TextEncoder().encode('<html><body>Forbidden</body></html>'))).toBe(false);
    expect(looksLikePicture(new Uint8Array(3))).toBe(false);
    expect(looksLikePicture(new Uint8Array(2_000_001).fill(0xff))).toBe(false);
  });
});

describe('the pictures the overlay asks for', () => {
  const jpeg = (n: number): Uint8Array => Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, n & 0xff, n >> 8, 0, 0, 0, 0, 0, 0, 0]);
  const b64 = (bytes: Uint8Array): string => btoa(String.fromCharCode(...bytes));

  function setup(over: { url?: (id: string) => Promise<string | undefined>; bytes?: (url: string) => Promise<Uint8Array | null> } = {}) {
    const got: string[] = [];
    const p = new Pictures({
      imageUrl: over.url ?? (async (id) => `${IMAGE_ORIGIN}/made-up/${id}.png?accountingTag=RB`),
      getBytes:
        over.bytes ??
        (async (url) => {
          got.push(url);
          return jpeg(Number(/(\d+)\.png/.exec(url)?.[1] ?? 0));
        }),
    });
    return { p, got };
  }

  it('come from the gallery as base64, at the size and format asked for', async () => {
    const { p, got } = setup();
    expect(await p.get('7')).toBe(b64(jpeg(7)));
    expect(got).toEqual([`${IMAGE_ORIGIN}/made-up/7.png?accountingTag=RB&w=400&fm=jpg&q=80`]);
  });

  it('are kept, 64 of them, in memory: the one used longest ago goes first', async () => {
    const { p, got } = setup();
    expect(PICTURES_KEPT).toBe(64);
    for (let i = 1; i <= 64; i++) await p.get(String(i));
    expect(p.size).toBe(64);
    await p.get('1'); // used again: no request, and now the newest
    expect(got).toHaveLength(64);
    await p.get('65'); // one too many: 2 goes, not 1
    expect(p.size).toBe(64);
    await p.get('1');
    expect(got).toHaveLength(65); // 1 was kept
    await p.get('2');
    expect(got).toHaveLength(66); // 2 was not: it is fetched again
    for (let i = 100; i < 200; i++) await p.get(String(i));
    expect(p.size).toBe(64);
  });

  it('are asked for once when they are asked for together', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const asked: string[] = [];
    const { p } = setup({
      bytes: async (url) => {
        asked.push(url);
        await gate;
        return jpeg(1);
      },
    });
    const together = Promise.all([p.get('1'), p.get('1'), p.get('1')]);
    release();
    expect(await together).toEqual([b64(jpeg(1)), b64(jpeg(1)), b64(jpeg(1))]);
    expect(asked).toHaveLength(1);
  });

  it('are null when there is none, and asked for again the next time', async () => {
    let fail = true;
    const { p } = setup({
      bytes: async () => {
        if (fail) throw new Error('offline');
        return jpeg(1);
      },
    });
    expect(await p.get('1')).toBeNull();
    expect(p.size).toBe(0);
    fail = false;
    expect(await p.get('1')).toBe(b64(jpeg(1)));
  });

  it('are null for a card the list does not name, an address that is not Riot\'s, and an answer that is not a picture', async () => {
    const asked: string[] = [];
    const { p } = setup({
      url: async (id) => (id === 'none' ? undefined : id === 'elsewhere' ? 'https://example.com/a.png' : `${IMAGE_ORIGIN}/made-up/${id}.png`),
      bytes: async (url) => {
        asked.push(url);
        return url.includes('html') ? new TextEncoder().encode('<html>Not found</html>') : url.includes('nothing') ? null : jpeg(1);
      },
    });
    expect(await p.get('none')).toBeNull();
    expect(await p.get('elsewhere')).toBeNull();
    expect(asked).toEqual([]); // nothing was asked of anyone
    expect(await p.get('html')).toBeNull();
    expect(await p.get('nothing')).toBeNull();
    expect(p.size).toBe(0);
  });
});
