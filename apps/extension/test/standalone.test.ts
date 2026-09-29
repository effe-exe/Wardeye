import { describe, expect, it, vi } from 'vitest';
import { badge, frameInterval } from '../src/geometry';
import { Standalone, UNAVAILABLE_RETRY_MS, cannotReadState, type Env } from '../src/standalone';
import type { EngineReply, EngineRequest } from '../src/protocol';

const PKG = JSON.stringify({
  format: 1,
  runtime: 'auto',
  detector: { id: 'd', fp16: 'models/d.fp16.onnx' },
  embedder: { id: 'e', fp16: 'models/e.fp16.onnx' },
});

function setup(files: Record<string, Uint8Array | string> = { 'standalone.json': PKG }, store = false) {
  let clock = 0;
  const sent: EngineRequest[] = [];
  const st = { ensured: 0, replies: [] as (EngineReply | Error | unknown)[], clock: () => clock };
  const bytes = (v: Uint8Array | string) => (typeof v === 'string' ? new TextEncoder().encode(v) : v);
  const env: Env = {
    now: () => clock,
    read: async (path) => (path in files ? bytes(files[path]!) : null),
    ensureDocument: async () => void st.ensured++,
    send: async (request) => {
      sent.push(request);
      const next = st.replies.length > 0 ? st.replies.shift() : { kind: 'state', state: null };
      if (next instanceof Error) throw next;
      return next;
    },
  };
  const s = new Standalone(env, 60_000, store);
  return { s, sent, st, tick: (ms: number) => void (clock += ms) };
}

const frame = { kind: 'frame' as const, t: 3, video: '/videos/1', jpeg: 'QUJD' };
const state = { t: 3, status: 'live', message: '', frame: { width: 1, height: 1 }, tracks: [] };

describe('standalone mode, from the worker', () => {
  it('is not there in the public build: nothing is made, nothing asked', async () => {
    const { s, sent, st } = setup({});
    expect(await s.usable()).toBe(false);
    expect(await s.frame(1, frame)).toBeNull();
    await s.warm();
    expect(st.ensured).toBe(0);
    expect(sent).toEqual([]);
    expect(await s.art('SFD-195a')).toBeNull();
  });

  it('is not used when standalone.json cannot be read', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { s } = setup({ 'standalone.json': '{"format": 9}' });
    expect(await s.usable()).toBe(false);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('format 9'));
    warn.mockRestore();
  });

  it('is not used when the package says to use the live runner, and nothing is made for it', async () => {
    const { s, sent, st } = setup({ 'standalone.json': PKG.replace('"auto"', '"companion"') });
    expect(await s.usable()).toBe(false);
    await s.warm();
    expect(await s.frame(1, frame)).toBeNull();
    expect(st.ensured).toBe(0);
    expect(sent).toEqual([]);
  });

  it('makes the document when a tab connects, and asks it whether the engine can run', async () => {
    const { s, sent, st } = setup();
    st.replies.push({ kind: 'hello' });
    await s.warm();
    expect(st.ensured).toBe(1);
    expect(sent).toEqual([{ target: 'engine', kind: 'hello' }]);
    expect(await s.usable()).toBe(true);
  });

  it("hands a tab's frame to the document, under the tab's id, and its state back", async () => {
    const { s, sent, st } = setup();
    st.replies.push({ kind: 'state', state });
    expect(await s.frame(4, frame)).toEqual({ state, events: [] });
    expect(sent).toEqual([{ target: 'engine', kind: 'frame', tab: 4, t: 3, video: '/videos/1', jpeg: 'QUJD' }]);
    st.replies.push({ kind: 'state', state: null }); // passed over: the board on screen stays
    expect(await s.frame(4, frame)).toEqual({ state: null, events: [] });
    // what happened on the table comes with the board, for the plays panel
    const played = { t: 3, kind: 'played', text: 'Sivir, Ambitious played', printing_id: 'SFD-120', track: 't71', side: 'right' };
    st.replies.push({ kind: 'state', state, events: [played] });
    expect(await s.frame(4, frame)).toEqual({ state, events: [played] });
  });

  it('is written off when the document says the engine cannot run here, and asked again a while later', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { s, sent, st, tick } = setup();
    st.replies.push({ kind: 'unavailable', reason: 'this browser has no WebGPU adapter' });
    expect(await s.frame(1, frame)).toBeNull(); // the live runner takes this one
    expect(s.lastReason).toContain('no WebGPU adapter');
    expect(await s.usable()).toBe(false);
    const n = sent.length;
    expect(await s.frame(1, frame)).toBeNull();
    expect(sent.length).toBe(n); // not asked again yet
    tick(61_000);
    expect(await s.usable()).toBe(true);
    st.replies.push({ kind: 'state', state });
    expect(await s.frame(1, frame)).toEqual({ state, events: [] });
    warn.mockRestore();
  });

  it('makes the document again, once, when it is gone', async () => {
    const { s, st } = setup();
    st.replies.push(new Error('Could not establish connection. Receiving end does not exist.'), { kind: 'state', state });
    expect(await s.frame(1, frame)).toEqual({ state, events: [] });
    expect(st.ensured).toBe(2);
  });

  it('gives up on a document that never answers, and uses the live runner', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { s, st } = setup();
    st.replies.push(new Error('gone'), new Error('gone'));
    expect(await s.frame(1, frame)).toBeNull();
    expect(st.ensured).toBe(2);
    expect(s.lastReason).toContain('did not answer');
    warn.mockRestore();
  });

  it('does not take an answer of the wrong shape for one', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { s, st } = setup();
    st.replies.push({ nonsense: true }, undefined);
    expect(await s.frame(1, frame)).toBeNull();
    warn.mockRestore();
  });

  it("lets a tab's board go without waking a document that is not there", async () => {
    const { s, sent } = setup({});
    await s.forget(3);
    expect(sent).toEqual([]);
    const on = setup();
    await on.s.forget(3);
    expect(on.sent).toEqual([{ target: 'engine', kind: 'forget', tab: 3 }]);
    expect(on.st.ensured).toBe(0);
  });

  it("serves a card's hover picture from the package, as base64", async () => {
    const jpeg = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]);
    const { s } = setup({ 'standalone.json': PKG, 'data/thumbs/OGN-299_2a.jpg': jpeg });
    expect(await s.art('OGN-299*')).toBe(btoa(String.fromCharCode(...jpeg)));
    expect(await s.art('SFD-195a')).toBeNull(); // no picture for it
  });
});

describe('standalone mode in the store build, which has no live runner', () => {
  it("gets a card's picture from the engine document, which holds Riot's card list: no file of the package is read", async () => {
    const { s, sent, st } = setup({ 'standalone.json': PKG }, true);
    st.replies.push({ kind: 'art', jpeg: 'QUJD' });
    expect(await s.art('OGN-299*')).toBe('QUJD');
    expect(sent).toEqual([{ target: 'engine', kind: 'art', printing_id: 'OGN-299*' }]);
    expect(st.ensured).toBe(1); // the document is made when it is gone: it holds the list
    st.replies.push({ kind: 'art', jpeg: null });
    expect(await s.art('OGN-001')).toBeNull();
  });

  it('has no picture when the document does not answer, or answers something else, or when there is no package at all', async () => {
    const { s, st } = setup({ 'standalone.json': PKG }, true);
    st.replies.push(new Error('gone'), new Error('gone'));
    expect(await s.art('OGN-001')).toBeNull();
    st.replies.push({ kind: 'hello' }, { kind: 'hello' });
    expect(await s.art('OGN-001')).toBeNull();
    const none = setup({}, true);
    expect(await none.s.art('OGN-001')).toBeNull();
    expect(none.sent).toEqual([]);
  });

  it('asks the engine even when standalone.json says to use the live runner: the document says it has nothing to run', async () => {
    const { s, sent, st } = setup({ 'standalone.json': PKG.replace('"auto"', '"companion"') }, true);
    expect(await s.usable()).toBe(true);
    st.replies.push({ kind: 'unavailable', reason: 'standalone.json says to use the live runner, which this build does not have' });
    expect(await s.frame(1, frame)).toBeNull();
    expect(sent).toHaveLength(1);
    expect(s.lastReason).toContain('this build does not have');
  });

  it('does not say the live runner is used when the engine is written off', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const store = setup({ 'standalone.json': PKG }, true);
    store.st.replies.push({ kind: 'unavailable', reason: 'nothing to run' });
    await store.s.frame(1, frame);
    expect(warn).toHaveBeenLastCalledWith(expect.stringContaining('no engine in this browser (nothing to run); the overlay says so'));
    const dev = setup();
    dev.st.replies.push({ kind: 'unavailable', reason: 'nothing to run' });
    await dev.s.frame(1, frame);
    expect(warn).toHaveBeenLastCalledWith(expect.stringContaining('the live runner is used'));
    warn.mockRestore();
  });

  it('tells the overlay plainly why there is no board, in the badge, and asks for a frame now and then, not four times a second', () => {
    const state = cannotReadState(12, false);
    expect(badge(true, state)).toBe('Wardeye: this browser cannot run the engine (WebGPU or WebAssembly needed)');
    expect(state).toMatchObject({ t: 12, status: 'error', frame: { width: 0, height: 0 }, tracks: [] });
    expect(frameInterval(state)).toBe(UNAVAILABLE_RETRY_MS);
    expect(frameInterval(state)).toBeGreaterThanOrEqual(1000);
    expect(badge(true, cannotReadState(12, true))).toBe("Wardeye: the engine's files are missing from this install, reinstall Wardeye");
  });
});
