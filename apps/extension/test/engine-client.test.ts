import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { StandalonePackage } from '../src/assets';
import { WorkerEngine } from '../src/engine-client';
import type { FromEngine, ToEngine } from '../src/protocol';

const pkg: StandalonePackage = { format: 1, runtime: 'auto', detector: { id: 'd', fp16: 'd16' }, embedder: { id: 'e', fp16: 'e16' }, data: 'data/' };
const attempt = { runtime: 'webgpu' as const, detector: 'fp32' as const, embedder: 'fp16' as const };

/** A Worker that records what it is sent, and says what the test makes it say. */
class FakeWorker {
  static last: FakeWorker;
  sent: ToEngine[] = [];
  terminated = false;
  onmessage: ((e: { data: FromEngine }) => void) | null = null;
  onerror: ((e: { message: string }) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  constructor(readonly url: URL, readonly options: unknown) {
    FakeWorker.last = this;
  }
  postMessage(m: ToEngine): void {
    this.sent.push(m);
  }
  terminate(): void {
    this.terminated = true;
  }
  say(m: FromEngine): void {
    this.onmessage?.({ data: m });
  }
}

const state = { t: 1, status: 'live', message: '', frame: { width: 1, height: 1 }, tracks: [] };

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
  vi.stubGlobal('location', { href: 'chrome-extension://abc/offscreen.html' });
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('the link to the engine worker', () => {
  it('starts a module worker next to the document, and loads it: progress on the way, ready at the end', async () => {
    const e = new WorkerEngine('engine-webgpu.js');
    const w = FakeWorker.last;
    expect(w.url.href).toBe('chrome-extension://abc/engine-webgpu.js');
    expect(w.options).toEqual({ type: 'module' });
    const progress: string[] = [];
    const ready = e.init(pkg, attempt, (m) => progress.push(m));
    expect(w.sent).toEqual([{ kind: 'init', pkg, attempt }]);
    w.say({ kind: 'progress', message: 'reading the gallery' });
    w.say({ kind: 'ready' });
    await ready;
    expect(progress).toEqual(['reading the gallery']);
  });

  it("gives the engine worker the card list's rows with its init (the store build), a copy of them, and none for the developer build", async () => {
    const rows = [{ printing_id: 'A-1', card_id: 'a', name: 'A', type: 'Unit' }];
    const e = new WorkerEngine('engine-webgpu.js');
    const w = FakeWorker.last;
    void e.init(pkg, attempt, () => {}, rows);
    expect(w.sent).toEqual([{ kind: 'init', pkg, attempt, cards: rows }]);
    expect((w.sent[0] as { cards: unknown }).cards).not.toBe(rows);
    const none = new WorkerEngine('engine-webgpu.js');
    void none.init(pkg, attempt, () => {}, []); // the list could not be read: an empty one is still said
    expect(FakeWorker.last.sent).toEqual([{ kind: 'init', pkg, attempt, cards: [] }]);
  });

  it('says why a load failed, and does not wait for a load that never ends', async () => {
    const e = new WorkerEngine('engine-webgpu.js');
    const w = FakeWorker.last;
    const failed = e.init(pkg, attempt, () => {});
    w.say({ kind: 'failed', error: 'GridSample is not supported' });
    await expect(failed).rejects.toThrow('GridSample');

    const slow = new WorkerEngine('engine-webgpu.js', 30_000, 60_000);
    const w2 = FakeWorker.last;
    const stuck = slow.init(pkg, attempt, () => {});
    const caught = expect(stuck).rejects.toThrow('did not start in 60 s');
    await vi.advanceTimersByTimeAsync(60_000);
    await caught;
    expect(w2.terminated).toBe(true);
  });

  it('matches each frame to its answer by id, and to the state and the events', async () => {
    const e = new WorkerEngine('engine-webgpu.js');
    const w = FakeWorker.last;
    const a = e.frame({ tab: 1, t: 1, video: 'v', jpeg: 'QQ==' });
    const b = e.frame({ tab: 2, t: 2, video: 'w', jpeg: 'Qg==' });
    expect(w.sent).toEqual([
      { kind: 'frame', id: 1, tab: 1, t: 1, video: 'v', jpeg: 'QQ==' },
      { kind: 'frame', id: 2, tab: 2, t: 2, video: 'w', jpeg: 'Qg==' },
    ]);
    const events = [{ t: 2, kind: 'played', text: 'x played', printing_id: 'A-1', track: 't1', side: 'left' }];
    w.say({ kind: 'state', id: 2, state, events });
    w.say({ kind: 'error', id: 1, error: 'the GPU device was lost' });
    expect(await b).toEqual({ state, events });
    await expect(a).rejects.toThrow('device was lost');
  });

  it('takes a worker that does not answer a frame to have hung: it is stopped, and everything waiting fails', async () => {
    const e = new WorkerEngine('engine-webgpu.js', 30_000);
    const w = FakeWorker.last;
    const crashes: string[] = [];
    e.onCrash((why) => crashes.push(why));
    const a = e.frame({ tab: 1, t: 1, video: 'v', jpeg: '' });
    const caught = expect(a).rejects.toThrow('no answer to a frame in 30 s');
    await vi.advanceTimersByTimeAsync(30_000);
    await caught;
    expect(w.terminated).toBe(true);
    expect(crashes).toEqual(['no answer to a frame in 30 s']);
    e.forget(3); // a dead worker is sent nothing
    expect(w.sent.some((m) => m.kind === 'forget')).toBe(false);
  });

  it('reports a worker that stops on its own once, and fails the frames waiting for it', async () => {
    const e = new WorkerEngine('engine-wasm.js');
    const w = FakeWorker.last;
    const crashes: string[] = [];
    e.onCrash((why) => crashes.push(why));
    const a = e.frame({ tab: 1, t: 1, video: 'v', jpeg: '' });
    w.onerror?.({ message: 'out of memory' });
    w.onerror?.({ message: 'again' });
    await expect(a).rejects.toThrow('out of memory');
    expect(crashes).toEqual(['out of memory']);
    w.onmessageerror?.();
    expect(crashes).toHaveLength(1);
  });

  it('fails a load that the worker dies in, and says a tab is gone', async () => {
    const e = new WorkerEngine('engine-wasm.js');
    const w = FakeWorker.last;
    const crashes: string[] = [];
    e.onCrash((why) => crashes.push(why));
    const loading = e.init(pkg, attempt, () => {});
    w.onerror?.({ message: 'the script did not load' });
    await expect(loading).rejects.toThrow('did not load');
    expect(crashes).toEqual([]); // a load that fails is the loader's to handle, not a crash

    const f = new WorkerEngine('engine-wasm.js');
    f.forget(4);
    expect(FakeWorker.last.sent).toEqual([{ kind: 'forget', tab: 4 }]);
  });

  it('stops the worker when it is let go, and fails what waits', async () => {
    const e = new WorkerEngine('engine-wasm.js');
    const w = FakeWorker.last;
    const a = e.frame({ tab: 1, t: 1, video: 'v', jpeg: '' });
    e.terminate();
    await expect(a).rejects.toThrow('was stopped');
    expect(w.terminated).toBe(true);
  });
});
