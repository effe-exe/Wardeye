import type { CatalogRow } from '@rifteye/engine';
import { describe, expect, it, vi } from 'vitest';
import type { StandalonePackage } from '../src/assets';
import { Controller, type ControllerEnv, type EngineWorker, type FrameOut, type FrameReq } from '../src/controller';
import type { Attempt, Capabilities } from '../src/mode';

const pkg: StandalonePackage = {
  format: 1,
  runtime: 'auto',
  detector: { id: 'd', fp16: 'd16', fp32: 'd32' },
  embedder: { id: 'e', fp16: 'e16', fp32: 'e32' },
  data: 'data/',
};

class FakeWorker implements EngineWorker {
  initError: Error | null = null;
  holdInit: Promise<void> | null = null;
  frames: FrameReq[] = [];
  forgotten: number[] = [];
  terminated = false;
  crash: ((why: string) => void) | null = null;
  failFrames: Error | null = null;
  hold: Promise<void> | null = null;
  /** The rows the engine was given with its init (the store build); undefined when it was given none. */
  cards: readonly CatalogRow[] | undefined;
  /** Its states say how it runs, as the engine host's do. */
  engine = false;
  constructor(readonly attempt: Attempt) {}
  async init(_pkg: StandalonePackage, _attempt: Attempt, progress: (m: string) => void, cards?: readonly CatalogRow[]): Promise<void> {
    this.cards = cards;
    progress('reading the gallery');
    if (this.holdInit) await this.holdInit;
    if (this.initError) throw this.initError;
  }
  async frame(req: FrameReq): Promise<FrameOut> {
    this.frames.push(req);
    if (this.hold) await this.hold;
    if (this.failFrames) throw this.failFrames;
    const state = { t: req.t, status: 'live', message: '', frame: { width: 64, height: 36 }, tracks: [] };
    if (!this.engine) return { state, events: [] };
    const timing = { decode: 1, detect: 1, embed: 1, track: 1, total: 4 };
    return { state: { ...state, engine: { ...this.attempt, every_ms: 200, reads_per_s: 1, timing, layout: 'auto' } }, events: [] };
  }
  forget(tab: number): void {
    this.forgotten.push(tab);
  }
  onCrash(cb: (why: string) => void): void {
    this.crash = cb;
  }
  terminate(): void {
    this.terminated = true;
  }
}

function setup(
  over: { pkg?: StandalonePackage | null; caps?: Partial<Capabilities>; setup?: (w: FakeWorker) => void; store?: boolean; cards?: () => Promise<readonly CatalogRow[]> } = {},
) {
  let clock = 0;
  const workers: FakeWorker[] = [];
  const env: ControllerEnv = {
    now: () => clock,
    readPackage: async () => (over.pkg === undefined ? pkg : over.pkg),
    probe: async () => ({ webgpu: true, shaderF16: true, jspi: true, ...over.caps }),
    spawn: (attempt) => {
      const w = new FakeWorker(attempt);
      over.setup?.(w);
      workers.push(w);
      return w;
    },
    ...(over.store === undefined ? {} : { store: over.store }),
    ...(over.cards ? { cards: over.cards } : {}),
  };
  const c = new Controller(env, 60_000);
  return { c, workers, tick: (ms: number) => void (clock += ms) };
}

const req = (tab = 1, t = 1): FrameReq => ({ tab, t, video: '/videos/1', jpeg: 'AAAA' });
const settle = () => new Promise((r) => setTimeout(r, 0));

describe('the engine document', () => {
  it('says at once, before any model is loaded, whether the engine can run here', async () => {
    expect(await setup().c.hello()).toEqual({ kind: 'hello' });
    expect(await setup({ pkg: null }).c.hello()).toEqual({ kind: 'unavailable', reason: 'this build has no models' });
    const noGpu = await setup({ caps: { webgpu: false, shaderF16: false } }).c.hello();
    expect(noGpu).toMatchObject({ kind: 'unavailable', reason: expect.stringContaining('no WebGPU adapter') });
  });

  it('answers a frame that comes while the models load with what it is doing, and then with the board', async () => {
    let release!: () => void;
    const { c, workers } = setup({ setup: (w) => (w.holdInit = new Promise<void>((r) => (release = r))) });
    const first = await c.frame(req());
    expect(first).toMatchObject({ kind: 'state', state: { status: 'starting', message: 'starting the engine', retry_ms: 1000 } });
    await settle();
    const second = await c.frame(req());
    expect(second).toMatchObject({ kind: 'state', state: { status: 'starting', message: 'reading the gallery' } });
    release();
    await settle();
    expect(await c.frame(req(1, 2))).toMatchObject({ kind: 'state', state: { status: 'live', t: 2 } });
    expect(workers.map((w) => w.attempt)).toEqual([{ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' }]);
    expect(c.running).toEqual({ runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' });
  });

  it('starts the next way when one will not start, and says why when none does', async () => {
    const { c, workers } = setup({ setup: (w) => { if (w.attempt.runtime === 'webgpu') w.initError = new Error('GridSample is not supported'); } });
    await c.frame(req());
    await settle();
    expect(workers.map((w) => `${w.attempt.runtime} ${w.attempt.detector}/${w.attempt.embedder}`)).toEqual(['webgpu fp32/fp16', 'wasm fp32/fp32']);
    expect(workers[0]!.terminated).toBe(true);
    expect(c.running).toEqual({ runtime: 'wasm', detector: 'fp32', embedder: 'fp32' });

    const bad = setup({ setup: (w) => (w.initError = new Error(`no ${w.attempt.runtime}`)) });
    await bad.c.frame(req());
    await settle();
    expect(await bad.c.frame(req())).toEqual({ kind: 'unavailable', reason: 'webgpu: detector fp32, embedder fp16: no webgpu; wasm: detector fp32, embedder fp32: no wasm' });
  });

  it('says with the board why it runs on WASM: what the plan left out, or the ways that would not start', async () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => {});
    const noteOf = async (c: Controller): Promise<unknown> => {
      await c.frame(req());
      await settle();
      const r = await c.frame(req());
      return r.kind === 'state' ? (r.state?.engine as { note?: string } | undefined)?.note : r.kind;
    };
    const failed = setup({ setup: (w) => { w.engine = true; if (w.attempt.runtime === 'webgpu') w.initError = new Error('GridSample is not supported'); } });
    expect(await noteOf(failed.c)).toBe('webgpu: detector fp32, embedder fp16: GridSample is not supported');
    const left = setup({ caps: { jspi: false, gpu: 'nvidia ampere' }, setup: (w) => (w.engine = true) });
    expect(await noteOf(left.c)).toBe('this browser has no WebAssembly JSPI (the native WebGPU runtime needs it)');
    const listed: CatalogRow[] = [{ printing_id: 'A-1', card_id: 'a', name: 'Card A', type: 'Unit' }];
    const none = setup({ store: true, cards: async () => listed, caps: { webgpu: false, shaderF16: false, gpu: 'navigator.gpu gave no adapter' }, setup: (w) => (w.engine = true) });
    expect(await noteOf(none.c)).toBe('this browser has no WebGPU adapter (navigator.gpu gave no adapter)');
    expect(info).toHaveBeenCalledWith('Wardeye: the engine runs on WASM, not WebGPU: this browser has no WebGPU adapter (navigator.gpu gave no adapter)');
    const gpu = setup({ setup: (w) => (w.engine = true) });
    expect(await noteOf(gpu.c)).toBeUndefined(); // on WebGPU, nothing to say
    info.mockRestore();
  });

  it('is written off for a while after it could not start, and tried again after that', async () => {
    const { c, workers, tick } = setup({ setup: (w) => (w.initError = new Error('nope')) });
    await c.frame(req());
    await settle();
    expect((await c.frame(req())).kind).toBe('unavailable');
    const made = workers.length;
    tick(59_000);
    expect((await c.frame(req())).kind).toBe('unavailable');
    expect(workers.length).toBe(made); // not tried again yet
    tick(2_000);
    await c.frame(req()); // the time is up: it tries again
    await settle();
    expect(workers.length).toBeGreaterThan(made);
  });

  it('drops the frame of a tab that a newer one replaced, and answers null to keep its board', async () => {
    let release!: () => void;
    const { c, workers } = setup();
    await c.frame(req());
    await settle();
    workers[0]!.hold = new Promise<void>((r) => (release = r));
    const a = c.frame(req(1, 2));
    const b = c.frame(req(2, 2)); // waits for a
    const b2 = c.frame(req(2, 3)); // replaces b
    release();
    expect(await a).toMatchObject({ state: { t: 2 } });
    expect(await b).toEqual({ kind: 'state', state: null });
    expect(await b2).toMatchObject({ state: { t: 3 } });
    expect(workers[0]!.frames.map((f) => `${f.tab}:${f.t}`)).toEqual(['1:2', '2:3']); // the first frame came while it loaded
  });

  it('starts the engine again after a crash, with the boards gone, unless it keeps happening', async () => {
    const { c, workers, tick } = setup();
    await c.frame(req());
    await settle();
    workers[0]!.crash!('the worker stopped');
    expect(workers[0]!.terminated).toBe(true);
    expect(c.state).toBe('idle');
    await c.frame(req()); // the next frame starts it afresh
    await settle();
    expect(workers.length).toBe(2);
    workers[1]!.crash!('again');
    tick(1000);
    await c.frame(req());
    await settle();
    workers[2]!.crash!('and again'); // the third in two minutes
    expect(c.state).toBe('failed');
    expect(await c.frame(req())).toMatchObject({ kind: 'unavailable', reason: expect.stringContaining('keeps stopping') });
  });

  it('starts the engine again when three frames in a row fail (a GPU device that was lost does not come back)', async () => {
    const { c, workers } = setup();
    await c.frame(req());
    await settle();
    workers[0]!.failFrames = new Error('device lost');
    const a = await c.frame(req(1, 2));
    expect(a).toMatchObject({ kind: 'state', state: { status: 'error', message: 'device lost' } });
    await c.frame(req(1, 3));
    expect(c.state).toBe('ready');
    await c.frame(req(1, 4));
    expect(c.state).toBe('idle');
    expect(workers[0]!.terminated).toBe(true);
  });

  it('forgets a tab in the worker, and lets the worker go when it closes', async () => {
    const { c, workers } = setup();
    await c.frame(req());
    await settle();
    c.forget(7);
    expect(workers[0]!.forgotten).toEqual([7]);
    c.dispose();
    expect(workers[0]!.terminated).toBe(true);
    expect(c.state).toBe('idle');
  });

  it('notes when a frame last came, for the document to close itself when none does', async () => {
    const { c, tick } = setup();
    tick(5000);
    await c.frame(req());
    expect(c.lastFrame).toBe(5000);
  });
});

describe("the engine document in the store build, which names the cards from Riot's card list", () => {
  const rows: CatalogRow[] = [{ printing_id: 'A-1', card_id: 'a', name: 'Card A', type: 'Unit' }];

  it('gives the engine the list with its init, and says what it waits for meanwhile', async () => {
    let release!: (r: readonly CatalogRow[]) => void;
    const { c, workers } = setup({ store: true, cards: () => new Promise((r) => (release = r)) });
    expect(await c.frame(req())).toMatchObject({ state: { status: 'starting', message: 'starting the engine' } });
    await settle();
    expect(await c.frame(req())).toMatchObject({ state: { status: 'starting', message: 'loading the card list' } });
    expect(workers).toHaveLength(0); // no engine until the list is in (or a moment has passed)
    release(rows);
    await settle();
    expect(workers).toHaveLength(1);
    expect(workers[0]!.cards).toEqual(rows);
    expect(await c.frame(req(1, 2))).toMatchObject({ state: { status: 'live', t: 2 } });
  });

  it('leaves an engine that was given the list alone when the list arrives, and one that has no need of it', async () => {
    const fed = setup({ store: true, cards: async () => rows });
    await fed.c.frame(req());
    await settle();
    fed.c.cardsArrived();
    expect(fed.c.state).toBe('ready');
    expect(fed.workers[0]!.terminated).toBe(false);
    const dev = setup();
    await dev.c.frame(req());
    await settle();
    expect(dev.workers[0]!.cards).toBeUndefined(); // the developer build reads catalog.json: no rows are given
    dev.c.cardsArrived();
    expect(dev.c.state).toBe('ready');
  });

  it('starts an engine that was given no rows afresh when the list arrives, with the rows, and only then', async () => {
    let list: readonly CatalogRow[] = []; // the first try at the list failed
    const { c, workers } = setup({ store: true, cards: async () => list });
    await c.frame(req());
    await settle();
    expect(workers[0]!.cards).toEqual([]); // every printing named by its id
    expect(c.state).toBe('ready');
    list = rows; // a later try worked
    c.cardsArrived();
    expect(workers[0]!.terminated).toBe(true);
    expect(c.state).toBe('idle');
    expect(c.running).toBeNull();
    await c.frame(req(1, 2)); // the next frame starts it afresh
    await settle();
    expect(workers).toHaveLength(2);
    expect(workers[1]!.cards).toEqual(rows);
    expect(c.state).toBe('ready');
    c.cardsArrived(); // and the engine that has them is not touched again
    expect(c.state).toBe('ready');
    expect(workers[1]!.terminated).toBe(false);
  });

  it('starts the engine afresh once it is up when the list arrived while it was loading without it', async () => {
    let release!: () => void;
    const { c, workers } = setup({ store: true, cards: async () => [], setup: (w) => (w.holdInit = new Promise<void>((r) => (release = r))) });
    await c.frame(req());
    await settle();
    expect(c.state).toBe('loading');
    c.cardsArrived(); // too late for this engine's init
    expect(workers[0]!.terminated).toBe(false);
    release();
    await settle();
    expect(workers[0]!.terminated).toBe(true);
    expect(c.state).toBe('idle');
  });

  it("runs the engine on WASM in a browser with no WebGPU, where the developer build says it is unavailable", async () => {
    const store = setup({ store: true, cards: async () => rows, caps: { webgpu: false, shaderF16: false } });
    await store.c.frame(req());
    await settle();
    expect(store.workers.map((w) => w.attempt)).toEqual([{ runtime: 'wasm', detector: 'fp32', embedder: 'fp32' }]); // float32 files first on WASM
    const dev = setup({ caps: { webgpu: false, shaderF16: false } });
    expect(await dev.c.hello()).toMatchObject({ kind: 'unavailable', reason: expect.stringContaining('no WebGPU adapter') });
  });

  it('is unavailable, with the reason, when the package says to use the live runner (which this build does not have)', async () => {
    const { c } = setup({ store: true, cards: async () => rows, pkg: { ...pkg, runtime: 'companion' } });
    expect(await c.hello()).toMatchObject({ kind: 'unavailable', reason: expect.stringContaining('this build does not have') });
  });
});
