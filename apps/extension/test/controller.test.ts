import { describe, expect, it } from 'vitest';
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
  constructor(readonly attempt: Attempt) {}
  async init(_pkg: StandalonePackage, _attempt: Attempt, progress: (m: string) => void): Promise<void> {
    progress('reading the gallery');
    if (this.holdInit) await this.holdInit;
    if (this.initError) throw this.initError;
  }
  async frame(req: FrameReq): Promise<FrameOut> {
    this.frames.push(req);
    if (this.hold) await this.hold;
    if (this.failFrames) throw this.failFrames;
    return { state: { t: req.t, status: 'live', message: '', frame: { width: 64, height: 36 }, tracks: [] }, events: [] };
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

function setup(over: { pkg?: StandalonePackage | null; caps?: Partial<Capabilities>; setup?: (w: FakeWorker) => void } = {}) {
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
