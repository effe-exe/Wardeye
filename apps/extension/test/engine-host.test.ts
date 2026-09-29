import { describe, expect, it } from 'vitest';
import { EngineHost, FPS } from '../src/engine-host';
import { Timer } from '../src/timer';
import { FakeParts, LA, jpegOf } from './fakes';

function setup(over: { maxSessions?: number } = {}) {
  let clock = 1000;
  const now = () => clock;
  const timer = new Timer(now);
  const parts = new FakeParts();
  parts.hooks = {
    decode: () => void (clock += 6),
    step: () => {
      timer.add('detect', 20);
      timer.add('embed', 12);
      clock += 50; // the whole step: 18 ms of it neither
    },
  };
  const host = new EngineHost({ parts, timer, attempt: { runtime: 'webgpu', detector: 'fp32', embedder: 'fp16' }, layout: LA, now, ...over });
  return { host, parts, tick: (ms: number) => void (clock += ms) };
}

const req = (tab: number, t: number, video = '/videos/1') => ({ tab, t, video, jpeg: jpegOf(64, 36) });

describe('the engine host', () => {
  it('adds the engine and its timings to the board the recogniser gives', async () => {
    const { host } = setup();
    const { state } = await host.frame(req(1, 10));
    expect(state.status).toBe('live');
    expect(state.engine).toEqual({
      runtime: 'webgpu',
      detector: 'fp32',
      embedder: 'fp16',
      every_ms: 200,
      reads_per_s: 0,
      timing: { decode: 6, detect: 20, embed: 12, track: 18, total: 56 },
      layout: 'la-rq',
    });
    expect(state.fps).toEqual({ source: FPS, processed: 0 });
    expect(state.latency_s).toBe(0.06);
  });

  it("gives each tab's board the decklists its frames carry, and says what it made of them", async () => {
    const { host, parts } = setup();
    expect((await host.frame(req(1, 10))).state.lists).toEqual([]);
    const { state } = await host.frame({ ...req(1, 10.2), lists: ['Gleaming Anvil\na'] });
    expect(state.lists).toEqual([{ legends: ['Gleaming Anvil'], cards: 1, unmapped: [], error: null }]);
    expect(parts.boardLists[0]).toEqual([['Gleaming Anvil\na']]);
    expect((await host.frame(req(2, 10))).state.lists).toEqual([]); // another tab: none
  });

  it('reports the reads a second it manages', async () => {
    const { host, tick } = setup();
    await host.frame(req(1, 10));
    let last = 0;
    for (let i = 1; i <= 5; i++) {
      tick(200 - 56); // a frame every 200 ms
      last = (await host.frame(req(1, 10 + i / 5))).state.engine!.reads_per_s;
    }
    expect(last).toBeCloseTo(5, 1);
  });

  it('gives each tab its own board, and lets the least recently used go', async () => {
    const { host, parts } = setup({ maxSessions: 2 });
    await host.frame(req(1, 10));
    await host.frame(req(2, 10));
    await host.frame(req(1, 10.2)); // tab 1 is the newer now
    await host.frame(req(3, 10)); // tab 2 goes
    expect(host.boards).toBe(2);
    expect(parts.made.length).toBe(3);
    await host.frame(req(1, 10.4));
    expect(parts.made.length).toBe(3); // tab 1 kept its board
    await host.frame(req(2, 10.2));
    expect(parts.made.length).toBe(4); // tab 2 starts a new one
  });

  it("lets a tab's board go when the tab does", async () => {
    const { host, parts } = setup();
    await host.frame(req(1, 10));
    host.forget(1);
    expect(host.boards).toBe(0);
    await host.frame(req(1, 10.2));
    expect(parts.made.length).toBe(2);
  });

  it('says what the layout is while it is being found', async () => {
    const parts = new FakeParts();
    const timer = new Timer(() => 0);
    const host = new EngineHost({ parts, timer, attempt: { runtime: 'wasm', detector: 'fp32', embedder: 'fp32' }, now: () => 0 });
    const { state } = await host.frame(req(1, 0));
    expect(state).toMatchObject({ status: 'starting', engine: { layout: 'auto', runtime: 'wasm', detector: 'fp32', embedder: 'fp32' } });
  });

  it('frees the models when it is let go', async () => {
    const { host, parts } = setup();
    await host.frame(req(1, 10));
    await host.dispose();
    expect(parts.disposed).toBe(true);
    expect(host.boards).toBe(0);
  });
});
