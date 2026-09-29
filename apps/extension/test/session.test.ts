import { describe, expect, it } from 'vitest';
import { layouts } from '@rifteye/engine';
import { Session } from '../src/session';
import { FakeParts, LA, picture } from './fakes';

const FPS = 5;
const frame = () => picture(64, 36, [10, 20, 30]);

/** Feeds frames at 5 a second from `from` to `to` (s of video), returning the states' statuses. */
async function play(s: Session, video: string, from: number, to: number): Promise<string[]> {
  const out: string[] = [];
  for (let t = from; t <= to + 1e-9; t += 1 / FPS) out.push((await s.step(Math.round(t * 100) / 100, video, frame())).state.status);
  return out;
}

describe("a tab's layout", () => {
  it('is looked for once a second over the last five looks, and the frames it is looked for in are not read as a board', async () => {
    const parts = new FakeParts();
    const s = new Session({ parts, fps: FPS });
    const statuses = await play(s, '/videos/1', 0, 4.8);
    expect(new Set(statuses)).toEqual(new Set(['starting']));
    expect(parts.finds).toEqual([5]); // the fifth look, at t = 4: five frames
    await play(s, '/videos/1', 5, 6.8); // looks at 5 and 6: asked again each time
    expect(parts.finds).toEqual([5, 5, 5]);
    expect(parts.made).toEqual([]);
    expect(parts.steps).toEqual([]);
    expect(s.layoutName).toBe('auto');
  });

  it('is read from at the next frame once it is found, by a board of its own', async () => {
    const parts = new FakeParts();
    parts.found = LA;
    const s = new Session({ parts, fps: FPS });
    const statuses = await play(s, '/videos/1', 0, 5.2);
    expect(statuses.slice(0, 21)).toEqual(new Array(21).fill('starting')); // up to t = 4.0, the look that found it
    expect(statuses.at(-1)).toBe('live'); // after it
    expect(parts.made).toEqual([LA]);
    expect(parts.steps.map((x) => x.t)).toEqual([4.2, 4.4, 4.6, 4.8, 5, 5.2]);
    expect(s.layoutName).toBe('la-rq');
  });

  it('says what it is doing while it looks', async () => {
    const s = new Session({ parts: new FakeParts(), fps: FPS });
    const { state } = await s.step(0, '/videos/1', frame());
    expect(state).toMatchObject({ status: 'starting', message: 'finding the table and the size of a card', tracks: [], frame: { width: 64, height: 36 } });
  });

  it('is one of the presets when the footage will not give one but a preset\'s mat fills the table window', async () => {
    const parts = new FakeParts();
    const s = new Session({ parts, fps: FPS, presetAfter: 3 });
    const mat = layouts.LAYOUTS.plusrb.mat!;
    const tableFrame = () => picture(384, 216, mat);
    for (let t = 0; t <= 6.01; t += 1) await s.step(t, '/videos/1', tableFrame());
    // the looks at 0..3 gather five frames only at t = 4; the third failure is at t = 6
    expect(parts.finds.length).toBe(3);
    expect(s.layoutName).toBe('plusrb');
    const { state } = await s.step(6.2, '/videos/1', tableFrame());
    expect(state.status).toBe('live');
  });

  it('is not a preset when no preset\'s mat is on screen, however long it looks', async () => {
    const s = new Session({ parts: new FakeParts(), fps: FPS, presetAfter: 1 });
    for (let t = 0; t < 12; t++) await s.step(t, '/videos/1', picture(384, 216, [250, 250, 250]));
    expect(s.layoutName).toBe('auto');
  });

  it('is not looked for at all when the package names one', async () => {
    const parts = new FakeParts();
    const s = new Session({ parts, fps: FPS, layout: layouts.LAYOUTS.shenyang });
    expect((await s.step(0, '/videos/1', frame())).state.status).toBe('live');
    expect(parts.finds).toEqual([]);
    expect(s.layoutName).toBe('shenyang');
  });

  it('starts its looks over after a seek: the earlier looks were another scene', async () => {
    const parts = new FakeParts();
    const s = new Session({ parts, fps: FPS });
    for (const t of [0, 1, 2, 3]) await s.step(t, '/videos/1', frame());
    await s.step(200, '/videos/1', frame()); // a jump far ahead: the fifth look would have come here
    expect(parts.finds).toEqual([]);
    for (const t of [201, 202, 203, 204]) await s.step(t, '/videos/1', frame());
    expect(parts.finds).toEqual([5]);
    await s.step(50, '/videos/1', frame()); // and back
    await s.step(51, '/videos/1', frame());
    expect(parts.finds).toEqual([5]); // two looks so far: not enough
  });
});

describe("a tab's board", () => {
  const live = async (video = '/videos/1') => {
    const parts = new FakeParts();
    const s = new Session({ parts, fps: FPS, layout: LA });
    await s.step(10, video, frame());
    return { parts, s };
  };

  it('goes on across small steps in the video, forward or back a little', async () => {
    const { parts, s } = await live();
    for (const t of [10.2, 10.4, 25.4, 24.9, 24.5]) await s.step(t, '/videos/1', frame()); // 15 s ahead at most; 1 s back at most
    expect(parts.made.length).toBe(1);
  });

  it('starts over on a jump: more than 15 s ahead, or more than a second back', async () => {
    const { parts, s } = await live();
    await s.step(25.5, '/videos/1', frame()); // 15.5 s ahead
    expect(parts.made.length).toBe(2);
    await s.step(24.4, '/videos/1', frame()); // 1.1 s back
    expect(parts.made.length).toBe(3);
    await s.step(24.6, '/videos/1', frame());
    expect(parts.made.length).toBe(3);
    expect(parts.steps.map((x) => x.board)).toEqual([0, 1, 2, 2]);
  });

  it('starts over on another video, which finds its own layout', async () => {
    const parts = new FakeParts();
    parts.found = LA;
    const s = new Session({ parts, fps: FPS });
    await play(s, '/videos/1', 0, 5.2);
    expect(s.layoutName).toBe('la-rq');
    parts.found = null;
    const { state } = await s.step(5.4, '/videos/2', frame()); // the same time, another video
    expect(state.status).toBe('starting');
    expect(s.layoutName).toBe('auto');
    expect(parts.made.length).toBe(1);
    await play(s, '/videos/2', 5.6, 10.6);
    expect(parts.finds.length).toBe(3); // it looked again: at 9.4, the fifth look of the new video, and at 10.4
  });

  it('starts over on another video with the same package layout', async () => {
    const { parts, s } = await live('/videos/1');
    await s.step(10.2, '/videos/2', frame());
    expect(parts.made.length).toBe(2);
    expect(parts.finds).toEqual([]);
  });

  it('lets what the board throws through, and goes on with the same board', async () => {
    const { parts, s } = await live();
    parts.fail = new Error('the GPU device was lost');
    await expect(s.step(10.2, '/videos/1', frame())).rejects.toThrow('device was lost');
    parts.fail = null;
    expect((await s.step(10.4, '/videos/1', frame())).state.status).toBe('live');
    expect(parts.made.length).toBe(1);
  });
});
