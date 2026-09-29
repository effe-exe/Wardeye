// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The recognizer against its Python reference on the first steps of the LA final, in Node for a fast loop. It needs
// the private data (RIFTEYE_M3, e.g. ~/rifteye-data/m3) and skips without it: the frames as raw RGB
// (fixtures/recognizer/frames-rgb/, written by gen/recognizer_frames.py, since Node has no JPEG decoder), and the
// finder's boxes and the encoder's rows as gen/recognizer_replay.py recorded them. The browser test
// (e2e/recognizer.spec.ts) replays all 240 steps from the JPEGs, decoded as the extension decodes them.
// RIFTEYE_REPLAY_LENIENT=1 goes on past pictures unlike Python's (counting them), to look further along.

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { rgbImage } from '../src/image';
import { parseFixture, picture, replay, summary, type ReplayFixture } from './recognizer-replay';

const M3 = process.env.RIFTEYE_M3;
const DIR = M3 ? join(M3, 'fixtures', 'recognizer') : '';
const RGB = join(DIR, 'frames-rgb');
const HAVE = !!M3 && existsSync(join(DIR, 'meta.json')) && existsSync(RGB);

const bytes = (path: string): ArrayBuffer => {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

function load(dir: string): ReplayFixture {
  const text = (f: string): string => readFileSync(join(dir, f), 'utf8');
  const meta = JSON.parse(text('meta.json')) as { levels: number[] };
  return parseFixture({
    meta: text('meta.json'),
    rows: text('rows.json'),
    steps: text('steps.jsonl'),
    embeds: text('embeds.json'),
    embedsBin: bytes(join(dir, 'embeds.bin')),
    levels: new Map(meta.levels.map((s) => [s, bytes(join(dir, 'levels', `${s}.bin`))])),
    internals: gunzipSync(readFileSync(join(dir, 'internals.jsonl.gz'))).toString('utf8'),
  });
}

const raw = (file: string): string => join(RGB, file.replace(/\.jpg$/, '.rgb.gz'));

/** A frame from frames-rgb/ (1920 x 1080). */
const frame = async (file: string) => rgbImage(1920, 1080, new Uint8Array(gunzipSync(readFileSync(raw(file)))));

describe.skipIf(!HAVE)('the recognizer on the LA final, against Python (RIFTEYE_M3)', () => {
  const lenient = process.env.RIFTEYE_REPLAY_LENIENT === '1';

  it('gives Python\'s state and events on the first steps, from raw frames', async () => {
    const fx = load(DIR);
    const have = new Set(readdirSync(RGB));
    let n = 0;
    while (n < fx.steps.length && have.has(fx.steps[n]!.file.replace(/\.jpg$/, '.rgb.gz'))) n++;
    expect(n).toBeGreaterThan(0);
    const report = await replay(fx, (s) => picture(s, frame), { steps: n, lenient });
    console.log(summary(report, n));
    expect(report.firstDivergence).toBeNull();
    expect(report.identical).toBe(n);
  }, 300_000);

  // recognizer_replay.py --scenario cut: the table, another shot (grey), then the table re-framed (moved 60 px left
  // and 40 px down): the scene goes away and back, the board is cut, found again by name and re-anchored
  it.skipIf(!existsSync(join(DIR, 'cut', 'meta.json')))('goes away, comes back re-framed and re-anchors the board as Python does', async () => {
    const fx = load(join(DIR, 'cut'));
    const report = await replay(fx, (s) => picture(s, frame), { lenient });
    console.log(summary(report, fx.steps.length));
    expect(report.firstDivergence).toBeNull();
    expect(report.identical).toBe(fx.steps.length);
    expect(fx.steps.filter((s) => s.state.status === 'away').length).toBeGreaterThan(0);
  }, 300_000);
});
