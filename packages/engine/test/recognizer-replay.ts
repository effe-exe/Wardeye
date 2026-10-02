// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The recognizer replayed against its Python reference (fixtures/recognizer/, written by
// test/gen/recognizer_replay.py; the format is in the README.md written next to it). The finder gives the boxes
// Python's finder gave; the encoder checks that each picture it is handed is the one Python's encoder was handed
// (its size and the SHA-256 of its RGB bytes), then gives the rows Python's gave; and each step's state and events
// are compared with Python's, numbers to 1e-6. Shared by the Node test (recognizer.replay.test.ts) and the browser
// page (e2e/recognizer-page.ts), so both replay the same way.

import * as image from '../src/image';
import { LAYOUTS, type LayoutName } from '../src/layouts';
import { Recognizer, type RecognizerEvent, type RecognizerState, type Track } from '../src/recognizer';
import { Pyramid } from '../src/retrieval';
import type { CardBox, CatalogRow, Encoder, Finder, Layout, RgbImage } from '../src/types';

export interface ReplayMeta {
  steps: number;
  layout: Layout;
  fps: number;
  det_score: number;
  temperature: number;
  title: string;
  /** Whether the legend rule was on; fixtures written before it had none, and ran without it. */
  legend_rule?: boolean;
  levels: number[];
  rows: number;
  dim: number;
  encoder: string;
  embed_calls: number;
  embed_rows: number;
}

/** Where a step's picture comes from: a frame, perhaps moved by (dx, dy) px (black where it left), or plain grey. */
export interface FrameSource {
  file?: string;
  shift?: [number, number] | null;
  grey?: number;
}

/** One line of steps.jsonl. */
export interface ReplayStep {
  i: number;
  /** The picture's name: the frame file, `<file>@dx,dy` when moved, or `grey`. */
  file: string;
  /** What the picture is (the LA final's own steps have none: the frame `file` as it is). */
  source?: FrameSource;
  t: number;
  /** The finder's boxes, or null when it was not asked (the scene said the video was not the table). */
  boxes: CardBox[] | null;
  /** The tracks read this step, in order. */
  todo: string[];
  /** The embed calls made during this step (indices into EmbedCall[]). */
  embeds: number[];
  state: RecognizerState;
  events: RecognizerEvent[];
}

/** One embed() call of embeds.json: its pictures and where its rows are in embeds.bin. */
export interface EmbedCall {
  step: number;
  call: number;
  /** read (the tracks' crops, four turns each) or watch (a region the change gate saw settle). */
  where: string;
  /** For a read, the track of each crop. */
  tracks: (string | null)[] | null;
  sizes: [number, number][];
  sha256: string[];
  /** The first row in embeds.bin, and how many. */
  offset: number;
  count: number;
}

export interface ReplayFixture {
  meta: ReplayMeta;
  rows: CatalogRow[];
  levels: Map<number, Float32Array>;
  steps: ReplayStep[];
  calls: EmbedCall[];
  /** embeds.bin: every call's rows, one after another, float32. */
  embedRows: Float32Array;
  /** internals.jsonl.gz, a line per step: the Recognizer's bookkeeping after it (for finding where a port first
   * goes its own way). */
  internals?: Snapshot[];
}

/** The files as text and bytes, as either environment reads them. */
export interface FixtureFiles {
  meta: string;
  rows: string;
  steps: string;
  embeds: string;
  embedsBin: ArrayBuffer;
  levels: Map<number, ArrayBuffer>;
  internals?: string;
}

const lines = (text: string): string[] => text.split('\n').filter((l) => l.trim() !== '');

export function parseFixture(files: FixtureFiles): ReplayFixture {
  const meta = JSON.parse(files.meta) as ReplayMeta;
  const levels = new Map<number, Float32Array>();
  for (const [s, buf] of files.levels) levels.set(s, new Float32Array(buf));
  const fx: ReplayFixture = {
    meta,
    rows: JSON.parse(files.rows) as CatalogRow[],
    levels,
    steps: lines(files.steps).map((l) => JSON.parse(l) as ReplayStep),
    calls: (JSON.parse(files.embeds) as { calls: EmbedCall[] }).calls,
    embedRows: new Float32Array(files.embedsBin),
  };
  if (files.internals !== undefined) fx.internals = lines(files.internals).map((l) => JSON.parse(l) as Snapshot);
  return fx;
}

// --- comparing ----------------------------------------------------------------------------------------------------

/** The first place `got` and `want` differ, as a path and the two values, or null when they agree: numbers within
 * `tol`, anything else exactly. A list of tracks that differs in length says which ids are missing or extra. */
export function firstDifference(got: unknown, want: unknown, path: string, tol = 1e-6): string | null {
  if (typeof want === 'number' && typeof got === 'number') {
    if (Math.abs(got - want) <= tol || (Number.isNaN(got) && Number.isNaN(want))) return null;
    return `${path}: ${got} here, ${want} in Python`;
  }
  if (want === null || typeof want !== 'object' || got === null || typeof got !== 'object') {
    return got === want ? null : `${path}: ${JSON.stringify(got)} here, ${JSON.stringify(want)} in Python`;
  }
  if (Array.isArray(want) || Array.isArray(got)) {
    if (!Array.isArray(want) || !Array.isArray(got)) return `${path}: ${JSON.stringify(got)} here, ${JSON.stringify(want)} in Python`;
    if (got.length !== want.length) {
      const ids = (xs: unknown[]): string[] => xs.map((x) => (x && typeof x === 'object' && 'id' in x ? String((x as { id: unknown }).id) : '?'));
      const g = ids(got);
      const w = ids(want);
      const extra = g.filter((id) => !w.includes(id));
      const missing = w.filter((id) => !g.includes(id));
      const which = g.includes('?') ? '' : ` (extra here: ${extra.join(', ') || 'none'}; missing here: ${missing.join(', ') || 'none'})`;
      return `${path}: ${got.length} items here, ${want.length} in Python${which}`;
    }
    for (let i = 0; i < want.length; i++) {
      const d = firstDifference(got[i], want[i], `${path}[${i}]`, tol);
      if (d) return d;
    }
    return null;
  }
  const g = got as Record<string, unknown>;
  const w = want as Record<string, unknown>;
  for (const k of Object.keys(w)) {
    if (!(k in g)) return `${path}.${k}: missing here (${JSON.stringify(w[k])} in Python)`;
    const d = firstDifference(g[k], w[k], `${path}.${k}`, tol);
    if (d) return d;
  }
  for (const k of Object.keys(g)) if (!(k in w) && g[k] !== undefined) return `${path}.${k}: ${JSON.stringify(g[k])} here, not in Python`;
  return null;
}

// --- the Recognizer's bookkeeping, as the Python reference writes it after each step ---------------------------

export type Snapshot = Record<string, unknown>;

function boxJson(b: CardBox): Record<string, unknown> {
  const d: Record<string, unknown> = { centre: [b.centre[0], b.centre[1]], long_px: b.long_px, short_px: b.short_px, angle_deg: b.angle_deg, fill: b.fill };
  if (b.back !== undefined) d.back = b.back;
  if (b.score !== undefined) d.score = b.score;
  if (b.vis !== undefined) d.vis = b.vis;
  return d;
}

function trackJson(tr: Track): Record<string, unknown> {
  const prob = [...tr.prob];
  let sum = 0; // Python 3.11's sum(): one after another
  for (const [, p] of prob) sum += p;
  const best = [...prob].sort((a, b) => b[1] - a[1]).slice(0, 5);
  return {
    id: tr.id,
    box: boxJson(tr.box),
    first: tr.first,
    last: tr.last,
    hits: tr.hits,
    reads: tr.reads,
    down: tr.down,
    prob_n: prob.length,
    prob_sum: sum,
    prob_top: best.map(([c, p]) => [c, p]),
    best_row: best.filter(([c]) => tr.bestRow.has(c)).map(([c]) => [c, tr.bestRow.get(c)![0], tr.bestRow.get(c)![1]]),
    last_read: tr.lastRead,
    named: tr.named,
    side: tr.side,
    kind: tr.kind,
    pinned: tr.pinned,
    free_since: tr.freeSince,
    free_at: tr.freeAt,
    placed: tr.placed,
  };
}

/** What recognizer_replay.py's snapshot() writes, from this Recognizer. */
export function snapshot(rec: Recognizer): Snapshot {
  const sc = rec.scene;
  return {
    tracks: [...rec.tracks.values()].map(trackJson),
    next_id: rec.nextId,
    t0: rec.t0,
    last_t: rec.lastT,
    away: rec.away,
    cut_at: rec.cutAt,
    pending: rec.pending.map(([w, b]) => [w, [...b]]),
    plays: rec.plays.map((p) => [...p]),
    ghosts: rec.ghosts,
    legends: Object.fromEntries(rec.legends),
    flashes: rec.flashes,
    prev_seen: [...rec.prevSeen].sort(),
    before_away: [...rec.beforeAway].sort(),
    anchor_base: [...rec.anchorBase.keys()].sort(),
    anchor_pairs: rec.anchorPairs.map(([a, b]) => [[...a], [...b]]),
    boxes_now: Object.fromEntries([...rec.boxesNow].map(([k, v]) => [k, [...v]])),
    scene: { n: sc.n, last_learn: sc.lastLearn, away_since: sc.awaySince, looks: sc.looks, last_look: sc.lastLook },
  };
}

// --- the stand-ins ----------------------------------------------------------------------------------------------

/** A replay that cannot go on: the port asked for something Python's run did not (or not the same pictures). */
export class ReplayError extends Error {}

const hex = (buf: ArrayBuffer): string => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

/** SHA-256 of a picture's RGB bytes, as hashlib gives it for np.asarray(im).tobytes(). */
export async function rgbSha256(im: RgbImage): Promise<string> {
  const bytes = im.data.byteOffset === 0 && im.data.byteLength === im.data.buffer.byteLength ? im.data : im.data.slice();
  return hex(await globalThis.crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
}

/** The Recognizer with what it is reading written down, for the encoder's messages. */
class Traced extends Recognizer {
  where = '';
  reading: string[] = [];

  override async read(t: number, frame: RgbImage, todo: readonly Track[]): Promise<void> {
    this.where = 'read';
    this.reading = todo.filter((tr) => !tr.box.back).map((tr) => tr.id);
    try {
      await super.read(t, frame, todo);
    } finally {
      this.where = '';
    }
  }

  override async watch(t: number, frame: RgbImage): Promise<RecognizerEvent[]> {
    this.where = 'watch';
    try {
      return await super.watch(t, frame);
    } finally {
      this.where = '';
    }
  }
}

/** The encoder: Python's rows, once the pictures are checked against the recording. */
class RecordedEncoder implements Encoder {
  readonly name: string;
  readonly dim: number;
  /** The next recorded call. */
  next = 0;
  step = -1;
  /** Time spent in here (hashing), ms: not the recogniser's own. */
  ms = 0;
  mismatches = 0;
  firstMismatch: string | null = null;
  rec: Traced | null = null;

  constructor(
    private readonly fx: ReplayFixture,
    private readonly lenient: boolean,
  ) {
    this.name = fx.meta.encoder;
    this.dim = fx.meta.dim;
  }

  private context(call: EmbedCall | undefined): string {
    const s = this.fx.steps[this.step];
    const here = this.rec ? `${this.rec.where}${this.rec.where === 'read' ? ` of ${this.rec.reading.join(', ')}` : ''}` : '?';
    const there = call ? `${call.where}${call.tracks ? ` of ${call.tracks.join(', ')}` : ''}` : 'no call';
    return `step ${this.step} (${s?.file}, t=${s?.t}), embed call ${call?.call ?? this.next}: here ${here}, in Python ${there}`;
  }

  async embed(images: readonly RgbImage[]): Promise<Float32Array> {
    const tic = performance.now();
    const call = this.fx.calls[this.next];
    if (call === undefined || call.step !== this.step) throw new ReplayError(`${this.context(call)}: Python made no such call`);
    const sizes = images.map((im) => `${im.width}x${im.height}`);
    const want = call.sizes.map(([w, h]) => `${w}x${h}`);
    if (images.length !== call.count) {
      throw new ReplayError(`${this.context(call)}: ${images.length} pictures here (${sizes.join(' ')}), ${call.count} in Python (${want.join(' ')})`);
    }
    for (let k = 0; k < images.length; k++) {
      const same = sizes[k] === want[k] && (await rgbSha256(images[k]!)) === call.sha256[k];
      if (same) continue;
      const crop = call.where === 'read' ? ` (track ${call.tracks?.[k >> 2] ?? '?'}, turn ${(k % 4) * 90})` : '';
      const msg = `${this.context(call)}: picture ${k}${crop} is ${sizes[k]} here, ${want[k]} in Python${sizes[k] === want[k] ? ', with other pixels' : ''}`;
      if (!this.lenient) throw new ReplayError(msg);
      this.mismatches += 1;
      this.firstMismatch ??= msg;
    }
    this.next += 1;
    this.ms += performance.now() - tic;
    return this.fx.embedRows.slice(call.offset * this.dim, (call.offset + call.count) * this.dim);
  }
}

// --- the replay -----------------------------------------------------------------------------------------------

/** A step's picture: the frame `load` gives, moved as Pillow's crop moves it (recognizer_replay.py's picture()), or
 * plain grey. */
export async function picture(s: ReplayStep, load: (file: string) => Promise<RgbImage>): Promise<RgbImage> {
  const src: FrameSource = s.source ?? { file: s.file };
  const { width, height } = s.state.frame;
  if (src.grey !== undefined) return image.rgbImage(width, height, new Uint8Array(width * height * 3).fill(src.grey));
  const im = await load(src.file!);
  if (!src.shift) return im;
  const [dx, dy] = src.shift;
  return image.crop(im, [-dx, -dy, im.width - dx, im.height - dy]);
}

export interface ReplayOptions {
  /** Go on when a picture differs from Python's (the rows are Python's either way): count them instead. */
  lenient?: boolean;
  /** Only the first n steps. */
  steps?: number;
  /** How far a number of the state or an event may be from Python's (1e-6; 0 asks for the same bits). */
  tolerance?: number;
  /** After each step. */
  onStep?: (k: number, state: RecognizerState, events: RecognizerEvent[], ms: number) => void;
}

export interface ReplayReport {
  /** Steps run. */
  steps: number;
  /** Steps whose state and events were Python's. */
  identical: number;
  /** Steps whose events were Python's. */
  eventsEqual: number;
  /** The first step that differs, and where. */
  firstDivergence: string | null;
  /** The first step after which the Recognizer's own bookkeeping differs from Python's (it can come before a
   * difference shows in the state, which rounds). */
  firstInternal: string | null;
  /** Why the replay stopped early, if it did. */
  stopped: string | null;
  /** Pictures handed to the encoder that were not Python's (lenient runs). */
  pictureMismatches: number;
  firstPictureMismatch: string | null;
  /** The recogniser's own time per step, without the finder and the encoder stand-ins, ms. */
  ms: number[];
}

/** Replays the fixture's steps; `frame(step)` gives each step's picture. */
export async function replay(fx: ReplayFixture, frame: (s: ReplayStep) => Promise<RgbImage>, opts: ReplayOptions = {}): Promise<ReplayReport> {
  const layout = LAYOUTS[fx.meta.layout.name as LayoutName];
  if (layout === undefined) throw new Error(`no layout ${fx.meta.layout.name}`);
  const drift = firstDifference({ ...layout }, fx.meta.layout, 'layout', 0);
  if (drift) throw new Error(`the layout preset is not the one Python ran with: ${drift}`);
  const gallery = new Pyramid(fx.levels, fx.meta.dim);
  const enc = new RecordedEncoder(fx, opts.lenient ?? false);
  let k = 0;
  let asked = false;
  let finderMs = 0;
  const finder: Finder = async () => {
    const s = fx.steps[k]!;
    if (s.boxes === null) throw new ReplayError(`step ${k} (${s.file}, t=${s.t}): the finder was asked for boxes; Python's was not (its scene said the video was not the table)`);
    if (asked) throw new ReplayError(`step ${k}: the finder was asked twice`);
    asked = true;
    const tic = performance.now();
    const boxes = s.boxes.map((b): CardBox => ({ ...b, centre: [b.centre[0], b.centre[1]] }));
    finderMs += performance.now() - tic;
    return boxes;
  };
  const rec = new Traced(layout, fx.rows, enc, gallery, {
    title: fx.meta.title,
    fps: fx.meta.fps,
    finder,
    temperature: fx.meta.temperature,
    legendRule: fx.meta.legend_rule ?? false,
  });
  enc.rec = rec;
  const report: ReplayReport = {
    steps: 0,
    identical: 0,
    eventsEqual: 0,
    firstDivergence: null,
    firstInternal: null,
    stopped: null,
    pictureMismatches: 0,
    firstPictureMismatch: null,
    ms: [],
  };
  const n = Math.min(opts.steps ?? fx.steps.length, fx.steps.length);
  for (k = 0; k < n; k++) {
    const s = fx.steps[k]!;
    const im = await frame(s);
    enc.step = k;
    enc.ms = 0;
    asked = false;
    finderMs = 0;
    let state: RecognizerState;
    let events: RecognizerEvent[];
    const tic = performance.now();
    try {
      [state, events] = await rec.step(s.t, im);
    } catch (e) {
      if (!(e instanceof ReplayError)) throw e;
      report.stopped = e.message;
      report.firstDivergence ??= e.message;
      break;
    }
    const ms = performance.now() - tic - enc.ms - finderMs;
    report.ms.push(ms);
    report.steps += 1;
    const last = s.embeds.length ? s.embeds[s.embeds.length - 1]! + 1 : null;
    if (last !== null && enc.next !== last) {
      report.stopped = `step ${k} (${s.file}, t=${s.t}): Python made embed calls ${s.embeds.join(', ')}, this made ${enc.next - s.embeds[0]!} of them`;
      report.firstDivergence ??= report.stopped;
      break;
    }
    if (!asked && s.boxes !== null) {
      report.stopped = `step ${k} (${s.file}, t=${s.t}): Python's finder was asked for boxes, this one was not (the scene said away)`;
      report.firstDivergence ??= report.stopped;
      break;
    }
    const tol = opts.tolerance ?? 1e-6;
    const dEvents = firstDifference(events, s.events, 'events', tol);
    const d = firstDifference(state, s.state, 'state', tol) ?? dEvents;
    if (!dEvents) report.eventsEqual += 1;
    if (d === null) report.identical += 1;
    else report.firstDivergence ??= `step ${k} (${s.file}, t=${s.t}): ${d}`;
    const want = fx.internals?.[k];
    if (want !== undefined && report.firstInternal === null) {
      const di = firstDifference(snapshot(rec), want, 'recognizer', 1e-9);
      if (di) report.firstInternal = `after step ${k} (${s.file}, t=${s.t}): ${di}`;
    }
    opts.onStep?.(k, state, events, ms);
  }
  report.pictureMismatches = enc.mismatches;
  report.firstPictureMismatch = enc.firstMismatch;
  return report;
}

/** The report in a few lines. */
export function summary(r: ReplayReport, total: number): string {
  const sorted = [...r.ms].sort((a, b) => a - b);
  const at = (q: number): number => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? NaN;
  const mean = r.ms.reduce((a, b) => a + b, 0) / Math.max(1, r.ms.length);
  return [
    `steps identical: ${r.identical} of ${total} (run ${r.steps}); events equal on ${r.eventsEqual}`,
    `first divergence: ${r.firstDivergence ?? 'none'}`,
    `first internal divergence: ${r.firstInternal ?? 'none'}`,
    ...(r.stopped ? [`stopped: ${r.stopped}`] : []),
    ...(r.pictureMismatches ? [`pictures unlike Python's: ${r.pictureMismatches}; first: ${r.firstPictureMismatch}`] : []),
    `ms a step (no models): mean ${mean.toFixed(2)}, median ${at(0.5).toFixed(2)}, p90 ${at(0.9).toFixed(2)}, max ${Math.max(...r.ms, 0).toFixed(2)}`,
  ].join('\n');
}
