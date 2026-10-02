// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The recognizer's pure parts and its rules against live/pipeline.py, on the synthetic inputs of gen/recognizer.py
// (vectors/recognizer.json): the boxes' corners, smoothing, the similarity fit, numpy's sums, the scene's float32
// arithmetic, and the tracker's rules on random boards of tracks set up by hand. No private data.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LAYOUTS } from '../src/layouts';
import {
  Recognizer,
  Scene,
  Track,
  aabb,
  galleryScales,
  hypot,
  pairwiseSum,
  pairwiseSum32,
  quad,
  similarity,
  smooth,
  type Ghost,
  handShare,
  hiddenRunes,
} from '../src/recognizer';
import * as decklist from '../src/decklist';
import { rgbImage } from '../src/image';
import { Pyramid } from '../src/retrieval';
import type { CardBox, CatalogRow, Encoder, RgbImage } from '../src/types';
import { ROWS } from './priors-rows';
import { firstDifference } from './recognizer-replay';

interface TrackJson {
  id: string;
  box: CardBox;
  first: number;
  last: number;
  hits: number;
  reads: number;
  down: number;
  prob: [string, number][];
  best_row: [string, number, number][];
  last_read: number;
  named: string | null;
  side: string;
  kind: string;
  pinned: boolean;
  free_since: number | null;
  free_at: [number, number] | null;
  placed: boolean;
}

interface Setup {
  tracks: TrackJson[];
  next_id: number;
  t0: number | null;
  legends: Record<string, { printing_id: string; name: string }>;
  ghosts: Ghost[];
  plays: [number, string, number, number][];
  cut_at: number | null;
  anchor_base: Record<string, CardBox>;
  boxes_now: Record<string, [number, number, number, number]>;
  anchor_pairs?: [[number, number], [number, number]][];
}

interface Board {
  t: number;
  setup: Setup;
  per_track: {
    id: string;
    label: [string, number, unknown[]];
    due: boolean;
    covered: boolean;
    stacked_on: string | null;
    twin: boolean;
    vanished: string | null;
    face_down: boolean;
    on_legend: boolean;
    side_legend: string | null;
    misread_battlefield: boolean;
    pinned_twin: string | null;
    held: boolean;
    in_strip: boolean;
  }[];
  stacks: Record<string, string[]>;
  state: unknown;
  recently_played: [string, number, number, boolean][];
  announce: unknown;
  after: unknown;
  state_after: unknown;
  left_box: [number, number, number, number];
  left: unknown;
  after_left: { ids: string[]; ghosts: unknown };
}

const V = JSON.parse(readFileSync(new URL('./vectors/recognizer.json', import.meta.url), 'utf8')) as {
  frame: [number, number];
  rows: CatalogRow[];
  geometry: {
    quads: { box: CardBox; quad: [number, number][]; aabb: [number, number, number, number] }[];
    smooth: { old: CardBox; new: CardBox; out: CardBox }[];
    similarity: { src: [number, number][]; dst: [number, number][]; scale: number; rot: number[][]; shift: number[] }[];
    hypot: [number, number, number][];
    sums64: { n: number; sum: number }[];
    sums32: { n: number; sum: number }[];
  };
  scene: {
    name: string;
    steps: { k: number; flat: boolean; cut: boolean; t: number; score: number | null; learnt: boolean; n: number }[];
    mean_sum: number;
    var_sum: number;
    mean_head: number[];
    var_head: number[];
  }[];
  boards: Board[];
  reanchor: { setup: Setup; after: unknown }[];
  on_table: { t: number; shift: number; ok: boolean; n: number; away_since: number | null; looks: number; last_look: number; last_learn: number }[];
  bootstrap: {
    t: number;
    cards: { centre: [number, number]; long: number; short: number; cos: number; sin: number; rgb: [number, number, number] }[];
    mat: [number, number, number];
    boxes: CardBox[];
  }[];
};

const [W, H] = V.frame;
const LAYOUT = LAYOUTS['la-rq'];

/** A recogniser with the made-up catalogue and nothing to read with: the rules alone. */
function recognizer(): Recognizer {
  const none: Encoder = {
    name: 'none',
    dim: 4,
    embed: async () => {
      throw new Error('nothing is read in these tests');
    },
  };
  return new Recognizer(LAYOUT, V.rows, none, new Pyramid(new Map([[120, new Float32Array(V.rows.length * 4)]]), 4), { fps: 2.0 });
}

function setUp(rec: Recognizer, s: Setup): void {
  for (const d of s.tracks) {
    const tr = new Track(d.id, structuredClone(d.box), d.first, d.last, d.side);
    tr.hits = d.hits;
    tr.reads = d.reads;
    tr.down = d.down;
    tr.prob = new Map(d.prob);
    tr.bestRow = new Map(d.best_row.map(([c, sc, i]) => [c, [sc, i]]));
    tr.lastRead = d.last_read;
    tr.named = d.named;
    tr.kind = d.kind;
    tr.pinned = d.pinned;
    tr.freeSince = d.free_since;
    tr.freeAt = d.free_at;
    tr.placed = d.placed;
    rec.tracks.set(tr.id, tr);
  }
  rec.nextId = s.next_id;
  rec.t0 = s.t0;
  rec.legends = new Map(Object.entries(structuredClone(s.legends)));
  rec.ghosts = structuredClone(s.ghosts);
  rec.plays = structuredClone(s.plays);
  rec.cutAt = s.cut_at;
  rec.anchorBase = new Map(Object.entries(structuredClone(s.anchor_base)));
  rec.boxesNow = new Map(Object.entries(structuredClone(s.boxes_now)));
  if (s.anchor_pairs) rec.anchorPairs = structuredClone(s.anchor_pairs);
}

/** gen/recognizer.py's after_json: what announce and reanchor change. */
function after(rec: Recognizer): unknown {
  return {
    tracks: [...rec.tracks.values()].map((tr) => {
      let sum = 0; // Python 3.11's sum(): one after another
      for (const p of tr.prob.values()) sum += p;
      return {
        id: tr.id,
        box: tr.box,
        first: tr.first,
        last: tr.last,
        hits: tr.hits,
        reads: tr.reads,
        down: tr.down,
        last_read: tr.lastRead,
        named: tr.named,
        side: tr.side,
        kind: tr.kind,
        pinned: tr.pinned,
        free_since: tr.freeSince,
        free_at: tr.freeAt,
        placed: tr.placed,
        prob_n: tr.prob.size,
        prob_sum: sum,
      };
    }),
    legends: Object.fromEntries(rec.legends),
    ghosts: rec.ghosts,
    plays: rec.plays,
    anchor_pairs: rec.anchorPairs,
  };
}

const same = (got: unknown, want: unknown, what: string, tol = 0): void => {
  expect(firstDifference(got, want, what, tol)).toBeNull();
};

describe('boxes', () => {
  it('has the corners and extent Python gives, rounded as numpy rounds its floats', () => {
    for (const q of V.geometry.quads) {
      same(quad(q.box), q.quad, `quad ${JSON.stringify(q.box)}`);
      same(aabb(q.box), q.aabb, `aabb ${JSON.stringify(q.box)}`);
    }
  });

  it('eases a jittering box and follows one that moves or turns', () => {
    for (const s of V.geometry.smooth) same(smooth(s.old, s.new), s.out, `smooth ${JSON.stringify(s)}`);
  });

  it('measures math.hypot bit for bit', () => {
    for (const [x, y, h] of V.geometry.hypot) expect(hypot(x, y)).toBe(h);
  });

  it("fits the scale, turn and shift of Umeyama's SVD without one", () => {
    for (const s of V.geometry.similarity) {
      const [scale, rot, shift] = similarity(s.src, s.dst);
      expect(Math.abs(scale - s.scale)).toBeLessThan(1e-12 * Math.max(1, s.scale));
      same(rot, s.rot, 'rot', 1e-12);
      same(shift, s.shift, 'shift', 1e-9);
    }
  });

  it('asks for the gallery levels live/__main__.py builds', () => {
    expect(galleryScales(LAYOUTS['la-rq'])).toEqual([120, 140, 160]);
    expect(galleryScales(LAYOUTS.shenyang)).toEqual([100, 120, 130]);
  });
});

describe("numpy's sums", () => {
  const series64 = (n: number): number[] => Array.from({ length: n }, (_, i) => (((i * 7919 + n * 104729) % 20011) - 10005) / 3.0);
  const series32 = (n: number): Float32Array => Float32Array.from({ length: n }, (_, i) => (((i * 7919 + n * 104729) % 20011) - 10005) / 64.0);

  it('sums float64 pairwise, as np.sum does', () => {
    for (const { n, sum } of V.geometry.sums64) expect(0 + pairwiseSum(series64(n))).toBe(sum);
  });

  it('sums float32 pairwise in float32', () => {
    for (const { n, sum } of V.geometry.sums32) expect(0 + pairwiseSum32(series32(n))).toBe(sum);
  });
});

describe('the scene', () => {
  /** gen/recognizer.py's thumb(): a still pattern, a changing region and a little jitter; a cut is another pattern. */
  function thumb(k: number, flat: boolean, cut: boolean): Float32Array {
    const x = new Float32Array(54 * 96 * 3);
    let o = 0;
    for (let y = 0; y < 54; y++) {
      for (let xx = 0; xx < 96; xx++) {
        for (let c = 0; c < 3; c++, o++) {
          if (cut) {
            x[o] = (xx * 3 + y * 29 + c * 71 + k * 13) % 256;
          } else {
            const base = flat ? 60 : ((xx * 7 + y * 13 + c * 50) % 200) + 20;
            const noise = ((xx * 31 + y * 17 + c * 5 + k * 11) % 7) - 3;
            const board = xx >= 30 && xx < 60 && y >= 10 && y < 40;
            x[o] = board ? (k * 37 + xx * 3 + y * 5) % 256 : base + noise;
          }
        }
      }
    }
    return x;
  }

  it('learns the table camera and scores frames as numpy does, in float32', () => {
    for (const run of V.scene) {
      const sc = new Scene(LAYOUT);
      for (const s of run.steps) {
        const x = thumb(s.k, s.flat, s.cut);
        const score = sc.n >= 5 ? sc.score(x) : null;
        if (s.score === null) expect(score, `${run.name} ${s.k}`).toBeNull();
        else expect(Math.abs(score! - s.score), `${run.name} ${s.k}: ${score} vs ${s.score}`).toBeLessThan(1e-6);
        const ok = score === null || score >= sc.corr;
        expect(ok, `${run.name} ${s.k}`).toBe(s.learnt);
        if (ok) sc.learn(s.t, x);
        expect(sc.n).toBe(s.n);
      }
      // the running mean and variance are float32 to the last bit
      expect([...sc.mean!.subarray(0, 64)]).toEqual(run.mean_head);
      expect([...sc.var!.subarray(0, 64)]).toEqual(run.var_head);
      expect(0 + pairwiseSum(Float64Array.from(sc.mean!))).toBe(run.mean_sum);
      expect(0 + pairwiseSum(Float64Array.from(sc.var!))).toBe(run.var_sum);
    }
  });
});

describe('the scene on frames', () => {
  /** gen/recognizer.py's scene_frame(): the overlay a print of 40 px blocks (moved `shift` px), the table window la-rq's
   * mat. */
  function sceneFrame(shift: number): RgbImage {
    const im = rgbImage(W, H);
    const [x0, y0, x1, y1] = [365, 65, 1555, 1080]; // LAYOUT's table window at 1920 x 1080
    const mat = LAYOUT.mat!;
    for (let y = 0, o = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
        const bx = Math.floor((x + shift) / 40);
        const by = Math.floor(y / 40);
        for (let c = 0; c < 3; c++, o++) {
          im.data[o] = inside ? mat[c]! + ((x * 3 + y * 7 + c) % 5) : (bx * bx * 37 + by * by * 91 + bx * by * 13 + c * 50) % 256;
        }
      }
    }
    return im;
  }

  it('learns the table camera, loses it when the camera moves, and learns the new view after 20 s', async () => {
    const frames = new Map([0, 40].map((s) => [s, sceneFrame(s)]));
    const sc = new Scene(LAYOUT);
    for (const s of V.on_table) {
      const ok = await sc.onTable(s.t, frames.get(s.shift)!);
      const { t: _t, shift: _shift, ...want } = s;
      same({ ok, n: sc.n, away_since: sc.awaySince, looks: sc.looks, last_look: sc.lastLook, last_learn: sc.lastLearn }, want, `on_table t=${s.t}`);
    }
    expect(V.on_table.some((s) => !s.ok) && V.on_table.at(-1)!.ok).toBe(true);
  }, 60_000);
});

describe("the tracker's rules on boards of tracks", () => {
  it('labels, locks, stacks, lists and announces the cards as Python does', () => {
    V.boards.forEach((b, n) => {
      const rec = recognizer();
      setUp(rec, b.setup);
      const t = b.t;
      for (const p of b.per_track) {
        const tr = rec.tracks.get(p.id)!;
        const where = `board ${n} ${p.id}`;
        same(rec.label(tr), p.label, `${where} label`);
        expect(rec.due(tr, t), `${where} due`).toBe(p.due);
        expect(rec.covered(t, tr), `${where} covered`).toBe(p.covered);
        expect(rec.stackedOn(t, tr)?.id ?? null, `${where} stacked on`).toBe(p.stacked_on);
        expect(rec.twin(t, tr), `${where} twin`).toBe(p.twin);
        expect(tr.named ? (rec.vanished(t, tr)?.id ?? null) : null, `${where} vanished`).toBe(p.vanished);
        expect(rec.faceDown(tr), `${where} face down`).toBe(p.face_down);
        expect(rec.onLegend(tr.box), `${where} on a legend`).toBe(p.on_legend);
        expect(rec.sideLegend(tr)?.id ?? null, `${where} side legend`).toBe(p.side_legend);
        expect(tr.pinned && tr.kind === 'Battlefield' ? rec.misreadBattlefield(tr) : false, `${where} misread battlefield`).toBe(p.misread_battlefield);
        expect(tr.named ? (rec.pinnedTwin(t, tr)?.id ?? null) : null, `${where} pinned twin`).toBe(p.pinned_twin);
        expect(rec.held(tr.box, W, H), `${where} held`).toBe(p.held);
        expect(rec.inStrip(tr.box.centre[0], tr.box.centre[1]), `${where} on the strip`).toBe(p.in_strip);
      }
      same(Object.fromEntries([...rec.stacks(t)].map(([k, v]) => [k, v.map((u) => u.id)])), b.stacks, `board ${n} stacks`);
      same(rec.state(t, W, H), b.state, `board ${n} state`);
      for (const [card, x, y, p] of b.recently_played) expect(rec.recentlyPlayed(t, card, x, y), `board ${n} recently played`).toBe(p);
      same(rec.announce(t), b.announce, `board ${n} announce`);
      same(after(rec), b.after, `board ${n} after announce`);
      same(rec.state(t, W, H), b.state_after, `board ${n} state after announce`);
      same(rec.left(t + 6.0, b.left_box), b.left, `board ${n} left`);
      same({ ids: [...rec.tracks.keys()], ghosts: rec.ghosts }, b.after_left, `board ${n} after left`);
    });
  });

  it('re-anchors the tracks from before a cut on the ones found again', () => {
    V.reanchor.forEach((r, n) => {
      const rec = recognizer();
      setUp(rec, r.setup);
      rec.reanchor();
      same(after(rec), r.after, `reanchor ${n}`, 1e-9);
    });
  });
});

describe('the bootstrap finder', () => {
  /** gen/recognizer.py's mat_frame(): la-rq's mat with a faint print, and cards as rotated rectangles of one colour. */
  function matFrame(cards: (typeof V.bootstrap)[number]['cards']): RgbImage {
    const im = rgbImage(W, H);
    const mat = LAYOUT.mat!;
    for (let y = 0, o = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++, o++) im.data[o] = mat[c]! + ((x * 3 + y * 7 + c) % 5);
    for (const cd of cards) {
      const [cx, cy] = cd.centre;
      const reach = Math.ceil(Math.hypot(cd.long, cd.short) / 2) + 1;
      for (let y = Math.max(0, Math.floor(cy - reach)); y < Math.min(H, cy + reach); y++) {
        for (let x = Math.max(0, Math.floor(cx - reach)); x < Math.min(W, cx + reach); x++) {
          const px = x + 0.5;
          const py = y + 0.5;
          const u = (px - cx) * cd.cos + (py - cy) * cd.sin;
          const v = -(px - cx) * cd.sin + (py - cy) * cd.cos;
          if (Math.abs(u) <= cd.long / 2 && Math.abs(v) <= cd.short / 2) im.data.set(cd.rgb, (y * W + x) * 3);
        }
      }
    }
    return im;
  }

  it('finds the isolated cards on the mat as Recognizer.find does with no finder, measuring the mat again after 30 s', async () => {
    const rec = recognizer();
    for (const b of V.bootstrap) {
      const boxes = await rec.find(b.t, matFrame(b.cards));
      expect(rec.mat).toEqual(b.mat);
      same(boxes, b.boxes, `bootstrap t=${b.t}`, 1e-9);
    }
  });
});

// --- the legend rule (D-026) --------------------------------------------------------------------------------------
// ml/tests/test_live_pipeline.py's two tests of it, on test_decklist.py's rows: gallery row i is the unit vector i,
// and the stub encoder gives every crop the same scores, so a read's candidates are these scores less the rows the
// side's legend rules out. The candidates are Python's: cards, scores and rows exactly, probabilities to 1e-12.

describe('the legend rule', () => {
  const SCORES: Record<string, number> = { 'OGN-914': 0.9, 'OGN-920': 0.85, 'OGN-918': 0.8, 'SFD-902a': 0.75, 'VEN-906': 0.7 }; // the rest 0.1
  const f = Math.fround;
  const TENTH = f(0.1);
  const BEFORE: [string, number, number, number][] = [
    ['blaze-fist', 0.9054438069724243, f(0.9), 14],
    ['ember-rune', 0.08562154120964982, f(0.85), 23],
    ['hush-rune', 0.008096612568507557, f(0.8), 20],
    ['fakesmith-hammerer', 0.000765638344724473, 0.75, 5],
    ['spark-bolt', 7.240090469346561e-5, f(0.7), 13],
    ...(
      [
        ['gleaming-anvil', 0], ['thunder-crown', 2], ['silent-loom', 3], ['fakesmith-hammerer-promo', 10], ['pocket-gadget', 11],
        ['quick-trick', 12], ['iron-wall', 15], ['tidal-edict', 16], ['plain-lantern', 17], ['quiet-glade', 18], ['far-tower', 19],
        ['muse-rune', 22], ['wisp', 24], ['squire', 26], ['squire-qx', 27], ['zed-ka', 28], ['zedka', 29],
      ] as [string, number][]
    ).map(([c, i]): [string, number, number, number] => [c, 3.701612953946511e-17, TENTH, i]),
  ];
  const LEFT: [string, number, number, number][] = [
    ['hush-rune', 0.9136067854294455, f(0.8), 20],
    ['fakesmith-hammerer', 0.08639321457049592, 0.75, 5],
    ...(
      [
        ['gleaming-anvil', 0], ['silent-loom', 3], ['fakesmith-hammerer-promo', 10], ['pocket-gadget', 11], ['quick-trick', 12],
        ['plain-lantern', 17], ['quiet-glade', 18], ['far-tower', 19], ['muse-rune', 22], ['wisp', 24], ['squire', 26],
        ['squire-qx', 27], ['zed-ka', 28], ['zedka', 29],
      ] as [string, number][]
    ).map(([c, i]): [string, number, number, number] => [c, 4.176831586227724e-15, TENTH, i]),
  ];

  function ruled(legendRule?: boolean): Recognizer {
    const n = ROWS.length;
    const scores = new Float32Array(ROWS.map((r) => (r.language === 'en' ? (SCORES[r.printing_id] ?? 0.1) : 0.1)));
    const stub: Encoder = {
      name: 'stub',
      dim: n,
      embed: async (images) => {
        const out = new Float32Array(images.length * n);
        images.forEach((_, k) => out.set(scores, k * n));
        return out;
      },
    };
    const eye = new Float32Array(n * n);
    for (let i = 0; i < n; i++) eye[i * n + i] = 1;
    return new Recognizer(LAYOUT, ROWS, stub, new Pyramid(new Map([[80, eye]]), n), { fps: 5.0, gate: false, ...(legendRule === undefined ? {} : { legendRule }) });
  }

  const same = (got: [string, number, number, number][], want: [string, number, number, number][]): void => {
    expect(got.map(([c, , sc, i]) => [c, sc, i])).toEqual(want.map(([c, , sc, i]) => [c, sc, i]));
    got.forEach(([, p], k) => expect(Math.abs(p - want[k]![1])).toBeLessThan(1e-12));
  };

  it('holds a side to its pinned legend, and nowhere else', async () => {
    const rec = ruled();
    const crop = rgbImage(56, 78);
    const before = (await rec.identify([crop], ['left']))[0]!;
    same(before, BEFORE);
    expect(rec.masks.size).toBe(0);
    rec.legends.set('left', { printing_id: 'SFD-901', name: 'Gleaming Anvil' }); // Calm and Mind, pinned on the left
    const [left, right, nowhere] = await rec.identify([crop, crop, crop], ['left', 'right', '']);
    same(left!, LEFT); // Fury and Order+Chaos ruled out; the softmax over the cards left
    same(right!, BEFORE);
    same(nowhere!, BEFORE);
    expect([...rec.masks.keys()]).toEqual(['gleaming-anvil']);
    expect(rec.allowed('left')).toBe(rec.masks.get('gleaming-anvil'));
    expect([rec.allowed('right'), rec.allowed(''), rec.allowed('top')]).toEqual([null, null, null]);
    same((await rec.identify([crop]))[0]!, BEFORE); // no sides: the whole gallery
  });

  it("holds a side whose legend a pasted list names to that list's cards, and the other side to its legend", async () => {
    const crop = rgbImage(56, 78);
    const plain = ruled();
    plain.legends.set('right', { printing_id: 'VEN-912', name: 'Thunder Crown' });
    const [ruleRight] = await plain.identify([crop], ['right']);
    const rec = ruled();
    rec.legends.set('left', { printing_id: 'SFD-901', name: 'Gleaming Anvil' });
    rec.legends.set('right', { printing_id: 'VEN-912', name: 'Thunder Crown' });
    // the Gleaming Anvil list, blaze-fist on its side board; the other player's list is not given
    const list = decklist.parse('1 Fakesmith - Gleaming Anvil (SFD-901)\n3 Fakesmith - Hammerer (SFD-902)\n7 Hush Rune (OGN-918)\n1 Quiet Glade (OGN-907)\nSide Board:\n2 Blaze Fist (OGN-914)', rec.catalogue());
    rec.setLists([list]);
    const [left, right] = await rec.identify([crop, crop], ['left', 'right']);
    const cards = left!.map(([c]) => c);
    expect(cards.slice(0, 3)).toEqual(['blaze-fist', 'hush-rune', 'fakesmith-hammerer']); // listed, in every printing
    expect(cards).not.toContain('ember-rune'); // a Fury rune the list does not hold, which the legend alone allowed ...
    expect(cards).not.toContain('spark-bolt');
    expect(cards).toContain('quiet-glade');
    expect(cards).toContain('wisp'); // ... and the tokens, which no list names
    expect(right!.map(([c]) => c)).toEqual(ruleRight!.map(([c]) => c)); // no list names Thunder Crown: the legend rule
    rec.setLists([]);
    const [again] = await rec.identify([crop], ['left']);
    same(again!, LEFT); // the lists taken away: the legend rule again
  });

  it("gives the other player the other list once one player's legend is a list's, never for lists of another match", async () => {
    const crop = rgbImage(56, 78);
    const rec = ruled();
    const anvil = decklist.parse('1 Fakesmith - Gleaming Anvil (SFD-901)\n3 Fakesmith - Hammerer (SFD-902)\n7 Hush Rune (OGN-918)\n1 Quiet Glade (OGN-907)', rec.catalogue());
    const crown = decklist.parse('1 Fakezap - Thunder Crown (VEN-912)\n3 Spark Bolt (VEN-906)\n1 Far Tower (UNL-909)', rec.catalogue());
    rec.setLists([anvil, crown]);
    expect(rec.legendByElimination('right')).toBeNull(); // no legend read yet
    rec.legends.set('left', { printing_id: 'SFD-901', name: 'Gleaming Anvil' });
    expect(rec.legendByElimination('right')).toBe('thunder-crown');
    expect(rec.legendByElimination('left')).toBeNull();
    const [right] = await rec.identify([crop], ['right']);
    const cards = right!.map(([c]) => c);
    expect(cards[0]).toBe('spark-bolt'); // the Thunder Crown list's, though that legend is not read: under dice, say
    expect(cards).not.toContain('blaze-fist');
    rec.legends.set('left', { printing_id: 'OGN-913', name: 'Silent Loom' }); // lists for another match
    expect(rec.legendByElimination('right')).toBeNull();
  });

  it('holds a legend on neither pasted list to a sure read, and keeps a named card competing as itself', async () => {
    const rec = ruled();
    const tr = new Track('t0', { centre: [300, 300], long_px: 78, short_px: 56, angle_deg: 90, fill: 1 }, 0, 10, 'left');
    tr.hits = 20;
    tr.reads = 4;
    tr.prob.set('silent-loom', 1.6);
    tr.bestRow.set('silent-loom', [0.9, 3]);
    expect(rec.label(tr)[0]).toBe('named'); // four reads of a legend at 0.4: named, with no list
    rec.setLists([decklist.parse('1 Fakesmith - Gleaming Anvil (SFD-901)', rec.catalogue())]);
    expect(rec.label(tr)[0]).toBe('unsure');
    const left = Uint8Array.from(ROWS, (r) => (r.card_id === 'hush-rune' ? 1 : 0));
    const both = rec.withCard(left, 'blaze-fist'); // a unit taken from the other player, now on this side
    expect([...both].map((v, i) => (v ? ROWS[i]!.card_id : '')).filter(Boolean)).toEqual(['blaze-fist', 'hush-rune', 'hush-rune']);
  });

  it("holds a player's rune count through a hand, a box between runes and a camera cut", () => {
    const rec = ruled();
    const track = (k: number, x: number, y: number, last: number, hits = 5): Track => {
      const tr = new Track(`t${k}`, { centre: [x, y], long_px: 78, short_px: 56, angle_deg: 0, fill: 1 }, 0, last, 'left');
      tr.hits = hits;
      tr.reads = 3;
      tr.named = 'hush-rune';
      tr.kind = 'Rune';
      return tr;
    };
    const counts: number[] = [];
    const seen: number[] = [];
    for (let k = 0; k < 20; k++) {
      const t = 0.5 * k;
      const covered = k >= 4 && k < 7 ? 1.5 : t; // a hand rests on two of the three runes for 1.5 s
      const trs = [track(0, 100, 300, t), track(1, 160, 300, covered), track(2, 220, 300, covered), track(3, 104, 302, t - 0.5)];
      if (k === 10) trs.push(track(4, 130, 340, t, 2)); // a box between two runes, for a frame
      rec.tracks = new Map(trs.map((tr) => [tr.id, tr]));
      seen.push(rec.runesSeen(t, 'left', false)[0]);
      counts.push(rec.runes(t, 'left', false).count);
    }
    expect([seen[6], seen[10]]).toEqual([1, 4]); // frame by frame, the hand and the box move it
    expect(counts).toEqual(Array(20).fill(3));
    rec.away = true; // off the table camera: the count holds, and the board's clocks stop
    rec.tracks = new Map();
    rec.pause(30);
    expect(rec.runes(40, 'left', false).count).toBe(3);
    rec.away = false;
    expect(rec.runes(40.5, 'left', false).count).toBe(3);
    for (let k = 0; k < 22; k++) rec.runes(41 + 0.5 * k, 'left', false);
    expect(rec.runes(52, 'left', false)).toEqual({ count: 0, exhausted: 0 });
  });

  it("counts a stack's covered runes from the card size and its step, and a foil rune from its reads, as Python does", () => {
    const rec = ruled();
    rec.frameWh = [960, 540]; // 77.5 px cards
    rec.typeOf.set('hush-rune', 'Rune');
    const track = (k: number, x: number, kind = 'Rune', prob?: [string, number][], side = 'left'): Track => {
      const tr = new Track(`t${k}`, { centre: [x, 300], long_px: 78, short_px: 56, angle_deg: 90, fill: 1 }, 0, 10, side);
      tr.hits = 5;
      tr.kind = kind;
      tr.placed = true;
      if (prob) {
        tr.reads = 4;
        tr.prob = new Map(prob);
      }
      return tr;
    };
    // a fan of six runes a quarter of a card apart: the detector boxed five, the strip between the third and the fifth is missing
    const fan = [track(0, 100), track(1, 120), track(2, 140), track(3, 180), track(4, 200)];
    rec.tracks = new Map(fan.map((tr) => [tr.id, tr]));
    expect(rec.runesNow(10, 'left')).toBe(6);
    expect([hiddenRunes(fan.map((tr) => tr.box), 78), hiddenRunes(fan.slice(0, 2).map((tr) => tr.box), 78)]).toEqual([1, 0]);
    // a covered strip boxed but not read joins the runes beside it; a card two cards away does not
    for (const tr of [track(2, 140, ''), track(5, 300, '')]) rec.tracks.set(tr.id, tr);
    expect(rec.runesNow(10, 'left')).toBe(6);
    // a foil rune read as a spell counts on the share of its reads that say rune; a spell that only looks like one does not
    for (const tr of [track(6, 600, 'Spell', [['hush-rune', 1.6], ['blaze-fist', 2.4]], 'right'), track(7, 800, 'Spell', [['hush-rune', 0.8], ['blaze-fist', 3.2]], 'right')])
      rec.tracks.set(tr.id, tr);
    expect(rec.runesNow(10, 'right')).toBe(1);
  });

  it('sees a hand around a card off the other cards, as Python does', () => {
    const mat = rgbImage(960, 540, Uint8Array.from({ length: 960 * 540 * 3 }, (_, i) => [30, 40, 55][i % 3]!));
    const fingers = rgbImage(960, 540, Uint8Array.from(mat.data));
    for (let y = 200; y < 278; y++) for (let x = 326; x < 350; x++) fingers.data.set([200, 140, 110], (y * 960 + x) * 3);
    const box: CardBox = { centre: [378, 239], long_px: 78, short_px: 56, angle_deg: 90, fill: 1 };
    const other: CardBox = { centre: [338, 239], long_px: 78, short_px: 30, angle_deg: 90, fill: 1 };
    const tilted: CardBox = { ...box, angle_deg: 63 };
    const table = [0, 0, 960, 540] as const;
    // the numbers live/pipeline.py's hand_share gives on the same pictures
    expect([handShare(mat, box, table), handShare(fingers, box, table), handShare(fingers, box, table, [other])]).toEqual([0, 0.1875, 0]);
    expect([handShare(fingers, tilted, table), handShare(fingers, tilted, [340, 0, 960, 540])]).toEqual([0.1875, 0.1]);
  });

  it("reads a track's crops on its side, and can be turned off", async () => {
    let x = 12345;
    const noise = rgbImage(960, 540, Uint8Array.from({ length: 960 * 540 * 3 }, () => ((x = (x * 1103515245 + 12345) >>> 0) >>> 16) & 255));
    for (const rule of [true, false]) {
      const rec = ruled(rule);
      rec.legends.set('left', { printing_id: 'SFD-901', name: 'Gleaming Anvil' });
      const box = (cx: number): CardBox => ({ centre: [cx, 200.0], long_px: 78.0, short_px: 56.0, angle_deg: 90.0, fill: 1.0 });
      const a = new Track('a', box(200.0), 0.0, 0.0, 'left');
      const b = new Track('b', box(700.0), 0.0, 0.0, 'right');
      a.hits = b.hits = 2;
      await rec.read(0.0, noise, [a, b]);
      expect([a.reads, b.reads]).toEqual([1, 1]);
      expect(a.prob.has('blaze-fist')).toBe(!rule);
      expect(b.prob.has('blaze-fist')).toBe(true);
      const best = [...a.prob].sort((p, q) => q[1] - p[1])[0]![0];
      expect(best).toBe(rule ? 'hush-rune' : 'blaze-fist');
    }
  });
});
