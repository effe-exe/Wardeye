// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Stand-in parts for the browser test (built in place of src/parts-engine.ts): the real host, on stand-in models
// (tiny ONNX graphs made at test time) and a stand-in recogniser. It finds the test video's coloured blocks, embeds
// each with the tiny embedder (mean colour), and names it from the stand-in gallery, so the frame's whole way is
// exercised: the page's JPEG, the parity decode, onnxruntime-web in a worker (WebGPU or WASM), the gallery and the
// catalogue, the state, the overlay.

import { decklist, layouts, type CatalogRow, type Layout, type RgbImage } from '@rifteye/engine';
import type { Track } from '../src/geometry';
import type { Board, BoardResult, ListSummary, LoadContext, Parts } from '../src/parts';

const STANDIN: Layout = layouts.makeLayout({ name: 'standin', title: 'Stand-in table', table: [0, 0, 1, 1], card_long_1080: 252 });
const BACKGROUND = [16, 48, 62]; // '#10303e', the test video's floor
const T = 0.05;

interface Block {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** The blocks on the floor: 4-connected cells of a coarse grid that are not the floor's colour. */
export function blocks(f: RgbImage, cell = 4): Block[] {
  const gw = Math.floor(f.width / cell);
  const gh = Math.floor(f.height / cell);
  const on = new Uint8Array(gw * gh);
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const p = ((j * cell + 1) * f.width + (i * cell + 1)) * 3;
      const d = Math.abs(f.data[p]! - BACKGROUND[0]!) + Math.abs(f.data[p + 1]! - BACKGROUND[1]!) + Math.abs(f.data[p + 2]! - BACKGROUND[2]!);
      on[j * gw + i] = d > 90 ? 1 : 0;
    }
  }
  const seen = new Uint8Array(gw * gh);
  const out: Block[] = [];
  for (let s = 0; s < on.length; s++) {
    if (!on[s] || seen[s]) continue;
    const stack = [s];
    seen[s] = 1;
    let x0 = gw;
    let y0 = gh;
    let x1 = 0;
    let y1 = 0;
    let n = 0;
    while (stack.length) {
      const c = stack.pop()!;
      const cx = c % gw;
      const cy = (c - cx) / gw;
      n++;
      x0 = Math.min(x0, cx);
      y0 = Math.min(y0, cy);
      x1 = Math.max(x1, cx + 1);
      y1 = Math.max(y1, cy + 1);
      for (const [nx, ny] of [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]] as const) {
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
        const k = ny * gw + nx;
        if (on[k] && !seen[k]) {
          seen[k] = 1;
          stack.push(k);
        }
      }
    }
    if (n >= 12) out.push({ x0: x0 * cell, y0: y0 * cell, x1: x1 * cell, y1: y1 * cell });
  }
  return out.sort((a, b) => a.x0 - b.x0);
}

/** A block's pixels as the embedder's input: 8 x 8, channel by channel, values 0..255 (mean over each cell). */
export function crop8(f: RgbImage, b: Block): Float32Array {
  const out = new Float32Array(3 * 8 * 8);
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) {
      const xa = b.x0 + Math.floor(((b.x1 - b.x0) * i) / 8);
      const xb = Math.max(xa + 1, b.x0 + Math.floor(((b.x1 - b.x0) * (i + 1)) / 8));
      const ya = b.y0 + Math.floor(((b.y1 - b.y0) * j) / 8);
      const yb = Math.max(ya + 1, b.y0 + Math.floor(((b.y1 - b.y0) * (j + 1)) / 8));
      const sum = [0, 0, 0];
      let n = 0;
      for (let y = ya; y < yb; y++) {
        for (let x = xa; x < xb; x++) {
          const p = (y * f.width + x) * 3;
          for (let c = 0; c < 3; c++) sum[c]! += f.data[p + c]!;
          n++;
        }
      }
      for (let c = 0; c < 3; c++) out[c * 64 + j * 8 + i] = sum[c]! / n;
    }
  }
  return out;
}

export async function loadParts(ctx: LoadContext): Promise<Parts> {
  const { ort, pkg, attempt, read, gallery, timer, decode } = ctx;
  const ep = attempt.runtime;
  const file = (m: { fp16?: string; fp32?: string }, precision: 'fp16' | 'fp32'): string => m[precision] ?? m.fp32 ?? m.fp16!;
  ctx.progress('loading the stand-in models');
  const detector = await ort.InferenceSession.create(await read(file(pkg.detector, attempt.detector)), { executionProviders: [ep] });
  const embedder = await ort.InferenceSession.create(await read(file(pkg.embedder, attempt.embedder)), { executionProviders: [ep] });
  const rows: readonly CatalogRow[] = gallery.rows;
  const level = gallery.levels.get(gallery.index.levels[0]!)!;
  const dim = gallery.index.dim;

  const board = (layout: Layout): Board => {
    const announced = new Set<string>(); // as the recogniser does: a card is played once, when it is first named
    return {
      async step(t: number, f: RgbImage): Promise<BoardResult> {
        // the tiny detector, timed as the real one is: it sees the frame's first 8 x 8 values
        await timer.span('detect', async () => {
          const x = new ort.Tensor('float32', Float32Array.from({ length: 192 }, (_, i) => f.data[i]!), [1, 3, 8, 8]);
          const out = await detector.run({ x });
          if (out.z!.dims[1] !== 192) throw new Error('the stand-in detector answered with the wrong shape');
        });
        const tracks: Track[] = [];
        for (const [k, b] of blocks(f).entries()) {
          const emb = await timer.span('embed', async () => {
            const out = await embedder.run({ crops: new ort.Tensor('float32', crop8(f, b), [1, 3, 8, 8]) });
            const v = Float32Array.from(out.embedding!.data as Float32Array);
            const n = Math.hypot(...v) || 1;
            return v.map((c) => c / n);
          });
          const sims = rows.map((_, r) => emb.reduce((s, c, d) => s + c * level[r * dim + d]!, 0));
          const best = sims.map((s, r) => [Math.exp((s - Math.max(...sims)) / T), r] as const);
          const total = best.reduce((s, [e]) => s + e, 0);
          const ranked = best.map(([e, r]) => ({ r, p: e / total })).sort((a, c) => c.p - a.p);
          const top = ranked[0]!;
          const named = top.p >= 0.85;
          const row = rows[top.r]!;
          tracks.push({
            id: `t${k}`,
            quad: [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]] as Track['quad'],
            side: b.x0 + b.x1 < f.width ? 'left' : 'right',
            state: named ? 'named' : 'unsure',
            printing_id: named ? row.printing_id : null,
            name: named ? row.name : '',
            confidence: Math.round(top.p * 1000) / 1000,
            guesses: ranked.slice(0, 3).map(({ r, p }) => ({ printing_id: rows[r]!.printing_id, card_id: rows[r]!.card_id, name: rows[r]!.name, p: Math.round(p * 1000) / 1000 })),
            kind: 'card',
            hidden: false,
          });
        }
        const events = tracks
          .filter((tr) => tr.state === 'named' && !announced.has(tr.id))
          .map((tr) => {
            announced.add(tr.id);
            return { t, kind: 'played', text: `${tr.name} played`, printing_id: tr.printing_id, track: tr.id, side: tr.side };
          });
        const players = [
          { side: 'left', label: 'Player 1', legend: null },
          { side: 'right', label: 'Player 2', legend: null },
        ];
        return {
          state: { t, status: 'live', message: '', title: layout.title, frame: { width: f.width, height: f.height }, players, tracks },
          events,
        };
      },
    };
  };

  // decklists as the engine's parts read them (parts-engine.ts), through the stand-in gallery's rows
  const cat = new decklist.Catalogue(rows);
  const readLists = (texts: readonly string[]) => {
    const read = texts.map((text): [decklist.Deck | null, ListSummary] => {
      try {
        const deck = decklist.parse(text, cat);
        return [deck, { legends: deck.legends(), cards: deck.cardIds().size, unmapped: deck.unmapped, error: null }];
      } catch (e) {
        return [null, { legends: [], cards: 0, unmapped: [], error: e instanceof Error ? e.message : String(e) }];
      }
    });
    return { lists: read.flatMap(([d]) => (d ? [d] : [])), summaries: read.map(([, sm]) => sm) };
  };

  return {
    decode,
    findLayout: async () => STANDIN,
    presets: () => [STANDIN],
    board,
    readLists,
    dispose: async () => {
      await detector.release();
      await embedder.release();
    },
  };
}
