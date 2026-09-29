// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// How the engine's run on the LA final's frames (the browser replay) is compared with the Python reference
// (fixtures/recognizer/steps.jsonl, from packages/engine/test/gen/recognizer_replay.py): each step's state and
// events, exact and to a few tolerances. The engine host adds to a state what the live runner adds (fps, latency)
// and what only the browser has (engine); the reference has none of it, so it is not compared.

/** What a step of the run gives: the recogniser's state and events, and how long the frame took. */
export interface GotStep {
  i: number;
  file: string;
  t: number;
  state: Record<string, unknown>;
  events: unknown[];
}

/** A line of the reference's steps.jsonl (it holds more: the finder's boxes, the embed calls). */
export interface WantStep {
  i: number;
  file: string;
  t: number;
  state: Record<string, unknown>;
  events: unknown[];
}

const HOST = ['fps', 'latency_s', 'engine'];

/** The state as the Python recogniser gives it: without what the host adds. */
export function recogniserState(state: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(state).filter(([k]) => !HOST.includes(k)));
}

/** The first place `got` and `want` differ, as a path and the two values, or null when they agree: numbers within
 * `tol`, anything else exactly. */
export function firstDifference(got: unknown, want: unknown, path: string, tol = 0): string | null {
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

export interface Agreement {
  /** The tolerance for numbers (0: exact). */
  tol: number;
  /** Steps compared. */
  steps: number;
  /** Steps whose state and events agree. */
  identical: number;
  /** Steps whose events agree. */
  eventsEqual: number;
  /** The first step that differs, and where. */
  firstDivergence: string | null;
  /** The steps that differ. */
  divergent: number[];
}

/** The two runs compared step by step at one tolerance. */
export function compareRuns(got: readonly GotStep[], want: readonly WantStep[], tol: number): Agreement {
  const n = Math.min(got.length, want.length);
  const out: Agreement = { tol, steps: n, identical: 0, eventsEqual: 0, firstDivergence: null, divergent: [] };
  for (let k = 0; k < n; k++) {
    const g = got[k]!;
    const w = want[k]!;
    const dEvents = firstDifference(g.events, w.events, 'events', tol);
    const d = firstDifference(recogniserState(g.state), w.state, 'state', tol) ?? dEvents;
    if (!dEvents) out.eventsEqual++;
    if (d === null) {
      out.identical++;
    } else {
      out.divergent.push(k);
      out.firstDivergence ??= `step ${k} (${w.file}, t=${w.t}): ${d}`;
    }
  }
  return out;
}

/** The comparison in a few lines, one for each tolerance. */
export function summarize(runs: readonly Agreement[], got: number, want: number): string {
  const lines = [`steps: ${got} run, ${want} in the reference`];
  for (const a of runs) {
    const tol = a.tol === 0 ? 'exact' : `numbers within ${a.tol}`;
    lines.push(`${tol}: ${a.identical} of ${a.steps} steps identical, events equal on ${a.eventsEqual}${a.firstDivergence ? `; first divergence: ${a.firstDivergence}` : ''}`);
  }
  return lines.join('\n');
}

/** The lines of a .jsonl file as objects. */
export const jsonLines = <T>(text: string): T[] => text.split('\n').filter((l) => l.trim() !== '').map((l) => JSON.parse(l) as T);

/** A card box as the finder gives it (live/pipeline.py detector_boxes; types.CardBox). */
export interface Box {
  centre: [number, number];
  long_px: number;
  short_px: number;
  angle_deg: number;
  fill: number;
  back?: boolean;
  score?: number;
  vis?: number;
}

/** How near the finder's boxes for a frame are to Python's: the same number of them, each matched to the box of Python's
 * whose centre is nearest (one to one), the largest difference in any number of the matched boxes (`back` must be
 * equal), and whether they come in the same order (the tracker matches in the order it is given). */
export function compareBoxes(got: readonly Box[], want: readonly Box[]): { same: boolean; maxDelta: number; back: boolean; inOrder: boolean } {
  if (got.length !== want.length) return { same: false, maxDelta: NaN, back: true, inOrder: false };
  const pairs = got
    .flatMap((g, i) => want.map((w, j) => ({ i, j, d: Math.hypot(g.centre[0] - w.centre[0], g.centre[1] - w.centre[1]) })))
    .sort((a, b) => a.d - b.d);
  const partner = new Array<number>(got.length).fill(-1);
  const taken = new Set<number>();
  for (const { i, j } of pairs) {
    if (partner[i]! >= 0 || taken.has(j)) continue;
    partner[i] = j;
    taken.add(j);
  }
  let maxDelta = 0;
  let back = true;
  got.forEach((g, i) => {
    const w = want[partner[i]!]!;
    const d = [g.centre[0] - w.centre[0], g.centre[1] - w.centre[1], g.long_px - w.long_px, g.short_px - w.short_px, g.angle_deg - w.angle_deg, g.fill - w.fill];
    if (w.score !== undefined) d.push((g.score ?? NaN) - w.score);
    if (w.vis !== undefined) d.push((g.vis ?? NaN) - w.vis);
    for (const x of d) maxDelta = Math.max(maxDelta, Math.abs(x));
    if (Boolean(g.back) !== Boolean(w.back)) back = false;
  });
  return { same: true, maxDelta, back, inOrder: partner.every((j, i) => j === i) };
}

/** How near the rows of one embed() call are to Python's (rows of `dim`, L2-normalised): the lowest cosine between a
 * row and its counterpart, and the largest difference in any value. */
export function compareRows(got: Float32Array, want: Float32Array, dim: number): { minCos: number; maxAbs: number } {
  let minCos = 1;
  let maxAbs = 0;
  for (let at = 0; at + dim <= want.length; at += dim) {
    let dot = 0;
    for (let k = 0; k < dim; k++) {
      dot += got[at + k]! * want[at + k]!;
      maxAbs = Math.max(maxAbs, Math.abs(got[at + k]! - want[at + k]!));
    }
    minCos = Math.min(minCos, dot);
  }
  return { minCos, maxAbs };
}

/** What a step's diagnostics say (the page compares the finder and the embedder with Python's and sends this). */
export interface StepDiag {
  i: number;
  boxes: { got: number; want: number; maxDelta: number; back: boolean; inOrder: boolean };
  embeds: { got: number[]; want: number[]; rows: number; minCos: number; maxAbs: number };
}

/** The diagnostics of all the steps in a few lines: where the detector and the embedder first parted from Python's, and how far. */
export function summarizeDiag(steps: readonly StepDiag[]): string {
  const boxesOff = steps.filter((d) => d.boxes.got !== d.boxes.want || !d.boxes.back);
  const reordered = steps.filter((d) => !d.boxes.inOrder && d.boxes.got === d.boxes.want);
  const embedsOff = steps.filter((d) => d.embeds.got.length !== d.embeds.want.length || d.embeds.got.some((n, k) => n !== d.embeds.want[k]));
  const delta = Math.max(0, ...steps.map((d) => (Number.isNaN(d.boxes.maxDelta) ? 0 : d.boxes.maxDelta)));
  const rows = steps.reduce((a, d) => a + d.embeds.rows, 0);
  const cos = Math.min(1, ...steps.filter((d) => d.embeds.rows > 0).map((d) => d.embeds.minCos));
  const abs = Math.max(0, ...steps.map((d) => d.embeds.maxAbs));
  return [
    `finder: ${steps.length - boxesOff.length} of ${steps.length} steps gave as many boxes as Python's, the largest difference in a number of a matched box ${delta.toExponential(2)}, ${reordered.length} steps with the boxes in another order${boxesOff.length ? `; first step with other boxes: ${boxesOff[0]!.i}` : ''}`,
    `embedder: ${rows} rows compared with Python's, lowest cosine ${cos.toFixed(9)}, largest difference in a value ${abs.toExponential(2)}${embedsOff.length ? `; first step whose embed calls differ from Python's: ${embedsOff[0]!.i}` : '; every step made the same embed calls'}`,
  ].join('\n');
}
