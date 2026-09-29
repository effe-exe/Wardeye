// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The decoded detector check: the check tile's outputs turned into cards with the engine's port of the detector's
// postprocess (@rifteye/engine, as Detector.detect_tiles does it), and matched to the cards Python found on the same
// tile. Comparing the raw outputs query by query cannot judge float16: it reorders near-tied queries, which moves
// whole rows of the outputs but not the cards. Pure functions, no DOM.
//
// The rule, for a threshold T (the live runner's --det-score, 0.4) and a precision's tolerances (corners in px,
// score): cards pair up one to one, same class, closest corners first (the largest of the four corner distances),
// no pair farther apart than the corner tolerance. An expected card scoring T + the score tolerance or more must be
// found within both tolerances; a found card scoring that much must be an expected one. Cards scoring less may be
// missing on either side: a score that moves by its tolerance can cross T, so being kept or not there is a coin
// toss, not a fault. The check passes when every card that must be found is found and nothing extra is.

import { detector, type Detection } from '@rifteye/engine';

export interface DetectionTolerance {
  corner_px: number;
  score: number;
}

export interface DecodedCheck {
  pass: boolean;
  threshold: number;
  tolerance: DetectionTolerance | null;
  /** Expected cards that must be found: those scoring threshold + score tolerance or more. */
  required: number;
  /** Of those, found within both tolerances. */
  matched: number;
  /** matched / required (1 when nothing is required). */
  share: number;
  /** Over every pair: the largest corner distance (px) and score difference. */
  worstCornerPx: number;
  worstScore: number;
  /** Cards that had to be found and were not (or were found too far off), and found ones no expected card explains. */
  missing: string[];
  extra: string[];
  /** Cards near the threshold left unpaired on either side: allowed. */
  near: number;
  /** Why there is no verdict, when there is none. */
  note: string;
}

/** The expected detections file: the cards Detector.detect_tiles found on the check tile, in tile px. */
export interface ExpectedDetections {
  tile: number;
  threshold: number;
  detections: Detection[];
}

type Json = Record<string, unknown>;
const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const numbers = (v: unknown, n: number): v is number[] => Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number');

/** Reads and checks an expected detections file; throws an Error that names what is wrong. */
export function parseDetections(json: unknown): ExpectedDetections {
  if (!isObject(json) || !Array.isArray(json.detections)) throw new Error('expected detections: no "detections" list');
  const detections = json.detections.map((d, i): Detection => {
    const at = `detections[${i}]`;
    if (!isObject(d)) throw new Error(`${at}: expected an object`);
    if (d.cls !== 'card' && d.cls !== 'card_back') throw new Error(`${at}.cls: expected "card" or "card_back"`);
    if (typeof d.score !== 'number') throw new Error(`${at}.score: expected a number`);
    if (!numbers(d.box, 4)) throw new Error(`${at}.box: expected four numbers`);
    if (!Array.isArray(d.quad) || d.quad.length !== 4 || !d.quad.every((p) => numbers(p, 2))) throw new Error(`${at}.quad: expected four [x, y]`);
    const found = Array.isArray(d.found) ? (d.found as number[]) : [];
    const visible = Array.isArray(d.visible) ? (d.visible as number[]) : [];
    return { cls: d.cls, score: d.score, box: d.box as [number, number, number, number], quad: d.quad as [number, number][], found, visible };
  });
  return { tile: typeof json.tile === 'number' ? json.tile : 576, threshold: typeof json.threshold === 'number' ? json.threshold : 0.3, detections };
}

/** The detections of the first item of a batch of the detector's outputs, as detect_tiles makes them. */
export function decodeFirst(outputs: { pred_logits: Float32Array; pred_boxes: Float32Array; pred_keypoints: Float32Array }, batch: number, threshold = 0.3): Detection[] {
  const first = detector.tileOutputs(outputs, 0, batch);
  return detector.decodeTile(first.pred_logits, first.pred_boxes, first.pred_keypoints, threshold);
}

/** The largest of the four distances between two cards' corners. */
export function cornerDistance(a: Detection, b: Detection): number {
  return Math.max(...a.quad.map(([x, y], k) => Math.hypot(x - b.quad[k]![0], y - b.quad[k]![1])));
}

const label = (d: Detection) => `${d.cls} ${d.score.toFixed(3)}`;

/** Judges found cards against the expected ones by the rule above. */
export function judgeDetections(got: readonly Detection[], want: readonly Detection[], threshold: number, tolerance: DetectionTolerance | undefined): DecodedCheck {
  const base = { threshold, tolerance: tolerance ?? null, required: 0, matched: 0, share: NaN, worstCornerPx: NaN, worstScore: NaN, missing: [], extra: [], near: 0 };
  if (!tolerance) return { ...base, pass: false, note: 'no tolerance for this precision' };
  const pairs: [number, number, number][] = [];
  want.forEach((w, i) =>
    got.forEach((g, j) => {
      if (g.cls !== w.cls) return;
      const c = cornerDistance(g, w);
      if (c <= tolerance.corner_px) pairs.push([c, i, j]);
    }),
  );
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const pairOfWant = new Map<number, number>();
  const pairedGot = new Set<number>();
  let worstCornerPx = 0;
  let worstScore = 0;
  for (const [c, i, j] of pairs) {
    if (pairOfWant.has(i) || pairedGot.has(j)) continue;
    pairOfWant.set(i, j);
    pairedGot.add(j);
    worstCornerPx = Math.max(worstCornerPx, c);
    worstScore = Math.max(worstScore, Math.abs(got[j]!.score - want[i]!.score));
  }
  const must = threshold + tolerance.score;
  const missing: string[] = [];
  const extra: string[] = [];
  let required = 0;
  let matched = 0;
  let near = 0;
  want.forEach((w, i) => {
    const j = pairOfWant.get(i);
    if (w.score < must) {
      if (j === undefined) near++;
      return;
    }
    required++;
    if (j !== undefined && Math.abs(got[j]!.score - w.score) <= tolerance.score) matched++;
    else missing.push(j === undefined ? label(w) : `${label(w)} (found at ${got[j]!.score.toFixed(3)})`);
  });
  got.forEach((g, j) => {
    if (pairedGot.has(j)) return;
    if (g.score >= must) extra.push(label(g));
    else near++;
  });
  return {
    pass: missing.length === 0 && extra.length === 0,
    threshold,
    tolerance,
    required,
    matched,
    share: required ? matched / required : 1,
    worstCornerPx: pairOfWant.size ? worstCornerPx : NaN,
    worstScore: pairOfWant.size ? worstScore : NaN,
    missing,
    extra,
    near,
    note: '',
  };
}

/** "decoded 57/58 cards above 0.6 (98.3%), worst corner 4.70 px, score 0.137; missing card 0.671 FAIL". */
export function decodedText(d: DecodedCheck): string {
  if (d.note) return `decoded: ${d.note}`;
  const fmt = (v: number, digits: number) => (Number.isFinite(v) ? v.toFixed(digits) : '-');
  const above = d.tolerance ? (d.threshold + d.tolerance.score).toFixed(2) : String(d.threshold);
  const parts = [`decoded ${d.matched}/${d.required} cards above ${above} (${(d.share * 100).toFixed(1)}%), worst corner ${fmt(d.worstCornerPx, 2)} px, score ${fmt(d.worstScore, 3)}`];
  if (d.missing.length) parts.push(`missing ${d.missing.join(', ')}`);
  if (d.extra.length) parts.push(`extra ${d.extra.join(', ')}`);
  if (d.near) parts.push(`${d.near} near the threshold unpaired`);
  return `${parts.join('; ')} ${d.pass ? 'PASS' : 'FAIL'}`;
}
