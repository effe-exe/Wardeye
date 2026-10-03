// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The live recogniser, ported from ml/rifteye_ml/live/pipeline.py: cards found, tracked and named frame by frame, as
// a stream plays.
//
// Each frame the finder gives card boxes inside the layout's table window: the trained detector's (detector_boxes),
// or without one the bootstrap finder's (matcrops.find_cards, isolated cards on the mat). Boxes are matched to the
// tracks of cards already on the table by position and size, so a card is identified
// once, not once per frame: new tracks and uncertain ones get a crop identified against the gallery pyramid, and
// named tracks are re-checked now and then. A track's name is its readings' mean probability (softmax over each
// crop's card scores). Plain crops are face-down cards: they are never identified (D-005). Only the table window is
// ever looked at; the broadcast's panels with the players' hands are hidden information and never read.
//
// A card named on the table becomes a "played" event; a named card gone for a while, "left". The finder only sees
// cards lying on their own, so the change gate watches the whole table too: when a region settles after a change
// (a card put on a stack, a rune channelled), the region is read like a crop, and a confident read is a play even
// though no track holds the card.
//
// The legend rule (D-026, priors.legendMask): once a side's legend is pinned, every crop read on that side competes
// only with the printings that fit that legend's domains, runes included, and with every battlefield and token.
// Until then, and with `legendRule: false`, a crop competes with the whole gallery.
//
// The port keeps pipeline.py's names (camelCase), its order and its arithmetic, so the two read side by side and,
// fed the same frames, boxes and embeddings, give the same state and events. Where numpy computes in float32 (the
// scene's thumbnails), so does this (Math.fround at each step). A few things can differ in their last bits: the
// scene's two vector norms (numpy's BLAS sums them in its own order), exp, cos and sin (the platform's), and the
// similarity fit (a closed form for the SVD); none changes a step of the LA final replay. `step` is asynchronous
// because the finder and the encoder are.

import { ChangeGate, gateSettings, skin, viewHeight, viewToFrame, type GateSettings } from './changegate';
import { dist, mean as meanOf, npRound, overlapArea } from './geometry';
import * as image from './image';
import { box as layoutBox, cardPx, side as layoutSide, sides as layoutSides } from './layouts';
import { linearSumAssignment } from './lsap';
import { FACE_DOWN_DETAIL, detail, findCards, matColour, notmatMask, type Mask } from './matcrops';
import { Catalogue, listMask, type Deck } from './decklist';
import { binaryFillHoles, findObjects, label } from './ndimage';
import { legendMask, tokenRows } from './priors';
import { pyMod, pyRound } from './pynum';
import { ROTATIONS, argsortDescending, bestSimilarities, type Pyramid } from './retrieval';
import type { CardBox, CatalogRow, Encoder, Finder, Layout, RgbImage } from './types';

/** Fitted on the M0 real labels (reviewpack identity). */
export const TEMPERATURE = 0.0212;
/** A named card out of sight this long has gone: not a hand over it, dice, a card on top. */
export const KEEP_S = 60.0;
/** A named card that vanished this recently and is named again elsewhere has moved. */
export const MOVE_S = 10.0;
/** Set up before the game: pinned where they are named (a moved battlefield's pin follows it). */
export const STATIC: readonly string[] = ['Legend', 'Battlefield'];
/** Tracked, but never labelled, listed or announced: not worth watching. */
export const QUIET: readonly string[] = ['Rune'];
/** A pinned battlefield read this often, and as itself on fewer than one read in `UNPIN_P`, was misread once: a rune or a
 * unit turned sideways (exhausted) looks like a battlefield's landscape art. */
export const UNPIN_READS = 4;
export const UNPIN_P = 0.2;
/** A legend or battlefield out of sight this long is not drawn (a hand resting on it is shorter). */
export const PIN_HIDE_S = 20.0;
/** A card past the table window's edge on a player's side by this much of its width is in a hand. */
export const HELD_OUT = 0.15;
/** The battlefield strip: the band this far either side of the midline between the players, as a share of the table's
 * width across them (the official mat's is 0.11; battlefields measured within 0.09). */
export const STRIP = 0.12;
/** Off the strip, a battlefield is named only after this many reads, and never pinned: a rune or a unit turned sideways
 * looks like a battlefield's landscape art, and players lay out their tables differently. */
export const STRIP_READS = 3;
/** A rune counts in a frame while seen this recently (a hand passing over it is shorter). */
export const RUNE_SEEN_S = 1.0;
/** Two rune boxes closer than this share of a card's length are one rune (two tracks on one card). */
export const RUNE_APART = 0.15;
/** A rune box longer than this for its width, or than RUNE_LONG cards, spans two runes: the detector's slip. */
export const RUNE_ASPECT = 1.7;
export const RUNE_LONG = 1.3;
/** A player's rune count is the upper quartile of the frames' counts over this long, and the exhausted ones the median
 * over RUNE_EXHAUSTED_S: hands over the runes and boxes between two come and go, runes stay. */
export const RUNE_WINDOW_S = 12.0;
export const RUNE_EXHAUSTED_S = 3.0;
/** A box whose reads put this share on runes is one, whatever it is named: runes are counted, never named, and a foil rune
 * or a stacked one's strip is often read as another card. */
export const RUNE_SHARE = 0.3;
/** An unread or unnamed card-sized box this close to a rune (in cards), turned its way (within RUNE_TURN degrees, and its
 * length within RUNE_SIZE of a card's), is one too: a stack's covered strips. */
export const RUNE_JOIN = 0.6;
export const RUNE_TURN = 20.0;
export const RUNE_SIZE = 0.2;
/** Runes this close (in cards) are one stack: a column or a fan. */
export const RUNE_LINK = 0.6;
/** A stack's step from strip to strip, as shares of a card's length: the gaps of that size. */
export const RUNE_STEP: readonly [number, number] = [0.18, 0.35];
/** A gap this many steps wide hides runes the detector missed: about gap / step - 1 of them. */
export const RUNE_GAP = 1.6;
/** A card with this share of skin in the band around it is in a hand: held, or being put down. */
export const HAND_SKIN = 0.1;
/** Out of a hand and still this long before a card is read: a card held over the table is never named. */
export const HAND_FREE_S = 0.5;
/** Still: moved less than this share of a card's length since (a hand that holds a card moves it). */
export const HAND_STILL = 0.04;
/** The band looked at, as distances out from the card's edges in shares of its width. */
export const HAND_RING = [0.1, 0.2, 0.3] as const;
/** Points along each side of the band, at each distance. */
export const HAND_POINTS = 8;
/** Fewer of them off the other cards and in the window: no hand to see. */
export const HAND_MIN_POINTS = 8;
/** A skin-coloured point this unlike the still table there (`StillTable`) is a hand; one like it is the table itself: a
 * wooden table is skin-coloured, and the cards lying beside it are not in a hand. */
export const HAND_DIFF = 40;
/** The still table: the frame at a quarter of 1080p, first the median of HAND_BG_FRAMES table frames HAND_BG_EVERY apart
 * (hands move; the table does not), then every HAND_BG_EVERY HAND_BG_STEP nearer the frame, so a hand passing over it
 * stays a hand. */
export const HAND_BG: readonly [number, number] = [480, 270];
export const HAND_BG_FRAMES = 5;
export const HAND_BG_EVERY = 0.5;
export const HAND_BG_STEP = 2;
/** A box longer than this many cards is not one: the co-stream's chat, two cards or a card and the printed zone beside it
 * outlined as one (named cards: 99.5% within 1.24 on two finals). */
export const SIZE_MAX = 1.45;
/** A box the mat's own colour inside as around it (medians, this close), and plain inside (the middle half of its points
 * within PLAIN_SPREAD), is a zone printed on the mat, not a card. */
export const PLAIN_TOL = 18;
export const PLAIN_SPREAD = 24;
/** This many cards named for the first time on the table within BURST_S are not that many plays: a graphic of cards (a
 * sideboard, a decklist) or a view framed anew; nobody plays four cards a second. */
export const BURST = 4;
export const BURST_S = 1.0;
/** A thumbnail pixel within this of the frame before, through a cut, stayed: the broadcast's overlay. */
export const OVERLAY_TOL = 16;
/** A frame whose thumbnail changed this much from the one before is a cut (play changes a quarter at most). */
export const CUT_SHARE = 0.5;
/** The overlay: what stayed through this share of the cuts, once there are OVERLAY_CUTS, in patches reaching within
 * OVERLAY_EDGE pixels of the frame's edge (a co-streamer's webcam and chat, a scoreboard, a sponsor banner), holes filled. */
export const OVERLAY_SHARE = 0.9;
export const OVERLAY_CUTS = 4;
export const OVERLAY_EDGE = 2;
/** A pixel once overlay stays so while it stayed through this share of the cuts: a face moving in a webcam changes it
 * at some cuts, and opens the webcam's frame to the table. */
export const OVERLAY_KEEP = 0.6;
/** Before then, what stayed through every cut so far is suspect: nothing there is read, until the cuts make it overlay,
 * or for this long after the last cut (one cut alone can be the camera reframed). */
export const SUSPECT_S = 60.0;
/** The scene's score is the mean of its scores in a 6 x 3 grid of blocks of the thumbnail: an arm or a banner over the
 * table spoils a block or two, another shot all of them. A block counts with SCENE_BLOCK still pixels of the table
 * window in it. */
export const SCENE_GRID: readonly [number, number] = [6, 3];
export const SCENE_BLOCK = 20;
/** The scene learns a moved camera again only when this many card-sized boxes lie in the view, and half the most the
 * table camera showed in CARDS_S of its last time on screen. */
export const RELEARN_CARDS = 5;
export const CARDS_S = 60.0;
export const KINDS: ReadonlyMap<string, string> = new Map([
  ['Legend', 'legend'],
  ['Battlefield', 'battlefield'],
  ['Rune', 'rune'],
]);

/** kind in STATIC + QUIET */
const settled = (kind: string): boolean => STATIC.includes(kind) || QUIET.includes(kind);

const DEG_TO_RAD = Math.PI / 180; // math.radians
const RAD_TO_DEG = 180 / Math.PI; // math.degrees
const f32 = Math.fround;

// --- what the overlay draws ---------------------------------------------------------------------------------------

/** One of a track's candidate cards: that card's best-matching printing, and its probability. */
export interface Guess {
  printing_id: string;
  card_id: string;
  name: string;
  p: number;
}

/** A card that lies under another. */
export interface Under {
  id: string;
  name: string;
  printing_id: string;
}

/** A card on the table as the overlay draws it (apps/extension/src/geometry.ts's Track). The gate's plays are
 * listed the same way for a few seconds, without `under`. */
export interface StateTrack {
  id: string;
  /** The corners in frame px, rounded to 0.1. */
  quad: [number, number][];
  side: string;
  /** new, named, unsure or facedown. */
  state: string;
  printing_id: string | null;
  name: string;
  confidence: number;
  guesses: Guess[];
  /** When it was first seen, in s. */
  since: number;
  /** card, legend, battlefield or rune. */
  kind: string;
  /** Out of sight (under a hand or another card) but still on the board: listed, not drawn. */
  hidden: boolean;
  under?: Under[];
}

export interface Legend {
  printing_id: string;
  name: string;
}

/** A player's runes on the table: counted, never named, and how many are exhausted (used). */
export interface Runes {
  count: number;
  exhausted: number;
}

export interface Player {
  side: string;
  label: string;
  legend: Legend | null;
  runes: Runes;
}

/** What `step` returns: the state the overlay draws (apps/extension/src/geometry.ts's State). */
export interface RecognizerState {
  t: number;
  /** live, or away while the video is not the table camera. */
  status: string;
  message: string;
  title: string;
  frame: { width: number; height: number };
  players: Player[];
  layout: { name: string; table: number[] };
  tracks: StateTrack[];
}

/** played, moved, changed or left. */
export interface RecognizerEvent {
  t: number;
  kind: string;
  text: string;
  printing_id: string | null;
  track: string;
  side: string;
}

/** A play the gate found, drawn until `until`. */
export interface Flash extends StateTrack {
  until: number;
}

/** A named card the finder lost, remembered for a while. */
export interface Ghost {
  t: number;
  card: string;
  x: number;
  y: number;
  name: string;
  printing_id: string | null;
  side: string;
  id: string;
}

/** Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first. */
export type Candidate = [card: string, p: number, score: number, row: number];

type Box4 = [number, number, number, number];

// --- tracks and boxes ---------------------------------------------------------------------------------------------

export class Track {
  id: string;
  box: CardBox;
  first: number;
  last: number;
  hits = 1;
  /** Identifications of this card so far. */
  reads = 0;
  /** Face-down looks in a row. */
  down = 0;
  /** card_id -> summed probability. */
  prob = new Map<string, number>();
  /** card_id -> (score, gallery row). */
  bestRow = new Map<string, [number, number]>();
  lastRead = -1e9;
  /** card_id once the track has been announced. */
  named: string | null = null;
  side = '';
  /** The named card's type (Unit, Rune, Legend, ...). */
  kind = '';
  /** A legend or battlefield: kept where it is all game. */
  pinned = false;
  /** Seen out of a hand and still since (`handShare`), from freeAt. */
  freeSince: number | null = null;
  freeAt: [number, number] | null = null;
  /** Out of a hand and still HAND_FREE_S once: on the table. */
  placed = false;

  constructor(id: string, box: CardBox, first: number, last: number, side = '') {
    this.id = id;
    this.box = box;
    this.first = first;
    this.last = last;
    this.side = side;
  }

  top(): [string, number][] {
    if (!this.reads) return [];
    // sorted by -p: stable, so equal probabilities keep the order the cards were first read in
    return [...this.prob].map(([c, p]): [string, number] => [c, p / this.reads]).sort((a, b) => b[1] - a[1]);
  }
}

/** The rotated rectangle's corners in frame px. The detector's boxes hold numpy floats in Python (detector_boxes
 * averages a numpy quad), so round() here is numpy's, rint(x * 10) / 10, not Python's. */
export function quad(box: CardBox): [number, number][] {
  const [cx, cy] = box.centre;
  const a = box.angle_deg * DEG_TO_RAD;
  const ux = (Math.cos(a) * box.long_px) / 2;
  const uy = (Math.sin(a) * box.long_px) / 2;
  const vx = (-Math.sin(a) * box.short_px) / 2;
  const vy = (Math.cos(a) * box.short_px) / 2;
  return [
    [npRound(cx + ux + vx, 1), npRound(cy + uy + vy, 1)],
    [npRound(cx + ux - vx, 1), npRound(cy + uy - vy, 1)],
    [npRound(cx - ux - vx, 1), npRound(cy - uy - vy, 1)],
    [npRound(cx - ux + vx, 1), npRound(cy - uy + vy, 1)],
  ];
}

export function aabb(box: CardBox): Box4 {
  const q = quad(box);
  const xs = q.map((p) => p[0]);
  const ys = q.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

/** Out of sight: not seen for more than a second. */
/** The point lies on the card. */
export function onBox(box: CardBox, x: number, y: number): boolean {
  const a = box.angle_deg * (Math.PI / 180);
  const ux = Math.cos(a);
  const uy = Math.sin(a);
  const dx = x - box.centre[0];
  const dy = y - box.centre[1];
  return Math.abs(dx * ux + dy * uy) <= box.long_px / 2 && Math.abs(dy * ux - dx * uy) <= box.short_px / 2;
}

/** A picture of whole numbers, three to a pixel, rows top to bottom: the still table (`StillTable.bg`). */
export interface StillPicture {
  width: number;
  height: number;
  data: ArrayLike<number>;
}

/** The share of skin-coloured points in a band around the box, inside the table window and off the other cards
 * (`others`): the fingers holding a card in a hand over the table, or putting it down. No card's art is looked at:
 * gold, faces and fire are skin-coloured too. Hemmed in by other cards, it sees no hand. With `still`, the still table
 * (`StillTable`), and `now`, this frame at its size (HAND_BG), a point counts only where the frame there is HAND_DIFF
 * unlike the table: on a wooden table the wood is skin-coloured too, and a hand is what is not the table (compared at
 * the same size, so a mat's thin printed lines are the table too). */
export function handShare(frame: RgbImage, box: CardBox, table: readonly [number, number, number, number], others: readonly CardBox[] = [],
  still: StillPicture | null = null, now: StillPicture | null = null): number {
  const [x0, y0, x1, y1] = table;
  const { width: w, height: h, data } = frame;
  const a = box.angle_deg * (Math.PI / 180);
  const ux = Math.cos(a);
  const uy = Math.sin(a);
  const [cx, cy] = box.centre;
  const near = others.filter((o) => dist(o.centre, box.centre) < o.long_px + box.long_px);
  const px: number[] = [];
  const at: number[] = []; // (yi, xi) of each point
  for (const f of HAND_RING) {
    const d = f * box.short_px;
    const hl = box.long_px / 2 + d;
    const hs = box.short_px / 2 + d;
    for (let k = 0; k < HAND_POINTS; k++) {
      const alongL = ((k + 0.5) / HAND_POINTS) * 2 * hl - hl;
      const alongS = ((k + 0.5) / HAND_POINTS) * 2 * hs - hs;
      for (const [su, sv] of [[alongL, hs], [alongL, -hs], [hl, alongS], [-hl, alongS]] as const) {
        const x = cx + su * ux - sv * uy;
        const y = cy + su * uy + sv * ux;
        const xi = Math.floor(x);
        const yi = Math.floor(y);
        if (xi >= Math.max(x0, 0) && xi < Math.min(x1, w) && yi >= Math.max(y0, 0) && yi < Math.min(y1, h) && !near.some((o) => onBox(o, x, y))) {
          const o = (yi * w + xi) * 3;
          px.push(data[o]!, data[o + 1]!, data[o + 2]!);
          at.push(yi, xi);
        }
      }
    }
  }
  const n = px.length / 3;
  if (n < HAND_MIN_POINTS) return 0;
  const m = skin({ width: n, height: 1, data: Uint8Array.from(px) }).data;
  let k = 0;
  for (let i = 0; i < n; i++) {
    if (!m[i]) continue;
    if (still !== null && now !== null) {
      const by = Math.min(still.height - 1, Math.floor((at[2 * i]! * still.height) / h));
      const bx = Math.min(still.width - 1, Math.floor((at[2 * i + 1]! * still.width) / w));
      const o = (by * still.width + bx) * 3;
      const d = Math.max(Math.abs(now.data[o]! - still.data[o]!), Math.abs(now.data[o + 1]! - still.data[o + 1]!), Math.abs(now.data[o + 2]! - still.data[o + 2]!));
      if (d < HAND_DIFF) continue; // the table itself, however skin-coloured
    }
    k++;
  }
  return k / n;
}

/** The table camera's picture without the hands over it, for the hand rule (`handShare`): HAND_BG_FRAMES table frames
 * HAND_BG_EVERY apart, their median per pixel (hands move, the table does not), then every HAND_BG_EVERY a step of
 * HAND_BG_STEP nearer the frame: a hand passing over the table for a few seconds stays unlike it, a card put down becomes
 * part of it within half a minute. Whole numbers throughout, as pipeline.py's are. Fed the frames at its size (`small`). */
export class StillTable {
  first: Uint8Array[] = [];
  bg: StillPicture | null = null;
  last = -1e9;

  /** A frame at the still table's size. */
  static small(frame: RgbImage): RgbImage {
    return image.resize(frame, HAND_BG, 'box');
  }

  feed(t: number, small: RgbImage): void {
    if (t - this.last < HAND_BG_EVERY) return;
    this.last = t;
    const x = small.data;
    if (this.bg === null) {
      this.first.push(Uint8Array.from(x));
      if (this.first.length === HAND_BG_FRAMES) {
        const bg = new Int16Array(x.length);
        const v = new Array<number>(HAND_BG_FRAMES);
        for (let i = 0; i < x.length; i++) {
          for (let f = 0; f < HAND_BG_FRAMES; f++) v[f] = this.first[f]![i]!;
          v.sort((a, b) => a - b);
          bg[i] = v[HAND_BG_FRAMES >> 1]!;
        }
        this.bg = { width: HAND_BG[0], height: HAND_BG[1], data: bg };
        this.first = [];
      }
      return;
    }
    const bg = this.bg.data as Int16Array;
    for (let i = 0; i < x.length; i++) {
      const d = x[i]! - bg[i]!;
      bg[i] = bg[i]! + (d > HAND_BG_STEP ? HAND_BG_STEP : d < -HAND_BG_STEP ? -HAND_BG_STEP : d);
    }
  }
}

/** A box of the mat's own colour inside as around it, and plain inside: a zone printed on the mat (outlined, a card's
 * size) or the mat's logo, which the detector outlines like a card. A card's face is never plain, and a face-down card is
 * plain in its sleeve's colour, not the mat's. Inside: 5 x 5 points over the middle 60% of the box; around: the band
 * HAND_RING[1] out from its edges. Their medians, and the inside's quartiles, of whole numbers. */
export function plainZone(frame: RgbImage, box: CardBox): boolean {
  const { width: w, height: h, data } = frame;
  const a = box.angle_deg * (Math.PI / 180);
  const ux = Math.cos(a);
  const uy = Math.sin(a);
  const [cx, cy] = box.centre;
  const inside: number[][] = [[], [], []];
  const ring: number[][] = [[], [], []];
  const take = (to: number[][], xi: number, yi: number): void => {
    if (xi < 0 || xi >= w || yi < 0 || yi >= h) return;
    const o = (yi * w + xi) * 3;
    for (let c = 0; c < 3; c++) to[c]!.push(data[o + c]!);
  };
  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
      const su = (i - 2) * 0.15 * box.long_px;
      const sv = (j - 2) * 0.15 * box.short_px;
      take(inside, Math.floor(cx + su * ux - sv * uy), Math.floor(cy + su * uy + sv * ux));
    }
  }
  const d = HAND_RING[1] * box.short_px;
  const hl = box.long_px / 2 + d;
  const hs = box.short_px / 2 + d;
  for (let k = 0; k < HAND_POINTS; k++) {
    const alongL = ((k + 0.5) / HAND_POINTS) * 2 * hl - hl;
    const alongS = ((k + 0.5) / HAND_POINTS) * 2 * hs - hs;
    for (const [su, sv] of [[alongL, hs], [alongL, -hs], [hl, alongS], [-hl, alongS]] as const) {
      take(ring, Math.floor(cx + su * ux - sv * uy), Math.floor(cy + su * uy + sv * ux));
    }
  }
  const n = ring[0]!.length;
  if (inside[0]!.length < 25 || n < 2 * HAND_POINTS) return false;
  let near = 0;
  let spread = 0;
  for (let c = 0; c < 3; c++) {
    const si = inside[c]!.sort((p, q) => p - q);
    const sr = ring[c]!.sort((p, q) => p - q);
    near = Math.max(near, Math.abs(si[12]! - (sr[(n - 1) >> 1]! + sr[n >> 1]!) / 2));
    spread = Math.max(spread, si[18]! - si[6]!);
  }
  return near < PLAIN_TOL && spread <= PLAIN_SPREAD;
}

/** Out of a hand and still for HAND_FREE_S: a card on the table, not one held over it. */
export function putDown(t: number, tr: Track): boolean {
  return tr.freeSince !== null && t - tr.freeSince >= HAND_FREE_S;
}

export function hiddenNow(t: number, tr: Track): boolean {
  return t - tr.last > 1.0;
}

/** The card's long side runs up the picture rather than across it. */
export function upright(box: CardBox): boolean {
  const a = pyMod(box.angle_deg, 180);
  return a >= 45 && a < 135;
}

/** The new box eased from the old one while the card barely moves: the detector's boxes jitter by a few pixels
 * from frame to frame. A card that moves or turns (exhausted) is followed at once. */
export function smooth(old: CardBox, newBox: CardBox, k = 0.35): CardBox {
  const da = pyMod(newBox.angle_deg - old.angle_deg + 90, 180) - 90;
  if (dist(old.centre, newBox.centre) > 0.1 * newBox.long_px || Math.abs(da) > 10) return newBox;
  return {
    centre: [(1 - k) * old.centre[0] + k * newBox.centre[0], (1 - k) * old.centre[1] + k * newBox.centre[1]],
    long_px: (1 - k) * old.long_px + k * newBox.long_px,
    short_px: (1 - k) * old.short_px + k * newBox.short_px,
    angle_deg: pyMod(old.angle_deg + k * da, 180),
    fill: newBox.fill,
    back: newBox.back ?? false,
  };
}

/** math.hypot(x, y), as CPython computes it: the same vector_norm as math.dist. */
export function hypot(x: number, y: number): number {
  return dist([x, y], [0, 0]);
}

/** The card upright, long side vertical. Only the card's neighbourhood is rotated, not the frame. */
export function cardCrop(frame: RgbImage, box: CardBox): RgbImage {
  const [cx, cy] = box.centre;
  const r = Math.ceil(hypot(box.long_px, box.short_px) / 2) + 2;
  const x0 = Math.trunc(cx) - r;
  const y0 = Math.trunc(cy) - r;
  const local = image.crop(frame, [x0, y0, x0 + 2 * r, y0 + 2 * r]);
  const lx = cx - x0;
  const ly = cy - y0;
  const w = box.short_px;
  const h = box.long_px;
  // local.rotate(...).crop(...) in one: rotateCrop computes only the pixels the crop keeps, the same bytes
  return image.rotateCrop(local, box.angle_deg - 90, { resample: 'bicubic', center: [lx, ly] }, [
    pyRound(lx - w / 2),
    pyRound(ly - h / 2),
    pyRound(lx + w / 2),
    pyRound(ly + h / 2),
  ]);
}

/** The scale, rotation and shift that take the points `src` onto `dst`, least squares (Umeyama). In two dimensions
 * the rotation of U diag(1, sign det(U Vt)) Vt is the turn by atan2(m10 - m01, m00 + m11) of the cross-covariance m,
 * and the singular values summed with those signs are the length of (m00 + m11, m10 - m01), so no SVD is needed. */
export function similarity(
  src: readonly (readonly [number, number])[],
  dst: readonly (readonly [number, number])[],
): [number, [[number, number], [number, number]], [number, number]] {
  const n = src.length;
  const ms = meanOf(src.map((p): [number, number] => [p[0], p[1]]));
  const md = meanOf(dst.map((p): [number, number] => [p[0], p[1]]));
  const a = src.map((p) => [p[0] - ms[0], p[1] - ms[1]] as const);
  const b = dst.map((p) => [p[0] - md[0], p[1] - md[1]] as const);
  let m00 = 0; // (b.T @ a)[0, 0]
  let m01 = 0;
  let m10 = 0;
  let m11 = 0;
  for (let i = 0; i < n; i++) {
    m00 += b[i]![0] * a[i]![0];
    m01 += b[i]![0] * a[i]![1];
    m10 += b[i]![1] * a[i]![0];
    m11 += b[i]![1] * a[i]![1];
  }
  m00 /= n;
  m01 /= n;
  m10 /= n;
  m11 /= n;
  const turn = Math.atan2(m10 - m01, m00 + m11);
  const c = Math.cos(turn);
  const s = Math.sin(turn);
  const rot: [[number, number], [number, number]] = [
    [c, -s],
    [s, c],
  ];
  const squares: number[] = [];
  for (const [x, y] of a) squares.push(x * x, y * y);
  const scale = hypot(m00 + m11, m10 - m01) / Math.max(1e-9, pairwiseSum(squares) / n);
  const shift: [number, number] = [
    md[0] - (scale * rot[0][0] * ms[0] + scale * rot[0][1] * ms[1]),
    md[1] - (scale * rot[1][0] * ms[0] + scale * rot[1][1] * ms[1]),
  ];
  return [scale, rot, shift];
}

/** numpy's sum of a float64 vector: pairwise, eight running sums in blocks of up to 128 (0 + the pairwise sum). */
export function pairwiseSum(a: ArrayLike<number>, lo = 0, n = a.length - lo): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[lo + i]!;
    return res;
  }
  if (n <= 128) {
    const r = [a[lo]!, a[lo + 1]!, a[lo + 2]!, a[lo + 3]!, a[lo + 4]!, a[lo + 5]!, a[lo + 6]!, a[lo + 7]!];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = r[j]! + a[lo + i + j]!;
    let res = r[0]! + r[1]! + (r[2]! + r[3]!) + (r[4]! + r[5]! + (r[6]! + r[7]!));
    for (; i < n; i++) res += a[lo + i]!;
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}

/** The same for a float32 vector, in float32. */
export function pairwiseSum32(a: ArrayLike<number>, lo = 0, n = a.length - lo): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f32(res + a[lo + i]!);
    return res;
  }
  if (n <= 128) {
    const r = new Float32Array(8);
    for (let j = 0; j < 8; j++) r[j] = a[lo + j]!;
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = r[j]! + a[lo + i + j]!;
    let res = f32(f32(f32(r[0]! + r[1]!) + f32(r[2]! + r[3]!)) + f32(f32(r[4]! + r[5]!) + f32(r[6]! + r[7]!)));
    for (; i < n; i++) res = f32(res + a[lo + i]!);
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return f32(pairwiseSum32(a, lo, n2) + pairwiseSum32(a, lo + n2, n - n2));
}

/** x.mean() of a float32 vector. */
function mean32(x: Float32Array): number {
  return f32(f32(0 + pairwiseSum32(x)) / x.length);
}

/** x.std() of a float32 vector (numpy's _var: the mean, the squared deviations, their mean, its root). */
function std32(x: Float32Array): number {
  const m = mean32(x);
  const d = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = f32(x[i]! - m);
    d[i] = v * v;
  }
  return f32(Math.sqrt(f32(f32(0 + pairwiseSum32(d)) / x.length)));
}

/** np.quantile(x, q) of float32 values (the default, linear): q, the index and the interpolation are float32, as
 * numpy keeps a float32 array's quantile float32. */
function quantile32(x: Float32Array, q: number): number {
  const s = Float32Array.from(x).sort();
  const n = s.length;
  const virtual = f32((n - 1) * f32(q));
  let prev = Math.floor(virtual);
  let next = prev + 1;
  if (virtual >= n - 1) prev = next = n - 1;
  if (virtual < 0) prev = next = 0;
  const gamma = f32(virtual - prev);
  const a = s[prev]!;
  const b = s[next]!;
  const diff = f32(b - a);
  return gamma >= 0.5 ? f32(b - f32(diff * f32(1 - gamma))) : f32(a + f32(diff * gamma));
}

/** The dot product of two float32 vectors, rounded to float32 (np.linalg.norm's BLAS sdot sums in its own order; this
 * is within a float32 step or two of it). */
function dot32(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return f32(s);
}

/** The correlation of a frame's thumbnail with the scene's mean at the pixels `idx`, in float32 as numpy works it out,
 * or null when the mean has no pattern there. */
function correlation(x: Float32Array, mean: Float32Array, idx: readonly number[]): number | null {
  const a = new Float32Array(idx.length * 3);
  const b = new Float32Array(idx.length * 3);
  idx.forEach((p, i) => {
    for (let c = 0; c < 3; c++) {
      a[3 * i + c] = x[3 * p + c]!;
      b[3 * i + c] = mean[3 * p + c]!;
    }
  });
  if (std32(b) < 8) return null;
  const am = mean32(a);
  const bm = mean32(b);
  const ab = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    a[i] = a[i]! - am;
    b[i] = b[i]! - bm;
    ab[i] = a[i]! * b[i]!;
  }
  const norms = f32(f32(Math.sqrt(dot32(a, a))) * f32(Math.sqrt(dot32(b, b))));
  return f32(f32(0 + pairwiseSum32(ab)) / f32(norms + f32(1e-6)));
}

// --- the scene ----------------------------------------------------------------------------------------------------

const SMALL_W = 96;
const SMALL_H = 54;

/** Whether a frame shows the table camera, on any broadcast. The other shots (player cams, a wide shot of the stage,
 * a title card) show players' hands and the cards they hold: hidden information, never processed (D-005), and
 * nothing on the board changes while they are on.
 *
 * It learns the table camera from the footage itself: the parts of a 96 x 54 thumbnail that stay put while the board
 * changes (the broadcast's overlay, the mat's edges and print) and their colours. A frame is the table camera when
 * those parts match; a cut replaces them, play does not (the M0 final: table frames score 0.5 to 0.8, other shots
 * about 0). To start, a frame is taken as the table camera when the mat fills the table window as it does there (a
 * layout that knows its mat colour) or when at least five cards lie in it; if the view stays unrecognised but looks
 * like a table again for a few seconds, with five cards of the table's size in it (the camera itself moved), it learns
 * again. A mat's colour alone is no proof: on a co-stream every shot, a player in a maroon shirt included, had it. The
 * thumbnail is too coarse to show any card. Its numbers are float32, as numpy's are.
 *
 * Only the table window is scored: the panels beside it are laid over every shot, a close-up of a hand included. It
 * is scored in blocks (SCENE_GRID), and the score is their mean: an arm or a banner over the table spoils a block or
 * two, where a cut to another shot spoils them all.
 *
 * What stays put through the cuts is not the table camera's at all: a co-streamer's webcam and chat, a scoreboard, a
 * sponsor banner, laid over every shot (`overlay`). Kept in the score, it makes every shot look like the table, so it is
 * left out once known; and a frame is learnt only when the one before was the table camera too, so the first frame
 * after a cut, another shot that happens to look alike, never teaches the scene what the table is. */
export class Scene {
  layout: Layout;
  corr: number;
  learnEvery: number;
  relearnAfter: number;
  n = 0;
  mean: Float32Array | null = null;
  var: Float32Array | null = null;
  lastLearn = -1e9;
  awaySince: number | null = null;
  /** Table-like frames in a row while away (checked every learnEvery). */
  looks = 0;
  lastLook = -1e9;
  /** The frame before was the table camera. */
  wasOn = false;
  /** The frame before's thumbnail. */
  prev: Float32Array | null = null;
  /** Cuts each thumbnail pixel stayed through, the board's own. */
  same = new Int32Array(SMALL_W * SMALL_H);
  cuts = 0;
  /** ... and with those seen before the board (`Recognizer.prime`). */
  sameAll = new Int32Array(SMALL_W * SMALL_H);
  cutsAll = 0;
  /** SMALL_H x SMALL_W, 1 where the overlay is, once OVERLAY_CUTS cuts are seen. */
  overlay: Uint8Array | null = null;
  /** How often it was worked out: the board drops what lies in it then. */
  overlayN = 0;
  /** SMALL_H x SMALL_W, 1 where what stayed through every cut so far is, before then. */
  suspect: Uint8Array | null = null;
  /** When the last cut was (the cuts seen before the board, when it began). */
  cutT: number | null = null;
  /** (t, card-sized boxes) on the table camera, in its last minute on screen (the board tells it). */
  cardsSeen: [number, number][] = [];

  /** SMALL_H x SMALL_W, 1 in the table window: the only pixels scored. */
  window: Uint8Array;

  constructor(layout: Layout, corr = 0.45, learnEvery = 2.0, relearnAfter = 20.0) {
    this.layout = layout;
    this.corr = corr;
    this.learnEvery = learnEvery;
    this.relearnAfter = relearnAfter;
    const [tx0, ty0, tx1, ty1] = layout.table;
    this.window = new Uint8Array(SMALL_W * SMALL_H);
    for (let r = Math.floor(ty0 * SMALL_H); r < Math.min(SMALL_H, Math.ceil(ty1 * SMALL_H)); r++)
      for (let c = Math.floor(tx0 * SMALL_W); c < Math.min(SMALL_W, Math.ceil(tx1 * SMALL_W)); c++) this.window[r * SMALL_W + c] = 1;
  }

  static small(frame: RgbImage): Float32Array {
    return Float32Array.from(image.resize(frame, [SMALL_W, SMALL_H], 'box').data);
  }

  /** A cut, when half the thumbnail changed from the frame before (play changes a quarter at most): every pixel that
   * stayed counts once more as overlay, and the overlay is worked out again. A cut seen `before` the board (the frames the
   * table was looked for in) makes what stayed suspect, never overlay: a scoreboard that comes with the table camera did
   * not stay through the cut from a player cam to it, and is overlay all the same. */
  see(x: Float32Array, before = false): void {
    if (this.prev !== null) {
      const px = SMALL_W * SMALL_H;
      const moved = new Uint8Array(px);
      let n = 0;
      for (let p = 0; p < px; p++) {
        const d = Math.max(Math.abs(x[3 * p]! - this.prev[3 * p]!), Math.abs(x[3 * p + 1]! - this.prev[3 * p + 1]!), Math.abs(x[3 * p + 2]! - this.prev[3 * p + 2]!));
        if (d >= OVERLAY_TOL) {
          moved[p] = 1;
          n++;
        }
      }
      if (n * 2 >= px) {
        for (let p = 0; p < px; p++) if (!moved[p]) this.sameAll[p] = this.sameAll[p]! + 1;
        this.cutsAll += 1;
        if (!before) {
          for (let p = 0; p < px; p++) if (!moved[p]) this.same[p] = this.same[p]! + 1;
          this.cuts += 1;
        }
        this.cutT = null; // stamped by onTable, with the time
        if (this.cuts >= OVERLAY_CUTS) {
          this.overlay = overlayPatches(this.same, this.cuts, this.overlay);
          this.overlayN += 1;
          this.suspect = null;
        } else {
          this.suspect = overlayPatches(this.sameAll, this.cutsAll);
        }
      }
    }
    this.prev = x;
  }

  /** The frame point (x, y) lies where the overlay may be, before the cuts have shown it. */
  inSuspect(x: number, y: number, w: number, h: number): boolean {
    if (this.suspect === null) return false;
    const r = Math.min(SMALL_H - 1, Math.max(0, Math.floor((y * SMALL_H) / h)));
    const c = Math.min(SMALL_W - 1, Math.max(0, Math.floor((x * SMALL_W) / w)));
    return this.suspect[r * SMALL_W + c] === 1;
  }

  /** The frame point (x, y) lies in the overlay. */
  inOverlay(x: number, y: number, w: number, h: number): boolean {
    if (this.overlay === null) return false;
    const r = Math.min(SMALL_H - 1, Math.max(0, Math.floor((y * SMALL_H) / h)));
    const c = Math.min(SMALL_W - 1, Math.max(0, Math.floor((x * SMALL_W) / w)));
    return this.overlay[r * SMALL_W + c] === 1;
  }

  async tableLike(frame: RgbImage, count: () => Promise<number>): Promise<boolean> {
    const mat = this.layout.mat;
    if (mat !== null) {
      const [x0, y0, x1, y1] = layoutBox(this.layout, frame.width, frame.height);
      const a = image.resize(image.crop(frame, [x0, y0, x1, y1]), [160, 120], 'box').data;
      let n = 0;
      for (let o = 0; o < a.length; o += 3) {
        const d = Math.max(Math.abs(a[o]! - mat[0]), Math.abs(a[o + 1]! - mat[1]), Math.abs(a[o + 2]! - mat[2]));
        if (d < this.layout.mat_tol) n++;
      }
      return n / (a.length / 3) >= this.layout.mat_share;
    }
    return (await count()) >= 5;
  }

  /** The board found `n` card-sized boxes on a frame of the table camera at `t`. */
  sawCards(t: number, n: number): void {
    this.cardsSeen = [...this.cardsSeen.filter(([tt]) => t - tt < CARDS_S), [t, n]];
  }

  /** A view to learn as the table camera again: table-like, with as many cards of the table's size in it as the table
   * camera showed in its last minute on screen (half the most, and at least RELEARN_CARDS). A close-up of one side of
   * the table, or of a hand over it, shows a few of its cards. */
  async tableAgain(frame: RgbImage, count: () => Promise<number>): Promise<boolean> {
    if (!(await this.tableLike(frame, count))) return false;
    const n = await count();
    return n >= RELEARN_CARDS && n * 2 >= this.cardsSeen.reduce((m, [, c]) => Math.max(m, c), 0);
  }

  /** How well the frame's still parts in the table window match the table camera's: the mean of their correlations
   * block by block (SCENE_GRID), or null when no block has a pattern to match (a plain mat): then the mat share or the
   * cards decide. The overlay laid over every shot is not the table camera's: it is left out. As pipeline.py's. */
  score(x: Float32Array): number | null {
    const mean = this.mean!;
    const v = this.var!;
    const px = x.length / 3;
    const std = new Float32Array(px);
    let steady = 0;
    for (let p = 0; p < px; p++) {
      std[p] = Math.max(f32(Math.sqrt(v[3 * p]!)), f32(Math.sqrt(v[3 * p + 1]!)), f32(Math.sqrt(v[3 * p + 2]!)));
      if (std[p]! < 12) steady++;
    }
    const cut = steady / px >= 0.1 ? null : quantile32(std, 0.3);
    const overlay = this.overlay;
    const still = new Uint8Array(px);
    for (let p = 0; p < px; p++) if ((cut === null ? std[p]! < 12 : std[p]! <= cut) && (overlay === null || !overlay[p]) && this.window[p]) still[p] = 1;
    const [gx, gy] = SCENE_GRID;
    const bw = Math.trunc(SMALL_W / gx);
    const bh = Math.trunc(SMALL_H / gy);
    let total = 0;
    let n = 0;
    for (let by = 0; by < gy; by++) {
      for (let bx = 0; bx < gx; bx++) {
        const idx: number[] = [];
        for (let r = by * bh; r < (by + 1) * bh; r++) for (let c = bx * bw; c < (bx + 1) * bw; c++) if (still[r * SMALL_W + c]) idx.push(r * SMALL_W + c);
        if (idx.length < SCENE_BLOCK) continue;
        const sc = correlation(x, mean, idx);
        if (sc === null) continue;
        total += sc;
        n += 1;
      }
    }
    return n ? total / n : null;
  }

  learn(t: number, x: Float32Array): void {
    if (t - this.lastLearn < this.learnEvery) return;
    this.lastLearn = t;
    this.n += 1;
    if (this.mean === null || this.var === null) {
      this.mean = Float32Array.from(x);
      this.var = new Float32Array(x.length).fill(400.0); // unsure at first: nothing counts as still
      return;
    }
    const k = Math.max(1.0 / this.n, 0.02); // the mean of the first fifty, then a slow drift (light, overlay updates)
    const kf = f32(k); // numpy takes the Python float as a float32
    const keep = f32(1 - k);
    for (let i = 0; i < x.length; i++) {
      const d = f32(x[i]! - this.mean[i]!);
      this.mean[i] = this.mean[i]! + f32(kf * d);
      this.var[i] = keep * f32(this.var[i]! + f32(f32(kf * d) * d));
    }
  }

  async onTable(t: number, frame: RgbImage, count: () => Promise<number> = async () => 0): Promise<boolean> {
    const x = Scene.small(frame);
    this.see(x);
    if (this.cutsAll && this.cutT === null) this.cutT = t; // a cut now, or the ones seen before the board began
    if (this.suspect !== null && t - this.cutT! > SUSPECT_S) this.suspect = null; // no cut for a minute: nothing there is held back any longer
    const sc = this.n >= 5 ? this.score(x) : null;
    let ok: boolean;
    if (sc === null) {
      ok = await this.tableLike(frame, count);
    } else {
      ok = sc >= this.corr;
      if (!ok && this.awaySince !== null && t - this.awaySince > this.relearnAfter && t - this.lastLook >= this.learnEvery) {
        this.lastLook = t;
        this.looks = (await this.tableAgain(frame, count)) ? this.looks + 1 : 0;
        if (this.looks >= 3) {
          // the table again, but not as it was learnt: the camera moved
          this.n = 0;
          this.mean = null;
          this.var = null;
          this.looks = 0;
          ok = true;
        }
      }
    }
    if (ok) {
      if (this.wasOn) this.learn(t, x); // not the first frame after a cut: another shot may look like the table for a frame
      this.awaySince = null;
    } else if (this.awaySince === null) {
      this.awaySince = t;
    }
    this.wasOn = ok;
    return ok;
  }
}

/** The overlay: the thumbnail pixels that stayed through OVERLAY_SHARE of the cuts, in patches reaching within
 * OVERLAY_EDGE pixels of the frame's edge (a broadcast lays its graphics along the edges; the mat lies inside), each
 * patch's holes filled (a webcam's frame stays, the face in it moves). The overlay worked out before, `was`, stays
 * where it stayed through OVERLAY_KEEP of the cuts. As pipeline.py's `overlay_patches`. */
export function overlayPatches(same: Int32Array, cuts: number, was: Uint8Array | null = null): Uint8Array {
  const w = SMALL_W;
  const h = SMALL_H;
  const need = cuts * Math.round(OVERLAY_SHARE * 10);
  const keepNeed = cuts * Math.round(OVERLAY_KEEP * 10);
  const ov = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const s = same[p]! * 10;
    ov[p] = s >= need || (was !== null && was[p] === 1 && s >= keepNeed) ? 1 : 0;
  }
  const { labels, count } = label({ width: w, height: h, data: ov });
  const objects = findObjects(labels, w, h, count);
  const keepIds = new Set<number>();
  objects.forEach((o, i) => {
    if (o === null || o === undefined) return;
    const [y0, y1, x0, x1] = o;
    if (y0 <= OVERLAY_EDGE || x0 <= OVERLAY_EDGE || y1 >= h - OVERLAY_EDGE || x1 >= w - OVERLAY_EDGE) keepIds.add(i + 1);
  });
  const keep = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) if (keepIds.has(labels[p]!)) keep[p] = 1;
  return binaryFillHoles({ width: w, height: h, data: keep }).data;
}

// --- the recogniser -----------------------------------------------------------------------------------------------

export interface RecognizerOptions {
  title?: string;
  minP?: number;
  sureP?: number;
  recheckS?: number;
  forgetS?: number;
  maxReads?: number;
  settleS?: number;
  /** Watch the whole table with the change gate. */
  gate?: boolean;
  fps?: number;
  gateP?: number;
  /** What finds the cards; without one, the bootstrap finder (isolated cards on the mat). */
  finder?: Finder | null;
  /** Turns an encoder's scores into how sure a read is; fitted per encoder. */
  temperature?: number;
  /** Hold each side's crops to its pinned legend (the legend rule, D-026); on by default. */
  legendRule?: boolean;
}

/** Holds the gallery and the table's tracks; `step` takes one frame and returns the state and events. `finder(t,
 * image)` replaces the bootstrap finder, e.g. with the trained detector (detector_boxes). */
/** Degrees between two boxes' long sides (a box's angle is modulo 180). */
export function turnApart(a: number, b: number): number {
  return Math.abs((((a - b + 90) % 180) + 180) % 180 - 90);
}

/** The boxes in groups, each linked box closer than `reach` to another of its group. */
export function stacksOf(boxes: readonly CardBox[], reach: number): CardBox[][] {
  const parent = boxes.map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]!]!;
      i = parent[i]!;
    }
    return i;
  };
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) if (dist(boxes[i]!.centre, boxes[j]!.centre) <= reach) parent[find(i)] = find(j);
  const groups = new Map<number, CardBox[]>();
  boxes.forEach((b, i) => {
    const r = find(i);
    const g = groups.get(r);
    if (g) g.push(b);
    else groups.set(r, [b]);
  });
  return [...groups.values()];
}

/** Runes a column or fan hides from the detector: along the stack, its usual step from strip to strip (the gaps of
 * RUNE_STEP cards), and each gap of RUNE_GAP steps or more holding about gap / step - 1 more. Cards are one size. */
export function hiddenRunes(stack: readonly CardBox[], card: number): number {
  if (stack.length < 3) return 0;
  let a = stack[0]!, b = stack[0]!, span = -1;
  for (const x of stack)
    for (const y of stack) {
      const d = dist(x.centre, y.centre);
      if (d > span) [a, b, span] = [x, y, d];
    }
  if (span < 1e-6) return 0;
  const ux = (b.centre[0] - a.centre[0]) / span, uy = (b.centre[1] - a.centre[1]) / span;
  const at = stack.map((o) => (o.centre[0] - a.centre[0]) * ux + (o.centre[1] - a.centre[1]) * uy).sort((p, q) => p - q);
  const gaps = at.slice(1).map((y, i) => y - at[i]!);
  const steps = gaps.filter((g) => RUNE_STEP[0] * card <= g && g <= RUNE_STEP[1] * card).sort((p, q) => p - q);
  if (!steps.length) return 0;
  const step = steps[Math.floor(steps.length / 2)]!;
  return gaps.filter((g) => g >= RUNE_GAP * step).reduce((n, g) => n + Math.max(0, Math.floor(g / step + 0.5) - 1), 0);
}

export class Recognizer {
  layout: Layout;
  rows: CatalogRow[];
  enc: Encoder;
  gallery: Pyramid;
  temperature: number;
  cards: string[];
  firstRow = new Map<string, number>();
  rowOf = new Map<string, CatalogRow>();
  /** card_id -> the card's type (Unit, Rune, ...) */
  typeOf = new Map<string, string>();
  title: string;
  minP: number;
  sureP: number;
  recheckS: number;
  forgetS: number;
  maxReads: number;
  settleS: number;
  tracks = new Map<string, Track>();
  nextId = 0;
  t0: number | null = null;
  mat: [number, number, number] | null = null;
  matT = -1e9;
  timing: Record<string, number> = {};
  gate: ChangeGate | null;
  gateP: number;
  finder: Finder | null;
  /** (read at, frame box) */
  pending: [number, Box4][] = [];
  /** Plays the gate found, drawn for a few seconds. */
  flashes: Flash[] = [];
  /** (t, card, x, y) of recent plays, for de-duplication. */
  plays: [number, string, number, number][] = [];
  // Named cards the finder lost (a hand over them, a card touching them): remembered for a while, so the same card
  // found again at the same spot is not played twice. A card leaves the table when the gate sees its spot change,
  // not when the finder loses it.
  ghosts: Ghost[] = [];
  // A legend never changes during a game (M0 section 5.4): once one is named on a side, that player keeps it.
  // ponytail: the first confident legend wins for the whole run; reset per game once games are detected
  legends = new Map<string, Legend>();
  /** The legend rule: a side's crops compete only with the rows its pinned legend allows (`allowed`). */
  legendRule: boolean;
  /** Legend card_id -> the gallery rows its side's crops compete with (1 for each allowed row). */
  masks = new Map<string, Uint8Array>();
  /** priors.tokenRows, once a legend needs it. */
  tokens: Uint8Array | null = null;
  /** The decklists the viewer gave (`setLists`): a side whose pinned legend a list names competes with that list's cards. */
  decks: Deck[] = [];
  private cat: Catalogue | null = null;
  /** Track id -> its box's extent, this frame. */
  boxesNow = new Map<string, Box4>();
  /** The frame's size, for where a card lies on the table. */
  frameWh: [number, number] = [1920, 1080];
  /** Card_id -> its gallery rows (1 for each), for a named card read again. */
  cardRows = new Map<string, Uint8Array>();
  /** Side -> [t, runes, exhausted] of the recent frames. */
  runeCounts = new Map<string, [number, number, number][]>();
  // Camera cuts: frames off the table camera are skipped and the board's clocks stop (`pause`). After a cut the view
  // may be framed differently, so tracks found again by name re-anchor the rest (`cut`).
  scene: Scene;
  gateSettings: GateSettings | null;
  lastT: number | null = null;
  away = false;
  cutAt: number | null = null;
  /** Track id -> its box before the cut. */
  anchorBase = new Map<string, CardBox>();
  /** (before, after) centres. */
  anchorPairs: [[number, number], [number, number]][] = [];
  /** Confirmed tracks the last frame matched. */
  prevSeen = new Set<string>();
  /** ... the last frame before a cut away. */
  beforeAway = new Set<string>();
  /** The table without the hands over it, for the hand rule. */
  still = new StillTable();
  /** The scene's overlay as last swept off the board (`dropOverlaid`). */
  overlayN = 0;
  /** When the cards first named lately were, for BURST. */
  firstNamed: number[] = [];
  /** The plays announced, by track (or flash) id: where they were, so a play read off the overlay before the cuts showed
   * it can be withdrawn (`dropOverlaid`). */
  announced = new Map<string, [x: number, y: number, name: string, printingId: string | null, side: string]>();

  constructor(layout: Layout, rows: readonly CatalogRow[], encoder: Encoder, gallery: Pyramid, opts: RecognizerOptions = {}) {
    const { title = '', minP = 0.5, sureP = 0.85, recheckS = 8.0, forgetS = 4.0, maxReads = 12, settleS = 3.0 } = opts;
    const { gate = true, fps = 5.0, gateP = 0.7, finder = null, temperature = TEMPERATURE, legendRule = true } = opts;
    this.layout = layout;
    this.rows = [...rows];
    this.enc = encoder;
    this.gallery = gallery;
    this.temperature = temperature; // turns an encoder's scores into how sure a read is; fitted per encoder
    this.cards = this.rows.map((r) => r.card_id);
    this.rows.forEach((r, i) => {
      if (!this.firstRow.has(r.card_id)) this.firstRow.set(r.card_id, i);
    });
    for (const r of this.rows) this.rowOf.set(r.printing_id, r);
    for (const r of this.rows) this.typeOf.set(r.card_id, r.type);
    this.title = title || layout.title;
    this.minP = minP;
    this.sureP = sureP;
    this.recheckS = recheckS;
    this.forgetS = forgetS;
    this.maxReads = maxReads;
    this.settleS = settleS;
    this.gate = gate ? new ChangeGate(gateSettings({ fps, card_long_frac: layout.card_long_1080 / 1080 })) : null;
    this.gateP = gateP;
    this.finder = finder;
    this.scene = new Scene(layout);
    this.gateSettings = gate ? gateSettings({ fps, card_long_frac: layout.card_long_1080 / 1080 }) : null;
    this.legendRule = legendRule;
  }

  // --- finding ------------------------------------------------------------------------------------------------------

  async find(t: number, rgb: RgbImage): Promise<CardBox[]> {
    if (this.finder !== null) return this.finder(t, rgb);
    const { width: w, height: h } = rgb;
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const roi = image.crop(rgb, [x0, y0, x1, y1]);
    let mask: Mask | undefined;
    if (this.layout.mask === 'notmat') {
      if (this.mat === null || t - this.matT > 30) {
        // the light drifts over a match
        this.mat = matColour(roi, 4);
        this.matT = t;
      }
      mask = notmatMask(roi, this.mat, this.layout.mat_tol);
    }
    const boxes = findCards(roi, cardPx(this.layout, h), mask === undefined ? {} : { mask });
    return boxes.map((b) => ({ ...b, centre: [b.centre[0] + x0, b.centre[1] + y0] }));
  }

  /** The frames seen before the board began, while the table was looked for (`autoLayout`), in order: a cut among them
   * (from a player cam to the table, most often) shows the scene some of the overlay from the start. */
  prime(frames: readonly RgbImage[]): void {
    for (const f of frames) this.scene.see(Scene.small(f), true);
  }

  /** The finder's boxes that can be cards: no longer than SIZE_MAX cards, off the overlay laid over every shot
   * (`Scene.overlay`), and not a zone printed on the mat (`plainZone`). */
  keep(boxes: readonly CardBox[], frame: RgbImage, w: number, h: number): CardBox[] {
    const px = cardPx(this.layout, h);
    return boxes.filter((b) => b.long_px <= SIZE_MAX * px && !this.scene.inOverlay(b.centre[0], b.centre[1], w, h) && !plainZone(frame, b));
  }

  /** What lies in the overlay, now that the cuts have shown it: the cards read off a webcam or a banner before then go,
   * a legend one of them gave its side with it, and the plays they made are withdrawn. */
  dropOverlaid(t: number, w: number, h: number): RecognizerEvent[] {
    this.overlayN = this.scene.overlayN;
    const events: RecognizerEvent[] = [];
    for (const [key, [x, y, name, pid, side]] of [...this.announced]) {
      if (!this.scene.inOverlay(x, y, w, h)) continue;
      this.announced.delete(key);
      events.push({ t: pyRound(t, 2), kind: 'withdrawn', text: `${name} withdrawn: it was the stream's overlay`, printing_id: pid, track: key, side });
    }
    for (const tr of [...this.tracks.values()]) {
      if (!this.scene.inOverlay(tr.box.centre[0], tr.box.centre[1], w, h)) continue;
      this.tracks.delete(tr.id);
      const lg = this.legends.get(tr.side);
      if (lg !== undefined && tr.named !== null && this.rowOf.get(lg.printing_id)?.card_id === tr.named) this.legends.delete(tr.side);
    }
    this.ghosts = this.ghosts.filter((gh) => !this.scene.inOverlay(gh.x, gh.y, w, h));
    return events;
  }

  // --- tracking -----------------------------------------------------------------------------------------------------

  /** Boxes to tracks one to one at the least total distance (Hungarian assignment). A box continues a track of about
   * its size within a third of a card, in view or out of sight for a while, so a card found again where it was keeps
   * its id and name; any other box starts a new track. */
  match(t: number, found: readonly CardBox[], w: number, h: number): Track[] {
    const boxes = found.filter((b) => !this.held(b, w, h));
    const tracks = [...this.tracks.values()];
    const pairs = new Map<number, Track>();
    if (tracks.length && boxes.length) {
      const cost = new Float64Array(boxes.length * tracks.length).fill(1e6);
      boxes.forEach((b, i) => {
        tracks.forEach((tr, j) => {
          const d = dist(tr.box.centre, b.centre) / b.long_px;
          if (d < 0.35 && Math.abs(tr.box.long_px / b.long_px - 1) < 0.25) {
            cost[i * tracks.length + j] = d + (t - tr.last > 1.0 ? 0.25 : 0.0); // the ones in view first
          }
        });
      });
      const [rows, cols] = linearSumAssignment({ rows: boxes.length, cols: tracks.length, data: cost });
      rows.forEach((i, n) => {
        const j = cols[n]!;
        if (cost[i * tracks.length + j]! < 1e6) pairs.set(i, tracks[j]!);
      });
    }
    const seen: Track[] = [];
    boxes.forEach((b, i) => {
      let tr = pairs.get(i);
      if (tr === undefined && this.onLegend(b)) return; // the detector's second outline of a legend's case, or of the die on it: not a card
      if (tr === undefined) {
        tr = new Track(`t${this.nextId}`, b, t, t, layoutSide(this.layout, b.centre[0], b.centre[1], w, h));
        this.nextId += 1;
        this.tracks.set(tr.id, tr);
      } else {
        tr.box = t - tr.last <= 1.0 ? smooth(tr.box, b) : b;
        tr.last = t;
        tr.hits = tr.hits + 1;
        if (!tr.pinned) tr.side = layoutSide(this.layout, tr.box.centre[0], tr.box.centre[1], w, h);
      }
      seen.push(tr);
    });
    return seen;
  }

  /** A card across the table window's edge on a player's side: in a hand held over the table, or on its way there, not
   * lying on it. Never tracked, so a player's hand is never read (D-005). (A card held just inside the edge looks like
   * the cards lying there, on some broadcasts a quarter of a card from it: not caught.) */
  held(box: CardBox, w: number, h: number): boolean {
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const [bx0, by0, bx1, by1] = aabb(box);
    const m = HELD_OUT * box.short_px;
    if (this.layout.split === 'horizontal') return by0 < y0 - m || by1 > y1 + m;
    return bx0 < x0 - m || bx1 > x1 + m;
  }

  /** The point lies in the battlefield strip: the band along the table's midline between the players, where the
   * battlefields lie and where either player's units go to fight over them. */
  inStrip(x: number, y: number): boolean {
    const [w, h] = this.frameWh;
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const u = this.layout.split === 'horizontal' ? (y - y0) / Math.max(1, y1 - y0) : (x - x0) / Math.max(1, x1 - x0);
    return Math.abs(u - 0.5) <= STRIP;
  }

  otherSide(side: string): string {
    const [a, b] = layoutSides(this.layout);
    return side === a ? b : a;
  }

  /** The box's centre lies well inside a named legend. Only dice and counters go on a legend; the champion and the
   * cards beside it lie about a card width away. */
  onLegend(box: CardBox): boolean {
    for (const o of this.tracks.values()) {
      if (o.pinned && o.kind === 'Legend' && dist(o.box.centre, box.centre) < 0.35 * o.box.long_px) return true;
    }
    return false;
  }

  /** The pinned legend of the track's side, when that is another track. */
  sideLegend(tr: Track): Track | null {
    for (const o of this.tracks.values()) if (o.pinned && o.kind === 'Legend' && o.side === tr.side && o !== tr) return o;
    return null;
  }

  /** The side already has its legend, and it is another card: one player, one legend (a rune column or a champion
   * read as a legend is not a second one). */
  otherLegend(side: string, card: string): boolean {
    const lg = this.legends.get(side);
    return lg !== undefined && this.rowOf.get(lg.printing_id)!.card_id !== card;
  }

  /** Something newer lies on this card: a card put on it or overlapping it. A covered card keeps its name (no re-reads
   * of a half-hidden face) and stays on the board until its spot clears. */
  covered(t: number, tr: Track): boolean {
    const [ax0, ay0, ax1, ay1] = this.boxesNow.get(tr.id) ?? aabb(tr.box);
    const area = Math.max(1.0, (ax1 - ax0) * (ay1 - ay0));
    for (const [oid, [bx0, by0, bx1, by1]] of this.boxesNow) {
      const o = this.tracks.get(oid);
      if (o === undefined || o === tr || o.first <= tr.first || t - o.last > 1.0) continue;
      const ix = Math.min(ax1, bx1) - Math.max(ax0, bx0);
      const iy = Math.min(ay1, by1) - Math.max(ay0, by0);
      if (ix > 0 && iy > 0 && ix * iy >= 0.25 * area) return true;
    }
    return false;
  }

  /** The part of card `a` that card `b` overlaps, on their outlines. */
  share(a: Track, b: Track): number {
    return overlapArea(quad(a.box), quad(b.box)) / Math.max(1.0, a.box.long_px * a.box.short_px);
  }

  /** Another named card in view, not this card outlined twice, that overlaps a quarter of this one or more: gear
   * tucked under a unit, a card put on another. Legends, battlefields and runes are not stacks. (The detector's
   * corner visibility would say which lies on top, but it is not reliable enough yet.) */
  stackedOn(t: number, tr: Track): Track | null {
    if (settled(tr.kind) || t - tr.last > 1.0) return null;
    let best = 0.25;
    let top: Track | null = null;
    for (const o of this.tracks.values()) {
      if (o === tr || !o.named || o.named === tr.named || settled(o.kind) || t - o.last > 1.0) continue;
      const sh = this.share(tr, o);
      if (sh >= best) {
        best = sh;
        top = o;
      }
    }
    return top;
  }

  /** An older track in view with the same name that covers half of this one: the same card outlined twice. */
  twin(t: number, tr: Track): boolean {
    if (!tr.named || tr.pinned || hiddenNow(t, tr)) return false;
    const n = Number(tr.id.slice(1));
    for (const o of this.tracks.values()) {
      if (o === tr || o.named !== tr.named) continue;
      const older = o.first < tr.first || (o.first === tr.first && Number(o.id.slice(1)) < n); // (o.first, int(o.id[1:])) < ...
      if (older && !hiddenNow(t, o) && this.share(tr, o) >= 0.5) return true;
    }
    return false;
  }

  /** The named cards under each card: a gear that overlaps a unit goes with the unit, and a card out of sight under a
   * newer one (`covered`) lies under it. Units side by side at a battlefield are no stack. */
  stacks(t: number): Map<string, Track[]> {
    const out = new Map<string, Track[]>();
    for (const u of this.tracks.values()) {
      if (!u.named || settled(u.kind)) continue;
      let host = u.kind === 'Gear' ? this.stackedOn(t, u) : null;
      if (host !== null && host.kind === 'Gear') host = null;
      if (host === null && t - u.last > 1.0 && this.covered(t, u)) {
        for (const o of this.tracks.values()) {
          // max(..., key=first): the first of the newest
          if (o !== u && o.named && o.named !== u.named && !settled(o.kind) && o.first > u.first && t - o.last <= 1.0 && this.share(u, o) >= 0.25) {
            if (host === null || o.first > host.first) host = o;
          }
        }
      }
      if (host !== null) {
        const list = out.get(host.id);
        if (list) list.push(u);
        else out.set(host.id, [u]);
      }
    }
    return out;
  }

  /** A pinned battlefield whose reads since say it is another card: a rune or a unit turned sideways (exhausted), read
   * once as a battlefield, which the pin would otherwise keep all game. */
  misreadBattlefield(tr: Track): boolean {
    const top = tr.top();
    if (tr.reads < UNPIN_READS || !top.length || top[0]![0] === tr.named) return false;
    return (top.find(([c]) => c === tr.named)?.[1] ?? 0.0) < UNPIN_P;
  }

  /** The pinned track of the battlefield `tr` is read as, when it is that card again: out of sight (the battlefield was
   * moved, and its pin follows it), or in sight and overlapping `tr` (outlined twice). */
  pinnedTwin(t: number, tr: Track): Track | null {
    for (const o of this.tracks.values()) {
      if (o !== tr && o.pinned && o.kind === 'Battlefield' && o.named === tr.named
          && (hiddenNow(t, o) || Math.max(this.share(tr, o), this.share(o, tr)) >= 0.25)) return o;
    }
    return null;
  }

  /** Another track of `tr`'s card that went out of sight around when `tr` appeared: the card moved. */
  vanished(t: number, tr: Track): Track | null {
    let best: Track | null = null;
    for (const o of this.tracks.values()) {
      if (o !== tr && o.named === tr.named && !o.pinned && t - o.last > 0.5 && o.last < tr.first + 0.5 && t - o.last < MOVE_S && !this.covered(t, o)) {
        if (best === null || o.last > best.last) best = o; // max(gone, key=last): the first of the latest
      }
    }
    return best;
  }

  // --- naming -------------------------------------------------------------------------------------------------------

  /** Two face-down looks in a row on a card never named. A hand resting on a named card or a blurred frame does not
   * hide it, and a card played under a hand is looked at again later. */
  faceDown(tr: Track): boolean {
    return tr.named === null && tr.down >= 2;
  }

  due(tr: Track, t: number): boolean {
    if (this.faceDown(tr)) return t - tr.lastRead > this.recheckS; // looked at now and then, never identified
    if (tr.named && (this.covered(t, tr) || this.stackedOn(t, tr) !== null)) return false; // its name is locked while something lies on it, or it lies on something
    const top = tr.top();
    if (!top.length || (top[0]![1] < this.sureP && tr.reads < this.maxReads)) return true;
    if (tr.pinned && tr.kind === 'Battlefield' && top[0]![0] !== tr.named && tr.reads < this.maxReads) return true; // read as another card than its pin: read again at once, until the reads can undo it
    return t - tr.lastRead > this.recheckS;
  }

  /** The gallery rows a crop on `side` competes with: once the side's legend is pinned, the cards of the lists that name
   * it (decklist.listMask: every printing of them, both lists' battlefields and the tokens) or, when no list does, those
   * that fit the legend (priors.legendMask, runes held to its domains; every battlefield and token), one mask per
   * legend. Null, the whole gallery, before that, on a side with no legend, or with the rule off. */
  allowed(side: string): Uint8Array | null {
    const lg = this.legendRule && side ? this.legends.get(side) : undefined;
    const row = lg !== undefined ? this.rowOf.get(lg.printing_id) : undefined;
    const card = row !== undefined ? row.card_id : this.legendRule && side ? this.legendByElimination(side) : null;
    if (card === null) return null;
    let mask = this.masks.get(card);
    if (mask === undefined) {
      this.tokens ??= tokenRows(this.rows).mask;
      mask = (this.decks.length ? listMask(this.catalogue(), this.decks, card) : null) ?? legendMask(this.rows, [card], { tokens: this.tokens, runes: true }).mask;
      this.masks.set(card, mask);
    }
    return mask;
  }

  /** With two lists given, once the other player's legend is one list's, this side's legend is the other list's: its
   * cards are read against that list, its own legend under dice or not read yet. Null otherwise: lists for another
   * match name neither legend on the table, and then neither is used. */
  legendByElimination(side: string): string | null {
    if (this.decks.length !== 2) return null;
    const lg = this.legends.get(this.otherSide(side));
    const row = lg !== undefined ? this.rowOf.get(lg.printing_id) : undefined;
    if (row === undefined) return null;
    const named = this.decks.map((d) => new Set(d.legends()));
    const theirs = [0, 1].filter((k) => named[k]!.has(row.card_id));
    if (theirs.length !== 1 || named[1 - theirs[0]!]!.size !== 1) return null;
    return [...named[1 - theirs[0]!]!][0]!;
  }

  /** A side's rows (`allowed`) and the rows of `card`, the card a track is named: a card named on one side stays itself
   * wherever it goes, a unit moved to a battlefield across the midline or taken by the other player. (Both players'
   * cards on the battlefield strip would name the cards of a hand held over it: the side's rows stay.) */
  withCard(allowed: Uint8Array, card: string): Uint8Array {
    let rows = this.cardRows.get(card);
    if (rows === undefined) {
      rows = Uint8Array.from(this.cards, (c) => (c === card ? 1 : 0));
      this.cardRows.set(card, rows);
    }
    const out = new Uint8Array(allowed.length);
    for (let g = 0; g < out.length; g++) out[g] = allowed[g]! | rows[g]!;
    return out;
  }

  /** The gallery's rows, indexed for decklists (made once, when a list is read). */
  catalogue(): Catalogue {
    this.cat ??= new Catalogue(this.rows);
    return this.cat;
  }

  /** The decklists the viewer gave, read through `catalogue()`: from the next read on, a side whose pinned legend a list
   * names competes with that list's cards; the other sides keep the legend rule. None: the legend rule everywhere. */
  setLists(decks: readonly Deck[]): void {
    this.decks = [...decks];
    this.masks.clear(); // the legend masks, and the other side's list by elimination, follow the lists
  }

  /** Per crop, its candidate cards as (card_id, probability, best score, best gallery row), best first. All four
   * turns go in one batch: which way up a card lies is unknown (exhausted, opponent side). `sides[n]` is where crop n
   * lies: under the legend rule, the rows its side's legend rules out score -Infinity before the best 60 are taken,
   * and the softmax runs over the cards left. `keeps[n]`, when given, is the card crop n's track is named: it competes
   * too, wherever the crop lies (`withCard`). */
  async identify(crops: readonly RgbImage[], sides?: readonly string[], keeps?: readonly (string | null)[]): Promise<Candidate[][]> {
    if (!crops.length) return [];
    const views: RgbImage[] = [];
    for (const c of crops) for (const r of ROTATIONS) views.push(r ? image.rotate(c, r, { expand: true }) : c);
    const emb = await this.enc.embed(views);
    const dim = emb.length / views.length;
    const turns = ROTATIONS.length;
    const out: Candidate[][] = [];
    crops.forEach((c, n) => {
      const level = { data: this.gallery.level(Math.max(c.width, c.height)), rows: this.gallery.rows, dim: this.gallery.dim };
      const sims = bestSimilarities({ data: emb.subarray(n * turns * dim, (n + 1) * turns * dim), rows: turns, dim }, level);
      let allowed = sides !== undefined ? this.allowed(sides[n]!) : null;
      const keep = keeps !== undefined ? keeps[n]! : null;
      if (allowed !== null && keep !== null) allowed = this.withCard(allowed, keep);
      if (allowed !== null) for (let g = 0; g < sims.length; g++) if (!allowed[g]) sims[g] = -Infinity;
      const scores = new Map<string, [number, number]>();
      for (const i of argsortDescending(sims).subarray(0, 60)) {
        if (!Number.isFinite(sims[i]!)) continue; // a row the legend rules out
        const card = this.cards[i]!;
        if (!scores.has(card)) scores.set(card, [sims[i]!, i]);
      }
      if (!scores.size) {
        out.push([]);
        return;
      }
      const vals = [...scores.values()].map((v) => v[0]);
      const top = Math.max(...vals);
      const p = vals.map((v) => Math.exp((v - top) / this.temperature));
      const sum = 0 + pairwiseSum(p);
      out.push([...scores].map(([card, [sc, i]], j): Candidate => [card, p[j]! / sum, sc, i]));
    });
    return out;
  }

  async read(t: number, frame: RgbImage, todo: readonly Track[]): Promise<void> {
    const crops: RgbImage[] = [];
    const owners: Track[] = [];
    for (const tr of todo) {
      tr.lastRead = t;
      if (tr.box.back) {
        tr.down += 1; // the detector saw a card back
        continue;
      }
      const c = cardCrop(frame, tr.box);
      if (detail(c) < FACE_DOWN_DETAIL) {
        tr.down += 1;
        continue;
      }
      tr.down = 0;
      crops.push(c);
      owners.push(tr);
    }
    const keeps = owners.map((tr) => (tr.named && tr.kind !== 'Battlefield' ? tr.named : null));
    const found = await this.identify(crops, owners.map((tr) => tr.side), keeps);
    owners.forEach((tr, n) => {
      tr.reads += 1;
      for (const [card, pc, sc, i] of found[n]!) {
        tr.prob.set(card, (tr.prob.get(card) ?? 0.0) + pc);
        if (sc > (tr.bestRow.get(card) ?? [-1.0, 0])[0]) tr.bestRow.set(card, [sc, i]);
      }
    });
  }

  /** The change gate on the whole table; a settled change is read a moment later, like a crop. */
  async watch(t: number, frame: RgbImage): Promise<RecognizerEvent[]> {
    if (this.gate === null) return [];
    const { width: w, height: h } = frame;
    const [x0, y0, x1, y1] = layoutBox(this.layout, w, h);
    const vw = this.gate.s.width;
    const vh = viewHeight(this.layout.table, vw);
    const view = image.resize(image.crop(frame, [x0, y0, x1, y1]), [vw, vh], 'bilinear');
    const events: RecognizerEvent[] = [];
    for (const ev of this.gate.feed(t, view)) {
      const fb = viewToFrame(ev.box, this.layout.table, vw, w, h);
      if (ev.kind === 'appeared' || ev.kind === 'changed') this.pending.push([t + 0.4, fb]);
      else if (ev.kind === 'disappeared') events.push(...this.left(t, fb));
    }
    const due = this.pending.filter(([when]) => when <= t).map(([, b]) => b);
    this.pending = this.pending.filter(([when]) => when > t);
    const cardLong = cardPx(this.layout, h);
    for (const [bx0, by0, bx1, by1] of due) {
      const mx = (bx0 + bx1) / 2;
      const my = (by0 + by1) / 2;
      if (this.scene.inOverlay(mx, my, w, h) || this.scene.inSuspect(mx, my, w, h)) continue; // a webcam or a banner changing, laid over the table: not a card
      const gx = (bx1 - bx0) * 0.08;
      const gy = (by1 - by0) * 0.08;
      const box: Box4 = [Math.max(0, bx0 - gx), Math.max(0, by0 - gy), Math.min(w, bx1 + gx), Math.min(h, by1 + gy)];
      let region = image.crop(frame, [pyRound(box[0]), pyRound(box[1]), pyRound(box[2]), pyRound(box[3])]);
      if (Math.min(region.width, region.height) < (0.45 * cardLong * 63) / 88 || Math.max(region.width, region.height) > 2.2 * cardLong) {
        continue; // a die or counter, or a whole area at once: not one card
      }
      if (region.width > region.height * 1.15) region = image.rotate(region, 90, { expand: true }); // cards stand portrait
      if (detail(region) < FACE_DOWN_DETAIL) continue; // a face-down card: never identified
      const cx = (box[0] + box[2]) / 2;
      const cy = (box[1] + box[3]) / 2;
      const side = layoutSide(this.layout, cx, cy, w, h); // before the read: the side's legend rules its candidates
      // ponytail: the whole changed region is read as one card; the visible-band matcher (stacks.py) for cards put on stacks
      const cands = (await this.identify([region], [side]))[0]!;
      if (!cands.length) continue;
      const [card, p, , i] = cands[0]!;
      const r = this.rows[i]!;
      if (p < this.gateP || settled(r.type ?? '') || this.recentlyPlayed(t, card, cx, cy)) continue;
      this.plays.push([t, card, cx, cy]);
      const q: [number, number][] = [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
      ];
      const guesses = cands.slice(0, 3).map(([c, pc, , j]): Guess => ({ printing_id: this.rows[j]!.printing_id, card_id: c, name: this.rows[j]!.name, p: pyRound(pc, 3) }));
      const flash: Flash = {
        id: `g${this.plays.length}`,
        quad: q.map(([x, y]): [number, number] => [pyRound(x, 1), pyRound(y, 1)]),
        side,
        state: 'named',
        printing_id: r.printing_id,
        name: r.name,
        confidence: pyRound(p, 3),
        guesses,
        since: pyRound(t, 2),
        until: t + 5,
        kind: 'card',
        hidden: false,
      };
      this.flashes.push(flash);
      events.push({ t: pyRound(t, 2), kind: 'played', text: `${r.name} played`, printing_id: r.printing_id, track: flash.id, side });
      this.announced.set(flash.id, [cx, cy, r.name, r.printing_id, side]);
    }
    return events;
  }

  /** A named card whose spot the gate saw cleared (a ghost, or a track the finder still holds). */
  left(t: number, box: Box4): RecognizerEvent[] {
    const [x0, y0, x1, y1] = box;
    const gx = (x1 - x0) * 0.1;
    const gy = (y1 - y0) * 0.1;
    const inside = (x: number, y: number): boolean => x0 - gx <= x && x <= x1 + gx && y0 - gy <= y && y <= y1 + gy;
    for (const tr of [...this.tracks.values()].sort((a, b) => b.last - a.last)) {
      if (tr.named && !tr.pinned && t - tr.last > this.forgetS && inside(tr.box.centre[0], tr.box.centre[1])) {
        this.tracks.delete(tr.id);
        const g = this.label(tr)[2];
        return [this.event(t, 'left', `${g.length ? g[0]!.name : tr.named} left the table`, tr, g.length ? g[0]!.printing_id : null)];
      }
    }
    for (const gh of [...this.ghosts].sort((a, b) => b.t - a.t)) {
      if (inside(gh.x, gh.y)) {
        this.ghosts.splice(this.ghosts.indexOf(gh), 1);
        return [{ t: pyRound(t, 2), kind: 'left', text: `${gh.name} left the table`, printing_id: gh.printing_id, track: gh.id, side: gh.side }];
      }
    }
    return [];
  }

  /** The same card announced nearby a moment ago, by the gate or by another track. */
  recentlyPlayed(t: number, card: string, x: number, y: number, window = 12.0, skip = ''): boolean {
    const reach = 1.5 * this.layout.card_long_1080;
    this.plays = this.plays.filter((pl) => t - pl[0] < window);
    if (this.plays.some(([, c, px, py]) => c === card && dist([x, y], [px, py]) < reach)) return true;
    for (const tr of this.tracks.values()) {
      if (tr.id !== skip && tr.named === card && t - tr.first < window && dist([x, y], tr.box.centre) < reach) return true;
    }
    return false;
  }

  // --- one frame ----------------------------------------------------------------------------------------------------

  /** Off the table camera: the board's clocks stop, so nothing is forgotten or re-read for the time away. */
  pause(dt: number): void {
    for (const tr of this.tracks.values()) {
      tr.last += dt;
      tr.lastRead += dt;
    }
    for (const gh of this.ghosts) gh.t += dt;
    this.plays = this.plays.map(([pt, c, x, y]): [number, string, number, number] => [pt + dt, c, x, y]);
    for (const tr of this.tracks.values()) if (tr.freeSince !== null) tr.freeSince += dt;
    for (const [side, h] of this.runeCounts) this.runeCounts.set(side, h.map(([pt, n, ex]): [number, number, number] => [pt + dt, n, ex]));
    this.pending = [];
    this.flashes = [];
  }

  /** The view is framed anew: cards found again by name re-anchor the rest (`reanchor`), the gate starts over, and
   * the cards first seen now were on the table already, not played. */
  cut(t: number): void {
    this.cutAt = t;
    this.anchorBase = new Map([...this.tracks].map(([k, tr]) => [k, tr.box]));
    this.anchorPairs = [];
    this.still = new StillTable(); // another view of the table: its still picture is taken again
    if (this.gateSettings !== null) this.gate = new ChangeGate(this.gateSettings);
  }

  /** Move the tracks not seen since the cut as the view moved (scale, turn and shift fitted to the cards found again
   * by name); a new track lying where a moved one now is, is that card and takes its id. */
  reanchor(): void {
    if (this.anchorPairs.length < 2 || this.cutAt === null) return;
    const cutAt = this.cutAt;
    const [scale, rot, shift] = similarity(this.anchorPairs.map(([a]) => a), this.anchorPairs.map(([, b]) => b));
    if (!(scale >= 0.5 && scale <= 2.0)) return;
    const turn = Math.atan2(rot[1][0], rot[0][0]) * RAD_TO_DEG;
    const old = [...this.tracks.values()].filter((o) => o.first < cutAt && o.last < cutAt && this.anchorBase.has(o.id));
    for (const o of old) {
      const b = this.anchorBase.get(o.id)!;
      const [bx, by] = b.centre;
      const c: [number, number] = [scale * rot[0][0] * bx + scale * rot[0][1] * by + shift[0], scale * rot[1][0] * bx + scale * rot[1][1] * by + shift[1]];
      o.box = { centre: c, long_px: b.long_px * scale, short_px: b.short_px * scale, angle_deg: pyMod(b.angle_deg + turn, 180), fill: b.fill };
    }
    for (const n of [...this.tracks.values()].filter((n) => n.first >= cutAt)) {
      const near = old.filter(
        (o) =>
          dist(o.box.centre, n.box.centre) < 0.35 * n.box.long_px &&
          Math.abs(o.box.long_px / n.box.long_px - 1) < 0.25 &&
          (o.named === null || n.named === null || n.named === o.named),
      );
      if (!near.length) continue;
      let o = near[0]!; // min(near, key=distance): the first of the nearest
      for (const m of near) if (dist(m.box.centre, n.box.centre) < dist(o.box.centre, n.box.centre)) o = m;
      o.box = n.box;
      o.last = n.last;
      o.hits = o.hits + n.hits;
      o.side = n.side;
      if (o.named === null && n.reads) {
        o.reads = n.reads;
        o.prob = n.prob;
        o.bestRow = n.bestRow;
        o.down = n.down;
        o.named = n.named;
        o.kind = n.kind;
      }
      this.tracks.delete(n.id);
      old.splice(old.indexOf(o), 1);
    }
  }

  /** One frame: the state the overlay draws and the events it saw. `budget` caps the crops read this frame. Steps
   * must not overlap: await one before starting the next. */
  async step(t: number, frame: RgbImage, budget = 10): Promise<[RecognizerState, RecognizerEvent[]]> {
    const tic = performance.now();
    if (this.t0 === null) this.t0 = t;
    const { width: w, height: h } = frame;
    this.frameWh = [w, h];
    const dt = this.lastT !== null ? t - this.lastT : 0.0;
    this.lastT = t;
    let boxes: CardBox[] | null = null;
    // the scene asks while it learns a broadcast with no known mat colour, or learns it again: the card-sized boxes (a
    // close-up's cards are bigger)
    const count = async (): Promise<number> => {
      if (boxes === null) boxes = await this.find(t, frame);
      const px = cardPx(this.layout, h);
      return boxes.filter((b) => b.long_px <= SIZE_MAX * px).length;
    };
    const on = await this.scene.onTable(t, frame, count);
    const withdrawn = this.scene.overlayN !== this.overlayN ? this.dropOverlaid(t, w, h) : [];
    if (!on) {
      if (!this.away) {
        this.away = true;
        this.beforeAway = new Set(this.prevSeen);
      }
      this.pause(dt);
      const state = this.state(t, w, h);
      for (const tr of state.tracks) tr.hidden = true; // the video is not the table: list the board, draw nothing on it
      state.status = 'away';
      state.message = 'the table camera is off; nothing is looked at until it is back';
      return [state, withdrawn];
    }
    const back = this.away;
    this.away = false;
    const small = StillTable.small(frame);
    this.still.feed(t, small);
    const all = boxes ?? (await this.find(t, frame));
    boxes = all;
    this.scene.sawCards(t, await count()); // what a view must show to be learnt as the table camera again
    const found: CardBox[] = this.keep(all, frame, w, h);
    const tf = performance.now();
    const seen = this.match(t, found, w, h);
    const before = back ? this.beforeAway : this.prevSeen;
    const ids = new Set(seen.map((tr) => tr.id));
    const gone = [...before].filter((id) => !ids.has(id)).length;
    if (before.size >= 6 && gone >= 0.7 * before.size && found.length >= 3) {
      this.cut(t); // most of the board moved at once: the view is framed anew
    } else if (back && this.gateSettings !== null) {
      this.gate = new ChangeGate(this.gateSettings); // the same view again: only the gate starts over
    }
    this.prevSeen = new Set(seen.filter((tr) => tr.hits >= 2).map((tr) => tr.id));
    this.boxesNow = new Map([...this.tracks].map(([k, tr]) => [k, aabb(tr.box)]));
    // New and uncertain cards first, then the oldest re-checks; a budget keeps each frame in time.
    // A box seen once may be the detector's slip (between two cards): only tracks seen twice are read.
    const table = layoutBox(this.layout, w, h);
    const still = HAND_STILL * cardPx(this.layout, h);
    for (const tr of seen) {
      // a card in a hand is not read, and not shown until it has been put down (D-005)
      if (this.scene.inSuspect(tr.box.centre[0], tr.box.centre[1], w, h)
        || handShare(frame, tr.box, table, seen.filter((o) => o !== tr).map((o) => o.box), this.still.bg, small) >= HAND_SKIN) {
        tr.freeSince = tr.freeAt = null;
      } else if (tr.freeSince === null || dist(tr.box.centre, tr.freeAt!) > still) {
        tr.freeSince = t; // out of the hand, or moved since: still from now
        tr.freeAt = [tr.box.centre[0], tr.box.centre[1]];
      }
      tr.placed = tr.placed || putDown(t, tr);
    }
    const todo = seen
      .filter((tr) => tr.hits >= 2 && putDown(t, tr) && this.due(tr, t))
      .sort((a, b) => Number(a.reads > 0) - Number(b.reads > 0) || a.lastRead - b.lastRead)
      .slice(0, budget);
    await this.read(t, frame, todo);
    const tr_ = performance.now();
    const events = [...withdrawn, ...this.announce(t), ...(await this.watch(t, frame))];
    this.flashes = this.flashes.filter((f) => f.until > t);
    this.timing = { find_ms: tf - tic, read_ms: tr_ - tf, gate_ms: performance.now() - tr_, reads: todo.length, boxes: found.length };
    return [this.state(t, w, h), events];
  }

  /** The track's state, confidence and up to three guesses (each card's best-matching printing). */
  label(tr: Track): [string, number, Guess[]] {
    if (this.faceDown(tr)) return ['facedown', 0.0, []];
    let top = tr.top();
    if (!top.length) return ['new', 0.0, []];
    if (tr.pinned) top = [...top].sort((a, b) => Number(a[0] !== tr.named) - Number(b[0] !== tr.named)); // a legend or battlefield keeps its name all game: re-reads move only its confidence
    const guesses: Guess[] = [];
    for (const [c, p] of top.slice(0, 3)) {
      const r = this.rows[(tr.bestRow.get(c) ?? [0.0, this.firstRow.get(c)!])[1]]!;
      guesses.push({ printing_id: r.printing_id, card_id: c, name: r.name, p: pyRound(p, 3) });
    }
    const p0 = top[0]![1];
    const kind = this.rows[this.firstRow.get(top[0]![0])!]!.type;
    const legend = kind === 'Legend';
    if (!tr.pinned && legend && this.sideLegend(tr) !== null) return ['unsure', p0, guesses]; // one player, one legend: another outline of it, or a card misread as one
    if (!tr.pinned && kind === 'Battlefield' && !this.inStrip(tr.box.centre[0], tr.box.centre[1])) {
      // off the battlefield strip, a battlefield only once the reads agree: a card turned sideways is not one
      return [tr.reads >= STRIP_READS && p0 >= this.minP ? 'named' : 'unsure', p0, guesses];
    }
    let named = tr.pinned || p0 >= this.sureP || (p0 >= this.minP && tr.reads >= 2);
    if (!named && tr.reads >= 4 && p0 >= 0.3 && legend && !this.otherLegend(tr.side, top[0]![0]) && this.listedLegend(top[0]![0])) {
      named = true; // a legend, read the same way four times: one a player, and its frame is like no other card's
    }
    return [named ? 'named' : 'unsure', p0, guesses];
  }

  /** No list given, or a given list names this legend: with lists, another legend needs a sure read to be named. */
  listedLegend(card: string): boolean {
    return !this.decks.length || this.decks.some((d) => d.legends().includes(card));
  }

  /** 'played' when a card is first named, 'moved' when a named card that just vanished is named again elsewhere (it
   * keeps its first id). A card out of sight keeps its track: an unnamed one `forgetS`, a named one `KEEP_S` or as
   * long as something lies on it, a legend or battlefield all game. Then a named card becomes a ghost (see `ghosts`).
   * Runes, legends and battlefields are never announced, and nor are the cards of a burst: BURST of them first named
   * within BURST_S (a graphic of a deck, a view framed anew). */
  announce(t: number): RecognizerEvent[] {
    const events: RecognizerEvent[] = [];
    const played: RecognizerEvent[] = []; // this step's plays, kept back until every card first named now is counted
    this.ghosts = this.ghosts.filter((gh) => t - gh.t < 60);
    this.firstNamed = this.firstNamed.filter((ft) => t - ft < BURST_S);
    for (const tr of [...this.tracks.values()]) {
      if (!this.tracks.has(tr.id)) continue; // merged into the track it moved from
      const gone = t - tr.last;
      const limit = tr.named ? KEEP_S : tr.hits >= 2 ? this.forgetS : 1.0;
      if (!tr.pinned && gone > limit && !(tr.named && this.covered(t, tr))) {
        this.tracks.delete(tr.id);
        if (tr.named) {
          const g = this.label(tr)[2];
          this.ghosts.push({
            t: tr.last,
            card: tr.named,
            x: tr.box.centre[0],
            y: tr.box.centre[1],
            name: g.length ? g[0]!.name : tr.named,
            printing_id: g.length ? g[0]!.printing_id : null,
            side: tr.side,
            id: tr.id,
          });
        }
        continue;
      }
      if (tr.pinned && tr.kind === 'Battlefield' && this.misreadBattlefield(tr)) tr.pinned = false; // read once as a battlefield, as another card since: named again as that card
      if (!tr.pinned && tr.kind === 'Battlefield' && tr.named && !hiddenNow(t, tr) && this.inStrip(tr.box.centre[0], tr.box.centre[1])) {
        const o = this.pinnedTwin(t, tr);
        if (o !== null && hiddenNow(t, o)) {
          o.box = tr.box;
          o.last = tr.last;
          o.hits = o.hits + tr.hits;
          this.tracks.delete(tr.id); // a pinned battlefield's second outline, in sight where it is not: the pin goes there
          continue;
        }
      }
      const [state, , g] = this.label(tr);
      if (state === 'named' && tr.named !== g[0]!.card_id) {
        const changed = tr.named !== null;
        const named = g[0]!.card_id;
        tr.named = named;
        tr.kind = this.rowOf.get(g[0]!.printing_id)?.type ?? '';
        const cutAt = this.cutAt;
        const afterCut = cutAt !== null && !changed && tr.first - cutAt >= 0 && tr.first - cutAt < 30;
        if (afterCut) {
          // a card on the table before the cut, found again by name in the new view
          const olds = [...this.tracks.values()].filter((o) => o !== tr && o.named === named && o.first < cutAt && o.last < cutAt);
          if (olds.length === 1) {
            const o = olds[0]!;
            this.anchorPairs.push([(this.anchorBase.get(o.id) ?? o.box).centre, tr.box.centre]);
            o.box = tr.box;
            o.last = tr.last;
            o.hits = o.hits + tr.hits;
            o.side = tr.side;
            this.tracks.delete(tr.id); // it keeps its first id
            this.reanchor();
            continue;
          }
        }
        const strip = this.inStrip(tr.box.centre[0], tr.box.centre[1]);
        if (tr.kind === 'Battlefield' && strip) {
          const o = this.pinnedTwin(t, tr);
          if (o !== null) {
            if (hiddenNow(t, o)) {
              // the battlefield was moved: its pin follows it, under its first id
              o.box = tr.box;
              o.last = tr.last;
              o.hits = o.hits + tr.hits;
              this.tracks.delete(tr.id);
            }
            continue; // or the same battlefield outlined again: not a second one, and not drawn (state)
          }
        }
        if (STATIC.includes(tr.kind) && (tr.kind !== 'Battlefield' || strip)
            && !(tr.kind === 'Legend' && (this.otherLegend(tr.side, named) || this.sideLegend(tr) !== null))) {
          tr.pinned = true; // set up before the game: nothing to announce, and it stays put
          if (tr.kind === 'Legend' && !this.legends.has(tr.side)) {
            // the side's legend, however sure its reads are under the dice
            this.legends.set(tr.side, { printing_id: g[0]!.printing_id, name: g[0]!.name });
          }
          for (const o of [...this.tracks.values()].filter((o) => !o.pinned && this.onLegend(o.box))) {
            this.tracks.delete(o.id); // a second outline of a legend, read as a card of its own
          }
        }
        if (settled(tr.kind)) continue;
        const was = !changed && !afterCut ? this.vanished(t, tr) : null;
        if (was !== null) {
          const far = dist(was.box.centre, tr.box.centre) > 0.6 * tr.box.long_px;
          was.box = tr.box;
          was.last = tr.last;
          was.hits = was.hits + tr.hits;
          was.side = tr.side;
          this.tracks.delete(tr.id); // the same card: it keeps its first id and what was read of it
          if (far) events.push(this.event(t, 'moved', `${g[0]!.name} moved`, was, g[0]!.printing_id));
          continue;
        }
        const reach = 1.5 * this.layout.card_long_1080;
        const back = this.ghosts.filter((gh) => gh.card === named && dist([gh.x, gh.y], tr.box.centre) < reach);
        if (back.length && !changed) {
          this.ghosts.splice(this.ghosts.indexOf(back[0]!), 1); // the same card, found again
          continue;
        }
        if (!changed) this.firstNamed.push(t); // a card new to the board
        if (tr.first - (this.t0 ?? 0.0) < this.settleS && !changed) continue; // on the table when we tuned in, not played now
        if (afterCut && tr.first - cutAt < this.settleS + 2) continue; // on the table when the view changed
        if (changed) {
          events.push(this.event(t, 'changed', `${g[0]!.name} (read again)`, tr, g[0]!.printing_id));
        } else if (!this.recentlyPlayed(t, named, tr.box.centre[0], tr.box.centre[1], 12.0, tr.id)) {
          this.plays.push([t, named, tr.box.centre[0], tr.box.centre[1]]);
          played.push(this.event(t, 'played', `${g[0]!.name} played`, tr, g[0]!.printing_id));
        } else {
          this.plays.push([t, named, tr.box.centre[0], tr.box.centre[1]]);
        }
      }
    }
    if (this.firstNamed.length < BURST) {
      for (const ev of played) {
        const tr = this.tracks.get(ev.track);
        if (tr !== undefined) this.announced.set(tr.id, [tr.box.centre[0], tr.box.centre[1], ev.text.slice(0, -' played'.length), ev.printing_id, tr.side]);
        events.push(ev);
      }
    }
    return events;
  }

  event(t: number, kind: string, text: string, tr: Track, pid: string | null): RecognizerEvent {
    return { t: pyRound(t, 2), kind, text, printing_id: pid, track: tr.id, side: tr.side };
  }

  /** Whether a ready card stands upright in the picture. From the player's seat, a ready card points straight at them
   * and an exhausted (used) one lies across, along their edge. Battlefields, printed landscape and never exhausted, lie
   * along the edges too, so a ready card stands at right angles to them; before one is named, the layout says where the
   * players sit (left and right: a ready card lies across the picture). */
  readyUpright(): boolean {
    const fields = [...this.tracks.values()].filter((tr) => tr.pinned && tr.kind === 'Battlefield');
    if (fields.length) return fields.filter((tr) => upright(tr.box)).length * 2 < fields.length;
    return this.layout.split === 'horizontal';
  }

  /** A player's runes in this frame, one per card (the newest of two tracks on one card), and how many are exhausted. */
  runesSeen(t: number, side: string, ready: boolean): [number, number] {
    const px = cardPx(this.layout, this.frameWh[1]);
    const kept: Track[] = [];
    const runes = [...this.tracks.values()]
      .filter(
        (tr) =>
          tr.kind === 'Rune' &&
          tr.side === side &&
          tr.hits >= 2 &&
          t - tr.last <= RUNE_SEEN_S &&
          tr.box.long_px / Math.max(1e-6, tr.box.short_px) <= RUNE_ASPECT &&
          tr.box.long_px <= RUNE_LONG * px,
      )
      .sort((a, b) => b.last - a.last || b.hits - a.hits);
    for (const tr of runes) if (kept.every((k) => dist(tr.box.centre, k.box.centre) > RUNE_APART * px)) kept.push(tr);
    return [kept.length, kept.filter((tr) => upright(tr.box) !== ready).length];
  }

  /** Read as a rune: named one, or a rune on RUNE_SHARE of its reads. */
  runeLike(tr: Track): boolean {
    if (tr.kind === 'Rune') return true;
    if (tr.reads < 2) return false;
    let p = 0;
    for (const [c, pc] of tr.prob) if (this.typeOf.get(c) === 'Rune') p += pc;
    return p >= RUNE_SHARE * tr.reads;
  }

  /** A player's runes in this frame: the boxes read as runes, the unnamed card-sized boxes beside them turned their way
   * (a stack's covered strips), and the runes a stack's wider gaps hide (`hiddenRunes`). */
  runesNow(t: number, side: string): number {
    const px = cardPx(this.layout, this.frameWh[1]);
    const now = [...this.tracks.values()].filter(
      (tr) =>
        tr.side === side &&
        tr.hits >= 2 &&
        tr.last === t &&
        tr.box.long_px / Math.max(1e-6, tr.box.short_px) <= RUNE_ASPECT &&
        tr.box.long_px <= RUNE_LONG * px,
    );
    const runes = now.filter((tr) => this.runeLike(tr));
    for (;;) {
      const more = now.filter(
        (tr) =>
          tr.kind === '' &&
          !runes.includes(tr) &&
          Math.abs(tr.box.long_px / px - 1) <= RUNE_SIZE &&
          runes.some((r) => dist(tr.box.centre, r.box.centre) < RUNE_JOIN * px && turnApart(tr.box.angle_deg, r.box.angle_deg) <= RUNE_TURN),
      );
      if (!more.length) break;
      runes.push(...more);
    }
    return runes.length + stacksOf(runes.map((tr) => tr.box), RUNE_LINK * px).reduce((n, g) => n + hiddenRunes(g, px), 0);
  }

  /** A player's runes on the table: counted, never named, and how many are exhausted (used this turn), over the last
   * frames (RUNE_WINDOW_S); off the table camera it holds. The count is this frame's (`runesNow`), the exhausted ones
   * those named runes seen in the last second that lie across (`runesSeen`). */
  runes(t: number, side: string, ready: boolean): Runes {
    let recent = this.runeCounts.get(side);
    if (recent === undefined) this.runeCounts.set(side, (recent = []));
    if (!this.away) {
      const n = this.runesNow(t, side);
      const ex = this.runesSeen(t, side, ready)[1];
      if (recent.length && recent[recent.length - 1]![0] >= t) recent.pop(); // the state asked again for this frame
      recent.push([t, n, ex]);
      while (recent[0]![0] <= t - RUNE_WINDOW_S) recent.shift();
    }
    if (!recent.length) return { count: 0, exhausted: 0 };
    const counts = recent.map(([, n]) => n).sort((a, b) => a - b);
    const count = counts[Math.min(counts.length - 1, Math.floor((3 * counts.length) / 4))]!;
    const newest = recent[recent.length - 1]![0];
    const used = recent.filter(([pt]) => pt > newest - RUNE_EXHAUSTED_S).map(([, , ex]) => ex).sort((a, b) => a - b);
    return { count, exhausted: Math.min(count, used[Math.floor(used.length / 2)]!) };
  }

  state(t: number, w: number, h: number): RecognizerState {
    const tracks: StateTrack[] = [];
    const under = this.stacks(t);
    const ready = this.readyUpright();
    for (const tr of this.tracks.values()) {
      if (tr.hits < 2 || !tr.placed) continue; // seen once (maybe the detector's slip, a box between two cards), or only ever in a hand
      const hidden = t - tr.last > (tr.pinned ? PIN_HIDE_S : 1.0);
      if (hidden && !tr.named) continue;
      const [state, p, g] = this.label(tr);
      const lg = this.sideLegend(tr);
      if (lg !== null && !tr.pinned && g.length && g[0]!.card_id === lg.named) continue; // another outline of the side's legend (its case, the die on it): not a card
      if (this.twin(t, tr)) continue; // the same card outlined again (a sleeve's or a toploader's edge): drawn once
      if (!tr.pinned && tr.kind === 'Battlefield') {
        const o = this.pinnedTwin(t, tr);
        if (o !== null && !hiddenNow(t, o)) continue; // a pinned battlefield's second outline: drawn once
      }
      const top = g.length && state === 'named' ? g[0]! : null;
      // hidden: out of sight (under a hand or another card) but still on the board, so listed, not drawn
      const unders: Under[] = [];
      for (const u of under.get(tr.id) ?? []) {
        const gu = this.label(u)[2];
        if (gu.length) unders.push({ id: u.id, name: gu[0]!.name, printing_id: gu[0]!.printing_id });
      }
      tracks.push({
        id: tr.id,
        quad: quad(tr.box),
        side: tr.side,
        state,
        printing_id: top ? top.printing_id : null,
        name: top ? top.name : '',
        confidence: pyRound(p, 3),
        guesses: state !== 'facedown' ? g : [],
        since: pyRound(tr.first, 2),
        kind: KINDS.get(tr.kind) ?? 'card',
        hidden,
        under: unders,
      });
    }
    for (const f of this.flashes) {
      const { until: _until, ...shown } = f;
      tracks.push(shown);
    }
    for (const tr of tracks) {
      const pid = tr.printing_id;
      if (tr.state === 'named' && pid && !this.legends.has(tr.side) && tr.confidence >= this.sureP) {
        const r = this.rowOf.get(pid);
        if (r !== undefined && r.type === 'Legend') this.legends.set(tr.side, { printing_id: pid, name: r.name });
      }
    }
    return {
      t: pyRound(t, 2),
      status: 'live',
      message: '',
      title: this.title,
      frame: { width: w, height: h },
      players: layoutSides(this.layout).map((s, k) => ({ side: s, label: `Player ${k + 1}`, legend: this.legends.get(s) ?? null, runes: this.runes(t, s, ready) })),
      layout: { name: this.layout.name, table: [...this.layout.table] },
      tracks,
    };
  }
}

/** The gallery levels live/__main__.py embeds for a layout: a card's long side at 1080p times 0.8, 0.9 and 1, each
 * rounded to 10 px (the Pyramid the Recognizer is given should hold these). */
export function galleryScales(layout: Layout): number[] {
  const px = cardPx(layout, 1080);
  return [...new Set([0.8, 0.9, 1.0].map((f) => Math.trunc(pyRound((px * f) / 10) * 10)))].sort((a, b) => a - b);
}
