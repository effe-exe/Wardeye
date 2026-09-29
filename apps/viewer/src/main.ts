// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// DOM wiring for the viewer preview. The logic lives in demo.ts (unit-tested).

import {
  EVENT_WORDS,
  checkBundle,
  corners,
  currentEvent,
  eventGuesses,
  formatTime,
  hoverState,
  sampleAt,
  type DemoBundle,
  type Track,
} from './demo';

declare global {
  interface Window {
    RIFTEYE_DEMO?: unknown;
  }
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const video = $<HTMLVideoElement>('video');
const overlay = document.getElementById('overlay') as unknown as SVGSVGElement;
const hover = $('hover');
const SVG = 'http://www.w3.org/2000/svg';

const issues = checkBundle(window.RIFTEYE_DEMO);
if (issues.length > 0) {
  $('problem').hidden = false;
  $('problem').textContent = `This page needs its demo files next to it (data.js, the video and art/). ${issues.join('; ')}.`;
  throw new Error(issues.join('; '));
}
const bundle = window.RIFTEYE_DEMO as DemoBundle;
const [fw, fh] = bundle.frame;
$('title').textContent = bundle.title || 'preview';
$('player').style.aspectRatio = `${fw} / ${fh}`;
overlay.setAttribute('viewBox', `0 0 ${fw} ${fh}`);
video.src = bundle.video;

let hot: string | null = null; // the track under the pointer
let pointer = { x: 0, y: 0 };

// --------------------------------------------------------------------------------------
// Hover card
// --------------------------------------------------------------------------------------

function img(src: string, alt: string): HTMLImageElement {
  const el = document.createElement('img');
  el.src = src;
  el.alt = alt;
  return el;
}

function showHover(track: Track): void {
  const state = hoverState(bundle, track.guesses, track.faceDown);
  hover.replaceChildren();
  if (state.kind === 'sure') {
    const one = document.createElement('div');
    one.className = 'one';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = state.card.name;
    const sure = document.createElement('div');
    sure.className = 'sure';
    sure.textContent = `Wardeye is ${Math.round(state.p * 100)}% sure`;
    one.append(img(state.card.art, state.card.name), name, sure);
    hover.append(one);
  } else if (state.kind === 'unsure') {
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = 'Not sure yet. Best guesses:';
    const three = document.createElement('div');
    three.className = 'three';
    for (const o of state.options) {
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = `${o.card.name} · ${Math.round(o.p * 100)}%`;
      fig.append(img(o.card.art, o.card.name), cap);
      three.append(fig);
    }
    hover.append(note, three);
  } else if (state.kind === 'face-down') {
    hover.textContent = 'A face-down card. It is hidden information, so Wardeye never tries to identify it.';
  } else {
    hover.textContent = 'A card Wardeye does not know yet.';
  }
  hover.hidden = false;
  placeHover();
}

function placeHover(): void {
  if (hover.hidden) return;
  const r = hover.getBoundingClientRect();
  let x = pointer.x + 18;
  let y = pointer.y - r.height / 2;
  if (x + r.width > window.innerWidth - 8) x = pointer.x - r.width - 18;
  y = Math.max(8, Math.min(window.innerHeight - r.height - 8, y));
  hover.style.left = `${x}px`;
  hover.style.top = `${y}px`;
}

function leave(): void {
  if (hot) polys.get(hot)?.classList.remove('hot');
  hot = null;
  hover.hidden = true;
}

document.addEventListener('pointermove', (e) => {
  pointer = { x: e.clientX, y: e.clientY };
  // Anywhere but a card box closes the hover card, whatever events the boxes missed.
  if (hot && !(e.target instanceof SVGPolygonElement)) leave();
  placeHover();
});

// --------------------------------------------------------------------------------------
// Boxes, drawn every frame at the video's current time
// --------------------------------------------------------------------------------------

const polys = new Map<string, SVGPolygonElement>();

function polygonFor(track: Track): SVGPolygonElement {
  let p = polys.get(track.id);
  if (!p) {
    p = document.createElementNS(SVG, 'polygon');
    p.dataset.track = track.id;
    const g = track.guesses?.[0];
    if (track.faceDown) p.classList.add('facedown');
    else if (!g || g.p < 0.75) p.classList.add('unsure');
    p.addEventListener('pointerenter', () => {
      if (hot && hot !== track.id) leave();
      hot = track.id;
      p!.classList.add('hot');
      showHover(track);
    });
    p.addEventListener('pointerleave', () => {
      if (hot === track.id) leave();
    });
    polys.set(track.id, p);
    overlay.append(p); // added once and kept: moving boxes in and out of the page loses pointer events
  }
  return p;
}

let lastNow = '';
function draw(): void {
  const t = video.currentTime || 0;
  const names = new Set<string>();
  for (const track of bundle.tracks) {
    const s = sampleAt(track, t);
    const p = polys.get(track.id) ?? (s ? polygonFor(track) : undefined);
    if (!p) continue;
    if (!s) {
      p.style.display = 'none';
      if (hot === track.id) leave();
      continue;
    }
    p.style.display = '';
    p.setAttribute('points', corners(s, bundle.frame).map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '));
    const g = track.guesses?.[0];
    if (track.faceDown) names.add('a face-down card');
    else if (g && bundle.cards[g.card]) names.add(bundle.cards[g.card]!.name + (g.p < 0.75 ? '?' : ''));
  }
  const now = [...names].join(', ') || '–';
  if (now !== lastNow) {
    $('now-list').textContent = now;
    lastNow = now;
  }
  highlight(t);
  requestAnimationFrame(draw);
}

$<HTMLInputElement>('show-boxes').addEventListener('change', (e) => {
  overlay.classList.toggle('hidden-boxes', !(e.target as HTMLInputElement).checked);
});

// --------------------------------------------------------------------------------------
// Timeline
// --------------------------------------------------------------------------------------

const rows = bundle.events.map((e) => {
  const li = document.createElement('li');
  const time = document.createElement('span');
  time.className = 'time';
  time.textContent = formatTime(e.t);
  const guesses = eventGuesses(bundle, e);
  const g = guesses?.[0];
  const card = g ? bundle.cards[g.card] : undefined;
  const pic = card ? img(card.art, card.name) : document.createElement('span');
  const what = document.createElement('span');
  what.className = 'what';
  what.textContent = card ? `${card.name}${g && g.p < 0.75 ? '?' : ''}` : 'a card';
  const small = document.createElement('small');
  small.textContent = EVENT_WORDS[e.kind];
  what.append(small);
  li.append(time, pic, what);
  li.title = 'Jump to this moment';
  li.addEventListener('click', () => {
    video.currentTime = Math.max(0, e.tBefore - 1);
    void video.play();
  });
  return li;
});
$('events').replaceChildren(...rows);

let lastCurrent = -2;
function highlight(t: number): void {
  const k = currentEvent(bundle.events, t);
  if (k === lastCurrent) return;
  lastCurrent = k;
  rows.forEach((li, i) => {
    li.classList.toggle('current', i === k);
    li.classList.toggle('future', i > k);
  });
  rows[k]?.scrollIntoView({ block: 'nearest' });
}

requestAnimationFrame(draw);
