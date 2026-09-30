// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Wardeye on the player itself. While the video plays, this grabs its current frame a few times a
// second and hands it to the extension's worker, which asks the live runner on this machine; the
// board it answers with is drawn over the picture: each card's box, its name once Wardeye is sure,
// and a hover card when you point at it. Paused, or turned off with Alt+R, nothing is read or sent and the board stays.
// The plays panel (panel.ts) connects here while it is open: this tab's board and plays (plays.ts) go to it, and a play
// clicked there jumps the video to it.
// How it looks is overlay.css (the brand book's section 8); this file makes the markup and never sets a colour.

import { NAME, badge, badgeDetail, badgeParts, becameNamed, boxClass, captureSize, contentRect, drawn, frameInterval, hoverCard, label, labelAnchor, type State, type Track } from './geometry';
import { MARK_SHAPES, MARK_VIEWBOX } from './mark';
import type { BoardEvent } from './parts';
import { PlayLog, sidesOf, type Snapshot } from './plays';

const SVG = 'http://www.w3.org/2000/svg';
const EVERY_MS = 250; // the live runner: at most four frames a second; a laptop reads about two. The engine in the extension sets its own pace

let port: chrome.runtime.Port | null = null;
let inFlight = false;
let lastSent = 0;
let online = true;
let shown = true;
let state: State | null = null;
let video: HTMLVideoElement | null = null;
const grab = document.createElement('canvas');
const art = new Map<string, Promise<ImageBitmap | null>>();
const waiting = new Map<string, (jpeg: string | null) => void>();
// Keyed by track id, so a card that stays on the table keeps its box, and the hover under the pointer, across updates.
const drawnBoxes = new Map<string, { poly: SVGPolygonElement; text: SVGTextElement; track: Track; pulsing: boolean }>();
let hovered: string | null = null;
let hoverAt: PointerEvent | null = null;
const log = new PlayLog(); // this video's plays, for the plays panel
let lists: string[] = []; // the plays panel's decklist boxes for this video, one a player ('' for an empty one)
let listsVideo = ''; // the video they were pasted for
const LIST_SLOTS = 2; // a box for each player
const panels = new Set<chrome.runtime.Port>(); // the plays panels open on this tab
let boardSent = 0;
const BOARD_EVERY_MS = 1000; // the panel's board, at most once a second; a new play at once
const SEEK_BEFORE_S = 2; // a play clicked in the panel: the video goes back to just before it

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, text = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  if (text) e.textContent = text;
  return e;
}

/** The brand mark, small, for the badge: the brand's shapes, painted by the stylesheet's tokens. */
function markSvg(): SVGSVGElement {
  const mark = document.createElementNS(SVG, 'svg');
  mark.setAttribute('class', 'rifteye-mark');
  mark.setAttribute('viewBox', MARK_VIEWBOX);
  mark.setAttribute('aria-hidden', 'true');
  for (const shape of MARK_SHAPES) {
    const e = document.createElementNS(SVG, shape.tag);
    e.setAttribute('class', shape.cls);
    for (const [name, value] of Object.entries(shape.attrs)) e.setAttribute(name, value);
    mark.append(e);
  }
  return mark;
}

/** The off switch's icon: a ring open at the top and a stroke through the gap, drawn with the brand's 1.5 px line. */
function powerSvg(): SVGSVGElement {
  const icon = document.createElementNS(SVG, 'svg');
  icon.setAttribute('viewBox', '0 0 24 24');
  icon.setAttribute('aria-hidden', 'true');
  for (const d of ['M12 3.5v8', 'M7 6.5a7.5 7.5 0 1 0 10 0']) {
    const path = document.createElementNS(SVG, 'path');
    path.setAttribute('d', d);
    icon.append(path);
  }
  return icon;
}

/** The plays panel's icon: three short rules, each after a dot, drawn with the brand's line. */
function playsSvg(): SVGSVGElement {
  const icon = document.createElementNS(SVG, 'svg');
  icon.setAttribute('viewBox', '0 0 24 24');
  icon.setAttribute('aria-hidden', 'true');
  for (const d of ['M9 6h11', 'M9 12h11', 'M9 18h11', 'M4.5 6h.01', 'M4.5 12h.01', 'M4.5 18h.01']) {
    const path = document.createElementNS(SVG, 'path');
    path.setAttribute('d', d);
    icon.append(path);
  }
  return icon;
}

const root = el('div', 'rifteye-root');
const svg = document.createElementNS(SVG, 'svg');
svg.setAttribute('class', 'rifteye-svg');
svg.setAttribute('preserveAspectRatio', 'none');
const badgeEl = el('div', 'rifteye-badge');
const badgeMain = el('div', 'rifteye-badge-main'); // the mark, then the name and the status in one run of text
const badgeText = el('span', 'rifteye-badge-text');
const badgeName = el('span', 'rifteye-badge-name', NAME);
const badgeStatus = el('span', 'rifteye-badge-status');
badgeText.append(badgeName, badgeStatus);
const offButton = el('button', 'rifteye-off'); // turns Wardeye off in this tab, as Alt+R and the toolbar button do
offButton.type = 'button';
offButton.title = 'Turn Wardeye off (Alt+R)';
offButton.setAttribute('aria-label', 'Turn Wardeye off (Alt+R)');
offButton.append(powerSvg());
const playsButton = el('button', 'rifteye-plays'); // opens the plays panel beside the page
playsButton.type = 'button';
playsButton.title = 'Plays and players';
playsButton.setAttribute('aria-label', 'Open the plays panel');
playsButton.append(playsSvg());
badgeMain.append(markSvg(), badgeText, playsButton, offButton);
const badgeDetailEl = el('div', 'rifteye-badge-detail'); // the engine's timings, when the engine in the extension reads
badgeDetailEl.hidden = true;
badgeEl.append(badgeMain, badgeDetailEl);
const card = el('div', 'rifteye-card');
card.hidden = true;
root.append(svg, badgeEl, card);

/** The badge's text: the name in one typeface, what follows it in another (the text is the same). */
function setBadge(text: string): void {
  const parts = badgeParts(text);
  badgeName.textContent = parts.name;
  badgeStatus.textContent = parts.status;
}

function b64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

function connect(): void {
  const p = chrome.runtime.connect({ name: 'rifteye' });
  p.onMessage.addListener((msg: { kind: string; online?: boolean; state?: State | null; events?: BoardEvent[]; printing_id?: string; jpeg?: string | null }) => {
    if (msg.kind === 'state') {
      inFlight = false;
      online = Boolean(msg.online);
      state = online ? (msg.state ?? state) : null;
      draw();
      const fresh = Array.isArray(msg.events) ? log.add(location.pathname, msg.events) : [];
      sendBoard(fresh.length > 0);
    } else if (msg.kind === 'art' && msg.printing_id) {
      waiting.get(msg.printing_id)?.(msg.jpeg ?? null);
      waiting.delete(msg.printing_id);
    }
  });
  p.onDisconnect.addListener(() => {
    port = null;
    inFlight = false;
    setTimeout(connect, 1000); // the worker was put to sleep: wake it again
  });
  port = p;
}

/** The main player: the largest video on the page that has a picture. */
function findVideo(): HTMLVideoElement | null {
  let best: HTMLVideoElement | null = null;
  let area = 0;
  for (const v of Array.from(document.querySelectorAll('video'))) {
    const r = v.getBoundingClientRect();
    if (v.readyState >= 2 && r.width >= 320 && r.width * r.height > area) {
      best = v;
      area = r.width * r.height;
    }
  }
  return best;
}

async function capture(v: HTMLVideoElement): Promise<string | null> {
  const [w, h] = captureSize(v.videoWidth, v.videoHeight);
  grab.width = w;
  grab.height = h;
  grab.getContext('2d')?.drawImage(v, 0, 0, w, h);
  const blob = await new Promise<Blob | null>((resolve) => grab.toBlob(resolve, 'image/jpeg', 0.85));
  return blob ? b64(await blob.arrayBuffer()) : null;
}

function tick(): void {
  video = findVideo();
  place();
  if (!video || !shown || video.paused || video.ended || inFlight || !port) return; // off (Alt+R) or paused: nothing is read
  const now = performance.now();
  if (now - lastSent < frameInterval(state, EVERY_MS)) return;
  inFlight = true;
  lastSent = now;
  const t = video.currentTime;
  capture(video)
    .then((jpeg) => {
      if (!jpeg || !port) inFlight = false;
      else port.postMessage({ kind: 'frame', t, video: location.pathname, jpeg, ...(filledLists().length ? { lists: filledLists() } : {}) });
    })
    .catch(() => {
      inFlight = false; // a player whose picture cannot be read (a protected stream): nothing to send
      setBadge(`${NAME}: this player cannot be read`);
    });
}

/** Keeps the overlay on the picture: theatre mode, resizing, scrolling, fullscreen. */
function place(): void {
  const host = document.fullscreenElement ?? document.body;
  if (root.parentElement !== host) host.append(root);
  if (!video || !shown) {
    root.style.display = 'none';
    return;
  }
  const r = video.getBoundingClientRect();
  const c = contentRect({ left: r.left, top: r.top, width: r.width, height: r.height }, video.videoWidth, video.videoHeight);
  root.style.display = '';
  root.style.left = `${c.left}px`;
  root.style.top = `${c.top}px`;
  root.style.width = `${c.width}px`;
  root.style.height = `${c.height}px`;
}

function draw(): void {
  setBadge(badge(online, state));
  const detail = badgeDetail(state);
  badgeDetailEl.textContent = detail;
  badgeDetailEl.hidden = detail === '';
  const live = new Set<string>();
  if (state?.frame.width && state.frame.height) {
    svg.setAttribute('viewBox', `0 0 ${state.frame.width} ${state.frame.height}`);
    for (const tr of state.tracks) {
      if (!drawn(tr)) continue;
      live.add(tr.id);
      let d = drawnBoxes.get(tr.id);
      const before = d?.track; // what this box drew last time; nothing, for a card that is new on the player
      if (!d) {
        const poly = document.createElementNS(SVG, 'polygon');
        const text = document.createElementNS(SVG, 'text');
        text.setAttribute('class', 'rifteye-label');
        const id = tr.id;
        poly.addEventListener('pointerenter', (e) => {
          hovered = id;
          hoverAt = e;
          const cur = drawnBoxes.get(id);
          if (cur) showCard(cur.track, e);
        });
        poly.addEventListener('pointermove', (e) => {
          hoverAt = e;
          moveCard(e);
        });
        poly.addEventListener('pointerleave', () => {
          if (hovered === id) hovered = null;
          card.hidden = true;
        });
        // the pulse of a card just named is one turn of the animation: when it ends, the box is as any other again
        const endPulse = (): void => {
          const cur = drawnBoxes.get(id);
          if (cur) cur.pulsing = false;
          poly.classList.remove('rifteye-pulse');
        };
        poly.addEventListener('animationend', endPulse);
        poly.addEventListener('animationcancel', endPulse);
        svg.append(poly, text);
        d = { poly, text, track: tr, pulsing: false };
        drawnBoxes.set(tr.id, d);
      }
      const changed = d.track.state !== tr.state || d.track.name !== tr.name || d.track.under?.length !== tr.under?.length;
      d.track = tr;
      d.pulsing = becameNamed(before, tr) || (d.pulsing && tr.state === 'named'); // one turn of the pulse, and only while it is named
      d.poly.setAttribute('points', tr.quad.map((p) => `${p[0]},${p[1]}`).join(' '));
      d.poly.setAttribute('class', d.pulsing ? `${boxClass(tr)} rifteye-pulse` : boxClass(tr));
      const [x, y] = labelAnchor(tr.quad);
      d.text.setAttribute('x', String(x));
      d.text.setAttribute('y', String(y - 8));
      d.text.textContent = label(tr);
      if (changed && hovered === tr.id && hoverAt) showCard(tr, hoverAt); // what it says under the pointer changed
    }
  }
  for (const [id, d] of drawnBoxes) {
    if (live.has(id)) continue;
    d.poly.remove();
    d.text.remove();
    drawnBoxes.delete(id);
    if (hovered === id) {
      hovered = null;
      card.hidden = true;
    }
  }
}

function picture(printingId: string | null): Promise<ImageBitmap | null> {
  if (!printingId || !port) return Promise.resolve(null);
  let p = art.get(printingId);
  if (!p) {
    p = new Promise<string | null>((resolve) => {
      waiting.set(printingId, resolve);
      port?.postMessage({ kind: 'art', printing_id: printingId });
    }).then(async (jpeg) => {
      if (!jpeg) {
        art.delete(printingId); // not there this time: the next hover asks again
        return null;
      }
      const bytes = Uint8Array.from(atob(jpeg), (ch) => ch.charCodeAt(0));
      return createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
    });
    art.set(printingId, p);
  }
  return p;
}

/** A card's picture on a canvas: pages may forbid outside images, never drawing on a canvas. */
function artCanvas(printingId: string | null, width: number): HTMLCanvasElement {
  const c = el('canvas', 'rifteye-art');
  c.width = width;
  c.height = Math.round(width * 1.4);
  void picture(printingId).then((bm) => {
    if (!bm) return;
    c.height = Math.round((width * bm.height) / bm.width);
    c.getContext('2d')?.drawImage(bm, 0, 0, c.width, c.height);
  });
  return c;
}

function showCard(tr: Track, e: PointerEvent): void {
  const hc = hoverCard(tr);
  if (!hc) return;
  card.replaceChildren();
  if (hc.kind === 'named') {
    card.append(artCanvas(hc.printing_id, 200), el('div', 'rifteye-name', hc.name));
    if (hc.meta) card.append(el('div', 'rifteye-meta', hc.meta));
    card.append(el('div', 'rifteye-sure', hc.sure));
  } else if (hc.kind === 'unsure') {
    const row = el('div', 'rifteye-guesses');
    for (const g of hc.guesses) {
      const fig = el('figure', '');
      const caption = el('figcaption', '', `${g.name} · `);
      caption.append(el('span', 'rifteye-pct', `${Math.round(g.p * 100)}%`)); // the numbers in the mono face
      fig.append(artCanvas(g.printing_id, 104), caption);
      row.append(fig);
    }
    card.append(el('div', 'rifteye-note', 'Not sure yet. Best guesses:'), row);
  } else {
    card.append(el('div', 'rifteye-plain', hc.text));
  }
  if (hc.kind !== 'text' && hc.under) card.append(el('div', 'rifteye-under', hc.under));
  card.hidden = false;
  moveCard(e);
}

function moveCard(e: PointerEvent): void {
  const w = card.offsetWidth || 240;
  const h = card.offsetHeight || 340;
  const x = e.clientX + 18 + w > window.innerWidth ? e.clientX - 18 - w : e.clientX + 18;
  const y = Math.min(Math.max(8, e.clientY - h / 2), window.innerHeight - h - 8);
  card.style.left = `${Math.max(8, x)}px`;
  card.style.top = `${y}px`;
}

/** The boxes' lists for this video: another video starts without them, as its plays do. */
function listsNow(): string[] {
  if (listsVideo !== location.pathname) lists = [];
  return lists;
}

/** The lists the engine is given with each frame: the boxes that hold one. */
function filledLists(): string[] {
  return listsNow().filter((x) => x !== '');
}

/** What the plays panel shows of this tab. */
function snapshot(): Snapshot {
  log.add(location.pathname, []); // another video: its plays start over
  const read = (state as { lists?: unknown } | null)?.lists;
  return {
    kind: 'board',
    video: location.pathname,
    title: document.title,
    on: shown,
    live: !/^\/videos\/\d+/.test(location.pathname), // a live channel: its plays cannot be jumped to
    status: state?.status ?? '',
    message: state?.message ?? '',
    sides: sidesOf(state),
    plays: [...log.plays],
    // each box, with what the engine made of its list (the engine reads the filled boxes, in order)
    lists: listsNow().map((text, i) => {
      const k = listsNow().slice(0, i).filter((x) => x !== '').length;
      return { text, read: text && Array.isArray(read) ? ((read[k] as Snapshot['lists'][number]['read']) ?? null) : null };
    }),
  };
}

/** The board and the plays to the open plays panels: at once for a new play, else at most once a second. */
function sendBoard(now = false): void {
  if (panels.size === 0 || (!now && performance.now() - boardSent < BOARD_EVERY_MS)) return;
  boardSent = performance.now();
  const s = snapshot();
  for (const p of panels) {
    try {
      p.postMessage(s);
    } catch {
      panels.delete(p); // the panel was closed
    }
  }
}

/** A play clicked in the panel: the video goes to just before it (a replay; a live stream cannot go back). */
function seek(t: number): void {
  if (!video || !Number.isFinite(t) || !/^\/videos\/\d+/.test(location.pathname)) return;
  video.currentTime = Math.max(0, t - SEEK_BEFORE_S);
}

chrome.runtime.onConnect.addListener((p) => {
  if (p.name !== 'plays') return;
  panels.add(p);
  p.onDisconnect.addListener(() => panels.delete(p));
  p.onMessage.addListener((msg: { kind?: unknown; t?: unknown; texts?: unknown }) => {
    if (msg?.kind === 'seek' && typeof msg.t === 'number') seek(msg.t);
    else if (msg?.kind === 'lists' && Array.isArray(msg.texts)) {
      // the decklists pasted in the panel, for this video: they go with the next frame, and the panel hears what was made of them
      lists = msg.texts.slice(0, LIST_SLOTS).map((x) => (typeof x === 'string' ? x.trim().slice(0, 20_000) : ''));
      listsVideo = location.pathname;
      sendBoard(true);
    }
  });
  p.postMessage(snapshot());
});

/** Wardeye on or off in this tab: off, the overlay is hidden and no frame is read or sent; the board stays. Three ways lead here: Alt+R
 * (Option+R on a Mac), the badge's off button, and the toolbar button (the worker's 'toggle'). The worker is told, so the toolbar
 * button says OFF while it is off. */
function setOn(on: boolean): void {
  shown = on;
  place();
  chrome.runtime.sendMessage({ kind: 'switched', on }).catch(() => {}); // the worker may be asleep: it is told again next time
  sendBoard(true);
}

document.addEventListener('keydown', (e) => {
  if (e.altKey && (e.key === 'r' || e.key === 'R' || e.code === 'KeyR')) setOn(!shown);
});
offButton.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation(); // the player underneath must not take it as a click on the video
  setOn(false);
});
playsButton.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  chrome.runtime.sendMessage({ kind: 'open-panel' }).catch(() => {}); // the worker opens the browser's side panel on it
});
chrome.runtime.onMessage.addListener((msg: { kind?: unknown }, _sender, sendResponse) => {
  if (msg?.kind !== 'toggle') return;
  setOn(!shown);
  sendResponse({ on: shown });
});
setOn(true); // a page that loads again starts on, and the toolbar button says so

connect();
setInterval(tick, 100);
const follow = (): void => {
  place();
  requestAnimationFrame(follow);
};
requestAnimationFrame(follow);
