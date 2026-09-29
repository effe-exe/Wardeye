// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The worker's side of standalone mode: whether there is an engine here (the private build's package, an engine
// document that can run), asking it for a tab's frame, and the card pictures: from the package (the developer build), or
// from Riot's gallery through the engine document (the store build). When it cannot serve, it says so and the worker uses
// the live runner instead (the store build has none: it tells the overlay, `cannotReadState`); it asks again after a while.

import { parsePackage, type StandalonePackage } from './assets';
import { b64Of } from './base64';
import type { State } from './geometry';
import { RETRY_AFTER_MS } from './mode';
import { isEngineReply, type EngineReply, type EngineRequest, type FrameMessage } from './protocol';
import { thumbPath } from './thumbs';

/** What the worker has that a test can stand in for. */
export interface Env {
  /** A clock in ms. */
  now(): number;
  /** A file of the package, or null when it is not there. */
  read(path: string): Promise<Uint8Array | null>;
  /** Makes sure the engine document exists. */
  ensureDocument(): Promise<void>;
  /** A request to the engine document, and what it answers. */
  send(request: EngineRequest): Promise<unknown>;
}

/** How long the document may take to answer: at once to `hello` and `forget`; a frame may take long on the WASM build
 * (finding the layout is dozens of detector runs), and the document has its own limit for a hung engine. */
const ANSWER_MS = 30_000;
const FRAME_ANSWER_MS = 5 * 60_000;

function within<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`no answer in ${ms} ms`)), ms);
    p.then(resolve, reject).finally(() => clearTimeout(timer));
  });
}

/** How long the overlay waits before it sends another frame while the store build has no engine (ms). */
export const UNAVAILABLE_RETRY_MS = 5000;

/** What the overlay is told when the store build cannot read a frame: it has no live runner to hand the frame to, so the
 * badge says plainly why (geometry.badge shows an error state's message). `missing`: the package is not there at all. */
export function cannotReadState(t: number, missing: boolean): State {
  return {
    t,
    status: 'error',
    message: missing ? "the engine's files are missing from this install, reinstall Wardeye" : 'this browser cannot run the engine (WebGPU or WebAssembly needed)',
    frame: { width: 0, height: 0 },
    tracks: [],
    retry_ms: UNAVAILABLE_RETRY_MS,
  };
}

export class Standalone {
  private pkg: Promise<StandalonePackage | null> | null = null;
  private offUntil = 0;
  /** Why the engine was last written off (for the console). */
  lastReason = '';

  /** `store`: the Chrome Web Store build. It has no live runner, so a package that says `companion` is asked like any other
   * (the engine document says it has nothing to run), and a card's picture comes from Riot's gallery, through the document. */
  constructor(
    private readonly env: Env,
    private readonly retryMs = RETRY_AFTER_MS,
    private readonly store = false,
  ) {}

  /** standalone.json's package, or null when this build has none (the public build) or it cannot be read. Read once. */
  package(): Promise<StandalonePackage | null> {
    this.pkg ??= (async () => {
      const bytes = await this.env.read('standalone.json');
      if (!bytes) return null;
      try {
        return parsePackage(JSON.parse(new TextDecoder().decode(bytes)));
      } catch (e) {
        console.warn(`Wardeye: standalone.json cannot be used${this.store ? '' : ', the live runner is used instead'}: ${e instanceof Error ? e.message : String(e)}`);
        return null;
      }
    })();
    return this.pkg;
  }

  /** There is an engine to ask: the package has one (and does not say to use the live runner), and it was not written
   * off a moment ago. */
  async usable(): Promise<boolean> {
    const pkg = await this.package();
    return pkg !== null && (pkg.runtime !== 'companion' || this.store) && this.env.now() >= this.offUntil;
  }

  private writeOff(reason: string): void {
    this.lastReason = reason;
    this.offUntil = this.env.now() + this.retryMs;
    const then = this.store ? 'the overlay says so' : 'the live runner is used';
    console.warn(`Wardeye: no engine in this browser (${reason}); ${then}, and the engine tried again in ${Math.round(this.retryMs / 60000)} min`);
  }

  /** One request to the engine document, made again once if the document is gone or does not answer. */
  private async ask(request: EngineRequest): Promise<EngineReply | null> {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        await this.env.ensureDocument();
        const reply = await within(this.env.send(request), request.kind === 'frame' ? FRAME_ANSWER_MS : ANSWER_MS);
        if (isEngineReply(reply)) return reply;
      } catch {
        // the document went away or did not answer: make it again
      }
    }
    return null;
  }

  /** A Twitch tab connected: the engine's document is made ready (its models load with the first frame). */
  async warm(): Promise<void> {
    if (!(await this.usable())) return;
    const reply = await this.ask({ target: 'engine', kind: 'hello' });
    if (reply?.kind !== 'hello') this.writeOff(reply?.kind === 'unavailable' ? reply.reason : 'the engine document did not answer');
  }

  /** The state for a tab's frame: `{ state }` (null: keep the board), or null when the engine cannot serve it. */
  async frame(tab: number, msg: FrameMessage): Promise<{ state: State | null } | null> {
    if (!(await this.usable())) return null;
    const reply = await this.ask({ target: 'engine', kind: 'frame', tab, t: msg.t, video: msg.video, jpeg: msg.jpeg });
    if (reply?.kind === 'state') return { state: reply.state };
    this.writeOff(reply?.kind === 'unavailable' ? reply.reason : 'the engine document did not answer');
    return null;
  }

  /** A tab went away: its board is let go. Nothing is made for it if the document is not there. */
  async forget(tab: number): Promise<void> {
    if (!(await this.usable())) return;
    try {
      await within(this.env.send({ target: 'engine', kind: 'forget', tab }), ANSWER_MS);
    } catch {
      // no document: no board
    }
  }

  /** A card's hover picture, base64; null when there is none. From the package (the developer build), or, in the store build, from
   * Riot's gallery: the engine document holds the card list and fetches the picture. */
  async art(printingId: string): Promise<string | null> {
    if (!(await this.package())) return null;
    if (this.store) {
      const reply = await this.ask({ target: 'engine', kind: 'art', printing_id: printingId });
      return reply?.kind === 'art' ? reply.jpeg : null;
    }
    const bytes = await this.env.read(thumbPath(printingId));
    return bytes ? b64Of(bytes) : null;
  }
}
