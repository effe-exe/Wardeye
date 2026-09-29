// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A fake of Riot's public card gallery for the store build's browser tests: its card list (content.publishing.riotgames.com)
// and its image server (cmsassets.rgpub.io), with made-up cards and pictures generated in the browser, so that no Riot data
// is in git and the tests reach no real network.
//
// The engine document is not a page Playwright can route (context.route does not see its requests), so the browser is
// pointed here instead: Chromium maps the two hosts to this server (--host-resolver-rules), trusts its throwaway
// certificate (--ignore-certificate-errors) and makes no use of a proxy (--no-proxy-server). The extension's own requests
// then go the way they go for real: https to the two hosts, under the manifest's host permissions. Like the real image
// server, this one sends no CORS header, so a picture that arrives arrives by that permission.

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer, type Server } from 'node:https';
import type { IncomingHttpHeaders } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Page } from '@playwright/test';
import { FEED_URL, IMAGE_ORIGIN } from '../src/feed';

const FEED_HOST = new URL(FEED_URL).host;
const FEED_PATH = new URL(FEED_URL).pathname;
const IMAGE_HOST = new URL(IMAGE_ORIGIN).host;

/** An item of the gallery's list, as Riot's has them (made up; `text` is the card's rules text, which the extension never keeps). */
export function item(publicCode: string, name: string, over: Record<string, unknown> = {}): Record<string, unknown> {
  const slug = publicCode.toLowerCase().replace(/[^a-z0-9]/g, '-');
  return {
    publicCode,
    name,
    cardType: { type: [{ id: 'unit', label: 'Unit' }] },
    cardImage: { url: `${IMAGE_ORIGIN}/fake/${slug}.png?accountingTag=RB` },
    set: { value: { id: 'TST', label: 'Test Set' } },
    rarity: { value: { id: 'common', label: 'Common' } },
    domain: { values: [{ id: 'fury', label: 'Fury' }] },
    text: { richText: { body: 'Made-up rules text.' } },
    ...over,
  };
}

/** The path a made-up card's picture is served at (the address `item` gives it). */
export const pictureOf = (publicCode: string): string => `/fake/${publicCode.toLowerCase().replace(/[^a-z0-9]/g, '-')}.png`;

export interface Seen {
  host: string;
  /** Path and query. */
  url: string;
  headers: IncomingHttpHeaders;
}

export interface FakeRiot {
  /** Chromium's arguments that send Riot's two hosts here. */
  args: string[];
  /** Answer the card list with a 503 (it is down), or with the list again. */
  down: boolean;
  /** The pictures, by path (see `pictureOf`): what the image server answers with; anything else it answers 404. */
  pictures: Map<string, Buffer>;
  /** The requests it has had, in order. */
  seen: Seen[];
  close(): Promise<void>;
}

let certificate: { key: string; cert: string } | null = null;

/** A certificate for the fake server, made once with the openssl command and thrown away with the folder. */
function throwawayCertificate(): { key: string; cert: string } {
  if (certificate) return certificate;
  const dir = mkdtempSync(join(tmpdir(), 'wardeye-fake-riot-'));
  try {
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', join(dir, 'key.pem'), '-out', join(dir, 'cert.pem'), '-days', '2', '-subj', '/CN=fake-riot'], { stdio: 'ignore' });
    certificate = { key: readFileSync(join(dir, 'key.pem'), 'utf8'), cert: readFileSync(join(dir, 'cert.pem'), 'utf8') };
    return certificate;
  } catch (e) {
    throw new Error(`the store build's browser tests need the openssl command, for a throwaway certificate (${e instanceof Error ? e.message : String(e)})`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Starts the fake: `items` are what the card list holds, served a page at a time as the real one is; or a function that answers a page
 * (`from`, `limit`) with its whole body, to serve a recording. */
export async function fakeRiot(items: readonly unknown[] | ((from: number, limit: number) => unknown)): Promise<FakeRiot> {
  const riot: FakeRiot = { args: [], down: false, pictures: new Map(), seen: [], close: async () => {} };
  const server: Server = createServer(throwawayCertificate(), (req, res) => {
    const host = String(req.headers.host ?? '').split(':')[0]!;
    const url = new URL(req.url ?? '/', 'https://fake.invalid');
    riot.seen.push({ host, url: `${url.pathname}${url.search}`, headers: req.headers });
    if (host === FEED_HOST && url.pathname === FEED_PATH) {
      if (riot.down) {
        res.writeHead(503, { 'content-type': 'text/plain' }).end('the card gallery is down');
        return;
      }
      const from = Number(url.searchParams.get('from') ?? 0);
      const limit = Number(url.searchParams.get('limit') ?? 200);
      const origin = req.headers.origin; // the real one sends Access-Control-Allow-Origin reflecting the extension's origin
      const body = typeof items === 'function' ? items(from, limit) : { metadata: { totalItems: items.length, totalPages: Math.ceil(items.length / limit) }, data: items.slice(from, from + limit) };
      res
        .writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-cache', ...(origin ? { 'access-control-allow-origin': origin } : {}) })
        .end(JSON.stringify(body));
    } else if (host === IMAGE_HOST) {
      const jpeg = riot.pictures.get(url.pathname);
      if (jpeg) res.writeHead(200, { 'content-type': 'image/jpeg', 'content-length': jpeg.length }).end(jpeg); // no CORS header: the real image server sends none
      else res.writeHead(404).end();
    } else {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = (server.address() as AddressInfo).port;
  riot.args = [`--host-resolver-rules=MAP ${FEED_HOST} 127.0.0.1:${port}, MAP ${IMAGE_HOST} 127.0.0.1:${port}`, '--ignore-certificate-errors', '--no-proxy-server'];
  riot.close = () =>
    new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
  return riot;
}

/** A solid-colour JPEG of `width` x `height`, made in the browser (no picture is in git). */
export async function makeJpeg(page: Page, colour: string, width = 400, height = 559): Promise<Buffer> {
  const b64 = await page.evaluate(
    async ({ colour, width, height }) => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = colour;
      ctx.fillRect(0, 0, width, height);
      const blob = await new Promise<Blob>((resolve) => canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.9));
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let bin = '';
      for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      return btoa(bin);
    },
    { colour, width, height },
  );
  return Buffer.from(b64, 'base64');
}
