// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// A stand-in for the host the models are downloaded from (D-038), for the browser tests. It answers as Hugging Face does
// (measured on huggingface.co/ilDoc/wardeye-models): a file's address redirects to a CDN on another host, the redirect
// reflects the extension's origin in Access-Control-Allow-Origin, and the CDN answers with "*". So the extension needs no
// permission for either. Chromium is pointed here as it is for the fake Riot (fake-riot.ts): both hosts are mapped to this
// server, and its throwaway certificate is trusted.

import { createServer, type Server } from 'node:https';
import type { AddressInfo } from 'node:net';
import { throwawayCertificate } from './fake-riot';

export const MODELS_HOST = 'models.wardeye.test';
export const MODELS_URL = `https://${MODELS_HOST}/v1/`;
/** Where the host redirects each file, as Hugging Face sends its files to its CDN. */
export const CDN_HOST = 'cdn.wardeye.test';

export interface FakeModels {
  /** Chromium's arguments that send the host here. */
  args: string[];
  /** What it serves, by file name; change one to serve a file that is not what the package says. */
  files: Map<string, Uint8Array>;
  /** The paths it was asked for at MODELS_URL, in order (not the CDN's). */
  seen: string[];
  /** The paths the CDN was asked for, in order. */
  cdn: string[];
  close(): Promise<void>;
}

/** Starts the host, serving each of `models` (by its path in the package) at MODELS_URL followed by its file name. */
export async function fakeModels(models: Record<string, Uint8Array>): Promise<FakeModels> {
  const host: FakeModels = { args: [], files: new Map(Object.entries(models).map(([p, b]) => [p.split('/').pop()!, b])), seen: [], cdn: [], close: async () => {} };
  const server: Server = createServer(throwawayCertificate(), (req, res) => {
    const path = new URL(req.url ?? '/', MODELS_URL).pathname;
    if (String(req.headers.host ?? '').split(':')[0] === CDN_HOST) {
      host.cdn.push(path);
      const file = host.files.get(path.slice(1));
      if (!file) return void res.writeHead(404, { 'access-control-allow-origin': '*' }).end();
      return void res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': file.length, 'access-control-allow-origin': '*' }).end(file);
    }
    host.seen.push(path);
    const name = path.startsWith('/v1/') ? path.slice('/v1/'.length) : '';
    const allow = { 'access-control-allow-origin': String(req.headers.origin ?? '*') };
    if (!host.files.has(name)) return void res.writeHead(404, allow).end();
    res.writeHead(302, { ...allow, location: `https://${CDN_HOST}/${name}?signed=1` }).end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = (server.address() as AddressInfo).port;
  host.args = [`--host-resolver-rules=MAP ${MODELS_HOST} 127.0.0.1:${port}, MAP ${CDN_HOST} 127.0.0.1:${port}`, '--ignore-certificate-errors', '--no-proxy-server'];
  host.close = () =>
    new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
  return host;
}
