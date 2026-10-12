import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { fetchFile, prune, type Download, type Fetching, type Keep } from '../src/downloads';

const hex = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
const file = (n: number, seed = 1) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) % 251);

function mapKeep(): Keep & { map: Map<string, Uint8Array> } {
  const map = new Map<string, Uint8Array>();
  return {
    map,
    get: async (k) => map.get(k) ?? null,
    put: async (k, b) => void map.set(k, b),
    keys: async () => [...map.keys()],
    delete: async (k) => void map.delete(k),
  };
}

/** A network that serves `body` in chunks of `chunk` bytes, and counts what it is asked. */
function net(body: Uint8Array, opts: { status?: number; chunk?: number } = {}) {
  const asked: string[] = [];
  const said: string[] = [];
  const f: Fetching = {
    fetch: async (url) => {
      asked.push(url);
      const chunk = opts.chunk ?? 1000;
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          for (let i = 0; i < body.length; i += chunk) c.enqueue(body.slice(i, i + chunk));
          c.close();
        },
      });
      return new Response(stream, { status: opts.status ?? 200 });
    },
    sha256: async (b) => hex(b),
    progress: (m) => said.push(m),
  };
  return { f, asked, said };
}

describe('a model file the package does not hold', () => {
  const bytes = file(12_000_000);
  const d: Download = { url: 'https://models.example/detector.onnx', sha256: hex(bytes), bytes: bytes.length };

  it('is downloaded once, checked, kept, and read from the keep the next time', async () => {
    const keep = mapKeep();
    const first = net(bytes, { chunk: 65536 });
    expect(hex(await fetchFile('models/detector.onnx', d, keep, first.f))).toBe(d.sha256); // a hash: toEqual on 12 MB is slow
    expect(first.asked).toEqual([d.url]);
    expect(hex(keep.map.get(d.sha256)!)).toBe(d.sha256);
    // the badge hears how far it is, a few times, not for every chunk
    expect(first.said[0]).toBe('downloading detector.onnx, the first time only: 0 of 12 MB');
    expect(first.said.length).toBeGreaterThan(1);
    expect(first.said.length).toBeLessThan(5);
    const again = net(bytes);
    expect(hex(await fetchFile('models/detector.onnx', d, keep, again.f))).toBe(d.sha256);
    expect(again.asked).toEqual([]); // no network
  });

  it('is refused and not kept when it is not the file the package names, or not all of it', async () => {
    const other = file(bytes.length, 7);
    for (const [body, error] of [
      [other, 'is not the file this version of Wardeye was made with'],
      [bytes.slice(0, 1000), 'came incomplete'],
      [file(bytes.length + 10), 'is bigger than this version of Wardeye expects'],
    ] as const) {
      const keep = mapKeep();
      await expect(fetchFile('models/detector.onnx', d, keep, net(body).f)).rejects.toThrow(error);
      expect(keep.map.size).toBe(0);
    }
  });

  it('says so when the host does not give it', async () => {
    await expect(fetchFile('models/detector.onnx', d, mapKeep(), net(bytes, { status: 404 }).f)).rejects.toThrow('detector.onnx could not be downloaded (HTTP 404)');
  });

  it('is downloaded again when the kept copy is not whole', async () => {
    const keep = mapKeep();
    keep.map.set(d.sha256, bytes.slice(0, 10));
    const n = net(bytes);
    expect(hex(await fetchFile('models/detector.onnx', d, keep, n.f))).toBe(d.sha256);
    expect(n.asked).toEqual([d.url]);
  });
});

describe('the kept files', () => {
  it('lose the ones the package no longer names (an update brought new models)', async () => {
    const keep = mapKeep();
    for (const k of ['old', 'new', 'other']) keep.map.set(k, file(4));
    await prune(keep, ['new']);
    expect([...keep.map.keys()]).toEqual(['new']);
  });
});
