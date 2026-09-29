import { describe, expect, it } from 'vitest';
import { checkPairing, floatsOf, halfToFloat, loadGallery, loadLevels, parseCatalog, parseGalleryIndex, parsePackage, sha256Hex } from '../src/assets';
import { floatToHalf } from '../../bench/src/tiny-onnx';

const halves = (values: number[]): Uint8Array => {
  const out = new Uint8Array(values.length * 2);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => view.setUint16(i * 2, floatToHalf(v), true));
  return out;
};

const pkg = {
  format: 1,
  runtime: 'auto',
  detector: { id: 'detector-v0', fp16: 'models/detector-v0.fp16.onnx' },
  embedder: { id: 'embedder-v1', fp16: 'models/embedder-v1.fp16.onnx', fp32: 'models/embedder-v1.onnx' },
};

const index = {
  format: 1,
  encoder: 'onnx:embedder-v1-3f0a857c',
  model: 'embedder-v1',
  sha256: 'a'.repeat(64),
  fp16_sha256: 'b'.repeat(64),
  dim: 2,
  dtype: 'float16',
  levels: [80, 90],
  rows: ['A-1', 'B-2', 'C-3'],
};

const catalog = [
  { printing_id: 'A-1', card_id: 'a', name: 'A', type: 'Unit' },
  { printing_id: 'B-2', card_id: 'b', name: 'B', type: 'Spell' },
  { printing_id: 'C-3', card_id: 'c', name: 'C', type: '' },
];

describe('standalone.json', () => {
  it('names the models, and by default the data folder and the runtime', () => {
    expect(parsePackage(pkg)).toEqual({ ...pkg, data: 'data/' });
    expect(parsePackage({ ...pkg, data: 'gallery', runtime: 'wasm', layout: 'la-rq', fps: 2, threads: 2, trace: true })).toMatchObject({ data: 'gallery/', runtime: 'wasm', layout: 'la-rq', fps: 2, threads: 2, trace: true });
  });

  it('is refused, with the field at fault, when it says too little', () => {
    expect(() => parsePackage(null)).toThrow('not an object');
    expect(() => parsePackage({ ...pkg, format: 2 })).toThrow('format 2');
    expect(() => parsePackage({ ...pkg, runtime: 'gpu' })).toThrow('runtime must be');
    expect(() => parsePackage({ ...pkg, detector: { id: 'd' } })).toThrow('names no model file');
    expect(() => parsePackage({ ...pkg, embedder: undefined })).toThrow('embedder must be an object');
    expect(() => parsePackage({ ...pkg, layout: 3 })).toThrow('layout must be');
    expect(() => parsePackage({ ...pkg, fps: 0 })).toThrow('fps must be');
    expect(() => parsePackage({ ...pkg, threads: 0.5 })).toThrow('threads must be');
    expect(() => parsePackage({ ...pkg, trace: 'yes' })).toThrow('trace must be');
  });
});

describe('the gallery index and the catalogue', () => {
  it('read what web_assets.py writes', () => {
    expect(parseGalleryIndex(index)).toEqual(index);
    expect(parseGalleryIndex({ ...index, fp16_sha256: null }).fp16_sha256).toBeNull();
    expect(parseCatalog(catalog)).toEqual(catalog);
    expect(parseCatalog([{ printing_id: 'A-1', card_id: 'a', name: 'A' }])[0]!.type).toBe('');
  });

  it('are refused when they are not what the extension reads', () => {
    expect(() => parseGalleryIndex({ ...index, dtype: 'float64' })).toThrow('dtype must be float16 or float32');
    expect(() => parseGalleryIndex({ ...index, levels: [] })).toThrow('levels must be');
    expect(() => parseGalleryIndex({ ...index, rows: [1] })).toThrow('rows must be');
    expect(() => parseCatalog([])).toThrow('not a list');
    expect(() => parseCatalog([{ printing_id: 'A-1', name: 'A' }])).toThrow('row 0 has no card_id');
  });

  it('go together row for row', () => {
    const idx = parseGalleryIndex(index);
    expect(() => checkPairing(idx, parseCatalog(catalog))).not.toThrow();
    expect(() => checkPairing(idx, parseCatalog(catalog.slice(0, 2)))).toThrow('3 rows and the catalogue 2');
    expect(() => checkPairing(idx, parseCatalog([catalog[1]!, catalog[0]!, catalog[2]!]))).toThrow("row 0 is A-1, the catalogue's is B-2");
  });
});

describe('float16 files', () => {
  it('are read as the float32 values they hold', () => {
    const values = [0, 1, -1, 0.5, 0.099975586, 65504, 6.1035156e-5, 5.9604645e-8, -0.33325195];
    const got = halfToFloat(halves(values));
    values.forEach((v, i) => expect(got[i]).toBeCloseTo(v, 6));
    expect(Object.is(halfToFloat(Uint8Array.from([0x00, 0x80]))[0], -0)).toBe(true);
    expect(halfToFloat(Uint8Array.from([0x00, 0x7c]))[0]).toBe(Infinity);
    expect(halfToFloat(Uint8Array.from([0x01, 0x7e]))[0]).toBeNaN();
  });

  it('round-trip every finite value exactly', () => {
    const all: number[] = [];
    for (let h = 0; h < 0x7c00; h++) all.push(halfToFloat(Uint8Array.from([h & 0xff, h >> 8]))[0]!);
    const back = halfToFloat(halves(all));
    expect(Array.from(back)).toEqual(all);
  });

  it('are refused when their length cannot be', () => {
    expect(() => halfToFloat(Uint8Array.from([1, 2, 3]))).toThrow('odd number');
    expect(() => floatsOf(Uint8Array.from([1, 2, 3]))).toThrow('multiple of 4');
  });

  it('come as float32 too, from bytes that need not be aligned', () => {
    const f = new Float32Array([0.1, -2.5, 3e-9]);
    const bytes = new Uint8Array(f.buffer.byteLength + 1);
    bytes.set(new Uint8Array(f.buffer), 1);
    expect(Array.from(floatsOf(bytes.subarray(1)))).toEqual(Array.from(f));
  });
});

describe('loading the gallery', () => {
  const files: Record<string, Uint8Array> = {
    'data/gallery/index.json': new TextEncoder().encode(JSON.stringify(index)),
    'data/catalog.json': new TextEncoder().encode(JSON.stringify(catalog)),
    'data/gallery/L80.bin': halves([1, 0, 0, 1, 0.6, 0.8]),
    'data/gallery/L90.bin': halves([0.6, 0.8, 1, 0, 0, 1]),
  };
  const read = async (path: string): Promise<Uint8Array> => {
    const f = files[path];
    if (!f) throw new Error(`${path}: HTTP 404`);
    return f;
  };

  it('gives each level as rows x dim float32, with the catalogue', async () => {
    const g = await loadGallery(read, 'data/');
    expect(g.rows.map((r) => r.printing_id)).toEqual(['A-1', 'B-2', 'C-3']);
    expect([...g.levels.keys()]).toEqual([80, 90]);
    expect(Array.from(g.levels.get(80)!)).toEqual([1, 0, 0, 1, expect.closeTo(0.6, 3), expect.closeTo(0.8, 3)]);
    expect(g.levels.get(90)!.length).toBe(6);
  });

  it('reads a float32 gallery as it is', async () => {
    const f32 = (v: number[]) => new Uint8Array(new Float32Array(v).buffer);
    const g = await loadGallery(
      async (p) => (p.endsWith('index.json') ? new TextEncoder().encode(JSON.stringify({ ...index, dtype: 'float32' })) : p.endsWith('L80.bin') ? f32([1, 0, 0, 1, 0.6, 0.8]) : p.endsWith('L90.bin') ? f32([0.6, 0.8, 1, 0, 0, 1]) : files[p]!),
      'data/',
    );
    expect(Array.from(g.levels.get(80)!)).toEqual(Array.from(new Float32Array([1, 0, 0, 1, 0.6, 0.8])));
  });

  it('is refused when a level has the wrong size, or is missing', async () => {
    await expect(loadLevels(async (p) => (p.endsWith('L80.bin') ? halves([1, 2]) : files[p]!), 'data/', parseGalleryIndex(index))).rejects.toThrow('L80.bin is 4 bytes, 3 rows of 2 float16 are 12');
    await expect(loadGallery(async (p) => (p.endsWith('L90.bin') ? read('nope') : read(p)), 'data/')).rejects.toThrow('HTTP 404');
  });

  it('can tell a file by its hash', async () => {
    expect(await sha256Hex(new TextEncoder().encode('abc'))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });
});
