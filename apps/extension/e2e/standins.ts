// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The private build's package with stand-ins for what is private, made at test time (nothing here is committed as a
// file): two tiny ONNX models, a gallery of three "cards" and their catalogue, hover pictures and standalone.json.
// The embedder is the mean colour of an 8 x 8 crop; the gallery's rows are colours.

import { encodeModel, floatToHalf, type TinyGraph } from '../../bench/src/tiny-onnx';
import { thumbName } from '../src/thumbs';

const ITEM = 3 * 8 * 8;
const shape = ['batch', 3, 8, 8];

/** y = x * 2 and z = flatten(x + 1): what the stand-in detector runs, for its timing and its output's shape. */
function detectorGraph(): TinyGraph {
  return {
    inputs: [{ name: 'x', type: 'float32', shape }],
    outputs: [
      { name: 'y', type: 'float32', shape },
      { name: 'z', type: 'float32', shape: ['batch', ITEM] },
    ],
    constants: [
      { name: 'two', type: 'float32', dims: [], values: [2] },
      { name: 'one', type: 'float32', dims: [], values: [1] },
    ],
    nodes: [
      { op: 'Mul', inputs: ['x', 'two'], outputs: ['y'] },
      { op: 'Add', inputs: ['x', 'one'], outputs: ['t'] },
      { op: 'Flatten', inputs: ['t'], outputs: ['z'], ints: { axis: 1 } },
    ],
  };
}

/** [mean red, mean green, mean blue, 0] / 255 of an 8 x 8 crop: flatten, then a 192 x 4 matrix of 1 / (64 * 255). The
 * float16 copy takes the crop in half precision, the matrix 1 / 64 (exact in half), and gives the means (0..255): the
 * page normalises the row, so the direction is what counts. */
function embedderGraph(half = false): TinyGraph {
  const w = new Array<number>(ITEM * 4).fill(0);
  for (let c = 0; c < 3; c++) for (let i = 0; i < 64; i++) w[(c * 64 + i) * 4 + c] = half ? 1 / 64 : 1 / (64 * 255);
  if (half) {
    return {
      inputs: [{ name: 'crops', type: 'float32', shape }],
      outputs: [{ name: 'embedding', type: 'float32', shape: ['batch', 4] }],
      constants: [{ name: 'w', type: 'float16', dims: [ITEM, 4], values: w }],
      nodes: [
        { op: 'Cast', inputs: ['crops'], outputs: ['h'], ints: { to: 10 } },
        { op: 'Flatten', inputs: ['h'], outputs: ['flat'], ints: { axis: 1 } },
        { op: 'MatMul', inputs: ['flat', 'w'], outputs: ['m'] },
        { op: 'Cast', inputs: ['m'], outputs: ['embedding'], ints: { to: 1 } },
      ],
    };
  }
  return {
    inputs: [{ name: 'crops', type: 'float32', shape }],
    outputs: [{ name: 'embedding', type: 'float32', shape: ['batch', 4] }],
    constants: [{ name: 'w', type: 'float32', dims: [ITEM, 4], values: w }],
    nodes: [
      { op: 'Flatten', inputs: ['crops'], outputs: ['flat'], ints: { axis: 1 } },
      { op: 'MatMul', inputs: ['flat', 'w'], outputs: ['embedding'] },
    ],
  };
}

const unit = (v: number[]): number[] => {
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};

/** The gallery's rows: gold (the first block), and two that the blue block cannot tell apart. */
const ROWS = [
  { printing_id: 'TST-001', card_id: 'test-unit', name: 'Test Unit', type: 'Unit', emb: unit([192 / 255, 160 / 255, 64 / 255, 0]) },
  { printing_id: 'TST-002', card_id: 'guess-two', name: 'Guess Two', type: 'Unit', emb: unit([0.3, 0.6, 0.75, 0]) },
  { printing_id: 'TST-003', card_id: 'guess-three', name: 'Guess Three', type: 'Spell', emb: unit([0.2, 0.65, 0.75, 0]) },
];

// A 16 x 16 grey JPEG: the hover picture of every card (a made-up picture, no card art in git).
const JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/wAALCAAQABABAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q==',
  'base64',
);

const json = (v: unknown): string => JSON.stringify(v);

export interface StandIn {
  runtime?: 'auto' | 'webgpu' | 'wasm' | 'companion';
  /** Leave out the models (a package whose files are missing). */
  noModels?: boolean;
  /** The embedder in float16 only, as the private build ships it (the detector is float32 either way). */
  embedderFp16?: boolean;
}

/** Every file of the package's private part, by its path in the package. */
export function standInFiles(opts: StandIn = {}): Record<string, Uint8Array | string> {
  const levels = [80, 90];
  const files: Record<string, Uint8Array | string> = {
    'standalone.json': json({
      format: 1,
      runtime: opts.runtime ?? 'auto',
      detector: { id: 'standin-detector', fp32: 'models/standin-detector.onnx' },
      embedder: { id: 'standin-embedder', ...(opts.embedderFp16 ? { fp16: 'models/standin-embedder.fp16.onnx' } : { fp32: 'models/standin-embedder.onnx' }) },
    }),
    'data/gallery/index.json': json({
      format: 1,
      encoder: 'onnx:standin-embedder',
      model: 'standin-embedder',
      sha256: '0'.repeat(64),
      fp16_sha256: null,
      dim: 4,
      dtype: 'float16',
      levels,
      rows: ROWS.map((r) => r.printing_id),
    }),
    'data/catalog.json': json(ROWS.map(({ emb: _emb, ...r }) => r)),
  };
  if (!opts.noModels) {
    files['models/standin-detector.onnx'] = encodeModel(detectorGraph());
    if (opts.embedderFp16) files['models/standin-embedder.fp16.onnx'] = encodeModel(embedderGraph(true));
    else files['models/standin-embedder.onnx'] = encodeModel(embedderGraph());
  }
  const half = new Uint8Array(ROWS.length * 4 * 2);
  const view = new DataView(half.buffer);
  ROWS.forEach((r, i) => r.emb.forEach((v, d) => view.setUint16((i * 4 + d) * 2, floatToHalf(v), true)));
  for (const lv of levels) files[`data/gallery/L${lv}.bin`] = half;
  for (const r of ROWS) files[`data/thumbs/${thumbName(r.printing_id)}.jpg`] = JPEG;
  return files;
}
