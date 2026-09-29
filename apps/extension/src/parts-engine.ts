// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The real parts: @rifteye/engine on onnxruntime-web. The only file that reaches into the engine for its detector,
// its embedder and its recogniser. It builds them as ml/rifteye_ml/live/__main__.py builds the live runner's:
// the trained detector's finder at --det-score 0.4, the ONNX embedder with embedder-v1's temperature, the gallery
// pyramid at the layout's levels, and a Recognizer for each table.

import { autolayout, detector, image, layouts, ort as engineOrt, recognizer, retrieval, type Layout, type RgbImage } from '@rifteye/engine';
import type { ModelFiles } from './assets';
import { pickLevels } from './levels';
import type { Board, BoardResult, BoardState, LoadContext, Parts } from './parts';

/** live/__main__.py EMBEDDER_T: how embedder-v1's scores become confidence (fitted on held-out Barcelona and Los
 * Angeles grand final crops; it belongs in the packed weights, but is here until it is). */
export const EMBEDDER_T = 0.0347;

export async function loadParts(ctx: LoadContext): Promise<Parts> {
  const { pkg, attempt, read, gallery, timer, decode, fps, progress, trace } = ctx;
  const ort = ctx.ort as unknown as engineOrt.OrtModule;
  // The runtime was chosen by the engine document: this worker holds the one build of onnxruntime-web it is for, so
  // a model that will not run on WebGPU is an error here (the document then starts the WASM worker), not a fallback.
  // Each model has its own precision (mode.ts): the runtime that opens it says which of its files it takes.
  const runtime = (precision: engineOrt.Precision): engineOrt.Runtime => ({ ep: attempt.runtime, precision, ort, wasm: ort, reason: `${attempt.runtime} ${precision}` });
  // A model's files as the engine asks for them. Given only a float32 file, the engine runs that on WebGPU too, even
  // on a GPU with shader-f16; a float16 file goes in both places, so that WASM (which takes the float32 slot) runs it.
  const filesOf = (m: ModelFiles, precision: engineOrt.Precision): engineOrt.ModelFiles => {
    const path = m[precision];
    if (!path) throw new Error(`${m.id} has no ${precision} file`);
    const file = { name: path.split('/').pop()!, bytes: () => read(path) };
    return precision === 'fp16' ? { fp32: file, fp16: file } : { fp32: file };
  };

  progress('loading the detector');
  const { detector: det, model: detModel } = await engineOrt.openDetector(runtime(attempt.detector), filesOf(pkg.detector, attempt.detector));
  progress('loading the embedder');
  const { encoder, model: encModel } = await engineOrt.openEncoder(runtime(attempt.embedder), filesOf(pkg.embedder, attempt.embedder));
  for (const m of [detModel, encModel]) if (m.fallback) throw new Error(m.fallback);
  // the gallery was embedded by the float32 file: the file that runs must be that one, or its float16 copy
  const want = attempt.embedder === 'fp16' ? gallery.index.fp16_sha256 : gallery.index.sha256;
  if (want && encModel.sha256 !== want) {
    throw new Error(`the embedder (${encModel.file}, sha256 ${encModel.sha256?.slice(0, 12)}) is not the one the gallery was made with (${want.slice(0, 12)})`);
  }
  // one run of each, so that the first frame does not wait for the GPU's shaders to compile
  progress('warming up the detector');
  await det.detectTiles([image.rgbImage(detector.TILE, detector.TILE)]);
  progress('warming up the embedder');
  await encoder.embed([image.rgbImage(64, 90)]);

  const pyramids = new Map<string, retrieval.Pyramid>();
  const pyramidFor = (layout: Layout): retrieval.Pyramid => {
    const levels = pickLevels(layouts.cardPx(layout, 1080), [...gallery.levels.keys()]);
    const key = levels.join(',');
    let p = pyramids.get(key);
    if (!p) {
      p = new retrieval.Pyramid(new Map(levels.map((l) => [l, gallery.levels.get(l)!])), gallery.index.dim);
      pyramids.set(key, p);
    }
    return p;
  };
  const embed = async (images: readonly RgbImage[]): Promise<Float32Array> => {
    const rows = await encoder.embed(images);
    trace?.embed(images.length, rows);
    return rows;
  };
  const timedEncoder = { name: encoder.name, dim: encoder.dim, embed: timer.wrap('embed', embed) };

  return {
    decode,
    findLayout: (frames) => autolayout.autoLayout(frames, timer.wrap('detect', async (frame, box, cardPx) => det.detect(frame, box, cardPx))),
    presets: () => Object.values(layouts.LAYOUTS),
    board(layout: Layout): Board {
      const find = detector.makeFinder(det, layout, { detScore: detector.DET_SCORE });
      const finder = timer.wrap('detect', async (t: number, frame: RgbImage) => {
        const boxes = await find(t, frame);
        trace?.boxes(t, boxes);
        return boxes;
      });
      const rec = new recognizer.Recognizer(layout, gallery.rows, timedEncoder, pyramidFor(layout), { title: layout.title, fps, finder, temperature: EMBEDDER_T });
      return {
        async step(t: number, frame: RgbImage): Promise<BoardResult> {
          const [state, events] = await rec.step(t, frame);
          return { state: state as unknown as BoardState, events };
        },
      };
    },
    async dispose() {
      await detModel.session.release();
      await encModel.session.release();
    },
  };
}
