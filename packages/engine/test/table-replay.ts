// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// The detector, replaced by what Python's said: a callback for the layout finder that answers each call with the answer
// the recorded call had, and throws when it is asked something else. Free of Node's modules, so the browser tests
// bundle it as well.

import type { Detect } from '../src/autolayout';
import type { Detection, RgbImage } from '../src/types';

/** One call of Python's detector as test/gen/table.py and test/gen/table_fixtures.py keep it: which frame, the
 * window in px, a card's long side in px, and the cards it found (flat corners x0, y0, ..., x3, y3). */
export interface DetectorCall {
  frame: number;
  box: number[];
  px: number;
  out: { cls?: string; score: number; quad: number[] }[];
}

export function replayDetector(calls: readonly DetectorCall[], frames: readonly RgbImage[]): { detect: Detect; used: () => number } {
  let n = 0;
  const detect: Detect = (frame, win, px) => {
    const call = calls[n++];
    if (!call) throw new Error(`the detector was asked ${n} times and Python asked ${calls.length}`);
    if (frames.indexOf(frame) !== call.frame) throw new Error(`call ${n}: frame ${frames.indexOf(frame)}, Python's was ${call.frame}`);
    if (win.length !== call.box.length || win.some((v, i) => !Object.is(v, call.box[i]))) throw new Error(`call ${n}: window ${win}, Python's was ${call.box}`);
    if (!Object.is(px, call.px)) throw new Error(`call ${n}: card size ${px}, Python's was ${call.px}`);
    return call.out.map(
      (d): Detection => ({
        cls: (d.cls ?? 'card') as 'card',
        score: d.score,
        box: [0, 0, 0, 0],
        quad: [
          [d.quad[0]!, d.quad[1]!],
          [d.quad[2]!, d.quad[3]!],
          [d.quad[4]!, d.quad[5]!],
          [d.quad[6]!, d.quad[7]!],
        ],
        found: [1, 1, 1, 1],
        visible: [1, 1, 1, 1],
      }),
    );
  };
  return { detect, used: () => n };
}
