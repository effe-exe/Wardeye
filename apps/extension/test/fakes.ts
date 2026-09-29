// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Stand-ins for the engine's parts, for the host's unit tests: a picture, a board that says what it was given, and
// parts that record what they are asked.

import { layouts, type Layout, type RgbImage } from '@rifteye/engine';
import type { Board, BoardResult, Parts } from '../src/parts';

export const LA = layouts.LAYOUTS['la-rq'];

/** A picture of one colour. */
export function picture(width = 64, height = 36, rgb: readonly [number, number, number] = [0, 0, 0]): RgbImage {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < data.length; i += 3) data.set(rgb, i);
  return { width, height, data };
}

/** A "JPEG" the fake decoder reads: the picture's size, then its colour. */
export const jpegOf = (width = 64, height = 36, rgb: readonly [number, number, number] = [0, 0, 0]): Uint8Array => Uint8Array.from([width, height, ...rgb]);

export interface StepLog {
  board: number;
  t: number;
}

/** Parts that record what they are asked: how often a layout was looked for, the boards made, the steps taken. */
export class FakeParts implements Parts {
  /** What findLayout answers; null until the test sets one. */
  found: Layout | null = null;
  finds: number[] = []; // the number of frames each findLayout call was given
  made: Layout[] = [];
  steps: StepLog[] = [];
  disposed = false;
  presetList: readonly Layout[] = Object.values(layouts.LAYOUTS);
  /** Set to make the next board step throw. */
  fail: Error | null = null;
  /** Run inside decode and inside a board's step, to spend time. */
  hooks: { decode?: () => void; step?: () => void } = {};

  async decode(jpeg: Uint8Array): Promise<RgbImage> {
    this.hooks.decode?.();
    return picture(jpeg[0], jpeg[1], [jpeg[2] ?? 0, jpeg[3] ?? 0, jpeg[4] ?? 0]);
  }

  async findLayout(frames: readonly RgbImage[]): Promise<Layout | null> {
    this.finds.push(frames.length);
    return this.found;
  }

  presets(): readonly Layout[] {
    return this.presetList;
  }

  board(layout: Layout): Board {
    const id = this.made.length;
    this.made.push(layout);
    return {
      step: async (t: number, frame: RgbImage): Promise<BoardResult> => {
        if (this.fail) throw this.fail;
        this.hooks.step?.();
        this.steps.push({ board: id, t });
        return {
          state: { t, status: 'live', message: '', title: layout.title, frame: { width: frame.width, height: frame.height }, tracks: [] },
          events: [],
        };
      },
    };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
