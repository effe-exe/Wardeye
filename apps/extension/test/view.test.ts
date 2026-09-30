// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors

import { describe, expect, it } from 'vitest';
import { DEFAULT_VIEW, VIEWS, VIEW_TEXT, asView } from '../src/view';

describe('the view', () => {
  it('reads a stored view, and anything else as the default: outlines and names', () => {
    for (const v of VIEWS) expect(asView(v)).toBe(v);
    expect(DEFAULT_VIEW).toBe('full');
    for (const x of [undefined, null, '', 'hover', 1, {}]) expect(asView(x)).toBe(DEFAULT_VIEW);
  });

  it('names each view and says what it leaves on the video', () => {
    expect(VIEWS).toEqual(['full', 'marks', 'clean']);
    for (const v of VIEWS) {
      expect(VIEW_TEXT[v].name.length).toBeGreaterThan(0);
      expect(VIEW_TEXT[v].about.length).toBeGreaterThan(0);
    }
  });
});
