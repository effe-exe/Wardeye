// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors

import { describe, expect, it } from 'vitest';
import { DEFAULT_POWER, POWERS, POWER_MS, POWER_TEXT, asPower, paced } from '../src/power';

describe('power', () => {
  it('reads a stored level, and anything else as the default', () => {
    for (const p of POWERS) expect(asPower(p)).toBe(p);
    expect(DEFAULT_POWER).toBe('full');
    for (const x of [undefined, null, '', 'max', 3, {}]) expect(asPower(x)).toBe(DEFAULT_POWER);
  });

  it('never goes faster than the engine, and slows it down to the level', () => {
    expect(paced(200, 'full')).toBe(200);
    expect(paced(200, 'balanced')).toBe(500);
    expect(paced(200, 'light')).toBe(1000);
    expect(paced(2000, 'light')).toBe(2000);
  });

  it('gets lighter level by level, and names each one', () => {
    expect(POWER_MS.full).toBeLessThan(POWER_MS.balanced);
    expect(POWER_MS.balanced).toBeLessThan(POWER_MS.light);
    for (const p of POWERS) expect(POWER_TEXT[p].name.length).toBeGreaterThan(0);
  });
});
