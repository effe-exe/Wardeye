// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// How hard Wardeye works: the viewer's choice in the plays panel's settings, kept in the extension's storage (the only thing
// it stores). It sets the least time between two frames the overlay sends: fewer frames, less of the graphics chip and the
// processor, for a computer that runs other things too. The board is read the same way; it only follows the table more slowly.

export type Power = 'full' | 'balanced' | 'light';
export const POWERS: readonly Power[] = ['full', 'balanced', 'light'];
export const DEFAULT_POWER: Power = 'full';

/** The least ms between two frames at each level: full is the engine's own pace (5 a second), balanced 2, light 1. */
export const POWER_MS: Readonly<Record<Power, number>> = { full: 0, balanced: 500, light: 1000 };

/** What the panel says of each level. */
export const POWER_TEXT: Readonly<Record<Power, { name: string; about: string }>> = {
  full: { name: 'Full', about: 'Up to 5 frames a second: the fastest to follow the table.' },
  balanced: { name: 'Balanced', about: '2 frames a second: about half the work.' },
  light: { name: 'Light', about: '1 frame a second: for a computer busy with other apps. Cards are named more slowly.' },
};

/** A stored value as a level; anything else is the default. */
export function asPower(x: unknown): Power {
  return POWERS.includes(x as Power) ? (x as Power) : DEFAULT_POWER;
}

/** The time to wait between frames: the engine's pace, or the level's, whichever is longer. */
export function paced(engineMs: number, power: Power): number {
  return Math.max(engineMs, POWER_MS[power]);
}
