// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What Wardeye draws over the stream: the viewer's choice in the plays panel's settings, kept in the extension's storage
// beside the performance level. Pointing at a card shows it in every view; the view only decides what stays on the video.

export type View = 'full' | 'marks' | 'clean';
export const VIEWS: readonly View[] = ['full', 'marks', 'clean'];
export const DEFAULT_VIEW: View = 'full';

/** What the panel says of each view. */
export const VIEW_TEXT: Readonly<Record<View, { name: string; about: string }>> = {
  full: { name: 'Outlines and names', about: 'Each card Wardeye has named is marked, with its name above it.' },
  marks: { name: 'Outlines only', about: 'The cards are marked, without names. Point at one to see it.' },
  clean: { name: 'Clean', about: 'Nothing over the stream until you point at a card: then its outline and the card show.' },
};

/** A stored value as a view; anything else is the default. */
export function asView(x: unknown): View {
  return VIEWS.includes(x as View) ? (x as View) : DEFAULT_VIEW;
}
