// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// What the browser tests share to check the overlay's look: the brand's tokens as the browser reports them, computed styles,
// the fonts that have loaded, and a log of every CSS animation that starts on the page.

import { readFileSync } from 'node:fs';
import type { Page } from '@playwright/test';

const TOKENS = readFileSync(new URL('../../../assets/brand/tokens.css', import.meta.url), 'utf8');

/** A colour token as the stylesheet writes it: #rrggbb. */
export function hex(token: string): string {
  const m = new RegExp(`--wd-${token}:\\s*(#[0-9a-f]{6})`, 'i').exec(TOKENS);
  if (!m) throw new Error(`no colour token --wd-${token} in assets/brand/tokens.css`);
  return m[1]!.toLowerCase();
}

/** A colour token as getComputedStyle reports it: rgb(r, g, b). */
export function rgb(token: string): string {
  const h = hex(token);
  return `rgb(${parseInt(h.slice(1, 3), 16)}, ${parseInt(h.slice(3, 5), 16)}, ${parseInt(h.slice(5, 7), 16)})`;
}

/** Some computed properties of the first element `selector` matches (custom properties by their name). */
export function styleOf(page: Page, selector: string, props: string[]): Promise<Record<string, string>> {
  return page.evaluate(
    ([sel, names]) => {
      const s = getComputedStyle(document.querySelector(sel)!);
      return Object.fromEntries(names!.map((n) => [n, s.getPropertyValue(n).trim()]));
    },
    [selector, props] as const,
  );
}

/** The families of the fonts the page has loaded: what the overlay's stylesheet fetched from the extension's own files. */
export function loadedFonts(page: Page): Promise<string[]> {
  return page.evaluate(() => [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family.replace(/["']/g, '')).sort());
}

export interface AnimationSeen {
  /** The keyframes' name. */
  name: string;
  /** The class of the element it started on. */
  on: string;
  ms: number;
  iterations: number;
}

/** Records every CSS animation that starts on the pages the tab opens from now on (call it before the page is opened). */
export async function logAnimations(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const log: AnimationSeen[] = [];
    (window as unknown as { __wardeyeAnimations: AnimationSeen[] }).__wardeyeAnimations = log;
    document.addEventListener(
      'animationstart',
      (e) => {
        const target = e.target as Element;
        const running = target.getAnimations().find((a) => (a as CSSAnimation).animationName === e.animationName);
        const timing = running?.effect?.getTiming();
        log.push({ name: e.animationName, on: target.getAttribute('class') ?? '', ms: Number(timing?.duration), iterations: Number(timing?.iterations) });
      },
      true,
    );
  });
}

export function animationsSeen(page: Page): Promise<AnimationSeen[]> {
  return page.evaluate(() => (window as unknown as { __wardeyeAnimations: AnimationSeen[] }).__wardeyeAnimations.slice());
}
