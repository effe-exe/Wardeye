// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// Where a card's hover picture is in the package: data/thumbs/<name>.jpg. The name is the printing id with what a
// file name may not hold (the `*` of OGN-299*) written as `_` and the byte's two hex digits: the same as
// ml/rifteye_ml/web_assets.thumb_name, which makes the files.

const SAFE = /[A-Za-z0-9.-]/;

export function thumbName(printingId: string): string {
  let out = '';
  for (const ch of printingId) {
    if (SAFE.test(ch)) out += ch;
    else for (const b of new TextEncoder().encode(ch)) out += `_${b.toString(16).padStart(2, '0')}`;
  }
  return out;
}

/** The picture's path inside the package. */
export function thumbPath(printingId: string): string {
  return `data/thumbs/${thumbName(printingId)}.jpg`;
}
