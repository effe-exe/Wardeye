// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// Python's and numpy's number rules, where the ports need them exactly: round() goes to the even neighbour on a
// tie, % and // follow the divisor's sign, and np.median is the mean of the two middle values.

/** Python's round(x) and round(x, ndigits) for ndigits >= 0: the nearest value, and on an exact tie the even one
 * (JavaScript's Math.round goes up, and toFixed goes away from zero). With ndigits the decimal string is rounded
 * from the double's exact value, as CPython does, so round(2.675, 2) is 2.67 and round(0.25, 1) is 0.2. */
export function pyRound(x: number, ndigits = 0): number {
  if (!Number.isFinite(x)) return x;
  if (ndigits === 0) {
    const f = Math.floor(x);
    const d = x - f; // exact
    const r = d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
    return r + 0; // never -0
  }
  if (!(ndigits > 0)) throw new RangeError('pyRound: ndigits must be 0 or more');
  const a = Math.abs(x);
  if (a >= 4503599627370496) return x; // 2^52: no fraction left
  const digits = a.toFixed(100); // the double's exact expansion (a double below 2^52 needs at most 1074 digits, and these values far fewer)
  const dot = digits.indexOf('.');
  const frac = digits.slice(dot + 1);
  const head = `${digits.slice(0, dot)}.${frac.slice(0, ndigits)}`;
  const tie = /^50*$/.test(frac.slice(ndigits));
  const lastEven = (head.charCodeAt(head.length - 1) - 48) % 2 === 0;
  const r = Number(tie && lastEven ? head : a.toFixed(ndigits));
  return x < 0 || Object.is(x, -0) ? -r : r;
}

/** Python's a % b for numbers: the result has the sign of b. */
export function pyMod(a: number, b: number): number {
  const m = a % b;
  if (m === 0) return b < 0 ? -0 : 0;
  return m < 0 !== b < 0 ? m + b : m;
}

/** Python's a // b for numbers: the floor of the quotient, as CPython computes it. */
export function pyFloorDiv(a: number, b: number): number {
  let m = a % b;
  let div = (a - m) / b;
  if (m !== 0 && m < 0 !== b < 0) {
    m += b;
    div -= 1;
  }
  if (div === 0) return a / b < 0 || Object.is(a / b, -0) ? -0 : 0;
  const floor = Math.floor(div);
  return div - floor > 0.5 ? floor + 1 : floor;
}

/** np.median: the middle value, or the mean of the two middle values when the count is even. NaN when empty. */
export function median(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return NaN;
  const s = Float64Array.from(values).sort();
  const mid = n >> 1;
  return n % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}
