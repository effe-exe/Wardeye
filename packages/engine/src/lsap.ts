// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
//
// scipy.optimize.linear_sum_assignment, ported from SciPy 1.17.1 (scipy/optimize/_lsap.c and
// scipy/optimize/rectangular_lsap/rectangular_lsap.cpp, BSD-3-Clause, notice at the end of this file). It is
// the shortest augmenting path algorithm of Crouse, "On implementing 2D rectangular assignment algorithms"
// (IEEE TAES 52(4), 2016), so a tie or a rectangular matrix comes out as SciPy has it, and the arithmetic is
// SciPy's own doubles.

/** A cost matrix: an array of rows, or its values row by row with the shape. */
export type CostMatrix = readonly (readonly number[])[] | { readonly rows: number; readonly cols: number; readonly data: ArrayLike<number> };

/** Row indices (ascending) and the column each is assigned, like scipy's linear_sum_assignment. There are
 * min(rows, cols) pairs. Throws Error("cost matrix is infeasible") when a full assignment needs an infinite
 * cost, and Error("matrix contains invalid numeric entries") for NaN, or -Infinity (+Infinity when maximizing). */
export function linearSumAssignment(cost: CostMatrix, maximize = false): [number[], number[]] {
  let nr: number;
  let nc: number;
  let flat: Float64Array;
  if (Array.isArray(cost)) {
    const rows = cost as readonly (readonly number[])[];
    nr = rows.length;
    nc = nr ? rows[0]!.length : 0;
    flat = new Float64Array(nr * nc);
    for (let i = 0; i < nr; i++) {
      const row = rows[i]!;
      if (row.length !== nc) throw new Error('expected a matrix (2-D array), got a ragged array');
      for (let j = 0; j < nc; j++) flat[i * nc + j] = row[j]!;
    }
  } else {
    const m = cost as { rows: number; cols: number; data: ArrayLike<number> };
    nr = m.rows;
    nc = m.cols;
    if (m.data.length !== nr * nc) throw new Error(`a ${nr} x ${nc} cost matrix needs ${nr * nc} values, not ${m.data.length}`);
    flat = Float64Array.from(m.data);
  }
  const n = Math.min(nr, nc);
  const a = new Array<number>(n);
  const b = new Array<number>(n);
  solve(nr, nc, flat, maximize, a, b);
  return [a, b];
}

/** The C++ `solve`: fills a and b, throws when the matrix is invalid or infeasible. */
function solve(nr: number, nc: number, cost: Float64Array, maximize: boolean, a: number[], b: number[]): void {
  // handle trivial inputs
  if (nr === 0 || nc === 0) return;

  // tall rectangular cost matrix must be transposed
  const transpose = nc < nr;

  // make a copy of the cost matrix if we need to modify it
  if (transpose || maximize) {
    const temp = new Float64Array(nr * nc);
    if (transpose) {
      for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) temp[j * nr + i] = cost[i * nc + j]!;
      [nr, nc] = [nc, nr];
    } else {
      temp.set(cost);
    }
    // negate cost matrix for maximization
    if (maximize) for (let i = 0; i < nr * nc; i++) temp[i] = -temp[i]!;
    cost = temp;
  }

  // test for NaN and -inf entries
  for (let i = 0; i < nr * nc; i++) {
    const c = cost[i]!;
    if (c !== c || c === -Infinity) throw new Error('matrix contains invalid numeric entries');
  }

  // initialize variables
  const u = new Float64Array(nr);
  const v = new Float64Array(nc);
  const shortestPathCosts = new Float64Array(nc);
  const path = new Int32Array(nc).fill(-1);
  const col4row = new Int32Array(nr).fill(-1);
  const row4col = new Int32Array(nc).fill(-1);
  const SR = new Uint8Array(nr);
  const SC = new Uint8Array(nc);
  const remaining = new Int32Array(nc);

  // iteratively build the solution
  for (let curRow = 0; curRow < nr; curRow++) {
    const [sink, minVal] = augmentingPath(nc, cost, u, v, path, row4col, shortestPathCosts, curRow, SR, SC, remaining);
    if (sink < 0) throw new Error('cost matrix is infeasible');

    // update dual variables
    u[curRow]! += minVal;
    for (let i = 0; i < nr; i++) {
      if (SR[i] && i !== curRow) u[i]! += minVal - shortestPathCosts[col4row[i]!]!;
    }
    for (let j = 0; j < nc; j++) {
      if (SC[j]) v[j]! -= minVal - shortestPathCosts[j]!;
    }

    // augment previous solution
    let j = sink;
    for (;;) {
      const i = path[j]!;
      row4col[j] = i;
      const swap = col4row[i]!;
      col4row[i] = j;
      j = swap;
      if (i === curRow) break;
    }
  }

  if (transpose) {
    // col4row holds distinct rows of the original matrix, so its argsort has no ties
    const order = Array.from({ length: nr }, (_, i) => i).sort((i, j) => col4row[i]! - col4row[j]!);
    order.forEach((v2, i) => {
      a[i] = col4row[v2]!;
      b[i] = v2;
    });
  } else {
    for (let i = 0; i < nr; i++) {
      a[i] = i;
      b[i] = col4row[i]!;
    }
  }
}

/** The C++ `augmenting_path`: the shortest path from row i to a free column, as [sink, minVal]; sink is -1
 * when there is none. */
function augmentingPath(
  nc: number,
  cost: Float64Array,
  u: Float64Array,
  v: Float64Array,
  path: Int32Array,
  row4col: Int32Array,
  shortestPathCosts: Float64Array,
  i: number,
  SR: Uint8Array,
  SC: Uint8Array,
  remaining: Int32Array,
): [number, number] {
  let minVal = 0;

  // Crouse's pseudocode uses set complements to keep track of remaining nodes. Here we use an array.
  let numRemaining = nc;
  for (let it = 0; it < nc; it++) {
    // Filling this up in reverse order ensures that the solution of a constant cost matrix is the identity
    // matrix (SciPy issue 11602).
    remaining[it] = nc - it - 1;
  }

  SR.fill(0);
  SC.fill(0);
  shortestPathCosts.fill(Infinity);

  // find shortest augmenting path
  let sink = -1;
  while (sink === -1) {
    let index = -1;
    let lowest = Infinity;
    SR[i] = 1;

    for (let it = 0; it < numRemaining; it++) {
      const j = remaining[it]!;

      const r = minVal + cost[i * nc + j]! - u[i]! - v[j]!;
      if (r < shortestPathCosts[j]!) {
        path[j] = i;
        shortestPathCosts[j] = r;
      }

      // When multiple nodes have the minimum cost, we select one which gives us a new sink node. This is
      // particularly important for integer cost matrices with small co-efficients.
      if (shortestPathCosts[j]! < lowest || (shortestPathCosts[j] === lowest && row4col[j] === -1)) {
        lowest = shortestPathCosts[j]!;
        index = it;
      }
    }

    minVal = lowest;
    if (minVal === Infinity) return [-1, minVal]; // infeasible cost matrix

    const j = remaining[index]!;
    if (row4col[j] === -1) sink = j;
    else i = row4col[j]!;

    SC[j] = 1;
    remaining[index] = remaining[--numRemaining]!;
  }

  return [sink, minVal];
}

// SciPy's licence, which covers the algorithm above (rectangular_lsap.cpp, by PM Larsen):
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//    following disclaimer in the documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
//    promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
