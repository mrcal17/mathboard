// Tests for static/graph/linalg.js. Run from the project dir: node --test tests/linalg.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as L from '../static/graph/linalg.js';

// ---------------------------------------------------------------- helpers

const TOL = 1e-9;
function close(a, b, tol = TOL, msg = '') {
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${msg} ${a} ≉ ${b}`);
}
function closeV(u, v, tol = TOL, msg = '') {
  assert.equal(u.length, v.length, `${msg} length`);
  u.forEach((x, i) => close(x, v[i], tol, `${msg}[${i}]`));
}
function closeM(A, B, tol = TOL, msg = '') {
  assert.equal(A.length, B.length, `${msg} rows`);
  A.forEach((r, i) => closeV(r, B[i], tol, `${msg}[${i}]`));
}
const dot = (u, v) => u.reduce((s, x, i) => s + x * v[i], 0);
const norm = v => Math.sqrt(dot(v, v));
const T = A => A[0].map((_, j) => A.map(r => r[j]));
const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0)));
const mv = (A, v) => A.map(r => dot(r, v));
const diag = (d, m = d.length, n = d.length) =>
  Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? d[i] ?? 0 : 0)));
const isOrthonormalCols = (Q, tol = 1e-9) => closeM(mul(T(Q), Q), L.identity(Q[0].length), tol, 'QᵀQ');
const isUpper = R => R.forEach((r, i) => r.forEach((x, j) => { if (j < i) assert.equal(x, 0, `R[${i}][${j}]`); }));
function deepFreeze(x) {
  if (Array.isArray(x)) { x.forEach(deepFreeze); Object.freeze(x); }
  return x;
}
// Deterministic PRNG (mulberry32)
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randInt = (r, n, lo = -4, hi = 4) =>
  Array.from({ length: n }, () => Array.from({ length: n }, () => lo + Math.floor(r() * (hi - lo + 1))));
const randReal = (r, m, n = m) => Array.from({ length: m }, () => Array.from({ length: n }, () => 2 * r() - 1));
const rot2 = t => [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
const rotZ = t => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];

// Full consistency check of an eig() result against A.
function checkEig(A, E, tol = 1e-7) {
  const n = A.length;
  const s = Math.max(1, ...A.flat().map(Math.abs));
  assert.equal(E.values.length, n, 'n eigenvalues');
  const nReal = E.values.filter(v => v.im === 0).length;
  assert.equal(E.vectors.length, nReal, 'one vectors entry per real value');
  assert.equal(E.planes.length * 2, n - nReal, 'one plane per complex pair');
  // real ones first, descending; then complex pairs
  for (let i = 1; i < nReal; i++) assert.ok(E.values[i - 1].re >= E.values[i].re, 'descending');
  for (let i = nReal; i < n; i++) assert.notEqual(E.values[i].im, 0, 'complex after real');
  for (let i = 0; i < nReal; i++) {
    const lam = E.values[i].re;
    assert.ok(E.vectors[i].length >= 1, 'at least one eigenvector');
    for (const v of E.vectors[i]) {
      close(norm(v), 1, 1e-9, 'unit');
      closeV(mv(A, v), v.map(x => lam * x), tol * s, `A v = λ v (λ=${lam})`);
    }
  }
  for (const { value: { re: a, im: b }, basis: [p, q] } of E.planes) {
    assert.ok(b > 0);
    closeV(mv(A, p), p.map((x, i) => a * x + b * q[i]), tol * s, 'A p = a p + b q');
    closeV(mv(A, q), q.map((x, i) => -b * p[i] + a * x), tol * s, 'A q = −b p + a q');
    close(norm(p), 1, 1e-9, '|p| = 1');
    close(dot(p, q), 0, 1e-9, 'p ⟂ q');
  }
  // trace = Σλ, det = Πλ
  const tr = A.reduce((t, r, i) => t + r[i], 0);
  close(E.values.reduce((t, v) => t + v.re, 0), tr, 1e-7, 'trace');
  let pr = [1, 0];
  for (const { re, im } of E.values) pr = [pr[0] * re - pr[1] * im, pr[0] * im + pr[1] * re];
  close(pr[0], L.det(A), 1e-6, 'det');
}

// ---------------------------------------------------------------- basics

describe('basic operations', () => {
  test('identity, transpose, matmul, matvec, add, scale, trace', () => {
    assert.deepEqual(L.identity(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    assert.deepEqual(L.transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
    assert.deepEqual(L.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]]);
    assert.deepEqual(L.matmul([[1, 2, 3]], [[1], [1], [1]]), [[6]]);
    assert.deepEqual(L.matvec([[1, 2], [3, 4], [5, 6]], [1, -1]), [-1, -1, -1]);
    assert.deepEqual(L.add([[1, 2]], [[3, -2]]), [[4, 0]]);
    assert.deepEqual(L.scale([[1, -2], [0, 3]], -2), [[-2, 4], [0, -6]]);
    assert.equal(L.trace([[1, 9], [9, 4]]), 5);
  });

  test('outputs snap near-integers and never contain -0', () => {
    assert.deepEqual(L.scale([[0.1, 0.7]], 10), [[1, 7]]); // 0.7*10 = 7.000000000000001
    assert.deepEqual(L.matmul([[0.1, 0.2]], [[10], [10]]), [[3]]); // 3.0000000000000004
    const r = L.scale([[0, 1]], -1);
    assert.ok(Object.is(r[0][0], 0), 'no -0');
  });

  test('bad input throws short errors', () => {
    assert.throws(() => L.det([[1, 2, 3], [4, 5, 6]]), /square/);
    assert.throws(() => L.trace([[1, 2]]), /square/);
    assert.throws(() => L.matmul([[1, 2]], [[1, 2]]), /Dimension mismatch/);
    assert.throws(() => L.matvec([[1, 2]], [1, 2, 3]), /Dimension mismatch/);
    assert.throws(() => L.add([[1]], [[1, 2]]), /Dimension mismatch/);
    assert.throws(() => L.inv([[1, 2], [2, 4]]), /singular/);
    assert.throws(() => L.inv([[0, 0], [0, 0]]), /singular/);
    assert.throws(() => L.rank([[1, 2], [3]]), /same length/);
    assert.throws(() => L.rank([[1, NaN]]), /finite/);
    assert.throws(() => L.rank([]), /non-empty/);
    assert.throws(() => L.identity(0), /positive/);
    assert.throws(() => L.solve([[1, 2], [3, 4]], [1]), /Dimension mismatch/);
    assert.throws(() => L.eig([[1, 2, 3], [4, 5, 6]]), /square/);
    assert.throws(() => L.eig(L.identity(4)), /3×3/);
    assert.throws(() => L.gramSchmidt([[1, 2], [1, 2, 3]]), /Dimension mismatch/);
  });

  test('no function mutates its inputs', () => {
    const A = deepFreeze([[2, 1, 0], [1, 3, 1], [0, 1, 4]]);
    const W = deepFreeze([[0, 2, 4, 1], [1, 1, 1, 1], [2, 0, -2, 1]]);
    const b = deepFreeze([1, 2, 3]);
    const all = [
      () => L.transpose(A), () => L.matmul(A, A), () => L.matvec(A, b), () => L.add(A, A), () => L.scale(A, 2),
      () => L.trace(A), () => L.det(A), () => L.inv(A), () => L.rref(W), () => L.eliminationSteps(W),
      () => L.rank(W), () => L.nullspace(W), () => L.colspace(W), () => L.rowspace(W), () => L.leftNullspace(W),
      () => L.solve(A, b), () => L.lstsq(W, b), () => L.gramSchmidt(A), () => L.qr(W), () => L.lu(W),
      () => L.charpoly(A), () => L.eig(A), () => L.svd(W), () => L.svd(T(W)),
    ];
    for (const f of all) f();
    assert.deepEqual(A, [[2, 1, 0], [1, 3, 1], [0, 1, 4]]);
  });
});

describe('det / inv', () => {
  test('det small and 4×4', () => {
    assert.equal(L.det([[1, 2], [3, 4]]), -2);
    assert.equal(L.det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]), -306);
    assert.equal(L.det([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0);
    assert.equal(L.det([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]), 0);
    const U4 = [[2, 1, 0, 3], [0, 3, 1, 1], [0, 0, 4, 2], [0, 0, 0, 5]];
    assert.equal(L.det(U4), 120);
    assert.equal(L.det([U4[1], U4[0], U4[2], U4[3]]), -120);
    assert.equal(L.det([[1, 2, 3, 4], [2, 4, 6, 8], [0, 1, 0, 1], [1, 0, 1, 0]]), 0);
  });

  test('inv gives clean fractions and A·A⁻¹ = I', () => {
    assert.deepEqual(L.inv([[2, 1], [1, 1]]), [[1, -1], [-1, 2]]);
    closeM(L.inv([[4, 7], [2, 6]]), [[0.6, -0.7], [-0.2, 0.4]]);
    const A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]];
    closeM(L.matmul(A, L.inv(A)), L.identity(3));
    closeM(L.inv(A), [[0.75, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.75]]);
    const B = [[1e-3, 2e-3], [3e-3, 1e-3]];
    closeM(mul(B, L.inv(B)), L.identity(2), 1e-9);
  });
});

// ---------------------------------------------------------------- elimination

describe('rref and elimination steps', () => {
  const W = [[1, 2, 1, 4], [2, 4, 0, 2], [3, 6, 1, 6]];

  test('rref of a rank-2 3×4 matrix', () => {
    const { R, pivots } = L.rref(W);
    assert.deepEqual(R, [[1, 2, 0, 1], [0, 0, 1, 3], [0, 0, 0, 0]]);
    assert.deepEqual(pivots, [0, 2]);
  });

  test('rref of tall, wide, zero and full-rank matrices', () => {
    assert.deepEqual(L.rref([[1, 2], [2, 4], [3, 6]]), { R: [[1, 2], [0, 0], [0, 0]], pivots: [0] });
    assert.deepEqual(L.rref([[0, 0, 0], [0, 0, 0]]), { R: [[0, 0, 0], [0, 0, 0]], pivots: [] });
    assert.deepEqual(L.rref([[0, 3, 6], [2, 4, 2]]), { R: [[1, 0, -3], [0, 1, 2]], pivots: [0, 1] });
    assert.deepEqual(L.rref([[2, 1], [1, 3]]).R, [[1, 0], [0, 1]]);
  });

  test('rank uses a relative tolerance', () => {
    assert.equal(L.rank([[0.1, 0.2], [0.3, 0.6]]), 1);
    assert.equal(L.rank([[1, 1], [1, 1 + 1e-12]]), 1);
    assert.equal(L.rank([[1, 1], [1, 1 + 1e-6]]), 2);
    assert.equal(L.rank([[1e-12, 2e-12], [3e-12, 1e-12]]), 2); // tiny but well-conditioned
    assert.equal(L.rank([[1e6, 2e6], [2e6, 4e6 + 1e-5]]), 1);
    assert.equal(L.rank([[0, 0], [0, 0]]), 0);
  });

  test('steps for [[2,4],[1,3]] read like hand work', () => {
    const steps = L.eliminationSteps([[2, 4], [1, 3]]);
    assert.deepEqual(steps.map(s => s.op), ['start', 'scale', 'add', 'add']);
    assert.deepEqual(steps[0].M, [[2, 4], [1, 3]]);
    assert.equal(steps[1].latex, 'R_1 \\leftarrow \\tfrac{1}{2}R_1');
    assert.equal(steps[1].text, 'R1 ← (1/2)R1');
    assert.deepEqual(steps[1].M, [[1, 2], [1, 3]]);
    assert.equal(steps[2].latex, 'R_2 \\leftarrow R_2 - R_1');
    assert.equal(steps[3].latex, 'R_1 \\leftarrow R_1 - 2R_2');
    assert.equal(steps[3].text, 'R1 ← R1 - 2R2');
    assert.deepEqual(steps.at(-1).M, [[1, 0], [0, 1]]);
  });

  test('steps: negative fractional scale and swap on a zero pivot', () => {
    const s1 = L.eliminationSteps([[1, 2], [3, 4]]);
    assert.deepEqual(s1.slice(1).map(s => s.latex), [
      'R_2 \\leftarrow R_2 - 3R_1',
      'R_2 \\leftarrow -\\tfrac{1}{2}R_2',
      'R_1 \\leftarrow R_1 - 2R_2',
    ]);
    assert.equal(s1[2].text, 'R2 ← -(1/2)R2');
    const s2 = L.eliminationSteps([[0, 1], [2, 0]]);
    assert.equal(s2[1].op, 'swap');
    assert.equal(s2[1].latex, 'R_1 \\leftrightarrow R_2');
    assert.equal(s2[1].text, 'R1 ↔ R2');
    assert.deepEqual(s2[1].rows, [0, 1]);
    // first nonzero pivot, not the largest: no swap for [[1,..],[5,..]]
    assert.notEqual(L.eliminationSteps([[1, 2], [5, 3]])[1].op, 'swap');
    // pretty coefficients: thirds, and a decimal when no simple fraction exists
    const s3 = L.eliminationSteps([[3, 1], [2, 1]]);
    assert.equal(s3[1].latex, 'R_1 \\leftarrow \\tfrac{1}{3}R_1');
    assert.equal(s3[2].latex, 'R_2 \\leftarrow R_2 - 2R_1');
    const s4 = L.eliminationSteps([[1, 0], [Math.SQRT2, 1]]);
    assert.equal(s4[1].latex, 'R_2 \\leftarrow R_2 - 1.414R_1');
  });

  test('each recorded op reproduces the next matrix; final matrix is the RREF', () => {
    const cases = [W, [[0, 2, 4], [1, 1, 1], [2, 0, -2]], [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]],
      [[0, 0, 1], [0, 3, 2]], [[1, 2], [3, 4], [5, 6]], [[0.5, 0.25], [0.2, 0.7]]];
    for (const M of cases) {
      const steps = L.eliminationSteps(M);
      assert.equal(steps[0].op, 'start');
      for (let k = 1; k < steps.length; k++) {
        const prev = steps[k - 1].M.map(r => r.slice());
        const { op, rows, factor } = steps[k];
        if (op === 'swap') [prev[rows[0]], prev[rows[1]]] = [prev[rows[1]], prev[rows[0]]];
        else if (op === 'scale') prev[rows[0]] = prev[rows[0]].map(x => factor * x);
        else prev[rows[0]] = prev[rows[0]].map((x, j) => x + factor * prev[rows[1]][j]);
        closeM(steps[k].M, prev, 1e-9, `step ${k}`);
      }
      closeM(steps.at(-1).M, L.rref(M).R);
    }
    // Gaussian elimination classic: x = 2, y = 3, z = -1
    assert.deepEqual(L.eliminationSteps([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]).at(-1).M,
      [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]]);
  });
});

// ---------------------------------------------------------------- subspaces

describe('four fundamental subspaces', () => {
  const W = [[1, 2, 1, 4], [2, 4, 0, 2], [3, 6, 1, 6]];

  test('3×4 rank-2 example', () => {
    assert.equal(L.rank(W), 2);
    assert.deepEqual(L.nullspace(W), [[-2, 1, 0, 0], [-1, 0, -3, 1]]);
    assert.deepEqual(L.colspace(W), [[1, 2, 3], [1, 0, 1]]);
    assert.deepEqual(L.rowspace(W), [[1, 2, 0, 1], [0, 0, 1, 3]]);
    assert.deepEqual(L.leftNullspace(W), [[-1, -1, 1]]);
  });

  test('dimension counts and orthogonality on assorted shapes', () => {
    const cases = [W, T(W), [[1, 2], [2, 4]], [[1, 2, 3]], [[1], [2], [3]], L.identity(3), [[0, 0], [0, 0]],
      [[1, 1, 0], [0, 1, 1], [1, 2, 1]], [[2, -1, 0, 1], [4, -2, 1, 3]]];
    for (const A of cases) {
      const m = A.length, n = A[0].length, r = L.rank(A);
      const N = L.nullspace(A), LN = L.leftNullspace(A);
      assert.equal(N.length, n - r);
      assert.equal(LN.length, m - r);
      assert.equal(L.colspace(A).length, r);
      assert.equal(L.rowspace(A).length, r);
      for (const v of N) closeV(mv(A, v), new Array(m).fill(0));
      for (const y of LN) closeV(mv(T(A), y), new Array(n).fill(0));
      for (const row of L.rowspace(A)) for (const v of N) close(dot(row, v), 0);
    }
  });

  test('rank-1 and full-rank edge cases', () => {
    assert.deepEqual(L.nullspace([[1, 2], [2, 4]]), [[-2, 1]]);
    assert.deepEqual(L.nullspace(L.identity(2)), []);
    assert.deepEqual(L.nullspace([[0, 0], [0, 0]]), [[1, 0], [0, 1]]);
    assert.deepEqual(L.leftNullspace([[1, 2], [2, 4]]), [[-2, 1]]);
  });
});

// ---------------------------------------------------------------- solve / lstsq

describe('solve', () => {
  test('unique', () => {
    const r = L.solve([[2, 1], [1, 3]], [3, 5]);
    assert.equal(r.kind, 'unique');
    closeV(r.x, [0.8, 1.4]);
    assert.deepEqual(L.solve([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3]), { kind: 'unique', x: [2, 3, -1] });
  });

  test('infinite (underdetermined and singular square)', () => {
    const r = L.solve([[1, 2, 3], [2, 4, 6]], [6, 12]);
    assert.equal(r.kind, 'infinite');
    assert.deepEqual(r.x, [6, 0, 0]);
    assert.deepEqual(r.directions, [[-2, 1, 0], [-3, 0, 1]]);
    const A = [[1, 1, 1], [1, 2, 3]], b = [3, 6];
    const s = L.solve(A, b);
    assert.equal(s.kind, 'infinite');
    closeV(mv(A, s.x), b);
    for (const d of s.directions) closeV(mv(A, d), [0, 0]);
    const q = L.solve([[1, 2], [2, 4]], [3, 6]);
    assert.equal(q.kind, 'infinite');
    assert.deepEqual(q.directions, [[-2, 1]]);
  });

  test('inconsistent', () => {
    assert.deepEqual(L.solve([[1, 1], [1, 1]], [1, 2]), { kind: 'none' });
    assert.deepEqual(L.solve([[1, 0], [0, 1], [1, 1]], [1, 1, 3]), { kind: 'none' });
    assert.deepEqual(L.solve([[0, 0]], [1]), { kind: 'none' });
    assert.equal(L.solve([[1, 0], [0, 1], [1, 1]], [1, 1, 2]).kind, 'unique'); // consistent tall system
  });
});

describe('lstsq', () => {
  test('overdetermined line fit (Strang: points (0,6), (1,0), (2,0))', () => {
    const A = [[1, 0], [1, 1], [1, 2]], b = [6, 0, 0];
    const r = L.lstsq(A, b);
    assert.deepEqual(r.x, [5, -3]);
    assert.deepEqual(r.projection, [5, 2, -1]);
    assert.deepEqual(r.residual, [1, -2, 1]);
    for (let j = 0; j < 2; j++) close(dot(r.residual, A.map(row => row[j])), 0); // e ⟂ C(A)
  });

  test('matches the normal equations on a random tall system', () => {
    const r = rng(7), A = randReal(r, 8, 3), b = Array.from({ length: 8 }, () => r());
    const { x } = L.lstsq(A, b);
    closeV(mv(mul(T(A), A), x), mv(T(A), b), 1e-9);
  });

  test('rank-deficient and underdetermined give the minimum-norm solution', () => {
    const r = L.lstsq([[1, 1], [1, 1]], [2, 0]);
    closeV(r.x, [0.5, 0.5]);
    closeV(r.projection, [1, 1]);
    closeV(r.residual, [1, -1]);
    closeV(L.lstsq([[1, 1]], [2]).x, [1, 1]);
    closeV(L.lstsq([[1, 2, 2]], [9]).x, [1, 2, 2]);
    closeV(L.lstsq([[0, 0], [0, 0]], [1, 2]).x, [0, 0]);
    closeV(L.lstsq([[2, 1], [1, 3]], [3, 5]).x, [0.8, 1.4]); // square invertible = solve
  });
});

// ---------------------------------------------------------------- orthogonality

describe('gramSchmidt / qr / lu', () => {
  test('gramSchmidt skips dependent vectors and records steps', () => {
    const { basis, steps } = L.gramSchmidt([[1, 1, 0], [1, 0, 1], [2, 1, 1], [0, 0, 5]]);
    assert.equal(basis.length, 3);
    closeM(mul(basis, T(basis)), L.identity(3));
    assert.equal(steps.length, 4);
    assert.deepEqual(steps[0].projections, []);
    assert.deepEqual(steps[1].input, [1, 0, 1]);
    closeV(steps[1].projections[0], [0.5, 0.5, 0]);
    closeV(steps[1].orthogonal, [0.5, -0.5, 1]);
    closeV(steps[1].unit, [0.5, -0.5, 1].map(x => x / Math.sqrt(1.5)));
    assert.equal(steps[2].unit, null); // (2,1,1) = (1,1,0) + (1,0,1)
    assert.deepEqual(steps[2].orthogonal, [0, 0, 0]);
    // input = Σ projections + orthogonal part
    for (const s of steps) {
      const sum = s.projections.reduce((acc, p) => acc.map((x, i) => x + p[i]), s.orthogonal);
      closeV(sum, s.input);
    }
    assert.deepEqual(L.gramSchmidt([]), { basis: [], steps: [] });
    assert.equal(L.gramSchmidt([[0, 0]]).steps[0].unit, null);
  });

  test('qr: Q R = A, QᵀQ = I, R upper triangular with nonnegative diagonal', () => {
    const cases = [[[1, 1, 0], [1, 0, 1], [0, 1, 1]], [[3, 1], [4, 2], [0, 5]], [[1, 2, 3], [4, 5, 6]],
      [[1, 2], [2, 4]], [[0, 1], [1, 0]], [[1, 2, 3], [2, 4, 6], [1, 0, 1]], [[-2, 1], [0, -3]], [[5]]];
    for (const A of cases) {
      const { Q, R } = L.qr(A);
      const k = Math.min(A.length, A[0].length);
      assert.equal(Q.length, A.length);
      assert.equal(Q[0].length, k);
      assert.equal(R.length, k);
      closeM(mul(Q, R), A);
      isOrthonormalCols(Q);
      isUpper(R);
      for (let i = 0; i < k; i++) assert.ok(R[i][i] >= 0);
    }
    const { Q, R } = L.qr([[3, 1], [4, 2]]);
    closeM(Q, [[0.6, -0.8], [0.8, 0.6]]);
    closeM(R, [[5, 2.2], [0, 0.4]]);
  });

  test('lu: P A = L U with row exchanges only when needed', () => {
    const a = L.lu([[2, 1], [4, 3]]);
    assert.deepEqual(a, { P: [[1, 0], [0, 1]], L: [[1, 0], [2, 1]], U: [[2, 1], [0, 1]] });
    const b = L.lu([[0, 1], [1, 1]]);
    assert.deepEqual(b.P, [[0, 1], [1, 0]]);
    assert.deepEqual(b.U, [[1, 1], [0, 1]]);
    const cases = [[[1, 2, 3], [2, 4, 5], [1, 3, 4]], [[1, 2], [2, 4]], [[0, 0], [0, 1]], [[1, 2], [3, 4], [5, 6]],
      [[0, 2, 1], [3, 1, 1]], [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]];
    for (const A of cases) {
      const { P, L: Lo, U } = L.lu(A);
      closeM(mul(P, A), mul(Lo, U));
      Lo.forEach((r, i) => { assert.equal(r[i], 1); r.forEach((x, j) => { if (j > i) assert.equal(x, 0); }); });
      isUpper(U);
    }
    assert.deepEqual(L.lu([[1, 2], [2, 4]]).U, [[1, 2], [0, 0]]);
  });
});

// ---------------------------------------------------------------- charpoly / eig

describe('charpoly', () => {
  test('known polynomials', () => {
    assert.deepEqual(L.charpoly([[1, 2], [3, 4]]), [1, -5, -2]);
    assert.deepEqual(L.charpoly([[0, -1], [1, 0]]), [1, 0, 1]);
    assert.deepEqual(L.charpoly(L.identity(3)), [1, -3, 3, -1]);
    assert.deepEqual(L.charpoly([[2, 0, 0], [0, 3, 4], [0, 4, 9]]), [1, -14, 35, -22]);
    assert.deepEqual(L.charpoly([[7]]), [1, -7]);
    const A = [[1, 2, 0], [3, -1, 4], [2, 2, 5]];
    const c = L.charpoly(A);
    assert.equal(c[1], -L.trace(A));
    assert.equal(c[3], -L.det(A));
  });
});

describe('eig', () => {
  test('rotations have complex eigenvalues and a rotation plane', () => {
    const E = L.eig([[0, -1], [1, 0]]);
    assert.deepEqual(E.values, [{ re: 0, im: 1 }, { re: 0, im: -1 }]);
    assert.deepEqual(E.vectors, []);
    assert.deepEqual(E.planes, [{ value: { re: 0, im: 1 }, basis: [[1, 0], [0, 1]] }]);
    const t = Math.PI / 6, R = rot2(t), F = L.eig(R);
    close(F.values[0].re, Math.cos(t));
    close(F.values[0].im, Math.sin(t));
    closeM(F.planes[0].basis, [[1, 0], [0, 1]]);
    checkEig(R, F);
    // rotation-scaling and a non-circular (elliptical) case
    checkEig([[1, -1], [1, 1]], L.eig([[1, -1], [1, 1]]));
    const G = L.eig([[0, -2], [1, 0]]);
    closeV(G.planes[0].basis[0], [1, 0]);
    closeV(G.planes[0].basis[1], [0, Math.SQRT1_2]);
    checkEig([[0, -2], [1, 0]], G);
  });

  test('3D rotation: real axis + complex plane', () => {
    const t = Math.PI / 3, R = rotZ(t), E = L.eig(R);
    assert.equal(E.values[0].re, 1);
    assert.equal(E.values[0].im, 0);
    assert.deepEqual(E.vectors[0], [[0, 0, 1]]);
    close(E.values[1].re, 0.5);
    close(E.values[1].im, Math.sin(t));
    close(E.values[2].im, -Math.sin(t));
    closeM(E.planes[0].basis, [[1, 0, 0], [0, 1, 0]]);
    checkEig(R, E);
    // rotation about (1,1,1) by 120° permutes the axes
    const P = [[0, 0, 1], [1, 0, 0], [0, 1, 0]], EP = L.eig(P);
    closeV(EP.vectors[0][0], [1, 1, 1].map(x => x / Math.sqrt(3)));
    for (const v of EP.planes[0].basis) close(dot(v, [1, 1, 1]), 0);
    checkEig(P, EP);
  });

  test('shears are defective', () => {
    const E = L.eig([[1, 1], [0, 1]]);
    assert.deepEqual(E.values, [{ re: 1, im: 0 }, { re: 1, im: 0 }]);
    assert.deepEqual(E.vectors, [[[1, 0]], [[1, 0]]]);
    assert.deepEqual(E.planes, []);
    const S = [[1, 0], [2.5, 1]];
    assert.deepEqual(L.eig(S).vectors[0], [[0, 1]]);
    const J = [[2, 1, 0], [0, 2, 1], [0, 0, 2]], EJ = L.eig(J);
    assert.deepEqual(EJ.values.map(v => v.re), [2, 2, 2]);
    assert.deepEqual(EJ.vectors[0], [[1, 0, 0]]);
    const J2 = [[3, 1, 0], [0, 3, 0], [0, 0, 3]], E2 = L.eig(J2);
    assert.deepEqual(E2.vectors[0], [[1, 0, 0], [0, 0, 1]]); // geometric multiplicity 2 < 3
    checkEig(J, EJ);
    checkEig(J2, E2);
    // 3D shear with a distinct third eigenvalue
    const K = [[2, 1, 0], [0, 2, 0], [0, 0, 5]], EK = L.eig(K);
    assert.deepEqual(EK.values.map(v => v.re), [5, 2, 2]);
    assert.deepEqual(EK.vectors.map(v => v.length), [1, 1, 1]);
  });

  test('diagonal and repeated eigenvalues', () => {
    const E = L.eig([[2, 0, 0], [0, 5, 0], [0, 0, 2]]);
    assert.deepEqual(E.values.map(v => v.re), [5, 2, 2]);
    assert.deepEqual(E.vectors, [[[0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1]]]);
    const I = L.eig(L.identity(3));
    assert.deepEqual(I.values.map(v => v.re), [1, 1, 1]);
    assert.deepEqual(I.vectors[0], L.identity(3));
    const S = L.eig([[3, 0], [0, 3]]);
    assert.deepEqual(S.vectors[0], [[1, 0], [0, 1]]);
    const Z = L.eig([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    assert.deepEqual(Z.values.map(v => v.re), [0, 0, 0]);
    assert.equal(Z.vectors[0].length, 3);
    // eigenvalue 1 on the plane x + y + z = 0, eigenvalue 4 along (1,1,1)
    const A = [[2, 1, 1], [1, 2, 1], [1, 1, 2]], EA = L.eig(A);
    assert.deepEqual(EA.values.map(v => v.re), [4, 1, 1]);
    assert.equal(EA.vectors[1].length, 2);
    closeV(EA.vectors[1][0], [1, 0, -1].map(x => x / Math.SQRT2));
    checkEig(A, EA);
    const B = [[2, 1], [1, 2]], EB = L.eig(B);
    assert.deepEqual(EB.values.map(v => v.re), [3, 1]);
    closeV(EB.vectors[0][0], [Math.SQRT1_2, Math.SQRT1_2]);
    closeV(EB.vectors[1][0], [Math.SQRT1_2, -Math.SQRT1_2]);
    assert.deepEqual(L.eig([[7]]), { values: [{ re: 7, im: 0 }], vectors: [[[1]]], planes: [] });
  });

  test('singular and rank-1 matrices', () => {
    const E = L.eig([[1, 2], [2, 4]]);
    assert.deepEqual(E.values.map(v => v.re), [5, 0]);
    closeV(E.vectors[0][0], [1, 2].map(x => x / Math.sqrt(5)));
    closeV(E.vectors[1][0], [2, -1].map(x => x / Math.sqrt(5)));
    const R1 = [[1, 2, 3], [2, 4, 6], [3, 6, 9]], F = L.eig(R1);
    assert.deepEqual(F.values.map(v => v.re), [14, 0, 0]);
    assert.equal(F.vectors[1].length, 2);
    checkEig(R1, F);
    const N = [[0, 1], [0, 0]];
    assert.deepEqual(L.eig(N).vectors, [[[1, 0]], [[1, 0]]]);
    const P = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]; // projection
    assert.deepEqual(L.eig(P).values.map(v => v.re), [1, 1, 0]);
  });

  test('distinct eigenvalues, non-symmetric', () => {
    const A = [[2, 0, 0], [1, 3, 0], [4, 5, 6]], E = L.eig(A);
    assert.deepEqual(E.values.map(v => v.re), [6, 3, 2]);
    checkEig(A, E);
    const B = [[4, 1], [2, 3]], F = L.eig(B);
    assert.deepEqual(F.values.map(v => v.re), [5, 2]);
    checkEig(B, F);
    // one real and a complex pair with nonzero real part
    const C = [[1, -2, 0], [2, 1, 0], [0, 0, -3]], G = L.eig(C);
    assert.deepEqual(G.values, [{ re: -3, im: 0 }, { re: 1, im: 2 }, { re: 1, im: -2 }]);
    checkEig(C, G);
  });

  test('repeated eigenvalues survive floating-point similarity transforms', () => {
    const S = [[1, 0.3, 0.2], [0.1, 1, 0.4], [0.5, 0.2, 1]], Si = L.inv(S);
    const A = mul(mul(S, diag([1, 1, 3])), Si), E = L.eig(A);
    closeV(E.values.map(v => v.re), [3, 1, 1], 1e-9);
    assert.deepEqual(E.vectors.map(v => v.length), [1, 2, 2]);
    checkEig(A, E);
    const J = mul(mul(S, [[2, 1, 0], [0, 2, 0], [0, 0, 5]]), Si), F = L.eig(J);
    closeV(F.values.map(v => v.re), [5, 2, 2], 1e-9);
    assert.deepEqual(F.vectors.map(v => v.length), [1, 1, 1]);
    checkEig(J, F, 1e-6);
    const Q = L.qr([[1, 2, 0], [2, -1, 1], [0, 1, 3]]).Q;
    const Sym = mul(mul(Q, diag([4, -1, -1])), T(Q)), G = L.eig(Sym);
    closeV(G.values.map(v => v.re), [4, -1, -1], 1e-9);
    assert.equal(G.vectors[1].length, 2);
    checkEig(Sym, G);
    const T3 = mul(mul(S, [[0.1, 1, 0], [0, 0.1, 1], [0, 0, 0.1]]), Si), H = L.eig(T3);
    closeV(H.values.map(v => v.re), [0.1, 0.1, 0.1], 1e-6);
    assert.equal(H.vectors[0].length, 1);
  });

  test('near-repeated eigenvalues behave (merged, with a full eigenspace)', () => {
    const E = L.eig([[1, 0], [0, 1 + 1e-12]]);
    assert.equal(E.values[0].re, E.values[1].re);
    assert.equal(E.vectors[0].length, 2);
    const F = L.eig([[1, 1e-6], [0, 1]]); // exact repeat, genuinely defective
    assert.equal(F.vectors[0].length, 1);
    const F3 = L.eig([[2, 0, 1e-6], [0, 2, 0], [0, 0, 2]]);
    assert.deepEqual(F3.vectors[0], [[1, 0, 0], [0, 1, 0]]);
    const G = L.eig(rot2(1e-3)); // small but real rotation stays complex
    assert.equal(G.planes.length, 1);
    close(G.values[0].im, Math.sin(1e-3));
    const tiny = [[0, -1e-11], [1e-11, 0]], H = L.eig(tiny); // tiny scale: pair must stay complex
    assert.equal(H.planes.length, 1);
    assert.ok(H.values[0].im > 0);
    checkEig(tiny, H);
  });

  test('random integer and real matrices satisfy A v = λ v', () => {
    const r = rng(12345);
    for (let k = 0; k < 300; k++) {
      const n = 2 + (k % 2);
      const A = k % 3 === 2 ? randReal(r, n) : randInt(r, n);
      checkEig(A, L.eig(A));
    }
    for (let k = 0; k < 50; k++) { // symmetric: real eigenvalues, orthonormal eigenvectors
      const M = randInt(r, 3), A = M.map((row, i) => row.map((x, j) => x + M[j][i]));
      const E = L.eig(A);
      assert.equal(E.planes.length, 0);
      checkEig(A, E);
      // one basis per distinct eigenvalue; together they must form an orthonormal basis of R³
      const all = [];
      E.vectors.forEach((b, i) => { if (i === 0 || E.values[i].re !== E.values[i - 1].re) all.push(...b); });
      assert.equal(all.length, 3);
      closeM(mul(all, T(all)), L.identity(3), 1e-7);
    }
  });
});

// ---------------------------------------------------------------- svd

describe('svd', () => {
  function checkSvd(A) {
    const m = A.length, n = A[0].length, k = Math.min(m, n);
    const { U, S, V } = L.svd(A);
    assert.equal(U.length, m); assert.equal(U[0].length, m);
    assert.equal(V.length, n); assert.equal(V[0].length, n);
    assert.equal(S.length, k);
    for (let i = 1; i < k; i++) assert.ok(S[i - 1] >= S[i], 'descending');
    for (const s of S) assert.ok(s >= 0);
    isOrthonormalCols(U);
    isOrthonormalCols(V);
    closeM(mul(mul(U, diag(S, m, n)), T(V)), A, 1e-9, 'U Σ Vᵀ = A');
    return { U, S, V };
  }

  test('known 2×2', () => {
    const { S } = checkSvd([[3, 0], [4, 5]]);
    closeV(S, [3 * Math.sqrt(5), Math.sqrt(5)]);
    const { U, V } = checkSvd(L.identity(2));
    assert.deepEqual(U, L.identity(2));
    assert.deepEqual(V, L.identity(2));
    assert.deepEqual(checkSvd([[2, 0], [0, -3]]).S, [3, 2]);
  });

  test('rank-deficient: U and V are completed to orthonormal bases', () => {
    assert.deepEqual(checkSvd([[1, 2], [2, 4]]).S, [5, 0]);
    const r1 = checkSvd([[1, 2, 3], [2, 4, 6], [3, 6, 9]]);
    close(r1.S[0], 14);
    assert.deepEqual(r1.S.slice(1), [0, 0]);
    assert.deepEqual(checkSvd([[0, 0, 0], [0, 0, 0]]).S, [0, 0]);
    const rot = checkSvd(rotZ(0.7));
    closeV(rot.S, [1, 1, 1]);
  });

  test('non-square shapes', () => {
    closeV(checkSvd([[1, 0], [0, 1], [1, 1]]).S, [Math.sqrt(3), 1]);
    closeV(checkSvd([[1, 0, 1], [0, 1, 1]]).S, [Math.sqrt(3), 1]);
    closeV(checkSvd([[1, 2, 2]]).S, [3]);
    closeV(checkSvd([[3], [4]]).S, [5]);
    checkSvd([[1, 2, 3], [4, 5, 6]]);
    checkSvd([[1, 2], [3, 4], [5, 6]]);
  });

  test('random matrices up to 3×3', () => {
    const r = rng(99);
    for (let k = 0; k < 100; k++) {
      const m = 1 + (k % 3), n = 1 + (Math.floor(k / 3) % 3);
      const A = Array.from({ length: m }, () =>
        Array.from({ length: n }, () => (k % 2 ? 2 * r() - 1 : Math.floor(r() * 7) - 3)));
      checkSvd(A);
    }
  });
});
