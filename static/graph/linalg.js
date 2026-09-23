// Small numeric linear algebra for the linear-algebra grapher.
// Pure ES module, no dependencies; runs in browsers and Node.
//
// Conventions
// - Matrices are number[][] (row-major, any m×n), vectors are number[]. Inputs are never mutated.
// - Zero / rank / pivot decisions use a relative tolerance: 1e-9 × (largest |entry| of the input).
// - Outputs snap near-integers (|x - round(x)| < 1e-10, or < 1e-12·|x| for large x) and turn -0
//   into 0, so teaching displays stay clean.
// - Bad input throws an Error with a short human message.

const REL = 1e-9;

// ---------------------------------------------------------------- helpers

function snap(x) {
  const r = Math.round(x);
  if (Math.abs(x - r) < Math.max(1e-10, 1e-12 * Math.abs(x))) return r === 0 ? 0 : r;
  return x;
}
const snapV = v => v.map(snap);
const snapM = A => A.map(snapV);

const isNum = x => typeof x === 'number' && Number.isFinite(x);
const shape = A => `${A.length}×${A[0].length}`;

function checkMatrix(A, name = 'Matrix') {
  if (!Array.isArray(A) || A.length === 0 || !A.every(Array.isArray)) {
    throw new Error(`${name} must be a non-empty list of rows`);
  }
  const n = A[0].length;
  if (n === 0) throw new Error(`${name} must have at least one column`);
  for (const row of A) {
    if (row.length !== n) throw new Error(`${name} rows must all have the same length`);
    for (const x of row) if (!isNum(x)) throw new Error(`${name} entries must be finite numbers`);
  }
  return A;
}

function checkVector(v, name = 'Vector') {
  if (!Array.isArray(v) || v.length === 0) throw new Error(`${name} must be a non-empty list of numbers`);
  for (const x of v) if (!isNum(x)) throw new Error(`${name} entries must be finite numbers`);
  return v;
}

function checkSquare(A, what) {
  checkMatrix(A);
  if (A.length !== A[0].length) throw new Error(`${what} needs a square matrix (got ${shape(A)})`);
  return A;
}

const copyM = A => A.map(r => r.slice());
const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const column = (A, j) => A.map(r => r[j]);
const transposeRaw = A => A[0].map((_, j) => A.map(r => r[j]));
const colsToRows = (cols, m) => Array.from({ length: m }, (_, i) => cols.map(c => c[i]));

function dot(u, v) {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}
const norm = v => Math.sqrt(dot(v, v));
const neg = v => v.map(x => -x);

function maxAbs(A) {
  let m = 0;
  for (const r of A) for (const x of r) if (Math.abs(x) > m) m = Math.abs(x);
  return m;
}
const maxAbsV = v => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

function mulRaw(A, B) {
  return A.map(r => B[0].map((_, j) => {
    let s = 0;
    for (let k = 0; k < r.length; k++) s += r[k] * B[k][j];
    return s;
  }));
}
const mulVecRaw = (A, v) => A.map(r => dot(r, v));

// Sign of the first entry that is not negligible (1 for the zero vector).
function firstSign(v) {
  const m = maxAbsV(v);
  for (const x of v) if (Math.abs(x) > 1e-9 * m) return Math.sign(x);
  return 1;
}
const canon = v => (firstSign(v) < 0 ? neg(v) : v);

// ---------------------------------------------------------------- basic operations

export function identity(n) {
  if (!Number.isInteger(n) || n < 1) throw new Error('identity(n) needs a positive whole number n');
  return eye(n);
}

export function transpose(A) {
  checkMatrix(A);
  return snapM(transposeRaw(A));
}

export function matmul(A, B) {
  checkMatrix(A, 'A');
  checkMatrix(B, 'B');
  if (A[0].length !== B.length) throw new Error(`Dimension mismatch: can't multiply ${shape(A)} by ${shape(B)}`);
  return snapM(mulRaw(A, B));
}

export function matvec(A, v) {
  checkMatrix(A);
  checkVector(v);
  if (A[0].length !== v.length) {
    throw new Error(`Dimension mismatch: ${shape(A)} matrix times a vector of length ${v.length}`);
  }
  return snapV(mulVecRaw(A, v));
}

export function add(A, B) {
  checkMatrix(A, 'A');
  checkMatrix(B, 'B');
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error(`Dimension mismatch: can't add ${shape(A)} and ${shape(B)}`);
  }
  return snapM(A.map((r, i) => r.map((x, j) => x + B[i][j])));
}

export function scale(A, s) {
  checkMatrix(A);
  if (!isNum(s)) throw new Error('Scale factor must be a finite number');
  return snapM(A.map(r => r.map(x => x * s)));
}

export function trace(A) {
  checkSquare(A, 'trace');
  let t = 0;
  for (let i = 0; i < A.length; i++) t += A[i][i];
  return snap(t);
}

function det3(M) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

export function det(A) {
  checkSquare(A, 'det');
  const n = A.length;
  if (n === 1) return snap(A[0][0]);
  if (n === 2) return snap(A[0][0] * A[1][1] - A[0][1] * A[1][0]);
  if (n === 3) return snap(det3(A));
  // n ≥ 4: elimination with partial pivoting; a (relatively) zero pivot column means det = 0.
  const U = copyM(A), tol = REL * maxAbs(A);
  let d = 1;
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(U[i][k]) > Math.abs(U[p][k])) p = i;
    if (Math.abs(U[p][k]) <= tol) return 0;
    if (p !== k) { [U[p], U[k]] = [U[k], U[p]]; d = -d; }
    d *= U[k][k];
    for (let i = k + 1; i < n; i++) {
      const f = U[i][k] / U[k][k];
      if (f !== 0) for (let j = k; j < n; j++) U[i][j] -= f * U[k][j];
    }
  }
  return snap(d);
}

export function inv(A) {
  checkSquare(A, 'inv');
  const n = A.length, tol = REL * maxAbs(A);
  const aug = A.map((row, i) => [...row, ...eye(n)[i]]);
  const tols = [...new Array(n).fill(tol), ...new Array(n).fill(0)];
  const { R, pivots } = gaussJordan(aug, { tols });
  if (tol === 0 || pivots.filter(c => c < n).length < n) throw new Error('Matrix is singular (not invertible)');
  return snapM(R.map(row => row.slice(n)));
}

// ---------------------------------------------------------------- elimination

// Gauss-Jordan elimination to RREF. `tols[j]` is the zero threshold for column j.
// Partial pivoting by default; `firstNonzero` picks the first usable pivot (hand style).
// `onStep(op, rows, factor, R)` is called after every row operation; when present the working
// matrix is also tidied (near-integers snapped) so recorded steps look like hand work.
function gaussJordan(A, { tols, firstNonzero = false, onStep = null }) {
  const R = copyM(A), m = R.length, n = R[0].length;
  const tidy = onStep !== null;
  const clean = i => {
    const row = R[i];
    for (let j = 0; j < n; j++) {
      if (Math.abs(row[j]) <= tols[j]) row[j] = 0;
      else if (tidy) row[j] = snap(row[j]);
    }
  };
  for (let i = 0; i < m; i++) clean(i);
  const pivots = [];
  let r = 0;
  for (let c = 0; c < n && r < m; c++) {
    let p = -1;
    if (firstNonzero) {
      for (let i = r; i < m; i++) if (Math.abs(R[i][c]) > tols[c]) { p = i; break; }
    } else {
      let best = tols[c];
      for (let i = r; i < m; i++) if (Math.abs(R[i][c]) > best) { best = Math.abs(R[i][c]); p = i; }
    }
    if (p < 0) {
      for (let i = r; i < m; i++) R[i][c] = 0;
      continue;
    }
    if (p !== r) {
      [R[p], R[r]] = [R[r], R[p]];
      onStep?.('swap', [r, p], null, R);
    }
    const piv = R[r][c];
    if (piv !== 1) {
      R[r] = R[r].map(x => x / piv);
      R[r][c] = 1;
      clean(r);
      onStep?.('scale', [r], 1 / piv, R);
    }
    for (let i = 0; i < m; i++) {
      const f = R[i][c];
      if (i === r || f === 0) continue;
      for (let j = 0; j < n; j++) R[i][j] -= f * R[r][j];
      R[i][c] = 0;
      clean(i);
      onStep?.('add', [i, r], -f, R);
    }
    pivots.push(c);
    r++;
  }
  return { R, pivots };
}

function rrefRaw(A) {
  const tol = REL * maxAbs(A);
  return gaussJordan(A, { tols: A[0].map(() => tol) });
}

export function rref(A) {
  checkMatrix(A);
  const { R, pivots } = rrefRaw(A);
  return { R: snapM(R), pivots };
}

// Coefficient formatting for step labels: integers, p/q with q ≤ 12 (or 1/n), else 4 significant digits.
function simpleRational(a) {
  for (let q = 1; q <= 12; q++) {
    const p = Math.round(a * q);
    if (p > 0 && Math.abs(a - p / q) <= 1e-9 * Math.max(1, a)) return [p, q];
  }
  const n = Math.round(1 / a);
  if (n > 12 && n <= 1000 && Math.abs(1 / a - n) <= 1e-9 * n) return [1, n];
  return null;
}

function coef(a) { // a > 0; empty string for 1
  const r = simpleRational(a);
  if (r) {
    const [p, q] = r;
    if (q === 1) return p === 1 ? { latex: '', text: '' } : { latex: `${p}`, text: `${p}` };
    return { latex: `\\tfrac{${p}}{${q}}`, text: `(${p}/${q})` };
  }
  const s = String(Number(a.toPrecision(4)));
  const e = s.match(/^([\d.]+)e([+-]\d+)$/);
  return { latex: e ? `${e[1]}\\times 10^{${Number(e[2])}}` : s, text: s };
}

const rowTex = i => (i < 9 ? `R_${i + 1}` : `R_{${i + 1}}`);
const rowTxt = i => `R${i + 1}`;

function describeOp(op, [i, j], k) {
  if (op === 'swap') {
    return { latex: `${rowTex(i)} \\leftrightarrow ${rowTex(j)}`, text: `${rowTxt(i)} ↔ ${rowTxt(j)}` };
  }
  const c = coef(Math.abs(k));
  if (op === 'scale') {
    const sg = k < 0 ? '-' : '';
    return {
      latex: `${rowTex(i)} \\leftarrow ${sg}${c.latex}${rowTex(i)}`,
      text: `${rowTxt(i)} ← ${sg}${c.text}${rowTxt(i)}`,
    };
  }
  const sg = k < 0 ? '-' : '+';
  return {
    latex: `${rowTex(i)} \\leftarrow ${rowTex(i)} ${sg} ${c.latex}${rowTex(j)}`,
    text: `${rowTxt(i)} ← ${rowTxt(i)} ${sg} ${c.text}${rowTxt(j)}`,
  };
}

// Gauss-Jordan steps from M to its RREF, pivoting on the first nonzero entry like hand work.
// Each step: { M, op, latex, text, rows, factor? }. rows are 0-based: swap [i, j]; scale [i]
// (R_i ← factor·R_i); add [i, j] (R_i ← R_i + factor·R_j). Step 0 is the start state.
export function eliminationSteps(M) {
  checkMatrix(M);
  const tol = REL * maxAbs(M);
  const steps = [{ M: snapM(M), op: 'start', latex: '\\text{start}', text: 'start', rows: [] }];
  gaussJordan(M, {
    tols: M[0].map(() => tol),
    firstNonzero: true,
    onStep: (op, rows, factor, R) => {
      const step = { M: snapM(R), op, ...describeOp(op, rows, factor), rows };
      if (factor !== null) step.factor = snap(factor);
      steps.push(step);
    },
  });
  return steps;
}

// ---------------------------------------------------------------- the four subspaces

function nullFromRref(R, pivots, n) {
  const isPivot = new Array(n).fill(false);
  for (const c of pivots) if (c < n) isPivot[c] = true;
  const out = [];
  for (let f = 0; f < n; f++) {
    if (isPivot[f]) continue;
    const v = new Array(n).fill(0);
    v[f] = 1;
    pivots.forEach((c, i) => { if (c < n) v[c] = -R[i][f]; });
    out.push(snapV(v));
  }
  return out;
}

export function rank(A) {
  checkMatrix(A);
  return rrefRaw(A).pivots.length;
}

export function nullspace(A) {
  checkMatrix(A);
  const { R, pivots } = rrefRaw(A);
  return nullFromRref(R, pivots, A[0].length);
}

export function colspace(A) {
  checkMatrix(A);
  return rrefRaw(A).pivots.map(j => snapV(column(A, j)));
}

export function rowspace(A) {
  checkMatrix(A);
  const { R, pivots } = rrefRaw(A);
  return R.slice(0, pivots.length).map(snapV);
}

export function leftNullspace(A) {
  checkMatrix(A);
  return nullspace(transposeRaw(A));
}

// ---------------------------------------------------------------- solving

export function solve(A, b) {
  checkMatrix(A);
  checkVector(b, 'b');
  const m = A.length, n = A[0].length;
  if (b.length !== m) throw new Error(`Dimension mismatch: A has ${m} rows but b has ${b.length} entries`);
  const tolA = REL * maxAbs(A), tolB = REL * Math.max(maxAbs(A), maxAbsV(b));
  const aug = A.map((row, i) => [...row, b[i]]);
  const { R, pivots } = gaussJordan(aug, { tols: [...new Array(n).fill(tolA), tolB] });
  if (pivots.includes(n)) return { kind: 'none' };
  const x = new Array(n).fill(0);
  pivots.forEach((c, i) => { x[c] = R[i][n]; });
  if (pivots.length === n) return { kind: 'unique', x: snapV(x) };
  return { kind: 'infinite', x: snapV(x), directions: nullFromRref(R, pivots, n) };
}

// Least squares via the SVD pseudoinverse: x is the minimum-norm minimiser of |b - Ax|
// (so rank-deficient and underdetermined systems get the solution lying in the row space).
export function lstsq(A, b) {
  checkMatrix(A);
  checkVector(b, 'b');
  const m = A.length, n = A[0].length;
  if (b.length !== m) throw new Error(`Dimension mismatch: A has ${m} rows but b has ${b.length} entries`);
  const { sig, U, V } = svdCore(A);
  const x = new Array(n).fill(0);
  sig.forEach((s, j) => {
    if (s === 0) return;
    const c = dot(U[j], b) / s;
    for (let i = 0; i < n; i++) x[i] += c * V[j][i];
  });
  const Ax = mulVecRaw(A, x);
  return { x: snapV(x), projection: snapV(Ax), residual: snapV(b.map((y, i) => y - Ax[i])) };
}

// ---------------------------------------------------------------- orthogonality

export function gramSchmidt(vectors) {
  if (!Array.isArray(vectors)) throw new Error('gramSchmidt needs a list of vectors');
  if (vectors.length === 0) return { basis: [], steps: [] };
  vectors.forEach(v => checkVector(v));
  const n = vectors[0].length;
  if (vectors.some(v => v.length !== n)) throw new Error('Dimension mismatch: vectors must all have the same length');
  const basis = [], steps = [];
  for (const v of vectors) {
    const w = v.slice();
    const coeffs = basis.map(() => 0);
    // Modified Gram-Schmidt with one re-orthogonalisation pass; the recorded projection onto each
    // basis vector is the total amount removed along it.
    for (let pass = 0; pass < 2; pass++) {
      basis.forEach((e, k) => {
        const c = dot(w, e);
        coeffs[k] += c;
        for (let i = 0; i < n; i++) w[i] -= c * e[i];
      });
    }
    const nv = norm(v), nw = norm(w);
    const dependent = nv === 0 || nw <= REL * nv;
    const u = dependent ? null : w.map(x => x / nw);
    steps.push({
      input: snapV(v),
      projections: basis.map((e, k) => snapV(e.map(x => x * coeffs[k]))),
      orthogonal: dependent ? new Array(n).fill(0) : snapV(w),
      unit: u && snapV(u),
    });
    if (u) basis.push(u);
  }
  return { basis: basis.map(snapV), steps };
}

// Reduced QR by Householder reflections: Q is m×k with orthonormal columns, R is k×n upper
// triangular with a nonnegative diagonal, k = min(m, n). For full column rank Q matches Gram-Schmidt.
export function qr(A) {
  checkMatrix(A);
  const m = A.length, n = A[0].length, k = Math.min(m, n);
  const R = copyM(A), Q = eye(m);
  for (let j = 0; j < Math.min(m - 1, n); j++) {
    let nx = 0;
    for (let i = j; i < m; i++) nx += R[i][j] * R[i][j];
    nx = Math.sqrt(nx);
    if (nx === 0) continue;
    const alpha = R[j][j] > 0 ? -nx : nx;
    const v = [];
    for (let i = j; i < m; i++) v.push(R[i][j]);
    v[0] -= alpha;
    const vv = dot(v, v);
    if (vv === 0) continue;
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let i = 0; i < v.length; i++) s += v[i] * R[j + i][c];
      const f = (2 * s) / vv;
      for (let i = 0; i < v.length; i++) R[j + i][c] -= f * v[i];
    }
    for (let r = 0; r < m; r++) {
      let s = 0;
      for (let i = 0; i < v.length; i++) s += Q[r][j + i] * v[i];
      const f = (2 * s) / vv;
      for (let i = 0; i < v.length; i++) Q[r][j + i] -= f * v[i];
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < Math.min(i, n); j++) R[i][j] = 0;
  for (let i = 0; i < k; i++) {
    if (R[i][i] < 0) {
      R[i] = neg(R[i]);
      for (let r = 0; r < m; r++) Q[r][i] = -Q[r][i];
    }
  }
  return { Q: snapM(Q.map(r => r.slice(0, k))), R: snapM(R.slice(0, k)) };
}

// P A = L U. Rows are exchanged only when the pivot is (relatively) zero, like hand elimination,
// so P = I whenever no exchange is needed. Works for m×n: L is m×m unit lower, U is m×n.
export function lu(A) {
  checkMatrix(A);
  const m = A.length, n = A[0].length;
  const U = copyM(A), L = eye(m), perm = [...Array(m).keys()];
  const tol = REL * maxAbs(A);
  for (let k = 0; k < Math.min(m, n); k++) {
    let p = -1;
    for (let i = k; i < m; i++) if (Math.abs(U[i][k]) > tol) { p = i; break; }
    if (p < 0) {
      for (let i = k; i < m; i++) U[i][k] = 0;
      continue;
    }
    if (p !== k) {
      [U[p], U[k]] = [U[k], U[p]];
      [perm[p], perm[k]] = [perm[k], perm[p]];
      for (let j = 0; j < k; j++) [L[p][j], L[k][j]] = [L[k][j], L[p][j]];
    }
    for (let i = k + 1; i < m; i++) {
      const f = U[i][k] / U[k][k];
      L[i][k] = f;
      if (f !== 0) for (let j = k; j < n; j++) U[i][j] -= f * U[k][j];
      U[i][k] = 0;
    }
  }
  const P = perm.map(pi => Array.from({ length: m }, (_, j) => (j === pi ? 1 : 0)));
  return { P, L: snapM(L), U: snapM(U) };
}

// ---------------------------------------------------------------- SVD

// One-sided Jacobi on the columns of B (p×q, p ≥ q). Returns W = B·V (mutually orthogonal
// columns) and V (orthonormal), both as arrays of column vectors.
function jacobiCols(B) {
  const q = B[0].length;
  const W = Array.from({ length: q }, (_, j) => column(B, j));
  const V = eye(q);
  const rot = (x, y, c, s) => {
    for (let k = 0; k < x.length; k++) {
      const a = x[k], b = y[k];
      x[k] = c * a - s * b;
      y[k] = s * a + c * b;
    }
  };
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let i = 0; i < q - 1; i++) {
      for (let j = i + 1; j < q; j++) {
        const a = dot(W[i], W[i]), b = dot(W[j], W[j]), g = dot(W[i], W[j]);
        if (g === 0 || Math.abs(g) <= 1e-15 * Math.sqrt(a * b)) continue;
        rotated = true;
        const z = (b - a) / (2 * g);
        const t = (z >= 0 ? 1 : -1) / (Math.abs(z) + Math.sqrt(1 + z * z));
        const c = 1 / Math.sqrt(1 + t * t);
        rot(W[i], W[j], c, c * t);
        rot(V[i], V[j], c, c * t);
      }
    }
    if (!rotated) break;
  }
  return { W, V };
}

// Thin SVD: k = min(m, n) triples sorted by descending σ. Singular values below 1e-9·σ_max are 0;
// for those the partner vector that cannot be derived is null (U when m ≥ n, V when m < n).
function svdCore(A) {
  const m = A.length, n = A[0].length, tr = m < n;
  const { W, V } = jacobiCols(tr ? transposeRaw(A) : A);
  const norms = W.map(norm);
  const order = norms.map((_, j) => j).sort((i, j) => norms[j] - norms[i]);
  const tol = REL * norms[order[0]];
  const sig = [], Uc = [], Vc = [];
  for (const j of order) {
    const s = norms[j];
    if (s > 0 && s > tol) { sig.push(s); Uc.push(W[j].map(x => x / s)); } else { sig.push(0); Uc.push(null); }
    Vc.push(V[j]);
  }
  return tr ? { sig, U: Vc, V: Uc } : { sig, U: Uc, V: Vc };
}

// Extend orthonormal columns (nulls allowed, at the end) to an orthonormal basis of R^dim.
function completeBasis(cols, dim) {
  const out = cols.filter(Boolean).map(c => c.slice());
  while (out.length < dim) {
    let best = null, bestNorm = -1;
    for (let e = 0; e < dim; e++) {
      const w = new Array(dim).fill(0);
      w[e] = 1;
      for (let pass = 0; pass < 2; pass++) {
        for (const u of out) {
          const c = dot(w, u);
          for (let i = 0; i < dim; i++) w[i] -= c * u[i];
        }
      }
      const nw = norm(w);
      if (nw > bestNorm) { bestNorm = nw; best = w; }
    }
    out.push(best.map(x => x / bestNorm));
  }
  return out;
}

// Full SVD: A = U Σ Vᵀ with U m×m, V n×n orthonormal and S the min(m, n) singular values
// (descending). Each column of V is signed so its first nonzero entry is positive.
export function svd(A) {
  checkMatrix(A);
  const m = A.length, n = A[0].length, k = Math.min(m, n);
  const { sig, U, V } = svdCore(A);
  const Uc = completeBasis(U, m), Vc = completeBasis(V, n);
  for (let j = 0; j < Math.max(m, n); j++) {
    if (j < k) {
      if (firstSign(Vc[j]) < 0) { Vc[j] = neg(Vc[j]); Uc[j] = neg(Uc[j]); }
    } else {
      if (j < m) Uc[j] = canon(Uc[j]);
      if (j < n) Vc[j] = canon(Vc[j]);
    }
  }
  return { U: snapM(colsToRows(Uc, m)), S: sig.map(snap), V: snapM(colsToRows(Vc, n)) };
}

// ---------------------------------------------------------------- eigen

// Coefficients [1, c1, …, cn] of det(λI − A) (Faddeev–LeVerrier; exact for small integer matrices).
export function charpoly(A) {
  checkSquare(A, 'charpoly');
  const n = A.length;
  const coeffs = [1];
  let M = A.map(r => r.map(() => 0));
  for (let k = 1; k <= n; k++) {
    M = mulRaw(A, M).map((r, i) => r.map((x, j) => x + (i === j ? coeffs[k - 1] : 0)));
    const AM = mulRaw(A, M);
    let tr = 0;
    for (let i = 0; i < n; i++) tr += AM[i][i];
    coeffs.push(-tr / k);
  }
  return coeffs.map(snap);
}

// Guarded Newton: stop if the step leaves the cap or the residual stops shrinking.
function newton(f, df, t, cap) {
  let ft = f(t);
  for (let it = 0; it < 8 && ft !== 0; it++) {
    const d = df(t);
    if (!d) break;
    const step = ft / d;
    if (!(Math.abs(step) <= cap)) break;
    const tn = t - step, fn = f(tn);
    if (!(Math.abs(fn) < Math.abs(ft))) break;
    t = tn;
    ft = fn;
  }
  return t;
}

// Roots of the characteristic polynomial of a centred, normalised matrix C (entries O(1)).
// Returns real root groups { t, mult, spread } (spread = how far apart the merged roots were)
// and complex pairs { re, im > 0 }. Nearly repeated roots are merged: the thresholds sit well
// above floating-point noise in the coefficients so exact repeats (shears, scalar matrices) stay
// repeated even after rounding.
function roots2(C) {
  const center = (C[0][0] + C[1][1]) / 2;
  const a = (C[0][0] - C[1][1]) / 2;
  const disc = a * a + C[0][1] * C[1][0];
  if (Math.abs(disc) <= 1e-13) return { reals: [{ t: center, mult: 2, spread: Math.sqrt(Math.abs(disc)) }], pairs: [] };
  if (disc > 0) {
    const r = Math.sqrt(disc);
    return { reals: [{ t: center + r, mult: 1, spread: 0 }, { t: center - r, mult: 1, spread: 0 }], pairs: [] };
  }
  return { reals: [], pairs: [{ re: center, im: Math.sqrt(-disc) }] };
}

function roots3(C) {
  const [[a, b, c], [d, e, f], [g, h, i]] = C;
  const c1 = -(a + e + i);
  const c2 = (a * e - b * d) + (a * i - c * g) + (e * i - f * h);
  const c3 = -det3(C);
  // Depressed cubic t³ + P t + Q with λ = t − c1/3.
  const sh = c1 / 3;
  const P = c2 - (c1 * c1) / 3;
  const Q = (2 * c1 * c1 * c1) / 27 - (c1 * c2) / 3 + c3;
  const fn = t => (t * t + P) * t + Q;
  const df = t => 3 * t * t + P;
  if (Math.abs(P) <= 1e-13 && Math.abs(Q) <= 1e-13) {
    return { reals: [{ t: -sh, mult: 3, spread: Math.max(Math.sqrt(Math.abs(P)), Math.cbrt(Math.abs(Q))) }], pairs: [] };
  }
  const D = (Q * Q) / 4 + (P * P * P) / 27; // −discriminant/108
  // Near a double root with gap L to the simple root, coefficient noise η splits the double root
  // by ~2√(η/L), and |D| ≈ δ²L⁴/108. So compare D against L³ rather than against its own scale:
  // this keeps doubles merged even when the gap is small next to the matrix entries.
  const L0 = Math.sqrt(3 * Math.abs(P));
  if (Math.abs(D) <= 1e-14 * L0 * L0 * L0) {
    // A double root at −3Q/(2P) and a simple one at 3Q/P.
    const ts = newton(fn, df, (3 * Q) / P, 0.5 * L0);
    const td = -ts / 2;
    const L = Math.abs(ts - td);
    const spread = Math.sqrt(108 * Math.abs(D)) / (L * L) / 2;
    return { reals: [{ t: ts - sh, mult: 1, spread: 0 }, { t: td - sh, mult: 2, spread }], pairs: [] };
  }
  if (D < 0) {
    // Three distinct real roots (trigonometric form), each polished.
    const r = 2 * Math.sqrt(-P / 3);
    const arg = Math.min(1, Math.max(-1, ((3 * Q) / (2 * P)) * Math.sqrt(-3 / P)));
    const phi = Math.acos(arg) / 3;
    const ts = [0, 1, 2].map(k => r * Math.cos(phi - (2 * Math.PI * k) / 3));
    const polished = ts.map((t, k) => {
      const gap = Math.min(...ts.filter((_, j) => j !== k).map(u => Math.abs(u - t)));
      return newton(fn, df, t, 0.5 * gap);
    });
    return { reals: polished.map(t => ({ t: t - sh, mult: 1, spread: 0 })), pairs: [] };
  }
  // One real root (stable Cardano) and a complex-conjugate pair from Vieta.
  const sq = Math.sqrt(D);
  const A0 = -(Q >= 0 ? 1 : -1) * Math.cbrt(Math.abs(Q) / 2 + sq);
  const B0 = A0 === 0 ? 0 : -P / (3 * A0);
  const t0 = A0 + B0;
  const cap = 0.5 * Math.hypot(1.5 * t0, (Math.sqrt(3) / 2) * (A0 - B0));
  const t1 = newton(fn, df, t0, cap);
  const im = Math.sqrt(Math.max(0, P + 0.75 * t1 * t1));
  return { reals: [{ t: t1 - sh, mult: 1, spread: 0 }], pairs: [{ re: -t1 / 2 - sh, im }] };
}

// Orthonormal basis of the eigenspace for λ with algebraic multiplicity `mult`. The dimension is
// the number of singular values of A − λI at or below `tol` (at least 1, at most mult). Bases of
// 2D+ eigenspaces are tidied (RREF, then Gram-Schmidt) so the first vector is the hand-work one.
function eigenspace(A, lam, mult, tol) {
  const n = A.length;
  const B = A.map((row, i) => row.map((x, j) => x - (i === j ? lam : 0)));
  const { sig, V } = svdCore(B);
  let g = 0;
  for (let j = n - 1; j >= n - mult && sig[j] <= tol; j--) g++;
  const raw = V.slice(n - Math.max(1, g));
  if (raw.length === 1) {
    const v = raw[0], nv = norm(v);
    return [snapV(canon(v.map(x => x / nv)))];
  }
  const { R, pivots } = gaussJordan(raw, { tols: raw[0].map(() => REL * maxAbs(raw)) });
  const out = [];
  for (const row of R.slice(0, pivots.length)) {
    const w = row.slice();
    for (const u of out) {
      const c = dot(w, u);
      for (let i = 0; i < n; i++) w[i] -= c * u[i];
    }
    const nw = norm(w);
    out.push(w.map(x => x / nw));
  }
  return out.map(v => snapV(canon(v)));
}

// Real invariant plane for the pair a ± ib (b > 0): returns [p, q] with
// A p = a p + b q and A q = −b p + a q, i.e. A acts on span{p, q} as |λ|·(rotation by arg λ)
// turning p toward q. p ⟂ q, |p| = 1 ≥ |q| (p is the long axis of the elliptical orbit).
function complexPlane(A, a, b) {
  const n = A.length;
  // Rows of A − (a − ib)I as complex numbers [re, im]; its null vector is p + iq.
  const row = i => A[i].map((x, j) => [x - (i === j ? a : 0), i === j ? b : 0]);
  const cmul = (x, y) => [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
  const csub = (x, y) => [x[0] - y[0], x[1] - y[1]];
  const size = w => w.reduce((s, z) => s + z[0] * z[0] + z[1] * z[1], 0);
  let w;
  if (n === 2) {
    const r = size(row(0)) >= size(row(1)) ? row(0) : row(1);
    w = [r[1], [-r[0][0], -r[0][1]]];
  } else {
    let best = -1;
    for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
      const u = row(i), v = row(j);
      const c = [
        csub(cmul(u[1], v[2]), cmul(u[2], v[1])),
        csub(cmul(u[2], v[0]), cmul(u[0], v[2])),
        csub(cmul(u[0], v[1]), cmul(u[1], v[0])),
      ];
      if (size(c) > best) { best = size(c); w = c; }
    }
  }
  const p = w.map(z => z[0]), q = w.map(z => z[1]);
  const al = dot(p, p), be = dot(q, q), ga = dot(p, q);
  let th;
  if (Math.hypot(al - be, 2 * ga) <= 1e-9 * (al + be)) {
    // Circular orbit: any phase works; make the first significant component of p + iq real.
    const big = Math.sqrt(Math.max(...w.map(z => z[0] * z[0] + z[1] * z[1])));
    const k = w.findIndex(z => Math.hypot(z[0], z[1]) > 1e-9 * big);
    th = -Math.atan2(w[k][1], w[k][0]);
  } else {
    th = 0.5 * Math.atan2(-2 * ga, al - be); // phase that makes p ⟂ q with |p| ≥ |q|
  }
  const c = Math.cos(th), s = Math.sin(th);
  let P = p.map((x, i) => x * c - q[i] * s);
  let Qv = p.map((x, i) => x * s + q[i] * c);
  const np = norm(P);
  P = P.map(x => x / np);
  Qv = Qv.map(x => x / np);
  if (firstSign(P) < 0) { P = neg(P); Qv = neg(Qv); }
  return [snapV(P), snapV(Qv)];
}

// Eigen-decomposition for 1×1 to 3×3.
// values: all n eigenvalues with algebraic multiplicity, real ones descending, then each complex
//   pair as (re, +im), (re, −im).
// vectors: one entry per real value (same index as in `values`): an orthonormal basis of that
//   eigenvalue's eigenspace (repeated eigenvalues repeat the same basis; a defective eigenvalue
//   has fewer basis vectors than its multiplicity).
// planes: per complex pair { value: {re, im > 0}, basis: [p, q] } (see complexPlane).
export function eig(A) {
  checkSquare(A, 'eig');
  const n = A.length;
  if (n > 3) throw new Error('eig supports matrices up to 3×3');
  if (n === 1) return { values: [{ re: snap(A[0][0]), im: 0 }], vectors: [[[1]]], planes: [] };
  const s = maxAbs(A);
  if (s === 0) {
    return {
      values: Array.from({ length: n }, () => ({ re: 0, im: 0 })),
      vectors: Array.from({ length: n }, () => eye(n)),
      planes: [],
    };
  }
  // Shift by the mean eigenvalue and normalise: clustered eigenvalues then give tiny, accurately
  // computed coefficients instead of cancellation-prone ones.
  let mu = 0;
  for (let i = 0; i < n; i++) mu += A[i][i];
  mu /= n;
  const C = A.map((row, i) => row.map((x, j) => (x - (i === j ? mu : 0)) / s));
  const { reals, pairs } = n === 2 ? roots2(C) : roots3(C);

  const groups = reals
    .map(g => ({ lam: snap(mu + s * g.t), mult: g.mult, tol: Math.max(REL, 10 * g.spread) * s }))
    .sort((x, y) => y.lam - x.lam);
  const values = [], vectors = [], planes = [];
  for (const g of groups) {
    const basis = eigenspace(A, g.lam, g.mult, g.tol);
    for (let k = 0; k < g.mult; k++) {
      values.push({ re: g.lam, im: 0 });
      vectors.push(basis.map(v => v.slice()));
    }
  }
  for (const pr of pairs.sort((x, y) => y.re - x.re)) {
    const a = snap(mu + s * pr.re), b = snap(s * pr.im) || s * pr.im; // never snap a pair to real
    values.push({ re: a, im: b }, { re: a, im: -b });
    planes.push({ value: { re: a, im: b }, basis: complexPlane(A, a, b) });
  }
  return { values, vectors, planes };
}
