// Systems of equations, subspaces and factorisation readouts for the 3D tab.
//   rowpicture(A, b)        each equation's plane (3 unknowns) or line (2 unknowns) + the solution set
//   colpicture(A, b)        x1 a1 + x2 a2 (+ x3 a3) tip-to-tail reaching b, or the closest point + gap
//   eliminate(A, b, k)      Gauss-Jordan on [A | b]: row picture of step ⌊k⌋ turning toward the next
//   lstsq(A, b)             column space, b, p = A x̂ and the residual meeting C(A) at a right angle
//   subspaces(A[, t])       row space ⟂ null space and column space ⟂ left null space
//                           (side by side; with a slider t: inputs at t = 0, outputs at t = 1)
//   gramschmidt(u, v[, w][, k])   projections subtracted step by step, then the orthonormal basis
//   basis(b1, b2[, b3])     a skewed lattice;  coords(v, B) = [v]_B, drawn tip-to-tail along it
//   rref rank nullspace colspace rowspace leftnull eig svd qr lu charpoly tr solve   (readouts)
// Pure helpers are exported for tests; scene.js loads in install().
import { registerFunction, registerType, values, formatNumber, parseLine } from '../lang.js';
import * as la from '../linalg.js';

const { vec, mat, kindOf, describe, isVecLike, dot3, cross3, len3 } = values;

const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const pad3 = (a) => [a[0] ?? 0, a[1] ?? 0, a[2] ?? 0];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const colsOf = (M) => M[0].map((_, j) => M.map((r) => r[j]));
const maxAbs = (a) => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

export function anyPerp(a) {
  const u = scale3(a, 1 / len3(a));
  const w = cross3(u, Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
  return scale3(w, 1 / len3(w));
}

// ---------------------------------------------------------------- colours

function hexToHsl(hex) {
  const n = parseInt(String(hex).replace('#', '').slice(0, 6), 16);
  const r = ((n >> 16) & 255) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  if (!d) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, s, l];
}

function hslToHex(h, s, l) {
  h = ((h % 360) + 360) % 360;
  const C = (1 - Math.abs(2 * l - 1)) * s, X = C * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - C / 2;
  const [r, g, b] = h < 60 ? [C, X, 0] : h < 120 ? [X, C, 0] : h < 180 ? [0, C, X]
    : h < 240 ? [0, X, C] : h < 300 ? [X, 0, C] : [C, 0, X];
  return '#' + [r, g, b].map((x) => Math.round((x + m) * 255).toString(16).padStart(2, '0')).join('');
}

// Relative luminance (WCAG) of a #rrggbb colour.
export function luminance(hex) {
  const n = parseInt(String(hex).replace('#', '').slice(0, 6), 16);
  const lin = (x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
}

// hsl -> hex with the lightness nudged until it reads on the theme's background (contrast >= ~3.8).
export function legible(h, s, l, theme = 'dark') {
  const light = theme === 'light';
  for (let k = 0; k < 40; k++) {
    const hex = hslToHex(h, s, l), y = luminance(hex);
    if (light ? y <= 0.2 : y >= 0.2) return hex;
    l = clamp(l + (light ? -0.02 : 0.02), 0.05, 0.95);
  }
  return hslToHex(h, s, l);
}

// Shades of one colour (hue turned, lightness varied) for the equations / columns of one row.
const TONES = [[0, 0], [50, 0.08], [-50, -0.06], [0, 0.2], [95, 0], [-95, 0.1]];
export function shades(hex, n, theme = 'dark') {
  const [h, s, l] = hexToHsl(hex || '#4a90e2');
  const light = theme === 'light', [lo, hi] = light ? [0.25, 0.5] : [0.5, 0.78];
  return Array.from({ length: n }, (_, i) => {
    const [dh, dl] = TONES[i % TONES.length];
    return legible(h + dh, clamp(s, 0.5, 1), clamp(l + (light ? -dl : dl), lo, hi), theme);
  });
}

export function mixHex(a, b, t) {
  const pa = parseInt(a.slice(1, 7), 16), pb = parseInt(b.slice(1, 7), 16);
  const ch = (p, sh) => (p >> sh) & 255;
  return '#' + [16, 8, 0].map((sh) => Math.round(ch(pa, sh) + (ch(pb, sh) - ch(pa, sh)) * t).toString(16).padStart(2, '0')).join('');
}

// Row space / null space (inputs, cool) and column space / left null space (outputs, warm).
const FOUR = {
  dark: { row: '#5b9ef0', nul: '#b08cf2', col: '#f0695c', lnul: '#f2b441' },
  light: { row: '#2c6fc4', nul: '#7b4fcf', col: '#c8412f', lnul: '#b87300' },
};

// ---------------------------------------------------------------- numbers -> LaTeX

const dec = (x, d = 4) => {
  if (Math.abs(x) >= 1e15) return formatNumber(x);
  const s = x.toFixed(d).replace(/\.?0+$/, '');
  return s === '-0' ? '0' : s;
};

// Integers, p/q with q ≤ 12, else d decimals.
export function fracTex(x, d = 4) {
  if (!Number.isFinite(x)) return formatNumber(x);
  const r = Math.round(x);
  if (Math.abs(x - r) <= 1e-9 * Math.max(1, Math.abs(x))) return String(r + 0);
  const a = Math.abs(x);
  for (let q = 2; q <= 12; q++) {
    const p = Math.round(a * q);
    if (p > 0 && Math.abs(a - p / q) <= 1e-9 * Math.max(1, a)) return `${x < 0 ? '-' : ''}\\tfrac{${p}}{${q}}`;
  }
  return dec(x, d);
}

const f3 = (x) => fracTex(x, 3);
const tupleTex = (a, f = fracTex) => `(${a.map((x) => f(x)).join(', ')})`;
const matTex = (M, f = fracTex) => `\\begin{bmatrix}${M.map((r) => r.map((x) => f(x)).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`;
const paint = (color, tex) => (color ? `\\textcolor{${color}}{${tex}}` : tex);
const coefMag = (a, d = 4) => (Math.abs(a - 1) < 1e-12 ? '' : fracTex(a, d));
const aligned = (lines) => `\\begin{aligned}${lines.map((l) => `&${l}`).join(' \\\\ ')}\\end{aligned}`;

// Coefficient in front of a symbol for labels: 1 -> '', -1 -> '-', else 2 decimals / small fraction.
function coefTex(k) {
  if (Math.abs(k - 1) < 1e-9) return '';
  if (Math.abs(k + 1) < 1e-9) return '-';
  return fracTex(k, 2);
}

// k1 s1 + k2 s2 ... with signs as operators.
function comboTex(ks, syms) {
  return ks.map((k, j) => {
    const mag = coefMag(Math.abs(k));
    const t = `${Math.abs(k) < 1e-12 ? '0' : mag}${syms[j]}`;
    return j === 0 ? `${k < 0 ? '-' : ''}${t}` : ` ${k < 0 ? '-' : '+'} ${t}`;
  }).join('');
}

export function nameTex(name, vector) {
  const m = name.match(/^(.*?)(?:_(\w+)|(\d+))$/);
  let base = m && m[1] ? m[1] : name;
  const sub = m && m[1] ? m[2] || m[3] : '';
  if (base.length > 1) base = `\\mathrm{${base}}`;
  const b = vector ? `\\vec{${base}}` : base;
  return sub ? `${b}_{${sub}}` : b;
}

// Coefficients [1, c1, …, cn] of λ^n + c1 λ^(n-1) + … + cn.
export function polyTex(cs, x = '\\lambda') {
  const n = cs.length - 1, parts = [];
  cs.forEach((k, i) => {
    const p = n - i;
    if (Math.abs(k) < 1e-12) return;
    const pw = p === 0 ? '' : p === 1 ? x : `${x}^{${p}}`;
    const t = `${p > 0 ? coefMag(Math.abs(k)) : fracTex(Math.abs(k))}${pw}`;
    parts.push(parts.length ? `${k < 0 ? '-' : '+'} ${t}` : `${k < 0 ? '-' : ''}${t}`);
  });
  return parts.length ? parts.join(' ') : '0';
}

// A direction rescaled to small integers when possible: (0.7071, 0.7071, 0) -> (1, 1, 0).
export function niceDir(v) {
  const nz = v.filter((x) => Math.abs(x) > 1e-9 * maxAbs(v));
  if (!nz.length) return v;
  const base = Math.min(...nz.map(Math.abs));
  for (let q = 1; q <= 12; q++) {
    const w = v.map((x) => (x / base) * q);
    if (w.every((x) => Math.abs(x - Math.round(x)) < 1e-6 && Math.abs(x) <= 40)) return w.map((x) => Math.round(x) + 0);
  }
  return v;
}

const VARS = ['x', 'y', 'z'];

// [1, -2, 0], 5 -> ['x - 2y', '5']
export function equationParts(row, rhs) {
  const parts = [];
  row.forEach((a, j) => {
    if (Math.abs(a) < 1e-12) return;
    const t = `${coefMag(Math.abs(a))}${VARS[j] ?? `x_{${j + 1}}`}`;
    parts.push(parts.length ? `${a < 0 ? '-' : '+'} ${t}` : `${a < 0 ? '-' : ''}${t}`);
  });
  return [parts.length ? parts.join(' ') : '0', fracTex(rhs)];
}

export function affineTex(p, dirs, f = fracTex) {
  const names = dirs.length === 1 ? ['t'] : ['s', 't', 'u', 'w', 'r'];
  return `${tupleTex(p, f)}${dirs.map((d, i) => ` + ${names[i]}\\,${tupleTex(d, f)}`).join('')}`;
}

// [what the solution set is, its formula or null]
function solutionParts(sol, n) {
  const vars = n <= 3 ? `(${VARS.slice(0, n).join(', ')})` : '\\vec x';
  if (sol.kind === 'none') return ['\\text{no common point}', null];
  if (sol.kind === 'unique') return [`${vars} = ${tupleTex(sol.x)}`, null];
  const k = sol.directions.length;
  if (k === n) return ['\\text{every point}', null];
  const what = k === 1 ? 'a line' : k === 2 ? 'a plane' : `a ${k}-dim family`;
  return [`\\text{${what}:}`, `${vars} = ${affineTex(sol.x, sol.directions)}`];
}

// ---------------------------------------------------------------- pure geometry

const CUBE_EDGES = [];
for (let i = 0; i < 8; i++) for (let a = 0; a < 3; a++) if (!(i & (1 << a))) CUBE_EDGES.push([i, i | (1 << a)]);
const cubeCorner = (i, B) => [i & 1 ? B : -B, i & 2 ? B : -B, i & 4 ? B : -B];

// The plane n·x = c cut by the cube |x|,|y|,|z| ≤ B: polygon vertices in order, or null.
export function planeInCube(nv, c, B) {
  const L = len3(nv);
  if (!(L > 0)) return null;
  const eps = 1e-12 * (L * B + Math.abs(c));
  const pts = [];
  const push = (p) => { if (!pts.some((q) => len3(sub3(p, q)) <= 1e-9 * B)) pts.push(p); };
  for (const [i, j] of CUBE_EDGES) {
    const p = cubeCorner(i, B), q = cubeCorner(j, B);
    const fp = dot3(nv, p) - c, fq = dot3(nv, q) - c;
    if (Math.abs(fp) <= eps) push(p);
    if (Math.abs(fq) <= eps) push(q);
    if ((fp < -eps && fq > eps) || (fp > eps && fq < -eps)) push(lerp3(p, q, fp / (fp - fq)));
  }
  if (pts.length < 3) return null;
  const n = scale3(nv, 1 / L), e1 = anyPerp(n), e2 = cross3(n, e1);
  const ctr = scale3(pts.reduce(add3, [0, 0, 0]), 1 / pts.length);
  const ang = (p) => { const d = sub3(p, ctr); return Math.atan2(dot3(d, e2), dot3(d, e1)); };
  return pts.map((p) => [ang(p), p]).sort((a, b) => a[0] - b[0]).map(([, p]) => p);
}

// Clip p + t d to the cube |x|,|y|,|z| ≤ B. Returns [t0, t1] or null.
export function clipLine3(p, d, B) {
  let t0 = -Infinity, t1 = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-12) { if (Math.abs(p[i]) > B) return null; continue; }
    let a = (-B - p[i]) / d[i], b = (B - p[i]) / d[i];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
  }
  return t0 < t1 ? [t0, t1] : null;
}

// Line where n1·x = c1 meets n2·x = c2 ({p closest to the origin, d}), or null if parallel.
export function meetPlanes(n1, c1, n2, c2) {
  const d = cross3(n1, n2), dd = dot3(d, d), s1 = dot3(n1, n1), s2 = dot3(n2, n2);
  if (dd <= 1e-12 * s1 * s2) return null;
  const n12 = dot3(n1, n2);
  return { p: scale3(add3(scale3(n1, c1 * s2 - c2 * n12), scale3(n2, c2 * s1 - c1 * n12)), 1 / dd), d };
}

// Point where a1·(x, y) = c1 meets a2·(x, y) = c2 in the plane, or null.
export function meetLines(a1, c1, a2, c2) {
  const det = a1[0] * a2[1] - a1[1] * a2[0];
  if (Math.abs(det) <= 1e-12 * Math.hypot(a1[0], a1[1]) * Math.hypot(a2[0], a2[1])) return null;
  return [(c1 * a2[1] - c2 * a1[1]) / det, (a1[0] * c2 - a2[0] * c1) / det, 0];
}

// Integer lattice lines of 2 or 3 basis vectors, clipped to the cube: [[p, q], ...].
export function latticeSegments(vs, B, cap = 40) {
  const segs = [];
  if (vs.length === 2) {
    const [u, v] = vs, area = len3(cross3(u, v));
    for (const [a, b] of [[u, v], [v, u]]) {
      const K = Math.min(cap, Math.ceil((B * Math.sqrt(3)) / (area / len3(b))));
      for (let k = -K; k <= K; k++) {
        const p = scale3(a, k), t = clipLine3(p, b, B);
        if (t) segs.push([add3(p, scale3(b, t[0])), add3(p, scale3(b, t[1]))]);
      }
    }
    return segs;
  }
  const K = 2; // a 4×4×4 block of cells around the origin
  for (let f = 0; f < 3; f++) {
    const d = vs[f], a = vs[(f + 1) % 3], b = vs[(f + 2) % 3];
    for (let i = -K; i <= K; i++) {
      for (let j = -K; j <= K; j++) {
        const p = add3(scale3(a, i), scale3(b, j)), t = clipLine3(p, d, B);
        if (!t) continue;
        const t0 = Math.max(t[0], -K), t1 = Math.min(t[1], K);
        if (t0 < t1) segs.push([add3(p, scale3(d, t0)), add3(p, scale3(d, t1))]);
      }
    }
  }
  return segs;
}

// ---------------------------------------------------------------- steppers

// Step s = ⌊k⌋ of S, fraction f toward s + 1 (k missing: the last step).
export function stepFrame(S, k) {
  const kk = Number.isFinite(k) ? clamp(k, 0, S) : S;
  const s = Math.min(S, Math.floor(kk + 1e-9));
  const f = s >= S ? 0 : kk - s;
  return { s, f: f > 1e-9 ? f : 0, S };
}

// Row picture of elimination step ⌊k⌋ moving toward the next. Only R_i ← R_i + c R_j moves a
// plane (it turns about the solution set); scaling and swapping leave the planes where they are.
export function elimFrame(steps, k) {
  const N = steps.length - 1, { s, f } = stepFrame(N, k);
  const cur = steps[s], next = f > 0 ? steps[s + 1] : null;
  const rows = cur.M.map((r) => r.slice());
  if (next?.op === 'add') {
    const [i] = next.rows;
    rows[i] = cur.M[i].map((x, j) => x + f * (next.M[i][j] - x));
  }
  const show = next ?? cur;
  return {
    s, f, N, rows, pending: !!next, op: show.op, opRows: show.rows.slice(), factor: show.factor ?? null,
    swap: next?.op === 'swap' ? next.rows.slice() : null,
  };
}

// Gram-Schmidt pieces: ws orthogonal results, qs unit vectors (null when dependent),
// coefs[j][i] = (v_j·w_i)/(w_i·w_i), projs[j][i] = that multiple of w_i (null if none).
export function gsData(vs) {
  const res = la.gramSchmidt(vs);
  const idx = [], ws = [], qs = [], coefs = [], projs = [];
  res.steps.forEach((st, j) => {
    const cs = vs.map(() => 0), ps = vs.map(() => null);
    st.projections.forEach((pv, kk) => {
      const i = idx[kk], w = ws[i], ww = dot3(w, w);
      cs[i] = ww > 0 ? dot3(pv, w) / ww : 0;
      ps[i] = pv;
    });
    ws.push(st.orthogonal);
    qs.push(st.unit);
    coefs.push(cs);
    projs.push(ps);
    if (st.unit) idx.push(j);
  });
  return { vs: vs.map((v) => v.slice()), ws, qs, coefs, projs, p: vs.length };
}

// ---------------------------------------------------------------- argument checks

function needMat(x, fname) {
  if (kindOf(x) !== 'mat') throw new Error(`${fname} needs a matrix, got ${describe(x)}`);
  return x.m;
}

// b as a list of m numbers (a vector's trailing zeros are dropped; a number works for one row).
export function needRhs(b, m, fname) {
  let v;
  if (kindOf(b) === 'num') v = [b];
  else if (isVecLike(b)) v = b.v;
  else throw new Error(`${fname} needs a right-hand side vector b, got ${describe(b)}`);
  if (m > 3) throw new Error(`${fname}: A has ${m} rows, but b can have at most 3 entries`);
  for (let i = m; i < v.length; i++) {
    if (v[i] !== 0) throw new Error(`A has ${m} row${m === 1 ? '' : 's'} but b has ${i + 1} entries`);
  }
  if (v.length < m) throw new Error(`A has ${m} rows but b is a single number`);
  return v.slice(0, m);
}

function needSystem(fname, A, b) {
  const M = needMat(A, fname);
  return [M, needRhs(b, M.length, fname)];
}

function needUnknowns(M, fname) {
  const n = M[0].length;
  if (n < 2 || n > 3) {
    throw new Error(`${fname} draws 2 unknowns (lines) or 3 unknowns (planes), but A has ${n} column${n === 1 ? '' : 's'}`);
  }
}

function needFewColumns(M, fname) {
  if (M[0].length > 6) throw new Error(`${fname} draws at most 6 columns, A has ${M[0].length}`);
}

function needStep(k, fname) {
  if (k != null && kindOf(k) !== 'num') throw new Error(`${fname} needs a number for the step (use a slider), got ${describe(k)}`);
  return k ?? null;
}

// ---------------------------------------------------------------- values

export function rowPicture(A, b) {
  return { type: 'sy-rows', A: A.map((r) => r.slice()), b: b.slice(), m: A.length, n: A[0].length, sol: la.solve(A, b) };
}

export function colPicture(A, b) {
  const sol = la.solve(A, b), exact = sol.kind !== 'none';
  const x = exact ? sol.x : la.lstsq(A, b).x;
  return {
    type: 'sy-cols', m: A.length, n: A[0].length, cols: colsOf(A).map(pad3), b: pad3(b), x,
    p: pad3(la.matvec(A, x)), exact, kind: sol.kind, rank: la.rank(A), basis: la.colspace(A).map(pad3),
  };
}

export function elimination(A, b, k = null) {
  let steps = la.eliminationSteps(A.map((r, i) => [...r, b[i]]));
  // Stop at the first 0 = c row: clearing the b column after that only moves planes for no reason.
  const n = A[0].length;
  const contradiction = steps.findIndex(st => st.M.some(row =>
    row.slice(0, n).every(x => Math.abs(x) < 1e-9) && Math.abs(row[n]) > 1e-9));
  if (contradiction >= 0) steps = steps.slice(0, contradiction + 1);
  return { type: 'sy-elim', m: A.length, n: A[0].length, steps, k, frame: elimFrame(steps, k), sol: la.solve(A, b) };
}

export function lstsqPicture(A, b) {
  const r = la.lstsq(A, b);
  return {
    type: 'sy-lstsq', m: A.length, n: A[0].length, cols: colsOf(A).map(pad3), b: pad3(b), x: r.x,
    p: pad3(r.projection), e: pad3(r.residual), rank: la.rank(A), basis: la.colspace(A).map(pad3),
  };
}

export function fourSubspaces(A, t = null) {
  return {
    type: 'sy-four', m: A.length, n: A[0].length, r: la.rank(A), t,
    row: la.rowspace(A), nul: la.nullspace(A), col: la.colspace(A), lnul: la.leftNullspace(A),
  };
}

// Where the input pair (domain) and output pair (codomain) sit, how big, how opaque.
export function fourLayout(t, E) {
  if (t == null) {
    const size = 0.4 * E;
    return [
      { which: 'domain', center: [0, -0.56 * E, 0], size, alpha: 1, apart: true },
      { which: 'codomain', center: [0, 0.56 * E, 0], size, alpha: 1, apart: true },
    ];
  }
  const u = clamp(t, 0, 1), size = 0.72 * E;
  return [
    { which: 'domain', center: [0, 0, 0], size, alpha: 1 - u, apart: false },
    { which: 'codomain', center: [0, 0, 0], size, alpha: u, apart: false },
  ];
}

export function gsValue(vs, k = null) {
  return { type: 'sy-gs', ...gsData(vs), k, frame: stepFrame(vs.length + 1, k), flat: vs.every((v) => v[2] === 0) };
}

// Columns of a basis given as a matrix or a basis(...) value, as 3D vectors.
function basisCols(B, fname) {
  if (kindOf(B) === 'sy-lattice') return B.vecs.map((v) => v.slice());
  const M = needMat(B, fname);
  if (M.length > 3) throw new Error(`${fname} needs basis vectors with at most 3 entries`);
  return colsOf(M).map(pad3);
}

function checkBasis(cols, fname) {
  if (cols.length < 2 || cols.length > 3) throw new Error(`${fname} needs 2 or 3 basis vectors, got ${cols.length}`);
  if (la.rank(cols) < cols.length) throw new Error('the basis vectors are dependent, so they are not a basis');
  return cols;
}

// [v]_B: the coefficients c with c1 b1 + c2 b2 (+ c3 b3) = v.
export function coordsOf(v, cols) {
  checkBasis(cols, 'coords');
  const flat = cols.every((c) => c[2] === 0);
  if (flat && v[2] !== 0) throw new Error('v has a z component, but the basis vectors lie in the xy-plane');
  const B = colsToRows(cols, flat ? 2 : 3), rhs = flat ? v.slice(0, 2) : v.slice(0, 3);
  const sol = la.solve(B, rhs);
  if (sol.kind === 'none') throw new Error(`v is not in the plane of the ${cols.length} basis vectors`);
  return sol.x;
}
const colsToRows = (cols, m) => Array.from({ length: m }, (_, i) => cols.map((c) => c[i]));

export function spaceValue(basis, ambient, which) {
  if (ambient <= 3) return { type: 'span', vecs: basis.map(pad3), sy: { kind: 'space', which, basis, ambient } };
  return { type: 'sy-space', basis, ambient, which };
}

const textValue = (tex, text) => ({ type: 'sy-text', tex, text });

function noSolutionTex(A, b) {
  const r = la.rank(A);
  return aligned([
    '\\text{no solution: elimination reaches } 0 = 1',
    `\\operatorname{rank} A = ${r} < \\operatorname{rank}\\,[A\\,|\\,\\vec b] = ${r + 1}`,
  ]);
}

export function solveValue(A, b) {
  const sol = la.solve(A, b), n = A[0].length;
  if (sol.kind === 'unique') {
    return n <= 3 ? { ...vec(pad3(sol.x)), sy: { kind: 'solve', x: sol.x } }
      : textValue(`\\vec x = ${tupleTex(sol.x)}`, `x = (${sol.x.map(formatNumber).join(', ')})`);
  }
  if (sol.kind === 'infinite') {
    if (n <= 3) return { type: 'sy-affine', n, x: sol.x, directions: sol.directions, p: pad3(sol.x), dirs: sol.directions.map(pad3) };
    return textValue(`\\vec x = ${affineTex(sol.x, sol.directions)}`, 'infinitely many solutions');
  }
  return textValue(noSolutionTex(A, b), 'no solution');
}

// ---------------------------------------------------------------- readouts (KaTeX source)

const matName = (a, fallback = 'A') => (a?.name ? nameTex(a.name, false) : fallback);
const colLetter = (a, fallback = 'a') => (a?.name && /^[A-Z]$/.test(a.name) ? a.name.toLowerCase() : fallback);
const argVec = (a, fallback) => (a?.name ? nameTex(a.name, true) : fallback);

function rowsTex(v, tones = []) {
  const eqs = v.A.map((row, i) => {
    const [l, r] = equationParts(row, v.b[i]);
    return `${paint(tones[i], l)} &${paint(tones[i], `{}= ${r}`)}`;
  });
  const block = v.m === 1 ? eqs[0].replace('&', '')
    : `\\left.\\begin{aligned}${eqs.join(' \\\\ ')}\\end{aligned}\\right\\}`;
  // a family of solutions: "a line:" beside the system, its formula on the next line
  const [head, formula] = solutionParts(v.sol, v.n);
  const first = `${block}\\ \\Rightarrow\\ ${head}`;
  return formula ? aligned([first, formula]) : first;
}

function colsTex(v, tones = [], args = []) {
  const a = colLetter(args[0]), bT = argVec(args[1], '\\vec b');
  const syms = v.cols.map((_, j) => paint(tones[j], `\\vec ${a}_{${j + 1}}`));
  const x = tupleTex(v.x);
  if (v.exact) {
    const many = v.kind === 'infinite' ? '\\ \\ (\\text{one of many})' : '';
    return `${comboTex(v.x, syms)} = ${bT}\\quad \\vec x = ${x}${many}`;
  }
  const gap = len3(sub3(v.b, v.p));
  return aligned([
    `\\text{no exact solution: } ${bT} \\notin C(${matName(args[0])})`,
    `\\text{closest: } ${comboTex(v.x, syms)},\\ \\ \\hat x = ${tupleTex(v.x, f3)}`,
    `\\text{gap } \\|${bT} - A\\hat x\\| = ${dec(gap, 3)}`,
  ]);
}

function elimTex(v, tones = []) {
  const { frame, n, steps } = v, M = steps[frame.s].M;
  const strut = '\\vphantom{\\tfrac{1}{2}}';
  const start = frame.op === 'start';
  const target = start ? -1 : frame.opRows[0];
  const other = start || frame.op === 'scale' ? -1 : frame.opRows[1];
  const bold = (i) => i === target || (frame.op === 'swap' && i === other);
  const labels = M.map((_, i) => `${paint(tones[i], `R_{${i + 1}}`)}${strut}`).join(' \\\\ ');
  const body = M.map((row, i) => row.map((x) => {
    const t = fracTex(x);
    if (bold(i)) return paint(tones[i], `\\boldsymbol{${t}}`);
    return i === other ? paint(tones[i], t) : t;
  }).join(' & ') + strut).join(' \\\\ ');
  const note = M.map((_, i) => (i === target ? opTex(frame.op, frame.opRows, frame.factor, tones) : '') + strut).join(' \\\\ ');
  const count = frame.pending ? `${frame.s}\\to ${frame.s + 1}` : `${frame.s}`;
  const done = !frame.pending && frame.s === frame.N ? '\\ \\ \\text{RREF}' : '';
  return aligned([
    `{\\scriptstyle\\text{step } ${count}/${frame.N}}${done}`,
    `\\begin{array}{r}${labels}\\end{array}\\!\\left[\\begin{array}{${'c'.repeat(n)}|c}${body}\\end{array}\\right]`
      + `\\begin{array}{l}${note}\\end{array}`,
  ]);
}

function opTex(op, rows, factor, tones = []) {
  const R = (i) => paint(tones[i], `R_{${i + 1}}`);
  if (op === 'swap') return `${R(rows[0])} \\leftrightarrow ${R(rows[1])}`;
  const sg = factor < 0 ? '-' : '';
  if (op === 'scale') return `${R(rows[0])} \\leftarrow ${sg}${coefMag(Math.abs(factor))}${R(rows[0])}`;
  return `${R(rows[0])} \\leftarrow ${R(rows[0])} ${factor < 0 ? '-' : '+'} ${coefMag(Math.abs(factor))}${R(rows[1])}`;
}

function lstsqTex(v, args = []) {
  const bT = argVec(args[1], '\\vec b'), A = matName(args[0]);
  return aligned([
    `\\hat x = ${tupleTex(v.x, f3)}\\quad (${A}^{T}${A}\\,\\hat x = ${A}^{T}${bT})`,
    `\\vec p = ${A}\\hat x = ${tupleTex(v.p.slice(0, v.m), f3)},\\ \\ \\|${bT} - \\vec p\\| = ${dec(len3(v.e), 3)}`,
  ]);
}

function fourTex(v, pal, args = []) {
  const A = matName(args[0]), T = `${A}^{T}`;
  return `${aligned([
    `\\mathbb{R}^{${v.n}}:\\ ${paint(pal.row, `C(${T})`)}\\ (${v.r}) \\perp ${paint(pal.nul, `N(${A})`)}\\ (${v.n - v.r})`,
    `\\mathbb{R}^{${v.m}}:\\ ${paint(pal.col, `C(${A})`)}\\ (${v.r}) \\perp ${paint(pal.lnul, `N(${T})`)}\\ (${v.m - v.r})`,
  ])}\\quad \\text{rank } ${v.r}`;
}

function gsTex(v, tones = [], args = []) {
  const { s, f } = v.frame, p = v.p, stage = f > 0 ? s + 1 : s, n = v.flat ? 2 : 3;
  const W = (i) => paint(tones[i], `\\vec w_{${i + 1}}`);
  const Vin = (j) => argVec(args[j], `\\vec v_{${j + 1}}`);
  if (stage === 0) {
    return '\\vec w_j = \\vec v_j - {\\textstyle\\sum_{i<j}} \\tfrac{\\vec v_j\\cdot\\vec w_i}{\\vec w_i\\cdot\\vec w_i}\\,\\vec w_i,'
      + '\\quad \\hat q_j = \\tfrac{\\vec w_j}{\\|\\vec w_j\\|}';
  }
  if (stage <= p) {
    const j = stage - 1;
    const terms = v.coefs[j].slice(0, j).map((cf, i) => (v.qs[i] && Math.abs(cf) > 1e-12
      ? ` ${cf < 0 ? '+' : '-'} ${coefMag(Math.abs(cf), 3)}${W(i)}` : '')).join('');
    const dep = v.qs[j] ? '' : '\\ \\ (\\text{dependent: dropped})';
    return `${W(j)} = ${Vin(j)}${terms} = ${tupleTex(v.ws[j].slice(0, n), f3)}${dep}`;
  }
  return aligned(v.qs.map((q, j) => (q
    ? `${paint(tones[j], `\\hat q_{${j + 1}}`)} = ${tupleTex(q.slice(0, n), f3)}`
    : `\\vec w_{${j + 1}} = \\vec 0\\ \\text{(dropped)}`)));
}

function latticeTex(v) {
  const flat = v.vecs.every((b) => b[2] === 0), m = flat ? 2 : 3;
  const B = colsToRows(v.vecs, m);
  const tail = B.length === B[0].length ? `,\\quad \\det B = ${fracTex(la.det(B))}` : '\\quad (\\text{a plane})';
  return `B = ${matTex(B)}${tail}`;
}

function coordsTex(sy, args = []) {
  const vT = argVec(args[0], '\\vec v'), B = matName(args[1], 'B'), b = colLetter(args[1], 'b');
  const syms = sy.c.map((_, j) => `\\vec ${b}_{${j + 1}}`);
  return `[${vT}]_{${B}} = ${tupleTex(sy.c)}:\\quad ${vT} = ${comboTex(sy.c, syms)}`;
}

function spaceTex(basis, ambient) {
  const k = basis.length;
  if (k === 0) return '= \\{\\vec 0\\}\\quad (\\dim 0)';
  const span = `\\operatorname{span}\\{${basis.map((b) => tupleTex(b)).join(', ')}\\}`;
  return k === ambient ? `= \\mathbb{R}^{${ambient}} = ${span}` : `= ${span}\\quad (\\dim ${k})`;
}

function eigTex(E) {
  const groups = [];
  E.values.forEach((val, i) => {
    if (val.im !== 0) return;
    const g = groups.find((x) => x.re === val.re);
    if (g) g.mult++;
    else groups.push({ re: val.re, mult: 1, vecs: E.vectors[i] });
  });
  const lines = groups.map((g) => {
    const mult = g.mult > 1 ? `\\ (\\times ${g.mult})` : '';
    const defective = g.vecs.length < g.mult ? '\\ \\text{(defective)}' : '';
    return `\\lambda = ${fracTex(g.re)}${mult}:\\ ${g.vecs.map((w) => tupleTex(niceDir(w), f3)).join(',\\ ')}${defective}`;
  });
  for (const { value, basis: [p, q] } of E.planes) {
    const both = niceDir([...p, ...q]), n = p.length;
    const im = Math.abs(value.im - 1) < 1e-12 ? '' : f3(value.im), re = f3(value.re);
    lines.push(`\\lambda = ${re === '0' ? '' : `${re} `}\\pm ${im}i:\\ ${tupleTex(both.slice(0, n), f3)} \\mp i\\,${tupleTex(both.slice(n), f3)}`);
  }
  return aligned(lines);
}

function svdTex(S) {
  const m = S.U.length, n = S.V.length;
  const Sig = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? S.S[i] : 0)));
  const VT = S.V[0].map((_, j) => S.V.map((r) => r[j]));
  return aligned([`U = ${matTex(S.U, f3)}\\quad \\Sigma = ${matTex(Sig, f3)}`, `V^{T} = ${matTex(VT, f3)}`]);
}

function luTex(F) {
  const isEye = F.P.every((r, i) => r.every((x, j) => x === (i === j ? 1 : 0)));
  const lu = `L = ${matTex(F.L)}\\quad U = ${matTex(F.U)}`;
  return isEye ? lu : aligned([`P = ${matTex(F.P)},\\ \\ PA = LU`, lu]);
}

function rrefTex(R, pivots, color) {
  const piv = new Set(pivots.map((c, i) => `${i},${c}`));
  const body = R.map((r, i) => r.map((x, j) => (piv.has(`${i},${j}`) ? paint(color, `\\boldsymbol{${fracTex(x)}}`) : fracTex(x))).join(' & ')).join(' \\\\ ');
  return `= \\begin{bmatrix}${body}\\end{bmatrix}\\quad \\text{rank } ${pivots.length}`;
}

// Full row readout for our values (no leading "= " unless it reads naturally), or null.
export function readoutTex(v, { color = null, theme = 'dark', args = [] } = {}) {
  if (!v || typeof v !== 'object') return null;
  const t = theme === 'light' ? 'light' : 'dark';
  const tones = (n) => (color ? shades(color, n, t) : []);
  switch (v.sy?.kind) {
    case 'rref': return rrefTex(v.m, v.sy.pivots, color);
    case 'space': return spaceTex(v.sy.basis, v.sy.ambient);
    case 'solve': return `\\vec x = ${tupleTex(v.sy.x)}`;
    case 'coords': return coordsTex(v.sy, args);
  }
  switch (v.type) {
    case 'sy-rows': return rowsTex(v, tones(v.m));
    case 'sy-cols': return colsTex(v, tones(v.n), args);
    case 'sy-elim': return elimTex(v, tones(v.m));
    case 'sy-lstsq': return lstsqTex(v, args);
    case 'sy-four': return fourTex(v, FOUR[t], args);
    case 'sy-gs': return gsTex(v, tones(v.p), args);
    case 'sy-lattice': return latticeTex(v);
    case 'sy-affine': return `\\vec x = ${affineTex(v.x, v.directions)}`;
    case 'sy-space': return spaceTex(v.basis, v.ambient);
    case 'sy-text': return v.tex;
    case 'sy-eig': return eigTex(v);
    case 'sy-svd': return svdTex(v);
    case 'sy-qr': return `Q = ${matTex(v.Q, f3)}\\quad R = ${matTex(v.R, f3)}`;
    case 'sy-lu': return luTex(v);
    case 'sy-poly': return `\\det(\\lambda I - A) = ${polyTex(v.coeffs)}`;
  }
  return null;
}

// ---------------------------------------------------------------- language

const TYPES = {
  'sy-rows': ['a row picture', (v) => [...v.A.flat(), ...v.b]],
  'sy-cols': ['a column picture', (v) => [...v.b, ...v.x]],
  'sy-elim': ['an elimination', (v) => v.frame.rows.flat()],
  'sy-lstsq': ['a least-squares fit', (v) => [...v.x, ...v.p]],
  'sy-four': ['the four subspaces', (v) => [v.r]],
  'sy-gs': ['a Gram-Schmidt process', (v) => v.ws.flat()],
  'sy-lattice': ['a basis', (v) => v.vecs.flat()],
  'sy-affine': ['a solution set', (v) => [...v.p, ...v.dirs.flat()]],
  'sy-coords': ['coordinates', (v) => v.v],
  'sy-space': ['a subspace', (v) => v.basis.flat(), false],
  'sy-text': ['a message', () => [], false],
  'sy-eig': ['an eigen-decomposition', (v) => v.values.flatMap((z) => [z.re, z.im]), false],
  'sy-svd': ['an SVD', (v) => v.S, false],
  'sy-qr': ['a QR factorization', (v) => v.R.flat(), false],
  'sy-lu': ['an LU factorization', (v) => v.U.flat(), false],
  'sy-poly': ['a polynomial', (v) => v.coeffs, false],
};
for (const [type, [desc, numbers, drawable = true]] of Object.entries(TYPES)) {
  registerType(type, {
    describe: desc,
    format: (v) => (type === 'sy-text' ? v.text : desc),
    latex: (v) => readoutTex(v),
    drawable,
    numbers,
  });
}

registerFunction('rowpicture', {
  n: 2,
  f: ([A, b], name) => {
    const [M, r] = needSystem(name, A, b);
    needUnknowns(M, name);
    return rowPicture(M, r);
  },
});

registerFunction('colpicture', {
  n: 2,
  f: ([A, b], name) => {
    const [M, r] = needSystem(name, A, b);
    needFewColumns(M, name);
    return colPicture(M, r);
  },
});

registerFunction('eliminate', {
  n: [2, 3],
  f: ([A, b, k], name) => {
    const [M, r] = needSystem(name, A, b);
    needUnknowns(M, name);
    return elimination(M, r, needStep(k, name));
  },
});

registerFunction('lstsq', {
  n: 2,
  f: ([A, b], name) => {
    const [M, r] = needSystem(name, A, b);
    needFewColumns(M, name);
    return lstsqPicture(M, r);
  },
});

registerFunction('subspaces', {
  n: [1, 2],
  f: ([A, t], name) => {
    const M = needMat(A, name);
    if (M.length > 3 || M[0].length > 3) throw new Error(`${name} draws matrices up to 3×3, got ${describe(A)}`);
    return fourSubspaces(M, needStep(t, name));
  },
});

registerFunction('gramschmidt', {
  n: [2, 4],
  f: (args, name) => {
    const list = args.slice();
    const k = kindOf(list.at(-1)) === 'num' ? list.pop() : null;
    if (list.length < 2 || list.length > 3) throw new Error(`${name} needs 2 or 3 vectors, then an optional step k`);
    for (const a of list) if (kindOf(a) !== 'vec') throw new Error(`${name} needs vectors, got ${describe(a)}`);
    return gsValue(list.map((a) => a.v.slice()), k);
  },
});

registerFunction('basis', {
  n: [1, 3],
  f: (args, name) => {
    let cols;
    if (args.length === 1) cols = basisCols(args[0], name);
    else {
      for (const a of args) if (kindOf(a) !== 'vec') throw new Error(`${name} needs vectors, got ${describe(a)}`);
      cols = args.map((a) => a.v.slice());
    }
    checkBasis(cols, name);
    return { type: 'sy-lattice', vecs: cols, dim: cols.length };
  },
});

registerFunction('coords', {
  n: 2,
  f: ([v, B], name) => {
    if (!isVecLike(v)) throw new Error(`${name} needs a vector first: coords(v, B), got ${describe(v)}`);
    const cols = basisCols(B, name), c = coordsOf(v.v, cols);
    return { ...vec(pad3(c)), sy: { kind: 'coords', cols, c, v: v.v.slice() } };
  },
});

const matFn = (f) => ({ n: 1, kind: 'mat', f: ([A]) => f(A.m) });
registerFunction('rref', matFn((A) => {
  const { R, pivots } = la.rref(A);
  return { ...mat(R), sy: { kind: 'rref', pivots } };
}));
registerFunction('rank', matFn((A) => la.rank(A)));
registerFunction('tr', matFn((A) => la.trace(A)));
registerFunction('nullspace', matFn((A) => spaceValue(la.nullspace(A), A[0].length, 'null')));
registerFunction('colspace', matFn((A) => spaceValue(la.colspace(A), A.length, 'col')));
registerFunction('rowspace', matFn((A) => spaceValue(la.rowspace(A), A[0].length, 'row')));
registerFunction('leftnull', matFn((A) => spaceValue(la.leftNullspace(A), A.length, 'left')));
registerFunction('eig', matFn((A) => ({ type: 'sy-eig', ...la.eig(A) })));
registerFunction('svd', matFn((A) => ({ type: 'sy-svd', ...la.svd(A) })));
registerFunction('qr', matFn((A) => ({ type: 'sy-qr', ...la.qr(A) })));
registerFunction('lu', matFn((A) => ({ type: 'sy-lu', ...la.lu(A) })));
registerFunction('charpoly', matFn((A) => ({ type: 'sy-poly', coeffs: la.charpoly(A) })));
registerFunction('solve', {
  n: 2,
  f: ([A, b], name) => solveValue(...needSystem(name, A, b)),
});

// ---------------------------------------------------------------- items (browser)

// Follow `name` rows back to the call that made a value, e.g. P = rowpicture(A, b); P.
function callOf(src, rows, results, depth = 0) {
  const body = parseLine(src)?.body;
  if (!body || depth > 4) return null;
  if (body.t === 'call') return body;
  if (body.t !== 'name') return null;
  const i = results.findIndex((r) => r?.name === body.name);
  return i < 0 ? null : callOf(rows[i].src, rows, results, depth + 1);
}

const sameOrigin = (a, b) => len3(sub3(a || [0, 0, 0], b || [0, 0, 0])) < 1e-9;

// Per argument: {name, color, drawn}; drawn = its own row already shows that vector from our origin.
function argInfo(index, origin, rows, results) {
  const call = callOf(rows[index]?.src ?? '', rows, results);
  if (!call) return [];
  return call.args.map((a) => {
    if (a.t !== 'name') return { name: null, color: null, drawn: false };
    const i = results.findIndex((r) => r?.name === a.name);
    const row = rows[i], res = results[i];
    const drawn = !!(row && !row.hidden && res && !res.error && kindOf(res.value) === 'vec'
      && res.value.sy?.kind !== 'coords' && sameOrigin(res.origin, origin));
    return { name: a.name, color: row?.color ?? null, drawn };
  });
}

function extentPoints(it) {
  switch (it.kind) {
    case 'sy-rows': case 'sy-elim': return it.sol.kind === 'unique' ? [pad3(it.sol.x)] : [];
    case 'sy-cols': {
      const pts = [...it.cols, it.b];
      it.cols.reduce((s, c, j) => { const t = add3(s, scale3(c, it.x[j])); pts.push(t); return t; }, [0, 0, 0]);
      return pts;
    }
    case 'sy-lstsq': return [...it.cols, it.b, it.p];
    case 'sy-gs': return [...it.vs, ...it.ws];
    case 'sy-lattice': return [...it.vecs, it.vecs.reduce(add3, [0, 0, 0])];
    case 'sy-coords': return [it.sy.v, ...it.sy.cols.map((c, j) => scale3(c, it.sy.c[j]))];
    case 'sy-affine': return [it.p];
  }
  return [];
}

function tagItems(items, { rows, results }) {
  return items.map((it) => {
    const coords = it.sy?.kind === 'coords';
    if (!coords && !String(it.kind).startsWith('sy-')) return it;
    const out = { ...it };
    if (coords) out.kind = out.type = 'sy-coords'; // not a plain arrow: the value is [v]_B, not v
    out.args = argInfo(it.index, it.o, rows, results);
    out.extentPoints = extentPoints(out);
    return out;
  });
}

// ---------------------------------------------------------------- label placement

// How clear `spot` is of the `obstacles` on screen: >= 1 means no overlap, < 0 off screen.
// half = half the combined label size in NDC (x scaled by the aspect).
export function clearance(camera, spot, obstacles, half) {
  const q = spot.clone().project(camera);
  if (!(Math.abs(q.x) < 0.92 && Math.abs(q.y) < 0.92 && q.z < 1)) return -1;
  const k = camera.isPerspectiveCamera ? camera.aspect : (camera.right - camera.left) / (camera.top - camera.bottom);
  let best = Infinity;
  for (const o of obstacles) {
    const r = o.clone().project(camera);
    best = Math.min(best, Math.max((Math.abs(q.x - r.x) * k) / half[0], Math.abs(q.y - r.y) / half[1]));
  }
  return best;
}

// Index of the first clear spot (last frame's choice first, then in order), else the clearest.
export function pickSpot(camera, spots, obstacles, prev, half) {
  const order = [...spots.keys()];
  if (prev != null && prev < spots.length) order.unshift(...order.splice(prev, 1));
  let best = order[0], score = -Infinity;
  for (const i of order) {
    const sc = clearance(camera, spots[i], obstacles, half);
    if (sc >= 1) return i;
    if (sc > score) { score = sc; best = i; }
  }
  return best;
}
const lastSpot = new Map(); // `${rowId}|${what}|${i}` -> index picked last time

// ---------------------------------------------------------------- scene (browser)

function installRenderers(registerRenderer) {
  const SM = 'g-label sy-sm';

  const mesh = (c, geo, color, opacity) => new c.THREE.Mesh(geo, opacity != null && opacity < 1 ? c.mat('glass', color, opacity) : c.mat('solid', color));
  const rod = (c, a, b, color, r, opacity) => {
    const d = b.clone().sub(a), L = d.length();
    if (L < 1e-9) return;
    const m = mesh(c, c.GEO.cyl, color, opacity);
    c.placeAlong(m, a, d, L, r);
    c.add(m);
  };
  const head = (c, tip, dir, color, opacity) => {
    const L = dir.length();
    if (L < 1e-9) return;
    const len = Math.min(0.3 * c.s, 0.45 * L), m = mesh(c, c.GEO.cone, color, opacity);
    c.placeAlong(m, tip.clone().addScaledVector(dir, -len / L), dir, len, 0.085 * c.s);
    c.add(m);
  };
  const dashedArrow = (c, from, v, color, opacity = 0.9) => {
    if (v.length() < 1e-9) return;
    const tip = from.clone().add(v);
    c.lines([from, tip], color, { dashed: true, opacity });
    head(c, tip, v, color, opacity);
  };
  const rightMark = (c, corner, a, b, size, color, opacity = 0.9) => {
    if (a.lengthSq() < 1e-18 || b.lengthSq() < 1e-18 || !(size > 0)) return;
    const A = a.clone().normalize().multiplyScalar(size), B = b.clone().normalize().multiplyScalar(size);
    const p1 = corner.clone().add(A), p2 = p1.clone().add(B), p3 = corner.clone().add(B);
    for (const [x, y] of [[p1, p2], [p2, p3]]) {
      const m = new c.THREE.Mesh(c.GEO.cyl, c.mat('glass', color, opacity));
      c.placeAlong(m, x, y.clone().sub(x), size, 0.012 * c.s);
      c.add(m);
    }
    const g = c.own(new c.THREE.BufferGeometry().setFromPoints([corner, p1, p2, corner, p2, p3]));
    c.add(new c.THREE.Mesh(g, c.mat('surface', color, 0.18 * opacity)));
  };
  // Unit vector across v so labels sit beside a shaft: roughly screen-up in the default view, and
  // within the xy-plane for flat vectors (so it still shows when looking straight down).
  const across = (c, v) => {
    const z = new c.THREE.Vector3(0, 0, 1);
    const flat = Math.abs(v.z) <= 1e-9 * v.length();
    const side = flat ? z.cross(v).normalize() : z.cross(v).cross(v).negate().normalize();
    if (side.lengthSq() < 0.5) side.set(0, 1, 0);
    return side;
  };
  // Label just past the tip of v; `drop` > 0 moves it that many units to the other side of the shaft.
  const tipLabel = (c, from, v, latex, color, cls = SM, drop = 0) => {
    const tip = from.clone().add(v), L = v.length();
    if (L > 1e-9) tip.addScaledVector(v, (0.32 * c.s) / L);
    if (drop && L > 1e-9) tip.addScaledVector(across(c, v), -drop * c.s);
    return c.label(tip, latex, color, cls);
  };
  const midLabel = (c, from, v, latex, color, k = 0.5) => {
    if (v.length() < 1e-9) return;
    c.label(from.clone().addScaledVector(v, k).addScaledVector(across(c, v), 0.32 * c.s), latex, color, SM);
  };
  const polygon = (c, pts, color, opacity, edge) => {
    const tri = [];
    for (let i = 1; i + 1 < pts.length; i++) tri.push(pts[0], pts[i], pts[i + 1]);
    c.add(new c.THREE.Mesh(c.own(new c.THREE.BufferGeometry().setFromPoints(tri)), c.mat('surface', color, opacity)));
    if (edge != null) c.polyline([...pts, pts[0]], color, edge >= 1 ? {} : { opacity: edge });
  };
  const at = (c, O, p) => O.clone().add(c.v3(p));
  const fade = (obj, alpha) => { if (obj && alpha < 1) obj.element.style.opacity = String(alpha); };

  // A label spot on a clipped plane: toward the top-right corner as seen from the default view.
  const planeSpot = (poly, i = 0) => {
    let best = poly[0], sc = -Infinity;
    for (const p of poly) {
      const v = 2 * p[2] + p[1] - 0.6 * p[0];
      if (v > sc) { sc = v; best = p; }
    }
    const ctr = scale3(poly.reduce(add3, [0, 0, 0]), 1 / poly.length);
    return lerp3(best, ctr, 0.2 + 0.12 * i);
  };
  // More spots on a clipped plane for when that one is taken: in from each corner, up/right first.
  const planeSpots = (poly, i) => {
    const ctr = scale3(poly.reduce(add3, [0, 0, 0]), 1 / poly.length), score = (p) => 2 * p[2] + p[1] - 0.6 * p[0];
    const ranked = poly.slice().sort((p, q) => score(q) - score(p));
    return [planeSpot(poly, i), ...ranked.flatMap((p) => [lerp3(p, ctr, 0.25), lerp3(p, ctr, 0.45)])];
  };

  // Equations [a | beta] as clipped planes (n = 3) or lines in the xy-plane (n = 2).
  // emph[i]: 0 plain, 1 source row, 2 changing row. Returns candidate label spots (or null) per row.
  const drawFlats = (c, O, rows, n, colors, emph = []) => rows.map((row, i) => {
    const a = pad3(row.slice(0, n)), beta = row[n], e = emph[i] ?? 0;
    if (len3(a) <= 1e-9 * Math.max(1, maxAbs(row))) return null;
    if (n === 3) {
      const poly = planeInCube(a, beta, c.E);
      if (!poly) return null;
      polygon(c, poly.map((p) => at(c, O, p)), colors[i], [0.2, 0.26, 0.36][e], [0.6, 0.85, 1][e]);
      return planeSpots(poly, i);
    }
    const L2 = a[0] * a[0] + a[1] * a[1], p = [(a[0] * beta) / L2, (a[1] * beta) / L2, 0], d = [-a[1], a[0], 0];
    const t = clipLine3(p, d, c.E);
    if (!t) return null;
    const P = add3(p, scale3(d, t[0])), Q = add3(p, scale3(d, t[1]));
    rod(c, at(c, O, P), at(c, O, Q), colors[i], [0.03, 0.036, 0.048][e] * c.s);
    const end = (P[1] - 0.6 * P[0]) > (Q[1] - 0.6 * Q[0]) ? P : Q, other = P === end ? Q : P;
    return [0.1 + 0.08 * i, 0.25, 0.4, 0.9, 0.75].map((f) => add3(lerp3(end, other, f), [0, 0, 0.3 * c.s]));
  });

  // One spot per row from its candidates, each clear on screen of those already placed (by this
  // row or an earlier row picture / elimination in the same build) and of `taken`.
  const builds = new WeakMap(); // ctx (one per build) -> equation label spots placed so far
  const pickLabels = (c, O, key, cands, taken, half) => {
    if (!builds.has(c)) builds.set(c, []);
    const placed = builds.get(c), others = [...taken];
    return cands.map((list, i) => {
      if (!list) return null;
      const spots = list.map((p) => at(c, O, p)), k = pickSpot(c.camera, spots, [...others, ...placed], lastSpot.get(`${key}|${i}`), half);
      lastSpot.set(`${key}|${i}`, k);
      placed.push(spots[k]);
      return spots[k];
    });
  };
  // The unique solution's dot and its label, which the equation labels keep clear of.
  const solutionSpots = (c, O, sol) => {
    if (sol.kind !== 'unique') return [];
    const P = at(c, O, pad3(sol.x));
    return [P, P.clone().add(new c.THREE.Vector3(0.2 * c.s, 0.35 * c.s, 0.4 * c.s))];
  };

  // Where pairs of equations meet: lines (3D) or points (2D). Returns the spots for a label.
  const drawMeets = (c, O, rows, n, inconsistent) => {
    const fg = c.colors.fg, spots = [];
    for (let i = 0; i < rows.length; i++) {
      for (let j = i + 1; j < rows.length; j++) {
        const a = pad3(rows[i].slice(0, n)), b = pad3(rows[j].slice(0, n));
        if (n === 3) {
          const L = meetPlanes(a, rows[i][n], b, rows[j][n]);
          const t = L && clipLine3(L.p, L.d, c.E);
          if (!t) continue;
          const P = add3(L.p, scale3(L.d, t[0])), Q = add3(L.p, scale3(L.d, t[1]));
          c.lines([at(c, O, P), at(c, O, Q)], fg, { opacity: inconsistent ? 0.85 : 0.45, dashed: inconsistent });
          spots.push(lerp3(P, Q, 0.5));
        } else if (inconsistent) {
          const p = meetLines(a, rows[i][n], b, rows[j][n]);
          if (!p || Math.abs(p[0]) > c.E || Math.abs(p[1]) > c.E) continue;
          c.dot(at(c, O, p), fg, 0.08 * c.s, 0.45);
          spots.push(p);
        }
      }
    }
    return spots;
  };

  const drawSolution = (c, O, sol, n, label = true) => {
    const fg = c.colors.fg;
    if (sol.kind === 'unique') {
      const P = at(c, O, pad3(sol.x));
      c.dot(P, fg, 0.1 * c.s);
      c.dot(P, fg, 0.2 * c.s, 0.18);
      // up and to the right, so the label clears the point in the top view too
      if (label) c.label(P.clone().add(new c.THREE.Vector3(0.2 * c.s, 0.35 * c.s, 0.4 * c.s)), tupleTex(sol.x), fg, SM);
      return;
    }
    if (sol.kind !== 'infinite') return;
    const p = pad3(sol.x), dirs = sol.directions.map(pad3);
    if (dirs.length === 1) {
      const t = clipLine3(p, dirs[0], c.E);
      if (t) rod(c, at(c, O, add3(p, scale3(dirs[0], t[0]))), at(c, O, add3(p, scale3(dirs[0], t[1]))), fg, 0.042 * c.s);
    } else if (dirs.length === 2 && n === 3) {
      const nv = cross3(dirs[0], dirs[1]), poly = planeInCube(nv, dot3(nv, p), c.E);
      if (poly) polygon(c, poly.map((q) => at(c, O, q)), fg, 0.1, 1);
    } else if (dirs.length === 2) {
      polygon(c, [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([x, y]) => at(c, O, [x * c.E, y * c.E, 0])), fg, 0.08, 1);
    }
  };

  const noCommon = (c, O, spots, n) => {
    let p = spots.length ? scale3(spots.reduce(add3, [0, 0, 0]), 1 / spots.length) : n === 3 ? [0, 0, 0.8 * c.E] : [0, 0.8 * c.E, 0];
    p = p.map((x) => clamp(x, -0.85 * c.E, 0.85 * c.E));
    c.label(at(c, O, add3(p, [0, 0, 0.5 * c.s])), '\\text{no common point}', c.colors.fg, SM);
  };

  // A subspace through O spanned by `basis` (3D arrays), clipped to the cube: line or plane.
  const drawSpanFlat = (c, O, basis, color, opacity) => {
    if (basis.length === 1) {
      const t = clipLine3([0, 0, 0], basis[0], c.E);
      if (!t) return null;
      rod(c, at(c, O, scale3(basis[0], t[0])), at(c, O, scale3(basis[0], t[1])), color, 0.016 * c.s, 0.55);
      return add3(scale3(basis[0], 0.8 * t[1]), [0, 0, 0.3 * c.s]);
    }
    if (basis.length !== 2) return null;
    const poly = planeInCube(cross3(basis[0], basis[1]), 0, c.E);
    if (!poly) return null;
    polygon(c, poly.map((p) => at(c, O, p)), color, opacity, 0.45);
    return planeSpot(poly);
  };

  registerRenderer('sy-rows', (it, c) => {
    const O = c.v3(it.o), tones = shades(it.color, it.m, c.theme);
    const rows = it.A.map((r, i) => [...r, it.b[i]]);
    const spots = pickLabels(c, O, `${it.rowId}|eq`, drawFlats(c, O, rows, it.n, tones), solutionSpots(c, O, it.sol), [0.26, 0.075]);
    spots.forEach((p, i) => {
      if (!p) return;
      const [l, r] = equationParts(it.A[i], it.b[i]);
      c.label(p, `${l} = ${r}`, tones[i], SM);
    });
    const none = it.sol.kind === 'none';
    const meets = drawMeets(c, O, rows, it.n, none);
    drawSolution(c, O, it.sol, it.n);
    if (none) noCommon(c, O, meets, it.n);
  });

  registerRenderer('sy-elim', (it, c) => {
    const O = c.v3(it.o), fr = it.frame, base = shades(it.color, it.m, c.theme), tones = base.slice();
    if (fr.swap) {
      const [a, b] = fr.swap;
      tones[a] = mixHex(base[a], base[b], fr.f);
      tones[b] = mixHex(base[b], base[a], fr.f);
    }
    const emph = [];
    if (fr.op !== 'start') {
      emph[fr.opRows[0]] = 2;
      if (fr.op === 'add') emph[fr.opRows[1]] = 1;
      if (fr.op === 'swap') emph[fr.opRows[1]] = 2;
    }
    const spots = pickLabels(c, O, `${it.rowId}|R`, drawFlats(c, O, fr.rows, it.n, tones, emph), solutionSpots(c, O, it.sol), [0.1, 0.075]);
    spots.forEach((p, i) => { if (p) c.label(p, `R_{${i + 1}}`, tones[i], SM); });
    drawMeets(c, O, fr.rows, it.n, false);
    drawSolution(c, O, it.sol, it.n);
    // a final row 0 = c ≠ 0: the plane has gone off to infinity
    if (it.sol.kind === 'none' && !fr.pending && fr.s === fr.N) noCommon(c, O, [], it.n);
  });

  registerRenderer('sy-cols', (it, c) => {
    const O = c.v3(it.o), fg = c.colors.fg, tones = shades(it.color, it.n, c.theme);
    const [aA, aB] = it.args ?? [], a = colLetter(aA);
    if (it.rank > 0 && it.rank < it.m) drawSpanFlat(c, O, it.basis, it.color, 0.1);
    const first = it.x.findIndex((x, j) => Math.abs(x) > 1e-9 && len3(it.cols[j]) > 1e-9);
    it.cols.forEach((col, j) => {
      const V = c.v3(col);
      if (V.length() < 1e-9 || (j === first && Math.abs(it.x[j] - 1) < 1e-9)) return; // the piece is this very arrow
      c.arrow(O, V, tones[j], { opacity: 0.35, thickness: 0.6, head: 0.75 });
      tipLabel(c, O, V, `\\vec ${a}_{${j + 1}}`, tones[j], SM, 0.3); // below the shaft; x_j a_j goes above
    });
    const tip = O.clone();
    it.cols.forEach((col, j) => {
      const piece = c.v3(col).multiplyScalar(it.x[j]);
      if (piece.length() < 1e-9) return;
      c.arrow(tip, piece, tones[j], { thickness: 0.9 });
      midLabel(c, tip, piece, `${coefTex(it.x[j])}\\vec ${a}_{${j + 1}}`, tones[j]);
      tip.add(piece);
    });
    const Bv = c.v3(it.b), T = O.clone().add(Bv);
    if (!aB?.drawn) {
      c.arrow(O, Bv, aB?.color ?? it.color, { thickness: 1.1 });
      tipLabel(c, O, Bv, argVec(aB, '\\vec b'), aB?.color ?? it.color, 'g-label');
    }
    if (it.exact) return;
    const P = at(c, O, it.p), gap = T.clone().sub(P);
    c.dot(P, fg, 0.075 * c.s);
    c.label(P.clone().add(new c.THREE.Vector3(0, 0, -0.42 * c.s)), `${nameTex(aA?.name ?? 'A', false)}\\hat x`, fg, SM);
    c.lines([P, T], fg, { dashed: true, opacity: 0.95 });
    const inPlane = c.v3(it.p).length() > 1e-6 ? c.v3(it.p).negate() : c.v3(it.basis[0] ?? [1, 0, 0]);
    rightMark(c, P, gap, inPlane, Math.min(0.28 * c.s, 0.4 * gap.length()), fg);
  });

  registerRenderer('sy-lstsq', (it, c) => {
    const O = c.v3(it.o), fg = c.colors.fg, tones = shades(it.color, it.n + 1, c.theme);
    const [aA, aB] = it.args ?? [], a = colLetter(aA), A = matName(aA);
    const spot = drawSpanFlat(c, O, it.basis, it.color, 0.16);
    if (spot) c.label(at(c, O, spot), `C(${A})`, it.color, SM);
    it.cols.forEach((col, j) => {
      const V = c.v3(col);
      if (V.length() < 1e-9) return;
      c.arrow(O, V, tones[j], { opacity: 0.4, thickness: 0.6, head: 0.75 });
      tipLabel(c, O, V, `\\vec ${a}_{${j + 1}}`, tones[j], SM, 0.3);
    });
    const Bv = c.v3(it.b), P = c.v3(it.p), T = O.clone().add(Bv), F = O.clone().add(P);
    const bColor = aB?.color ?? tones[it.n];
    if (!aB?.drawn) {
      c.arrow(O, Bv, bColor, { thickness: 1.05 });
      tipLabel(c, O, Bv, argVec(aB, '\\vec b'), bColor, 'g-label');
    }
    c.arrow(O, P, it.color, { thickness: 1.2 });
    tipLabel(c, O, P, `\\vec p = ${A}\\hat x`, it.color);
    const e = T.clone().sub(F);
    if (e.length() < 1e-6 * Math.max(1, Bv.length())) return;
    c.lines([F, T], fg, { dashed: true, opacity: 0.95 });
    const inPlane = P.length() > 1e-6 ? P.clone().negate() : c.v3(it.basis[0] ?? [1, 0, 0]);
    rightMark(c, F, e, inPlane, Math.min(0.3 * c.s, 0.4 * e.length(), P.length() > 1e-6 ? 0.4 * P.length() : Infinity), fg);
    midLabel(c, F, e, '\\vec e = \\vec b - \\vec p', fg);
  });

  // One subspace (basis in R^amb) around C, `size` = half-extent of the drawn piece. The label goes
  // to the candidate spot farthest from `avoid` (or the top-right one); returns that spot.
  const drawSub = (c, C, basis, amb, color, size, alpha, latex, avoid = null) => {
    const vecs = basis.map(pad3), k = vecs.length, op = (x) => Math.max(0.04, x * alpha);
    const solidOp = alpha < 1 ? Math.max(0.05, alpha) : undefined;
    const square = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
    let spots;
    if (k === 0) {
      c.dot(C, color, 0.1 * c.s, solidOp);
      spots = [C.clone().add(new c.THREE.Vector3(0, 0, 0.45 * c.s))];
    } else if (k === amb && amb === 3) {
      const h = 0.8 * size, pts = [];
      for (const [i, j] of CUBE_EDGES) pts.push(at(c, C, cubeCorner(i, h)), at(c, C, cubeCorner(j, h)));
      c.lines(pts, color, { opacity: op(0.7) });
      spots = [at(c, C, [h, h, h + 0.3 * c.s]), at(c, C, [-h, -h, h + 0.3 * c.s])];
    } else if (k === amb && amb === 2) {
      polygon(c, square.map(([x, y]) => at(c, C, [x * size, y * size, 0])), color, op(0.16), op(0.8));
      spots = square.map(([x, y]) => at(c, C, [0.8 * x * size, 0.8 * y * size, 0.3 * c.s]));
    } else if (k === 1) {
      const d = c.v3(vecs[0]).normalize();
      rod(c, C.clone().addScaledVector(d, -size), C.clone().addScaledVector(d, size), color, 0.032 * c.s, solidOp);
      spots = [1, -1].map((sg) => C.clone().addScaledVector(d, sg * (size + 0.4 * c.s)));
    } else {
      const [e1, e2] = c.orthoBasis(vecs).basis;
      polygon(c, square.map(([x, y]) => C.clone().addScaledVector(e1, x * size).addScaledVector(e2, y * size)), color, op(0.26), op(0.9));
      spots = square.map(([x, y]) => C.clone().addScaledVector(e1, 0.75 * x * size).addScaledVector(e2, 0.75 * y * size));
    }
    // compare on screen (camera at build time): far from the `avoid` spots, else up and to the right
    const scr = (p) => p.clone().project(c.camera), sa = (avoid ?? []).map(scr), sc = scr(C);
    const score = (p) => {
      const q = scr(p);
      return sa.length ? Math.min(...sa.map((a) => Math.hypot(q.x - a.x, q.y - a.y))) : 2 * (q.y - sc.y) + (q.x - sc.x);
    };
    const spot = spots.reduce((best, p) => (score(p) > score(best) ? p : best));
    if (latex && alpha > 0.3) fade(c.label(spot, latex, color, SM), alpha);
    return spot;
  };

  registerRenderer('sy-four', (it, c) => {
    const pal = FOUR[c.theme === 'light' ? 'light' : 'dark'], fg = c.colors.fg, O = c.v3(it.o);
    const A = matName(it.args?.[0]), T = `${A}^{T}`;
    const groups = fourLayout(it.t, c.E);
    for (const g of groups) {
      if (g.alpha < 0.02) continue;
      const C = at(c, O, g.center), dom = g.which === 'domain', amb = dom ? it.n : it.m;
      const [U, W] = dom ? [it.row, it.nul] : [it.col, it.lnul];
      const [cu, cw] = dom ? [pal.row, pal.nul] : [pal.col, pal.lnul];
      const [tu, tw] = dom ? [`C(${T})`, `N(${A})`] : [`C(${A})`, `N(${T})`];
      const avoid = [];
      if (g.apart) {
        // a small frame for each space: its own axes (and the plane itself for R^2)
        const h = 1.12 * g.size, axes = [], caption = at(c, C, [0, 0, -g.size - 0.6 * c.s]);
        for (let a = 0; a < Math.min(3, amb); a++) {
          const d = [0, 0, 0];
          d[a] = h;
          axes.push(at(c, C, scale3(d, -1)), at(c, C, d));
        }
        c.lines(axes, c.colors.text, { opacity: 0.55 });
        if (amb === 2) c.polyline([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]].map(([x, y]) => at(c, C, [x * h, y * h, 0])), c.colors.text, { opacity: 0.45 });
        c.label(caption, `\\mathbb{R}^{${amb}}\\ \\text{(${dom ? 'inputs' : 'outputs'})}`, fg, SM);
        avoid.push(caption);
      }
      const su = drawSub(c, C, U, amb, cu, g.size, g.alpha, tu, avoid);
      drawSub(c, C, W, amb, cw, g.size, g.alpha, tw, [...avoid, su]);
      if (U.length > 0 && U.length < amb && W.length > 0 && W.length < amb) {
        const du = c.orthoBasis(U.map(pad3)).basis[0], dw = c.orthoBasis(W.map(pad3)).basis[0];
        rightMark(c, C, du, dw, (g.apart ? 0.45 : 0.6) * c.s, fg, 0.9 * g.alpha);
      }
    }
    if (groups[0].apart) {
      // A carries the inputs to the outputs
      const from = at(c, O, add3(groups[0].center, [0, 0.2 * c.E, 0.5 * c.E]));
      const to = at(c, O, add3(groups[1].center, [0, -0.2 * c.E, 0.5 * c.E]));
      const mid = from.clone().add(to).multiplyScalar(0.5).add(new c.THREE.Vector3(0, 0, 0.3 * c.E));
      const curve = new c.THREE.QuadraticBezierCurve3(from, mid, to), pts = curve.getPoints(24);
      c.polyline(pts, fg, { opacity: 0.75 });
      head(c, pts.at(-1), curve.getTangent(1), fg, 0.75);
      c.label(curve.getPoint(0.3).add(new c.THREE.Vector3(0, 0, 0.35 * c.s)), A, fg, SM);
    }
  });

  registerRenderer('sy-gs', (it, c) => {
    const O = c.v3(it.o), { vs, ws, qs, projs, coefs, p } = it, { s, f } = it.frame;
    const tones = shades(it.color, p, c.theme), args = it.args ?? [];
    const W = (j) => `\\vec w_{${j + 1}}`;
    vs.forEach((v, j) => {
      const a = args[j];
      if (a?.drawn) return;
      const V = c.v3(v), col = a?.color ?? tones[j];
      c.arrow(O, V, col, { opacity: 0.3, thickness: 0.7 });
      tipLabel(c, O, V, argVec(a, `\\vec v_{${j + 1}}`), col);
    });
    const norm = s === p + 1 ? 1 : s === p ? f : 0; // progress toward unit length
    for (let j = 0; j < p; j++) {
      const done = s >= j + 1, active = s === j && f > 0;
      if (!done && !active) continue;
      const V = c.v3(vs[j]);
      if (active || (s === j + 1 && f === 0)) {
        // the projections being taken away (dashed, along earlier w's) and the path v_j -> w_j
        // drawn in the colour of the vector being built, so they stand out on the w_i they lie along
        const k = active ? f : 1, path = [O.clone().add(V)];
        projs[j].forEach((pv, i) => {
          if (!pv || len3(pv) < 1e-9) return;
          const P = c.v3(pv).multiplyScalar(k);
          dashedArrow(c, O, P, tones[j], 0.95);
          if (!active) midLabel(c, O, P, `${coefTex(coefs[j][i])}${W(i)}`, tones[j], 0.45);
          path.push(path.at(-1).clone().sub(P));
        });
        if (path.length > 1) c.polyline(path, tones[j], { dashed: true, opacity: 0.8 });
        if (active) {
          c.arrow(O, path.at(-1).clone().sub(O), tones[j], { opacity: 0.8 });
          continue;
        }
      }
      if (!qs[j]) {
        c.dot(O, tones[j], 0.1 * c.s);
        c.label(O.clone().add(new c.THREE.Vector3(0, 0, -0.45 * c.s)), `${W(j)} = \\vec 0`, tones[j], SM);
        continue;
      }
      const Wv = c.v3(ws[j]).lerp(c.v3(qs[j]), norm);
      c.arrow(O, Wv, tones[j], { thickness: s === p + 1 ? 1.15 : 1 });
      // w_j = v_j (nothing to take away, e.g. w_1): its label goes below the shaft, clear of v_j's
      const drop = Wv.distanceTo(c.v3(vs[j])) < 0.3 * c.s ? 0.5 : 0;
      tipLabel(c, O, Wv, s === p + 1 ? `\\hat q_{${j + 1}}` : W(j), tones[j], 'g-label', drop);
      // right angles with the earlier results, shown for the newest one (all of them at the end)
      if (s === j + 1 || s >= p) {
        for (let i = 0; i < j; i++) {
          if (!qs[i]) continue;
          const Wi = c.v3(ws[i]).lerp(c.v3(qs[i]), norm);
          rightMark(c, O, Wv, Wi, Math.min(0.3 * c.s, 0.3 * Wv.length(), 0.3 * Wi.length()), c.colors.fg, 0.8);
        }
      }
    }
    if (norm > 0) {
      const live = qs.filter(Boolean);
      if (live.length >= 3) c.dot(O, c.colors.fg, 1, 0.07 * norm);
      else if (live.length === 2) {
        const e1 = c.v3(live[0]), e2 = c.v3(live[1]), ring = [];
        for (let k = 0; k <= 64; k++) {
          const t = (2 * Math.PI * k) / 64;
          ring.push(O.clone().addScaledVector(e1, Math.cos(t)).addScaledVector(e2, Math.sin(t)));
        }
        c.polyline(ring, c.colors.fg, { dashed: true, opacity: 0.6 * norm });
      }
    }
  });

  registerRenderer('sy-lattice', (it, c) => {
    const O = c.v3(it.o), tones = shades(it.color, it.dim, c.theme), vs = it.vecs, args = it.args ?? [];
    const segs = latticeSegments(vs, c.E);
    if (segs.length) c.lines(segs.flatMap(([p, q]) => [at(c, O, p), at(c, O, q)]), it.color, { opacity: it.dim === 3 ? 0.22 : 0.45 });
    if (it.dim === 2) {
      const poly = planeInCube(cross3(vs[0], vs[1]), 0, c.E);
      if (poly) polygon(c, poly.map((p) => at(c, O, p)), it.color, 0.05, null);
      polygon(c, [[0, 0, 0], vs[0], add3(vs[0], vs[1]), vs[1]].map((p) => at(c, O, p)), it.color, 0.22, 0.9);
    } else {
      const M = new c.THREE.Matrix4().makeBasis(c.v3(vs[0]), c.v3(vs[1]), c.v3(vs[2])).setPosition(O);
      for (const obj of [new c.THREE.Mesh(c.GEO.box, c.mat('surface', it.color, 0.12)), new c.THREE.LineSegments(c.GEO.boxEdges, c.mat('edge', it.color))]) {
        obj.matrixAutoUpdate = false;
        obj.matrix.copy(M);
        c.add(obj);
      }
    }
    const letter = args.length === 1 ? colLetter(args[0], 'b') : 'b';
    vs.forEach((v, j) => {
      const a = args.length > 1 ? args[j] : null;
      if (a?.drawn) return;
      const V = c.v3(v);
      c.arrow(O, V, tones[j], { thickness: 1.05 });
      tipLabel(c, O, V, argVec(a, `\\vec ${letter}_{${j + 1}}`), tones[j], 'g-label');
    });
  });

  registerRenderer('sy-coords', (it, c) => {
    const O = c.v3(it.o), { cols, c: co, v } = it.sy, k = cols.length, tones = shades(it.color, k, c.theme);
    const [aV, aB] = it.args ?? [], letter = colLetter(aB, 'b');
    const pieces = cols.map((b, j) => c.v3(scale3(b, co[j])));
    const corner = (mask) => pieces.reduce((acc, P, j) => (mask & (1 << j) ? acc.add(P) : acc), O.clone());
    const box = [];
    for (let mask = 0; mask < 1 << k; mask++) {
      for (let j = 0; j < k; j++) {
        if (mask & (1 << j) || mask === (1 << j) - 1) continue; // the tip-to-tail path is drawn solid
        box.push(corner(mask), corner(mask | (1 << j)));
      }
    }
    if (box.length) c.lines(box, it.color, { dashed: true, opacity: 0.55 });
    const tip = O.clone();
    pieces.forEach((P, j) => {
      if (P.length() < 1e-9) return;
      c.arrow(tip, P, tones[j], { thickness: 0.85 });
      midLabel(c, tip, P, `${coefTex(co[j])}\\vec ${letter}_{${j + 1}}`, tones[j]);
      tip.add(P);
    });
    if (!aV?.drawn) {
      const V = c.v3(v);
      c.arrow(O, V, it.color, { thickness: 1.1 });
      tipLabel(c, O, V, argVec(aV, '\\vec v'), it.color, 'g-label');
    }
  });

  registerRenderer('sy-affine', (it, c) => {
    const O = c.v3(it.o), p = it.p, dirs = it.dirs;
    if (dirs.length === 1) {
      const t = clipLine3(p, dirs[0], c.E);
      if (t) rod(c, at(c, O, add3(p, scale3(dirs[0], t[0]))), at(c, O, add3(p, scale3(dirs[0], t[1]))), it.color, 0.035 * c.s);
    } else if (dirs.length === 2 && it.n === 3) {
      const nv = cross3(dirs[0], dirs[1]), poly = planeInCube(nv, dot3(nv, p), c.E);
      if (poly) polygon(c, poly.map((q) => at(c, O, q)), it.color, 0.22, 1);
    } else if (dirs.length === 2) {
      polygon(c, [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([x, y]) => at(c, O, [x * c.E, y * c.E, 0])), it.color, 0.12, 1);
    }
    const P = at(c, O, p);
    c.dot(P, it.color, 0.085 * c.s);
    if (it.label) c.label(P.clone().add(new c.THREE.Vector3(0, 0, 0.4 * c.s)), `${nameTex(it.label, true)}_p`, it.color, SM);
  });
}

// ---------------------------------------------------------------- install

const CSS = `
.g-label.sy-sm { font-size: 16px; }
`;
const HELP = `<h4 class="ui-overline">Systems and subspaces</h4>
<p><code>rowpicture(A, b)</code> <code>colpicture(A, b)</code> <code>eliminate(A, b, k)</code> (k: step slider)
<code>lstsq(A, b)</code> <code>subspaces(A)</code> or <code>subspaces(A, t)</code> <code>gramschmidt(u, v, w, k)</code>
<code>basis(b1, b2)</code> <code>coords(v, B)</code></p>
<p>Readouts: ${'solve rref rank nullspace colspace rowspace leftnull eig svd qr lu charpoly tr'.split(' ').map((f) => `<code>${f}</code>`).join(' ')}</p>`;

// Which argument is the step slider, and its natural range.
const STEP_ARG = {
  'sy-elim': (v) => [2, 0, Math.max(1, v.frame.N)],
  'sy-gs': (v) => [v.p, 0, v.p + 1],
  'sy-four': () => [1, 0, 1],
};

export async function install(api) {
  const { registerRenderer } = await import('../scene.js');
  installRenderers(registerRenderer);
  api.addStyles(CSS);
  document.getElementById('g-help')?.insertAdjacentHTML('beforeend', HELP);
  api.addItemsHook(tagItems);

  const theme = () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  const html = new Map();
  const render = (tex) => {
    let h = html.get(tex);
    if (h == null) {
      h = katex.renderToString(tex, { throwOnError: false });
      if (html.size > 400) html.clear();
      html.set(tex, h);
    }
    return h;
  };
  const ours = (v) => v && typeof v === 'object' && (v.sy || String(v.type).startsWith('sy-'));
  api.addRowDecorator((row, res, el) => {
    if (!res || res.error || !ours(res.value) || typeof katex === 'undefined') return;
    const args = argInfo(api.rows.indexOf(row), res.origin, api.rows, api.results);
    const tex = readoutTex(res.value, { color: row.color, theme: theme(), args });
    if (tex != null) el.out.innerHTML = render(tex);
  });

  // Give an untouched step slider the range of its stepper (eliminate: 0..steps, etc.).
  const ranged = new Map(); // slider row id -> "min,max" we set
  // Several steppers may share one slider: merge their ranges first, or they would keep
  // overwriting each other and re-triggering recompute forever.
  api.onRecompute((results, rows) => {
    const want = new Map(); // slider row -> [lo, hi] over every stepper using it
    rows.forEach((r, i) => {
      const res = results[i], v = res?.value;
      const spec = v && !res.error && STEP_ARG[v.type]?.(v);
      if (!spec) return;
      const [argIdx, lo, hi] = spec;
      const arg = callOf(r.src, rows, results)?.args?.[argIdx];
      if (arg?.t !== 'name') return;
      const j = results.findIndex((x) => x?.name === arg.name), sr = rows[j];
      if (!sr || results[j].slider == null) return;
      const w = want.get(sr);
      want.set(sr, w ? [Math.min(w[0], lo), Math.max(w[1], hi)] : [lo, hi]);
    });
    let changed = false;
    for (const [sr, [lo, hi]] of want) {
      const key = `${lo},${hi}`, now = `${sr.min},${sr.max}`, mine = ranged.get(sr.id);
      if (now === key || !((sr.min == null && sr.max == null) || now === mine)) continue;
      sr.min = lo;
      sr.max = hi;
      ranged.set(sr.id, key);
      changed = true;
    }
    if (changed) queueMicrotask(() => api.recompute());
  });

  // Readout colours follow the theme.
  new MutationObserver(() => { if (api.results.some((r) => ours(r?.value))) api.recompute(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // Equation and subspace labels are placed for the camera at build time: redo them once a turn settles.
  const placed = new Set(['sy-rows', 'sy-elim', 'sy-four']);
  api.onSceneReady((scene) => {
    let key = '', since = 0, dirty = false;
    scene.onFrame(() => {
      const cam = scene.camera, now = performance.now();
      const k = [...cam.position.toArray(), ...cam.quaternion.toArray(), cam.zoom].map((x) => x.toFixed(3)).join();
      if (k !== key) { key = k; since = now; dirty = true; return; }
      if (!dirty || now - since < 250) return;
      dirty = false;
      if (scene.items.some((it) => placed.has(it.kind))) scene.rebuild();
    });
  });
}
