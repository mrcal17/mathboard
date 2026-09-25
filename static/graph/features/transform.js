// Linear maps as motion (3Blue1Brown style) for the 3D tab.
//   transform(A, t)     space under M(t) = (1-t)I + tA: the warped integer lattice, the images of
//                       i, j, k (the columns of M), the unit cube with its signed volume det M, and the
//                       null space collapsing when A is singular. A may be 2×2 (drawn in the xy-plane).
//   transform(A, B, t)  t in [0, 2]: I -> A over [0, 1], then A -> BA (B applied after A).
//   fixed(x)            x is drawn unmoved; every other vector, point, span, plane, parallelogram and
//                       parallelepiped row is carried along by the first visible transform row.
//   eigen(A)            real eigenlines labelled with λ; complex pairs as their invariant plane with a
//                       spiral cue. Vectors on an eigenline glow while a transform keeps them on it.
//   svdview(A[, t])     unit sphere -> ellipsoid A·S², axes σᵢuᵢ, right singular vectors vᵢ on the
//                       sphere; t in [0, 3] morphs in stages: rotate by Vᵀ, stretch by Σ, rotate by U.
// A slider passed as t gets the range 0..1 (0..2, 0..3) unless its min/max were already set.
// Other features can make their own item kinds ride along with registerCarrier(kind, fn).
// Pure helpers are exported for tests; scene.js loads in install().
import { registerFunction, registerType, values, formatNumber, parseLine } from '../lang.js';
import * as la from '../linalg.js';

const { kindOf, describe } = values;

export const GLOW_TOL = 2e-3; // |sin angle| below which a vector counts as lying on an eigenline
// images of i, j, k; v1/σ1u1, v2/σ2u2, v3/σ3u3 (darker on the whiteboard so labels stay readable)
const BASIS = { dark: ['#43b05c', '#e05a4f', '#4a90e2'], light: ['#2e8b45', '#e05a4f', '#4a90e2'] };
const HATS = ['\\hat\\imath', '\\hat\\jmath', '\\hat k'];
const SV = { dark: ['#f5a623', '#26b5b5', '#e056a0'], light: ['#c77700', '#138f8f', '#c2307f'] };
const FLIP_TEX = '#b05ce0';
const TONES = {
  dark: { pos: '#f5c542', neg: '#c77dff', nul: '#ff4d6d', glow: '#ffe066' },
  light: { pos: '#a87700', neg: '#9446c8', nul: '#d7263d', glow: '#f0a000' },
};
const MINE = new Set(['tf-map', 'tf-eigen', 'tf-svd', 'tf-glow']);
const pal = (set, theme) => set[theme] ?? set.dark;

// ---------------------------------------------------------------- small math

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const pad3 = (v) => [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const unit3 = (a) => { const L = len3(a); return L > 0 ? scale3(a, 1 / L) : a.slice(); };
const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const column = (M, j) => M.map((r) => r[j]);
const fromCols = (cs) => [0, 1, 2].map((i) => cs.map((c) => c[i]));
const transpose3 = (A) => fromCols(A);
const diag3 = (d) => [[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]];
const maxAbs = (M) => M.reduce((m, r) => Math.max(m, ...r.map(Math.abs)), 0);
const lerpM = (A, B, s) => A.map((r, i) => r.map((x, j) => x + s * (B[i][j] - x)));

export const mulMV = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * (v[2] ?? 0));
export const mulMM = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
export const det3 = (M) => dot3(M[0], cross3(M[1], M[2]));

// Short numbers for readouts and labels: `d` decimals, no trailing zeros, no "-0".
export function fmt(x, d = 2) {
  if (!Number.isFinite(x)) return formatNumber(x);
  const s = String(Number(x.toFixed(d)));
  return s === '-0' ? '0' : s;
}

function perpPair(n) {
  const u = unit3(n);
  const e1 = unit3(cross3(u, Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  return [e1, cross3(u, e1)];
}

// Distance from w to span(basis), basis orthonormal.
function residual(w, basis) {
  let r = w.slice();
  for (const b of basis) r = sub3(r, scale3(b, dot3(r, b)));
  return len3(r);
}

// Scale an eigenvector so its largest entry is ±1: (0.7071, 0.7071) -> (1, 1).
export function niceDir(v) {
  const m = v.reduce((a, x) => (Math.abs(x) > Math.abs(a) ? x : a), 0);
  return m ? v.map((x) => x / Math.abs(m)) : v.slice();
}

// 2×2 matrices act on the xy-plane and leave z alone.
export function embed3(m, what = 'transform') {
  const n = m.length;
  if (n === 3 && m.every((r) => r.length === 3)) return m.map((r) => r.slice());
  if (n === 2 && m.every((r) => r.length === 2)) return [[m[0][0], m[0][1], 0], [m[1][0], m[1][1], 0], [0, 0, 1]];
  throw new Error(`${what} needs a 2×2 or 3×3 matrix, got a ${n}×${m[0].length} matrix`);
}

// Rotation matrix <-> axis-angle (Rodrigues). rotLog handles angles near 0 and π.
export function rotExp(axis, angle) {
  const [x, y, z] = unit3(axis), c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
}

export function rotLog(R) {
  const angle = Math.acos(clamp((R[0][0] + R[1][1] + R[2][2] - 1) / 2, -1, 1));
  const skew = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (angle < 1e-9) return { axis: [0, 0, 1], angle: 0 };
  if (angle < Math.PI - 1e-3) return { axis: unit3(skew), angle };
  // near π the skew part vanishes: read the axis off sym(R) = cos θ I + (1 - cos θ) a aᵀ instead
  const c = Math.cos(angle);
  const B = R.map((r, i) => r.map((x, j) => ((x + R[j][i]) / 2 - (i === j ? c : 0)) / (1 - c)));
  const k = [0, 1, 2].reduce((a, i) => (B[i][i] > B[a][a] ? i : a), 0);
  let axis = unit3(column(B, k));
  if (dot3(axis, skew) < 0) axis = scale3(axis, -1);
  return { axis, angle };
}

// Clip segment pq to the cube |x|,|y|,|z| <= L (Liang-Barsky). Returns [p', q'] or null.
export function clipSegment(p, q, L) {
  const d = sub3(q, p);
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-12) { if (Math.abs(p[i]) > L) return null; continue; }
    let a = (-L - p[i]) / d[i], b = (L - p[i]) / d[i];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    if (t0 > t1) return null;
  }
  return [add3(p, scale3(d, t0)), add3(p, scale3(d, t1))];
}

export const niceStep = (E) => (E <= 8 ? 1 : E <= 16 ? 2 : E <= 40 ? 5 : E <= 80 ? 10 : E <= 200 ? 25 : 100);

// Image under M of the integer grid, clipped to the cube of half-size E, as flat point-pair lists:
// axes (images of the coordinate axes), plane (the xy-plane grid, taken from |x|,|y| <= 2E so that
// shrinking maps still fill the view) and, in 3D, cage (axis-parallel lines through lattice points
// with |coords| <= about E/2, minus the ones already in the plane grid).
export function latticeSegments(M, E, dim) {
  const step = niceStep(E), P = Math.ceil((2 * E) / step) * step;
  const out = { axes: [], plane: [], cage: [] };
  const push = (list, a, b) => {
    const s = clipSegment(mulMV(M, a), mulMV(M, b), E);
    if (s) list.push(s[0], s[1]);
  };
  for (let k = -P; k <= P; k += step) {
    const list = k === 0 ? out.axes : out.plane;
    push(list, [k, -P, 0], [k, P, 0]);
    push(list, [-P, k, 0], [P, k, 0]);
  }
  if (dim === 3) {
    push(out.axes, [0, 0, -P], [0, 0, P]);
    const C = Math.max(step, step * Math.floor(E / 2 / step));
    for (let a = -C; a <= C; a += step) {
      for (let b = -C; b <= C; b += step) {
        if (a || b) push(out.cage, [a, b, -C], [a, b, C]);
        if (b) {
          push(out.cage, [-C, a, b], [C, a, b]);
          push(out.cage, [a, -C, b], [a, C, b]);
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- transform(A[, B], t)

// The matrix at time t and the stage it is in. start/target are the stage's end matrices.
export function mapPath(A, B, t) {
  if (!B) {
    const s = clamp(t, 0, 1);
    return { M: lerpM(I3(), A, s), t: s, stage: 0, start: I3(), target: A };
  }
  const u = clamp(t, 0, 2);
  if (u <= 1) return { M: lerpM(I3(), A, u), t: u, stage: 1, start: I3(), target: A };
  const BA = mulMM(B, A);
  return { M: lerpM(A, BA, u - 1), t: u, stage: 2, start: A, target: BA };
}

const sub2 = (M) => [[M[0][0], M[0][1]], [M[1][0], M[1][1]]];

// transform(A), transform(A, t), transform(A, B), transform(A, B, t)
export function makeMap(args) {
  const [a0, a1, a2] = args;
  if (kindOf(a0) !== 'mat') throw new Error(`transform needs a matrix first, got ${describe(a0)}`);
  let B = null, t = 1;
  if (args.length === 2) {
    if (kindOf(a1) === 'mat') { B = a1; t = 2; }
    else if (kindOf(a1) === 'num') t = a1;
    else throw new Error(`transform(A, t) needs a number t, got ${describe(a1)}`);
  } else if (args.length === 3) {
    if (kindOf(a1) !== 'mat') throw new Error(`transform(A, B, t) needs a second matrix, got ${describe(a1)}`);
    if (kindOf(a2) !== 'num') throw new Error(`transform(A, B, t) needs a number t, got ${describe(a2)}`);
    B = a1;
    t = a2;
  }
  const A3 = embed3(a0.m), B3 = B ? embed3(B.m) : null;
  const dim = a0.m.length === 2 && (!B || B.m.length === 2) ? 2 : 3;
  const path = mapPath(A3, B3, t);
  const T = dim === 2 ? sub2(path.target) : path.target;
  const rank = maxAbs(T) === 0 ? 0 : la.rank(T);
  const nul = rank < dim ? (rank === 0 ? T.map((_, i) => T.map((__, j) => (i === j ? 1 : 0))) : la.nullspace(T)).map(pad3) : [];
  return {
    type: 'tf-map', A: A3, B: B3, dim, t: path.t, stage: path.stage,
    M: path.M, start: path.start, det: det3(path.M), rank, nul,
    extentPoints: [0, 1, 2].slice(0, dim).map((j) => column(path.M, j)).concat([mulMV(path.M, [1, 1, dim === 3 ? 1 : 0])]),
  };
}

const detTol = (M) => 1e-9 * Math.max(1, maxAbs(M)) ** 3;
export const flipped = (v) => v.det < -detTol(v.M);

// ---------------------------------------------------------------- carrying other rows along

const CARRIERS = new Map();
// fn(item, M, f) -> item | null, where f(v) = M v; lets another feature's item kinds ride along.
export function registerCarrier(kind, fn) { CARRIERS.set(kind, fn); }

// The item as seen after applying M to space (null drops it).
export function carryItem(it, M) {
  const f = (v) => mulMV(M, pad3(v));
  const o = f(it.o ?? [0, 0, 0]);
  const big = 1e-9 * Math.max(1, maxAbs(M));
  switch (it.kind) {
    case 'vec': case 'point': {
      const out = { ...it, o, v: f(it.v) };
      const cb = it.combo;
      // linear pictures stay exact under M; projections, boxes and trails would lie, so drop them
      if (cb?.mode === 'explain') out.combo = { ...cb, cols: cb.cols.map(f) };
      else if (cb?.mode === 'chain') out.combo = { ...cb, parts: cb.parts.map(f) };
      else if (cb?.mode === 'target') out.combo = { ...cb, b: f(cb.b) };
      else if (cb) delete out.combo;
      return out;
    }
    case 'span': return { ...it, o, vecs: it.vecs.map(f) };
    case 'parallelogram': return { ...it, o, u: f(it.u), v: f(it.v) };
    case 'parallelepiped': return { ...it, o, u: f(it.u), v: f(it.v), w: f(it.w) };
    case 'plane': {
      const [a, b] = perpPair(it.normal).map(f), n = cross3(a, b);
      if (len3(n) > big * Math.max(len3(a), len3(b), 1)) return { ...it, o, normal: n };
      const d = [a, b].sort((x, y) => len3(y) - len3(x))[0];
      return { ...it, kind: 'span', type: 'span', o, vecs: len3(d) > big ? [d] : [] };
    }
    case 'cb-line': {
      const d = f(it.d);
      if (len3(d) <= big) return { ...it, kind: 'point', type: 'point', o, v: f(it.p) };
      return { ...it, o, p: f(it.p), d, pts: it.pts?.map(f) };
    }
    case 'cb-plane': {
      const nn = dot3(it.n, it.n), p0 = scale3(it.n, it.c / nn);
      const [a, b] = perpPair(it.n).map(f), n = cross3(a, b), P = f(p0);
      if (len3(n) > big * Math.max(len3(a), len3(b), 1)) return { ...it, o, n, c: dot3(n, P), pts: it.pts?.map(f) };
      const d = [a, b].sort((x, y) => len3(y) - len3(x))[0];
      if (len3(d) <= big) return { ...it, kind: 'point', type: 'point', o, v: P };
      return { ...it, kind: 'cb-line', type: 'cb-line', o, p: P, d, pts: [] };
    }
    case 'cb-arc': {
      const u = f(it.u), v = f(it.v), L = len3(u) * len3(v);
      if (!(L > 0)) return null;
      return { ...it, o, u, v, theta: Math.acos(clamp(dot3(u, v) / L, -1, 1)) };
    }
    case 'cb-dist': {
      const p = f(it.p), foot = f(it.foot);
      return { ...it, o, p, foot, d: len3(sub3(p, foot)), along: null };
    }
  }
  // graphs (plots.js) are sampled at draw time, so they carry the matrix itself
  if (it.kind === 'graph' || it.kind === 'softmaxmap') return { ...it, o, M: it.M ? mulMM(M, it.M) : M };
  const custom = CARRIERS.get(it.kind);
  return custom ? custom(it, M, f) : it;
}

// Does the transform's path keep x on its own span? Stage 1 needs A x ∥ x; stage 2 also B x ∥ x.
export function keepsSpan(x, tf) {
  const on = (y) => len3(y) <= 1e-9 * Math.max(1, maxAbs(tf.A)) * len3(x) || len3(cross3(x, y)) <= GLOW_TOL * len3(x) * len3(y);
  const Ax = mulMV(tf.A, x);
  if (!on(Ax)) return false;
  if (!tf.B || tf.t <= 1 || len3(Ax) <= 1e-9 * Math.max(1, maxAbs(tf.A)) * len3(x)) return true;
  return on(mulMV(tf.B, x));
}

// Should this (untransformed) vector or point glow: on a drawn eigenspace and kept there by tf?
export function glows(it, spaces, tf) {
  if ((it.kind !== 'vec' && it.kind !== 'point') || !spaces.length) return false;
  const o = pad3(it.o ?? [0, 0, 0]), v = pad3(it.v);
  const x = it.kind === 'point' ? add3(o, v) : v;
  if (len3(x) < 1e-9) return false;
  const inside = (w, basis) => len3(w) < 1e-12 || residual(w, basis) <= GLOW_TOL * len3(w);
  const extra = it.kind === 'vec' && len3(o) >= 1e-12 ? [o] : [];
  if (!spaces.some((b) => inside(x, b) && extra.every((w) => inside(w, b)))) return false;
  return [x, ...extra].every((w) => keepsSpan(w, tf));
}

// Items hook: the first transform row carries every other drawable row (except fixed(...) ones and
// this feature's own overlays); eigen rows learn about the active transform for their arrows.
export function carryAll(items) {
  const tf = items.find((it) => it.kind === 'tf-map');
  if (!tf) return items;
  const spaces = items.filter((it) => it.kind === 'tf-eigen').flatMap((it) => it.lines.map((l) => l.basis));
  const out = [];
  for (const it of items) {
    if (it.kind === 'tf-eigen') { out.push({ ...it, carry: tf }); continue; }
    if (MINE.has(it.kind) || it.fixed) { out.push(it); continue; }
    const moved = carryItem(it, tf.M);
    if (!moved) continue;
    out.push(moved);
    if (glows(it, spaces, tf)) {
      out.push({ kind: 'tf-glow', o: moved.o, v: moved.v, point: it.kind === 'point', color: it.color, label: null, index: it.index, rowId: it.rowId });
    }
  }
  return out;
}

// Where the (carried) vectors and points end and put their labels, handed to our transform and
// eigen items so the det and λ labels can sit clear of them: [{p: tip, d: direction | null}].
export function labelHints(items) {
  if (!items.some((it) => it.kind === 'tf-map' || it.kind === 'tf-eigen')) return items;
  const avoid = [];
  for (const it of items) {
    if ((it.kind !== 'vec' && it.kind !== 'point') || !Array.isArray(it.v)) continue;
    const o = pad3(it.o ?? [0, 0, 0]), v = pad3(it.v);
    avoid.push({ p: add3(o, v), d: it.kind === 'vec' ? v : null });
  }
  return items.map((it) => (it.kind === 'tf-map' || it.kind === 'tf-eigen' ? { ...it, avoid } : it));
}

// ---------------------------------------------------------------- eigen(A)

export function makeEigen(m, what = 'eigen') {
  const dim = m.length;
  if (!(dim === 2 || dim === 3) || m.some((r) => r.length !== dim)) {
    throw new Error(`${what} needs a 2×2 or 3×3 matrix, got a ${dim}×${m[0].length} matrix`);
  }
  const { values: vals, vectors, planes } = la.eig(m);
  const lines = [];
  vals.forEach((val, k) => {
    if (val.im !== 0) return;
    const last = lines.at(-1);
    if (last && last.lam === val.re) { last.mult++; return; }
    lines.push({ lam: val.re, mult: 1, basis: vectors[k].map(pad3) });
  });
  const pairs = planes.map((p) => ({ re: p.value.re, im: p.value.im, p: pad3(p.basis[0]), q: pad3(p.basis[1]) }));
  const extentPoints = [...lines.flatMap((l) => l.basis.map(niceDir)), ...pairs.flatMap((p) => [p.p, p.q])];
  return { type: 'tf-eigen', A: embed3(m, what), dim, lines, pairs, extentPoints };
}

// Points of the orbit x(θ) = ρ^(θ/φ) (cos θ p + sin θ q), λ = a + bi = ρ e^(iφ): A maps x(θ) to
// x(θ + φ), so it turns p toward q while growing (ρ > 1) or shrinking (ρ < 1). The growth is capped
// at 3× over the drawn turns and the largest radius is R.
export function spiralPoints(p, q, re, im, R, turns = 1.25, n = 90) {
  const phi = Math.atan2(im, re), rho = Math.hypot(re, im), span = 2 * Math.PI * turns;
  const cap = Math.log(3) / span;
  const g = clamp(Math.log(rho) / phi, -cap, cap);
  const r0 = g > 0 ? R * Math.exp(-g * span) : R;
  const pts = [];
  for (let k = 0; k <= n; k++) {
    const th = (span * k) / n, r = r0 * Math.exp(g * th);
    pts.push(add3(scale3(p, r * Math.cos(th)), scale3(q, r * Math.sin(th))));
  }
  return pts;
}

// ---------------------------------------------------------------- svdview(A[, t])

// SVD as proper rotations: A = U·diag(sgn)·Vᵀ with det U = det V = +1 (3×3, 2×2 embedded).
// Column signs are chosen so V is the smallest rotation; a reflection in A shows up as a negative
// last entry of sgn. sig are the singular values (descending, dim of them).
export function svdFrames(m) {
  const dim = m.length;
  const { U, S, V } = la.svd(m);
  let us = [...Array(dim)].map((_, j) => pad3(column(U, j)));
  let vs = [...Array(dim)].map((_, j) => pad3(column(V, j)));
  if (dim === 2) { us.push([0, 0, 1]); vs.push([0, 0, 1]); }
  let best = null;
  for (let mask = 0; mask < 1 << dim; mask++) {
    const s = [0, 1, 2].map((j) => ((mask >> j) & 1 ? -1 : 1));
    const Vs = vs.map((v, j) => scale3(v, s[j]));
    if (det3(fromCols(Vs)) < 0) continue;
    const tr = Vs[0][0] + Vs[1][1] + Vs[2][2];
    if (!best || tr > best.tr + 1e-12) best = { s, tr };
  }
  vs = vs.map((v, j) => scale3(v, best.s[j]));
  us = us.map((u, j) => scale3(u, best.s[j]));
  const sig = S.slice(0, dim), sgn = [...sig, 1, 1].slice(0, 3);
  if (det3(fromCols(us)) < 0) {
    us[dim - 1] = scale3(us[dim - 1], -1);
    sgn[dim - 1] = -sgn[dim - 1] || 0;
  }
  const Um = fromCols(us), Vm = fromCols(vs);
  return { dim, sig, sgn, U: Um, V: Vm, ru: rotLog(Um), rv: rotLog(transpose3(Vm)) };
}

// t in [0, 3]: [0, 1] rotate by Vᵀ, [1, 2] stretch by Σ (through 0 for a reflection), [2, 3] rotate by U.
export function svdPath(F, t) {
  const u = clamp(t, 0, 3), Vt = transpose3(F.V);
  if (u <= 1) return { M: rotExp(F.rv.axis, u * F.rv.angle), stage: 1, t: u };
  if (u <= 2) return { M: mulMM(diag3(F.sgn.map((x) => 1 + (u - 1) * (x - 1))), Vt), stage: 2, t: u };
  return { M: mulMM(mulMM(rotExp(F.ru.axis, (u - 2) * F.ru.angle), diag3(F.sgn)), Vt), stage: 3, t: u };
}

export function makeSvd(args, what = 'svdview') {
  const [A, t] = args;
  if (kindOf(A) !== 'mat') throw new Error(`${what} needs a matrix, got ${describe(A)}`);
  const dim = A.m.length;
  if (!(dim === 2 || dim === 3) || A.m.some((r) => r.length !== dim)) {
    throw new Error(`${what} needs a 2×2 or 3×3 matrix, got ${describe(A)}`);
  }
  if (t != null && kindOf(t) !== 'num') throw new Error(`${what}(A, t) needs a number t, got ${describe(t)}`);
  const F = svdFrames(A.m), A3 = embed3(A.m, what);
  const path = t == null ? { M: A3, stage: 0, t: null } : svdPath(F, t);
  const vs = [0, 1, 2].slice(0, dim).map((j) => column(F.V, j));
  return {
    type: 'tf-svd', A: A3, dim, sig: F.sig, sgn: F.sgn, U: F.U, V: F.V, M: path.M, stage: path.stage, t: path.t,
    extentPoints: [...vs, ...vs.map((v) => mulMV(A3, v))],
  };
}

// ---------------------------------------------------------------- readouts

const paint = (color, s) => `\\color{${color}}{${s}}`;

// Readout lines: [stage, M (columns coloured like the arrows), det] and [orientation / rank notes].
export function mapLines(v, names = {}, theme = 'dark') {
  const A = names.A ?? 'A', B = names.B ?? 'B', n = v.dim, cols = pal(BASIS, theme);
  const body = v.M.slice(0, n).map((r) => r.slice(0, n).map((x, j) => paint(cols[j], fmt(x))).join(' & ')).join(' \\\\ ');
  const main = [], notes = [];
  if (v.B) main.push(v.stage === 1 ? `I \\to ${A}` : `${A} \\to ${B}${A}`);
  main.push(`M = \\begin{bmatrix}${body}\\end{bmatrix}`, `\\det M = ${fmt(v.det, 3)}`);
  if (flipped(v)) notes.push(paint(FLIP_TEX, '\\text{orientation flipped}'));
  if (v.rank < n) notes.push(`\\operatorname{rank}(${v.stage === 2 ? B + A : A}) = ${v.rank}\\text{: null space} \\to \\vec 0`);
  return [main.join('\\quad '), notes.join('\\quad ')].filter(Boolean);
}
export const mapLatex = (v, names) => mapLines(v, names).join('\\quad ');

const pairTex = (re, im, d) => {
  const b = fmt(im, d);
  return `${fmt(re, d) === '0' ? '' : `${fmt(re, d)} `}\\pm ${b === '1' ? '' : b}i`;
};

export function eigenLatex(v) {
  const vals = v.lines.map((l) => `${fmt(l.lam, 4)}${l.mult > 1 ? `\\ (\\times ${l.mult})` : ''}`);
  for (const p of v.pairs) vals.push(pairTex(p.re, p.im, 4));
  let s = `\\lambda = ${vals.join(',\\ ')}`;
  if (v.lines.some((l) => l.basis.length === v.dim)) s += '\\quad \\text{every vector is an eigenvector}';
  else if (v.lines.some((l) => l.basis.length < l.mult)) s += '\\quad \\text{defective: too few eigenvectors}';
  return s;
}

const SVD_STAGE = ['', 'V^{\\mathsf T}\\text{: rotate}', '\\Sigma\\text{: stretch}', 'U\\text{: rotate}'];
export function svdLatex(v) {
  const s = v.sig.map((x, i) => `\\sigma_${i + 1} = ${fmt(x, 3)}`).join(',\\ ');
  if (!v.stage) return s;
  const flip = v.stage >= 2 && v.sgn.some((x) => x < 0) ? '\\text{ (through 0: a flip)}' : '';
  return `${s}\\quad ${SVD_STAGE[v.stage]}${v.stage === 2 ? flip : ''}`;
}

export function readoutLines(v, names, theme) {
  if (v?.type === 'tf-map') return mapLines(v, names, theme);
  if (v?.type === 'tf-eigen') return [eigenLatex(v)];
  if (v?.type === 'tf-svd') return [svdLatex(v)];
  return [];
}

// Plain names of a call's arguments when the row is `[name =] fname(...)`.
export function callArgs(src, fname) {
  const body = parseLine(src)?.body;
  if (body?.t !== 'call' || body.name !== fname) return [];
  return body.args.map((a) => (a.t === 'name' ? a.name : null));
}

// Slider rows used as the t of transform/svdview -> the upper end of the range they need.
export function sliderRanges(srcs, results) {
  const out = new Map();
  const walk = (n) => {
    if (!n || typeof n !== 'object') return;
    if (n.t === 'call' && (n.name === 'transform' || n.name === 'svdview')) {
      const k = n.args.length, last = n.args.at(-1);
      const hi = n.name === 'svdview' ? (k === 2 ? 3 : 0) : k === 3 ? 2 : k === 2 ? 1 : 0;
      const j = hi && last?.t === 'name' ? results.findIndex((r) => r?.name === last.name) : -1;
      if (j >= 0 && results[j].slider != null) out.set(j, Math.max(out.get(j) ?? 0, hi));
    }
    for (const c of [n.a, n.b, ...(n.items ?? []), ...(n.args ?? []), ...(n.rows?.flat() ?? [])]) walk(c);
  };
  for (const src of srcs) walk(parseLine(src)?.body);
  return out;
}

// ---------------------------------------------------------------- language

registerType('tf-map', {
  describe: 'a transformation',
  format: (v) => `det ${fmt(v.det, 3)}`,
  latex: (v) => mapLatex(v),
  numbers: (v) => [...v.M.flat(), v.det],
});
registerType('tf-eigen', {
  describe: 'an eigen-picture',
  format: (v) => `eigenvalues ${[...v.lines.map((l) => fmt(l.lam, 4)), ...v.pairs.map((p) => `${fmt(p.re, 4)} ± ${fmt(p.im, 4)}i`)].join(', ')}`,
  latex: (v) => eigenLatex(v),
  numbers: (v) => [...v.lines.flatMap((l) => [l.lam, ...l.basis.flat()]), ...v.pairs.flatMap((p) => [p.re, p.im, ...p.p, ...p.q])],
});
registerType('tf-svd', {
  describe: 'an SVD picture',
  format: (v) => `singular values ${v.sig.map((x) => fmt(x, 3)).join(', ')}`,
  latex: (v) => svdLatex(v),
  numbers: (v) => [...v.sig, ...v.M.flat()],
});

registerFunction('transform', { n: [1, 3], f: (args) => makeMap(args) });
registerFunction('eigen', { n: 1, kind: 'mat', f: ([A], name) => makeEigen(A.m, name) });
registerFunction('svdview', { n: [1, 2], f: (args, name) => makeSvd(args, name) });
registerFunction('fixed', { n: 1, f: ([x]) => (x && typeof x === 'object' ? { ...x, fixed: true } : x) });

// ---------------------------------------------------------------- label placement

const DET_HALF = [0.16, 0.075];   // det label vs another label, in NDC (x scaled by the aspect)
const LAMBDA_HALF = [0.14, 0.075];

// How clear `spot` is of the `obstacles` on screen: >= 1 means no overlap, < 0 off screen.
export function clearance(camera, spot, obstacles, half = DET_HALF) {
  const q = spot.clone().project(camera);
  if (!(Math.abs(q.x) < 0.9 && Math.abs(q.y) < 0.9 && q.z < 1)) return -1;
  const k = camera.isPerspectiveCamera ? camera.aspect : (camera.right - camera.left) / (camera.top - camera.bottom);
  let best = Infinity;
  for (const o of obstacles) {
    const r = o.clone().project(camera);
    best = Math.min(best, Math.max((Math.abs(q.x - r.x) * k) / half[0], Math.abs(q.y - r.y) / half[1]));
  }
  return best;
}

// Index of the first clear spot (last frame's choice first, then in order of preference), else
// the clearest one. Keeping last frame's choice stops labels jumping while a slider plays.
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
const lastSpot = new Map(); // `${rowId}|${what}` -> index picked last time

// ---------------------------------------------------------------- scene (browser)

function installRenderers(registerRenderer, nameTex) {
  const tone = (c) => TONES[c.theme] ?? TONES.dark;
  // The spot a tip label of o + v uses, and the tips and label spots of `avoid` hints.
  const labelSpot = (c, tip, d) => {
    const L = d ? len3(d) : 0;
    return L > 1e-9 ? c.v3(tip).addScaledVector(c.v3(d), (0.3 * c.s) / L) : c.v3(tip).add(new c.THREE.Vector3(0, 0, 0.3 * c.s));
  };
  const hintPoints = (c, avoid = []) => avoid.flatMap(({ p, d }) => [c.v3(p), labelSpot(c, p, d)]);
  const hatSpots = (c, M, dim) => [0, 1, 2].slice(0, dim).map((j) => column(M, j)).filter((v) => len3(v) > 0.05).map((v) => labelSpot(c, v, v));
  // Pull a spot back inside the axes box so its label stays on screen.
  const inBox = (c, p) => {
    const m = Math.max(Math.abs(p.x), Math.abs(p.y), Math.abs(p.z));
    return m > 0.9 * c.E ? p.multiplyScalar((0.9 * c.E) / m) : p;
  };
  // det and λ labels placed so far in this build (one ctx per build), so later ones avoid them too
  const placed = new WeakMap();
  const place = (c, key, spots, obstacles, half) => {
    if (!placed.has(c)) placed.set(c, []);
    const mine = placed.get(c), k = pickSpot(c.camera, spots, [...obstacles, ...mine], lastSpot.get(key), half);
    lastSpot.set(key, k);
    mine.push(spots[k]);
    return spots[k];
  };
  const tipLabel = (c, o, v, latex, color, cls) => {
    const len = v.length(), tip = o.clone().add(v);
    if (len > 1e-9) tip.addScaledVector(v, (0.3 * c.s) / len);
    c.label(tip, latex, color, cls);
  };
  const rod = (c, a, b, color, r, material) => {
    const d = b.clone().sub(a), L = d.length();
    if (L < 1e-9) return;
    const m = new c.THREE.Mesh(c.GEO.cyl, material ?? c.mat('solid', color));
    c.placeAlong(m, a, d, L, r);
    c.add(m);
  };
  const mat4 = (THREE, M, o) => new THREE.Matrix4().set(
    M[0][0], M[0][1], M[0][2], o?.x ?? 0,
    M[1][0], M[1][1], M[1][2], o?.y ?? 0,
    M[2][0], M[2][1], M[2][2], o?.z ?? 0,
    0, 0, 0, 1);
  // A pulsing translucent sleeve around a vector (or a halo round a point).
  const glow = (c, o, v, point) => {
    const { THREE } = c, m = c.own(new THREE.MeshBasicMaterial({ color: tone(c).glow, transparent: true, opacity: 0.35, depthWrite: false }));
    const halo = new THREE.Mesh(c.GEO.sphere, m);
    halo.scale.setScalar((point ? 0.26 : 0.2) * c.s);
    halo.position.copy(o).add(v);
    c.add(halo);
    if (!point && v.length() > 1e-9) rod(c, o, o.clone().add(v), null, 0.1 * c.s, m);
    c.onFrame((dt, t) => { m.opacity = 0.3 + 0.14 * Math.sin(t * 4.5); });
  };
  const names = (it) => ({ A: it.names?.[0] ? nameTex(it.names[0], false) : 'A', B: it.names?.[1] ? nameTex(it.names[1], false) : 'B' });

  registerRenderer('tf-glow', (it, c) => glow(c, c.v3(it.o), c.v3(it.v), it.point));

  registerRenderer('tf-map', (it, c) => {
    const { THREE } = c, M = it.M, O = new THREE.Vector3(), light = c.theme === 'light', tn = tone(c), hues = pal(BASIS, c.theme);
    const seg = latticeSegments(M, c.E, it.dim);
    if (seg.cage.length) c.lines(seg.cage.map(c.v3), it.color, { opacity: light ? 0.2 : 0.14 });
    if (seg.plane.length) c.lines(seg.plane.map(c.v3), it.color, { opacity: light ? 0.6 : 0.5 });
    if (seg.axes.length) c.lines(seg.axes.map(c.v3), c.colors.fg, { opacity: 0.6 });

    // the unit cube (square) and its signed volume; a flipped orientation changes its colour
    const color = flipped(it) ? tn.neg : tn.pos;
    if (it.dim === 3) {
      const m4 = mat4(THREE, M);
      c.add(new THREE.Mesh(c.own(c.GEO.box.clone().applyMatrix4(m4)), c.mat('surface', color, 0.2)));
      c.add(new THREE.LineSegments(c.own(c.GEO.boxEdges.clone().applyMatrix4(m4)), c.mat('line', color, 0.95)));
    } else {
      const a = c.v3(column(M, 0)), b = c.v3(column(M, 1)), d = a.clone().add(b);
      c.add(new THREE.Mesh(c.own(new THREE.BufferGeometry().setFromPoints([O, a, d, O, d, b])), c.mat('surface', color, 0.22)));
      c.polyline([O, a, d, b, O], color, { opacity: 0.95 });
    }
    // det label: inside the parallelogram in 2D; in 3D just past the far corner, pushed away from
    // the origin as seen on screen. When a label or vector tip is in the way, the clearest of a
    // few spots just outside the other edges / corners.
    const right = new THREE.Vector3().setFromMatrixColumn(c.camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(c.camera.matrixWorld, 1);
    const push = (p, from, k) => {
      const d = p.clone().sub(from);
      let dx = d.dot(right), dy = d.dot(up);
      const dl = Math.hypot(dx, dy);
      [dx, dy] = dl > 1e-6 ? [dx / dl, dy / dl] : [0.6, 0.8];
      return p.clone().addScaledVector(right, k * dx * c.s).addScaledVector(up, 0.5 * k * dy * c.s);
    };
    const cell = (x, y, z) => c.v3(mulMV(M, [x, y, z]));
    const outer = it.dim === 2 ? [[0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0]] : [[1, 1, 0], [1, 0, 1], [0, 1, 1]];
    const mid = it.dim === 2 ? cell(0.5, 0.5, 0) : cell(0.5, 0.5, 0.5);
    const detSpots = [it.dim === 2 ? mid : push(cell(1, 1, 1), O, 1.1)];
    for (const k of [1.1, 2.4]) detSpots.push(...outer.map((p) => push(cell(...p), mid, k)));
    const obstacles = [O, ...hatSpots(c, M, it.dim), ...hintPoints(c, it.avoid)];

    // null space of the stage's target: its stage-start image (dashed) and M applied to it (shrinking)
    const nul = it.nul;
    if (nul.length && nul.length < it.dim) {
      const nm = names(it), T = it.stage === 2 ? `${nm.B}${nm.A}` : nm.A;
      const S0 = it.start, L = c.E;
      if (nul.length === 1) {
        const n = unit3(nul[0]);
        const ghost = clipSegment(mulMV(S0, scale3(n, -2 * L)), mulMV(S0, scale3(n, 2 * L)), L);
        if (ghost) c.lines(ghost.map(c.v3), tn.nul, { dashed: true, opacity: 0.9 });
        const a = c.v3(mulMV(M, scale3(n, -L))), b = c.v3(mulMV(M, scale3(n, L)));
        rod(c, a, b, tn.nul, 0.03 * c.s);
        const K = Math.min(6, Math.floor(L / len3(nul[0])));
        for (let k = -K; k <= K; k++) if (k) c.dot(c.v3(mulMV(M, scale3(nul[0], k))), tn.nul, 0.06 * c.s);
        if (ghost) {
          const end = c.v3(ghost[1]).multiplyScalar(0.85).add(new THREE.Vector3(0, 0, 0.35 * c.s));
          c.label(end, `\\mathrm{null}(${T})`, tn.nul, 'g-label tf-sm');
          obstacles.push(end);
        }
      } else {
        const e1 = unit3(nul[0]), e2 = unit3(sub3(nul[1], scale3(e1, dot3(nul[1], e1)))), r = 0.6 * L;
        const quad = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([x, y]) => add3(scale3(e1, x * r), scale3(e2, y * r)));
        const start = quad.map((p) => c.v3(mulMV(S0, p))), now = quad.map((p) => c.v3(mulMV(M, p)));
        c.polyline([...start, start[0]], tn.nul, { dashed: true, opacity: 0.9 });
        const g = c.own(new THREE.BufferGeometry().setFromPoints([now[0], now[1], now[2], now[0], now[2], now[3]]));
        c.add(new THREE.Mesh(g, c.mat('surface', tn.nul, 0.15)));
        c.polyline([...now, now[0]], tn.nul, { opacity: 0.95 });
        const spot = start[0].clone().multiplyScalar(0.9).add(new THREE.Vector3(0, 0, 0.35 * c.s));
        c.label(spot, `\\mathrm{null}(${T})`, tn.nul, 'g-label tf-sm');
        obstacles.push(spot);
      }
    }
    const detAt = place(c, `${it.rowId}|det`, detSpots.map((p) => inBox(c, p)), obstacles, DET_HALF);
    c.label(detAt, `\\det = ${fmt(it.det)}`, color, 'g-label tf-sm');

    for (let j = 0; j < it.dim; j++) {
      const v = c.v3(column(M, j));
      c.arrow(O, v, hues[j], { thickness: 1.3 });
      if (v.length() <= 0.05) continue;
      // a vector row ending on this tip (u = (1, 0) is carried onto î) keeps the tip label spot
      const taken = (it.avoid ?? []).some(({ p }) => c.v3(p).distanceTo(v) < 0.25 * c.s);
      if (!taken) { tipLabel(c, O, v, HATS[j], hues[j]); continue; }
      const side = right.clone().multiplyScalar(-v.dot(up)).addScaledVector(up, v.dot(right));
      if (side.lengthSq() < 1e-12) side.copy(up);
      c.label(v.clone().multiplyScalar(0.5).addScaledVector(side.normalize(), 0.35 * c.s), HATS[j], hues[j]);
    }
  });

  registerRenderer('tf-eigen', (it, c) => {
    const { THREE } = c, o = c.v3(it.o), L = 1.25 * c.E, tf = it.carry;
    // λ labels sit beside the line (in the plane for 2×2, above it in 3D), clear of vector tips
    const obstacles = [o, ...hintPoints(c, it.avoid), ...(tf ? hatSpots(c, tf.M, tf.dim) : [])];
    it.lines.forEach((line, idx) => {
      const k = line.basis.length;
      if (k === it.dim) return; // every vector: nothing to single out
      const dirs = line.basis.map(c.v3);
      if (k === 1) {
        const m = new THREE.Mesh(c.GEO.cyl, c.mat('ghost', it.color, 0.6));
        c.placeAlong(m, o.clone().addScaledVector(dirs[0], -L), dirs[0], 2 * L, 0.02 * c.s);
        c.add(m);
      } else {
        c.planePatch(o, dirs[0], dirs[1], it.color, null, { size: 0.6 * c.E, opacity: 0.12 });
      }
      const d = dirs[0].clone().normalize();
      let side;
      if (it.dim === 2) {
        side = new THREE.Vector3(-d.y, d.x, 0);
        if (side.y < -1e-9 || (Math.abs(side.y) <= 1e-9 && side.x < 0)) side.negate();
        side.multiplyScalar(0.45 * c.s);
      } else side = Math.abs(d.z) > 0.9 ? new THREE.Vector3(0.35 * c.s, 0, 0) : new THREE.Vector3(0, 0, 0.35 * c.s);
      const spots = [0.72, 0.55, 0.9, -0.72, -0.55].map((f) => o.clone().addScaledVector(d, f * c.E).add(side));
      const at = place(c, `${it.rowId}|lam${idx}`, spots, obstacles, LAMBDA_HALF);
      c.label(at, `\\lambda = ${fmt(line.lam, 3)}`, it.color, 'g-label tf-sm');
      for (const b of line.basis) {
        const v = niceDir(b), w = c.v3(tf ? mulMV(tf.M, v) : v);
        if (tf && keepsSpan(v, tf)) glow(c, o, w, false);
        c.arrow(o, w, it.color, { thickness: 1.1 });
      }
    });
    for (const pr of it.pairs) {
      const p = c.v3(pr.p), qh = c.v3(unit3(pr.q));
      if (it.dim === 3) c.planePatch(o, p, qh, it.color, null, { size: 0.55 * c.E, opacity: 0.12 });
      const pts = spiralPoints(pr.p, pr.q, pr.re, pr.im, 0.42 * c.E).map((x) => o.clone().add(c.v3(x)));
      const tube = c.own(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 120, 0.022 * c.s, 6, false));
      c.add(new THREE.Mesh(tube, c.mat('solid', it.color)));
      const end = pts.at(-1), dir = end.clone().sub(pts.at(-3));
      const head = new THREE.Mesh(c.GEO.cone, c.mat('solid', it.color));
      c.placeAlong(head, end, dir, 0.24 * c.s, 0.09 * c.s);
      c.add(head);
      c.label(end.clone().addScaledVector(end.clone().sub(o).normalize(), 0.55 * c.s), `\\lambda = ${pairTex(pr.re, pr.im, 3)}`, it.color, 'g-label tf-sm');
    }
  });

  registerRenderer('tf-svd', (it, c) => {
    const { THREE } = c, o = c.v3(it.o), M = it.M, fg = c.colors.fg, dim = it.dim;
    const V = [0, 1, 2].map((j) => column(it.V, j));
    const circle = (a, b, map) => {
      const pts = [];
      for (let k = 0; k <= 72; k++) {
        const th = (2 * Math.PI * k) / 72, p = add3(scale3(a, Math.cos(th)), scale3(b, Math.sin(th)));
        pts.push(o.clone().add(c.v3(map ? mulMV(map, p) : p)));
      }
      return pts;
    };
    const pairs = dim === 3 ? [[0, 1], [1, 2], [0, 2]] : [[0, 1]];
    // the unit sphere (circle) with its great circles through the vᵢ
    if (dim === 3) {
      const s = new THREE.Mesh(c.GEO.sphere, c.mat('ghost', fg, 0.07));
      s.position.copy(o);
      c.add(s);
    }
    for (const [a, b] of pairs) c.polyline(circle(V[a], V[b]), fg, { opacity: 0.4 });
    // its image under M
    const scale = Math.max(1e-12, maxAbs(M));
    if (dim === 3) {
      const solid = Math.abs(det3(M)) > 1e-6 * scale ** 3;
      const e = new THREE.Mesh(c.GEO.sphere, solid ? c.mat('glass', it.color, 0.28) : c.mat('surface', it.color, 0.22));
      e.matrixAutoUpdate = false;
      e.matrix.copy(mat4(THREE, M, o));
      c.add(e);
    } else {
      const ring = circle(V[0], V[1], M), fan = [];
      for (let k = 0; k < ring.length - 1; k++) fan.push(o, ring[k], ring[k + 1]);
      c.add(new THREE.Mesh(c.own(new THREE.BufferGeometry().setFromPoints(fan)), c.mat('surface', it.color, 0.2)));
    }
    for (const [a, b] of pairs) c.polyline(circle(V[a], V[b], M), it.color, { opacity: 0.95 });
    const done = it.t == null || it.t >= 3, hues = pal(SV, c.theme);
    const right = new THREE.Vector3().setFromMatrixColumn(c.camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(c.camera.matrixWorld, 1);
    for (let i = 0; i < dim; i++) {
      const v = c.v3(V[i]), w = c.v3(mulMV(M, V[i])), col = hues[i];
      c.arrow(o, v, col, { opacity: 0.5, thickness: 0.7, head: 0.8 });
      c.dot(o.clone().add(v), col, 0.07 * c.s);
      const wl = done && w.length() >= 1e-9 * scale;
      // σᵢuᵢ ending where vᵢ does (σᵢ ≈ 1, uᵢ ≈ vᵢ): vᵢ's label moves beside the middle of its shaft
      if (wl && w.distanceTo(v) < 0.6 * c.s) {
        const side = right.clone().multiplyScalar(-v.dot(up)).addScaledVector(up, v.dot(right));
        if (side.lengthSq() < 1e-12) side.copy(up);
        c.label(o.clone().addScaledVector(v, 0.5).addScaledVector(side.normalize(), 0.4 * c.s), `\\vec v_{${i + 1}}`, col, 'g-label tf-sm');
      } else tipLabel(c, o, v, `\\vec v_{${i + 1}}`, col, 'g-label tf-sm');
      if (w.length() < 1e-9 * scale) continue;
      c.arrow(o, w, col, { thickness: 1.25 });
      if (wl) tipLabel(c, o, w, `\\sigma_{${i + 1}}\\vec u_{${i + 1}}`, col, 'g-label tf-sm');
    }
  });
}

// ---------------------------------------------------------------- install

const CSS = `
.g-label.tf-sm { font-size: 17px; }
`;
const HELP = `<h4 class="ui-overline">Transformations</h4>
<p><code>transform(A, t)</code> space moving under (1&minus;t)I + tA, t 0&ndash;1 &middot;
<code>transform(A, B, t)</code> A then B, t 0&ndash;2 &middot; <code>fixed(u)</code> stays put while the rest moves</p>
<p><code>eigen(A)</code> eigenlines with &lambda; &middot; <code>svdview(A)</code> sphere &rarr; ellipsoid &middot;
<code>svdview(A, t)</code> rotate V&#7488;, stretch &Sigma;, rotate U (t 0&ndash;3)</p>`;
const FNAME = { 'tf-map': 'transform', 'tf-eigen': 'eigen', 'tf-svd': 'svdview' };

export async function install(api) {
  const { registerRenderer, nameTex } = await import('../scene.js');
  installRenderers(registerRenderer, nameTex);
  api.addStyles(CSS);
  document.getElementById('g-help')?.insertAdjacentHTML('beforeend', HELP);

  api.addItemsHook((items, { rows }) => labelHints(carryAll(items.map((it) => (
    FNAME[it.kind] ? { ...it, names: callArgs(rows[it.index]?.src, FNAME[it.kind]) } : it)))));

  const theme = () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  api.addRowDecorator((row, res, el) => {
    const v = res?.value;
    if (!v || res.error || !FNAME[v.type] || typeof katex === 'undefined') return;
    const [a, b] = callArgs(row.src, FNAME[v.type]);
    const names = { A: a ? nameTex(a, false) : 'A', B: b ? nameTex(b, false) : 'B' };
    el.out.innerHTML = readoutLines(v, names, theme()).map((s) => katex.renderToString(s, { throwOnError: false })).join('<br>');
  });
  // the readout's column colours follow the theme
  new MutationObserver(() => { if (api.results.some((r) => r?.value?.type === 'tf-map')) api.recompute(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  // det and λ labels are placed for the camera at build time: once a turn or zoom settles, redo them
  api.onSceneReady((scene) => {
    let key = '', since = 0, dirty = false;
    scene.onFrame(() => {
      const cam = scene.camera, now = performance.now();
      const k = [...cam.position.toArray(), ...cam.quaternion.toArray(), cam.zoom].map((x) => x.toFixed(3)).join();
      if (k !== key) { key = k; since = now; dirty = true; return; }
      if (!dirty || now - since < 250) return;
      dirty = false;
      if (scene.items.some((it) => it.kind === 'tf-map' || it.kind === 'tf-eigen')) scene.rebuild();
    });
  });

  // a slider that drives t starts out ranging over t's domain instead of the default -5..5
  api.onRecompute((results, rows) => {
    let changed = false;
    for (const [j, hi] of sliderRanges(rows.map((r) => r.src), results)) {
      const r = rows[j];
      if (r && r.min == null && r.max == null) { r.min = 0; r.max = hi; changed = true; }
    }
    if (changed) queueMicrotask(() => api.recompute());
  });
}
