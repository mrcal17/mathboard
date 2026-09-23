// Dynamics and quadratic forms for the 3D tab.
//   flow(A)           x' = Ax: a sparse direction field, particles streaming along the flow (RK4)
//                     with short trails, real eigenvector lines as highways, a sample orbit for
//                     complex eigenvalues. A 2×2 A flows in the xy-plane.
//   iterate(A, v, n)  v, Av, …, Aⁿv as fading arrows with the tips joined; value Aⁿv
//   power(A, v, n)    the same normalised each step (arrows keep |v|) plus the dominant eigenvector
//                     line; value |v|·qₙ, readout qₙ and the Rayleigh quotient
//   quadric(A[, c])   xᵀAx = c (default 1) using (A + Aᵀ)/2, classified by eigenvalue signs, drawn in
//                     the eigenbasis and clipped to the axes box, principal axes labelled (2×2: a conic)
// n can be a slider; a fractional n animates the next step. Pure helpers are exported for tests;
// scene.js loads in install().
import { registerFunction, registerType, values, formatNumber, parseLine } from '../lang.js';
import * as la from '../linalg.js';

const { vec, kindOf, describe } = values;

export const MAX_ITERATE = 100;
export const MAX_POWER = 500;
export const SHOWN = 40;                          // arrows drawn per sequence (older ones are left out)
export const PARTICLES = { 2: 450, 3: 700 };
export const TRAIL = { segs: 10, seconds: 0.45 }; // trail = the particle's last 0.45 s of motion
const FNS = new Set(['flow', 'iterate', 'power', 'quadric']);
const TYPES = new Set(['fl-flow', 'fl-quadric']);

const pad3 = (x) => [x[0], x[1], x[2] ?? 0];
const matvec = (A, x) => A.map((r) => r.reduce((s, a, j) => s + a * x[j], 0));
const dotN = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (x) => Math.sqrt(dotN(x, x));
const maxAbs = (A) => A.flat().reduce((m, x) => Math.max(m, Math.abs(x)), 0);
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
const mmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
const snap = (x) => (Math.abs(x - Math.round(x)) < 1e-10 ? Math.round(x) || 0 : x);
const n3 = (x) => formatNumber(Math.round(x * 1000) / 1000);
const quant = (a) => Math.round(a * 20) / 20;     // opacities share cached materials
const sgn = (x, tol) => (Math.abs(x) <= tol ? 0 : Math.sign(x));

// ---------------------------------------------------------------- colours

const rgbOf = (h) => {
  const n = typeof h === 'number' ? h : parseInt(String(h).replace('#', '').slice(0, 6), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
export function mixHex(a, b, t) {
  const pa = rgbOf(a), pb = rgbOf(b);
  return '#' + pa.map((x, i) => Math.round(x + (pb[i] - x) * t).toString(16).padStart(2, '0')).join('');
}

function hexToHsl(hex) {
  const [r, g, b] = rgbOf(hex).map((x) => x / 255);
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
  const lin = (x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
  const [r, g, b] = rgbOf(hex).map(lin);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// n well-separated hues starting at the row colour, readable on the theme: yellows and greens are
// darkened on the whiteboard, blues lightened on the chalkboard (contrast >= ~3.8).
export function tones(hex, n, theme = 'dark') {
  const [h, s] = hexToHsl(hex), light = theme === 'light';
  return Array.from({ length: n }, (_, i) => {
    let L = light ? 0.42 : 0.64, out = hslToHex(h + (i * 360) / n, Math.max(0.55, s), L);
    for (let k = 0; k < 30 && (light ? luminance(out) > 0.2 : luminance(out) < 0.2); k++) {
      L += light ? -0.02 : 0.02;
      out = hslToHex(h + (i * 360) / n, Math.max(0.55, s), L);
    }
    return out;
  });
}

// ---------------------------------------------------------------- flow

// One RK4 step of x' = Ax with step h is exactly x ← P x, P = I + hA + (hA)²/2 + (hA)³/6 + (hA)⁴/24.
export function rk4Matrix(A, h) {
  const n = A.length, hA = A.map((r) => r.map((x) => x * h));
  let P = eye(n), term = eye(n);
  for (let k = 1; k <= 4; k++) {
    term = mmul(term, hA).map((r) => r.map((x) => x / k));
    P = P.map((r, i) => r.map((x, j) => x + term[i][j]));
  }
  return P;
}

function flowKind(d, A, groups, pairs, st, un, ne, tol) {
  if (maxAbs(A) === 0) return 'no motion (A = 0)';
  const center = pairs.length && sgn(pairs[0].re, tol) === 0;
  if (d === 2) {
    if (pairs.length) return center ? 'center' : pairs[0].re < 0 ? 'stable spiral' : 'unstable spiral';
    if (st && un) return 'saddle';
    if (ne === 2) return 'shear flow';
    if (ne) return st ? 'line of fixed points, attracting' : 'line of fixed points, repelling';
    const w = st ? 'stable' : 'unstable';
    if (groups.length === 1) return groups[0].basis.length === 2 ? `${w} star` : `${w} degenerate node`;
    return `${w} node`;
  }
  if (!ne) {
    const base = st === 3 ? 'sink' : un === 3 ? 'source' : 'saddle';
    return pairs.length ? `spiral ${base}` : base;
  }
  if (center) return 'center (closed orbits)';
  return ne === 3 ? 'shear flow' : 'non-isolated fixed points';
}

// Eigen-structure of x' = Ax for 2×2 / 3×3 A: real eigenvalue groups with their eigenspace
// bases, complex pairs with their invariant planes, a phase-portrait name, and `rate`: the
// time scale that keeps the animation between 0.4 and 1.5 "speed units" (1 = real time).
export function flowInfo(A) {
  const d = A.length, { values: ev, vectors, planes } = la.eig(A);
  const tol = 1e-9 * (maxAbs(A) || 1);
  const groups = [];
  ev.forEach((z, i) => {
    if (z.im !== 0) return;
    const g = groups.at(-1);
    if (g && g.lam === z.re) g.mult++;
    else groups.push({ lam: z.re, mult: 1, basis: vectors[i].map(pad3) });
  });
  const pairs = planes.map(({ value, basis }) => ({ re: value.re, im: value.im, p: pad3(basis[0]), q: pad3(basis[1]) }));
  const s = ev.map((z) => sgn(z.re, tol));
  const st = s.filter((x) => x < 0).length, un = s.filter((x) => x > 0).length, ne = d - st - un;
  const nu = Math.hypot(...A.flat()) / Math.sqrt(d);
  const rate = nu > 0 ? Math.min(Math.max(nu, 0.4), 1.5) / nu : 1;
  const detail = [st && `${st} stable`, un && `${un} unstable`, ne && `${ne} neutral`].filter(Boolean).join(', ');
  return {
    dim: d, A: A.map((r) => r.slice()), values: ev, groups, pairs, stable: st, unstable: un, neutral: ne,
    rate, kind: flowKind(d, A, groups, pairs, st, un, ne, tol), detail,
  };
}

// Arrow glyphs of the field Ax on a grid over [-E, E]^dim (relative to the flow's origin).
// Length runs from 0.25 to 0.8 of the spacing (0.2 to 0.55 in 3D) and alpha from 0.3 to 1, both
// on a square-root scale of |Ax| / max|Ax|, so the slow middle stays visible and the edge tidy.
export function fieldGlyphs(A, dim, E) {
  const ticks = dim === 2 ? Array.from({ length: 11 }, (_, i) => -E + (i * E) / 5) : [-0.8, -0.4, 0, 0.4, 0.8].map((t) => t * E);
  const [l0, l1] = dim === 2 ? [0.25, 0.8] : [0.2, 0.55];
  const spacing = ticks[1] - ticks[0], raw = [];
  for (const x of ticks) for (const y of ticks) for (const z of dim === 3 ? ticks : [0]) {
    const p = [x, y, z], f = pad3(matvec(A, p.slice(0, dim)));
    raw.push({ p, f, m: norm(f) });
  }
  const M = Math.max(...raw.map((g) => g.m));
  const glyphs = [];
  if (M > 0) {
    for (const { p, f, m } of raw) {
      const r = m / M;
      if (r < 1e-6) continue;
      const k = Math.sqrt(r);
      glyphs.push({ p, dir: f.map((x) => x / m), len: spacing * (l0 + (l1 - l0) * k), alpha: 0.3 + 0.7 * k });
    }
  }
  return { spacing, glyphs };
}

// Sample orbits for a complex pair: a spiral from the edge in (sink) or from the middle out
// (source); two closed orbits for a center.
export function sampleOrbits(A, pair, E) {
  const d = A.length, tol = 1e-9 * (maxAbs(A) || 1), re = sgn(pair.re, tol);
  const P = rk4Matrix(A, (2 * Math.PI) / pair.im / 90), p = pair.p.slice(0, d);
  const run = (r0, steps) => {
    let x = p.map((c) => c * r0);
    const pts = [pad3(x)];
    for (let k = 0; k < steps; k++) {
      x = matvec(P, x);
      pts.push(pad3(x));
      if (x.some((c) => Math.abs(c) > E) || norm(x) < 0.03 * E) break;
    }
    return pts;
  };
  if (re === 0) return [run(0.45 * E, 90), run(0.9 * E, 90)];
  return [re < 0 ? run(0.9 * E, 720) : run(0.04 * E, 720)];
}

// Particle swarm in coordinates relative to the flow's origin (z = 0 in 2D).
export function makeSwarm(dim, E, n = PARTICLES[dim], rand = Math.random) {
  const sw = { dim, E, n, pos: new Float32Array(3 * n), age: new Float32Array(n), life: new Float32Array(n), alpha: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    spawn(sw, i, rand);
    sw.age[i] = rand() * sw.life[i]; // staggered, so they don't all respawn together
  }
  return sw;
}

function spawn(sw, i, rand) {
  for (let j = 0; j < 3; j++) sw.pos[3 * i + j] = j < sw.dim ? (2 * rand() - 1) * sw.E : 0;
  sw.age[i] = 0;
  sw.life[i] = 3 + 3 * rand();
}

const unpack3 = (P) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => P[i]?.[j] ?? 0));

// Advance every particle by x ← P x (P = rk4Matrix(A, h)); respawn those that leave the box, age
// out, or reach the origin (a sink would otherwise pile them into one bright blob).
export function stepSwarm(sw, P, dt, rand = Math.random) {
  const { pos, age, life, alpha, n, E } = sw, B = 1.03 * E, core2 = (0.02 * E) ** 2;
  const [[a, b, c], [d, e, f], [g, h, k]] = unpack3(P);
  for (let i = 0; i < n; i++) {
    const j = 3 * i, x = pos[j], y = pos[j + 1], z = pos[j + 2];
    const nx = a * x + b * y + c * z, ny = d * x + e * y + f * z, nz = g * x + h * y + k * z;
    pos[j] = nx; pos[j + 1] = ny; pos[j + 2] = nz;
    age[i] += dt;
    if (Math.abs(nx) > B || Math.abs(ny) > B || Math.abs(nz) > B || age[i] > life[i] || nx * nx + ny * ny + nz * nz < core2) {
      spawn(sw, i, rand);
    }
    alpha[i] = Math.max(0, Math.min(1, age[i] / 0.4, (life[i] - age[i]) / 0.6));
  }
}

// Trails are each particle's exact recent past: `segs` steps of Bk = rk4Matrix(A, -Δ) backwards,
// written as line-segment pairs; alpha (every 4th float of `col`) fades toward the tail.
export function fillTrails(sw, Bk, segs, out, col) {
  const { pos, alpha, n, E } = sw, lim = 1.15 * E;
  const [[a, b, c], [d, e, f], [g, h, k]] = unpack3(Bk);
  for (let i = 0; i < n; i++) {
    let x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2], v = i * segs * 2;
    const a0 = 0.7 * alpha[i];
    for (let s = 0; s < segs; s++, v += 2) {
      let px = a * x + b * y + c * z, py = d * x + e * y + f * z, pz = g * x + h * y + k * z;
      if (Math.abs(px) > lim || Math.abs(py) > lim || Math.abs(pz) > lim) { px = x; py = y; pz = z; }
      out[3 * v] = x; out[3 * v + 1] = y; out[3 * v + 2] = z;
      out[3 * v + 3] = px; out[3 * v + 4] = py; out[3 * v + 5] = pz;
      col[4 * v + 3] = a0 * (1 - s / segs);
      col[4 * v + 7] = a0 * (1 - (s + 1) / segs);
      x = px; y = py; z = pz;
    }
  }
}

// ---------------------------------------------------------------- iterate / power

export function iterateInfo(A, v, n, frac = 0) {
  const seq = [v.slice()];
  let x = v.slice();
  for (let k = 1; k <= n + (frac > 0 ? 1 : 0); k++) {
    x = matvec(A, x);
    if (!x.every(Number.isFinite)) throw new Error(`A^${k} v is too large; try a smaller n`);
    seq.push(x);
  }
  const next = frac > 0 ? seq.pop() : null;
  return { mode: 'iterate', dim: A.length, n, frac, seq: seq.map(pad3), next: next && pad3(next) };
}

// The eigenvalue of largest modulus and whether power iteration can settle on one line.
// `note` is KaTeX for the readout.
export function dominant(A) {
  const { values: ev, vectors } = la.eig(A);
  const mods = ev.map((z) => Math.hypot(z.re, z.im)), top = Math.max(...mods);
  if (top === 0) return { kind: 'zero', note: '\\text{all eigenvalues are 0}' };
  const tol = 1e-9 * top, lead = ev.map((_, i) => i).filter((i) => mods[i] >= top - tol);
  if (lead.some((i) => ev[i].im !== 0)) {
    return { kind: 'complex', note: '\\text{dominant eigenvalues are complex: the iterates keep turning}' };
  }
  if (new Set(lead.map((i) => ev[i].re)).size > 1) {
    return { kind: 'tie', note: `\\lambda = \\pm ${n3(top)}\\text{: no single dominant direction}` };
  }
  const i = lead[0], lam = ev[i].re, basis = vectors[i].map(pad3);
  const rest = mods.filter((_, j) => !lead.includes(j)), ratio = rest.length ? Math.max(...rest) / top : 0;
  if (basis.length > 1) return { kind: 'plane', lam, basis, note: '\\text{the dominant eigenspace is a plane}' };
  if (lead.length > 1) return { kind: 'line', lam, dir: basis[0], note: '\\text{repeated } \\lambda_1 \\text{ with one eigenvector: slow convergence}' };
  return { kind: 'line', lam, dir: basis[0], ratio };
}

// q_{k+1} = A q_k / |A q_k| from q_0 = v/|v|. Arrows are drawn at |v| (so step 0 is v itself).
export function powerInfo(A, v, n, frac = 0) {
  const R = norm(v);
  if (R === 0) throw new Error('power needs a nonzero starting vector');
  const scale = maxAbs(A) || 1, qs = [v.map((x) => x / R)];
  let dead = 0;
  for (let k = 1; k <= n + (frac > 0 ? 1 : 0); k++) {
    const y = matvec(A, qs.at(-1)), ny = norm(y);
    if (!(ny > 1e-13 * scale)) { dead = k; break; }
    qs.push(y.map((x) => x / ny));
  }
  const m = Math.min(n, qs.length - 1), q = qs[m];
  const next = frac > 0 && qs[m + 1] ? pad3(qs[m + 1].map((x) => x * R)) : null;
  return {
    mode: 'power', dim: A.length, n: m, frac: next ? frac : 0, R, dead: dead && dead <= n ? dead : 0,
    seq: qs.slice(0, m + 1).map((x) => pad3(x.map((c) => c * R))), next,
    q: pad3(q), rayleigh: dotN(q, matvec(A, q)), dom: dominant(A),
  };
}

// ---------------------------------------------------------------- quadric

// Symmetric eigen-decomposition by cyclic Jacobi: values descending, orthonormal vectors.
export function symEig(S) {
  const n = S.length, scale = maxAbs(S) || 1;
  let A = S.map((r) => r.slice()), V = eye(n);
  const off = () => A.reduce((s, r, i) => s + r.reduce((t, x, j) => t + (j > i ? x * x : 0), 0), 0);
  for (let sweep = 0; sweep < 60 && off() > 1e-30 * scale * scale; sweep++) {
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (A[p][q] === 0) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c, J = eye(n);
        J[p][p] = c; J[q][q] = c; J[p][q] = s; J[q][p] = -s;
        A = mmul(mmul(J[0].map((_, i) => J.map((r) => r[i])), A), J);
        V = mmul(V, J);
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => A[j][j] - A[i][i]);
  const canon = (v) => {
    const k = v.findIndex((x) => Math.abs(x) > 1e-9);
    return v.map((x) => snap(k >= 0 && v[k] < 0 ? -x : x));
  };
  return { values: order.map((i) => snap(A[i][i])), vectors: order.map((i) => canon(V.map((r) => r[i]))) };
}

const NAMES = {
  ellipsoid: 'ellipsoid', hyperboloid1: 'hyperboloid of one sheet', hyperboloid2: 'hyperboloid of two sheets',
  ellcyl: 'elliptic cylinder', hypcyl: 'hyperbolic cylinder', planes2: 'two parallel planes',
  cone: 'elliptic cone', line: 'a line (counted twice)', xplanes: 'two intersecting planes', plane: 'a plane (counted twice)',
  ellipse: 'ellipse', hyperbola: 'hyperbola', lines2: 'two parallel lines', xlines: 'two crossing lines',
  point: 'just the origin', empty: 'no real points',
};

// Kind by the sign pattern of k = λ/c (c ≠ 0) or of λ (c = 0, flipped so + is the majority), plus
// the eigen-index order the parametrisation wants: [+, +, −] etc.
const PATTERNS = {
  3: [
    ['+++', 'ellipsoid'], ['++-', 'hyperboloid1'], ['--+', 'hyperboloid2'], ['++0', 'ellcyl'], ['+-0', 'hypcyl'], ['+00', 'planes2'],
  ],
  30: [['+++', 'point'], ['++-', 'cone'], ['++0', 'line'], ['+-0', 'xplanes'], ['+00', 'plane']],
  2: [['++', 'ellipse'], ['+-', 'hyperbola'], ['+0', 'lines2']],
  20: [['++', 'point'], ['+-', 'xlines'], ['+0', 'line']],
};

export function quadricInfo(M, c = 1) {
  const d = M.length, scale = maxAbs(M) || 1;
  const S = M.map((r, i) => r.map((x, j) => (x + M[j][i]) / 2));
  const symmetric = M.every((r, i) => r.every((x, j) => Math.abs(x - M[j][i]) <= 1e-12 * scale));
  const { values, vectors } = symEig(S);
  const tol = 1e-9 * (Math.max(...values.map(Math.abs)) || 1);
  const lams = values.map((l) => (Math.abs(l) <= tol ? 0 : l));
  let ks = c === 0 ? lams.slice() : lams.map((l) => l / c);
  const count = (s) => ks.filter((k) => Math.sign(k) === s).length;
  if (c === 0 && count(-1) > count(1)) ks = ks.map((k) => -k);
  const sg = (k) => (k > 0 ? '+' : k < 0 ? '-' : '0');
  let kind = c === 0 && lams.every((l) => l === 0) ? 'all' : 'empty', order = [...Array(d).keys()];
  for (const [pat, name] of PATTERNS[c === 0 ? d * 10 : d]) {
    const used = new Set(), pick = [...pat].map((ch) => {
      const i = ks.findIndex((k, j) => !used.has(j) && sg(k) === ch);
      used.add(i);
      return i;
    });
    if (pick.every((i) => i >= 0)) { kind = name; order = pick; break; }
  }
  const a = order.map((i) => (ks[i] ? 1 / Math.sqrt(Math.abs(ks[i])) : null));
  const same = (x, y) => Math.abs(x - y) <= 1e-9 * Math.max(x, y);
  let name = kind === 'all' ? (d === 3 ? 'every point of space' : 'every point of the plane') : NAMES[kind];
  if (kind === 'ellipsoid') name = same(a[0], a[1]) && same(a[1], a[2]) ? 'sphere' : same(a[0], a[1]) || same(a[1], a[2]) || same(a[0], a[2]) ? 'spheroid (ellipsoid of revolution)' : name;
  if (kind === 'ellcyl' && same(a[0], a[1])) name = 'circular cylinder';
  if (kind === 'cone' && same(ks[order[0]], ks[order[1]])) name = 'circular cone';
  if (kind === 'ellipse' && same(a[0], a[1])) name = 'circle';
  // Principal axes: eigenvectors scaled by √|c/λ| (1/√|λ| when c = 0); `real` = the surface meets it.
  const axes = lams.map((l, i) => ({
    lam: l, dir: pad3(vectors[i]),
    len: l === 0 ? null : c === 0 ? 1 / Math.sqrt(Math.abs(l)) : Math.sqrt(Math.abs(c / l)),
    real: c === 0 || l / c > 0,
  }));
  return { dim: d, S, c, symmetric, lams, vecs: vectors.map(pad3), ks, kind, name, order, axes };
}

// Parametric grids of the surface (3×3), points relative to the origin, covering the ball of
// radius R: [{cols, rows, pts: (rows+1)(cols+1) points, wu, wv: iso-lines to draw}].
export function quadricPatches(info, R) {
  if (info.dim !== 3) return [];
  const { kind, ks, order, vecs } = info, [q1, q2, q3] = order.map((i) => vecs[i]);
  const [a1, a2, a3] = order.map((i) => (ks[i] ? 1 / Math.sqrt(Math.abs(ks[i])) : 0));
  const X = ([y1, y2, y3]) => [0, 1, 2].map((j) => y1 * q1[j] + y2 * q2[j] + y3 * q3[j]);
  const grid = (cols, rows, f, wu = 8, wv = 8) => {
    const pts = [];
    for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) pts.push(X(f(c / cols, r / rows)));
    return { cols, rows, pts, wu, wv };
  };
  const TAU = 2 * Math.PI, span = (u) => R * (2 * u - 1);
  switch (kind) {
    case 'ellipsoid':
      return [grid(64, 32, (u, v) => {
        const th = TAU * u, ph = Math.PI * v;
        return [a1 * Math.sin(ph) * Math.cos(th), a2 * Math.sin(ph) * Math.sin(th), a3 * Math.cos(ph)];
      })];
    case 'hyperboloid1': {
      const T = Math.asinh(R / a3);
      return [grid(64, 40, (u, v) => {
        const th = TAU * u, t = T * (2 * v - 1);
        return [a1 * Math.cosh(t) * Math.cos(th), a2 * Math.cosh(t) * Math.sin(th), a3 * Math.sinh(t)];
      }, 8, 10)];
    }
    case 'hyperboloid2': {
      const T = Math.asinh(R / Math.min(a1, a2));
      return [1, -1].map((s) => grid(64, 24, (u, v) => {
        const th = TAU * u, t = T * v;
        return [a1 * Math.sinh(t) * Math.cos(th), a2 * Math.sinh(t) * Math.sin(th), s * a3 * Math.cosh(t)];
      }, 8, 6));
    }
    case 'ellcyl':
      return [grid(64, 8, (u, v) => [a1 * Math.cos(TAU * u), a2 * Math.sin(TAU * u), span(v)], 8, 8)];
    case 'hypcyl': {
      const T = Math.asinh(R / a2);
      return [1, -1].map((s) => grid(48, 8, (u, v) => {
        const t = T * (2 * u - 1);
        return [s * a1 * Math.cosh(t), a2 * Math.sinh(t), span(v)];
      }, 6, 8));
    }
    case 'planes2':
      return [1, -1].map((s) => grid(6, 6, (u, v) => [s * a1, span(u), span(v)], 6, 6));
    case 'plane':
      return [grid(6, 6, (u, v) => [0, span(u), span(v)], 6, 6)];
    case 'cone': {
      const b1 = a1 / a3, b2 = a2 / a3; // √(|k3|/k1), √(|k3|/k2)
      return [grid(64, 32, (u, v) => {
        const th = TAU * u, h = span(v);
        return [h * b1 * Math.cos(th), h * b2 * Math.sin(th), h];
      }, 8, 8)];
    }
    case 'xplanes': {
      const r = a1 / a2, w = Math.sqrt(1 + r * r); // y1 = ±r y2 with r = √(|k2|/k1)
      return [1, -1].map((s) => grid(6, 6, (u, v) => [(s * r * span(u)) / w, span(u) / w, span(v)], 6, 6));
    }
  }
  return [];
}

// Curves of a conic (2×2), relative to the origin, before clipping.
export function conicCurves(info, R) {
  if (info.dim !== 2) return [];
  const { kind, ks, order, vecs } = info, [q1, q2] = order.map((i) => vecs[i]);
  const [a1, a2] = order.map((i) => (ks[i] ? 1 / Math.sqrt(Math.abs(ks[i])) : 0));
  const X = (y1, y2) => [y1 * q1[0] + y2 * q2[0], y1 * q1[1] + y2 * q2[1], 0];
  const line = (f) => [f(-R), f(R)];
  switch (kind) {
    case 'ellipse':
      return [Array.from({ length: 161 }, (_, k) => X(a1 * Math.cos((k * Math.PI) / 80), a2 * Math.sin((k * Math.PI) / 80)))];
    case 'hyperbola': {
      const T = Math.asinh(R / a2);
      return [1, -1].map((s) => Array.from({ length: 121 }, (_, k) => {
        const t = T * (k / 60 - 1);
        return X(s * a1 * Math.cosh(t), a2 * Math.sinh(t));
      }));
    }
    case 'lines2': return [1, -1].map((s) => line((t) => X(s * a1, t)));
    case 'line': return [line((t) => X(0, t))];
    case 'xlines': {
      const r = a1 / a2, w = Math.sqrt(1 + r * r);
      return [1, -1].map((s) => line((t) => X((s * r * t) / w, t / w)));
    }
  }
  return [];
}

// Split a polyline into the runs inside the box |x_i| ≤ B (i < dims), cutting at the walls.
export function clipPolyline(pts, B, dims = 2) {
  const runs = [];
  let cur = null;
  for (let i = 0; i + 1 < pts.length; i++) {
    const a = pts[i], b = pts[i + 1];
    let t0 = 0, t1 = 1;
    for (let j = 0; j < dims && t0 <= t1; j++) {
      const dj = b[j] - a[j];
      if (Math.abs(dj) < 1e-15) { if (Math.abs(a[j]) > B) t1 = -1; continue; }
      let u = (-B - a[j]) / dj, w = (B - a[j]) / dj;
      if (u > w) [u, w] = [w, u];
      t0 = Math.max(t0, u);
      t1 = Math.min(t1, w);
    }
    if (t0 >= t1) { cur = null; continue; }
    const at = (t) => a.map((x, j) => x + (b[j] - x) * t);
    if (!cur || t0 > 0) { cur = [at(t0)]; runs.push(cur); }
    cur.push(at(t1));
    if (t1 < 1) cur = null;
  }
  return runs;
}

// ---------------------------------------------------------------- readouts

function nm(name, vector = false) {
  const m = name.match(/^(.*?)(?:_(\w+)|(\d+))$/);
  let base = m && m[1] ? m[1] : name;
  const sub = m && m[1] ? m[2] || m[3] : '';
  if (base.length > 1) base = `\\mathrm{${base}}`;
  const b = vector ? `\\vec{${base}}` : base;
  return sub ? `${b}_{${sub}}` : b;
}

const tupleTex = (v, dim = 3) => `(${v.slice(0, dim).map(n3).join(', ')})`;
const powTex = (A, k) => (k === 0 ? '' : k === 1 ? A : `${A}^{${k}}`);

export function eigTex(ev) {
  const out = [];
  for (const z of ev) {
    if (z.im < 0) continue;
    if (z.im === 0) { out.push(n3(z.re)); continue; }
    const im = n3(z.im), it = im === '1' ? 'i' : `${im}i`;
    out.push(n3(z.re) === '0' ? `\\pm ${it}` : `${n3(z.re)} \\pm ${it}`);
  }
  return out.join(',\\ ');
}

// Slider rows used as the n of iterate/power -> the upper end of the range they start with.
export function stepRanges(srcs, results) {
  const out = new Map();
  for (const src of srcs) {
    const body = parseLine(src ?? '')?.body;
    if (body?.t !== 'call' || (body.name !== 'iterate' && body.name !== 'power') || body.args[2]?.t !== 'name') continue;
    const j = results.findIndex((r) => r?.name === body.args[2].name);
    if (j >= 0 && results[j].slider != null) out.set(j, Math.max(out.get(j) ?? 0, body.name === 'power' ? 20 : 8));
  }
  return out;
}

// Names of the arguments when a row is a direct call to one of our functions.
export function callArgs(src) {
  const body = parseLine(src ?? '')?.body;
  if (body?.t !== 'call' || !FNS.has(body.name)) return [];
  return body.args.map((a) => (a.t === 'name' ? a.name : null));
}

// KaTeX lines for the row readout, or null for values that aren't ours.
export function readoutLines(v, { args = [], theme = 'dark', color = '#e05a4f' } = {}) {
  if (!v || typeof v !== 'object') return null;
  const A = args[0] ? nm(args[0]) : 'A';
  if (v.type === 'fl-flow') {
    const f = v.info, lines = [`\\text{${f.kind}}${f.dim === 3 && f.detail ? `\\ \\ (\\text{${f.detail}})` : ''}`];
    const speed = f.rate !== 1 ? `\\quad \\text{shown at } ${formatNumber(Math.round(f.rate * 100) / 100)}\\times \\text{ speed}` : '';
    lines.push(`\\lambda = ${eigTex(f.values)}${speed}`);
    return lines;
  }
  if (v.type === 'fl-quadric') {
    const q = v.info, tn = tones(color, q.dim, theme);
    const lams = q.lams.map((l, i) => `\\color{${tn[i]}}{${n3(l)}}`).join(',\\ ');
    const lines = [`\\vec x^{\\mathsf T}\\!${A}\\,\\vec x = ${n3(q.c)}:\\ \\text{${q.name}}`, `\\lambda = ${lams}`];
    if (!q.symmetric) lines.push(`\\text{not symmetric: using } \\tfrac12(${A} + ${A}^{\\mathsf T})`);
    return lines;
  }
  const f = v.fl;
  if (f?.mode === 'iterate') {
    const x = nm(args[1] ?? 'v', true);
    return [`${powTex(A, f.n)}${x} = ${tupleTex(v.v, f.dim)}`];
  }
  if (f?.mode === 'power') {
    const k = f.n, qk = `\\vec q_{${k}}`, dom = f.dom;
    const lines = [`${qk} = ${tupleTex(f.q, f.dim)}`];
    const lam1 = dom.lam != null ? `\\quad \\lambda_1 = ${n3(dom.lam)}` : '';
    lines.push(`${qk}^{\\mathsf T}${A}\\,${qk} = ${formatNumber(f.rayleigh)}${lam1}`);
    if (f.dead) lines.push(`${powTex(A, f.dead)}\\vec v = \\vec 0 \\text{: stopped}`);
    else if (dom.note) lines.push(dom.note);
    else if (dom.ratio != null) lines.push(`|\\lambda_2 / \\lambda_1| = ${n3(dom.ratio)}${dom.lam < 0 ? '\\quad\\text{(sign flips each step)}' : ''}`);
    return lines;
  }
  return null;
}

// ---------------------------------------------------------------- language

function squareMat(A, name) {
  if (kindOf(A) !== 'mat') throw new Error(`${name} needs a matrix first, got ${describe(A)}`);
  const n = A.m.length;
  if (n !== A.m[0].length || n < 2 || n > 3) throw new Error(`${name} needs a 2×2 or 3×3 matrix, got ${describe(A)}`);
  return A.m.map((r) => r.slice());
}

function vecArg(v, d, name) {
  if (kindOf(v) !== 'vec') throw new Error(`${name} needs a vector after the matrix, got ${describe(v)}`);
  if (d === 2 && v.v[2] !== 0) throw new Error("a 2×2 matrix can't act on a vector with nonzero z");
  return v.v.slice(0, d);
}

function stepsArg(n, name, max) {
  if (kindOf(n) !== 'num') throw new Error(`${name} needs a step count n last, got ${describe(n)}`);
  const x = Math.min(Math.max(n, 0), max), k = Math.floor(x);
  return { k, frac: x - k };
}

registerType('fl-flow', {
  describe: 'a flow',
  format: (v) => `flow: ${v.info.kind}`,
  latex: (v) => `\\text{${v.info.kind}}`,
  numbers: (v) => v.A.flat(),
});
registerFunction('flow', {
  n: 1,
  f: ([A], name) => {
    const M = squareMat(A, name);
    return { type: 'fl-flow', A: M, info: flowInfo(M) };
  },
});

registerFunction('iterate', {
  n: 3,
  f: ([A, v, n], name) => {
    const M = squareMat(A, name), { k, frac } = stepsArg(n, name, MAX_ITERATE);
    const fl = iterateInfo(M, vecArg(v, M.length, name), k, frac);
    return { ...vec(fl.seq[fl.n]), fl };
  },
});

registerFunction('power', {
  n: 3,
  f: ([A, v, n], name) => {
    const M = squareMat(A, name), { k, frac } = stepsArg(n, name, MAX_POWER);
    const fl = powerInfo(M, vecArg(v, M.length, name), k, frac);
    return { ...vec(fl.seq[fl.n]), fl };
  },
});

registerType('fl-quadric', {
  describe: 'a quadric',
  format: (v) => v.info.name,
  latex: (v) => `\\text{${v.info.name}}`,
  numbers: (v) => [...v.info.S.flat(), v.info.c],
});
registerFunction('quadric', {
  n: [1, 2],
  f: ([A, c = 1], name) => {
    const M = squareMat(A, name);
    if (kindOf(c) !== 'num') throw new Error(`${name} needs a number c for xᵀAx = c, got ${describe(c)}`);
    const info = quadricInfo(M, c);
    const extentPoints = info.axes.filter((a) => a.len != null && a.len < 50).flatMap((a) => [a.dir.map((x) => x * a.len), a.dir.map((x) => -x * a.len)]);
    return { type: 'fl-quadric', info, extentPoints };
  },
});

// ---------------------------------------------------------------- scene (browser)

const rid = (it) => it.rowId ?? `i${it.index}`;
const cache = new Map();  // `${rowId}|${slot}` -> {id, key, obj, trash}: static groups reused across rebuilds
const swarms = new Map(); // rowId -> particle state

function cached(id, slot, key, build) {
  const k = `${id}|${slot}`, hit = cache.get(k);
  if (hit && hit.key === key) return hit.obj;
  if (hit) hit.trash.forEach((x) => x.dispose());
  const trash = [], obj = build(trash);
  cache.set(k, { id, key, obj, trash });
  return obj;
}

function purge(live) {
  for (const [k, e] of cache) if (!live.has(e.id)) { e.trash.forEach((x) => x.dispose()); cache.delete(k); }
  for (const [id, st] of swarms) if (!live.has(id)) { st.trash.forEach((x) => x.dispose()); swarms.delete(id); }
}

let DOT = null;
function dotTexture(THREE) {
  if (DOT) return DOT;
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d'), grad = g.createRadialGradient(32, 32, 0, 32, 32, 31);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.6, 'rgba(255,255,255,1)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  DOT = new THREE.CanvasTexture(cv);
  return DOT;
}

// Half-length of the line t·u (|u| = 1) inside the box |x_i| ≤ B, i < dim.
const reach = (u, B, dim = 3) => B / Math.max(...u.slice(0, dim).map(Math.abs), 1e-9);

function installRenderers(registerRenderer, THREE) {
  const Y = new THREE.Vector3(0, 1, 0);
  const SM = 'g-label fl-sm';

  const rod = (c, a, b, material, r, parent) => {
    const d = b.clone().sub(a), L = d.length();
    if (L < 1e-9) return null;
    const m = new THREE.Mesh(c.GEO.cyl, material);
    c.placeAlong(m, a, d, L, r);
    if (parent) parent.add(m); else c.add(m);
    return m;
  };
  // Beside the far end of the line o + t u (off the line, so lines along an axis don't cover
  // the axis labels): in-plane for 2D, toward +z otherwise.
  const endLabel = (c, o, u, T, dim, latex, color) => {
    const side = dim === 2 ? new THREE.Vector3(-u.y, u.x, 0)
      : Math.abs(u.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1).addScaledVector(u, -u.z).normalize();
    c.label(o.clone().addScaledVector(u, 0.9 * T).addScaledVector(side, 0.45 * c.s), latex, color, SM);
  };
  const boxPlanes = (o, B) => {
    const out = [];
    for (let j = 0; j < 3; j++) {
      for (const s of [1, -1]) {
        const n = new THREE.Vector3();
        n.setComponent(j, s);
        out.push(new THREE.Plane(n, B - s * o.getComponent(j)));
      }
    }
    return out;
  };

  // -------------------------------------------------------------- flow
  function flowStatic(c, it, info, hw, trash) {
    const grp = new THREE.Group(), E = c.E, s = c.s, dim = info.dim;
    grp.position.copy(c.v3(it.o));
    const { glyphs } = fieldGlyphs(info.A, dim, E), n = glyphs.length;
    if (n) {
      const m = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const shafts = new THREE.InstancedMesh(c.GEO.cyl, m, n), heads = new THREE.InstancedMesh(c.GEO.cone, m, n);
      trash.push(m, shafts, heads);
      const M4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), p = new THREE.Vector3();
      const d = new THREE.Vector3(), col = new THREE.Color(), amax = dim === 3 ? 0.5 : 0.85, bg = c.colors.bg;
      glyphs.forEach((g, i) => {
        d.fromArray(g.dir);
        q.setFromUnitVectors(Y, d);
        const L = g.len, hl = Math.min(0.4 * L, 0.26 * s), r = 0.017 * s;
        p.fromArray(g.p).addScaledVector(d, -L / 2 + (L - hl) / 2);
        M4.compose(p, q, sc.set(r, L - hl, r));
        shafts.setMatrixAt(i, M4);
        p.fromArray(g.p).addScaledVector(d, L / 2 - hl / 2);
        M4.compose(p, q, sc.set(0.075 * s, hl, 0.075 * s));
        heads.setMatrixAt(i, M4);
        col.set(mixHex(bg, it.color, g.alpha * amax));
        shafts.setColorAt(i, col);
        heads.setColorAt(i, col);
      });
      for (const o of [shafts, heads]) { o.frustumCulled = false; grp.add(o); }
    }
    const glass = c.mat('glass', hw, 0.6), solid = c.mat('solid', hw);
    for (const g of info.groups) {
      if (g.basis.length === 1) {
        const u = c.v3(g.basis[0]).normalize(), T = reach(g.basis[0], E, dim);
        rod(c, u.clone().multiplyScalar(-T), u.clone().multiplyScalar(T), glass, 0.03 * s, grp);
        if (g.lam === 0) continue;
        for (const f of [-0.75, -0.3, 0.3, 0.75]) { // chevrons: outward for λ > 0, inward for λ < 0
          const dir = u.clone().multiplyScalar(Math.sign(f) * Math.sign(g.lam));
          const head = new THREE.Mesh(c.GEO.cone, solid);
          c.placeAlong(head, u.clone().multiplyScalar(f * T).addScaledVector(dir, -0.14 * s), dir, 0.28 * s, 0.1 * s);
          grp.add(head);
        }
      } else if (g.basis.length === 2 && dim === 3) {
        const [e1, e2] = c.orthoBasis(g.basis).basis, pg = new THREE.PlaneGeometry(2 * E, 2 * E);
        trash.push(pg);
        const mesh = new THREE.Mesh(pg, c.mat('surface', hw, 0.12));
        mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(e1, e2, e1.clone().cross(e2)));
        grp.add(mesh);
      }
    }
    for (const pr of info.pairs) {
      for (const pts of sampleOrbits(info.A, pr, E)) {
        const lg = new THREE.BufferGeometry().setFromPoints(pts.map((x) => new THREE.Vector3(...x)));
        trash.push(lg);
        grp.add(new THREE.Line(lg, c.mat('line', hw, 0.8)));
      }
    }
    return grp;
  }

  function swarmFor(c, it, info) {
    const id = rid(it), reset = `${info.dim}|${c.E}|${it.o.join(',')}`;
    let st = swarms.get(id);
    if (st && st.reset !== reset) { st.trash.forEach((x) => x.dispose()); st = null; }
    if (!st) {
      const sw = makeSwarm(info.dim, c.E), n = sw.n, segs = TRAIL.segs;
      const pg = new THREE.BufferGeometry(), tg = new THREE.BufferGeometry();
      const pPos = new THREE.BufferAttribute(sw.pos, 3), pCol = new THREE.BufferAttribute(new Float32Array(4 * n).fill(1), 4);
      const tPos = new THREE.BufferAttribute(new Float32Array(n * segs * 6), 3);
      const tCol = new THREE.BufferAttribute(new Float32Array(n * segs * 8).fill(1), 4);
      for (const a of [pPos, pCol, tPos, tCol]) a.setUsage(THREE.DynamicDrawUsage);
      pg.setAttribute('position', pPos);
      pg.setAttribute('color', pCol);
      tg.setAttribute('position', tPos);
      tg.setAttribute('color', tCol);
      const pm = new THREE.PointsMaterial({
        size: 4.5, sizeAttenuation: false, map: dotTexture(THREE), vertexColors: true,
        transparent: true, depthWrite: false, alphaTest: 0.02,
      });
      const tm = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
      const points = new THREE.Points(pg, pm), trails = new THREE.LineSegments(tg, tm);
      for (const o of [points, trails]) { o.frustumCulled = false; o.raycast = () => {}; }
      st = { reset, sw, points, trails, pm, tm, pPos, pCol, tPos, tCol, akey: null, lastT: -1, trash: [pg, tg, pm, tm] };
      st.tick = (dt) => {
        stepSwarm(st.sw, rk4Matrix(st.A, dt * st.rate), dt);
        const pc = st.pCol.array, al = st.sw.alpha;
        for (let i = 0; i < al.length; i++) pc[4 * i + 3] = al[i];
        fillTrails(st.sw, st.back, TRAIL.segs, st.tPos.array, st.tCol.array);
        st.pPos.needsUpdate = st.pCol.needsUpdate = st.tPos.needsUpdate = st.tCol.needsUpdate = true;
      };
      swarms.set(id, st);
    }
    const akey = info.A.flat().join(',');
    if (st.akey !== akey) { // A changed (e.g. a slider): keep the particles, swap the dynamics
      st.akey = akey;
      st.A = info.A;
      st.rate = info.rate;
      st.back = rk4Matrix(info.A, (-TRAIL.seconds / TRAIL.segs) * info.rate);
      st.tick(0);
    }
    const col = mixHex(it.color, c.colors.fg, 0.3);
    st.pm.color.set(col);
    st.tm.color.set(col);
    st.points.position.copy(c.v3(it.o));
    st.trails.position.copy(c.v3(it.o));
    return st;
  }

  registerRenderer('fl-flow', (it, c) => {
    const info = it.info, o = c.v3(it.o), E = c.E, id = rid(it);
    const hw = mixHex(it.color, c.colors.fg, 0.45);
    const key = JSON.stringify([info.A, it.o, E, c.theme, it.color]);
    c.add(cached(id, 'flow', key, (trash) => flowStatic(c, it, info, hw, trash)));
    for (const g of info.groups) {
      const u = c.v3(g.basis[0]).normalize(), latex = `\\lambda = ${n3(g.lam)}`;
      if (g.basis.length === 1) endLabel(c, o, u, reach(g.basis[0], E, info.dim), info.dim, latex, hw);
      else if (g.basis.length === 2 && info.dim === 3) {
        c.label(o.clone().addScaledVector(u, 0.85 * E).addScaledVector(c.v3(g.basis[1]).normalize(), 0.85 * E), latex, hw, SM);
      }
    }
    const st = swarmFor(c, it, info);
    c.add(st.points, st.trails);
    c.onFrame((dt, t) => {
      if (st.lastT === t) return; // one step per frame even if the row is drawn twice
      st.lastT = t;
      st.tick(dt);
    });
  });

  // -------------------------------------------------------------- iterate / power
  function drawSequence(it, c) {
    const f = it.fl, o = c.v3(it.o), s = c.s, lim = 1000 * c.E, args = it.args ?? [];
    const n = f.seq.length - 1, first = Math.max(0, n - SHOWN + 1), tips = [];
    for (const x of f.seq) {
      const v = c.v3(x);
      if (v.length() > lim) break;
      tips.push(o.clone().add(v));
    }
    for (let k = first; k < tips.length; k++) {
      const last = k === n, w = (k - first + 1) / (n - first + 1);
      const opacity = last ? undefined : quant(0.12 + 0.6 * w ** 1.5);
      c.arrow(o, tips[k].clone().sub(o), it.color, { opacity, thickness: last ? 1.1 : 0.7, head: last ? 1 : 0.8 });
      if (!last) c.dot(tips[k], it.color, 0.045 * s, opacity);
    }
    const path = tips.slice();
    if (f.frac > 0 && f.next && tips.length === n + 1) {
      const p = c.v3(f.seq[n]).lerp(c.v3(f.next), f.frac);
      if (f.mode === 'power' && p.length() > 1e-9) p.setLength(f.R);
      if (p.length() < lim) {
        c.arrow(o, p, it.color, { opacity: quant(0.25 + 0.5 * f.frac), thickness: 0.8 });
        path.push(o.clone().add(p));
      }
    }
    if (path.length > 1) c.polyline(path, it.color, { opacity: 0.75 });
    const A = args[0] ? c.nameTex(args[0], false) : 'A', x = c.nameTex(args[1] ?? 'v', true);
    const tipLabel = (k, latex) => {
      const v = tips[k].clone().sub(o), L = v.length();
      c.label(L > 1e-9 ? tips[k].clone().addScaledVector(v, (0.32 * s) / L) : tips[k], latex, it.color, SM);
    };
    if (f.mode === 'iterate') {
      const every = n <= 5;
      for (let k = 0; k < tips.length; k++) if (every || k === 0 || k === n) tipLabel(k, `${powTex(A, k)}${x}`);
    } else {
      tipLabel(0, x);
      if (n > 0 && tips.length === n + 1) tipLabel(n, `\\vec q_{${n}}`);
    }
  }

  registerRenderer('fl-iterate', drawSequence);
  registerRenderer('fl-power', (it, c) => {
    const dom = it.fl.dom, o = c.v3(it.o), hw = mixHex(it.color, c.colors.fg, 0.45), dim = it.fl.dim;
    if (dom.kind === 'line') {
      const u = c.v3(dom.dir).normalize(), T = reach(dom.dir, c.E, dim);
      rod(c, o.clone().addScaledVector(u, -T), o.clone().addScaledVector(u, T), c.mat('glass', hw, 0.55), 0.024 * c.s);
      endLabel(c, o, u, T, dim, `\\lambda_1 = ${n3(dom.lam)}`, hw);
    } else if (dom.kind === 'plane' && dim === 3) {
      const [e1, e2] = c.orthoBasis(dom.basis).basis;
      c.planePatch(o, e1, e2, hw, null, { size: c.E, opacity: 0.12 });
    }
    drawSequence(it, c);
  });

  // -------------------------------------------------------------- quadric
  function quadricStatic(c, it, info, trash) {
    const grp = new THREE.Group(), o = c.v3(it.o), E = c.E, s = c.s;
    grp.position.copy(o);
    if (info.dim === 2) {
      const tube = c.mat('solid', it.color);
      for (const curve of conicCurves(info, Math.SQRT2 * E)) {
        for (const run of clipPolyline(curve, E, 2)) {
          const pts = run.map((p) => new THREE.Vector3(...p));
          const g = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.max(2, 2 * pts.length), 0.035 * s, 8, false);
          trash.push(g);
          grp.add(new THREE.Mesh(g, tube));
        }
      }
      return grp;
    }
    const planes = boxPlanes(o, E);
    const surf = new THREE.MeshStandardMaterial({
      color: it.color, transparent: true, opacity: 0.36, side: THREE.DoubleSide, depthWrite: false,
      roughness: 0.6, metalness: 0, clippingPlanes: planes,
    });
    const wire = new THREE.LineBasicMaterial({
      color: mixHex(it.color, c.colors.fg, 0.35), transparent: true, opacity: 0.5, depthWrite: false, clippingPlanes: planes,
    });
    trash.push(surf, wire);
    for (const { cols, rows, pts, wu, wv } of quadricPatches(info, Math.sqrt(3) * E)) {
      const pos = new Float32Array(pts.flat()), idx = [], segs = [];
      const at = (r, k) => r * (cols + 1) + k;
      for (let r = 0; r < rows; r++) {
        for (let k = 0; k < cols; k++) idx.push(at(r, k), at(r, k + 1), at(r + 1, k), at(r, k + 1), at(r + 1, k + 1), at(r + 1, k));
      }
      for (const k of new Set(Array.from({ length: wu + 1 }, (_, i) => Math.round((i * cols) / wu)))) {
        for (let r = 0; r < rows; r++) segs.push(...pts[at(r, k)], ...pts[at(r + 1, k)]);
      }
      for (const r of new Set(Array.from({ length: wv + 1 }, (_, i) => Math.round((i * rows) / wv)))) {
        for (let k = 0; k < cols; k++) segs.push(...pts[at(r, k)], ...pts[at(r, k + 1)]);
      }
      const g = new THREE.BufferGeometry(), lg = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setIndex(idx);
      g.computeVertexNormals();
      lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs), 3));
      trash.push(g, lg);
      grp.add(new THREE.Mesh(g, surf), new THREE.LineSegments(lg, wire));
    }
    return grp;
  }

  registerRenderer('fl-quadric', (it, c) => {
    const info = it.info, o = c.v3(it.o), s = c.s, E = c.E, d = info.dim;
    const key = JSON.stringify([info.S, info.c, it.o, E, c.theme, it.color]);
    c.add(cached(rid(it), 'quadric', key, (trash) => quadricStatic(c, it, info, trash)));
    if (info.kind === 'point') c.dot(o, it.color, 0.1 * s);
    if (info.kind === 'line' && d === 3) {
      const u = c.v3(info.vecs[info.order[2]]).normalize(), T = reach(info.vecs[info.order[2]], E);
      rod(c, o.clone().addScaledVector(u, -T), o.clone().addScaledVector(u, T), c.mat('solid', it.color), 0.035 * s);
    }
    const tn = tones(it.color, d, c.theme);
    info.axes.forEach((ax, i) => {
      const u = c.v3(ax.dir).normalize(), T = reach(ax.dir, E, d), col = tn[i];
      c.lines([o.clone().addScaledVector(u, -1.1 * T), o.clone().addScaledVector(u, 1.1 * T)], col, { dashed: true, opacity: 0.5 });
      if (ax.len != null) c.arrow(o, u.clone().multiplyScalar(Math.min(ax.len, 1.15 * E)), col, { opacity: ax.real ? undefined : 0.45, thickness: 0.8 });
      endLabel(c, o, u, 0.85 * T, d, `\\lambda_{${i + 1}} = ${n3(ax.lam)}`, col);
    });
  });
}

// ---------------------------------------------------------------- install

const CSS = `
.g-label.fl-sm { font-size: 16px; }
.g-out .fl-line { line-height: 1.55; }
`;
const HELP = `<p><code>flow(A)</code> x&#8242; = Ax: field, particles, eigenvector highways &middot;
<code>iterate(A, v, n)</code> v, Av, A&sup2;v, &hellip; &middot; <code>power(A, v, n)</code> power method &middot;
<code>quadric(A, c)</code> x&#7488;Ax = c (c = 1 if left out). n can be a slider.</p>`;

function tagItems(items, { rows }) {
  return items.map((it) => {
    const kind = it.fl ? `fl-${it.fl.mode}` : TYPES.has(it.kind) ? it.kind : null;
    if (!kind) return it;
    const out = { ...it, kind, args: callArgs(rows[it.index]?.src) };
    if (it.fl) out.extentPoints = it.fl.seq.filter((x) => Math.hypot(...x) < 1e4);
    return out;
  });
}

const ours = (v) => !!v && typeof v === 'object' && (TYPES.has(v.type) || !!v.fl);

export async function install(api) {
  const { registerRenderer, THREE } = await import('../scene.js');
  installRenderers(registerRenderer, THREE);
  api.addStyles(CSS);
  const help = document.getElementById('g-help');
  if (help) help.insertAdjacentHTML('beforeend', HELP);
  // quadric surfaces are clipped to the axes box with material clipping planes
  api.onSceneReady((scene) => { scene.renderer.localClippingEnabled = true; });
  api.addItemsHook(tagItems);
  api.onRecompute((results, rows) => {
    purge(new Set(rows.filter((r, i) => ours(results[i]?.value)).map((r) => r.id)));
    // a slider used as n starts out at 0..8 (iterate) or 0..20 (power) instead of -5..5
    let changed = false;
    for (const [j, hi] of stepRanges(rows.map((r) => r.src), results)) {
      const r = rows[j];
      if (r && r.min == null && r.max == null) { r.min = 0; r.max = Math.max(hi, Math.ceil(results[j].slider)); changed = true; }
    }
    if (changed) queueMicrotask(() => api.recompute());
  });
  // quadric readouts colour the eigenvalues with the theme's tones
  new MutationObserver(() => { if (api.results.some((r) => r?.value?.type === 'fl-quadric')) api.recompute(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  api.addRowDecorator((row, res, el) => {
    if (!res || res.error || !ours(res.value) || typeof katex === 'undefined') return;
    const lines = readoutLines(res.value, { args: callArgs(row.src), theme: document.documentElement.dataset.theme, color: row.color });
    if (!lines) return;
    el.out.innerHTML = lines.map((t) => `<div class="fl-line">${katex.renderToString(t, { throwOnError: false })}</div>`).join('');
  });
}
