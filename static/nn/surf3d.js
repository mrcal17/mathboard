// 3D plots panel for the Net tab (docs/NN_3D_PLOTS.md), floating over #nn-stage like the Train and
// Attention panels. Four plots, drawn with Three.js (imported the first time the panel opens):
//   surface    2-input nets: one neuron's activation (the output, or the selected neuron) as a height
//              over the input plane, with the training points; it follows training live.
//   landscape  the dataset loss on a plane through the weights θ₀, spanned by two random
//              filter-normalised directions (Li et al. 2018), the top two principal directions of the
//              training path, or two chosen weights; the path is projected onto it, at its true loss.
//   space      the dataset in each layer's activation space (up to three neurons as the axes, a wider
//              layer through its top three principal components), morphing from the inputs through
//              every z = W a + b and a = f(z).
//   simplex    3-class softmax nets: the predicted probabilities on the triangle ŷ₁ + ŷ₂ + ŷ₃ = 1.
//
// The panel is open exactly when state.s3d is set (null = closed), so the audience mirror drives it
// through the store: the mode, its settings, the landscape's plane and path, the morph stage and the
// camera (throttled). UI state (not in the store): localStorage 'mathboard.nn.s3d' = { x, y, w, h, mode }.
// ctx.surf3d = { MODES, open, mode, show(mode?, patch?), hide(), toggle(mode?), cycle(dir), recenter(),
//                clearPath(), available(), probe(), el }. P: open / close, Shift+P: next plot.
// The pure helpers below are exported for tests/surf3d.test.mjs.

import { colorFor, HI } from './store.js';
import { readSettings, forwardMany, datasetLoss } from './train.js';

export const MODES = ['surface', 'landscape', 'space', 'simplex'];
const MODE_LABEL = { surface: 'Surface', landscape: 'Landscape', space: 'Space', simplex: 'Simplex' };
const MODE_TITLE = {
  surface: 'A neuron (the output, or the one you select) as a height over the input plane (Shift+P: next plot)',
  landscape: 'The loss over a plane through the weights, with the training path on it (Shift+P: next plot)',
  space: 'The data in each layer\'s activation space, morphing from the inputs to the outputs (Shift+P: next plot)',
  simplex: 'A 3-class softmax: the predicted probabilities on the triangle ŷ₁ + ŷ₂ + ŷ₃ = 1 (Shift+P: next plot)',
};
const UI_KEY = 'mathboard.nn.s3d';
const W_DEFAULT = 460, H_DEFAULT = 340, W_MIN = 320, H_MIN = 180;
const FOV = 32;
const ZH = 1;                       // surface and landscape: the box is [-1, 1]² × [0, ZH]
const SURF_G = 49;                  // surface: grid vertices per side
const LAND_G0 = 11, LAND_G = 31;    // landscape: preview and final grid (odd, so the centre is a vertex)
const LAND_NMAX = 500;              // landscape: at most this many samples in the loss
const BUDGET_MS = 9;                // landscape grid work per frame
const REFRESH_MS = 90;              // data refresh interval while training runs
const CAM_MS = 120;                 // camera -> state.s3d.cam, at most this often
const TRAIL_MS = 60, TRAIL_MAX = 400, PATH_MS = 150, PMAX = 5000;   // PMAX: the landscape's largest net
const LINES = 9, PER = 40;          // input grid lines carried into layer space
const SPEED = 0.8;                  // morph: stages per second (slower near each stage)
const EXTRA = ['#3ec27a', '#b07cff', '#ff6fa8', '#2ec4c4', '#c9a227'];   // classes 3+, as in train.js
const BOUNDED = { sigmoid: [0, 1], softmax: [0, 1], tanh: [-1, 1] };
const FN = { relu: '\\mathrm{ReLU}', leaky: '\\mathrm{LReLU}', sigmoid: '\\sigma', tanh: '\\tanh', softmax: '\\mathrm{softmax}' };
const RANGES = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10];
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
// default views: the orbit target t and the camera's offset d from it (z up)
const CAM = {
  surface: { t: [0.05, 0, 0.36], d: [2.6, -3.85, 2.5] },
  landscape: { t: [0.05, 0, 0.36], d: [2.6, -3.85, 2.5] },
  space: { t: [0, 0, -0.04], d: [3.15, -3.95, 2.45] },
  simplex: { t: [-1 / 3, -1 / 3, -1 / 3], d: [4.3, 4.3, 4.3] },
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const texEsc = s => String(s).replace(/[\\{}$&#^_%~]/g, c => ({ '\\': '\\textbackslash{}', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}' }[c] || `\\${c}`));
const isTie = t => typeof t === 'string' && t !== '';
const TIE_RE = /^(.*):\s*(\d+)\s*,\s*(\d+)\s*$/;
const round = (v, p = 10) => (Number.isFinite(v) ? +v.toPrecision(p) : 0);
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

// ---------------------------------------------------------------- pure helpers (exported for tests)

// The trainable parameters, in reading order (layer by layer: its weights row by row, then its
// biases), exactly as model.trainStep steps them: fixed edges and attention biases are left out, a
// tie group is one parameter. filter: the group filter normalisation scales together, a neuron's
// incoming weights and its bias (a shared matrix W:i,j: its column j).
export function paramList(net, model, M = model.matrices(net)) {
  const edges = new Map(net.edges.map(e => [e.id, e]));
  const out = [], tied = new Map();
  const add = (key, kind, id, filter, tie) => {
    if (tie) {
      const p = tied.get(key);
      if (p) { p.ids.push(id); return; }
      const q = { key, kind, ids: [id], filter, tie };
      tied.set(key, q);
      out.push(q);
      return;
    }
    out.push({ key, kind, ids: [id], filter, tie: null });
  };
  for (const m of M) {
    if (m.kind !== 'dense') continue;
    for (const term of m.terms) {
      for (const row of term.edge) {
        for (const id of row) {
          const e = id != null ? edges.get(id) : null;
          if (!e || e.fixed === true) continue;
          if (isTie(e.tie)) {
            const t = TIE_RE.exec(e.tie);
            add('t:' + e.tie, 'w', id, t ? `t:${t[1]}|${t[3]}` : 'n:' + e.to, e.tie);
          } else add('e:' + id, 'w', id, 'n:' + e.to);
        }
      }
    }
    for (const id of m.rows) {
      const nd = model.node(net, id);
      if (!nd) continue;
      if (isTie(nd.tie)) add('bt:' + nd.tie, 'b', id, 'n:' + id, nd.tie);
      else add('b:' + id, 'b', id, 'n:' + id);
    }
  }
  return out;
}

function lookups(net) {
  return { e: new Map(net.edges.map(e => [e.id, e])), n: new Map(net.nodes.map(n => [n.id, n])) };
}

export function readParams(net, params) {
  const L = lookups(net), v = new Float64Array(params.length);
  params.forEach((p, k) => {
    const x = p.kind === 'w' ? L.e.get(p.ids[0])?.w : L.n.get(p.ids[0])?.bias;
    v[k] = Number.isFinite(x) ? x : 0;
  });
  return v;
}

export function writeParams(net, params, vec) {
  const L = lookups(net);
  params.forEach((p, k) => {
    for (const id of p.ids) {
      const o = p.kind === 'w' ? L.e.get(id) : L.n.get(id);
      if (o) { if (p.kind === 'w') o.w = vec[k]; else o.bias = vec[k]; }
    }
  });
}

function gauss(rand) {
  let u = 0;
  while (u <= 1e-12) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };

// Filter normalisation (Li et al. 2018): every filter's slice of dir gets the norm of the same
// filter of theta. A filter whose weights are all 0 (the W = 0 presets) gets the RMS of the other
// filters' norms instead (1 when every filter is 0), so the plane is never flat in it.
export function filterNormalize(dir, theta, params) {
  const groups = new Map();
  params.forEach((p, k) => { const g = groups.get(p.filter); if (g) g.push(k); else groups.set(p.filter, [k]); });
  const list = [...groups.values()];
  const tn = list.map(ks => Math.sqrt(ks.reduce((s, k) => s + theta[k] * theta[k], 0)));
  const nz = tn.filter(v => v > 1e-12);
  const fallback = nz.length ? Math.sqrt(nz.reduce((s, v) => s + v * v, 0) / nz.length) : 1;
  const out = Float64Array.from(dir);
  list.forEach((ks, g) => {
    const dn = Math.sqrt(ks.reduce((s, k) => s + dir[k] * dir[k], 0));
    if (dn < 1e-15) return;
    const f = (tn[g] > 1e-12 ? tn[g] : fallback) / dn;
    for (const k of ks) out[k] = dir[k] * f;
  });
  return out;
}

// Two filter-normalised Gaussian directions. In a small net two draws can come out nearly parallel;
// d2 is redrawn (up to 12 times) until |cos(d1, d2)| <= 0.5.
export function randomDirections(params, theta, rand) {
  const P = params.length;
  const draw = () => filterNormalize(Float64Array.from({ length: P }, () => gauss(rand)), theta, params);
  const d1 = draw();
  let d2 = draw();
  const cos = (a, b) => dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);
  for (let k = 0; k < 12 && P > 1 && Math.abs(cos(d1, d2)) > 0.5; k++) d2 = draw();
  return [d1, d2];
}

// Least-squares coordinates of delta in the plane of d1, d2: delta ≈ a d1 + b d2, r = the distance
// from the plane.
export function project(delta, d1, d2) {
  let g11 = 0, g12 = 0, g22 = 0, r1 = 0, r2 = 0, dd = 0;
  for (let k = 0; k < delta.length; k++) {
    const x = delta[k], u = d1[k], v = d2[k];
    g11 += u * u; g12 += u * v; g22 += v * v; r1 += u * x; r2 += v * x; dd += x * x;
  }
  const det = g11 * g22 - g12 * g12;
  let a, b;
  if (det > 1e-12 * (g11 * g22 || 1)) { a = (g22 * r1 - g12 * r2) / det; b = (g11 * r2 - g12 * r1) / det; }
  else { a = g11 > 0 ? r1 / g11 : 0; b = 0; }
  return { a, b, r: Math.sqrt(Math.max(0, dd - (a * r1 + b * r2))) };
}

// Symmetric eigendecomposition (cyclic Jacobi). A: n x n row-major. Returns the eigenvalues in
// decreasing order and the matching eigenvectors as the columns of an n x n row-major array.
export function symEig(A, n) {
  const a = Float64Array.from(A), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  let scale = 0;
  for (let i = 0; i < n * n; i++) scale += a[i] * a[i];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off <= 1e-24 * (scale || 1)) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const kp = a[k * n + p], kq = a[k * n + q];
          a[k * n + p] = c * kp - s * kq; a[k * n + q] = s * kp + c * kq;
        }
        for (let k = 0; k < n; k++) {
          const pk = a[p * n + k], qk = a[q * n + k];
          a[p * n + k] = c * pk - s * qk; a[q * n + k] = s * pk + c * qk;
        }
        for (let k = 0; k < n; k++) {
          const kp = v[k * n + p], kq = v[k * n + q];
          v[k * n + p] = c * kp - s * kq; v[k * n + q] = s * kp + c * kq;
        }
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = order.map(i => a[i * n + i]);
  const vectors = new Float64Array(n * n);
  order.forEach((i, c) => { for (let k = 0; k < n; k++) vectors[k * n + c] = v[k * n + i]; });
  return { values, vectors };
}

// The top two principal directions of a path of parameter vectors (unit vectors), and the share of
// the path's variance they hold. null when the path doesn't move.
export function pathDirections(rows) {
  const T = rows.length, P = T ? rows[0].length : 0;
  if (T < 2 || P < 2) return null;
  const mean = new Float64Array(P);
  for (const r of rows) for (let k = 0; k < P; k++) mean[k] += r[k] / T;
  const X = rows.map(r => Float64Array.from(r, (v, k) => v - mean[k]));
  const K = new Float64Array(T * T);
  for (let i = 0; i < T; i++) for (let j = i; j < T; j++) K[i * T + j] = K[j * T + i] = dot(X[i], X[j]);
  const { values, vectors } = symEig(K, T);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0);
  if (!(values[0] > 1e-18) || !(total > 0)) return null;
  const dir = c => {
    const d = new Float64Array(P);
    for (let t = 0; t < T; t++) { const u = vectors[t * T + c]; for (let k = 0; k < P; k++) d[k] += u * X[t][k]; }
    const n = Math.sqrt(dot(d, d)) || 1;
    for (let k = 0; k < P; k++) d[k] /= n;
    return d;
  };
  const d1 = dir(0);
  let d2;
  if (values[1] > 1e-9 * values[0]) d2 = dir(1);
  else {
    // a straight path: any unit vector orthogonal to d1 (the axis least along it)
    let best = 0;
    for (let k = 1; k < P; k++) if (Math.abs(d1[k]) < Math.abs(d1[best])) best = k;
    d2 = new Float64Array(P);
    d2[best] = 1;
    const c = d1[best];
    for (let k = 0; k < P; k++) d2[k] -= c * d1[k];
    const n = Math.sqrt(dot(d2, d2)) || 1;
    for (let k = 0; k < P; k++) d2[k] /= n;
  }
  return { d1, d2, share: (Math.max(0, values[0]) + Math.max(0, values[1] || 0)) / total };
}

// Marching squares on a G x G grid (vals[j * G + i], i along x). Returns [u0, v0, u1, v1, ...] in
// grid index coordinates, one pair of points per segment.
export function isoLines(vals, G, level) {
  const out = [];
  for (let j = 0; j < G - 1; j++) {
    for (let i = 0; i < G - 1; i++) {
      const a = vals[j * G + i], b = vals[j * G + i + 1], c = vals[(j + 1) * G + i + 1], d = vals[(j + 1) * G + i];
      if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || !Number.isFinite(d)) continue;
      const code = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const f = (p, q) => (q === p ? 0.5 : (level - p) / (q - p));
      const E = [
        [i + f(a, b), j], [i + 1, j + f(b, c)], [i + f(d, c), j + 1], [i, j + f(a, d)],
      ];
      const seg = (p, q) => out.push(E[p][0], E[p][1], E[q][0], E[q][1]);
      const mid = (a + b + c + d) / 4 > level;
      switch (code) {
        case 1: case 14: seg(3, 0); break;
        case 2: case 13: seg(0, 1); break;
        case 3: case 12: seg(3, 1); break;
        case 4: case 11: seg(1, 2); break;
        case 6: case 9: seg(0, 2); break;
        case 7: case 8: seg(2, 3); break;
        case 5: if (mid) { seg(0, 1); seg(2, 3); } else { seg(3, 0); seg(1, 2); } break;
        case 10: if (mid) { seg(3, 0); seg(1, 2); } else { seg(0, 1); seg(2, 3); } break;
        default: break;
      }
    }
  }
  return out;
}

export function niceTicks(lo, hi, max = 5) {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return Number.isFinite(lo) ? [lo] : [];
  const raw = (hi - lo) / max, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
  return out;
}

// A round number >= v: 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 times a power of ten.
export function niceUp(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v)), m = v / p;
  return ([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find(x => x >= m - 1e-9) || 10) * p;
}

const SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '(': '⁽', ')': '⁾', '+': '⁺', '-': '⁻', n: 'ⁿ', i: 'ⁱ', T: 'ᵀ', ' ': '' };
const SUB = {
  0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋', ',': ',', ' ': '',
  a: 'ₐ', e: 'ₑ', h: 'ₕ', i: 'ᵢ', j: 'ⱼ', k: 'ₖ', l: 'ₗ', m: 'ₘ', n: 'ₙ', o: 'ₒ', p: 'ₚ', r: 'ᵣ', s: 'ₛ', t: 'ₜ', u: 'ᵤ', v: 'ᵥ', x: 'ₓ',
};
const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', theta: 'θ', lambda: 'λ', mu: 'μ', pi: 'π', sigma: 'σ', phi: 'φ', psi: 'ψ', omega: 'ω', to: '→' };

// A KaTeX label as plain text for a <select>: x_{1} -> x₁, h^{(1)}_{2} -> h⁽¹⁾₂, \hat y_{1} -> ŷ₁.
export function plainLabel(tex) {
  let t = String(tex ?? '');
  t = t.replace(/\\hat\s*\{?\s*y\s*\}?/g, 'ŷ').replace(/\\hat\s*\{?\s*([a-zA-Z])\s*\}?/g, '$1̂');
  t = t.replace(/\\(?:mathrm|text|mathbf|mathit|operatorname|boldsymbol|mathsf)\s*\{([^{}]*)\}/g, '$1');
  t = t.replace(/\\([a-zA-Z]+)\s*/g, (m, w) => GREEK[w] ?? '');
  const conv = (s, map) => ([...s].every(c => c in map) ? [...s].map(c => map[c]).join('') : null);
  t = t.replace(/\^\{([^{}]*)\}|\^(\S)/g, (m, a, b) => conv(a ?? b, SUP) ?? `^${a ?? b}`);
  t = t.replace(/_\{([^{}]*)\}|_(\S)/g, (m, a, b) => conv(a ?? b, SUB) ?? `_${a ?? b}`);
  return t.replace(/[{}]/g, '').replace(/\\[,;!: ]/g, ' ').replace(/\s+/g, ' ').trim();
}

// ---- token layers (as train.js reads them): softmax runs per token (and group)
const layerTokens = ly => (ly && Number.isInteger(ly.tokens) && ly.tokens > 1 ? ly.tokens : 1);
const layerGroups = ly => (ly && Array.isArray(ly.groups) && ly.groups.length ? ly.groups.length : 1);
function softmaxBlock(ly, R) {
  const k = layerTokens(ly) * layerGroups(ly);
  return k > 1 && R % k === 0 ? R / k : R;
}

// Every layer's pre-activations and activations for n inputs at once (train.js's forwardMany, which
// keeps only a). X: row-major n x size(layer 0). Returns { a, z, hasZ } with a[l] / z[l] n x size_l
// (z[0] = null). A net with attention goes through model.predict, which has no z (hasZ false).
export function forwardZA(net, model, X, n, M = model.matrices(net)) {
  const L = net.layers.length;
  const sizes = [], pos = Object.create(null);
  for (let l = 0; l < L; l++) {
    const ns = model.nodesIn(net, l);
    sizes.push(ns.length);
    ns.forEach((nd, i) => { pos[nd.id] = i; });
  }
  if (M.some(m => m.kind === 'attention')) {
    const d0 = sizes[0], rows = new Array(n);
    for (let s = 0; s < n; s++) rows[s] = Array.from(X.subarray ? X.subarray(s * d0, (s + 1) * d0) : X.slice(s * d0, (s + 1) * d0));
    const per = model.predict(net, rows, { layer: 'all' });
    const a = sizes.map(sz => new Float64Array(n * sz));
    for (let s = 0; s < n; s++) {
      for (let l = 0; l < L; l++) {
        const v = per[s]?.[l];
        if (!v) return null;
        for (let i = 0; i < sizes[l]; i++) a[l][s * sizes[l] + i] = v[i];
      }
    }
    return { a, z: sizes.map(() => null), hasZ: false };
  }
  const A = new Array(L).fill(null), Zs = new Array(L).fill(null);
  A[0] = X;
  for (const m of [...M].sort((p, q) => p.l - q.l)) {
    const l = m.l, R = m.rows.length, Z = new Float64Array(n * R);
    for (let i = 0; i < R; i++) {
      const b = m.b[i] || 0;
      if (b) for (let s = 0; s < n; s++) Z[s * R + i] = b;
    }
    for (const term of m.terms) {
      const Ak = A[term.k];
      if (!Ak) return null;
      const C = sizes[term.k], idx = term.cols.map(id => pos[id]);
      for (let i = 0; i < R; i++) {
        const w = term.W[i];
        for (let j = 0; j < idx.length; j++) {
          const wij = w[j];
          if (!wij) continue;
          const c = idx[j];
          for (let s = 0; s < n; s++) Z[s * R + i] += wij * Ak[s * C + c];
        }
      }
    }
    Zs[l] = Z;
    const out = new Float64Array(Z), act = m.act;
    if (act === 'softmax') {
      const B = softmaxBlock(net.layers[l], R);
      for (let o = 0; o < n * R; o += B) {
        let mx = -Infinity, sum = 0;
        for (let i = 0; i < B; i++) mx = Math.max(mx, out[o + i]);
        for (let i = 0; i < B; i++) { out[o + i] = Math.exp(out[o + i] - mx); sum += out[o + i]; }
        for (let i = 0; i < B; i++) out[o + i] /= sum;
      }
    } else {
      const f = model.ACTS?.[act]?.f;
      if (f && act !== 'identity') for (let k = 0; k < out.length; k++) out[k] = f(out[k]);
    }
    A[l] = out;
  }
  return { a: A, z: Zs, hasZ: true };
}

// The loss at any parameter vector, over (at most LAND_NMAX of) the dataset's samples. A net without
// attention patches the weight matrices in place and reuses train.js's forwardMany; a net with
// attention writes the vector into a scratch copy and runs model.predict.
export function makeEvaluator(net, model, data, nmax = LAND_NMAX) {
  const M = model.matrices(net);
  const params = paramList(net, model, M);
  const L = net.layers.length, last = net.layers[L - 1], K = data.K;
  const ns = Math.min(data.n, nmax), dim = data.dim;
  const idx = Array.from({ length: ns }, (_, k) => (ns === data.n ? k : Math.floor(k * data.n / ns)));
  const Xs = new Float64Array(ns * dim), Ys = idx.map(s => data.Y[s]);
  idx.forEach((s, k) => { for (let j = 0; j < dim; j++) Xs[k * dim + j] = data.Xf[s * dim + j]; });
  const lossOf = out => datasetLoss(out, Ys, ns, K, net.meta?.loss || 'mse', last.act, K / softmaxBlock(last, K));
  let evalAt;
  if (M.some(m => m.kind === 'attention')) {
    const scratch = model.clone(net);
    evalAt = vec => {
      writeParams(scratch, params, vec);
      const acts = forwardMany(scratch, model, Xs, ns);
      return acts && acts[L - 1] ? lossOf(acts[L - 1]) : NaN;
    };
  } else {
    const at = new Map();
    for (const m of M) {
      if (m.kind !== 'dense') continue;
      for (const term of m.terms) term.edge.forEach((row, i) => row.forEach((id, j) => { if (id != null) at.set('w' + id, [term.W[i], j]); }));
      m.rows.forEach((id, i) => at.set('b' + id, [m.b, i]));
    }
    const locs = params.map(p => p.ids.map(id => at.get(p.kind + id)).filter(Boolean));
    evalAt = vec => {
      for (let p = 0; p < locs.length; p++) { const v = vec[p]; for (const [arr, j] of locs[p]) arr[j] = v; }
      const acts = forwardMany(net, model, Xs, ns, 0, M);
      return acts && acts[L - 1] ? lossOf(acts[L - 1]) : NaN;
    };
  }
  return { params, evalAt, n: ns, total: data.n };
}

// ---------------------------------------------------------------- small colour utilities

function parseRGB(str) {
  const s = String(str || '').trim();
  let m = /^#([0-9a-f]{3})$/i.exec(s);
  if (m) return [...m[1]].map(c => parseInt(c + c, 16));
  m = /^#([0-9a-f]{6})/i.exec(s);
  if (m) return [0, 2, 4].map(k => parseInt(m[1].slice(k, k + 2), 16));
  m = /rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(s);
  if (m) return [+m[1], +m[2], +m[3]];
  return null;
}
const lin = c => { const x = c / 255; return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
function mixInto(arr, o, base, rgb, a) {   // linear vertex colour of base + (rgb - base) a
  for (let k = 0; k < 3; k++) arr[o + k] = lin(base[k] + (rgb[k] - base[k]) * a);
}
function viridis(u) {
  const x = clamp(u, 0, 1) * (VIRIDIS.length - 1), i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  return VIRIDIS[i].map((c, k) => c + (VIRIDIS[i + 1][k] - c) * f);
}

// ---------------------------------------------------------------- the panel

export function install(ctx) {
  const { store, model } = ctx;
  const ro = !!ctx.audience;
  const stage = ctx.el.stage;
  if (getComputedStyle(stage).position === 'static') stage.style.position = 'relative';
  const shownTab = () => document.body.dataset.view === 'nn' && !ctx.el.root.hidden;
  const themeNow = () => (ctx.theme ? ctx.theme() : document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  const fmt = (v, dg = 2) => (Number.isFinite(v) ? model.fmt(v, dg).replace(/^-/, '−') : '?');
  const fmtK = (v, dg = 2) => (Number.isFinite(v) ? model.fmt(v, dg) : '?');   // inside KaTeX
  const tickText = v => String(+(+v).toPrecision(3)).replace(/^-/, '−');
  const lossText = v => (!Number.isFinite(v) ? '?' : v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(2) : v.toFixed(4));
  let lastValues = 0;
  // while training (or, in the audience, while nets keep arriving) data refreshes are throttled
  const busy = () => !!ctx.train?.running || (ro && performance.now() - lastValues < 300);

  // ---- panel UI state: spot, width, view height, last mode (the audience reads the presenter's)
  const ui = { x: null, y: null, w: W_DEFAULT, h: H_DEFAULT, mode: 'surface' };
  const readUi = () => { try { return JSON.parse(localStorage.getItem(UI_KEY)) || {}; } catch { return {}; } };
  const loadUi = () => {
    const s = readUi();
    ui.x = Number.isFinite(s.x) ? s.x : null;
    ui.y = Number.isFinite(s.y) ? s.y : null;
    ui.w = Number.isFinite(s.w) ? Math.max(W_MIN, s.w) : W_DEFAULT;
    ui.h = Number.isFinite(s.h) ? Math.max(H_MIN, s.h) : H_DEFAULT;
    ui.mode = MODES.includes(s.mode) ? s.mode : 'surface';
  };
  loadUi();
  const saveUi = () => { if (!ro) try { localStorage.setItem(UI_KEY, JSON.stringify(ui)); } catch { /* ignore */ } };
  if (ro) window.addEventListener('storage', e => { if (e.key === UI_KEY || e.key === null) { loadUi(); placePanel(); } });

  // ---- DOM
  const panel = document.createElement('div');
  panel.className = 'nn-s3d' + (ro ? ' ro' : '');
  panel.hidden = true;
  panel.innerHTML = `
    <header class="s3-head">
      <b class="s3-title" title="Drag to move; double-click to put it back">3D plots</b>
      <div class="s3-modes">${MODES.map(m => `<button type="button" data-mode="${m}" title="${esc(MODE_TITLE[m])}">${MODE_LABEL[m]}</button>`).join('')}</div>
      <span class="s3-flex"></span>
      <button type="button" class="s3-home" data-act="home" title="Reset the view. Drag to orbit, right-drag to pan, wheel to zoom">&#8962;</button>
      <button type="button" class="s3-close" data-act="close" title="Close (P)">&times;</button>
    </header>
    <div class="s3-ctl"></div>
    <div class="s3-view"><div class="s3-msg" hidden></div></div>
    <div class="s3-cap"><div class="s3-tex"></div><div class="s3-note"></div></div>
    <div class="s3-grip" title="Drag to resize"></div>`;
  stage.appendChild(panel);
  const $ = s => panel.querySelector(s);
  const head = $('.s3-head'), ctl = $('.s3-ctl'), viewEl = $('.s3-view'), msgEl = $('.s3-msg');
  const capTex = $('.s3-tex'), capNote = $('.s3-note'), grip = $('.s3-grip');
  for (const type of ['pointerdown', 'mousedown', 'click', 'dblclick', 'wheel', 'contextmenu', 'touchstart']) {
    panel.addEventListener(type, e => e.stopPropagation(), { passive: type === 'wheel' || type === 'touchstart' });
  }
  // A clicked button must not keep focus: Space belongs to training.
  panel.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

  // Default spot: top left, or right of an open attention panel sitting there.
  function defaultSpot(w) {
    const sw = stage.clientWidth;
    const a = ctx.attnviz?.el;
    if (a && !a.hidden && a.offsetWidth) {
      const x = a.offsetLeft + a.offsetWidth + 10;
      if (a.offsetLeft < 40 && x + w <= sw - 10) return { x, y: 10 };
    }
    return { x: 10, y: 10 };
  }
  // Width, spot and view height from ui, kept inside the stage.
  function placePanel() {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const w = sw > 0 ? Math.min(ui.w, Math.max(W_MIN, sw - 20)) : ui.w;
    panel.style.width = `${Math.round(w)}px`;
    let x, y;
    if (ui.x == null || ui.y == null) ({ x, y } = defaultSpot(w));
    else {
      x = sw > 0 ? clamp(ui.x, 0, Math.max(0, sw - w)) : ui.x;
      y = sh > 0 ? clamp(ui.y, 0, Math.max(0, sh - 120)) : ui.y;
    }
    panel.style.left = `${Math.round(x)}px`;
    panel.style.top = `${Math.round(y)}px`;
    const chrome = panel.hidden ? 150 : Math.max(80, panel.offsetHeight - viewEl.offsetHeight);
    const h = sh > 0 ? clamp(ui.h, 120, Math.max(120, sh - y - chrome - 8)) : ui.h;
    viewEl.style.height = `${Math.round(h)}px`;
  }
  placePanel();
  new ResizeObserver(() => placePanel()).observe(stage);

  head.addEventListener('pointerdown', e => {
    if (e.button !== 0 || ro || e.target.closest('button, select')) return;
    const sr = stage.getBoundingClientRect(), pr = panel.getBoundingClientRect();
    const dx = e.clientX - pr.left, dy = e.clientY - pr.top;
    head.setPointerCapture(e.pointerId);
    const move = ev => {
      ui.x = Math.round(clamp(ev.clientX - sr.left - dx, 0, Math.max(0, sr.width - pr.width)));
      ui.y = Math.round(clamp(ev.clientY - sr.top - dy, 0, Math.max(0, sr.height - 40)));
      placePanel();
    };
    const up = () => {
      head.removeEventListener('pointermove', move);
      head.removeEventListener('pointerup', up);
      head.removeEventListener('pointercancel', up);
      saveUi();
    };
    head.addEventListener('pointermove', move);
    head.addEventListener('pointerup', up);
    head.addEventListener('pointercancel', up);
  });
  head.addEventListener('dblclick', e => {
    if (ro || e.target.closest('button, select')) return;
    ui.x = ui.y = null;
    placePanel();
    saveUi();
  });
  // The corner grip sets the width and the view's height.
  grip.addEventListener('pointerdown', e => {
    if (e.button !== 0 || ro) return;
    e.preventDefault();
    const sr = stage.getBoundingClientRect(), pr = panel.getBoundingClientRect();
    const x0 = e.clientX, y0 = e.clientY, w0 = pr.width, h0 = viewEl.offsetHeight;
    grip.setPointerCapture(e.pointerId);
    const move = ev => {
      ui.w = Math.round(clamp(w0 + ev.clientX - x0, W_MIN, Math.max(W_MIN, sr.right - pr.left - 6)));
      ui.h = Math.round(clamp(h0 + ev.clientY - y0, H_MIN, 2400));
      placePanel();
    };
    const up = () => {
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', up);
      grip.removeEventListener('pointercancel', up);
      ui.h = viewEl.offsetHeight;   // what the stage allowed
      saveUi();
    };
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', up);
    grip.addEventListener('pointercancel', up);
  });

  // ---------------------------------------------------------------- state.s3d
  const cur = () => { const s = store.state.s3d; return s && MODES.includes(s.mode) ? s : null; };
  const fresh = mode => ({
    mode, neuron: 'sel', pre: false, dirs: null, seed: 1, wa: null, wb: null, range: 0, log: false,
    basis: null, trail: [], stage: null, cam: null,
  });
  function setS3d(patch) {
    if (ro) return false;
    const s = cur();
    if (!s) return false;
    const next = { ...s, ...patch };
    if (same(next, s)) return false;
    store.set('s3d', next);
    return true;
  }
  function show(mode, patch = {}) {
    if (ro) return false;
    const s = cur();
    const m = MODES.includes(mode) ? mode : s?.mode || ui.mode;
    if (ui.mode !== m) { ui.mode = m; saveUi(); }
    const next = { ...fresh(m), ...(s || {}), ...patch, mode: m };
    if (!s || s.mode !== m) next.cam = patch.cam ?? null;
    if (!same(next, s)) store.set('s3d', next);
    return true;
  }
  function hide() { if (!ro && store.state.s3d) store.set('s3d', null); }
  function toggle(mode) { const s = cur(); if (s && (mode == null || s.mode === mode)) hide(); else show(mode); }
  function cycle(dir = 1) {
    const s = cur();
    if (!s) return show(ui.mode);
    return show(MODES[(MODES.indexOf(s.mode) + dir + MODES.length) % MODES.length]);
  }

  // ---------------------------------------------------------------- data
  const shape = net => {
    const L = net.layers.length;
    return { L, inputs: L ? model.nodesIn(net, 0).length : 0, outputs: L > 1 ? model.nodesIn(net, L - 1).length : 0 };
  };
  let dataC = null;
  function getData() {
    const net = store.net, t = readSettings(net, model), ds = model.DATASETS?.[t.dataset];
    if (!ds) return null;
    const key = [t.dataset, t.n, t.noise, t.seed].join('|');
    if (dataC && dataC.key === key) return dataC;
    let X, Y;
    try { ({ X, Y } = ds.make(t.n, t.seed, t.noise)); } catch (err) { console.warn('[nn/surf3d] dataset:', err); return null; }
    const flat = r => (Array.isArray(r) && r.some(Array.isArray) ? r.flat(Infinity) : r);
    if (ds.kind === 'seq') { X = X.map(flat); Y = Y.map(flat); }
    const n = X.length, dim = ds.inputs, K = ds.outputs;
    const Xf = new Float64Array(n * dim);
    X.forEach((x, s) => { for (let j = 0; j < dim; j++) Xf[s * dim + j] = x[j]; });
    let lo = Infinity, hi = -Infinity;
    for (const y of Y) { lo = Math.min(lo, y[0]); hi = Math.max(hi, y[0]); }
    const mid = (lo + hi) / 2, half = (hi - lo) / 2 || 1;
    const cls = new Int32Array(n);
    Y.forEach((y, s) => {
      if (y.length > 1) { let b = 0; for (let k = 1; k < y.length; k++) if (y[k] > y[b]) b = k; cls[s] = b; }
      else cls[s] = y[0] > mid ? 1 : 0;
    });
    let dom = null;
    if (ds.kind !== 'seq' && dim === 2) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const x of X) { x0 = Math.min(x0, x[0]); x1 = Math.max(x1, x[0]); y0 = Math.min(y0, x[1]); y1 = Math.max(y1, x[1]); }
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, h = (Math.max(x1 - x0, y1 - y0) / 2 || 1) * 1.1;
      dom = { x0: cx - h, x1: cx + h, y0: cy - h, y1: cy + h };
    }
    dataC = { key, ds, kind: ds.kind, X, Y, Xf, n, dim, K, cls, mid, half, dom, T: ds.kind === 'seq' ? Math.max(1, ds.tokens | 0 || 1) : 1 };
    return dataC;
  }
  const fitsIn = (net, d) => {
    if (!d || d.dim !== shape(net).inputs) return false;
    const T0 = layerTokens(net.layers[0]);
    return T0 === 1 || T0 === d.T;
  };
  const fitsAll = (net, d) => fitsIn(net, d) && d.K === shape(net).outputs;

  // What the parameters and the evaluator depend on beyond the store's structural signature: ties,
  // fixed edges (and their weights), activations and attention settings.
  const structSig = net => JSON.stringify([
    net.layers.map(l => [l.id, l.act, l.kind, l.heads, l.causal, l.scale, l.tokens, l.groups]),
    net.nodes.map(q => [q.id, q.layer, q.tie]),
    net.edges.map(e => [e.id, e.from, e.to, e.tie, e.fixed === true, e.fixed === true ? e.w : 0]),
  ]);
  let evalC = null;
  function evaluator() {
    const net = store.net, d = getData();
    if (!d || !fitsAll(net, d)) return null;
    const sig = structSig(net), key = sig + '|' + d.key + '|' + (net.meta?.loss || 'mse');
    if (evalC && evalC.key === key) return evalC;
    try {
      const ev = makeEvaluator(net, model, d);
      evalC = { ...ev, key, psig: ev.params.map(p => p.key).join(','), dataKey: d.key };
    } catch (err) {
      console.warn('[nn/surf3d] evaluator:', err);
      evalC = null;
    }
    return evalC;
  }

  const nodeTex = (net, id) => {
    const q = model.node(net, id);
    return (q && String(q.label ?? '').trim()) || model.defaultLabel?.(net, id) || '?';
  };
  const outNodes = net => model.nodesIn(net, net.layers.length - 1);
  const isSimplexNet = net => {
    const L = net.layers.length, last = net.layers[L - 1];
    return L > 1 && last.act === 'softmax' && last.kind !== 'attention' && outNodes(net).length === 3
      && layerTokens(last) === 1 && layerGroups(last) === 1;
  };

  // Why a mode can't draw for this net ('' = it can).
  function availability() {
    const net = store.net, sh = shape(net), d = getData();
    const dsName = d ? d.ds.label || 'the dataset' : 'the dataset';
    const out = { surface: '', landscape: '', space: '', simplex: '' };
    if (sh.inputs !== 2) out.surface = `The surface needs a net with 2 inputs; this one has ${sh.inputs}. Try xor, circles or the classifier.`;
    if (!d) out.landscape = out.space = out.simplex = 'No dataset: pick one in the Train panel.';
    else {
      if (!fitsAll(store.net, d)) out.landscape = `The loss needs a dataset that fits the net: ${dsName} doesn't (Train panel: Adapt network).`;
      else {
        const ev = evaluator();
        if (!ev) out.landscape = 'Could not evaluate the loss for this net.';
        else if (ev.params.length < 2) out.landscape = 'The loss landscape needs at least 2 trainable parameters.';
        else if (ev.params.length > PMAX) out.landscape = `The loss landscape handles up to ${PMAX} parameters; this net has ${ev.params.length}.`;
      }
      if (!fitsIn(net, d)) out.space = out.simplex = `${dsName} has ${d.dim} inputs and the net ${sh.inputs}: pick a dataset that fits (Train panel).`;
    }
    if (!isSimplexNet(net)) out.simplex = 'The simplex needs a softmax output with exactly 3 classes: try the classifier, softmax_reg or funnel preset.';
    return out;
  }

  // ---------------------------------------------------------------- the training path
  // Recorded from the Train panel's 'train' events (presenter only), whatever the panel shows:
  // { steps, theta, loss }, at most every TRAIL_MS while running, TRAIL_MAX points (halved when full).
  let trail = [], trailSig = '', trailVer = 0, lastRec = 0, basisStale = false;
  function recordTrail(p) {
    if (ro) return;
    const net = store.net, ev = evaluator();
    if (!ev || ev.params.length > PMAX || ev.params.length < 1) return;
    if (trailSig !== ev.key) { trail = []; trailSig = ev.key; trailVer++; }
    const steps = readSettings(net, model).steps;
    let last = trail[trail.length - 1];
    if (last && steps < last.steps) {
      trail = steps === 0 ? [] : trail.filter(q => q.steps <= steps);
      trailVer++;
      if (steps === 0) basisStale = true;   // Reset: new weights, a plane through them
      last = trail[trail.length - 1];
    }
    const theta = readParams(net, ev.params);
    if (last && last.steps === steps) {
      // the same step count with other weights: Reset at step 0 again (a new plane), else a hand edit
      if (steps === 0 && theta.some((v, k) => v !== last.theta[k])) {
        trail = [];
        basisStale = true;
        trailVer++;
      } else return;
    }
    const now = performance.now();
    if (p?.running && trail.length && now - lastRec < TRAIL_MS) return;
    trail.push({ steps, theta, loss: ev.evalAt(theta) });
    lastRec = now;
    if (trail.length > TRAIL_MAX) trail = trail.filter((_, i) => i % 2 === 0 || i === trail.length - 1);
    trailVer++;
    dataDirty = true;
    kick();
  }
  store.on('train', recordTrail);

  // ---------------------------------------------------------------- the landscape's plane (presenter)
  let basisSeq = Math.floor(Math.random() * 1e6) * 1000;
  const paramIndex = (ev, key) => (key ? ev.params.findIndex(p => p.key === key) : -1);
  function defaultPair(ev) {
    const P = ev.params.length, sel = store.state.sel;
    let a = -1;
    if (sel?.kind === 'edge') {
      const e = model.edge(store.net, sel.id);
      a = ev.params.findIndex(p => p.kind === 'w' && (p.ids.includes(sel.id) || (e && isTie(e.tie) && p.tie === e.tie)));
    } else if (sel?.kind === 'node') a = ev.params.findIndex(p => p.kind === 'b' && p.ids.includes(sel.id));
    if (a < 0) a = 0;
    return [a, (a + 1) % P];
  }
  const dirsOf = (s, ev) => s.dirs || (ev.params.length <= 2 ? 'weights' : 'random');
  function autoSpan(kind, t0, d1, d2) {
    let ext = 0;
    const theta = readParams(store.net, evaluator().params);
    for (const th of [...trail.map(q => q.theta), theta]) {
      const delta = Float64Array.from(th, (v, k) => v - t0[k]);
      const { a, b } = project(delta, d1, d2);
      ext = Math.max(ext, Math.abs(a), Math.abs(b));
    }
    const base = kind === 'pca' ? 0 : 1;
    return niceUp(Math.max(base, ext * 1.15) || 1);
  }
  function makeBasis(s, ev) {
    const net = store.net, params = ev.params, P = params.length;
    const theta = readParams(net, params);
    const dirs = dirsOf(s, ev);
    let kind = dirs, d1, d2, share = null, note = '', wa = null, wb = null;
    if (dirs === 'weights') {
      let [a, b] = defaultPair(ev);
      const ia = paramIndex(ev, s.wa), ib = paramIndex(ev, s.wb);
      if (ia >= 0) a = ia;
      if (ib >= 0 && ib !== a) b = ib;
      else if (b === a) b = (a + 1) % P;
      d1 = new Float64Array(P); d1[a] = 1;
      d2 = new Float64Array(P); d2[b] = 1;
      wa = params[a].key; wb = params[b].key;
    } else if (dirs === 'pca') {
      const rows = trail.length > 100 ? trail.filter((_, i) => i % Math.ceil(trail.length / 100) === 0).map(q => q.theta) : trail.map(q => q.theta);
      const r = rows.length >= 3 ? pathDirections([...rows, theta]) : null;
      if (r) { d1 = r.d1; d2 = r.d2; share = round(r.share, 3); }
      else { kind = 'random'; note = 'PCA needs a training path: press Play, then Re-center.'; }
    }
    if (kind === 'random') [d1, d2] = randomDirections(params, theta, model.rng(s.seed || 1));
    const rnd = v => Array.from(v, x => round(x));
    const t0 = rnd(theta), r1 = rnd(d1), r2 = rnd(d2);
    const span = s.range > 0 ? s.range : autoSpan(kind, t0, r1, r2);
    return { id: ++basisSeq, sig: ev.psig, data: ev.dataKey, kind, wa, wb, theta0: t0, d1: r1, d2: r2, span, share, note };
  }
  let pubVer = '', lastPub = -1e9;
  function projectTrail(B) {
    const t0 = B.theta0, d1 = B.d1, d2 = B.d2;
    return trail.map(q => {
      const delta = Float64Array.from(q.theta, (v, k) => v - t0[k]);
      const { a, b } = project(delta, d1, d2);
      return [round(a, 5), round(b, 5), round(q.loss, 5)];
    });
  }
  // A new plane through the current weights (patch: settings changed with it).
  function rebase(patch = {}) {
    if (ro) return false;
    const s = cur(), ev = evaluator();
    if (!s) return false;
    const next = { ...s, ...patch };
    if (!ev || ev.params.length < 2) return setS3d(patch);
    basisStale = false;
    const basis = makeBasis(next, ev);
    pubVer = trailVer + ':' + basis.id;
    lastPub = performance.now();
    return setS3d({ ...patch, basis, trail: projectTrail(basis) });
  }
  function clearPath() {
    if (ro) return;
    const ev = evaluator(), net = store.net;
    trail = ev ? [{ steps: readSettings(net, model).steps, theta: readParams(net, ev.params), loss: ev.evalAt(readParams(net, ev.params)) }] : [];
    trailVer++;
    const B = cur()?.basis;
    if (B) { pubVer = trailVer + ':' + B.id; setS3d({ trail: projectTrail(B) }); }
  }

  // ---------------------------------------------------------------- Three.js
  let THREE = null, CSS2DObject = null, GL = null, glError = '', glLoading = null;
  function ensureGL() {
    if (GL || glError) return Promise.resolve(GL);
    if (!glLoading) {
      glLoading = Promise.all([
        import('three'), import('three/addons/controls/OrbitControls.js'), import('three/addons/renderers/CSS2DRenderer.js'),
      ]).then(([T, OC, C2]) => { GL = createGL(T, OC.OrbitControls, C2); })
        .catch(err => { console.error('[nn/surf3d] 3D setup failed:', err); glError = err?.message || String(err); })
        .finally(() => { built = null; dataDirty = true; needRender = true; kick(); });
    }
    return glLoading;
  }
  function createGL(T, Orbit, C2) {
    THREE = T;
    CSS2DObject = C2.CSS2DObject;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.domElement.className = 's3-canvas';
    viewEl.prepend(renderer.domElement);
    const labels = new C2.CSS2DRenderer();
    labels.domElement.className = 's3-labels';
    viewEl.insertBefore(labels.domElement, msgEl);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 200);
    camera.up.set(0, 0, 1);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x4a4a4a, 1.9);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const head = new THREE.DirectionalLight(0xffffff, 1.5);
    head.position.set(0.8, 1.4, 2);
    camera.add(head);
    scene.add(camera);
    const controls = new Orbit(camera, renderer.domElement);
    Object.assign(controls, { enableDamping: true, dampingFactor: 0.14, rotateSpeed: 0.9, zoomSpeed: 0.9, minDistance: 1.2, maxDistance: 40 });
    controls.enabled = !ro;
    controls.addEventListener('change', () => { needRender = true; if (!ro && !applying) camMoved(); kick(); });
    const root = new THREE.Group();
    scene.add(root);
    const g = { renderer, labels, scene, camera, controls, root, sphere: new THREE.SphereGeometry(1, 12, 8), tmp: new THREE.Color() };
    new ResizeObserver(() => resizeGL(g)).observe(viewEl);
    resizeGL(g);
    return g;
  }
  function resizeGL(g) {
    const w = viewEl.clientWidth, h = viewEl.clientHeight;
    if (!w || !h) return;
    g.renderer.setSize(w, h);
    g.labels.setSize(w, h);
    g.camera.aspect = w / h;
    g.camera.updateProjectionMatrix();
    needRender = true;
    kick();
  }

  // ---- camera: the presenter writes state.s3d.cam (throttled); the audience eases toward it
  let applying = false, lastCamSent = 'null', camWanted = null, camTimer = 0, camLast = -1e9;
  function defaultCam(mode) {
    const c = CAM[mode] || CAM.surface, asp = GL?.camera.aspect || 1.35, k = Math.max(1, 1.25 / asp);
    return { p: c.t.map((t, i) => t + c.d[i] * k), t: [...c.t] };
  }
  function camFromState(s) {
    const j = JSON.stringify(s.cam ?? null);
    if (!ro && j === lastCamSent) return;
    lastCamSent = j;
    camWanted = { ...(s.cam || defaultCam(s.mode)), jump: !ro || !s.cam };
  }
  function followCam() {
    if (!camWanted || !GL) return false;
    const { camera, controls } = GL, c = camWanted;
    const P = new THREE.Vector3(...c.p), Tg = new THREE.Vector3(...c.t);
    applying = true;
    if (c.jump) { camera.position.copy(P); controls.target.copy(Tg); }
    else { camera.position.lerp(P, 0.3); controls.target.lerp(Tg, 0.3); }
    camera.lookAt(controls.target);
    if (!ro) controls.update();
    applying = false;
    needRender = true;
    const done = c.jump || (camera.position.distanceTo(P) < 1e-3 && controls.target.distanceTo(Tg) < 1e-3);
    if (done) camWanted = null;
    return !done;
  }
  const r3 = v => [v.x, v.y, v.z].map(x => Math.round(x * 1000) / 1000);
  function sendCam() {
    camTimer = 0;
    camLast = performance.now();
    if (!GL || !cur()) return;
    const c = { p: r3(GL.camera.position), t: r3(GL.controls.target) };
    lastCamSent = JSON.stringify(c);
    setS3d({ cam: c });
  }
  function camMoved() {
    const wait = CAM_MS - (performance.now() - camLast);
    if (wait <= 0) sendCam();
    else if (!camTimer) camTimer = setTimeout(sendCam, wait);
  }
  function home() {
    if (ro) return;
    const s = cur();
    if (!s) return;
    lastCamSent = 'null';
    camWanted = { ...defaultCam(s.mode), jump: true };
    setS3d({ cam: null });
    kick();
  }

  // ---- palette (both themes), rebuilt on a theme change
  let PAL = null;
  function pal() {
    if (PAL) return PAL;
    const cs = getComputedStyle(document.documentElement), th = themeNow();
    const v = (name, d) => cs.getPropertyValue(name).trim() || d;
    const rgb = (c, d) => parseRGB(c) || d;
    const pos = rgb(colorFor(1, 1, th), [74, 163, 255]), neg = rgb(colorFor(-1, 1, th), [255, 122, 89]);
    const light = th === 'light';
    PAL = {
      theme: th, light, bg: v('--bg', light ? '#fbfbf8' : '#1d2327'), fg: v('--ui-fg', light ? '#222' : '#e6e6e0'),
      muted: v('--ui-muted', light ? '#6b7075' : '#9aa3a8'), pos, neg,
      cls: [neg, pos, ...EXTRA.map(c => parseRGB(c))],
      neutral: light ? [206, 211, 215] : [86, 96, 104],
      hi: light ? '#e8a800' : HI,
      grid: light ? '#b9bfc4' : '#56626a', wall: light ? '#d9dde0' : '#3b454b', axis: light ? '#5b6166' : '#9aa3a8',
      wire: light ? '#1d1d1f' : '#ffffff', path: light ? '#1d1d1f' : '#ffffff', gridLine: light ? '#3d4246' : '#c4cbd0',
    };
    return PAL;
  }
  const colorOf = rgb => GL.tmp.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);

  // ---------------------------------------------------------------- scene building blocks
  // built: the current mode's objects. Everything it owns is disposed when the mode, the structure
  // or the theme changes.
  let built = null;
  const texCache = new Map();
  function texHtml(s) {
    s = String(s);
    let h = texCache.get(s);
    if (h == null) {
      if (typeof katex === 'undefined') h = esc(s);
      else { try { h = katex.renderToString(s, { throwOnError: false }); } catch { h = esc(s); } }
      if (texCache.size > 500) texCache.clear();
      texCache.set(s, h);
    }
    return h;
  }
  const mixed = s => String(s).split('$').map((part, i) => (i % 2 ? texHtml(part) : esc(part))).join('');

  function clearBuilt() {
    if (!built) return;
    GL.root.clear();
    for (const d of built.own) d.dispose?.();
    built = null;
  }
  function begin(mode) {
    clearBuilt();
    PAL = null;
    built = { mode, own: [], labels: new Map(), boxKey: '' };
    GL.root.visible = true;
    return built;
  }
  const own = x => { built.own.push(x); return x; };
  const add = (...objs) => { GL.root.add(...objs); return objs[0]; };
  const lineMat = (color, opacity = 1, extra = {}) => own(new THREE.LineBasicMaterial({
    color: new THREE.Color(color), transparent: opacity < 1 || extra.depthTest === false, opacity, ...extra,
  }));
  // Dynamic line segments: set(flat xyz list), 2 points per segment; cap in segments.
  function segments(cap, color, opacity = 1, extra = {}) {
    const geo = own(new THREE.BufferGeometry());
    const arr = new Float32Array(cap * 6);
    geo.setAttribute('position', new THREE.BufferAttribute(arr, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    const obj = add(new THREE.LineSegments(geo, lineMat(color, opacity, extra)));
    obj.frustumCulled = false;
    return {
      obj,
      set(list) {
        const m = Math.min(list.length, arr.length);
        for (let k = 0; k < m; k++) arr[k] = list[k];
        geo.setDrawRange(0, Math.floor(m / 3));
        geo.attributes.position.needsUpdate = true;
      },
    };
  }
  function polyline(cap, color, opacity = 1, extra = {}) {
    const geo = own(new THREE.BufferGeometry());
    const arr = new Float32Array(cap * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(arr, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    const obj = add(new THREE.Line(geo, lineMat(color, opacity, extra)));
    obj.frustumCulled = false;
    return {
      obj,
      set(list) {
        const m = Math.min(list.length, arr.length);
        for (let k = 0; k < m; k++) arr[k] = list[k];
        geo.setDrawRange(0, Math.floor(m / 3));
        geo.attributes.position.needsUpdate = true;
      },
    };
  }
  function marker(color, r = 0.045) {
    const m = add(new THREE.Mesh(GL.sphere, own(new THREE.MeshBasicMaterial({ color: new THREE.Color(color), depthTest: false, transparent: true }))));
    m.scale.setScalar(r);
    m.renderOrder = 10;
    const ring = add(new THREE.Mesh(GL.sphere, own(new THREE.MeshBasicMaterial({ color: new THREE.Color(pal().bg), depthTest: false, transparent: true, side: THREE.BackSide }))));
    ring.scale.setScalar(r * 1.45);
    ring.renderOrder = 9;
    const stem = segments(1, color, 0.8);
    return {
      set(x, y, z, stemZ = null) {
        const ok = [x, y, z].every(Number.isFinite);
        m.visible = ring.visible = ok;
        if (!ok) { stem.set([]); return; }
        m.position.set(x, y, z);
        ring.position.set(x, y, z);
        stem.set(stemZ == null ? [] : [x, y, stemZ, x, y, z]);
      },
    };
  }
  // Instanced spheres for the dataset points.
  function points(cap) {
    const mesh = add(new THREE.InstancedMesh(GL.sphere, own(new THREE.MeshLambertMaterial({ color: 0xffffff })), Math.max(16, cap)));
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.count = 0;
    const arr = mesh.instanceMatrix.array;
    return {
      mesh, cap: Math.max(16, cap),
      place(i, x, y, z, r) {
        const o = 16 * i;
        arr[o] = r; arr[o + 1] = 0; arr[o + 2] = 0; arr[o + 3] = 0;
        arr[o + 4] = 0; arr[o + 5] = r; arr[o + 6] = 0; arr[o + 7] = 0;
        arr[o + 8] = 0; arr[o + 9] = 0; arr[o + 10] = r; arr[o + 11] = 0;
        arr[o + 12] = x; arr[o + 13] = y; arr[o + 14] = z; arr[o + 15] = 1;
      },
      color(i, rgb) { mesh.setColorAt(i, colorOf(rgb)); },
      done(n) {
        mesh.count = n;
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      },
    };
  }
  function pointsFor(n) {
    if (!built.pts || built.pts.cap < n) {
      if (built.pts) built.pts.mesh.removeFromParent();
      built.pts = points(n);
    }
    return built.pts;
  }
  const pointR = n => (n <= 300 ? 0.03 : n <= 1000 ? 0.022 : 0.015);
  function classRGB(d, s) {
    const p = pal();
    if (d.kind === 'class') return p.cls[d.cls[s] % p.cls.length];
    const u = clamp((d.Y[s][0] - d.mid) / d.half, -1, 1);
    const c = u >= 0 ? p.pos : p.neg;
    return p.neutral.map((b, k) => b + (c[k] - b) * (0.25 + 0.75 * Math.abs(u)));
  }

  // Labels in 3D: CSS2D objects keyed so they are reused between refreshes.
  function setLabels(list) {
    const keep = new Set();
    for (const L of list) {
      keep.add(L.key);
      let r = built.labels.get(L.key);
      if (!r) {
        const el = document.createElement('div');
        const obj = new CSS2DObject(el);
        add(obj);
        r = { obj, el, html: null, cls: null };
        built.labels.set(L.key, r);
      }
      const html = L.tex != null ? texHtml(L.tex) : esc(L.text ?? '');
      if (r.html !== html) { r.el.innerHTML = html; r.html = html; }
      const cls = `s3-lab ${L.cls || ''}`;
      if (r.cls !== cls) { r.el.className = cls; r.cls = cls; }
      r.obj.position.set(L.p[0], L.p[1], L.p[2]);
      if (L.anchor) r.obj.center.set(L.anchor[0], L.anchor[1]); else r.obj.center.set(0.5, 0.5);
      r.obj.visible = !L.hidden;
    }
    for (const [k, r] of built.labels) if (!keep.has(k)) { r.obj.removeFromParent(); built.labels.delete(k); }
  }

  // The floor box of the surface and the landscape: a grid on z = 0 at the x and y ticks, two back
  // walls with lines at the z ticks, and the tick values and axis titles. spec.zmap(v) -> world z;
  // spec.zticks: values; *t: KaTeX titles.
  function floorBox(spec) {
    const key = JSON.stringify([spec.xr, spec.yr, spec.zticks, spec.xt, spec.yt, spec.zt, spec.zr]);
    const labels = [];
    const xs = niceTicks(spec.xr[0], spec.xr[1], 4), ys = niceTicks(spec.yr[0], spec.yr[1], 4);
    const X = v => -1 + 2 * (v - spec.xr[0]) / (spec.xr[1] - spec.xr[0]);
    const Y = v => -1 + 2 * (v - spec.yr[0]) / (spec.yr[1] - spec.yr[0]);
    if (built.boxKey !== key) {
      built.boxKey = key;
      const g = [], w = [];
      for (const x of xs) { const u = X(x); if (u > -0.999 && u < 0.999) g.push(u, -1, 0, u, 1, 0); }
      for (const y of ys) { const u = Y(y); if (u > -0.999 && u < 0.999) g.push(-1, u, 0, 1, u, 0); }
      g.push(-1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 0, -1, 1, 0, -1, -1, 0);
      for (const z of spec.zticks) {
        const h = spec.zmap(z);
        if (h < 0.001 || h > ZH + 0.001) continue;
        w.push(-1, 1, h, 1, 1, h, -1, -1, h, -1, 1, h);
      }
      w.push(-1, 1, 0, -1, 1, ZH, 1, 1, 0, 1, 1, ZH, -1, -1, 0, -1, -1, ZH, -1, 1, ZH, 1, 1, ZH, -1, -1, ZH, -1, 1, ZH);
      built.box.grid.set(g);
      built.box.wall.set(w);
    }
    xs.forEach((x, k) => { const u = X(x); if (u >= -1.001 && u <= 1.001) labels.push({ key: 'x' + k, p: [u, -1.12, 0], text: tickText(x), cls: 'tick' }); });
    ys.forEach((y, k) => { const u = Y(y); if (u >= -1.001 && u <= 1.001) labels.push({ key: 'y' + k, p: [1.12, u, 0], text: tickText(y), cls: 'tick' }); });
    spec.zticks.forEach((z, k) => { const h = spec.zmap(z); if (h >= -0.001 && h <= ZH + 0.001) labels.push({ key: 'z' + k, p: [-1.1, -1.1, h], text: spec.ztext ? spec.ztext(z) : tickText(z), cls: 'tick' }); });
    labels.push({ key: 'xt', p: [0, -1.38, 0], tex: spec.xt, cls: 'ttl' });
    labels.push({ key: 'yt', p: [1.4, 0, 0], tex: spec.yt, cls: 'ttl' });
    labels.push({ key: 'zt', p: [-1.06, -1.06, ZH + 0.14], tex: spec.zt, cls: 'ttl', anchor: [0.15, 1] });
    return labels;
  }
  function floorParts() {
    const p = pal();
    built.box = { grid: segments(40, p.grid, 0.75), wall: segments(40, p.wall, 0.9) };
  }

  // ---------------------------------------------------------------- captions and messages
  let capKey = '';
  function setCaption(tex, note = '') {
    const key = tex + '\u0000' + note;
    if (key === capKey) return;
    capKey = key;
    capTex.innerHTML = mixed(tex);
    capNote.textContent = note;
    capNote.hidden = !note;
  }
  function showMsg(t) {
    msgEl.textContent = t;
    msgEl.hidden = false;
    if (GL) GL.root.visible = false;
    setCaption('', '');
  }
  function hideMsg() {
    msgEl.hidden = true;
    if (GL) GL.root.visible = true;
  }

  // ---------------------------------------------------------------- surface
  const gridC = { key: '', X: null };
  function gridFor(dom) {
    const key = [dom.x0, dom.x1, dom.y0, dom.y1].join(',');
    if (gridC.key === key) return gridC.X;
    const G = SURF_G, X = new Float64Array(G * G * 2);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const s = j * G + i;
      X[2 * s] = dom.x0 + (dom.x1 - dom.x0) * i / (G - 1);
      X[2 * s + 1] = dom.y0 + (dom.y1 - dom.y0) * j / (G - 1);
    }
    gridC.key = key;
    gridC.X = X;
    return X;
  }
  function fallbackDom(net) {
    const ins = model.nodesIn(net, 0).map(q => Math.abs(Number(q.value) || 0));
    const h = Math.max(1.5, ...ins.map(v => v + 0.5));
    return { x0: -h, x1: h, y0: -h, y1: h };
  }
  function resolveNeuron(net, s) {
    const L = net.layers.length, outs = outNodes(net);
    let id = null;
    if (s.neuron === 'sel') {
      const sel = store.state.sel;
      if (sel?.kind === 'node') id = sel.id;
      else if (sel?.kind === 'edge') id = model.edge(net, sel.id)?.to ?? null;
      else if (sel?.kind === 'layer') { const l = model.layerIndex(net, sel.id); if (l >= 0) id = model.nodesIn(net, l)[0]?.id ?? null; }
    } else if (s.neuron && s.neuron !== 'out') id = s.neuron;
    let l = id != null && model.node(net, id) ? model.nodeLayerIndex(net, id) : -1;
    if (l < 0) {
      if (outs.length > 1) return { argmax: true, l: L - 1, R: outs.length };
      id = outs[0]?.id;
      l = L - 1;
    }
    const ns = model.nodesIn(net, l), i = ns.findIndex(q => q.id === id);
    const attn = net.layers[l].kind === 'attention';
    return { argmax: false, l, i, id, R: ns.length, pre: !!s.pre && l > 0 && !attn, attn };
  }
  function buildSurface() {
    const b = begin('surface'), p = pal(), G = SURF_G;
    const geo = own(new THREE.BufferGeometry());
    const pos = new Float32Array(G * G * 3), col = new Float32Array(G * G * 3);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const k = j * G + i;
      pos[3 * k] = -1 + 2 * i / (G - 1);
      pos[3 * k + 1] = -1 + 2 * j / (G - 1);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage));
    const idx = [], wire = [];
    for (let j = 0; j < G - 1; j++) for (let i = 0; i < G - 1; i++) {
      const a = j * G + i;
      idx.push(a, a + 1, a + G + 1, a, a + G + 1, a + G);
    }
    for (let j = 0; j < G; j += 4) for (let i = 0; i < G - 1; i++) wire.push(j * G + i, j * G + i + 1);
    for (let i = 0; i < G; i += 4) for (let j = 0; j < G - 1; j++) wire.push(j * G + i, (j + 1) * G + i);
    geo.setIndex(idx);
    b.mesh = add(new THREE.Mesh(geo, own(new THREE.MeshLambertMaterial({
      vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    }))));
    b.mesh.frustumCulled = false;
    const wg = own(new THREE.BufferGeometry());
    wg.setAttribute('position', geo.getAttribute('position'));
    wg.setIndex(wire);
    b.wire = add(new THREE.LineSegments(wg, lineMat(p.wire, p.light ? 0.22 : 0.2, { depthWrite: false })));
    b.wire.frustumCulled = false;
    b.plane = add(new THREE.Mesh(own(new THREE.PlaneGeometry(2, 2)), own(new THREE.MeshBasicMaterial({
      color: new THREE.Color(p.muted), transparent: true, opacity: p.light ? 0.16 : 0.14, side: THREE.DoubleSide, depthWrite: false,
    }))));
    b.planeEdge = segments(4, p.muted, 0.7);
    b.contour = segments(4 * G * G, p.hi, 1);
    floorParts();
    b.mark = marker(p.hi);
  }
  function refreshSurface(s) {
    const net = store.net, L = net.layers.length, b = built, p = pal(), G = SURF_G, n = G * G;
    const d = getData(), dataOk = !!d && d.dim === 2 && d.kind !== 'seq' && !!d.dom && fitsIn(net, d);
    const dom = dataOk ? d.dom : fallbackDom(net);
    const gx = gridFor(dom);
    const M = model.matrices(net);
    const F = forwardZA(net, model, gx, n, M);
    if (!F) { showMsg('Could not run the net over the input plane.'); return; }
    hideMsg();
    const tg = resolveNeuron(net, s);
    const vals = new Float64Array(n);
    let win = null, marg = null;
    if (tg.argmax) {
      const A = F.a[L - 1], R = tg.R;
      win = new Int32Array(n); marg = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        let b1 = 0, b2 = -1;
        for (let c = 1; c < R; c++) {
          if (A[k * R + c] > A[k * R + b1]) { b2 = b1; b1 = c; } else if (b2 < 0 || A[k * R + c] > A[k * R + b2]) b2 = c;
        }
        vals[k] = A[k * R + b1]; win[k] = b1; marg[k] = A[k * R + b1] - (b2 >= 0 ? A[k * R + b2] : 0);
      }
    } else {
      const src = tg.pre ? F.z[tg.l] : F.a[tg.l];
      for (let k = 0; k < n; k++) vals[k] = src ? src[k * tg.R + tg.i] : NaN;
    }
    const act = tg.argmax ? 'softmax' : tg.pre || tg.l === 0 || tg.attn ? 'identity' : net.layers[tg.l].act;
    const isOut = !tg.argmax && tg.l === L - 1 && !tg.pre && dataOk && d.K === tg.R;
    const Fd = dataOk && !isOut && !tg.argmax ? forwardZA(net, model, d.Xf, d.n, M) : null;
    // the height range: an activation's bounds, else the values (and targets), 0 included, eased while training
    let lo, hi;
    const bd = BOUNDED[act];
    if (bd) [lo, hi] = bd;
    else {
      let mn = Infinity, mx = -Infinity;
      for (const v of vals) if (Number.isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
      if (isOut) for (const y of d.Y) if (Number.isFinite(y[tg.i])) { mn = Math.min(mn, y[tg.i]); mx = Math.max(mx, y[tg.i]); }
      if (!Number.isFinite(mn)) { mn = -1; mx = 1; }
      if (tg.pre || act === 'relu' || act === 'leaky' || act === 'identity') { mn = Math.min(mn, 0); mx = Math.max(mx, 0); }
      if (!(mx - mn > 1e-9)) { mn -= 0.5; mx += 0.5; }
      const pad = (mx - mn) * 0.04;
      mn -= pad; mx += pad;
      const key = `${tg.id}|${tg.pre}|${dom.x0}|${dom.x1}`;
      if (!b.range || b.range.key !== key || !busy()) b.range = { key, lo: mn, hi: mx };
      else {
        const r = b.range, k = 0.1;   // grow at once, shrink slowly: no jitter while training
        r.lo = mn < r.lo ? mn : r.lo + (mn - r.lo) * k;
        r.hi = mx > r.hi ? mx : r.hi + (mx - r.hi) * k;
      }
      ({ lo, hi } = b.range);
    }
    const Z = v => ZH * clamp((v - lo) / (hi - lo), 0, 1);
    const X = v => -1 + 2 * (v - dom.x0) / (dom.x1 - dom.x0), Y = v => -1 + 2 * (v - dom.y0) / (dom.y1 - dom.y0);
    const pa = b.mesh.geometry.attributes.position, ca = b.mesh.geometry.attributes.color;
    const c0 = bd ? (bd[0] + bd[1]) / 2 : 0, h0 = bd ? (bd[1] - bd[0]) / 2 : Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    for (let k = 0; k < n; k++) {
      const v = vals[k];
      pa.array[3 * k + 2] = Number.isFinite(v) ? Z(v) : 0;
      if (!Number.isFinite(v)) { mixInto(ca.array, 3 * k, p.neutral, p.neutral, 0); continue; }
      if (tg.argmax) mixInto(ca.array, 3 * k, p.neutral, p.cls[win[k] % p.cls.length], 0.25 + 0.65 * clamp(marg[k], 0, 1));
      else {
        const u = clamp((v - c0) / h0, -1, 1);
        mixInto(ca.array, 3 * k, p.neutral, u >= 0 ? p.pos : p.neg, 0.1 + 0.75 * Math.abs(u));
      }
    }
    pa.needsUpdate = ca.needsUpdate = true;
    b.mesh.geometry.computeVertexNormals();
    // the level plane: the decision boundary (sigmoid 0.5, tanh 0) or z = 0, and where the surface crosses it
    const level = tg.argmax ? null : tg.pre ? 0 : act === 'tanh' ? 0 : act === 'sigmoid' ? 0.5 : null;
    const showLevel = level != null && level > lo && level < hi;
    b.plane.visible = showLevel;
    if (showLevel) {
      const zl = Z(level);
      b.plane.position.z = zl;
      b.planeEdge.set([-1, -1, zl, 1, -1, zl, 1, -1, zl, 1, 1, zl, 1, 1, zl, -1, 1, zl, -1, 1, zl, -1, -1, zl]);
      const uv = isoLines(vals, G, level), w = [];
      for (let k = 0; k < uv.length; k += 2) w.push(-1 + 2 * uv[k] / (G - 1), -1 + 2 * uv[k + 1] / (G - 1), zl + 0.006);
      b.contour.set(w);
    } else { b.planeEdge.set([]); b.contour.set([]); }
    // the training points: at their targets for the output, else at this neuron's value; the winning
    // class view keeps them on the floor
    if (dataOk) {
      const P = pointsFor(d.n), r = pointR(d.n), src = Fd && (tg.pre ? Fd.z[tg.l] : Fd.a[tg.l]);
      for (let k = 0; k < d.n; k++) {
        const x = d.X[k][0], y = d.X[k][1];
        const h = tg.argmax ? r : isOut ? Z(d.Y[k][tg.i]) : src ? Z(src[k * tg.R + tg.i]) : 0;
        P.place(k, X(x), Y(y), h, r);
        P.color(k, classRGB(d, k));
      }
      P.done(d.n);
      P.mesh.visible = true;
    } else if (built.pts) built.pts.mesh.visible = false;
    // the net's current input
    const ins = model.nodesIn(net, 0), fw = store.state.fwd;
    const cx = Number(ins[0]?.value), cy = Number(ins[1]?.value);
    let cv = NaN;
    if (fw) {
      if (tg.argmax) cv = Math.max(...outNodes(net).map(q => fw.node?.[q.id]?.a ?? -Infinity));
      else if (tg.l === 0) cv = Number(model.node(net, tg.id)?.value);
      else cv = fw.node?.[tg.id]?.[tg.pre ? 'z' : 'a'];
    }
    const inside = cx >= dom.x0 && cx <= dom.x1 && cy >= dom.y0 && cy <= dom.y1;
    if (inside && Number.isFinite(cv)) b.mark.set(X(cx), Y(cy), Z(cv), 0); else b.mark.set(NaN, 0, 0);
    // axes and caption
    const zt = tg.argmax ? '\\max_k \\hat y_k' : tg.pre ? `z^{(${tg.l})}_{${tg.i + 1}}` : nodeTex(net, tg.id);
    const labels = floorBox({
      xr: [dom.x0, dom.x1], yr: [dom.y0, dom.y1], zr: [lo, hi], zticks: niceTicks(lo, hi, 4), zmap: Z,
      xt: nodeTex(net, ins[0]?.id), yt: nodeTex(net, ins[1]?.id), zt,
    });
    setLabels(labels);
    surfaceCaption(net, M, tg, act, level, showLevel, dataOk, isOut);
    probeData.surface = {
      G, dom, lo, hi, zh: ZH, neuron: tg.argmax ? { argmax: true } : { id: tg.id, l: tg.l, i: tg.i, pre: tg.pre },
      xs: Array.from({ length: G }, (_, i) => gx[2 * i]), ys: Array.from({ length: G }, (_, j) => gx[2 * j * G + 1]),
      values: Array.from(vals), meshZ: Array.from({ length: n }, (_, k) => pa.array[3 * k + 2]),
    };
  }
  // z of neuron i of layer l as numbers, when it reads only the inputs: 0.83 x₁ − 1.20 x₂ + 0.10
  function zFormula(net, M, l, i) {
    const m = M[l - 1];
    if (!m || m.kind !== 'dense' || m.terms.length !== 1 || m.terms[0].k !== 0) return null;
    const t = m.terms[0];
    let s = '';
    const term = (v, body) => { s += (s ? (v < 0 ? ' - ' : ' + ') : v < 0 ? '-' : '') + (body ? `${fmtK(Math.abs(v))}\\,${body}` : fmtK(Math.abs(v))); };
    t.cols.forEach((id, j) => { if (t.edge[i][j] != null) term(t.W[i][j], nodeTex(net, id)); });
    const bias = m.b[i] || 0;
    if (bias || !s) term(bias, '');
    return s;
  }
  function surfaceCaption(net, M, tg, act, level, showLevel, dataOk, isOut) {
    const L = net.layers.length;
    const follow = cur()?.neuron === 'sel' && !store.state.sel ? ' Click a neuron to see its surface.' : '';
    if (tg.argmax) {
      setCaption('Colour: the class the net picks, $\\arg\\max_k \\hat y_k$; height: how sure it is, $\\max_k \\hat y_k$, over the input plane.',
        (dataOk ? 'Dots: the training points on the floor, in their class colour.' : '') + follow);
      return;
    }
    const name = nodeTex(net, tg.id), f = FN[act];
    const lvl = showLevel ? (tg.pre ? ' The flat plane is $z = 0$, where it changes sign.' : act === 'sigmoid' ? ' The flat plane is 0.5' + (tg.l === L - 1 ? ', the decision boundary.' : ', its midpoint.') : ' The flat plane is 0' + (tg.l === L - 1 ? ', the decision boundary.' : '.')) : '';
    let tex;
    if (tg.l === 0) tex = `$${name}$ itself: an input is a tilted plane over the input plane.`;
    else if (tg.attn) tex = `$${name}$ over the input plane: an attention output.`;
    else {
      const z = zFormula(net, M, tg.l, tg.i);
      const zt = `z^{(${tg.l})}_{${tg.i + 1}}`;
      if (tg.pre) tex = z ? `$${zt} = ${z}$ is a tilted plane over $(x_1, x_2)$.` : `$${zt}$, the neuron's input sum, over the input plane: layer ${tg.l} adds up layer ${tg.l - 1}'s bent surfaces.`;
      else if (z && f) tex = `$${name} = ${f}(${z})$: the plane $${zt}$, bent by $${f}$.`;
      else if (z) tex = `$${name} = ${z}$: a plane over the input plane.`;
      else if (f) tex = `$${name} = ${f}(\\textstyle\\sum_j w_j a_j + b)$ over the input plane: layer ${tg.l} combines layer ${tg.l - 1}'s bent surfaces.`;
      else tex = `$${name} = \\textstyle\\sum_j w_j a_j + b$ over the input plane: a sum of layer ${tg.l - 1}'s surfaces.`;
    }
    const pts = dataOk ? (isOut ? ' Dots: the training points at their targets.' : ' Dots: the training points at this neuron\'s value.') : '';
    setCaption(tex + lvl, (pts + follow).trim());
  }

  // ---------------------------------------------------------------- landscape
  let job = null, grid = null, gridVer = 0;
  const landKey = (B, ev) => [ev.key, B.id, B.span].join('|');
  function startJob(key, B, ev, G) {
    job = {
      key, G, k: 0, ev, vals: new Float64Array(G * G), span: B.span,
      t0: Float64Array.from(B.theta0), d1: Float64Array.from(B.d1), d2: Float64Array.from(B.d2), vec: new Float64Array(B.theta0.length),
    };
  }
  function stepJob(budget) {
    const t = performance.now();
    while (job) {
      const { G, vals, ev, span, t0, d1, d2, vec } = job, P = vec.length;
      while (job.k < G * G) {
        const i = job.k % G, j = Math.floor(job.k / G);
        const a = span * (2 * i / (G - 1) - 1), b = span * (2 * j / (G - 1) - 1);
        for (let q = 0; q < P; q++) vec[q] = t0[q] + a * d1[q] + b * d2[q];
        vals[job.k++] = ev.evalAt(vec);
        if (performance.now() - t > budget) return;
      }
      grid = { key: job.key, G, vals, span, preview: G < LAND_G, ver: ++gridVer };
      if (G < LAND_G) startJob(job.key, { theta0: t0, d1, d2, span }, ev, LAND_G);
      else job = null;
    }
  }
  function heightMap(vals, extra, log) {
    const f = Array.from(vals).filter(Number.isFinite).sort((a, b) => a - b);
    if (!f.length) return null;
    const lo = f[0], mx = f[f.length - 1], q = f[Math.floor(0.85 * (f.length - 1))];
    let hi = Math.min(mx, lo + (Math.max(q, extra) - lo) * 1.3);
    if (!(hi > lo)) hi = lo + (Math.abs(lo) || 1) * 1e-3;
    const g = log ? v => Math.log10(Math.max(v, 1e-12)) : v => v;
    const glo = g(lo), ghi = g(hi) > glo ? g(hi) : glo + 1e-9;
    const u = v => clamp((g(v) - glo) / (ghi - glo), 0, 1);
    return { lo, hi, max: mx, log, clipped: mx > hi * (1 + 1e-6) + 1e-12, u, z: v => ZH * u(v), g, glo, ghi };
  }
  function buildLandscape() {
    const b = begin('landscape'), p = pal();
    b.meshG = 0;
    b.contour = segments(24000, p.wire, p.light ? 0.35 : 0.3, { depthWrite: false });
    b.floorC = segments(24000, p.grid, 0.55, { depthWrite: false });
    b.shadow = polyline(TRAIL_MAX + 2, p.path, 0.4);
    b.cross = segments(2, p.axis, 0.9);
    b.cross.set([-0.06, 0, 0.003, 0.06, 0, 0.003, 0, -0.06, 0.003, 0, 0.06, 0.003]);
    floorParts();
    b.mark = marker(p.hi, 0.042);
    b.tube = null;
    b.tubeKey = '';
    b.meshKey = '';
  }
  function landMesh(G) {
    const b = built;
    if (b.meshG === G) return;
    if (b.mesh) { b.mesh.removeFromParent(); b.wire.removeFromParent(); b.mesh.geometry.dispose(); b.wire.geometry.dispose(); }
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(G * G * 3), col = new Float32Array(G * G * 3);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const k = j * G + i;
      pos[3 * k] = -1 + 2 * i / (G - 1);
      pos[3 * k + 1] = -1 + 2 * j / (G - 1);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const idx = [], wire = [];
    for (let j = 0; j < G - 1; j++) for (let i = 0; i < G - 1; i++) { const a = j * G + i; idx.push(a, a + 1, a + G + 1, a, a + G + 1, a + G); }
    const step = G > 20 ? 3 : 1;
    for (let j = 0; j < G; j += step) for (let i = 0; i < G - 1; i++) wire.push(j * G + i, j * G + i + 1);
    for (let i = 0; i < G; i += step) for (let j = 0; j < G - 1; j++) wire.push(j * G + i, (j + 1) * G + i);
    geo.setIndex(idx);
    if (!b.surfMat) b.surfMat = own(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    if (!b.wireMat) b.wireMat = lineMat(pal().wire, pal().light ? 0.14 : 0.12, { depthWrite: false });
    b.mesh = add(new THREE.Mesh(geo, b.surfMat));
    b.mesh.frustumCulled = false;
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', geo.getAttribute('position'));
    wg.setIndex(wire);
    b.wire = add(new THREE.LineSegments(wg, b.wireMat));
    b.wire.frustumCulled = false;
    own(geo); own(wg);
    b.meshG = G;
  }
  let landView = null;   // what the last landscape drawing used: for the probe and the caption
  function refreshLandscape(s) {
    const net = store.net, b = built, p = pal();
    const ev = evaluator();
    if (!ev) { showMsg('Could not evaluate the loss for this net.'); return; }
    if (!ro && (basisStale || !s.basis || s.basis.sig !== ev.psig || s.basis.data !== ev.dataKey)) {
      if (rebase()) return;   // the new s3d event refreshes again
    }
    const B = s.basis;
    if (!B || B.sig !== ev.psig || B.data !== ev.dataKey) { showMsg('Waiting for the presenter\'s plane…'); return; }
    hideMsg();
    // publish the projected path (presenter, throttled)
    const now = performance.now();
    if (!ro && pubVer !== trailVer + ':' + B.id) {
      if (now - lastPub >= PATH_MS) { pubVer = trailVer + ':' + B.id; lastPub = now; if (setS3d({ trail: projectTrail(B) })) return; }
      else { dataDirty = true; }
    }
    const key = landKey(B, ev);
    if ((!grid || grid.key !== key) && (!job || job.key !== key)) startJob(key, B, ev, LAND_G0);
    const T = Array.isArray(s.trail) ? s.trail : [];
    // the current weights, projected, at their true loss
    const theta = readParams(net, ev.params);
    const t0 = B.theta0, delta = Float64Array.from(theta, (v, k) => v - t0[k]);
    const pr = project(delta, B.d1, B.d2), loss = ev.evalAt(theta);
    const ext = Math.max(Math.abs(pr.a), Math.abs(pr.b), ...T.map(q => Math.max(Math.abs(q[0]), Math.abs(q[1]))));
    if (!ro && !(s.range > 0) && ext > 0.98 * B.span) {
      if (setS3d({ basis: { ...B, span: niceUp(ext * 1.3) } })) return;
    }
    const G0 = grid && grid.key === key ? grid : null;
    const trailMax = T.reduce((m, q) => (Number.isFinite(q[2]) ? Math.max(m, q[2]) : m), -Infinity);
    const hm = G0 ? heightMap(G0.vals, Math.max(trailMax, loss), !!s.log) : null;
    const mkey = G0 ? [G0.ver, s.log, hm && hm.hi].join('|') : '';
    if (G0 && hm && b.meshKey !== mkey) {
      b.meshKey = mkey;
      landMesh(G0.G);
      const G = G0.G, pa = b.mesh.geometry.attributes.position, ca = b.mesh.geometry.attributes.color;
      const U = new Float64Array(G * G);
      for (let k = 0; k < G * G; k++) {
        const v = G0.vals[k];
        const u = Number.isFinite(v) ? hm.u(v) : 1;
        U[k] = u;
        pa.array[3 * k + 2] = ZH * u;
        mixInto(ca.array, 3 * k, [0, 0, 0], viridis(u), 1);
      }
      pa.needsUpdate = ca.needsUpdate = true;
      b.mesh.geometry.computeVertexNormals();
      const on = [], fl = [];
      for (let c = 1; c <= 8; c++) {
        const lev = c / 9, uv = isoLines(U, G, lev);
        for (let k = 0; k < uv.length; k += 2) {
          const x = -1 + 2 * uv[k] / (G - 1), y = -1 + 2 * uv[k + 1] / (G - 1);
          on.push(x, y, ZH * lev + 0.004);
          fl.push(x, y, 0.002);
        }
      }
      b.contour.set(on);
      b.floorC.set(fl);
      b.hm = hm;
    }
    const H = b.hm && G0 ? b.hm : null;
    b.mesh && (b.mesh.visible = b.wire.visible = !!H);
    // the path: a tube at its true loss, a faint copy of it through the surface, its shadow on the floor
    const W = v => clamp(v / B.span, -1.05, 1.05);
    if (H) {
      const pts = T.filter(q => q.every(Number.isFinite)).map(q => [W(q[0]), W(q[1]), H.z(q[2])]);
      const tkey = JSON.stringify(pts.length) + '|' + (pts.length ? pts[pts.length - 1].join(',') + pts[0].join(',') : '') + '|' + b.meshKey;
      if (tkey !== b.tubeKey) {
        b.tubeKey = tkey;
        if (b.tube) { b.tube.removeFromParent(); b.xray.removeFromParent(); b.tube.geometry.dispose(); b.tube = b.xray = null; }
        if (pts.length >= 2) {
          const curve = new THREE.CatmullRomCurve3(pts.map(q => new THREE.Vector3(...q)), false, 'centripetal');
          const tg = new THREE.TubeGeometry(curve, Math.min(800, pts.length * 4), 0.013, 6, false);
          if (!b.tubeMat) b.tubeMat = own(new THREE.MeshLambertMaterial({ color: new THREE.Color(p.path) }));
          b.tube = add(new THREE.Mesh(tg, b.tubeMat));
          b.tube.frustumCulled = false;
          // the same tube seen through the surface, fainter: where the path runs below the slice
          if (!b.xrayMat) b.xrayMat = own(new THREE.MeshBasicMaterial({ color: new THREE.Color(p.path), transparent: true, opacity: 0.32, depthTest: false, depthWrite: false }));
          b.xray = add(new THREE.Mesh(tg, b.xrayMat));
          b.xray.frustumCulled = false;
          b.xray.renderOrder = 4;
        }
        b.shadow.set(pts.flatMap(q => [q[0], q[1], 0.003]));
      }
      b.mark.set(W(pr.a), W(pr.b), H.z(loss), 0);
    } else {
      b.mark.set(NaN, 0, 0);
      b.shadow.set([]);
    }
    // axes
    const kind = B.kind, P = ev.params;
    const ia = kind === 'weights' ? P.findIndex(q => q.key === B.wa) : -1, ib = kind === 'weights' ? P.findIndex(q => q.key === B.wb) : -1;
    const xa = ia >= 0 ? B.theta0[ia] : 0, yb = ib >= 0 ? B.theta0[ib] : 0;
    const zticks = H ? (H.log ? niceTicks(H.glo, H.ghi, 4).map(t => 10 ** t) : niceTicks(H.lo, H.hi, 4)) : [];
    const labels = floorBox({
      xr: [xa - B.span, xa + B.span], yr: [yb - B.span, yb + B.span], zr: H ? [H.lo, H.hi, H.log] : [0, 1],
      zticks, zmap: H ? H.z : () => -1, ztext: H?.log ? v => String(+v.toPrecision(2)) : null,
      xt: kind === 'weights' ? paramTex(net, P[ia]) : '\\alpha \\;(\\delta_1)', yt: kind === 'weights' ? paramTex(net, P[ib]) : '\\beta \\;(\\delta_2)',
      zt: s.log ? '\\log_{10} L' : 'L',
    });
    labels.push({ key: 'th0', p: [0, 0, -0.05], tex: '\\theta_0', cls: 'tick th' });
    setLabels(labels);
    // caption
    const lossName = net.meta?.loss === 'xent' ? 'cross-entropy' : 'MSE';
    const sub = ev.n < ev.total ? ` on ${ev.n} of the ${ev.total} points` : '';
    let tex;
    if (kind === 'weights') tex = `$L$ (${lossName}${sub}) as $${paramTex(net, P[ia])}$ and $${paramTex(net, P[ib])}$ move, every other parameter held at $\\theta_0$.`;
    else if (kind === 'pca') tex = `$L(\\theta_0 + \\alpha\\,\\delta_1 + \\beta\\,\\delta_2)$ (${lossName}${sub}): $\\delta_1, \\delta_2$ are the path's top two principal directions${B.share != null ? ` (${Math.round(B.share * 100)}% of its variance)` : ''}.`;
    else tex = `$L(\\theta_0 + \\alpha\\,\\delta_1 + \\beta\\,\\delta_2)$ (${lossName}${sub}) on a random plane through the weights $\\theta_0$; each neuron's share of $\\delta_1, \\delta_2$ has the norm of its own weights (filter-normalised).`;
    const off = pr.r > 1e-9 * (1 + Math.sqrt(dot(theta, theta))) ? `, ${fmt(pr.r)} off the plane` : ', on the plane';
    const notes = [];
    if (B.note) notes.push(B.note);
    notes.push(`● now: L = ${lossText(loss)}${off}.`);
    if (T.length > 1) notes.push(`Path: ${T.length} points of training, at their true loss${!ro && kind !== 'pca' && T.length > 2 && pathOff(pr, B) ? ' (it leaves this plane: try PCA of the path)' : ''}.`);
    else if (!ro) notes.push('Train to draw the path.');
    if (H?.clipped) notes.push(`Heights stop at L = ${lossText(H.hi)} (the top reaches ${lossText(H.max)}).`);
    if (G0?.preview || job) notes.push('Computing…');
    setCaption(tex, notes.join(' '));
    const c = grid && grid.key === key ? grid.vals[(grid.G * grid.G - 1) / 2] : NaN;
    landView = { G: G0?.G ?? 0, span: B.span, kind, center: c, current: { a: pr.a, b: pr.b, r: pr.r, loss }, trail: T.length, done: !!G0 && !G0.preview && !job, lo: H?.lo, hi: H?.hi, n: ev.n, basis: B.id };
    probeData.landscape = landView;
  }
  // The path mostly leaves the plane: the current weights are further from it than they moved in it.
  function pathOff(v, B) {
    return v.r > 2 * Math.hypot(v.a, v.b) && v.r > 0.1 * B.span;
  }
  function paramTex(net, prm) {
    if (!prm) return '?';
    if (prm.tie) return `\\text{${texEsc(prm.tie)}}`;
    if (prm.kind === 'b') return `b\\,(${nodeTex(net, prm.ids[0])})`;
    const e = model.edge(net, prm.ids[0]);
    return e ? `w\\,(${nodeTex(net, e.from)} \\to ${nodeTex(net, e.to)})` : 'w';
  }
  function paramText(net, prm) {
    if (prm.tie) return `${prm.tie} (shared ×${prm.ids.length})`;
    if (prm.kind === 'b') return `b ${plainLabel(nodeTex(net, prm.ids[0]))}`;
    const e = model.edge(net, prm.ids[0]);
    return e ? `w ${plainLabel(nodeTex(net, e.from))} → ${plainLabel(nodeTex(net, e.to))}` : 'w';
  }

  // ---------------------------------------------------------------- space and simplex
  function stagesFor(net, hasZ) {
    const L = net.layers.length, out = [{ l: 0, part: 'a', d: model.nodesIn(net, 0).length }];
    for (let l = 1; l < L; l++) {
      const ly = net.layers[l], d = model.nodesIn(net, l).length;
      const attn = ly.kind === 'attention';
      if (hasZ && !attn && ly.act !== 'identity') out.push({ l, part: 'z', d });
      out.push({ l, part: 'a', d });
    }
    return out;
  }
  const stageTex = (net, S) => (S.l === 0 ? 'x' : S.part === 'z' ? `z^{(${S.l})}` : S.l === net.layers.length - 1 ? '\\hat y' : `h^{(${S.l})}`);
  function defaultStage(net, st) {
    const L = net.layers.length;
    let k = st.findIndex(S => S.part === 'a' && S.l > 0 && S.l < L - 1 && S.d === 3);
    if (k < 0) k = st.findIndex(S => S.part === 'a' && S.l > 0 && S.l < L - 1 && S.d > 3);
    return k >= 0 ? k : st.length - 1;
  }
  function linesFor(dom) {
    const n = 2 * LINES * PER, X = new Float64Array(2 * n);
    let o = 0;
    for (let dir = 0; dir < 2; dir++) for (let i = 0; i < LINES; i++) for (let j = 0; j < PER; j++) {
      const u = i / (LINES - 1), w = j / (PER - 1);
      X[o++] = dom.x0 + (dir ? w : u) * (dom.x1 - dom.x0);
      X[o++] = dom.y0 + (dir ? u : w) * (dom.y1 - dom.y0);
    }
    return { X, n };
  }
  // The dataset at one stage in the shared 3-D box: d <= 3 coordinates as the first d axes (the rest
  // 0), d > 3 through the top three principal components (signs matched to the previous stage),
  // scaled so the points fill the box. simplex: the probabilities themselves, world = 2p - 1.
  function embedStage(net, S, F, FL, FC, n, nl, prev, simplex) {
    const d = S.d, pick = R => (R ? (S.part === 'z' ? R.z[S.l] : R.a[S.l]) : null);
    const V = pick(F), VL = pick(FL), VC = pick(FC);
    let U = null, share = null;
    if (d > 3 && V) {
      const mean = new Float64Array(d), C = new Float64Array(d * d);
      for (let s = 0; s < n; s++) for (let j = 0; j < d; j++) mean[j] += V[s * d + j] / n;
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < d; i++) {
          const a = V[s * d + i] - mean[i];
          for (let j = i; j < d; j++) C[i * d + j] += a * (V[s * d + j] - mean[j]);
        }
      }
      for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) C[i * d + j] = C[j * d + i];
      const { values, vectors } = symEig(C, d);
      const tot = values.reduce((sum, v) => sum + Math.max(0, v), 0) || 1;
      U = new Float64Array(d * 3);
      for (let j = 0; j < d; j++) for (let k = 0; k < 3; k++) U[j * 3 + k] = vectors[j * d + k];
      share = (Math.max(0, values[0]) + Math.max(0, values[1]) + Math.max(0, values[2])) / tot;
    }
    const proj = (X, m) => {
      const C = new Float64Array(m * 3);
      if (!X) return null;
      for (let s = 0; s < m; s++) {
        for (let k = 0; k < 3; k++) {
          let v = 0;
          if (U) for (let j = 0; j < d; j++) v += X[s * d + j] * U[j * 3 + k];
          else if (k < d) v = X[s * d + k];
          C[s * 3 + k] = v;
        }
      }
      return C;
    };
    const C = proj(V, n), CL = proj(VL, nl), CC = proj(VC, 1);
    if (U && prev && C) {
      for (let k = 0; k < 3; k++) {
        let s0 = 0;
        for (let s = 0; s < n; s++) s0 += C[s * 3 + k] * prev.raw[s * 3 + k];
        if (s0 < 0) for (const A of [C, CL, CC]) if (A) for (let s = k; s < A.length; s += 3) A[s] = -A[s];
      }
    }
    let e = 1, off = 0;
    const act = S.l > 0 && S.part === 'a' && net.layers[S.l].kind !== 'attention' ? net.layers[S.l].act : 'identity';
    if (simplex) { e = 0.5; off = -1; }
    else if (!U && BOUNDED[act]) e = 1;   // tanh, sigmoid, softmax: the box is their range, so saturation shows at its faces
    else if (C) {
      const abs = Float64Array.from(C, Math.abs).sort();
      e = (abs[Math.floor(0.985 * (abs.length - 1))] || abs[abs.length - 1] || 1) * 1.1;
    }
    const norm = A => (A ? Float32Array.from(A, v => clamp(v / e + off, -1.6, 1.6)) : null);
    const nodes = model.nodesIn(net, S.l);
    const axes = [0, 1, 2].map(k => (U ? `\\mathrm{PC}_{${k + 1}}` : k < d ? (S.part === 'z' ? `z^{(${S.l})}_{${k + 1}}` : nodeTex(net, nodes[k].id)) : null));
    return { S, raw: C, C: norm(C), CL: norm(CL), CC: norm(CC), e, off, pca: !!U, share, axes, d };
  }
  function buildSpace(mode) {
    const b = begin(mode), p = pal();
    b.lines = segments(2 * LINES * (PER - 1), p.gridLine, p.light ? 0.55 : 0.5, { depthWrite: false });
    b.cube = segments(12, p.wall, 1);
    b.axes = segments(3, p.axis, 0.85);
    const c = [];
    for (let k = 0; k < 3; k++) {   // the 4 edges along axis k
      for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) {
        const P = [0, 0, 0], Q = [0, 0, 0], a = (k + 1) % 3, bb = (k + 2) % 3;
        P[a] = Q[a] = s1; P[bb] = Q[bb] = s2; P[k] = -1; Q[k] = 1;
        c.push(...P, ...Q);
      }
    }
    b.cube.set(c);
    // the probability simplex: its triangle, edges and the lines where two classes tie
    b.tri = add(new THREE.Mesh(own(new THREE.BufferGeometry()), own(new THREE.MeshBasicMaterial({
      color: new THREE.Color(p.muted), transparent: true, opacity: p.light ? 0.13 : 0.1, side: THREE.DoubleSide, depthWrite: false,
    }))));
    b.tri.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
    b.triEdge = segments(3, p.axis, 0.9);
    b.triTie = segments(3, p.axis, 0.55);
    b.mark = marker(p.hi, 0.04);
  }
  let spaceData = null;
  function refreshSpace(s) {
    const net = store.net, d = getData(), simplex = s.mode === 'simplex';
    hideMsg();
    const M = model.matrices(net);
    const hasZ = !M.some(m => m.kind === 'attention');
    const st = stagesFor(net, hasZ);
    const F = forwardZA(net, model, d.Xf, d.n, M);
    if (!F) { showMsg('Could not run the net on the dataset.'); return; }
    const ln = d.dim === 2 && d.dom ? linesFor(d.dom) : null;
    const FL = ln ? forwardZA(net, model, ln.X, ln.n, M) : null;
    const x0 = Float64Array.from(model.nodesIn(net, 0), q => Number(q.value) || 0);
    const FC = forwardZA(net, model, x0, 1, M);
    const list = simplex ? [st[st.length - 1]] : st;
    const emb = [];
    let prev = null;
    for (const S of list) { const e = embedStage(net, S, F, FL, FC, d.n, ln ? ln.n : 0, prev, simplex); emb.push(e); prev = e; }
    const P = pointsFor(d.n);
    for (let k = 0; k < d.n; k++) P.color(k, classRGB(d, k));
    spaceData = { st: list, all: st, emb, n: d.n, d, ln, simplex, net };
    placeSpace(s);
  }
  const stageNow = s => {
    const K = spaceData ? spaceData.emb.length : 1;
    if (s.mode === 'simplex') return 0;
    const v = Number.isFinite(s.stage) ? s.stage : spaceData ? defaultStage(store.net, spaceData.st) : 0;
    return clamp(v, 0, K - 1);
  };
  function placeSpace(s) {
    const SP = spaceData, b = built;
    if (!SP || !b || !b.lines) return;
    const net = store.net, p = pal();
    const t = playing ? playing.v : stageNow(s);
    const K = SP.emb.length, a = clamp(Math.floor(t + 1e-9), 0, K - 1), c = Math.min(K - 1, a + 1), u = K > 1 ? clamp(t - a, 0, 1) : 0;
    const A = SP.emb[a], Bm = SP.emb[c], n = SP.n, r = pointR(n);
    const P = pointsFor(n), lerp = (x, y) => x + (y - x) * u;
    if (A.C && Bm.C) {
      for (let k = 0; k < n; k++) P.place(k, lerp(A.C[3 * k], Bm.C[3 * k]), lerp(A.C[3 * k + 1], Bm.C[3 * k + 1]), lerp(A.C[3 * k + 2], Bm.C[3 * k + 2]), r);
      P.done(n);
      P.mesh.visible = true;
    }
    if (A.CL && Bm.CL && SP.ln) {
      const w = [];
      for (let i = 0; i < 2 * LINES; i++) {
        for (let j = 0; j < PER - 1; j++) {
          const q0 = 3 * (i * PER + j), q1 = q0 + 3;
          w.push(lerp(A.CL[q0], Bm.CL[q0]), lerp(A.CL[q0 + 1], Bm.CL[q0 + 1]), lerp(A.CL[q0 + 2], Bm.CL[q0 + 2]),
            lerp(A.CL[q1], Bm.CL[q1]), lerp(A.CL[q1 + 1], Bm.CL[q1 + 1]), lerp(A.CL[q1 + 2], Bm.CL[q1 + 2]));
        }
      }
      b.lines.set(w);
    } else b.lines.set([]);
    if (A.CC && Bm.CC) b.mark.set(lerp(A.CC[0], Bm.CC[0]), lerp(A.CC[1], Bm.CC[1]), lerp(A.CC[2], Bm.CC[2]));
    else b.mark.set(NaN, 0, 0);
    const near = u < 0.5 ? A : Bm, exact = u < 0.02 || u > 0.98;
    // the simplex: in simplex mode, or when the stage is a 3-class softmax output
    const L = net.layers.length;
    const tri = SP.simplex || (near.S.part === 'a' && near.S.l === L - 1 && isSimplexNet(net));
    b.tri.visible = tri;
    const labels = [];
    if (tri) {
      const e = near.e, o = near.off, V = [[1 / e + o, o, o], [o, 1 / e + o, o], [o, o, 1 / e + o]];
      const arr = b.tri.geometry.attributes.position.array;
      V.flat().forEach((v, k) => { arr[k] = v; });
      b.tri.geometry.attributes.position.needsUpdate = true;
      b.tri.geometry.computeBoundingSphere();
      b.triEdge.set([...V[0], ...V[1], ...V[1], ...V[2], ...V[2], ...V[0]]);
      const cen = [0, 1, 2].map(k => (V[0][k] + V[1][k] + V[2][k]) / 3), mid = (i, j) => [0, 1, 2].map(k => (V[i][k] + V[j][k]) / 2);
      b.triTie.set([...cen, ...mid(0, 1), ...cen, ...mid(1, 2), ...cen, ...mid(0, 2)]);
      const outs = outNodes(net);
      V.forEach((v, k) => {
        const push = [0, 1, 2].map(q => v[q] + (v[q] - cen[q]) * 0.16);
        labels.push({ key: 'v' + k, p: push, tex: `${nodeTex(net, outs[k].id)} = 1`, cls: 'ttl' });
      });
    } else { b.triEdge.set([]); b.triTie.set([]); }
    // axes: through the data origin in space mode (the ends read ±e), from the corner in the simplex
    if (SP.simplex) b.axes.set([]);   // seen along (1, 1, 1) the cube's axes would all point from the triangle's centre
    else {
      const ax = near.axes, w = [];
      ax.forEach((tx, k) => { if (tx != null) { const P0 = [0, 0, 0], P1 = [0, 0, 0]; P0[k] = -1.08; P1[k] = 1.08; w.push(...P0, ...P1); } });
      b.axes.set(w);
      ax.forEach((tx, k) => {
        const q = [0, 0, 0];
        q[k] = 1.3;
        labels.push({ key: 't' + k, p: q, tex: tx ?? '', cls: 'ttl', hidden: tx == null });
        const e1 = [0, 0, 0], e0 = [0, 0, 0];
        e1[k] = 1; e0[k] = -1;
        const dn = k === 2 ? [0.12, -0.12, 0] : [0, 0, -0.12];
        labels.push({ key: 'p' + k, p: e1.map((v, i) => v + dn[i]), text: tickText(near.e), cls: 'tick', hidden: tx == null || !exact });
        labels.push({ key: 'n' + k, p: e0.map((v, i) => v + dn[i]), text: tickText(-near.e), cls: 'tick', hidden: tx == null || !exact || near.pca });
      });
    }
    setLabels(labels);
    spaceCaption(net, SP, A, Bm, u, near, exact);
    if (!ro && built?.stops) paintStops(t);
    probeData.space = {
      stage: t, stages: SP.emb.map(E => ({ l: E.S.l, part: E.S.part, d: E.d, e: E.e, off: E.off, pca: E.pca, share: E.share })),
      coords: Array.from((near.C || []).slice(0, 15)), X: SP.d.X.slice(0, 5), n,
    };
  }
  function spaceCaption(net, SP, A, Bm, u, near, exact) {
    const L = net.layers.length, S = near.S, name = stageTex(net, S);
    const ly = net.layers[S.l], act = S.l > 0 ? ly.act : 'identity';
    if (SP.simplex) {
      const outs = outNodes(net).map(q => nodeTex(net, q.id));
      setCaption(`$\\hat y = \\mathrm{softmax}(z)$ always lands on the triangle $${outs.join(' + ')} = 1$: each corner is one class for certain, and the lines from the centre mark where two classes tie.`,
        `Dots: the training points in their class colour${SP.ln ? '; lines: the input grid, carried onto the simplex' : ''}; ● the current input.`);
      return;
    }
    const dimNote = E => (E.pca ? `the top 3 principal components of its ${E.d} dimensions (${Math.round(E.share * 100)}% of the spread)`
      : E.d === 3 ? 'its 3 neurons as the axes' : E.d === 2 ? 'its 2 neurons as the axes, in the plane of the third = 0' : `its ${E.d} neuron on one axis`);
    const f = FN[act];
    let tex;
    if (!exact) {
      const from = stageTex(net, A.S), to = stageTex(net, Bm.S);
      const how = Bm.S.part === 'z' ? 'the affine map moves every point at once: straight lines stay straight and parallel ones stay parallel'
        : act === 'relu' || act === 'leaky' ? `$${FN[act]}$ folds each negative coordinate ${act === 'relu' ? 'onto 0' : 'to a tenth'}`
          : act === 'softmax' ? 'softmax squeezes every point onto the triangle of probabilities'
            : f ? `$${f}$ squashes each axis on its own` : 'the layer passes it on';
      tex = `From $${from}$ to $${to}$: ${how}.`;
    } else if (S.l === 0) tex = `The training points in input space $x$, ${dimNote(near)}.`;
    else if (S.part === 'z') tex = `$z^{(${S.l})} = W^{(${S.l})} ${S.l === 1 ? 'x' : stageTex(net, { l: S.l - 1, part: 'a' })} + b^{(${S.l})}$: the data after layer ${S.l}'s affine map, ${dimNote(near)}.`;
    else if (ly.kind === 'attention') tex = `$${name}$: after attention in layer ${S.l}, ${dimNote(near)}.`;
    else tex = `$${name}${f ? ` = ${f}(z^{(${S.l})})` : ''}$: the data in layer ${S.l}${S.l === L - 1 ? ' (the output)' : ''}'s activation space, ${dimNote(near)}.`;
    const note = `Each stage is scaled to fit the box${near.pca ? '; a PCA view is a shadow: points close here may be far apart in the other dimensions' : ''}. ● the current input.`;
    setCaption(tex, note);
  }

  // ---- morph playback (presenter): moves s3d.stage, slowing near each stage so it lingers there
  let playing = null;
  function togglePlay() {
    if (playing) { stopPlay(); return; }
    const s = cur();
    if (!s || !spaceData) return;
    const K = spaceData.emb.length;
    let v = stageNow(s);
    if (v >= K - 1 - 1e-6) v = 0;
    playing = { v, target: K - 1 };
    paintPlay();
    kick();
  }
  function goStage(k) {
    const s = cur();
    if (!s || !spaceData) return;
    playing = { v: stageNow(s), target: clamp(k, 0, spaceData.emb.length - 1) };
    paintPlay();
    kick();
  }
  function stopPlay() {
    if (!playing) return;
    const v = playing.v;
    playing = null;
    paintPlay();
    setS3d({ stage: Math.round(v * 1000) / 1000 });
  }
  function stepPlay(dt) {
    const pl = playing, dir = Math.sign(pl.target - pl.v);
    if (!dir) { stopPlay(); return false; }
    const fr = pl.v - Math.floor(pl.v);
    const sp = SPEED * (0.3 + 0.7 * Math.sin(Math.PI * clamp(fr, 0.02, 0.98)));
    pl.v += dir * sp * dt;
    if ((pl.v - pl.target) * dir >= 0) { pl.v = pl.target; stopPlay(); return false; }
    setS3d({ stage: Math.round(pl.v * 1000) / 1000 });
    placeDirty = true;
    return true;
  }

  // ---------------------------------------------------------------- controls row
  let ctlSig = '';
  function neuronOptions(net) {
    const L = net.layers.length, outs = outNodes(net);
    const opts = [{ v: 'sel', t: 'follow the selection' }, { v: 'out', t: outs.length > 1 ? 'output: the winning class' : `output ${plainLabel(nodeTex(net, outs[0]?.id))}` }];
    for (let l = 1; l < L; l++) {
      for (const q of model.nodesIn(net, l)) {
        if (l === L - 1 && outs.length === 1) continue;
        opts.push({ v: q.id, t: `${l === L - 1 ? 'output' : 'layer ' + l}: ${plainLabel(nodeTex(net, q.id))}` });
      }
    }
    return opts;
  }
  const opt = (v, t) => `<option value="${esc(v)}">${esc(t)}</option>`;
  function syncCtl(s) {
    const net = store.net;
    if (ro) return;
    let sig, html = '';
    if (s.mode === 'surface') {
      const opts = neuronOptions(net);
      sig = 'surface|' + JSON.stringify(opts);
      html = `<label title="Which neuron's value is the height (click a neuron on the canvas to follow it)">height <select data-k="neuron">${opts.map(o => opt(o.v, o.t)).join('')}</select></label>
        <div class="s3-seg" title="a: the neuron's activation f(z); z: its input sum W a + b, before the activation"><button type="button" data-k="pre" data-v="0">a = f(z)</button><button type="button" data-k="pre" data-v="1">z</button></div>`;
    } else if (s.mode === 'landscape') {
      const ev = evaluator(), P = ev ? ev.params : [];
      const popts = P.map(q => opt(q.key, paramText(net, q))).join('');
      sig = 'landscape|' + (ev ? ev.psig : '') + '|' + P.map(q => paramText(net, q)).join(',');
      html = `<label title="The plane through θ₀ the loss is drawn on">plane <select data-k="dirs">${opt('random', 'random')}${opt('pca', 'PCA of the path')}${opt('weights', 'two weights')}</select></label>
        <button type="button" data-act="reroll" title="Two new random directions">&#8635;</button>
        <label class="s3-w">x <select data-k="wa">${popts}</select></label>
        <label class="s3-w">y <select data-k="wb">${popts}</select></label>
        <label title="Half-width of the plane (auto: fits the path)">range <select data-k="range">${RANGES.map(v => opt(String(v), v ? '±' + v : 'auto')).join('')}</select></label>
        <label class="s3-check" title="Height by log₁₀ of the loss"><input type="checkbox" data-k="log"> log</label>
        <button type="button" data-act="recenter" title="Put the plane's centre θ₀ at the current weights">Re-center</button>
        <button type="button" data-act="clear" title="Forget the training path so far">Clear path</button>`;
    } else if (s.mode === 'space') {
      const st = spaceData?.emb.map(E => E.S) || stagesFor(net, !model.matrices(net).some(m => m.kind === 'attention'));
      sig = 'space|' + JSON.stringify(st.map(S => [S.l, S.part])) + '|' + st.map(S => stageTex(net, S)).join(',');
      html = `<button type="button" class="s3-play" data-act="play" title="Morph through every layer, from the inputs to the outputs">&#9654;</button>
        <div class="s3-track"><input type="range" data-k="stage" min="0" max="${Math.max(0, st.length - 1)}" step="0.001">
        <div class="s3-stops">${st.map((S, k) => `<button type="button" data-act="stop" data-v="${k}" title="Go to this stage">${texHtml(stageTex(net, S))}</button>`).join('')}</div></div>`;
    } else sig = 'simplex';
    if (sig !== ctlSig) {
      ctlSig = sig;
      ctl.innerHTML = html;
      if (built) built.stops = s.mode === 'space' ? [...ctl.querySelectorAll('.s3-stops button')] : null;
      requestAnimationFrame(placePanel);
    }
    const set = (k, v) => {
      const el = ctl.querySelector(`[data-k="${k}"]`);
      if (!el || el === document.activeElement) return;
      if (el.type === 'checkbox') el.checked = !!v;
      else if (el.value !== String(v)) el.value = String(v);
    };
    if (s.mode === 'surface') {
      set('neuron', s.neuron && [...ctl.querySelector('[data-k="neuron"]').options].some(o => o.value === s.neuron) ? s.neuron : 'sel');
      const tg = resolveNeuron(net, s), canPre = !tg.argmax && tg.l > 0 && !tg.attn;
      for (const b of ctl.querySelectorAll('[data-k="pre"]')) {
        b.classList.toggle('on', (b.dataset.v === '1') === !!(s.pre && canPre));
        b.disabled = !canPre;
      }
    } else if (s.mode === 'landscape') {
      const ev = evaluator(), B = s.basis;
      const dirs = ev ? dirsOf(s, ev) : 'random';
      set('dirs', dirs);
      set('range', String(s.range || 0));
      set('log', !!s.log);
      if (B?.wa) set('wa', B.wa);
      if (B?.wb) set('wb', B.wb);
      for (const el of ctl.querySelectorAll('.s3-w')) el.hidden = dirs !== 'weights';
      const rr = ctl.querySelector('[data-act="reroll"]');
      if (rr) rr.hidden = dirs !== 'random';
    } else if (s.mode === 'space') {
      if (built) built.stops = [...ctl.querySelectorAll('.s3-stops button')];
      paintPlay();
    }
  }
  function paintStops(t) {
    const r = ctl.querySelector('[data-k="stage"]');
    if (r && r !== document.activeElement) r.value = String(t);
    (built?.stops || []).forEach((el, k) => el.classList.toggle('on', Math.abs(k - t) < 0.02));
  }
  function paintPlay() {
    const b = ctl.querySelector('.s3-play');
    if (b) { b.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;'; b.classList.toggle('on', !!playing); }
  }
  ctl.addEventListener('change', e => {
    const el = e.target, k = el.dataset.k, s = cur();
    if (!k || !s || ro) return;
    if (el.tagName === 'SELECT' || el.type === 'checkbox') el.blur();
    if (k === 'neuron') setS3d({ neuron: el.value });
    else if (k === 'dirs') rebase({ dirs: el.value });
    else if (k === 'wa' || k === 'wb') rebase({ dirs: 'weights', wa: k === 'wa' ? el.value : s.basis?.wa ?? s.wa, wb: k === 'wb' ? el.value : s.basis?.wb ?? s.wb });
    else if (k === 'range') rebase({ range: Number(el.value) || 0 });
    else if (k === 'log') setS3d({ log: el.checked });
  });
  ctl.addEventListener('input', e => {
    const el = e.target;
    if (el.dataset.k !== 'stage' || ro) return;
    if (playing) { playing = null; paintPlay(); }
    setS3d({ stage: Number(el.value) });
  });
  ctl.addEventListener('pointerup', e => { if (e.target.dataset?.k === 'stage') e.target.blur(); });
  ctl.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || ro) return;
    const s = cur();
    if (!s) return;
    if (b.dataset.k === 'pre') setS3d({ pre: b.dataset.v === '1' });
    const act = b.dataset.act;
    if (act === 'reroll') rebase({ dirs: 'random', seed: (s.seed || 1) + 1 });
    else if (act === 'recenter') rebase();
    else if (act === 'clear') clearPath();
    else if (act === 'play') togglePlay();
    else if (act === 'stop') goStage(Number(b.dataset.v));
    b.blur();
  });
  panel.addEventListener('click', e => {
    const b = e.target.closest('.s3-head button');
    if (!b || ro) return;
    if (b.dataset.mode) show(b.dataset.mode);
    else if (b.dataset.act === 'close') hide();
    else if (b.dataset.act === 'home') home();
  });

  // ---------------------------------------------------------------- the frame loop
  let raf = 0, needRender = true, dataDirty = true, placeDirty = false, lastRefresh = -1e9, lastT = 0, lastMode = null;
  const probeData = {};
  const stats = { refresh: 0, refreshes: 0, render: 0, renders: 0 };   // ms spent, for tests
  function kick() { if (!raf && cur() && shownTab()) raf = requestAnimationFrame(frame); }
  function build(mode) {
    for (const k of Object.keys(probeData)) delete probeData[k];
    if (mode === 'surface') buildSurface();
    else if (mode === 'landscape') buildLandscape();
    else buildSpace(mode);
    capKey = '';
    ctlSig = '';
  }
  function paintHead(s) {
    const why = availability();
    for (const b of panel.querySelectorAll('.s3-modes button')) {
      const m = b.dataset.mode;
      b.classList.toggle('on', m === s.mode);
      b.classList.toggle('na', !!why[m]);
      b.title = why[m] ? `${MODE_TITLE[m].replace(/ \(Shift\+P.*$/, '')}. ${why[m]}` : MODE_TITLE[m];
    }
    return why;
  }
  function refresh(s) {
    const why = paintHead(s);
    if (why[s.mode]) {
      showMsg(why[s.mode]);
      for (const k of Object.keys(probeData)) delete probeData[k];
      return;
    }
    if (s.mode === 'surface') refreshSurface(s);
    else if (s.mode === 'landscape') refreshLandscape(s);
    else refreshSpace(s);
  }
  function frame(now) {
    raf = 0;
    const s = cur();
    if (!s || !shownTab() || panel.hidden) return;
    if (!GL) {
      if (glError) showMsg(`3D plots need WebGL, which is not available here (${glError}).`);
      else ensureGL();
      return;
    }
    let again = false;
    const dt = Math.min(0.1, (now - (lastT || now)) / 1000);
    lastT = now;
    if (!built || built.mode !== s.mode) { build(s.mode); dataDirty = true; }
    syncCtl(s);
    if (playing) { if (stepPlay(dt)) again = true; }
    if (camWanted) { if (followCam()) again = true; }
    else if (!ro && GL.controls.update()) again = true;
    if (dataDirty) {
      if (now - lastRefresh >= (busy() ? REFRESH_MS : 0)) {
        dataDirty = false;
        lastRefresh = now;
        placeDirty = false;
        const t0 = performance.now();
        try { refresh(cur()); } catch (err) { console.error('[nn/surf3d] draw:', err); showMsg(`Could not draw: ${err.message}`); }
        stats.refresh += performance.now() - t0; stats.refreshes++;
        needRender = true;
      } else again = true;
    }
    if (placeDirty && (s.mode === 'space' || s.mode === 'simplex')) {
      placeDirty = false;
      try { placeSpace(cur()); } catch (err) { console.error('[nn/surf3d] place:', err); }
      needRender = true;
    }
    if (job) {
      stepJob(BUDGET_MS);
      dataDirty = true;   // redraw with the finished grid
      again = true;
    }
    if (dataDirty) again = true;
    if (needRender) {
      needRender = false;
      const P = pal();
      if (!P.clear) P.clear = new THREE.Color(P.bg);
      GL.renderer.setClearColor(P.clear, 1);
      const t0 = performance.now();
      GL.renderer.render(GL.scene, GL.camera);
      GL.labels.render(GL.scene, GL.camera);
      stats.render += performance.now() - t0; stats.renders++;
    }
    if (again) kick();
  }

  // ---------------------------------------------------------------- store events
  let lastTitle = store.net.meta?.title ?? '';
  let prevState = null;
  function onState() {
    const s = cur(), open = !!s, was = prevState;
    prevState = s;
    const opened = panel.hidden === open && open;
    if (panel.hidden === open) {
      panel.hidden = !open;
      toolBtn?.classList.toggle('on', open);
      placePanel();
    }
    if (!open) { if (playing) { playing = null; paintPlay(); } job = null; lastMode = null; return; }
    panel.dataset.mode = s.mode;
    const moded = s.mode !== lastMode;
    if (moded) {
      lastMode = s.mode;
      ctlSig = '';
      playing = null;
      if (!ro && ui.mode !== s.mode) { ui.mode = s.mode; saveUi(); }
    }
    for (const b of panel.querySelectorAll('.s3-modes button')) b.classList.toggle('on', b.dataset.mode === s.mode);
    if (opened || moded) lastCamSent = '\u0000';   // a fresh view: apply the state's camera (null: the default)
    camFromState(s);
    // only the stage or the camera moved: re-place, no new forward passes
    const viewOnly = !opened && !moded && was && Object.keys({ ...s, ...was }).every(k => k === 'stage' || k === 'cam' || same(s[k], was[k]));
    if (!viewOnly) dataDirty = true;
    if ((s.mode === 'space' || s.mode === 'simplex') && (!was || !same(s.stage, was.stage))) placeDirty = true;
    if (!GL) ensureGL();
    kick();
  }
  store.on('s3d', onState);
  store.on('values', () => { lastValues = performance.now(); if (cur()) { dataDirty = true; kick(); } });
  store.on('sel', () => { if (cur()?.mode === 'surface') { dataDirty = true; kick(); } });
  store.on('net', p => {
    const net = store.net, s = cur();
    if (p?.structural) {
      if (GL && built) clearBuilt();
      spaceData = null;
      grid = null; job = null;
      gridC.key = '';
    }
    ctlSig = p?.structural ? '' : ctlSig;
    const title = net.meta?.title ?? '';
    if (title !== lastTitle) {
      lastTitle = title;
      trail = []; trailVer++;
      if (s && !ro) setS3d({ neuron: 'sel', pre: false, stage: null, basis: null, trail: [], wa: null, wb: null, dirs: null });
    } else if (s && !ro && s.mode === 'surface' && s.neuron !== 'sel' && s.neuron !== 'out' && !model.node(net, s.neuron)) setS3d({ neuron: 'sel' });
    if (s) { dataDirty = true; kick(); }
  });
  ctx.onTheme?.(() => { PAL = null; capKey = ''; texCache.clear(); if (GL && built) clearBuilt(); dataDirty = true; needRender = true; kick(); });
  ctx.onShow?.(v => { if (v) { placePanel(); dataDirty = true; needRender = true; kick(); } });

  // ---------------------------------------------------------------- toolbar button and keys
  let toolBtn = null;
  if (!ro && ctx.addButton) {
    try {
      toolBtn = ctx.addButton({
        label: '3D plots', icon: '&#8779;', group: 'surf3d',
        title: '3D plots: a neuron as a surface over the inputs, the loss landscape with the training path, the data through the layers, the softmax simplex (P: open / close, Shift+P: next plot)',
        onClick: () => toggle(),
      });
    } catch (err) { console.warn('[nn/surf3d] addButton:', err); }
  }
  window.addEventListener('keydown', e => {
    if (ro || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    if (e.key !== 'p' && e.key !== 'P') return;
    if (ctx.active && !ctx.active(e)) return;
    e.preventDefault();
    if (e.shiftKey) cycle(1);
    else toggle();
  });

  // ---------------------------------------------------------------- handle
  const api = {
    MODES,
    get open() { return !!cur(); },
    get mode() { return cur()?.mode ?? null; },
    show, hide, toggle, cycle,
    recenter: () => rebase(),
    clearPath,
    available: () => availability(),
    // For tests: what the last drawing used (surface heights and values, the landscape's centre and
    // current point, the space stage's first coordinates) and whether the work has settled.
    probe: () => ({ mode: cur()?.mode ?? null, gl: !!GL, error: glError || null, msg: msgEl.hidden ? null : msgEl.textContent, busy: !!job || dataDirty || !!playing, stats: { ...stats }, ...probeData }),
    el: panel,
  };
  ctx.surf3d = api;
  panel.nnS3d = api;
  onState();
}
