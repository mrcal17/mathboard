// Linear-combination and geometry teaching tools for the 3D tab.
//   explain(A, v)     A v drawn as v1 a1 + v2 a2 (+ v3 a3), tip-to-tail, plus the resultant
//   chain(u, v, ...)  arguments tip-to-tail with a faint resultant; value = their sum
//   trail(expr)       the vector plus a trail of its tip as sliders move ("Clear trails" button)
//   target(b, g)      ghost target b that lights up (and toasts) when the guess g lands on it
//   arc(u, v)  shadow(u, v)  components(u)  crossview(u, v)
//   line(p, q)  plane3(p, q, r)  intersect(X, Y)  distance(p, X)
// Also: automatic right-angle markers between perpendicular vectors that share an origin.
// Vector-valued helpers return ordinary vectors tagged with `combo`, so they still compose
// (w = explain(A, v); w + u). Pure helpers are exported for tests; scene.js loads in install().
import { registerFunction, registerType, values, formatNumber, parseLine } from '../lang.js';

const { vec, point, kindOf, describe, isVecLike, dot3, cross3, len3 } = values;

export const TARGET_TOL = 0.15;  // default |guess - target| that counts as a hit
export const RIGHT_TOL = 3e-3;   // |cos θ| below this counts as perpendicular (~0.17°)
export const TRAIL_CAP = 3000;   // points kept per trail

const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const clamp1 = (x) => Math.max(-1, Math.min(1, x));
const tagged = (base, combo) => ({ ...base, combo });

// ---------------------------------------------------------------- colours

export function hexToHsl(hex) {
  const n = parseInt(String(hex).replace('#', '').slice(0, 6), 16);
  const r = ((n >> 16) & 255) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  if (!d) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, s, l];
}

export function hslToHex(h, s, l) {
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
function legible(h, s, l, theme) {
  const light = theme === 'light';
  for (let k = 0; k < 40; k++) {
    const hex = hslToHex(h, s, l), y = luminance(hex);
    if (light ? y <= 0.2 : y >= 0.2) return hex;
    l = Math.min(0.95, Math.max(0.05, l + (light ? -0.02 : 0.02)));
  }
  return hslToHex(h, s, l);
}

// Distinct shades of one colour (hue nudged, lightness varied), readable on the given theme:
// lighter variants on the chalkboard, darker ones on the whiteboard.
const TONES = [[-16, 0.16], [16, -0.13], [0, 0.29], [-30, -0.03], [30, 0.07]];
export function shades(hex, n, theme = 'dark') {
  const [h, s, l] = hexToHsl(hex);
  const light = theme === 'light', sign = light ? -1 : 1;
  const [lo, hi] = light ? [0.22, 0.56] : [0.5, 0.86];
  return Array.from({ length: n }, (_, i) => {
    const [dh, dl0] = TONES[i % TONES.length], dl = sign * dl0;
    let L = Math.min(hi, Math.max(lo, l + dl + sign * 0.05 * Math.floor(i / TONES.length)));
    if (Math.abs(L - l) < 0.08) L = Math.min(hi, Math.max(lo, l + (dl >= 0 ? -0.12 : 0.12)));
    return legible(h + dh, Math.min(1, s * 0.95 + 0.05), L, theme);
  });
}

// ---------------------------------------------------------------- pure geometry

export function explainParts(A, x) {
  if (kindOf(A) !== 'mat') {
    if (isVecLike(A) && kindOf(x) === 'mat') throw new Error('explain needs the matrix first: explain(A, v)');
    throw new Error(`explain needs a matrix and a vector, got ${describe(A)}`);
  }
  if (kindOf(x) !== 'vec') throw new Error(`explain needs a vector after the matrix, got ${describe(x)}`);
  const rows = A.m.length, n = A.m[0].length;
  if (rows < 2 || rows > 3 || n < 2 || n > 3) throw new Error(`explain needs a matrix with 2 or 3 rows and columns, got ${describe(A)}`);
  if (n === 2 && x.v[2] !== 0) throw new Error(`can't multiply ${describe(A)} by a vector with nonzero z`);
  const cols = Array.from({ length: n }, (_, j) => [A.m[0][j], A.m[1][j], rows === 3 ? A.m[2][j] : 0]);
  const coeffs = x.v.slice(0, n);
  const sum = cols.reduce((s, c, j) => add3(s, scale3(c, coeffs[j])), [0, 0, 0]);
  return { cols, coeffs, dim: rows, sum };
}

// Clip p + t d to the cube |x|,|y|,|z| <= B. Returns [t0, t1] or null.
export function clipLine(p, d, B) {
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

// The polygon where the plane n·x = c cuts the cube |x|,|y|,|z| <= B, in order around its
// boundary; [] when the plane misses the cube.
export function planeInBox(n, c, B) {
  const pts = [], f = (p) => dot3(n, p) - c;
  const corners = [];
  for (let i = 0; i < 8; i++) corners.push([i & 1 ? B : -B, i & 2 ? B : -B, i & 4 ? B : -B]);
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue;
      const a = corners[i], b = corners[i | bit], fa = f(a), fb = f(b);
      if ((fa > 0 && fb > 0) || (fa < 0 && fb < 0) || fa === fb) continue;
      const p = add3(a, scale3(sub3(b, a), fa / (fa - fb)));
      if (!pts.some((q) => len3(sub3(p, q)) < 1e-9 * B)) pts.push(p);
    }
  }
  if (pts.length < 3) return [];
  const m = scale3(pts.reduce(add3, [0, 0, 0]), 1 / pts.length);
  const e1 = anyPerp(n), e2 = cross3(scale3(n, 1 / len3(n)), e1);
  const ang = (p) => Math.atan2(dot3(sub3(p, m), e2), dot3(sub3(p, m), e1));
  return pts.sort((p, q) => ang(p) - ang(q));
}

const isInt = (x) => Math.abs(x - Math.round(x)) < 1e-9;
const gcd = (a, b) => (b ? gcd(b, a % b) : a);

// Integer vectors are divided by their gcd; the first nonzero entry is made positive.
export function niceDirection(d, extra = []) {
  const all = [...d, ...extra];
  let k = 1;
  if (all.every(isInt)) {
    const g = all.map((x) => Math.abs(Math.round(x))).filter(Boolean).reduce(gcd, 0);
    if (g > 1) k = 1 / g;
  }
  const first = d.find((x) => Math.abs(x) > 1e-12) ?? 1;
  if (first < 0) k = -k;
  return { d: d.map((x) => (isInt(x) ? Math.round(x) : x) * k || 0), extra: extra.map((x) => (isInt(x) ? Math.round(x) : x) * k || 0) };
}

const makeLine = (p, d, pts = null) => ({ type: 'cb-line', p: p.slice(), d: d.slice(), pts });
function makePlane(n, c, pts = null) {
  const nice = niceDirection(n, [c]);
  return { type: 'cb-plane', n: nice.d, c: nice.extra[0], pts };
}

// Lines and planes in the language -> {kind: 'line', p, d} | {kind: 'plane', n, c} (n·x = c).
export function toFlat(x, fname = 'intersect') {
  switch (kindOf(x)) {
    case 'cb-line': return { kind: 'line', p: x.p, d: x.d };
    case 'cb-plane': return { kind: 'plane', n: x.n, c: x.c };
    case 'plane': return { kind: 'plane', n: x.normal, c: 0 };
    case 'span':
      if (x.vecs.length === 1) return { kind: 'line', p: [0, 0, 0], d: x.vecs[0] };
      if (x.vecs.length === 2) return { kind: 'plane', n: cross3(x.vecs[0], x.vecs[1]), c: 0 };
      throw new Error(`${fname} needs a line or plane, got a span that is ${x.vecs.length ? 'all of R^3' : 'just the origin'}`);
  }
  throw new Error(`${fname} needs lines or planes, got ${describe(x)}`);
}

const scaleOf = (...vs) => Math.max(1, ...vs.flat().map(Math.abs));

export function intersectFlats(A, B) {
  if (A.kind === 'plane' && B.kind === 'line') [A, B] = [B, A];
  if (A.kind === 'plane' && B.kind === 'plane') {
    const n1 = A.n, n2 = B.n, d = cross3(n1, n2);
    const dd = dot3(d, d), s1 = dot3(n1, n1), s2 = dot3(n2, n2);
    if (dd <= 1e-12 * s1 * s2) {
      // parallel: same plane iff c1/|n1| = c2/|n2| once the normals point the same way
      const sign = dot3(n1, n2) < 0 ? -1 : 1;
      const same = Math.abs(A.c / Math.sqrt(s1) - (sign * B.c) / Math.sqrt(s2)) < 1e-9 * scaleOf(A.c, B.c);
      throw new Error(same ? 'the planes are the same plane' : 'the planes are parallel, so they never meet');
    }
    const n12 = dot3(n1, n2);
    const p = add3(scale3(n1, A.c * s2 - B.c * n12), scale3(n2, B.c * s1 - A.c * n12)).map((x) => x / dd);
    return makeLine(p, niceDirection(d).d);
  }
  if (A.kind === 'line' && B.kind === 'plane') {
    const nd = dot3(B.n, A.d), off = B.c - dot3(B.n, A.p);
    if (Math.abs(nd) <= 1e-9 * len3(B.n) * len3(A.d)) {
      throw new Error(Math.abs(off) <= 1e-9 * len3(B.n) * scaleOf(A.p, B.c)
        ? 'the line lies in the plane' : 'the line is parallel to the plane, so they never meet');
    }
    return point(add3(A.p, scale3(A.d, off / nd)));
  }
  // line ∩ line: closest points p1 + s d1 and p2 + t d2
  const { p: p1, d: d1 } = A, { p: p2, d: d2 } = B, w = sub3(p1, p2);
  const a = dot3(d1, d1), b = dot3(d1, d2), c = dot3(d2, d2), d = dot3(d1, w), e = dot3(d2, w);
  const den = a * c - b * b, tol = 1e-7 * scaleOf(p1, p2);
  if (den <= 1e-12 * a * c) {
    const gap = len3(cross3(w, d1)) / Math.sqrt(a);
    throw new Error(gap <= tol ? 'the lines are the same line' : 'the lines are parallel, so they never meet');
  }
  const s = (b * e - c * d) / den, t = (a * e - b * d) / den;
  const q1 = add3(p1, scale3(d1, s)), q2 = add3(p2, scale3(d2, t)), gap = len3(sub3(q1, q2));
  if (gap > tol) throw new Error(`the lines are skew: they miss each other by ${formatNumber(gap)}`);
  return point(scale3(add3(q1, q2), 0.5));
}

// Unit vector perpendicular to a (a nonzero).
export function anyPerp(a) {
  const L = len3(a), u = scale3(a, 1 / L);
  const w = cross3(u, Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
  return scale3(w, 1 / len3(w));
}

// Point-to-(point | line | plane): {p, foot, d, along} with `along` a unit direction in X at the foot.
export function distanceTo(p, X) {
  if (isVecLike(X)) return { p: p.slice(), foot: X.v.slice(), d: len3(sub3(p, X.v)), along: null };
  const F = toFlat(X, 'distance');
  if (F.kind === 'line') {
    const dd = dot3(F.d, F.d), t = dot3(sub3(p, F.p), F.d) / dd;
    const foot = add3(F.p, scale3(F.d, t));
    return { p: p.slice(), foot, d: len3(sub3(p, foot)), along: scale3(F.d, 1 / Math.sqrt(dd)) };
  }
  const nn = dot3(F.n, F.n), k = (dot3(F.n, p) - F.c) / nn;
  const foot = sub3(p, scale3(F.n, k));
  return { p: p.slice(), foot, d: Math.abs(k) * Math.sqrt(nn), along: anyPerp(F.n) };
}

// Pairs of perpendicular vectors that share an origin -> right-angle markers.
// vs: [{o, v}]; returns [{o, a, b, size}] with a, b unit directions and size = shorter length.
export function perpendicularPairs(vs, tol = RIGHT_TOL, max = 60) {
  const out = [], seen = new Set();
  const key = (a) => a.map((x) => Math.round(x * 1e5)).join(',');
  for (let i = 0; i < vs.length; i++) {
    for (let j = i + 1; j < vs.length; j++) {
      const A = vs[i], B = vs[j];
      if (len3(sub3(A.o, B.o)) > 1e-9 * scaleOf(A.o)) continue;
      const la = len3(A.v), lb = len3(B.v);
      if (la < 1e-9 || lb < 1e-9 || Math.abs(dot3(A.v, B.v)) / (la * lb) > tol) continue;
      const a = scale3(A.v, 1 / la), b = scale3(B.v, 1 / lb);
      const k = `${key(A.o)}|${[key(a), key(b)].sort().join('|')}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ o: A.o.slice(), a, b, size: Math.min(la, lb) });
      if (out.length >= max) return out;
    }
  }
  return out;
}

// Trail bookkeeping: one list of tip positions per row, reset when the row's text changes.
export function pushTrail(map, id, src, tip, cap = TRAIL_CAP) {
  let t = map.get(id);
  if (!t || t.src !== src) map.set(id, (t = { src, pts: [] }));
  const last = t.pts.at(-1);
  if (!last || len3(sub3(last, tip)) > 1e-6) t.pts.push(tip.slice());
  if (t.pts.length > cap) t.pts.splice(0, t.pts.length - cap);
  return t;
}

// ---------------------------------------------------------------- LaTeX

export function nameTex(name, vector) {
  const m = name.match(/^(.*?)(?:_(\w+)|(\d+))$/);
  let base = m && m[1] ? m[1] : name;
  const sub = m && m[1] ? m[2] || m[3] : '';
  if (base.length > 1) base = `\\mathrm{${base}}`;
  const b = vector ? `\\vec{${base}}` : base;
  return sub ? `${b}_{${sub}}` : b;
}

const num = formatNumber;
const short = (x) => { const s = (Math.round(x * 100) / 100).toFixed(2).replace(/\.?0+$/, ''); return s === '-0' ? '0' : s; };
export const tupleTex = (a, n = 3) => `(${a.slice(0, n).map(num).join(', ')})`;
const paint = (color, tex) => (color ? `\\textcolor{${color}}{${tex}}` : tex);

// terms: [{k, body, color}] -> "2\cdot(1, 0, 0) - 1\cdot(1, 1, 0)"
export function sumTex(terms) {
  return terms.map(({ k, body, color }, i) => {
    const neg = k < 0;
    const t = paint(color, `${num(Math.abs(k))}\\cdot ${body}`);
    return i === 0 ? (neg ? `-${t}` : t) : `${neg ? '-' : '+'} ${t}`;
  }).join(' ');
}

export function planeTex(n, c) {
  const parts = [];
  n.forEach((k, i) => {
    if (Math.abs(k) < 1e-12) return;
    const mag = Math.abs(k) === 1 ? '' : num(Math.abs(k));
    const t = `${mag}${'xyz'[i]}`;
    parts.push(parts.length ? `${k < 0 ? '-' : '+'} ${t}` : `${k < 0 ? '-' : ''}${t}`);
  });
  return `${parts.join(' ')} = ${num(c)}`;
}

export function componentsTex(v) {
  const hats = ['\\hat\\imath', '\\hat\\jmath', '\\hat k'], parts = [];
  v.forEach((k, i) => {
    if (Math.abs(k) < 1e-12) return;
    const mag = Math.abs(k) === 1 ? '' : `${num(Math.abs(k))}\\,`;
    const t = `${mag}${hats[i]}`;
    parts.push(parts.length ? `${k < 0 ? '-' : '+'} ${t}` : `${k < 0 ? '-' : ''}${t}`);
  });
  return parts.length ? parts.join(' ') : '\\vec 0';
}

const HIT_COLOR = '#43b05c';
const deg = (rad) => `${short((rad * 180) / Math.PI)}^\\circ`;

// Full row readout (KaTeX source) for our values, or null to keep the grapher's default.
export function readoutTex(v, color = null, theme = 'dark') {
  if (!v || typeof v !== 'object') return null;
  const cb = v.combo;
  switch (cb?.mode ?? v.type) {
    case 'explain': {
      const tones = color ? shades(color, cb.cols.length, theme) : [];
      const terms = cb.cols.map((col, j) => ({ k: cb.coeffs[j], body: tupleTex(col, cb.dim), color: tones[j] }));
      return `= ${sumTex(terms)} = ${tupleTex(v.v, cb.dim)}`;
    }
    case 'chain': return `= ${cb.parts.map((p) => tupleTex(p)).join(' + ')} = ${tupleTex(v.v)}`;
    case 'target':
      return cb.hit
        ? `= ${tupleTex(v.v)}\\quad ${paint(HIT_COLOR, '\\checkmark\\ \\text{on target}')}`
        : `= ${tupleTex(v.v)}\\quad \\text{off by } ${num(cb.miss)}`;
    case 'shadow': return `= \\tfrac{${num(cb.uv)}}{${num(cb.vv)}}\\,${tupleTex(cb.onto)} = ${tupleTex(v.v)}`;
    case 'components': return `= ${tupleTex(v.v)} = ${componentsTex(v.v)}`;
    case 'crossview': return `= ${tupleTex(v.v)}\\quad \\text{area } ${num(cb.area)}`;
    case 'cb-arc': return `\\theta = ${deg(v.theta)}\\ \\ (${num(v.theta)}\\text{ rad})`;
    case 'cb-line': return `\\vec r(t) = ${tupleTex(v.p)} + t\\,${tupleTex(v.d)}`;
    case 'cb-plane': return planeTex(v.n, v.c);
    case 'cb-dist': return `d = ${num(v.d)}`;
  }
  return null;
}

// ---------------------------------------------------------------- language

const needVecLike = (x, fname, what = 'a vector or point') => {
  if (!isVecLike(x)) throw new Error(`${fname} needs ${what}, got ${describe(x)}`);
  return x;
};

registerFunction('explain', {
  n: 2,
  f: ([A, x]) => {
    const { cols, coeffs, dim, sum } = explainParts(A, x);
    return tagged(vec(sum), { mode: 'explain', cols, coeffs, dim });
  },
});

registerFunction('chain', {
  n: [1, Infinity], kind: 'vec',
  f: (vs) => {
    const parts = vs.map((u) => u.v.slice());
    return tagged(vec(parts.reduce(add3, [0, 0, 0])), { mode: 'chain', parts });
  },
});

registerFunction('trail', {
  n: 1,
  f: ([x], name) => {
    needVecLike(x, name);
    return tagged(kindOf(x) === 'point' ? point(x.v) : vec(x.v), { mode: 'trail' });
  },
});

registerFunction('target', {
  n: [2, 3],
  f: ([b, g, tol], name) => {
    needVecLike(b, name);
    needVecLike(g, name);
    if (tol != null && (kindOf(tol) !== 'num' || !(tol > 0))) throw new Error('target tolerance must be a positive number');
    const miss = len3(sub3(g.v, b.v)), t = tol ?? TARGET_TOL;
    return tagged(kindOf(g) === 'point' ? point(g.v) : vec(g.v), { mode: 'target', b: b.v.slice(), miss, tol: t, hit: miss <= t });
  },
});

registerFunction('shadow', {
  n: 2, kind: 'vec',
  f: ([u, v]) => {
    const vv = dot3(v.v, v.v);
    if (vv === 0) throw new Error("can't project onto the zero vector");
    const uv = dot3(u.v, v.v);
    return tagged(vec(scale3(v.v, uv / vv)), { mode: 'shadow', u: u.v.slice(), onto: v.v.slice(), uv, vv });
  },
});

registerFunction('components', {
  n: 1,
  f: ([u], name) => tagged(kindOf(needVecLike(u, name)) === 'point' ? point(u.v) : vec(u.v), { mode: 'components' }),
});

registerFunction('crossview', {
  n: 2, kind: 'vec',
  f: ([u, v]) => {
    const n = cross3(u.v, v.v);
    return tagged(vec(n), { mode: 'crossview', u: u.v.slice(), w: v.v.slice(), area: len3(n) });
  },
});

registerType('cb-arc', {
  describe: 'an angle',
  format: (v) => `${short((v.theta * 180) / Math.PI)}°`,
  latex: (v) => deg(v.theta),
  numbers: (v) => [v.theta, ...v.u, ...v.v],
});
registerFunction('arc', {
  n: 2, kind: 'vec',
  f: ([u, v]) => {
    const d = len3(u.v) * len3(v.v);
    if (d === 0) throw new Error('arc needs two nonzero vectors');
    return { type: 'cb-arc', u: u.v.slice(), v: v.v.slice(), theta: Math.acos(clamp1(dot3(u.v, v.v) / d)) };
  },
});

registerType('cb-line', {
  describe: 'a line',
  format: (v) => `line through (${v.p.map(num).join(', ')}) along (${v.d.map(num).join(', ')})`,
  latex: (v) => `${tupleTex(v.p)} + t\\,${tupleTex(v.d)}`,
  numbers: (v) => [...v.p, ...v.d],
});
registerFunction('line', {
  n: 2,
  f: ([p, q], name) => {
    needVecLike(p, name, 'two points');
    needVecLike(q, name, 'two points');
    const d = sub3(q.v, p.v);
    if (len3(d) <= 1e-12 * scaleOf(p.v, q.v)) throw new Error('line needs two different points');
    return makeLine(p.v, d, [p.v.slice(), q.v.slice()]);
  },
});

registerType('cb-plane', {
  describe: 'a plane',
  format: (v) => planeTex(v.n, v.c),
  latex: (v) => `\\{${planeTex(v.n, v.c)}\\}`,
  numbers: (v) => [...v.n, v.c],
});
registerFunction('plane3', {
  n: 3,
  f: ([p, q, r], name) => {
    for (const x of [p, q, r]) needVecLike(x, name, 'three points');
    const n = cross3(sub3(q.v, p.v), sub3(r.v, p.v));
    if (len3(n) <= 1e-12 * scaleOf(p.v, q.v, r.v) ** 2) throw new Error("the three points lie on one line, so they don't pick out a plane");
    return makePlane(n, dot3(n, p.v), [p.v.slice(), q.v.slice(), r.v.slice()]);
  },
});

registerFunction('intersect', {
  n: 2,
  f: ([X, Y], name) => intersectFlats(toFlat(X, name), toFlat(Y, name)),
});

registerType('cb-dist', {
  describe: 'a distance',
  format: (v) => `distance ${num(v.d)}`,
  latex: (v) => num(v.d),
  numbers: (v) => [...v.p, ...v.foot, v.d],
});
registerFunction('distance', {
  n: 2,
  f: ([a, b], name) => {
    if (!isVecLike(a) && isVecLike(b)) [a, b] = [b, a];
    if (!isVecLike(a)) throw new Error(`${name} needs a point and a point, line or plane, got ${describe(a)} and ${describe(b)}`);
    return { type: 'cb-dist', ...distanceTo(a.v, b) };
  },
});

// ---------------------------------------------------------------- items (browser)

// Follow `name` rows back to the call that made a value, e.g. w = explain(A, v); w.
function callOf(src, rows, results, depth = 0) {
  const body = parseLine(src)?.body;
  if (!body || depth > 4) return null;
  if (body.t === 'call') return body;
  if (body.t !== 'name') return null;
  const i = results.findIndex((r) => r?.name === body.name);
  return i < 0 ? null : callOf(rows[i].src, rows, results, depth + 1);
}

const sameOrigin = (a, b) => len3(sub3(a || [0, 0, 0], b || [0, 0, 0])) < 1e-9;

// Per argument: {name, color, drawn} where drawn = its own row already shows it from our origin.
function argInfo(it, rows, results) {
  const call = callOf(rows[it.index]?.src ?? '', rows, results);
  if (!call) return [];
  return call.args.map((a) => {
    if (a.t !== 'name') return { name: null, color: null, drawn: false };
    const i = results.findIndex((r) => r?.name === a.name);
    const row = rows[i], res = results[i];
    const drawn = !!(row && !row.hidden && res && !res.error && kindOf(res.value) === 'vec' && sameOrigin(res.origin, it.o));
    return { name: a.name, color: row?.color ?? null, drawn };
  });
}

function extentPoints(it) {
  const cb = it.combo;
  switch (it.kind) {
    case 'cb-explain': {
      const pts = [];
      cb.cols.reduce((s, c, j) => { const t = add3(s, scale3(c, cb.coeffs[j])); pts.push(t); return t; }, [0, 0, 0]);
      return pts;
    }
    case 'cb-chain': {
      const pts = [];
      cb.parts.reduce((s, p) => { const t = add3(s, p); pts.push(t); return t; }, [0, 0, 0]);
      return pts;
    }
    case 'cb-target': return [it.v, cb.b];
    case 'cb-shadow': return [it.v, cb.u];
    case 'cb-crossview': return [it.v, cb.u, cb.w, add3(cb.u, cb.w)];
    case 'cb-line': return it.pts ?? [it.p];
    case 'cb-plane': return it.pts ?? [];
    case 'cb-dist': return [it.p, it.foot];
    default: return [it.v ?? [0, 0, 0]];
  }
}

const OURS = new Set(['cb-arc', 'cb-line', 'cb-plane', 'cb-dist']);
function tagItems(items, { rows, results }) {
  return items.map((it) => {
    if (!it.combo?.mode && !OURS.has(it.kind)) return it;
    const out = { ...it, kind: it.combo?.mode ? `cb-${it.combo.mode}` : it.kind };
    out.args = argInfo(out, rows, results);
    out.extentPoints = extentPoints(out);
    return out;
  });
}

function vectorsOf(it) {
  if (it.type !== 'vec' || !Array.isArray(it.v)) return [];
  const o = it.o || [0, 0, 0], out = [{ o, v: it.v }];
  if (it.combo?.mode === 'crossview') out.push({ o, v: it.combo.u }, { o, v: it.combo.w });
  return out;
}

// ---------------------------------------------------------------- scene (browser)

let TEX = null;
function textures(THREE) {
  if (TEX) return TEX;
  const make = (draw) => {
    const cv = document.createElement('canvas');
    cv.width = cv.height = 128;
    draw(cv.getContext('2d'));
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  const circle = (g, r) => { g.beginPath(); g.arc(64, 64, r, 0, 2 * Math.PI); };
  TEX = {
    ring: make((g) => {
      g.strokeStyle = '#fff';
      g.lineWidth = 8; circle(g, 56); g.stroke();
      g.lineWidth = 6; circle(g, 32); g.stroke();
      g.fillStyle = '#fff'; circle(g, 9); g.fill();
    }),
    glow: make((g) => {
      const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      r.addColorStop(0, 'rgba(255,255,255,0.95)');
      r.addColorStop(0.35, 'rgba(255,255,255,0.45)');
      r.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = r;
      g.fillRect(0, 0, 128, 128);
    }),
    dot: make((g) => { g.fillStyle = '#fff'; circle(g, 58); g.fill(); }),
  };
  return TEX;
}

function installRenderers(registerRenderer, trails) {
  const SM = 'g-label cb-sm';

  const rod = (c, a, b, color, r, opacity) => {
    const d = b.clone().sub(a), L = d.length();
    if (L < 1e-9) return null;
    const m = new c.THREE.Mesh(c.GEO.cyl, opacity != null ? c.mat('glass', color, opacity) : c.mat('solid', color));
    c.placeAlong(m, a, d, L, r);
    c.add(m);
    return m;
  };
  const longLine = (c, p, d, color, r, opacity) => {
    const t = clipLine(p.toArray(), d.toArray(), 1.1 * c.E);
    if (!t) return null;
    rod(c, p.clone().addScaledVector(d, t[0]), p.clone().addScaledVector(d, t[1]), color, r, opacity);
    return t;
  };
  const tipLabel = (c, o, v, latex, color, cls) => {
    const tip = o.clone().add(v), L = v.length();
    if (L > 1e-9) tip.addScaledVector(v, (0.3 * c.s) / L);
    c.label(tip, latex, color, cls);
  };
  const midLabel = (c, from, v, latex, color) => {
    const L = v.length();
    if (L < 1e-9) return;
    // nudge sideways (roughly screen-up) so the label doesn't sit on the shaft
    const side = new c.THREE.Vector3(0, 0, 1).cross(v).cross(v).negate().normalize();
    if (side.lengthSq() < 0.5) side.set(0, 1, 0);
    c.label(from.clone().addScaledVector(v, 0.5).addScaledVector(side, 0.28 * c.s), latex, color, SM);
  };
  const rightMark = (c, corner, a, b, size, color, opacity = 0.9) => {
    const A = a.clone().normalize().multiplyScalar(size), B = b.clone().normalize().multiplyScalar(size);
    const p1 = corner.clone().add(A), p2 = p1.clone().add(B), p3 = corner.clone().add(B);
    const r = 0.011 * c.s;
    const legMat = c.mat('glass', color, opacity);
    for (const [x, y] of [[p1, p2], [p2, p3]]) {
      const m = new c.THREE.Mesh(c.GEO.cyl, legMat);
      c.placeAlong(m, x, y.clone().sub(x), size, r);
      c.add(m);
    }
    const g = c.own(new c.THREE.BufferGeometry().setFromPoints([corner, p1, p2, corner, p2, p3]));
    c.add(new c.THREE.Mesh(g, c.mat('surface', color, 0.16)));
  };
  // The item's own vector (or point) with its row label, drawn like a plain row.
  const drawMain = (c, it, opts = {}) => {
    const o = c.v3(it.o), v = c.v3(it.v);
    if (it.type === 'point') {
      const p = o.clone().add(v);
      c.dot(p, it.color, 0.08 * c.s);
      if (it.label) c.label(p.clone().add(new c.THREE.Vector3(0, 0, 0.3 * c.s)), nameTex(it.label, false), it.color);
      return;
    }
    c.arrow(o, v, it.color, opts);
    const latex = opts.latex ?? (it.label ? nameTex(it.label, true) : null);
    if (latex) tipLabel(c, o, v, latex, it.color, opts.cls);
  };
  const argTex = (arg, fallback) => (arg?.name ? nameTex(arg.name, true) : fallback);
  const colName = (it) => {
    const n = it.args?.[0]?.name;
    return n && /^[A-Z]$/.test(n) ? n.toLowerCase() : 'a';
  };

  registerRenderer('cb-explain', (it, c) => {
    const { cols, coeffs } = it.combo, o = c.v3(it.o), tones = shades(it.color, cols.length, c.theme);
    const a = colName(it);
    cols.forEach((col, j) => c.arrow(o, c.v3(col), tones[j], { opacity: 0.22, thickness: 0.6, head: 0.7 }));
    const at = o.clone();
    cols.forEach((col, j) => {
      const piece = c.v3(col).multiplyScalar(coeffs[j]);
      if (piece.length() < 1e-9) return;
      c.arrow(at, piece, tones[j], { thickness: 0.85 });
      midLabel(c, at, piece, `${short(coeffs[j])}\\,\\vec ${a}_{${j + 1}}`, tones[j]);
      at.add(piece);
    });
    const [A, x] = it.args ?? [];
    const latex = it.label ? nameTex(it.label, true)
      : A?.name && x?.name ? `${nameTex(A.name, false)}${nameTex(x.name, true)}` : null;
    drawMain(c, it, { thickness: 1.2, latex });
  });

  registerRenderer('cb-chain', (it, c) => {
    const { parts } = it.combo, o = c.v3(it.o), tones = shades(it.color, parts.length, c.theme);
    const at = o.clone();
    parts.forEach((p, j) => {
      const piece = c.v3(p), arg = it.args?.[j], color = arg?.color ?? tones[j];
      if (piece.length() < 1e-9) return;
      c.arrow(at, piece, color, { thickness: 0.9 });
      if (arg?.name) midLabel(c, at, piece, nameTex(arg.name, true), color);
      at.add(piece);
    });
    const names = (it.args ?? []).map((a) => a.name);
    const latex = it.label ? nameTex(it.label, true)
      : names.length > 1 && names.length <= 4 && names.every(Boolean) ? names.map((n) => nameTex(n, true)).join('+') : null;
    drawMain(c, it, { opacity: 0.4, thickness: 0.8, latex, cls: SM });
  });

  registerRenderer('cb-trail', (it, c) => {
    drawMain(c, it);
    const tr = trails.get(it.rowId);
    if (!tr || tr.pts.length < 2) return;
    const { THREE } = c, n = tr.pts.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const bg = new THREE.Color(c.colors.bg), fg = new THREE.Color(it.color), tmp = new THREE.Color();
    const idx = [], jump = 0.25 * c.E;
    tr.pts.forEach((p, i) => {
      pos.set(p, 3 * i);
      tmp.lerpColors(bg, fg, 0.25 + (0.75 * (i + 1)) / n);
      col.set([tmp.r, tmp.g, tmp.b], 3 * i);
      if (i && len3(sub3(p, tr.pts[i - 1])) < jump) idx.push(i - 1, i);
    });
    const g = c.own(new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const dots = new THREE.Points(g, c.own(new THREE.PointsMaterial({
      size: 5, sizeAttenuation: false, map: textures(THREE).dot, vertexColors: true,
      transparent: true, alphaTest: 0.3, depthWrite: false,
    })));
    c.add(dots);
    if (idx.length) {
      const lg = c.own(new THREE.BufferGeometry());
      lg.setAttribute('position', g.getAttribute('position'));
      lg.setAttribute('color', g.getAttribute('color'));
      lg.setIndex(idx);
      c.add(new THREE.LineSegments(lg, c.own(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6, depthWrite: false }))));
    }
  });

  registerRenderer('cb-target', (it, c) => {
    const { THREE } = c, o = c.v3(it.o), T = o.clone().add(c.v3(it.combo.b)), tex = textures(THREE);
    const hit = it.combo.hit, [h, s, l] = hexToHsl(it.color);
    const lit = hslToHex(h, Math.min(1, s + 0.1), Math.min(0.72, l + 0.12));
    const bArg = it.args?.[0];
    if (!bArg?.drawn) c.lines([o, T], it.color, { dashed: true, opacity: hit ? 0.9 : 0.6 });
    const ring = new THREE.Sprite(c.own(new THREE.SpriteMaterial({
      map: tex.ring, color: hit ? lit : it.color, transparent: true, opacity: hit ? 1 : 0.7, depthWrite: false,
    })));
    ring.position.copy(T);
    ring.scale.set(0.8 * c.s, 0.8 * c.s, 1);
    c.add(ring);
    const guessTip = o.clone().add(c.v3(it.v));
    if (hit) {
      const glow = new THREE.Sprite(c.own(new THREE.SpriteMaterial({
        map: tex.glow, color: lit, transparent: true, opacity: 0.85, depthWrite: false,
      })));
      glow.position.copy(T);
      c.add(glow);
      c.onFrame((dt, t) => {
        const k = (1.5 + 0.25 * Math.sin(t * 5)) * c.s;
        glow.scale.set(k, k, 1);
      });
      glow.scale.set(1.5 * c.s, 1.5 * c.s, 1);
    } else if (guessTip.distanceTo(T) > 1e-9) {
      c.lines([guessTip, T], c.colors.fg, { dashed: true, opacity: 0.45 });
    }
    if (bArg?.name && !bArg.drawn) c.label(T.clone().add(new THREE.Vector3(0, 0, 0.55 * c.s)), nameTex(bArg.name, true), it.color, SM);
    drawMain(c, it);
  });

  registerRenderer('cb-shadow', (it, c) => {
    const { u, onto } = it.combo, o = c.v3(it.o), U = c.v3(u), W = c.v3(onto), p = c.v3(it.v);
    const [aU, aW] = it.args ?? [], tones = shades(it.color, 2, c.theme);
    longLine(c, o, W, aW?.color ?? tones[1], 0.012 * c.s, 0.35);
    if (!aU?.drawn) c.arrow(o, U, aU?.color ?? tones[0], { opacity: 0.55, thickness: 0.8 });
    if (!aW?.drawn) c.arrow(o, W, aW?.color ?? tones[1], { opacity: 0.55, thickness: 0.8 });
    const tip = o.clone().add(U), foot = o.clone().add(p), drop = tip.clone().sub(foot);
    const dropped = drop.length() > 1e-6 * Math.max(1, U.length());
    if (dropped) {
      c.lines([tip, foot], it.color, { dashed: true, opacity: 0.9 });
      const leg = p.length() > 1e-9 ? p.clone().negate() : W.clone();
      const size = Math.min(0.26 * c.s, 0.4 * drop.length(), p.length() > 1e-9 ? 0.4 * p.length() : Infinity);
      rightMark(c, foot, drop, leg, size, c.colors.fg);
    }
    const latex = it.label ? nameTex(it.label, true)
      : aU?.name && aW?.name ? `\\mathrm{proj}_{${nameTex(aW.name, true)}}${nameTex(aU.name, true)}` : null;
    if (!dropped || p.length() < 1e-9 || !latex) { drawMain(c, it, { thickness: 1.15, latex }); return; }
    // the label sits across the line from u, by the foot: clear of u's label and of the onto vector's
    c.arrow(o, p, it.color, { thickness: 1.15 });
    c.label(foot.clone().addScaledVector(drop.normalize(), -0.45 * c.s), latex, it.color);
  });

  registerRenderer('cb-components', (it, c) => {
    const o = c.v3(it.o), P = it.v, tones = shades(it.color, 3, c.theme);
    const corner = (i, j, k) => o.clone().add(new c.THREE.Vector3(i * P[0], j * P[1], k * P[2]));
    const pairs = [], seen = new Set();
    const key = (v) => v.toArray().map((x) => x.toFixed(6)).join(',');
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) {
      if (!i && !j && !k) continue; // edges out of the origin are the component arrows
      for (const [di, dj, dk] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
        if ((di && i) || (dj && j) || (dk && k)) continue;
        const A = corner(i, j, k), B = corner(i + di, j + dj, k + dk);
        const id = [key(A), key(B)].sort().join('|');
        if (A.distanceTo(B) < 1e-9 || seen.has(id)) continue;
        seen.add(id);
        pairs.push(A, B);
      }
    }
    if (pairs.length) c.lines(pairs, it.color, { dashed: true, opacity: 0.75 });
    const hats = ['\\hat\\imath', '\\hat\\jmath', '\\hat k'];
    const axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    P.forEach((x, a) => {
      if (Math.abs(x) < 1e-9) return;
      const piece = c.v3(axes[a]).multiplyScalar(x);
      c.arrow(o, piece, tones[a], { thickness: 0.75, head: 0.85 });
      // label beside the middle of the component, on the side away from the box
      const off = new c.THREE.Vector3(...axes[(a + 1) % 3]).multiplyScalar(-Math.sign(P[(a + 1) % 3] || 1) * 0.38 * c.s);
      c.label(o.clone().addScaledVector(piece, 0.5).add(off), `${short(x)}\\,${hats[a]}`, tones[a], SM);
    });
    drawMain(c, it, { thickness: 1.1 });
  });

  registerRenderer('cb-crossview', (it, c) => {
    const { THREE } = c, { u, w } = it.combo, o = c.v3(it.o), U = c.v3(u), W = c.v3(w), N = c.v3(it.v);
    const [aU, aW] = it.args ?? [], tones = shades(it.color, 3, c.theme);
    const cu = aU?.color ?? tones[0], cw = aW?.color ?? tones[1];
    const A = o, B = o.clone().add(U), C = B.clone().add(W), D = o.clone().add(W);
    if (N.length() > 1e-9) {
      const g = c.own(new THREE.BufferGeometry().setFromPoints([A, B, C, A, C, D]));
      c.add(new THREE.Mesh(g, c.mat('surface', it.color, 0.22)));
      c.polyline([B, C, D], it.color, { opacity: 0.7 });
    }
    if (!aU?.drawn) { c.arrow(o, U, cu); tipLabel(c, o, U, argTex(aU, '\\vec u'), cu); }
    if (!aW?.drawn) { c.arrow(o, W, cw); tipLabel(c, o, W, argTex(aW, '\\vec v'), cw); }
    const latex = it.label ? nameTex(it.label, true) : `${argTex(aU, '\\vec u')}\\times ${argTex(aW, '\\vec v')}`;
    drawMain(c, it, { latex });
    if (N.length() < 1e-9 || U.length() < 1e-9) return;
    // right-hand-rule cue: a curled arrow around u×v turning from u toward v
    const n = N.clone().normalize(), e1 = U.clone().sub(n.clone().multiplyScalar(U.dot(n))).normalize();
    const e2 = n.clone().cross(e1);
    const r = Math.max(0.22 * c.s, Math.min(0.42 * c.s, 0.3 * Math.min(U.length(), W.length())));
    const center = o.clone().addScaledVector(n, Math.min(0.45 * N.length(), 1.4 * c.s));
    const end = 1.5 * Math.PI, pts = [];
    for (let k = 0; k <= 40; k++) {
      const t = (end * k) / 40;
      pts.push(center.clone().addScaledVector(e1, r * Math.cos(t)).addScaledVector(e2, r * Math.sin(t)));
    }
    const cue = tones[2];
    const tube = c.own(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 48, 0.022 * c.s, 8, false));
    c.add(new THREE.Mesh(tube, c.mat('solid', cue)));
    const tangent = e1.clone().multiplyScalar(-Math.sin(end)).addScaledVector(e2, Math.cos(end));
    const head = new THREE.Mesh(c.GEO.cone, c.mat('solid', cue));
    c.placeAlong(head, pts.at(-1), tangent, 0.2 * c.s, 0.075 * c.s);
    c.add(head);
  });

  registerRenderer('cb-arc', (it, c) => {
    const { THREE } = c, o = c.v3(it.o), U = c.v3(it.u), W = c.v3(it.v), theta = it.theta;
    const e1 = U.clone().normalize();
    let e2 = W.clone().sub(e1.clone().multiplyScalar(W.dot(e1)));
    e2 = e2.length() > 1e-9 * W.length() ? e2.normalize() : c.v3(anyPerp(e1.toArray()));
    const r = Math.max(0.3 * c.s, Math.min(0.9 * c.s, 0.4 * Math.min(U.length(), W.length())));
    const dir = (t) => e1.clone().multiplyScalar(Math.cos(t)).addScaledVector(e2, Math.sin(t));
    if (Math.abs(theta - Math.PI / 2) < 1e-4) {
      rightMark(c, o, e1, e2, 0.7 * r, it.color);
    } else if (theta > 1e-4) {
      const N = Math.max(8, Math.ceil(theta * 24)), pts = [], fan = [];
      for (let k = 0; k <= N; k++) pts.push(o.clone().addScaledVector(dir((theta * k) / N), r));
      for (let k = 0; k < N; k++) fan.push(o, pts[k], pts[k + 1]);
      c.add(new THREE.Mesh(c.own(new THREE.BufferGeometry().setFromPoints(fan)), c.mat('surface', it.color, 0.2)));
      const tube = c.own(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 2 * N, 0.016 * c.s, 6, false));
      c.add(new THREE.Mesh(tube, c.mat('solid', it.color)));
    }
    c.label(o.clone().addScaledVector(dir(theta / 2), r + 0.42 * c.s), deg(theta), it.color, SM);
  });

  registerRenderer('cb-line', (it, c) => {
    const o = c.v3(it.o), p = o.clone().add(c.v3(it.p)), d = c.v3(it.d);
    const t = longLine(c, p, d, it.color, 0.02 * c.s);
    const pts = (it.pts ?? []).map((q) => o.clone().add(c.v3(q)));
    if (!t && pts.length === 2) rod(c, pts[0], pts[1], it.color, 0.02 * c.s);
    for (const q of pts) c.dot(q, it.color, 0.075 * c.s);
    if (it.label) {
      const at = (pts[1] ?? p).clone().addScaledVector(d, (0.5 * c.s) / d.length());
      c.label(at.add(new c.THREE.Vector3(0, 0, 0.32 * c.s)), nameTex(it.label, false), it.color);
    }
  });

  registerRenderer('cb-plane', (it, c) => {
    const { THREE } = c, o = c.v3(it.o), n = c.v3(it.n), nn = n.lengthSq();
    if (nn < 1e-18) return;
    const center = o.clone().addScaledVector(n, it.c / nn), nh = n.clone().normalize();
    // the part of the plane inside the grid box, filled and outlined
    const poly = planeInBox(it.n, it.c + dot3(it.n, it.o), c.E).map((p) => c.v3(p));
    if (poly.length) {
      const mid = poly.reduce((s, q) => s.add(q), new THREE.Vector3()).divideScalar(poly.length), fan = [];
      poly.forEach((q, k) => fan.push(mid, q, poly[(k + 1) % poly.length]));
      c.add(new THREE.Mesh(c.own(new THREE.BufferGeometry().setFromPoints(fan)), c.mat('surface', it.color, 0.2)));
      c.polyline([...poly, poly[0]], it.color, { opacity: 0.75 });
    }
    const pts = (it.pts ?? []).map((q) => o.clone().add(c.v3(q)));
    if (pts.length === 3) {
      c.polyline([...pts, pts[0]], it.color, { opacity: 0.8 });
      for (const q of pts) c.dot(q, it.color, 0.075 * c.s);
    }
    if (it.label) {
      const at = pts.length ? pts.reduce((s, q) => s.add(q), new c.THREE.Vector3()).divideScalar(pts.length) : center;
      c.label(at.addScaledVector(nh, 0.35 * c.s), nameTex(it.label, false), it.color);
    }
  });

  registerRenderer('cb-dist', (it, c) => {
    const o = c.v3(it.o), P = o.clone().add(c.v3(it.p)), F = o.clone().add(c.v3(it.foot));
    c.dot(P, it.color, 0.075 * c.s);
    if (it.d < 1e-9) return;
    rod(c, P, F, it.color, 0.022 * c.s);
    c.dot(F, it.color, 0.05 * c.s);
    if (it.along) rightMark(c, F, P.clone().sub(F), c.v3(it.along), Math.min(0.26 * c.s, 0.4 * it.d), c.colors.fg);
    const mid = P.clone().add(F).multiplyScalar(0.5), perp = P.clone().sub(F);
    const side = new c.THREE.Vector3(0, 0, 1).cross(perp);
    if (side.lengthSq() < 1e-12) side.set(0.7, -0.7, 0);
    c.label(mid.addScaledVector(side.normalize(), 0.45 * c.s), `d = ${short(it.d)}`, it.color, SM);
  });

  registerRenderer('cb-right', (it, c) => {
    rightMark(c, c.v3(it.o), c.v3(it.a), c.v3(it.b), Math.min(0.36 * c.s, 0.3 * it.size), c.colors.fg, 0.85);
  });
}

// ---------------------------------------------------------------- install

const STORE = 'mathboard.combos';
const CSS = `
.g-label.cb-sm { font-size: 17px; }
#g-tools button.on { border-color: var(--accent); }
`;
const HELP = `<p><code>explain(A, v)</code> A v as a combination of columns &middot; <code>chain(u, v, w)</code> tip-to-tail
&middot; <code>trail(a u + b v)</code> leaves a trail as sliders move &middot; <code>target(b, a u + b v)</code> find the combination</p>
<p><code>arc(u,v)</code> <code>shadow(u,v)</code> <code>components(u)</code> <code>crossview(u,v)</code>
<code>line(p,q)</code> <code>plane3(p,q,r)</code> <code>intersect(X,Y)</code> <code>distance(p,X)</code></p>`;

// Slider rows that the guess expression depends on, e.g. "a = 2, b = 1".
function sliderSettings(call, rows, results) {
  const found = new Map(), seen = new Set();
  const walk = (n, depth) => {
    if (!n || depth > 6) return;
    if (n.t === 'name' && !seen.has(n.name)) {
      seen.add(n.name);
      const i = results.findIndex((r) => r?.name === n.name);
      if (i < 0) return;
      if (results[i].slider != null) found.set(n.name, results[i].slider);
      else walk(parseLine(rows[i].src)?.body, depth + 1);
      return;
    }
    for (const k of [n.a, n.b, ...(n.items ?? []), ...(n.args ?? []), ...(n.rows?.flat() ?? [])]) walk(k, depth);
  };
  walk(call?.args?.[1], 0);
  return [...found].map(([k, v]) => `${k} = ${num(v)}`).join(', ');
}

export async function install(api) {
  const { registerRenderer } = await import('../scene.js');
  const trails = new Map(); // rowId -> {src, pts}
  const hits = new Map();   // rowId -> last hit state of a target row
  let primed = false;
  let prefs = {};
  try { prefs = JSON.parse(localStorage.getItem(STORE)) || {}; } catch { /* defaults */ }
  let rightAngles = prefs.rightAngles !== false;

  installRenderers(registerRenderer, trails);
  api.addStyles(CSS);
  const help = document.getElementById('g-help');
  if (help) help.insertAdjacentHTML('beforeend', HELP);

  api.onRecompute((results, rows) => {
    const live = new Set(rows.map((r) => r.id));
    for (const id of trails.keys()) if (!live.has(id)) trails.delete(id);
    for (const id of hits.keys()) if (!live.has(id)) hits.delete(id);
    rows.forEach((r, i) => {
      const res = results[i], v = res?.value;
      if (!v || res.error || !v.combo) return;
      if (v.combo.mode === 'trail') pushTrail(trails, r.id, r.src, add3(res.origin || [0, 0, 0], v.v));
      if (v.combo.mode === 'target') {
        const was = hits.get(r.id);
        hits.set(r.id, v.combo.hit);
        if (v.combo.hit && !was && primed) {
          const set = sliderSettings(callOf(r.src, rows, results), rows, results);
          api.toast(set ? `On target with ${set}` : 'On target!');
        }
      }
    });
    primed = true;
  });

  api.addItemsHook(tagItems);
  // Markers look at the final item list, so register after every feature's install.
  api.onSceneReady(() => api.addItemsHook((items) => {
    if (!rightAngles) return items;
    const marks = perpendicularPairs(items.flatMap(vectorsOf));
    return [...items, ...marks.map((m) => ({ ...m, kind: 'cb-right', color: null, label: null, index: -1, rowId: null }))];
  }));

  api.addRowDecorator((row, res, el) => {
    if (!res || res.error || !res.value || typeof katex === 'undefined') return;
    const tex = readoutTex(res.value, row.color, document.documentElement.dataset.theme);
    if (tex) el.out.innerHTML = katex.renderToString(tex, { throwOnError: false });
  });
  // explain's readout is coloured with the theme's shades
  new MutationObserver(() => { if (api.results.some((r) => r?.value?.combo?.mode === 'explain')) api.recompute(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  api.addToolbarButton({
    label: 'Clear trails',
    title: 'Erase the trails left by trail(...) rows',
    onClick: () => { trails.clear(); api.scene?.rebuild(); },
  });
  const btn = api.addToolbarButton({
    label: '&#8735; Right angles',
    title: 'Mark perpendicular vectors that start at the same point',
    onClick: () => {
      rightAngles = !rightAngles;
      btn.classList.toggle('on', rightAngles);
      try { localStorage.setItem(STORE, JSON.stringify({ ...prefs, rightAngles })); } catch { /* private mode */ }
      api.recompute();
    },
  });
  btn.classList.toggle('on', rightAngles);
}
