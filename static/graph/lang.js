// Expression language for the 3D vector grapher.
// evaluate(lines) turns editor lines into rows of {name, value, origin, error, slider}.
// Pure ES module, no dependencies; runs in browsers and Node.

const NAME = '[A-Za-z\\u0370-\\u03FF][A-Za-z0-9_\\u0370-\\u03FF]*';
const NAME_RE = new RegExp(NAME, 'y');
const DEF_RE = new RegExp(`^(${NAME})\\s*=`);
const FNDEF_RE = new RegExp(`^(${NAME})\\s*\\(\\s*(${NAME}(?:\\s*,\\s*${NAME})*)?\\s*\\)\\s*=`); // f(x) = ..., g(x, y) = ...
const NUM_RE = /(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/y;
const OPS = "+-*/·×^()[],;|=@.'{}<>";
const OP_ALIASES = { '−': '-', '⋅': '·', '∙': '·', '′': "'", '≤': '<=', '≥': '>=' };
// Comparisons, only inside a restriction: y = -ln(x) {0 < x <= 1}
const RELS = { '<': (a, b) => a < b, '>': (a, b) => a > b, '<=': (a, b) => a <= b, '>=': (a, b) => a >= b };
// Coordinates: a row that uses x, y or z without defining them is a graph (y = x^2, z = x^2 - y^2).
const COORDS = ['x', 'y', 'z'];

// ---------- values ----------

const KIND = {
  num: 'a number', vec: 'a vector', point: 'a point', mat: 'a matrix', span: 'a span',
  plane: 'a plane', parallelogram: 'a parallelogram', parallelepiped: 'a parallelepiped',
};
const PLURAL = { num: 'numbers', vec: 'vectors', mat: 'matrices' };

const kindOf = (x) => (typeof x === 'number' ? 'num' : x?.type);
const describe = (x) =>
  kindOf(x) === 'mat' ? `a ${x.m.length}×${x.m[0].length} matrix` : KIND[kindOf(x)] ?? 'an unknown value';

const vec = (a) => ({ type: 'vec', v: [a[0], a[1], a[2] ?? 0] });
const point = (a) => ({ type: 'point', v: [a[0], a[1], a[2] ?? 0] });
const mat = (m) => ({ type: 'mat', m });
const isVecLike = (x) => kindOf(x) === 'vec' || kindOf(x) === 'point';
// A column with 2-3 entries is a vector (so (u^T)^T = u and A*[[1],[2],[3]] draws as an arrow).
const colToVec = (m) => (m[0].length === 1 && (m.length === 2 || m.length === 3) ? vec(m.map((r) => r[0])) : mat(m));

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const clamp1 = (x) => Math.max(-1, Math.min(1, x));

const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const matMul = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((s, x, k) => s + x * b[k][j], 0)));

function square(A, what) {
  if (A.m.length !== A.m[0].length) throw new Error(`${what} needs a square matrix, got ${describe(A)}`);
  return A.m;
}

function det(m) {
  const n = m.length;
  if (n === 1) return m[0][0];
  if (n === 2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (n === 3) return dot3(m[0], cross3(m[1], m[2]));
  const a = m.map((r) => r.slice());
  let d = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (a[p][c] === 0) return 0;
    if (p !== c) [a[p], a[c], d] = [a[c], a[p], -d];
    d *= a[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = a[r][c] / a[c][c];
      for (let k = c; k < n; k++) a[r][k] -= f * a[c][k];
    }
  }
  return d;
}

// Gauss-Jordan with partial pivoting; pivots below a relative tolerance count as singular.
function inverse(m) {
  const n = m.length;
  const tol = 1e-12 * Math.max(...m.flat().map(Math.abs));
  const a = m.map((r, i) => [...r, ...identity(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (!(Math.abs(a[p][c]) > tol)) throw new Error('matrix is not invertible (det = 0)');
    [a[p], a[c]] = [a[c], a[p]];
    const piv = a[c][c];
    a[c] = a[c].map((x) => x / piv);
    for (let r = 0; r < n; r++) {
      if (r === c || a[r][c] === 0) continue;
      const f = a[r][c];
      a[r] = a[r].map((x, k) => x - f * a[c][k]);
    }
  }
  return a.map((r) => r.slice(n));
}

// Greedy linearly independent subset (original vectors kept, dependent ones dropped).
function independent(vs) {
  const basis = [];
  const ortho = [];
  for (const v of vs) {
    let r = v.slice();
    for (const q of ortho) {
      const d = dot3(r, q);
      r = r.map((x, i) => x - d * q[i]);
    }
    const L = len3(r);
    if (L > 1e-9 * len3(v)) {
      ortho.push(r.map((x) => x / L));
      basis.push(v.slice());
    }
  }
  return basis;
}

function numbersIn(v) {
  switch (kindOf(v)) {
    case 'num': return [v];
    case 'vec': case 'point': return v.v;
    case 'mat': return v.m.flat();
    case 'span': return v.vecs.flat();
    case 'plane': return v.normal;
    case 'parallelogram': return [...v.u, ...v.v];
    case 'parallelepiped': return [...v.u, ...v.v, ...v.w];
    default: return TYPES.get(kindOf(v))?.numbers?.(v) ?? [];
  }
}

function finite(v) {
  for (const x of numbersIn(v)) {
    if (Number.isNaN(x)) throw new Error('result is undefined');
    if (!Number.isFinite(x)) throw new Error('result is too large');
  }
  return v;
}

// ---------- operators ----------

function add(a, b, sign) {
  const ka = kindOf(a), kb = kindOf(b);
  if (ka === 'num' && kb === 'num') return a + sign * b;
  if (isVecLike(a) && isVecLike(b)) {
    const r = a.v.map((x, i) => x + sign * b.v[i]);
    if (ka === 'vec' && kb === 'vec') return vec(r);
    if (kb === 'vec') return point(r); // point ± vec
    if (sign < 0 && ka === 'point') return vec(r); // point − point
    if (sign > 0 && ka === 'vec') return point(r); // vec + point
    if (sign > 0) throw new Error("can't add two points (subtract them to get a vector)");
  }
  if (ka === 'mat' && kb === 'mat' && a.m.length === b.m.length && a.m[0].length === b.m[0].length) {
    return mat(a.m.map((row, i) => row.map((x, j) => x + sign * b.m[i][j])));
  }
  throw new Error(sign > 0
    ? `can't add ${describe(a)} and ${describe(b)}`
    : `can't subtract ${describe(b)} from ${describe(a)}`);
}

function matVec(A, u) {
  const rows = A.m.length, cols = A.m[0].length;
  let x;
  if (cols === 3) x = u.v;
  else if (cols === 2 && u.v[2] === 0) x = u.v.slice(0, 2);
  else if (cols === 2) throw new Error(`can't multiply ${describe(A)} by a vector with nonzero z`);
  if (!x || rows > 3) throw new Error(`can't multiply ${describe(A)} by a vector`);
  const out = A.m.map((row) => row.reduce((s, a, k) => s + a * x[k], 0));
  return rows === 1 ? out[0] : vec(out); // a row times a vector is a number: u^T v
}

function mul(a, b) {
  const ka = kindOf(a), kb = kindOf(b);
  if (ka === 'num' && kb === 'num') return a * b;
  if (ka === 'num' && kb === 'vec') return vec(b.v.map((x) => a * x));
  if (ka === 'vec' && kb === 'num') return vec(a.v.map((x) => x * b));
  if (ka === 'num' && kb === 'mat') return mat(b.m.map((r) => r.map((x) => a * x)));
  if (ka === 'mat' && kb === 'num') return mat(a.m.map((r) => r.map((x) => x * b)));
  if (ka === 'mat' && kb === 'vec') return matVec(a, b);
  if (ka === 'mat' && kb === 'mat') {
    if (a.m[0].length !== b.m.length) {
      throw new Error(`can't multiply ${describe(a)} by ${describe(b)} (${a.m[0].length} columns vs ${b.m.length} rows)`);
    }
    const P = matMul(a.m, b.m);
    return P.length === 1 && P[0].length === 1 ? P[0][0] : colToVec(P); // 1×1 results (x^T A x) are numbers
  }
  if (ka === 'vec' && kb === 'vec') throw new Error("can't multiply two vectors with *; use dot(u, v) or cross(u, v)");
  if (ka === 'vec' && kb === 'mat') {
    if (b.m.length === 1) return mat(a.v.map((x) => b.m[0].map((y) => x * y))); // u v^T: outer product
    throw new Error("can't multiply a vector by a matrix; put the matrix first");
  }
  throw new Error(`can't multiply ${describe(a)} by ${describe(b)}`);
}

function div(a, b) {
  if (kindOf(b) !== 'num') throw new Error(`can't divide by ${describe(b)}`);
  if (b === 0) throw new Error('division by zero');
  const ka = kindOf(a);
  if (ka === 'num') return a / b;
  if (ka === 'vec') return vec(a.v.map((x) => x / b));
  if (ka === 'mat') return mat(a.m.map((r) => r.map((x) => x / b)));
  throw new Error(`can't divide ${describe(a)} by a number`);
}

function pow(a, b) {
  if (kindOf(b) !== 'num') throw new Error(`exponents must be numbers, got ${describe(b)}`);
  const ka = kindOf(a);
  if (ka === 'num') {
    if (a < 0 && !Number.isInteger(b)) throw new Error("can't raise a negative number to a fractional power");
    if (a === 0 && b < 0) throw new Error('division by zero');
    return a ** b;
  }
  if (ka === 'mat') {
    if (a.m.length !== a.m[0].length) throw new Error(`only square matrices have powers, got ${describe(a)}`);
    if (!Number.isInteger(b)) throw new Error('matrix powers must be whole numbers');
    let base = b < 0 ? inverse(a.m) : a.m;
    let result = identity(a.m.length);
    for (let e = Math.abs(b); e > 0; e = Math.floor(e / 2)) {
      if (e % 2) result = matMul(result, base);
      if (e > 1) base = matMul(base, base);
    }
    return mat(result);
  }
  if (ka === 'vec') throw new Error("can't raise a vector to a power; try dot(u, u) or |u|^2");
  throw new Error(`can't raise ${describe(a)} to a power`);
}

function neg(a) {
  switch (kindOf(a)) {
    case 'num': return -a;
    case 'vec': return vec(a.v.map((x) => -x));
    case 'mat': return mat(a.m.map((r) => r.map((x) => -x)));
    default: throw new Error(`can't negate ${describe(a)}`);
  }
}

function magnitude(x, what) {
  if (kindOf(x) === 'num') return Math.abs(x);
  if (kindOf(x) === 'vec') return len3(x.v);
  const hint = kindOf(x) === 'mat' && what === '|...|' ? '; use det(A) for a determinant' : `, got ${describe(x)}`;
  throw new Error(`${what} needs a number or vector${hint}`);
}

function binary(op, a, b) {
  if ((op === '·' || op === '×') && isVecLike(a) && isVecLike(b) && !(kindOf(a) === 'vec' && kindOf(b) === 'vec')) {
    throw new Error(`${op} needs two vectors, got a point`);
  }
  switch (op) {
    case '+': return add(a, b, 1);
    case '-': return add(a, b, -1);
    case '*': return mul(a, b);
    case '/': return div(a, b);
    case '^': return pow(a, b);
    // · and × are dot/cross between vectors, plain multiplication otherwise
    case '·': return kindOf(a) === 'vec' && kindOf(b) === 'vec' ? dot3(a.v, b.v) : mul(a, b);
    case '×': return kindOf(a) === 'vec' && kindOf(b) === 'vec' ? vec(cross3(a.v, b.v)) : mul(a, b);
  }
  throw new Error(`unknown operator '${op}'`);
}

// ---------- builtins ----------

const numFn = (f, ok, msg) => ({
  n: 1, kind: 'num',
  f: ([x], name) => {
    if (ok && !ok(x)) throw new Error(`${name} ${msg}`);
    return f(x);
  },
});
const inUnit = (x) => Math.abs(x) <= 1 + 1e-12;
const norm = { n: 1, f: ([x], name) => magnitude(x, name) };
const unit = {
  n: 1, kind: 'vec',
  f: ([u]) => {
    const L = len3(u.v);
    if (L === 0) throw new Error("can't normalize the zero vector");
    return vec(u.v.map((x) => x / L));
  },
};

// Activations and friends (stable forms: no overflow for large |x|).
const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
const softplus = (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
function erf(x) { // Abramowitz and Stegun 7.1.26, |error| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
const leaky = (fallback, g) => ({ n: [1, 2], kind: 'num', f: ([x, a = fallback]) => (x >= 0 ? x : g(x, a)) });
const leakyrelu = leaky(0.01, (x, a) => a * x);
const silu = numFn((x) => x * sigmoid(x));

// softmax(v [, T]) is a probability vector; softmax(T) or a bare `softmax` row draws the map
// from logits to the probability triangle (a 'softmaxmap' value).
const SIMPLEX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const softmaxMap = (T) => ({ type: 'softmaxmap', T, extentPoints: SIMPLEX });
const softmax = {
  n: [1, 2], bare: () => softmaxMap(1),
  f: ([z, T], name) => {
    if (T !== undefined && !(typeof T === 'number' && T > 0)) throw new Error(`${name}'s temperature must be a positive number`);
    if (kindOf(z) === 'num') {
      if (T !== undefined) throw new Error(`${name} needs a vector of logits, got a number`);
      if (!(z > 0)) throw new Error(`${name}'s temperature must be a positive number`);
      return softmaxMap(z);
    }
    if (kindOf(z) !== 'vec') throw new Error(`${name} needs a vector of logits (or a temperature), got ${describe(z)}`);
    const t = T ?? 1, m = Math.max(...z.v), e = z.v.map((x) => Math.exp((x - m) / t)), s = e[0] + e[1] + e[2];
    return vec(e.map((x) => x / s));
  },
};

// n: exact argument count or [min, max]; kind: required type of every argument (checked before f runs).
// bare: the value of a row that is just the name (`softmax`); one-number functions plot as y = f(x).
const FUNCS = {
  dot: { n: 2, kind: 'vec', f: ([u, v]) => dot3(u.v, v.v) },
  cross: { n: 2, kind: 'vec', f: ([u, v]) => vec(cross3(u.v, v.v)) },
  norm, length: norm, mag: norm,
  unit, normalize: unit,
  proj: {
    n: 2, kind: 'vec',
    f: ([u, v]) => {
      const vv = dot3(v.v, v.v);
      if (vv === 0) throw new Error("can't project onto the zero vector");
      const s = dot3(u.v, v.v) / vv;
      return vec(v.v.map((x) => s * x));
    },
  },
  angle: {
    n: 2, kind: 'vec',
    f: ([u, v]) => {
      const d = len3(u.v) * len3(v.v);
      if (d === 0) throw new Error('angle with the zero vector is undefined');
      return Math.acos(clamp1(dot3(u.v, v.v) / d));
    },
  },
  deg: numFn((x) => (x * 180) / Math.PI),
  rad: numFn((x) => (x * Math.PI) / 180),
  sin: numFn(Math.sin),
  cos: numFn(Math.cos),
  tan: numFn(Math.tan),
  asin: numFn((x) => Math.asin(clamp1(x)), inUnit, 'needs a number between -1 and 1'),
  acos: numFn((x) => Math.acos(clamp1(x)), inUnit, 'needs a number between -1 and 1'),
  atan: numFn(Math.atan),
  sqrt: numFn(Math.sqrt, (x) => x >= 0, 'of a negative number is undefined'),
  abs: norm,
  exp: numFn(Math.exp),
  ln: numFn(Math.log, (x) => x > 0, 'needs a positive number'),
  log: numFn(Math.log10, (x) => x > 0, 'needs a positive number'),
  sinh: numFn(Math.sinh),
  cosh: numFn(Math.cosh),
  tanh: numFn(Math.tanh),
  sigmoid: numFn(sigmoid),
  σ: numFn(sigmoid),
  relu: numFn((x) => Math.max(0, x)),
  leakyrelu, leaky_relu: leakyrelu,
  elu: leaky(1, (x, a) => a * Math.expm1(x)),
  gelu: numFn((x) => 0.5 * x * (1 + erf(x / Math.SQRT2))),
  softplus: numFn(softplus),
  silu, swish: silu,
  mish: numFn((x) => x * Math.tanh(softplus(x))),
  erf: numFn(erf),
  heaviside: numFn((x) => (x > 0 ? 1 : x < 0 ? 0 : 0.5)),
  sign: numFn(Math.sign),
  floor: numFn(Math.floor),
  ceil: numFn(Math.ceil),
  round: numFn(Math.round),
  mod: {
    n: 2, kind: 'num',
    f: ([a, b]) => {
      if (b === 0) throw new Error('mod by zero');
      return a - b * Math.floor(a / b);
    },
  },
  softmax,
  logsumexp: {
    n: 1, kind: 'vec',
    f: ([z]) => {
      const m = Math.max(...z.v);
      return m + Math.log(z.v.reduce((s, x) => s + Math.exp(x - m), 0));
    },
  },
  min: { n: [1, Infinity], kind: 'num', f: (xs) => Math.min(...xs) },
  max: { n: [1, Infinity], kind: 'num', f: (xs) => Math.max(...xs) },
  det: { n: 1, kind: 'mat', f: ([A], name) => det(square(A, name)) },
  inv: { n: 1, kind: 'mat', f: ([A], name) => mat(inverse(square(A, name))) },
  transpose: {
    n: 1,
    f: ([A], name) => {
      if (kindOf(A) === 'vec') return mat([A.v.slice()]); // a vector becomes a 1×3 row
      if (kindOf(A) !== 'mat') throw new Error(`${name} needs a matrix or vector, got ${describe(A)}`);
      return colToVec(A.m[0].map((_, j) => A.m.map((r) => r[j])));
    },
  },
  matrix: {
    n: [2, 3], kind: 'vec',
    // two flat (z = 0) columns make a 2×2 matrix; otherwise 3 rows
    f: (cols) => {
      const rows = cols.length === 2 && cols.every((c) => c.v[2] === 0) ? 2 : 3;
      return mat(Array.from({ length: rows }, (_, i) => cols.map((c) => c.v[i])));
    },
  },
  point: {
    n: [1, 3],
    f: (args) => {
      if (args.length === 1) {
        if (isVecLike(args[0])) return point(args[0].v);
        throw new Error(`point needs x, y, z or a vector, got ${describe(args[0])}`);
      }
      for (const x of args) if (kindOf(x) !== 'num') throw new Error(`point needs numbers, got ${describe(x)}`);
      return point(args);
    },
  },
  span: { n: [1, 3], kind: 'vec', f: (vs) => ({ type: 'span', vecs: independent(vs.map((u) => u.v)) }) },
  plane: {
    n: 1, kind: 'vec',
    f: ([n]) => {
      if (len3(n.v) === 0) throw new Error('plane needs a nonzero normal vector');
      return { type: 'plane', normal: n.v.slice() };
    },
  },
  parallelogram: { n: 2, kind: 'vec', f: ([u, v]) => ({ type: 'parallelogram', u: u.v.slice(), v: v.v.slice() }) },
  parallelepiped: {
    n: 3, kind: 'vec',
    f: ([u, v, w]) => ({ type: 'parallelepiped', u: u.v.slice(), v: v.v.slice(), w: w.v.slice() }),
  },
};

const CONSTS = {
  pi: () => Math.PI,
  π: () => Math.PI,
  e: () => Math.E,
  i: () => vec([1, 0, 0]),
  j: () => vec([0, 1, 0]),
  k: () => vec([0, 0, 1]),
};

const isFunc = (name) => Object.hasOwn(FUNCS, name);

// ---------- extension API (used by graph/features/*) ----------

const TYPES = new Map(); // value type -> { describe, format, latex, drawable, numbers }

// spec: { n: exact count | [min, max], kind?: 'num'|'vec'|'mat'|<type>, f: (args, name) => value }
export function registerFunction(name, spec) {
  if (!new RegExp(`^${NAME}$`).test(name)) throw new Error(`bad function name ${name}`);
  if (Object.hasOwn(FUNCS, name)) console.warn(`[lang] function ${name} registered twice; the later one wins`);
  FUNCS[name] = spec;
}
// opts: { describe: 'a transformation', format(v) -> string, latex(v) -> string (row readout),
//         drawable: false for readout-only values, numbers(v) -> number[] checked for NaN / infinity }
export function registerType(type, opts) {
  TYPES.set(type, opts);
  if (opts.describe) KIND[type] = opts.describe;
}
export function registerConstant(name, fn) { CONSTS[name] = fn; }
export function isDrawable(v) {
  const t = TYPES.get(v?.type);
  if (!t) return v?.type !== 'mat';
  return typeof t.drawable === 'function' ? !!t.drawable(v) : t.drawable !== false;
}
export const valueLatex = (v) => TYPES.get(v?.type)?.latex?.(v) ?? null;
// A readout shown as is, without the "= ": {latex} or {text} (both may be empty).
export const valueReadout = (v) => TYPES.get(v?.type)?.readout?.(v) ?? null;

// Graphs: {type: 'graph', mode: 'curve' | 'surface' | 'param' | null, ins: ['x'] | ['x', 'y'] | ...,
// dep: the coordinate that is the value ('y' in y = x^2), at(env) -> value, free, callable, params,
// call(args), fname and d for a plotted built-in}. graph/features/plots.js draws them.
const FN_TEX = { sigmoid: '\\sigma', σ: '\\sigma', tanh: '\\tanh', sinh: '\\sinh', cosh: '\\cosh', erf: '\\operatorname{erf}' };
const FORMULAS = {
  sigmoid: '\\frac{1}{1 + e^{-x}}', σ: '\\frac{1}{1 + e^{-x}}',
  tanh: '\\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}', sinh: '\\frac{e^{x} - e^{-x}}{2}', cosh: '\\frac{e^{x} + e^{-x}}{2}',
  relu: '\\max(0, x)', leakyrelu: '\\max(0.01\\,x,\\ x)', leaky_relu: '\\max(0.01\\,x,\\ x)',
  elu: '\\max(0, x) + \\min(0,\\ e^{x} - 1)', gelu: 'x\\,\\Phi(x)', softplus: '\\ln(1 + e^{x})',
  silu: 'x\\,\\sigma(x)', swish: 'x\\,\\sigma(x)', mish: 'x \\tanh(\\ln(1 + e^{x}))',
  heaviside: '\\mathbb{1}[x > 0]', erf: '\\tfrac{2}{\\sqrt{\\pi}} \\textstyle\\int_0^x e^{-t^2} dt',
};
const DERIVATIVES = {
  sigmoid: '\\sigma(x)\\,(1 - \\sigma(x))', σ: '\\sigma(x)\\,(1 - \\sigma(x))', tanh: '1 - \\tanh^2 x',
  relu: '\\mathbb{1}[x > 0]', softplus: '\\sigma(x)', sin: '\\cos x', cos: '-\\sin x', exp: 'e^{x}',
};
function graphReadout(v) {
  if (!v.fname) return {};
  const body = v.d === 1 ? DERIVATIVES[v.fname] : v.d ? null : FORMULAS[v.fname];
  if (!body) return {};
  const head = FN_TEX[v.fname] ?? `\\operatorname{${v.fname.replace(/_/g, '\\_')}}`;
  return { latex: `${head}${primes(v.d)}(x) = ${body}` };
}
registerType('graph', { describe: 'a function', format: () => '', numbers: () => [], drawable: (v) => !!v.mode, readout: graphReadout });
registerType('softmaxmap', {
  describe: 'the softmax map',
  format: (v) => `temperature ${formatNumber(v.T)}`,
  numbers: (v) => [v.T],
  readout: (v) => ({
    latex: v.T === 1 ? '\\operatorname{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}'
      : `\\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}},\\ T = ${formatNumber(v.T)}`,
  }),
});
// AST of one line: {name, params, body, where, at, literal, error}; where (a restriction) is null or
// [{items, ops}], a chain items[0] ops[0] items[1] ... Nodes: num{v} name{name, d?} vec{items}
// mat{rows} neg{a} abs{a} comp{a,i} bin{op,a,b,implicit?} call{name,args,user?,d?} (d: derivative order)
// userFns: names defined as f(x) = ... elsewhere, so f(2) parses as a call.
export const parseLine = (line, userFns) => parseStatement(line, userFns);
export const functionNames = () => Object.keys(FUNCS);
export const values = { vec, point, mat, kindOf, describe, isVecLike, dot3, cross3, len3, identity, matMul, det, inverse };

function arity(lo, hi) {
  const s = (n) => `${n} argument${n === 1 ? '' : 's'}`;
  if (lo === hi) return s(lo);
  return hi === Infinity ? `at least ${s(lo)}` : `${lo} to ${hi} arguments`;
}

function call(name, args) {
  const fn = FUNCS[name];
  const [lo, hi] = Array.isArray(fn.n) ? fn.n : [fn.n, fn.n];
  if (args.length < lo || args.length > hi) throw new Error(`${name} needs ${arity(lo, hi)}, got ${args.length}`);
  if (fn.kind) {
    for (const a of args) {
      if (kindOf(a) !== fn.kind) {
        throw new Error(`${name} needs ${hi === 1 ? KIND[fn.kind] : PLURAL[fn.kind]}, got ${describe(a)}`);
      }
    }
  }
  return fn.f(args, name);
}

// ---------- tokenizer & parser ----------

function tokenize(src) {
  const toks = [];
  let i = 0, sp = false; // sp: whitespace before this token (spaces separate entries inside [ ])
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; sp = true; continue; }
    NUM_RE.lastIndex = i;
    let m = NUM_RE.exec(src);
    if (m) {
      toks.push({ t: 'num', v: parseFloat(m[0]), s: m[0], sp });
      sp = false;
      i += m[0].length;
      continue;
    }
    NAME_RE.lastIndex = i;
    m = NAME_RE.exec(src);
    if (m) {
      toks.push({ t: 'name', v: m[0], s: m[0], sp });
      sp = false;
      i += m[0].length;
      continue;
    }
    const op = OP_ALIASES[c] ?? c;
    const wide = (op === '<' || op === '>') && src[i + 1] === '='; // <= and >= (≤ and ≥ are aliases)
    if (wide || op === '<=' || op === '>=') {
      toks.push({ t: 'op', v: wide ? `${op}=` : op, s: wide ? `${c}=` : c, sp });
      sp = false;
      i += wide ? 2 : 1;
      continue;
    }
    if (OPS.includes(op)) {
      toks.push({ t: 'op', v: op, s: c, sp });
      sp = false;
      i++;
      continue;
    }
    throw new Error(`unexpected character '${String.fromCodePoint(src.codePointAt(i))}'`);
  }
  return toks;
}

const bin = (op, a, b) => ({ t: 'bin', op, a, b });

class Parser {
  constructor(toks, userFns) {
    this.toks = toks;
    this.userFns = userFns ?? new Set();
    this.pos = 0;
    this.absDepth = 0; // inside |...|, a '|' closes rather than starting an implicit product
    this.rowMode = 0;  // directly inside [ ... ]: a space starts the next entry (MATLAB style)
  }

  // [1 2 3], [1 -2; 3 4]: a spaced token ends the entry; unspaced juxtaposition still multiplies ([2a 3]).
  breaksEntry(t) { return this.rowMode > 0 && !!t?.sp; }

  peek() { return this.toks[this.pos]; }
  isOp(v) { const t = this.peek(); return t?.t === 'op' && t.v === v; }
  eat(v) { if (this.isOp(v)) { this.pos++; return true; } return false; }

  unexpected(wanted) {
    const t = this.peek();
    if (!t) return new Error(wanted ? `missing ${wanted}` : 'unexpected end of line');
    return new Error(`unexpected '${t.s}'${wanted ? `, expected ${wanted}` : ''}`);
  }

  expect(v) { if (!this.eat(v)) throw this.unexpected(`'${v}'`); }

  startsPrimary(t) {
    if (!t) return false;
    if (t.t !== 'op') return true;
    return t.v === '(' || t.v === '[' || (t.v === '|' && this.absDepth === 0);
  }

  expr() {
    let a = this.term();
    for (;;) {
      const t = this.peek(), n = this.toks[this.pos + 1];
      // "[1 -2]": a spaced sign glued to what follows is the sign of the next entry
      if (t?.t === 'op' && (t.v === '+' || t.v === '-') && this.breaksEntry(t) && n && !n.sp) return a;
      if (this.eat('+')) a = bin('+', a, this.term());
      else if (this.eat('-')) a = bin('-', a, this.term());
      else return a;
    }
  }

  term() {
    let a = this.unary();
    for (;;) {
      const t = this.peek();
      if (t?.t === 'op' && (t.v === '*' || t.v === '/')) {
        this.pos++;
        a = bin(t.v, a, this.unary());
      } else if (t?.t === 'op' && (t.v === '·' || t.v === '×')) {
        // juxtaposition on the right binds tighter here, so u × 2v means u × (2v)
        this.pos++;
        a = bin(t.v, a, this.juxtaposed(this.unary()));
      } else if (this.startsPrimary(t) && !this.breaksEntry(t)) {
        a = { ...bin('*', a, this.power()), implicit: true };
      } else {
        return a;
      }
    }
  }

  juxtaposed(a) {
    while (this.startsPrimary(this.peek()) && !this.breaksEntry(this.peek())) a = { ...bin('*', a, this.power()), implicit: true };
    return a;
  }

  unary() {
    if (this.eat('-')) return { t: 'neg', a: this.unary() };
    if (this.eat('+')) return this.unary();
    return this.power();
  }

  power() {
    const base = this.postfix();
    if (!this.isOp('^')) return base;
    const n = this.toks[this.pos + 1];
    if (n?.t === 'name' && n.v === 'T') { // u^T, A^T: transpose
      this.pos += 2;
      return { t: 'call', name: 'transpose', args: [base] };
    }
    this.pos++;
    return bin('^', base, this.unary());
  }

  postfix() {
    let a = this.primary();
    while (this.eat('.')) {
      const t = this.peek();
      const i = t?.t === 'name' ? ['x', 'y', 'z'].indexOf(t.v) : -1;
      if (i < 0) throw new Error("expected x, y or z after '.'");
      this.pos++;
      a = { t: 'comp', a, i };
    }
    return a;
  }

  primary() {
    const t = this.peek();
    if (!t) throw this.unexpected();
    if (t.t === 'num') {
      this.pos++;
      return { t: 'num', v: t.v };
    }
    if (t.t === 'name') {
      this.pos++;
      const user = !isFunc(t.v) && this.userFns.has(t.v);
      let d = 0; // sigmoid', f'': derivatives
      while (this.isOp("'")) { this.pos++; d++; }
      if (d && !user && !isFunc(t.v)) throw new Error(`' (a derivative) goes after a function, like sigmoid' or f'`);
      if ((user || isFunc(t.v)) && this.isOp('(')) {
        const node = { t: 'call', name: t.v, args: this.list('(', ')', true) };
        if (user) node.user = true;
        if (d) node.d = d;
        return node;
      }
      return d ? { t: 'name', name: t.v, d } : { t: 'name', name: t.v };
    }
    if (t.v === '(') {
      const items = this.list('(', ')');
      return items.length === 1 ? items[0] : vectorNode(items);
    }
    if (t.v === '[') {
      const next = this.toks[this.pos + 1];
      if (next?.t === 'op' && next.v === '[') return this.matrix();
      return this.bracket();
    }
    if (t.v === '|') {
      this.pos++;
      this.absDepth++;
      const a = this.expr();
      this.expect('|');
      this.absDepth--;
      return { t: 'abs', a };
    }
    throw this.unexpected();
  }

  // open expr (, expr)* close
  list(open, close, allowEmpty = false) {
    this.expect(open);
    const saved = this.absDepth, savedRow = this.rowMode;
    this.absDepth = 0;
    this.rowMode = 0;
    const items = [];
    if (!(allowEmpty && this.isOp(close))) {
      do items.push(this.expr());
      while (this.eat(','));
    }
    this.absDepth = saved;
    this.rowMode = savedRow;
    if (!this.eat(close)) throw this.unexpected(`'${close}'`);
    return items;
  }

  // [ entries ]: entries split by ',' or spaces, rows by ';'. Returns { rows, spaced }.
  bracketRows() {
    this.expect('[');
    const saved = this.absDepth, savedRow = this.rowMode;
    this.absDepth = 0;
    this.rowMode = 1;
    const rows = [[]];
    let spaced = false;
    for (;;) {
      rows.at(-1).push(this.expr());
      if (this.eat(',')) continue;
      if (this.eat(';')) { if (this.isOp(']')) break; rows.push([]); continue; }
      if (this.isOp(']')) break;
      if (this.breaksEntry(this.peek())) { spaced = true; continue; }
      throw this.unexpected("']'");
    }
    this.absDepth = saved;
    this.rowMode = savedRow;
    this.expect(']');
    return { rows, spaced };
  }

  // [1, 2, 3] and [1; 2; 3] are vectors; [1 2 3] is a 1×3 matrix; [1 2; 3 4] a 2×2.
  bracket() {
    const { rows, spaced } = this.bracketRows();
    if (rows.length === 1 && !spaced) return vectorNode(rows[0]);
    if (!spaced && rows.every((r) => r.length === 1) && rows.length <= 3) return vectorNode(rows.map((r) => r[0]));
    if (rows.some((r) => r.length !== rows[0].length)) throw new Error('matrix rows must all be the same length');
    return { t: 'mat', rows };
  }

  matrix() {
    this.expect('[');
    const rows = [];
    do {
      const r = this.bracketRows();
      if (r.rows.length > 1) throw new Error("inside [[...]] each [...] is one row; use ';' only in the [a b; c d] form");
      rows.push(r.rows[0]);
    } while (this.eat(','));
    this.expect(']');
    if (rows.some((r) => r.length !== rows[0].length)) throw new Error('matrix rows must all be the same length');
    return { t: 'mat', rows };
  }

  // {0 < x <= 1, y > 0}: comparisons (chains like a < x < b) that must all hold. -> [{items, ops}]
  restriction() {
    this.expect('{');
    const saved = this.absDepth, savedRow = this.rowMode, conds = [];
    this.absDepth = 0;
    this.rowMode = 0;
    do {
      const items = [this.expr()], ops = [];
      while (this.peek()?.t === 'op' && Object.hasOwn(RELS, this.peek().v)) {
        ops.push(this.peek().v);
        this.pos++;
        items.push(this.expr());
      }
      if (!ops.length) throw new Error(this.isOp('=') ? "use < or <= in a restriction, not '='" : 'a restriction needs a comparison, like {x > 0}');
      conds.push({ items, ops });
    } while (this.eat(','));
    this.absDepth = saved;
    this.rowMode = savedRow;
    if (!this.eat('}')) throw this.unexpected("'}'");
    return conds;
  }
}

function vectorNode(items) {
  if (items.length < 2 || items.length > 3) throw new Error(`a vector needs 2 or 3 components, got ${items.length}`);
  return { t: 'vec', items };
}

function stripComment(s) {
  const i = s.search(/#|\/\//);
  return i < 0 ? s : s.slice(0, i);
}

// Parses one line into {name, params, body, where, at, literal, error}, or null for blank/comment
// lines. params is set for a function definition, f(x) = x^2; where for a restriction, {x > 0}.
function parseStatement(line, userFns) {
  const text = stripComment(String(line ?? '')).trim();
  if (!text) return null;
  const fdef = FNDEF_RE.exec(text);
  const def = fdef ?? DEF_RE.exec(text);
  const st = { name: def ? def[1] : null, params: null, body: null, where: null, at: null, literal: false, error: null };
  try {
    if (st.name && isFunc(st.name)) throw new Error(`${st.name} is a built-in function; pick another name`);
    if (fdef) {
      st.params = fdef[2] ? fdef[2].split(',').map((s) => s.trim()) : [];
      const bad = st.params.find((p, i) => isFunc(p) || st.params.indexOf(p) !== i);
      if (bad) throw new Error(isFunc(bad) ? `${bad} is a built-in function; pick another parameter name` : `${bad} is listed twice`);
    }
    const toks = tokenize(def ? text.slice(def[0].length) : text);
    if (def && !toks.length) throw new Error("missing value after '='");
    const p = new Parser(toks, userFns);
    st.body = p.expr();
    while (p.isOp('{')) (st.where ??= []).push(...p.restriction());
    if (p.eat('@')) {
      if (!p.peek()) throw new Error("missing origin after '@'");
      st.at = p.expr();
    }
    const rest = p.peek();
    if (rest) {
      if (rest.t === 'op' && rest.v === '=' && !def) throw new Error("only a single name can go left of '='");
      if (rest.t === 'op' && Object.hasOwn(RELS, rest.v)) throw new Error('comparisons go in a restriction after the expression: y = x^2 {x > 0}');
      throw p.unexpected();
    }
    st.literal = !!def && !fdef && toks.at(-1).t === 'num' &&
      (toks.length === 1 || (toks.length === 2 && toks[0].t === 'op' && toks[0].v === '-'));
  } catch (e) {
    st.body = st.at = st.where = null;
    st.error = e.message;
  }
  return st;
}

// ---------- evaluation ----------

function refs(n, out = []) {
  if (n.t === 'name' || (n.t === 'call' && n.user)) out.push(n.name);
  for (const c of [n.a, n.b, ...(n.items ?? []), ...(n.args ?? []), ...(n.rows?.flat() ?? [])]) {
    if (c) refs(c, out);
  }
  return out;
}
// A row's names: its body and its restriction ({x > a} makes a row a graph of x that uses a).
const stmtRefs = (s) => {
  const out = s.body ? refs(s.body) : [];
  for (const c of s.where ?? []) for (const n of c.items) refs(n, out);
  return out;
};

// Shortest dependency path start → … → start, or null.
function findCycle(start, deps) {
  const prev = new Map();
  const queue = [start];
  while (queue.length) {
    const cur = queue.shift();
    for (const d of deps.get(cur) ?? []) {
      if (d === start) {
        const path = [start];
        for (let p = cur; p !== start; p = prev.get(p)) path.splice(1, 0, p);
        return [...path, start];
      }
      if (deps.has(d) && !prev.has(d)) {
        prev.set(d, cur);
        queue.push(d);
      }
    }
  }
  return null;
}

// A row compiles once into a closure env => value, so a graph can sample it thousands of times.
// env holds the free coordinates (x, y, z) and a function's parameters. cx: {lookup, free, params, coordFree}.

const toNum = (x, what) => {
  if (kindOf(x) !== 'num') throw new Error(`${what} must be numbers, got ${describe(x)}`);
  return x;
};
const mulNum = (a, b) => a * b;
const FAST = { // number op number, with the same errors as binary()
  '+': (a, b) => a + b,
  '-': (a, b) => a - b,
  '*': mulNum, '·': mulNum, '×': mulNum,
  '/': (a, b) => { if (b === 0) throw new Error('division by zero'); return a / b; },
  '^': (a, b) => ((a < 0 && !Number.isInteger(b)) || (a === 0 && b < 0) ? pow(a, b) : a ** b),
};
const NO_ENV = Object.freeze({});
const primes = (d) => "'".repeat(d);

// The d-th derivative of a one-number function by central differences (nested for d > 1, with a
// wider step, since rounding grows like 1/h^d).
function derivative(g, d, name) {
  let f = (x) => {
    const v = g([x]);
    if (typeof v !== 'number') throw new Error(`${name}${primes(d)} needs a function with a number value`);
    return v;
  };
  const h0 = d === 1 ? 1e-5 : 2e-3;
  for (let k = 0; k < d; k++) {
    const inner = f;
    f = (x) => { const h = h0 * Math.max(1, Math.abs(x)); return (inner(x + h) - inner(x - h)) / (2 * h); };
  }
  return (args) => {
    if (args.length !== 1 || typeof args[0] !== 'number') throw new Error(`${name}${primes(d)} needs one number`);
    return f(args[0]);
  };
}

// A restriction compiles to a check that throws OUTSIDE where a comparison fails: a gap in the
// graph. One shared error, since a surface may be sampled outside thousands of times per frame.
const OUTSIDE = new Error('outside the restriction { }');
function compileWhere(where, cx) {
  const conds = where.map(({ items, ops }) => ({ fs: items.map((n) => compile(n, cx)), rel: ops.map((o) => RELS[o]) }));
  return (e) => {
    for (const { fs, rel } of conds) {
      let a = toNum(fs[0](e), 'comparisons');
      for (let i = 0; i < rel.length; i++) {
        const b = toNum(fs[i + 1](e), 'comparisons');
        if (!rel[i](a, b)) throw OUTSIDE;
        a = b;
      }
    }
  };
}
// A graph's samples cut down to a restriction; one of y = ... that may name y ({|y| <= 2}) checks
// the value too.
const guarded = (f, check, dep) => (dep
  ? (e) => { const v = f(e); check({ ...e, [dep]: v }); return v; }
  : (e) => { check(e); return f(e); });

function compile(n, cx) {
  const C = (x) => compile(x, cx);
  switch (n.t) {
    case 'num': { const v = n.v; return () => v; }
    case 'name': return compileName(n, cx);
    case 'vec': {
      const items = n.items.map(C);
      return (e) => vec(items.map((f) => toNum(f(e), 'vector components')));
    }
    case 'mat': {
      const rows = n.rows.map((r) => r.map(C));
      return (e) => colToVec(rows.map((r) => r.map((f) => toNum(f(e), 'matrix entries'))));
    }
    case 'neg': { const a = C(n.a); return (e) => { const x = a(e); return typeof x === 'number' ? -x : neg(x); }; }
    case 'abs': { const a = C(n.a); return (e) => magnitude(a(e), '|...|'); }
    case 'comp': {
      const a = C(n.a), i = n.i;
      return (e) => {
        const x = a(e);
        if (!isVecLike(x)) throw new Error(`.${'xyz'[i]} needs a vector or point, got ${describe(x)}`);
        return x.v[i];
      };
    }
    case 'bin': {
      const a = C(n.a), b = C(n.b), op = n.op, fast = FAST[op];
      return (e) => {
        const x = a(e), y = b(e);
        return typeof x === 'number' && typeof y === 'number' ? fast(x, y) : binary(op, x, y);
      };
    }
    case 'call': return compileCall(n, cx);
  }
  throw new Error(`unknown node '${n.t}'`);
}

function compileName(n, cx) {
  const name = n.name;
  if (n.d) throw new Error(`${name}${primes(n.d)} is a function; call it like ${name}${primes(n.d)}(x)`);
  if (cx.params?.includes(name)) return (e) => e[name];
  if (cx.coordFree(name)) {
    cx.free.add(name);
    return (e) => e[name];
  }
  const v = cx.lookup(name);
  if (v?.type === 'graph') { // a = x^2 used in another row: that row is a graph of x too
    if (v.callable) throw new Error(`${name} is a function; call it like ${name}(...)`);
    for (const c of v.free) cx.free.add(c);
    return (e) => v.at(e);
  }
  return () => v;
}

function compileCall(n, cx) {
  const args = n.args.map((a) => compile(a, cx)), name = n.name;
  let g;
  if (n.user) {
    const f = cx.lookup(name);
    if (f?.type !== 'graph' || !f.callable) throw new Error(`${name} is not a function`);
    const k = f.params.length;
    if (args.length !== k) throw new Error(`${name} needs ${arity(k, k)}, got ${args.length}`);
    if (n.d && k !== 1) throw new Error(`${name}${primes(n.d)} needs a function of one variable`);
    g = f.call;
  } else {
    g = (vals) => call(name, vals);
  }
  if (n.d) g = derivative(g, n.d, name);
  if (args.length === 1) { const a = args[0]; return (e) => g([a(e)]); }
  return (e) => g(args.map((a) => a(e)));
}

// What a graph's samples are: 'num' or 'vec'. 1/x fails at 0, so try a few points; {error} if all
// fail. A restriction can leave out all of them, so then search a grid; a graph that is outside
// its restriction everywhere there draws nothing, which is not an error.
function probe(at, ins) {
  let err = null, outside = false;
  const tryAt = (vals) => {
    try {
      const e = {};
      ins.forEach((c, i) => { e[c] = vals[i]; });
      const k = kindOf(at(e));
      return k === 'point' ? 'vec' : k;
    } catch (e) {
      if (e === OUTSIDE) outside = true;
      else err = e;
      return null;
    }
  };
  for (const t of [0.37, -0.61, 1.3, 2.9, -4.1, 5.3]) {
    const k = tryAt(ins.map((_, i) => t * (i ? -0.7 : 1)));
    if (k) return k;
  }
  if (outside) {
    const G = [0.05, 0.2, 0.5, 0.8, 0.95, -0.5, 1.5, -1.5, 3, -3, 7, -7];
    for (const a of G) {
      for (const b of ins.length > 1 ? G : [0]) {
        const k = tryAt([a, b]);
        if (k) return k;
      }
    }
    if (!err) return 'num';
  }
  return { error: err };
}

// A row with free coordinates: a curve (y = f(x), x = g(y), ...), a surface (z = f(x, y)) or,
// for a vector of one coordinate, a parametric curve. dep is the row's name for y = ..., else null.
function graphOf(at, free, dep) {
  if (dep && free.has(dep)) throw new Error(`${dep} is on both sides of the =`);
  const ins = COORDS.filter((c) => free.has(c)), named = !!dep;
  if (!dep) {
    if (free.has('z')) throw new Error('z is the height here: write z = f(x, y), or y = f(x, z)');
    dep = ins.length === 1 && ins[0] === 'x' ? 'y' : 'z';
  }
  const k = probe(at, ins);
  if (k?.error) throw k.error;
  if (k === 'vec') {
    if (named) throw new Error(`${dep} = ... needs a number, got a vector`);
    if (ins.length !== 1) throw new Error('a surface needs a number; a vector of one coordinate draws a curve');
    return { type: 'graph', mode: 'param', ins, dep: null, free: ins, at };
  }
  if (k !== 'num') throw new Error(`can't draw a graph of ${KIND[k] ?? 'this'}`);
  return { type: 'graph', mode: ins.length === 1 ? 'curve' : 'surface', ins, dep, free: ins, at };
}

export function evaluate(lines) {
  const src = Array.from(lines ?? []);
  const userFns = new Set(); // f in f(x) = ..., so f(2) parses as a call on every row
  for (const l of src) {
    const m = FNDEF_RE.exec(stripComment(String(l ?? '')).trim());
    if (m && !isFunc(m[1])) userFns.add(m[1]);
  }
  const stmts = src.map((l) => parseStatement(l, userFns));

  // A row named x, y or z is a value unless it uses a coordinate that isn't a value itself:
  // x = [1, 2, 3] and y = A x are values, y = x^2 is a graph of y (s.eq) and defines nothing.
  const valueCoords = new Set();
  for (let grew = true; grew;) {
    grew = false;
    for (const s of stmts) {
      if (!s?.name || s.params || !COORDS.includes(s.name) || valueCoords.has(s.name)) continue;
      const cs = stmtRefs(s).filter((n) => COORDS.includes(n));
      if (cs.every((c) => c !== s.name && valueCoords.has(c))) { valueCoords.add(s.name); grew = true; }
    }
  }
  for (const s of stmts) if (s?.name && !s.params && COORDS.includes(s.name) && !valueCoords.has(s.name)) s.eq = true;
  const coordFree = (name) => COORDS.includes(name) && !valueCoords.has(name) && !userFns.has(name);

  const defs = new Map(); // name -> defining row indices
  stmts.forEach((s, i) => {
    if (s?.name && !isFunc(s.name) && !s.eq) defs.set(s.name, [...(defs.get(s.name) ?? []), i]);
  });
  const deps = new Map(); // uniquely defined name -> names its value refers to
  for (const [name, idx] of defs) {
    const s = stmts[idx[0]];
    if (idx.length === 1 && s.body) deps.set(name, stmtRefs(s).filter((r) => !s.params?.includes(r)));
  }

  const staticError = (s) => {
    if (s.error) return s.error;
    if (s.name && defs.get(s.name)?.length > 1) return `${s.name} is defined more than once`;
    const cycle = s.name && deps.has(s.name) && findCycle(s.name, deps);
    return cycle ? `circular definition: ${cycle.join(' → ')}` : null;
  };

  const memo = new Map(); // row index -> {value} | {error}
  const valueOf = (i) => {
    if (!memo.has(i)) {
      memo.set(i, { error: 'circular definition' }); // guard; cycles are normally caught statically
      const s = stmts[i];
      const err = staticError(s);
      let r;
      if (err) r = { error: err };
      else {
        try { r = { value: rowValue(s) }; }
        catch (e) { r = { error: e.message }; }
      }
      memo.set(i, r);
    }
    return memo.get(i);
  };

  const lookup = (name) => {
    const idx = defs.get(name);
    if (idx) {
      if (idx.length > 1) throw new Error(`${name} is defined more than once`);
      const r = valueOf(idx[0]);
      if ('error' in r) throw new Error(`${name} has an error`);
      return r.value;
    }
    if (Object.hasOwn(CONSTS, name)) return CONSTS[name]();
    if (isFunc(name)) throw new Error(`${name} is a function; call it like ${name}(...)`);
    throw new Error(`${name} is not defined`);
  };

  const cxFor = (params) => ({ lookup, free: new Set(), params, coordFree });
  const evalConst = (node) => {
    const cx = cxFor(null), f = compile(node, cx);
    if (cx.free.size) throw new Error(`${[...cx.free][0]} is not defined`);
    return f(NO_ENV);
  };

  // f(x) = ...: callable from other rows; one or two parameters also draw it.
  const userFunction = (s) => {
    const params = s.params, cx = cxFor(params), body = compile(s.body, cx);
    const check = s.where && compileWhere(s.where, cx);
    const extra = [...cx.free].filter((c) => !params.includes(c));
    if (extra.length) throw new Error(`${s.name} uses ${extra.join(' and ')}; add ${extra.length > 1 ? 'them' : 'it'} to ${s.name}(${params.join(', ')})`);
    const fcall = (vals) => {
      const e = {};
      for (let i = 0; i < params.length; i++) e[params[i]] = vals[i];
      if (check) check(e);
      return body(e);
    };
    const g = { type: 'graph', callable: true, params, call: fcall, free: [], mode: null };
    if (params.length === 1) {
      const at = (e) => fcall([e.x]), k = probe(at, ['x']);
      Object.assign(g, { mode: k === 'vec' ? 'param' : 'curve', ins: ['x'], dep: k === 'vec' ? null : 'y', at });
    } else if (params.length === 2) {
      Object.assign(g, { mode: 'surface', ins: ['x', 'y'], dep: 'z', at: (e) => fcall([e.x, e.y]) });
    }
    return g;
  };

  // A row that is only a function's name draws it: sigmoid, relu', f'. softmax has its own picture.
  const bareGraph = (body) => {
    if (body.t !== 'name') return null;
    const { name, d = 0 } = body;
    let g;
    if (isFunc(name)) {
      const spec = FUNCS[name];
      if (!d && spec.bare) return spec.bare();
      const lo = Array.isArray(spec.n) ? spec.n[0] : spec.n;
      if (lo !== 1 || (spec.kind !== 'num' && name !== 'abs')) {
        if (d) throw new Error(`${name}${primes(d)} needs a function of one number`);
        return null;
      }
      g = (vals) => call(name, vals);
    } else if (userFns.has(name) && defs.has(name)) {
      const f = lookup(name);
      if (!d) return f.mode ? { type: 'graph', mode: f.mode, ins: f.ins, dep: f.dep, free: f.ins, at: f.at } : null;
      if (f.params.length !== 1) throw new Error(`${name}${primes(d)} needs a function of one variable`);
      g = f.call;
    } else {
      return null;
    }
    if (d) g = derivative(g, d, name);
    return { type: 'graph', mode: 'curve', ins: ['x'], dep: 'y', free: ['x'], at: (e) => g([e.x]), fname: name, d };
  };

  // sigmoid {x > 0}: a drawn function cut down to its restriction.
  const restrictGraph = (g, check, free, dep) => {
    if (g.type !== 'graph' || !g.mode) throw new Error('only a graph can have a restriction { }');
    const extra = [...free].find((c) => !g.ins.includes(c));
    if (extra) throw new Error(`this graph is drawn over ${g.ins.join(' and ')}, so its restriction can't use ${extra}`);
    return { ...g, at: guarded(g.at, check, dep && g.dep) };
  };

  const rowValue = (s) => {
    if (s.params) return userFunction(s);
    // y = ... {|y| <= 2}: in the restriction of an equation, its own coordinate is the value
    const dep = s.eq ? s.name : null, cx = cxFor(null), wx = dep ? cxFor([dep]) : cx;
    const check = s.where && compileWhere(s.where, wx);
    for (const c of wx.free) cx.free.add(c);
    const own = dep && s.where?.some((c) => c.items.some((n) => refs(n).includes(dep))) ? dep : null;
    const bare = bareGraph(s.body);
    if (bare) return check ? restrictGraph(bare, check, cx.free, own) : bare;
    const f = compile(s.body, cx), at = check ? guarded(f, check, own) : f;
    if (!cx.free.size) {
      if (dep && check) throw new Error(`${dep} = ... needs x, y or z on the right to draw`);
      return finite(at(NO_ENV));
    }
    return graphOf(at, cx.free, dep);
  };

  return stmts.map((s, i) => {
    const row = { name: null, value: null, origin: null, error: null, slider: null };
    if (!s) return row;
    row.name = s.name;
    const r = valueOf(i);
    if ('error' in r) {
      row.error = r.error;
      return row;
    }
    row.value = r.value;
    if (s.at) {
      try {
        const o = finite(evalConst(s.at));
        if (!isVecLike(o)) throw new Error(`@ needs a vector or point, got ${describe(o)}`);
        row.origin = o.v.slice();
      } catch (e) {
        row.error = e.message;
      }
    }
    if (s.literal && !row.error) row.slider = r.value;
    return row;
  });
}

// ---------- formatting ----------

export function formatNumber(x) {
  if (typeof x !== 'number') return String(x);
  if (Number.isNaN(x)) return 'undefined';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞';
  if (Math.abs(x) >= 1e15) return x.toExponential(4).replace(/\.?0+e/, 'e');
  const s = x.toFixed(4).replace(/\.?0+$/, '');
  return s === '-0' ? '0' : s;
}

const tuple = (a) => `(${a.map(formatNumber).join(', ')})`;
const SPAN_DESC = ['just the origin', 'line through origin', 'plane through origin', 'all of R^3'];

export function formatValue(v) {
  if (v == null) return '';
  if (typeof v === 'number') return formatNumber(v);
  switch (v.type) {
    case 'vec': case 'point': return tuple(v.v);
    case 'mat': return `[${v.m.map((r) => `[${r.map(formatNumber).join(', ')}]`).join(', ')}]`;
    case 'span': return SPAN_DESC[v.vecs.length] ?? 'span';
    case 'plane': return `plane with normal ${tuple(v.normal)}`;
    case 'parallelogram': return `area ${formatNumber(len3(cross3(v.u, v.v)))}`;
    case 'parallelepiped': return `volume ${formatNumber(Math.abs(dot3(v.u, cross3(v.v, v.w))))}`;
  }
  const custom = TYPES.get(v.type)?.format;
  return custom ? custom(v) : String(v);
}
