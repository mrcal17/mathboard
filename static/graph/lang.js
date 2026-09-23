// Expression language for the 3D vector grapher.
// evaluate(lines) turns editor lines into rows of {name, value, origin, error, slider}.
// Pure ES module, no dependencies; runs in browsers and Node.

const NAME = '[A-Za-z\\u0370-\\u03FF][A-Za-z0-9_\\u0370-\\u03FF]*';
const NAME_RE = new RegExp(NAME, 'y');
const DEF_RE = new RegExp(`^(${NAME})\\s*=`);
const NUM_RE = /(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/y;
const OPS = '+-*/·×^()[],;|=@.';
const OP_ALIASES = { '−': '-', '⋅': '·', '∙': '·' };

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

// n: exact argument count or [min, max]; kind: required type of every argument (checked before f runs).
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
  return t ? t.drawable !== false : v?.type !== 'mat';
}
export const valueLatex = (v) => TYPES.get(v?.type)?.latex?.(v) ?? null;
// AST of one line: {name, body, at, literal, error}. Nodes: num{v} name{name} vec{items} mat{rows}
// neg{a} abs{a} comp{a,i} bin{op,a,b,implicit?} call{name,args}
export const parseLine = (line) => parseStatement(line);
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
  constructor(toks) {
    this.toks = toks;
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
      if (isFunc(t.v) && this.isOp('(')) return { t: 'call', name: t.v, args: this.list('(', ')', true) };
      return { t: 'name', name: t.v };
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
}

function vectorNode(items) {
  if (items.length < 2 || items.length > 3) throw new Error(`a vector needs 2 or 3 components, got ${items.length}`);
  return { t: 'vec', items };
}

function stripComment(s) {
  const i = s.search(/#|\/\//);
  return i < 0 ? s : s.slice(0, i);
}

// Parses one line into {name, body, at, literal, error}, or null for blank/comment lines.
function parseStatement(line) {
  const text = stripComment(String(line ?? '')).trim();
  if (!text) return null;
  const def = DEF_RE.exec(text);
  const st = { name: def ? def[1] : null, body: null, at: null, literal: false, error: null };
  try {
    if (st.name && isFunc(st.name)) throw new Error(`${st.name} is a built-in function; pick another name`);
    const toks = tokenize(def ? text.slice(def[0].length) : text);
    if (def && !toks.length) throw new Error("missing value after '='");
    const p = new Parser(toks);
    st.body = p.expr();
    if (p.eat('@')) {
      if (!p.peek()) throw new Error("missing origin after '@'");
      st.at = p.expr();
    }
    const rest = p.peek();
    if (rest) {
      if (rest.t === 'op' && rest.v === '=' && !def) throw new Error("only a single name can go left of '='");
      throw p.unexpected();
    }
    st.literal = !!def && toks.at(-1).t === 'num' &&
      (toks.length === 1 || (toks.length === 2 && toks[0].t === 'op' && toks[0].v === '-'));
  } catch (e) {
    st.body = st.at = null;
    st.error = e.message;
  }
  return st;
}

// ---------- evaluation ----------

function refs(n, out = []) {
  if (n.t === 'name') out.push(n.name);
  for (const c of [n.a, n.b, ...(n.items ?? []), ...(n.args ?? []), ...(n.rows?.flat() ?? [])]) {
    if (c) refs(c, out);
  }
  return out;
}

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

function evalNode(n, lookup) {
  const ev = (x) => evalNode(x, lookup);
  const toNum = (x, what) => {
    if (kindOf(x) !== 'num') throw new Error(`${what} must be numbers, got ${describe(x)}`);
    return x;
  };
  switch (n.t) {
    case 'num': return n.v;
    case 'name': return lookup(n.name);
    case 'vec': return vec(n.items.map((it) => toNum(ev(it), 'vector components')));
    case 'mat': return colToVec(n.rows.map((r) => r.map((it) => toNum(ev(it), 'matrix entries'))));
    case 'neg': return neg(ev(n.a));
    case 'abs': return magnitude(ev(n.a), '|...|');
    case 'comp': {
      const a = ev(n.a);
      if (!isVecLike(a)) throw new Error(`.${'xyz'[n.i]} needs a vector or point, got ${describe(a)}`);
      return a.v[n.i];
    }
    case 'bin': return binary(n.op, ev(n.a), ev(n.b));
    case 'call': return call(n.name, n.args.map(ev));
  }
  throw new Error(`unknown node '${n.t}'`);
}

export function evaluate(lines) {
  const stmts = Array.from(lines ?? [], parseStatement);

  const defs = new Map(); // name -> defining row indices
  stmts.forEach((s, i) => {
    if (s?.name && !isFunc(s.name)) defs.set(s.name, [...(defs.get(s.name) ?? []), i]);
  });
  const deps = new Map(); // uniquely defined name -> names its value refers to
  for (const [name, idx] of defs) {
    const body = stmts[idx[0]].body;
    if (idx.length === 1 && body) deps.set(name, refs(body));
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
        try { r = { value: finite(evalNode(s.body, lookup)) }; }
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
        const o = finite(evalNode(s.at, lookup));
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
