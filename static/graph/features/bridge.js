// Bridge between the handwriting board (static/app.js) and the 3D tab.
//   Board -> 3D: "Send to 3D" in the board's expression editor fires `mathboard:to-graph` {latex}.
//     latexToGrapher() turns the LaTeX into a grapher row, which is added (or replaces the row that
//     defines the same name) and the 3D tab is shown. Input it can't convert only toasts.
//   3D -> board: the "To board" toolbar button snapshots the view (labels drawn in), downscales it
//     to a JPEG and fires `mathboard:to-board` {src, w, h}; app.js puts it on the current page.
// Listeners set detail.handled (plus detail.ok / detail.page) so the sender knows someone answered.
// latexToGrapher is pure (no DOM) and tested in tests/bridge.test.mjs.
import { functionNames, parseLine } from '../lang.js';

const SNAP_MAX = 960;       // px, longest side of the JPEG put on the board
const SNAP_QUALITY = 0.85;

// ================================================================ LaTeX -> grapher syntax

const GREEK = {
  __proto__: null,
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η',
  theta: 'θ', vartheta: 'θ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', rho: 'ρ',
  varrho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
};
const FN_CMDS = {
  __proto__: null,
  sin: 'sin', cos: 'cos', tan: 'tan', arcsin: 'asin', arccos: 'acos', arctan: 'atan',
  ln: 'ln', log: 'log', exp: 'exp', det: 'det', min: 'min', max: 'max',
};
const INVERSE = { __proto__: null, sin: 'asin', cos: 'acos', tan: 'atan' };
const OP_CMDS = { __proto__: null, cdot: '·', centerdot: '·', bullet: '·', times: '×', ast: '*', div: '/' };
const OP_CHARS = {
  __proto__: null,
  '+': '+', '-': '-', '−': '-', '*': '*', '∗': '*', '/': '/', '÷': '/', '=': '=', ',': ',',
  '·': '·', '⋅': '·', '∙': '·', '•': '·', '×': '×',
};
const NAME_CMDS = new Set(['vec', 'vv', 'mathbf', 'boldsymbol', 'bm', 'overrightarrow', 'underline']);
const DECOR_CMDS = { __proto__: null, hat: 'hat', widehat: 'hat', bar: 'bar', tilde: 'tilde', widetilde: 'tilde' }; // \hat{u} -> u_hat
const TEXT_CMDS = new Set(['text', 'textrm', 'textit', 'textbf', 'textsf', 'mathrm', 'mathit', 'mathsf', 'mathtt', 'operatorname']);
const SKIP_CMDS = new Set(['', ',', ';', ':', '!', ' ', 'quad', 'qquad', 'displaystyle', 'textstyle', 'limits',
  'nolimits', 'hline', 'big', 'Big', 'bigg', 'Bigg', 'bigl', 'bigr', 'Bigl', 'Bigr']);
const RELATIONS = new Set(['le', 'leq', 'leqslant', 'ge', 'geq', 'geqslant', 'ne', 'neq', 'approx', 'equiv', 'sim',
  'simeq', 'cong', 'lt', 'gt', 'll', 'gg', 'propto']);
const ARROWS = new Set(['to', 'rightarrow', 'Rightarrow', 'implies', 'iff', 'mapsto', 'longrightarrow',
  'leftarrow', 'Leftarrow', 'Leftrightarrow', 'longmapsto']);
const MULTILINE = /^(?:aligned|align\*?|alignat\*?|gathered|gather\*?|split|cases|eqnarray\*?|multline\*?)$/;
const RELATION_MSG = "the 3D tab can't show relations like < or ≈";
const PRIME_MSG = "primes (u') can't be sent; use a name like u2";

// Output pieces: num | name | fn (always followed by '(') | op | open | close.
const P = {
  num: s => ({ k: 'num', s }), name: s => ({ k: 'name', s }), fn: s => ({ k: 'fn', s }),
  op: s => ({ k: 'op', s }), open: (s = '(') => ({ k: 'open', s }), close: (s = ')') => ({ k: 'close', s }),
};
const paren = ps => [P.open(), ...ps, P.close()];
const call = (name, ps) => [P.fn(name), ...paren(stripOuter(ps))];
const join = lists => lists.flatMap((ps, i) => (i ? [P.op(','), ...ps] : ps));
const nameUnit = s => ({ p: [P.name(s)], name: s });
const isCh = (tk, v) => tk?.t === 'ch' && tk.v === v;
const isCmd = (tk, v) => tk?.t === 'cmd' && tk.v === v;

function barKind(tk) {
  if (isCh(tk, '|')) return '|';
  if (tk?.t !== 'cmd') return null;
  if (['|', 'Vert', 'lVert', 'rVert'].includes(tk.v)) return '‖';
  if (['vert', 'lvert', 'rvert', 'mid'].includes(tk.v)) return '|';
  return null;
}

function closing(ps, i) { // index of the close matching the open at i
  for (let d = 0, j = i; j < ps.length; j++) {
    if (ps[j].k === 'open') d++;
    else if (ps[j].k === 'close' && --d === 0) return j;
  }
  return -1;
}
function topComma(ps) {
  let d = 0;
  for (const x of ps) {
    if (x.k === 'open') d++;
    else if (x.k === 'close') d--;
    else if (!d && x.s === ',') return true;
  }
  return false;
}
// One operand on its own: a number, a name, a call, a bracketed group or |x|.
function atomic(ps) {
  if (ps.length < 2) return ps.length === 1 && ps[0].k !== 'op';
  const i = ps[0].k === 'fn' ? 1 : 0;
  if (ps[i]?.k === 'open') return closing(ps, i) === ps.length - 1;
  return ps[0].s === '|' && ps.at(-1).s === '|' && ps.filter(x => x.s === '|').length === 2;
}
const tight = ps => (atomic(ps) ? ps : paren(ps));
function stripOuter(ps) { // (a + b) -> a + b, but keep (1, 2, 3)
  while (ps.length > 2 && ps[0].s === '(' && closing(ps, 0) === ps.length - 1 && !topComma(ps.slice(1, -1))) ps = ps.slice(1, -1);
  return ps;
}

function lex(src) {
  const toks = [];
  let sp = false;
  for (const m of src.matchAll(/(\s+)|\\([A-Za-z]+|[^A-Za-z]?)|(\d+(?:\.\d+)?|\.\d+)|([A-Za-z\u0370-\u03FF])|([^])/gu)) {
    if (m[1]) { sp = true; continue; }
    const tk = m[2] != null ? { t: 'cmd', v: m[2] } : m[3] ? { t: 'num', v: m[3] } : m[4] ? { t: 'letter', v: m[4] } : { t: 'ch', v: m[5] };
    tk.sp = sp; // whitespace before it (splits letter runs: "det A" vs "detA")
    sp = false;
    toks.push(tk);
  }
  return toks;
}

// n x 1 with 2-3 entries -> vector; 1 x n stays a row matrix; vmatrix -> det, Vmatrix -> |v|.
function matrixUnit(rows, kind) {
  const R = rows.length, C = rows[0].length, cells = rows.flat();
  const list = join(cells.map(c => c.p));
  const colShaped = C === 1 && R >= 2 && R <= 3, rowShaped = R === 1 && C >= 2 && C <= 3;
  const vecShaped = colShaped || rowShaped;
  const literal = () => [P.open('['), ...join(rows.map(r => [P.open('['), ...join(r.map(c => c.p)), P.close(']')])), P.close(']')];
  let p;
  if (kind === 'det' && R === 3 && C === 3 && rows[0].every((c, i) => c.p.length === 1 && c.p[0].s === 'ijk'[i])) {
    p = call('cross', [...paren(join(rows[1].map(c => c.p))), P.op(','), ...paren(join(rows[2].map(c => c.p)))]);
  } else if (kind === 'det' && R === C) p = call('det', literal());
  else if (kind !== 'matrix' && vecShaped) p = [P.op('|'), ...paren(list), P.op('|')];
  else if (kind === 'det') throw new Error(`a determinant needs a square matrix, got ${R}×${C}`);
  else if (kind === 'norm') throw new Error(`‖…‖ needs a vector, got a ${R}×${C} matrix`);
  else if (rowShaped && cells.every(c => c.vec)) p = call('matrix', list); // [u v w]: columns
  else if (colShaped) p = paren(list);
  else p = literal();
  return { p, mat: rows, kind, vecShaped: colShaped && kind === 'matrix' };
}

function wrapGroup(open, inner) {
  if (inner.length === 1 && inner[0].mat) {
    return matrixUnit(inner[0].mat, open === '|' ? 'det' : open === '‖' ? 'norm' : inner[0].kind);
  }
  const p = finish(inner);
  if (!p.length) throw new Error(open === '(' ? 'empty brackets' : 'empty |…|');
  if (open !== '(') return { p: [P.op('|'), ...p, P.op('|')] };
  return { p: paren(p), group: p };
}

// Units -> pieces: bind function arguments (innermost first) and flatten.
function finish(units) {
  units = units.slice();
  for (let i = units.length - 1; i >= 0; i--) {
    const u = units[i];
    if (!u.fn) continue;
    let j = i + 1, arg;
    if (units[j]?.group) arg = units[j++].group;            // \sin(x)
    else {                                                  // \sin 2x, \det A: the juxtaposed run
      while (units[j] && !units[j].op) j++;
      arg = units.slice(i + 1, j).flatMap(x => x.p);
    }
    if (!arg.length) throw new Error(`${u.fn} needs an argument`);
    if (u.sub) arg = [...arg, P.op(','), ...u.sub];         // proj_v u -> proj(u, v)
    let p = call(u.fn, arg);
    if (u.sup) p = [...p, P.op('^'), ...tight(u.sup)];      // \sin^2 x -> sin(x)^2
    units.splice(i, j - i, { p });
  }
  return units.flatMap(u => (u.op ? [P.op(u.op)] : u.p));
}

class Conv {
  constructor(toks, fns) { this.t = toks; this.i = 0; this.fns = fns; }
  peek() { return this.t[this.i]; }
  next() { return this.t[this.i++]; }
  sub(toks) { return new Conv(toks, this.fns); }
  all() { return finish(this.seq()); }
  expect(test, what) {
    if (!test(this.peek())) throw new Error(`missing ${what}`);
    this.i++;
  }

  // Units until stop(token) (left unconsumed) or the end. A unit is {op} or an operand
  // {p: pieces, name?, vec?, group?, mat?} or an unbound function {fn, sub?, sup?}.
  seq(stop = () => false) {
    const units = [];
    while (this.i < this.t.length && !stop(this.peek())) {
      const u = this.unit();
      if (Array.isArray(u)) units.push(...u);
      else if (u) units.push(u);
    }
    return units;
  }

  unit() {
    const tk = this.next();
    if (tk.t === 'num') return this.postfix({ p: [P.num(tk.v)] });
    if (tk.t === 'letter') return this.postfix(this.letters(tk));
    if (tk.t === 'cmd') return this.command(tk.v);
    const c = tk.v;
    if (OP_CHARS[c]) return { op: OP_CHARS[c] };
    if (c === '(' || c === '[') return this.postfix(this.closeWith(t => isCh(t, c === '(' ? ')' : ']'), c === '(' ? "')'" : "']'"));
    if (c === '{') return this.brace();
    if (c === '|') return this.postfix(this.bars('|'));
    if (c === '<' || c === '>') throw new Error(RELATION_MSG);
    if (c === "'" || c === '′') throw new Error(PRIME_MSG);
    if (c === '^' || c === '_') throw new Error(`${c === '^' ? 'a power' : 'a subscript'} needs something in front of it`);
    if (c === ')' || c === ']' || c === '}') throw new Error('unbalanced brackets');
    if (c === '&') throw new Error("can't send a table; put it in a matrix");
    throw new Error(`can't convert '${c}'`);
  }

  command(v) {
    if (SKIP_CMDS.has(v)) return null;
    if (v === '\\' || v === 'cr' || v === 'newline') throw new Error("can't send several lines at once; send one expression");
    if (GREEK[v]) return this.postfix(nameUnit(GREEK[v]));
    if (v === 'pi') return this.postfix(nameUnit('pi'));
    if (v === 'imath' || v === 'jmath') return this.postfix(nameUnit(v[0]));
    if (v === 'top' || v === 'intercal') return nameUnit('T');
    if (OP_CMDS[v]) return { op: OP_CMDS[v] };
    if (FN_CMDS[v]) return this.postfix({ fn: FN_CMDS[v] });
    if (NAME_CMDS.has(v) || DECOR_CMDS[v]) return this.postfix(this.marked(v));
    if (TEXT_CMDS.has(v)) return this.postfix(this.text());
    if (['frac', 'dfrac', 'tfrac', 'cfrac'].includes(v)) return this.postfix(this.frac());
    if (v === 'sqrt') return this.postfix(this.sqrt());
    if (v === 'left') return this.postfix(this.left());
    if (v === 'begin') return this.postfix(this.env());
    if (v === 'langle') return this.postfix(this.closeWith(t => isCmd(t, 'rangle'), "'⟩'"));
    if (v === '{' || v === 'lbrace') return this.postfix(this.closeWith(t => isCmd(t, '}') || isCmd(t, 'rbrace'), "'\\}'"));
    if (v === 'lbrack') return this.postfix(this.closeWith(t => isCmd(t, 'rbrack') || isCh(t, ']'), "']'"));
    const bar = barKind({ t: 'cmd', v });
    if (bar) return this.postfix(this.bars(bar));
    if (RELATIONS.has(v)) throw new Error(RELATION_MSG);
    if (ARROWS.has(v)) throw new Error(`can't convert arrows (\\${v})`);
    if (v === 'prime') throw new Error(PRIME_MSG);
    if (['right', 'end', 'rangle', 'rbrace', '}', 'rbrack'].includes(v)) throw new Error('unbalanced brackets');
    throw new Error(`can't convert \\${v}`);
  }

  postfix(u) {
    while (u) {
      const tk = this.peek();
      if (isCh(tk, '_')) { this.i++; u = this.subscript(u); }
      else if (isCh(tk, '^')) { this.i++; u = this.superscript(u); }
      else if (isCh(tk, "'") || isCh(tk, '′') || isCmd(tk, 'prime')) throw new Error(PRIME_MSG);
      else break;
    }
    return u;
  }

  // The next TeX argument as tokens: {…} or a single token (one digit of a number: \frac12, x^23).
  arg() {
    const tk = this.next();
    if (!tk) throw new Error('missing argument');
    if (isCh(tk, '{')) {
      const start = this.i;
      for (let d = 1; this.i < this.t.length; this.i++) {
        const x = this.t[this.i];
        if (isCh(x, '{')) d++;
        else if (isCh(x, '}') && --d === 0) return this.t.slice(start, this.i++);
      }
      throw new Error('unbalanced braces');
    }
    if (isCh(tk, '}')) throw new Error('missing argument');
    if (tk.t === 'num' && tk.v.length > 1 && /\d/.test(tk.v[0])) {
      this.t[--this.i] = { ...tk, v: tk.v.slice(1), sp: false };
      return [{ t: 'num', v: tk.v[0] }];
    }
    return [tk];
  }

  letters(tk) { // a run of plain letters spelling a known function (span, proj, det…) is that function
    let j = this.i, run = tk.v;
    while (this.t[j]?.t === 'letter' && !this.t[j].sp) run += this.t[j++].v;
    if (run.length > 1 && this.fns.has(run)) { this.i = j; return { fn: run }; }
    return nameUnit(tk.v);
  }

  marked(cmd) { // \vec{u}, \mathbf{u} -> u; \hat{u} -> u_hat, but \hat{i} -> i (basis vector)
    const inner = this.sub(this.arg()).seq();
    if (inner.length === 1 && inner[0].p?.length === 1 && inner[0].p[0].s === '0') {
      return { p: paren(join([[P.num('0')], [P.num('0')], [P.num('0')]])) };
    }
    if (!inner.length || !inner.every(u => u.name)) throw new Error(`only a name can go inside \\${cmd}{…}`);
    let name = inner.map(u => u.name).join('');
    if (DECOR_CMDS[cmd] && !(cmd === 'hat' && /^[ijk]$/.test(name))) name += `_${DECOR_CMDS[cmd]}`;
    return { ...nameUnit(name), vec: true };
  }

  text() { // \text{span}, \operatorname{proj}, \mathrm{A}: a function or a name
    const toks = this.arg();
    const shown = toks.map((tk, i) => (i && tk.sp ? ' ' : '') + (tk.t === 'cmd' ? `\\${tk.v}` : tk.v)).join('');
    if (!toks.length || !shown.trim()) return null;
    if (toks.some((tk, i) => (i && tk.sp) || (tk.t !== 'letter' && tk.t !== 'num')) || toks[0].t !== 'letter') {
      throw new Error(`can't convert the text "${shown.trim()}"`);
    }
    const word = toks.map(tk => tk.v).join('');
    const fn = [word, word.toLowerCase()].find(w => this.fns.has(w));
    return fn ? { fn } : nameUnit(word);
  }

  frac() {
    const a = this.sub(this.arg()).all(), b = this.sub(this.arg()).all();
    if (!a.length || !b.length) throw new Error('a fraction needs a top and a bottom');
    return { p: paren([...tight(a), P.op('/'), ...tight(b)]) };
  }

  sqrt() {
    let n = null;
    if (isCh(this.peek(), '[')) {
      this.i++;
      n = finish(this.seq(t => isCh(t, ']')));
      this.expect(t => isCh(t, ']'), "']'");
    }
    const a = this.sub(this.arg()).all();
    if (!a.length) throw new Error('empty square root');
    if (!n?.length) return { p: call('sqrt', a) };
    return { p: [...tight(a), P.op('^'), ...paren([P.num('1'), P.op('/'), ...tight(n)])] };
  }

  subscript(u) {
    const toks = this.arg();
    if (u.fn === 'proj') return { ...u, sub: this.sub(toks).all() };
    if (!u.name) throw new Error(u.fn ? `can't convert ${u.fn} with a subscript` : 'only names can have subscripts (like v_1)');
    const parts = this.sub(toks).seq().filter(x => x.op !== ',')
      .map(x => x.name ?? (x.p?.length === 1 && x.p[0].k === 'num' ? x.p[0].s : null));
    if (!parts.length || parts.some(s => s == null)) throw new Error(`can't convert the subscript of ${u.name}`);
    const sub = parts.join('').replace(/\./g, '');
    const name = /^\d+$/.test(sub) ? u.name + sub : `${u.name}_${sub}`; // v_1 -> v1, v_x -> v_x
    return { ...u, p: [P.name(name)], name };
  }

  superscript(u) {
    const toks = this.arg();
    if (toks.length === 1 && isCmd(toks[0], 'circ')) { // 30^\circ
      if (u.fn) throw new Error(`can't convert ${u.fn}^\\circ`);
      return { p: call('rad', u.p) };
    }
    const sup = this.sub(toks).all();
    if (!sup.length) throw new Error('empty power');
    if (u.fn) {
      if (sup.length === 2 && sup[0].s === '-' && sup[1].s === '1' && INVERSE[u.fn]) return { fn: INVERSE[u.fn] };
      return { ...u, sup };
    }
    if (sup.length === 1 && sup[0].k === 'name' && sup[0].s === 'T') { // A^T, u^T (a vector's transpose is a row)
      return { p: call('transpose', u.p) };
    }
    const exp = atomic(sup) || (sup[0].s === '-' && atomic(sup.slice(1))) ? sup : paren(sup);
    return { p: [...tight(u.p), P.op('^'), ...exp] };
  }

  closeWith(stop, what) { // ( … ), [ … ], \langle … \rangle, \{ … \}: parentheses (or the matrix inside)
    const inner = this.seq(stop);
    this.expect(stop, what);
    return wrapGroup('(', inner);
  }

  brace() { // {…} is invisible TeX grouping, so its contents join the surrounding sequence
    const inner = this.seq(t => isCh(t, '}'));
    this.expect(t => isCh(t, '}'), "'}'");
    if (!isCh(this.peek(), '^') && !isCh(this.peek(), '_')) return inner;
    return this.postfix(inner.length === 1 && !inner[0].op ? inner[0] : { p: paren(finish(inner)) });
  }

  bars(kind) { // |x|, ‖x‖ (and |matrix| -> det)
    const inner = this.seq(t => barKind(t) === kind);
    this.expect(t => barKind(t) === kind, `closing ${kind}`);
    return wrapGroup(kind, inner);
  }

  left() {
    const open = this.delim();
    const inner = this.seq(t => isCmd(t, 'right'));
    this.expect(t => isCmd(t, 'right'), '\\right');
    this.delim();
    return wrapGroup(open, inner);
  }
  delim() {
    const tk = this.next();
    const bar = barKind(tk);
    if (bar) return bar;
    if ((tk?.t === 'ch' && '()[].'.includes(tk.v)) ||
        (tk?.t === 'cmd' && ['{', '}', 'langle', 'rangle', 'lbrace', 'rbrace', 'lbrack', 'rbrack'].includes(tk.v))) return '(';
    throw new Error(tk ? `can't convert the bracket ${tk.t === 'cmd' ? '\\' : ''}${tk.v}` : 'missing bracket after \\left or \\right');
  }

  env() {
    const name = this.rawArg();
    if (!/^(?:[pbBvV]?matrix\*?|smallmatrix|array)$/.test(name)) {
      throw new Error(MULTILINE.test(name) ? "can't send several lines at once; send one expression" : `can't convert the ${name} environment`);
    }
    if (name === 'array') this.rawArg();                                   // column spec
    if (isCh(this.peek(), '[')) while (this.i < this.t.length && !isCh(this.next(), ']'));  // bmatrix*[r]
    const rows = [[]];
    const stop = t => isCh(t, '&') || isCmd(t, '\\') || isCmd(t, 'cr') || isCmd(t, 'end');
    for (;;) {
      const units = this.seq(stop), tk = this.next();
      if (!tk) throw new Error(`missing \\end{${name}}`);
      rows.at(-1).push({ p: stripOuter(finish(units)), vec: units.length === 1 && !!units[0].vec });
      if (isCmd(tk, 'end')) { this.rawArg(); break; }
      if (!isCh(tk, '&')) rows.push([]);
    }
    while (rows.length && rows.at(-1).every(c => !c.p.length)) rows.pop(); // trailing \\
    if (!rows.length) throw new Error('empty matrix');
    if (rows.some(r => r.length !== rows[0].length)) throw new Error('matrix rows have different lengths');
    if (rows.some(r => r.some(c => !c.p.length))) throw new Error('a matrix entry is empty');
    return matrixUnit(rows, name[0] === 'v' ? 'det' : name[0] === 'V' ? 'norm' : 'matrix');
  }
  rawArg() { // {text} as a plain string
    this.expect(t => isCh(t, '{'), "'{'");
    let s = '';
    while (this.i < this.t.length && !isCh(this.peek(), '}')) s += this.next().v;
    this.expect(t => isCh(t, '}'), "'}'");
    return s;
  }
}

function format(ps) {
  let out = '', prev = null, bars = 0;
  for (const x of ps) {
    let cur = x;
    if (x.k === 'op' && x.s === '|') {
      cur = { k: bars++ % 2 ? 'close' : 'open', s: '|' };
      out += '|';
    } else if (x.k === 'op') {
      const unary = (x.s === '-' || x.s === '+') && (!prev || prev.k === 'op' || prev.k === 'open');
      out += x.s === ',' ? ', ' : unary || '*/^'.includes(x.s) ? x.s : ` ${x.s} `;
    } else {
      const after = prev && (prev.k === 'num' || prev.k === 'name' || prev.k === 'close');
      if (after && (x.k === 'num' || x.k === 'name' || x.k === 'fn' || (x.k === 'open' && prev.k === 'close'))) out += ' ';
      out += x.s;
    }
    prev = cur;
  }
  return out;
}

// LaTeX from the board -> { src: grapher row, name: defined name or null, note: '' or what was dropped }.
// Throws an Error with a short human message when it can't convert.
export function latexToGrapher(latex, { functions = functionNames() } = {}) {
  const src = String(latex ?? '')
    .replace(/^\s*(?:\$\$?|\\\[|\\\()|(?:\$\$?|\\\]|\\\))\s*$/g, '')
    .replace(/(?<!\\)[.,;]\s*$/, '')
    .trim();
  if (!src) throw new Error('there is no LaTeX to send');
  const units = new Conv(lex(src), new Set(functions)).seq();
  const parts = [[]];
  for (const u of units) {
    if (u.op === '=') parts.push([]);
    else parts.at(-1).push(u);
  }
  let name = null, body = parts[0], note = '';
  if (parts.length > 1) {
    if (parts[0].length === 1 && parts[0][0].name) {
      name = parts[0][0].name;
      body = parts[1];
      if (parts.length > 2) note = 'kept the first right-hand side';
    } else {
      note = 'the 3D tab has no equations, so only the left side was sent';
    }
  }
  if (!body.length) throw new Error(name ? `nothing after ${name} =` : 'there is nothing to send');
  const expr = format(stripOuter(finish(body)));
  const out = name ? `${name} = ${expr}` : expr;
  const st = parseLine(out);
  if (st?.error) throw new Error(`"${out}" isn't valid in the 3D tab (${st.error})`);
  return { src: out, name, note };
}

// ================================================================ install

export function install(api) {
  window.addEventListener('mathboard:to-graph', e => {
    const d = e.detail || {};
    d.handled = true;
    d.ok = sendToGraph(api, d.latex);
  });
  let busy = false;
  api.addToolbarButton({
    label: 'To board',
    title: 'Put a snapshot of this view on the current board page',
    group: 'more',
    icon: 'board',
    async onClick() {
      if (busy) return;
      busy = true;
      try { await snapshotToBoard(api); } finally { busy = false; }
    },
  });
}

function sendToGraph(api, latex) {
  let res;
  try { res = latexToGrapher(latex); }
  catch (err) {
    api.toast(`Can't send to 3D: ${err.message}`, 4000);
    return false;
  }
  const named = res.name ? api.rowByName(res.name) : null, last = api.rows.at(-1);
  const old = named?.src;
  let row;
  if (named) api.setRowSource(row = named, res.src);
  else if (last && !last.src.trim()) api.setRowSource(row = last, res.src);
  else row = api.addRow(res.src, { focus: false });
  api.setView('graph');
  row?.el?.li?.scrollIntoView({ block: 'nearest' });
  const what = named && old !== res.src ? `Updated ${res.src} (was ${old})` : `Added ${res.src}`;
  api.toast(res.note ? `${what}; ${res.note}` : what, 3500);
  return true;
}

async function snapshotToBoard(api) {
  const scene = api.scene;
  if (!scene) return api.toast('The 3D view is not ready');
  let shot;
  try { shot = await captureView(scene); }
  catch (err) {
    console.error('[bridge] snapshot failed:', err);
    return api.toast('Snapshot failed');
  }
  const detail = shot;
  window.dispatchEvent(new CustomEvent('mathboard:to-board', { detail }));
  api.toast(detail.handled ? `Snapshot placed on board page ${detail.page}` : 'The board is not available', 2600);
}

const loadImage = async src => {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
};

// WebGL frame + the DOM labels (KaTeX) drawn on top, as a JPEG no wider/taller than SNAP_MAX.
// While map(A) splits the view (graph/features/dual.js adds a .dual-pane next to it), the codomain
// pane and both readout heads come along.
async function captureView(scene) {
  const graph = scene.container.parentElement, pane = graph?.querySelector(':scope > .dual-pane');
  const side = pane?.querySelector('canvas') ?? null;
  let sideSrc = null;
  if (side) { // its drawing buffer is only readable right after its own frame renders
    await new Promise(res => { requestAnimationFrame(res); setTimeout(res, 250); });
    sideSrc = side.toDataURL('image/png');
  }
  scene.render();
  const frame = await loadImage(scene.snapshot());
  const main = scene.canvas.getBoundingClientRect(), sr = side?.getBoundingClientRect();
  const rect = !sr ? main : {
    left: main.left, top: Math.min(main.top, sr.top), right: Math.max(main.right, sr.right), bottom: Math.max(main.bottom, sr.bottom),
  };
  rect.width = rect.right - rect.left;
  rect.height = rect.bottom - rect.top;
  const k = Math.min(SNAP_MAX / Math.max(rect.width, rect.height), window.devicePixelRatio || 1);
  const cv = document.createElement('canvas');
  cv.width = Math.round(rect.width * k);
  cv.height = Math.round(rect.height * k);
  const g = cv.getContext('2d');
  if (pane) { // background, then the divider colour on top: it shows in the gap between the panes
    for (const c of [getComputedStyle(document.documentElement).getPropertyValue('--bg'), getComputedStyle(pane).borderLeftColor]) {
      g.fillStyle = c.trim() || '#000';
      g.fillRect(0, 0, cv.width, cv.height);
    }
  }
  g.setTransform(k, 0, 0, k, 0, 0);
  g.drawImage(frame, main.left - rect.left, main.top - rect.top, main.width, main.height);
  if (sideSrc) g.drawImage(await loadImage(sideSrc), sr.left - rect.left, sr.top - rect.top, sr.width, sr.height);
  try {
    await drawLabels(g, scene.labelLayer, rect, main);
    if (side) {
      await drawLabels(g, pane.querySelector('.g-labels'), rect, sr);
      await drawLabels(g, { children: [...graph.querySelectorAll('.dual-head')].filter(el => !el.hidden) }, rect);
    }
  } catch (err) { console.warn('[bridge] labels left out of the snapshot:', err); }
  return { src: cv.toDataURL('image/jpeg', SNAP_QUALITY), w: rect.width, h: rect.height };
}

// Copies each visible label glyph run (and KaTeX's SVG accents such as the \vec arrow) at its
// on-screen position, font and colour, with a background-coloured halo where the label has one.
// Labels with a background (the split view's readout heads) get their box drawn first.
async function drawLabels(g, layer, base, clip = base) { // clip: the pane the layer shows in
  const runs = [], range = document.createRange();
  const inside = r => r.width > 0 && r.right > clip.left && r.left < clip.right && r.bottom > clip.top && r.top < clip.bottom;
  for (const el of layer?.children ?? []) {
    if (el.style.display === 'none') continue;
    const cs = getComputedStyle(el), alpha = Number(cs.opacity);
    if (!(alpha > 0)) continue;
    if (!/^(transparent|rgba\(.*,\s*0\))$/.test(cs.backgroundColor)) {
      const r = el.getBoundingClientRect();
      runs.push({ box: cs.backgroundColor, x: r.left - base.left, top: r.top - base.top, w: r.width, h: r.height,
        radius: parseFloat(cs.borderTopLeftRadius) || 0, alpha });
    }
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.nodeValue.trim() || n.parentElement.closest('.katex-mathml')) continue;
      range.selectNodeContents(n);
      const r = range.getBoundingClientRect();
      if (!inside(r)) continue;
      const cs = getComputedStyle(n.parentElement);
      runs.push({
        text: n.nodeValue, x: r.left - base.left, top: r.top - base.top, h: r.height, alpha, color: cs.color,
        font: `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`, halo: cs.textShadow !== 'none',
      });
    }
    for (const svg of el.querySelectorAll('svg')) {
      const r = svg.getBoundingClientRect();
      if (!inside(r) || !r.height) continue;
      const color = getComputedStyle(svg).color, copy = svg.cloneNode(true);
      copy.setAttribute('width', r.width);
      copy.setAttribute('height', r.height);
      copy.setAttribute('style', `fill: ${color}; stroke: ${color}; stroke-width: 0`);
      const img = await loadImage(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(copy))}`);
      runs.push({ img, x: r.left - base.left, top: r.top - base.top, w: r.width, h: r.height, alpha });
    }
  }
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#000';
  g.textBaseline = 'alphabetic';
  g.lineJoin = 'round';
  for (const pass of ['box', 'halo', 'fill']) {
    for (const t of runs) {
      g.globalAlpha = t.alpha;
      if (t.box) {
        if (pass !== 'box') continue;
        g.fillStyle = t.box;
        g.beginPath();
        if (g.roundRect) g.roundRect(t.x, t.top, t.w, t.h, t.radius);
        else g.rect(t.x, t.top, t.w, t.h);
        g.fill();
        continue;
      }
      if (pass === 'box') continue;
      if (t.img) {
        if (pass === 'fill') g.drawImage(t.img, t.x, t.top, t.w, t.h);
        continue;
      }
      if (pass === 'halo' && !t.halo) continue;
      g.font = t.font;
      const m = g.measureText(t.text);
      const y = t.top + (t.h - m.fontBoundingBoxAscent - m.fontBoundingBoxDescent) / 2 + m.fontBoundingBoxAscent;
      if (pass === 'halo') {
        g.strokeStyle = bg;
        g.lineWidth = 5;
        g.strokeText(t.text, t.x, y);
      } else {
        g.fillStyle = t.color;
        g.fillText(t.text, t.x, y);
      }
    }
  }
  g.globalAlpha = 1;
}
