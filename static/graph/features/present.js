// Presentation polish for the 3D tab:
//   - pretty rows: rows that aren't being edited show their expression typeset with KaTeX
//   - snapshot (P): PNG of the view with its KaTeX labels composited in
//   - recording (V): the same compositing every frame, fed to MediaRecorder (mp4 if possible, else webm)
// rowLatex / exprLatex / nameLatex are pure (no DOM) and unit-tested in tests/present.test.mjs.
import { parseLine } from '../lang.js';

// ================================================================ AST -> LaTeX
const SUM = 1, PROD = 2, POW = 3, ATOM = 4;

const GREEK = new Set(('alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi ' +
  'pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega ' +
  'Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega').split(' '));
const UNI = {}; // unicode Greek -> LaTeX
[...'αβγδεζηθικλμνξοπρςστυφχψω'].forEach((ch, i) => {
  UNI[ch] = ('\\alpha \\beta \\gamma \\delta \\varepsilon \\zeta \\eta \\theta \\iota \\kappa \\lambda \\mu \\nu \\xi o ' +
    '\\pi \\rho \\varsigma \\sigma \\tau \\upsilon \\varphi \\chi \\psi \\omega').split(' ')[i];
});
[...'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'].forEach((ch, i) => {
  UNI[ch] = 'A B \\Gamma \\Delta E Z H \\Theta I K \\Lambda M N \\Xi O \\Pi P \\Sigma T \\Upsilon \\Phi X \\Psi \\Omega'.split(' ')[i];
});
Object.assign(UNI, { 'ϑ': '\\vartheta', 'ϕ': '\\phi', 'ϖ': '\\varpi', 'ϵ': '\\epsilon', 'ϱ': '\\varrho' });

const CONST_TEX = { pi: '\\pi', 'π': '\\pi', e: 'e', i: '\\hat{\\imath}', j: '\\hat{\\jmath}', k: '\\hat{k}' };
const CONST_KIND = { pi: 'num', 'π': 'num', e: 'num', i: 'vec', j: 'vec', k: 'vec' };
const NORMS = new Set(['norm', 'length', 'mag']);
const FN_NAME = { asin: 'arcsin', acos: 'arccos', atan: 'arctan', σ: '\\sigma' };
const FN_KIND = {
  dot: 'num', angle: 'num', det: 'num', deg: 'num', rad: 'num', sin: 'num', cos: 'num', tan: 'num', asin: 'num',
  acos: 'num', atan: 'num', sqrt: 'num', exp: 'num', ln: 'num', log: 'num', min: 'num', max: 'num',
  norm: 'num', length: 'num', mag: 'num', abs: 'num',
  sinh: 'num', cosh: 'num', tanh: 'num', sigmoid: 'num', σ: 'num', relu: 'num', leakyrelu: 'num', leaky_relu: 'num',
  elu: 'num', gelu: 'num', softplus: 'num', silu: 'num', swish: 'num', mish: 'num', erf: 'num', heaviside: 'num',
  sign: 'num', floor: 'num', ceil: 'num', round: 'num', mod: 'num', logsumexp: 'num', softmax: 'vec',
  cross: 'vec', unit: 'vec', normalize: 'vec', proj: 'vec',
  inv: 'mat', transpose: 'mat', matrix: 'mat', point: 'point', span: 'span', plane: 'plane',
};

// Plain brackets hug ordinary content; \left( only for content that needs taller ones (a \vec
// accent alone would bump \left( up a size). \mathopen{} drops the thin space TeX puts between a
// function name and \left(.
const TALL = /\\(?:begin|frac|sqrt|left|overrightarrow|widehat)/;
const paren = s => (TALL.test(s) ? `\\mathopen{}\\left(${s}\\right)` : `(${s})`);
const braces = s => (TALL.test(s) ? `\\mathopen{}\\left\\{${s}\\right\\}` : `\\{${s}\\}`);
const fence = (s, vector) => (vector ? `\\left\\lVert ${s}\\right\\rVert` : `\\left|${s}\\right|`);
const NAME_CALL = /(?<![\w.Ͱ-Ͽ])([A-Za-zͰ-Ͽ][\wͰ-Ͽ]*)\s*\(/gu;
const macroSafe = t => (t.startsWith('\\') ? `{${t}}` : t);

function nameParts(name) {
  const m = /^(.+?)(?:_(\w+)|(\d+))$/.exec(name);
  return m ? { base: m[1], sub: m[2] ?? m[3] } : { base: name, sub: '' };
}
const isWordy = base => [...base].length > 1 && !GREEK.has(base);

function baseLatex(s) {
  if (GREEK.has(s)) return `\\${s}`;
  const chars = [...s];
  if (chars.length === 1) return UNI[s] ?? s;
  return `\\mathrm{${chars.map(ch => (UNI[ch] ? macroSafe(UNI[ch]) : ch === '_' ? '\\_' : ch)).join('')}}`;
}
const subLatex = sub => (/^\d+$/.test(sub) ? sub : baseLatex(sub));

// v1 / v_1 -> v_{1}; theta -> \theta; long names upright; vector names get an arrow.
export function nameLatex(name, vector = false) {
  const { base, sub } = nameParts(name);
  let b = baseLatex(base);
  if (vector) b = isWordy(base) ? `\\overrightarrow{${b}}` : `\\vec{${b}}`;
  return sub ? `${b}_{${subLatex(sub)}}` : b;
}

function numLatex(v) {
  const s = String(v);
  const m = /^([\d.]+)e([+-])(\d+)$/.exec(s);
  return m ? `${m[1]} \\times 10^{${m[2] === '-' ? '-' : ''}${m[3]}}` : s;
}

const isConst = (name, c) => !c.isUser(name) && Object.hasOwn(CONST_TEX, name);

// Best-effort static type of an expression: 'num' | 'vec' | 'point' | 'mat' | ... | null.
export function inferKind(n, c) {
  const k = x => inferKind(x, c);
  switch (n.t) {
    case 'num': case 'abs': case 'comp': return 'num';
    case 'vec': return 'vec';
    case 'mat': return 'mat';
    case 'name': return c.kinds.get(n.name) ?? (isConst(n.name, c) ? CONST_KIND[n.name] : null);
    case 'neg': return k(n.a);
    case 'call': return FN_KIND[n.name] ?? null;
    case 'bin': {
      const a = k(n.a), b = k(n.b);
      if (n.op === '/' || n.op === '^') return a;
      if (n.op === '+' || n.op === '-') {
        if (a === 'point' || b === 'point') return a === 'point' && b === 'point' ? 'vec' : 'point';
        return a === 'num' ? b : a ?? b;
      }
      if (n.op === '·' && a === 'vec' && b === 'vec') return 'num';
      if (a === 'num') return b;
      if (b === 'num') return a;
      if (a === 'mat') return b;
      return null;
    }
  }
  return null;
}

// Returns { s: latex, l: precedence level, lead: 'num'|'frac'|null, wl/wr: starts/ends with an
// upright word, imp: implicit product, neg: leading unary minus, frac: \frac }.
function tex(n, c) {
  switch (n.t) {
    case 'num': {
      const s = numLatex(n.v);
      return { s, l: s.includes('\\times') ? PROD : ATOM, lead: 'num' };
    }
    case 'name': {
      if (isConst(n.name, c)) return { s: CONST_TEX[n.name], l: ATOM };
      const w = isWordy(nameParts(n.name).base), d = "'".repeat(n.d ?? 0); // sigmoid' on its own
      return { s: nameLatex(n.name, c.kinds.get(n.name) === 'vec') + d, l: ATOM, wl: w, wr: w };
    }
    case 'vec': {
      const items = n.items.map(x => tex(x, c).s);
      return { s: c.inline ? paren(items.join(', ')) : `\\begin{pmatrix}${items.join(' \\\\ ')}\\end{pmatrix}`, l: ATOM };
    }
    case 'mat':
      return { s: `\\begin{bmatrix}${n.rows.map(r => r.map(x => tex(x, c).s).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`, l: ATOM };
    case 'neg': {
      const a = tex(n.a, c);
      return { s: `-${a.l === SUM || a.neg ? paren(a.s) : a.s}`, l: PROD, neg: true };
    }
    case 'abs': return { s: fence(tex(n.a, c).s, inferKind(n.a, c) === 'vec'), l: ATOM };
    case 'comp': {
      const xyz = 'xyz'[n.i];
      if (n.a.t === 'name' && !isConst(n.a.name, c)) {
        const { base, sub } = nameParts(n.a.name);
        return { s: `${baseLatex(base)}_{${sub ? `${subLatex(sub)},` : ''}${xyz}}`, l: ATOM };
      }
      const a = tex(n.a, c); // softmax(z).x -> softmax(z)_x, a call already has its brackets
      return { s: `${n.a.t === 'call' && a.l === ATOM && /\)$/.test(a.s) ? a.s : paren(a.s)}_{${xyz}}`, l: ATOM };
    }
    case 'bin': return binTex(n, c);
    case 'call': return callTex(n, c);
  }
  return { s: '?', l: ATOM };
}

function binTex(n, c) {
  const { op } = n;
  // `f(x)` with no function f parses as f times x: show it as the call the user typed
  if (n.implicit && n.a.t === 'name' && c.calls?.has(n.a.name) && !c.isUser(n.a.name) && !isConst(n.a.name, c)) {
    const args = n.b.t === 'vec' ? n.b.items : [n.b];
    return { s: `${nameLatex(n.a.name)}${paren(args.map(x => tex(x, c).s).join(', '))}`, l: ATOM };
  }
  const a = tex(n.a, c);
  if (op === '^') {
    const e = tex(n.b, { ...c, inline: true });
    const wrap = a.l < ATOM || a.frac;
    return { s: `${wrap ? paren(a.s) : a.s}^{${e.s}}`, l: POW, lead: wrap ? null : a.lead, wl: a.wl };
  }
  const b = tex(n.b, c);
  if (op === '+' || op === '-') {
    return { s: `${a.s} ${op} ${b.l === SUM || b.neg ? paren(b.s) : b.s}`, l: SUM, lead: a.lead, wl: a.wl };
  }
  if (op === '/' && !c.inline) return { s: `\\frac{${a.s}}{${b.s}}`, l: ATOM, lead: 'frac', frac: true };
  // * / · × and juxtaposition
  const wrapA = a.l === SUM;
  const wrapB = b.l === SUM || b.neg || (op === '/' ? b.l < POW : b.l === PROD && !b.imp);
  let sep;
  if (n.implicit) {
    const lead = wrapB ? null : b.lead;
    sep = lead === 'num' || lead === 'frac' ? ' \\cdot ' : (!wrapA && a.wr) || (!wrapB && b.wl) ? '\\,' : '';
  } else {
    sep = { '*': ' \\cdot ', '·': ' \\cdot ', '×': ' \\times ', '/': ' / ' }[op];
  }
  const left = wrapA ? paren(a.s) : a.s, right = wrapB ? paren(b.s) : b.s;
  if (!sep && /\\[A-Za-z]+$/.test(left) && /^[A-Za-z]/.test(right)) sep = ' '; // eta a -> \eta a, not \etaa
  return {
    s: `${left}${sep}${right}`,
    l: PROD, imp: !!n.implicit, lead: wrapA ? null : a.lead, wl: !wrapA && a.wl, wr: !wrapB && b.wr,
  };
}

function callTex(n, c) {
  const { name, args } = n;
  const all = () => args.map(x => tex(x, c).s);
  const one = args.length === 1;
  if (name === 'sqrt' && one) return { s: `\\sqrt{${tex(args[0], c).s}}`, l: ATOM };
  if ((name === 'abs' || NORMS.has(name)) && one) {
    const k = inferKind(args[0], c);
    return { s: fence(tex(args[0], c).s, k === 'vec' || (k == null && name !== 'abs')), l: ATOM };
  }
  if ((name === 'dot' || name === 'cross') && args.length === 2) {
    return binTex({ t: 'bin', op: name === 'dot' ? '·' : '×', a: args[0], b: args[1] }, c);
  }
  if ((name === 'inv' || name === 'transpose') && one) {
    const a = tex(args[0], c);
    const base = a.l < ATOM || a.frac ? paren(a.s) : a.s;
    return { s: `${base}^{${name === 'inv' ? '-1' : '\\mathsf{T}'}}`, l: POW, wl: a.wl };
  }
  if ((name === 'unit' || name === 'normalize') && one && args[0].t === 'name' && !isConst(args[0].name, c)) {
    const { base, sub } = nameParts(args[0].name);
    const hat = `\\${isWordy(base) ? 'widehat' : 'hat'}{${baseLatex(base)}}`;
    return { s: sub ? `${hat}_{${subLatex(sub)}}` : hat, l: ATOM };
  }
  if (name === 'proj' && args.length === 2) {
    const u = tex(args[0], c), v = tex(args[1], { ...c, inline: true });
    return { s: `\\operatorname{proj}_{${v.s}}${u.l === ATOM && !u.frac ? u.s : paren(u.s)}`, l: POW };
  }
  if (name === 'span' && args.length) return { s: `\\operatorname{span}${braces(all().join(', '))}`, l: ATOM };
  if (name === 'matrix' && args.length) {
    if (args.every(a => a.t === 'vec')) { // literal columns -> the actual matrix
      const rows = Math.max(...args.map(a => a.items.length));
      const cell = (a, i) => (a.items[i] ? tex(a.items[i], c).s : '0');
      const body = Array.from({ length: rows }, (_, i) => args.map(a => cell(a, i)).join(' & ')).join(' \\\\ ');
      return { s: `\\begin{bmatrix}${body}\\end{bmatrix}`, l: ATOM };
    }
    return { s: `\\begin{bmatrix}${all().join(' & ')}\\end{bmatrix}`, l: ATOM };
  }
  const d = "'".repeat(n.d ?? 0); // f'(x), sigmoid''(x)
  if (n.user) return { s: `${nameLatex(name, false)}${d}${paren(all().join(', '))}`, l: ATOM };
  const fn = FN_NAME[name] ?? name.replace(/_/g, '\\_');
  return { s: `\\operatorname{${fn}}${d}${paren(all().join(', '))}`, l: ATOM };
}

function context(opts = {}, self = null) {
  const kinds = opts.kinds instanceof Map ? opts.kinds : new Map(Object.entries(opts.kinds ?? {}));
  const defined = opts.defined ?? new Set(kinds.keys());
  return { kinds, inline: !!opts.inline, isUser: name => name === self || defined.has(name) };
}

// LaTeX for one AST node. opts: { kinds: Map|object name -> value kind, defined: Set of row names, inline }
export function exprLatex(node, opts = {}) { return tex(node, context(opts)).s; }

// LaTeX for one editor line: { tex, at, note } (at = origin after @, note = trailing comment),
// or null for blank lines and parse errors (those rows stay raw). opts.fns: the names defined as
// f(x) = ... on other rows, so f(2) and f' parse as the evaluator reads them.
export function rowLatex(src, opts = {}) {
  const text = String(src ?? '');
  const ci = text.search(/#|\/\//);
  const note = ci < 0 ? '' : text.slice(ci).replace(/^(?:#|\/\/)\s*/, '').trim();
  const st = parseLine(text, opts.fns);
  if (!st) return note ? { tex: '', at: '', note } : null;
  if (st.error) return null;
  const c = context(opts, st.name);
  c.calls = new Set([...(ci < 0 ? text : text.slice(0, ci)).matchAll(NAME_CALL)].map(m => m[1]));
  let lhs = '';
  if (st.params) lhs = `${nameLatex(st.name, false)}${paren(st.params.map(q => nameLatex(q, false)).join(', '))} = `; // f(x) = ...
  else if (st.name) lhs = `${nameLatex(st.name, (c.kinds.get(st.name) ?? inferKind(st.body, c)) === 'vec')} = `;
  return { tex: lhs + tex(st.body, c).s + whereLatex(st.where, c), at: st.at ? tex(st.at, { ...c, inline: true }).s : '', note };
}

// A restriction, {0 < x <= 1}, after the expression as Desmos writes it.
const REL_TEX = { '<': '<', '>': '>', '<=': '\\le', '>=': '\\ge' };
function whereLatex(where, c) {
  if (!where?.length) return '';
  const ic = { ...c, inline: true };
  const conds = where.map(({ items, ops }) => items.map((n, i) => (i ? ` ${REL_TEX[ops[i - 1]]} ` : '') + tex(n, ic).s).join(''));
  return `\\ \\{${conds.join(',\\ ')}\\}`;
}

// ================================================================ label compositing
const PAD = 12; // px around each label in the atlas (text-shadow glow)
const TF = /translate\((-?[\d.e+-]+)%, (-?[\d.e+-]+)%\) translate\((-?[\d.e+-]+)px, (-?[\d.e+-]+)px\)(?: rotate\((-?[\d.e+-]+)rad\))?/;
const COPY = ['color', 'font-family', 'font-size', 'font-style', 'font-weight', 'line-height', 'letter-spacing',
  'text-shadow', 'padding', 'background-color', 'border', 'border-radius', 'text-align'];
const cells = new Map(); // label key -> { src canvas, sx, sy, sw, sh, w, h, base opacity }
let atlasBroken = false, rasterizing = null, lastRaster = 0, cssPromise = null;

const themeName = () => document.documentElement.dataset.theme || 'dark';
const labelKey = (el, scale) =>
  `${themeName()}|${scale}|${el.className}|${el.style.color}|${el.dataset.latex ?? el.textContent}`;

// KaTeX CSS with its woff2 fonts inlined, so an SVG <foreignObject> image can render labels.
function katexCss() {
  cssPromise ||= (async () => {
    const link = [...document.querySelectorAll('link[rel="stylesheet"]')].find(l => /katex/i.test(l.href));
    if (!link) throw new Error('KaTeX stylesheet not found');
    let css = await (await fetch(link.href)).text();
    const faces = css.match(/@font-face\s*{[^}]*}/g) || [];
    const inlined = await Promise.all(faces.map(async face => {
      const m = /url\(["']?([^"')]+\.woff2)["']?\)/.exec(face);
      if (!m) return '';
      const blob = await (await fetch(new URL(m[1], link.href))).blob();
      const url = await new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(String(fr.result).replace(/^data:[^;,]*/, 'data:font/woff2'));
        fr.onerror = rej;
        fr.readAsDataURL(blob);
      });
      return face.replace(/src:[^;}]*/, `src:url(${url}) format("woff2")`);
    }));
    faces.forEach((face, i) => { css = css.replace(face, inlined[i]); });
    return css.replaceAll(']]>', '');
  })();
  cssPromise.catch(() => { cssPromise = null; });
  return cssPromise;
}

// The atlas SVG only carries KaTeX's CSS: give the other elements inside a label (the split view's
// readout heads have a few) their computed look inline.
const INNER = ['display', 'color', 'font-size', 'font-style', 'font-weight', 'line-height', 'text-align'];
function inlineInner(el, clone) {
  const src = el.querySelectorAll('*'), dst = clone.querySelectorAll('*');
  src.forEach((x, i) => {
    if (x.closest('.katex')) return;
    const cs = getComputedStyle(x);
    dst[i].setAttribute('style', INNER.map(p => `${p}:${cs.getPropertyValue(p)}`).join(';'));
  });
}

// What a capture shows: the 3D view and, while map(A) splits it (graph/features/dual.js puts a
// .dual-pane next to the view), the codomain pane and both readout heads. x, y: CSS px from `box`.
function captureArea(scene) {
  const graph = scene.container.parentElement, pane = graph?.querySelector(':scope > .dual-pane');
  const panes = [{ canvas: scene.canvas, layer: scene.labelLayer }];
  const side = pane?.querySelector('canvas');
  if (side) panes.push({ canvas: side, layer: pane.querySelector('.g-labels') });
  const rects = panes.map(p => p.canvas.getBoundingClientRect());
  const left = Math.min(...rects.map(r => r.left)), top = Math.min(...rects.map(r => r.top));
  const box = { left, top, width: Math.max(...rects.map(r => r.right)) - left, height: Math.max(...rects.map(r => r.bottom)) - top };
  panes.forEach((p, i) => Object.assign(p, { x: rects[i].left - left, y: rects[i].top - top, w: rects[i].width, h: rects[i].height }));
  const heads = side ? [...graph.querySelectorAll('.dual-head')].filter(el => !el.hidden && el.offsetWidth) : [];
  const css = getComputedStyle(document.documentElement);
  return { panes, heads, box, bg: css.getPropertyValue('--bg').trim() || '#000', line: side ? getComputedStyle(pane).borderLeftColor : null };
}

// Resolves inside the next animation frame, after the scenes' own frame callbacks have rendered:
// the only moment another scene's canvas (whose drawing buffer isn't preserved) can be read.
const afterRender = () => new Promise(res => {
  requestAnimationFrame(() => res());
  setTimeout(res, 250); // hidden tab: no frames
});

function visibleLabels(layer) {
  const out = [];
  if (!layer || layer.style.display === 'none') return out;
  for (const el of layer.children) {
    if (el.style.display === 'none') continue;
    const m = TF.exec(el.style.transform);
    if (m) out.push({ el, m, z: Number(el.style.zIndex) || 0 });
  }
  return out.sort((a, b) => a.z - b.z);
}

// Renders the given labels into one atlas canvas (one SVG decode for the whole batch).
async function rasterize(els, scale) {
  if (atlasBroken) return;
  const css = await katexCss();
  const seen = new Set();
  let batch = [], y = 0, W = 0;
  const flush = async () => {
    if (!batch.length) return;
    const parts = batch.map(b => b.html).join('');
    const H = y;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><foreignObject width="100%" height="100%">` +
      `<div xmlns="http://www.w3.org/1999/xhtml" style="position:relative;width:${W}px;height:${H}px">` +
      `<style><![CDATA[${css}]]></style>${parts}</div></foreignObject></svg>`;
    const img = new Image();
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`; // a blob: URL would taint the canvas
    await img.decode();
    const cv = document.createElement('canvas');
    cv.width = Math.ceil(W * scale);
    cv.height = Math.ceil(H * scale);
    const g = cv.getContext('2d');
    g.drawImage(img, 0, 0, cv.width, cv.height);
    g.getImageData(0, 0, 1, 1); // throws if the SVG tainted the canvas
    if (cells.size > 800) cells.clear();
    for (const b of batch) {
      cells.set(b.key, { src: cv, sx: 0, sy: b.y * scale, sw: (b.w + 2 * PAD) * scale, sh: (b.h + 2 * PAD) * scale, w: b.w, h: b.h, base: b.base });
    }
    batch = [];
    y = 0;
    W = 0;
  };
  const ser = new XMLSerializer();
  for (const el of els) {
    const key = labelKey(el, scale);
    if (seen.has(key) || cells.has(key)) continue;
    seen.add(key);
    const w = el.offsetWidth, h = el.offsetHeight;
    if (!w || !h) continue;
    const cs = getComputedStyle(el);
    const clone = el.cloneNode(true);
    const style = [`position:absolute`, `left:${PAD}px`, `top:${y + PAD}px`, `width:${w}px`, `height:${h}px`,
      'margin:0', 'box-sizing:border-box', 'white-space:nowrap', 'opacity:1'];
    for (const p of COPY) style.push(`${p}:${cs.getPropertyValue(p)}`);
    clone.setAttribute('style', style.join(';'));
    inlineInner(el, clone);
    const kx = el.querySelector('.katex'), kc = clone.querySelector('.katex');
    if (kx && kc) kc.style.fontSize = getComputedStyle(kx).fontSize;
    batch.push({ key, w, h, y, html: ser.serializeToString(clone), base: el.style.opacity === '' ? Number(cs.opacity) : 1 });
    y += h + 2 * PAD;
    W = Math.max(W, w + 2 * PAD);
    if (y > 3000) await flush();
  }
  await flush();
}

function rasterizeSoon(els, scale) {
  const now = performance.now();
  if (!els.length || atlasBroken || rasterizing || now - lastRaster < 250) return rasterizing;
  lastRaster = now;
  rasterizing = rasterize(els, scale)
    .catch(err => { atlasBroken = true; console.warn('[present] labels fall back to plain text:', err); })
    .finally(() => { rasterizing = null; });
  return rasterizing;
}

async function prepareLabels(area, scale) {
  if (rasterizing) await rasterizing;
  const els = [...area.panes.flatMap(p => visibleLabels(p.layer).map(l => l.el)), ...area.heads]
    .filter(el => !cells.has(labelKey(el, scale)));
  lastRaster = 0;
  await rasterizeSoon(els, scale);
}

// Plain-text stand-in while a label's atlas cell is being made (or if SVG rasterization fails).
const textStyles = new Map(); // label key -> cached fallback style (getComputedStyle once per label)
function textStyle(el, key) {
  let st = textStyles.get(key);
  if (!st) {
    const cs = getComputedStyle(el);
    const text = (el.querySelector('.katex-html') ?? el).textContent.replace(/\p{Cf}/gu, '').trim();
    const glow = cs.textShadow !== 'none' && /rgba?\([^)]*\)|#[0-9a-f]{3,8}/i.exec(cs.textShadow);
    st = {
      text, px: parseFloat(cs.fontSize) * 1.21, italic: /^[A-Za-z]$/.test(text), color: cs.color, glow: glow ? glow[0] : null,
      arrow: /\\(vec|overrightarrow)/.test(el.dataset.latex || ''), base: el.style.opacity === '' ? Number(cs.opacity) : 1,
    };
    if (textStyles.size > 800) textStyles.clear();
    textStyles.set(key, st);
  }
  return st;
}

function drawText(g, st, k) {
  if (!st.text) return;
  const px = st.px * k;
  g.font = `${st.italic ? 'italic ' : ''}${px}px KaTeX_Main, "Times New Roman", serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  if (st.glow) { // halo instead of the CSS text-shadow (a blur per label per frame is too slow)
    g.strokeStyle = st.glow;
    g.lineJoin = 'round';
    g.lineWidth = 0.3 * px;
    g.strokeText(st.text, 0, 0);
  }
  g.fillStyle = st.color;
  g.fillText(st.text, 0, 0);
  if (st.arrow) {
    const w = g.measureText(st.text).width, y = -0.55 * px, hw = Math.max(w / 2, 0.25 * px);
    g.strokeStyle = st.color;
    g.lineWidth = Math.max(1, 0.06 * px);
    g.beginPath();
    g.moveTo(-hw, y); g.lineTo(hw, y);
    g.moveTo(hw - 0.15 * px, y - 0.1 * px); g.lineTo(hw, y); g.lineTo(hw - 0.15 * px, y + 0.1 * px);
    g.stroke();
  }
}

// Draws the WebGL canvases of a captureArea() plus every visible label into a 2D context of size
// W x H and returns the labels that had no atlas cell yet. Must run in the same task as the WebGL
// renders (the drawing buffers aren't preserved): see afterRender.
function compose(g, W, H, area, scale) {
  const { panes, heads, box } = area;
  const k = Math.min(W / box.width, H / box.height);
  g.fillStyle = area.bg;
  g.fillRect(0, 0, W, H);
  const missing = [];
  for (const p of panes) {
    g.drawImage(p.canvas, p.x * k, p.y * k, p.w * k, p.h * k);
    if (p !== panes[0] && area.line) {
      g.fillStyle = area.line;
      g.fillRect((p.x - 1) * k, p.y * k, Math.max(1, k), p.h * k);
    }
    g.save(); // labels stay inside their pane, as on screen
    g.beginPath();
    g.rect(p.x * k, p.y * k, p.w * k, p.h * k);
    g.clip();
    for (const { el, m } of visibleLabels(p.layer)) {
      const key = labelKey(el, scale);
      const cell = cells.get(key);
      const st = cell ? null : textStyle(el, key);
      const alpha = el.style.opacity !== '' ? Number(el.style.opacity) : (cell ?? st).base;
      if (!(alpha > 0)) continue;
      g.save();
      g.globalAlpha = Math.min(1, alpha);
      g.translate((p.x + Number(m[3])) * k, (p.y + Number(m[4])) * k);
      if (m[5] && Number(m[5])) g.rotate(Number(m[5]));
      if (cell) {
        g.drawImage(cell.src, cell.sx, cell.sy, cell.sw, cell.sh,
          ((Number(m[1]) / 100) * cell.w - PAD) * k, ((Number(m[2]) / 100) * cell.h - PAD) * k,
          (cell.w + 2 * PAD) * k, (cell.h + 2 * PAD) * k);
      } else {
        drawText(g, st, k);
        missing.push(el);
      }
      g.restore();
    }
    g.restore();
  }
  for (const el of heads) { // no plain-text stand-in: a head appears once its cell is ready
    const cell = cells.get(labelKey(el, scale));
    if (!cell) { missing.push(el); continue; }
    const r = el.getBoundingClientRect();
    g.drawImage(cell.src, cell.sx, cell.sy, cell.sw, cell.sh, (r.left - box.left - PAD) * k, (r.top - box.top - PAD) * k,
      (cell.w + 2 * PAD) * k, (cell.h + 2 * PAD) * k);
  }
  return missing;
}

// ================================================================ UI
const PRETTY_KEY = 'mathboard.present.pretty';
const FRAME_MS = 1000 / 60;
const STOP_GRACE_MS = 500, STOP_MAX_WAIT_MS = 8000;
const MIMES = ['video/mp4;codecs=avc1.640028', 'video/mp4;codecs=avc1', 'video/mp4',
  'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];

const CSS = `
.pp-tex { display: none; }
#g-panel.pp-on .g-row.pp-ok:not(.pp-edit) .pp-tex {
  display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 0.6em;
  min-height: 30px; padding: 4px 2px; cursor: text;
  font-size: 17px; line-height: 1.3; overflow-x: auto; overflow-y: hidden; scrollbar-width: thin;
}
/* a comment that doesn't fit beside the maths wraps onto its own line under it */
.pp-math, .pp-at { flex: none; white-space: nowrap; }
.pp-math:empty { display: none; }
#g-panel.pp-on .g-row.pp-ok:not(.pp-edit) .g-src {
  position: absolute; width: 1px; height: 1px; padding: 0; opacity: 0; pointer-events: none;
}
.pp-tex .katex { font-size: 1.08em; }
.pp-at { color: var(--text-3); font-size: 0.72em; }
.pp-at .katex { font-size: 1.15em; }
.pp-note { color: var(--text-3); font: var(--fs-md) var(--font-ui); }
#g-tools .ui-btn.pp-recording, #g-tools .ui-btn.pp-recording:hover { color: var(--danger); background: var(--danger-soft); }
.pp-rec {
  position: absolute; top: 12px; right: 12px; z-index: 4; pointer-events: none;
  display: flex; align-items: center; gap: 7px; height: 30px; padding: 0 13px 0 11px; border-radius: var(--r-pill);
  background: var(--float); color: var(--text-1); border: 1px solid var(--float-line); box-shadow: var(--shadow-2);
  font: var(--fw-strong) var(--fs-sm)/1 var(--font-ui); font-variant-numeric: tabular-nums; letter-spacing: 0.04em;
}
.pp-rec[hidden] { display: none; }
.pp-rec i { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); animation: pp-pulse 1.2s var(--ease) infinite; }
.pp-rec span { font-weight: var(--fw-medium); color: var(--text-2); letter-spacing: 0; }
@keyframes pp-pulse { 50% { opacity: 0.25; } }
`;

const esc = s => s.replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);

function stamp() {
  const d = new Date(), p = x => String(x).padStart(2, '0');
  return `mathboard-3d-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export function install(api) {
  api.addStyles(CSS);
  const panel = api.panelEl;
  let pretty = true;
  try { pretty = localStorage.getItem(PRETTY_KEY) !== '0'; } catch { /* default on */ }
  panel.classList.toggle('pp-on', pretty);

  // ---------------------------------------------------------------- pretty rows
  let kindsFor = null, kindsInfo = null, sticky = new Map();
  // name -> value kind; a row that is temporarily broken keeps its last kind so arrows don't flicker
  function currentKinds() {
    const res = api.results;
    if (res === kindsFor) return kindsInfo;
    const defined = new Set(res.map(r => r?.name).filter(Boolean));
    const kinds = new Map([...sticky].filter(([n]) => defined.has(n)));
    for (const r of res) if (r?.name && !r.error && r.value != null) kinds.set(r.name, api.lang.values.kindOf(r.value));
    const fns = new Set(api.rows.map(r => parseLine(r.src)).filter(st => st?.params && st.name).map(st => st.name));
    sticky = kinds;
    kindsFor = res;
    kindsInfo = { kinds, defined, fns };
    return kindsInfo;
  }

  function renderRow(row, el) {
    const box = el.ppTex;
    let info = null;
    try { info = globalThis.katex ? rowLatex(row.src, currentKinds()) : null; } catch (err) { console.error('[present]', err); }
    el.li.classList.toggle('pp-ok', !!info);
    const key = info ? `${info.tex}\u0000${info.at}\u0000${info.note}` : '';
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    if (!info) { box.textContent = ''; return; }
    const k = s => katex.renderToString(`\\displaystyle ${s}`, { throwOnError: false });
    box.innerHTML = `<span class="pp-math">${info.tex ? k(info.tex) : ''}</span>` +
      (info.at ? `<span class="pp-at">from ${k(info.at)}</span>` : '') +
      (info.note ? `<span class="pp-note">${esc(info.note)}</span>` : '');
  }

  function decorate(row, res, el) {
    if (!el.ppTex) {
      const box = el.ppTex = document.createElement('div');
      box.className = 'pp-tex';
      box.title = 'Click to edit';
      el.main.insertBefore(box, el.src);
      box.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        e.preventDefault();
        el.li.classList.add('pp-edit');
        el.src.focus();
        const n = el.src.value.length;
        el.src.setSelectionRange(n, n);
      });
      el.src.addEventListener('focus', () => el.li.classList.add('pp-edit'));
      el.src.addEventListener('blur', () => el.li.classList.remove('pp-edit'));
      el.li.classList.toggle('pp-edit', document.activeElement === el.src);
    }
    if (pretty) renderRow(row, el);
  }
  api.addRowDecorator(decorate);

  const prettyBtn = api.addToolbarButton({
    label: 'TeX', title: 'Typeset rows while they are not being edited', group: 'display',
    onClick: () => {
      pretty = !pretty;
      try { localStorage.setItem(PRETTY_KEY, pretty ? '1' : '0'); } catch { /* private mode */ }
      panel.classList.toggle('pp-on', pretty);
      prettyBtn.classList.toggle('on', pretty);
      if (pretty) api.rows.forEach(r => r.el && decorate(r, null, r.el));
    },
  });
  prettyBtn.classList.toggle('on', pretty);

  // ---------------------------------------------------------------- snapshot
  const pixelRatio = scene => scene.renderer.getPixelRatio?.() || window.devicePixelRatio || 1;
  let shooting = false;
  async function snapshot() {
    const scene = api.scene;
    if (!scene || shooting || api.view !== 'graph') return;
    shooting = true;
    try {
      const scale = pixelRatio(scene);
      await prepareLabels(captureArea(scene), scale);
      await afterRender();
      const area = captureArea(scene), out = document.createElement('canvas');
      out.width = Math.round(area.box.width * scale);
      out.height = Math.round(area.box.height * scale);
      scene.render();
      compose(out.getContext('2d'), out.width, out.height, area, scale);
      const blob = await new Promise(res => out.toBlob(res, 'image/png'));
      if (!blob) throw new Error('could not encode the image');
      download(blob, `${stamp()}.png`);
      api.toast('Snapshot saved');
    } catch (err) {
      console.error('[present] snapshot:', err);
      api.toast('Snapshot failed');
    } finally {
      shooting = false;
    }
  }

  // ---------------------------------------------------------------- recording
  api.addToolbarButton({ label: 'PNG', title: 'Download a PNG of the 3D view (P)', onClick: () => snapshot(), group: 'output', icon: 'image' });
  const REC_TITLE = 'Record the 3D view to a video file (V)';
  const recBtn = api.addToolbarButton({ label: 'Rec', title: REC_TITLE, onClick: () => toggleRecording(), group: 'output', icon: 'record' });
  const badge = document.createElement('div');
  badge.className = 'pp-rec';
  badge.hidden = true;
  badge.innerHTML = '<i></i>REC <span>0:00</span>';
  api.addOverlay(badge);
  const badgeTime = badge.querySelector('span');

  let rec = null;
  const paintRec = () => {
    const on = !!rec?.recorder;
    recBtn.innerHTML = api.icon?.(on ? 'stop' : 'record') || (on ? '&#9632; Stop' : '&#9679; Rec');
    recBtn.title = on ? 'Stop recording and save the video (V)' : REC_TITLE;
    recBtn.setAttribute('aria-label', on ? 'Stop' : 'Rec');
    recBtn.classList.toggle('pp-recording', on);
    badge.hidden = !on;
  };

  async function startRecording() {
    const scene = api.scene;
    if (!scene || rec || api.view !== 'graph') return;
    const mime = typeof MediaRecorder !== 'undefined' && MIMES.find(t => MediaRecorder.isTypeSupported(t));
    if (!mime || !HTMLCanvasElement.prototype.captureStream) { api.toast('Recording is not supported in this browser'); return; }
    const token = rec = { pending: true };
    try {
      const scale = pixelRatio(scene), area0 = captureArea(scene);
      let w = area0.box.width * scale, h = area0.box.height * scale;
      const s = Math.min(1, 1920 / w, 1080 / h);
      w = Math.max(2, 2 * Math.round((w * s) / 2));
      h = Math.max(2, 2 * Math.round((h * s) / 2));
      await prepareLabels(area0, scale).catch(() => {});
      if (rec !== token) return; // stopped (or restarted) while preparing
      if (api.view !== 'graph') { rec = null; return; }
      const cv = document.createElement('canvas');
      cv.width = w;
      cv.height = h;
      const g = cv.getContext('2d', { alpha: false });
      scene.render();
      compose(g, w, h, { ...area0, panes: area0.panes.slice(0, 1) }, scale); // first frame: the codomain pane can't be read here
      const stream = cv.captureStream(60);
      const recorder = new MediaRecorder(stream, {
        mimeType: mime, videoBitsPerSecond: Math.round(Math.min(16e6, Math.max(4e6, w * h * 60 * 0.08))),
      });
      const chunks = [];
      recorder.ondataavailable = e => {
        if (!e.data?.size) return;
        chunks.push(e.data);
        stats.bytes += e.data.size;
      };
      recorder.onerror = e => console.error('[present] recorder:', e.error ?? e);
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        const type = (recorder.mimeType || mime).split(';')[0];
        if (chunks.length) {
          download(new Blob(chunks, { type }), `${stamp()}.${type.includes('mp4') ? 'mp4' : 'webm'}`);
          api.toast('Recording saved');
        } else {
          api.toast('Recording was empty');
        }
      };
      let due = 0, live = true, raf = 0, side = area0.panes[1]?.canvas ?? null;
      const stats = { w, h, mime, frames: 0, composeMs: 0, bytes: 0 };
      // Our own frame callback, queued from a timer so it runs after the scenes' callbacks (which
      // re-queue themselves first each frame): every canvas it reads has just been rendered. A
      // codomain pane that appears later queues behind us, so then we re-queue once.
      const arm = () => setTimeout(() => { if (live) raf = requestAnimationFrame(loop); });
      const loop = () => {
        raf = requestAnimationFrame(loop);
        const area = captureArea(scene), cvSide = area.panes[1]?.canvas ?? null;
        if (cvSide !== side) {
          side = cvSide;
          if (cvSide) { cancelAnimationFrame(raf); arm(); return; }
        }
        const now = performance.now();
        if (now < due - 2) return;
        due = Math.max(due + FRAME_MS, now + FRAME_MS / 2);
        rasterizeSoon(compose(g, w, h, area, scale), scale);
        stats.frames++;
        stats.composeMs += performance.now() - now;
      };
      arm();
      const t0 = performance.now();
      const timer = setInterval(() => {
        const sec = Math.floor((performance.now() - t0) / 1000);
        badgeTime.textContent = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
      }, 250);
      badgeTime.textContent = '0:00';
      recorder.start(1000);
      rec = {
        recorder, stats,
        freeze() { live = false; cancelAnimationFrame(raf); clearInterval(timer); },
        hold() { g.drawImage(cv, 0, 0); }, // repeat the last frame
      };
      paintRec();
    } catch (err) {
      console.error('[present] recording:', err);
      api.toast('Recording failed');
      rec = null;
      paintRec();
    }
  }

  function stopRecording() {
    if (!rec) return;
    const r = rec;
    rec = null;
    paintRec();
    if (!r.recorder) return; // still starting: the pending start sees rec === null and bails
    r.freeze();
    // Hold the last frame for a moment before stopping. The encoder can take a second or two to
    // start (software H.264 under load); stopping before it has emitted anything gives an empty
    // file, so for very short clips keep holding until the first data arrives.
    const t = performance.now();
    const pump = setInterval(() => {
      r.hold();
      const waited = performance.now() - t;
      if ((waited >= STOP_GRACE_MS && r.stats.bytes > 0) || waited > STOP_MAX_WAIT_MS) {
        clearInterval(pump);
        if (r.recorder.state !== 'inactive') r.recorder.stop();
      }
    }, 2 * FRAME_MS);
  }

  const toggleRecording = () => (rec ? stopRecording() : startRecording());
  api.onViewChange(v => { if (v !== 'graph') stopRecording(); });

  window.addEventListener('keydown', e => {
    if (api.view !== 'graph' || api.params.has('audience') || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    if (e.target?.closest?.('input:not([type="range"]), textarea, select, [contenteditable="true"]')) return;
    const key = e.key.toLowerCase();
    if (key === 'p') { e.preventDefault(); snapshot(); }
    else if (key === 'v') { e.preventDefault(); toggleRecording(); }
  });

  api.present = { snapshot, startRecording, stopRecording, get recording() { return rec?.recorder ? rec.stats : null; } };
}
