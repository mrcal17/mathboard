// Net tab: the network canvas in #nn-stage. See docs/NN_CONTRACT.md.
//
// One SVG fills the stage. Layer lanes, edges, neurons and layer headers sit in a world <g> under a
// pan/zoom transform (screen = world * k + (x, y)). Structural changes diff the DOM by id;
// everything else (values at training speed, positions while dragging, hover, selection and the
// matrix panel's step animation) patches attributes in place, batched into one animation frame.
//
// Editing (all undoable): drag a neuron or a layer header to move it; drag empty space to pan,
// wheel to zoom; double-click empty space to add a neuron to the nearest layer, or a new layer
// when the click is clearly between two columns or beyond the ends; drag from a neuron's handle
// (the dot on its right edge) to another neuron to connect them. W: numbers on the edges.
//
// Sequences (docs/NN_ATTENTION.md): a token layer's neurons sit in one soft box per token, grouped
// under their group's letter (Q / K / V). The token labels (t₁…tₙ, or net.meta.tokenNames) show once
// per row, left of the first token layer; any other box shows its label while it is hovered or
// followed. An attention layer draws its data-dependent edges V_j -> Z_i (width and opacity from
// A_ij) and an n×n heatmap of A left of its header's name; hovering a Z token (or neuron, or a
// heatmap cell) shows its attention row, hovering a V token or neuron its column. Token boxes,
// heatmap cells and attention edges hover through the store as { kind: 'token', layer, t, g?, h? },
// so the matrix panel, the cards and the audience see it too. A hovered or selected tied edge
// lights its whole tie group; fixed edges are dotted and neutral (not parameters). Edits keep token
// layers whole: a double-click adds a feature to every token (or says why it can't), and a new edge
// between tokenwise-tied layers becomes a new shared entry, added for every token.
//
// Look (docs/DESIGN.md section 7 A): no lane or header boxes by default. A lane tints while its
// header is hovered and takes an HI outline when selected; edges get a crisp HI outline and
// neurons an HI ring on hover and selection. Headers that would collide drop a sub line first.
//
// Lens (docs/NN_LENS.md, rules in focus.js): what state.lens leaves out of its story is dimmed
// (opacity --le, about 0.1, and no numbers); what its show toggles and thresholds remove is not
// drawn. Hover, selection and the step-through still light what they point at. Following a token
// shows its row of A on the attention edges, as hovering it would.

import { colorFor } from './store.js';
import { emphasis, tokenNames } from './focus.js';

const NS = 'http://www.w3.org/2000/svg';
const XHTML = 'http://www.w3.org/1999/xhtml';
const R = 26;                          // neuron radius, world px (CSS px at zoom 1)
// The HI ring's radius: a 3 px gap outside the rim, or outside the δ ring (r R + 4, up to 5.5 wide).
const HL_R = R + 4.5, HL_R_D = R + 10;
const MIN_K = 0.15, MAX_K = 4, FIT_MAX_K = 1.6;
const HEAD_UP = R + 50;                // header centre sits this far above the topmost neuron's centre
const LANE_TOP = 27;                   // lanes start this far below the header centre (clear of its pill)
const LANE_DOWN = R + 30;              // lanes end this far below the lowest neuron's centre
const COL = 160;                       // column spacing for empty layers (as model.js)
const SNAP = 10;                       // a dragged neuron snaps onto its column within this (Alt: off)
const TEXT_K = 0.85, TEXT_MAX = 1.5;   // zoomed out past TEXT_K, text grows (up to TEXT_MAX) to stay legible
const BEYOND = 2.5 * R;                // a double-click this far past the end columns adds a layer
const GAP = 170;                       // a layer added by double-click keeps this far from its neighbours (as nn.js)
const ROW = 80;                        // neuron spacing in a column (as model.js)
// Token boxes: padding around the token's neurons (the bottom holds the value under the circle),
// and the group band's padding around its boxes (Q / K / V sit to its left). A row's token label
// sits TOK_LAB left of its box.
const TOK = { l: 8, r: 8, t: 6, b: 24, gap: 5 };
const GRP = { l: 6, r: 6, t: 6, b: 6 };
const TOK_LAB = 7;
// Headers (in header px, before the text boost): the name's and the sub line's baselines, the
// pill's padding round the text, the heatmap's gap to the name, and the air between two pills.
const HEAD = { name: -3, sub: 13, top: -20, bot: 20, botName: 7, padX: 10, hmGap: 8, air: 4 };
const HM_H = 26;                       // the heatmap of A is about this tall: two lines of header text
const VARS = ['--nnv-bg', '--nnv-text', '--nnv-text-2', '--nnv-muted', '--nnv-line', '--nnv-line-3', '--nnv-hover',
  '--nnv-hi', '--nnv-hi-text', '--nnv-att', '--nnv-bad', '--nnv-node', '--nnv-rim', '--nnv-dot', '--nnv-tok'];
const MARKS = ['sel', 'hov', 'rel', 'lit', 'bias', 'show', 'drop', 'tie', 'foc'];
// The dot grid goes once text grows past this (zoomed out).
const FAR_TS = 1.3;
// Lens: a dimmed thing is drawn at DIM + (1 - DIM)·emphasis; below NUM_MIN it also loses its numbers.
const DIM = 0.1, NUM_MIN = 0.25;
// Token names: at most NAME_MAX characters beside the box (the full name is the box's tooltip).
const NAME_MAX = 8;
// Fit: the panels floating over the stage that the net fits beside (px of air around each), the
// smallest free area worth fitting into, and how much scale a larger free area may cost.
const FLOATS = '.nn-train, .nn-attnviz, .nn-s3d', FLOAT_GAP = 6, FREE_MIN = 160, FIT_SLACK = 0.9;

const r1 = v => Math.round(v * 10) / 10;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const isNum = v => typeof v === 'number' && Number.isFinite(v);

// A layer's token shape: tokens rows of d features, per group (node k is group floor(k / (T d)),
// token floor(k / d) % T, feature k % d). null for a plain vector; d = 0 when the size doesn't split.
function shapeOf(layer, size) {
  const att = layer?.kind === 'attention';
  const T = Number.isInteger(layer?.tokens) && layer.tokens >= 1 ? layer.tokens : 1;
  const groups = !att && Array.isArray(layer?.groups) && layer.groups.length ? layer.groups.map(String) : null;
  if (T === 1 && !groups && !att) return null;
  const G = groups ? groups.length : 1, d = size / (T * G);
  const heads = att && Number.isInteger(layer.heads) && layer.heads > 0 ? layer.heads : 1;
  return { T, G, d: Number.isInteger(d) && d >= 1 ? d : 0, groups, att, heads, causal: att && !!layer.causal };
}
// 'W_{Q}:1,2' -> { name: 'W_{Q}', i: 1, j: 2 }.
function tieParse(tie) {
  const m = /^(.*):\s*(\d+)\s*,\s*(\d+)\s*$/.exec(String(tie ?? ''));
  return m ? { name: m[1], i: +m[2], j: +m[3] } : null;
}
// KaTeX tie name -> plain base + subscript for SVG text: 'W_{Q}' -> ['W', 'Q'], '\\alpha' -> ['α', ''].
const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', theta: 'θ', lambda: 'λ', mu: 'μ', sigma: 'σ', phi: 'φ', omega: 'ω' };
function tiePlain(name) {
  const s = String(name).replace(/\\(?:mathrm|mathbf|mathit|text|operatorname|boldsymbol)\s*/g, '')
    .replace(/\\([a-zA-Z]+)/g, (_, w) => GREEK[w] || w);
  const m = /^([^_]*)(?:_(\{[^}]*\}|.))?(.*)$/.exec(s);
  const clean = x => (x || '').replace(/[{}\s]/g, '').replace(/\^/g, '');
  return [clean(m[1]) + clean(m[3]), clean(m[2])];
}
function maxAbs(...xs) {   // as matrix.js: nested arrays, non-finite entries skipped
  let m = 0;
  const walk = x => {
    if (Array.isArray(x)) x.forEach(walk);
    else if (isNum(x)) m = Math.max(m, Math.abs(x));
  };
  xs.forEach(walk);
  return m;
}

function mk(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const a in attrs) e.setAttribute(a, attrs[a]);
  if (parent) parent.appendChild(e);
  return e;
}
// Setters that skip the DOM when nothing changed: values repaint every frame while training.
function put(e, name, v) {
  const c = e.__a || (e.__a = {});
  if (c[name] === v) return;
  c[name] = v;
  if (v == null) e.removeAttribute(name);
  else e.setAttribute(name, v);
}
function txt(e, s) {
  if (e.__t !== s) { e.__t = s; e.textContent = s; }
}

// KaTeX font size for a label, so longer labels still fit inside the circle.
function texSize(tex) {
  const n = String(tex || '')
    .replace(/\\(?:hat|bar|tilde|vec|dot|mathbf|mathrm|mathit|boldsymbol|operatorname|text)\b/g, '')
    .replace(/\\[a-zA-Z]+/g, 'x').replace(/[{}\s^_]/g, '').length;
  return n <= 3 ? 17 : n <= 5 ? 15.5 : n <= 7 ? 13.5 : Math.max(8, 95 / n);
}

// ---------------------------------------------------------------- geometry (world px)
function toward(p, q, d) {
  const dx = q.x - p.x, dy = q.y - p.y, l = Math.hypot(dx, dy) || 1;
  return { x: p.x + (dx / l) * d, y: p.y + (dy / l) * d };
}
function straight(a, b, rb = R) {
  const s = toward(a, b, R), t = toward(b, a, rb);
  return {
    d: `M${r1(s.x)},${r1(s.y)}L${r1(t.x)},${r1(t.y)}`,
    at: u => ({ x: s.x + (t.x - s.x) * u, y: s.y + (t.y - s.y) * u }),
  };
}
function curve(a, c, b) {
  const s = toward(a, c, R), t = toward(b, c, R);
  return {
    d: `M${r1(s.x)},${r1(s.y)}Q${r1(c.x)},${r1(c.y)} ${r1(t.x)},${r1(t.y)}`,
    at: u => {
      const v = 1 - u;
      return { x: v * v * s.x + 2 * v * u * c.x + u * u * t.x, y: v * v * s.y + 2 * v * u * c.y + u * u * t.y };
    },
  };
}

// ---------------------------------------------------------------- png() helpers
// KaTeX CSS with its woff2 fonts inlined, so the SVG-as-image can render the labels
// (same approach as graph/features/present.js).
let katexCssP = null, viewCssP = null;
const readAsDataUrl = blob => new Promise((res, rej) => {
  const fr = new FileReader();
  fr.onload = () => res(String(fr.result));
  fr.onerror = rej;
  fr.readAsDataURL(blob);
});
function katexCss() {
  katexCssP ||= (async () => {
    const link = [...document.querySelectorAll('link[rel="stylesheet"]')].find(l => /katex/i.test(l.href));
    if (!link) return '';
    let css = await (await fetch(link.href)).text();
    const faces = css.match(/@font-face\s*{[^}]*}/g) || [];
    const inlined = await Promise.all(faces.map(async face => {
      const m = /url\(["']?([^"')]+\.woff2)["']?\)/.exec(face);
      if (!m) return '';
      const url = (await readAsDataUrl(await (await fetch(new URL(m[1], link.href))).blob()))
        .replace(/^data:[^;,]*/, 'data:font/woff2');
      return face.replace(/src:[^;}]*/, `src:url(${url}) format("woff2")`);
    }));
    faces.forEach((face, i) => { css = css.replace(face, inlined[i]); });
    return css;
  })();
  katexCssP.catch(() => { katexCssP = null; });
  return katexCssP;
}
function viewCss() {
  viewCssP ||= fetch(new URL('./view.css', import.meta.url)).then(r => r.text());
  viewCssP.catch(() => { viewCssP = null; });
  return viewCssP;
}

// ================================================================ install
export function install(ctx) {
  const { store } = ctx;
  const model = ctx.model || store.model;
  const stage = ctx.el?.stage || document.getElementById('nn-stage');
  const audience = !!ctx.audience;
  const theme = () => ctx.theme?.() || (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  const num = v => (isNum(v) ? model.fmt(v, 2).replace(/^-/, '−') : '—');
  // gradients: below 0.01 two significant figures (0.0034, 3.4e−4), as in the matrix panel
  const numg = v => (isNum(v) ? (model.fmtg || model.fmt)(v, 2).replace(/-/g, '−') : '—');

  // ---------------------------------------------------------------- DOM
  const svg = mk('svg', { class: `nnv${audience ? ' audience' : ''}`, role: 'img', 'aria-label': 'Neural network' });
  const defs = mk('defs', null, svg);
  const pat = mk('pattern', { id: 'nnv-dots', width: 28, height: 28, patternUnits: 'userSpaceOnUse' }, defs);
  mk('circle', { class: 'nnv-dot', cx: 14, cy: 14, r: 1.2 }, pat);
  mk('circle', { r: R }, mk('clipPath', { id: 'nnv-clip', clipPathUnits: 'userSpaceOnUse' }, defs));
  mk('rect', { class: 'nnv-grid', width: '100%', height: '100%', fill: 'url(#nnv-dots)' }, svg);
  const world = mk('g', { class: 'nnv-world' }, svg);
  const content = mk('g', { class: 'nnv-content' }, world);   // untransformed: its getBBox is in world px
  const gBands = mk('g', { class: 'nnv-bands' }, content);
  const gGroups = mk('g', { class: 'nnv-grps' }, content);
  const gToks = mk('g', { class: 'nnv-toks' }, content);
  const gEdges = mk('g', { class: 'nnv-edges' }, content);
  const gAtt = mk('g', { class: 'nnv-atts' }, content);
  const pairEl = mk('path', { class: 'nnv-pair', display: 'none' }, content);
  const gPulses = mk('g', { class: 'nnv-pulses' }, content);
  const gLabels = mk('g', { class: 'nnv-wls' }, content);
  const gTokLabs = mk('g', { class: 'nnv-tls' }, content);   // t₁ and Q / K / V: over the edges that cross them
  const gAttLabs = mk('g', { class: 'nnv-als' }, content);
  const gNodes = mk('g', { class: 'nnv-nodes' }, content);
  const gHeads = mk('g', { class: 'nnv-heads' }, content);
  const ghost = mk('path', { class: 'nnv-ghost', display: 'none' }, content);
  // The empty-net hint: HTML in the SVG, for its key chips (screen px: placeHint centres it).
  const HINT_W = 520, HINT_H = 32;
  const hint = mk('foreignObject', { class: 'nnv-hint', x: 0, y: 0, width: HINT_W, height: HINT_H, display: 'none' }, svg);
  const hintIn = document.createElementNS(XHTML, 'div');
  hintIn.className = 'nnv-hint-in';
  hintIn.innerHTML = '<kbd class="lg">Double-click</kbd><span>to add a neuron,</span><kbd class="lg">N</kbd><span>for a new net</span>';
  hint.appendChild(hintIn);
  stage.prepend(svg);

  // ---------------------------------------------------------------- state
  const nodes = new Map(), edges = new Map(), layers = new Map(), images = new Map();
  const toks = new Map(), atts = new Map(), attLines = new Map();   // per layer id; attention edges by id
  const V = { k: 1, x: 0, y: 0 };
  let I = null, stale = true;
  let dirty = { build: true }, raf = 0;
  let shown = document.body.dataset.view === 'nn';
  let needFit = 0, everFit = false, userMoved = false, fitAnim = 0;   // needFit: false | ms of the pending fit
  let showW = false, marked = [], pulseKey = '', pairIds = null, dropId = null, textScale = 1;
  let focus = [];     // the shown attention rows (or column)
  let E = null;       // the lens's emphasis (focus.js), or null without a lens

  // Per-flush index of the live net. store.net's contents are replaced on undo, so never cache
  // node objects across events.
  function ix() {
    if (I && !stale) return I;
    stale = false;
    const net = store.net, li = new Map(), byLayer = [], nodeById = new Map(), edgeById = new Map(), rank = new Map();
    net.layers.forEach((l, i) => { li.set(l.id, i); byLayer.push([]); });
    let box = null;
    for (const n of net.nodes) {
      nodeById.set(n.id, n);
      const i = li.get(n.layer);
      if (i === undefined) continue;
      rank.set(n.id, byLayer[i].length);
      byLayer[i].push(n);
      if (!box) box = { minX: n.x, maxX: n.x, minY: n.y, maxY: n.y };
      else {
        box.minX = Math.min(box.minX, n.x); box.maxX = Math.max(box.maxX, n.x);
        box.minY = Math.min(box.minY, n.y); box.maxY = Math.max(box.maxY, n.y);
      }
    }
    const ties = new Map(), biasTies = new Map();   // tie id -> edge ids; node.tie (shared bias) -> node ids
    for (const e of net.edges) {
      edgeById.set(e.id, e);
      if (typeof e.tie === 'string' && e.tie) {
        if (!ties.has(e.tie)) ties.set(e.tie, []);
        ties.get(e.tie).push(e.id);
      }
    }
    for (const n of net.nodes) {
      if (typeof n.tie !== 'string' || !n.tie) continue;
      if (!biasTies.has(n.tie)) biasTies.set(n.tie, []);
      biasTies.get(n.tie).push(n.id);
    }
    const tok = net.layers.map((l, i) => shapeOf(l, byLayer[i].length));
    // An attention layer draws only when it has what it reads: a Q/K/V layer before it with the same tokens.
    const att = tok.map((s, l) => {
      const p = tok[l - 1];
      if (!s?.att || !s.d || !p?.groups || !p.d || p.T !== s.T || p.d !== s.d) return null;
      const vG = p.groups.indexOf('V'), dh = s.d / s.heads;
      if (vG < 0 || !Number.isInteger(dh)) return null;
      return { l, T: s.T, d: s.d, heads: s.heads, dh, vG, qG: p.groups.indexOf('Q'), kG: p.groups.indexOf('K'), causal: s.causal };
    });
    const real = byLayer.map(ns => {
      if (!ns.length) return null;
      const c = { x: 0, minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, n: ns.length };
      for (const n of ns) {
        c.x += n.x / ns.length;
        c.minX = Math.min(c.minX, n.x); c.maxX = Math.max(c.maxX, n.x);
        c.minY = Math.min(c.minY, n.y); c.maxY = Math.max(c.maxY, n.y);
      }
      return c;
    });
    const cy = box ? (box.minY + box.maxY) / 2 : 280, L = real.length;
    const cols = real.map((c, i) => {
      if (c) return c;
      let p = i - 1, q = i + 1;
      while (p >= 0 && !real[p]) p--;
      while (q < L && !real[q]) q++;
      const P = real[p], Q = real[q];
      const x = P && Q ? P.x + ((Q.x - P.x) * (i - p)) / (q - p)
        : P ? P.x + COL * (i - p) : Q ? Q.x - COL * (q - i) : 450 + COL * (i - (L - 1) / 2);
      return { x, minX: x, maxX: x, minY: cy, maxY: cy, n: 0 };
    });
    I = { li, byLayer, nodeById, edgeById, rank, cols, box, cy, full: real.every(Boolean), ties, biasTies, tok, att };
    return I;
  }
  // Where node id sits in its token layer: { l, g, t, f } (null on a plain layer or a bad shape).
  function tokPos(id) {
    const I = ix(), n = I.nodeById.get(id), l = n && I.li.get(n.layer), s = l !== undefined ? I.tok[l] : null;
    if (!s?.d) return null;
    const k = I.rank.get(id);
    return { l, g: Math.floor(k / (s.T * s.d)), t: Math.floor(k / s.d) % s.T, f: k % s.d };
  }
  const tokNode = (l, g, t, f) => { const I = ix(), s = I.tok[l]; return s?.d ? I.byLayer[l][(g * s.T + t) * s.d + f] : undefined; };

  // ---------------------------------------------------------------- build (structural)
  function diff(map, items, make, drop) {
    const seen = new Set();
    for (const it of items) {
      seen.add(it.id);
      if (!map.has(it.id)) map.set(it.id, make(it));
    }
    for (const [id, r] of map) if (!seen.has(id)) { drop(r); map.delete(id); }
  }

  function makeNode(n) {
    const g = mk('g', { class: 'nnv-node', 'data-kind': 'node', 'data-id': n.id }, gNodes);
    const hl = mk('circle', { class: 'nnv-hl', r: HL_R }, g);
    const gring = mk('circle', { class: 'nnv-gring', r: R + 4 }, g);
    mk('circle', { class: 'nnv-base', r: R }, g);
    const fill = mk('circle', { class: 'nnv-fill', r: R }, g);
    mk('circle', { class: 'nnv-rim', r: R }, g);
    const fo = mk('foreignObject', { class: 'nnv-lab', x: -R - 14, y: -R, width: 2 * R + 28, height: 2 * R }, g);
    const div = document.createElementNS(XHTML, 'div');
    div.className = 'nnv-tex';
    fo.appendChild(div);
    const val = mk('text', { class: 'nnv-val', y: R + 8 }, g);
    const grad = mk('text', { class: 'nnv-grad', x: R + 9, y: -R * 0.5 }, g);
    const tgt = mk('text', { class: 'nnv-tgt', x: R + 9, y: R * 0.5 }, g);
    const bias = mk('text', { class: 'nnv-bias', x: -R - 9, y: -R * 0.5 }, g);
    const bn = mk('tspan', { class: 'nnv-wl-n' }, bias), bs = mk('tspan', { class: 'nnv-wl-s' }, bias), bv = mk('tspan', null, bias);
    if (!audience) mk('circle', { class: 'nnv-handle', 'data-kind': 'handle', 'data-id': n.id, cx: R + 1, r: 5 }, g);
    const r = { id: n.id, g, hl, gring, fill, fo, div, val, grad, tgt, bias, bn, bs, bv, img: null, label: undefined, x: NaN, y: NaN };
    if (images.has(n.id)) applyImage(r, images.get(n.id));
    return r;
  }
  function makeEdge(e) {
    const g = mk('g', { class: 'nnv-edge', 'data-kind': 'edge', 'data-id': e.id }, gEdges);
    const hit = mk('path', { class: 'nnv-hit' }, g);
    const ol = mk('path', { class: 'nnv-ol' }, g);        // the HI outline, under the line
    const line = mk('path', { class: 'nnv-line' }, g);
    const lab = mk('text', { class: 'nnv-wl' }, gLabels);
    const t1 = mk('tspan', { x: 0 }, lab);
    const tn = mk('tspan', { class: 'nnv-wl-n' }, t1);   // a tied edge's matrix, e.g. W
    const ts = mk('tspan', { class: 'nnv-wl-s' }, t1);   // its subscript, e.g. Q
    const tv = mk('tspan', null, t1);                     // (i,j) = value
    const t2 = mk('tspan', { x: 0, dy: '1.2em', class: 'nnv-wl-g' }, lab);
    return { id: e.id, g, hit, ol, line, lab, t1, tn, ts, tv, t2, G: null, kind: '' };
  }
  function makeLayer(l) {
    const band = mk('rect', { class: 'nnv-band', rx: 18 }, gBands);
    const g = mk('g', { class: 'nnv-head', 'data-kind': 'layer', 'data-id': l.id }, gHeads);
    const tip = mk('title', null, g);                   // the whole header, while its sub line gives way
    const gi = mk('g', { class: 'nnv-head-in' }, g);   // carries the text boost (see placeHead)
    if (textScale !== 1) put(gi, 'transform', `scale(${textScale})`);
    const bg = mk('rect', { class: 'nnv-head-bg', rx: 8, y: HEAD.top, height: HEAD.bot - HEAD.top }, gi);   // the pill: hit area, hover, selection
    const name = mk('text', { class: 'nnv-head-name', y: HEAD.name }, gi);
    const sub = mk('text', { class: 'nnv-head-sub', y: HEAD.sub }, gi);
    return { id: l.id, band, g, tip, gi, bg, name, sub, nameS: null, subS: null, nameW: 0, subW: 0, measured: false };
  }
  const drop = r => { r.g.remove(); r.lab?.remove(); r.band?.remove(); r.hm = null; };

  function rebuild() {
    const net = store.net, before = new Set(nodes.keys());
    diff(nodes, net.nodes, makeNode, drop);
    diff(edges, net.edges, makeEdge, drop);
    diff(layers, net.layers, makeLayer, drop);
    for (const id of images.keys()) if (!nodes.has(id)) images.delete(id);
    // Keep paint order = net order, so later layers draw over earlier ones.
    for (const n of net.nodes) gNodes.appendChild(nodes.get(n.id).g);
    put(hint, 'display', net.nodes.length || audience ? 'none' : null);
    // A wholly different net (preset, import, undo past a load): frame it.
    if (nodes.size && ![...nodes.keys()].some(id => before.has(id))) needFit = everFit ? 300 : 0;
    pulseKey = '';
  }

  // ---------------------------------------------------------------- tokens and attention (structure)
  // A layer's token shape lives in its fields, so changing it is not a structural event (no id
  // changes): compare shape keys on every net event and rebuild only what changed.
  // Token names (net.meta.tokenNames) are part of the key: renaming rebuilds the boxes' labels.
  const tokKey = (s, names) => (s?.d ? `${s.T}|${s.d}|${s.groups ? s.groups.join('\u0001') : ''}|${s.att}|${names.slice(0, s.T).join('\u0001')}` : '');
  const attKey = a => (a ? `${a.T}|${a.d}|${a.heads}|${a.vG}|${a.qG}|${a.kG}` : '');
  function syncTokens() {
    const net = store.net, I = ix(), live = new Set(), names = tokenNames(net);
    // The row labels (t₁, or the token's word) sit beside the first token layer only.
    const rowL = I.tok.findIndex(s => s?.d && s.T > 1);
    let changed = false;
    net.layers.forEach((l, i) => {
      live.add(l.id);
      const base = tokKey(I.tok[i], names), key = base && i === rowL ? `${base}|row` : base, r = toks.get(l.id);
      if ((r?.key ?? '') !== key) {
        if (r) dropTok(r);
        toks.delete(l.id);
        if (key) toks.set(l.id, makeTok(l.id, I.tok[i], key, names, i === rowL));
        changed = true;
      }
      const akey = attKey(I.att[i]), q = atts.get(l.id), head = layers.get(l.id);
      if ((q?.key ?? '') !== akey || (q && q.head !== head)) {
        if (q) dropAtt(q);
        atts.delete(l.id);
        if (akey) atts.set(l.id, makeAtt(l.id, I.att[i], akey, head));
        changed = true;
      }
    });
    for (const [id, r] of toks) if (!live.has(id)) { dropTok(r); toks.delete(id); changed = true; }
    for (const [id, q] of atts) if (!live.has(id)) { dropAtt(q); atts.delete(id); changed = true; }
    return changed;
  }
  // A token's name, shortened for its label (the full name is the box's tooltip).
  const shortName = n => (n.length > NAME_MAX ? `${n.slice(0, NAME_MAX - 1)}…` : n);
  function makeTok(id, s, key, names = [], row = false) {
    const r = { key, row, bands: [], boxes: [], gut: TOK.l };
    (s.groups || []).forEach((name, g) => {
      const rect = mk('rect', { class: 'nnv-grp', rx: 16 }, gGroups);
      const lab = mk('text', { class: 'nnv-grp-lab' }, gTokLabs);
      const [base, lo] = tiePlain(name);
      lab.textContent = base;
      if (lo) mk('tspan', { class: 'nnv-sub', dy: '0.3em' }, lab).textContent = lo;
      r.bands.push({ rect, lab, g });
    });
    for (let g = 0; g < s.G; g++) for (let t = 0; t < s.T; t++) {
      const el = mk('g', { class: 'nnv-tok', 'data-kind': 'token', 'data-id': id, 'data-g': g, 'data-t': t }, gToks);
      const rect = mk('rect', { class: 'nnv-tok-box', rx: 12 }, el);
      const lab = mk('text', { class: row ? 'nnv-tok-lab row' : 'nnv-tok-lab' }, gTokLabs);
      if (names[t]) {
        lab.classList.add('name');
        lab.textContent = shortName(names[t]);
        mk('title', null, el).textContent = names[t];
      } else {
        lab.textContent = 't';
        mk('tspan', { class: 'nnv-sub', dy: '0.3em' }, lab).textContent = String(t + 1);
      }
      if (s.T === 1) { el.setAttribute('display', 'none'); lab.setAttribute('display', 'none'); }   // one token: the band is enough
      r.boxes.push({ el, rect, lab, g, t, name: !!names[t] });
    }
    return r;
  }
  function dropTok(r) {
    for (const b of r.bands) { b.rect.remove(); b.lab.remove(); }
    for (const b of r.boxes) { b.el.remove(); b.lab.remove(); }
  }
  // Attention: an edge V_j,f -> Z_i,f per (i, j, f), an A_ij label per (head, i, j) shown with its row,
  // and the heatmap (one T×T block per head) in the layer's header.
  function makeAtt(id, a, key, head) {
    const q = { key, head, lines: [], labs: [], hm: null };
    for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) for (let f = 0; f < a.d; f++) {
      const h = Math.floor(f / a.dh), lid = `att:${id}:${i}:${j}:${f}`;
      const g = mk('g', { class: 'nnv-att', 'data-kind': 'attedge', 'data-id': id, 'data-i': i, 'data-j': j, 'data-h': h }, gAtt);
      const rec = { id: lid, g, hit: mk('path', { class: 'nnv-att-hit' }, g), line: mk('path', { class: 'nnv-att-line' }, g), i, j, f, h, G: null };
      q.lines.push(rec);
      attLines.set(lid, rec);
    }
    for (let h = 0; h < a.heads; h++) for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) {
      q.labs.push({ el: mk('text', { class: 'nnv-al' }, gAttLabs), h, i, j });
    }
    if (head) {
      // About as tall as the header's two lines of text; cells 1 px apart, no frame.
      const cell = clamp(Math.floor(HM_H / a.T), 4, 13), n = a.T * cell - 1, gap = 5;
      const g = mk('g', { class: 'nnv-hm' }, head.gi);   // left of the header's name, placed by placeHead
      const hm = { g, cell, n, W: a.heads * n + (a.heads - 1) * gap, H: n, cells: [], hl: [] };
      for (let h = 0; h < a.heads; h++) {
        const x0 = h * (n + gap);
        for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) {
          const el = mk('rect', { class: 'nnv-hm-c', 'data-kind': 'attcell', 'data-id': id, 'data-i': i, 'data-j': j, 'data-h': h,
            x: x0 + j * cell, y: i * cell, width: cell - 1, height: cell - 1 }, g);
          hm.cells.push({ el, h, i, j });
        }
        hm.hl.push({ el: mk('rect', { class: 'nnv-hm-hl', display: 'none', rx: 1.5 }, g), x0 });
      }
      q.hm = head.hm = hm;   // placed by placeHead
    }
    return q;
  }
  function dropAtt(q) {
    for (const r of q.lines) { r.g.remove(); attLines.delete(r.id); }
    for (const t of q.labs) t.el.remove();
    q.hm?.g.remove();
    if (q.head && q.head.hm === q.hm) q.head.hm = null;
  }

  // ---------------------------------------------------------------- meta (labels, headers)
  function renderLabel(r, tex) {
    r.label = tex;
    r.div.style.fontSize = `${texSize(tex)}px`;
    const k = window.katex;
    if (!k || !tex) { r.div.textContent = tex || ''; return; }
    try { k.render(String(tex), r.div, { throwOnError: false, output: 'html' }); }
    catch { r.div.textContent = tex; }
  }
  // A header is its name over a sub line (act · size), with an attention layer's heatmap of A left
  // of both, and a pill round them that shows only on hover and selection. Widths are header px,
  // before the text boost; the sub line is measured while hidden too (visibility, not display).
  function measureHead(r) {
    let a = 0, b = 0;
    try { a = r.name.getComputedTextLength(); b = r.sub.getComputedTextLength(); } catch { /* not rendered */ }
    if (!a) return;
    r.measured = true;
    r.nameW = a;
    r.subW = b;
  }
  const headW = (r, sub) => (r.hm ? r.hm.W + HEAD.hmGap : 0) + Math.max(24, r.nameW, sub ? r.subW : 0) + 2 * HEAD.padX;
  function headSub(l, i, count) {
    const s = ix().tok[i], act = model.ACTS?.[l.act]?.label || l.act || 'Identity';
    if (!s) return i === 0 ? `${count} input${count === 1 ? '' : 's'}` : `${act} · ${count}`;
    const tokens = `${s.T} token${s.T === 1 ? '' : 's'}`;
    if (!s.d) return `${i === 0 ? '' : `${act} · `}${count}: not ${s.G > 1 ? `${s.G} × ` : ''}${tokens} ✕`;
    const shape = `${s.T} × ${s.d}${s.groups ? ' each' : ''}`;
    if (s.att) return `${s.causal ? 'causal ' : ''}attention · ${shape}${s.heads > 1 ? ` · ${s.heads} heads` : ''}`;
    return i === 0 ? `${tokens} × ${s.d}${s.groups ? ' each' : ''}` : `${act} · ${shape}`;
  }
  // Returns true when a header's text changed (it needs measuring and fitting between its neighbours).
  function syncMeta() {
    const net = store.net, I = ix(), L = net.layers.length;
    let heads = false;
    for (const n of net.nodes) {
      const r = nodes.get(n.id);
      if (r && r.label !== n.label) renderLabel(r, n.label);
    }
    for (const e of net.edges) {   // fixed (dashed) and tied edges: plain fields, so no structural event
      const r = edges.get(e.id), kind = e.fixed ? 'fixed' : typeof e.tie === 'string' && e.tie ? 'tied' : '';
      if (!r || r.kind === kind) continue;
      r.kind = kind;
      r.g.classList.toggle('fixed', kind === 'fixed');
      r.g.classList.toggle('tied', kind === 'tied');
    }
    net.layers.forEach((l, i) => {
      const r = layers.get(l.id);
      if (!r) return;
      const count = I.byLayer[i].length;
      const name = l.name || (i === 0 ? 'Input' : i === L - 1 ? 'Output' : 'Hidden');
      const sub = headSub(l, i, count);
      if (r.nameS === name && r.subS === sub) return;
      r.nameS = name; r.subS = sub;
      txt(r.name, name); txt(r.sub, sub);
      r.g.setAttribute('aria-label', `${name}: ${sub}`);
      r.measured = false;
      r.nameW = name.length * 8.2;   // until measured
      r.subW = sub.length * 6.6;
      r.band.classList.toggle('empty', !count);
      heads = true;
    });
    return heads;
  }

  // ---------------------------------------------------------------- layout (positions)
  // Skip edges bow around the columns they jump over: above them if they start and end in the
  // upper half, below otherwise.
  // Between token layers (a transformer's residual stream) they only bow a little instead: going
  // around a tall Q/K/V column would take them off the page. A token's upper features bow up, the
  // lower ones down, so the two strands of a d = 2 stream separate.
  function geom(a, b) {
    const I = ix(), la = I.li.get(a.layer), lb = I.li.get(b.layer);
    if (lb - la >= 2 && I.tok[la]?.d && I.tok[lb]?.d) {
      const p = tokPos(a.id), s = I.tok[la], dir = p && p.f > (s.d - 1) / 2 ? -1 : 1;
      const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, off = 2 * 34 * dir;
      return curve(a, { x: (a.x + b.x) / 2 + (dy / len) * off, y: (a.y + b.y) / 2 - (dx / len) * off }, b);
    }
    if (lb - la >= 2) {
      let top = Infinity, bot = -Infinity;
      for (let l = la + 1; l < lb; l++) {
        const c = I.cols[l];
        if (c.n) { top = Math.min(top, c.minY); bot = Math.max(bot, c.maxY); }
      }
      if (top < Infinity) {
        const my = (a.y + b.y) / 2;
        const apex = my <= (top + bot) / 2
          ? Math.min(top - R - 20, Math.min(a.y, b.y) - 36)
          : Math.max(bot + R + 34, Math.max(a.y, b.y) + 36);
        return curve(a, { x: (a.x + b.x) / 2, y: 2 * apex - my }, b);
      }
    }
    return straight(a, b);
  }
  function posChanged() {
    for (const n of store.net.nodes) {
      const r = nodes.get(n.id);
      if (r && (r.x !== n.x || r.y !== n.y)) return true;
    }
    return false;
  }
  function layoutAll() {
    const net = store.net, I = ix();
    for (const n of net.nodes) {
      const r = nodes.get(n.id);
      if (!r || (r.x === n.x && r.y === n.y)) continue;
      r.x = n.x; r.y = n.y;
      r.g.setAttribute('transform', `translate(${r1(n.x)},${r1(n.y)})`);
    }
    for (const e of net.edges) {
      const r = edges.get(e.id), a = I.nodeById.get(e.from), b = I.nodeById.get(e.to);
      if (!r || !a || !b) continue;
      r.G = geom(a, b);
      put(r.line, 'd', r.G.d);
      put(r.hit, 'd', r.G.d);
      if (r.olOn) put(r.ol, 'd', r.G.d);   // the outline follows only while it shows (see outline())
      // The number sits about 40% along from the source, where the edges fanning out of it have
      // parted (a skip edge's halfway, on its bow, clear of the columns it jumps); the edges into one
      // target are staggered by source row, so their numbers don't stack.
      const la = I.li.get(a.layer), n = I.byLayer[la]?.length || 1, u0 = Math.abs(I.li.get(b.layer) - la) >= 2 ? 0.5 : 0.4;
      const p = r.G.at(u0 + 0.3 * (((I.rank.get(a.id) ?? 0) + 0.5) / n - 0.5));
      put(r.lab, 'transform', `translate(${r1(p.x)},${r1(p.y)})`);
    }
    const ext = layoutTokens();
    layoutAtt();
    const top = (I.box ? I.box.minY : I.cy) - HEAD_UP;
    const bottom = (I.box ? I.box.maxY : I.cy) + LANE_DOWN;
    net.layers.forEach((l, i) => {
      const r = layers.get(l.id), c = I.cols[i];
      if (!r) return;
      put(r.g, 'transform', `translate(${r1(c.x)},${r1(top)})`);
      const x0 = Math.min(c.minX - R - 14, (ext[i]?.x0 ?? Infinity) - 8), x1 = Math.max(c.maxX + R + 14, (ext[i]?.x1 ?? -Infinity) + 8);
      put(r.band, 'x', r1(x0));
      put(r.band, 'width', r1(x1 - x0));
      put(r.band, 'y', r1(top + LANE_TOP));
      put(r.band, 'height', r1(Math.max(0, Math.max(bottom, (ext[i]?.y1 ?? -Infinity) + 6) - top - LANE_TOP)));
      if (!r.measured && shown) measureHead(r);
    });
    fitHeads();
    for (const p of gPulses.children) put(p, 'd', pathOf(p.dataset.id) ?? '');
    layoutPair();
    placeHint();
    placeAttLabels();
  }
  const pathOf = id => (edges.get(id) || attLines.get(id))?.G?.d;
  // Headers get the text boost (ts), and may be wider than their lane, but never run into a
  // neighbour. Two headers that would touch first drop a sub line where that makes one narrower
  // (the narrower header's alone, else the other's, else both), and only then shrink by one factor
  // (long layer names, columns packed close). Returns layer id -> { s: scale, sub: sub line shown }.
  function headFit(ts) {
    const I = ix(), out = new Map();
    const hs = store.net.layers.map((l, i) => ({ r: layers.get(l.id), x: I.cols[i].x, sub: true, s: ts }))
      .filter(h => h.r).sort((a, b) => a.x - b.x);
    const W = h => headW(h.r, h.sub);
    for (let k = 0; k + 1 < hs.length; k++) {
      const a = hs[k], b = hs[k + 1], room = b.x - a.x - HEAD.air;
      const fits = () => ((W(a) + W(b)) * ts) / 2 <= room;
      if (fits()) continue;
      const can = (W(a) <= W(b) ? [a, b] : [b, a]).filter(h => h.sub && h.r.subW > h.r.nameW);
      const tries = can.length === 2 ? [[can[0]], [can[1]], can] : [can];
      for (const t of tries) {
        for (const h of t) h.sub = false;
        if (fits()) break;
        if (t !== tries[tries.length - 1]) for (const h of t) h.sub = true;
      }
      if (fits()) continue;
      const s = (2 * room) / (W(a) + W(b));
      a.s = Math.min(a.s, s);
      b.s = Math.min(b.s, s);
    }
    for (const h of hs) out.set(h.r.id, { s: Math.round(Math.max(0.6, h.s) * 100) / 100, sub: h.sub });
    return out;
  }
  function fitHeads() {
    for (const [id, f] of headFit(textScale)) placeHead(layers.get(id), f);
  }
  // Lay out one header for fit f: the boost, the sub line, the heatmap left of the text (centred on
  // the two lines), and the pill round it all. el: a png() clone's elements instead of the live ones.
  function placeHead(r, f, el = null) {
    const E = el || { g: r.g, gi: r.gi, bg: r.bg, name: r.name, sub: r.sub, tip: r.tip, hm: r.hm?.g };
    const sub = f.sub, tw = Math.max(24, r.nameW, sub ? r.subW : 0), w = headW(r, sub);
    const cx = r.hm ? (r.hm.W + HEAD.hmGap) / 2 : 0;   // the text's centre: the whole header is centred on the column
    put(E.gi, 'transform', f.s === 1 ? null : `scale(${f.s})`);
    if (E.g.__nosub !== !sub) { E.g.__nosub = !sub; E.g.classList.toggle('nosub', !sub); }
    if (E.tip) txt(E.tip, sub ? '' : `${r.nameS}: ${r.subS}`);
    put(E.name, 'x', cx ? r1(cx) : null);
    put(E.sub, 'x', cx ? r1(cx) : null);
    let bot = sub ? HEAD.bot : HEAD.botName;
    if (r.hm && E.hm) {
      const capTop = HEAD.name - 10, y = capTop + (HEAD.sub - capTop - r.hm.H) / 2;
      put(E.hm, 'transform', `translate(${r1(cx - tw / 2 - HEAD.hmGap - r.hm.W)},${r1(y)})`);
      bot = Math.max(bot, y + r.hm.H + 6);
    }
    put(E.bg, 'x', r1(-w / 2));
    put(E.bg, 'width', r1(w));
    put(E.bg, 'height', r1(bot - HEAD.top));
  }

  // Token boxes around each token's neurons, group bands around each group's boxes. Returns each
  // token layer's extent (by layer index) so its lane can hold them.
  function layoutTokens() {
    const net = store.net, I = ix(), ext = [];
    net.layers.forEach((l, li) => {
      const r = toks.get(l.id), s = I.tok[li], ns = I.byLayer[li];
      if (!r || !s?.d || ns.length !== s.G * s.T * s.d) return;
      const boxes = r.boxes.map(b => {
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let f = 0; f < s.d; f++) {
          const n = ns[(b.g * s.T + b.t) * s.d + f];
          x0 = Math.min(x0, n.x); x1 = Math.max(x1, n.x); y0 = Math.min(y0, n.y); y1 = Math.max(y1, n.y);
        }
        return { b, cy0: y0, cy1: y1, x0: x0 - R - r.gut, x1: x1 + R + TOK.r, y0: y0 - R - TOK.t, y1: y1 + R + TOK.b };
      });
      // Tokens packed tighter than their boxes: neighbours split the space between their neurons.
      const order = [...boxes].sort((a, b) => a.cy0 - b.cy0);
      for (let k = 0; k + 1 < order.length; k++) {
        const a = order[k], b = order[k + 1];
        if (a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 + TOK.gap <= b.y0) continue;
        const mid = (a.cy1 + b.cy0) / 2;
        a.y1 = Math.max(a.cy1 + R + 2, mid - TOK.gap / 2);
        b.y0 = Math.min(b.cy0 - R - 2, mid + TOK.gap / 2);
      }
      let E = null;
      const grow = (x0, x1, y0, y1) => {
        E = E ? { x0: Math.min(E.x0, x0), x1: Math.max(E.x1, x1), y0: Math.min(E.y0, y0), y1: Math.max(E.y1, y1) } : { x0, x1, y0, y1 };
      };
      for (const q of boxes) {
        const { b } = q;
        put(b.rect, 'x', r1(q.x0)); put(b.rect, 'y', r1(q.y0));
        put(b.rect, 'width', r1(q.x1 - q.x0)); put(b.rect, 'height', r1(q.y1 - q.y0));
        // The label sits left of the box, level with the middle of its neurons.
        put(b.lab, 'x', r1(q.x0 - TOK_LAB));
        put(b.lab, 'y', r1((q.cy0 + q.cy1) / 2));
        grow(q.x0, q.x1, q.y0, q.y1);
      }
      const bands = r.bands.map(band => {
        const mine = boxes.filter(q => q.b.g === band.g);
        return mine.length && {
          band, x0: Math.min(...mine.map(q => q.x0)) - GRP.l, x1: Math.max(...mine.map(q => q.x1)) + GRP.r,
          y0: Math.min(...mine.map(q => q.y0)) - GRP.t, y1: Math.max(...mine.map(q => q.y1)) + GRP.b,
          in0: Math.min(...mine.map(q => q.y0)), in1: Math.max(...mine.map(q => q.y1)),
        };
      }).filter(Boolean).sort((a, b) => a.y0 - b.y0);
      for (let k = 0; k + 1 < bands.length; k++) {   // groups packed tight: bands give way, never into a box
        const a = bands[k], b = bands[k + 1];
        if (a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 + 2 <= b.y0) continue;
        const mid = (a.in1 + b.in0) / 2;
        a.y1 = Math.max(a.in1, mid - 1);
        b.y0 = Math.min(b.in0, mid + 1);
      }
      for (const b of bands) {
        const { band, x0, x1, y0, y1 } = b;
        put(band.rect, 'x', r1(x0)); put(band.rect, 'y', r1(y0));
        put(band.rect, 'width', r1(x1 - x0)); put(band.rect, 'height', r1(y1 - y0));
        // Left of the band, as in Q = [ ... ]; above it when another group sits right there (dragged
        // side by side), or when the row labels do.
        const top = r.row || bands.some(o => o !== b && o.x1 <= x0 + 1 && x0 - o.x1 < 34 && o.y0 < y1 && y0 < o.y1);
        band.lab.classList.toggle('top', top);
        put(band.lab, 'x', r1(top ? x0 + 10 : x0 - 5));
        put(band.lab, 'y', r1(top ? y0 - 10 : (y0 + y1) / 2));
        grow(top ? x0 : x0 - 26, x1, top ? y0 - 22 : y0, y1);
      }
      ext[li] = E;
    });
    return ext;
  }
  // Attention edges run from V_j,f (the layer before) to Z_i,f.
  function layoutAtt() {
    const I = ix();
    for (const [id, q] of atts) {
      const l = I.li.get(id), a = I.att[l];
      for (const rec of q.lines) {
        const v = a && tokNode(l - 1, a.vG, rec.j, rec.f), z = a && tokNode(l, 0, rec.i, rec.f);
        rec.G = v && z ? straight(v, z) : null;
        put(rec.line, 'd', rec.G?.d ?? '');
        put(rec.hit, 'd', rec.G?.d ?? '');
      }
    }
  }
  // The empty-net hint sits under the lanes, centred on them (screen px: it doesn't scale).
  function placeHint() {
    if (hint.getAttribute('display') === 'none') return;
    const I = ix(), xs = I.cols.map(c => c.x);
    const x = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 450;
    put(hint, 'x', r1(x * V.k + V.x - HINT_W / 2));
    put(hint, 'y', r1((I.cy + LANE_DOWN) * V.k + V.y + 30 - HINT_H / 2));
  }
  function layoutPair() {
    const I = ix(), a = pairIds && I.nodeById.get(pairIds[0]), b = pairIds && I.nodeById.get(pairIds[1]);
    if (a && b) {
      const [s, t] = (I.li.get(a.layer) ?? 0) <= (I.li.get(b.layer) ?? 0) ? [a, b] : [b, a];
      put(pairEl, 'd', geom(s, t).d);
      put(pairEl, 'display', null);
    } else put(pairEl, 'display', 'none');
  }

  // ---------------------------------------------------------------- paint (values)
  // An edge's numbers. Hovered or selected (mode true), a tied edge also names its shared entry,
  // e.g. W_Q(1,2) = 0.53 with ∂L/∂w of its own and Σ over the group, and a fixed edge says so. The
  // rest of a lit tie group (mode 'tie') shows just the shared value.
  function paintLabel(r, e, mode = r.lab.__show) {
    if (mode === 'tie') {
      txt(r.tn, ''); txt(r.ts, ''); put(r.ts, 'dy', null); put(r.tv, 'dy', null);
      txt(r.tv, num(e.w)); txt(r.t2, '');
      return;
    }
    const full = !!mode, bwd = store.state.bwd, tied = typeof e.tie === 'string' && e.tie;
    const tp = full && tied ? tieParse(e.tie) || { name: e.tie, i: null } : null;
    const [base, lo] = tp ? tiePlain(tp.name) : ['', ''];
    txt(r.tn, base);
    txt(r.ts, lo);
    put(r.ts, 'dy', lo ? '0.3em' : null);
    put(r.tv, 'dy', lo ? '-0.3em' : null);
    txt(r.tv, tp ? `${tp.i ? `(${tp.i},${tp.j})` : ''} = ${num(e.w)}` : full && e.fixed ? `${num(e.w)} fixed` : num(e.w));
    // ∂L/∂w only on the hovered or selected edge: with W on, the rest show their value alone.
    const g = full ? bwd?.edge?.[e.id] : undefined, sum = tp ? bwd?.tie?.[e.tie] : undefined;
    txt(r.t2, e.fixed || !isNum(g) ? '' : `∂L/∂w ${numg(g)}${isNum(sum) ? ` · Σ ${numg(sum)}` : ''}`);
  }
  function paint() {
    const net = store.net, I = ix(), th = theme(), L = net.layers.length;
    const { fwd, bwd } = store.state;
    svg.classList.toggle('bwd', !!bwd);
    const act = n => {
      const v = fwd?.node?.[n.id]?.a;
      return isNum(v) ? v : I.li.get(n.layer) === 0 && isNum(n.value) ? n.value : NaN;
    };
    // The matrix panel's scales, so a neuron, an edge and their cells share one colour.
    const maxW = maxAbs(net.edges.map(e => e.w)) || 1;
    const maxA = (fwd ? maxAbs(fwd.a, fwd.z) : maxAbs(net.nodes.map(act))) || 1;
    let maxD = 0;
    if (bwd && fwd) {
      maxD = maxAbs(bwd.dZ, bwd.dA);
      const ks = net.layers.map((_, l) => new Set(l ? [l - 1] : []));
      for (const e of net.edges) {
        const a = I.nodeById.get(e.from), b = I.nodeById.get(e.to);
        if (a && b) ks[I.li.get(b.layer)]?.add(I.li.get(a.layer));
      }
      ks.forEach((set, l) => { for (const k of set) maxD = Math.max(maxD, maxAbs(bwd.dZ?.[l]) * maxAbs(fwd.a?.[k])); });
    }
    maxD ||= 1;
    for (const n of net.nodes) {
      const r = nodes.get(n.id);
      if (!r) continue;
      const l = I.li.get(n.layer), a = act(n);
      put(r.fill, 'fill', colorFor(a, maxA, th));
      txt(r.val, num(a));
      const d = l > 0 ? bwd?.node?.[n.id]?.dz : undefined;
      if (isNum(d)) {
        put(r.gring, 'stroke', colorFor(d, maxD, th));
        put(r.gring, 'stroke-width', r1(1.5 + 4 * Math.min(1, Math.abs(d) / maxD)));
        txt(r.grad, `δ ${numg(d)}`);
      } else {
        put(r.gring, 'stroke', 'none');
        txt(r.grad, '');
      }
      put(r.hl, 'r', isNum(d) ? HL_R_D : HL_R);   // outside the δ ring when there is one
      txt(r.tgt, L > 1 && l === L - 1 && isNum(n.target) ? `y = ${num(n.target)}` : '');
      paintBias(r, n, l > 0 && !I.tok[l]?.att);
    }
    for (const e of net.edges) {
      const r = edges.get(e.id);
      if (!r) continue;
      const w = isNum(e.w) ? e.w : 0;
      r.sw = r1(1.2 + 5 * Math.min(1, Math.abs(w) / maxW));
      put(r.line, 'stroke', colorFor(w, maxW, th));
      put(r.line, 'stroke-width', r.sw);
      if (r.olOn) put(r.ol, 'stroke-width', r1(r.sw + 3));
      if (showW || r.lab.__show) paintLabel(r, e);
    }
    paintAtt();
    paintLens();
  }
  // b = 0.12, or a shared bias by name: b_Q(2) = 0.12 (node.tie 'b_Q:2').
  function paintBias(r, n, on) {
    const m = on && typeof n.tie === 'string' && n.tie ? /^(.*?)(?::(\d+))?$/.exec(n.tie) : null;
    const [base, lo] = m ? tiePlain(m[1]) : [on ? 'b' : '', ''];
    txt(r.bn, base);
    txt(r.bs, lo);
    put(r.bs, 'dy', lo ? '0.3em' : null);
    put(r.bv, 'dy', lo ? '-0.3em' : null);
    txt(r.bv, on ? `${m?.[2] ? `(${m[2]})` : ''} = ${num(n.bias ?? 0)}` : '');
  }
  // A_ij: the attention edges' width and opacity, the heatmap cells, the shown row's labels.
  // Causally masked pairs (j > i) have no edge (paintLens hides them); their cells are hatched.
  function attA(l, h, i, j) {
    const v = store.state.fwd?.attn?.[l]?.heads?.[h]?.A?.[i]?.[j];
    return isNum(v) ? v : NaN;
  }
  function paintAtt() {
    const I = ix();
    for (const [id, q] of atts) {
      const l = I.li.get(id), a = I.att[l];
      if (!a) continue;
      for (const rec of q.lines) {
        const v = attA(l, rec.h, rec.i, rec.j), u = isNum(v) ? clamp(v, 0, 1) : 0;
        put(rec.line, 'stroke-width', r1(0.8 + 6.4 * u));
        put(rec.line, 'stroke-opacity', isNum(v) ? (0.06 + 0.88 * u).toFixed(3) : '0.12');
      }
      for (const c of q.hm?.cells || []) {
        const v = attA(l, c.h, c.i, c.j), masked = a.causal && c.j > c.i;
        put(c.el, 'class', masked ? 'nnv-hm-c mask' : isNum(v) ? 'nnv-hm-c' : 'nnv-hm-c nan');
        put(c.el, 'fill-opacity', masked || !isNum(v) ? null : (0.1 + 0.9 * clamp(v, 0, 1)).toFixed(3));
      }
      for (const t of q.labs) if (t.on) txt(t.el, num(attA(l, t.h, t.i, t.j)));
    }
  }

  // ---------------------------------------------------------------- lens (dim, hide)
  // --le on an element is its opacity (view.css), 1 when unset; the hover and step dimming
  // multiply it. Runs after every paint (A, and so a followed token's keys and values, change
  // with the values) and on every lens change; only what changed touches the DOM.
  function setLe(el, v) {
    if (!el) return;
    const s = v >= 0.995 ? '' : (DIM + (1 - DIM) * clamp(v, 0, 1)).toFixed(2);
    if (el.__le === s) return;
    el.__le = s;
    if (s) el.style.setProperty('--le', s);
    else el.style.removeProperty('--le');
  }
  function cls(el, c, on) {
    if (!el) return;
    const k = `__${c}`;
    if (el[k] === on) return;
    el[k] = on;
    el.classList.toggle(c, on);
  }
  function paintLens() {
    const net = store.net, I = ix(), lens = store.state.lens;
    E = null;
    if (lens) {
      try { E = emphasis(net, store.state.fwd, lens); } catch (err) { console.error('[nn] lens:', err); }
    }
    const dim = !!E?.any, hides = !!E?.hides;
    const nv = id => (dim ? E.node(id) : 1);
    for (const n of net.nodes) {
      const r = nodes.get(n.id);
      if (!r) continue;
      const v = nv(n.id);
      setLe(r.g, v);
      cls(r.g, 'lz-dim', v < NUM_MIN);
    }
    for (const e of net.edges) {
      const r = edges.get(e.id);
      if (!r) continue;
      const gone = hides && E.hidden.edge(e.id), v = dim ? E.edge(e.id) : 1;
      put(r.g, 'display', gone ? 'none' : null);
      setLe(r.g, v);
      setLe(r.lab, v);
      cls(r.lab, 'lz-gone', gone);
      cls(r.lab, 'lz-dim', v < NUM_MIN);
    }
    for (const [id, q] of atts) {
      const l = I.li.get(id), a = I.att[l];
      if (!a) continue;
      for (const rec of q.lines) {
        const gone = (a.causal && rec.j > rec.i) || (hides && E.hidden.attn(l, rec.i, rec.j, rec.h));
        put(rec.g, 'display', gone ? 'none' : null);
        setLe(rec.g, dim ? E.attn(l, rec.i, rec.j, rec.h) : 1);
      }
      for (const t of q.labs) cls(t.el, 'lz-gone', hides && E.hidden.attn(l, t.i, t.j, t.h));   // A moved under minA
      // The heatmap dims with its header; within it, the followed token's row and the kept head.
      const rows = dim ? E.rows(l) : null, hs = dim ? E.heads(l) : null;
      for (const c of q.hm?.cells || []) setLe(c.el, (!rows?.size || rows.has(c.i)) && (!hs || hs.has(c.h)) ? 1 : 0);
    }
    for (const [id, r] of toks) {
      const l = I.li.get(id), s = I.tok[l], ns = I.byLayer[l];
      if (!s?.d || ns?.length !== s.G * s.T * s.d) continue;
      const most = (g, t0, t1) => {
        let v = 0;
        for (let t = t0; t < t1; t++) for (let f = 0; f < s.d; f++) v = Math.max(v, nv(ns[(g * s.T + t) * s.d + f].id));
        return v;
      };
      for (const b of r.boxes) { const v = most(b.g, b.t, b.t + 1); setLe(b.el, v); setLe(b.lab, v); }
      for (const b of r.bands) { const v = most(b.g, 0, s.T); setLe(b.rect, v); setLe(b.lab, v); }
    }
    const fl = E?.lens.focus?.layer;
    net.layers.forEach((l, i) => {
      const r = layers.get(l.id);
      if (!r) return;
      const v = dim ? E.layer(i) : 1;
      setLe(r.g, v);
      setLe(r.band, v);
      cls(r.g, 'lz-foc', fl === l.id);
      cls(r.band, 'lz-foc', fl === l.id);
    });
  }

  // ---------------------------------------------------------------- highlight (sel, hover, anim)
  // What a sel / hover target lights up. Matrix rows and columns name layers by index or id.
  function resolve(t) {
    const out = { nodes: [], edges: [], layers: [], relNodes: [], relEdges: [], ties: [], tokens: [], pair: null, bias: null, any: false };
    if (!t) return out;
    const net = store.net, I = ix();
    const li = v => (typeof v === 'number' ? v : I.li.get(v));
    const into = (id, pick) => {
      for (const e of net.edges) {
        if (e.to !== id || (pick && !pick(e))) continue;
        out.relEdges.push(e.id); out.relNodes.push(e.from);
      }
    };
    switch (t.kind) {
      case 'node': case 'bias':
        if (!I.nodeById.has(t.id)) break;
        out.nodes.push(t.id);
        if (t.kind === 'bias') {
          // A shared bias (node.tie): every neuron that uses it shows it.
          out.bias = [t.id, ...(I.biasTies.get(I.nodeById.get(t.id).tie) || []).filter(id => id !== t.id)];
          break;
        }
        for (const e of net.edges) {
          if (e.from === t.id) { out.relEdges.push(e.id); out.relNodes.push(e.to); }
          else if (e.to === t.id) { out.relEdges.push(e.id); out.relNodes.push(e.from); }
        }
        break;
      case 'edge': {
        const e = I.edgeById.get(t.id);
        if (!e) break;
        out.edges.push(e.id);
        out.relNodes.push(e.from, e.to);
        // A tied edge is one entry of a shared matrix: light every edge that uses it.
        for (const o of (e.tie && I.ties.get(e.tie)) || []) {
          if (o === e.id) continue;
          const x = I.edgeById.get(o);
          out.ties.push(o);
          out.relNodes.push(x.from, x.to);
        }
        break;
      }
      case 'layer': {
        const i = I.li.get(t.id);
        if (i === undefined) break;
        out.layers.push(t.id);
        for (const n of I.byLayer[i]) { out.relNodes.push(n.id); into(n.id); }
        break;
      }
      case 'pair':
        if (I.nodeById.has(t.from) && I.nodeById.has(t.to)) { out.pair = [t.from, t.to]; out.relNodes.push(t.from, t.to); }
        break;
      case 'row': {
        const n = I.byLayer[li(t.layer)]?.[t.i];
        if (n) { out.nodes.push(n.id); into(n.id); }
        break;
      }
      case 'col': {
        const l = li(t.layer), src = I.byLayer[li(t.k)]?.[t.j];
        if (!src) break;
        out.nodes.push(src.id);
        for (const e of net.edges) {
          const to = I.nodeById.get(e.to);
          if (e.from === src.id && to && I.li.get(to.layer) === l) { out.relEdges.push(e.id); out.relNodes.push(e.to); }
        }
        break;
      }
      case 'token': {
        // the token's neurons (of group g, and of head h in attention's Z or V) and their edges
        const l = li(t.layer), s = I.tok[l];
        if (!s?.d || !Number.isInteger(t.t) || t.t < 0 || t.t >= s.T) break;
        const hd = Number.isInteger(t.h) ? t.h : null, a = I.att[l], b = I.att[l + 1];
        for (let g = 0; g < s.G; g++) {
          if (Number.isInteger(t.g) && t.g !== g) continue;
          out.tokens.push({ l, g, t: t.t });
          const dh = hd === null ? 0 : a ? a.dh : b ? b.dh : 0;   // heads split Q, K and V by column too
          for (let f = 0; f < s.d; f++) {
            const n = dh && Math.floor(f / dh) !== hd ? null : tokNode(l, g, t.t, f);
            if (!n) continue;
            out.nodes.push(n.id);
            for (const e of net.edges) {
              if (e.from === n.id) { out.relEdges.push(e.id); out.relNodes.push(e.to); }
              else if (e.to === n.id) { out.relEdges.push(e.id); out.relNodes.push(e.from); }
            }
          }
        }
        break;
      }
      default: break;
    }
    out.any = !!(out.nodes.length || out.edges.length || out.layers.length || out.pair);
    return out;
  }
  // state.anim { dir, l, i, phase }: neuron i of layer l; fwd lights its incoming edges, bwd its outgoing ones.
  // An attention layer steps a token at a time (matrix.js: anim.i is the token's first neuron), in
  // phases: 'scores' (q_i · k_j: the query and the keys light up), 'softmax' (row i of A: its edges
  // and numbers), 'sum' (Σ_j A_ij v_j: pulses from the values); backward, the gradient runs back
  // along those edges and out of the token's outgoing weights.
  function resolveAnim(a) {
    const out = { node: null, nodes: [], dir: 'fwd', edges: [], rel: [], phase: '', focus: null };
    if (!a) return out;
    const I = ix(), l = typeof a.l === 'number' ? a.l : I.li.get(a.l), n = I.byLayer[l]?.[a.i];
    if (!n) return out;
    out.node = n.id;
    out.nodes.push(n.id);
    out.dir = a.dir === 'bwd' ? 'bwd' : 'fwd';
    out.phase = String(a.phase ?? '');
    const p = tokPos(n.id), at = p && I.att[p.l], bwd = out.dir === 'bwd';
    const lines = L => atts.get(store.net.layers[L]?.id)?.lines || [];
    const outOf = id => {
      for (const e of store.net.edges) if (e.from === id) { out.edges.push(e.id); out.rel.push(e.to); }
    };
    if (at) {
      const t = p.t, ph = bwd ? 'bwd' : out.phase, zs = [];
      for (let f = 0; f < at.d; f++) { const z = tokNode(p.l, 0, t, f); if (z) zs.push(z.id); }
      out.nodes = zs;
      out.focus = { l: p.l, dir: 'row', t, h: null, f: null, labels: ph !== 'scores' };
      if ((ph === 'scores' || ph === 'bwd') && at.qG >= 0 && at.kG >= 0) {
        for (let f = 0; f < at.d; f++) {
          out.rel.push(tokNode(p.l - 1, at.qG, t, f)?.id);
          for (let j = 0; j < at.T; j++) if (!(at.causal && j > t)) out.rel.push(tokNode(p.l - 1, at.kG, j, f)?.id);
        }
      }
      if (ph === 'sum' || ph === 'bwd') {
        for (const rec of lines(p.l)) {
          if (rec.i !== t || (at.causal && rec.j > rec.i)) continue;
          out.edges.push(rec.id);
          out.rel.push(tokNode(p.l - 1, at.vG, rec.j, rec.f)?.id);
        }
      }
      if (bwd) zs.forEach(outOf);
      out.rel = out.rel.filter(Boolean);
      return out;
    }
    if (bwd) outOf(n.id);
    else for (const e of store.net.edges) if (e.to === n.id) { out.edges.push(e.id); out.rel.push(e.from); }
    // A V neuron's gradient comes back from every Z_i,f it fed (dV = Aᵀ dZ).
    const nx = p && I.att[p.l + 1];
    if (bwd && nx && p.g === nx.vG) {
      for (const rec of lines(p.l + 1)) {
        if (rec.f !== p.f || rec.j !== p.t || (nx.causal && rec.j > rec.i)) continue;
        out.edges.push(rec.id);
        const z = tokNode(p.l + 1, 0, rec.i, rec.f);
        if (z) out.rel.push(z.id);
      }
    }
    return out;
  }

  // The attention row (or column) to show: a hovered token (a Z token, heatmap cell or attention
  // edge: its row; a V token: its column), else a hovered Z neuron (its token's row, in its head)
  // or V neuron (its column, that feature), else the step-through's token, else the selected
  // neuron. { l, dir: 'row' | 'col', t, h: head | null, f: feature | null, hover }.
  function focusOfNode(id) {
    const I = ix(), p = tokPos(id);
    if (!p) return null;
    const a = I.att[p.l], b = I.att[p.l + 1];
    if (a) return { l: p.l, dir: 'row', t: p.t, h: Math.floor(p.f / a.dh), f: null };
    if (b && p.g === b.vG) return { l: p.l + 1, dir: 'col', t: p.t, h: Math.floor(p.f / b.dh), f: p.f };
    return null;
  }
  function focusOfToken(hv) {
    const I = ix(), l = typeof hv.layer === 'number' ? hv.layer : I.li.get(hv.layer);
    if (l === undefined || !Number.isInteger(hv.t)) return null;
    const h = Number.isInteger(hv.h) ? hv.h : null, b = I.att[l + 1];
    if (I.att[l]) return { l, dir: 'row', t: hv.t, h, f: null };
    if (!b) return null;   // on the Q, K, V layer: a query token's row, a key or value token's column
    return { l: l + 1, dir: Number.isInteger(hv.g) && hv.g === b.qG ? 'row' : 'col', t: hv.t, h, f: null };
  }
  // After hover and the step-through, a followed token (lens.token) shows its row of A in every
  // attention layer (lens.head: that head's), with no extra lighting: the lens does the dimming.
  function attFoci(A) {
    if (!atts.size) return [];
    const I = ix(), st = store.state, li = v => (typeof v === 'number' ? v : I.li.get(v));
    const hv = st.hover, hn = hv?.kind === 'node' ? hv.id : hv?.kind === 'row' ? I.byLayer[li(hv.layer)]?.[hv.i]?.id : null;
    const fh = hv?.kind === 'token' ? focusOfToken(hv) : hn && focusOfNode(hn);
    if (fh) return [{ ...fh, hover: true }];
    const fa = A?.focus || (A?.node && focusOfNode(A.node));
    if (fa) return [fa];
    const t = E?.lens.token, h = E?.lens.head;
    if (Number.isInteger(t)) {
      const out = [];
      I.att.forEach((a, l) => { if (a && t < a.T) out.push({ l, dir: 'row', t, h: Number.isInteger(h) && h < a.heads ? h : null, f: null, lens: true }); });
      if (out.length) return out;
    }
    const fs = st.sel?.kind === 'node' && focusOfNode(st.sel.id);
    return fs ? [fs] : [];
  }
  function highlightAtt(foci, mark, onNode) {
    for (const q of atts.values()) {
      for (const t of q.labs) t.on = false;
      for (const hl of q.hm?.hl || []) put(hl.el, 'display', 'none');
    }
    for (const F of foci) showAtt(F, mark, onNode);
    placeAttLabels();
  }
  function showAtt(F, mark, onNode) {
    const I = ix(), a = F && I.att[F.l], q = a && atts.get(store.net.layers[F.l].id);
    if (!q) return;
    const row = F.dir === 'row', inHead = h => F.h == null || h === F.h;
    const hit = rec => (row ? rec.i : rec.j) === F.t && inHead(rec.h) && (F.f == null || rec.f === F.f) && !(a.causal && rec.j > rec.i);
    for (const rec of q.lines) {
      if (F.labels === false || F.lens || !hit(rec)) continue;   // labels false: the scores step, before A exists
      mark(rec.g, 'rel');
      if (F.hover) {
        onNode(tokNode(F.l - 1, a.vG, rec.j, rec.f)?.id, 'rel');
        onNode(tokNode(F.l, 0, rec.i, rec.f)?.id, 'rel');
      }
    }
    // A_ij = softmax_j(q_i · k_j): the query of the row and every key of its head.
    if (F.hover && row && a.qG >= 0 && a.kG >= 0) {
      for (let f = 0; f < a.d; f++) {
        if (!inHead(Math.floor(f / a.dh))) continue;
        onNode(tokNode(F.l - 1, a.qG, F.t, f)?.id, 'rel');
        for (let j = 0; j < a.T; j++) if (!(a.causal && j > F.t)) onNode(tokNode(F.l - 1, a.kG, j, f)?.id, 'rel');
      }
    }
    const box = toks.get(store.net.layers[row ? F.l : F.l - 1].id)?.boxes.find(b => b.t === F.t && b.g === (row ? 0 : a.vG));
    if (box) { mark(box.el, 'foc'); mark(box.lab, 'foc'); }
    // A number rides a visible edge only: not one the lens hides, nor (for its own row) dims.
    const off = t => !!E && (E.hidden.attn(F.l, t.i, t.j, t.h) || (F.lens && E.any && E.attn(F.l, t.i, t.j, t.h) < NUM_MIN));
    for (const t of q.labs) {
      if (F.labels === false || (row ? t.i : t.j) !== F.t || !inHead(t.h) || (a.causal && t.j > t.i) || off(t)) continue;
      t.on = true;
      mark(t.el, 'show');
      put(t.el, 'data-dir', F.dir);
      txt(t.el, num(attA(F.l, t.h, t.i, t.j)));
    }
    const hm = q.hm;
    for (const [h, hl] of (hm?.hl || []).entries()) {
      if (!inHead(h)) continue;
      put(hl.el, 'display', null);
      put(hl.el, 'x', r1(hl.x0 + (row ? 0 : F.t * hm.cell) - 1.5));
      put(hl.el, 'y', r1((row ? F.t * hm.cell : 0) - 1.5));
      put(hl.el, 'width', r1((row ? a.T : 1) * hm.cell + 2));   // 1.5 px round the cells (each cell - 1 wide)
      put(hl.el, 'height', r1((row ? 1 : a.T) * hm.cell + 2));
    }
  }
  // A shown A_ij sits on its edge: near V_j for a row (the edges fan into Z_i), near Z_i for a column.
  // It rides the edge of the head's last feature: in a token grid that is V's right-hand column,
  // so the number lands clear of the Q, K, V neurons.
  function placeAttLabels() {
    const I = ix();
    for (const [id, q] of atts) {
      const l = I.li.get(id), a = I.att[l];
      if (!a) continue;
      for (const t of q.labs) {
        if (!t.on) continue;
        const f = (t.h + 1) * a.dh - 1, rec = attLines.get(`att:${id}:${t.i}:${t.j}:${f}`), G = rec?.G;
        if (!G) continue;
        const p = G.at(t.el.getAttribute('data-dir') === 'col' ? 0.72 : 0.3);
        put(t.el, 'x', r1(p.x));
        put(t.el, 'y', r1(p.y));
      }
    }
  }
  // The edges whose HI outline can show (hovered, selected, tied to those, lit): only these keep
  // their outline's path and width current, so training and dragging don't pay for the rest.
  let outlined = [];
  function outline(ids) {
    for (const r of outlined) r.olOn = false;
    outlined = [];
    for (const id of ids) {
      const r = edges.get(id);
      if (!r || r.olOn) continue;
      r.olOn = true;
      outlined.push(r);
      put(r.ol, 'd', r.G?.d ?? '');
      put(r.ol, 'stroke-width', r1((r.sw ?? 1.2) + 3));
    }
  }
  function highlight() {
    for (const [e, c] of marked) { e.classList.remove(c); if (c === 'show') e.__show = false; }
    marked = [];
    const mark = (e, c) => { if (e) { e.classList.add(c); marked.push([e, c]); } };
    const onNode = (id, c) => mark(nodes.get(id)?.g, c);
    const onEdge = (id, c) => { const r = edges.get(id) || attLines.get(id); if (r) { mark(r.g, c); mark(r.lab, c); } };
    const onLayer = (id, c) => { const r = layers.get(id); if (r) { mark(r.g, c); mark(r.band, c); } };
    const I = ix(), S = resolve(store.state.sel), H = resolve(store.state.hover), A = resolveAnim(store.state.anim);
    outline([...S.edges, ...S.ties, ...H.edges, ...H.ties, ...A.edges]);
    S.nodes.forEach(id => onNode(id, 'sel'));
    S.edges.forEach(id => onEdge(id, 'sel'));
    S.ties.forEach(id => onEdge(id, 'tie'));
    S.layers.forEach(id => onLayer(id, 'sel'));
    H.nodes.forEach(id => onNode(id, 'hov'));
    H.edges.forEach(id => onEdge(id, 'hov'));
    H.ties.forEach(id => onEdge(id, 'tie'));
    H.layers.forEach(id => onLayer(id, 'hov'));
    H.relNodes.forEach(id => onNode(id, 'rel'));
    H.relEdges.forEach(id => onEdge(id, 'rel'));
    for (const { l, g, t } of H.tokens) {   // a hovered box shows its label too
      const b = toks.get(store.net.layers[l]?.id)?.boxes.find(x => x.g === g && x.t === t);
      mark(b?.el, 'hov');
      mark(b?.lab, 'hov');
    }
    if (H.bias) H.bias.forEach((id, k) => { onNode(id, 'bias'); if (k) onNode(id, 'rel'); });
    if (A.node) {
      A.nodes.forEach(id => onNode(id, 'lit'));
      A.edges.forEach(id => onEdge(id, 'lit'));
      A.rel.forEach(id => onNode(id, 'rel'));
    }
    focus = attFoci(A);
    highlightAtt(focus, mark, onNode);
    // The hovered / selected edge shows its numbers even with W off; the rest of its tie group its value.
    const shows = [...S.edges, ...H.edges].map(id => [id, true]).concat([...S.ties, ...H.ties].map(id => [id, 'tie']));
    for (const [id, mode] of shows) {
      const r = edges.get(id), e = I.edgeById.get(id);
      if (!r || !e || r.lab.__show) continue;
      mark(r.lab, 'show');
      r.lab.__show = mode;
      paintLabel(r, e, mode);
    }
    // Labels that just lost 'show' drop the tie name (with W on they stay, as plain numbers).
    if (showW) for (const e of store.net.edges) { const r = edges.get(e.id); if (r && !r.lab.__show) paintLabel(r, e, false); }
    svg.classList.toggle('focus', H.any || !!A.node || !!focus[0]?.hover);
    pairIds = H.pair;
    layoutPair();
    const key = A.node ? `${A.dir}|${A.node}|${A.edges.join()}|${A.phase}` : '';
    if (key !== pulseKey) {
      pulseKey = key;
      gPulses.textContent = '';
      for (const id of A.edges) {
        const d = pathOf(id);
        if (d) mk('path', { class: `nnv-pulse ${A.dir}`, d, pathLength: 100, 'data-id': id }, gPulses);
      }
    }
  }

  // ---------------------------------------------------------------- frame scheduling
  // Covered: the Flow or 3D view hides the SVG (visibility: hidden), so nothing on it can be seen.
  // What training brings every frame (HOLD: the values' paint and the meta check) waits until it
  // shows again (the style observer below). Structure, layout, hover, the lens and a fit go on as
  // before, and contentBox() (fit, outOfView, contentRect, png) flushes in full: they measure or
  // copy the SVG.
  const covered = () => svg.style.visibility === 'hidden';
  const HOLD = ['paint', 'meta'], QUIET_MS = 300;
  let held = null, quietT = 0;   // HOLD flags put off under the cover
  function schedule() {
    if (!raf && shown) raf = requestAnimationFrame(() => { raf = 0; flush(); });
  }
  function flush(force = false) {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    let d = dirty;
    dirty = {};
    if (!force && covered()) {
      for (const k of HOLD) if (d[k]) { (held ||= {})[k] = true; delete d[k]; }
      if (held) {   // quiet a while (training paused): catch up under the cover, so it goes at once
        clearTimeout(quietT);
        quietT = setTimeout(() => { if (held && shown && covered() && !ctx.train?.running) flush(true); }, QUIET_MS);
      }
    } else if (held) {
      // The catch-up shows at once, as the canvas painted under the cover did: without nnv-snap every
      // edge and neuron would start its fill and stroke transitions together.
      d = { ...held, ...d };
      held = null;
      snap();
    }
    if (d.build) rebuild();
    const reshaped = (d.build || d.meta) && syncTokens();
    const retitled = (d.build || d.meta) && syncMeta();
    const moved = d.build || d.layout || reshaped || retitled || (d.meta && posChanged());
    if (moved) layoutAll();
    if (d.build || d.paint || reshaped) paint();
    else if (d.lens) paintLens();
    if (d.build || d.hl || d.lens || reshaped) highlight();
    if (audience && moved && shown && needFit === false && outOfView()) needFit = 300;
    if (needFit !== false && shown && stage.clientWidth && stage.clientHeight) fit(everFit ? needFit : 0);
  }
  const invalidate = (...keys) => { for (const k of keys) dirty[k] = true; schedule(); };
  // No transitions (view.css zeroes --dur-1 and --dur-2 under .nnv-snap) until two frames have drawn.
  function snap() {
    svg.classList.add('nnv-snap');
    requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.remove('nnv-snap')));
  }

  store.on('net', p => { stale = true; invalidate(p?.structural ? 'build' : 'meta'); });
  store.on('values', () => invalidate('paint'));
  store.on('layout', () => { stale = true; invalidate('layout'); });
  for (const k of ['sel', 'hover', 'anim']) store.on(k, () => invalidate('hl'));
  store.on('lens', () => invalidate('lens'));
  ctx.onTheme?.(() => invalidate('paint'));
  ctx.onShow?.(v => {
    shown = v;
    if (!v) return;
    for (const r of layers.values()) r.measured = false;
    invalidate('layout', 'paint', 'hl');
  });
  // Uncovered (the Flow or 3D view closed): catch up on what piled up meanwhile, before the next
  // frame draws (this runs as a microtask), so the canvas never shows a stale frame.
  if (typeof MutationObserver === 'function') {
    new MutationObserver(() => { if (held && shown && !covered()) flush(); })
      .observe(svg, { attributes: true, attributeFilter: ['style'] });
  }
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => {
      if (!stage.clientWidth || !stage.clientHeight) return;
      if (!userMoved || audience) needFit = 0;   // follow the resize without easing
      invalidate('layout');
    }).observe(stage);
    // A floating panel (Train, Attention) opening, folding, resizing, moving or closing changes
    // fitArea(): refit, unless the user has panned or zoomed since the last fit.
    let areaKey = '';
    const areaChanged = () => {
      if (!stage.clientWidth || !stage.clientHeight) return;
      const F = floats(), key = [F.w, F.h, ...F.rects.flatMap(r => [r.x0, r.y0, r.x1, r.y1])].map(Math.round).join(',');
      if (key === areaKey) return;
      const first = !areaKey;
      areaKey = key;
      if (!first && !userMoved) { needFit = 300; invalidate('layout'); }
    };
    const panels = new ResizeObserver(areaChanged);
    const watch = () => { for (const c of stage.children) if (c !== svg) panels.observe(c); };
    new MutationObserver(watch).observe(stage, { childList: true });
    watch();
    // Moving a panel changes it too: a drag ends with a pointerup on the panel's header.
    stage.addEventListener('pointerup', () => requestAnimationFrame(areaChanged), true);
  }

  // ---------------------------------------------------------------- view transform
  function applyView() {
    const t = `translate(${r1(V.x)},${r1(V.y)}) scale(${V.k.toFixed(4)})`;
    world.setAttribute('transform', t);
    pat.setAttribute('patternTransform', t);
    placeHint();
    const ts = Math.round(clamp(TEXT_K / V.k, 1, TEXT_MAX) * 20) / 20;
    if (ts === textScale) return;
    textScale = ts;
    svg.style.setProperty('--nnv-ts', ts);
    svg.classList.toggle('far', ts > FAR_TS);   // zoomed out: no dot grid
    fitHeads();
  }
  function moveTo(k, x, y, ms = 0) {
    cancelAnimationFrame(fitAnim);
    if (!ms || !shown) { Object.assign(V, { k, x, y }); applyView(); return; }
    const k0 = V.k, x0 = V.x, y0 = V.y, t0 = performance.now();
    const step = now => {
      const u = Math.min(1, (now - t0) / ms), e = u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
      V.k = k0 + (k - k0) * e; V.x = x0 + (x - x0) * e; V.y = y0 + (y - y0) * e;
      applyView();
      if (u < 1) fitAnim = requestAnimationFrame(step);
    };
    fitAnim = requestAnimationFrame(step);
  }
  const toWorld = p => ({ x: (p.x - V.x) / V.k, y: (p.y - V.y) / V.k });
  function worldToScreen(x, y) {
    const s = stage.getBoundingClientRect(), v = svg.getBoundingClientRect();
    return { x: x * V.k + V.x + v.left - s.left - stage.clientLeft, y: y * V.k + V.y + v.top - s.top - stage.clientTop };
  }
  function nodeRect(id) {
    const n = model.node(store.net, id);
    if (!n) return null;
    const c = worldToScreen(n.x, n.y), r = R * V.k;
    return { x: c.x - r, y: c.y - r, w: 2 * r, h: 2 * r };
  }

  // Content bounds in world px: measured when rendered, estimated from the net otherwise.
  function contentBox() {
    flush(true);
    try {
      const b = content.getBBox();
      if (b.width > 0 && b.height > 0) return { x: b.x, y: b.y, w: b.width, h: b.height };
    } catch { /* not rendered */ }
    const I = ix();
    const xs = I.cols.flatMap(c => [c.minX, c.maxX]);
    const x0 = Math.min(...xs, 0) - R - 60, x1 = Math.max(...xs, 0) + R + 60;
    const y0 = (I.box ? I.box.minY : I.cy) - HEAD_UP - 24, y1 = (I.box ? I.box.maxY : I.cy) + LANE_DOWN + 20;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }
  // The part of the stage the net should fit into: above the lens bar (lens.js) at the bottom and
  // clear of the floating panels (FLOATS: Train, Attention) wherever they sit. Of the free
  // rectangles between the panels it is the one where content box b comes out largest, or nearly
  // (FIT_SLACK) and roomier; without b, the largest one. When none is at least FREE_MIN each way
  // (a small window, panels over most of it), the panels are ignored.
  function floats() {
    const s = stage.getBoundingClientRect(), rects = [];
    let bottom = s.height;
    for (const c of stage.children) {
      if (c === svg) continue;
      const r = c.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if (c.classList.contains('nn-lens')) {
        if (r.bottom >= s.bottom - 40) bottom = Math.min(bottom, r.top - s.top - 4);
      } else if (c.matches(FLOATS)) {
        rects.push({ x0: r.left - s.left - FLOAT_GAP, y0: r.top - s.top - FLOAT_GAP, x1: r.right - s.left + FLOAT_GAP, y1: r.bottom - s.top + FLOAT_GAP });
      }
    }
    return { w: s.width, h: Math.max(s.height * 0.5, bottom), rects };
  }
  const fitPad = A => clamp(Math.min(A.w, A.h) * 0.05, 12, 40);
  const fitScale = (A, b) => Math.min((A.w - 2 * fitPad(A)) / b.w, (A.h - 2 * fitPad(A)) / b.h);
  function fitArea(b = null) {
    const F = floats(), all = { x: 0, y: 0, w: F.w, h: F.h };
    const obs = F.rects.filter(r => r.x1 > 0 && r.x0 < F.w && r.y1 > 0 && r.y0 < F.h);
    if (!obs.length) return all;
    // A largest free rectangle has each side on the stage's edge or on a panel's.
    const cuts = (lo, hi, k0, k1) => [...new Set([lo, hi, ...obs.flatMap(r => [r[k0], r[k1]]).filter(v => v > lo && v < hi)])].sort((a, c) => a - c);
    const xs = cuts(0, F.w, 'x0', 'x1'), ys = cuts(0, F.h, 'y0', 'y1');
    const free = [];
    for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) {
      if (xs[j] - xs[i] < FREE_MIN) continue;
      for (let p = 0; p < ys.length; p++) for (let q = p + 1; q < ys.length; q++) {
        const A = { x: xs[i], y: ys[p], w: xs[j] - xs[i], h: ys[q] - ys[p] };
        if (A.h < FREE_MIN || obs.some(r => r.x0 < A.x + A.w && r.x1 > A.x && r.y0 < A.y + A.h && r.y1 > A.y)) continue;
        free.push({ A, k: b ? Math.min(fitScale(A, b), FIT_MAX_K) : 0 });
      }
    }
    if (!free.length) return all;
    // Within FIT_SLACK of the largest scale the larger area wins, so a folded panel's header doesn't
    // push the net into a corner for a few percent of size.
    const kMax = Math.max(...free.map(c => c.k));
    return free.filter(c => c.k >= kMax * FIT_SLACK - 1e-9).reduce((a, c) => (c.A.w * c.A.h > a.A.w * a.A.h ? c : a)).A;
  }
  function fit(ms = 0) {
    if (!stage.clientWidth || !stage.clientHeight) { needFit = 0; return; }
    needFit = false;                     // before contentBox(): it flushes, and a flush may fit
    userMoved = false;
    everFit = true;
    const b = contentBox(), A = fitArea(b);
    const k = clamp(fitScale(A, b), MIN_K, FIT_MAX_K);
    moveTo(k, A.x + A.w / 2 - (b.x + b.w / 2) * k, A.y + A.h / 2 - (b.y + b.h / 2) * k, ms);
  }
  // Stage px box of the whole net (lanes and headers included), e.g. to place a card beside it.
  function contentRect() {
    const b = contentBox(), o = worldToScreen(b.x, b.y);
    return { x: o.x, y: o.y, w: b.w * V.k, h: b.h * V.k };
  }
  // Is any of the net outside the stage, or under a panel that fit() avoids?
  function outOfView() {
    if (!stage.clientWidth || !stage.clientHeight) return false;
    const b = contentBox(), A = fitArea(b);
    const x0 = b.x * V.k + V.x, y0 = b.y * V.k + V.y;
    return x0 < A.x - 2 || y0 < A.y - 2 || x0 + b.w * V.k > A.x + A.w + 2 || y0 + b.h * V.k > A.y + A.h + 2;
  }

  // ---------------------------------------------------------------- node images (train heatmaps)
  function applyImage(r, url) {
    if (!url) {
      r.img?.remove();
      r.img = null;
      r.g.classList.remove('img');
      return;
    }
    if (!r.img) {
      r.img = mk('image', { class: 'nnv-img', x: -R, y: -R, width: 2 * R, height: 2 * R,
        preserveAspectRatio: 'xMidYMid slice', 'clip-path': 'url(#nnv-clip)' });
      r.fill.after(r.img);
      r.g.classList.add('img');
    }
    put(r.img, 'href', url);
  }
  function setNodeImage(id, url) {
    if (url) images.set(id, url);
    else images.delete(id);
    const r = nodes.get(id);
    if (r) applyImage(r, url || null);
  }

  // ---------------------------------------------------------------- png
  async function png(scale = 2) {
    const b = contentBox(), pad = 24;
    const W = Math.ceil(b.w + 2 * pad), H = Math.ceil(b.h + 2 * pad);
    const cs = getComputedStyle(svg);
    const clone = svg.cloneNode(true);
    for (const e of clone.querySelectorAll('.nnv-grid, pattern, .nnv-pulses, .nnv-ghost, .nnv-pair, .nnv-handle, .nnv-hint, .nnv-hm-hl')) e.remove();
    for (const c of MARKS) for (const e of clone.querySelectorAll(`.${c}`)) e.classList.remove(c);
    clone.classList.remove('focus', 'panning', 'wiring', 'dragging', 'far');
    const fit1 = headFit(1);   // headers drawn at zoom 1, still fitted between neighbours
    for (const h of clone.querySelectorAll('.nnv-head')) {
      const r = layers.get(h.dataset.id), f = r && fit1.get(r.id);
      if (!f) continue;
      const q = s => h.querySelector(s);
      placeHead(r, f, { g: h, gi: q('.nnv-head-in'), bg: q('.nnv-head-bg'), name: q('.nnv-head-name'), sub: q('.nnv-head-sub'), hm: q('.nnv-hm') });
    }
    clone.querySelector('.nnv-world').setAttribute('transform', `translate(${r1(pad - b.x)},${r1(pad - b.y)})`);
    clone.setAttribute('xmlns', NS);
    clone.setAttribute('width', W);
    clone.setAttribute('height', H);
    clone.setAttribute('viewBox', `0 0 ${W} ${H}`);
    // The standalone image has neither the page's variables nor its theme selector: resolve them here.
    clone.setAttribute('style', [...VARS.map(v => `${v}:${cs.getPropertyValue(v).trim()}`), '--nnv-ts:1',
      `font-family:${cs.fontFamily}`, `width:${W}px`, `height:${H}px`].join(';'));
    const [kcss, vcss] = await Promise.all([katexCss().catch(() => ''), viewCss().catch(() => '')]);
    const style = document.createElementNS(NS, 'style');
    style.textContent = `${kcss}\n${vcss}`;
    clone.insertBefore(mk('rect', { width: W, height: H, fill: cs.getPropertyValue('--nnv-bg').trim() || '#1d2327' }), clone.firstChild);
    clone.insertBefore(style, clone.firstChild);
    await Promise.all([...clone.querySelectorAll('image')].map(async im => {
      const href = im.getAttribute('href');
      if (!href || href.startsWith('data:')) return;
      try { im.setAttribute('href', await readAsDataUrl(await (await fetch(href)).blob())); } catch { im.remove(); }
    }));
    const img = new Image();
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`;
    await img.decode();
    const cv = document.createElement('canvas');
    cv.width = Math.max(1, Math.round(W * scale));
    cv.height = Math.max(1, Math.round(H * scale));
    cv.getContext('2d').drawImage(img, 0, 0, cv.width, cv.height);
    return cv.toDataURL('image/png');
  }

  // ---------------------------------------------------------------- weight labels (W)
  let wBtn = null;
  function toggleW(on = !showW) {
    showW = on;
    svg.classList.toggle('show-w', on);
    wBtn?.classList.toggle('on', on);
    invalidate('paint');
  }

  ctx.view = {
    svg, worldToScreen, nodeRect, fit, setNodeImage, png, outOfView, contentRect,
    screenToWorld: (x, y) => toWorld({ x, y }),
    get weights() { return showW; },
    set weights(on) { toggleW(!!on); },
  };

  dirty = { build: true };
  flush();
  applyView();
  if (audience) return;

  // ================================================================ editing (presenter only)
  wBtn = ctx.addButton?.({ label: 'Weights', icon: '&#8649;', title: 'Numbers on the edges: weights, and ∂L/∂w once targets are set (W)', onClick: () => toggleW(), group: 'view' }) || null;
  window.addEventListener('keydown', e => {
    if (e.key !== 'w' && e.key !== 'W') return;
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || !(ctx.active ? ctx.active(e) : true)) return;
    e.preventDefault();
    toggleW();
  });

  const pt = e => { const r = svg.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  const hitOf = t => (t && svg.contains(t) ? t.closest?.('[data-kind]') || null : null);
  const layerNodes = id => store.net.nodes.filter(n => n.layer === id);
  function select(t) {
    const cur = store.state.sel;
    if ((cur?.kind ?? null) === (t?.kind ?? null) && cur?.id === t?.id) return;
    store.set('sel', t);
  }

  // Hover: set on entering a node / edge / header; on leaving, clear it only if it is still ours.
  // A token box, heatmap cell or attention edge hovers a token, { kind: 'token', layer, t, g?, h? }:
  // a box its token (g: its group), a cell or attention edge A_ij's row, token i of head h.
  let hoverKey = '', myHover = null;
  const TOKEN = new Set(['token', 'attcell', 'attedge']);
  function tokenHover(h) {
    const I = ix(), l = I.li.get(h.dataset.id), ds = h.dataset;
    if (l === undefined) return null;
    if (ds.kind === 'token') return { kind: 'token', layer: l, t: +ds.t, ...(I.tok[l]?.groups ? { g: +ds.g } : {}) };
    return { kind: 'token', layer: l, t: +ds.i, ...(I.att[l]?.heads > 1 ? { h: +ds.h } : {}) };
  }
  function hoverFrom(target) {
    const h = hitOf(target), kind = h?.dataset.kind === 'handle' ? 'node' : h?.dataset.kind, ds = h?.dataset;
    const key = h ? `${kind}:${ds.id}:${ds.t ?? ds.i ?? ''}:${ds.g ?? ds.h ?? ''}` : '';
    if (key === hoverKey) return;
    hoverKey = key;
    const spec = !h ? null : TOKEN.has(kind) ? tokenHover(h) : { kind, id: ds.id };
    if (spec) { myHover = spec; store.set('hover', spec); }
    else if (myHover && store.state.hover === myHover) { myHover = null; store.set('hover', null); }
  }

  function nearest(w, exclude) {
    let best = null, bd = R + 10;
    for (const n of store.net.nodes) {
      if (n.id === exclude) continue;
      const d = Math.hypot(n.x - w.x, n.y - w.y);
      if (d < bd) { bd = d; best = n; }
    }
    return best;
  }
  function setDrop(id) {
    if (id === dropId) return;
    if (dropId) nodes.get(dropId)?.g.classList.remove('drop');
    dropId = id;
    if (id) nodes.get(id)?.g.classList.add('drop');
  }
  // What a wire from a to b would do: { why } refuses; { tie, pairs } adds one shared entry of a
  // tokenwise tied matrix, for every token; else a plain edge. Between the same two groups of two
  // token layers, the existing (non-fixed) edges decide: all tied to one matrix N, token t to token t.
  const texName = name => { const [b, s] = tiePlain(name); return s ? `${b}_${s}` : b; };
  function wirePlan(a, b) {
    const I = ix();
    let la = I.li.get(a.layer), lb = I.li.get(b.layer);
    if (la === lb) return { why: 'Edges run between layers, not within one' };
    if (la > lb) { [a, b] = [b, a]; [la, lb] = [lb, la]; }
    if (I.tok[lb]?.att) return { why: 'Attention has no weights coming in: Z = softmax(QKᵀ)V is computed from the Q, K and V before it' };
    const pa = tokPos(a.id), pb = tokPos(b.id);
    if (!pa || !pb || I.tok[la].T !== I.tok[lb].T || I.tok[la].T < 2) return {};
    let name = null, conv = 'both';
    for (const e of store.net.edges) {
      const p = tokPos(e.from), q = tokPos(e.to);
      if (!p || !q || e.fixed || p.l !== la || q.l !== lb || p.g !== pa.g || q.g !== pb.g) continue;
      const tp = tieParse(e.tie);
      if (!tp || p.t !== q.t || (name !== null && tp.name !== name)) return {};
      name = tp.name;
      // X W convention (Q = X W_Q): (i, j) = (sending, receiving) feature. Accept the transpose too.
      const xw = tp.i === p.f + 1 && tp.j === q.f + 1, rw = tp.i === q.f + 1 && tp.j === p.f + 1;
      conv = conv === 'both' ? (xw && rw ? 'both' : xw ? 'xw' : rw ? 'rw' : '') : conv === 'xw' ? (xw ? 'xw' : '') : rw ? 'rw' : '';
      if (!conv) return {};
    }
    if (name === null) return {};
    if (pa.t !== pb.t) return { why: `Only attention mixes tokens: ${texName(name)} acts on each token by itself` };
    const [i, j] = conv === 'rw' ? [pb.f + 1, pa.f + 1] : [pa.f + 1, pb.f + 1];
    const pairs = [];
    for (let t = 0; t < I.tok[la].T; t++) {
      const s = tokNode(la, pa.g, t, pa.f), d = tokNode(lb, pb.g, t, pb.f);
      if (s && d) pairs.push([s.id, d.id]);
    }
    return { tie: `${name}:${i},${j}`, label: `${texName(name)}(${i},${j})`, pairs, T: I.tok[la].T };
  }
  let planFor = null;   // the last wirePlan, while dragging a wire: { key, P }
  function wireMove(d, w) {
    const I = ix(), src = I.nodeById.get(d.from);
    if (!src) return;
    const tgt = nearest(w, d.from), key = tgt ? `${d.from}>${tgt.id}` : '';
    if (tgt && planFor?.key !== key) planFor = { key, P: wirePlan(src, tgt) };
    const ok = !!tgt && !planFor.P.why;
    put(ghost, 'd', (tgt ? straight(src, tgt) : straight(src, w, 0)).d);
    put(ghost, 'display', null);
    ghost.classList.toggle('bad', !!tgt && !ok);
    setDrop(ok ? tgt.id : null);
  }
  function wireEnd(d, w) {
    put(ghost, 'display', 'none');
    ghost.classList.remove('bad');
    setDrop(null);
    planFor = null;
    const net = store.net, I = ix(), src = I.nodeById.get(d.from), tgt = w && nearest(w, d.from);
    if (!src || !tgt) return;
    const P = wirePlan(src, tgt);
    if (P.why) { ctx.toast?.(P.why); return; }
    const old = model.edgeBetween(net, src.id, tgt.id);
    if (old) { select({ kind: 'edge', id: old.id }); ctx.toast?.('Those two are already connected'); return; }
    const la = I.li.get(src.layer), lb = I.li.get(tgt.layer);
    const lim = Math.sqrt(6 / (I.byLayer[la].length + I.byLayer[lb].length));
    const w0 = (Math.random() < 0.5 ? -1 : 1) * lim * (0.25 + 0.75 * Math.random());
    if (P.tie) {
      // One new shared parameter: the same entry of the matrix, for every token.
      const have = I.ties.get(P.tie);
      const w1 = have?.length ? I.edgeById.get(have[0]).w : Math.round(w0 * 100) / 100;
      for (const [f, t] of P.pairs) {
        if (model.edgeBetween(net, f, t)) continue;
        const id = model.connect(net, f, t, w1);
        const e = id && model.edge(net, id);
        if (e) e.tie = P.tie;
      }
      const mine = model.edgeBetween(net, src.id, tgt.id);
      if (!mine) return;
      store.commit('Connect');
      select({ kind: 'edge', id: mine.id });
      ctx.toast?.(`New shared weight ${P.label}: one edge per token (${P.pairs.length})`);
      return;
    }
    const id = model.connect(net, src.id, tgt.id, Math.round(w0 * 100) / 100);
    if (!id) return;
    store.commit('Connect');
    select({ kind: 'edge', id });
  }

  // Double-click on empty space.
  function addAt(p) {
    const net = store.net, I = ix(), L = net.layers.length;
    if (!L) return;
    const cols = I.cols.map((c, l) => ({ l, x: c.x })).sort((a, b) => a.x - b.x);
    let at = -1;
    if (I.full && L) {
      const lo = cols[0], hi = cols[cols.length - 1];
      if (p.x < lo.x - BEYOND && lo.l === 0) at = 0;
      else if (p.x > hi.x + BEYOND && hi.l === L - 1) at = L;
      else {
        for (let i = 0; i + 1 < cols.length; i++) {
          const a = cols[i], b = cols[i + 1], gap = b.x - a.x;
          if (p.x <= a.x || p.x >= b.x || Math.abs(a.l - b.l) !== 1) continue;
          if (Math.min(p.x - a.x, b.x - p.x) > Math.max(0.3 * gap, R + 12)) at = Math.max(a.l, b.l);
          break;
        }
      }
    }
    if (at >= 0) insertLayer(at, p);
    else {
      const near = cols.reduce((m, c) => (Math.abs(c.x - p.x) < Math.abs(m.x - p.x) ? c : m));
      addNeuron(near.l, p);
    }
  }
  function addNeuron(l, p) {
    const net = store.net, I = ix(), c = I.cols[l], layer = net.layers[l];
    if (I.tok[l]) { addFeature(l, p); return; }
    const x = c.n && Math.abs(p.x - c.x) < R + 16 ? c.x : p.x;
    const index = I.byLayer[l].filter(n => n.y < p.y).length;   // rows follow the picture, top to bottom
    const id = model.addNode(net, layer.id, { x, y: p.y, index, connect: true, seed: (Math.random() * 2 ** 31) | 0 });
    if (!id) return;
    store.commit('Add neuron');
    select({ kind: 'node', id });
  }
  // A token layer grows by a whole feature: one neuron per token, each wired like the token's last
  // feature (same neighbours, fresh weights; a tied entry N(i, d) gets a new shared N(i, d + 1)).
  // Attention, Q/K/V and residual-stream layers keep their width, and say why.
  function addFeature(l, p) {
    const net = store.net, I = ix(), s = I.tok[l], lay = net.layers[l], nm = lay.name || 'This layer';
    const no = msg => { ctx.toast?.(msg, 4500); };
    if (!s.d) return no(`${nm} doesn't split into ${s.T} tokens evenly: fix its size first`);
    if (s.att) return no('Attention has no weights of its own: Z is tokens × d_v, one row per token of V');
    if (s.groups) return no(`${s.groups.map(texName).join(', ')} share one shape (tokens × d each), so they can't grow one neuron at a time`);
    if (I.att[l + 1]) return no('Attention reads this layer as Q, K and V: its shape is fixed');
    const T = s.T, d = s.d, ns = I.byLayer[l];
    const plan = [], conv = new Map();   // per token: the template's edges; per tie name: its (i, j) convention
    for (let t = 0; t < T; t++) {
      const tn = ns[t * d + d - 1], list = [];
      for (const e of net.edges) {
        if (e.to !== tn.id && e.from !== tn.id) continue;
        if (e.fixed) return no(`${nm} is on a fixed (residual) path, so its width stays d = ${d}`);
        const inc = e.to === tn.id, tp = typeof e.tie === 'string' && e.tie ? tieParse(e.tie) : null;
        if (e.tie && !tp) return no(`The shared weight ${e.tie} can't grow a new feature`);
        if (tp) {
          const xw = inc ? tp.j === d : tp.i === d, rw = inc ? tp.i === d : tp.j === d, was = conv.get(tp.name) ?? 'both';
          const now = was === 'both' ? (xw && rw ? 'both' : xw ? 'xw' : rw ? 'rw' : '') : was === 'xw' ? (xw ? 'xw' : '') : rw ? 'rw' : '';
          if (!now) return no(`The shared weight ${texName(tp.name)} can't grow a new feature`);
          conv.set(tp.name, now);
        }
        list.push({ other: inc ? e.from : e.to, inc, tp });
      }
      plan.push(list);
    }
    const newTie = ({ inc, tp }) => {
      const rw = conv.get(tp.name) === 'rw';
      return inc === !rw ? `${tp.name}:${tp.i},${d + 1}` : `${tp.name}:${d + 1},${tp.j}`;
    };
    // Where: one step past the token's last feature (the step between its last two, or a row down).
    // Tokens stacked along that step move apart to make room, then the layer is recentred.
    const step = d >= 2 ? { x: ns[d - 1].x - ns[d - 2].x, y: ns[d - 1].y - ns[d - 2].y } : { x: 0, y: ROW };
    const along = T > 1 && (ns[d].x - ns[0].x) * step.x + (ns[d].y - ns[0].y) * step.y > 0;
    const back = along ? T / 2 : 0.5;
    const at = (n, k) => ({ x: r1(n.x + (k - back) * step.x), y: r1(n.y + (k - back) * step.y) });
    const moves = ns.map((n, k) => [n.id, at(n, along ? Math.floor(k / d) : 0)]);
    const nIn = plan[0].filter(x => x.inc).length, nOut = plan[0].length - nIn;
    const rand = () => {
      const lim = Math.sqrt(6 / Math.max(2, nIn + nOut + 1));
      return Math.round((Math.random() < 0.5 ? -1 : 1) * lim * (0.25 + 0.75 * Math.random()) * 100) / 100;
    };
    // A shared bias (node.tie 'b:d', one per feature) gets a new entry too.
    const bt = /^(.*):(\d+)$/.exec(typeof ns[d - 1].tie === 'string' ? ns[d - 1].tie : '');
    const biasTie = bt && +bt[2] === d ? `${bt[1]}:${d + 1}` : null;
    const tieW = new Map(), added = [];
    for (let t = 0; t < T; t++) {
      const tn = ns[t * d + d - 1], pos = at(tn, (along ? t : 0) + 1);
      // Midway the layer doesn't split into its tokens, so model.addNode makes it a plain vector:
      // the tokens come back below, once every token has its new neuron.
      const id = model.addNode(net, lay.id, { index: t * (d + 1) + d, x: pos.x, y: pos.y, label: nextLabel(tn, d) });
      if (!id) continue;
      added.push(id);
      const nn = model.node(net, id);
      if (biasTie && nn) nn.tie = biasTie;
      for (const x of plan[t]) {
        const tie = x.tp ? newTie(x) : null;
        if (tie && !tieW.has(tie)) tieW.set(tie, rand());
        const eid = model.connect(net, x.inc ? x.other : id, x.inc ? id : x.other, tie ? tieW.get(tie) : rand());
        const e = eid && model.edge(net, eid);
        if (e && tie) e.tie = tie;
      }
    }
    for (const [id, q] of moves) model.setNode(net, id, q);
    if (added.length === T) lay.tokens = T;
    store.commit('Add feature');
    const near = added.map(id => model.node(net, id)).filter(Boolean)
      .reduce((m, n) => (!m || Math.hypot(n.x - p.x, n.y - p.y) < Math.hypot(m.x - p.x, m.y - p.y) ? n : m), null);
    if (near) select({ kind: 'node', id: near.id });
    ctx.toast?.(`${nm}: a new feature in every token (d = ${d} → ${d + 1})${tieW.size ? `, with ${tieW.size} new shared weights` : ''}`);
    requestAnimationFrame(() => { if (outOfView()) fit(300); });
  }
  // A custom label that ends in the feature number (x_{2,3}) counts on; default labels renumber themselves.
  function nextLabel(n, d) {
    if (!n.label || model.defaultLabel?.(store.net, n.id) === n.label) return undefined;
    const m = /^(.*?)(\d+)(\D*)$/.exec(n.label);
    return m && +m[2] === d ? `${m[1]}${d + 1}${m[3]}` : undefined;
  }
  // As the toolbar's "+ Layer": the direct edges between the two neighbours would become skip
  // edges, so they go; a new hidden layer copies the activation of the hidden layer before it.
  // The new column keeps GAP from its neighbours: later columns move right to make room.
  function insertLayer(at, p) {
    const net = store.net, L = net.layers.length, hidden = at > 0 && at < L, before = net.layers[at - 1];
    // Next to token layers a new (plain, dense) layer would cut their shared weights and residuals,
    // and before an attention layer it would leave attention without its Q, K, V.
    const I = ix();
    if (I.tok[at]?.att) { ctx.toast?.('Attention reads Q, K and V straight from the layer before it: nothing goes between them', 4500); return; }
    if (I.tok[at - 1] || I.tok[at]) { ctx.toast?.('Between token layers a plain dense layer would cut their shared (tied) weights, so none is added here', 4500); return; }
    const meanX = l => { const ns = l ? layerNodes(l.id) : []; return ns.length ? ns.reduce((s, n) => s + n.x, 0) / ns.length : null; };
    const px = meanX(net.layers[at - 1]), nx = meanX(net.layers[at]);
    let x = p.x;
    if (px !== null) x = Math.max(x, px + GAP);
    else if (nx !== null) x = Math.min(x, nx - GAP);
    const shift = px !== null && nx !== null ? Math.max(0, x + GAP - nx) : 0;
    if (hidden) {
      const a = new Set(layerNodes(net.layers[at - 1].id).map(n => n.id)), b = new Set(layerNodes(net.layers[at].id).map(n => n.id));
      for (const e of net.edges.filter(e => a.has(e.from) && b.has(e.to))) model.disconnect(net, e.id);
    }
    const act = hidden && at > 1 && before.act !== 'softmax' ? before.act : undefined;
    const outer = at === 0 ? net.layers[0] : at === L ? net.layers[L - 1] : null;
    const later = shift ? net.layers.slice(at).map(l => l.id) : [];
    const id = model.addLayer(net, at, { act, size: 1, dense: true, seed: (Math.random() * 2 ** 31) | 0 });
    // The old first / last layer is hidden now; don't leave it called "Input" / "Output".
    if (outer && /^(input|output)$/i.test(outer.name || '')) model.setLayer(net, outer.id, { name: 'Hidden' });
    for (const lid of later) for (const m of layerNodes(lid)) model.setNode(net, m.id, { x: r1(m.x + shift) });
    const [n] = layerNodes(id);
    if (n) model.setNode(net, n.id, { x: r1(x), y: p.y });
    store.commit('Insert layer');
    select({ kind: 'layer', id });
    if (shift || x !== p.x) requestAnimationFrame(() => { if (outOfView()) fit(300); });
  }

  // ---------------------------------------------------------------- pointer
  let drag = null;
  svg.addEventListener('pointerdown', e => {
    if (drag || (e.button !== 0 && e.button !== 1)) return;
    const h = e.button === 0 ? hitOf(e.target) : null, kind = h?.dataset.kind, id = h?.dataset.id;
    const base = { pid: e.pointerId, p0: pt(e), moved: false };
    const n = kind === 'node' ? model.node(store.net, id) : null;
    if (kind === 'handle') drag = { ...base, type: 'wire', from: id };
    else if (n) {
      const peers = layerNodes(n.layer).filter(m => m.id !== id);
      drag = { ...base, type: 'node', id, ox: n.x, oy: n.y, colX: peers.length ? peers.reduce((s, m) => s + m.x, 0) / peers.length : null };
    } else if (kind === 'layer' || kind === 'attcell') drag = { ...base, type: 'layer', id, start: layerNodes(id).map(m => [m.id, m.x, m.y]) };
    else if (kind === 'token') {
      // A token box drags its token's neurons (of its group); a click selects the layer.
      const s = ix().tok[ix().li.get(id)], g = +h.dataset.g, t = +h.dataset.t;
      const mine = s?.d ? layerNodes(id).slice((g * s.T + t) * s.d, (g * s.T + t + 1) * s.d) : [];
      drag = { ...base, type: 'layer', id, token: true, start: mine.map(m => [m.id, m.x, m.y]) };
    } else drag = { ...base, type: 'pan', x0: V.x, y0: V.y, edge: kind === 'edge' ? id : null, click: e.button === 0 };
  });
  svg.addEventListener('pointermove', e => {
    if (!drag) { hoverFrom(e.target); return; }
    if (e.pointerId !== drag.pid) return;
    const p = pt(e), dx = p.x - drag.p0.x, dy = p.y - drag.p0.y;
    if (!drag.moved) {
      if (Math.hypot(dx, dy) < 4) return;
      drag.moved = true;
      try { svg.setPointerCapture(e.pointerId); } catch { /* pointer gone */ }
      svg.classList.add(drag.type === 'pan' ? 'panning' : drag.type === 'wire' ? 'wiring' : 'dragging');
      cancelAnimationFrame(fitAnim);
    }
    if (drag.type === 'pan') {
      V.x = drag.x0 + dx; V.y = drag.y0 + dy;
      userMoved = true;
      applyView();
    } else if (drag.type === 'node') {
      const n = model.node(store.net, drag.id);
      if (!n) return;
      let x = drag.ox + dx / V.k;
      if (drag.colX !== null && !e.altKey && Math.abs(x - drag.colX) < SNAP) x = drag.colX;
      n.x = r1(x); n.y = r1(drag.oy + dy / V.k);
      store.layout();
    } else if (drag.type === 'layer') {
      for (const [id, x0, y0] of drag.start) {
        const n = model.node(store.net, id);
        if (n) { n.x = r1(x0 + dx / V.k); n.y = r1(y0 + dy / V.k); }
      }
      store.layout();
    } else if (drag.type === 'wire') wireMove(drag, toWorld(p));
  });
  const end = e => {
    if (!drag || e.pointerId !== drag.pid) return;
    const d = drag;
    drag = null;
    svg.classList.remove('panning', 'wiring', 'dragging');
    try { if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId); } catch { /* gone */ }
    const cancel = e.type === 'pointercancel';
    if (d.moved) {
      if (d.type === 'node') store.commit('Move neuron');
      else if (d.type === 'layer') store.commit(d.token ? 'Move token' : 'Move layer');
      else if (d.type === 'wire') wireEnd(d, cancel ? null : toWorld(pt(e)));
      return;
    }
    if (cancel) return;
    if (d.type === 'node') select({ kind: 'node', id: d.id });
    else if (d.type === 'wire') select({ kind: 'node', id: d.from });
    else if (d.type === 'layer') select({ kind: 'layer', id: d.id });
    else if (d.edge) select({ kind: 'edge', id: d.edge });
    else if (d.click) select(null);
  };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
  svg.addEventListener('pointerleave', () => { if (!drag) hoverFrom(null); });
  // Edges don't block it: between two dense columns there is hardly a spot that misses every edge.
  // Nor do token boxes: they fill their column.
  svg.addEventListener('dblclick', e => {
    const t = document.elementFromPoint(e.clientX, e.clientY), h = hitOf(t);
    if (!t || !svg.contains(t) || (h && !['edge', 'attedge', 'token'].includes(h.dataset.kind))) return;
    e.preventDefault();
    addAt(toWorld(pt(e)));
  });
  svg.addEventListener('wheel', e => {
    e.preventDefault();
    const p = pt(e), w = toWorld(p);
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
    const k = clamp(V.k * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0015)), MIN_K, MAX_K);
    cancelAnimationFrame(fitAnim);
    Object.assign(V, { k, x: p.x - w.x * k, y: p.y - w.y * k });
    userMoved = true;
    applyView();
  }, { passive: false });
  svg.addEventListener('contextmenu', e => { if (hitOf(e.target)) e.preventDefault(); });
}
