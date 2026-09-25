// Attention panel for the Net tab (docs/NN_LENS.md: state.viz), floating over #nn-stage like the
// Train panel. Four pictures of one attention layer, all drawn from store.state.fwd.attn:
//   arcs  queries on the left, keys on the right, a line per pair as thick as A_ij, a colour per
//         head (BertViz-style). Masked pairs have no line; hovering a token isolates its lines.
//   dots  why: the k_j arrows and the query q_i in the plane (a number line when d_h = 1, two
//         chosen dimensions when d_h > 2), each score q_i·k_j·scale and the softmax bars. The scale
//         slider edits the layer's scale live (touch while dragging, commit on release).
//   mix   what comes out: z_i = Σ_j A_ij v_j tip-to-tail from the scaled A_ij v_j, inside the shaded
//         convex hull of the v_j. "Send to 3D" writes the same construction into the 3D tab.
//   heat  S and A per head with numbers; masked cells hatched; row lens.token ringed.
//
// The panel is open exactly when state.viz = { mode, layer } is set, so tour.js and the audience
// mirror drive it through the store alone. Its query token and head are lens.token and lens.head
// (docs/NN_LENS.md); picking a query here sets lens.token. Dots and mix need one query: with no
// token followed they show the most decisive row (autoQuery, the row tour.js captions use) and
// head 1, with a dashed chip. Arcs follow focus.js's emphasis() and hide what the lens hides.
// Token names (net.meta.tokenNames) are renamed here: click the followed token's chip again, or
// double-click any token chip.
//
// ctx.attnviz = { show(mode?, { layer }?), hide(), toggle(mode?), cycle(dir), open, mode, layer,
//                 layers(), send3d(), rename(i, name), geometry(), el }. A: open / close, M: next mode.
// Panel UI state (not mirrored through the store): localStorage 'mathboard.nn.attnviz' = { x, y, w, mode }.

import { colorFor, HI } from './store.js';
import { lensOf } from './lens.js';
import { tokenNames, emphasis } from './focus.js';

const MODES = ['arcs', 'dots', 'mix', 'heat'];
const MODE_LABEL = { arcs: 'Arcs', dots: 'Dots', mix: 'Mix', heat: 'Heat' };
const MODE_TITLE = {
  arcs: 'Who reads whom: a line from query i to key j, as thick as A_ij (M: next mode)',
  dots: 'Why: each score is the dot product q_i · k_j times the scale, then softmax (M: next mode)',
  mix: 'What comes out: z_i = Σ_j A_ij v_j, a weighted average of the values (M: next mode)',
  heat: 'The numbers: S and A as heatmaps, per head (M: next mode)',
};
// How to read each picture: the help icon's title (the drawing keeps only notes about the state).
const MODE_HELP = {
  arcs: 'Arcs: who reads whom. A line from query i to key j, as thick as A_ij, one colour per head. Hover a token to see its lines alone with their weights; click it to follow it.',
  dots: 'Dots: why. Each score s_j is the dot product q · k_j times the scale, and softmax turns the scores into the row of A. A key\'s score is |q| times the length of its shadow on q\'s line (with one dimension, q times k). The scale sharpens A (larger) or flattens it (smaller).',
  mix: 'Mix: what comes out. z_i = Σ_j A_ij v_j, the values scaled by A and laid tip to tail. A\'s row sums to 1, so z lies in the shaded hull of the values. Send to 3D rebuilds it in the 3D tab, with a slider per weight.',
  heat: 'Heat: the numbers. S and A for each head. Rows are queries, columns are keys. Hover a cell for what it means; click a row to follow that query.',
};
const HELP_COMMON = 'The chips pick the query (and the head); double-click a query chip to rename its token. Drag the header to move the panel, double-click it to put it back. A: open / close, M: next mode.';
const UI_KEY = 'mathboard.nn.attnviz';
const W_DEFAULT = 400, W_MIN = 300;
const SCALE_SPAN = 4;             // the scale slider covers default × 2^±SCALE_SPAN
const NAME_MAX = 16;
// Heads 1 to 3: the canvas's attention purple, then aqua and amber (checked for colour-blind
// separation against both backgrounds). More heads than that are shown one at a time, each in its
// own colour (--head-4 to --head-6), as in the matrix panel and the 3D view.
const HEAD_VARS = ['--att', '--head-2', '--head-3', '--head-4', '--head-5', '--head-6'];
const HEAD_COLORS = { dark: ['#b794ff', '#199e70', '#c98500'], light: ['#7442d6', '#1baf7a', '#eda100'] };   // if the tokens can't be read
const SUBS = '₀₁₂₃₄₅₆₇₈₉';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const f1 = v => (Math.round(v * 10) / 10).toString();
const round3 = v => Math.round(v * 1000) / 1000 + 0;
const r3 = v => String(round3(v));
const sub = k => String(k).split('').map(c => SUBS[+c] ?? c).join('');
const texEsc = s => String(s).replace(/[\\{}$&#^_%~]/g, c => ({
  '\\': '\\textbackslash{}', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}',
}[c] || `\\${c}`));
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const icon = (name, fallback = '') => window.mathboardIcons?.svg(name) || fallback;
// A heatmap cell's fill: colorFor with its alpha capped (as the matrix panel's cells), so the number
// on it can always be text-1.
const cellFill = (v, max, th) => colorFor(v, max, th).replace(/,\s*([\d.]+)\)$/, (m, a) => `,${Math.min(0.72, +a).toFixed(3)})`);

// A token's stored name, read by focus.js as the canvas and the matrix panel read it (a slot holding
// its own default, 't2' for token 2, counts as unnamed); null when it has none.
const tokenName = (net, i) => tokenNames(net)[i] || null;
const plainName = (net, i) => tokenName(net, i) ?? `t${sub(i + 1)}`;
const texName = (net, i) => { const s = tokenName(net, i); return s ? `\\text{${texEsc(s)}}` : `t_{${i + 1}}`; };

// With no token followed, dots and mix show the query that attends most decisively: the row
// holding the largest A_ij (rows that see a single key skipped, the first on ties), the row
// tour.js's captions pick as their example.
function autoQuery(at, h) {
  const A = at?.heads?.[h]?.A || [], n = A.length;
  let best = null;
  A.forEach((row, i) => {
    const vis = row.map((a, j) => (at.causal && j > i ? null : a));
    if (vis.filter(Number.isFinite).length < 2 && n > 1) return;
    vis.forEach(a => { if (Number.isFinite(a) && (!best || a > best.a + 1e-9)) best = { i, a }; });
  });
  return best ? best.i : Math.max(0, n - 1);
}

// Convex hull (monotone chain) of 2-D points, counter-clockwise.
function hull(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo.at(-2), lo.at(-1), q) <= 1e-12) lo.pop(); lo.push(q); }
  for (const q of [...p].reverse()) { while (up.length >= 2 && cross(up.at(-2), up.at(-1), q) <= 1e-12) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

let measureCtx = null;
function textW(s, px = 12, weight = '') {
  if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d');
  measureCtx.font = `${weight} ${px}px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif`;   // --font-ui
  return measureCtx.measureText(String(s)).width;
}
function fitText(s, px, maxW, weight = '') {
  s = String(s);
  if (textW(s, px, weight) <= maxW) return s;
  while (s.length > 1 && textW(s + '…', px, weight) > maxW) s = s.slice(0, -1);
  return s + '…';
}

// ---------------------------------------------------------------- SVG pieces

function arrow(x1, y1, x2, y2, { color, w = 2, op = 1, dash = false, head = 8, attrs = '' } = {}) {
  const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
  const st = `stroke:${color};stroke-width:${f1(w)};opacity:${op.toFixed(3)}`;
  if (L < 2) return `<circle cx="${f1(x2)}" cy="${f1(y2)}" r="${f1(w + 2)}" style="fill:${color};opacity:${op.toFixed(3)}" ${attrs}/>`;
  const hl = Math.min(head + w * 1.6, L * 0.5), ux = dx / L, uy = dy / L, hw = hl * 0.48;
  const bx = x2 - ux * hl, by = y2 - uy * hl;
  return `<g ${attrs}><line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(bx + ux * 0.5)}" y2="${f1(by + uy * 0.5)}" style="${st};stroke-linecap:round${dash ? ';stroke-dasharray:5 4' : ''}"/>`
    + `<polygon points="${f1(x2)},${f1(y2)} ${f1(bx - uy * hw)},${f1(by + ux * hw)} ${f1(bx + uy * hw)},${f1(by - ux * hw)}" style="fill:${color};opacity:${op.toFixed(3)}"/></g>`;
}
// A label next to a tip, pushed outward along the direction it points.
function tipLabel(x, y, dx, dy, main, subText, color = 'var(--text-1)', extra = '') {
  const L = Math.hypot(dx, dy) || 1, ox = dx / L, oy = dy / L;
  const lx = x + ox * 13, ly = y + oy * 13;
  const anchor = ox > 0.35 ? 'start' : ox < -0.35 ? 'end' : 'middle';
  return `<text class="na-tl" x="${f1(lx)}" y="${f1(ly + 4)}" text-anchor="${anchor}" style="fill:${color}" ${extra}>${esc(main)}`
    + `${subText != null ? `<tspan class="na-sub" dy="4">${esc(subText)}</tspan>` : ''}</text>`;
}

// Labels that keep out of each other's way: each one tries spots around its anchor, starting in
// the preferred direction, and takes the first that overlaps no earlier box and stays in bounds.
function placer(bx0, by0, bw, bh) {
  const boxes = [];
  const free = b => b.x >= bx0 && b.y >= by0 && b.x + b.w <= bx0 + bw && b.y + b.h <= by0 + bh
    && !boxes.some(o => b.x < o.x + o.w && b.x + b.w > o.x && b.y < o.y + o.h && b.y + b.h > o.y);
  return {
    block(x, y, w, h) { boxes.push({ x, y, w, h }); },
    // A w x h box near (x, y) at distance r, preferring direction (dx, dy) in screen px.
    // strict: null when nothing fits (else the preferred spot anyway).
    spot(x, y, w, h, dx, dy, { r = 10, strict = false } = {}) {
      const a0 = Math.atan2(dy, dx), tries = [0, 1, -1, 2, -2, 3, -3, 4];
      let first = null;
      for (const k of tries) {
        const a = a0 + k * Math.PI / 4, ux = Math.cos(a), uy = Math.sin(a);
        const cx = x + ux * (r + w / 2 * Math.abs(ux)), cy = y + uy * (r + h / 2 * Math.abs(uy));
        const b = { x: cx - w / 2, y: cy - h / 2, w, h };
        first = first || b;
        if (free(b)) { boxes.push(b); return b; }
      }
      if (strict) return null;
      boxes.push(first);
      return first;
    },
  };
}
// A vector's label (italic symbol, upright subscript) at a free spot next to its tip. strict: ''
// when no spot is free (the caller then tries a shorter label).
function vecLabel(P, x, y, dx, dy, main, subText, color = 'var(--text-1)', extra = '', strict = false) {
  if (!Math.hypot(dx, dy)) { dx = 1; dy = -1; }
  const w = textW(main, 14, 'italic 600') + (subText ? textW(subText, 11, '500') + 1 : 0) + 4, h = 19;
  const b = P.spot(x, y, w, h, dx, dy, { r: 7, strict });
  if (!b) return '';
  return `<text class="na-tl" x="${f1(b.x + 1)}" y="${f1(b.y + 13)}" style="fill:${color}" ${extra}>${esc(main)}`
    + `${subText ? `<tspan class="na-sub" dy="4">${esc(subText)}</tspan>` : ''}</text>`;
}

// Equal-aspect frame around points (data units) in a box of w × h px, the origin always in view.
function frame(points, x0, y0, w, h) {
  let xa = 0, xb = 0, ya = 0, yb = 0;
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    xa = Math.min(xa, x); xb = Math.max(xb, x); ya = Math.min(ya, y); yb = Math.max(yb, y);
  }
  const span = Math.max(xb - xa, yb - ya, 1e-6);
  const pad = span * 0.14 + 0.05;
  xa -= pad; xb += pad; ya -= pad; yb += pad;
  const sc = Math.min(w / (xb - xa), h / (yb - ya));
  const cx = (xa + xb) / 2, cy = (ya + yb) / 2;
  const ox = x0 + w / 2 - cx * sc, oy = y0 + h / 2 + cy * sc;
  return { sc, ox, oy, X: x => ox + x * sc, Y: y => oy - y * sc, box: [x0, y0, w, h], range: [xa, xb, ya, yb] };
}
// Faint grid on a "nice" step, the two axes through the origin, and tick labels.
function axes(F, labels = ['', ''], P = null) {
  const [x0, y0, w, h] = F.box, [xa, xb, ya, yb] = F.range;
  const raw = Math.max(xb - xa, yb - ya) / 6, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * mag).find(s => s >= raw) || raw;
  let s = `<rect x="${f1(x0)}" y="${f1(y0)}" width="${f1(w)}" height="${f1(h)}" rx="8" class="na-plotbg"/>`;
  const X0 = Math.max(x0, Math.min(x0 + w, F.X(0))), Y0 = Math.max(y0, Math.min(y0 + h, F.Y(0)));
  for (let v = Math.ceil(xa / step) * step; v <= xb; v += step) {
    const px = F.X(v);
    if (px < x0 || px > x0 + w) continue;
    s += `<line x1="${f1(px)}" y1="${f1(y0)}" x2="${f1(px)}" y2="${f1(y0 + h)}" class="na-grid"/>`;
    if (Math.abs(v) > step / 2) {
      const t = r3(v).replace('-', '−'), ty = Math.min(y0 + h - 4, Y0 + 14), tw = textW(t, 11);
      s += `<text class="na-tick" x="${f1(px)}" y="${f1(ty)}" text-anchor="middle">${esc(t)}</text>`;
      P?.block(px - tw / 2 - 1, ty - 10, tw + 2, 13);
    }
  }
  for (let v = Math.ceil(ya / step) * step; v <= yb; v += step) {
    const py = F.Y(v);
    if (py < y0 || py > y0 + h) continue;
    s += `<line x1="${f1(x0)}" y1="${f1(py)}" x2="${f1(x0 + w)}" y2="${f1(py)}" class="na-grid"/>`;
    if (Math.abs(v) > step / 2) {
      const t = r3(v).replace('-', '−'), tw = textW(t, 11), tx = Math.max(x0 + 5 + tw, X0 - 5);
      s += `<text class="na-tick" x="${f1(tx)}" y="${f1(py + 4)}" text-anchor="end">${esc(t)}</text>`;
      P?.block(tx - tw - 1, py - 7, tw + 2, 13);
    }
  }
  s += `<line x1="${f1(x0)}" y1="${f1(Y0)}" x2="${f1(x0 + w)}" y2="${f1(Y0)}" class="na-axis"/>`
    + `<line x1="${f1(X0)}" y1="${f1(y0)}" x2="${f1(X0)}" y2="${f1(y0 + h)}" class="na-axis"/>`;
  if (labels[0]) {
    const tw = textW(labels[0], 11);
    s += `<text class="na-axlab" x="${f1(x0 + w - 6)}" y="${f1(Y0 - 6)}" text-anchor="end">${esc(labels[0])}</text>`;
    P?.block(x0 + w - 7 - tw, Y0 - 17, tw + 2, 13);
  }
  if (labels[1]) {
    s += `<text class="na-axlab" x="${f1(X0 + 6)}" y="${f1(y0 + 15)}">${esc(labels[1])}</text>`;
    P?.block(X0 + 5, y0 + 4, textW(labels[1], 11) + 2, 13);
  }
  return s;
}

// ---------------------------------------------------------------- the panel

export function install(ctx) {
  const { store, model } = ctx;
  const ro = !!ctx.audience;
  const stage = ctx.el.stage;
  if (getComputedStyle(stage).position === 'static') stage.style.position = 'relative';
  const fmt = v => (Number.isFinite(v) ? model.fmt(v, 2).replace(/^-/, '−') : v === -Infinity ? '−∞' : '?');

  // ---- panel UI state: position, width, last mode (the audience reads the presenter's, like Train)
  const ui = { x: null, y: null, w: W_DEFAULT, mode: 'arcs' };
  const readUi = () => { try { return JSON.parse(localStorage.getItem(UI_KEY)) || {}; } catch { return {}; } };
  const loadUi = () => {
    const s = readUi();
    ui.x = Number.isFinite(s.x) ? s.x : null;
    ui.y = Number.isFinite(s.y) ? s.y : null;
    ui.w = Number.isFinite(s.w) ? Math.max(W_MIN, s.w) : W_DEFAULT;
    ui.mode = MODES.includes(s.mode) ? s.mode : 'arcs';
  };
  loadUi();
  const saveUi = () => { if (!ro) try { localStorage.setItem(UI_KEY, JSON.stringify(ui)); } catch { /* ignore */ } };
  if (ro) window.addEventListener('storage', e => { if (e.key === UI_KEY || e.key === null) { loadUi(); placePanel(); schedule(); } });

  // ---- lens (lens.js owns it: lensOf is a complete, valid copy; always set a complete object)
  const lensNow = () => lensOf(store);
  function setLens(patch) {
    if (ro) return;
    const next = { ...lensOf(store), ...patch };
    if (!same(next, store.state.lens)) store.set('lens', next);
  }

  // ---- DOM
  const panel = document.createElement('div');
  panel.className = 'nn-attnviz ui-float' + (ro ? ' ro' : '');
  panel.hidden = true;
  panel.innerHTML = `
    <header class="na-head ui-float-head drag">
      <b class="na-title ui-float-title" title="Drag to move; double-click to put it back">Attention</b>
      <div class="na-modes ui-seg sm">${MODES.map(m => `<button type="button" data-mode="${m}" title="${esc(MODE_TITLE[m])}">${MODE_LABEL[m]}</button>`).join('')}</div>
      <select class="na-layer ui-field sm" title="Which attention layer" hidden></select>
      <span class="na-flex ui-float-sp"></span>
      <button type="button" class="na-help ui-help ui-btn sm icon ui-chrome" tabindex="-1" aria-label="How to read it">${icon('help', '?')}</button>
      <button type="button" class="na-close ui-btn sm icon" data-act="close" title="Close (A)" aria-label="Close">${icon('close', '&times;')}</button>
    </header>
    <div class="na-picks"></div>
    <div class="na-stage"><svg class="na-svg" xmlns="http://www.w3.org/2000/svg"></svg><div class="na-msg" hidden></div></div>
    <div class="na-ctl"></div>
    <div class="na-cap"></div>
    <div class="na-tip ui-tip" hidden></div>
    <div class="na-grip ui-grip" title="Drag to resize"></div>`;
  stage.appendChild(panel);
  const $ = s => panel.querySelector(s);
  const head = $('.na-head'), picks = $('.na-picks'), svgWrap = $('.na-stage'), svg = $('.na-svg');
  const msg = $('.na-msg'), ctl = $('.na-ctl'), cap = $('.na-cap'), layerSel = $('.na-layer'), grip = $('.na-grip');
  const helpBtn = $('.na-help'), tip = $('.na-tip');
  for (const type of ['pointerdown', 'mousedown', 'click', 'dblclick', 'wheel', 'contextmenu', 'touchstart']) {
    panel.addEventListener(type, e => e.stopPropagation(), { passive: type === 'wheel' || type === 'touchstart' });
  }
  // A clicked button or select must not keep focus: Space belongs to training.
  panel.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

  // Width and spot from ui, kept inside the stage (a smaller window, the audience's own size); the
  // panel may be no taller than the room below its top, and its drawing scrolls past that.
  function placePanel() {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const w = sw > 0 ? Math.min(ui.w, Math.max(W_MIN, sw - 20)) : ui.w;
    panel.style.width = `${Math.round(w)}px`;
    let top = 10;
    if (ui.x == null || ui.y == null) { panel.style.left = ''; panel.style.top = ''; }
    else {
      const x = sw > 0 ? clamp(ui.x, 0, Math.max(0, sw - w)) : ui.x;
      top = sh > 0 ? clamp(ui.y, 0, Math.max(0, sh - 120)) : ui.y;
      panel.style.left = `${x}px`;
      panel.style.top = `${top}px`;
    }
    panel.style.maxHeight = `calc(100% - ${Math.round(top) + 10}px)`;
  }
  placePanel();
  new ResizeObserver(() => placePanel()).observe(stage);

  // ---- state helpers
  const vizNow = () => { const v = store.state.viz; return v && MODES.includes(v.mode) ? v : null; };
  const shown = () => document.body.dataset.view === 'nn' && !ctx.el.root.hidden;
  function attnLayers(net) {
    const out = [];
    net.layers.forEach((ly, l) => { if (ly.kind === 'attention' && model.attnSpec(net, l)) out.push(l); });
    return out;
  }
  function layerOf(net, viz) {
    const ls = attnLayers(net);
    if (!ls.length) return -1;
    const l = viz?.layer != null ? model.layerIndex(net, viz.layer) : -1;
    return ls.includes(l) ? l : ls[0];
  }
  const tokenCount = net => {
    const t0 = model.tokenShape(net, 0).tokens;
    if (t0 > 1) return t0;
    const l = attnLayers(net)[0];
    return l != null ? model.attnSpec(net, l).tokens : t0;
  };
  const pickQ = (lens, n) => (Number.isInteger(lens?.token) && lens.token >= 0 && lens.token < n ? lens.token : null);
  const pickH = (lens, H) => (Number.isInteger(lens?.head) && lens.head >= 0 && lens.head < H ? lens.head : null);

  // Colours for SVG styles: CSS variables where the stylesheet can resolve them, computed values
  // where the colour depends on a number (heatmap cells).
  const theme = () => (ctx.theme ? ctx.theme() : document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  const headColor = h => `var(${HEAD_VARS[h % HEAD_VARS.length]})`;
  const HIc = 'var(--na-hi)';

  // ---- hover: the panel's own pointer, else a token hover from the canvas / matrix panel
  let local = null;   // { side: 'q' | 'k' | 'v', t, h? }
  let hoverSet = false;
  function isoOf(net, l) {
    if (local) return local;
    const hv = store.state.hover;
    if (hv?.kind !== 'token') return null;
    const hl = typeof hv.layer === 'number' ? hv.layer : model.layerIndex(net, hv.layer);
    if (hl === l) return { side: 'q', t: hv.t, h: hv.h };
    if (hl === l - 1) return { side: hv.g === 1 ? 'k' : hv.g === 2 ? 'v' : 'q', t: hv.t, h: hv.h };
    return null;
  }
  function hoverStore(next) {
    if (ro) return;
    if (!same(next, store.state.hover)) store.set('hover', next);
    hoverSet = !!next;
  }

  // ---------------------------------------------------------------- mode drawings
  // Each returns { svg, h, cap (KaTeX source), info (plain text), geom }.

  function drawArcs(D) {
    const { net, at, n, H, W, l } = D;
    // what the lens hides (masked pairs, A below minA, attention edges off) and dims, as on the canvas
    const E = emphasis(net, D.fwd, D.lens);
    const heads = D.h != null ? [D.h] : H <= 3 ? [...Array(H).keys()] : [0];
    const pad = 1, top = 24;
    const bw = clamp(W * 0.22, 62, 120), bh = 28;
    const rh = clamp(Math.floor(300 / n), 36, 52);
    const Hpx = top + n * rh + 4;
    const rowY = t => top + t * rh + rh / 2;
    const xL = pad + bw, xR = W - pad - bw;
    const iso = D.iso && (D.iso.side === 'q' || D.iso.side === 'k' || D.iso.side === 'v') ? { side: D.iso.side === 'q' ? 'q' : 'k', t: D.iso.t } : null;
    const focusRow = iso ? null : D.q;
    let s = `<text class="na-colhd" x="${f1(pad + bw / 2)}" y="12" text-anchor="middle">query i</text>`
      + `<text class="na-colhd" x="${f1(W - pad - bw / 2)}" y="12" text-anchor="middle">key j</text>`;
    if (heads.length > 1) {
      heads.forEach((hh, k) => {
        const x = W / 2 + (k - (heads.length - 1) / 2) * 60;
        s += `<line x1="${f1(x - 25)}" y1="8" x2="${f1(x - 13)}" y2="8" style="stroke:${headColor(hh)};stroke-width:3;stroke-linecap:round"/>`
          + `<text class="na-leg" x="${f1(x - 8)}" y="12">head ${hh + 1}</text>`;
      });
    }
    // a ribbon per pair: it leaves the query box and reaches the key box level, so it reads as a flow
    const geomLines = [], arcs = [];
    const x1 = xL + 2, x2 = xR - 2, bend = (x2 - x1) * 0.42;
    heads.forEach((hh, k) => {
      const A = at.heads[hh].A, S = at.heads[hh].S, off = (k - (heads.length - 1) / 2) * 3.2;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const a = A[i][j];
        if (!Number.isFinite(S[i][j]) || !Number.isFinite(a) || E.hidden.attn(l, i, j, hh)) continue;   // masked or hidden: no line
        let op = 0.1 + 0.9 * a;
        if (iso && !(iso.side === 'q' ? i === iso.t : j === iso.t)) continue;
        if (!iso && E.any && E.attn(l, i, j, hh) < 1) op *= 0.13;
        const w = 0.7 + 7.5 * a;
        const y1 = rowY(i) + off, y2 = rowY(j) + off;
        arcs.push({ a, svg: `<path class="na-arc" d="M${f1(x1)} ${f1(y1)}C${f1(x1 + bend)} ${f1(y1)} ${f1(x2 - bend)} ${f1(y2)} ${f1(x2)} ${f1(y2)}" data-i="${i}" data-j="${j}" data-h="${hh}" style="stroke:${headColor(hh)};stroke-width:${f1(w)};opacity:${op.toFixed(3)}"/>` });
        geomLines.push({ i, j, h: hh, a, w, op });
      }
    });
    // the strongest pairs on top
    const lines = arcs.sort((p, q) => p.a - q.a).map(p => p.svg).join('');
    let nums = '';
    // numbers at the far end of the isolated (or followed) row, or of the isolated column
    const lab = iso || (focusRow != null ? { side: 'q', t: focusRow } : null);
    if (lab) {
      heads.forEach((hh, k) => {
        const A = at.heads[hh].A, S = at.heads[hh].S, dy = (k - (heads.length - 1) / 2) * 13;
        for (let o = 0; o < n; o++) {
          const [i, j] = lab.side === 'q' ? [lab.t, o] : [o, lab.t];
          if (!Number.isFinite(S[i]?.[j])) continue;
          const y = rowY(o) + dy + 4;
          const x = lab.side === 'q' ? xR - 8 : xL + 8, anchor = lab.side === 'q' ? 'end' : 'start';
          const dot = heads.length > 1 ? `<circle cx="${f1(lab.side === 'q' ? x - textW(fmt(A[i][j]), 11, '500') - 7 : x + 3)}" cy="${f1(y - 4)}" r="3" style="fill:${headColor(hh)};stroke:var(--float);stroke-width:2;paint-order:stroke"/>` : '';
          const tx = heads.length > 1 && lab.side === 'k' ? x + 10 : x;
          nums += `${dot}<text class="na-anum" x="${f1(tx)}" y="${f1(y)}" text-anchor="${anchor}">${esc(fmt(A[i][j]))}</text>`;
        }
      });
    }
    let boxes = '';
    for (const side of ['q', 'k']) {
      for (let t = 0; t < n; t++) {
        const cx = side === 'q' ? pad + bw / 2 : W - pad - bw / 2, y = rowY(t);
        const on = side === 'q' && t === D.q, hot = iso && iso.side === side && iso.t === t;
        const name = fitText(plainName(net, t), 12.5, bw - 14, '500');
        boxes += `<g class="na-tok${on ? ' on' : ''}${hot ? ' hot' : ''}" data-side="${side}" data-t="${t}">`
          + `<rect x="${f1(cx - bw / 2)}" y="${f1(y - bh / 2)}" width="${f1(bw)}" height="${bh}" rx="8"/>`
          + `<text x="${f1(cx)}" y="${f1(y + 4.5)}" text-anchor="middle">${esc(name)}</text></g>`;
      }
    }
    s += `<g class="na-arcs">${lines}</g>${boxes}${nums}`;
    const scaleTex = scaleText(D);
    let capTex = `A = \\operatorname{softmax}\\!\\big(QK^{\\top}\\!\\cdot ${scaleTex}${at.causal ? ' + M' : ''}\\big)`;
    const row = lab?.side === 'q' ? lab.t : null;
    if (row != null && heads.length === 1) {
      const A = at.heads[heads[0]].A;
      capTex += `,\\quad A_{${texName(net, row)},\\,\\cdot} = (${A[row].map(v => fmt(v).replace('−', '-')).join(',\\ ')})`;
    }
    const lz = E.lens;
    // how to read it is behind the help icon; the note says what the lens hides, or what a hover shows
    const info = !lz.show.attention ? 'Attention edges are hidden by the lens (Edges: attention).'
      : iso && !ro ? (iso.side === 'q' ? `${plainName(net, iso.t)} as a query: where it looks.` : `${plainName(net, iso.t)} as a key: who looks at it.`)
        : lz.minA > 0 ? `Lines with A below ${fmt(lz.minA)} are hidden.` : '';
    return { svg: s, h: Hpx, cap: capTex, info, reserve: !ro, geom: { mode: 'arcs', lines: geomLines, heads, rowY: [...Array(n).keys()].map(rowY), xL, xR } };
  }

  // The plane (or number line) a head's d_h dimensions are shown in.
  let dims = [0, 1];
  function dimsFor(dh) {
    if (dh === 1) return [0];
    if (dh === 2) return [0, 1];
    let [a, b] = dims;
    if (!(a >= 0 && a < dh)) a = 0;
    if (!(b >= 0 && b < dh) || b === a) b = a === 0 ? 1 : 0;
    return (dims = [a, b]);
  }
  const dimName = (dh, k) => (dh === 1 ? '' : `dim ${k + 1}`);

  function scaleText(D) {
    const sc = D.at.scale, dh = D.dh, def = 1 / Math.sqrt(dh);
    if (Math.abs(sc - def) < 1e-9) return dh === 1 ? '1' : `\\tfrac{1}{\\sqrt{${dh}}}`;
    return fmt(sc).replace('−', '-');
  }

  function drawDots(D) {
    const { net, at, n, dh, W } = D;
    const hh = D.h ?? 0, i = D.q ?? D.auto, hd = at.heads[hh];
    const { Q, K, S, A } = hd;
    const masked = j => !Number.isFinite(S[i][j]);
    const dot = j => Q[i].reduce((s, v, f) => s + v * K[j][f], 0);
    const col = headColor(hh);
    const wide = W >= 560;
    const PW = wide ? Math.round(W * 0.55) : W;
    const tableX = wide ? PW + 14 : 0, tableW = wide ? W - PW - 14 : W;
    let s = '', geom = { mode: 'dots', i, h: hh, dims: null, keys: [] };
    let plotH;
    const qn = Math.hypot(...Q[i]);
    if (dh === 1) {
      // number line: q on its own lane on top, one lane per key
      const LH = 30, y0 = 24;
      plotH = y0 + (n + 1) * LH + 18;
      const vals = [0, Q[i][0], ...K.map(k => k[0])];
      const lo = Math.min(...vals), hi = Math.max(...vals), padv = (hi - lo || 1) * 0.12;
      const x0 = 46, x1 = PW - 14, X = v => x0 + (v - (lo - padv)) / ((hi + padv) - (lo - padv)) * (x1 - x0);
      s += `<rect x="${f1(x0)}" y="${f1(y0 - 12)}" width="${f1(x1 - x0)}" height="${f1((n + 1) * LH + 6)}" rx="8" class="na-plotbg"/>`;
      s += `<line x1="${f1(X(0))}" y1="${f1(y0 - 12)}" x2="${f1(X(0))}" y2="${f1(y0 + (n + 1) * LH - 6)}" class="na-axis"/>`;
      s += `<text class="na-tick" x="${f1(X(0))}" y="${f1(y0 + (n + 1) * LH + 9)}" text-anchor="middle">0</text>`;
      const lane = k => y0 + k * LH + 4;
      s += `<text class="na-lane" x="4" y="${f1(lane(0) + 4)}">q</text>`;
      s += arrow(X(0), lane(0), X(Q[i][0]), lane(0), { color: HIc, w: 3.2 });
      s += tipLabel(X(Q[i][0]), lane(0), Q[i][0] >= 0 ? 1 : -1, 0, 'q', plainName(net, i), 'var(--text-1)');
      geom.q = [Q[i][0]];
      for (let j = 0; j < n; j++) {
        const y = lane(j + 1), m = masked(j), a = m ? 0 : A[i][j], v = K[j][0];
        s += `<text class="na-lane" x="4" y="${f1(y + 4)}">${esc(fitText(plainName(net, j), 11, 40))}</text>`;
        s += arrow(X(0), y, X(v), y, { color: m ? 'var(--text-4)' : col, w: 1.6 + 3 * a, op: m ? 0.8 : 0.35 + 0.65 * a, dash: m, attrs: `data-kj="${j}"` });
        s += tipLabel(X(v), y, v >= 0 ? 1 : -1, 0, 'k', plainName(net, j), m ? 'var(--text-3)' : 'var(--text-1)');
        geom.keys.push({ j, k: [v], px: [X(v), y], masked: m, s: S[i][j], a: A[i][j], dot: dot(j) });
      }
      geom.x0 = X(0); geom.unit = X(1) - X(0);
    } else {
      const [d0, d1] = dimsFor(dh);
      geom.dims = [d0, d1];
      const P = v => [v[d0], v[d1]];
      const q = P(Q[i]), ks = K.map(P);
      plotH = Math.round(clamp(wide ? W * 0.5 : W * 0.72, 190, 380));
      const F = frame([q, ...ks], 0, 0, PW, plotH - 4);
      const LP = placer(0, 0, PW, plotH - 4);
      s += axes(F, [dimName(dh, d0), dimName(dh, d1)], LP);
      const qq = Math.hypot(q[0], q[1]);
      // the line through q: each key's score is |q| times the length of its shadow on this line
      if (qq > 1e-9) {
        const ux = q[0] / qq, uy = q[1] / qq, R = Math.hypot(F.range[1] - F.range[0], F.range[3] - F.range[2]);
        s += `<line class="na-qline" x1="${f1(F.X(-ux * R))}" y1="${f1(F.Y(-uy * R))}" x2="${f1(F.X(ux * R))}" y2="${f1(F.Y(uy * R))}"/>`;
        ks.forEach((k, j) => {
          const t = k[0] * ux + k[1] * uy, fx = t * ux, fy = t * uy;
          s += `<line class="na-perp" x1="${f1(F.X(k[0]))}" y1="${f1(F.Y(k[1]))}" x2="${f1(F.X(fx))}" y2="${f1(F.Y(fy))}"/>`
            + `<circle cx="${f1(F.X(fx))}" cy="${f1(F.Y(fy))}" r="3" style="fill:${masked(j) ? 'var(--text-4)' : col}"/>`;
        });
      }
      s = `<g clip-path="url(#na-clip)">${s}</g>`;
      // q first, under the keys, so a key along q's direction stays visible on top of it
      const Ox = F.X(0), Oy = F.Y(0), qx = F.X(q[0]), qy = F.Y(q[1]);
      s += arrow(Ox, Oy, qx, qy, { color: HIc, w: 3.6 });
      // labels keep off q's whole arrow and every arrowhead; the masked keys' come last and drop
      // "(masked)" when it doesn't fit (the table says it too)
      const keepOff = (x1, y1, x2, y2, from = 0) => {
        const L = Math.hypot(x2 - x1, y2 - y1), n = Math.max(1, Math.ceil(L / 8));
        for (let t = 0; t <= n; t++) if (L * t / n >= from) LP.block(x1 + (x2 - x1) * t / n - 4, y1 + (y2 - y1) * t / n - 4, 8, 8);
      };
      if (qq > 1e-9) keepOff(Ox, Oy, qx, qy);
      const kp = ks.map(k => [F.X(k[0]), F.Y(k[1])]);
      kp.forEach(([x2, y2]) => keepOff(Ox, Oy, x2, y2, Math.max(0, Math.hypot(x2 - Ox, y2 - Oy) - 14)));
      let labs = qq > 1e-9 ? vecLabel(LP, qx, qy, qx - Ox, qy - Oy, 'q', plainName(net, i), 'var(--text-1)')
        : `<text class="na-tl" x="${f1(Ox + 8)}" y="${f1(Oy - 8)}" style="fill:var(--text-1)">q = 0</text>`;
      ks.forEach((k, j) => {
        const m = masked(j), a = m ? 0 : A[i][j];
        const [x2, y2] = kp[j];
        s += arrow(Ox, Oy, x2, y2, { color: m ? 'var(--text-4)' : col, w: 1.6 + 3 * a, op: m ? 0.8 : 0.35 + 0.65 * a, dash: m, attrs: `data-kj="${j}"` });
        if (!m) labs += vecLabel(LP, x2, y2, x2 - Ox, y2 - Oy, 'k', plainName(net, j), 'var(--text-1)');
        geom.keys.push({ j, k, px: [x2, y2], masked: m, s: S[i][j], a: A[i][j], dot: dot(j) });
      });
      ks.forEach((k, j) => {
        if (!masked(j)) return;
        const [x2, y2] = kp[j], lab = (sub, strict) => vecLabel(LP, x2, y2, x2 - Ox, y2 - Oy, 'k', sub, 'var(--text-3)', '', strict);
        labs += lab(`${plainName(net, j)} (masked)`, true) || lab(plainName(net, j), false);
      });
      s += labs;
      s = `<defs><clipPath id="na-clip"><rect x="0" y="0" width="${f1(PW)}" height="${f1(plotH - 4)}" rx="8"/></clipPath></defs>${s}`;
      geom.q = q; geom.o = [F.ox, F.oy]; geom.sc = F.sc;
    }
    // the scores and the softmax, one row per key
    const ty0 = wide ? 4 : plotH + 10, rowH = 30;
    const cName = tableX + 8, cDot = tableX + tableW * 0.3, cS = tableX + tableW * 0.5, cBar = tableX + tableW * 0.57, barW = tableW * 0.43 - 44;
    s += `<text class="na-colhd" x="${f1(cName)}" y="${f1(ty0 + 11)}">key</text>`
      + `<text class="na-colhd" x="${f1(cDot)}" y="${f1(ty0 + 11)}" text-anchor="end">q·k</text>`
      + `<text class="na-colhd" x="${f1(cS)}" y="${f1(ty0 + 11)}" text-anchor="end">s</text>`
      + `<text class="na-colhd" x="${f1(cBar)}" y="${f1(ty0 + 11)}">A = softmax(s)</text>`;
    for (let j = 0; j < n; j++) {
      const y = ty0 + 20 + j * rowH, m = masked(j), a = m ? 0 : A[i][j];
      s += `<g class="na-krow" data-side="k" data-t="${j}">`
        + `<rect x="${f1(tableX)}" y="${f1(y)}" width="${f1(tableW)}" height="${rowH - 2}" rx="6" class="na-rowbg"/>`
        + `<text class="na-kname" x="${f1(cName)}" y="${f1(y + 18.5)}">${esc(fitText(plainName(net, j), 12.5, cDot - cName - 44, '500'))}</text>`
        + `<text class="na-num" x="${f1(cDot)}" y="${f1(y + 18.5)}" text-anchor="end">${m ? '' : esc(fmt(dot(j)))}</text>`
        + `<text class="na-num na-strong" x="${f1(cS)}" y="${f1(y + 18.5)}" text-anchor="end">${m ? '−∞' : esc(fmt(S[i][j]))}</text>`
        + `<rect x="${f1(cBar)}" y="${f1(y + 7)}" width="${f1(Math.max(0, barW))}" height="14" rx="4" class="na-bartrack"/>`
        + (m ? `<text class="na-num na-muted" x="${f1(cBar + 8)}" y="${f1(y + 18.5)}">masked</text>`
          : `<rect x="${f1(cBar)}" y="${f1(y + 7)}" width="${f1(Math.max(0, barW) * a)}" height="14" rx="4" style="fill:${col}"/>`
          + `<text class="na-num na-strong" x="${f1(cBar + Math.max(0, barW) + 8)}" y="${f1(y + 18.5)}">${esc(fmt(a))}</text>`)
        + '</g>';
    }
    const tableH = 20 + n * rowH + 2;
    const Hpx = wide ? Math.max(plotH, tableH + 8) : plotH + 8 + tableH;
    const qTex = `q_{${texName(net, i)}}`;
    let capTex = `s_j = ${qTex}\\cdot k_j \\times ${scaleText(D)},\\quad A_{${texName(net, i)},j} = \\operatorname{softmax}_j(s)`;
    // how to read it (the shadows, the one-dimensional case) is behind the help icon
    let info = '';
    if (qn < 1e-9) info = 'q = 0: every score is 0, so the attention is spread evenly.';
    else if (dh > 2) info = `The plane of dims ${dims[0] + 1} and ${dims[1] + 1}; the scores use all ${dh}.`;
    return { svg: s, h: Hpx, cap: capTex, info, geom };
  }

  function drawMix(D) {
    const { net, at, n, dh, W } = D;
    const hh = D.h ?? 0, i = D.q ?? D.auto, hd = at.heads[hh];
    const { V, A, S, Z } = hd;
    const vis = [...Array(n).keys()].filter(j => Number.isFinite(S[i][j]));
    const col = headColor(hh);
    let s = '';
    const geom = { mode: 'mix', i, h: hh, dims: null, steps: [], v: [], z: null };
    let Hpx;
    if (dh === 1) {
      const LH = 28, y0 = 22, CH = Math.max(LH, 12 * vis.length + 8);   // the chain's lane holds a staircase
      Hpx = y0 + (n + 1) * LH + CH + 16;
      const chain = [];
      let cur = 0;
      for (const j of vis) { const nx = cur + A[i][j] * V[j][0]; chain.push({ j, a: cur, b: nx, w: A[i][j] }); cur = nx; }
      const vals = [0, ...V.map(v => v[0]), cur];
      const lo = Math.min(...vals), hi = Math.max(...vals), padv = (hi - lo || 1) * 0.12;
      const x0 = 46, x1 = W - 14, X = v => x0 + (v - (lo - padv)) / ((hi + padv) - (lo - padv)) * (x1 - x0);
      const lane = k => y0 + k * LH + 4;
      const vv = vis.map(j => V[j][0]);
      s += `<rect x="${f1(x0)}" y="${f1(y0 - 12)}" width="${f1(x1 - x0)}" height="${f1((n + 1) * LH + CH + 2)}" rx="8" class="na-plotbg"/>`;
      if (vv.length) {
        const a = Math.min(...vv), b = Math.max(...vv);
        s += `<rect x="${f1(X(a))}" y="${f1(y0 - 10)}" width="${f1(Math.max(2, X(b) - X(a)))}" height="${f1((n + 1) * LH + CH)}" class="na-hull" style="fill:${col};stroke:${col}"/>`;
      }
      s += `<line x1="${f1(X(0))}" y1="${f1(y0 - 12)}" x2="${f1(X(0))}" y2="${f1(y0 + (n + 1) * LH + CH - 6)}" class="na-axis"/>`;
      s += `<text class="na-tick" x="${f1(X(0))}" y="${f1(y0 + (n + 1) * LH + CH + 9)}" text-anchor="middle">0</text>`;
      for (let j = 0; j < n; j++) {
        const y = lane(j), m = !vis.includes(j), v = V[j][0];
        s += `<text class="na-lane" x="4" y="${f1(y + 4)}">${esc(fitText(plainName(net, j), 11, 40))}</text>`;
        s += arrow(X(0), y, X(v), y, { color: m ? 'var(--text-4)' : 'var(--text-3)', w: 1.6, op: m ? 0.8 : 0.9, dash: m, attrs: `data-vj="${j}"` });
        s += tipLabel(X(v), y, v >= 0 ? 1 : -1, 0, 'v', plainName(net, j), 'var(--text-3)');
        geom.v.push({ j, v: [v], px: [X(v), y], masked: m });
      }
      const yc = lane(n) - LH / 2 + 10;
      s += `<text class="na-lane" x="4" y="${f1(yc + CH / 2 - 6)}">A·v</text>`;
      chain.forEach((c, k) => {
        const y = yc + k * 12;
        if (k) s += `<line class="na-perp" x1="${f1(X(c.a))}" y1="${f1(y - 12)}" x2="${f1(X(c.a))}" y2="${f1(y)}"/>`;
        s += arrow(X(c.a), y, X(c.b), y, { color: col, w: 2.6, op: 0.45 + 0.55 * c.w, head: 6 });
        geom.steps.push({ j: c.j, a: c.w, from: [c.a], to: [c.b] });
      });
      const yz = yc + CH + LH / 2 - 6;
      s += `<text class="na-lane" x="4" y="${f1(yz + 4)}">z</text>`;
      s += arrow(X(0), yz, X(cur), yz, { color: HIc, w: 3.2 });
      s += tipLabel(X(cur), yz, cur >= 0 ? 1 : -1, 0, 'z', plainName(net, i), 'var(--text-1)');
      geom.z = [cur]; geom.zNet = Z[i]; geom.x0 = X(0); geom.unit = X(1) - X(0);
    } else {
      const [d0, d1] = dimsFor(dh);
      geom.dims = [d0, d1];
      const P = v => [v[d0], v[d1]];
      const vs = V.map(P);
      let cur = [0, 0];
      const chain = [];
      for (const j of vis) {
        const nx = [cur[0] + A[i][j] * vs[j][0], cur[1] + A[i][j] * vs[j][1]];
        chain.push({ j, a: cur, b: nx, w: A[i][j] });
        cur = nx;
      }
      Hpx = Math.round(clamp(W * 0.72, 200, 400));
      const F = frame([...vs, cur], 0, 0, W, Hpx - 4);
      const LP = placer(0, 0, W, Hpx - 4);
      s += axes(F, [dimName(dh, d0), dimName(dh, d1)], LP);
      const hp = hull(vis.map(j => vs[j]));
      if (hp.length >= 3) s += `<polygon class="na-hull" points="${hp.map(p => `${f1(F.X(p[0]))},${f1(F.Y(p[1]))}`).join(' ')}" style="fill:${col};stroke:${col}"/>`;
      else if (hp.length === 2) s += `<line class="na-hull2" x1="${f1(F.X(hp[0][0]))}" y1="${f1(F.Y(hp[0][1]))}" x2="${f1(F.X(hp[1][0]))}" y2="${f1(F.Y(hp[1][1]))}" style="stroke:${col}"/>`;
      s = `<defs><clipPath id="na-clip"><rect x="0" y="0" width="${f1(W)}" height="${f1(Hpx - 4)}" rx="8"/></clipPath></defs><g clip-path="url(#na-clip)">${s}</g>`;
      const Ox = F.X(0), Oy = F.Y(0), zx = F.X(cur[0]), zy = F.Y(cur[1]);
      let labs = '';
      vs.forEach((v, j) => {
        const m = !vis.includes(j), x2 = F.X(v[0]), y2 = F.Y(v[1]);
        s += arrow(Ox, Oy, x2, y2, { color: m ? 'var(--text-4)' : 'var(--text-3)', w: 1.5, op: m ? 0.8 : 0.85, dash: m, attrs: `data-vj="${j}"` });
        s += `<circle cx="${f1(x2)}" cy="${f1(y2)}" r="3" class="na-vtip"/>`;
        if (!m) labs += vecLabel(LP, x2, y2, x2 - Ox, y2 - Oy, 'v', plainName(net, j), 'var(--text-3)');
        geom.v.push({ j, v, px: [x2, y2], masked: m });
      });
      // the masked values' labels after z's and the weights', and without "(masked)" if it doesn't fit
      const maskedLabs = () => vs.forEach((v, j) => {
        if (vis.includes(j)) return;
        const x2 = F.X(v[0]), y2 = F.Y(v[1]), lab = (sub, strict) => vecLabel(LP, x2, y2, x2 - Ox, y2 - Oy, 'v', sub, 'var(--text-3)', '', strict);
        labs += lab(`${plainName(net, j)} (masked)`, true) || lab(plainName(net, j), false);
      });
      // the resultant z first, then the chain on top of it: both end at the same point
      s += arrow(Ox, Oy, zx, zy, { color: HIc, w: 4.2, op: 0.95 });
      // z's label goes across the resultant (a value may sit right at its tip)
      labs += vecLabel(LP, zx, zy, (Oy - zy) || 0.5, (zx - Ox) || 1, 'z', plainName(net, i), 'var(--text-1)');
      chain.forEach(c => {
        const ax = F.X(c.a[0]), ay = F.Y(c.a[1]), bx = F.X(c.b[0]), by = F.Y(c.b[1]);
        s += arrow(ax, ay, bx, by, { color: col, w: 2.4, op: 0.6 + 0.4 * c.w, head: 7 });
        const L = Math.hypot(bx - ax, by - ay);
        if (L > 26) {
          const t = fmt(c.w), tw = textW(t, 11, '500') + 4, mx = (ax + bx) / 2, my = (ay + by) / 2;
          const b = LP.spot(mx, my, tw, 14, -(by - ay) / L, (bx - ax) / L, { r: 5, strict: true });
          if (b) labs += `<text class="na-wlab" x="${f1(b.x + tw / 2)}" y="${f1(b.y + 11)}" text-anchor="middle">${esc(t)}</text>`;
        }
        geom.steps.push({ j: c.j, a: c.w, from: c.a, to: c.b, px: [[ax, ay], [bx, by]] });
      });
      maskedLabs();
      s += labs;
      geom.z = cur; geom.zNet = P(Z[i]); geom.o = [F.ox, F.oy]; geom.sc = F.sc;
    }
    const terms = vis.map(j => `${fmt(A[i][j]).replace('−', '-')}\\,v_{${texName(net, j)}}`).join(' + ');
    const zt = (dh <= 3 ? Z[i] : dims.map(d => Z[i][d])).map(v => fmt(v).replace('−', '-')).join(',\\ ');
    const capTex = `z_{${texName(net, i)}} = ${terms || '0'} = (${zt})`;
    // why z stays in the hull is behind the help icon; the note says what drops out and what is drawn
    const notes = [];
    if (dh > 2) notes.push(`The plane of dims ${dims[0] + 1} and ${dims[1] + 1}.`);
    if (vis.length < n) notes.push('Masked keys drop out of the sum.');
    return { svg: s, h: Hpx, cap: capTex, info: notes.join(' '), geom };
  }

  function drawHeat(D) {
    const { net, at, n, H, W } = D, th = theme();
    const heads = D.h != null ? [D.h] : [...Array(H).keys()];
    const names = [...Array(n).keys()].map(t => plainName(net, t));
    const RL = clamp(Math.max(...names.map(s => textW(s, 11))) + 10, 24, 96);
    const gap = 16;
    const c = Math.floor(clamp((W - RL - gap) / (2 * n), 18, 80));
    const numPx = clamp(Math.round(c * 0.22), 11, 14);
    let maxS = 0;
    for (const hh of heads) for (const row of at.heads[hh].S) for (const v of row) if (Number.isFinite(v)) maxS = Math.max(maxS, Math.abs(v));
    let s = `<defs><pattern id="na-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="na-hatchline"/></pattern></defs>`;
    const iso = D.iso;
    let y = 2;
    const geom = { mode: 'heat', heads, c, blocks: [] };
    for (const hh of heads) {
      const { S, A } = at.heads[hh];
      const xS = RL, xA = RL + n * c + gap;
      // with 2+ heads each pair of maps is headed by its head's swatch and number
      const sw = H > 1 ? 14 : 0;
      if (H > 1) {
        for (const x0 of [xS, xA]) s += `<circle cx="${f1(x0 + 5)}" cy="${f1(y + 8)}" r="4" style="fill:${headColor(hh)}"/>`;
      }
      const hs = H > 1 ? `<tspan class="na-leg" dx="6">head ${hh + 1}</tspan>` : '';
      s += `<text class="na-mtitle" x="${f1(xS + sw)}" y="${f1(y + 12)}">S = QKᵀ · scale${hs}</text>`
        + `<text class="na-mtitle" x="${f1(xA + sw)}" y="${f1(y + 12)}">A = softmax(S)${hs}</text>`;
      y += 22;
      for (const x0 of [xS, xA]) {
        for (let j = 0; j < n; j++) s += `<text class="na-chd" x="${f1(x0 + j * c + c / 2)}" y="${f1(y + 8)}" text-anchor="middle">${esc(fitText(names[j], 11, c - 3))}</text>`;
      }
      y += 14;
      const top = y;
      for (let i = 0; i < n; i++) {
        s += `<text class="na-rhd na-chd${i === D.q ? ' on' : ''}" data-side="q" data-t="${i}" x="${f1(RL - 8)}" y="${f1(y + i * c + c / 2 + 4)}" text-anchor="end">${esc(fitText(names[i], 11, RL - 10))}</text>`;
        for (let j = 0; j < n; j++) {
          const m = !Number.isFinite(S[i][j]);
          for (const [x0, v, max, which] of [[xS, S[i][j], maxS || 1, 'S'], [xA, A[i][j], 1, 'A']]) {
            const x = x0 + j * c, yy = y + i * c;
            const fill = m ? 'url(#na-hatch)' : cellFill(v, max, th);
            const txt = m ? (which === 'S' ? '−∞' : '0') : fmt(v);
            const hot = iso && ((iso.side === 'q' && iso.t === i) || (iso.side !== 'q' && iso.t === j)) && (iso.h == null || iso.h === hh);
            s += `<g class="na-cell${hot ? ' hot' : ''}" data-i="${i}" data-j="${j}" data-h="${hh}" data-m="${which}">`
              + `<rect x="${f1(x + 1)}" y="${f1(yy + 1)}" width="${c - 2}" height="${c - 2}" rx="4" style="fill:${fill}"/>`
              + (c >= 24 ? `<text x="${f1(x + c / 2)}" y="${f1(yy + c / 2 + numPx * 0.36)}" text-anchor="middle" class="${m ? 'na-muted' : ''}" style="font-size:${numPx}px">${esc(txt)}</text>` : '')
              + '</g>';
          }
        }
      }
      if (D.q != null) {
        for (const x0 of [xS, xA]) s += `<rect class="na-ring" x="${f1(x0)}" y="${f1(top + D.q * c)}" width="${n * c}" height="${c}" rx="5"/>`;
      }
      geom.blocks.push({ h: hh, xS, xA, top });
      y = top + n * c + 14;
    }
    const capTex = `S = QK^{\\top}\\!\\cdot ${scaleText(D)}${at.causal ? ' + M' : ''},\\quad A = \\operatorname{softmax}(S)${at.causal ? '' : '\\ \\text{row by row}'}`;
    // the hatching's key stays as a note; how to read the maps is behind the help icon, and a hovered
    // cell's meaning shows in a tooltip
    const info = at.causal ? 'Hatched: masked. M = −∞ where a key comes after its query.' : '';
    return { svg: s, h: y - 4, cap: capTex, info, geom };
  }

  const DRAW = { arcs: drawArcs, dots: drawDots, mix: drawMix, heat: drawHeat };

  // ---------------------------------------------------------------- header, picks, controls, caption

  let layerSig = '';
  function paintHead(net, viz, l) {
    for (const b of panel.querySelectorAll('.na-modes button')) b.classList.toggle('on', b.dataset.mode === viz.mode);
    const ls = attnLayers(net);
    const sig = ls.map(k => `${net.layers[k].id}:${net.layers[k].name}`).join('|');
    if (sig !== layerSig) {
      layerSig = sig;
      layerSel.innerHTML = ls.map(k => `<option value="${esc(net.layers[k].id)}">${esc(net.layers[k].name || `layer ${k}`)}</option>`).join('');
    }
    layerSel.hidden = ls.length < 2;
    layerSel.disabled = ro;
    if (l >= 0 && layerSel.value !== net.layers[l].id) layerSel.value = net.layers[l].id;
    const help = `${MODE_HELP[viz.mode]}\n\n${HELP_COMMON}`;
    if (helpBtn.title !== help) helpBtn.title = help;
  }

  let editing = null, picksSig = '';
  function paintPicks(D) {
    if (editing != null) return;
    const { net, n, H, mode } = D;
    const needOne = mode === 'dots' || mode === 'mix';
    const qShown = D.q ?? (needOne ? D.auto : null);
    const hShown = D.h ?? (needOne ? 0 : null);
    const names = [...Array(n).keys()].map(t => plainName(net, t));
    const sig = JSON.stringify([mode, n, H, names, D.q, D.h, qShown, hShown, ro]);
    if (sig === picksSig) return;
    picksSig = sig;
    // .on: all (nothing followed); .hi: the followed token or head (the lens); .auto: shown because
    // dots and mix need one, but not followed
    const chip = (attr, v, label, cls, title, extra = '') =>
      `<button type="button" class="na-chip ui-chip${cls}" ${attr}="${v}" title="${esc(title)}"${ro ? ' disabled' : ''}>${extra}${label}</button>`;
    const state = (on, followed) => (!on ? '' : followed ? ' hi' : ' auto');
    let h = `<span class="na-lab">query</span>`;
    if (!needOne) h += chip('data-q', -1, 'all', D.q == null ? ' on' : '', 'Every query token');
    for (let t = 0; t < n; t++) {
      const on = t === qShown;
      const title = ro ? names[t] : on && D.q === t ? 'Following this token: click to rename it'
        : `${on ? 'Shown because it attends most decisively; click to follow' : 'Follow'} ${names[t]} (${t + 1}); double-click to rename`;
      h += chip('data-q', t, esc(names[t]), state(on, D.q === t), title);
    }
    if (H > 1) {
      h += `<span class="na-lab na-lab2">head</span>`;
      if (!needOne) h += chip('data-h', -1, 'all', D.h == null ? ' on' : '', 'Every head, one colour each');
      for (let k = 0; k < H; k++) {
        const on = k === hShown;
        h += chip('data-h', k, `${k + 1}`, state(on, D.h === k), on && D.h !== k ? `Head ${k + 1}, shown because one is needed; click to follow it` : `Head ${k + 1} only`,
          `<i class="na-sw ui-sw" style="background:${headColor(k)}"></i>`);
      }
    }
    picks.innerHTML = h;
  }

  let ctlSig = '', dragging = false;
  function paintCtl(D, mode) {
    const { net, l, dh } = D;
    const sig = [mode, l, dh, ro, net.layers[l].id].join('|');
    if (sig !== ctlSig) {
      ctlSig = sig;
      let h = '';
      if ((mode === 'dots' || mode === 'mix') && dh > 2) {
        const opts = [...Array(dh).keys()].map(k => `<option value="${k}">dim ${k + 1}</option>`).join('');
        const sel = k => `<select class="ui-field sm" data-dim="${k}"${ro ? ' disabled' : ''}>${opts}</select>`;
        h += `<label class="na-dims" title="Which two of the head's ${dh} dimensions to draw">plane ${sel(0)}${sel(1)}</label>`;
      }
      // the scale (the temperature): in dots it moves the scores, in mix it moves z
      if (mode === 'dots' || mode === 'mix') {
        h += `<label class="na-scale" title="The layer's scale: larger sharpens A, smaller flattens it (drag; release to keep it)">scale`
          + `<input type="range" class="ui-range" min="${-SCALE_SPAN}" max="${SCALE_SPAN}" step="0.05" value="0"${ro ? ' disabled' : ''}>`
          + `<output></output></label>`
          + `<button type="button" class="na-reset ui-btn xs" data-act="scale-reset" title="Back to 1/√d_k">1/√d</button>`;
      }
      if (mode === 'mix') h += `<button type="button" class="na-send ui-btn sm soft" data-act="send3d" title="Write this construction into the 3D tab: the v_j, sliders for A_ij, the A_ij v_j tip-to-tail with @, and z">${icon('cube')}Send to 3D</button>`;
      ctl.innerHTML = h;
    }
    const dsel = ctl.querySelectorAll('select[data-dim]');
    if (dsel.length) { dsel[0].value = String(dims[0]); dsel[1].value = String(dims[1]); }
    const range = ctl.querySelector('input[type="range"]');
    if (range) {
      const def = 1 / Math.sqrt(dh), sc = D.at.scale, m = Math.log2(sc / def);
      if (!dragging) range.value = String(clamp(Number.isFinite(m) ? m : 0, -SCALE_SPAN, SCALE_SPAN));
      const isDef = Math.abs(sc - def) < 1e-9;
      ctl.querySelector('output').textContent = isDef ? `${fmt(sc)} = 1/√${dh}` : `${fmt(sc)} = ${fmt(sc / def)} × 1/√${dh}`;
      const rb = ctl.querySelector('.na-reset');
      if (rb) rb.disabled = ro || isDef;
    }
  }

  let capKey = '';
  // The formula, and a note: what is true now (masking, the lens, a projection), never how to use it
  // (that is the help icon). reserve: keep the note's line while it is empty (arcs' hover line).
  function paintCap(tex, info, reserve = false) {
    const key = tex + '\u0000' + info + (reserve ? '\u0000r' : '');
    if (key === capKey) return;
    capKey = key;
    let html = '';
    if (tex) {
      try { html = window.katex ? window.katex.renderToString(tex, { throwOnError: false }) : esc(tex); }
      catch { html = esc(tex); }
    }
    cap.innerHTML = `<div class="na-tex">${html}</div>`
      + (info || reserve ? `<div class="na-note ui-caption${reserve ? ' reserve' : ''}">${esc(info)}</div>` : '');
  }

  // ---------------------------------------------------------------- render

  let raf = 0, geom = null, lastMode = null;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(render); };

  function render() {
    raf = 0;
    const viz = vizNow();
    toolBtn?.classList.toggle('on', !!viz);
    panel.hidden = !viz;
    if (!viz || !shown()) return;
    if (viz.mode !== lastMode) { lastMode = viz.mode; capKey = ''; }
    const net = store.net, fwd = store.state.fwd, l = layerOf(net, viz);
    paintHead(net, viz, l);
    const at = l >= 0 ? fwd?.attn?.[l] : null;
    // the drawing's width in CSS px: the stage's content box (its padding and scrollbar gutter out),
    // so the SVG is drawn 1:1. The gutter takes the place of the right padding, so the drawing sits
    // 12 px from both edges.
    const gutter = svgWrap.offsetWidth - svgWrap.clientWidth;
    const padR = `${Math.max(0, 12 - gutter)}px`;
    if (svgWrap.style.paddingRight !== padR) svgWrap.style.paddingRight = padR;
    const cs = getComputedStyle(svgWrap);
    const W = Math.max(0, Math.floor(svgWrap.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)));
    if (viz.mode !== 'heat') tip.hidden = true;
    if (!at || !W) {
      svg.innerHTML = '';
      svg.style.height = '0px';
      picks.innerHTML = ''; picksSig = '';
      ctl.innerHTML = ''; ctlSig = '';
      cap.innerHTML = ''; capKey = '';
      msg.hidden = false;
      msg.textContent = l < 0 ? 'This net has no attention layer. Load one from New net → Attention.'
        : 'The attention layer could not be computed.';
      geom = null;
      return;
    }
    msg.hidden = true;
    const spec = model.attnSpec(net, l), lens = lensNow();
    const n = spec.tokens, H = spec.heads;
    const D = {
      net, fwd, at, spec, l, n, H, dh: spec.dh, W, lens, mode: viz.mode,
      q: pickQ(lens, n), h: pickH(lens, H), iso: isoOf(net, l),
    };
    D.auto = autoQuery(at, D.h ?? 0);
    paintPicks(D);
    paintCtl(D, viz.mode);
    let out;
    try { out = DRAW[viz.mode](D); }
    catch (err) { console.error('[nn/attnviz] draw:', err); out = { svg: '', h: 0, cap: '', info: `Could not draw: ${err.message}`, geom: null }; }
    svg.setAttribute('viewBox', `0 0 ${W} ${Math.max(1, Math.ceil(out.h))}`);
    svg.setAttribute('width', String(W));
    svg.setAttribute('height', String(Math.ceil(out.h)));
    svg.style.height = `${Math.ceil(out.h)}px`;
    svg.innerHTML = out.svg;
    paintCap(out.cap, out.info, !!out.reserve);
    geom = out.geom && { ...out.geom, l, layer: net.layers[l].id, n, H, dh: spec.dh, scale: at.scale };
  }

  // ---------------------------------------------------------------- actions

  function show(mode, { layer } = {}) {
    if (ro) return false;
    const net = store.net, ls = attnLayers(net);
    if (!ls.length) { ctx.toast?.('This net has no attention layer: try a preset from New net → Attention', 3200); return false; }
    const m = MODES.includes(mode) ? mode : vizNow()?.mode || ui.mode;
    let l = layer != null ? (typeof layer === 'number' ? layer : model.layerIndex(net, layer)) : layerOf(net, vizNow());
    if (!ls.includes(l)) l = ls[0];
    const next = { mode: m, layer: net.layers[l].id };
    if (ui.mode !== m) { ui.mode = m; saveUi(); }
    if (!same(next, store.state.viz)) store.set('viz', next);
    return true;
  }
  function hide() { if (!ro && store.state.viz) store.set('viz', null); }
  function toggle(mode) { if (vizNow() && (mode == null || vizNow().mode === mode)) hide(); else show(mode); }
  function cycle(dir = 1) {
    const v = vizNow();
    if (!v) return show(ui.mode);
    return show(MODES[(MODES.indexOf(v.mode) + dir + MODES.length) % MODES.length]);
  }

  function rename(i, raw) {
    if (ro) return false;
    const net = store.net, n = tokenCount(net);
    if (!(i >= 0 && i < n)) return false;
    const s = String(raw ?? '').replace(/\s+/g, ' ').trim().slice(0, NAME_MAX);
    // Other slots keep their names; unnamed ones hold their default (t1, t2, ...), so a list
    // with gaps never renders an empty name anywhere.
    const names = Array.from({ length: n }, (_, k) => tokenName(net, k) ?? `t${k + 1}`);
    names[i] = s || `t${i + 1}`;
    const before = JSON.stringify(net.meta?.tokenNames ?? null);
    net.meta = net.meta || {};
    if (names.every((v, k) => v === `t${k + 1}`)) delete net.meta.tokenNames;
    else net.meta.tokenNames = names;
    if (JSON.stringify(net.meta.tokenNames ?? null) === before) return false;
    store.commit(s ? `Rename token ${i + 1} to ${s}` : `Unname token ${i + 1}`);
    return true;
  }

  function startRename(i) {
    if (ro) return;
    const chip = picks.querySelector(`button[data-q="${i}"]`);
    if (!chip) return;
    editing = i;
    const inp = document.createElement('input');
    inp.className = 'na-name ui-field sm';
    inp.type = 'text';
    inp.maxLength = NAME_MAX;
    inp.value = tokenName(store.net, i) ?? '';
    inp.placeholder = `t${i + 1}`;
    inp.title = 'Enter: rename (empty: back to t' + (i + 1) + '), Esc: cancel';
    inp.style.width = `${Math.max(56, chip.offsetWidth + 16)}px`;
    chip.replaceWith(inp);
    inp.focus();
    inp.select();
    let done = false;
    const finish = ok => {
      if (done) return;
      done = true;
      editing = null;
      picksSig = '';
      if (ok) rename(i, inp.value);
      schedule();
    };
    inp.addEventListener('keydown', e => {
      e.stopPropagation();
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    inp.addEventListener('blur', () => finish(true));
  }

  // z_i = Σ_j A_ij v_j into the 3D tab: the value vectors, one slider per weight (A_ij, named by its
  // indices), the scaled vectors tip-to-tail with @, and z_i itself. Rows from an earlier send are
  // replaced. More than 3 dimensions per head: the plane that is drawn here.
  function send3d() {
    if (ro) return false;
    const g = ctx.graph || window.mathboardGraph;
    if (!g?.addRow) { ctx.toast?.('The 3D tab is not ready'); return false; }
    const net = store.net, fwd = store.state.fwd, l = layerOf(net, vizNow());
    const at = l >= 0 ? fwd?.attn?.[l] : null;
    if (!at) { ctx.toast?.('No attention layer to send'); return false; }
    const spec = model.attnSpec(net, l), lens = lensNow(), n = spec.tokens, dh = spec.dh;
    const hh = pickH(lens, spec.heads) ?? 0, i = pickQ(lens, n) ?? autoQuery(at, hh);
    const { V, A, S } = at.heads[hh];
    const vis = [...Array(n).keys()].filter(j => Number.isFinite(S[i][j]));
    const [d0, d1] = dh > 3 ? dimsFor(dh) : [0, 1];
    const vec = v => (dh === 1 ? `(${r3(v[0])}, 0)` : dh === 2 ? `(${r3(v[0])}, ${r3(v[1])})`
      : dh === 3 ? `(${v.map(r3).join(', ')})` : `(${r3(v[d0])}, ${r3(v[d1])})`);
    const an = (a, b) => (n >= 10 ? `A${a}_${b}` : `A${a}${b}`);
    const term = j => `${an(i + 1, j + 1)} v${j + 1}`;
    // the heads' colours (and HI for z) from the tokens, as the panel draws them
    const cs = getComputedStyle(document.documentElement), th = theme();
    const tok = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
    const cols = HEAD_VARS.map((name, k) => tok(name, HEAD_COLORS[th][k] || HEAD_COLORS[th][0]));
    const defs = [
      ...V.map((v, j) => ({ name: `v${j + 1}`, src: `v${j + 1} = ${vec(v)}`, o: { color: '#9aa3a8' } })),
      ...vis.map(j => ({ name: an(i + 1, j + 1), src: `${an(i + 1, j + 1)} = ${r3(A[i][j])}`, o: { min: 0, max: 1 } })),
    ];
    const chain = vis.map((j, k) => (k ? `${term(j)} @ ${vis.slice(0, k).map(term).join(' + ')}` : term(j)));
    const zsrc = `z${i + 1} = ${vis.map(term).join(' + ') || '0 v1'}`;
    try {
      const GEN = /^(A\d+(?:_\d+)?\s+v\d+(?:\s*@.*)?|z\d+\s*=\s*A\d+(?:_\d+)?\s+v\d+.*)$/;
      const SLIDER = /^(A\d\d|A\d+_\d+)\s*=\s*-?[\d.]+(e-?\d+)?$/;
      const keep = new Set([...chain, zsrc]);
      const names = new Set(defs.map(d => d.name));
      for (const r of [...g.rows]) {
        const src = r.src.trim();
        if ((GEN.test(src) && !keep.has(src)) || (SLIDER.test(src) && !names.has(src.split('=')[0].trim()))) g.removeRow(r);
      }
      const add = (src, o = {}) => {
        const last = g.rows.at(-1);
        if (last && !last.src.trim()) { Object.assign(last, o); g.setRowSource(last, src); } else g.addRow(src, { focus: false, ...o });
      };
      for (const d of defs) {
        const row = g.rowByName(d.name);
        if (row) { if (row.src !== d.src) g.setRowSource(row, d.src); } else add(d.src, d.o);
      }
      for (const src of chain) if (!g.rows.some(r => r.src.trim() === src)) add(src, { color: cols[hh % cols.length] });
      const zr = g.rowByName(`z${i + 1}`);
      if (zr) { if (zr.src !== zsrc) g.setRowSource(zr, zsrc); } else add(zsrc, { color: tok('--hi', th === 'light' ? '#e8a800' : HI) });
      g.setView?.('graph');
      // frame it: these vectors are about unit length, the 3D tab's default extent is 6; a
      // construction in the plane is seen from the top, as the panel draws it
      const shown = [...V.map(v => (dh > 3 ? [v[d0], v[d1]] : v)), dh > 3 ? [at.heads[hh].Z[i][d0], at.heads[hh].Z[i][d1]] : at.heads[hh].Z[i]];
      const ext = Math.max(0.5, ...shown.flat().map(Math.abs).filter(Number.isFinite)) * 1.25;
      try {
        g.scene?.setExtent?.(ext);
        if (dh !== 3) g.scene?.viewPreset?.('top', 600);
      } catch (err) { console.warn('[nn/attnviz] 3D framing:', err); }
      const who = tokenName(net, i) ? ` (query ${tokenName(net, i)})` : '';
      const proj = dh > 3 ? `, dims ${d0 + 1} and ${d1 + 1}` : '';
      g.toast?.(`Sent ${zsrc}${who}${proj} to the 3D tab: drag the A sliders`, 4000);
      return true;
    } catch (err) {
      console.error('[nn/attnviz] send to 3D:', err);
      ctx.toast?.(`Send to 3D failed: ${err.message}`);
      return false;
    }
  }

  // ---------------------------------------------------------------- input

  panel.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || ro) return;
    if (b.dataset.mode) show(b.dataset.mode);
    else if (b.dataset.act === 'close') hide();
    else if (b.dataset.act === 'send3d') send3d();
    else if (b.dataset.act === 'scale-reset') setScale(null, true);
    else if (b.dataset.q != null) {
      const t = +b.dataset.q, cur = pickQ(lensNow(), 1e9);
      if (t >= 0 && cur === t) startRename(t);
      else setLens({ token: t < 0 ? null : t });
    } else if (b.dataset.h != null) {
      const k = +b.dataset.h;
      setLens({ head: k < 0 ? null : k });
    }
  });
  picks.addEventListener('dblclick', e => {
    const b = e.target.closest('button[data-q]');
    if (b && +b.dataset.q >= 0) startRename(+b.dataset.q);
  });
  layerSel.addEventListener('change', () => { show(vizNow()?.mode, { layer: layerSel.value }); layerSel.blur(); });
  ctl.addEventListener('change', e => {
    const sel = e.target.closest('select[data-dim]');
    if (sel) {
      const k = +sel.dataset.dim, v = +sel.value;
      const other = dims[1 - k];
      dims[k] = v;
      if (v === other) dims[1 - k] = dims[k] === 0 ? 1 : 0;
      sel.blur();
      schedule();
    }
  });

  // The scale slider: value = log2(scale / default). Touch while dragging, commit on release.
  function setScale(v, commit) {
    if (ro) return;
    const net = store.net, l = layerOf(net, vizNow());
    if (l < 0) return;
    const id = net.layers[l].id;
    model.setLayer(net, id, { scale: v });
    if (commit) store.commit(v == null ? 'Attention scale: default' : 'Attention scale');
    else store.touch();
  }
  ctl.addEventListener('input', e => {
    const r = e.target.closest('input[type="range"]');
    if (!r || ro) return;
    dragging = true;
    const m = +r.value, spec = model.attnSpec(store.net, layerOf(store.net, vizNow()));
    if (!spec) return;
    setScale(Math.abs(m) < 0.03 ? null : round3(2 ** m / Math.sqrt(spec.dh)), false);
  });
  ctl.addEventListener('change', e => {
    const r = e.target.closest('input[type="range"]');
    if (!r || ro) return;
    dragging = false;
    const m = +r.value, spec = model.attnSpec(store.net, layerOf(store.net, vizNow()));
    if (spec) setScale(Math.abs(m) < 0.03 ? null : round3(2 ** m / Math.sqrt(spec.dh)), true);
    r.blur();
  });

  // Hover in the drawing: tokens isolate their lines (and light the canvas through the shared
  // hover); heatmap cells hover their query row, per head.
  svg.addEventListener('pointermove', e => {
    const net = store.net, viz = vizNow(), l = layerOf(net, viz);
    if (l < 0) return;
    const H = model.attnSpec(net, l).heads;
    const tok = e.target.closest?.('[data-side]');
    const cell = e.target.closest?.('.na-cell');
    const kj = e.target.closest?.('[data-kj]'), vj = e.target.closest?.('[data-vj]');
    let next = null, loc = null, tipAt = null;
    if (tok) {
      const t = +tok.dataset.t, side = tok.dataset.side;
      loc = { side, t };
      next = side === 'q' ? { kind: 'token', layer: l, t } : { kind: 'token', layer: l - 1, t, g: 1 };
    } else if (cell) {
      const i = +cell.dataset.i, j = +cell.dataset.j, hh = +cell.dataset.h, which = cell.dataset.m;
      loc = { side: 'q', t: i, h: hh };
      next = { kind: 'token', layer: l, t: i, ...(H > 1 ? { h: hh } : {}) };
      const at = store.state.fwd?.attn?.[l], v = at?.heads[hh]?.[which]?.[i]?.[j];
      const a = plainName(net, i), b = plainName(net, j), hs = H > 1 ? `, head ${hh + 1}` : '';
      const [main, what] = !Number.isFinite(v) || (which === 'A' && !Number.isFinite(at.heads[hh].S[i][j]))
        ? [`${which}(${a}, ${b}): masked`, `${a} comes before ${b}${hs}`]
        : which === 'S' ? [`S(${a}, ${b}) = ${fmt(v)}`, `q·k × scale: how well ${a}'s query matches ${b}'s key${hs}`]
          : [`A(${a}, ${b}) = ${fmt(v)}`, `how much ${a} reads ${b}${hs}`];
      tipAt = { el: cell, key: `${which}|${i}|${j}|${hh}|${main}`, html: `${esc(main)}<small>${esc(what)}</small>` };
    } else if (kj) {
      loc = { side: 'k', t: +kj.dataset.kj };
      next = { kind: 'token', layer: l - 1, t: +kj.dataset.kj, g: 1 };
    } else if (vj) {
      loc = { side: 'v', t: +vj.dataset.vj };
      next = { kind: 'token', layer: l - 1, t: +vj.dataset.vj, g: 2 };
    }
    showTip(tipAt);
    if (!same(loc, local)) { local = loc; schedule(); }
    if (next || hoverSet) hoverStore(next);
  });
  svg.addEventListener('pointerleave', () => {
    showTip(null);
    if (local) { local = null; schedule(); }
    if (hoverSet) hoverStore(null);
  });
  // A heatmap cell's meaning, in a tooltip over it (below it when there is no room above).
  let tipKey = '';
  function showTip(t) {
    if (!t) { tip.hidden = true; tipKey = ''; return; }
    if (t.key !== tipKey) { tipKey = t.key; tip.innerHTML = t.html; }
    tip.hidden = false;
    const pr = panel.getBoundingClientRect(), cr = t.el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const x = clamp(cr.left + cr.width / 2 - pr.left - tw / 2, 6, Math.max(6, pr.width - tw - 6));
    let y = cr.top - pr.top - th - 6;
    if (y < 4) y = cr.bottom - pr.top + 6;
    tip.style.left = `${Math.round(x)}px`;
    tip.style.top = `${Math.round(y)}px`;
  }
  svg.addEventListener('click', e => {
    if (ro) return;
    const tok = e.target.closest?.('[data-side="q"], .na-tok[data-side="k"]');
    const cell = e.target.closest?.('.na-cell');
    const t = tok ? +tok.dataset.t : cell ? +cell.dataset.i : null;
    if (t == null) return;
    const cur = pickQ(lensNow(), 1e9);
    setLens({ token: cur === t ? null : t });
  });

  // Drag the panel by its header (not by its buttons); double-click puts it back.
  head.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest('button, select')) return;
    if (document.body.classList.contains('clean') && ro) return;
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
    if (e.target.closest('button, select') || ro) return;
    ui.x = ui.y = null;
    placePanel();
    saveUi();
  });
  // Resize from the corner grip: the width; the drawings follow it.
  grip.addEventListener('pointerdown', e => {
    if (e.button !== 0 || ro) return;
    e.preventDefault();
    const sr = stage.getBoundingClientRect(), pr = panel.getBoundingClientRect(), x0 = e.clientX, w0 = pr.width;
    grip.setPointerCapture(e.pointerId);
    const move = ev => {
      ui.w = Math.round(clamp(w0 + ev.clientX - x0, W_MIN, Math.max(W_MIN, sr.right - pr.left - 6)));
      placePanel();
    };
    const up = () => {
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', up);
      grip.removeEventListener('pointercancel', up);
      saveUi();
    };
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', up);
    grip.addEventListener('pointercancel', up);
  });

  let toolBtn = null;
  if (!ro && ctx.addButton) {
    try {
      toolBtn = ctx.addButton({
        label: 'Attention', icon: icon('attention', '&#8978;'), group: 'attnviz',
        title: 'Attention panel: who reads whom, the dot products, the weighted sum, the heatmaps (A: open / close, M: next mode)',
        onClick: () => toggle(),
      });
    } catch (err) { console.warn('[nn/attnviz] addButton:', err); }
  }

  window.addEventListener('keydown', e => {
    if (ro || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : '';   // the typed letter, as lens.js and tour.js read theirs
    if (k !== 'a' && k !== 'm') return;
    if (ctx.active && !ctx.active(e)) return;
    e.preventDefault();
    if (k === 'a') toggle();
    else cycle(e.shiftKey ? -1 : 1);
  });

  // ---------------------------------------------------------------- store events

  store.on('viz', schedule);
  store.on('lens', schedule);
  store.on('values', schedule);
  store.on('hover', () => { if (vizNow()) schedule(); });
  store.on('net', p => {
    if (p?.structural) { layerSig = ''; picksSig = ''; ctlSig = ''; }
    const v = vizNow();
    // keep viz.layer pointing at a working attention layer (the presenter fixes it; the audience mirrors)
    if (v && !ro) {
      const l = layerOf(store.net, v);
      if (l >= 0 && store.net.layers[l].id !== v.layer) store.set('viz', { ...v, layer: store.net.layers[l].id });
    }
    schedule();
  });
  ctx.onTheme?.(() => { capKey = ''; schedule(); });
  ctx.onShow?.(v => { if (v) schedule(); });
  new ResizeObserver(() => schedule()).observe(svgWrap);

  const api = {
    MODES,
    get open() { return !!vizNow(); },
    get mode() { return vizNow()?.mode ?? null; },
    get layer() { const l = layerOf(store.net, vizNow()); return l >= 0 ? store.net.layers[l].id : null; },
    layers: () => attnLayers(store.net).map(l => store.net.layers[l].id),
    show, hide, toggle, cycle, send3d, rename,
    geometry: () => geom,
    el: panel,
  };
  ctx.attnviz = api;
  panel.nnAttn = api;
  schedule();
}
