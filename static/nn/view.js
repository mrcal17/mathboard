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

import { colorFor } from './store.js';

const NS = 'http://www.w3.org/2000/svg';
const XHTML = 'http://www.w3.org/1999/xhtml';
const R = 26;                          // neuron radius, world px (CSS px at zoom 1)
const MIN_K = 0.15, MAX_K = 4, FIT_MAX_K = 1.6;
const HEAD_UP = R + 50;                // header centre sits this far above the topmost neuron's centre
const LANE_DOWN = R + 30;              // lanes end this far below the lowest neuron's centre
const COL = 160;                       // column spacing for empty layers (as model.js)
const SNAP = 10;                       // a dragged neuron snaps onto its column within this (Alt: off)
const TEXT_K = 0.85, TEXT_MAX = 1.5;   // zoomed out past TEXT_K, text grows (up to TEXT_MAX) to stay legible
const BEYOND = 2.5 * R;                // a double-click this far past the end columns adds a layer
const GAP = 170;                       // a layer added by double-click keeps this far from its neighbours (as nn.js)
const VARS = ['--nnv-hi', '--nnv-bg', '--nnv-text', '--nnv-muted', '--nnv-node', '--nnv-rim', '--nnv-band',
  '--nnv-band-line', '--nnv-head', '--nnv-head-line', '--nnv-dot', '--nnv-bad'];
const MARKS = ['sel', 'hov', 'rel', 'lit', 'bias', 'show', 'drop'];

const r1 = v => Math.round(v * 10) / 10;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const isNum = v => typeof v === 'number' && Number.isFinite(v);
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
  const gEdges = mk('g', { class: 'nnv-edges' }, content);
  const pairEl = mk('path', { class: 'nnv-pair', display: 'none' }, content);
  const gPulses = mk('g', { class: 'nnv-pulses' }, content);
  const gLabels = mk('g', { class: 'nnv-wls' }, content);
  const gNodes = mk('g', { class: 'nnv-nodes' }, content);
  const gHeads = mk('g', { class: 'nnv-heads' }, content);
  const ghost = mk('path', { class: 'nnv-ghost', display: 'none' }, content);
  const hint = mk('text', { class: 'nnv-hint', x: '50%', y: '50%', display: 'none' }, svg);
  hint.textContent = 'Double-click to add a neuron';
  stage.prepend(svg);

  // ---------------------------------------------------------------- state
  const nodes = new Map(), edges = new Map(), layers = new Map(), images = new Map();
  const V = { k: 1, x: 0, y: 0 };
  let I = null, stale = true;
  let dirty = { build: true }, raf = 0;
  let shown = document.body.dataset.view === 'nn';
  let needFit = 0, everFit = false, userMoved = false, fitAnim = 0;   // needFit: false | ms of the pending fit
  let showW = false, marked = [], pulseKey = '', pairIds = null, dropId = null, textScale = 1;

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
    for (const e of net.edges) edgeById.set(e.id, e);
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
    I = { li, byLayer, nodeById, edgeById, rank, cols, box, cy, full: real.every(Boolean) };
    return I;
  }

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
    mk('circle', { class: 'nnv-hl', r: R + 9 }, g);
    const gring = mk('circle', { class: 'nnv-gring', r: R + 4 }, g);
    mk('circle', { class: 'nnv-base', r: R }, g);
    const fill = mk('circle', { class: 'nnv-fill', r: R }, g);
    mk('circle', { class: 'nnv-rim', r: R }, g);
    const fo = mk('foreignObject', { class: 'nnv-lab', x: -R - 14, y: -R, width: 2 * R + 28, height: 2 * R }, g);
    const div = document.createElementNS(XHTML, 'div');
    div.className = 'nnv-tex';
    fo.appendChild(div);
    const val = mk('text', { class: 'nnv-val', y: R + 10 }, g);
    const grad = mk('text', { class: 'nnv-grad', x: R + 7, y: -R * 0.45 }, g);
    const tgt = mk('text', { class: 'nnv-tgt', x: R + 7, y: R * 0.5 }, g);
    const bias = mk('text', { class: 'nnv-bias', x: -R - 7, y: -R * 0.45 }, g);
    if (!audience) mk('circle', { class: 'nnv-handle', 'data-kind': 'handle', 'data-id': n.id, cx: R + 1, r: 7 }, g);
    const r = { id: n.id, g, gring, fill, fo, div, val, grad, tgt, bias, img: null, label: undefined, x: NaN, y: NaN };
    if (images.has(n.id)) applyImage(r, images.get(n.id));
    return r;
  }
  function makeEdge(e) {
    const g = mk('g', { class: 'nnv-edge', 'data-kind': 'edge', 'data-id': e.id }, gEdges);
    const hit = mk('path', { class: 'nnv-hit' }, g);
    const line = mk('path', { class: 'nnv-line' }, g);
    const lab = mk('text', { class: 'nnv-wl' }, gLabels);
    const t1 = mk('tspan', { x: 0 }, lab);
    const t2 = mk('tspan', { x: 0, dy: '1.2em', class: 'nnv-wl-g' }, lab);
    return { id: e.id, g, hit, line, lab, t1, t2, G: null };
  }
  function makeLayer(l) {
    const band = mk('rect', { class: 'nnv-band', rx: 18 }, gBands);
    const g = mk('g', { class: 'nnv-head', 'data-kind': 'layer', 'data-id': l.id }, gHeads);
    const gi = mk('g', { class: 'nnv-head-in' }, g);   // carries the text boost
    if (textScale !== 1) gi.setAttribute('transform', `scale(${textScale})`);
    const bg = mk('rect', { class: 'nnv-head-bg', rx: 9, y: -21, height: 40 }, gi);
    const name = mk('text', { class: 'nnv-head-name', y: -4 }, gi);
    const sub = mk('text', { class: 'nnv-head-sub', y: 12 }, gi);
    return { id: l.id, band, g, gi, bg, name, sub, nameS: null, subS: null, measured: false };
  }
  const drop = r => { r.g.remove(); r.lab?.remove(); r.band?.remove(); };

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

  // ---------------------------------------------------------------- meta (labels, headers)
  function renderLabel(r, tex) {
    r.label = tex;
    r.div.style.fontSize = `${texSize(tex)}px`;
    const k = window.katex;
    if (!k || !tex) { r.div.textContent = tex || ''; return; }
    try { k.render(String(tex), r.div, { throwOnError: false, output: 'html' }); }
    catch { r.div.textContent = tex; }
  }
  function measureHead(r) {
    let w = 0;
    try { w = Math.max(r.name.getComputedTextLength(), r.sub.getComputedTextLength()); } catch { /* not rendered */ }
    if (!w) return;
    r.measured = true;
    w = Math.max(64, w + 26);
    put(r.bg, 'x', r1(-w / 2));
    put(r.bg, 'width', r1(w));
  }
  function syncMeta() {
    const net = store.net, I = ix(), L = net.layers.length;
    for (const n of net.nodes) {
      const r = nodes.get(n.id);
      if (r && r.label !== n.label) renderLabel(r, n.label);
    }
    net.layers.forEach((l, i) => {
      const r = layers.get(l.id);
      if (!r) return;
      const count = I.byLayer[i].length;
      const name = l.name || (i === 0 ? 'Input' : i === L - 1 ? 'Output' : 'Hidden');
      const sub = i === 0 ? `${count} input${count === 1 ? '' : 's'}`
        : `${model.ACTS?.[l.act]?.label || l.act || 'Identity'} · ${count}`;
      if (r.nameS === name && r.subS === sub) return;
      r.nameS = name; r.subS = sub;
      txt(r.name, name); txt(r.sub, sub);
      r.g.setAttribute('aria-label', `${name}: ${sub}`);
      r.measured = false;
      const w = Math.max(64, Math.max(name.length * 8.6, sub.length * 7) + 26);   // until measured
      put(r.bg, 'x', r1(-w / 2));
      put(r.bg, 'width', r1(w));
      r.band.classList.toggle('empty', !count);
    });
  }

  // ---------------------------------------------------------------- layout (positions)
  // Skip edges bow around the columns they jump over: above them if they start and end in the
  // upper half, below otherwise.
  function geom(a, b) {
    const I = ix(), la = I.li.get(a.layer), lb = I.li.get(b.layer);
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
      // Stagger the numbers by source row so the labels of edges into one neuron don't stack.
      const n = I.byLayer[I.li.get(a.layer)]?.length || 1;
      const p = r.G.at(0.3 + (0.4 * ((I.rank.get(a.id) ?? 0) + 0.5)) / n);
      put(r.lab, 'transform', `translate(${r1(p.x)},${r1(p.y)})`);
    }
    const top = (I.box ? I.box.minY : I.cy) - HEAD_UP;
    const bottom = (I.box ? I.box.maxY : I.cy) + LANE_DOWN;
    net.layers.forEach((l, i) => {
      const r = layers.get(l.id), c = I.cols[i];
      if (!r) return;
      put(r.g, 'transform', `translate(${r1(c.x)},${r1(top)})`);
      const x0 = c.minX - R - 14, x1 = c.maxX + R + 14;
      put(r.band, 'x', r1(x0));
      put(r.band, 'width', r1(x1 - x0));
      put(r.band, 'y', r1(top + 23));
      put(r.band, 'height', r1(Math.max(0, bottom - top - 23)));
      if (!r.measured && shown) measureHead(r);
    });
    for (const p of gPulses.children) put(p, 'd', edges.get(p.dataset.id)?.G?.d ?? '');
    layoutPair();
    placeHint();
  }
  // The empty-net hint sits under the lanes, centred on them (screen px: it doesn't scale).
  function placeHint() {
    if (hint.getAttribute('display') === 'none') return;
    const I = ix(), xs = I.cols.map(c => c.x);
    const x = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 450;
    put(hint, 'x', r1(x * V.k + V.x));
    put(hint, 'y', r1((I.cy + LANE_DOWN) * V.k + V.y + 30));
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
  function paintLabel(r, e) {
    const bwd = store.state.bwd;
    txt(r.t1, num(e.w));
    txt(r.t2, bwd && isNum(bwd.edge?.[e.id]) ? `∂L/∂w ${num(bwd.edge[e.id])}` : '');
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
        txt(r.grad, `δ ${num(d)}`);
      } else {
        put(r.gring, 'stroke', 'none');
        txt(r.grad, '');
      }
      txt(r.tgt, L > 1 && l === L - 1 && isNum(n.target) ? `y = ${num(n.target)}` : '');
      txt(r.bias, l > 0 ? `b = ${num(n.bias ?? 0)}` : '');
    }
    for (const e of net.edges) {
      const r = edges.get(e.id);
      if (!r) continue;
      const w = isNum(e.w) ? e.w : 0;
      put(r.line, 'stroke', colorFor(w, maxW, th));
      put(r.line, 'stroke-width', r1(1.2 + 5 * Math.min(1, Math.abs(w) / maxW)));
      if (showW || r.lab.__show) paintLabel(r, e);
    }
  }

  // ---------------------------------------------------------------- highlight (sel, hover, anim)
  // What a sel / hover target lights up. Matrix rows and columns name layers by index or id.
  function resolve(t) {
    const out = { nodes: [], edges: [], layers: [], relNodes: [], relEdges: [], pair: null, bias: null, any: false };
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
        if (t.kind === 'bias') { out.bias = t.id; break; }
        for (const e of net.edges) {
          if (e.from === t.id) { out.relEdges.push(e.id); out.relNodes.push(e.to); }
          else if (e.to === t.id) { out.relEdges.push(e.id); out.relNodes.push(e.from); }
        }
        break;
      case 'edge': {
        const e = I.edgeById.get(t.id);
        if (e) { out.edges.push(e.id); out.relNodes.push(e.from, e.to); }
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
      default: break;
    }
    out.any = !!(out.nodes.length || out.edges.length || out.layers.length || out.pair);
    return out;
  }
  // state.anim { dir, l, i, phase }: neuron i of layer l; fwd lights its incoming edges, bwd its outgoing ones.
  function resolveAnim(a) {
    const out = { node: null, dir: 'fwd', edges: [], rel: [], phase: '' };
    if (!a) return out;
    const I = ix(), l = typeof a.l === 'number' ? a.l : I.li.get(a.l), n = I.byLayer[l]?.[a.i];
    if (!n) return out;
    out.node = n.id;
    out.dir = a.dir === 'bwd' ? 'bwd' : 'fwd';
    out.phase = String(a.phase ?? '');
    for (const e of store.net.edges) {
      if (out.dir === 'fwd' ? e.to !== n.id : e.from !== n.id) continue;
      out.edges.push(e.id);
      out.rel.push(out.dir === 'fwd' ? e.from : e.to);
    }
    return out;
  }
  function highlight() {
    for (const [e, c] of marked) { e.classList.remove(c); if (c === 'show') e.__show = false; }
    marked = [];
    const mark = (e, c) => { if (e) { e.classList.add(c); marked.push([e, c]); } };
    const onNode = (id, c) => mark(nodes.get(id)?.g, c);
    const onEdge = (id, c) => { const r = edges.get(id); if (r) { mark(r.g, c); mark(r.lab, c); } };
    const onLayer = (id, c) => { const r = layers.get(id); if (r) { mark(r.g, c); mark(r.band, c); } };
    const I = ix(), S = resolve(store.state.sel), H = resolve(store.state.hover), A = resolveAnim(store.state.anim);
    S.nodes.forEach(id => onNode(id, 'sel'));
    S.edges.forEach(id => onEdge(id, 'sel'));
    S.layers.forEach(id => onLayer(id, 'sel'));
    H.nodes.forEach(id => onNode(id, 'hov'));
    H.edges.forEach(id => onEdge(id, 'hov'));
    H.layers.forEach(id => onLayer(id, 'hov'));
    H.relNodes.forEach(id => onNode(id, 'rel'));
    H.relEdges.forEach(id => onEdge(id, 'rel'));
    if (H.bias) onNode(H.bias, 'bias');
    if (A.node) {
      onNode(A.node, 'lit');
      A.edges.forEach(id => onEdge(id, 'lit'));
      A.rel.forEach(id => onNode(id, 'rel'));
    }
    // The hovered / selected edge shows its numbers even with W off.
    for (const id of [...S.edges, ...H.edges]) {
      const r = edges.get(id), e = I.edgeById.get(id);
      if (!r || !e) continue;
      mark(r.lab, 'show');
      r.lab.__show = true;
      paintLabel(r, e);
    }
    svg.classList.toggle('focus', H.any || !!A.node);
    pairIds = H.pair;
    layoutPair();
    const key = A.node ? `${A.dir}|${A.node}|${A.edges.join()}|${A.phase}` : '';
    if (key !== pulseKey) {
      pulseKey = key;
      gPulses.textContent = '';
      for (const id of A.edges) {
        const r = edges.get(id);
        if (r?.G) mk('path', { class: `nnv-pulse ${A.dir}`, d: r.G.d, pathLength: 100, 'data-id': id }, gPulses);
      }
    }
  }

  // ---------------------------------------------------------------- frame scheduling
  function schedule() {
    if (!raf && shown) raf = requestAnimationFrame(() => { raf = 0; flush(); });
  }
  function flush() {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    const d = dirty;
    dirty = {};
    if (d.build) rebuild();
    if (d.build || d.meta) syncMeta();
    const moved = d.build || d.layout || (d.meta && posChanged());
    if (moved) layoutAll();
    if (d.build || d.paint) paint();
    if (d.build || d.hl) highlight();
    if (audience && moved && shown && needFit === false && outOfView()) needFit = 300;
    if (needFit !== false && shown && stage.clientWidth && stage.clientHeight) fit(everFit ? needFit : 0);
  }
  const invalidate = (...keys) => { for (const k of keys) dirty[k] = true; schedule(); };

  store.on('net', p => { stale = true; invalidate(p?.structural ? 'build' : 'meta'); });
  store.on('values', () => invalidate('paint'));
  store.on('layout', () => { stale = true; invalidate('layout'); });
  for (const k of ['sel', 'hover', 'anim']) store.on(k, () => invalidate('hl'));
  ctx.onTheme?.(() => invalidate('paint'));
  ctx.onShow?.(v => {
    shown = v;
    if (!v) return;
    for (const r of layers.values()) r.measured = false;
    invalidate('layout', 'paint', 'hl');
  });
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => {
      if (!stage.clientWidth || !stage.clientHeight) return;
      if (!userMoved || audience) needFit = 0;   // follow the resize without easing
      invalidate('layout');
    }).observe(stage);
    // A floating panel (train) opening, folding or closing changes fitArea(): refit, unless the
    // user has panned or zoomed since the last fit.
    let areaKey = '';
    const areaChanged = () => {
      if (!stage.clientWidth || !stage.clientHeight) return;
      const A = fitArea(), key = `${Math.round(A.x)},${Math.round(A.w)}`;
      if (key === areaKey) return;
      const first = !areaKey;
      areaKey = key;
      if (!first && !userMoved) { needFit = 300; invalidate('layout'); }
    };
    const panels = new ResizeObserver(areaChanged);
    const watch = () => { for (const c of stage.children) if (c !== svg) panels.observe(c); };
    new MutationObserver(watch).observe(stage, { childList: true });
    watch();
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
    for (const r of layers.values()) {
      if (ts === 1) r.gi.removeAttribute('transform');
      else r.gi.setAttribute('transform', `scale(${ts})`);
    }
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
    flush();
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
  // The part of the stage the net should fit into: beside a tall panel floating on one side (train).
  function fitArea() {
    const s = stage.getBoundingClientRect();
    let left = 0, right = s.width;
    for (const c of stage.children) {
      if (c === svg) continue;
      const r = c.getBoundingClientRect();
      if (!r.width || r.height < s.height * 0.35 || r.width > s.width * 0.45) continue;
      if (r.right >= s.right - 40 && r.left - s.left > s.width * 0.5) right = Math.min(right, r.left - s.left);
      else if (r.left <= s.left + 40 && r.right - s.left < s.width * 0.5) left = Math.max(left, r.right - s.left);
    }
    return { x: left, y: 0, w: right - left, h: s.height };
  }
  function fit(ms = 0) {
    if (!stage.clientWidth || !stage.clientHeight) { needFit = 0; return; }
    needFit = false;                     // before contentBox(): it flushes, and a flush may fit
    userMoved = false;
    everFit = true;
    const b = contentBox(), A = fitArea();
    const pad = clamp(Math.min(A.w, A.h) * 0.05, 12, 40);
    const k = clamp(Math.min((A.w - 2 * pad) / b.w, (A.h - 2 * pad) / b.h), MIN_K, FIT_MAX_K);
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
    const b = contentBox(), A = fitArea();
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
    for (const e of clone.querySelectorAll('.nnv-grid, pattern, .nnv-pulses, .nnv-ghost, .nnv-pair, .nnv-handle, .nnv-hint')) e.remove();
    for (const c of MARKS) for (const e of clone.querySelectorAll(`.${c}`)) e.classList.remove(c);
    clone.classList.remove('focus', 'panning', 'wiring', 'dragging');
    for (const e of clone.querySelectorAll('.nnv-head-in')) e.removeAttribute('transform');   // drawn at zoom 1
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
  wBtn = ctx.addButton?.({ label: 'Weights', title: 'Numbers on the edges: weights, and ∂L/∂w once targets are set (W)', onClick: () => toggleW(), group: 'view' }) || null;
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
  let hoverKey = '', myHover = null;
  function hoverFrom(target) {
    const h = hitOf(target), kind = h?.dataset.kind === 'handle' ? 'node' : h?.dataset.kind;
    const key = h ? `${kind}:${h.dataset.id}` : '';
    if (key === hoverKey) return;
    hoverKey = key;
    if (h) { myHover = { kind, id: h.dataset.id }; store.set('hover', myHover); }
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
  function wireMove(d, w) {
    const I = ix(), src = I.nodeById.get(d.from);
    if (!src) return;
    const tgt = nearest(w, d.from), ok = !!tgt && I.li.get(tgt.layer) !== I.li.get(src.layer);
    put(ghost, 'd', (tgt ? straight(src, tgt) : straight(src, w, 0)).d);
    put(ghost, 'display', null);
    ghost.classList.toggle('bad', !!tgt && !ok);
    setDrop(ok ? tgt.id : null);
  }
  function wireEnd(d, w) {
    put(ghost, 'display', 'none');
    ghost.classList.remove('bad');
    setDrop(null);
    const net = store.net, I = ix(), src = I.nodeById.get(d.from), tgt = w && nearest(w, d.from);
    if (!src || !tgt) return;
    const la = I.li.get(src.layer), lb = I.li.get(tgt.layer);
    if (la === lb) { ctx.toast?.('Edges run between layers, not within one'); return; }
    const old = model.edgeBetween(net, src.id, tgt.id);
    if (old) { select({ kind: 'edge', id: old.id }); ctx.toast?.('Those two are already connected'); return; }
    const lim = Math.sqrt(6 / (I.byLayer[la].length + I.byLayer[lb].length));
    const w0 = (Math.random() < 0.5 ? -1 : 1) * lim * (0.25 + 0.75 * Math.random());
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
    const x = c.n && Math.abs(p.x - c.x) < R + 16 ? c.x : p.x;
    const index = I.byLayer[l].filter(n => n.y < p.y).length;   // rows follow the picture, top to bottom
    const id = model.addNode(net, layer.id, { x, y: p.y, index, connect: true, seed: (Math.random() * 2 ** 31) | 0 });
    if (!id) return;
    store.commit('Add neuron');
    select({ kind: 'node', id });
  }
  // As the toolbar's "+ Layer": the direct edges between the two neighbours would become skip
  // edges, so they go; a new hidden layer copies the activation of the hidden layer before it.
  // The new column keeps GAP from its neighbours: later columns move right to make room.
  function insertLayer(at, p) {
    const net = store.net, L = net.layers.length, hidden = at > 0 && at < L, before = net.layers[at - 1];
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
    } else if (kind === 'layer') drag = { ...base, type: 'layer', id, start: layerNodes(id).map(m => [m.id, m.x, m.y]) };
    else drag = { ...base, type: 'pan', x0: V.x, y0: V.y, edge: kind === 'edge' ? id : null, click: e.button === 0 };
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
      else if (d.type === 'layer') store.commit('Move layer');
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
  svg.addEventListener('dblclick', e => {
    const t = document.elementFromPoint(e.clientX, e.clientY), h = hitOf(t);
    if (!t || !svg.contains(t) || (h && h.dataset.kind !== 'edge')) return;
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
