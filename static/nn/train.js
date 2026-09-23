// Training panel for the Net tab (docs/NN_CONTRACT.md), floating over #nn-stage.
//
// Settings live in net.meta.train. While playing, each animation frame runs up to `speed`
// model.trainStep calls inside a small time budget, then one store.touch(); pausing commits once.
// Plots: the input space (decision boundary for 2-input nets, fitted curve for 1-input nets),
// a hidden 2-neuron layer's activation space, and per-neuron heatmaps drawn into the nodes
// through ctx.view.setNodeImage (throttled).

import { colorFor, HI } from './store.js';

const LRS = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
const BATCHES = [1, 2, 4, 8, 10, 16, 32, 64, 128, 0];   // 0 = the whole dataset
const SPEEDS = [1, 2, 5, 10, 25, 100];                  // max training steps per frame
const GRID = 40;          // heatmap samples per side
const CURVE = 120;        // samples along a 1-D input
const LINES = 9;          // input grid lines carried into layer space (per direction)
const MAP_MS = 150;       // per-neuron heatmap throttle
const BUDGET_MS = 7;      // training time per frame
const HIST_MAX = 400;
const EXTRA = ['#3ec27a', '#b07cff', '#ff6fa8', '#2ec4c4', '#c9a227'];   // classes 3+ (0/1 use NEG/POS)
const BOUNDED = { sigmoid: [0, 1], softmax: [0, 1], tanh: [-1, 1] };
const UI_KEY = 'mathboard.nn.train';

// ---------------------------------------------------------------- pure helpers

export function netShape(net, model) {
  const L = net.layers.length;
  return {
    inputs: L ? model.nodesIn(net, 0).length : 0,
    outputs: L > 1 ? model.nodesIn(net, L - 1).length : 0,
  };
}

// Presets record their dataset in meta.train.dataset, which readSettings reads directly. Nets
// saved before that carry only the preset's title, so for them (and only when meta.train names
// no dataset) the title still picks the preset's dataset.
let legacyTitles = null;
function legacyPresetDataset(net, model) {
  if (!legacyTitles) {
    legacyTitles = new Map();
    for (const p of Object.values(model.PRESETS || {})) {
      if (!p.dataset) continue;
      try { legacyTitles.set(p.build(1).meta.title, p.dataset); } catch { /* skip */ }
    }
  }
  return legacyTitles.get(net.meta && net.meta.title);
}

// The dataset for a net whose meta.train.dataset is missing or unknown.
export function defaultDataset(net, model) {
  const all = model.DATASETS || {};
  const keys = Object.keys(all);
  const { inputs, outputs } = netShape(net, model);
  const fit = k => all[k] && all[k].inputs === inputs && all[k].outputs === outputs;
  const hint = legacyPresetDataset(net, model);
  return (fit(hint) && hint) || keys.find(fit) || (all.xor ? 'xor' : keys[0]);
}

const num = (v, d, lo = -Infinity, hi = Infinity) =>
  (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);

// net.meta.train with defaults filled in and junk repaired. Never mutates the net.
// A dataset recorded in meta.train (a preset's, or the user's choice) wins even when the net no
// longer fits it: the panel then offers "Adapt network" instead of silently switching.
export function readSettings(net, model) {
  const raw = (net.meta && net.meta.train) || {};
  return {
    dataset: (model.DATASETS || {})[raw.dataset] ? raw.dataset : defaultDataset(net, model),
    n: Math.round(num(raw.n, 200, 4, 5000)),
    noise: num(raw.noise, 0.1, 0, 10),
    seed: Math.round(num(raw.seed, 1)),
    lr: num(raw.lr, 0.1, 1e-7, 1000),
    batch: Math.round(num(raw.batch, 10, 0, 1e6)),
    speed: Math.round(num(raw.speed, 5, 1, 1000)),
    initSeed: Math.round(num(raw.initSeed, 1)),
    seen: Math.round(num(raw.seen, 0, 0)),
    steps: Math.round(num(raw.steps, 0, 0)),
    hist: Array.isArray(raw.hist) ? raw.hist.filter(Number.isFinite) : [],
    every: Math.round(num(raw.every, 1, 1)),
    maps: raw.maps !== false,
    space: typeof raw.space === 'string' ? raw.space : '',
  };
}

// Activations of every layer for n inputs at once, straight from model.matrices (the heatmaps
// need thousands of forward passes per frame). X is row-major n x size(layer `from`).
// Returns acts[l] (Float64Array n x size_l, null below `from`), or null when a skip edge
// bypasses layer `from` (then later layers are not a function of it alone).
export function forwardMany(net, model, X, n, from = 0, M = model.matrices(net)) {
  const L = net.layers.length;
  const sizes = [], pos = Object.create(null);
  for (let l = 0; l < L; l++) {
    const ns = model.nodesIn(net, l);
    sizes.push(ns.length);
    ns.forEach((nd, i) => { pos[nd.id] = i; });
  }
  const acts = new Array(L).fill(null);
  acts[from] = X;
  for (const m of [...M].sort((a, b) => a.l - b.l)) {
    const l = m.l;
    if (l <= from) continue;
    const R = m.rows.length;
    const Z = new Float64Array(n * R);
    for (let i = 0; i < R; i++) {
      const b = m.b[i] || 0;
      if (b) for (let s = 0; s < n; s++) Z[s * R + i] = b;
    }
    for (const term of m.terms) {
      if (term.k < from || !acts[term.k]) return null;
      const A = acts[term.k], C = sizes[term.k];
      const idx = term.cols.map(id => pos[id]);
      for (let i = 0; i < R; i++) {
        const w = term.W[i];
        for (let j = 0; j < idx.length; j++) {
          const wij = w[j];
          if (!wij) continue;
          const c = idx[j];
          for (let s = 0; s < n; s++) Z[s * R + i] += wij * A[s * C + c];
        }
      }
    }
    const act = net.layers[l].act;
    if (act === 'softmax') {
      for (let s = 0; s < n; s++) {
        const o = s * R;
        let mx = -Infinity, sum = 0;
        for (let i = 0; i < R; i++) mx = Math.max(mx, Z[o + i]);
        for (let i = 0; i < R; i++) { Z[o + i] = Math.exp(Z[o + i] - mx); sum += Z[o + i]; }
        for (let i = 0; i < R; i++) Z[o + i] /= sum;
      }
    } else {
      const f = model.ACTS && model.ACTS[act] && model.ACTS[act].f;
      if (f && act !== 'identity') for (let k = 0; k < Z.length; k++) Z[k] = f(Z[k]);
    }
    acts[l] = Z;
  }
  return acts;
}

// Mean loss over a dataset, per the contract's definitions (xent falls back to mse unless the
// output layer is softmax or sigmoid). P: n x K predictions, Y: rows of targets.
export function datasetLoss(P, Y, n, K, loss, outAct) {
  const eps = 1e-12;
  const xent = loss === 'xent' && (outAct === 'softmax' || outAct === 'sigmoid');
  let sum = 0;
  for (let s = 0; s < n; s++) {
    const y = Y[s], o = s * K;
    let e = 0;
    if (xent && outAct === 'softmax') {
      for (let k = 0; k < K; k++) e -= y[k] * Math.log(Math.max(P[o + k], eps));
    } else if (xent) {
      for (let k = 0; k < K; k++) {
        const a = Math.min(1 - eps, Math.max(eps, P[o + k]));
        e -= y[k] * Math.log(a) + (1 - y[k]) * Math.log(1 - a);
      }
      e /= K;
    } else {
      for (let k = 0; k < K; k++) { const d = P[o + k] - y[k]; e += d * d; }
      e = 0.5 * e / K;
    }
    sum += e;
  }
  return sum / n;
}

// Resize the input and output layers to fit a dataset; hidden layers are kept. New nodes are
// wired densely to the neighbouring layer with small Xavier weights. The output activation and
// the loss are set to suit the task. Returns a sentence describing the result.
export function adaptNet(net, model, ds, { seed = 1 } = {}) {
  const rand = model.rng(seed);
  const outAct = ds.kind === 'class' ? (ds.outputs > 1 ? 'softmax' : 'sigmoid') : 'identity';
  if (!net.layers.length) model.addLayer(net, 0, { name: 'input', act: 'identity', size: 0 });
  if (net.layers.length < 2) model.addLayer(net, 1, { name: 'output', act: outAct, size: 0 });
  const L = net.layers.length;
  resizeLayer(net, model, 0, ds.inputs, rand);
  resizeLayer(net, model, L - 1, ds.outputs, rand);
  model.setLayer(net, net.layers[L - 1].id, { act: outAct });
  net.meta = net.meta || {};
  net.meta.loss = ds.kind === 'class' ? 'xent' : 'mse';
  const sizes = net.layers.map((_, l) => model.nodesIn(net, l).length).join(' → ');
  return `Adapted to ${sizes}: ${outAct} output, ${net.meta.loss === 'xent' ? 'cross-entropy' : 'MSE'} loss`;
}

function resizeLayer(net, model, l, size, rand) {
  const layerId = net.layers[l].id;
  let nodes = model.nodesIn(net, l);
  if (nodes.length === size) return;
  const all = net.nodes;
  const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  const ys = nodes.map(n => n.y);
  const cy = ys.length ? mean(ys) : mean(all.map(n => n.y));
  const gap = ys.length > 1 ? Math.max(40, (Math.max(...ys) - Math.min(...ys)) / (ys.length - 1)) : 70;
  const x = nodes.length ? mean(nodes.map(n => n.x))
    : (all.length ? Math.max(...all.map(n => n.x)) + 160 : 0);
  while (nodes.length > size) {
    model.removeNode(net, nodes[nodes.length - 1].id);
    nodes = model.nodesIn(net, l);
  }
  while (nodes.length < size) {
    const y = nodes.length ? nodes[nodes.length - 1].y + gap : cy;
    const id = model.addNode(net, layerId, { x, y });
    const nb = l === 0 ? model.nodesIn(net, 1) : model.nodesIn(net, l - 1);
    const lim = Math.sqrt(6 / (nb.length + size));
    for (const o of nb) {
      const w = (rand() * 2 - 1) * lim;
      if (l === 0) model.connect(net, id, o.id, w); else model.connect(net, o.id, id, w);
    }
    nodes = model.nodesIn(net, l);
  }
  const dy = cy - mean(nodes.map(n => n.y));
  for (const n of nodes) n.y += dy;
}

// ---------------------------------------------------------------- small utilities

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const round4 = v => +v.toPrecision(4);
const fmtLoss = v => (Number.isNaN(v) ? 'NaN' : !Number.isFinite(v) ? '∞' : v === 0 ? '0' : v < 0.001 ? v.toExponential(2) : v.toFixed(4));
const hash3 = (a, b, c) => ((Math.imul(a | 0, 73856093) ^ Math.imul(b | 0, 19349663) ^ Math.imul(c | 0, 83492791)) >>> 0) || 1;

let probe = null;
function rgbOf(color) {
  if (!probe) {
    const c = document.createElement('canvas');
    c.width = c.height = 1;
    probe = c.getContext('2d', { willReadFrequently: true });
  }
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = '#000';
  probe.fillStyle = color;
  probe.fillRect(0, 0, 1, 1);
  const d = probe.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}
const css = rgb => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;

function sizeCanvas(c, w, h) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  if (c.style.height !== h + 'px') c.style.height = h + 'px';
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

// ---------------------------------------------------------------- the panel

export function install(ctx) {
  const { store, model } = ctx;
  const ro = !!ctx.audience;
  const stage = ctx.el.stage;   // train.css is linked by the shell
  if (getComputedStyle(stage).position === 'static') stage.style.position = 'relative';

  const ui = { open: true, fold: false, x: null, y: null };
  const savedUi = () => { try { return JSON.parse(localStorage.getItem(UI_KEY)) || {}; } catch { return {}; } };
  if (!ro) Object.assign(ui, savedUi());
  else {
    // The audience window mirrors the presenter's open / folded panel. Both windows share this
    // origin's localStorage, and 'storage' fires here whenever the presenter saves its UI state.
    const mirror = () => { const s = savedUi(); ui.open = s.open !== false; ui.fold = !!s.fold; };
    mirror();
    window.addEventListener('storage', e => {
      if (e.key !== UI_KEY && e.key !== null) return;
      mirror(); applyUi(); dirty = true; kick();
    });
  }
  const saveUi = () => { if (!ro) try { localStorage.setItem(UI_KEY, JSON.stringify(ui)); } catch { /* ignore */ } };

  const panel = document.createElement('div');
  panel.className = 'nn-train' + (ro ? ' ro' : '');
  panel.innerHTML = `
    <header class="nt-head">
      <button class="nt-fold" data-act="fold" title="Collapse / expand">&#9662;</button>
      <b class="nt-title">Train</b>
      <span class="nt-mini"></span>
      <button class="nt-go nt-go-mini" data-act="play" title="Play / pause training (Space)">&#9654;</button>
    </header>
    <div class="nt-body">
      <div class="nt-grid nt-g3">
        <label class="nt-span3">data <select data-k="dataset"></select></label>
        <label>points <input type="number" data-k="n" min="4" max="5000" step="10"></label>
        <label>noise <input type="number" data-k="noise" min="0" max="10" step="0.05"></label>
        <label>seed <input type="number" data-k="seed" step="1"></label>
      </div>
      <div class="nt-warn" hidden><span></span><button data-act="adapt" title="Resize the input and output layers to fit the dataset; hidden layers are kept">Adapt network</button></div>
      <div class="nt-grid nt-g4">
        <label>loss <select data-k="loss"><option value="mse">MSE</option><option value="xent">x-entropy</option></select></label>
        <label>rate <select data-k="lr"></select></label>
        <label>batch <select data-k="batch"></select></label>
        <label title="Maximum training steps per animation frame">speed <select data-k="speed"></select></label>
      </div>
      <div class="nt-hint" hidden></div>
      <div class="nt-ctl">
        <button class="nt-go" data-act="play" title="Play / pause training (Space)">&#9654; Play</button>
        <button data-act="step" title="One mini-batch gradient step (T)">Step</button>
        <button data-act="reset" title="Re-randomize the weights from the init seed">Reset</button>
        <label title="Seed for Reset">init <input type="number" data-k="initSeed" step="1"></label>
        <button data-act="dice" title="New random init seed, then reset">&#8635;</button>
      </div>
      <div class="nt-read">
        <span>epoch <b data-r="epoch">0</b></span>
        <span>step <b data-r="steps">0</b></span>
        <span>loss <b data-r="loss">&ndash;</b></span>
        <span data-r="accw" hidden>acc <b data-r="acc"></b></span>
      </div>
      <canvas class="nt-chart"></canvas>
      <div class="nt-grid nt-g2b">
        <label>plot <select data-k="space"></select></label>
        <label class="nt-check" title="Each neuron's activation over the input domain, drawn inside its node"><input type="checkbox" data-k="maps"> neuron maps</label>
      </div>
      <canvas class="nt-plot"></canvas>
      <div class="nt-cap"><span class="nt-axes"></span><span class="nt-info"></span></div>
    </div>`;
  stage.appendChild(panel);

  const $ = s => panel.querySelector(s);
  const K = k => panel.querySelector(`[data-k="${k}"]`);
  const R = k => panel.querySelector(`[data-r="${k}"]`);
  const chart = $('.nt-chart'), plot = $('.nt-plot');
  const warn = $('.nt-warn'), hint = $('.nt-hint'), mini = $('.nt-mini');
  const axesEl = $('.nt-axes'), infoEl = $('.nt-info');
  const goBtns = [...panel.querySelectorAll('.nt-go')], stepBtn = $('[data-act="step"]');
  const HINT = ro ? '' : 'click a point to load it as the current sample';

  K('dataset').innerHTML = Object.entries(model.DATASETS || {})
    .map(([k, v]) => `<option value="${k}">${esc(v.label || k)} (${v.inputs}&rarr;${v.outputs})</option>`).join('');
  K('lr').innerHTML = LRS.map(v => `<option value="${v}">${v}</option>`).join('');
  K('batch').innerHTML = BATCHES.map(v => `<option value="${v}">${v || 'all'}</option>`).join('');
  K('speed').innerHTML = SPEEDS.map(v => `<option value="${v}">${v}&times;</option>`).join('');
  if (ro) panel.querySelectorAll('input, select, .nt-body button, .nt-go').forEach(el => { el.disabled = true; });

  // Keep pointer and wheel input on the panel away from the stage (pan, deselect, ...).
  for (const type of ['pointerdown', 'mousedown', 'click', 'dblclick', 'wheel', 'contextmenu', 'touchstart']) {
    panel.addEventListener(type, e => e.stopPropagation(), { passive: type === 'wheel' || type === 'touchstart' });
  }

  // ---------------------------------------------------------------- state

  let running = false, trained = false, inTick = false;
  let raf = 0, dirty = true, mapsDirty = true, lastMaps = -1e9, mapTimer = 0;
  let data = null, order = null, P = null, lastEval = null;
  let pts = null, spaceSig = null, axesKey = '', lsBox = null, chartKey = '';
  const mapUrls = new Map();

  // Cheap checks only: this runs every frame, and reading layout here would force a reflow.
  const shown = () => document.body.dataset.view === 'nn' && !ctx.el.root.hidden;
  const size = { chart: 0, plot: 0 };
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };

  function palette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, d) => cs.getPropertyValue(name).trim() || d;
    const theme = ctx.theme ? ctx.theme() : (document.documentElement.dataset.theme || 'dark');
    const pos = rgbOf(colorFor(1, 1, theme)), neg = rgbOf(colorFor(-1, 1, theme));
    const bg = v('--bg', theme === 'light' ? '#fbfbf8' : '#1d2327');
    return {
      theme, bg, bgRGB: rgbOf(bg), pos, neg,
      fg: v('--ui-fg', '#e6e6e0'), muted: v('--ui-muted', '#9aa3a8'),
      line: v('--ui-line', 'rgba(255,255,255,0.1)'), accent: v('--accent', '#5ac8fa'),
      cls: [neg, pos, ...EXTRA.map(rgbOf)],
      stroke: theme === 'light' ? 'rgba(20,20,20,0.85)' : 'rgba(255,255,255,0.9)',
    };
  }
  const pal = () => (P || (P = palette()));

  // ---------------------------------------------------------------- data

  function getData(t) {
    const key = [t.dataset, t.n, t.noise, t.seed].join('|');
    if (data && data.key === key) return data;
    hover = -1;   // the hovered index belonged to the old samples
    const ds = model.DATASETS[t.dataset];
    const { X, Y } = ds.make(t.n, t.seed, t.noise);
    const n = X.length, dim = ds.inputs, Kout = ds.outputs;
    const Xf = new Float64Array(n * dim);
    X.forEach((x, s) => { for (let j = 0; j < dim; j++) Xf[s * dim + j] = x[j]; });
    let lo = Infinity, hi = -Infinity;
    for (const y of Y) { lo = Math.min(lo, y[0]); hi = Math.max(hi, y[0]); }
    const cls = new Int32Array(n);
    const mid = (lo + hi) / 2, half = (hi - lo) / 2 || 1;
    Y.forEach((y, s) => {
      if (y.length > 1) { let b = 0; for (let k = 1; k < y.length; k++) if (y[k] > y[b]) b = k; cls[s] = b; }
      else cls[s] = y[0] > mid ? 1 : 0;
    });
    // Plot domain: square box around the points (2-D), padded range (1-D).
    let dom;
    if (dim === 2) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const x of X) { x0 = Math.min(x0, x[0]); x1 = Math.max(x1, x[0]); y0 = Math.min(y0, x[1]); y1 = Math.max(y1, x[1]); }
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, h = (Math.max(x1 - x0, y1 - y0) / 2 || 1) * 1.1;
      dom = { x0: cx - h, x1: cx + h, y0: cy - h, y1: cy + h };
    } else if (dim === 1) {
      let x0 = Infinity, x1 = -Infinity;
      for (const x of X) { x0 = Math.min(x0, x[0]); x1 = Math.max(x1, x[0]); }
      const px = (x1 - x0 || 1) * 0.06, py = (hi - lo || 1) * 0.18;
      dom = { x0: x0 - px, x1: x1 + px, y0: lo - py, y1: hi + py };
    }
    data = { key, ds, X, Y, Xf, n, dim, K: Kout, cls, mid, half, dom, kind: ds.kind, grid: null, curve: null, lines: null };
    if (dom && dim === 2) {
      const g = new Float64Array(GRID * GRID * 2);
      for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
        const s = r * GRID + c;
        g[2 * s] = dom.x0 + (c + 0.5) / GRID * (dom.x1 - dom.x0);
        g[2 * s + 1] = dom.y1 - (r + 0.5) / GRID * (dom.y1 - dom.y0);
      }
      data.grid = g;
      const per = 48, ln = new Float64Array(2 * LINES * per * 2);
      let o = 0;
      for (let dir = 0; dir < 2; dir++) for (let i = 0; i < LINES; i++) for (let j = 0; j < per; j++) {
        const u = dom.x0 + i / (LINES - 1) * (dom.x1 - dom.x0), w = dom.x0 + j / (per - 1) * (dom.x1 - dom.x0);
        const a = dom.y0 + i / (LINES - 1) * (dom.y1 - dom.y0), b = dom.y0 + j / (per - 1) * (dom.y1 - dom.y0);
        ln[o++] = dir ? w : u;
        ln[o++] = dir ? a : b;
      }
      data.lines = { X: ln, count: 2 * LINES, per };
    }
    if (dom && dim === 1) {
      const c = new Float64Array(CURVE);
      for (let i = 0; i < CURVE; i++) c[i] = dom.x0 + i / (CURVE - 1) * (dom.x1 - dom.x0);
      data.curve = c;
      data.lines = { X: c, count: 1, per: CURVE };
    }
    order = null;
    return data;
  }

  const fits = (net, ds) => {
    const sh = netShape(net, model);
    return !!ds && ds.inputs === sh.inputs && ds.outputs === sh.outputs;
  };

  function orderFor(d, t, epoch) {
    const key = d.key + '|' + t.initSeed + '|' + epoch;
    if (order && order.key === key) return order.idx;
    const idx = new Int32Array(d.n);
    for (let i = 0; i < d.n; i++) idx[i] = i;
    const r = model.rng(hash3(t.seed, t.initSeed, epoch));
    for (let i = d.n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
    order = { key, idx };
    return idx;
  }

  // Loss (and accuracy for classification) over the whole dataset at the current weights.
  function evaluate(net, d, A) {
    const L = net.layers.length;
    const acts = A || forwardMany(net, model, d.Xf, d.n);
    const out = acts && acts[L - 1];
    if (!out) return null;
    const loss = datasetLoss(out, d.Y, d.n, d.K, net.meta?.loss || 'mse', net.layers[L - 1].act);
    let acc = null;
    if (d.kind === 'class') {
      let hit = 0;
      for (let s = 0; s < d.n; s++) {
        let c;
        if (d.K > 1) { c = 0; for (let k = 1; k < d.K; k++) if (out[s * d.K + k] > out[s * d.K + c]) c = k; }
        else c = out[s] > d.mid ? 1 : 0;
        if (c === d.cls[s]) hit++;
      }
      acc = hit / d.n;
    }
    return { loss, acc };
  }

  // ---------------------------------------------------------------- training

  function trainSteps(net, t, d, max, budget = Infinity) {
    const start = performance.now();
    const loss = net.meta?.loss || 'mse';
    if (!t.hist.length) { const ev = evaluate(net, d); if (ev) t.hist.push(round4(ev.loss)); }
    let k = 0, bad = false;
    while (k < max && (k === 0 || performance.now() - start < budget)) {
      const e0 = Math.floor(t.seen / d.n), c = t.seen % d.n;
      const bs = t.batch > 0 ? Math.min(t.batch, d.n) : d.n;
      const idx = orderFor(d, t, e0).subarray(c, Math.min(c + bs, d.n));
      const X = new Array(idx.length), Y = new Array(idx.length);
      for (let j = 0; j < idx.length; j++) { X[j] = d.X[idx[j]]; Y[j] = d.Y[idx[j]]; }
      const l = model.trainStep(net, { X, Y }, { lr: t.lr, loss });
      t.seen += idx.length;
      t.steps++;
      k++;
      trained = true;
      if (!Number.isFinite(l)) { bad = true; break; }
      const e1 = Math.floor(t.seen / d.n);
      if (e1 > e0 && e1 % t.every === 0) {
        const ev = evaluate(net, d);
        if (ev && Number.isFinite(ev.loss)) t.hist.push(round4(ev.loss));
        if (t.hist.length > HIST_MAX) { t.hist = t.hist.filter((_, i) => i % 2 === 0); t.every *= 2; }
      }
    }
    return !bad;
  }

  // Settings object written into the net (training ticks mutate it in place).
  function live(net) {
    net.meta = net.meta || {};
    const t = readSettings(net, model);
    net.meta.train = t;
    return t;
  }

  function trainFrame() {
    const net = store.net;
    const t0 = readSettings(net, model);
    const ds = model.DATASETS[t0.dataset];
    if (!fits(net, ds)) { pause(); return; }
    const t = live(net), d = getData(t);
    const ok = trainSteps(net, t, d, t.speed, BUDGET_MS);
    inTick = true;
    try { store.touch(); } finally { inTick = false; }
    if (!ok) { pause(); ctx.toast?.('The loss blew up: lower the learning rate, then Reset'); }
  }

  function emitTrain() {
    const net = store.net, t = readSettings(net, model);
    const n = data ? data.n : t.n;
    store.emit('train', { epoch: Math.floor(t.seen / n), loss: lastEval ? lastEval.loss : null, running });
  }

  function setRunning(v) {
    running = v;
    panel.classList.toggle('running', v);
    for (const b of goBtns) b.innerHTML = b.classList.contains('nt-go-mini') ? (v ? '&#10074;&#10074;' : '&#9654;') : (v ? '&#10074;&#10074; Pause' : '&#9654; Play');
  }

  function play() {
    if (ro || running) return;
    const net = store.net, t = readSettings(net, model);
    if (!fits(net, model.DATASETS[t.dataset])) { ctx.toast?.('The network does not fit this dataset: Adapt network first'); return; }
    trained = false;
    setRunning(true);
    emitTrain();
    kick();
  }

  function pause() {
    if (!running) return;
    setRunning(false);
    if (trained) { trained = false; store.commit('Train'); }
    emitTrain();
  }

  const toggle = () => (running ? pause() : play());

  function step() {
    if (ro) return;
    if (running) pause();
    const net = store.net;
    if (!fits(net, model.DATASETS[readSettings(net, model).dataset])) { ctx.toast?.('The network does not fit this dataset: Adapt network first'); return; }
    const t = live(net), d = getData(t);
    const ok = trainSteps(net, t, d, 1);
    trained = false;
    store.commit('Training step');
    if (!ok) ctx.toast?.('The loss blew up: lower the learning rate, then Reset');
    emitTrain();
  }

  function reset(seed) {
    if (ro) return;
    if (running) pause();
    const net = store.net, t = live(net);
    if (seed != null) t.initSeed = seed;
    const acts = net.layers.slice(1).map(l => l.act);
    model.randomize(net, { seed: t.initSeed, scheme: acts.some(a => a === 'relu' || a === 'leaky') ? 'he' : 'xavier' });
    Object.assign(t, { seen: 0, steps: 0, hist: [], every: 1 });
    const ds = model.DATASETS[t.dataset];
    if (fits(net, ds)) { const ev = evaluate(net, getData(t)); if (ev && Number.isFinite(ev.loss)) t.hist.push(round4(ev.loss)); }
    store.commit('Reset weights');
    emitTrain();
  }

  function adapt() {
    if (ro) return;
    if (running) pause();
    const net = store.net, t = readSettings(net, model), ds = model.DATASETS[t.dataset];
    if (!ds) return;
    const msg = adaptNet(net, model, ds, { seed: t.initSeed });
    net.meta.train = { ...t, seen: 0, steps: 0, hist: [], every: 1 };
    store.commit('Adapt network');
    try { ctx.view?.fit?.(300); } catch { /* optional */ }
    ctx.toast?.(msg);
  }

  function loadSample(i) {
    const net = store.net, t = readSettings(net, model), ds = model.DATASETS[t.dataset];
    const sh = netShape(net, model);
    if (!ds || ds.inputs !== sh.inputs) { ctx.toast?.('The network does not fit this dataset: Adapt network first'); return; }
    const d = getData(t), L = net.layers.length;
    model.nodesIn(net, 0).forEach((nd, j) => model.setNode(net, nd.id, { value: d.X[i][j] }));
    if (ds.outputs === sh.outputs) model.nodesIn(net, L - 1).forEach((nd, j) => model.setNode(net, nd.id, { target: d.Y[i][j] }));
    if (!running) store.commit(`Load sample ${i + 1}`);
  }

  function writeSettings(patch, label) {
    const net = store.net;
    net.meta = net.meta || {};
    net.meta.train = { ...readSettings(net, model), ...patch };
    store.commit(label);
  }

  // ---------------------------------------------------------------- controls

  function setVal(el, v) {
    if (!el || el === document.activeElement) return;
    if (el.type === 'checkbox') { el.checked = !!v; return; }
    const s = String(v);
    if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value === s)) el.add(new Option(s, s));
    if (el.value !== s) el.value = s;
  }

  function spaceLayers(net) {
    const L = net.layers.length, out = [];
    for (let l = 1; l < L - 1; l++) if (model.nodesIn(net, l).length === 2) out.push(l);
    return out;
  }

  function syncControls() {
    const net = store.net, t = readSettings(net, model);
    for (const k of ['dataset', 'n', 'noise', 'seed', 'lr', 'batch', 'speed', 'initSeed', 'maps']) setVal(K(k), t[k]);
    setVal(K('loss'), net.meta?.loss === 'xent' ? 'xent' : 'mse');
    const list = spaceLayers(net);
    const sig = list.map(l => net.layers[l].id + ':' + net.layers[l].name).join(',');
    const sel = K('space');
    if (sig !== spaceSig) {
      spaceSig = sig;
      sel.innerHTML = '<option value="">input space</option>' + list.map(l =>
        `<option value="${esc(net.layers[l].id)}">layer ${l}${net.layers[l].name ? ': ' + esc(net.layers[l].name) : ''}</option>`).join('');
      sel.title = list.length ? 'Scatter the data in a 2-neuron hidden layer\'s activation space'
        : 'Add a hidden layer with exactly 2 neurons to see its activation space';
    }
    setVal(sel, list.some(l => net.layers[l].id === t.space) ? t.space : '');
  }

  // A focused select or checkbox would swallow Space (play/pause): let go of it once it's used.
  panel.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.target.matches('input[type="number"]')) e.target.blur();
  });
  panel.addEventListener('change', e => {
    const el = e.target, k = el.dataset.k;
    if (!k || ro) return;
    if (el.tagName === 'SELECT' || el.type === 'checkbox') el.blur();
    if (k === 'loss') {
      store.net.meta = store.net.meta || {};
      store.net.meta.loss = el.value;
      store.commit('Loss');
      return;
    }
    let v = el.type === 'checkbox' ? el.checked : el.value;
    if (!['dataset', 'space', 'maps'].includes(k)) {
      v = Number(v);
      if (!Number.isFinite(v)) { syncControls(); return; }
    }
    const patch = { [k]: v };
    if (['dataset', 'n', 'noise', 'seed'].includes(k)) Object.assign(patch, { seen: 0, steps: 0, hist: [], every: 1 });
    if (k === 'maps' && !v) clearMaps();
    writeSettings(patch, 'Training settings');
    syncControls();
    if (['dataset', 'n', 'noise', 'seed'].includes(k) && running && !fits(store.net, model.DATASETS[readSettings(store.net, model).dataset])) pause();
  });

  panel.addEventListener('click', e => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const act = b.dataset.act;
    if (act === 'fold') { ui.fold = !ui.fold; saveUi(); applyUi(); dirty = true; kick(); }
    if (ro) return;
    if (act === 'play') toggle();
    else if (act === 'step') step();
    else if (act === 'reset') reset();
    else if (act === 'dice') { const s = 1 + Math.floor(Math.random() * 9999); reset(s); }
    else if (act === 'adapt') adapt();
    b.blur();   // so Space goes to the shortcut, not the focused button
  });

  // Drag the panel by its header.
  const head = $('.nt-head');
  head.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest('button')) return;
    const sr = stage.getBoundingClientRect(), pr = panel.getBoundingClientRect();
    const dx = e.clientX - pr.left, dy = e.clientY - pr.top;
    head.setPointerCapture(e.pointerId);
    const move = ev => {
      ui.x = clamp(ev.clientX - sr.left - dx, 0, Math.max(0, sr.width - pr.width));
      ui.y = clamp(ev.clientY - sr.top - dy, 0, Math.max(0, sr.height - 40));
      applyUi();
    };
    const up = () => { head.removeEventListener('pointermove', move); head.removeEventListener('pointerup', up); head.removeEventListener('pointercancel', up); saveUi(); };
    head.addEventListener('pointermove', move);
    head.addEventListener('pointerup', up);
    head.addEventListener('pointercancel', up);
  });
  head.addEventListener('dblclick', e => {
    if (e.target.closest('button')) return;
    ui.x = ui.y = null; saveUi(); applyUi();
  });

  function applyUi() {
    panel.hidden = !ui.open;
    panel.classList.toggle('folded', !!ui.fold);
    $('.nt-fold').innerHTML = ui.fold ? '&#9656;' : '&#9662;';
    if (ui.x == null || ui.y == null) { panel.style.left = ''; panel.style.top = ''; panel.style.right = ''; }
    else { panel.style.left = ui.x + 'px'; panel.style.top = ui.y + 'px'; panel.style.right = 'auto'; }
    if (toolBtn) toolBtn.classList.toggle('on', !!ui.open);
  }

  let toolBtn = null;
  if (!ro && ctx.addButton) {
    try {
      toolBtn = ctx.addButton({
        label: 'Train', group: 'train',
        title: 'Training panel (Space: play/pause, T: one step)',
        onClick: () => { ui.open = !ui.open; saveUi(); applyUi(); dirty = true; kick(); },
      });
    } catch (err) { console.warn('[nn/train] addButton:', err); }
  }
  applyUi();

  window.addEventListener('keydown', e => {
    if (ro || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code !== 'Space' && e.code !== 'KeyT') return;
    if (ctx.active && !ctx.active(e)) return;
    if (e.target && e.target.closest && e.target.closest('input[type="number"], input[type="text"], textarea, select, [contenteditable=""], [contenteditable="true"]')) return;
    if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) toggle(); }
    else if (!e.shiftKey) { e.preventDefault(); step(); }
  });

  // ---------------------------------------------------------------- store events

  store.on('net', payload => {
    if (payload && payload.structural) { mapUrls.clear(); lsBox = null; }
    if (!inTick) syncControls();
  });
  store.on('values', () => { dirty = true; mapsDirty = true; kick(); });
  ctx.onTheme?.(() => { P = null; mapUrls.clear(); dirty = true; mapsDirty = true; kick(); });
  ctx.onShow?.(v => { if (v) { dirty = true; mapsDirty = true; kick(); } });
  const sizer = new ResizeObserver(() => {
    if (size.chart === chart.clientWidth && size.plot === plot.clientWidth) return;
    size.chart = chart.clientWidth;
    size.plot = plot.clientWidth;
    chartKey = '';
    dirty = true;
    kick();
  });
  sizer.observe(chart);
  sizer.observe(plot);

  // ---------------------------------------------------------------- rendering

  function frame() {
    raf = 0;
    const ticking = running;
    if (running) { trainFrame(); if (running) kick(); }
    if (dirty || mapsDirty) render();
    if (ticking && running) emitTrain();   // after render, so it carries this frame's loss
  }

  function render() {
    if (!shown()) return;
    const net = store.net, t = readSettings(net, model), sh = netShape(net, model);
    const ds = model.DATASETS[t.dataset];
    const d = ds ? getData(t) : null;
    const inOk = !!d && ds.inputs === sh.inputs && sh.outputs > 0;
    const ok = inOk && ds.outputs === sh.outputs;
    let M = null;
    if (inOk) try { M = model.matrices(net); } catch (err) { console.warn('[nn/train] matrices:', err); }
    const A = M ? forwardMany(net, model, d.Xf, d.n, 0, M) : null;
    let gridActs;
    const grid = () => (gridActs !== undefined ? gridActs
      : (gridActs = M && d.grid ? forwardMany(net, model, d.grid, GRID * GRID, 0, M) : null));
    if (dirty) {
      dirty = false;
      lastEval = ok && A ? evaluate(net, d, A) : null;
      status(net, t, sh, ds, d, ok);
      if (ui.open && !ui.fold) {
        drawChart(t);
        drawPlot(net, t, d, sh, M, A, ok, inOk, grid);
      }
    }
    if (mapsDirty) {
      const now = performance.now(), wait = MAP_MS - (now - lastMaps);
      if (wait <= 0) {
        mapsDirty = false;
        lastMaps = now;
        drawMaps(net, t, d, inOk && M ? M : null, grid);
      } else if (!mapTimer) {
        mapTimer = setTimeout(() => { mapTimer = 0; kick(); }, wait + 1);
      }
    }
  }

  function status(net, t, sh, ds, d, ok) {
    const epoch = d ? Math.floor(t.seen / d.n) : 0;
    R('epoch').textContent = epoch;
    R('steps').textContent = t.steps;
    R('loss').textContent = lastEval ? fmtLoss(lastEval.loss) : '–';
    const acc = lastEval && lastEval.acc != null;
    R('accw').hidden = !acc;
    if (acc) R('acc').textContent = Math.round(lastEval.acc * 100) + '%';
    mini.textContent = `${ro && ds ? (ds.label || t.dataset) + ' · ' : ''}epoch ${epoch} · ${lastEval ? fmtLoss(lastEval.loss) : '–'}`;
    const mis = !!ds && !ok;
    warn.hidden = !mis;
    if (mis) {
      warn.firstChild.textContent = `This net is ${sh.inputs} → … → ${sh.outputs}; ${ds.label || t.dataset} needs ${ds.inputs} → … → ${ds.outputs}. `;
    }
    for (const b of goBtns) b.disabled = ro || !ok;
    stepBtn.disabled = ro || !ok;
    const L = net.layers.length, outAct = L ? net.layers[L - 1].act : '';
    let h = '';
    if (net.meta?.loss === 'xent' && outAct !== 'softmax' && outAct !== 'sigmoid') h = 'Cross-entropy needs a sigmoid or softmax output layer; MSE is used instead.';
    else if (outAct === 'softmax' && sh.outputs === 1) h = 'Softmax over one neuron is always 1: use sigmoid for a single output.';
    if (hint.textContent !== h) hint.textContent = h;
    hint.hidden = !h;
  }

  function drawChart(t) {
    const w = size.chart, hs = t.hist;
    const key = [w, hs.length, hs[hs.length - 1], hs[0], t.every, pal().theme].join('|');
    if (!w || key === chartKey) return;
    chartKey = key;
    const h = 56, p = pal(), g = sizeCanvas(chart, w, h);
    g.clearRect(0, 0, w, h);
    g.font = '10px system-ui, "Segoe UI", sans-serif';
    if (hs.length < 2) {
      g.fillStyle = p.muted;
      g.fillText(hs.length ? `loss ${fmtLoss(hs[0])} at epoch 0; press Play` : 'loss per epoch', 6, h / 2 + 3);
      return;
    }
    let max = 0;
    for (const v of hs) max = Math.max(max, v);
    max = max || 1;
    const pad = 4, top = 13;
    const X = i => pad + i / (hs.length - 1) * (w - 2 * pad), Y = v => h - pad - v / max * (h - top - pad);
    g.strokeStyle = p.line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, h - pad + 0.5); g.lineTo(w - pad, h - pad + 0.5); g.stroke();
    g.beginPath();
    hs.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.strokeStyle = p.accent; g.lineWidth = 1.6; g.lineJoin = 'round'; g.stroke();
    g.fillStyle = p.muted;
    g.textAlign = 'left'; g.fillText(fmtLoss(max), pad, 9);
    g.textAlign = 'right'; g.fillText(`epoch ${(hs.length - 1) * t.every}`, w - pad, 9);
    g.textAlign = 'left';
  }

  // Paint a GRID x GRID output field (n x K values) into the plot.
  let tile = null;
  function heat(g, S, out, Kout, d, amax) {
    if (!tile) { const c = document.createElement('canvas'); c.width = c.height = GRID; tile = c.getContext('2d'); }
    const p = pal(), bg = p.bgRGB, img = tile.createImageData(GRID, GRID), px = img.data;
    for (let s = 0; s < GRID * GRID; s++) {
      let rgb, a;
      if (Kout === 1) {
        const v = clamp((out[s] - d.mid) / d.half, -1, 1);
        rgb = v >= 0 ? p.pos : p.neg;
        a = 0.05 + amax * Math.abs(v);
      } else {
        let b = 0, b2 = -1;
        for (let k = 1; k < Kout; k++) {
          if (out[s * Kout + k] > out[s * Kout + b]) { b2 = b; b = k; } else if (b2 < 0 || out[s * Kout + k] > out[s * Kout + b2]) b2 = k;
        }
        rgb = p.cls[b % p.cls.length];
        a = 0.05 + amax * clamp(out[s * Kout + b] - out[s * Kout + b2], 0, 1);
      }
      const o = 4 * s;
      px[o] = bg[0] + (rgb[0] - bg[0]) * a;
      px[o + 1] = bg[1] + (rgb[1] - bg[1]) * a;
      px[o + 2] = bg[2] + (rgb[2] - bg[2]) * a;
      px[o + 3] = 255;
    }
    tile.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(tile.canvas, 0, 0, S, S);
  }

  function axes(g, S, x0px, y0px) {
    const p = pal();
    g.strokeStyle = p.muted; g.globalAlpha = 0.35; g.lineWidth = 1;
    g.beginPath();
    if (x0px > 0 && x0px < S) { g.moveTo(Math.round(x0px) + 0.5, 0); g.lineTo(Math.round(x0px) + 0.5, S); }
    if (y0px > 0 && y0px < S) { g.moveTo(0, Math.round(y0px) + 0.5); g.lineTo(S, Math.round(y0px) + 0.5); }
    g.stroke();
    g.globalAlpha = 1;
  }

  // Range labels: x0 and x1 along the bottom edge, y1 and y0 down the left edge (y0 just above
  // x0). The box is rarely centred on 0, so the bottom-left corner needs both of its values.
  function ticks(g, S, dom) {
    const p = pal(), f = v => (model.fmt ? model.fmt(v, 1) : v.toFixed(1));
    g.font = '10px system-ui, "Segoe UI", sans-serif';
    g.fillStyle = p.muted;
    g.textAlign = 'left';
    g.fillText(f(dom.x0), 3, S - 3);
    g.fillText(f(dom.y0), 3, S - 15);
    g.fillText(f(dom.y1), 3, 11);
    g.textAlign = 'right'; g.fillText(f(dom.x1), S - 3, S - 3);
    g.textAlign = 'left';
  }

  function points(g, d, at, hl) {
    const p = pal(), r = d.n > 400 ? 2.2 : 3.2;
    pts = new Float32Array(d.n * 2);
    for (let s = 0; s < d.n; s++) {
      const [x, y] = at(s);
      pts[2 * s] = x; pts[2 * s + 1] = y;
    }
    // One path per class: a few draw calls instead of two per point.
    const groups = d.kind === 'class' ? p.cls.length : 1;
    g.lineWidth = 1;
    g.strokeStyle = p.stroke;
    for (let c = 0; c < groups; c++) {
      g.beginPath();
      let any = false;
      for (let s = 0; s < d.n; s++) {
        if (groups > 1 && d.cls[s] % groups !== c) continue;
        const x = pts[2 * s], y = pts[2 * s + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        g.moveTo(x + r, y);
        g.arc(x, y, r, 0, Math.PI * 2);
        any = true;
      }
      if (!any) continue;
      g.fillStyle = groups > 1 ? css(p.cls[c]) : p.fg;
      g.fill();
      if (groups > 1) g.stroke();
    }
    if (hl >= 0 && hl < d.n) ring(g, pts[2 * hl], pts[2 * hl + 1], 6, p.fg);
  }

  function ring(g, x, y, r = 7, color = HI) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.strokeStyle = color; g.lineWidth = 2.2;
    g.stroke();
  }

  let hover = -1;
  function drawPlot(net, t, d, sh, M, A, ok, inOk, grid) {
    const S = size.plot;
    pts = null;
    if (!S) return;
    const p = pal(), g = sizeCanvas(plot, S, S);
    g.fillStyle = p.bg;
    g.fillRect(0, 0, S, S);
    if (!d || !d.dom) {
      g.fillStyle = p.muted; g.font = '12px system-ui, sans-serif';
      g.fillText('Plots need a dataset with 1 or 2 inputs', 10, S / 2);
      setAxes('', []);
      return;
    }
    const L = net.layers.length;
    const lay = inOk && A ? spaceLayers(net).find(l => net.layers[l].id === t.space) : undefined;
    if (lay !== undefined) drawLayerSpace(g, S, net, d, M, A, lay, ok);
    else if (d.dim === 2) {
      const { x0, x1, y0, y1 } = d.dom;
      const sx = x => (x - x0) / (x1 - x0) * S, sy = y => S - (y - y0) / (y1 - y0) * S;
      const G = ok ? grid() : null;
      if (G) heat(g, S, G[L - 1], d.K, d, 0.62);
      axes(g, S, sx(0), sy(0));
      ticks(g, S, d.dom);
      points(g, d, s => [sx(d.X[s][0]), sy(d.X[s][1])], hover);
      if (inOk) {
        const ins = model.nodesIn(net, 0);
        ring(g, sx(ins[0].value), sy(ins[1].value));
      }
      setAxes('in', inOk ? model.nodesIn(net, 0).map(n => n.label) : ['x_{1}', 'x_{2}']);
    } else {
      const { x0, x1, y0, y1 } = d.dom;
      const sx = x => (x - x0) / (x1 - x0) * S, sy = y => S - (y - y0) / (y1 - y0) * S;
      axes(g, S, sx(0), sy(0));
      ticks(g, S, d.dom);
      points(g, d, s => [sx(d.X[s][0]), sy(d.Y[s][0])], hover);
      if (ok) {
        const C = forwardMany(net, model, d.curve, CURVE, 0, M);
        const out = C && C[L - 1];
        if (out) {
          for (let k = 0; k < d.K; k++) {
            g.beginPath();
            for (let i = 0; i < CURVE; i++) {
              const X = sx(d.curve[i]), Y = clamp(sy(out[i * d.K + k]), -S, 2 * S);
              i ? g.lineTo(X, Y) : g.moveTo(X, Y);
            }
            g.strokeStyle = d.K > 1 ? css(p.cls[k % p.cls.length]) : p.accent;
            g.lineWidth = 2.4; g.lineJoin = 'round';
            g.stroke();
          }
        }
      }
      if (inOk) {
        const xin = model.nodesIn(net, 0)[0].value;
        g.setLineDash([4, 4]); g.strokeStyle = HI; g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(sx(xin), 0); g.lineTo(sx(xin), S); g.stroke();
        g.setLineDash([]);
        const outs = L > 1 ? model.nodesIn(net, L - 1) : [];
        if (outs[0] && typeof outs[0].target === 'number') ring(g, sx(xin), sy(outs[0].target));
      }
      const outs = L > 1 ? model.nodesIn(net, L - 1) : [];
      setAxes('in', [inOk ? model.nodesIn(net, 0)[0].label : 'x_{1}', ok && outs[0] ? outs[0].label : 'y']);
    }
  }

  function drawLayerSpace(g, S, net, d, M, A, l, ok) {
    const p = pal(), L = net.layers.length, H = A[l];
    const ly = net.layers[l], nodes = model.nodesIn(net, l);
    let x0, x1, y0, y1;
    const b = BOUNDED[ly.act];
    if (b) {
      const pd = (b[1] - b[0]) * 0.06;
      x0 = y0 = b[0] - pd; x1 = y1 = b[1] + pd;
    } else {
      let a0 = 0, a1 = 0, c0 = 0, c1 = 0;
      for (let s = 0; s < d.n; s++) {
        const u = H[2 * s], v = H[2 * s + 1];
        if (Number.isFinite(u)) { a0 = Math.min(a0, u); a1 = Math.max(a1, u); }
        if (Number.isFinite(v)) { c0 = Math.min(c0, v); c1 = Math.max(c1, v); }
      }
      const cx = (a0 + a1) / 2, cy = (c0 + c1) / 2, h = (Math.max(a1 - a0, c1 - c0) / 2 || 1) * 1.12;
      const want = { x0: cx - h, x1: cx + h, y0: cy - h, y1: cy + h };
      if (!lsBox || lsBox.id !== ly.id || !running) lsBox = { id: ly.id, ...want };
      else {
        // Grow at once, shrink slowly: the view doesn't jitter while training.
        const k = 0.08;
        lsBox.x0 = want.x0 < lsBox.x0 ? want.x0 : lsBox.x0 + (want.x0 - lsBox.x0) * k;
        lsBox.y0 = want.y0 < lsBox.y0 ? want.y0 : lsBox.y0 + (want.y0 - lsBox.y0) * k;
        lsBox.x1 = want.x1 > lsBox.x1 ? want.x1 : lsBox.x1 + (want.x1 - lsBox.x1) * k;
        lsBox.y1 = want.y1 > lsBox.y1 ? want.y1 : lsBox.y1 + (want.y1 - lsBox.y1) * k;
        const s = Math.max(lsBox.x1 - lsBox.x0, lsBox.y1 - lsBox.y0) / 2;
        const mx = (lsBox.x0 + lsBox.x1) / 2, my = (lsBox.y0 + lsBox.y1) / 2;
        Object.assign(lsBox, { x0: mx - s, x1: mx + s, y0: my - s, y1: my + s });
      }
      ({ x0, x1, y0, y1 } = lsBox);
    }
    const sx = x => (x - x0) / (x1 - x0) * S, sy = y => S - (y - y0) / (y1 - y0) * S;
    let note = '';
    if (ok) {
      const gx = new Float64Array(GRID * GRID * 2);
      for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
        const s = r * GRID + c;
        gx[2 * s] = x0 + (c + 0.5) / GRID * (x1 - x0);
        gx[2 * s + 1] = y1 - (r + 0.5) / GRID * (y1 - y0);
      }
      const G = forwardMany(net, model, gx, GRID * GRID, l, M);
      if (G) heat(g, S, G[L - 1], d.K, d, 0.62);
      else note = 'skip edges bypass this layer: no output field';
    }
    axes(g, S, sx(0), sy(0));
    // The input domain's grid lines, carried into this layer: how the net bends space.
    const W = d.lines ? forwardMany(net, model, d.lines.X, d.lines.count * d.lines.per, 0, M) : null;
    if (W && W[l]) {
      const V = W[l];
      g.strokeStyle = p.muted; g.globalAlpha = 0.5; g.lineWidth = 1;
      for (let i = 0; i < d.lines.count; i++) {
        g.beginPath();
        for (let j = 0; j < d.lines.per; j++) {
          const s = i * d.lines.per + j;
          const X = clamp(sx(V[2 * s]), -S, 2 * S), Y = clamp(sy(V[2 * s + 1]), -S, 2 * S);
          j ? g.lineTo(X, Y) : g.moveTo(X, Y);
        }
        g.stroke();
      }
      g.globalAlpha = 1;
    }
    ticks(g, S, { x0, x1, y0, y1 });
    points(g, d, s => [sx(H[2 * s]), sy(H[2 * s + 1])], hover);
    const fn = store.state.fwd && store.state.fwd.node;
    if (fn && fn[nodes[0].id] && fn[nodes[1].id]) ring(g, sx(fn[nodes[0].id].a), sy(fn[nodes[1].id].a));
    setAxes('ls', nodes.map(n => n.label), note);
  }

  function tex(s) {
    if (typeof katex === 'undefined') return esc(s);
    try { return katex.renderToString(String(s), { throwOnError: false }); } catch { return esc(s); }
  }
  let infoNote = '';
  function setAxes(kind, labels, note = '') {
    const key = kind + '|' + labels.join('|');
    if (key !== axesKey) {
      axesKey = key;
      axesEl.innerHTML = labels.length === 2 ? `&rarr; ${tex(labels[0])} &nbsp; &uarr; ${tex(labels[1])}` : '';
    }
    infoNote = note;
    showInfo(hover);   // every redraw: the data, the plot or the net may have changed under a resting pointer
  }

  function showInfo(i) {
    const d = data;
    let s = infoNote || HINT;
    if (i >= 0 && d && i < d.n) {
      const f = v => (model.fmt ? model.fmt(v, 2) : v.toFixed(2));
      s = `#${i + 1}: (${d.X[i].map(f).join(', ')}) → ${d.Y[i].length > 1 ? 'class ' + (d.cls[i] + 1) : f(d.Y[i][0])}`;
    }
    if (infoEl.textContent !== s) infoEl.textContent = s;
  }

  function pick(e) {
    if (!pts) return -1;
    const r = plot.getBoundingClientRect(), k = r.width ? size.plot / r.width : 1;
    const x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
    let best = -1, bd = 81;
    for (let s = 0; s < pts.length / 2; s++) {
      const dx = pts[2 * s] - x, dy = pts[2 * s + 1] - y, dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = s; }
    }
    return best;
  }
  plot.addEventListener('pointermove', e => {
    const i = pick(e);
    plot.style.cursor = i >= 0 && !ro ? 'pointer' : '';
    if (i !== hover) { hover = i; showInfo(i); dirty = true; kick(); }
  });
  plot.addEventListener('pointerleave', () => { if (hover >= 0) { hover = -1; showInfo(-1); dirty = true; kick(); } });
  plot.addEventListener('click', e => {
    if (ro) return;
    const i = pick(e);
    if (i >= 0) loadSample(i);
  });

  // ---------------------------------------------------------------- per-neuron maps

  let mapCanvas = null;
  // CPU-backed on purpose: toDataURL on a GPU canvas stalls on a readback, 19 times per update.
  function mapCtx(size) {
    if (!mapCanvas) mapCanvas = document.createElement('canvas');
    if (mapCanvas.width !== size) { mapCanvas.width = mapCanvas.height = size; }
    return mapCanvas.getContext('2d', { willReadFrequently: true });
  }

  function scaleOf(act, A, R, i, n) {
    const b = BOUNDED[act];
    if (b) return { c: (b[0] + b[1]) / 2, h: (b[1] - b[0]) / 2 };
    let m = 0;
    for (let s = 0; s < n; s++) { const v = Math.abs(A[s * R + i]); if (v > m && Number.isFinite(v)) m = v; }
    return { c: 0, h: m || 1 };
  }

  function heatUrl(A, R, i, act) {
    const p = pal(), n = GRID * GRID, { c, h } = scaleOf(act, A, R, i, n);
    const g = mapCtx(GRID), img = g.createImageData(GRID, GRID), px = img.data, bg = p.bgRGB;
    for (let s = 0; s < n; s++) {
      const v = clamp((A[s * R + i] - c) / h, -1, 1);
      const rgb = v >= 0 ? p.pos : p.neg, a = Number.isFinite(v) ? 0.12 + 0.88 * Math.abs(v) : 0;
      const o = 4 * s;
      px[o] = bg[0] + (rgb[0] - bg[0]) * a;
      px[o + 1] = bg[1] + (rgb[1] - bg[1]) * a;
      px[o + 2] = bg[2] + (rgb[2] - bg[2]) * a;
      px[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return mapCanvas.toDataURL();
  }

  function curveUrl(A, R, i, act) {
    const p = pal(), N = 48, { c, h } = scaleOf(act, A, R, i, CURVE);
    const g = mapCtx(N);
    g.fillStyle = p.bg; g.fillRect(0, 0, N, N);
    const mid = N / 2, amp = N * 0.36;
    const Y = k => mid - clamp((A[k * R + i] - c) / h, -1.2, 1.2) * amp;
    for (let k = 0; k < CURVE; k++) {
      const x = k / (CURVE - 1) * N, y = Y(k);
      g.fillStyle = css(y <= mid ? p.pos : p.neg);
      g.globalAlpha = 0.45;
      g.fillRect(x, Math.min(y, mid), N / CURVE + 0.6, Math.abs(y - mid));
    }
    g.globalAlpha = 1;
    g.beginPath();
    for (let k = 0; k < CURVE; k++) { const x = k / (CURVE - 1) * N; k ? g.lineTo(x, Y(k)) : g.moveTo(x, Y(k)); }
    g.strokeStyle = p.fg; g.lineWidth = 2.5; g.lineJoin = 'round';
    g.stroke();
    return mapCanvas.toDataURL();
  }

  function clearMaps() {
    for (const id of mapUrls.keys()) { try { ctx.view?.setNodeImage?.(id, null); } catch { /* node gone */ } }
    mapUrls.clear();
  }

  function drawMaps(net, t, d, M, grid) {
    if (!ctx.view || !ctx.view.setNodeImage) return;
    const acts = t.maps && M && d ? (d.dim === 2 ? grid() : d.dim === 1 ? forwardMany(net, model, d.curve, CURVE, 0, M) : null) : null;
    if (!acts) { clearMaps(); return; }
    const keep = new Set();
    net.layers.forEach((layer, l) => {
      const A = acts[l];
      if (!A) return;
      const nodes = model.nodesIn(net, l), Rn = nodes.length;
      nodes.forEach((nd, i) => {
        const act = l === 0 ? 'identity' : layer.act;
        const url = d.dim === 2 ? heatUrl(A, Rn, i, act) : curveUrl(A, Rn, i, act);
        keep.add(nd.id);
        if (mapUrls.get(nd.id) !== url) {
          mapUrls.set(nd.id, url);
          try { ctx.view.setNodeImage(nd.id, url); } catch (err) { console.warn('[nn/train] setNodeImage:', err); }
        }
      });
    });
    for (const id of [...mapUrls.keys()]) {
      if (!keep.has(id)) { mapUrls.delete(id); try { ctx.view.setNodeImage(id, null); } catch { /* node gone */ } }
    }
  }

  // ---------------------------------------------------------------- go

  syncControls();
  setRunning(false);
  kick();
  // Handle for tests and the console.
  panel.nnTrain = {
    play, pause, step, reset, adapt, loadSample,
    get running() { return running; }, get eval() { return lastEval; },
    point: i => (pts && i >= 0 && 2 * i < pts.length ? [pts[2 * i], pts[2 * i + 1]] : null),
  };
}
