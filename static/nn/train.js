// Training panel for the Net tab (docs/NN_CONTRACT.md), floating over #nn-stage.
//
// Settings live in net.meta.train. While playing, each animation frame runs up to `speed`
// model.trainStep calls inside a small time budget, then one store.touch(); pausing commits once.
// Plots: the input space (decision boundary for 2-input nets, fitted curve for 1-input nets),
// a hidden 2-neuron layer's activation space, and per-neuron heatmaps drawn into the nodes
// through ctx.view.setNodeImage (throttled).
//
// Sequence datasets (kind 'seq', docs/NN_ATTENTION.md): the plot is the current sample instead,
// one row per token (its features, its row of the chosen attention matrix, its outputs against
// its targets), a ◀ ▶ sample stepper replaces click-a-point, and neuron maps are off. Nets with
// attention are evaluated through model.predict (attention multiplies activations, which the
// matrices can't express); Adapt network keeps token structure or refuses (see adaptNet).
// Word datasets (model.WORDS, ds.decode) also name things: the stepper shows the sentence, loading
// a sample sets meta.tokenNames to its words, the plot names each output by its nearest word, and
// the readout adds the word accuracy.

import { colorFor, HI } from './store.js';
import { tokenLabel } from './focus.js';

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

// Presets record their dataset (and learning rate) in meta.train, which readSettings reads
// directly. Nets saved before that carry only the preset's title, so for them (and only when
// meta.train names none) the title still picks the preset's dataset and lr.
let legacyTitles = null;
function legacyPreset(net, model) {
  if (!legacyTitles) {
    legacyTitles = new Map();
    for (const p of Object.values(model.PRESETS || {})) {
      if (!p.dataset && !Number.isFinite(p.lr)) continue;
      try { legacyTitles.set(p.build(1).meta.title, p); } catch { /* skip */ }
    }
  }
  return legacyTitles.get(net.meta && net.meta.title) || null;
}
const legacyPresetDataset = (net, model) => legacyPreset(net, model)?.dataset || undefined;

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

// meta.train.init: { <shared matrix name>: scheme } for Reset (model.randomize's init option), e.g.
// the word presets' { W_Q: 'small', W_K: 'small', W_V: 'identity' }. Unknown schemes are dropped.
function cleanInit(v, model) {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return null;
  const ok = model.INIT_SCHEMES || [];
  const out = Object.fromEntries(Object.entries(v).filter(([k, s]) => k && ok.includes(s)));
  return Object.keys(out).length ? out : null;
}

// net.meta.train with defaults filled in and junk repaired. Never mutates the net.
// A dataset recorded in meta.train (a preset's, or the user's choice) wins even when the net no
// longer fits it: the panel then offers "Adapt network" instead of silently switching.
export function readSettings(net, model) {
  const raw = (net.meta && net.meta.train) || {};
  const init = cleanInit(raw.init, model);
  return {
    ...(init ? { init } : {}),
    dataset: (model.DATASETS || {})[raw.dataset] ? raw.dataset : defaultDataset(net, model),
    n: Math.round(num(raw.n, 200, 4, 5000)),
    noise: num(raw.noise, 0.1, 0, 10),
    seed: Math.round(num(raw.seed, 1)),
    lr: num(raw.lr, Number.isFinite(raw.lr) ? 0.1 : num(legacyPreset(net, model)?.lr, 0.1, 1e-7, 1000), 1e-7, 1000),
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
  // Attention multiplies activations together, which the matrices can't express: the model's own
  // forward pass then does every sample (only from the inputs).
  if (M.some(m => m.kind === 'attention')) return from === 0 ? predictMany(net, model, X, n) : null;
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
      const B = softmaxBlock(net.layers[l], R);   // per token (and group) on a token layer
      for (let o = 0; o < n * R; o += B) {
        let mx = -Infinity, sum = 0;
        for (let i = 0; i < B; i++) mx = Math.max(mx, Z[o + i]);
        for (let i = 0; i < B; i++) { Z[o + i] = Math.exp(Z[o + i] - mx); sum += Z[o + i]; }
        for (let i = 0; i < B; i++) Z[o + i] /= sum;
      }
    } else {
      const f = model.ACTS && model.ACTS[act] && model.ACTS[act].f;
      if (f && act !== 'identity') for (let k = 0; k < Z.length; k++) Z[k] = f(Z[k]);
    }
    acts[l] = Z;
  }
  return acts;
}

// ---- token layers (docs/NN_ATTENTION.md): tokens x d nodes per group, token-major

const layerTokens = ly => (ly && Number.isInteger(ly.tokens) && ly.tokens > 1 ? ly.tokens : 1);
const layerGroups = ly => (ly && Array.isArray(ly.groups) && ly.groups.length ? ly.groups.length : 1);
const isTokenLayer = ly => !!ly && (layerTokens(ly) > 1 || layerGroups(ly) > 1 || ly.kind === 'attention');
function softmaxBlock(ly, R) {
  const k = layerTokens(ly) * layerGroups(ly);
  return k > 1 && R % k === 0 ? R / k : R;
}

// forwardMany through model.predict: acts[l] = Float64Array n x size_l, acts[0] = X.
function predictMany(net, model, X, n) {
  const sizes = net.layers.map((_, l) => model.nodesIn(net, l).length), d0 = sizes[0];
  const rows = new Array(n);
  for (let s = 0; s < n; s++) rows[s] = X.subarray ? X.subarray(s * d0, (s + 1) * d0) : X.slice(s * d0, (s + 1) * d0);
  const per = model.predict(net, rows, { layer: 'all' });
  const acts = sizes.map(sz => new Float64Array(n * sz));
  for (let s = 0; s < n; s++) {
    const all = per[s];
    if (!all) return null;
    for (let l = 1; l < sizes.length; l++) {
      const a = all[l], o = s * sizes[l];
      if (!a) return null;
      for (let i = 0; i < sizes[l]; i++) acts[l][o + i] = a[i];
    }
  }
  acts[0] = X;
  return acts;
}

// Mean loss over a dataset, per the contract's definitions (xent falls back to mse unless the
// output layer is softmax or sigmoid). P: n x K predictions, Y: rows of targets. segments: the
// softmax blocks of a token output layer (tokens x groups); its cross-entropy is their mean.
export function datasetLoss(P, Y, n, K, loss, outAct, segments = 1) {
  const eps = 1e-12;
  const xent = loss === 'xent' && (outAct === 'softmax' || outAct === 'sigmoid');
  const m = segments > 1 && K % segments === 0 ? segments : 1;
  let sum = 0;
  for (let s = 0; s < n; s++) {
    const y = Y[s], o = s * K;
    let e = 0;
    if (xent && outAct === 'softmax') {
      for (let k = 0; k < K; k++) e -= y[k] * Math.log(Math.max(P[o + k], eps));
      e /= m;
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

// Word datasets (model.WORDS) name every output token by its nearest word: ds.decode(row) -> words.
const isWords = ds => !!ds && typeof ds.decode === 'function' && !!ds.vocab;

// The share of output tokens whose nearest word (ds.decode) is the target word. P: n x K outputs
// (flat), targetWords: per sample, one word per token.
export function wordAccuracy(P, targetWords, n, K, ds) {
  let hit = 0, all = 0;
  for (let s = 0; s < n; s++) {
    const want = targetWords[s] || [];
    const got = ds.decode(P.subarray ? P.subarray(s * K, (s + 1) * K) : P.slice(s * K, (s + 1) * K));
    want.forEach((w, t) => { all++; if (got[t] === w) hit++; });
  }
  return all ? hit / all : null;
}

// Resize the input and output layers to fit a dataset; hidden layers are kept. New nodes are
// wired densely to the neighbouring layer with small Xavier weights. The output activation and
// the loss are set to suit the task. Returns a sentence describing the result.
//
// Token nets (docs/NN_ATTENTION.md) are adapted token by token instead, so their structure survives:
// - a sequence dataset on a plain net: the input and output are resized flat, then marked as
//   `tokens` rows (the hidden layers stay dense, an MLP on the whole sequence);
// - a net with attention, groups, token hidden layers, shared or fixed edges: only the number of
//   features per token changes. A new feature copies the wiring of its token's last feature,
//   with a new entry of each shared matrix (one draw per entry, so the copies stay tied) and no
//   fixed edges (a residual has no partner for it).
// Anything else (a token net on a plain dataset, another token count, an attention output of the
// wrong width) throws an Error whose message says why, and leaves the net untouched.
export function adaptNet(net, model, ds, { seed = 1 } = {}) {
  if (ds.kind === 'seq' || net.layers.some(isTokenLayer)) {
    const msg = adaptTokens(net, model, ds, seed);
    if (msg) return msg;   // null: a plain net again, adapted below
  }
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

// The output activation and loss a sequence dataset asks for (identity + mse unless it says).
function seqHead(ds) {
  const act = typeof ds.outAct === 'string' ? ds.outAct : ds.loss === 'xent' ? 'softmax' : 'identity';
  return { act, loss: ds.loss === 'xent' || ds.loss === 'mse' ? ds.loss : act === 'identity' ? 'mse' : 'xent' };
}

function adaptTokens(net, model, ds, seed) {
  const name = ds.label || 'this dataset';
  const L = net.layers.length, last = L - 1;
  const tokenHidden = net.layers.some((ly, l) => ly.kind === 'attention' || layerGroups(ly) > 1 || (l > 0 && l < last && layerTokens(ly) > 1));
  const structured = tokenHidden || net.edges.some(e => e.tie || e.fixed);
  if (ds.kind !== 'seq') {
    if (tokenHidden) {
      throw new Error(`This net works on token sequences and ${name} is not one: pick a sequence dataset, or load a preset made for ${name}.`);
    }
    for (const l of [0, last]) delete net.layers[l].tokens;   // a flat vector again
    return null;
  }
  const T = Math.max(1, ds.tokens | 0 || 1);
  const dIn = ds.inputs / T, dOut = ds.outputs / T;
  if (!Number.isInteger(dIn) || !Number.isInteger(dOut)) throw new Error(`${name} does not split into ${T} tokens.`);
  const head = seqHead(ds);
  const rand = model.rng(seed);
  const outLayer = net.layers[last];
  if (structured && !tokenHidden) {
    throw new Error(`This net's shared or fixed weights are laid out for its own inputs, so Adapt can't rewire it for ${T} tokens × ${dIn} features: load a preset made for ${name}.`);
  }
  if (!structured) {
    resizeLayer(net, model, 0, ds.inputs, rand);
    resizeLayer(net, model, last, ds.outputs, rand);
    for (const l of [0, last]) { if (T > 1) net.layers[l].tokens = T; else delete net.layers[l].tokens; }
    model.setLayer(net, outLayer.id, { act: head.act });
  } else {
    const bad = net.layers.find(ly => isTokenLayer(ly) && layerTokens(ly) !== T);
    if (bad) {
      throw new Error(`This net has ${layerTokens(bad)} tokens and ${name} has ${T}. Every token layer is built for its token count: load a preset made for ${name}.`);
    }
    const plan = [[0, dIn], [last, dOut]];
    for (const [l, d1] of plan) {
      const ly = net.layers[l], n = model.nodesIn(net, l).length, d0 = n / T;
      if (d0 === d1) continue;
      if (ly.kind === 'attention') throw new Error(`The output is an attention layer, whose width comes from V: it can't be resized to ${d1} features per token.`);
      if (layerGroups(ly) > 1 || !Number.isInteger(d0) || d0 < 1) throw new Error(`Layer ${l} can't be resized token by token.`);
    }
    for (const [l, d1] of plan) {
      const d0 = model.nodesIn(net, l).length / T;
      if (d0 !== d1) resizeTokenLayer(net, model, l, T, d0, d1, rand);
      if (T > 1) net.layers[l].tokens = T;
    }
    if (outLayer.kind !== 'attention') model.setLayer(net, outLayer.id, { act: head.act });
  }
  net.meta = net.meta || {};
  net.meta.loss = head.loss;
  const act = net.layers[last].act;
  return `Adapted to ${T}×${dIn} → … → ${T}×${dOut} tokens: ${act} output, ${head.loss === 'xent' ? 'cross-entropy' : 'MSE'} loss`;
}

// Change the features per token of layer l from d0 to d1, keeping T tokens (token-major order).
function resizeTokenLayer(net, model, l, T, d0, d1, rand) {
  const lay = net.layers[l];
  const nodes = model.nodesIn(net, l);
  // A tokens × d grid (a row per token, a column per feature, as the 3-token attention presets lay
  // out their inputs) stays a grid: the same rows, the columns at the same step.
  const grid = d0 > 1 && new Set(nodes.map(n => n.x)).size === d0 && nodes.every((n, k) => n.y === nodes[k - (k % d0)].y);
  const gx = grid ? nodes[0].x : 0, fx = grid ? nodes[1].x - nodes[0].x : 0;
  const rows = grid ? Array.from({ length: T }, (_, t) => nodes[t * d0].y) : null;
  const ys = nodes.map(n => n.y), cy = ys.reduce((s, y) => s + y, 0) / (ys.length || 1);
  const gap = ys.length > 1 ? Math.max(30, (Math.max(...ys) - Math.min(...ys)) / (ys.length - 1)) : 60;
  if (d1 < d0) {
    for (let t = 0; t < T; t++) for (let f = d1; f < d0; f++) model.removeNode(net, nodes[t * d0 + f].id);
  } else {
    const tieRe = /^(.*):\s*(\d+)\s*,\s*(\d+)\s*$/;
    const drawn = new Map();                 // new tie id -> its one weight
    const lim = Math.sqrt(3 / d1);
    for (let t = 0; t < T; t++) {
      const tmpl = nodes[t * d0 + d0 - 1];   // the token's last feature: new ones copy its wiring
      const wiring = net.edges.filter(e => (e.from === tmpl.id || e.to === tmpl.id) && !e.fixed).map(e => ({ ...e }));
      const lab = /^(.*)_\{\s*(\d+)\s*,\s*(\d+)\s*\}$/.exec(tmpl.label || '');
      const btie = typeof tmpl.tie === 'string' ? /^(.*):\s*(\d+)\s*$/.exec(tmpl.tie) : null;   // shared bias 'b_Q:f'
      for (let f = d0; f < d1; f++) {
        const id = model.addNode(net, lay.id, {
          index: t * d1 + f, x: tmpl.x, y: tmpl.y,
          label: lab ? `${lab[1]}_{${t + 1},${f + 1}}` : undefined,
        });
        if (id == null) continue;
        if (btie) {
          const nd = model.node(net, id);
          nd.tie = `${btie[1]}:${f + 1}`;
          nd.bias = 0;
        }
        for (const e of wiring) {
          const out = e.from === tmpl.id, other = out ? e.to : e.from;
          const m = e.tie ? tieRe.exec(e.tie) : null;
          const tie = m ? (out ? `${m[1]}:${f + 1},${m[3]}` : `${m[1]}:${m[2]},${f + 1}`) : e.tie || null;
          let w = tie && drawn.has(tie) ? drawn.get(tie) : (rand() * 2 - 1) * lim;
          if (tie) drawn.set(tie, w);
          const eid = out ? model.connect(net, id, other, w) : model.connect(net, other, id, w);
          const ne = eid != null ? model.edge(net, eid) : null;
          if (ne && tie) ne.tie = tie;
        }
      }
    }
  }
  const now = model.nodesIn(net, l), dn = now.length / T, extra = gap * 0.35;
  if (grid && dn > 1) {
    now.forEach((n, k) => { n.x = gx + (k % dn) * fx; n.y = rows[Math.floor(k / dn)]; });
    return;
  }
  // even spacing around the old centre, a little extra between tokens
  const span = (now.length - 1) * gap + (T - 1) * extra;
  now.forEach((n, k) => { n.y = Math.round((cy - span / 2 + k * gap + Math.floor(k / dn) * extra) * 10) / 10; });
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
      <div class="nt-warn" hidden><span></span><button data-act="adapt" title="Resize the input and output layers to fit the dataset; hidden layers are kept. A token net keeps its structure: it changes the features per token, or says why it can't">Adapt network</button></div>
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
        <button data-act="reset" title="Re-randomize the weights from the init seed (a preset that starts some shared matrices its own way, such as W_V = I, keeps that)">Reset</button>
        <label title="Seed for Reset">init <input type="number" data-k="initSeed" step="1"></label>
        <button data-act="dice" title="New random init seed, then reset">&#8635;</button>
      </div>
      <div class="nt-read">
        <span>epoch <b data-r="epoch">0</b></span>
        <span>step <b data-r="steps">0</b></span>
        <span>loss <b data-r="loss">&ndash;</b></span>
        <span data-r="accw" hidden><span data-r="accl">acc</span> <b data-r="acc"></b></span>
      </div>
      <canvas class="nt-chart"></canvas>
      <div class="nt-grid nt-g2b">
        <label>plot <select data-k="space"></select></label>
        <label class="nt-check" title="Each neuron's activation over the input domain, drawn inside its node"><input type="checkbox" data-k="maps"> neuron maps</label>
      </div>
      <canvas class="nt-plot"></canvas>
      <div class="nt-steps" hidden>
        <button data-act="prev" title="Load the previous sample">&#9664;</button>
        <span class="nt-steps-lab"></span>
        <button data-act="next" title="Load the next sample">&#9654;</button>
      </div>
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
  const steps = $('.nt-steps'), stepsLab = $('.nt-steps-lab');
  const HINT = ro ? '' : 'click a point to load it as the current sample';
  const SEQ_HINT = ro ? '' : 'hover a cell for its value; ◀ ▶ load the samples';

  // A sequence dataset's shape reads tokens × features: 3×2 → 3×2.
  const dsShape = v => (v.kind === 'seq' && v.tokens > 1
    ? `${v.tokens}&times;${v.inputs / v.tokens}&rarr;${v.tokens}&times;${v.outputs / v.tokens}` : `${v.inputs}&rarr;${v.outputs}`);
  K('dataset').innerHTML = Object.entries(model.DATASETS || {})
    .map(([k, v]) => `<option value="${k}">${esc(v.label || k)} (${dsShape(v)})</option>`).join('');
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
    const made = ds.make(t.n, t.seed, t.noise);
    let { X, Y } = made;
    // A word dataset's samples carry their sentences: the words in, and the word each token should output.
    const words = isWords(ds) && Array.isArray(made.words) && Array.isArray(made.targetWords) ? made : null;
    // A sequence sample may come as tokens x features rows: the net reads it flat, token-major.
    const flat = r => (Array.isArray(r) && r.some(Array.isArray) ? r.flat(Infinity) : r);
    if (ds.kind === 'seq') { X = X.map(flat); Y = Y.map(flat); }
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
    if (ds.kind === 'seq') {
      dom = null;   // no input-space plot: drawSeq shows the sample's tokens instead
    } else if (dim === 2) {
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
    const T = ds.kind === 'seq' ? Math.max(1, ds.tokens | 0 || 1) : 1;
    data = { key, ds, X, Y, Xf, n, dim, K: Kout, cls, mid, half, dom, kind: ds.kind, grid: null, curve: null, lines: null, T,
      words: words ? words.words : null, twords: words ? words.targetWords : null };
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

  // Sizes match; a net with token inputs must also have the dataset's token count.
  const tokensOk = (net, ds) => {
    const T0 = layerTokens(net.layers[0]);
    return T0 === 1 || T0 === (ds.kind === 'seq' ? Math.max(1, ds.tokens | 0) : 1);
  };
  const fits = (net, ds) => {
    const sh = netShape(net, model);
    return !!ds && ds.inputs === sh.inputs && ds.outputs === sh.outputs && tokensOk(net, ds);
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
    const lastL = net.layers[L - 1];
    const loss = datasetLoss(out, d.Y, d.n, d.K, net.meta?.loss || 'mse', lastL.act, d.K / softmaxBlock(lastL, d.K));
    let acc = null;
    if (d.twords) acc = wordAccuracy(out, d.twords, d.n, d.K, d.ds);
    else if (d.kind === 'class') {
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
    // a preset's own recipe for some shared matrices (meta.train.init) is kept
    model.randomize(net, { seed: t.initSeed, scheme: acts.some(a => a === 'relu' || a === 'leaky') ? 'he' : 'xavier', init: t.init || null });
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
    // Token nets refuse what they can't do without breaking their structure: work on a copy, so
    // a refusal leaves the live net exactly as it was.
    const tmp = model.clone(net);
    let msg;
    try { msg = adaptNet(tmp, model, ds, { seed: t.initSeed }); } catch (err) { ctx.toast?.(err.message, 6000); return; }
    for (const k of Object.keys(net)) delete net[k];
    Object.assign(net, tmp);
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
    // a sentence names the tokens (docs/NN_LENS.md), so the canvas, the matrix panel, the lens and
    // the attention panel show its words; part of the same commit, so undo brings the old names back
    // Names that are all vocabulary words were left by a word dataset: a sample without words drops them.
    if (d.words?.[i]) { net.meta = net.meta || {}; net.meta.tokenNames = d.words[i].slice(); }
    else if (Array.isArray(net.meta?.tokenNames) && model.WORDS && net.meta.tokenNames.length
      && net.meta.tokenNames.every(w => Object.hasOwn(model.WORDS, w))) delete net.meta.tokenNames;
    lastLoaded = i;
    if (!running) store.commit(`Load sample ${i + 1}`);
    else { dirty = true; kick(); }
  }

  // Which dataset sample the net's inputs hold (-1: none of them). The stepper moves from there.
  let lastLoaded = -1, curKey = '', curIdx = -1;
  function currentSample(net, d) {
    const ins = model.nodesIn(net, 0);
    const key = d.key + '|' + ins.map(q => q.value).join(',');
    if (key === curKey) return curIdx;
    curKey = key;
    const eq = s => d.X[s] && ins.every((q, j) => Math.abs(d.X[s][j] - q.value) < 1e-9);
    if (lastLoaded >= 0 && lastLoaded < d.n && eq(lastLoaded)) return (curIdx = lastLoaded);
    curIdx = -1;
    for (let s = 0; s < d.n; s++) if (eq(s)) { curIdx = s; break; }
    return curIdx;
  }

  function stepSample(dir) {
    if (ro) return;
    const net = store.net, t = readSettings(net, model), ds = model.DATASETS[t.dataset];
    if (!ds) return;
    const d = getData(t), cur = currentSample(net, d);
    loadSample(cur < 0 ? (dir > 0 ? 0 : d.n - 1) : (cur + dir + d.n) % d.n);
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

  // The attention layers (and heads) a sequence plot can show: values `${layerId}#${head}`.
  function attnChoices(net) {
    const out = [];
    net.layers.forEach((ly, l) => {
      if (ly.kind !== 'attention') return;
      const H = Number.isInteger(ly.heads) && ly.heads > 0 ? ly.heads : 1;
      for (let hd = 0; hd < H; hd++) {
        out.push({ l, hd, value: `${ly.id}#${hd}`, label: `attention: ${ly.name || 'layer ' + l}${H > 1 ? ` · head ${hd + 1}` : ''}` });
      }
    });
    return out;
  }
  const attnChoice = (net, t) => { const cs = attnChoices(net); return cs.find(c => c.value === t.space) || cs[0] || null; };

  function syncControls() {
    const net = store.net, t = readSettings(net, model);
    const seq = model.DATASETS[t.dataset]?.kind === 'seq';
    for (const k of ['dataset', 'n', 'noise', 'seed', 'lr', 'batch', 'speed', 'initSeed', 'maps']) setVal(K(k), t[k]);
    setVal(K('loss'), net.meta?.loss === 'xent' ? 'xent' : 'mse');
    const maps = K('maps');
    maps.disabled = ro || seq;
    if (seq) maps.checked = false;   // shown off; the saved setting comes back with a plain dataset
    maps.parentElement.title = seq ? 'Neuron maps are off for sequence datasets: a neuron sees a whole sequence, not a point in a plane'
      : 'Each neuron\'s activation over the input domain, drawn inside its node';
    if (seq) {
      const cs = attnChoices(net);
      const sig = 'seq|' + cs.map(c => c.value + ':' + c.label).join(',');
      const sel = K('space');
      if (sig !== spaceSig) {
        spaceSig = sig;
        sel.innerHTML = cs.length ? cs.map(c => `<option value="${esc(c.value)}">${esc(c.label)}</option>`).join('')
          : '<option value="">outputs vs targets</option>';
        sel.title = cs.length ? 'Which attention matrix to show for the current sample' : 'This net has no attention layer';
      }
      setVal(sel, attnChoice(net, t)?.value ?? '');
      return;
    }
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
    if (act === 'fold') fold();
    if (ro) return;
    if (act === 'play') toggle();
    else if (act === 'step') step();
    else if (act === 'reset') reset();
    else if (act === 'dice') { const s = 1 + Math.floor(Math.random() * 9999); reset(s); }
    else if (act === 'adapt') adapt();
    else if (act === 'prev') stepSample(-1);
    else if (act === 'next') stepSample(1);
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

  // Fold the panel to its header (on), unfold it (false), or toggle; saved and mirrored like the button.
  function fold(on = !ui.fold) {
    if (!!ui.fold === !!on) return;
    ui.fold = !!on; saveUi(); applyUi(); dirty = true; kick();
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
    if (acc) {
      R('acc').textContent = Math.round(lastEval.acc * 100) + '%';
      const wd = !!d?.twords, lab = wd ? 'words' : 'acc';
      if (R('accl').textContent !== lab) {
        R('accl').textContent = lab;
        R('accw').title = wd ? 'Word accuracy: the share of output tokens, over the whole dataset, whose nearest word is the target word'
          : 'Accuracy: the share of points classified correctly';
      }
    }
    mini.textContent = `${ro && ds ? (ds.label || t.dataset) + ' · ' : ''}epoch ${epoch} · ${lastEval ? fmtLoss(lastEval.loss) : '–'}`;
    const mis = !!ds && !ok;
    warn.hidden = !mis;
    if (mis) {
      // token shapes read tokens × features on both sides
      const T0 = layerTokens(net.layers[0]), TL = layerTokens(net.layers[net.layers.length - 1]);
      const side = (n, T) => (T > 1 && n % T === 0 ? `${T}×${n / T}` : `${n}`);
      const Td = ds.kind === 'seq' ? Math.max(1, ds.tokens | 0) : 1;
      warn.firstChild.textContent = `This net is ${side(sh.inputs, T0)} → … → ${side(sh.outputs, TL)}; ` +
        `${ds.label || t.dataset} needs ${side(ds.inputs, Td)} → … → ${side(ds.outputs, Td)}. `;
    }
    const seq = !!d && d.kind === 'seq';
    steps.hidden = !seq;
    if (seq) {
      const cur = sh.inputs === ds.inputs ? currentSample(net, d) : -1;
      const said = cur >= 0 && d.words?.[cur] ? `: “${d.words[cur].join(' ')}”` : '';
      const lab = cur >= 0 ? `sample ${cur + 1} of ${d.n}${said}` : `not a sample (${d.n} in the set)`;
      if (stepsLab.textContent !== lab) { stepsLab.textContent = lab; stepsLab.title = lab; }
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

  // ---------------------------------------------------------------- sequence plot
  // One row per token of the current sample (the net's live inputs): its features x_t, its row of
  // the chosen attention matrix A (which tokens it reads), then its outputs ŷ_t as bars with the
  // targets y_t as ticks. Canvas height follows the token count.

  let seqCells = [];   // hover boxes { x, y, w, h, text } in canvas px
  function drawSeq(net, t, d, ok, inOk) {
    const S = size.plot;
    pts = null;
    seqCells = [];
    if (!S) return;
    const p = pal(), T = d.T, L = net.layers.length, fw = store.state.fwd;
    const f2 = v => (model.fmt ? model.fmt(v, 2) : v.toFixed(2));
    const font = (px, w = '') => `${w}${px}px system-ui, "Segoe UI", sans-serif`;
    if (!inOk) {
      const g = sizeCanvas(plot, S, 64);
      g.fillStyle = p.bg; g.fillRect(0, 0, S, 64);
      g.fillStyle = p.muted; g.font = font(12);
      g.fillText('The network does not fit this dataset', 10, 36);
      setAxes('seq', []);
      return;
    }
    const ins = model.nodesIn(net, 0), outs = L > 1 ? model.nodesIn(net, L - 1) : [];
    const dIn = Math.max(1, Math.round(ins.length / T)), dOut = ok ? Math.max(1, Math.round(outs.length / T)) : 0;
    const ch = attnChoice(net, t);
    const A = ch ? fw?.attn?.[ch.l]?.heads?.[ch.hd]?.A : null;
    const causal = ch ? !!net.layers[ch.l].causal : false;
    const nA = ch ? T : 0;
    // token names (net.meta.tokenNames, docs/NN_LENS.md) replace t1…tn; the row labels widen to fit
    const names = Array.from({ length: T }, (_, i) => tokenLabel(net, i));   // focus.js: its name, else t1, t2, ...
    const g0 = plot.getContext('2d');
    g0.font = font(10);
    const pad = 8, lw = Math.min(64, Math.max(20, Math.ceil(Math.max(...names.map(s => g0.measureText(s).width))) + 7)), gap = 9, head = 30, minOut = 66;
    const who = i => (names[i] === `t${i + 1}` ? `token ${i + 1}` : `${names[i]} (token ${i + 1})`);
    const fit = k => Math.floor((S - 2 * pad - lw - 2 * gap - minOut) / Math.max(1, k));
    let showX = true, c = Math.min(36, fit(dIn + nA));
    if (c < 15) { showX = false; c = Math.min(36, fit(nA || 1)); }
    c = Math.max(12, c);
    const H = pad + head + T * c + 26;
    const g = sizeCanvas(plot, S, H);
    g.fillStyle = p.bg; g.fillRect(0, 0, S, H);
    g.textBaseline = 'middle';
    const y0 = pad + head;
    const xX = pad + lw, xA = showX ? xX + dIn * c + gap : xX, xO = xA + nA * c + (nA || showX ? gap : 0), wO = S - pad - xO;
    const numPx = c >= 30 ? 10 : c >= 25 ? 9 : 8;   // cell values, when they fit in the cell
    const title = (x, s) => { g.font = font(10.5, '600 '); g.fillStyle = p.fg; g.textAlign = 'left'; g.fillText(s, x, pad + 5); };
    const colLab = (x, s) => { g.font = font(9.5); g.fillStyle = p.muted; g.textAlign = 'center'; g.fillText(s, x, pad + 20); };
    const box = (x, y, v, max, txt, masked) => {
      if (masked) {
        g.save();
        g.beginPath(); g.rect(x + 1, y + 1, c - 2, c - 2); g.clip();
        g.strokeStyle = p.line; g.lineWidth = 1;
        g.beginPath();
        for (let k = -c; k < c; k += 5) { g.moveTo(x + k, y + c); g.lineTo(x + k + c, y); }
        g.stroke();
        g.restore();
      } else {
        g.fillStyle = Number.isFinite(v) ? colorFor(v, max, p.theme) : 'rgba(128,128,128,0.25)';
        g.fillRect(x + 1, y + 1, c - 2, c - 2);
      }
      const s = masked ? '–' : txt;
      g.font = font(numPx);
      if (masked || g.measureText(s).width <= c - 3) {
        g.fillStyle = masked ? p.muted : p.fg; g.textAlign = 'center';
        g.fillText(s, x + c / 2, y + c / 2 + 0.5);
      }
    };
    // row labels: token t
    g.font = font(10); g.fillStyle = p.muted; g.textAlign = 'right';
    const clip = (s, w) => { if (g.measureText(s).width <= w) return s; while (s.length > 1 && g.measureText(s + '…').width > w) s = s.slice(0, -1); return s + '…'; };
    for (let i = 0; i < T; i++) g.fillText(clip(names[i], lw - 7), xX - 5, y0 + i * c + c / 2);
    // x_t
    if (showX) {
      title(xX, 'x');
      let mx = 0;
      for (const q of ins) if (Number.isFinite(q.value)) mx = Math.max(mx, Math.abs(q.value));
      for (let f = 0; f < dIn; f++) colLab(xX + f * c + c / 2, String(f + 1));
      for (let i = 0; i < T; i++) for (let f = 0; f < dIn; f++) {
        const v = ins[i * dIn + f]?.value, x = xX + f * c, y = y0 + i * c;
        box(x, y, v, mx || 1, f2(v), false);
        seqCells.push({ x, y, w: c, h: c, text: `x: ${who(i)}, feature ${f + 1} = ${f2(v)}` });
      }
    }
    // A: rows are queries (this token), columns the keys it reads
    if (ch) {
      title(xA, net.layers[ch.l].heads > 1 ? `A, head ${ch.hd + 1}` : 'A (attention)');
      g.font = font(9.5);
      for (let j = 0; j < T; j++) colLab(xA + j * c + c / 2, clip(names[j], c - 2));
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const a = A?.[i]?.[j], masked = causal && j > i, x = xA + j * c, y = y0 + i * c;
        box(x, y, a, 1, Number.isFinite(a) ? f2(a) : '?', masked);
        const ij = names[i] === `t${i + 1}` && names[j] === `t${j + 1}` ? `${i + 1},${j + 1}` : `${names[i]}, ${names[j]}`;
        seqCells.push({ x, y, w: c, h: c, text: masked ? `A(${ij}): masked, ${who(i)} can't read the later ${who(j)}`
          : `A(${ij}) = ${Number.isFinite(a) ? f2(a) : '?'}: how much ${who(i)} reads ${who(j)}` });
      }
    }
    // outputs against targets: a word dataset names both by their nearest words
    const wordsOut = ok && !!d.twords && wO >= 30;
    title(xO, wordsOut ? 'ŷ → word' : ok ? 'ŷ vs y' : 'outputs');
    if (wordsOut) {
      const yh = outs.map(q => fw?.node?.[q.id]?.a);
      const cur = currentSample(net, d);
      const got = d.ds.decode(yh), want = cur >= 0 ? d.twords[cur] : d.ds.decode(outs.map(q => q.target));
      const two = c >= 22, bad = css(p.neg);
      g.textAlign = 'left';
      for (let i = 0; i < T; i++) {
        const top = y0 + i * c;
        if (i) { g.strokeStyle = p.line; g.lineWidth = 1; g.beginPath(); g.moveTo(xO, top + 0.5); g.lineTo(xO + wO, top + 0.5); g.stroke(); }
        const w = got[i], y = want[i], hit = !!w && w === y;
        const mark = w && y ? (hit ? '✓' : '✗') : '';
        g.font = font(10.5, '600 ');
        g.fillStyle = w && y && !hit ? bad : p.fg;
        g.textAlign = 'left';
        g.fillText(clip(w || '?', wO - 14), xO + 2, two ? top + c / 2 - 5 : top + c / 2);
        g.textAlign = 'right';
        g.fillText(mark, xO + wO, two ? top + c / 2 - 5 : top + c / 2);
        if (two) {
          g.font = font(9); g.fillStyle = p.muted; g.textAlign = 'left';
          g.fillText(clip(`y ${y || '–'}`, wO - 2), xO + 2, top + c / 2 + 6);
        }
        const v = yh.slice(i * dOut, (i + 1) * dOut).map(a => (Number.isFinite(a) ? f2(a) : '?')).join(', ');
        seqCells.push({ x: xO, y: top, w: wO, h: c, text: `${who(i)}: ŷ = (${v}), nearest word “${w || '?'}”; target “${y || 'none'}”${mark ? ' ' + mark : ''}` });
      }
    } else if (!ok || wO < 30) {
      g.font = font(10); g.fillStyle = p.muted; g.textAlign = 'left';
      g.fillText(ok ? '' : 'outputs don\'t fit', xO, y0 + c / 2);
    } else {
      const yh = outs.map(q => fw?.node?.[q.id]?.a), ys = outs.map(q => q.target);
      let vmax = 0.1;
      for (const v of [...yh, ...ys]) if (Number.isFinite(v)) vmax = Math.max(vmax, Math.abs(v));
      const x0 = xO + wO / 2, half = wO / 2 - 4, X = v => x0 + clamp(v / vmax, -1, 1) * half;
      g.strokeStyle = p.muted; g.globalAlpha = 0.45; g.lineWidth = 1;
      g.beginPath(); g.moveTo(Math.round(x0) + 0.5, y0); g.lineTo(Math.round(x0) + 0.5, y0 + T * c); g.stroke();
      g.globalAlpha = 1;
      g.font = font(9); g.fillStyle = p.muted;
      g.textAlign = 'left'; g.fillText(`−${f2(vmax)}`, xO, pad + 20);
      g.textAlign = 'right'; g.fillText(f2(vmax), xO + wO, pad + 20);
      const lh = (c - 4) / dOut;
      for (let i = 0; i < T; i++) {
        if (i) { g.strokeStyle = p.line; g.beginPath(); g.moveTo(xO, y0 + i * c + 0.5); g.lineTo(xO + wO, y0 + i * c + 0.5); g.stroke(); }
        for (let f = 0; f < dOut; f++) {
          const k = i * dOut + f, a = yh[k], y = ys[k], ly = y0 + i * c + 2 + f * lh;
          if (Number.isFinite(a)) {
            g.fillStyle = css(a >= 0 ? p.pos : p.neg);
            g.globalAlpha = 0.85;
            const xa = X(a);
            g.fillRect(Math.min(x0, xa), ly + 1, Math.abs(xa - x0), Math.max(1, lh - 2));
            g.globalAlpha = 1;
          }
          if (typeof y === 'number' && Number.isFinite(y)) {
            const xy = X(y);
            g.strokeStyle = HI; g.lineWidth = 2.5;
            g.beginPath(); g.moveTo(xy, ly - 0.5); g.lineTo(xy, ly + lh + 0.5); g.stroke();
          }
          seqCells.push({ x: xO, y: ly, w: wO, h: lh, text: `${who(i)}, output ${f + 1}: ŷ = ${Number.isFinite(a) ? f2(a) : '?'}, y = ${typeof y === 'number' ? f2(y) : 'none'}` });
        }
      }
    }
    // legend + this sample's loss
    const ly = y0 + T * c + 14;
    g.font = font(10); g.textAlign = 'left';
    let x = pad;
    if (wordsOut) {
      g.fillStyle = p.muted; g.fillText('ŷ: nearest word · y: target', x, ly);
    } else if (ok) {
      g.fillStyle = css(p.pos); g.fillRect(x, ly - 4, 12, 8); x += 16;
      g.fillStyle = p.muted; g.fillText('output ŷ', x, ly); x += g.measureText('output ŷ').width + 12;
      g.strokeStyle = HI; g.lineWidth = 2.5; g.beginPath(); g.moveTo(x + 2, ly - 6); g.lineTo(x + 2, ly + 6); g.stroke(); x += 8;
      g.fillStyle = p.muted; g.fillText('target y', x, ly); x += g.measureText('target y').width + 12;
    }
    const bl = store.state.bwd?.loss;
    if (Number.isFinite(bl)) { g.fillStyle = p.muted; g.textAlign = 'right'; g.fillText(`this sample: loss ${fmtLoss(bl)}`, S - pad, ly); }
    setAxes('seq', [], ch ? '' : 'no attention layer: each token only mixes through the dense weights');
    if (seqPtr) seqInfo();   // a resting pointer keeps its readout through training frames
  }

  let seqPtr = null;
  function seqInfo() {
    const q = seqPtr && seqCells.find(b => seqPtr.x >= b.x && seqPtr.x < b.x + b.w && seqPtr.y >= b.y && seqPtr.y < b.y + b.h);
    const s = q ? q.text : infoNote || SEQ_HINT;
    if (infoEl.textContent !== s) infoEl.textContent = s;
  }

  let hover = -1;
  function drawPlot(net, t, d, sh, M, A, ok, inOk, grid) {
    const S = size.plot;
    pts = null;
    if (!S) return;
    if (d && d.kind === 'seq') { drawSeq(net, t, d, ok, inOk); return; }
    seqCells = [];
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
    let s = infoNote || (d && d.kind === 'seq' ? SEQ_HINT : HINT);
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
    if (seqCells.length) {
      // sequence plot: read the cell under the pointer (no points to load)
      const r = plot.getBoundingClientRect(), k = r.width ? size.plot / r.width : 1;
      seqPtr = { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
      seqInfo();
      plot.style.cursor = '';
      return;
    }
    const i = pick(e);
    plot.style.cursor = i >= 0 && !ro ? 'pointer' : '';
    if (i !== hover) { hover = i; showInfo(i); dirty = true; kick(); }
  });
  plot.addEventListener('pointerleave', () => {
    seqPtr = null;
    if (seqCells.length) showInfo(-1);
    if (hover >= 0) { hover = -1; showInfo(-1); dirty = true; kick(); }
  });
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
    // off for sequences: a neuron there sees a whole sequence, not a point of a 1-D or 2-D domain
    const acts = t.maps && M && d && d.kind !== 'seq'
      ? (d.dim === 2 ? grid() : d.dim === 1 ? forwardMany(net, model, d.curve, CURVE, 0, M) : null) : null;
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
  // Handle for tests, the console and other modules (ctx.train: the walkthrough folds the panel).
  panel.nnTrain = ctx.train = {
    play, pause, step, reset, adapt, loadSample, stepSample, fold,
    get running() { return running; }, get eval() { return lastEval; },
    get open() { return !!ui.open; }, get folded() { return !!ui.fold; },
    point: i => (pts && i >= 0 && 2 * i < pts.length ? [pts[2 * i], pts[2 * i + 1]] : null),
  };
}
