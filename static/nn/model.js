// Net tab model: the network JSON, edit operations and the maths. Pure: no DOM.
// Binding spec: docs/NN_CONTRACT.md. Notes for consumers beyond the contract:
//
// Indexing
// - Every per-layer array returned by forward() and backward() is indexed by layer index l
//   (0 = inputs). Layer-0 slots are placeholders: fwd.z[0] = null, bwd.dZ[0] = [], bwd.db[0] = [],
//   bwd.dW[0] = []. bwd.dA[0] is real (dL/dx, the gradient with respect to the inputs).
// - matrices() is a list over l >= 1 (so matrices()[l - 1].l === l). Each entry also has `act`.
// - bwd.dW[l][t] is a rows x cols matrix parallel to matrices()[l - 1].terms[t].W.
//   Masked (missing-edge) entries are 0 in dW; bwd.edge only has existing edges.
// - bwd.node[id] = { da, dz }; dz is null on the input layer. bwd.note is a string or null.
//
// Losses (meta.loss): mse = 1/2 mean_i (a_i - y_i)^2. xent + softmax output = -sum_i y_i log p_i
// (dz = p * sum(y) - y, i.e. p - y for a proper distribution). xent + sigmoid output = binary
// cross-entropy, mean over outputs. xent on anything else falls back to mse and sets bwd.note.
//
// Edits (all mutate in place and keep validate(net) empty)
// - A valid net always has >= 2 layers: removeLayer refuses (returns false) at 2 layers.
//   removeLayer(net, id, { bridge: true, seed }) also wires the removed hidden layer's two
//   neighbours densely when no edge joins them yet (default: they are left unconnected).
// - Structural edits renumber labels that still look like defaults (x_{j}, h^{(l)}_{j},
//   \hat y_{j}, or empty); custom labels are never touched. relabel(net) does this on demand.
// - addLayer(net, at, { name, act, size = 2, dense = false, seed }): new nodes are placed in a
//   column between the neighbours. dense: true also wires it densely to both neighbour layers.
// - addNode(net, layer, { x, y, label, bias, value, target, params, index, connect, seed }):
//   index = order within the layer; connect: true wires it to every node of both neighbour
//   layers. On the output layer a new node gets target 0 if every other output has a target.
// - connect() without w picks a seeded random weight in (-1, 1).
// - connectDense() returns every edge id between the two layers; existing edges keep their w.
// - randomize(net, { seed, scheme, biases: 'zero' | 'small' | 'keep' }) (default 'zero').
// - setNode accepts x y label bias value target params layer (moving layers drops edges that
//   would become sideways/backward). Numeric strings are accepted; bad values are ignored.
// - edgeBetween(a, b) matches either direction.
//
// Tokens, ties and attention (docs/NN_ATTENTION.md)
// - layer.tokens / layer.groups shape a layer as token rows; node k is group floor(k / (tokens d)),
//   token floor(k / d) % tokens, feature k % d. softmax on such a layer runs per token (and group).
// - edge.tie: edges with the same tie share one weight (setWeight / connect set the whole group,
//   randomize draws once, trainStep steps by the summed gradient). edge.fixed: never trained,
//   randomized or re-weighted (setWeight refuses it). node.tie (an extension of the spec): nodes
//   with the same tie share one bias, e.g. 'b_Q:2'; setNode's bias sets the whole group.
// - An attention layer (kind 'attention') has no incoming edges, zero biases and act identity.
//   connect / connectDense refuse to wire into it. Other edits never refuse on token grounds: a
//   layer whose node count no longer splits into its tokens x groups loses tokens / groups, and an
//   attention layer whose Q, K, V input broke becomes a plain dense layer (settle()).
// - fwd.attn / bwd.attn are per-layer arrays (null except on attention layers). bwd.tie[tieId] is
//   the summed gradient of a shared weight or bias. bwd.db on an attention layer is all 0.
// - xent + softmax on a token layer is the mean over its tokens (and groups) of each one's -sum y log p.
// - Input nodes and attention nodes carry no bias tie (settle() drops them).
// - Helpers: tokenShape, tokenPos, reshape, attnSpec, tiedMatrices. PRESETS[key].lr (or null) is a
//   learning rate that suits the preset, also recorded as meta.train.lr.
//
// Extras: fmtg(x, digits) (fmt that keeps small values readable: 0.0034, 3.4e-4; the UI uses it
// for gradients), activate(act, z[], out?, seg?) -> a[], relabel(net), defaultLabel(net, nodeId),
// predict(net, X, { layer }) where layer is an index / id (activations of that layer) or 'all'
// (per sample, every layer's activations). PRESETS[key].dataset names a DATASETS key that suits
// the preset (or null); a built preset also records it as net.meta.train.dataset, the Train
// panel's setting. PRESETS[key].group is its New net menu section (PRESETS is in menu order) and
// .note one line on what to notice. ACTS[name].f / df are scalar; softmax is { vector: true } and its scalar
// f is the 2-class curve softmax([z, 0])_1 = sigmoid(z), for plotting only.

// ------------------------------------------------------------------ small helpers

const isNum = v => typeof v === 'number' && Number.isFinite(v);
const isObj = v => v !== null && typeof v === 'object' && !Array.isArray(v);
const num = (v, d) => (isNum(v) ? v : d);
// Lenient number: finite numbers and numeric strings; anything else -> undefined.
function toNum(v) {
  if (isNum(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return undefined;
}
const clampInt = (v, lo, hi, d) => (isNum(v) ? Math.max(lo, Math.min(hi, Math.round(v))) : d);

function filterInPlace(arr, keep) {
  let w = 0;
  for (let r = 0; r < arr.length; r++) if (keep(arr[r])) arr[w++] = arr[r];
  arr.length = w;
  return arr;
}

// ------------------------------------------------------------------ random numbers

function hashStr(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
function seedInt(seed) {
  if (seed === undefined || seed === null) return (Math.random() * 4294967296) >>> 0;
  if (typeof seed === 'number' && Number.isInteger(seed)) return seed >>> 0;
  return hashStr(String(seed));
}

// mulberry32. rng(seed) -> () => [0, 1). No seed -> a random one.
export function rng(seed) {
  let s = seedInt(seed) | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

// ------------------------------------------------------------------ formatting

// Fixed-point with ASCII '-', never "-0.00". Non-finite -> 'NaN' / 'inf' / '-inf'.
export function fmt(x, digits = 2) {
  const v = typeof x === 'number' ? x : typeof x === 'string' && x.trim() !== '' ? Number(x) : NaN;
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const d = clampInt(digits, 0, 20, 2);
  let s = v.toFixed(d);
  if (/^-0(\.0*)?$/.test(s)) s = s.slice(1);
  return s;
}

// fmt for values that are often tiny (gradients): |x| >= 0.01 as fmt(x, digits); a smaller value
// keeps two significant figures, '0.0034' down to 0.001 and '3.4e-4' below. Below 1e-9 it is
// rounding noise of an exact 0 (e.g. a key bias's gradient) and reads as fmt(0). ASCII '-'.
// For KaTeX, write the exponent as \mathrm{e}{-4}.
export function fmtg(x, digits = 2) {
  const v = typeof x === 'number' ? x : typeof x === 'string' && x.trim() !== '' ? Number(x) : NaN;
  if (Number.isFinite(v) && Math.abs(v) < 1e-9) return fmt(0, digits);
  if (!Number.isFinite(v) || Math.abs(v) >= 0.01) return fmt(v, digits);
  if (Math.abs(v) >= 0.001) return v.toFixed(4);
  const [m, e] = v.toExponential(1).split('e');
  return `${m}e${Number(e)}`;
}

// ------------------------------------------------------------------ activations

const LEAK = 0.1;
function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}
const softplus = z => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

export const ACTS = {
  identity: { label: 'Identity', tex: 'z', f: z => z, df: () => 1 },
  relu: { label: 'ReLU', tex: '\\max(0,\\, z)', f: z => (z > 0 ? z : 0), df: z => (z > 0 ? 1 : 0) },
  leaky: {
    label: 'Leaky ReLU', tex: '\\max(0.1z,\\, z)', slope: LEAK,
    f: z => (z > 0 ? z : LEAK * z), df: z => (z > 0 ? 1 : LEAK),
  },
  sigmoid: {
    label: 'Sigmoid', tex: '\\frac{1}{1+e^{-z}}',
    f: sigmoid, df: (z, a = sigmoid(z)) => a * (1 - a),
  },
  tanh: { label: 'tanh', tex: '\\tanh(z)', f: Math.tanh, df: (z, a = Math.tanh(z)) => 1 - a * a },
  softmax: {
    label: 'Softmax', tex: '\\frac{e^{z_i}}{\\sum_j e^{z_j}}', vector: true,
    f: sigmoid, df: (z, a = sigmoid(z)) => a * (1 - a),   // scalar view only; see header
  },
};

const ACT_ALIASES = {
  linear: 'identity', none: 'identity', id: 'identity', ident: 'identity',
  leakyrelu: 'leaky', leaky_relu: 'leaky', lrelu: 'leaky',
  logistic: 'sigmoid', sigma: 'sigmoid', sig: 'sigmoid', softargmax: 'softmax',
};
function actName(a) {
  if (typeof a !== 'string') return 'identity';
  const k = a.trim().toLowerCase().replace(/[\s-]+/g, '_');
  return ACTS[k] ? k : ACT_ALIASES[k] || ACT_ALIASES[k.replace(/_/g, '')] || 'identity';
}

// Apply a layer activation to a whole vector (softmax needs the layer). Works on typed arrays.
// seg: softmax runs on each run of seg entries (one token of a token layer); default the whole vector.
export function activate(act, z, out, seg) {
  const n = z.length;
  if (!out) out = new Array(n);
  if (act === 'softmax') {
    const w = seg > 0 && n % seg === 0 ? seg : n;
    for (let s0 = 0; s0 < n; s0 += w) {
      let m = -Infinity;
      for (let i = s0; i < s0 + w; i++) if (z[i] > m) m = z[i];
      let s = 0;
      for (let i = s0; i < s0 + w; i++) { const e = Math.exp(z[i] - m); out[i] = e; s += e; }
      for (let i = s0; i < s0 + w; i++) out[i] /= s;
    }
    return out;
  }
  const f = (ACTS[act] || ACTS.identity).f;
  for (let i = 0; i < n; i++) out[i] = f(z[i]);
  return out;
}

// ------------------------------------------------------------------ ids, construction

function maxSuffix(net) {
  let m = 0;
  for (const key of ['layers', 'nodes', 'edges']) {
    for (const o of Array.isArray(net[key]) ? net[key] : []) {
      const id = isObj(o) ? o.id : null;
      const hit = /(\d+)$/.exec(typeof id === 'number' ? String(id) : typeof id === 'string' ? id : '');
      if (hit) m = Math.max(m, Math.min(Number(hit[1]), 1e15));
    }
  }
  return m;
}

export function uid(net, prefix = 'id') {
  if (!isObj(net.meta)) net.meta = {};
  let n = net.meta.nextId;
  if (!Number.isInteger(n) || n < 1) n = maxSuffix(net) + 1;
  net.meta.nextId = n + 1;
  return `${prefix}${n}`;
}

export function emptyNet() {
  const net = { v: 1, layers: [], nodes: [], edges: [], meta: { title: 'Untitled', loss: 'mse', nextId: 1, train: {} } };
  net.layers.push({ id: uid(net, 'L'), name: 'Input', act: 'identity' });
  net.layers.push({ id: uid(net, 'L'), name: 'Output', act: 'identity' });
  return net;
}

export function clone(net) {
  return JSON.parse(JSON.stringify(net));
}

function blankNode(net, layerId) {
  return { id: uid(net, 'n'), layer: layerId, x: 0, y: 0, label: '', bias: 0, value: 0, target: null, params: {} };
}

function cleanParams(p) {
  const out = {};
  if (!isObj(p)) return out;
  for (const [k, v] of Object.entries(p)) {
    if (typeof v === 'string' || isNum(v)) out[k] = v;
    else if (typeof v === 'boolean') out[k] = String(v);
  }
  return out;
}

// ------------------------------------------------------------------ lookups

export function layerIndex(net, layerId) {
  return net.layers.findIndex(l => l.id === layerId);
}
// Layer reference -> index: an index, a layer id, or a layer object. -1 if unknown.
function layerRef(net, ref) {
  if (typeof ref === 'number') return Number.isInteger(ref) && ref >= 0 && ref < net.layers.length ? ref : -1;
  if (isObj(ref)) ref = ref.id;
  return layerIndex(net, ref);
}
export function node(net, id) {
  return net.nodes.find(n => n.id === id) || null;
}
export function edge(net, id) {
  return net.edges.find(e => e.id === id) || null;
}
export function nodeLayerIndex(net, nodeId) {
  const n = node(net, nodeId);
  return n ? layerIndex(net, n.layer) : -1;
}
export function nodesIn(net, layerIdOrIndex) {
  const l = layerRef(net, layerIdOrIndex);
  if (l < 0) return [];
  const id = net.layers[l].id;
  return net.nodes.filter(n => n.layer === id);
}
export function edgeBetween(net, fromId, toId) {
  return net.edges.find(e => e.from === fromId && e.to === toId)
    || net.edges.find(e => e.from === toId && e.to === fromId) || null;
}

// ------------------------------------------------------------------ labels

function labelFor(l, L, j) {
  if (l === 0) return `x_{${j}}`;
  if (l === L - 1) return `\\hat y_{${j}}`;
  return `h^{(${l})}_{${j}}`;
}
const DEFAULT_LABEL = /^\s*(?:x_\{?\d+\}?|h\^\{?\(\d+\)\}?_\{?\d+\}?|\\hat\s*\{?\s*y\s*\}?(?:_\{?\d+\}?)?)?\s*$/;

export function defaultLabel(net, nodeId) {
  const n = node(net, nodeId);
  if (!n) return '';
  const l = layerIndex(net, n.layer);
  if (l < 0) return '';
  return labelFor(l, net.layers.length, nodesIn(net, l).indexOf(n) + 1);
}

// Rewrite default-looking (or empty) labels to the current default. onlyEmpty: fill blanks only.
export function relabel(net, { onlyEmpty = false } = {}) {
  const L = net.layers.length;
  const li = new Map(net.layers.map((l, i) => [l.id, i]));
  const count = new Array(L).fill(0);
  for (const n of net.nodes) {
    const l = li.get(n.layer);
    if (l === undefined) continue;
    const j = ++count[l];
    const blank = typeof n.label !== 'string' || n.label.trim() === '';
    if (blank || (!onlyEmpty && DEFAULT_LABEL.test(n.label))) n.label = labelFor(l, L, j);
  }
  return net;
}

// ------------------------------------------------------------------ tokens, ties, attention

const QKV = ['Q', 'K', 'V'];
const isTie = t => typeof t === 'string' && t !== '';
const isTokens = t => Number.isInteger(t) && t >= 1;
const isGroups = g => Array.isArray(g) && g.length > 0 && g.every(s => typeof s === 'string' && s !== '')
  && new Set(g).size === g.length;
const isQKV = g => Array.isArray(g) && g.length === 3 && g.every((s, i) => s === QKV[i]);
const isAttn = layer => isObj(layer) && layer.kind === 'attention';

// Shape of a layer with `size` nodes: tokens rows of d features in each group. A plain vector
// (1 token, no groups) when the fields are absent or don't split the size. Attention layers have no groups.
function shapeOf(layer, size) {
  const groups = !isAttn(layer) && isGroups(layer?.groups) ? layer.groups : null;
  const tokens = isTokens(layer?.tokens) ? layer.tokens : 1;
  const G = groups ? groups.length : 1;
  if (size > 0 && size % (tokens * G) === 0) return { tokens, d: size / (tokens * G), groups };
  return { tokens: 1, d: size, groups: null };
}

// Why attention layer l (given every layer's size) can't work, or null when it can.
function attnProblem(layers, sizes, l) {
  const lay = layers[l], prev = layers[l - 1];
  if (!isObj(prev)) return 'an attention layer cannot be the input layer';
  if (!isQKV(prev.groups)) return 'the layer before an attention layer must have groups Q, K, V';
  const n = isTokens(lay.tokens) ? lay.tokens : 1;
  if ((isTokens(prev.tokens) ? prev.tokens : 1) !== n) return 'an attention layer needs the same tokens as its Q, K, V layer';
  const ps = sizes[l - 1];
  if (!(ps > 0) || ps % (3 * n)) return 'its Q, K, V layer must have 3 · tokens · d nodes';
  const d = ps / (3 * n);
  if (sizes[l] !== n * d) return `an attention layer needs tokens · d_v = ${n * d} nodes, not ${sizes[l]}`;
  const H = lay.heads === undefined ? 1 : lay.heads;
  if (!Number.isInteger(H) || H < 1 || d % H) return `heads must be an integer that divides d = ${d}`;
  return null;
}

// Geometry of a working attention layer: n tokens, d = d_k = d_v, H heads of dh = d / H.
function attnGeom(layers, sizes, l) {
  const lay = layers[l];
  if (!isAttn(lay) || attnProblem(layers, sizes, l)) return null;
  const n = isTokens(lay.tokens) ? lay.tokens : 1, d = sizes[l] / n;
  const H = lay.heads === undefined ? 1 : lay.heads, dh = d / H;
  return { l, n, d, H, dh, scale: isNum(lay.scale) ? lay.scale : 1 / Math.sqrt(dh), causal: lay.causal === true };
}

function layerSizes(net) {
  const count = new Map();
  for (const n of Array.isArray(net.nodes) ? net.nodes : []) if (isObj(n)) count.set(n.layer, (count.get(n.layer) || 0) + 1);
  return (Array.isArray(net.layers) ? net.layers : []).map(l => (isObj(l) ? count.get(l.id) || 0 : 0));
}

// After an edit (or a load): a layer whose node count no longer splits into its tokens and groups
// becomes a plain vector, and an attention layer whose Q, K, V input broke becomes a dense layer.
// A working attention layer keeps no incoming edges, zero biases (no bias ties) and act identity.
// Input nodes have no bias, so they drop bias ties too.
function settle(net) {
  const sizes = layerSizes(net);
  const input = net.layers[0]?.id;
  for (const n of net.nodes) if (n.layer === input && isTie(n.tie)) delete n.tie;
  net.layers.forEach((l, i) => {
    if (!('tokens' in l) && !('groups' in l)) return;
    const t = isTokens(l.tokens) ? l.tokens : 1, G = !isAttn(l) && isGroups(l.groups) ? l.groups.length : 1;
    if (!(sizes[i] > 0 && sizes[i] % (t * G) === 0)) { delete l.tokens; delete l.groups; }
  });
  net.layers.forEach((l, i) => {
    if (!isAttn(l)) return;
    if (attnProblem(net.layers, sizes, i)) {
      for (const k of ['kind', 'heads', 'causal', 'scale']) delete l[k];
      return;
    }
    l.act = 'identity';
    const ids = new Set();
    for (const n of net.nodes) {
      if (n.layer !== l.id) continue;
      ids.add(n.id);
      if (n.bias !== 0) n.bias = 0;
      if (isTie(n.tie)) delete n.tie;
    }
    if (net.edges.some(e => ids.has(e.to))) filterInPlace(net.edges, e => !ids.has(e.to));
  });
  return net;
}

// Set a weight, or every weight of its tie group.
function setShared(net, e, w) {
  if (isTie(e.tie)) { for (const x of net.edges) if (x.tie === e.tie) x.w = w; }
  else e.w = w;
}

// { tokens, d, groups } of a layer (index, id or object). A plain layer is 1 token of d = its size.
export function tokenShape(net, layer) {
  const l = layerRef(net, layer);
  if (l < 0) return { tokens: 1, d: 0, groups: null };
  const s = shapeOf(net.layers[l], nodesIn(net, l).length);
  return { tokens: s.tokens, d: s.d, groups: s.groups ? s.groups.slice() : null };
}

// Where a node sits in its layer's token grid: { l, index, g, group, token, feature } (0-based).
export function tokenPos(net, nodeId) {
  const n = node(net, nodeId);
  const l = n ? layerIndex(net, n.layer) : -1;
  if (l < 0) return null;
  const peers = nodesIn(net, l), k = peers.indexOf(n);
  const { tokens, d, groups } = shapeOf(net.layers[l], peers.length);
  const g = d ? Math.floor(k / (tokens * d)) : 0;
  return { l, index: k, g, group: groups ? groups[g] : null, token: d ? Math.floor(k / d) % tokens : 0, feature: d ? k % d : k };
}

// A layer vector (activations, biases, gradients...) as tokens x d matrices, one per group ('X'
// for a layer without groups). Missing entries read 0.
export function reshape(net, layer, vec) {
  const { tokens, d, groups } = tokenShape(net, layer);
  const out = {};
  (groups || ['X']).forEach((g, gi) => {
    out[g] = Array.from({ length: tokens }, (_, t) => Array.from({ length: d }, (_, f) => num(vec?.[gi * tokens * d + t * d + f], 0)));
  });
  return out;
}

// A working attention layer's settings, defaults filled: { l, tokens, d, heads, dh, scale, causal }, else null.
export function attnSpec(net, layer) {
  const l = layerRef(net, layer);
  const g = l < 0 ? null : attnGeom(net.layers, layerSizes(net), l);
  return g && { l, tokens: g.n, d: g.d, heads: g.H, dh: g.dh, scale: g.scale, causal: g.causal };
}

const TIE_ID = /^(.*):(\d+),(\d+)$/;

// The small shared matrices behind layer l, when every non-fixed edge into it is tied with an id
// '<name>:<i>,<j>' (else []). W is in the X W convention (rows = input feature, cols = output
// feature), so Q = X W_Q: an edge from input feature i to output feature j has weight W[i-1][j-1],
// and the layer's receiving-row matrix for one token is Wᵀ. Fixed edges (residuals) are left out.
// Each: { name, W, ties: (tieId | null)[][], k (source layer, or null if mixed), fromGroup, toGroup,
// edges: edgeId[], tokenwise } where tokenwise means the layer matrix is exactly I_tokens ⊗ Wᵀ.
export function tiedMatrices(net, layer) {
  const l = layerRef(net, layer);
  if (l < 1) return [];
  const P = plan(net), T = P.T[l];
  const byName = new Map();
  for (let q = 0; q < T.n; q++) {
    if (T.fixed[q]) continue;
    const e = T.edges[q], hit = isTie(e.tie) && TIE_ID.exec(e.tie);
    if (!hit || +hit[2] < 1 || +hit[3] < 1) return [];
    if (!byName.has(hit[1])) byName.set(hit[1], []);
    byName.get(hit[1]).push({ q, e, i: +hit[2], j: +hit[3] });
  }
  const at = (li, k) => {
    const s = P.shapes[li], d = s.d || 1;
    return { g: Math.floor(k / (s.tokens * d)), t: Math.floor(k / d) % s.tokens, f: k % d, s };
  };
  const out = [];
  for (const [name, list] of byName) {
    const R = Math.max(...list.map(x => x.i)), C = Math.max(...list.map(x => x.j));
    const W = Array.from({ length: R }, () => new Array(C).fill(0));
    const ties = Array.from({ length: R }, () => new Array(C).fill(null));
    const ks = new Set(), from = new Set(), to = new Set();
    let tokenwise = true;
    for (const { q, e, i, j } of list) {
      W[i - 1][j - 1] = T.w[q];
      ties[i - 1][j - 1] = e.tie;
      ks.add(T.k[q]);
      const sp = at(T.k[q], T.j[q]), tp = at(l, T.i[q]);
      from.add(sp.s.groups ? sp.s.groups[sp.g] : null);
      to.add(tp.s.groups ? tp.s.groups[tp.g] : null);
      if (sp.s.tokens !== tp.s.tokens || sp.t !== tp.t || sp.f !== i - 1 || tp.f !== j - 1) tokenwise = false;
    }
    const cells = ties.flat().filter(Boolean).length;
    if (from.size !== 1 || to.size !== 1 || list.length !== cells * P.shapes[l].tokens) tokenwise = false;
    out.push({
      name, W, ties, k: ks.size === 1 ? [...ks][0] : null,
      fromGroup: from.size === 1 ? [...from][0] : null, toGroup: to.size === 1 ? [...to][0] : null,
      edges: list.map(x => x.e.id), tokenwise,
    });
  }
  return out;
}

// ------------------------------------------------------------------ validation / repair

export function validate(net) {
  const errs = [];
  if (!isObj(net)) return ['net is not an object'];
  if (net.v !== 1) errs.push('v must be 1');
  for (const key of ['layers', 'nodes', 'edges']) if (!Array.isArray(net[key])) errs.push(`${key} must be an array`);
  if (errs.some(e => e.endsWith('an array'))) return errs;
  if (net.layers.length < 2) errs.push('a net needs at least 2 layers (inputs and outputs)');

  const all = new Set();
  const layerAt = new Map();
  net.layers.forEach((l, i) => {
    if (!isObj(l)) return errs.push(`layer ${i} is not an object`);
    if (typeof l.id !== 'string' || !l.id) errs.push(`layer ${i}: bad id`);
    else if (all.has(l.id)) errs.push(`layer ${l.id}: duplicate id`);
    else { all.add(l.id); layerAt.set(l.id, i); }
    if (typeof l.name !== 'string') errs.push(`layer ${l.id}: name must be a string`);
    if (!Object.hasOwn(ACTS, l.act)) errs.push(`layer ${l.id}: unknown act ${JSON.stringify(l.act)}`);
  });

  const nodeLayer = new Map();
  net.nodes.forEach((n, i) => {
    if (!isObj(n)) return errs.push(`node ${i} is not an object`);
    const tag = `node ${typeof n.id === 'string' ? n.id : i}`;
    if (typeof n.id !== 'string' || !n.id) errs.push(`${tag}: bad id`);
    else if (all.has(n.id)) errs.push(`${tag}: duplicate id`);
    else all.add(n.id);
    if (!layerAt.has(n.layer)) errs.push(`${tag}: layer ${JSON.stringify(n.layer)} not found`);
    else if (typeof n.id === 'string') nodeLayer.set(n.id, layerAt.get(n.layer));
    for (const k of ['x', 'y', 'bias', 'value']) if (!isNum(n[k])) errs.push(`${tag}: ${k} must be a finite number`);
    if (typeof n.label !== 'string') errs.push(`${tag}: label must be a string`);
    if (!(n.target === null || isNum(n.target))) errs.push(`${tag}: target must be a number or null`);
    if (!isObj(n.params)) errs.push(`${tag}: params must be an object`);
    else for (const [k, v] of Object.entries(n.params)) {
      if (!(typeof v === 'string' || isNum(v))) errs.push(`${tag}: param ${k} must be a string or number`);
    }
    if (n.tie !== undefined && n.tie !== null && !isTie(n.tie)) errs.push(`${tag}: tie must be a non-empty string or null`);
  });

  // token shapes and attention layers
  const sizes = layerSizes(net);
  const attn = new Set();
  net.layers.forEach((l, i) => {
    if (!isObj(l)) return;
    const tag = `layer ${l.id}`;
    const tokOk = l.tokens === undefined || isTokens(l.tokens), grpOk = l.groups === undefined || isGroups(l.groups);
    if (!tokOk) errs.push(`${tag}: tokens must be an integer >= 1`);
    if (!grpOk) errs.push(`${tag}: groups must be distinct non-empty strings`);
    if (l.kind !== undefined && l.kind !== 'dense' && l.kind !== 'attention') errs.push(`${tag}: unknown kind ${JSON.stringify(l.kind)}`);
    const att = l.kind === 'attention';
    if (att && l.groups !== undefined) errs.push(`${tag}: an attention layer has no groups`);
    if (tokOk && grpOk && (l.tokens !== undefined || l.groups !== undefined)) {
      const t = l.tokens ?? 1, G = !att && l.groups ? l.groups.length : 1;
      if (!(sizes[i] > 0 && sizes[i] % (t * G) === 0)) errs.push(`${tag}: ${sizes[i]} nodes don't split into ${t} tokens x ${G} groups`);
    }
    if (!att) return;
    attn.add(l.id);
    const p = attnProblem(net.layers, sizes, i);
    if (p) errs.push(`${tag}: ${p}`);
    if (l.act !== 'identity') errs.push(`${tag}: an attention layer's act must be identity`);
    if (l.causal !== undefined && typeof l.causal !== 'boolean') errs.push(`${tag}: causal must be a boolean`);
    if (l.scale !== undefined && !isNum(l.scale)) errs.push(`${tag}: scale must be a finite number`);
  });

  const pairs = new Set();
  const tieW = new Map();
  net.edges.forEach((e, i) => {
    if (!isObj(e)) return errs.push(`edge ${i} is not an object`);
    const tag = `edge ${typeof e.id === 'string' ? e.id : i}`;
    if (typeof e.id !== 'string' || !e.id) errs.push(`${tag}: bad id`);
    else if (all.has(e.id)) errs.push(`${tag}: duplicate id`);
    else all.add(e.id);
    const lf = nodeLayer.get(e.from), lt = nodeLayer.get(e.to);
    if (lf === undefined) errs.push(`${tag}: from ${JSON.stringify(e.from)} not found`);
    if (lt === undefined) errs.push(`${tag}: to ${JSON.stringify(e.to)} not found`);
    if (lf !== undefined && lt !== undefined && lf >= lt) errs.push(`${tag}: must go forward (layer ${lf} -> ${lt})`);
    if (lt !== undefined && attn.has(net.layers[lt].id)) errs.push(`${tag}: an attention layer has no incoming edges`);
    const key = `${e.from}\u0000${e.to}`;
    if (pairs.has(key)) errs.push(`${tag}: duplicate edge ${e.from} -> ${e.to}`);
    pairs.add(key);
    if (!isNum(e.w)) errs.push(`${tag}: w must be a finite number`);
    if (e.tie !== undefined && e.tie !== null && !isTie(e.tie)) errs.push(`${tag}: tie must be a non-empty string or null`);
    if (e.fixed !== undefined && typeof e.fixed !== 'boolean') errs.push(`${tag}: fixed must be a boolean`);
    if (e.fixed === true && isTie(e.tie)) errs.push(`${tag}: an edge cannot be both fixed and tied`);
    if (isTie(e.tie) && isNum(e.w)) {
      if (!tieW.has(e.tie)) tieW.set(e.tie, e.w);
      else if (tieW.get(e.tie) !== e.w) errs.push(`${tag}: tie ${e.tie} has differing weights`);
    }
  });

  // shared and attention biases
  const tieB = new Map();
  net.nodes.forEach((n, i) => {
    if (!isObj(n)) return;
    const tag = `node ${typeof n.id === 'string' ? n.id : i}`;
    if (attn.has(n.layer)) {
      if (isNum(n.bias) && n.bias !== 0) errs.push(`${tag}: an attention layer's bias must be 0`);
      if (isTie(n.tie)) errs.push(`${tag}: an attention layer has no bias to share`);
      return;
    }
    if (!isTie(n.tie)) return;
    if (n.layer === net.layers[0]?.id) return errs.push(`${tag}: an input node has no bias to share`);
    if (tieW.has(n.tie)) errs.push(`${tag}: tie ${n.tie} is also an edge tie`);
    if (isNum(n.bias)) {
      if (!tieB.has(n.tie)) tieB.set(n.tie, n.bias);
      else if (tieB.get(n.tie) !== n.bias) errs.push(`${tag}: tie ${n.tie} has differing biases`);
    }
  });

  if (!isObj(net.meta)) errs.push('meta must be an object');
  else {
    const m = net.meta;
    if (typeof m.title !== 'string') errs.push('meta.title must be a string');
    if (m.loss !== 'mse' && m.loss !== 'xent') errs.push('meta.loss must be mse or xent');
    if (!Number.isInteger(m.nextId) || m.nextId <= maxSuffix(net)) errs.push('meta.nextId must exceed every numeric id suffix');
    if (m.train !== undefined && !isObj(m.train)) errs.push('meta.train must be an object');
  }
  return errs;
}

const LOSS_ALIASES = { xent: 'xent', ce: 'xent', crossentropy: 'xent', cross_entropy: 'xent', bce: 'xent', nll: 'xent' };

// Optional token / tie fields: keep valid values, coerce lenient ones, drop the rest. Never adds a field.
const toBool = v => (v === 'true' ? true : v === 'false' ? false : v);
function cleanLayerFields(l) {
  if ('tokens' in l) { const t = toNum(l.tokens); if (isTokens(t)) l.tokens = t; else delete l.tokens; }
  if ('groups' in l && !isGroups(l.groups)) delete l.groups;
  if ('kind' in l) {
    const k = typeof l.kind === 'string' ? l.kind.trim().toLowerCase() : '';
    if (k === 'dense' || k === 'attention') l.kind = k; else delete l.kind;
  }
  if (l.kind !== 'attention') return l;
  delete l.groups;
  l.act = 'identity';
  if ('heads' in l) { const h = toNum(l.heads); if (isTokens(h)) l.heads = h; else delete l.heads; }
  if ('causal' in l) { const c = toBool(l.causal); if (typeof c === 'boolean') l.causal = c; else delete l.causal; }
  if ('scale' in l) { const s = toNum(l.scale); if (s !== undefined) l.scale = s; else delete l.scale; }
  return l;
}
function cleanTie(o) {
  if (!('tie' in o) || o.tie === null || isTie(o.tie)) return o;
  if (isNum(o.tie)) o.tie = String(o.tie); else delete o.tie;
  return o;
}
function cleanEdgeFields(e) {
  cleanTie(e);
  if ('fixed' in e) { const f = toBool(e.fixed); if (typeof f === 'boolean') e.fixed = f; else delete e.fixed; }
  if (e.fixed === true && isTie(e.tie)) delete e.tie;
  return e;
}
// Every member of a tie group takes its first member's weight (or bias). A bias tie that collides
// with an edge tie is dropped.
function equalizeTies(net) {
  const ew = new Map();
  for (const e of net.edges) {
    if (!isTie(e.tie)) continue;
    if (ew.has(e.tie)) e.w = ew.get(e.tie); else ew.set(e.tie, e.w);
  }
  const nb = new Map();
  for (const n of net.nodes) {
    if (!isTie(n.tie)) continue;
    if (ew.has(n.tie)) { delete n.tie; continue; }
    if (nb.has(n.tie)) n.bias = nb.get(n.tie); else nb.set(n.tie, n.bias);
  }
}

// Repair loaded JSON (object or string) into a valid net. Valid nets come back deep-equal.
export function normalize(input) {
  let src = input;
  if (typeof src === 'string') { try { src = JSON.parse(src); } catch { src = null; } }
  if (!isObj(src)) return emptyNet();

  const meta = isObj(src.meta) ? { ...src.meta } : {};
  const out = { v: 1, layers: [], nodes: [], edges: [], meta };
  for (const k of Object.keys(src)) if (!(k in out)) out[k] = src[k];
  meta.title = typeof meta.title === 'string' ? meta.title : 'Untitled';
  meta.loss = typeof meta.loss === 'string' && LOSS_ALIASES[meta.loss.trim().toLowerCase()] ? 'xent' : 'mse';
  meta.train = isObj(meta.train) ? meta.train : {};
  const floor = maxSuffix(src) + 1;
  meta.nextId = Number.isInteger(meta.nextId) && meta.nextId >= floor ? meta.nextId : floor;

  const idOf = v => (typeof v === 'string' && v ? v : isNum(v) ? String(v) : null);
  const arr = v => (Array.isArray(v) ? v : []);
  const all = new Set();

  // layers
  for (const r of arr(src.layers)) {
    if (!isObj(r)) continue;
    let id = idOf(r.id);
    if (id && out.layers.some(l => l.id === id)) continue;          // duplicate layer: drop
    if (!id) id = uid(out, 'L');
    all.add(id);
    out.layers.push(cleanLayerFields({ ...r, id, name: typeof r.name === 'string' ? r.name : '', act: actName(r.act) }));
  }
  const layerIds = new Set(out.layers.map(l => l.id));
  const layerOfRef = ref => {
    const s = idOf(ref);
    if (s && layerIds.has(s)) return s;
    if (Number.isInteger(ref) && ref >= 0 && ref < out.layers.length) return out.layers[ref].id;
    return null;
  };

  // nodes
  const rename = new Map();
  const nodeIds = new Set();
  const unplaced = [];
  for (const r of arr(src.nodes)) {
    if (!isObj(r)) continue;
    const layer = layerOfRef(r.layer);
    if (!layer) continue;
    let id = idOf(r.id);
    if (id && nodeIds.has(id)) continue;                             // duplicate node: drop
    if (!id || all.has(id)) {
      const fresh = uid(out, 'n');
      if (id) rename.set(id, fresh);                                // collided with a layer id
      id = fresh;
    }
    all.add(id); nodeIds.add(id);
    const x = toNum(r.x), y = toNum(r.y), bias = toNum(r.bias), value = toNum(r.value), target = toNum(r.target);
    const n = {
      ...r, id, layer, x: x ?? 0, y: y ?? 0,
      label: typeof r.label === 'string' ? r.label : isNum(r.label) ? String(r.label) : '',
      bias: bias ?? 0, value: value ?? 0, target: target ?? null, params: cleanParams(r.params),
    };
    if (x === undefined || y === undefined) unplaced.push(n);
    out.nodes.push(cleanTie(n));
  }

  while (out.layers.length < 2) {
    const id = uid(out, 'L');
    all.add(id);
    out.layers.push({ id, name: '', act: 'identity' });
  }
  out.layers.forEach((l, i) => {
    if (!l.name) l.name = i === 0 ? 'Input' : i === out.layers.length - 1 ? 'Output' : `Hidden ${i}`;
  });

  // edges
  const where = new Map();
  out.nodes.forEach(n => where.set(n.id, layerIndex(out, n.layer)));
  const pairs = new Set();
  const edgeIds = new Set();
  for (const r of arr(src.edges)) {
    if (!isObj(r)) continue;
    let from = idOf(r.from), to = idOf(r.to);
    from = rename.get(from) || from; to = rename.get(to) || to;
    if (!where.has(from) || !where.has(to)) continue;
    const lf = where.get(from), lt = where.get(to);
    if (lf === lt) continue;
    if (lf > lt) [from, to] = [to, from];
    const key = `${from}\u0000${to}`;
    if (pairs.has(key)) continue;
    pairs.add(key);
    let id = idOf(r.id);
    if (!id || all.has(id) || edgeIds.has(id)) id = uid(out, 'e');
    all.add(id); edgeIds.add(id);
    out.edges.push(cleanEdgeFields({ ...r, id, from, to, w: toNum(r.w) ?? 0 }));
  }

  settle(out);
  equalizeTies(out);
  relabel(out, { onlyEmpty: true });
  if (unplaced.length) {
    if (unplaced.length === out.nodes.length) autoLayout(out);
    else for (const n of unplaced) placeNode(out, n);
  }
  return out;
}

// ------------------------------------------------------------------ layout helpers

const COL = 160, ROW = 80;

function meanX(net, l) {
  const ns = nodesIn(net, l);
  return ns.length ? ns.reduce((s, n) => s + n.x, 0) / ns.length : null;
}
function centreY(net) {
  const ys = net.nodes.map(n => n.y).filter(isNum);
  return ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : 280;
}
// x of layer l's column: its own nodes, else interpolated from the nearest non-empty neighbours.
function columnX(net, l) {
  const own = meanX(net, l);
  if (own !== null) return own;
  let ip = -1, xp = null, iq = -1, xq = null;
  for (let i = l - 1; i >= 0 && xp === null; i--) { xp = meanX(net, i); ip = i; }
  for (let i = l + 1; i < net.layers.length && xq === null; i++) { xq = meanX(net, i); iq = i; }
  if (xp !== null && xq !== null) return xp + ((xq - xp) * (l - ip)) / (iq - ip);
  if (xp !== null) return xp + COL * (l - ip);
  if (xq !== null) return xq - COL * (iq - l);
  return 450;
}
// Give a node without a position a spot at the bottom of its layer's column.
function placeNode(net, n) {
  const l = layerIndex(net, n.layer);
  const peers = nodesIn(net, l).filter(p => p !== n);
  n.x = peers.length ? peers.reduce((s, p) => s + p.x, 0) / peers.length : columnX(net, l);
  n.y = peers.length ? Math.max(...peers.map(p => p.y)) + ROW : centreY(net);
}

// Token layers leave an extra quarter row between tokens, and another between groups.
export function autoLayout(net, { width = 900, height = 560 } = {}) {
  const L = net.layers.length;
  const mx = Math.min(110, width * 0.12), my = Math.min(50, height * 0.1);
  net.layers.forEach((layer, l) => {
    const ns = net.nodes.filter(n => n.layer === layer.id);
    const x = L === 1 ? width / 2 : mx + ((width - 2 * mx) * l) / (L - 1);
    const { tokens, d, groups } = shapeOf(layer, ns.length);
    const flat = tokens === 1 && !groups;
    const off = ns.map((_, k) => (flat ? k : k + 0.25 * Math.floor(k / d) + 0.25 * Math.floor(k / (tokens * d))));
    const span = ns.length ? off[ns.length - 1] : 0;
    const gap = span > 0 ? Math.min(96, (height - 2 * my) / span) : 0;
    ns.forEach((n, j) => {
      n.x = Math.round(x * 10) / 10;
      n.y = Math.round((height / 2 + (off[j] - span / 2) * gap) * 10) / 10;
    });
  });
  return net;
}

// ------------------------------------------------------------------ edits

function sampleW(r, scheme, fanIn, fanOut) {
  fanIn = Math.max(1, fanIn); fanOut = Math.max(1, fanOut);
  if (scheme === 'he') return gauss(r) * Math.sqrt(2 / fanIn);
  if (scheme === 'small') return gauss(r) * 0.1;
  const a = Math.sqrt(6 / (fanIn + fanOut));
  return (2 * r() - 1) * a;
}

export function addLayer(net, atIndex, { name, act, size = 2, dense = false, seed } = {}) {
  const L0 = net.layers.length;
  const at = clampInt(atIndex, 0, L0, L0);
  const hidden = at > 0 && at < L0;
  const id = uid(net, 'L');
  net.layers.splice(at, 0, {
    id,
    name: typeof name === 'string' && name ? name : at === 0 ? 'Input' : at === L0 ? 'Output' : 'Hidden',
    act: typeof act === 'string' && Object.hasOwn(ACTS, act) ? act : hidden ? 'tanh' : 'identity',
  });
  const n = clampInt(size, 0, 256, 2);
  const x = columnX(net, at), cy = centreY(net);
  // A new output layer inherits "the backward pass is on" from the old one.
  const oldOut = at === L0 && L0 > 1 ? nodesIn(net, L0 - 1) : [];
  const keepTargets = oldOut.length > 0 && oldOut.every(o => isNum(o.target));
  for (let j = 0; j < n; j++) {
    const nd = blankNode(net, id);
    nd.x = x;
    nd.y = cy + (j - (n - 1) / 2) * ROW;
    if (keepTargets) nd.target = 0;
    net.nodes.push(nd);
  }
  settle(net);   // a layer between Q, K, V and their attention layer turns that layer dense
  if (dense) {
    const r = rng(seed ?? net.meta.nextId);
    if (at > 0) connectDense(net, at - 1, at, { seed: Math.floor(r() * 1e9) });
    if (at < net.layers.length - 1) connectDense(net, at, at + 1, { seed: Math.floor(r() * 1e9) });
  }
  relabel(net);
  return id;
}

export function removeLayer(net, id, { bridge = false, seed } = {}) {
  const i = layerIndex(net, id);
  if (i < 0 || net.layers.length <= 2) return false;
  const gone = new Set(net.nodes.filter(n => n.layer === id).map(n => n.id));
  filterInPlace(net.edges, e => !gone.has(e.from) && !gone.has(e.to));
  filterInPlace(net.nodes, n => n.layer !== id);
  net.layers.splice(i, 1);
  settle(net);   // removing a Q, K, V layer turns its attention layer dense
  // bridge: a removed hidden layer's neighbours (now i - 1 and i) get wired densely, unless some
  // edge already joins them. Seeded init, He for a ReLU-like receiving layer, else Xavier.
  if (bridge && i > 0 && i < net.layers.length) {
    const prev = new Set(nodesIn(net, i - 1).map(n => n.id)), next = new Set(nodesIn(net, i).map(n => n.id));
    if (prev.size && next.size && !net.edges.some(e => prev.has(e.from) && next.has(e.to))) {
      const scheme = /relu|leaky/.test(net.layers[i].act) ? 'he' : 'xavier';
      connectDense(net, i - 1, i, { seed: seed ?? net.meta?.nextId ?? 1, scheme });
    }
  }
  relabel(net);
  return true;
}

// patch: { name, act } for any layer; { causal: bool, heads: int dividing d, scale: number | null
// (null = the default 1/sqrt(d_k / heads)) } for an attention layer, whose act stays identity.
export function setLayer(net, id, patch = {}) {
  const l = net.layers.find(x => x.id === id);
  if (!l || !isObj(patch)) return false;
  if (typeof patch.name === 'string') l.name = patch.name;
  const att = isAttn(l);
  if (typeof patch.act === 'string' && Object.hasOwn(ACTS, patch.act) && (!att || patch.act === 'identity')) l.act = patch.act;
  if (att) {
    if (typeof patch.causal === 'boolean') l.causal = patch.causal;
    if (patch.scale === null) delete l.scale;
    else if (toNum(patch.scale) !== undefined) l.scale = toNum(patch.scale);
    const h = toNum(patch.heads);
    if (isTokens(h)) {
      const had = 'heads' in l, old = l.heads;
      l.heads = h;
      if (attnProblem(net.layers, layerSizes(net), net.layers.indexOf(l))) { if (had) l.heads = old; else delete l.heads; }
    }
  }
  return true;
}

function insertAt(net, n, index) {
  const peers = net.nodes.filter(m => m.layer === n.layer);
  const at = clampInt(index, 0, peers.length, peers.length);
  if (!peers.length) net.nodes.push(n);
  else if (at < peers.length) net.nodes.splice(net.nodes.indexOf(peers[at]), 0, n);
  else net.nodes.splice(net.nodes.indexOf(peers[peers.length - 1]) + 1, 0, n);
}

export function addNode(net, layerId, opts = {}) {
  const l = layerRef(net, layerId);
  if (l < 0) return null;
  const L = net.layers.length;
  const lay = net.layers[l];
  const peers = nodesIn(net, l);
  const n = blankNode(net, lay.id);
  if (L > 1 && l === L - 1 && peers.length && peers.every(p => isNum(p.target))) n.target = 0;
  const hasX = toNum(opts.x) !== undefined, hasY = toNum(opts.y) !== undefined;
  insertAt(net, n, opts.index);
  applyNodePatch(net, n, { ...opts, layer: undefined });
  settle(net);   // one more node: a token layer becomes a plain vector (and its attention layer dense)
  if (!hasX || !hasY) {
    const px = n.x, py = n.y;
    placeNode(net, n);
    if (hasX) n.x = px;
    if (hasY) n.y = py;
  }
  if (opts.connect) {
    const r = rng(opts.seed ?? net.meta.nextId);
    const into = k => k >= 0 && k < L && !isAttn(net.layers[k]);
    const prev = l > 0 && into(l) ? nodesIn(net, l - 1) : [], next = l < L - 1 && into(l + 1) ? nodesIn(net, l + 1) : [];
    for (const p of prev) addEdge(net, p.id, n.id, sampleW(r, 'xavier', prev.length, next.length || 1));
    for (const q of next) addEdge(net, n.id, q.id, sampleW(r, 'xavier', peers.length + 1, 1));
  }
  relabel(net);
  return n.id;
}

export function removeNode(net, id) {
  const i = net.nodes.findIndex(n => n.id === id);
  if (i < 0) return false;
  net.nodes.splice(i, 1);
  filterInPlace(net.edges, e => e.from !== id && e.to !== id);
  settle(net);
  relabel(net);
  return true;
}

// Returns true when the node's layer changed. A bias sets the node's whole bias tie group; an
// attention layer's nodes keep bias 0.
function applyNodePatch(net, n, patch) {
  for (const k of ['x', 'y', 'value']) {
    const v = toNum(patch[k]);
    if (v !== undefined) n[k] = v;
  }
  const bv = toNum(patch.bias);
  if (bv !== undefined && !isAttn(net.layers.find(l => l.id === n.layer))) {
    if (isTie(n.tie)) { for (const m of net.nodes) if (m.tie === n.tie) m.bias = bv; }
    else n.bias = bv;
  }
  if (typeof patch.label === 'string') n.label = patch.label;
  else if (isNum(patch.label)) n.label = String(patch.label);
  if ('target' in patch && patch.target !== undefined) {
    const v = toNum(patch.target);
    if (v !== undefined) n.target = v;
    else if (patch.target === null || patch.target === '') n.target = null;
  }
  if (isObj(patch.params)) n.params = cleanParams(patch.params);
  if (patch.layer !== undefined) {
    const l = layerRef(net, patch.layer);
    if (l >= 0 && net.layers[l].id !== n.layer) {
      n.layer = net.layers[l].id;
      const li = new Map(net.layers.map((x, i) => [x.id, i]));
      const at = new Map(net.nodes.map(m => [m.id, li.get(m.layer)]));
      filterInPlace(net.edges, e => at.get(e.from) < at.get(e.to));
      settle(net);
      return true;
    }
  }
  return false;
}

export function setNode(net, id, patch = {}) {
  const n = node(net, id);
  if (!n || !isObj(patch)) return false;
  if (applyNodePatch(net, n, patch)) relabel(net);
  return true;
}

export function moveNode(net, id, toIndex) {
  const i = net.nodes.findIndex(n => n.id === id);
  if (i < 0) return false;
  const [n] = net.nodes.splice(i, 1);
  const peers = net.nodes.filter(m => m.layer === n.layer);
  if (!peers.length) net.nodes.splice(i, 0, n);
  else insertAt(net, n, toIndex);
  relabel(net);
  return true;
}

// extra: optional fields for the new edge ({ tie } or { fixed: true }).
function addEdge(net, from, to, w, extra) {
  const id = uid(net, 'e');
  net.edges.push(extra ? { id, from, to, w, ...extra } : { id, from, to, w });
  return id;
}

// null into an attention layer. A w on an existing tied edge sets its group; a fixed edge keeps its w.
export function connect(net, fromId, toId, w) {
  let a = node(net, fromId), b = node(net, toId);
  if (!a || !b) return null;
  let la = layerIndex(net, a.layer), lb = layerIndex(net, b.layer);
  if (la < 0 || lb < 0 || la === lb) return null;
  if (la > lb) { [a, b] = [b, a]; [la, lb] = [lb, la]; }
  if (isAttn(net.layers[lb])) return null;
  const wv = toNum(w);
  const old = net.edges.find(e => e.from === a.id && e.to === b.id);
  if (old) {
    if (wv !== undefined && old.fixed !== true) setShared(net, old, wv);
    return old.id;
  }
  return addEdge(net, a.id, b.id, wv ?? rng(net.meta?.nextId ?? 1)() * 2 - 1);
}

export function disconnect(net, edgeId) {
  const i = net.edges.findIndex(e => e.id === edgeId);
  if (i < 0) return false;
  net.edges.splice(i, 1);
  return true;
}

// A tied edge sets its whole group. A fixed edge refuses (false).
export function setWeight(net, edgeId, w) {
  const e = edge(net, edgeId), v = toNum(w);
  if (!e || v === undefined || e.fixed === true) return false;
  setShared(net, e, v);
  return true;
}

// Every pair between two layers. New edges: w if given, else a seeded init. Returns all edge ids.
// Nothing into an attention layer ([]).
export function connectDense(net, fromLayerId, toLayerId, { seed, scheme = 'xavier', w } = {}) {
  let a = layerRef(net, fromLayerId), b = layerRef(net, toLayerId);
  if (a < 0 || b < 0 || a === b) return [];
  if (a > b) [a, b] = [b, a];
  if (isAttn(net.layers[b])) return [];
  const src = nodesIn(net, a), dst = nodesIn(net, b);
  const have = new Map(net.edges.map(e => [`${e.from}\u0000${e.to}`, e.id]));
  const r = rng(seed ?? net.meta?.nextId ?? 1);
  const wv = toNum(w);
  const ids = [];
  for (const t of dst) {
    for (const s of src) {
      const old = have.get(`${s.id}\u0000${t.id}`);
      ids.push(old ?? addEdge(net, s.id, t.id, wv ?? sampleW(r, scheme, src.length, dst.length)));
    }
  }
  return ids;
}

// Fixed edges keep their w; a tie group (edges or biases) draws one value; attention biases stay 0.
export function randomize(net, { seed, scheme = 'xavier', biases = 'zero' } = {}) {
  const r = rng(seed);
  const fanIn = new Map(), fanOut = new Map();
  for (const e of net.edges) {
    fanIn.set(e.to, (fanIn.get(e.to) || 0) + 1);
    fanOut.set(e.from, (fanOut.get(e.from) || 0) + 1);
  }
  const drawn = new Map();
  for (const e of net.edges) {
    if (e.fixed === true) continue;
    if (isTie(e.tie) && drawn.has(e.tie)) { e.w = drawn.get(e.tie); continue; }
    e.w = sampleW(r, scheme, fanIn.get(e.to), fanOut.get(e.from));
    if (isTie(e.tie)) drawn.set(e.tie, e.w);
  }
  if (biases !== 'keep') {
    const input = net.layers[0]?.id, attn = new Set(net.layers.filter(isAttn).map(l => l.id));
    const shared = new Map();
    for (const n of net.nodes) {
      if (n.layer === input) continue;
      if (attn.has(n.layer)) { n.bias = 0; continue; }
      if (isTie(n.tie) && shared.has(n.tie)) { n.bias = shared.get(n.tie); continue; }
      n.bias = biases === 'small' ? (2 * r() - 1) * 0.1 : 0;
      if (isTie(n.tie)) shared.set(n.tie, n.bias);
    }
  }
  return net;
}

// ------------------------------------------------------------------ compiled plan

// Everything the maths needs, in flat arrays. Nodes whose layer is missing and edges that
// don't go forward are skipped, so a damaged net degrades instead of throwing.
function plan(net) {
  const layers = net.layers || [];
  const L = layers.length;
  const li = new Map();
  layers.forEach((l, i) => li.set(l.id, i));
  const nodes = layers.map(() => []);
  const where = new Map();
  for (const n of net.nodes || []) {
    const l = li.get(n.layer);
    if (l === undefined || where.has(n.id)) continue;
    where.set(n.id, [l, nodes[l].length]);
    nodes[l].push(n);
  }
  const sizes = nodes.map(ns => ns.length);
  const shapes = layers.map((l, i) => shapeOf(l, sizes[i]));
  // A working attention layer ignores stray incoming edges and biases (normalize drops them).
  const att = layers.map((_, l) => attnGeom(layers, sizes, l));
  const lists = layers.map(() => []);
  for (const e of net.edges || []) {
    const f = where.get(e.from), t = where.get(e.to);
    if (!f || !t || f[0] >= t[0] || att[t[0]]) continue;
    lists[t[0]].push([e, t[1], f[0], f[1]]);
  }
  // Shared parameters: tie id -> [[l, q], ...] (edges) and [[l, i], ...] (biases).
  const ties = new Map(), bties = new Map();
  const T = lists.map((list, l) => {
    const n = list.length;
    const T = {
      n, i: new Int32Array(n), k: new Int32Array(n), j: new Int32Array(n), w: new Float64Array(n), edges: new Array(n),
      fixed: new Uint8Array(n), tied: new Uint8Array(n),
    };
    list.forEach(([e, i, k, j], q) => {
      T.i[q] = i; T.k[q] = k; T.j[q] = j; T.w[q] = num(e.w, 0); T.edges[q] = e;
      if (e.fixed === true) T.fixed[q] = 1;
      else if (isTie(e.tie)) {
        T.tied[q] = 1;
        if (!ties.has(e.tie)) ties.set(e.tie, []);
        ties.get(e.tie).push([l, q]);
      }
    });
    return T;
  });
  nodes.forEach((ns, l) => {
    if (l === 0 || att[l]) return;
    ns.forEach((n, i) => {
      if (!isTie(n.tie)) return;
      if (!bties.has(n.tie)) bties.set(n.tie, []);
      bties.get(n.tie).push([l, i]);
    });
  });
  const b = nodes.map((ns, l) => (att[l] ? new Float64Array(ns.length) : Float64Array.from(ns, n => num(n.bias, 0))));
  const acts = layers.map((l, i) => (att[i] ? 'identity' : Object.hasOwn(ACTS, l.act) ? l.act : 'identity'));
  // softmax runs per token (and group) on a token layer, else over the whole layer
  const seg = shapes.map((s, l) => (s.tokens > 1 || s.groups ? s.d : sizes[l]));
  return { L, nodes, where, T, b, acts, sizes, shapes, att, seg, ties, bties };
}

const attnBuf = g => ({ S: new Float64Array(g.H * g.n * g.n), A: new Float64Array(g.H * g.n * g.n) });

function alloc(P) {
  return {
    z: P.sizes.map((n, l) => (l ? new Float64Array(n) : null)),
    a: P.sizes.map(n => new Float64Array(n)),
    att: P.att.map(g => g && attnBuf(g)),
  };
}

// Attention scores of every head into S (masked cells -Infinity) and their row softmax into A.
// x is the Q, K, V layer's activations: Q block, then K, then V, each token-major.
function attnScores(g, x, S, A) {
  const { n, d, H, dh, scale, causal } = g, ko = n * d;
  for (let h = 0; h < H; h++) {
    const o = h * n * n, c = h * dh;
    for (let i = 0; i < n; i++) {
      const row = o + i * n;
      let m = -Infinity;
      for (let j = 0; j < n; j++) {
        let s = -Infinity;
        if (!causal || j <= i) {
          s = 0;
          for (let f = 0; f < dh; f++) s += x[i * d + c + f] * x[ko + j * d + c + f];
          s *= scale;
        }
        S[row + j] = s;
        if (s > m) m = s;
      }
      let t = 0;
      for (let j = 0; j < n; j++) { const e = Math.exp(S[row + j] - m); A[row + j] = e; t += e; }
      for (let j = 0; j < n; j++) A[row + j] /= t;
    }
  }
}

// Z = A V per head, heads side by side in each token's row.
function attnForward(g, x, out, st) {
  attnScores(g, x, st.S, st.A);
  const { n, d, H, dh } = g, vo = 2 * n * d, A = st.A;
  for (let h = 0; h < H; h++) {
    const o = h * n * n, c = h * dh;
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < dh; f++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += A[o + i * n + j] * x[vo + j * d + c + f];
        out[i * d + c + f] = s;
      }
    }
  }
}

// Gradients through one attention layer. dZ = dL/dZ; adds dL/dQ, dL/dK, dL/dV into dX (the Q, K,
// V layer's dL/da). st keeps S, A, dA = dZ Vᵀ, dS = A ⊙ (dA - rowsum(dA ⊙ A)) and own = this
// layer's [dQ | dK | dV] with dQ = dS K · scale, dK = dSᵀ Q · scale, dV = Aᵀ dZ.
function attnBackward(g, x, dZ, dX, st) {
  attnScores(g, x, st.S, st.A);
  const { n, d, H, dh, scale } = g, ko = n * d, vo = 2 * n * d;
  const { A, dAt, dS, own } = st;
  for (let h = 0; h < H; h++) {
    const o = h * n * n, c = h * dh;
    for (let i = 0; i < n; i++) {
      const row = o + i * n;
      let r = 0;
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let f = 0; f < dh; f++) s += dZ[i * d + c + f] * x[vo + j * d + c + f];
        dAt[row + j] = s;
        r += s * A[row + j];
      }
      for (let j = 0; j < n; j++) dS[row + j] = A[row + j] * (dAt[row + j] - r);
    }
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < dh; f++) {
        let q = 0, k = 0, v = 0;
        for (let j = 0; j < n; j++) {
          q += dS[o + i * n + j] * x[ko + j * d + c + f];   // (dS K)[i][f]
          k += dS[o + j * n + i] * x[j * d + c + f];        // (dSᵀ Q)[i][f]
          v += A[o + j * n + i] * dZ[j * d + c + f];        // (Aᵀ dZ)[i][f]
        }
        own[i * d + c + f] = q * scale;
        own[ko + i * d + c + f] = k * scale;
        own[vo + i * d + c + f] = v;
      }
    }
  }
  for (let q = 0; q < own.length; q++) dX[q] += own[q];
}

function run(P, x, buf = alloc(P)) {
  const { z, a } = buf;
  if (!P.L) return buf;
  const a0 = a[0], ins = P.nodes[0];
  for (let j = 0; j < a0.length; j++) a0[j] = num(x ? x[j] : ins[j].value, 0);
  for (let l = 1; l < P.L; l++) {
    const zl = z[l], T = P.T[l], g = P.att[l];
    if (g) {
      attnForward(g, a[l - 1], a[l], buf.att[l]);
      zl.set(a[l]);
      continue;
    }
    zl.set(P.b[l]);
    for (let q = 0; q < T.n; q++) zl[T.i[q]] += T.w[q] * a[T.k[q]][T.j[q]];
    activate(P.acts[l], zl, a[l], P.seg[l]);
  }
  return buf;
}

// Source layers feeding layer l, sorted descending; l - 1 always included.
function termKs(P, l) {
  const ks = new Set([l - 1]);
  const T = P.T[l];
  for (let q = 0; q < T.n; q++) ks.add(T.k[q]);
  return [...ks].sort((p, q) => q - p);
}

// ------------------------------------------------------------------ maths

// An attention layer's entry is { l, id, kind: 'attention', act: 'identity', rows, b (all 0), terms: [] };
// every other layer's has kind 'dense'.
export function matrices(net) {
  const P = plan(net);
  const out = [];
  for (let l = 1; l < P.L; l++) {
    const rows = P.nodes[l].map(n => n.id);
    if (P.att[l]) {
      out.push({ l, id: net.layers[l].id, kind: 'attention', act: 'identity', rows, b: rows.map(() => 0), terms: [] });
      continue;
    }
    const terms = termKs(P, l).map(k => {
      const cols = P.nodes[k].map(n => n.id);
      return {
        k, cols,
        W: rows.map(() => cols.map(() => 0)),
        edge: rows.map(() => cols.map(() => null)),
      };
    });
    const byK = new Map(terms.map(t => [t.k, t]));
    const T = P.T[l];
    for (let q = 0; q < T.n; q++) {
      const t = byK.get(T.k[q]);
      t.W[T.i[q]][T.j[q]] = T.w[q];
      t.edge[T.i[q]][T.j[q]] = T.edges[q].id;
    }
    out.push({ l, id: net.layers[l].id, kind: 'dense', act: P.acts[l], rows, b: Array.from(P.b[l]), terms });
  }
  return out;
}

const grid = (rows, cols, f) => Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => f(i, j)));

// fwd.attn[l]: per head the n x dh matrices Q, K, V, Z and the n x n S (masked -Infinity) and A.
function attnReport(g, x, out, st) {
  const { n, d, H, dh } = g, ko = n * d, vo = 2 * n * d;
  const heads = [];
  for (let h = 0; h < H; h++) {
    const c = h * dh, o = h * n * n;
    heads.push({
      Q: grid(n, dh, (i, f) => x[i * d + c + f]),
      K: grid(n, dh, (i, f) => x[ko + i * d + c + f]),
      V: grid(n, dh, (i, f) => x[vo + i * d + c + f]),
      S: grid(n, n, (i, j) => st.S[o + i * n + j]),
      A: grid(n, n, (i, j) => st.A[o + i * n + j]),
      Z: grid(n, dh, (i, f) => out[i * d + c + f]),
    });
  }
  return { heads, tokens: n, dk: dh, scale: g.scale, causal: g.causal };
}

// bwd.attn[l]: per head dZ, dQ, dK, dV (n x dh; this layer's own contribution) and dA, dS (n x n).
function attnGradReport(g, dZ, st) {
  const { n, d, H, dh } = g, ko = n * d, vo = 2 * n * d;
  const heads = [];
  for (let h = 0; h < H; h++) {
    const c = h * dh, o = h * n * n;
    heads.push({
      dZ: grid(n, dh, (i, f) => dZ[i * d + c + f]),
      dA: grid(n, n, (i, j) => st.dAt[o + i * n + j]),
      dS: grid(n, n, (i, j) => st.dS[o + i * n + j]),
      dQ: grid(n, dh, (i, f) => st.own[i * d + c + f]),
      dK: grid(n, dh, (i, f) => st.own[ko + i * d + c + f]),
      dV: grid(n, dh, (i, f) => st.own[vo + i * d + c + f]),
    });
  }
  return { heads };
}

export function forward(net, x) {
  const P = plan(net);
  const buf = run(P, x);
  const Z = buf.z.map(v => (v ? Array.from(v) : null));
  const A = buf.a.map(v => Array.from(v));
  const nodeMap = {};
  P.nodes.forEach((ns, l) => ns.forEach((n, i) => { nodeMap[n.id] = { z: l ? Z[l][i] : null, a: A[l][i] }; }));
  const attn = P.att.map((g, l) => (g ? attnReport(g, buf.a[l - 1], buf.a[l], buf.att[l]) : null));
  return { z: Z, a: A, node: nodeMap, attn };
}

// Loss head on the output layer. Fills dA (dL/da) and, for the fused xent cases, dZ directly.
// seg: softmax segment length (a token layer: xent is the mean over its m segments).
function head(act, z, a, y, loss, dA, dZ, seg) {
  const n = a.length;
  const t = i => (y && isNum(y[i]) ? y[i] : 0);
  if (loss === 'xent' && act === 'softmax') {
    const w = seg > 0 && n % seg === 0 ? seg : n, m = w ? n / w : 1;
    let L = 0;
    for (let s0 = 0; s0 < n; s0 += w) {
      let mx = -Infinity;
      for (let i = s0; i < s0 + w; i++) if (z[i] > mx) mx = z[i];
      let s = 0;
      for (let i = s0; i < s0 + w; i++) s += Math.exp(z[i] - mx);
      const lse = mx + Math.log(s);
      let S = 0;
      for (let i = s0; i < s0 + w; i++) { const yi = t(i); if (yi) L -= yi * (z[i] - lse); S += yi; }
      for (let i = s0; i < s0 + w; i++) {
        const yi = t(i);
        dZ[i] = (a[i] * S - yi) / m;
        dA[i] = -yi / (m * Math.max(a[i], 1e-12));
      }
    }
    return { loss: L / m, fused: true, note: null };
  }
  if (loss === 'xent' && act === 'sigmoid') {
    let L = 0;
    for (let i = 0; i < n; i++) {
      const yi = t(i);
      L += softplus(z[i]) - yi * z[i];
      dZ[i] = (a[i] - yi) / n;
      dA[i] = (a[i] - yi) / (n * Math.max(a[i] * (1 - a[i]), 1e-12));
    }
    return { loss: n ? L / n : 0, fused: true, note: null };
  }
  let L = 0;
  for (let i = 0; i < n; i++) { const r = a[i] - t(i); L += r * r; dA[i] = r / n; }
  return {
    loss: n ? L / (2 * n) : 0, fused: false,
    note: loss === 'xent' ? 'cross-entropy needs a softmax or sigmoid output layer; using mse instead' : null,
  };
}

function actBack(act, z, a, dA, dZ, seg) {
  const n = dZ.length;
  if (act === 'softmax') {
    const w = seg > 0 && n % seg === 0 ? seg : n;
    for (let s0 = 0; s0 < n; s0 += w) {
      let s = 0;
      for (let i = s0; i < s0 + w; i++) s += a[i] * dA[i];
      for (let i = s0; i < s0 + w; i++) dZ[i] = a[i] * (dA[i] - s);
    }
    return;
  }
  const df = (ACTS[act] || ACTS.identity).df;
  for (let i = 0; i < n; i++) dZ[i] = dA[i] * df(z[i], a[i]);
}

function workspace(P) {
  return {
    dA: P.sizes.map(n => new Float64Array(n)),
    dZ: P.sizes.map(n => new Float64Array(n)),
    gw: P.T.map(T => new Float64Array(T.n)),
    att: P.att.map(g => g && {
      ...attnBuf(g), dAt: new Float64Array(g.H * g.n * g.n), dS: new Float64Array(g.H * g.n * g.n),
      own: new Float64Array(3 * g.n * g.d),
    }),
  };
}

// Backward pass for one sample into workspace W (dA, dZ overwritten). gw is overwritten, or
// added to when accumulate is true (trainStep sums over the batch without reallocating).
function backprop(P, z, a, y, loss, W = workspace(P), accumulate = false) {
  const { dA, dZ, gw } = W;
  for (const v of dA) v.fill(0);
  const last = P.L - 1;
  if (P.L < 2) return { loss: 0, note: null, ...W };
  const h = head(P.acts[last], z[last], a[last], y, loss, dA[last], dZ[last], P.seg[last]);
  for (let l = last; l >= 1; l--) {
    const geo = P.att[l];
    if (geo) {   // identity activation, no edges: the gradient flows to Q, K, V
      dZ[l].set(dA[l]);
      attnBackward(geo, a[l - 1], dZ[l], dA[l - 1], W.att[l]);
      continue;
    }
    if (l !== last || !h.fused) actBack(P.acts[l], z[l], a[l], dA[l], dZ[l], P.seg[l]);
    const T = P.T[l], d = dZ[l], g = gw[l], ti = T.i, tk = T.k, tj = T.j, tw = T.w;
    for (let q = 0; q < T.n; q++) {
      const i = ti[q], di = d[i], ak = a[tk[q]], j = tj[q];
      if (accumulate) g[q] += di * ak[j]; else g[q] = di * ak[j];
      dA[tk[q]][j] += tw[q] * di;
    }
  }
  return { loss: h.loss, note: h.note, ...W };
}

function fwdMatches(fwd, P) {
  return fwd && Array.isArray(fwd.a) && Array.isArray(fwd.z) && fwd.a.length === P.L && fwd.z.length === P.L
    && P.sizes.every((n, l) => fwd.a[l]?.length === n && (l === 0 || fwd.z[l]?.length === n));
}

export function backward(net, fwd, y, loss) {
  loss = loss || net.meta?.loss || 'mse';
  const P = plan(net);
  if (!fwdMatches(fwd, P)) fwd = forward(net);
  const r = backprop(P, fwd.z, fwd.a, y, loss);
  const dA = r.dA.map(v => Array.from(v));
  const dZ = r.dZ.map((v, l) => (l ? Array.from(v) : []));
  const db = dZ.map((v, l) => (P.att[l] ? v.map(() => 0) : v.slice()));
  const dW = [[]];
  const edgeMap = {};
  for (let l = 1; l < P.L; l++) {
    if (P.att[l]) { dW.push([]); continue; }   // parallel to matrices(): no terms
    const ks = termKs(P, l), rows = P.sizes[l];
    const mats = ks.map(k => Array.from({ length: rows }, () => new Array(P.sizes[k]).fill(0)));
    const at = new Map(ks.map((k, t) => [k, t]));
    const T = P.T[l];
    for (let q = 0; q < T.n; q++) {
      const g = r.gw[l][q];
      mats[at.get(T.k[q])][T.i[q]][T.j[q]] = g;
      edgeMap[T.edges[q].id] = g;
    }
    dW.push(mats);
  }
  const nodeMap = {};
  P.nodes.forEach((ns, l) => ns.forEach((n, i) => { nodeMap[n.id] = { da: dA[l][i], dz: l ? dZ[l][i] : null }; }));
  const attn = P.att.map((g, l) => (g ? attnGradReport(g, r.dZ[l], r.att[l]) : null));
  // A shared parameter's gradient is the sum of its members' (bwd.edge / bwd.node[].dz stay per member).
  const tie = {};
  for (const [id, list] of P.ties) tie[id] = list.reduce((s, [l, q]) => s + r.gw[l][q], 0);
  for (const [id, list] of P.bties) tie[id] = list.reduce((s, [l, i]) => s + r.dZ[l][i], tie[id] || 0);
  return { loss: r.loss, note: r.note, dA, dZ, dW, db, node: nodeMap, edge: edgeMap, attn, tie };
}

const asRow = v => (Array.isArray(v) || ArrayBuffer.isView(v) ? v : [v]);

// One full-batch gradient-descent step. Returns the mean loss before the update. If the loss or
// any gradient is not finite, nothing is updated (the net stays valid) and the loss is returned.
export function trainStep(net, { X, Y } = {}, { lr = 0.1, loss } = {}) {
  loss = loss || net.meta?.loss || 'mse';
  const P = plan(net);
  const m = Array.isArray(X) ? X.length : 0;
  if (!m || P.L < 2) return 0;
  const W = workspace(P), gw = W.gw;
  const gb = P.sizes.map(n => new Float64Array(n));
  const buf = alloc(P);
  let total = 0;
  for (let s = 0; s < m; s++) {
    run(P, asRow(X[s]), buf);
    total += backprop(P, buf.z, buf.a, asRow(Y?.[s]), loss, W, true).loss;
    for (let l = 1; l < P.L; l++) {
      const b = gb[l], d = W.dZ[l];
      for (let i = 0; i < b.length; i++) b[i] += d[i];
    }
  }
  const mean = total / m;
  if (!Number.isFinite(mean)) return mean;
  const ok = arrs => arrs.every(v => !v || v.every(Number.isFinite));
  if (!ok(gw) || !ok(gb)) return mean;
  const step = num(lr, 0.1) / m;
  for (let l = 1; l < P.L; l++) {
    const T = P.T[l];
    for (let q = 0; q < T.n; q++) if (!T.fixed[q] && !T.tied[q]) T.edges[q].w -= step * gw[l][q];
    if (P.att[l]) continue;   // no biases
    P.nodes[l].forEach((n, i) => { if (!isTie(n.tie)) n.bias -= step * gb[l][i]; });
  }
  // A shared parameter steps by the summed gradient of its members, and they stay equal.
  for (const list of P.ties.values()) {
    let g = 0;
    for (const [l, q] of list) g += gw[l][q];
    const [l0, q0] = list[0], w = P.T[l0].edges[q0].w - step * g;
    for (const [l, q] of list) P.T[l].edges[q].w = w;
  }
  for (const list of P.bties.values()) {
    let g = 0;
    for (const [l, i] of list) g += gb[l][i];
    const [l0, i0] = list[0], b = P.nodes[l0][i0].bias - step * g;
    for (const [l, i] of list) P.nodes[l][i].bias = b;
  }
  return mean;
}

// Outputs for every row of X (fast path for heatmaps). { layer } picks another layer's
// activations (index or id), or 'all' for every layer per sample.
export function predict(net, X, { layer } = {}) {
  const P = plan(net);
  if (!P.L || !Array.isArray(X)) return [];
  const want = layer === undefined || layer === null ? P.L - 1 : layer === 'all' ? 'all' : layerRef(net, layer);
  const buf = alloc(P);
  return X.map(x => {
    run(P, asRow(x), buf);
    if (want === 'all') return buf.a.map(v => Array.from(v));
    return want >= 0 ? Array.from(buf.a[want]) : [];
  });
}

// If every non-input layer is identity, the whole net is one affine map y = W x + b.
export function collapse(net) {
  if (net.layers.length < 2 || net.layers.slice(1).some(l => l.act !== 'identity')) return null;
  const P = plan(net);
  if (P.att.some(Boolean)) return null;   // attention multiplies activations: not affine
  const n0 = P.sizes[0];
  const A = [Array.from({ length: n0 }, (_, i) => Array.from({ length: n0 }, (_, j) => (i === j ? 1 : 0)))];
  const c = [new Array(n0).fill(0)];
  for (let l = 1; l < P.L; l++) {
    const Al = Array.from({ length: P.sizes[l] }, () => new Array(n0).fill(0));
    const cl = Array.from(P.b[l]);
    const T = P.T[l];
    for (let q = 0; q < T.n; q++) {
      const i = T.i[q], src = A[T.k[q]][T.j[q]], w = T.w[q];
      for (let j = 0; j < n0; j++) Al[i][j] += w * src[j];
      cl[i] += w * c[T.k[q]][T.j[q]];
    }
    A.push(Al); c.push(cl);
  }
  const last = P.L - 1;
  return { W: A[last], b: c[last], rows: P.nodes[last].map(n => n.id), cols: P.nodes[0].map(n => n.id) };
}

// ------------------------------------------------------------------ datasets

function sample(n, seed, noise, kind, fn) {
  const r = rng(seed ?? 1);
  const count = clampInt(n, 0, 1e6, 200);
  const s = num(noise, 0);
  const X = [], Y = [];
  for (let i = 0; i < count; i++) {
    const [x, y] = fn(r, i);
    if (s) {
      if (kind === 'class') for (let j = 0; j < x.length; j++) x[j] += s * gauss(r);
      else for (let j = 0; j < y.length; j++) y[j] += s * gauss(r);
    }
    X.push(x); Y.push(y);
  }
  return { X, Y };
}

function dataset(label, inputs, outputs, kind, fn) {
  return { label, inputs, outputs, kind, make: (n = 200, seed = 1, noise = 0) => sample(n, seed, noise, kind, fn) };
}

// A sequence task: { label, inputs = tokens · d_in, outputs = tokens · d_out, kind: 'seq', tokens, make }.
// fn(r, noise) -> [x, y], both flat and token-major.
function seqData(label, tokens, dIn, dOut, fn) {
  return {
    label, inputs: tokens * dIn, outputs: tokens * dOut, kind: 'seq', tokens,
    make: (n = 200, seed = 1, noise = 0) => {
      const r = rng(seed ?? 1), count = clampInt(n, 0, 1e6, 200), s = num(noise, 0);
      const X = [], Y = [];
      for (let i = 0; i < count; i++) { const [x, y] = fn(r, s); X.push(x); Y.push(y); }
      return { X, Y };
    },
  };
}

// n tokens (x₁, x₂): x₁ uniform on [-0.9, 0.9], redrawn until every two are at least 0.3 apart
// (so the largest is clear), x₂ uniform on [-1, 1]. Then Gaussian noise s on both.
function ladder(r, n, s) {
  let a;
  do a = Array.from({ length: n }, () => 1.8 * r() - 0.9);
  while (a.some((u, i) => a.some((v, j) => j > i && Math.abs(u - v) < 0.3)));
  return a.map(u => {
    const t = [u, 2 * r() - 1];
    if (s) { t[0] += s * gauss(r); t[1] += s * gauss(r); }
    return t;
  });
}
// The token with the largest first feature.
const argmax = T => T.reduce((best, t, j) => (t[0] > T[best][0] ? j : best), 0);
// Positions of 3 tokens on the unit circle, 120° apart: (cos θ_j, sin θ_j) with θ_j = 2πj/3.
const POS3 = [[1, 0], [-0.5, Math.sqrt(3) / 2], [-0.5, -Math.sqrt(3) / 2]];

// Orthonormal axes of the 'cloud' dataset, widest spread first.
const CLOUD_AXES =[[2, 1, 2], [1, 2, -2], [2, -2, -1]].map(v => v.map(c => c / 3));

export const DATASETS = {
  xor: dataset('XOR', 2, 1, 'class', (r, i) => {
    const sx = i & 1 ? 1 : -1, sy = i & 2 ? 1 : -1;
    const x = sx * (0.1 + 0.9 * r()), y = sy * (0.1 + 0.9 * r());
    return [[x, y], [sx !== sy ? 1 : 0]];
  }),
  circles: dataset('Circles', 2, 1, 'class', (r, i) => {
    const inner = i % 2 === 0;
    const t = 2 * Math.PI * r();
    const rad = inner ? 0.5 * Math.sqrt(r()) : 0.7 + 0.3 * r();
    return [[rad * Math.cos(t), rad * Math.sin(t)], [inner ? 1 : 0]];
  }),
  spiral: dataset('Spiral', 2, 1, 'class', (r, i) => {
    const c = i % 2;
    const t = 0.08 + 0.92 * r();
    const th = 3 * Math.PI * t + c * Math.PI;
    return [[t * Math.cos(th), t * Math.sin(th)], [c]];
  }),
  blobs: dataset('Two blobs', 2, 1, 'class', (r, i) => {
    const c = i % 2, cx = c ? 0.45 : -0.45, cy = c ? 0.35 : -0.35;
    return [[cx + 0.2 * gauss(r), cy + 0.2 * gauss(r)], [c]];
  }),
  moons: dataset('Moons', 2, 1, 'class', (r, i) => {
    const c = i % 2, t = Math.PI * r();
    const x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t);
    return [[(x - 0.5) / 1.5, (y - 0.25) / 1.5], [c]];
  }),
  three: dataset('Three classes', 2, 3, 'class', (r, i) => {
    const c = i % 3, ang = Math.PI / 2 + (c * 2 * Math.PI) / 3;
    return [[0.55 * Math.cos(ang) + 0.18 * gauss(r), 0.55 * Math.sin(ang) + 0.18 * gauss(r)], [0, 1, 2].map(k => (k === c ? 1 : 0))];
  }),
  line: dataset('Line', 1, 1, 'reg', r => {
    const x = 2 * r() - 1;
    return [[x], [0.7 * x + 0.2]];
  }),
  sine: dataset('Sine', 1, 1, 'reg', r => {
    const x = 2 * r() - 1;
    return [[x], [0.8 * Math.sin(Math.PI * x)]];
  }),
  // A tilted pancake in 3-D, the target being the point itself: spread 0.55 and 0.3 along
  // CLOUD_AXES[0] and [1] (its plane), 0.04 along [2]. For the linear autoencoder (PCA) preset.
  cloud: dataset('Flat cloud (3-D)', 3, 3, 'reg', r => {
    const g = () => Math.max(-2.5, Math.min(2.5, gauss(r)));
    const s = [0.55 * g(), 0.3 * g(), 0.04 * g()];
    const x = [0, 1, 2].map(j => s[0] * CLOUD_AXES[0][j] + s[1] * CLOUD_AXES[1][j] + s[2] * CLOUD_AXES[2][j]);
    return [x, x.slice()];
  }),

  // Sequence tasks: X and Y rows are token-major (tokens x d_in, tokens x d_out); noise goes on
  // the token features and the targets follow the noisy tokens.
  seq_max: seqData('Copy the max token (3 tokens × 2)', 3, 2, 2, (r, s) => {
    const T = ladder(r, 3, s), top = T[argmax(T)];
    return [T.flat(), T.flatMap(() => top)];
  }),
  seq_minmax: seqData('Max and min of x₁ (3 tokens × 2)', 3, 2, 2, (r, s) => {
    const T = ladder(r, 3, s), a = T.map(t => t[0]);
    return [T.flat(), T.flatMap(() => [Math.max(...a), Math.min(...a)])];
  }),
  seq_prev: seqData('Previous token, causal (3 tokens, c + position)', 3, 3, 1, (r, s) => {
    const c = [0, 1, 2].map(() => 2 * r() - 1 + (s ? s * gauss(r) : 0));
    return [c.flatMap((v, j) => [v, ...POS3[j]]), c.map((_, i) => c[Math.max(0, i - 1)])];
  }),
  seq_addmax: seqData('ReLU(own + max token) (2 tokens × 2)', 2, 2, 2, (r, s) => {
    const T = ladder(r, 2, s), top = T[argmax(T)];
    return [T.flat(), T.flatMap(t => t.map((v, f) => Math.max(0, v + top[f])))];
  }),
};

// ------------------------------------------------------------------ presets

// A densely connected stack: specs = [{ name, act, size }], layer 0 first.
function stack(title, loss, specs, { seed = 1, scheme = 'xavier', inputs = [], targets = null } = {}) {
  const net = { v: 1, layers: [], nodes: [], edges: [], meta: { title, loss, nextId: 1, train: {} } };
  for (const s of specs) {
    const id = uid(net, 'L');
    net.layers.push({ id, name: s.name, act: s.act });
    for (let j = 0; j < s.size; j++) net.nodes.push(blankNode(net, id));
  }
  for (let l = 1; l < specs.length; l++) connectDense(net, l - 1, l, { w: 0 });
  randomize(net, { seed, scheme, biases: 'small' });
  nodesIn(net, 0).forEach((n, j) => { n.value = inputs[j] ?? 0.5; });
  if (targets) nodesIn(net, specs.length - 1).forEach((n, j) => { n.target = targets[j] ?? 0; });
  relabel(net);
  return net;
}
// Evenly spaced columns in a width x height box. The first eight presets use 900 x 560; columns of
// 8 or more get a taller box, so rows stay >= 76 px apart (a neuron's value sits under its circle).
const layout = (net, width = 900, height = 560) => autoLayout(net, { width, height });
const IN = size => ({ name: 'Input', act: 'identity', size });
const tex = words => words.map(w => `\\text{${w}}`);
const r3 = v => Math.round(v * 1000) / 1000 + 0;   // 3 decimals; + 0 turns -0 into 0

// ---- hand-wired nets: layers first, then the edges one layer pair at a time

// Layers and nodes without edges. specs = [{ name, act, size, labels? }], layer 0 first.
function blank(title, loss, specs) {
  const net = { v: 1, layers: [], nodes: [], edges: [], meta: { title, loss, nextId: 1, train: {} } };
  for (const s of specs) {
    const id = uid(net, 'L');
    net.layers.push({ id, name: s.name, act: s.act });
    for (let j = 0; j < s.size; j++) {
      const n = blankNode(net, id);
      if (s.labels) n.label = s.labels[j];
      net.nodes.push(n);
    }
  }
  return net;
}

// Edges from layer a into layer b. M(i, j) is the matrix entry in row i (node i of b) and
// column j (node j of a): a weight, or null for no edge (a masked entry). M may be a nested array.
// extra(i, j): optional fields for that edge ({ tie } or { fixed: true }).
function wire(net, a, b, M, extra) {
  const src = nodesIn(net, a), dst = nodesIn(net, b);
  const at = typeof M === 'function' ? M : (i, j) => M[i]?.[j];
  dst.forEach((t, i) => src.forEach((s, j) => {
    const w = at(i, j);
    if (isNum(w)) addEdge(net, s.id, t.id, w, extra?.(i, j));
  }));
}
// A conv kernel's taps as ties: the entry at offset o = j - stride i is tap o + 1 of kernel `name`.
const kernelTie = (name, stride = 1) => (i, j) => ({ tie: `${name}:1,${j - stride * i + 1}` });
const FIXED = () => ({ fixed: true });

// Seeded Xavier (or He) weights on the entries keep(i, j) allows. fanIn defaults to the kept
// entries in the row; pass the layer's total when other terms feed it too.
function wireRandom(net, a, b, r, { scheme = 'xavier', keep = () => true, fanIn } = {}) {
  const rows = nodesIn(net, b).length, cols = nodesIn(net, a).length;
  const count = (n, f) => Array.from({ length: n }, (_, k) => f(k)).filter(Boolean).length;
  const rowN = Array.from({ length: rows }, (_, i) => count(cols, j => keep(i, j)));
  const colN = Array.from({ length: cols }, (_, j) => count(rows, i => keep(i, j)));
  wire(net, a, b, (i, j) => (keep(i, j) ? sampleW(r, scheme, fanIn ?? rowN[i], colN[j]) : null));
}

// b: an array, a function of the row, or one value for every node of layer l.
function setBias(net, l, b) {
  nodesIn(net, l).forEach((n, i) => { n.bias = typeof b === 'function' ? b(i) : Array.isArray(b) ? b[i] : b; });
}
// Small seeded biases on every non-input node, as stack() gives its nets.
function smallBiases(net, r) {
  for (let l = 1; l < net.layers.length; l++) setBias(net, l, () => (2 * r() - 1) * 0.1);
}

// Input values and output targets, then default labels. targets 'self': the net's own output,
// for demos that compute something exactly (the loss is 0: the weights already do the job).
function io(net, inputs, targets) {
  nodesIn(net, 0).forEach((n, j) => { n.value = inputs[j]; });
  const L = net.layers.length;
  const y = targets === 'self' ? forward(net).a[L - 1] : targets;
  nodesIn(net, L - 1).forEach((n, j) => { n.target = y[j] + 0; });
  relabel(net);
  return net;
}

// Put node i of layer l at the mean height of nodes at(i) of layer k (its receptive field).
function alignTo(net, l, k, at) {
  const src = nodesIn(net, k);
  nodesIn(net, l).forEach((n, i) => {
    const ys = at(i).map(j => src[j].y);
    n.y = Math.round((ys.reduce((s, y) => s + y, 0) / ys.length) * 10) / 10;
  });
  return net;
}
// Pull the groups of layer l apart (group(i) = 0, 1, ...): each moves gap px from the next.
function spread(net, l, group, gap) {
  const ns = nodesIn(net, l), G = Math.max(...ns.map((_, i) => group(i))) + 1;
  ns.forEach((n, i) => { n.y += (group(i) - (G - 1) / 2) * gap; });
  return net;
}

// ---- token nets: layers of tokens x d features, tokenwise tied matrices, fixed residual edges

// specs = [{ name, act, tokens, d, groups?, attention?: { heads, causal }, label(t, f, group) }] (t, f 1-based).
function seqBlank(title, loss, specs) {
  const net = { v: 1, layers: [], nodes: [], edges: [], meta: { title, loss, nextId: 1, train: {} } };
  for (const s of specs) {
    const id = uid(net, 'L');
    const layer = { id, name: s.name, act: s.act || 'identity', tokens: s.tokens };
    if (s.groups) layer.groups = s.groups.slice();
    if (s.attention) Object.assign(layer, { kind: 'attention', heads: s.attention.heads ?? 1, causal: !!s.attention.causal });
    net.layers.push(layer);
    for (const g of s.groups || [null]) {
      for (let t = 1; t <= s.tokens; t++) {
        for (let f = 1; f <= s.d; f++) {
          const n = blankNode(net, id);
          n.label = s.label(t, f, g);
          net.nodes.push(n);
        }
      }
    }
  }
  return net;
}

// The nodes of one group of layer l as [token][feature].
function tokenGrid(net, l, group = null) {
  const { tokens, d, groups } = tokenShape(net, l), ns = nodesIn(net, l);
  const o = group && groups ? groups.indexOf(group) * tokens * d : 0;
  return Array.from({ length: tokens }, (_, t) => ns.slice(o + t * d, o + (t + 1) * d));
}

// Tokenwise tied edges from group `from` of layer a to group `to` of layer b: in every token,
// feature i feeds feature j with the shared weight W[i][j] (X W convention), tie '<name>:<i>,<j>'.
function wireTied(net, a, b, name, W, { from = null, to = null } = {}) {
  const src = tokenGrid(net, a, from), dst = tokenGrid(net, b, to);
  dst.forEach((row, t) => row.forEach((q, j) => src[t].forEach((p, i) => {
    if (isNum(W[i]?.[j])) addEdge(net, p.id, q.id, W[i][j], { tie: `${name}:${i + 1},${j + 1}` });
  })));
}
// A fixed residual: token t's feature f of layer a adds into the same slot of layer b.
function wireResidual(net, a, b) {
  const src = tokenGrid(net, a);
  tokenGrid(net, b).forEach((row, t) => row.forEach((q, f) => addEdge(net, src[t][f].id, q.id, 1, { fixed: true })));
}
// One shared bias per feature of a group of layer l: node.tie '<name>:<f>'.
function tieBias(net, l, name, b, group = null) {
  tokenGrid(net, l, group).forEach(row => row.forEach((n, f) => { n.bias = b[f]; n.tie = `${name}:${f + 1}`; }));
}
// A shared bias for every node of a plain layer (a conv's one bias per kernel).
function tieBiasAll(net, l, name) {
  const ns = nodesIn(net, l);
  ns.forEach(n => { n.bias = ns[0].bias; n.tie = `${name}:1`; });
}
const randW = (r, rows, cols, scheme = 'xavier') =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => sampleW(r, scheme, rows, cols)));
const randB = (r, d) => Array.from({ length: d }, () => (2 * r() - 1) * 0.1);

// X (layer x, d_in features) -> Q, K, V (layer qkv, d each): tied W_Q, W_K, W_V and b_Q, b_K, b_V.
function wireQKV(net, x, qkv, r) {
  const dIn = tokenShape(net, x).d, d = tokenShape(net, qkv).d;
  for (const g of QKV) {
    wireTied(net, x, qkv, `W_${g}`, randW(r, dIn, d), { to: g });
    tieBias(net, qkv, `b_${g}`, randB(r, d), g);
  }
}
// A 3-token attention net laid out like the matrix panel: X and the Q, K, V layer as tokens × d
// grids (a row per token, a column per feature, the groups stacked), so Q, K, V is 9 rows high
// instead of an 18-neuron column and the net fits the stage at a readable zoom. X sits halfway
// between the Q and K blocks, so every edge into Q, K and V slants: none runs level through the
// next neuron of a row. Later layers are columns (a token's features stacked) at the same height.
function gridLayout(net) {
  const FX = 88, RP = 90, GG = 60, GAP = 170, TOP = 120;
  const q = tokenShape(net, 1), block = q.tokens * RP + GG;
  const mid = TOP + (block * (q.groups.length - 1) + (q.tokens - 1) * RP) / 2;
  let x = 110;
  net.layers.forEach((_, l) => {
    const { tokens, d } = tokenShape(net, l), ns = nodesIn(net, l);
    if (l < 2) {
      ns.forEach((n, k) => {
        const g = Math.floor(k / (tokens * d)), t = Math.floor(k / d) % tokens;
        n.x = x + (k % d) * FX;
        n.y = (l ? TOP + g * block : TOP + block / 2) + t * RP;
      });
      x += (d - 1) * FX + GAP;
      return;
    }
    const off = ns.map((_, k) => k * ROW + Math.floor(k / d) * ROW / 4), span = off.at(-1) ?? 0;
    ns.forEach((n, k) => { n.x = x; n.y = Math.round((mid - span / 2 + off[k]) * 10) / 10; });
    x += GAP;
  });
  return net;
}
// Input values and targets from sample 0 of a dataset (seed 1, no noise).
function seqIO(net, key) {
  const { X, Y } = DATASETS[key].make(1, 1, 0);
  return io(net, X[0].map(r3), Y[0].map(r3));
}
const tokLabel = sym => (t, f) => `${sym}_{${t},${f}}`;
const qkvLabel = (t, f, g) => `${g.toLowerCase()}_{${t},${f}}`;

const eye = (i, j) => (i === j ? 1 : null);                        // identity shortcut, rest masked
const band = (k, stride = 1) => (i, j) => {                         // convolution: kernel k per row
  const o = j - stride * i;
  return o >= 0 && o < k.length ? k[o] : null;
};
const pairs = w => (i, j) => (Math.floor(j / 2) === i ? w : null);  // pool size 2
// The graph convolution preset's graph: a triangle 1-2-3 with a tail 3-4-5 (no self-loops here).
const GRAPH = [[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]];

// A preset records the dataset that suits it in meta.train.dataset (the Train panel's setting),
// so the net carries it through saves and exports. Presets without a dataset leave meta.train empty.
// group: its section of the New net menu. note: one line on what to notice (matrix panel first).
// lr: a learning rate that suits the preset (one the Train panel lists), recorded as meta.train.lr.
function preset({ group, label, dataset = null, note, lr = null }, make) {
  return {
    label, group, note, dataset, lr,
    build: (seed = 1) => {
      const net = make(seed);
      if (dataset) net.meta.train = { ...net.meta.train, dataset };
      if (lr) net.meta.train = { ...net.meta.train, lr };
      return net;
    },
  };
}

const BASICS = 'Basics', MLPS = 'MLPs', SKIPS = 'Skip connections', STRUCT = 'Structure in W';
const SEQ = 'Sequences', ATT = 'Attention', EMB = 'Embeddings & autoencoders', DEMO = 'Teaching demos';

// Menu order: by group, then as listed.
export const PRESETS = {
  // ---------------------------------------------------------------- basics
  gates: preset({
    group: BASICS, label: 'Logic gates: AND, OR, NAND (hand-set)',
    note: 'Each row of W is one gate. Weights (20, 20) with bias −30 make AND, with −10 OR; flipping every sign gives NAND. The bias sets how many inputs must be on.',
  }, () => {
    const net = blank('Logic gates', 'xent', [IN(2),
      { name: 'Gates', act: 'sigmoid', size: 3, labels: tex(['AND', 'OR', 'NAND']) }]);
    wire(net, 0, 1, [[20, 20], [20, 20], [-20, -20]]);
    setBias(net, 1, [-30, -10, 30]);
    return layout(io(net, [1, 0], [0, 1, 1]));
  }),
  xor_gates: preset({
    group: BASICS, label: 'XOR by hand: OR + NAND, then AND',
    note: 'Row 1 of W⁽¹⁾ is OR and row 2 is NAND; W⁽²⁾ ANDs them. That is XOR with no training: try all four 0/1 inputs.',
  }, () => {
    const net = blank('XOR by hand', 'xent', [IN(2),
      { name: 'OR, NAND', act: 'sigmoid', size: 2, labels: tex(['OR', 'NAND']) },
      { name: 'AND', act: 'sigmoid', size: 1, labels: tex(['XOR']) }]);
    wire(net, 0, 1, [[20, 20], [-20, -20]]);
    setBias(net, 1, [-10, 30]);
    wire(net, 1, 2, [[20, 20]]);
    setBias(net, 2, [-30]);
    return layout(io(net, [1, 0], [1]));
  }),
  xor_relu: preset({
    group: BASICS, label: 'XOR by hand: two ReLUs (exact)',
    note: 'Both rows of W⁽¹⁾ are (1, 1); only the biases differ (0, −1). ReLU sends (0,1) and (1,0) to the same hidden point, so the linear output h₁ − 2h₂ is exactly XOR.',
  }, () => {
    const net = blank('XOR with ReLUs', 'mse', [IN(2), { name: 'Hidden', act: 'relu', size: 2 },
      { name: 'Output', act: 'identity', size: 1 }]);
    wire(net, 0, 1, [[1, 1], [1, 1]]);
    setBias(net, 1, [0, -1]);
    wire(net, 1, 2, [[1, -2]]);
    return layout(io(net, [1, 1], [0]));
  }),
  xor: preset({
    group: BASICS, label: 'XOR (2-4-1)', dataset: 'xor',
    note: 'No single line separates XOR, so W⁽¹⁾ first maps the plane to 4 tanh features, where one row of W⁽²⁾ is enough. Train it and watch the boundary bend.',
  }, seed => layout(stack('XOR', 'xent',
    [IN(2), { name: 'Hidden', act: 'tanh', size: 4 }, { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, inputs: [0.9, -0.7], targets: [1] }))),
  perceptron: preset({
    group: BASICS, label: 'Perceptron', dataset: 'blobs',
    note: 'One neuron is one row of W: σ(w·x + b). The boundary is the line w·x + b = 0, with w as its normal. The same model as logistic regression.',
  }, seed => layout(stack('Perceptron', 'xent',
    [IN(2), { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, inputs: [0.8, -0.5], targets: [1] }))),
  logreg: preset({
    group: BASICS, label: 'Logistic regression (from w = 0)', dataset: 'blobs',
    note: 'p = σ(w·x + b), starting at w = 0 (p = ½ everywhere). The loss is convex, so zero is a fine start: train and watch w turn to point from blob 0 to blob 1.',
  }, () => {
    const net = blank('Logistic regression', 'xent', [IN(2), { name: 'Output', act: 'sigmoid', size: 1 }]);
    wire(net, 0, 1, [[0, 0]]);
    return layout(io(net, [0.45, 0.35], [1]));
  }),
  linreg: preset({
    group: BASICS, label: 'Linear regression (1 → 1)', dataset: 'line',
    note: 'W is 1×1: ŷ = w x + b. From w = b = 0, gradient descent walks to the least-squares line, slope 0.7 and intercept 0.2.',
  }, () => {
    const net = blank('Linear regression', 'mse', [IN(1), { name: 'Output', act: 'identity', size: 1 }]);
    wire(net, 0, 1, [[0]]);
    return layout(io(net, [0.5], [0.55]));
  }),
  softmax_reg: preset({
    group: BASICS, label: 'Softmax regression (2 → 3)', dataset: 'three',
    note: 'W is 3×2, one row per class: z_k = w_k·x + b_k, and softmax turns the scores into probabilities. The class boundaries are where two rows tie. Starts at W = 0.',
  }, () => {
    const net = blank('Softmax regression', 'xent', [IN(2), { name: 'Scores', act: 'softmax', size: 3 }]);
    wire(net, 0, 1, () => 0);
    return layout(io(net, [0, 0.55], [1, 0, 0]));
  }),
  linear: preset({
    group: BASICS, label: 'Linear (collapses to one matrix)',
    note: 'No activations, so the layers collapse: W⁽²⁾W⁽¹⁾ is a single 2×2 matrix (Collapse in the matrix panel). Depth adds nothing without a nonlinearity.',
  }, seed => layout(stack('Linear net', 'mse',
    [IN(2), { name: 'Hidden', act: 'identity', size: 3 }, { name: 'Output', act: 'identity', size: 2 }],
    { seed, inputs: [1, 0.5], targets: [0.5, -0.5] }))),

  // ---------------------------------------------------------------- MLPs
  mlp: preset({
    group: MLPS, label: 'MLP (2-6-4-1, ReLU)', dataset: 'circles',
    note: 'W⁽¹⁾ is 6×2, W⁽²⁾ 4×6, W⁽³⁾ 1×4. Each first-layer ReLU creases the plane along a line; together the creases can enclose the inner circle.',
  }, seed => layout(stack('MLP', 'xent',
    [IN(2), { name: 'Hidden 1', act: 'relu', size: 6 }, { name: 'Hidden 2', act: 'relu', size: 4 },
      { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, scheme: 'he', inputs: [0.3, -0.4], targets: [1] }))),
  deep: preset({
    group: MLPS, label: 'Deep (4 hidden layers)', dataset: 'spiral',
    note: 'Four tanh layers of 6, so the middle W\'s are 6×6. Each layer warps what the one before produced; the spiral takes many warps to untangle.',
  }, seed => layout(stack('Deep net', 'xent',
    [IN(2), ...[1, 2, 3, 4].map(i => ({ name: `Hidden ${i}`, act: 'tanh', size: 6 })),
      { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, inputs: [0.5, 0.3], targets: [1] }))),
  classifier: preset({
    group: MLPS, label: 'Classifier (2-4-3 softmax)', dataset: 'three',
    note: 'The output W is 3×4, one row per class, and softmax turns the three scores into a distribution. With cross-entropy the output δ is just ŷ − y.',
  }, seed => layout(stack('Classifier', 'xent',
    [IN(2), { name: 'Hidden', act: 'tanh', size: 4 }, { name: 'Output', act: 'softmax', size: 3 }],
    { seed, inputs: [0.6, -0.4], targets: [1, 0, 0] }))),
  wide: preset({
    group: MLPS, label: 'Wide & shallow (2-12-1)', dataset: 'circles',
    note: 'One hidden layer: W⁽¹⁾ is 12×2 and W⁽²⁾ 1×12, 49 parameters in all. Narrow & deep has exactly as many; train both on Circles and compare.',
  }, seed => layout(stack('Wide & shallow', 'xent',
    [IN(2), { name: 'Hidden', act: 'tanh', size: 12 }, { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, inputs: [0.2, -0.3], targets: [1] }), 700, 940)),
  narrow_deep: preset({
    group: MLPS, label: 'Narrow & deep (2-3-3-3-3-1)', dataset: 'circles',
    note: 'Four hidden layers of 3: W is 3×2, then three 3×3, then 1×3, 49 parameters like Wide & shallow. Depth composes functions instead of adding more of them.',
  }, seed => layout(stack('Narrow & deep', 'xent',
    [IN(2), ...[1, 2, 3, 4].map(i => ({ name: `Hidden ${i}`, act: 'tanh', size: 3 })),
      { name: 'Output', act: 'sigmoid', size: 1 }],
    { seed, inputs: [0.2, -0.3], targets: [1] }))),
  funnel: preset({
    group: MLPS, label: 'Funnel classifier (2-8-6-4-3)', dataset: 'three',
    note: 'W shrinks down the chain, 8×2, 6×8, 4×6, 3×4: each layer maps into a smaller space, ending in the 3 class scores.',
  }, seed => layout(stack('Funnel classifier', 'xent',
    [IN(2), { name: 'Hidden 1', act: 'relu', size: 8 }, { name: 'Hidden 2', act: 'relu', size: 6 },
      { name: 'Hidden 3', act: 'relu', size: 4 }, { name: 'Output', act: 'softmax', size: 3 }],
    { seed, scheme: 'he', inputs: [0, 0.55], targets: [1, 0, 0] }), 900, 640)),
  uat: preset({
    group: MLPS, label: 'Universal approximation (1-10-1, hand-built)', dataset: 'sine',
    note: 'Built by hand: unit i is ReLU(x − kᵢ) with knots kᵢ = −1, −0.8, …, 0.8 (W⁽¹⁾ all 1s, b⁽¹⁾ = −k). W⁽²⁾ holds the slope changes, so ŷ joins the sine\'s values at the knots.',
  }, () => {
    const K = 10, f = x => 0.8 * Math.sin(Math.PI * x);
    const knot = j => -1 + (2 * j) / K;
    const slope = j => (f(knot(j + 1)) - f(knot(j))) / (knot(j + 1) - knot(j));
    const net = blank('Universal approximation', 'mse', [IN(1),
      { name: 'Hinges', act: 'relu', size: K }, { name: 'Output', act: 'identity', size: 1 }]);
    wire(net, 0, 1, () => 1);
    setBias(net, 1, i => r3(-knot(i)));
    wire(net, 1, 2, (i, j) => r3(j ? slope(j) - slope(j - 1) : slope(0)));
    setBias(net, 2, r3(f(-1)));
    return layout(io(net, [0.5], [0.8]), 700, 790);
  }),

  // ---------------------------------------------------------------- skip connections
  residual: preset({
    group: SKIPS, label: 'Residual block (skip edges)', dataset: 'moons',
    note: 'The x + F(x) layer has two terms: W⁽²⁾a⁽¹⁾ plus the identity shortcut from x (off-diagonal masked). The gradient reaches x through that identity unchanged.',
  }, seed => {
    const net = stack('Residual block', 'xent',
      [IN(2), { name: 'F(x)', act: 'relu', size: 3 }, { name: 'x + F(x)', act: 'identity', size: 2 },
        { name: 'Output', act: 'sigmoid', size: 1 }],
      { seed, scheme: 'he', inputs: [0.6, -0.3], targets: [1] });
    const xs = nodesIn(net, 0), sum = nodesIn(net, 2);
    xs.forEach((x, i) => connect(net, x.id, sum[i].id, 1));   // identity shortcut: masked off-diagonal
    return layout(net);
  }),
  bottleneck: preset({
    group: SKIPS, label: 'ResNet bottleneck block', dataset: 'moons',
    note: 'Reduce 4 → 2, transform 2 → 2, expand 2 → 4, then add the stem a⁽¹⁾ through an identity term before the ReLU: a⁽⁴⁾ = ReLU(W⁽⁴⁾a⁽³⁾ + a⁽¹⁾ + b).',
  }, seed => {
    const r = rng(seed);
    const net = blank('ResNet bottleneck', 'xent', [IN(2), { name: 'Stem', act: 'relu', size: 4 },
      { name: 'Reduce', act: 'relu', size: 2 }, { name: 'Transform', act: 'relu', size: 2 },
      { name: 'Expand', act: 'relu', size: 4 }, { name: 'Output', act: 'sigmoid', size: 1 }]);
    for (let l = 1; l <= 4; l++) wireRandom(net, l - 1, l, r, { scheme: 'he' });
    wireRandom(net, 4, 5, r);
    wire(net, 1, 4, eye);
    smallBiases(net, r);
    return layout(io(net, [0.6, -0.3], [1]), 1000);
  }),
  ffn: preset({
    group: SKIPS, label: 'Transformer FFN block (d → 4d → d)', dataset: 'circles',
    note: 'W⁽¹⁾ is 8×2 (d → 4d) and W⁽²⁾ 2×8 (4d → d); layer 2 also adds x through an identity term, so it holds x + FFN(x), the residual stream. A sigmoid head reads it.',
  }, seed => {
    const r = rng(seed);
    const net = blank('Transformer FFN block', 'xent', [IN(2), { name: 'Up to 4d', act: 'relu', size: 8 },
      { name: 'x + FFN(x)', act: 'identity', size: 2 }, { name: 'Head', act: 'sigmoid', size: 1 }]);
    wireRandom(net, 0, 1, r, { scheme: 'he' });
    wireRandom(net, 1, 2, r);
    wire(net, 0, 2, eye);
    wireRandom(net, 2, 3, r);
    smallBiases(net, r);
    return layout(io(net, [0.2, -0.3], [1]), 900, 640);
  }),
  densenet: preset({
    group: SKIPS, label: 'DenseNet-style (every layer to every later one)', dataset: 'moons',
    note: 'Every layer has a term from every earlier layer: z⁽³⁾ = W⁽³⁾a⁽²⁾ + W a⁽¹⁾ + W x + b. That is DenseNet\'s concatenation [x; a⁽¹⁾; a⁽²⁾] times one wide matrix, split into blocks.',
  }, seed => {
    const r = rng(seed);
    const net = blank('DenseNet-style', 'xent', [IN(2),
      ...[1, 2].map(i => ({ name: `Layer ${i}`, act: 'relu', size: 3 })), { name: 'Output', act: 'sigmoid', size: 1 }]);
    for (let b = 1; b <= 3; b++) {
      for (let a = b - 1; a >= 0; a--) wireRandom(net, a, b, r, { scheme: b < 3 ? 'he' : 'xavier', fanIn: 2 + 3 * (b - 1) });
    }
    smallBiases(net, r);
    return layout(io(net, [0.6, -0.3], [1]));
  }),
  unet: preset({
    group: SKIPS, label: 'U-Net-style encoder-decoder skips',
    note: 'Mirror-image skips, encoder to decoder and input to output, each a masked identity, so detail can bypass the 2-unit bottleneck. U-Net concatenates instead, which would make them full blocks.',
  }, seed => {
    const r = rng(seed);
    const net = blank('U-Net-style', 'mse', [IN(4), { name: 'Encoder', act: 'relu', size: 3 },
      { name: 'Bottleneck', act: 'relu', size: 2 }, { name: 'Decoder', act: 'relu', size: 3 },
      { name: 'Output', act: 'identity', size: 4 }]);
    ['he', 'he', 'he', 'xavier'].forEach((scheme, l) => wireRandom(net, l, l + 1, r, { scheme }));
    wire(net, 1, 3, eye);
    wire(net, 0, 4, eye);
    smallBiases(net, r);
    const x = [0.9, 0.2, -0.4, 0.6];
    return layout(io(net, x, x));
  }),
  wide_deep: preset({
    group: SKIPS, label: 'Wide & deep (linear + MLP)', dataset: 'moons',
    note: 'The output adds two terms before one sigmoid: W⁽³⁾a⁽²⁾ from the deep MLP and W x straight from the inputs, a plain linear model (the wide part).',
  }, seed => {
    const r = rng(seed);
    const net = blank('Wide & deep', 'xent', [IN(2), { name: 'Deep 1', act: 'relu', size: 4 },
      { name: 'Deep 2', act: 'relu', size: 4 }, { name: 'Output', act: 'sigmoid', size: 1 }]);
    wireRandom(net, 0, 1, r, { scheme: 'he' });
    wireRandom(net, 1, 2, r, { scheme: 'he' });
    wireRandom(net, 2, 3, r, { fanIn: 6 });   // deep
    wireRandom(net, 0, 3, r, { fanIn: 6 });   // wide
    smallBiases(net, r);
    return layout(io(net, [0.6, -0.3], [1]));
  }),

  // ---------------------------------------------------------------- structure in W
  conv1d: preset({
    group: STRUCT, label: '1D convolution (banded W)',
    note: 'W is banded Toeplitz: every row is one tied kernel (−1, 2, −1) moved one column right, off-band masked. Training moves every copy together. The box input lights up only at its two edges.',
  }, () => {
    const net = blank('1D convolution', 'mse', [IN(8), { name: 'Conv, kernel 3', act: 'identity', size: 6 }]);
    wire(net, 0, 1, band([-1, 2, -1]), kernelTie('k'));
    tieBiasAll(net, 1, 'b');
    layout(io(net, [0, 0, 1, 1, 1, 1, 0, 0], 'self'), 600, 640);
    return alignTo(net, 1, 0, i => [i + 1]);
  }),
  conv1d_s2: preset({
    group: STRUCT, label: '1D convolution, stride 2',
    note: 'Stride 2: every row is the same tied kernel (¼, ½, ¼) moved two columns right, so W is 4×9 and the signal comes out half as long.',
  }, () => {
    const net = blank('1D convolution, stride 2', 'mse', [IN(9), { name: 'Conv, stride 2', act: 'identity', size: 4 }]);
    wire(net, 0, 1, band([0.25, 0.5, 0.25], 2), kernelTie('k', 2));
    tieBiasAll(net, 1, 'b');
    layout(io(net, [0, 1, 0, 1, 0, 1, 1, 1, 1], 'self'), 600, 720);
    return alignTo(net, 1, 0, i => [2 * i + 1]);
  }),
  avgpool: preset({
    group: STRUCT, label: 'Average pooling (fixed weights)',
    note: 'Each row of W is ½, ½ over its own pair of inputs, masked elsewhere: pooling is a fixed linear map (dashed edges never train).',
  }, () => {
    const net = blank('Average pooling', 'mse', [IN(8), { name: 'Avg pool, size 2', act: 'identity', size: 4 }]);
    wire(net, 0, 1, pairs(0.5), FIXED);
    layout(io(net, [0.2, 0.8, 1, 0.4, -0.6, -0.2, 0.9, 0.1], 'self'), 600, 640);
    return alignTo(net, 1, 0, i => [2 * i, 2 * i + 1]);
  }),
  maxpool: preset({
    group: STRUCT, label: 'Max pooling from ReLUs (exact)',
    note: 'max(a, b) = b + ReLU(a − b): the rows of W⁽¹⁾ take differences (1, −1), and the output adds b back through a skip term. Exact, with no max anywhere in the model.',
  }, () => {
    const net = blank('Max pooling from ReLUs', 'mse', [IN(4), { name: 'ReLU(a − b)', act: 'relu', size: 2 },
      { name: 'b + ReLU(a − b)', act: 'identity', size: 2 }]);
    wire(net, 0, 1, (i, j) => (j === 2 * i ? 1 : j === 2 * i + 1 ? -1 : null));
    wire(net, 1, 2, eye);
    wire(net, 0, 2, (i, j) => (j === 2 * i + 1 ? 1 : null));
    layout(io(net, [0.3, 0.9, 0.7, -0.2], 'self'));
    alignTo(net, 1, 0, i => [2 * i, 2 * i + 1]);
    return alignTo(net, 2, 1, i => [i]);
  }),
  lenet: preset({
    group: STRUCT, label: 'Tiny LeNet: conv, pool, dense, softmax',
    note: 'Three kinds of W in one net: the conv is banded with one tied kernel (½, 1, ½), the pool is fixed ½ pairs (dashed), the last is dense. A bump in the middle of the 8 pixels comes out as mid.',
  }, () => {
    const net = blank('Tiny LeNet', 'xent', [{ name: 'Input: 8 pixels', act: 'identity', size: 8 },
      { name: 'Conv', act: 'relu', size: 6 }, { name: 'Avg pool', act: 'identity', size: 3 },
      { name: 'Dense + softmax', act: 'softmax', size: 3, labels: tex(['left', 'mid', 'right']) }]);
    wire(net, 0, 1, band([0.5, 1, 0.5]), kernelTie('k'));
    setBias(net, 1, -0.5);
    tieBiasAll(net, 1, 'b');
    wire(net, 1, 2, pairs(0.5), FIXED);
    wire(net, 2, 3, (i, j) => (i === j ? 2 : -1));
    layout(io(net, [0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 0]), 780, 640);
    alignTo(net, 1, 0, i => [i + 1]);
    alignTo(net, 2, 1, i => [2 * i, 2 * i + 1]);
    return alignTo(net, 3, 2, i => [i]);
  }),
  gnn: preset({
    group: STRUCT, label: 'Graph convolution (W = adjacency)',
    note: 'W is the graph: row i averages node i and its neighbours (A + I, each row divided by its count) and every non-edge is masked. Two layers are two hops: node 1\'s signal reaches node 4, not 5.',
  }, () => {
    const net = blank('Graph convolution', 'mse', [{ name: 'Node features', act: 'identity', size: 5 },
      { name: 'Hop 1', act: 'relu', size: 5 }, { name: 'Hop 2', act: 'relu', size: 5 }]);
    const deg = GRAPH.map(row => row.reduce((s, v) => s + v, 0));
    const hat = (i, j) => (i === j || GRAPH[i][j] ? 1 / (deg[i] + 1) : null);
    wire(net, 0, 1, hat);
    wire(net, 1, 2, hat);
    return layout(io(net, [1, 0, 0, 0, 0], 'self'));
  }),
  towers: preset({
    group: STRUCT, label: 'Two towers (block-diagonal W)', dataset: 'circles',
    note: 'W⁽¹⁾ and W⁽²⁾ are block-diagonal: tower A sees only x₁, tower B only x₂, and the output adds them. f(x₁) + g(x₂) can still fit Circles, since x₁² + x₂² is a sum too.',
  }, seed => {
    const r = rng(seed);
    const t1 = i => (i < 3 ? 0 : 1), t2 = i => (i < 2 ? 0 : 1);
    const net = blank('Two towers', 'xent', [IN(2), { name: 'Towers, layer 1', act: 'tanh', size: 6 },
      { name: 'Towers, layer 2', act: 'tanh', size: 4 }, { name: 'Output', act: 'sigmoid', size: 1 }]);
    wireRandom(net, 0, 1, r, { keep: (i, j) => t1(i) === j });
    wireRandom(net, 1, 2, r, { keep: (i, j) => t2(i) === t1(j) });
    wireRandom(net, 2, 3, r);
    smallBiases(net, r);
    layout(io(net, [0.2, -0.3], [1]));
    spread(net, 1, t1, 40);
    return spread(net, 2, t2, 40);
  }),
  multitask: preset({
    group: STRUCT, label: 'Multi-task: shared trunk, 3 heads', dataset: 'three',
    note: 'W⁽¹⁾ is the trunk every task shares; W⁽³⁾ is block-diagonal, so each sigmoid reads only its own head. Three yes/no tasks (is it class k?) train on one summed loss.',
  }, seed => {
    const r = rng(seed);
    const head = i => Math.floor(i / 2);
    const net = blank('Multi-task', 'xent', [IN(2), { name: 'Shared trunk', act: 'tanh', size: 4 },
      { name: 'Task heads', act: 'tanh', size: 6 }, { name: 'Tasks', act: 'sigmoid', size: 3 }]);
    wireRandom(net, 0, 1, r);
    wireRandom(net, 1, 2, r);
    wireRandom(net, 2, 3, r, { keep: (i, j) => head(j) === i });
    smallBiases(net, r);
    layout(io(net, [0, 0.55], [1, 0, 0]));
    spread(net, 2, head, 24);
    return alignTo(net, 3, 2, i => [2 * i, 2 * i + 1]);
  }),

  // ---------------------------------------------------------------- sequences
  rnn: preset({
    group: SEQ, label: 'RNN unrolled over 4 steps',
    note: 'Step t computes h⁽ᵗ⁾ = tanh(h⁽ᵗ⁻¹⁾W_hh + x_t w_x + b), x_t on a skip edge. W_hh, w_x and b are tied: one set of parameters shared by every step, so training moves all the copies together.',
  }, () => {
    const T = 4, wx = [0.8, -0.5], Whh = [[0.5, -0.4], [0.3, 0.6]], bh = [0, 0.1];
    const net = blank('RNN unrolled', 'mse', [{ name: 'Inputs x₁…x₄', act: 'identity', size: T },
      ...Array.from({ length: T }, (_, t) => ({ name: `Step ${t + 1}`, act: 'tanh', size: 2 })),
      { name: 'Output', act: 'identity', size: 1 }]);
    // Ties in the row-vector (X W) convention: entry (source feature, target feature).
    for (let t = 1; t <= T; t++) {
      wire(net, 0, t, (i, j) => (j === t - 1 ? wx[i] : null), i => ({ tie: `w_x:1,${i + 1}` }));
      if (t > 1) wire(net, t - 1, t, Whh, (i, j) => ({ tie: `W_{hh}:${j + 1},${i + 1}` }));
      setBias(net, t, bh);
      nodesIn(net, t).forEach((n, i) => { n.tie = `b_h:${i + 1}`; });
    }
    wire(net, T, T + 1, [[1, -1]]);
    return layout(io(net, [0.5, -0.3, 0.9, 0.2], [0.5]));
  }),
  wavenet: preset({
    group: SEQ, label: 'Dilated causal conv (WaveNet-style)',
    note: 'Each W is lower-triangular: one tied kernel (½, ½) on the diagonal and at offset 1, 2, then 4, so no row reads the future and the receptive field doubles per layer. The last output is the mean of all 8.',
  }, () => {
    const D = [1, 2, 4];
    const net = blank('Dilated causal conv', 'mse', [{ name: 'Input over time', act: 'identity', size: 8 },
      ...D.map(d => ({ name: `Dilation ${d}`, act: 'identity', size: 8 }))]);
    D.forEach((d, l) => {
      // tap 1 reads x_{t-d}, tap 2 reads x_t
      wire(net, l, l + 1, (i, j) => (j === i || j === i - d ? 0.5 : null), (i, j) => ({ tie: `k^{(${l + 1})}:1,${j === i ? 2 : 1}` }));
      tieBiasAll(net, l + 1, `b^{(${l + 1})}`);
    });
    return layout(io(net, [0.8, -0.2, 0.5, 0.1, -0.4, 0.9, 0.3, -0.6], 'self'), 780, 640);
  }),

  // ---------------------------------------------------------------- attention
  // Hand-set: one-hot word types, so W_Q's rows read "what each kind of word asks for" and W_K's
  // "what it offers". sat (verb) asks for a noun, cat (noun) for a determiner, the asks nothing.
  words: preset({
    group: ATT, label: 'Word attention: the cat sat (hand-set)',
    note: 'Hand-set. Words come in one-hot (det, noun, verb). W_Q asks and W_K answers: sat\'s query points at the noun cat, cat\'s at the determiner the. The asks nothing (q = 0), so its row of A is even.',
  }, () => {
    const net = seqBlank('Word attention', 'mse', [
      { name: 'Words: det, noun, verb', tokens: 3, d: 3, label: (t, f) => [`\\mathrm{det}_{${t}}`, `\\mathrm{noun}_{${t}}`, `\\mathrm{verb}_{${t}}`][f - 1] },
      { name: 'Q, K, V', tokens: 3, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Attention Z', tokens: 3, d: 2, attention: { heads: 1 }, label: tokLabel('z') }]);
    const W = {
      Q: [[0, 0], [0, 4], [4, 0]],        // det asks nothing, noun asks "det?", verb asks "noun?"
      K: [[0, 1], [1, 0], [0, 0]],        // det offers (0, 1), noun (1, 0), verb nothing
      V: [[-0.8, 0.6], [1, 0], [0, -1]],  // three spread values, so the mix has a clear triangle
    };
    for (const g of QKV) {
      wireTied(net, 0, 1, `W_${g}`, W[g], { to: g });
      tieBias(net, 1, `b_${g}`, [0, 0], g);
    }
    net.meta.tokenNames = ['the', 'cat', 'sat'];
    return gridLayout(io(net, [1, 0, 0, 0, 1, 0, 0, 0, 1], 'self'));
  }),
  attention: preset({
    group: ATT, label: 'Self-attention (3 tokens × 2)', dataset: 'seq_max', lr: 0.3,
    note: 'Q = XW_Q, K = XW_K, V = XW_V, each one tied 2×2 matrix shared by all 3 tokens. Z = softmax(QKᵀ/√2)V; row i of A is where token i looks. Train: every token learns to look at the largest x₁.',
  }, seed => {
    const r = rng(seed);
    const net = seqBlank('Self-attention', 'mse', [
      { name: 'Tokens X', tokens: 3, d: 2, label: tokLabel('x') },
      { name: 'Q, K, V', tokens: 3, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Attention Z', tokens: 3, d: 2, attention: { heads: 1 }, label: tokLabel('z') }]);
    wireQKV(net, 0, 1, r);
    return gridLayout(seqIO(net, 'seq_max'));
  }),
  causal: preset({
    group: ATT, label: 'Causal attention: the previous token', dataset: 'seq_prev', lr: 0.3,
    note: 'The mask sets S_ij = −∞ for j > i, so token i reads only tokens up to i. Positions come in as (cos θ, sin θ); to copy the previous token, W_Q W_Kᵀ has to learn a rotation by one position.',
  }, seed => {
    const r = rng(seed);
    const net = seqBlank('Causal attention', 'mse', [
      { name: 'Tokens: c, cos θ, sin θ', tokens: 3, d: 3, label: (t, f) => [`c_{${t}}`, `\\cos\\theta_{${t}}`, `\\sin\\theta_{${t}}`][f - 1] },
      { name: 'Q, K, V', tokens: 3, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Masked attention Z', tokens: 3, d: 2, attention: { heads: 1, causal: true }, label: tokLabel('z') },
      { name: 'Output', tokens: 3, d: 1, label: t => `\\hat y_{${t}}` }]);
    wireQKV(net, 0, 1, r);
    wireTied(net, 2, 3, 'W_{out}', randW(r, 2, 1));
    tieBias(net, 3, 'b_{out}', randB(r, 1));
    return gridLayout(seqIO(net, 'seq_prev'));
  }),
  // Hand-set solution of the causal preset's task: W_Q is 4 R(120°) on the position rows, so
  // x W_Q turns token i's position (cos θ_i, sin θ_i) back one step, onto k_{i-1} = its position.
  causal_rot: preset({
    group: ATT, label: 'Previous token by rotation (hand-set)', dataset: 'seq_prev',
    note: 'Hand-set causal attention. Positions come in as (cos θ, sin θ) and W_Q turns them back by 120°, so q_i points at k_(i−1) and token i copies c_(i−1). The loss starts near 0.',
  }, () => {
    const g = 4, co = -0.5, si = Math.sqrt(3) / 2;   // g R(120°)
    const net = seqBlank('Previous token, hand-set', 'mse', [
      { name: 'Tokens: c, cos θ, sin θ', tokens: 3, d: 3, label: (t, f) => [`c_{${t}}`, `\\cos\\theta_{${t}}`, `\\sin\\theta_{${t}}`][f - 1] },
      { name: 'Q, K, V', tokens: 3, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Masked attention Z', tokens: 3, d: 2, attention: { heads: 1, causal: true }, label: tokLabel('z') },
      { name: 'Output', tokens: 3, d: 1, label: t => `\\hat y_{${t}}` }]);
    const W = {
      Q: [[0, 0], [r3(g * co), r3(-g * si)], [r3(g * si), r3(g * co)]],
      K: [[0, 0], [1, 0], [0, 1]],        // the key is the position itself
      V: [[1, 0], [0, 0], [0, 1]],        // the value carries c (and sin θ, so the values span a plane)
    };
    for (const k of QKV) {
      wireTied(net, 0, 1, `W_${k}`, W[k], { to: k });
      tieBias(net, 1, `b_${k}`, [0, 0], k);
    }
    wireTied(net, 2, 3, 'W_{out}', [[1], [0]]);
    tieBias(net, 3, 'b_{out}', [0]);
    return gridLayout(seqIO(net, 'seq_prev'));
  }),
  multihead: preset({
    group: ATT, label: 'Two heads: max and min', dataset: 'seq_minmax', lr: 1,
    note: 'heads = 2 splits Q, K and V by column: head 1 uses column 1 of each, head 2 column 2, and each head has its own A. Train: one head learns to find the largest x₁, the other the smallest.',
  }, seed => {
    const r = rng(seed);
    const net = seqBlank('Two attention heads', 'mse', [
      { name: 'Tokens X', tokens: 3, d: 2, label: tokLabel('x') },
      { name: 'Q, K, V', tokens: 3, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Attention Z, 2 heads', tokens: 3, d: 2, attention: { heads: 2 }, label: tokLabel('z') }]);
    wireQKV(net, 0, 1, r);
    return gridLayout(seqIO(net, 'seq_minmax'));
  }),
  transformer: preset({
    group: ATT, label: 'Transformer block (2 tokens)', dataset: 'seq_addmax', lr: 0.3,
    note: 'One block, all weights tied: Z = softmax(QKᵀ/√2)V, H = X + ZW_O, Y = H + ReLU(HW₁)W₂, the + X and + H being fixed residual edges. No LayerNorm: with d = 2 it sends every token to (±1, ∓1).',
  }, seed => {
    const r = rng(seed);
    const net = seqBlank('Transformer block', 'mse', [
      { name: 'Tokens X', tokens: 2, d: 2, label: tokLabel('x') },
      { name: 'Q, K, V', tokens: 2, d: 2, groups: QKV, label: qkvLabel },
      { name: 'Attention Z', tokens: 2, d: 2, attention: { heads: 1 }, label: tokLabel('z') },
      { name: 'H = X + Z W_O', tokens: 2, d: 2, label: tokLabel('h') },
      { name: 'FFN: ReLU(H W₁)', act: 'relu', tokens: 2, d: 4, label: tokLabel('f') },
      { name: 'Y = H + FFN', tokens: 2, d: 2, label: (t, f) => `\\hat y_{${t},${f}}` }]);
    wireQKV(net, 0, 1, r);
    wireTied(net, 2, 3, 'W_O', randW(r, 2, 2, 'small'));   // small residual branches: the block starts near Y = X
    wireResidual(net, 0, 3);
    tieBias(net, 3, 'b_O', randB(r, 2));
    wireTied(net, 3, 4, 'W_1', randW(r, 2, 4, 'he'));
    tieBias(net, 4, 'b_1', randB(r, 4));
    wireTied(net, 4, 5, 'W_2', randW(r, 4, 2, 'small'));
    wireResidual(net, 3, 5);
    tieBias(net, 5, 'b_2', randB(r, 2));
    return layout(seqIO(net, 'seq_addmax'), 1100, 1100);
  }),

  // ---------------------------------------------------------------- embeddings and autoencoders
  autoencoder: preset({
    group: EMB, label: 'Autoencoder (4-2-4)',
    note: 'W⁽¹⁾ (2×4) squeezes the 4 inputs into a 2-number code and W⁽²⁾ (4×2) rebuilds them. The target is the input itself.',
  }, seed => {
    const x = [0.9, 0.1, 0.6, 0.3];
    return layout(stack('Autoencoder', 'mse',
      [IN(4), { name: 'Code', act: 'tanh', size: 2 }, { name: 'Reconstruction', act: 'sigmoid', size: 4 }],
      { seed, inputs: x, targets: x }));
  }),
  embedding: preset({
    group: EMB, label: 'Word embedding (one-hot lookup)',
    note: 'x is one-hot, so W⁽¹⁾x is just the cat column of W⁽¹⁾: an embedding is a table lookup. W⁽²⁾ starts as W⁽¹⁾ᵀ, so each score is a dot product with the cat vector.',
  }, () => {
    const words = tex(['the', 'cat', 'sat', 'on', 'mat']);
    const E = [[0.1, 0.8, -0.6, -0.3, 0.7], [0.9, 0.3, 0.4, -0.8, -0.1]];   // column k: word k's vector
    const net = blank('Word embedding', 'xent', [{ name: 'One-hot word', act: 'identity', size: 5, labels: words },
      { name: 'Embedding', act: 'identity', size: 2 },
      { name: 'Context word', act: 'softmax', size: 5, labels: words }]);
    wire(net, 0, 1, E);
    wire(net, 1, 2, (i, j) => E[j][i]);
    return layout(io(net, [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]));
  }),
  pca_ae: preset({
    group: EMB, label: 'Linear autoencoder = PCA (3-2-3)', dataset: 'cloud',
    note: 'All linear, so x̂ = W⁽²⁾W⁽¹⁾x + b has rank 2. Train on the flat cloud: the two columns of W⁽²⁾ come to span its plane, the span of the top two principal components.',
  }, seed => {
    const x = [0.47, 0.33, 0.27];
    return layout(stack('Linear autoencoder (PCA)', 'mse',
      [IN(3), { name: 'Code', act: 'identity', size: 2 }, { name: 'Reconstruction', act: 'identity', size: 3 }],
      { seed, inputs: x, targets: x }));
  }),

  // ---------------------------------------------------------------- teaching demos
  vanishing: preset({
    group: DEMO, label: 'Vanishing gradients (sigmoid chain)', dataset: 'line',
    note: 'Turn on Backward: each sigmoid multiplies δ by w·σ′(z) ≤ ¼, so ∂L/∂W shrinks about 4× per layer toward the input. Train on Line: the early layers barely move.',
  }, () => {
    const net = blank('Vanishing gradients', 'mse', [IN(1),
      ...[1, 2, 3, 4, 5].map(i => ({ name: `σ ${i}`, act: 'sigmoid', size: 1 })),
      { name: 'Output', act: 'identity', size: 1 }]);
    for (let l = 1; l < net.layers.length; l++) wire(net, l - 1, l, [[1]]);
    return layout(io(net, [1], [0.9]), 1100);
  }),
  gan: preset({
    group: DEMO, label: 'GAN as one net: D(G(z))',
    note: 'Only the composition D(G(z)). With target 1 the backward pass is G\'s update (fool D); a real GAN also shows D real data and trains D and G in turns.',
  }, seed => {
    const r = rng(seed);
    const net = blank('GAN: D(G(z))', 'xent', [
      { name: 'Noise z', act: 'identity', size: 2, labels: ['z_{1}', 'z_{2}'] },
      { name: 'G hidden', act: 'relu', size: 4 },
      { name: 'G(z): fake x', act: 'identity', size: 2, labels: ['\\tilde x_{1}', '\\tilde x_{2}'] },
      { name: 'D hidden', act: 'relu', size: 4 },
      { name: 'D: real?', act: 'sigmoid', size: 1, labels: ['D'] }]);
    ['he', 'xavier', 'he', 'xavier'].forEach((scheme, l) => wireRandom(net, l, l + 1, r, { scheme }));
    smallBiases(net, r);
    return layout(io(net, [0.5, -0.8], [1]));
  }),
};
