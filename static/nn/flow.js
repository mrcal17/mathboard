// Net tab: the Flow view (docs/NN_FLOW.md). While it is on, it covers #nn-stage as the 3D view does
// and draws one whole forward pass left to right, as live matrix tiles with their shapes:
//   words (one-hot) -> X = O W_E + P -> per head Q, K, V -> scores S -> softmax A -> A V -> concat
//   -> Z W_O -> + residual -> FFN -> F W_2 -> + residual -> logits -> the next word.
// A language model's ending shows only the last position (the next word's distribution, as
// generating reads it); `every` shows every position's prediction, as training scores them.
// The sentence heads the flow as numbered words, every matrix names its rows by position and word,
// and brackets under the tiles give their widths (d_model for the residual stream, the FFN's, the
// vocabulary's).
// Any net gets a flow: attention layers as their stages, tokenwise tied layers as products and
// sums, anything else as one tile per layer (with a note when the net has no attention).
// Stepping (◀ ▶, ← →) lights one stage at a time and fades the ones not computed yet; ▶ plays.
// Hovering a cell traces what it was computed from (its sources get a frame, the tip does the
// sum with numbers), clicking a stage focuses its layer through the lens, clicking a cell selects
// its neuron. Head chips knock a head out: its columns of the concat are 0 and everything after
// is recomputed (the net itself is not changed).
//
// state.flow (owner: this module) is null (off) or a cleanFlow() object; the audience mirrors it.

import { colorFor } from './store.js';
import { cleanLens, copyLens, tokenLabel } from './focus.js';
import * as M from './model.js';

const isNum = v => typeof v === 'number' && Number.isFinite(v);
const isObj = v => !!v && typeof v === 'object' && !Array.isArray(v);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const grid = (r, c, f) => Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => f(i, j)));
const argmax = row => row.reduce((b, v, i) => (v > row[b] ? i : b), 0);
const QKV = ['Q', 'K', 'V'];
const isQKV = g => Array.isArray(g) && g.length === 3 && g.every((s, i) => s === QKV[i]);
const f2 = v => (v === -Infinity ? '−∞' : M.fmt(v, 2).replace('-', '−'));
// A number for a narrow cell: two figures without the leading 0 (.12, −.34, 1.2, 12), as the 3D view
// writes them on narrow faces.
const fc = v => {
  if (v === -Infinity) return '−∞';
  if (!isNum(v)) return '';
  const a = Math.abs(v), s = a >= 9.5 ? String(Math.round(a)) : a >= 0.995 ? a.toFixed(1) : a.toFixed(2).slice(1);
  return s === '.00' ? '0' : `${v < 0 ? '−' : ''}${s}`;
};
const range = n => Array.from({ length: n }, (_, i) => i);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ================================================================ pure helpers (tests: tests/nn_flow.test.mjs)

// state.flow, completed and repaired (null stays null): stage null = the whole pass, else the lit
// stage (0-based); play only with a stage; off = knocked-out heads (0-based, sorted); nums = numbers
// in the cells; every = a language model's ending shows every position (how it's trained), not only
// the last; hover = the cell the presenter points at, { t: tile id, i, j }.
export function cleanFlow(v) {
  if (!isObj(v)) return null;
  const stage = Number.isInteger(v.stage) && v.stage >= 0 ? v.stage : null;
  const off = [...new Set((Array.isArray(v.off) ? v.off : []).filter(h => Number.isInteger(h) && h >= 0 && h < 64))].sort((a, b) => a - b);
  const h = v.hover;
  const hover = isObj(h) && typeof h.t === 'string' && Number.isInteger(h.i) && Number.isInteger(h.j) ? { t: h.t, i: h.i, j: h.j } : null;
  return { stage, play: v.play === true && stage !== null, off, nums: v.nums !== false, every: v.every === true, hover };
}

// Every head of an attention layer (g = model.attnSpec) from its Q, K, V layer's activations x (the Q
// block, then K, then V, each token-major), computed by model.js itself (model.attend: the variants'
// RoPE, ALiBi, window and linear attention included). out is the heads' Z side by side, token-major,
// with the columns of the heads in `off` left at 0 (knocked out).
export function attendHeads(x, g, off = new Set()) {
  const { tokens: n, d, heads: H, dh } = g, r = M.attend(x, g), out = r.out;
  for (let h = 0; h < H; h++) if (off.has(h)) for (let i = 0; i < n; i++) for (let f = 0; f < dh; f++) out[i * d + h * dh + f] = 0;
  return { heads: r.heads, out };
}

// softmax (and any activation) runs per token on a token layer, as in model.js's plan().
function segOf(net, l) {
  const s = M.tokenShape(net, l);
  return s.tokens > 1 || s.groups ? s.d : M.nodesIn(net, l).length;
}

// The forward pass again from layer `from` on, given every layer's activations a (layers before
// `from` are kept): dense layers as b + Σ W a then the activation (model.matrices), attention layers
// with attendHeads, the heads in `off` knocked out. Returns { a, z, attn } (per layer, as forward).
export function propagate(net, a0, from = 1, { off = [] } = {}) {
  const L = net.layers.length, offs = new Set(off);
  const a = a0.map(r => Array.from(r)), z = a.map(() => null), attn = a.map(() => null);
  const mats = M.matrices(net);
  for (let l = Math.max(1, from); l < L; l++) {
    const g = M.attnSpec(net, l);
    if (g) {
      const r = attendHeads(a[l - 1], g, offs);
      a[l] = r.out;
      z[l] = r.out.slice();
      attn[l] = { heads: r.heads, tokens: g.tokens, dk: g.dh, scale: g.scale, causal: g.causal };
      for (const k of ['window', 'pos', 'linear']) if (g[k]) attn[l][k] = g[k];
      continue;
    }
    const m = mats[l - 1], zl = m.b.slice();
    for (const t of m.terms) {
      const src = a[t.k];
      t.W.forEach((row, i) => {
        let s = 0;
        row.forEach((w, j) => { if (w) s += w * src[j]; });
        zl[i] += s;
      });
    }
    z[l] = zl;
    a[l] = M.activate(m.act, zl, null, segOf(net, l));
  }
  return { a, z, attn };
}

// fwd with the heads in `off` knocked out of every attention layer that has them: from the first
// such layer on everything is recomputed (propagate); before it fwd is kept as it is. from: that
// layer (-1 when nothing changed, and then a, z and attn are fwd's own).
export function ablate(net, fwd, off = []) {
  const offs = [...new Set(off)].filter(h => Number.isInteger(h) && h >= 0);
  let first = -1;
  for (let l = 1; l < net.layers.length && first < 0; l++) {
    const g = M.attnSpec(net, l);
    if (g && offs.some(h => h < g.heads)) first = l;
  }
  if (first < 0) return { a: fwd.a, z: fwd.z, attn: fwd.attn, from: -1 };
  const r = propagate(net, fwd.a, first, { off: offs });
  for (let l = 0; l < first; l++) { r.z[l] = fwd.z[l]; r.attn[l] = fwd.attn[l]; }
  return { ...r, from: first };
}

// A layer's symbol: the letter before '=' in its name ('H = X + Z W_O' -> H), else its neurons'
// label letter (h_{1,2} -> H, \hat y_{1} -> Ŷ), else a^{(l)}.
function symOf(net, l) {
  const name = String(net.layers[l]?.name || '');
  const m = /^\s*([A-Za-z])([₀-₉]*)\s*=/.exec(name);   // H₂ = ... (a block's) -> H_{2}
  if (m) return m[1].toUpperCase() + (m[2] ? `_{${[...m[2]].map(c => c.charCodeAt(0) - 0x2080).join('')}}` : '');
  const lab = String(M.nodesIn(net, l)[0]?.label || '');
  const h = /^\s*\\hat\s*\{?\s*([a-zA-Z])\s*\}?\s*_/.exec(lab);
  if (h) return `\\hat ${h[1].toUpperCase()}`;
  const b = /^\s*([a-zA-Z])\s*_/.exec(lab);
  return b ? b[1].toUpperCase() : `a^{(${l})}`;
}
const tieBase = tie => (typeof tie === 'string' ? tie.replace(/:[\d,]+$/, '') : null);
// a cell's name in a tip: H<sub>2,3</sub>, or (N<sub>1</sub>)<sub>2,3</sub> for a symbol that has its own subscript (i, j 0-based)
const cellName = (sym, i, j) => `${/_/.test(sym) ? `(${texHtml(sym)})` : texHtml(sym)}<sub>${i + 1},${j + 1}</sub>`;

// KaTeX-ish source to plain HTML for tips and captions: W_{out} -> W<sub>out</sub>, K^{\top} -> Kᵀ.
export function texHtml(s) {
  let t = String(s ?? '');
  t = t.replace(/\\text\{([^}]*)\}/g, '$1').replace(/\\hat\s*\{?\s*([A-Za-z])\s*\}?/g, '$1̂')
    .replace(/\\ell/g, 'ℓ').replace(/\^\{?\\top\}?/g, 'ᵀ').replace(/\\,|\\;|\\!/g, ' ')
    .replace(/\\operatorname\{([^}]*)\}/g, '$1').replace(/\\(?:left|right|big|Big)/g, '');
  t = esc(t.replace(/\\([A-Za-z]+)/g, '$1'));
  t = t.replace(/_\{([^}]*)\}/g, '<sub>$1</sub>').replace(/_([A-Za-z0-9])/g, '<sub>$1</sub>')
    .replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>').replace(/\^([A-Za-z0-9])/g, '<sup>$1</sup>');
  return t.replace(/[{}]/g, '');
}

// The tiny language model's family (model.PRESETS with family 'tiny_lm'): the preset a net was built
// from, matched by its title (as train.js matches presets), as { key, axis, short, note }, else null.
let familyTitles = null;
export function variantOf(net) {
  if (!familyTitles) {
    familyTitles = new Map();
    for (const [key, p] of Object.entries(M.PRESETS)) if (p.family === 'tiny_lm') familyTitles.set(p.title, key);
  }
  const key = familyTitles.get(net?.meta?.title);
  const p = key && M.PRESETS[key];
  return p ? { key, axis: p.axis, short: p.short, note: p.note } : null;
}
// The family's presets for the variant picker, by axis in menu order: [[axis, [{ key, short }]]].
export function variantMenu() {
  const by = new Map();
  for (const [key, p] of Object.entries(M.PRESETS)) {
    if (p.family !== 'tiny_lm') continue;
    if (!by.has(p.axis)) by.set(p.axis, []);
    by.get(p.axis).push({ key, short: p.short });
  }
  return [...by];
}

const fwdFits = (net, fwd) => !!fwd && Array.isArray(fwd.a) && fwd.a.length === net.layers.length
  && net.layers.every((_, l) => fwd.a[l]?.length === M.nodesIn(net, l).length);

// The whole flow of net's forward pass (fwd: the store's, reused when it fits and no head is off;
// every: a language model's ending at every position, else only at the last).
// {
//   stages: [{ key, title, tex, text, layer (id), part, mode, rows: [{ head, tiles: [tile id], ops }] }],
//   tiles: { [id]: { id, l, tex, kind: 'mat' | 'bars' | 'dist', rows, cols, v, pos, rowLab, rowNo, colLab,
//            head, off, mask, scale: 'act' | 'attn' | 'prob', onehot, target, dim, dimRows, note,
//            node(i, j) -> id | null, src(i, j) -> [[tile, i, j]], tip(i, j) -> html } },
//   nodeCell: { [nodeId]: [tile, i, j] },   // where each neuron is drawn
//   words: (string | null)[] per position (a one-hot input over meta.vocab), labels: per position,
//   numbered: the positions are words (the sentence reads 1 the, 2 cat, ...),
//   next: null | { word, p, target, t },    // the last position's prediction
//   lm: a language model's ending (a softmax over meta.vocab at every position), every: shown so,
//   residual: [{ l, sym, from: [sym], branch: [name], d, tiles, src }], ffn: null | { id, d, from },
//   dims: null | { model, ffn, vocab }, dimsText (html: why the residual stream keeps one width),
//   attention: bool, why: null | 'empty' | 'no-attention', max: { act }, heads: the most heads,
//   T: the most tokens of any layer,
// }
// A tile's pos maps its rows to positions (the last-position tiles have one row); dim is the
// bracket under it ({ kind: 'model' | 'ffn' | 'vocab', html }); dimRows are rows the next stage
// does not read (Y's earlier rows, when only the last position goes on).
export function buildFlow(net, { fwd = null, off = [], every = false } = {}) {
  const L = net?.layers?.length || 0;
  const F = {
    stages: [], tiles: {}, nodeCell: {}, words: [], labels: [], numbered: false, next: null, lm: false, every: !!every,
    residual: [], ffn: null, dims: null, dimsText: '', attention: false, why: null, max: { act: 1, bias: 1 }, heads: 1, T: 1,
    norms: [], variant: variantOf(net),
  };
  if (L < 2 || !net.nodes?.length) { F.why = 'empty'; return F; }
  const base = fwdFits(net, fwd) ? fwd : M.forward(net);
  const V = ablate(net, base, off);
  const offs = new Set(off);
  const ns = net.layers.map((_, l) => M.nodesIn(net, l));
  const shp = net.layers.map((_, l) => M.tokenShape(net, l));
  const att = net.layers.map((_, l) => M.attnSpec(net, l));
  const T = Math.max(1, ...shp.map(s => s.tokens));
  F.T = T;
  F.attention = att.some(Boolean);
  F.heads = Math.max(1, ...att.map(g => g?.heads || 1));
  if (!F.attention) F.why = 'no-attention';
  let mx = 0;
  V.a.forEach(r => r?.forEach(v => { if (isNum(v)) mx = Math.max(mx, Math.abs(v)); }));
  V.z.forEach(r => r?.forEach(v => { if (isNum(v)) mx = Math.max(mx, Math.abs(v)); }));
  F.max.act = mx || 1;
  // ALiBi's biases have their own colour scale (they reach -slope · (n - 1))
  let mb = 0;
  V.attn.forEach(r => r?.heads?.forEach(h => h.B?.forEach(row => row.forEach(v => { mb = Math.max(mb, Math.abs(v)); }))));
  F.max.bias = mb || 1;
  // how position reaches the model when X has no position vector, and whether every attention layer is causal
  const causalAll = att.filter(Boolean).every(g => g.causal);
  const pe = att.find(g => g?.pos)?.pos;
  const posHow = pe === 'rope' ? 'RoPE brings position in at the attention layer, turning q and k by their positions.'
    : pe === 'alibi' ? 'ALiBi brings position in at the attention layer, as a penalty on the scores of distant words.'
      : causalAll ? `the order reaches the model only through the causal mask (position 1 sees one word, position ${T} sees all ${T}), so it has to tell positions apart by what they can see.`
        : 'with no mask either, nothing tells the model the order of the words.';

  // ---- positions: a one-hot input over meta.vocab names its words
  const voc = Array.isArray(net.meta?.vocab) && net.meta.vocab.length ? net.meta.vocab.map(w => String(w)) : null;
  const s0 = shp[0], a0 = V.a[0];
  const oneHot = !!voc && s0.d === voc.length && !s0.groups && Array.from({ length: s0.tokens }, (_, t) => {
    const row = a0.slice(t * s0.d, (t + 1) * s0.d);
    return row.filter(v => Math.abs(v - 1) < 1e-6).length === 1 && row.every(v => Math.abs(v) < 1e-6 || Math.abs(v - 1) < 1e-6);
  }).every(Boolean);
  F.words = Array.from({ length: T }, (_, t) => (oneHot && t < s0.tokens ? voc[argmax(a0.slice(t * s0.d, (t + 1) * s0.d))] : null));
  F.labels = F.words.map((w, t) => w ?? tokenLabel(net, t));
  F.numbered = T > 1 && F.words.every(Boolean);
  const posName = t => (F.words[t] ? `${t + 1} “${esc(F.words[t])}”` : `position ${t + 1}`);
  // the words a position may see (itself and the ones before it), for "after the cat"
  const seen = t => (F.words.slice(0, t + 1).every(Boolean) ? F.words.slice(0, t + 1).join(' ') : null);

  // ---- tiles
  const tile = (id, o) => (F.tiles[id] = {
    id, kind: 'mat', head: null, off: false, mask: null, rowLab: null, rowNo: null, colLab: null, scale: 'act', onehot: false,
    target: null, pos: null, dim: null, dimRows: null, note: null,
    node: () => null, src: () => [], tip: () => '', ...o,
  });
  const layerTileOf = [];   // layer -> its output tile id (per group: layerTileOf[l][g])
  const cellOfNode = (id, cell) => { F.nodeCell[id] = cell; };
  const inEdges = new Map();
  for (const e of net.edges) {
    if (!inEdges.has(e.to)) inEdges.set(e.to, []);
    inEdges.get(e.to).push(e);
  }
  const nodeIdx = new Map();
  ns.forEach((list, l) => list.forEach((n, k) => nodeIdx.set(n.id, [l, k])));
  const val = (l, k) => V.a[l]?.[k];
  const actOf = l => (att[l] ? 'identity' : Object.hasOwn(M.ACTS, net.layers[l].act) ? net.layers[l].act : 'identity');
  // A neuron's incoming sum, told as its parts: each source layer's shared matrix (or plain
  // weights), the fixed residual and the bias. Returns html.
  function nodeSum(id) {
    const [l, k] = nodeIdx.get(id) || [];
    if (l === undefined || l === 0) return '';
    const parts = new Map();
    for (const e of inEdges.get(id) || []) {
      const src = nodeIdx.get(e.from);
      if (!src) continue;
      const key = e.fixed ? `res:${src[0]}` : `${src[0]}:${tieBase(e.tie) || 'W'}`;
      const p = parts.get(key) || { l: src[0], name: e.fixed ? null : tieBase(e.tie), s: 0 };
      p.s += (e.w || 0) * (val(src[0], src[1]) || 0);
      parts.set(key, p);
    }
    const n = ns[l][k], out = [];
    for (const p of parts.values()) {
      const who = texHtml(symOf(net, p.l));
      out.push(p.name === null ? `${who} ${f2(p.s)}` : `${who}·${texHtml(p.name)} ${f2(p.s)}`);
    }
    const b = n.bias || 0;
    if (b || n.tie) out.push(`${n.tie ? texHtml(tieBase(n.tie)) : 'b'} ${f2(b)}`);
    return out.join(' + ');
  }
  const nodeTip = (id, title) => {
    const [l, k] = nodeIdx.get(id) || [];
    if (l === undefined) return '';
    const act = actOf(l), z = V.z[l]?.[k], a = V.a[l]?.[k];
    const sum = nodeSum(id);
    if (!l) return `${title} = ${f2(a)}`;
    const fn = act === 'identity' ? '' : act === 'softmax' ? 'softmax' : M.ACTS[act]?.label || act;
    return `${title} = ${fn ? `${fn}(` : ''}${sum || f2(z)}${fn ? ')' : ''}<br>= ${fn && act !== 'softmax' ? `${fn}(${f2(z)}) = ` : ''}${f2(a)}`;
  };
  const nodeSrc = id => {
    const out = [];
    for (const e of inEdges.get(id) || []) { const c = F.nodeCell[e.from]; if (c) out.push(c); }
    return out;
  };
  // a layer's activations (group g) as a tile; nodes map one to one
  function layerTile(l, g, o = {}) {
    const { tokens, d, groups } = shp[l], gi = g ?? 0, off0 = gi * tokens * d;
    const id = o.id || (groups ? `L${l}.${groups[gi]}` : `L${l}`);
    const vec = o.vec || V.a[l];
    const t = tile(id, {
      l, rows: tokens, cols: d, v: grid(tokens, d, (i, j) => vec[off0 + i * d + j]),
      tex: o.tex || (groups ? groups[gi] : symOf(net, l)),
      node: (i, j) => ns[l][off0 + i * d + j]?.id ?? null,
      src: (i, j) => nodeSrc(ns[l][off0 + i * d + j]?.id),
      tip: (i, j) => nodeTip(ns[l][off0 + i * d + j]?.id, cellName(t.tex, i, j)),
      ...o,
    });
    for (let i = 0; i < tokens; i++) for (let j = 0; j < d; j++) {
      const nid = ns[l][off0 + i * d + j]?.id;
      if (nid && !F.nodeCell[nid]) cellOfNode(nid, [id, i, j]);
    }
    (layerTileOf[l] ||= [])[gi] = id;
    return id;
  }
  const rowSrc = (id, i) => (F.tiles[id] ? Array.from({ length: F.tiles[id].cols }, (_, j) => [id, i, j]) : []);
  // Row t of layer k (of one group) wherever its neurons are drawn (a Q, K, V layer is split by heads).
  const rowCells = (k, group, t) => {
    const { tokens, d, groups } = shp[k], g = group && groups ? groups.indexOf(group) : 0;
    return ns[k].slice(g * tokens * d + t * d, g * tokens * d + (t + 1) * d).map(n => F.nodeCell[n.id]).filter(Boolean);
  };
  const colSrc = (id, j) => (F.tiles[id] ? Array.from({ length: F.tiles[id].rows }, (_, i) => [id, i, j]) : []);
  const stage = s => { F.stages.push({ part: null, text: '', ...s }); };
  const lid = l => net.layers[l].id;

  // ---- layer 0
  {
    const id = layerTile(0, null, {
      id: 'in', tex: oneHot ? 'O' : symOf(net, 0), rowLab: s0.tokens > 1 ? F.labels.slice(0, s0.tokens) : null, colLab: oneHot ? voc : null, onehot: oneHot,
      tip: (i, j) => (oneHot ? `${posName(i)}: ${esc(voc[j])} ${f2(a0[i * s0.d + j])}` : `${texHtml(symOf(net, 0))}<sub>${i + 1},${j + 1}</sub> = ${f2(a0[i * s0.d + j])}`),
    });
    stage({
      key: 'in', title: oneHot ? 'Words' : 'Input', layer: lid(0), rows: [{ head: null, tiles: [id], ops: [] }],
      tex: oneHot ? 'O_{t} = \\text{one-hot}(w_t)' : `${symOf(net, 0)}`,
      text: oneHot ? 'The words, a row per position: a 1 in the column of its word.' : 'The input, a row per token.',
    });
  }

  for (let l = 1; l < L; l++) {
    if (att[l]) { attentionStages(l); continue; }
    if (att[l + 1] && isQKV(shp[l].groups)) { qkvStage(l); continue; }
    denseStages(l);
  }

  // ---- Q, K, V split into heads
  function qkvStage(l) {
    const g = att[l + 1], H = g.heads, dh = g.dh, d = g.d, n = g.tokens, heads = V.attn[l + 1]?.heads || [];
    const tm = M.tiedMatrices(net, l), src = tm.length && tm[0].k !== null ? symOf(net, tm[0].k) : 'X';
    // Heads that share their K and V (multi-query, grouped-query): every column of theirs is the same
    // shared weights and bias (the same tie ids), so their K (V) are one matrix. rep[G][h] = the first
    // head with head h's K (V).
    const sig = (G, h) => range(dh).map(f => {
      const nd = ns[l][QKV.indexOf(G) * n * d + h * dh + f];
      return [nd?.tie || nd?.id, ...(inEdges.get(nd?.id) || []).map(e => e.tie || e.id).sort()].join('|');
    }).join('/');
    const rep = Object.fromEntries(['K', 'V'].map(G => { const s = range(H).map(h => sig(G, h)); return [G, s.map(x => s.indexOf(x))]; }));
    const shared = H > 1 && ['K', 'V'].some(G => rep[G].some((r, h) => r !== h));
    const rows = [];
    for (let h = 0; h < H; h++) {
      const ids = QKV.map((G, gi) => {
        const id = H > 1 ? `${G.toLowerCase()}${l}.${h}` : `L${l}.${G}`;
        const at = (i, f) => gi * n * d + i * d + h * dh + f;
        const texName = H > 1 ? `${G}_{${h + 1}}` : G;
        const same = G !== 'Q' && H > 1 && rep[G][h] !== h ? `${G}_{${rep[G][h] + 1}}` : null;
        tile(id, {
          l, head: H > 1 ? h : null, rows: n, cols: dh, tex: texName, same,
          v: grid(n, dh, (i, f) => (heads[h] ? heads[h][G][i][f] : V.a[l][at(i, f)])),
          node: (i, f) => ns[l][at(i, f)]?.id ?? null,
          src: (i, f) => nodeSrc(ns[l][at(i, f)]?.id),
          tip: (i, f) => nodeTip(ns[l][at(i, f)]?.id, H > 1 ? `${G}<sub>${h + 1}</sub>[${i + 1},${f + 1}]` : `${G}<sub>${i + 1},${f + 1}</sub>`),
        });
        for (let i = 0; i < n; i++) for (let f = 0; f < dh; f++) { const nid = ns[l][at(i, f)]?.id; if (nid) cellOfNode(nid, [id, i, f]); }
        return id;
      });
      rows.push({ head: H > 1 ? h : null, tiles: ids, ops: [] });
    }
    for (let gi = 0; gi < 3; gi++) (layerTileOf[l] ||= [])[gi] = rows[0].tiles[gi];
    const W = G => tm.find(m => m.toGroup === G)?.name || `W_${G}`;
    const b = G => tieBase(ns[l][QKV.indexOf(G) * n * d]?.tie) || `b_${G}`;
    let text = `Three shared matrices turn every position's ${texHtml(src)} row into a query, a key and a value${H > 1 ? `, each then cut by columns into ${H} heads of ${dh}` : ''}.`;
    if (shared) {
      const kv = new Set(rep.K).size, wk = tm.find(m => m.toGroup === 'K')?.W, size = wk ? `${wk.length} × ${wk[0]?.length}` : '';
      const groups = [...new Set(rep.K)].map(r => range(H).filter(h => rep.K[h] === r).map(h => h + 1).join(' and ')).join(', ');
      text += kv === 1 ? ` Multi-query: all ${H} heads read one K and one V (heads ${groups}); only Q is their own.`
        : ` Grouped-query: heads ${groups} share a K and a V each, ${kv} K, V heads for ${H} queries.`;
      text += `${size ? ` W_K and W_V are ${size}, not ${d} × ${d}` : ''}: while generating, a model stores ${H / kv} times fewer keys and values.`;
    }
    stage({
      key: `qkv${l}`, title: H > 1 ? 'Q, K, V per head' : 'Q, K, V', layer: lid(l), rows,
      tex: QKV.map(G => `${G} = ${src}\\,${W(G)} + ${b(G)}`).join(',\\;'),
      text,
    });
  }

  // ---- attention: scores, softmax, weighted sum, concat
  function attentionStages(l) {
    const g = att[l], H = g.heads, dh = g.dh, n = g.tokens, d = g.d, A = V.attn[l] || base.attn[l];
    const qkv = layerTileOf[l - 1] || [];
    const hid = (p, h) => `${p}${l}.${h}`;
    const tq = h => (H > 1 ? `q${l - 1}.${h}` : qkv[0]), tk = h => (H > 1 ? `k${l - 1}.${h}` : qkv[1]), tv = h => (H > 1 ? `v${l - 1}.${h}` : qkv[2]);
    const sub = h => (H > 1 ? `_{${h + 1}}` : '');
    const at = (sym, h, i, j) => (H > 1 ? `${sym}<sub>${h + 1}</sub>[${i + 1},${j + 1}]` : `${sym}<sub>${i + 1},${j + 1}</sub>`);
    // rows are the queries, columns the keys: by word, or by position number when the rows name the words
    const lab = F.labels.slice(0, n), keys = F.numbered ? range(n).map(t => String(t + 1)) : lab;
    const mask = grid(n, n, (i, j) => !M.attnVisible(g, i, j));
    const rows = (...ps) => Array.from({ length: H }, (_, h) => ({ head: H > 1 ? h : null, tiles: ps.map(p => hid(p, h)), ops: ps.slice(1).map(() => '') }));
    const hs = h => A?.heads?.[h];
    const rope = g.pos === 'rope', alibi = g.pos === 'alibi', lin = !!g.linear, hsub = h => (H > 1 ? `<sub>${h + 1}</sub>` : '');
    const deg = p => (M.ropeFreq(p, dh) * 180) / Math.PI;
    // RoPE: q and k turned by their position's angles, in their own stage before the scores
    if (rope) {
      for (let h = 0; h < H; h++) {
        for (const [G, key, srcT] of [['Q', 'Qr', tq(h)], ['K', 'Kr', tk(h)]]) {
          const raw = () => hs(h)?.[G], rot = () => hs(h)?.[key], s = G.toLowerCase();
          tile(hid(`r${s}`, h), {
            l, head: H > 1 ? h : null, rows: n, cols: dh, tex: `\\tilde ${G}${sub(h)}`, v: rot() || grid(n, dh, () => NaN), rowLab: lab,
            src: (i, f) => { const p0 = f - (f % 2); return [[srcT, i, p0], ...(p0 + 1 < dh ? [[srcT, i, p0 + 1]] : [])]; },
            tip: (i, f) => {
              const p = Math.floor(f / 2), x = raw()[i], a = `${i} × ${M.fmt(deg(p), 1)}°`, y = rot()[i][f];
              if (2 * p + 1 >= dh) return `${s}̃<sub>${i + 1},${f + 1}</sub> = ${s}<sub>${i + 1},${f + 1}</sub> = ${f2(y)} (an odd last column is not turned)`;
              const u = x[2 * p], v = x[2 * p + 1], th = i * M.ropeFreq(p, dh), co = Math.cos(th), si = Math.sin(th);
              const expr = f % 2 === 0 ? `${f2(u)}·cos − ${f2(v)}·sin = ${f2(u)}·${f2(co)} − ${f2(v)}·${f2(si)}` : `${f2(u)}·sin + ${f2(v)}·cos = ${f2(u)}·${f2(si)} + ${f2(v)}·${f2(co)}`;
              return `${s}̃${hsub(h)}[${i + 1},${f + 1}]: pair ${p + 1} of ${posName(i)} turned by ${a}<br>= ${expr} = ${f2(y)}`;
            },
          });
        }
      }
      const angles = range(Math.floor(dh / 2)).map(p => `pair ${p + 1} by ${M.fmt(deg(p), deg(p) < 1 ? 2 : 1)}°`).join(', ');
      stage({
        key: `rope${l}`, title: 'Rotate q, k (RoPE)', layer: lid(l), part: 'scores', rows: rows('rq', 'rk'),
        tex: `\\tilde q_t = R(t\\,\\theta)\\,q_t,\\;\\; \\tilde k_t = R(t\\,\\theta)\\,k_t`,
        text: `RoPE turns every pair of a head's columns of q and k by an angle that grows with the position, ${angles} per position (position 1 is not turned). `
          + 'A turned q against a turned k sees only the difference of their angles, so the scores depend on how far apart two words are, not where they are. No position vector was added to X.',
      });
    }
    // Linear attention: every entry of q and k through phi = elu + 1, so the scores are positive
    if (lin) {
      for (let h = 0; h < H; h++) {
        for (const [G, key, srcT] of [['Q', 'Qf', rope ? hid('rq', h) : tq(h)], ['K', 'Kf', rope ? hid('rk', h) : tk(h)]]) {
          const pre = () => (rope ? hs(h)?.[`${G}r`] : hs(h)?.[G]), out = () => hs(h)?.[key], s = G.toLowerCase();
          tile(hid(`f${s}`, h), {
            l, head: H > 1 ? h : null, rows: n, cols: dh, tex: `\\phi(${G}${sub(h)})`, v: out() || grid(n, dh, () => NaN), rowLab: lab,
            src: (i, f) => [[srcT, i, f]],
            tip: (i, f) => {
              const x = pre()[i][f];
              return `φ(${s}${hsub(h)}[${i + 1},${f + 1}]) = ${x > 0 ? `${f2(x)} + 1` : `e<sup>${f2(x)}</sup>`} = ${f2(out()[i][f])}<br>φ(x) = x + 1 above 0, e<sup>x</sup> below: always positive`;
            },
          });
        }
      }
      stage({
        key: `phi${l}`, title: 'Feature map φ', layer: lid(l), part: 'scores', rows: rows('fq', 'fk'),
        tex: '\\phi(x) = \\operatorname{elu}(x) + 1',
        text: 'Linear attention has no softmax. Instead φ makes every entry of q and k positive, so every score φ(q)·φ(k) is positive too and the weights can be the scores over their sum.',
      });
    }
    const dq = h => (lin ? hid('fq', h) : rope ? hid('rq', h) : tq(h)), dkT = h => (lin ? hid('fk', h) : rope ? hid('rk', h) : tk(h));
    const qOf = h => (lin ? hs(h).Qf : rope ? hs(h).Qr : hs(h).Q), kOf = h => (lin ? hs(h).Kf : rope ? hs(h).Kr : hs(h).K);
    const qs = lin ? 'φ(q' : rope ? 'q̃' : 'q', ks = lin ? 'φ(k' : rope ? 'k̃' : 'k', cl = lin ? ')' : '';
    const why = (i, j) => (j > i && g.causal ? `can't see the later ${posName(j)}` : g.window && Math.abs(i - j) >= g.window
      ? `can't see ${posName(j)}: it is ${Math.abs(i - j)} away, outside the window of ${g.window}` : `can't see ${posName(j)}`);
    const slope = h => M.alibiSlope(h), frac = v => (v === 0.5 ? '½' : v === 0.25 ? '¼' : M.fmt(v, 3));
    for (let h = 0; h < H; h++) {
      const off = offs.has(h) && H > 1;
      const dot = (i, j) => qOf(h)[i].reduce((s, x, f) => s + x * kOf(h)[j][f], 0);
      if (alibi) {
        tile(hid('qk', h), {
          l, head: H > 1 ? h : null, rows: n, cols: n, tex: `Q${sub(h)}K${sub(h)}^{\\top}\\!/\\sqrt{d}`, mask, rowLab: lab, colLab: keys, off,
          v: hs(h) ? grid(n, n, (i, j) => (mask[i][j] ? -Infinity : dot(i, j) * g.scale)) : grid(n, n, () => NaN),
          src: (i, j) => (mask[i][j] ? [] : [...rowSrc(tq(h), i), ...rowSrc(tk(h), j)]),
          tip: (i, j) => (mask[i][j] ? `masked: ${posName(i)} ${why(i, j)}` : `q<sub>${i + 1}</sub>·k<sub>${j + 1}</sub> × ${f2(g.scale)} = ${f2(dot(i, j) * g.scale)}: the content part of the score`),
        });
        tile(hid('ab', h), {
          l, head: H > 1 ? h : null, rows: n, cols: n, tex: `B${sub(h)}`, v: hs(h)?.B || grid(n, n, () => NaN), mask, rowLab: lab, colLab: keys, off, scale: 'bias',
          tip: (i, j) => `B${hsub(h)}[${i + 1},${j + 1}] = −${frac(slope(h))} × ${Math.abs(i - j)} = ${f2(hs(h).B[i][j])}<br>head ${h + 1}'s slope ${frac(slope(h))} times how far apart ${posName(i)} and ${posName(j)} are: fixed, never trained`,
        });
      }
      tile(hid('s', h), {
        l, head: H > 1 ? h : null, rows: n, cols: n, tex: `S${sub(h)}`, v: hs(h)?.S || grid(n, n, () => NaN), mask, rowLab: lab, colLab: keys, off, maskBlank: lin,
        src: (i, j) => (mask[i][j] ? [] : alibi ? [[hid('qk', h), i, j], [hid('ab', h), i, j]] : [...rowSrc(dq(h), i), ...rowSrc(dkT(h), j)]),
        tip: (i, j) => {
          if (mask[i][j]) return `${at('S', h, i, j)} = −∞: ${posName(i)} ${why(i, j)}`;
          if (alibi) return `${at('S', h, i, j)} = q<sub>${i + 1}</sub>·k<sub>${j + 1}</sub> × ${f2(g.scale)} + B${hsub(h)}[${i + 1},${j + 1}]<br>= ${f2(dot(i, j) * g.scale)} + (${f2(hs(h).B[i][j])}) = ${f2(hs(h).S[i][j])}`;
          const q = qOf(h)[i], k = kOf(h)[j], sc = lin ? '' : ` × ${f2(g.scale)}`;
          return `${at('S', h, i, j)} = ${qs}<sub>${i + 1}</sub>${cl}·${ks}<sub>${j + 1}</sub>${cl}${sc}`
            + `<br>= (${q.map((x, f) => `${f2(x)}·${f2(k[f])}`).join(' + ')})${sc} = ${f2(hs(h).S[i][j])}`;
        },
      });
      tile(hid('a', h), {
        l, head: H > 1 ? h : null, rows: n, cols: n, tex: `A${sub(h)}`, v: hs(h)?.A || grid(n, n, () => NaN), mask, rowLab: lab, colLab: keys, scale: 'attn', off,
        src: (i, j) => (mask[i][j] ? [] : rowSrc(hid('s', h), i).filter(([, , jj]) => !mask[i][jj])),
        tip: (i, j) => (mask[i][j] ? `${at('A', h, i, j)} = 0 (masked)`
          : lin ? `${at('A', h, i, j)} = S<sub>${i + 1},${j + 1}</sub> / Σ<sub>k</sub> S<sub>${i + 1},k</sub> = ${f2(hs(h).S[i][j])} / ${f2(hs(h).S[i].reduce((s, v) => s + (v === -Infinity ? 0 : v), 0))} = ${f2(hs(h).A[i][j])}`
            + `<br>how much ${posName(i)} reads ${posName(j)}`
          : `${at('A', h, i, j)} = e<sup>S<sub>${i + 1},${j + 1}</sub></sup> / Σ<sub>k</sub> e<sup>S<sub>${i + 1},k</sub></sup> = ${f2(hs(h).A[i][j])}`
            + `<br>how much ${posName(i)} reads ${posName(j)}`),
      });
      const zt = hid('z', h), zAt = (i, f) => i * d + h * dh + f;
      tile(zt, {
        l, head: H > 1 ? h : null, rows: n, cols: dh, tex: `Z${sub(h)}`, v: hs(h)?.Z || grid(n, dh, () => NaN), off,
        node: (i, f) => ns[l][zAt(i, f)]?.id ?? null,
        src: (i, f) => [...rowSrc(hid('a', h), i).filter(([, , j]) => !mask[i][j]), ...colSrc(tv(h), f).filter(([, j]) => !mask[i][j])],
        tip: (i, f) => `${at('Z', h, i, f)} = Σ<sub>j</sub> A<sub>${i + 1},j</sub> V<sub>j,${f + 1}</sub><br>= `
          + hs(h).A[i].map((w, j) => (mask[i][j] ? null : `${f2(w)}·${f2(hs(h).V[j][f])}`)).filter(Boolean).join(' + ')
          + ` = ${f2(hs(h).Z[i][f])}${off ? '<br>head off: its columns of the concat are 0' : ''}`,
      });
      if (H === 1) for (let i = 0; i < n; i++) for (let f = 0; f < dh; f++) { const nid = ns[l][zAt(i, f)]?.id; if (nid) cellOfNode(nid, [zt, i, f]); }
    }
    const sc = H > 1 ? '_h' : '', dk = H > 1 ? 'd_h' : 'd_k';
    const masked = g.causal || !!g.window;
    // the mask's shape: causal (a triangle), a window (a band), or none
    const maskText = g.causal && g.window ? ` Causal, window ${g.window}: a position sees itself and the ${g.window - 1} before it, nothing older or later (−∞).`
      : g.window ? ` Window ${g.window}: a position sees the positions less than ${g.window} away, on both sides (−∞ elsewhere).`
        : g.causal ? ' Causal: a position sees itself and the positions before it, never after (−∞).'
          : oneHot && n > 1 ? ' No mask: every position sees every other, the later ones too, so each can read the very word it is trained to predict next.' : '';
    let sTex = `S${sc} = Q${sc} K${sc}^{\\top} / \\sqrt{${dk}}${masked ? ' + M' : ''}`;
    let sText = `Every query against every key, times 1/√${dh} = ${f2(g.scale)}.${maskText}`;
    if (rope) {
      sTex = `S${sc} = \\tilde Q${sc} \\tilde K${sc}^{\\top} / \\sqrt{${dk}}${masked ? ' + M' : ''}`;
      sText = `Every turned query against every turned key, times 1/√${dh} = ${f2(g.scale)}.${maskText}`;
    } else if (lin) {
      sTex = `S${sc} = \\phi(Q${sc})\\,\\phi(K${sc})^{\\top}`;
      sText = `Every φ(q) against every φ(k): all positive, and no scale (the weights divide it out).${maskText.replace(' (−∞).', ' (left out).').replace(' (−∞ elsewhere).', ' (left out elsewhere).')}`;
    } else if (alibi) {
      sTex = `S${sc} = Q${sc} K${sc}^{\\top} / \\sqrt{${dk}} + B${sc}${masked ? ' + M' : ''}`;
      sText = `The content scores plus B${H > 1 ? '<sub>h</sub>' : ''}, ALiBi's fixed penalty: ${range(H).map(h => `head ${h + 1} loses ${frac(slope(h))}`).join(', ')} per position of distance, `
        + `so each head starts out favouring nearby words, head ${H} the least. No position vector was added to X: this is where position comes in.${maskText}`;
    }
    stage({
      key: `scores${l}`, title: 'Scores', layer: lid(l), part: 'scores', rows: alibi ? rows('qk', 'ab', 's').map(r => ({ ...r, ops: ['+', '='] })) : rows('s'),
      tex: sTex, text: sText,
    });
    const last = n - 1, peek = h => (hs(h) ? argmax(hs(h).A[last]) : 0);
    const reads = Array.from({ length: H }, (_, h) => `${posName(peek(h))} most${H > 1 ? ` in head ${h + 1}` : ''} (${f2(hs(h)?.A[last][peek(h)])})`).join(', ');
    stage({
      key: `softmax${l}`, title: lin ? 'Weights A' : 'Attention A', layer: lid(l), part: 'softmax', rows: rows('a'),
      tex: lin ? `A${sc} = S${sc} \\,/\\, \\textstyle\\sum_j S${sc}[:, j]` : `A${sc} = \\operatorname{softmax}(S${sc})`,
      text: lin ? `Each row of S over its sum, so it sums to 1 as softmax rows do, but with no exponential to sharpen them the weights stay flatter. ${posName(last)} reads ${reads}. `
          + 'With no softmax, Z = φ(Q)(φ(K)ᵀV) can be summed right to left: a d_h × d_h running state instead of the n × n matrix, which is what makes linear attention cheap on long inputs.'
        : `Each row sums to 1: where that position looks. ${posName(last)} reads ${reads}.`,
    });
    stage({
      key: `mix${l}`, title: 'A V', layer: lid(l), part: 'mix', rows: rows('z'),
      tex: `Z${sc} = A${sc} V${sc}`,
      text: 'Each position\'s row of A times V: a weighted mix of the values it looked at.',
    });
    if (H > 1) {
      const id = layerTile(l, null, {
        id: `L${l}`, tex: 'Z', vec: V.a[l], colHeads: Array.from({ length: d }, (_, c) => Math.floor(c / dh)),
        src: (i, c) => [[hid('z', Math.floor(c / dh)), i, c % dh]],
        tip: (i, c) => {
          const h = Math.floor(c / dh);
          return `Z<sub>${i + 1},${c + 1}</sub> = Z<sub>${h + 1}</sub>[${i + 1},${(c % dh) + 1}] = ${f2(V.a[l][i * d + c])}${offs.has(h) ? ` (head ${h + 1} is off: 0)` : ''}`;
        },
      });
      F.tiles[id].offCols = Array.from({ length: d }, (_, c) => offs.has(Math.floor(c / dh)));
      const on = Array.from({ length: H }, (_, h) => h).filter(h => !offs.has(h));
      stage({
        key: `concat${l}`, title: 'Concat', layer: lid(l), part: null, rows: [{ head: null, tiles: [id], ops: [] }],
        tex: `Z = [\\,${Array.from({ length: H }, (_, h) => `Z_{${h + 1}}`).join('\\;')}\\,]`,
        text: on.length === H ? `The ${H} heads side by side, back to d = ${d} columns.`
          : `Heads ${[...offs].filter(h => h < H).map(h => h + 1).join(', ')} knocked out: their columns are 0, and every later stage is recomputed without them.`,
      });
    } else {
      (layerTileOf[l] ||= [])[0] = hid('z', 0);
    }
  }

  // ---- dense layers: tokenwise products and sums, else one tile per layer
  // A layer's activation as a function name in KaTeX (post-norm's LN included), and its norm line for a tip.
  function fnTexOf(act) {
    return act === 'identity' ? '' : act === 'softmax' ? '\\operatorname{softmax}' : act === 'layernorm' ? '\\operatorname{LN}'
      : `\\operatorname{${M.ACTS[act]?.label || act}}`;
  }
  function isNorm(act) { return act === 'layernorm' || act === 'rmsnorm'; }
  // row t of layer l normalized (LayerNorm / RMSNorm): its mean, its sigma and cell j's arithmetic
  function normLine(l, t, j) {
    const { d } = shp[l], act = actOf(l), z = V.z[l].slice(t * d, (t + 1) * d);
    const mu = act === 'layernorm' ? z.reduce((s, v) => s + v, 0) / d : 0;
    const sg = Math.sqrt(z.reduce((s, v) => s + (v - mu) ** 2, 0) / d + M.NORM_EPS), a = V.a[l][t * d + j];
    return act === 'layernorm'
      ? `= (${f2(z[j])} − μ) / σ = (${f2(z[j])} − ${f2(mu)}) / ${f2(sg)} = ${f2(a)}<br>μ = the row's mean, σ = √(its variance + ε): each row comes out with mean 0, spread 1`
      : `= ${f2(z[j])} / rms = ${f2(z[j])} / ${f2(sg)} = ${f2(a)}<br>rms = √(mean of the row's squares + ε): no mean taken out`;
  }
  // a matrix's name, transposed when a shared matrix is read the other way (tied embeddings: W_Eᵀ)
  function mname(m) { return m.transposed ? `${m.name}^{\\top}` : m.name; }
  function denseStages(l) {
    const { tokens: n, d, groups } = shp[l], nodes = ns[l], act = actOf(l);
    const tm = M.tiedMatrices(net, l);
    const fixed = [];
    for (const nd of nodes) for (const e of inEdges.get(nd.id) || []) if (e.fixed) fixed.push(e);
    // A fixed residual: every fixed edge copies slot (t, f) of one earlier layer with weight 1.
    const resK = new Set();
    let resOk = true;
    const slot = id => { const [ll, k] = nodeIdx.get(id) || []; return ll === undefined ? null : { l: ll, t: Math.floor(k / shp[ll].d) % shp[ll].tokens, f: k % shp[ll].d }; };
    for (const e of fixed) {
      const a = slot(e.from), b = slot(e.to);
      if (!a || !b || e.w !== 1 || a.t !== b.t || a.f !== b.f || shp[a.l].groups || shp[a.l].d !== d) { resOk = false; break; }
      resK.add(a.l);
    }
    if (resOk) for (const k of resK) if (fixed.filter(e => slot(e.from).l === k).length !== n * d) resOk = false;
    const tokenwise = !groups && tm.length > 0 && tm.every(m => m.tokenwise && m.k !== null && shp[m.k].tokens === n) && resOk;
    const title = String(net.layers[l].name || `Layer ${l}`);
    // biases fixed (node.fixed): at 0, there is no bias term at all; else a fixed P (sinusoidal)
    const fixedB = nodes.length > 0 && nodes.every(nd => nd.fixed === true), noB = fixedB && nodes.every(nd => !nd.bias);
    // A pre-norm layer: a fixed identity copy of one earlier layer, normalized per position.
    if (!groups && isNorm(act) && !tm.length && resOk && resK.size === 1 && noB) { normStage(l, [...resK][0]); return; }
    // SwiGLU: groups G, U from one source, silu(G) ⊙ U.
    if (act === 'swiglu' && groups?.length === 2 && !fixed.length && tm.length === 2
      && tm.every(m => m.tokenwise && m.k !== null && m.k === tm[0].k && shp[m.k].tokens === n) && new Set(tm.map(m => m.toGroup)).size === 2) {
      gluStages(l, tm);
      return;
    }
    if (!tokenwise) {
      const ids = (groups || [null]).map((_, g) => layerTile(l, groups ? g : null, { rowLab: n > 1 ? F.labels.slice(0, n) : null }));
      const fn = fnTexOf(act);
      stage({
        key: `layer${l}`, title, layer: lid(l), rows: [{ head: null, tiles: ids, ops: [] }],
        tex: `${symOf(net, l)} = ${fn}${fn ? '(' : ''}W a + b${fn ? ')' : ''}`,
        text: `${nodes.length} neuron${nodes.length === 1 ? '' : 's'}: each sums its weighted inputs and its bias${fn ? ', then the activation' : ''}. Hover one to see its sum.`,
      });
      return;
    }
    const sym = symOf(net, l), bias = tieBase(nodes[0]?.tie);
    const untied = n > 1 && !fixedB && nodes.every(nd => !nd.tie);   // a bias per position: P
    const fixedP = n > 1 && fixedB && !noB;                            // a fixed P (sinusoidal positions)
    // products: the source rows times each shared matrix
    const prods = tm.map((m, mi) => {
      const src = (M.reshape(net, m.k, V.a[m.k])[m.fromGroup || 'X']);
      const rows = src.length, dout = m.W[0]?.length || 0;
      const v = grid(rows, d, (t, j) => (j < dout ? src[t].reduce((s, x, i) => s + x * (m.W[i]?.[j] ?? 0), 0) : 0));
      // SwiGLU's gated half is F, whatever its group is called
      const ssym = actOf(m.k) === 'swiglu' && m.fromGroup && m.fromGroup === shp[m.k].groups?.[0] ? 'F' : m.fromGroup || symOf(net, m.k);
      const id = `P${l}.${mi}`, onehotSrc = m.k === 0 && oneHot, nm = mname(m);
      tile(id, {
        l, rows, cols: d, v, tex: `${oneHot && m.k === 0 ? 'O' : ssym}\\,${nm}`,
        src: t => rowCells(m.k, m.fromGroup, t),
        tip: (t, j) => (onehotSrc
          ? `(O ${texHtml(nm)})<sub>${t + 1},${j + 1}</sub> = ${texHtml(nm)}[${esc(F.words[t])}, ${j + 1}] = ${f2(v[t][j])}<br>a one-hot row picks one row of ${texHtml(nm)}`
          : `(${texHtml(ssym)} ${texHtml(nm)})<sub>${t + 1},${j + 1}</sub> = ${texHtml(ssym)}<sub>${t + 1}</sub> · ${texHtml(nm)}[:,${j + 1}]<br>= `
            + `${src[t].map((x, i) => `${f2(x)}·${f2(m.W[i]?.[j] ?? 0)}`).join(' + ')} = ${f2(v[t][j])}`),
      });
      return { id, m, ssym, nm };
    });
    const res = [...resK].map(k => ({ k, id: layerTileOf[k]?.[0], sym: symOf(net, k) }));
    const outTip = (t, j) => {
      const parts = [];
      for (const r of res) parts.push(`${texHtml(r.sym)} ${f2(V.a[r.k][t * d + j])}`);
      for (const p of prods) parts.push(`${texHtml(p.m.k === 0 && oneHot ? 'O' : p.ssym)}·${texHtml(p.nm)} ${f2(F.tiles[p.id].v[t][j])}`);
      const b = nodes[t * d + j]?.bias || 0;
      if (!noB) parts.push(`${untied || fixedP ? 'P' : bias ? texHtml(bias) : 'b'} ${f2(b)}`);
      const fn = act === 'identity' ? '' : act === 'softmax' ? 'softmax' : act === 'layernorm' ? 'LN' : M.ACTS[act]?.label || act;
      const z = V.z[l]?.[t * d + j], a = V.a[l]?.[t * d + j];
      return `${cellName(sym, t, j)} = ${fn ? `${fn}(` : ''}${parts.join(' + ')}${fn ? ')' : ''}`
        + (isNorm(act) ? `<br>${normLine(l, t, j)}` : `<br>= ${fn && act !== 'softmax' ? `${fn}(${f2(z)}) = ` : ''}${f2(a)}`);
    };
    const outSrc = (t, j) => [
      ...res.map(r => (r.id ? [r.id, t, j] : null)).filter(Boolean),
      ...prods.map(p => [p.id, t, j]),
      ...(untied || fixedP ? [[`B${l}`, t, j]] : []),
    ];
    const fnTex = fnTexOf(act);
    const sumTex = [...res.map(r => r.sym), ...prods.map(p => `${oneHot && p.m.k === 0 ? 'O' : p.ssym}\\,${p.nm}`), ...(noB ? [] : [untied || fixedP ? 'P' : bias || 'b'])].join(' + ');
    if (res.length) {
      stage({
        key: `proj${l}`, title: prods.map(p => `· ${texHtml(p.nm)}`).join(', '), layer: lid(l), part: null,
        rows: [{ head: null, tiles: prods.map(p => p.id), ops: [] }],
        tex: prods.map(p => `${p.ssym}\\,${p.nm}`).join(',\\;'),
        text: prods.map(p => `${texHtml(p.ssym)} times ${texHtml(p.nm)}`).join(', ')
          + (prods.some(p => p.ssym === 'Z') ? ': the heads\' outputs mixed back together' : ': the branch the residual adds')
          + `, brought back to ${d} columns so it can be added.`,
      });
      const id = layerTile(l, null, { src: outSrc, tip: outTip });
      // the branch a residual adds, by what it is: attention (its heads) or the FFN
      const branch = p => (att[p.m.k] ? 'attention' : ['relu', 'gelu', 'swiglu'].includes(actOf(p.m.k)) && shp[p.m.k].d > d ? 'FFN' : `${texHtml(p.ssym)} ${texHtml(p.nm)}`);
      F.residual.push({ l, sym: texHtml(sym), from: res.map(r => texHtml(r.sym)), branch: prods.map(branch), d, tiles: [id, ...prods.map(p => p.id)], src: res.map(r => r.id).filter(Boolean) });
      const who = res.map(r => texHtml(r.sym)).join(', ');
      stage({
        key: `sum${l}`, title: isNorm(act) ? `+ residual, ${M.ACTS[act].label}` : '+ residual', layer: lid(l), rows: [{ head: null, tiles: [id], ops: [] }],
        tex: `${sym} = ${fnTex}${fnTex ? '(' : ''}${sumTex}${fnTex ? ')' : ''}`,
        text: `The residual: ${who} comes through unchanged and ${prods.map(p => `${texHtml(p.ssym)} ${texHtml(p.nm)}`).join(', ')} is added${bias ? `, plus ${texHtml(bias)}` : ''}. `
          + `The sum goes cell by cell, so ${texHtml(sym)} keeps ${who}'s ${d} columns: that is why the residual stream has one width.`
          + (isNorm(act) ? ` Then each row is normalized (post-norm): ${act === 'layernorm' ? 'its mean taken out and divided by its spread' : 'divided by its root mean square'}, so the stream is rescaled after every branch.` : ''),
      });
      return;
    }
    if (untied || fixedP) {
      const bid = `B${l}`;
      tile(bid, {
        l, rows: n, cols: d, tex: 'P', v: grid(n, d, (t, j) => nodes[t * d + j]?.bias || 0),
        tip: (t, j) => (fixedP
          ? `P<sub>${t + 1},${j + 1}</sub> = ${j % 2 ? 'cos' : 'sin'}(${t} / 10000<sup>${2 * Math.floor(j / 2)}/${d}</sup>) = ${f2(nodes[t * d + j]?.bias || 0)}: fixed, the same in every sentence and never trained`
          : `P<sub>${t + 1},${j + 1}</sub> = ${f2(nodes[t * d + j]?.bias || 0)}: the learned vector of position ${t + 1} (the layer's own biases)`),
      });
      const id = layerTile(l, null, { src: outSrc, tip: outTip });
      stage({
        key: `embed${l}`, title: oneHot && prods.some(p => p.m.k === 0) ? `Embed + position${fixedP ? ' (fixed)' : ''}` : `${texHtml(sym)} + position`, layer: lid(l),
        rows: [{ head: null, tiles: [...prods.map(p => p.id), bid, id], ops: [...prods.map((_, i) => (i ? '+' : '')).slice(1), '+', '='] }],
        tex: `${sym} = ${fnTex}${fnTex ? '(' : ''}${sumTex}${fnTex ? ')' : ''}`,
        text: fixedP ? `Each one-hot row picks its word's row of the embedding; P adds a fixed vector per position: sin and cos of the position at ${Math.ceil(d / 2)} frequencies (1, 1/10, 1/100, ...), one per pair of columns. Nothing learns it: every sentence gets the same P.`
          : oneHot ? 'Each one-hot row picks its word\'s row of the embedding; P adds a learned vector for each position.'
            : `The shared product plus P, a learned vector for each position.`,
      });
      return;
    }
    // One product and nothing added: the layer's own tile is enough, traced to the rows it reads.
    const single = prods.length === 1 ? prods[0] : null;
    for (const p of prods) delete F.tiles[p.id];
    const rowsIn = t => prods.flatMap(p => rowCells(p.m.k, p.m.fromGroup, t));
    if (act === 'softmax') {
      const lg = `G${l}`;
      const vocOut = !!voc && d === voc.length;
      // A language model's head: a softmax over the words at every position. Generating reads only
      // the last position's row, so that is all the ending shows unless `every` asks for the rest.
      const lm = vocOut && l === L - 1 && n > 1;
      if (lm) F.lm = true;
      const pos = lm && !F.every ? [n - 1] : range(n), one = pos.length === 1 && n > 1;
      const at = (i, j) => pos[i] * d + j;
      const pSub = one ? `_{${n}}` : '';
      const ctxOf = t => seen(t) ?? tokenLabel(net, t);
      const src0 = single ? layerTileOf[single.m.k]?.[single.m.fromGroup && shp[single.m.k].groups ? shp[single.m.k].groups.indexOf(single.m.fromGroup) : 0] : null;
      tile(lg, {
        l, rows: pos.length, cols: d, pos, tex: `\\ell${pSub}`, v: pos.map(t => V.z[l].slice(t * d, (t + 1) * d)), colLab: vocOut ? voc : null,
        src: i => rowsIn(pos[i]),
        tip: (i, j) => `ℓ<sub>${pos[i] + 1},${j + 1}</sub>${vocOut ? ` (${esc(voc[j])})` : ''} = ${nodeSum(nodes[at(i, j)]?.id) || f2(V.z[l][at(i, j)])} = ${f2(V.z[l][at(i, j)])}`,
      });
      // only the last row of the layer before goes on: its other rows fade
      if (one && src0 && F.tiles[src0]?.rows === n) F.tiles[src0].dimRows = range(n - 1);
      const ysym = single ? single.ssym : 'Y', yrow = ysym.includes('_') ? `(${ysym.toLowerCase()})` : ysym.toLowerCase();   // (n_{3})_{5}: row 5 of N₃
      const tied = !!single?.m.transposed && vocOut;
      stage({
        key: `logits${l}`, title: 'Logits', layer: lid(l), rows: [{ head: null, tiles: [lg], ops: [] }],
        tex: one ? `\\ell_{${n}} = ${yrow}_{${n}}\\,${single ? single.nm : 'W'} + ${bias || 'b'}` : `\\ell = ${prods.map(p => `${p.ssym}\\,${p.nm}`).join(' + ')} + ${bias || 'b'}`,
        text: (one ? `One score per word, from the last position's row of ${texHtml(ysym)} only (position ${n}, “${esc(F.labels[n - 1])}”): the row that predicts the next word.`
          : vocOut ? 'One score per word, at each position.' : 'One score per output, before the softmax.')
          + (tied ? ` Tied embeddings: the matrix is ${texHtml(single.m.name)} itself, turned around, the one that embedded the words, so a word's score is the row times that word's embedding.` : ''),
      });
      const targets = nodes.map(nd => nd.target);
      const hasT = targets.every(isNum);
      const tgt = hasT ? range(n).map(t => {
        const row = targets.slice(t * d, (t + 1) * d), k = argmax(row);
        return row[k] >= 0.5 ? k : null;
      }) : null;
      const id = `L${l}`;
      const probsOf = t => V.a[l].slice(t * d, (t + 1) * d);
      tile(id, {
        l, rows: pos.length, cols: d, pos, tex: `p${pSub}`, kind: lm ? (one ? 'dist' : 'bars') : vocOut ? 'bars' : 'mat', scale: 'prob',
        v: pos.map(probsOf), colLab: vocOut ? voc : null, target: tgt ? pos.map(t => tgt[t]) : null,
        rowLab: vocOut ? pos.map(ctxOf) : null,
        note: lm && !one ? `Training scores every position against its true next word, and the loss is the mean over the ${n}; generating reads only the last one.` : null,
        node: (i, j) => ns[l][at(i, j)]?.id ?? null,
        src: i => rowSrc(lg, i),
        tip: (i, j) => {
          const t = pos[i], p = V.a[l][at(i, j)], top = argmax(probsOf(t)) === j;
          return `p(${vocOut ? esc(voc[j]) : `${j + 1}`} | ${seen(t) ? esc(seen(t)) : `position ${t + 1}`}) = e<sup>ℓ</sup> / Σ e<sup>ℓ</sup> = ${f2(p)}`
            + `${top ? '<br>the most likely word' : ''}${tgt?.[t] === j ? `<br>the true next word${F.words.every(Boolean) && t < n - 1 ? ` (position ${t + 2})` : ''}` : ''}`;
        },
      });
      for (let i = 0; i < pos.length; i++) for (let j = 0; j < d; j++) { const nid = ns[l][at(i, j)]?.id; if (nid) cellOfNode(nid, [id, i, j]); }
      (layerTileOf[l] ||= [])[0] = id;
      if (vocOut && l === L - 1) {
        const t = n - 1, row = probsOf(t), k = argmax(row);
        F.next = { word: voc[k], p: row[k], target: tgt?.[t] !== null && tgt?.[t] !== undefined ? voc[tgt[t]] : null, t };
      }
      const nx = F.next;
      const upTo = causalAll ? 'given the words up to it (causal)' : 'given the whole sentence (no mask: the later words too)';
      stage({
        key: `probs${l}`, title: lm ? (one ? 'Next word' : 'Next word, every position') : vocOut ? 'Next word' : 'Softmax', layer: lid(l), mode: lm,
        rows: [{ head: null, tiles: [id], ops: [] }],
        tex: one ? `p_{${n}} = \\operatorname{softmax}(\\ell_{${n}})` : 'p = \\operatorname{softmax}(\\ell)',
        text: one ? `The next word's distribution, after “${esc(ctxOf(n - 1))}”: ${esc(nx.word)} gets ${f2(nx.p)}${nx.target ? `, and the true next word is ${esc(nx.target)}` : ''}. `
            + 'Generating reads only this last row; every position (how it\'s trained) shows the rest.'
          : lm ? `Every position predicts its own next word, ${upTo}: the whole sentence trains at once. The loss is the mean over the ${n} positions of −log p(true next word); generating reads only the last row.`
            : vocOut ? `A probability for every word at each position, ${upTo}. The ringed bar is the target.`
              : 'Each row turned into probabilities that sum to 1.',
      });
      return;
    }
    const id = layerTile(l, null, { src: t => rowsIn(t), tip: (t, j) => nodeTip(nodes[t * d + j]?.id, cellName(sym, t, j)) });
    const widens = (act === 'relu' || act === 'gelu') && !!single && d > shp[single.m.k].d;   // the FFN: d -> 4d
    const embeds = noB && oneHot && !!single && single.m.k === 0;   // X = O W_E, no position vector
    const how = single ? `${texHtml(single.ssym)} ${texHtml(single.nm)}` : 'the products';
    stage({
      key: `layer${l}`, title: widens ? 'FFN' : embeds ? 'Embed' : title, layer: lid(l),
      rows: [{ head: null, tiles: [id], ops: [] }],
      tex: `${sym} = ${fnTex}${fnTex ? '(' : ''}${sumTex}${fnTex ? ')' : ''}`,
      text: act === 'relu'
        ? `The feed-forward layer, at each position on its own: ${how} + ${bias ? texHtml(bias) : 'b'}, then ReLU (${V.a[l].filter(v => v > 0).length} of ${n * d} units are on).`
          + (widens ? ` It is wider than the stream, ${d} columns from ${shp[single.m.k].d}: room to compute in, and the next matrix brings it back to ${shp[single.m.k].d}.` : '')
        : act === 'gelu'
          ? `The feed-forward layer, at each position on its own: ${how} + ${bias ? texHtml(bias) : 'b'}, then GELU, z·Φ(z): smooth, and a little below 0 for negative z, where ReLU would give exactly 0 (${V.a[l].filter(v => v < 0).length} of ${n * d} units are slightly negative, so none is off).`
            + (widens ? ` It is wider than the stream, ${d} columns from ${shp[single.m.k].d}; the next matrix brings it back.` : '')
          : embeds ? `Each one-hot row picks its word's row of the embedding, and nothing marks its position: ${posHow}`
            : `${sumTex.includes('+') ? 'The products and the bias' : noB ? 'The product' : 'The product and the bias'}${act === 'identity' ? '' : ', then the activation'}, at each position.`,
    });
    if (widens) F.ffn = { id, d, from: shp[single.m.k].d };
  }

  // A pre-norm layer: row t of layer k normalized on its own (LayerNorm or RMSNorm, no gain or shift).
  function normStage(l, k) {
    const act = actOf(l), rms = act === 'rmsnorm', sym = symOf(net, l), from = symOf(net, k);
    const id = layerTile(l, null, { src: t => rowCells(k, null, t), tip: (t, j) => `${cellName(sym, t, j)} ${normLine(l, t, j)}` });
    F.norms.push(id);
    stage({
      key: `norm${l}`, title: rms ? 'RMSNorm' : 'LayerNorm', layer: lid(l), rows: [{ head: null, tiles: [id], ops: [] }],
      tex: rms ? `${sym} = ${from} \\,/\\, \\operatorname{rms}(${from})` : `${sym} = (${from} - \\mu) \\,/\\, \\sigma`,
      text: (rms ? `Each position's row of ${texHtml(from)} divided by its root mean square, √(mean(x²) + ε), with no mean taken out: cheaper than LayerNorm, and it trains about as well.`
        : `Each position's row of ${texHtml(from)} on its own: its mean μ taken out, then divided by its spread σ, so every row of ${texHtml(sym)} has mean 0 and spread 1 whatever the size of the stream.`)
        + ` Pre-norm: only the branch reads ${texHtml(sym)}; the residual stream ${texHtml(from)} goes on unnormalized. No learned gain or shift here: right before a matrix they would fold into it.`,
    });
  }

  // SwiGLU: G = H W_1 + b_1 and U = H W_3 + b_3 (the layer's groups), then F = silu(G) ⊙ U (group 1's activations).
  function gluStages(l, tm) {
    const { tokens: n, d, groups } = shp[l], k = tm[0].k, from = symOf(net, k), off = n * d;
    const mG = tm.find(m => m.toGroup === groups[0]), mU = tm.find(m => m.toGroup === groups[1]);
    const gid = `gate${l}`, zg = (t, j) => V.z[l][t * d + j];
    const bOf = g => tieBase(ns[l][g * off]?.tie) || `b_{${groups[g]}}`;
    tile(gid, {
      l, rows: n, cols: d, tex: 'G', v: grid(n, d, zg), src: t => rowCells(k, mG.fromGroup, t),
      tip: (t, j) => `G<sub>${t + 1},${j + 1}</sub> = ${texHtml(from)}<sub>${t + 1}</sub> · ${texHtml(mG.name)}[:,${j + 1}] + ${texHtml(bOf(0))} = ${f2(zg(t, j))}: the gate`,
    });
    const uid = layerTile(l, 1, {
      tex: 'U', src: t => rowCells(k, mU.fromGroup, t),
      tip: (t, j) => `U<sub>${t + 1},${j + 1}</sub> = ${texHtml(from)}<sub>${t + 1}</sub> · ${texHtml(mU.name)}[:,${j + 1}] + ${texHtml(bOf(1))} = ${f2(V.a[l][off + t * d + j])}: the value the gate lets through`,
    });
    stage({
      key: `gate${l}`, title: 'Gate and up', layer: lid(l), rows: [{ head: null, tiles: [gid, uid], ops: [] }],
      tex: `G = ${from}\\,${mG.name} + ${bOf(0)},\\;\\; U = ${from}\\,${mU.name} + ${bOf(1)}`,
      text: `Two matrices up from ${texHtml(from)}, both to ${d} columns: G will gate, U carries the values. A ReLU FFN has one matrix up; SwiGLU's second one is its 50% more weights.`,
    });
    const silu = v => v / (1 + Math.exp(-v));
    const fid = layerTile(l, 0, {
      tex: 'F', src: (t, j) => [[gid, t, j], [uid, t, j]],
      tip: (t, j) => {
        const g = zg(t, j), u = V.a[l][off + t * d + j];
        return `F<sub>${t + 1},${j + 1}</sub> = silu(G) · U = silu(${f2(g)}) · ${f2(u)} = ${f2(silu(g))} · ${f2(u)} = ${f2(V.a[l][t * d + j])}<br>silu(g) = g·σ(g): near 0 for negative g (the gate shut), near g for positive`;
      },
    });
    const open = range(n * d).filter(i => silu(V.z[l][i]) > 0.5).length;
    stage({
      key: `glu${l}`, title: 'SwiGLU', layer: lid(l), rows: [{ head: null, tiles: [fid], ops: [] }],
      tex: 'F = \\operatorname{silu}(G) \\odot U',
      text: `Each unit of U times its gate silu(G), cell by cell: the gate depends on the input too, so the FFN multiplies two functions of it where ReLU only cuts one (${open} of ${n * d} gates are above 0.5). The next matrix reads F and brings it back to ${shp[k].d}.`,
    });
    F.ffn = { id: fid, d, from: shp[k].d };
  }

  // ---- every matrix of positions names its rows: the first tile of each stage row carries the
  // labels (the tiles beside it share its rows), numbered by position when the tokens are words
  for (const s of F.stages) for (const r of s.rows) {
    const t0 = F.tiles[r.tiles[0]];
    if (!t0 || t0.kind !== 'mat') continue;
    const pos = t0.pos || (T > 1 && t0.rows === T ? range(T) : null);
    if (!pos) continue;
    if (!t0.rowLab) t0.rowLab = pos.map(t => F.labels[t]);
    if (F.numbered) t0.rowNo = pos.map(t => t + 1);
  }

  // ---- widths: the residual stream keeps one (d_model), the FFN widens, the words set the vocabulary's
  const model = F.residual[0]?.d ?? null;
  const vocab = voc && (oneHot || F.lm) ? voc.length : null;
  if (model) {
    const ffn = F.ffn && F.residual.some(r => r.d === model) ? F.ffn.d : null;
    F.dims = { model, ffn, vocab };
    const dmodel = `d<sub>model</sub> = ${model}`;
    const mark = (id, kind, html) => { if (F.tiles[id] && F.tiles[id].cols === (kind === 'model' ? model : F.tiles[id].cols)) F.tiles[id].dim = { kind, html }; };
    for (const r of F.residual) if (r.d === model) for (const id of [...r.src, ...r.tiles]) mark(id, 'model', dmodel);
    for (const id of F.norms) mark(id, 'model', dmodel);   // a pre-norm N: the stream's width
    if (ffn) mark(F.ffn.id, 'ffn', ffn === 4 * model ? `4 d<sub>model</sub> = ${ffn}` : `d<sub>ff</sub> = ${ffn}`);
    const chain = [...new Set([...F.residual[0].from, ...F.residual.map(r => r.sym)])];
    const list = a => (a.length > 1 ? `${a.slice(0, -1).join(', ')} and ${a.at(-1)}` : a[0]);
    F.dimsText = `Residual additions (${F.residual.map(r => `${r.from.join(' + ')} + ${r.branch.join(' + ')}`).join(', ')}) add cell by cell, `
      + `so the residual stream ${list(chain)} keeps one width, ${dmodel}.`
      + (ffn ? ` The FFN widens to ${ffn} and back` : '')
      + (vocab ? `${ffn ? ';' : ''} the logits have one column per word, ${vocab}.` : ffn ? '.' : '');
  }
  if (vocab) for (const t of Object.values(F.tiles)) if ((t.id === 'in' && t.onehot) || (t.cols === vocab && (t.scale === 'prob' || /^G\d/.test(t.id)))) t.dim = { kind: 'vocab', html: `vocab = ${vocab}` };
  return F;
}

// A structure key: the DOM is rebuilt only when it changes (not on new values).
export function flowKey(F) {
  // a variant's (not the tiny language model's own): its key, the shared K, V tiles and the masks
  const v = F.variant && F.variant.key !== 'tiny_lm'
    ? [F.variant.key, Object.values(F.tiles).filter(t => t.same || t.mask).map(t => [t.id, t.same || null, t.mask])] : null;
  return JSON.stringify([F.why, F.every, F.dimsText, F.stages.map(s => [s.key, s.title, s.tex, s.mode, s.rows.map(r => [r.head, r.tiles, r.ops])]),
    Object.values(F.tiles).map(t => [t.id, t.kind, t.rows, t.cols, t.tex, t.rowLab, t.rowNo, t.colLab, t.head, t.off, t.offCols, t.onehot, t.dim, t.dimRows, t.note]),
    ...(v ? [v] : [])]);
}

// ================================================================ the view

const PLAY_MS = 1600;
const HEAD_VARS = ['--att', '--head-2', '--head-3', '--head-4', '--head-5', '--head-6'];   // a head keeps its colour in every panel
const ALPHA_MAX = 0.72;   // docs/DESIGN.md (matrix cells): fills stay light enough for --text-1 digits
const capped = c => c.replace(/^rgba\((\d+),(\d+),(\d+),([\d.]+)\)$/, (m, r, g, b, a) => `rgba(${r},${g},${b},${Math.min(ALPHA_MAX, +a).toFixed(3)})`);
// Icons from static/icons.js (docs/DESIGN.md, Icons), with a glyph when the script is missing.
const icon = (name, glyph) => window.mathboardIcons?.svg?.(name, { size: 14 }) || glyph;
// Zooms tried, largest first, until the whole flow fits its box. Below the last one the text gets
// too small to read, so the box scrolls instead (the lit stage scrolls into view).
const SCALES = [1.4, 1.25, 1.12, 1, 0.9, 0.8, 0.72];

export function install(ctx) {
  const { store } = ctx;
  const stage = ctx.el?.stage || document.getElementById('nn-stage');
  const audience = !!ctx.audience;
  const theme = () => ctx.theme?.() || (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  if (!('flow' in store.state)) store.state.flow = null;

  const btn = audience ? null : ctx.addButton?.({
    label: 'Flow', icon: 'flow', group: 'view', onClick: () => toggle(),
    title: 'Flow view (G): the whole forward pass as matrices, stage by stage, down to the next-word softmax. Shift+G: play',
  }) || null;

  const cur = () => cleanFlow(store.state.flow);
  function put(patch) {
    if (audience) return;
    const v = cur();
    if (v) store.set('flow', cleanFlow({ ...v, ...patch }));
  }
  function open(patch = {}) {
    if (audience) return;
    if (store.state.v3d) store.set('v3d', null);   // one view replaces the canvas at a time
    store.set('flow', cleanFlow({ stage: null, nums: true, ...(cur() || {}), ...patch }));
  }
  function toggle(on = !store.state.flow) {
    if (audience) return;
    if (on) open();
    else store.set('flow', null);
  }

  let V = null;   // the view while it is on
  function sync() {
    const v = cur();
    btn?.classList.toggle('on', !!v);
    // the flow draws the matrices itself: the matrix panel steps aside while it is on (nn.js
    // brings back what the user had when it closes)
    if (!v) { if (V) { const g = V; V = null; g.dispose(); ctx.matrixAway?.(false); } return; }
    if (!V) { ctx.matrixAway?.(true); V = createView(); }
    V.apply(v);
  }
  store.on('flow', sync);
  store.on('v3d', v => { if (v && store.state.flow && !audience) store.set('flow', null); });
  store.on('net', p => V?.onNet(p));
  store.on('values', () => V?.invalidate());
  for (const k of ['sel', 'hover', 'lens', 'anim']) store.on(k, () => V?.restate());
  ctx.onTheme?.(() => V?.invalidate(true));
  ctx.onShow?.(on => V?.shown(on));

  // A net that asks for the Flow view (meta.flow, the tiny language model) opens in it when it loads.
  let title = store.net.meta?.title;
  store.on('net', p => {
    const t = store.net.meta?.title;
    if (!p?.structural || t === title) return;
    title = t;
    if (store.net.meta?.flow === true && !store.state.flow) open({ stage: null, off: [] });
  });
  if (!audience && store.net.meta?.flow === true) open();

  if (!audience) {
    window.addEventListener('keydown', e => {
      if (e.ctrlKey || e.metaKey || e.altKey || !(ctx.active ? ctx.active(e) : true)) return;
      const k = e.key;
      if (k === 'g' || k === 'G') {
        e.preventDefault();
        if (e.repeat) return;
        if (e.shiftKey) { if (!store.state.flow) open(); V?.play(); }
        else toggle();
      } else if ((k === 'ArrowRight' || k === 'ArrowLeft') && store.state.flow) {
        e.preventDefault();
        V?.step(k === 'ArrowRight' ? 1 : -1);
      }
    });
  }

  // Test / console handle (docs/NN_FLOW.md).
  ctx.flow = {
    get on() { return !!store.state.flow; },
    toggle, open,
    step: dir => V?.step(dir),
    play: on => V?.play(on),
    every: (on = !cur()?.every) => put({ every: !!on }),
    head: h => V?.head(h),
    variant: key => V?.variant(key),   // another variant of the tiny language model, on the same data
    fit: () => V?.fit(),
    info: () => V?.info() ?? null,
  };

  // ============================================================== the view
  function createView() {
    const mk = (tag, cls, parent, html) => {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      if (html != null) e.innerHTML = html;
      if (parent) parent.appendChild(e);
      return e;
    };
    const root = mk('div', `nn-flow${audience ? ' ro' : ''}`);
    const scroll = mk('div', 'nnf-scroll', root);
    const sizer = mk('div', 'nnf-sizer', scroll);
    const content = mk('div', 'nnf-content', sizer);
    const head = mk('div', 'nnf-head', content);
    const varEl = mk('div', 'nnf-variant', content);   // a variant of the tiny language model: what it changes
    const dimsEl = mk('div', 'nnf-dims', content);   // why the residual stream keeps one width
    const msg = mk('div', 'nnf-msg', content);
    const strip = mk('div', 'nnf-strip', content);
    const bar = mk('div', 'nnf-bar ui-float', root);
    const ctl = mk('div', 'nnf-ctl ui-chrome', bar);
    const cap = mk('div', 'nnf-cap', bar);
    const tip = mk('div', 'nnf-tip ui-tip', root);
    tip.hidden = true;
    stage.appendChild(root);
    const svg = ctx.view?.svg || null;
    if (svg) svg.style.visibility = 'hidden';
    // Its buttons never take focus, so Space stays with training (as the shell's toolbar).
    root.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

    let F = null, key = '', cells = new Map(), stageEls = [], tileEls = new Map(), alive = true, visible = document.body.dataset.view === 'nn';
    let raf = 0, needBuild = true, needPaint = true, needState = true, needFit = true, playTimer = 0, enterT = 0, lastStage = null;
    let myHover = null, scale = 1;

    // ---------------------------------------------------------------- compute
    function compute() {
      const v = cur();
      try { F = buildFlow(store.net, { fwd: store.state.fwd, off: v?.off || [], every: !!v?.every }); }
      catch (err) {
        console.error('[nn/flow] build:', err);
        F = {
          stages: [], tiles: {}, nodeCell: {}, words: [], labels: [], numbered: false, next: null, lm: false, every: false,
          residual: [], ffn: null, dims: null, dimsText: '', attention: false, why: 'error', max: { act: 1 }, heads: 1, T: 1,
        };
      }
      const k = flowKey(F);
      if (k !== key) { key = k; needBuild = true; }
      // another net with fewer stages: back to the whole pass
      if (!audience && v && v.stage !== null && v.stage >= F.stages.length) queueMicrotask(() => put({ stage: null, play: false }));
    }

    // ---------------------------------------------------------------- build (structure)
    const texOf = (s, fallback) => {
      const k = window.katex;
      try { return k ? k.renderToString(String(s), { throwOnError: false, strict: 'ignore' }) : texHtml(fallback ?? s); } catch { return texHtml(fallback ?? s); }
    };
    // One cell width for the whole flow, so a tile's width shows how many columns it has (the FFN
    // four times the residual stream, the words wider still): 34 px, with room for the numbers, or
    // 18 px square cells with short ones (.12) when some tile has more than 10 columns.
    let CW = 34;
    const cellW = () => CW;
    // a row's name: its position's number and word (1 the), or the token's label
    const rowHead = (t, i) => (t.rowNo ? `<span class="nnf-pn">${t.rowNo[i]}</span>` : '') + esc(t.rowLab?.[i] ?? '');
    function build() {
      needBuild = false;
      strip.textContent = '';
      cells = new Map();
      tileEls = new Map();
      stageEls = [];
      CW = Object.values(F.tiles).some(t => t.cols > 10) ? 18 : 34;
      root.classList.toggle('cmp', CW < 28);
      dimsEl.innerHTML = F.dimsText || '';
      dimsEl.hidden = !F.dimsText;
      const vr = F.variant && F.variant.key !== 'tiny_lm' ? F.variant : null;
      varEl.innerHTML = vr ? `<b>${esc(vr.short)}</b>, one change from the tiny language model: ${esc(vr.note)}` : '';
      varEl.hidden = !vr;
      msg.hidden = !F.why;
      msg.innerHTML = F.why === 'error' ? 'The forward pass failed on this net, so there is nothing to draw.'
        : F.why === 'empty' ? 'This net has no neurons yet: nothing flows.'
        : F.why === 'no-attention' ? `<b>No attention layer here.</b> The flow shows this net layer by layer: each tile is one layer's vector, and hovering a neuron traces its inputs.${audience ? '' : ' <button class="ui-btn sm soft ui-chrome nnf-lm">Open the tiny language model</button>'}` : '';
      F.stages.forEach((s, si) => {
        // a stage and the arrow after it wrap together, so a line ends with an arrow, never starts with one
        const pair = mk('div', 'nnf-pair', strip);
        const el = mk('div', 'nnf-st', pair);
        if (si < F.stages.length - 1) mk('div', 'nnf-arrow', pair, '<svg viewBox="0 0 16 16"><path d="M2 8h11M9 4l4 4-4 4"/></svg>');
        el.dataset.i = si;
        const sh = mk('button', 'nnf-sh', el);
        sh.innerHTML = `<span class="nnf-no">${si + 1}</span>${esc(s.title).replace(/&lt;(\/?)(sub|sup)&gt;/g, '<$1$2>')}`;
        sh.title = `Focus ${net().layers.find(x => x.id === s.layer)?.name || 'this layer'}${s.part ? ` · ${s.part}` : ''} on the canvas and the matrix panel`;
        // a language model's ending: the last position (as generating reads it), or every position (as training scores them)
        if (s.mode) {
          mk('div', 'nnf-mode ui-seg sm ui-chrome', el,
            `<button data-every="0" class="${F.every ? '' : 'on'}" title="Only the last position's distribution: the next word, as generating reads it">last position</button>`
            + `<button data-every="1" class="${F.every ? 'on' : ''}" title="Every position's prediction of its own next word: training scores them all at once">every position (how it's trained)</button>`);
        }
        const body = mk('div', 'nnf-rows', el);
        for (const r of s.rows) {
          const row = mk('div', 'nnf-row', body);
          if (r.head !== null) {
            row.dataset.h = r.head;
            row.style.setProperty('--hc', `var(${HEAD_VARS[r.head % HEAD_VARS.length]})`);
            mk('span', 'nnf-hl', row, `h${r.head + 1}`);
          }
          r.tiles.forEach((id, ti) => {
            if (ti && r.ops[ti - 1]) mk('span', 'nnf-op', row, r.ops[ti - 1]);
            row.appendChild(buildTile(F.tiles[id]));
          });
        }
        stageEls.push(el);
      });
      needPaint = needState = needFit = true;
    }
    function buildTile(t) {
      const el = mk('div', `nnf-tile${t.kind === 'bars' || t.kind === 'dist' ? ` ${t.kind}` : ''}`);
      el.dataset.t = t.id;
      if (t.head !== null) el.style.setProperty('--hc', `var(${HEAD_VARS[t.head % HEAD_VARS.length]})`);
      const nm = mk('div', 'nnf-tn', el);
      mk('span', 'nnf-name', nm, texOf(t.tex));
      mk('span', 'nnf-shape', nm, `${t.rows}×${t.cols}`);
      if (t.same) {   // a head that shares its K or V (multi-query, grouped-query)
        const b = mk('span', 'nnf-same', nm, `= ${texOf(t.same)}`);
        b.title = `The same matrix as ${texHtml(t.same).replace(/<[^>]+>/g, '')}: these heads share it`;
        el.classList.add('shared');
      }
      const cw = cellW(t), list = [];
      // the bracket under a tile: its width, by what sets it (d_model, the FFN's, the vocabulary)
      const bracket = (g, from) => {
        if (!t.dim) return;
        const b = mk('div', `nnf-dim ${t.dim.kind}`, g, `<span>${t.dim.html}</span>`);
        b.style.gridColumn = `${from} / -1`;
      };
      if (t.kind === 'bars' || t.kind === 'dist') {
        // the next word: a bar per word, the words under the bars; dist is one position's, bars a row per position
        const dist = t.kind === 'dist';
        if (dist && t.rowLab?.[0]) mk('span', 'nnf-ctx', nm, `after “${esc(t.rowLab[0])}”`);
        if (dist && t.target) mk('span', 'nnf-key', nm, '<i class="k-top"></i>most likely<i class="k-tgt"></i>true next word');
        const g = mk('div', `nnf-bg${dist ? ' dist' : ''}`, el);
        g.style.gridTemplateColumns = `${dist ? '' : 'auto '}repeat(${t.cols}, ${cw}px)`;
        const preds = [];
        for (let i = 0; i < t.rows; i++) {
          if (!dist) {
            const rh = mk('div', 'nnf-rh bar', g);
            rh.innerHTML = `<small>after</small><span class="nnf-ctxw">${esc(t.rowLab?.[i] ?? `t${i + 1}`)}</span>`;
            preds.push(mk('span', 'nnf-pred', rh));
          }
          for (let j = 0; j < t.cols; j++) {
            const c = mk('div', 'nnf-b', g);
            c.dataset.t = t.id; c.dataset.i = i; c.dataset.j = j;
            c.style.setProperty('--k', i + j);
            const val = mk('span', '', c), bar = mk('i', '', c);
            list.push({ el: c, i, j, bar, val });
          }
        }
        if (!dist) mk('div', 'nnf-corner', g);
        const labs = [];
        for (let j = 0; j < t.cols; j++) labs.push(mk('div', 'nnf-ch vert bot', g, esc(t.colLab?.[j] ?? String(j + 1))));
        bracket(g, dist ? 1 : 2);
        list.preds = preds;
        list.labs = labs;
      } else {
        const g = mk('div', 'nnf-g', el);
        g.style.gridTemplateColumns = `${t.rowLab ? 'auto ' : ''}repeat(${t.cols}, ${cw}px)`;
        if (t.colLab || t.colHeads) {
          if (t.rowLab) mk('div', 'nnf-corner', g);
          for (let j = 0; j < t.cols; j++) {
            const ch = mk('div', `nnf-ch${t.colLab && t.cols > 3 && t.colLab.some(s => String(s).length > 2) ? ' vert' : ''}`, g);
            if (t.colLab) ch.textContent = t.colLab[j];
            else { ch.classList.add('hc'); ch.style.setProperty('--hc', `var(${HEAD_VARS[t.colHeads[j] % HEAD_VARS.length]})`); }
          }
        }
        const dimR = new Set(t.dimRows || []);
        for (let i = 0; i < t.rows; i++) {
          if (t.rowLab) mk('div', `nnf-rh${dimR.has(i) ? ' dimr' : ''}`, g, rowHead(t, i));
          for (let j = 0; j < t.cols; j++) {
            const c = mk('div', 'nnf-c', g);
            c.dataset.t = t.id; c.dataset.i = i; c.dataset.j = j;
            c.style.setProperty('--k', i + j);
            if (t.mask?.[i]?.[j]) c.classList.add('mask');
            if (t.offCols?.[j]) c.classList.add('offc');
            if (dimR.has(i)) c.classList.add('dimr');
            list.push({ el: c, i, j });
          }
        }
        bracket(g, t.rowLab ? 2 : 1);
      }
      if (t.note) mk('div', 'nnf-note', el, esc(t.note));
      if (t.off) el.classList.add('off');
      el.style.setProperty('--cw', `${cw}px`);
      cells.set(t.id, list);
      tileEls.set(t.id, el);
      return el;
    }

    // ---------------------------------------------------------------- paint (values)
    function paint() {
      needPaint = false;
      const th = theme(), v = cur(), nums = v?.nums !== false;
      for (const t of Object.values(F.tiles)) {
        const list = cells.get(t.id);
        if (!list) continue;
        const max = t.scale === 'act' ? F.max.act : t.scale === 'bias' ? F.max.bias : 1, fmtC = cellW(t) >= 28 ? f2 : fc;
        const bars = t.kind === 'bars' || t.kind === 'dist', tops = bars ? t.v.map(row => argmax(row || [])) : null;
        for (const c of list) {
          const x = t.v[c.i]?.[c.j];
          if (bars) {
            const p = isNum(x) ? clamp(x, 0, 1) : 0;
            const h = `${(p * 100).toFixed(1)}%`;
            if (c.bar._h !== h) { c.bar._h = h; c.el.style.setProperty('--p', p.toFixed(4)); }
            const top = tops[c.i] === c.j;
            c.el.classList.toggle('top', top);
            c.el.classList.toggle('tgt', t.target?.[c.i] === c.j);
            // the most likely word's value always; the chart's others from 5%, the rows' from 20%
            const s = top || p >= (t.kind === 'dist' ? 0.05 : 0.2) ? fc(p) : '';
            if (c.val._t !== s) c.val.textContent = c.val._t = s;
            continue;
          }
          const masked = !!t.mask?.[c.i]?.[c.j], zero = isNum(x) && Math.abs(x) < 0.005;
          const col = masked || zero || t.offCols?.[c.j] ? '' : capped(colorFor(x, max, th));
          if (c.el._c !== col) { c.el._c = col; c.el.style.background = col; }
          if (c.el._z !== zero) { c.el._z = zero; c.el.classList.toggle('zero', zero); }
          // narrow cells leave a 0 blank (the unfilled cell says it: a ReLU that is off)
          const s = masked ? (t.scale === 'attn' || t.scale === 'bias' || t.maskBlank ? '' : '−∞') : t.onehot ? (x === 1 ? '1' : '') : nums && !(zero && fmtC === fc) ? fmtC(x) : '';
          if (c.el._t !== s) c.el.textContent = c.el._t = s;
        }
        if (bars) {
          // each row's prediction beside it (after "the cat" → sat .34); the chart's words mark the top and the true one
          list.preds?.forEach((el, i) => {
            const k = tops[i], s = t.colLab && isNum(t.v[i]?.[k]) ? `→ <b>${esc(t.colLab[k])}</b> ${fc(t.v[i][k])}` : '';
            if (el._h !== s) el.innerHTML = el._h = s;
          });
          if (t.kind === 'dist') list.labs?.forEach((el, j) => { el.classList.toggle('top', tops[0] === j); el.classList.toggle('tgt', t.target?.[0] === j); });
        }
      }
      paintHead();
      paintCaption();
      if (tip._cell) showTip(tip._cell);
    }
    // The sentence in order, a numbered chip per position (1 the, 2 cat, ...), then the next word:
    // "?" until the lit stage reaches the end of the pass, then the prediction and the true word.
    function paintHead() {
      const words = F.words, has = words.length && words.every(Boolean);
      const toks = has ? words : F.labels;
      let h = '';
      if (has || F.T > 1) {   // a plain net has no tokens to name
        h += `<span class="nnf-lab">${has ? 'Input' : 'Tokens'}</span>`;
        h += toks.map((w, t) => `<span class="nnf-w" data-t="${t}">${F.numbered ? `<span class="nnf-pn">${t + 1}</span>` : ''}${esc(w)}</span>`).join('');
      }
      if (F.next) {
        const i = cur()?.stage, n = F.stages.length, done = i === null || i === undefined || i >= n - 1;
        const no = F.numbered ? `<span class="nnf-pn">${F.next.t + 2}</span>` : '';
        h += '<span class="nnf-to">&rarr;</span>';
        if (!done) h += `<span class="nnf-next q" title="The next word: the last stage computes it">${no}?</span>`;
        else {
          const ok = F.next.target ? (F.next.target === F.next.word ? ' ok' : ' bad') : '';
          h += `<span class="nnf-next${ok}">${no}${esc(F.next.word)}<small>${M.fmt(F.next.p * 100, 0)}%</small></span>`;
          if (F.next.target) h += `<span class="nnf-tgt">${F.next.target === F.next.word ? '&#10003; ' : '&#10007; '}true next word: ${esc(F.next.target)}</span>`;
        }
      }
      const off = cur()?.off || [];
      if (off.length && F.attention) h += `<span class="nnf-offn">head ${off.map(x => x + 1).join(', ')} off</span>`;
      if (head._h !== h) { head._h = h; head.innerHTML = h; }
    }
    function paintCaption() {
      const v = cur(), i = v?.stage;
      const s = i !== null && i !== undefined ? F.stages[i] : null;
      let h;
      if (s) h = `<b>${i + 1} · ${esc(s.title).replace(/&lt;(\/?)(sub|sup)&gt;/g, '<$1$2>')}</b><span class="nnf-tex">${texOf(s.tex)}</span><span class="nnf-txt">${s.text}</span>`;
      else {
        const n = F.stages.length;
        h = `<b>Forward pass</b><span class="nnf-txt">${n} stage${n === 1 ? '' : 's'}, left to right${F.attention ? ', one tile per matrix' : ''}. `
          + `${audience ? '' : '◀ ▶ (← →) step through them, ▶ plays; hover a cell to trace it, click a stage to focus its layer.'}</span>`;
      }
      if (cap._h !== h) { cap._h = h; cap.innerHTML = h; }
    }

    // ---------------------------------------------------------------- controls
    function renderCtl() {
      const v = cur(), n = F?.stages.length || 0, i = v?.stage ?? null;
      let h = '<span class="nnf-title">Flow</span>';
      // the tiny language model's variants: the same data, one change each (the picker rebuilds on it)
      if (F?.variant && !audience) {
        h += `<select class="ui-field sm nnf-varsel" title="Variants of the tiny language model: the same data and sizes, one change each. Picking one builds it (Ctrl+Z goes back)" aria-label="Variant">`
          + variantMenu().map(([axis, list]) => `<optgroup label="${esc(axis)}">${list.map(it => `<option value="${esc(it.key)}"${it.key === F.variant.key ? ' selected' : ''}>${esc(it.short)}</option>`).join('')}</optgroup>`).join('')
          + '</select><span class="nnf-sep"></span>';
      }
      h += `<button class="ui-btn sm icon" data-a="whole" title="The whole pass: no stage lit"${i === null ? ' disabled' : ''}>${icon('stop', '&#9632;')}</button>`;
      h += `<button class="ui-btn sm icon" data-a="prev" title="Previous stage (←)"${!n ? ' disabled' : ''}>${icon('step-back', '&#9664;')}</button>`;
      h += `<button class="ui-btn sm icon${v?.play ? ' on' : ''}" data-a="play" title="${v?.play ? 'Pause' : 'Play the stages one by one'} (Shift+G)"${!n ? ' disabled' : ''}>${v?.play ? icon('pause', '&#10074;&#10074;') : icon('play', '&#9654;')}</button>`;
      h += `<button class="ui-btn sm icon" data-a="next" title="Next stage (→)"${!n ? ' disabled' : ''}>${icon('step-forward', '&#9654;')}</button>`;
      h += '<span class="ui-seg sm nnf-steps">' + (F?.stages || []).map((s, k) => `<button data-a="go" data-i="${k}" class="${k === i ? 'on' : ''}" title="${esc(s.title.replace(/<[^>]+>/g, ''))}">${k + 1}</button>`).join('') + '</span>';
      if ((F?.heads || 1) > 1) {
        h += '<span class="nnf-sep"></span><span class="nnf-lab2">heads</span>';
        for (let k = 0; k < F.heads; k++) {
          const off = v?.off?.includes(k);
          h += `<button class="ui-chip${off ? '' : ' on'}" data-a="head" data-h="${k}" title="${off ? 'Turn head ' + (k + 1) + ' back on' : 'Knock head ' + (k + 1) + ' out: its columns of the concat become 0 and the rest is recomputed'}"><span class="ui-sw" style="background: var(${HEAD_VARS[k % HEAD_VARS.length]})"></span>head ${k + 1}${off ? ' off' : ''}</button>`;
        }
      }
      h += `<span class="nnf-sep"></span><button class="ui-btn sm${v?.nums !== false ? ' on' : ''}" data-a="nums" title="Numbers in the cells">1.2</button>`;
      if (ctx.train?.stepSample) {
        h += `<span class="nnf-sep"></span><button class="ui-btn sm icon" data-a="sample" data-d="-1" title="Previous sample of the Train panel's dataset">${icon('chevron-left', '&#8249;')}</button>`
          + '<span class="nnf-lab2">sample</span>'
          + `<button class="ui-btn sm icon" data-a="sample" data-d="1" title="Next sample of the Train panel's dataset">${icon('chevron-right', '&#8250;')}</button>`;
      }
      if (ctl._h !== h) { ctl._h = h; ctl.innerHTML = h; }
    }
    ctl.addEventListener('change', e => {
      const sel = e.target.closest?.('.nnf-varsel');
      if (!sel || audience) return;
      sel.blur();   // the keys go back to the tab
      switchVariant(sel.value);
    });
    ctl.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b || audience) return;
      const a = b.dataset.a;
      if (a === 'whole') { setPlay(false); put({ stage: null }); }
      else if (a === 'prev') step(-1);
      else if (a === 'next') step(1);
      else if (a === 'play') play();
      else if (a === 'go') { setPlay(false); put({ stage: +b.dataset.i }); }
      else if (a === 'head') headToggle(+b.dataset.h);
      else if (a === 'nums') put({ nums: !(cur()?.nums !== false) });
      else if (a === 'sample') { try { ctx.train.stepSample(+b.dataset.d); } catch (err) { console.error('[nn/flow] sample:', err); } }
    });
    msg.addEventListener('click', e => {
      if (!e.target.closest('.nnf-lm') || audience) return;
      const p = M.PRESETS.tiny_lm;
      if (!p) return;
      store.load(p.build(1));
      ctx.toast?.(p.note, 5000);
    });

    function step(dir) {
      if (!F && !audience) compute();
      if (audience || !F?.stages.length) return;
      setPlay(false);
      const n = F.stages.length, i = cur()?.stage;
      const next = i === null || i === undefined ? (dir > 0 ? 0 : n - 1) : i + dir;
      put({ stage: next < 0 || next >= n ? null : next });
    }
    function setPlay(on) {
      clearInterval(playTimer);
      playTimer = 0;
      if (!on) { if (cur()?.play) put({ play: false }); return; }
      if (!F && !audience) compute();   // opened this frame (Shift+G on a closed view): not built yet
      const n = F?.stages.length || 0;
      if (!n || audience) return;
      let i = cur()?.stage;
      if (i === null || i === undefined || i >= n - 1) i = 0;
      put({ stage: i, play: true });
      playTimer = setInterval(() => {
        const v = cur(), m = F?.stages.length || 0;
        if (!v?.play || !alive || v.stage === null) { clearInterval(playTimer); playTimer = 0; return; }
        if (v.stage >= m - 1) { setPlay(false); return; }
        put({ stage: v.stage + 1 });
      }, PLAY_MS);
    }
    function play(on) {
      if (audience) return;
      setPlay(on === undefined ? !cur()?.play : !!on);
    }
    // Another variant of the tiny language model on the same data: the Train panel's data settings,
    // batch, speed and init seed carry over, and so does the sentence in the inputs (with its targets and
    // token names); the variant keeps its own recorded rate. store.load: Ctrl+Z comes back.
    function switchVariant(key) {
      const p = M.PRESETS[key], old = store.net;
      if (!p?.family || audience || variantOf(old)?.key === key) return;
      const ot = isObj(old.meta?.train) ? old.meta.train : {};
      const seed = Number.isInteger(ot.initSeed) ? ot.initSeed : 1;
      let net;
      try { net = p.build(seed); } catch (err) { console.error('[nn/flow] variant:', err); ctx.toast?.(`Could not build that variant: ${err.message}`); return; }
      for (const k of ['n', 'noise', 'seed', 'batch', 'speed', 'initSeed']) if (ot[k] !== undefined) net.meta.train[k] = ot[k];
      const io = n => [M.nodesIn(n, 0), M.nodesIn(n, n.layers.length - 1)], [i0, o0] = io(old), [i1, o1] = io(net);
      if (i0.length === i1.length && o0.length === o1.length) {
        i1.forEach((nd, j) => { nd.value = i0[j].value; });
        o1.forEach((nd, j) => { nd.target = o0[j].target; });
        if (Array.isArray(old.meta?.tokenNames)) net.meta.tokenNames = old.meta.tokenNames.slice();
      }
      setPlay(false);
      store.load(net);
      put({ stage: null, off: [], hover: null });
      ctx.toast?.(`${p.short}: ${p.note}`, 5000);
    }
    function headToggle(h) {
      if (audience) return;
      const off = new Set(cur()?.off || []);
      if (off.has(h)) off.delete(h); else off.add(h);
      if (off.size >= (F?.heads || 1)) { ctx.toast?.('Keep at least one head on', 1600); return; }
      put({ off: [...off] });
    }

    // ---------------------------------------------------------------- state: lit stage, trace, lens, hover
    const net = () => store.net;
    function cellEl(t, i, j) {
      const list = cells.get(t);
      return list ? list.find(c => c.i === i && c.j === j)?.el || null : null;
    }
    function paintState() {
      needState = false;
      paintHead();   // the next word shows once the lit stage reaches the end (before the chips get their token ring)
      const v = cur(), i = v?.stage ?? null;
      stageEls.forEach((el, k) => {
        el.classList.toggle('cur', k === i);
        el.classList.toggle('later', i !== null && k > i);
      });
      if (i !== lastStage) {
        lastStage = i;
        if (i !== null && stageEls[i]) {
          const el = stageEls[i];
          for (const e of stageEls) e.classList.remove('enter');   // the last lit one too, if cut short
          void el.offsetWidth;
          el.classList.add('enter');
          clearTimeout(enterT);
          enterT = setTimeout(() => el.classList.remove('enter'), 900);
          revealStage(el);
        }
      }
      // lens: the focused layer's stages get a frame, a followed token its rows, a kept head the rest dimmed
      const L = cleanLens(net(), store.state.lens);
      stageEls.forEach((el, k) => {
        const s = F.stages[k], f = L.focus;
        // a Q, K or V part is not a stage of its own: it frames the Q, K, V stage
        el.classList.toggle('foc', !!f && f.layer === s.layer && (!f.part || f.part === s.part || (!s.part && QKV.includes(f.part))));
      });
      for (const el of root.querySelectorAll('.nnf-row[data-h]')) el.classList.toggle('dimh', L.head !== null && +el.dataset.h !== L.head);
      // hover: the presenter's cell (mirrored), else the shared hover (a neuron or a token)
      for (const el of root.querySelectorAll('.nnf-c.hov, .nnf-c.src, .nnf-b.hov, .nnf-b.src, .nnf-c.tok, .nnf-b.tok, .nnf-c.sel, .nnf-b.sel')) el.classList.remove('hov', 'src', 'tok', 'sel');
      root.classList.remove('tracing');
      let cell = v?.hover && F.tiles[v.hover.t] ? v.hover : null;
      const sh = store.state.hover;
      if (!cell && sh?.kind === 'node' && F.nodeCell[sh.id]) { const [t, a, b] = F.nodeCell[sh.id]; cell = { t, i: a, j: b }; }
      if (cell) {
        const el = cellEl(cell.t, cell.i, cell.j);
        if (el) {
          el.classList.add('hov');
          root.classList.add('tracing');
          for (const [t, a, b] of F.tiles[cell.t].src(cell.i, cell.j) || []) cellEl(t, a, b)?.classList.add('src');
        }
      }
      const tokT = L.token !== null ? L.token : sh?.kind === 'token' && !cell ? sh.t : null;
      if (tokT !== null && tokT !== undefined) {
        for (const [id, list] of cells) {
          const t = F.tiles[id];
          if (t.rows < 2 && !t.pos) continue;
          for (const c of list) if ((t.pos ? t.pos[c.i] : c.i) === tokT) c.el.classList.add('tok');
        }
      }
      const sel = store.state.sel;
      if (sel?.kind === 'node' && F.nodeCell[sel.id]) { const [t, a, b] = F.nodeCell[sel.id]; cellEl(t, a, b)?.classList.add('sel'); }
      // the matrix panel's step-through (S): its neuron gets the HI frame here too
      const an = store.state.anim, al = an ? (typeof an.l === 'number' ? an.l : net().layers.findIndex(x => x.id === an.l)) : -1;
      const aid = al > 0 ? M.nodesIn(net(), al)[an.i]?.id : null;
      if (aid && F.nodeCell[aid]) { const [t, a, b] = F.nodeCell[aid]; cellEl(t, a, b)?.classList.add('src'); }
      for (const w of head.querySelectorAll('.nnf-w')) w.classList.toggle('tok', +w.dataset.t === tokT);
      if (cell) showTip(cell); else hideTip();
      paintCaption();
      renderCtl();
    }
    // Scroll the lit stage into view when the flow is taller than its box.
    function revealStage(el) {
      if (scroll.scrollHeight <= scroll.clientHeight + 2) return;
      const r = el.getBoundingClientRect(), b = scroll.getBoundingClientRect();
      if (r.top < b.top || r.bottom > b.bottom) scroll.scrollTo({ top: scroll.scrollTop + r.top - b.top - 12, behavior: 'smooth' });
    }

    function showTip(cell) {
      const t = F.tiles[cell.t], el = cellEl(cell.t, cell.i, cell.j);
      if (!t || !el) { hideTip(); return; }
      let html = '';
      try { html = t.tip(cell.i, cell.j); } catch { html = ''; }
      if (!html) { hideTip(); return; }
      const tp = t.pos ? t.pos[cell.i] : cell.i;   // the row's position
      const who = t.rows > 1 || t.pos ? `<small>${esc(F.labels[tp] && t.kind === 'mat' && !t.colLab ? `position ${tp + 1} (${F.labels[tp]})` : `position ${tp + 1}`)}</small>` : '';
      if (tip._h !== html + who) { tip._h = html + who; tip.innerHTML = who + html; }
      tip._cell = cell;
      tip.hidden = false;
      tip.style.left = '0px';   // measure it unsqueezed by the right edge
      tip.style.top = '0px';
      // above the tile (so the tile stays readable), centred on the cell; below it when there is no room
      const r = el.getBoundingClientRect(), s = root.getBoundingClientRect();
      const box = (el.closest('.nnf-tile') || el).getBoundingClientRect();
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let x = r.left - s.left + r.width / 2 - w / 2, y = box.top - s.top - h - 6;
      if (y < 4) y = box.bottom - s.top + 6;
      tip.style.left = `${Math.round(clamp(x, 4, s.width - w - 4))}px`;
      tip.style.top = `${Math.round(clamp(y, 4, s.height - h - 4))}px`;
    }
    function hideTip() { tip.hidden = true; tip._cell = null; }

    // ---------------------------------------------------------------- input
    const cellAt = e => {
      const c = e.target.closest?.('.nnf-c, .nnf-b');
      return c && root.contains(c) ? { t: c.dataset.t, i: +c.dataset.i, j: +c.dataset.j } : null;
    };
    const sameCell = (a, b) => (a?.t ?? null) === (b?.t ?? null) && a?.i === b?.i && a?.j === b?.j;
    function hoverCell(c) {
      if (audience) return;
      const v = cur();
      if (!v || sameCell(v.hover, c)) return;
      put({ hover: c });
      // the shared hover: its neuron, or for a score or weight its query token
      const t = c && F.tiles[c.t];
      let h = null;
      const nid = t ? t.node(c.i, c.j) : null;
      if (nid) h = { kind: 'node', id: nid };
      else if (t && /^[sa]\d/.test(t.id)) h = { kind: 'token', layer: t.l, t: c.i, ...(t.head !== null ? { h: t.head } : {}) };
      else if (t && (t.rows > 1 || t.pos)) h = { kind: 'token', layer: t.l, t: t.pos ? t.pos[c.i] : c.i };
      const was = store.state.hover;
      if (JSON.stringify(h) !== JSON.stringify(was) && (h || (was && JSON.stringify(was) === JSON.stringify(myHover)))) {
        myHover = h;
        store.set('hover', h);
      }
    }
    content.addEventListener('pointermove', e => { if (!audience) hoverCell(cellAt(e)); });
    content.addEventListener('pointerleave', () => { if (!audience) hoverCell(null); });
    content.addEventListener('click', e => {
      if (audience) return;
      const md = e.target.closest('[data-every]');
      if (md) { put({ every: md.dataset.every === '1' }); return; }
      const sh = e.target.closest('.nnf-sh');
      if (sh) { focusStage(+sh.parentElement.dataset.i); return; }
      const c = cellAt(e);
      if (c) {
        const t = F.tiles[c.t], nid = t?.node(c.i, c.j);
        if (nid) store.set('sel', { kind: 'node', id: nid });
        else if (t && /^[sa]\d/.test(t.id)) store.set('sel', { kind: 'layer', id: net().layers[t.l].id });
        return;
      }
      const w = e.target.closest('.nnf-w');
      if (w && F.T > 1) {
        const L = cleanLens(net(), store.state.lens), tt = +w.dataset.t;
        store.set('lens', copyLens(cleanLens(net(), { ...L, token: L.token === tt ? null : tt })));
        return;
      }
      if (!e.target.closest('.nnf-tile, .nnf-head, button')) store.set('sel', null);
    });
    // the room round the flow (the scroll box's margins, the stage beside it) is empty space too
    root.addEventListener('click', e => {
      if (audience || content.contains(e.target) || e.target.closest('.nnf-bar, .nnf-tip')) return;
      store.set('sel', null);
    });
    // A stage click: light it, focus its layer (and part) through the lens, reveal it in the matrix panel.
    function focusStage(i) {
      const s = F.stages[i];
      if (!s) return;
      setPlay(false);
      put({ stage: i });
      const L = cleanLens(net(), store.state.lens);
      const focus = { layer: s.layer, ...(s.part ? { part: s.part } : {}) };
      store.set('lens', copyLens(cleanLens(net(), { ...L, focus })));
      try { ctx.matrix?.reveal?.(s.layer, s.part || null); } catch { /* optional */ }
    }

    // ---------------------------------------------------------------- framing
    // The free part of the stage: clear of the floating panels (Train, Attention, 3D plots) and
    // above this bar and the lens bar; the roomiest rectangle at least 320 × 200.
    function freeRect() {
      const s = stage.getBoundingClientRect();
      let bottom = s.height;
      const br = bar.getBoundingClientRect();
      if (br.height) bottom = Math.min(bottom, br.top - s.top - 6);
      const rects = [];
      const skip = c => c === root || c.tagName?.toLowerCase() === 'svg' || c.classList.contains('nn-insp-layer') || c.classList.contains('nn-tour') || c.classList.contains('nn3d');
      for (const c of stage.children) {
        if (skip(c) || c.hidden) continue;
        const r = c.getBoundingClientRect();
        if (!r.width || !r.height || getComputedStyle(c).display === 'none' || getComputedStyle(c).visibility === 'hidden') continue;
        if (c.classList.contains('nn-lens')) { if (r.bottom >= s.bottom - 40) bottom = Math.min(bottom, r.top - s.top - 6); continue; }
        if (r.width * r.height > 0.8 * s.width * s.height) continue;
        rects.push({ x0: r.left - s.left - 8, y0: r.top - s.top - 8, x1: r.right - s.left + 8, y1: r.bottom - s.top + 8 });
      }
      const W = s.width, H = Math.max(s.height * 0.45, bottom), all = { x: 0, y: 0, w: W, h: H };
      const obs = rects.filter(r => r.x1 > 0 && r.x0 < W && r.y1 > 0 && r.y0 < H);
      if (!obs.length) return all;
      const cuts = (lo, hi, k0, k1) => [...new Set([lo, hi, ...obs.flatMap(r => [r[k0], r[k1]]).filter(x => x > lo && x < hi)])].sort((a, b) => a - b);
      const xs = cuts(0, W, 'x0', 'x1'), ys = cuts(0, H, 'y0', 'y1');
      let best = null;
      for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) {
        if (xs[j] - xs[i] < 320) continue;
        for (let p = 0; p < ys.length; p++) for (let q = p + 1; q < ys.length; q++) {
          const A = { x: xs[i], y: ys[p], w: xs[j] - xs[i], h: ys[q] - ys[p] };
          if (A.h < 200 || obs.some(r => r.x0 < A.x + A.w && r.x1 > A.x && r.y0 < A.y + A.h && r.y1 > A.y)) continue;
          if (!best || A.w * A.h > best.w * best.h) best = A;
        }
      }
      return best || all;
    }
    let area = null;
    function fit() {
      needFit = false;
      const A = freeRect();
      area = A;
      Object.assign(scroll.style, { left: `${A.x}px`, top: `${A.y}px`, width: `${A.w}px`, height: `${A.h}px` });
      const pad = 12, W = Math.max(200, A.w - 2 * pad), Hh = Math.max(120, A.h - 2 * pad);
      const fits = s => {
        content.style.width = `${Math.floor(W / s)}px`;
        return content.offsetHeight * s <= Hh && content.scrollWidth * s <= W + 1;
      };
      let pick = SCALES[SCALES.length - 1], hPick = 0, over = 0;
      for (let k = 0; k < SCALES.length; k++) {
        if (fits(SCALES[k])) { pick = SCALES[k]; over = SCALES[k - 1] || 0; break; }
      }
      // then as large as fits between that step and the one above it (text is what reads)
      for (let k = 0; over && k < 4; k++) {
        const mid = (pick + over) / 2;
        if (fits(mid)) pick = mid; else over = mid;
      }
      content.style.width = `${Math.floor(W / pick)}px`;
      hPick = content.offsetHeight;
      scale = pick;
      content.style.transform = `scale(${pick})`;
      sizer.style.width = `${Math.round(W)}px`;
      sizer.style.height = `${Math.ceil(hPick * pick)}px`;
      sizer.style.marginTop = `${Math.max(0, Math.round((Hh - hPick * pick) / 2))}px`;
      nudgeCards();
    }
    let nudgeT = 0;
    function nudgeCards() {   // the inspector re-places its cards on wheel input (as view3d.js does)
      const t = performance.now();
      if (t - nudgeT < 300) return;
      nudgeT = t;
      requestAnimationFrame(() => root.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 0 })));
    }
    // The bar runs along the bottom, between the floating panels that reach down there (a Train
    // panel on the right ends it early), and above the lens bar when that one is under it.
    function placeBar() {
      bar.style.bottom = bar.style.left = bar.style.right = '';
      const s = stage.getBoundingClientRect(), band = s.bottom - bar.offsetHeight - 70;
      let L = 10, R = 10;
      for (const c of stage.children) {
        if (c === root || c.hidden || c.tagName?.toLowerCase() === 'svg' || ['nn-insp-layer', 'nn-tour', 'nn-lens', 'nn3d'].some(k => c.classList.contains(k))) continue;
        const r = c.getBoundingClientRect();
        if (!r.width || !r.height || r.bottom < band || r.width > 0.8 * s.width || getComputedStyle(c).display === 'none') continue;
        if ((r.left + r.right) / 2 > s.left + s.width / 2) R = Math.max(R, Math.round(s.right - r.left + 8));
        else L = Math.max(L, Math.round(r.right - s.left + 8));
      }
      if (s.width - L - R >= 280) { bar.style.left = `${L}px`; bar.style.right = `${R}px`; }
      const lens = stage.querySelector(':scope > .nn-lens');
      if (!lens || lens.hidden || !lens.offsetWidth || getComputedStyle(lens).display === 'none') return;
      const a = bar.getBoundingClientRect(), b = lens.getBoundingClientRect();
      if (a.left < b.right + 8 && b.left < a.right && a.top < b.bottom && b.top < a.bottom) {
        bar.style.bottom = `${Math.round(root.getBoundingClientRect().bottom - b.top + 8)}px`;
      }
    }

    // ---------------------------------------------------------------- ctx.view while on
    const orig = ctx.view ? { nodeRect: ctx.view.nodeRect, contentRect: ctx.view.contentRect, fit: ctx.view.fit } : null;
    if (ctx.view) {
      ctx.view.nodeRect = id => {
        const c = F?.nodeCell[id], el = c && cellEl(c[0], c[1], c[2]);
        if (!el) return null;
        const r = el.getBoundingClientRect(), s = stage.getBoundingClientRect();
        return { x: r.left - s.left, y: r.top - s.top, w: r.width, h: r.height };
      };
      ctx.view.contentRect = () => {
        const r = content.getBoundingClientRect(), s = stage.getBoundingClientRect();
        return { x: r.left - s.left, y: r.top - s.top, w: r.width, h: r.height };
      };
      ctx.view.fit = ms => { try { orig.fit?.(ms); } catch { /* the 2D view */ } needFit = true; kick(); };
    }

    // ---------------------------------------------------------------- loop
    function frame() {
      raf = 0;
      if (!alive || !visible) return;
      if (!F || needPaint) compute();
      if (needBuild) build();
      if (needFit) { placeBar(); fit(); }
      if (needPaint) paint();
      if (needState) paintState();
    }
    const kick = () => { if (!raf && alive && visible) raf = requestAnimationFrame(frame); };
    const areaObs = new ResizeObserver(() => { needFit = true; kick(); });
    const watch = () => { areaObs.observe(stage); areaObs.observe(bar); for (const c of stage.children) if (c !== root) areaObs.observe(c); };
    const mo = new MutationObserver(watch);
    mo.observe(stage, { childList: true });
    watch();
    const onUp = () => requestAnimationFrame(() => {
      const A = freeRect();
      if (!area || ['x', 'y', 'w', 'h'].some(k => Math.abs(A[k] - area[k]) > 1)) { needFit = true; kick(); }
    });
    stage.addEventListener('pointerup', onUp, true);
    kick();

    function dispose() {
      alive = false;
      clearInterval(playTimer);
      clearTimeout(enterT);
      cancelAnimationFrame(raf);
      areaObs.disconnect();
      mo.disconnect();
      stage.removeEventListener('pointerup', onUp, true);
      if (myHover && JSON.stringify(store.state.hover) === JSON.stringify(myHover)) store.set('hover', null);
      root.remove();
      if (svg) svg.style.visibility = '';
      if (ctx.view && orig) Object.assign(ctx.view, orig);
      requestAnimationFrame(() => stage.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 0 })));
    }

    return {
      apply(v) {
        if (!v.play && playTimer) { clearInterval(playTimer); playTimer = 0; }
        const offKey = JSON.stringify(v.off);
        if (offKey !== this._off) { this._off = offKey; needPaint = true; }
        if (v.nums !== this._nums) { this._nums = v.nums; needPaint = true; }
        if (v.every !== this._every) { this._every = v.every; needPaint = true; }   // recomputed: the ending's tiles change
        needState = true;
        kick();
      },
      onNet(p) { if (p?.structural) { needFit = true; } needPaint = true; needState = true; kick(); },
      invalidate(all) { if (all) for (const list of cells.values()) for (const c of list) { c.el._c = undefined; } needPaint = true; needState = true; kick(); },
      restate() { needState = true; kick(); },
      shown(on) { visible = on; if (on) { needFit = needPaint = needState = true; kick(); } },
      step, play, head: headToggle, variant: switchVariant, fit: () => { needFit = true; kick(); }, dispose,
      info: () => ({
        stages: F?.stages.map(s => s.key) || [], stage: cur()?.stage ?? null, scale, area,
        tiles: F ? Object.keys(F.tiles).length : 0, why: F?.why ?? null, next: F?.next ?? null,
        lm: !!F?.lm, every: !!F?.every, cell: CW, variant: F?.variant?.key ?? null,
      }),
    };
  }
}
