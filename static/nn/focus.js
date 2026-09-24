// Net tab lens (docs/NN_LENS.md): what a lens emphasizes, dims and hides. Pure: no DOM.
//
// emphasis(net, fwd, lens) tells the canvas, the matrix panel and the attention panel alike how
// strongly each neuron, edge and attention edge belongs to the lens's story (1 = emphasized,
// 0 = dimmed), and what `show` and the thresholds remove. lens.js owns state.lens and uses
// cleanLens, lensInfo and stepStage from here too; tests: tests/nn_focus.test.mjs.
//
// Layer arguments (l) are layer indices; layer ids are accepted too. Token and head numbers are
// 0-based. Anything invalid in the lens (a deleted layer, a token the net doesn't have) is
// ignored, exactly as cleanLens would drop it.

import { tokenShape, attnSpec } from './model.js';

const deepFreeze = o => {
  for (const v of Object.values(o)) if (v && typeof v === 'object') deepFreeze(v);
  return Object.freeze(o);
};
export const DEFAULT_LENS = deepFreeze({
  focus: null, token: null, head: null, show: { weights: true, attention: true, fixed: true }, minW: 0, minA: 0,
});
// Parts of an attention layer's focus, in the matrix panel's order (1 · scores, 2 · softmax, 3 · sum).
export const ATTN_PARTS = ['scores', 'softmax', 'mix'];

const isNum = v => typeof v === 'number' && Number.isFinite(v);
const isObj = v => !!v && typeof v === 'object' && !Array.isArray(v);
const clamp01 = v => Math.min(1, Math.max(0, v));

// A complete, independent copy (callers may mutate it).
export function copyLens(lens) {
  const L = isObj(lens) ? lens : DEFAULT_LENS;
  const s = isObj(L.show) ? L.show : DEFAULT_LENS.show;
  return {
    focus: isObj(L.focus) ? { ...L.focus } : null,
    token: L.token ?? null, head: L.head ?? null,
    show: { weights: s.weights !== false, attention: s.attention !== false, fixed: s.fixed !== false },
    minW: L.minW ?? 0, minA: L.minA ?? 0,
  };
}

// One pass over the net: per layer its token shape and (a working attention layer) its attnSpec,
// the Q, K, V layer before each attention layer, and per node { l, g, t, f }.
function scan(net) {
  const layers = Array.isArray(net?.layers) ? net.layers : [];
  const li = new Map();
  layers.forEach((l, i) => li.set(l?.id, i));
  const byLayer = layers.map(() => []);
  for (const n of Array.isArray(net?.nodes) ? net.nodes : []) {
    const l = li.get(n?.layer);
    if (l !== undefined) byLayer[l].push(n);
  }
  const shape = layers.map((_, l) => {
    const s = tokenShape(net, l);
    return { T: s.tokens, d: s.d, groups: s.groups, G: s.groups ? s.groups.length : 1 };
  });
  const att = layers.map((l, i) => (l?.kind === 'attention' ? attnSpec(net, i) : null));
  const qkv = layers.map((_, l) => {
    const a = att[l + 1], g = shape[l].groups;
    return a && g ? { a, qG: g.indexOf('Q'), kG: g.indexOf('K'), vG: g.indexOf('V') } : null;
  });
  const pos = new Map();
  byLayer.forEach((ns, l) => {
    const { T, d } = shape[l];
    ns.forEach((n, k) => pos.set(n.id, d ? { l, g: Math.floor(k / (T * d)), t: Math.floor(k / d) % T, f: k % d } : { l, g: 0, t: 0, f: k }));
  });
  let tokens = 0, heads = 1;
  for (const s of shape) if (s.T > 1) tokens = Math.max(tokens, s.T);
  for (const a of att) if (a) heads = Math.max(heads, a.heads);
  return { layers, li, byLayer, shape, att, qkv, pos, tokens, heads };
}

const layerName = (S, i) => {
  const l = S.layers[i], L = S.layers.length;
  return String(l?.name || (i === 0 ? 'Input' : i === L - 1 ? 'Output' : 'Hidden'));
};

// net.meta.tokenNames, cleaned: a trimmed string or null per entry. A slot holding its own
// default ('t2' for token 2) counts as unnamed, as in the attention panel and the cards.
export function tokenNames(net) {
  const a = net?.meta?.tokenNames;
  return Array.isArray(a) ? a.map((v, i) => {
    const s = typeof v === 'string' || isNum(v) ? String(v).trim() : '';
    return s && s !== `t${i + 1}` ? s : null;
  }) : [];
}
// Plain text for token t (0-based): its name, else t1, t2, ...
export function tokenLabel(net, t) {
  return tokenNames(net)[t] || `t${t + 1}`;
}

// The stage focuses [ and ] step through, in order: every layer after the input, with an
// attention layer split into its three parts.
function stagesOf(S) {
  const out = [];
  S.layers.forEach((l, i) => {
    if (i === 0) return;
    if (S.att[i]) for (const part of ATTN_PARTS) out.push({ layer: l.id, part });
    else out.push({ layer: l.id });
  });
  return out;
}
export function stages(net) { return stagesOf(scan(net)); }

// What the lens bar can offer for this net.
//   tokens: the most tokens of any token layer (0 when there is none: token does nothing),
//   heads: the most heads of any attention layer (1: head does nothing), names: tokenNames,
//   layers: [{ id, index, name, kind: 'dense' | 'qkv' | 'attention', parts }] (parts: the focus
//   parts it takes: its group names, or ATTN_PARTS), stages, hasWeights / hasFixed / hasAttention,
//   maxW: the largest |w| of a non-fixed edge.
export function lensInfo(net) {
  const S = scan(net);
  let hasWeights = false, hasFixed = false, maxW = 0;
  for (const e of Array.isArray(net?.edges) ? net.edges : []) {
    if (e.fixed) hasFixed = true;
    else {
      hasWeights = true;
      if (isNum(e.w)) maxW = Math.max(maxW, Math.abs(e.w));
    }
  }
  const layers = S.layers.map((l, i) => ({
    id: l.id, index: i, name: layerName(S, i),
    kind: S.att[i] ? 'attention' : S.qkv[i] ? 'qkv' : 'dense',
    parts: S.att[i] ? ATTN_PARTS.slice() : S.shape[i].groups ? S.shape[i].groups.slice() : [],
  }));
  return {
    tokens: S.tokens, heads: S.heads, names: tokenNames(net), layers, stages: stagesOf(S),
    hasWeights, hasFixed, hasAttention: S.att.some(Boolean), maxW,
  };
}

function cleanWith(S, lens) {
  const src = isObj(lens) ? lens : DEFAULT_LENS;
  let focus = null;
  if (isObj(src.focus)) {
    let ref = src.focus.layer;
    if (typeof ref === 'number' && Number.isInteger(ref) && S.layers[ref]) ref = S.layers[ref].id;
    const l = S.li.get(ref);
    if (l !== undefined) {
      const part = src.focus.part;
      const ok = S.att[l] ? ATTN_PARTS.includes(part) : !!S.shape[l].groups?.includes(part);
      focus = ok ? { layer: ref, part } : { layer: ref };
    }
  }
  const token = Number.isInteger(src.token) && src.token >= 0 && src.token < S.tokens ? src.token : null;
  const head = S.heads > 1 && Number.isInteger(src.head) && src.head >= 0 && src.head < S.heads ? src.head : null;
  const s = isObj(src.show) ? src.show : {};
  const show = { weights: s.weights !== false, attention: s.attention !== false, fixed: s.fixed !== false };
  const minW = isNum(src.minW) && src.minW > 0 ? src.minW : 0;
  const minA = isNum(src.minA) && src.minA > 0 ? Math.min(1, src.minA) : 0;
  const f0 = src.focus;
  const same = Object.keys(src).length === 6
    && (focus === null ? f0 === null : isObj(f0) && Object.keys(f0).length === ('part' in focus ? 2 : 1) && f0.layer === focus.layer && f0.part === focus.part)
    && src.token === token && src.head === head && isObj(src.show) && Object.keys(src.show).length === 3
    && src.show.weights === show.weights && src.show.attention === show.attention && src.show.fixed === show.fixed
    && src.minW === minW && src.minA === minA;
  return same ? src : { focus, token, head, show, minW, minA };
}
// The lens with every field that no longer makes sense for this net reset (a focus on a deleted
// layer or a part that layer doesn't have, a token or head the net doesn't have, junk values).
// Returns the same object when there was nothing to fix; null gives DEFAULT_LENS.
export function cleanLens(net, lens) {
  return cleanWith(scan(net), lens);
}

// The next (dir 1) or previous (dir -1) stage focus from `focus`: from none, the first or the
// last; past either end, null. A focus that isn't a stage (a Q, K or V part, a whole attention
// layer, the input layer) steps from where it sits: from the input, ] gives the first stage and
// [ null.
export function stepStage(net, focus, dir) {
  const S = scan(net), list = stagesOf(S);
  if (!list.length) return null;
  if (!isObj(focus)) return dir > 0 ? list[0] : list[list.length - 1];
  let pos = list.findIndex(s => s.layer === focus.layer && s.part === focus.part);
  if (pos < 0) {
    const first = list.findIndex(s => s.layer === focus.layer), l = S.li.get(focus.layer);
    if (first >= 0) pos = first + (focus.part ? 0.5 : -0.5);
    else if (l === undefined) return dir > 0 ? list[0] : list[list.length - 1];
    else {
      const after = list.findIndex(s => S.li.get(s.layer) > l);
      pos = (after < 0 ? list.length : after) - 0.5;
    }
  }
  const next = dir > 0 ? Math.floor(pos) + 1 : Math.ceil(pos) - 1;
  return list[next] || null;
}

// ================================================================ emphasis
export function emphasis(net, fwd, lens) {
  const S = scan(net), L = cleanWith(S, lens);
  const edges = Array.isArray(net?.edges) ? net.edges : [];
  const lix = l => (typeof l === 'number' ? l : S.li.get(l));
  const Aof = (l, h, i, j) => fwd?.attn?.[l]?.heads?.[h]?.A?.[i]?.[j];
  const edgeById = new Map();
  for (const e of edges) edgeById.set(e.id, e);

  // ---- hidden (show, thresholds): independent of the emphasis
  const hides = !L.show.weights || !L.show.attention || !L.show.fixed || L.minW > 0 || L.minA > 0;
  const hideEdge = id => {
    const e = edgeById.get(id);
    if (!e) return false;
    if (e.fixed) return !L.show.fixed;
    return !L.show.weights || (L.minW > 0 && isNum(e.w) && Math.abs(e.w) < L.minW);
  };
  // Causally masked pairs (j > i) have no attention edge at all: always hidden.
  const hideAttn = (l, i, j, h) => {
    const a = S.att[lix(l)];
    if (!a) return false;
    if (!L.show.attention || (a.causal && j > i)) return true;
    const v = Aof(lix(l), h, i, j);
    return L.minA > 0 && isNum(v) && v < L.minA;
  };
  const hidden = { edge: hideEdge, attn: hideAttn };

  const tok = L.token, head = L.head;
  if (!L.focus && tok === null && head === null) {
    return {
      any: false, hides, lens: L, hidden,
      node: () => 1, edge: () => 1, attn: () => 1, layer: () => 1,
      rows: () => null, heads: () => null, groups: () => null,
    };
  }

  // ---- token and head: which layers they apply to
  const tokOn = l => tok !== null && S.shape[l]?.T > 1 && tok < S.shape[l].T;
  const headSpec = l => S.att[l] || S.qkv[l]?.a || null;          // an attention layer, or the Q, K, V layer it reads
  const headOn = l => head !== null && !!headSpec(l) && head < headSpec(l).heads;

  // ---- focus: layer fl (and a part), the layers feeding it, the groups kept
  let fl = -1, keep = null, prevKeep = null, attnFocus = true;
  const feed = new Set();
  if (L.focus) {
    fl = S.li.get(L.focus.layer);
    const part = L.focus.part;
    if (S.att[fl]) {
      const q = S.qkv[fl - 1];
      feed.add(fl - 1);
      prevKeep = part === 'scores' || part === 'softmax' ? new Set([q.qG, q.kG]) : part === 'mix' ? new Set([q.vG]) : null;
      attnFocus = !part || part === 'mix';
    } else {
      const g = part ? S.shape[fl].groups.indexOf(part) : -1;
      keep = g >= 0 ? new Set([g]) : null;
      attnFocus = false;
      for (const e of edges) {
        const pf = S.pos.get(e.from), pt = S.pos.get(e.to);
        if (pf && pt && pt.l === fl && (!keep || keep.has(pt.g))) feed.add(pf.l);
      }
    }
  }
  const focusLayer = l => fl < 0 || l === fl || feed.has(l);
  const focusNode = p => {
    if (fl < 0) return 1;
    if (p.l === fl) return !keep || keep.has(p.g) ? 1 : 0;
    if (!feed.has(p.l)) return 0;
    return prevKeep && p.l === fl - 1 && !prevKeep.has(p.g) ? 0 : 1;
  };
  const focusEdge = (pf, pt) => (fl < 0 ? 1 : pt.l === fl && !S.att[fl] && (!keep || keep.has(pt.g)) ? 1 : 0);
  // Token t: token t of every token layer. A key or value token j of the Q, K, V layer gets A_tj
  // (the head of its feature), so what t attends to stays visible; unknown A (no forward pass) 1.
  const tokenNode = p => {
    if (!tokOn(p.l)) return 1;
    const q = S.qkv[p.l];
    if (q && (p.g === q.kG || p.g === q.vG)) {
      const v = Aof(p.l + 1, Math.floor(p.f / q.a.dh), tok, p.t);
      return isNum(v) ? clamp01(v) : 1;
    }
    return p.t === tok ? 1 : 0;
  };
  // Head h: features [h·dh, (h+1)·dh) of every Q, K, V group and of Z.
  const headNode = p => (headOn(p.l) && Math.floor(p.f / headSpec(p.l).dh) !== head ? 0 : 1);

  const nodeE = new Map(), edgeE = new Map(), layerE = S.layers.map((_, l) => (S.byLayer[l].length ? 0 : focusLayer(l) ? 1 : 0));
  let any = false;
  for (const [id, p] of S.pos) {
    const v = Math.min(focusNode(p), tokenNode(p), headNode(p));
    nodeE.set(id, v);
    if (v < 1) any = true;
    if (v > layerE[p.l]) layerE[p.l] = v;
  }
  for (const e of edges) {
    const pf = S.pos.get(e.from), pt = S.pos.get(e.to);
    const v = pf && pt ? Math.min(focusEdge(pf, pt), tokenNode(pf), tokenNode(pt), headNode(pf), headNode(pt)) : 1;
    edgeE.set(e.id, v);
    if (v < 1) any = true;
  }
  const attnE = (l, i, j, h) => {
    if (!S.att[l]) return 1;
    if (fl >= 0 && !(fl === l && attnFocus)) return 0;
    if (tokOn(l) && i !== tok) return 0;
    if (headOn(l) && h !== head) return 0;
    return 1;
  };
  if (!any) {
    S.att.forEach((a, l) => {
      if (!a || any) return;
      for (let h = 0; h < a.heads && !any; h++) for (let i = 0; i < a.tokens && !any; i++) {
        for (let j = 0; j < a.tokens; j++) if (!(a.causal && j > i) && attnE(l, i, j, h) < 1) { any = true; break; }
      }
    });
  }

  const valid = l => Number.isInteger(l) && l >= 0 && l < S.layers.length;
  return {
    any, hides, lens: L, hidden,
    node: id => nodeE.get(id) ?? 1,
    edge: id => edgeE.get(id) ?? 1,
    attn: (l, i, j, h) => (valid(lix(l)) ? attnE(lix(l), i, j, h) : 1),
    layer: l => (valid(lix(l)) ? layerE[lix(l)] : 1),
    rows: l => {
      l = lix(l);
      if (!valid(l)) return null;
      if (layerE[l] === 0) return new Set();
      return tokOn(l) ? new Set([tok]) : null;
    },
    heads: l => (valid(lix(l)) && headOn(lix(l)) ? new Set([head]) : null),
    groups: l => {
      l = lix(l);
      if (!valid(l)) return null;
      if (layerE[l] === 0) return new Set();
      if (l === fl && keep) return new Set(keep);
      if (l === fl - 1 && prevKeep) return new Set(prevKeep);
      return null;
    },
  };
}
