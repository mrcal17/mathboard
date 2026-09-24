// Net tab: Explain, a guided walkthrough (docs/NN_LENS.md: state.tour).
//
// The Explain button or E starts it; → / PageDown and ← / PageUp move, Esc (or E) ends it. Esc is
// caught in the capture phase, so it ends the tour before the shell's Esc deselects anything.
//
// Steps come from the net's structure (buildSteps, pure). A token net with attention gets the
// transformer story: the tokens, Q, K, V, the scores, the softmax, the weighted sum, one step per
// layer after attention (residual, FFN, output), then one token followed through the block. A plain
// net gets the layer-by-layer z = W a + b story, ending on collapse (all linear) or on why the
// activations matter. Each step sets state.lens (focus, token, head), state.viz (the attention
// panel) and state.anim (the step-through) to show what its caption says, then state.tour =
// { i, n, title, text }. The caption card renders state.tour alone, so the audience window shows
// it from the mirror. Ending the tour puts lens, viz and anim back as they were. While it runs an
// open Train panel is folded to its header (ctx.train.fold), so the net has room beside the
// attention panel; the end unfolds it.
//
// Captions quote the current forward pass and are re-read (at most every REFRESH_MS) whenever it
// changes, so a tour over a net that is training keeps up with it.

import { DEFAULT_LENS, stages as lensStages, tokenNames } from './focus.js';

const REFRESH_MS = 150;
const MAX_ROWS = 3;    // rows of a matrix quoted in a caption before "…"
const MAX_COLS = 4;    // entries of a vector before \dots
const MAX_TERMS = 4;   // terms of a written-out sum before it is shortened
const ACT_TEX = {
  relu: '\\mathrm{ReLU}', leaky: '\\mathrm{LReLU}', sigmoid: '\\sigma', tanh: '\\tanh', softmax: '\\mathrm{softmax}',
};
const ACT_WORD = { relu: 'ReLU', leaky: 'leaky ReLU', sigmoid: 'the sigmoid', tanh: 'tanh', softmax: 'softmax' };

// ================================================================ text helpers (pure)

const plainSafe = s => String(s).replace(/\$/g, '＄');   // a $ in a token name must not open maths
const cap = s => (s ? s[0].toUpperCase() + s.slice(1) : s);
const joinAnd = xs => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
const isNum = v => typeof v === 'number' && Number.isFinite(v);

function kit(M) {
  const f = x => (isNum(x) ? M.fmt(x, 2) : x === -Infinity ? '-\\infty' : x === Infinity ? '\\infty' : '?');
  const par = x => { const s = f(x); return s.startsWith('-') ? `(${s})` : s; };
  // (a,\ b, …): a row of numbers, the middle elided past MAX_COLS entries. Up to twice that, not
  // when the middle holds every entry that isn't 0 (a one-hot input would read as all zeros).
  const zero = x => isNum(x) && Math.abs(x) < 0.005;
  const vec = row => {
    const r = Array.isArray(row) ? row : [];
    if (r.length === 1) return f(r[0]);   // a 1-vector reads as its number
    const shown = [...r.slice(0, MAX_COLS - 1), r.at(-1)];
    const elide = r.length > MAX_COLS && !(r.length <= 2 * MAX_COLS && shown.every(zero) && !r.every(zero));
    const xs = elide ? [...r.slice(0, MAX_COLS - 1).map(f), '\\dots', f(r.at(-1))] : r.map(f);
    return `(${xs.join(',\\ ')})`;
  };
  // a b, juxtaposed when b is a bracketed negative
  const prod = (a, b) => { const pb = par(b); return pb.startsWith('(') ? `${par(a)}${pb}` : `${par(a)} \\cdot ${pb}`; };
  return { f, par, vec, prod };
}

// Token names (net.meta.tokenNames, read by focus.js's rule): plain text for captions, 1-based fallbacks.
function tokenWords(net) {
  const all = tokenNames(net), nm = t => all[t] ?? null;
  const names = all.some(Boolean) ? all : null;
  return {
    names,
    tok: t => (nm(t) ? `“${plainSafe(nm(t))}”` : `token ${t + 1}`),
    Tok: t => (nm(t) ? `“${plainSafe(nm(t))}”` : `Token ${t + 1}`),
    list: T => (names ? ` (${Array.from({ length: T }, (_, t) => plainSafe(nm(t) ?? `token ${t + 1}`)).join(', ')})` : ''),
  };
}

// The bias name of a group of nodes: its tie ('b_Q:2' -> b_Q), or b when untied, or null when all 0.
function biasName(nodes, fallback = 'b') {
  const tie = nodes.find(n => typeof n.tie === 'string' && n.tie)?.tie;
  if (tie) { const c = tie.lastIndexOf(':'); return `{${c > 0 ? tie.slice(0, c) : tie}}`; }
  return nodes.some(n => isNum(n.bias) && n.bias !== 0) ? fallback : null;
}

// Row t (1-based) of a matrix symbol: X -> x_{t}, \hat Y -> \hat y_{t}, H^{(3)} -> h^{(3)}_{t}.
function rowOf(sym, t) {
  if (sym === '\\hat Y') return `\\hat y_{${t}}`;
  const m = /^([A-Z])(\^\{\(\d+\)\})?$/.exec(sym);
  return m ? `${m[1].toLowerCase()}${m[2] || ''}_{${t}}` : `(${sym})_{${t}}`;
}

// Where the attention in row i of A goes: the row whose largest weight is largest, over rows that
// see more than one key (the first row of a causal layer sees only itself).
function pickRow(A, causal) {
  let best = null;
  (A || []).forEach((row, i) => {
    const vis = (row || []).map((a, j) => (causal && j > i ? null : a));
    if (vis.filter(isNum).length < 2 && A.length > 1) return;
    vis.forEach((a, j) => { if (isNum(a) && (!best || a > best.a + 1e-9)) best = { i, j, a }; });
  });
  return best;
}
const spread = (row, i, causal) => {
  const vis = (row || []).filter((a, j) => isNum(a) && !(causal && j > i));
  return vis.length ? Math.max(...vis) - Math.min(...vis) : 0;
};

// ================================================================ steps (pure)
// A step: { key, title(env), text(env), lens: { focus, token, head }, viz, anim, reveal, collapse }.
// env = { net, fwd, bwd }: the live net and forward pass when the caption is (re)read.
//
// The steps walk the lens's stages in order (focus.js stages(net): every layer after the input, an
// attention layer split into scores, softmax and mix), so each step's focus is one the lens bar and
// [ / ] know. opts.stages defaults to focus.js's stages; if it fails the same walk is made here.

export function buildSteps(net, M, fwd = null, { stages = lensStages } = {}) {
  if (!net?.layers?.length || !net.nodes?.length || !M) return [];
  const att = [];
  for (let l = 1; l < net.layers.length; l++) {
    if (net.layers[l].kind === 'attention' && M.attnSpec?.(net, l)) att.push(l);
  }
  let order = null;
  if (typeof stages === 'function') {
    try { order = stages(net); } catch { order = null; }
  }
  if (!Array.isArray(order) || !order.length) {
    order = [];
    net.layers.forEach((lay, l) => {
      if (l === 0) return;
      if (att.includes(l)) for (const part of ['scores', 'softmax', 'mix']) order.push({ layer: lay.id, part });
      else order.push({ layer: lay.id });
    });
  }
  order = order.map(s => ({ l: M.layerIndex(net, s?.layer), part: s?.part ?? null })).filter(s => s.l > 0);
  return att.length ? tokenStory(net, M, att, fwd, order) : denseStory(net, M, order);
}

// ---------------------------------------------------------------- transformer story
function tokenStory(net, M, att, fwd0, order) {
  const K = kit(M), { f, vec } = K;
  const L = net.layers.length, last = L - 1, lays = net.layers;
  const id = l => lays[l].id;
  const shape = l => M.tokenShape(net, l);
  const mats = l => { try { return M.tiedMatrices(net, l) || []; } catch { return []; } };
  const layerOfNode = nid => M.nodeLayerIndex(net, nid);
  const isQKV = l => att.includes(l + 1);
  const A0 = att[0];

  // Matrix symbols: X for the input, Z for attention, a layer named "H = ..." is H, the output is
  // \hat Y, a hidden nonlinear layer F (the FFN), anything else H^{(l)}.
  const sym = [], used = new Set();
  for (let l = 0; l < L; l++) {
    const lay = lays[l];
    let s;
    if (att.includes(l)) s = used.has('Z') ? `Z^{(${l})}` : 'Z';
    else if (isQKV(l)) s = (lay.groups || ['Q', 'K', 'V']).join(',');
    else if (l === 0) s = 'X';
    else {
      const m = /^\s*([A-Z])\s*=/.exec(lay.name || '');
      s = m ? m[1] : null;
      if (l === last && (!s || s === 'Y')) s = '\\hat Y';
      if (!s || used.has(s)) s = lay.act !== 'identity' && !used.has('F') ? 'F' : !used.has('H') ? 'H' : `H^{(${l})}`;
    }
    sym[l] = s;
    used.add(s);
  }
  const rowSym = (l, t) => rowOf(sym[l], t);
  const rowsOf = (env, l) => M.reshape(env.net, l, env.fwd?.a?.[l]);   // { X | Q, K, V: tokens x d }
  const words = env => tokenWords(env.net);
  const targetRow = (env, t) => {
    if (!env.bwd) return null;
    const outs = M.nodesIn(env.net, last), d = shape(last).d;
    const ys = outs.slice(t * d, t * d + d).map(n => n.target);
    return ys.length && ys.every(isNum) ? ys : null;
  };

  // How layer l (after attention) is made: residual sources, shared matrices, plain edges, bias.
  function recipe(l) {
    const ids = new Set(M.nodesIn(net, l).map(n => n.id));
    const fixed = new Set();
    let plain = false;
    for (const e of net.edges) {
      if (!ids.has(e.to)) continue;
      if (e.fixed) fixed.add(layerOfNode(e.from));
      else if (!(typeof e.tie === 'string' && e.tie)) plain = true;
    }
    const tm = plain ? [] : mats(l).filter(t => t.k != null);
    return { fixed: [...fixed].sort((a, b) => a - b), tm, plain, bias: biasName(M.nodesIn(net, l)), act: lays[l].act || 'identity' };
  }
  // "H = X + Z W_O + b_O", or row t of it ("h_1 = x_1 + z_1 W_O + b_O") when t is given.
  function formula(l, t = null) {
    const r = recipe(l), s = k => (t == null ? sym[k] : rowSym(k, t));
    const parts = [...r.fixed.map(s), ...r.tm.map(m => `${s(m.k)}${m.name.length > 1 ? `{${m.name}}` : m.name}`)];
    if (r.plain) parts.push(`W^{(${l})}a^{(${l - 1})}`);
    if (r.bias) parts.push(r.bias);
    const inner = parts.join(' + ') || '0';
    const fn = ACT_TEX[r.act];
    return `${s(l)} = ${r.act === 'identity' || !fn ? inner : `${fn}(${inner})`}`;
  }

  const steps = [];

  // tokens: the rows of X
  const qkv0 = mats(A0 - 1);
  const src = Number.isInteger(qkv0[0]?.k) ? qkv0[0].k : Math.max(0, A0 - 2);
  steps.push({
    key: 'tokens',
    title: () => `Tokens are the rows of $${sym[src]}$`,
    text: env => {
      const { tokens: T, d } = shape(src), X = rowsOf(env, src).X || [];
      const shown = X.slice(0, MAX_ROWS).map((r, t) => `$${rowSym(src, t + 1)} = ${vec(r)}$`);
      const more = T > MAX_ROWS ? `${shown.join(', ')}, …` : joinAnd(shown);
      return `$${sym[src]}$ has ${T} rows, one per token${words(env).list(T)}, and ${d} column${d === 1 ? '' : 's'}, one per feature. ` +
        `Here ${more}.`;
    },
    lens: { focus: { layer: id(src) } },
    reveal: { layer: id(src) },
  });

  // The stages in order: the Q, K, V layer's stage gives the Q, K and V steps, an attention layer's
  // parts give scores, softmax and mix (heads after mix); every other layer one step.
  const vizOf = (mode, A) => ({ mode, layer: id(A) });
  const done = new Set(), parts = new Map();
  const once = (key, fn) => { if (!done.has(key)) { done.add(key); steps.push(...fn()); } };
  for (const { l, part } of order) {
    if (isQKV(l)) once(`qkv:${l}`, () => qkvSteps(l + 1));
    else if (att.includes(l)) {
      if (!parts.has(l)) parts.set(l, attnSteps(l));
      const P = parts.get(l);
      for (const p of P[part] ? [part] : Object.keys(P)) once(`attn:${l}:${p}`, () => P[p]);
    } else if (l > src) once(`layer:${l}`, () => [layerStep(l)]);
  }
  steps.push(...followSteps());
  return steps;

  // Q = X W_Q, K = X W_K, V = X W_V: one step each, the lens on that group.
  function qkvSteps(A) {
    const Lq = A - 1, groups = lays[Lq].groups || ['Q', 'K', 'V'], tm = mats(Lq);
    const sh = shape(Lq), T = sh.tokens;
    const out = [];
    groups.forEach((g, gi) => {
      const m = tm.find(x => x.toGroup === g) || tm[gi] || null;
      const k = Number.isInteger(m?.k) ? m.k : src;
      const W = m ? (m.name.length > 1 ? `{${m.name}}` : m.name) : `W_${g}`;
      const nodes = M.nodesIn(net, Lq).slice(gi * T * sh.d, (gi + 1) * T * sh.d);
      const b = biasName(nodes, `b_${g}`);
      const eq = `${g} = ${sym[k]}${W}${b ? ` + ${b}` : ''}`;
      const lo = g.toLowerCase();
      const rowEq = t => `${lo}_{${t}} = ${rowSym(k, t)}${W}${b ? ` + ${b}` : ''}`;
      const size = m ? `${m.W.length}×${m.W[0]?.length ?? 0}` : '';
      const rows = env => rowsOf(env, Lq)[g] || [];
      const role = ['Queries', 'Keys', 'Values'][gi] || g;
      out.push({
        key: `qkv:${A}:${g}`,
        title: () => `${role}: $${eq}$`,
        text: env => {
          const R = rows(env), list = R.slice(0, 2).map((r, t) => `$${lo}_{${t + 1}} = ${vec(r)}$`);
          const tail = R.length > 2 ? `${list.join(', ')}, …` : joinAnd(list);
          if (gi === 0) {
            return `One shared ${size ? size + ' ' : ''}matrix turns every token into a query, what that token is looking for: ` +
              `$${rowEq(1)} = ${vec(R[0])}$. Shared (tied) means the same numbers for every token.`;
          }
          if (gi === 1) return `Keys are what each token offers to be found by, from a second shared matrix: ${tail}.`;
          if (gi === 2) {
            return `Values are what a token hands on once another token attends to it: ${tail}. ` +
              `$Q$, $K$ and $V$ are three views of the same rows of $${sym[k]}$.`;
          }
          return `$${eq}$: ${tail}.`;
        },
        lens: { focus: { layer: id(Lq), part: g } },
        reveal: { layer: id(Lq), part: g },
      });
    });
    return out;
  }

  // S = QK^T / sqrt(d_k), A = softmax(S), Z = AV (head 1 of a multi-head layer), then the other
  // heads and their concatenation.
  function attnSteps(A) {
    const sp = M.attnSpec(net, A), H = sp.heads, head = H > 1 ? 0 : null;
    const dflt = Math.abs(sp.scale - 1 / Math.sqrt(sp.dh)) < 1e-9;
    const Seq = dflt ? `S = QK^{\\top}/\\sqrt{d_k}` : `S = c\\,QK^{\\top}`;
    const sij = dflt ? `S_{ij} = q_i \\cdot k_j/\\sqrt{${sp.dh}}` : `S_{ij} = ${f(sp.scale)}\\; q_i \\cdot k_j`;
    const heads = env => env.fwd?.attn?.[A]?.heads || [];
    const hd = (env, h) => heads(env)[h] || null;
    const inHead = H > 1 ? 'In head 1, ' : '';
    const sup = h => (H > 1 ? `^{(${h + 1})}` : '');
    const out = { scores: [], softmax: [], mix: [] };   // by lens part; the heads follow mix
    out.scores.push({
      key: `attn:${A}:scores`,
      title: () => `Scores: $${Seq}$`,
      text: env => {
        const F = hd(env, 0), T = sp.tokens;
        const first = `Every query is scored against every key, $${sij}$, filling a ${T}×${T} table${H > 1 ? ' in each head' : ''}.`;
        if (sp.causal) return `${first} The causal mask sets $S_{ij} = -\\infty$ for $j > i$, so no token can see a later one.`;
        let best = null;
        (F?.S || []).forEach((row, i) => row.forEach((s, j) => { if (isNum(s) && (!best || s > best.s)) best = { i, j, s }; }));
        if (!best) return first;
        return `${first} ${cap(`${inHead}the best match is $S${sup(0)}_{${best.i + 1},${best.j + 1}} = ${f(best.s)}$: ` +
          `$q_{${best.i + 1}}$ lines up most with $k_{${best.j + 1}}$.`)}`;
      },
      lens: { focus: { layer: id(A), part: 'scores' }, head },
      viz: vizOf('dots', A),
      reveal: { layer: id(A), part: 'scores' },
    });
    out.softmax.push({
      key: `attn:${A}:softmax`,
      title: () => 'Attention: $A = \\mathrm{softmax}(S)$',
      text: env => {
        const F = hd(env, 0), w = words(env);
        const first = 'Softmax turns each row of $S$ into positive weights that sum to 1: row $i$ is how much token $i$ reads each token.';
        const p = pickRow(F?.A, sp.causal);
        if (!p) return first;
        const row = F.A[p.i];
        if (spread(row, p.i, sp.causal) < 0.1) {
          return `${first} ${inHead ? 'In head 1 the' : 'Untrained, the'} rows are still nearly even: row ${p.i + 1} is $${vec(row)}$.`;
        }
        return `${first} ${cap(`${inHead}${w.tok(p.i)} puts ${f(p.a)} of its attention on ${p.j === p.i ? 'itself' : w.tok(p.j)}: ` +
          `row ${p.i + 1} is $${vec(row)}$.`)}`;
      },
      lens: { focus: { layer: id(A), part: 'softmax' }, head },
      viz: vizOf('heat', A),
      reveal: { layer: id(A), part: 'softmax' },
    });
    out.mix.push({
      key: `attn:${A}:mix`,
      title: () => 'Mix: $Z = AV$',
      text: env => {
        const F = hd(env, 0);
        const first = 'Each $z_i$ is a weighted average of the value rows, with row $i$ of $A$ as the weights';
        const p = pickRow(F?.A, sp.causal);
        if (!p || !F?.V || !F?.Z) return `${first}.`;
        const i = p.i, vis = F.A[i].map((a, j) => ({ a, j })).filter(x => !(sp.causal && x.j > i));
        const terms = vis.map(x => `${f(x.a)}\\,v${sup(0)}_{${x.j + 1}}`);
        const sum = terms.length > MAX_TERMS ? `${terms.slice(0, 2).join(' + ')} + \\dots + ${terms.at(-1)}` : terms.join(' + ');
        return `${first}: $z${sup(0)}_{${i + 1}} = ${sum} = ${vec(F.Z[i])}$.${H > 1 ? ' (Head 1’s columns.)' : ''}`;
      },
      lens: { focus: { layer: id(A), part: 'mix' }, head },
      viz: vizOf('mix', A),
      reveal: { layer: id(A), part: 'mix' },
    });
    if (H > 1) {
      const cols = h => (sp.dh === 1 ? `column ${h * sp.dh + 1}` : `columns ${h * sp.dh + 1}–${(h + 1) * sp.dh}`);
      for (let h = 1; h < H; h++) {
        out.mix.push({
          key: `attn:${A}:head:${h}`,
          title: () => `Head ${h + 1}: its own $A${sup(h)}$`,
          text: env => {
            const F = hd(env, h), F0 = hd(env, 0), w = words(env);
            const first = `Head ${h + 1} runs the same three steps on ${cols(h)} of $Q$, $K$ and $V$, with its own attention matrix.`;
            const p = pickRow(F?.A, sp.causal);
            if (!p) return first;
            if (spread(F.A[p.i], p.i, sp.causal) < 0.1) return `${first} Its rows are still nearly even: row ${p.i + 1} is $${vec(F.A[p.i])}$.`;
            const j0 = F0?.A?.[p.i] ? F0.A[p.i].indexOf(Math.max(...F0.A[p.i].filter(isNum))) : -1;
            const vs = j0 >= 0 && j0 !== p.j ? `, where head 1 looks mostly at ${j0 === p.i ? 'itself' : w.tok(j0)}` : '';
            return `${first} Here ${w.tok(p.i)} puts ${f(p.a)} of its attention on ${p.j === p.i ? 'itself' : w.tok(p.j)}${vs}.`;
          },
          lens: { focus: { layer: id(A), part: 'softmax' }, head: h },
          viz: vizOf('heat', A),
          reveal: { layer: id(A), part: 'softmax' },
        });
      }
      const cat = Array.from({ length: H }, (_, h) => `Z${sup(h)}`).join('\\ ');
      out.mix.push({
        key: `attn:${A}:concat`,
        title: () => `Heads side by side: $Z = [\\,${cat}\\,]$`,
        text: () => `Each head writes its own ${sp.dh === 1 ? 'column' : `${sp.dh} columns`} of $Z$, so ${H} heads can look for ${H} different things at once. ` +
          'Every later layer reads all of them together.',
        lens: { focus: { layer: id(A), part: 'mix' } },
        viz: vizOf('mix', A),
        reveal: { layer: id(A), part: 'mix' },
      });
    }
    return out;
  }

  // A layer after attention: the residual projection, the FFN, the output, or a generic map.
  function layerStep(l) {
    const r = recipe(l), sh = shape(l), eq = formula(l);
    const fromAttn = r.tm.some(m => att.includes(m.k));
    const ffnIn = r.tm.find(m => lays[m.k]?.act !== 'identity' && !att.includes(m.k) && m.k > A0);
    let role = 'map';
    if (r.plain) role = 'dense';
    else if (fromAttn && r.fixed.length) role = 'residual';
    else if (r.act !== 'identity' && r.tm.length && l < last) role = 'ffn';
    else if (ffnIn) role = 'down';
    const wName = m => (m ? (m.name.length > 1 ? `{${m.name}}` : m.name) : 'W');
    const main = r.tm[0] || null;
    const dIn = main ? main.W.length : shape(l - 1).d, dOut = sh.d;
    const resid = r.fixed.map(k => `$+${sym[k]}$`).join(' and ');
    const outRow = env => {
      const R = rowsOf(env, l).X || [];
      const y = l === last ? targetRow(env, 0) : null;
      return `$${formula(l, 1)} = ${vec(R[0])}$${y ? `, target $y_{1} = ${vec(y)}$` : ''}`;
    };
    const title = {
      residual: () => `Project and add: $${eq}$`,
      ffn: () => 'FFN: the same small MLP on every token',
      down: () => `${l === last ? 'Output' : 'Back down'}: $${eq}$`,
      dense: () => `Layer ${l}: $a^{(${l})} = ${ACT_TEX[r.act] ? `${ACT_TEX[r.act]}(W^{(${l})}a^{(${l - 1})} + b)` : `W^{(${l})}a^{(${l - 1})} + b`}$`,
      map: () => `${l === last ? 'Output: ' : ''}$${eq}$`,
    }[role];
    const text = env => {
      const R = rowsOf(env, l).X || [];
      if (role === 'residual') {
        return `$${wName(r.tm.find(m => att.includes(m.k)))}$ projects each row of $${sym[r.tm.find(m => att.includes(m.k)).k]}$, and the fixed ${resid} edges add each token’s own row back (the residual). ` +
          `So ${outRow(env)}.`;
      }
      if (role === 'ffn') {
        const on = r.act === 'relu' || r.act === 'leaky' ? (R[0] || []).filter(v => v > 0).length : null;
        return `$${eq}$ widens every token from ${dIn} to ${dOut} features with one shared $${wName(main)}$; no token looks at another here. ` +
          (on == null ? `${cap(words(env).tok(0))}: ${outRow(env)}.` : `For ${words(env).tok(0)}, ${on} of the ${dOut} units are on: ${outRow(env)}.`);
      }
      if (role === 'down') {
        return `$${wName(ffnIn)}$ brings every token back to ${dOut} feature${dOut === 1 ? '' : 's'}${resid ? `, and the fixed ${resid} edges add the residual` : ''}: ${outRow(env)}.`;
      }
      if (role === 'dense') {
        const n = M.nodesIn(net, l).length, m = M.nodesIn(net, l - 1).length;
        return `Its ${n}×${m} matrix $W^{(${l})}$ connects every feature of every token, so this layer mixes the tokens directly: ` +
          `$a^{(${l})} = ${vec(env.fwd?.a?.[l])}$.`;
      }
      const act = r.act !== 'identity' ? `, then ${ACT_WORD[r.act] || r.act}` : '';
      return `The same ${main ? `${main.W.length}×${main.W[0]?.length ?? 0} ` : ''}matrix $${wName(main)}$ is applied to every token${act}${resid ? `, plus ${resid}` : ''}: ${outRow(env)}.`;
    };
    return { key: `layer:${l}:${role}`, title, text, lens: { focus: { layer: id(l) } }, reveal: { layer: id(l) } };
  }

  // One token followed through the block: its row of scores and A, its mix, then (when layers
  // follow attention) the rest of the way to the output.
  function followSteps() {
    const A = A0, sp = M.attnSpec(net, A), T = sp.tokens;
    const p = pickRow(fwd0?.attn?.[A]?.heads?.[0]?.A, sp.causal);
    const t = p ? p.i : T - 1;
    const out = [];
    const after = [];
    for (let l = A + 1; l <= last; l++) if (!isQKV(l) && !att.includes(l)) after.push(l);
    out.push({
      key: `follow:${t}`,
      title: env => `Follow ${words(env).tok(t)}`,
      text: env => {
        const F = env.fwd?.attn?.[A]?.heads?.[0], w = words(env);
        if (!F?.S || !F?.A) return `${w.Tok(t)} scores every key, softmax turns the scores into weights and $z_{${t + 1}}$ mixes the values with them.`;
        const Srow = F.S[t], Arow = F.A[t];
        const vis = Arow.map((a, j) => ({ a, j })).filter(x => !(sp.causal && x.j > t) && isNum(x.a));
        const top = vis.reduce((b, x) => (!b || x.a > b.a ? x : b), null);
        const h = sp.heads > 1 ? ' (head 1)' : '';
        const first = `${w.Tok(t)} scores every key, $S_{${t + 1},\\cdot} = ${vec(Srow)}$, and softmax makes that $A_{${t + 1},\\cdot} = ${vec(Arow)}$${h}.`;
        const end = after.length ? '' : ` That is this net’s output for ${w.tok(t)}.`;
        if (!top) return first;
        if (spread(Arow, t, sp.causal) < 0.1) return `${first} So $z_{${t + 1}}$ is still close to the plain average of the values.${end}`;
        return `${first} So ${f(top.a)} of $z_{${t + 1}}$ comes from ${top.j === t ? 'its own value' : `the value of ${w.tok(top.j)}`}.${end}`;
      },
      lens: { token: t },
      viz: vizOf('arcs', A),
      anim: { dir: 'fwd', l: A, i: t * sp.d, phase: 'sum' },
      reveal: { layer: id(A), part: 'mix', token: t },
    });
    if (after.length) {
      const chain = [src, A, ...after].map(l => rowSym(l, t + 1)).join(' \\to ');
      out.push({
        key: `follow:${t}:out`,
        title: env => `${cap(words(env).tok(t))}, through the rest of the block`,
        text: env => {
          const w = words(env), y = rowsOf(env, last).X?.[t], tg = targetRow(env, t);
          return `After attention, the row of ${w.tok(t)} only meets shared weights: $${chain} = ${vec(y)}$${tg ? `, target $${vec(tg)}$` : ''}. ` +
            `Other tokens reach it only through $A$.`;
        },
        lens: { token: t },
        reveal: { layer: id(last), token: t },
      });
    }
    return out;
  }
}

// ---------------------------------------------------------------- plain (dense) story
function denseStory(net, M, order) {
  const K = kit(M), { f, vec, prod } = K;
  const L = net.layers.length, last = L - 1, lays = net.layers;
  const id = l => lays[l].id;
  const size = l => M.nodesIn(net, l).length;
  const aSym = l => (l === 0 ? 'x' : l === last ? '\\hat y' : `a^{(${l})}`);
  const steps = [];
  if (!size(0)) return steps;

  steps.push({
    key: 'input',
    title: () => 'The input: $x = a^{(0)}$',
    text: env => `The net reads one vector, $x = ${vec(env.fwd?.a?.[0] ?? M.nodesIn(env.net, 0).map(n => n.value))}$. ` +
      'Each layer multiplies the vector before it by a matrix, adds a bias and applies an activation.',
    lens: { focus: { layer: id(0) } },
    reveal: { layer: id(0) },
  });

  let rowShown = false;
  for (const l of [...new Set(order.map(s => s.l))]) {   // the lens's stages: every layer after the input
    const n = size(l), m = size(l - 1);
    if (!n) continue;
    const ids = new Set(M.nodesIn(net, l).map(q => q.id));
    const into = net.edges.filter(e => ids.has(e.to));
    const skips = [...new Set(into.map(e => M.nodeLayerIndex(net, e.from)).filter(k => k < l - 1))].sort((a, b) => b - a);
    const act = lays[l].act || 'identity', fn = ACT_TEX[act];
    const terms = [`W^{(${l})}${aSym(l - 1)}`, ...skips.map(k => `W^{(${l},${k})}${aSym(k)}`), `b^{(${l})}`].join(' + ');
    const eq = `${aSym(l)} = ${act === 'identity' || !fn ? terms : `${fn}(${terms})`}`;
    steps.push({
      key: `layer:${l}`,
      title: () => `${l === last ? 'Output' : `Layer ${l}`}: $${eq}$`,
      text: env => {
        if (!into.length) return `No edges come into this layer yet, so $${aSym(l)}$ is just $${fn && act !== 'identity' ? `${fn}(b^{(${l})})` : `b^{(${l})}`}$.`;
        const outs = M.nodesIn(env.net, l), ys = outs.map(q => q.target);
        const target = l === last && env.bwd && ys.every(isNum) ? `, against the target $y = ${ys.length === 1 ? f(ys[0]) : vec(ys)}$` : '';
        const val = env.fwd?.a?.[l], valTex = n === 1 ? f(val?.[0]) : vec(val);
        const how = act === 'identity' ? '' : act === 'softmax' ? '; softmax then turns the scores into probabilities'
          : `; ${ACT_WORD[act] || act} then bends ${n === 1 ? 'it' : 'each entry'}`;
        const skip = skips.length ? ` A skip term adds $${skips.map(k => `W^{(${l},${k})}${aSym(k)}`).join(' + ')}$ straight from an earlier layer.` : '';
        return `$W^{(${l})}$ is ${n}×${m}, one row per neuron, so $W^{(${l})}${aSym(l - 1)}$ is ${n === 1 ? 'one dot product' : `${n} dot products at once`}${how}: ` +
          `$${aSym(l)} = ${valTex}$${target}.${skip}`;
      },
      lens: { focus: { layer: id(l) } },
      reveal: { layer: id(l) },
    });
    // One row with its numbers, once: the first layer that has edges coming in.
    if (!rowShown && into.length) {
      rowShown = true;
      const row0 = M.nodesIn(net, l)[0].id;
      steps.push({
        key: `row:${l}`,
        title: () => `One row: neuron 1 of ${l === last ? 'the output' : `layer ${l}`}`,
        text: env => {
          const nd = M.node(env.net, row0), fw = env.fwd?.node?.[row0];
          const ins = env.net.edges.filter(e => e.to === row0 && M.nodeLayerIndex(env.net, e.from) === l - 1)
            .map(e => ({ w: e.w, a: env.fwd?.node?.[e.from]?.a, j: M.nodesIn(env.net, l - 1).findIndex(q => q.id === e.from) }))
            .sort((p, q) => p.j - q.j);
          const b = nd?.bias ?? 0;
          const sum = ins.length <= MAX_TERMS
            ? [...ins.map(x => prod(x.w, x.a)), b < 0 ? `- ${f(-b)}` : f(b)].join(' + ').replace(/\+ - /g, '- ')
            : `w_{1} \\cdot ${aSym(l - 1)} + b_{1}`;
          const zt = `z_{1} = ${sum} = ${f(fw?.z)}`;
          const at = act === 'identity' ? `and $a_{1} = z_{1}$` : act === 'softmax'
            ? `and softmax over the layer gives $a_{1} = ${f(fw?.a)}$` : `and $a_{1} = ${fn || act}(${f(fw?.z)}) = ${f(fw?.a)}$`;
          return `Row 1 of $W^{(${l})}$ dotted with $${aSym(l - 1)}$, plus the bias: $${zt}$, ${at}. The matrix panel shows the same row.`;
        },
        lens: { focus: { layer: id(l) } },
        anim: { dir: 'fwd', l, i: 0, phase: 'dot' },
        reveal: { layer: id(l) },
      });
    }
  }

  if (L > 2) {
    const linear = lays.slice(1).every(q => (q.act || 'identity') === 'identity');
    const prodTex = Array.from({ length: L - 1 }, (_, k) => `W^{(${L - 1 - k})}`).join('');
    if (linear) {
      steps.push({
        key: 'collapse',
        title: () => 'Linear layers collapse',
        text: env => {
          let c = null;
          try { c = M.collapse(env.net); } catch { /* shown without numbers */ }
          const nested = L === 3 ? `W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)}` : `${prodTex}x + \\dots`;
          const W = c?.W, small = W && W.length <= 3 && (W[0]?.length ?? 9) <= 3;
          const shown = small ? ` = \\begin{bmatrix} ${W.map(r => r.map(f).join(' & ')).join(' \\\\ ')} \\end{bmatrix}` : '';
          const dims = W ? `${W.length}×${W[0]?.length ?? 0}` : '';
          return `No layer has an activation, so $\\hat y = ${nested} = W_{\\text{eff}}x + b_{\\text{eff}}$: one ${dims} matrix does the whole net${small ? `, $W_{\\text{eff}}${shown}$` : ''}. ` +
            'Depth adds nothing without a nonlinearity.';
        },
        lens: {},
        collapse: true,
      });
    } else {
      const acts = [...new Set(lays.slice(1).map(q => q.act).filter(a => a && a !== 'identity'))].map(a => ACT_WORD[a] || a);
      steps.push({
        key: 'why',
        title: () => 'Why the activations matter',
        text: () => `Take away ${joinAnd(acts)}, and the layers collapse into one matrix, $${prodTex}$, which can only draw straight boundaries. ` +
          'The bends between the matrices are what depth buys.',
        lens: {},
      });
    }
  }
  return steps;
}

// ================================================================ caption card + runner (DOM)

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
const escHtml = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Plain text with $…$ KaTeX.
function richHTML(s) {
  const k = globalThis.katex;
  return String(s ?? '').split(/(\$[^$]*\$)/g).map(p => {
    if (p.length >= 2 && p[0] === '$' && p.at(-1) === '$') {
      const src = p.slice(1, -1);
      if (k) try { return k.renderToString(src, { throwOnError: false, strict: 'ignore' }); } catch { /* as text */ }
      return escHtml(src);
    }
    return escHtml(p);
  }).join('');
}

export function install(ctx) {
  const { store } = ctx;
  const M = ctx.model || store.model;
  const stage = ctx.el?.stage || document.getElementById('nn-stage');
  if (!stage) throw new Error('tour: #nn-stage missing');
  const audience = !!ctx.audience;
  const steps = () => buildSteps(store.net, M, store.state.fwd);

  // ---------------------------------------------------------------- the card
  const card = el('div', 'nn-tour');
  card.hidden = true;
  card.setAttribute('role', 'status');
  card.setAttribute('aria-live', 'polite');
  const count = el('span', 'nn-tour-count');
  const title = el('span', 'nn-tour-title');
  const nav = el('span', 'nn-tour-nav');
  const btn = (html, tip, fn, cls = '') => {
    const b = el('button', `nn-tour-btn ${cls}`.trim());
    b.type = 'button';
    b.innerHTML = html;
    b.title = tip;
    b.addEventListener('mousedown', e => e.preventDefault());   // never take focus (Space = training)
    b.addEventListener('click', fn);
    nav.append(b);
    return b;
  };
  const prevBtn = btn('&#8592;', 'Back (← / PageUp)', () => go(-1));
  const nextBtn = btn('&#8594;', 'Next (→ / PageDown)', () => go(1), 'next');
  btn('&times;', 'End the walkthrough (Esc)', () => end(), 'close');
  const head = el('div', 'nn-tour-head');
  head.append(count, title, nav);
  const text = el('div', 'nn-tour-text');
  const bar = el('div', 'nn-tour-bar');
  const fill = el('i');
  bar.append(fill);
  card.append(head, text, bar);
  for (const ev of ['pointerdown', 'mousedown', 'click', 'dblclick', 'wheel']) card.addEventListener(ev, e => e.stopPropagation());
  stage.append(card);

  let shown = { title: null, text: null };
  function render() {
    const t = store.state.tour;
    toolBtn?.classList.toggle('on', !!t);
    if (!t) { card.hidden = true; return; }
    const wasHidden = card.hidden;
    card.hidden = false;
    const n = Math.max(1, t.n | 0), i = Math.min(n - 1, Math.max(0, t.i | 0));
    count.textContent = `${i + 1} / ${n}`;
    if (shown.title !== t.title) { shown.title = t.title; title.innerHTML = richHTML(t.title); }
    if (shown.text !== t.text) { shown.text = t.text; text.innerHTML = richHTML(t.text); }
    fill.style.width = `${((i + 1) / n) * 100}%`;
    prevBtn.disabled = i === 0;
    const lastStep = i === n - 1;
    nextBtn.innerHTML = lastStep ? 'Done' : '&#8594;';
    nextBtn.title = lastStep ? 'End the walkthrough (→ or Esc)' : 'Next (→ / PageDown)';
    nextBtn.classList.toggle('done', lastStep);
    if (wasHidden || shown.i !== i) placeSoon();   // again once the step's panels and the view settle
    shown.i = i;
  }

  // ---------------------------------------------------------------- placement
  // The card runs along the bottom or the top edge of the stage, wherever it hides the least:
  // the neurons it would cover (those of the focused layers count 4) and the layer headers (2, the
  // focused one 8), the floating panels by the share of them covered (the attention panel, which
  // the steps talk about, counts double, and its header 3 more; a folded Train panel, just a
  // header, little), then the net's box. Ties go to the bottom centre at full width; a narrower card (taller, a small
  // penalty) is tried too, to fit between two panels.
  const PAD = 12, STEP_X = 24, NARROW = 0.4;
  function place() {
    if (card.hidden) return;
    const sw = stage.clientWidth, sh = stage.clientHeight;
    if (!sw || !sh) return;
    const full = Math.round(Math.min(640, Math.max(380, sw * 0.62)));
    const widths = [...new Set([full, Math.max(340, full - 120), Math.max(340, full - 240)])];
    const sr = stage.getBoundingClientRect();
    const rectOf = e => { const r = e.getBoundingClientRect(); return { x: r.left - sr.left, y: r.top - sr.top, w: r.width, h: r.height }; };
    const panels = [];
    const visibleEl = e => !e.hidden && e.offsetWidth > 0 && e.offsetHeight > 0 && getComputedStyle(e).display !== 'none';
    const viz = ctx.attnviz?.el || null;
    for (const c of stage.children) {
      if (c === card || c.tagName === 'svg' || !visibleEl(c)) continue;
      const r = rectOf(c);
      if (r.w * r.h > 0.8 * sw * sh) {   // an overlay layer (the inspector's): its children are the panels
        for (const k of c.children) if (visibleEl(k)) panels.push({ ...rectOf(k), k: 6 });
      } else panels.push({ ...r, k: c === viz || c.contains(viz) ? 12 : c.matches('.nn-train.folded') ? 1 : 6 });
    }
    const v = ctx.view, st = store.state;
    const focusIds = new Set();
    const f = st.lens?.focus?.layer;
    if (f != null) {
      const l = M.layerIndex(store.net, f);
      for (const k of [l, l - 1]) if (k >= 0) for (const q of M.nodesIn(store.net, k)) focusIds.add(q.id);
    }
    const nodes = [];
    if (v?.nodeRect) for (const q of store.net.nodes) { const r = v.nodeRect(q.id); if (r) nodes.push({ r, w: focusIds.has(q.id) ? 4 : 1 }); }
    // layer headers count like two neurons (the focused layer's like two focused ones)
    for (const hd of v?.svg?.querySelectorAll?.('.nnv-head') || []) {
      const r = rectOf(hd);
      if (r.w && r.h) nodes.push({ r, w: hd.dataset.id === String(f) ? 8 : 2 });
    }
    // the attention panel's header (its mode buttons and ×) counts on top of the panel's share
    const vizHead = viz && visibleEl(viz) ? viz.querySelector('.na-head') : null;
    if (vizHead) nodes.push({ r: rectOf(vizHead), w: 3 });
    const net = v?.contentRect?.() || null;
    const ov = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    let best = null;
    widths.forEach((W, level) => {
      card.style.width = `${W}px`;
      const w = card.offsetWidth, hh = card.offsetHeight;
      const cx = Math.round((sw - w) / 2), top = PAD, bottom = Math.max(PAD, sh - hh - PAD), right = Math.max(PAD, sw - w - PAD);
      for (const y of [bottom, top]) {
        for (const x of [cx, ...Array.from({ length: Math.ceil((right - PAD) / STEP_X) }, (_, k) => PAD + k * STEP_X), right]) {
          const box = { x, y, w, h: hh };
          // tie-breaks: full width, then the bottom, then the centre
          let cost = level * NARROW + (y === top && top !== bottom ? 0.15 : 0) + (0.3 * Math.abs(x - cx)) / sw;
          for (const q of nodes) if (ov(box, q.r) > 0) cost += q.w;
          for (const p of panels) cost += (p.k * ov(box, p)) / Math.max(1, Math.min(w * hh, p.w * p.h));
          if (net) cost += (0.5 * ov(box, net)) / Math.max(1, w * hh);
          if (!best || cost < best.cost - 1e-9) best = { x, y, W, cost };
        }
      }
    });
    card.style.width = `${best.W}px`;
    card.style.left = `${Math.max(PAD, best.x)}px`;
    card.style.top = `${Math.max(PAD, best.y)}px`;
  }
  let placeRaf = 0, placeTimer = 0;
  function placeSoon(again = true) {
    if (!placeRaf) placeRaf = requestAnimationFrame(() => { placeRaf = 0; place(); });
    clearTimeout(placeTimer);
    if (again) placeTimer = setTimeout(place, 420);   // after a view fit or a panel settles
  }
  if (typeof ResizeObserver === 'function') new ResizeObserver(() => placeSoon(false)).observe(stage);
  store.on('layout', () => placeSoon(false));
  store.on('lens', () => placeSoon(false));
  store.on('viz', () => placeSoon());
  ctx.onShow?.(on => { if (on) placeSoon(); });
  ctx.onTheme?.(() => { shown = { title: null, text: null }; render(); });
  store.on('tour', render);

  const api = { buildSteps: steps, get running() { return !!run; }, start, end, go, place };
  ctx.tour = api;
  let toolBtn = null;
  let run = null;
  if (audience) { render(); return api; }

  // ---------------------------------------------------------------- runner (presenter only)
  const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  const setIf = (k, v) => { if (!same(store.state[k], v)) store.set(k, v ?? null); };
  const storyKey = net => JSON.stringify(net.layers.map(l => [l.id, l.kind, l.act, l.heads, l.causal, l.tokens, l.groups]));
  const layerExists = lid => store.net.layers.some(l => l.id === lid);

  function start() {
    if (run || audience) return;   // the audience window only renders state.tour
    const list = steps();
    if (!list.length) { ctx.toast?.('Nothing to explain yet: add some neurons first'); return; }
    run = {
      steps: list, i: 0, key: storyKey(store.net), title: store.net.meta?.title ?? '',
      saved: { lens: store.state.lens ?? null, viz: store.state.viz ?? null, anim: store.state.anim ?? null },
      collapseOn: false, batchOff: false, trainFolded: false,
    };
    // The attention panel opens on several steps: fold an open Train panel to its header so the
    // net keeps room beside them (end() unfolds it), and fit the net to what is left.
    const tr = ctx.train;
    try { if (tr?.open && !tr.folded) { tr.fold(true); run.trainFolded = true; } }
    catch (err) { console.warn('[nn/tour] train:', err); }
    apply();
    requestAnimationFrame(() => { if (run) ctx.view?.fit?.(300); });
  }

  function go(d) {
    if (!run) return;
    const i = run.i + d;
    if (i >= run.steps.length) { end(); return; }
    if (i < 0) return;
    run.i = i;
    apply();
  }

  function apply() {
    const s = run.steps[run.i];
    const base = { ...DEFAULT_LENS, ...(run.saved.lens || {}) };
    setIf('lens', { ...base, focus: s.lens?.focus ?? null, token: s.lens?.token ?? null, head: s.lens?.head ?? null });
    setIf('viz', s.viz ?? null);
    const mx = ctx.matrix?.opt;
    try {
      if (mx && typeof ctx.matrix.toggle === 'function') {
        if (s.collapse && !mx.collapse) { ctx.matrix.toggle('collapse'); run.collapseOn = true; }
        else if (!s.collapse && run.collapseOn && mx.collapse) { ctx.matrix.toggle('collapse'); run.collapseOn = false; }
        if (s.anim && mx.batch) { mx.batch = false; run.batchOff = true; }   // as a step does (matrix.js)
      }
    } catch (err) { console.warn('[nn/tour] matrix:', err); }
    setIf('anim', s.anim ?? null);
    if (s.reveal) {
      const r = s.reveal;
      try { ctx.matrix?.reveal?.(r.layer, r.part ?? null, Number.isInteger(r.token) ? { token: r.token } : undefined); }
      catch (err) { console.warn('[nn/tour] reveal:', err); }
    }
    publish();
  }

  function publish() {
    if (!run) return;
    const s = run.steps[run.i], env = { net: store.net, fwd: store.state.fwd, bwd: store.state.bwd };
    let t = '', x = '';
    try { t = s.title(env); x = s.text(env); } catch (err) { console.error('[nn/tour] caption:', err); t = t || 'Explain'; }
    setIf('tour', { i: run.i, n: run.steps.length, title: t, text: x });
  }

  // Put lens, viz and anim back as they were before the tour. anim: false leaves the step-through
  // off; fresh (another net was loaded): lens.js and attnviz.js have already reset lens and viz for
  // it, so the old net's are not put back.
  function end({ anim = true, fresh = false } = {}) {
    if (!run) return;
    const r = run;
    run = null;
    clearTimeout(refreshTimer);
    refreshTimer = 0;
    try { if (r.trainFolded && ctx.train?.folded) ctx.train.fold(false); }   // unless the user unfolded it meanwhile
    catch (err) { console.warn('[nn/tour] train:', err); }
    const mx = ctx.matrix?.opt;
    try {
      if (mx && r.collapseOn && mx.collapse) ctx.matrix.toggle('collapse');
      if (mx && r.batchOff && !mx.batch) { mx.batch = true; ctx.matrix.render?.(); }
    } catch (err) { console.warn('[nn/tour] matrix:', err); }
    if (!fresh) {
      setIf('lens', r.saved.lens);
      setIf('viz', r.saved.viz && layerExists(r.saved.viz.layer) ? r.saved.viz : null);
    } else if (store.state.viz && !layerExists(store.state.viz.layer)) setIf('viz', null);
    setIf('anim', anim ? r.saved.anim : null);
    setIf('tour', null);
  }

  const toggle = () => (run ? end() : start());

  // Captions follow the numbers (training, sliders), at most every REFRESH_MS.
  let refreshTimer = 0;
  store.on('values', () => {
    if (!run || refreshTimer) return;
    refreshTimer = setTimeout(() => { refreshTimer = 0; publish(); }, REFRESH_MS);
  });
  // A new structure ends the tour; a new story on the same structure (heads, causal, an activation)
  // rebuilds the steps and stays on the same step number.
  store.on('net', p => {
    if (!run) return;
    if (p?.structural) { end({ anim: false, fresh: (store.net.meta?.title ?? '') !== run.title }); return; }
    const key = storyKey(store.net);
    if (key === run.key) return;
    const list = steps();
    if (!list.length) { end({ anim: false }); return; }
    run.key = key;
    run.steps = list;
    run.i = Math.min(run.i, list.length - 1);
    apply();
  });

  toolBtn = ctx.addButton?.({
    label: 'Explain', title: 'Walk through this net step by step (E). ← → move, Esc ends', group: 'tour',
    onClick: () => toggle(),
  }) || null;

  // E starts / ends; while running, → / PageDown, ← / PageUp and Esc. Capture phase, so Esc ends the
  // tour before the shell's Esc (deselect, close the cheat sheet) sees it.
  window.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey || !(ctx.active ? ctx.active(e) : true)) return;
    const k = e.key;
    let fn = null;
    if (k === 'e' || k === 'E') fn = e.repeat ? () => {} : toggle;
    else if (!run) return;
    else if (k === 'Escape') fn = () => end();
    else if (k === 'ArrowRight' || k === 'PageDown') fn = e.repeat ? () => {} : () => go(1);
    else if (k === 'ArrowLeft' || k === 'PageUp') fn = e.repeat ? () => {} : () => go(-1);
    if (!fn) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    fn();
  }, true);

  render();
  return api;
}
