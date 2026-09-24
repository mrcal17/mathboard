// Net tab: the live network as matrix multiplication, in #nn-matrix. See docs/NN_CONTRACT.md.
//   Each layer is z = W a (+ skip terms) + b, a = σ(z), with the actual numbers. Numbers are HTML
//   grids updated in place (training ticks at 60 fps); KaTeX renders symbols, only on rebuild.
//   Hover and click link both ways through store.hover / store.sel.
//   Step-through (S / Shift+S, buttons) owns state.anim: the forward pass row by row, then the
//   backward pass (δ, ∂L/∂W, ∂L/∂b) when every output has a target.
//   Toggles: Backward, bias trick [W | b] (B), batch Z = W X + b 1ᵀ, collapse, labels.
//   Layers whose W is at most 3×3 can be sent to the 3D tab (map / transform).
//   Token layers (docs/NN_ATTENTION.md) read as tokens × d matrices: a tied tokenwise layer is
//   Q = X W_Q (an expander shows the flattened I ⊗ W_Qᵀ form of z = W a), a fixed identity term is
//   + X, and an attention layer is S = QKᵀ/√d_k, A = softmax(S), Z = A V. Attention steps go token
//   by token in three phases ('scores', 'softmax', 'sum'); their anim.i is the token's first node.
//   The lens (state.lens, docs/NN_LENS.md) filters the panel without a rebuild: focus folds every
//   other layer to a one-line summary and scrolls to the focused one (its part is highlighted);
//   token dims the other rows of every token matrix, outlines row t and opens a token trace card;
//   head hides the other heads; minW / minA dim small weights / attention weights. Every dimming
//   comes from focus.js's emphasis(), the same function the canvas uses.
//   ctx.matrix.reveal(layer, part) and ctx.matrix.parts(layer) are for tour.js.
import { colorFor } from './store.js';
import { copyLens, emphasis, tokenNames } from './focus.js';
import { lensOf } from './lens.js';

const BATCH = 4;         // samples shown as columns in the batch view
const STRONG = 0.55;     // |v| / max above which a tinted cell switches to contrasting text
const DIM = 0.16;        // opacity of a cell outside the lens's emphasis
const ACT_TEX = {
  identity: '', relu: '\\operatorname{ReLU}', leaky: '\\operatorname{LReLU}', sigmoid: '\\sigma',
  tanh: '\\tanh', softmax: '\\operatorname{softmax}',
};
const KX = { throwOnError: false, strict: 'ignore', trust: c => c.command === '\\htmlData' };
const NN_VISUAL = /^(?:transform\(W(?:\d+|eff), t\)|map\(W(?:\d+|eff)\))$/;   // rows we add to the 3D tab

function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

function tex(el, src, display = false) {
  if (el._tex === src) return el;
  el._tex = src;
  if (typeof katex === 'undefined') { el.textContent = src; return el; }
  try { katex.render(src, el, { ...KX, displayMode: display }); } catch { el.textContent = src; }
  return el;
}

function maxAbs(...xs) {
  let m = 0;
  const walk = x => {
    if (Array.isArray(x)) x.forEach(walk);
    else if (Number.isFinite(x)) m = Math.max(m, Math.abs(x));
  };
  xs.forEach(walk);
  return m;
}

const r3 = x => { const s = String(Math.round(x * 1000) / 1000); return s === '-0' ? '0' : s; };
const wName = (l, k) => (k === l - 1 ? `W^{(${l})}` : `W^{(${l},${k})}`);
const aName = k => (k === 0 ? '\\mathbf x' : `a^{(${k})}`);
const aT = k => (k === 0 ? '\\mathbf x^{\\top}' : `a^{(${k})\\top}`);
const AName = k => (k === 0 ? 'X' : `A^{(${k})}`);
const idx = i => `{${i + 1}}`;

// Token layers: node k of a layer is group floor(k / (tokens·d)), token floor(k / d) % tokens,
// feature k % d (docs/NN_ATTENTION.md).
const tpos = (S, i) => ({ g: Math.floor(i / (S.tokens * S.d)), t: Math.floor(i / S.d) % S.tokens, f: i % S.d });
const tidx = (S, g, t, f) => (g * S.tokens + t) * S.d + f;
const hasTie = e => !!e && e.tie != null && e.tie !== '';
const tieName = tie => { const s = String(tie), c = s.lastIndexOf(':'); return c > 0 ? s.slice(0, c) : s; };
const wT = s => `{${s}}^{\\top}`;
const PHASES = ['scores', 'softmax', 'sum'];
// Plain text inside KaTeX \text{...}
const texEsc = s => String(s).replace(/[\\{}$&#^_%~]/g, c => ({ '\\': '\\textbackslash{}', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}' }[c] || `\\${c}`));
const unit = v => (Number.isFinite(+v) ? Math.max(0, Math.min(1, +v)) : 1);

export function install(ctx) {
  const { store, model } = ctx;
  const host = ctx.el?.matrix;
  if (!host) return;

  // expand: comma-separated ids of token layers whose flattened z = W a form is open (a string, so
  // the audience mirror compares it by value)
  const opt = { mode: 'fwd', bias: false, batch: false, collapse: false, labels: true, expand: '' };
  const theme = () => (ctx.theme ? ctx.theme() : document.documentElement.dataset.theme) || 'dark';
  const f2 = x => (Number.isFinite(x) ? model.fmt(x, 2) : '\\text{--}');                  // TeX
  const num = x => (Number.isFinite(x) ? model.fmt(x, 2).replace(/^-/, '−') : '·'); // HTML
  const par = x => { const s = f2(x); return s.startsWith('-') ? `(${s})` : s; };
  // Gradients are often tiny: below 0.01 they keep two significant figures (0.0034, 3.4e-4).
  const fmtg = x => (model.fmtg ? model.fmtg(x, 2) : model.fmt(x, 2));
  const g2 = x => (Number.isFinite(x) ? fmtg(x).replace(/e(-?\d+)$/, '\\mathrm{e}{$1}') : '\\text{--}');   // TeX
  const numg = x => (Number.isFinite(x) ? fmtg(x).replace(/-/g, '−') : '·');                              // HTML
  const parg = x => { const s = g2(x); return s.startsWith('-') ? `(${s})` : s; };
  const GRAD = new Set(['g', 'ga', 'gt']);   // grid scales that hold gradients
  const actTex = act => ACT_TEX[act] ?? `\\operatorname{${act}}`;
  const actLabel = act => model.ACTS?.[act]?.label || act;
  const label = id => model.node(store.net, id)?.label || '?';
  const layerRef = l => l;   // hover {row|col}.layer is the layer index

  // ---------------------------------------------------------------- skeleton + toolbar
  const root = h('div', 'nm');
  if (ctx.audience) root.classList.add('nm-audience');
  const bar = h('div', 'nm-bar');
  const body = h('div', 'nm-body');
  root.append(bar, body);
  host.append(root);

  const btn = (html, title, onClick, cls = '') => {
    const b = h('button', cls);
    b.type = 'button';
    b.innerHTML = html;
    b.title = title;
    b.onclick = onClick;
    bar.append(b);
    return b;
  };
  btn('&#9664;', 'Step back (Shift+S)', () => step(-1), 'nm-sq');
  btn('Step &#9654;', 'Step through the forward pass row by row, then backprop (S)', () => step(1));
  const bStop = btn('&#9632;', 'Stop stepping', () => store.set('anim', null), 'nm-sq');
  const info = h('span', 'nm-info');
  bar.append(info, h('span', 'nm-gap'));
  const tg = {
    mode: btn('Backward', 'Backprop: δ, ∂L/∂W and ∂L/∂b (needs a target on every output node)', () => toggle('mode')),
    bias: btn('[W&thinsp;|&thinsp;b]', 'Bias trick: b becomes the last column of W and a 1 is appended to the input (B)', () => toggle('bias')),
    batch: btn('Batch', 'Z = W X + b 1ᵀ with a few samples as the columns of X', () => toggle('batch')),
    collapse: btn('Collapse', 'Multiply the layers into one matrix (works only when every layer is identity)', () => toggle('collapse')),
    labels: btn('Labels', 'Row and column headers with the node labels', () => toggle('labels')),
  };
  btn('&rarr; 3D', 'Send the selected layer\'s weight matrix (at most 3×3) to the 3D tab', () => send3d(pickLayer()));

  // Backward mode chosen with the button (not switched on by stepping into backprop): only then
  // does a fresh step-through start at the backward pass.
  let userBwd = false;
  function toggle(k) {
    if (k === 'mode') { opt.mode = opt.mode === 'bwd' ? 'fwd' : 'bwd'; userBwd = opt.mode === 'bwd'; }
    else opt[k] = !opt[k];
    if (k === 'batch' && opt.batch && store.state.anim) store.set('anim', null);
    if (k === 'mode' && opt.mode === 'fwd' && store.state.anim?.dir === 'bwd') store.set('anim', null);
    schedule();
  }
  const expanded = () => new Set(String(opt.expand || '').split(',').filter(Boolean));
  function toggleExpand(id) {
    const s = expanded();
    if (s.has(id)) s.delete(id); else s.add(id);
    opt.expand = [...s].join(',');
    schedule();
  }

  // ---------------------------------------------------------------- per-frame data
  // The batch view samples the Train panel's dataset: train.js's readSettings fills the same
  // defaults it uses (meta.train can be empty for a fresh preset). Without train.js, raw meta.train.
  let batchCache = { key: null, X: [], live: false, src: '' };
  let trainSettings = net => net.meta?.train || {};
  import('./train.js').then(m => {
    if (typeof m.readSettings === 'function') trainSettings = net => m.readSettings(net, model);
    batchCache.key = null;
    if (opt.batch) schedule();
  }).catch(() => {});

  // The lens (docs/NN_LENS.md): focus.js's emphasis() is what the canvas dims by, and the panel too.
  const lensNow = () => lensOf(store);   // complete and valid, a fresh copy
  const setLens = patch => store.set('lens', copyLens({ ...lensNow(), ...patch }));

  function batchInputs(net) {
    const ins = model.nodesIn(net, 0), n0 = ins.length;
    let t = {};
    try { t = trainSettings(net) || {}; } catch { /* keep {} */ }
    const dkey = t.dataset;
    const ds = dkey != null ? model.DATASETS?.[dkey] : null;
    const key = JSON.stringify([n0, dkey, t.n, t.seed, t.noise]);
    if (batchCache.key !== key) {
      batchCache = { key, X: [], live: true, src: 'x and seeded samples' };
      try {
        if (ds && ds.inputs === n0) {
          const { X } = ds.make(t.n ?? 200, t.seed ?? 1, t.noise ?? 0);
          // the first few: the datasets cycle through the classes by index
          batchCache = { key, X: X.slice(0, BATCH).map(x => x.slice()), live: false, src: ds.label || dkey };
        }
      } catch (err) { console.warn('[nn/matrix] batch from dataset:', err); }
      if (batchCache.live) {
        const r = model.rng(7);
        batchCache.X = Array.from({ length: BATCH - 1 }, () => ins.map(() => Math.round((r() * 2 - 1) * 10) / 10));
      }
    }
    return batchCache.live ? [ins.map(n => +n.value || 0), ...batchCache.X] : batchCache.X;
  }

  function compute() {
    const net = store.net, L = net.layers.length - 1;
    const d = { net, L, M: [], fwd: store.state.fwd, bwd: null, max: { w: 1, b: 1, a: 1, g: 1, one: 1, eff: 1, s: 1, ga: 1, gt: 1 } };
    if (L < 1 || !d.fwd) return d;
    for (const m of model.matrices(net)) d.M[m.l] = m;
    d.T = structure(net, d.M);
    // token layers shown compactly write their bias when it is a shared (tied) row vector, or
    // when some entry of it is nonzero
    d.bflag = d.T.map((S, l) => (l && S.mode === 'tok'
      ? Array.from({ length: S.G }, (_, g) => !!S.bias[g]?.tied
        || d.M[l].b.slice(g * S.tokens * S.d, (g + 1) * S.tokens * S.d).some(v => v !== 0)) : null));
    d.max.w = maxAbs(net.edges.map(e => e.w)) || 1;     // same scales as the inspector's sliders
    d.max.b = maxAbs(net.nodes.map(n => n.bias)) || 1;
    d.max.a = maxAbs(d.fwd.a, d.fwd.z) || 1;
    d.max.s = maxAbs(d.T.map((S, l) => (S.mode === 'attn' ? d.fwd.attn?.[l]?.heads?.map(x => x.S) : null))) || 1;
    if (opt.batch) {
      const X = batchInputs(net);
      const runs = X.map(x => model.forward(net, x));
      d.batch = { X, src: batchCache.src, a: [], z: [] };
      for (let l = 0; l <= L; l++) {
        d.batch.a[l] = (runs[0]?.a[l] || []).map((_, i) => runs.map(r => r.a[l][i]));
        d.batch.z[l] = l ? (runs[0]?.z[l] || []).map((_, i) => runs.map(r => r.z[l][i])) : null;
      }
      // colours keep the live sample's scale (as in the view); larger batch values just saturate
    }
    const bwd = store.state.bwd;
    if (opt.mode === 'bwd' && bwd) {
      d.bwd = bwd;
      d.y = model.nodesIn(net, L).map(n => n.target);
      d.sp = [];
      let g = maxAbs(bwd.dZ, bwd.dA);
      for (let l = 1; l <= L; l++) {
        const act = net.layers[l].act, z = d.fwd.z[l], a = d.fwd.a[l];
        d.sp[l] = act === 'softmax' ? z.map(() => NaN) : z.map((zi, i) => model.ACTS[act].df(zi, a[i]));
        for (const t of d.M[l].terms) g = Math.max(g, maxAbs(bwd.dZ[l]) * maxAbs(d.fwd.a[t.k]));
      }
      d.max.g = g || 1;
      d.sumY = d.y.reduce((s, v) => s + v, 0);
      // attention's ∂A, ∂S and a tied matrix's summed gradient have their own scales
      d.max.ga = maxAbs(d.T.map((S, l) => (S.mode === 'attn' ? bwd.attn?.[l]?.heads?.map(x => [x.dA, x.dS]) : null))) || 1;
      d.max.gt = Math.max(d.max.g, maxAbs(Object.values(bwd.tie || {})));
    }
    if (opt.collapse) {
      d.col = collapseData(d);
      d.max.eff = maxAbs(d.col.W, d.col.b) || 1;
    }
    return d;
  }

  function collapseData(d) {
    const net = d.net, x = d.fwd.a[0];
    const apply = c => c.W.map((r, i) => r.reduce((s, w, j) => s + w * x[j], 0) + c.b[i]);
    const skip = d.M.some(m => m && m.terms.length > 1);
    if (net.layers.some(L => L.kind === 'attention')) return { ok: false, attn: true, skip };
    const c = model.collapse(net);
    if (c) return { ok: true, W: c.W, b: c.b, y: apply(c), skip };
    const lin = model.clone(net);
    for (const L of lin.layers.slice(1)) L.act = 'identity';
    const cl = model.collapse(lin);
    const bad = net.layers.map((L, l) => (l && L.act !== 'identity' ? l : 0)).filter(Boolean);
    return { ok: false, W: cl?.W, b: cl?.b, y: cl ? apply(cl) : null, skip, bad };
  }

  // Structure that needs a DOM rebuild (values alone never do).
  function signature(d) {
    const net = d.net;
    return [
      opt.mode, opt.bias, opt.batch, opt.collapse, opt.labels, !!d.bwd, d.bwd?.note || '', net.meta?.loss,
      d.bwd && outCase(d) === 'softmax-xent' ? d.sumY.toFixed(4) : '',
      d.batch ? d.batch.X.length + ':' + d.batch.X[0]?.length : '', d.col ? `${d.col.ok}:${d.col.skip}:${!!d.col.W}` : '',
      !!d.fwd, net.layers.map(l => `${l.id}:${l.act}:${l.name}`).join(),
      net.nodes.map(n => `${n.id}:${n.layer}:${n.label}`).join(),
      net.edges.map(e => `${e.id}:${e.from}>${e.to}`).join(),
      skey || '', opt.expand, (d.bflag || []).map(f => (f ? f.map(Number).join('') : '')).join(),
      !!d.fwd?.attn, !!d.bwd?.attn, !!d.bwd?.tie, d.bwd ? String(outRows(d)?.proper ?? '') : '',
      JSON.stringify(net.meta?.tokenNames ?? null),
    ].join('|');
  }

  // ---------------------------------------------------------------- token structure
  // Per layer: its token shape and how it reads as token matrices. mode is 'dense' (the plain
  // z = W a + b view), 'tok' (every term is a tied tokenwise block S W or a fixed identity + S),
  // 'attn' or 'attnbad' (an attention layer whose input isn't a Q/K/V layer). Structural only,
  // cached by skeyOf; numbers are read per frame through a representative (row, col) of M.
  let skey = null, struct = null;
  const skeyOf = net => [
    net.layers.map(l => [l.id, l.kind || '', l.tokens ?? '', (l.groups || []).join('/'), l.heads ?? '', l.causal ? 1 : 0, l.scale ?? ''].join(':')).join(),
    net.nodes.map(n => `${n.id}:${n.layer}:${n.tie ?? ''}`).join(),
    net.edges.map(e => `${e.id}:${e.from}>${e.to}:${e.tie ?? ''}:${e.fixed ? `F${e.w}` : ''}`).join(),
  ].join('|');

  function structure(net, M) {
    const k = skeyOf(net);
    if (k !== skey || !struct) { struct = analyze(net, M); skey = k; }
    return struct;
  }

  function analyze(net, M) {
    const L = net.layers.length - 1, eById = new Map(net.edges.map(e => [e.id, e]));
    const tieCount = new Map();
    for (const e of net.edges) if (hasTie(e)) tieCount.set(e.tie, (tieCount.get(e.tie) || 0) + 1);
    const nodeTies = new Map();
    for (const n of net.nodes) if (hasTie(n)) nodeTies.set(n.tie, (nodeTies.get(n.tie) || 0) + 1);
    const T = net.layers.map((lay, l) => {
      const ns = model.nodesIn(net, l), ids = ns.map(n => n.id);
      const tokens = Math.max(1, Math.floor(+lay.tokens) || 1);
      const groups = Array.isArray(lay.groups) && lay.groups.length ? lay.groups.map(String) : null;
      const G = groups ? groups.length : 1, w = ids.length / (tokens * G);
      const ok = ids.length > 0 && Number.isInteger(w);
      const S = { l, id: lay.id, ids, tokens, groups, G, d: ok ? w : 0, ok, tok: ok && (tokens > 1 || !!groups),
        act: lay.act, mode: lay.kind === 'attention' ? 'attnbad' : 'dense', blocks: [], resid: [], bias: [] };
      // A bias tied per feature (node.tie '<name>:<f>' on every token) is one row vector b: + 1 bᵀ.
      if (S.tok && l) {
        for (let g = 0; g < G; g++) {
          const ties = Array.from({ length: S.d }, (_, f) => ns[tidx(S, g, 0, f)].tie);
          let tied = ties.every(x => typeof x === 'string' && x !== '') && new Set(ties).size === S.d && new Set(ties.map(tieName)).size === 1;
          for (let t = 1; tied && t < tokens; t++) for (let f = 0; f < S.d; f++) if (ns[tidx(S, g, t, f)].tie !== ties[f]) tied = false;
          S.bias[g] = tied ? { tied, name: tieName(ties[0]), ties, extra: ties.some(x => nodeTies.get(x) > tokens) } : { tied: false };
        }
      }
      return S;
    });
    for (let l = 1; l <= L; l++) {
      const S = T[l], m = M[l];
      if (S.mode === 'attnbad') { attnInfo(net.layers[l], S, T[l - 1]); continue; }
      if (!S.tok || !m) continue;
      let ok = true;
      const blocks = [], resid = [];
      m.terms.forEach((term, ti) => {
        if (!ok) return;
        const src = T[term.k], list = [];
        term.edge.forEach((r, i) => r.forEach((eid, j) => { if (eid != null) list.push({ e: eById.get(eid), i, j }); }));
        if (!list.length) return;
        if (!src.ok || src.tokens !== S.tokens || list.some(x => !x.e)) { ok = false; return; }
        if (list.every(x => x.e.fixed)) {   // + S: fixed weight-1 edges, node i to node i
          const ident = src.G === S.G && src.d === S.d && list.length === S.ids.length && list.every(x => x.i === x.j && +x.e.w === 1);
          if (ident) resid.push({ k: term.k, ti }); else ok = false;
          return;
        }
        if (!list.every(x => hasTie(x.e) && !x.e.fixed)) { ok = false; return; }
        // tied: one small matrix per (source group, target group), the same on every token
        const byBlk = new Map();
        for (const { e, i, j } of list) {
          const pd = tpos(S, i), ps = tpos(src, j);
          if (pd.t !== ps.t) { ok = false; return; }
          const key = `${ps.g}>${pd.g}`;
          let b = byBlk.get(key);
          if (!b) {
            byBlk.set(key, b = { k: term.k, ti, gs: ps.g, gd: pd.g, name: tieName(e.tie),
              cells: Array.from({ length: src.d }, () => new Array(S.d).fill(null)) });
          }
          if (b.name !== tieName(e.tie)) { ok = false; return; }
          let c = b.cells[ps.f][pd.f];
          if (!c) c = b.cells[ps.f][pd.f] = { tie: e.tie, rep: e.id, i, j, t: pd.t, byTok: [], rowByTok: [], dst: [], src: [], rows: [], cols: [] };
          if (c.tie !== e.tie || c.byTok[pd.t] != null) { ok = false; return; }
          c.byTok[pd.t] = e.id;
          c.rowByTok[pd.t] = i;
          c.dst.push(S.ids[i]); c.src.push(src.ids[j]); c.rows.push(i); c.cols.push(j);
          if (pd.t < c.t) Object.assign(c, { rep: e.id, i, j, t: pd.t });
        }
        for (const b of byBlk.values()) {
          for (const row of b.cells) {
            for (const c of row) {
              if (!c) continue;
              if (c.rows.length !== S.tokens) ok = false;
              c.extra = (tieCount.get(c.tie) || 0) > c.rows.length;   // the tie also reaches edges outside this block
            }
          }
          blocks.push(b);
        }
      });
      if (ok && (blocks.length || resid.length)) Object.assign(S, { mode: 'tok', blocks, resid });
    }
    return T;
  }

  function attnInfo(lay, S, P) {
    const names = P?.groups || [], gq = names.indexOf('Q'), gk = names.indexOf('K'), gv = names.indexOf('V');
    const heads = Math.max(1, Math.floor(+lay.heads) || 1);
    if (!(S.ok && P?.ok && gq >= 0 && gk >= 0 && gv >= 0 && P.tokens === S.tokens && P.d % heads === 0 && S.d % heads === 0)) return;
    const dh = P.d / heads, def = 1 / Math.sqrt(dh);
    const scale = lay.scale != null && lay.scale !== '' && Number.isFinite(+lay.scale) ? +lay.scale : def;
    Object.assign(S, { mode: 'attn', heads, dh, dvh: S.d / heads, causal: !!lay.causal, scale,
      scaleDef: Math.abs(scale - def) < 1e-12, gq, gk, gv, P });
  }

  // Node id behind entry (t, c) of head hh's Q / K / V / Z matrix of attention layer S.
  const headNode = (S, hh, which, t, c) => (which === 'z' ? S.ids[tidx(S, 0, t, hh * S.dvh + c)]
    : S.P.ids[tidx(S.P, which === 'q' ? S.gq : which === 'k' ? S.gk : S.gv, t, hh * (which === 'v' ? S.dvh : S.dh) + c)]);

  // ∂L/∂(node i of layer m - 1) that attention layer m sends back: its ∂Q, ∂K or ∂V entry.
  function attnGradAt(d, m, i) {
    const S = d.T[m], P = S.P, p = tpos(P, i), B = d.bwd?.attn?.[m];
    if (!B) return NaN;
    const v = p.g === S.gv, w = v ? S.dvh : S.dh, hh = Math.floor(p.f / w), c = p.f % w;
    const M = p.g === S.gq ? 'dQ' : p.g === S.gk ? 'dK' : v ? 'dV' : null;
    return M ? B.heads?.[hh]?.[M]?.[p.t]?.[c] : 0;
  }

  // Names in the token view: X, Q / K / V, Z^{(l)} (attention), H^{(l)}, and \hat Y for the output.
  function symOf(d, k, g = 0) {
    const S = d.T[k];
    if (S.groups) {
      const nm = S.groups[g] ?? '?';
      return d.T.some(o => o !== S && o.groups?.includes(nm)) ? `${nm}^{(${k})}` : nm;
    }
    if (k === 0) return 'X';
    if (S.mode === 'attn' || S.mode === 'attnbad') return `Z^{(${k})}`;
    return k === d.L ? '\\hat Y' : `H^{(${k})}`;
  }
  const preSym = (d, l, g) => (d.T[l].groups ? `\\tilde{${symOf(d, l, g)}}` : `Z^{(${l})}`);   // before the activation
  const bTied = (d, l, g) => d.T[l].bias?.[g]?.tied ? d.T[l].bias[g] : null;
  const bSym = (d, l, g) => (bTied(d, l, g) ? `\\mathbf 1\\,${wT(bTied(d, l, g).name)}`
    : d.T[l].groups ? `B_{${d.T[l].groups[g]}}` : `B^{(${l})}`);
  const dSym = (d, l, g) => (d.T[l].groups ? `\\delta_{${d.T[l].groups[g]}}` : `\\delta^{(${l})}`);
  const scoreTex = (S, q, k) => (S.scaleDef ? `\\frac{${q}\\,${k}}{\\sqrt{d_k}}` : `${f2(S.scale)}\\,${q}\\,${k}`);
  const scaleNum = S => (S.scaleDef ? `\\tfrac{1}{\\sqrt{${S.dh}}}` : f2(S.scale));

  // ---------------------------------------------------------------- DOM registry
  let keyMap = new Map(), binds = [], stepBoxes = [], sig = '', eMap = new Map();
  let hoverOf = new WeakMap(), clickOf = new WeakMap();
  // Lens registry (rebuilt with the DOM). emEls: { el, em } where em says what the element stands
  // for, so emphasis() can weigh it: n (node ids, max), e (edge ids, max; w: minW / show apply),
  // p ([from, to] of a missing edge), r ([layer, token row]), a ([l, i, j, h]: attention pair; thr:
  // minA applies), lg (keep legible when emphasized). toks: row / column outlines and token
  // headers { el, l, t }. secs[l]: layer sections. heads: per-head boxes { el, l, hh }.
  let emEls = [], toks = [], secs = [], headBoxes = [], formulaEls = [], trace = null, names = null;

  const reg = (el, keys) => {
    for (const k of keys) {
      if (!k) continue;
      let a = keyMap.get(k);
      if (!a) keyMap.set(k, a = []);
      a.push(el);
    }
  };
  const op = s => tex(h('span', 'nm-op'), s);
  const line = (s, cls = 'nm-sym') => tex(h('div', cls), s);
  // An equation row: each group ([op, grid, ...]) stays on one line; rows wrap between groups.
  const eqOf = (role, ...groups) => {
    const e = h('div', 'nm-eq');
    e.dataset.eq = role;
    for (const g of groups) {
      if (!g?.length) continue;
      const s = h('span', 'nm-grp');
      s.append(...g.filter(Boolean));
      e.append(s);
    }
    return e;
  };

  // A bracketed grid of numbers. cell(i, j) -> { f(d) -> number, s: scale key, mask, faint, fixed,
  // keys, hover, click, em }. o: { cap, rowHdr, colHdr: [{ tex, hover, click, em, tok }], aug: column
  // index with a bar before it, augRow: row index with a bar above it, rowTok / colTok: the layer
  // whose tokens the rows / columns are (they get the lens's row outline) }.
  function grid(n, m, cell, o = {}) {
    const blk = h('div', o.cls ? `nm-blk ${o.cls}` : 'nm-blk');
    const rh = opt.labels && o.rowHdr, ch = opt.labels && o.colHdr;
    const c0 = rh ? 2 : 1;
    blk.style.gridTemplateColumns = `${rh ? 'auto ' : ''}5px repeat(${m}, auto) 5px`;
    blk.style.gridTemplateRows = `var(--hdr) repeat(${n}, auto) var(--cap)`;
    const lb = h('i', 'nm-lb'), rb = h('i', 'nm-rb');
    lb.style.gridArea = `2 / ${c0} / ${n + 2} / ${c0 + 1}`;
    rb.style.gridArea = `2 / ${c0 + m + 1} / ${n + 2} / ${c0 + m + 2}`;
    blk.append(lb, rb);
    const hdr = (s, cls, area) => {
      if (!s) return;
      const e = tex(h('span', cls), s.tex);
      e.style.gridArea = area;
      if (s.hover) hoverOf.set(e, s.hover);
      if (s.click) clickOf.set(e, s.click);
      reg(e, s.keys || []);
      if (s.em) emEls.push({ el: e, em: s.em });
      if (s.tok) toks.push({ el: e, l: s.tok.l, t: s.tok.t, cls: 'nm-tok-on' });
      blk.append(e);
    };
    if (ch) ch.forEach((s, j) => hdr(s, 'nm-ch', `1 / ${c0 + 1 + j}`));
    if (rh) rh.forEach((s, i) => hdr(s, 'nm-rh', `${i + 2} / 1`));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const c = h('span', 'nm-c'), sp = cell(i, j);
        c.style.gridArea = `${i + 2} / ${c0 + 1 + j}`;
        if (o.aug != null && j === o.aug) c.classList.add('nm-augc');
        if (o.augRow != null && i === o.augRow) c.classList.add('nm-augr');
        if (sp.mask) c.classList.add('nm-mask');
        if (sp.faint) c.classList.add('nm-faint');
        if (sp.cls) c.classList.add(...sp.cls.split(' ').filter(Boolean));
        if (sp.fixed != null) c.textContent = sp.fixed;
        else binds.push({ el: c, f: sp.f, s: sp.mask ? null : sp.s });
        reg(c, sp.keys || []);
        if (sp.hover) hoverOf.set(c, sp.hover);
        if (sp.click) clickOf.set(c, sp.click);
        if (sp.em) emEls.push({ el: c, em: sp.em });
        blk.append(c);
      }
    }
    // the lens's outline of token t's row (or column, in a transposed grid), drawn over the cells
    const band = (area, l, t) => {
      const b = h('i', 'nm-band');
      b.style.gridArea = area;
      toks.push({ el: b, l, t, cls: 'on' });
      blk.append(b);
    };
    if (o.rowTok != null) for (let i = 0; i < n; i++) band(`${i + 2} / ${c0 + 1} / ${i + 3} / ${c0 + 1 + m}`, o.rowTok, i);
    if (o.colTok != null) for (let j = 0; j < m; j++) band(`2 / ${c0 + 1 + j} / ${n + 2} / ${c0 + 2 + j}`, o.colTok, j);
    if (o.cap) {
      const cap = tex(h('span', 'nm-cap'), o.cap);
      cap.style.gridArea = `${n + 2} / 1 / ${n + 3} / -1`;
      blk.append(cap);
    }
    return blk;
  }

  // Column vector of node values. f(d, i) -> number.
  const nodeVec = (ids, f, s, keys, cap, extra = {}) => grid(ids.length, 1, i => ({
    f: d => f(d, i), s, keys: [`n:${ids[i]}`, ...keys(i)], hover: { kind: 'node', id: ids[i] }, click: { kind: 'node', id: ids[i] },
    em: { n: [ids[i]] },
  }), { cap, ...extra });

  const nodeHdr = (ids, extra) => ids.map((id, j) => ({ tex: label(id), em: { n: [id] }, ...extra(id, j) }));

  // W (or its transpose) of term t in layer l; aug appends b as a last column.
  function wGrid(d, l, ti, { aug = false, T = false } = {}) {
    const m = d.M[l], t = m.terms[ti], n = m.rows.length, c = t.cols.length;
    // transposed (inside δ^{(k)}): its row j is node j of layer k, the row being stepped backward
    const extraKeys = T ? (i, j) => [`brow:${t.k}:${j}`] : (i, j) => [`row:${l}:${i}`, `col:${l}:${t.k}:${j}`];
    const cell = (i, j) => {
      const to = m.rows[i];
      if (j === c) {
        return { f: dd => dd.M[l].b[i], s: 'b', keys: [`b:${to}`, `row:${l}:${i}`, `dst:${to}`],
          hover: { kind: 'bias', id: to }, click: { kind: 'node', id: to }, em: { n: [to] } };
      }
      const from = t.cols[j], eid = t.edge[i][j];
      const keys = [...extraKeys(i, j), `dst:${to}`, `src:${from}`];
      if (eid == null) return { mask: true, fixed: '0', keys: [...keys, `p:${from}>${to}`], hover: { kind: 'pair', from, to }, em: { p: [from, to] } };
      const e = eMap.get(eid);
      return { f: dd => dd.M[l].terms[ti].W[i][j], s: 'w', keys: [...keys, `e:${eid}`, hasTie(e) && `tie:${e.tie}`],
        cls: e?.fixed ? 'nm-fixedc' : hasTie(e) ? 'nm-tied' : '',
        hover: { kind: 'edge', id: eid }, click: { kind: 'edge', id: eid }, em: { e: [eid], w: true } };
    };
    const rowHdr = nodeHdr(m.rows, (id, i) => ({ hover: { kind: 'row', layer: layerRef(l), i }, click: { kind: 'node', id } }));
    const colHdr = nodeHdr(t.cols, (id, j) => ({ hover: { kind: 'col', layer: layerRef(l), k: t.k, j }, click: { kind: 'node', id } }));
    if (T) {   // rows = source nodes (layer t.k), columns = this layer's nodes
      return grid(c, n, (i, j) => cell(j, i), {
        cap: `\\big(${wName(l, t.k)}\\big)^{\\!\\top}`, rowHdr: colHdr, colHdr: rowHdr,
      });
    }
    return grid(n, c + (aug ? 1 : 0), cell, {
      aug: aug ? c : null, rowHdr,
      colHdr: aug ? [...colHdr, { tex: '1' }] : colHdr,
      cap: aug ? `\\big[\\,${wName(l, t.k)}\\mid b^{(${l})}\\big]` : wName(l, t.k),
    });
  }

  // a^{(k)} feeding term t of layer l (single sample), with a trailing 1 for the bias trick.
  function inVec(d, l, ti, aug) {
    const t = d.M[l].terms[ti], c = t.cols.length;
    return grid(c + (aug ? 1 : 0), 1, i => (i === c
      ? { fixed: '1', keys: [`in:${l}`] }
      : { f: dd => dd.fwd.a[t.k][i], s: 'a', keys: [`n:${t.cols[i]}`, `col:${l}:${t.k}:${i}`, `in:${l}`],
        hover: { kind: 'node', id: t.cols[i] }, click: { kind: 'node', id: t.cols[i] }, em: { n: [t.cols[i]] } }), {
      augRow: aug ? c : null,
      cap: aug ? `\\left[\\begin{smallmatrix}${aName(t.k)}\\\\ 1\\end{smallmatrix}\\right]` : aName(t.k),
    });
  }

  // A^{(k)} (batch): source nodes as rows, samples as columns.
  function inMat(d, l, ti, aug) {
    const t = d.M[l].terms[ti], c = t.cols.length, B = d.batch.X.length;
    return grid(c + (aug ? 1 : 0), B, (i, s) => (i === c
      ? { fixed: '1', keys: [`in:${l}`] }
      : { f: dd => dd.batch.a[t.k][i][s], s: 'a', keys: [`n:${t.cols[i]}`, `col:${l}:${t.k}:${i}`, `in:${l}`],
        hover: { kind: 'node', id: t.cols[i] }, click: { kind: 'node', id: t.cols[i] }, em: { n: [t.cols[i]] } }), {
      augRow: aug ? c : null,
      colHdr: t.k === 0 ? Array.from({ length: B }, (_, s) => ({ tex: `\\#${s + 1}` })) : null,
      cap: aug ? `\\left[\\begin{smallmatrix}${AName(t.k)}\\\\ \\mathbf 1^{\\top}\\end{smallmatrix}\\right]` : AName(t.k),
    });
  }

  function outVec(d, l, which) {
    const m = d.M[l], isOut = l === d.L;
    const cap = which === 'z' ? `z^{(${l})}` : isOut ? '\\hat{\\mathbf y}' : `a^{(${l})}`;
    if (d.batch) {
      const B = d.batch.X.length;
      return grid(m.rows.length, B, (i, s) => ({
        f: dd => dd.batch[which][l][i][s], s: 'a', keys: [`n:${m.rows[i]}`, `out:${l}:${i}`],
        hover: { kind: 'node', id: m.rows[i] }, click: { kind: 'node', id: m.rows[i] }, em: { n: [m.rows[i]] },
      }), { cap: which === 'z' ? `Z^{(${l})}` : isOut ? '\\hat Y' : `A^{(${l})}` });
    }
    return nodeVec(m.rows, (dd, i) => dd.fwd[which][l][i], 'a', i => [`out:${l}:${i}`], cap);
  }

  const biasVec = (d, l) => {
    const m = d.M[l];
    return grid(m.rows.length, 1, i => ({
      f: dd => dd.M[l].b[i], s: 'b', keys: [`b:${m.rows[i]}`, `row:${l}:${i}`, `dst:${m.rows[i]}`],
      hover: { kind: 'bias', id: m.rows[i] }, click: { kind: 'node', id: m.rows[i] }, em: { n: [m.rows[i]] },
    }), { cap: `b^{(${l})}` });
  };

  const arrow = (d, l) => `\\xrightarrow{\\;${actTex(d.net.layers[l].act) || '\\mathrm{id}'}\\;}`;

  // z^{(l)} = W a + b = [z] -σ-> [a]   (or the batch version Z = W X + b 1ᵀ)
  function fwdEq(d, l) {
    const m = d.M[l], B = d.batch?.X.length, groups = [];
    const lead = () => op(groups.length ? '+' : d.batch ? `Z^{(${l})} =` : `z^{(${l})} =`);
    m.terms.forEach((t, ti) => {
      if (!t.cols.length) return;
      const aug = opt.bias && ti === 0;
      groups.push([lead(), wGrid(d, l, ti, { aug }), d.batch ? inMat(d, l, ti, aug) : inVec(d, l, ti, aug)]);
    });
    if (!opt.bias || !m.terms[0]?.cols.length) {
      groups.push([lead(), biasVec(d, l), d.batch && grid(1, B, () => ({ fixed: '1' }), { cap: '\\mathbf 1^{\\top}' })]);
    }
    return eqOf('fwd', ...groups, [op('='), outVec(d, l, 'z')], [op(arrow(d, l)), outVec(d, l, 'a')]);
  }

  function shapeNote(d, l) {
    const m = d.M[l], n = m.rows.length, B = d.batch ? d.batch.X.length : 1, x = (a, b) => `${a}×${b}`;
    const augB = opt.bias && m.terms[0]?.cols.length > 0;
    const parts = m.terms.map((t, ti) => {
      const c = t.cols.length + (augB && ti === 0 ? 1 : 0);
      return t.cols.length ? `${x(n, c)} · ${x(c, B)}` : '';
    }).filter(Boolean);
    if (!augB) parts.push(d.batch ? `${x(n, 1)} · ${x(1, B)}` : x(n, 1));
    return `${parts.join(' + ')} → ${x(n, B)}`;
  }

  // ---------------------------------------------------------------- backward equations
  function outCase(d) {
    const loss = d.net.meta?.loss || 'mse', act = d.net.layers[d.L].act;
    if (loss === 'xent' && act === 'softmax') return 'softmax-xent';
    if (loss === 'xent' && act === 'sigmoid') return 'bce';
    return act === 'softmax' ? 'softmax-mse' : 'mse';
  }

  const primeTex = (act, arg) => (act === 'identity' ? '1' : `${actTex(act)}'(${arg})`);
  const primeVec = (act, arg) => (act === 'identity' ? '\\mathbf 1' : primeTex(act, arg));   // σ'(z) as a vector
  // Slope values: ReLU / leaky ReLU slopes are exact (1, 0, 0.1), the rest rounded like the grids.
  const spTex = (act, v) => ((act === 'relu' || act === 'leaky' || act === 'identity') && Number.isFinite(v) ? String(v) : f2(v));

  // A token output layer's softmax runs on each row (token, and group): with cross-entropy the
  // loss is the mean over those m rows, so δ = (Σ_row y · ŷ − y) / m. null for a plain output layer.
  function outRows(d) {
    const S = d.T?.[d.L];
    if (!S?.tok || S.mode === 'attn' || S.mode === 'attnbad') return null;
    const m = S.tokens * S.G, w = S.d;
    const sums = Array.from({ length: m }, (_, r) => d.y ? d.y.slice(r * w, (r + 1) * w).reduce((s, v) => s + (+v || 0), 0) : 1);
    return { m, w, sums, proper: sums.every(s => Math.abs(s - 1) < 1e-9), row: i => Math.floor(i / w) };
  }

  // Right-hand side of the loss over n outputs, as model.js computes it (ln = natural log).
  function lossTex(kind, n, rows = null) {
    const one = n === 1, s = one ? '1' : 'i', yh = `\\hat y_{${s}}`, y = `y_{${s}}`;
    const sum = one ? '' : `\\sum_{i=1}^{${n}}`;
    if (kind === 'softmax-xent' && rows && rows.m > 1) {
      return `-\\tfrac{1}{${rows.m}}\\sum_{t=1}^{${rows.m}}\\sum_{j=1}^{${rows.w}} Y_{t,j} \\ln \\hat Y_{t,j}`;
    }
    if (kind === 'softmax-xent') return `-${sum} ${y} \\ln ${yh}`;
    if (kind === 'bce') return `-${one ? '' : `\\tfrac{1}{${n}}`}${sum} \\big[${y} \\ln ${yh} + (1 - ${y}) \\ln (1 - ${yh})\\big]`;
    return `\\tfrac{1}{2}${one ? '' : `\\cdot\\tfrac{1}{${n}}`}${sum} (${yh} - ${y})^2`;
  }

  // Layers m > l with a term fed by layer l: where δ^{(l)} comes from.
  const upstream = (d, l) => {
    const out = [];
    for (let m = l + 1; m <= d.L; m++) d.M[m].terms.forEach((t, ti) => { if (t.k === l) out.push({ m, ti, t }); });
    return out;
  };

  function bwdEqs(d, l) {
    const els = [], m = d.M[l], ids = m.rows, act = d.net.layers[l].act, n = ids.length;
    const bk = i => [`brow:${l}:${i}`];
    const dVec = cap => nodeVec(ids, (dd, i) => dd.bwd.dZ[l][i], 'g', bk, cap);
    const spVec = () => nodeVec(ids, (dd, i) => dd.sp[l][i], 'one', bk, primeVec(act, `z^{(${l})}`));
    if (l === d.L) {
      const kind = outCase(d), frac = n > 1 ? `\\tfrac{1}{${n}}` : '';
      const yhat = nodeVec(ids, (dd, i) => dd.fwd.a[l][i], 'a', i => [`brow:${l}:${i}`], '\\hat{\\mathbf y}');
      const y = nodeVec(ids, (dd, i) => dd.y[i], 'a', i => [`brow:${l}:${i}`], '\\mathbf y');
      const dAcap = `\\partial L / \\partial a^{(${l})}`;
      const dAvec = () => nodeVec(ids, (dd, i) => dd.bwd.dA[l][i], 'g', bk, dAcap);
      const rows = kind === 'softmax-xent' ? outRows(d) : null;
      if (rows && rows.m > 1) {
        // a token output: softmax + xent on each row, averaged over the m rows
        const fm = `\\tfrac{1}{${rows.m}}`;
        els.push(line(`\\delta^{(${l})} = \\frac{\\partial L}{\\partial z^{(${l})}} = ${fm}\\big(${rows.proper ? '' : '(\\textstyle\\sum_{\\text{row}} y)\\,'}\\hat{\\mathbf y} - \\mathbf y\\big)
          \\quad\\text{(softmax + cross-entropy on each of the ${rows.m} token rows, averaged)}`));
        els.push(rows.proper ? eqOf('delta', [op(`\\delta^{(${l})} = ${fm}\\Big(`), yhat, op('-'), y, op('\\Big)')], [op('='), dVec(`\\delta^{(${l})}`)])
          : eqOf('delta', [op(`\\delta^{(${l})} =`), dVec(`\\delta^{(${l})}`)]));
      } else if (kind === 'softmax-xent' || kind === 'bce') {
        // softmax + xent: δ = (Σ y) ŷ − y, which is ŷ − y for a one-hot (or any sum-to-1) target
        const f = kind === 'bce' ? frac : '', sc = kind === 'softmax-xent' && Math.abs(d.sumY - 1) > 1e-9;
        els.push(line(`\\delta^{(${l})} = \\frac{\\partial L}{\\partial z^{(${l})}} = ${f}\\big(${sc ? '(\\textstyle\\sum_j y_j)\\,' : ''}\\hat{\\mathbf y} - \\mathbf y\\big)
          \\quad\\text{(${kind === 'bce' ? 'sigmoid + cross-entropy' : 'softmax + cross-entropy'})}`));
        els.push(eqOf('delta', [op(`\\delta^{(${l})} = ${f}\\Big(${sc ? `${f2(d.sumY)}\\,` : ''}`), yhat, op('-'), y, op('\\Big)')],
          [op('='), dVec(`\\delta^{(${l})}`)]));
      } else {
        els.push(line(`\\frac{\\partial L}{\\partial a^{(${l})}} = ${frac}\\big(\\hat{\\mathbf y} - \\mathbf y\\big),\\qquad
          \\delta^{(${l})} = ${kind === 'softmax-mse' ? `J_{\\operatorname{softmax}}^{\\top}\\,\\frac{\\partial L}{\\partial a^{(${l})}}`
          : `\\frac{\\partial L}{\\partial a^{(${l})}} \\odot ${primeVec(act, `z^{(${l})}`)}`}`));
        els.push(eqOf('dA', [op(`\\frac{\\partial L}{\\partial a^{(${l})}} = ${frac}\\Big(`), yhat, op('-'), y, op('\\Big)')],
          [op('='), dAvec()]));
        els.push(kind === 'softmax-mse'
          ? eqOf('delta', [op(`\\delta^{(${l})} = J^{\\top}`), dAvec()], [op('='), dVec(`\\delta^{(${l})}`)])
          : eqOf('delta', [op(`\\delta^{(${l})} =`), dAvec()], [op('\\odot'), spVec()], [op('='), dVec(`\\delta^{(${l})}`)]));
      }
    } else {
      // δ^{(l)} = (Σ_m W^{(m,l)ᵀ} δ^{(m)}) ⊙ σ'(z^{(l)}): the product is bracketed so ⊙ can't be read
      // as binding first. ups is never empty: layer l + 1 always has a term fed by l.
      const ups = upstream(d, l);
      // the attention layer above has no edges: it sends back ∂Q, ∂K, ∂V directly
      const att = d.T?.[l + 1]?.mode === 'attn' ? l + 1 : 0;
      const sum = [...ups.map(u => `\\big(${wName(u.m, l)}\\big)^{\\!\\top}\\delta^{(${u.m})}`),
        ...(att ? [`\\partial [Q\\,K\\,V]^{(${att})}`] : [])].join(' + ');
      const soft = act === 'softmax';
      els.push(line(soft
        ? `\\delta^{(${l})} = J_{\\operatorname{softmax}}^{\\top}\\big(${sum}\\big)`
        : `\\delta^{(${l})} = \\big(${sum}\\big) \\odot ${primeVec(act, `z^{(${l})}`)}`));
      const groups = ups.map((u, j) => [
        op(j ? '+' : `\\delta^{(${l})} = ${soft ? 'J^{\\top}' : ''}\\Big(`),
        wGrid(d, u.m, u.ti, { T: true }),
        nodeVec(d.M[u.m].rows, (dd, i) => dd.bwd.dZ[u.m][i], 'g', () => [`bin:${l}`], `\\delta^{(${u.m})}`),
        j === ups.length - 1 && !att && op('\\Big)'),
      ]);
      if (att) {
        groups.push([op(ups.length ? '+' : `\\delta^{(${l})} = ${soft ? 'J^{\\top}' : ''}\\Big(`),
          nodeVec(ids, (dd, i) => attnGradAt(dd, att, i), 'g', () => [`bin:${l}`], `\\partial [Q\\,K\\,V]`), op('\\Big)')]);
      }
      els.push(eqOf('delta', ...groups, !soft && [op('\\odot'), spVec()], [op('='), dVec(`\\delta^{(${l})}`)]));
    }
    // ∂L/∂W^{(l,k)} = δ^{(l)} a^{(k)ᵀ} (outer product), ∂L/∂b^{(l)} = δ^{(l)}
    const terms = m.terms.map((t, ti) => ({ t, ti })).filter(x => x.t.cols.length);
    const augB = opt.bias && terms[0]?.ti === 0;   // b rides along as the last column of the first term
    els.push(line([...terms.map(({ t, ti }) => (augB && ti === 0
      ? `\\frac{\\partial L}{\\partial [${wName(l, t.k)} \\mid b^{(${l})}]} = \\delta^{(${l})} \\big[${aT(t.k)}\\ 1\\big]`
      : `\\frac{\\partial L}{\\partial ${wName(l, t.k)}} = \\delta^{(${l})}\\, ${aT(t.k)}`)),
    ...(augB ? [] : [`\\frac{\\partial L}{\\partial b^{(${l})}} = \\delta^{(${l})}`])].join(',\\qquad ')));
    for (const { t, ti } of terms) {
      const aug = augB && ti === 0, c = t.cols.length;
      const aRow = grid(1, c + (aug ? 1 : 0), (_, j) => (j === c
        ? { fixed: '1', keys: [`bin:${l}`] }
        : { f: dd => dd.fwd.a[t.k][j], s: 'a', keys: [`n:${t.cols[j]}`, `bin:${l}`],
          hover: { kind: 'node', id: t.cols[j] }, click: { kind: 'node', id: t.cols[j] }, em: { n: [t.cols[j]] } }), {
        aug: aug ? c : null, cap: aug ? `\\big[${aT(t.k)}\\ 1\\big]` : aT(t.k),
      });
      const dW = grid(n, c + (aug ? 1 : 0), (i, j) => {
        const to = ids[i];
        if (j === c) {
          return { f: dd => dd.bwd.db[l][i], s: 'g', keys: [`b:${to}`, `brow:${l}:${i}`],
            hover: { kind: 'bias', id: to }, click: { kind: 'node', id: to }, em: { n: [to] } };
        }
        const from = t.cols[j], eid = t.edge[i][j], e = eid == null ? null : eMap.get(eid);
        const keys = [`brow:${l}:${i}`, `dst:${to}`, `src:${from}`, eid == null ? `p:${from}>${to}` : `e:${eid}`, hasTie(e) && `tie:${e.tie}`];
        return { f: dd => dd.bwd.dZ[l][i] * dd.fwd.a[t.k][j], s: 'g', faint: eid == null || !!e?.fixed, keys,
          em: eid == null ? { p: [from, to] } : { e: [eid] },
          hover: eid == null ? { kind: 'pair', from, to } : { kind: 'edge', id: eid }, click: eid == null ? null : { kind: 'edge', id: eid } };
      }, { aug: aug ? c : null, cap: aug ? `\\partial L / \\partial [${wName(l, t.k)} \\mid b]` : `\\partial L / \\partial ${wName(l, t.k)}` });
      const lead = aug ? `\\frac{\\partial L}{\\partial [${wName(l, t.k)} \\mid b]} =` : `\\frac{\\partial L}{\\partial ${wName(l, t.k)}} =`;
      els.push(eqOf('grad', [op(lead), dVec(`\\delta^{(${l})}`), aRow], [op('='), dW]));
    }
    // δ aᵀ is the full outer product; a masked entry's value is shown faint because nothing uses it
    // (model.backward reports 0 there and training leaves the fixed 0 alone).
    const termEdges = terms.flatMap(({ t }) => t.edge.flat().filter(e => e != null).map(id => eMap.get(id)));
    const anyFixed = termEdges.some(e => e?.fixed);
    if (terms.some(({ t }) => t.edge.some(r => r.includes(null)))) {
      els.push(h('p', 'nm-note', anyFixed
        ? 'Faint entries have no edge (a fixed 0) or a fixed edge, so training never applies them.'
        : 'Faint entries have no edge: that weight is a fixed 0, so training never applies them.'));
    } else if (anyFixed) els.push(h('p', 'nm-note', 'Faint entries are fixed edges, so training never applies them.'));
    const tiedB = ids.some(id => hasTie(model.node(d.net, id)));
    if (termEdges.some(hasTie) || tiedB) {
      els.push(h('p', 'nm-note', `${termEdges.some(hasTie) ? 'Tied entries (hover one to light its group) are' : 'Tied biases are'} one shared parameter: `
        + `its gradient is the sum of those entries, and training moves them all together.${termEdges.some(hasTie) && tiedB ? ' The biases here are tied the same way.' : ''}`));
    }
    if (!augB) {
      els.push(eqOf('db', [op(`\\frac{\\partial L}{\\partial b^{(${l})}} =`), grid(n, 1, i => ({
        f: dd => dd.bwd.db[l][i], s: 'g', keys: [`b:${ids[i]}`, `brow:${l}:${i}`],
        hover: { kind: 'bias', id: ids[i] }, click: { kind: 'node', id: ids[i] }, em: { n: [ids[i]] },
      }), { cap: `\\partial L / \\partial b^{(${l})}` })]));
    }
    return els;
  }

  // ---------------------------------------------------------------- composed formula + collapse
  function composed(d) {
    const { L, M } = d, uses = new Array(L + 1).fill(0);
    const isAttn = l => d.net.layers[l]?.kind === 'attention';
    for (let l = 1; l <= L; l++) {
      for (const t of M[l].terms) uses[t.k]++;
      if (isAttn(l)) uses[l - 1]++;
    }
    const named = l => l > 0 && l < L && uses[l] > 1;
    const expr = (l, top) => {
      if (l === 0) return { s: '\\mathbf x', atom: true };
      if (!top && named(l)) return { s: `a^{(${l})}`, atom: true };
      if (isAttn(l)) return { s: `\\htmlData{l=${l}}{\\operatorname{attn}\\!\\left(${expr(l - 1, false).s}\\right)}`, atom: true };
      const parts = M[l].terms.map(t => {
        const e = expr(t.k, false);
        return `${wName(l, t.k)}${e.atom ? `\\,${e.s}` : `\\left(${e.s}\\right)`}`;
      });
      const sum = [...parts, `b^{(${l})}`].join(' + ');
      const act = actTex(d.net.layers[l].act);
      return act
        ? { s: `\\htmlData{l=${l}}{${act}\\!\\left(${sum}\\right)}`, atom: true }
        : { s: `\\htmlData{l=${l}}{${sum}}`, atom: false };
    };
    const lines = [`\\hat{\\mathbf y} = ${expr(L, true).s}`];
    let first = true;
    for (let l = 1; l < L; l++) {
      if (!named(l)) continue;
      lines.push(`${first ? '\\text{where }' : ''}a^{(${l})} = ${expr(l, true).s}`);
      first = false;
    }
    return lines;
  }

  // The same for a net of token layers (every layer compact or attention), in token matrices:
  // a transformer block as one line, with layers used more than once named in "where" lines.
  const tokNet = d => d.T && d.L >= 1 && d.T.slice(1).every(S => S.mode === 'tok' || S.mode === 'attn');

  function tokComposed(d) {
    const { L, T } = d, uses = T.map(S => new Array(S.G).fill(0)), force = new Set();
    for (let l = 1; l <= L; l++) {
      const S = T[l];
      if (S.mode === 'attn') {
        for (const g of [S.gq, S.gk, S.gv]) uses[l - 1][g]++;
        if (S.heads > 1) force.add(l - 1);
      }
      for (const b of S.blocks) uses[b.k][b.gs]++;
      for (const r of S.resid) uses[r.k].forEach((_, g) => uses[r.k][g]++);
    }
    const named = k => k > 0 && k < L && (force.has(k) || uses[k].some(u => u > 1));
    const mul = e => (e.kind === 'sum' ? `\\left(${e.s}\\right)` : e.s);
    const tr = e => (e.kind === 'sym' ? wT(e.s) : `\\left(${e.s}\\right)^{\\!\\top}`);
    const expr = (k, g, top) => {
      if (k === 0 || (!top && named(k))) return { s: symOf(d, k, g), kind: 'sym' };
      const S = T[k], wrap = s => `\\htmlData{l=${k}}{${s}}`, M = S.causal ? ' + M' : '';
      if (S.mode === 'attn') {
        if (S.heads > 1) {   // heads side by side; Q_h is head h's columns of Q
          const [q, kk, v] = [S.gq, S.gk, S.gv].map(x => symOf(d, k - 1, x));
          const head = h => `\\operatorname{softmax}\\!\\Big(${scoreTex(S, `${q}_{${h}}`, wT(`${kk}_{${h}}`))}${M}\\Big)\\,${v}_{${h}}`;
          const hs = S.heads <= 3 ? Array.from({ length: S.heads }, (_, h) => head(h + 1)) : [head(1), '\\cdots', head(S.heads)];
          return { s: wrap(`\\Big[\\,${hs.join('\\;\\Big|\\;')}\\,\\Big]`), kind: 'sym' };
        }
        const q = expr(k - 1, S.gq, false), kk = expr(k - 1, S.gk, false), v = expr(k - 1, S.gv, false);
        return { s: wrap(`\\operatorname{softmax}\\!\\Big(${scoreTex(S, mul(q), tr(kk))}${M}\\Big)\\,${mul(v)}`), kind: 'prod' };
      }
      const parts = [];
      for (const b of S.blocks) if (b.gd === g) parts.push(`${mul(expr(b.k, b.gs, false))}\\,${b.name}`);
      for (const r of S.resid) parts.push(expr(r.k, g, false).s);
      if (d.bflag[k]?.[g]) parts.push(bSym(d, k, g));
      if (!parts.length) parts.push('0');
      const act = actTex(S.act), sum = parts.join(' + ');
      if (act) return { s: wrap(`${act}\\!\\left(${sum}\\right)`), kind: 'sym' };
      return { s: wrap(sum), kind: parts.length > 1 ? 'sum' : 'prod' };
    };
    const def = (k, lhs) => (T[k].groups
      ? T[k].groups.map((_, g) => `${symOf(d, k, g)} = ${expr(k, g, true).s}`).join(',\\quad ')
      : `${lhs ?? symOf(d, k)} = ${expr(k, 0, true).s}`);
    const lines = [def(L, '\\hat Y')];
    let first = true;
    for (let k = 1; k < L; k++) {
      if (!named(k)) continue;
      lines.push(`${first ? '\\text{where }' : ''}${def(k)}`);
      first = false;
      const S = T[k + 1];
      if (S?.mode === 'attn' && S.heads > 1) {
        const [q, kk, v] = [S.gq, S.gk, S.gv].map(x => symOf(d, k, x));
        lines.push(`${q}_h, ${kk}_h, ${v}_h = \\text{head } h\\text{'s ${S.dh} column${S.dh > 1 ? 's' : ''} of } ${q}, ${kk}, ${v}`);
      }
    }
    return lines;
  }

  function buildFormula(d) {
    const box = h('div', 'nm-formula');
    const tn = tokNet(d);
    for (const s of tn ? tokComposed(d) : composed(d)) box.append(line(s, 'nm-fl'));
    if (tn) {
      box.append(h('p', 'nm-note', 'Rows are tokens. Each W is one shared matrix applied to every token (weight tying); '
        + 'softmax acts on each row. Every step is a matrix product: hover a symbol to find its layer.'));
    }
    for (const el of box.querySelectorAll('[data-l]')) {
      const l = +el.dataset.l, id = d.net.layers[l]?.id;
      reg(el, [`L:${l}`]);
      hoverOf.set(el, { kind: 'layer', id });
      clickOf.set(el, { kind: 'layer', id });
      formulaEls.push({ el, l });
    }
    return box;
  }

  function buildCollapse(d) {
    const box = h('div', 'nm-collapse'), c = d.col, L = d.L;
    const n0 = model.nodesIn(d.net, 0).length, nL = d.M[L].rows.length;
    const vec = (f, n, s, cap) => grid(n, 1, i => ({ f: dd => f(dd, i), s }), { cap });
    const Weff = () => grid(nL, n0, (i, j) => ({ f: dd => dd.col.W[i][j], s: 'eff' }), { cap: 'W_{\\text{eff}}' });
    const beff = () => vec((dd, i) => dd.col.b[i], nL, 'eff', 'b_{\\text{eff}}');
    const yLin = cap => vec((dd, i) => dd.col.y[i], nL, 'a', cap);
    const yhat = () => nodeVec(d.M[L].rows, (dd, i) => dd.fwd.a[L][i], 'a', () => [], '\\hat{\\mathbf y}');
    if (c.attn) {
      box.append(h('p', 'nm-note', 'Can\'t collapse: attention multiplies activations together (Q Kᵀ, then A V), '
        + 'so even with every activation removed the network is not one matrix. Its A changes with every input.'));
      return box;
    }
    if (c.ok) {
      box.append(line('\\text{Every layer is identity, so the whole network is one affine map: }'
        + '\\hat{\\mathbf y} = W_{\\text{eff}}\\,\\mathbf x + b_{\\text{eff}}'));
      if (!c.skip) {
        const Ws = [];
        for (let l = L; l >= 1; l--) Ws.push([wGrid(d, l, 0)]);
        box.append(eqOf('chain', ...Ws, [op('='), Weff()]));
      } else {
        box.append(eqOf('chain', [op('W_{\\text{eff}} ='), Weff()]));
        box.append(line('\\text{(skip edges add their own products to } W_{\\text{eff}}\\text{)}', 'nm-note'));
      }
      // without skips, b_eff = W^{(L)}⋯W^{(2)} b^{(1)} + ⋯ + W^{(L)} b^{(L-1)} + b^{(L)}
      const bSym = c.skip ? '' : Array.from({ length: L }, (_, q) => {
        let s = '';
        for (let m = L; m > q + 1; m--) s += `W^{(${m})}`;
        return `${s}b^{(${q + 1})}`;
      }).join(' + ') + ' =';
      box.append(eqOf('eff', [op(`b_{\\text{eff}} = ${bSym}`), beff()],
        [op('\\qquad W_{\\text{eff}}\\,\\mathbf x + b_{\\text{eff}} ='), yLin(''), op('= \\hat{\\mathbf y}')]));
      if (n0 <= 3 && nL <= 3 && n0 * nL > 1) {
        const b = h('button', 'nm-3d', '→ 3D');
        b.title = 'Send W_eff to the 3D tab';
        b.onclick = () => send3d('eff');
        box.append(b);
      }
    } else {
      const hidden = c.bad.filter(l => l < L), actOf = l => d.net.layers[l].act;
      const note = text => box.append(h('p', 'nm-note', text));
      if (!hidden.length) {
        note(`Only the output layer (${actLabel(actOf(L))}) is nonlinear. Everything inside it collapses to one affine map:`);
        box.append(line(`\\hat{\\mathbf y} = ${actTex(actOf(L))}\\big(W_{\\text{eff}}\\,\\mathbf x + b_{\\text{eff}}\\big)`));
        if (c.W) {
          box.append(eqOf('lin', [op('W_{\\text{eff}}\\,\\mathbf x + b_{\\text{eff}} ='), yLin(''), op(`= z^{(${L})}`)],
            [op(`\\qquad \\hat{\\mathbf y} = ${actTex(actOf(L))}(z^{(${L})}) =`), yhat()]));
        }
      } else {
        const names = c.bad.map(l => `layer ${l} (${actLabel(actOf(l))})`);
        const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names.at(-1)}` : names[0];
        const k = hidden[0], s = actTex(actOf(k));
        note(`Can't collapse: ${list} ${names.length > 1 ? 'are' : 'is'} nonlinear. A product of matrices is one matrix, `
          + 'but the nonlinearity sits between the layers:');
        box.append(line(`W^{(${k + 1})}\\,${s}\\big(W^{(${k})}\\,${aName(k - 1)} + b^{(${k})}\\big) \\;\\ne\\; `
          + `\\big(W^{(${k + 1})}W^{(${k})}\\big)\\,${aName(k - 1)} + \\dots`));
        // "vs", not "≠": at one input the two can agree (ReLU units that are all active, say)
        note('That is why depth adds expressive power. With every activation removed, the network would give:');
        if (c.W) box.append(eqOf('lin', [op('W_{\\text{eff}}\\,\\mathbf x + b_{\\text{eff}} ='), yLin('')], [op('\\quad\\text{vs.}\\quad \\hat{\\mathbf y} ='), yhat()]));
      }
    }
    return box;
  }

  // ---------------------------------------------------------------- token layers: grids
  // t_1 ... t_n row (or column) headers. at(t) -> { hover, keys } makes each one hover its token
  // (store hover { kind: 'token', layer, t, g?, h? }) and light with it.
  // net.meta.tokenNames (docs/NN_LENS.md) replace t_1 ... t_n wherever a name is set (as focus.js's tokenLabel).
  const tokName = t => (typeof names?.[t] === 'string' && names[t] ? names[t] : null);
  const tokTex = t => { const s = tokName(t); return s ? `\\text{${texEsc(s)}}` : `t_{${t + 1}}`; };
  const tokHdr = (n, at = null) => Array.from({ length: n }, (_, t) => ({ tex: tokTex(t), ...(at ? at(t) : {}) }));
  const tokSpec = (layer, t, g = null, hh = null) => ({ kind: 'token', layer, t, ...(g == null ? {} : { g }), ...(hh == null ? {} : { h: hh }) });
  // A token header of layer k (group g): hovers the token, lights with its neurons.
  const tokAt = (d, k, g) => t => {
    const S = d.T[k], ids = Array.from({ length: S.d }, (_, f) => S.ids[tidx(S, g, t, f)]);
    return { hover: tokSpec(k, t, S.groups ? g : null), keys: ids.map(id => `n:${id}`), em: { n: ids }, tok: { l: k, t } };
  };
  const sub = (text, cls = 'nm-sub') => h('div', cls, text);
  // A stage of a layer that lens.focus.part can pick ('scores', 'softmax', 'mix'; groups mark their eq)
  const stage = (part, ...els) => {
    const s = h('div', 'nm-stage');
    s.dataset.part = part;
    s.append(...els);
    return s;
  };
  // One head's block of an attention layer: lens.head hides the others.
  const headBox = (l, hh) => {
    const b = h('div', 'nm-headbox');
    b.dataset.head = hh;
    headBoxes.push({ el: b, l, hh });
    return b;
  };
  // A wrapping note mixing text and TeX: parts are strings, or { t: tex } for inline maths.
  const note = (...parts) => {
    const p = h('p', 'nm-note');
    for (const x of parts) p.append(typeof x === 'string' ? x : tex(h('span'), x.t));
    return p;
  };
  const ones = (n, m, cap) => grid(n, m, () => ({ fixed: '1', cls: 'nm-one' }), { cap });

  // tokens × d grid of layer l, group g: f(dd, i) with i the node index. keys(i, t) adds keys.
  function tvals(d, l, g, f, s, cap, keys = () => [], o = {}) {
    const S = d.T[l];
    return grid(S.tokens, S.d, (t, ff) => {
      const i = tidx(S, g, t, ff), id = S.ids[i];
      return { f: dd => f(dd, i), s, keys: [`n:${id}`, ...keys(i, t)], hover: { kind: 'node', id }, click: { kind: 'node', id }, em: { n: [id] } };
    }, { cap, rowTok: l, ...o });
  }

  // Layer k's activations (group g) as the left operand of layer l's product, or transposed.
  function srcGrid(d, l, k, g, { T = false } = {}) {
    const S = d.T[k], n = S.tokens, w = S.d, sym = symOf(d, k, g);
    const cell = (t, f) => {
      const i = tidx(S, g, t, f), id = S.ids[i];
      return { f: dd => dd.fwd.a[k][i], s: 'a', keys: [`n:${id}`, `${T ? 'tbin' : 'tin'}:${l}:${t}`],
        hover: { kind: 'node', id }, click: { kind: 'node', id }, em: { n: [id] } };
    };
    if (T) return grid(w, n, (f, t) => cell(t, f), { cap: wT(sym), colHdr: tokHdr(n, tokAt(d, k, g)), colTok: k });
    return grid(n, w, cell, { cap: sym, rowHdr: tokHdr(n, tokAt(d, k, g)), rowTok: k });
  }

  const valGrid = (d, l, g, which, cap) => tvals(d, l, g, (dd, i) => dd.fwd[which][l][i], 'a',
    cap ?? (which === 'z' ? preSym(d, l, g) : symOf(d, l, g)), i => [`out:${l}:${i}`]);

  // + B: a shared row vector as 1 bᵀ (a column of ones times the row), else the tokens × d matrix
  const biasEls = (d, l, g) => {
    const S = d.T[l], bt = bTied(d, l, g);
    if (bt) {
      const col = f => Array.from({ length: S.tokens }, (_, t) => tidx(S, g, t, f));
      return [ones(S.tokens, 1, '\\mathbf 1'),
        grid(1, S.d, (_, f) => {
          const i = tidx(S, g, 0, f), id = S.ids[i];
          return { f: dd => dd.M[l].b[i], s: 'b', keys: [...col(f).flatMap(k => [`b:${S.ids[k]}`, `row:${l}:${k}`, `dst:${S.ids[k]}`])],
            hover: { kind: 'bias', id }, click: { kind: 'node', id }, em: { n: col(f).map(k => S.ids[k]) } };
        }, { cap: wT(bt.name) })];
    }
    return [grid(S.tokens, S.d, (t, f) => {
      const i = tidx(S, g, t, f), id = S.ids[i];
      return { f: dd => dd.M[l].b[i], s: 'b', keys: [`b:${id}`, `row:${l}:${i}`, `dst:${id}`],
        hover: { kind: 'bias', id }, click: { kind: 'node', id }, em: { n: [id] } };
    }, { cap: bSym(d, l, g), rowTok: l })];
  };

  // A tied block's small matrix (rows = input feature, cols = output feature: Q = X W_Q), or its
  // transpose inside a δ. Each cell is one shared parameter: hover lights its whole tie group.
  function tieGrid(d, l, b, { T = false } = {}) {
    const S = d.T[l], src = d.T[b.k];
    const cell = (fs, fd) => {
      const c = b.cells[fs][fd];
      if (!c) return { mask: true, fixed: '0' };
      const keys = [`tie:${c.tie}`, ...c.byTok.map(id => `e:${id}`), ...c.dst.map(id => `dst:${id}`), ...c.src.map(id => `src:${id}`),
        ...(T ? c.cols.map(j => `brow:${b.k}:${j}`) : c.rows.map(i => `row:${l}:${i}`))];
      return { f: dd => dd.M[l].terms[b.ti].W[c.i][c.j], s: 'w', keys, cls: 'nm-tied',
        hover: { kind: 'edge', id: c.rep }, click: { kind: 'edge', id: c.rep }, em: { e: c.byTok.filter(x => x != null), w: true } };
    };
    if (T) return grid(S.d, src.d, (fd, fs) => cell(fs, fd), { cap: wT(b.name) });
    return grid(src.d, S.d, cell, { cap: b.name });
  }

  // ---------------------------------------------------------------- token layers: forward
  function localTex(d, l) {
    const S = d.T[l];
    const one = g => {
      const parts = [];
      for (const b of S.blocks) if (b.gd === g) parts.push(`${symOf(d, b.k, b.gs)}\\,${b.name}`);
      for (const r of S.resid) parts.push(symOf(d, r.k, g));
      if (d.bflag[l]?.[g]) parts.push(bSym(d, l, g));
      const sum = parts.join(' + ') || '0', act = actTex(S.act);
      return `${symOf(d, l, g)} = ${act ? `${act}\\!\\left(${sum}\\right)` : sum}`;
    };
    return Array.from({ length: S.G }, (_, g) => one(g)).join(',\\qquad ');
  }

  function tokFwd(d, l) {
    const S = d.T[l], els = [line(localTex(d, l))];
    for (let g = 0; g < S.G; g++) {
      const grps = [];
      const lead = () => op(grps.length ? '+' : `${S.act === 'identity' ? symOf(d, l, g) : preSym(d, l, g)} =`);
      for (const b of S.blocks) if (b.gd === g) grps.push([lead(), srcGrid(d, l, b.k, b.gs), tieGrid(d, l, b)]);
      for (const r of S.resid) grps.push([lead(), srcGrid(d, l, r.k, g)]);
      if (d.bflag[l]?.[g] || !grps.length) grps.push([lead(), ...biasEls(d, l, g)]);
      const outs = S.act === 'identity' ? [[op('='), valGrid(d, l, g, 'a')]]
        : [[op('='), valGrid(d, l, g, 'z')], [op(arrow(d, l)), valGrid(d, l, g, 'a')]];
      const eq = eqOf('fwd', ...grps, ...outs);
      if (S.groups) eq.dataset.part = S.groups[g];   // lens.focus.part 'Q' / 'K' / 'V'
      els.push(eq);
    }
    for (const r of S.resid) {
      els.push(note({ t: `+\\,${symOf(d, r.k)}` }, `: a residual (skip) connection from layer ${r.k}, a fixed identity `
        + '(edges fixed at 1). It adds that layer back in and never trains.'));
    }
    if (S.act === 'softmax') els.push(h('p', 'nm-note', 'Softmax acts on each token\'s row separately.'));
    return els;
  }

  // The flattened view: the same layer as z = W a with W built from I ⊗ Wᵀ blocks.
  function flatEls(d, l) {
    const S = d.T[l], m = d.M[l], els = [];
    const used = m.terms.map((t, ti) => ({ t, ti })).filter(({ t }) => t.edge.some(r => r.some(e => e != null)));
    const kron = (t, ti) => {
      if (S.resid.some(r => r.ti === ti)) return `I_{${S.ids.length}}\\ \\text{(fixed)}`;
      const src = d.T[t.k];
      const cell = (gd, gs) => {
        const b = S.blocks.find(x => x.ti === ti && x.gd === gd && x.gs === gs);
        return b ? `I_{${S.tokens}} \\otimes ${wT(b.name)}` : '0';
      };
      if (S.G === 1 && src.G === 1) return cell(0, 0);
      const rows = Array.from({ length: S.G }, (_, gd) => Array.from({ length: src.G }, (_, gs) => cell(gd, gs)).join(' & '));
      return `\\begin{bmatrix}${rows.join(' \\\\ ')}\\end{bmatrix}`;
    };
    els.push(line(`z^{(${l})} = ${used.map(({ t }) => `${wName(l, t.k)}\\,${aName(t.k)}`).join(' + ')} + b^{(${l})},\\qquad `
      + used.map(({ t, ti }) => `${wName(l, t.k)} = ${kron(t, ti)}`).join(',\\quad ')));
    const b0 = S.blocks[0];
    if (b0) {
      const xs = symOf(d, b0.k, b0.gs);
      els.push(line(`\\operatorname{vec}\\big(${xs}\\,${b0.name}\\big) = \\big(I_{${S.tokens}} \\otimes ${wT(b0.name)}\\big)\\operatorname{vec}(${xs})`));
      els.push(h('p', 'nm-note', 'vec stacks the token rows into one column, in node order (token-major). The block '
        + `repeats down the diagonal once per token (${S.tokens} copies), and every copy is the same parameter: that is weight tying.`));
    }
    els.push(fwdEq(d, l));
    return els;
  }

  function expander(d, l) {
    const id = d.net.layers[l].id, open = expanded().has(id);
    const box = h('div', open ? 'nm-expand open' : 'nm-expand');
    const b = h('button', 'nm-exp-btn', `${open ? '▾' : '▸'} Flattened: z = W a`);
    b.type = 'button';
    b.title = 'The same layer as one matrix product z = W a + b on the flattened vectors (I ⊗ Wᵀ blocks)';
    b.onclick = () => toggleExpand(id);
    box.append(b);
    if (open) box.append(...flatEls(d, l));
    return box;
  }

  // ---------------------------------------------------------------- attention: forward
  // Head hh's Q / K / V / Z (tokens × width, or transposed). f(dd, t, c).
  function headMat(d, l, hh, which, f, s, cap, { T = false, keys = () => [] } = {}) {
    const S = d.T[l], w = which === 'q' || which === 'k' ? S.dh : S.dvh, n = S.tokens;
    const cell = (t, c) => {
      const id = headNode(S, hh, which, t, c);
      return { f: dd => f(dd, t, c), s, keys: [`n:${id}`, ...keys(t, c)], hover: { kind: 'node', id }, click: { kind: 'node', id }, em: { n: [id] } };
    };
    // headers hover the token: Z's (this layer, head hh) or Q's, K's, V's (the layer before)
    const tl = which === 'z' ? l : l - 1;
    const at = t => {
      const w2 = which === 'q' || which === 'k' ? S.dh : S.dvh;
      const ids = Array.from({ length: w2 }, (_, c) => headNode(S, hh, which, t, c));
      return { hover: which === 'z' ? tokSpec(l, t, null, S.heads > 1 ? hh : null)
          : tokSpec(l - 1, t, which === 'q' ? S.gq : which === 'k' ? S.gk : S.gv, S.heads > 1 ? hh : null),
        keys: ids.map(id => `n:${id}`), em: { n: ids }, tok: { l: tl, t } };
    };
    return T ? grid(w, n, (c, t) => cell(t, c), { cap, colHdr: tokHdr(n, at), colTok: tl })
      : grid(n, w, cell, { cap, rowHdr: tokHdr(n, at), rowTok: tl });
  }

  // tokens × tokens (S, A, ∂A, ∂S): row i = query token, column j = key token. Hovering a cell
  // hovers token i's attention row in head hh ({ kind: 'token' }: the view shows A_i, the cards
  // light row i); keys ahr / ahc light that row / column of every such matrix of the head.
  // masked: text for causal cells, or null.
  // The lens weighs a cell by its query row (rows(l)); an A cell below minA is dimmed too (thr).
  function sqMat(d, l, hh, f, s, cap, { T = false, keys = () => [], masked = null, heat = false, thr = false } = {}) {
    const S = d.T[l], n = S.tokens, H = S.heads > 1 ? hh : null;
    const cell = (i, j) => {
      const z = Array.from({ length: S.dvh }, (_, c) => headNode(S, hh, 'z', i, c));
      const src = [...Array.from({ length: S.dh }, (_, c) => [headNode(S, hh, 'q', i, c), headNode(S, hh, 'k', j, c)]).flat(),
        ...Array.from({ length: S.dvh }, (_, c) => headNode(S, hh, 'v', j, c))];
      const sp = { keys: [...z.map(id => `dst:${id}`), ...src.map(id => `src:${id}`), `ahr:${l}:${hh}:${i}`, `ahc:${l}:${hh}:${j}`, ...keys(i, j)],
        hover: tokSpec(l, i, null, H), click: { kind: 'node', id: z[0] }, cls: heat ? 'nm-heat' : '',
        em: { r: [l, i], ...(thr ? { a: [l, i, j, hh], lg: true } : {}) } };
      if (masked != null && S.causal && j > i) return { ...sp, mask: true, fixed: masked, cls: `${sp.cls} nm-inf`, em: { r: [l, i] } };
      return { ...sp, f: dd => f(dd, i, j), s };
    };
    // row headers: token i's row (Z_i); column headers: key / value token j of the layer before
    const rows = tokHdr(n, i => ({ hover: tokSpec(l, i, null, H), keys: [`ahr:${l}:${hh}:${i}`], em: { r: [l, i] }, tok: { l, t: i } }));
    const cols = tokHdr(n, j => ({ hover: tokSpec(l - 1, j, S.gv, H), keys: [`ahc:${l}:${hh}:${j}`] }));
    return T ? grid(n, n, (i, j) => cell(j, i), { cap, colTok: l }) : grid(n, n, cell, { cap, rowHdr: rows, colHdr: cols, rowTok: l });
  }

  function attnLocalTex(d, l) {
    const S = d.T[l], M = S.causal ? ' + M' : '', H = S.heads;
    const z = `Z^{(${l})}`;
    const main = H === 1
      ? `S = ${scoreTex(S, 'Q', wT('K'))}${M},\\qquad A = \\operatorname{softmax}(S),\\qquad ${z} = A\\,V`
      : `S_h = ${scoreTex(S, 'Q_h', wT('K_h'))}${M},\\quad A_h = \\operatorname{softmax}(S_h),\\quad Z_h = A_h V_h,\\qquad ${z} = \\big[${
        H <= 3 ? Array.from({ length: H }, (_, h) => `Z_{${h + 1}}`).join(' \\mid ') : `Z_1 \\mid \\cdots \\mid Z_{${H}}`}\\big]`;
    return main;
  }

  function attnHeadNote(d, l) {
    const S = d.T[l], P = S.P;
    const src = [S.gq, S.gk, S.gv].map(g => symOf(d, l - 1, g)).join(', ');
    return `Q, K, V are layer ${l - 1}'s groups (${src}). d_k = ${S.dh}${S.heads > 1 ? ` per head (${S.heads} heads split the ${P.d} features)` : ''}`
      + `, scale ${S.scaleDef ? `1/√${S.dh}` : model.fmt(S.scale, 3)}${S.causal ? '; causal: M is −∞ above the diagonal, so token i only sees tokens ≤ i' : ''}.`;
  }

  function attnFwd(d, l) {
    const S = d.T[l], H = S.heads, n = S.tokens, els = [line(attnLocalTex(d, l)), h('p', 'nm-note', attnHeadNote(d, l))];
    if (!d.fwd.attn?.[l]?.heads?.length) { els.push(h('p', 'nm-note', 'The model returned no attention data for this layer.')); return els; }
    const sc = scaleNum(S);
    const zFull = () => tvals(d, l, 0, (dd, i) => dd.fwd.a[l][i], 'a', `Z^{(${l})}`, (i, t) => [`out:${l}:${i}`, `az:${l}:${t}`]);
    for (let hh = 0; hh < H; hh++) {
      const s = H > 1 ? `_{${hh + 1}}` : '', F = dd => dd.fwd.attn[l].heads[hh];
      const hb = headBox(l, hh);
      if (H > 1) hb.append(sub(`Head ${hh + 1}`, 'nm-sub nm-head'));
      const A = () => sqMat(d, l, hh, (dd, i, j) => F(dd).A[i][j], 'one', `A${s}`, { keys: i => [`aa:${l}:${i}`], masked: '0', heat: true, thr: true });
      const gS = [[op(`S${s} = ${sc}`),
        headMat(d, l, hh, 'q', (dd, t, c) => F(dd).Q[t][c], 'a', `Q${s}`, { keys: t => [`aq:${l}:${t}`] }),
        headMat(d, l, hh, 'k', (dd, t, c) => F(dd).K[t][c], 'a', wT(`K${s}`), { T: true, keys: () => [`ak:${l}`] })]];
      if (S.causal) {
        gS.push([op('+'), grid(n, n, (i, j) => ({ fixed: j > i ? '−∞' : '0', cls: j > i ? 'nm-inf' : 'nm-zero', em: { r: [l, i] } }), { cap: 'M', rowTok: l })]);
      }
      const Z = H > 1 ? headMat(d, l, hh, 'z', (dd, t, c) => F(dd).Z[t][c], 'a', `Z${s}`, { keys: t => [`az:${l}:${t}`] }) : zFull();
      hb.append(
        stage('scores', sub('1 · scores: every query against every key'),
          eqOf('scores', ...gS, [op('='), sqMat(d, l, hh, (dd, i, j) => F(dd).S[i][j], 's', `S${s}`, { keys: i => [`as:${l}:${i}`], masked: '−∞' })])),
        stage('softmax', sub('2 · softmax of each row: the attention weights (each row sums to 1)'),
          eqOf('softmax', [op(`A${s} = \\operatorname{softmax}(S${s}) =`), A()])),
        stage('mix', sub('3 · weighted sum of the values'),
          eqOf('mix', [op(`${H > 1 ? `Z${s}` : `Z^{(${l})}`} =`), A(),
            headMat(d, l, hh, 'v', (dd, t, c) => F(dd).V[t][c], 'a', `V${s}`, { keys: () => [`av:${l}`] })], [op('='), Z])));
      els.push(hb);
    }
    if (H > 1) {
      els.push(stage('mix', eqOf('concat', [op(`Z^{(${l})} = \\big[${Array.from({ length: H }, (_, hh) => `Z_{${hh + 1}}`).join(' \\mid ')}\\big] =`), zFull()])));
    }
    els.push(h('p', 'nm-note', 'No fixed W here: A is computed from this input, so this layer is not z = W a + b. '
      + 'Flattened token-major it is vec Z = (A ⊗ I) vec V, a matrix that changes with every input.'));
    return els;
  }

  function attnBatch(d, l) {
    const S = d.T[l], m = d.M[l], B = d.batch.X.length;
    const out = grid(m.rows.length, B, (i, s) => ({
      f: dd => dd.batch.a[l][i][s], s: 'a', keys: [`n:${m.rows[i]}`, `out:${l}:${i}`],
      hover: { kind: 'node', id: m.rows[i] }, click: { kind: 'node', id: m.rows[i] }, em: { n: [m.rows[i]] },
    }), { cap: `Z^{(${l})}\\ \\text{(one column per sample)}` });
    return [line(attnLocalTex(d, l)),
      eqOf('fwd', [op(`Z^{(${l})} =`), out]),
      h('p', 'nm-note', `Attention has no fixed W: each sample gets its own A, so the batch can't be one product W X. `
        + `The columns are the ${S.tokens * S.d} outputs of each sample, token-major.`)];
  }

  // ---------------------------------------------------------------- token layers: backward
  function tokDelta(d, l) {
    const S = d.T[l], els = [], act = S.act, idn = act === 'identity', soft = act === 'softmax';
    const bk = i => [`brow:${l}:${i}`];
    const dZg = (g, cap) => tvals(d, l, g, (dd, i) => dd.bwd.dZ[l][i], 'g', cap ?? dSym(d, l, g), bk);
    const dAg = (g, cap) => tvals(d, l, g, (dd, i) => dd.bwd.dA[l][i], 'g', cap ?? `\\partial L / \\partial ${symOf(d, l, g)}`, bk);
    const spg = g => tvals(d, l, g, (dd, i) => dd.sp[l][i], 'one', primeTex(act, preSym(d, l, g)), bk);
    const deltaFrom = (g, dA) => (soft
      ? eqOf('delta', [op(`${dSym(d, l, g)} = J^{\\top}`), dA()], [op('='), dZg(g)])
      : eqOf('delta', [op(`${dSym(d, l, g)} =`), dA()], [op('\\odot'), spg(g)], [op('='), dZg(g)]));
    if (l === d.L) {
      const kind = outCase(d), n = S.ids.length, fr = n > 1 ? `\\tfrac{1}{${n}}` : '';
      for (let g = 0; g < S.G; g++) {
        const Yh = () => tvals(d, l, g, (dd, i) => dd.fwd.a[l][i], 'a', S.groups ? symOf(d, l, g) : '\\hat Y', bk);
        const Y = () => tvals(d, l, g, (dd, i) => dd.y[i], 'a', 'Y', bk);
        if (kind === 'softmax-xent' || kind === 'bce') {
          // bce: mean over all n outputs; softmax + xent: softmax on each of the m rows, loss averaged over them
          const rows = kind === 'softmax-xent' ? outRows(d) : null, m = rows?.m || 1;
          const f = kind === 'bce' ? fr : m > 1 ? `\\tfrac{1}{${m}}` : '', proper = !rows || rows.proper;
          els.push(line(`${dSym(d, l, g)} = \\frac{\\partial L}{\\partial ${preSym(d, l, g)}} = ${f}\\big(${proper ? '' : '(\\textstyle\\sum_{\\text{row}} Y)\\,'}\\hat Y - Y\\big)\\quad`
            + `\\text{(${kind === 'bce' ? 'sigmoid + cross-entropy' : `softmax + cross-entropy on each row${m > 1 ? `, averaged over the ${m} rows` : ''}`})}`));
          els.push(proper ? eqOf('delta', [op(`${dSym(d, l, g)} = ${f}\\Big(`), Yh(), op('-'), Y(), op('\\Big)')], [op('='), dZg(g)])
            : eqOf('delta', [op(`${dSym(d, l, g)} =`), dZg(g)]));
        } else {
          const lead = `${idn ? `${dSym(d, l, g)} = ` : ''}\\frac{\\partial L}{\\partial \\hat Y} = ${fr}\\Big(`;
          els.push(eqOf(idn ? 'delta' : 'dA', [op(lead), Yh(), op('-'), Y(), op('\\Big)')], [op('='), idn ? dZg(g) : dAg(g, '\\partial L / \\partial \\hat Y')]));
          if (!idn) els.push(deltaFrom(g, () => dAg(g, '\\partial L / \\partial \\hat Y')));
        }
      }
      return els;
    }
    // ∂L/∂(this layer), group by group: what each layer above sends back
    const contrib = Array.from({ length: S.G }, () => []);
    const up = (m, g) => tvals(d, m, g, (dd, i) => dd.bwd.dZ[m][i], 'g', dSym(d, m, g), () => [`bin:${l}`]);
    for (let m = l + 1; m <= d.L; m++) {
      const Sm = d.T[m];
      if (Sm.mode === 'attn' && m === l + 1) {
        [[Sm.gq, 'Q'], [Sm.gk, 'K'], [Sm.gv, 'V']].forEach(([g, nm]) => contrib[g].push({
          tex: `\\partial ${nm}`, plain: true,
          grids: () => [tvals(d, l, g, (dd, i) => attnGradAt(dd, m, i), 'g', `\\partial ${nm}`, () => [`bin:${l}`])],
        }));
        continue;
      }
      if (Sm.mode === 'tok') {
        for (const b of Sm.blocks) {
          if (b.k === l) contrib[b.gs].push({ tex: `${dSym(d, m, b.gd)}\\,${wT(b.name)}`, grids: () => [up(m, b.gd), tieGrid(d, m, b, { T: true })] });
        }
        for (const r of Sm.resid) {
          if (r.k === l) for (let g = 0; g < S.G; g++) contrib[g].push({ tex: dSym(d, m, g), plain: true, grids: () => [up(m, g)] });
        }
        continue;
      }
      d.M[m].terms.forEach((t, ti) => {
        if (t.k !== l || !t.edge.some(r => r.some(e => e != null))) return;
        const tx = `\\big(${wName(m, l)}\\big)^{\\!\\top}\\delta^{(${m})}`;
        for (let g = 0; g < S.G; g++) {
          contrib[g].push({ tex: tx, plain: true, grids: () => [tvals(d, l, g,
            (dd, i) => dd.M[m].terms[ti].W.reduce((s, row, r) => s + row[i] * dd.bwd.dZ[m][r], 0), 'g', tx, () => [`bin:${l}`])] });
        }
      });
    }
    const symLines = [];
    for (let g = 0; g < S.G; g++) {
      const cs = contrib[g], sym = symOf(d, l, g);
      const dL = `\\frac{\\partial L}{\\partial ${sym}}`, sum = cs.map(c => c.tex).join(' + ') || '0';
      symLines.push(idn ? `${dSym(d, l, g)} = ${dL} = ${sum}`
        : `${dL} = ${sum},\\quad ${dSym(d, l, g)} = ${soft ? `J^{\\top}_{\\operatorname{softmax}}${dL}` : `${dL} \\odot ${primeTex(act, preSym(d, l, g))}`}`);
    }
    // (the equations below already read δ = ∂L/∂(·) = ∂Q when nothing is multiplied)
    if (!idn || contrib.some(cs => cs.length > 1 || cs.some(c => !c.plain))) els.push(line(symLines.join(',\\qquad ')));
    if (contrib.some(cs => cs.some(c => c.tex.startsWith('\\delta') && c.plain))) {
      els.push(h('p', 'nm-note', 'A residual passes δ straight back: the fixed identity adds the upper layer\'s δ unchanged.'));
    }
    for (let g = 0; g < S.G; g++) {
      const cs = contrib[g], sym = symOf(d, l, g);
      const lead = idn ? `${dSym(d, l, g)} = \\frac{\\partial L}{\\partial ${sym}}` : `\\frac{\\partial L}{\\partial ${sym}}`;
      const res = () => (idn ? dZg(g) : dAg(g));
      if (!cs.length) els.push(eqOf('delta', [op(`${lead} =`), res()]));
      else if (cs.length === 1 && cs[0].plain) els.push(eqOf('delta', [op(`${lead} = ${cs[0].tex} =`), res()]));
      else els.push(eqOf('delta', ...cs.map((c, j) => [op(j ? '+' : `${lead} =`), ...c.grids()]), [op('='), res()]));
      if (!idn) els.push(deltaFrom(g, () => dAg(g)));
    }
    return els;
  }

  // ∂L/∂W for each tied block into layer l: Xᵀ δ, which is the sum of one outer product per token.
  function tieGradEls(d, l) {
    const S = d.T[l], els = [], n = S.tokens;
    const lines = S.blocks.map(b => {
      const xs = symOf(d, b.k, b.gs), ds = dSym(d, l, b.gd);
      return `\\frac{\\partial L}{\\partial ${b.name}} = ${wT(xs)}\\,${ds} = \\sum_{t=1}^{${n}} \\big(${xs}_{t,:}\\big)^{\\top} \\big(${ds}\\big)_{t,:}`;
    });
    for (let g = 0; g < S.G; g++) {
      const bt = bTied(d, l, g), ds = dSym(d, l, g);
      lines.push(bt ? `\\frac{\\partial L}{\\partial ${wT(bt.name)}} = \\mathbf 1^{\\top}${ds} = \\sum_{t=1}^{${n}} \\big(${ds}\\big)_{t,:}`
        : `\\frac{\\partial L}{\\partial ${bSym(d, l, g)}} = ${ds}`);
    }
    for (const s of lines) els.push(line(s));
    const names = [...new Set([...S.blocks.map(b => b.name), ...S.bias.filter(b => b?.tied).map(b => b.name)])];
    if (!names.length) return els;
    const list = names.flatMap((nm, j) => [j ? (j === names.length - 1 ? ' and ' : ', ') : '', { t: nm }]).filter(Boolean);
    els.push(note(...list, `${names.length > 1 ? ' are shared parameters' : ' is one shared parameter'}: every token uses the same copy, `
      + `so each gradient is the sum over the ${n} tokens of the per-token terms. Training moves every copy by that one step.`));
    // ∂L/∂bᵀ = 1ᵀ δ: a shared bias sums δ over the tokens (bwd.tie)
    const biasGrads = () => {
      for (let g = 0; g < S.G; g++) {
        const bt = bTied(d, l, g);
        if (!bt) continue;
        const col = f => Array.from({ length: n }, (_, t) => tidx(S, g, t, f));
        const sumDb = (dd, f) => col(f).reduce((s, i) => s + dd.bwd.db[l][i], 0);
        els.push(eqOf('grad', [op(`\\frac{\\partial L}{\\partial ${wT(bt.name)}} =`), ones(1, n, '\\mathbf 1^{\\top}'),
          tvals(d, l, g, (dd, i) => dd.bwd.dZ[l][i], 'g', dSym(d, l, g), i => [`brow:${l}:${i}`])], [op('='), grid(1, S.d, (_, f) => {
          const id = S.ids[tidx(S, g, 0, f)];
          return { f: dd => (bt.extra ? sumDb(dd, f) : dd.bwd.tie?.[bt.ties[f]] ?? sumDb(dd, f)),
            s: 'gt', keys: col(f).flatMap(i => [`b:${S.ids[i]}`, `brow:${l}:${i}`]), hover: { kind: 'bias', id }, click: { kind: 'node', id },
            em: { n: col(f).map(i => S.ids[i]) } };
        }, { cap: `\\partial L / \\partial ${wT(bt.name)}` })]));
      }
    };
    const totalOf = (dd, c) => {
      let s = 0;
      for (const id of c.byTok) s += dd.bwd.edge?.[id] ?? NaN;
      return c.extra ? s : (dd.bwd.tie?.[c.tie] ?? s);
    };
    for (const b of S.blocks) {
      const src = d.T[b.k];
      const cellOf = (fs, fd, f, s, keys, hov, em) => {
        const c = b.cells[fs][fd];
        if (!c) return { mask: true, fixed: '0' };
        return { f: dd => f(dd, c), s, keys: [`tie:${c.tie}`, ...keys(c)], cls: 'nm-tied',
          hover: { kind: 'edge', id: hov(c) }, click: { kind: 'edge', id: hov(c) }, em: { e: em(c).filter(x => x != null) } };
      };
      // token t's outer product: the lens weighs it by token t's own edge
      const perTok = t => grid(src.d, S.d, (fs, fd) => cellOf(fs, fd, (dd, c) => dd.bwd.edge?.[c.byTok[t]], 'g',
        c => [`e:${c.byTok[t]}`, `brow:${l}:${c.rowByTok[t]}`, `tbin:${l}:${t}`], c => c.byTok[t], c => [c.byTok[t]]),
      { cap: tokName(t) ? tokTex(t) : `t = ${t + 1}` });
      const total = grid(src.d, S.d, (fs, fd) => cellOf(fs, fd, totalOf, 'gt', c => c.rows.map(i => `brow:${l}:${i}`), c => c.rep, c => c.byTok),
        { cap: `\\partial L / \\partial ${b.name}` });
      const grps = [[op(`\\frac{\\partial L}{\\partial ${b.name}} =`), srcGrid(d, l, b.k, b.gs, { T: true }),
        tvals(d, l, b.gd, (dd, i) => dd.bwd.dZ[l][i], 'g', dSym(d, l, b.gd), i => [`brow:${l}:${i}`])]];
      if (n <= 4) for (let t = 0; t < n; t++) grps.push([op(t ? '+' : '='), perTok(t)]);
      grps.push([op('='), total]);
      els.push(eqOf('grad', ...grps));
      if (b.cells.some(r => r.some(c => c?.extra))) {
        els.push(note({ t: b.name }, ' is also used by edges outside this layer: the total gradient training applies adds their terms too.'));
        els.push(eqOf('grad', [op(`\\text{total } \\frac{\\partial L}{\\partial ${b.name}} =`), grid(src.d, S.d, (fs, fd) => cellOf(fs, fd,
          (dd, c) => dd.bwd.tie?.[c.tie], 'gt', () => [], c => c.rep, c => c.byTok), { cap: '\\text{all edges}' })]));
      }
    }
    biasGrads();
    return els;
  }

  // ---------------------------------------------------------------- attention: backward
  function attnBwd(d, l) {
    const S = d.T[l], H = S.heads, els = [];
    const sc = S.scaleDef ? '\\tfrac{1}{\\sqrt{d_k}}' : f2(S.scale);
    els.push(line(`\\partial V = A^{\\top}\\partial Z,\\qquad \\partial A = \\partial Z\\,V^{\\top},\\qquad `
      + '\\partial S = A \\odot \\big(\\partial A - \\mathbf r\\,\\mathbf 1^{\\top}\\big),\\ \\ \\mathbf r = \\operatorname{rowsum}(\\partial A \\odot A)'));
    els.push(line(`\\partial Q = ${sc}\\,\\partial S\\,K,\\qquad \\partial K = ${sc}\\,\\partial S^{\\top} Q\\qquad
      (\\partial X \\text{ is short for } \\partial L / \\partial X;\\ \\partial Z = \\delta^{(${l})})`));
    if (!d.bwd.attn?.[l]?.heads?.length || !d.fwd.attn?.[l]?.heads?.length) {
      els.push(h('p', 'nm-note', 'The model returned no attention gradients for this layer.'));
      return els;
    }
    const scN = scaleNum(S);
    for (let hh = 0; hh < H; hh++) {
      const s = H > 1 ? `_{${hh + 1}}` : '', F = dd => dd.fwd.attn[l].heads[hh], B = dd => dd.bwd.attn[l].heads[hh];
      const hb = headBox(l, hh);
      if (H > 1) hb.append(sub(`Head ${hh + 1}`, 'nm-sub nm-head'));
      const bk = t => [`bat:${l}:${t}`];
      const hm = (which, f, sc2, cap, T = false) => headMat(d, l, hh, which, f, sc2, cap, { T, keys: bk });
      const sq = (f, sc2, cap, o = {}) => sqMat(d, l, hh, f, sc2, cap, { keys: bk, ...o });
      const dZ = () => hm('z', (dd, t, c) => B(dd).dZ[t][c], 'g', `\\partial Z${s}`);
      const dA = () => sq((dd, i, j) => B(dd).dA[i][j], 'ga', `\\partial A${s}`);
      const dS = (T = false) => sq((dd, i, j) => B(dd).dS[i][j], 'ga', T ? wT(`\\partial S${s}`) : `\\partial S${s}`, { T, masked: T ? null : '0' });
      const A = (T = false) => sq((dd, i, j) => F(dd).A[i][j], 'one', T ? wT(`A${s}`) : `A${s}`, { T, masked: T ? null : '0', heat: true, thr: true });
      hb.append(eqOf('dV', [op(`\\partial V${s} =`), A(true), dZ()], [op('='), hm('v', (dd, t, c) => B(dd).dV[t][c], 'g', `\\partial V${s}`)]));
      hb.append(eqOf('dA', [op(`\\partial A${s} =`), dZ(), hm('v', (dd, t, c) => F(dd).V[t][c], 'a', wT(`V${s}`), true)], [op('='), dA()]));
      const rVec = grid(S.tokens, 1, i => ({
        f: dd => F(dd).A[i].reduce((acc, a, j) => acc + a * B(dd).dA[i][j], 0), s: 'ga', keys: bk(i), em: { r: [l, i] },
      }), { cap: `\\mathbf r${s}`, rowTok: l });
      hb.append(eqOf('dS', [op(`\\partial S${s} =`), A(), op('\\odot\\Big('), dA(), op('-'), rVec, op('\\mathbf 1^{\\top}\\Big)')], [op('='), dS()]));
      hb.append(eqOf('dQ', [op(`\\partial Q${s} = ${scN}`), dS(), hm('k', (dd, t, c) => F(dd).K[t][c], 'a', `K${s}`)],
        [op('='), hm('q', (dd, t, c) => B(dd).dQ[t][c], 'g', `\\partial Q${s}`)]));
      hb.append(eqOf('dK', [op(`\\partial K${s} = ${scN}`), dS(true), hm('q', (dd, t, c) => F(dd).Q[t][c], 'a', `Q${s}`)],
        [op('='), hm('k', (dd, t, c) => B(dd).dK[t][c], 'g', `\\partial K${s}`)]));
      els.push(hb);
    }
    els.push(h('p', 'nm-note', `∂Q, ∂K and ∂V are what layer ${l - 1} receives: its δ for the Q, K and V groups. `
      + 'The attention layer itself has no parameters to train.'));
    return els;
  }

  // ---------------------------------------------------------------- build
  // How layer l is shown: 'attn', 'attnbad', 'tok' (compact token matrices; the batch view keeps
  // the flat z = W a form) or 'dense'.
  function layerMode(d, l) {
    const mode = d.T?.[l]?.mode || 'dense';
    return mode === 'tok' && d.batch ? 'dense' : mode;
  }

  function tokShapeNote(d, l) {
    const S = d.T[l], n = S.tokens, x = (a, b) => `${a}×${b}`;
    if (S.mode === 'attn') {
      const hs = S.heads > 1 ? ` (each of ${S.heads} heads)` : '';
      return `${x(n, S.dh)} · ${x(S.dh, n)} → ${x(n, n)} → ${x(n, n)} · ${x(n, S.dvh)} → ${x(n, S.dvh)}${hs}`;
    }
    const parts = [];
    for (const b of S.blocks) if (b.gd === 0) parts.push(`${x(n, d.T[b.k].d)} · ${x(d.T[b.k].d, S.d)}`);
    for (const r of S.resid) parts.push(x(n, S.d));
    return `${parts.join(' + ')} → ${x(n, S.d)}${S.G > 1 ? ` (each of ${S.groups.join(', ')})` : ''}`;
  }

  // One line for a folded layer: its local equation.
  function sumTex(d, l) {
    const S = d.T?.[l];
    if (S?.mode === 'attn') return attnLocalTex(d, l);
    if (S?.mode === 'attnbad') return '';
    if (S?.mode === 'tok') return localTex(d, l);
    const parts = d.M[l].terms.filter(t => t.cols.length).map(t => `${wName(l, t.k)}\\,${aName(t.k)}`);
    const sum = [...parts, `b^{(${l})}`].join(' + '), act = actTex(d.net.layers[l].act);
    return `${l === d.L ? '\\hat{\\mathbf y}' : `a^{(${l})}`} = ${act ? `${act}\\!\\left(${sum}\\right)` : sum}`;
  }

  function buildLayer(d, l) {
    const lay = d.net.layers[l], m = d.M[l], mode = layerMode(d, l), S = d.T?.[l];
    const sec = h('section', 'nm-layer');
    reg(sec, [`L:${l}`]);
    const head = h('div', 'nm-lhead');
    const what = mode === 'attn' || mode === 'attnbad'
      ? `attention${S.heads > 1 ? ` · ${S.heads} heads` : ''}${S.causal ? ' · causal' : ''} · ${S.tokens} tokens`
      : `${actLabel(lay.act)}${S?.tok && mode === 'tok' ? ` · ${S.tokens} tokens × ${S.d}${S.groups ? ` · ${S.groups.join(' / ')}` : ''}` : ''}`;
    head.append(h('b', null, `Layer ${l}`), h('span', null, `${lay.name ? lay.name + ' · ' : ''}${what}`),
      h('span', 'nm-shape', !m.rows.length ? '' : (mode === 'tok' || mode === 'attn') ? tokShapeNote(d, l) : mode === 'attnbad' ? '' : shapeNote(d, l)));
    hoverOf.set(head, { kind: 'layer', id: lay.id });
    clickOf.set(head, { kind: 'layer', id: lay.id });
    if (eligible(m)) {
      const b = h('button', 'nm-3d', '→ 3D');
      b.title = `Send W${l} to the 3D tab`;
      b.onclick = e => { e.stopPropagation(); send3d(l); };
      head.append(b);
    }
    sec.append(head);
    // the one-line summary shown while lens.focus folds this layer; a click focuses it instead
    const sum = line(sumTex(d, l) || '\\text{}', 'nm-sum');
    sum.title = 'Focus this layer';
    if (!ctx.audience) sum.onclick = e => { e.stopPropagation(); setLens({ focus: { layer: lay.id } }); };
    sec.append(sum);
    secs[l] = { el: sec, id: lay.id, sum };
    if (!m.rows.length) { sec.append(h('p', 'nm-note', 'This layer has no nodes.')); return sec; }
    if (mode === 'attnbad') {
      sec.append(h('p', 'nm-note', 'Attention needs the layer before it to have groups Q, K, V with the same number of tokens, '
        + 'and the head count must divide the feature widths.'));
      return sec;
    }
    if (mode === 'attn') sec.append(...(d.batch ? attnBatch(d, l) : attnFwd(d, l)));
    else if (mode === 'tok') sec.append(...tokFwd(d, l), expander(d, l));
    else sec.append(fwdEq(d, l));
    const box = kind => {
      const b = h('div', 'nm-step');
      b.hidden = true;
      b.lines = [h('div'), h('div'), h('div'), h('div')];   // empty lines are hidden; more are added as needed
      b.append(...b.lines);
      return b;
    };
    stepBoxes[l] = { fwd: box(), bwd: null };
    sec.append(stepBoxes[l].fwd);
    if (d.bwd) {
      const bw = h('div', 'nm-bwd');
      if (mode === 'attn') bw.append(...tokDelta(d, l), ...attnBwd(d, l));
      else if (mode === 'tok') bw.append(...tokDelta(d, l), ...tieGradEls(d, l));
      else bw.append(...bwdEqs(d, l));
      stepBoxes[l].bwd = box();
      bw.append(stepBoxes[l].bwd);
      sec.append(bw);
    }
    return sec;
  }

  function build(d) {
    keyMap = new Map();
    binds = [];
    stepBoxes = [];
    hoverOf = new WeakMap();
    clickOf = new WeakMap();
    painted = { hi: new Set(), hi2: new Set(), sel: new Set(), an: new Set(), an2: new Set() };
    emEls = []; toks = []; secs = []; headBoxes = []; formulaEls = []; trace = null;
    emOn = true;   // the new DOM has no lens state yet: apply everything once
    // focus.js's tokenNames (a trimmed name or null per token), so the names match the canvas and cards
    names = tokenNames(d.net);
    if (!names.some(Boolean)) names = null;
    eMap = new Map(d.net.edges.map(e => [e.id, e]));
    body.replaceChildren();
    root.classList.toggle('nm-labels', opt.labels);
    if (d.L < 1 || !d.fwd) {
      body.append(h('p', 'nm-note', d.L < 1 ? 'Add a layer after the inputs to see z = W a + b.' : 'The network has no values yet.'));
      return;
    }
    body.append(buildFormula(d));
    trace = buildTrace(d);
    if (trace) body.append(trace.el);
    if (d.col) body.append(buildCollapse(d));
    if (d.batch) {
      body.append(h('p', 'nm-note', `Batch: ${d.batch.X.length} samples (${d.batch.src}) as the columns of X.`
        + (d.bwd ? ' The backward pass below is for the current sample x only.' : '')));
    }
    if (opt.mode === 'bwd' && !d.bwd) body.append(h('p', 'nm-note', 'Backward: set a target on every output node to see backprop.'));
    if (d.bwd) {
      // The formula backward() actually used: an xent fallback (bwd.note) is mse.
      const p = h('p', 'nm-loss');
      const v = h('b');
      binds.push({ el: v, t: dd => model.fmt(dd.bwd.loss, 4).replace(/^-/, '−') });
      const kind = outCase(d), name = kind === 'softmax-xent' ? 'cross-entropy' : kind === 'bce' ? 'binary cross-entropy' : 'MSE';
      p.append(`Loss (${name}) `, tex(h('span'), `L = ${lossTex(kind, d.M[d.L].rows.length, outRows(d))} =`), ' ', v);
      if (d.bwd.note) p.append(h('span', 'nm-note', ` ${d.bwd.note}`));
      body.append(p);
    }
    for (let l = 1; l <= d.L; l++) body.append(buildLayer(d, l));
    for (const s of secs) if (s) s.parts = [...s.el.querySelectorAll(':scope > [data-part], :scope > .nm-headbox > [data-part]')];
  }

  // ---------------------------------------------------------------- update in place
  function update(d) {
    const th = theme();
    for (const b of binds) {
      if (b.t) {
        const s = b.t(d);
        if (b.el._t !== s) b.el.textContent = b.el._t = s;
        continue;
      }
      let v;
      try { v = b.f(d); } catch { v = NaN; }
      const s = v === -Infinity ? '−∞' : GRAD.has(b.s) ? numg(v) : num(v);   // −∞: a masked score in the trace
      if (b.el._t !== s) b.el.textContent = b.el._t = s;
      if (!b.s) continue;
      const max = d.max[b.s] || 1, strong = Number.isFinite(v) && Math.abs(v) / max > STRONG;
      const c = v === -Infinity ? 'transparent' : colorFor(v, max, th);
      if (b.el._c !== c) { b.el._c = c; b.el.style.background = c; }
      if (b.el._s !== strong) { b.el._s = strong; b.el.classList.toggle('nm-strong', strong); }
    }
  }

  // ---------------------------------------------------------------- step-through (state.anim)
  const sizeOf = l => model.nodesIn(store.net, l).length;

  // An attention layer (from the last analysed structure) steps token by token: forward in three
  // phases (scores, softmax, weighted sum), backward once per token. anim.i is the token's first node.
  const attnAt = l => {
    const S = struct?.[l];
    return S?.mode === 'attn' && S.ids.length === sizeOf(l) ? S : null;
  };

  function stepList() {
    const L = store.net.layers.length - 1, list = [];
    for (let l = 1; l <= L; l++) {
      const S = attnAt(l);
      if (S) for (let r = 0; r < S.tokens; r++) for (const phase of PHASES) list.push({ dir: 'fwd', l, i: r * S.d, phase });
      else for (let i = 0; i < sizeOf(l); i++) list.push({ dir: 'fwd', l, i, phase: 'dot' });
    }
    if (store.state.bwd) {
      for (let l = L; l >= 1; l--) {
        const S = attnAt(l);
        if (S) for (let r = 0; r < S.tokens; r++) list.push({ dir: 'bwd', l, i: r * S.d, phase: 'delta' });
        else for (let i = 0; i < sizeOf(l); i++) list.push({ dir: 'bwd', l, i, phase: 'delta' });
      }
    }
    return list;
  }

  function step(dir) {
    const list = stepList();
    if (!list.length) return;
    const a = store.state.anim;
    let at = a ? list.findIndex(s => s.dir === a.dir && s.l === a.l && s.i === a.i && s.phase === a.phase) : -1;
    if (a && at < 0) at = list.findIndex(s => s.dir === a.dir && s.l === a.l && s.i === a.i);
    let next;
    if (at < 0) {
      const bwdFirst = userBwd && opt.mode === 'bwd' && store.state.bwd;
      next = dir > 0 ? (bwdFirst ? list.find(s => s.dir === 'bwd') : list[0]) : list.at(-1);
    } else next = list[at + dir] || null;
    if (next) opt.batch = false;
    if (next?.dir === 'bwd') opt.mode = 'bwd';
    store.set('anim', next);
  }

  function animValid(a) {
    if (!a) return true;
    const L = store.net.layers.length - 1;
    if (!(a.l >= 1 && a.l <= L && a.i >= 0 && a.i < sizeOf(a.l))) return false;
    return a.dir !== 'bwd' || !!store.state.bwd;
  }

  function fwdStepTex(d, l, i) {
    const m = d.M[l], act = d.net.layers[l].act, zi = d.fwd.z[l][i], ai = d.fwd.a[l][i], b = m.b[i];
    const syms = [], nums = [];
    for (const t of m.terms) {
      if (!t.cols.length) continue;
      syms.push(`${wName(l, t.k)}_{${i + 1},:}\\,${aName(t.k)}`);
      t.cols.forEach((_, j) => { if (t.edge[i][j] != null) nums.push(`${par(t.W[i][j])} \\cdot ${par(d.fwd.a[t.k][j])}`); });
    }
    syms.push(`b^{(${l})}_${idx(i)}`);
    nums.push(opt.bias ? `${par(b)} \\cdot 1` : par(b));
    const zs = `z^{(${l})}_${idx(i)}`, as = `a^{(${l})}_${idx(i)}`;
    const l1 = `${zs} = ${syms.join(' + ')} = ${nums.join(' + ')} = ${f2(zi)}`;
    let l2 = actStepTex(act, zi, ai, as, zs, d.fwd.z[l], `z^{(${l})}_j`);
    if (l === d.L) l2 = l2.replace(as, `\\hat y_${idx(i)} = ${as}`);
    return [l1, l2, ''];
  }

  // a = σ(z) with numbers. softZ: the z values the softmax normalises over; zj their symbol.
  function actStepTex(act, zi, ai, as, zs, softZ, zj) {
    switch (act) {
      case 'identity': return `${as} = ${zs} = ${f2(ai)}`;
      case 'relu': return `${as} = \\operatorname{ReLU}(${f2(zi)}) = \\max(0,\\ ${f2(zi)}) = ${f2(ai)}`;
      case 'leaky': {
        const k = model.ACTS.leaky.slope ?? -model.ACTS.leaky.f(-1);
        return zi < 0 ? `${as} = \\operatorname{LReLU}(${f2(zi)}) = ${k} \\cdot ${par(zi)} = ${f2(ai)}`
          : `${as} = \\operatorname{LReLU}(${f2(zi)}) = ${f2(ai)}`;
      }
      case 'sigmoid': return `${as} = \\sigma(${f2(zi)}) = \\frac{1}{1 + e^{${f2(-zi)}}} = ${f2(ai)}`;
      case 'tanh': return `${as} = \\tanh(${f2(zi)}) = ${f2(ai)}`;
      case 'softmax': {
        const den = softZ.length <= 5 ? softZ.map(v => `e^{${f2(v)}}`).join(' + ') : `\\sum_{j=1}^{${softZ.length}} e^{${zj}}`;
        return `${as} = \\frac{e^{${zs}}}{\\sum_j e^{${zj}}} = \\frac{e^{${f2(zi)}}}{${den}} = ${f2(ai)}`;
      }
      default: return `${as} = ${actTex(act)}(${f2(zi)}) = ${f2(ai)}`;
    }
  }

  // ---- token layers: node i is entry (t, f) of group g
  function tokFwdStep(d, l, i) {
    const S = d.T[l], p = tpos(S, i), m = d.M[l], tf = `${p.t + 1},${p.f + 1}`, act = S.act;
    const syms = [], nums = [];
    m.terms.forEach((term, ti) => {
      if (S.resid.some(r => r.ti === ti)) {
        syms.push(`${symOf(d, term.k, p.g)}_{${tf}}`);
        term.edge[i].forEach((e, j) => { if (e != null) nums.push(par(d.fwd.a[term.k][j])); });
        return;
      }
      for (const b of S.blocks) if (b.ti === ti && b.gd === p.g) syms.push(`${symOf(d, b.k, b.gs)}_{${p.t + 1},:}\\,(${b.name})_{:,${p.f + 1}}`);
      term.edge[i].forEach((e, j) => { if (e != null) nums.push(`${par(term.W[i][j])} \\cdot ${par(d.fwd.a[term.k][j])}`); });
    });
    if (d.bflag[l]?.[p.g]) {
      const bt = bTied(d, l, p.g);
      syms.push(bt ? `(${bt.name})_{${p.f + 1}}` : `(${bSym(d, l, p.g)})_{${tf}}`);
      nums.push(par(m.b[i]));
    }
    const as = `${symOf(d, l, p.g)}_{${tf}}`, zs = act === 'identity' ? as : `${preSym(d, l, p.g)}_{${tf}}`;
    const l1 = `${zs} = ${syms.join(' + ') || '0'} = ${nums.join(' + ') || '0'} = ${f2(d.fwd.z[l][i])}`;
    if (act === 'identity') return [l1];
    // softmax on a token layer normalises this token's row of the group
    const row = Array.from({ length: S.d }, (_, f) => d.fwd.z[l][tidx(S, p.g, p.t, f)]);
    return [l1, actStepTex(act, d.fwd.z[l][i], d.fwd.a[l][i], as, zs, row, `${preSym(d, l, p.g)}_{${p.t + 1},j}`)];
  }

  function tokBwdStep(d, l, i) {
    const S = d.T[l], p = tpos(S, i), tf = `${p.t + 1},${p.f + 1}`, act = S.act, n = S.ids.length;
    const dz = d.bwd.dZ[l][i], sp = d.sp[l][i], ds = `(${dSym(d, l, p.g)})_{${tf}}`;
    const as = `${symOf(d, l, p.g)}_{${tf}}`, zs = `${preSym(d, l, p.g)}_{${tf}}`, dAs = `\\frac{\\partial L}{\\partial ${as}}`;
    const side = act === 'relu' || act === 'leaky' ? `\\qquad (${zs} ${d.fwd.z[l][i] > 0 ? '>' : '\\le'} 0)` : '';
    const softTex = () => {   // δ = a (∂L/∂a − Σ_j a_j ∂L/∂a_j) over this token's row
      const js = Array.from({ length: S.d }, (_, f) => tidx(S, p.g, p.t, f));
      const s = js.reduce((acc, j) => acc + d.fwd.a[l][j] * d.bwd.dA[l][j], 0);
      return `${ds} = ${as}\\Big(${dAs} - \\sum_j ${symOf(d, l, p.g)}_{${p.t + 1},j} \\frac{\\partial L}{\\partial ${symOf(d, l, p.g)}_{${p.t + 1},j}}\\Big)
        = ${f2(d.fwd.a[l][i])}\\,\\big(${g2(d.bwd.dA[l][i])} - ${parg(s)}\\big) = ${g2(dz)}`;
    };
    let l0 = '', l1;
    if (l === d.L) {
      const kind = outCase(d), yh = d.fwd.a[l][i], y = d.y[i], fr = n > 1 ? `\\tfrac{1}{${n}}` : '';
      const Yh = `\\hat Y_{${tf}}`, Yt = `Y_{${tf}}`;
      if (kind === 'softmax-xent') {
        const rows = outRows(d), m = rows?.m || 1, fm = m > 1 ? `\\tfrac{1}{${m}}` : '', s = rows ? rows.sums[rows.row(i)] : 1;
        l1 = !rows || rows.proper ? `${ds} = ${fm}(${Yh} - ${Yt}) = ${fm}(${f2(yh)} - ${par(y)}) = ${g2(dz)}`
          : `${ds} = ${fm}\\big((\\textstyle\\sum_j Y_{${p.t + 1},j})\\,${Yh} - ${Yt}\\big) = ${fm}(${f2(s)} \\cdot ${f2(yh)} - ${par(y)}) = ${g2(dz)}`;
      }
      else if (kind === 'bce') l1 = `${ds} = ${fr}(${Yh} - ${Yt}) = ${fr}(${f2(yh)} - ${par(y)}) = ${g2(dz)}`;
      else if (act === 'softmax') { l0 = `${dAs} = ${fr}(${Yh} - ${Yt}) = ${g2(d.bwd.dA[l][i])}`; l1 = softTex(); }
      else {
        const g = act === 'identity' ? '' : `\\,${primeTex(act, zs)}`, gv = act === 'identity' ? '' : ` \\cdot ${spTex(act, sp)}`;
        l1 = `${ds} = ${fr}(${Yh} - ${Yt})${g} = ${fr}(${f2(yh)} - ${par(y)})${gv} = ${g2(dz)}${side}`;
      }
    } else {
      const syms = [], nums = [];
      for (let m = l + 1; m <= d.L; m++) {
        const Sm = d.T[m];
        if (Sm.mode === 'attn' && m === l + 1) {
          const nm = p.g === Sm.gq ? 'Q' : p.g === Sm.gk ? 'K' : p.g === Sm.gv ? 'V' : null;
          if (nm) { syms.push(`(\\partial ${nm})_{${tf}}`); nums.push(parg(attnGradAt(d, m, i))); }
          continue;
        }
        d.M[m].terms.forEach((term, ti) => {
          if (term.k !== l) return;
          const rows = [];
          term.edge.forEach((row, r) => { if (row[i] != null) rows.push(r); });
          if (!rows.length) return;
          if (Sm.mode === 'tok' && Sm.resid.some(r => r.ti === ti)) {
            syms.push(`(${dSym(d, m, p.g)})_{${tf}}`);
            rows.forEach(r => nums.push(parg(d.bwd.dZ[m][r])));
            return;
          }
          const blks = Sm.mode === 'tok' ? Sm.blocks.filter(b => b.ti === ti && b.gs === p.g) : [];
          if (blks.length) blks.forEach(b => syms.push(`\\sum_{f'} (${b.name})_{${p.f + 1},f'}\\,(${dSym(d, m, b.gd)})_{${p.t + 1},f'}`));
          else syms.push(`\\sum_r ${wName(m, l)}_{r,${i + 1}}\\,\\delta^{(${m})}_r`);
          rows.forEach(r => nums.push(`${par(term.W[r][i])} \\cdot ${parg(d.bwd.dZ[m][r])}`));
        });
      }
      const ups = syms.join(' + ') || '0', sum = nums.join(' + ') || '0';
      const bare = nums.length === 1 && !nums[0].includes('\\cdot');   // one plain term: its number is the result
      if (act === 'identity') l1 = `${ds} = ${dAs} = ${ups} = ${bare ? '' : `${sum} = `}${g2(dz)}`;
      else if (act === 'softmax') { l0 = `${dAs} = ${ups} = ${bare ? '' : `${sum} = `}${g2(d.bwd.dA[l][i])}`; l1 = softTex(); }
      else l1 = `${ds} = \\Big(${ups}\\Big)\\,${primeTex(act, zs)} = (${sum}) \\cdot ${spTex(act, sp)} = ${g2(dz)}${side}`;
    }
    const out = [l0, l1];
    for (const b of S.blocks) {
      if (b.gd !== p.g) continue;
      const src = d.T[b.k], x = Array.from({ length: src.d }, (_, f) => d.fwd.a[b.k][tidx(src, b.gs, p.t, f)]);
      out.push(`\\frac{\\partial L}{\\partial (${b.name})_{:,${p.f + 1}}} \\mathrel{+}= ${ds}\\,\\big(${symOf(d, b.k, b.gs)}_{${p.t + 1},:}\\big)^{\\top}
        = ${parg(dz)} \\cdot \\big[${x.map(f2).join(',\\ ')}\\big]^{\\top} = \\big[${x.map(v => g2(dz * v)).join(',\\ ')}\\big]^{\\top}
        \\quad\\text{(token ${p.t + 1}'s term; the shared }${b.name}\\text{ adds all ${S.tokens})}`);
    }
    const bt = bTied(d, l, p.g);
    out.push(bt
      ? `\\frac{\\partial L}{\\partial (${bt.name})_{${p.f + 1}}} \\mathrel{+}= ${ds} = ${g2(d.bwd.db[l][i])}\\quad\\text{(token ${p.t + 1}'s term; the shared bias adds all ${S.tokens})}`
      : `\\frac{\\partial L}{\\partial (${bSym(d, l, p.g)})_{${tf}}} = ${ds} = ${g2(d.bwd.db[l][i])}`);
    return out;
  }

  // ---- attention: token r = floor(anim.i / d), phases scores / softmax / sum; backward once per token
  function attnFwdStep(d, l, a) {
    const S = d.T[l], r = Math.floor(a.i / S.d), n = S.tokens, H = S.heads, R = r + 1, out = [];
    const at = d.fwd.attn?.[l];
    if (!at?.heads) return ['\\text{no attention data}'];
    const sc = scaleNum(S), masked = j => S.causal && j > r;
    const vec = v => `\\big[${v.map(f2).join(',\\ ')}\\big]`;
    for (let hh = 0; hh < H; hh++) {
      const F = at.heads[hh], hs = H > 1 ? `^{(${hh + 1})}` : '';
      if (a.phase === 'softmax') {
        const live = F.S[r].map((s, j) => (masked(j) ? null : s));
        const top = live.map(s => (s == null ? '0' : `e^{${f2(s)}}`)).join(',\\ ');
        const den = live.filter(s => s != null).map(s => `e^{${f2(s)}}`).join(' + ');
        const sum = F.A[r].reduce((acc, v) => acc + v, 0);
        out.push(`A${hs}_{${R},:} = \\operatorname{softmax}\\big(S${hs}_{${R},:}\\big) = \\frac{\\big[${top}\\big]}{${den}} = ${vec(F.A[r])}`
          + `\\qquad \\textstyle\\sum_j A${hs}_{${R},j} = ${f2(sum)}${S.causal && r < n - 1 ? '\\quad (e^{-\\infty} = 0)' : ''}`);
      } else if (a.phase === 'sum') {
        const terms = F.V.map((v, j) => `${f2(F.A[r][j])}\\,${vec(v)}`).join(' + ');
        out.push(`\\mathbf z${hs}_{${R}} = \\sum_j A${hs}_{${R},j}\\,\\mathbf v${hs}_j = ${terms} = ${vec(F.Z[r])}`);
      } else {
        for (let j = 0; j < n; j++) {
          const s = `s${hs}_{${R},${j + 1}}`;
          if (masked(j)) { out.push(`${s} = -\\infty\\quad\\text{(masked: token ${j + 1} comes after token ${R})}`); continue; }
          const dot = F.Q[r].map((q, c) => `${par(q)} \\cdot ${par(F.K[j][c])}`).join(' + ');
          out.push(`${s} = ${sc}\\;\\mathbf q${hs}_{${R}} \\cdot \\mathbf k${hs}_{${j + 1}} = ${sc}\\,\\big(${dot}\\big) = ${f2(F.S[r][j])}`);
        }
      }
    }
    return out;
  }

  function attnBwdStep(d, l, a) {
    const S = d.T[l], r = Math.floor(a.i / S.d), H = S.heads, R = r + 1, out = [];
    const at = d.fwd.attn?.[l], bt = d.bwd.attn?.[l];
    if (!at?.heads || !bt?.heads) return ['\\text{no attention gradients}'];
    const sc = scaleNum(S), vecg = v => `\\big[${v.map(g2).join(',\\ ')}\\big]`;
    for (let hh = 0; hh < H; hh++) {
      const F = at.heads[hh], B = bt.heads[hh], hs = H > 1 ? `^{(${hh + 1})}` : '';
      const c = F.A[r].reduce((acc, v, j) => acc + v * B.dA[r][j], 0);
      out.push(`\\partial\\mathbf z${hs}_{${R}} = ${vecg(B.dZ[r])},\\qquad \\partial A${hs}_{${R},:} = \\partial\\mathbf z${hs}_{${R}}\\,V^{\\top}
        = \\big[\\partial\\mathbf z_{${R}}\\cdot\\mathbf v_j\\big]_j = ${vecg(B.dA[r])}`);
      out.push(`\\partial S${hs}_{${R},:} = A_{${R},:} \\odot \\big(\\partial A_{${R},:} - r_{${R}}\\big),\\ \\ r_{${R}} = \\textstyle\\sum_j A_{${R},j}\\,\\partial A_{${R},j} = ${g2(c)}
        \\ \\Rightarrow\\ \\big[${F.A[r].map((v, j) => `${f2(v)}\\,(${g2(B.dA[r][j])} - ${parg(c)})`).join(',\\ ')}\\big] = ${vecg(B.dS[r])}`);
      out.push(`\\partial\\mathbf q${hs}_{${R}} = ${sc}\\sum_j \\partial S_{${R},j}\\,\\mathbf k_j = ${vecg(B.dQ[r])}`);
      out.push(`\\partial\\mathbf k${hs}_{${R}} = ${sc}\\sum_i \\partial S_{i,${R}}\\,\\mathbf q_i = ${vecg(B.dK[r])},\\qquad
        \\partial\\mathbf v${hs}_{${R}} = \\sum_i A_{i,${R}}\\,\\partial\\mathbf z_i = ${vecg(B.dV[r])}`);
    }
    return out;
  }

  function bwdStepTex(d, l, i) {
    const m = d.M[l], act = d.net.layers[l].act, n = m.rows.length;
    const dz = d.bwd.dZ[l][i], sp = d.sp[l][i], ds = `\\delta^{(${l})}_${idx(i)}`;
    const zs = `z^{(${l})}_${idx(i)}`, as = `a^{(${l})}_${idx(i)}`, dAs = `\\frac{\\partial L}{\\partial ${as}}`;
    // σ' factor: juxtaposed for a named slope, "· 1" for identity; ReLU / leaky name their side of 0
    const times = act === 'identity' ? ' \\cdot ' : '\\,';
    const side = act === 'relu' || act === 'leaky' ? `\\qquad (${zs} ${d.fwd.z[l][i] > 0 ? '>' : '\\le'} 0)` : '';
    // softmax: δ_i = a_i (∂L/∂a_i − Σ_j a_j ∂L/∂a_j)
    const softTex = () => {
      const a = d.fwd.a[l], dA = d.bwd.dA[l], s = a.reduce((acc, v, j) => acc + v * dA[j], 0);
      return `${ds} = ${as}\\Big(${dAs} - \\sum_j a^{(${l})}_j \\frac{\\partial L}{\\partial a^{(${l})}_j}\\Big)
        = ${f2(a[i])}\\,\\big(${g2(dA[i])} - ${parg(s)}\\big) = ${g2(dz)}`;
    };
    let l0 = '', l1;   // l0: dL/da on its own line when a softmax needs it before δ
    if (l === d.L) {
      const kind = outCase(d), yh = d.fwd.a[l][i], y = d.y[i], fr = n > 1 ? `\\tfrac{1}{${n}}` : '';
      const sc = Math.abs(d.sumY - 1) > 1e-9, rows = kind === 'softmax-xent' ? outRows(d) : null;
      if (rows && rows.m > 1) {   // token rows: δ_i = (Σ_row y · ŷ_i − y_i) / m
        const r = rows.row(i), s = rows.sums[r], fm = `\\tfrac{1}{${rows.m}}`;
        l1 = rows.proper ? `${ds} = ${fm}(\\hat y_${idx(i)} - y_${idx(i)}) = ${fm}(${f2(yh)} - ${par(y)}) = ${g2(dz)}`
          : `${ds} = ${fm}\\big((\\textstyle\\sum_{\\text{row}} y)\\,\\hat y_${idx(i)} - y_${idx(i)}\\big) = ${fm}(${f2(s)} \\cdot ${f2(yh)} - ${par(y)}) = ${g2(dz)}`;
      } else if (kind === 'softmax-xent') {
        l1 = sc ? `${ds} = \\big(\\textstyle\\sum_j y_j\\big)\\hat y_${idx(i)} - y_${idx(i)} = ${f2(d.sumY)} \\cdot ${f2(yh)} - ${par(y)} = ${g2(dz)}`
          : `${ds} = \\hat y_${idx(i)} - y_${idx(i)} = ${f2(yh)} - ${par(y)} = ${g2(dz)}`;
      } else if (kind === 'bce') l1 = `${ds} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)}) = ${fr}(${f2(yh)} - ${par(y)}) = ${g2(dz)}`;
      else if (kind === 'softmax-mse') {
        l0 = `${dAs} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)}) = ${g2(d.bwd.dA[l][i])}`;
        l1 = softTex();
      } else {
        const g = act === 'identity' ? '' : `\\,${primeTex(act, zs)}`;
        const gv = act === 'identity' ? '' : ` \\cdot ${spTex(act, sp)}`;
        l1 = `${ds} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)})${g} = ${fr}(${f2(yh)} - ${par(y)})${gv} = ${g2(dz)}${side}`;
      }
    } else {
      const terms = [];
      for (const u of upstream(d, l)) {
        const um = d.M[u.m];
        um.rows.forEach((_, r) => { if (u.t.edge[r][i] != null) terms.push(`${par(u.t.W[r][i])} \\cdot ${parg(d.bwd.dZ[u.m][r])}`); });
      }
      const upSyms = upstream(d, l).map(u => `\\sum_r ${wName(u.m, l)}_{r,${i + 1}}\\,\\delta^{(${u.m})}_r`);
      if (d.T?.[l + 1]?.mode === 'attn') {   // the attention layer above sends back ∂Q, ∂K, ∂V directly
        terms.push(parg(attnGradAt(d, l + 1, i)));
        upSyms.push(`(\\partial [Q\\,K\\,V])_{${i + 1}}`);
      }
      const sum = terms.length ? terms.join(' + ') : '0';
      const ups = upSyms.join(' + ');
      if (act === 'softmax') {
        l0 = `${dAs} = ${ups} = ${sum} = ${g2(d.bwd.dA[l][i])}`;
        l1 = softTex();
      } else {
        l1 = `${ds} = \\Big(${ups}\\Big)${times}${primeTex(act, zs)} = (${sum}) \\cdot ${spTex(act, sp)} = ${g2(dz)}${side}`;
      }
    }
    const grads = m.terms.filter(t => t.cols.length).map(t => {
      const a = d.fwd.a[t.k];
      return `\\frac{\\partial L}{\\partial ${wName(l, t.k)}_{${i + 1},:}} = ${ds}\\,${aT(t.k)} = ${parg(dz)} \\cdot
        \\big[${a.map(f2).join(',\\ ')}\\big] = \\big[${a.map(v => g2(dz * v)).join(',\\ ')}\\big]`;
    });
    const l3 = `\\frac{\\partial L}{\\partial b^{(${l})}_${idx(i)}} = ${ds} = ${g2(d.bwd.db[l][i])}`;
    return [l0, l1, grads.join(',\\qquad '), l3];
  }

  let lastAnim = '';
  function updateSteps(d) {
    const a = store.state.anim;
    stepBoxes.forEach((boxes, l) => {
      if (!boxes) return;
      for (const dir of ['fwd', 'bwd']) {
        const box = boxes[dir];
        if (!box) continue;
        const on = !!a && a.l === l && a.dir === dir && !d.batch;
        box.hidden = !on;
        if (!on) continue;
        let lines;
        const mode = layerMode(d, l);
        try {
          lines = dir === 'fwd'
            ? (mode === 'attn' ? attnFwdStep(d, l, a) : mode === 'tok' ? tokFwdStep(d, l, a.i) : fwdStepTex(d, l, a.i))
            : (mode === 'attn' ? attnBwdStep(d, l, a) : mode === 'tok' ? tokBwdStep(d, l, a.i) : bwdStepTex(d, l, a.i));
        } catch (err) {
          console.error('[nn/matrix] step:', err);
          lines = [];
        }
        while (box.lines.length < lines.length) { const el = h('div'); box.lines.push(el); box.append(el); }
        box.lines.forEach((el, j) => { el.hidden = !lines[j]; if (lines[j]) tex(el, lines[j]); });
      }
    });
    const key = a ? `${a.dir}:${a.l}:${a.i}:${a.phase}` : '';
    if (key !== lastAnim) {
      lastAnim = key;
      if (a) stepBoxes[a.l]?.[a.dir]?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    }
    info.textContent = a ? stepInfo(d, a) : '';
    bStop.disabled = !a;
  }

  function stepInfo(d, a) {
    const dir = a.dir === 'fwd' ? 'forward' : 'backward', mode = layerMode(d, a.l), S = d.T?.[a.l];
    const nm = t => { const s = tokName(t); return s ? ` “${s}”` : ''; };
    if (mode === 'attn') {
      const r = Math.floor(a.i / S.d) + 1;
      return `${dir} · layer ${a.l} · token ${r}/${S.tokens}${nm(r - 1)}${a.dir === 'fwd' ? ` · ${a.phase === 'sum' ? 'weighted sum' : a.phase}` : ''}`;
    }
    if (mode === 'tok') {
      const p = tpos(S, a.i);
      return `${dir} · layer ${a.l} · ${S.groups ? `${S.groups[p.g]} ` : ''}token ${p.t + 1}${nm(p.t)}, feature ${p.f + 1} · row ${a.i + 1}/${sizeOf(a.l)}`;
    }
    return `${dir} · layer ${a.l} · row ${a.i + 1}/${sizeOf(a.l)}`;
  }

  // ---------------------------------------------------------------- highlight (hover, sel, anim)
  let painted = { hi: new Set(), hi2: new Set(), sel: new Set(), an: new Set(), an2: new Set() };

  function paint(cls, keys) {
    const next = new Set();
    for (const k of keys) for (const el of keyMap.get(k) || []) next.add(el);
    for (const el of painted[cls]) if (!next.has(el)) el.classList.remove(`nm-${cls}`);
    for (const el of next) if (!painted[cls].has(el)) el.classList.add(`nm-${cls}`);
    painted[cls] = next;
  }

  const layerIdx = ref => (typeof ref === 'number' ? ref : model.layerIndex(store.net, ref));

  function keysFor(t) {
    if (!t) return [[], []];
    switch (t.kind) {
      case 'edge': {
        const e = model.edge(store.net, t.id);
        if (!hasTie(e)) return [[`e:${t.id}`], e ? [`n:${e.from}`, `n:${e.to}`] : []];
        // a tied edge lights its whole group: every cell of that shared parameter, and its endpoints
        const rel = [];
        for (const o of store.net.edges) if (o.tie === e.tie) rel.push(`n:${o.from}`, `n:${o.to}`);
        return [[`e:${t.id}`, `tie:${e.tie}`], rel];
      }
      case 'pair': return [[`p:${t.from}>${t.to}`], [`n:${t.from}`, `n:${t.to}`]];
      case 'bias': return [[`b:${t.id}`], [`n:${t.id}`]];
      case 'node': return [[`n:${t.id}`, `b:${t.id}`], [`dst:${t.id}`, `src:${t.id}`]];
      case 'layer': return [[`L:${layerIdx(t.id)}`], []];
      case 'row': return [[`row:${layerIdx(t.layer)}:${t.i}`], []];
      case 'col': return [[`col:${layerIdx(t.layer)}:${layerIdx(t.k)}:${t.j}`], []];
      case 'token': {
        // the token's cells (group g, head h when given); on attention, its row of S and A, and
        // for a key / value token of the layer before, its column
        const l = layerIdx(t.layer), S = struct?.[l], tt = t.t;
        if (!S?.d || !Number.isInteger(tt) || tt < 0 || tt >= S.tokens) return [[], []];
        const hh = Number.isInteger(t.h) ? t.h : null, A = struct[l + 1]?.mode === 'attn' ? struct[l + 1] : null;
        const keys = [];
        for (let g = 0; g < S.G; g++) {
          if (Number.isInteger(t.g) && t.g !== g) continue;
          const w = hh === null ? 0 : S.mode === 'attn' ? S.dvh : A ? (g === A.gv ? A.dvh : A.dh) : 0;
          for (let f = 0; f < S.d; f++) if (!w || Math.floor(f / w) === hh) keys.push(`n:${S.ids[tidx(S, g, tt, f)]}`);
        }
        const heads = X => Array.from({ length: X.heads }, (_, k) => k).filter(k => hh === null || k === hh);
        if (S.mode === 'attn') for (const k of heads(S)) keys.push(`ahr:${l}:${k}:${tt}`);
        if (A && t.g === A.gq) for (const k of heads(A)) keys.push(`ahr:${l + 1}:${k}:${tt}`);
        else if (A) for (const k of heads(A)) keys.push(`ahc:${l + 1}:${k}:${tt}`);
        return [keys, []];
      }
      default: return [[], []];
    }
  }

  function paintState() {
    const [hp, hs] = keysFor(store.state.hover);
    paint('hi', hp);
    paint('hi2', hs);
    paint('sel', keysFor(store.state.sel)[0]);
    const a = store.state.anim;
    const [an, an2] = animKeys(a);
    paint('an', an);
    paint('an2', an2);
  }

  function animKeys(a) {
    if (!a) return [[], []];
    const fwd = a.dir === 'fwd', l = a.l, S = struct?.[l];
    if (S?.mode === 'attn' && S.d) {
      const r = Math.floor(a.i / S.d);
      if (!fwd) return [[`bat:${l}:${r}`], []];
      if (a.phase === 'softmax') return [[`aa:${l}:${r}`], [`as:${l}:${r}`]];
      if (a.phase === 'sum') return [[`az:${l}:${r}`], [`aa:${l}:${r}`, `av:${l}`]];
      return [[`aq:${l}:${r}`, `as:${l}:${r}`], [`ak:${l}`]];
    }
    const t = S?.mode === 'tok' && S.d ? tpos(S, a.i).t : null;   // compact inputs light the token's row
    return [fwd ? [`row:${l}:${a.i}`, `out:${l}:${a.i}`] : [`brow:${l}:${a.i}`],
      [fwd ? `in:${l}` : `bin:${l}`, ...(t == null ? [] : [`${fwd ? 'tin' : 'tbin'}:${l}:${t}`])]];
  }

  // ---------------------------------------------------------------- token trace (lens.token)
  // A card at the top: the followed token's row through every token layer (X; Q K V; S, A and Z per
  // head; H; ...; Ŷ), live. Built with the DOM; the token is read per frame (trTok), so following
  // another token never rebuilds. Clicking a line reveals that matrix.
  let trTok = null;
  function buildTrace(d) {
    const T = d.T;
    if (!T || !T.some(S => S.ok && S.tok)) return null;
    const el = h('div', 'nm-trace'), head = h('div', 'nm-tr-head'), title = h('span', 'nm-tr-title');
    el.hidden = true;
    head.append(title, h('span', 'nm-tr-hint', 'its row is outlined in every matrix below'));
    if (!ctx.audience) {
      const x = h('button', 'nm-tr-x', '×');
      x.type = 'button';
      x.title = 'Stop following the token (0)';
      x.onclick = e => { e.stopPropagation(); setLens({ token: null }); };
      head.append(x);
    }
    const box = h('div', 'nm-tr-lines');
    el.append(head, box);
    const labels = [];                                   // { el, f(t) -> tex }
    const has = S => trTok != null && trTok < S.tokens;
    const row = (w, f, s, hover, cls = '') => grid(1, w, (_, j) => ({ f: dd => f(dd, j), s, cls, hover: () => hover(j) }));
    const add = (l, part, pieces, hh = null) => {
      const ln = h('div', 'nm-tr-line');
      pieces.forEach((p, k) => {
        const sym = h('span', k ? 'nm-tr-sym nm-tr-more' : 'nm-tr-sym');
        labels.push({ el: sym, f: p.sym });
        hoverOf.set(sym, () => (trTok == null ? null : tokSpec(l, trTok, p.g ?? null, hh != null && T[l].heads > 1 ? hh : null)));
        ln.append(sym, p.grid);
        if (p.how) { const how = h('span', 'nm-tr-how'); labels.push({ el: how, f: p.how }); ln.append(how); }
      });
      ln.title = 'Show this matrix';
      ln.onclick = () => reveal(d.net.layers[l].id, part, { token: trTok });
      if (hh != null) headBoxes.push({ el: ln, l, hh });
      box.append(ln);
    };
    const node = (S, l, g, f) => () => (has(S) ? { kind: 'node', id: S.ids[tidx(S, g, trTok, f)] } : null);
    for (let l = 0; l <= d.L; l++) {
      const S = T[l];
      if (!S?.ok || !S.tok || S.mode === 'attnbad') continue;
      if (S.mode === 'attn') {
        const n = S.tokens, H = S.heads, heads = dd => dd.fwd.attn?.[l]?.heads;
        for (let hh = 0; hh < H; hh++) {
          const nm = x => (H > 1 ? `(${x}_{${hh + 1}})` : x), tokH = () => (trTok == null ? null : tokSpec(l, trTok, null, H > 1 ? hh : null));
          const q = H > 1 ? `\\mathbf q^{(${hh + 1})}` : '\\mathbf q', K = H > 1 ? `K_{${hh + 1}}` : 'K';
          add(l, 'scores', [{ sym: t => `${nm('S')}_{${t + 1},:}`, how: t => `= ${scoreTex(S, `${q}_{${t + 1}}`, wT(K))}`,
            grid: row(n, (dd, j) => (!has(S) ? NaN : S.causal && j > trTok ? -Infinity : heads(dd)[hh].S[trTok][j]), 's', tokH) }], hh);
          add(l, 'softmax', [{ sym: t => `${nm('A')}_{${t + 1},:}`, how: t => `= \\operatorname{softmax}\\big(${nm('S')}_{${t + 1},:}\\big)`,
            grid: row(n, (dd, j) => (has(S) ? heads(dd)[hh].A[trTok][j] : NaN), 'one', tokH, 'nm-heat') }], hh);
        }
        const A = t => (H > 1 ? `(A_h)_{${t + 1},j}` : `A_{${t + 1},j}`), v = H > 1 ? '\\mathbf v^{(h)}' : '\\mathbf v';
        add(l, 'mix', [{ sym: t => `${symOf(d, l)}_{${t + 1},:}`,
          how: t => `= ${H > 1 ? '\\big[\\,' : ''}\\textstyle\\sum_j ${A(t)}\\,${v}_j${H > 1 ? '\\,\\big]_{h=1..' + H + '}' : ''}`,
          grid: row(S.d, (dd, f) => (has(S) ? dd.fwd.a[l][tidx(S, 0, trTok, f)] : NaN), 'a', f => node(S, l, 0, f)()) }]);
        continue;
      }
      add(l, null, Array.from({ length: S.G }, (_, g) => ({
        sym: t => `${symOf(d, l, g)}_{${t + 1},:}`, g: S.groups ? g : null,
        grid: row(S.d, (dd, f) => (has(S) ? dd.fwd.a[l][tidx(S, g, trTok, f)] : NaN), 'a', f => node(S, l, g, f)()),
      })));
    }
    return { el, title, labels, tok: undefined };
  }

  function syncTrace(d, lens) {
    const t = Number.isInteger(lens?.token) && lens.token >= 0 ? lens.token : null;
    trTok = t;
    if (!trace) return;
    const show = t != null && !!d.T?.some(S => S.ok && S.tok && t < S.tokens);
    if (trace.el.hidden !== !show) trace.el.hidden = !show;
    if (!show || trace.tok === t) return;
    trace.tok = t;
    const nm = tokName(t);
    trace.title.textContent = `Following token ${t + 1}${nm ? ` “${nm}”` : ''}`;
    for (const x of trace.labels) tex(x.el, x.f(t));
  }

  // ---------------------------------------------------------------- lens
  let emOn = true, lastFocus = null, lastD = null;
  const peek = new Set();   // layer ids reveal() opened while lens.focus folds them; cleared when the focus moves

  const focusIndex = (d, lens) => {
    const f = lens?.focus;
    if (!f || f.layer == null) return null;
    const l = typeof f.layer === 'number' ? f.layer : model.layerIndex(d.net, f.layer);
    return l >= 0 && l <= d.L ? l : null;
  };
  const readsFrom = (d, l, k) => d.M[l]?.terms.some(t => t.k === k && t.cols.length) || (k === l - 1 && d.T?.[l]?.mode === 'attn');

  function emphasisOf(d, lens) {
    if (!lens || !d.fwd) return null;
    try { return emphasis(d.net, d.fwd, lens); } catch (err) { console.error('[nn/matrix] emphasis:', err); return null; }
  }

  // How strongly the lens keeps an element (0 = dimmed, 1 = emphasized). thr.w / thr.a: minW / minA
  // are set (their edge type shown), so W cells and A cells the canvas hides for them are dimmed.
  // lens.show alone dims nothing here: it only declutters the canvas.
  function emWeight(E, em, rowsOf, thr) {
    let f = 1;
    try {
      if (em.n) { f = 0; for (const id of em.n) f = Math.max(f, unit(E.node(id))); }
      else if (em.e) {
        f = em.e.length ? 0 : 1;
        let hid = thr.w && !!em.w && em.e.length > 0;
        for (const id of em.e) {
          f = Math.max(f, unit(E.edge(id)));
          if (hid && (eMap.get(id)?.fixed || !E.hidden?.edge?.(id))) hid = false;
        }
        if (hid) f = 0;   // |w| < minW
      } else if (em.p) f = Math.min(unit(E.node(em.p[0])), unit(E.node(em.p[1])));
      if (em.r) { const R = rowsOf(em.r[0]); if (R && !R.has(em.r[1])) f = 0; }
      if (em.a && thr.a && E.hidden?.attn?.(...em.a)) f = 0;   // A_ij < minA
    } catch { f = 1; }
    return f;
  }

  // Everything the lens changes, in place: folds, the focused part, heads, row outlines, dimming.
  function applyLens(d, lens) {
    // any: something is dimmed; hides: show / minW / minA remove something (dimmed here)
    const E = emphasisOf(d, lens), on = !!(E?.any || E?.hides), memo = new Map();
    const setOf = (k, l) => {
      const key = `${k}${l}`;
      if (!memo.has(key)) {
        let r = null;
        try { r = E?.[k]?.(l) ?? null; } catch { r = null; }
        memo.set(key, r instanceof Set ? r : Array.isArray(r) ? new Set(r) : null);
      }
      return memo.get(key);
    };
    const rowsOf = l => (on ? setOf('rows', l) : null), headsOf = l => (on ? setOf('heads', l) : null);
    const cls = (el, c, v) => { const k = `_${c}`; if (el[k] !== v) { el[k] = v; el.classList.toggle(c, v); } };

    // focus: fold every other layer (the step-through's layer and revealed ones stay open)
    const fl = focusIndex(d, lens), part = fl != null && typeof lens.focus.part === 'string' ? lens.focus.part : null;
    const fkey = fl == null ? '' : `${d.net.layers[fl].id}|${part || ''}`;
    const moved = fkey !== lastFocus;
    if (moved) { lastFocus = fkey; peek.clear(); }
    const open = new Set();
    if (fl != null) {
      if (fl === 0) { for (let l = 1; l <= d.L; l++) if (readsFrom(d, l, 0)) open.add(l); } else open.add(fl);
      if (store.state.anim) open.add(store.state.anim.l);
      secs.forEach((s, l) => { if (s && peek.has(s.id)) open.add(l); });
    }
    secs.forEach((s, l) => {
      if (!s) return;
      cls(s.el, 'nm-folded', fl != null && !open.has(l));
      cls(s.el, 'nm-focus', fl != null && (l === fl || (fl === 0 && open.has(l) && readsFrom(d, l, 0))));
      s.parts ||= [];
      const hit = part && l === fl && s.parts.some(p => p.dataset.part === part);
      for (const p of s.parts) {
        cls(p, 'nm-part-on', !!hit && p.dataset.part === part);
        cls(p, 'nm-part-off', !!hit && p.dataset.part !== part);
      }
    });
    for (const x of formulaEls) cls(x.el, 'nm-foc', x.l === fl);

    // head: the other heads' blocks (and trace lines) are hidden
    for (const b of headBoxes) {
      const Hs = headsOf(b.l), hide = !!Hs && !Hs.has(b.hh);
      if (b.el.hidden !== hide) b.el.hidden = hide;
    }
    // token: row t's outline and header in every token matrix
    for (const x of toks) { const R = rowsOf(x.l); cls(x.el, x.cls, !!R && R.has(x.t)); }
    // dimming, weighted like the canvas
    if (on || emOn) {
      const thr = { w: on && +lens.minW > 0 && lens.show?.weights !== false, a: on && +lens.minA > 0 && lens.show?.attention !== false };
      for (const x of emEls) {
        const f = on ? emWeight(E, x.em, rowsOf, thr) : 1;
        const o = on ? Math.round((DIM + (1 - DIM) * f) * 50) / 50 : 1;
        if (x.el._em !== o) {
          x.el._em = o;
          if (o >= 1) x.el.style.removeProperty('--em'); else x.el.style.setProperty('--em', String(o));
        }
        if (x.em.lg) cls(x.el, 'nm-legible', on && f > 0.99 && !!rowsOf(x.em.a[0]));
      }
      emOn = on;
    }
    cls(root, 'nm-lens', on);

    // a new focus scrolls its layer (or part) to the top of the panel
    if (moved && fl != null) {
      const s = fl === 0 ? secs.find((x, l) => x && open.has(l)) : secs[fl];
      const tgt = (part && s?.parts.find(p => p.dataset.part === part && p.offsetParent !== null)) || s?.el;
      scrollTo(tgt);
    }
  }

  const reduceMotion = () => !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  function scrollTo(el, smooth = true) {
    if (!el?.isConnected) return;
    const r = el.getBoundingClientRect(), b = body.getBoundingClientRect();
    if (!r.width && !r.height) return;
    body.scrollTo({ top: Math.max(0, body.scrollTop + r.top - b.top - 6), behavior: smooth && !reduceMotion() ? 'smooth' : 'auto' });
  }

  // For tour.js: bring layer (id or index) into view, opening it if lens.focus folds it, scrolled to
  // its part ('Q' | 'K' | 'V' | 'scores' | 'softmax' | 'mix') when given, and flash it. With
  // token, the token's outlined row in that part is centred instead. Returns false if not shown.
  function reveal(layer, part = null, { token = null, flash = true, smooth = true } = {}) {
    if (queued || !secs.length) render();
    let l = typeof layer === 'number' ? layer : model.layerIndex(store.net, layer);
    if (l === 0 && lastD?.M) l = secs.findIndex((s, k) => s && readsFrom(lastD, k, 0));   // X shows where it is read
    const s = secs[l];
    if (!s) return false;
    if (s.el.classList.contains('nm-folded')) { peek.add(s.id); render(); }
    const tgt = (part && s.parts.find(p => p.dataset.part === part && p.offsetParent !== null)) || s.el;
    // with a token and a part, centre the part's last outlined row of it (the stage's result: S, A or Z)
    const bands = Number.isInteger(token) && tgt !== s.el
      ? toks.filter(x => x.cls === 'on' && x.t === token && x.el._on && tgt.contains(x.el)) : [];
    const br = bands.at(-1)?.el.getBoundingClientRect();
    if (br?.height) {
      const b = body.getBoundingClientRect(), top = body.scrollTop + br.top - b.top - body.clientHeight / 2 + br.height / 2;
      body.scrollTo({ top: Math.max(0, top), behavior: smooth && !reduceMotion() ? 'smooth' : 'auto' });
    } else scrollTo(tgt, smooth);
    if (flash) {
      tgt.classList.remove('nm-flash');
      void tgt.offsetWidth;   // restart the animation
      tgt.classList.add('nm-flash');
      clearTimeout(tgt._flash);
      tgt._flash = setTimeout(() => tgt.classList.remove('nm-flash'), 1400);
    }
    return true;
  }

  // The lens parts layer (id or index) shows, in order: e.g. ['Q', 'K', 'V'] or ['scores', 'softmax', 'mix'].
  function parts(layer) {
    const l = typeof layer === 'number' ? layer : model.layerIndex(store.net, layer);
    return [...new Set((secs[l]?.parts || []).map(p => p.dataset.part))];
  }

  // ---------------------------------------------------------------- render loop
  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; render(); });
  }

  function render() {
    if (ctx.el.root?.hidden) return;   // onShow re-schedules
    if (!animValid(store.state.anim)) { store.set('anim', null); return; }
    if (store.state.anim?.dir === 'bwd' && opt.mode !== 'bwd') opt.mode = 'bwd';
    let d;
    try { d = compute(); } catch (err) { console.error('[nn/matrix] compute:', err); return; }
    const s = signature(d);
    if (s !== sig) {
      sig = s;
      try { build(d); } catch (err) { console.error('[nn/matrix] build:', err); body.replaceChildren(h('p', 'nm-note', `Matrix view failed: ${err.message}`)); }
    }
    lastD = d;
    const lens = store.state.lens ? lensNow() : null;
    syncTrace(d, lens);   // before update(): the trace's cells read the followed token
    update(d);
    try { applyLens(d, lens); } catch (err) { console.error('[nn/matrix] lens:', err); }
    updateSteps(d);
    paintState();
    for (const [k, b] of Object.entries(tg)) b.classList.toggle('on', k === 'mode' ? opt.mode === 'bwd' : opt[k]);
    tg.mode.classList.toggle('nm-dim', !store.state.bwd);
  }

  // render() forces a rebuild and update() is one per-frame in-place update, as during training
  // (debug / test handles). reveal(layer, part, { token, flash, smooth }) and parts(layer) are for
  // tour.js (see reveal above).
  ctx.matrix = {
    step, toggle, opt, reveal, parts,
    render: () => { sig = ''; render(); },
    update: () => {
      const d = compute(), lens = store.state.lens ? lensNow() : null;
      syncTrace(d, lens); update(d); applyLens(d, lens); updateSteps(d);
    },
  };

  // ---------------------------------------------------------------- 3D tab
  const eligible = m => {
    const W = m?.terms[0]?.W, n = W?.length || 0, c = W?.[0]?.length || 0;
    return n >= 1 && c >= 1 && n <= 3 && c <= 3 && n * c > 1;
  };

  function pickLayer() {
    const net = store.net, s = store.state.sel, mats = model.matrices(net);
    let l = null;
    if (s?.kind === 'layer') l = model.layerIndex(net, s.id);
    else if (s?.kind === 'node') l = model.nodeLayerIndex(net, s.id);
    else if (s?.kind === 'edge') l = model.nodeLayerIndex(net, model.edge(net, s.id)?.to);
    const m = mats.find(x => x.l === l);
    if (m && eligible(m)) return l;
    return mats.find(eligible)?.l ?? null;
  }

  function send3d(l) {
    const g = ctx.graph || window.mathboardGraph;
    if (!g) return ctx.toast('The 3D tab is not ready');
    if (l == null) return ctx.toast('No layer has a weight matrix of at most 3×3');
    const net = store.net, fwd = store.state.fwd;
    let W, name, input, inName;
    if (l === 'eff') {
      const c = model.collapse(net);
      if (!c) return ctx.toast('W_eff exists only when every layer is identity');
      [W, name, input, inName] = [c.W, 'Weff', fwd?.a[0], 'a0'];
    } else {
      const m = model.matrices(net).find(x => x.l === l);
      [W, name, input, inName] = [m.terms[0].W, `W${l}`, fwd?.a[l - 1], `a${l - 1}`];
    }
    const n = W.length, c = W[0]?.length || 0;
    const defs = [];
    let visual = null;
    if (n >= 2 && c >= 2) {
      defs.push([name, `${name} = [${W.map(r => `[${r.map(r3).join(', ')}]`).join(', ')}]`]);
      if (input) defs.push([inName, `${inName} = (${input.map(r3).join(', ')})`]);
      visual = n === c ? `transform(${name}, t)` : `map(${name})`;
    } else if (n * c >= 2) {   // one row or one column: show it as a vector
      const v = n === 1 ? W[0] : W.map(r => r[0]), vn = name.replace(/^W/, 'w');
      defs.push([vn, `${vn} = (${v.map(r3).join(', ')})`]);
    } else return ctx.toast('A 1×1 matrix has nothing to show in 3D');
    try {
      const add = (src, o = {}) => {   // reuse a trailing empty row, as the board's "Send to 3D" does
        const last = g.rows.at(-1);
        if (last && !last.src.trim()) { Object.assign(last, o); g.setRowSource(last, src); } else g.addRow(src, { focus: false, ...o });
      };
      const upsert = (nm, src, o) => {
        const row = g.rowByName(nm);
        if (row) { if (row.src !== src) g.setRowSource(row, src); } else add(src, o);
      };
      for (const [nm, src] of defs) upsert(nm, src);
      if (visual) {   // one nn visual at a time: map / transform rows from an earlier send are replaced
        for (const r of [...g.rows]) if (NN_VISUAL.test(r.src.trim()) && r.src.trim() !== visual) g.removeRow(r);
        if (visual.startsWith('transform') && !g.rowByName('t')) upsert('t', 't = 1', { min: 0, max: 1 });
        if (!g.rows.some(r => r.src.trim() === visual)) add(visual);
      }
      g.setView('graph');
      g.toast?.(`Sent ${defs.map(x => x[0]).join(', ')}${visual ? ` and ${visual}` : ''} to the 3D tab`, 3500);
    } catch (err) {
      console.error('[nn/matrix] send to 3D:', err);
      ctx.toast(`Send to 3D failed: ${err.message}`);
    }
  }

  // ---------------------------------------------------------------- input
  const specEl = (el, map) => {
    for (; el && el !== root; el = el.parentElement) if (map.has(el)) return el;
    return null;
  };
  let myHover = null;
  function setHover(spec) {
    const k = spec ? JSON.stringify(spec) : null;
    if (k === myHover) return;
    const had = myHover;
    myHover = k;
    if (!spec && had == null) return;
    store.set('hover', spec);
  }
  if (!ctx.audience) {
    body.addEventListener('mouseover', e => {
      const el = specEl(e.target, hoverOf), v = el ? hoverOf.get(el) : null;
      setHover(typeof v === 'function' ? v() : v);   // functions: the trace, whose token changes
    });
    body.addEventListener('mouseleave', () => setHover(null));
    body.addEventListener('click', e => {
      if (e.target.closest('button')) return;
      const el = specEl(e.target, clickOf);
      if (el) store.set('sel', clickOf.get(el));
    });
    window.addEventListener('keydown', e => {
      if (e.ctrlKey || e.metaKey || e.altKey || !ctx.active(e)) return;
      if (e.code === 'KeyS') { e.preventDefault(); step(e.shiftKey ? -1 : 1); }
      else if (e.code === 'KeyB' && !e.shiftKey) { e.preventDefault(); toggle('bias'); }
    });
  }

  store.on('net', schedule);
  store.on('values', schedule);
  store.on('anim', () => render());   // user-paced: answer the key press in the same frame
  store.on('lens', () => render());
  store.on('hover', t => {
    if ((t ? JSON.stringify(t) : null) !== myHover) myHover = undefined;
    paintState();
  });
  store.on('sel', paintState);
  ctx.onTheme?.(() => { for (const b of binds) b.el._c = undefined; schedule(); });
  ctx.onShow?.(v => { if (v) schedule(); });
  schedule();
}
