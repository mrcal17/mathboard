// Net tab: the live network as matrix multiplication, in #nn-matrix. See docs/NN_CONTRACT.md.
//   Each layer is z = W a (+ skip terms) + b, a = σ(z), with the actual numbers. Numbers are HTML
//   grids updated in place (training ticks at 60 fps); KaTeX renders symbols, only on rebuild.
//   Hover and click link both ways through store.hover / store.sel.
//   Step-through (S / Shift+S, buttons) owns state.anim: the forward pass row by row, then the
//   backward pass (δ, ∂L/∂W, ∂L/∂b) when every output has a target.
//   Toggles: Backward, bias trick [W | b] (B), batch Z = W X + b 1ᵀ, collapse, labels.
//   Layers whose W is at most 3×3 can be sent to the 3D tab (map / transform).
import { colorFor } from './store.js';

const BATCH = 4;         // samples shown as columns in the batch view
const STRONG = 0.55;     // |v| / max above which a tinted cell switches to contrasting text
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

export function install(ctx) {
  const { store, model } = ctx;
  const host = ctx.el?.matrix;
  if (!host) return;

  const opt = { mode: 'fwd', bias: false, batch: false, collapse: false, labels: true };
  const theme = () => (ctx.theme ? ctx.theme() : document.documentElement.dataset.theme) || 'dark';
  const f2 = x => (Number.isFinite(x) ? model.fmt(x, 2) : '\\text{--}');                  // TeX
  const num = x => (Number.isFinite(x) ? model.fmt(x, 2).replace(/^-/, '−') : '·'); // HTML
  const par = x => { const s = f2(x); return s.startsWith('-') ? `(${s})` : s; };
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
    const d = { net, L, M: [], fwd: store.state.fwd, bwd: null, max: { w: 1, b: 1, a: 1, g: 1, one: 1, eff: 1 } };
    if (L < 1 || !d.fwd) return d;
    for (const m of model.matrices(net)) d.M[m.l] = m;
    d.max.w = maxAbs(net.edges.map(e => e.w)) || 1;     // same scales as the inspector's sliders
    d.max.b = maxAbs(net.nodes.map(n => n.bias)) || 1;
    d.max.a = maxAbs(d.fwd.a, d.fwd.z) || 1;
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
    ].join('|');
  }

  // ---------------------------------------------------------------- DOM registry
  let keyMap = new Map(), binds = [], stepBoxes = [], sig = '';
  let hoverOf = new WeakMap(), clickOf = new WeakMap();

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
  // keys, hover, click }. o: { cap, rowHdr, colHdr: [{ tex, hover, click }], aug: column index with a
  // bar before it, augRow: row index with a bar above it }.
  function grid(n, m, cell, o = {}) {
    const blk = h('div', 'nm-blk');
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
        if (sp.fixed != null) c.textContent = sp.fixed;
        else binds.push({ el: c, f: sp.f, s: sp.mask ? null : sp.s });
        reg(c, sp.keys || []);
        if (sp.hover) hoverOf.set(c, sp.hover);
        if (sp.click) clickOf.set(c, sp.click);
        blk.append(c);
      }
    }
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
  }), { cap, ...extra });

  const nodeHdr = (ids, extra) => ids.map((id, j) => ({ tex: label(id), ...extra(id, j) }));

  // W (or its transpose) of term t in layer l; aug appends b as a last column.
  function wGrid(d, l, ti, { aug = false, T = false } = {}) {
    const m = d.M[l], t = m.terms[ti], n = m.rows.length, c = t.cols.length;
    // transposed (inside δ^{(k)}): its row j is node j of layer k, the row being stepped backward
    const extraKeys = T ? (i, j) => [`brow:${t.k}:${j}`] : (i, j) => [`row:${l}:${i}`, `col:${l}:${t.k}:${j}`];
    const cell = (i, j) => {
      const to = m.rows[i];
      if (j === c) {
        return { f: dd => dd.M[l].b[i], s: 'b', keys: [`b:${to}`, `row:${l}:${i}`, `dst:${to}`],
          hover: { kind: 'bias', id: to }, click: { kind: 'node', id: to } };
      }
      const from = t.cols[j], eid = t.edge[i][j];
      const keys = [...extraKeys(i, j), `dst:${to}`, `src:${from}`];
      if (eid == null) return { mask: true, fixed: '0', keys: [...keys, `p:${from}>${to}`], hover: { kind: 'pair', from, to } };
      return { f: dd => dd.M[l].terms[ti].W[i][j], s: 'w', keys: [...keys, `e:${eid}`],
        hover: { kind: 'edge', id: eid }, click: { kind: 'edge', id: eid } };
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
        hover: { kind: 'node', id: t.cols[i] }, click: { kind: 'node', id: t.cols[i] } }), {
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
        hover: { kind: 'node', id: t.cols[i] }, click: { kind: 'node', id: t.cols[i] } }), {
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
        hover: { kind: 'node', id: m.rows[i] }, click: { kind: 'node', id: m.rows[i] },
      }), { cap: which === 'z' ? `Z^{(${l})}` : isOut ? '\\hat Y' : `A^{(${l})}` });
    }
    return nodeVec(m.rows, (dd, i) => dd.fwd[which][l][i], 'a', i => [`out:${l}:${i}`], cap);
  }

  const biasVec = (d, l) => {
    const m = d.M[l];
    return grid(m.rows.length, 1, i => ({
      f: dd => dd.M[l].b[i], s: 'b', keys: [`b:${m.rows[i]}`, `row:${l}:${i}`, `dst:${m.rows[i]}`],
      hover: { kind: 'bias', id: m.rows[i] }, click: { kind: 'node', id: m.rows[i] },
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

  // Right-hand side of the loss over n outputs, as model.js computes it (ln = natural log).
  function lossTex(kind, n) {
    const one = n === 1, s = one ? '1' : 'i', yh = `\\hat y_{${s}}`, y = `y_{${s}}`;
    const sum = one ? '' : `\\sum_{i=1}^{${n}}`;
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
      if (kind === 'softmax-xent' || kind === 'bce') {
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
      const sum = ups.map(u => `\\big(${wName(u.m, l)}\\big)^{\\!\\top}\\delta^{(${u.m})}`).join(' + ');
      const soft = act === 'softmax';
      els.push(line(soft
        ? `\\delta^{(${l})} = J_{\\operatorname{softmax}}^{\\top}\\big(${sum}\\big)`
        : `\\delta^{(${l})} = \\big(${sum}\\big) \\odot ${primeVec(act, `z^{(${l})}`)}`));
      const groups = ups.map((u, j) => [
        op(j ? '+' : `\\delta^{(${l})} = ${soft ? 'J^{\\top}' : ''}\\Big(`),
        wGrid(d, u.m, u.ti, { T: true }),
        nodeVec(d.M[u.m].rows, (dd, i) => dd.bwd.dZ[u.m][i], 'g', () => [`bin:${l}`], `\\delta^{(${u.m})}`),
        j === ups.length - 1 && op('\\Big)'),
      ]);
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
          hover: { kind: 'node', id: t.cols[j] }, click: { kind: 'node', id: t.cols[j] } }), {
        aug: aug ? c : null, cap: aug ? `\\big[${aT(t.k)}\\ 1\\big]` : aT(t.k),
      });
      const dW = grid(n, c + (aug ? 1 : 0), (i, j) => {
        const to = ids[i];
        if (j === c) {
          return { f: dd => dd.bwd.db[l][i], s: 'g', keys: [`b:${to}`, `brow:${l}:${i}`],
            hover: { kind: 'bias', id: to }, click: { kind: 'node', id: to } };
        }
        const from = t.cols[j], eid = t.edge[i][j];
        const keys = [`brow:${l}:${i}`, `dst:${to}`, `src:${from}`, eid == null ? `p:${from}>${to}` : `e:${eid}`];
        return { f: dd => dd.bwd.dZ[l][i] * dd.fwd.a[t.k][j], s: 'g', faint: eid == null, keys,
          hover: eid == null ? { kind: 'pair', from, to } : { kind: 'edge', id: eid }, click: eid == null ? null : { kind: 'edge', id: eid } };
      }, { aug: aug ? c : null, cap: aug ? `\\partial L / \\partial [${wName(l, t.k)} \\mid b]` : `\\partial L / \\partial ${wName(l, t.k)}` });
      const lead = aug ? `\\frac{\\partial L}{\\partial [${wName(l, t.k)} \\mid b]} =` : `\\frac{\\partial L}{\\partial ${wName(l, t.k)}} =`;
      els.push(eqOf('grad', [op(lead), dVec(`\\delta^{(${l})}`), aRow], [op('='), dW]));
    }
    // δ aᵀ is the full outer product; a masked entry's value is shown faint because nothing uses it
    // (model.backward reports 0 there and training leaves the fixed 0 alone).
    if (terms.some(({ t }) => t.edge.some(r => r.includes(null)))) {
      els.push(h('p', 'nm-note', 'Faint entries have no edge: that weight is a fixed 0, so training never applies them.'));
    }
    if (!augB) {
      els.push(eqOf('db', [op(`\\frac{\\partial L}{\\partial b^{(${l})}} =`), grid(n, 1, i => ({
        f: dd => dd.bwd.db[l][i], s: 'g', keys: [`b:${ids[i]}`, `brow:${l}:${i}`],
        hover: { kind: 'bias', id: ids[i] }, click: { kind: 'node', id: ids[i] },
      }), { cap: `\\partial L / \\partial b^{(${l})}` })]));
    }
    return els;
  }

  // ---------------------------------------------------------------- composed formula + collapse
  function composed(d) {
    const { L, M } = d, uses = new Array(L + 1).fill(0);
    for (let l = 1; l <= L; l++) for (const t of M[l].terms) uses[t.k]++;
    const named = l => l > 0 && l < L && uses[l] > 1;
    const expr = (l, top) => {
      if (l === 0) return { s: '\\mathbf x', atom: true };
      if (!top && named(l)) return { s: `a^{(${l})}`, atom: true };
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

  function buildFormula(d) {
    const box = h('div', 'nm-formula');
    for (const s of composed(d)) box.append(line(s, 'nm-fl'));
    for (const el of box.querySelectorAll('[data-l]')) {
      const l = +el.dataset.l, id = d.net.layers[l]?.id;
      reg(el, [`L:${l}`]);
      hoverOf.set(el, { kind: 'layer', id });
      clickOf.set(el, { kind: 'layer', id });
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

  // ---------------------------------------------------------------- build
  function buildLayer(d, l) {
    const lay = d.net.layers[l], m = d.M[l];
    const sec = h('section', 'nm-layer');
    reg(sec, [`L:${l}`]);
    const head = h('div', 'nm-lhead');
    head.append(h('b', null, `Layer ${l}`), h('span', null, `${lay.name ? lay.name + ' · ' : ''}${actLabel(lay.act)}`),
      h('span', 'nm-shape', m.rows.length ? shapeNote(d, l) : ''));
    hoverOf.set(head, { kind: 'layer', id: lay.id });
    clickOf.set(head, { kind: 'layer', id: lay.id });
    if (eligible(m)) {
      const b = h('button', 'nm-3d', '→ 3D');
      b.title = `Send W${l} to the 3D tab`;
      b.onclick = e => { e.stopPropagation(); send3d(l); };
      head.append(b);
    }
    sec.append(head);
    if (!m.rows.length) { sec.append(h('p', 'nm-note', 'This layer has no nodes.')); return sec; }
    sec.append(fwdEq(d, l));
    const box = kind => {
      const b = h('div', 'nm-step');
      b.hidden = true;
      b.lines = [h('div'), h('div'), h('div'), h('div')];   // empty lines are hidden
      b.append(...b.lines);
      return b;
    };
    stepBoxes[l] = { fwd: box(), bwd: null };
    sec.append(stepBoxes[l].fwd);
    if (d.bwd) {
      const bw = h('div', 'nm-bwd');
      bw.append(...bwdEqs(d, l));
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
    body.replaceChildren();
    root.classList.toggle('nm-labels', opt.labels);
    if (d.L < 1 || !d.fwd) {
      body.append(h('p', 'nm-note', d.L < 1 ? 'Add a layer after the inputs to see z = W a + b.' : 'The network has no values yet.'));
      return;
    }
    body.append(buildFormula(d));
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
      p.append(`Loss (${name}) `, tex(h('span'), `L = ${lossTex(kind, d.M[d.L].rows.length)} =`), ' ', v);
      if (d.bwd.note) p.append(h('span', 'nm-note', ` ${d.bwd.note}`));
      body.append(p);
    }
    for (let l = 1; l <= d.L; l++) body.append(buildLayer(d, l));
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
      const s = num(v);
      if (b.el._t !== s) b.el.textContent = b.el._t = s;
      if (!b.s) continue;
      const max = d.max[b.s] || 1, strong = Math.abs(v) / max > STRONG;
      const c = colorFor(v, max, th);
      if (b.el._c !== c) { b.el._c = c; b.el.style.background = c; }
      if (b.el._s !== strong) { b.el._s = strong; b.el.classList.toggle('nm-strong', strong); }
    }
  }

  // ---------------------------------------------------------------- step-through (state.anim)
  const sizeOf = l => model.nodesIn(store.net, l).length;

  function stepList() {
    const L = store.net.layers.length - 1, list = [];
    for (let l = 1; l <= L; l++) for (let i = 0; i < sizeOf(l); i++) list.push({ dir: 'fwd', l, i, phase: 'dot' });
    if (store.state.bwd) for (let l = L; l >= 1; l--) for (let i = 0; i < sizeOf(l); i++) list.push({ dir: 'bwd', l, i, phase: 'delta' });
    return list;
  }

  function step(dir) {
    const list = stepList();
    if (!list.length) return;
    const a = store.state.anim;
    const at = a ? list.findIndex(s => s.dir === a.dir && s.l === a.l && s.i === a.i) : -1;
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
    let l2;
    switch (act) {
      case 'identity': l2 = `${as} = ${zs} = ${f2(ai)}`; break;
      case 'relu': l2 = `${as} = \\operatorname{ReLU}(${f2(zi)}) = \\max(0,\\ ${f2(zi)}) = ${f2(ai)}`; break;
      case 'leaky': {
        const k = model.ACTS.leaky.slope ?? -model.ACTS.leaky.f(-1);
        l2 = zi < 0 ? `${as} = \\operatorname{LReLU}(${f2(zi)}) = ${k} \\cdot ${par(zi)} = ${f2(ai)}`
          : `${as} = \\operatorname{LReLU}(${f2(zi)}) = ${f2(ai)}`;
        break;
      }
      case 'sigmoid': l2 = `${as} = \\sigma(${f2(zi)}) = \\frac{1}{1 + e^{${f2(-zi)}}} = ${f2(ai)}`; break;
      case 'tanh': l2 = `${as} = \\tanh(${f2(zi)}) = ${f2(ai)}`; break;
      case 'softmax': {
        const z = d.fwd.z[l], zj = `z^{(${l})}_j`;
        const den = z.length <= 5 ? z.map(v => `e^{${f2(v)}}`).join(' + ') : `\\sum_{j=1}^{${z.length}} e^{${zj}}`;
        l2 = `${as} = \\frac{e^{z^{(${l})}_${idx(i)}}}{\\sum_j e^{${zj}}} = \\frac{e^{${f2(zi)}}}{${den}} = ${f2(ai)}`;
        break;
      }
      default: l2 = `${as} = ${actTex(act)}(${f2(zi)}) = ${f2(ai)}`;
    }
    if (l === d.L) l2 = l2.replace(as, `\\hat y_${idx(i)} = ${as}`);
    return [l1, l2, ''];
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
        = ${f2(a[i])}\\,\\big(${f2(dA[i])} - ${par(s)}\\big) = ${f2(dz)}`;
    };
    let l0 = '', l1;   // l0: dL/da on its own line when a softmax needs it before δ
    if (l === d.L) {
      const kind = outCase(d), yh = d.fwd.a[l][i], y = d.y[i], fr = n > 1 ? `\\tfrac{1}{${n}}` : '';
      const sc = Math.abs(d.sumY - 1) > 1e-9;
      if (kind === 'softmax-xent') {
        l1 = sc ? `${ds} = \\big(\\textstyle\\sum_j y_j\\big)\\hat y_${idx(i)} - y_${idx(i)} = ${f2(d.sumY)} \\cdot ${f2(yh)} - ${par(y)} = ${f2(dz)}`
          : `${ds} = \\hat y_${idx(i)} - y_${idx(i)} = ${f2(yh)} - ${par(y)} = ${f2(dz)}`;
      } else if (kind === 'bce') l1 = `${ds} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)}) = ${fr}(${f2(yh)} - ${par(y)}) = ${f2(dz)}`;
      else if (kind === 'softmax-mse') {
        l0 = `${dAs} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)}) = ${f2(d.bwd.dA[l][i])}`;
        l1 = softTex();
      } else {
        const g = act === 'identity' ? '' : `\\,${primeTex(act, zs)}`;
        const gv = act === 'identity' ? '' : ` \\cdot ${spTex(act, sp)}`;
        l1 = `${ds} = ${fr}(\\hat y_${idx(i)} - y_${idx(i)})${g} = ${fr}(${f2(yh)} - ${par(y)})${gv} = ${f2(dz)}${side}`;
      }
    } else {
      const terms = [];
      for (const u of upstream(d, l)) {
        const um = d.M[u.m];
        um.rows.forEach((_, r) => { if (u.t.edge[r][i] != null) terms.push(`${par(u.t.W[r][i])} \\cdot ${par(d.bwd.dZ[u.m][r])}`); });
      }
      const sum = terms.length ? terms.join(' + ') : '0';
      const ups = upstream(d, l).map(u => `\\sum_r ${wName(u.m, l)}_{r,${i + 1}}\\,\\delta^{(${u.m})}_r`).join(' + ');
      if (act === 'softmax') {
        l0 = `${dAs} = ${ups} = ${sum} = ${f2(d.bwd.dA[l][i])}`;
        l1 = softTex();
      } else {
        l1 = `${ds} = \\Big(${ups}\\Big)${times}${primeTex(act, zs)} = (${sum}) \\cdot ${spTex(act, sp)} = ${f2(dz)}${side}`;
      }
    }
    const grads = m.terms.filter(t => t.cols.length).map(t => {
      const a = d.fwd.a[t.k];
      return `\\frac{\\partial L}{\\partial ${wName(l, t.k)}_{${i + 1},:}} = ${ds}\\,${aT(t.k)} = ${par(dz)} \\cdot
        \\big[${a.map(f2).join(',\\ ')}\\big] = \\big[${a.map(v => f2(dz * v)).join(',\\ ')}\\big]`;
    });
    const l3 = `\\frac{\\partial L}{\\partial b^{(${l})}_${idx(i)}} = ${ds} = ${f2(d.bwd.db[l][i])}`;
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
        try { lines = dir === 'fwd' ? fwdStepTex(d, l, a.i) : bwdStepTex(d, l, a.i); } catch (err) {
          console.error('[nn/matrix] step:', err);
          lines = [];
        }
        box.lines.forEach((el, j) => { el.hidden = !lines[j]; if (lines[j]) tex(el, lines[j]); });
      }
    });
    const key = a ? `${a.dir}:${a.l}:${a.i}` : '';
    if (key !== lastAnim) {
      lastAnim = key;
      if (a) stepBoxes[a.l]?.[a.dir]?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    }
    info.textContent = a ? `${a.dir === 'fwd' ? 'forward' : 'backward'} · layer ${a.l} · row ${a.i + 1}/${sizeOf(a.l)}` : '';
    bStop.disabled = !a;
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
        return [[`e:${t.id}`], e ? [`n:${e.from}`, `n:${e.to}`] : []];
      }
      case 'pair': return [[`p:${t.from}>${t.to}`], [`n:${t.from}`, `n:${t.to}`]];
      case 'bias': return [[`b:${t.id}`], [`n:${t.id}`]];
      case 'node': return [[`n:${t.id}`, `b:${t.id}`], [`dst:${t.id}`, `src:${t.id}`]];
      case 'layer': return [[`L:${layerIdx(t.id)}`], []];
      case 'row': return [[`row:${layerIdx(t.layer)}:${t.i}`], []];
      case 'col': return [[`col:${layerIdx(t.layer)}:${layerIdx(t.k)}:${t.j}`], []];
      default: return [[], []];
    }
  }

  function paintState() {
    const [hp, hs] = keysFor(store.state.hover);
    paint('hi', hp);
    paint('hi2', hs);
    paint('sel', keysFor(store.state.sel)[0]);
    const a = store.state.anim;
    const fwd = a?.dir === 'fwd';
    paint('an', !a ? [] : fwd ? [`row:${a.l}:${a.i}`, `out:${a.l}:${a.i}`] : [`brow:${a.l}:${a.i}`]);
    paint('an2', !a ? [] : [fwd ? `in:${a.l}` : `bin:${a.l}`]);
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
    update(d);
    updateSteps(d);
    paintState();
    for (const [k, b] of Object.entries(tg)) b.classList.toggle('on', k === 'mode' ? opt.mode === 'bwd' : opt[k]);
    tg.mode.classList.toggle('nm-dim', !store.state.bwd);
  }

  // Debug / test handle (not part of the contract): render() forces a rebuild, update() is one
  // per-frame in-place update, as during training.
  ctx.matrix = { step, toggle, opt, render: () => { sig = ''; render(); }, update: () => { const d = compute(); update(d); updateSteps(d); } };

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
      const el = specEl(e.target, hoverOf);
      setHover(el ? hoverOf.get(el) : null);
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
  store.on('hover', t => {
    if ((t ? JSON.stringify(t) : null) !== myHover) myHover = undefined;
    paintState();
  });
  store.on('sel', paintState);
  ctx.onTheme?.(() => { for (const b of binds) b.el._c = undefined; schedule(); });
  ctx.onShow?.(v => { if (v) schedule(); });
  schedule();
}
