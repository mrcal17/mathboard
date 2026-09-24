// Net tab inspector: a floating card for the selected node, edge or layer (state.sel).
// See docs/NN_CONTRACT.md.
//
// One card follows the selection. Pinning it keeps it open on its target, and the next selection
// opens another card. A card sits next to its target (ctx.view.nodeRect) until it is dragged by
// its header; double-clicking the header puts it back. Card DOM is rebuilt only when the
// selection or the structure changes. Everything else (training ticks at 60 fps, undo, edits made
// in the view or the matrix) updates text and attributes in place through cached setters. While
// the Train panel runs, the KaTeX boxes refresh at 10 Hz (sliders and bars every frame).
//
// Token layers, shared (tied) and fixed weights and attention layers (docs/NN_ATTENTION.md) get
// their own cards: an attention neuron shows its scores q_i . k_j, its row of A as bars and
// z = sum_j A_ij V_jf; a tied edge is a shared parameter whose slider moves its whole group and
// whose gradient is the sum over its edges; a fixed edge is read-only; an attention layer has
// tokens, heads, the causal mask, the scale and this sample's A per head.

import { colorFor } from './store.js';
import { emphasis, tokenNames } from './focus.js';

const CARD_W = 322;
const GAP = 16;        // px between a card and its target
const EDGE = 8;        // px kept clear of the stage border
const SPAN = 3;        // default slider half-range; grows to fit larger values
const MAX_TERMS = 9;   // longer sums are elided with \cdots
const FN = {
  identity: '', relu: '\\operatorname{ReLU}', leaky: '\\operatorname{LReLU}',
  sigmoid: '\\sigma', tanh: '\\tanh', softmax: '\\operatorname{softmax}',
};
const DEF = {
  identity: 'a = z', relu: 'a = \\max(0,\\, z)', sigmoid: 'a = \\dfrac{1}{1 + e^{-z}}',
  tanh: 'a = \\tanh z', softmax: 'a_i = \\dfrac{e^{z_i}}{\\sum_j e^{z_j}}',
};
const SCHEMES = ['xavier', 'he', 'small'];
const STOP = ['pointerdown', 'mousedown', 'click', 'dblclick', 'contextmenu', 'wheel', 'touchstart'];
const NUM_RE = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i;

// ---------------------------------------------------------------- DOM helpers

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (typeof v === 'function') el.addEventListener(k, v);
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c);
  return el;
}

function sv(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function setAttr(el, k, v) {
  const key = '_a_' + k;
  if (el[key] !== v) { el[key] = v; el.setAttribute(k, v); }
}
function setText(el, s) { if (el._text !== s) { el._text = s; el.textContent = s; } }
function setStyle(el, k, v) { if (el.style[k] !== v) el.style[k] = v; }
function setHidden(el, on) { if (el.hidden !== !!on) el.hidden = !!on; }
function setVal(el, v) { if (document.activeElement !== el && el.value !== v) el.value = v; }

function setTex(el, src, display = false) {
  if (el._tex === src) return;
  el._tex = src;
  const k = globalThis.katex;
  if (!k) { el.textContent = src; return; }
  try { k.render(src, el, { throwOnError: false, displayMode: display, strict: 'ignore' }); }
  catch { el.textContent = src; }
}

function texString(src) {
  const k = globalThis.katex;
  if (!k) return null;
  try { return k.renderToString(src, { throwOnError: false, strict: 'ignore' }); } catch { return null; }
}

// Rough rendered width of a TeX snippet in digit widths: sub/superscripts count 0.7, a command
// (\cdot, \delta, ...) 1, thin spaces and braces nothing; a binary + or - adds its spacing.
function texWidth(s) {
  let t = String(s).replace(/\\textcolor\{[^}]*\}/g, '').replace(/\\[,;!]|\\quad|&/g, ' ');
  let w = 0;
  t = t.replace(/[\^_](\{(?:[^{}]|\{[^{}]*\})*\}|\\[a-zA-Z]+|.)/g, m => {
    w += 0.7 * texWidth(m.slice(1).replace(/^\{|\}$/g, ''));
    return ' ';
  });
  t = t.replace(/\\[a-zA-Z]+/g, () => { w += 1; return ' '; });
  for (const ch of t) if (!/[\s{}]/.test(ch)) w += /[+\-=−]/.test(ch) ? 1.6 : 1;
  return w;
}

// A card's KaTeX line holds about this many texWidth units (measured: 296 px at ~7 px a unit).
const CARD_UNITS = 40;

// Terms of a sum for an aligned block, packed greedily into right-hand sides of at most `budget`
// texWidth units (CARD_UNITS minus the widest left-hand side of the block). The first term keeps
// its own sign; continuation lines are indented by a quad.
function wrapSum(lead, parts, budget) {
  const signed = parts.map((s, q) => (q && !s.startsWith('-') ? `+ ${s}` : s));
  const lines = [];
  let cur = [], used = texWidth(lead.slice(lead.lastIndexOf('&') + 1));
  for (const t of signed) {
    const tw = texWidth(t);
    if (cur.length && used + tw > budget) { lines.push(cur); cur = []; used = 2.5; }
    cur.push(t);
    used += tw;
  }
  if (cur.length) lines.push(cur);
  return lines.map((c, q) => (q ? '&\\quad {} ' : lead) + c.join(' '));
}

const numText = x => (Number.isFinite(x) ? String(+x.toFixed(3)) : '');
function parseNum(s) {
  const t = String(s ?? '').trim().replace(/−/g, '-');
  return NUM_RE.test(t) ? +t : null;
}
const fitSpan = v => (Math.abs(v) <= SPAN ? SPAN : Math.ceil(Math.abs(v) * 1.25));
const TEXT_ESC = { '\\': '\\textbackslash{}', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}' };
const texSafe = s => String(s).replace(/[\\{}$&#^_%~]/g, ch => TEXT_ESC[ch] || '\\' + ch);
const same = (a, b) => !!a && !!b && a.kind === b.kind && a.id === b.id;
const cap = s => (s ? s[0].toUpperCase() + s.slice(1) : s);
const hoverKey = t => (!t ? ''
  : t.kind === 'pair' ? `pair:${t.from}>${t.to}`
  : t.kind === 'row' ? `row:${t.layer}:${t.i}`
  : t.kind === 'col' ? `col:${t.layer}:${t.k}:${t.j}`
  : t.kind === 'token' ? `token:${t.layer}:${t.t}:${t.g ?? ''}:${t.h ?? ''}`
  : `${t.kind}:${t.id}`);
// A hovered token { kind: 'token', layer (index), t, g?, h? } against an element's token spec: the
// same layer and token, and the same group and head where both give one.
const tokenMatch = (hv, s) => hv.layer === s.layer && hv.t === s.t
  && (hv.g == null || s.g == null || hv.g === s.g) && (hv.h == null || s.h == null || hv.h === s.h);

function pinIcon() {
  const s = sv('svg', { viewBox: '0 0 16 16', width: 14, height: 14, 'aria-hidden': 'true' });
  s.append(sv('path', { d: 'M6 1.5h4l-.6 4.2 2.6 2.3v1.2H8.6V15h-1.2V9.2H4V8l2.6-2.3z', fill: 'currentColor' }));
  return s;
}

function loadCss() {
  if (document.querySelector('link[href$="nn/inspector.css"], link[data-nn="inspector"]')) return;
  document.head.append(h('link', { rel: 'stylesheet', href: new URL('./inspector.css', import.meta.url).href, 'data-nn': 'inspector' }));
}

// ---------------------------------------------------------------- install

export function install(ctx) {
  const { store } = ctx;
  const M = ctx.model || store.model;
  const stage = ctx.el?.stage || document.getElementById('nn-stage');
  if (!stage) throw new Error('inspector: #nn-stage missing');
  const ro = !!ctx.audience;
  const net = () => store.net;
  const theme = () => (ctx.theme ? ctx.theme() : document.documentElement.dataset.theme) || 'dark';
  const fmt = (x, d = 2) => {
    if (!Number.isFinite(x)) return '?';
    if (M.fmt) return M.fmt(x, d);
    const s = (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d);
    return s;
  };

  loadCss();
  if (getComputedStyle(stage).position === 'static') stage.style.position = 'relative';
  const layerEl = h('div', { class: 'nn-insp-layer' });
  stage.append(layerEl);

  const cards = [];
  const folded = new Set();   // section keys the user folded; shared by every card
  let follow = null;          // the unpinned card that shows state.sel
  let zTop = 1;
  let visible = !document.body.dataset.view || document.body.dataset.view === 'nn';
  let raf = 0;
  let maxW = 1, maxB = 1, maxA = 1;   // the shared colour scales (contract): weights, biases, activations
  // While the Train panel runs, the maths boxes (KaTeX: re-render + relayout) refresh at TEX_HZ;
  // sliders, bars and heatmap cells still follow every frame, and a pause brings everything current.
  const TEX_MS = 100;
  let training = false, texTurn = true, texDue = 0;

  const sizer = typeof ResizeObserver === 'function'
    ? new ResizeObserver(() => { for (const c of cards) c.akey = ''; ensureLoop(); })
    : null;
  sizer?.observe(stage);

  // ---------------------------------------------------------------- model access

  const node = id => M.node(net(), id);
  const edge = id => M.edge(net(), id);
  const layerOf = id => net().layers.find(l => l.id === id);
  const label = id => node(id)?.label || '?';
  const lastIndex = () => net().layers.length - 1;
  const fwdNode = id => store.state.fwd?.node?.[id] || null;
  const bwdNode = id => store.state.bwd?.node?.[id] || null;
  const actOf = L => net().layers[L]?.act || 'identity';
  const actNames = () => Object.keys(M.ACTS || FN);
  const actLabel = k => M.ACTS?.[k]?.label || k;
  const exists = t => !!t && (t.kind === 'node' ? !!node(t.id)
    : t.kind === 'edge' ? !!edge(t.id)
    : t.kind === 'layer' ? !!layerOf(t.id) : false);
  const indexIn = id => M.nodesIn(net(), M.nodeLayerIndex(net(), id)).findIndex(n => n.id === id);
  const wTex = w => `\\textcolor{${solid(w)}}{${fmt(w)}}`;
  const factor = s => (s.startsWith('-') ? `(${s})` : s);
  const select = t => { if (!ro) store.set('sel', t); };

  // ---------------------------------------------------------------- tokens, ties, attention
  // docs/NN_ATTENTION.md. The geometry is read straight from the layer fields, so a net that has
  // no token layers takes none of these paths.

  const layerAt = L => net().layers[L];
  const isAttn = L => layerAt(L)?.kind === 'attention';
  // Node k of a token layer is group floor(k / (tokens d)), token floor(k / d) % tokens, feature k % d.
  function shapeOf(L) {
    const lay = layerAt(L), size = M.nodesIn(net(), L).length;
    const tokens = Number.isInteger(lay?.tokens) && lay.tokens > 1 ? lay.tokens : 1;
    const groups = Array.isArray(lay?.groups) && lay.groups.length ? lay.groups : null;
    const d = Math.max(1, Math.floor(size / (tokens * (groups ? groups.length : 1))));
    return { tokens, d, groups, attn: lay?.kind === 'attention', size };
  }
  const isTok = L => { const s = shapeOf(L); return s.tokens > 1 || !!s.groups || s.attn; };
  function tokPos(L, k) {
    const s = shapeOf(L);
    return { ...s, g: Math.floor(k / (s.tokens * s.d)), t: Math.floor(k / s.d) % s.tokens, f: k % s.d };
  }
  // The indices softmax normalizes over together: the node's token (and group) on a token layer.
  function softBlock(L, i) {
    if (!isTok(L)) return [0, M.nodesIn(net(), L).length];
    const d = shapeOf(L).d, s = Math.floor(i / d) * d;
    return [s, s + d];
  }
  // Matrix names, as in Q = X W_Q: X for the inputs, Z for an attention layer, else H^(l).
  const matSym = L => (L === 0 ? 'X' : isAttn(L) ? 'Z' : L === lastIndex() ? '\\hat Y' : `H^{(${L})}`);
  const grpSym = (s, g) => (s.groups ? `{${s.groups[g]}}` : null);

  // Token names (docs/NN_LENS.md): net.meta.tokenNames, one per token, replace t_1 ... t_n. Read
  // live (in binds), since a name edit is not a structural change and cards are not rebuilt.
  // focus.js's tokenNames is the rule: a slot holding its own default ('t2' for token 2) is unnamed.
  const tokName = t => tokenNames(net())[t] ?? null;
  const tokNameTex = t => (tokName(t) ? `\\text{${texSafe(tokName(t))}}` : null);
  const tokWord = t => (tokName(t) ? `“${tokName(t)}”` : `token ${t + 1}`);   // plain text (tooltips)
  const headsOf = L => { const v = layerAt(L)?.heads; return Number.isInteger(v) && v > 0 ? v : 1; };
  const causalOf = L => !!layerAt(L)?.causal;
  // d_k per head, from the Q/K/V layer that feeds attention layer L.
  const dkHead = L => shapeOf(L - 1).d / headsOf(L);
  const scaleOf = L => {
    const v = layerAt(L)?.scale;
    return typeof v === 'number' && Number.isFinite(v) ? v : 1 / Math.sqrt(dkHead(L));
  };
  const customScale = L => typeof layerAt(L)?.scale === 'number' && Number.isFinite(layerAt(L).scale);
  const scaleTex = L => (customScale(L) ? fmt(scaleOf(L), 3) : `\\tfrac{1}{\\sqrt{${+dkHead(L).toFixed(3)}}}`);
  const scaleEq = L => (customScale(L) ? fmt(scaleOf(L), 3) : `${scaleTex(L)} = ${fmt(scaleOf(L), 3)}`);
  const attnFwd = (L, hd) => store.state.fwd?.attn?.[L]?.heads?.[hd] || null;
  const attnBwd = (L, hd) => store.state.bwd?.attn?.[L]?.heads?.[hd] || null;
  // Node k of attention layer L: token i, feature f of Z; head hd, column fh inside that head.
  function attnAt(L, k) {
    const s = shapeOf(L), heads = headsOf(L), dh = Math.max(1, Math.floor(s.d / heads));
    const i = Math.floor(k / s.d) % s.tokens, f = k % s.d;
    return { T: s.tokens, d: s.d, heads, dh, i, f, hd: Math.min(heads - 1, Math.floor(f / dh)), fh: f % dh };
  }
  // The node of the Q/K/V layer before attention layer L: group g (0 Q, 1 K, 2 V), token t, feature f.
  const qkvNode = (L, g, t, f) => {
    const s = shapeOf(L - 1);
    return M.nodesIn(net(), L - 1)[g * s.tokens * s.d + t * s.d + f] || null;
  };
  const hsup = (L, hd) => (headsOf(L) > 1 ? `^{(${hd + 1})}` : '');

  const tieOf = e => (e && typeof e.tie === 'string' && e.tie ? e.tie : null);
  const tieGroup = tie => net().edges.filter(e => e.tie === tie);
  // 'W_Q:1,2' is entry (1, 2) of W_Q; a shared bias (model.js's node.tie) is 'b_Q:2', entry 2 of b_Q.
  function tieParts(tie) {
    const m = /^(.*):\s*(\d+)\s*(?:,\s*(\d+)\s*)?$/.exec(tie);
    return m ? { name: m[1], i: +m[2], j: m[3] == null ? null : +m[3] } : { name: tie, i: null, j: null };
  }
  const tieIdx = p => (p.j == null ? `${p.i}` : `${p.i},${p.j}`);
  const tieTex = tie => { const p = tieParts(tie); return p.i ? `{${p.name}}(${tieIdx(p)})` : `{${p.name}}`; };
  const tieText = tie => { const p = tieParts(tie); return p.i ? `${p.name}(${tieIdx(p)})` : p.name; };
  const biasTie = id => { const t = node(id)?.tie; return typeof t === 'string' && t ? t : null; };
  const biasGroup = tie => net().nodes.filter(q => q.tie === tie);
  // Every weight edit goes through here: a tied edge moves its whole group, a fixed one never moves.
  function setW(eid, v) {
    const e = edge(eid);
    if (!e || e.fixed) return;
    M.setWeight(net(), eid, v);
    const t = tieOf(e);
    if (t) for (const o of net().edges) if (o.tie === t && o !== e) o.w = e.w;
  }

  // Notation, shared with the matrix panel: W^{(l)}_{i,j} has row i = receiving neuron and column
  // j = sending neuron, 1-based. A skip term from layer k is W^{(l,k)}. a^{(0)} = x.
  const wUp = (lt, lf) => (lf === lt - 1 ? `${lt}` : `${lt},${lf}`);
  const wSym = (lt, lf, i, j) => `W^{(${wUp(lt, lf)})}_{${i},${j}}`;
  const wText = (lt, lf, i, j) => `W^(${wUp(lt, lf)})_${i},${j}`;   // plain text, for tooltips
  const aSym = (l, j) => (l === 0 ? `x_{${j}}` : `a^{(${l})}_{${j}}`);
  // gradients: 3 decimals, and below 0.01 two significant figures (0.0034, 3.4e-4) as the matrix panel
  const g3 = v => (!Number.isFinite(v) ? '?' : M.fmtg ? M.fmtg(v, 3).replace(/e(-?\d+)$/, '\\mathrm{e}{$1}') : fmt(v, 3));
  const aligned = lines => `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
  // A term "w · v" with a negative weight's minus pulled out front (it keeps its sign colour).
  const wTerm = (w, v) => {
    const ww = Number.isFinite(w) ? w : NaN;
    const mag = `\\textcolor{${solid(ww)}}{${fmt(Math.abs(ww))}}\\cdot ${factor(v)}`;
    return ww < 0 ? `- ${mag}` : mag;
  };

  // The loss head backward() used (model.js): cross-entropy needs a softmax ('ce') or sigmoid
  // ('bce') output layer; anything else is mse, including the xent fallback (bwd.note says so).
  function lossKind() {
    const loss = net().meta?.loss, act = actOf(lastIndex());
    if (loss === 'xent' && act === 'softmax') return 'ce';
    if (loss === 'xent' && act === 'sigmoid') return 'bce';
    return 'mse';
  }
  // Right-hand side of the loss over n outputs, exactly as model.js computes it (ln = natural log).
  function lossTex(kind, n) {
    const one = n === 1, s = one ? '1' : 'i', yh = `\\hat y_{${s}}`, y = `y_{${s}}`;
    const sum = one ? '' : `\\sum_{i=1}^{${n}}`;
    if (kind === 'ce') return `-${sum} ${y} \\ln ${yh}`;
    if (kind === 'bce') return `-${one ? '' : `\\tfrac{1}{${n}}`}${sum} \\big[${y} \\ln ${yh} + (1 - ${y}) \\ln (1 - ${yh})\\big]`;
    return `\\tfrac{1}{2}${one ? '' : `\\cdot\\tfrac{1}{${n}}`}${sum} (${yh} - ${y})^2`;
  }
  // The Train panel's learning rate: train.js's readSettings once it has loaded (a preset's lr for
  // an old net matched by title), else meta.train.lr with the same default and clamp.
  let trainSettings = null;
  import('./train.js').then(m => { if (typeof m.readSettings === 'function') trainSettings = m.readSettings; }).catch(() => {});
  function learningRate() {
    if (trainSettings) try { return trainSettings(net(), M).lr; } catch { /* fall back */ }
    const lr = net().meta?.train?.lr;
    return typeof lr === 'number' && Number.isFinite(lr) ? Math.min(1000, Math.max(1e-7, lr)) : 0.1;
  }

  function solid(v) {
    const m = /(\d+),\s*(\d+),\s*(\d+)/.exec(colorFor(v >= 0 ? 1 : -1, 1, theme()));
    return m ? '#' + m.slice(1, 4).map(x => (+x).toString(16).padStart(2, '0')).join('') : '#888888';
  }

  function actFn(act) {
    const A = M.ACTS?.[act];
    return A && typeof A.f === 'function' ? z => A.f(z) : z => z;
  }

  // Layer-wise activations (softmax): the scalar ACTS f is only a stand-in, never plot it.
  const isVec = act => !!M.ACTS?.[act]?.vector || act === 'softmax';

  function actDef(act) {
    const tex = M.ACTS?.[act]?.tex;
    if (tex) return `${isVec(act) ? 'a_i' : 'a'} = \\displaystyle ${tex}`;
    return DEF[act] || `a = ${FN[act] || act}(z)`;
  }

  // ---------------------------------------------------------------- selection, cards

  function syncSel() {
    const s = store.state.sel;
    const t = s && exists(s) ? { kind: s.kind, id: s.id } : null;
    const pinned = t && cards.find(c => c.pinned && same(c.target, t));
    if (!t || pinned) {
      if (follow) close(follow);
      if (pinned) { raise(pinned); flash(pinned); }
    } else if (follow) {
      if (!same(follow.target, t)) {
        follow.target = t;
        build(follow);
        follow.akey = '';
        place(follow);
      }
    } else {
      follow = open(t);
    }
    for (const c of cards) c.el.classList.toggle('sel', same(c.target, s));
    ensureLoop();
  }

  function open(target, { pin = false } = {}) {
    const c = { target, pinned: pin, dragged: false, x: 0, y: 0, binds: [], hov: [], dims: [], akey: '', failed: false };
    c.title = h('span', { class: 'nn-insp-title' });
    c.kind = h('span', { class: 'nn-insp-kind' });
    c.pinBtn = h('button', { class: 'nn-insp-btn nn-insp-pin', title: 'Pin: keep this card open (the next selection opens another)' }, pinIcon());
    c.closeBtn = h('button', { class: 'nn-insp-btn', title: 'Close' }, '×');
    c.head = h('header', { class: 'nn-insp-head', title: 'Drag to move. Double-click to put it back next to its target.' },
      c.title, c.kind, h('span', { class: 'nn-insp-sp' }), ro ? null : c.pinBtn, ro ? null : c.closeBtn);
    c.body = h('div', { class: 'nn-insp-body' });
    c.el = h('div', { class: 'nn-insp' + (ro ? ' ro' : '') }, c.head, c.body);
    c.el.style.width = CARD_W + 'px';
    for (const ev of STOP) c.el.addEventListener(ev, e => e.stopPropagation(), { passive: true });
    c.el.addEventListener('pointerdown', () => raise(c));
    // Mouse clicks don't leave focus on card buttons, so Space still reaches play/pause.
    c.el.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });
    c.el.addEventListener('change', e => { if (e.target.matches('select')) e.target.blur(); });
    c.pinBtn.addEventListener('click', () => togglePin(c));
    c.closeBtn.addEventListener('click', () => {
      if (c === follow || same(c.target, store.state.sel)) store.set('sel', null);
      if (cards.includes(c)) close(c);
    });
    c.head.addEventListener('pointerenter', () => { if (!ro) store.set('hover', { kind: c.target.kind, id: c.target.id }); });
    c.head.addEventListener('pointerleave', () => clearHover({ kind: c.target.kind, id: c.target.id }));
    c.head.addEventListener('dblclick', e => {
      if (e.target.closest('button')) return;
      c.dragged = false; c.akey = ''; ensureLoop();
    });
    dragHandle(c);
    layerEl.append(c.el);
    cards.push(c);
    raise(c);
    paintPin(c);
    build(c);
    place(c);
    sizer?.observe(c.el);
    return c;
  }

  function close(c) {
    const i = cards.indexOf(c);
    if (i < 0) return;
    const hk = hoverKey(store.state.hover);
    if (hk && (c.hov.some(el => el.dataset.hk === hk) || hk === hoverKey(c.target))) store.set('hover', null);
    sizer?.unobserve(c.el);
    c.el.remove();
    cards.splice(i, 1);
    if (follow === c) follow = null;
  }

  function togglePin(c) {
    c.pinned = !c.pinned;
    if (c.pinned) {
      if (follow === c) follow = null;
    } else if (same(c.target, store.state.sel)) {
      if (follow && follow !== c) close(follow);
      follow = c;
    } else {
      close(c);
      return;
    }
    paintPin(c);
  }

  function paintPin(c) {
    c.el.classList.toggle('pinned', c.pinned);
    c.pinBtn.classList.toggle('on', c.pinned);
    c.pinBtn.title = c.pinned ? 'Unpin' : 'Pin: keep this card open (the next selection opens another)';
  }

  function raise(c) { c.el.style.zIndex = String(++zTop); }

  function flash(c) {
    c.el.classList.remove('flash');
    void c.el.offsetWidth;
    c.el.classList.add('flash');
  }

  function clearHover(t) {
    if (hoverKey(store.state.hover) === hoverKey(t)) store.set('hover', null);
  }

  // ---------------------------------------------------------------- placement

  function anchor(t) {
    const v = ctx.view;
    if (!v?.nodeRect) return null;
    if (t.kind === 'node') return v.nodeRect(t.id);
    if (t.kind === 'edge') {
      const e = edge(t.id);
      if (!e) return null;
      const a = v.nodeRect(e.from), b = v.nodeRect(e.to);
      if (!a || !b) return a || b;
      const x = (a.x + a.w / 2 + b.x + b.w / 2) / 2, y = (a.y + a.h / 2 + b.y + b.h / 2) / 2;
      return { x: x - 6, y: y - 6, w: 12, h: 12 };
    }
    if (t.kind === 'layer') {
      const L = M.layerIndex(net(), t.id);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const n of M.nodesIn(net(), L)) {
        const r = v.nodeRect(n.id);
        if (!r) continue;
        x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y);
        x1 = Math.max(x1, r.x + r.w); y1 = Math.max(y1, r.y + r.h);
      }
      return x0 < Infinity ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null;
    }
    return null;
  }

  function clampTo(c, x, y) {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const w = c.el.offsetWidth || CARD_W, hh = c.el.offsetHeight || 200;
    c.x = Math.round(Math.max(EDGE, Math.min(x, sw - w - EDGE)));
    c.y = Math.round(Math.max(EDGE, Math.min(y, sh - Math.min(hh, 60) - EDGE)));
    setStyle(c.el, 'left', c.x + 'px');
    setStyle(c.el, 'top', c.y + 'px');
  }

  // Where a card may go. view.fit() frames the net in all the room beside the Train panel, so a
  // card level with its target usually has to cover something. Each x along the stage is scored
  // by what the card would hide:
  // - its target: never;
  // - neurons in the target's layers and their neighbours (the ones being explained): 4 each;
  //   any other neuron: 1;
  // - the rest of the net's box, where the next neuron gets added while building: up to 3;
  // - the Train panel: up to 3; a card placed before this one: up to 2;
  // then the nearest to the target wins a tie. So a card sits off the net when there is room,
  // over the Train panel before over the net's middle, and otherwise over the far layers.
  const overlap = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

  function targetLayers(t) {
    const n = net();
    if (t.kind === 'node') return [M.nodeLayerIndex(n, t.id)];
    if (t.kind === 'edge') { const e = edge(t.id); return e ? [M.nodeLayerIndex(n, e.from), M.nodeLayerIndex(n, e.to)] : []; }
    return [M.layerIndex(n, t.id)];
  }

  function bestX(c, r, y, w, hh, sw) {
    const v = ctx.view, n = net(), card = x => ({ x, y, w, h: hh });
    const t = c.target;
    const own = new Set(t.kind === 'node' ? [t.id] : t.kind === 'edge' ? [edge(t.id)?.from, edge(t.id)?.to]
      : M.nodesIn(n, M.layerIndex(n, t.id)).map(q => q.id));
    const ls = targetLayers(c.target).filter(l => l >= 0);
    const near = l => ls.some(k => Math.abs(k - l) <= 1);
    const nodes = [];
    for (const q of n.nodes) {
      const qr = v?.nodeRect?.(q.id);
      if (qr) nodes.push({ r: qr, cost: own.has(q.id) ? 1e6 : near(M.nodeLayerIndex(n, q.id)) ? 4 : 1 });
    }
    const soft = [];   // [box, full cost]
    const cr = v?.contentRect?.();
    if (cr && cr.w > 0 && cr.h > 0) soft.push([cr, 3]);
    const sr = stage.getBoundingClientRect();
    for (const el of stage.querySelectorAll('.nn-train, .nn-attnviz, .nn-s3d')) {
      if (el.hidden || !el.offsetWidth) continue;
      const tr = el.getBoundingClientRect();
      soft.push([{ x: tr.left - sr.left, y: tr.top - sr.top, w: tr.width, h: tr.height }, 3]);
    }
    for (const o of cards) {
      if (o === c) break;
      if (o.el.offsetWidth) soft.push([{ x: o.x, y: o.y, w: o.el.offsetWidth, h: o.el.offsetHeight }, 2]);
    }
    const lo = EDGE, hi = Math.max(EDGE, sw - w - EDGE), cx = r.x + r.w / 2;
    const xs = new Set([lo, hi, r.x + r.w + GAP, r.x - GAP - w]);
    for (const [b] of soft) { xs.add(b.x - GAP - w); xs.add(b.x + b.w + GAP); }
    for (let x = lo; x < hi; x += 16) xs.add(x);
    let best = null;
    for (const x0 of xs) {
      const x = Math.round(Math.max(lo, Math.min(hi, x0))), box = card(x);
      let cost = 0;
      for (const q of nodes) if (overlap(box, q.r) > 0) cost += q.cost;
      for (const [b, full] of soft) cost += (full * overlap(box, b)) / Math.max(1, Math.min(w * hh, b.w * b.h));
      cost += Math.abs(x + w / 2 - cx) / (sw * 100);   // tie-break: nearer the target
      if (!best || cost < best.cost) best = { x, cost };
    }
    return best ? best.x : lo;
  }

  // Level with the target, at the x that hides the least (bestX).
  function place(c) {
    if (c.dragged) { clampTo(c, c.x, c.y); return; }
    const r = anchor(c.target);
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const w = c.el.offsetWidth || CARD_W, hh = c.el.offsetHeight || 200;
    let x, y;
    if (!r) {
      x = sw - w - EDGE - 18 * (cards.indexOf(c) % 6);
      y = EDGE + 40 + 18 * (cards.indexOf(c) % 6);
    } else {
      y = Math.max(EDGE, Math.min(r.y + r.h / 2 - 22, sh - Math.min(hh, sh) - EDGE));
      x = bestX(c, r, y, w, Math.min(hh, sh - 2 * EDGE), sw);
    }
    clampTo(c, x, y);
  }

  function dragHandle(c) {
    c.head.addEventListener('pointerdown', e => {
      if (e.button !== 0 || e.target.closest('button, input, select')) return;
      e.preventDefault();
      raise(c);
      const ox = e.clientX - c.x, oy = e.clientY - c.y;
      const x0 = e.clientX, y0 = e.clientY;
      c.head.setPointerCapture?.(e.pointerId);
      c.head.classList.add('grabbing');
      const move = ev => {
        if (!c.dragged && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 3) return;
        c.dragged = true;
        clampTo(c, ev.clientX - ox, ev.clientY - oy);
      };
      const up = () => {
        c.head.removeEventListener('pointermove', move);
        c.head.removeEventListener('pointerup', up);
        c.head.removeEventListener('pointercancel', up);
        c.head.classList.remove('grabbing');
      };
      c.head.addEventListener('pointermove', move);
      c.head.addEventListener('pointerup', up);
      c.head.addEventListener('pointercancel', up);
    });
  }

  // Cards follow their targets for a short while after anything that can move them on screen:
  // pointer / wheel / key input in the view (pan, zoom, fit), node drags ('layout'), structure
  // changes and resizes. Then the loop stops, so training ticks cost no layout reads here.
  let trackUntil = 0;
  function ensureLoop(ms = 900) {
    trackUntil = Math.max(trackUntil, performance.now() + ms);
    if (!raf && visible && cards.length) raf = requestAnimationFrame(tick);
  }
  function tick() {
    raf = 0;
    if (!visible || !cards.length) return;
    for (const c of cards) {
      if (c.dragged) {
        if (c.akey !== 'dragged') { c.akey = 'dragged'; place(c); }
        continue;
      }
      const r = anchor(c.target), cr = r && ctx.view?.contentRect?.();
      const box = q => (q ? `${Math.round(q.x)},${Math.round(q.y)},${Math.round(q.w)},${Math.round(q.h)}` : '');
      const k = r ? `${box(r)}|${box(cr)}` : '-';   // the net's box too: a card moves off a growing net
      if (k !== c.akey) { c.akey = k; place(c); }
    }
    if (performance.now() < trackUntil) raf = requestAnimationFrame(tick);
  }
  const nudge = e => {
    if (!cards.length || layerEl.contains(e.target)) return;
    if (e.type === 'pointermove' && !e.buttons) return;
    ensureLoop();
  };
  const inputRoot = ctx.el?.root || stage;   // includes the toolbar (Fit) and the split handle
  for (const ev of ['pointerdown', 'pointermove', 'wheel']) inputRoot.addEventListener(ev, nudge, { capture: true, passive: true });
  window.addEventListener('keydown', nudge, true);

  // ---------------------------------------------------------------- build + update

  function build(c) {
    const top = c.body.scrollTop;
    const hk = hoverKey(store.state.hover);
    if (hk && c.hov.some(el => el.dataset.hk === hk)) store.set('hover', null);
    c.binds = [];
    c.hov = [];
    c.dims = [];
    c.failed = false;
    c.body.replaceChildren();
    c.el.dataset.kind = c.target.kind;
    try {
      if (c.target.kind === 'node') buildNode(c, c.target.id);
      else if (c.target.kind === 'edge') buildEdge(c, c.target.id);
      else buildLayer(c, c.target.id);
    } catch (err) {
      console.error('[nn] inspector build:', err);
      c.body.append(h('div', { class: 'nn-hint err', text: 'Could not show this: ' + err.message }));
    }
    if (ro) for (const el of c.body.querySelectorAll('input, select, button, textarea')) el.disabled = true;
    update(c);
    c.body.scrollTop = top;
    paintHover();
  }

  function update(c) {
    if (!exists(c.target)) { close(c); return; }
    for (const f of c.binds) {
      try { f(); } catch (err) {
        if (!c.failed) console.error('[nn] inspector update:', err);   // once per card, not per frame
        c.failed = true;
      }
    }
    paintLens(c);
  }

  function updateAll() {
    let w = 0, b = 0;
    for (const e of net().edges) if (Number.isFinite(e.w)) w = Math.max(w, Math.abs(e.w));
    for (const n of net().nodes) if (Number.isFinite(n.bias)) b = Math.max(b, Math.abs(n.bias));
    maxW = w || 1;
    maxB = b || 1;
    // activations: every entry of fwd.a and fwd.z, as the view's neuron fill and the matrix x cells
    const f = store.state.fwd;
    let a = 0;
    for (const vs of [...(f?.a || []), ...(f?.z || [])]) for (const v of vs || []) if (Number.isFinite(v)) a = Math.max(a, Math.abs(v));
    if (!f) for (const n of M.nodesIn(net(), 0)) if (Number.isFinite(n.value)) a = Math.max(a, Math.abs(n.value));
    maxA = a || 1;
    const now = performance.now();
    texTurn = !training || now >= texDue;
    if (texTurn && training) texDue = now + TEX_MS;
    try { for (const c of [...cards]) update(c); } finally { texTurn = true; }
  }

  function rebuildAll() {
    for (const c of [...cards]) {
      if (!exists(c.target)) close(c);
      else build(c);
    }
  }

  function paintHover() {
    const t = store.state.hover;
    const k = hoverKey(t);
    const nodeId = t?.kind === 'node' ? t.id : null;
    const tok = t?.kind === 'token' ? { ...t, layer: typeof t.layer === 'number' ? t.layer : M.layerIndex(net(), t.layer) } : null;
    for (const c of cards) {
      for (const el of c.hov) {
        el.classList.toggle('hi', (!!k && (el.dataset.hk === k || (!!nodeId && el.dataset.other === nodeId)))
          || (!!tok && !!el._toks?.some(s => tokenMatch(tok, s))));
      }
    }
  }

  // ---------------------------------------------------------------- lens (docs/NN_LENS.md)
  // What the lens de-emphasizes on the canvas fades in the cards too, weighed by focus.js's
  // emphasis(): a weight row or used-by chip by its edge, a bias or input slider by its neuron, an
  // attention bar or heatmap cell by its A_ij edge, a query header by the lens's token rows, a
  // head's heatmap by the lens's heads. What the lens hides (edge types off, below a threshold)
  // fades further. The card's own target never fades.
  const DIM_MIN = 0.3, DIM_HID = 0.16;
  const emphasisFn = emphasis;
  let emWarned = false;
  let emCache = null, emLens, emFwd, emNet = '';
  function lensEm() {
    const lens = store.state.lens, fwd = store.state.fwd;
    if (!emphasisFn || !lens) return null;
    const sig = `${net().layers.length}|${net().nodes.length}|${net().edges.length}`;
    if (lens === emLens && fwd === emFwd && sig === emNet) return emCache;
    emLens = lens; emFwd = fwd; emNet = sig;
    try { emCache = emphasisFn(net(), fwd, lens) || null; } catch (err) {
      emCache = null;
      if (!emWarned) { emWarned = true; console.warn('[nn] inspector: lens emphasis failed:', err); }
    }
    return emCache;
  }
  // 0..1 emphasis, 'hid' (hidden by the lens) or null (no opinion: leave it as it is)
  const edgeEm = (E, id) => (E.hidden?.edge?.(id) ? 'hid' : E.any ? E.edge?.(id) ?? null : null);
  const nodeEm = (E, id) => (E.any ? E.node?.(id) ?? null : null);
  const attnEm = (E, l, i, j, hd) => (E.hidden?.attn?.(l, i, j, hd) ? 'hid' : E.any ? E.attn?.(l, i, j, hd) ?? null : null);
  const rowEm = (E, l, i) => { const r = E.rows?.(l); return r && !r.has(i) ? 0 : null; };
  const headEm = (E, l, hd) => { const s = E.heads?.(l); return s && !s.has(hd) ? 0 : null; };
  // el fades by fn(emphasis) while the lens is on
  function dimBy(c, el, fn) { c.dims.push({ el, fn }); }
  function applyDim(el, v) {
    const hid = v === 'hid';
    const o = hid ? DIM_HID : typeof v !== 'number' || !(v < 0.999) ? 1 : DIM_MIN + (1 - DIM_MIN) * Math.max(0, v);
    const s = o >= 1 ? '' : o.toFixed(2);
    if (el._lensO === s) return;
    el._lensO = s;
    el.style.opacity = s;
    el.classList.toggle('nn-lens-out', !!s);
    el.classList.toggle('nn-lens-hid', hid);
  }
  function paintLens(c) {
    if (!c.dims.length) return;
    const E = lensEm();
    for (const d of c.dims) {
      let v = null;
      if (E) try { v = d.fn(E); } catch { v = null; }
      applyDim(d.el, v);
    }
  }
  function paintLensAll() { for (const c of cards) paintLens(c); }

  // ---------------------------------------------------------------- widgets

  function section(c, key, title, sub, ...kids) {
    const head = h('h4', { class: 'nn-sec-h' }, h('span', { class: 'nn-fold' }), title);
    if (sub) {
      const s = h('span', { class: 'nn-sec-sub' });
      setTex(s, sub);
      head.append(s);
    }
    const body = h('div', { class: 'nn-sec-b' }, ...kids);
    const sec = h('section', { class: 'nn-sec' + (folded.has(key) ? ' folded' : ''), 'data-sec': key }, head, body);
    head.addEventListener('click', () => {
      if (sec.classList.toggle('folded')) folded.add(key); else { folded.delete(key); update(c); }   // folded maths is not kept current
    });
    c.body.append(sec);
    return { sec, head, body };
  }

  function hoverable(c, el, key, other = null) {
    el.dataset.hk = hoverKey(key);
    if (other) el.dataset.other = other;
    c.hov.push(el);
    if (ro) return;
    el.addEventListener('pointerenter', () => store.set('hover', key));
    el.addEventListener('pointerleave', () => clearHover(key));
  }
  // el also lights while one of these tokens is hovered ([{ layer, t, g?, h? }], see tokenMatch).
  function tokenLit(c, el, specs) {
    el._toks = specs;
    if (!c.hov.includes(el)) c.hov.push(el);
  }

  // A labelled value slider with a number box. `set` writes into store.net; dragging calls
  // store.touch(), releasing (or leaving the number box) calls store.commit(what).
  function slider(c, { lab, get, set, what, hover, other, scale = () => maxW, onLabel, labelTitle }) {
    const sw = h('span', { class: 'nn-sw' });
    const name = h('span', { class: 'nn-sl-lab' + (onLabel ? ' link' : ''), title: labelTitle || null });
    const range = h('input', { type: 'range', min: -SPAN, max: SPAN, step: '0.01' });
    const box = h('input', { type: 'number', step: '0.01', class: 'nn-num' });
    const row = h('div', { class: 'nn-sl' }, sw, name, range, box);
    let span = SPAN, dragging = false, released = false, before = '';
    if (onLabel && !ro) name.addEventListener('click', onLabel);
    if (hover) hoverable(c, row, hover, other);
    // Pointer gestures commit on release; the 'change' that follows the release is swallowed so
    // keyboard steps (which only fire 'change') still commit once each.
    const endDrag = () => {
      if (!dragging) return;
      dragging = false;
      window.removeEventListener('pointerup', endDrag, true);
      window.removeEventListener('pointercancel', endDrag, true);
      released = true;
      setTimeout(() => { released = false; }, 0);
      store.commit(what);
    };
    range.addEventListener('pointerdown', () => {
      dragging = true;
      window.addEventListener('pointerup', endDrag, true);
      window.addEventListener('pointercancel', endDrag, true);
    });
    range.addEventListener('input', () => {
      const v = +range.value;
      set(v);
      box.value = numText(v);
      store.touch();
    });
    range.addEventListener('change', () => { if (!dragging && !released) store.commit(what); });
    range.addEventListener('dblclick', () => { set(0); store.commit(what); });
    range.title = 'Drag to change. Double-click for 0.';
    box.addEventListener('focus', () => { before = box.value; });
    box.addEventListener('input', () => {
      const v = parseNum(box.value);
      if (v != null) { set(v); store.touch(); }
    });
    box.addEventListener('change', () => {
      const v = parseNum(box.value);
      if (v != null) set(v);
      store.commit(what);
    });
    box.addEventListener('keydown', e => {
      if (e.key === 'Enter') box.blur();
      else if (e.key === 'Escape') {
        const v = parseNum(before);
        if (v != null) { set(v); store.touch(); }
        box.value = before;
        box.blur();
      }
    });
    c.binds.push(() => {
      lab?.(name);
      const v = get();
      const ok = Number.isFinite(v);
      if (!dragging && ok && Math.abs(v) > span) {
        span = fitSpan(v);
        range.min = String(-span);
        range.max = String(span);
      }
      if (!dragging && ok && range._v !== v) { range._v = v; range.value = String(v); }
      setVal(box, numText(v));
      setStyle(sw, 'background', ok ? colorFor(v, scale(), theme()) : 'transparent');
      setStyle(range, 'accentColor', ok ? solid(v) : '');
    });
    return row;
  }

  // An entry of W that has no edge: shown as a masked 0 with a connect button.
  function maskedRow(c, fromId, toId, otherId) {
    const name = h('span', { class: 'nn-sl-lab' });
    const btn = h('button', { class: 'nn-mini', title: 'Add this edge' }, '+ connect');
    const row = h('div', { class: 'nn-sl masked' }, h('span', { class: 'nn-sw' }), name,
      h('span', { class: 'nn-masked-txt', text: 'no edge: fixed 0' }), btn);
    hoverable(c, row, { kind: 'pair', from: fromId, to: toId }, otherId);
    btn.addEventListener('click', () => {
      if (M.connect(net(), fromId, toId) != null) store.commit('connect');
    });
    c.binds.push(() => setTex(name, label(otherId)));
    return row;
  }

  function texBox(c, fn, display = false, cls = 'nn-math') {
    const el = h('div', { class: cls });
    c.binds.push(() => {
      if (!texTurn && el._tex !== undefined) return;         // training: at TEX_MS, not every frame
      if (el._tex !== undefined && el.closest('.nn-sec.folded')) return;
      const s = fn();
      setHidden(el, s == null);
      if (s != null) setTex(el, s, display);
    });
    return el;
  }

  function textInput(c, { get, set, what, cls = 'nn-text', placeholder, live = true }) {
    const inp = h('input', { type: 'text', class: cls, spellcheck: 'false', autocomplete: 'off', placeholder });
    let before = '';
    inp.addEventListener('focus', () => { before = inp.value; });
    if (live) inp.addEventListener('input', () => { set(inp.value); store.touch(); });
    inp.addEventListener('change', () => { set(inp.value); store.commit(what); });
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter') inp.blur();
      else if (e.key === 'Escape') {
        inp.value = before;
        set(before);
        store.touch();
        inp.blur();
      }
    });
    c.binds.push(() => setVal(inp, get() ?? ''));
    return inp;
  }

  function actSelect(c, layerId) {
    const sel = h('select', { class: 'nn-act', title: 'Activation of the whole layer' },
      actNames().map(k => h('option', { value: k }, actLabel(k))));
    sel.addEventListener('change', () => {
      M.setLayer(net(), layerId, { act: sel.value });
      store.commit('activation');
    });
    c.binds.push(() => setVal(sel, layerOf(layerId)?.act || 'identity'));
    return sel;
  }

  // Plot of an activation over z, with a dot at each (z, a) point.
  function makePlot(W = 296, H = 80) {
    const P = 8;
    const svg = sv('svg', { class: 'nn-plot', viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const axes = sv('path', { class: 'nn-plot-axis' });
    const curve = sv('path', { class: 'nn-plot-curve' });
    const guide = sv('path', { class: 'nn-plot-guide' });
    const dots = sv('g', { class: 'nn-plot-dots' });
    const tz = sv('text', { class: 'nn-plot-tick', x: W - P, y: H - 2, 'text-anchor': 'end' });
    const tz0 = sv('text', { class: 'nn-plot-tick', x: P, y: H - 2 });
    const tzl = sv('text', { class: 'nn-plot-tick nn-plot-var', x: W - P, 'text-anchor': 'end' });
    const ta = sv('text', { class: 'nn-plot-tick nn-plot-var', y: P + 4 });
    tzl.textContent = 'z';
    ta.textContent = 'a';
    svg.append(axes, guide, curve, dots, tz, tz0, tzl, ta);
    let key = '', X = z => z, Y = a => a;
    return {
      el: svg,
      draw(f, pts, curveKey) {
        let m = 0;
        for (const p of pts) if (Number.isFinite(p.z)) m = Math.max(m, Math.abs(p.z));
        const R = Math.max(4, Math.ceil(m * 1.15));
        const k = `${curveKey}|${R}`;
        if (k !== key) {
          key = k;
          const N = 80, zs = [], as = [];
          let lo = 0, hi = 0;
          for (let q = 0; q <= N; q++) {
            const z = -R + (2 * R * q) / N;
            let a = f(z);
            if (!Number.isFinite(a)) a = 0;
            zs.push(z); as.push(a);
            lo = Math.min(lo, a); hi = Math.max(hi, a);
          }
          if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
          const pad = (hi - lo) * 0.08;
          lo -= pad; hi += pad;
          X = z => P + ((z + R) / (2 * R)) * (W - 2 * P);
          Y = a => H - P - 6 - ((a - lo) / (hi - lo)) * (H - 2 * P - 6);
          setAttr(axes, 'd', `M${P},${Y(0).toFixed(1)}H${W - P}M${X(0).toFixed(1)},${P}V${H - P - 6}`);
          setAttr(curve, 'd', zs.map((z, q) => `${q ? 'L' : 'M'}${X(z).toFixed(1)},${Y(as[q]).toFixed(1)}`).join(''));
          setText(tz, `${R}`);
          setText(tz0, `−${R}`);
          setAttr(tzl, 'y', (Y(0) - 4).toFixed(1));
          setAttr(ta, 'x', (X(0) + 5).toFixed(1));
        }
        while (dots.childNodes.length < pts.length) dots.append(sv('circle', { r: 4 }));
        while (dots.childNodes.length > pts.length) dots.lastChild.remove();
        pts.forEach((p, q) => {
          const d = dots.childNodes[q];
          const ok = Number.isFinite(p.z) && Number.isFinite(p.a);
          setAttr(d, 'visibility', ok ? 'visible' : 'hidden');
          if (ok) {
            setAttr(d, 'cx', X(p.z).toFixed(1));
            setAttr(d, 'cy', Y(p.a).toFixed(1));
          }
          setAttr(d, 'class', p.hi ? 'hi' : '');
        });
        const p = pts.length === 1 ? pts[0] : null;
        const g = p && Number.isFinite(p.z) && Number.isFinite(p.a)
          ? `M${X(p.z).toFixed(1)},${Y(0).toFixed(1)}V${Y(p.a).toFixed(1)}H${X(0).toFixed(1)}` : '';
        setAttr(guide, 'd', g);
      },
    };
  }

  // Softmax output i as a function of z_i with the other logits held fixed.
  function softmaxSlice(L, i) {
    const [b0, b1] = softBlock(L, i);   // a token layer normalizes each token on its own
    const zs = (store.state.fwd?.z?.[L] || []).slice(b0, b1);
    i -= b0;
    let m = -Infinity;
    zs.forEach((z, k) => { if (k !== i && Number.isFinite(z)) m = Math.max(m, z); });
    if (m === -Infinity) return { f: () => 1, key: 'softmax|1' };
    let s = 0;
    zs.forEach((z, k) => { if (k !== i && Number.isFinite(z)) s += Math.exp(z - m); });
    return { f: z => 1 / (1 + s * Math.exp(m - z)), key: `softmax|${m.toFixed(2)}|${s.toFixed(3)}` };
  }

  // ---------------------------------------------------------------- node card

  function buildNode(c, id) {
    const L = M.nodeLayerIndex(net(), id);
    const layerId = net().layers[L].id;
    const last = lastIndex();
    const i = indexIn(id);
    const I = i + 1;
    const kind = L === 0 ? 'input' : L === last ? 'output' : 'hidden';
    const attn = isAttn(L);                          // no incoming edges, no bias: z = sum_j A_ij V_jf
    const tp = isTok(L) ? tokPos(L, i) : null;
    const feedsAttn = L < last && isAttn(L + 1);     // a Q / K / V neuron
    c.binds.push(() => {
      setTex(c.title, label(id));
      const name = layerOf(layerId)?.name;
      const k = attn && L < last ? 'attention' : kind;
      setText(c.kind, `${k} · ${name && name.trim().toLowerCase() !== k ? name : 'layer ' + L}`);
    });

    // where it sits + label
    const where = L === 0
      ? `\\text{entry } ${I} \\text{ of } x = a^{(0)}` + (last > 0 && !feedsAttn ? `,\\ \\text{column } ${I} \\text{ of } W^{(1)}` : '')
      : attn ? `\\text{entry } (${tp.t + 1}, ${tp.f + 1}) \\text{ of } Z = AV` +
        (L < last ? `;\\ \\text{column } ${I} \\text{ of } W^{(${L + 1})}` : `;\\ \\hat y_{${I}}`)
      : `\\text{row } ${I} \\text{ of } W^{(${L})},\\ b^{(${L})}_{${I}}` +
        (L === last ? `;\\ \\hat y_{${I}}` : feedsAttn ? '' : `;\\ \\text{column } ${I} \\text{ of } W^{(${L + 1})}`);
    const whereEl = h('div', { class: 'nn-where' });
    setTex(whereEl, `${where}\\,`);   // the thin space holds \hat y's overhang (else a 2 px scrollbar)
    // which token and which feature, on a token layer: row t, column f of X / Q / K / V / Z / H
    let tokEl = null;
    if (tp) {
      const m = grpSym(tp, tp.g) || matSym(L);
      const ah = attn ? attnAt(L, i) : null;
      tokEl = h('div', { class: 'nn-where nn-tok' });
      c.binds.push(() => {
        const nm = tokNameTex(tp.t);
        setTex(tokEl, (nm ? `${nm}\\ (\\text{token } ${tp.t + 1})` : `\\text{token } ${tp.t + 1}`) + `,\\ \\text{feature } ${tp.f + 1}` +
          `\\quad (\\text{row } ${tp.t + 1},\\ \\text{column } ${tp.f + 1} \\text{ of } ${m})` +
          (ah && ah.heads > 1 ? `,\\ \\text{head } ${ah.hd + 1}` : ''));
      });
    }
    const chip = h('button', { class: 'nn-chip', title: 'Open the layer' });
    chip.addEventListener('click', () => select({ kind: 'layer', id: layerId }));
    c.binds.push(() => setText(chip, `${layerOf(layerId)?.name || 'layer ' + L} →`));
    const lab = textInput(c, {
      get: () => node(id)?.label,
      set: v => M.setNode(net(), id, { label: v }),
      what: 'label', cls: 'nn-text nn-tex-src', placeholder: 'KaTeX label',
    });
    c.body.append(h('div', { class: 'nn-top' },
      h('div', { class: 'nn-grid2' }, h('span', { class: 'nn-k', text: 'label' }), lab),
      tokEl,
      h('div', { class: 'nn-where-row' }, whereEl, chip)));

    if (L === 0) {
      section(c, 'node.value', 'Input value', null, slider(c, {
        lab: el => setTex(el, label(id)),
        get: () => node(id)?.value,
        set: v => M.setNode(net(), id, { value: v }),
        what: 'input value', hover: { kind: 'node', id },
        scale: () => maxA,
      }));
    }

    // incoming: the row of W in column order, then skip terms, then the bias
    const incoming = [];
    if (attn) buildAttnNode(c, id, L, i);
    else if (L > 0) {
      const byFrom = new Map(net().edges.filter(e => e.to === id).map(e => [e.from, e]));
      const rows = [];
      // j is the column: the sender's order in its whole layer (skip terms list only their edges)
      const addTerm = (k, sub) => {
        const list = [];
        M.nodesIn(net(), k).forEach((p, j) => {
          const e = byFrom.get(p.id);
          if (e) {
            incoming.push({ eid: e.id, from: p.id, k, j });
            list.push(weightRow(c, e.id, p.id, wText(L, k, I, j + 1)));
          } else if (k === L - 1) {
            list.push(maskedRow(c, p.id, id, p.id));
          }
        });
        if (list.length) rows.push(sub ? h('div', { class: 'nn-sub' }, sub) : null, ...list);
      };
      addTerm(L - 1, null);
      for (let k = L - 2; k >= 0; k--) {
        if (M.nodesIn(net(), k).some(p => byFrom.has(p.id))) {
          addTerm(k, texEl(`\\text{skip from ${texSafe(net().layers[k].name || 'layer ' + k)}: } W^{(${L},${k})}`));
        }
      }
      if (!rows.length) rows.push(h('div', { class: 'nn-hint', text: 'No incoming edges.' }));
      const bt = biasTie(id);   // a shared bias (one per feature, used by every token): setNode moves them all
      const brow = slider(c, {
        lab: el => setTex(el, bt ? tieTex(bt) : `b^{(${L})}_{${I}}`),
        get: () => node(id)?.bias,
        set: v => M.setNode(net(), id, { bias: v }),
        what: 'bias', hover: { kind: 'bias', id },
        scale: () => maxB,
        labelTitle: bt ? `b^(${L})_${I} = ${tieText(bt)}, shared by ${biasGroup(bt).length} neurons` : null,
      });
      if (bt) { brow.classList.add('tied'); brow.title = `Shared bias ${tieText(bt)}: moving it moves all ${biasGroup(bt).length} copies`; }
      rows.push(h('div', { class: 'nn-bias' }, brow));
      const s = section(c, 'node.in', 'Weights in + bias', `\\text{row } ${I} \\text{ of } W^{(${L})},\\ b^{(${L})}_{${I}}`, ...rows);
      hoverable(c, s.head, { kind: 'row', layer: L, i });
    }

    // the arithmetic, then the activation with a plot marking the current z
    const math = texBox(c, () => (attn ? attnArith(id, L, I, i) : arithmetic(id, L, I, incoming)), true);
    if (attn) {
      section(c, 'node.math', 'Forward', null, math,
        h('div', { class: 'nn-hint', text: 'An attention layer has no weights in and no bias, and its activation is the identity.' }));
    } else if (L > 0) {
      const plot = makePlot();
      const def = h('div', { class: 'nn-def' });
      c.binds.push(() => {
        const act = actOf(L);
        setTex(def, actDef(act));
        const fw = fwdNode(id);
        const pt = { z: fw?.z, a: fw?.a };
        if (isVec(act)) {
          const { f, key } = softmaxSlice(L, i);
          plot.draw(f, [pt], key);
        } else {
          plot.draw(actFn(act), [pt], act);
        }
      });
      section(c, 'node.math', 'Forward', null, math,
        h('div', { class: 'nn-act-row' }, actSelect(c, layerId), def), plot.el);
    } else {
      section(c, 'node.math', 'Forward', null, math);
    }

    // target (outputs)
    if (L === last && L > 0) {
      const tgt = h('input', { type: 'number', step: '0.1', class: 'nn-num nn-target', placeholder: 'none' });
      const clr = h('button', { class: 'nn-mini', title: 'Clear the target' }, 'clear');
      tgt.addEventListener('input', () => {
        const v = parseNum(tgt.value);
        if (v != null) { M.setNode(net(), id, { target: v }); store.touch(); }
      });
      tgt.addEventListener('change', () => {
        M.setNode(net(), id, { target: parseNum(tgt.value) });
        store.commit('target');
      });
      tgt.addEventListener('keydown', e => { if (e.key === 'Enter') tgt.blur(); });
      clr.addEventListener('click', () => {
        M.setNode(net(), id, { target: null });
        tgt.value = '';
        store.commit('clear target');
      });
      const err = texBox(c, () => {
        const y = node(id)?.target, a = fwdNode(id)?.a;
        return typeof y === 'number' && Number.isFinite(a)
          ? `\\hat y_{${I}} - y_{${I}} = ${fmt(a)} - ${factor(fmt(y))} = ${fmt(a - y)}` : null;
      }, false, 'nn-math nn-inline');
      c.binds.push(() => {
        const y = node(id)?.target;
        setVal(tgt, typeof y === 'number' ? numText(y) : '');
        clr.disabled = ro || typeof y !== 'number';
      });
      section(c, 'node.target', 'Target', null,
        h('div', { class: 'nn-target-row' }, h('span', { class: 'nn-tex-y' }, texEl(`y_{${I}}`)), tgt, clr), err);
    }

    // gradients: the chain rule with this neuron's numbers
    const grad = texBox(c, () => gradTex(id, L, I), true);
    const gradHint = h('div', { class: 'nn-hint' });
    c.binds.push(() => {
      const b = store.state.bwd;
      setHidden(gradHint, !!b && !b.note);
      setText(gradHint, b ? b.note || '' : targetsHint());
    });
    section(c, 'node.grad', 'Gradients', null, grad, gradHint);

    // outgoing: column of the next W, then skip edges. A Q / K / V neuron has no edges into the
    // attention layer: it enters through the scores or the weighted sum instead.
    if (L < last) {
      const byTo = new Map(net().edges.filter(e => e.from === id).map(e => [e.to, e]));
      const rows = [];
      if (feedsAttn) {
        const g = tp?.groups ? tp.g : -1, t = (tp?.t ?? 0) + 1;
        const dh = Math.max(1, Math.floor((tp?.d ?? 1) / headsOf(L + 1))), hs = hsup(L + 1, Math.floor((tp?.f ?? 0) / dh));
        const how = g === 0 ? `q${hs}_{${t}} \\text{ is scored against every key: } s${hs}_{${t},j} = q${hs}_{${t}} \\cdot k${hs}_{j}\\, c`
          : g === 1 ? `k${hs}_{${t}} \\text{ is scored by every query: } s${hs}_{i,${t}} = q${hs}_{i} \\cdot k${hs}_{${t}}\\, c`
          : g === 2 ? `v${hs}_{${t}} \\text{ is mixed into every } z_{i} \\text{ with weight } A${hs}_{i,${t}}`
          : '\\text{feeds the attention layer}';
        const el = h('div', { class: 'nn-math nn-inline' });
        setTex(el, how);
        rows.push(el);
      } else {
        M.nodesIn(net(), L + 1).forEach((q, r) => {
          const e = byTo.get(q.id);
          rows.push(e ? edgeRow(c, e.id, q.id, wText(L + 1, L, r + 1, I)) : maskedRow(c, id, q.id, q.id));
        });
      }
      for (let k = L + 2; k <= last; k++) {
        const all = M.nodesIn(net(), k);
        if (!all.some(q => byTo.has(q.id))) continue;
        rows.push(h('div', { class: 'nn-sub' }, texEl(`\\text{skip to ${texSafe(net().layers[k].name || 'layer ' + k)}: } W^{(${k},${L})}`)));
        all.forEach((q, r) => {
          const e = byTo.get(q.id);
          if (e) rows.push(edgeRow(c, e.id, q.id, wText(k, L, r + 1, I)));
        });
      }
      const s = section(c, 'node.out', feedsAttn ? 'Into attention' : 'Outgoing weights',
        feedsAttn ? null : `\\text{column } ${I} \\text{ of } W^{(${L + 1})}`, ...rows);
      if (!feedsAttn) hoverable(c, s.head, { kind: 'col', layer: L + 1, k: L, j: i });
    }

    buildParams(c, id);
  }

  const edgeRow = (c, eid, otherId, wt) => weightRow(c, eid, otherId, wt);

  // One weight in a node's row or column: a slider, a shared-parameter slider (moves its whole tie
  // group) or a read-only fixed value. wt names the matrix entry, for the tooltip.
  function weightRow(c, eid, otherId, wt) {
    const e = edge(eid), tie = tieOf(e);
    if (e?.fixed) return fixedRow(c, eid, otherId, `${wt}: fixed. Open this weight`);
    const n = tie ? tieGroup(tie).length : 0;
    const row = slider(c, {
      lab: el => setTex(el, label(otherId)),
      get: () => edge(eid)?.w,
      set: v => setW(eid, v),
      what: 'weight', hover: { kind: 'edge', id: eid }, other: otherId,
      onLabel: () => select({ kind: 'edge', id: eid }),
      labelTitle: tie ? `${wt} = ${tieText(tie)}, shared by ${n} edges: open this weight` : `${wt}: open this weight`,
    });
    if (tie) {
      row.classList.add('tied');
      row.title = `Shared parameter ${tieText(tie)}: moving it moves all ${n} edges`;
    }
    dimBy(c, row, E => edgeEm(E, eid));
    return row;
  }

  // A fixed edge (a residual identity, a pooling weight): shown, never edited.
  function fixedRow(c, eid, otherId, tip) {
    const sw = h('span', { class: 'nn-sw' });
    const name = h('span', { class: 'nn-sl-lab link', title: tip });
    const val = h('span', { class: 'nn-fixed-val' });
    const row = h('div', { class: 'nn-sl fixed', title: 'Fixed weight: training, Randomize and connect never change it' },
      sw, name, h('span', { class: 'nn-masked-txt', text: 'fixed' }), val);
    hoverable(c, row, { kind: 'edge', id: eid }, otherId);
    dimBy(c, row, E => edgeEm(E, eid));
    if (!ro) name.addEventListener('click', () => select({ kind: 'edge', id: eid }));
    c.binds.push(() => {
      setTex(name, label(otherId));
      const w = edge(eid)?.w;
      setText(val, numText(w));
      setStyle(sw, 'background', Number.isFinite(w) ? colorFor(w, maxW, theme()) : 'transparent');
    });
    return row;
  }

  // ---------------------------------------------------------------- attention neuron

  // a b, juxtaposed when b is a bracketed negative: (-0.12)(-0.19), 0.31(-0.24), but 0.31 · 0.24
  const prod = (a, b) => { const fa = factor(a), fb = factor(b); return fb.startsWith('(') ? `${fa}${fb}` : `${fa} \\cdot ${fb}`; };
  const cutTerms = xs => (xs.length > MAX_TERMS ? [...xs.slice(0, 4), '\\cdots', ...xs.slice(-2)] : xs);

  // Scores s_ij = q_i . k_j c for this neuron's token i, then its attention row A_i as bars.
  function buildAttnNode(c, id, L, k) {
    const at = attnAt(L, k), { T, i, hd } = at;
    const hs = hsup(L, hd);
    const scores = texBox(c, () => {
      const F = attnFwd(L, hd);
      if (!F?.Q || !F?.K) return null;
      const lhsW = texWidth(`s${hs}_{${i + 1},${T}}`);
      const room = CARD_UNITS - lhsW;
      const lines = [`s${hs}_{${i + 1},j} &= c\\; q${hs}_{${i + 1}} \\cdot k${hs}_{j}, \\quad c = ${scaleEq(L)}`];
      const q = F.Q[i] || [];
      for (let j = 0; j < T; j++) {
        const sv = F.S?.[i]?.[j];
        const lhs = `s${hs}_{${i + 1},${j + 1}} &= `;
        if ((causalOf(L) && j > i) || sv === -Infinity) { lines.push(`${lhs}-\\infty \\quad \\text{masked: } ${j + 1} > ${i + 1}`); continue; }
        const kj = F.K[j] || [];
        let dot = 0;
        const terms = q.map((qc, cc) => { dot += qc * (kj[cc] ?? NaN); return prod(fmt(qc), fmt(kj[cc])); });
        const val = Number.isFinite(sv) ? sv : dot * scaleOf(L);
        const shown = cutTerms(terms);
        const body = `c\\,\\big(${shown.join(' + ')}\\big)`, res = `= ${fmt(val)}`;
        if (texWidth(`${body} ${res}`) <= room) lines.push(`${lhs}${body} ${res}`);
        else if (texWidth(body) <= room) lines.push(lhs + body, `&${res}`);
        else {
          const w = wrapSum(`${lhs}c\\,\\big(`, shown, room);
          w[w.length - 1] += '\\big)';
          lines.push(...w, `&${res}`);
        }
      }
      return aligned(lines);
    }, true);
    const hint = h('div', { class: 'nn-hint' });
    c.binds.push(() => {
      const ok = !!attnFwd(L, hd);
      setHidden(hint, ok);
      setText(hint, ok ? '' : 'No attention values: the forward pass did not return them.');
    });
    section(c, 'node.scores', 'Scores', `q${hs}_{${i + 1}} \\cdot k${hs}_{j}`, scores, hint);

    // the attention row: A_ij = softmax_j(s_ij), one bar per key token j, hover lights v_j
    const def = h('div', { class: 'nn-def' });
    setTex(def, `A${hs}_{${i + 1},j} = \\dfrac{e^{s${hs}_{${i + 1},j}}}{\\sum_{j'} e^{s${hs}_{${i + 1},j'}}}` +
      (causalOf(L) ? `,\\quad A${hs}_{${i + 1},j} = 0 \\text{ for } j > ${i + 1}` : ''));
    const bars = [];
    for (let j = 0; j < T; j++) {
      const lab = h('span', { class: 'nn-att-lab' });
      const tex = h('span');
      setTex(tex, `A${hs}_{${i + 1},${j + 1}}`);
      const name = h('span', { class: 'nn-att-name' });   // key token j's name, when the tokens have names
      lab.append(tex, name);
      const fill = h('i');
      const val = h('span', { class: 'nn-att-val' });
      const row = h('div', { class: 'nn-att-row' }, lab, h('span', { class: 'nn-att-bar' }, fill), val);
      const v = qkvNode(L, 2, j, at.f);
      if (v) hoverable(c, row, { kind: 'node', id: v.id }, v.id);
      tokenLit(c, row, [{ layer: L - 1, t: j, g: 2, h: at.heads > 1 ? hd : null }]);   // value token j (V is group 2)
      dimBy(c, row, E => (causalOf(L) && j > i ? null : attnEm(E, L, i, j, hd)));
      bars.push({ row, fill, val, name, j });
    }
    c.binds.push(() => {
      const A = attnFwd(L, hd)?.A?.[i];
      for (const b of bars) {
        const a = A?.[b.j];
        const masked = causalOf(L) && b.j > i;
        b.row.classList.toggle('masked', masked);
        setStyle(b.fill, 'width', Number.isFinite(a) ? `${Math.max(0, Math.min(1, a)) * 100}%` : '0%');
        setText(b.val, masked ? '0 (masked)' : Number.isFinite(a) ? fmt(a) : '?');
        setText(b.name, tokName(b.j) || '');
        setHidden(b.name, !tokName(b.j));
        const title = `${cap(tokWord(i))} attends to ${tokWord(b.j)} with this weight (hover: v${b.j + 1})`;
        if (b.row.title !== title) b.row.title = title;
      }
    });
    const attendsTex = () => `${tokNameTex(i) || `\\text{token } ${i + 1}`} \\text{ attends to}`;
    const rowSec = section(c, 'node.attn', 'Attention row', attendsTex(), def, ...bars.map(b => b.row));
    const rowSub = rowSec.head.querySelector('.nn-sec-sub');
    if (rowSub) c.binds.push(() => setTex(rowSub, attendsTex()));
    const hh = at.heads > 1 ? hd : null;   // lit by this token's row: its Z token, its query, a heatmap row
    tokenLit(c, rowSec.head, [{ layer: L, t: i, h: hh }, { layer: L - 1, t: i, g: 0, h: hh }]);
  }

  // z_if = sum_j A_ij V_jf with numbers (a = z: the layer is identity).
  function attnArith(id, L, I, k) {
    const { T, i, hd, fh, f } = attnAt(L, k);
    const F = attnFwd(L, hd), fw = fwdNode(id);
    const hs = hsup(L, hd);
    const zS = `z^{(${L})}_{${I}}`, aS = aSym(L, I);
    const room = CARD_UNITS - Math.max(texWidth(zS), texWidth(aS));
    const lines = [`${zS} &= Z_{${i + 1},${f + 1}} = \\sum_{j=1}^{${T}} A${hs}_{${i + 1},j}\\, V${hs}_{j,${fh + 1}}`];
    if (F?.A && F?.V) {
      const terms = [];
      for (let j = 0; j < T; j++) terms.push(prod(fmt(F.A[i]?.[j]), fmt(F.V[j]?.[fh])));
      lines.push(...wrapSum('&= ', cutTerms(terms), room));
    }
    lines.push(`&= ${fmt(fw?.z)}`, `${aS} &= ${zS} = ${fmt(fw?.a)}`);
    return aligned(lines);
  }

  function texEl(src) {
    const el = h('span');
    setTex(el, src);
    return el;
  }

  function targetsHint() {
    const last = lastIndex();
    if (last < 1) return 'Add a layer after the inputs to get gradients.';
    const outs = M.nodesIn(net(), last);
    const n = outs.filter(q => typeof q.target === 'number').length;
    return `Gradients need a target on every output (${n} of ${outs.length} set).`;
  }

  function arithmetic(id, L, I, incoming) {
    const n = node(id);
    const fw = fwdNode(id);
    if (!n) return null;
    if (L === 0) {
      return `a^{(0)}_{${I}} = {${n.label || 'x'}} = ${fmt(n.value)}`;
    }
    // z^{(L)}_I = sum_j W^{(L)}_{I,j} a^{(L-1)}_j (+ skip terms) + b^{(L)}_I: existing edges only,
    // since a masked entry is a fixed 0. Factors are the senders' labels.
    // a shared weight is written by its parameter's name: q = x W_Q reads W_Q(i, j) x_i
    const terms = incoming.map(({ eid, from, k, j }) => {
      const tie = tieOf(edge(eid));
      return {
        sym: `${tie ? tieTex(tie) : wSym(L, k, I, j + 1)}\\,{${label(from)}}`,
        w: edge(eid)?.w, a: fwdNode(from)?.a,
      };
    });
    const shown = terms.length > MAX_TERMS ? [...terms.slice(0, 4), null, ...terms.slice(-2)] : terms;
    const num = shown.map(t => (t ? wTerm(t.w, fmt(t.a)) : '\\cdots'));
    const b = Number.isFinite(n.bias) ? n.bias : 0;
    const z = fw?.z, a = fw?.a;
    const zS = `z^{(${L})}_{${I}}`, aS = aSym(L, I);
    const room = CARD_UNITS - Math.max(texWidth(zS), texWidth(aS));
    const lines = [
      ...wrapSum(`${zS} &= `, [...shown.map(t => (t ? t.sym : '\\cdots')), `b^{(${L})}_{${I}}`], room),
      ...wrapSum('&= ', [...num, b < 0 ? `- ${fmt(-b)}` : fmt(b)], room),
      `&= ${fmt(z)}`,
    ];
    const act = actOf(L);
    const fn = FN[act] ?? `\\operatorname{${act}}`;
    if (act === 'identity') lines.push(`${aS} &= ${zS} = ${fmt(a)}`);
    else if (isVec(act)) {
      lines.push(`${aS} &= ${fn || `\\operatorname{${act}}`}(z^{(${L})})_{${I}} = \\frac{e^{${zS}}}{\\sum_j e^{z^{(${L})}_{j}}} = ${fmt(a)}`);
    } else lines.push(`${aS} &= ${fn}(${zS}) = ${fn}(${fmt(z)}) = ${fmt(a)}`);
    return aligned(lines);
  }

  // The chain rule for one neuron, with its numbers: dL/da (from the loss on an output, from the
  // next layers' deltas otherwise), the activation's slope, delta = dL/dz, and dL/db = delta.
  // Each quantity is a symbolic line, then its numbers (the card is narrow).
  function gradTex(id, L, I) {
    const b = store.state.bwd, g = bwdNode(id), fw = fwdNode(id);
    if (!b || !g) return null;
    const last = lastIndex(), act = actOf(L), aS = aSym(L, I), zS = `z^{(${L})}_{${I}}`;
    const dS = `\\delta^{(${L})}_{${I}}`, dA = `\\frac{\\partial L}{\\partial ${aS}}`;
    const dZ = `\\frac{\\partial L}{\\partial ${zS}}`;
    const lines = [];
    let fused = false;   // the output's delta came straight from the loss (cross-entropy)
    if (L === last && L > 0) {
      const outs = M.nodesIn(net(), L), n = outs.length, kind = lossKind();
      const y = node(id)?.target, a = fw?.a, yh = `\\hat y_{${I}}`, yy = `y_{${I}}`;
      const fr = n > 1 ? `\\tfrac{1}{${n}}` : '';
      // a token output layer normalizes each token: its cross-entropy is the mean over the m tokens
      const [b0, b1] = softBlock(L, I - 1), m = kind === 'ce' ? Math.max(1, Math.round(n / (b1 - b0))) : 1;
      const fm = m > 1 ? `\\tfrac{1}{${m}}` : '';
      lines.push(`L &= ${m > 1 ? `\\tfrac{1}{${m}}\\sum_{\\text{tokens}}\\Big(${lossTex(kind, b1 - b0)}\\Big)` : lossTex(kind, n)}`, `&= ${fmt(b.loss, 4)}`);
      if (kind === 'ce') {
        // softmax + cross-entropy: delta = (sum_j y_j) yhat - y, i.e. yhat - y for a one-hot target
        const S = outs.slice(b0, b1).reduce((s, q) => s + (Number.isFinite(q.target) ? q.target : 0), 0);
        const one = Math.abs(S - 1) < 1e-9;
        lines.push(`${dA} &= -${fm}\\frac{${yy}}{${yh}} = -${fm}\\frac{${fmt(y)}}{${fmt(a)}} = ${g3(g.da)}`);
        lines.push(one ? `${dS} &= ${dZ} = ${fm}(${yh} - ${yy})` : `${dS} &= ${dZ} = ${fm}\\big(${yh}\\textstyle\\sum_j y_j - ${yy}\\big)`);
        lines.push(one ? `&= ${fm}(${fmt(a)} - ${factor(fmt(y))}) = ${g3(g.dz)}`
          : `&= ${fm}\\big(${fmt(a)} \\cdot ${factor(fmt(S))} - ${factor(fmt(y))}\\big) = ${g3(g.dz)}`);
        fused = true;
      } else if (kind === 'bce') {
        // sigmoid + binary cross-entropy: sigma'(z) = yhat (1 - yhat) cancels, delta = (yhat - y) / n
        lines.push(`${dA} &= ${fr}\\frac{${yh} - ${yy}}{${yh}\\,(1 - ${yh})} = ${g3(g.da)}`);
        lines.push(`${dS} &= ${dZ} = ${dA}\\,\\sigma'(${zS})`);
        lines.push(`&= ${fr}(${yh} - ${yy})`);
        lines.push(`&= ${fr}(${fmt(a)} - ${factor(fmt(y))}) = ${g3(g.dz)}`);
        fused = true;
      } else {
        lines.push(`${dA} &= ${fr}(${yh} - ${yy})`, `&= ${fr}(${fmt(a)} - ${factor(fmt(y))}) = ${g3(g.da)}`);
      }
    } else if (L < last && isAttn(L + 1)) {
      // a Q / K / V neuron: dQ = dS K c, dK = dS^T Q c, dV = A^T dZ (docs/NN_ATTENTION.md)
      lines.push(...attnGradLines(id, L, aS, dA));
      lines.push(`&= ${g3(g.da)}`);
    } else {
      // dL/da = sum over this neuron's outgoing edges of W delta (column I of the next W, plus skips)
      const out = net().edges.filter(e => e.from === id)
        .map(e => ({ e, m: M.nodeLayerIndex(net(), e.to), r: indexIn(e.to) }))
        .filter(t => t.m > L)
        .sort((p, q) => p.m - q.m || p.r - q.r);
      const ms = [...new Set(out.map(t => t.m))];
      const sym = ms.length ? ms.map(m => `\\sum_r W^{(${wUp(m, L)})}_{r,${I}}\\,\\delta^{(${m})}_r`).join(' + ') : '0';
      lines.push(`${dA} &= ${sym}`);
      if (out.length) {
        const shown = out.length > MAX_TERMS ? [...out.slice(0, 4), null, ...out.slice(-2)] : out;
        // the block's widest left-hand side: dL/da's denominator, or the activation's slope
        const fn = FN[act] ?? `\\operatorname{${act}}`;
        const lhs = Math.max(texWidth(`\\partial ${aS}`), L > 0 && !isVec(act) && act !== 'identity' ? texWidth(`${fn}'(${zS})`) : 0);
        lines.push(...wrapSum('&= ', shown.map(t => (t ? wTerm(t.e.w, g3(bwdNode(t.e.to)?.dz)) : '\\cdots')), CARD_UNITS - lhs));
      }
      lines.push(`&= ${g3(g.da)}`);
    }
    if (L > 0 && !fused) {
      const z = fw?.z, a = fw?.a;
      if (act === 'identity') lines.push(`${dS} &= ${dZ} = ${dA} = ${g3(g.dz)}`);
      else if (isVec(act)) {
        // softmax couples the layer: delta_i = a_i (dL/da_i - sum_j a_j dL/da_j)
        const aj = `a^{(${L})}_{j}`, dAj = `\\frac{\\partial L}{\\partial ${aj}}`;
        const [b0, b1] = softBlock(L, I - 1);   // per token on a token layer
        const as = (store.state.fwd?.a?.[L] || []).slice(b0, b1), das = (b.dA?.[L] || []).slice(b0, b1);
        const s = as.reduce((t, v, j) => t + v * (das[j] ?? 0), 0);
        const over = isTok(L) ? `_{j \\in \\text{token}}` : '_j';
        lines.push(`${dS} &= ${dZ} = ${aS}\\Big(${dA} - \\sum${over} ${aj}\\,${dAj}\\Big)`);
        lines.push(`&= ${fmt(a)}\\,\\big(${g3(g.da)} - ${factor(g3(s))}\\big) = ${g3(g.dz)}`);
      } else {
        const fn = FN[act] ?? `\\operatorname{${act}}`;
        const sp = M.ACTS?.[act]?.df ? M.ACTS[act].df(z, a) : NaN;
        const kink = act === 'relu' || act === 'leaky';   // piecewise slope: exact, no rounding
        const spT = kink && Number.isFinite(sp) ? String(sp) : g3(sp);
        lines.push(...slopeLines(act, `${fn}'(${zS})`, aS, zS, z, a, spT));
        lines.push(`${dS} &= ${dZ} = ${dA}\\,${fn}'(${zS})`);
        lines.push(`&= ${factor(g3(g.da))} \\cdot ${factor(spT)} = ${g3(g.dz)}`);
      }
    }
    if (L > 0 && !isAttn(L)) {
      lines.push(`\\frac{\\partial L}{\\partial b^{(${L})}_{${I}}} &= ${dS} = ${g3(g.dz)}`);
      const bt = biasTie(id);   // a shared bias steps by the sum over every neuron that uses it
      const tot = bt ? b.tie?.[bt] : null;
      if (bt && Number.isFinite(tot)) {
        lines.push(`\\frac{\\partial L}{\\partial ${tieTex(bt)}} &= \\sum_{\\text{copies}} \\delta = ${g3(tot)}\\quad (${biasGroup(bt).length} \\text{ copies})`);
      }
    }
    return aligned(lines);
  }

  // dL/da for a neuron of the Q / K / V layer L, through attention layer L + 1, with numbers:
  //   q_{t,c}: c sum_j dS_tj K_jc      k_{t,c}: c sum_i dS_it Q_ic      v_{t,c}: sum_i A_it dZ_ic
  function attnGradLines(id, L, aS, dA) {
    const k = indexIn(id), s = shapeOf(L), A = L + 1;
    const g = s.groups ? Math.floor(k / (s.tokens * s.d)) : 2;
    const t = Math.floor(k / s.d) % s.tokens, f = k % s.d;
    const dh = Math.max(1, Math.floor(s.d / headsOf(A))), hd = Math.min(headsOf(A) - 1, Math.floor(f / dh)), ch = f % dh;
    const F = attnFwd(A, hd), B = attnBwd(A, hd), hs = hsup(A, hd);
    const T = s.tokens, room = CARD_UNITS - texWidth(`\\partial ${aS}`);
    const dS = (p, q) => `\\mathrm{d}S${hs}_{${p},${q}}`;
    let sym, terms = [], pre = '', post = '';
    if (g === 0) {
      sym = `(\\mathrm{d}Q${hs})_{${t + 1},${ch + 1}} = c \\sum_j ${dS(t + 1, 'j')}\\, K${hs}_{j,${ch + 1}}`;
      if (F && B) for (let j = 0; j < T; j++) terms.push(prod(g3(B.dS?.[t]?.[j]), fmt(F.K?.[j]?.[ch])));
      pre = 'c\\,\\big('; post = '\\big)';
    } else if (g === 1) {
      sym = `(\\mathrm{d}K${hs})_{${t + 1},${ch + 1}} = c \\sum_i ${dS('i', t + 1)}\\, Q${hs}_{i,${ch + 1}}`;
      if (F && B) for (let i = 0; i < T; i++) terms.push(prod(g3(B.dS?.[i]?.[t]), fmt(F.Q?.[i]?.[ch])));
      pre = 'c\\,\\big('; post = '\\big)';
    } else {
      sym = `(\\mathrm{d}V${hs})_{${t + 1},${ch + 1}} = \\sum_i A${hs}_{i,${t + 1}}\\, \\mathrm{d}Z${hs}_{i,${ch + 1}}`;
      if (F && B) for (let i = 0; i < T; i++) terms.push(prod(fmt(F.A?.[i]?.[t]), g3(B.dZ?.[i]?.[ch])));
    }
    const lines = [`${dA} &= ${sym}`];
    if (pre) lines.push(`&\\quad c = ${scaleEq(A)}`);
    if (terms.length) {
      const shown = cutTerms(terms);
      const w = wrapSum(`&= ${pre}`, shown, room);
      w[w.length - 1] += post;
      lines.push(...w);
    } else {
      lines.push('&\\quad \\text{(no attention gradients from the backward pass)}');
    }
    return lines;
  }

  // The slope of an elementwise activation at this neuron, in terms students can check. ReLU and
  // leaky ReLU use the model's convention at the kink: z = 0 takes the left-hand slope.
  function slopeLines(act, lhs, aS, zS, z, a, spT) {
    const pos = z > 0, side = `\\quad (${zS} ${pos ? '>' : '\\le'} 0)`;
    if (act === 'sigmoid') return [`${lhs} &= ${aS}\\,(1 - ${aS})`, `&= ${fmt(a)}\\,(1 - ${fmt(a)}) = ${spT}`];
    if (act === 'tanh') return [`${lhs} &= 1 - (${aS})^2`, `&= 1 - ${factor(fmt(a))}^2 = ${spT}`];
    if (act === 'relu' || act === 'leaky') return [`${lhs} &= ${spT}${side}`];
    return [`${lhs} &= ${spT}`];
  }

  // ---------------------------------------------------------------- params

  function buildParams(c, id) {
    const list = h('div', { class: 'nn-params' });
    const empty = h('div', { class: 'nn-hint', text: 'Stash anything on this neuron: a note, a unit, a role. Numbers stay numbers; $\\tex$ renders as maths.' });
    const add = h('button', { class: 'nn-mini nn-add', title: 'Add a key/value pair' }, '+ add');
    let shown = null;

    const current = () => node(id)?.params || {};
    const rows = () => [...list.children];

    function save() {
      const obj = {};
      const seen = new Map();
      let bad = false;
      for (const r of rows()) {
        const k = r._key.value.trim();
        r.classList.remove('dup');
        if (!k) continue;
        if (seen.has(k)) { r.classList.add('dup'); seen.get(k).classList.add('dup'); bad = true; continue; }
        seen.set(k, r);
        const raw = r._val.value;
        const n = parseNum(raw);
        obj[k] = n != null ? n : raw;
      }
      if (bad) return false;
      const next = JSON.stringify(obj);
      shown = next;
      if (next === JSON.stringify(current())) return true;
      M.setNode(net(), id, { params: obj });
      store.commit('param');
      return true;
    }

    function preview(r) {
      const v = r._val.value.trim();
      const m = /^\$(.+)\$$/s.exec(v);
      const html = m ? texString('\\displaystyle ' + m[1]) : null;
      r._pre.hidden = !html;
      if (html && r._pre._src !== m[1]) { r._pre._src = m[1]; r._pre.innerHTML = html; }
      r._val.classList.toggle('num', parseNum(v) != null);
    }

    function row(k = '', v = '') {
      const key = h('input', { type: 'text', class: 'nn-text nn-pk', placeholder: 'key', spellcheck: 'false', autocomplete: 'off', value: k });
      const val = h('input', { type: 'text', class: 'nn-text nn-pv', placeholder: 'value', spellcheck: 'false', autocomplete: 'off', value: String(v) });
      const del = h('button', { class: 'nn-insp-btn nn-pdel', title: 'Remove' }, '×');
      const pre = h('div', { class: 'nn-ppre', hidden: true });
      const r = h('div', { class: 'nn-prow' }, key, val, del, pre);
      Object.assign(r, { _key: key, _val: val, _pre: pre });
      let keyBefore = k, valBefore = String(v);
      key.addEventListener('focus', () => { keyBefore = key.value; });
      val.addEventListener('focus', () => { valBefore = val.value; });
      key.addEventListener('change', save);
      val.addEventListener('change', save);
      val.addEventListener('input', () => preview(r));
      key.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); val.focus(); val.select(); }
        else if (e.key === 'Escape') { key.value = keyBefore; key.blur(); }
        else if (e.key === 'Backspace' && !key.value && !val.value) {
          e.preventDefault();
          const prev = r.previousElementSibling;
          r.remove();
          save();
          syncEmpty();
          prev?._val.focus();
        }
      });
      val.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          if (!save()) { key.focus(); key.select(); return; }
          const next = r.nextElementSibling;
          if (next) next._key.focus();
          else if (key.value.trim()) newRow();
          else val.blur();
        } else if (e.key === 'Escape') { val.value = valBefore; preview(r); val.blur(); }
      });
      del.addEventListener('click', () => { r.remove(); save(); syncEmpty(); });
      preview(r);
      return r;
    }

    function newRow() {
      const r = row();
      list.append(r);
      syncEmpty();
      r._key.focus();
    }

    function render() {
      const p = current();
      shown = JSON.stringify(p);
      list.replaceChildren(...Object.entries(p).map(([k, v]) => row(k, v)));
      syncEmpty();
    }

    function syncEmpty() { empty.hidden = list.children.length > 0; }

    add.addEventListener('click', newRow);
    render();
    c.binds.push(() => {
      if (JSON.stringify(current()) !== shown && !list.contains(document.activeElement)) render();
    });
    const s = section(c, 'node.params', 'Params', null, list, empty, h('div', { class: 'nn-row' }, add));
    s.sec.classList.add('nn-params-sec');
  }

  // ---------------------------------------------------------------- edge card

  function buildEdge(c, eid) {
    const e0 = edge(eid);
    const from = e0.from, to = e0.to;
    const tie = tieOf(e0), fixed = !!e0.fixed;
    const lt = M.nodeLayerIndex(net(), to), lf = M.nodeLayerIndex(net(), from);
    const i = indexIn(to), j = indexIn(from);
    const Wsym = wSym(lt, lf, i + 1, j + 1);
    const zS = `z^{(${lt})}_{${i + 1}}`;
    c.binds.push(() => {
      setTex(c.title, `{${label(from)}} \\to {${label(to)}}`);
      setText(c.kind, fixed ? 'fixed weight' : tie ? 'shared weight' : 'weight');
    });

    const fromChip = h('button', { class: 'nn-chip', title: 'Open the source neuron' });
    const toChip = h('button', { class: 'nn-chip', title: 'Open the target neuron' });
    fromChip.addEventListener('click', () => select({ kind: 'node', id: from }));
    toChip.addEventListener('click', () => select({ kind: 'node', id: to }));
    hoverable(c, fromChip, { kind: 'node', id: from });
    hoverable(c, toChip, { kind: 'node', id: to });
    c.binds.push(() => { setTex(fromChip, label(from)); setTex(toChip, label(to)); });
    const whereEl = h('div', { class: 'nn-where' });
    setTex(whereEl, `w = ${Wsym}` + `\\quad \\text{row } ${i + 1}\\ (\\text{to}),\\ \\text{column } ${j + 1}\\ (\\text{from})` +
      (lf === lt - 1 ? '' : `\\quad \\text{skip}`));
    const kids = [h('div', { class: 'nn-where-row' }, fromChip, h('span', { class: 'nn-arrow', text: '→' }), toChip), whereEl];
    if (tie) {
      // "shared parameter W_Q(1,2), used by n edges", then one chip per edge (per token)
      const share = h('div', { class: 'nn-where nn-share' });
      c.binds.push(() => setTex(share, `\\text{shared parameter } ${tieTex(tie)},\\ \\text{used by } ${tieGroup(tie).length} \\text{ edges}`));
      kids.push(share, usedBy(c, eid, tie));
    } else if (fixed) {
      kids.push(h('div', { class: 'nn-hint', text: 'Fixed: a residual identity or pooling weight. Training, Randomize and connect never change it.' }));
    }
    section(c, 'edge.where', 'Edge', null, ...kids);

    if (fixed) {
      section(c, 'edge.w', 'Weight', '\\text{read-only}',
        texBox(c, () => `w = ${wTex(edge(eid)?.w)}`, false, 'nn-math nn-inline'));
    } else {
      const row = slider(c, {
        lab: el => setTex(el, tie ? tieTex(tie) : 'w'),
        get: () => edge(eid)?.w,
        set: v => setW(eid, v),
        what: 'weight', hover: { kind: 'edge', id: eid },
      });
      if (tie) row.classList.add('tied', 'wide');
      section(c, 'edge.w', tie ? 'Shared parameter' : 'Weight',
        tie ? `\\text{moves all } ${tieGroup(tie).length} \\text{ edges}` : null, row);
    }

    section(c, 'edge.contrib', 'Contribution', null, texBox(c, () => {
      const w = edge(eid)?.w, a = fwdNode(from)?.a, z = fwdNode(to)?.z;
      return aligned([
        `w\\,{${label(from)}} &= ${wTex(w)} \\cdot ${factor(fmt(a))} = ${fmt(w * a)}`,
        `&\\text{one term of } ${zS} = ${fmt(z)} \\text{ at } {${label(to)}}`,
      ]);
    }, true));

    // dL/dw by the chain rule (z depends on w through the term w a), then one descent step. A shared
    // parameter sums the terms of every edge that uses it; a fixed weight is never stepped.
    const grad = texBox(c, () => {
      if (tie) return tieGradTex(tie);
      const dw = store.state.bwd?.edge?.[eid];
      if (!Number.isFinite(dw)) return null;
      const dz = bwdNode(to)?.dz, a = fwdNode(from)?.a, w = edge(eid)?.w;
      const lines = [
        `\\frac{\\partial L}{\\partial w} &= \\frac{\\partial L}{\\partial ${zS}}\\,\\frac{\\partial ${zS}}{\\partial w} = \\delta^{(${lt})}_{${i + 1}}\\, ${aSym(lf, j + 1)}`,
        `&= ${factor(g3(dz))} \\cdot ${factor(fmt(a))} = ${g3(dw)}`,
      ];
      const lr = learningRate();
      if (Number.isFinite(w) && !fixed) {
        lines.push(`w_{\\text{new}} &= w - \\eta\\,\\frac{\\partial L}{\\partial w}`);
        lines.push(`&= ${fmt(w, 3)} - ${+lr.toPrecision(3)} \\cdot ${factor(g3(dw))} = ${fmt(w - lr * dw, 3)}`);
      }
      return aligned(lines);
    }, true);
    const hint = h('div', { class: 'nn-hint' });
    c.binds.push(() => {
      const b = store.state.bwd, eta = +learningRate().toPrecision(3);
      setText(hint, !b ? targetsHint() : b.note
        || (fixed ? 'Fixed: training computes this gradient but never applies it.'
          : tie ? `One term per edge that uses ${tieText(tie)}: training adds them up, so every copy takes the same step (η = ${eta}; training also averages over a mini-batch, this is the current sample alone).`
            : `η = ${eta}, the Train panel's rate. Training averages ∂L/∂w over a mini-batch; this step uses the current sample alone.`));
    });
    section(c, 'edge.grad', 'Gradient', null, grad, hint);

    if (!fixed) {
      const del = h('button', {
        class: 'nn-mini danger',
        title: tie ? 'Remove this one edge (its entry becomes a fixed 0; the shared parameter keeps its other edges)' : 'Remove this edge (its entry becomes a fixed 0)',
      }, 'remove edge');
      del.addEventListener('click', () => { M.disconnect(net(), eid); store.commit('disconnect'); });
      c.body.append(h('div', { class: 'nn-row nn-actions' }, del));
    }
  }

  // The edges of a tie group as chips: by token when they all land in one token layer, else from -> to.
  function usedBy(c, eid, tie) {
    const es = tieGroup(tie);
    const ls = new Set(es.map(e => M.nodeLayerIndex(net(), e.to)));
    const tokL = ls.size === 1 ? [...ls][0] : -1;
    const byTok = tokL > 0 && isTok(tokL);
    const wrap = h('div', { class: 'nn-used' });
    for (const e of es) {
      const chip = h('button', { class: 'nn-chip nn-used-chip' + (e.id === eid ? ' on' : ''), title: 'Open this edge' });
      const t = byTok ? tokPos(tokL, indexIn(e.to)).t : -1;
      c.binds.push(() => setTex(chip, (byTok ? `${tokNameTex(t) || `t_{${t + 1}}`}\\!:\\ ` : '') + `{${label(e.from)}} \\to {${label(e.to)}}`));
      chip.addEventListener('click', () => select({ kind: 'edge', id: e.id }));
      hoverable(c, chip, { kind: 'edge', id: e.id });
      if (e.id !== eid) dimBy(c, chip, E => edgeEm(E, e.id));
      wrap.append(chip);
    }
    return wrap;
  }

  // dL/dW(i, j) of a shared parameter = the sum over its edges of delta_to a_from (one per token).
  function tieGradTex(tie) {
    const b = store.state.bwd;
    if (!b) return null;
    const es = tieGroup(tie);
    const P = tieTex(tie), dP = `\\frac{\\partial L}{\\partial ${P}}`;
    const room = CARD_UNITS - Math.max(texWidth(`\\partial ${P}`), texWidth('w_{new}'));
    const syms = [], nums = [];
    let sum = 0;
    for (const e of es) {
      const lt = M.nodeLayerIndex(net(), e.to), lf = M.nodeLayerIndex(net(), e.from);
      const dz = bwdNode(e.to)?.dz, a = fwdNode(e.from)?.a, ge = b.edge?.[e.id];
      syms.push(`\\delta^{(${lt})}_{${indexIn(e.to) + 1}}\\, ${aSym(lf, indexIn(e.from) + 1)}`);
      nums.push(prod(g3(dz), fmt(a)));
      sum += Number.isFinite(ge) ? ge : dz * a;
    }
    const total = Number.isFinite(b.tie?.[tie]) ? b.tie[tie] : sum;
    const cut = xs => (xs.length > MAX_TERMS ? [...xs.slice(0, 4), '\\cdots', ...xs.slice(-2)] : xs);
    const lines = [`${dP} &= \\sum_{e} \\delta_{\\text{to}(e)}\\, a_{\\text{from}(e)}`,
      ...wrapSum('&= ', cut(syms), room), ...wrapSum('&= ', cut(nums), room), `&= ${g3(total)}`];
    const w = es[0]?.w, lr = learningRate();
    if (Number.isFinite(w) && Number.isFinite(total)) {
      lines.push(`w_{\\text{new}} &= w - \\eta\\, ${dP}`);
      lines.push(`&= ${fmt(w, 3)} - ${+lr.toPrecision(3)} \\cdot ${factor(g3(total))} = ${fmt(w - lr * total, 3)}`);
    }
    return aligned(lines);
  }

  // ---------------------------------------------------------------- layer card

  function buildLayer(c, lid) {
    const L = M.layerIndex(net(), lid);
    const last = lastIndex();
    const attn = isAttn(L), tok = isTok(L), sh = shapeOf(L);
    const kind = attn ? (L === last ? 'attention output layer' : 'attention layer')
      : L === 0 ? 'input layer' : L === last ? 'output layer' : 'hidden layer';
    const nodes = () => M.nodesIn(net(), L);
    c.binds.push(() => {
      setText(c.title, layerOf(lid)?.name || `layer ${L}`);
      setText(c.kind, `${kind} · ${L}`);
    });

    const name = textInput(c, {
      get: () => layerOf(lid)?.name,
      set: v => M.setLayer(net(), lid, { name: v }),
      what: 'layer name', placeholder: 'name',
    });
    const size = h('span', { class: 'nn-size' });
    const minus = h('button', { class: 'nn-mini', title: 'Remove the last neuron' }, '−');
    const plus = h('button', { class: 'nn-mini', title: 'Add a neuron, wired like its neighbours' }, '+');
    minus.addEventListener('click', () => resize(lid, -1));
    plus.addEventListener('click', () => resize(lid, +1));
    // A token layer holds tokens x features neurons (per group): one more neuron would break that.
    const sizeRow = tok
      ? h('span', { class: 'nn-size-row', title: 'tokens × features' + (sh.groups ? ' per group' : '') }, size)
      : h('span', { class: 'nn-size-row' }, minus, size, plus);
    c.binds.push(() => {
      const n = nodes().length, s = shapeOf(L);
      setText(size, tok ? `${s.groups ? s.groups.length + ' × ' : ''}${s.tokens} × ${s.d} = ${n}` : String(n));
      minus.disabled = ro || n <= 1;
    });
    const shape = texBox(c, () => {
      const m = nodes().length, s = shapeOf(L);
      if (attn) {
        return `Z = \\operatorname{softmax}\\big(QK^{\\top} c${causalOf(L) ? ' + M' : ''}\\big)\\,V \\in \\mathbb{R}^{${s.tokens} \\times ${s.d}}`;
      }
      const tokTex = tok ? (s.groups ? s.groups.map(g => `{${g}}`).join(',\\ ') : matSym(L)) + ` \\in \\mathbb{R}^{${s.tokens} \\times ${s.d}}` : '';
      // named tokens: the rows, in order (docs/NN_LENS.md)
      const named = tok && s.tokens > 1 && Array.from({ length: s.tokens }, (_, t) => tokName(t)).some(Boolean)
        ? `\\text{rows: } ${Array.from({ length: s.tokens }, (_, t) => tokNameTex(t) || `t_{${t + 1}}`).join(',\\ ')}` : '';
      if (L === 0) {
        const top = tok ? `${tokTex}\\quad (x = a^{(0)} \\in \\mathbb{R}^{${m}})` : `x = a^{(0)} \\in \\mathbb{R}^{${m}}`;
        return named ? `\\begin{aligned} &${top} \\\\ &${named} \\end{aligned}` : top;
      }
      const n = M.nodesIn(net(), L - 1).length;
      const ids = new Set(nodes().map(q => q.id));
      const prev = new Set(M.nodesIn(net(), L - 1).map(q => q.id));
      let have = 0, skip = 0;
      for (const e of net().edges) {
        if (!ids.has(e.to)) continue;
        if (prev.has(e.from)) have++; else skip++;
      }
      const masked = m * n - have;
      const flat = `W^{(${L})} \\in \\mathbb{R}^{${m} \\times ${n}},\\ b^{(${L})} \\in \\mathbb{R}^{${m}}` +
        (masked ? `\\quad ${masked} \\text{ masked}` : '') + (skip ? `\\quad ${skip} \\text{ skip}` : '');
      return tok ? `\\begin{aligned} &${tokTex} \\\\ &${flat} \\end{aligned}` : flat;
    }, false, 'nn-math nn-inline');
    section(c, 'layer.main', 'Layer', null,
      h('div', { class: 'nn-field nn-grid2' }, h('span', { class: 'nn-k', text: 'name' }), name),
      h('div', { class: 'nn-field nn-grid2' }, h('span', { class: 'nn-k', text: 'size' }), sizeRow),
      shape);

    if (attn) { buildAttnLayer(c, L, lid); return; }

    if (L > 0) {
      const def = h('div', { class: 'nn-def' });
      const plot = makePlot();
      c.binds.push(() => {
        const act = actOf(L);
        setTex(def, actDef(act) + (isVec(act) && tok ? '\\quad \\text{(per token)}' : ''));
        setHidden(plot.el, isVec(act));
        if (!isVec(act)) {
          const hv = store.state.hover;
          plot.draw(actFn(act), nodes().map(q => ({ ...(fwdNode(q.id) || {}), hi: hv && (hv.id === q.id) })), act);
        }
      });
      section(c, 'layer.act', 'Activation', null, h('div', { class: 'nn-act-row' }, actSelect(c, lid), def), plot.el);
      sharedSection(c, L);
    }

    // wiring (never into an attention layer: it takes no edges)
    const acts = [];
    const mix = t => (tok || (L > 0 && isTok(L - 1)) ? `${t}. Dense edges mix the tokens and are not shared.` : t);
    if (L > 0) {
      const b = h('button', { class: 'nn-mini', title: mix('Connect every neuron of the previous layer to every neuron here') }, '← connect previous');
      b.addEventListener('click', () => dense(net().layers[L - 1].id, lid));
      acts.push(b);
    }
    if (L < last && !isAttn(L + 1)) {
      const b = h('button', { class: 'nn-mini', title: mix('Connect every neuron here to every neuron of the next layer') }, 'connect next →');
      b.addEventListener('click', () => dense(lid, net().layers[L + 1].id));
      acts.push(b);
    }
    if (L > 0) {
      const scheme = h('select', { class: 'nn-scheme', title: 'Initialisation scheme' }, SCHEMES.map(s => h('option', { value: s }, s)));
      scheme.value = schemeFor(L);
      const b = h('button', { class: 'nn-mini', title: 'Re-draw the incoming weights and biases of this layer (one value per shared parameter; fixed weights stay)' }, 'randomize');
      b.addEventListener('click', () => randomizeLayer(lid, scheme.value));
      acts.push(h('span', { class: 'nn-rand' }, b, scheme));
    }
    if (acts.length) section(c, 'layer.wire', 'Wiring', null, h('div', { class: 'nn-row' }, ...acts));

    // bias vector (or the input vector)
    const vec = texBox(c, () => {
      const vs = nodes().map(q => fmt(L === 0 ? q.value : q.bias));
      // elide the middle until the row fits the card (each cell also carries its column gap)
      const rowW = cells => cells.reduce((s, v) => s + texWidth(v) + 1.5, 6);
      let shown = vs;
      for (let keep = vs.length - 1; rowW(shown) > CARD_UNITS && keep >= 3; keep--) {
        shown = [...vs.slice(0, keep - 1), '\\cdots', ...vs.slice(-1)];
      }
      return `${L === 0 ? 'x' : `b^{(${L})}`} = \\begin{bmatrix} ${shown.join(' & ')} \\end{bmatrix}^{\\top}`;
    }, false, 'nn-math nn-inline');
    const rows = nodes().map(q => slider(c, L === 0 ? {
      lab: el => setTex(el, label(q.id)),
      get: () => node(q.id)?.value,
      set: v => M.setNode(net(), q.id, { value: v }),
      what: 'input value', hover: { kind: 'node', id: q.id }, other: q.id, scale: () => maxA,
      onLabel: () => select({ kind: 'node', id: q.id }), labelTitle: 'Open this neuron',
    } : {
      lab: el => setTex(el, label(q.id)),
      get: () => node(q.id)?.bias,
      set: v => M.setNode(net(), q.id, { bias: v }),
      what: 'bias', hover: { kind: 'bias', id: q.id }, other: q.id, scale: () => maxB,
      onLabel: () => select({ kind: 'node', id: q.id }),
      labelTitle: biasTie(q.id) ? `Shared bias ${tieText(biasTie(q.id))}: open this neuron` : 'Open this neuron',
    }));
    rows.forEach((r, k) => {
      const q = nodes()[k];
      if (biasTie(q?.id)) r.classList.add('tied');
      if (q) dimBy(c, r, E => nodeEm(E, q.id));
    });
    section(c, 'layer.vec', L === 0 ? 'Inputs' : 'Biases', null, vec, ...rows);
  }

  // The shared parameters feeding layer L as small matrices (X W convention: rows = input feature,
  // columns = output feature), and the layer as a matrix product: Q = X W_Q, H = ReLU(Z W_O + X).
  function sharedSection(c, L) {
    const s = shapeOf(L), pos = new Map(M.nodesIn(net(), L).map((q, k) => [q.id, k]));
    const mats = new Map(), fixedFrom = new Set(), plainFrom = new Set();
    for (const e of net().edges) {
      if (!pos.has(e.to)) continue;
      const lf = M.nodeLayerIndex(net(), e.from), t = tieOf(e);
      if (e.fixed) { fixedFrom.add(lf); continue; }
      if (!t) { plainFrom.add(lf); continue; }
      const p = tieParts(t);
      let m = mats.get(p.name);
      if (!m) mats.set(p.name, m = { name: p.name, cell: new Map(), rows: 0, cols: 0, groups: new Set(), from: lf, n: 0 });
      m.n++;
      if (p.i && !m.cell.has(`${p.i},${p.j}`)) {
        m.cell.set(`${p.i},${p.j}`, e.id);
        m.rows = Math.max(m.rows, p.i);
        m.cols = Math.max(m.cols, p.j);
      }
      if (s.groups) m.groups.add(Math.floor(pos.get(e.to) / (s.tokens * s.d)));
    }
    if (!mats.size) return;
    const list = [...mats.values()];
    // The layer as a product (Q = X W_Q) only when that is exactly what it computes: every shared
    // matrix tokenwise (model.tiedMatrices: the layer matrix is I ⊗ Wᵀ), every other edge a fixed
    // identity residual. A conv kernel or an RNN's x_t w_x is shared but not such a product.
    const tw = typeof M.tiedMatrices === 'function' ? M.tiedMatrices(net(), L) : [];
    const residual = [...fixedFrom].every(k => net().edges.every(e => {
      if (!e.fixed || !pos.has(e.to) || M.nodeLayerIndex(net(), e.from) !== k) return true;
      const a = tokPos(k, indexIn(e.from)), b = tokPos(L, pos.get(e.to));
      return e.w === 1 && a.t === b.t && a.f === b.f;
    }));
    const exact = !plainFrom.size && residual && list.every(m => tw.some(t => t.name === m.name && t.tokenwise));
    const rel = !exact ? null : texBox(c, () => {
      const act = actOf(L), fn = FN[act] ?? `\\operatorname{${act}}`;
      const anyB = M.nodesIn(net(), L).some(q => Number.isFinite(q.bias) && q.bias !== 0);
      const wrap = x => (act === 'identity' ? x : `${fn}\\big(${x}\\big)`);
      const bias = anyB ? ' + b' : '';
      if (s.groups) {
        return s.groups.map((g, gi) => {
          const ms = list.filter(m => m.groups.has(gi));
          return ms.length ? `{${g}} = ${wrap(ms.map(m => `${matSym(m.from)}\\,{${m.name}}`).join(' + ') + bias)}` : null;
        }).filter(Boolean).join(',\\quad ');
      }
      const parts = [...list.map(m => `${matSym(m.from)}\\,{${m.name}}`), ...[...fixedFrom].map(k => matSym(k))];
      return `${isTok(L) ? matSym(L) : `a^{(${L})}`} = ${wrap(parts.join(' + ') + bias)}`;
    }, true);
    const cut = (n, max = 6) => (n > max ? [0, 1, 2, 3, -1, n - 1] : [...Array(n).keys()]);   // -1 = dots
    const boxes = list.map(m => texBox(c, () => {
      if (!m.rows || !m.cols) return `{${m.name}}\\ \\text{(${m.n} edges)}`;
      const rows = cut(m.rows).map(i => cut(m.cols).map(j => {
        if (i < 0 || j < 0) return i < 0 && j < 0 ? '\\ddots' : i < 0 ? '\\vdots' : '\\cdots';
        const eid = m.cell.get(`${i + 1},${j + 1}`);
        return eid ? wTex(edge(eid)?.w) : '\\cdot';
      }).join(' & '));
      return `{${m.name}} = \\begin{bmatrix} ${rows.join(' \\\\ ')} \\end{bmatrix} \\in \\mathbb{R}^{${m.rows} \\times ${m.cols}}`;
    }, false, 'nn-math nn-inline'));
    const note = h('div', { class: 'nn-hint' });
    const T = s.tokens;
    note.textContent = exact && T > 1
      ? `Each shared matrix is applied to every one of the ${T} tokens: in z = W a it is the block-diagonal I${sub(T)} ⊗ Wᵀ. Moving one entry moves all its copies.`
      : 'Each shared entry is used by several edges: moving it moves all its copies.';
    section(c, 'layer.shared', 'Shared weights', null, rel, ...boxes, note);
  }
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(d => SUBS[+d] || d).join('');

  // Tokens, heads, the causal mask and the scale, then this sample's attention matrix per head.
  function buildAttnLayer(c, L, lid) {
    const prevD = shapeOf(L - 1).d;
    const divs = [];
    for (let k = 1; k <= prevD; k++) if (prevD % k === 0) divs.push(k);
    const tokEl = h('span', { class: 'nn-size' });
    const heads = h('select', { class: 'nn-heads', title: 'Heads split d_k and d_v into equal slices, one softmax per slice' },
      divs.map(k => h('option', { value: String(k) }, String(k))));
    heads.addEventListener('change', () => setAttnLayer(lid, { heads: +heads.value }, 'heads'));
    const causal = h('input', { type: 'checkbox', class: 'nn-check', title: 'Causal mask: token i only attends to tokens j ≤ i' });
    causal.addEventListener('change', () => { setAttnLayer(lid, { causal: causal.checked }, 'causal'); causal.blur(); });
    const scale = h('input', { type: 'number', step: '0.05', class: 'nn-num nn-scale', title: 'The factor c in S = Q Kᵀ c' });
    scale.addEventListener('change', () => {
      const v = parseNum(scale.value);
      setAttnLayer(lid, { scale: v }, 'scale');
    });
    scale.addEventListener('keydown', e => { if (e.key === 'Enter') scale.blur(); });
    const def = h('button', { class: 'nn-mini', title: 'Back to the default 1/√(d_k / heads)' }, 'default');
    def.addEventListener('click', () => setAttnLayer(lid, { scale: null }, 'scale'));
    const scaleNote = h('span', { class: 'nn-k nn-scale-note' });
    c.binds.push(() => {
      const T = shapeOf(L).tokens, names = Array.from({ length: T }, (_, t) => tokName(t));
      setText(tokEl, names.some(Boolean) ? `${T}: ${names.map((n, t) => n || `t${t + 1}`).join(', ')}` : String(T));
      setVal(heads, String(headsOf(L)));
      if (document.activeElement !== causal) causal.checked = causalOf(L);
      setVal(scale, numText(scaleOf(L)));
      def.disabled = ro || !customScale(L);
      setTex(scaleNote, customScale(L) ? '\\text{custom}' : `= 1/\\sqrt{${+dkHead(L).toFixed(3)}}`);
    });
    // what the Layer line's c, M and heads mean
    const formula = texBox(c, () => {
      const H = headsOf(L), lines = [];
      if (causalOf(L)) lines.push('M_{i,j} = -\\infty \\text{ for } j > i, \\text{ else } 0');
      const dh = +dkHead(L).toFixed(3);
      if (H > 1) lines.push(`\\text{each head: its own } ${dh} \\text{ column${dh === 1 ? '' : 's'} of } Q, K, V`);
      return lines.length ? aligned(lines.map(s => `&${s}`)) : null;
    }, true);
    const maps = h('div', { class: 'nn-amaps' });
    let mapKey = '', cells = [], hdrs = [];
    c.binds.push(() => {
      const T = shapeOf(L).tokens, H = headsOf(L), cz = causalOf(L);
      const key = `${T}|${H}|${cz}`;
      if (key !== mapKey) {
        mapKey = key;
        const old = new Set(maps.querySelectorAll('div, span'));
        c.hov = c.hov.filter(el => !old.has(el));
        c.dims = c.dims.filter(d => !old.has(d.el));
        cells = [];
        hdrs = [];
        maps.replaceChildren();
        // Row i is token i's attention row, column j key / value token j: a cell or header hovers
        // that token ({ kind: 'token' }), and lights with it (Q, K and V are groups 0, 1, 2).
        for (let hd = 0; hd < H; hd++) {
          const hh = H > 1 ? hd : null, tok = (layer, t, g = null) => ({ kind: 'token', layer, t, ...(g == null ? {} : { g }), ...(hh == null ? {} : { h: hh }) });
          const grid = h('div', { class: 'nn-amap' });
          grid.style.gridTemplateColumns = `auto repeat(${T}, minmax(0, 1fr))`;
          grid.append(h('span', { class: 'nn-amap-c', text: H > 1 ? `h${hd + 1}` : 'A' }));
          dimBy(c, grid, E => headEm(E, L, hd));
          for (let j = 0; j < T; j++) {
            const kh = h('span', { class: 'nn-amap-h' });
            hoverable(c, kh, tok(L - 1, j, 1));
            tokenLit(c, kh, [{ layer: L - 1, t: j, g: 1, h: hh }, { layer: L - 1, t: j, g: 2, h: hh }]);
            grid.append(kh);
            hdrs.push({ el: kh, t: j, key: true });
          }
          for (let i = 0; i < T; i++) {
            const qh = h('span', { class: 'nn-amap-h' });
            hoverable(c, qh, tok(L, i));
            tokenLit(c, qh, [{ layer: L, t: i, h: hh }, { layer: L - 1, t: i, g: 0, h: hh }]);
            dimBy(c, qh, E => rowEm(E, L, i));
            grid.append(qh);
            hdrs.push({ el: qh, t: i, key: false });
            for (let j = 0; j < T; j++) {
              const cell = h('span', { class: 'nn-amap-v' + (cz && j > i ? ' masked' : '') });
              hoverable(c, cell, tok(L, i));
              tokenLit(c, cell, [{ layer: L, t: i, h: hh }, { layer: L - 1, t: i, g: 0, h: hh },
                { layer: L - 1, t: j, g: 1, h: hh }, { layer: L - 1, t: j, g: 2, h: hh }]);
              dimBy(c, cell, E => (cz && j > i ? null : attnEm(E, L, i, j, hd)));
              grid.append(cell);
              cells.push({ cell, hd, i, j });
            }
          }
          maps.append(grid);
        }
        paintHover();
      }
      // headers: q_i / k_j, or the token's name (named tokens)
      for (const q of hdrs) {
        const nm = tokName(q.t);
        setText(q.el, nm || `${q.key ? 'k' : 'q'}${q.t + 1}`);
        q.el.classList.toggle('named', !!nm);
        const tip = `${q.key ? 'key' : 'query'} ${q.t + 1}${nm ? `: ${nm}` : ''}`;
        if (q.el.title !== tip) q.el.title = tip;
      }
      for (const q of cells) {
        const a = attnFwd(L, q.hd)?.A?.[q.i]?.[q.j];
        const masked = cz && q.j > q.i;
        setText(q.cell, masked ? '–' : Number.isFinite(a) ? fmt(a) : '?');
        setStyle(q.cell, 'background', masked || !Number.isFinite(a) ? 'transparent' : colorFor(a, 1, theme()));
        const tip = `A(${q.i + 1},${q.j + 1}): how much ${tokWord(q.i)} reads ${tokWord(q.j)}`;
        if (q.cell.title !== tip) q.cell.title = tip;
      }
    });
    section(c, 'layer.attn', 'Attention', null,
      h('div', { class: 'nn-attn-grid' },
        h('span', { class: 'nn-k', text: 'tokens' }), tokEl,
        h('span', { class: 'nn-k', text: 'heads' }), heads,
        h('span', { class: 'nn-k', text: 'causal' }), h('label', { class: 'nn-check-row' }, causal, h('span', { text: 'mask j > i' })),
        h('span', { class: 'nn-k', text: 'scale' }), h('span', { class: 'nn-scale-row' }, scale, scaleNote, def)),
      formula, maps,
      h('div', { class: 'nn-hint', text: 'A for the current sample. No weights come in and there is no bias: Q, K and V come from the previous layer.' }));

    if (L < lastIndex()) {
      const b = h('button', { class: 'nn-mini', title: 'Connect every neuron here to every neuron of the next layer. Dense edges mix the tokens and are not shared.' }, 'connect next →');
      b.addEventListener('click', () => dense(lid, net().layers[L + 1].id));
      section(c, 'layer.wire', 'Wiring', null, h('div', { class: 'nn-row' }, b));
    }
  }

  // heads / causal / scale through model.setLayer, which checks heads divides d; scale null = default.
  function setAttnLayer(lid, patch, what) {
    if (!layerOf(lid)) return;
    M.setLayer(net(), lid, patch);
    store.commit(what);
  }

  const schemeFor = L => (/relu|leaky/.test(actOf(L)) ? 'he' : 'xavier');
  const seed = () => (Math.random() * 2 ** 31) | 0;

  // Missing edges only: existing weights are kept.
  function dense(fromLid, toLid) {
    M.connectDense(net(), fromLid, toLid, { seed: seed(), scheme: schemeFor(M.layerIndex(net(), toLid)) });
    store.commit('connect dense');
  }

  // Re-draw only this layer's incoming weights and biases, using the model's own schemes.
  function randomizeLayer(lid, scheme) {
    const n = net();
    const tmp = M.clone(n);
    M.randomize(tmp, { seed: seed(), scheme, biases: 'small' });
    const ids = new Set(M.nodesIn(n, M.layerIndex(n, lid)).map(q => q.id));
    const done = new Set();   // tie groups already drawn: one value per shared parameter
    for (const e of n.edges) {
      if (!ids.has(e.to) || e.fixed) continue;
      const t = tieOf(e);
      if (t && done.has(t)) continue;
      if (t) done.add(t);
      const w = M.edge(tmp, e.id)?.w;
      if (Number.isFinite(w)) setW(e.id, w);
    }
    for (const id of ids) {
      const b = M.node(tmp, id)?.bias;
      if (Number.isFinite(b)) M.setNode(n, id, { bias: b });
    }
    store.commit('randomize layer');
  }

  // Add or remove the last neuron of a layer, keeping the column centred. A new neuron is wired
  // to the previous / next layer when the layer already has edges there, with fresh weights.
  function resize(lid, d) {
    const n = net();
    const L = M.layerIndex(n, lid);
    const nodes = M.nodesIn(n, L);
    const lastN = nodes[nodes.length - 1], prevN = nodes[nodes.length - 2];
    let gap = lastN && prevN ? lastN.y - prevN.y : 0;
    if (!(Math.abs(gap) > 4)) gap = 70;
    if (d < 0) {
      if (nodes.length <= 1) return;
      M.removeNode(n, lastN.id);
      for (const q of M.nodesIn(n, L)) M.setNode(n, q.id, { y: q.y + gap / 2 });
      store.commit('remove neuron');
      return;
    }
    const x = lastN ? lastN.x : 120 + 160 * L;
    const y = lastN ? lastN.y + gap : 120;
    const id = M.addNode(n, lid, { x, y });
    for (const q of M.nodesIn(n, L)) M.setNode(n, q.id, { y: q.y - gap / 2 });
    const others = new Set(nodes.map(q => q.id));
    const fresh = [];
    const wire = (k, incoming) => {
      if (k < 0 || k > n.layers.length - 1) return;
      const ks = M.nodesIn(n, k);
      const kIds = new Set(ks.map(q => q.id));
      const used = n.edges.some(e => (incoming ? others.has(e.to) && kIds.has(e.from) : others.has(e.from) && kIds.has(e.to)));
      if (!used) return;
      for (const q of ks) {
        const eid = incoming ? M.connect(n, q.id, id) : M.connect(n, id, q.id);
        if (eid != null) fresh.push(eid);
      }
    };
    wire(L - 1, true);
    wire(L + 1, false);
    if (fresh.length) {
      const tmp = M.clone(n);
      M.randomize(tmp, { seed: seed(), scheme: schemeFor(L), biases: 'keep' });
      for (const eid of fresh) {
        const w = M.edge(tmp, eid)?.w;
        if (Number.isFinite(w)) M.setWeight(n, eid, w);
      }
    }
    store.commit('add neuron');
  }

  // ---------------------------------------------------------------- wiring

  store.on('sel', syncSel);
  store.on('net', p => { if (p?.structural) { rebuildAll(); ensureLoop(); } });
  // At most one refresh per frame, however often 'values' fires (slider input, training).
  let uraf = 0;
  store.on('values', () => {
    if (!uraf && cards.length) uraf = requestAnimationFrame(() => { uraf = 0; if (visible) updateAll(); });
  });
  store.on('train', p => {
    const was = training;
    training = !!p?.running;
    if (was && !training && cards.length && visible) updateAll();   // paused: every box current again
  });
  store.on('layout', () => { for (const c of cards) c.akey = ''; ensureLoop(); });
  store.on('hover', paintHover);
  store.on('lens', paintLensAll);
  ctx.onTheme?.(() => { for (const c of cards) update(c); });
  ctx.onShow?.(v => {
    visible = !!v;
    if (visible) { updateAll(); for (const c of cards) c.akey = ''; ensureLoop(); }
  });

  const api = {
    // Open a (by default pinned) card; a target that already has a card reuses it.
    open(target, opts = {}) {
      if (!exists(target)) return null;
      if (opts.pin === false) {   // unpinned = the card that follows the selection
        store.set('sel', { kind: target.kind, id: target.id });
        return cards.find(k => same(k.target, target))?.el || null;
      }
      let c = cards.find(k => same(k.target, target));
      if (c) {
        if (!c.pinned) togglePin(c);
        raise(c);
        flash(c);
      } else {
        c = open({ kind: target.kind, id: target.id }, { pin: true });
      }
      ensureLoop();
      return c.el;
    },
    closeAll() { for (const c of [...cards]) close(c); },
    get cards() { return cards.map(c => ({ target: { ...c.target }, pinned: c.pinned, el: c.el })); },
  };
  ctx.inspector = api;
  updateAll();
  syncSel();
  return api;
}
