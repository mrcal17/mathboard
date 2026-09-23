// The 3D tab: a Desmos-style expression list driving scene.js, plus the Board/3D/Net tab switch
// (the Net tab itself is static/nn/nn.js; this file only shows and hides its section).
// Vectors are drawn from the origin unless the row says `@ <point>`.
//
// Features live in graph/features/<name>.js and export `install(api)`; see `api` below.
// A broken or missing feature is logged and skipped, never fatal.
//
// Test hook: index.html#graph=<encodeURIComponent(lines joined by \n)>[&view=iso|top|front|side]
// opens the 3D tab with those rows without touching the saved list.
import * as lang from './lang.js';
import { createScene } from './scene.js';

const { evaluate, formatValue, formatNumber } = lang;
const $ = id => document.getElementById(id);
const uid = () => Math.random().toString(36).slice(2, 10);
const PALETTE = ['#e05a4f', '#4a90e2', '#43b05c', '#9b6ade', '#f5a623', '#26b5b5', '#e056a0', '#a1887f'];
const FEATURES = ['transform', 'fields', 'combos', 'systems', 'dual', 'lecture', 'present', 'bridge', 'drag'];
const STORE = 'mathboard.graph';
// Different speeds let two playing sliders sweep out an area instead of retracing one line.
const SPEEDS = [1, 1.618, 2, 0.5, 0.25];
const SPEEDS_LABEL = { 1: '1', 1.618: '&phi;', 2: '2', 0.5: '&frac12;', 0.25: '&frac14;' };
const EXAMPLE = [
  'u = (1, 2, 0)',
  'v = (-1, 1, 2)',
  'a = 1.5',
  'b = 1',
  'a u',
  'b v @ a u',
  'w = a u + b v',
  'span(u, v)',
];
// Right-hand sides that are plain literals don't need an "= value" echo underneath.
const LITERAL = /^\s*(?:[([]\s*-?[\d.]+(?:e-?\d+)?\s*(?:,\s*-?[\d.]+(?:e-?\d+)?\s*){1,2}[)\]]|-?[\d.]+)\s*$/;

const hashParams = new URLSearchParams(location.hash.slice(1));
const preset = hashParams.has('graph') ? hashParams.get('graph').split('\n') : null;
// Audience/mirror windows (?audience) and #graph= test links never write the saved list.
const persist = !preset && !new URLSearchParams(location.search).has('audience');
// #nn= test links (static/nn/nn.js) open the Net tab without remembering it as the last view.
const persistView = persist && !hashParams.has('nn');
const VIEWS = ['board', 'graph', 'nn'];

let saved = null;
try { saved = JSON.parse(localStorage.getItem(STORE)); } catch { /* fresh start */ }
if (preset) saved = null;
let rows = [];
let nextColor = 0;
let results = [];
let scene = null;

const hooks = { recompute: [], items: [], rowDecor: [], sceneReady: [], view: [] };

function normalizeRows(list) {
  return list.map(r => {
    const src = typeof r === 'string' ? r : r.src;
    const o = typeof r === 'string' ? {} : r;
    return { id: uid(), src, color: o.color || PALETTE[nextColor++ % PALETTE.length], hidden: !!o.hidden, min: o.min, max: o.max, speed: o.speed, playing: false };
  });
}

// ================================================================ rows UI
const rowsEl = $('g-rows');

function makeRowEl(r) {
  const li = document.createElement('li');
  li.className = 'g-row';
  li.innerHTML = `
    <button class="g-dot" title="Show / hide"></button>
    <div class="g-main">
      <input class="g-src" spellcheck="false" autocomplete="off">
      <div class="g-out"></div>
      <div class="g-slider" hidden>
        <input class="g-min" type="number" title="Slider minimum">
        <input class="g-range" type="range">
        <input class="g-max" type="number" title="Slider maximum">
        <button class="g-play" title="Animate">&#9654;</button>
        <button class="g-speed" title="Animation speed (click to change)">1&times;</button>
      </div>
    </div>
    <button class="g-del" title="Delete">&times;</button>`;
  const q = sel => li.querySelector(sel);
  r.el = { li, dot: q('.g-dot'), src: q('.g-src'), out: q('.g-out'), slider: q('.g-slider'), main: q('.g-main'),
    min: q('.g-min'), range: q('.g-range'), max: q('.g-max'), play: q('.g-play'), speed: q('.g-speed') };
  r.el.src.value = r.src;
  r.el.src.style.setProperty('--c', r.color);
  r.el.dot.style.setProperty('--c', r.color);

  r.el.src.addEventListener('input', () => { r.src = r.el.src.value; recompute(); });
  r.el.src.addEventListener('keydown', e => onRowKey(e, r));
  r.el.dot.onclick = () => { r.hidden = !r.hidden; recompute(); };
  q('.g-del').onclick = () => removeRow(r, true);
  r.el.range.addEventListener('input', () => setSlider(r, Number(r.el.range.value)));
  const bound = which => () => {
    const x = Number(r.el[which].value);
    if (Number.isFinite(x)) r[which] = x;
    recompute();
  };
  r.el.min.addEventListener('change', bound('min'));
  r.el.max.addEventListener('change', bound('max'));
  r.el.play.onclick = () => { r.playing = !r.playing; r.dir ||= 1; if (r.playing) kickAnimation(); paintRows(); };
  r.el.speed.onclick = () => { r.speed = SPEEDS[(SPEEDS.indexOf(r.speed ?? 1) + 1) % SPEEDS.length]; paintRows(); save(); };
  return li;
}

function buildRows() {
  rowsEl.textContent = '';
  for (const r of rows) rowsEl.appendChild(makeRowEl(r));
}

function addRow(after, src = '', opts = {}) {
  const [r] = normalizeRows([{ src, color: opts.color, hidden: opts.hidden, min: opts.min, max: opts.max, speed: opts.speed }]);
  const i = after ? rows.indexOf(after) + 1 : rows.length;
  rows.splice(i, 0, r);
  rowsEl.insertBefore(makeRowEl(r), rowsEl.children[i] || null);
  recompute();
  if (opts.focus !== false) r.el.src.focus();
  return r;
}

function removeRow(r, focusPrev) {
  const i = rows.indexOf(r);
  if (i < 0) return;
  rows.splice(i, 1);
  r.el.li.remove();
  if (!rows.length) addRow(null);
  else if (focusPrev) rows[Math.max(0, i - 1)].el.src.focus();
  recompute();
}

function setRowSource(r, src) {
  r.src = src;
  r.el.src.value = src;
  recompute();
}

function setRows(list) {
  for (const r of rows) r.playing = false;
  rows = normalizeRows(list);
  buildRows();
  recompute();
}

function onRowKey(e, r) {
  const i = rows.indexOf(r);
  if (e.key === 'Enter') { e.preventDefault(); addRow(r); }
  else if (e.key === 'Backspace' && !r.el.src.value && rows.length > 1) { e.preventDefault(); removeRow(r, true); }
  else if (e.key === 'ArrowUp' && i > 0) { e.preventDefault(); rows[i - 1].el.src.focus(); }
  else if (e.key === 'ArrowDown' && i < rows.length - 1) { e.preventDefault(); rows[i + 1].el.src.focus(); }
  else if (e.key === 'Escape') r.el.src.blur();
}

// ================================================================ sliders
function sliderRange(r, x) {
  const min = r.min ?? Math.min(-5, Math.floor(x));
  const max = r.max ?? Math.max(5, Math.ceil(x));
  return min < max ? [min, max] : [min, min + 1];
}

function setSlider(r, x) {
  const res = results[rows.indexOf(r)];
  if (!res?.name) return;
  r.src = `${res.name} = ${formatNumber(x)}`;
  r.el.src.value = r.src;
  recompute();
}

let animating = false, lastT = 0;
function kickAnimation() {
  if (!animating) { animating = true; lastT = 0; requestAnimationFrame(animate); }
}
function animate(t) {
  const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
  lastT = t;
  let any = false;
  rows.forEach((r, i) => {
    const res = results[i];
    if (!r.playing) return;
    if (res?.slider == null) { r.playing = false; return; }
    any = true;
    const [mn, mx] = sliderRange(r, res.slider);
    let x = res.slider + r.dir * (mx - mn) * (r.speed ?? 1) * dt / 4; // full sweep in 4 s at 1x, ping-pong
    if (x >= mx) { x = mx; r.dir = -1; }
    if (x <= mn) { x = mn; r.dir = 1; }
    r.src = `${res.name} = ${formatNumber(x)}`;
    r.el.src.value = r.src;
  });
  if (any) { recompute(); requestAnimationFrame(animate); }
  else { animating = false; paintRows(); }
}

// ================================================================ evaluation -> rows + scene
function recompute() {
  results = evaluate(rows.map(r => r.src));
  paintRows();
  for (const fn of hooks.recompute) safe(fn, results, rows);
  if (scene) scene.setContent(sceneItems());
  save();
}

const drawable = v => v != null && typeof v === 'object' && v.type !== 'mat' && lang.isDrawable?.(v) !== false;

function paintRows() {
  rows.forEach((r, i) => {
    const res = results[i] || {}, el = r.el;
    const draws = !res.error && drawable(res.value);
    el.dot.classList.toggle('off', r.hidden);
    el.dot.classList.toggle('inert', !draws);
    el.out.classList.toggle('err', !!res.error);

    const isSlider = !res.error && res.slider != null;
    el.slider.hidden = !isSlider;
    if (isSlider) {
      const [mn, mx] = sliderRange(r, res.slider);
      if (document.activeElement !== el.min) el.min.value = mn;
      if (document.activeElement !== el.max) el.max.value = mx;
      el.range.min = mn;
      el.range.max = mx;
      el.range.step = (mx - mn) / 1000;
      el.range.value = res.slider;
      el.play.innerHTML = r.playing ? '&#10074;&#10074;' : '&#9654;';
      el.speed.innerHTML = `${SPEEDS_LABEL[r.speed ?? 1]}&times;`;
    }

    if (res.error) el.out.textContent = res.error;
    else if (res.value == null || isSlider) el.out.textContent = '';
    else if (res.value.type === 'mat' && /^\s*\[[\d\s.,;eE+\-[\]]*$/.test(r.src.split('@')[0].split('=').pop())) {
      el.out.textContent = ''; // a literal matrix is already visible in the row itself
    } else if (res.value.type === 'mat') {
      const body = res.value.m.map(row => row.map(formatNumber).join(' & ')).join(' \\\\ ');
      el.out.innerHTML = katex.renderToString(`= \\begin{bmatrix}${body}\\end{bmatrix}`, { throwOnError: false });
    } else if (lang.valueLatex?.(res.value)) {
      el.out.innerHTML = katex.renderToString(`= ${lang.valueLatex(res.value)}`, { throwOnError: false });
    } else {
      const rhs = r.src.split('@')[0].split('=').pop();
      el.out.textContent = LITERAL.test(rhs) && res.value.type !== 'span' ? '' : `= ${formatValue(res.value)}`;
    }
    for (const fn of hooks.rowDecor) safe(fn, r, res, el);
  });
}

function sceneItems() {
  let items = [];
  rows.forEach((r, i) => {
    const res = results[i];
    if (r.hidden || !res || res.error || !drawable(res.value)) return;
    items.push({ ...res.value, kind: res.value.type, o: res.origin || [0, 0, 0], color: r.color, label: res.name, index: i, rowId: r.id });
  });
  for (const fn of hooks.items) {
    try { items = fn(items, { rows, results }) || items; } catch (err) { console.error('[graph] items hook:', err); }
  }
  return items;
}

function safe(fn, ...args) {
  try { return fn(...args); } catch (err) { console.error('[graph] hook failed:', err); }
}

let saveTimer = 0;
function save() {
  if (!persist) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const data = {
      rows: rows.map(({ src, color, hidden, min, max, speed }) => ({ src, color, hidden, min, max, speed })),
      nextColor, extent: scene?.extent ?? saved?.extent, collapsed: $('graph').classList.contains('collapsed'),
    };
    localStorage.setItem(STORE, JSON.stringify(data));
  }, 300);
}

// ================================================================ tabs + panel chrome
const theme = () => document.documentElement.dataset.theme || 'dark';
let view = 'board';

function setView(next) {
  view = next;
  document.body.dataset.view = next;
  $('graph').hidden = next !== 'graph';
  if ($('nn')) $('nn').hidden = next !== 'nn';
  for (const b of document.querySelectorAll('#tabs button')) b.classList.toggle('on', b.dataset.view === next);
  if (persistView) localStorage.setItem('mathboard.view', next);
  if (next === 'graph') {
    if (!scene) {
      scene = createScene($('g-view'));
      if (saved?.extent) scene.setExtent(saved.extent);
      for (const fn of hooks.sceneReady) safe(fn, scene);
    }
    scene.setTheme(theme());
    scene.setContent(sceneItems());
    scene.start();
  } else if (scene) {
    scene.stop();
  }
  for (const fn of hooks.view) safe(fn, next);
}

const setCollapsed = on => {
  $('graph').classList.toggle('collapsed', on);
  $('g-expand').hidden = !on;
  save();
};

for (const b of document.querySelectorAll('#tabs button')) b.onclick = () => setView(b.dataset.view);
$('g-add').onclick = () => addRow(null);
$('g-fit').onclick = () => { scene?.fit(); save(); };
$('g-reset').onclick = () => scene?.reset();
$('g-help-btn').onclick = () => { $('g-help').hidden = !$('g-help').hidden; };
$('g-collapse').onclick = () => setCollapsed(true);
$('g-expand').onclick = () => setCollapsed(false);

window.addEventListener('keydown', e => {
  const i = ['1', '2', '3'].indexOf(e.key);
  if (e.altKey && i >= 0) { e.preventDefault(); setView(VIEWS[i]); }
});
new MutationObserver(() => scene?.setTheme(theme()))
  .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

let toastTimer = 0;
function toast(msg, ms = 2200) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

// ================================================================ feature API
const api = {
  lang, PALETTE, params: new URLSearchParams(location.search), hashParams, persist,
  get rows() { return rows; },
  get results() { return results; },
  get scene() { return scene; },
  get view() { return view; },
  // rows
  addRow: (src, opts = {}) => addRow(opts.after || null, src, opts),
  removeRow: r => removeRow(r, false),
  setRowSource, setRows, recompute,
  rowByName: name => rows[results.findIndex(res => res?.name === name)] || null,
  // state snapshot for slides / presenter sync
  getState: () => ({
    rows: rows.map(({ src, color, hidden, min, max, speed }) => ({ src, color, hidden, min, max, speed })),
    camera: scene?.getPose() ?? null,
    collapsed: $('graph').classList.contains('collapsed'),
  }),
  setState(state, { cameraMs = 900 } = {}) {
    if (state.rows) setRows(state.rows);
    if (state.collapsed != null) setCollapsed(state.collapsed);
    if (state.camera && scene) scene.setPose(state.camera, cameraMs);
  },
  // hooks
  onRecompute: fn => hooks.recompute.push(fn),
  addItemsHook: fn => hooks.items.push(fn),
  addRowDecorator: fn => hooks.rowDecor.push(fn),
  onSceneReady: fn => { if (scene) safe(fn, scene); else hooks.sceneReady.push(fn); },
  onViewChange: fn => hooks.view.push(fn),
  // chrome
  addToolbarButton({ label, title, onClick }) {
    const b = document.createElement('button');
    b.innerHTML = label;
    if (title) b.title = title;
    b.onclick = onClick;
    $('g-tools').appendChild(b);
    $('g-tools').hidden = false;
    return b;
  },
  addOverlay(el) { $('g-view').appendChild(el); return el; },
  addStyles(css) { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); return s; },
  setView, setCollapsed, toast,
  get panelEl() { return $('g-panel'); },
  get viewEl() { return $('g-view'); },
};
window.mathboardGraph = api;

// ================================================================ start
const modules = await Promise.all(FEATURES.map(name => import(`./features/${name}.js`).catch(err => {
  if (!/Failed to fetch|404|not found/i.test(String(err))) console.error(`[graph] feature ${name}:`, err);
  return null;
})));
for (const [i, m] of modules.entries()) {
  if (!m?.install) continue;
  try { await m.install(api); } catch (err) { console.error(`[graph] feature ${FEATURES[i]} install:`, err); }
}

rows = normalizeRows(preset || (saved?.rows?.length ? saved.rows : EXAMPLE));
if (saved?.nextColor != null) nextColor = saved.nextColor;
buildRows();
recompute();
if (saved?.collapsed) setCollapsed(true);
const lastView = localStorage.getItem('mathboard.view');
setView(preset ? 'graph' : hashParams.has('nn') ? 'nn' : VIEWS.includes(lastView) ? lastView : 'board');
if (preset && hashParams.get('view')) scene?.viewPreset(hashParams.get('view'), 0);
document.body.dataset.graphReady = '1';
