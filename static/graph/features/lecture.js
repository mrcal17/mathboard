// Lecture tools for the 3D tab.
//   Steps: named snapshots of api.getState() (+ 2D mode and auto-rotate). Left/Right or
//   PageUp/PageDown walk through them with camera flights; when only slider values differ the
//   sliders are tweened. Kept in localStorage 'mathboard.steps'; Export/Import as .json.
//   Camera: view presets (1-4), orthographic (O), slow auto-rotate (R), 2D mode. H hides the UI.
//   Audience: index.html?audience=1 shows only the 3D view and mirrors the presenter over a
//   BroadcastChannel (rows once per frame, camera on change, theme, step overlay). It never posts.
//   It follows the presenter into the Net tab (static/nn/nn.js) and mirrors the net and its
//   sel / hover / anim state read-only; while the presenter is on the board it shows the 3D view.
//   api.audience = { open(), isOpen } lets the Net tab's toolbar open the same window.
import { evaluate, formatNumber } from '../lang.js';

const STORE = 'mathboard.steps';
const CHANNEL = 'mathboard.audience';
const STEP_MS = 900;
const PRESETS = [['iso', 'Iso', 'Isometric'], ['top', 'Top', 'Top'], ['front', 'Front', 'Front'], ['side', 'Side', 'Side']];

// ================================================================ pure helpers
export const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

// A slider row (`name = <number>`) -> { name, value }; anything else -> null.
export function sliderOf(src) {
  const [r] = evaluate([String(src ?? '')]);
  return r?.name && r.slider != null ? { name: r.name, value: r.slider } : null;
}

// Same row count and colours (a missing colour in `b` matches anything): rows can be patched in place.
export function sameShape(a, b) {
  return a.length === b.length && b.every((r, i) => !r.color || r.color === a[i].color);
}

// How to get from rows `a` to rows `b`:
//   tween   only slider values (or visibility / slider ranges) differ; `tweens` = sliders to animate
//   patch   same shape, other sources differ: rewrite them in place
//   rebuild anything else
export function planTransition(a, b) {
  if (!sameShape(a, b)) return { mode: 'rebuild', tweens: [] };
  const tweens = [];
  for (let i = 0; i < a.length; i++) {
    if (a[i].src === b[i].src) continue;
    const x = sliderOf(a[i].src), y = sliderOf(b[i].src);
    if (!x || !y || x.name !== y.name) return { mode: 'patch', tweens: [] };
    if (x.value !== y.value) tweens.push({ i, name: y.name, from: x.value, to: y.value });
  }
  return { mode: 'tween', tweens };
}

// The view preset a pose looks along (its direction only, so panning keeps it), or null.
// `views`: name -> camera direction from the target, as scene.js VIEWS.
export function presetOf(pose, views) {
  if (!pose?.position || !views) return null;
  const d = pose.position.map((x, i) => x - (pose.target?.[i] ?? 0)), n = Math.hypot(...d);
  if (!(n > 0)) return null;
  for (const [name, v] of Object.entries(views)) {
    const m = Math.hypot(...v);
    if (m > 0 && d.reduce((s, x, i) => s + x * v[i], 0) / (n * m) > 0.99995) return name;
  }
  return null;
}

const near = (a, b, eps) => Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b));
const near3 = (a, b, eps) => [0, 1, 2].every(i => near(a[i], b[i], eps));
export function samePose(a, b, eps = 1e-6) {
  if (!a || !b) return a === b;
  return near3(a.position, b.position, eps) && near3(a.target || [0, 0, 0], b.target || [0, 0, 0], eps)
    && near(a.zoom ?? 1, b.zoom ?? 1, eps) && !!a.ortho === !!b.ortho
    && (a.extent == null || b.extent == null || a.extent === b.extent);
}

// Index to move to from `cur` by `d` among `n` steps, or -1 for "nowhere".
export function stepTarget(cur, d, n) {
  if (!n) return -1;
  if (cur < 0) return d > 0 ? 0 : -1;
  if (cur >= n) return n - 1;
  const j = cur + d;
  return j >= 0 && j < n ? j : -1;
}

const num3 = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);

function normCamera(c) {
  if (!c || !num3(c.position)) return null;
  const out = { position: c.position.slice(), target: num3(c.target) ? c.target.slice() : [0, 0, 0] };
  if (Number.isFinite(c.zoom) && c.zoom > 0) out.zoom = c.zoom;
  if (c.ortho != null) out.ortho = !!c.ortho;
  if (Number.isFinite(c.extent) && c.extent > 0) out.extent = c.extent;
  return out;
}

function normRow(r, i, palette) {
  const o = typeof r === 'string' ? { src: r } : r && typeof r.src === 'string' ? { src: r.src } : null;
  if (!o) return null;
  if (typeof r.color === 'string') o.color = r.color;
  else if (palette?.length) o.color = palette[i % palette.length];
  if (r.hidden) o.hidden = true;
  if (Number.isFinite(r.min)) o.min = r.min;
  if (Number.isFinite(r.max)) o.max = r.max;
  return o;
}

// Steps from storage, an imported file or a capture. Accepts `[...]` or `{ steps: [...] }`; rows may
// be plain strings. A missing camera or collapsed means "leave as is"; flat (2D) and spin default off
// so stepping back always undoes them. Throws on garbage.
export function normalizeSteps(data, palette = null) {
  const list = Array.isArray(data) ? data : data?.steps;
  if (!Array.isArray(list)) throw new Error('not a steps file');
  return list.map((s, k) => {
    if (!s || !Array.isArray(s.rows)) throw new Error(`step ${k + 1} has no rows`);
    const rows = s.rows.map((r, i) => normRow(r, i, palette)).filter(Boolean);
    return {
      title: typeof s.title === 'string' ? s.title : '',
      rows: rows.length ? rows : [{ src: '' }],
      camera: normCamera(s.camera),
      collapsed: s.collapsed == null ? null : !!s.collapsed, flat: !!s.flat, spin: !!s.spin,
    };
  });
}

// ================================================================ shared DOM helpers
const CSS = `
#lec-steps {
  display: flex; flex-direction: column; flex: none; min-height: 0; max-height: 38%;
  padding: var(--sp-8) var(--sp-8) var(--sp-8) 10px; border-bottom: 1px solid var(--line-1);
}
#lec-steps[hidden] { display: none; }
.lec-head { display: flex; gap: var(--sp-2); align-items: center; }
.lec-head .lec-sp { flex: 1; }
.lec-head .lec-cap { margin-right: var(--sp-4); }
.lec-head .ui-btn.icon { color: var(--text-3); }
.lec-head .ui-btn.icon:hover { color: var(--text-1); }
.lec-pos { min-width: 40px; text-align: center; font-size: var(--fs-sm); color: var(--text-3); font-variant-numeric: tabular-nums; }
.lec-list { list-style: none; margin: var(--sp-6) 0 0; padding: 0; overflow-y: auto; min-height: 0; }
.lec-item {
  display: grid; grid-template-columns: 26px minmax(0, 1fr) auto; gap: var(--sp-2); align-items: center;
  min-height: 28px; padding-right: 3px; border-radius: var(--r-md); cursor: pointer;
}
.lec-item + .lec-item { margin-top: 1px; }
.lec-item:hover { background: var(--hover); }
.lec-item.cur { background: var(--on); box-shadow: inset 0 0 0 1px var(--on-line); }
.lec-num { height: 22px; padding: 0; border: 0; font-size: var(--fs-sm); color: var(--text-3); font-variant-numeric: tabular-nums; }
.lec-num:hover { background: transparent; color: var(--text-1); }
.lec-item.cur .lec-num { color: var(--text-1); font-weight: var(--fw-strong); }
.lec-title { min-width: 0; width: 100%; padding: 0 6px; font-size: var(--fs-md); color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lec-item.cur .lec-title { color: var(--text-1); }
.lec-title.untitled { color: var(--text-3); }
.lec-item.cur .lec-title.untitled { color: var(--text-2); }
input.lec-title.ui-field { height: 24px; font-size: var(--fs-md); color: var(--text-1); }
.lec-acts { display: flex; opacity: 0; transition: opacity var(--dur-1) var(--ease); }
.lec-item:hover .lec-acts, .lec-item.cur .lec-acts, .lec-acts:focus-within { opacity: 1; } /* the current step keeps its actions in view */
.lec-acts .ui-btn { color: var(--text-3); }
.lec-empty { margin: var(--sp-8) 2px 2px; }
.lec-empty[hidden] { display: none; }
.lec-overlay {
  position: absolute; right: 12px; bottom: 12px; z-index: 4; pointer-events: none;
  display: flex; align-items: baseline; gap: 8px;
  max-width: 60%; overflow: hidden; white-space: nowrap;
  padding: 5px 13px; border-radius: var(--r-pill);
  background: var(--float); color: var(--text-1); border: 1px solid var(--float-line); box-shadow: var(--shadow-2);
  font-size: var(--fs-lg); line-height: var(--lh); font-variant-numeric: tabular-nums;
}
.lec-overlay[hidden] { display: none; }
.lec-ov-n { flex: none; color: var(--text-3); }
.lec-ov-t { min-width: 0; overflow: hidden; text-overflow: ellipsis; font-weight: var(--fw-medium); }
body.clean #graph { grid-template-columns: 1fr; }
body.clean #g-panel, body.clean #g-expand { display: none; }
`;

const AUDIENCE_CSS = `
body.lec-audience #tabs, body.lec-audience #toolbar, body.lec-audience #status, body.lec-audience #editor,
body.lec-audience #settings, body.lec-audience #toast, body.lec-audience #typeset, body.lec-audience #ink,
body.lec-audience #fx, body.lec-audience #g-panel, body.lec-audience #g-expand { display: none !important; }
body.lec-audience #graph { grid-template-columns: 1fr !important; }
body.lec-audience .lec-overlay { right: 16px; bottom: 16px; padding: 6px 16px; font-size: 18px; }
`;

function whenReady(fn) {
  if (document.body.dataset.graphReady === '1') fn();
  else setTimeout(() => whenReady(fn), 50);
}

// The Net tab publishes window.mathboardNet ({ ready, mirrorState, applyMirror, onMirror }) once
// its modules are installed. It may load before or after this feature, or not at all.
function whenNet(fn) {
  if (window.mathboardNet?.ready) fn(window.mathboardNet);
  else window.addEventListener('mathboard:nn-ready', () => fn(window.mathboardNet), { once: true });
}

// Rewrite the live rows to `list` (same shape) in place, keeping row ids; one recompute.
function patchRows(api, list, skip = null) {
  api.rows.forEach((r, i) => {
    const t = list[i];
    r.hidden = !!t.hidden;
    r.min = t.min;
    r.max = t.max;
    if (skip?.has(i) || r.src === t.src) return;
    r.src = t.src;
    r.el.src.value = t.src;
    r.playing = false;
  });
  api.recompute();
}

const overlayText = d => `${d.index >= 0 ? d.index + 1 : '–'} / ${d.count}${d.title ? ` · ${d.title}` : ''}`;

export function install(api) {
  api.addStyles(CSS);
  const overlay = document.createElement('div');
  overlay.className = 'lec-overlay';
  overlay.hidden = true;
  api.addOverlay(overlay);
  const paintOverlay = d => {
    overlay.hidden = !d;
    if (!d || overlay.dataset.key === overlayText(d)) return;
    overlay.dataset.key = overlayText(d);
    const part = (cls, text) => Object.assign(document.createElement('span'), { className: cls, textContent: text });
    overlay.replaceChildren(part('lec-ov-n', `${d.index >= 0 ? d.index + 1 : '–'} / ${d.count}`));
    if (d.title) overlay.append(part('lec-ov-t', d.title));
  };
  const chan = typeof BroadcastChannel === 'function' ? new BroadcastChannel(CHANNEL) : null;
  if (api.params.has('audience')) installAudience(api, chan, paintOverlay);
  else installPresenter(api, chan, paintOverlay);
}

// ================================================================ audience window
function installAudience(api, chan, paintOverlay) {
  document.body.classList.add('lec-audience');
  api.addStyles(AUDIENCE_CSS);
  let ready = false, lastRows = '';
  let want = 'graph'; // the presenter's tab: 'nn', or 'graph' for both the 3D tab and the board
  let pendingNet = null;
  const pending = {};
  const applyNet = n => {
    if (window.mathboardNet?.ready) window.mathboardNet.applyMirror(n);
    else pendingNet = { ...pendingNet, ...n };
  };
  const apply = m => {
    if (m.theme && document.documentElement.dataset.theme !== m.theme) document.documentElement.dataset.theme = m.theme;
    if (m.view) {
      want = m.view === 'nn' ? 'nn' : 'graph';
      if (api.view !== want) api.setView(want);
    }
    if (m.nn) applyNet(m.nn);
    if ('overlay' in m) paintOverlay(m.overlay);
    if (Array.isArray(m.rows) && m.rows.length) {
      const json = JSON.stringify(m.rows);
      if (json !== lastRows) {
        lastRows = json;
        if (sameShape(api.rows, m.rows)) patchRows(api, m.rows);
        else api.setRows(m.rows);
      }
    }
    const sc = api.scene;
    if (!sc) return;
    const flip = m.flat != null && sc.is2D !== m.flat;
    if (flip) sc.set2D(m.flat);
    if (m.camera && (flip || !samePose(sc.getPose(), m.camera))) sc.setPose(m.camera, 0);
  };
  if (chan) chan.onmessage = e => {
    if (ready) return apply(e.data);
    const nn = pending.nn && e.data.nn ? { ...pending.nn, ...e.data.nn } : null; // a later nn message may omit the net
    Object.assign(pending, e.data);
    if (nn) pending.nn = nn;
  };
  api.onSceneReady(sc => { sc.controls.enabled = false; });
  api.onViewChange(v => { if (v !== want) setTimeout(() => { if (api.view !== want) api.setView(want); }); });
  whenNet(net => {
    if (pendingNet) net.applyMirror(pendingNet);
    pendingNet = null;
  });
  whenReady(() => {
    if (pending.view) want = pending.view === 'nn' ? 'nn' : 'graph';
    api.setView(want);
    ready = true;
    apply(pending);
  });
}

// ================================================================ presenter window
function installPresenter(api, chan, paintOverlay) {
  (document.getElementById('g-help-keys') ?? document.getElementById('g-help'))?.insertAdjacentHTML('beforeend',
    '<div class="g-keys"><span><kbd>1</kbd>&ndash;<kbd>4</kbd></span><span>views</span><kbd>O</kbd><span>orthographic</span>' +
    '<kbd>R</kbd><span>rotate</span><kbd>H</kbd><span>hide the UI</span>' +
    '<span><kbd>&larr;</kbd> <kbd>&rarr;</kbd></span><span>steps</span><kbd>P</kbd><span>PNG</span>' +
    '<kbd>V</kbd><span>record</span></div>');
  const ic = name => api.icon?.(name) ?? '';
  let steps = [], cur = -1, open = false;
  try {
    const saved = JSON.parse(localStorage.getItem(STORE));
    if (saved) {
      steps = normalizeSteps(saved, api.PALETTE);
      cur = Number.isInteger(saved.cur) && saved.cur < steps.length ? saved.cur : -1;
      open = !!saved.open;
    }
  } catch { /* no saved steps */ }

  function saveSteps() {
    if (!api.persist) return;
    try { localStorage.setItem(STORE, JSON.stringify({ version: 1, cur, open, steps })); }
    catch { api.toast('Could not save the steps: browser storage is full. Export them.'); }
  }

  // ---------------------------------------------------------------- broadcast to the audience
  const post = msg => {
    try { chan?.postMessage(msg); } catch (err) { console.error('[lecture] broadcast:', err); }
  };
  const theme = () => document.documentElement.dataset.theme || 'dark';
  const overlayData = () => (steps.length ? { index: cur, count: steps.length, title: steps[cur]?.title || '' } : null);
  const fullState = () => {
    const sc = api.scene, m = { rows: api.getState().rows, theme: theme(), overlay: overlayData(), view: api.view };
    if (sc) Object.assign(m, { camera: sc.getPose(), flat: sc.is2D });
    if (window.mathboardNet?.ready) m.nn = window.mathboardNet.mirrorState();
    return m;
  };

  let rowsQueued = false;
  api.onRecompute(() => {
    if (rowsQueued || !chan) return;
    rowsQueued = true;
    requestAnimationFrame(() => { rowsQueued = false; post({ rows: api.getState().rows }); });
  });
  api.onViewChange(v => post({ view: v }));

  // Net tab: at most one message per frame, and the net itself only when it changed (training
  // changes it every frame; hover and selection don't).
  let netQueued = false, lastNet = '';
  whenNet(net => net.onMirror(() => {
    if (netQueued || !chan) return;
    netQueued = true;
    requestAnimationFrame(() => {
      netQueued = false;
      const m = net.mirrorState(), json = JSON.stringify(m.net);
      if (json === lastNet) delete m.net;
      else lastNet = json;
      post({ nn: m });
    });
  }));
  new MutationObserver(() => post({ theme: theme() }))
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  let lastPose = null, lastFlat = null;
  api.onSceneReady(sc => sc.onFrame(() => {
    const pose = sc.getPose();
    if (lastFlat !== sc.is2D || !samePose(pose, lastPose)) {
      lastPose = pose;
      lastFlat = sc.is2D;
      post({ camera: pose, flat: sc.is2D });
      paintPreset(pose);
    }
    sync();
  }));
  // A full snapshot now and every second, so a freshly opened or reloaded audience catches up.
  whenReady(() => {
    if (!chan) return;
    post(fullState());
    setInterval(() => post(fullState()), 1000);
  });

  let audienceWin = null;
  function openAudience() {
    if (audienceWin && !audienceWin.closed) { audienceWin.focus(); return; }
    audienceWin = window.open(new URL('index.html?audience=1', location.href).href, 'mathboard-audience', 'popup,width=1280,height=720');
    if (!audienceWin) { api.toast('The browser blocked the audience window: allow pop-ups for this page'); return; }
    if (!chan) api.toast('This browser has no BroadcastChannel: the audience window will not follow along');
    const t0 = Date.now(), win = audienceWin;
    const poll = () => {
      let up = false;
      try { up = win.document.body?.dataset.graphReady === '1'; } catch { return; }
      if (up) post(fullState());
      else if (!win.closed && Date.now() - t0 < 30000) setTimeout(poll, 150);
    };
    setTimeout(poll, 150);
    sync();
  }
  // The Net tab's toolbar opens the same window (static/nn/nn.js).
  api.audience = { open: openAudience, get isOpen() { return !!audienceWin && !audienceWin.closed; } };

  // ---------------------------------------------------------------- camera tools
  let orthoBefore2D = false;
  function setFlat(on) {
    const sc = api.scene;
    if (!sc || sc.is2D === on) return;
    if (on) { orthoBefore2D = sc.ortho; sc.setAutoRotate(false); sc.set2D(true); }
    else { sc.set2D(false); sc.setOrtho(orthoBefore2D); }
    sync();
  }
  function preset(name) {
    const sc = api.scene;
    if (!sc) return;
    if (name !== 'top') setFlat(false);
    sc.viewPreset(name);
  }
  const toggleOrtho = () => { api.scene?.setOrtho(!api.scene.ortho); sync(); };
  const toggleRotate = () => { api.scene?.setAutoRotate(!api.scene.autoRotate); sync(); };
  const toggleFlat = () => { if (api.scene) setFlat(!api.scene.is2D); };

  const bar = (label, title, onClick, opts) => api.addToolbarButton({ label, title, onClick, group: 'display', ...opts });
  // The view presets share one segmented track; the one the camera looks along is marked.
  const presetBtns = PRESETS.map(([name, label, long], i) => bar(label, `${long} view (${i + 1})`, () => preset(name),
    { group: 'camera', seg: 'view' }));
  let viewDirs = null, shownPreset;
  import('../scene.js').then(S => { viewDirs = S.VIEWS; paintPreset(api.scene?.getPose()); }).catch(() => {});
  function paintPreset(pose) {
    const name = presetOf(pose, viewDirs);
    if (name === shownPreset) return;
    shownPreset = name;
    presetBtns.forEach((b, i) => b.classList.toggle('on', PRESETS[i][0] === name));
  }
  const btn = {
    ortho: bar('Ortho', 'Orthographic projection: no perspective (O)', toggleOrtho, { group: 'camera' }),
    flat: bar('2D', '2D mode: top-down, flat, no rotation (for R² material)', toggleFlat, { group: 'camera' }),
    rotate: bar('Rotate', 'Slow auto-rotate (R)', toggleRotate),
    steps: bar('Steps', 'Lecture steps: capture, reorder, jump. ←/→ or PageUp/PageDown step through them', () => {
      open = !open;
      panel.hidden = !open;
      saveSteps();
      sync();
    }),
    audience: bar('Audience', 'Open an audience window: no UI, follows this window live in the 3D and Net tabs (drag it to the projector)',
      openAudience, { group: 'output', icon: 'audience' }),
  };
  function sync() {
    const sc = api.scene, on = (b, x) => b.classList.toggle('on', !!x);
    on(btn.ortho, sc?.ortho);
    on(btn.rotate, sc?.autoRotate);
    on(btn.flat, sc?.is2D);
    on(btn.steps, open);
    on(btn.audience, audienceWin && !audienceWin.closed);
  }

  // ---------------------------------------------------------------- steps panel
  const panel = document.createElement('div');
  panel.id = 'lec-steps';
  panel.hidden = !open;
  panel.innerHTML = `
    <div class="lec-head">
      <button class="lec-cap ui-btn sm soft" title="Capture the rows, sliders and camera as a new step after the current one">${ic('plus')}Step</button>
      <span class="lec-sp"></span>
      <button class="lec-prev ui-btn sm icon" title="Previous step (← / PageUp)">${ic('chevron-left')}</button>
      <span class="lec-pos"></span>
      <button class="lec-next ui-btn sm icon" title="Next step (→ / PageDown)">${ic('chevron-right')}</button>
      <span class="lec-sp"></span>
      <button class="lec-exp ui-btn sm icon" title="Export: download the steps as a .json file">${ic('download')}</button>
      <button class="lec-imp ui-btn sm icon" title="Import: load steps from a .json file (replaces the current ones)">${ic('upload')}</button>
      <input class="lec-file" type="file" accept=".json,application/json" hidden>
    </div>
    <ol class="lec-list"></ol>
    <p class="lec-empty ui-caption">No steps yet. Set up the rows and camera, then press <b>Step</b>.</p>`;
  const q = sel => panel.querySelector(sel);
  const list = q('.lec-list'), pos = q('.lec-pos'), empty = q('.lec-empty'), file = q('.lec-file');
  const tools = api.panelEl.querySelector('#g-tools');
  if (tools) tools.after(panel);
  else api.panelEl.querySelector('header').after(panel);

  function renderList() {
    list.textContent = '';
    steps.forEach((s, i) => {
      const li = document.createElement('li');
      li.className = 'lec-item';
      li.dataset.i = i;
      li.innerHTML = `
        <button class="lec-num" title="Go to this step">${i + 1}</button>
        <span class="lec-title" title="Click to go here, double-click to rename"></span>
        <span class="lec-acts">
          <button class="ui-btn xs icon" data-a="recap" title="Re-capture: replace this step with the current state">${ic('reset')}</button>
          <button class="ui-btn xs icon" data-a="up" title="Move up">${ic('chevron-up')}</button>
          <button class="ui-btn xs icon" data-a="down" title="Move down">${ic('chevron-down')}</button>
          <button class="ui-btn xs icon danger" data-a="del" title="Delete this step">${ic('close')}</button>
        </span>`;
      const t = li.querySelector('.lec-title');
      t.textContent = s.title || `Step ${i + 1}`;
      t.classList.toggle('untitled', !s.title);
      list.appendChild(li);
    });
    empty.hidden = steps.length > 0;
  }
  function markCur() {
    [...list.children].forEach((li, i) => li.classList.toggle('cur', i === cur));
    list.children[cur]?.scrollIntoView({ block: 'nearest' });
  }
  function renderPos() {
    const d = overlayData();
    pos.textContent = d ? `${d.index >= 0 ? d.index + 1 : '–'} / ${d.count}` : '0 / 0';
    paintOverlay(d);
    post({ overlay: d });
  }
  function changed(relist = true) {
    saveSteps();
    if (relist) renderList();
    markCur();
    renderPos();
  }

  function snapshot(title = '') {
    const s = api.getState(), sc = api.scene;
    return normalizeSteps([{ ...s, title, flat: !!sc?.is2D, spin: !!sc?.autoRotate }], api.PALETTE)[0];
  }
  function capture() {
    const i = cur < 0 ? steps.length : cur + 1;
    steps.splice(i, 0, snapshot());
    cur = i;
    changed();
    editTitle(i);
  }
  function editTitle(i) {
    const span = list.children[i]?.querySelector('span.lec-title');
    if (!span) return;
    const input = document.createElement('input');
    Object.assign(input, { className: 'lec-title ui-field sm', value: steps[i].title, placeholder: `Step ${i + 1}`, spellcheck: false, autocomplete: 'off' });
    span.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = keep => {
      if (done) return;
      done = true;
      if (keep) steps[i].title = input.value.trim();
      span.textContent = steps[i].title || `Step ${i + 1}`;
      span.classList.toggle('untitled', !steps[i].title);
      input.replaceWith(span); // in place, so a click that caused the blur still lands
      changed(false);
    };
    input.onkeydown = e => {
      if (e.key === 'Enter') finish(true);
      else if (e.key === 'Escape') finish(false);
    };
    input.onblur = () => finish(true);
  }

  let tweenRaf = 0;
  function tweenRows(plan, target, ms) {
    patchRows(api, target, new Set(plan.tweens.map(w => w.i)));
    if (!plan.tweens.length) return;
    const live = plan.tweens.map(w => ({ ...w, r: api.rows[w.i], end: target[w.i].src }));
    for (const w of live) w.r.playing = false;
    const t0 = performance.now();
    const tick = now => {
      if (!live.every(w => api.rows.includes(w.r))) { tweenRaf = 0; return; }
      const k = Math.min(1, Math.max(0, (now - t0) / ms)), e = ease(k);
      for (const w of live) {
        w.r.src = k < 1 ? `${w.name} = ${formatNumber(w.from + (w.to - w.from) * e)}` : w.end;
        w.r.el.src.value = w.r.src;
      }
      api.recompute();
      tweenRaf = k < 1 ? requestAnimationFrame(tick) : 0;
    };
    tweenRaf = requestAnimationFrame(tick);
  }

  function goTo(i, ms = STEP_MS) {
    const s = steps[i];
    if (!s) return;
    cancelAnimationFrame(tweenRaf);
    tweenRaf = 0;
    cur = i;
    const sc = api.scene;
    if (sc) { setFlat(s.flat); sc.setAutoRotate(s.spin); }
    const plan = planTransition(api.getState().rows, s.rows);
    if (plan.mode === 'patch') patchRows(api, s.rows);
    else if (plan.mode === 'tween') tweenRows(plan, s.rows, ms);
    api.setState({ rows: plan.mode === 'rebuild' ? s.rows : undefined, collapsed: s.collapsed, camera: s.camera }, { cameraMs: ms });
    sync();
    changed(false);
  }
  function step(d) {
    const j = stepTarget(cur, d, steps.length);
    if (j >= 0) goTo(j);
    else if (steps.length) api.toast(d > 0 ? 'Last step' : 'First step', 900);
  }

  function act(a, i) {
    if (a === 'recap') {
      steps[i] = snapshot(steps[i].title);
      cur = i;
      api.toast(`Step ${i + 1} re-captured`);
    } else if (a === 'del') {
      steps.splice(i, 1);
      if (cur >= i) cur--;
    } else {
      const j = a === 'up' ? i - 1 : i + 1;
      if (j < 0 || j >= steps.length) return;
      [steps[i], steps[j]] = [steps[j], steps[i]];
      if (cur === i) cur = j;
      else if (cur === j) cur = i;
    }
    changed();
  }

  list.addEventListener('click', e => {
    const li = e.target.closest('.lec-item');
    if (!li) return;
    const i = Number(li.dataset.i), b = e.target.closest('button[data-a]');
    if (b) act(b.dataset.a, i);
    else if (!e.target.matches('input')) goTo(i);
  });
  list.addEventListener('dblclick', e => {
    if (e.target.matches('span.lec-title')) editTitle(Number(e.target.closest('.lec-item').dataset.i));
  });

  function exportSteps() {
    if (!steps.length) { api.toast('No steps to export yet'); return; }
    const blob = new Blob([JSON.stringify({ format: 'mathboard-steps', version: 1, steps }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `mathboard-steps-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  file.onchange = async () => {
    const f = file.files[0];
    file.value = '';
    if (!f) return;
    let next;
    try { next = normalizeSteps(JSON.parse(await f.text()), api.PALETTE); }
    catch (err) { api.toast(`Could not import ${f.name}: ${err.message}`); return; }
    if (!next.length) { api.toast(`${f.name} has no steps`); return; }
    if (steps.length && !confirm(`Replace the ${steps.length} current steps with the ${next.length} in ${f.name}?`)) return;
    steps = next;
    cur = -1;
    renderList();
    goTo(0);
    api.toast(`Loaded ${steps.length} steps`);
  };

  q('.lec-cap').onclick = capture;
  q('.lec-prev').onclick = () => step(-1);
  q('.lec-next').onclick = () => step(1);
  q('.lec-exp').onclick = exportSteps;
  q('.lec-imp').onclick = () => file.click();

  // H hides the UI here too (the same body.clean the board's H sets, so both tabs agree).
  function toggleClean() {
    if (document.body.classList.toggle('clean')) api.toast('UI hidden. Press H to show it again', 1800);
  }

  // ---------------------------------------------------------------- keyboard (3D tab only)
  // A focused slider still lets the keys through (after dragging one, → should mean "next step").
  window.addEventListener('keydown', e => {
    if (api.view !== 'graph' || e.defaultPrevented || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t?.closest?.('input:not([type="range"]), textarea, select') || t?.isContentEditable) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    let run = null;
    if (k === 'ArrowRight' || k === 'PageDown') run = steps.length && (() => step(1));
    else if (k === 'ArrowLeft' || k === 'PageUp') run = steps.length && (() => step(-1));
    else if (k.length === 1 && '1234'.includes(k)) run = () => preset(PRESETS[Number(k) - 1][0]);
    else if (k === 'o') run = toggleOrtho;
    else if (k === 'r') run = toggleRotate;
    else if (k === 'h') run = toggleClean;
    if (!run) return;
    e.preventDefault();
    run();
  });

  renderList();
  [...list.children].forEach((li, i) => li.classList.toggle('cur', i === cur)); // the saved current step
  renderPos();
  sync();
}
