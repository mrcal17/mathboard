// Net tab shell (see docs/NN_CONTRACT.md): layout with a resizable matrix panel, the toolbar,
// the shell's keys, persistence and module loading. view, inspector, matrix and train each
// export install(ctx); they are imported in parallel and installed in that order, and a missing
// or broken one is logged and skipped.
//
// Persistence: localStorage 'mathboard.nn' = { v: 1, net, split, matrixHidden }.
// Test hook: index.html#nn=<preset key> | #nn=<base64url JSON> | #nn=<URI-encoded JSON> opens the
// tab with that net and never saves. body[data-nn-ready="1"] once every module has installed.
// Audience windows (?audience) are read-only mirrors. graph/features/lecture.js carries the
// presenter's state over its BroadcastChannel using window.mathboardNet (see the end of start()).

const MODULES = ['view', 'inspector', 'matrix', 'train'];
const STORE_KEY = 'mathboard.nn';
const DEFAULT_PRESET = 'xor';
const PRESET_SEED = 1;                     // presets always build the same weights; Randomize reshuffles
const NOTE_MS = 5000;                      // how long a preset's note stays up after "New net"
const SPLIT_DEFAULT = 0.4;                 // matrix panel width as a fraction of the tab
const MATRIX_MIN = 220, STAGE_MIN = 280;   // px (the CSS enforces the same limits)
const COLLAPSE_BELOW = 90;                 // px: dragging the matrix panel narrower than this hides it
const LAYER_GAP = 170;                     // px: room "+ Layer" makes between two close columns
const SNAP_MAX = 960, SNAP_QUALITY = 0.85; // "To board" image, as in graph/features/bridge.js

const $ = id => document.getElementById(id);
const el = { root: $('nn'), bar: $('nn-bar'), main: $('nn-main'), stage: $('nn-stage'), split: $('nn-split'), matrix: $('nn-matrix') };
const audience = new URLSearchParams(location.search).has('audience');
const hash = new URLSearchParams(location.hash.slice(1));
const preload = hash.has('nn') ? hash.get('nn') : null;
const persist = !audience && preload == null;

const visible = () => document.body.dataset.view === 'nn';
const theme = () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);
const typing = t => !!(t?.closest?.('input:not([type="range"]):not([type="checkbox"]):not([type="radio"]):not([type="button"]), textarea, select')
  || t?.isContentEditable);
const safe = (fn, ...args) => {
  try { return fn(...args); } catch (err) { console.error('[nn] handler failed:', err); }
};

let toastTimer = 0;
function toast(msg, ms = 2400) {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

function download(href, name) {
  const a = document.createElement('a');
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

const loadImage = async src => {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
};

// ================================================================ initial net
// '#nn=xor', '#nn=<base64url of the JSON>' or '#nn=<encodeURIComponent of the JSON>'.
function decodeNet(s, model) {
  s = s.trim();
  if (model.PRESETS?.[s]) return model.PRESETS[s].build(PRESET_SEED);
  if (s.startsWith('{')) return JSON.parse(s);
  const b64 = s.replace(/ /g, '+').replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(b64 + '==='.slice((b64.length + 3) % 4));
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0))));
}

function defaultNet(model) {
  const p = model.PRESETS?.[DEFAULT_PRESET] || Object.values(model.PRESETS || {})[0];
  return p ? p.build(PRESET_SEED) : model.emptyNet();
}

// Imported or hand-written JSON may have no positions (or all the same one).
function needsLayout(net) {
  const ns = net.nodes || [];
  return ns.length > 1 && ns.every(n => !Number.isFinite(n.x) || !Number.isFinite(n.y) || (n.x === ns[0].x && n.y === ns[0].y));
}

function initialState(model) {
  let raw = null, saved = null, note = '';
  if (preload != null) {
    try { raw = decodeNet(preload, model); }
    catch (err) { note = `Could not read the #nn= net (${err.message}); showing the default`; }
  } else {
    try { saved = JSON.parse(localStorage.getItem(STORE_KEY)); } catch { /* fresh start */ }
    raw = saved?.net?.layers ? saved.net : saved?.layers ? saved : null;
  }
  let net = null;
  try { if (raw) net = model.normalize(model.clone(raw)); }
  catch (err) { note = `The saved net could not be loaded (${err.message}); showing the default`; }
  if (!net) net = model.normalize(defaultNet(model));
  return { net, note, split: saved?.split, matrixHidden: !!saved?.matrixHidden };
}

// ================================================================ start
async function boot() {
  let core;
  try {
    const [model, { createStore }] = await Promise.all([import('./model.js'), import('./store.js')]);
    core = { model, createStore };
  } catch (err) {
    console.error('[nn] core failed to load:', err);
    el.stage.innerHTML = `<p class="nn-fail">The Net tab could not start: ${esc(err.message)}</p>`;
    document.body.dataset.nnReady = 'error';
    return;
  }
  await start(core);
}

async function start({ model, createStore }) {
  const init = initialState(model);
  const store = createStore(init.net);
  let split = Number.isFinite(init.split) && init.split > 0 && init.split < 1 ? init.split : SPLIT_DEFAULT;
  let matrixHidden = init.matrixHidden;
  el.root.classList.toggle('nn-audience', audience);

  // ---------------------------------------------------------------- ctx
  const showFns = new Set(), themeFns = new Set(), mirrorFns = new Set();
  const active = e => visible() && !audience && !(e && typing(e.target));
  const ctx = {
    store, model, audience, view: null,
    el: { root: el.root, bar: el.bar, stage: el.stage, matrix: el.matrix },
    addButton, toast, theme, active,
    onTheme: fn => { themeFns.add(fn); return () => themeFns.delete(fn); },
    onShow: fn => { showFns.add(fn); return () => showFns.delete(fn); },
    get graph() { return window.mathboardGraph || null; },
  };

  // ---------------------------------------------------------------- toolbar
  // Shell groups first, module groups after them (in install order), then a spacer and '?'.
  const spacer = document.createElement('span'), tail = document.createElement('div');
  spacer.className = 'nn-spacer';
  tail.className = 'nn-group nn-tail';
  el.bar.append(spacer, tail);
  const groups = new Map();
  function groupEl(name) {
    let g = groups.get(name);
    if (!g) {
      g = document.createElement('div');
      g.className = 'nn-group';
      g.dataset.group = name;
      spacer.before(g);
      groups.set(name, g);
    }
    return g;
  }
  function addButton({ label = '', title = '', onClick = null, group = 'modules' } = {}) {
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = label;
    if (title) b.title = title;
    if (onClick) b.addEventListener('click', onClick);
    groupEl(group).appendChild(b);
    return b;
  }
  // Clicking a toolbar button must not focus it, or Space (play/pause training) would click it again.
  el.bar.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

  // Presets are grouped by p.group (sections in PRESETS order); an option's tooltip is p.note.
  const pick = document.createElement('select');
  pick.className = 'nn-pick';
  pick.title = 'Start a new net from a preset (Ctrl+Z goes back)';
  const sections = new Map();
  for (const [k, p] of Object.entries(model.PRESETS || {})) {
    const g = p.group || 'Other';
    if (!sections.has(g)) sections.set(g, []);
    sections.get(g).push(`<option value="${esc(k)}"${p.note ? ` title="${esc(p.note)}"` : ''}>${esc(p.label || k)}</option>`);
  }
  pick.innerHTML = '<option value="">New net&hellip;</option>'
    + [...sections].map(([g, opts]) => `<optgroup label="${esc(g)}">${opts.join('')}</optgroup>`).join('')
    + '<option value="-blank">Blank</option>';
  pick.onchange = () => {
    const key = pick.value;
    pick.value = '';
    pick.blur();
    if (key) newNet(key);
  };
  groupEl('net').appendChild(pick);
  addButton({ label: '+ Layer', title: 'Insert a dense hidden layer after the selected layer (or before the outputs)', onClick: addLayer, group: 'net' });
  addButton({ label: 'Layout', title: 'Auto layout: evenly spaced columns', onClick: autoLayout, group: 'net' });
  addButton({ label: 'Fit', title: 'Fit the network to the view (F)', onClick: () => fit(), group: 'net' });
  addButton({ label: 'Randomize', title: 'New random weights: He for ReLU nets, Xavier otherwise (Shift+click: small weights)', onClick: randomize, group: 'net' });
  const undoBtn = addButton({ label: '&#8630;', title: 'Undo (Ctrl+Z)', onClick: undo, group: 'edit' });
  const redoBtn = addButton({ label: '&#8631;', title: 'Redo (Ctrl+Y)', onClick: redo, group: 'edit' });
  addButton({ label: 'Export', title: 'Download the net as a .json file', onClick: exportNet, group: 'file' });
  addButton({ label: 'Import', title: 'Load a net from a .json file (Ctrl+Z goes back)', onClick: () => file.click(), group: 'file' });
  addButton({ label: 'PNG', title: 'Download a picture of the network', onClick: () => png(), group: 'file' });
  addButton({ label: 'To board', title: 'Put a picture of the network on the current board page', onClick: () => toBoard(), group: 'file' });
  // The audience window belongs to graph/features/lecture.js (api.audience); it mirrors this tab too.
  const audienceBtn = document.createElement('button');
  audienceBtn.type = 'button';
  audienceBtn.className = 'nn-audience-btn';
  audienceBtn.textContent = 'Audience';
  audienceBtn.title = 'Open an audience window: no UI, follows this window live (drag it to the projector)';
  audienceBtn.onclick = () => {
    const a = ctx.graph?.audience;
    if (!a?.open) return toast('The audience window is not available (the 3D tab did not load)');
    a.open();
    paintAudience();
  };
  const paintAudience = () => audienceBtn.classList.toggle('on', !!ctx.graph?.audience?.isOpen);
  setInterval(() => { if (visible()) paintAudience(); }, 1000);   // notices the window being closed
  const helpBtn = document.createElement('button');
  helpBtn.type = 'button';
  helpBtn.textContent = '?';
  helpBtn.title = 'Keys and tools (?)';
  helpBtn.onclick = () => toggleHelp();
  tail.append(audienceBtn, helpBtn);

  const file = document.createElement('input');
  Object.assign(file, { type: 'file', accept: '.json,application/json', hidden: true });
  el.root.appendChild(file);
  file.onchange = () => {
    const f = file.files[0];
    file.value = '';
    if (f) importFile(f);
  };

  let historyQueued = false;
  function paintHistory() {
    if (historyQueued) return;
    historyQueued = true;
    requestAnimationFrame(() => {
      historyQueued = false;
      undoBtn.disabled = !store.canUndo;
      redoBtn.disabled = !store.canRedo;
    });
  }
  store.on('net', paintHistory);
  paintHistory();

  // ---------------------------------------------------------------- actions
  const fit = (ms = 350) => ctx.view?.fit?.(ms);
  const fitSoon = (ms = 350) => requestAnimationFrame(() => fit(ms));
  function stageSize() {
    const r = el.stage.getBoundingClientRect();
    return { width: r.width > 50 ? r.width : 900, height: r.height > 50 ? r.height : 560 };
  }
  function offStage() {
    if (ctx.view?.outOfView) return ctx.view.outOfView();   // also counts the area under the train panel
    const w = el.stage.clientWidth, h = el.stage.clientHeight;
    if (!w || !h || !ctx.view?.nodeRect) return false;
    return store.net.nodes.some(n => {
      const r = ctx.view.nodeRect(n.id);
      return r && (r.x < 0 || r.y < 0 || r.x + r.w > w || r.y + r.h > h);
    });
  }

  function undo() { if (!store.undo()) toast('Nothing to undo', 900); }
  function redo() { if (!store.redo()) toast('Nothing to redo', 900); }

  // Run an edit on the live net and commit it. If it throws half way, put the net back exactly as
  // it was (in place: store.net keeps its identity) without touching the undo history.
  function attempt(what, edit) {
    const net = store.net, before = JSON.stringify(net);
    try { edit(net); }
    catch (err) {
      console.error(`[nn] ${what}:`, err);
      if (JSON.stringify(net) !== before) {
        for (const k of Object.keys(net)) delete net[k];
        Object.assign(net, JSON.parse(before));
        store.touch();
      }
      toast(`Could not ${what}: ${err.message}`);
      return false;
    }
    store.commit(what);
    return true;
  }

  // A different net: no selection, and no step-through left lit on a neuron that merely has the same index.
  function clearMarks() {
    if (store.state.sel) store.set('sel', null);
    if (store.state.anim) store.set('anim', null);
  }

  // A preset with a note toasts it for a while: what to look for in this net.
  function newNet(key) {
    let net, label, note = '';
    try {
      if (key === '-blank') { net = model.emptyNet(); label = 'blank'; }
      else {
        const p = model.PRESETS[key];
        net = p.build(PRESET_SEED);
        label = p.label || key;
        note = p.note || '';
      }
      if (needsLayout(net)) model.autoLayout(net, stageSize());
      clearMarks();
      store.load(net);
    } catch (err) {
      console.error('[nn] new net:', err);
      return toast(`Could not build that net: ${err.message}`);
    }
    fitSoon();
    if (note) toast(`${label}: ${note}`, NOTE_MS);
    else toast(`New net: ${label}. Ctrl+Z goes back`, 2200);
  }

  // Token layers keep their shape (docs/NN_ATTENTION.md). The model never refuses an edit on token
  // grounds: a token layer that loses one neuron quietly becomes a plain vector, and an attention
  // layer whose Q, K, V input breaks becomes a dense layer. So the shell refuses those edits with
  // a toast, by view.js's rules (its addFeature and insertLayer) and in the same words.
  const layerAt = l => store.net.layers[l];
  const isAttn = l => layerAt(l)?.kind === 'attention';
  const isTokenLayer = l => {
    const lay = layerAt(l);
    return !!lay && (lay.kind === 'attention' || (Number.isInteger(lay.tokens) && lay.tokens > 1)
      || (Array.isArray(lay.groups) && lay.groups.length > 0));
  };
  const plainName = s => String(s || '').replace(/\\[a-zA-Z]+\s*/g, '').replace(/[{}_^$]/g, '');
  function refuseRemove(s) {
    const net = store.net;
    if (s.kind === 'edge') {
      return model.edge(net, s.id)?.fixed ? 'Fixed edges (a residual or a pooling weight) are read-only' : null;
    }
    if (s.kind === 'layer') {
      const l = model.layerIndex(net, s.id);
      return isAttn(l + 1) ? 'Attention reads Q, K and V from this layer: delete the attention layer first' : null;
    }
    if (s.kind !== 'node') return null;
    const l = model.nodeLayerIndex(net, s.id);
    if (!isTokenLayer(l)) return null;
    const lay = layerAt(l), nm = lay.name || 'This layer', sh = model.tokenShape?.(net, l);
    if (isAttn(l)) return 'Attention has no weights of its own: Z is tokens × d_v, one row per token of V, so it can\'t lose one neuron';
    if (lay.groups?.length) return `${lay.groups.map(plainName).join(', ')} share one shape (tokens × d each), so they can't lose one neuron at a time`;
    if (isAttn(l + 1)) return 'Attention reads this layer as Q, K and V: its shape is fixed';
    const shape = sh?.d ? `${sh.tokens} tokens × ${sh.d}` : `${lay.tokens} tokens`;
    return `${nm} is ${shape}: one neuron less would break its token rows. Delete the whole layer instead`;
  }

  // A dense layer after the selected layer (by default just before the outputs). The direct edges
  // between its two neighbours would become skip edges, so they are replaced. model.addLayer puts
  // the neurons midway between the neighbouring columns; if those are too close, later columns
  // move right to make room. Never next to token layers (see refuseRemove).
  function addLayer() {
    const net = store.net, L = net.layers.length, sel = store.state.sel;
    const li = !sel ? -1 : sel.kind === 'layer' ? model.layerIndex(net, sel.id) : sel.kind === 'node' ? model.nodeLayerIndex(net, sel.id) : -1;
    const at = L < 2 ? L : li >= 0 ? clamp(li + 1, 1, L - 1) : L - 1;
    if (isAttn(at)) return toast('Attention reads Q, K and V straight from the layer before it: nothing goes between them', 4500);
    if (isTokenLayer(at - 1) || isTokenLayer(at)) return toast('Between token layers a plain dense layer would cut their shared (tied) weights, so none is added here', 4500);
    const hidden = at > 0 && at < L, before = net.layers[at - 1];
    const meanX = i => {
      const ns = model.nodesIn(net, i);
      return ns.length ? ns.reduce((s, n) => s + n.x, 0) / ns.length : null;
    };
    let id;
    const ok = attempt('add a layer', () => {
      if (hidden) {
        const a = new Set(model.nodesIn(net, at - 1).map(n => n.id)), b = new Set(model.nodesIn(net, at).map(n => n.id));
        for (const e of net.edges.filter(e => a.has(e.from) && b.has(e.to))) model.disconnect(net, e.id);
      }
      // A new hidden layer copies the activation of the hidden layer before it; else the model's default.
      const act = hidden && at > 1 && before.act !== 'softmax' ? before.act : undefined;
      id = model.addLayer(net, at, { act, size: hidden ? 3 : at === 0 ? 2 : 1, dense: true, seed: randomSeed() });
      const px = meanX(at - 1), nx = at + 1 < net.layers.length ? meanX(at + 1) : null;
      if (px === null || nx === null || nx <= px || nx - px >= 1.6 * LAYER_GAP) return;
      const shift = px + 2 * LAYER_GAP - nx;
      for (let i = at + 1; i < net.layers.length; i++) {
        for (const n of model.nodesIn(net, i)) model.setNode(net, n.id, { x: n.x + shift });
      }
      for (const n of model.nodesIn(net, at)) model.setNode(net, n.id, { x: px + LAYER_GAP });
    });
    if (!ok) return;
    store.set('sel', { kind: 'layer', id });
    requestAnimationFrame(() => { if (offStage()) fit(); });
  }

  function autoLayout() {
    if (!attempt('lay out the net', net => model.autoLayout(net, stageSize()))) return;
    store.layout();
    fitSoon();
  }

  function randomize(e) {
    const relu = store.net.layers.some((l, i) => i > 0 && (l.act === 'relu' || l.act === 'leaky'));
    const scheme = e?.shiftKey ? 'small' : relu ? 'he' : 'xavier';
    if (attempt('randomize the weights', net => model.randomize(net, { seed: randomSeed(), scheme }))) {
      toast(`New random weights (${{ he: 'He', xavier: 'Xavier', small: 'small' }[scheme]})`, 1600);
    }
  }

  function removeSelection() {
    const s = store.state.sel;
    const remove = { node: model.removeNode, edge: model.disconnect, layer: model.removeLayer }[s?.kind];
    if (s?.kind === 'layer' && store.net.layers.length <= 2) return toast('A net keeps at least an input and an output layer');
    const why = s && refuseRemove(s);
    if (why) return toast(why, 4500);
    // Deleting a hidden layer reconnects its neighbours, the reverse of "+ Layer".
    const opts = s?.kind === 'layer' ? { bridge: true, seed: randomSeed() } : undefined;
    if (!remove || !attempt(`delete that ${s.kind}`, net => remove(net, s.id, opts))) return;
    if (store.state.sel) store.set('sel', null);
  }

  const stem = () => {
    const t = String(store.net.meta?.title || '').trim().replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '');
    return `mathboard-net${t ? `-${t}` : ''}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}`;
  };

  function exportNet() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(store.net, null, 2)], { type: 'application/json' }));
    download(url, `${stem()}.json`);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function importFile(f) {
    let data;
    try { data = JSON.parse(await f.text()); }
    catch { return toast(`Could not read ${f.name}: it is not JSON`); }
    const raw = data?.net?.layers ? data.net : data;
    if (!raw || !Array.isArray(raw.layers)) return toast(`${f.name} is not a Mathboard net`);
    let problems = [];
    try { problems = model.validate(raw) || []; } catch { /* normalize repairs what it can */ }
    try {
      const net = model.normalize(model.clone(raw));
      if (needsLayout(net)) model.autoLayout(net, stageSize());
      clearMarks();
      store.load(net);
    } catch (err) {
      console.error('[nn] import:', err);
      return toast(`Could not load ${f.name}: ${err.message}`);
    }
    fitSoon();
    const fixed = problems.length ? ` (repaired: ${problems[0]}${problems.length > 1 ? `, +${problems.length - 1} more` : ''})` : '';
    toast(`Loaded ${f.name}${fixed}. Ctrl+Z goes back`, 3500);
  }

  let shooting = false;
  async function shoot(fn) {
    if (shooting) return;
    if (!ctx.view?.png) return toast('The network view is not loaded');
    shooting = true;
    try { await fn(); }
    catch (err) { console.error('[nn] snapshot:', err); toast('Snapshot failed'); }
    finally { shooting = false; }
  }
  const png = () => shoot(async () => {
    download(await ctx.view.png(2), `${stem()}.png`);
    toast('PNG saved', 1400);
  });
  // Same hand-off as the 3D tab's "To board" (graph/features/bridge.js): a JPEG on the page's
  // background colour, no bigger than SNAP_MAX, sent as `mathboard:to-board` {src, w, h} to app.js.
  const toBoard = () => shoot(async () => {
    const scale = 2, img = await loadImage(await ctx.view.png(scale));
    const w = img.naturalWidth / scale, h = img.naturalHeight / scale;
    const k = Math.min(SNAP_MAX / Math.max(w, h), window.devicePixelRatio || 1);
    const cv = document.createElement('canvas');
    cv.width = Math.max(1, Math.round(w * k));
    cv.height = Math.max(1, Math.round(h * k));
    const g = cv.getContext('2d');
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#000';
    g.fillRect(0, 0, cv.width, cv.height);
    g.drawImage(img, 0, 0, cv.width, cv.height);
    const detail = { src: cv.toDataURL('image/jpeg', SNAP_QUALITY), w, h };
    window.dispatchEvent(new CustomEvent('mathboard:to-board', { detail }));
    toast(detail.handled ? `Snapshot placed on board page ${detail.page}` : 'The board is not available', 2600);
  });

  function toggleClean() {
    if (document.body.classList.toggle('clean')) toast('UI hidden. Press H to show it again', 1800);
  }

  // ---------------------------------------------------------------- cheat sheet
  const help = document.createElement('div');
  help.id = 'nn-help';
  help.className = 'panel';
  help.hidden = true;
  const row = (keys, what) => `<tr><td>${keys}</td><td>${what}</td></tr>`;
  help.innerHTML = `
    <h3>Net tab</h3>
    <p>Each layer computes <b>z = W a + b</b>, then <b>a = f(z)</b>. The panel on the right shows the same numbers as matrices.</p>
    <h4>Keys (not while typing)</h4>
    <table>
      ${row('<kbd>Ctrl+Z</kbd> <kbd>Ctrl+Y</kbd>', 'undo / redo')}
      ${row('<kbd>Delete</kbd>', 'remove the selected neuron, edge or layer')}
      ${row('<kbd>Esc</kbd>', 'deselect')}
      ${row('<kbd>F</kbd>', 'fit the network to the view')}
      ${row('<kbd>H</kbd>', 'hide the toolbars; cards and plots stay, as the audience sees them')}
      ${row('<kbd>W</kbd>', 'weight labels on the edges')}
      ${row('<kbd>S</kbd> <kbd>Shift+S</kbd>', 'step through the matrix product / step back')}
      ${row('<kbd>B</kbd>', 'bias trick: fold b into W')}
      ${row('<kbd>Space</kbd> <kbd>T</kbd>', 'play / pause training, one training step')}
      ${row('<kbd>Alt+1</kbd> <kbd>2</kbd> <kbd>3</kbd>', 'Board / 3D / Net tab')}
    </table>
    <h4>Toolbar</h4>
    <table>
      ${row('New net', 'start from a preset (or blank); Ctrl+Z goes back')}
      ${row('+ Layer', 'insert a dense hidden layer after the selected layer, or before the outputs')}
      ${row('Layout, Fit', 'evenly spaced columns; zoom to fit')}
      ${row('Randomize', 'new weights: He for ReLU nets, Xavier otherwise; Shift+click for small ones')}
      ${row('Export, Import', 'the net as a .json file')}
      ${row('PNG, To board', 'download a picture, or put it on the current board page')}
      ${row('Audience', 'a window without UI that mirrors this one live, for the projector')}
    </table>
    <h4>Mouse</h4>
    <table>
      ${row('Click', 'inspect and edit a neuron, edge or layer')}
      ${row('Drag a neuron / header', 'move it (Alt: no snapping to the column)')}
      ${row('Drag from a neuron&rsquo;s dot', 'connect it to another neuron')}
      ${row('Double-click empty space', 'add a neuron to the nearest layer, or a layer between / beyond the columns')}
      ${row('Drag empty space, wheel', 'pan, zoom')}
      ${row('Divider', 'drag to resize the matrix panel; double-click to hide or show it')}
    </table>`;
  el.root.appendChild(help);
  function toggleHelp(open = help.hidden) {
    help.hidden = !open;
    helpBtn.classList.toggle('on', open);
  }
  document.addEventListener('pointerdown', e => {
    if (!help.hidden && !help.contains(e.target) && e.target !== helpBtn) toggleHelp(false);
  });

  // ---------------------------------------------------------------- keys (Net tab only)
  window.addEventListener('keydown', e => {
    if (!active(e)) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (k === 'z' || k === 'y')) {
      e.preventDefault();
      if (k === 'y' || e.shiftKey) redo();
      else undo();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    let run = null;
    if (k === 'Delete' || k === 'Backspace') run = store.state.sel && !e.repeat && removeSelection;
    else if (k === 'Escape') run = !help.hidden ? () => toggleHelp(false) : store.state.sel && (() => store.set('sel', null));
    else if (e.repeat) return;
    else if (k === 'f') run = () => fit();
    else if (k === 'h') run = toggleClean;
    else if (k === '?') run = () => toggleHelp();
    if (!run) return;
    e.preventDefault();
    run();
  });

  // ---------------------------------------------------------------- splitter
  function applySplit() {
    el.main.style.setProperty('--nn-split', String(split));
    el.main.classList.toggle('matrix-hidden', matrixHidden);
  }
  function setSplit(f, hidden = matrixHidden) {
    if (f === split && hidden === matrixHidden) return;
    split = f;
    matrixHidden = hidden;
    applySplit();
    mirrorChanged();
  }
  applySplit();
  el.split.addEventListener('pointerdown', e => {
    if (audience || e.button !== 0) return;
    e.preventDefault();
    getSelection()?.removeAllRanges();
    const r = el.main.getBoundingClientRect(), start = split; // hiding keeps the width from before the drag
    el.split.setPointerCapture(e.pointerId);
    el.root.classList.add('nn-resizing');
    const move = ev => {
      const w = r.right - ev.clientX;
      if (w < COLLAPSE_BELOW) return setSplit(start, true);
      const lo = Math.min(MATRIX_MIN, r.width / 2), hi = Math.max(lo, r.width - STAGE_MIN);
      setSplit(clamp(w, lo, hi) / r.width, false);
    };
    const end = () => {
      el.split.removeEventListener('pointermove', move);
      el.split.removeEventListener('pointerup', end);
      el.split.removeEventListener('pointercancel', end);
      el.root.classList.remove('nn-resizing');
      save();
    };
    el.split.addEventListener('pointermove', move);
    el.split.addEventListener('pointerup', end);
    el.split.addEventListener('pointercancel', end);
  });
  el.split.addEventListener('mousedown', e => e.preventDefault()); // a double-click would select matrix text
  el.split.addEventListener('dblclick', () => {
    if (audience) return;
    getSelection()?.removeAllRanges();
    setSplit(split, !matrixHidden);
    save();
  });

  // ---------------------------------------------------------------- persistence
  let saveTimer = 0, warned = false;
  function save() {
    if (persist && !saveTimer) saveTimer = setTimeout(flush, 500);
  }
  function flush() {
    clearTimeout(saveTimer);
    saveTimer = 0;
    if (!persist) return;
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ v: 1, net: store.net, split, matrixHidden })); }
    catch {
      if (!warned) toast('Could not save the net: browser storage is full. Export it.');
      warned = true;
    }
  }
  store.on('net', save);
  store.on('layout', save);
  window.addEventListener('pagehide', () => { if (saveTimer) flush(); });

  // ---------------------------------------------------------------- theme, visibility
  new MutationObserver(() => { const t = theme(); for (const fn of themeFns) safe(fn, t); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // The toolbar starts right of the fixed Board/3D/Net tabs.
  function placeBar() {
    const t = $('tabs')?.getBoundingClientRect();
    el.root.style.setProperty('--nn-bar-left', `${Math.round((t?.width ? t.right : 0) + 12)}px`);
  }
  placeBar();
  document.fonts?.ready.then(placeBar);
  window.addEventListener('resize', () => { if (visible()) placeBar(); });

  let shown = false, everShown = false;
  function showChanged() {
    const v = visible();
    if (v === shown) return;
    shown = v;
    if (v) {
      placeBar();
      if (!everShown) { everShown = true; fitSoon(0); } // the view may have been built into a hidden, zero-size stage
    } else {
      toggleHelp(false);
    }
    for (const fn of showFns) safe(fn, v);
  }

  // ---------------------------------------------------------------- audience mirror API
  // Presenter: lecture.js calls onMirror(fn) and posts mirrorState() (at most once per frame).
  // Audience: lecture.js passes what it receives to applyMirror (ignored outside audience windows).
  // The matrix panel's toggles (ctx.matrix.opt) and the view's weight labels are UI state with no
  // store event: a click in the toolbars or the matrix panel, or a key, re-posts after it has run.
  function mirrorChanged() { for (const fn of mirrorFns) safe(fn); }
  for (const evt of ['net', 'layout', 'sel', 'hover', 'anim']) store.on(evt, mirrorChanged);
  if (!audience) {
    const later = () => { if (mirrorFns.size) requestAnimationFrame(mirrorChanged); };
    el.bar.addEventListener('click', later);
    el.matrix.addEventListener('click', later);
    window.addEventListener('keyup', e => { if (active(e)) later(); });
  }
  let lastMirrorNet = '';
  const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  const netApi = window.mathboardNet = {
    store, ctx, audience, ready: false,
    mirrorState: () => ({
      net: store.net, sel: store.state.sel, hover: store.state.hover, anim: store.state.anim, split, matrixHidden,
      matrix: ctx.matrix?.opt ? { ...ctx.matrix.opt } : null, weights: !!ctx.view?.weights,
    }),
    applyMirror(m) {
      if (!audience || !m) return;
      if (m.net) {
        const json = JSON.stringify(m.net);
        if (json !== lastMirrorNet) {
          lastMirrorNet = json;
          try { store.load(m.net, { history: false }); } catch (err) { console.error('[nn] mirror:', err); }
        }
      }
      const opt = ctx.matrix?.opt;
      if (opt && m.matrix && Object.keys(opt).some(k => k in m.matrix && opt[k] !== m.matrix[k])) {
        for (const k of Object.keys(opt)) if (k in m.matrix) opt[k] = m.matrix[k];
        safe(() => ctx.matrix.render?.());
      }
      if ('weights' in m && ctx.view && ctx.view.weights !== !!m.weights) ctx.view.weights = !!m.weights;
      for (const k of ['sel', 'hover', 'anim']) if (k in m && !same(m[k], store.state[k])) store.set(k, m[k] ?? null);
      if (Number.isFinite(m.split)) setSplit(m.split, !!m.matrixHidden);
    },
    onMirror(fn) { mirrorFns.add(fn); return () => mirrorFns.delete(fn); },
  };

  // ---------------------------------------------------------------- modules
  const linkCss = name => {
    const href = new URL(`./${name}.css`, import.meta.url).href;
    if ([...document.querySelectorAll('link[rel="stylesheet"]')].some(l => l.href === href)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  };
  const mods = await Promise.all(MODULES.map(name => import(`./${name}.js`).catch(err => {
    if (/Failed to fetch|404|not found/i.test(String(err))) console.warn(`[nn] module ${name} is missing`);
    else console.error(`[nn] module ${name}:`, err);
    return null;
  })));
  for (const [i, m] of mods.entries()) {
    if (!m?.install) continue;
    linkCss(MODULES[i]);
    try { await m.install(ctx); } catch (err) { console.error(`[nn] module ${MODULES[i]} install:`, err); }
  }

  new MutationObserver(showChanged).observe(document.body, { attributes: true, attributeFilter: ['data-view'] });
  showChanged();
  netApi.ready = true;
  document.body.dataset.nnReady = '1';
  window.dispatchEvent(new Event('mathboard:nn-ready'));
  if (init.note) toast(init.note, 4000);
}

await boot();
