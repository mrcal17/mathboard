// Net tab shell (see docs/NN_CONTRACT.md): layout with a resizable matrix panel, the toolbar,
// the shell's keys, persistence and module loading. view, inspector, matrix, train, lens, attnviz
// and tour each export install(ctx); they are imported in parallel and installed in that order,
// and a missing or broken one is logged and skipped.
//
// Persistence: localStorage 'mathboard.nn' = { v: 1, net, split, matrixHidden }.
// Test hook: index.html#nn=<preset key> | #nn=<base64url JSON> | #nn=<URI-encoded JSON> opens the
// tab with that net and never saves. body[data-nn-ready="1"] once every module has installed.
// Audience windows (?audience) are read-only mirrors. graph/features/lecture.js carries the
// presenter's state over its BroadcastChannel using window.mathboardNet (see the end of start()).

const MODULES = ['view', 'inspector', 'matrix', 'train', 'lens', 'attnviz', 'tour', 'view3d', 'surf3d', 'flow'];
const STORE_KEY = 'mathboard.nn';
const DEFAULT_PRESET = 'xor';
const PRESET_SEED = 1;                     // presets always build the same weights; Randomize reshuffles
const NOTE_MS = 5000;                      // how long a preset's note stays up after "New net"
const SPLIT_DEFAULT = 0.4;                 // matrix panel width as a fraction of the tab
const MATRIX_MIN = 220, STAGE_MIN = 280;   // px (the CSS enforces the same limits)
const COLLAPSE_BELOW = 90;                 // px: dragging the matrix panel narrower than this hides it
const LAYER_GAP = 170;                     // px: room "+ Layer" makes between two close columns
const SNAP_MAX = 960, SNAP_QUALITY = 0.85; // "To board" image, as in graph/features/bridge.js
// The toolbar's rounded clusters, left to right ('tail' sits at the right end), and the sections
// inside each, in order. An addButton group joins the section of the same name, or GROUP_SECTION's.
const BAR = { build: ['new', 'net', 'edit'], show: ['view', 'panels', 'tour'], tail: ['file', 'tail'] };
const GROUP_SECTION = { train: 'panels', attnviz: 'panels', surf3d: 'panels' };
const BLANK_NOTE = 'An input and an output layer with no neurons: double-click to add some.';

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

// Line icons from static/icons.js (docs/DESIGN.md, Icons). An addButton icon is an icon name, an
// <svg> string or older glyph HTML; a module button whose label LABEL_ICON knows gets that line
// icon in place of a glyph, so the whole bar draws one set.
const ICONS = window.mathboardIcons || null;
const LABEL_ICON = { Weights: 'weights', Lens: 'lens', '3D': 'layers', Train: 'train', Attention: 'attention', '3D plots': 'surface', Explain: 'explain' };
const svgIcon = (name, size = 16) => ICONS?.svg(name, { size }) || '';
function iconHtml(icon, label) {
  if (typeof icon === 'string' && /^[a-z][a-z-]*$/.test(icon)) return svgIcon(icon);
  if (typeof icon === 'string' && icon.trim().startsWith('<svg')) return icon;
  const named = LABEL_ICON[String(label).replace(/<[^>]*>/g, '').trim()];
  return (named && svgIcon(named)) || icon || '';
}
// An icon-only button's label and icon: the glyph stands in if the icons did not load.
const iconOnly = (name, glyph) => (svgIcon(name) ? { label: '', icon: name } : { label: glyph });
const caret = () => `<span class="nn-caret" aria-hidden="true">${svgIcon('chevron-down', 12) || '&#9662;'}</span>`;

let toastTimer = 0;
function toast(msg, ms = 2400) {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.hidden = false;
  placeToast(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}
// In the Net tab the toast is centred over the stage (not the window), above any bar docked at the
// stage's bottom under it: the lens bar, the 3D and Flow views' bars, a card parked there. A bar is
// a shown element of the stage (or of a view covering it) that ends near the bottom and is less
// than 40% of the stage tall. nn.css reads the variables, only while the tab shows.
function placeToast(t) {
  const s = el.stage.getBoundingClientRect(), b = document.body.style;
  if (!visible() || s.width < 50) {
    for (const k of ['--nn-toast-x', '--nn-toast-bottom', '--nn-toast-max']) b.removeProperty(k);
    return;
  }
  const cx = s.left + s.width / 2, max = Math.max(200, s.width - 40);
  const half = Math.min(t.offsetWidth, max) / 2;
  let bottom = 36;
  for (const c of el.stage.querySelectorAll(':scope > *, :scope > * > *')) {
    const r = c.getBoundingClientRect();
    if (!r.width || !r.height || r.height > 0.4 * s.height || r.bottom < s.bottom - 60 || r.bottom > s.bottom + 1) continue;
    if (r.right < cx - half || r.left > cx + half) continue;
    const cs = getComputedStyle(c);
    if (cs.visibility === 'hidden' || cs.position === 'static' || +cs.opacity === 0) continue;
    bottom = Math.max(bottom, innerHeight - r.top + 12);
  }
  b.setProperty('--nn-toast-x', `${Math.round(cx)}px`);
  b.setProperty('--nn-toast-bottom', `${Math.round(bottom)}px`);
  b.setProperty('--nn-toast-max', `${Math.round(max)}px`);
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
  let matrixAway = false;   // hidden for a while by a view that shows the matrices itself (Flow); not saved
  el.root.classList.toggle('nn-audience', audience);

  // ---------------------------------------------------------------- ctx
  const showFns = new Set(), themeFns = new Set(), mirrorFns = new Set();
  const active = e => visible() && !audience && !(e && typing(e.target));
  const ctx = {
    store, model, audience, view: null,
    el: { root: el.root, bar: el.bar, stage: el.stage, matrix: el.matrix },
    addButton, toast, theme, active,
    // matrixAway(true): hide the matrix panel while a view shows the matrices itself (the Flow
    // view); matrixAway(false) brings back what the user had. Dragging the divider ends it.
    matrixAway: on => setAway(on),
    onTheme: fn => { themeFns.add(fn); return () => themeFns.delete(fn); },
    onShow: fn => { showFns.add(fn); return () => showFns.delete(fn); },
    get graph() { return window.mathboardGraph || null; },
  };

  // ---------------------------------------------------------------- toolbar
  // Rounded clusters, like the board's toolbar (BAR): build and show on the left, the tail at the
  // right end. Each addButton group joins a divider-separated section of a cluster, placed in BAR
  // order whatever the install order; a group BAR doesn't list gets a section of its own at the end
  // of show. Sections are made on first use, so none is empty.
  const clusters = {};
  for (const name of Object.keys(BAR)) {
    clusters[name] = document.createElement('div');
    clusters[name].className = 'nn-cluster';
    clusters[name].dataset.cluster = name;
  }
  el.bar.append(...Object.values(clusters));
  const sections = new Map();
  function groupEl(group) {
    const name = GROUP_SECTION[group] || group;
    let g = sections.get(name);
    if (g) return g;
    g = document.createElement('div');
    g.className = 'nn-group';
    g.dataset.group = name;
    const home = Object.keys(BAR).find(c => BAR[c].includes(name)) || 'show', order = BAR[home], at = order.indexOf(name);
    const later = s => { const i = order.indexOf(s.dataset.group); return i < 0 || i > at; };
    clusters[home].insertBefore(g, at < 0 ? null : [...clusters[home].children].find(later) || null);
    sections.set(name, g);
    return g;
  }
  // label is HTML. icon (an icons.js name, an <svg> string or glyph HTML, see iconHtml) goes before
  // the label and is dropped when the bar is short of room; with no label the button is icon-only
  // (square, and it keeps its icon), and its title is also its accessible name.
  function addButton({ label = '', title = '', onClick = null, group = 'modules', icon = '' } = {}) {
    const b = document.createElement('button');
    b.type = 'button';
    const ico = iconHtml(icon, label);
    b.innerHTML = ico ? `<span class="nn-ico" aria-hidden="true">${ico}</span>${label}` : label;
    if (ico && !label) {
      b.classList.add('nn-iconbtn');
      if (title) b.setAttribute('aria-label', title.replace(/\s*\([^)]*\)\s*$/, ''));
    }
    if (title) b.title = title;
    if (onClick) b.addEventListener('click', onClick);
    groupEl(group).appendChild(b);
    queueFit();
    return b;
  }
  // Clicking a toolbar button must not focus it, or Space (play/pause training) would click it again.
  // Nor may a click in a menu (but its search field, or the list's scrollbar): the field keeps the keys.
  el.bar.addEventListener('mousedown', e => {
    const t = e.target;
    if (t.closest('button') || (t.closest('.nn-pop') && !t.matches('input, .np-scroll'))) e.preventDefault();
  });

  // Short of room, the bar first tightens the room round its buttons (icons kept), then drops the
  // buttons' icons, then some padding, and only then wraps.
  const FIT = [[], ['nn-snug'], ['nn-compact'], ['nn-compact', 'nn-tight']];
  let fitQueued = false;
  function fitBar() {
    if (!visible()) return;   // hidden, everything measures 0
    const cs = Object.values(clusters).filter(c => c.childElementCount);
    for (const cls of FIT) {
      el.bar.classList.remove('nn-snug', 'nn-compact', 'nn-tight');
      el.bar.classList.add(...cls);
      if (cs.every(c => Math.abs(c.offsetTop - cs[0].offsetTop) < 8)) break;
    }
    el.root.style.setProperty('--nn-bar-h', `${el.bar.offsetHeight}px`);   // the cheat sheet opens under it
  }
  function queueFit() {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fitBar(); });
  }

  // ---------------------------------------------------------------- menus
  // New net and File open a popover under their button. It lives in #nn-bar, so H and the audience
  // window hide it with the bar. One is open at a time; Esc, a click outside it or on its button
  // again, or leaving the tab closes it.
  let openMenu = null;
  function makeMenu(btn, pop, opts = {}) {
    const m = { btn, pop, ...opts };   // opts: onOpen(), onKey(e) -> true if it used the key, field (its search input)
    pop.classList.add('nn-pop');
    pop.hidden = true;
    el.bar.appendChild(pop);
    btn.setAttribute('aria-haspopup', 'true');
    btn.setAttribute('aria-expanded', 'false');
    btn.addEventListener('click', () => (openMenu === m ? closeMenu() : showMenu(m)));
    return m;
  }
  function showMenu(m) {
    if (openMenu === m) return;
    closeMenu();
    toggleHelp(false);
    openMenu = m;
    m.pop.hidden = false;
    m.btn.classList.add('on');
    m.btn.setAttribute('aria-expanded', 'true');
    placeMenu();
    m.onOpen?.();
  }
  function closeMenu() {
    const m = openMenu;
    if (!m) return;
    openMenu = null;
    if (m.pop.contains(document.activeElement)) document.activeElement.blur();   // Space goes back to training
    m.pop.hidden = true;
    m.btn.classList.remove('on');
    m.btn.setAttribute('aria-expanded', 'false');
  }
  // Just under the button's cluster, lined up with its left end (its right end in the right half
  // of the window), and inside the window.
  function placeMenu() {
    const m = openMenu;
    if (!m) return;
    const c = m.btn.closest('.nn-cluster').getBoundingClientRect(), w = m.pop.offsetWidth, top = Math.round(c.bottom + 6);
    const left = c.left + c.width / 2 > innerWidth / 2 ? c.right - w : c.left;
    m.pop.style.top = `${top}px`;
    m.pop.style.left = `${Math.round(clamp(left, 12, innerWidth - w - 12))}px`;
    m.pop.style.maxHeight = `${Math.max(200, innerHeight - top - 12)}px`;
  }
  document.addEventListener('pointerdown', e => {
    if (openMenu && !openMenu.pop.contains(e.target) && !openMenu.btn.contains(e.target)) closeMenu();
  }, true);
  // An open menu has the keys first: capture phase, registered before any module's listener. Esc
  // closes it, and a key its onKey uses goes no further. A menu with a search field sends every
  // other key there, so no shortcut fires while it is open; any other menu closes and lets the key
  // through (Alt+1..3 always do).
  window.addEventListener('keydown', e => {
    const m = openMenu;
    if (!m || ['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) return;
    if (e.key === 'Escape') closeMenu();
    else if (!m.onKey?.(e)) {
      if (!m.field || e.altKey) return closeMenu();
      if (document.activeElement !== m.field) m.field.focus();   // the key then types into it
      e.stopImmediatePropagation();
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);

  // ---------------------------------------------------------------- New net: the preset menu
  // Presets in sections by p.group (in PRESETS order), in as many columns as fit, Blank last. The
  // search field filters on label, key, group and note (every word must match somewhere); the line
  // at the bottom shows the active preset's note, which is also its tooltip. Keys: typing filters,
  // ↑ ↓ (or Tab) move, ← → change column while the field is empty, Enter opens, Esc closes.
  const newBtn = addButton({ label: `New net${caret()}`, icon: svgIcon('sparkle') ? 'sparkle' : '&#10022;', group: 'new',
    title: 'Start a new net from a preset (N). Ctrl+Z goes back' });
  newBtn.classList.add('nn-new');
  const presets = Object.entries(model.PRESETS || {}).map(([key, p]) => ({ key, label: p.label || key, note: p.note || '', group: p.group || 'Other' }));
  const presetCount = presets.length;
  presets.push({ key: '-blank', label: 'Blank net', note: BLANK_NOTE, group: 'From scratch' });
  const presetSecs = new Map();
  for (const it of presets) {
    if (!presetSecs.has(it.group)) presetSecs.set(it.group, []);
    presetSecs.get(it.group).push(it);
  }
  // "XOR (2-4-1)": the trailing parenthesis in muted type.
  const presetLabel = s => {
    const m = /^(.*\S)\s+(\([^()]*\))$/.exec(s);
    return m ? `${esc(m[1])} <span class="np-aside">${esc(m[2])}</span>` : esc(s);
  };
  const presetPop = document.createElement('div');
  presetPop.className = 'nn-presets';
  presetPop.innerHTML = `
    <div class="np-top">
      <label class="np-field">${svgIcon('search')}<input class="np-find ui-field" type="text" spellcheck="false" autocomplete="off" placeholder="Search ${presetCount} presets"
        role="combobox" aria-label="Search the presets" aria-controls="np-list" aria-expanded="true" aria-autocomplete="list"></label>
      <span class="np-keys"><span><kbd>&uarr;</kbd><kbd>&darr;</kbd><kbd>&larr;</kbd><kbd>&rarr;</kbd> choose</span>
        <span><kbd>Enter</kbd> open</span> <span><kbd>Esc</kbd> close</span></span>
    </div>
    <div class="np-scroll">
      <div class="np-cols" id="np-list" role="listbox" aria-label="Presets">${[...presetSecs].map(([g, list], i) => `
        <section class="np-sec" role="group" aria-labelledby="np-h${i}"><h4 class="ui-overline" id="np-h${i}">${esc(g)}</h4>${list.map(it => `
          <button type="button" class="np-item" role="option" aria-selected="false" id="np-${esc(it.key)}" data-key="${esc(it.key)}"${it.note ? ` title="${esc(it.note)}"` : ''}>${presetLabel(it.label)}</button>`).join('')}
        </section>`).join('')}
      </div>
      <p class="np-none" hidden></p>
    </div>
    <p class="np-note"></p>`;
  const find = presetPop.querySelector('.np-find'), presetScroll = presetPop.querySelector('.np-scroll');
  const noMatch = presetPop.querySelector('.np-none'), presetNote = presetPop.querySelector('.np-note');
  const presetSecEls = [...presetPop.querySelectorAll('.np-sec')];
  const fold = s => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();   // W⁽¹⁾ -> w(1), é -> e
  const byKey = new Map(presets.map(it => [it.key, it]));
  const items = [...presetPop.querySelectorAll('.np-item')].map(b => Object.assign(byKey.get(b.dataset.key), { el: b }));   // screen order: down each column
  for (const it of items) {
    it.name = fold(`${it.label} ${it.key}`);
    it.hay = fold(`${it.label} ${it.key} ${it.group} ${it.note}`);
  }
  let act = null;
  function setActive(it, scroll = true) {
    if (act) { act.el.classList.remove('act'); act.el.setAttribute('aria-selected', 'false'); }
    act = it || null;
    if (act) {
      act.el.classList.add('act');
      act.el.setAttribute('aria-selected', 'true');
      find.setAttribute('aria-activedescendant', act.el.id);
      if (scroll) act.el.scrollIntoView({ block: 'nearest' });
    } else find.removeAttribute('aria-activedescendant');
    presetNote.innerHTML = act ? `<b>${esc(act.label)}</b> ${esc(act.note)}`
      : 'Pick a preset to start a new net: point at one to read what it shows. Ctrl+Z comes back to this net.';
  }
  // With a search, the first preset whose label or key has every word is active (else the first match).
  function filterPresets() {
    const words = fold(find.value).split(/\s+/).filter(Boolean);
    let first = null, named = null;
    for (const it of items) {
      const hit = words.every(w => it.hay.includes(w));
      it.el.hidden = !hit;
      if (!hit) continue;
      first = first || it;
      if (!named && words.every(w => it.name.includes(w))) named = it;
    }
    for (const s of presetSecEls) s.hidden = !s.querySelector('.np-item:not([hidden])');
    noMatch.hidden = !!first;
    noMatch.textContent = first ? '' : `No preset matches "${find.value.trim()}"`;
    setActive(words.length ? named || first : null);
  }
  const shownItems = () => items.filter(it => !it.el.hidden);
  function stepPreset(d) {
    const list = shownItems(), i = list.indexOf(act);
    if (list.length) setActive(list[i < 0 ? (d > 0 ? 0 : list.length - 1) : (i + d + list.length) % list.length]);
  }
  // ← →: the item in the nearest column that way, at about the same height.
  function stepColumn(d) {
    const list = shownItems();
    if (!act) return setActive(list[0]);
    const r = act.el.getBoundingClientRect();
    let best = null, bestD = Infinity;
    for (const it of list) {
      const q = it.el.getBoundingClientRect(), dx = (q.left - r.left) * d;
      if (dx < 20) continue;   // the same column, or the other way
      const dist = dx * 1e3 + Math.abs(q.top - r.top);
      if (dist < bestD) { best = it; bestD = dist; }
    }
    if (best) setActive(best);
  }
  function choosePreset(key) {
    closeMenu();
    newNet(key);
  }
  const presetMenu = makeMenu(newBtn, presetPop, {
    field: find,
    onOpen() {
      find.value = '';
      filterPresets();
      presetScroll.scrollTop = 0;
      find.focus({ preventScroll: true });
    },
    onKey(e) {
      const k = e.key;
      if (k === 'ArrowDown' || k === 'ArrowUp' || k === 'Tab') stepPreset(k === 'ArrowUp' || (k === 'Tab' && e.shiftKey) ? -1 : 1);
      else if ((k === 'ArrowLeft' || k === 'ArrowRight') && !find.value) stepColumn(k === 'ArrowRight' ? 1 : -1);
      else if (k === 'Enter') { if (act) choosePreset(act.key); }
      else return false;
      return true;
    },
  });
  find.addEventListener('input', filterPresets);
  presetPop.addEventListener('pointermove', e => {   // not pointerover: keys that scroll the list under a still mouse keep their pick
    const b = e.target.closest?.('.np-item');
    if (b && b !== act?.el) setActive(byKey.get(b.dataset.key), false);
  });
  presetPop.addEventListener('click', e => {
    const b = e.target.closest?.('.np-item');
    if (b) choosePreset(b.dataset.key);
  });

  // "+ Layer": the plus is its icon, kept when the bar drops the others (.nn-keepico).
  addButton({ ...(svgIcon('plus') ? { label: 'Layer', icon: 'plus' } : { label: '+ Layer' }), onClick: addLayer, group: 'net',
    title: 'Insert a dense hidden layer after the selected layer (or before the outputs)' }).classList.add('nn-keepico');
  addButton({ label: 'Layout', icon: 'layout', title: 'Auto layout: evenly spaced columns', onClick: autoLayout, group: 'net' });
  addButton({ label: 'Fit', icon: 'fit', title: 'Fit the network to the view (F)', onClick: () => fit(), group: 'net' });
  addButton({ label: 'Randomize', icon: 'dice', title: 'New random weights: He for ReLU nets, Xavier otherwise (Shift+click: small weights)', onClick: randomize, group: 'net' });
  const undoBtn = addButton({ ...iconOnly('undo', '&#8630;'), title: 'Undo (Ctrl+Z)', onClick: undo, group: 'edit' });
  const redoBtn = addButton({ ...iconOnly('redo', '&#8631;'), title: 'Redo (Ctrl+Y)', onClick: redo, group: 'edit' });

  // ---------------------------------------------------------------- File menu
  // ↑ ↓ and Enter work too. Each item keeps its tooltip, and shows it as its second line.
  const fileBtn = addButton({ label: `File${caret()}`, icon: 'file', group: 'file',
    title: 'Export or import the net as a .json file, or take a picture of it' });
  const filePop = document.createElement('div');
  filePop.className = 'nn-filemenu ui-menu';
  filePop.setAttribute('role', 'menu');
  const fileItems = [];
  function fileItem(label, title, fn, icon) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ui-menu-item';
    b.setAttribute('role', 'menuitem');
    b.title = title;
    b.innerHTML = `${svgIcon(icon)}<span>${label}<small>${esc(title)}</small></span>`;
    b.addEventListener('click', () => { closeMenu(); fn(); });
    fileItems.push(b);
    return b;
  }
  const fileSep = document.createElement('hr');
  fileSep.className = 'ui-menu-sep';
  filePop.append(
    fileItem('Export', 'Download the net as a .json file', exportNet, 'download'),
    fileItem('Import', 'Load a net from a .json file (Ctrl+Z goes back)', () => file.click(), 'upload'),
    fileSep,
    fileItem('PNG', 'Download a picture of the network', () => png(), 'image'),
    fileItem('To board', 'Put a picture of the network on the current board page', () => toBoard(), 'board'),
  );
  let fileAct = -1;
  const paintFileAct = i => { fileAct = i; fileItems.forEach((b, j) => b.classList.toggle('act', j === i)); };
  makeMenu(fileBtn, filePop, {
    onOpen: () => paintFileAct(-1),
    onKey(e) {
      const n = fileItems.length, d = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
      if (d) paintFileAct(fileAct < 0 ? (d > 0 ? 0 : n - 1) : (fileAct + d + n) % n);
      else if (e.key === 'Enter') { if (fileAct >= 0) fileItems[fileAct].click(); }
      else return false;
      return true;
    },
  });
  filePop.addEventListener('pointermove', e => {
    const i = fileItems.indexOf(e.target.closest?.('button'));
    if (i >= 0 && i !== fileAct) paintFileAct(i);
  });

  // The audience window belongs to graph/features/lecture.js (api.audience); it mirrors this tab too.
  const audienceBtn = addButton({
    label: 'Audience', icon: svgIcon('audience') ? 'audience' : '&#10697;', group: 'tail',
    title: 'Open an audience window: no UI, follows this window live (drag it to the projector)',
    onClick: () => {
      const a = ctx.graph?.audience;
      if (!a?.open) return toast('The audience window is not available (the 3D tab did not load)');
      a.open();
      paintAudience();
    },
  });
  audienceBtn.classList.add('nn-audience-btn');
  const paintAudience = () => audienceBtn.classList.toggle('on', !!ctx.graph?.audience?.isOpen);
  setInterval(() => { if (visible()) paintAudience(); }, 1000);   // notices the window being closed
  const helpBtn = addButton({ ...iconOnly('help', '?'), title: 'Keys and tools (?)', onClick: () => toggleHelp(), group: 'tail' });
  helpBtn.classList.add('nn-help-btn');

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
    // A preset's init recipe (meta.train.init, e.g. the word presets' W_V = I) keeps it out of known bad minima.
    const init = store.net.meta?.train?.init || null;
    if (attempt('randomize the weights', net => model.randomize(net, { seed: randomSeed(), scheme, init }))) {
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
  help.setAttribute('role', 'dialog');
  help.setAttribute('aria-label', 'Net tab: keys and tools');
  const row = (keys, what) => `<tr><td>${keys}</td><td>${what}</td></tr>`;
  // A toolbar row: the button's icon before its name, as the bar draws it.
  const tool = (icons, name, what) => row(`<span class="nn-help-tool">${[].concat(icons).map(n => svgIcon(n, 14)).join('')}${name}</span>`, what);
  const tex = s => { try { return window.katex ? window.katex.renderToString(s, { throwOnError: false }) : `<b>${esc(s)}</b>`; } catch { return `<b>${esc(s)}</b>`; } };
  help.innerHTML = `
    <div class="nn-help-head"><h3>Net tab</h3><button type="button" class="nn-help-x ui-btn sm icon" title="Close (Esc)" aria-label="Close">${svgIcon('close') || '&times;'}</button></div>
    <p class="nn-help-lead">Each layer computes ${tex('z = W a + b')}, then ${tex('a = f(z)')}. The panel on the right shows the same numbers as matrices.</p>
    <h4>Keys (not while typing)</h4>
    <table>
      ${row('<kbd>Ctrl+Z</kbd> <kbd>Ctrl+Y</kbd>', 'undo / redo')}
      ${row('<kbd>Delete</kbd>', 'remove the selected neuron, edge or layer')}
      ${row('<kbd>Esc</kbd>', 'deselect')}
      ${row('<kbd>F</kbd>', 'fit the network to the view')}
      ${row('<kbd>N</kbd>', 'New net: the preset menu; type to search, arrows and Enter to pick')}
      ${row('<kbd>H</kbd>', 'hide the toolbars; cards and plots stay, as the audience sees them')}
      ${row('<kbd>W</kbd>', 'weight labels on the edges')}
      ${row('<kbd>S</kbd> <kbd>Shift+S</kbd>', 'step through the matrix product / step back')}
      ${row('<kbd>B</kbd>', 'bias trick: fold b into W')}
      ${row('<kbd>Space</kbd> <kbd>T</kbd>', 'play / pause training, one training step')}
      ${row('<kbd>Alt+1</kbd> <kbd>2</kbd> <kbd>3</kbd>', 'Board / 3D / Net tab')}
      ${row('<kbd>?</kbd>', 'this cheat sheet')}
    </table>
    <h4>Lens, Attention, Explain</h4>
    <table>
      ${row('<kbd>1</kbd>&hellip;<kbd>9</kbd> <kbd>0</kbd>', 'follow token n (token nets); 0 stops following')}
      ${row('<kbd>[</kbd> <kbd>]</kbd>', 'focus the previous / next stage')}
      ${row('<kbd>L</kbd>', 'show / hide the lens bar')}
      ${row('<kbd>A</kbd>', 'open / close the Attention panel')}
      ${row('<kbd>M</kbd> <kbd>Shift+M</kbd>', 'next / previous Attention view: arcs, dots, mix, heat')}
      ${row('<kbd>E</kbd>', 'start / end Explain, the step-by-step walkthrough')}
      ${row('<kbd>&rarr;</kbd> <kbd>&larr;</kbd>', 'Explain: next / previous step (also <kbd>PageDown</kbd> <kbd>PageUp</kbd>)')}
      ${row('<kbd>Esc</kbd>', 'end Explain (before anything else)')}
    </table>
    <h4>3D</h4>
    <table>
      ${row('<kbd>D</kbd> <kbd>Shift+D</kbd>', '3D view on / off; next 3D view: stack, heads, tensor')}
      ${row('<kbd>&larr;</kbd> <kbd>&rarr;</kbd>', '3D tensor view: previous / next reshape step (when Explain is not running)')}
      ${row('<kbd>P</kbd> <kbd>Shift+P</kbd>', 'open / close the 3D plots panel; next plot: surface, landscape, space, simplex')}
    </table>
    <h4>Flow</h4>
    <table>
      ${row('<kbd>G</kbd> <kbd>Shift+G</kbd>', 'Flow view on / off; play the stages')}
      ${row('<kbd>&larr;</kbd> <kbd>&rarr;</kbd>', 'Flow view: previous / next stage (when Explain is not running)')}
    </table>
    <h4>Toolbar, left to right</h4>
    <p>Build the net and undo; what the view shows and the panels; then File, Audience and this sheet.</p>
    <table class="nn-help-tools">
      ${tool('sparkle', 'New net', 'the preset menu (<kbd>N</kbd>): presets by topic, a search field, Blank net last; Ctrl+Z goes back')}
      ${tool('plus', svgIcon('plus') ? 'Layer' : '+ Layer', 'insert a dense hidden layer after the selected layer, or before the outputs')}
      ${tool(['layout', 'fit'], 'Layout, Fit', 'evenly spaced columns; zoom to fit')}
      ${tool('dice', 'Randomize', 'new weights: He for ReLU nets, Xavier otherwise; Shift+click for small ones')}
      ${svgIcon('undo') ? tool(['undo', 'redo'], 'Undo, Redo', 'undo / redo') : row('&#8630; &#8631;', 'undo / redo')}
      ${tool('weights', 'Weights', 'numbers on the edges (W)')}
      ${tool('lens', 'Lens', 'focus one stage, follow a token or a head, hide weak edges; the rest dims')}
      ${tool('layers', '3D', 'the net in 3D: layers in depth, an attention layer as one slab per head, or the multi-head reshape as moving cubes')}
      ${row('Flow', 'the whole forward pass as matrix tiles, stage by stage, down to the next-word softmax; hover a cell to trace it')}
      ${tool('train', 'Train', 'the Train panel: datasets, training and plots')}
      ${tool('attention', 'Attention', 'one attention layer as arcs, dot products, the weighted sum or heatmaps')}
      ${tool('surface', '3D plots', 'a neuron as a surface over the inputs, the loss landscape with the training path, the data morphing through the layers, the softmax simplex')}
      ${tool('explain', 'Explain', 'a guided walkthrough of this net, one caption per step; it sets the lens and panels as it goes')}
      ${tool('file', 'File: Export, Import', 'the net as a .json file')}
      ${tool('file', 'File: PNG, To board', 'download a picture, or put it on the current board page')}
      ${tool('audience', 'Audience', 'a window without UI that mirrors this one live, for the projector')}
    </table>
    <h4>Mouse</h4>
    <table>
      ${row('Click', 'inspect and edit a neuron, edge or layer')}
      ${row('3D: drag, right-drag, wheel', 'orbit, pan, zoom; click a neuron or edge to inspect it')}
      ${row('Drag a neuron / header', 'move it (Alt: no snapping to the column)')}
      ${row('Drag from a neuron&rsquo;s dot', 'connect it to another neuron')}
      ${row('Double-click empty space', 'add a neuron to the nearest layer, or a layer between / beyond the columns')}
      ${row('Drag empty space, wheel', 'pan, zoom')}
      ${row('Divider', 'drag to resize the matrix panel; double-click to hide or show it')}
    </table>`;
  for (const h of help.querySelectorAll('h4')) h.classList.add('ui-overline');
  help.querySelector('.nn-help-x').addEventListener('click', () => toggleHelp(false));
  el.root.appendChild(help);
  // A toolbar row written without an icon (a module's) borrows its button's, once the bar has it.
  function helpIcons() {
    const btns = [...el.bar.querySelectorAll('.nn-cluster button')];
    for (const td of help.querySelectorAll('.nn-help-tools td:first-child')) {
      if (td.querySelector('svg, .nn-help-tool')) continue;
      const b = btns.find(x => x.textContent.trim() === td.textContent.trim() && x.querySelector('.nn-ico svg'));
      if (b) td.innerHTML = `<span class="nn-help-tool">${b.querySelector('.nn-ico').innerHTML.replace(/width="16" height="16"/, 'width="14" height="14"')}${td.innerHTML}</span>`;
    }
  }
  function toggleHelp(open = help.hidden) {
    if (open) helpIcons();
    help.hidden = !open;
    helpBtn.classList.toggle('on', open);
  }
  document.addEventListener('pointerdown', e => {
    if (!help.hidden && !help.contains(e.target) && !helpBtn.contains(e.target)) toggleHelp(false);
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
    else if (k === 'n') run = () => showMenu(presetMenu);
    else if (k === 'h') run = toggleClean;
    else if (k === '?') run = () => toggleHelp();
    if (!run) return;
    e.preventDefault();
    run();
  });

  // ---------------------------------------------------------------- splitter
  function applySplit() {
    el.main.style.setProperty('--nn-split', String(split));
    el.main.classList.toggle('matrix-hidden', matrixHidden || matrixAway);
  }
  function setSplit(f, hidden = matrixHidden) {
    if (f === split && hidden === matrixHidden && !matrixAway) return;
    split = f;
    matrixHidden = hidden;
    matrixAway = false;   // the user's own choice (or the presenter's, mirrored) ends a view's hiding
    applySplit();
    mirrorChanged();
  }
  function setAway(on) {
    if (audience || !!on === matrixAway) return;   // the audience follows the presenter's panel
    matrixAway = !!on;
    applySplit();
    mirrorChanged();
  }
  applySplit();
  el.split.setAttribute('role', 'separator');
  el.split.setAttribute('aria-orientation', 'vertical');
  el.split.setAttribute('aria-label', 'Matrix panel divider');
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
    setSplit(split, !(matrixHidden || matrixAway));
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
    fitBar();
    placeMenu();
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
      closeMenu();
    }
    for (const fn of showFns) safe(fn, v);
  }

  // ---------------------------------------------------------------- audience mirror API
  // Presenter: lecture.js calls onMirror(fn) and posts mirrorState() (at most once per frame).
  // Audience: lecture.js passes what it receives to applyMirror (ignored outside audience windows).
  // The matrix panel's toggles (ctx.matrix.opt) and the view's weight labels are UI state with no
  // store event: a click in the toolbars or the matrix panel, or a key, re-posts after it has run.
  function mirrorChanged() { for (const fn of mirrorFns) safe(fn); }
  for (const evt of ['net', 'layout', 'sel', 'hover', 'anim', 'lens', 'viz', 'tour', 'v3d', 's3d', 'flow']) store.on(evt, mirrorChanged);
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
      net: store.net, sel: store.state.sel, hover: store.state.hover, anim: store.state.anim, split, matrixHidden: matrixHidden || matrixAway,
      lens: store.state.lens, viz: store.state.viz, tour: store.state.tour, v3d: store.state.v3d, s3d: store.state.s3d, flow: store.state.flow ?? null,
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
      for (const k of ['sel', 'hover', 'anim', 'lens', 'viz', 'tour', 'v3d', 's3d', 'flow']) if (k in m && !same(m[k], store.state[k])) store.set(k, m[k] ?? null);
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
