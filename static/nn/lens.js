// Net tab: the lens bar (docs/NN_LENS.md). Owns state.lens: which stage to focus, which token
// to follow, which head to keep, which edge types to show and the |w| and A thresholds. The
// rules (what is emphasized, dimmed or hidden) live in focus.js; view.js, matrix.js and
// attnviz.js draw them.
//
// The bar floats at the bottom left of #nn-stage (view.js fits the net above it) and shows only
// the controls that apply to the current net. It starts hidden: L or the toolbar's Lens button
// shows it, and that button carries a dot whenever the lens is not the default. It keeps to one
// row: short of room (the stage, or a floating panel beside it), the edge thresholds move into a
// "more" menu. It keeps the lens valid: a new net (another meta.title) starts from the default
// lens, and after any other edit fields that stop making sense are reset. Keys (ctx.active): 1-9
// follow token n, 0 stop following, [ / ] previous / next stage, L show / hide the bar. UI
// state: localStorage 'mathboard.nn.lens' = { open }.

import { DEFAULT_LENS, cleanLens, copyLens, lensInfo, stepStage, tokenLabel } from './focus.js';

export { DEFAULT_LENS };

// The current lens, complete and valid for store.net (a fresh copy: change it and store.set it).
export function lensOf(store) {
  return copyLens(cleanLens(store.net, store.state?.lens));
}

const UI_KEY = 'mathboard.nn.lens';
const PART = { scores: '1 scores', softmax: '2 softmax', mix: '3 mix' };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const isDefault = L => !L.focus && L.token === null && L.head === null && L.show.weights && L.show.attention && L.show.fixed
  && !L.minW && !L.minA;
// A line icon from static/icons.js (docs/DESIGN.md, Icons), else the glyph it replaced.
const icon = (name, alt = '') => globalThis.mathboardIcons?.svg(name) || alt;
const HEAD_VARS = ['--att', '--head-2', '--head-3', '--head-4', '--head-5', '--head-6'];   // a head keeps its colour in every panel
const CHIPS_MAX = 5;        // more tokens than this: a menu instead of chips
const EDGE = 10, GAP = 8;   // the bar's margin in the stage; its gap to a panel beside it
const BTN_TIP = 'Lens bar: focus one stage, follow a token or a head, hide edges (L)';

export function install(ctx) {
  const { store } = ctx;
  if (ctx.audience) return;   // the audience window only mirrors state.lens; the canvas draws it
  const stage = ctx.el?.stage || document.getElementById('nn-stage');

  // hidden until L or the Lens button shows it (docs/DESIGN.md C13); the choice is remembered
  let open = false;
  try { open = JSON.parse(localStorage.getItem(UI_KEY))?.open === true; } catch { /* default: hidden */ }

  const bar = document.createElement('div');
  bar.className = 'nn-lens ui-float';
  bar.setAttribute('role', 'toolbar');
  bar.setAttribute('aria-label', 'Lens');
  bar.hidden = !open;
  stage.appendChild(bar);
  // Its buttons never take focus, so Space stays with training (as the shell's toolbar).
  bar.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

  const btn = ctx.addButton?.({ label: 'Lens', icon: icon('lens', '&#9678;'), title: BTN_TIP, group: 'view', onClick: () => toggle() }) || null;

  function toggle(on = !open) {
    open = on;
    bar.hidden = !open;
    try { localStorage.setItem(UI_KEY, JSON.stringify({ open })); } catch { /* private mode */ }
    if (open) fitBar(); else setPop(false);
    paintButton();
  }
  const barShown = () => open && !document.body.classList.contains('clean');

  function set(patch) {
    const L = { ...lensOf(store), ...patch };
    store.set('lens', copyLens(cleanLens(store.net, L)));
  }

  // ---------------------------------------------------------------- build
  let info = null, shapeKey = '', wMax = 1;
  const focusValue = (L, inf = info) => {
    if (!L.focus) return '';
    const i = inf.layers.findIndex(l => l.id === L.focus.layer);
    return i < 0 ? '' : L.focus.part ? `${i}:${L.focus.part}` : String(i);
  };
  const focusText = (f, inf = info) => {
    const l = f && inf.layers.find(x => x.id === f.layer);
    if (!l) return 'whole net';
    return f.part ? `${l.name} · ${PART[f.part] || f.part}` : l.name;
  };
  const tokText = t => {
    const n = info.names[t];
    return `<span>${n ? esc(n.length > 10 ? `${n.slice(0, 9)}…` : n) : `t<sub>${t + 1}</sub>`}</span>`;   // one flex item in the chip
  };

  // One row (a .ui-float, docs/DESIGN.md C14): Lens, Focus (a menu), Token and Head (chips, or a
  // menu past CHIPS_MAX tokens), Edges (a show / hide chip per edge type, when there are two or
  // more), the |w| and A thresholds, then Clear. Groups are spaced, not ruled. fitBar() moves the
  // thresholds into the "more" menu when the row doesn't fit.
  function build() {
    const inf = info, parts = [];
    const opts = ['<option value="">Whole net</option>'];
    for (const l of inf.layers) {   // the input layer too ([ and ] skip it; Explain starts there)
      opts.push(`<option value="${l.index}">${esc(l.name)}</option>`);
      for (const p of l.parts) opts.push(`<option value="${l.index}:${esc(p)}">${esc(`${l.name} · ${PART[p] || p}`)}</option>`);
    }
    parts.push(`<label class="nl-grp" title="Focus one stage: that layer, the layers feeding it and the edges into it stay lit; the rest dims ([ and ] step through the stages)">
      <span class="nl-lab">Focus</span><select class="nl-focus ui-field sm">${opts.join('')}</select></label>`);
    const key = t => (t + 1 <= 9 ? t + 1 : '-');
    if (inf.tokens > CHIPS_MAX) {
      const o = Array.from({ length: inf.tokens }, (_, t) => `<option value="${t}">${esc(tokenLabel(store.net, t))}${t < 9 ? ` (${t + 1})` : ''}</option>`);
      parts.push(`<label class="nl-grp" title="Follow a token (keys 1 to 9, 0 for every token): its rows stay lit, and the keys and values it attends to by how much">
        <span class="nl-lab">Token</span><select class="nl-tok ui-field sm"><option value="">all (0)</option>${o.join('')}</select></label>`);
    } else if (inf.tokens > 1) {
      const chips = Array.from({ length: inf.tokens }, (_, t) => `<button type="button" class="ui-chip" data-token="${t}" title="Follow ${esc(tokenLabel(store.net, t))} (key ${key(t)}): its rows stay lit, and the keys and values it attends to by how much">${tokText(t)}</button>`);
      parts.push(`<div class="nl-grp"><span class="nl-lab">Token</span><button type="button" class="ui-chip" data-token="" title="Every token (key 0)">all</button>${chips.join('')}</div>`);
    }
    if (inf.heads > 1) {
      const sw = h => (h < HEAD_VARS.length ? `<i class="ui-sw" style="background: var(${HEAD_VARS[h]})"></i>` : '');
      const chips = Array.from({ length: inf.heads }, (_, h) => `<button type="button" class="ui-chip" data-head="${h}" title="Keep head ${h + 1}: its columns of Q, K, V and Z, and its attention">${sw(h)}${h + 1}</button>`);
      parts.push(`<div class="nl-grp"><span class="nl-lab">Head</span><button type="button" class="ui-chip" data-head="" title="Every head">all</button>${chips.join('')}</div>`);
    }
    // Edges: a show / hide chip per edge type (when there are two or more); weights and attention
    // each have their threshold.
    const kinds = [inf.hasWeights, inf.hasAttention, inf.hasFixed].filter(Boolean).length;
    const kindChip = (k, label, tip) => `<button type="button" class="ui-chip" data-show="${k}" title="${esc(tip)}: show or hide">${label}</button>`;
    if (kinds > 1) {
      const chips = [
        inf.hasWeights && kindChip('weights', 'W', 'Weight edges, the trainable W'),
        inf.hasAttention && kindChip('attention', 'A', 'Attention edges V → Z, as wide as A_ij'),
        inf.hasFixed && kindChip('fixed', 'fixed', 'Fixed edges: residuals and pooling, dashed'),
      ].filter(Boolean);
      parts.push(`<div class="nl-grp"><span class="nl-lab">Edges</span>${chips.join('')}</div>`);
    }
    const thr = (k, label, tip) => `<label class="nl-grp nl-thr" title="${esc(tip)}"><span class="nl-lab">${label} ≥</span>`
      + `<input type="range" class="ui-range" data-thr="${k}" min="0" max="1" step="0.01" value="0"><output></output></label>`;
    const thrs = [
      inf.hasWeights && thr('minW', '|w|', 'Hide weight edges with |w| below this (fixed edges stay)'),
      inf.hasAttention && thr('minA', 'A', 'Hide attention edges with A_ij below this'),
    ].filter(Boolean);
    if (thrs.length) {
      parts.push(`<div class="nl-thrs">${thrs.join('')}</div>`
        + `<div class="nl-more" hidden><button type="button" class="nl-more-btn ui-btn xs icon" title="Edge thresholds" aria-haspopup="true" aria-expanded="false">${icon('more', '&hellip;')}</button>`
        + '<div class="nl-pop ui-menu" hidden></div></div>');
    }
    parts.push('<button type="button" class="nl-clear ui-btn xs" title="The whole net again: no focus, every token and head, every edge">Clear</button>');
    bar.innerHTML = `<span class="nl-title" title="Lens: what the canvas and the matrix panel light up (L hides this bar)">Lens</span>${parts.join('')}`;
    popOpen = false;
    compactOn = false;
  }

  function render() {
    info = lensInfo(store.net);
    const key = JSON.stringify([info.tokens, info.heads, info.names.slice(0, info.tokens), info.layers.map(l => [l.id, l.name, l.kind, l.parts]),
      info.hasWeights, info.hasAttention, info.hasFixed]);
    if (key !== shapeKey) { shapeKey = key; build(); sync(); fitBar(); return; }
    sync();
  }

  // ---------------------------------------------------------------- one row
  // The room: the stage less its margins, and short of a floating panel (Train, Attention, 3D
  // plots) that reaches down beside the bar. Past it, the thresholds go into the "more" menu;
  // only when even that doesn't fit does the bar wrap. Runs when the bar is built or shown, and
  // when the stage or a panel resizes: never per frame.
  let popOpen = false, compactOn = false;
  const PANELS = ':scope > .nn-train, :scope > .nn-attnviz, :scope > .nn-s3d';
  const watched = new WeakSet();
  const sizer = typeof ResizeObserver === 'function' ? new ResizeObserver(() => fitSoon()) : null;
  sizer?.observe(stage);
  function setPop(on) {
    const pop = bar.querySelector('.nl-pop'), mb = bar.querySelector('.nl-more-btn');
    popOpen = !!on && !!pop;
    if (pop) pop.hidden = !popOpen;
    mb?.classList.toggle('on', popOpen);
    mb?.setAttribute('aria-expanded', String(popOpen));
  }
  function compact(on) {
    const thrs = bar.querySelector('.nl-thrs'), more = bar.querySelector('.nl-more');
    if (!thrs || !more) { compactOn = false; return; }
    if (on === compactOn) return;
    compactOn = on;
    if (on) more.querySelector('.nl-pop').append(thrs);
    else { more.before(thrs); setPop(false); }
    more.hidden = !on;
  }
  function fitBar() {
    if (bar.hidden || !stage.clientWidth) return;
    const s = stage.getBoundingClientRect(), h = Math.max(40, bar.offsetHeight);
    let room = s.width - 2 * EDGE;
    for (const c of stage.querySelectorAll(PANELS)) {
      if (sizer && !watched.has(c)) { watched.add(c); sizer.observe(c); }
      if (c.hidden) continue;
      const r = c.getBoundingClientRect();
      if (!r.width || r.top >= s.bottom - EDGE || r.bottom <= s.bottom - EDGE - h || r.left - s.left < EDGE + 160) continue;
      room = Math.min(room, r.left - s.left - EDGE - GAP);
    }
    room = Math.max(200, Math.floor(room));
    if (bar.style.maxWidth !== `${room}px`) bar.style.maxWidth = `${room}px`;
    bar.classList.remove('wrap');
    compact(false);
    const over = () => bar.scrollWidth > bar.clientWidth + 1;
    if (over()) compact(true);
    if (over()) bar.classList.add('wrap');
  }
  let fitRaf = 0;
  function fitSoon() {
    if (!fitRaf) fitRaf = requestAnimationFrame(() => { fitRaf = 0; fitBar(); });
  }

  // Slider range for |w|: up to the largest trainable weight, rounded up (and never below minW).
  function syncW(L) {
    const inp = bar.querySelector('[data-thr="minW"]');
    if (!inp) return;
    const m = Math.max(0.1, Math.ceil(Math.max(info.maxW, L.minW) * 10) / 10);
    if (m !== wMax || inp.max !== String(m)) { wMax = m; inp.max = String(m); }
    if (+inp.value !== L.minW) inp.value = String(L.minW);
    text(inp.nextElementSibling, L.minW ? L.minW.toFixed(2) : 'off');
    inp.disabled = !L.show.weights;
  }
  const text = (el, s) => { if (el.textContent !== s) el.textContent = s; };
  function sync() {
    const L = lensOf(store);
    const sel = bar.querySelector('.nl-focus');
    if (sel) {
      const v = focusValue(L);
      if (sel.value !== v) sel.value = v;
      sel.classList.toggle('on', !!L.focus);
      const tip = `Focus: ${focusText(L.focus)}`;
      if (sel.title !== tip) sel.title = tip;
    }
    const tok = L.token === null ? '' : String(L.token);
    // the followed token wears HI, as in the attention panel; "all" is a plain toggle
    for (const b of bar.querySelectorAll('[data-token]')) {
      b.classList.toggle('on', b.dataset.token === tok);
      b.classList.toggle('hi', tok !== '' && b.dataset.token === tok);
    }
    const ts = bar.querySelector('.nl-tok');
    if (ts) {
      if (ts.value !== tok) ts.value = tok;
      ts.classList.toggle('on', tok !== '');
    }
    for (const b of bar.querySelectorAll('[data-head]')) b.classList.toggle('on', b.dataset.head === (L.head === null ? '' : String(L.head)));
    for (const b of bar.querySelectorAll('[data-show]')) b.classList.toggle('on', !!L.show[b.dataset.show]);
    syncW(L);
    const a = bar.querySelector('[data-thr="minA"]');
    if (a) {
      if (+a.value !== L.minA) a.value = String(L.minA);
      text(a.nextElementSibling, L.minA ? L.minA.toFixed(2) : 'off');
      a.disabled = !L.show.attention;
    }
    // folded thresholds: the "more" button says when one of them is set
    bar.querySelector('.nl-more-btn')?.classList.toggle('nl-set', !!(L.minW || L.minA));
    const idle = isDefault(L);
    bar.querySelector('.nl-clear').disabled = idle;
    bar.classList.toggle('active', !idle);
    paintButton(idle, L);
  }
  // What the lens does, for the Lens button's tooltip.
  function describe(L) {
    const d = [];
    if (L.focus) d.push(`focus ${focusText(L.focus)}`);
    if (L.token !== null) d.push(`following ${tokenLabel(store.net, L.token)}`);
    if (L.head !== null) d.push(`head ${L.head + 1}`);
    const off = ['weights', 'attention', 'fixed'].filter(k => !L.show[k]);
    if (off.length) d.push(`${off.join(' and ')} edges hidden`);
    if (L.minW) d.push(`|w| ≥ ${L.minW.toFixed(2)}`);
    if (L.minA) d.push(`A ≥ ${L.minA.toFixed(2)}`);
    return d.join(', ');
  }
  function paintButton(idle = null, L = lensOf(store)) {
    if (idle === null) idle = isDefault(L);
    btn?.classList.toggle('on', open);
    btn?.classList.toggle('nl-act', !idle);   // a dot: the lens is doing something (while the bar is hidden too)
    if (btn) {
      const tip = idle || !info ? BTN_TIP : `${BTN_TIP}. Lens on: ${describe(L)}`;
      if (btn.title !== tip) btn.title = tip;
    }
  }

  // ---------------------------------------------------------------- input
  bar.addEventListener('change', e => {
    const t = e.target;
    if (t.matches('.nl-tok')) { set({ token: t.value === '' ? null : +t.value }); t.blur(); return; }
    if (!t.matches('.nl-focus')) return;
    const [i, part] = t.value.split(':');
    const l = t.value === '' ? null : info.layers[+i];
    set({ focus: l ? (part ? { layer: l.id, part } : { layer: l.id }) : null });
    t.blur();   // keys (1-9, [ ]) work again
  });
  bar.addEventListener('input', e => {
    const k = e.target.dataset?.thr;
    if (k) set({ [k]: Math.max(0, +e.target.value || 0) });
  });
  bar.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || b.disabled) return;
    if (b.classList.contains('nl-more-btn')) { setPop(!popOpen); return; }
    const L = lensOf(store), ds = b.dataset;
    if ('token' in ds) set({ token: ds.token === '' || +ds.token === L.token ? null : +ds.token });
    else if ('head' in ds) set({ head: ds.head === '' || +ds.head === L.head ? null : +ds.head });
    else if ('show' in ds) set({ show: { ...L.show, [ds.show]: !L.show[ds.show] } });
    else if (b.classList.contains('nl-clear')) store.set('lens', copyLens(DEFAULT_LENS));
  });

  // The "more" menu closes on a press outside it, and on Esc (before the shell's Esc sees it).
  document.addEventListener('pointerdown', e => { if (popOpen && !e.target.closest?.('.nl-more')) setPop(false); }, true);
  window.addEventListener('keydown', e => {
    if (!popOpen || e.key !== 'Escape') return;
    e.preventDefault();
    e.stopImmediatePropagation();
    setPop(false);
  }, true);

  window.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || !ctx.active?.(e)) return;
    const k = e.key;
    if (k.length === 1 && k >= '0' && k <= '9') {
      const inf = lensInfo(store.net);
      if (inf.tokens < 2) return;
      e.preventDefault();
      const n = +k;
      if (n > inf.tokens) { ctx.toast?.(`This net has ${inf.tokens} tokens`, 1400); return; }
      set({ token: n ? n - 1 : null });
      if (!barShown()) ctx.toast?.(n ? `Following ${tokenLabel(store.net, n - 1)}` : 'Every token', 1200);
    } else if (k === '[' || k === ']') {
      e.preventDefault();
      const f = stepStage(store.net, lensOf(store).focus, k === ']' ? 1 : -1);
      set({ focus: f });
      if (!barShown()) ctx.toast?.(`Focus: ${focusText(f, lensInfo(store.net))}`, 1400);
    } else if (k === 'l' || k === 'L') {
      e.preventDefault();
      toggle();
    }
  });

  // ---------------------------------------------------------------- keep it valid
  let title = store.net.meta?.title ?? '';
  store.on('net', p => {
    const cur = store.state.lens, now = store.net.meta?.title ?? '';
    const fresh = p?.structural && now !== title;   // a preset, an import, undo past a load
    title = now;
    if (cur && fresh && !isDefault(copyLens(cur))) store.set('lens', copyLens(DEFAULT_LENS));
    else if (cur) {
      const c = cleanLens(store.net, cur);
      if (c !== cur) store.set('lens', copyLens(c));
    }
    later();
  });
  // A panel opening or closing beside the bar changes its room (the attention panel, the 3D plots).
  store.on('viz', () => fitSoon());
  ctx.onShow?.(on => { if (on) fitSoon(); });
  // A lens set by someone else (the attention panel, the walkthrough) is cleaned the same way.
  store.on('lens', cur => {
    if (cur) {
      const c = cleanLens(store.net, cur);
      if (c !== cur) { store.set('lens', copyLens(c)); return; }
    }
    sync();
  });
  // 'net' fires every frame while training (and moves max |w|, the slider's range): the bar
  // catches up once a frame.
  let queued = false;
  function later() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; render(); });
  }
  render();
  paintButton();
}
