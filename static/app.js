'use strict';
// Mathboard: write math with the Math pen, it is grouped into expressions by proximity (the reach
// scales with your handwriting, static/board/geom.js), each expression is rasterized and sent to
// the local vision model once you move on or pause, and the handwriting is replaced in place by
// KaTeX. docs/BOARD_UX.md describes the grouping, timing and correction rules.
(() => {
  const { geom, tex } = globalThis.mathboardBoard; // static/board/*.js, loaded before this file

  // ================================================================ settings
  const DEFAULTS = {
    tool: 'math', mathColor: 'ink', drawColor: 'c1',
    delay: 1200,       // ms of mouse idle on an expression before converting it (a pen waits 75% of it)
    reach: 1.0,        // multiplier on the grouping reach (geom.js: 0.9 x and 0.35 x the writing size)
    maxFont: 110,      // px cap for typeset size (typeset is sized like the ink under it)
    inkAfter: 'hide',  // 'hide' | 'faint': handwriting under converted expressions
    theme: 'dark',
    mode: 'preview',   // 'preview' (reading under the ink, commit on intent) | 'auto' (replace in place) | 'off'
    commitOnMove: true, // Preview: writing elsewhere commits the readings you leave behind, when sure
    backend: '',       // recognizer: '' is the server's default, else one of /api/status backends
  };
  const SETTINGS_V = 2;    // v2: delay 700 -> 1200 and maxFont 56 -> 110 unless you had changed them
  const ERASER_R = 12;
  const PEN_W = 3.2;
  const RASTER_MAX = 768;  // longest side of the image sent to the model, unless symbols would get small
  const SYM_PX = 72;       // ... render so a symbol is at least this tall; the server normalizes the rest
  const MAX_AREA = 1.5 * (1 << 20); // the model's context fits about 1.5 MP (docs/BOARD_DIAGNOSIS.md 1.1)
  // Re-recognize alternates variants 1 and 2 so the model gets a different input (the same image at
  // temperature 0 gives the same answer). scale: symbol size, sent to the server as a symbol_px hint
  // (a 1.5x variant asks for symbols 1.5x the server's target); width: line width.
  const VARIANTS = [{ scale: 1, width: 1 }, { scale: 1.5, width: 1.4 }, { scale: 0.7, width: 0.8 }];
  const LASER_MS = 650;
  const COLORS = {
    dark:  { ink: '#ecebe4', c1: '#ffd166', c2: '#5ac8fa', c3: '#ff6b9a', c4: '#7ee08a', c5: '#ff9f43' },
    light: { ink: '#1d1d1f', c1: '#c98500', c2: '#0a6fd8', c3: '#d6245a', c4: '#1f9d55', c5: '#e8590c' },
  };
  const BACKEND_NAMES = { qwen: 'Qwen3-VL (general)', unimumer: 'Uni-MuMER (handwriting)', ensemble: 'Both, commit when they agree' };

  // The audience window (index.html?audience, see graph/features/lecture.js) is a read-only mirror:
  // it never recognizes (the presenter does) and never saves (its copy of the board goes stale).
  const mirror = new URLSearchParams(location.search).has('audience');
  const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
  const stored = load('mathboard.settings', {}) || {};
  if (!(stored.v >= SETTINGS_V)) { // the old defaults were saved with everything else: move them on
    if (stored.delay === 700) delete stored.delay;
    if (stored.maxFont === 56) delete stored.maxFont;
  }
  const settings = Object.assign({}, DEFAULTS, stored, { v: SETTINGS_V });
  const saveSettings = () => { if (!mirror) localStorage.setItem('mathboard.settings', JSON.stringify(settings)); };

  // ================================================================ helpers
  const $ = id => document.getElementById(id);
  const uid = () => Math.random().toString(36).slice(2, 10);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const r1 = v => Math.round(v * 10) / 10;
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const color = token => (COLORS[settings.theme] || COLORS.dark)[token] || token;
  const { grow, overlaps, inRect, unionBox } = geom;
  const shift = (r, dx, dy) => [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy];

  function strokeBox(s) {
    const u = [Infinity, Infinity, -Infinity, -Infinity];
    for (const [x, y] of s.pts) { u[0] = Math.min(u[0], x); u[1] = Math.min(u[1], y); u[2] = Math.max(u[2], x); u[3] = Math.max(u[3], y); }
    return grow(u, s.w / 2, s.w / 2);
  }
  function segDist(px, py, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
    const t = L ? clamp(((px - a[0]) * dx + (py - a[1]) * dy) / L, 0, 1) : 0;
    return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
  }
  function hitStroke(s, x, y, r) {
    const P = s.pts, rr = r + s.w / 2;
    if (P.length === 1) return Math.hypot(P[0][0] - x, P[0][1] - y) <= rr;
    for (let i = 1; i < P.length; i++) if (segDist(x, y, P[i - 1], P[i]) <= rr) return true;
    return false;
  }

  // ================================================================ board state
  // page   = { id, strokes: [stroke], blocks: [block], images: [image] }
  // stroke = { id, kind: 'math'|'draw', color, w, pr (pressure?), pts: [[x,y,p,t]], t0, box, block, conv }
  //   t: ms since the stroke started, t0: when it started (epoch ms). Both are missing in old boards.
  // block  = one math expression: { id, latex, status, ver, due, t0, box, tsBox, tsRect, keepTs, color,
  //          pv, manual, locks, movedOn, force, forceCommit, commitOnResult }
  //   status: pending -> busy -> done | empty (the model saw no math) | error (unrenderable / failed)
  //   latex: the committed reading, typeset in place. pv: a reading waiting for you, shown as a chip
  //     under the ink: { latex, cands: [{ model, latex }] (the ensemble disagreed: pick one), suggest
  //     (the group has a hand edit, so the model's reading is only a suggestion), at }
  //   manual: latex was typed by hand, so later readings never replace it on their own
  //   locks: symbols picked from the look-alike menu, re-applied to every new reading (board/tex.js)
  //   stroke.conv: the stroke is covered by the block's current typeset (so its ink is hidden)
  // image  = a picture under the ink (e.g. a 3D-tab snapshot): { id, box }. Its data URL lives in
  //   media[id], outside the page, so undo snapshots (page JSON) stay small.
  const blankPage = () => ({ id: uid(), strokes: [], blocks: [], images: [] });
  let pages = [blankPage()], pageIdx = 0;
  let media = {};          // image id -> data URL
  let active = null;       // the pointer gesture in progress
  let inflight = null;     // the recognition request in progress
  let server = { ready: false, state: 'connecting' };
  let lastMs = null, lastTokens = null, lastMath = null, penSeen = false, hover = null;
  let lastPen = settings.tool === 'draw' ? 'draw' : 'math';
  let inputType = 'mouse'; // pointerType of the last pen-down: switches timing and thresholds
  let hold = false;        // Hold (P): nothing converts until it is released or you press Enter
  let sel = null;          // a lasso selection on the current page: { ids: Set of stroke ids, box }

  const saved = load('mathboard.board', null);
  if (saved && Array.isArray(saved.pages) && saved.pages.length) {
    pages = saved.pages;
    pageIdx = clamp(saved.pageIdx | 0, 0, pages.length - 1);
    media = saved.media || {};
    for (const p of pages) {
      p.id ||= uid();
      p.images = (p.images || []).filter(im => media[im.id]);
      requeue(p);
    }
  }

  const pg = () => pages[pageIdx];
  const blockOf = (p, id) => p.blocks.find(b => b.id === id);
  const strokesOf = (p, b) => p.strokes.filter(s => s.block === b.id);
  function findBlock(id) {
    for (const p of pages) { const b = blockOf(p, id); if (b) return { p, b }; }
    return null;
  }
  function requeue(p) {
    for (const b of p.blocks) if (b.status === 'pending' || b.status === 'busy') { b.status = 'pending'; b.due = 0; }
  }
  function tsVisible(b) {
    return !!b.latex && (b.status === 'done' || (!!b.keepTs && (b.status === 'pending' || b.status === 'busy')));
  }
  const inkHidden = (s, b) => s.kind === 'math' && !!b && s.conv && tsVisible(b);
  const refreshBox = (p, b) => { b.box = unionBox(strokesOf(p, b).map(s => s.box)); };
  const removeBlock = (p, b) => p.blocks.splice(p.blocks.indexOf(b), 1);
  const waiting = b => b.status === 'pending' || b.status === 'busy';
  const mode = () => (['auto', 'preview', 'off'].includes(settings.mode) ? settings.mode : 'auto');

  // ================================================================ input profile, writing size, visible geometry
  const prof = () => geom.profile(inputType);
  const idleMs = () => Math.round(settings.delay * prof().idle);
  const knownSize = {}; // input type -> the last writing size measured from real glyphs
  // The writing size S of a page: the median height of its last glyphs (geom.writingSize), or the
  // last one measured, or the input profile's default until there are 3 glyphs.
  function pageSize(p) {
    const boxes = [];
    for (const s of p.strokes) if (s.kind === 'math' && s.box) boxes.push(s.box);
    const S = geom.writingSize(boxes.slice(-120), NaN, prof().dot);
    if (Number.isFinite(S)) return (knownSize[inputType] = S);
    return knownSize[inputType] ?? prof().size;
  }
  // The symbol size of one group (for symbol_px and the typeset fit); any glyph counts here.
  function groupSize(p, b) {
    const S = geom.writingSize(strokesOf(p, b).map(s => s.box), NaN, prof().dot, 1);
    return Number.isFinite(S) ? S : pageSize(p);
  }
  // What you see of a group: its typeset (plus any ink not yet converted) when the typeset shows,
  // otherwise its ink. Grouping, the outlines and the chips all use this.
  function hitBox(p, b, strokes = strokesOf(p, b)) {
    if (tsVisible(b) && b.tsRect) return unionBox([b.tsRect, ...strokes.filter(s => !s.conv).map(s => s.box)]);
    return b.box;
  }
  // The groups of a page as geom.joinTargets wants them: visible box and visible fraction bars.
  function groupsOf(p, S) {
    const by = new Map(p.blocks.map(b => [b.id, []]));
    for (const s of p.strokes) if (s.block && by.has(s.block)) by.get(s.block).push(s);
    return p.blocks.filter(b => b.box).map(b => {
      const ss = by.get(b.id);
      return { id: b.id, box: hitBox(p, b, ss), bars: ss.filter(s => !inkHidden(s, b) && geom.isBar(s.box, S)).map(s => s.box) };
    });
  }
  const burstId = () => (lastMath && Date.now() - lastMath.t < prof().burst ? lastMath.id : null);
  // S is measured before the new stroke counts, so a small exponent doesn't shrink its own reach.
  function joinIds(p, sbox, S = pageSize(p)) {
    return geom.joinTargets(groupsOf(p, S), sbox, S, settings.reach, { burst: burstId() });
  }

  // Mark an expression as changed. keepTs: leave the old typeset up (dimmed) while it re-converts;
  // otherwise its handwriting is revealed, and the old typeset (and a hand edit) is dropped: the ink
  // is what counts until the new reading arrives.
  function touch(p, b, keepTs) {
    const shown = tsVisible(b);
    b.ver = uid();
    b.status = 'pending';
    b.t0 = Date.now();
    b.due = b.t0 + idleMs();
    b.errors = 0;
    b.movedOn = b.force = b.forceCommit = b.commitOnResult = false;
    b.keepTs = keepTs && shown;
    if (!b.keepTs) {
      b.latex = '';
      b.manual = false;
      for (const s of strokesOf(p, b)) s.conv = false;
    }
  }

  // ================================================================ undo (page snapshots)
  const undoStack = [], redoStack = [];
  const snapshot = () => ({ id: pg().id, data: JSON.stringify(pg()) });
  function commitUndo(snap) {
    undoStack.push(snap);
    if (undoStack.length > 300) undoStack.shift();
    redoStack.length = 0;
  }
  function markUndo() { // first real change of the current gesture
    if (active && !active.undone) { commitUndo(active.snap); active.undone = true; }
  }
  function stepHistory(from, to) {
    if (active) return;
    const e = from.pop();
    if (!e) return;
    const i = pages.findIndex(p => p.id === e.id);
    if (i < 0) return;
    to.push({ id: e.id, data: JSON.stringify(pages[i]) });
    pages[i] = JSON.parse(e.data);
    requeue(pages[i]);
    pageIdx = i;
    closePanels();
    clearSel();
    refreshAll();
    scheduleSave();
  }
  const undo = () => stepHistory(undoStack, redoStack);
  const redo = () => stepHistory(redoStack, undoStack);

  // ================================================================ ink rendering
  const inkCv = $('ink'), fxCv = $('fx'), tsLayer = $('typeset'), pvLayer = $('previews');
  const ctx = inkCv.getContext('2d'), fx = fxCv.getContext('2d');
  let W = 0, H = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = innerWidth; H = innerHeight;
    for (const c of [inkCv, fxCv]) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
      // Pin the on-screen size: a canvas otherwise displays at its pixel size, which with
      // display scaling (dpr > 1) makes ink land further and further from the pointer.
      c.style.width = W + 'px';
      c.style.height = H + 'px';
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  }

  const penWidth = (s, p) => (s.pr ? s.w * (0.45 + 0.8 * p) : s.w);

  // One smoothed piece of a stroke, around point i (quadratic through midpoints).
  // Live drawing and full redraws share it, so ink doesn't shift when the pen lifts.
  function drawSeg(c, s, i) {
    const P = s.pts, a = P[i - 1], b = P[i], n = P[i + 1];
    const m1 = i === 1 ? a : mid(a, b), m2 = n ? mid(b, n) : b;
    c.lineWidth = penWidth(s, b[2]);
    c.beginPath();
    c.moveTo(m1[0], m1[1]);
    c.quadraticCurveTo(b[0], b[1], m2[0], m2[1]);
    c.stroke();
  }
  function drawStroke(c, s, alpha = 1) {
    const P = s.pts;
    c.globalAlpha = alpha;
    c.strokeStyle = c.fillStyle = color(s.color);
    if (P.length === 1) {
      c.beginPath(); c.arc(P[0][0], P[0][1], penWidth(s, P[0][2]) / 2, 0, 7); c.fill();
    } else {
      for (let i = 1; i < P.length; i++) drawSeg(c, s, i);
    }
    c.globalAlpha = 1;
  }

  // The outlines' colours are tokens (style.css): --line-3 waiting, --danger failed, --text-3 the
  // group a stroke will join, --hi a selection. Read once per theme.
  let outlineUi = null;
  function outlineColors() {
    if (outlineUi?.theme !== settings.theme) {
      const cs = getComputedStyle(document.documentElement), v = n => cs.getPropertyValue(n).trim();
      outlineUi = {
        theme: settings.theme, wait: v('--line-3') || 'rgba(127,127,127,.3)', error: v('--danger') || '#e5484d',
        join: v('--text-3') || '#939ca2', hi: v('--hi') || '#ffd54a',
      };
    }
    return outlineUi;
  }
  function outline(c, r, pad = 6) {
    const g = grow(r, pad, pad);
    c.beginPath();
    c.roundRect(g[0], g[1], g[2] - g[0], g[3] - g[1], 8);
    c.stroke();
  }

  function redraw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = ctx.lineJoin = 'round';
    const p = pg(), bm = new Map(p.blocks.map(b => [b.id, b]));
    const faint = settings.inkAfter === 'faint';
    if (sel && !document.body.classList.contains('clean')) drawSelection(p, bm);
    for (const s of p.strokes) {
      if (inkHidden(s, s.block && bm.get(s.block))) { if (faint) drawStroke(ctx, s, 0.16); continue; }
      drawStroke(ctx, s);
    }
    if (active?.stroke) drawStroke(ctx, active.stroke);

    if (!document.body.classList.contains('clean')) { // faint outline = "converting soon"
      const ui = outlineColors();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      for (const b of p.blocks) {
        if (!b.box || (!waiting(b) && b.status !== 'error')) continue;
        ctx.strokeStyle = b.status === 'error' ? ui.error : ui.wait;
        ctx.globalAlpha = b.status === 'error' ? 0.8 : 1;
        outline(ctx, hitBox(p, b));
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }
  }
  // A lasso selection: an --hi halo under the selected ink (a tint over selected typeset) and the
  // selection's box, 2 px dashed --text-3 (docs/DESIGN.md E7).
  function drawSelection(p, bm) {
    const ui = outlineColors();
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = ui.hi;
    const tinted = new Set();
    for (const s of p.strokes) {
      if (!sel.ids.has(s.id)) continue;
      const b = s.block && bm.get(s.block);
      if (inkHidden(s, b)) {
        if (b.tsRect && !tinted.has(b.id)) {
          tinted.add(b.id);
          const r = grow(b.tsRect, 4, 4);
          ctx.globalAlpha = 0.16;
          ctx.beginPath(); ctx.roundRect(r[0], r[1], r[2] - r[0], r[3] - r[1], 6); ctx.fill();
        }
        continue;
      }
      ctx.globalAlpha = 0.32;
      ctx.lineWidth = s.w + 8;
      if (s.pts.length === 1) { ctx.beginPath(); ctx.arc(s.pts[0][0], s.pts[0][1], (s.w + 8) / 2, 0, 7); ctx.fill(); }
      else { tracePath(ctx, s); ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = ui.join;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    outline(ctx, sel.box, 10);
    ctx.restore();
  }

  // ================================================================ typeset rendering
  const htmlCache = new Map();
  function balanceBraces(src) {
    let depth = 0, out = '';
    for (let i = 0; i < src.length; i++) {
      const ch = src[i];
      if (ch === '\\') { out += ch + (src[i + 1] ?? ''); i++; continue; }
      if (ch === '}') { if (!depth) continue; depth--; }
      if (ch === '{') depth++;
      out += ch;
    }
    return out + '}'.repeat(depth);
  }
  const KATEX = { displayMode: true, throwOnError: true, strict: 'ignore', output: 'html' };
  function toHTML(src) { // null if KaTeX can't render it (after a couple of repairs)
    if (htmlCache.has(src)) return htmlCache.get(src);
    const stacked = tex.stackLines(src); // lines on one line in KaTeX: stack them
    const tries = stacked ? [stacked, src] : [src];
    const bal = balanceBraces(src);
    if (bal !== src) tries.push(bal);
    let html = null;
    for (const t of tries) {
      try { html = katex.renderToString(t, KATEX); break; }
      catch { /* try the next repair */ }
    }
    if (htmlCache.size > 500) htmlCache.clear();
    htmlCache.set(src, html);
    return html;
  }
  // The same, with each tappable symbol wrapped in \htmlData{mbt=K} (board/tex.js scan), so a tap on
  // the typeset finds its token. KaTeX's trust is limited to \htmlData. null when it doesn't render;
  // the typeset then falls back to toHTML and simply isn't tappable.
  const tapCache = new Map();
  const TAP_KATEX = { ...KATEX, trust: c => c.command === '\\htmlData' };
  function tapHTML(src) {
    if (tapCache.has(src)) return tapCache.get(src);
    const { tex: wrapped } = tex.scan(tex.stackLines(src) || src); // stacking adds no tappable tokens
    let html = null;
    if (wrapped) try { html = katex.renderToString(wrapped, TAP_KATEX); } catch { /* not tappable */ }
    if (tapCache.size > 500) tapCache.clear();
    tapCache.set(src, html);
    return html;
  }

  const tsEls = new Map(); // block id -> element

  // The glyphs' own box inside a rendered element, relative to the element's top-left corner: the
  // width of the KaTeX bases and the height of their struts (a strut spans the glyphs' height and
  // depth; a base's own box also holds the 1.2 line height, which made the typeset half the ink's size).
  function glyphArea(el) {
    const e = el.getBoundingClientRect(), bases = el.querySelectorAll('.katex-html > .base');
    if (!bases.length) return { x: 0, y: 0, w: e.width, h: e.height };
    const rect = n => { const r = n.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
    const u = unionBox([...bases].map(rect)), struts = el.querySelectorAll('.katex-html > .base > .strut');
    if (struts.length) { const v = unionBox([...struts].map(rect)); if (v[3] > v[1]) { u[1] = v[1]; u[3] = v[3]; } }
    return { x: u[0] - e.left, y: u[1] - e.top, w: u[2] - u[0], h: u[3] - u[1] };
  }

  // Size the typeset like the handwriting under it: its glyphs as tall as the ink (a flat stroke like
  // = counts as at most three quarters of a symbol) and no wider than 1.1x the ink, capped by maxFont.
  function paintTypeset(el, b, box, S) {
    el.innerHTML = tapHTML(b.latex) || toHTML(b.latex) || '';
    el.style.color = color(b.color);
    el.style.fontSize = '40px';
    let m = glyphArea(el);
    const bw = Math.max(box[2] - box[0], 24), bh = Math.max(box[3] - box[1], 0.75 * S, 24);
    const fit = Math.min(bh / (m.h || 1), (bw * 1.1) / (m.w || 1));
    el.style.fontSize = clamp(40 * fit, 14, settings.maxFont) + 'px';
    m = glyphArea(el);
    el._m = m;
    if (!m.w) el._key = null; // board hidden (3D tab): measure again once it is shown
  }

  function renderTypeset() {
    const p = pg(), live = new Set();
    for (const b of p.blocks) {
      if (!b.box || !tsVisible(b)) continue;
      const box = b.tsBox || b.box; // geometry of the last conversion, so re-converting doesn't jump
      live.add(b.id);
      let el = tsEls.get(b.id);
      if (!el) {
        el = document.createElement('div');
        el.className = 'ts';
        tsLayer.appendChild(el);
        tsEls.set(b.id, el);
      }
      const key = [b.latex, Math.round(box[2] - box[0]), Math.round(box[3] - box[1]), b.color, settings.maxFont, settings.theme].join('|');
      if (el._key !== key) { el._key = key; paintTypeset(el, b, box, groupSize(p, b)); }
      // The glyphs start at the ink's left edge and are centred on it vertically.
      const m = el._m, cy = (box[1] + box[3]) / 2;
      el.style.transform = `translate(${box[0] - m.x}px, ${cy - m.h / 2 - m.y}px)`;
      el.classList.toggle('stale', b.status !== 'done');
      b.tsRect = [box[0], cy - m.h / 2, box[0] + m.w, cy + m.h / 2].map(r1);
    }
    for (const [id, el] of tsEls) if (!live.has(id)) { el.remove(); tsEls.delete(id); }
  }

  // ================================================================ reading chips (under the ink)
  // A group with a reading waiting for you (block.pv) gets a chip: click it (or press Enter) to
  // commit, Esc to keep the ink. When the ensemble disagrees it shows both readings to pick from
  // (1, 2); for a hand-edited group it is a suggestion ("Use ...").
  const pvEls = new Map(); // block id -> chip
  let hoverChip = null;    // the chip under the pointer: Esc and 1 / 2 act on it
  pvLayer.addEventListener('pointerover', e => { hoverChip = e.target.closest('.pv')?.dataset.id || null; });
  pvLayer.addEventListener('pointerout', e => { if (!e.relatedTarget?.closest?.('.pv')) hoverChip = null; });
  // A chip sits where the next line goes: a pen stroke that starts on it is drawn on the board (the
  // pointer is captured by #fx), and only a tap on it counts as a click (onUp).
  pvLayer.addEventListener('pointerdown', e => {
    const btn = e.target.closest('.pv button');
    if (!btn || active || (settings.tool !== 'math' && settings.tool !== 'draw') || e.button !== 0) return;
    e.preventDefault(); // no focus and no click on the button: the gesture is the board's now
    onDown(e);
    if (active) active.chipBtn = btn;
  });
  function renderPreviews() {
    const p = pg(), live = new Set();
    for (const b of p.blocks) {
      if (!b.pv || !b.box) continue;
      live.add(b.id);
      let el = pvEls.get(b.id);
      if (!el) {
        el = document.createElement('div');
        el.className = 'pv';
        el.dataset.id = b.id;
        pvLayer.appendChild(el);
        pvEls.set(b.id, el);
      }
      const S = groupSize(p, b), fs = Math.round(clamp(0.4 * S, 16, 30)); // smaller than the ink
      const key = [JSON.stringify(b.pv), fs, b.color, settings.theme].join('|');
      if (el._key !== key) { el._key = key; paintChip(el, b, fs); }
      el.classList.toggle('stale', waiting(b));
      const r = hitBox(p, b), cw = el.offsetWidth, chh = el.offsetHeight;
      let top = r[3] + 10;
      if (top + chh > H - 8) top = Math.max(8, r[1] - chh - 10);
      el.style.transform = `translate(${Math.round(clamp(r[0], 8, W - cw - 8))}px, ${Math.round(top)}px)`;
    }
    for (const [id, el] of pvEls) if (!live.has(id)) { el.remove(); pvEls.delete(id); }
  }
  function paintChip(el, b, fs) {
    const pv = b.pv, choice = pv.cands?.length > 1, opts = choice ? pv.cands : [{ latex: pv.latex, backend: pv.by }];
    el.textContent = '';
    el.style.fontSize = fs + 'px';
    el.classList.toggle('choice', choice);
    const label = pv.suggest ? 'Use' : choice ? 'Pick one' : pv.unsure ? 'Unsure' : '';
    if (label) {
      const lbl = document.createElement('span');
      lbl.className = 'pv-lbl';
      lbl.textContent = label;
      el.appendChild(lbl);
    }
    opts.forEach((c, i) => {
      const btn = document.createElement('button');
      btn.className = 'pv-opt';
      btn.style.color = color(b.color);
      btn.innerHTML = toHTML(c.latex) || '';
      if (choice || pv.unsure) {
        const tag = document.createElement('small');
        tag.textContent = choice ? `${backendShort(c.backend) || 'reading'} · ${i + 1}` : `${backendShort(c.backend) || 'one model'} only`;
        btn.appendChild(tag);
      }
      btn.title = choice ? `Use this reading (${i + 1})${i ? '' : '. The server\'s pick'}`
        : pv.suggest ? 'Use the model\'s reading instead of your edit'
          : pv.unsure ? 'Only one recognizer read this. Click to use it' : 'Commit: replace the ink with this (Enter)';
      btn.onclick = () => takeReading(b.id, i);
      el.appendChild(btn);
    });
    if (choice || pv.unsure || pv.suggest) { // the escape to typing it: Edit LaTeX
      const ed = document.createElement('button');
      ed.className = 'pv-edit ui-btn xs icon';
      ed.title = 'Edit the LaTeX';
      ed.innerHTML = window.mathboardIcons?.svg('board') || '';
      ed.onclick = () => openEditor(b);
      el.appendChild(ed);
    }
  }
  // Commit a waiting reading (option i of a disagreement).
  function takeReading(id, i = 0) {
    const f = findBlock(id);
    if (!f?.b.pv) return;
    if (waiting(f.b)) return toast('This reading is out of date: a new one is on its way'); // the group changed
    const { p, b } = f, c = b.pv.cands?.length > 1 ? b.pv.cands[i] : b.pv;
    if (!c) return;
    commitUndo(snapshot());
    b.manual = false;
    commitReading(p, b, c.latex);
    afterChange(p);
  }

  // ================================================================ images (under the typeset and the ink)
  const imgLayer = document.createElement('div'); // first child of #typeset, so typeset paints over it
  tsLayer.prepend(imgLayer);
  const imgEls = new Map(); // image id -> <img>
  const CORNER = 18;        // px: grabbing this close to an image's bottom-right corner resizes it

  function renderImages() {
    const p = pg(), live = new Set();
    p.images.forEach((im, i) => {
      live.add(im.id);
      let el = imgEls.get(im.id);
      if (!el) {
        el = document.createElement('img');
        el.src = media[im.id];
        el.alt = '';
        el.className = 'board-img';
        imgEls.set(im.id, el);
      }
      if (imgLayer.children[i] !== el) imgLayer.insertBefore(el, imgLayer.children[i] || null); // page order = z order
      el.style.transform = `translate(${im.box[0]}px, ${im.box[1]}px)`;
      el.style.width = `${im.box[2] - im.box[0]}px`;
      el.style.height = `${im.box[3] - im.box[1]}px`;
      el.classList.toggle('sel', im.id === editingImg);
    });
    for (const [id, el] of imgEls) if (!live.has(id)) { el.remove(); imgEls.delete(id); }
  }

  // The 3D tab's "To board" button (graph/features/bridge.js) sends a snapshot of its view here.
  window.addEventListener('mathboard:to-board', e => {
    const d = e.detail || {};
    if (typeof d.src !== 'string' || !d.src.startsWith('data:image/') || !(d.w > 0 && d.h > 0)) return;
    const p = pg(), k = Math.min(1, (W * 0.5) / d.w, (H * 0.6) / d.h);
    const w = d.w * k, h = d.h * k;
    let x = (W - w) / 2, y = (H - h) / 2;
    while (p.images.some(im => Math.abs(im.box[0] - x) < 1 && Math.abs(im.box[1] - y) < 1)) { x += 28; y += 28; } // don't hide an earlier one
    commitUndo(snapshot());
    const im = { id: uid(), box: [r1(x), r1(y), r1(x + w), r1(y + h)] };
    media[im.id] = d.src;
    p.images.push(im);
    renderImages();
    scheduleSave();
    d.handled = true;
    d.page = pageIdx + 1;
  });

  function refreshAll() { redraw(); renderImages(); renderTypeset(); renderPreviews(); paintToolbar(); paintStatus(); }
  function afterChange(p) {
    if (p === pg()) { redraw(); renderTypeset(); renderPreviews(); }
    paintStatus();
    scheduleSave();
  }

  // ================================================================ grouping strokes into expressions
  function commitMath(p, s, S) {
    const hits = joinIds(p, s.box, S).map(id => blockOf(p, id));
    let b;
    if (!hits.length) {
      b = { id: uid(), latex: '', status: 'pending', color: s.color, box: null };
      p.blocks.push(b);
    } else {
      b = hits[0];
      for (const o of hits.slice(1)) { // stroke bridges several expressions (e.g. a fraction bar, brackets): merge
        for (const t of p.strokes) if (t.block === o.id) t.block = b.id;
        removeBlock(p, o);
      }
    }
    const merged = hits.length > 1;
    if (merged) { b.latex = ''; b.pv = null; b.manual = false; b.locks = []; }
    s.block = b.id;
    s.conv = false;
    refreshBox(p, b);
    touch(p, b, !merged);
    lastMath = { id: b.id, t: Date.now() };
  }

  // A pen-down away from a waiting group means you have moved on from it: it converts at once.
  // Returns the ids of the groups you moved on from.
  function markMovedOn(x, y) {
    const p = pg(), S = pageSize(p), burst = burstId(), out = [];
    for (const g of groupsOf(p, S)) {
      const b = blockOf(p, g.id);
      if (!waiting(b) && !b.pv) continue;
      if (geom.isNear(g, x, y, S, settings.reach, g.id === burst)) continue;
      if (b.status === 'pending') b.movedOn = true;
      out.push(b.id);
    }
    return out;
  }
  // While the pointer hovers near a waiting group its idle time doesn't count, up to the profile's
  // ceiling after its last change (touch can't hover: no pause). Preview shows a reading without
  // replacing anything, so it waits two thirds as long.
  function holdForHover(now, dt) {
    const ceil = prof().ceiling * (mode() === 'preview' ? 2 / 3 : 1);
    if (!hover || !ceil || active || inputType === 'touch') return;
    const p = pg(), S = pageSize(p);
    for (const g of groupsOf(p, S)) {
      const b = blockOf(p, g.id);
      if (b.status !== 'pending' || b.movedOn || b.force) continue;
      const lim = (b.t0 || now) + ceil;
      if (b.due < lim && geom.isNear(g, hover.x, hover.y, S, settings.reach)) b.due = Math.min(lim, b.due + dt);
    }
  }

  // ================================================================ recognition
  function tracePath(c, s) {
    const P = s.pts;
    c.beginPath();
    c.moveTo(P[0][0], P[0][1]);
    if (P.length === 1) { c.lineTo(P[0][0] + 0.01, P[0][1]); return; }
    for (let i = 1; i < P.length; i++) {
      const b = P[i], n = P[i + 1], m = n ? mid(b, n) : b;
      c.quadraticCurveTo(b[0], b[1], m[0], m[1]);
    }
  }

  // Black-on-white crop of one expression, uniform line width (colour/pressure only confuse the model).
  // Also returns symbol_px (the median symbol height in image px, which the server normalizes to its
  // own target) and the strokes in image px with times, for the server's debug log.
  function rasterize(p, b, variant = 0) {
    const strokes = strokesOf(p, b), box = unionBox(strokes.map(s => s.box));
    const S = groupSize(p, b), v = VARIANTS[variant] || VARIANTS[0];
    const bw = box[2] - box[0], bh = box[3] - box[1];
    const pad = Math.max(14, 0.1 * Math.max(bw, bh));
    const w = bw + 2 * pad, h = bh + 2 * pad;
    let k = Math.min(2, Math.max(RASTER_MAX / Math.max(w, h), SYM_PX / S)) * v.scale;
    k = Math.min(k, Math.sqrt(MAX_AREA / (w * h)));
    const cw = Math.max(56, Math.round(w * k)), ch = Math.max(56, Math.round(h * k));
    const ox = (cw - w * k) / 2 + (pad - box[0]) * k, oy = (ch - h * k) / 2 + (pad - box[1]) * k;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const g = cv.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, cw, ch);
    g.setTransform(k, 0, 0, k, ox, oy);
    g.strokeStyle = '#000';
    g.lineCap = g.lineJoin = 'round';
    // Lines 8% of a symbol (5 px at 64 px symbols, the best rendering in docs/BOARD_DIAGNOSIS.md), so
    // the proportion survives the server's rescaling. Thin mouse strokes made the model misread 0s.
    g.lineWidth = (clamp(0.08 * S * k, 4, 16) * v.width) / k;
    for (const s of strokes) { tracePath(g, s); g.stroke(); }
    const t0 = Math.min(...strokes.map(s => s.t0 ?? Infinity));
    const T = Number.isFinite(t0) ? t0 : 0;
    const pts = strokes.map(s => s.pts.map(pt => [r1(pt[0] * k + ox), r1(pt[1] * k + oy), Math.max(0, Math.round((s.t0 ?? T) - T + (pt[3] ?? 0)))]));
    return { image: cv.toDataURL('image/png'), symbolPx: r1((S * k) / v.scale), strokes: pts };
  }

  let lastPump = Date.now();
  function pump() {
    const now = Date.now(), dt = Math.min(1000, now - lastPump);
    lastPump = now;
    holdForHover(now, dt);
    if (mirror || inflight || !server.ready) return;
    const drawing = !!active?.stroke, off = mode() === 'off';
    let pick = null;
    pages.forEach((p, i) => {
      for (const b of p.blocks) {
        if (b.status !== 'pending') continue;
        const at = b.force || (!hold && !off && b.movedOn); // Enter, Re-recognize, or you moved on
        if (!at && (hold || off || drawing || b.due > now)) continue;
        const rank = (i === pageIdx ? 0 : 1e13) + (at ? 0 : b.due); // current page first
        if (!pick || rank < pick.rank) pick = { p, b, rank };
      }
    });
    if (pick) recognize(pick.p, pick.b);
  }

  async function recognize(p, b) {
    const id = b.id, ver = b.ver, variant = b.rv || 0;
    const backend = b.rb || backendChoice(); // Re-recognize may ask another backend
    b.rv = 0;
    b.rb = '';
    b.status = 'busy';
    const r = rasterize(p, b, variant);
    const body = { image: r.image, symbol_px: r.symbolPx, strokes: r.strokes };
    if (backend) body.backend = backend;
    window.__mathboardLastImage = r.image;
    window.__mathboardLastRequest = { symbol_px: r.symbolPx, strokes: r.strokes.length, backend: backend || null, variant };
    inflight = { id, ver };
    paintStatus();
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      lastMs = data.ms ?? null;
      if (Number.isFinite(data.prompt_tokens)) lastTokens = data.prompt_tokens;
      applyResult(id, ver, tex.reading(data));
    } catch (err) {
      const f = findBlock(id);
      if (f && f.b.ver === ver) {
        f.b.errors = (f.b.errors || 0) + 1;
        f.b.status = f.b.errors < 3 ? 'pending' : 'error';
        f.b.due = Date.now() + 2000;
        if (f.p === pg()) redraw();
      }
      toast(`Recognition failed: ${err.message}`);
      pollStatus();
    } finally {
      inflight = null;
      paintStatus();
    }
  }

  // Typeset a reading in place of the ink.
  function commitReading(p, b, latex) {
    b.latex = latex;
    b.pv = null;
    b.status = 'done';
    b.keepTs = false;
    b.tsBox = b.box.slice();
    b.commitOnResult = b.forceCommit = false;
    for (const s of strokesOf(p, b)) s.conv = true;
  }

  // A reading arrived. Locks are applied first. Nothing recognized keeps the ink. A reading is
  // committed at once unless the ensemble disagreed (or half of it failed), the group was edited by
  // hand, or (Preview) you haven't moved on: then it waits in a chip under the ink.
  function applyResult(id, ver, rd) {
    const f = findBlock(id);
    if (!f || f.b.ver !== ver) return; // edited while we waited; a newer request will follow
    const { p, b } = f;
    const locks = b.locks || [], asked = b.asked;
    b.asked = b.force = b.keepTs = false;
    b.by = rd.by;
    const latex = rd.empty ? '' : tex.applyLocks(rd.latex, locks);
    const cands = rd.differ ? rd.cands.map(c => ({ backend: c.backend, model: c.model, latex: tex.applyLocks(c.latex, locks) })).filter(c => toHTML(c.latex)) : [];
    const differ = cands.length > 1, sure = rd.sure && !differ;
    if (!latex) { // the model saw no maths: keep the ink (and the typeset it already had)
      b.status = b.latex ? 'done' : 'empty';
      b.pv = null;
      b.commitOnResult = b.forceCommit = false;
      if (asked) toast('The model saw no maths there');
    } else if (!differ && !toHTML(latex)) {
      if (b.latex) { b.status = 'done'; toast('KaTeX can\'t render the new reading'); }
      else { b.latex = latex; b.status = 'error'; }
      b.pv = null;
    } else if (!differ && tex.sameReading(latex, b.latex)) { // nothing new: the ink is covered again
      commitReading(p, b, b.latex);
      if (asked) toast('Same reading again. Edit the LaTeX, or Re-recognize once more');
    } else if (sure && !b.manual && (mode() !== 'preview' || b.commitOnResult || b.forceCommit)) {
      commitReading(p, b, latex);
    } else {
      b.pv = { latex, cands: differ ? cands : null, suggest: !!b.manual, unsure: !sure && !differ, by: rd.by, at: Date.now() };
      b.status = 'done';
      b.commitOnResult = b.forceCommit = false;
    }
    afterChange(p);
  }
  // A waiting reading that may be committed without a pick: not a choice, a suggestion or unsure.
  const pvSure = pv => !!pv && !pv.cands && !pv.suggest && !pv.unsure;

  setInterval(pump, 100);

  // Enter: convert every waiting group on this page now, and commit the readings that wait for you
  // (not a choice, a suggestion or an unsure one: those need a click).
  function convertNow() {
    const p = pg(), ready = p.blocks.filter(b => !waiting(b) && pvSure(b.pv));
    let pick = 0;
    for (const b of p.blocks) {
      if (waiting(b)) b.force = b.forceCommit = true;
      else if (b.pv && !pvSure(b.pv)) pick++;
    }
    if (ready.length) commitUndo(snapshot());
    for (const b of ready) commitReading(p, b, b.pv.latex);
    if (pick) toast(pick === 1 ? 'One reading needs a click (or 1 / 2 for a choice)' : `${pick} readings need a click`);
    afterChange(p);
  }
  // The reading Esc and the number keys act on: the one under the pointer, else the newest.
  function targetPv() {
    const p = pg(), S = pageSize(p), withPv = p.blocks.filter(b => b.pv);
    if (hoverChip) { const b = blockOf(p, hoverChip); if (b?.pv) return b; }
    if (hover) for (const g of groupsOf(p, S)) {
      const b = blockOf(p, g.id);
      if (b.pv && geom.isNear(g, hover.x, hover.y, S, settings.reach)) return b;
    }
    return withPv.reduce((a, b) => (!a || (b.pv.at || 0) > (a.pv.at || 0) ? b : a), null);
  }
  // Esc on a reading: keep the ink, drop the reading. The group stays ink until it changes.
  function discardPv(b) {
    const p = pg();
    commitUndo(snapshot());
    b.pv = null;
    if (waiting(b)) { // the new reading on its way goes too
      b.ver = uid();
      b.keepTs = false;
      b.status = b.latex ? 'done' : 'empty';
    } else if (!b.latex) b.status = 'empty';
    afterChange(p);
  }
  // Esc: close the open panel or menu, else drop the lasso selection, else keep the ink of a reading.
  function escape() {
    const open = !editor.hidden || !$('settings').hidden || !alts.hidden;
    if (open) return closePanels();
    if (sel) return clearSel();
    const b = targetPv();
    if (b) discardPv(b);
  }
  // 1 to 9: a look-alike in the open menu, else a reading of the targeted choice.
  function pickNumber(n) {
    if (!alts.hidden) return pickAlt(n - 1);
    const b = targetPv();
    if (!b) return;
    const choice = b.pv.cands?.length > 1;
    if (choice ? n <= b.pv.cands.length : n === 1) takeReading(b.id, n - 1);
  }
  function setHold(on) {
    hold = on;
    if (!hold) { // release: whatever waited converts shortly
      const t = Date.now() + 250;
      for (const p of pages) for (const b of p.blocks) if (b.status === 'pending') b.due = Math.min(b.due, t);
    }
    paintToolbar();
    paintStatus();
  }

  // ================================================================ tools: eraser, select, laser
  function eraseAt(x, y) {
    const p = pg(), touched = new Set();
    let changed = false;
    // Touching typeset reveals the handwriting underneath so it can be corrected.
    for (const b of p.blocks) {
      if (!tsVisible(b) || !b.tsRect || !inRect(x, y, grow(b.tsRect, ERASER_R, ERASER_R))) continue;
      markUndo();
      b.keepTs = false;
      b.status = 'pending';
      for (const s of strokesOf(p, b)) s.conv = false;
      touched.add(b);
      changed = true;
    }
    const bm = new Map(p.blocks.map(b => [b.id, b]));
    for (let i = p.strokes.length - 1; i >= 0; i--) {
      const s = p.strokes[i], b = s.block ? bm.get(s.block) : null;
      if (inkHidden(s, b) || !inRect(x, y, grow(s.box, ERASER_R, ERASER_R)) || !hitStroke(s, x, y, ERASER_R)) continue;
      markUndo();
      p.strokes.splice(i, 1);
      if (b) touched.add(b);
      changed = true;
    }
    for (const b of touched) {
      if (!strokesOf(p, b).length) { removeBlock(p, b); continue; }
      refreshBox(p, b);
      touch(p, b, false);
    }
    return changed;
  }
  function eraseLine(a, b) {
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / (ERASER_R * 0.6)));
    let changed = false;
    for (let i = 1; i <= n; i++) changed = eraseAt(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) || changed;
    if (changed) { redraw(); renderTypeset(); renderPreviews(); }
  }

  function pickAt(x, y) {
    const p = pg();
    for (let i = p.blocks.length - 1; i >= 0; i--) {
      const b = p.blocks[i], r = tsVisible(b) && b.tsRect ? b.tsRect : b.box;
      if (r && inRect(x, y, grow(r, 6, 6))) return { b };
    }
    for (let i = p.strokes.length - 1; i >= 0; i--) {
      const s = p.strokes[i];
      if (s.kind === 'draw' && inRect(x, y, grow(s.box, 8, 8)) && hitStroke(s, x, y, 8)) return { s };
    }
    for (let i = p.images.length - 1; i >= 0; i--) { // images sit under everything else
      const r = p.images[i].box;
      if (!inRect(x, y, grow(r, 4, 4))) continue;
      return { img: p.images[i], resize: x > r[2] - CORNER && y > r[3] - CORNER, grab: r[2] - x };
    }
    return null;
  }
  function moveStroke(s, dx, dy) {
    for (const pt of s.pts) { pt[0] = r1(pt[0] + dx); pt[1] = r1(pt[1] + dy); }
    s.box = shift(s.box, dx, dy);
  }
  function dragSelect(x, y) {
    const a = active;
    if (!a.hit) return;
    if (!a.moved) {
      if (Math.hypot(x - a.start[0], y - a.start[1]) < 5) return;
      a.moved = true;
      markUndo();
    }
    const dx = x - a.last[0], dy = y - a.last[1];
    if (a.hit.img) return dragImage(a, x, y);
    if (a.hit.b) {
      const b = a.hit.b;
      for (const s of strokesOf(pg(), b)) moveStroke(s, dx, dy);
      b.box = shift(b.box, dx, dy);
      if (b.tsBox) b.tsBox = shift(b.tsBox, dx, dy);
    } else {
      moveStroke(a.hit.s, dx, dy);
    }
    redraw();
    renderTypeset();
    renderPreviews();
  }
  function dragImage(a, x, y) {
    const hit = a.hit, r = hit.box0 ||= hit.img.box.slice();
    if (hit.resize) { // keeps its aspect ratio; the top-left corner stays put
      const w = Math.max(60, x + hit.grab - r[0]);
      hit.img.box = [r[0], r[1], r1(r[0] + w), r1(r[1] + (w * (r[3] - r[1])) / (r[2] - r[0]))];
    } else {
      hit.img.box = shift(r, x - a.start[0], y - a.start[1]).map(r1);
    }
    renderImages();
  }

  // ================================================================ regrouping: lasso, scratch-out
  // After strokes left groups: an emptied group goes, the others show their ink and are read again.
  function settleBlocks(p, ids) {
    for (const id of ids) {
      const b = blockOf(p, id);
      if (!b) continue;
      if (!strokesOf(p, b).length) { removeBlock(p, b); continue; }
      refreshBox(p, b);
      touch(p, b, false);
      b.pv = null;
    }
  }
  // Make these strokes one new group and read it now (turning Draw strokes into math if asked).
  function regroup(p, strokes, toMath = false) {
    const from = new Set(strokes.map(s => s.block).filter(Boolean));
    const b = { id: uid(), latex: '', status: 'pending', color: strokes[0].color, box: null };
    p.blocks.push(b);
    for (const s of strokes) { if (toMath) s.kind = 'math'; s.block = b.id; s.conv = false; }
    refreshBox(p, b);
    touch(p, b, false);
    b.force = true;
    settleBlocks(p, from);
    return b;
  }

  // The lasso (Select, drag on empty board) selects ink whose points are mostly inside it, and
  // converted expressions whose typeset is mostly inside it.
  function finishLasso(poly) {
    const p = pg();
    if (poly.length < 4 || geom.pathLength(poly) < 30) return clearSel();
    const bm = new Map(p.blocks.map(b => [b.id, b])), ids = new Set();
    for (const s of p.strokes) {
      const b = s.block && bm.get(s.block);
      if (!inkHidden(s, b) && geom.fracInside(s.pts, poly) >= 0.6) ids.add(s.id);
    }
    for (const b of p.blocks) {
      if (tsVisible(b) && b.tsRect && geom.rectInside(b.tsRect, poly) >= 0.5) for (const s of strokesOf(p, b)) ids.add(s.id);
    }
    if (!ids.size) return clearSel();
    sel = { ids, box: null };
    selChanged();
  }
  function selBox(p) {
    const bm = new Map(p.blocks.map(b => [b.id, b]));
    return unionBox(p.strokes.filter(s => sel.ids.has(s.id)).map(s => {
      const b = s.block && bm.get(s.block);
      return inkHidden(s, b) && b.tsRect ? b.tsRect : s.box;
    }));
  }
  function selChanged() {
    const p = pg();
    if (sel) {
      sel.ids = new Set([...sel.ids].filter(id => p.strokes.some(s => s.id === id)));
      if (!sel.ids.size) sel = null;
    }
    if (sel) sel.box = selBox(p);
    paintSelBar();
    redraw();
  }
  function clearSel() {
    if (!sel) return;
    sel = null;
    paintSelBar();
    redraw();
  }
  function dragSelection(x, y) {
    const a = active, p = pg();
    if (!a.moved) {
      if (Math.hypot(x - a.start[0], y - a.start[1]) < 5) return;
      a.moved = true;
      markUndo();
    }
    const dx = x - a.last[0], dy = y - a.last[1];
    for (const s of p.strokes) if (sel.ids.has(s.id)) moveStroke(s, dx, dy);
    for (const b of p.blocks) {
      const ss = strokesOf(p, b), n = ss.filter(s => sel.ids.has(s.id)).length;
      if (!n) continue;
      if (n === ss.length) { // the whole expression moves, typeset and all
        b.box = shift(b.box, dx, dy);
        if (b.tsBox) b.tsBox = shift(b.tsBox, dx, dy);
      } else { refreshBox(p, b); a.selDrag.partial.add(b.id); }
    }
    sel.box = shift(sel.box, dx, dy);
    redraw();
    renderTypeset();
    renderPreviews();
    paintSelBar();
  }

  // The lasso's actions. Group: one expression, read now. Split: the selected part of each
  // expression becomes its own. To math: Draw ink becomes math. Keep as ink: never converted.
  const selBar = $('lasso-bar');
  function selParts(p) {
    const chosen = p.strokes.filter(s => sel.ids.has(s.id));
    const math = chosen.filter(s => s.kind === 'math'), draw = chosen.filter(s => s.kind === 'draw');
    const partial = [...new Set(math.map(s => s.block).filter(Boolean))]
      .filter(id => p.strokes.some(s => s.block === id && !sel.ids.has(s.id)));
    return { chosen, math, draw, partial };
  }
  function paintSelBar() {
    if (!sel || document.body.classList.contains('clean')) { selBar.hidden = true; return; }
    const { chosen, math, draw, partial } = selParts(pg());
    const act = n => selBar.querySelector(`[data-act="${n}"]`);
    act('group').disabled = !math.length;
    act('split').disabled = !partial.length;
    act('math').disabled = !draw.length;
    act('ink').disabled = !math.length;
    $('lasso-n').textContent = `${chosen.length} stroke${chosen.length === 1 ? '' : 's'}`;
    selBar.hidden = false;
    const r = sel.box, bw = selBar.offsetWidth, bh = selBar.offsetHeight;
    let top = r[3] + 18;
    if (top + bh > H - 8) top = Math.max(8, r[1] - bh - 18);
    selBar.style.left = clamp((r[0] + r[2]) / 2 - bw / 2, 8, W - bw - 8) + 'px';
    selBar.style.top = top + 'px';
  }
  function selAction(name) {
    const p = pg();
    if (!sel) return;
    const { chosen, math, draw, partial } = selParts(p);
    commitUndo(snapshot());
    if (name === 'group' && math.length) regroup(p, math);
    else if (name === 'math' && draw.length) regroup(p, [...draw, ...math], true);
    else if (name === 'split') {
      for (const id of partial) regroup(p, math.filter(s => s.block === id));
    } else if (name === 'ink') {
      const from = new Set(math.map(s => s.block).filter(Boolean));
      for (const s of math) { s.kind = 'draw'; s.block = null; s.conv = false; }
      settleBlocks(p, from);
    } else if (name === 'delete') {
      const from = new Set(chosen.map(s => s.block).filter(Boolean));
      p.strokes = p.strokes.filter(s => !sel.ids.has(s.id));
      settleBlocks(p, from);
    }
    sel = null;
    paintSelBar();
    afterChange(p);
  }
  for (const btn of selBar.querySelectorAll('[data-act]')) btn.onclick = () => selAction(btn.dataset.act);

  // Scratch-out: a zig-zag Math-pen stroke erases the ink it crosses (twice, or once when the ink is
  // mostly inside the zig-zag), and typeset it covers by half. Returns false, and changes nothing,
  // when it crosses no ink: then it is an ordinary stroke.
  function scratchOut(p, s, snap) {
    const bm = new Map(p.blocks.map(b => [b.id, b])), zone = grow(s.box, 4, 4), poly = geom.rectPoly(zone);
    const gone = new Set();
    for (const t of p.strokes) {
      const b = t.block && bm.get(t.block);
      if (inkHidden(t, b) || !overlaps(t.box, s.box)) continue;
      const n = geom.crossings(s.pts, t.pts), inside = geom.fracInside(t.pts, poly);
      if (n >= 2 || (n >= 1 && inside >= 0.5) || (t.pts.length <= 2 && inside === 1)) gone.add(t.id);
    }
    for (const b of p.blocks) {
      if (tsVisible(b) && b.tsRect && geom.coverage(b.tsRect, zone) >= 0.5) for (const t of strokesOf(p, b)) gone.add(t.id);
    }
    if (!gone.size) return false;
    commitUndo(snap);
    const from = new Set(p.strokes.filter(t => gone.has(t.id) && t.block).map(t => t.block));
    p.strokes = p.strokes.filter(t => !gone.has(t.id));
    settleBlocks(p, from);
    return true;
  }

  // ================================================================ look-alikes: tap a typeset symbol
  // The typeset symbol under (x, y): { b, k (its token index, board/tex.js taps), rect }.
  function glyphAt(x, y) {
    const p = pg();
    for (let i = p.blocks.length - 1; i >= 0; i--) {
      const b = p.blocks[i], el = tsEls.get(b.id);
      if (!el || b.status !== 'done' || !tsVisible(b) || !b.tsRect || !inRect(x, y, grow(b.tsRect, 6, 6))) continue;
      let best = null;
      for (const node of el.querySelectorAll('[data-mbt]')) {
        const r = node.getBoundingClientRect();
        if (x < r.left - 3 || x > r.right + 3 || y < r.top - 3 || y > r.bottom + 3) continue;
        const area = r.width * r.height;
        if (!best || area < best.area) best = { b, k: Number(node.dataset.mbt), area, rect: [r.left, r.top, r.right, r.bottom] };
      }
      if (best) return best;
    }
    return null;
  }
  const alts = $('alts');
  let altOf = null; // { id, k, list } while the menu is open
  function openAlts(g) {
    const tok = tex.taps(g.b.latex)[g.k];
    if (!tok) return;
    const list = tex.alternatives(tok.t);
    altOf = { id: g.b.id, k: g.k, list };
    const row = $('alts-row');
    row.textContent = '';
    list.forEach((a, i) => {
      const btn = document.createElement('button');
      btn.className = 'ui-chip';
      btn.innerHTML = toHTML(a) || a;
      btn.title = `Use ${a} (${i + 1})`;
      btn.onclick = () => pickAlt(i);
      row.appendChild(btn);
    });
    $('alts-head').textContent = list.length ? 'Look-alikes' : 'No look-alikes';
    row.hidden = !list.length;
    alts.hidden = false;
    const r = g.rect, aw = alts.offsetWidth, ah = alts.offsetHeight;
    let top = r[3] + 8;
    if (top + ah > H - 8) top = Math.max(8, r[1] - ah - 8);
    alts.style.left = clamp(r[0] - 6, 8, W - aw - 8) + 'px';
    alts.style.top = top + 'px';
  }
  // Replace the symbol and lock it, so later readings of this expression keep your pick.
  function pickAlt(i) {
    const f = altOf && findBlock(altOf.id), to = altOf?.list[i];
    if (!f || !to) return;
    const { p, b } = f, T = tex.taps(b.latex), tok = T[altOf.k];
    if (!tok) return;
    const next = tex.replaceAt(b.latex, tok, to);
    if (!toHTML(next)) return toast('KaTeX can\'t render that');
    commitUndo(snapshot());
    const lock = tex.makeLock(next, altOf.k, to);
    lock.from = tok.t;
    b.locks = [...(b.locks || []).filter(L => L.pos !== altOf.k), lock];
    b.latex = next;
    b.ver = uid(); // a reading already on its way is for the old text
    closePanels();
    afterChange(p);
  }
  $('alts-edit').onclick = () => {
    const f = altOf && findBlock(altOf.id);
    closePanels();
    if (f) openEditor(f.b);
  };

  const laser = []; // {x, y, t, brk}
  let fxOn = false;
  function kickFx() { if (!fxOn) { fxOn = true; requestAnimationFrame(fxFrame); } }
  function fxFrame() {
    const now = performance.now();
    while (laser.length && now - laser[0].t > LASER_MS) laser.shift();
    fx.clearRect(0, 0, W, H);
    fx.lineCap = fx.lineJoin = 'round';
    for (let i = 1; i < laser.length; i++) {
      const a = laser[i - 1], b = laser[i];
      if (b.brk) continue;
      const life = 1 - (now - b.t) / LASER_MS;
      fx.strokeStyle = `rgba(255, 45, 45, ${life.toFixed(3)})`;
      fx.lineWidth = 2 + 5 * life;
      fx.beginPath(); fx.moveTo(a.x, a.y); fx.lineTo(b.x, b.y); fx.stroke();
    }
    const laserHead = active?.tool === 'laser' ? active.last : settings.tool === 'laser' && hover ? [hover.x, hover.y] : null;
    if (laserHead) {
      fx.save();
      fx.shadowColor = 'rgba(255, 30, 30, 0.9)';
      fx.shadowBlur = 16;
      fx.fillStyle = 'rgba(255, 50, 50, 0.95)';
      fx.beginPath(); fx.arc(laserHead[0], laserHead[1], 6, 0, 7); fx.fill();
      fx.restore();
    }
    const erasing = active?.tool === 'erase' || (settings.tool === 'erase' && !active);
    const eraserAt = active?.tool === 'erase' ? active.last : hover && [hover.x, hover.y];
    if (erasing && eraserAt) {
      fx.lineWidth = 1.5;
      fx.strokeStyle = settings.theme === 'dark' ? 'rgba(255,255,255,.6)' : 'rgba(0,0,0,.5)';
      fx.beginPath(); fx.arc(eraserAt[0], eraserAt[1], ERASER_R, 0, 7); fx.stroke();
    }
    if (!document.body.classList.contains('clean')) drawLive();
    if (laser.length || active?.tool === 'laser' || active?.lasso || active?.stroke?.kind === 'math') requestAnimationFrame(fxFrame);
    else fxOn = false;
  }
  // While you write: an outline round the group the stroke will join (all of them when it would
  // merge several). While you lasso: the lasso.
  function drawLive() {
    const ui = outlineColors();
    if (active?.stroke?.kind === 'math' && active.join?.length) {
      const p = pg(), boxes = active.join.map(id => blockOf(p, id)).filter(b => b?.box).map(b => hitBox(p, b));
      if (boxes.length) {
        fx.save();
        fx.strokeStyle = ui.join;
        fx.lineWidth = 1.5;
        fx.setLineDash([6, 4]);
        outline(fx, unionBox([...boxes, active.sbox]), 8);
        fx.restore();
      }
    }
    if (active?.lasso?.length > 1) {
      const L = active.lasso;
      fx.save();
      fx.strokeStyle = ui.join;
      fx.lineWidth = 1.5;
      fx.setLineDash([5, 4]);
      fx.beginPath();
      fx.moveTo(L[0][0], L[0][1]);
      for (const [x, y] of L.slice(1)) fx.lineTo(x, y);
      fx.closePath();
      fx.stroke();
      fx.restore();
    }
  }

  // ================================================================ pointer input
  function onDown(e) {
    if (active) return;
    if (e.pointerType === 'pen') penSeen = true;
    else if (e.pointerType === 'touch' && penSeen) return; // palm rejection once a pen has been used
    inputType = e.pointerType || 'mouse';
    closePanels();
    let tool = settings.tool;
    if (e.button === 2 || e.button === 5 || (e.buttons & 32)) tool = 'erase'; // right button, pen eraser end
    try { fxCv.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    const x = e.clientX, y = e.clientY;
    active = { id: e.pointerId, tool, snap: snapshot(), undone: false, last: [x, y], start: [x, y], t: performance.now() };
    const inSel = tool === 'select' && sel && inRect(x, y, grow(sel.box, 10, 10));
    if (sel && !inSel) clearSel();

    if (tool === 'math' || tool === 'draw') {
      active.movedFrom = markMovedOn(x, y);
      active.S = pageSize(pg());
      const s = {
        id: uid(), kind: tool, color: settings[tool + 'Color'], w: PEN_W, pr: e.pointerType === 'pen', t0: Date.now(),
        pts: [[r1(x), r1(y), Math.round((e.pressure || 0.5) * 100) / 100, 0]], block: null, conv: false,
      };
      active.stroke = s;
      ctx.lineCap = ctx.lineJoin = 'round';
      drawStroke(ctx, s);
      if (tool === 'math') kickFx();
    } else if (tool === 'erase') {
      if (eraseAt(x, y)) { redraw(); renderTypeset(); renderPreviews(); }
      kickFx();
    } else if (tool === 'select') {
      active.moved = false;
      if (inSel) active.selDrag = { partial: new Set() };
      else {
        active.hit = pickAt(x, y);
        if (!active.hit) { active.lasso = [[x, y]]; kickFx(); } // empty board: lasso
      }
    } else if (tool === 'laser') {
      laser.push({ x, y, t: performance.now(), brk: true });
      kickFx();
    }
  }

  function onMove(e) {
    hover = { x: e.clientX, y: e.clientY };
    if (!active && (settings.tool === 'erase' || settings.tool === 'laser')) kickFx();
    if (!active) { // Select over an image's resize corner
      const cur = settings.tool === 'select' && pickAt(hover.x, hover.y)?.resize ? 'nwse-resize' : '';
      if (fxCv.style.cursor !== cur) fxCv.style.cursor = cur;
    }
    if (!active || e.pointerId !== active.id) return;
    const evs = (e.getCoalescedEvents && e.getCoalescedEvents()) || [];
    if (!evs.length) evs.push(e);
    for (const ev of evs) {
      const x = ev.clientX, y = ev.clientY;
      if (active.stroke) {
        const s = active.stroke, P = s.pts, last = P[P.length - 1];
        if (Math.hypot(x - last[0], y - last[1]) < 1) continue;
        P.push([r1(x), r1(y), Math.round((ev.pressure || 0.5) * 100) / 100, Math.round(performance.now() - active.t)]);
        if (P.length >= 3) {
          ctx.strokeStyle = color(s.color);
          ctx.lineCap = ctx.lineJoin = 'round';
          drawSeg(ctx, s, P.length - 2);
        }
        if (s.kind === 'math' && P.length % 4 === 0) { // the group this stroke would join, for the outline
          active.sbox = strokeBox(s);
          active.join = joinIds(pg(), active.sbox, active.S);
        }
      } else if (active.tool === 'erase') {
        eraseLine(active.last, [x, y]);
      } else if (active.lasso) {
        const L = active.lasso, q = L[L.length - 1];
        if (Math.hypot(x - q[0], y - q[1]) >= 3) L.push([r1(x), r1(y)]);
      } else if (active.selDrag) {
        dragSelection(x, y);
      } else if (active.tool === 'select') {
        dragSelect(x, y);
      } else if (active.tool === 'laser') {
        laser.push({ x, y, t: performance.now() });
      }
      active.last = [x, y];
    }
    if (active.tool === 'erase' || active.tool === 'laser') kickFx();
  }

  function onUp(e) {
    if (!active || e.pointerId !== active.id) return;
    const a = active;
    active = null;
    if (a.stroke || a.lasso) fx.clearRect(0, 0, W, H); // the join outline or the lasso, gone with the pen
    if (a.stroke) {
      const s = a.stroke, p = pg(), S = a.S ?? pageSize(p), P = s.pts, pr = prof();
      s.box = strokeBox(s);
      // Started on a reading chip and barely moved: a click on the chip, not a dot.
      if (a.chipBtn && P.every(q => Math.hypot(q[0] - P[0][0], q[1] - P[0][1]) <= 2 * pr.tapPx)) {
        redraw();
        if (a.chipBtn.isConnected) a.chipBtn.click();
        kickFx();
        return;
      }
      // A tap on a typeset symbol opens its look-alikes instead of leaving a dot.
      const tap = s.kind === 'math' && performance.now() - a.t <= pr.tapMs && P.every(q => Math.hypot(q[0] - P[0][0], q[1] - P[0][1]) <= pr.tapPx);
      const glyph = tap && glyphAt(P[0][0], P[0][1]);
      if (glyph) { redraw(); openAlts(glyph); kickFx(); return; }
      // A zig-zag across ink erases it (scratch-out); across nothing it is just a stroke.
      if (s.kind === 'math' && geom.looksLikeScratch(P, pr.scratchRev) && scratchOut(p, s, a.snap)) {
        afterChange(p);
        kickFx();
        return;
      }
      commitUndo(a.snap);
      p.strokes.push(s);
      if (s.kind === 'math') commitMath(p, s, S);
      afterMove(p, a.movedFrom, s.block);
      redraw();
      renderTypeset();
      renderPreviews();
      paintStatus();
      scheduleSave();
    } else if (a.lasso) {
      finishLasso(a.lasso);
    } else if (a.selDrag) {
      if (a.moved) { settleBlocks(pg(), a.selDrag.partial); afterChange(pg()); }
    } else if (a.tool === 'select') { // a tap opens the editor, as before (the Math pen taps symbols)
      if (a.moved) scheduleSave();
      else if (a.hit?.b) openEditor(a.hit.b);
      else if (a.hit?.img) openImageEditor(a.hit.img);
    } else if (a.undone) {
      scheduleSave();
    }
    kickFx();
  }
  // Preview: moving on commits the readings you left behind (when they are sure), or commits them
  // as soon as they arrive.
  function afterMove(p, ids, joined) {
    if (mode() !== 'preview' || !settings.commitOnMove) return;
    for (const id of ids || []) {
      const b = blockOf(p, id);
      if (!b || b.id === joined || b.manual) continue;
      if (pvSure(b.pv) && !waiting(b)) commitReading(p, b, b.pv.latex);
      else if (waiting(b)) b.commitOnResult = true;
    }
  }

  fxCv.addEventListener('pointerdown', onDown);
  fxCv.addEventListener('pointermove', onMove);
  fxCv.addEventListener('pointerup', onUp);
  fxCv.addEventListener('pointercancel', onUp);
  fxCv.addEventListener('pointerleave', () => { hover = null; kickFx(); });
  fxCv.addEventListener('contextmenu', e => e.preventDefault());

  // ================================================================ editor popover
  let editing = null, editingImg = null; // the popover edits an expression or an image
  const edSrc = $('ed-src'), edPrev = $('ed-preview'), editor = $('editor');
  const edBlock = () => (editing ? blockOf(pg(), editing) : null);
  const edSend = document.createElement('button');
  edSend.className = 'ui-btn sm soft';
  edSend.textContent = 'Send to 3D';
  edSend.title = 'Add this expression as a row in the 3D tab';
  $('ed-copy').after(edSend);
  const edHint = document.createElement('p'); // shown when editing a picture, beside its Delete
  edHint.className = 'hint';
  edHint.textContent = 'Drag to move. Drag the bottom-right corner to resize.';
  editor.querySelector('.row').prepend(edHint);

  function editorMode(image) {
    edSrc.hidden = edPrev.hidden = image;
    edHint.hidden = !image;
    for (const btn of editor.querySelectorAll('.row button')) btn.hidden = image && btn.id !== 'ed-del';
    editor.style.width = image ? 'auto' : '';
    $('ed-del').title = `Delete this ${image ? 'picture' : 'expression'} (Ctrl+Z brings it back)`;
  }
  function placeEditor(r) {
    editor.hidden = false;
    const eh = editor.offsetHeight;
    let top = r[3] + 10;
    if (top + eh > H - 8) top = Math.max(8, r[1] - eh - 10);
    editor.style.left = clamp(r[0], 8, W - editor.offsetWidth - 8) + 'px';
    editor.style.top = top + 'px';
  }
  function openEditor(b) {
    editing = b.id;
    editorMode(false);
    edSrc.value = b.latex || b.pv?.latex || '';
    previewEdit();
    placeEditor(hitBox(pg(), b));
    edSrc.focus();
    edSrc.select();
  }
  function openImageEditor(im) {
    editingImg = im.id;
    editorMode(true);
    placeEditor(im.box);
    renderImages();
  }
  function previewEdit() {
    const src = edSrc.value.trim();
    const html = src ? toHTML(src) : '';
    edPrev.innerHTML = html ?? '<span class="err">KaTeX can\'t render this yet</span>';
  }
  // A hand edit sticks: later strokes in this group bring the model's reading as a suggestion only.
  function applyEdit() {
    const p = pg(), b = edBlock();
    if (!b) return closePanels();
    const src = edSrc.value.trim();
    if (src && !toHTML(src)) return toast('KaTeX can\'t render that');
    commitUndo(snapshot());
    b.latex = src;
    b.ver = uid(); // drop any in-flight result
    b.status = src ? 'done' : 'empty';
    b.keepTs = false;
    b.tsBox = b.box.slice();
    b.manual = !!src;
    b.pv = null;
    b.locks = [];
    b.force = b.forceCommit = b.commitOnResult = b.movedOn = false;
    for (const s of strokesOf(p, b)) s.conv = !!src;
    closePanels();
    refreshAll();
    scheduleSave();
  }
  edSrc.addEventListener('input', previewEdit);
  edSrc.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); applyEdit(); }
    if (e.key === 'Escape') closePanels();
  });
  $('ed-apply').onclick = applyEdit;
  // Re-recognize is a real second opinion: the other backend when the server has two, else a
  // different rendering each time (VARIANTS). The typeset stays up meanwhile.
  $('ed-rerun').onclick = () => {
    const b = edBlock();
    if (!b) return;
    touch(pg(), b, true);
    b.reruns = (b.reruns || 0) + 1;
    b.rb = secondOpinion(b);
    b.rv = b.rb ? 0 : 1 + ((b.reruns - 1) % 2);
    b.force = b.asked = true;
    b.due = 0;
    closePanels();
    refreshAll();
  };
  $('ed-copy').onclick = async () => {
    try { await navigator.clipboard.writeText(edSrc.value.trim()); toast('LaTeX copied'); }
    catch { toast('Clipboard blocked by the browser'); }
  };
  // Converted and added to the 3D tab by graph/features/bridge.js, which also reports problems.
  edSend.onclick = () => {
    const latex = edSrc.value.trim();
    if (!latex) return toast('Nothing to send');
    const detail = { latex };
    window.dispatchEvent(new CustomEvent('mathboard:to-graph', { detail }));
    if (!detail.handled) toast('The 3D tab is not ready');
    else if (detail.ok) closePanels();
  };
  $('ed-ink').onclick = () => {
    const p = pg(), b = edBlock();
    if (!b) return;
    commitUndo(snapshot());
    for (const s of strokesOf(p, b)) { s.kind = 'draw'; s.block = null; s.conv = false; }
    removeBlock(p, b);
    closePanels();
    refreshAll();
    scheduleSave();
  };
  $('ed-del').onclick = () => {
    const p = pg(), b = edBlock();
    if (editingImg) {
      commitUndo(snapshot());
      p.images = p.images.filter(im => im.id !== editingImg);
      closePanels(); // re-renders the images
      scheduleSave();
      return;
    }
    if (!b) return;
    commitUndo(snapshot());
    p.strokes = p.strokes.filter(s => s.block !== b.id);
    removeBlock(p, b);
    closePanels();
    refreshAll();
    scheduleSave();
  };

  function closePanels() {
    // A hidden textarea can keep focus in Chrome and swallow the board's shortcuts.
    if (document.activeElement?.closest?.('.panel, .ui-menu')) document.activeElement.blur();
    editor.hidden = true;
    $('settings').hidden = true;
    $('gear').classList.remove('on');
    alts.hidden = true;
    altOf = null;
    editing = null;
    if (editingImg) { editingImg = null; renderImages(); }
  }

  // ================================================================ toolbar, pages, settings
  const swatchKeys = ['ink', 'c1', 'c2', 'c3', 'c4', 'c5'];
  const swatchNames = { ink: 'Default ink', c1: 'Yellow', c2: 'Blue', c3: 'Pink', c4: 'Green', c5: 'Orange' };
  const swatchEls = swatchKeys.map(key => {
    const el = document.createElement('button');
    el.className = 'swatch';
    el.dataset.c = key;
    el.title = swatchNames[key];
    el.onclick = () => {
      settings[lastPen + 'Color'] = key;
      if (settings.tool !== 'math' && settings.tool !== 'draw') settings.tool = lastPen;
      saveSettings();
      paintToolbar();
    };
    $('swatches').appendChild(el);
    return el;
  });

  function setTool(t) {
    settings.tool = t;
    if (t === 'math' || t === 'draw') lastPen = t;
    fxCv.style.cursor = '';
    saveSettings();
    paintToolbar();
    kickFx();
  }
  function paintToolbar() {
    document.body.dataset.tool = settings.tool;
    for (const btn of document.querySelectorAll('#tools button')) btn.classList.toggle('on', btn.dataset.tool === settings.tool);
    const current = settings[lastPen + 'Color'];
    for (const sw of swatchEls) {
      sw.style.backgroundColor = color(sw.dataset.c); // not the shorthand: the CSS clips it to the dot
      sw.classList.toggle('on', sw.dataset.c === current);
    }
    $('pageno').textContent = `${pageIdx + 1} / ${pages.length}`;
    $('prev').disabled = pageIdx === 0;
    $('next').disabled = pageIdx === pages.length - 1;
    $('hold').classList.toggle('on', hold);
  }
  for (const btn of document.querySelectorAll('#tools button')) btn.onclick = () => setTool(btn.dataset.tool);

  function gotoPage(i) {
    if (active) return;
    pageIdx = clamp(i, 0, pages.length - 1);
    closePanels();
    clearSel();
    refreshAll();
    scheduleSave();
  }
  function addPage() {
    pages.splice(pageIdx + 1, 0, blankPage());
    gotoPage(pageIdx + 1);
  }
  function clearPage() {
    const p = pg();
    if (!p.strokes.length && !p.images.length) return;
    commitUndo(snapshot());
    p.strokes = [];
    p.blocks = [];
    p.images = [];
    clearSel();
    refreshAll();
    scheduleSave();
    toast('Page cleared (Ctrl+Z brings it back)');
  }
  function setTheme(t) {
    settings.theme = t;
    document.documentElement.dataset.theme = t;
    $('set-theme').value = t; // T with the settings open
    saveSettings();
    refreshAll();
  }
  function toggleClean() {
    const on = document.body.classList.toggle('clean');
    closePanels();
    clearSel();
    redraw();
    if (on) toast('UI hidden. Press H to show it again', 1800);
  }

  $('undo').onclick = undo;
  $('redo').onclick = redo;
  $('prev').onclick = () => gotoPage(pageIdx - 1);
  $('next').onclick = () => gotoPage(pageIdx + 1);
  $('addpage').onclick = addPage;
  $('clear').onclick = clearPage;
  $('export').onclick = exportNotes;
  $('hold').onclick = () => setHold(!hold);
  $('convert').onclick = convertNow;
  $('gear').onclick = () => ($('settings').hidden ? openSettings() : closePanels());
  $('set-close').onclick = closePanels;

  const ranges = [
    ['set-delay', 'delay', v => `${v} ms`],
    ['set-reach', 'reach', v => `${Number(v).toFixed(1)}×`],
    ['set-font', 'maxFont', v => `${v} px`],
  ];
  for (const [id, key, fmt] of ranges) {
    const input = $(id), out = input.nextElementSibling;
    input.addEventListener('input', () => {
      settings[key] = Number(input.value);
      out.textContent = fmt(input.value);
      saveSettings();
      if (key === 'maxFont') renderTypeset();
    });
  }
  $('set-ink').onchange = e => { settings.inkAfter = e.target.checked ? 'faint' : 'hide'; saveSettings(); redraw(); }; // a switch
  $('set-theme').onchange = e => setTheme(e.target.value);
  $('set-backend').onchange = e => { settings.backend = e.target.value; saveSettings(); fillBackends(); paintStatus(); };
  // Convert: Auto (typeset replaces the ink), Preview (the reading waits under the ink), Off (Enter).
  function paintMode() {
    for (const btn of document.querySelectorAll('#set-mode button')) btn.classList.toggle('on', btn.dataset.mode === mode());
    $('set-move').checked = !!settings.commitOnMove;
    $('set-move').disabled = mode() !== 'preview';
  }
  for (const btn of document.querySelectorAll('#set-mode button')) {
    btn.onclick = () => { settings.mode = btn.dataset.mode; saveSettings(); paintMode(); paintStatus(); };
  }
  $('set-move').onchange = e => { settings.commitOnMove = e.target.checked; saveSettings(); };
  $('set-wipe').onclick = () => {
    if (!confirm('Erase every page? This cannot be undone.')) return;
    pages = [blankPage()];
    pageIdx = 0;
    media = {};
    undoStack.length = redoStack.length = 0;
    closePanels();
    clearSel();
    refreshAll();
    scheduleSave();
  };
  $('set-model').onchange = async e => {
    const model = e.target.value;
    try {
      await fetch('/api/model', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) });
      server = { ...server, ready: false, state: 'loading', model };
      paintStatus();
      setTimeout(pollStatus, 500);
    } catch { toast('Could not switch model'); }
  };

  // The recognizers the server offers (GET /api/status: backend is the default mode, backends
  // [{ name, label, default, ready, state, model, error }], docs/BOARD_BACKENDS.md). An old server
  // sends neither, and the Recognizer row stays hidden. Your pick is kept in the board's settings
  // and sent with each request; the server's default is never changed.
  function backendList() {
    const raw = server.backends;
    const list = Array.isArray(raw) ? raw
      : raw && typeof raw === 'object' ? Object.entries(raw).map(([name, v]) => ({ name, ...(v && typeof v === 'object' ? v : {}) })) : [];
    return list.map(x => (typeof x === 'string' ? { id: x, ok: true, state: '', label: '', model: '', error: '', def: false } : {
      id: String(x?.name ?? x?.id ?? ''), label: String(x?.label || ''), state: String(x?.state || ''),
      ok: x?.state !== 'error', def: !!x?.default, model: String(x?.model || ''), error: String(x?.error || ''),
    })).filter(x => x.id);
  }
  const defaultBackend = () => String(server.backend || backendList().find(x => x.def)?.id || backendList()[0]?.id || '');
  const backendName = id => backendList().find(x => x.id === id)?.label || BACKEND_NAMES[id] || id;
  const backendShort = id => ({ qwen: 'Qwen', unimumer: 'Uni-MuMER' })[id] || id;
  function backendChoice() { // '' = let the server choose
    const c = settings.backend;
    return c && backendList().some(x => x.id === c && x.ok) ? c : '';
  }
  // Re-recognize asks the other single backend for a second opinion, when the server has two.
  function secondOpinion(b) {
    const singles = backendList().filter(x => x.ok && x.id !== 'ensemble');
    if (singles.length < 2) return '';
    const cur = b.by || backendChoice() || defaultBackend();
    return (singles.find(x => x.id !== cur) || singles[0]).id;
  }
  const STATE_NOTE = { idle: 'loads on first use', loading: 'loading', starting: 'starting', downloading: 'downloading', error: 'unavailable' };
  function fillBackends() {
    const sel = $('set-backend'), list = backendList();
    $('set-backend-row').hidden = !list.length;
    if (!list.length) return;
    sel.textContent = '';
    const add = (value, text, disabled) => {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = text;
      o.disabled = !!disabled;
      o.selected = value === (backendChoice() || '');
      sel.appendChild(o);
    };
    const def = defaultBackend();
    add('', def ? `Server default: ${backendName(def)}` : 'Server default');
    for (const x of list) add(x.id, backendName(x.id) + (STATE_NOTE[x.state] ? ` · ${STATE_NOTE[x.state]}` : ''), !x.ok);
    const cur = list.find(x => x.id === (backendChoice() || def));
    $('set-backend-note').textContent = !cur ? '' : cur.error ? cur.error
      : [cur.model, cur.state, server.normalize === false ? 'crops sent unscaled' : ''].filter(Boolean).join(' · ');
  }

  async function openSettings() {
    closePanels();
    const panel = $('settings'), bar = $('toolbar').getBoundingClientRect();
    panel.hidden = false;
    $('gear').classList.add('on');
    fillBackends();
    // Drop from the gear: right-aligned with the toolbar, just under it.
    panel.style.left = clamp(bar.right - panel.offsetWidth, 8, W - panel.offsetWidth - 8) + 'px';
    panel.style.top = bar.bottom + 8 + 'px';
    for (const [id, key, fmt] of ranges) { $(id).value = settings[key]; $(id).nextElementSibling.textContent = fmt(settings[key]); }
    $('set-ink').checked = settings.inkAfter === 'faint';
    $('set-theme').value = settings.theme;
    paintMode();
    const models = $('set-model');
    const fill = (names, activeName) => {
      models.textContent = '';
      for (const name of names) {
        const o = document.createElement('option');
        o.value = o.textContent = name;
        o.selected = name === activeName;
        models.appendChild(o);
      }
    };
    fill([server.model || '…'], server.model);
    try {
      const r = await fetch('/api/models');
      const d = await r.json();
      if (!r.ok) throw new Error(d.error);
      fill(d.models.includes(d.active) ? d.models : [d.active, ...d.models], d.active);
    } catch { /* keep the single entry */ }
  }

  // ================================================================ keyboard
  // The 3D and Net tabs have their own keys; body[data-view] is unset until grapher.js starts.
  const onBoard = () => (document.body.dataset.view || 'board') === 'board';
  window.addEventListener('keydown', e => {
    if (!onBoard()) return;
    if (e.target.closest && e.target.closest('input, textarea, select')) {
      if (e.key === 'Escape') closePanels();
      return;
    }
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); redo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const actions = {
      m: () => setTool('math'), d: () => setTool('draw'), e: () => setTool('erase'),
      s: () => setTool('select'), l: () => setTool('laser'),
      h: toggleClean, t: () => setTheme(settings.theme === 'dark' ? 'light' : 'dark'), n: addPage,
      arrowright: () => gotoPage(pageIdx + 1), pagedown: () => gotoPage(pageIdx + 1),
      arrowleft: () => gotoPage(pageIdx - 1), pageup: () => gotoPage(pageIdx - 1),
      enter: convertNow, p: () => setHold(!hold),
      escape,
      delete: () => sel && selAction('delete'), backspace: () => sel && selAction('delete'),
    };
    if (/^[1-9]$/.test(k)) { e.preventDefault(); pickNumber(Number(k)); return; }
    if (actions[k]) { e.preventDefault(); actions[k](); }
  });

  // ================================================================ status, saving, export
  let pollTimer = 0;
  async function pollStatus() {
    clearTimeout(pollTimer);
    try {
      const r = await fetch('/api/status');
      server = await r.json();
    } catch {
      server = { ready: false, state: 'offline', error: 'Mathboard server is not running' };
    }
    paintStatus();
    pollTimer = setTimeout(pollStatus, server.ready ? 10000 : 1500);
  }
  function paintStatus() {
    const el = $('status'), txt = $('status-text');
    const queued = pages.reduce((n, p) => n + p.blocks.filter(waiting).length, 0);
    const be = backendChoice() || (backendList().length ? defaultBackend() : '');
    if (server.ready && hold && !inflight) {
      el.className = 'held';
      txt.textContent = queued ? `Hold · ${queued} waiting (Enter converts now)` : 'Hold · nothing converts until you release it';
    } else if (server.ready) { // ready shows only the dot, and this text on hover (style.css)
      el.className = inflight ? 'busy' : 'ready';
      txt.textContent = inflight ? (queued > 1 ? `Recognizing · ${queued - 1} more queued` : 'Recognizing…')
        : [server.model, be && backendName(be), mode() === 'off' ? 'conversion off (Enter converts)' : mode() === 'auto' && 'auto',
          lastMs != null && `${lastMs} ms`, lastTokens != null && `${lastTokens} tokens`,
          queued && `${queued} ${mode() === 'off' ? 'waiting' : 'queued'}`].filter(Boolean).join(' · ');
    } else if (server.state === 'error' || server.state === 'offline') {
      el.className = 'error';
      txt.textContent = server.error || 'Offline';
    } else {
      el.className = 'busy';
      txt.textContent = server.state === 'downloading' ? server.error : `Loading ${server.model || 'the model'}…`;
    }
  }

  let saveTimer = 0;
  function scheduleSave() {
    if (mirror) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const used = {}; // only images still on a page (the undo history isn't saved)
      for (const p of pages) for (const im of p.images) used[im.id] = media[im.id];
      try { localStorage.setItem('mathboard.board', JSON.stringify({ pages, pageIdx, media: used })); }
      catch { toast('Autosave failed: browser storage is full. Export your notes.'); }
    }, 600);
  }

  // Converted expressions, and readings still waiting under their ink; Draw-pen ink and images are
  // left out.
  function exportNotes() {
    const out = ['# Mathboard notes', '', `_${new Date().toLocaleString()}_`, ''];
    let count = 0;
    const texOf = b => (b.status === 'done' && b.latex) || b.pv?.latex || '';
    pages.forEach((p, i) => {
      const blocks = p.blocks.filter(b => b.box && texOf(b));
      if (!blocks.length) return;
      out.push(`## Page ${i + 1}`, '');
      const rows = []; // reading order: rows top to bottom, left to right within a row
      for (const b of [...blocks].sort((a, c) => a.box[1] - c.box[1])) {
        const cy = (b.box[1] + b.box[3]) / 2;
        const row = rows.find(r => cy >= r.top && cy <= r.bottom);
        if (row) row.items.push(b);
        else rows.push({ top: b.box[1], bottom: b.box[3], items: [b] });
      }
      for (const r of rows) {
        for (const b of r.items.sort((a, c) => a.box[0] - c.box[0])) { out.push('$$', texOf(b), '$$', ''); count++; }
      }
    });
    if (!count) return toast('Nothing converted yet');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([out.join('\n')], { type: 'text/markdown' }));
    a.download = `mathboard-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    toast(`Exported ${count} expression${count === 1 ? '' : 's'}`);
  }

  let toastTimer = 0;
  function toast(msg, ms = 2400) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, ms);
  }

  // ================================================================ start
  document.documentElement.dataset.theme = settings.theme;
  window.addEventListener('resize', () => { resize(); renderTypeset(); renderPreviews(); });
  // KaTeX fonts load lazily; re-measure typeset once they arrive.
  document.fonts.addEventListener('loadingdone', () => {
    for (const el of tsEls.values()) el._key = null;
    for (const el of pvEls.values()) el._key = null;
    renderTypeset();
    renderPreviews();
  });
  // Typeset laid out while another tab hid the board has no size yet (see paintTypeset). H in the
  // 3D and Net tabs (graph/features/lecture.js, nn/nn.js) toggles body.clean too: the outlines follow it.
  let wasClean = document.body.classList.contains('clean');
  new MutationObserver(() => {
    const clean = document.body.classList.contains('clean');
    if (clean !== wasClean) { wasClean = clean; redraw(); }
    if (onBoard()) { renderTypeset(); renderPreviews(); }
  }).observe(document.body, { attributes: true, attributeFilter: ['data-view', 'class'] });
  resize();
  refreshAll();
  pollStatus();

  window.mathboard = {
    get pages() { return pages; }, get server() { return server; }, get pageIdx() { return pageIdx; },
    get settings() { return settings; }, get hold() { return hold; }, get inputType() { return inputType; },
    writingSize: () => pageSize(pg()),
    debug: { join: () => active?.join || null, get selection() { return sel && [...sel.ids]; }, }, // for browser checks
  };
})();
