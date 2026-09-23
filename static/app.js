'use strict';
// Mathboard: write math with the Math pen, it is grouped into expressions by proximity,
// each expression is rasterized and sent to the local vision model after the pen goes idle,
// and the handwriting is replaced in place by KaTeX.
(() => {
  // ================================================================ settings
  const DEFAULTS = {
    tool: 'math', mathColor: 'ink', drawColor: 'c1',
    delay: 700,        // ms of pen idle on an expression before converting it
    reach: 1.0,        // multiplier on the grouping distances below
    maxFont: 56,       // px cap for typeset size
    inkAfter: 'hide',  // 'hide' | 'faint': handwriting under converted expressions
    theme: 'dark',
  };
  const REACH_X = 56;      // px sideways a stroke may sit from an expression and still join it
  const REACH_Y = 16;      // px vertically (small, so separate lines stay separate)
  const ERASER_R = 12;
  const PEN_W = 3.2;
  const RASTER_MAX = 768;  // longest side of the image sent to the model
  const LASER_MS = 650;
  const COLORS = {
    dark:  { ink: '#ecebe4', c1: '#ffd166', c2: '#5ac8fa', c3: '#ff6b9a', c4: '#7ee08a', c5: '#ff9f43' },
    light: { ink: '#1d1d1f', c1: '#c98500', c2: '#0a6fd8', c3: '#d6245a', c4: '#1f9d55', c5: '#e8590c' },
  };

  // The audience window (index.html?audience, see graph/features/lecture.js) is a read-only mirror:
  // it never recognizes (the presenter does) and never saves (its copy of the board goes stale).
  const mirror = new URLSearchParams(location.search).has('audience');
  const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
  const settings = Object.assign({}, DEFAULTS, load('mathboard.settings', {}));
  const saveSettings = () => { if (!mirror) localStorage.setItem('mathboard.settings', JSON.stringify(settings)); };

  // ================================================================ helpers
  const $ = id => document.getElementById(id);
  const uid = () => Math.random().toString(36).slice(2, 10);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const r1 = v => Math.round(v * 10) / 10;
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const color = token => (COLORS[settings.theme] || COLORS.dark)[token] || token;
  const grow = (r, dx, dy) => [r[0] - dx, r[1] - dy, r[2] + dx, r[3] + dy];
  const shift = (r, dx, dy) => [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy];
  const overlaps = (a, b) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
  const inRect = (x, y, r) => x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];

  function unionBox(boxes) {
    const u = [Infinity, Infinity, -Infinity, -Infinity];
    for (const b of boxes) { u[0] = Math.min(u[0], b[0]); u[1] = Math.min(u[1], b[1]); u[2] = Math.max(u[2], b[2]); u[3] = Math.max(u[3], b[3]); }
    return u;
  }
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
  // stroke = { id, kind: 'math'|'draw', color, w, pr (pressure?), pts: [[x,y,p]], box, block, conv }
  // block  = one math expression: { id, latex, status, ver, due, box, tsBox, tsRect, keepTs, color }
  //   status: pending -> busy -> done | empty (model saw no math) | error (unrenderable / failed)
  //   stroke.conv: the stroke is covered by the block's current typeset (so its ink is hidden)
  // image  = a picture under the ink (e.g. a 3D-tab snapshot): { id, box }. Its data URL lives in
  //   media[id], outside the page, so undo snapshots (page JSON) stay small.
  const blankPage = () => ({ id: uid(), strokes: [], blocks: [], images: [] });
  let pages = [blankPage()], pageIdx = 0;
  let media = {};          // image id -> data URL
  let active = null;       // the pointer gesture in progress
  let inflight = null;     // the recognition request in progress
  let server = { ready: false, state: 'connecting' };
  let lastMs = null, lastMath = null, penSeen = false, hover = null;
  let lastPen = settings.tool === 'draw' ? 'draw' : 'math';

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

  // Mark an expression as changed. keepTs: leave the old typeset up (dimmed) while it re-converts;
  // otherwise its handwriting is revealed until the new result arrives.
  function touch(p, b, keepTs) {
    const shown = tsVisible(b);
    b.ver = uid();
    b.status = 'pending';
    b.due = Date.now() + settings.delay;
    b.errors = 0;
    b.keepTs = keepTs && shown;
    if (!b.keepTs) for (const s of strokesOf(p, b)) s.conv = false;
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
    refreshAll();
    scheduleSave();
  }
  const undo = () => stepHistory(undoStack, redoStack);
  const redo = () => stepHistory(redoStack, undoStack);

  // ================================================================ ink rendering
  const inkCv = $('ink'), fxCv = $('fx'), tsLayer = $('typeset');
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

  function redraw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = ctx.lineJoin = 'round';
    const p = pg(), bm = new Map(p.blocks.map(b => [b.id, b]));
    const faint = settings.inkAfter === 'faint';
    for (const s of p.strokes) {
      if (inkHidden(s, s.block && bm.get(s.block))) { if (faint) drawStroke(ctx, s, 0.16); continue; }
      drawStroke(ctx, s);
    }
    if (active?.stroke) drawStroke(ctx, active.stroke);

    if (!document.body.classList.contains('clean')) { // faint outline = "converting soon"
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1;
      for (const b of p.blocks) {
        const waiting = b.status === 'pending' || b.status === 'busy';
        if (!b.box || (!waiting && b.status !== 'error')) continue;
        ctx.strokeStyle = b.status === 'error' ? 'rgba(255,90,90,.75)'
          : settings.theme === 'dark' ? 'rgba(255,255,255,.16)' : 'rgba(0,0,0,.16)';
        const r = grow(b.box, 6, 6);
        ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      }
      ctx.setLineDash([]);
    }
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
  function toHTML(src) { // null if KaTeX can't render it (after a couple of repairs)
    if (htmlCache.has(src)) return htmlCache.get(src);
    const tries = [src];
    if (src.includes('\\\\') && !src.includes('\\begin')) tries.push(`\\begin{gathered}${src}\\end{gathered}`);
    const bal = balanceBraces(src);
    if (bal !== src) tries.push(bal);
    let html = null;
    for (const t of tries) {
      try { html = katex.renderToString(t, { displayMode: true, throwOnError: true, strict: 'ignore', output: 'html' }); break; }
      catch { /* try the next repair */ }
    }
    if (htmlCache.size > 500) htmlCache.clear();
    htmlCache.set(src, html);
    return html;
  }

  const tsEls = new Map(); // block id -> element

  // Size the typeset to fit inside the handwriting's box (capped by maxFont).
  function paintTypeset(el, b, box) {
    el.innerHTML = toHTML(b.latex) || '';
    el.style.color = color(b.color);
    const k = el.querySelector('.katex') || el;
    el.style.fontSize = '40px';
    let r = k.getBoundingClientRect();
    const bw = Math.max(box[2] - box[0], 24), bh = Math.max(box[3] - box[1], 24);
    const fit = Math.min(bh / (r.height || 1), (bw * 1.1) / (r.width || 1));
    el.style.fontSize = clamp(40 * fit, 14, settings.maxFont) + 'px';
    r = k.getBoundingClientRect();
    el._w = r.width;
    el._h = r.height;
    if (!r.width) el._key = null; // board hidden (3D tab): measure again once it is shown
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
      if (el._key !== key) { el._key = key; paintTypeset(el, b, box); }
      const top = (box[1] + box[3]) / 2 - el._h / 2;
      el.style.transform = `translate(${box[0]}px, ${top}px)`;
      el.classList.toggle('stale', b.status !== 'done');
      b.tsRect = [box[0], top, box[0] + el._w, top + el._h];
    }
    for (const [id, el] of tsEls) if (!live.has(id)) { el.remove(); tsEls.delete(id); }
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
        el.style.cssText = 'position: absolute; left: 0; top: 0; border-radius: 4px; outline-offset: 3px;';
        imgEls.set(im.id, el);
      }
      if (imgLayer.children[i] !== el) imgLayer.insertBefore(el, imgLayer.children[i] || null); // page order = z order
      el.style.transform = `translate(${im.box[0]}px, ${im.box[1]}px)`;
      el.style.width = `${im.box[2] - im.box[0]}px`;
      el.style.height = `${im.box[3] - im.box[1]}px`;
      el.style.outline = im.id === editingImg ? '2px dashed var(--accent)' : '';
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

  function refreshAll() { redraw(); renderImages(); renderTypeset(); paintToolbar(); paintStatus(); }

  // ================================================================ grouping strokes into expressions
  function commitMath(p, s) {
    const rx = REACH_X * settings.reach, ry = REACH_Y * settings.reach;
    let hits = p.blocks.filter(b => b.box && overlaps(grow(b.box, rx, ry), s.box));
    // Writing quickly: allow a wider sideways gap to the expression you just wrote in.
    if (!hits.length && lastMath && Date.now() - lastMath.t < 1500) {
      const b = blockOf(p, lastMath.id);
      if (b?.box && overlaps(grow(b.box, rx * 2, ry), s.box)) hits = [b];
    }
    let b;
    if (!hits.length) {
      b = { id: uid(), latex: '', status: 'pending', color: s.color, box: null };
      p.blocks.push(b);
    } else {
      b = hits[0];
      for (const o of hits.slice(1)) { // stroke bridges several expressions (e.g. matrix brackets): merge
        for (const t of p.strokes) if (t.block === o.id) t.block = b.id;
        removeBlock(p, o);
      }
    }
    const merged = hits.length > 1;
    if (merged) b.latex = '';
    s.block = b.id;
    s.conv = false;
    refreshBox(p, b);
    touch(p, b, !merged);
    lastMath = { id: b.id, t: Date.now() };
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
  function rasterize(p, b) {
    const strokes = strokesOf(p, b), box = unionBox(strokes.map(s => s.box));
    const bw = box[2] - box[0], bh = box[3] - box[1];
    const pad = Math.max(14, 0.1 * Math.max(bw, bh));
    const w = bw + 2 * pad, h = bh + 2 * pad;
    const k = Math.min(2, RASTER_MAX / Math.max(w, h));
    const cw = Math.max(56, Math.round(w * k)), ch = Math.max(56, Math.round(h * k));
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const g = cv.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, cw, ch);
    g.setTransform(k, 0, 0, k, (cw - w * k) / 2 + (pad - box[0]) * k, (ch - h * k) / 2 + (pad - box[1]) * k);
    g.strokeStyle = '#000';
    g.lineCap = g.lineJoin = 'round';
    // ~7 px strokes at full size: thin mouse strokes made the model read 0s as circles and drop columns.
    g.lineWidth = Math.max(5, 0.009 * Math.max(cw, ch)) / k;
    for (const s of strokes) { tracePath(g, s); g.stroke(); }
    return cv.toDataURL('image/png');
  }

  function pump() {
    if (mirror || inflight || !server.ready) return;
    if (active?.stroke?.kind === 'math') return; // wait for the pen to lift
    const now = Date.now();
    let pick = null;
    pages.forEach((p, i) => {
      for (const b of p.blocks) {
        if (b.status !== 'pending' || b.due > now) continue;
        const rank = (i === pageIdx ? 0 : 1e13) + b.due; // current page first
        if (!pick || rank < pick.rank) pick = { p, b, rank };
      }
    });
    if (pick) recognize(pick.p, pick.b);
  }

  async function recognize(p, b) {
    const id = b.id, ver = b.ver;
    b.status = 'busy';
    const image = rasterize(p, b);
    window.__mathboardLastImage = image;
    inflight = { id, ver };
    paintStatus();
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      lastMs = data.ms;
      applyResult(id, ver, data.latex || '');
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

  function applyResult(id, ver, latex) {
    const f = findBlock(id);
    if (!f || f.b.ver !== ver) return; // edited while we waited; a newer request will follow
    const { p, b } = f;
    latex = latex.trim();
    b.latex = latex;
    b.status = !latex ? 'empty' : toHTML(latex) ? 'done' : 'error';
    b.keepTs = false;
    if (b.status === 'done') b.tsBox = b.box.slice();
    for (const s of strokesOf(p, b)) s.conv = b.status === 'done';
    if (p === pg()) { redraw(); renderTypeset(); }
    scheduleSave();
  }

  setInterval(pump, 100);

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
    if (changed) { redraw(); renderTypeset(); }
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
    if (laser.length || active?.tool === 'laser') requestAnimationFrame(fxFrame);
    else fxOn = false;
  }

  // ================================================================ pointer input
  function onDown(e) {
    if (active) return;
    if (e.pointerType === 'pen') penSeen = true;
    else if (e.pointerType === 'touch' && penSeen) return; // palm rejection once a pen has been used
    closePanels();
    let tool = settings.tool;
    if (e.button === 2 || e.button === 5 || (e.buttons & 32)) tool = 'erase'; // right button, pen eraser end
    try { fxCv.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    const x = e.clientX, y = e.clientY;
    active = { id: e.pointerId, tool, snap: snapshot(), undone: false, last: [x, y], start: [x, y] };

    if (tool === 'math' || tool === 'draw') {
      const s = {
        id: uid(), kind: tool, color: settings[tool + 'Color'], w: PEN_W, pr: e.pointerType === 'pen',
        pts: [[r1(x), r1(y), Math.round((e.pressure || 0.5) * 100) / 100]], block: null, conv: false,
      };
      active.stroke = s;
      ctx.lineCap = ctx.lineJoin = 'round';
      drawStroke(ctx, s);
    } else if (tool === 'erase') {
      if (eraseAt(x, y)) { redraw(); renderTypeset(); }
      kickFx();
    } else if (tool === 'select') {
      active.hit = pickAt(x, y);
      active.moved = false;
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
        P.push([r1(x), r1(y), Math.round((ev.pressure || 0.5) * 100) / 100]);
        if (P.length >= 3) {
          ctx.strokeStyle = color(s.color);
          ctx.lineCap = ctx.lineJoin = 'round';
          drawSeg(ctx, s, P.length - 2);
        }
      } else if (active.tool === 'erase') {
        eraseLine(active.last, [x, y]);
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
    if (a.stroke) {
      const s = a.stroke, p = pg();
      s.box = strokeBox(s);
      commitUndo(a.snap);
      p.strokes.push(s);
      if (s.kind === 'math') commitMath(p, s);
      redraw();
      renderTypeset();
      paintStatus();
      scheduleSave();
    } else if (a.tool === 'select') {
      if (a.moved) scheduleSave();
      else if (a.hit?.b) openEditor(a.hit.b);
      else if (a.hit?.img) openImageEditor(a.hit.img);
    } else if (a.undone) {
      scheduleSave();
    }
    kickFx();
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
  edSend.textContent = 'Send to 3D';
  edSend.title = 'Add this expression as a row in the 3D tab';
  $('ed-copy').after(edSend);
  const edHint = document.createElement('p');
  edHint.className = 'hint';
  edHint.style.marginTop = '0';
  edHint.textContent = 'Drag to move. Drag the bottom-right corner to resize.';
  edPrev.after(edHint);

  function editorMode(image) {
    edSrc.hidden = edPrev.hidden = image;
    edHint.hidden = !image;
    for (const btn of editor.querySelectorAll('.row button')) btn.hidden = image && btn.id !== 'ed-del';
    editor.style.width = image ? 'auto' : '';
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
    edSrc.value = b.latex || '';
    previewEdit();
    placeEditor(tsVisible(b) && b.tsRect ? b.tsRect : b.box);
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
  $('ed-rerun').onclick = () => {
    const b = edBlock();
    if (!b) return;
    touch(pg(), b, false);
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
    if (document.activeElement?.closest?.('.panel')) document.activeElement.blur();
    editor.hidden = true;
    $('settings').hidden = true;
    editing = null;
    if (editingImg) { editingImg = null; renderImages(); }
  }

  // ================================================================ toolbar, pages, settings
  const swatchKeys = ['ink', 'c1', 'c2', 'c3', 'c4', 'c5'];
  const swatchEls = swatchKeys.map(key => {
    const el = document.createElement('button');
    el.className = 'swatch';
    el.dataset.c = key;
    el.title = key === 'ink' ? 'Default ink' : 'Colour';
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
      sw.style.background = color(sw.dataset.c);
      sw.classList.toggle('on', sw.dataset.c === current);
    }
    $('pageno').textContent = `${pageIdx + 1} / ${pages.length}`;
  }
  for (const btn of document.querySelectorAll('#tools button')) btn.onclick = () => setTool(btn.dataset.tool);

  function gotoPage(i) {
    if (active) return;
    pageIdx = clamp(i, 0, pages.length - 1);
    closePanels();
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
    refreshAll();
    scheduleSave();
    toast('Page cleared (Ctrl+Z brings it back)');
  }
  function setTheme(t) {
    settings.theme = t;
    document.documentElement.dataset.theme = t;
    saveSettings();
    refreshAll();
  }
  function toggleClean() {
    const on = document.body.classList.toggle('clean');
    closePanels();
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
  $('gear').onclick = () => ($('settings').hidden ? openSettings() : closePanels());

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
  $('set-ink').onchange = e => { settings.inkAfter = e.target.value; saveSettings(); redraw(); };
  $('set-theme').onchange = e => setTheme(e.target.value);
  $('set-wipe').onclick = () => {
    if (!confirm('Erase every page? This cannot be undone.')) return;
    pages = [blankPage()];
    pageIdx = 0;
    media = {};
    undoStack.length = redoStack.length = 0;
    closePanels();
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

  async function openSettings() {
    closePanels();
    $('settings').hidden = false;
    for (const [id, key, fmt] of ranges) { $(id).value = settings[key]; $(id).nextElementSibling.textContent = fmt(settings[key]); }
    $('set-ink').value = settings.inkAfter;
    $('set-theme').value = settings.theme;
    const sel = $('set-model');
    const fill = (names, activeName) => {
      sel.textContent = '';
      for (const name of names) {
        const o = document.createElement('option');
        o.value = o.textContent = name;
        o.selected = name === activeName;
        sel.appendChild(o);
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
      escape: closePanels,
    };
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
    const queued = pages.reduce((n, p) => n + p.blocks.filter(b => b.status === 'pending' || b.status === 'busy').length, 0);
    if (server.ready) {
      el.className = inflight ? 'busy' : 'ready';
      txt.textContent = [server.model, lastMs != null && `${lastMs} ms`, queued && `${queued} queued`].filter(Boolean).join(' · ');
    } else if (server.state === 'error' || server.state === 'offline') {
      el.className = 'error';
      txt.textContent = server.error || 'offline';
    } else {
      el.className = 'busy';
      txt.textContent = server.state === 'downloading' ? server.error : `loading ${server.model || 'model'}…`;
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

  function exportNotes() { // converted expressions only: Draw-pen ink and images are left out
    const out = ['# Mathboard notes', '', `_${new Date().toLocaleString()}_`, ''];
    let count = 0;
    pages.forEach((p, i) => {
      const blocks = p.blocks.filter(b => b.status === 'done' && b.latex && b.box);
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
        for (const b of r.items.sort((a, c) => a.box[0] - c.box[0])) { out.push('$$', b.latex, '$$', ''); count++; }
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
  window.addEventListener('resize', () => { resize(); renderTypeset(); });
  // KaTeX fonts load lazily; re-measure typeset once they arrive.
  document.fonts.addEventListener('loadingdone', () => {
    for (const el of tsEls.values()) el._key = null;
    renderTypeset();
  });
  // Typeset laid out while another tab hid the board has no size yet (see paintTypeset). H in the
  // 3D and Net tabs (graph/features/lecture.js, nn/nn.js) toggles body.clean too: the outlines follow it.
  let wasClean = document.body.classList.contains('clean');
  new MutationObserver(() => {
    const clean = document.body.classList.contains('clean');
    if (clean !== wasClean) { wasClean = clean; redraw(); }
    if (onBoard()) renderTypeset();
  }).observe(document.body, { attributes: true, attributeFilter: ['data-view', 'class'] });
  resize();
  refreshAll();
  pollStatus();

  window.mathboard = { get pages() { return pages; }, get server() { return server; }, get pageIdx() { return pageIdx; } };
})();
