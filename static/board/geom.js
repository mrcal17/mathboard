'use strict';
// Board geometry, shared by app.js and the node tests (tests/board_geom.test.mjs): the writing size,
// which expression a stroke joins (size-relative reach, the fraction-bar rule, the new-line guard),
// input profiles for mouse / pen / touch, scratch-out, and lasso hit tests. Pure functions on
// boxes [x0, y0, x1, y1] and point lists [[x, y, ...], ...]. A classic script loaded before app.js
// that registers globalThis.mathboardBoard.geom. docs/BOARD_UX.md explains the numbers.
(() => {
  // ------------------------------------------------------------ constants (in writing sizes S)
  const REACH_X = 0.9;     // sideways reach from an expression's visible box
  const REACH_Y = 0.35;    // vertical reach (small, so separate lines stay separate)
  const BAR_MIN_W = 1.3;   // a fraction bar is at least 1.3 S wide ...
  const BAR_ASPECT = 4;    // ... and 4x wider than tall (an = sign is about 1 S wide)
  const BAR_REACH = 1.2;   // a bar gathers what sits within 1.2 S above and below it ...
  const BAR_SIDE = 0.15;   // ... over its own width plus 15% on each side
  const BURST_X = 2;       // sideways reach multiplier for the expression you just wrote in
  const SCRIPT_Y = 0.6;    // a small stroke just right of a group (an exponent, a subscript) may float
  const SCRIPT_H = 0.8;    // ... this far above or below it, if it is at most 0.8 S tall
  const GLYPHS = 30;       // the writing size is the median height of the last 30 glyphs
  const MIN_GLYPHS = 3;    // below this, the input profile's default size is used
  const SIZE_MIN = 20, SIZE_MAX = 180;

  // Timing and thresholds by pointer type. idle multiplies Settings > Convert after (the mouse
  // value). ceiling: the longest a hovering pointer can hold off conversion (touch can't hover).
  // burst: ms after a stroke in which the next one may sit twice as far sideways. size: the
  // writing size before there are 3 glyphs. dot: glyphs smaller than this are not counted.
  const PROFILES = {
    mouse: { size: 72, idle: 1, ceiling: 3000, burst: 2500, dot: 6, scratchRev: 3, tapPx: 4, tapMs: 350 },
    pen: { size: 44, idle: 0.75, ceiling: 2400, burst: 1500, dot: 4, scratchRev: 4, tapPx: 6, tapMs: 300 },
    touch: { size: 56, idle: 0.85, ceiling: 0, burst: 1800, dot: 6, scratchRev: 4, tapPx: 10, tapMs: 300 },
  };
  const profile = type => PROFILES[type] || PROFILES.mouse;

  // ------------------------------------------------------------ boxes
  const grow = (r, dx, dy = dx) => [r[0] - dx, r[1] - dy, r[2] + dx, r[3] + dy];
  const overlaps = (a, b) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
  const inRect = (x, y, r) => x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];
  const area = r => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
  const center = r => [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2];
  function unionBox(boxes) {
    const u = [Infinity, Infinity, -Infinity, -Infinity];
    for (const b of boxes) {
      if (!b) continue;
      u[0] = Math.min(u[0], b[0]); u[1] = Math.min(u[1], b[1]); u[2] = Math.max(u[2], b[2]); u[3] = Math.max(u[3], b[3]);
    }
    return u;
  }
  function interArea(a, b) {
    return Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  }
  const coverage = (r, by) => (area(r) ? interArea(r, by) / area(r) : 0); // share of r inside `by`
  function median(v) {
    if (!v.length) return NaN;
    const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // ------------------------------------------------------------ writing size
  // Strokes in writing order become glyphs: a stroke that overlaps the previous glyph by a quarter
  // of the smaller box joins it (the two strokes of an x or a 4), otherwise it starts a new one.
  function glyphBoxes(boxes) {
    const out = [];
    for (const b of boxes) {
      const g = out[out.length - 1];
      if (g && interArea(g, b) >= 0.25 * Math.max(1, Math.min(area(g), area(b)))) {
        g[0] = Math.min(g[0], b[0]); g[1] = Math.min(g[1], b[1]); g[2] = Math.max(g[2], b[2]); g[3] = Math.max(g[3], b[3]);
      } else out.push(b.slice(0, 4));
    }
    return out;
  }
  const isDot = (g, dot) => Math.max(g[2] - g[0], g[3] - g[1]) < dot;
  const isFlat = g => g[2] - g[0] > BAR_ASPECT * Math.max(1, g[3] - g[1]); // bars, minus, the strokes of =
  // Median height of the last GLYPHS glyphs, skipping dots and bars; `fallback` below `min` glyphs.
  function writingSize(boxes, fallback = 48, dot = 6, min = MIN_GLYPHS) {
    const hs = glyphBoxes(boxes).filter(g => !isDot(g, dot) && !isFlat(g)).slice(-GLYPHS).map(g => g[3] - g[1]);
    if (hs.length < min) return fallback;
    return Math.min(SIZE_MAX, Math.max(SIZE_MIN, median(hs)));
  }

  // ------------------------------------------------------------ fraction bars
  function isBar(box, S) {
    const w = box[2] - box[0], h = Math.max(1, box[3] - box[1]);
    return w >= BAR_MIN_W * S && w >= BAR_ASPECT * h;
  }
  function barZone(box, S, reach = 1) {
    const side = BAR_SIDE * (box[2] - box[0]), dy = BAR_REACH * S * reach;
    return [box[0] - side, box[1] - dy, box[2] + side, box[3] + dy];
  }
  // A group sits in a bar's zone when its middle is within the zone's height and at least 30% of
  // its width is over the zone (so a neighbour on the bar's own line is not pulled in).
  function groupInZone(zone, g) {
    const cy = (g[1] + g[3]) / 2;
    if (cy < zone[1] || cy > zone[3]) return false;
    const ov = Math.min(zone[2], g[2]) - Math.max(zone[0], g[0]);
    return ov >= 0.3 * Math.max(1, g[2] - g[0]);
  }
  // A new stroke lands in a bar's zone when its centre does.
  const strokeInZone = (zone, s) => inRect(...center(s), zone);

  // ------------------------------------------------------------ joining
  // New-line guard: a stroke that starts left of a group (or within 0.3 S of its left edge, since
  // lines written under each other share a margin) and lies entirely below it starts a new line, so
  // the vertical reach doesn't merge stacked lines.
  const NEWLINE_X = 0.3;
  const newLine = (s, g, S = 0) => s[0] < g[0] + NEWLINE_X * S && s[1] > g[3];

  // The zone right of a group where a small exponent or subscript may float.
  const scriptZone = (g, S, reach = 1) => [g[2] - 0.5 * S * reach, g[1] - SCRIPT_Y * S * reach, g[2] + REACH_X * S * reach, g[3] + SCRIPT_Y * S * reach];

  // Which groups a stroke with box `sbox` joins. groups: [{ id, box, bars: [box] }] where box is
  // the group's visible geometry and bars its visible fraction-bar strokes. opt.burst: the id of
  // the group the last stroke went to, if it was within the profile's burst time. Returns ids in
  // `groups` order (the first one survives a merge).
  function joinTargets(groups, sbox, S, reach = 1, opt = {}) {
    const rx = REACH_X * S * reach, ry = REACH_Y * S * reach;
    const bar = isBar(sbox, S), zone = bar ? barZone(sbox, S, reach) : null;
    const small = sbox[3] - sbox[1] <= SCRIPT_H * S;
    const near = [], strong = []; // by reach / by a fraction bar (bars always win)
    for (const g of groups) {
      if (!g.box) continue;
      if ((zone && groupInZone(zone, g.box)) || (g.bars || []).some(b => strokeInZone(barZone(b, S, reach), sbox))) strong.push(g);
      else if (!newLine(sbox, g.box, S) && (overlaps(grow(g.box, rx, ry), sbox) || (small && overlaps(scriptZone(g.box, S, reach), sbox)))) near.push(g);
    }
    // The new-line guard again, for the line the stroke is on: a stroke that reaches both a line and
    // a group that started as a new line under it (b = 1 written under A = ...) stays with the new one.
    const kept = near.filter(a => !near.some(o => o !== a && newLine(unionBox([o.box, sbox]), a.box, S)));
    const hits = groups.filter(g => strong.includes(g) || kept.includes(g)).map(g => g.id);
    if (!hits.length && opt.burst) { // writing fast: the expression you're in reaches further sideways
      const g = groups.find(x => x.id === opt.burst);
      if (g?.box && !newLine(sbox, g.box, S) && overlaps(grow(g.box, rx * BURST_X, ry), sbox)) hits.push(g.id);
    }
    return hits;
  }
  // Where a pen-down still counts as "writing in this group" (hover pause, moving on).
  function nearZones(g, S, reach = 1, burst = false) {
    const rx = REACH_X * S * reach * (burst ? BURST_X : 1), ry = REACH_Y * S * reach;
    return [grow(g.box, rx, ry), scriptZone(g.box, S, reach), ...(g.bars || []).map(b => barZone(b, S, reach))];
  }
  const isNear = (g, x, y, S, reach, burst) => nearZones(g, S, reach, burst).some(z => inRect(x, y, z));

  // ------------------------------------------------------------ scratch-out
  // Direction reversals of a sequence, with hysteresis `tol` so jitter isn't counted.
  function reversals(vals, tol) {
    let dir = 0, hi = vals[0], lo = vals[0], n = 0;
    for (const v of vals) {
      if (dir === 0) {
        hi = Math.max(hi, v); lo = Math.min(lo, v);
        if (v - lo > tol) { dir = 1; hi = v; } else if (hi - v > tol) { dir = -1; lo = v; }
      } else if (dir > 0) {
        if (v > hi) hi = v; else if (hi - v > tol) { n++; dir = -1; lo = v; }
      } else if (v < lo) lo = v; else if (v - lo > tol) { n++; dir = 1; hi = v; }
    }
    return n;
  }
  function pathLength(pts) {
    let L = 0;
    for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    return L;
  }
  function ptsBox(pts) {
    return unionBox(pts.map(p => [p[0], p[1], p[0], p[1]]));
  }
  // A zig-zag: at least minRev sideways reversals, each swing a quarter of the extent, with a path
  // several times the extent. Up-and-down zig-zags need minRev + 2 reversals packed closer than a
  // fifth of their height each, since m, w and M have 3 to 5 reversals spread over their width.
  // Only the shape: app.js also requires the zig-zag to cross existing ink before it erases.
  function looksLikeScratch(pts, minRev = 3) {
    if (pts.length < 6) return false;
    const b = ptsBox(pts), w = b[2] - b[0], h = b[3] - b[1];
    if (Math.max(w, h) < 10) return false;
    const rx = reversals(pts.map(p => p[0]), Math.max(4, 0.25 * w));
    const ry = reversals(pts.map(p => p[1]), Math.max(4, 0.25 * h));
    const L = pathLength(pts);
    return (rx >= minRev && L >= 2.5 * w) || (ry >= minRev + 2 && w / ry <= 0.22 * h && L >= 2.5 * h);
  }
  function segCross(a, b, c, d) {
    const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    const d1 = o(c, d, a), d2 = o(c, d, b), d3 = o(a, b, c), d4 = o(a, b, d);
    return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0)) && d1 !== 0 && d2 !== 0 && d3 !== 0 && d4 !== 0;
  }
  // How many times polyline A crosses polyline B.
  function crossings(A, B) {
    let n = 0;
    for (let i = 1; i < A.length; i++) {
      const a = A[i - 1], b = A[i];
      const ax0 = Math.min(a[0], b[0]), ax1 = Math.max(a[0], b[0]), ay0 = Math.min(a[1], b[1]), ay1 = Math.max(a[1], b[1]);
      for (let j = 1; j < B.length; j++) {
        const c = B[j - 1], d = B[j];
        if (Math.max(c[0], d[0]) < ax0 || Math.min(c[0], d[0]) > ax1 || Math.max(c[1], d[1]) < ay0 || Math.min(c[1], d[1]) > ay1) continue;
        if (segCross(a, b, c, d)) n++;
      }
    }
    return n;
  }

  // ------------------------------------------------------------ lasso
  function pointInPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  const fracInside = (pts, poly) => (pts.length ? pts.filter(p => pointInPoly(p[0], p[1], poly)).length / pts.length : 0);
  // Share of a rectangle inside a polygon, sampled on a 5 x 5 grid.
  function rectInside(r, poly) {
    let n = 0;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      if (pointInPoly(r[0] + ((r[2] - r[0]) * (i + 0.5)) / 5, r[1] + ((r[3] - r[1]) * (j + 0.5)) / 5, poly)) n++;
    }
    return n / 25;
  }
  const rectPoly = r => [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]];

  const geom = {
    REACH_X, REACH_Y, BAR_MIN_W, BAR_ASPECT, BAR_REACH, BAR_SIDE, BURST_X, SCRIPT_Y, SCRIPT_H, NEWLINE_X, GLYPHS, MIN_GLYPHS, PROFILES,
    profile, grow, overlaps, inRect, area, center, unionBox, interArea, coverage, median,
    glyphBoxes, writingSize, isBar, barZone, groupInZone, strokeInZone, newLine, scriptZone, joinTargets, nearZones, isNear,
    reversals, pathLength, ptsBox, looksLikeScratch, segCross, crossings,
    pointInPoly, fracInside, rectInside, rectPoly,
  };
  globalThis.mathboardBoard = Object.assign(globalThis.mathboardBoard || {}, { geom });
})();
