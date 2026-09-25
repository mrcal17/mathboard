// Domain / codomain side-by-side view of a linear map.
//   map(A)   A is m×n with m, n in {2, 3}. While the row is visible the 3D view splits: the main
//            scene becomes the domain R^n (left) and a second scene shows the codomain R^m (right).
//            Vector, point, span, parallelogram and parallelepiped rows are mapped by A in their own
//            colour (labelled A u), the domain's integer grid is mapped too, the kernel is shaded red
//            in the domain and the image in the codomain, and the readout checks rank + nullity = n.
// "Link cameras" (toolbar) ties the two rotations together when both sides are 3D.
// Pure helpers are exported for tests/dual.test.mjs; scene.js loads in install().
import { registerFunction, registerType, parseLine } from '../lang.js';
import * as la from '../linalg.js';

export const KER_COLOR = '#e5484d';
export const IMG_COLOR = '#12a594';
const MAPPABLE = new Set(['vec', 'point', 'span', 'parallelogram', 'parallelepiped']);
const STORE = 'mathboard.dual';

export const pad3 = v => [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

// ================================================================ pure helpers

export function mapInfo(M) {
  const kernel = la.nullspace(M).map(pad3), image = la.colspace(M).map(pad3);
  return { m: M.length, n: M[0].length, rank: image.length, nullity: kernel.length, kernel, image };
}

export function mapValue(M) {
  const m = M.length, n = M[0].length;
  if (m < 2 || m > 3 || n < 2 || n > 3) {
    throw new Error(`map needs a matrix with 2 or 3 rows and 2 or 3 columns, got a ${m}×${n} matrix`);
  }
  return { type: 'map', M: M.map(r => r.slice()), info: mapInfo(M) };
}

// A p for a 3-entry p (only the first n entries are used), padded back to 3 entries.
export const applyMap = (M, p) => pad3(M.map(row => row.reduce((s, a, k) => s + a * p[k], 0)));

function itemPoints(it) {
  const o = it.o || [0, 0, 0];
  switch (it.kind) {
    case 'vec': case 'point': return [o, it.v];
    case 'span': return [o, ...it.vecs];
    case 'parallelogram': return [o, it.u, it.v];
    case 'parallelepiped': return [o, it.u, it.v, it.w];
  }
  return [];
}
export const fitsDomain = (it, n) => n === 3 || itemPoints(it).every(p => Math.abs(p[2] ?? 0) < 1e-9);

// Image of one scene item under A (null for kinds that aren't mapped).
export function mapItem(it, M, matTex = 'A') {
  const f = p => applyMap(M, p), o = it.o || [0, 0, 0];
  const base = { color: it.color, index: it.index, rowId: it.rowId, o: f(o) };
  switch (it.kind) {
    case 'vec': {
      const v = f(it.v);
      return { ...base, kind: 'dual-vec', v, mat: matTex, label: it.label || null, extentPoints: [v] };
    }
    case 'point': {
      const v = f(add3(o, it.v));
      return { ...base, kind: 'dual-pt', o: [0, 0, 0], v, mat: matTex, label: it.label || null, extentPoints: [v] };
    }
    case 'span': return { ...base, kind: 'span', vecs: it.vecs.map(f) };
    case 'parallelogram': return { ...base, kind: 'parallelogram', u: f(it.u), v: f(it.v) };
    case 'parallelepiped': return { ...base, kind: 'parallelepiped', u: f(it.u), v: f(it.v), w: f(it.w) };
  }
  return null;
}

export const niceStep = E => (E <= 8 ? 1 : E <= 16 ? 2 : E <= 40 ? 5 : E <= 80 ? 10 : E <= 200 ? 25 : 100);

// The domain's integer grid as segments: R^2 -> lines x, y = k step with |k step| <= R (R = E is the
// drawn floor grid); R^3 -> a 5×5×5 lattice. `axes` holds the lines through the origin.
export function gridSegments(n, E, R = E) {
  const step = niceStep(E), segs = [], axes = [];
  if (n === 2) {
    const K = Math.floor(R / step + 1e-9);
    for (let k = -K; k <= K; k++) {
      const a = k * step;
      (k ? segs : axes).push([[a, -R, 0], [a, R, 0]], [[-R, a, 0], [R, a, 0]]);
    }
  } else {
    const L = 2 * step;
    for (let a = -2; a <= 2; a++) {
      for (let b = -2; b <= 2; b++) {
        const p = a * step, q = b * step;
        (a || b ? segs : axes).push([[-L, p, q], [L, p, q]], [[p, -L, q], [p, L, q]], [[p, q, -L], [p, q, L]]);
      }
    }
  }
  return { segs, axes };
}

// How far out the R^2 grid must go for its image to fill the codomain box |y| <= Ec: points with
// |x| > Ec sqrt(m) / sigma_min land outside it. Rank-deficient maps keep the drawn grid (E).
export function gridRange(M, E, Ec) {
  if (M[0].length !== 2) return E;
  let a = 0, b = 0, c = 0;
  for (const [x, y] of M) { a += x * x; b += x * y; c += y * y; }
  const smin = Math.sqrt(Math.max(0, (a + c) / 2 - Math.hypot((a - c) / 2, b)));
  if (!(smin > 1e-9 * Math.sqrt(a + c))) return E;
  return Math.max(E, Math.min(40 * niceStep(E), (Ec * Math.sqrt(M.length)) / smin));
}

export function mapSegments(M, list) {
  const out = [];
  for (const [p, q] of list) {
    const a = applyMap(M, p), b = applyMap(M, q);
    if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > 1e-9) out.push([a, b]);
  }
  return out;
}

// Clip segment pq to the cube |x|, |y|, |z| <= S (Liang-Barsky). Returns [p', q'] or null.
export function clipSegment(p, q, S) {
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 3; i++) {
    const d = q[i] - p[i];
    if (Math.abs(d) < 1e-12) { if (Math.abs(p[i]) > S) return null; continue; }
    let a = (-S - p[i]) / d, b = (S - p[i]) / d;
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    if (t0 >= t1) return null;
  }
  const at = t => p.map((x, i) => x + t * (q[i] - p[i]));
  return [at(t0), at(t1)];
}

// Largest coordinate reached by the mapped items (for the codomain extent).
export function needExtent(items) {
  let M = 0;
  const see = p => { for (const x of p) if (Number.isFinite(x)) M = Math.max(M, Math.abs(x)); };
  for (const it of items) {
    const o = it.o || [0, 0, 0];
    const tips = it.kind === 'dual-vec' || it.kind === 'dual-pt' ? [it.v]
      : it.kind === 'parallelogram' ? [it.u, it.v, add3(it.u, it.v)]
      : it.kind === 'parallelepiped' ? [it.u, it.v, it.w, add3(it.u, it.v), add3(it.v, it.w), add3(it.u, it.w), add3(add3(it.u, it.v), it.w)]
      : it.kind === 'span' ? it.vecs : [];
    see(o);
    for (const t of tips) see(add3(o, t));
  }
  return M;
}

// Split the scene items of a map(A) row into the domain list (main scene) and the codomain list.
// `from` is the map item (its row id tags the extras); `skipped` names rows that don't live in R^n.
export function dualItems(items, info, M, { E = 6, matTex = 'A', from = {} } = {}) {
  const tag = { index: from.index, rowId: from.rowId };
  const keep = [], images = [], skipped = [];
  for (const it of items) {
    if (it.kind === 'map') continue;
    if (!MAPPABLE.has(it.kind)) { keep.push(it); continue; }
    if (!fitsDomain(it, info.n)) { skipped.push(it.label || null); continue; }
    keep.push(it);
    const img = mapItem(it, M, matTex);
    if (img) images.push(img);
  }
  const domain = [...keep, {
    ...tag, kind: 'dual-sub', basis: info.kernel, amb: info.n, whole: true, color: KER_COLOR, opacity: 0.2, latex: `\\ker ${matTex}`,
    avoid: labelTips(keep),
  }];
  if (info.n === 3) domain.unshift({ ...tag, kind: 'dual-grid', n: 3, E, M: null, noAxes: true, color: null, opacity: 0.2 });
  const codomain = [
    { ...tag, kind: 'dual-grid', n: info.n, E, M, color: IMG_COLOR, opacity: 0.3, axisOpacity: 0.7 },
    { ...tag, kind: 'dual-sub', basis: info.image, amb: info.m, whole: false, color: IMG_COLOR, opacity: 0.12, latex: `\\operatorname{im} ${matTex}`, avoid: labelTips(images) },
    ...images,
  ];
  return { domain, codomain, skipped };
}

// Tips of the vectors and points in a list (where their labels go), for the kernel / image label
// to keep clear of.
const TIPPED = new Set(['vec', 'point', 'dual-vec', 'dual-pt']);
export const labelTips = list => list.filter(it => TIPPED.has(it.kind)).map(it => add3(it.o || [0, 0, 0], it.v));

// Which camera moved since the last sync ('a', 'b' or null); directions are unit 3-arrays.
export function linkSource(a, b, last, eps = 1e-6) {
  if (!last) return 'a';
  const da = Math.hypot(a[0] - last[0], a[1] - last[1], a[2] - last[2]);
  const db = Math.hypot(b[0] - last[0], b[1] - last[1], b[2] - last[2]);
  if (Math.max(da, db) < eps) return null;
  return da >= db ? 'a' : 'b';
}

// Name of the matrix in `map(<name>)`, else T.
export function matrixName(src) {
  const st = parseLine(src);
  const arg = st?.body?.t === 'call' && st.body.name === 'map' ? st.body.args[0] : null;
  return arg?.t === 'name' ? arg.name : 'T';
}

const RR = k => `\\mathbb{R}^{${k}}`;
export function subspaceTex(dim, amb) {
  if (dim === 0) return '= \\{\\vec{0}\\}';
  if (dim === amb) return `= ${RR(amb)}`;
  return dim === 1 ? '\\text{ is a line}' : '\\text{ is a plane}';
}

export function readout(info, A = 'A') {
  const { m, n, rank, nullity } = info, sq = m === n;
  return {
    domain: `\\text{${sq ? 'before: ' : ''}domain } ${RR(n)}`,
    codomain: `\\text{${sq ? 'after: ' : ''}codomain } ${RR(m)}`,
    ker: `\\ker ${A} ${subspaceTex(nullity, n)}\\quad \\text{nullity } ${nullity}`,
    im: `\\operatorname{im} ${A} ${subspaceTex(rank, m)}\\quad \\text{rank } ${rank}`,
    check: `\\text{rank} + \\text{nullity} = ${rank} + ${nullity} = ${rank + nullity} = \\dim ${RR(n)}\\ \\checkmark`,
    row: `${A}: ${RR(n)} \\to ${RR(m)},\\quad \\text{rank } ${rank} + \\text{nullity } ${nullity} = ${rank + nullity} = n\\ \\checkmark`,
  };
}

// ================================================================ language
registerFunction('map', { n: 1, kind: 'mat', f: ([A]) => mapValue(A.m) });
registerType('map', {
  describe: 'a linear map view',
  format: v => `R^${v.info.n} -> R^${v.info.m}: rank ${v.info.rank}, nullity ${v.info.nullity}`,
  latex: v => readout(v.info).row,
  numbers: v => v.M.flat(),
});

// ================================================================ renderers
function installRenderers(reg, THREE) {
  reg('dual-vec', (it, c) => {
    const o = c.v3(it.o), v = c.v3(it.v), len = v.length();
    c.arrow(o, v, it.color);
    if (!it.label) return;
    const tip = o.clone().add(v);
    if (len > 1e-9) tip.addScaledVector(v, (0.3 * c.s) / len);
    else tip.add(new THREE.Vector3(0.3, 0.3, 0.2).multiplyScalar(c.s));
    c.label(tip, `${it.mat}${c.nameTex(it.label, true)}`, it.color);
  });
  reg('dual-pt', (it, c) => {
    const p = c.v3(it.v);
    c.dot(p, it.color, 0.08 * c.s);
    if (it.label) c.label(p.clone().add(new THREE.Vector3(0.22, 0.22, 0.3).multiplyScalar(c.s)), `${it.mat}\\,${c.nameTex(it.label, false)}`, it.color);
  });
  // Subspace through the origin: point, line or plane (3 dims fill space: nothing to draw). A plane
  // that is the whole ambient R^2 is only shaded when `whole` is set.
  reg('dual-sub', (it, c) => {
    const { basis } = c.orthoBasis(it.basis), o = new THREE.Vector3(), E = c.E, s = c.s;
    // Label spot: of the candidate points, the one nearest the middle of the screen that keeps
    // clear of the vector and point labels (a vector in the kernel ends right on the kernel line).
    const cam = c.camera, aspect = cam.isPerspectiveCamera ? cam.aspect : (cam.right - cam.left) / (cam.top - cam.bottom);
    const busy = (it.avoid ?? []).map(p => c.v3(p).project(cam));
    const pick = (cands, extra) => {
      let best = cands[0], bd = Infinity;
      for (const [k, p] of [...cands, ...extra].entries()) {
        const q = p.clone().project(cam), d = Math.max(Math.abs(q.x), Math.abs(q.y));
        const crowded = busy.some(r => Math.abs(q.x - r.x) * aspect < 0.2 && Math.abs(q.y - r.y) < 0.09);
        const score = d + (crowded ? 10 : 0) + (k >= cands.length ? 5 : 0);
        if (Number.isFinite(score) && score < bd) { bd = score; best = p; }
      }
      return best;
    };
    let at = null;
    if (basis.length === 0) c.dot(o, it.color, 0.11 * s, 0.85);
    else if (basis.length === 1) {
      const b = basis[0], L = E * 1.4, m = new THREE.Mesh(c.GEO.cyl, c.mat('ghost', it.color, 0.6));
      c.placeAlong(m, b.clone().multiplyScalar(-L), b, 2 * L, 0.03 * s);
      c.add(m);
      const perp = Math.abs(b.z) < 0.9 ? b.clone().cross(new THREE.Vector3(0, 0, 1)).normalize() : new THREE.Vector3(1, 0, 0);
      const on = k => b.clone().multiplyScalar(k * E).addScaledVector(perp, -0.4 * s).add(new THREE.Vector3(0, 0, 0.25 * s));
      at = pick([0.7, -0.7].map(on), [0.45, -0.45, 0.9, -0.9].map(on));
    } else if (basis.length === 2 && (it.whole || it.amb > 2)) {
      c.planePatch(o, basis[0], basis[1], it.color, null, { opacity: it.opacity, size: E });
      const corners = f => [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([x, y]) => basis[0].clone().multiplyScalar(x).addScaledVector(basis[1], y).multiplyScalar(E * f));
      at = pick(corners(0.55), corners(0.35));
    }
    if (at && it.latex) c.label(at, it.latex, it.color);
  });
  // The domain's grid (mapped by it.M when set), clipped to this scene's box. color null = axis colour.
  reg('dual-grid', (it, c) => {
    const R = it.M ? gridRange(it.M, it.E, c.E) : it.E;
    let { segs, axes } = gridSegments(it.n, it.E, R);
    if (it.M) { segs = mapSegments(it.M, segs); axes = mapSegments(it.M, axes); }
    if (it.noAxes) axes = [];
    const lists = [[], []];
    [segs, axes].forEach((list, k) => {
      for (const [p, q] of list) {
        const r = clipSegment(p, q, c.E);
        if (r) lists[k].push(c.v3(r[0]), c.v3(r[1]));
      }
    });
    const color = it.color ?? c.colors.axis;
    if (lists[0].length) c.lines(lists[0], color, { opacity: it.opacity });
    if (lists[1].length) c.lines(lists[1], color, { opacity: it.axisOpacity ?? it.opacity });
  });
}

// createScene() re-registers the built-in span / plane / parallelogram renderers bound to the scene
// it creates, so after making the codomain scene put back versions that only use ctx.
function restoreBuiltins(reg, THREE) {
  reg('span', (it, c) => {
    const o = c.v3(it.o), { basis, picked } = c.orthoBasis(it.vecs);
    if (basis.length === 0) c.dot(o, it.color);
    else if (basis.length === 1) {
      const L = c.E * 1.4, m = new THREE.Mesh(c.GEO.cyl, c.mat('ghost', it.color));
      c.placeAlong(m, o.clone().addScaledVector(basis[0], -L), basis[0], 2 * L, 0.014 * c.s);
      c.add(m);
    } else if (basis.length === 2) c.planePatch(o, basis[0], basis[1], it.color, picked);
  });
  reg('plane', (it, c) => {
    const n = c.v3(it.normal);
    if (n.lengthSq() < 1e-18) return;
    n.normalize();
    const e1 = n.clone().cross(Math.abs(n.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize();
    c.planePatch(c.v3(it.o), e1, n.clone().cross(e1), it.color, null);
  });
  reg('parallelogram', (it, c) => {
    const o = c.v3(it.o), u = c.v3(it.u), v = c.v3(it.v);
    const A = o, B = o.clone().add(u), C = B.clone().add(v), D = o.clone().add(v);
    const geo = pts => c.own(new THREE.BufferGeometry().setFromPoints(pts));
    c.add(new THREE.Mesh(geo([A, B, C, A, C, D]), c.mat('surface', it.color)));
    c.add(new THREE.LineLoop(geo([A, B, C, D]), c.mat('edge', it.color)));
  });
}

// ================================================================ split view
const CSS = `
#graph.dual-on > #g-view, #graph.dual-on > .dual-pane { grid-row: 1; grid-column: -2 / -1; }
#graph.dual-on > #g-view { margin-right: 50%; }
.dual-pane {
  position: relative; min-width: 0; min-height: 0; overflow: hidden;
  margin-left: 50%; border-left: 1px solid var(--line-2);
}
.dual-pane canvas { display: block; }
.dual-head {
  position: absolute; z-index: 900; top: 12px; left: 50%; transform: translateX(-50%);
  max-width: calc(100% - 24px); overflow: hidden; pointer-events: none;
  padding: 6px 14px 7px; border-radius: var(--r-lg); text-align: center; white-space: nowrap;
  background: var(--float); color: var(--text-1); border: 1px solid var(--float-line);
  box-shadow: var(--shadow-2); font-size: var(--fs-md); line-height: 1.55;
}
.dual-head[hidden] { display: none; }
.dual-head .dual-title { font-size: var(--fs-xl); }
.dual-head .dual-note { color: var(--text-3); font-size: var(--fs-sm); }
`;
const HELP = '<h4 class="ui-overline">Split view</h4>' +
  '<p><code>map(A)</code> split view: domain | codomain of a 2×2 to 3×3 matrix, with its kernel (red) and image</p>';

export async function install(api) {
  const S = await import('../scene.js');
  installRenderers(S.registerRenderer, S.THREE);
  api.addStyles(CSS);
  document.getElementById('g-help')?.insertAdjacentHTML('beforeend', HELP);
  const view = splitView(api, S);
  api.addItemsHook(view.items);
  api.addRowDecorator((row, res, el) => {
    if (res?.error || res?.value?.type !== 'map') return;
    el.out.innerHTML = S.tex(readout(res.value.info, S.nameTex(matrixName(row.src), false)).row);
  });
  api.onViewChange(view.onView);
  new MutationObserver(view.onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  document.getElementById('g-fit')?.addEventListener('click', view.fit);
  document.getElementById('g-reset')?.addEventListener('click', view.reset);
}

function splitView(api, S) {
  const { THREE } = S, ISO = S.VIEWS.iso, TOP = S.VIEWS.top;
  const graphEl = document.getElementById('graph');
  const theme = () => document.documentElement.dataset.theme || 'dark';
  const makeHead = () => { const el = document.createElement('div'); el.className = 'dual-head'; return el; };
  const mainHead = makeHead();
  mainHead.hidden = true;
  api.addOverlay(mainHead);

  let side = null, pane = null, sideHead = null, unsubs = [];
  // flat: we switched the main scene to 2D; moved: we backed its camera off for the narrow pane.
  let dims = null, saved = null, flat = false, moved = false, lastE = 0, forceFit = false, lastDir = null;
  let link = true;
  try { link = JSON.parse(localStorage.getItem(STORE))?.link !== false; } catch { /* default on */ }
  const btn = api.addToolbarButton({
    label: 'Link cameras',
    title: 'Rotate the domain and codomain views together',
    group: 'camera',
    onClick: () => {
      link = !link;
      lastDir = null;
      try { localStorage.setItem(STORE, JSON.stringify({ link })); } catch { /* private mode */ }
      paintButton();
    },
  });
  paintButton();

  const linked = () => link && side && dims?.m === 3 && dims?.n === 3;

  function paintButton() { // a toolbar group with nothing shown folds away (style.css section 6)
    btn.hidden = !(side && dims?.m === 3 && dims?.n === 3);
    btn.classList.toggle('on', link);
    if (btn.nextElementSibling) btn.parentElement.appendChild(btn); // after the view presets, Ortho and 2D
  }

  // Camera direction (unit, target -> camera) as an array, and setting it at the same distance.
  const dirOf = sc => sc.camera.position.clone().sub(sc.controls.target).normalize().toArray();
  function setDir(sc, d) {
    const t = sc.controls.target, dist = sc.camera.position.distanceTo(t);
    sc.camera.position.copy(t).addScaledVector(new THREE.Vector3(...d), dist);
  }
  // Runs in both scenes' frame hooks (before their controls update). Linked cameras: whichever
  // turned since the last sync leads.
  function sync() {
    // Keep both panes sharp when the device pixel ratio changes (browser zoom, other monitor).
    const dpr = window.devicePixelRatio || 1;
    for (const sc of [api.scene, side]) if (sc.renderer.getPixelRatio() !== dpr) sc.renderer.setPixelRatio(dpr);
    if (!linked()) { lastDir = null; return; }
    const main = api.scene, a = dirOf(main), b = dirOf(side);
    const src = linkSource(a, b, lastDir);
    if (src === 'a') { setDir(side, a); lastDir = a; }
    else if (src === 'b') { setDir(main, b); lastDir = b; }
  }

  // Camera distance that shows about +-E across a pane of this shape (viewPreset uses 3.4 E, which
  // crops the sides of a tall half-width pane). k ~ how wide the view is relative to its height.
  const paneDist = (sc, k) => {
    const w = sc.container.clientWidth, h = sc.container.clientHeight;
    return sc.extent * 3.4 * Math.max(1, w && h ? (k * h) / w : 1);
  };
  const along = (dir, d) => new THREE.Vector3(...dir).normalize().multiplyScalar(d).toArray();
  const distOf = sc => sc.camera.position.distanceTo(sc.controls.target);
  // Ortho top view framed for the extent and pane (scene.fit() doesn't rescale ortho cameras).
  function frameFlat(sc) {
    sc.setOrtho(false);
    sc.flyTo({ position: along(TOP, paneDist(sc, 0.9)), target: [0, 0, 0], zoom: 1 }, 0);
    sc.setOrtho(true);
  }
  // Perspective main view at distance d along dir from target (default: keep the current one).
  function aim(sc, d, ms, dir = dirOf(sc), target = sc.controls.target.toArray()) {
    const p = along(dir, d).map((x, i) => x + target[i]);
    sc.flyTo({ position: p, target, zoom: sc.camera.zoom }, ms);
  }
  // Back the main camera off so the half-width pane isn't cropped (grow: never move it closer).
  function frameMain(main, ms, dir, grow = false) {
    if (main.ortho) return;
    const d = paneDist(main, 0.85);
    if (grow && d <= distOf(main) * 1.02) return;
    aim(main, d, ms, dir, grow ? undefined : [0, 0, 0]);
    moved = true;
  }

  function create(main) {
    saved = { pose: main.getPose(), is2D: main.is2D };
    pane = document.createElement('div');
    pane.className = 'dual-pane';
    main.container.after(pane);
    graphEl.classList.add('dual-on');
    side = S.createScene(pane);
    restoreBuiltins(S.registerRenderer, THREE);
    side.setTheme(theme());
    sideHead = makeHead();
    pane.appendChild(sideHead);
    unsubs = [main.onFrame(sync), side.onFrame(sync)];
    if (api.view === 'graph') side.start();
    mainHead.hidden = false;
    dims = { m: 0, n: 0 };
    lastDir = null;
  }

  function close() {
    if (!side) return;
    for (const u of unsubs) u();
    unsubs = [];
    side.stop();
    side.setContent([]);
    side.controls.dispose();
    side.renderer.dispose();
    side.renderer.forceContextLoss();
    pane.remove();
    side = pane = sideHead = null;
    graphEl.classList.remove('dual-on');
    mainHead.hidden = true;
    mainHead.dataset.key = '';
    // Put the main camera back: the saved pose after a 2D domain, else just the old distance.
    const main = api.scene, { pose } = saved;
    if (flat) {
      main.set2D(saved.is2D);
      if (!saved.is2D) main.setPose({ ...pose, extent: undefined }, 500);
    } else if (moved && !main.ortho) {
      aim(main, Math.hypot(...pose.position.map((x, i) => x - pose.target[i])), 400);
    }
    flat = moved = false;
    saved = dims = lastDir = null;
    paintButton();
  }

  function setDomain(main, n, first) {
    if (n === 2) {
      if (!main.is2D) { main.set2D(true); flat = true; }
      frameFlat(main);
    } else if (flat) {
      main.set2D(false);
      main.setOrtho(!!saved.pose.ortho);
      flat = false;
      frameMain(main, 500, ISO);
    } else if (first) frameMain(main, 400, undefined, true);
  }

  function frameSide(m, reset) {
    if (m === 2) {
      if (!side.is2D) side.set2D(true);
      frameFlat(side);
      return;
    }
    const was2D = side.is2D;
    if (was2D) { side.set2D(false); side.setOrtho(false); }
    const snap = reset || was2D;
    aim(side, paneDist(side, 0.85), snap || linked() ? 0 : 400, snap ? ISO : undefined, [0, 0, 0]);
    lastDir = null;
  }

  function paint(el, parts) {
    const key = JSON.stringify(parts);
    if (el.dataset.key === key) return;
    el.dataset.key = key;
    el.innerHTML = parts.map(([cls, latex, color, text]) =>
      `<div class="${cls}"${color ? ` style="color:${color}"` : ''}>${latex ? S.tex(latex) : text}</div>`).join('');
  }

  function paintHeads(info, A, skipped) {
    const r = readout(info, A);
    const named = skipped.filter(Boolean), anon = skipped.length - named.length;
    const who = [...named, ...(anon ? [`${anon} unnamed row${anon > 1 ? 's' : ''}`] : [])].join(', ');
    paint(mainHead, [
      ['dual-title', r.domain], ['dual-sub', r.ker, KER_COLOR],
      ...(who ? [['dual-note', null, null, `hidden (not in ℝ²): ${who}`]] : []),
    ]);
    paint(sideHead, [['dual-title', r.codomain], ['dual-sub', r.im, IMG_COLOR], ['dual-note', r.check]]);
  }

  function items(list, { rows, results }) {
    const main = api.scene;
    const mapIt = main && list.find(it => it.kind === 'map');
    if (!mapIt) {
      // Keep the split while a map(...) row is being retyped (error), so it doesn't flicker.
      const typing = rows.some((r, i) => !r.hidden && results[i]?.error && /\bmap\s*\(/.test(r.src));
      if (!typing) close();
      return list;
    }
    const { info, M } = mapIt;
    const A = S.nameTex(matrixName(rows[mapIt.index]?.src ?? ''), false);
    const first = !side;
    if (first) create(main);
    const eChanged = main.extent !== lastE, mChanged = info.m !== dims.m, nChanged = info.n !== dims.n;
    const fitting = forceFit, refit = first || fitting || eChanged || mChanged;
    forceFit = false;
    dims = { m: info.m, n: info.n };
    if (nChanged) setDomain(main, info.n, first);
    else if (fitting) main.is2D ? frameFlat(main) : frameMain(main, 500);
    const { domain, codomain, skipped } = dualItems(list, info, M, { E: main.extent, matTex: A, from: mapIt });
    if (refit) {
      side.setExtent(Math.max(main.extent, needExtent(codomain) * 1.15));
      frameSide(info.m, first || mChanged);
    }
    lastE = main.extent;
    side.setContent(codomain);
    paintHeads(info, A, skipped);
    paintButton();
    return domain;
  }

  return {
    items,
    onView: v => { if (side) v === 'graph' ? side.start() : side.stop(); },
    onTheme: () => side?.setTheme(theme()),
    fit: () => { if (side) { forceFit = true; api.recompute(); } },
    // "Reset view" flies the main camera to the 3D iso view; a flat R^2 domain has to stay a top
    // view (its rotation is locked), and both panes go back to their framing for the half-width pane.
    reset: () => {
      if (!side || !dims) return;
      const main = api.scene;
      if (main.is2D) frameFlat(main);
      else frameMain(main, 600, ISO);
      frameSide(dims.m, true);
    },
  };
}
