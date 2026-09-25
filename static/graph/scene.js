// Three.js view for the 3D tab: z-up axes + grid, and drawable items supplied by grapher.js.
//
// Extension points (used by graph/features/*):
//   registerRenderer(kind, (item, ctx) => void)   draw items of a new kind (see makeCtx for ctx)
//   scene.onFrame(fn(dt, time))                     persistent per-frame hook, returns unsubscribe
//   ctx.onFrame(fn(dt, time))                       per-frame hook that lives until the next rebuild
//   scene.getPose / setPose / flyTo / viewPreset / setOrtho / set2D / setAutoRotate
//   scene.pick(clientX, clientY)                    raycast -> { item, object, point }
//   scene.snapshot()                                PNG data URL of the WebGL canvas (no labels)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export { THREE };

export const THEMES = {
  dark: { bg: 0x1d2327, grid: 0x323b41, gridMajor: 0x46525a, axis: 0x9aa3a8, text: '#9aa3a8', fg: '#ecebe4' },
  light: { bg: 0xfbfbf8, grid: 0xe3e5e8, gridMajor: 0xc7cbcf, axis: 0x5b6166, text: '#5b6166', fg: '#1d1d1f' },
};
const NICE = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000];
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const FOV = 40;
export const VIEWS = { // camera direction from the target
  iso: [10, 5, 6], top: [0, -1e-3, 1], front: [1, 0, 0], side: [0, -1, 0],
};

export const v3 = a => new THREE.Vector3(a[0], a[1], a[2] ?? 0);

// Clip the 2D line p + t d to the square |x|,|y| <= S (Liang-Barsky). Returns [t0, t1] or null.
export function clipToSquare(p, d, S) {
  let t0 = -Infinity, t1 = Infinity;
  for (let i = 0; i < 2; i++) {
    if (Math.abs(d[i]) < 1e-12) { if (Math.abs(p[i]) > S) return null; continue; }
    let a = (-S - p[i]) / d[i], b = (S - p[i]) / d[i];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
  }
  return t0 < t1 ? [t0, t1] : null;
}

// Orthonormal basis of span(vecs) by Gram-Schmidt; `picked` = the inputs that were independent.
export function orthoBasis(vecs) {
  const basis = [], picked = [];
  for (const a of vecs) {
    const w = v3(a), scale = Math.max(1, w.length());
    for (const q of basis) w.addScaledVector(q, -w.dot(q));
    const n = w.length();
    if (n > 1e-9 * scale) { basis.push(w.divideScalar(n)); picked.push(v3(a)); }
  }
  return { basis, picked };
}

const RENDERERS = new Map();
export function registerRenderer(kind, fn) { RENDERERS.set(kind, fn); }

const GEO = {
  cyl: new THREE.CylinderGeometry(1, 1, 1, 18),
  cone: new THREE.ConeGeometry(1, 1, 24),
  sphere: new THREE.SphereGeometry(1, 24, 16),
  box: new THREE.BoxGeometry(1, 1, 1).translate(0.5, 0.5, 0.5),
};
GEO.boxEdges = new THREE.EdgesGeometry(GEO.box);
export { GEO };

const materials = new Map();
// kinds: solid | glass (solid, transparent) | ghost (unlit, transparent) | surface (unlit, double-sided)
//        line | edge   (dashed lines: use ctx.lines/polyline with { dashed: true })
export function mat(kind, color, opacity) {
  const key = `${kind}|${color}|${opacity ?? ''}`;
  if (!materials.has(key)) {
    const c = new THREE.Color(color);
    const make = {
      solid: () => new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, metalness: 0.05 }),
      glass: () => new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, transparent: true, opacity: opacity ?? 0.35, depthWrite: false }),
      ghost: () => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: opacity ?? 0.5, depthWrite: false }),
      surface: () => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: opacity ?? 0.16, side: THREE.DoubleSide, depthWrite: false }),
      line: () => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: opacity ?? 0.45, depthWrite: false }),
      edge: () => new THREE.LineBasicMaterial({ color: c, transparent: opacity != null, opacity: opacity ?? 1 }),
    }[kind];
    if (!make) throw new Error(`unknown material kind ${kind}`);
    materials.set(key, make());
  }
  return materials.get(key);
}

const texCache = new Map();
export function tex(src) {
  if (!texCache.has(src)) texCache.set(src, katex.renderToString(src, { throwOnError: false }));
  return texCache.get(src);
}

// LaTeX for a user name: v1 -> v_{1}, v_old -> v_{old}, long names upright; vector adds an arrow.
export function nameTex(name, vector) {
  const m = name.match(/^(.*?)(?:_(\w+)|(\d+))$/);
  let base = m && m[1] ? m[1] : name;
  const sub = m && m[1] ? m[2] || m[3] : '';
  if (base.length > 1) base = `\\mathrm{${base}}`;
  const b = vector ? `\\vec{${base}}` : base;
  return sub ? `${b}_{${sub}}` : b;
}

// ------------------------------------------------------------------ built-in item kinds
// Registered once and written only against ctx, so several scenes can coexist.
registerRenderer('vec', (it, c) => {
  const o = v3(it.o), v = v3(it.v);
  c.arrow(o, v, it.color, it.style);
  if (it.label) {
    const len = v.length(), tip = o.clone().add(v);
    if (len > 1e-9) tip.addScaledVector(v, (0.3 * c.s) / len);
    c.label(tip, nameTex(it.label, true), it.color);
  }
});
registerRenderer('point', (it, c) => {
  const p = v3(it.o).add(v3(it.v));
  c.dot(p, it.color, 0.08 * c.s);
  if (it.label) c.label(p.clone().add(new THREE.Vector3(0, 0, 0.3 * c.s)), nameTex(it.label, false), it.color);
});
registerRenderer('span', (it, c) => {
  const o = v3(it.o), { basis, picked } = orthoBasis(it.vecs);
  if (basis.length === 0) c.dot(o, it.color);
  else if (basis.length === 1) {
    const L = c.E * 1.4, m = new THREE.Mesh(GEO.cyl, mat('ghost', it.color));
    c.placeAlong(m, o.clone().addScaledVector(basis[0], -L), basis[0], 2 * L, 0.014 * c.s);
    c.add(m);
  } else if (basis.length === 2) c.planePatch(o, basis[0], basis[1], it.color, picked);
  // rank 3 is all of R^3: nothing sensible to draw
});
registerRenderer('plane', (it, c) => {
  const n = v3(it.normal);
  if (n.lengthSq() < 1e-18) return;
  n.normalize();
  const e1 = n.clone().cross(Math.abs(n.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize();
  c.planePatch(v3(it.o), e1, n.clone().cross(e1), it.color, null);
});
registerRenderer('parallelogram', (it, c) => {
  const o = v3(it.o), u = v3(it.u), v = v3(it.v);
  const A = o, B = o.clone().add(u), C = B.clone().add(v), D = o.clone().add(v);
  c.add(new THREE.Mesh(c.geometry([A, B, C, A, C, D]), mat('surface', it.color)));
  c.add(new THREE.LineLoop(c.geometry([A, B, C, D]), mat('edge', it.color)));
});
registerRenderer('parallelepiped', (it, c) => {
  const M = new THREE.Matrix4().makeBasis(v3(it.u), v3(it.v), v3(it.w)).setPosition(v3(it.o));
  for (const obj of [new THREE.Mesh(GEO.box, mat('surface', it.color)), new THREE.LineSegments(GEO.boxEdges, mat('edge', it.color))]) {
    obj.matrixAutoUpdate = false; // basis may be skewed, so set the matrix directly
    obj.matrix.copy(M);
    c.add(obj);
  }
});

// Label text needs more contrast than a thick arrow: on the whiteboard, darken light hues
// (orange, teal, green) until their luminance is <= 0.2, about 4:1 against white.
const readable = new Map();
function readableOnLight(color) {
  if (!readable.has(color)) {
    const c = new THREE.Color(color), hsl = {};
    c.getHSL(hsl);
    const lum = () => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; // Color keeps linear components
    while (lum() > 0.2 && hsl.l > 0.05) { hsl.l -= 0.02; c.setHSL(hsl.h, hsl.s, hsl.l); }
    readable.set(color, `#${c.getHexString()}`);
  }
  return readable.get(color);
}

const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

export function createScene(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = 'g-labels';
  container.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  const persp = new THREE.PerspectiveCamera(FOV, 1, 0.01, 5000);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -5000, 5000);
  persp.up.set(0, 0, 1);
  ortho.up.set(0, 0, 1);
  let camera = persp, orthoHalfH = 5;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.autoRotateSpeed = 0.8;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3a3a, 1.7));
  const headlight = new THREE.DirectionalLight(0xffffff, 1.5);
  headlight.position.set(1, 2, 3);
  camera.add(headlight);
  scene.add(camera);

  const axes = new THREE.Group(), content = new THREE.Group();
  scene.add(axes, content);

  let E = 6;                 // axis half-extent
  let theme = 'dark';
  let items = [];
  let showGrid = true, showZ = true;
  const owned = { axes: [], content: [] }; // per-build geometries to dispose
  const frameHooks = new Set();
  let contentHooks = [];

  function clear(group, list) {
    group.clear(); // CSS2DObjects remove their DOM elements on 'removed'
    for (const g of list) g.dispose();
    list.length = 0;
  }
  function label(group, pos, latex, cls, color) {
    const el = document.createElement('div');
    el.className = cls;
    el.innerHTML = tex(latex);
    el.dataset.latex = latex;
    if (color) el.style.color = theme === 'light' ? readableOnLight(color) : color;
    const obj = new CSS2DObject(el);
    obj.position.copy(pos);
    group.add(obj);
    return obj;
  }
  function lineGeometry(points, list) {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    list.push(g);
    return g;
  }

  // ------------------------------------------------------------------ axes + grid
  function niceStep() { return E <= 1.5 ? 0.25 : E <= 3 ? 0.5 : E <= 8 ? 1 : E <= 16 ? 2 : E <= 40 ? 5 : E <= 80 ? 10 : E <= 200 ? 25 : 100; }

  // The XY floor: one line per tick step (so lines meet the ticks at every extent), shaded by
  // vertex colour from full strength near the middle to almost the page colour at the corners,
  // so the floor has no hard square edge. Built once per theme / extent change.
  function floorGrid(t, step, bg) {
    const n = Math.floor(E / step + 1e-9), K = 16, pts = [], cols = [];
    const minor = new THREE.Color(t.grid), major = new THREE.Color(t.gridMajor), c = new THREE.Color();
    const r0 = 0.8 * E, r1 = Math.SQRT2 * E;
    const shade = (base, x, y) => {
      const k = Math.min(1, Math.max(0, (Math.hypot(x, y) - r0) / (r1 - r0)));
      return c.copy(base).lerp(bg, 0.85 * k * (2 - k)); // ease out: the fade starts gently
    };
    for (let i = -n; i <= n; i++) {
      const a = i * step, base = i === 0 ? major : minor;
      for (let k = 0; k < K; k++) {
        const b0 = -E + (2 * E * k) / K, b1 = -E + (2 * E * (k + 1)) / K;
        for (const [x0, y0, x1, y1] of [[a, b0, a, b1], [b0, a, b1, a]]) {
          pts.push(x0, y0, 0, x1, y1, 0);
          shade(base, x0, y0).toArray(cols, cols.length);
          shade(base, x1, y1).toArray(cols, cols.length);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    const m = new THREE.LineBasicMaterial({ vertexColors: true });
    owned.axes.push(g, m);
    return new THREE.LineSegments(g, m);
  }

  // An axis seen end-on (Front, Side, Top) would pile its ticks and name onto the origin: hide
  // them while it points at the camera. Cheap, and it only touches the labels when that flips.
  let axisLabels = [[], [], []], endOn = [false, false, false];
  const toCamera = new THREE.Vector3();
  function hideEndOnAxes() {
    toCamera.copy(camera.position).sub(controls.target).normalize();
    for (let i = 0; i < 3; i++) {
      const end = Math.abs(toCamera.getComponent(i)) > 0.985;
      if (end === endOn[i]) continue;
      endOn[i] = end;
      for (const o of axisLabels[i]) o.visible = !end;
    }
  }

  function buildAxes() {
    clear(axes, owned.axes);
    axisLabels = [[], [], []];
    endOn = [false, false, false];
    const t = THEMES[theme], step = niceStep(), s = E / 6, bg = new THREE.Color(t.bg);
    scene.background = bg;

    if (showGrid) axes.add(floorGrid(t, step, bg));

    // Positive half-axes end in a small head; in 3D the negative halves are quieter, which says
    // which way each axis points at a glance. Neither is data, so both stay neutral.
    const axisMat = new THREE.MeshBasicMaterial({ color: t.axis });
    const negMat = showZ ? new THREE.MeshBasicMaterial({ color: new THREE.Color(t.axis).lerp(bg, 0.45) }) : axisMat;
    owned.axes.push(axisMat, negMat);
    const L = E * 1.1, r = 0.012 * s, headLen = 0.24 * s, o = new THREE.Vector3();
    const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
    dirs.forEach((d, i) => {
      if (i === 2 && !showZ) return;
      const plus = new THREE.Mesh(GEO.cyl, axisMat), minus = new THREE.Mesh(GEO.cyl, negMat), head = new THREE.Mesh(GEO.cone, axisMat);
      placeAlong(plus, o, d, L - headLen, r);
      placeAlong(minus, o, d.clone().negate(), L, r);
      placeAlong(head, d.clone().multiplyScalar(L - headLen), d, headLen, 0.055 * s);
      axes.add(plus, minus, head);
      axisLabels[i].push(label(axes, d.clone().multiplyScalar(L + 0.35 * s), 'xyz'[i], 'g-axis'));
      for (let n = -Math.floor(E / step) * step; n <= E + 1e-9; n += step) {
        if (Math.abs(n) < 1e-9) continue;
        const pos = d.clone().multiplyScalar(n);
        // x, y: just below the floor; in 2D (looking straight down) beside the axis instead. z is
        // upright on screen, so its labels sit on the axis and CSS moves them right (.g-tz).
        if (i < 2 && showZ) pos.z -= 0.28 * s;
        else if (i === 0) pos.y -= 0.3 * s;
        else if (i === 1) pos.x -= 0.3 * s;
        const cls = `g-tick${i === 2 ? ' g-tz' : ''}${n < 0 && showZ ? ' neg' : ''}`;
        axisLabels[i].push(label(axes, pos, String(+(Math.round(n / step) * step).toFixed(2)), cls));
      }
    });
  }

  // ------------------------------------------------------------------ primitives
  function placeAlong(mesh, from, dir, len, radius) { // unit, Y-aligned, centred geometry
    mesh.scale.set(radius, len, radius);
    mesh.quaternion.setFromUnitVectors(Y_AXIS, dir.lengthSq() > 0 ? dir.clone().normalize() : Y_AXIS);
    mesh.position.copy(from).addScaledVector(dir.clone().normalize(), len / 2);
  }
  function dot(pos, color, r, opacity) {
    const m = new THREE.Mesh(GEO.sphere, opacity != null ? mat('glass', color, opacity) : mat('solid', color));
    m.scale.setScalar(r ?? 0.07 * (E / 6));
    m.position.copy(pos);
    content.add(m);
    return m;
  }
  // opts: { opacity, thickness (multiplier), head (multiplier) }
  function arrow(o, v, color, opts = {}) {
    const s = E / 6, len = v.length(), k = opts.thickness ?? 1, hk = opts.head ?? 1;
    const material = opts.opacity != null ? mat('glass', color, opts.opacity) : mat('solid', color);
    if (len < 1e-9) return [dot(o, color, 0.07 * s * k, opts.opacity)];
    const dir = v.clone().divideScalar(len);
    const headLen = Math.min(0.35 * len, 0.34 * s * hk);
    const headR = Math.min(0.12 * s * hk, headLen * 0.45);
    const shaft = new THREE.Mesh(GEO.cyl, material);
    placeAlong(shaft, o, dir, len - headLen, Math.min(0.035 * s * k, headR * 0.45));
    const head = new THREE.Mesh(GEO.cone, material);
    placeAlong(head, o.clone().addScaledVector(dir, len - headLen), dir, headLen, headR);
    content.add(shaft, head);
    return [shaft, head];
  }
  function dashedMat(color, opacity) { // dash length follows the axis scale, so one per build
    const s = E / 6, m = new THREE.LineDashedMaterial({
      color: new THREE.Color(color), dashSize: 0.14 * s, gapSize: 0.1 * s,
      transparent: true, opacity: opacity ?? 0.8, depthWrite: false,
    });
    owned.content.push(m);
    return m;
  }
  // opts: { opacity, dashed }
  function lines(points, color, opts = {}) { // pairs of points -> segments
    const g = lineGeometry(points, owned.content);
    const obj = new THREE.LineSegments(g, opts.dashed ? dashedMat(color, opts.opacity) : mat('line', color, opts.opacity));
    if (opts.dashed) obj.computeLineDistances();
    content.add(obj);
    return obj;
  }
  function polyline(points, color, opts = {}) {
    const g = lineGeometry(points, owned.content);
    const obj = new THREE.Line(g, opts.dashed ? dashedMat(color, opts.opacity) : mat(opts.opacity != null ? 'line' : 'edge', color, opts.opacity));
    if (opts.dashed) obj.computeLineDistances();
    content.add(obj);
    return obj;
  }
  function planePatch(o, e1, e2, color, lattice, opts = {}) {
    const S = opts.size ?? E * 1.2;
    const g = new THREE.PlaneGeometry(2 * S, 2 * S);
    owned.content.push(g);
    const m = new THREE.Mesh(g, mat('surface', color, opts.opacity));
    m.applyMatrix4(new THREE.Matrix4().makeBasis(e1, e2, e1.clone().cross(e2)).setPosition(o));
    content.add(m);
    if (!lattice) return m;
    // Integer combinations a*u + b*v: two families of lines, clipped to the patch.
    const [u, v] = lattice, to2 = w => [w.dot(e1), w.dot(e2)];
    const area = u.clone().cross(v).length(), pts = [];
    for (const [a, b] of [[u, v], [v, u]]) {
      const gap = area / b.length(); // distance between neighbouring lines of this family
      const K = Math.min(40, Math.ceil((S * Math.SQRT2) / gap));
      for (let k = -K; k <= K; k++) {
        const p = a.clone().multiplyScalar(k), t = clipToSquare(to2(p), to2(b), S);
        if (!t) continue;
        pts.push(o.clone().add(p).addScaledVector(b, t[0]), o.clone().add(p).addScaledVector(b, t[1]));
      }
    }
    lines(pts, color);
    return m;
  }

  function makeCtx() {
    const s = E / 6;
    return {
      THREE, E, s, theme, colors: THEMES[theme], camera, group: content,
      add: (...objs) => content.add(...objs),
      own: x => { owned.content.push(x); return x; }, // geometry/material to dispose on rebuild
      v3, GEO, mat, tex, nameTex, orthoBasis, clipToSquare, placeAlong,
      arrow, dot, lines, polyline, planePatch,
      geometry: points => lineGeometry(points, owned.content), // disposed on rebuild
      label: (pos, latex, color, cls = 'g-label') => label(content, pos, latex, cls, color),
      onFrame: fn => { contentHooks.push(fn); },
    };
  }

  function buildContent() {
    clear(content, owned.content);
    contentHooks = [];
    const ctx = makeCtx();
    for (const it of items) {
      const draw = RENDERERS.get(it.kind);
      if (!draw) continue;
      const before = content.children.length;
      try { draw(it, ctx); } catch (err) { console.error(`[scene] ${it.kind}:`, err); }
      for (let k = before; k < content.children.length; k++) content.children[k].userData.item = it;
    }
  }

  // ------------------------------------------------------------------ cameras
  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    persp.aspect = w / h;
    persp.updateProjectionMatrix();
    const a = w / h;
    Object.assign(ortho, { left: -orthoHalfH * a, right: orthoHalfH * a, top: orthoHalfH, bottom: -orthoHalfH });
    ortho.updateProjectionMatrix();
  }
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(container);

  const halfH = dist => dist * Math.tan((FOV * Math.PI) / 360);
  // Framing that shows about +-1.25E across the pane's narrower direction (half-width panes too).
  const fitDistance = () => (E * 1.25) / Math.tan((FOV * Math.PI) / 360) / Math.min(1, persp.aspect || 1);
  const fitZoom = () => (camera === ortho ? (orthoHalfH * Math.min(1, persp.aspect || 1)) / (E * 1.25) : 1);

  function setOrtho(on) {
    const next = on ? ortho : persp;
    if (next === camera) return;
    const dist = camera.position.distanceTo(controls.target);
    if (on) { orthoHalfH = halfH(dist) / persp.zoom; ortho.zoom = 1; }
    else {
      const want = orthoHalfH / ortho.zoom / Math.tan((FOV * Math.PI) / 360);
      const dir = camera.position.clone().sub(controls.target).normalize();
      next.position.copy(controls.target).addScaledVector(dir, want);
    }
    if (on) next.position.copy(camera.position);
    next.quaternion.copy(camera.quaternion);
    camera.remove(headlight);
    scene.remove(camera);
    next.add(headlight);
    scene.add(next);
    camera = next;
    controls.object = camera;
    resize();
    controls.update();
  }

  let flight = null;
  function flyTo(pose, ms = 900) {
    if (pose.extent && pose.extent !== E) api.setExtent(pose.extent);
    if (pose.ortho != null) setOrtho(!!pose.ortho);
    const target = pose.target ? v3(pose.target) : controls.target.clone();
    const position = v3(pose.position);
    if (!ms) {
      controls.target.copy(target);
      camera.position.copy(position);
      if (pose.zoom) { camera.zoom = pose.zoom; camera.updateProjectionMatrix(); }
      controls.update();
      flight = null;
      return;
    }
    const off0 = camera.position.clone().sub(controls.target), off1 = position.clone().sub(target);
    flight = {
      t: 0, ms, t0: controls.target.clone(), t1: target,
      len0: off0.length(), len1: off1.length(),
      rot: new THREE.Quaternion().setFromUnitVectors(off0.clone().normalize(), off1.clone().normalize()),
      dir0: off0.normalize(), zoom0: camera.zoom, zoom1: pose.zoom ?? camera.zoom,
    };
  }
  function stepFlight(dt) {
    if (!flight) return;
    flight.t = Math.min(1, flight.t + (dt * 1000) / flight.ms);
    const k = easeInOut(flight.t);
    const q = new THREE.Quaternion().slerp(flight.rot, k);
    const dir = flight.dir0.clone().applyQuaternion(q);
    controls.target.lerpVectors(flight.t0, flight.t1, k);
    camera.position.copy(controls.target).addScaledVector(dir, flight.len0 + (flight.len1 - flight.len0) * k);
    camera.zoom = flight.zoom0 + (flight.zoom1 - flight.zoom0) * k;
    camera.updateProjectionMatrix();
    if (flight.t >= 1) flight = null;
  }

  function viewPreset(name, ms = 700) {
    const d = v3(VIEWS[name] || VIEWS.iso).normalize();
    flyTo({ position: d.multiplyScalar(fitDistance()).toArray(), target: [0, 0, 0], zoom: fitZoom() }, ms);
  }

  // ------------------------------------------------------------------ loop
  let running = false, lastT = 0;
  function frame(t) {
    if (!running) return;
    const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
    lastT = t;
    stepFlight(dt);
    for (const fn of frameHooks) fn(dt, t / 1000);
    for (const fn of contentHooks) fn(dt, t / 1000);
    controls.update();
    hideEndOnAxes();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  function extentOf(list) {
    let M = 0;
    const see = a => { for (const x of a) if (Number.isFinite(x)) M = Math.max(M, Math.abs(x)); };
    for (const it of list) {
      const o = it.o || [0, 0, 0];
      see(o);
      const tips = it.kind === 'vec' || it.kind === 'point' ? [it.v]
        : it.kind === 'parallelogram' ? [it.u, it.v, [0, 1, 2].map(i => it.u[i] + it.v[i])]
        : it.kind === 'parallelepiped' ? [[0, 1, 2].map(i => it.u[i] + it.v[i] + it.w[i]), it.u, it.v, it.w]
        : it.kind === 'span' ? it.vecs : (it.extentPoints || []);
      for (const t of tips) see([0, 1, 2].map(i => o[i] + (t[i] ?? 0)));
    }
    return M;
  }

  const raycaster = new THREE.Raycaster();

  const api = {
    THREE, container, get camera() { return camera; }, controls, renderer, scene,
    canvas: renderer.domElement, labelLayer: labelRenderer.domElement,
    get extent() { return E; },
    get theme() { return theme; },
    get items() { return items; },
    // Labels are positioned right away: rebuilds often happen after this frame's render (slider
    // animation runs in its own rAF), and unpositioned new labels would never reach the screen.
    setContent(list) { items = list; buildContent(); labelRenderer.render(scene, camera); },
    rebuild() { buildContent(); labelRenderer.render(scene, camera); },
    setTheme(name) { theme = THEMES[name] ? name : 'dark'; buildAxes(); buildContent(); },
    setExtent(x) { E = NICE.find(n => n >= x) || x; buildAxes(); buildContent(); },
    setAxes({ grid, z } = {}) { if (grid != null) showGrid = grid; if (z != null) showZ = z; buildAxes(); },
    fit() {
      this.setExtent(Math.max(0.5, extentOf(items) * 1.15));
      const dir = camera.position.clone().sub(controls.target).normalize();
      flyTo({ position: dir.multiplyScalar(fitDistance()).toArray(), target: [0, 0, 0], zoom: fitZoom() }, 500);
    },
    reset() { viewPreset(this.is2D ? 'top' : 'iso', 600); },
    viewPreset, flyTo,
    setPose(pose, ms = 900) { flyTo(pose, ms); },
    getPose() {
      return { position: camera.position.toArray(), target: controls.target.toArray(), zoom: camera.zoom, ortho: camera === ortho, extent: E };
    },
    setOrtho, get ortho() { return camera === ortho; },
    set2D(on) {
      controls.enableRotate = !on;
      showZ = !on;
      buildAxes();
      if (on) { setOrtho(true); viewPreset('top', 500); }
    },
    get is2D() { return !controls.enableRotate; },
    setAutoRotate(on) { controls.autoRotate = !!on; },
    get autoRotate() { return controls.autoRotate; },
    onFrame(fn) { frameHooks.add(fn); return () => frameHooks.delete(fn); },
    // First hit under the cursor; `filter(item, object)` skips hits (e.g. see-through planes in front).
    pick(clientX, clientY, filter) {
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      raycaster.params.Line.threshold = 0.06 * (E / 6);
      for (const hit of raycaster.intersectObjects(content.children, true)) {
        let o = hit.object;
        while (o && !o.userData.item) o = o.parent;
        if (o && (!filter || filter(o.userData.item, hit.object))) return { item: o.userData.item, object: hit.object, point: hit.point };
      }
      return null;
    },
    render() { renderer.render(scene, camera); labelRenderer.render(scene, camera); },
    snapshot() { renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); },
    start() { if (!running) { running = true; lastT = 0; resize(); requestAnimationFrame(frame); } },
    dispose() {
      running = false;
      resizeObs.disconnect();
      clear(content, owned.content);
      clear(axes, owned.axes);
      controls.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
    },
    stop() { running = false; },
  };

  buildAxes();
  viewPreset('iso', 0);
  return api;
}
