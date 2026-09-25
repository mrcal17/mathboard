// Graphs of functions: x^2 or y = sin(x) (curves), z = x^2 - y^2 (surfaces), f(x) = ... (drawn and
// callable), c(t) = (cos t, sin t, t/4) (parametric), a bare sigmoid / relu' (the function, or its
// derivative) and softmax (logits squashed onto the probability triangle). lang.js turns such rows
// into {type: 'graph'} and {type: 'softmaxmap'} values; this module samples them over the axes box.
import { registerRenderer, THREE } from '../scene.js';

const AXIS = { x: 0, y: 1, z: 2 };
const CURVE_N = 720, SURF_N = 80, LINE_N = 160;
const CURVE_H = 1.2, SURF_H = 1; // graphs are clipped to |value| <= H * E (surfaces at the axis tips)
const inks = new Map(); // colour -> unlit material: a curve is ink in its row's colour, in any light
const ink = (color) => {
  if (!inks.has(color)) inks.set(color, new THREE.MeshBasicMaterial({ color: new THREE.Color(color) }));
  return inks.get(color);
};

const HELP = `
<h4 class="ui-overline">Functions</h4>
<p><code>x^2</code> or <code>y = sin(x)</code> a curve &middot; <code>x = y^2</code> &middot; <code>z = x^2 - y^2</code> a surface</p>
<p><code>f(x) = x^3 - x</code> defines and draws f; then <code>f(2)</code>, <code>f'(x)</code> or a bare <code>f'</code> for its slope</p>
<p><code>c(t) = (cos(t), sin(t), t/4)</code> a curve through space &middot; <code>a = 2</code> then <code>y = a x^2</code> to animate</p>
<p><code>sigmoid</code> <code>tanh</code> <code>relu</code> <code>leakyrelu</code> <code>gelu</code> <code>softplus</code> <code>silu</code> <code>elu</code>: type one alone to draw it, <code>sigmoid'</code> for its derivative</p>
<p><code>softmax</code> the probability triangle and where the logit grid lands on it &middot; <code>softmax(T)</code> with a temperature &middot; <code>softmax(v)</code> a vector of logits</p>`;

const num = (at, env) => {
  try {
    const v = at(env);
    return typeof v === 'number' && Number.isFinite(v) ? v : NaN;
  } catch { return NaN; }
};
const vecAt = (at, env) => {
  try {
    const v = at(env);
    return v && Array.isArray(v.v) && v.v.every(Number.isFinite) ? v.v : null;
  } catch { return null; }
};
// A transform(A, t) row carries graphs like everything else (it.M, see transform.js).
const place = (it) => {
  const o = it.o || [0, 0, 0], M = it.M;
  return (p) => (M
    ? new THREE.Vector3(...[0, 1, 2].map((i) => o[i] + M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2]))
    : new THREE.Vector3(o[0] + p[0], o[1] + p[1], o[2] + p[2]));
};

// ------------------------------------------------------------------ curves

// y = f(x) sampled over |x| <= E, split where it is undefined or leaves |y| <= HEIGHT E (ending on
// the clip line, so steep curves reach the edge of the box).
export function curvePolylines(at, input, dep, E) {
  const lim = CURVE_H * E, a = AXIS[input], b = AXIS[dep], env = {}, out = [];
  const put = (t, v) => { const p = [0, 0, 0]; p[a] = t; p[b] = v; return p; };
  let cur = null, pt = 0, pv = NaN;
  for (let k = 0; k <= CURVE_N; k++) {
    const t = -E + (2 * E * k) / CURVE_N;
    env[input] = t;
    const v = num(at, env), inside = Math.abs(v) <= lim;
    if (inside) {
      if (cur && Math.abs(v - pv) > lim) cur = null; // a jump: don't draw the riser
      if (!cur) {
        cur = [];
        out.push(cur);
        if (Math.abs(pv) > lim) { const edge = Math.sign(pv) * lim; cur.push(put(pt + ((edge - pv) / (v - pv)) * (t - pt), edge)); }
      }
      cur.push(put(t, v));
    } else {
      if (cur && Number.isFinite(v) && Number.isFinite(pv)) { const edge = Math.sign(v) * lim; cur.push(put(pt + ((edge - pv) / (v - pv)) * (t - pt), edge)); }
      cur = null;
    }
    pt = t;
    pv = v;
  }
  return out.filter((s) => s.length > 1);
}

// c(t) for |t| <= E, split where undefined or far outside the box.
export function paramPolylines(at, input, E) {
  const far = 4 * E, env = {}, out = [];
  let cur = null;
  for (let k = 0; k <= CURVE_N; k++) {
    env[input] = -E + (2 * E * k) / CURVE_N;
    const p = vecAt(at, env);
    if (p && p.every((x) => Math.abs(x) <= far)) {
      if (!cur) { cur = []; out.push(cur); }
      cur.push(p);
    } else cur = null;
  }
  return out.filter((s) => s.length > 1);
}

// A tube round a polyline, with parallel-transported frames (no twist at inflections).
function tubeGeometry(pts, r, radial = 8) {
  const n = pts.length, pos = new Float32Array(n * radial * 3), nor = new Float32Array(n * radial * 3);
  const T = new THREE.Vector3(), prev = new THREE.Vector3(), N = new THREE.Vector3(), B = new THREE.Vector3();
  const axis = new THREE.Vector3(), tmp = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    T.subVectors(pts[Math.min(n - 1, i + 1)], pts[Math.max(0, i - 1)]).normalize();
    if (i === 0) {
      N.set(0, 0, 1);
      if (Math.abs(T.z) > 0.9) N.set(1, 0, 0);
    } else {
      axis.crossVectors(prev, T);
      const s = axis.length();
      if (s > 1e-9) N.applyAxisAngle(axis.divideScalar(s), Math.acos(Math.max(-1, Math.min(1, prev.dot(T)))));
    }
    N.addScaledVector(T, -N.dot(T)).normalize();
    B.crossVectors(T, N);
    prev.copy(T);
    for (let j = 0; j < radial; j++) {
      const th = (2 * Math.PI * j) / radial, k = (i * radial + j) * 3;
      tmp.copy(N).multiplyScalar(Math.cos(th)).addScaledVector(B, Math.sin(th));
      nor[k] = tmp.x; nor[k + 1] = tmp.y; nor[k + 2] = tmp.z;
      pos[k] = pts[i].x + r * tmp.x; pos[k + 1] = pts[i].y + r * tmp.y; pos[k + 2] = pts[i].z + r * tmp.z;
    }
  }
  const idx = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * radial + j, b = i * radial + ((j + 1) % radial), c = a + radial, d = b + radial;
      idx.push(a, c, b, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setIndex(idx);
  return g;
}

function drawCurves(it, c, lines) {
  const at = place(it), r = 0.032 * c.s, material = ink(it.color);
  let end = null;
  for (const line of lines) {
    const pts = line.map(at);
    c.add(new THREE.Mesh(c.own(tubeGeometry(pts, r)), material));
    for (const p of [pts[0], pts.at(-1)]) { // round the ends
      const cap = new THREE.Mesh(c.GEO.sphere, material);
      cap.scale.setScalar(r);
      cap.position.copy(p);
      c.add(cap);
    }
    if (!end || pts.at(-1).x > end.x) end = pts.at(-1);
  }
  // f(x) = ... and a = x^2 get their name; y = ... doesn't (y is the axis, not a name)
  if (end && it.label && !AXIS.hasOwnProperty(it.label)) {
    c.label(end.clone().add(new THREE.Vector3(0.25 * c.s, 0, 0.25 * c.s)), c.nameTex(it.label, false), it.color);
  }
}

// ------------------------------------------------------------------ surfaces

// Heights on an (N+1)^2 grid over the two inputs; NaN where undefined.
export function surfaceGrid(at, ins, E, N = SURF_N) {
  const h = new Float64Array((N + 1) * (N + 1)), env = {}, [ku, kv] = ins;
  for (let i = 0; i <= N; i++) {
    env[ku] = -E + (2 * E * i) / N;
    for (let j = 0; j <= N; j++) {
      env[kv] = -E + (2 * E * j) / N;
      h[i * (N + 1) + j] = num(at, env);
    }
  }
  return h;
}

function drawSurface(it, c) {
  const E = c.E, N = SURF_N, lim = SURF_H * E, [ku, kv] = it.ins, ua = AXIS[ku], va = AXIS[kv], d = AXIS[it.dep];
  const at = place(it), carried = !!it.M, heights = surfaceGrid(it.at, it.ins, E, N), W = N + 1;
  const ok = (h) => Number.isFinite(h) && (!carried || Math.abs(h) <= lim); // carried: no clip planes
  const pos = new Float32Array(W * W * 3), col = new Float32Array(W * W * 3);
  const base = new THREE.Color(it.color), low = base.clone().lerp(new THREE.Color(c.theme === 'light' ? '#1d1d1f' : '#000000'), 0.45);
  const high = base.clone().lerp(new THREE.Color('#ffffff'), c.theme === 'light' ? 0.15 : 0.3), tmp = new THREE.Color();
  const p = [0, 0, 0];
  for (let i = 0; i < W; i++) {
    for (let j = 0; j < W; j++) {
      const k = i * W + j, h = heights[k], hh = Number.isFinite(h) ? Math.max(-8 * lim, Math.min(8 * lim, h)) : 0;
      p[ua] = -E + (2 * E * i) / N; p[va] = -E + (2 * E * j) / N; p[d] = hh;
      const q = at(p);
      pos[3 * k] = q.x; pos[3 * k + 1] = q.y; pos[3 * k + 2] = q.z;
      tmp.copy(low).lerp(high, Math.max(0, Math.min(1, (hh + lim) / (2 * lim)))); // height shading
      col[3 * k] = tmp.r; col[3 * k + 1] = tmp.g; col[3 * k + 2] = tmp.b;
    }
  }
  const idx = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = i * W + j, b = a + 1, e = a + W, f = e + 1;
      if (ok(heights[a]) && ok(heights[b]) && ok(heights[e]) && ok(heights[f])) idx.push(a, e, b, b, e, f);
    }
  }
  if (!idx.length) return;
  const g = c.own(new THREE.BufferGeometry());
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  // clip to the box along the height axis (unless a transform moved the axes)
  const n = new THREE.Vector3(), clip = [];
  if (!carried) {
    n.setComponent(d, 1);
    clip.push(new THREE.Plane(n.clone().negate(), lim), new THREE.Plane(n.clone(), lim));
  }
  const skin = c.own(new THREE.MeshStandardMaterial({
    color: 0xffffff, vertexColors: true, side: THREE.DoubleSide, roughness: 0.8, metalness: 0,
    transparent: true, opacity: 0.9, clippingPlanes: clip,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, // the grid lines sit on top
  }));
  c.add(new THREE.Mesh(g, skin));

  // grid lines on the surface at the axis ticks
  const step = niceStep(E), segs = [], env = {};
  for (const [kf, kl] of [[ku, kv], [kv, ku]]) {
    for (let gv = Math.ceil(-E / step) * step; gv <= E + 1e-9; gv += step) {
      env[kf] = gv;
      let prevP = null;
      for (let s = 0; s <= LINE_N; s++) {
        env[kl] = -E + (2 * E * s) / LINE_N;
        const h = num(it.at, env);
        if (!ok(h) || (carried && Math.abs(h) > lim)) { prevP = null; continue; }
        p[AXIS[kf]] = gv; p[AXIS[kl]] = env[kl]; p[d] = Math.max(-8 * lim, Math.min(8 * lim, h));
        const q = at(p);
        if (prevP) segs.push(prevP, q);
        prevP = q;
      }
    }
  }
  if (segs.length) {
    const lineMat = c.own(new THREE.LineBasicMaterial({
      color: c.theme === 'light' ? 0x1d1d1f : 0xffffff, transparent: true, opacity: c.theme === 'light' ? 0.16 : 0.2,
      depthWrite: false, clippingPlanes: clip,
    }));
    c.add(new THREE.LineSegments(c.geometry(segs), lineMat));
  }
  if (it.label && !AXIS.hasOwnProperty(it.label)) {
    const k = N * W + N, h = heights[k];
    if (ok(h) && Math.abs(h) <= lim) { p[ua] = E; p[va] = E; p[d] = h; c.label(at(p), c.nameTex(it.label, false), it.color); }
  }
}

// The axis tick spacing (scene.js niceStep), so the surface grid meets the ticks.
const niceStep = (E) => (E <= 1.5 ? 0.25 : E <= 3 ? 0.5 : E <= 8 ? 1 : E <= 16 ? 2 : E <= 40 ? 5 : E <= 80 ? 10 : E <= 200 ? 25 : 100);

// ------------------------------------------------------------------ softmax

// softmax((a, b, 0) / T): the logit plane z = 0 covers every probability vector (adding the same
// number to every logit changes nothing), so its grid lines show how softmax squashes logits.
export function softmaxPoint(a, b, T) {
  const m = Math.max(a, b, 0), e = [Math.exp((a - m) / T), Math.exp((b - m) / T), Math.exp(-m / T)], s = e[0] + e[1] + e[2];
  return [e[0] / s, e[1] / s, e[2] / s];
}

function drawSoftmax(it, c) {
  const at = place(it), T = it.T, L = 6, K = 96;
  const A = at([1, 0, 0]), B = at([0, 1, 0]), C = at([0, 0, 1]), mid = at([1 / 3, 1 / 3, 1 / 3]);
  c.add(new THREE.Mesh(c.geometry([A, B, C]), c.mat('surface', it.color, 0.1)));
  c.add(new THREE.LineLoop(c.geometry([A, B, C]), c.mat('edge', it.color)));
  for (let g = -L; g <= L; g++) {
    for (const first of [true, false]) {
      const pts = [];
      for (let s = 0; s <= K; s++) {
        const t = -L + (2 * L * s) / K;
        pts.push(at(first ? softmaxPoint(g, t, T) : softmaxPoint(t, g, T)));
      }
      c.polyline(pts, it.color, { opacity: g === 0 ? 0.9 : 0.45 });
    }
  }
  c.dot(mid, it.color, 0.03 * Math.min(c.s, 1));
  [A, B, C].forEach((p, i) => c.label(p.clone().add(p.clone().sub(mid).multiplyScalar(0.18)), `e_${i + 1}`, it.color));
  if (it.label) c.label(mid.clone().add(new THREE.Vector3(0, 0, 0.35)), c.nameTex(it.label, false), it.color);
}

registerRenderer('graph', (it, c) => {
  if (it.mode === 'surface') drawSurface(it, c);
  else if (it.mode === 'curve') drawCurves(it, c, curvePolylines(it.at, it.ins[0], it.dep, c.E));
  else if (it.mode === 'param') drawCurves(it, c, paramPolylines(it.at, it.ins[0], c.E));
});
registerRenderer('softmaxmap', drawSoftmax);

export function install(api) {
  const help = document.getElementById('g-help'), keys = document.getElementById('g-help-keys');
  if (help) (keys ?? help).insertAdjacentHTML(keys ? 'beforebegin' : 'beforeend', HELP);
  // surfaces are clipped to the axes box with material clipping planes
  api.onSceneReady((scene) => { scene.renderer.localClippingEnabled = true; });
}
