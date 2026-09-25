// Net tab: the 3D net view (docs/NN_3D.md). While it is on, a WebGL stage covers #nn-stage and
// replaces the SVG canvas, in one of three modes:
//   stack   any net: each layer a sheet of neurons (a tokens × features grid on a token layer), the
//           sheets in depth; weight edges coloured as on the canvas, attention edges violet
//   heads   one attention layer: Q, K and V split by columns into a slab per head, each slab doing
//           its own Q_h K_hᵀ → A_h → A_h V_h = Z_h, then concat and W_O
//   tensor  the multi-head reshape as moving cubes: [T, d] → [T, h, d/h] → [h, T, d/h], attention
//           per head, and back to [T, d]; also the common bug (a view without the transpose)
// It follows the shared hover, selection, lens and step-through, and sets hover and selection as
// the canvas does, so the matrix panel, the cards and Explain stay linked. While it is on,
// ctx.view.nodeRect, contentRect and fit answer for the 3D stage (the SVG view's are put back after).
//
// state.v3d (owner: this module) is null (off) or a cleanV3d() object. The audience window mirrors
// it, the presenter's camera included (throttled). Three.js is imported the first time it opens.

import { colorFor } from './store.js';
import { emphasis, tokenNames, tokenLabel } from './focus.js';
import * as M from './model.js';

// ================================================================ pure helpers (tests: tests/nn_view3d.test.mjs)
export const MODES = ['stack', 'heads', 'tensor'];
export const EXAMPLE = Object.freeze({ T: 4, d: 6, seed: 5 });
const UI_KEY = 'mathboard.nn.view3d';      // { mode }: the view D opens in

const isNum = v => typeof v === 'number' && Number.isFinite(v);
const isObj = v => !!v && typeof v === 'object' && !Array.isArray(v);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const r3 = v => Math.round(v * 1000) / 1000;

export function divisors(n) {
  const out = [];
  for (let k = 1; k <= n; k++) if (n % k === 0) out.push(k);
  return out;
}

// A complete v3d from anything (null stays null): unknown or junk fields get their defaults.
//   mode: 'stack' | 'heads' | 'tensor'; camera: null | { p: [x, y, z], t: [x, y, z], k? } (position,
//   orbit target and distance / fit distance; null: the mode's own framing); layer: an attention layer id or null (the first);
//   step: the tensor step; src: 'net' | 'example' | null (auto); h: the example's heads (a divisor of
//   EXAMPLE.d); bug: the no-transpose bug; color: 'value' | 'token'; nums: numbers on cells.
export function cleanV3d(v) {
  if (!isObj(v)) return null;
  const vec = a => (Array.isArray(a) && a.length === 3 && a.every(isNum) ? a.slice() : null);
  const c = isObj(v.camera) ? { p: vec(v.camera.p), t: vec(v.camera.t) } : null;
  if (c && isNum(v.camera.k) && v.camera.k > 0) c.k = v.camera.k;
  return {
    mode: MODES.includes(v.mode) ? v.mode : 'stack',
    camera: c?.p && c?.t ? c : null,
    layer: typeof v.layer === 'string' && v.layer ? v.layer : null,
    step: Number.isInteger(v.step) && v.step >= 0 ? v.step : 0,
    src: v.src === 'net' || v.src === 'example' ? v.src : null,
    h: divisors(EXAMPLE.d).includes(v.h) ? v.h : 3,
    bug: v.bug === true,
    color: v.color === 'token' ? 'token' : 'value',
    nums: v.nums !== false,
  };
}

// Where element (t, f) of a row-major [T, d] tensor sits in each form of the multi-head reshape:
//   'TD'   [T, d]        row t, column f (head: the chunk it will go to)
//   'THD'  [T, h, d/h]   row t, chunk floor(f / dh), column f mod dh in it      Q.view(T, h, dh)
//   'HTD'  [h, T, d/h]   slab floor(f / dh), row t, column f mod dh             ... .transpose(0, 1)
//   'BUG'  [h, T, d/h]   Q.view(h, T, dh) straight from [T, d]: slab, row and column by the flat
//                        index k = t·d + f, so a "head" is a run of memory, not a column chunk
export function slotOf(form, t, f, { T, d, h }) {
  const dh = d / h;
  if (form === 'BUG') {
    const k = t * d + f;
    return { head: Math.floor(k / (T * dh)), row: Math.floor(k / dh) % T, col: k % dh };
  }
  return { head: Math.floor(f / dh), row: t, col: form === 'TD' ? f : f % dh };
}
// T × d -> h × T × dh, as 'HTD' (or 'BUG') lays it out; fromHeads is its inverse.
export function toHeads(X, form, h) {
  const T = X.length, d = T ? X[0].length : 0, dh = d / h;
  const out = Array.from({ length: h }, () => Array.from({ length: T }, () => new Array(dh).fill(0)));
  for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) {
    const s = slotOf(form, t, f, { T, d, h });
    out[s.head][s.row][s.col] = X[t][f];
  }
  return out;
}
export function fromHeads(Hs, form, T, d) {
  const h = Hs.length, out = Array.from({ length: T }, () => new Array(d).fill(0));
  for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) {
    const s = slotOf(form, t, f, { T, d, h });
    out[t][f] = Hs[s.head][s.row][s.col];
  }
  return out;
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export function matmul(A, B) {
  return A.map(row => (B[0] || []).map((_, j) => row.reduce((s, x, i) => s + x * B[i][j], 0)));
}
function softmaxRow(row) {
  const m = Math.max(...row.filter(x => x > -Infinity));
  const e = row.map(x => (x === -Infinity ? 0 : Math.exp(x - m)));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map(x => x / s);
}
// Per head: S = Q_h K_hᵀ · scale (masked cells -Infinity), A = softmax(S) by rows, Z = A V_h.
export function attend(Qh, Kh, Vh, { scale = 1, causal = false } = {}) {
  return Qh.map((Q, h) => {
    const K = Kh[h], V = Vh[h];
    const S = Q.map((q, i) => K.map((k, j) => (causal && j > i ? -Infinity : dot(q, k) * scale)));
    const A = S.map(softmaxRow);
    const Z = A.map(a => (V[0] || []).map((_, e) => a.reduce((s, w, j) => s + w * V[j][e], 0)));
    return { S, A, Z };
  });
}
// The steps of the tensor view: { key, form, show } with show 'qkv' (Q, K, V cubes), 'z' (the V
// cubes carry Z) or 'out' (Z W_O). The bug skips the transpose both ways.
export function tensorSteps({ bug = false, proj = false } = {}) {
  const list = bug
    ? [['td', 'TD', 'qkv'], ['view', 'BUG', 'qkv'], ['attend', 'BUG', 'z'], ['bugback', 'TD', 'z']]
    : [['td', 'TD', 'qkv'], ['split', 'THD', 'qkv'], ['heads', 'HTD', 'qkv'], ['attend', 'HTD', 'z'], ['back', 'THD', 'z'], ['merge', 'TD', 'z']];
  if (proj) list.push(['proj', 'TD', 'out']);
  return list.map(([key, form, show]) => ({ key, form, show }));
}
// Everything the tensor view shows, from Q, K, V (T × d each): the head stacks in the chosen
// layout, per-head attention, Z back as T × d, and Z W_O when WO is given.
export function tensorStory({ Q, K, V, h, scale = 1, causal = false, WO = null, bug = false }) {
  const T = Q.length, d = T ? Q[0].length : 0, form = bug ? 'BUG' : 'HTD';
  const [Qh, Kh, Vh] = [Q, K, V].map(X => toHeads(X, form, h));
  const heads = attend(Qh, Kh, Vh, { scale, causal });
  const Z = fromHeads(heads.map(x => x.Z), form, T, d);
  return { T, d, h, dh: d / h, form, steps: tensorSteps({ bug, proj: !!WO }), Qh, Kh, Vh, heads, Z, out: WO ? matmul(Z, WO) : null };
}
// The 4 × 6 example: fixed pseudo-random Q, K, V and W_O (two decimals).
export function exampleQKV({ T, d, seed } = EXAMPLE) {
  const r = M.rng(seed);
  const mk = (R, C, k) => Array.from({ length: R }, () => Array.from({ length: C }, () => Math.round((r() * 2 - 1) * k * 100) / 100));
  return { Q: mk(T, d, 1.3), K: mk(T, d, 1.3), V: mk(T, d, 1), WO: mk(d, d, 0.7) };
}
// The tied, tokenwise matrix from attention layer l into layer l + 1 (W_O), if there is one.
export function projOf(net, l) {
  if (!(l + 1 < (net?.layers?.length ?? 0))) return null;
  const m = M.tiedMatrices(net, l + 1).find(x => x.k === l && x.tokenwise);
  return m ? { name: m.name, W: m.W, ties: m.ties, edges: m.edges, l: l + 1 } : null;
}
export function attnLayers(net) {
  return (net?.layers || []).map((l, i) => (l.kind === 'attention' && M.attnSpec(net, i) ? i : -1)).filter(i => i >= 0);
}

// ---------------------------------------------------------------- captions (tensor mode)
const TENSOR_TEXT = {
  td: { title: 'one row per token', code: 'Q = X @ W_Q', shape: '[T, d]', note: 'Q, K and V are T tokens × d features. Every step below does the same to all three.' },
  split: { title: 'split the features', code: 'Q = Q.view(T, h, d_h)', shape: '[T, h, d/h]', note: 'Each row is cut into h chunks of d/h features, one per head. Same numbers in the same memory order: only the shape changed.' },
  heads: { title: 'heads to the front', code: 'Q = Q.transpose(0, 1)', shape: '[h, T, d/h]', note: 'Now head is the leading axis: head h is a T × d/h matrix holding chunk h of every token.' },
  attend: { title: 'attend, every head at once', code: 'Z = softmax(Q @ K.mT * s) @ V', shape: '[h, T, d/h]', note: 'One batched matmul over the head axis: each head gets its own T × T attention A_h (the bars), and Z_h = A_h V_h replaces V.' },
  back: { title: 'tokens to the front', code: 'Z = Z.transpose(0, 1)', shape: '[T, h, d/h]', note: 'Undo the transpose: row t gathers token t’s chunk from every head.' },
  merge: { title: 'concat the heads', code: 'Z = Z.reshape(T, d)', shape: '[T, d]', note: 'The head axis folds back into the features: Z = concat(Z_1, …, Z_h). After a transpose the memory is out of order, so this needs reshape (or .contiguous().view).' },
  proj: { title: 'mix the heads', code: 'out = Z @ W_O', shape: '[T, d]', note: 'W_O is d × d: every output feature reads every head.' },
  view: { title: 'the bug: a view, no transpose', code: 'Q = Q.view(h, T, d_h)   # wrong', shape: '[h, T, d/h]', note: '' },
  bugback: { title: 'back to [T, d], wrong numbers', code: 'Z = Z.view(T, d)', shape: '[T, d]', note: '' },
};

// ================================================================ install
export function install(ctx) {
  const { store } = ctx;
  const stage = ctx.el?.stage || document.getElementById('nn-stage');
  const audience = !!ctx.audience;
  const theme = () => ctx.theme?.() || (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  let lastMode = 'stack';
  try { const s = JSON.parse(localStorage.getItem(UI_KEY)); if (MODES.includes(s?.mode)) lastMode = s.mode; } catch { /* default */ }
  let G = null, gen = 0, threeP = null;

  const btn = audience ? null : ctx.addButton?.({
    label: '3D', icon: window.mathboardIcons?.svg('layers') ? 'layers' : '&#11041;', group: 'view', onClick: () => toggle(),
    title: '3D view (D): the net in depth, attention heads as slabs, or the multi-head reshape as moving cubes. Shift+D: next view',
  }) || null;

  const cur = () => cleanV3d(store.state.v3d);
  const hasAttn = () => attnLayers(store.net).length > 0;
  function put(patch) {
    if (audience) return;
    const v = cur();
    if (v) store.set('v3d', { ...v, ...patch });
  }
  function open(mode = lastMode) {
    if (audience) return;
    if (mode === 'heads' && !hasAttn()) mode = 'stack';
    store.set('v3d', cleanV3d({ mode, nums: mode === 'tensor' }));
  }
  function toggle(on = !store.state.v3d) {
    if (audience) return;
    if (on) open();
    else store.set('v3d', null);
  }
  function setMode(mode) {
    if (!MODES.includes(mode)) return;
    if (mode === 'heads' && !hasAttn()) { ctx.toast?.('Heads needs an attention layer: try the "Two heads" or "Transformer block" preset', 3200); return; }
    if (!cur()) open(mode);
    else if (cur().mode !== mode) put({ mode, camera: null, step: 0, nums: mode === 'tensor' });
  }
  function cycle(dir = 1) {
    const list = MODES.filter(m => m !== 'heads' || hasAttn()), v = cur();
    const i = list.indexOf(v ? v.mode : lastMode);
    const next = list[(Math.max(0, i) + (v ? dir : 0) + list.length) % list.length];
    if (!v) open(next);
    else setMode(next);
    ctx.toast?.(`3D: ${next}`, 900);
  }

  async function loadThree() {
    const [THREE, oc, css] = await Promise.all([
      import('three'), import('three/addons/controls/OrbitControls.js'), import('three/addons/renderers/CSS2DRenderer.js')]);
    return { THREE, OrbitControls: oc.OrbitControls, CSS2DRenderer: css.CSS2DRenderer, CSS2DObject: css.CSS2DObject };
  }
  async function sync() {
    const v = cur();
    btn?.classList.toggle('on', !!v);
    if (!v) {
      gen++;
      if (G) { const g = G; G = null; g.dispose(); }
      return;
    }
    if (v.mode !== lastMode && !audience) {
      lastMode = v.mode;
      try { localStorage.setItem(UI_KEY, JSON.stringify({ mode: lastMode })); } catch { /* full */ }
    }
    if (!G) {
      const my = ++gen;
      let K;
      try { K = await (threeP ||= loadThree()); }
      catch (err) {
        threeP = null;
        console.error('[nn/view3d] three:', err);
        ctx.toast?.('The 3D view could not load Three.js');
        if (!audience) store.set('v3d', null);
        return;
      }
      if (my !== gen || !store.state.v3d) return;
      if (!G) {
        try { G = createStage(K); }
        catch (err) {
          console.error('[nn/view3d] stage:', err);
          ctx.toast?.('The 3D view needs WebGL');
          G = null;
          if (!audience) store.set('v3d', null);
          return;
        }
      }
    }
    G.apply(cur());
  }

  store.on('v3d', () => { sync(); });
  store.on('net', p => G?.onNet(p));
  store.on('values', () => G?.invalidate());
  for (const k of ['sel', 'hover', 'anim', 'lens']) store.on(k, () => G?.invalidate());
  ctx.onTheme?.(() => G?.retheme());
  ctx.onShow?.(on => G?.shown(on));

  if (!audience) {
    window.addEventListener('keydown', e => {
      if (e.ctrlKey || e.metaKey || e.altKey || !(ctx.active ? ctx.active(e) : true)) return;
      const k = e.key;
      if (k === 'd' || k === 'D') {
        e.preventDefault();
        if (e.repeat) return;
        if (e.shiftKey) cycle(1);
        else toggle();
      } else if ((k === 'ArrowRight' || k === 'ArrowLeft') && cur()?.mode === 'tensor') {
        e.preventDefault();
        G?.step(k === 'ArrowRight' ? 1 : -1);
      }
    });
  }

  // Test / console handle (docs/NN_3D.md).
  ctx.view3d = {
    MODES,
    get on() { return !!store.state.v3d; },
    get mode() { return cur()?.mode ?? null; },
    get ready() { return !!G?.built; },
    toggle, open, setMode, cycle,
    step: dir => G?.step(dir),
    fit: ms => G?.fit(ms),
    info: () => G?.info() ?? null,
  };

  // ============================================================== the stage
  function createStage(K) {
    const { THREE, OrbitControls, CSS2DRenderer, CSS2DObject } = K;
    const FOV = 24;
    const DIRS = { stack: [0.36, 0.42, 0.84], heads: [0.34, 0.3, 1], tensor: [0.62, 0.28, 1] };
    const DIM = 0.1, NUM_MIN = 0.25, FOCUS_NODE = 0.42, FOCUS_EDGE = 0.35, FOCUS_ATT = 0.25;
    const FIXED_A = 0.34;                    // a fixed edge's strength at rest (full while held)
    const NUM_PX = 28;                       // heads, tensor: a cell's number shows when its face is this wide on screen
    const PUB_MS = 120, STEP_MS = 1100, PLAY_MS = 2600, SETTLE_MS = 800;
    const NODE_R = 0.32, CUBE = 0.8, TD = 0.34;   // TD: a cell tile's depth
    const CELL_A = 0.72;                     // cells cap colorFor's alpha as the matrix panel does, so numbers read on them
    const MAX_DPR = 1.5;                     // pixel ratio cap: MSAA does the rest
    const calm = () => !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    // ---------------------------------------------------------------- DOM
    const mk = (tag, cls, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (parent) parent.appendChild(e); return e; };
    const root = mk('div', `nn3d${audience ? ' ro' : ''}`);
    const gl = mk('div', 'nn3d-gl', root);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const dpr = () => Math.min(MAX_DPR, window.devicePixelRatio || 1);
    renderer.setPixelRatio(dpr());
    gl.appendChild(renderer.domElement);
    const labelR = new CSS2DRenderer();
    labelR.domElement.className = 'nn3d-labels';
    gl.appendChild(labelR.domElement);
    const bar = mk('div', 'nn3d-bar', root);
    const msg = mk('div', 'nn3d-msg', root);
    msg.hidden = true;
    stage.appendChild(root);
    const svg = ctx.view?.svg || null;
    if (svg) svg.style.visibility = 'hidden';
    bar.addEventListener('mousedown', e => { if (e.target.closest('button')) e.preventDefault(); });

    const scene = new THREE.Scene();
    scene.matrixWorldAutoUpdate = false;   // the loop updates the scene graph once a frame, for both renderers
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 5000);
    scene.add(camera);   // it carries the key light
    // Soft, matte light (Lambert: no specular): a hemisphere, sky over a grey ground, and one gentle
    // key that rides with the camera, a little up and to the left of it. The face turned to the
    // viewer gets about 1 (so a value's colour reads as in the matrix panel), the sides shade off
    // to about 0.85 and the undersides to about 0.6. Intensities carry the π of three's lights.
    const hemi = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.9 * Math.PI);
    hemi.groundColor.setRGB(0.35, 0.35, 0.35, THREE.LinearSRGBColorSpace);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 0.45 * Math.PI);
    key.position.set(-1.1, 1.6, 4);
    key.target.position.set(0, 0, -4);
    camera.add(key, key.target);
    // Light fog toward the page colour: the back of the content falls off by about a third.
    scene.fog = new THREE.Fog(0x000000, 10, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.6;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.8;
    controls.screenSpacePanning = true;
    controls.maxPolarAngle = Math.PI * 0.62;   // a little under the horizon at most: the ground stays a floor
    controls.enabled = !audience;
    camera.position.set(-10, 8, 12);

    // ---------------------------------------------------------------- shared geometry and materials
    const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
    const UP = V3(0, 1, 0);
    const tm = new THREE.Matrix4(), tq = new THREE.Quaternion(), ts = V3(), tp = V3(), tv = V3(), tc = new THREE.Color();
    // A rounded rectangle, w × h, centred.
    function roundShape(w, h, r) {
      const s = new THREE.Shape(), x = w / 2, y = h / 2;
      r = Math.min(r, x, y);
      s.moveTo(-x + r, -y);
      s.lineTo(x - r, -y); s.absarc(x - r, -y + r, r, -Math.PI / 2, 0);
      s.lineTo(x, y - r); s.absarc(x - r, y - r, r, 0, Math.PI / 2);
      s.lineTo(-x + r, y); s.absarc(-x + r, y - r, r, Math.PI / 2, Math.PI);
      s.lineTo(-x, -y + r); s.absarc(-x + r, -y + r, r, Math.PI, 1.5 * Math.PI);
      return s;
    }
    // A cell: a 1 × 1 rounded tile, `depth` deep, centred. Indexed and built by hand (66 vertices,
    // not an ExtrudeGeometry's 324): the corners' sides shade smoothly, the faces are flat.
    function tileGeo(depth, r = 0.14, seg = 3) {
      const h = 0.5, pts = [];
      for (const [cx, cy, a0] of [[h - r, -h + r, -Math.PI / 2], [h - r, h - r, 0], [-h + r, h - r, Math.PI / 2], [-h + r, -h + r, Math.PI]]) {
        for (let k = 0; k <= seg; k++) { const a = a0 + (k / seg) * (Math.PI / 2); pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a), Math.cos(a), Math.sin(a)]); }
      }
      const n = pts.length, z1 = depth / 2, z0 = -depth / 2, pos = [], nor = [], idx = [];
      const cap = (z, nz) => {   // a fan round the centre, facing nz
        const c = pos.length / 3;
        pos.push(0, 0, z); nor.push(0, 0, nz);
        for (const [x, y] of pts) { pos.push(x, y, z); nor.push(0, 0, nz); }
        for (let k = 0; k < n; k++) { const a = c + 1 + k, b = c + 1 + ((k + 1) % n); if (nz > 0) idx.push(c, a, b); else idx.push(c, b, a); }
      };
      cap(z1, 1);
      cap(z0, -1);
      const s0 = pos.length / 3;   // the sides: two rings with outward normals
      for (const [x, y, nx, ny] of pts) { pos.push(x, y, z1, x, y, z0); nor.push(nx, ny, 0, nx, ny, 0); }
      for (let k = 0; k < n; k++) { const a = s0 + 2 * k, c = s0 + 2 * ((k + 1) % n); idx.push(a, a + 1, c + 1, a, c + 1, c); }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
      g.setIndex(idx);
      return g;
    }
    const GEO = {
      sphere: new THREE.SphereGeometry(1, 22, 16), sphereLo: new THREE.SphereGeometry(1, 14, 10),   // Lo: nets of 300+ neurons
      tile: tileGeo(TD), bar: tileGeo(1),
      plane: new THREE.PlaneGeometry(1, 1), ring: new THREE.TorusGeometry(1, 0.05, 6, 48), ringThin: new THREE.TorusGeometry(1, 0.032, 6, 48),
      ball: new THREE.SphereGeometry(1, 12, 8),
      // a square frame (half-side 0.5 to 0.58) for a lit cell's front face
      frame: new THREE.RingGeometry(0.5 * Math.SQRT2, 0.58 * Math.SQRT2, 4, 1, Math.PI / 4),
    };
    // Screen-space lines (weights, attention, the heads' fans): one instance per segment, a width in
    // CSS px whatever the zoom, so thin ones stay crisp instead of shimmering; under 1 px a line is
    // drawn 1 px wide and fainter. Colours are sRGB, already blended toward the page (as the canvas
    // blends colorFor's alpha), and fog toward it by depth as the meshes do.
    const LINE_VS = `
      attribute vec3 iA;
      attribute vec3 iB;
      attribute vec3 iC;
      attribute float iW;
      uniform vec2 uRes;
      uniform float uPx;
      uniform float uNear;
      uniform vec3 uBg;
      uniform vec2 uFog;
      varying vec3 vC;
      void main() {
        vC = uBg;
        vec4 a = modelViewMatrix * vec4(iA, 1.0);
        vec4 b = modelViewMatrix * vec4(iB, 1.0);
        float zn = -uNear;
        if (iW <= 0.0 || (a.z > zn && b.z > zn)) { gl_Position = vec4(0.0, 0.0, -2.0, 1.0); return; }
        if (a.z > zn) a = mix(a, b, (a.z - zn) / (a.z - b.z));
        else if (b.z > zn) b = mix(b, a, (b.z - zn) / (b.z - a.z));
        vec4 ca = projectionMatrix * a;
        vec4 cb = projectionMatrix * b;
        vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * uRes;
        float len = length(d);
        d = len > 1e-5 ? d / len : vec2(1.0, 0.0);
        vec4 c = position.x < 0.5 ? ca : cb;
        c.xy += vec2(-d.y, d.x) * position.y * max(iW, 1.0) * uPx / uRes * c.w;
        gl_Position = c;
        float depth = -(position.x < 0.5 ? a.z : b.z);
        vC = mix(mix(uBg, iC, clamp(iW, 0.0, 1.0)), uBg, smoothstep(uFog.x, uFog.y, depth));
      }`;
    const LINE_FS = `
      varying vec3 vC;
      void main() { gl_FragColor = vec4(vC, 1.0); }`;
    // The ground: a faint dot grid under the content, fading out toward its rim (aF) and with the
    // fog. One point sprite per dot, so it costs only the dots' pixels; a dot under a pixel fades
    // instead of sparkling, as a thin line does.
    const GROUND_VS = `
      attribute float aF;
      uniform float uDot;
      uniform float uA;
      uniform vec2 uRes;
      uniform vec2 uFog;
      varying float vA;
      varying float vR;
      varying float vS;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        float px = uDot * projectionMatrix[1][1] * uRes.y * 0.5 / max(1e-3, -mv.z);
        float d = max(px, 1.0);
        vS = d + 2.0;
        gl_PointSize = vS;
        vR = 0.5 * d / vS;
        vA = uA * aF * clamp(px, 0.0, 1.0) * (1.0 - smoothstep(uFog.x, uFog.y, -mv.z));
      }`;
    const GROUND_FS = `
      uniform vec3 uC;
      varying float vA;
      varying float vR;
      varying float vS;
      void main() {
        float e = 0.7 / vS;
        float a = vA * (1.0 - smoothstep(vR - e, vR + e, length(gl_PointCoord - 0.5)));
        if (a < 0.002) discard;
        gl_FragColor = vec4(uC, a);
      }`;
    const fogU = { value: new THREE.Vector2(1e4, 2e4) };
    const MAT = {
      lit: new THREE.MeshLambertMaterial(),   // neurons, cells, bars: instance colours under the matte light
      flat: new THREE.MeshBasicMaterial(),    // the head accents: exact colour
      band: new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.05, depthWrite: false, side: THREE.DoubleSide }),    // stack: the held layer
      plate: new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.05, depthWrite: false, side: THREE.DoubleSide }),   // heads, tensor
      halo: new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, fog: false }),
      ring: new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, fog: false }),
      line: new THREE.ShaderMaterial({
        vertexShader: LINE_VS, fragmentShader: LINE_FS,
        uniforms: { uRes: { value: new THREE.Vector2(1, 1) }, uPx: { value: 1 }, uNear: { value: camera.near }, uBg: { value: new THREE.Vector3() }, uFog: fogU },
      }),
      ground: new THREE.ShaderMaterial({
        vertexShader: GROUND_VS, fragmentShader: GROUND_FS, transparent: true, depthWrite: false,
        uniforms: { uC: { value: new THREE.Vector3(1, 1, 1) }, uA: { value: 0.1 }, uDot: { value: 0.1 }, uRes: null, uFog: fogU },
      }),
    };

    MAT.ground.uniforms.uRes = MAT.line.uniforms.uRes;   // one resolution for both (size() sets it)

    // ---------------------------------------------------------------- theme
    const hexRGB = s => {
      const m = /^#?([0-9a-f]{6})$/i.exec(String(s).trim());
      if (m) return [0, 2, 4].map(k => parseInt(m[1].slice(k, k + 2), 16));
      const r = /rgba?\(([^)]+)\)/.exec(String(s));
      return r ? r[1].split(',').slice(0, 3).map(x => +x) : null;
    };
    // Colour is for data (the value pair, attention, the heads, HI). Structure is neutral: the
    // neurons' and cells' base (--n3-node), plates (--n3-plate), the ground (--n3-ground) and fixed
    // edges and fans in the muted text colour (--text-3), as on the 2D canvas. The --n3-* tokens
    // live in view3d.css (literal hex / rgba: read here).
    const PAL = {
      dark: { node: [78, 88, 95], hi: [255, 213, 74], att: [183, 148, 255], line: [255, 255, 255], fixed: [147, 156, 162], bgFall: [29, 35, 39],
        plate: [255, 255, 255, 0.045], ground: [255, 255, 255, 0.12],
        toks: ['#f0a23b', '#4fb3d9', '#7bd672', '#e0629a', '#c9a0ff', '#e8d94a', '#ff8a5c', '#9fb0bd'] },
      light: { node: [226, 229, 227], hi: [232, 168, 0], att: [116, 66, 214], line: [0, 0, 0], fixed: [107, 112, 117], bgFall: [251, 251, 248],
        plate: [0, 0, 0, 0.04], ground: [0, 0, 0, 0.11],
        toks: ['#d9822b', '#1f86b8', '#3c9a35', '#c2185b', '#7442d6', '#a38c00', '#e0562b', '#5c6b77'] },
    };
    // a head keeps its colour in every panel: the head tokens (style.css section 1, literal hex)
    const HEAD_VARS = ['--att', '--head-2', '--head-3', '--head-4', '--head-5', '--head-6'];
    const rgbaOf = s => {   // '#rrggbb' or 'rgba(r, g, b, a)' -> [r, g, b, a] | null
      const m = /^#?([0-9a-f]{6})$/i.exec(String(s).trim());
      if (m) return [...[0, 2, 4].map(k => parseInt(m[1].slice(k, k + 2), 16)), 1];
      const r = /rgba?\(([^)]+)\)/.exec(String(s));
      if (!r) return null;
      const p = r[1].split(',').map(x => +x);
      return p.length >= 3 && p.slice(0, 3).every(Number.isFinite) ? [p[0], p[1], p[2], Number.isFinite(p[3]) ? p[3] : 1] : null;
    };
    const srgb = (c, v3 = V3()) => v3.set(c[0] / 255, c[1] / 255, c[2] / 255);
    let P = null;
    function retheme() {
      const th = theme(), base = PAL[th], cs = getComputedStyle(document.documentElement), rs = getComputedStyle(root);
      const bg = hexRGB(cs.getPropertyValue('--bg')) || base.bgFall;
      const heads = HEAD_VARS.map(v => hexRGB(cs.getPropertyValue(v)) || base.att);
      const tok = (name, fall) => rgbaOf(rs.getPropertyValue(name)) || fall;
      const node = tok('--n3-node', base.node).slice(0, 3), plate = tok('--n3-plate', base.plate), ground = tok('--n3-ground', base.ground);
      P = { ...base, th, bg, heads, node, plate, ground, toks: base.toks.map(hexRGB) };
      cfInit(th);
      const bgc = new THREE.Color().setRGB(bg[0] / 255, bg[1] / 255, bg[2] / 255, THREE.SRGBColorSpace);
      scene.background = bgc;
      scene.fog.color.copy(bgc);
      MAT.band.color.setRGB(plate[0] / 255, plate[1] / 255, plate[2] / 255, THREE.SRGBColorSpace);
      MAT.band.opacity = Math.min(1, plate[3] * 1.5);
      MAT.plate.opacity = plate[3];
      MAT.ring.color.setRGB(...base.hi.map(x => x / 255), THREE.SRGBColorSpace);
      MAT.halo.color.setRGB(...base.hi.map(x => x / 255), THREE.SRGBColorSpace);
      srgb(bg, MAT.line.uniforms.uBg.value);
      srgb(ground, MAT.ground.uniforms.uC.value);
      MAT.ground.uniforms.uA.value = ground[3];
      invalidate();
    }
    const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    // colorFor as numbers, so the hot paths (every edge, every frame while training) parse no
    // strings: [r, g, b, a]. cfInit checks it against store.js colorFor for the theme and falls
    // back to parsing colorFor's string if they ever differ.
    const rgba = s => { const m = /rgba?\(([^)]+)\)/.exec(s); const p = m ? m[1].split(',').map(Number) : [128, 128, 128, 0.4]; return [p[0], p[1], p[2], p[3] ?? 1]; };
    let cf = (v, max) => rgba(colorFor(v, max, P.th));
    function cfInit(th) {
      const pos = rgba(colorFor(1, 1, th)), neg = rgba(colorFor(-1, 1, th)), bad = rgba(colorFor(NaN, 1, th));
      const fast = (v, max, out = [0, 0, 0, 0]) => {
        const c = !Number.isFinite(v) ? bad : v >= 0 ? pos : neg;
        out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
        out[3] = Number.isFinite(v) ? 0.12 + 0.88 * Math.min(1, Math.abs(v) / (max || 1)) : c[3];
        return out;
      };
      const same = [0.5, -0.25, 0, 2, -3, 0.04].every(v => { const a = rgba(colorFor(v, 1, th)), b = fast(v, 1); return a.every((x, i) => Math.abs(x - b[i]) < 2e-3); });
      cf = same ? fast : (v, max, out) => { const c = rgba(colorFor(v, max, th)); if (out) { out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3]; return out; } return c; };
    }
    // colorFor's colour laid over base (as its alpha would blend it on the canvas); cap: the most
    // alpha a cell takes (CELL_A), eased as the matrix panel eases it above 0.5.
    const easeCap = (a, cap) => (a <= 0.5 ? a : 0.5 + (cap - 0.5) * Math.min(1, (a - 0.5) / 0.5));
    const over = (v, max, base, cap = 1) => { const c = cf(v, max); return mix(base, c, cap < 1 ? easeCap(c[3], cap) : c[3]); };
    const setCol = (mesh, i, c) => { tc.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace); mesh.setColorAt(i, tc); };
    // Lens emphasis -> opacity, as on the canvas (0.1 + 0.9 v; 1 at full emphasis).
    const le = v => (v >= 0.995 ? 1 : DIM + (1 - DIM) * clamp(v, 0, 1));
    const num = v => (isNum(v) ? M.fmt(v, 2).replace(/^-/, '−') : '—');
    const katexHtml = tex => {
      const k = window.katex;
      try { return k ? k.renderToString(String(tex), { throwOnError: false }) : esc(tex); } catch { return esc(tex); }
    };
    const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const sub = (b, s) => `<i>${esc(b)}</i><sub>${esc(s)}</sub>`;
    function maxAbs(...xs) {
      let m = 0;
      const walk = x => { if (Array.isArray(x)) x.forEach(walk); else if (isNum(x)) m = Math.max(m, Math.abs(x)); };
      xs.forEach(walk);
      return m;
    }

    // ---------------------------------------------------------------- content scaffolding
    // A mode's scene content: its group, its labels (removed from the DOM on dispose) and its
    // instanced meshes (the geometries and materials are shared, so only the instances go).
    function newContent() {
      const group = new THREE.Group();
      group.matrixAutoUpdate = false;   // content objects are placed once (instances and labels carry the motion)
      scene.add(group);
      const labs = [], geos = [];
      return {
        group, labs, geos,
        label(cls, html, pos, cx = 0.5, cy = 0.5, parent = group) {
          const el = document.createElement('div');
          el.className = `nn3d-lab ${cls}`;
          if (html != null) el.innerHTML = html;
          const o = new CSS2DObject(el);
          o.position.copy(pos);
          o.center.set(cx, cy);
          parent.add(o);
          labs.push(o);
          return o;
        },
        inst(geo, mat, n, order = 0) {
          const m = new THREE.InstancedMesh(geo, mat, Math.max(1, n));
          m.count = n;
          m.frustumCulled = false;
          m.renderOrder = order;
          m.matrixAutoUpdate = false;
          tc.setRGB(1, 1, 1);
          for (let i = 0; i < n; i++) m.setColorAt(i, tc);
          group.add(m);
          return m;
        },
        // n screen-space line segments (LINE_VS): seg(i, a, b) once, then set(i, rgb, px) per paint
        // (px 0 hides one), then flush().
        lines(n, order = 0) {
          const m = Math.max(1, n), g = new THREE.InstancedBufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]), 3));
          g.setIndex([0, 1, 2, 0, 2, 3]);
          const at = k => new THREE.InstancedBufferAttribute(new Float32Array(m * k), k);
          const A = at(3), B = at(3), Cc = at(3), W = at(1);
          Cc.setUsage(THREE.DynamicDrawUsage);
          W.setUsage(THREE.DynamicDrawUsage);
          g.setAttribute('iA', A); g.setAttribute('iB', B); g.setAttribute('iC', Cc); g.setAttribute('iW', W);
          g.instanceCount = n;
          const mesh = new THREE.Mesh(g, MAT.line);
          mesh.frustumCulled = false;
          mesh.renderOrder = order;
          mesh.matrixAutoUpdate = false;
          group.add(mesh);
          geos.push(g);
          const a = A.array, b = B.array, c = Cc.array, w = W.array;
          return {
            n, mesh,
            seg(i, p, q) { const k = 3 * i; a[k] = p.x; a[k + 1] = p.y; a[k + 2] = p.z; b[k] = q.x; b[k + 1] = q.y; b[k + 2] = q.z; A.needsUpdate = B.needsUpdate = true; },
            set(i, rgb, px) { const k = 3 * i; c[k] = rgb[0] / 255; c[k + 1] = rgb[1] / 255; c[k + 2] = rgb[2] / 255; w[i] = px; },
            flush() { Cc.needsUpdate = W.needsUpdate = true; },
          };
        },
        // a rounded plate (w × h, facing +z; 'yz' faces +x), its geometry owned by this content
        plateGeo(w, h, r = 0.22) {
          const g = new THREE.ShapeGeometry(roundShape(w, h, r), 4);
          geos.push(g);
          return g;
        },
        // the faint ground under the content's box: a dot grid on a disc, fading out at its rim,
        // on world multiples of its step (so the pattern stays put across rebuilds)
        ground(box) {
          if (box.isEmpty()) return null;
          const c = box.getCenter(V3()), sz = box.getSize(V3());
          const R = Math.max(sz.x, sz.z) * 0.5 + 2, step = Math.max(1, Math.round(R / 10)), y = box.min.y - 0.9;
          const pos = [], fade = [];
          for (let x = Math.ceil((c.x - R) / step) * step; x <= c.x + R; x += step) {
            for (let z = Math.ceil((c.z - R) / step) * step; z <= c.z + R; z += step) {
              const r = Math.hypot(x - c.x, z - c.z) / R;
              if (r >= 1) continue;
              const u = clamp((r - 0.1) / 0.9, 0, 1);
              pos.push(x, y, z);
              fade.push(1 - u * u * (3 - 2 * u));
            }
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
          g.setAttribute('aF', new THREE.Float32BufferAttribute(fade, 1));
          geos.push(g);
          const o = new THREE.Points(g, MAT.ground);
          o.renderOrder = -10;
          o.frustumCulled = false;
          o.matrixAutoUpdate = false;
          group.add(o);
          MAT.ground.uniforms.uDot.value = 0.085 * step;
          return o;
        },
        dispose() {
          for (const o of labs) o.element.remove();
          group.traverse(o => { if (o.isInstancedMesh) o.dispose(); });
          for (const g of geos) g.dispose();
          scene.remove(group);
        },
      };
    }
    const setBox = (mesh, i, p, sx, sy = sx, sz = sx, q = null) => {
      ts.set(sx, sy, sz);
      tm.compose(p, q || tq.identity(), ts);
      mesh.setMatrixAt(i, tm);
    };
    const flushInst = m => { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; };
    // a plate facing +z turned to face +x (a stack sheet's y-z plane)
    const ROT_YZ = new THREE.Quaternion().setFromAxisAngle(UP, Math.PI / 2);

    // ---------------------------------------------------------------- net index (as view.js's ix)
    function index(net) {
      const li = new Map(), byLayer = net.layers.map(() => []), nodeById = new Map(), rank = new Map();
      net.layers.forEach((l, i) => li.set(l.id, i));
      for (const n of net.nodes) {
        nodeById.set(n.id, n);
        const i = li.get(n.layer);
        if (i === undefined) continue;
        rank.set(n.id, byLayer[i].length);
        byLayer[i].push(n);
      }
      const edgeById = new Map(), ties = new Map(), biasTies = new Map();
      for (const e of net.edges) {
        edgeById.set(e.id, e);
        if (typeof e.tie === 'string' && e.tie) { if (!ties.has(e.tie)) ties.set(e.tie, []); ties.get(e.tie).push(e.id); }
      }
      for (const n of net.nodes) {
        if (typeof n.tie !== 'string' || !n.tie) continue;
        if (!biasTies.has(n.tie)) biasTies.set(n.tie, []);
        biasTies.get(n.tie).push(n.id);
      }
      const shape = net.layers.map((lay, i) => {
        const s = M.tokenShape(net, i), G = s.groups ? s.groups.length : 1;
        const tokenish = s.tokens > 1 || !!s.groups || lay.kind === 'attention';
        return tokenish && s.d > 0 && byLayer[i].length === G * s.tokens * s.d ? { T: s.tokens, d: s.d, groups: s.groups, G } : null;
      });
      const att = net.layers.map((lay, i) => {
        const a = lay.kind === 'attention' ? M.attnSpec(net, i) : null, g = shape[i - 1]?.groups;
        return a && g ? { ...a, T: a.tokens, qG: g.indexOf('Q'), kG: g.indexOf('K'), vG: g.indexOf('V') } : null;
      });
      const lix = v => (typeof v === 'number' ? v : li.get(v));
      const pos = id => {
        const n = nodeById.get(id), l = n && li.get(n.layer), s = l !== undefined ? shape[l] : null;
        if (!s) return null;
        const k = rank.get(id);
        return { l, g: Math.floor(k / (s.T * s.d)), t: Math.floor(k / s.d) % s.T, f: k % s.d };
      };
      const tokNode = (l, g, t, f) => { const s = shape[l]; return s ? byLayer[l][(g * s.T + t) * s.d + f] : undefined; };
      return { li, lix, byLayer, nodeById, rank, edgeById, ties, biasTies, shape, att, pos, tokNode, of: [net.nodes, net.edges, net.layers] };
    }
    // Undo, redo and the audience mirror swap in new node and edge objects with the same ids (no
    // rebuild): a mode's paint and pick re-index when the arrays are no longer the ones indexed.
    const stale = (I, net) => I.of[0] !== net.nodes || I.of[1] !== net.edges || I.of[2] !== net.layers;
    // Stable key of what the 3D layout depends on: a change rebuilds, anything else repaints.
    function netKey(net) {
      return JSON.stringify([
        net.layers.map(l => [l.id, l.name, l.act, l.kind, l.tokens, l.groups, l.heads, l.causal]),
        net.nodes.map(n => [n.id, n.layer, n.label]), net.edges.map(e => `${e.id}${e.fixed ? '!' : ''}${e.tie || ''}`),
        net.meta?.tokenNames ?? null, net.meta?.title ?? null,
      ]);
    }

    // ---------------------------------------------------------------- highlight rules (as view.js)
    // What a sel / hover target lights: { nodes, edges, rel, relE, ties, tokens, any }.
    function resolve(I, t) {
      const out = { nodes: new Set(), edges: new Set(), rel: new Set(), relE: new Set(), ties: new Set(), any: false };
      if (!t) return out;
      const net = store.net;
      const touch = id => {
        for (const e of net.edges) {
          if (e.from === id) { out.relE.add(e.id); out.rel.add(e.to); }
          else if (e.to === id) { out.relE.add(e.id); out.rel.add(e.from); }
        }
      };
      const into = id => { for (const e of net.edges) if (e.to === id) { out.relE.add(e.id); out.rel.add(e.from); } };
      switch (t.kind) {
        case 'node': case 'bias': {
          if (!I.nodeById.has(t.id)) break;
          out.nodes.add(t.id);
          if (t.kind === 'bias') { for (const id of I.biasTies.get(I.nodeById.get(t.id).tie) || []) out.rel.add(id); break; }
          touch(t.id);
          break;
        }
        case 'edge': {
          const e = I.edgeById.get(t.id);
          if (!e) break;
          out.edges.add(e.id);
          out.rel.add(e.from); out.rel.add(e.to);
          for (const o of (e.tie && I.ties.get(e.tie)) || []) {
            if (o === e.id) continue;
            const x = I.edgeById.get(o);
            out.ties.add(o); out.rel.add(x.from); out.rel.add(x.to);
          }
          break;
        }
        case 'layer': {
          const i = I.li.get(t.id);
          if (i === undefined) break;
          for (const n of I.byLayer[i]) { out.rel.add(n.id); into(n.id); }
          out.layer = i;
          break;
        }
        case 'pair':
          if (I.nodeById.has(t.from) && I.nodeById.has(t.to)) { out.rel.add(t.from); out.rel.add(t.to); out.pair = true; }
          break;
        case 'row': {
          const n = I.byLayer[I.lix(t.layer)]?.[t.i];
          if (n) { out.nodes.add(n.id); into(n.id); }
          break;
        }
        case 'col': {
          const l = I.lix(t.layer), src = I.byLayer[I.lix(t.k)]?.[t.j];
          if (!src) break;
          out.nodes.add(src.id);
          for (const e of net.edges) {
            const to = I.nodeById.get(e.to);
            if (e.from === src.id && to && I.li.get(to.layer) === l) { out.relE.add(e.id); out.rel.add(e.to); }
          }
          break;
        }
        case 'token': {
          const l = I.lix(t.layer), s = I.shape[l];
          if (!s || !Number.isInteger(t.t) || t.t < 0 || t.t >= s.T) break;
          const hd = Number.isInteger(t.h) ? t.h : null, a = I.att[l], b = I.att[l + 1];
          const dh = hd === null ? 0 : a ? a.dh : b ? b.dh : 0;
          for (let g = 0; g < s.G; g++) {
            if (Number.isInteger(t.g) && t.g !== g) continue;
            for (let f = 0; f < s.d; f++) {
              if (dh && Math.floor(f / dh) !== hd) continue;
              const n = I.tokNode(l, g, t.t, f);
              if (n) { out.nodes.add(n.id); touch(n.id); }
            }
          }
          break;
        }
        default: break;
      }
      out.any = !!(out.nodes.size || out.edges.size || out.rel.size);
      return out;
    }
    // state.anim: the lit neuron(s), edges (weight edge ids, or attention keys 'l:i:j:f') and related.
    function resolveAnim(I, a) {
      const out = { lit: new Set(), litE: new Set(), rel: new Set(), dir: 'fwd', att: null, any: false };
      if (!a) return out;
      const l = I.lix(a.l), n = I.byLayer[l]?.[a.i];
      if (!n) return out;
      out.any = true;
      out.dir = a.dir === 'bwd' ? 'bwd' : 'fwd';
      const bwd = out.dir === 'bwd', ph = bwd ? 'bwd' : String(a.phase ?? '');
      const p = I.pos(n.id), at = p && I.att[p.l];
      const outOf = id => { for (const e of store.net.edges) if (e.from === id) { out.litE.add(e.id); out.rel.add(e.to); } };
      if (at) {
        const t = p.t;
        out.att = { l: p.l, t, ph };
        for (let f = 0; f < at.d; f++) { const z = I.tokNode(p.l, 0, t, f); if (z) out.lit.add(z.id); }
        if ((ph === 'scores' || ph === 'bwd') && at.qG >= 0 && at.kG >= 0) {
          for (let f = 0; f < at.d; f++) {
            const q = I.tokNode(p.l - 1, at.qG, t, f);
            if (q) out.rel.add(q.id);
            for (let j = 0; j < at.T; j++) if (!(at.causal && j > t)) { const k = I.tokNode(p.l - 1, at.kG, j, f); if (k) out.rel.add(k.id); }
          }
        }
        if (ph === 'sum' || ph === 'bwd' || ph === 'softmax') {
          for (let j = 0; j < at.T; j++) for (let f = 0; f < at.d; f++) {
            if (at.causal && j > t) continue;
            out.litE.add(`${p.l}:${t}:${j}:${f}`);
            if (ph !== 'softmax') { const v = I.tokNode(p.l - 1, at.vG, j, f); if (v) out.rel.add(v.id); }
          }
        }
        if (bwd) for (const id of out.lit) outOf(id);
        return out;
      }
      out.lit.add(n.id);
      if (bwd) outOf(n.id);
      else for (const e of store.net.edges) if (e.to === n.id) { out.litE.add(e.id); out.rel.add(e.from); }
      return out;
    }
    // The attention rows (or columns) to show, as the canvas does: hover, then the step-through,
    // then a followed token, then the selected neuron. [{ l, dir: 'row' | 'col', t, h, f, hover }]
    function attFoci(I, E, A) {
      const st = store.state, hv = st.hover;
      const ofNode = id => {
        const p = I.pos(id);
        if (!p) return null;
        const a = I.att[p.l], b = I.att[p.l + 1];
        if (a) return { l: p.l, dir: 'row', t: p.t, h: Math.floor(p.f / a.dh), f: null };
        if (b && p.g === b.vG) return { l: p.l + 1, dir: 'col', t: p.t, h: Math.floor(p.f / b.dh), f: p.f };
        return null;
      };
      const ofToken = x => {
        const l = I.lix(x.layer);
        if (l === undefined || !Number.isInteger(x.t)) return null;
        const h = Number.isInteger(x.h) ? x.h : null, b = I.att[l + 1];
        if (I.att[l]) return { l, dir: 'row', t: x.t, h, f: null };
        if (!b) return null;
        return { l: l + 1, dir: Number.isInteger(x.g) && x.g === b.qG ? 'row' : 'col', t: x.t, h, f: null };
      };
      const hn = hv?.kind === 'node' ? hv.id : hv?.kind === 'row' ? I.byLayer[I.lix(hv.layer)]?.[hv.i]?.id : null;
      const fh = hv?.kind === 'token' ? ofToken(hv) : hn && ofNode(hn);
      if (fh) return [{ ...fh, hover: true }];
      if (A.att && A.att.ph !== 'scores') return [{ l: A.att.l, dir: 'row', t: A.att.t, h: null, f: null }];
      const t = E?.lens.token, h = E?.lens.head;
      if (Number.isInteger(t)) {
        const out = [];
        I.att.forEach((a, l) => { if (a && t < a.T) out.push({ l, dir: 'row', t, h: Number.isInteger(h) && h < a.heads ? h : null, f: null, lens: true }); });
        if (out.length) return out;
      }
      const fs = st.sel?.kind === 'node' && ofNode(st.sel.id);
      return fs ? [fs] : [];
    }
    const lensOf = () => {
      const lens = store.state.lens;
      if (!lens) return null;
      try { return emphasis(store.net, store.state.fwd, lens); } catch (err) { console.error('[nn/view3d] lens:', err); return null; }
    };

    // ================================================================ mode: stack
    function buildStack() {
      const net = store.net, C = newContent(), L = net.layers.length;
      let I = index(net);
      if (!net.nodes.length) return { ...C, empty: 'An empty net: add neurons on the canvas (D goes back to it)' };
      // Sheets in y-z planes, one per layer along x. Features run down (y) as in the canvas's
      // columns, groups (Q, K, V) stacked; tokens go into depth (z), token 1 in front, so a
      // tokenwise layer keeps to its token's plane and only attention edges cross between planes.
      // A dense layer is a column, wrapped into a grid (columns in depth) past 4 neurons.
      const GAP_G = 0.9, ZT = 2, ZD = 1.7;
      const sheets = net.layers.map((lay, l) => {
        const ns = I.byLayer[l], s = I.shape[l], cells = new Map();
        let groups = [], T = 0;
        if (s) {
          const Hh = s.G * s.d + (s.G - 1) * GAP_G;
          ns.forEach((n, k) => {
            const g = Math.floor(k / (s.T * s.d)), t = Math.floor(k / s.d) % s.T, f = k % s.d;
            cells.set(n.id, { y: (Hh - 1) / 2 - (g * (s.d + GAP_G) + f), z: ((s.T - 1) / 2 - t) * ZT });
          });
          groups = (s.groups || []).map((name, g) => ({ name, ya: (Hh - 1) / 2 - g * (s.d + GAP_G), yb: (Hh - 1) / 2 - (g * (s.d + GAP_G) + s.d - 1) }));
          T = s.T;
        } else {
          const n = ns.length, c = n <= 4 ? 1 : Math.ceil(Math.sqrt(n / 2)), r = Math.max(1, Math.ceil(n / c));
          ns.forEach((nd, k) => cells.set(nd.id, { y: (r - 1) / 2 - Math.floor(k / c), z: ((c - 1) / 2 - (k % c)) * ZD }));
        }
        const ys = [...cells.values()].map(c => c.y), zs = [...cells.values()].map(c => c.z);
        const b = ys.length ? { y0: Math.min(...ys), y1: Math.max(...ys), z0: Math.min(...zs), z1: Math.max(...zs) } : { y0: 0, y1: 0, z0: 0, z1: 0 };
        return { cells, groups, T, tok: s, ...b, lid: b.y1 };
      });
      let x = 0;
      sheets.forEach((s, l) => {
        if (l) {
          const p = sheets[l - 1];
          x += 3.3 + 0.08 * Math.max(s.y1 - s.y0, p.y1 - p.y0) + 0.3 * Math.max(s.z1 - s.z0, p.z1 - p.z0);
        }
        s.x = x;
      });
      const pos = new Map();
      sheets.forEach(s => { for (const [id, c] of s.cells) pos.set(id, V3(s.x, c.y, c.z)); });

      // -- a faint plate per layer, shown only while the layer is hovered or selected (as the
      //    canvas's lanes): at rest the neurons' columns are the structure
      sheets.forEach(s => {
        if (!s.cells.size) return;
        const o = new THREE.Mesh(C.plateGeo(s.z1 - s.z0 + 1, s.y1 - s.y0 + 1, 0.35), MAT.band);
        o.position.set(s.x - 0.01, (s.y0 + s.y1) / 2, (s.z0 + s.z1) / 2);
        o.quaternion.copy(ROT_YZ);
        o.renderOrder = -2;
        o.visible = false;
        o.updateMatrix();
        o.matrixAutoUpdate = false;
        C.group.add(o);
        s.plate = o;
      });

      // -- neurons
      const nodeMesh = C.inst(net.nodes.length > 300 ? GEO.sphereLo : GEO.sphere, MAT.lit, net.nodes.length);
      net.nodes.forEach((n, i) => setBox(nodeMesh, i, pos.get(n.id) || V3(), pos.has(n.id) ? NODE_R : 0));
      flushInst(nodeMesh);

      // -- weight edges: straight, or arcing over the sheets they skip; fixed ones dashed
      const edgeRecs = [];
      for (const e of net.edges) {
        const a = pos.get(e.from), b = pos.get(e.to);
        if (!a || !b) continue;
        const la = I.li.get(I.nodeById.get(e.from).layer), lb = I.li.get(I.nodeById.get(e.to).layer);
        let pts;
        if (Math.abs(lb - la) >= 2) {
          let apex = Math.max(a.y, b.y) + 0.9;
          for (let l = Math.min(la, lb) + 1; l < Math.max(la, lb); l++) apex = Math.max(apex, sheets[l].y1 + 1.1);
          for (let l = Math.min(la, lb) + 1; l < Math.max(la, lb); l++) sheets[l].lid = Math.max(sheets[l].lid, apex);
          const mid = V3().addVectors(a, b).multiplyScalar(0.5), ctl = V3(mid.x, 2 * apex - mid.y, mid.z);
          pts = [];
          for (let k = 0; k <= 16; k++) {
            const u = k / 16, w = 1 - u;
            pts.push(V3(w * w * a.x + 2 * w * u * ctl.x + u * u * b.x, w * w * a.y + 2 * w * u * ctl.y + u * u * b.y, w * w * a.z + 2 * w * u * ctl.z + u * u * b.z));
          }
        } else pts = [a.clone(), b.clone()];
        // trim to the spheres' surfaces
        const trim = (p, q) => p.clone().addScaledVector(V3().subVectors(q, p).normalize(), NODE_R);
        pts[0] = trim(pts[0], pts[1]);
        pts[pts.length - 1] = trim(pts[pts.length - 1], pts[pts.length - 2]);
        const segs = [];
        if (e.fixed) {   // dashes: every other piece of about 0.3
          const dense = [];
          for (let k = 0; k + 1 < pts.length; k++) {
            const n = Math.max(1, Math.round(pts[k].distanceTo(pts[k + 1]) / 0.3));
            for (let q = 0; q < n; q++) dense.push(V3().lerpVectors(pts[k], pts[k + 1], q / n));
          }
          dense.push(pts[pts.length - 1]);
          for (let k = 0; k + 1 < dense.length; k += 2) segs.push([dense[k], dense[k + 1]]);
        } else for (let k = 0; k + 1 < pts.length; k++) segs.push([pts[k], pts[k + 1]]);
        edgeRecs.push({ id: e.id, pts, segs, first: 0 });
      }
      let nSeg = 0;
      for (const r of edgeRecs) { r.first = nSeg; nSeg += r.segs.length; }
      const edgeLines = C.lines(nSeg);
      for (const r of edgeRecs) r.segs.forEach((s, k) => edgeLines.seg(r.first + k, s[0], s[1]));
      // At rest weights are faint, and fainter and thinner the more there are; hovering or selecting
      // brings a neuron's (or a layer's) edges forward at their full colorFor strength.
      const crowd = 30 / Math.max(30, edgeRecs.filter(r => !I.edgeById.get(r.id)?.fixed).length);
      const K0 = clamp(0.62 * crowd ** 0.25, 0.16, 0.62), WS = clamp(crowd ** 0.2, 0.5, 1), KA = 0.75;

      // -- attention edges V_j,f -> Z_i,f, and the n × n tile of A (bars) in front of each attention sheet
      const attRecs = [], tiles = [];
      I.att.forEach((a, l) => {
        if (!a) return;
        for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) for (let f = 0; f < a.d; f++) {
          if (a.causal && j > i) continue;
          const v = I.tokNode(l - 1, a.vG, j, f), z = I.tokNode(l, 0, i, f);
          const pv = v && pos.get(v.id), pz = z && pos.get(z.id);
          if (!pv || !pz) continue;
          const dir = V3().subVectors(pz, pv).normalize();
          attRecs.push({ key: `${l}:${i}:${j}:${f}`, l, i, j, f, h: Math.floor(f / a.dh), pts: [pv.clone().addScaledVector(dir, NODE_R), pz.clone().addScaledVector(dir, -NODE_R)] });
        }
        const s = sheets[l], cell = 0.42, gap = 0.35, wAll = a.heads * a.T * cell + (a.heads - 1) * gap;
        const yTop = s.y0 - 1.15, zF = s.z1 + 0.5;
        for (let h = 0; h < a.heads; h++) for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) {
          tiles.push({ l, h, i, j, cell, masked: a.causal && j > i, z: zF,
            c: V3(s.x - wAll / 2 + h * (a.T * cell + gap) + (j + 0.5) * cell, yTop - (i + 0.5) * cell, zF) });
        }
        s.tile = { x0: s.x - wAll / 2, y: yTop - (a.T * cell) / 2, z: zF, yb: yTop - a.T * cell };
      });
      const attLines = C.lines(attRecs.length);
      attRecs.forEach((r, k) => attLines.seg(k, r.pts[0], r.pts[1]));
      const attByKey = new Map(attRecs.map(r => [r.key, r]));
      const tileMesh = C.inst(GEO.bar, MAT.lit, tiles.length);

      // -- highlight rings and step-through pulses
      const ringMesh = C.inst(GEO.ring, MAT.ring, 64, 3), ringThin = C.inst(GEO.ringThin, MAT.ring, 128, 3);
      const pulseMesh = C.inst(GEO.ball, MAT.ring, 256, 4);
      let rings = [], thin = [], pulses = [];

      // -- labels: headers, groups, token names, values, A_ij, the hover tip
      const headSub = (lay, l) => {
        const s = I.shape[l], a = I.att[l], n = I.byLayer[l].length, act = M.ACTS?.[lay.act]?.label || lay.act || 'identity';
        if (a) return `${a.causal ? 'causal ' : ''}attention · ${a.T} × ${a.d}${a.heads > 1 ? ` · ${a.heads} heads` : ''}`;
        if (s) return `${l ? `${act} · ` : ''}${s.T} × ${s.d}${s.groups ? ' each' : ''}`;
        return l === 0 ? `${n} input${n === 1 ? '' : 's'}` : `${act} · ${n}`;
      };
      // One label per layer: its name. The second line (activation, shape) opens while the layer or
      // one of its neurons is hovered, selected, stepped or the lens's focus.
      sheets.forEach((s, l) => {
        const lay = net.layers[l], name = lay.name || (l === 0 ? 'Input' : l === L - 1 ? 'Output' : 'Hidden');
        const o = C.label('nn3d-head', `<b>${esc(name)}</b><span>${esc(headSub(lay, l))}</span>`, V3(s.x, s.lid + 0.75, s.z0), 0.5, 1);
        o.userData.layer = l;
        o.userData.fit = true;
        s.headLab = o;
        for (const g of s.groups) C.label('nn3d-grp', `<i>${esc(g.name)}</i>`, V3(s.x, (g.ya + g.yb) / 2, s.z1 + 0.6), 1, 0.5);
      });
      const firstTok = sheets.findIndex(s => s.T > 1), tokLabs = [];
      if (firstTok >= 0) {
        const s = sheets[firstTok], names = tokenNames(net);
        for (let t = 0; t < s.T; t++) {
          tokLabs.push(C.label(`nn3d-tok${names[t] ? ' name' : ''}`, names[t] ? esc(names[t].length > 10 ? `${names[t].slice(0, 9)}…` : names[t]) : sub('t', t + 1), V3(s.x, s.y0 - 0.62, ((s.T - 1) / 2 - t) * ZT), 0.5, 0));
        }
      }
      const valLabs = new Map();
      if (net.nodes.length <= 160) for (const n of net.nodes) { const p = pos.get(n.id); if (p) valLabs.set(n.id, C.label('nn3d-val', '', V3(p.x, p.y - NODE_R - 0.05, p.z), 0.5, 0)); }
      const aLabs = new Map();
      I.att.forEach((a, l) => {
        if (!a) return;
        for (let h = 0; h < a.heads; h++) for (let i = 0; i < a.T; i++) for (let j = 0; j < a.T; j++) {
          const o = C.label('nn3d-al', '', V3(), 0.5, 0.5);
          o.visible = false;
          aLabs.set(`${l}:${h}:${i}:${j}`, o);
        }
      });
      const tip = C.label('nn3d-tip', '', V3(), 0, 1);
      tip.visible = false;
      let tipKey = '';

      const act = n => { const v = store.state.fwd?.node?.[n.id]?.a; return isNum(v) ? v : I.li.get(n.layer) === 0 && isNum(n.value) ? n.value : NaN; };
      const attA = (l, h, i, j) => { const v = store.state.fwd?.attn?.[l]?.heads?.[h]?.A?.[i]?.[j]; return isNum(v) ? v : NaN; };

      const bgMix = (out, c, t) => { out[0] = P.bg[0] + (c[0] - P.bg[0]) * t; out[1] = P.bg[1] + (c[1] - P.bg[1]) * t; out[2] = P.bg[2] + (c[2] - P.bg[2]) * t; return out; };
      const hiMix = (out, t) => { for (let k = 0; k < 3; k++) out[k] += (P.hi[k] - out[k]) * t; return out; };
      const openL = new Set();   // layers whose header shows its second line
      function paint() {
        if (stale(I, net)) I = index(net);
        const st = store.state, fwd = st.fwd, v3 = cur();
        const E = lensOf(), dim = !!E?.any, hides = !!E?.hides;
        const H = resolve(I, st.hover), S = resolve(I, st.sel), A = resolveAnim(I, st.anim);
        const foci = attFoci(I, E, A);
        const focusOn = H.any || A.any || !!foci[0]?.hover;
        let maxW = 0;
        for (const e of net.edges) if (isNum(e.w)) maxW = Math.max(maxW, Math.abs(e.w));
        maxW ||= 1;
        const maxA = (fwd ? maxAbs(fwd.a, fwd.z) : maxAbs(net.nodes.map(act))) || 1;
        const marked = id => H.nodes.has(id) || S.nodes.has(id) || A.lit.has(id) || (focusOn && (H.rel.has(id) || A.rel.has(id)));
        // neurons
        rings = []; thin = [];
        net.nodes.forEach((n, i) => {
          const p = pos.get(n.id);
          if (!p) return;
          const a = act(n), m = marked(n.id), e = dim ? E.node(n.id) : 1;
          const op = m ? 1 : le(e) * (focusOn ? FOCUS_NODE : 1);
          setCol(nodeMesh, i, mix(P.bg, over(a, maxA, P.node), op));
          const lab = valLabs.get(n.id);
          if (lab) {
            lab.visible = !!v3?.nums && (m || e >= NUM_MIN);
            const s = num(a);
            if (lab.element.__t !== s) { lab.element.__t = s; lab.element.textContent = s; }
            lab.element.style.opacity = op < 0.99 ? String(Math.max(0.3, op)) : '';
          }
          if (S.nodes.has(n.id)) rings.push({ p, r: NODE_R + 0.12, k: 'sel' });
          else if (A.lit.has(n.id)) rings.push({ p, r: NODE_R + 0.12, k: 'lit' });
          else if (H.nodes.has(n.id)) thin.push({ p, r: NODE_R + 0.1 });
        });
        flushInst(nodeMesh);
        // weight edges: faint at rest (K0, WS), their colorFor strength and a wider line when they
        // touch the hovered or selected neuron or layer; the held edge and its tie group get HI
        pulses = [];
        const c4 = [0, 0, 0, 0], rgb = [0, 0, 0];
        for (const r of edgeRecs) {
          const e = I.edgeById.get(r.id);
          if (!e) continue;
          const w = isNum(e.w) ? e.w : 0, u = Math.min(1, Math.abs(w) / maxW);
          const gone = hides && E.hidden.edge(e.id);
          const on = H.edges.has(e.id) || S.edges.has(e.id), tie = H.ties.has(e.id) || S.ties.has(e.id);
          const lit = A.litE.has(e.id), held = on || tie || lit;
          const fwdE = held || (focusOn && H.relE.has(e.id)) || S.relE.has(e.id);
          const ev = dim ? E.edge(e.id) : 1;
          let s, px;
          if (e.fixed) {   // not a parameter: muted text colour, thin, dashed
            c4[0] = P.fixed[0]; c4[1] = P.fixed[1]; c4[2] = P.fixed[2];
            s = held ? 1 : fwdE ? 0.7 : FIXED_A;
            px = held ? 1.6 : 1;
          } else {
            cf(w, maxW, c4);
            s = c4[3] * (fwdE ? 1 : K0);
            px = fwdE ? 0.9 + 2.6 * u : (0.55 + 2.2 * u) * WS;
          }
          if (!held) s *= le(ev) * (focusOn && !fwdE ? FOCUS_EDGE : 1);
          bgMix(rgb, c4, s);
          if (held) { hiMix(rgb, on ? 0.55 : 0.35); px += on ? 1.2 : 0.6; }
          for (let k = 0; k < r.segs.length; k++) edgeLines.set(r.first + k, rgb, gone ? 0 : px);
          if (lit && !gone) pulses.push({ pts: r.pts, rev: A.dir === 'bwd' });
        }
        edgeLines.flush();
        // attention edges: strength and width from A_ij, faint at rest, full on a shown row
        const rowOf = new Map();   // attention keys the shown rows light
        for (const F of foci) {
          for (const r of attRecs) {
            if (r.l !== F.l) continue;
            if ((F.dir === 'row' ? r.i : r.j) !== F.t || (F.h != null && r.h !== F.h) || (F.f != null && r.f !== F.f)) continue;
            rowOf.set(r.key, F);
          }
        }
        const attHi = mix(P.att, P.hi, 0.3);
        attRecs.forEach((r, k) => {
          const A0 = attA(r.l, r.h, r.i, r.j), u = isNum(A0) ? clamp(A0, 0, 1) : 0;
          const gone = hides && E.hidden.attn(r.l, r.i, r.j, r.h);
          const lit = A.litE.has(r.key), rel = rowOf.has(r.key) && !rowOf.get(r.key).lens;
          const ev = dim ? E.attn(r.l, r.i, r.j, r.h) : 1;
          let s = (isNum(A0) ? 0.06 + 0.88 * u : 0.12) * (lit || rel ? 1 : KA);
          if (!(lit || rel)) s *= le(ev) * (focusOn ? FOCUS_ATT : 1);
          bgMix(rgb, lit ? attHi : P.att, s);
          attLines.set(k, rgb, gone ? 0 : lit || rel ? 1 + 3 * u + (lit ? 0.6 : 0) : (0.55 + 2.4 * u) * WS);
          if (lit && !gone && A.att?.ph !== 'softmax') pulses.push({ pts: r.pts, rev: A.dir === 'bwd' });
        });
        attLines.flush();
        // A tiles (per head T × T): bars as tall as A_ij over a faint neutral cell, the shown row in HI
        const cellBase = mix(P.bg, P.node, 0.45);
        tiles.forEach((c, k) => {
          const v = attA(c.l, c.h, c.i, c.j), u = isNum(v) ? clamp(v, 0, 1) : 0;
          const rows = dim ? E.rows(c.l) : null, hs = dim ? E.heads(c.l) : null;
          const keep = (!rows?.size || rows.has(c.i)) && (!hs || hs.has(c.h));
          const shown = foci.some(F => F.l === c.l && (F.h == null || F.h === c.h) && (F.dir === 'row' ? c.i === F.t : c.j === F.t));
          const depth = c.masked ? 0.02 : 0.05 + 0.6 * u;
          tp.set(c.c.x, c.c.y, c.c.z + depth / 2);
          setBox(tileMesh, k, tp, c.cell * 0.86, c.cell * 0.86, depth);
          const base = c.masked ? mix(P.bg, P.node, 0.2) : mix(cellBase, P.att, 0.1 + 0.9 * u);
          setCol(tileMesh, k, mix(P.bg, shown ? mix(base, P.hi, 0.45) : base, keep ? 1 : 0.25));
        });
        flushInst(tileMesh);
        // A_ij numbers on the shown rows (not on a followed token's: the lens does the dimming)
        for (const o of aLabs.values()) o.visible = false;
        for (const F of foci) {
          if (F.lens) continue;
          const a = I.att[F.l];
          for (let h = 0; h < a.heads; h++) for (let o2 = 0; o2 < a.T; o2++) {
            if (F.h != null && h !== F.h) continue;
            const i = F.dir === 'row' ? F.t : o2, j = F.dir === 'row' ? o2 : F.t;
            if (a.causal && j > i) continue;
            if (hides && E.hidden.attn(F.l, i, j, h)) continue;
            const r = attByKey.get(`${F.l}:${i}:${j}:${(h + 1) * a.dh - 1}`), o = aLabs.get(`${F.l}:${h}:${i}:${j}`);
            if (!r || !o) continue;
            o.position.lerpVectors(r.pts[0], r.pts[1], F.dir === 'row' ? 0.3 : 0.72);
            const s = num(attA(F.l, h, i, j));
            if (o.element.__t !== s) { o.element.__t = s; o.element.textContent = s; }
            o.visible = true;
          }
        }
        // hover tip: the hovered (else selected) neuron's label and value
        const tipId = [...H.nodes][0] || (st.sel?.kind === 'node' ? st.sel.id : null);
        const tn = tipId && I.nodeById.get(tipId);
        if (tn && pos.get(tipId)) {
          const key = `${tipId}|${tn.label}|${num(act(tn))}`;
          if (key !== tipKey) {
            tipKey = key;
            tip.element.innerHTML = `${katexHtml(tn.label || '')}<span>= ${num(act(tn))}</span>`;
          }
          tip.position.copy(pos.get(tipId)).add(V3(0, NODE_R + 0.1, 0));
          tip.visible = true;
        } else tip.visible = false;
        // headers: dim with their layer; the second line opens for a held layer (the layer itself,
        // or a neuron of it hovered, selected or stepped) or the lens's focus, and its plate shows
        const open = new Set([H.layer, S.layer].filter(Number.isInteger));
        for (const id of [...H.nodes, ...S.nodes, ...A.lit]) { const n = I.nodeById.get(id); if (n) open.add(I.li.get(n.layer)); }
        for (const s of sheets) {
          if (!s.headLab) continue;
          const l = s.headLab.userData.layer, el = s.headLab.element;
          const lv = dim ? E.layer(l) : 1, foc = E?.lens.focus?.layer === net.layers[l].id, hov = H.layer === l || S.layer === l;
          el.style.opacity = lv < 0.995 ? String(le(lv)) : '';
          el.classList.toggle('foc', !!foc);
          el.classList.toggle('hov', hov);
          const wide = open.has(l) || !!foc;
          if (wide !== openL.has(l)) { if (wide) openL.add(l); else openL.delete(l); el.classList.toggle('open', wide); relabel = true; }
          if (s.plate) s.plate.visible = hov || !!foc;
        }
        placeRings();
      }
      function placeRings() {
        const pulse = 1 + 0.16 * Math.sin(performance.now() / 175);
        ringMesh.count = Math.min(64, rings.length);
        rings.slice(0, 64).forEach((r, i) => setBox(ringMesh, i, r.p, r.r * (r.k === 'lit' ? pulse : 1), r.r * (r.k === 'lit' ? pulse : 1), r.r, camera.quaternion));
        ringThin.count = Math.min(128, thin.length);
        thin.slice(0, 128).forEach((r, i) => setBox(ringThin, i, r.p, r.r, r.r, r.r, camera.quaternion));
        flushInst(ringMesh);
        flushInst(ringThin);
      }
      const pathAt = (pts, u) => {
        const n = pts.length - 1, x = clamp(u, 0, 1) * n, k = Math.min(n - 1, Math.floor(x));
        return tp.lerpVectors(pts[k], pts[k + 1], x - k);
      };
      function frame(t) {
        let busy = false;
        if (rings.length || thin.length) { placeRings(); busy = rings.some(r => r.k === 'lit'); }
        pulseMesh.count = Math.min(256, pulses.length);
        if (pulses.length) {
          const u = (t / 1100) % 1, e = u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
          pulses.slice(0, 256).forEach((q, i) => setBox(pulseMesh, i, pathAt(q.pts, q.rev ? 1 - e : e), 0.07));
          flushInst(pulseMesh);
          busy = true;
        }
        return busy;
      }
      // Screen-space picking: a neuron under the cursor, else an A tile cell, else the nearest edge.
      function pick(px, py, proj) {
        if (stale(I, net)) I = index(net);
        let best = null;
        for (const n of net.nodes) {
          const p = pos.get(n.id);
          if (!p) continue;
          const q = proj(p, NODE_R);
          if (q && Math.hypot(q.x - px, q.y - py) < q.r + 3 && (!best || q.z < best.z)) best = { z: q.z, spec: { kind: 'node', id: n.id } };
        }
        if (best) return best.spec;
        for (const c of tiles) {
          const q = proj(c.c, c.cell / 2);
          if (q && Math.abs(q.x - px) < q.r && Math.abs(q.y - py) < q.r) {
            return { kind: 'token', layer: c.l, t: c.i, ...(I.att[c.l].heads > 1 ? { h: c.h } : {}), cell: true };
          }
        }
        let bd = 6, hit = null;
        const near = (pts, spec) => {
          let prev = null;
          for (const p of pts) {
            const q = proj(p, 0);
            if (q && prev) {
              const d = segDist(px, py, prev, q);
              if (d < bd) { bd = d; hit = spec; }
            }
            prev = q;
          }
        };
        const E = lensOf();
        for (const r of edgeRecs) if (!(E?.hides && E.hidden.edge(r.id))) near(r.pts, { kind: 'edge', id: r.id });
        for (const r of attRecs) if (!(E?.hides && E.hidden.attn(r.l, r.i, r.j, r.h))) near(r.pts, { kind: 'token', layer: r.l, t: r.i, ...(I.att[r.l].heads > 1 ? { h: r.h } : {}), att: true });
        return hit;
      }
      // What the camera frames: the neurons, the A tiles and a little room over each header (the
      // headers themselves are fitted at their size on screen: userData.fit).
      const points = () => {
        const pts = [...pos.values()];
        for (const s of sheets) if (s.headLab) { const h = s.headLab.position; pts.push(V3(h.x, h.y + 0.4, h.z)); }
        for (const c of tiles) pts.push(c.c);
        return pts;
      };
      C.ground(new THREE.Box3().setFromPoints([...pos.values(), ...tiles.map(c => c.c)]));
      // At rest, headers that touch another on screen drop their second line, then each takes the
      // lowest of three rows (lifted LIFT px apart) where it touches none placed before it, left to
      // right; one that fits in none hides (a deep net in a small window). A held header (open,
      // hovered, the lens's focus) keeps its second line and goes in on top of that rest layout,
      // hiding only the headers it touches, so hovering never reshuffles the others. The rest
      // layout is worked out again only when the camera has moved. Token names that would touch
      // the one before them hide. Measured after a render, when the camera has moved or a header
      // opened or closed.
      const LIFT = 30;
      let restAt = -1;   // camMoves when the rest layout was last worked out
      function declutter() {
        const hs = sheets.filter(s => s.headLab?.visible).map(s => s.headLab.element);
        const held = e => e.classList.contains('open') || e.classList.contains('hov') || e.classList.contains('foc');
        const touch = (a, b) => a.right + 6 > b.left && b.right + 6 > a.left && a.bottom > b.top && b.bottom > a.top;
        const lift = (r, k) => ({ left: r.left, right: r.right, top: r.top - k * LIFT, bottom: r.bottom - k * LIFT });
        const setRow = (e, k) => { e.classList.toggle('up', k === 1); e.classList.toggle('up2', k === 2); e.classList.toggle('gone', k < 0); };
        const H = hs.filter(held), R = hs.filter(e => !held(e));
        if (!H.length || restAt !== camMoves) {
          restAt = camMoves;
          for (const e of R) { e.classList.remove('compact'); setRow(e, 0); }
          let rs = R.map(e => ({ e, r: e.getBoundingClientRect() })).filter(x => x.r.width);
          if (rs.some(a => rs.some(b => b !== a && touch(a.r, b.r)))) {
            for (const a of rs) if (rs.some(b => b !== a && touch(a.r, b.r))) a.e.classList.add('compact');
            rs = rs.map(x => ({ e: x.e, r: x.e.getBoundingClientRect() })).sort((a, b) => a.r.left - b.r.left);
            const placed = [];
            for (const x of rs) {
              let k = 0;
              while (k < 3 && placed.some(p => touch(lift(x.r, k), p))) k++;
              x.e.__row = k < 3 ? k : -1;
              if (k < 3) placed.push(lift(x.r, k));
            }
          } else for (const x of rs) x.e.__row = 0;
        }
        for (const e of R) setRow(e, e.__row ?? 0);
        if (H.length) {
          const shown = R.filter(e => !e.classList.contains('gone')).map(e => ({ e, r: e.getBoundingClientRect() }));
          const placed = [];
          for (const e of H.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left)) {
            e.classList.remove('compact');
            setRow(e, 0);
            // the row (clear of other held headers) where it hides the fewest others
            const r0 = e.getBoundingClientRect();
            let k = -1, cost = Infinity;
            for (let j = 0; j < 3; j++) {
              if (placed.some(p => touch(lift(r0, j), p))) continue;
              const c = shown.filter(x => touch(lift(r0, j), x.r)).length;
              if (c < cost) { cost = c; k = j; }
            }
            if (k < 0) k = 0;
            setRow(e, k);
            const r = lift(r0, k);
            placed.push(r);
            for (const x of shown) if (touch(r, x.r)) x.e.classList.add('gone');
          }
        }
        let last = null;
        for (const o of tokLabs) {
          const r = o.element.getBoundingClientRect(), hit = last && r.width && r.left < last.right + 3 && last.left < r.right + 3 && r.top < last.bottom && last.top < r.bottom;
          o.element.style.visibility = hit ? 'hidden' : '';
          if (!hit && r.width) last = r;
        }
      }
      return {
        ...C, paint, frame, pick, points, declutter, dir: DIRS.stack,
        nodePos: id => pos.get(id) || null, nodeR: NODE_R,
        click(spec) {
          if (!spec) return null;
          if (spec.kind === 'token' && spec.cell) return { kind: 'layer', id: net.layers[spec.layer].id };
          if (spec.kind === 'token') return null;
          return spec;
        },
      };
    }
    const segDist = (px, py, a, b) => {
      const dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy;
      const u = L2 ? clamp(((px - a.x) * dx + (py - a.y) * dy) / L2, 0, 1) : 0;
      return Math.hypot(px - a.x - u * dx, py - a.y - u * dy);
    };

    // ================================================================ cells (heads and tensor modes)
    // A cell's number shows on a held cell (hovered, selected, stepped) always, and with the 1.2
    // button on, on every cell the lens keeps: at full size once its face is NUM_PX wide on screen,
    // below that without its leading 0 and shrunk to the face (7.5 px at the least), so they don't
    // overlap. paint / draw set the label's userData { txt, nums (the button), on (held), want (not
    // faded by the lens), no (never: a masked bar, token colours) }; declutter measures face (px)
    // as the camera moves. showNum -> true if the visibility changed.
    // The face's drawn width: its left and right edges (cells are axis-aligned) projected, so a face
    // seen at an angle counts as narrower than one seen head on.
    const faceL = V3(), faceR = V3();
    const faceW = (p, half) => {
      const a = proj(faceL.set(p.x - half, p.y, p.z), 0), b = proj(faceR.set(p.x + half, p.y, p.z), 0);
      return a && b ? Math.hypot(b.x - a.x, b.y - a.y) : 0;
    };
    function showNum(o) {
      const d = o.userData, vis = !d.no && (!!d.on || (!!d.nums && !!d.want && !d.hid));
      const small = vis && !d.on && d.face != null && d.face < NUM_PX;
      const txt = small ? String(d.txt ?? '').replace(/^(−?)0\./, '$1.') : String(d.txt ?? '');
      if (o.element.__t !== txt) { o.element.__t = txt; o.element.textContent = txt; }
      const fs = small ? `${clamp(Math.round(23 * d.face / 22) / 2, 7.5, 11.5)}px` : '';
      if (o.element.__fs !== fs) { o.element.__fs = fs; o.element.style.fontSize = fs; }
      if (o.visible === vis) return false;
      o.visible = vis;
      return true;
    }
    // A number whose cell sits behind a nearer cell on screen (a slab behind a slab) hides, so
    // only the numbers of the faces you see show: labels ignore depth. Run after the faces are
    // measured; the held cell always shows its number. -> true if any visibility changed.
    function occlude(labs) {
      const qs = [];
      for (const o of labs) {
        const d = o.userData;
        d.hid = false;
        if (d.no || !d.nums || !d.want || !d.face) continue;
        const q = proj(o.position, 0);
        if (q) qs.push({ o, x: q.x, y: q.y, z: q.z, h: d.face * 0.45 });
      }
      qs.sort((a, b) => a.z - b.z);
      for (let i = 1; i < qs.length; i++) {
        const a = qs[i];
        for (let j = 0; j < i; j++) {
          const b = qs[j];
          if (!b.o.userData.hid && b.z < a.z - 1e-6 && Math.abs(a.x - b.x) < b.h && Math.abs(a.y - b.y) < b.h) { a.o.userData.hid = true; break; }
        }
      }
      // then, nearest first, a number whose text would touch one already shown hides too
      const kept = [];
      for (const a of qs) {
        const d = a.o.userData;
        if (d.hid) continue;
        const small = d.face < NUM_PX, fs = small ? clamp(Math.round(23 * d.face / 22) / 2, 7.5, 11.5) : 11.5;
        const txt = small ? String(d.txt ?? '').replace(/^(−?)0\./, '$1.') : String(d.txt ?? '');
        const w = txt.length * fs * 0.27 + 1, h = fs * 0.55;   // half-extents of the drawn text
        if (kept.some(k => Math.abs(a.x - k.x) < w + k.w && Math.abs(a.y - k.y) < h + k.h)) { d.hid = true; continue; }
        kept.push({ x: a.x, y: a.y, w, h });
      }
      let changed = false;
      for (const o of labs) changed = showNum(o) || changed;
      return changed;
    }
    // Value tiles in x-y planes (rows = tokens top to bottom), with a HI frame on lit ones.
    function cellSet(C, n) {
      const mesh = C.inst(GEO.tile, MAT.lit, n), halo = C.inst(GEO.frame, MAT.halo, n, 2);
      halo.count = 0;
      return { mesh, halo };
    }
    // The heads' marks: head colour only on a thin accent (a strip along a slab's left edge, a bar
    // under a chunk), the plates themselves neutral. A head label carries a swatch, not a colour.
    const headSw = h => `<span class="ui-sw" style="background: var(${HEAD_VARS[h % HEAD_VARS.length]})"></span>`;
    // W_O's value from its tied edges, read fresh (it trains): the tie's first edge.
    function woReader(wo, I) {
      const ids = wo ? wo.ties.map(r => r.map(t => (t && I.ties.get(t)?.[0]) || null)) : [];
      return (idx, i, j) => { const id = ids[i]?.[j], e = id && idx().edgeById.get(id); return e && isNum(e.w) ? e.w : wo.W[i][j]; };
    }

    // ================================================================ mode: heads
    function buildHeads(v) {
      const net = store.net, C = newContent(), list = attnLayers(net);
      let I = index(net);
      const idx = () => (stale(I, net) ? (I = index(net)) : I);
      if (!list.length) return { ...C, empty: 'Heads shows an attention layer, and this net has none. Try the "Two heads: max and min" or "Transformer block" preset' };
      const l = list.includes(I.li.get(v.layer)) ? I.li.get(v.layer) : list[0];
      const a = I.att[l], T = a.T, d = a.d, H = a.heads, dh = a.dh, qkv = l - 1;
      const wo = projOf(net, l), dOut = wo ? wo.W[0].length : 0, woAt = woReader(wo, I);
      // Cells in x-y planes (rows = tokens). Q | K | V on top (each head's column chunk marked),
      // split into one slab per head below them, stepping down and back in depth:
      // Q_h K_h -> A_h (bars), A_h V_h = Z_h; then, right of the slabs, concat Z × W_O = Z W_O.
      const GZ = 2.1, SLAB = T + 2.0, ROW = T + 2.1;
      const zMid = -((H - 1) / 2) * GZ;
      const yA = 0, yS = h => -(ROW + h * SLAB), yC = -(ROW + ((H - 1) * SLAB) / 2);
      const zOf = h => -h * GZ;
      const yOf = t => (T - 1) / 2 - t;
      const X = {};
      const row = (list, gap) => {   // [key, width, gap after?] centred on x = 0; returns the width, gaps included
        const gs = list.map(([, , g], i) => (i < list.length - 1 ? g ?? gap : 0));
        const W0 = list.reduce((s, [, w]) => s + w, 0) + gs.reduce((s, g) => s + g, 0);
        let x = -W0 / 2;
        list.forEach(([k, w], i) => { X[k] = x + (w - 1) / 2; x += w + gs[i]; });
        return W0;
      };
      row([['Qf', d], ['Kf', d], ['Vf', d]], 1.1);
      const slabW = row([['Q', dh, 0.6], ['K', dh, 1.2], ['A', T, 1.2], ['V', dh, 0.6], ['Z', dh]], 0);
      const wC = row(wo ? [['Zc', d], ['W', dOut], ['out', dOut]] : [['Zc', d]], 1.5);
      const xC = slabW / 2 + 3.4 + 0.45 * (H - 1) * GZ + wC / 2;   // the concat row sits right of the slabs (and of their depth shift)
      for (const k of ['Zc', 'W', 'out']) if (k in X) X[k] += xC;
      const cells = [];
      const node = (g, t, f) => I.tokNode(qkv, g, t, f)?.id;
      const G3 = [a.qG, a.kG, a.vG];
      ['Qf', 'Kf', 'Vf'].forEach((k, gi) => {
        for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) {
          cells.push({ kind: k, g: gi, t, f, h: Math.floor(f / dh), node: node(G3[gi], t, f), p: V3(X[k] - (d - 1) / 2 + f, yA + yOf(t), zMid) });
        }
      });
      for (let h = 0; h < H; h++) {
        ['Q', 'K', 'V', 'Z'].forEach((k, gi) => {
          for (let t = 0; t < T; t++) for (let e = 0; e < dh; e++) {
            const f = h * dh + e;
            cells.push({ kind: k, g: gi, t, f, e, h, node: gi < 3 ? node(G3[gi], t, f) : I.tokNode(l, 0, t, f)?.id, p: V3(X[k] - (dh - 1) / 2 + e, yS(h) + yOf(t), zOf(h)) });
          }
        });
      }
      for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) cells.push({ kind: 'Zc', t, f, h: Math.floor(f / dh), node: I.tokNode(l, 0, t, f)?.id, p: V3(X.Zc - (d - 1) / 2 + f, yC + yOf(t), zMid) });
      if (wo) {
        for (let i = 0; i < d; i++) for (let j = 0; j < dOut; j++) {
          const tie = wo.ties[i]?.[j], eid = tie && I.ties.get(tie)?.[0];
          cells.push({ kind: 'W', i, j, t: i, f: j, edge: eid || null, p: V3(X.W - (dOut - 1) / 2 + j, yC + (d - 1) / 2 - i, zMid) });
        }
        for (let t = 0; t < T; t++) for (let j = 0; j < dOut; j++) cells.push({ kind: 'out', t, f: j, node: I.tokNode(wo.l, 0, t, j)?.id, p: V3(X.out - (dOut - 1) / 2 + j, yC + yOf(t), zMid) });
      }
      const bars = [];
      for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        bars.push({ h, i, j, masked: a.causal && j > i, p: V3(X.A - (T - 1) / 2 + j, yS(h) + yOf(i), zOf(h)) });
      }
      const { mesh, halo } = cellSet(C, cells.length);
      cells.forEach((c, i) => setBox(mesh, i, c.p, CUBE, CUBE, 1));
      flushInst(mesh);
      const barMesh = C.inst(GEO.bar, MAT.lit, bars.length), barHalo = C.inst(GEO.frame, MAT.halo, bars.length, 2);
      barHalo.count = 0;

      // neutral plates: each head's slab, and one behind each of Q, K, V and concat Z; the head
      // colour on an accent beside a slab and under each head's column chunk
      const x0 = -slabW / 2 - 0.55, x1 = slabW / 2 + 0.55, backZ = TD / 2 + 0.12;
      const slabs = [], chunks = [], accents = [];
      for (let h = 0; h < H; h++) {
        slabs.push({ h, c: V3(0, yS(h), zOf(h) - backZ) });
        accents.push({ h, c: V3(x0 + 0.06, yS(h), zOf(h) - backZ + 0.004), w: 0.08, hh: T + 0.5 });
        const cx = k => X[k] - (d - 1) / 2 + h * dh + (dh - 1) / 2;
        for (const k of ['Qf', 'Kf', 'Vf']) accents.push({ h, c: V3(cx(k), yA - T / 2 - 0.2, zMid), w: dh - 0.12, hh: 0.08 });
        accents.push({ h, c: V3(cx('Zc'), yC + T / 2 + 0.2, zMid), w: dh - 0.12, hh: 0.08 });
      }
      for (const k of ['Qf', 'Kf', 'Vf', 'Zc']) chunks.push({ h: null, c: V3(X[k], k === 'Zc' ? yC : yA, zMid - backZ) });
      const slabMesh = C.inst(C.plateGeo(x1 - x0, T + 0.9, 0.3), MAT.plate, slabs.length, -2);
      slabs.forEach((p, i) => setBox(slabMesh, i, p.c, 1));
      const chunkMesh = C.inst(C.plateGeo(d + 0.3, T + 0.3, 0.22), MAT.plate, chunks.length, -2);
      chunks.forEach((p, i) => setBox(chunkMesh, i, p.c, 1));
      const accMesh = C.inst(GEO.plane, MAT.flat, accents.length);
      accents.forEach((p, i) => setBox(accMesh, i, p.c, p.w, p.hh, 1));
      [slabMesh, chunkMesh, accMesh].forEach(flushInst);
      // fans: split (chunk h of Q, K, V -> head h's piece) and concat (Z_h -> chunk h), thin and neutral
      const fans = [];
      const bot = y => y - T / 2 - 0.28, top = y => y + T / 2 + 0.28;
      for (let h = 0; h < H; h++) {
        const cx = k => X[k] - (d - 1) / 2 + h * dh + (dh - 1) / 2;
        for (const [f, p] of [['Qf', 'Q'], ['Kf', 'K'], ['Vf', 'V']]) fans.push({ h, a: V3(cx(f), bot(yA), zMid), b: V3(X[p], yS(h) + T / 2 + 0.12, zOf(h)) });
        fans.push({ h, a: V3(X.Z + dh / 2 + 0.1, yS(h), zOf(h)), b: V3(cx('Zc'), top(yC), zMid) });
      }
      const fanLines = C.lines(fans.length);
      fans.forEach((f, i) => fanLines.seg(i, f.a, f.b));
      // labels: the blocks' names; a slab's own names (Q_h ... Z_h) on head 1, and on another head
      // while one of its cells is held; "head n" with its swatch
      const lab = (html, p, cls = 'nn3d-name', cx = 0.5, cy = 1) => C.label(cls, html, p, cx, cy);
      const name = (k, i) => (i == null ? `<i>${k}</i>` : sub(k, i));
      for (const k of ['Q', 'K', 'V']) lab(name(k), V3(X[`${k}f`], yA + T / 2 + 0.2, zMid)).userData.fit = true;
      lab(`<i>Z</i> = [${sub('Z', 1)}${H > 1 ? ` ${H > 2 ? '⋯ ' : ''}${sub('Z', H)}` : ''}]`, V3(X.Zc, yC - T / 2 - 0.25, zMid), 'nn3d-name', 0.5, 0).userData.fit = true;
      if (wo) {
        lab(katexHtml(wo.name), V3(X.W, yC - d / 2 - 0.25, zMid), 'nn3d-name tex', 0.5, 0);
        lab(katexHtml(`Z${wo.name}`), V3(X.out, yC - T / 2 - 0.25, zMid), 'nn3d-name tex', 0.5, 0).userData.fit = true;
        lab('×', V3((X.Zc + X.W) / 2 + (d - dOut) / 4, yC, zMid), 'nn3d-op', 0.5, 0.5);
        lab('=', V3((X.W + X.out) / 2, yC, zMid), 'nn3d-op', 0.5, 0.5);
      }
      const slabNames = [];
      for (let h = 0; h < H; h++) {
        const z = zOf(h), n = h + 1, ty = yS(h) + T / 2 + 0.2;
        slabNames.push(['Q', 'K', 'A', 'V', 'Z'].map(k => lab(name(k, n), V3(X[k], ty, z))));
        if (H > 3 && h > 0 && h < H - 1) continue;
        lab(`${headSw(h)}head ${n}`, V3(x0 - 0.3, yS(h), z), 'nn3d-headlab', 1, 0.5).userData.fit = true;
        if (h === 0) {
          lab('→', V3((X.K + X.A) / 2 - (T - dh) / 4, yS(0), z), 'nn3d-op', 0.5, 0.5);
          lab('·', V3((X.A + X.V) / 2 + (T - dh) / 4, yS(0), z), 'nn3d-op', 0.5, 0.5);
          lab('=', V3((X.V + X.Z) / 2, yS(0), z), 'nn3d-op', 0.5, 0.5);
        }
      }
      const numLabs = cells.length + bars.length <= 220 ? cells.map(c => C.label('nn3d-num', '', c.p.clone().add(V3(0, 0, TD / 2)), 0.5, 0.5)) : [];
      const barLabs = cells.length + bars.length <= 220 ? bars.map(() => C.label('nn3d-num att', '', V3(), 0.5, 0.5)) : [];

      const val = c => {
        const fwd = store.state.fwd, fa = fwd?.attn?.[l];
        if (c.kind === 'W') return woAt(idx, c.i, c.j);
        if (c.kind === 'Q' || c.kind === 'K' || c.kind === 'V' || c.kind === 'Z') return fa?.heads?.[c.h]?.[c.kind]?.[c.t]?.[c.e];
        if (c.kind === 'out') {
          const zs = M.reshape(net, l, fwd?.a?.[l]).X;
          return zs[c.t].reduce((s, z, i) => s + z * woAt(idx, i, c.f), 0);
        }
        return c.node ? fwd?.node?.[c.node]?.a : NaN;
      };
      const barBase = () => mix(P.bg, P.node, 0.5);
      function paint() {
        idx();
        const st = store.state, fwd = st.fwd, fa = fwd?.attn?.[l], v3 = cur();
        const E = lensOf(), dim = !!E?.any;
        const maxA = (fwd ? maxAbs(fwd.a, fwd.z) : 1) || 1, maxW = maxAbs(net.edges.map(e => e.w)) || 1;
        const hl = heldCells(I, cells, bars, { l, qkv, wo, a }), focusOn = hl.any;
        const lit = [], hot = new Set();
        cells.forEach((c, i) => {
          const vv = val(c), on = hl.cells.has(i);
          const e = !dim ? 1 : c.node ? E.node(c.node) : c.edge ? E.edge(c.edge) : 1;
          const op = on ? 1 : le(e) * (focusOn ? FOCUS_NODE : 1);
          setCol(mesh, i, mix(P.bg, over(vv, c.kind === 'W' ? maxW : maxA, P.node, CELL_A), op));
          if (on) { lit.push(c.p); if (c.e !== undefined) hot.add(c.h); }
          const o = numLabs[i];
          if (o) {
            Object.assign(o.userData, { txt: num(vv), nums: !!v3?.nums, on, want: e >= NUM_MIN, no: false });
            showNum(o);
            o.element.style.opacity = op < 0.99 ? String(Math.max(0.35, op)) : '';
          }
        });
        flushInst(mesh);
        halo.count = lit.length;
        lit.forEach((p, i) => setBox(halo, i, tp.set(p.x, p.y, p.z + TD / 2 + 0.01), CUBE, CUBE, 1));
        flushInst(halo);
        const litBars = [], bb = barBase();
        bars.forEach((b, i) => {
          const A0 = fa?.heads?.[b.h]?.A?.[b.i]?.[b.j], u = isNum(A0) ? clamp(A0, 0, 1) : 0, on = hl.bars.has(i);
          const e = dim ? E.attn(l, b.i, b.j, b.h) : 1, op = on ? 1 : le(e) * (focusOn ? FOCUS_NODE : 1);
          const depth = b.masked ? 0.04 : 0.08 + 1.0 * u, z0 = b.p.z - TD / 2;
          setBox(barMesh, i, tp.set(b.p.x, b.p.y, z0 + depth / 2), CUBE, CUBE, depth);
          const base = b.masked ? mix(P.bg, P.node, 0.25) : mix(bb, P.att, 0.1 + 0.9 * u);
          setCol(barMesh, i, mix(P.bg, base, op));
          if (on) { litBars.push(V3(b.p.x, b.p.y, z0 + depth)); hot.add(b.h); }
          const o = barLabs[i];
          if (o) {
            o.position.set(b.p.x, b.p.y, z0 + depth + 0.02);
            Object.assign(o.userData, { txt: num(A0), nums: !!v3?.nums, on, want: e >= NUM_MIN, no: b.masked });
            showNum(o);
          }
        });
        flushInst(barMesh);
        barHalo.count = litBars.length;
        litBars.forEach((p, i) => setBox(barHalo, i, tp.set(p.x, p.y, p.z + 0.01), CUBE, CUBE, 1));
        flushInst(barHalo);
        // a head the lens drops fades: its plates, accents and fans
        const hs = dim ? E.heads(l) : null, kept = h => !hs || hs.has(h);
        fans.forEach((f, i) => fanLines.set(i, mix(P.bg, P.fixed, kept(f.h) ? (hot.has(f.h) ? 0.8 : 0.45) : 0.12), hot.has(f.h) ? 1.4 : 1));
        fanLines.flush();
        slabs.forEach((p, i) => setCol(slabMesh, i, kept(p.h) ? P.plate : P.bg));
        chunks.forEach((p, i) => setCol(chunkMesh, i, P.plate));
        accents.forEach((p, i) => setCol(accMesh, i, mix(P.bg, P.heads[p.h % P.heads.length], kept(p.h) ? 1 : 0.2)));
        [slabMesh, chunkMesh, accMesh].forEach(flushInst);
        // a slab's names: head 1's always, another's while it is held
        slabNames.forEach((ls, h) => { const on = h === 0 || hot.has(h); for (const o of ls) o.visible = on; });
      }
      function pick(px, py, proj) {
        idx();
        let best = null;
        const test = (p, spec, zoff) => {
          const q = proj(tp.set(p.x, p.y, p.z + zoff), CUBE / 2);
          if (q && Math.abs(q.x - px) < q.r && Math.abs(q.y - py) < q.r && (!best || q.z < best.z)) best = { z: q.z, spec };
        };
        const tokenOf = c => {
          if (c.kind === 'Z') return { kind: 'token', layer: l, t: c.t, ...(H > 1 ? { h: c.h } : {}) };
          if (c.kind === 'Q' || c.kind === 'K' || c.kind === 'V') return { kind: 'token', layer: qkv, t: c.t, g: G3[c.g], ...(H > 1 ? { h: c.h } : {}) };
          if (c.kind === 'W') return c.edge ? { kind: 'edge', id: c.edge } : null;
          return c.node ? { kind: 'node', id: c.node } : null;
        };
        cells.forEach(c => { const s = tokenOf(c); if (s) test(c.p, { ...s, node: c.node || null }, TD / 2); });
        bars.forEach(b => test(b.p, { kind: 'token', layer: l, t: b.i, ...(H > 1 ? { h: b.h } : {}), cell: true }, 0));
        return best?.spec || null;
      }
      const points = () => [...cells.map(c => c.p), ...bars.map(b => V3(b.p.x, b.p.y, b.p.z + 1.2)),
        V3(x0 - 0.4, yS(0), zOf(0)), V3(0, yA + T / 2 + 1.1, zMid), V3(X.Zc, yC - T / 2 - 1.1, zMid), V3(X.Zc, top(yC) + 0.4, zMid),
        V3(x0 - 0.4, yS(H - 1), zOf(H - 1)), V3(0, yS(H - 1) - T / 2 - 0.6, zOf(H - 1))];
      C.ground(new THREE.Box3().setFromPoints([...cells.map(c => c.p), ...bars.map(b => b.p)]));
      const sc = Math.abs(a.scale - 1 / Math.sqrt(dh)) < 1e-9 ? (dh === 1 ? '' : `/\\sqrt{${dh}}`) : `\\cdot ${M.fmt(a.scale, 2)}`;
      function declutter() {
        const all = [...numLabs, ...barLabs];
        for (const o of all) o.userData.face = faceW(o.position, CUBE / 2);
        return occlude(all);
      }
      const cap = `<b>${H > 1 ? `${H} heads, each on ${dh} of the ${d} columns` : `1 head on all ${d} columns: set heads to split them`}</b>`
        + `<span class="nn3d-tex">${katexHtml(`Z_h = \\operatorname{softmax}\\big(Q_h K_h^{\\top}${sc}${a.causal ? ' + M' : ''}\\big)\\, V_h`)}</span>`
        + `<span class="nn3d-tex">${katexHtml(`Z = [\\,Z_1${H > 2 ? ' \\cdots' : ''}${H > 1 ? ` \\; Z_{${H}}` : ''}\\,]${wo ? ` \\;\\to\\; Z${wo.name}` : ''}`)}</span>`;
      return {
        ...C, paint, pick, points, declutter, dir: DIRS.heads, fov: 18, layer: l, caption: cap, heads: H, d,
        nodePos: id => cells.find(c => c.node === id && (c.kind === 'Qf' || c.kind === 'Kf' || c.kind === 'Vf' || c.kind === 'Zc' || c.kind === 'out'))?.p || cells.find(c => c.node === id)?.p || null,
        nodeR: CUBE / 2,
        click(spec) {
          if (!spec) return null;
          if (spec.cell) return { kind: 'layer', id: net.layers[l].id };
          if (spec.node) return { kind: 'node', id: spec.node };
          return spec.kind === 'edge' || spec.kind === 'node' ? spec : null;
        },
      };
    }
    // Which heads-mode cells and bars the hover, selection and step-through light.
    function heldCells(I, cells, bars, { l, qkv, wo, a }) {
      const st = store.state, out = { cells: new Set(), bars: new Set(), any: false };
      const G3 = [a.qG, a.kG, a.vG];
      const byNode = id => cells.forEach((c, i) => { if (c.node === id) out.cells.add(i); });
      const light = (t, on) => {
        if (!t) return;
        if (t.kind === 'node' || t.kind === 'bias') byNode(t.id);
        else if (t.kind === 'row') { const n = I.byLayer[I.lix(t.layer)]?.[t.i]; if (n) byNode(n.id); }
        else if (t.kind === 'edge') {
          const e = I.edgeById.get(t.id);
          if (e?.tie) cells.forEach((c, i) => { if (c.kind === 'W' && c.edge && I.edgeById.get(c.edge)?.tie === e.tie) out.cells.add(i); });
          if (e) { byNode(e.from); byNode(e.to); }
        } else if (t.kind === 'token') {
          const L2 = I.lix(t.layer), hd = Number.isInteger(t.h) ? t.h : null;
          const inHead = c => hd === null || c.h === hd;
          if (L2 === l) {   // a query row: Q_h, A_h row, Z_h, concat and out rows
            cells.forEach((c, i) => { if (c.t === t.t && inHead(c) && ['Q', 'Z', 'Zc', 'out'].includes(c.kind)) out.cells.add(i); if (c.kind === 'Qf' && c.t === t.t && inHead(c)) out.cells.add(i); });
            bars.forEach((b, i) => { if (b.i === t.t && (hd === null || b.h === hd)) out.bars.add(i); });
          } else if (L2 === qkv) {
            const g = Number.isInteger(t.g) ? G3.indexOf(t.g) : -1;
            cells.forEach((c, i) => {
              if (c.t !== t.t || !inHead(c) || c.g === undefined || c.kind === 'Z' || c.kind === 'Zc' || c.kind === 'W' || c.kind === 'out') return;
              if (g < 0 || c.g === g) out.cells.add(i);
            });
            bars.forEach((b, i) => { if ((hd === null || b.h === hd) && (g === 0 ? b.i === t.t : g > 0 ? b.j === t.t : b.i === t.t || b.j === t.t)) out.bars.add(i); });
          } else if (wo && L2 === wo.l) cells.forEach((c, i) => { if (c.kind === 'out' && c.t === t.t) out.cells.add(i); });
          else {   // another token layer: its row t, if it is shown
            const s = I.shape[L2];
            if (s) for (let f = 0; f < s.d; f++) for (let g = 0; g < s.G; g++) { const n = I.tokNode(L2, g, t.t, f); if (n) byNode(n.id); }
          }
        }
        out.any ||= on;
      };
      light(st.hover, !!st.hover);
      light(st.sel, false);
      const A = st.anim;
      if (A) {
        const L2 = I.lix(A.l), n = I.byLayer[L2]?.[A.i], p = n && I.pos(n.id);
        if (p && L2 === l) {
          const ph = A.dir === 'bwd' ? 'bwd' : A.phase;
          cells.forEach((c, i) => {
            if ((ph === 'scores' || ph === 'bwd') && ((c.kind === 'Q' && c.t === p.t) || c.kind === 'K')) out.cells.add(i);
            if ((ph === 'sum' || ph === 'bwd') && ((c.kind === 'Z' && c.t === p.t) || c.kind === 'V')) out.cells.add(i);
            if (c.kind === 'Zc' && c.t === p.t) out.cells.add(i);
          });
          if (ph !== 'scores') bars.forEach((b, i) => { if (b.i === p.t && !(b.masked)) out.bars.add(i); });
        } else if (n) byNode(n.id);
        out.any = true;
      }
      return out;
    }

    // ================================================================ mode: tensor
    function buildTensor(v) {
      const net = store.net, C = newContent(), list = attnLayers(net);
      let I = index(net);
      const idx = () => (stale(I, net) ? (I = index(net)) : I);
      const l = list.includes(I.li.get(v.layer)) ? I.li.get(v.layer) : list[0] ?? -1;
      const src = v.src || (l >= 0 && I.att[l].heads > 1 ? 'net' : 'example');
      const useNet = src === 'net' && l >= 0;
      const spec = useNet ? I.att[l] : null;
      const T = useNet ? spec.T : EXAMPLE.T, d = useNet ? spec.d : EXAMPLE.d, H = useNet ? spec.heads : v.h, dh = d / H;
      const wo = useNet ? projOf(net, l) : null, ex = useNet ? null : exampleQKV(), woAt = woReader(wo, I);
      const proj = useNet ? !!wo && wo.W.length === d && wo.W[0].length === d : true;
      const steps = tensorSteps({ bug: v.bug, proj });
      // Q and K side by side on top, V below; within a block rows are tokens and columns features.
      // [h, T, d/h] puts head h in depth (head 1 in front): from the default camera the slabs
      // step to the right as they go back. Attention's A appears between Q and K, Z comes out of V,
      // and at the last step Z (top left) × W_O (top right) = out (below).
      const GH = 0.75, wTHD = d + (H - 1) * GH;
      const GZ = H > 1 ? clamp((dh + 0.7) / 0.55, 2.4, 5.5) : 0;
      const zOf = h => ((H - 1) / 2 - h) * GZ;
      const CX = (Math.max(wTHD, proj ? d : 0) + 2.8) / 2 + 0.3 * (H - 1) * GZ;
      const BY = Math.max(T, proj ? d : 0) / 2 + T / 2 + 2.6 + 0.12 * (H - 1) * GZ;
      const BP = [V3(-CX, 0, 0), V3(CX, 0, 0), V3(0, -BY, 0)];
      const by = b => BP[b].y;
      const yOf = r => (T - 1) / 2 - r;
      const deep = form => form === 'HTD' || form === 'BUG';
      const at = (form, b, t, f) => {
        const s = slotOf(form, t, f, { T, d, h: H }), o = BP[b];
        if (form === 'TD') return V3(o.x + f - (d - 1) / 2, o.y + yOf(t), 0);
        if (form === 'THD') return V3(o.x + s.head * (dh + GH) + s.col - (wTHD - 1) / 2, o.y + yOf(t), 0);
        return V3(o.x + s.col - (dh - 1) / 2, o.y + yOf(s.row), zOf(s.head));
      };
      const xL = form => (form === 'TD' ? -(d - 1) / 2 : form === 'THD' ? -(wTHD - 1) / 2 : -(dh - 1) / 2) - 0.5;
      const xR = form => -xL(form);
      const zF = form => (deep(form) ? zOf(0) : 0), zB = form => (deep(form) ? zOf(H - 1) : 0);
      const ap = H > 1 ? Math.min(1, (0.5 * GZ - 0.3) / T) : 1;   // A's cell pitch, so the heads' tiles don't overlap on screen
      // objects with a key frame per step: cubes (Q, K, V -> Z, out, W_O), A bars, head plates
      const cubes = [];
      for (let b = 0; b < 3; b++) for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) cubes.push({ b, t, f, kind: 'QKV'[b] });
      for (let t = 0; t < T; t++) for (let f = 0; f < d; f++) cubes.push({ b: 4, t, f, kind: 'out' });
      if (proj) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cubes.push({ b: 3, t: i, f: j, kind: 'W' });
      const bars = [];
      for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) bars.push({ h, i, j, masked: !!spec?.causal && j > i });
      const plates = [];
      for (const b of [0, 1, 2]) for (let h = 0; h < H; h++) plates.push({ b, h });
      const { mesh, halo } = cellSet(C, cubes.length);
      const barMesh = C.inst(GEO.bar, MAT.lit, bars.length);
      // neutral plates behind a head's chunk or slab, the head colour on a strip at their left edge
      const PW = dh + 0.24, PH = T + 0.3;
      const plateMesh = C.inst(C.plateGeo(PW, PH, 0.2), MAT.plate, plates.length, -2);
      const accMesh = C.inst(GEO.plane, MAT.flat, plates.length);
      const nums = cubes.length <= 230 ? cubes.map(() => C.label('nn3d-num', '', V3(), 0.5, 0.5)) : [];
      const blockLabs = [0, 1, 2].map(() => ({ name: C.label('nn3d-name side', '', V3(), 1, 0.5), shape: C.label('nn3d-shape', '', V3(), 0.5, 0) }));
      const headLabs = Array.from({ length: H }, (_, h) => C.label('nn3d-headlab top', `${headSw(h)}${H > 3 ? String(h + 1) : `head ${h + 1}`}`, V3(), 0.5, 1));

      // the numbers
      let story = null, maxV = 1, maxWO = 1, cmp = null;
      function compute() {
        const fwd = store.state.fwd;
        let Q, K, V, WO;
        if (useNet) {
          const R = M.reshape(net, l - 1, fwd?.a?.[l - 1]);
          Q = R.Q; K = R.K; V = R.V; WO = proj ? wo.W.map((r, i) => r.map((_, j) => woAt(idx, i, j))) : null;
        } else ({ Q, K, V, WO } = ex);
        const scale = useNet ? spec.scale : 1 / Math.sqrt(dh), causal = !!spec?.causal;
        story = tensorStory({ Q, K, V, h: H, scale, causal, WO, bug: v.bug });
        Object.assign(story, { Q, K, V, WO });
        const Zref = useNet ? M.reshape(net, l, fwd?.a?.[l]).X : v.bug ? tensorStory({ Q, K, V, h: H, scale, causal }).Z : story.Z;
        cmp = maxAbs(story.Z.map((row, t) => row.map((z, f) => z - (Zref[t]?.[f] ?? 0))));
        maxV = maxAbs(Q, K, V, story.Z, story.out) || 1;
        maxWO = maxAbs(WO) || 1;
      }
      // the key frames: per object and step { p, s (scale; 0 = hidden), val, max, tok, head }
      let keys = null;
      const lastQKV = steps.filter(x => x.show === 'qkv').pop().form;
      function frames() {
        keys = { cubes: [], bars: [], plates: [] };
        cubes.forEach(c => {
          keys.cubes.push(steps.map(st => {
            const head = Math.floor(c.f / dh);
            if (c.b <= 2) {
              // Q and K are used up at attend: they shrink where the last step left them. The V
              // cubes carry Z from attend on (fromHeads puts z back at (t, f)), and at proj move to
              // the top row as the Z that W_O multiplies.
              const zNow = st.show !== 'qkv';
              const form = st.show === 'out' ? 'TD' : zNow && c.b < 2 ? lastQKV : st.form;
              const p = at(form, st.show === 'out' && c.b === 2 ? 0 : c.b, c.t, c.f);
              const on = !zNow || c.b === 2;
              const val = c.b === 2 && zNow ? story.Z[c.t][c.f] : story['QKV'[c.b]][c.t][c.f];
              return { p, s: on ? 1 : 0, val, max: maxV, tok: c.t, head };
            }
            const on = st.show === 'out';
            if (c.kind === 'out') return { p: at('TD', 2, c.t, c.f), s: on ? 1 : 0, val: story.out?.[c.t]?.[c.f], max: maxV, tok: c.t, head: 0 };
            return { p: V3(BP[1].x + c.f - (d - 1) / 2, (d - 1) / 2 - c.t, 0), s: on ? 1 : 0, val: story.WO?.[c.t]?.[c.f], max: maxWO, tok: -1, head: 0 };
          }));
        });
        bars.forEach(b => {
          const A0 = story.heads[b.h]?.A?.[b.i]?.[b.j], u = isNum(A0) ? clamp(A0, 0, 1) : 0;
          const p = V3((b.j - (T - 1) / 2) * ap, ((T - 1) / 2 - b.i) * ap, zOf(b.h));
          keys.bars.push(steps.map(st => ({ p, s: st.key === 'attend' ? 1 : 0, depth: b.masked ? 0.04 : 0.08 + 1.2 * u, u })));
        });
        // a head's plate sits behind its chunk (THD) or its slab (HTD, BUG); Q's and K's go with
        // their cubes at attend, V's stays behind Z until the heads merge
        const bz = TD / 2 + 0.12;
        const plateAt = (form, b, hh) => (form === 'THD' ? V3(BP[b].x + hh * (dh + GH) + (dh - 1) / 2 - (wTHD - 1) / 2, by(b), -bz)
          : deep(form) ? V3(BP[b].x, by(b), zOf(hh) - bz)
            : V3(BP[b].x + hh * dh + (dh - 1) / 2 - (d - 1) / 2, by(b), -bz));
        plates.forEach(q => {
          keys.plates.push(steps.map(st => {
            const mine = st.show === 'qkv' || (q.b === 2 && st.show === 'z');
            const form = mine ? st.form : q.b === 2 ? 'TD' : lastQKV;
            return { p: plateAt(form, q.b, q.h), s: mine && form !== 'TD' ? 1 : 0 };
          }));
        });
      }
      // playback: from the drawn state to step `to`, staggered by head on the moves between forms
      const nSteps = steps.length;
      let shown = clamp(v.step, 0, nSteps - 1), anim = null;
      let now = null;   // the drawn state per object
      const colorOf = (k, mode) => (mode === 'token' && k.tok >= 0 ? P.toks[k.tok % P.toks.length] : over(k.val, k.max, P.node, CELL_A));
      function target(stepI, mode) {
        return {
          cubes: keys.cubes.map(ks => { const k = ks[stepI]; return { p: k.p, s: k.s, rgb: colorOf(k, mode), val: k.val, head: k.head }; }),
          bars: keys.bars.map(ks => ks[stepI]),
          plates: keys.plates.map(ks => ks[stepI]),
        };
      }
      let peekI = -1;   // the cube under the mouse: framed, and its number shown
      // Numbers only on the step's matrices (the cubes it shows, not the ones shrinking away): on a
      // held or hovered cube, and with the 1.2 button on, on all of them (showNum sizes them).
      function draw(state, hl) {
        const vv = cur(), byTok = vv?.color === 'token';   // token colours: no numbers
        state.cubes.forEach((k, i) => {
          setBox(mesh, i, k.p, CUBE * k.s, CUBE * k.s, k.s);
          setCol(mesh, i, hl.has(i) ? mix(k.rgb, P.hi, 0.35) : k.rgb);
          const o = nums[i];
          if (o) {
            o.position.set(k.p.x, k.p.y, k.p.z + (TD / 2) * k.s);
            Object.assign(o.userData, { txt: num(k.val), nums: !!vv?.nums, on: hl.has(i) || i === peekI, want: true, no: byTok || k.s <= 0.7, s: k.s });
            showNum(o);
          }
        });
        flushInst(mesh);
        const lit = state.cubes.map((k, i) => ((hl.has(i) || i === peekI) && k.s > 0.5 ? k.p : null)).filter(Boolean);
        halo.count = lit.length;
        lit.forEach((p, i) => setBox(halo, i, tp.set(p.x, p.y, p.z + TD / 2 + 0.01), CUBE, CUBE, 1));
        flushInst(halo);
        const bb = mix(P.bg, P.node, 0.5);
        state.bars.forEach((k, i) => {
          const depth = k.depth * k.s;
          setBox(barMesh, i, tp.set(k.p.x, k.p.y, k.p.z - TD / 2 + depth / 2), CUBE * ap * k.s, CUBE * ap * k.s, Math.max(1e-3, depth));
          setCol(barMesh, i, bars[i].masked ? mix(P.bg, P.node, 0.25) : mix(bb, P.att, 0.1 + 0.9 * k.u));
        });
        flushInst(barMesh);
        state.plates.forEach((k, i) => {
          setBox(plateMesh, i, k.p, k.s, k.s, 1);
          setCol(plateMesh, i, P.plate);
          setBox(accMesh, i, tp.set(k.p.x - (PW / 2 - 0.06) * k.s, k.p.y, k.p.z + 0.004), 0.08 * k.s, (PH - 0.2) * k.s, 1);
          setCol(accMesh, i, P.heads[plates[i].h % P.heads.length]);
        });
        flushInst(plateMesh);
        flushInst(accMesh);
      }
      // labels: a block's name at its left, its shape under it, head labels over chunks / slabs
      const shapeTxt = form => (form === 'TD' ? `[${T}, ${d}]` : form === 'THD' ? `[${T}, ${H}, ${dh}]` : `[${H}, ${T}, ${dh}]`);
      function labelsAt(stepI) {
        const st = steps[stepI], out = st.show === 'out', form = out ? 'TD' : st.form;
        const rows = [];   // per block: { name, shape, on, pl (name), pr (shape) }
        for (let b = 0; b < 3; b++) {
          let name = `<i>${'QKV'[b]}</i>`, f = form, on = st.show === 'qkv', tile = false, rowsH = T;
          if (st.show === 'z') {
            if (b === 0) on = false;
            if (b === 1) { on = st.key === 'attend'; name = `${sub('A', 'h')}`; tile = true; }
            if (b === 2) { on = true; name = '<i>Z</i>'; }
          }
          if (out) { on = true; name = ['<i>Z</i>', katexHtml('W_O'), katexHtml('ZW_O')][b]; f = 'TD'; if (b === 1) rowsH = d; }
          const shape = tile ? `[${H}, ${T}, ${T}]` : out && b === 1 ? `[${d}, ${d}]` : shapeTxt(f);
          const x0 = tile ? -(T * ap) / 2 - 0.35 : BP[b].x + xL(f), z0 = tile ? zOf(0) : zF(f);
          const yb = by(b) - (tile ? (T * ap) / 2 : rowsH / 2) - 0.3;
          rows.push({ name, shape, on, pl: V3(x0 - 0.25, by(b), z0), pr: V3(tile ? 0 : BP[b].x, yb, z0) });
        }
        const hb = st.show === 'qkv' ? 0 : st.show === 'z' ? 2 : -1, B0 = BP[Math.max(0, hb)];
        const heads = Array.from({ length: H }, (_, h) => ({
          on: hb >= 0 && form !== 'TD',
          p: form === 'THD' ? V3(B0.x + h * (dh + GH) + (dh - 1) / 2 - (wTHD - 1) / 2, B0.y + (T - 1) / 2 + 0.62, 0)
            : V3(B0.x, B0.y + (T - 1) / 2 + 0.62, zOf(h)),
        }));
        return { rows, heads };
      }
      function placeLabels(stepI, u = 1, from = stepI) {
        const A = labelsAt(from), B = labelsAt(stepI);
        blockLabs.forEach((L2, b) => {
          const a = A.rows[b], c = B.rows[b];
          L2.name.visible = L2.shape.visible = c.on;
          if (L2.name.element.__h !== c.name) { L2.name.element.__h = c.name; L2.name.element.innerHTML = c.name; }
          L2.shape.element.textContent = c.shape;
          L2.name.position.lerpVectors(a.pl, c.pl, u);
          L2.shape.position.lerpVectors(a.pr, c.pr, u);
        });
        headLabs.forEach((o, h) => {
          o.visible = B.heads[h].on && u > 0.5;
          o.position.copy(B.heads[h].p);
        });
      }
      let hlSet = new Set();
      function heldTensor() {
        const out = new Set();
        if (!useNet) return out;
        const st = store.state, ids = new Set();
        for (const t of [st.hover, st.sel]) if (t?.kind === 'node') ids.add(t.id);
        if (!ids.size) return out;
        const G3 = [spec.qG, spec.kG, spec.vG], zStep = steps[shown].show !== 'qkv';
        cubes.forEach((c, i) => {
          if (c.b > 2) return;
          const qn = I.tokNode(l - 1, G3[c.b], c.t, c.f)?.id, zn = I.tokNode(l, 0, c.t, c.f)?.id;
          if (ids.has(qn) && !(zStep && c.b === 2)) out.add(i);
          if (c.b === 2 && zStep && ids.has(zn)) out.add(i);
        });
        return out;
      }
      function paint() {
        compute();
        frames();
        hlSet = heldTensor();
        const t = target(shown, cur()?.color);
        if (!anim) { now = t; draw(now, hlSet); placeLabels(shown); }
        else anim.to = t;
      }
      function go(stepI) {
        stepI = clamp(stepI, 0, nSteps - 1);
        if (stepI === shown && !anim) return;
        const fromStep = anim ? anim.toStep : shown;
        shown = stepI;
        anim = { t0: performance.now(), from: now, to: target(stepI, cur()?.color), toStep: stepI, fromStep, stagger: steps[stepI].form !== steps[fromStep].form };
      }
      function frame(t) {
        if (!anim) return false;
        const e0 = (t - anim.t0) / STEP_MS;
        const ease = u => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);
        const stag = anim.stagger && H > 1 ? Math.min(0.35, 0.12 * (H - 1)) : 0;
        const uOf = head => ease(clamp((e0 - (H > 1 ? (stag * head) / (H - 1) : 0)) / (1 - stag), 0, 1));
        const F = anim.from, Tt = anim.to;
        now = {
          cubes: F.cubes.map((a, i) => {
            const b = Tt.cubes[i], u = uOf(b.head);
            // a cube moving between depth and width rises a little on the way, so the paths don't cross
            const p = V3().lerpVectors(a.p, b.p, u);
            if (a.p.z !== b.p.z) p.y += Math.sin(Math.PI * u) * 0.35;
            return { p, s: a.s + (b.s - a.s) * u, rgb: mix(a.rgb, b.rgb, u), val: u < 0.5 ? a.val : b.val, head: b.head };
          }),
          bars: F.bars.map((a, i) => { const b = Tt.bars[i], u = uOf(bars[i].h); return { p: b.p, s: a.s + (b.s - a.s) * u, depth: b.depth, u: b.u }; }),
          plates: F.plates.map((a, i) => { const b = Tt.plates[i], u = uOf(plates[i].h); return { p: V3().lerpVectors(a.p, b.p, u), s: a.s + (b.s - a.s) * u }; }),
        };
        draw(now, hlSet);
        placeLabels(anim.toStep, ease(clamp(e0, 0, 1)), anim.fromStep);
        if (e0 >= 1 + stag) { anim = null; now = Tt; draw(now, hlSet); placeLabels(shown); }
        return true;
      }
      // Any source: the front-most shown cube under (px, py) (null: the mouse left). True if it changed.
      function peek(px, py, proj) {
        let best = -1, bz = Infinity;
        if (px != null && now) {
          now.cubes.forEach((k, i) => {
            if (k.s < 0.5) return;
            const q = proj(tp.set(k.p.x, k.p.y, k.p.z + (TD / 2) * k.s), (CUBE / 2) * k.s);
            if (q && Math.abs(q.x - px) <= q.r && Math.abs(q.y - py) <= q.r && q.z < bz) { best = i; bz = q.z; }
          });
        }
        if (best === peekI) return false;
        peekI = best;
        if (now) draw(now, hlSet);
        return true;
      }
      function declutter() {
        for (const o of nums) o.userData.face = faceW(o.position, (CUBE / 2) * (o.userData.s ?? 1));
        return occlude(nums);
      }
      function pick(px, py, proj) {
        if (!useNet || !now) return null;
        let best = null;
        const G3 = [spec.qG, spec.kG, spec.vG], zStep = steps[shown].show !== 'qkv';
        now.cubes.forEach((k, i) => {
          const c = cubes[i];
          if (c.b > 2 || k.s < 0.5) return;
          const q = proj(tp.set(k.p.x, k.p.y, k.p.z + TD / 2), CUBE / 2);
          if (!q || Math.abs(q.x - px) > q.r || Math.abs(q.y - py) > q.r || (best && q.z > best.z)) return;
          const id = c.b === 2 && zStep ? I.tokNode(l, 0, c.t, c.f)?.id : I.tokNode(l - 1, G3[c.b], c.t, c.f)?.id;
          if (id) best = { z: q.z, spec: { kind: 'node', id } };
        });
        return best?.spec || null;
      }
      function caption() {
        const st = steps[shown], tx = TENSOR_TEXT[st.key], code = tx.code;
        let note = tx.note;
        if (st.key === 'view') {
          const toks = [...new Set(Array.from({ length: T * dh }, (_, k) => Math.floor(k / d) + 1))];
          note = `A view keeps the memory order, so “head 1” is just the first T·d/h = ${T * dh} numbers: pieces of token${toks.length > 1 ? 's' : ''} ${toks.join(' and ')}, not chunk 1 of every token.${v.color === 'token' ? '' : ' Colour by tokens to see it.'}`;
        } else if (st.key === 'bugback') {
          note = `Every number goes back to its old place, so the shape looks right, but each head attended over mixed-up rows: Z is off by up to ${num(cmp)}.`;
        } else if (st.key === 'attend' && v.bug) {
          note = 'The same batched matmul, but each slab’s rows are pieces of the wrong tokens, so A_h compares the wrong things.';
        } else if (st.key === 'merge') {
          note += useNet ? ` It matches the attention layer’s Z (largest difference ${num(cmp)}).` : '';
        }
        const legend = `T = ${T} tokens · d = ${d} · h = ${H} head${H > 1 ? 's' : ''} · d/h = ${dh}${useNet ? '' : ' · example numbers'}`;
        return { title: `${shown + 1} / ${nSteps} · ${tx.title}`, count: `${shown + 1} / ${nSteps}`, name: tx.title,
          code: `${code}${' '.repeat(Math.max(2, 32 - code.length))}# ${tx.shape}`, note, legend };
      }
      compute();
      frames();
      now = target(shown, v.color);
      draw(now, hlSet);
      placeLabels(shown);
      const points = () => {
        const pts = [];
        for (const form of v.bug ? ['TD', 'BUG'] : ['TD', 'THD', 'HTD']) {
          for (const b of [0, 1, 2]) {
            for (const t of [0, T - 1]) for (const f of [0, d - 1]) pts.push(at(form, b, t, f));
            pts.push(V3(BP[b].x + xL(form) - 1.3, by(b), zF(form)), V3(BP[b].x, by(b) - T / 2 - 1.1, zF(form)), V3(BP[b].x, by(b) + T / 2 + 1, zB(form)));
          }
        }
        if (proj) pts.push(V3(BP[1].x, (d - 1) / 2 + 0.5, 0), V3(BP[1].x, -(d - 1) / 2 - 1.2, 0));
        return pts;
      };
      C.ground(new THREE.Box3().setFromPoints(points()));
      return {
        ...C, paint, frame, pick, peek, points, declutter, dir: DIRS.tensor, fov: 15,
        tensor: { steps, go, get shown() { return shown; }, caption, src, useNet, H, T, d, hasNet: l >= 0, animating: () => !!anim },
        setColor() { if (now) { const t = target(shown, cur()?.color); if (anim) anim.to = t; else { now = t; draw(now, hlSet); } } },
        nodePos: id => {
          if (!useNet || !now) return null;
          const G3 = [spec.qG, spec.kG, spec.vG], zStep = steps[shown].show !== 'qkv';
          const i = cubes.findIndex(c => c.b <= 2 && (c.b === 2 && zStep ? I.tokNode(l, 0, c.t, c.f)?.id === id : I.tokNode(l - 1, G3[c.b], c.t, c.f)?.id === id));
          return i >= 0 && now.cubes[i].s > 0.5 ? now.cubes[i].p : null;
        },
        nodeR: CUBE / 2,
        click: spec => spec,
      };
    }

    // ================================================================ camera: fit, flights, mirror
    let content = null, builtKey = '', builtNetKey = '', builtTitle, built = false, builds = 0, paints = 0;
    let userMoved = false, flight = null, follow = null, lastPub = null, lastIn = null, camDirty = false, lastPubT = 0;
    let SW = 1, SH = 1, off = null;   // stage px; off: where the orbit target sits on screen (the free area's centre)
    function size() {
      const w = root.clientWidth, h = root.clientHeight;
      if (!w || !h) return false;
      const px = dpr();
      if (px !== renderer.getPixelRatio()) { renderer.setPixelRatio(px); SW = 0; }
      if (w !== SW || h !== SH) {
        SW = w; SH = h;
        renderer.setSize(w, h);
        labelR.setSize(w, h);
        camera.aspect = w / h;
        MAT.line.uniforms.uRes.value.set(w * px, h * px);
        MAT.line.uniforms.uPx.value = px;
        applyOffset();
      }
      return true;
    }
    function applyOffset() {
      const cx = off?.cx ?? SW / 2, cy = off?.cy ?? SH / 2;
      camera.setViewOffset(SW, SH, SW / 2 - cx, SH / 2 - cy, SW, SH);
      camera.updateProjectionMatrix();
    }
    // The free part of the stage: clear of the floating panels (Train, Attention, the 3D plots,
    // this bar) and above the lens bar, as view.js's fit does; the roomiest rectangle in which the
    // content comes out largest (within 10%).
    function freeRect(aspect) {
      const s = stage.getBoundingClientRect(), rects = [];
      let bottom = s.height;
      const skip = c => c === root || c.tagName?.toLowerCase() === 'svg' || c.classList.contains('nn-insp-layer') || c.classList.contains('nn-tour');
      for (const c of stage.children) {
        if (skip(c) || c.hidden) continue;
        const r = c.getBoundingClientRect();
        if (!r.width || !r.height || getComputedStyle(c).display === 'none' || getComputedStyle(c).visibility === 'hidden') continue;
        if (c.classList.contains('nn-lens')) { if (r.bottom >= s.bottom - 40) bottom = Math.min(bottom, r.top - s.top - 4); continue; }
        if (r.width * r.height > 0.8 * s.width * s.height) continue;
        rects.push({ x0: r.left - s.left - 6, y0: r.top - s.top - 6, x1: r.right - s.left + 6, y1: r.bottom - s.top + 6 });
      }
      const br = bar.getBoundingClientRect();
      if (br.width && br.height) rects.push({ x0: br.left - s.left - 6, y0: br.top - s.top - 6, x1: br.right - s.left + 6, y1: br.bottom - s.top + 6 });
      const Fw = s.width, Fh = Math.max(s.height * 0.5, bottom), all = { x: 0, y: 0, w: Fw, h: Fh };
      const obs = rects.filter(r => r.x1 > 0 && r.x0 < Fw && r.y1 > 0 && r.y0 < Fh);
      if (!obs.length) return all;
      const cuts = (lo, hi, k0, k1) => [...new Set([lo, hi, ...obs.flatMap(r => [r[k0], r[k1]]).filter(x => x > lo && x < hi)])].sort((a, b) => a - b);
      const xs = cuts(0, Fw, 'x0', 'x1'), ys = cuts(0, Fh, 'y0', 'y1'), free = [];
      for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) {
        if (xs[j] - xs[i] < 200) continue;
        for (let p = 0; p < ys.length; p++) for (let q = p + 1; q < ys.length; q++) {
          const A = { x: xs[i], y: ys[p], w: xs[j] - xs[i], h: ys[q] - ys[p] };
          if (A.h < 160 || obs.some(r => r.x0 < A.x + A.w && r.x1 > A.x && r.y0 < A.y + A.h && r.y1 > A.y)) continue;
          free.push({ A, k: Math.min(A.w / aspect, A.h) });
        }
      }
      if (!free.length) return all;
      const kMax = Math.max(...free.map(c => c.k));
      return free.filter(c => c.k >= kMax * 0.9 - 1e-9).reduce((a, c) => (c.A.w * c.A.h > a.A.w * a.A.h ? c : a)).A;
    }
    const cam2 = new THREE.PerspectiveCamera(FOV, 1, 0.05, 5000);
    // Frame content.points() from direction dir (default: the current one) in the free area:
    // { p, t, D (the distance), off (where t sits on screen) }. No side effects.
    function fitPose(dir) {
      const pts = content?.points?.() || [];
      if (!pts.length || !size()) return null;
      const box = new THREE.Box3().setFromPoints(pts), target = box.getCenter(V3());
      const d = dir ? V3(...dir).normalize() : camera.position.clone().sub(controls.target).normalize();
      // the content's aspect from this direction, for picking the free rectangle
      cam2.fov = camera.fov; cam2.aspect = SW / SH; cam2.clearViewOffset(); cam2.position.copy(target).addScaledVector(d, 100); cam2.up.copy(UP); cam2.lookAt(target);
      cam2.updateProjectionMatrix(); cam2.updateMatrixWorld();
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) { tv.copy(p).project(cam2); x0 = Math.min(x0, tv.x * SW); x1 = Math.max(x1, tv.x * SW); y0 = Math.min(y0, tv.y * SH); y1 = Math.max(y1, tv.y * SH); }
      const A = freeRect(Math.max(0.2, (x1 - x0) / Math.max(1e-6, y1 - y0)));
      const o = { cx: A.x + A.w / 2, cy: A.y + A.h / 2 };
      cam2.setViewOffset(SW, SH, SW / 2 - o.cx, SH / 2 - o.cy, SW, SH);
      cam2.updateProjectionMatrix();
      const pad = clamp(Math.min(A.w, A.h) * 0.06, 16, 44);
      // labels marked fit (a layer's header, a block's name) are kept in frame at their drawn size:
      // [anchor, width, height, cx, cy] in px, measured once they are on the page
      const boxes = (content?.labs || []).filter(o => o.userData.fit && o.visible).map(o => {
        const e = o.element, w = e.offsetWidth || 8 * (e.textContent || '').length + 12, h = e.offsetHeight || 20;
        return [o.position, w, h, o.center.x, o.center.y];   // content groups sit at the origin
      });
      const fits = D => {
        cam2.position.copy(target).addScaledVector(d, D);
        cam2.lookAt(target);
        cam2.updateMatrixWorld();
        for (const p of pts) {
          tv.copy(p).project(cam2);
          const px = ((tv.x + 1) / 2) * SW, py = ((1 - tv.y) / 2) * SH;
          if (tv.z > 1 || px < A.x + pad || px > A.x + A.w - pad || py < A.y + pad || py > A.y + A.h - pad) return false;
        }
        for (const [p, w, h, cx, cy] of boxes) {
          tv.copy(p).project(cam2);
          const x0 = ((tv.x + 1) / 2) * SW - cx * w, y0 = ((1 - tv.y) / 2) * SH - (1 - cy) * h;
          if (tv.z > 1 || x0 < A.x + 6 || x0 + w > A.x + A.w - 6 || y0 < A.y + 6 || y0 + h > A.y + A.h - 6) return false;
        }
        return true;
      };
      let hi = 4;
      while (!fits(hi) && hi < 5000) hi *= 1.6;
      let lo = hi / 1.6;
      for (let k = 0; k < 24; k++) { const m = (lo + hi) / 2; if (fits(m)) hi = m; else lo = m; }
      return { p: target.clone().addScaledVector(d, hi), t: target, D: hi, off: o };
    }
    const reOffset = () => { const pose = fitPose(null); if (pose) { off = pose.off; applyOffset(); } return pose; };
    // Zoom stays within a range of the fit, so the content can't be lost off into the fog.
    const limits = D => { controls.minDistance = D * 0.12; controls.maxDistance = D * 5; };
    function fit(ms = 400, dir = null) {
      const pose = fitPose(dir);
      if (!pose) return;
      off = pose.off;
      applyOffset();
      limits(pose.D);
      userMoved = false;
      if (!ms) { camera.position.copy(pose.p); controls.target.copy(pose.t); controls.update(); flight = null; camDirty = true; needRender = true; nudgeCards(); return; }
      flight = { t0: performance.now(), ms, p0: camera.position.clone(), q0: controls.target.clone(), p1: pose.p, q1: pose.t };
    }
    // A new view comes in calmly: framed from a little further out and turned a few degrees, it
    // eases into its fit while the stage fades in (none of it with reduced motion).
    function settle(dir) {
      fit(0, dir);
      fadeIn();
      if (calm() || !content?.points) return;
      const t = controls.target.clone(), rel = camera.position.clone().sub(t);
      const p0 = t.clone().add(rel.applyAxisAngle(UP, -0.14).multiplyScalar(1.16));
      flight = { t0: performance.now(), ms: SETTLE_MS, p0, q0: t.clone(), p1: camera.position.clone(), q1: t, out: true };
      camera.position.copy(p0);
      controls.update();
    }
    let fade = null;
    function fadeIn() {
      fade?.cancel();
      fade = calm() ? null : gl.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 320, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' });
    }
    function flyTo(pose, ms) {
      flight = { t0: performance.now(), ms, p0: camera.position.clone(), q0: controls.target.clone(), p1: V3(...pose.p), q1: V3(...pose.t) };
    }
    // The mirrored camera: position and target, plus k = distance / this window's fit distance, so
    // an audience window of another size frames the same view (camFrom scales it back).
    const pubCam = () => {
      const c = { p: camera.position.toArray().map(r3), t: controls.target.toArray().map(r3) };
      const pose = fitPose(null);
      if (pose?.D) c.k = r3(camera.position.distanceTo(controls.target) / pose.D);
      return c;
    };
    function camFrom(c) {
      const t = V3(...c.t), p = V3(...c.p);
      if (!isNum(c.k)) return { p, t };
      const dir = p.clone().sub(t).normalize(), pose = fitPose(dir.toArray());
      if (!pose) return { p, t };
      off = pose.off;
      applyOffset();
      return { p: t.clone().addScaledVector(dir, c.k * pose.D), t };
    }
    const sameCam = (a, b) => !!a && !!b && a.p.every((x, i) => Math.abs(x - b.p[i]) < 2e-3) && a.t.every((x, i) => Math.abs(x - b.t[i]) < 2e-3);
    function publish() {
      if (audience || !store.state.v3d) return;
      const c = pubCam();
      lastPub = c;
      lastPubT = performance.now();
      camDirty = false;
      const v = cur();
      if (!sameCam(v.camera, c)) store.set('v3d', { ...v, camera: c });
    }
    // The inspector re-places its cards on pointer and wheel input in the tab; a camera that
    // moves by itself (a fit, the mirror) sends it a no-op wheel so the cards follow.
    let nudgeT = 0;
    function nudgeCards() {
      const t = performance.now();
      if (t - nudgeT < 400) return;
      nudgeT = t;
      root.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 0 }));
    }
    controls.addEventListener('start', () => { userMoved = true; flight = null; });
    controls.addEventListener('change', () => { camDirty = true; needRender = true; });

    // ================================================================ build and apply
    let needPaint = true, needRender = true, relabel = true;   // relabel: re-measure the labels after the next render
    let camMoves = 0;   // bumped whenever the camera (or the stage size) changes: the stack's rest layout of headers
    function invalidate() { needPaint = true; }
    function modeKey(v) {
      return JSON.stringify([v.mode, v.mode === 'stack' ? null : v.layer, v.mode === 'tensor' ? [v.src, v.h, v.bug] : null]);
    }
    function rebuild(v, { keepCamera }) {
      content?.dispose();
      content = null;
      const make = { stack: buildStack, heads: buildHeads, tensor: buildTensor }[v.mode];
      try { content = make(v); }
      catch (err) { console.error('[nn/view3d] build:', err); content = { ...newContent(), empty: `The 3D view could not draw this net (${err.message})` }; }
      msg.hidden = !content.empty;
      camera.fov = content.fov || FOV;   // tensor and heads: flatter, so near and far blocks keep their size
      camera.updateProjectionMatrix();
      msg.textContent = content.empty || '';
      builtKey = modeKey(v);
      builtNetKey = netKey(store.net);
      builtTitle = store.net.meta?.title;
      built = true;
      builds++;
      needPaint = true;
      renderBar(v, true);   // first: the fit keeps clear of the bar
      depthCues();
      if (size()) { scene.updateMatrixWorld(); labelR.render(scene, camera); }   // the labels go on the page, so the fit can measure them
      if (audience && v.camera) {
        const c = camFrom(v.camera);
        camera.position.copy(c.p);
        controls.target.copy(c.t);
        controls.update();
        follow = null;
        if (!keepCamera) fadeIn();
      } else if (!keepCamera) settle(content.dir);
      else if (!userMoved && !audience) fit(300);
      relabel = true;
    }
    // The fog's range (and the lines' and the ground's) follows the camera's distance to the
    // content, so the back of it falls off by about a third at any zoom.
    let cueC = V3(), cueR = 1;
    function depthCues() {
      const pts = content?.points?.() || [];
      if (pts.length) { const s = new THREE.Box3().setFromPoints(pts).getBoundingSphere(new THREE.Sphere()); cueC = s.center; cueR = Math.max(1, s.radius); }
      updateFog();
    }
    function updateFog() {
      const dd = camera.position.distanceTo(cueC), near = Math.max(0.1, dd - cueR * 0.4), far = dd + cueR * 3.2;
      scene.fog.near = near;
      scene.fog.far = far;
      fogU.value.set(near, far);
    }
    let barKey = '';
    function apply(v) {
      if (!v) return;
      size();
      const key = modeKey(v);
      if (key !== builtKey || !content) {
        const sameMode = content && JSON.parse(builtKey || '[]')[0] === v.mode;
        rebuild(v, { keepCamera: sameMode && !(audience && v.camera) });
      } else if (content.tensor) {
        if (clamp(v.step, 0, content.tensor.steps.length - 1) !== content.tensor.shown) content.tensor.go(v.step);
        else content.setColor();
      }
      needPaint = true;
      if (audience) {
        if (v.camera && !sameCam(v.camera, lastIn)) { lastIn = v.camera; follow = camFrom(v.camera); }
      } else if (v.camera && lastPub && !sameCam(v.camera, lastPub) && !sameCam(v.camera, pubCam())) {
        flyTo(v.camera, 350);   // someone else set the camera
      }
      renderBar(v);
    }
    function onNet(p) {
      const v = cur();
      if (!v || !content) return;
      const k = netKey(store.net);
      if (p?.structural || k !== builtNetKey) {
        const fresh = store.net.meta?.title !== builtTitle;
        rebuild(v, { keepCamera: !fresh });
      } else needPaint = true;
      renderBar(v);
    }

    // ================================================================ the bar
    let barList = null, barListKey = null;   // attnLayers for the built net (the bar asks on every net event)
    function renderBar(v, force = false) {
      if (barListKey !== builtNetKey) { barListKey = builtNetKey; barList = attnLayers(store.net); }
      const net = store.net, list = barList, T = content?.tensor;
      const key = JSON.stringify([v.mode, v.step, v.src, v.h, v.bug, v.color, v.nums, v.layer, list, content?.heads, content?.d, T?.src, T?.shown, playing, !!content?.empty]);
      if (!force && key === barKey) return;
      barKey = key;
      // The shared components: a segment s() of a .ui-seg, a .ui-btn.sm b() (a toggle when on is
      // given), an icon button ib() (a glyph stands in if the icons did not load).
      const ico = name => window.mathboardIcons?.svg(name, { size: 14 }) || '';
      const btn = (cls, act, label, title, on = false, dis = false) =>
        `<button type="button"${cls || on ? ` class="${cls}${on ? `${cls ? ' ' : ''}on` : ''}"` : ''} data-act="${act}" title="${esc(title)}"${dis ? ' disabled' : ''}>${label}</button>`;
      const s = (act, label, title, on, dis) => btn('', act, label, title, on, dis);
      const b = (act, label, title, on, dis) => btn('ui-btn sm', act, label, title, on, dis);
      const ib = (act, icon, glyph, title, on, dis, cls = '') => btn(`ui-btn sm icon${cls}`, act, ico(icon) || glyph, title, on, dis);
      const seg = (inner, cls = '') => `<span class="ui-seg${cls}">${inner}</span>`;
      let html = `<div class="nn3d-ctls nn3d-ctl"><div class="nn3d-row"><span class="nn3d-title">3D</span>${seg(
        s('mode:stack', 'Stack', 'Every layer as a sheet of neurons, the sheets in depth', v.mode === 'stack')
        + s('mode:heads', 'Heads', list.length ? 'One attention layer, a slab per head' : 'Needs an attention layer', v.mode === 'heads', !list.length)
        + s('mode:tensor', 'Tensor', 'The multi-head reshape and transpose, as moving cubes', v.mode === 'tensor'))}`;
      if (v.mode !== 'stack' && list.length > 1) {
        const at = list.find(i => net.layers[i].id === v.layer) ?? list[0];
        html += `<select class="ui-field sm" data-act="layer" title="Attention layer">${list.map(i => `<option value="${esc(net.layers[i].id)}"${i === at ? ' selected' : ''}>${esc(net.layers[i].name || `layer ${i}`)}</option>`).join('')}</select>`;
      }
      html += '<span class="nn3d-sp"></span>';
      html += b('nums', '1.2', 'Numbers on the neurons and cells (off: only on the one under the mouse or lit)', v.nums);
      html += ib('fit', 'home', '&#10227;', 'Reset the view (F frames it)');
      html += ib('close', 'close', '&times;', 'Back to the 2D canvas (D)');
      html += '</div>';
      if (v.mode === 'heads' && content?.heads) {
        const dv = divisors(content.d);
        html += `<div class="nn3d-row"><span class="nn3d-lab2">Heads</span>${seg(dv.map(k => s(`heads:${k}`, String(k), `Split d = ${content.d} into ${k} head${k > 1 ? 's' : ''} of ${content.d / k} (edits the attention layer; Ctrl+Z undoes)`, content.heads === k)).join(''), ' sm steps')}</div>`;
      }
      if (v.mode === 'tensor' && T) {
        const n = T.steps.length;
        html += `<div class="nn3d-row"><span class="nn3d-pair">${ib('prev', 'chevron-left', '&#9664;', 'Previous step (←)', false, T.shown === 0)
          + ib('play', playing ? 'pause' : 'play', playing ? '&#10073;&#10073;' : '&#9654;', playing ? 'Pause' : 'Play the steps', false, false, ' soft')
          + ib('next', 'chevron-right', '&#9654;', 'Next step (→)', false, T.shown >= n - 1)}</span>`
          + seg(T.steps.map((st, i) => s(`step:${i}`, String(i + 1), TENSOR_TEXT[st.key].title, i === T.shown)).join(''), ' sm steps')
          + '</div><div class="nn3d-row">'
          + seg(s('src:net', 'Net', T.hasNet ? 'This net’s Q, K and V' : 'Needs an attention layer', T.src === 'net' && T.useNet, !T.hasNet) + s('src:example', `${EXAMPLE.T}×${EXAMPLE.d}`, 'A 4 × 6 example', !T.useNet), ' sm');
        if (!T.useNet) html += `<span class="nn3d-lab2">h</span>${seg(divisors(EXAMPLE.d).map(k => s(`h:${k}`, String(k), `${k} head${k > 1 ? 's' : ''} of ${EXAMPLE.d / k}`, v.h === k)).join(''), ' sm steps')}`;
        html += seg(s('color:value', 'Values', 'Colour the cubes by value', v.color === 'value') + s('color:token', 'Tokens', 'Colour the cubes by the token they came from', v.color === 'token'), ' sm')
          + b('bug', 'The bug', T.H > 1 ? 'Show the common mistake: view(h, T, d/h) without the transpose' : 'With one head the view and the transpose agree: pick 2+ heads to see the bug', v.bug, T.H === 1 && !v.bug) + '</div>';
      }
      html += '</div>';
      let cap = '';
      if (v.mode === 'tensor' && T) {
        const c = T.caption(), name = c.name.charAt(0).toUpperCase() + c.name.slice(1);
        cap = `<div class="nn3d-cap"><div class="nn3d-cap-h"><span class="nn3d-count">${esc(c.count)}</span><b>${esc(name)}</b></div>`
          + `<code>${esc(c.code)}</code>${c.note ? `<span>${esc(c.note)}</span>` : ''}<em>${esc(c.legend)}</em></div>`;
      } else if (v.mode === 'heads' && content?.caption) cap = `<div class="nn3d-cap heads">${content.caption}</div>`;
      bar.innerHTML = html + cap;
      bar.classList.toggle('wide', v.mode !== 'stack');
      placeBar();
    }
    // The bar sits at the bottom right (or along the bottom), above the lens bar when that one
    // reaches under it, and ends left of a floating panel on the right that reaches down to it
    // (a tall Train panel at 1280), while that leaves it 360 px.
    function placeBar() {
      bar.style.bottom = bar.style.right = '';
      const s = root.getBoundingClientRect();
      for (const c of stage.querySelectorAll(':scope > .nn-train, :scope > .nn-attnviz, :scope > .nn-s3d')) {
        if (c.hidden || !c.offsetWidth) continue;
        const r = c.getBoundingClientRect(), a = bar.getBoundingClientRect(), right = Math.round(s.right - r.left + 8);
        if (r.left > s.left + s.width / 2 && r.bottom > a.top - 8 && r.left < a.right + 8 && s.width - right - 10 >= 360) bar.style.right = `${right}px`;
      }
      const lens = stage.querySelector(':scope > .nn-lens');
      if (!lens || lens.hidden || !lens.offsetWidth) return;
      const a = bar.getBoundingClientRect(), b = lens.getBoundingClientRect();
      if (a.left < b.right + 8 && b.left < a.right && a.top < b.bottom && b.top < a.bottom) {
        bar.style.bottom = `${Math.round(root.getBoundingClientRect().bottom - b.top + 8)}px`;
      }
    }
    let playing = false, playTimer = 0;
    function setPlay(on) {
      playing = on;
      clearInterval(playTimer);
      if (on) {
        const T = content?.tensor;
        if (T && T.shown >= T.steps.length - 1) put({ step: 0 });
        playTimer = setInterval(() => {
          const t = content?.tensor;
          if (!t || !playing) return setPlay(false);
          if (t.shown >= t.steps.length - 1) return setPlay(false);
          put({ step: t.shown + 1 });
        }, PLAY_MS);
      }
      const v = cur();
      if (v) renderBar(v, true);
    }
    function step(dir) {
      const T = content?.tensor;
      if (!T) return;
      const n = clamp(T.shown + dir, 0, T.steps.length - 1);
      if (n !== T.shown) put({ step: n });
    }
    if (!audience) {
      bar.addEventListener('click', e => {
        const el = e.target.closest('[data-act]');
        if (!el || el.tagName === 'SELECT') return;
        const [act, arg] = el.dataset.act.split(':');
        const v = cur();
        if (!v) return;
        if (act === 'mode') setMode(arg);
        else if (act === 'nums') put({ nums: !v.nums });
        else if (act === 'fit') fit(450, content?.dir);
        else if (act === 'close') toggle(false);
        else if (act === 'prev') { setPlay(false); step(-1); }
        else if (act === 'next') { setPlay(false); step(1); }
        else if (act === 'play') setPlay(!playing);
        else if (act === 'step') { setPlay(false); put({ step: +arg }); }
        else if (act === 'src') put({ src: arg, step: 0 });
        else if (act === 'h') put({ h: +arg, step: Math.min(v.step, 5) });
        else if (act === 'color') put({ color: arg });
        else if (act === 'bug') put({ bug: !v.bug, step: 0, color: !v.bug ? 'token' : v.color });
        else if (act === 'heads') setHeads(+arg);
      });
      bar.addEventListener('change', e => {
        if (e.target.dataset.act === 'layer') { put({ layer: e.target.value, step: 0 }); e.target.blur(); }
      });
    }
    function setHeads(k) {
      const l = content?.layer;
      if (!Number.isInteger(l)) return;
      const id = store.net.layers[l]?.id;
      if (!id || !M.setLayer(store.net, id, { heads: k })) return ctx.toast?.(`heads must divide d = ${content.d}`);
      store.commit('Set heads');
    }

    // ================================================================ pointer: hover and click
    const rect = () => renderer.domElement.getBoundingClientRect();
    const camRight = V3();
    function proj(p, r) {
      tv.copy(p).project(camera);
      if (tv.z > 1 || tv.z < -1) return null;
      const x = ((tv.x + 1) / 2) * SW, y = ((1 - tv.y) / 2) * SH, z = tv.z;
      if (!r) return { x, y, z, r: 0 };
      camRight.setFromMatrixColumn(camera.matrixWorld, 0);
      tp.copy(p).addScaledVector(camRight, r).project(camera);
      return { x, y, z, r: Math.hypot(((tp.x + 1) / 2) * SW - x, ((1 - tp.y) / 2) * SH - y) };
    }
    let myHover = null, hoverKey = '', down = null;
    function hoverAt(e) {
      if (audience || !content?.pick) return;
      const r = rect(), x = e ? e.clientX - r.left : null, y = e ? e.clientY - r.top : null;
      if (content.peek?.(x, y, proj)) needRender = true;
      const spec = e ? content.pick(x, y, proj) : null;
      const clean = spec ? Object.fromEntries(Object.entries(spec).filter(([k]) => !['cell', 'att', 'node'].includes(k))) : null;
      const key = clean ? JSON.stringify(clean) : '';
      if (key === hoverKey) return;
      hoverKey = key;
      renderer.domElement.style.cursor = clean ? 'pointer' : '';
      if (clean) { myHover = clean; store.set('hover', clean); }
      else if (myHover && store.state.hover === myHover) { myHover = null; store.set('hover', null); }
    }
    const cv = renderer.domElement;
    cv.addEventListener('pointermove', e => { if (!e.buttons) hoverAt(e); });
    cv.addEventListener('pointerleave', () => hoverAt(null));
    cv.addEventListener('pointerdown', e => { down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null; });
    cv.addEventListener('pointerup', e => {
      if (audience || !down || e.button !== 0) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5;
      down = null;
      if (moved || !content?.pick) return;
      const r = rect(), spec = content.pick(e.clientX - r.left, e.clientY - r.top, proj);
      const sel = content.click ? content.click(spec) : spec;
      const clean = sel ? Object.fromEntries(Object.entries(sel).filter(([k]) => ['kind', 'id'].includes(k))) : null;
      const now = store.state.sel;
      if ((now?.kind ?? null) === (clean?.kind ?? null) && now?.id === clean?.id) return;
      store.set('sel', clean && clean.id ? clean : null);
    });
    cv.addEventListener('contextmenu', e => e.preventDefault());

    // ================================================================ ctx.view while on
    const orig = ctx.view ? { nodeRect: ctx.view.nodeRect, contentRect: ctx.view.contentRect, fit: ctx.view.fit } : null;
    if (ctx.view) {
      ctx.view.nodeRect = id => {
        const p = content?.nodePos?.(id);
        if (!p || !size()) return null;
        camera.updateMatrixWorld();
        const q = proj(p, content.nodeR || NODE_R);
        return q ? { x: q.x - q.r, y: q.y - q.r, w: 2 * q.r, h: 2 * q.r } : null;
      };
      ctx.view.contentRect = () => {
        const pts = content?.points?.() || [];
        if (!pts.length || !size()) return { x: 0, y: 0, w: 0, h: 0 };
        camera.updateMatrixWorld();
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const p of pts) { const q = proj(p, 0); if (!q) continue; x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }
        return x0 < Infinity ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : { x: 0, y: 0, w: 0, h: 0 };
      };
      ctx.view.fit = (ms = 350) => { try { orig.fit?.(ms); } catch { /* 2D view */ } fit(ms); };
    }

    // ================================================================ loop
    let raf = 0, visible = document.body.dataset.view === 'nn', alive = true, lastT = 0, declT = -1e9, declDue = false;
    const camWas = new Float64Array(32);   // the camera's world and projection matrices at the last render
    function camChanged() {
      const a = camera.matrixWorld.elements, b = camera.projectionMatrix.elements;
      let ch = false;
      for (let i = 0; i < 16; i++) {
        if (Math.abs(a[i] - camWas[i]) > 1e-6) { camWas[i] = a[i]; ch = true; }
        if (Math.abs(b[i] - camWas[16 + i]) > 1e-7) { camWas[16 + i] = b[i]; ch = true; }
      }
      if (ch) camMoves++;
      return ch;
    }
    function loop(t) {
      raf = 0;
      if (!alive || !visible) return;
      const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
      lastT = t;
      size();
      let moving = false;
      if (flight) {
        const u = clamp((t - flight.t0) / flight.ms, 0, 1), e = flight.out ? 1 - (1 - u) ** 3 : u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
        camera.position.lerpVectors(flight.p0, flight.p1, e);
        controls.target.lerpVectors(flight.q0, flight.q1, e);
        if (u >= 1) { flight = null; camDirty = true; nudgeCards(); }
        moving = true;
      }
      if (audience && follow) {
        const k = 1 - Math.exp(-dt / 0.09);
        camera.position.lerp(follow.p, k);
        controls.target.lerp(follow.t, k);
        if (camera.position.distanceTo(follow.p) < 1e-3 && controls.target.distanceTo(follow.t) < 1e-3) { camera.position.copy(follow.p); controls.target.copy(follow.t); follow = null; nudgeCards(); }
        moving = true;
      }
      if (controls.update()) moving = true;
      if (needPaint && content?.paint) { needPaint = false; paints++; try { content.paint(); } catch (err) { console.error('[nn/view3d] paint:', err); } needRender = true; }
      if (content?.frame) { try { if (content.frame(t)) needRender = true; } catch (err) { console.error('[nn/view3d] frame:', err); } }
      if (content?.tensor?.animating()) relabel = true;   // the cubes' faces change size as they move
      if (moving || needRender) {
        needRender = false;
        updateFog();
        scene.updateMatrixWorld();
        renderer.render(scene, camera);
        labelR.render(scene, camera);
        // Labels are re-measured only when what they depend on changed: the camera (or the stage),
        // a header that opened or closed, a new build. Never on a repaint alone, so nothing shifts
        // while training.
        if (camChanged() || relabel) { relabel = false; declDue = true; }
      }
      // At most every 150 ms, and once more after the last change. A declutter that shows or hides
      // a label asks for a render (CSS2D applies visibility there).
      if (declDue && content?.declutter && t - declT > 150) {
        declT = t;
        declDue = false;
        try { if (content.declutter()) needRender = true; } catch { /* labels gone */ }
      }
      if (!audience && camDirty && !flight && (t - lastPubT > PUB_MS || !moving)) publish();
      raf = requestAnimationFrame(loop);
    }
    const kick = () => { if (!raf && alive && visible) raf = requestAnimationFrame(loop); };
    // Refit when the free area changes (a panel opens, folds or moves), unless the user moved: once
    // the panels have settled (140 ms), and only when the framing would change noticeably, so a
    // panel that grows a line while training doesn't nudge the camera.
    let areaKey = '', areaT = 0;
    const areaSoon = () => { if (!alive) return; placeBar(); clearTimeout(areaT); areaT = setTimeout(areaCheck, 140); };
    const areaObs = new ResizeObserver(areaSoon);
    const samePose = pose => {
      const D = camera.position.distanceTo(controls.target);
      return !!off && Math.abs(pose.D - D) < 0.035 * D && Math.abs(pose.off.cx - off.cx) < 10 && Math.abs(pose.off.cy - off.cy) < 10 && pose.t.distanceTo(controls.target) < 0.02 * D;
    };
    function areaCheck() {
      if (!alive || !size()) return;
      placeBar();
      const k = [...stage.children].filter(c => c !== root).map(c => { const r = c.getBoundingClientRect(); return [r.left, r.top, r.width, r.height].map(Math.round).join(','); }).join('|') + `|${SW}x${SH}`;
      if (k === areaKey) return;
      const first = !areaKey;
      areaKey = k;
      if (!first && !userMoved && !audience) {
        const pose = flight ? null : fitPose(null);
        if (!pose || !samePose(pose)) fit(300);
      } else if (!first) {
        if (audience && lastIn) follow = camFrom(lastIn);   // the same view, framed for this window
        else { reOffset(); camDirty = true; }   // k changed with the free area: tell the audience
        needRender = true;
      }
    }
    const watch = () => { areaObs.observe(root); for (const c of stage.children) if (c !== root) areaObs.observe(c); };
    const mo = new MutationObserver(watch);
    mo.observe(stage, { childList: true });
    watch();
    const onUp = () => requestAnimationFrame(areaSoon);   // a panel dragged (not resized) moves the free area too
    stage.addEventListener('pointerup', onUp, true);

    retheme();
    size();
    kick();

    function dispose() {
      alive = false;
      setPlay(false);
      cancelAnimationFrame(raf);
      clearTimeout(areaT);
      fade?.cancel();
      areaObs.disconnect();
      mo.disconnect();
      stage.removeEventListener('pointerup', onUp, true);
      if (myHover && store.state.hover === myHover) store.set('hover', null);
      content?.dispose();
      content = null;
      controls.dispose();
      for (const g of Object.values(GEO)) g.dispose();
      for (const m of Object.values(MAT)) m.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      root.remove();
      if (svg) svg.style.visibility = '';
      if (ctx.view && orig) Object.assign(ctx.view, orig);
      requestAnimationFrame(nudgeCards);   // the cards go back beside the 2D neurons
    }

    return {
      get built() { return built && !!content; },
      apply, onNet, dispose, fit: ms => fit(ms ?? 400),
      step(dir) { setPlay(false); step(dir); },
      invalidate() { needPaint = true; kick(); },
      retheme() { retheme(); needPaint = true; kick(); },
      shown(on) { visible = on; if (on) { needPaint = true; needRender = true; kick(); } },
      info: () => ({
        mode: cur()?.mode, empty: content?.empty || null, camera: pubCam(), off, W: SW, H: SH,
        tensor: content?.tensor ? { shown: content.tensor.shown, steps: content.tensor.steps.map(s => s.key), src: content.tensor.src } : null,
        labels: content?.labs?.length ?? 0, builds, paints,
      }),
    };
  }
}
