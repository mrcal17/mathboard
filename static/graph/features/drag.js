// Drag vector tips and points in the 3D view, like Desmos. Only rows whose value is a plain literal
// are draggable: `u = (1, 2, 3)`, `(1, 2, 3)`, `v = [1, 0, 2] @ u`, `P = point(1, 2, 3)`.
// Drag moves the tip (origin + v) in the horizontal plane through it; Shift moves it along z.
// Moved components snap to integers when close (Alt = free); the row text is rewritten live.
import { formatNumber } from '../lang.js';

const NAME = '[A-Za-z\\u0370-\\u03FF][A-Za-z0-9_\\u0370-\\u03FF]*';
const NUM = '[-+\\u2212]?\\s*(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:[eE][-+]?\\d+)?';
const LITERAL_RE = new RegExp(`^(\\s*(?:${NAME}\\s*=\\s*)?)(point\\s*\\(|\\(|\\[)\\s*(${NUM})\\s*,\\s*(${NUM})\\s*` +
  `(?:,\\s*(${NUM})\\s*)?([)\\]])(\\s*(?:@[\\s\\S]*)?)$`);
const MIN_COS = 0.02;     // ignore a drag plane seen almost edge-on
const RING = [[0, 0], [7, 0], [-7, 0], [0, 7], [0, -7], [5, 5], [-5, 5], [5, -5], [-5, -5]]; // px offsets tried
const TIP_PX = 9;         // screen distance from a tip that grabs it

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const same = (a, b) => Array.isArray(b) && a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) < 1e-9);
const numberOf = tok => parseFloat(tok.replace(/\s+/g, '').replace('−', '-'));

// ---------------------------------------------------------------- pure helpers (tested)

// A row that is a plain vector/point literal -> its pieces, else null.
export function parseLiteral(src) {
  const s = String(src ?? '');
  const c = s.search(/#|\/\//);
  const code = c < 0 ? s : s.slice(0, c);
  const m = LITERAL_RE.exec(code);
  if (!m) return null;
  const [, prefix, open, a, b, z, close, suffix] = m;
  if ((open === '[') !== (close === ']')) return null;
  const tokens = [a, b, z].filter(t => t != null);
  const lit = code.slice(prefix.length, code.length - suffix.length);
  return {
    kind: open === '(' || open === '[' ? 'vec' : 'point',
    prefix, open, close, suffix, comment: c < 0 ? '' : s.slice(c),
    tokens, values: tokens.map(numberOf), sep: /,\s/.test(lit) ? ', ' : ',',
  };
}

// Same row with new components; unchanged components keep their original text.
export function rewriteSource(src, comps, fmt = formatNumber) {
  const p = parseLiteral(src);
  if (!p) return null;
  const n = p.tokens.length === 2 && comps[2] !== 0 ? 3 : p.tokens.length;
  const texts = comps.slice(0, n).map((x, i) => (i < p.tokens.length && x === p.values[i] ? p.tokens[i].trim() : fmt(x)));
  return `${p.prefix}${p.open}${texts.join(p.sep)}${p.close}${p.suffix}${p.comment}`;
}

// Ray origin + t dir (t > 0) against the plane through `point` with `normal`.
export function intersectRayPlane(origin, dir, point, normal, minCos = 1e-9) {
  const den = dot(dir, normal);
  if (Math.abs(den) <= minCos * Math.sqrt(dot(dir, dir) * dot(normal, normal))) return null;
  const t = dot(sub(point, origin), normal) / den;
  if (!(t > 0)) return null;
  return [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]];
}

// Point of the vertical line through `tip` closest to the ray (only z changes).
export function closestOnVertical(origin, dir, tip) {
  const w = sub(tip, origin), b = dir[2], c = dot(dir, dir), e = dot(dir, w);
  const den = c - b * b; // |dir|^2 sin^2(angle between the ray and z)
  if (den <= 1e-6 * c) return null;
  const s = (b * e - c * w[2]) / den, t = (e - b * w[2]) / den;
  if (!(t > 0)) return null;
  return [tip[0], tip[1], tip[2] + s];
}

// Snap to the nearest integer within `threshold`, else round to `step` (a power of ten).
export function snapValue(x, threshold, step) {
  const r = Math.round(x);
  if (threshold > 0 && Math.abs(x - r) <= threshold) return r + 0;
  if (!(step > 0)) return x;
  const digits = Math.max(0, Math.min(10, -Math.floor(Math.log10(step) + 1e-9)));
  return Number((Math.round(x / step) * step).toFixed(digits)) + 0;
}

// Both scale with the size of a screen pixel (in world units) at the tip.
export const snapThreshold = px => Math.min(0.3, 10 * px);
// Parsed from '1e<k>' because 10 ** -3 is not exactly 0.001 on every V8 build.
export const roundStep = px => Number(`1e${Math.max(-4, Math.min(0, Math.floor(Math.log10(3 * px))))}`);

// New components of v after moving the tip (origin + v) to `tip` along `axes` (clamped to +-limit).
export function moveComponents(v, o, tip, axes, threshold, step, limit = Infinity) {
  const next = v.slice();
  for (const k of axes) {
    const t = Math.max(-limit, Math.min(limit, tip[k]));
    next[k] = snapValue(t - o[k], threshold, step);
  }
  return next;
}

// ---------------------------------------------------------------- UI
const CSS = `
#g-view canvas.g-drag-hover { cursor: grab; }
html.g-dragging, html.g-dragging * { cursor: grabbing !important; user-select: none; }
.g-drag-readout { position: absolute; z-index: 6; pointer-events: none; padding: 3px 8px 3px 6px;
  border-left: 3px solid var(--c, var(--accent)); border-radius: 4px; background: var(--ui-bg); color: var(--ui-fg);
  box-shadow: 0 1px 5px rgba(0, 0, 0, 0.25); font: 12px/1.4 ui-monospace, Consolas, monospace; white-space: nowrap; }
.g-drag-readout[hidden] { display: none; }
.g-drag-readout small { display: block; color: var(--ui-muted); font-size: 11px; }
`;

export async function install(api) {
  if (api.params.has('audience')) return; // the audience window only mirrors the presenter
  const { THREE, GEO } = await import('../scene.js');
  api.addStyles(CSS);
  api.onSceneReady(scene => setup(api, scene, THREE, GEO));
}

function setup(api, scene, THREE, GEO) {
  const { canvas, controls } = scene;
  const root = document.documentElement;
  const halo = new THREE.Mesh(GEO.sphere, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false }));
  halo.visible = false;
  halo.renderOrder = 5;
  scene.scene.add(halo);
  const readout = document.createElement('div');
  readout.className = 'g-drag-readout';
  readout.hidden = true;
  api.addOverlay(readout);
  const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();

  let drag = null;                 // { rowId, pointerId, controlsWere, pending }
  let hoverId = null, hoverDirty = false, altArmed = false; // hover is re-picked after every recompute
  const pointer = { x: 0, y: 0, shift: false, alt: false, inside: false, buttons: 0 };

  // Current state of a draggable row (fresh from api.rows/results), or null.
  function handleInfo(rowId) {
    const i = api.rows.findIndex(r => r.id === rowId);
    const row = api.rows[i], res = api.results[i];
    if (!row || row.hidden || !res || res.error || !res.value) return null;
    const lit = parseLiteral(row.src);
    if (!lit || lit.kind !== res.value.type) return null;
    const o = res.origin || [0, 0, 0], v = res.value.v;
    return { row, lit, name: res.name, o, v, tip: add(o, v) };
  }

  // The handle for a drawn item, if it is a literal row's own vec/point.
  function handleFor(it) {
    if (it.kind !== 'vec' && it.kind !== 'point') return null;
    const h = handleInfo(it.rowId);
    return h && h.lit.kind === it.kind && same(h.v, it.v) && same(h.o, it.o || [0, 0, 0]) ? h : null;
  }
  // pick filter: only arrow heads and dots of handles count, so planes in front are skipped
  const onHead = (it, object) => (object.geometry === GEO.cone || object.geometry === GEO.sphere) && !!handleFor(it);

  function findHandle(x, y) {
    if (!api.rows.some(r => !r.hidden && parseLiteral(r.src))) return null;
    for (const [dx, dy] of RING) {
      const hit = scene.pick(x + dx, y + dy, onHead);
      if (hit) return handleFor(hit.item);
    }
    // The apex of a head is too thin to hit reliably: also accept a drawn tip within TIP_PX.
    const r = canvas.getBoundingClientRect(), p = new THREE.Vector3();
    let best = null, bestD = TIP_PX;
    for (const it of scene.items) {
      const h = handleFor(it);
      if (!h) continue;
      p.set(...h.tip).project(scene.camera);
      if (Math.abs(p.z) > 1) continue;
      const d = Math.hypot(r.left + ((p.x + 1) / 2) * r.width - x, r.top + ((1 - p.y) / 2) * r.height - y);
      if (d < bestD) { best = h; bestD = d; }
    }
    return best;
  }

  function rayAt(x, y) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, scene.camera);
    const { origin, direction } = raycaster.ray;
    const o = origin.clone();
    if (scene.camera.isOrthographicCamera) o.addScaledVector(direction, -1e4); // ortho sees behind its plane
    return [o.toArray(), direction.toArray()];
  }

  function worldPerPixel(tip) {
    const cam = scene.camera, h = canvas.clientHeight || 1;
    if (cam.isOrthographicCamera) return (cam.top - cam.bottom) / cam.zoom / h;
    const depth = new THREE.Vector3(...tip).sub(cam.position).dot(cam.getWorldDirection(new THREE.Vector3()));
    return (2 * Math.max(depth, 1e-3) * Math.tan((cam.fov * Math.PI) / 360)) / cam.zoom / h;
  }

  function showReadout(h, v) {
    const fmt = a => `(${a.map(formatNumber).join(', ')})`;
    readout.textContent = `${h.name ? `${h.name} = ` : ''}${fmt(v)}`;
    if (h.o.some(x => x !== 0)) {
      const tip = document.createElement('small');
      tip.textContent = `tip ${fmt(add(h.o, v))}`;
      readout.appendChild(tip);
    }
    readout.style.setProperty('--c', h.row.color);
    readout.hidden = false;
    const vr = api.viewEl.getBoundingClientRect();
    let left = pointer.x - vr.left + 16, top = pointer.y - vr.top + 14;
    if (left + readout.offsetWidth > vr.width - 4) left = pointer.x - vr.left - readout.offsetWidth - 12;
    if (top + readout.offsetHeight > vr.height - 4) top = pointer.y - vr.top - readout.offsetHeight - 10;
    readout.style.left = `${Math.max(2, left)}px`;
    readout.style.top = `${Math.max(2, top)}px`;
  }

  function moveDrag() {
    const h = handleInfo(drag.rowId);
    if (!h) { endDrag(false); return; }
    const vertical = pointer.shift && !scene.is2D;
    const [ro, rd] = rayAt(pointer.x, pointer.y);
    const tip = vertical ? closestOnVertical(ro, rd, h.tip) : intersectRayPlane(ro, rd, h.tip, [0, 0, 1], MIN_COS);
    if (!tip) return;
    const px = worldPerPixel(h.tip);
    const v = moveComponents(h.v, h.o, tip, vertical ? [2] : [0, 1],
      pointer.alt ? 0 : snapThreshold(px), roundStep(px), 3 * scene.extent);
    const src = rewriteSource(h.row.src, v);
    if (src && src !== h.row.src) api.setRowSource(h.row, src);
    showReadout(h, v);
  }

  function endDrag(apply) {
    if (!drag) return;
    if (apply && drag.pending) {
      drag.pending = false;
      moveDrag();
      if (!drag) return;
    }
    const d = drag;
    drag = null;
    controls.enabled = d.controlsWere;
    try { if (canvas.hasPointerCapture(d.pointerId)) canvas.releasePointerCapture(d.pointerId); } catch { /* gone */ }
    root.classList.remove('g-dragging');
    readout.hidden = true;
    hoverDirty = true;
  }

  const setPointer = e => {
    Object.assign(pointer, { x: e.clientX, y: e.clientY, shift: e.shiftKey, alt: e.altKey, buttons: e.buttons });
    if (drag && e.altKey) altArmed = true;
  };

  // Capture phase on the view, so this runs before OrbitControls' listener on the canvas.
  api.viewEl.addEventListener('pointerdown', e => {
    if (e.target !== canvas || drag || api.view !== 'graph') return;
    if (e.button !== 0 || !e.isPrimary || e.ctrlKey || e.metaKey) return;
    setPointer(e);
    const h = findHandle(e.clientX, e.clientY);
    if (!h) return;
    e.preventDefault();
    drag = { rowId: h.row.id, pointerId: e.pointerId, controlsWere: controls.enabled, pending: false };
    controls.enabled = false;
    altArmed = e.altKey;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    root.classList.add('g-dragging');
    canvas.classList.remove('g-drag-hover');
    hoverId = null;
    showReadout(h, h.v);
  }, true);

  canvas.addEventListener('pointermove', e => {
    setPointer(e);
    pointer.inside = true;
    if (drag) { if (e.pointerId === drag.pointerId) drag.pending = true; }
    else hoverDirty = true;
  });
  canvas.addEventListener('pointerleave', () => { pointer.inside = false; if (!drag) hoverDirty = true; });
  const finish = e => { if (drag && e.pointerId === drag.pointerId) endDrag(e.type === 'pointerup'); };
  canvas.addEventListener('pointerup', e => { setPointer(e); hoverDirty = true; finish(e); });
  canvas.addEventListener('pointercancel', finish);
  canvas.addEventListener('lostpointercapture', finish);

  // Shift / Alt pressed or released mid-drag re-apply at the current cursor.
  const onKey = e => {
    if (e.key === 'Alt' && (drag || altArmed)) { e.preventDefault(); altArmed = e.type === 'keydown'; } // no browser menu
    if (!drag || (e.key !== 'Shift' && e.key !== 'Alt')) return;
    pointer.shift = e.shiftKey;
    pointer.alt = e.altKey;
    drag.pending = true;
  };
  window.addEventListener('keydown', onKey, true);
  window.addEventListener('keyup', onKey, true);

  api.onRecompute(() => { hoverDirty = true; });
  api.onViewChange(view => { if (view !== 'graph') { endDrag(false); hoverId = null; halo.visible = false; } });

  scene.onFrame(() => {
    if (drag?.pending) { drag.pending = false; moveDrag(); }
    if (!drag && hoverDirty) {
      hoverDirty = false;
      const h = pointer.inside && !pointer.buttons ? findHandle(pointer.x, pointer.y) : null;
      hoverId = h ? h.row.id : null;
      canvas.classList.toggle('g-drag-hover', !!h);
    }
    const h = handleInfo(drag ? drag.rowId : hoverId);
    halo.visible = !!h;
    if (!h) return;
    halo.position.set(...h.tip);
    halo.scale.setScalar(0.22 * (scene.extent / 6));
    halo.material.color.set(h.row.color);
    halo.material.opacity = drag ? 0.3 : 0.2;
  });
}
