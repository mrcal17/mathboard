// Shared state + event bus for the Net tab. See docs/NN_CONTRACT.md.
//
// store.net keeps its identity for the life of the page: undo, redo and load replace its
// contents in place, so modules may hold on to the store but must read store.net fresh.

import * as model from './model.js';

export const POS = '#4aa3ff';
export const NEG = '#ff7a59';
export const HI = '#ffd54a';
const RGB = {
  dark: { pos: [74, 163, 255], neg: [255, 122, 89] },
  light: { pos: [31, 111, 209], neg: [216, 82, 44] },
};

// Diverging colour for a signed value: blue positive, orange negative, opacity by |v| / max.
export function colorFor(v, max = 1, theme = 'dark') {
  const c = RGB[theme] || RGB.dark;
  if (!Number.isFinite(v)) return 'rgba(128,128,128,0.4)';
  const [r, g, b] = v >= 0 ? c.pos : c.neg;
  const t = Math.min(1, Math.abs(v) / (max || 1));
  return `rgba(${r},${g},${b},${(0.12 + 0.88 * t).toFixed(3)})`;
}

const signature = net => [
  net.layers.map(l => l.id).join(','),
  net.nodes.map(n => n.id + ':' + n.layer).join(','),
  net.edges.map(e => e.id).join(','),
].join('|');

export function createStore(net, { undoLimit = 200 } = {}) {
  const handlers = new Map();
  const state = { sel: null, hover: null, anim: null, fwd: null, bwd: null };
  let undoStack = [], redoStack = [];
  let saved = JSON.stringify(net);   // the net as of the last commit: what undo returns to
  let sig = signature(net);

  function replace(next) {
    for (const k of Object.keys(net)) delete net[k];
    Object.assign(net, next);
  }

  function recompute() {
    try {
      state.fwd = model.forward(net);
      const outs = net.layers.length > 1 ? model.nodesIn(net, net.layers.length - 1) : [];
      const y = outs.map(n => n.target);
      state.bwd = outs.length && y.every(v => typeof v === 'number' && Number.isFinite(v))
        ? model.backward(net, state.fwd, y, net.meta?.loss || 'mse')
        : null;
    } catch (err) {
      console.error('[nn] recompute:', err);
      state.fwd = state.bwd = null;
    }
  }

  // Drop selection / hover that point at things that no longer exist.
  function prune() {
    const exists = t => !t || (t.kind === 'node' || t.kind === 'bias' ? model.node(net, t.id)
      : t.kind === 'edge' ? model.edge(net, t.id)
      : t.kind === 'layer' ? net.layers.some(l => l.id === t.id)
      : t.kind === 'token' ? (typeof t.layer === 'number' ? !!net.layers[t.layer] : net.layers.some(l => l.id === t.layer))
      : true);
    if (!exists(state.sel)) store.set('sel', null);
    if (!exists(state.hover)) store.set('hover', null);
  }

  function changed() {
    const s = signature(net);
    const structural = s !== sig;
    sig = s;
    recompute();
    if (structural) prune();
    store.emit('net', { structural });
    store.emit('values');
  }

  const store = {
    net, state, model,
    on(evt, fn) {
      if (!handlers.has(evt)) handlers.set(evt, new Set());
      handlers.get(evt).add(fn);
      return () => handlers.get(evt).delete(fn);
    },
    emit(evt, payload) {
      for (const fn of [...(handlers.get(evt) || [])]) {
        try { fn(payload, store); } catch (err) { console.error(`[nn] '${evt}' handler:`, err); }
      }
    },
    set(key, value) { state[key] = value; store.emit(key, value); },
    commit(label = '') {
      const now = JSON.stringify(net);
      if (now !== saved) {
        undoStack.push(saved);
        if (undoStack.length > undoLimit) undoStack.shift();
        redoStack = [];
        saved = now;
      }
      store.lastLabel = label;
      changed();
    },
    touch() { changed(); },
    layout() { store.emit('layout'); },
    undo() {
      const cur = JSON.stringify(net);
      if (cur !== saved) { redoStack.push(cur); replace(JSON.parse(saved)); changed(); return true; }
      if (!undoStack.length) return false;
      redoStack.push(saved);
      saved = undoStack.pop();
      replace(JSON.parse(saved));
      changed();
      return true;
    },
    redo() {
      if (!redoStack.length) return false;
      undoStack.push(saved);
      saved = redoStack.pop();
      replace(JSON.parse(saved));
      changed();
      return true;
    },
    // Swap in a whole new net (preset, import, audience sync). Undoable unless { history: false }.
    load(next, { history = true } = {}) {
      const fixed = model.normalize(typeof next === 'string' ? JSON.parse(next) : model.clone(next));
      if (history) { undoStack.push(saved); redoStack = []; } else { undoStack = []; redoStack = []; }
      replace(fixed);
      saved = JSON.stringify(net);
      changed();
    },
    get canUndo() { return undoStack.length > 0 || JSON.stringify(net) !== saved; },
    get canRedo() { return redoStack.length > 0; },
  };

  recompute();
  return store;
}
