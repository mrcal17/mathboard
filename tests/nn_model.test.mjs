// Tests for static/nn/model.js (Net tab maths + edits). Run: node --test tests/nn_model.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../static/nn/model.js';
import { createStore } from '../static/nn/store.js';

// ---------------------------------------------------------------- helpers

function near(a, b, tol = 1e-7, msg = '') {
  assert.ok(Number.isFinite(a) && Number.isFinite(b), `${msg} non-finite: ${a} vs ${b}`);
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${msg} ${a} vs ${b}`);
}
const nearV = (u, v, tol, msg = '') => {
  assert.equal(u.length, v.length, `${msg} length`);
  u.forEach((x, i) => near(x, v[i], tol, `${msg}[${i}]`));
};
const valid = (net, msg = '') => assert.deepEqual(M.validate(net), [], msg);
const last = net => net.layers.length - 1;
const targets = net => M.nodesIn(net, last(net)).map(n => n.target);
const ids = arr => arr.map(o => o.id);
const allFinite = v => (Array.isArray(v) ? v.every(allFinite) : v === null || Number.isFinite(v));

// Loss as a function of the current net state (inputs come from layer-0 node values).
const lossOf = (net, y, loss) => M.backward(net, M.forward(net), y, loss).loss;

// Central finite difference of the loss with respect to one scalar field.
function fd(net, y, loss, get, set, eps = 1e-5) {
  const v0 = get();
  set(v0 + eps); const lp = lossOf(net, y, loss);
  set(v0 - eps); const lm = lossOf(net, y, loss);
  set(v0);
  return (lp - lm) / (2 * eps);
}

// Check every edge weight, every bias (= dz) and every input value (= dA[0]) against finite
// differences, plus internal consistency of dW / db / dZ.
function gradCheck(net, y, loss, tol = 2e-7) {
  const fwd = M.forward(net);
  const bwd = M.backward(net, fwd, y, loss);
  assert.ok(Number.isFinite(bwd.loss), 'loss finite');
  for (const e of net.edges) {
    near(bwd.edge[e.id], fd(net, y, loss, () => e.w, v => { e.w = v; }), tol, `dL/dw ${e.id}`);
  }
  for (const n of net.nodes) {
    const l = M.nodeLayerIndex(net, n.id);
    if (l === 0) near(bwd.node[n.id].da, fd(net, y, loss, () => n.value, v => { n.value = v; }), tol, `dL/dx ${n.id}`);
    else near(bwd.node[n.id].dz, fd(net, y, loss, () => n.bias, v => { n.bias = v; }), tol, `dL/db ${n.id}`);
  }
  // dW is the edge gradient laid out like matrices(); masked entries are exactly 0.
  const mats = M.matrices(net);
  for (const m of mats) {
    assert.equal(bwd.dW[m.l].length, m.terms.length, `dW[${m.l}] terms`);
    m.terms.forEach((t, ti) => t.edge.forEach((row, i) => row.forEach((eid, j) => {
      const g = bwd.dW[m.l][ti][i][j];
      if (eid === null) assert.equal(g, 0, `masked dW[${m.l}][${ti}][${i}][${j}]`);
      else assert.equal(g, bwd.edge[eid]);
    })));
    assert.deepEqual(bwd.db[m.l], bwd.dZ[m.l]);
  }
  // Elementwise layers: dZ = dA * f'(z).
  net.layers.forEach((layer, l) => {
    if (l === 0 || layer.act === 'softmax' || (l === last(net) && loss === 'xent' && layer.act === 'sigmoid')) return;
    const df = M.ACTS[layer.act].df;
    bwd.dZ[l].forEach((dz, i) => near(dz, bwd.dA[l][i] * df(fwd.z[l][i], fwd.a[l][i]), 1e-12, `dZ[${l}][${i}]`));
  });
  return bwd;
}

// Nudge relu/leaky biases off the kink so finite differences are valid, and make sure each
// such layer has live units (otherwise the check would be vacuous).
function avoidKinks(net) {
  for (let pass = 0; pass < 20; pass++) {
    const f = M.forward(net);
    let moved = false;
    net.layers.forEach((layer, l) => {
      if (l === 0 || !['relu', 'leaky'].includes(layer.act)) return;
      const ns = M.nodesIn(net, l);
      if (!ns.some(n => f.node[n.id].z > 0.05)) { ns.forEach(n => { n.bias += 0.5; }); moved = true; return; }
      for (const n of ns) if (Math.abs(f.node[n.id].z) < 0.05) { n.bias += 0.2; moved = true; }
    });
    if (!moved) return net;
  }
  throw new Error('could not move off relu kinks');
}

// 3 inputs -> h1 (4) -> h2 (3) -> outputs, with skip edges and masked (missing) entries.
function gradNet({ hidden = 'tanh', hidden2 = hidden, out = 'identity', outSize = 2, seed = 7, loss = 'mse', y } = {}) {
  const net = M.emptyNet();
  net.meta.loss = loss;
  const [lin, lout] = ids(net.layers);
  const h1 = M.addLayer(net, 1, { act: hidden, size: 4 });
  const h2 = M.addLayer(net, 2, { act: hidden2, size: 3 });
  M.setLayer(net, lout, { act: out });
  [0.7, -1.1, 0.4].forEach(value => M.addNode(net, lin, { value }));
  for (let j = 0; j < outSize; j++) M.addNode(net, lout, {});
  M.connectDense(net, lin, h1);
  M.connectDense(net, h1, h2);
  M.connectDense(net, h2, lout);
  const X = M.nodesIn(net, 0), H1 = M.nodesIn(net, 1), H2 = M.nodesIn(net, 2), O = M.nodesIn(net, 3);
  // skip edges: input -> h2, input -> out, h1 -> out
  M.connect(net, X[0].id, H2[0].id); M.connect(net, X[2].id, H2[2].id);
  M.connect(net, X[1].id, O[0].id); M.connect(net, H1[3].id, O[outSize - 1].id);
  // masked entries
  M.disconnect(net, M.edgeBetween(net, X[1].id, H1[0].id).id);
  M.disconnect(net, M.edgeBetween(net, H1[2].id, H2[1].id).id);
  M.disconnect(net, M.edgeBetween(net, H2[0].id, O[outSize - 1].id).id);
  M.randomize(net, { seed, scheme: 'he', biases: 'small' });
  const ty = y || O.map((_, i) => [0.3, -0.6, 0.9][i % 3]);
  O.forEach((n, i) => { n.target = ty[i]; });
  avoidKinks(net);
  valid(net, 'gradNet');
  return net;
}

// Sanity: the gradient test net really has skip terms and masked entries.
function assertSkipAndMask(net) {
  const mats = M.matrices(net);
  assert.ok(mats.some(m => m.terms.length > 1), 'has a skip term');
  assert.ok(mats.some(m => m.terms[0].edge.some(r => r.includes(null))), 'has a masked entry');
}

function accuracy(net, X, Y) {
  const P = M.predict(net, X);
  let ok = 0;
  P.forEach((p, s) => {
    if (p.length === 1) ok += (p[0] > 0.5) === (Y[s][0] > 0.5);
    else ok += p.indexOf(Math.max(...p)) === Y[s].indexOf(Math.max(...Y[s]));
  });
  return ok / X.length;
}

// ---------------------------------------------------------------- basics

describe('rng / fmt / ACTS', () => {
  test('rng is deterministic, in [0, 1), and seeds differ', () => {
    const a = M.rng(42), b = M.rng(42), c = M.rng(43);
    const xs = Array.from({ length: 1000 }, () => a());
    assert.deepEqual(xs, Array.from({ length: 1000 }, () => b()));
    assert.ok(xs.every(x => x >= 0 && x < 1));
    assert.notDeepEqual(xs.slice(0, 5), Array.from({ length: 5 }, () => c()));
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    assert.ok(Math.abs(mean - 0.5) < 0.05);
    assert.equal(M.rng('abc')(), M.rng('abc')());
  });

  test('fmt: fixed digits, ASCII minus, no negative zero', () => {
    assert.equal(M.fmt(1.234), '1.23');
    assert.equal(M.fmt(-1.5, 1), '-1.5');
    assert.equal(M.fmt(2, 0), '2');
    assert.equal(M.fmt(-0.001), '0.00');
    assert.equal(M.fmt(-0.4, 0), '0');
    assert.equal(M.fmt(-0), '0.00');
    assert.equal(M.fmt(-12.345, 3), '-12.345');
    assert.equal(M.fmt(NaN), 'NaN');
    assert.equal(M.fmt(Infinity), 'inf');
    assert.equal(M.fmt(-Infinity), '-inf');
    assert.ok(!M.fmt(-3.2).includes('−'));
  });

  test('fmtg: fmt from 0.01 up, two significant figures below, rounding noise reads 0', () => {
    assert.equal(M.fmtg(0.123), '0.12');
    assert.equal(M.fmtg(-0.01), '-0.01');
    assert.equal(M.fmtg(0.0123, 3), '0.012');
    assert.equal(M.fmtg(0.00345), '0.0034');
    assert.equal(M.fmtg(-0.0072), '-0.0072');
    assert.equal(M.fmtg(0.000346), '3.5e-4');
    assert.equal(M.fmtg(-7.61e-5), '-7.6e-5');
    assert.equal(M.fmtg(8.7e-19), '0.00');
    assert.equal(M.fmtg(0, 3), '0.000');
    assert.equal(M.fmtg(NaN), 'NaN');
    assert.equal(M.fmtg(-Infinity), '-inf');
  });

  test('ACTS has the six contract activations with label, tex, f, df', () => {
    assert.deepEqual(Object.keys(M.ACTS).sort(), ['identity', 'leaky', 'relu', 'sigmoid', 'softmax', 'tanh']);
    for (const [k, a] of Object.entries(M.ACTS)) {
      assert.equal(typeof a.label, 'string', k);
      assert.equal(typeof a.tex, 'string', k);
      assert.equal(typeof a.f, 'function', k);
      assert.equal(typeof a.df, 'function', k);
    }
    assert.equal(M.ACTS.softmax.vector, true);
  });

  test('scalar df matches finite differences of f (with and without a)', () => {
    for (const k of ['identity', 'relu', 'leaky', 'sigmoid', 'tanh']) {
      const { f, df } = M.ACTS[k];
      for (const z of [-3.1, -0.7, -0.05, 0.08, 0.9, 2.6]) {
        const num = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6;
        near(df(z, f(z)), num, 1e-6, `${k}'(${z})`);
        near(df(z), num, 1e-6, `${k}'(${z}) no a`);
      }
    }
    assert.equal(M.ACTS.relu.f(-2), 0);
    assert.equal(M.ACTS.leaky.f(-2), -0.2);
    near(M.ACTS.sigmoid.f(-800), 0, 1e-12);
    near(M.ACTS.sigmoid.f(800), 1, 1e-12);
  });

  test('activate: softmax sums to 1 and survives huge logits', () => {
    const p = M.activate('softmax', [1000, 1001, 999]);
    near(p.reduce((s, v) => s + v, 0), 1, 1e-12);
    assert.ok(p[1] > p[0] && p[0] > p[2]);
    assert.deepEqual(M.activate('relu', [-1, 2]), [0, 2]);
  });
});

// ---------------------------------------------------------------- construction + lookups

describe('ids, emptyNet, clone, lookups', () => {
  test('emptyNet is valid with an input and an output layer', () => {
    const net = M.emptyNet();
    valid(net);
    assert.equal(net.layers.length, 2);
    assert.equal(net.nodes.length, 0);
    assert.equal(net.meta.loss, 'mse');
    assert.ok(M.forward(net));
    assert.equal(M.collapse(net).W.length, 0);
  });

  test('uid is unique and never reused after removal', () => {
    const net = M.emptyNet();
    const seen = new Set(ids(net.layers));
    const a = M.addNode(net, 0, {});
    assert.ok(!seen.has(a)); seen.add(a);
    M.removeNode(net, a);
    for (let i = 0; i < 50; i++) {
      const id = M.uid(net, 'n');
      assert.ok(!seen.has(id), id);
      seen.add(id);
    }
    const bare = { layers: [{ id: 'L9' }], nodes: [], edges: [] };
    assert.equal(M.uid(bare, 'x'), 'x10');
  });

  test('clone is deep and independent', () => {
    const net = M.PRESETS.xor.build(3);
    const c = M.clone(net);
    assert.deepEqual(c, net);
    c.edges[0].w = 99;
    assert.notEqual(net.edges[0].w, 99);
  });

  test('layerIndex / nodeLayerIndex / nodesIn / node / edge / edgeBetween', () => {
    const net = M.PRESETS.xor.build(1);
    const [l0, l1, l2] = ids(net.layers);
    assert.equal(M.layerIndex(net, l1), 1);
    assert.equal(M.layerIndex(net, 'nope'), -1);
    assert.deepEqual(ids(M.nodesIn(net, l1)), ids(M.nodesIn(net, 1)));
    assert.equal(M.nodesIn(net, 1).length, 4);
    assert.deepEqual(M.nodesIn(net, 9), []);
    const x = M.nodesIn(net, l0)[0], h = M.nodesIn(net, 1)[2];
    assert.equal(M.nodeLayerIndex(net, h.id), 1);
    assert.equal(M.nodeLayerIndex(net, 'ghost'), -1);
    assert.equal(M.node(net, x.id), x);
    assert.equal(M.node(net, 'ghost'), null);
    const e = M.edgeBetween(net, x.id, h.id);
    assert.ok(e && e.from === x.id && e.to === h.id);
    assert.equal(M.edgeBetween(net, h.id, x.id), e, 'either direction');
    assert.equal(M.edge(net, e.id), e);
    assert.equal(M.edge(net, 'ghost'), null);
    assert.equal(M.edgeBetween(net, x.id, M.nodesIn(net, l2)[0].id), null);
  });
});

// ---------------------------------------------------------------- presets + datasets

describe('PRESETS', () => {
  const KEYS = ['perceptron', 'xor', 'mlp', 'deep', 'autoencoder', 'residual', 'linear', 'classifier'];

  test('all contract presets exist', () => {
    for (const k of KEYS) {
      assert.ok(M.PRESETS[k], k);
      assert.equal(typeof M.PRESETS[k].label, 'string');
      assert.equal(typeof M.PRESETS[k].build, 'function');
    }
  });

  for (const k of KEYS) {
    test(`${k}: valid, deterministic, laid out, active, trainable targets`, () => {
      const net = M.PRESETS[k].build(5);
      valid(net);
      assert.deepEqual(M.PRESETS[k].build(5), net, 'same seed, same net');
      assert.notDeepEqual(M.PRESETS[k].build(6).edges.map(e => e.w), net.edges.map(e => e.w), 'seed matters');
      assert.deepEqual(M.normalize(M.clone(net)), net, 'normalize is a no-op on a preset');
      for (const n of net.nodes) {
        assert.ok(n.x >= 0 && n.x <= 900 && n.y >= 0 && n.y <= 560, `${n.id} in 900x560`);
        assert.ok(n.label.length > 0);
      }
      assert.ok(M.nodesIn(net, 0).every(n => n.value !== 0), 'inputs nonzero');
      assert.ok(targets(net).every(Number.isFinite), 'every output has a target');
      const f = M.forward(net);
      assert.ok(allFinite(f.a.flat()));
      for (let l = 1; l < net.layers.length; l++) {
        assert.ok(f.a[l].some(v => Math.abs(v) > 0.01), `layer ${l} shows activity`);
      }
      const b = M.backward(net, f, targets(net), net.meta.loss);
      assert.equal(b.note, null, 'loss suits the output layer');
      assert.ok(b.loss > 0 && Number.isFinite(b.loss));
      const ds = M.PRESETS[k].dataset;
      if (ds) {
        assert.equal(M.DATASETS[ds].inputs, M.nodesIn(net, 0).length, 'dataset inputs');
        assert.equal(M.DATASETS[ds].outputs, M.nodesIn(net, last(net)).length, 'dataset outputs');
      }
    });
  }

  test('specific shapes: residual skips, linear collapses, classifier is 2-4-3 softmax', () => {
    const res = M.PRESETS.residual.build(1);
    const mats = M.matrices(res);
    const sum = mats.find(m => m.terms.length === 2);
    assert.ok(sum, 'residual has a layer with a skip term');
    assert.deepEqual(sum.terms.map(t => t.k), [sum.l - 1, 0]);
    assert.deepEqual(sum.terms[1].W, [[1, 0], [0, 1]], 'identity shortcut');
    assert.equal(sum.terms[1].edge[0][1], null, 'off-diagonal masked');

    const lin = M.PRESETS.linear.build(1);
    assert.ok(lin.layers.every(l => l.act === 'identity'));
    assert.ok(M.collapse(lin));

    const cls = M.PRESETS.classifier.build(1);
    assert.deepEqual(cls.layers.map(l => M.nodesIn(cls, l.id).length), [2, 4, 3]);
    assert.equal(cls.layers[2].act, 'softmax');
    assert.equal(cls.meta.loss, 'xent');

    assert.equal(M.PRESETS.perceptron.build(1).layers.length, 2);
    assert.ok(M.PRESETS.deep.build(1).layers.length >= 5);
    const ae = M.PRESETS.autoencoder.build(1);
    const sizes = ae.layers.map(l => M.nodesIn(ae, l.id).length);
    assert.equal(sizes[0], sizes.at(-1));
    assert.ok(Math.min(...sizes) < sizes[0], 'bottleneck');
  });

  test('presets record their dataset in meta.train.dataset (the Train panel setting)', () => {
    for (const k of KEYS) {
      const p = M.PRESETS[k], net = p.build(3);
      if (p.dataset) {
        assert.ok(M.DATASETS[p.dataset], `${k}: known dataset`);
        assert.equal(net.meta.train.dataset, p.dataset, k);
      } else {
        assert.equal(net.meta.train.dataset, undefined, `${k}: no dataset recorded`);
      }
      assert.deepEqual(M.normalize(JSON.parse(JSON.stringify(net))).meta.train, net.meta.train, `${k}: survives a save`);
    }
  });

  test('store integration: every preset gets a forward and a backward pass', () => {
    for (const k of KEYS) {
      const store = createStore(M.PRESETS[k].build(1));
      assert.ok(store.state.fwd, k);
      assert.ok(store.state.bwd, k);
      assert.ok(Number.isFinite(store.state.bwd.loss));
    }
  });
});

describe('PRESETS: every preset in the menu', () => {
  const ALL = Object.keys(M.PRESETS);
  const GROUPS = ['Basics', 'MLPs', 'Skip connections', 'Structure in W', 'Sequences', 'Attention', 'Embeddings & autoencoders', 'Teaching demos'];
  const R = 26;   // view.js neuron radius

  test('menu: known groups, each in one run and in order; a one-line note each; unique titles', () => {
    const seen = [];
    for (const k of ALL) {
      const p = M.PRESETS[k];
      assert.ok(GROUPS.includes(p.group), `${k}: group ${p.group}`);
      if (seen.at(-1) !== p.group) {
        assert.ok(!seen.includes(p.group), `${k}: ${p.group} is split up`);
        seen.push(p.group);
      }
      assert.equal(typeof p.label, 'string');
      assert.equal(typeof p.note, 'string');
      assert.ok(p.note.length >= 40 && p.note.length <= 220, `${k}: note length ${p.note.length}`);
      assert.ok(!/[\n—]/.test(p.note), `${k}: one line, no em dash`);
      assert.ok(p.dataset === null || M.DATASETS[p.dataset], `${k}: dataset`);
    }
    assert.deepEqual(seen, GROUPS);
    const titles = ALL.map(k => M.PRESETS[k].build(1).meta.title);
    assert.equal(new Set(titles).size, titles.length, 'train.js matches legacy nets to presets by title');
  });

  for (const k of ALL) {
    test(`${k}: valid, deterministic, readable layout, finite forward and backward`, () => {
      const p = M.PRESETS[k], net = p.build(1);
      valid(net);
      assert.deepEqual(p.build(1), net, 'same seed, same net');
      assert.deepEqual(M.normalize(M.clone(net)), net, 'normalize is a no-op on a preset');
      // The tiny language model is made for the Flow view (docs/NN_FLOW.md), which draws it as matrices.
      assert.ok(net.nodes.length <= (k === 'tiny_lm' ? 180 : 40), `${net.nodes.length} nodes`);
      assert.ok(net.layers.every(l => l.name.trim()), 'layer names');
      assert.ok(net.nodes.every(n => n.label.trim()), 'labels');
      for (let i = 0; i < net.nodes.length; i++) {
        for (let j = i + 1; j < net.nodes.length; j++) {
          const a = net.nodes[i], b = net.nodes[j];
          assert.ok(Math.hypot(a.x - b.x, a.y - b.y) >= 2 * R + 24, `${a.id} and ${b.id} too close (value labels overlap)`);
        }
      }
      // A layer is one column, or (the 3-token attention presets) a tokens x d grid like the matrix
      // panel: a column per feature, a row per token. Each layer sits right of the one before.
      const ext = net.layers.map((_, l) => {
        const ns = M.nodesIn(net, l), xs = ns.map(n => n.x), { d } = M.tokenShape(net, l);
        if (!xs.every(x => x === xs[0])) {
          assert.ok(d > 1 && new Set(xs).size === d, `layer ${l} is one column, or one column per feature`);
          assert.ok(ns.every((n, k) => n.x === ns[k % d].x && n.y === ns[k - (k % d)].y), `layer ${l}: a column per feature, a row per token`);
        }
        return [Math.min(...xs), Math.max(...xs)];
      });
      ext.forEach(([x0], l) => assert.ok(l === 0 || x0 > ext[l - 1][1] + 2 * R, `layer ${l} is right of layer ${l - 1}`));

      const f = M.forward(net);
      assert.ok(allFinite(f.a) && allFinite(f.z), 'finite forward');
      assert.ok(targets(net).every(Number.isFinite), 'every output has a target');
      const b = M.backward(net, f, targets(net), net.meta.loss);
      assert.equal(b.note, null, 'the loss suits the output layer');
      assert.ok(Number.isFinite(b.loss) && b.loss >= 0);
      assert.ok(allFinite(b.dA) && allFinite(b.dZ));
      const store = createStore(M.clone(net));
      assert.ok(store.state.fwd && store.state.bwd, 'the store runs both passes');
      if (p.dataset) {
        assert.equal(net.meta.train.dataset, p.dataset);
        assert.equal(M.DATASETS[p.dataset].inputs, M.nodesIn(net, 0).length, 'dataset inputs');
        assert.equal(M.DATASETS[p.dataset].outputs, M.nodesIn(net, last(net)).length, 'dataset outputs');
      } else {
        assert.equal(net.meta.train.dataset, undefined);
      }
    });
  }
});

describe('hand-set presets compute what they claim', () => {
  const out = (net, x) => M.forward(net, x).a.at(-1);
  const BITS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const term = (net, l, k) => M.matrices(net)[l - 1].terms.find(t => t.k === k);
  const mask = t => t.edge.map(row => row.map(e => (e === null ? 0 : 1)));
  // Entry (i, j) of a term: its weight, or null where the edge is missing (masked).
  const entries = t => t.W.map((row, i) => row.map((w, j) => (t.edge[i][j] === null ? null : w)));
  const expect = (rows, cols, f) => Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => f(i, j)));

  test('gates: each output row is a truth table (AND, OR, NAND)', () => {
    const net = M.PRESETS.gates.build();
    for (const [a, b] of BITS) nearV(out(net, [a, b]), [a & b, a | b, 1 - (a & b)], 1e-3, `${a}${b}`);
  });

  test('xor_gates: OR and NAND in the hidden layer, XOR at the output, no training', () => {
    const net = M.PRESETS.xor_gates.build();
    for (const [a, b] of BITS) {
      const f = M.forward(net, [a, b]);
      nearV(f.a[1], [a | b, 1 - (a & b)], 1e-3, `hidden ${a}${b}`);
      near(f.a[2][0], a ^ b, 1e-3, `xor ${a}${b}`);
    }
  });

  test('xor_relu: exactly XOR; (0, 1) and (1, 0) land on the same hidden point', () => {
    const net = M.PRESETS.xor_relu.build();
    for (const [a, b] of BITS) assert.equal(out(net, [a, b])[0], a ^ b);
    assert.deepEqual(M.forward(net, [0, 1]).a[1], M.forward(net, [1, 0]).a[1]);
  });

  test('conv1d: banded Toeplitz W, the kernel moved one column per row, off-band masked', () => {
    const net = M.PRESETS.conv1d.build(), t = M.matrices(net)[0].terms[0], k = [-1, 2, -1];
    assert.deepEqual(entries(t), expect(6, 8, (i, j) => k[j - i] ?? null));
    for (let i = 1; i < 6; i++) assert.deepEqual(t.W[i].slice(i), t.W[0].slice(0, 8 - i), `row ${i} is row 0 shifted`);
    assert.deepEqual(M.forward(net).a[1], [-1, 1, 0, 0, 1, -1], 'the box lights up at its two edges');
  });

  test('conv1d_s2: rows move two columns, W is 4 x 9', () => {
    const net = M.PRESETS.conv1d_s2.build(), t = M.matrices(net)[0].terms[0], k = [0.25, 0.5, 0.25];
    assert.deepEqual(entries(t), expect(4, 9, (i, j) => k[j - 2 * i] ?? null));
    for (let i = 1; i < 4; i++) assert.deepEqual(t.W[i].slice(2 * i), t.W[0].slice(0, 9 - 2 * i));
  });

  test('avgpool: fixed half-half pairs', () => {
    const net = M.PRESETS.avgpool.build();
    assert.deepEqual(entries(M.matrices(net)[0].terms[0]), expect(4, 8, (i, j) => (Math.floor(j / 2) === i ? 0.5 : null)));
    nearV(out(net, [3, 1, -2, 4, 0.5, 0.5, 7, -7]), [2, 1, 0.5, 0], 1e-12);
  });

  test('maxpool: b + ReLU(a - b) is the max of each pair', () => {
    const net = M.PRESETS.maxpool.build(), r = M.rng(4);
    for (let s = 0; s < 50; s++) {
      const x = Array.from({ length: 4 }, () => 4 * r() - 2);
      nearV(out(net, x), [Math.max(x[0], x[1]), Math.max(x[2], x[3])], 1e-12);
    }
  });

  test('lenet: banded conv, fixed pool, dense head; the bump position picks the class', () => {
    const net = M.PRESETS.lenet.build(), m = M.matrices(net);
    const k = [0.5, 1, 0.5];
    assert.deepEqual(entries(m[0].terms[0]), expect(6, 8, (i, j) => k[j - i] ?? null));
    assert.equal(new Set(m[0].b).size, 1, 'one bias for the whole kernel');
    assert.deepEqual(entries(m[1].terms[0]), expect(3, 6, (i, j) => (Math.floor(j / 2) === i ? 0.5 : null)));
    assert.ok(m[2].terms[0].edge.flat().every(Boolean), 'the head is dense');
    assert.deepEqual(M.forward(net).a[2], [0, 1, 0], 'pooled features of the built-in bump');
    const bump = at => Array.from({ length: 8 }, (_, j) => (j === at || j === at + 1 ? 1 : 0));
    const cls = x => { const p = out(net, x); return p.indexOf(Math.max(...p)); };
    assert.deepEqual([cls(bump(0)), cls(bump(3)), cls(bump(6))], [0, 1, 2]);
  });

  test('gnn: each W is masked by A + I with weights 1 / (row count); two layers reach two hops', () => {
    const A = [[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]];
    const net = M.PRESETS.gnn.build();
    for (const m of M.matrices(net)) {
      assert.equal(m.terms.length, 1);
      assert.deepEqual(mask(m.terms[0]), A.map((row, i) => row.map((v, j) => (v || i === j ? 1 : 0))));
      m.terms[0].W.forEach((row, i) => {
        const count = A[i].reduce((s, v) => s + v, 1);
        row.forEach((w, j) => { if (m.terms[0].edge[i][j]) near(w, 1 / count, 1e-15); });
      });
      assert.ok(m.b.every(b => b === 0));
    }
    const a = M.forward(net).a;   // one-hot at node 1
    assert.ok(a[2][3] > 0, 'node 4 is two hops from node 1');
    assert.equal(a[2][4], 0, 'node 5 is three hops away');
  });

  test('rnn: every step shares one tied W_hh, x weights and bias; x_t only feeds step t', () => {
    const net = M.PRESETS.rnn.build(), T = 4;
    const same = (n, msg) => {
      const m = M.matrices(n), Whh = term(n, 2, 1).W, wx = term(n, 1, 0).W.map(row => row[0]);
      for (let t = 1; t <= T; t++) {
        const x = term(n, t, 0);
        x.edge.forEach(row => row.forEach((e, j) => assert.equal(e !== null, j === t - 1, `${msg} step ${t}, x_${j + 1}`)));
        assert.deepEqual(x.W.map(row => row[t - 1]), wx, msg);
        if (t > 1) assert.deepEqual(term(n, t, t - 1).W, Whh, msg);
        assert.deepEqual(m[t - 1].b, m[0].b, msg);
      }
      return { Whh, wx, b: m[0].b };
    };
    const before = same(net, 'built');
    assert.deepEqual(M.matrices(net)[T].terms.map(t => t.k), [T], 'the output reads the last step only');
    const tied = e => M.nodeLayerIndex(net, e.to) <= T;
    assert.ok(net.edges.filter(tied).every(e => /^(W_\{hh\}|w_x):\d,\d$/.test(e.tie)), 'every step edge is tied');
    // the X W convention: W_hh:j,i is the weight from h_j of step t-1 to h_i of step t
    const e = M.edgeBetween(net, M.nodesIn(net, 1)[1].id, M.nodesIn(net, 2)[0].id);
    assert.equal(e.tie, 'W_{hh}:2,1');
    assert.equal(M.tiedMatrices(net, 3).find(t => t.name === 'W_{hh}').W[1][0], e.w);
    const trained = M.clone(net);   // tied: a training step moves every copy together
    M.trainStep(trained, { X: [[0.5, -0.3, 0.9, 0.2]], Y: [[1]] }, { lr: 0.5 });
    valid(trained);
    const after = same(trained, 'trained');
    assert.notDeepEqual(after.Whh, before.Whh, 'W_hh moved');
    assert.notDeepEqual(after.wx, before.wx, 'w_x moved');
    assert.notDeepEqual(after.b, before.b, 'b moved');
  });

  test('wavenet: causal W with dilations 1, 2, 4; the last output is the mean of the inputs', () => {
    const net = M.PRESETS.wavenet.build(), m = M.matrices(net);
    [1, 2, 4].forEach((d, l) => {
      assert.deepEqual(entries(m[l].terms[0]), expect(8, 8, (i, j) => (j === i || j === i - d ? 0.5 : null)), `dilation ${d}`);
    });
    const r = M.rng(8);
    for (let s = 0; s < 10; s++) {
      const x = Array.from({ length: 8 }, () => 2 * r() - 1), y = out(net, x);
      near(y[7], x.reduce((p, q) => p + q, 0) / 8, 1e-12);
      const later = x.slice();
      later[7] += 5;
      assert.deepEqual(out(net, later).slice(0, 7), y.slice(0, 7), 'no output reads the future');
    }
  });

  test('embedding: a one-hot input picks a column of W1; W2 starts as its transpose', () => {
    const net = M.PRESETS.embedding.build(), [m1, m2] = M.matrices(net);
    const W1 = m1.terms[0].W;
    for (let k = 0; k < 5; k++) {
      const x = [0, 0, 0, 0, 0];
      x[k] = 1;
      assert.deepEqual(M.forward(net, x).a[1], W1.map(row => row[k]));
    }
    assert.deepEqual(m2.terms[0].W, W1[0].map((_, j) => W1.map(row => row[j])));
    assert.deepEqual(M.nodesIn(net, 0).map(n => n.label), M.nodesIn(net, 2).map(n => n.label));
  });

  test('uat: hinges at the knots; the output joins the sine\'s values at the knots', () => {
    const net = M.PRESETS.uat.build();
    assert.ok(M.matrices(net)[0].terms[0].W.every(row => row[0] === 1));
    for (let j = 0; j <= 10; j++) {
      const x = -1 + j / 5;
      near(out(net, [x])[0], 0.8 * Math.sin(Math.PI * x), 0.01, `knot ${x}`);
    }
    near(out(net, [0.1])[0], (out(net, [0])[0] + out(net, [0.2])[0]) / 2, 1e-9, 'linear between knots');
  });

  test('vanishing: dL/dW shrinks more than 3x per sigmoid layer toward the input', () => {
    const net = M.PRESETS.vanishing.build();
    const b = M.backward(net, M.forward(net), targets(net), 'mse');
    const g = net.layers.map((_, l) => (l ? Math.abs(b.dW[l][0][0][0]) : 0));
    for (let l = 1; l < last(net); l++) assert.ok(g[l] < g[l + 1] / 3, `layer ${l}: ${g[l]} vs ${g[l + 1]}`);
    assert.ok(g[1] < 2e-3 * g[last(net)]);
    assert.equal(M.fmt(g[1], 2), '0.00', 'the matrix panel shows the first layer\'s gradient as 0.00');
  });

  test('residual-style shortcuts are masked identities', () => {
    for (const [k, to, from] of [['residual', 2, 0], ['bottleneck', 4, 1], ['ffn', 2, 0], ['unet', 3, 1], ['unet', 4, 0]]) {
      const t = term(M.PRESETS[k].build(), to, from);
      assert.ok(t, `${k}: a term from layer ${from} into layer ${to}`);
      assert.deepEqual(entries(t), expect(t.W.length, t.W[0].length, (i, j) => (i === j ? 1 : null)), k);
    }
  });

  test('densenet: every layer has a dense term from every earlier layer', () => {
    for (const m of M.matrices(M.PRESETS.densenet.build())) {
      assert.deepEqual(m.terms.map(t => t.k), Array.from({ length: m.l }, (_, i) => m.l - 1 - i));
      assert.ok(m.terms.every(t => t.edge.flat().every(Boolean)));
    }
  });

  test('wide_deep: the output adds a deep term and a wide term straight from the inputs', () => {
    const net = M.PRESETS.wide_deep.build(), m = M.matrices(net).at(-1);
    assert.deepEqual(m.terms.map(t => t.k), [2, 0]);
    assert.ok(m.terms.every(t => t.edge.flat().every(Boolean)));
  });

  test('towers and multitask: block-diagonal masks where the towers and heads separate', () => {
    const [a, b, c] = M.matrices(M.PRESETS.towers.build());
    assert.deepEqual(mask(a.terms[0]), [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
    assert.deepEqual(mask(b.terms[0]), [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]);
    assert.ok(c.terms[0].edge.flat().every(Boolean), 'the output adds both towers');
    const [t1, t2, t3] = M.matrices(M.PRESETS.multitask.build());
    assert.ok([t1, t2].every(m => m.terms[0].edge.flat().every(Boolean)), 'the trunk and head inputs are dense');
    assert.deepEqual(mask(t3.terms[0]), [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]);
    assert.equal(t3.act, 'sigmoid');
  });

  test('convex models start at W = 0; wide and narrow_deep both have 49 parameters', () => {
    for (const k of ['logreg', 'linreg', 'softmax_reg']) {
      const net = M.PRESETS[k].build();
      assert.ok(net.edges.length && net.edges.every(e => e.w === 0), k);
      assert.ok(net.nodes.every(n => n.bias === 0), k);
    }
    const params = net => net.edges.length + net.nodes.filter(n => M.nodeLayerIndex(net, n.id) > 0).length;
    assert.equal(params(M.PRESETS.wide.build()), 49);
    assert.equal(params(M.PRESETS.narrow_deep.build()), 49);
  });
});

describe('presets with a dataset train on it', () => {
  const data = (p, n = 100, noise = 0) => M.DATASETS[p.dataset].make(n, 1, noise);
  function sgd(net, { X, Y }, steps, lr, batch) {
    const r = M.rng(3);
    for (let s = 0; s < steps; s++) {
      const idx = Array.from({ length: batch }, () => Math.floor(r() * X.length));
      M.trainStep(net, { X: idx.map(i => X[i]), Y: idx.map(i => Y[i]) }, { lr });
    }
    return net;
  }

  for (const [k, p] of Object.entries(M.PRESETS)) {
    if (!p.dataset) continue;
    test(`${k}: gradient descent on ${p.dataset} lowers the loss`, () => {
      const net = p.build(1), d = data(p);
      const before = M.trainStep(M.clone(net), d, { lr: 0 });
      let after = before;
      for (let s = 0; s < 150; s++) after = M.trainStep(net, d, { lr: 0.1 });
      assert.ok(after < before, `${before} -> ${after}`);
      valid(net);
    });
  }

  test('linreg walks to the least-squares line: slope 0.7, intercept 0.2', () => {
    const net = M.PRESETS.linreg.build(), d = data(M.PRESETS.linreg);
    for (let s = 0; s < 500; s++) M.trainStep(net, d, { lr: 0.3 });
    near(net.edges[0].w, 0.7, 1e-3);
    near(M.nodesIn(net, 1)[0].bias, 0.2, 1e-3);
  });

  test('logreg: w turns to point from blob 0 to blob 1', () => {
    const net = M.PRESETS.logreg.build(), d = data(M.PRESETS.logreg, 200);
    for (let s = 0; s < 300; s++) M.trainStep(net, d, { lr: 0.3 });
    const [w1, w2] = net.edges.map(e => e.w);
    assert.ok((w1 * 0.45 + w2 * 0.35) / Math.hypot(w1, w2) / Math.hypot(0.45, 0.35) > 0.99);
    assert.ok(accuracy(net, d.X, d.Y) > 0.95);
  });

  test('towers fit Circles although the net computes f(x1) + g(x2)', () => {
    const net = M.PRESETS.towers.build(), d = data(M.PRESETS.towers, 200);
    sgd(net, d, 3000, 0.3, 10);
    assert.ok(accuracy(net, d.X, d.Y) > 0.95);
  });

  test('pca_ae: the columns of W2 come to span the cloud\'s plane (top two principal components)', () => {
    const net = M.PRESETS.pca_ae.build(), d = data(M.PRESETS.pca_ae, 200);
    for (let s = 0; s < 1000; s++) M.trainStep(net, d, { lr: 0.1 });
    const W2 = M.matrices(net)[1].terms[0].W, c = [0, 1].map(j => W2.map(row => row[j]));
    const normal = [0, 1, 2].map(i => c[0][(i + 1) % 3] * c[1][(i + 2) % 3] - c[0][(i + 2) % 3] * c[1][(i + 1) % 3]);
    const thin = [2, -2, -1].map(v => v / 3);
    const cos = Math.abs(normal.reduce((s, v, i) => s + v * thin[i], 0)) / Math.hypot(...normal);
    assert.ok(cos > 0.99, `cos ${cos}`);
  });
});

describe('DATASETS', () => {
  const KEYS = ['xor', 'circles', 'spiral', 'blobs', 'moons', 'three', 'line', 'sine', 'cloud'];
  for (const k of KEYS) {
    test(`${k}: shape, determinism, labels`, () => {
      const d = M.DATASETS[k];
      assert.ok(d, k);
      assert.equal(typeof d.label, 'string');
      assert.ok(d.kind === 'class' || d.kind === 'reg');
      const { X, Y } = d.make(120, 3, 0);
      assert.equal(X.length, 120);
      assert.equal(Y.length, 120);
      assert.ok(X.every(x => x.length === d.inputs && x.every(Number.isFinite)));
      assert.ok(Y.every(y => y.length === d.outputs && y.every(Number.isFinite)));
      assert.ok(X.flat().every(v => Math.abs(v) <= 1.6), 'roughly in [-1, 1]');
      assert.deepEqual(d.make(120, 3, 0), { X, Y });
      assert.notDeepEqual(d.make(120, 4, 0).X, X);
      assert.notDeepEqual(d.make(120, 3, 0.2), { X, Y }, 'noise changes the data');
      if (d.kind === 'class') {
        if (d.outputs === 1) {
          assert.ok(Y.every(y => y[0] === 0 || y[0] === 1));
          const ones = Y.filter(y => y[0] === 1).length;
          assert.ok(ones > 30 && ones < 90, 'roughly balanced');
        } else {
          assert.ok(Y.every(y => y.every(v => v === 0 || v === 1) && y.reduce((s, v) => s + v, 0) === 1), 'one-hot');
        }
      }
    });
  }
  test('kinds and sizes match the contract', () => {
    assert.equal(M.DATASETS.three.outputs, 3);
    assert.equal(M.DATASETS.line.kind, 'reg');
    assert.equal(M.DATASETS.sine.kind, 'reg');
    assert.equal(M.DATASETS.xor.kind, 'class');
    const { X, Y } = M.DATASETS.xor.make(40, 1, 0);
    X.forEach((x, i) => assert.equal(Y[i][0], (x[0] > 0) !== (x[1] > 0) ? 1 : 0));
  });
  test('cloud: 3 -> 3, the target is the point, flat along (2, -2, -1)/3', () => {
    const d = M.DATASETS.cloud;
    assert.deepEqual([d.inputs, d.outputs, d.kind], [3, 3, 'reg']);
    const { X, Y } = d.make(300, 2, 0);
    assert.deepEqual(Y, X);
    assert.notEqual(Y[0], X[0], 'separate arrays, so target noise leaves X alone');
    const thin = [2, -2, -1].map(c => c / 3), wide = [2, 1, 2].map(c => c / 3);
    const spread = (P, u) => Math.sqrt(P.reduce((s, x) => s + (x[0] * u[0] + x[1] * u[1] + x[2] * u[2]) ** 2, 0) / P.length);
    assert.ok(spread(X, thin) < 0.06 && spread(X, wide) > 0.4, `${spread(X, thin)} ${spread(X, wide)}`);
    const noisy = d.make(300, 2, 0.1);
    assert.ok(spread(noisy.X, thin) < 0.06, 'regression noise goes on the targets, so the inputs stay flat');
    assert.ok(spread(noisy.Y, thin) > 0.08);
  });
  test('the new dataset changes no existing preset\'s default (nothing else is 3 -> 3)', () => {
    for (const k of ['perceptron', 'xor', 'mlp', 'deep', 'autoencoder', 'residual', 'linear', 'classifier']) {
      const net = M.PRESETS[k].build(1);
      assert.notDeepEqual([M.nodesIn(net, 0).length, M.nodesIn(net, last(net)).length], [3, 3], k);
    }
  });
});

// ---------------------------------------------------------------- forward / matrices / collapse

describe('forward, matrices, predict, collapse', () => {
  // x (2) -> h (2, relu) -> y (1, identity), plus a skip x1 -> y.
  function tiny() {
    const net = M.emptyNet();
    const [lin, lout] = ids(net.layers);
    const lh = M.addLayer(net, 1, { act: 'relu', size: 2 });
    const [x1, x2] = [1, 2].map(value => M.addNode(net, lin, { value }));
    const y = M.addNode(net, lout, {});
    const [h1, h2] = ids(M.nodesIn(net, lh));
    M.connect(net, x1, h1, 0.5); M.connect(net, x2, h1, -1);
    M.connect(net, x1, h2, 2); M.connect(net, x2, h2, 1);
    M.connect(net, h1, y, 3); M.connect(net, h2, y, -0.5);
    M.connect(net, x1, y, 10);
    M.setNode(net, h1, { bias: 0.25 }); M.setNode(net, h2, { bias: -1 }); M.setNode(net, y, { bias: 0.1 });
    return { net, x1, x2, h1, h2, y };
  }

  test('hand-computed forward pass with a skip edge', () => {
    const { net, h1, h2, y } = tiny();
    const f = M.forward(net);
    // h1: 0.5*1 - 1*2 + 0.25 = -1.25 -> relu 0; h2: 2 + 2 - 1 = 3 -> 3
    assert.deepEqual(f.z[1], [-1.25, 3]);
    assert.deepEqual(f.a[1], [0, 3]);
    // y: 3*0 - 0.5*3 + 10*1 + 0.1 = 8.6
    near(f.a[2][0], 8.6, 1e-12);
    assert.equal(f.z[0], null);
    assert.deepEqual(f.a[0], [1, 2]);
    assert.deepEqual(f.node[h1], { z: -1.25, a: 0 });
    assert.equal(f.node[h2].a, 3);
    near(f.node[y].z, 8.6, 1e-12);
    // x override; missing entries count as 0
    near(M.forward(net, [0, 1]).a[2][0], 3 * 0 - 0.5 * 0 + 0.1, 1e-12);
    near(M.forward(net, [1]).a[2][0], 3 * 0.75 - 0.5 * 1 + 10 + 0.1, 1e-12);
  });

  test('matrices: shapes, skip terms sorted k desc, masked cells', () => {
    const { net, x1, x2, h1, h2, y } = tiny();
    const mats = M.matrices(net);
    assert.equal(mats.length, 2);
    assert.deepEqual(mats.map(m => m.l), [1, 2]);
    assert.deepEqual(mats[0].rows, [h1, h2]);
    assert.deepEqual(mats[0].b, [0.25, -1]);
    assert.equal(mats[0].terms.length, 1);
    assert.deepEqual(mats[0].terms[0], {
      k: 0, cols: [x1, x2], W: [[0.5, -1], [2, 1]],
      edge: [[M.edgeBetween(net, x1, h1).id, M.edgeBetween(net, x2, h1).id], [M.edgeBetween(net, x1, h2).id, M.edgeBetween(net, x2, h2).id]],
    });
    assert.deepEqual(mats[1].terms.map(t => t.k), [1, 0]);
    assert.deepEqual(mats[1].terms[0].W, [[3, -0.5]]);
    assert.deepEqual(mats[1].terms[1].W, [[10, 0]]);
    assert.equal(mats[1].terms[1].edge[0][1], null, 'x2 -> y is masked');
    assert.equal(mats[1].id, net.layers[2].id);
    assert.equal(mats[1].act, 'identity');
    assert.deepEqual(mats[1].rows, [y]);
  });

  test('matrices: the k = l-1 term exists even with only skip edges into a layer', () => {
    const { net, h1, h2, y } = tiny();
    M.disconnect(net, M.edgeBetween(net, h1, y).id);
    M.disconnect(net, M.edgeBetween(net, h2, y).id);
    const m = M.matrices(net)[1];
    assert.deepEqual(m.terms.map(t => t.k), [1, 0]);
    assert.deepEqual(m.terms[0].W, [[0, 0]]);
    assert.deepEqual(m.terms[0].edge, [[null, null]]);
  });

  test('matrix form reproduces forward: z = b + sum_k W a (attention layers: see the attention tests)', () => {
    for (const k of Object.keys(M.PRESETS)) {
      const net = M.PRESETS[k].build(2);
      const f = M.forward(net);
      for (const m of M.matrices(net)) {
        if (m.kind === 'attention') continue;
        m.rows.forEach((_, i) => {
          let z = m.b[i];
          for (const t of m.terms) t.W[i].forEach((w, j) => { z += w * f.a[t.k][j]; });
          near(z, f.z[m.l][i], 1e-12, `${k} z[${m.l}][${i}]`);
        });
      }
    }
  });

  test('softmax layer outputs a distribution', () => {
    const net = M.PRESETS.classifier.build(1);
    M.nodesIn(net, 2).forEach((n, i) => { n.bias = 500 * (i + 1); });
    const a = M.forward(net).a[2];
    assert.ok(allFinite(a));
    near(a.reduce((s, v) => s + v, 0), 1, 1e-12);
  });

  test('predict matches forward, and can return other layers', () => {
    const net = M.PRESETS.mlp.build(1);
    const X = [[0.1, 0.2], [-0.5, 0.9], [1, -1]];
    const P = M.predict(net, X);
    X.forEach((x, s) => nearV(P[s], M.forward(net, x).a.at(-1), 1e-12));
    const H = M.predict(net, X, { layer: 1 });
    X.forEach((x, s) => nearV(H[s], M.forward(net, x).a[1], 1e-12));
    assert.deepEqual(M.predict(net, X, { layer: net.layers[1].id }), H);
    const all = M.predict(net, X, { layer: 'all' });
    assert.equal(all[0].length, net.layers.length);
    assert.deepEqual(all[1][1], H[1]);
    assert.deepEqual(M.predict(net, 'nope'), []);
  });

  test('collapse: linear net = one affine map, including skip edges', () => {
    const net = M.PRESETS.linear.build(4);
    const [x1, x2] = ids(M.nodesIn(net, 0));
    const o = M.nodesIn(net, 2)[1];
    M.connect(net, x2, o.id, 0.75);
    const c = M.collapse(net);
    assert.equal(c.W.length, 2);
    assert.equal(c.W[0].length, 2);
    assert.deepEqual(c.cols, [x1, x2]);
    assert.deepEqual(c.rows, ids(M.nodesIn(net, 2)));
    const r = M.rng(9);
    for (let t = 0; t < 10; t++) {
      const x = [r() * 4 - 2, r() * 4 - 2];
      const want = M.forward(net, x).a[2];
      const got = c.W.map((row, i) => row[0] * x[0] + row[1] * x[1] + c.b[i]);
      nearV(got, want, 1e-12);
    }
  });

  test('collapse: null once any non-input layer is nonlinear; layer-0 act is ignored', () => {
    const net = M.PRESETS.linear.build(1);
    M.setLayer(net, net.layers[0].id, { act: 'tanh' });
    assert.ok(M.collapse(net));
    M.setLayer(net, net.layers[1].id, { act: 'relu' });
    assert.equal(M.collapse(net), null);
    assert.equal(M.collapse(M.PRESETS.xor.build(1)), null);
  });
});

// ---------------------------------------------------------------- backward vs finite differences

describe('backward matches finite differences', () => {
  const ACT_NAMES = ['identity', 'relu', 'leaky', 'sigmoid', 'tanh', 'softmax'];

  for (const act of ACT_NAMES) {
    test(`hidden layers ${act} (mse, skip edges, masked entries)`, () => {
      const net = gradNet({ hidden: act, seed: 11 });
      assertSkipAndMask(net);
      gradCheck(net, targets(net), 'mse');
    });
  }

  for (const act of ACT_NAMES) {
    test(`output layer ${act} with mse`, () => {
      const net = gradNet({ hidden: 'tanh', hidden2: 'leaky', out: act, outSize: 3, seed: 5 });
      const b = gradCheck(net, targets(net), 'mse');
      assert.equal(b.note, null);
    });
  }

  test('xent + softmax: exact gradient, dZ = p - y for a distribution', () => {
    for (const y of [[0, 1, 0], [0.2, 0.5, 0.3]]) {
      const net = gradNet({ hidden: 'relu', hidden2: 'tanh', out: 'softmax', outSize: 3, loss: 'xent', y, seed: 3 });
      const b = gradCheck(net, y, 'xent');
      const p = M.forward(net).a[3];
      nearV(b.dZ[3], p.map((v, i) => v - y[i]), 1e-12);
      near(b.loss, -y.reduce((s, yi, i) => s + yi * Math.log(p[i]), 0), 1e-12, 'xent value');
      assert.equal(b.note, null);
    }
    // targets that don't sum to 1 still get the exact gradient
    const y = [1, 1, 0];
    gradCheck(gradNet({ out: 'softmax', outSize: 3, loss: 'xent', y, seed: 8 }), y, 'xent');
  });

  test('xent + sigmoid: binary cross-entropy, mean over outputs', () => {
    const y = [1, 0.25];
    const net = gradNet({ hidden: 'sigmoid', hidden2: 'relu', out: 'sigmoid', outSize: 2, loss: 'xent', y, seed: 2 });
    const b = gradCheck(net, y, 'xent');
    const a = M.forward(net).a[3];
    const bce = -a.reduce((s, ai, i) => s + y[i] * Math.log(ai) + (1 - y[i]) * Math.log(1 - ai), 0) / a.length;
    near(b.loss, bce, 1e-12);
    nearV(b.dZ[3], a.map((ai, i) => (ai - y[i]) / a.length), 1e-12);
    assert.equal(b.note, null);
  });

  test('xent on another output layer falls back to mse and says so', () => {
    const net = gradNet({ out: 'tanh', outSize: 2, loss: 'xent', seed: 4 });
    const y = targets(net);
    const b = gradCheck(net, y, 'xent');
    assert.equal(typeof b.note, 'string');
    assert.match(b.note, /mse/);
    const m = M.backward(net, M.forward(net), y, 'mse');
    assert.equal(b.loss, m.loss);
    assert.deepEqual(b.edge, m.edge);
  });

  test('mse value is 1/2 mean over outputs of (a - y)^2', () => {
    const net = gradNet({ outSize: 3, seed: 6 });
    const y = targets(net);
    const a = M.forward(net).a[3];
    const want = a.reduce((s, ai, i) => s + (ai - y[i]) ** 2, 0) / (2 * a.length);
    near(M.backward(net, M.forward(net), y, 'mse').loss, want, 1e-12);
  });

  test('loss defaults to meta.loss; stale fwd is recomputed', () => {
    const net = gradNet({ out: 'softmax', outSize: 3, loss: 'xent', y: [0, 0, 1], seed: 9 });
    const y = targets(net);
    const b1 = M.backward(net, M.forward(net), y);
    near(b1.loss, M.backward(net, M.forward(net), y, 'xent').loss, 0);
    const b2 = M.backward(net, { z: [], a: [] }, y, 'xent');
    assert.deepEqual(b2.edge, b1.edge);
  });

  test('result shape: per-layer arrays indexed by l, node and edge maps', () => {
    const net = gradNet({ seed: 1 });
    const b = M.backward(net, M.forward(net), targets(net), 'mse');
    assert.equal(b.dA.length, net.layers.length);
    assert.deepEqual(b.dZ[0], []);
    assert.deepEqual(b.db[0], []);
    assert.deepEqual(b.dW[0], []);
    assert.equal(b.dA[0].length, 3);
    assert.deepEqual(Object.keys(b.edge).sort(), ids(net.edges).sort(), 'only existing edges');
    assert.deepEqual(Object.keys(b.node).sort(), ids(net.nodes).sort());
    const x = M.nodesIn(net, 0)[0];
    assert.equal(b.node[x.id].dz, null);
    assert.equal(b.node[x.id].da, b.dA[0][0]);
  });

  test('a longer mixed chain: relu -> softmax (hidden) -> sigmoid xent', () => {
    const y = [0, 1];
    const net = gradNet({ hidden: 'relu', hidden2: 'softmax', out: 'sigmoid', outSize: 2, loss: 'xent', y, seed: 21 });
    gradCheck(net, y, 'xent');
  });
});

// The inspector and the matrix panel print dL/da next to delta = dL/dz and chain them by hand, so
// the model's dA must satisfy the same identities, including on the fused cross-entropy heads.
describe('chain-rule identities the UI prints', () => {
  const outDelta = (net, loss, y) => {
    const f = M.forward(net), b = M.backward(net, f, y, loss), o = last(net);
    return { f, b, a: f.a[o], dA: b.dA[o], dZ: b.dZ[o] };
  };

  test('sigmoid + BCE: dL/da = (a - y) / (n a (1 - a)) and delta = dL/da * a (1 - a) = (a - y) / n', () => {
    const y = [1, 0.25];
    const net = gradNet({ out: 'sigmoid', outSize: 2, loss: 'xent', y, seed: 2 });
    const { a, dA, dZ } = outDelta(net, 'xent', y);
    a.forEach((ai, i) => {
      near(dA[i], (ai - y[i]) / (a.length * ai * (1 - ai)), 1e-12, `dA[${i}]`);
      near(dZ[i], dA[i] * ai * (1 - ai), 1e-12, `dZ[${i}]`);
      near(dZ[i], (ai - y[i]) / a.length, 1e-12, `dZ[${i}] fused`);
    });
  });

  test('softmax + CE: dL/da = -y / a and delta_i = a_i (dL/da_i - sum_k a_k dL/da_k) = a_i sum(y) - y_i', () => {
    for (const y of [[0, 1, 0], [1, 1, 0]]) {
      const net = gradNet({ out: 'softmax', outSize: 3, loss: 'xent', y, seed: 8 });
      const { a, dA, dZ } = outDelta(net, 'xent', y);
      const S = y.reduce((s, v) => s + v, 0);
      const s = a.reduce((t, ak, k) => t + ak * dA[k], 0);
      a.forEach((ai, i) => {
        near(dA[i], -y[i] / ai, 1e-12, `dA[${i}]`);
        near(dZ[i], ai * (dA[i] - s), 1e-12, `dZ[${i}] via the Jacobian`);
        near(dZ[i], ai * S - y[i], 1e-12, `dZ[${i}] fused`);
      });
    }
  });

  test('mse: dL/da = (a - y) / n, also under a softmax output', () => {
    for (const out of ['identity', 'softmax']) {
      const net = gradNet({ out, outSize: 3, seed: 6 });
      const y = targets(net);
      const { a, dA } = outDelta(net, 'mse', y);
      a.forEach((ai, i) => near(dA[i], (ai - y[i]) / a.length, 1e-12, `${out} dA[${i}]`));
    }
  });

  test('hidden and input dL/da = sum over the outgoing edges (skips included) of w * delta', () => {
    const net = gradNet({ hidden: 'tanh', hidden2: 'relu', seed: 11 });
    const f = M.forward(net), b = M.backward(net, f, targets(net), 'mse');
    for (const n of net.nodes) {
      const l = M.nodeLayerIndex(net, n.id);
      if (l === last(net)) continue;
      const want = net.edges.filter(e => e.from === n.id).reduce((s, e) => s + e.w * b.node[e.to].dz, 0);
      near(b.node[n.id].da, want, 1e-12, n.id);
    }
  });

  test('the kink convention the UI states: slope at z = 0 is the left-hand one', () => {
    assert.equal(M.ACTS.relu.df(0), 0);
    assert.equal(M.ACTS.relu.f(0), 0);
    assert.equal(M.ACTS.leaky.df(0), M.ACTS.leaky.slope);
    assert.equal(M.ACTS.leaky.slope, 0.1);
    assert.equal(M.ACTS.leaky.f(-1), -M.ACTS.leaky.slope);
    assert.match(M.ACTS.leaky.tex, /0\.1z/, 'the formula shows the same slope');
    assert.equal(M.ACTS.relu.df(1e-9), 1);
  });
});

// ---------------------------------------------------------------- training

describe('trainStep', () => {
  test('solves XOR from a fixed seed', () => {
    const net = M.PRESETS.xor.build(1);
    const data = M.DATASETS.xor.make(200, 1, 0);
    const first = M.trainStep(net, data, { lr: 0.5 });
    let L = first;
    for (let i = 0; i < 1500; i++) L = M.trainStep(net, data, { lr: 0.5 });
    assert.ok(L < first / 5, `loss ${first} -> ${L}`);
    assert.ok(accuracy(net, data.X, data.Y) >= 0.97, 'train accuracy');
    const corners = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    assert.deepEqual(M.predict(net, corners).map(p => (p[0] > 0.5 ? 1 : 0)), [0, 1, 1, 0], 'truth table');
    valid(net);
  });

  test('solves circles from a fixed seed (and generalises)', () => {
    const net = M.PRESETS.mlp.build(1);
    const data = M.DATASETS.circles.make(200, 1, 0);
    let L;
    for (let i = 0; i < 800; i++) L = M.trainStep(net, data, { lr: 0.5 });
    assert.ok(L < 0.1, `loss ${L}`);
    assert.ok(accuracy(net, data.X, data.Y) >= 0.97);
    const test2 = M.DATASETS.circles.make(200, 2, 0);
    assert.ok(accuracy(net, test2.X, test2.Y) >= 0.9, 'held-out accuracy');
  });

  test('softmax classifier learns three classes; regression fits a sine', () => {
    const cls = M.PRESETS.classifier.build(1);
    const three = M.DATASETS.three.make(150, 1, 0);
    for (let i = 0; i < 500; i++) M.trainStep(cls, three, { lr: 0.5 });
    assert.ok(accuracy(cls, three.X, three.Y) >= 0.95);

    const reg = M.emptyNet();
    const [lin, lout] = ids(reg.layers);
    M.addNode(reg, lin, { value: 0.5 });
    M.addNode(reg, lout, {});
    M.addLayer(reg, 1, { act: 'tanh', size: 8, dense: true, seed: 3 });
    const sine = M.DATASETS.sine.make(100, 1, 0);
    let L;
    for (let i = 0; i < 3000; i++) L = M.trainStep(reg, sine, { lr: 0.3, loss: 'mse' });
    assert.ok(L < 0.005, `sine mse ${L}`);
  });

  test('one step = mean of per-sample backward gradients; returns the mean loss', () => {
    const net = gradNet({ hidden: 'tanh', hidden2: 'leaky', out: 'sigmoid', outSize: 2, seed: 13 });
    const X = [[0.1, 0.5, -0.3], [1, -1, 0.2], [-0.4, 0.3, 0.9]];
    const Y = [[1, 0], [0, 1], [0.5, 0.5]];
    const before = M.clone(net);
    const per = X.map((x, s) => M.backward(net, M.forward(net, x), Y[s], 'xent'));
    const lr = 0.3;
    const mean = M.trainStep(net, { X, Y }, { lr, loss: 'xent' });
    near(mean, per.reduce((s, b) => s + b.loss, 0) / X.length, 1e-12);
    for (const e of before.edges) {
      const g = per.reduce((s, b) => s + b.edge[e.id], 0) / X.length;
      near(M.edge(net, e.id).w, e.w - lr * g, 1e-12, e.id);
    }
    for (const n of before.nodes) {
      const l = M.nodeLayerIndex(before, n.id);
      const g = l ? per.reduce((s, b) => s + b.node[n.id].dz, 0) / X.length : 0;
      near(M.node(net, n.id).bias, n.bias - lr * g, 1e-12, n.id);
    }
    assert.deepEqual(ids(net.edges), ids(before.edges), 'masked entries stay masked');
    assert.deepEqual(M.nodesIn(net, 0).map(n => n.value), M.nodesIn(before, 0).map(n => n.value), 'inputs untouched');
  });

  test('defaults: loss from meta, scalar rows, empty batch', () => {
    const net = M.PRESETS.perceptron.build(1);
    M.removeNode(net, M.nodesIn(net, 0)[1].id);
    const copy = M.clone(net);
    const a = M.trainStep(net, { X: [0.5, -0.5], Y: [1, 0] }, { lr: 0.1 });
    const b = M.trainStep(copy, { X: [[0.5], [-0.5]], Y: [[1], [0]] }, { lr: 0.1, loss: 'xent' });
    assert.equal(a, b);
    assert.deepEqual(net, copy);
    const snap = M.clone(net);
    assert.equal(M.trainStep(net, { X: [], Y: [] }, { lr: 1 }), 0);
    assert.deepEqual(net, snap);
  });

  test('a diverged step changes nothing and reports the non-finite loss', () => {
    const net = M.PRESETS.linear.build(1);
    net.edges.forEach(e => { e.w = 1e200; });
    const snap = M.clone(net);
    const L = M.trainStep(net, { X: [[1, 1]], Y: [[0, 0]] }, { lr: 0.1, loss: 'mse' });
    assert.ok(!Number.isFinite(L));
    assert.deepEqual(net, snap);
    valid(net);
  });
});

// ---------------------------------------------------------------- edits

describe('edit operations keep the net valid', () => {
  test('addLayer: position, defaults, relabel, dense wiring', () => {
    const net = M.PRESETS.xor.build(1);
    const edges0 = net.edges.length;
    const id = M.addLayer(net, 2, { size: 3 });
    valid(net);
    assert.equal(net.layers[2].id, id);
    assert.equal(net.layers[2].act, 'tanh');
    assert.equal(net.layers[2].name, 'Hidden');
    assert.deepEqual(M.nodesIn(net, id).map(n => n.label), ['h^{(2)}_{1}', 'h^{(2)}_{2}', 'h^{(2)}_{3}']);
    assert.equal(net.edges.length, edges0, 'no wiring by default');
    const xs = [0, 1, 2, 3].map(l => M.nodesIn(net, l)[0].x);
    assert.ok(xs[1] < xs[2] && xs[2] < xs[3], 'new column between its neighbours');
    // the old hidden -> output edges are now skip edges
    assert.equal(M.matrices(net)[2].terms.length, 2);

    const out = M.addLayer(net, net.layers.length, { size: 1, act: 'sigmoid', dense: true, seed: 1 });
    valid(net);
    assert.equal(net.layers.at(-1).id, out);
    assert.equal(M.nodesIn(net, 3)[0].label, 'h^{(3)}_{1}', 'old output relabelled as hidden');
    assert.equal(M.nodesIn(net, out)[0].label, '\\hat y_{1}');
    assert.equal(M.nodesIn(net, out)[0].target, 0, 'backward pass stays on');
    assert.ok(createStore(net).state.bwd);
    assert.equal(net.edges.length, edges0 + 1);

    const inp = M.addLayer(net, 0, { size: 2, act: 'bogus' });
    valid(net);
    assert.equal(net.layers[0].id, inp);
    assert.equal(net.layers[0].act, 'identity');
    assert.deepEqual(M.nodesIn(net, 1).map(n => n.label), ['h^{(1)}_{1}', 'h^{(1)}_{2}']);
    assert.deepEqual(M.nodesIn(net, 0).map(n => n.label), ['x_{1}', 'x_{2}']);

    const empty = M.addLayer(net, 99, { size: 0, name: 'Z' });
    assert.equal(net.layers.at(-1).id, empty);
    assert.equal(net.layers.at(-1).name, 'Z');
    assert.equal(M.nodesIn(net, empty).length, 0);
    valid(net);
    assert.ok(allFinite(M.forward(net).a.flat()));
  });

  test('removeLayer drops its nodes and their edges; keeps at least 2 layers', () => {
    const net = M.PRESETS.deep.build(1);
    const l2 = net.layers[2].id;
    const gone = new Set(ids(M.nodesIn(net, l2)));
    assert.equal(M.removeLayer(net, l2), true);
    valid(net);
    assert.ok(!net.nodes.some(n => gone.has(n.id)));
    assert.ok(!net.edges.some(e => gone.has(e.from) || e.to === l2 || gone.has(e.to)));
    assert.equal(M.removeLayer(net, 'ghost'), false);
    while (net.layers.length > 2) assert.equal(M.removeLayer(net, net.layers[1].id), true);
    assert.equal(M.removeLayer(net, net.layers[0].id), false);
    valid(net);
  });

  test('removeLayer { bridge }: wires the neighbours of a removed hidden layer, only when unjoined', () => {
    const between = (net, a, b) => {
      const A = new Set(ids(M.nodesIn(net, a))), B = new Set(ids(M.nodesIn(net, b)));
      return net.edges.filter(e => A.has(e.from) && B.has(e.to));
    };
    // default: the neighbours are left unconnected
    const plain = M.PRESETS.mlp.build(1);
    M.removeLayer(plain, plain.layers[2].id);
    valid(plain);
    assert.equal(between(plain, 1, 2).length, 0);

    // bridge: dense and seeded
    const mk = () => { const n = M.PRESETS.mlp.build(1); M.removeLayer(n, n.layers[2].id, { bridge: true, seed: 4 }); return n; };
    const net = mk();
    valid(net);
    const e = between(net, 1, 2);
    assert.equal(e.length, M.nodesIn(net, 1).length * M.nodesIn(net, 2).length, 'dense');
    assert.ok(e.every(x => x.w !== 0 && Math.abs(x.w) < 2), 'initialised weights');
    assert.deepEqual(mk().edges, net.edges, 'seeded');
    const b = M.backward(net, M.forward(net), targets(net), 'xent');
    assert.ok(e.every(x => Number.isFinite(b.edge[x.id])), 'the new edges carry gradients');

    // a skip edge already joins the neighbours: nothing is added
    const skip = M.PRESETS.mlp.build(1);
    M.connect(skip, M.nodesIn(skip, 1)[0].id, M.nodesIn(skip, 3)[0].id, 0.5);
    const kept = skip.edges.length - between(skip, 1, 2).length - between(skip, 2, 3).length;
    M.removeLayer(skip, skip.layers[2].id, { bridge: true, seed: 1 });
    valid(skip);
    assert.equal(skip.edges.length, kept, 'kept the skip edge, added none');

    // removing the input or the output layer never bridges
    for (const at of ['first', 'last']) {
      const n = M.PRESETS.deep.build(1);
      const lid = at === 'first' ? n.layers[0].id : n.layers.at(-1).id;
      const ns = new Set(ids(M.nodesIn(n, lid)));
      const left = n.edges.filter(x => !ns.has(x.from) && !ns.has(x.to)).length;
      M.removeLayer(n, lid, { bridge: true, seed: 1 });
      valid(n);
      assert.equal(n.edges.length, left, at);
    }
  });

  test('setLayer validates act', () => {
    const net = M.PRESETS.xor.build(1);
    const id = net.layers[1].id;
    assert.equal(M.setLayer(net, id, { act: 'relu', name: 'H' }), true);
    assert.equal(net.layers[1].act, 'relu');
    assert.equal(net.layers[1].name, 'H');
    M.setLayer(net, id, { act: 'nope', name: 5 });
    assert.equal(net.layers[1].act, 'relu');
    assert.equal(net.layers[1].name, 'H');
    assert.equal(M.setLayer(net, 'ghost', { act: 'tanh' }), false);
    valid(net);
  });

  test('addNode: order, labels, placement, targets, wiring', () => {
    const net = M.PRESETS.xor.build(1);
    const hid = net.layers[1].id;
    const col = M.nodesIn(net, hid)[0].x;
    const a = M.addNode(net, hid, {});
    valid(net);
    assert.equal(M.nodesIn(net, hid).at(-1).id, a);
    assert.equal(M.node(net, a).label, 'h^{(1)}_{5}');
    assert.equal(M.node(net, a).x, col);
    assert.ok(M.node(net, a).y > Math.max(...M.nodesIn(net, hid).slice(0, -1).map(n => n.y)));
    const b = M.addNode(net, 1, { index: 0, label: 'custom', x: 5, y: 6, bias: '0.5' });
    assert.equal(M.nodesIn(net, hid)[0].id, b);
    assert.equal(M.node(net, b).label, 'custom');
    assert.equal(M.node(net, b).bias, 0.5);
    assert.deepEqual([M.node(net, b).x, M.node(net, b).y], [5, 6]);
    assert.equal(M.nodesIn(net, hid)[1].label, 'h^{(1)}_{2}', 'defaults renumbered around the new node');
    const c = M.addNode(net, hid, { connect: true, seed: 2 });
    assert.equal(net.edges.filter(e => e.to === c).length, 2);
    assert.equal(net.edges.filter(e => e.from === c).length, 1);
    const o = M.addNode(net, net.layers[2].id, {});
    assert.equal(M.node(net, o).target, 0, 'outputs keep full targets');
    assert.equal(M.node(net, o).label, '\\hat y_{2}');
    assert.equal(M.addNode(net, 'ghost', {}), null);
    valid(net);
    assert.ok(createStore(net).state.bwd);
  });

  test('removeNode removes its edges and renumbers default labels', () => {
    const net = M.PRESETS.mlp.build(1);
    const x1 = M.nodesIn(net, 0)[0];
    M.setNode(net, M.nodesIn(net, 1)[5].id, { label: 'mine' });
    assert.equal(M.removeNode(net, x1.id), true);
    valid(net);
    assert.ok(!net.edges.some(e => e.from === x1.id || e.to === x1.id));
    assert.equal(M.nodesIn(net, 0)[0].label, 'x_{1}');
    const h = M.nodesIn(net, 1);
    M.removeNode(net, h[0].id);
    assert.equal(M.nodesIn(net, 1)[0].label, 'h^{(1)}_{1}');
    assert.equal(M.nodesIn(net, 1).at(-1).label, 'mine', 'custom label kept');
    assert.equal(M.removeNode(net, 'ghost'), false);
    valid(net);
  });

  test('setNode: lenient numbers, invalid ignored, targets, params, layer moves', () => {
    const net = M.PRESETS.xor.build(1);
    const x = M.nodesIn(net, 0)[0], h = M.nodesIn(net, 1)[0], o = M.nodesIn(net, 2)[0];
    assert.equal(M.setNode(net, x.id, { value: '1.5', x: 12, y: 'bad', label: 'a', params: { u: 'm', n: 3, bad: {}, nan: NaN } }), true);
    assert.equal(x.value, 1.5);
    assert.equal(x.x, 12);
    assert.notEqual(x.y, 'bad');
    assert.equal(x.label, 'a');
    assert.deepEqual(x.params, { u: 'm', n: 3 });
    M.setNode(net, h.id, { bias: NaN, id: 'hacked' });
    assert.ok(Number.isFinite(h.bias));
    assert.notEqual(h.id, 'hacked');
    M.setNode(net, o.id, { target: '' });
    assert.equal(o.target, null);
    M.setNode(net, o.id, { target: 0.25 });
    assert.equal(o.target, 0.25);
    M.setNode(net, o.id, { target: 'abc' });
    assert.equal(o.target, 0.25);
    valid(net);
    // move a hidden node to the output layer: its edge to the output would go sideways -> dropped
    M.setNode(net, h.id, { layer: 2 });
    valid(net);
    assert.equal(h.layer, net.layers[2].id);
    assert.ok(!net.edges.some(e => e.from === h.id));
    assert.equal(net.edges.filter(e => e.to === h.id).length, 2, 'input edges still go forward');
    assert.equal(h.label, '\\hat y_{1}', 'relabelled for its new layer');
    assert.equal(M.setNode(net, 'ghost', { bias: 1 }), false);
  });

  test('moveNode reorders within the layer (matrix rows follow)', () => {
    const net = M.PRESETS.xor.build(1);
    const hid = net.layers[1].id;
    const before = ids(M.nodesIn(net, hid));
    const W0 = M.matrices(net)[0].terms[0].W;
    assert.equal(M.moveNode(net, before[0], 2), true);
    valid(net);
    const after = ids(M.nodesIn(net, hid));
    assert.deepEqual(after, [before[1], before[2], before[0], before[3]]);
    assert.deepEqual(M.matrices(net)[0].terms[0].W, [W0[1], W0[2], W0[0], W0[3]]);
    assert.deepEqual(M.matrices(net)[1].terms[0].cols, after);
    assert.deepEqual(M.nodesIn(net, hid).map(n => n.label), [1, 2, 3, 4].map(j => `h^{(1)}_{${j}}`));
    M.moveNode(net, before[3], -5);
    assert.equal(M.nodesIn(net, hid)[0].id, before[3]);
    M.moveNode(net, before[3], 99);
    assert.equal(M.nodesIn(net, hid).at(-1).id, before[3]);
    assert.equal(M.moveNode(net, 'ghost', 0), false);
    const x = net.nodes.find(n => n.layer === net.layers[0].id);
    const pos = [x.x, x.y];
    M.moveNode(net, x.id, 1);
    assert.deepEqual([x.x, x.y], pos, 'moving order never moves the node');
    valid(net);
  });

  test('connect: create, reuse, swap backward, refuse sideways', () => {
    const net = M.PRESETS.xor.build(1);
    const x = M.nodesIn(net, 0), o = M.nodesIn(net, 2)[0], h = M.nodesIn(net, 1);
    const id = M.connect(net, x[0].id, o.id);
    valid(net);
    const e = M.edge(net, id);
    assert.ok(e.w > -1 && e.w < 1);
    assert.equal(M.connect(net, x[0].id, o.id, 0.3), id, 'reuse');
    assert.equal(e.w, 0.3);
    assert.equal(M.connect(net, o.id, x[0].id), id, 'backward pair is swapped onto the same edge');
    const back = M.connect(net, o.id, x[1].id, -2);
    assert.equal(M.edge(net, back).from, x[1].id);
    assert.equal(M.edge(net, back).to, o.id);
    assert.equal(M.connect(net, h[0].id, h[1].id), null, 'within a layer');
    assert.equal(M.connect(net, h[0].id, h[0].id), null);
    assert.equal(M.connect(net, 'ghost', o.id), null);
    valid(net);
  });

  test('disconnect / setWeight', () => {
    const net = M.PRESETS.xor.build(1);
    const e = net.edges[0];
    assert.equal(M.setWeight(net, e.id, -0.5), true);
    assert.equal(e.w, -0.5);
    assert.equal(M.setWeight(net, e.id, NaN), false);
    assert.equal(M.setWeight(net, e.id, '0.25'), true);
    assert.equal(e.w, 0.25);
    assert.equal(M.setWeight(net, 'ghost', 1), false);
    assert.equal(M.disconnect(net, e.id), true);
    assert.equal(M.edge(net, e.id), null);
    assert.equal(M.disconnect(net, e.id), false);
    valid(net);
  });

  test('connectDense: all pairs, keeps existing weights, returns ids', () => {
    const net = M.PRESETS.xor.build(1);
    const w0 = net.edges.map(e => e.w);
    const again = M.connectDense(net, 1, 2);
    assert.equal(again.length, 4);
    assert.deepEqual(net.edges.map(e => e.w), w0);
    const skip = M.connectDense(net, net.layers[2].id, net.layers[0].id, { seed: 1 });
    valid(net);
    assert.equal(skip.length, 2);
    const bound = Math.sqrt(6 / 3);
    skip.forEach(id => {
      assert.equal(M.nodeLayerIndex(net, M.edge(net, id).from), 0);
      assert.ok(Math.abs(M.edge(net, id).w) <= bound);
    });
    assert.deepEqual(M.connectDense(net, 1, 1), []);
    assert.deepEqual(M.connectDense(net, 0, 2, { w: 0.5 }), skip, 'existing edges reused');
  });

  test('randomize: seeded, scheme scales, bias modes', () => {
    const a = M.PRESETS.deep.build(1), b = M.clone(a), c = M.clone(a);
    M.randomize(a, { seed: 7 }); M.randomize(b, { seed: 7 }); M.randomize(c, { seed: 8 });
    assert.deepEqual(a, b);
    assert.notDeepEqual(a.edges.map(e => e.w), c.edges.map(e => e.w));
    const fanIn = id => a.edges.filter(e => e.to === id).length;
    const fanOut = id => a.edges.filter(e => e.from === id).length;
    a.edges.forEach(e => assert.ok(Math.abs(e.w) <= Math.sqrt(6 / (fanIn(e.to) + fanOut(e.from))) + 1e-12));
    assert.ok(a.nodes.every(n => n.bias === 0), 'biases zeroed by default');
    const inputs = M.nodesIn(a, 0);
    inputs[0].bias = 0.123;
    M.randomize(a, { seed: 1, scheme: 'small', biases: 'small' });
    assert.equal(inputs[0].bias, 0.123, 'input biases untouched');
    const sd = Math.sqrt(a.edges.reduce((s, e) => s + e.w * e.w, 0) / a.edges.length);
    assert.ok(sd > 0.07 && sd < 0.13, `small sd ${sd}`);
    assert.ok(a.nodes.every(n => n.layer === a.layers[0].id || Math.abs(n.bias) <= 0.1));
    const kept = a.nodes.map(n => n.bias);
    M.randomize(a, { seed: 2, scheme: 'he', biases: 'keep' });
    assert.deepEqual(a.nodes.map(n => n.bias), kept);
    valid(a);
  });

  test('randomize { init }: a scheme per shared matrix name; identity and zero; fixed edges kept', () => {
    const net = M.PRESETS.transformer.build(1), fixed = net.edges.filter(e => e.fixed).map(e => e.w);
    const init = { W_V: 'identity', W_O: 'zero', W_1: 'small', W_Q: 'nonsense' };
    M.randomize(net, { seed: 4, init });
    const T = name => M.tiedMatrices(net, net.layers.findIndex((_, l) => M.tiedMatrices(net, l).some(t => t.name === name)))
      .find(t => t.name === name).W;
    assert.deepEqual(T('W_V'), [[1, 0], [0, 1]]);
    assert.deepEqual(T('W_O'), [[0, 0], [0, 0]]);
    assert.ok(T('W_1').flat().every(v => Math.abs(v) < 0.5), 'small');
    assert.notDeepEqual(T('W_Q'), [[0, 0], [0, 0]], 'an unknown scheme falls back to the default');
    assert.deepEqual(net.edges.filter(e => e.fixed).map(e => e.w), fixed);
    valid(net);
    // without init nothing changes
    const a = M.PRESETS.attention.build(1), b = M.clone(a);
    M.randomize(a, { seed: 3 }); M.randomize(b, { seed: 3, init: null });
    assert.deepEqual(a, b);
  });

  test('autoLayout: evenly spaced columns, centred, inside the box', () => {
    const net = M.PRESETS.deep.build(1);
    net.nodes.forEach(n => { n.x = 5000; n.y = -40; });
    M.autoLayout(net, { width: 900, height: 560 });
    const cols = net.layers.map((_, l) => M.nodesIn(net, l).map(n => n.x));
    cols.forEach(c => assert.ok(c.every(x => x === c[0]), 'one x per column'));
    const xs = cols.map(c => c[0]);
    const gaps = xs.slice(1).map((x, i) => x - xs[i]);
    gaps.forEach(g => near(g, gaps[0], 1e-2));
    net.layers.forEach((_, l) => {
      const ys = M.nodesIn(net, l).map(n => n.y);
      near((Math.min(...ys) + Math.max(...ys)) / 2, 280, 1e-2);
      for (let i = 1; i < ys.length; i++) assert.ok(ys[i] > ys[i - 1], 'order top to bottom');
    });
    assert.ok(net.nodes.every(n => n.x > 0 && n.x < 900 && n.y > 0 && n.y < 560));
  });

  test('fuzz: 600 random edits never produce an invalid net', () => {
    const r = M.rng(2024);
    const pick = a => a[Math.floor(r() * a.length)];
    const net = M.PRESETS.residual.build(1);
    const acts = Object.keys(M.ACTS);
    const ops = [
      () => M.addLayer(net, Math.floor(r() * (net.layers.length + 1)), { size: Math.floor(r() * 4), act: pick(acts), dense: r() < 0.5, seed: 1 }),
      () => M.removeLayer(net, pick(net.layers).id),
      () => M.setLayer(net, pick(net.layers).id, { act: pick(acts) }),
      () => M.addNode(net, Math.floor(r() * net.layers.length), { connect: r() < 0.5, index: Math.floor(r() * 4) }),
      () => net.nodes.length && M.removeNode(net, pick(net.nodes).id),
      () => net.nodes.length && M.setNode(net, pick(net.nodes).id, { bias: r() - 0.5, layer: r() < 0.3 ? Math.floor(r() * net.layers.length) : undefined, target: r() < 0.2 ? null : r() }),
      () => net.nodes.length && M.moveNode(net, pick(net.nodes).id, Math.floor(r() * 5) - 1),
      () => net.nodes.length > 1 && M.connect(net, pick(net.nodes).id, pick(net.nodes).id, r() * 2 - 1),
      () => net.edges.length && M.disconnect(net, pick(net.edges).id),
      () => net.edges.length && M.setWeight(net, pick(net.edges).id, r() * 4 - 2),
      () => M.connectDense(net, Math.floor(r() * net.layers.length), Math.floor(r() * net.layers.length), { seed: 3 }),
      () => M.randomize(net, { seed: Math.floor(r() * 100), scheme: pick(['xavier', 'he', 'small']) }),
      () => M.autoLayout(net),
    ];
    for (let step = 0; step < 600; step++) {
      pick(ops)();
      assert.deepEqual(M.validate(net), [], `after step ${step}`);
      const f = M.forward(net);
      assert.equal(f.a.length, net.layers.length);
      const y = targets(net);
      if (y.length && y.every(Number.isFinite)) {
        const b = M.backward(net, f, y, pick(['mse', 'xent']));
        assert.equal(b.dW.length, net.layers.length);
      }
      const mats = M.matrices(net);
      mats.forEach(m => m.terms.forEach(t => assert.ok(t.W.length === m.rows.length && t.W.every(row => row.length === t.cols.length))));
    }
    assert.deepEqual(M.normalize(M.clone(net)), net);
  });
});

// ---------------------------------------------------------------- validate / normalize

describe('validate and normalize', () => {
  test('validate reports each kind of damage', () => {
    const cases = [
      [n => { n.v = 2; }, /v must be 1/],
      [n => { n.layers.splice(1); }, /at least 2 layers/],
      [n => { n.layers[1].act = 'swish'; }, /unknown act/],
      [n => { n.layers[1].id = n.layers[0].id; }, /duplicate id/],
      [n => { n.nodes[0].layer = 'ghost'; }, /not found/],
      [n => { n.nodes[0].bias = NaN; }, /bias must be a finite number/],
      [n => { n.nodes[0].target = 'x'; }, /target/],
      [n => { n.nodes[0].params = { a: {} }; }, /param a/],
      [n => { n.nodes[1].id = n.nodes[0].id; }, /duplicate id/],
      [n => { n.edges[0].to = 'ghost'; }, /to "ghost" not found/],
      [n => { const e = n.edges[0]; [e.from, e.to] = [e.to, e.from]; }, /must go forward/],
      [n => { n.edges.push({ ...n.edges[0], id: 'e999' }); }, /duplicate edge/],
      [n => { n.edges[0].w = Infinity; }, /w must be a finite number/],
      [n => { n.meta.loss = 'hinge'; }, /meta.loss/],
      [n => { n.meta.nextId = 1; }, /nextId/],
      [n => { delete n.meta; }, /meta must be an object/],
    ];
    for (const [breakIt, re] of cases) {
      const net = M.PRESETS.xor.build(1);
      breakIt(net);
      const errs = M.validate(net);
      assert.ok(errs.some(e => re.test(e)), `${re}: ${errs.join(' | ')}`);
      valid(M.normalize(net), `normalize fixes ${re}`);
    }
    assert.deepEqual(M.validate(null), ['net is not an object']);
    assert.ok(M.validate({ v: 1 }).length > 0);
  });

  test('normalize repairs damaged JSON', () => {
    const bad = {
      v: 7,
      extra: { keep: true },
      layers: [
        { id: 'in', name: 'Inputs', act: 'LINEAR' },
        { id: 'in', name: 'dup', act: 'relu' },
        null,
        { name: 'hid', act: 'Leaky-ReLU', custom: 1 },
        { id: 'out', act: 'banana' },
      ],
      nodes: [
        { id: 'a', layer: 'in', x: 10, y: 20, value: '0.5', bias: 'x' },
        { id: 'a', layer: 'in', x: 0, y: 0 },
        { id: 'b', layer: 0, label: 7, target: 'abc', params: { k: 'v', n: 2, o: { deep: 1 }, f: NaN } },
        { id: 'h', layer: 1, x: 100, y: 50, bias: NaN },
        { id: 'o', layer: 'out', target: '1', x: 200, y: 50 },
        { id: 'z', layer: 'nope', x: 1, y: 1 },
        { layer: 'out', x: 200, y: 130 },
        'garbage',
      ],
      edges: [
        { id: 'e1', from: 'a', to: 'h', w: '0.25' },
        { id: 'e1', from: 'b', to: 'h', w: 1 },
        { from: 'o', to: 'h', w: 2 },
        { id: 'e3', from: 'a', to: 'b', w: 1 },
        { id: 'e4', from: 'a', to: 'h', w: 3 },
        { id: 'e5', from: 'a', to: 'ghost', w: 1 },
        { id: 'e6', from: 'b', to: 'o', w: Infinity },
        7,
      ],
      meta: { title: 5, loss: 'CrossEntropy', nextId: 2, train: { epoch: 3 } },
    };
    const net = M.normalize(bad);
    valid(net);
    assert.equal(net.v, 1);
    assert.deepEqual(net.extra, { keep: true });
    assert.deepEqual(net.layers.map(l => l.act), ['identity', 'leaky', 'identity']);
    assert.deepEqual(net.layers.map(l => l.name), ['Inputs', 'hid', 'Output']);
    assert.equal(net.layers[1].custom, 1);
    assert.equal(net.nodes.length, 5);
    const [a, b, h, o, fresh] = net.nodes;
    assert.deepEqual([a.id, b.id, h.id, o.id], ['a', 'b', 'h', 'o']);
    assert.deepEqual([a.value, a.bias, a.x], [0.5, 0, 10]);
    assert.equal(b.layer, 'in');
    assert.equal(b.label, '7');
    assert.equal(b.target, null);
    assert.deepEqual(b.params, { k: 'v', n: 2 });
    assert.ok(Number.isFinite(b.x) && Number.isFinite(b.y), 'unplaced node gets a position');
    assert.equal(b.x, 10, 'placed in its layer column');
    assert.equal(h.layer, net.layers[1].id);
    assert.equal(h.bias, 0);
    assert.equal(o.target, 1);
    assert.equal(fresh.layer, 'out');
    assert.equal(fresh.label, '\\hat y_{2}', 'missing label filled with the default');
    assert.equal(a.label, 'x_{1}');
    const pairs = net.edges.map(e => `${e.from}>${e.to}:${e.w}`);
    assert.deepEqual(pairs, ['a>h:0.25', 'b>h:1', 'h>o:2', 'b>o:0']);
    assert.equal(net.edges[0].id, 'e1');
    assert.equal(new Set(ids(net.edges)).size, 4);
    assert.equal(net.meta.title, 'Untitled');
    assert.equal(net.meta.loss, 'xent');
    assert.deepEqual(net.meta.train, { epoch: 3 });
    const everything = new Set([...ids(net.layers), ...ids(net.nodes), ...ids(net.edges)]);
    for (let i = 0; i < 20; i++) assert.ok(!everything.has(M.uid(net, 'e')), 'fresh ids never collide');
    assert.deepEqual(M.normalize(M.clone(net)), net, 'idempotent');
    assert.ok(allFinite(M.forward(net).a.flat()));
  });

  test('normalize: garbage in, empty net out; strings are parsed', () => {
    for (const junk of [null, undefined, 42, 'not json', [], [1, 2], true]) {
      const net = M.normalize(junk);
      valid(net, String(junk));
      assert.equal(net.layers.length, 2);
    }
    const p = M.PRESETS.classifier.build(2);
    assert.deepEqual(M.normalize(JSON.stringify(p)), p);
  });

  test('normalize: pads to 2 layers, lays out unplaced nets, fixes id collisions', () => {
    const one = M.normalize({ layers: [{ id: 'only' }], nodes: [{ id: 'x', layer: 'only' }, { id: 'y', layer: 'only' }] });
    valid(one);
    assert.equal(one.layers.length, 2);
    assert.equal(one.layers[0].name, 'Input');
    assert.equal(one.layers[1].name, 'Output');
    assert.notEqual(one.nodes[0].y, one.nodes[1].y, 'auto-laid out');

    const clash = M.normalize({
      layers: [{ id: 'A' }, { id: 'B' }],
      nodes: [{ id: 'A', layer: 'A', x: 0, y: 0 }, { id: 'n', layer: 'B', x: 100, y: 0 }],
      edges: [{ id: 'n', from: 'A', to: 'n', w: 1 }],
    });
    valid(clash);
    assert.equal(clash.edges.length, 1);
    assert.notEqual(clash.nodes[0].id, 'A');
    assert.equal(clash.edges[0].from, clash.nodes[0].id, 'edge follows the renamed node');
    assert.notEqual(clash.edges[0].id, 'n');
  });

  test('store round trip: load, commit, undo keep a valid net', () => {
    const store = createStore(M.PRESETS.xor.build(1));
    store.load(JSON.stringify({ layers: [{ id: 'a' }, { id: 'b' }], nodes: [{ id: 'p', layer: 'a' }, { id: 'q', layer: 'b', target: 1 }], edges: [{ from: 'p', to: 'q', w: 2 }] }));
    valid(store.net);
    assert.ok(store.state.bwd);
    M.addNode(store.net, 0, { value: 1 });
    store.commit('add');
    assert.equal(M.nodesIn(store.net, 0).length, 2);
    store.undo();
    valid(store.net);
    assert.equal(M.nodesIn(store.net, 0).length, 1);
  });
});

// ---------------------------------------------------------------- tokens, ties, attention

const ATT_KEYS = ['words', 'pronouns', 'agreement', 'attention', 'causal', 'causal_rot', 'multihead', 'transformer'];
const matmul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
const nearM = (A, B, tol, msg) => { assert.equal(A.length, B.length, `${msg} rows`); A.forEach((r, i) => nearV(r, B[i], tol, `${msg}[${i}]`)); };
const attnLayer = net => net.layers.findIndex(l => l.kind === 'attention');

// Attention from the formulas: per head S = Q Kᵀ · scale (masked -Infinity), A = row softmax, Z = A V.
function refAttention(Q, K, V, { heads = 1, causal = false, scale } = {}) {
  const n = Q.length, d = Q[0].length, dh = d / heads, sc = scale ?? 1 / Math.sqrt(dh);
  const Z = Q.map(() => new Array(d).fill(0)), out = [];
  for (let h = 0; h < heads; h++) {
    const c = f => h * dh + f;
    const S = Q.map((q, i) => K.map((k, j) => {
      if (causal && j > i) return -Infinity;
      let s = 0;
      for (let f = 0; f < dh; f++) s += q[c(f)] * k[c(f)];
      return sc * s;
    }));
    const A = S.map(row => {
      const m = Math.max(...row), e = row.map(s => Math.exp(s - m)), t = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / t);
    });
    A.forEach((row, i) => { for (let f = 0; f < dh; f++) Z[i][c(f)] = row.reduce((s, a, j) => s + a * V[j][c(f)], 0); });
    out.push({ S, A });
  }
  return { Z, heads: out };
}

// Finite differences on a net with ties, fixed edges and attention layers: every edge on its
// own (= bwd.edge), every tie group moved together (= bwd.tie, the sum of its members), every
// bias (and bias group), every input; an attention layer's biases have no effect (db = 0).
function gradCheckShared(net, y, loss, tol = 1e-6) {
  const fwd = M.forward(net), bwd = M.backward(net, fwd, y, loss);
  assert.ok(Number.isFinite(bwd.loss), 'loss finite');
  const groups = new Map(), bgroups = new Map();
  for (const e of net.edges) {
    near(bwd.edge[e.id], fd(net, y, loss, () => e.w, v => { e.w = v; }), tol, `dL/dw ${e.id}${e.fixed ? ' (fixed)' : ''}`);
    if (e.tie) { if (!groups.has(e.tie)) groups.set(e.tie, []); groups.get(e.tie).push(e); }
  }
  for (const [t, es] of groups) {
    near(bwd.tie[t], es.reduce((s, e) => s + bwd.edge[e.id], 0), 1e-12, `tie ${t} is the sum of its edges`);
    near(bwd.tie[t], fd(net, y, loss, () => es[0].w, v => es.forEach(e => { e.w = v; })), tol, `dL/d(${t})`);
  }
  for (const n of net.nodes) {
    const l = M.nodeLayerIndex(net, n.id);
    const fdB = () => fd(net, y, loss, () => n.bias, v => { n.bias = v; });
    if (l === 0) near(bwd.node[n.id].da, fd(net, y, loss, () => n.value, v => { n.value = v; }), tol, `dL/dx ${n.id}`);
    else if (net.layers[l].kind === 'attention') near(fdB(), 0, 1e-12, `attention bias ${n.id} is unused`);
    else {
      near(bwd.node[n.id].dz, fdB(), tol, `dL/db ${n.id}`);
      if (n.tie) { if (!bgroups.has(n.tie)) bgroups.set(n.tie, []); bgroups.get(n.tie).push(n); }
    }
  }
  for (const [t, ns] of bgroups) {
    near(bwd.tie[t], ns.reduce((s, n) => s + bwd.node[n.id].dz, 0), 1e-12, `bias tie ${t}`);
    near(bwd.tie[t], fd(net, y, loss, () => ns[0].bias, v => ns.forEach(n => { n.bias = v; })), tol, `dL/d(${t})`);
  }
  net.layers.forEach((layer, l) => {
    if (layer.kind === 'attention') assert.ok(bwd.db[l].every(v => v === 0), `db[${l}] is 0 on an attention layer`);
  });
  return { fwd, bwd };
}

// A hand-wired token net covering the corners: X (n x dIn) -> tied Q, K, V -> attention (heads,
// causal, scale) -> H, a dense untied mix across every token with a per-token activation
// (softmax by default) -> Y (n x dOut), tokenwise tied from H, plus untied skip edges from the
// Q, K, V layer and fixed edges from X. Y's act is per token too (softmax + xent by default).
function tokenNet({ n = 3, dIn = 2, d = 2, heads = 1, causal = false, scale, hidden = 'softmax', out = 'softmax', dOut = 3, loss = 'xent', seed = 5 } = {}) {
  const r = M.rng(seed), w = () => 2 * r() - 1;
  const net = M.emptyNet();
  net.meta.loss = loss;
  const [lx, ly] = ids(net.layers);
  const lq = M.addLayer(net, 1, { size: 0, act: 'identity' });
  const lz = M.addLayer(net, 2, { size: 0, act: 'identity' });
  const lh = M.addLayer(net, 3, { size: 0, act: hidden });
  for (let k = 0; k < n * dIn; k++) M.addNode(net, lx, { value: w() });
  for (let k = 0; k < 3 * n * d; k++) M.addNode(net, lq, {});
  for (let k = 0; k < n * d; k++) M.addNode(net, lz, {});
  for (let k = 0; k < n * d; k++) M.addNode(net, lh, {});
  for (let k = 0; k < n * dOut; k++) M.addNode(net, ly, {});
  Object.assign(net.layers[0], { tokens: n });
  Object.assign(net.layers[1], { tokens: n, groups: ['Q', 'K', 'V'] });
  Object.assign(net.layers[2], { tokens: n, kind: 'attention', heads, causal }, scale === undefined ? {} : { scale });
  Object.assign(net.layers[3], { tokens: n });
  Object.assign(net.layers[4], { tokens: n, act: out });
  const X = M.nodesIn(net, 0), QKV = M.nodesIn(net, 1), Z = M.nodesIn(net, 2), H = M.nodesIn(net, 3), Y = M.nodesIn(net, 4);
  const edge = (a, b, wv, extra = {}) => net.edges.push({ id: M.uid(net, 'e'), from: a.id, to: b.id, w: wv, ...extra });
  ['Q', 'K', 'V'].forEach((g, gi) => {
    const W = Array.from({ length: dIn }, () => Array.from({ length: d }, w));
    const b = Array.from({ length: d }, () => 0.3 * w());
    for (let t = 0; t < n; t++) {
      for (let j = 0; j < d; j++) {
        const q = QKV[gi * n * d + t * d + j];
        q.bias = b[j]; q.tie = `b_${g}:${j + 1}`;
        for (let i = 0; i < dIn; i++) edge(X[t * dIn + i], q, W[i][j], { tie: `W_${g}:${i + 1},${j + 1}` });
      }
    }
  });
  for (const h of H) { h.bias = 0.2 * w(); for (const z of Z) edge(z, h, w()); }
  const WY = Array.from({ length: d }, () => Array.from({ length: dOut }, w));
  const bY = Array.from({ length: dOut }, () => 0.3 * w());
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < dOut; j++) {
      const o = Y[t * dOut + j];
      o.bias = bY[j]; o.tie = `b_Y:${j + 1}`;
      for (let i = 0; i < d; i++) edge(H[t * d + i], o, WY[i][j], { tie: `W_Y:${i + 1},${j + 1}` });
    }
  }
  edge(QKV[0], Y[0], 0.7); edge(QKV[n * d + 1], Y[1], -0.4); edge(QKV[2 * n * d], Y[dOut], 0.5);   // skips from Q, K, V
  edge(X[0], Y[2], 1, { fixed: true }); edge(X[1], Y[dOut + 1], -0.5, { fixed: true });
  // per-token one-hot targets (class t % dOut for token t), or soft targets for mse
  Y.forEach((o, k) => { o.target = loss === 'xent' ? (k % dOut === Math.floor(k / dOut) % dOut ? 1 : 0) : 0.4 * w(); });
  valid(net, 'tokenNet');
  return net;
}

describe('tokens: shapes, reshape, positions', () => {
  test('tokenShape / tokenPos: group-major, then token-major, then feature', () => {
    const net = M.PRESETS.attention.build(1);
    assert.deepEqual(M.tokenShape(net, 0), { tokens: 3, d: 2, groups: null });
    assert.deepEqual(M.tokenShape(net, 1), { tokens: 3, d: 2, groups: ['Q', 'K', 'V'] });
    assert.deepEqual(M.tokenShape(net, net.layers[2].id), { tokens: 3, d: 2, groups: null });
    M.nodesIn(net, 1).forEach((n, k) => {
      assert.deepEqual(M.tokenPos(net, n.id), { l: 1, index: k, g: Math.floor(k / 6), group: 'QKV'[Math.floor(k / 6)], token: Math.floor(k / 2) % 3, feature: k % 2 });
    });
    assert.deepEqual(M.tokenShape(M.PRESETS.xor.build(1), 1), { tokens: 1, d: 4, groups: null }, 'a plain layer is one token');
    assert.equal(M.tokenPos(net, 'ghost'), null);
    const s = M.tokenShape(net, 1);
    s.groups.push('W');
    assert.deepEqual(net.layers[1].groups, ['Q', 'K', 'V'], 'a copy');
  });

  test('reshape: tokens x d matrices per group, X for a layer without groups', () => {
    const net = M.PRESETS.attention.build(1), f = M.forward(net);
    const x = M.reshape(net, 0, f.a[0]);
    assert.deepEqual(Object.keys(x), ['X']);
    assert.deepEqual(x.X, [f.a[0].slice(0, 2), f.a[0].slice(2, 4), f.a[0].slice(4, 6)]);
    const qkv = M.reshape(net, 1, f.a[1]);
    assert.deepEqual(Object.keys(qkv), ['Q', 'K', 'V']);
    assert.deepEqual(qkv.Q, f.attn[2].heads[0].Q);
    assert.deepEqual(qkv.K, f.attn[2].heads[0].K);
    assert.deepEqual(qkv.V, f.attn[2].heads[0].V);
    assert.deepEqual(M.reshape(net, 2, f.a[2]).X, f.attn[2].heads[0].Z);
    assert.deepEqual(M.reshape(net, 0, [1]).X, [[1, 0], [0, 0], [0, 0]], 'missing entries read 0');
  });

  test('attnSpec: defaults filled, null for anything else', () => {
    assert.deepEqual(M.attnSpec(M.PRESETS.attention.build(1), 2), { l: 2, tokens: 3, d: 2, heads: 1, dh: 2, scale: 1 / Math.sqrt(2), causal: false });
    assert.deepEqual(M.attnSpec(M.PRESETS.multihead.build(1), 2), { l: 2, tokens: 3, d: 2, heads: 2, dh: 1, scale: 1, causal: false });
    assert.equal(M.attnSpec(M.PRESETS.causal.build(1), 2).causal, true);
    assert.equal(M.attnSpec(M.PRESETS.causal.build(1), 1), null);
    assert.equal(M.attnSpec(M.PRESETS.xor.build(1), 1), null);
  });

  test('matrices: an attention layer has no terms; dW stays parallel', () => {
    const net = M.PRESETS.transformer.build(1), mats = M.matrices(net), f = M.forward(net);
    const m = mats[1];
    assert.deepEqual(m, { l: 2, id: net.layers[2].id, kind: 'attention', act: 'identity', rows: ids(M.nodesIn(net, 2)), b: [0, 0, 0, 0], terms: [] });
    assert.ok(mats.filter(x => x.l !== 2).every(x => x.kind === 'dense'));
    const b = M.backward(net, f, targets(net));
    assert.deepEqual(b.dW[2], []);
    mats.forEach(x => assert.equal(b.dW[x.l].length, x.terms.length));
    assert.equal(M.collapse(net), null, 'attention is not affine');
  });
});

describe('attention forward', () => {
  test('every attention preset matches the formulas: S = QKᵀ/√d_k (+ mask), A = softmax(S), Z = AV', () => {
    for (const k of ATT_KEYS) {
      const net = M.PRESETS[k].build(3), l = attnLayer(net), spec = M.attnSpec(net, l);
      const r = M.rng(4);
      for (let s = 0; s < 5; s++) {
        const x = M.nodesIn(net, 0).map(() => 2 * r() - 1), f = M.forward(net, x);
        const { Q, K, V } = M.reshape(net, l - 1, f.a[l - 1]);
        const ref = refAttention(Q, K, V, spec);
        nearM(M.reshape(net, l, f.a[l]).X, ref.Z, 1e-12, `${k} Z`);
        assert.deepEqual(f.z[l], f.a[l], `${k}: identity`);
        assert.equal(f.attn[l].heads.length, spec.heads);
        f.attn[l].heads.forEach((h, hi) => {
          nearM(h.A, ref.heads[hi].A, 1e-12, `${k} A`);
          h.S.forEach((row, i) => row.forEach((v, j) => {
            if (ref.heads[hi].S[i][j] === -Infinity) { assert.equal(v, -Infinity, 'masked'); assert.equal(h.A[i][j], 0); }
            else near(v, ref.heads[hi].S[i][j], 1e-12);
          }));
          h.A.forEach(row => near(row.reduce((a, b) => a + b, 0), 1, 1e-12, 'A rows sum to 1'));
          nearM(h.Z, matmul(h.A, h.V), 1e-12, 'Z = A V');
        });
        f.attn.forEach((a, i) => assert.equal(a === null, i !== l));
        nearV(M.predict(net, [x])[0], f.a.at(-1), 1e-12, 'predict');
      }
    }
  });

  test('causal: token i reads only tokens up to i; a later token cannot change an earlier output', () => {
    const net = M.PRESETS.causal.build(1), f = M.forward(net);
    const A = f.attn[2].heads[0].A;
    assert.deepEqual(A[0], [1, 0, 0]);
    assert.equal(A[1][2], 0);
    const x = f.a[0].slice(), later = x.slice();
    later[6] += 3;   // token 3's content
    nearV(M.forward(net, later).a[3].slice(0, 2), f.a[3].slice(0, 2), 1e-15);
  });

  test('two heads read their own columns of Q, K, V and sit side by side in Z', () => {
    const net = M.PRESETS.multihead.build(2), f = M.forward(net), [h1, h2] = f.attn[2].heads;
    const { Q, V } = M.reshape(net, 1, f.a[1]);
    assert.deepEqual(h1.Q, Q.map(row => [row[0]]));
    assert.deepEqual(h2.Q, Q.map(row => [row[1]]));
    assert.deepEqual(h2.V, V.map(row => [row[1]]));
    assert.deepEqual(M.reshape(net, 2, f.a[2]).X, h1.Z.map((row, i) => [row[0], h2.Z[i][0]]));
    assert.equal(f.attn[2].scale, 1);
  });

  test('softmax on a token layer runs per token; xent there is the mean over tokens', () => {
    const net = tokenNet({ seed: 3 }), f = M.forward(net);
    for (const [l, d] of [[3, 2], [4, 3]]) {
      for (let t = 0; t < 3; t++) near(f.a[l].slice(t * d, t * d + d).reduce((a, b) => a + b, 0), 1, 1e-12, `layer ${l} token ${t}`);
      near(f.a[l].reduce((a, b) => a + b, 0), 3, 1e-12);
    }
    const y = targets(net), p = f.a[4];
    const want = [0, 1, 2].reduce((s, t) => s - [0, 1, 2].reduce((u, j) => u + y[t * 3 + j] * Math.log(p[t * 3 + j]), 0), 0) / 3;
    const b = M.backward(net, f, y, 'xent');
    near(b.loss, want, 1e-12);
    b.dZ[4].forEach((dz, i) => near(dz, (p[i] - y[i]) / 3, 1e-12, 'dZ = (p - y) / tokens'));
    assert.equal(b.note, null);
  });
});

describe('attention and tied gradients match finite differences', () => {
  for (const k of ATT_KEYS) {
    test(`${k} preset: every edge, tie group, bias and input`, () => {
      const net = M.PRESETS[k].build(2), y = targets(net);
      const relu = net.layers.map((l, i) => (l.act === 'relu' ? i : -1)).filter(i => i >= 0);
      const f = M.forward(net);
      relu.forEach(l => {
        assert.ok(f.z[l].every(z => Math.abs(z) > 1e-3), 'no ReLU sits on its kink');
        assert.ok(f.z[l].some(z => z > 0), 'some ReLU is live');
      });
      gradCheckShared(net, y, net.meta.loss);
    });
  }

  for (const [name, opts] of [
    ['causal, 2 heads, custom scale, softmax token layers, xent', { heads: 2, causal: true, scale: 0.8 }],
    ['non-causal, 1 head, tanh hidden, mse', { hidden: 'tanh', out: 'identity', loss: 'mse', seed: 9 }],
    ['4 tokens, d = 4, 2 heads, sigmoid output, xent', { n: 4, d: 4, heads: 2, out: 'sigmoid', dOut: 2, hidden: 'leaky', seed: 12 }],
  ]) {
    test(`hand-wired token net: ${name}`, () => {
      const net = tokenNet(opts);
      if (opts.hidden === 'leaky') avoidKinks(net);
      const y = targets(net);
      const { fwd, bwd } = gradCheckShared(net, y, net.meta.loss);
      // the attention formulas the matrix panel prints, with the model's numbers
      const l = 2, spec = M.attnSpec(net, l);
      bwd.attn[l].heads.forEach((g, hi) => {
        const { A, Q, K, V } = fwd.attn[l].heads[hi];
        nearM(g.dV, matmul(transpose(A), g.dZ), 1e-12, 'dV = Aᵀ dZ');
        nearM(g.dA, matmul(g.dZ, transpose(V)), 1e-12, 'dA = dZ Vᵀ');
        const dS = A.map((row, i) => row.map((a, j) => a * (g.dA[i][j] - row.reduce((s, b, m) => s + b * g.dA[i][m], 0))));
        nearM(g.dS, dS, 1e-12, 'dS = A ⊙ (dA - rowsum(dA ⊙ A))');
        nearM(g.dQ, matmul(g.dS, K).map(r => r.map(v => v * spec.scale)), 1e-12, 'dQ = dS K · scale');
        nearM(g.dK, matmul(transpose(g.dS), Q).map(r => r.map(v => v * spec.scale)), 1e-12, 'dK = dSᵀ Q · scale');
        g.dZ.forEach((row, i) => row.forEach((v, f) => near(v, bwd.dZ[l][i * spec.d + hi * spec.dh + f], 1e-15)));
      });
      // dL/d(Q, K, V layer) = the attention's own dQ, dK, dV + what the skip edges send back
      const qkv = M.nodesIn(net, 1), own = [];
      for (const key of ['dQ', 'dK', 'dV']) {
        for (let t = 0; t < spec.tokens; t++) for (let h = 0; h < spec.heads; h++) own.push(...bwd.attn[l].heads[h][key][t]);
      }
      const flat = [];   // reorder own (key, token, head, f) into node order (key, token, head·dh + f)
      for (let k = 0; k < 3; k++) for (let t = 0; t < spec.tokens; t++) for (let h = 0; h < spec.heads; h++) for (let f = 0; f < spec.dh; f++) {
        flat[k * spec.tokens * spec.d + t * spec.d + h * spec.dh + f] = bwd.attn[l].heads[h][['dQ', 'dK', 'dV'][k]][t][f];
      }
      qkv.forEach((n, i) => {
        const skip = net.edges.filter(e => e.from === n.id).reduce((s, e) => s + (M.nodeLayerIndex(net, e.to) > 2 ? e.w * bwd.node[e.to].dz : 0), 0);
        near(bwd.dA[1][i], flat[i] + skip, 1e-12, `dL/da ${n.id}`);
      });
    });
  }

  test('an attention output layer (mse) and a causal row with a single allowed token', () => {
    const net = M.PRESETS.causal.build(4);
    const b = gradCheckShared(net, targets(net), 'mse').bwd;
    const g = b.attn[2].heads[0];
    assert.ok(g.dS[0].every(v => v === 0), 'row 1 attends only to itself: no score gradient');
  });
});

describe('ties and fixed edges', () => {
  test('trainStep: a tie group steps by the mean over samples of its summed gradient; fixed edges and attention biases stay', () => {
    const net = tokenNet({ heads: 2, causal: true, seed: 21 });
    const ds = [0, 1, 2].map(s => { const r = M.rng(40 + s); return { x: M.nodesIn(net, 0).map(() => 2 * r() - 1), y: targets(net).map((v, i) => (i + s) % 3 === 0 ? 1 : 0) }; });
    const before = M.clone(net), lr = 0.4;
    const per = ds.map(({ x, y }) => M.backward(net, M.forward(net, x), y, 'xent'));
    const mean = M.trainStep(net, { X: ds.map(d => d.x), Y: ds.map(d => d.y) }, { lr });
    near(mean, per.reduce((s, b) => s + b.loss, 0) / ds.length, 1e-12);
    for (const e of before.edges) {
      const g = e.tie ? per.reduce((s, b) => s + b.tie[e.tie], 0) : per.reduce((s, b) => s + b.edge[e.id], 0);
      const want = e.fixed ? e.w : e.w - (lr * g) / ds.length;
      near(M.edge(net, e.id).w, want, 1e-12, e.id);
      if (e.fixed) assert.equal(M.edge(net, e.id).w, e.w, 'fixed');
    }
    for (const n of before.nodes) {
      const l = M.nodeLayerIndex(before, n.id);
      if (l === 0) continue;
      const g = n.tie ? per.reduce((s, b) => s + b.tie[n.tie], 0) : per.reduce((s, b) => s + b.node[n.id].dz, 0);
      const want = before.layers[l].kind === 'attention' ? 0 : n.bias - (lr * g) / ds.length;
      near(M.node(net, n.id).bias, want, 1e-12, n.id);
    }
    valid(net, 'ties stay exactly equal');
  });

  test('setWeight / connect set the whole group; fixed edges refuse; attention layers take no edges', () => {
    const net = M.PRESETS.transformer.build(1);
    const tied = net.edges.find(e => e.tie === 'W_Q:1,2');
    assert.equal(M.setWeight(net, tied.id, 0.625), true);
    assert.ok(net.edges.filter(e => e.tie === 'W_Q:1,2').every(e => e.w === 0.625));
    assert.equal(net.edges.filter(e => e.tie === 'W_Q:1,2').length, 2, 'one per token');
    const other = net.edges.filter(e => e.tie === 'W_Q:1,2')[1];
    assert.equal(M.connect(net, other.from, other.to, -0.25), other.id);
    assert.ok(net.edges.filter(e => e.tie === 'W_Q:1,2').every(e => e.w === -0.25));
    const fixed = net.edges.find(e => e.fixed);
    assert.equal(M.setWeight(net, fixed.id, 3), false);
    assert.equal(fixed.w, 1);
    assert.equal(M.connect(net, fixed.from, fixed.to, 3), fixed.id);
    assert.equal(fixed.w, 1, 'connect keeps a fixed weight');
    const z = M.nodesIn(net, 2), x = M.nodesIn(net, 0);
    assert.equal(M.connect(net, x[0].id, z[0].id, 1), null, 'nothing into an attention layer');
    assert.deepEqual(M.connectDense(net, 1, 2), []);
    assert.deepEqual(M.connectDense(net, 0, 2), []);
    valid(net);
  });

  test('setNode: a bias sets its tie group; an attention node keeps bias 0', () => {
    const net = M.PRESETS.attention.build(1);
    const q = M.nodesIn(net, 1)[3];   // Q, token 2, feature 2
    assert.equal(q.tie, 'b_Q:2');
    M.setNode(net, q.id, { bias: 0.75 });
    assert.deepEqual(M.nodesIn(net, 1).filter(n => n.tie === 'b_Q:2').map(n => n.bias), [0.75, 0.75, 0.75]);
    assert.ok(M.nodesIn(net, 1).filter(n => n.tie === 'b_Q:1').every(n => n.bias !== 0.75));
    const z = M.nodesIn(net, 2)[0];
    M.setNode(net, z.id, { bias: 2 });
    assert.equal(z.bias, 0);
    valid(net);
  });

  test('randomize: one draw per tie group, fixed edges kept, attention biases 0, seeded', () => {
    const a = M.PRESETS.transformer.build(1), b = M.clone(a);
    const fixed = a.edges.filter(e => e.fixed).map(e => e.w);
    M.randomize(a, { seed: 4, scheme: 'he', biases: 'small' });
    M.randomize(b, { seed: 4, scheme: 'he', biases: 'small' });
    assert.deepEqual(a, b);
    valid(a, 'groups stay equal');
    assert.deepEqual(a.edges.filter(e => e.fixed).map(e => e.w), fixed);
    const draws = new Set(a.edges.filter(e => e.tie).map(e => e.w));
    assert.equal(draws.size, new Set(a.edges.filter(e => e.tie).map(e => e.tie)).size, 'one value per group');
    assert.ok(M.nodesIn(a, 2).every(n => n.bias === 0));
    assert.ok(new Set(a.nodes.filter(n => n.tie).map(n => n.bias)).size > 1, 'bias groups drawn');
  });

  test('tiedMatrices: the small matrices in the X W convention, Q = X W_Q + b_Q', () => {
    const net = M.PRESETS.attention.build(2), f = M.forward(net);
    const T = M.tiedMatrices(net, 1);
    assert.deepEqual(T.map(t => [t.name, t.k, t.fromGroup, t.toGroup, t.tokenwise, t.edges.length]),
      [['W_Q', 0, null, 'Q', true, 12], ['W_K', 0, null, 'K', true, 12], ['W_V', 0, null, 'V', true, 12]]);
    const X = M.reshape(net, 0, f.a[0]).X, qkv = M.reshape(net, 1, f.a[1]), B = M.reshape(net, 1, M.matrices(net)[0].b);
    for (const t of T) {
      const g = t.toGroup;
      B[g].forEach(row => assert.deepEqual(row, B[g][0], 'tied bias rows are equal'));
      nearM(qkv[g], matmul(X, t.W).map(row => row.map((v, j) => v + B[g][0][j])), 1e-12, `${g} = X W_${g} + b_${g}`);
      assert.deepEqual(t.ties, [[`W_${g}:1,1`, `W_${g}:1,2`], [`W_${g}:2,1`, `W_${g}:2,2`]]);
    }
    // the layer matrix is I ⊗ Wᵀ: token 2's Q rows read token 2's x columns through W_Qᵀ
    const m = M.matrices(net)[0].terms[0], W = T[0].W;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      assert.equal(m.W[2 + i][2 + j], W[j][i]);
      assert.equal(m.edge[2 + i][0 + j], null, 'off the block diagonal is masked');
    }
    // fixed residual edges are left out; untied layers and inputs give []
    const tr = M.PRESETS.transformer.build(1);
    assert.deepEqual(M.tiedMatrices(tr, 3).map(t => [t.name, t.k, t.tokenwise]), [['W_O', 2, true]]);
    assert.deepEqual(M.tiedMatrices(tr, 5).map(t => [t.name, t.k, t.W.length, t.W[0].length]), [['W_2', 4, 4, 2]]);
    assert.deepEqual(M.tiedMatrices(tr, 2), []);
    assert.deepEqual(M.tiedMatrices(tr, 0), []);
    assert.deepEqual(M.tiedMatrices(M.PRESETS.xor.build(1), 1), []);
    const conv = M.tiedMatrices(M.PRESETS.conv1d.build(), 1);
    assert.deepEqual(conv.map(t => [t.name, t.W, t.tokenwise]), [['k', [[-1, 2, -1]], false]]);
  });
});

describe('retrofitted presets keep their weights shared', () => {
  const kernelOf = (net, l) => M.tiedMatrices(net, l).find(t => /^k/.test(t.name)).W[0];
  const train = (net, steps = 20, lr = 0.2) => {
    const r = M.rng(6), n0 = M.nodesIn(net, 0).length, nL = M.nodesIn(net, last(net)).length;
    for (let s = 0; s < steps; s++) {
      M.trainStep(net, { X: [Array.from({ length: n0 }, () => 2 * r() - 1)], Y: [Array.from({ length: nL }, () => r())] }, { lr, loss: 'mse' });
    }
    valid(net);
    return net;
  };

  for (const [k, l, stride] of [['conv1d', 1, 1], ['conv1d_s2', 1, 2], ['lenet', 1, 1]]) {
    test(`${k}: every conv edge is a tap of one tied kernel; training keeps W Toeplitz`, () => {
      const net = M.PRESETS[k].build();
      const into = net.edges.filter(e => M.nodeLayerIndex(net, e.to) === l);
      assert.ok(into.every(e => /^k:1,[123]$/.test(e.tie)));
      assert.ok(M.nodesIn(net, l).every(n => n.tie === 'b:1'));
      const k0 = kernelOf(net, l);
      train(net);
      const k1 = kernelOf(net, l), W = M.matrices(net)[l - 1].terms[0];
      assert.notDeepEqual(k1, k0, 'the kernel moved');
      W.edge.forEach((row, i) => row.forEach((e, j) => { if (e) assert.equal(W.W[i][j], k1[j - stride * i]); }));
      assert.equal(new Set(M.matrices(net)[l - 1].b).size, 1, 'one bias');
    });
  }

  test('lenet and avgpool: the pooling weights are fixed', () => {
    for (const [k, l] of [['lenet', 2], ['avgpool', 1]]) {
      const net = M.PRESETS[k].build(), pool = net.edges.filter(e => M.nodeLayerIndex(net, e.to) === l);
      assert.ok(pool.length && pool.every(e => e.fixed === true && e.w === 0.5), k);
      train(net);
      assert.ok(pool.every(e => e.w === 0.5), `${k}: unchanged`);
    }
  });

  test('wavenet: one tied kernel per dilation, taps (x_{t-d}, x_t)', () => {
    const net = M.PRESETS.wavenet.build();
    [1, 2, 4].forEach((d, i) => {
      const l = i + 1;
      assert.ok(net.edges.filter(e => M.nodeLayerIndex(net, e.to) === l).every(e => e.tie === `k^{(${l})}:1,${M.nodesIn(net, l).findIndex(n => n.id === e.to) === M.nodesIn(net, l - 1).findIndex(n => n.id === e.from) ? 2 : 1}`));
    });
    train(net, 10, 0.05);
    [1, 2, 4].forEach((d, i) => {
      const [back, now] = kernelOf(net, i + 1), t = M.matrices(net)[i].terms[0];
      t.edge.forEach((row, r) => row.forEach((e, j) => { if (e) assert.equal(t.W[r][j], j === r ? now : back, `dilation ${d}`); }));
    });
  });
});

describe('validate and normalize: tokens, ties, attention', () => {
  test('validate reports each new kind of damage, and normalize repairs it', () => {
    const cases = [
      [n => { n.layers[1].tokens = 2.5; }, /tokens must be an integer/],
      [n => { n.layers[1].groups = ['Q', 'Q', 'V']; }, /groups must be distinct/],
      [n => { n.layers[0].tokens = 4; }, /don't split into 4 tokens/],
      [n => { n.layers[2].kind = 'conv'; }, /unknown kind/],
      [n => { n.layers[2].act = 'relu'; }, /act must be identity/],
      [n => { n.layers[2].heads = 3; }, /heads must be an integer that divides/],
      [n => { n.layers[2].causal = 'yes'; }, /causal must be a boolean/],
      [n => { n.layers[2].scale = 'big'; }, /scale must be a finite number/],
      [n => { n.layers[2].groups = ['Q']; }, /attention layer has no groups/],
      [n => { n.layers[1].groups = ['A', 'B', 'C']; }, /must have groups Q, K, V/],
      [n => { n.layers[2].tokens = 2; }, /same tokens|don't split/],
      [n => { M.nodesIn(n, 2)[0].bias = 0.5; }, /attention layer's bias must be 0/],
      [n => { M.nodesIn(n, 2)[0].tie = 'b_Z:1'; }, /no bias to share/],
      [n => { n.edges.push({ id: 'e999', from: M.nodesIn(n, 1)[0].id, to: M.nodesIn(n, 2)[0].id, w: 1 }); }, /no incoming edges/],
      [n => { n.edges.find(e => e.tie).w += 1; }, /differing weights/],
      [n => { n.edges[0].tie = 7; }, /tie must be a non-empty string/],
      [n => { n.edges[0].tie = ''; }, /tie must be a non-empty string/],
      [n => { n.edges[0].fixed = 'no'; }, /fixed must be a boolean/],
      [n => { n.edges[0].fixed = true; }, /both fixed and tied/],
      [n => { M.nodesIn(n, 1)[0].bias += 1; }, /differing biases/],
      [n => { M.nodesIn(n, 1)[0].tie = n.edges[0].tie; }, /also an edge tie/],
      [n => { M.nodesIn(n, 0)[0].tie = 'b_X:1'; }, /input node has no bias to share/],
      [n => { n.layers[2].kind = 'attention'; n.layers.reverse(); }, /cannot be the input layer|before an attention layer/],
    ];
    for (const [breakIt, re] of cases) {
      const net = M.PRESETS.attention.build(1);
      breakIt(net);
      const errs = M.validate(net);
      assert.ok(errs.some(e => re.test(e)), `${re}: ${errs.join(' | ')}`);
      const fixed = M.normalize(net);
      valid(fixed, `normalize fixes ${re}`);
      assert.ok(allFinite(M.forward(fixed).a));
    }
  });

  test('normalize: specific repairs', () => {
    const base = M.PRESETS.attention.build(1);
    const n1 = M.clone(base);
    n1.edges.push({ id: 'stray', from: M.nodesIn(n1, 0)[0].id, to: M.nodesIn(n1, 2)[0].id, w: 1 });
    M.nodesIn(n1, 2)[1].bias = 3;
    n1.layers[2].act = 'tanh';
    const r1 = M.normalize(n1);
    assert.deepEqual(r1, base, 'edges into attention dropped, biases zeroed, act identity');

    const n2 = M.clone(base);
    const tied = n2.edges.filter(e => e.tie === 'W_K:2,1');
    tied[1].w = 9; tied[2].w = -9;
    assert.deepEqual(M.normalize(n2), base, 'a tie group takes its first member\'s weight');

    const n3 = M.clone(base);
    Object.assign(n3.layers[1], { tokens: '3' });
    Object.assign(n3.layers[2], { kind: ' Attention ', heads: '1', causal: 'false' });
    n3.edges[0].fixed = 'false';
    const r3 = M.normalize(n3);
    assert.deepEqual([r3.layers[1].tokens, r3.layers[2].kind, r3.layers[2].heads, r3.layers[2].causal, r3.edges[0].fixed], [3, 'attention', 1, false, false]);
    valid(r3);

    const n4 = M.clone(base);
    n4.edges[0].fixed = true;   // fixed wins over tie
    const r4 = M.normalize(n4);
    assert.equal(r4.edges[0].fixed, true);
    assert.ok(!('tie' in r4.edges[0]));
    valid(r4);

    const n5 = M.clone(base);
    n5.nodes = n5.nodes.filter(n => n !== M.nodesIn(n5, 1)[17]);   // Q, K, V loses a node
    const r5 = M.normalize(n5);
    valid(r5);
    assert.ok(!('groups' in r5.layers[1]) && !('tokens' in r5.layers[1]), 'plain vector');
    assert.ok(!('kind' in r5.layers[2]) && !('heads' in r5.layers[2]), 'attention demoted to dense');
    assert.equal(r5.layers[2].tokens, 3, 'but still 3 tokens');

    const n6 = M.clone(base);
    M.nodesIn(n6, 1)[0].tie = 5;
    assert.equal(M.normalize(n6).nodes.find(n => n.id === M.nodesIn(n6, 1)[0].id).tie, '5');
  });

  test('old nets load unchanged; token nets survive a save', () => {
    for (const k of Object.keys(M.PRESETS)) {
      const net = M.PRESETS[k].build(1);
      assert.deepEqual(M.normalize(JSON.parse(JSON.stringify(net))), net, k);
    }
    const legacy = { v: 1, layers: [{ id: 'L1', name: 'In', act: 'identity' }, { id: 'L2', name: 'Out', act: 'sigmoid' }],
      nodes: [{ id: 'n3', layer: 'L1', x: 1, y: 2, label: 'x_{1}', bias: 0, value: 1, target: null, params: {} },
        { id: 'n4', layer: 'L2', x: 3, y: 2, label: '\\hat y_{1}', bias: 0.5, value: 0, target: 1, params: {} }],
      edges: [{ id: 'e5', from: 'n3', to: 'n4', w: -2 }], meta: { title: 'Old', loss: 'xent', nextId: 6, train: {} } };
    assert.deepEqual(M.normalize(M.clone(legacy)), legacy);
    for (const e of M.normalize(M.clone(legacy)).edges) assert.ok(!('tie' in e) && !('fixed' in e), 'no fields added');
  });
});

describe('edits never break a token net', () => {
  test('removing or adding a Q, K, V node turns the layer plain and the attention layer dense', () => {
    for (const edit of [
      n => M.removeNode(n, M.nodesIn(n, 1)[4].id),
      n => M.addNode(n, 1, { connect: true, seed: 2 }),
      n => M.setNode(n, M.nodesIn(n, 1)[0].id, { layer: 0 }),
    ]) {
      const net = M.PRESETS.attention.build(1);
      edit(net);
      valid(net);
      assert.equal(net.layers[1].groups, undefined);
      assert.equal(net.layers[2].kind, undefined);
      assert.ok(allFinite(M.forward(net).a));
    }
  });

  test('a layer between Q, K, V and attention demotes it; removing Q, K, V too; removing attention is clean', () => {
    const a = M.PRESETS.attention.build(1);
    const id = M.addLayer(a, 2, { size: 3, dense: true, seed: 1 });
    valid(a);
    assert.equal(a.layers[3].kind, undefined);
    assert.equal(M.nodesIn(a, 3).length, 6);
    assert.ok(a.edges.some(e => M.nodeLayerIndex(a, e.from) === 2 && M.nodeLayerIndex(a, e.to) === 3), 'wired densely');
    assert.equal(a.layers[2].id, id);

    const b = M.PRESETS.causal.build(1);
    M.removeLayer(b, b.layers[1].id, { bridge: true, seed: 3 });
    valid(b);
    assert.equal(b.layers[1].kind, undefined);

    const c = M.PRESETS.transformer.build(1);
    M.removeLayer(c, c.layers[2].id, { bridge: true, seed: 3 });
    valid(c);
    assert.deepEqual(c.layers[1].groups, ['Q', 'K', 'V'], 'Q, K, V stays a token layer');
    assert.ok(M.tiedMatrices(c, 1).length === 3);

    const d = M.PRESETS.attention.build(1);
    M.addNode(d, 0, {});
    valid(d);
    assert.equal(d.layers[0].tokens, undefined, 'X becomes a plain vector');
    assert.equal(d.layers[2].kind, 'attention', 'the attention layer is untouched');
  });

  test('setLayer on an attention layer: causal, heads (only if they divide d), scale; act stays identity', () => {
    const net = M.PRESETS.attention.build(1), id = net.layers[2].id;
    assert.equal(M.setLayer(net, id, { act: 'relu', causal: true, scale: 0.5 }), true);
    assert.deepEqual([net.layers[2].act, net.layers[2].causal, net.layers[2].scale], ['identity', true, 0.5]);
    M.setLayer(net, id, { heads: 2 });
    assert.equal(net.layers[2].heads, 2);
    M.setLayer(net, id, { heads: 3 });
    assert.equal(net.layers[2].heads, 2, 'heads must divide d');
    M.setLayer(net, id, { scale: null });
    assert.equal('scale' in net.layers[2], false);
    assert.equal(M.attnSpec(net, 2).scale, 1);
    valid(net);
    assert.equal(M.forward(net).attn[2].heads.length, 2);
  });

  test('fuzz: 500 random edits on the transformer never produce an invalid net', () => {
    const r = M.rng(77);
    const pick = a => a[Math.floor(r() * a.length)];
    const net = M.PRESETS.transformer.build(1);
    const acts = Object.keys(M.ACTS);
    const ops = [
      () => M.addLayer(net, Math.floor(r() * (net.layers.length + 1)), { size: Math.floor(r() * 4), act: pick(acts), dense: r() < 0.5, seed: 1 }),
      () => net.layers.length > 3 && M.removeLayer(net, pick(net.layers).id, { bridge: r() < 0.5, seed: 2 }),
      () => M.setLayer(net, pick(net.layers).id, { act: pick(acts), heads: 1 + Math.floor(r() * 3), causal: r() < 0.5, scale: r() < 0.3 ? null : r() * 2 }),
      () => r() < 0.3 && M.addNode(net, Math.floor(r() * net.layers.length), { connect: r() < 0.5 }),
      () => r() < 0.3 && net.nodes.length && M.removeNode(net, pick(net.nodes).id),
      () => net.nodes.length && M.setNode(net, pick(net.nodes).id, { bias: r() - 0.5, layer: r() < 0.1 ? Math.floor(r() * net.layers.length) : undefined }),
      () => net.nodes.length > 1 && M.connect(net, pick(net.nodes).id, pick(net.nodes).id, r() * 2 - 1),
      () => net.edges.length && M.disconnect(net, pick(net.edges).id),
      () => net.edges.length && M.setWeight(net, pick(net.edges).id, r() * 4 - 2),
      () => M.connectDense(net, Math.floor(r() * net.layers.length), Math.floor(r() * net.layers.length), { seed: 3 }),
      () => M.randomize(net, { seed: Math.floor(r() * 100), scheme: pick(['xavier', 'he', 'small']), biases: pick(['zero', 'small']) }),
      () => M.trainStep(net, { X: [M.nodesIn(net, 0).map(() => r())], Y: [M.nodesIn(net, last(net)).map(() => r())] }, { lr: 0.05, loss: 'mse' }),
      () => M.autoLayout(net),
    ];
    for (let step = 0; step < 500; step++) {
      pick(ops)();
      assert.deepEqual(M.validate(net), [], `after step ${step}`);
      const f = M.forward(net);
      const y = targets(net);
      if (y.length && y.every(Number.isFinite)) assert.equal(M.backward(net, f, y, 'mse').dW.length, net.layers.length);
    }
    assert.deepEqual(M.normalize(M.clone(net)), net);
  });
});

describe('sequence datasets', () => {
  const SEQ = { seq_max: [3, 2, 2], seq_minmax: [3, 2, 2], seq_prev: [3, 3, 1], seq_addmax: [2, 2, 2] };
  const tok = (row, n) => Array.from({ length: n }, (_, t) => row.slice((t * row.length) / n, ((t + 1) * row.length) / n));
  for (const [k, [n, dIn, dOut]] of Object.entries(SEQ)) {
    test(`${k}: kind seq, ${n} tokens x ${dIn} -> ${dOut}, deterministic, targets follow the (noisy) tokens`, () => {
      const d = M.DATASETS[k];
      assert.deepEqual([d.kind, d.tokens, d.inputs, d.outputs], ['seq', n, n * dIn, n * dOut]);
      assert.equal(typeof d.label, 'string');
      for (const noise of [0, 0.1]) {
        const { X, Y } = d.make(150, 3, noise);
        assert.equal(X.length, 150);
        assert.ok(X.every(x => x.length === n * dIn && x.every(Number.isFinite)) && Y.every(y => y.length === n * dOut && y.every(Number.isFinite)));
        assert.deepEqual(d.make(150, 3, noise), { X, Y });
        X.forEach((x, s) => {
          const T = tok(x, n), y = tok(Y[s], n), a = T.map(t => t[0]), top = T[a.indexOf(Math.max(...a))];
          if (k === 'seq_max') y.forEach(row => assert.deepEqual(row, top));
          if (k === 'seq_minmax') y.forEach(row => assert.deepEqual(row, [Math.max(...a), Math.min(...a)]));
          if (k === 'seq_addmax') y.forEach((row, t) => assert.deepEqual(row, T[t].map((v, f) => Math.max(0, v + top[f]))));
          if (k === 'seq_prev') {
            y.forEach((row, t) => assert.deepEqual(row, [T[Math.max(0, t - 1)][0]]));
            T.forEach((t, j) => { near(Math.hypot(t[1], t[2]), 1, 1e-12); near(Math.atan2(t[2], t[1]), [0, 2 * Math.PI / 3, -2 * Math.PI / 3][j], 1e-12); });
          }
          if (k !== 'seq_prev' && !noise) {
            assert.ok(a.every((u, i) => a.every((v, j) => i === j || Math.abs(u - v) >= 0.3)), 'first features at least 0.3 apart');
            assert.ok(x.every(v => Math.abs(v) <= 1));
          }
        });
      }
      assert.notDeepEqual(d.make(50, 4, 0).X, d.make(50, 3, 0).X);
      assert.notDeepEqual(d.make(50, 3, 0.2).X, d.make(50, 3, 0).X);
    });
  }

  test('the maximum and minimum really vary (no constant answers)', () => {
    const { Y } = M.DATASETS.seq_minmax.make(300, 1, 0);
    const sd = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length - (v.reduce((s, x) => s + x, 0) / v.length) ** 2);
    assert.ok(sd(Y.map(y => y[0])) > 0.2 && sd(Y.map(y => y[1])) > 0.2);
    // nor are they the mean plus a constant (uniform attention would get those)
    const { X } = M.DATASETS.seq_minmax.make(300, 1, 0);
    const gap = X.map((x, s) => Y[s][0] - (x[0] + x[2] + x[4]) / 3);
    assert.ok(sd(gap) > 0.1);
  });
});

describe('hand-set attention presets mean what they say', () => {
  const argmax = row => row.indexOf(Math.max(...row));

  test('words: sat reads cat, cat reads the, and the (q = 0) reads all three evenly', () => {
    const net = M.PRESETS.words.build(), f = M.forward(net), l = attnLayer(net);
    assert.deepEqual(net.meta.tokenNames, ['the', 'cat', 'sat']);
    assert.equal(net.meta.tokenNames.length, M.tokenShape(net, 0).tokens, 'one name per token');
    assert.deepEqual(M.reshape(net, 0, f.a[0]).X, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'one-hot det, noun, verb');
    const { Q, K, S, A } = f.attn[l].heads[0];
    assert.deepEqual(Q[0], [0, 0], 'the asks nothing');
    nearV(A[0], [1 / 3, 1 / 3, 1 / 3], 1e-12, 'a zero query scores every key 0');
    assert.equal(argmax(A[1]), 0, 'cat reads the');
    assert.equal(argmax(A[2]), 1, 'sat reads cat');
    assert.ok(A[1][0] > 0.85 && A[2][1] > 0.85, `peaked: ${A[1][0]}, ${A[2][1]}`);
    // X is the identity, so S is W_Q W_Kᵀ · scale: the tied matrices alone say who reads whom
    const T = M.tiedMatrices(net, 1), Wq = T.find(t => t.name === 'W_Q').W, Wk = T.find(t => t.name === 'W_K').W;
    nearM(S, matmul(Wq, transpose(Wk)).map(r => r.map(v => v / Math.sqrt(2))), 1e-12, 'S = W_Q W_Kᵀ / √2');
    nearM(Q, Wq, 1e-12, 'Q = W_Q'); nearM(K, Wk, 1e-12, 'K = W_K');
    assert.equal(M.backward(net, f, targets(net)).loss, 0, 'targets are the net\'s own output');
    assert.deepEqual(M.normalize(JSON.parse(JSON.stringify(net))).meta.tokenNames, ['the', 'cat', 'sat'], 'names survive a save');
  });

  test('causal_rot: W_Q is 4 R(120°) on the positions, so q_i = 4 k_(i-1) and each token copies the one before', () => {
    const net = M.PRESETS.causal_rot.build(), T = M.tiedMatrices(net, 1);
    const Wq = T.find(t => t.name === 'W_Q').W, c = Math.cos(2 * Math.PI / 3), s = Math.sin(2 * Math.PI / 3);
    nearM(Wq.slice(1), [[4 * c, -4 * s], [4 * s, 4 * c]], 1e-3, 'rotation block');
    near(Wq[1][0] * Wq[2][1] - Wq[1][1] * Wq[2][0], 16, 1e-2, 'det = 16: a rotation scaled by 4');
    const d = M.DATASETS.seq_prev.make(100, 7, 0);
    let loss = 0;
    d.X.forEach((x, n) => {
      const f = M.forward(net, x), { Q, K, A } = f.attn[2].heads[0];
      for (const i of [1, 2]) {
        nearV(Q[i], K[i - 1].map(v => 4 * v), 1e-3, `q_${i + 1} = 4 k_${i}`);
        assert.ok(A[i][i - 1] > 0.95, `sample ${n}: token ${i + 1} reads token ${i} (${A[i][i - 1]})`);
      }
      assert.deepEqual(A[0], [1, 0, 0], 'the mask leaves token 1 only itself');
      loss += M.backward(net, f, d.Y[n], 'mse').loss / d.X.length;
    });
    assert.ok(loss < 1e-3, `already solves Previous token: loss ${loss}`);
  });
});

describe('attention presets train on their datasets', () => {
  // As the Train panel runs them: 200 points, noise 0.1, batches of 10, the preset's own lr.
  function run(k, steps = 3000) {
    const p = M.PRESETS[k], net = p.build(1);
    assert.equal(net.meta.train.lr, p.lr);
    const d = M.DATASETS[p.dataset].make(200, 1, 0.1), test = M.DATASETS[p.dataset].make(200, 7, 0);
    const r = M.rng(3);
    const lossOn = D => D.X.reduce((s, x, i) => s + M.backward(net, M.forward(net, x), D.Y[i], 'mse').loss, 0) / D.X.length;
    const before = lossOn(test);
    for (let s = 0; s < steps; s++) {
      const idx = Array.from({ length: 10 }, () => Math.floor(r() * 200));
      M.trainStep(net, { X: idx.map(i => d.X[i]), Y: idx.map(i => d.Y[i]) }, { lr: p.lr });
    }
    valid(net);
    // mean attention weight on the token each head should find, over the held-out set
    const on = [0, 0], onMin = [0, 0];
    let rows = 0;
    test.X.forEach(x => {
      const at = M.forward(net, x).attn.find(Boolean), n = at.tokens, a = Array.from({ length: n }, (_, j) => x[(j * x.length) / n]);
      const jmax = a.indexOf(Math.max(...a)), jmin = a.indexOf(Math.min(...a));
      at.heads.forEach((h, hi) => h.A.forEach((row, i) => {
        on[hi] += k === 'causal' ? (i ? row[i - 1] : row[0]) : row[jmax];
        onMin[hi] += row[jmin];
        if (!hi) rows++;
      }));
    });
    return { net, before, after: lossOn(test), on: on.map(v => v / rows), onMin: onMin.map(v => v / rows) };
  }

  test('attention: every token learns to look at the largest x₁', () => {
    const t = run('attention');
    assert.ok(t.after < 2e-3 && t.after < t.before / 50, `${t.before} -> ${t.after}`);
    assert.ok(t.on[0] > 0.9, `A on the max: ${t.on[0]}`);
  });

  test('causal: token i looks at token i - 1, and W_Q W_Kᵀ turns the positions by one', () => {
    const t = run('causal');
    assert.ok(t.after < 1e-3 && t.after < t.before / 50, `${t.before} -> ${t.after}`);
    assert.ok(t.on[0] > 0.9, `A on the previous token: ${t.on[0]}`);
    // the positional block of W_Q W_Kᵀ scores token j - 1's position highest for token j
    const T = M.tiedMatrices(t.net, 1), Wq = T[0].W, Wk = T[1].W;
    const pos = [[1, 0], [-0.5, Math.sqrt(3) / 2], [-0.5, -Math.sqrt(3) / 2]];
    const P = pos.map(p => [0, ...p]);
    const Sq = matmul(matmul(P, Wq), transpose(matmul(P, Wk)));
    for (const i of [1, 2]) assert.equal(Sq[i].indexOf(Math.max(...Sq[i].slice(0, i + 1))), i - 1, `row ${i}`);
  });

  test('multihead: one head finds the largest x₁, the other the smallest', () => {
    const t = run('multihead');
    assert.ok(t.after < 2e-3 && t.after < t.before / 50, `${t.before} -> ${t.after}`);
    const [a, b] = t.on[0] > t.on[1] ? [0, 1] : [1, 0];
    assert.ok(t.on[a] > 0.6 && t.onMin[b] > 0.6, `max head ${t.on[a]}, min head ${t.onMin[b]}`);
    assert.ok(t.onMin[a] < 0.15 && t.on[b] < 0.15);
  });

  test('transformer: attention finds the max token, the FFN clips at 0; fixed residuals stay 1', () => {
    const t = run('transformer');
    assert.ok(t.after < 5e-3 && t.after < t.before / 20, `${t.before} -> ${t.after}`);
    assert.ok(t.on[0] > 0.8, `A on the max: ${t.on[0]}`);
    assert.ok(t.net.edges.filter(e => e.fixed).every(e => e.w === 1));
  });
});

// ---------------------------------------------------------------- word datasets and presets

describe('word datasets', () => {
  const WORD_KEYS = ['nl_pronoun', 'nl_agree'];
  // A tiny grammar check: the kind of each word, and number agreement.
  const KIND = { dog: 'N', cat: 'N', dogs: 'N', cats: 'N', sees: 'V', see: 'V', chases: 'V', chase: 'V', itself: 'R', themselves: 'R' };
  const NUMBER = { dog: 'sg', cat: 'sg', sees: 'sg', chases: 'sg', itself: 'sg', dogs: 'pl', cats: 'pl', see: 'pl', chase: 'pl', themselves: 'pl' };

  test('WORDS: a 5 x 2 grid, x = the kind of word, y = +0.6 singular and -0.6 plural', () => {
    assert.deepEqual(Object.keys(M.WORDS).sort(), Object.keys(KIND).sort());
    const x = { dog: 1.2, cat: 0.6, itself: 0, sees: -0.6, chases: -1.2 };
    const pl = { dog: 'dogs', cat: 'cats', itself: 'themselves', sees: 'see', chases: 'chase' };
    for (const [sg, v] of Object.entries(x)) {
      assert.deepEqual(M.WORDS[sg], [v, 0.6], sg);
      assert.deepEqual(M.WORDS[pl[sg]], [v, -0.6], pl[sg]);
    }
    assert.ok(Object.isFrozen(M.WORDS) && Object.isFrozen(M.WORDS.dog));
  });

  for (const k of WORD_KEYS) {
    test(`${k}: 3 words x 2 -> 3 x 2, deterministic, grammatical, targets are words of the sentence`, () => {
      const d = M.DATASETS[k];
      assert.deepEqual([d.kind, d.tokens, d.inputs, d.outputs], ['seq', 3, 6, 6]);
      assert.equal(typeof d.label, 'string');
      assert.ok(Object.keys(d.vocab).every(w => M.WORDS[w] === d.vocab[w]), 'vocab is a part of WORDS');
      for (const noise of [0, 0.1]) {
        const S = d.make(300, 3, noise);
        assert.deepEqual(d.make(300, 3, noise), S, 'same seed, same data');
        assert.deepEqual([S.X.length, S.Y.length, S.words.length, S.targetWords.length], [300, 300, 300, 300]);
        S.words.forEach((w, s) => {
          const [a, b, c] = w;
          assert.ok(w.every(v => d.vocab[v]), `${w}: known words`);
          if (k === 'nl_pronoun') {
            assert.deepEqual(w.map(v => KIND[v]), ['N', 'V', 'R'], w.join(' '));
            assert.ok(NUMBER[a] === NUMBER[b] && NUMBER[b] === NUMBER[c], `${w.join(' ')}: agreement`);
            assert.deepEqual(S.targetWords[s], [a, b, a], 'the reflexive outputs its noun');
          } else {
            assert.deepEqual(w.map(v => KIND[v]), ['N', 'V', 'N'], w.join(' '));
            assert.equal(NUMBER[a], NUMBER[b], `${w.join(' ')}: the verb agrees with its subject`);
            assert.notEqual(NUMBER[a], NUMBER[c], `${w.join(' ')}: the object has the other number`);
            assert.deepEqual(S.targetWords[s], [a, a, c], 'the verb outputs its subject');
          }
          // X is the words' vectors (plus noise); Y copies the vectors of the target positions
          const src = S.targetWords[s].map(t => w.indexOf(t));
          const T = [0, 1, 2].map(t => S.X[s].slice(2 * t, 2 * t + 2));
          assert.deepEqual(S.Y[s], src.flatMap(j => T[j]), 'targets follow the (noisy) tokens');
          if (!noise) assert.deepEqual(T, w.map(v => [...M.WORDS[v]]));
          else T.forEach((v, t) => assert.ok(Math.hypot(v[0] - M.WORDS[w[t]][0], v[1] - M.WORDS[w[t]][1]) < 0.6, 'noise is small'));
        });
        // every sentence of the grammar turns up
        const kinds = new Set(S.words.map(w => w.join(' ')));
        assert.equal(kinds.size, k === 'nl_pronoun' ? 8 : 16, [...kinds].join(', '));
      }
      // the noise has its own random stream: a seed gives the same sentences at any noise level
      assert.deepEqual(d.make(50, 3, 0.2).words, d.make(50, 3, 0).words);
      assert.notDeepEqual(d.make(50, 3, 0.2).X, d.make(50, 3, 0).X);
      assert.notDeepEqual(d.make(50, 4, 0).words, d.make(50, 3, 0).words);
    });

    test(`${k}: decode names each token by its nearest word`, () => {
      const d = M.DATASETS[k], { X, words } = d.make(40, 5, 0);
      X.forEach((x, s) => assert.deepEqual(d.decode(x), words[s]));
      const w = Object.keys(d.vocab)[0], [a, b] = d.vocab[w];
      assert.deepEqual(d.decode([a + 0.2, b - 0.2, NaN, 0, a, b]), [w, null, w]);
      if (k === 'nl_agree') assert.ok(!('itself' in d.vocab), 'only the words the task uses');
    });
  }
});

describe('word presets train on their datasets', () => {
  // As the Train panel runs them: 200 points, batches of 10, the preset's own lr and noise (0). Word
  // accuracy and attention are measured on a held-out set without noise.
  function run(k, { steps = 3000, init, noise = M.PRESETS[k].noise } = {}) {
    const p = M.PRESETS[k], net = p.build(1), ds = M.DATASETS[p.dataset];
    if (init) init(net);
    const d = ds.make(200, 1, noise), test = ds.make(200, 7, 0), r = M.rng(3);
    for (let s = 0; s < steps; s++) {
      const idx = Array.from({ length: 10 }, () => Math.floor(r() * 200));
      M.trainStep(net, { X: idx.map(i => d.X[i]), Y: idx.map(i => d.Y[i]) }, { lr: p.lr });
    }
    valid(net);
    let hit = 0;
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], argmax = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    test.X.forEach((x, s) => {
      const f = M.forward(net, x);
      ds.decode(f.a.at(-1)).forEach((w, t) => { if (w === test.targetWords[s][t]) hit++; });
      f.attn[2].heads[0].A.forEach((row, i) => {
        row.forEach((v, j) => { A[i][j] += v / test.X.length; });
        argmax[i][row.indexOf(Math.max(...row))]++;
      });
    });
    return { net, acc: hit / (3 * test.X.length), A, argmax };
  }

  test('pronouns and agreement start as the note says: W_V = I, b_V = 0, small W_Q and W_K, the first sentence named', () => {
    for (const k of ['pronouns', 'agreement']) {
      const p = M.PRESETS[k], net = p.build(1), T = M.tiedMatrices(net, 1);
      assert.deepEqual([net.meta.train.lr, net.meta.train.noise], [0.3, 0], 'lr 0.3 on the exact word vectors');
      assert.deepEqual(net.meta.train.init, { W_Q: 'small', W_K: 'small', W_V: 'identity' }, 'the recipe Reset uses');
      assert.deepEqual(T.find(t => t.name === 'W_V').W, [[1, 0], [0, 1]]);
      assert.ok(M.nodesIn(net, 1).every(n => n.bias === 0), 'biases start at 0');
      const again = M.clone(net);
      M.randomize(again, { seed: 1, scheme: 'xavier', init: net.meta.train.init });
      assert.deepEqual(again, net, 'Reset with init seed 1 gives back the preset');
      for (const nm of ['W_Q', 'W_K']) assert.ok(T.find(t => t.name === nm).W.flat().every(v => Math.abs(v) < 0.4), nm);
      const { words } = M.DATASETS[p.dataset].make(1, 1, 0);
      assert.deepEqual(net.meta.tokenNames, words[0]);
      assert.deepEqual(M.reshape(net, 0, M.forward(net).a[0]).X, words[0].map(w => [...M.WORDS[w]]), 'the inputs are its words');
      // before training every word reads all three about evenly
      M.forward(net).attn[2].heads[0].A.flat().forEach(v => assert.ok(v > 0.25 && v < 0.42, `${k}: ${v}`));
    }
  });

  test('pronouns: itself and themselves read the noun they refer to; the noun and the verb read themselves', () => {
    const t = run('pronouns');
    assert.ok(t.acc > 0.99, `word accuracy ${t.acc}`);
    assert.deepEqual(t.argmax, [[200, 0, 0], [0, 200, 0], [200, 0, 0]], 'row argmax of A on every held-out sentence');
    assert.ok(t.A[2][0] > 0.9, `the reflexive's weight on its noun: ${t.A[2][0]}`);
    assert.ok(t.A[0][0] > 0.9 && t.A[1][1] > 0.85, `self: ${t.A[0][0]}, ${t.A[1][1]}`);
  });

  test('agreement: the verb reads its subject, never the object, which has the other number', () => {
    const t = run('agreement');
    assert.ok(t.acc > 0.99, `word accuracy ${t.acc}`);
    assert.deepEqual(t.argmax, [[200, 0, 0], [200, 0, 0], [0, 0, 200]], 'row argmax of A on every held-out sentence');
    assert.ok(t.A[1][0] > 0.9 && t.A[1][2] < 0.03, `the verb on its subject ${t.A[1][0]}, on the object ${t.A[1][2]}`);
  });

  test('with noisy word vectors (noise 0.1) both still learn their pattern', () => {
    for (const k of ['pronouns', 'agreement']) {
      const t = run(k, { noise: 0.1 });
      assert.ok(t.acc > 0.99, `${k}: word accuracy ${t.acc}`);
      assert.ok(k === 'pronouns' ? t.A[2][0] > 0.9 : t.A[1][0] > 0.9, `${k}: ${JSON.stringify(t.A)}`);
    }
  });

  test('from a random W_V a run can lock into a swap instead (the reason the presets start at W_V = I)', () => {
    const t = run('pronouns', { init: net => M.randomize(net, { seed: 8, scheme: 'xavier' }) });
    assert.ok(t.acc < 0.7, `word accuracy ${t.acc}`);
    assert.ok(t.argmax[0][1] > 150, 'the noun reads the verb');
    const W = M.tiedMatrices(t.net, 1).find(x => x.name === 'W_V').W;
    assert.ok(W[0][0] * W[1][1] - W[0][1] * W[1][0] < 0, 'and W_V mirrors it back: det < 0');
  });

  test('Reset keeps the recipe (meta.train.init), so a new init seed trains as well as the preset', () => {
    for (const k of ['pronouns', 'agreement']) {
      for (const seed of [2, 3, 8, 9]) {
        const r = run(k, { init: net => M.randomize(net, { seed, scheme: 'xavier', init: net.meta.train.init }) });
        assert.ok(r.acc > 0.99, `${k}, init seed ${seed}: ${r.acc}`);
      }
    }
  });
});

// ---------------------------------------------------------------- train.js pure helpers

describe('train.js readouts agree with the model', async () => {
  const T = await import('../static/nn/train.js');

  test('a preset trains on the dataset it records; legacy nets still match by title', () => {
    for (const [k, p] of Object.entries(M.PRESETS)) {
      if (!p.dataset) continue;
      const net = p.build(1);
      assert.equal(T.readSettings(net, M).dataset, p.dataset, k);
      const renamed = M.clone(net);
      renamed.meta.title = 'My own net';
      assert.equal(T.readSettings(renamed, M).dataset, p.dataset, `${k}: the title no longer matters`);
      const legacy = M.clone(net);
      legacy.meta.train = {};   // saved before presets recorded a dataset
      assert.equal(T.readSettings(legacy, M).dataset, p.dataset, `${k}: legacy title fallback`);
    }
    const fresh = M.emptyNet();
    M.addNode(fresh, 0, {}); M.addNode(fresh, 0, {}); M.addNode(fresh, 1, {});
    assert.equal(T.readSettings(fresh, M).dataset, 'xor', 'first dataset that fits 2 -> 1');
  });

  test('datasetLoss is the mean of model.backward losses, per loss head (incl. the xent fallback)', () => {
    const cases = [
      ['xor', 'xent'], ['classifier', 'xent'], ['autoencoder', 'mse'], ['linear', 'mse'], ['residual', 'xent'],
    ];
    for (const [k, loss] of cases) {
      const net = M.PRESETS[k].build(2);
      const ins = M.nodesIn(net, 0).length, outs = M.nodesIn(net, last(net));
      const r = M.rng(5);
      const X = Array.from({ length: 7 }, () => Array.from({ length: ins }, () => 2 * r() - 1));
      const Y = X.map(() => outs.map((_, i) => (k === 'classifier' ? (i === 1 ? 1 : 0) : Math.round(r()))));
      const P = new Float64Array(M.predict(net, X).flat());
      const want = X.reduce((s, x, i) => s + M.backward(net, M.forward(net, x), Y[i], loss).loss, 0) / X.length;
      near(T.datasetLoss(P, Y, X.length, outs.length, loss, net.layers[last(net)].act), want, 1e-9, k);
    }
    // xent on a tanh output: both fall back to mse
    const net = M.PRESETS.linear.build(1);
    M.setLayer(net, net.layers[last(net)].id, { act: 'tanh' });
    const x = [[0.3, -0.2]], y = [[0.5, -0.5]];
    const P = new Float64Array(M.predict(net, x).flat());
    near(T.datasetLoss(P, y, 1, 2, 'xent', 'tanh'), M.backward(net, M.forward(net, x[0]), y[0], 'xent').loss, 1e-12);
  });

  test('wordAccuracy: the share of output tokens whose nearest word is the target word', () => {
    const net = M.PRESETS.agreement.build(1), ds = M.DATASETS.nl_agree, D = ds.make(30, 2, 0.1);
    const P = new Float64Array(M.predict(net, D.X).flat());
    let hit = 0;
    D.X.forEach((x, s) => ds.decode(M.predict(net, [x])[0]).forEach((w, t) => { if (w === D.targetWords[s][t]) hit++; }));
    near(T.wordAccuracy(P, D.targetWords, 30, 6, ds), hit / 90, 1e-12);
    // noise-free targets decode to their own words
    const D0 = ds.make(30, 2, 0);
    assert.equal(T.wordAccuracy(new Float64Array(D0.Y.flat()), D0.targetWords, 30, 6, ds), 1);
    const want = D.X.reduce((s, x, i) => s + M.backward(net, M.forward(net, x), D.Y[i], 'mse').loss, 0) / 30;
    near(T.datasetLoss(P, D.Y, 30, 6, 'mse', 'identity'), want, 1e-9, 'datasetLoss');
  });

  test('forwardMany (heatmaps, layer space) matches model.predict, skip edges included', () => {
    for (const k of ['residual', 'classifier', 'mlp']) {
      const net = M.PRESETS[k].build(4);
      const r = M.rng(9), n = 6, ins = M.nodesIn(net, 0).length;
      const X = Array.from({ length: n }, () => Array.from({ length: ins }, () => 2 * r() - 1));
      const acts = T.forwardMany(net, M, new Float64Array(X.flat()), n);
      const all = M.predict(net, X, { layer: 'all' });
      for (let l = 0; l < net.layers.length; l++) {
        const R = M.nodesIn(net, l).length;
        for (let s = 0; s < n; s++) for (let i = 0; i < R; i++) near(acts[l][s * R + i], all[s][l][i], 1e-12, `${k} a[${l}][${s}][${i}]`);
      }
    }
    // from a hidden layer: null when a skip edge bypasses it, exact otherwise
    const res = M.PRESETS.residual.build(1);
    assert.equal(T.forwardMany(res, M, new Float64Array([0.1, 0.2, 0.3]), 1, 1), null);
    const xor = M.PRESETS.xor.build(1);
    const h = M.predict(xor, [[0.4, -0.3]], { layer: 1 })[0];
    near(T.forwardMany(xor, M, new Float64Array(h), 1, 1)[2][0], M.predict(xor, [[0.4, -0.3]])[0][0], 1e-12);
  });
});

// ---------------------------------------------------------------- the tiny language model

describe('the tiny language model (nl_next, tiny_lm; docs/NN_FLOW.md)', () => {
  const ds = M.DATASETS.nl_next, VOCAB = ['.', 'dog', 'cat', 'dogs', 'cats', 'chases', 'chase'];
  const rows = x => [0, 1, 2].map(t => x.slice(t * 7, t * 7 + 7));
  const hot = row => {
    const k = row.indexOf(1);
    assert.ok(k >= 0 && row.every((v, i) => v === (i === k ? 1 : 0)), `one-hot: ${row}`);
    return VOCAB[k];
  };

  test('nl_next: a start token and two words in, the next word at each position out', () => {
    assert.deepEqual([ds.kind, ds.tokens, ds.inputs, ds.outputs], ['seq', 3, 21, 21]);
    assert.deepEqual(Object.keys(ds.vocab), VOCAB);
    assert.deepEqual(Object.values(ds.vocab), [0, 1, 2, 3, 4, 5, 6], 'each word\'s one-hot slot');
    assert.ok(VOCAB.slice(1).every(w => M.WORDS[w]), 'the words come from WORDS');
    const D = ds.make(200, 3, 0), seen = new Set();
    D.X.forEach((x, s) => {
      const w = rows(x).map(hot), y = rows(D.Y[s]).map(hot);
      assert.deepEqual(w, D.words[s]);
      assert.deepEqual(y, D.targetWords[s]);
      assert.equal(w[0], '.');
      assert.deepEqual(w.slice(1), y.slice(0, 2), 'each position\'s target is the next input');
      const [subj, verb, obj] = y;
      assert.ok(['dog', 'cat', 'dogs', 'cats'].includes(subj));
      assert.equal(verb, M.WORDS[subj][1] > 0 ? 'chases' : 'chase', 'the verb agrees with its subject');
      assert.notEqual(obj.replace(/s$/, ''), subj.replace(/s$/, ''), 'the object is the other animal');
      assert.notEqual(M.WORDS[obj][1], M.WORDS[subj][1], 'in the other number');
      assert.deepEqual(ds.decode(D.Y[s]), y, 'decode names the most likely word');
      seen.add(y.join(' '));
    });
    assert.deepEqual([...seen].sort(), ['cat chases dogs', 'cats chase dog', 'dog chases cats', 'dogs chase cat']);
    assert.deepEqual(ds.make(50, 3, 0), ds.make(50, 3, 0), 'deterministic');
    const noisy = ds.make(50, 3, 0.2), clean = ds.make(50, 3, 0);
    assert.deepEqual(noisy.words, clean.words, 'noise keeps the sentences');
    assert.deepEqual(noisy.Y, clean.Y, 'noise only on the inputs: the targets stay one-hot');
    assert.notDeepEqual(noisy.X, clean.X);
    assert.deepEqual(ds.decode([NaN, ...new Array(20).fill(0)]), [null, '.', '.']);
  });

  test('tiny_lm: embedding + position, 2 causal heads, W_O, a d -> 4d -> d FFN, a softmax per position', () => {
    const net = M.PRESETS.tiny_lm.build(1);
    assert.equal(net.meta.loss, 'xent');
    assert.deepEqual(net.meta.vocab, VOCAB);
    assert.equal(net.meta.flow, true);
    assert.deepEqual(net.meta.train, { init: { W_Q: 'small', W_K: 'small', W_O: 'small', W_2: 'small' }, dataset: 'nl_next', lr: 0.1, noise: 0 });
    assert.deepEqual(net.meta.tokenNames, ['.', 'dogs', 'chase'], 'the first sentence names the tokens');
    assert.deepEqual(net.layers.map((_, l) => { const s = M.tokenShape(net, l); return [s.tokens, s.d, s.groups]; }),
      [[3, 7, null], [3, 4, null], [3, 4, ['Q', 'K', 'V']], [3, 4, null], [3, 4, null], [3, 16, null], [3, 4, null], [3, 7, null]]);
    assert.deepEqual(net.layers.map(l => l.act), ['identity', 'identity', 'identity', 'identity', 'identity', 'relu', 'identity', 'softmax']);
    assert.deepEqual(M.attnSpec(net, 3), { l: 3, tokens: 3, d: 4, heads: 2, dh: 2, scale: 1 / Math.sqrt(2), causal: true });
    const mats = l => M.tiedMatrices(net, l).map(m => [m.name, m.W.length, m.W[0].length, m.k, m.tokenwise]);
    assert.deepEqual(mats(1), [['W_E', 7, 4, 0, true]]);
    assert.deepEqual(mats(2), [['W_Q', 4, 4, 1, true], ['W_K', 4, 4, 1, true], ['W_V', 4, 4, 1, true]]);
    assert.deepEqual(mats(4), [['W_O', 4, 4, 3, true]]);
    assert.deepEqual(mats(5), [['W_1', 4, 16, 4, true]]);
    assert.deepEqual(mats(6), [['W_2', 16, 4, 5, true]]);
    assert.deepEqual(mats(7), [['W_U', 4, 7, 6, true]]);
    // the two residuals: fixed identity edges X -> H and H -> Y, slot for slot
    const fixed = net.edges.filter(e => e.fixed);
    assert.equal(fixed.length, 24);
    for (const e of fixed) {
      const a = M.tokenPos(net, e.from), b = M.tokenPos(net, e.to);
      assert.equal(e.w, 1);
      assert.deepEqual([a.token, a.feature], [b.token, b.feature]);
      assert.ok((a.l === 1 && b.l === 4) || (a.l === 4 && b.l === 6), `${a.l} -> ${b.l}`);
    }
    // P: the embedding layer's biases are its own per position and feature; the others are shared per feature
    assert.ok(M.nodesIn(net, 1).every(n => !n.tie));
    for (const l of [2, 4, 5, 6, 7]) assert.ok(M.nodesIn(net, l).every(n => n.tie), `layer ${l} ties its biases`);
    // the Train panel's Reset (He, as the net has a ReLU, and meta.train.init) with init seed 1 gives the preset back
    const reset = M.clone(net);
    M.randomize(reset, { seed: 5 });
    M.randomize(reset, { seed: 1, scheme: 'he', init: reset.meta.train.init });
    assert.deepEqual(reset, net);
  });

  test('tiny_lm: every gradient against finite differences (cross-entropy over the words at each position)', () => {
    // P, every bias and every matrix non-zero (one draw per tie group, so the ties hold), and no
    // ReLU of the FFN on its kink, where a finite difference would be meaningless
    const net = M.PRESETS.tiny_lm.build(2);
    let seed = 5;
    for (;; seed++) {
      M.randomize(net, { seed, scheme: 'xavier', biases: 'small' });
      const z = M.forward(net).z[5];
      if (z.every(v => Math.abs(v) > 1e-3) && z.some(v => v > 0)) break;
    }
    const y = targets(net), { fwd, bwd } = gradCheckShared(net, y, 'xent');
    const p = fwd.a[7];
    const want = [0, 1, 2].reduce((s, t) => s - Math.log(p[t * 7 + y.slice(t * 7, t * 7 + 7).indexOf(1)]), 0) / 3;
    near(bwd.loss, want, 1e-12, 'the mean over the positions of -log p(next word)');
    bwd.dZ[7].forEach((dz, i) => near(dz, (p[i] - y[i]) / 3, 1e-12, `dL/dlogit ${i}`));
  });

  test('tiny_lm learns the grammar: the verb from the subject, the object through attention', () => {
    const net = M.PRESETS.tiny_lm.build(1), D = ds.make(200, 1, 0), r = M.rng(8);
    for (let s = 0; s < 800; s++) {
      const B = Array.from({ length: 10 }, () => Math.floor(r() * D.X.length));
      M.trainStep(net, { X: B.map(i => D.X[i]), Y: B.map(i => D.Y[i]) }, { lr: 0.1 });
    }
    for (const subj of ['dog', 'cat', 'dogs', 'cats']) {
      const i = D.words.findIndex(w => w[1] === subj), p = M.predict(net, [D.X[i]])[0], want = D.targetWords[i];
      assert.deepEqual(ds.decode(p).slice(1), want.slice(1), subj);
      assert.ok(p[7 + VOCAB.indexOf(want[1])] > 0.9, `${subj}: the verb`);
      assert.ok(p[14 + VOCAB.indexOf(want[2])] > 0.9, `${subj}: the object`);
      for (const k of [1, 2, 3, 4]) assert.ok(p[k] > 0.1 && p[k] < 0.5, `${subj}: the first word stays a guess among the nouns`);
    }
  });

  test('Adapt onto nl_next: a softmax per position and cross-entropy, as the preset has', async () => {
    const T = await import('../static/nn/train.js');
    const lm = M.PRESETS.tiny_lm.build(1), want = M.clone(lm);
    T.adaptNet(lm, M, ds, { seed: 1 });
    assert.deepEqual(lm, want, 'the preset already fits: Adapt keeps it as it is');
    T.adaptNet(lm, M, M.DATASETS.nl_agree, { seed: 1 });
    assert.deepEqual([lm.layers[7].act, lm.meta.loss], ['identity', 'mse']);
    for (const net of [lm, M.PRESETS.xor.build(1), M.PRESETS.causal.build(1)]) {
      assert.match(T.adaptNet(net, M, ds, { seed: 1 }), /3×7 tokens: softmax output, cross-entropy loss/);
      const out = net.layers[net.layers.length - 1];
      assert.deepEqual([out.act, out.tokens, net.meta.loss], ['softmax', 3, 'xent']);
      const p = M.predict(net, ds.make(1, 1, 0).X)[0];
      for (let t = 0; t < 3; t++) near(p.slice(t * 7, t * 7 + 7).reduce((s, v) => s + v, 0), 1, 1e-12, `position ${t}`);
    }
  });
});
