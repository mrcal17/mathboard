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
  const GROUPS = ['Basics', 'MLPs', 'Skip connections', 'Structure in W', 'Sequences', 'Embeddings & autoencoders', 'Teaching demos'];
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
      assert.ok(net.nodes.length <= 40, `${net.nodes.length} nodes`);
      assert.ok(net.layers.every(l => l.name.trim()), 'layer names');
      assert.ok(net.nodes.every(n => n.label.trim()), 'labels');
      for (let i = 0; i < net.nodes.length; i++) {
        for (let j = i + 1; j < net.nodes.length; j++) {
          const a = net.nodes[i], b = net.nodes[j];
          assert.ok(Math.hypot(a.x - b.x, a.y - b.y) >= 2 * R + 24, `${a.id} and ${b.id} too close (value labels overlap)`);
        }
      }
      const colX = net.layers.map((_, l) => {
        const xs = M.nodesIn(net, l).map(n => n.x);
        assert.ok(xs.every(x => x === xs[0]), `layer ${l} is one column`);
        return xs[0];
      });
      colX.forEach((x, l) => assert.ok(l === 0 || x > colX[l - 1] + 2 * R, `column ${l} is right of column ${l - 1}`));

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

  test('rnn: every step starts with the same W_hh, x weights and bias; x_t only feeds step t', () => {
    const net = M.PRESETS.rnn.build(), m = M.matrices(net), T = 4;
    const Whh = term(net, 2, 1).W, wx = term(net, 1, 0).W.map(row => row[0]);
    for (let t = 1; t <= T; t++) {
      const x = term(net, t, 0);
      x.edge.forEach(row => row.forEach((e, j) => assert.equal(e !== null, j === t - 1, `step ${t}, x_${j + 1}`)));
      assert.deepEqual(x.W.map(row => row[t - 1]), wx);
      if (t > 1) assert.deepEqual(term(net, t, t - 1).W, Whh);
      assert.deepEqual(m[t - 1].b, m[0].b);
    }
    assert.deepEqual(m[T].terms.map(t => t.k), [T], 'the output reads the last step only');
    const trained = M.clone(net);   // no weight tying: a training step moves the copies apart
    M.trainStep(trained, { X: [[0.5, -0.3, 0.9, 0.2]], Y: [[1]] }, { lr: 0.5 });
    assert.notDeepEqual(term(trained, 2, 1).W, term(trained, 3, 2).W);
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
    assert.ok(g[1] < 1e-3 * g[last(net)]);
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

  test('matrix form reproduces forward: z = b + sum_k W a', () => {
    for (const k of Object.keys(M.PRESETS)) {
      const net = M.PRESETS[k].build(2);
      const f = M.forward(net);
      for (const m of M.matrices(net)) {
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
