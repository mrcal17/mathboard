// Tests for the pure helpers of static/nn/surf3d.js (the 3D plots panel, docs/NN_3D_PLOTS.md).
// Run: node --test tests/surf3d.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../static/nn/model.js';
import { datasetLoss } from '../static/nn/train.js';
import {
  paramList, readParams, writeParams, filterNormalize, randomDirections, project, symEig, pathDirections,
  isoLines, niceTicks, niceUp, plainLabel, forwardZA, makeEvaluator,
} from '../static/nn/surf3d.js';

const build = key => M.PRESETS[key].build(1);
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(b)), `${msg}: ${a} vs ${b}`);
// The dataset of a preset, flattened as the panel reads it.
function data(net) {
  const ds = M.DATASETS[net.meta.train.dataset];
  let { X, Y } = ds.make(200, 1, 0.1);
  const flat = r => (r.some(Array.isArray) ? r.flat(Infinity) : r);
  X = X.map(flat); Y = Y.map(flat);
  const Xf = new Float64Array(X.length * ds.inputs);
  X.forEach((x, s) => x.forEach((v, j) => { Xf[s * ds.inputs + j] = v; }));
  return { X, Y, Xf, n: X.length, dim: ds.inputs, K: ds.outputs };
}
// The dataset loss through model.predict: what the landscape's centre must equal.
function lossByPredict(net, d) {
  const P = M.predict(net, d.X), K = d.K, flat = new Float64Array(d.n * K);
  P.forEach((p, s) => p.forEach((v, k) => { flat[s * K + k] = v; }));
  const last = net.layers[net.layers.length - 1], L = net.layers.length;
  const sh = M.tokenShape(net, L - 1), seg = sh.tokens * (sh.groups?.length || 1);
  return datasetLoss(flat, d.Y, d.n, K, net.meta.loss || 'mse', last.act, seg > 1 && K % seg === 0 ? seg : 1);
}

describe('parameters', () => {
  for (const key of ['xor', 'conv1d', 'lenet', 'transformer', 'avgpool']) {
    test(`${key}: the list holds exactly what trainStep moves`, () => {
      const net = build(key), params = paramList(net, M);
      const before = M.clone(net), d = key === 'transformer' ? data(net) : null;
      const X = d ? d.X.slice(0, 8) : [M.nodesIn(net, 0).map((_, j) => 0.3 + 0.1 * j)];
      const Y = d ? d.Y.slice(0, 8) : [M.nodesIn(net, net.layers.length - 1).map(() => 0.7)];
      M.trainStep(net, { X, Y }, { lr: 0.5 });
      const moved = new Set();
      net.edges.forEach((e, k) => { if (e.w !== before.edges[k].w) moved.add('w' + e.id); });
      net.nodes.forEach((q, k) => { if (q.bias !== before.nodes[k].bias) moved.add('b' + q.id); });
      const listed = new Set(params.flatMap(p => p.ids.map(id => p.kind + id)));
      for (const m of moved) assert.ok(listed.has(m), `${m} moved but is not a parameter`);
      for (const e of net.edges) if (e.fixed) assert.ok(!listed.has('w' + e.id), 'a fixed edge is not a parameter');
      // one entry per tie group, and writing a vector sets every member
      const ties = new Set(net.edges.filter(e => typeof e.tie === 'string' && e.tie && !e.fixed).map(e => e.tie));
      for (const t of ties) assert.equal(params.filter(p => p.tie === t && p.kind === 'w').length, 1, `tie ${t}`);
      const v = Float64Array.from(params, (_, k) => k + 0.5);
      writeParams(net, params, v);
      assert.deepEqual(Array.from(readParams(net, params)), Array.from(v));
      for (const p of params) for (const id of p.ids) {
        const o = p.kind === 'w' ? M.edge(net, id).w : M.node(net, id).bias;
        assert.equal(o, v[params.indexOf(p)]);
      }
    });
  }
});

describe('the loss landscape', () => {
  for (const key of ['xor', 'classifier', 'transformer']) {
    test(`${key}: the evaluator's centre is the dataset loss, and a step along one weight matches a changed copy`, () => {
      const net = build(key), d = data(net), ev = makeEvaluator(net, M, d);
      const th = readParams(net, ev.params);
      near(ev.evalAt(th), lossByPredict(net, d), 1e-12, 'centre');
      const k = 1, v = Float64Array.from(th);
      v[k] += 0.37;
      const copy = M.clone(net);
      writeParams(copy, ev.params, v);
      near(ev.evalAt(v), lossByPredict(copy, d), 1e-12, 'moved');
      near(ev.evalAt(th), lossByPredict(net, d), 1e-12, 'centre again (the patched matrices are restored by the vector)');
    });
  }
  test('filter normalisation: each neuron\'s slice of the direction has the norm of its weights', () => {
    const net = build('mlp'), params = paramList(net, M), th = readParams(net, params);
    const dir = Float64Array.from(params, (_, k) => Math.sin(k * 1.7) + 0.2);
    const out = filterNormalize(dir, th, params);
    const byF = new Map();
    params.forEach((p, k) => { const g = byF.get(p.filter) || []; g.push(k); byF.set(p.filter, g); });
    for (const ks of byF.values()) {
      const n = v => Math.sqrt(ks.reduce((s, k) => s + v[k] * v[k], 0));
      near(n(out), n(th), 1e-12, 'filter norm');
    }
  });
  test('W = 0 presets still get a plane (the zero filters get the RMS norm, or 1)', () => {
    const net = build('logreg'), params = paramList(net, M), th = readParams(net, params);
    assert.ok(th.every(v => v === 0));
    const [d1, d2] = randomDirections(params, th, M.rng(1));
    assert.ok(Math.sqrt(d1.reduce((s, v) => s + v * v, 0)) > 0.5);
    assert.ok(Math.abs(project(d2, d1, d2).b - 1) < 1e-12);
  });
  test('projection recovers the coordinates and the distance from the plane', () => {
    const d1 = [1, 2, 0, 1], d2 = [0, 1, 1, -1], extra = [-1, 0, 1, 1].map(v => v / Math.sqrt(3));
    // extra is orthogonal to d1 and d2
    assert.ok(Math.abs(extra.reduce((s, v, k) => s + v * d1[k], 0)) < 1e-12 && Math.abs(extra.reduce((s, v, k) => s + v * d2[k], 0)) < 1e-12);
    const delta = d1.map((v, k) => 0.7 * v - 1.3 * d2[k] + 0.4 * extra[k]);
    const r = project(delta, d1, d2);
    near(r.a, 0.7, 1e-12, 'a'); near(r.b, -1.3, 1e-12, 'b'); near(r.r, 0.4, 1e-9, 'r');
  });
  test('the principal directions of a path in a plane span that plane', () => {
    const u = [1, 0, 1, 0, 0].map(v => v / Math.SQRT2), w = [0, 1, 0, 0, 1].map(v => v / Math.SQRT2);
    const rows = Array.from({ length: 30 }, (_, t) => Float64Array.from(u, (v, k) => 2 * Math.cos(t / 5) * v + 0.5 * Math.sin(t / 3) * w[k] + 1));
    const r = pathDirections(rows);
    near(r.share, 1, 1e-9, 'share');
    for (const d of [r.d1, r.d2]) {
      const p = project(d, u, w);
      near(p.r, 0, 1e-6, 'in the plane');
    }
  });
});

describe('numerics', () => {
  test('symEig', () => {
    const A = [4, 1, 2, 1, 3, 0, 2, 0, 5];
    const { values, vectors } = symEig(A, 3);
    assert.ok(values[0] >= values[1] && values[1] >= values[2]);
    for (let c = 0; c < 3; c++) {
      const v = [0, 1, 2].map(k => vectors[k * 3 + c]);
      for (let i = 0; i < 3; i++) near([0, 1, 2].reduce((s, j) => s + A[i * 3 + j] * v[j], 0), values[c] * v[i], 1e-10, 'Av = λv');
    }
  });
  test('isoLines: a plane crosses its level on one straight line', () => {
    const G = 9, vals = new Float64Array(G * G);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) vals[j * G + i] = i + 0.5 * j;
    const seg = isoLines(vals, G, 4.25);
    assert.ok(seg.length > 0);
    for (let k = 0; k < seg.length; k += 2) near(seg[k] + 0.5 * seg[k + 1], 4.25, 1e-12, 'on the level');
  });
  test('ticks and labels', () => {
    assert.deepEqual(niceTicks(-1, 1, 4), [-1, -0.5, 0, 0.5, 1]);
    assert.equal(niceUp(1.37), 1.5);
    assert.equal(niceUp(0.0071), 0.008);
    assert.equal(plainLabel('h^{(1)}_{2}'), 'h⁽¹⁾₂');
    assert.equal(plainLabel('\\hat y_{1}'), 'ŷ₁');
    assert.equal(plainLabel('x_{1}'), 'x₁');
  });
  test('forwardZA agrees with model.forward, z and a', () => {
    for (const key of ['deep', 'residual', 'classifier', 'transformer']) {
      const net = build(key), fwd = M.forward(net);
      const x = Float64Array.from(M.nodesIn(net, 0), q => q.value);
      const F = forwardZA(net, M, x, 1);
      for (let l = 1; l < net.layers.length; l++) {
        M.nodesIn(net, l).forEach((q, i) => {
          near(F.a[l][i], fwd.node[q.id].a, 1e-12, `${key} a ${l}.${i}`);
          if (F.hasZ) near(F.z[l][i], fwd.node[q.id].z, 1e-12, `${key} z ${l}.${i}`);
        });
      }
    }
  });
});
