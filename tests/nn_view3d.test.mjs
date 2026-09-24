// Tests for the pure part of static/nn/view3d.js (the 3D net view, docs/NN_3D.md): the multi-head
// reshape and its bug, per-head attention against model.forward, and the v3d state cleaning.
// Run: node --test tests/nn_view3d.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../static/nn/model.js';
import {
  MODES, EXAMPLE, cleanV3d, divisors, slotOf, toHeads, fromHeads, attend, tensorSteps, tensorStory,
  exampleQKV, projOf, attnLayers,
} from '../static/nn/view3d.js';

const near = (a, b, eps = 1e-12) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);
const grid = (T, d) => Array.from({ length: T }, (_, t) => Array.from({ length: d }, (_, f) => 10 * t + f));

describe('cleanV3d', () => {
  test('null stays null, junk gets defaults', () => {
    assert.equal(cleanV3d(null), null);
    assert.equal(cleanV3d('stack'), null);
    const v = cleanV3d({ mode: 'nope', camera: { p: [1, 2], t: [0, 0, 0] }, step: -3, h: 5, bug: 'yes', color: 'x' });
    assert.deepEqual(v, { mode: 'stack', camera: null, layer: null, step: 0, src: null, h: 3, bug: false, color: 'value', nums: true });
  });
  test('good fields are kept', () => {
    const v = cleanV3d({ mode: 'tensor', camera: { p: [1, 2, 3], t: [0, 1, 0] }, layer: 'L2', step: 4, src: 'example', h: 2, bug: true, color: 'token', nums: false });
    assert.deepEqual(v, { mode: 'tensor', camera: { p: [1, 2, 3], t: [0, 1, 0] }, layer: 'L2', step: 4, src: 'example', h: 2, bug: true, color: 'token', nums: false });
    for (const m of MODES) assert.equal(cleanV3d({ mode: m }).mode, m);
  });
  test('the example heads divide d', () => {
    assert.deepEqual(divisors(EXAMPLE.d), [1, 2, 3, 6]);
    assert.deepEqual(divisors(12), [1, 2, 3, 4, 6, 12]);
  });
});

describe('the reshape', () => {
  const T = 4, d = 6, h = 3, dh = 2, X = grid(T, d);
  test('view(T, h, d/h) then transpose(0, 1): head k holds chunk k of every token', () => {
    const H = toHeads(X, 'HTD', h);
    for (let k = 0; k < h; k++) for (let t = 0; t < T; t++) {
      assert.deepEqual(H[k][t], X[t].slice(k * dh, (k + 1) * dh), `head ${k}, row ${t}`);
    }
    assert.deepEqual(slotOf('THD', 2, 5, { T, d, h }), { head: 2, row: 2, col: 1 });
    assert.deepEqual(slotOf('TD', 2, 5, { T, d, h }), { head: 2, row: 2, col: 5 });
  });
  test('the bug, view(h, T, d/h): head k is a run of T·d/h numbers in memory order', () => {
    const H = toHeads(X, 'BUG', h), flat = X.flat();
    for (let k = 0; k < h; k++) assert.deepEqual(H[k].flat(), flat.slice(k * T * dh, (k + 1) * T * dh));
    // so head 1's rows come from tokens 1 and 2 only (0-based 0 and 1)
    assert.deepEqual([...new Set(H[0].flat().map(v => Math.floor(v / 10)))], [0, 1]);
  });
  test('fromHeads undoes both', () => {
    for (const form of ['HTD', 'BUG']) assert.deepEqual(fromHeads(toHeads(X, form, h), form, T, d), X);
  });
  test('the steps', () => {
    assert.deepEqual(tensorSteps().map(s => s.key), ['td', 'split', 'heads', 'attend', 'back', 'merge']);
    assert.deepEqual(tensorSteps({ proj: true }).map(s => s.form), ['TD', 'THD', 'HTD', 'HTD', 'THD', 'TD', 'TD']);
    assert.deepEqual(tensorSteps({ bug: true, proj: true }).map(s => s.key), ['td', 'view', 'attend', 'bugback', 'proj']);
  });
});

describe('attention per head', () => {
  test('rows of A sum to 1; the causal mask zeroes j > i', () => {
    const { Q, K, V } = exampleQKV();
    const [Qh, Kh, Vh] = [Q, K, V].map(X => toHeads(X, 'HTD', 2));
    for (const causal of [false, true]) {
      for (const { A } of attend(Qh, Kh, Vh, { scale: 0.5, causal })) {
        A.forEach((row, i) => {
          near(row.reduce((s, x) => s + x, 0), 1, 1e-9);
          if (causal) row.forEach((x, j) => { if (j > i) assert.equal(x, 0); });
        });
      }
    }
  });
  for (const key of ['multihead', 'transformer', 'attention', 'causal', 'words']) {
    test(`${key}: the correct story reproduces the attention layer's Z, the bug does not`, () => {
      const net = M.PRESETS[key].build(1), fwd = M.forward(net);
      const [l] = attnLayers(net), spec = M.attnSpec(net, l);
      const { Q, K, V } = M.reshape(net, l - 1, fwd.a[l - 1]), Z = M.reshape(net, l, fwd.a[l]).X;
      const ok = tensorStory({ Q, K, V, h: spec.heads, scale: spec.scale, causal: spec.causal });
      ok.Z.forEach((row, t) => row.forEach((z, f) => near(z, Z[t][f], 1e-9)));
      ok.heads.forEach((hd, k) => hd.A.forEach((row, i) => row.forEach((a, j) => near(a, fwd.attn[l].heads[k].A[i][j], 1e-9))));
      if (spec.heads > 1) {
        const bug = tensorStory({ Q, K, V, h: spec.heads, scale: spec.scale, causal: spec.causal, bug: true });
        const diff = Math.max(...bug.Z.flat().map((z, k) => Math.abs(z - Z.flat()[k])));
        assert.ok(diff > 1e-6, 'the bug changes Z');
      }
    });
  }
  test('the example: out = Z W_O, and the bug differs from the right answer', () => {
    const ex = exampleQKV();
    const ok = tensorStory({ ...ex, h: 3, scale: 1 / Math.sqrt(2) }), bad = tensorStory({ ...ex, h: 3, scale: 1 / Math.sqrt(2), bug: true });
    assert.equal(ok.out.length, EXAMPLE.T);
    near(ok.out[1][2], ok.Z[1].reduce((s, z, i) => s + z * ex.WO[i][2], 0));
    assert.ok(Math.max(...ok.Z.flat().map((z, k) => Math.abs(z - bad.Z.flat()[k]))) > 0.01);
    // h = 1: the view and the transpose are the same thing, so there is no bug to see
    const one = tensorStory({ ...ex, h: 1 }), oneBug = tensorStory({ ...ex, h: 1, bug: true });
    one.Z.flat().forEach((z, k) => near(z, oneBug.Z.flat()[k]));
  });
});

describe('net helpers', () => {
  test('projOf finds the transformer\'s W_O and nothing on nets without one', () => {
    const tr = M.PRESETS.transformer.build(1), [l] = attnLayers(tr);
    const p = projOf(tr, l);
    assert.equal(p.l, l + 1);
    assert.equal(p.W.length, 2);
    assert.equal(projOf(M.PRESETS.multihead.build(1), attnLayers(M.PRESETS.multihead.build(1))[0]), null);
    assert.deepEqual(attnLayers(M.PRESETS.mlp.build(1)), []);
  });
});
