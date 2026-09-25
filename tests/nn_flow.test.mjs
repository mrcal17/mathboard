// Tests for the pure part of static/nn/flow.js (the Flow view, docs/NN_FLOW.md): the per-head
// attention and the forward pass it recomputes against model.forward, knocking a head out, and
// the stages and tiles of a flow. Run: node --test tests/nn_flow.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../static/nn/model.js';
import { cleanFlow, attendHeads, propagate, ablate, buildFlow, flowKey, texHtml } from '../static/nn/flow.js';

const near = (a, b, eps = 1e-12, msg = '') => assert.ok(Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b)), `${msg} ${a} vs ${b}`);
const nearV = (u, v, eps, msg = '') => { assert.equal(u.length, v.length, `${msg} length`); u.forEach((x, i) => near(x, v[i], eps, `${msg}[${i}]`)); };
const lm = (seed = 1) => M.PRESETS.tiny_lm.build(seed);

// A few steps of training, so the numbers are not the tiny ones of a fresh init.
function trained(net, steps = 300) {
  const d = M.DATASETS[net.meta.train.dataset].make(60, 1, 0), r = M.rng(4);
  for (let s = 0; s < steps; s++) {
    const B = Array.from({ length: 10 }, () => Math.floor(r() * d.X.length));
    M.trainStep(net, { X: B.map(i => d.X[i]), Y: B.map(i => d.Y[i]) }, { lr: 0.1 });
  }
  return net;
}

describe('cleanFlow', () => {
  test('null and junk', () => {
    assert.equal(cleanFlow(null), null);
    assert.equal(cleanFlow('on'), null);
    assert.deepEqual(cleanFlow({}), { stage: null, play: false, off: [], nums: true, every: false, hover: null });
    assert.deepEqual(cleanFlow({ stage: -1, play: true, off: [1, 'x', 1, -2, 0.5], nums: 0, every: 1, hover: { t: 3, i: 0, j: 0 } }),
      { stage: null, play: false, off: [1], nums: true, every: false, hover: null });
  });
  test('good fields are kept; play needs a stage', () => {
    assert.deepEqual(cleanFlow({ stage: 4, play: true, off: [2, 0], nums: false, every: true, hover: { t: 'a3.0', i: 2, j: 1, x: 9 } }),
      { stage: 4, play: true, off: [0, 2], nums: false, every: true, hover: { t: 'a3.0', i: 2, j: 1 } });
    assert.equal(cleanFlow({ stage: null, play: true }).play, false);
  });
});

describe('the recomputed forward pass', () => {
  test('attendHeads gives model.forward\'s S, A, Z per head, exactly', () => {
    for (const k of ['tiny_lm', 'multihead', 'causal', 'transformer']) {
      const net = trained(M.PRESETS[k].build(2), 50), f = M.forward(net);
      f.attn.forEach((at, l) => {
        if (!at) return;
        const g = M.attnSpec(net, l), r = attendHeads(f.a[l - 1], g);
        assert.deepEqual(r.out, f.a[l], `${k}: Z`);
        r.heads.forEach((h, i) => {
          for (const key of ['Q', 'K', 'V', 'S', 'A', 'Z']) assert.deepEqual(h[key], at.heads[i][key], `${k}: head ${i} ${key}`);
        });
      });
    }
  });

  test('propagate from any layer reproduces model.forward', () => {
    for (const k of ['tiny_lm', 'transformer', 'multihead', 'causal', 'mlp', 'residual', 'classifier', 'densenet']) {
      const net = M.PRESETS[k].build(3), f = M.forward(net);
      for (let from = 1; from < net.layers.length; from++) {
        const r = propagate(net, f.a, from);
        f.a.forEach((a, l) => nearV(r.a[l], a, 1e-12, `${k} from ${from}: a[${l}]`));
        for (let l = from; l < net.layers.length; l++) nearV(r.z[l], f.z[l], 1e-12, `${k} from ${from}: z[${l}]`);
      }
    }
  });

  test('ablate with no head off is the forward pass itself', () => {
    const net = lm(), f = M.forward(net);
    const r = ablate(net, f, []);
    assert.equal(r.from, -1);
    assert.equal(r.a, f.a);
    assert.equal(ablate(net, f, [5]).from, -1, 'a head the net does not have changes nothing');
  });

  test('knocking head h out = zeroing the rows of W_O that read its columns of Z', () => {
    const net = trained(lm()), f = M.forward(net), g = M.attnSpec(net, 3);
    for (let h = 0; h < g.heads; h++) {
      const r = ablate(net, f, [h]);
      assert.equal(r.from, 3);
      const cut = M.clone(net), zIds = M.nodesIn(cut, 3).map(n => n.id);
      for (const e of cut.edges) {
        const k = zIds.indexOf(e.from);
        if (k >= 0 && Math.floor((k % g.d) / g.dh) === h) e.w = 0;   // an edge out of head h's columns of Z
      }
      const want = M.forward(cut);
      for (let l = 4; l < net.layers.length; l++) nearV(r.a[l], want.a[l], 1e-12, `head ${h} off: a[${l}]`);
      for (let t = 0; t < g.tokens; t++) for (let c = 0; c < g.d; c++) {
        const v = r.a[3][t * g.d + c];
        if (Math.floor(c / g.dh) === h) assert.equal(v, 0, 'its concat columns are 0');
        else assert.equal(v, f.a[3][t * g.d + c], 'the other head is untouched');
      }
      for (let l = 0; l < 3; l++) assert.deepEqual(r.a[l], f.a[l], 'layers before the attention are the forward pass');
      assert.deepEqual(r.attn[3].heads[h].Z, f.attn[3].heads[h].Z, 'the knocked-out head still reports its own Z');
    }
  });

  test('knocking every head out of the multihead preset zeros its output', () => {
    const net = M.PRESETS.multihead.build(1), f = M.forward(net);
    assert.ok(ablate(net, f, [0, 1]).a[2].every(v => v === 0));
  });
});

describe('buildFlow', () => {
  test('the tiny language model: every stage, left to right', () => {
    const F = buildFlow(lm());
    assert.equal(F.why, null);
    assert.ok(F.attention);
    assert.equal(F.heads, 2);
    assert.deepEqual(F.stages.map(s => s.key), ['in', 'embed1', 'qkv2', 'scores3', 'softmax3', 'mix3', 'concat3', 'proj4', 'sum4',
      'layer5', 'proj6', 'sum6', 'logits7', 'probs7']);
    assert.deepEqual(F.stages.map(s => s.part), [null, null, null, 'scores', 'softmax', 'mix', null, null, null, null, null, null, null, null]);
    const net = lm();
    assert.deepEqual(F.stages.map(s => s.layer), [0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7].map(l => net.layers[l].id));
    const shape = id => [F.tiles[id].rows, F.tiles[id].cols];
    assert.deepEqual(shape('in'), [5, 23]);
    assert.ok(F.tiles.in.onehot);
    assert.deepEqual(F.tiles.in.colLab, net.meta.vocab);
    for (const id of ['P1.0', 'B1', 'L1', 'L3', 'P4.0', 'L4', 'P6.0', 'L6']) assert.deepEqual(shape(id), [5, 8], id);
    for (const id of ['q2.0', 'k2.0', 'v2.0', 'q2.1', 'k2.1', 'v2.1', 'z3.0', 'z3.1']) assert.deepEqual(shape(id), [5, 4], id);
    for (const id of ['s3.0', 'a3.0', 's3.1', 'a3.1']) assert.deepEqual(shape(id), [5, 5], id);
    assert.deepEqual(shape('L5'), [5, 32]);
    // the ending: only the last position's logits and next-word distribution
    assert.ok(F.lm && !F.every);
    assert.deepEqual(shape('G7'), [1, 23]);
    assert.deepEqual(shape('L7'), [1, 23]);
    assert.equal(F.tiles.L7.kind, 'dist');
    assert.deepEqual([F.tiles.G7.pos, F.tiles.L7.pos], [[4], [4]]);
    assert.deepEqual(F.tiles['s3.0'].mask, [0, 1, 2, 3, 4].map(i => [0, 1, 2, 3, 4].map(j => j > i)), 'causal');
    assert.deepEqual(F.words, ['the', 'cat', 'sat', 'on', 'the'], 'the headline sentence');
  });

  test('tiles hold the forward pass, and each sum adds up', () => {
    const net = trained(lm()), f = M.forward(net), F = buildFlow(net, { fwd: f, every: true });
    const R = l => M.reshape(net, l, f.a[l]);
    const T = id => F.tiles[id].v;
    assert.deepEqual(T('L1'), R(1).X);
    assert.deepEqual(T('L3'), R(3).X);
    assert.deepEqual(T('L4'), R(4).X);
    assert.deepEqual(T('L5'), R(5).X);
    assert.deepEqual(T('L6'), R(6).X);
    assert.deepEqual(T('L7'), R(7).X);
    assert.deepEqual(T('G7'), M.reshape(net, 7, f.z[7]).X, 'the logits are z of the output');
    for (let h = 0; h < 2; h++) {
      assert.deepEqual(T(`s3.${h}`), f.attn[3].heads[h].S);
      assert.deepEqual(T(`a3.${h}`), f.attn[3].heads[h].A);
      assert.deepEqual(T(`z3.${h}`), f.attn[3].heads[h].Z);
      assert.deepEqual(T(`q2.${h}`), f.attn[3].heads[h].Q);
    }
    const bias = (l, j) => M.nodesIn(net, l)[j].bias;
    for (let t = 0; t < 5; t++) for (let j = 0; j < 8; j++) {
      near(T('P1.0')[t][j] + T('B1')[t][j], T('L1')[t][j], 1e-12, 'X = O W_E + P');
      near(T('L1')[t][j] + T('P4.0')[t][j] + bias(4, j), T('L4')[t][j], 1e-12, 'H = X + Z W_O + b_O');
      near(T('L4')[t][j] + T('P6.0')[t][j] + bias(6, j), T('L6')[t][j], 1e-12, 'Y = H + F W_2 + b_2');
    }
    // the embedding product of a one-hot row is the word's row of W_E
    const WE = M.tiedMatrices(net, 1).find(m => m.name === 'W_E').W, vocab = net.meta.vocab;
    F.words.forEach((w, t) => assert.deepEqual(T('P1.0')[t], WE[vocab.indexOf(w)]));
    // the prediction: the last position's most likely word
    const p = T('L7')[4], k = p.indexOf(Math.max(...p));
    assert.deepEqual(F.next, { word: vocab[k], p: p[k], target: 'mat', t: 4 });
    // the default ending holds the same numbers, for the last position alone
    const L = buildFlow(net, { fwd: f });
    assert.deepEqual(L.tiles.L7.v, [T('L7')[4]]);
    assert.deepEqual(L.tiles.G7.v, [T('G7')[4]]);
    assert.deepEqual(L.next, F.next);
  });

  test('every neuron is drawn once, where its value is; every source is a cell that exists', () => {
    for (const k of Object.keys(M.PRESETS)) {
      const net = M.PRESETS[k].build(1), f = M.forward(net), F = buildFlow(net, { fwd: f, every: true });
      for (const n of net.nodes) {
        const c = F.nodeCell[n.id];
        assert.ok(c, `${k}: ${n.id} is drawn`);
        const [t, i, j] = c;
        near(F.tiles[t].v[i][j], f.node[n.id].a, 1e-12, `${k}: ${n.id}`);
      }
      const shown = new Set(F.stages.flatMap(s => s.rows.flatMap(r => r.tiles)));
      assert.deepEqual([...shown].sort(), Object.keys(F.tiles).sort(), `${k}: every tile is in a stage, once`);
      for (const t of Object.values(F.tiles)) {
        for (let i = 0; i < t.rows; i++) for (let j = 0; j < t.cols; j++) {
          for (const [s, a, b] of t.src(i, j)) assert.ok(F.tiles[s] && a < F.tiles[s].rows && b < F.tiles[s].cols, `${k}: ${t.id} -> ${s}`);
          assert.equal(typeof t.tip(i, j), 'string');
        }
      }
    }
  });

  test('sources trace one step back', () => {
    const F = buildFlow(lm()), src = (id, i, j) => F.tiles[id].src(i, j).map(c => c.join(':')).sort();
    // a score: the query row and the key row of its head
    const row = (id, i, n) => Array.from({ length: n }, (_, j) => `${id}:${i}:${j}`);
    assert.deepEqual(src('s3.1', 2, 1), [...row('k2.1', 1, 4), ...row('q2.1', 2, 4)]);
    assert.deepEqual(src('s3.0', 0, 2), [], 'a masked score has no sources');
    // a weight: the scores it may see; a mix: its row of A and the value column
    assert.deepEqual(src('a3.0', 1, 0), ['s3.0:1:0', 's3.0:1:1']);
    assert.deepEqual(src('z3.0', 1, 1), ['a3.0:1:0', 'a3.0:1:1', 'v2.0:0:1', 'v2.0:1:1']);
    // concat: the head's cell; the residual sum: its three parts
    assert.deepEqual(src('L3', 2, 5), ['z3.1:2:1']);
    assert.deepEqual(src('L4', 0, 2), ['L1:0:2', 'P4.0:0:2']);
    assert.deepEqual(src('L1', 1, 3), ['B1:1:3', 'P1.0:1:3']);
    // the FFN and the logits read the whole row of the layer before; the last position's logits its row of Y
    assert.deepEqual(src('L5', 2, 7), row('L4', 2, 8));
    assert.deepEqual(src('G7', 0, 3), row('L6', 4, 8));
    assert.deepEqual(src('L7', 0, 4), row('G7', 0, 23).sort());
  });

  test('a head knocked out: its tiles are marked, its concat columns are 0, the rest follows', () => {
    const net = trained(lm()), F0 = buildFlow(net), F = buildFlow(net, { off: [0] });
    assert.ok(F.tiles['z3.0'].off && F.tiles['s3.0'].off && !F.tiles['z3.1'].off);
    assert.deepEqual(F.tiles.L3.offCols, [true, true, true, true, false, false, false, false]);
    assert.ok(F.tiles.L3.v.every(row => row[0] === 0 && row[1] === 0));
    assert.notDeepEqual(F.tiles.L7.v, F0.tiles.L7.v);
    assert.notEqual(flowKey(F), flowKey(F0), 'the view rebuilds');
    assert.match(F.stages.find(s => s.key === 'concat3').text, /Heads 1 knocked out/);
  });

  test('the ending: only the last position by default, every position when asked', () => {
    const net = trained(lm()), f = M.forward(net), F = buildFlow(net, { fwd: f }), E = buildFlow(net, { fwd: f, every: true });
    const nodes = M.nodesIn(net, 7), V = 23;
    // by default only position 5's outputs are drawn, in the chart; every position draws them all
    nodes.forEach((n, k) => {
      if (Math.floor(k / V) === 4) assert.deepEqual(F.nodeCell[n.id], ['L7', 0, k % V]);
      else assert.equal(F.nodeCell[n.id], undefined, 'an earlier position is not drawn');
      assert.deepEqual(E.nodeCell[n.id], ['L7', Math.floor(k / V), k % V]);
    });
    // only Y's last row goes on: the other rows fade
    assert.deepEqual(F.tiles.L6.dimRows, [0, 1, 2, 3]);
    assert.equal(E.tiles.L6.dimRows, null);
    // the chart: its context, the true next word and no note; every position: a row each, with the note
    assert.deepEqual(F.tiles.L7.rowLab, ['the cat sat on the']);
    assert.deepEqual(F.tiles.L7.target, [net.meta.vocab.indexOf('mat')]);
    assert.equal(F.tiles.L7.note, null);
    assert.deepEqual(E.tiles.L7.rowLab, ['the', 'the cat', 'the cat sat', 'the cat sat on', 'the cat sat on the']);
    assert.deepEqual(E.tiles.L7.target, ['cat', 'sat', 'on', 'the', 'mat'].map(w => net.meta.vocab.indexOf(w)));
    assert.match(E.tiles.L7.note, /Training scores every position .* generating reads only the last/);
    const probs = X => X.stages.find(s => s.key === 'probs7');
    assert.deepEqual([probs(F).title, probs(F).mode, probs(E).title], ['Next word', true, 'Next word, every position']);
    assert.equal(F.stages.find(s => s.key === 'logits7').tex, '\\ell_{5} = y_{5}\\,W_U + b_U');
    // the tip names the word, its context and the true next word
    const tip = F.tiles.L7.tip(0, net.meta.vocab.indexOf('mat'));
    assert.match(tip, /p\(mat \| the cat sat on the\)/);
    assert.match(tip, /the true next word/);
  });

  test('the order is on every matrix: rows named by position and word, keys by position', () => {
    const net = lm(), F = buildFlow(net), words = ['the', 'cat', 'sat', 'on', 'the'];
    assert.ok(F.numbered);
    for (const s of F.stages) for (const r of s.rows) {
      const t = F.tiles[r.tiles[0]];
      if (t.kind !== 'mat') continue;
      const pos = t.pos || [0, 1, 2, 3, 4];
      assert.deepEqual(t.rowLab, pos.map(p => words[p]), `${t.id}: its rows by word`);
      assert.deepEqual(t.rowNo, pos.map(p => p + 1), `${t.id}: and by position`);
    }
    assert.deepEqual(F.tiles['s3.0'].colLab, ['1', '2', '3', '4', '5'], 'the keys by position');
    assert.deepEqual(F.tiles['a3.1'].colLab, ['1', '2', '3', '4', '5']);
    assert.match(F.tiles['s3.0'].tip(1, 3), /2 “cat” can't see the later 4 “on”/);
  });

  test('the widths: d_model for the residual stream, 4 d_model for the FFN, the vocabulary for the words', () => {
    const F = buildFlow(lm());
    assert.deepEqual(F.dims, { model: 8, ffn: 32, vocab: 23 });
    const kind = id => F.tiles[id].dim?.kind ?? null;
    for (const id of ['L1', 'P4.0', 'L4', 'P6.0', 'L6']) assert.equal(kind(id), 'model', id);
    assert.equal(kind('L5'), 'ffn');
    for (const id of ['in', 'G7', 'L7']) assert.equal(kind(id), 'vocab', id);
    for (const id of ['q2.0', 's3.0', 'a3.0', 'z3.1', 'B1']) assert.equal(kind(id), null, id);
    assert.equal(F.tiles.L1.dim.html, 'd<sub>model</sub> = 8');
    assert.equal(F.tiles.L5.dim.html, '4 d<sub>model</sub> = 32');
    assert.deepEqual(F.residual.map(r => [r.sym, r.from, r.branch]), [['H', ['X'], ['attention']], ['Y', ['H'], ['FFN']]]);
    assert.match(F.dimsText, /Residual additions \(X \+ attention, H \+ FFN\) .* residual stream X, H and Y keeps one width/);
    assert.match(F.dimsText, /FFN widens to 32 .* one column per word, 23/);
  });

  test('the structure key ignores new values', () => {
    const net = lm(), k0 = flowKey(buildFlow(net));
    trained(net, 20);
    assert.equal(flowKey(buildFlow(net)), k0);
    assert.notEqual(flowKey(buildFlow(net, { every: true })), k0, 'every position rebuilds the ending');
  });

  test('a net without attention: one stage per layer, and a note', () => {
    const F = buildFlow(M.PRESETS.mlp.build(1));
    assert.equal(F.why, 'no-attention');
    assert.deepEqual(F.stages.map(s => s.key), ['in', 'layer1', 'layer2', 'layer3']);
    assert.equal(F.next, null);
    assert.equal(buildFlow(M.emptyNet()).why, 'empty');
  });

  test('the transformer block: products, residual sums and the FFN', () => {
    const F = buildFlow(M.PRESETS.transformer.build(1));
    assert.deepEqual(F.stages.map(s => s.key), ['in', 'qkv1', 'scores2', 'softmax2', 'mix2', 'proj3', 'sum3', 'layer4', 'proj5', 'sum5']);
    assert.equal(F.stages.find(s => s.key === 'sum3').tex, 'H = X + Z\\,W_O + b_O');
    assert.equal(F.stages.find(s => s.key === 'layer4').title, 'FFN');
    // not a language model, so its ending is every position as before; rows by token, widths d_model 2, FFN 4
    assert.ok(!F.lm);
    assert.deepEqual(F.tiles.in.rowLab, ['t1', 't2']);
    assert.equal(F.tiles.in.rowNo, null);
    assert.deepEqual(F.dims, { model: 2, ffn: 4, vocab: null });
    assert.deepEqual(['in', 'L3', 'L4', 'L5'].map(id => F.tiles[id].dim?.kind), ['model', 'model', 'ffn', 'model']);
    assert.equal(F.tiles.L4.dim.html, 'd<sub>ff</sub> = 4');
  });
});

test('texHtml', () => {
  assert.equal(texHtml('W_{out}'), 'W<sub>out</sub>');
  assert.equal(texHtml('Z\\,W_O'), 'Z W<sub>O</sub>');
  assert.equal(texHtml('K_h^{\\top}'), 'K<sub>h</sub>ᵀ');
  assert.equal(texHtml('a^{(2)}'), 'a<sup>(2)</sup>');
  assert.equal(texHtml('\\text{<b>}'), '&lt;b&gt;');
});
