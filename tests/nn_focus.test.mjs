// Tests for static/nn/focus.js (the Net tab's lens rules, docs/NN_LENS.md).
// Run: node --test tests/nn_focus.test.mjs
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../static/nn/model.js';
import { emphasis, cleanLens, lensInfo, stages, stepStage, copyLens, tokenLabel, tokenNames, DEFAULT_LENS, ATTN_PARTS } from '../static/nn/focus.js';
import { buildSteps } from '../static/nn/tour.js';

// ---------------------------------------------------------------- helpers
const build = key => M.PRESETS[key].build(1);
const lens = patch => ({ ...copyLens(DEFAULT_LENS), ...patch });
const layerId = (net, l) => net.layers[l].id;
// Nodes of layer l matching { g, t, f } (any field left out matches everything).
const nodesAt = (net, l, want = {}) => M.nodesIn(net, l).filter(n => {
  const p = M.tokenPos(net, n.id);
  return ['g', 't', 'f'].every(k => want[k] === undefined || want[k] === { g: p.g, t: p.token, f: p.feature }[k]);
});
const edgesInto = (net, l) => { const ids = new Set(M.nodesIn(net, l).map(n => n.id)); return net.edges.filter(e => ids.has(e.to)); };
const every = (list, fn, v, msg) => { assert.ok(list.length > 0, `${msg}: nothing to check`); for (const x of list) assert.equal(fn(x.id), v, `${msg} (${x.id})`); };
const near = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-12, `${msg}: ${a} vs ${b}`);

// Transformer layers: 0 X, 1 Q/K/V, 2 attention Z, 3 H = X + Z W_O, 4 FFN, 5 Y.
const TF = { X: 0, QKV: 1, Z: 2, H: 3, FFN: 4, Y: 5 };

describe('the default lens', () => {
  for (const key of ['transformer', 'multihead', 'causal', 'deep']) {
    test(`${key}: nothing dimmed, nothing hidden`, () => {
      const net = build(key), fwd = M.forward(net);
      for (const L of [null, undefined, DEFAULT_LENS, copyLens(null)]) {
        const E = emphasis(net, fwd, L);
        assert.equal(E.any, false);
        assert.equal(E.hides, false);
        for (const n of net.nodes) assert.equal(E.node(n.id), 1);
        for (const e of net.edges) { assert.equal(E.edge(e.id), 1); assert.equal(E.hidden.edge(e.id), false); }
        net.layers.forEach((_, l) => { assert.equal(E.rows(l), null); assert.equal(E.heads(l), null); assert.equal(E.layer(l), 1); });
      }
    });
  }
  test('causal: masked pairs are always hidden, the rest never', () => {
    const net = build('causal'), E = emphasis(net, M.forward(net), null);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.equal(E.hidden.attn(2, i, j, 0), j > i, `${i},${j}`);
  });
});

describe('focus', () => {
  test('transformer: focus on H = X + Z W_O lights H, Z, X (residual) and the edges into H only', () => {
    const net = build('transformer'), E = emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, TF.H) } }));
    assert.equal(E.any, true);
    for (const l of [TF.X, TF.Z, TF.H]) every(M.nodesIn(net, l), E.node, 1, `layer ${l}`);
    for (const l of [TF.QKV, TF.FFN, TF.Y]) every(M.nodesIn(net, l), E.node, 0, `layer ${l}`);
    const into = new Set(edgesInto(net, TF.H).map(e => e.id));
    assert.ok(edgesInto(net, TF.H).some(e => e.fixed), 'the residual is among them');
    for (const e of net.edges) assert.equal(E.edge(e.id), into.has(e.id) ? 1 : 0, e.id);
    assert.equal(E.attn(TF.Z, 0, 1, 0), 0, 'attention edges go into Z, not H');
    assert.equal(E.layer(TF.FFN), 0);
    assert.deepEqual([...E.rows(TF.FFN)], []);
    assert.equal(E.rows(TF.H), null);
  });
  test('transformer: focus by layer index works the same as by id', () => {
    const net = build('transformer'), fwd = M.forward(net);
    const a = emphasis(net, fwd, lens({ focus: { layer: layerId(net, TF.FFN) } }));
    const b = emphasis(net, fwd, lens({ focus: { layer: TF.FFN } }));
    for (const n of net.nodes) assert.equal(a.node(n.id), b.node(n.id));
    assert.deepEqual(cleanLens(net, lens({ focus: { layer: TF.FFN } })).focus, { layer: layerId(net, TF.FFN) });
  });
  test('transformer: scores and softmax light Q, K and the attention layer', () => {
    const net = build('transformer'), fwd = M.forward(net);
    for (const part of ['scores', 'softmax']) {
      const E = emphasis(net, fwd, lens({ focus: { layer: layerId(net, TF.Z), part } }));
      every(nodesAt(net, TF.QKV, { g: 0 }), E.node, 1, `${part} Q`);
      every(nodesAt(net, TF.QKV, { g: 1 }), E.node, 1, `${part} K`);
      every(nodesAt(net, TF.QKV, { g: 2 }), E.node, 0, `${part} V`);
      every(M.nodesIn(net, TF.Z), E.node, 1, `${part} Z`);
      every(M.nodesIn(net, TF.X), E.node, 0, `${part} X`);
      every(net.edges, E.edge, 0, `${part} edges`);
      assert.equal(E.attn(TF.Z, 1, 0, 0), 0, `${part}: no attention edges`);
      assert.deepEqual([...E.groups(TF.QKV)].sort(), [0, 1]);
    }
  });
  test('transformer: mix lights V, the attention edges and Z', () => {
    const net = build('transformer'), E = emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, TF.Z), part: 'mix' } }));
    every(nodesAt(net, TF.QKV, { g: 2 }), E.node, 1, 'V');
    every(nodesAt(net, TF.QKV, { g: 0 }), E.node, 0, 'Q');
    every(nodesAt(net, TF.QKV, { g: 1 }), E.node, 0, 'K');
    every(M.nodesIn(net, TF.Z), E.node, 1, 'Z');
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) assert.equal(E.attn(TF.Z, i, j, 0), 1);
    assert.deepEqual([...E.groups(TF.QKV)], [2]);
  });
  test('transformer: the whole attention layer lights Q, K, V, Z and the attention edges', () => {
    const net = build('transformer'), E = emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, TF.Z) } }));
    every(M.nodesIn(net, TF.QKV), E.node, 1, 'Q, K, V');
    every(M.nodesIn(net, TF.Z), E.node, 1, 'Z');
    assert.equal(E.attn(TF.Z, 0, 1, 0), 1);
    assert.equal(E.groups(TF.QKV), null);
  });
  test('transformer: part Q on the Q, K, V layer keeps the Q group and the edges into it', () => {
    const net = build('transformer'), E = emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, TF.QKV), part: 'Q' } }));
    every(nodesAt(net, TF.QKV, { g: 0 }), E.node, 1, 'Q');
    every(nodesAt(net, TF.QKV, { g: 1 }), E.node, 0, 'K');
    every(nodesAt(net, TF.QKV, { g: 2 }), E.node, 0, 'V');
    every(M.nodesIn(net, TF.X), E.node, 1, 'X feeds it');
    const q = new Set(nodesAt(net, TF.QKV, { g: 0 }).map(n => n.id));
    for (const e of net.edges) assert.equal(E.edge(e.id), q.has(e.to) ? 1 : 0, e.id);
    assert.deepEqual([...E.groups(TF.QKV)], [0]);
  });
  test('deep (plain dense): focus on Hidden 2 lights Hidden 1, Hidden 2 and W between them', () => {
    const net = build('deep'), E = emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, 2) } }));
    net.layers.forEach((_, l) => every(M.nodesIn(net, l), E.node, l === 1 || l === 2 ? 1 : 0, `layer ${l}`));
    const into = new Set(edgesInto(net, 2).map(e => e.id));
    for (const e of net.edges) assert.equal(E.edge(e.id), into.has(e.id) ? 1 : 0);
  });
  test('a focus that covers everything dims nothing (any is false)', () => {
    const net = build('perceptron');
    assert.equal(net.layers.length, 2);
    assert.equal(emphasis(net, M.forward(net), lens({ focus: { layer: layerId(net, 1) } })).any, false);
  });
});

describe('token', () => {
  test('transformer: token 2 lights token 2 in every layer; keys and values get A_2j', () => {
    const net = build('transformer'), fwd = M.forward(net), t = 1;
    const E = emphasis(net, fwd, lens({ token: t }));
    assert.equal(E.any, true);
    for (const l of [TF.X, TF.Z, TF.H, TF.FFN, TF.Y]) {
      every(nodesAt(net, l, { t }), E.node, 1, `layer ${l} token 2`);
      every(nodesAt(net, l, { t: 0 }), E.node, 0, `layer ${l} token 1`);
    }
    every(nodesAt(net, TF.QKV, { g: 0, t }), E.node, 1, 'q_2');
    every(nodesAt(net, TF.QKV, { g: 0, t: 0 }), E.node, 0, 'q_1');
    const A = fwd.attn[TF.Z].heads[0].A;
    for (const g of [1, 2]) for (let j = 0; j < 2; j++) for (const n of nodesAt(net, TF.QKV, { g, t: j })) near(E.node(n.id), A[t][j], `K/V token ${j + 1}`);
    for (let j = 0; j < 2; j++) { assert.equal(E.attn(TF.Z, t, j, 0), 1); assert.equal(E.attn(TF.Z, 0, j, 0), 0); }
    for (const l of [TF.X, TF.QKV, TF.Z, TF.H, TF.FFN, TF.Y]) assert.deepEqual([...E.rows(l)], [t]);
    // edges: within token 2 lit (residuals too); anything touching token 1 dimmed
    for (const e of net.edges) {
      const a = M.tokenPos(net, e.from), b = M.tokenPos(net, e.to);
      if (a.token === 0 || b.token === 0) assert.equal(E.edge(e.id), 0, e.id);
      else if (a.l !== TF.QKV && b.l !== TF.QKV) assert.equal(E.edge(e.id), 1, e.id);
    }
  });
  test('transformer: without a forward pass, keys and values stay lit', () => {
    const net = build('transformer'), E = emphasis(net, null, lens({ token: 0 }));
    every(nodesAt(net, TF.QKV, { g: 2 }), E.node, 1, 'V');
  });
  test('causal: token 1 attends only to itself, so later keys and values dim', () => {
    const net = build('causal'), E = emphasis(net, M.forward(net), lens({ token: 0 }));
    for (const g of [1, 2]) {
      every(nodesAt(net, 1, { g, t: 0 }), E.node, 1, 'token 1');
      every(nodesAt(net, 1, { g, t: 1 }), E.node, 0, 'token 2');
      every(nodesAt(net, 1, { g, t: 2 }), E.node, 0, 'token 3');
    }
    every(nodesAt(net, 3, { t: 0 }), E.node, 1, 'output ŷ_1');
    every(nodesAt(net, 3, { t: 2 }), E.node, 0, 'output ŷ_3');
  });
  test('token and focus intersect', () => {
    const net = build('transformer'), E = emphasis(net, M.forward(net), lens({ token: 1, focus: { layer: layerId(net, TF.H) } }));
    for (const l of [TF.X, TF.Z, TF.H]) {
      every(nodesAt(net, l, { t: 1 }), E.node, 1, `layer ${l} token 2`);
      every(nodesAt(net, l, { t: 0 }), E.node, 0, `layer ${l} token 1`);
    }
    every(M.nodesIn(net, TF.FFN), E.node, 0, 'FFN is outside the focus');
    for (const e of edgesInto(net, TF.H)) assert.equal(E.edge(e.id), M.tokenPos(net, e.to).token === 1 ? 1 : 0);
  });
  test('deep (plain dense): token does nothing and is cleaned away', () => {
    const net = build('deep'), E = emphasis(net, M.forward(net), lens({ token: 0, head: 0 }));
    assert.equal(E.any, false);
    assert.equal(cleanLens(net, lens({ token: 0, head: 0 })).token, null);
    assert.equal(cleanLens(net, lens({ token: 0, head: 0 })).head, null);
  });
});

describe('head', () => {
  test('multihead: head 2 keeps feature 2 of Q, K, V and Z, and head 2\'s attention edges', () => {
    const net = build('multihead'), E = emphasis(net, M.forward(net), lens({ head: 1 }));
    assert.equal(E.any, true);
    every(nodesAt(net, 1, { f: 1 }), E.node, 1, 'Q, K, V feature 2');
    every(nodesAt(net, 1, { f: 0 }), E.node, 0, 'Q, K, V feature 1');
    every(nodesAt(net, 2, { f: 1 }), E.node, 1, 'Z feature 2');
    every(nodesAt(net, 2, { f: 0 }), E.node, 0, 'Z feature 1');
    every(M.nodesIn(net, 0), E.node, 1, 'X is not split by head');
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { assert.equal(E.attn(2, i, j, 1), 1); assert.equal(E.attn(2, i, j, 0), 0); }
    assert.deepEqual([...E.heads(1)], [1]);
    assert.deepEqual([...E.heads(2)], [1]);
    assert.equal(E.heads(0), null);
    for (const e of net.edges) assert.equal(E.edge(e.id), M.tokenPos(net, e.to).feature === 1 ? 1 : 0);
  });
  test('multihead: head and token intersect; keys get A of their own head', () => {
    const net = build('multihead'), fwd = M.forward(net), E = emphasis(net, fwd, lens({ head: 1, token: 2 }));
    const A = fwd.attn[2].heads[1].A;
    for (let j = 0; j < 3; j++) {
      for (const n of nodesAt(net, 1, { g: 1, t: j, f: 1 })) near(E.node(n.id), A[2][j], `k_${j + 1},2`);
      every(nodesAt(net, 1, { g: 1, t: j, f: 0 }), E.node, 0, `k_${j + 1},1 (head 1)`);
    }
    assert.equal(E.attn(2, 2, 0, 1), 1);
    assert.equal(E.attn(2, 1, 0, 1), 0);
    assert.equal(E.attn(2, 2, 0, 0), 0);
  });
  test('a single-head net has no head to pick', () => {
    const net = build('transformer');
    assert.equal(lensInfo(net).heads, 1);
    assert.equal(cleanLens(net, lens({ head: 0 })).head, null);
  });
});

describe('show and thresholds hide', () => {
  test('transformer: minW hides small trainable weights, never fixed ones; show.fixed hides the residuals', () => {
    const net = build('transformer');
    const E = emphasis(net, M.forward(net), lens({ minW: 0.3 }));
    assert.equal(E.hides, true);
    assert.equal(E.any, false, 'hiding is not dimming');
    for (const e of net.edges) assert.equal(E.hidden.edge(e.id), !e.fixed && Math.abs(e.w) < 0.3, e.id);
    const F = emphasis(net, M.forward(net), lens({ show: { weights: true, attention: true, fixed: false } }));
    for (const e of net.edges) assert.equal(F.hidden.edge(e.id), !!e.fixed);
    const W = emphasis(net, M.forward(net), lens({ show: { weights: false, attention: true, fixed: true } }));
    for (const e of net.edges) assert.equal(W.hidden.edge(e.id), !e.fixed);
  });
  test('multihead: minA hides attention edges below it; show.attention hides them all', () => {
    const net = build('multihead'), fwd = M.forward(net);
    const E = emphasis(net, fwd, lens({ minA: 0.34 }));
    for (let h = 0; h < 2; h++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      assert.equal(E.hidden.attn(2, i, j, h), fwd.attn[2].heads[h].A[i][j] < 0.34);
    }
    const G = emphasis(net, fwd, lens({ show: { weights: true, attention: false, fixed: true } }));
    assert.equal(G.hidden.attn(2, 0, 0, 0), true);
    assert.equal(G.hidden.edge(net.edges[0].id), false);
  });
  test('deep: thresholds declutter a big dense net', () => {
    const net = build('deep'), E = emphasis(net, M.forward(net), lens({ minW: 0.5 }));
    const gone = net.edges.filter(e => E.hidden.edge(e.id)).length;
    assert.ok(gone > 0 && gone < net.edges.length, `${gone} of ${net.edges.length} hidden`);
  });
});

describe('cleanLens, lensInfo, stages', () => {
  test('a clean lens comes back as the same object; null gives the default', () => {
    const net = build('transformer'), L = cleanLens(net, lens({ token: 1, focus: { layer: layerId(net, TF.Z), part: 'mix' } }));
    assert.equal(cleanLens(net, L), L);
    assert.equal(cleanLens(net, null), DEFAULT_LENS);
    assert.deepEqual(copyLens(null), { focus: null, token: null, head: null, show: { weights: true, attention: true, fixed: true }, minW: 0, minA: 0 });
  });
  test('fields that stop making sense are reset', () => {
    const net = build('transformer');
    const L = cleanLens(net, { focus: { layer: 'nope' }, token: 5, head: 1, show: { weights: 0 }, minW: -1, minA: 3 });
    assert.deepEqual(L, { focus: null, token: null, head: null, show: { weights: true, attention: true, fixed: true }, minW: 0, minA: 1 });
    assert.deepEqual(cleanLens(net, lens({ focus: { layer: layerId(net, TF.Z), part: 'Q' } })).focus, { layer: layerId(net, TF.Z) }, 'Q is not a part of attention');
    assert.deepEqual(cleanLens(net, lens({ focus: { layer: layerId(net, TF.QKV), part: 'mix' } })).focus, { layer: layerId(net, TF.QKV) });
    assert.deepEqual(cleanLens(net, lens({ focus: { layer: layerId(net, TF.QKV), part: 'K' } })).focus, { layer: layerId(net, TF.QKV), part: 'K' });
    // the net changes shape: a layer goes, the token count shrinks
    const small = build('transformer');
    const L2 = lens({ token: 1, focus: { layer: layerId(small, TF.FFN) } });
    M.removeLayer(small, layerId(small, TF.FFN), { bridge: true, seed: 1 });
    assert.equal(cleanLens(small, L2).focus, null);
    assert.equal(cleanLens(build('attention'), lens({ token: 2 })).token, 2);
    assert.equal(cleanLens(build('transformer'), lens({ token: 2 })).token, null);
  });
  test('lensInfo says which controls apply', () => {
    const t = lensInfo(build('transformer')), m = lensInfo(build('multihead')), d = lensInfo(build('deep'));
    assert.deepEqual([t.tokens, t.heads, t.hasAttention, t.hasFixed, t.hasWeights], [2, 1, true, true, true]);
    assert.deepEqual([m.tokens, m.heads, m.hasFixed], [3, 2, false]);
    assert.deepEqual([d.tokens, d.heads, d.hasAttention, d.hasFixed], [0, 1, false, false]);
    assert.deepEqual(t.layers.map(l => l.kind), ['dense', 'qkv', 'attention', 'dense', 'dense', 'dense']);
    assert.deepEqual(t.layers[TF.QKV].parts, ['Q', 'K', 'V']);
    assert.deepEqual(t.layers[TF.Z].parts, ATTN_PARTS);
    assert.ok(d.maxW > 0);
  });
  test('stages step through the transformer in reading order', () => {
    const net = build('transformer'), id = l => layerId(net, l), list = stages(net);
    assert.deepEqual(list, [{ layer: id(1) }, { layer: id(2), part: 'scores' }, { layer: id(2), part: 'softmax' },
      { layer: id(2), part: 'mix' }, { layer: id(3) }, { layer: id(4) }, { layer: id(5) }]);
    assert.deepEqual(stepStage(net, null, 1), list[0]);
    assert.deepEqual(stepStage(net, null, -1), list[6]);
    assert.deepEqual(stepStage(net, list[0], 1), list[1]);
    assert.equal(stepStage(net, list[6], 1), null);
    assert.equal(stepStage(net, list[0], -1), null);
    assert.deepEqual(stepStage(net, { layer: id(1), part: 'K' }, 1), list[1], 'from a part: on to the next stage');
    assert.deepEqual(stepStage(net, { layer: id(1), part: 'K' }, -1), list[0], '... or back to its layer');
    assert.deepEqual(stepStage(net, { layer: id(2) }, 1), list[1], 'a whole attention layer: into its parts');
    assert.deepEqual(stepStage(net, { layer: id(2) }, -1), list[0]);
    assert.deepEqual(stepStage(net, { layer: id(0) }, 1), list[0], 'the input (not a stage): ] goes to the first stage');
    assert.equal(stepStage(net, { layer: id(0) }, -1), null, '... and [ to none');
  });
  test('token labels use net.meta.tokenNames', () => {
    const net = build('transformer');
    assert.equal(tokenLabel(net, 1), 't2');
    net.meta.tokenNames = ['the', ' cat '];
    assert.equal(tokenLabel(net, 1), 'cat');
    assert.deepEqual(lensInfo(net).names, ['the', 'cat']);
    net.meta.tokenNames = ['t1', 'cat', 7];
    assert.deepEqual(lensInfo(net).names, [null, 'cat', '7'], 'a slot holding its own default is unnamed');
  });
});

// One rule for token names (docs/NN_LENS.md), read through focus.js by the canvas, the matrix panel,
// the cards, the Train plot, the attention panel, the lens bar and the Explain captions.
describe('token names: one rule', () => {
  test('a trimmed string or number; empty, junk or the slot\'s own default is unnamed', () => {
    const net = build('words');
    assert.deepEqual(tokenNames(net), ['the', 'cat', 'sat']);
    net.meta.tokenNames = ['t1', ' cat ', '', 3, null, { a: 1 }, 't2'];
    assert.deepEqual(tokenNames(net), [null, 'cat', null, '3', null, null, 't2'], "'t2' is only a default in slot 2");
    assert.equal(tokenLabel(net, 0), 't1');
    assert.equal(tokenLabel(net, 1), 'cat');
    assert.equal(tokenLabel(net, 2), 't3');
    assert.equal(tokenLabel(net, 9), 't10', 'past the list: the default');
    delete net.meta.tokenNames;
    assert.deepEqual(tokenNames(net), []);
    assert.deepEqual(tokenNames(null), []);
  });
  test('Explain captions follow it', () => {
    const net = M.normalize(build('words'));
    net.meta.tokenNames = ['t1', ' cat ', 't3'];
    const fwd = M.forward(net), env = { net, fwd, bwd: null };
    const steps = buildSteps(net, M, fwd);
    const text = steps.map(s => `${s.title(env)} ${s.text(env)}`).join('\n');
    assert.match(steps.find(s => s.key === 'tokens').text(env), /one per token \(token 1, cat, token 3\)/);
    assert.match(text, /Follow “cat”/);
    assert.doesNotMatch(text, /“t1”|“t3”/, 'a default is never quoted as a name');
    net.meta.tokenNames = ['t1', 't2', 't3'];
    assert.doesNotMatch(steps.find(s => s.key === 'tokens').text(env), /one per token \(/, 'all defaults: no name list');
  });
});
