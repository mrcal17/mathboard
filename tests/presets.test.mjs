import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate } from '../static/graph/lang.js';
// the features whose functions the presets use (they register them when imported)
import '../static/graph/features/transform.js';
import '../static/graph/features/combos.js';
import '../static/graph/features/systems.js';
import { PRESETS, presetRows, matches, pose2D, pose3D } from '../static/graph/features/presets.js';

const PALETTE = ['#e05a4f', '#4a90e2', '#43b05c', '#9b6ade', '#f5a623', '#26b5b5', '#e056a0', '#a1887f'];

test('presets have unique keys, a section, a label, a one-line note and a camera', () => {
  const keys = new Set();
  for (const p of PRESETS) {
    assert.ok(!keys.has(p.key), `duplicate key ${p.key}`);
    keys.add(p.key);
    assert.ok(p.group && p.label && p.note, p.key);
    assert.ok(!/\n/.test(p.note) && !/—/.test(p.note + p.label), `${p.key}: one line, no em-dashes`);
    const c = p.camera;
    assert.ok(c.extent > 0, `${p.key}: extent`);
    if (c.flat) assert.ok(c.frame.length === 4 && c.frame[0] < c.frame[1] && c.frame[2] < c.frame[3], `${p.key}: frame`);
    else assert.equal((c.dir ?? [1, 1, 1]).length, 3, `${p.key}: dir`);
  }
  assert.ok(PRESETS.length >= 20);
});

test('every preset row evaluates without an error', () => {
  for (const p of PRESETS) {
    const rows = presetRows(p, PALETTE);
    const res = evaluate(rows.map(r => r.src));
    res.forEach((r, i) => assert.equal(r.error, null, `${p.key} row ${i + 1} (${rows[i].src}): ${r.error}`));
    for (const r of rows) assert.match(r.color, /^#[0-9a-f]{6}$/, `${p.key}: colour of ${r.src}`);
  }
});

test('slider rows start inside their range', () => {
  for (const p of PRESETS) {
    const rows = presetRows(p, PALETTE), res = evaluate(rows.map(r => r.src));
    rows.forEach((r, i) => {
      if (r.min == null && r.max == null) return;
      const x = res[i].slider;
      assert.ok(x != null, `${p.key}: ${r.src} has a range but is not a slider`);
      assert.ok(r.min < r.max && x >= r.min && x <= r.max, `${p.key}: ${r.src} outside ${r.min}..${r.max}`);
    });
  }
});

// Samples a graph over the axes box the way features/plots.js does (curves over |t| <= E,
// surfaces on a grid), counting the samples that land inside the box.
function inside(g, E) {
  let hit = 0, n = 0;
  const lim = g.mode === 'surface' ? E : 1.2 * E;
  const ok = (v) => (typeof v === 'number' ? Number.isFinite(v) && Math.abs(v) <= lim
    : v?.v?.every((x) => Number.isFinite(x) && Math.abs(x) <= 4 * E));
  const at = (env) => { try { return g.at(env); } catch { return NaN; } };
  if (g.mode === 'surface') {
    const [u, w] = g.ins;
    for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++, n++) hit += ok(at({ [u]: -E + (2 * E * i) / 40, [w]: -E + (2 * E * j) / 40 })) ? 1 : 0;
  } else {
    for (let k = 0; k <= 400; k++, n++) hit += ok(at({ [g.ins[0]]: -E + (2 * E * k) / 400 })) ? 1 : 0;
  }
  return hit / n;
}

test('every visible graph draws inside the preset\'s axes box', () => {
  for (const p of PRESETS) {
    const rows = presetRows(p, PALETTE), res = evaluate(rows.map(r => r.src));
    res.forEach((r, i) => {
      if (rows[i].hidden || r.value?.type !== 'graph' || !r.value.mode) return;
      const share = inside(r.value, p.camera.extent);
      assert.ok(share > 0.05, `${p.key}: ${rows[i].src} is ${Math.round(share * 100)}% inside the box`);
    });
  }
});

test('restrictions keep the probability plots on 0 < p <= 1', () => {
  const p = PRESETS.find(x => x.key === 'nll-prob');
  const [nll, err] = evaluate(presetRows(p, PALETTE).map(r => r.src)).slice(1, 3).map(r => r.value);
  assert.throws(() => nll.at({ x: -0.5 }), /outside/);
  assert.throws(() => err.at({ x: 1.2 }), /outside/);
  assert.equal(err.at({ x: 0.2 }), 1);
  assert.ok(Math.abs(nll.at({ x: 1 })) < 1e-12);
});

test('search matches every word in the label, section, note or rows', () => {
  const find = (text) => PRESETS.filter(p => matches(p, text)).map(p => p.key);
  assert.equal(find('').length, PRESETS.length);
  assert.ok(find('softmax').includes('softmax-temp') && find('softmax').includes('ce-surface'));
  assert.deepEqual(find('learning rate'), ['gd-lr']);
  assert.ok(find('eigen').includes('transform'));
  assert.deepEqual(find('zzz'), []);
});

test('camera poses: 3D looks along dir at the target, 2D fits the frame', () => {
  const p3 = pose3D({ extent: 3, dir: [0, 0, 2], target: [1, 2, 3] }, 1.5);
  assert.deepEqual(p3.target, [1, 2, 3]);
  assert.equal(p3.position[0], 1);
  assert.ok(p3.position[2] > 3 && !p3.ortho);
  const cam = { flat: true, extent: 1.5, frame: [-0.1, 1.1, -0.15, 1.95] };
  for (const aspect of [0.7, 1.2, 2]) {
    const pose = pose2D(cam, aspect, 5), halfH = 5 / pose.zoom, halfW = halfH * aspect;
    assert.ok(pose.ortho);
    assert.ok(halfW >= 0.6 - 1e-9 && halfH >= 1.05 - 1e-9);
    assert.ok(Math.abs(halfW - 0.6) < 1e-9 || Math.abs(halfH - 1.05) < 1e-9, 'tight on one side');
    assert.deepEqual(pose.target, [0.5, 0.9, 0]);
  }
});
