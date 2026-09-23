import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ease, sliderOf, sameShape, planTransition, samePose, stepTarget, normalizeSteps } from '../static/graph/features/lecture.js';

const R = (src, color = '#e05a4f', extra = {}) => ({ src, color, ...extra });

test('ease hits its endpoints and is symmetric', () => {
  assert.equal(ease(0), 0);
  assert.equal(ease(1), 1);
  assert.equal(ease(0.5), 0.5);
  assert.ok(Math.abs(ease(0.25) + ease(0.75) - 1) < 1e-12);
});

test('sliderOf recognises number rows only', () => {
  assert.deepEqual(sliderOf('t = 0.5'), { name: 't', value: 0.5 });
  assert.deepEqual(sliderOf('a = -2 # scale'), { name: 'a', value: -2 });
  assert.equal(sliderOf('u = (1, 2, 3)'), null);
  assert.equal(sliderOf('2'), null);
  assert.equal(sliderOf('a = b'), null);
  assert.equal(sliderOf('a = 2 @ (1, 0, 0)'), null);
  assert.equal(sliderOf(''), null);
});

test('sameShape needs equal length and colours; a missing colour matches', () => {
  assert.ok(sameShape([R('a'), R('b', '#4a90e2')], [R('x'), { src: 'y' }]));
  assert.ok(!sameShape([R('a')], [R('a'), R('b')]));
  assert.ok(!sameShape([R('a')], [R('a', '#000000')]));
});

test('planTransition tweens when only slider values differ', () => {
  const a = [R('u = (1, 2, 0)'), R('t = 0.5'), R('k = 1'), R('t u')];
  const b = [R('u = (1, 2, 0)'), R('t = 2'), R('k = 1 # same'), R('t u', '#e05a4f', { hidden: true })];
  const p = planTransition(a, b);
  assert.equal(p.mode, 'tween');
  assert.deepEqual(p.tweens, [{ i: 1, name: 't', from: 0.5, to: 2 }]);
  assert.deepEqual(planTransition(a, a), { mode: 'tween', tweens: [] });
});

test('planTransition patches other source changes and rebuilds on shape changes', () => {
  const a = [R('u = (1, 2, 0)'), R('t = 0.5')];
  assert.equal(planTransition(a, [R('u = (0, 1, 0)'), R('t = 1')]).mode, 'patch');
  assert.equal(planTransition(a, [R('u = (1, 2, 0)'), R('s = 1')]).mode, 'patch'); // renamed slider
  assert.equal(planTransition(a, [R('u = (1, 2, 0)'), R('t = u')]).mode, 'patch'); // no longer a slider
  assert.equal(planTransition(a, [R('u = (1, 2, 0)')]).mode, 'rebuild');
  assert.equal(planTransition(a, [R('u = (1, 2, 0)'), R('t = 0.5', '#43b05c')]).mode, 'rebuild');
});

test('samePose compares position, target, zoom, projection and extent', () => {
  const p = { position: [10, 5, 6], target: [0, 0, 0], zoom: 1, ortho: false, extent: 6 };
  assert.ok(samePose(p, { ...p, position: [10, 5, 6 + 1e-9] }));
  assert.ok(!samePose(p, { ...p, position: [10, 5, 6.01] }));
  assert.ok(!samePose(p, { ...p, zoom: 2 }));
  assert.ok(!samePose(p, { ...p, ortho: true }));
  assert.ok(!samePose(p, { ...p, extent: 8 }));
  assert.ok(samePose(p, { position: [10, 5, 6] }));
  assert.ok(!samePose(p, null));
  assert.ok(samePose(null, null));
});

test('stepTarget walks forward and back and stops at the ends', () => {
  assert.equal(stepTarget(-1, 1, 3), 0);
  assert.equal(stepTarget(-1, -1, 3), -1);
  assert.equal(stepTarget(0, 1, 3), 1);
  assert.equal(stepTarget(2, 1, 3), -1);
  assert.equal(stepTarget(0, -1, 3), -1);
  assert.equal(stepTarget(1, -1, 3), 0);
  assert.equal(stepTarget(5, 1, 3), 2);
  assert.equal(stepTarget(0, 1, 0), -1);
});

test('normalizeSteps accepts files, plain-string rows and fills colours', () => {
  const pal = ['#111111', '#222222'];
  const steps = normalizeSteps({ format: 'mathboard-steps', steps: [
    { title: 'Span', rows: ['u = (1, 2, 0)', 'v = (0, 1, 1)', { src: 'w = u + v', color: '#abcdef', hidden: 1, min: -2, max: 'x' }] },
    { rows: [], camera: { position: [1, 2, 3], zoom: 2, ortho: 1, extent: 8 }, collapsed: true, flat: false },
  ] }, pal);
  assert.equal(steps.length, 2);
  assert.deepEqual(steps[0].rows, [
    { src: 'u = (1, 2, 0)', color: '#111111' },
    { src: 'v = (0, 1, 1)', color: '#222222' },
    { src: 'w = u + v', color: '#abcdef', hidden: true, min: -2 },
  ]);
  assert.equal(steps[0].title, 'Span');
  assert.equal(steps[0].camera, null);
  assert.equal(steps[0].collapsed, null);
  assert.deepEqual(steps[1].rows, [{ src: '' }]);
  assert.deepEqual(steps[1].camera, { position: [1, 2, 3], target: [0, 0, 0], zoom: 2, ortho: true, extent: 8 });
  assert.equal(steps[1].collapsed, true);
  assert.equal(steps[1].flat, false);
  assert.equal(steps[1].spin, false);
  assert.equal(steps[0].flat, false);
  assert.equal(normalizeSteps([]).length, 0);
});

test('normalizeSteps rejects garbage and drops bad cameras', () => {
  assert.throws(() => normalizeSteps(null), /not a steps file/);
  assert.throws(() => normalizeSteps({ rows: [] }), /not a steps file/);
  assert.throws(() => normalizeSteps([{ title: 'x' }]), /step 1 has no rows/);
  const [s] = normalizeSteps([{ rows: [null, 3, 'a = 1'], camera: { position: [1, 'b', 3] } }]);
  assert.deepEqual(s.rows, [{ src: 'a = 1' }]);
  assert.equal(s.camera, null);
});

test('normalizeSteps round-trips its own output', () => {
  const once = normalizeSteps([{ title: 't', rows: [R('a = 1')], camera: { position: [1, 2, 3], target: [0, 0, 1], zoom: 1, ortho: false, extent: 6 }, collapsed: false, flat: true, spin: false }]);
  assert.deepEqual(normalizeSteps(JSON.parse(JSON.stringify({ steps: once }))), once);
});
