import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate } from '../static/graph/lang.js';
import {
  parseLiteral, rewriteSource, intersectRayPlane, closestOnVertical,
  snapValue, snapThreshold, roundStep, moveComponents,
} from '../static/graph/features/drag.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const nearArr = (a, b, eps = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => near(x, b[i], eps));
};

test('plain literals are draggable', () => {
  for (const [src, kind, values] of [
    ['u = (1, 2, 3)', 'vec', [1, 2, 3]],
    ['(1, 2, 3)', 'vec', [1, 2, 3]],
    ['  w=[-1,0.5,2e1]', 'vec', [-1, 0.5, 20]],
    ['v = (1, -2) @ u', 'vec', [1, -2]],
    ['P = point(1, 2, 3)', 'point', [1, 2, 3]],
    ['point(-1, .5)', 'point', [-1, 0.5]],
    ['a1 = (−1, + 2, - 3) # note', 'vec', [-1, 2, -3]],
    ['pointA = (1, 2, 3)', 'vec', [1, 2, 3]],
  ]) {
    const p = parseLiteral(src);
    assert.ok(p, src);
    assert.equal(p.kind, kind, src);
    assert.deepEqual(p.values, values, src);
  }
});

test('computed and malformed rows are not draggable', () => {
  for (const src of ['u + v', 'w = u + v', '2 (1, 2, 3)', '(1, 2, 3) u', 'u = -(1, 2, 3)', '(1, 2, 3]',
    '[1, 2, 3)', 'point(u)', 'point(1, 2, 3', '(1, 2, 3, 4)', '(1)', 'a = 1', 'A = [[1, 0], [0, 1]]',
    'u = (1/2, 1, 0)', 'u = (pi, 1, 0)', '# (1, 2, 3)', '', 'u = (1., 2, 3)', 'u v = (1, 2, 3)']) {
    assert.equal(parseLiteral(src), null, src);
  }
});

test('the literal parser agrees with lang.js on values', () => {
  for (const src of ['u = (1, 2, 3)', 'w=[-1,0.5,2e1]', 'P = point(4, -2)', '(−1, +2, - 3) // x']) {
    const [row] = evaluate([src]);
    assert.equal(row.error, null, src);
    const p = parseLiteral(src);
    assert.equal(row.value.type, p.kind);
    nearArr(row.value.v.slice(0, p.values.length), p.values);
  }
});

test('rewrite keeps name, bracket style, origin, comment and separators', () => {
  assert.equal(rewriteSource('u = (1, 2, 3)', [2, -1.5, 3]), 'u = (2, -1.5, 3)');
  assert.equal(rewriteSource('v = [1, 2, 3] @ u', [0, 4, 3]), 'v = [0, 4, 3] @ u');
  assert.equal(rewriteSource('P = point(1, 2, 3)', [1, 2, 7]), 'P = point(1, 2, 7)');
  assert.equal(rewriteSource('(1,2,3)', [5, 2, 3]), '(5,2,3)');
  assert.equal(rewriteSource('  u=(1, 2, 3)  @ (1, 0, 0) # tip-to-tail', [1, 2.25, 3]), '  u=(1, 2.25, 3)  @ (1, 0, 0) # tip-to-tail');
  assert.equal(rewriteSource('u = (1, 2, 3)', [1 / 3, 2, 3]), 'u = (0.3333, 2, 3)');
  assert.equal(rewriteSource('u = (1, 2, 3)', [-0, 2, 3]), 'u = (0, 2, 3)');
  assert.equal(rewriteSource('u = (a, 2, 3)', [1, 2, 3]), null);
});

test('rewrite leaves untouched component text alone', () => {
  assert.equal(rewriteSource('u = (1.0e1, −2, 3)', [10, -2, 4]), 'u = (1.0e1, −2, 4)');
  assert.equal(rewriteSource('u = (0.123456, 2, 3)', [0.123456, 5, 3]), 'u = (0.123456, 5, 3)');
});

test('2D literals grow a z only when it becomes non-zero', () => {
  assert.equal(rewriteSource('u = (1, 2)', [3, 4, 0]), 'u = (3, 4)');
  assert.equal(rewriteSource('u = (1, 2)', [1, 2, 1.5]), 'u = (1, 2, 1.5)');
  assert.equal(rewriteSource('P = point(1, 2)', [1, 2, -1]), 'P = point(1, 2, -1)');
  assert.equal(rewriteSource('u = (1, 2, 0)', [1, 3, 0]), 'u = (1, 3, 0)');
});

test('rewritten rows still evaluate to the new values', () => {
  const src = rewriteSource('v = [1, 2, 3] @ (1, 1, 0) # c', [-2.5, 0, 4]);
  const [row] = evaluate([src]);
  assert.deepEqual(row.value.v, [-2.5, 0, 4]);
  assert.deepEqual(row.origin, [1, 1, 0]);
});

test('ray / horizontal plane intersection', () => {
  nearArr(intersectRayPlane([0, 0, 10], [1, 0, -1], [5, 5, 3], [0, 0, 1]), [7, 0, 3]);
  nearArr(intersectRayPlane([4, -2, 1], [0, 1, 2], [0, 0, 3], [0, 0, 1]), [4, -1, 3]);
  assert.equal(intersectRayPlane([0, 0, 10], [1, 0, 1], [0, 0, 3], [0, 0, 1]), null); // behind the ray
  assert.equal(intersectRayPlane([0, 0, 10], [1, 0, 0], [0, 0, 3], [0, 0, 1]), null); // parallel
  assert.equal(intersectRayPlane([0, 0, 10], [1, 0, -0.01], [0, 0, 3], [0, 0, 1], 0.02), null); // grazing
  nearArr(intersectRayPlane([0, 0, 0], [1, 1, 0], [2, 0, 0], [1, 0, 0]), [2, 2, 0]); // any plane
});

test('closest point on the vertical line through the tip', () => {
  nearArr(closestOnVertical([10, 0, 0], [-1, 0, 0.2], [0, 0, 1]), [0, 0, 2]);
  // a ray passing beside the line: the nearest approach sets z
  nearArr(closestOnVertical([10, 3, 5], [-2, 0, -1], [1, 1, 0]), [1, 1, 0.5]);
  assert.equal(closestOnVertical([0, 0, 10], [0, 0, -1], [0, 0, 0]), null);   // looking straight down
  assert.equal(closestOnVertical([10, 0, 0], [1, 0, 0.2], [0, 0, 1]), null);  // line is behind
});

test('snapping to the integer grid', () => {
  assert.equal(snapValue(2.08, 0.1, 0.01), 2);
  assert.equal(snapValue(-0.05, 0.1, 0.01), 0);
  assert.ok(Object.is(snapValue(-0.05, 0.1, 0.01), 0));
  assert.equal(snapValue(2.2, 0.1, 0.01), 2.2);
  assert.equal(snapValue(2.2345, 0.1, 0.01), 2.23);
  assert.equal(snapValue(2.08, 0, 0.01), 2.08);        // Alt: free, still tidy
  assert.equal(snapValue(0.1 + 0.2, 0, 0.1), 0.3);
  assert.equal(snapValue(1.23456, 0, 0), 1.23456);
});

test('snap threshold and rounding step follow the pixel size', () => {
  near(snapThreshold(0.0175), 0.175);
  assert.equal(snapThreshold(1), 0.3);
  assert.equal(roundStep(0.0175), 0.01);
  assert.equal(roundStep(0.175), 0.1);
  assert.equal(roundStep(0.002), 0.001);
  assert.equal(roundStep(1e-7), 1e-4);
  assert.equal(roundStep(5), 1);
});

test('moving the tip rewrites only the dragged components, relative to the origin', () => {
  const o = [1, 1, 0], v = [1, 2, 3];
  assert.deepEqual(moveComponents(v, o, [4.03, 0.5, 3], [0, 1], 0.1, 0.01), [3, -0.5, 3]);
  assert.deepEqual(moveComponents(v, o, [9, 9, 5.96], [2], 0.1, 0.01), [1, 2, 6]);
  assert.deepEqual(moveComponents(v, [0, 0, 0], [100, -3.3, 3], [0, 1], 0.1, 0.01, 18), [18, -3.3, 3]);
});
