import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate } from '../static/graph/lang.js';
import {
  explainParts, clipLine, planeInBox, niceDirection, toFlat, intersectFlats, distanceTo, anyPerp,
  perpendicularPairs, pushTrail, hexToHsl, hslToHex, shades, luminance, sumTex, planeTex, componentsTex,
  readoutTex, tupleTex, TARGET_TOL,
} from '../static/graph/features/combos.js';

const last = (...lines) => evaluate(lines).at(-1);
const val = (...lines) => {
  const r = last(...lines);
  assert.equal(r.error, null, `unexpected error: ${r.error}`);
  return r.value;
};
const err = (...lines) => {
  const r = last(...lines);
  assert.notEqual(r.error, null, `expected an error, got ${JSON.stringify(r.value)}`);
  return r.error;
};
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const nearArr = (a, b, eps = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => near(x, b[i], eps));
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => a.map((x, i) => x - b[i]);

// ---------------------------------------------------------------- explain / chain

test('explain splits A v into scaled columns', () => {
  const v = val('A = [[1,1,0],[0,1,0],[0,0,1]]', 'v = (2, 1, 0)', 'explain(A, v)');
  assert.equal(v.type, 'vec');
  nearArr(v.v, [3, 1, 0]);
  assert.equal(v.combo.mode, 'explain');
  assert.deepEqual(v.combo.cols, [[1, 0, 0], [1, 1, 0], [0, 0, 1]]);
  assert.deepEqual(v.combo.coeffs, [2, 1, 0]);
});

test('explain readout shows the actual numbers', () => {
  const v = val('A = [[1,1,0],[0,1,0],[0,0,1]]', 'explain(A, (2, 1, 0))');
  assert.equal(readoutTex(v), '= 2\\cdot (1, 0, 0) + 1\\cdot (1, 1, 0) + 0\\cdot (0, 0, 1) = (3, 1, 0)');
  const coloured = readoutTex(v, '#e05a4f');
  assert.match(coloured, /\\textcolor\{#[0-9a-f]{6}\}\{2\\cdot \(1, 0, 0\)\}/);
});

test('explain handles 2x2 matrices and negative coefficients', () => {
  const v = val('M = [[2, -1], [1, 1]]', 'explain(M, (1.5, -2))');
  nearArr(v.v, [5, -0.5, 0]);
  assert.equal(v.combo.dim, 2);
  assert.equal(readoutTex(v), '= 1.5\\cdot (2, 1) - 2\\cdot (-1, 1) = (5, -0.5)');
});

test('explain errors are readable', () => {
  assert.match(err('A = [[1,0],[0,1]]', 'explain((1, 2), A)'), /matrix first/);
  assert.match(err('explain((1, 2), (3, 4))'), /needs a matrix/);
  assert.match(err('A = [[1,0],[0,1]]', 'explain(A, (1, 2, 3))'), /nonzero z/);
  assert.match(err('A = [[1,0],[0,1]]', 'explain(A, 2)'), /vector after the matrix/);
  assert.throws(() => explainParts({ type: 'mat', m: [[1, 2, 3, 4]] }, { type: 'vec', v: [1, 0, 0] }), /2 or 3 rows/);
});

test('tagged values still compose like plain vectors', () => {
  const v = val('A = [[1,1,0],[0,1,0],[0,0,1]]', 'w = explain(A, (2, 1, 0))', 'w + (1, 1, 1)');
  assert.deepEqual(v, { type: 'vec', v: [4, 2, 1] });
  nearArr(val('u = (1, 2, 0)', 's = shadow(u, (1, 0, 0))', 'u - s').v, [0, 2, 0]);
});

test('chain sums its vectors and keeps the parts', () => {
  const v = val('u = (1, 2, 0)', 'v = (-1, 1, 2)', 'chain(u, v, (2, 0, 1))');
  nearArr(v.v, [2, 3, 3]);
  assert.deepEqual(v.combo.parts, [[1, 2, 0], [-1, 1, 2], [2, 0, 1]]);
  assert.equal(readoutTex(v), '= (1, 2, 0) + (-1, 1, 2) + (2, 0, 1) = (2, 3, 3)');
  assert.match(err('chain(point(1, 2, 3), (1, 0, 0))'), /chain needs vectors, got a point/);
});

// ---------------------------------------------------------------- trail / target

test('trail passes the value through', () => {
  const v = val('a = 2', 'trail(a (1, 1, 0))');
  assert.equal(v.type, 'vec');
  nearArr(v.v, [2, 2, 0]);
  assert.equal(val('trail(point(1, 2, 3))').type, 'point');
  assert.match(err('trail(3)'), /trail needs a vector or point/);
});

test('pushTrail dedupes, caps, and resets when the text changes', () => {
  const m = new Map();
  pushTrail(m, 'r1', 'trail(a u)', [0, 0, 0]);
  pushTrail(m, 'r1', 'trail(a u)', [0, 0, 0]);
  pushTrail(m, 'r1', 'trail(a u)', [1, 0, 0]);
  assert.equal(m.get('r1').pts.length, 2);
  for (let i = 0; i < 10; i++) pushTrail(m, 'r1', 'trail(a u)', [i + 2, 0, 0], 5);
  assert.equal(m.get('r1').pts.length, 5);
  assert.deepEqual(m.get('r1').pts.at(-1), [11, 0, 0]);
  pushTrail(m, 'r1', 'trail(b u)', [7, 7, 7]);
  assert.deepEqual(m.get('r1').pts, [[7, 7, 7]]);
});

test('target reports hits within the tolerance', () => {
  const ctx = ['u = (1, 2, 0)', 'v = (-1, 1, 2)', 'b = (1, 5, 2)'];
  const hit = val(...ctx, 'target(b, 2u + v)');
  assert.equal(hit.combo.hit, true);
  nearArr(hit.v, [1, 5, 2]);
  const miss = val(...ctx, 'target(b, 2u + 0.5v)');
  assert.equal(miss.combo.hit, false);
  near(miss.combo.miss, Math.hypot(0.5, 0.5, 1));
  const close = val(...ctx, 'target(b, 2.02u + v)');
  assert.ok(close.combo.miss < TARGET_TOL && close.combo.hit);
  assert.equal(val(...ctx, 'target(b, 2.02u + v, 0.01)').combo.hit, false);
  assert.match(err(...ctx, 'target(b, u, -1)'), /tolerance/);
  assert.match(readoutTex(hit), /on target/);
  assert.match(readoutTex(miss), /off by \} 1\.2247$/);
});

// ---------------------------------------------------------------- geometry helpers

test('shadow is the projection, with the scalar in the readout', () => {
  const v = val('shadow((3, 1, 0), (2, 0, 0))');
  nearArr(v.v, [3, 0, 0]);
  assert.equal(v.combo.uv, 6);
  assert.equal(v.combo.vv, 4);
  assert.equal(readoutTex(v), '= \\tfrac{6}{4}\\,(2, 0, 0) = (3, 0, 0)');
  assert.match(err('shadow((1, 2, 3), (0, 0, 0))'), /zero vector/);
});

test('components and crossview', () => {
  const c = val('components((3, -1, 2.5))');
  nearArr(c.v, [3, -1, 2.5]);
  assert.equal(readoutTex(c), '= (3, -1, 2.5) = 3\\,\\hat\\imath - \\hat\\jmath + 2.5\\,\\hat k');
  assert.equal(componentsTex([0, 0, 0]), '\\vec 0');
  const x = val('crossview((2, 0, 0), (1, 2, 0))');
  nearArr(x.v, [0, 0, 4]);
  near(x.combo.area, 4);
  assert.equal(readoutTex(x), '= (0, 0, 4)\\quad \\text{area } 4');
});

test('arc measures the angle', () => {
  const a = val('arc((1, 0, 0), (1, 1, 0))');
  assert.equal(a.type, 'cb-arc');
  near(a.theta, Math.PI / 4);
  assert.match(readoutTex(a), /\\theta = 45\^\\circ/);
  near(val('arc((1, 0, 0), (-3, 0, 0))').theta, Math.PI);
  assert.match(err('arc((0, 0, 0), (1, 0, 0))'), /nonzero/);
  assert.match(err('arc((1, 0, 0), 2)'), /arc needs vectors/);
});

test('line and plane3', () => {
  const L = val('line(point(1, 0, 0), point(0, 1, 0))');
  assert.equal(L.type, 'cb-line');
  assert.deepEqual(L.d, [-1, 1, 0]);
  assert.equal(readoutTex(L), '\\vec r(t) = (1, 0, 0) + t\\,(-1, 1, 0)');
  assert.match(err('line((1, 2, 3), (1, 2, 3))'), /two different points/);
  const P = val('plane3((2, 0, 0), (0, 2, 0), (0, 0, 2))');
  assert.deepEqual(P.n, [1, 1, 1]);
  assert.equal(P.c, 2);
  assert.equal(readoutTex(P), 'x + y + z = 2');
  assert.match(err('plane3((0, 0, 0), (1, 1, 1), (2, 2, 2))'), /one line/);
  assert.match(err('plane3((0, 0, 0), (1, 1, 1), 2)'), /three points/);
});

test('planeTex and sumTex formatting', () => {
  assert.equal(planeTex([3, -1, 0], 6), '3x - y = 6');
  assert.equal(planeTex([0, 0, -2], -1), '-2z = -1');
  assert.equal(sumTex([{ k: -2, body: 'a' }, { k: 1.5, body: 'b' }]), '-2\\cdot a + 1.5\\cdot b');
  assert.equal(tupleTex([1, 2, 3], 2), '(1, 2)');
});

test('niceDirection reduces integer vectors and fixes the sign', () => {
  assert.deepEqual(niceDirection([-2, 4, 0]).d, [1, -2, 0]);
  assert.deepEqual(niceDirection([0, -3, 6], [9]), { d: [0, 1, -2], extra: [-3] });
  assert.deepEqual(niceDirection([0.5, -1, 0]).d, [0.5, -1, 0]);
});

// ---------------------------------------------------------------- intersect / distance

test('plane ∩ plane is a line lying in both', () => {
  const L = val('A = plane3((1, 0, 0), (0, 1, 0), (0, 0, 1))', 'intersect(A, plane((0, 0, 1)))');
  assert.equal(L.type, 'cb-line');
  assert.deepEqual(L.d, [1, -1, 0]);
  for (const t of [-2, 0, 3]) {
    const p = L.p.map((x, i) => x + t * L.d[i]);
    near(p[0] + p[1] + p[2], 1);
    near(p[2], 0);
  }
  assert.match(err('intersect(plane((0, 0, 1)), plane((0, 0, 2)))'), /same plane/);
  assert.match(err('intersect(plane((0, 0, 1)), plane3((0,0,1), (1,0,1), (0,1,1)))'), /parallel/);
});

test('line ∩ plane is a point', () => {
  const p = val('intersect(line(point(0, 0, 3), point(1, 1, 2)), plane((0, 0, 1)))');
  assert.equal(p.type, 'point');
  nearArr(p.v, [3, 3, 0]);
  nearArr(val('intersect(plane((0, 0, 1)), line((0, 0, 3), (1, 1, 2)))').v, [3, 3, 0]);
  assert.match(err('intersect(line((0, 0, 1), (1, 0, 1)), plane((0, 0, 1)))'), /parallel to the plane/);
  assert.match(err('intersect(line((0, 0, 0), (1, 0, 0)), plane((0, 0, 1)))'), /lies in the plane/);
});

test('line ∩ line is a point, or an error for parallel and skew lines', () => {
  const p = val('intersect(line((0, 0, 0), (2, 2, 0)), line((0, 2, 0), (2, 0, 0)))');
  nearArr(p.v, [1, 1, 0]);
  assert.match(err('intersect(line((0, 0, 0), (1, 0, 0)), line((0, 1, 0), (1, 1, 0)))'), /parallel/);
  assert.match(err('intersect(line((0, 0, 0), (1, 0, 0)), line((2, 0, 0), (5, 0, 0)))'), /same line/);
  assert.match(err('intersect(line((0, 0, 0), (1, 0, 0)), line((0, 1, 1), (0, 2, 1)))'), /skew.*by 1$/);
});

test('intersect accepts spans and rejects other values', () => {
  nearArr(val('intersect(span((1, 0, 0)), plane3((2, 0, 0), (2, 1, 0), (2, 0, 1)))').v, [2, 0, 0]);
  const L = val('intersect(span((1, 0, 0), (0, 1, 0)), span((1, 0, 0), (0, 0, 1)))');
  assert.deepEqual(L.d, [1, 0, 0]);
  assert.match(err('intersect((1, 2, 3), plane((0, 0, 1)))'), /lines or planes, got a vector/);
  assert.match(err('intersect(span((1,0,0), (0,1,0), (0,0,1)), plane((0, 0, 1)))'), /all of R\^3/);
});

test('distance to a plane, a line and a point', () => {
  const d = val('distance(point(3, 3, 3), plane3((2, 0, 0), (0, 2, 0), (0, 0, 2)))');
  assert.equal(d.type, 'cb-dist');
  near(d.d, 7 / Math.sqrt(3));
  near(d.foot[0] + d.foot[1] + d.foot[2], 2);
  const dl = val('distance(point(2, -2, 2), line((0, 0, 0), (1, 1, 0)))');
  near(dl.d, Math.sqrt(12));
  near(dot(sub(dl.p, dl.foot), [1, 1, 0]), 0);
  near(val('distance(line((0, 0, 0), (1, 1, 0)), (2, -2, 2))').d, Math.sqrt(12));
  near(val('distance((1, 1, 1), (4, 5, 1))').d, 5);
  assert.equal(readoutTex(dl), 'd = 3.4641');
  assert.match(err('distance(2, plane((0, 0, 1)))'), /needs a point/);
});

test('toFlat, distanceTo and anyPerp helpers', () => {
  assert.deepEqual(toFlat({ type: 'plane', normal: [0, 0, 1] }), { kind: 'plane', n: [0, 0, 1], c: 0 });
  const r = distanceTo([0, 0, 5], { type: 'plane', normal: [0, 0, 2] });
  near(r.d, 5);
  nearArr(r.foot, [0, 0, 0]);
  near(dot(r.along, [0, 0, 1]), 0);
  for (const a of [[1, 0, 0], [0, 0, 3], [1, 2, 3]]) {
    const q = anyPerp(a);
    near(dot(q, a), 0);
    near(Math.hypot(...q), 1);
  }
  assert.throws(() => intersectFlats({ kind: 'line', p: [0, 0, 0], d: [1, 0, 0] }, { kind: 'line', p: [0, 0, 1], d: [0, 1, 0] }), /skew/);
});

// ---------------------------------------------------------------- drawing helpers

test('perpendicularPairs finds right angles at shared origins only', () => {
  const O = [0, 0, 0];
  const pairs = perpendicularPairs([
    { o: O, v: [2, 0, 0] }, { o: O, v: [0, 3, 0] }, { o: O, v: [1, 1, 0] },
    { o: [1, 0, 0], v: [0, 0, 1] }, // perpendicular to the first, but starts elsewhere
    { o: O, v: [4, 0, 0] },         // same direction as the first: no duplicate marker
    { o: O, v: [0, 0, 0] },
  ]);
  assert.equal(pairs.length, 1);
  assert.deepEqual(pairs[0].a, [1, 0, 0]);
  assert.deepEqual(pairs[0].b, [0, 1, 0]);
  assert.equal(pairs[0].size, 2);
  assert.equal(perpendicularPairs([{ o: O, v: [1, 0, 0] }, { o: O, v: [0.01, 1, 0] }]).length, 0);
  assert.equal(perpendicularPairs([{ o: O, v: [1, 0, 0] }, { o: O, v: [0.001, 1, 0] }]).length, 1);
});

test('clipLine and planeInBox stay inside the box', () => {
  const t = clipLine([0, 0, 0], [1, 2, 0], 6);
  nearArr(t, [-3, 3]);
  assert.equal(clipLine([0, 0, 9], [1, 0, 0], 6), null);
  assert.equal(planeInBox([0, 0, 1], 0, 6).length, 4);
  const hex = planeInBox([1, 1, 1], 0, 6);
  assert.equal(hex.length, 6);
  for (const p of hex) {
    near(p[0] + p[1] + p[2], 0);
    assert.ok(p.every((x) => Math.abs(x) <= 6 + 1e-9));
  }
  assert.deepEqual(planeInBox([0, 0, 1], 10, 6), []);
});

test('shades are distinct and readable', () => {
  const [h, s, l] = hexToHsl('#e05a4f');
  assert.equal(hslToHex(h, s, l), '#e05a4f');
  for (const theme of ['dark', 'light']) {
    for (const base of ['#e05a4f', '#4a90e2', '#43b05c', '#9b6ade', '#f5a623', '#26b5b5', '#e056a0', '#a1887f']) {
      const out = shades(base, 3, theme);
      assert.equal(new Set([base, ...out]).size, 4);
      for (const c of out) {
        const L = hexToHsl(c)[2];
        assert.ok(theme === 'dark' ? L >= 0.49 : L <= 0.63, `${c} on ${theme}`);
        // WCAG contrast with the theme background (#1d2327 / #fbfbf8) of at least 3.5
        const y = luminance(c), contrast = theme === 'dark' ? (y + 0.05) / 0.0655 : 1.01 / (y + 0.05);
        assert.ok(contrast >= 3.5, `${c} on ${theme}: contrast ${contrast.toFixed(2)}`);
      }
    }
  }
});
