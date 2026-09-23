import { test } from 'node:test';
import assert from 'node:assert/strict';
import katex from '../static/vendor/katex/katex.mjs';
import { evaluate } from '../static/graph/lang.js';
import * as la from '../static/graph/linalg.js';
import {
  planeInCube, clipLine3, meetPlanes, meetLines, latticeSegments, stepFrame, elimFrame, gsData,
  coordsOf, fracTex, polyTex, niceDir, shades, luminance, mixHex, equationParts, readoutTex, fourLayout, needRhs,
} from '../static/graph/features/systems.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const nearArr = (a, b, eps = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => near(x, b[i], eps));
};
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const run = (lines) => evaluate(lines);
const last = (lines) => run(lines).at(-1);
const valueOf = (lines) => {
  const r = last(lines);
  assert.equal(r.error, null, r.error);
  return r.value;
};
const renders = (tex) => katex.renderToString(tex, { throwOnError: true });

// ---------------------------------------------------------------- geometry

test('a plane cut by the cube', () => {
  const sq = planeInCube([0, 0, 1], 0, 2);
  assert.equal(sq.length, 4);
  const hex = planeInCube([1, 1, 1], 0, 1);
  assert.equal(hex.length, 6);
  for (const [n, c, B] of [[[1, 2, 3], 4, 6], [[1, 1, 1], 0, 1], [[0, 1, 0], -2, 3], [[2, -1, 0.5], 1, 6]]) {
    const poly = planeInCube(n, c, B);
    assert.ok(poly.length >= 3);
    for (const p of poly) {
      near(dot(n, p), c, 1e-9);
      for (const x of p) assert.ok(Math.abs(x) <= B + 1e-9);
    }
  }
  assert.equal(planeInCube([1, 0, 0], 7, 6), null); // misses the cube
  assert.equal(planeInCube([0, 0, 0], 1, 6), null);
});

test('polygon vertices go around in order (a convex fan works)', () => {
  const poly = planeInCube([1, 2, 3], 1, 6);
  const n = [1, 2, 3];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const signs = poly.map((p, i) => Math.sign(dot(cross(sub(p, poly[0]), sub(poly[(i + 1) % poly.length], poly[0])), n)));
  assert.ok(signs.every((s) => s >= 0) || signs.every((s) => s <= 0));
});

test('line clipping, plane and line intersections', () => {
  assert.deepEqual(clipLine3([0, 0, 0], [1, 0, 0], 6), [-6, 6]);
  assert.equal(clipLine3([0, 7, 0], [1, 0, 0], 6), null);
  const L = meetPlanes([1, 0, 0], 1, [0, 1, 0], 2);
  nearArr(L.p, [1, 2, 0]);
  nearArr(L.d, [0, 0, 1]);
  assert.equal(meetPlanes([1, 1, 0], 1, [2, 2, 0], 5), null);
  nearArr(meetLines([1, 2], 5, [3, -1], 1), [1, 2, 0]);
  assert.equal(meetLines([1, 1], 1, [2, 2], 3), null);
});

test('lattice segments stay in the cube and on lattice lines', () => {
  const segs = latticeSegments([[1, 0, 0], [1, 1, 0]], 3);
  assert.ok(segs.length > 5);
  for (const [p, q] of segs) for (const x of [...p, ...q]) assert.ok(Math.abs(x) <= 3 + 1e-9);
  const segs3 = latticeSegments([[1, 0, 0], [0, 1, 0], [1, 1, 1]], 6);
  assert.equal(segs3.length, 75);
});

// ---------------------------------------------------------------- formatting

test('fractions, polynomials, nice directions, colours', () => {
  assert.equal(fracTex(2), '2');
  assert.equal(fracTex(-0), '0');
  assert.equal(fracTex(-1 / 3), '-\\tfrac{1}{3}');
  assert.equal(fracTex(2.5), '\\tfrac{5}{2}');
  assert.equal(fracTex(Math.SQRT1_2), '0.7071');
  assert.equal(fracTex(Math.SQRT1_2, 3), '0.707');
  assert.equal(polyTex([1, -5, -2]), '\\lambda^{2} - 5\\lambda - 2');
  assert.equal(polyTex([1, 0, 1]), '\\lambda^{2} + 1');
  assert.equal(polyTex([1, -3, 3, -1]), '\\lambda^{3} - 3\\lambda^{2} + 3\\lambda - 1');
  assert.deepEqual(niceDir([Math.SQRT1_2, Math.SQRT1_2, 0]), [1, 1, 0]);
  assert.deepEqual(niceDir([2 / Math.sqrt(13), 3 / Math.sqrt(13)]), [2, 3]);
  assert.deepEqual(equationParts([1, -2, 0], 5), ['x - 2y', '5']);
  assert.deepEqual(equationParts([0, 0.5, -1], -1), ['\\tfrac{1}{2}y - z', '-1']);
  const t = shades('#e05a4f', 3, 'dark');
  assert.equal(new Set(t).size, 3);
  for (const x of [...t, ...shades('#4a90e2', 3, 'light')]) assert.match(x, /^#[0-9a-f]{6}$/);
  assert.equal(mixHex('#000000', '#ffffff', 0.5), '#808080');
  // every shade of every palette colour reads on its theme (WCAG contrast >= 3.5)
  for (const theme of ['dark', 'light']) {
    for (const base of ['#e05a4f', '#4a90e2', '#43b05c', '#9b6ade', '#f5a623', '#26b5b5', '#e056a0', '#a1887f']) {
      for (const c of shades(base, 6, theme)) {
        const y = luminance(c), contrast = theme === 'dark' ? (y + 0.05) / 0.0655 : 1.01 / (y + 0.05);
        assert.ok(contrast >= 3.5, `${base} -> ${c} on ${theme}: ${contrast.toFixed(2)}`);
      }
    }
  }
});

test('right-hand sides', () => {
  assert.deepEqual(needRhs({ type: 'vec', v: [5, 1, 0] }, 2, 'f'), [5, 1]);
  assert.deepEqual(needRhs(4, 1, 'f'), [4]);
  assert.throws(() => needRhs({ type: 'vec', v: [5, 1, 2] }, 2, 'f'), /2 rows but b has 3/);
  assert.throws(() => needRhs(3, 2, 'f'), /single number/);
});

// ---------------------------------------------------------------- elimination

test('every elimination frame keeps the solution on every plane', () => {
  const A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], b = [8, -11, -3], x = [2, 3, -1];
  const steps = la.eliminationSteps(A.map((r, i) => [...r, b[i]]));
  const N = steps.length - 1;
  for (let k = 0; k <= N; k += 0.05) {
    const fr = elimFrame(steps, k);
    for (const row of fr.rows) near(dot(row.slice(0, 3), x), row[3], 1e-9);
  }
  const end = elimFrame(steps, N);
  assert.equal(end.pending, false);
  nearArr(end.rows.map((r) => r[3]), x);
});

test('elimination frames interpolate only row additions', () => {
  const steps = la.eliminationSteps([[1, 2, 5], [3, -1, 1]]);
  const fr0 = elimFrame(steps, 0);
  assert.equal(fr0.op, 'start');
  assert.equal(fr0.pending, false);
  const add = steps.findIndex((s) => s.op === 'add');
  const half = elimFrame(steps, add - 0.5);
  assert.equal(half.pending, true);
  assert.equal(half.op, 'add');
  const [i] = steps[add].rows;
  nearArr(half.rows[i], steps[add - 1].M[i].map((x, j) => (x + steps[add].M[i][j]) / 2));
  assert.deepEqual(stepFrame(4, 9), { s: 4, f: 0, S: 4 });
  assert.deepEqual(stepFrame(4, -1), { s: 0, f: 0, S: 4 });
  assert.deepEqual(stepFrame(4, null), { s: 4, f: 0, S: 4 });
  const sf = stepFrame(4, 1.25);
  assert.equal(sf.s, 1);
  near(sf.f, 0.25);
});

test('swaps and scalings do not move the planes', () => {
  const steps = la.eliminationSteps([[0, 1, 2], [2, 4, 6]]);
  const swap = steps.findIndex((s) => s.op === 'swap');
  assert.ok(swap > 0);
  const fr = elimFrame(steps, swap - 0.4);
  assert.deepEqual(fr.swap, steps[swap].rows);
  assert.deepEqual(fr.rows, steps[swap - 1].M);
  const scale = steps.findIndex((s) => s.op === 'scale');
  assert.deepEqual(elimFrame(steps, scale - 0.5).rows, steps[scale - 1].M);
});

// ---------------------------------------------------------------- language

test('rowpicture and colpicture values', () => {
  const r = valueOf(['A = [[1, 2], [3, -1]]', 'b = (5, 1)', 'rowpicture(A, b)']);
  assert.equal(r.type, 'sy-rows');
  assert.equal(r.n, 2);
  assert.deepEqual(r.sol, { kind: 'unique', x: [1, 2] });
  const c = valueOf(['A = [[1, 2], [3, -1]]', 'b = (5, 1)', 'colpicture(A, b)']);
  assert.equal(c.exact, true);
  assert.deepEqual(c.x, [1, 2]);
  const bad = valueOf(['A = [[1, 0], [0, 1], [1, 1]]', 'b = (1, 1, 3)', 'colpicture(A, b)']);
  assert.equal(bad.exact, false);
  nearArr(bad.p, [4 / 3, 4 / 3, 8 / 3]);
  assert.match(last(['A = [[1, 2, 3, 4]]', 'rowpicture(A, 1)']).error, /2 unknowns .* 4 columns/);
  assert.match(last(['A = [[1, 2], [3, 4]]', 'rowpicture(A, (1, 2, 3))']).error, /2 rows but b has 3/);
  assert.match(last(['rowpicture((1, 2), (1, 2))']).error, /needs a matrix/);
});

test('eliminate follows the k slider', () => {
  const lines = ['A = [[1, 2], [3, -1]]', 'b = (5, 1)', 'k = 1.5', 'eliminate(A, b, k)'];
  const v = valueOf(lines);
  assert.equal(v.frame.s, 1);
  near(v.frame.f, 0.5);
  assert.deepEqual(v.sol.x, [1, 2]);
  const end = valueOf(['A = [[1, 2], [3, -1]]', 'b = (5, 1)', 'eliminate(A, b)']);
  assert.equal(end.frame.s, end.frame.N);
  assert.deepEqual(end.steps.at(-1).M, [[1, 0, 1], [0, 1, 2]]);
  assert.match(last(['A = [[1, 2], [3, -1]]', 'eliminate(A, (5, 1), (1, 1))']).error, /number for the step/);
});

test('lstsq projects b onto the column space', () => {
  const v = valueOf(['A = [[1, 0], [1, 1], [1, 2]]', 'b = (6, 0, 0)', 'lstsq(A, b)']);
  nearArr(v.x, [5, -3]);
  nearArr(v.p, [5, 2, -1]);
  near(dot(v.e, [1, 1, 1]), 0);
  near(dot(v.e, [0, 1, 2]), 0);
});

test('subspaces: dimensions and orthogonality', () => {
  const v = valueOf(['A = [[1, 2, 3], [2, 4, 6]]', 'subspaces(A)']);
  assert.equal(v.r, 1);
  assert.equal(v.row.length, 1);
  assert.equal(v.nul.length, 2);
  assert.equal(v.col.length, 1);
  assert.equal(v.lnul.length, 1);
  for (const r of v.row) for (const n of v.nul) near(dot(r, n), 0);
  for (const c of v.col) for (const n of v.lnul) near(dot(c, n), 0);
  assert.match(last(['subspaces([[1, 2, 3, 4]])']).error, /up to 3×3/);
  const lay = fourLayout(null, 6);
  assert.ok(lay.every((g) => g.alpha === 1 && g.apart));
  const mid = fourLayout(0.25, 6);
  near(mid[0].alpha, 0.75);
  near(mid[1].alpha, 0.25);
});

test('gramschmidt builds an orthonormal basis', () => {
  const v = valueOf(['u = (1, 1, 0)', 'v = (1, 0, 1)', 'w = (0, 1, 1)', 'gramschmidt(u, v, w)']);
  assert.equal(v.frame.s, 4);
  const qs = v.qs;
  for (let i = 0; i < 3; i++) {
    near(dot(qs[i], qs[i]), 1);
    for (let j = 0; j < i; j++) near(dot(qs[i], qs[j]), 0);
  }
  nearArr(v.ws[1], [0.5, -0.5, 1]);
  near(v.coefs[1][0], 0.5);
  const d = gsData([[1, 0, 0], [2, 0, 0], [0, 1, 0]]);
  assert.equal(d.qs[1], null);
  nearArr(d.ws[2], [0, 1, 0]);
  const k = valueOf(['gramschmidt((1, 0), (1, 1), 1.5)']);
  assert.equal(k.frame.s, 1);
  assert.equal(k.p, 2);
  assert.match(last(['gramschmidt((1, 0, 0), 2)']).error, /2 or 3 vectors/);
});

test('basis and coords', () => {
  const b = valueOf(['basis((1, 0), (1, 1))']);
  assert.equal(b.type, 'sy-lattice');
  assert.match(last(['basis((1, 0), (2, 0))']).error, /dependent/);
  const lines = ['b1 = (1, 0)', 'b2 = (1, 1)', 'B = matrix(b1, b2)', 'v = (3, 2)', 'coords(v, B)'];
  const c = valueOf(lines);
  assert.equal(c.type, 'vec');
  assert.deepEqual(c.v, [1, 2, 0]);
  assert.deepEqual(valueOf(['c = coords((3, 2), basis((1, 0), (1, 1)))', 'matrix((1, 0), (1, 1)) c']).v, [3, 2, 0]);
  const cols = [[1, 1, 0], [0, 1, 1], [1, 0, 1]], v = [2, 3, 4], x = coordsOf(v, cols);
  nearArr(cols.reduce((s, col, j) => s.map((y, i) => y + x[j] * col[i]), [0, 0, 0]), v);
  assert.throws(() => coordsOf([1, 2, 0], [[1, 0, 1], [0, 1, 1]]), /not in the plane/);
  nearArr(coordsOf([1, 2, 3], [[1, 0, 1], [0, 1, 1]]), [1, 2]);
  assert.throws(() => coordsOf([1, 2, 1], [[1, 0, 0], [0, 1, 0]]), /z component/);
  assert.throws(() => coordsOf([1, 2, 1], [[1, 0, 0], [2, 0, 0]]), /dependent/);
});

test('readouts: rref, rank, tr, spaces, eig, svd, qr, lu, charpoly', () => {
  const A = 'A = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]';
  const R = valueOf([A, 'rref(A)']);
  assert.equal(R.type, 'mat');
  assert.deepEqual(R.sy.pivots, [0, 1]);
  assert.equal(valueOf([A, 'rank(A)']), 2);
  assert.equal(valueOf(['tr([[1, 2], [3, 4]])']), 5);
  const N = valueOf([A, 'nullspace(A)']);
  assert.equal(N.type, 'span');
  assert.deepEqual(N.vecs, [[-1, -1, 1]]);
  assert.equal(valueOf([A, 'colspace(A)']).vecs.length, 2);
  assert.equal(valueOf([A, 'rowspace(A)']).vecs.length, 2);
  assert.equal(valueOf([A, 'leftnull(A)']).vecs.length, 1);
  const wide = valueOf(['nullspace([[1, 2, 3, 4]])']);
  assert.equal(wide.type, 'sy-space');
  assert.equal(valueOf(['eig([[2, 0], [0, 3]])']).type, 'sy-eig');
  assert.equal(valueOf(['svd([[3, 0], [0, 2]])']).type, 'sy-svd');
  assert.equal(valueOf(['qr([[1, 1], [0, 1]])']).type, 'sy-qr');
  assert.equal(valueOf(['lu([[2, 1], [4, 5]])']).type, 'sy-lu');
  assert.deepEqual(valueOf(['charpoly([[1, 2], [3, 4]])']).coeffs, [1, -5, -2]);
  assert.match(last(['eig([[1, 2, 3], [4, 5, 6]])']).error, /square/);
});

test('solve: unique vector, affine line / plane, readable no-solution', () => {
  const u = valueOf(['solve([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], (8, -11, -3))']);
  assert.equal(u.type, 'vec');
  assert.deepEqual(u.v, [2, 3, -1]);
  const line = valueOf(['solve([[1, 1, 0], [0, 1, 1]], (2, 3))']);
  assert.equal(line.type, 'sy-affine');
  assert.equal(line.dirs.length, 1);
  const plane = valueOf(['solve([[1, 2, 3]], 6)']);
  assert.equal(plane.dirs.length, 2);
  const none = valueOf(['solve([[1, 1], [1, 1]], (1, 2))']);
  assert.equal(none.type, 'sy-text');
  assert.match(none.tex, /no solution/);
  const r = run(['solve([[1, 1], [1, 1]], (1, 2))'])[0];
  assert.equal(r.error, null);
});

test('every readout is valid KaTeX, with and without colours', () => {
  const rows = [
    'A = [[1, 2, 1], [3, 8, 1], [0, 4, 1]]', 'b = (2, 12, 2)', 'k = 2.5', 'M = [[1, 2], [3, -1]]', 'c = (5, 1)',
    'rowpicture(A, b)', 'rowpicture(M, c)', 'rowpicture([[1, 1, 1], [1, 1, 1], [1, 2, 3]], (1, 2, 3))',
    'rowpicture([[1, 2, 3]], 6)', 'colpicture(A, b)', 'colpicture([[1, 0], [0, 1], [1, 1]], (1, 1, 3))',
    'eliminate(A, b, k)', 'eliminate(A, b, 0)', 'eliminate(A, b)', 'eliminate(M, c, 1.3)',
    'eliminate([[0, 1], [2, 4]], (1, 2), 0.5)', 'lstsq([[1, 0], [1, 1], [1, 2]], (6, 0, 0))',
    'subspaces(A)', 'subspaces([[1, 2, 3], [2, 4, 6]], 0.3)', 'gramschmidt((1, 1, 0), (1, 0, 1), (0, 1, 1), 0)',
    'gramschmidt((1, 1, 0), (1, 0, 1), (0, 1, 1), 2)', 'gramschmidt((1, 1, 0), (1, 0, 1), (0, 1, 1), 2.5)',
    'gramschmidt((1, 0), (2, 0), 2)', 'gramschmidt((1, 1), (1, 0))', 'basis((1, 0), (1, 1))',
    'basis((1, 0, 0), (1, 1, 0), (0, 1, 1))', 'basis((1, 0, 1), (0, 1, 1))', 'coords((3, 2), matrix((1, 0), (1, 1)))',
    'rref(A)', 'nullspace([[1, 2], [2, 4]])', 'colspace(A)', 'rowspace([[1, 2, 3], [2, 4, 6]])', 'leftnull(M)',
    'nullspace([[1, 2, 3, 4]])', 'eig([[2, 1], [1, 2]])', 'eig([[0, -1], [1, 0]])', 'eig([[2, 1, 0], [0, 2, 0], [0, 0, 3]])',
    'eig([[1, 2, 0], [-2, 1, 0], [0, 0, 3]])', 'svd(A)', 'svd([[1, 2, 3], [4, 5, 6]])', 'qr(A)', 'lu(A)',
    'lu([[0, 1], [1, 1]])', 'charpoly(A)', 'solve(A, b)', 'solve([[1, 1, 0], [0, 1, 1]], (2, 3))',
    'solve([[1, 1], [1, 1]], (1, 2))', 'solve([[1, 2, 3, 4], [0, 1, 0, 1]], (1, 2))',
  ];
  const results = run(rows);
  results.forEach((r, i) => {
    if (i < 5) return;
    assert.equal(r.error, null, `${rows[i]}: ${r.error}`);
    for (const opts of [{}, { color: '#e05a4f', theme: 'dark' }, { color: '#4a90e2', theme: 'light', args: [{ name: 'A' }, { name: 'b' }] }]) {
      const tex = readoutTex(r.value, opts);
      assert.ok(tex, rows[i]);
      assert.doesNotThrow(() => renders(tex), `${rows[i]}: ${tex}`);
    }
  });
});

test('readout content', () => {
  const t = (src) => readoutTex(valueOf([src]));
  assert.match(t('rowpicture([[1, 2], [3, -1]], (5, 1))'), /\(x, y\) = \(1, 2\)/);
  assert.match(t('rowpicture([[1, 1], [1, 1]], (1, 2))'), /no common point/);
  assert.match(t('eliminate([[1, 2], [3, -1]], (5, 1), 0)'), /step \} 0\/3/);
  assert.match(t('rowpicture([[1, 1, 1], [1, -1, 0], [2, 0, 1]], (1, 1, 2))'), /a line:\}.*\\\\ &\(x, y, z\) = \(1, 0, 0\) \+ t/);
  assert.match(t('eliminate([[1, 2], [3, -1]], (5, 1), 1.5)'), /step \} 1\\to 2\/3/);
  assert.match(t('eliminate([[1, 2], [3, -1]], (5, 1))'), /RREF/);
  assert.match(t('charpoly([[1, 2], [3, 4]])'), /\\lambda\^\{2\} - 5\\lambda - 2/);
  assert.match(t('eig([[0, -1], [1, 0]])'), /\\pm i/);
  assert.match(t('nullspace([[1, 2], [2, 4]])'), /span\}\\\{\(-2, 1\)\\\}/);
  assert.match(t('nullspace([[1, 0], [0, 1]])'), /\\vec 0/);
  assert.match(t('coords((3, 2), matrix((1, 0), (1, 1)))'), /\[\\vec v\]_\{B\} = \(1, 2\)/);
  assert.match(t('subspaces([[1, 2, 3], [2, 4, 6]])'), /rank \} 1/);
});

test('eig readout drops a zero real part', () => {
  const tex = readoutTex({ type: 'sy-eig', ...la.eig([[0, -1, 0], [1, 0, 0], [0, 0, 2]]) });
  assert.ok(tex.includes('\\lambda = \\pm i:'), tex);
  assert.ok(readoutTex({ type: 'sy-eig', ...la.eig([[1, -2], [2, 1]]) }).includes('\\lambda = 1 \\pm 2i:'));
});
