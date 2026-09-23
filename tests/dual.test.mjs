import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate } from '../static/graph/lang.js';
import {
  mapInfo, mapValue, applyMap, fitsDomain, mapItem, gridSegments, mapSegments, clipSegment,
  needExtent, dualItems, linkSource, matrixName, subspaceTex, readout, niceStep, gridRange, KER_COLOR, IMG_COLOR, labelTips,
} from '../static/graph/features/dual.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const nearArr = (a, b, eps = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => near(x, b[i], eps));
};
const last = (...lines) => evaluate(lines).at(-1);
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

test('map(A) evaluates to a map value with rank and nullity', () => {
  const r = last('A = [[1, 2, 3], [4, 5, 6]]', 'map(A)');
  assert.equal(r.error, null);
  assert.equal(r.value.type, 'map');
  assert.deepEqual(r.value.M, [[1, 2, 3], [4, 5, 6]]);
  const { m, n, rank, nullity } = r.value.info;
  assert.deepEqual([m, n, rank, nullity], [2, 3, 2, 1]);
  const r2 = last('map([[1, 0], [0, 1], [1, 1]])');
  assert.deepEqual([r2.value.info.m, r2.value.info.n, r2.value.info.rank, r2.value.info.nullity], [3, 2, 2, 0]);
});

test('map rejects non-matrices and sizes outside 2..3', () => {
  assert.match(last('u = (1, 2, 3)', 'map(u)').error, /map needs a matrix/);
  assert.match(last('map([[1, 2, 3, 4], [1, 2, 3, 4]])').error, /2×4 matrix/);
  assert.match(last('map([[1, 2]])').error, /1×2 matrix/);
  assert.match(last('map(A)').error, /A is not defined/);
  assert.throws(() => mapValue([[1], [2]]), /2×1/);
});

test('kernel vectors are sent to zero and the image is spanned by the columns', () => {
  for (const M of [
    [[1, 2, 3], [4, 5, 6]], [[1, 2], [2, 4]], [[1, 2, 3], [2, 4, 6], [1, 1, 1]], [[0, 0], [0, 0], [0, 0]],
    [[1, 0], [0, 1], [2, 3]], [[1, 1, 1], [1, 1, 1]], [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
  ]) {
    const info = mapInfo(M);
    assert.equal(info.rank + info.nullity, info.n, JSON.stringify(M));
    for (const k of info.kernel) nearArr(applyMap(M, k), [0, 0, 0]);
    assert.equal(info.image.length, info.rank);
    for (const c of info.image) assert.equal(c.length, 3);
  }
  const z = mapInfo([[0, 0, 0], [0, 0, 0]]);
  assert.deepEqual([z.rank, z.nullity], [0, 3]);
  const s = mapInfo([[1, 2], [2, 4]]);
  assert.deepEqual([s.rank, s.nullity], [1, 1]);
  near(dot3(s.kernel[0], [1, 2, 0]), 0);
});

test('applyMap lifts R^2 into R^3 and projects R^3 onto R^2', () => {
  assert.deepEqual(applyMap([[1, 0], [0, 1], [1, 1]], [2, 3, 0]), [2, 3, 5]);
  assert.deepEqual(applyMap([[1, 0, 0], [0, 1, 0]], [2, 3, 7]), [2, 3, 0]);
  assert.deepEqual(applyMap([[0, -1], [1, 0]], [1, 0, 0]), [0, 1, 0]);
});

test('only flat items live in a 2D domain', () => {
  assert.ok(fitsDomain({ kind: 'vec', v: [1, 2, 0], o: [0, 0, 0] }, 2));
  assert.ok(!fitsDomain({ kind: 'vec', v: [1, 2, 3], o: [0, 0, 0] }, 2));
  assert.ok(!fitsDomain({ kind: 'vec', v: [1, 2, 0], o: [0, 0, 1] }, 2));
  assert.ok(!fitsDomain({ kind: 'span', vecs: [[1, 0, 0], [0, 0, 1]], o: [0, 0, 0] }, 2));
  assert.ok(fitsDomain({ kind: 'vec', v: [1, 2, 3], o: [0, 0, 0] }, 3));
});

test('mapItem maps tails, tips and shapes, keeping colour and labelling with the matrix', () => {
  const M = [[1, 0], [0, 1], [1, 1]];
  const v = mapItem({ kind: 'vec', v: [1, 2, 0], o: [1, 0, 0], color: '#123456', label: 'u', rowId: 'r1', index: 3 }, M, 'A');
  assert.equal(v.kind, 'dual-vec');
  assert.deepEqual(v.o, [1, 0, 1]);
  assert.deepEqual(v.v, [1, 2, 3]);
  assert.equal(v.color, '#123456');
  assert.equal(v.mat, 'A');
  assert.equal(v.label, 'u');
  assert.equal(v.rowId, 'r1');
  const p = mapItem({ kind: 'point', v: [1, 1, 0], o: [1, 0, 0], color: 'red', label: 'P' }, M);
  assert.deepEqual(p.v, [2, 1, 3]);
  assert.deepEqual(p.o, [0, 0, 0]);
  const g = mapItem({ kind: 'parallelogram', u: [1, 0, 0], v: [0, 1, 0], o: [0, 0, 0] }, M);
  assert.deepEqual([g.u, g.v], [[1, 0, 1], [0, 1, 1]]);
  const s = mapItem({ kind: 'span', vecs: [[1, 0, 0]], o: [0, 0, 0] }, M);
  assert.deepEqual(s.vecs, [[1, 0, 1]]);
  assert.equal(mapItem({ kind: 'plane', normal: [0, 0, 1] }, M), null);
});

test('grid segments: floor grid for R^2, 5x5x5 lattice for R^3', () => {
  assert.equal(niceStep(6), 1);
  assert.equal(niceStep(12), 2);
  const g2 = gridSegments(2, 6);
  assert.equal(g2.axes.length, 2);
  assert.equal(g2.segs.length, 24);
  for (const [p, q] of [...g2.segs, ...g2.axes]) { assert.equal(p[2], 0); assert.equal(q[2], 0); }
  const g3 = gridSegments(3, 6);
  assert.equal(g3.axes.length, 3);
  assert.equal(g3.segs.length, 72);
  const g12 = gridSegments(2, 12);
  assert.equal(g12.segs.length + g12.axes.length, 26); // step 2 out to 12
});

test('gridRange extends the R^2 grid until its image leaves the codomain box', () => {
  near(gridRange([[1, 0], [0, 1]], 6, 6), 6 * Math.SQRT2);
  near(gridRange([[2, 0], [0, 2]], 6, 6), 6); // never below the drawn grid
  near(gridRange([[0.5, 0], [0, 1]], 6, 6), 12 * Math.SQRT2);
  near(gridRange([[1, 0], [0, 1], [0, 0]], 6, 6), 6 * Math.sqrt(3));
  assert.equal(gridRange([[1, 2], [2, 4]], 6, 6), 6); // rank 1
  assert.equal(gridRange([[1e-6, 0], [0, 1e-6]], 6, 6), 40); // capped
  assert.equal(gridRange([[1, 0, 0], [0, 1, 0]], 6, 6), 6); // R^3 domain uses the lattice
  const g = gridSegments(2, 6, 8.5);
  assert.equal(g.segs.length + g.axes.length, 34);
  assert.deepEqual(g.axes[0], [[0, -8.5, 0], [0, 8.5, 0]]);
});

test('mapped grid of a 3x2 matrix lies in its column-space plane', () => {
  const M = [[1, 0], [0, 1], [1, 2]], nrm = cross3([1, 0, 1], [0, 1, 2]);
  const { segs, axes } = gridSegments(2, 6);
  const img = mapSegments(M, [...segs, ...axes]);
  assert.equal(img.length, 26);
  for (const [p, q] of img) { near(dot3(p, nrm), 0); near(dot3(q, nrm), 0); }
});

test('mapSegments drops segments squashed to a point', () => {
  const M = [[1, 0, 0], [0, 1, 0]]; // kills z
  const { segs, axes } = gridSegments(3, 6);
  const img = mapSegments(M, [...segs, ...axes]);
  assert.equal(img.length, 50); // the 25 z-lines vanish
  for (const [p] of img) assert.equal(p[2], 0);
});

test('clipSegment keeps inside parts only', () => {
  assert.deepEqual(clipSegment([-1, 0, 0], [1, 0, 0], 5), [[-1, 0, 0], [1, 0, 0]]);
  const r = clipSegment([-10, 0, 0], [10, 0, 0], 5);
  nearArr(r[0], [-5, 0, 0]);
  nearArr(r[1], [5, 0, 0]);
  assert.equal(clipSegment([6, -1, 0], [6, 1, 0], 5), null);
  assert.equal(clipSegment([10, 0, 0], [20, 0, 0], 5), null);
  const d = clipSegment([-10, -10, -10], [10, 10, 10], 2);
  nearArr(d[0], [-2, -2, -2]);
  nearArr(d[1], [2, 2, 2]);
});

test('needExtent sees mapped tips, origins and shape corners', () => {
  assert.equal(needExtent([{ kind: 'dual-vec', o: [1, 0, 0], v: [0, 7, 0] }]), 7);
  assert.equal(needExtent([{ kind: 'dual-vec', o: [4, 0, 0], v: [5, 0, 0] }]), 9);
  assert.equal(needExtent([{ kind: 'parallelepiped', o: [0, 0, 0], u: [1, 0, 0], v: [1, 0, 0], w: [1, 0, 0] }]), 3);
  assert.equal(needExtent([{ kind: 'dual-grid', segs: [[[100, 0, 0], [0, 0, 0]]], axes: [] }]), 0);
});

test('dualItems splits items into domain and codomain lists', () => {
  const r = evaluate(['A = [[1, 2, 3], [4, 5, 6]]', 'map(A)']).at(-1);
  const mapIt = { ...r.value, kind: 'map', rowId: 'm', index: 1 };
  const items = [
    mapIt,
    { kind: 'vec', v: [1, 0, 0], o: [0, 0, 0], color: '#e05a4f', label: 'u', rowId: 'a', index: 2 },
    { kind: 'point', v: [0, 1, 1], o: [0, 0, 0], color: '#4a90e2', label: 'P', rowId: 'b', index: 3 },
    { kind: 'plane', normal: [0, 0, 1], o: [0, 0, 0], color: '#43b05c', rowId: 'c', index: 4 },
  ];
  const { domain, codomain, skipped } = dualItems(items, r.value.info, r.value.M, { E: 6, matTex: 'A', from: mapIt });
  assert.deepEqual(skipped, []);
  assert.ok(!domain.some(it => it.kind === 'map'));
  assert.ok(domain.some(it => it.kind === 'plane'));
  const ker = domain.find(it => it.kind === 'dual-sub');
  assert.equal(ker.color, KER_COLOR);
  assert.equal(ker.basis.length, 1);
  assert.equal(ker.rowId, 'm');
  assert.ok(domain.some(it => it.kind === 'dual-grid' && it.n === 3 && !it.M)); // R^3 lattice shown in the domain
  const grid = codomain.find(it => it.kind === 'dual-grid');
  assert.deepEqual([grid.n, grid.E, grid.M], [3, 6, r.value.M]);
  const img = codomain.find(it => it.kind === 'dual-sub');
  assert.equal(img.color, IMG_COLOR);
  assert.equal(img.basis.length, 2);
  const u = codomain.find(it => it.rowId === 'a');
  assert.deepEqual([u.kind, u.v, u.color], ['dual-vec', [1, 4, 0], '#e05a4f']);
  const P = codomain.find(it => it.rowId === 'b');
  assert.deepEqual([P.kind, P.v], ['dual-pt', [5, 11, 0]]);
  assert.ok(!codomain.some(it => it.rowId === 'c'));
});

test('dualItems hides rows that are not in a 2D domain', () => {
  const M = [[1, 0], [0, 1], [1, 1]], info = mapInfo(M);
  const items = [
    { kind: 'vec', v: [1, 2, 0], o: [0, 0, 0], label: 'u' },
    { kind: 'vec', v: [1, 2, 3], o: [0, 0, 0], label: 'w' },
    { kind: 'vec', v: [0, 0, 1], o: [0, 0, 0], label: null },
  ];
  const { domain, codomain, skipped } = dualItems(items, info, M, { E: 6 });
  assert.deepEqual(skipped, ['w', null]);
  assert.equal(domain.filter(it => it.kind === 'vec').length, 1);
  assert.equal(codomain.filter(it => it.kind === 'dual-vec').length, 1);
  assert.ok(!domain.some(it => it.kind === 'dual-grid')); // the scene's own floor grid is the R^2 grid
});

test('linkSource follows whichever camera turned', () => {
  const a = [1, 0, 0], b = [0, 1, 0];
  assert.equal(linkSource(a, b, null), 'a');
  assert.equal(linkSource(a, a, a), null);
  assert.equal(linkSource([0.9, 0.1, 0], a, a), 'a');
  assert.equal(linkSource(a, [0.9, 0.1, 0], a), 'b');
});

test('matrixName reads the argument of map()', () => {
  assert.equal(matrixName('map(A)'), 'A');
  assert.equal(matrixName('V = map(B1)  # comment'), 'B1');
  assert.equal(matrixName('map([[1, 0], [0, 1]])'), 'T');
  assert.equal(matrixName('map(transpose(A))'), 'T');
  assert.equal(matrixName(''), 'T');
});

test('readout states the subspaces and the rank-nullity check', () => {
  assert.equal(subspaceTex(0, 3), '= \\{\\vec{0}\\}');
  assert.equal(subspaceTex(1, 3), '\\text{ is a line}');
  assert.equal(subspaceTex(2, 3), '\\text{ is a plane}');
  assert.equal(subspaceTex(2, 2), '= \\mathbb{R}^{2}');
  const r = readout(mapInfo([[1, 2, 3], [4, 5, 6]]), 'A');
  assert.match(r.domain, /domain \} \\mathbb\{R\}\^\{3\}/);
  assert.match(r.codomain, /codomain \} \\mathbb\{R\}\^\{2\}/);
  assert.match(r.ker, /\\ker A \\text\{ is a line\}.*nullity \} 1/);
  assert.match(r.im, /\\operatorname\{im\} A = \\mathbb\{R\}\^\{2\}.*rank \} 2/);
  assert.match(r.check, /2 \+ 1 = 3 = \\dim \\mathbb\{R\}\^\{3\}/);
  assert.match(r.row, /A: \\mathbb\{R\}\^\{3\} \\to \\mathbb\{R\}\^\{2\}/);
  const sq = readout(mapInfo([[1, 0], [0, 1]]));
  assert.match(sq.domain, /before/);
  assert.match(sq.codomain, /after/);
});

test('kernel and image labels learn where the vector labels are', () => {
  const M = [[1, 0, 1], [0, 1, 1]], info = mapInfo(M);
  const items = [
    { kind: 'vec', v: [1, 1, -1], o: [0, 0, 0], color: '#e05a4f', label: 'n', rowId: 'a', index: 1 },
    { kind: 'point', v: [1, 2, 0], o: [0, 0, 0], color: '#4a90e2', label: 'P', rowId: 'b', index: 2 },
  ];
  const { domain, codomain } = dualItems(items, info, M);
  assert.deepEqual(domain.find((it) => it.kind === 'dual-sub').avoid, [[1, 1, -1], [1, 2, 0]]);
  assert.deepEqual(codomain.find((it) => it.kind === 'dual-sub').avoid, [[0, 0, 0], [1, 2, 0]]);
  assert.deepEqual(labelTips([{ kind: 'span', vecs: [[1, 0, 0]] }]), []);
});
