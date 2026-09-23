import { test } from 'node:test';
import assert from 'node:assert/strict';
import katex from '../static/vendor/katex/katex.mjs';
import { evaluate } from '../static/graph/lang.js';
import {
  embed3, mapPath, makeMap, carryItem, carryAll, keepsSpan, glows, makeEigen, spiralPoints,
  svdFrames, svdPath, makeSvd, rotExp, rotLog, clipSegment, latticeSegments, mulMV, mulMM, det3,
  mapLatex, eigenLatex, svdLatex, sliderRanges, callArgs, niceDir, flipped, fmt, registerCarrier,
  labelHints, clearance, pickSpot, mapLines,
} from '../static/graph/features/transform.js';
import * as THREE from '../static/vendor/three/three.module.js';

const near = (a, b, tol = 1e-9) => {
  if (Array.isArray(a)) {
    assert.equal(a.length, b.length);
    a.forEach((x, i) => near(x, b[i], tol));
  } else assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);
};
const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const T = (M) => [0, 1, 2].map((j) => M.map((r) => r[j]));
const run = (lines) => evaluate(lines);
const valueOf = (lines) => {
  const rows = run(lines), last = rows.at(-1);
  assert.equal(last.error, null, last.error);
  return last.value;
};
const valid = (s) => katex.renderToString(s, { throwOnError: true, strict: 'ignore' });

test('embed3 keeps 3×3, puts 2×2 in the xy-plane, rejects other shapes', () => {
  assert.deepEqual(embed3([[1, 2], [3, 4]]), [[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
  assert.deepEqual(embed3(I), I);
  assert.throws(() => embed3([[1, 2], [3, 4], [5, 6]]), /2×2 or 3×3/);
});

test('mapPath interpolates I -> A, then A -> BA, clamping t', () => {
  const A = [[2, 0, 0], [0, 1, 0], [0, 0, 1]], B = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  near(mapPath(A, null, 0).M.flat(), I.flat());
  near(mapPath(A, null, 1).M.flat(), A.flat());
  near(mapPath(A, null, 0.5).M.flat(), [1.5, 0, 0, 0, 1, 0, 0, 0, 1]);
  near(mapPath(A, null, 7).M.flat(), A.flat());
  near(mapPath(A, null, -3).M.flat(), I.flat());
  const BA = mulMM(B, A);
  assert.equal(mapPath(A, B, 0.5).stage, 1);
  near(mapPath(A, B, 1).M.flat(), A.flat());
  near(mapPath(A, B, 2).M.flat(), BA.flat());
  near(mapPath(A, B, 1.5).M.flat(), A.map((r, i) => r.map((x, j) => (x + BA[i][j]) / 2)).flat());
  assert.equal(mapPath(A, B, 1.5).stage, 2);
  near(mapPath(A, B, 1.5).start.flat(), A.flat());
});

test('transform rows evaluate; order of composition matters', () => {
  const v = valueOf(['A = [[2,0],[0,3]]', 't = 0.5', 'transform(A, t)']);
  assert.equal(v.type, 'tf-map');
  assert.equal(v.dim, 2);
  near(v.det, 1.5 * 2);
  near(v.M.flat(), [1.5, 0, 0, 0, 2, 0, 0, 0, 1]);
  const AB = valueOf(['A = [[1,1],[0,1]]', 'B = [[0,-1],[1,0]]', 'transform(A, B, 2)']);
  const BA = valueOf(['A = [[1,1],[0,1]]', 'B = [[0,-1],[1,0]]', 'transform(B, A, 2)']);
  near(AB.M.flat(), mulMM(embed3([[0, -1], [1, 0]]), embed3([[1, 1], [0, 1]])).flat());
  assert.notDeepEqual(AB.M, BA.M);
  assert.equal(valueOf(['A = [[1,2,0],[0,1,0],[0,0,1]]', 'transform(A)']).t, 1);
  assert.equal(valueOf(['A = [[1,2,0],[0,1,0],[0,0,1]]', 'transform(A, A)']).stage, 2);
  assert.equal(valueOf(['A = [[1,2,0],[0,1,0],[0,0,1]]', 'B = [[1,0],[0,2]]', 'transform(A, B, 1)']).dim, 3);
});

test('transform rejects bad arguments with short messages', () => {
  const err = (lines) => run(lines).at(-1).error;
  assert.match(err(['transform((1,2,3), 1)']), /matrix first/);
  assert.match(err(['A = [[1,2],[3,4],[5,6]]', 'transform(A, 1)']), /2×2 or 3×3/);
  assert.match(err(['A = [[1,0],[0,1]]', 'transform(A, (1,0,0))']), /number t/);
  assert.match(err(['A = [[1,0],[0,1]]', 'transform(A, A, A)']), /number t/);
  assert.match(err(['A = [[1,0],[0,1]]', 'transform(A, 1, 1)']), /second matrix/);
});

test('singular maps report rank and null space; null vectors shrink to 0', () => {
  const v = valueOf(['A = [[1,2],[2,4]]', 't = 0.25', 'transform(A, t)']);
  assert.equal(v.rank, 1);
  assert.equal(v.nul.length, 1);
  near(mulMV(embed3([[1, 2], [2, 4]]), v.nul[0]), [0, 0, 0]);
  near(mulMV(v.M, v.nul[0]), v.nul[0].map((x) => 0.75 * x));
  const end = valueOf(['A = [[1,2],[2,4]]', 'transform(A, 1)']);
  near(mulMV(end.M, end.nul[0]), [0, 0, 0]);
  const flat = valueOf(['P = [[1,0,0],[0,1,0],[0,0,0]]', 'transform(P, 0.5)']);
  assert.equal(flat.rank, 2);
  near(flat.nul, [[0, 0, 1]]);
  const rank1 = valueOf(['A = [[1,1,1],[1,1,1],[1,1,1]]', 'transform(A, 0.5)']);
  assert.equal(rank1.rank, 1);
  assert.equal(rank1.nul.length, 2);
  const zero = valueOf(['Z = [[0,0],[0,0]]', 'transform(Z, 0.5)']);
  assert.equal(zero.rank, 0);
  const inv = valueOf(['A = [[1,1],[0,1]]', 'transform(A, 0.5)']);
  assert.equal(inv.rank, 2);
  assert.deepEqual(inv.nul, []);
});

test('composition: the null space is that of BA in the second stage', () => {
  const v = valueOf(['A = [[0,-1],[1,0]]', 'B = [[1,0],[0,0]]', 'transform(A, B, 1.5)']);
  assert.equal(v.rank, 1);
  const BA = mulMM(embed3([[1, 0], [0, 0]]), embed3([[0, -1], [1, 0]]));
  near(mulMV(BA, v.nul[0]), [0, 0, 0]);
  assert.equal(valueOf(['A = [[0,-1],[1,0]]', 'B = [[1,0],[0,0]]', 'transform(A, B, 0.5)']).rank, 2);
});

test('determinant sign flips colour: a reflection passes through 0', () => {
  const v = (t) => valueOf(['S = [[0,1],[1,0]]', `transform(S, ${t})`]);
  assert.ok(!flipped(v(0.2)));
  near(v(0.5).det, 0);
  assert.ok(!flipped(v(0.5)));
  assert.ok(flipped(v(0.8)));
  near(v(1).det, -1);
});

test('carryItem applies M to vectors, points, spans, planes and solids', () => {
  const M = [[1, 1, 0], [0, 1, 0], [0, 0, 2]];
  assert.deepEqual(carryItem({ kind: 'vec', v: [1, 2, 3], o: [0, 1, 0] }, M), { kind: 'vec', v: [3, 2, 6], o: [1, 1, 0] });
  assert.deepEqual(carryItem({ kind: 'point', v: [1, 0, 1], o: [0, 0, 0] }, M).v, [1, 0, 2]);
  assert.deepEqual(carryItem({ kind: 'span', vecs: [[0, 1, 0]], o: [0, 0, 0] }, M).vecs, [[1, 1, 0]]);
  const pg = carryItem({ kind: 'parallelogram', u: [1, 0, 0], v: [0, 1, 0], o: [0, 0, 0] }, M);
  assert.deepEqual([pg.u, pg.v], [[1, 0, 0], [1, 1, 0]]);
  const pp = carryItem({ kind: 'parallelepiped', u: [1, 0, 0], v: [0, 1, 0], w: [0, 0, 1], o: [0, 0, 0] }, M);
  assert.deepEqual(pp.w, [0, 0, 2]);
  // plane with normal n maps to a plane with normal ∝ M^{-T} n
  const pl = carryItem({ kind: 'plane', normal: [1, 0, 0], o: [0, 0, 0] }, M);
  const Minv = [[1, -1, 0], [0, 1, 0], [0, 0, 0.5]];
  const want = mulMV(T(Minv), [1, 0, 0]);
  near(pl.normal.map((x) => x / Math.hypot(...pl.normal)), want.map((x) => x / Math.hypot(...want)).map((x) => x * Math.sign(pl.normal[0] * want[0] || 1)));
  // a plane squashed by a singular map becomes a line (span)
  const P = [[1, 0, 0], [0, 0, 0], [0, 0, 0]];
  const sq = carryItem({ kind: 'plane', normal: [0, 0, 1], o: [0, 0, 0] }, P);
  assert.equal(sq.kind, 'span');
  assert.equal(sq.vecs.length, 1);
  assert.equal(carryItem({ kind: 'plane', normal: [1, 0, 0], o: [0, 0, 0] }, P).vecs.length, 0);
  // unknown kinds ride along only through registerCarrier
  const odd = { kind: 'zz-thing', p: [1, 0, 0] };
  assert.equal(carryItem(odd, M), odd);
  registerCarrier('zz-thing', (it, _M, f) => ({ ...it, p: f(it.p) }));
  assert.deepEqual(carryItem(odd, M).p, [1, 0, 0]);
  assert.deepEqual(carryItem({ ...odd, p: [0, 1, 0] }, M).p, [1, 1, 0]);
});

test('carryItem keeps linear combo pictures and drops the ones a map would falsify', () => {
  const M = [[2, 0, 0], [0, 1, 0], [0, 0, 1]];
  const ex = carryItem({ kind: 'vec', v: [1, 1, 0], o: [0, 0, 0], combo: { mode: 'explain', cols: [[1, 0, 0], [0, 1, 0]], coeffs: [1, 1] } }, M);
  assert.deepEqual(ex.combo.cols, [[2, 0, 0], [0, 1, 0]]);
  assert.deepEqual(ex.v, [2, 1, 0]);
  const sh = carryItem({ kind: 'vec', v: [1, 0, 0], o: [0, 0, 0], combo: { mode: 'shadow', u: [1, 1, 0], onto: [1, 0, 0] } }, M);
  assert.equal(sh.combo, undefined);
  const line = carryItem({ kind: 'cb-line', p: [0, 1, 0], d: [1, 0, 0], pts: [[0, 1, 0], [1, 1, 0]], o: [0, 0, 0] }, M);
  assert.deepEqual([line.p, line.d], [[0, 1, 0], [2, 0, 0]]);
  const arc = carryItem({ kind: 'cb-arc', u: [1, 0, 0], v: [1, 1, 0], theta: Math.PI / 4, o: [0, 0, 0] }, M);
  near(arc.theta, Math.atan2(1, 2));
});

test('carryAll: first transform carries the rest, fixed rows and overlays stay', () => {
  const rows = run(['A = [[2,0],[0,1]]', 't = 1', 'transform(A, t)', 'u = (1, 1, 0)', 'fixed(u)', 'transform(A, 0)']);
  const items = rows.map((r, i) => r.value && typeof r.value === 'object' && r.value.type !== 'mat'
    ? { ...r.value, kind: r.value.type, o: [0, 0, 0], index: i } : null).filter(Boolean);
  const out = carryAll(items);
  assert.deepEqual(out.find((it) => it.index === 3).v, [2, 1, 0]);
  assert.deepEqual(out.find((it) => it.index === 4).v, [1, 1, 0]);
  assert.equal(out.find((it) => it.index === 4).fixed, true);
  assert.deepEqual(out.find((it) => it.index === 5).M, items.find((it) => it.index === 5).M);
  const still = items.filter((it) => it.kind !== 'tf-map');
  assert.equal(carryAll(still), still);
});

test('eigenvectors glow while the transform keeps them on their span', () => {
  const rows = run(['A = [[2,1],[1,2]]', 'transform(A, 0.5)', 'eigen(A)', 'u = (1, 1, 0)', 'w = (1, 0, 0)', 'P = point(-2, 2, 0)']);
  const items = rows.map((r, i) => r.value && typeof r.value === 'object' && r.value.type !== 'mat'
    ? { ...r.value, kind: r.value.type, o: [0, 0, 0], index: i } : null).filter(Boolean);
  const out = carryAll(items);
  const glowing = out.filter((it) => it.kind === 'tf-glow').map((it) => it.index);
  assert.deepEqual(glowing, [3, 5]);
  near(out.find((it) => it.index === 3 && it.kind === 'vec').v, [2, 2, 0]); // λ = 3 at t = 0.5 -> ×2
  assert.ok(out.find((it) => it.kind === 'tf-eigen').carry);
  // no eigen row, no glow
  assert.equal(carryAll(items.filter((it) => it.kind !== 'tf-eigen')).filter((it) => it.kind === 'tf-glow').length, 0);
});

test('keepsSpan: stage 2 of a composition also needs B to keep the vector', () => {
  const A = embed3([[2, 0], [0, 3]]), B = embed3([[0, -1], [1, 0]]);
  assert.ok(keepsSpan([1, 0, 0], { A, B: null, t: 1 }));
  assert.ok(keepsSpan([1, 0, 0], { A, B, t: 0.7 }));
  assert.ok(!keepsSpan([1, 0, 0], { A, B, t: 1.5 }));
  assert.ok(!keepsSpan([1, 1, 0], { A, B: null, t: 1 }));
  const spaces = [[[1, 0, 0]]];
  assert.ok(glows({ kind: 'vec', v: [3, 0, 0], o: [0, 0, 0] }, spaces, { A, B: null, t: 1 }));
  assert.ok(!glows({ kind: 'vec', v: [3, 0, 0], o: [0, 1, 0] }, spaces, { A, B: null, t: 1 }));
  assert.ok(!glows({ kind: 'vec', v: [0, 0, 0], o: [0, 0, 0] }, spaces, { A, B: null, t: 1 }));
});

test('eigen: real lines, repeated and defective values, complex planes', () => {
  const sym = makeEigen([[2, 1], [1, 2]]);
  assert.deepEqual(sym.lines.map((l) => l.lam), [3, 1]);
  near(niceDir(sym.lines[0].basis[0]), [1, 1, 0]);
  near(niceDir(sym.lines[1].basis[0]).map(Math.abs), [1, 1, 0]);
  const shear = makeEigen([[1, 1], [0, 1]]);
  assert.equal(shear.lines.length, 1);
  assert.equal(shear.lines[0].mult, 2);
  assert.equal(shear.lines[0].basis.length, 1);
  assert.match(eigenLatex(shear), /defective/);
  assert.match(eigenLatex(makeEigen([[2, 0], [0, 2]])), /every vector/);
  const rot = makeEigen([[0, -1], [1, 0]]);
  assert.equal(rot.lines.length, 0);
  assert.equal(rot.pairs.length, 1);
  near([rot.pairs[0].re, rot.pairs[0].im], [0, 1]);
  const r3 = makeEigen([[0, -1, 0], [1, 0, 0], [0, 0, 2]]);
  assert.deepEqual(r3.lines.map((l) => l.lam), [2]);
  near(r3.pairs[0].p[2], 0);
  near(r3.pairs[0].q[2], 0);
  assert.throws(() => makeEigen([[1, 2, 3], [4, 5, 6]]), /2×2 or 3×3/);
  assert.match(run(['eigen((1,2,3))']).at(-1).error, /matrix/);
});

test('spiralPoints follows the action of A on its invariant plane', () => {
  const p = [1, 0, 0], q = [0, 1, 0];
  const circle = spiralPoints(p, q, 0, 1, 2);
  near(circle[0], [2, 0, 0]);
  for (const x of circle) near(Math.hypot(...x), 2, 1e-9);
  const out = spiralPoints(p, q, 1, 1, 2); // |λ| = √2: grows, capped at 3×
  const r = out.map((x) => Math.hypot(...x));
  near(r.at(-1), 2);
  near(r[0], 2 / 3, 1e-9);
  assert.ok(r.every((x, i) => i === 0 || x >= r[i - 1]));
  const inward = spiralPoints(p, q, 0.5, 0.5, 2).map((x) => Math.hypot(...x));
  near(inward[0], 2);
  assert.ok(inward.at(-1) < inward[0]);
  // turning direction: from p toward q
  assert.ok(circle[5][1] > 0);
});

test('rotExp/rotLog round-trip, including half turns', () => {
  for (const [axis, angle] of [[[1, 2, 3], 0.7], [[0, 0, 1], Math.PI], [[1, -1, 0.5], Math.PI - 1e-5], [[0, 1, 0], 0]]) {
    const R = rotExp(axis, angle), { axis: a, angle: th } = rotLog(R);
    near(rotExp(a, th).flat(), R.flat(), 1e-8);
    near(det3(R), 1);
  }
});

test('svdFrames: A = U diag(sgn) Vᵀ with proper rotations', () => {
  for (const m of [
    [[3, 0], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 0]], [[2, 1, 0], [0, 1, 1], [1, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]], [[1, 2, 3], [2, 4, 6], [1, 1, 1]], [[0, 0], [0, 0]],
  ]) {
    const F = svdFrames(m), A = embed3(m);
    const rebuilt = mulMM(mulMM(F.U, [[F.sgn[0], 0, 0], [0, F.sgn[1], 0], [0, 0, F.sgn[2]]]), T(F.V));
    near(rebuilt.flat(), A.flat(), 1e-8);
    near(det3(F.U), 1, 1e-9);
    near(det3(F.V), 1, 1e-9);
    assert.equal(F.sig.length, m.length);
    F.sig.forEach((s, i) => near(Math.abs(F.sgn[i]), s, 1e-9));
  }
  assert.ok(svdFrames([[0, 1], [1, 0]]).sgn.some((x) => x < 0));
  near(svdFrames([[3, 0], [0, 1]]).sig, [3, 1]);
});

test('svdPath runs I -> Vᵀ -> ΣVᵀ -> A continuously', () => {
  const m = [[2, 1, 0], [0, 1, 1], [1, 0, -1]], F = svdFrames(m), A = embed3(m);
  near(svdPath(F, 0).M.flat(), I.flat(), 1e-9);
  near(svdPath(F, 3).M.flat(), A.flat(), 1e-8);
  near(svdPath(F, 1).M.flat(), T(F.V).flat(), 1e-8);
  for (const t of [1, 2]) near(svdPath(F, t - 1e-7).M.flat(), svdPath(F, t + 1e-7).M.flat(), 1e-5);
  const mid = svdPath(F, 0.4).M;
  near(mulMM(mid, T(mid)).flat(), I.flat(), 1e-9); // still a rotation
  assert.equal(svdPath(F, 2.5).stage, 3);
  const v = valueOf(['A = [[3,0],[0,1]]', 't = 1.5', 'svdview(A, t)']);
  assert.equal(v.stage, 2);
  near(v.M[0][0], 2);
  assert.match(svdLatex(v), /stretch/);
  const s = makeSvd([{ type: 'mat', m: [[1, 2], [3, 4]] }]);
  assert.equal(s.t, null);
  near(s.M.flat(), embed3([[1, 2], [3, 4]]).flat());
  assert.match(run(['A = [[1,2,3],[4,5,6]]', 'svdview(A)']).at(-1).error, /2×2 or 3×3/);
});

test('clipSegment and the warped lattice stay inside the view cube', () => {
  near(clipSegment([-10, 0, 0], [10, 0, 0], 6).flat(), [-6, 0, 0, 6, 0, 0]);
  assert.equal(clipSegment([-10, 8, 0], [10, 8, 0], 6), null);
  near(clipSegment([0, 0, 0], [1, 1, 1], 6).flat(), [0, 0, 0, 1, 1, 1]);
  const id = latticeSegments(I, 6, 3);
  assert.equal(id.axes.length, 6);          // x, y, z axes
  assert.equal(id.plane.length, 2 * 24);    // |k| <= 6 after clipping, k != 0
  assert.ok(id.cage.length > 0);
  for (const p of [...id.axes, ...id.plane, ...id.cage]) assert.ok(p.every((x) => Math.abs(x) <= 6 + 1e-9));
  const flat = latticeSegments(embed3([[1, 1], [0, 1]]), 6, 2);
  assert.equal(flat.cage.length, 0);
  assert.ok(flat.plane.every((p) => p[2] === 0));
  const squash = latticeSegments([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], 6, 3);
  assert.ok(squash.plane.some((p) => Math.abs(p[0]) > 5.9)); // shrinking maps still fill the view
});

test('readouts render in KaTeX and mention det, rank, stage, singular values', () => {
  const v = valueOf(['A = [[1,2],[2,4]]', 'B = [[0,1],[1,0]]', 'transform(A, B, 1.5)']);
  const tex = mapLatex(v, { A: 'R', B: 'S' });
  valid(tex);
  assert.match(tex, /\\det M = /);
  assert.match(tex, /R \\to SR/);
  assert.match(tex, /rank\}\(SR\) = 1/);
  assert.match(mapLatex(valueOf(['S = [[0,1],[1,0]]', 'transform(S, 1)'])), /orientation flipped/);
  valid(eigenLatex(makeEigen([[1, -2], [2, 1]])));
  assert.match(eigenLatex(makeEigen([[1, -2], [2, 1]])), /1 \\pm 2i/);
  assert.match(eigenLatex(makeEigen([[0, -1], [1, 0]])), /= \\pm i$/);
  valid(svdLatex(valueOf(['svdview([[1,1,0],[0,1,0],[0,0,1]], 0.5)'])));
  assert.match(svdLatex(valueOf(['svdview([[3,0],[0,1]])'])), /\\sigma_1 = 3,\\ \\sigma_2 = 1$/);
  assert.equal(fmt(-0.0001), '0');
  assert.equal(fmt(1.23456, 3), '1.235');
});

test('fixed() marks objects and passes numbers through', () => {
  assert.equal(valueOf(['u = (1,2,3)', 'fixed(u)']).fixed, true);
  assert.equal(valueOf(['fixed(span((1,0,0)))']).type, 'span');
  assert.equal(valueOf(['fixed(2)']), 2);
  assert.equal(valueOf(['u = fixed((1,2,3))', '2u']).fixed, undefined);
});

test('sliderRanges finds sliders used as t; callArgs reads argument names', () => {
  const srcs = ['A = [[1,1],[0,1]]', 't = 0', 's = 0.5', 'transform(A, t)', 'T = svdview(A, s)', 'transform(A, A)', 'r = 1', 'transform(A, A, r)'];
  const ranges = sliderRanges(srcs, run(srcs));
  assert.deepEqual([...ranges].sort(), [[1, 1], [2, 3], [6, 2]]);
  assert.deepEqual(callArgs('M = transform(R, S, t)', 'transform'), ['R', 'S', 't']);
  assert.deepEqual(callArgs('transform([[1,0],[0,1]], t)', 'transform'), [null, 't']);
  assert.deepEqual(callArgs('eigen(A)', 'transform'), []);
});

test('labelHints hands vector and point tips to transform and eigen items only', () => {
  const items = [
    { kind: 'tf-map', M: I }, { kind: 'tf-eigen', lines: [] },
    { kind: 'vec', v: [1, 2, 0], o: [1, 0, 0] }, { kind: 'point', v: [0, 0, 3], o: [0, 0, 0] }, { kind: 'span', vecs: [[1, 0, 0]] },
  ];
  const out = labelHints(items);
  assert.deepEqual(out[0].avoid, [{ p: [2, 2, 0], d: [1, 2, 0] }, { p: [0, 0, 3], d: null }]);
  assert.equal(out[1].avoid, out[0].avoid);
  assert.equal(out[2], items[2]);
  const plain = [items[2], items[3]];
  assert.equal(labelHints(plain), plain);
});

test('label spots: clearance on screen, first clear spot wins, last choice kept while clear', () => {
  const cam = new THREE.PerspectiveCamera(40, 1.5, 0.01, 100);
  cam.position.set(0, 0, 20);
  cam.lookAt(0, 0, 0);
  cam.updateMatrixWorld();
  const V = (x, y) => new THREE.Vector3(x, y, 0);
  assert.ok(clearance(cam, V(0, 0), [V(0.05, 0)]) < 1);
  assert.ok(clearance(cam, V(0, 0), [V(3, 0)]) >= 1);
  assert.ok(clearance(cam, V(0, 0), []) >= 1);
  assert.equal(clearance(cam, V(100, 0), []), -1); // off screen
  const spots = [V(0, 0), V(3, 3), V(-3, 3)];
  assert.equal(pickSpot(cam, spots, [V(0, 0.1)]), 1);
  assert.equal(pickSpot(cam, spots, [V(0, 0.1)], 2), 2);
  assert.equal(pickSpot(cam, spots, [], 2), 2);
  assert.equal(pickSpot(cam, spots, [V(0, 0), V(3, 3), V(-3, 3)]), 0); // nothing clear: the clearest
});

test('readout colours follow the theme', () => {
  const v = valueOf(['A = [[1,2],[3,4]]', 'transform(A, 1)']);
  assert.match(mapLines(v, {}, 'dark')[0], /#43b05c/);
  assert.match(mapLines(v, {}, 'light')[0], /#2e8b45/);
});
