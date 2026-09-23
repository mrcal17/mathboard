import { test } from 'node:test';
import assert from 'node:assert/strict';
import katex from '../static/vendor/katex/katex.mjs';
import { evaluate } from '../static/graph/lang.js';
import {
  rk4Matrix, flowInfo, fieldGlyphs, sampleOrbits, makeSwarm, stepSwarm, fillTrails,
  iterateInfo, powerInfo, dominant, symEig, quadricInfo, quadricPatches, conicCurves, clipPolyline,
  readoutLines, callArgs, eigTex, tones, luminance, stepRanges, mixHex, PARTICLES, TRAIL,
} from '../static/graph/features/fields.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
const nearArr = (a, b, eps = 1e-9) => {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => near(x, b[i], eps));
};
const mv = (A, x) => A.map((r) => r.reduce((s, a, j) => s + a * x[j], 0));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const len = (a) => Math.sqrt(dot(a, a));
const form = (S, x) => dot(x.slice(0, S.length), mv(S, x.slice(0, S.length)));
function lcg(seed = 1) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
}
const valid = (s) => katex.renderToString(s, { throwOnError: true, strict: 'ignore' });

test('rk4Matrix is one classical RK4 step', () => {
  const A = [[0.3, -1.2, 0.5], [1, -0.4, 0], [0.2, 0.7, -0.9]], x = [1, -2, 0.5], h = 0.13;
  const f = (y) => mv(A, y), add = (a, b, k) => a.map((v, i) => v + k * b[i]);
  const k1 = f(x), k2 = f(add(x, k1, h / 2)), k3 = f(add(x, k2, h / 2)), k4 = f(add(x, k3, h));
  const want = x.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  nearArr(mv(rk4Matrix(A, h), x), want, 1e-12);
  nearArr(rk4Matrix(A, 0).flat(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
});

test('flowInfo names 2D phase portraits', () => {
  const cases = [
    [[[1, 0], [0, -1]], 'saddle'],
    [[[-1, -2], [2, -1]], 'stable spiral'],
    [[[0.5, -2], [2, 0.5]], 'unstable spiral'],
    [[[0, -1], [1, 0]], 'center'],
    [[[-2, 0], [0, -2]], 'stable star'],
    [[[2, 1], [0, 2]], 'unstable degenerate node'],
    [[[-1, 0], [0, -3]], 'stable node'],
    [[[2, 1], [1, 2]], 'unstable node'],
    [[[0, 1], [0, 0]], 'shear flow'],
    [[[1, 0], [0, 0]], 'line of fixed points, repelling'],
    [[[-1, 0], [0, 0]], 'line of fixed points, attracting'],
    [[[0, 0], [0, 0]], 'no motion (A = 0)'],
  ];
  for (const [A, kind] of cases) assert.equal(flowInfo(A).kind, kind, JSON.stringify(A));
  const sad = flowInfo([[1, 0], [0, -1]]);
  assert.equal(sad.groups.length, 2);
  assert.deepEqual(sad.groups.map((g) => g.lam), [1, -1]);
  nearArr(sad.groups[0].basis[0], [1, 0, 0]);
  const sp = flowInfo([[-1, -2], [2, -1]]);
  assert.equal(sp.pairs.length, 1);
  near(sp.pairs[0].re, -1);
  near(sp.pairs[0].im, 2);
  assert.equal(sp.groups.length, 0);
});

test('flowInfo names 3D flows and counts directions', () => {
  assert.equal(flowInfo([[-1, 0, 0], [0, -2, 0], [0, 0, -3]]).kind, 'sink');
  assert.equal(flowInfo([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).kind, 'source');
  const s = flowInfo([[1, 0, 0], [0, -1, 0], [0, 0, -2]]);
  assert.equal(s.kind, 'saddle');
  assert.equal(s.detail, '2 stable, 1 unstable');
  assert.equal(flowInfo([[-0.2, -1, 0], [1, -0.2, 0], [0, 0, 0.5]]).kind, 'spiral saddle');
  assert.equal(flowInfo([[-0.2, -1, 0], [1, -0.2, 0], [0, 0, -0.5]]).kind, 'spiral sink');
  assert.equal(flowInfo([[0, -1, 0], [1, 0, 0], [0, 0, 0]]).kind, 'center (closed orbits)');
  assert.equal(flowInfo([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).kind, 'non-isolated fixed points');
  assert.equal(flowInfo([[0, 1, 0], [0, 0, 1], [0, 0, 0]]).kind, 'shear flow');
  const plane = flowInfo([[2, 0, 0], [0, 2, 0], [0, 0, -1]]);
  assert.equal(plane.groups[0].basis.length, 2);
});

test('flow speed is rescaled only for very fast or very slow matrices', () => {
  assert.equal(flowInfo([[1, 0], [0, -1]]).rate, 1);
  assert.equal(flowInfo([[0, -1], [1, 0]]).rate, 1);
  near(flowInfo([[10, 0], [0, 10]]).rate, 0.15);
  near(flowInfo([[0.1, 0], [0, 0.1]]).rate, 4);
  assert.equal(flowInfo([[0, 0], [0, 0]]).rate, 1);
});

test('field glyphs: clamped lengths, alphas and directions along Ax', () => {
  const A = [[1, 2], [-3, 0.5]], E = 6, { spacing, glyphs } = fieldGlyphs(A, 2, E);
  near(spacing, 1.2);
  assert.equal(glyphs.length, 11 * 11 - 1); // the origin has no arrow
  for (const g of glyphs) {
    assert.ok(g.len >= 0.25 * spacing - 1e-12 && g.len <= 0.8 * spacing + 1e-12);
    assert.ok(g.alpha >= 0.3 - 1e-12 && g.alpha <= 1 + 1e-12);
    near(len(g.dir), 1);
    const f = [...mv(A, g.p.slice(0, 2)), 0];
    nearArr(g.dir, f.map((x) => x / len(f)));
    assert.equal(g.p[2], 0);
  }
  assert.ok(glyphs.some((g) => Math.abs(g.len - 0.8 * spacing) < 1e-9));
  const g3 = fieldGlyphs([[1, 0, 0], [0, -1, 0], [0, 0, 0.5]], 3, 6);
  assert.equal(g3.glyphs.length, 124);
  assert.equal(fieldGlyphs([[0, 0], [0, 0]], 2, 6).glyphs.length, 0);
});

test('sample orbits spiral in, spiral out, or close up', () => {
  const E = 6;
  const sink = flowInfo([[-0.3, -2], [2, -0.3]]), src = flowInfo([[0.3, -2], [2, 0.3]]), ctr = flowInfo([[0, -2], [2, 0]]);
  const [a] = sampleOrbits(sink.A, sink.pairs[0], E);
  assert.ok(len(a[0]) > len(a.at(-1)) * 5);
  const [b] = sampleOrbits(src.A, src.pairs[0], E);
  assert.ok(len(b.at(-1)) > len(b[0]) * 5);
  const c = sampleOrbits(ctr.A, ctr.pairs[0], E);
  assert.equal(c.length, 2);
  for (const orbit of c) near(len(orbit.at(-1)), len(orbit[0]), 1e-3);
});

test('swarm: spawns inside the box, respawns escapers, keeps RK4 motion', () => {
  const E = 6, rand = lcg(7), sw = makeSwarm(2, E, 200, rand);
  assert.equal(PARTICLES[2] > 0 && PARTICLES[3] > PARTICLES[2], true);
  for (let i = 0; i < sw.n; i++) {
    assert.ok(Math.abs(sw.pos[3 * i]) <= E && Math.abs(sw.pos[3 * i + 1]) <= E);
    assert.equal(sw.pos[3 * i + 2], 0);
  }
  // rotation: radii are kept (RK4 error is tiny for small steps)
  const R = [[0, -1], [1, 0]], P = rk4Matrix(R, 0.01);
  const before = Array.from({ length: sw.n }, (_, i) => Math.hypot(sw.pos[3 * i], sw.pos[3 * i + 1]));
  const ages = Array.from(sw.age);
  stepSwarm(sw, P, 0.01, rand);
  for (let i = 0; i < sw.n; i++) {
    if (sw.age[i] === 0) continue; // respawned
    near(Math.hypot(sw.pos[3 * i], sw.pos[3 * i + 1]), before[i], 1e-4);
    near(sw.age[i], ages[i] + 0.01, 1e-6);
    assert.ok(sw.alpha[i] >= 0 && sw.alpha[i] <= 1);
  }
  // pushed outside -> respawned inside with age 0; same for one that reached the origin
  sw.pos[0] = 50; sw.pos[1] = 0; sw.age[0] = 1; sw.life[0] = 5;
  sw.pos[3] = 0.001; sw.pos[4] = 0; sw.age[1] = 1; sw.life[1] = 5;
  stepSwarm(sw, [[1, 0], [0, 1]], 0.01, rand);
  for (const i of [0, 1]) {
    assert.equal(sw.age[i], 0);
    assert.ok(Math.abs(sw.pos[3 * i]) <= E && Math.abs(sw.pos[3 * i + 1]) <= E);
    assert.equal(sw.alpha[i], 0);
  }
  // aged out
  sw.age[2] = 10; sw.life[2] = 4;
  stepSwarm(sw, [[1, 0], [0, 1]], 0.01, rand);
  assert.equal(sw.age[2], 0);
});

test('3D swarm fills the cube and trails trace the recent past', () => {
  const E = 6, sw = makeSwarm(3, E, 50, lcg(3));
  assert.ok(Array.from(sw.pos).some((x, i) => i % 3 === 2 && Math.abs(x) > 1));
  stepSwarm(sw, rk4Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0), 0);
  const A = [[-0.5, 1, 0], [-1, -0.5, 0], [0, 0, 0.3]], dt = TRAIL.seconds / TRAIL.segs;
  const Bk = rk4Matrix(A, -dt), segs = TRAIL.segs;
  const out = new Float32Array(sw.n * segs * 6), col = new Float32Array(sw.n * segs * 8).fill(1);
  fillTrails(sw, Bk, segs, out, col);
  const i = 4, head = [sw.pos[3 * i], sw.pos[3 * i + 1], sw.pos[3 * i + 2]], v = i * segs * 2;
  nearArr(Array.from(out.slice(3 * v, 3 * v + 3)), head, 1e-6);
  const back = mv(Bk, head);
  if (back.every((x) => Math.abs(x) <= 1.15 * E)) nearArr(Array.from(out.slice(3 * v + 3, 3 * v + 6)), back, 1e-5);
  // alpha fades from the head to the tail
  for (let s = 0; s < segs; s++) assert.ok(col[4 * (v + 2 * s) + 3] >= col[4 * (v + 2 * s + 1) + 3]);
  near(col[4 * (v + 2 * segs - 1) + 3], 0, 1e-7);
});

test('iterate: v, Av, A²v, … and a partial next step', () => {
  const A = [[2, 1], [0, 0.5]], v = [1, 1];
  const r = iterateInfo(A, v, 3);
  assert.deepEqual(r.seq, [[1, 1, 0], [3, 0.5, 0], [6.5, 0.25, 0], [13.25, 0.125, 0]]);
  assert.equal(r.next, null);
  const p = iterateInfo(A, v, 2, 0.4);
  assert.equal(p.seq.length, 3);
  assert.deepEqual(p.next, [13.25, 0.125, 0]);
  assert.equal(p.frac, 0.4);
  assert.throws(() => iterateInfo([[1e200, 0], [0, 1]], [1e200, 0], 3), /too large/);
});

test('power iteration converges to the dominant eigenvector with the Rayleigh quotient', () => {
  const A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]], v = [1, 0, 0];
  const r = powerInfo(A, v, 60);
  const dom = dominant(A);
  assert.equal(dom.kind, 'line');
  near(r.rayleigh, dom.lam, 1e-8);
  near(Math.abs(dot(r.q, dom.dir)), 1, 1e-8);
  for (const x of r.seq) near(len(x), 1, 1e-12);
  assert.ok(dom.ratio > 0 && dom.ratio < 1);
  // arrows keep |v|; step 0 is v itself
  const s = powerInfo([[3, 0], [0, 1]], [2, 2], 4, 0.5);
  nearArr(s.seq[0], [2, 2, 0]);
  for (const x of s.seq) near(len(x), 2 * Math.SQRT2, 1e-12);
  assert.ok(s.next && s.frac === 0.5);
  assert.throws(() => powerInfo(A, [0, 0, 0], 3), /nonzero/);
  // a starting vector killed by A stops the iteration
  const z = powerInfo([[1, 1], [1, 1]], [1, -1], 5);
  assert.equal(z.dead, 1);
  assert.equal(z.n, 0);
});

test('dominant spots complex pairs, ± ties, planes and defective leads', () => {
  assert.equal(dominant([[0, -2], [2, 0]]).kind, 'complex');
  assert.equal(dominant([[1, 0], [0, -1]]).kind, 'tie');
  assert.equal(dominant([[2, 0, 0], [0, 2, 0], [0, 0, 1]]).kind, 'plane');
  const d = dominant([[2, 1], [0, 2]]);
  assert.equal(d.kind, 'line');
  assert.match(d.note, /slow/);
  assert.equal(dominant([[0, 1], [0, 0]]).kind, 'zero');
  const neg = dominant([[-3, 0], [0, 1]]);
  assert.equal(neg.lam, -3);
});

test('symEig diagonalises symmetric matrices', () => {
  const rand = lcg(11);
  for (let t = 0; t < 30; t++) {
    const n = t % 2 ? 3 : 2, M = Array.from({ length: n }, () => Array.from({ length: n }, () => 6 * rand() - 3));
    const S = M.map((r, i) => r.map((x, j) => (x + M[j][i]) / 2));
    const { values, vectors } = symEig(S);
    for (let i = 1; i < n; i++) assert.ok(values[i - 1] >= values[i]);
    vectors.forEach((q, i) => {
      nearArr(mv(S, q), q.map((x) => values[i] * x), 1e-9);
      near(len(q), 1, 1e-12);
      for (let j = 0; j < i; j++) near(dot(q, vectors[j]), 0, 1e-9);
    });
  }
  const r = symEig([[2, 0, 0], [0, 2, 0], [0, 0, 5]]);
  assert.deepEqual(r.values, [5, 2, 2]);
});

const D = (a, b, c) => (c === undefined ? [[a, 0], [0, b]] : [[a, 0, 0], [0, b, 0], [0, 0, c]]);

test('quadricInfo classifies by eigenvalue signs', () => {
  const cases = [
    [D(1, 2, 3), 1, 'ellipsoid', 'ellipsoid'],
    [D(1, 1, 1), 1, 'ellipsoid', 'sphere'],
    [D(1, 1, 4), 1, 'ellipsoid', 'spheroid (ellipsoid of revolution)'],
    [D(1, 2, -1), 1, 'hyperboloid1', 'hyperboloid of one sheet'],
    [D(1, -2, -1), 1, 'hyperboloid2', 'hyperboloid of two sheets'],
    [D(1, 2, 0), 1, 'ellcyl', 'elliptic cylinder'],
    [D(3, 3, 0), 1, 'ellcyl', 'circular cylinder'],
    [D(1, -2, 0), 1, 'hypcyl', 'hyperbolic cylinder'],
    [D(2, 0, 0), 1, 'planes2', 'two parallel planes'],
    [D(-1, -2, -3), 1, 'empty', 'no real points'],
    [D(-1, -2, 0), 1, 'empty', 'no real points'],
    [D(0, 0, 0), 1, 'empty', 'no real points'],
    [D(1, 2, 3), -1, 'empty', 'no real points'],
    [D(-1, -2, -3), -1, 'ellipsoid', 'ellipsoid'],
    [D(1, 2, -1), -1, 'hyperboloid2', 'hyperboloid of two sheets'],
    [D(1, 2, -1), 0, 'cone', 'elliptic cone'],
    [D(1, 1, -1), 0, 'cone', 'circular cone'],
    [D(-1, -1, 2), 0, 'cone', 'circular cone'],
    [D(1, 2, 3), 0, 'point', 'just the origin'],
    [D(1, 2, 0), 0, 'line', 'a line (counted twice)'],
    [D(1, -2, 0), 0, 'xplanes', 'two intersecting planes'],
    [D(0, -2, 0), 0, 'plane', 'a plane (counted twice)'],
    [D(0, 0, 0), 0, 'all', 'every point of space'],
    [D(1, 4), 1, 'ellipse', 'ellipse'],
    [D(2, 2), 1, 'ellipse', 'circle'],
    [D(1, -4), 1, 'hyperbola', 'hyperbola'],
    [D(1, 0), 1, 'lines2', 'two parallel lines'],
    [D(-1, -1), 1, 'empty', 'no real points'],
    [D(1, -4), 0, 'xlines', 'two crossing lines'],
    [D(1, 0), 0, 'line', 'a line (counted twice)'],
    [D(1, 1), 0, 'point', 'just the origin'],
    [D(0, 0), 0, 'all', 'every point of the plane'],
  ];
  for (const [M, c, kind, name] of cases) {
    const q = quadricInfo(M, c);
    assert.equal(q.kind, kind, `${JSON.stringify(M)} c=${c}`);
    assert.equal(q.name, name, `${JSON.stringify(M)} c=${c}`);
    assert.ok(q.symmetric);
  }
});

test('quadricInfo symmetrises and scales the principal axes', () => {
  const q = quadricInfo([[2, 2], [0, 2]], 1);
  assert.equal(q.symmetric, false);
  assert.deepEqual(q.S, [[2, 1], [1, 2]]);
  assert.deepEqual(q.lams, [3, 1]);
  near(q.axes[0].len, 1 / Math.sqrt(3));
  near(q.axes[1].len, 1);
  near(Math.abs(q.axes[0].dir[0]), Math.SQRT1_2);
  const h = quadricInfo(D(1, 4, -1), 4);
  assert.deepEqual(h.axes.map((a) => a.real), [true, true, false]);
  nearArr(h.axes.map((a) => a.len), [1, 2, 2]);
  const z = quadricInfo(D(1, 0, 0), 1);
  assert.equal(z.axes[1].len, null);
  near(quadricInfo(D(4, 1, -1), 0).axes[0].len, 0.5);
});

test('quadric patches lie on xᵀSx = c', () => {
  const rot = (t) => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];
  const conj = (Q, L) => {
    const QL = Q.map((r) => r.map((x, j) => x * L[j]));
    return QL.map((r) => Q.map((s) => dot(r, s)));
  };
  const Q = rot(0.7).map((r, i) => r.map((x, j) => (j === 2 && i === 2 ? 1 : x)));
  const cases = [
    [[1, 2, 3], 1], [[2, 1, -1], 1], [[1, -2, -1], 2], [[1, 2, 0], 1], [[1, -2, 0], 1], [[2, 0, 0], 1],
    [[1, 2, -1], 0], [[1, -3, 0], 0], [[0, 2, 0], 0], [[-1, -2, -3], -1],
  ];
  for (const [L, c] of cases) {
    const S = conj(Q, L), info = quadricInfo(S, c), patches = quadricPatches(info, 10);
    assert.ok(patches.length >= 1, `${L} c=${c} (${info.kind})`);
    for (const p of patches) {
      assert.equal(p.pts.length, (p.rows + 1) * (p.cols + 1));
      for (const x of p.pts) {
        const r = form(info.S, x), scale = Math.max(1, dot(x, x));
        near(r, c, 1e-8 * scale);
      }
    }
  }
  // coverage: the one-sheet hyperboloid reaches the covering radius along its axis
  const hp = quadricPatches(quadricInfo(D(1, 1, -1), 1), 10)[0];
  assert.ok(Math.max(...hp.pts.map((x) => Math.abs(x[2]))) >= 10 - 1e-9);
  assert.deepEqual(quadricPatches(quadricInfo(D(1, 2, 3), -1), 10), []);
});

test('conic curves lie on the conic; clipping keeps runs inside the box', () => {
  for (const [M, c] of [[[[2, 1], [1, 2]], 1], [[[1, 2], [2, -2]], 1], [[[1, 0], [0, 0]], 4], [[[1, 2], [2, -2]], 0], [[[0, 0], [0, 3]], 0]]) {
    const info = quadricInfo(M, c), curves = conicCurves(info, 12);
    assert.ok(curves.length >= 1, info.kind);
    for (const curve of curves) {
      for (const x of curve) near(form(info.S, x), c, 1e-8 * Math.max(1, dot(x, x)));
      for (const run of clipPolyline(curve, 6)) {
        assert.ok(run.length >= 2);
        for (const x of run) assert.ok(Math.abs(x[0]) <= 6 + 1e-9 && Math.abs(x[1]) <= 6 + 1e-9);
      }
    }
  }
  const runs = clipPolyline([[-10, 0, 0], [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 1, 0]], 5);
  assert.equal(runs.length, 2);
  nearArr(runs[0][0], [-5, 0, 0]);
  nearArr(runs[0].at(-1), [5, 0, 0]);
  assert.equal(runs[0].length, 3);
  nearArr(runs[1].at(-1), [0, 1, 0]);
  assert.deepEqual(clipPolyline([[7, 7, 0], [8, 9, 0]], 5), []);
});

test('language: flow, iterate, power, quadric evaluate and reject bad input', () => {
  const rows = evaluate([
    'A = [[1, 1], [0, -1]]', 'B = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]', 'v = (0, 1)', 'w = (1, 1, 1)', 'n = 2.5',
    'flow(A)', 'x = iterate(A, v, n)', 'y = power(B, w, 20)', 'quadric(B)', 'quadric(A, -2)', 'x + v',
  ]);
  for (const r of rows) assert.equal(r.error, null);
  assert.equal(rows[5].value.type, 'fl-flow');
  assert.equal(rows[5].value.info.kind, 'saddle');
  assert.deepEqual(rows[6].value.v, [0, 1, 0]); // A² v with n = 2.5 -> step 2, partial third
  assert.deepEqual(rows[6].value.fl.next, [1, -1, 0]);
  assert.equal(rows[6].value.fl.frac, 0.5);
  assert.equal(rows[7].value.type, 'vec');
  near(len(rows[7].value.v), Math.sqrt(3), 1e-12);
  assert.equal(rows[8].value.info.kind, 'ellipsoid');
  assert.equal(rows[9].value.info.symmetric, false);
  assert.deepEqual(rows[10].value, { type: 'vec', v: [0, 2, 0] });

  const errs = evaluate([
    'flow([[1, 2, 3], [4, 5, 6]])', 'flow((1, 2, 3))', 'iterate([[1, 0], [0, 1]], (1, 1, 1), 3)',
    'iterate([[1, 0], [0, 1]], (1, 1), (1, 2))', 'power([[1, 0], [0, 1]], (0, 0), 3)', 'quadric([[1, 0], [0, 1]], (1, 1))',
    'iterate([[1, 0], [0, 1]], (1, 1))', 'flow([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])',
  ]).map((r) => r.error);
  assert.match(errs[0], /2×2 or 3×3/);
  assert.match(errs[1], /matrix first/);
  assert.match(errs[2], /nonzero z/);
  assert.match(errs[3], /step count/);
  assert.match(errs[4], /nonzero starting vector/);
  assert.match(errs[5], /number c/);
  assert.match(errs[6], /needs 3 arguments/);
  assert.match(errs[7], /2×2 or 3×3/);
  // n is clamped: negative -> 0, huge -> the cap
  const [neg] = evaluate(['iterate([[2, 0], [0, 2]], (1, 0), -3)']);
  assert.deepEqual(neg.value.v, [1, 0, 0]);
  const [big] = evaluate(['power([[2, 0], [0, 1]], (1, 1), 1e9)']);
  assert.equal(big.error, null);
});

test('readouts are valid KaTeX and say the important thing', () => {
  const rows = evaluate([
    'A = [[1, 1], [0, -1]]', 'S = [[0, -2], [2, 0]]', 'v = (1, 2)', 'M = [[1, 2, 0], [0, 3, 0], [0, 0, -1]]',
    'flow(A)', 'flow(S)', 'flow([[10, 0, 0], [0, -10, 0], [0, 0, 1]])', 'iterate(A, v, 3)', 'power(A, v, 5)',
    'power(S, v, 5)', 'quadric(M)', 'quadric(M, 0)', 'quadric([[-1, 0], [0, -1]])', 'power([[1, 1], [1, 1]], (1, -1), 3)',
  ]);
  const src = ['', '', '', '', 'flow(A)', 'flow(S)', '', 'iterate(A, v, 3)', 'power(A, v, 5)', 'power(S, v, 5)', 'quadric(M)', 'quadric(M, 0)', '', ''];
  const all = rows.map((r, i) => readoutLines(r.value, { args: callArgs(src[i]) }));
  for (const lines of all.slice(4)) for (const t of lines) valid(t);
  assert.equal(all[0], null);
  assert.match(all[4][0], /saddle/);
  assert.match(all[4][1], /\\lambda = 1,\\ -1/);
  assert.match(all[5][0], /center/);
  assert.match(all[5][1], /\\pm 2i/);
  assert.match(all[6][1], /shown at \} 0\.18\\times/);
  assert.match(all[6][0], /2 stable|1 stable/);
  assert.match(all[7][0], /^A\^\{3\}\\vec\{v\} = \(/);
  assert.match(all[8][1], /\\vec q_\{5\}\^\{\\mathsf T\}A/);
  assert.match(all[9].join(' '), /complex/);
  assert.match(all[10][0], /hyperboloid of one sheet/);
  assert.match(all[10][2], /not symmetric/);
  assert.match(all[11][0], /cone/);
  assert.match(all[12][0], /no real points/);
  assert.match(all[13].join(' '), /stopped/);
});

test('helpers: eigenvalue text, call arguments, colours', () => {
  assert.equal(eigTex([{ re: 2, im: 0 }, { re: -1, im: 0 }]), '2,\\ -1');
  assert.equal(eigTex([{ re: 0, im: 1 }, { re: 0, im: -1 }]), '\\pm i');
  assert.equal(eigTex([{ re: -0.5, im: 2 }, { re: -0.5, im: -2 }, { re: 3, im: 0 }]), '-0.5 \\pm 2i,\\ 3');
  assert.deepEqual(callArgs('q = quadric(M, c)'), ['M', 'c']);
  assert.deepEqual(callArgs('iterate([[1,0],[0,1]], v, 3)'), [null, 'v', null]);
  assert.deepEqual(callArgs('w = u + v'), []);
  assert.deepEqual(callArgs('dot(u, v)'), []);
  assert.equal(mixHex('#000000', '#ffffff', 0.5), '#808080');
  assert.equal(mixHex(0x102030, '#102030', 0.3), '#102030');
  const t = tones('#e05a4f', 3, 'light');
  assert.equal(t.length, 3);
  assert.equal(new Set(t).size, 3);
  for (const c of t) assert.match(c, /^#[0-9a-f]{6}$/);
});

test('tones read on both themes; step sliders get a 0..n range', () => {
  for (const theme of ['dark', 'light']) {
    for (const base of ['#e05a4f', '#4a90e2', '#43b05c', '#9b6ade', '#f5a623', '#26b5b5', '#e056a0', '#a1887f']) {
      for (const n of [2, 3]) {
        for (const c of tones(base, n, theme)) {
          const y = luminance(c), contrast = theme === 'dark' ? (y + 0.05) / 0.0655 : 1.01 / (y + 0.05);
          assert.ok(contrast >= 3.5, `${base} -> ${c} on ${theme}: ${contrast.toFixed(2)}`);
        }
      }
    }
  }
  const srcs = ['A = [[2,1],[1,2]]', 'v = (1, 0)', 'n = 3', 'k = 1', 'iterate(A, v, n)', 'power(A, v, k)', 'iterate(A, v, 2)'];
  const ranges = stepRanges(srcs, evaluate(srcs));
  assert.deepEqual([...ranges.entries()].sort(), [[2, 8], [3, 20]]);
});
