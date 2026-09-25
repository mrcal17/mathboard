import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, valueReadout, isDrawable } from '../static/graph/lang.js';

const rows = (...lines) => evaluate(lines);
const last = (...lines) => evaluate(lines).at(-1);
const ok = (...lines) => {
  const r = last(...lines);
  assert.equal(r.error, null, `unexpected error: ${r.error}`);
  return r.value;
};
const err = (...lines) => {
  const r = last(...lines);
  assert.notEqual(r.error, null, `expected an error, got ${JSON.stringify(r.value)}`);
  return r.error;
};
const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('a free x makes a curve, free x and y a surface', () => {
  const g = ok('x^2');
  assert.equal(g.type, 'graph');
  assert.deepEqual([g.mode, g.ins, g.dep], ['curve', ['x'], 'y']);
  assert.equal(g.at({ x: 3 }), 9);
  const s = ok('x^2 - y^2');
  assert.deepEqual([s.mode, s.ins, s.dep], ['surface', ['x', 'y'], 'z']);
  assert.equal(s.at({ x: 2, y: 1 }), 3);
  assert.ok(isDrawable(g) && isDrawable(s));
});

test('y = ..., z = ... and x = ... are equations to graph, not definitions', () => {
  const g = ok('y = 2x + 1');
  assert.deepEqual([g.mode, g.dep], ['curve', 'y']);
  assert.equal(g.at({ x: 1 }), 3);
  const s = ok('z = sin(x) cos(y)');
  assert.deepEqual([s.mode, s.ins, s.dep], ['surface', ['x', 'y'], 'z']);
  const c = ok('x = y^2');
  assert.deepEqual([c.mode, c.ins, c.dep], ['curve', ['y'], 'x']);
  // several graphs of y are fine: none of them defines y
  const r = rows('y = x', 'y = -x', 'y = x^2');
  assert.ok(r.every((row) => row.error === null));
  assert.match(err('y = y + 1'), /both sides/);
  assert.match(err('x + y + z'), /z is the height/);
});

test('defined coordinates stay values', () => {
  const r = rows('A = [[1, 2], [3, 4]]', 'x = (1, 1, 0)', 'y = A x', 'y');
  assert.deepEqual(r[2].value, { type: 'vec', v: [3, 7, 0] });
  assert.deepEqual(r[3].value, { type: 'vec', v: [3, 7, 0] });
  assert.equal(ok('x = 2', 'y = x + 1'), 3);
  assert.equal(ok('y = 3'), 3);
  // a slider scales the graph
  const g = ok('a = 2', 'y = a x^2');
  assert.equal(g.at({ x: 3 }), 18);
});

test('rows built from a graph are graphs of the same coordinates', () => {
  const g = ok('a = x^2', 'a + 1');
  assert.equal(g.mode, 'curve');
  assert.equal(g.at({ x: 2 }), 5);
});

test('f(x) = ... defines a function that other rows can call and draws it', () => {
  const r = rows('f(x) = x^3 - x', 'f(2)', 'g(t) = f(t) + 1', 'g(0)', 'h(x, y) = x y', 'h(2, 3)');
  assert.equal(r[0].value.mode, 'curve');
  assert.equal(r[1].value, 6);
  assert.equal(r[3].value, 1);
  assert.equal(r[4].value.mode, 'surface');
  assert.equal(r[5].value, 6);
  assert.match(err('f(x) = x^2', 'f(1, 2)'), /f needs 1 argument/);
  assert.match(err('f(t) = t + x'), /f uses x/);
  assert.match(err('sin(x) = 2'), /built-in/);
  assert.match(err('f(x, x) = 1'), /listed twice/);
  assert.equal(ok('f(x) = 2', 'f(5)'), 2); // a constant function, not a slider
  assert.equal(last('f(x) = 2').slider, null);
  // a vector of one parameter is a parametric curve
  assert.equal(ok('c(t) = (cos(t), sin(t), t / 4)').mode, 'param');
});

test('derivatives with primes', () => {
  near(ok("sigmoid'(0)"), 0.25);
  near(ok("f(x) = x^3", "f'(2)"), 12, 1e-4);
  near(ok("f(x) = x^3", "f''(2)"), 12, 1e-3);
  const g = ok("relu'");
  assert.equal(g.mode, 'curve');
  near(g.at({ x: 3 }), 1);
  near(g.at({ x: -3 }), 0);
  assert.match(err("u = (1, 2, 3)", "u'"), /goes after a function/);
});

test('activations', () => {
  near(ok('sigmoid(0)'), 0.5);
  near(ok('σ(2)'), 1 / (1 + Math.exp(-2)));
  near(ok('sigmoid(-800)'), 0);
  near(ok('tanh(1)'), Math.tanh(1));
  assert.equal(ok('relu(-2)'), 0);
  near(ok('leakyrelu(-2)'), -0.02);
  near(ok('leakyrelu(-2, 0.1)'), -0.2);
  near(ok('softplus(0)'), Math.log(2));
  near(ok('softplus(800)'), 800);
  near(ok('gelu(1)'), 0.8413447 * 1, 1e-6);
  near(ok('silu(1)'), 1 / (1 + Math.exp(-1)));
  near(ok('elu(-1)'), Math.exp(-1) - 1);
  assert.equal(ok('mod(-1, 3)'), 2);
  near(ok('logsumexp((0, 0, 0))'), Math.log(3));
});

test('a bare activation name draws it, with its formula as the readout', () => {
  const g = ok('sigmoid');
  assert.deepEqual([g.type, g.mode, g.fname], ['graph', 'curve', 'sigmoid']);
  near(g.at({ x: 0 }), 0.5);
  assert.match(valueReadout(g).latex, /\\sigma\(x\) = \\frac\{1\}\{1 \+ e\^\{-x\}\}/);
  assert.match(valueReadout(ok("sigmoid'")).latex, /\\sigma'\(x\) = \\sigma\(x\)/);
  assert.deepEqual(valueReadout(ok('x^2')), {});
});

test('softmax: a probability vector, or the map onto the triangle', () => {
  const p = ok('softmax((1, 2, 3))');
  assert.equal(p.type, 'vec');
  near(p.v[0] + p.v[1] + p.v[2], 1);
  assert.ok(p.v[2] > p.v[1] && p.v[1] > p.v[0]);
  const sharp = ok('softmax((1, 2, 3), 0.1)');
  assert.ok(sharp.v[2] > 0.99);
  assert.deepEqual(ok('softmax'), { type: 'softmaxmap', T: 1, extentPoints: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
  assert.equal(ok('T = 0.5', 'softmax(T)').T, 0.5);
  assert.match(err('softmax(-1)'), /positive/);
  assert.match(valueReadout(ok('softmax')).latex, /softmax/);
});

test('sampling errors do not fail a graph, but a row that fails everywhere does', () => {
  const g = ok('1 / x');
  assert.throws(() => g.at({ x: 0 }), /division by zero/);
  assert.equal(g.at({ x: 2 }), 0.5);
  assert.equal(ok('sqrt(x)').mode, 'curve');
  assert.match(err('x + [1, 2, 3]'), /can't add/);
});

test('origins can still use values but not free coordinates', () => {
  assert.match(err('(1, 0, 0) @ x'), /x is not defined/);
  assert.deepEqual(last('p = (1, 2, 3)', '(1, 0, 0) @ p').origin, [1, 2, 3]);
});
