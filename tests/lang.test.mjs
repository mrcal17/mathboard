import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, formatNumber, formatValue } from '../static/graph/lang.js';

// Evaluate lines and return the last row (earlier lines act as context).
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
const V = (x, y, z) => ({ type: 'vec', v: [x, y, z] });
const P = (x, y, z) => ({ type: 'point', v: [x, y, z] });
const EMPTY = { name: null, value: null, origin: null, error: null, slider: null };

test('blank lines and comments produce empty rows', () => {
  const rows = evaluate(['', '   ', '# note', '  // note', '#', '//u = 2']);
  for (const r of rows) assert.deepEqual(r, EMPTY);
});

test('trailing comments are ignored', () => {
  const r = last('a = 2 # the scale');
  assert.equal(r.value, 2);
  assert.equal(r.slider, 2);
  assert.deepEqual(val('u = (1, 2, 3) // arrow'), V(1, 2, 3));
});

test('number literals', () => {
  assert.equal(val('2'), 2);
  assert.equal(val('2.5'), 2.5);
  assert.equal(val('.5'), 0.5);
  assert.equal(val('1e-3'), 0.001);
  assert.equal(val('1.5E2'), 150);
});

test('vector literals', () => {
  assert.deepEqual(val('(1, 2, 3)'), V(1, 2, 3));
  assert.deepEqual(val('[1, 2, 3]'), V(1, 2, 3));
  assert.deepEqual(val('(1, 2)'), V(1, 2, 0));
  assert.deepEqual(val('[4, 5]'), V(4, 5, 0));
  assert.deepEqual(val('(1+1, 2*3, -1)'), V(2, 6, -1));
  assert.equal(val('(2)'), 2);
  assert.equal(val('((3))'), 3);
  assert.match(err('(1, 2, 3, 4)'), /2 or 3 components/);
  assert.match(err('[5]'), /2 or 3 components/);
  assert.match(err('u = (1,0,0)', '(u, 1)'), /components must be numbers, got a vector/);
});

test('matrix literals', () => {
  assert.deepEqual(val('[[1, 2], [3, 4]]'), { type: 'mat', m: [[1, 2], [3, 4]] });
  assert.deepEqual(val('[[1, 2, 3]]'), { type: 'mat', m: [[1, 2, 3]] });
  assert.deepEqual(val('[[1,0,0],[0,1,0],[0,0,1]]').m, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  assert.deepEqual(val('[[1, 2], [3, 4], [5, 6]]').m, [[1, 2], [3, 4], [5, 6]]);
  assert.match(err('[[1, 2], [3]]'), /same length/);
  assert.match(err('[[1, 2], (3, 4)]'), /unexpected '\('/);
});

test('precedence and associativity', () => {
  assert.equal(val('1 + 2 * 3'), 7);
  assert.equal(val('(1 + 2) * 3'), 9);
  assert.equal(val('10 - 4 - 3'), 3);
  assert.equal(val('8 / 4 / 2'), 1);
  assert.equal(val('-2^2'), -4);
  assert.equal(val('2^3^2'), 512);
  assert.equal(val('2^-1'), 0.5);
  assert.equal(val('2 * -3'), -6);
  assert.equal(val('--2'), 2);
  assert.equal(val('+3'), 3);
  assert.equal(val('(1, 2, 3).y'), 2);
  assert.equal(val('u = (3, 4, 5)', '-u.x'), -3);
  assert.equal(val('u = (3, 4, 5)', 'u.x^2'), 9);
  assert.equal(val('u = (3, 4, 5)', 'u.z * 2 + u.y'), 14);
});

test('implicit multiplication', () => {
  const ctx = ['u = (1, 0, 0)', 'v = (0, 2, 0)', 'a = 3', 'b = 2'];
  assert.deepEqual(val(...ctx, '2u'), V(2, 0, 0));
  assert.deepEqual(val(...ctx, 'a u'), V(3, 0, 0));
  assert.deepEqual(val(...ctx, '2(1, 0, 0)'), V(2, 0, 0));
  assert.deepEqual(val(...ctx, 'a(u + v)'), V(3, 6, 0));
  assert.deepEqual(val(...ctx, '(u + v)b'), V(2, 4, 0));
  assert.equal(val(...ctx, 'a b'), 6);
  assert.equal(val(...ctx, '2 3'), 6);
  near(val('2pi'), 2 * Math.PI);
  near(val('2π'), 2 * Math.PI);
  assert.equal(val(...ctx, '2|v|'), 4);
  assert.equal(val(...ctx, '|u| |v|'), 2);
  assert.equal(val(...ctx, '2a^2'), 18); // ^ binds tighter than juxtaposition
  assert.equal(val(...ctx, 'a/2b'), 3); // same level as / : (a/2)b
});

test('implicit multiplication binds to the right of · and ×', () => {
  const ctx = ['u = (1, 0, 0)', 'v = (0, 1, 0)', 'w = (1, 2, 3)'];
  assert.deepEqual(val(...ctx, 'u × 2v'), V(0, 0, 2));
  assert.equal(val(...ctx, 'u · 3w'), 3);
  assert.deepEqual(val(...ctx, '2u × v'), V(0, 0, 2));
  // projection formula keeps / and implicit product left-to-right
  assert.deepEqual(val(...ctx, '(w·u)/(u·u) u'), V(1, 0, 0));
});

test('function name followed by ( is a call, other names multiply', () => {
  assert.equal(val('sin(0)'), 0);
  assert.equal(val('2sin(0) + 1'), 1);
  assert.deepEqual(val('a = 2', 'a(1, 2, 3)'), V(2, 4, 6));
  assert.match(err('sin'), /sin is a function/);
  assert.match(err('u = (1,0,0)', 'det u'), /det is a function/);
});

test('absolute value and length', () => {
  assert.equal(val('|-2|'), 2);
  assert.equal(val('|(3, 4, 0)|'), 5);
  assert.equal(val('u = (1, 2, 2)', '|2u - u|'), 3);
  assert.equal(val('||-3||'), 3);
  assert.match(err('|[[1, 2], [3, 4]]|'), /det\(A\)/);
  assert.match(err('|point(1, 2, 3)|'), /got a point/);
  assert.match(err('|2'), /missing '\|'/);
});

test('addition and subtraction by type', () => {
  const ctx = ['u = (1, 2, 3)', 'p = point(1, 1, 1)', 'q = point(0, 1, 2)'];
  assert.deepEqual(val(...ctx, 'u + (1, 1, 1)'), V(2, 3, 4));
  assert.deepEqual(val(...ctx, 'u - (1, 1, 1)'), V(0, 1, 2));
  assert.deepEqual(val(...ctx, 'p + u'), P(2, 3, 4));
  assert.deepEqual(val(...ctx, 'u + p'), P(2, 3, 4));
  assert.deepEqual(val(...ctx, 'p - u'), P(0, -1, -2));
  assert.deepEqual(val(...ctx, 'p - q'), V(1, 0, -1));
  assert.match(err(...ctx, 'p + q'), /two points/);
  assert.match(err(...ctx, 'u - p'), /subtract a point from a vector/);
  assert.match(err(...ctx, '2 + u'), /can't add a number and a vector/);
  assert.match(err(...ctx, 'u - 1'), /can't subtract a number from a vector/);
  assert.deepEqual(val('[[1, 2], [3, 4]] + [[1, 1], [1, 1]]').m, [[2, 3], [4, 5]]);
  assert.deepEqual(val('[[1, 2], [3, 4]] - [[1, 1], [1, 1]]').m, [[0, 1], [2, 3]]);
  assert.match(err('[[1, 2], [3, 4]] + [[1, 2, 3]]'), /2×2 matrix and a 1×3 matrix/);
});

test('multiplication and division by type', () => {
  const A = 'A = [[1, 2], [3, 4]]';
  const R = 'R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]';
  assert.equal(val('3 * 4'), 12);
  assert.deepEqual(val('2 * (1, 2, 3)'), V(2, 4, 6));
  assert.deepEqual(val('(1, 2, 3) * 2'), V(2, 4, 6));
  assert.deepEqual(val(A, '2A').m, [[2, 4], [6, 8]]);
  assert.deepEqual(val(A, 'A * 2').m, [[2, 4], [6, 8]]);
  assert.deepEqual(val(R, 'R (1, 0, 0)'), V(0, 1, 0));
  assert.deepEqual(val(A, 'A * (1, 1)'), V(3, 7, 0));
  assert.deepEqual(val(A, 'A * A').m, [[7, 10], [15, 22]]);
  assert.equal(val('[[1, 2, 3]] * [[1], [1], [1]]'), 6); // 1×1 products are numbers
  assert.match(err('(1, 2, 3) * (1, 2, 3)'), /dot\(u, v\).*cross\(u, v\)/);
  assert.match(err('u = (1,0,0)', 'u u'), /dot\(u, v\)/);
  assert.match(err(A, '(1, 1) A'), /put the matrix first/);
  assert.match(err(A, 'A (1, 1, 1)'), /nonzero z/);
  assert.match(err(A, 'A [[1, 2, 3]]'), /2 columns vs 1 rows/);
  assert.equal(val('7 / 2'), 3.5);
  assert.deepEqual(val('(2, 4, 6) / 2'), V(1, 2, 3));
  assert.deepEqual(val(A, 'A / 2').m, [[0.5, 1], [1.5, 2]]);
  assert.match(err('1 / 0'), /division by zero/);
  assert.match(err('2 / (1, 2, 3)'), /divide by a vector/);
  assert.match(err('2 * point(1, 2, 3)'), /multiply a number by a point/);
});

test('dot and cross operators', () => {
  const ctx = ['u = (1, 2, 3)', 'v = (4, 5, 6)'];
  assert.equal(val(...ctx, 'u · v'), 32);
  assert.equal(val(...ctx, 'u ⋅ v'), 32);
  assert.deepEqual(val(...ctx, 'u × v'), V(-3, 6, -3));
  assert.deepEqual(val('i × j'), V(0, 0, 1));
  assert.equal(val('2 · 3'), 6);
  assert.equal(val('2 × 3'), 6);
  assert.deepEqual(val(...ctx, '2 × u'), V(2, 4, 6));
  assert.equal(val(...ctx, 'u · v + 1'), 33);
  assert.equal(val('3 − 1'), 2);
  assert.equal(err(...ctx, 'u · point(1, 2, 3)'), '· needs two vectors, got a point');
  assert.equal(err('norm([[1, 2], [3, 4]])'), 'norm needs a number or vector, got a 2×2 matrix');
});

test('powers', () => {
  assert.equal(val('3^2'), 9);
  assert.deepEqual(val('A = [[1, 1], [0, 1]]', 'A^3').m, [[1, 3], [0, 1]]);
  assert.deepEqual(val('A = [[2, 0], [0, 4]]', 'A^0').m, [[1, 0], [0, 1]]);
  assert.deepEqual(val('A = [[2, 0], [0, 4]]', 'A^-1').m, [[0.5, 0], [0, 0.25]]);
  assert.match(err('A = [[2, 0], [0, 4]]', 'A^0.5'), /whole numbers/);
  assert.match(err('[[1, 2, 3]]^2'), /square/);
  assert.match(err('(1, 2, 3)^2'), /\|u\|\^2/);
  assert.match(err('(-8)^(1/3)'), /fractional power/);
  assert.match(err('0^-1'), /division by zero/);
  assert.match(err('[[1, 2], [2, 4]]^-1'), /not invertible/);
});

test('builtin constants and overriding them', () => {
  near(val('pi'), Math.PI);
  near(val('π'), Math.PI);
  near(val('e'), Math.E);
  assert.deepEqual(val('i'), V(1, 0, 0));
  assert.deepEqual(val('j'), V(0, 1, 0));
  assert.deepEqual(val('k'), V(0, 0, 1));
  assert.deepEqual(val('2i + 3j - k'), V(2, 3, -1));
  assert.equal(val('e = 2', 'e + 1'), 3);
  assert.equal(val('i = 5', 'i'), 5);
  assert.equal(val('pi = 3', '2pi'), 6);
});

test('vector functions', () => {
  const ctx = ['u = (1, 2, 2)', 'v = (1, 0, 0)', 'w = (0, 1, 0)'];
  assert.equal(val(...ctx, 'dot(u, v)'), 1);
  assert.deepEqual(val(...ctx, 'cross(v, w)'), V(0, 0, 1));
  assert.equal(val(...ctx, 'norm(u)'), 3);
  assert.equal(val(...ctx, 'length(u)'), 3);
  assert.equal(val(...ctx, 'mag(u)'), 3);
  nearArr(val(...ctx, 'unit(u)').v, [1 / 3, 2 / 3, 2 / 3]);
  nearArr(val(...ctx, 'normalize(u)').v, [1 / 3, 2 / 3, 2 / 3]);
  assert.deepEqual(val(...ctx, 'proj(u, v)'), V(1, 0, 0));
  assert.deepEqual(val(...ctx, 'proj(u, 2w)'), V(0, 2, 0));
  near(val(...ctx, 'angle(v, w)'), Math.PI / 2);
  near(val(...ctx, 'angle(v, v)'), 0);
  near(val(...ctx, 'deg(angle(v, (1, 1, 0)))'), 45);
  assert.equal(val(...ctx, 'cross(v, w).z'), 1);
  assert.match(err('unit((0, 0, 0))'), /zero vector/);
  assert.match(err('proj((1, 2, 3), (0, 0, 0))'), /zero vector/);
  assert.match(err('angle((1, 2, 3), (0, 0, 0))'), /zero vector/);
});

test('numeric functions', () => {
  near(val('deg(pi)'), 180);
  near(val('rad(180)'), Math.PI);
  near(val('sin(pi/2)'), 1);
  near(val('cos(0)'), 1);
  near(val('tan(pi/4)'), 1);
  near(val('asin(1)'), Math.PI / 2);
  near(val('acos(0)'), Math.PI / 2);
  near(val('atan(1)'), Math.PI / 4);
  assert.equal(val('sqrt(16)'), 4);
  assert.equal(val('abs(-3)'), 3);
  assert.equal(val('abs((0, 3, 4))'), 5);
  near(val('exp(1)'), Math.E);
  near(val('ln(e)'), 1);
  assert.equal(val('log(1000)'), 3);
  assert.equal(val('min(3, 1, 2)'), 1);
  assert.equal(val('max(3, 1, 2)'), 3);
  assert.equal(val('sin(0)^2 + cos(0)^2'), 1);
  assert.match(err('sqrt(-1)'), /negative/);
  assert.match(err('ln(0)'), /positive/);
  assert.match(err('log(-2)'), /positive/);
  assert.match(err('asin(2)'), /between -1 and 1/);
  assert.match(err('exp(1000)'), /too large/);
});

test('matrix functions', () => {
  assert.equal(val('det([[1, 2], [3, 4]])'), -2);
  assert.equal(val('det([[2, 0, 0], [0, 3, 0], [0, 0, 4]])'), 24);
  assert.equal(val('det([[5]])'), 5);
  near(val('det([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 2, 1], [0, 0, 1, 1]])'), -2);
  assert.deepEqual(val('inv([[2, 0], [0, 4]])').m, [[0.5, 0], [0, 0.25]]);
  const I = val('A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]', 'inv(A) A').m;
  I.forEach((row, r) => row.forEach((x, c) => near(x, r === c ? 1 : 0)));
  assert.deepEqual(val('transpose([[1, 2, 3], [4, 5, 6]])').m, [[1, 4], [2, 5], [3, 6]]);
  assert.deepEqual(val('matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))').m, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
  assert.deepEqual(val('matrix((1, 2), (3, 4))').m, [[1, 3], [2, 4]]);
  assert.deepEqual(val('matrix((1, 2, 1), (3, 4, 0))').m, [[1, 3], [2, 4], [1, 0]]);
  assert.equal(val('det(matrix(i, j, k))'), 1);
  assert.match(err('det([[1, 2, 3]])'), /square matrix, got a 1×3 matrix/);
  assert.match(err('inv([[1, 2], [2, 4]])'), /not invertible/);
  assert.match(err('inv([[1, 2, 3], [4, 5, 6], [7, 8, 9]])'), /not invertible/);
  assert.match(err('det((1, 2, 3))'), /det needs a matrix, got a vector/);
  assert.match(err('matrix((1, 2, 3))'), /matrix needs 2 to 3 arguments, got 1/);
});

test('points and shapes', () => {
  assert.deepEqual(val('point(1, 2, 3)'), P(1, 2, 3));
  assert.deepEqual(val('point(1, 2)'), P(1, 2, 0));
  assert.deepEqual(val('point((4, 5, 6))'), P(4, 5, 6));
  assert.equal(val('point(1, 2, 3).z'), 3);
  assert.match(err('point(2)'), /point needs/);
  assert.match(err('point((1, 0, 0), 2)'), /point needs numbers, got a vector/);

  assert.deepEqual(val('span((1, 0, 0))'), { type: 'span', vecs: [[1, 0, 0]] });
  assert.deepEqual(val('span(i, j)').vecs, [[1, 0, 0], [0, 1, 0]]);
  assert.deepEqual(val('span(i, 2i)').vecs, [[1, 0, 0]]); // dependent vectors dropped
  assert.deepEqual(val('span(i, j, i + j)').vecs, [[1, 0, 0], [0, 1, 0]]);
  assert.equal(val('span(i, j, k)').vecs.length, 3);
  assert.equal(val('span((0, 0, 0))').vecs.length, 0);
  assert.deepEqual(val('plane((0, 0, 2))'), { type: 'plane', normal: [0, 0, 2] });
  assert.deepEqual(val('parallelogram(i, j)'), { type: 'parallelogram', u: [1, 0, 0], v: [0, 1, 0] });
  assert.deepEqual(val('parallelepiped(i, j, k)'),
    { type: 'parallelepiped', u: [1, 0, 0], v: [0, 1, 0], w: [0, 0, 1] });

  assert.match(err('plane((0, 0, 0))'), /nonzero/);
  assert.match(err('span(point(1, 2, 3))'), /span needs vectors, got a point/);
  assert.match(err('span(i, j, k, i)'), /1 to 3 arguments, got 4/);
  assert.match(err('parallelogram(i)'), /parallelogram needs 2 arguments, got 1/);
  assert.match(err('span(i) + j'), /can't add a span and a vector/);
});

test('argument count and type checks', () => {
  assert.equal(err('dot((1, 2, 3))'), 'dot needs 2 arguments, got 1');
  assert.equal(err('sin(1, 2)'), 'sin needs 1 argument, got 2');
  assert.equal(err('cross()'), 'cross needs 2 arguments, got 0');
  assert.equal(err('min()'), 'min needs at least 1 argument, got 0');
  assert.equal(err('sqrt((1, 2, 3))'), 'sqrt needs a number, got a vector');
  assert.equal(err('dot(1, 2)'), 'dot needs vectors, got a number');
  assert.equal(err('max(1, (1, 2))'), 'max needs numbers, got a vector');
  assert.equal(err('transpose(3)'), 'transpose needs a matrix or vector, got a number');
});

test('definitions are order independent', () => {
  const rows = evaluate(['c = a + b', 'a = 2', 'b = a * 10']);
  assert.deepEqual(rows.map((r) => r.value), [22, 2, 20]);
  assert.deepEqual(rows.map((r) => r.name), ['c', 'a', 'b']);
  assert.deepEqual(rows.map((r) => r.error), [null, null, null]);
});

test('greek and subscripted names', () => {
  near(val('θ = pi/2', 'sin(θ)'), 1);
  near(val('θ = pi/2', '2θ'), Math.PI);
  assert.equal(val('v_1 = 3', 'x2 = 4', 'v_1 x2'), 12);
});

test('circular definitions', () => {
  const rows = evaluate(['a = b + 1', 'b = 2a', 'c = a + 1', 'd = 5']);
  assert.equal(rows[0].error, 'circular definition: a → b → a');
  assert.equal(rows[1].error, 'circular definition: b → a → b');
  assert.equal(rows[2].error, 'a has an error');
  assert.equal(rows[3].value, 5);
  assert.equal(last('u = u + (1, 0, 0)').error, 'circular definition: u → u');
  const three = evaluate(['x = y', 'y = z', 'z = x']);
  assert.equal(three[2].error, 'circular definition: z → x → y → z');
});

test('duplicate definitions error on every defining row', () => {
  const rows = evaluate(['a = 1', 'b = a + 1', 'a = 2', 'c = 3']);
  assert.equal(rows[0].error, 'a is defined more than once');
  assert.equal(rows[2].error, 'a is defined more than once');
  assert.match(rows[1].error, /a is defined more than once/);
  assert.equal(rows[3].value, 3);
  assert.equal(rows[0].slider, null);
});

test('undefined names and errors in dependencies', () => {
  assert.equal(err('q + 1'), 'q is not defined');
  assert.equal(err('u = (1, 2, 3)', '2w'), 'w is not defined');
  const rows = evaluate(['a = q', 'b = a + 1']);
  assert.equal(rows[0].error, 'q is not defined');
  assert.equal(rows[1].error, 'a has an error');
  assert.equal(err('a = (1, 2', 'a'), 'a has an error');
  // names that exist on Object.prototype are not special
  assert.equal(err('toString(1)'), 'toString is not defined');
  assert.equal(val('constructor = 2', 'constructor + 1'), 3);
});

test('built-in function names cannot be redefined', () => {
  const r = last('dot = 3');
  assert.equal(r.name, 'dot');
  assert.match(r.error, /built-in function/);
  assert.equal(val('dot = 3', 'dot((1, 0, 0), (1, 0, 0))'), 1);
});

test('row shape for definitions and expressions', () => {
  const rows = evaluate(['u = (1, 2, 3)', 'u + u']);
  assert.deepEqual(rows[0], { name: 'u', value: V(1, 2, 3), origin: null, error: null, slider: null });
  assert.deepEqual(rows[1], { name: null, value: V(2, 4, 6), origin: null, error: null, slider: null });
});

test('sliders are numeric-literal definitions only', () => {
  assert.equal(last('a = 3').slider, 3);
  assert.equal(last('a = -1.5').slider, -1.5);
  assert.equal(last('a=.5').slider, 0.5);
  assert.equal(last('a = 1e-3').slider, 0.001);
  assert.equal(last('a = −2').slider, -2);
  assert.equal(last('a = 1 + 2').slider, null);
  assert.equal(last('a = --2').slider, null);
  assert.equal(last('a = pi').slider, null);
  assert.equal(last('a = (1, 2, 3)').slider, null);
  assert.equal(last('3').slider, null);
  assert.equal(last('a = 2 @ (1, 1, 1)').slider, null);
});

test('@ sets the origin without changing the value', () => {
  const rows = evaluate([
    'u = (1, 2, 3) @ (1, 0, 0)',
    'v = u @ point(0, 0, 5)',
    'w = u + v',
    '(0, 0, 1) @ u',
    'n = plane(k) @ point(0, 0, 2)',
    'u2 = (1, 0, 0) @ u2',
  ]);
  assert.deepEqual(rows[0].value, V(1, 2, 3));
  assert.deepEqual(rows[0].origin, [1, 0, 0]);
  assert.deepEqual(rows[1].origin, [0, 0, 5]);
  assert.deepEqual(rows[2].value, V(2, 4, 6));
  assert.equal(rows[2].origin, null);
  assert.deepEqual(rows[3].origin, [1, 2, 3]);
  assert.deepEqual(rows[4].origin, [0, 0, 2]);
  assert.deepEqual(rows[5].origin, [1, 0, 0]); // origin may refer to the row's own value
  assert.equal(rows[5].error, null);
});

test('@ errors', () => {
  assert.equal(err('(1, 2, 3) @ 5'), '@ needs a vector or point, got a number');
  assert.equal(err('(1, 2, 3) @'), "missing origin after '@'");
  assert.equal(err('(1, 2, 3) @ q'), 'q is not defined');
  assert.match(err('(1, 2, 3) @ (1, 0, 0) @ (0, 1, 0)'), /unexpected '@'/);
  // an origin error does not break rows that use the value
  const rows = evaluate(['u = (1, 2, 3) @ 7', 'v = 2u']);
  assert.ok(rows[0].error);
  assert.deepEqual(rows[1].value, V(2, 4, 6));
});

test('parse errors say what was unexpected', () => {
  assert.equal(err('2 +'), 'unexpected end of line');
  assert.equal(err('(1, 2'), "missing ')'");
  assert.equal(err('2 * * 3'), "unexpected '*'");
  assert.equal(err(')'), "unexpected ')'");
  assert.equal(err('(1, 2]'), "unexpected ']', expected ')'");
  assert.equal(err('a ='), "missing value after '='");
  assert.equal(err('u = (1, 2, 3) $'), "unexpected character '$'");
  assert.equal(err('2a = 3'), "only a single name can go left of '='");
  assert.equal(err('a = b = 3'), "unexpected '='");
  assert.equal(err('1, 2'), "unexpected ','");
  assert.equal(err('u = (1,0,0)', 'u.w'), "expected x, y or z after '.'");
  assert.equal(err('2.x'), '.x needs a vector or point, got a number');
  const r = last('a = (1, 2');
  assert.equal(r.name, 'a');
  assert.equal(r.value, null);
});

test('evaluate never throws on garbage', () => {
  const pieces = ['a', 'u', '=', '@', '(', ')', '[', ']', ',', '|', '.', 'x', '+', '-', '*', '/', '^',
    '·', '×', '2', '.5', 'pi', 'dot', 'span', 'det', ' ', '#', 'e', '1e3', 'i', 'k', 'point', 'π'];
  let seed = 12345;
  const rand = (n) => ((seed = (seed * 1103515245 + 12345) % 2147483648) % n);
  for (let t = 0; t < 300; t++) {
    const lines = Array.from({ length: 1 + rand(4) }, () =>
      Array.from({ length: rand(12) }, () => pieces[rand(pieces.length)]).join(''));
    const rows = evaluate(lines);
    assert.equal(rows.length, lines.length);
    for (const r of rows) {
      assert.ok(r.error === null || typeof r.error === 'string');
      assert.ok(r.error === null || r.value === null || r.origin === null);
      formatValue(r.value);
    }
  }
});

test('formatNumber', () => {
  assert.equal(formatNumber(2), '2');
  assert.equal(formatNumber(2.5), '2.5');
  assert.equal(formatNumber(1 / 3), '0.3333');
  assert.equal(formatNumber(2 / 3), '0.6667');
  assert.equal(formatNumber(-2.5), '-2.5');
  assert.equal(formatNumber(100), '100');
  assert.equal(formatNumber(0), '0');
  assert.equal(formatNumber(-0), '0');
  assert.equal(formatNumber(-0.00001), '0');
  assert.equal(formatNumber(1.23456), '1.2346');
  assert.equal(formatNumber(0.1 + 0.2), '0.3');
  assert.equal(formatNumber(1e20), '1e+20');
});

test('formatValue', () => {
  assert.equal(formatValue(3.14159), '3.1416');
  assert.equal(formatValue(V(1, 2.5, 0)), '(1, 2.5, 0)');
  assert.equal(formatValue(P(-0, 1 / 3, 2)), '(0, 0.3333, 2)');
  assert.equal(formatValue({ type: 'mat', m: [[1, 0], [0, 1]] }), '[[1, 0], [0, 1]]');
  assert.equal(formatValue(val('span(i, j)')), 'plane through origin');
  assert.equal(formatValue(val('span(i, 2i)')), 'line through origin');
  assert.equal(formatValue(val('span(i, j, k)')), 'all of R^3');
  assert.equal(formatValue(val('span((0, 0, 0))')), 'just the origin');
  assert.equal(formatValue(val('plane(k)')), 'plane with normal (0, 0, 1)');
  assert.equal(formatValue(val('parallelogram((2, 0, 0), (1, 1.6, 0))')), 'area 3.2');
  assert.equal(formatValue(val('parallelepiped((1, 0, 0), (0, 2, 0), (0, 0, 3))')), 'volume 6');
  assert.equal(formatValue(null), '');
});

test('row matrices, MATLAB-style brackets and transposes', () => {
  assert.deepEqual(val('[1 2 3]').m, [[1, 2, 3]]);
  assert.deepEqual(val('[[1 2 3]]').m, [[1, 2, 3]]);
  assert.deepEqual(val('[1 2; 3 4]').m, [[1, 2], [3, 4]]);
  assert.deepEqual(val('[1 -2 3; 4 5 -6]').m, [[1, -2, 3], [4, 5, -6]]);
  assert.deepEqual(val('[1 - 2 3]').m, [[-1, 3]]); // spaced on both sides: subtraction
  assert.deepEqual(val('[2(1+1) 3]').m, [[4, 3]]); // unspaced juxtaposition still multiplies
  assert.deepEqual(val('[1, 2, 3]'), V(1, 2, 3)); // commas: still a vector
  assert.deepEqual(val('[1; 2; 3]'), V(1, 2, 3)); // one column: a vector
  assert.deepEqual(val('(1,2,3)^T').m, [[1, 2, 3]]);
  assert.deepEqual(val('[1 2; 3 4]^T').m, [[1, 3], [2, 4]]);
  assert.equal(val('[1 2 3] (1, 1, 1)'), 6); // row times vector: a number
  assert.equal(val('(1,2,3)^T (1,2,3)'), 14);
  assert.equal(val('(1,0,1)^T [2 0 0; 0 1 0; 0 0 3] (1,0,1)'), 5); // quadratic form
  assert.deepEqual(val('(1,2,0) (1,1,1)^T').m, [[1, 1, 1], [2, 2, 2], [0, 0, 0]]); // outer product
  assert.equal(err('[1 2; 3]'), 'matrix rows must all be the same length');
  assert.deepEqual(val('[1 2 3]^T'), V(1, 2, 3)); // a column is a vector
  assert.deepEqual(val('[[1],[2],[3]]'), V(1, 2, 3));
  assert.deepEqual(val('[1 0 0; 0 2 0; 0 0 3] [[1],[1],[1]]'), V(1, 2, 3));
});
