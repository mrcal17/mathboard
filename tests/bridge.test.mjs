import { test } from 'node:test';
import assert from 'node:assert/strict';
import { latexToGrapher } from '../static/graph/features/bridge.js';
import { evaluate, parseLine } from '../static/graph/lang.js';

const src = latex => latexToGrapher(latex).src;
const fails = (latex, re) => assert.throws(() => latexToGrapher(latex), err => err.constructor === Error && re.test(err.message));
const value = (...lines) => {
  const r = evaluate(lines).at(-1);
  assert.equal(r.error, null, `unexpected error: ${r.error}`);
  return r.value;
};
const V = (x, y, z) => ({ type: 'vec', v: [x, y, z] });

test('column, row and bracketed vectors become vector literals', () => {
  assert.equal(src(String.raw`\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}`), '(1, 2, 3)');
  assert.equal(src(String.raw`\begin{pmatrix}1\\-2\\3\end{pmatrix}`), '(1, -2, 3)');
  assert.equal(src(String.raw`\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}`), '[[1, 2, 3]]', 'a row stays a row');
  assert.equal(src(String.raw`\begin{matrix} 1 \\ 2 \end{matrix}`), '(1, 2)');
  assert.equal(src(String.raw`\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}`), '(1, 2, 3)', 'trailing \\\\ ignored');
  assert.equal(src(String.raw`\left[\begin{array}{c} 4 \\ 5 \\ 6 \end{array}\right]`), '(4, 5, 6)');
  assert.equal(src(String.raw`\langle 1, 2, 3 \rangle`), '(1, 2, 3)');
  assert.equal(src(String.raw`\left[ 1, 2, 3 \right]`), '(1, 2, 3)');
  assert.equal(src(String.raw`\begin{bmatrix} \frac{1}{2} \\ -1 \\ \sqrt{2} \end{bmatrix}`), '(1/2, -1, sqrt(2))');
  assert.equal(src(String.raw`\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}^T`), 'transpose([[1, 2, 3]])', 'a transposed row (evaluates to a vector)');
});

test('other shapes become matrix literals; a row of vectors becomes matrix(columns)', () => {
  assert.equal(src(String.raw`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`), '[[1, 2], [3, 4]]');
  assert.equal(src(String.raw`A=\left[\begin{array}{ccc}1&0&0\\0&1&0\\0&0&1\end{array}\right]`), 'A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]');
  assert.equal(src(String.raw`\left( \begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix} \right)`), '[[1, 2], [3, 4]]');
  assert.equal(src(String.raw`\begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \end{bmatrix}`), '[[1], [2], [3], [4]]');
  assert.equal(src(String.raw`\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}`), '[[1, 0], [0, 1], [1, 1]]');
  assert.equal(src(String.raw`\begin{bmatrix} \vec{u} & \vec{v} & \vec{w} \end{bmatrix}`), 'matrix(u, v, w)');
  assert.equal(src(String.raw`\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}`), '[[1, 2], [3, 4]] (x, y)');
});

test('vmatrix and |matrix| are determinants; Vmatrix is a norm', () => {
  assert.equal(src(String.raw`\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix}`), 'det([[1, 2], [3, 4]])');
  assert.equal(src(String.raw`\left| \begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix} \right|`), 'det([[1, 2], [3, 4]])');
  assert.equal(src(String.raw`|\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}|`), 'det([[1, 2], [3, 4]])');
  assert.equal(src(String.raw`\det\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}`), 'det([[2, 0], [0, 3]])');
  assert.equal(src(String.raw`\begin{Vmatrix} 3 \\ 4 \end{Vmatrix}`), '|(3, 4)|');
  assert.equal(src(String.raw`\begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ 1 & 2 & 3 \\ 4 & 5 & 6 \end{vmatrix}`),
    'cross((1, 2, 3), (4, 5, 6))', 'the i j k determinant is a cross product');
  fails(String.raw`\begin{vmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{vmatrix}`, /square matrix/);
});

test('names on the left are kept; vector decorations are dropped', () => {
  assert.deepEqual(latexToGrapher(String.raw`\vec{u} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}`), { src: 'u = (1, 2, 3)', name: 'u', note: '' });
  assert.equal(latexToGrapher('u = (1, 2, 3)').name, 'u');
  assert.equal(src(String.raw`\mathbf{w} = (0, 1, 0)`), 'w = (0, 1, 0)');
  assert.equal(src(String.raw`\vec{v}_1 = (1, 0, 0)`), 'v1 = (1, 0, 0)');
  assert.equal(src(String.raw`\overrightarrow{AB} = B - A`), 'AB = B - A');
  assert.equal(src(String.raw`\vec{u}+\boldsymbol{v}`), 'u + v');
  assert.equal(src(String.raw`\hat{i} + \hat{j} + \hat\imath`), 'i + j + i', 'hatted i j k are the basis vectors');
  assert.equal(src(String.raw`\hat{u} = \frac{\vec{u}}{\|\vec{u}\|}`), 'u_hat = u/|u|', 'a hat makes its own name');
  assert.equal(src(String.raw`\vec{0}`), '(0, 0, 0)');
  assert.equal(src(String.raw`\theta = 1.5`), 'θ = 1.5');
});

test('dot, cross and the other operators', () => {
  assert.equal(src(String.raw`\vec{u}\cdot\vec{v}`), 'u · v');
  assert.equal(src(String.raw`\vec{u} \times \vec{v}`), 'u × v');
  assert.equal(src('u − v'), 'u - v');
  assert.equal(src(String.raw`a \ast b \div c`), 'a*b/c');
  assert.equal(src(String.raw`-\vec{u}`), '-u');
  assert.equal(src('(-1, 2, -3)'), '(-1, 2, -3)');
});

test('fractions and roots', () => {
  assert.equal(src(String.raw`\frac{1}{2}`), '1/2');
  assert.equal(src(String.raw`x = \frac{1}{2}`), 'x = 1/2');
  assert.equal(src(String.raw`\frac12`), '1/2');
  assert.equal(src(String.raw`\frac{a+b}{2}`), '(a + b)/2');
  assert.equal(src(String.raw`2\vec{u} - \frac{1}{2}\vec{v}`), '2 u - (1/2) v');
  assert.equal(src(String.raw`\dfrac{1}{\sqrt{2}}`), '1/sqrt(2)');
  assert.equal(src(String.raw`\sqrt[3]{8}`), '8^(1/3)');
  assert.equal(src(String.raw`x / \frac{1}{2}`), 'x/(1/2)', 'a fraction stays one operand');
  assert.equal(src(String.raw`\frac{\vec{u}\cdot\vec{v}}{\|\vec{v}\|^2}\vec{v}`), '((u · v)/(|v|^2)) v');
});

test('implicit products and subscripts', () => {
  assert.equal(src(String.raw`2\vec{u}+3\vec{v}`), '2 u + 3 v');
  assert.equal(src('ab'), 'a b', 'adjacent letters are separate variables');
  assert.equal(src('3e_1 + 2e_2'), '3 e1 + 2 e2', 'no 3e1 (= 30)');
  assert.equal(src(String.raw`c_1\vec{v}_1 + c_2\vec{v}_2`), 'c1 v1 + c2 v2');
  assert.equal(src(String.raw`\lambda_1 \vec{v}_1`), 'λ1 v1');
  assert.equal(src('v_{12}'), 'v12');
  assert.equal(src('x_{a}'), 'x_a');
  assert.equal(src('a_{1,2}'), 'a12');
  assert.equal(src(String.raw`A\vec{x}`), 'A x');
  assert.equal(src('2(a+b)'), '2(a + b)');
  assert.equal(src(String.raw`\pi r`), 'pi r');
  fails('(a+b)_1', /subscript/);
});

test('powers, inverses and transposes', () => {
  assert.equal(src('x^{2} + y^2'), 'x^2 + y^2');
  assert.equal(src('2^{10}'), '2^10');
  assert.equal(src('x^23'), 'x^2 3', 'TeX takes one digit');
  assert.equal(src('x^{n+1}'), 'x^(n + 1)');
  assert.equal(src(String.raw`A^{-1}\vec{b}`), 'A^-1 b');
  assert.equal(src('A^T'), 'transpose(A)');
  assert.equal(src(String.raw`A^{\top}\vec{x}`), 'transpose(A) x');
  assert.equal(src(String.raw`(AB)^\intercal`), 'transpose(A B)');
  assert.equal(src(String.raw`\theta = 30^\circ`), 'θ = rad(30)');
});

test('functions, norms and projections', () => {
  assert.equal(src(String.raw`\det(A)`), 'det(A)');
  assert.equal(src(String.raw`\det A`), 'det(A)');
  assert.equal(src(String.raw`\sin 2x + 1`), 'sin(2 x) + 1');
  assert.equal(src(String.raw`\sin^2 x + \cos^2 x`), 'sin(x)^2 + cos(x)^2');
  assert.equal(src(String.raw`\sin^{-1}(0.5)`), 'asin(0.5)');
  assert.equal(src(String.raw`\cos\left(\frac{\pi}{3}\right)`), 'cos(pi/3)');
  assert.equal(src(String.raw`\sqrt{2}`), 'sqrt(2)');
  assert.equal(src(String.raw`\|\vec{u}\|`), '|u|');
  assert.equal(src(String.raw`\left\|\vec{u}\right\|^2`), '|u|^2');
  assert.equal(src(String.raw`\lVert \vec u \rVert + |\vec v|`), '|u| + |v|');
  assert.equal(src(String.raw`\text{proj}_{\vec{v}}\vec{u}`), 'proj(u, v)');
  assert.equal(src(String.raw`\operatorname{proj}_{\vec v}(\vec u)`), 'proj(u, v)');
  assert.equal(src(String.raw`\operatorname{span}\{\vec{u}, \vec{v}\}`), 'span(u, v)');
  assert.equal(src(String.raw`\text{span}(\vec{u},\vec{v})`), 'span(u, v)');
  assert.equal(src('span(u, v)'), 'span(u, v)', 'a plain word that is a grapher function');
  assert.equal(src(String.raw`\mathrm{rank}(A)`), 'rank(A)', 'unknown words stay names');
  assert.equal(latexToGrapher(String.raw`\text{rref}(A)`, { functions: ['rref'] }).src, 'rref(A)');
});

test('equations: name = expression, otherwise only the left side (with a note)', () => {
  const eq = latexToGrapher(String.raw`A\vec{x} = \vec{b}`);
  assert.equal(eq.src, 'A x');
  assert.equal(eq.name, null);
  assert.match(eq.note, /left side/);
  const chain = latexToGrapher(String.raw`\vec{w} = 2\vec{u} + \vec{v} = (3, 5, 2)`);
  assert.equal(chain.src, 'w = 2 u + v');
  assert.match(chain.note, /first/);
});

test('delimiters, spacing commands and trailing punctuation are tolerated', () => {
  assert.equal(src(String.raw`$\vec{u} = (1, 2)$`), 'u = (1, 2)');
  assert.equal(src(String.raw`\[ \vec{u} = (1, 2) \]`), 'u = (1, 2)');
  assert.equal(src(String.raw`\vec{u} = (1,\,2,\;3).`), 'u = (1, 2, 3)');
  assert.equal(src(String.raw`\displaystyle \left( 1 + 2 \right) \quad`), '1 + 2');
  assert.equal(src(String.raw`\bigl( a + b \bigr)`), 'a + b');
});

test('unconvertible input throws an Error with a short message', () => {
  fails('', /no LaTeX/);
  fails('   ', /no LaTeX/);
  fails(String.raw`\int_0^1 x\,dx`, /\\int/);
  fails(String.raw`\sum_{i=1}^n x_i`, /\\sum/);
  fails(String.raw`x \le 3`, /relations/);
  fails('x < 3', /relations/);
  fails(String.raw`f \to g`, /arrows/);
  fails("u'", /primes/);
  fails(String.raw`\vec{u}^\prime`, /primes/);
  fails(String.raw`\begin{aligned} a &= 1 \\ b &= 2 \end{aligned}`, /several lines/);
  fails(String.raw`a \\ b`, /several lines/);
  fails(String.raw`\frac{1}{`, /unbalanced/);
  fails('}{', /unbalanced/);
  fails('(1, 2', /missing/);
  fails(String.raw`\begin{bmatrix} 1 & 2 \\ 3 \end{bmatrix}`, /different lengths/);
  fails(String.raw`\begin{bmatrix} 1 & \\ 3 & 4 \end{bmatrix}`, /empty/);
  fails(String.raw`\begin{bmatrix} 1 & 2`, /missing/);
  fails(String.raw`\vec{u} =`, /nothing after u =/);
  fails(String.raw`\vec{u} \in \mathbb{R}^3`, /\\in/);
  fails('x!', /'!'/);
  fails(String.raw`\det = 3`, /det needs an argument/);
  fails(String.raw`\vec{1 + 2}`, /only a name/);
  fails(String.raw`\text{for all } x`, /text/);
  fails('u = (1, 2, 3, 4)', /isn't valid in the 3D tab/);
  fails('dot = 2', /dot needs an argument/);
  fails(String.raw`\toString`, /\\toString/);
  fails(String.raw`\constructor{u}`, /\\constructor/);
});

test('converted rows evaluate to the intended values', () => {
  assert.deepEqual(value(src(String.raw`\vec{u} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}`)), V(1, 2, 3));
  assert.deepEqual(value('u = (1, 2, 3)', 'v = (0, 1, 0)', src(String.raw`\vec{u} \times \vec{v}`)), V(-3, 0, 1));
  assert.equal(value('u = (1, 2, 3)', 'v = (0, 1, 0)', src(String.raw`\vec{u}\cdot\vec{v}`)), 2);
  assert.equal(value(src(String.raw`\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix}`)), -2);
  assert.deepEqual(value(src(String.raw`\begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{vmatrix}`)), V(0, 0, 1));
  assert.deepEqual(value('u = (2, 1, 0)', 'v = (1, 0, 0)', src(String.raw`\frac{\vec{u}\cdot\vec{v}}{\|\vec{v}\|^2}\vec{v}`)), V(2, 0, 0));
  assert.deepEqual(value('A = [[2, 0], [0, 4]]', 'b = (2, 4, 0)', src(String.raw`A^{-1}\vec{b}`)), V(1, 1, 0));
  assert.deepEqual(value(src(String.raw`\vec{u} = 1\hat{i} + 2\hat{j} + 3\hat{k}`)), V(1, 2, 3));
  const x = value(src(String.raw`x = \frac{1}{2} + \frac{\sqrt{9}}{2}`));
  assert.equal(x, 2);
  assert.deepEqual(value('u = (3, 4, 0)', src(String.raw`\hat{u} = \frac{\vec{u}}{\|\vec{u}\|}`)), V(0.6, 0.8, 0));
});

test('never crashes: random LaTeX-ish input either converts to a parsable row or throws an Error', () => {
  const bits = [String.raw`\vec{u}`, 'v', '2', '1.5', '+', '-', '=', ',', '(', ')', '[', ']', '{', '}', '|', '^', '_',
    String.raw`\frac`, String.raw`\sqrt`, String.raw`\cdot`, String.raw`\times`, String.raw`\left(`, String.raw`\right)`,
    String.raw`\begin{bmatrix}`, String.raw`\end{bmatrix}`, '&', String.raw`\\`, String.raw`\det`, String.raw`\sin`,
    String.raw`\|`, 'T', String.raw`\alpha`, String.raw`\text{proj}`, String.raw`\hat`, "'", String.raw`\,`, ' ', 'x', 'e'];
  let seed = 12345;
  const rand = n => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
  let converted = 0;
  for (let i = 0; i < 20000; i++) {
    const s = Array.from({ length: 1 + rand(12) }, () => bits[rand(bits.length)]).join('');
    let res;
    try { res = latexToGrapher(s); }
    catch (err) {
      assert.ok(err.constructor === Error && typeof err.message === 'string' && err.message.length, `bad throw for ${s}`);
      continue;
    }
    converted++;
    assert.equal(typeof res.src, 'string');
    assert.equal(parseLine(res.src)?.error ?? null, null, `${s} -> ${res.src} does not parse`);
  }
  assert.ok(converted > 100, `only ${converted} random inputs converted`);
});
