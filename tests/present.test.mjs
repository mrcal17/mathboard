import { test } from 'node:test';
import assert from 'node:assert/strict';
import katex from '../static/vendor/katex/katex.mjs';
import { rowLatex, nameLatex, exprLatex } from '../static/graph/features/present.js';
import { parseLine } from '../static/graph/lang.js';

const KINDS = { u: 'vec', v: 'vec', w: 'vec', v1: 'vec', v_2: 'vec', a: 'num', b: 'num', t: 'num', A: 'mat', P: 'point' };
const tex = (src, kinds = KINDS) => rowLatex(src, { kinds })?.tex;
const valid = s => katex.renderToString(s, { throwOnError: true, strict: 'ignore' });

test('names: subscripts, Greek, upright words, vector arrows', () => {
  assert.equal(nameLatex('u'), 'u');
  assert.equal(nameLatex('u', true), '\\vec{u}');
  assert.equal(nameLatex('v1', true), '\\vec{v}_{1}');
  assert.equal(nameLatex('v_1', true), '\\vec{v}_{1}');
  assert.equal(nameLatex('x12'), 'x_{12}');
  assert.equal(nameLatex('v_old'), 'v_{\\mathrm{old}}');
  assert.equal(nameLatex('theta'), '\\theta');
  assert.equal(nameLatex('lambda2'), '\\lambda_{2}');
  assert.equal(nameLatex('α'), '\\alpha');
  assert.equal(nameLatex('vel'), '\\mathrm{vel}');
  assert.equal(nameLatex('vel', true), '\\overrightarrow{\\mathrm{vel}}');
});

test('vector and matrix literals', () => {
  assert.equal(tex('u = (1, 2, 0)'), '\\vec{u} = \\begin{pmatrix}1 \\\\ 2 \\\\ 0\\end{pmatrix}');
  assert.equal(tex('[4, 5]', {}), '\\begin{pmatrix}4 \\\\ 5\\end{pmatrix}');
  assert.equal(tex('(a, 2b, -1)'), '\\begin{pmatrix}a \\\\ 2b \\\\ -1\\end{pmatrix}');
  assert.equal(tex('A = [[1, 2], [3, 4]]'), 'A = \\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}');
  assert.equal(tex('matrix(u, v, w)'), '\\begin{bmatrix}\\vec{u} & \\vec{v} & \\vec{w}\\end{bmatrix}');
  assert.equal(tex('matrix((1,0), (2,3))'), '\\begin{bmatrix}1 & 2 \\\\ 0 & 3\\end{bmatrix}');
});

test('left-hand side takes the value kind, or the inferred one when unknown', () => {
  assert.equal(tex('w = a u + b v'), '\\vec{w} = a\\vec{u} + b\\vec{v}');
  assert.equal(tex('z = u - v', { u: 'vec', v: 'vec' }), '\\vec{z} = \\vec{u} - \\vec{v}');
  assert.equal(tex('s = dot(u, v)', { u: 'vec', v: 'vec' }), 's = \\vec{u} \\cdot \\vec{v}');
  assert.equal(tex('P = point(1, 2, 3)'), 'P = \\operatorname{point}(1, 2, 3)');
  assert.equal(tex('t = 0.5'), 't = 0.5');
});

test('products: juxtaposition, explicit * as cdot, dot and cross', () => {
  assert.equal(tex('2 * u'), '2 \\cdot \\vec{u}');
  assert.equal(tex('2u'), '2\\vec{u}');
  assert.equal(tex('A u'), 'A\\vec{u}');
  assert.equal(tex('2 3'), '2 \\cdot 3');
  assert.equal(tex('u · v'), '\\vec{u} \\cdot \\vec{v}');
  assert.equal(tex('u × v'), '\\vec{u} \\times \\vec{v}');
  assert.equal(tex('dot(u, v)'), '\\vec{u} \\cdot \\vec{v}');
  assert.equal(tex('cross(u, v)'), '\\vec{u} \\times \\vec{v}');
  assert.equal(tex('u × 2v'), '\\vec{u} \\times 2\\vec{v}');
  assert.equal(tex('u · (v × w)'), '\\vec{u} \\cdot (\\vec{v} \\times \\vec{w})');
  assert.equal(tex('2 cross(u, v)'), '2(\\vec{u} \\times \\vec{v})');
  assert.equal(tex('vel t', { t: 'num' }), '\\mathrm{vel}\\,t');
  assert.equal(tex('a * -b'), 'a \\cdot (-b)');
  assert.equal(tex('-2u'), '-2\\vec{u}');
});

test('sums, negation and parentheses', () => {
  assert.equal(tex('(a + b) u'), '(a + b)\\vec{u}');
  assert.equal(tex('a (u + v)'), 'a(\\vec{u} + \\vec{v})');
  assert.equal(tex('a - (b - t)'), 'a - (b - t)');
  assert.equal(tex('a - b + t'), 'a - b + t');
  assert.equal(tex('u + -v'), '\\vec{u} + (-\\vec{v})');
  assert.equal(tex('-(a + b)'), '-(a + b)');
  assert.equal(tex('--a'), '-(-a)');
});

test('powers and division', () => {
  assert.equal(tex('a^2'), 'a^{2}');
  assert.equal(tex('(-a)^2'), '(-a)^{2}');
  assert.equal(tex('-a^2'), '-a^{2}');
  assert.equal(tex('(a b)^2'), '(ab)^{2}');
  assert.equal(tex('A^-1'), 'A^{-1}');
  assert.equal(tex('a^(1/2)'), 'a^{1 / 2}');
  assert.equal(tex('1/2 u'), '\\frac{1}{2}\\vec{u}');
  assert.equal(tex('u / |u|'), '\\frac{\\vec{u}}{\\left\\lVert \\vec{u}\\right\\rVert}');
  assert.equal(tex('(1/2)^2'), '\\mathopen{}\\left(\\frac{1}{2}\\right)^{2}');
  assert.equal(tex('2 (1/2)'), '2 \\cdot \\frac{1}{2}');
  assert.equal(tex('1e-7'), '1 \\times 10^{-7}');
});

test('norms, abs, sqrt and special functions', () => {
  assert.equal(tex('|u|'), '\\left\\lVert \\vec{u}\\right\\rVert');
  assert.equal(tex('|u - v|'), '\\left\\lVert \\vec{u} - \\vec{v}\\right\\rVert');
  assert.equal(tex('|a|'), '\\left|a\\right|');
  assert.equal(tex('|A u|'), '\\left\\lVert A\\vec{u}\\right\\rVert');
  assert.equal(tex('norm(u)'), '\\left\\lVert \\vec{u}\\right\\rVert');
  assert.equal(tex('abs(a)'), '\\left|a\\right|');
  assert.equal(tex('sqrt(a^2 + b^2)'), '\\sqrt{a^{2} + b^{2}}');
  assert.equal(tex('inv(A) u'), 'A^{-1}\\vec{u}');
  assert.equal(tex('transpose(A)'), 'A^{\\mathsf{T}}');
  assert.equal(tex('inv(2A)'), '(2A)^{-1}');
  assert.equal(tex('unit(u)'), '\\hat{u}');
  assert.equal(tex('proj(u, v)'), '\\operatorname{proj}_{\\vec{v}}\\vec{u}');
  assert.equal(tex('proj(u + w, v)'), '\\operatorname{proj}_{\\vec{v}}(\\vec{u} + \\vec{w})');
  assert.equal(tex('span(u, v)'), '\\operatorname{span}\\{\\vec{u}, \\vec{v}\\}');
  assert.equal(tex('det(A)'), '\\operatorname{det}(A)');
  assert.equal(tex('angle(u, v)'), '\\operatorname{angle}(\\vec{u}, \\vec{v})');
  assert.equal(tex('acos(a)'), '\\operatorname{arccos}(a)');
});

test('constants, components, and user names shadowing constants', () => {
  assert.equal(tex('i + 2j - k'), '\\hat{\\imath} + 2\\hat{\\jmath} - \\hat{k}');
  assert.equal(tex('2 pi'), '2\\pi');
  assert.equal(tex('|i|'), '\\left\\lVert \\hat{\\imath}\\right\\rVert');
  assert.equal(tex('e + i', { e: 'vec', i: 'num' }), '\\vec{e} + i');
  assert.equal(tex('u.x'), 'u_{x}');
  assert.equal(tex('v1.z'), 'v_{1,z}');
  assert.equal(tex('(u + v).y'), '(\\vec{u} + \\vec{v})_{y}');
});

test('@ origin, comments, blanks and parse errors', () => {
  assert.deepEqual(rowLatex('b v @ a u', { kinds: KINDS }), { tex: 'b\\vec{v}', at: 'a\\vec{u}', note: '' });
  assert.deepEqual(rowLatex('v @ (1, 0, 0)', { kinds: KINDS }), { tex: '\\vec{v}', at: '(1, 0, 0)', note: '' });
  assert.deepEqual(rowLatex('u = (1, 2, 3) # velocity', { kinds: KINDS }).note, 'velocity');
  assert.deepEqual(rowLatex('// just a note'), { tex: '', at: '', note: 'just a note' });
  assert.equal(rowLatex(''), null);
  assert.equal(rowLatex('   '), null);
  assert.equal(rowLatex('(1, 2'), null);
  assert.equal(rowLatex('u = '), null);
  assert.equal(rowLatex('u = 2 +'), null);
});

test('unknown functions look like calls; tall arguments get growing brackets', () => {
  assert.equal(tex('f(x)', {}), 'f(x)');
  assert.equal(tex('myFunc(A, t)'), '\\mathrm{myFunc}(A, t)');
  assert.equal(tex('g_2(u, 3)'), 'g_{2}(\\vec{u}, 3)');
  assert.equal(tex('y = sinh(t) + 1', {}), 'y = \\operatorname{sinh}(t) + 1'); // built in since functions were added
  assert.equal(tex('y = myfn(t) + 1', {}), 'y = \\mathrm{myfn}(t) + 1');
  assert.equal(tex('f(x) = x^3 - x', {}), 'f(x) = x^{3} - x');
  assert.equal(tex("sigmoid'(x)", {}), "\\operatorname{sigmoid}'(x)");
  assert.equal(tex('a (u + v)'), 'a(\\vec{u} + \\vec{v})'); // a is a row: still a product
  assert.equal(tex('det([[1, 0], [0, 1]])'), '\\operatorname{det}\\mathopen{}\\left(\\begin{bmatrix}1 & 0 \\\\ 0 & 1\\end{bmatrix}\\right)');
  assert.equal(tex('span(u, (1, 0, 0))'), '\\operatorname{span}\\mathopen{}\\left\\{\\vec{u}, \\begin{pmatrix}1 \\\\ 0 \\\\ 0\\end{pmatrix}\\right\\}');
  assert.equal(tex('foo(A u)'), '\\mathrm{foo}(A\\vec{u})');
});

test('exprLatex works on a bare AST node', () => {
  assert.equal(exprLatex(parseLine('a u + v').body, { kinds: KINDS }), 'a\\vec{u} + \\vec{v}');
  assert.equal(exprLatex(parseLine('(1, 2)').body, { inline: true }), '(1, 2)');
});

test('everything produced is valid KaTeX', () => {
  const lines = [
    'u = (1, 2, 0)', 'A = [[1,0,0],[0,2,0],[0,0,1]]', 'w = a u + b v', 'v1 + v_2', 'b v @ a u', '|u| + |a|',
    'u / |u|', '1/2 u', 'a^(1/2)', 'proj(u + w, v)', 'span(u, v, w)', 'matrix(u, v, w)', 'inv(A) transpose(A)',
    'i + 2j - k', 'theta = 0.5', 'vel t', 'λ_max u', 'x_old.y', 'sqrt(dot(u,u))', 'unit(v_old)', '1.5e20 u',
    'plane(cross(u, v)) @ point(1,1,1)', 'u × (v × w) · u', '(u + v).y', '-(-a)^2', 'null_vec',
    'Ω = 2', 'ϑ ϕ', 'u = (1, 2, 3) # note',
  ];
  for (const line of lines) {
    const r = rowLatex(line, { kinds: KINDS });
    assert.ok(r, `no latex for ${line}`);
    if (r.tex) valid(r.tex);
    if (r.at) valid(r.at);
  }
});
