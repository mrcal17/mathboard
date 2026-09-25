import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import '../static/board/tex.js';

const T = globalThis.mathboardBoard.tex;

// The vendored KaTeX, run in a sandbox (it is a UMD script, not a module).
const sandbox = {};
sandbox.self = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(new URL('../static/vendor/katex/katex.min.js', import.meta.url), 'utf8'), sandbox);
const katex = sandbox.katex;
const render = (src, trust = false) => katex.renderToString(src, {
  displayMode: true, throwOnError: true, strict: 'ignore', output: 'html',
  trust: trust ? c => c.command === '\\htmlData' : false,
});
const renders = src => { try { render(src); return true; } catch { return false; } };

// Reply shapes seen from the model (subscripted words, glued digits, stray \text), plus common lecture notation.
const CORPUS = [
  '2 + 2 = 4', '2 + \\infty', '12 \\times 34', '1234', 'A x = b', 'C_{\\mu n}', 'W_{0}d +', 'Wand +42', '\\alpha',
  '\\begin{bmatrix} 1 & 2 & 3 \\\\ \\frac{1}{2} & 2 & 3 \\end{bmatrix}', '\\begin{bmatrix} 4 & 8 \\end{bmatrix}',
  'A = \\begin{bmatrix} 1 & 0 \\\\ 4 & 9 \\end{bmatrix}', '\\overline{a} w', '\\text{Word}', '\\text{d} \\downarrow \\quad X',
  '\\frac{d}{dx}', '\\frac{\\partial L}{\\partial w}', 'x^2', 'x^{2} + y_{i}', '\\frac12', '\\sqrt2', '\\sqrt[3]{8}',
  '\\hat x + \\vec{v}', 'e^{i\\pi} + 1 = 0', '\\int_0^1 f(x)\\,dx', '\\sum_{i=1}^{n} i^2', '\\lim_{x \\to 0} \\frac{\\sin x}{x}',
  '\\left( x + 1 \\right)^2', '\\left.\\frac{df}{dx}\\right|_{x=0}', "f'(x) = 2x", '10^{-3}', '\\mathrm{d}x', '\\operatorname{tr}(A)',
  '\\begin{aligned} x &= 1 \\\\ y &= 2 \\end{aligned}', '\\begin{array}{c|c} 1 & 2 \\end{array}', '\\not= 3',
  '\\textcolor{red}{x} + 1', '\\binom{n}{k}', '\\frac{a}{b}^2', '|x| \\leq 1', 'a \\cdot b \\neq c', '\\bar{6}',
  '\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1}', 'x_1, x_2, \\ldots, x_n', '\\{ 1, 2 \\}', '\\mathbb{R}^n',
  '\\nabla f = 0', 'p \\propto q', 'y = mx + b', '\\sigma(z) = \\frac{1}{1 + e^{-z}}', '\\\\[4pt] x',
];

test('tokenize splits control words, control symbols and characters', () => {
  assert.deepEqual(T.tokenize('\\alpha x^2 \\\\ \\,').map(t => t.t), ['\\alpha', 'x', '^', '2', '\\\\', '\\,']);
  const [a] = T.tokenize('  \\infty');
  assert.deepEqual([a.i, a.j], [2, 8]);
});

test('scan wraps tappable symbols; every corpus entry still renders, with one data-mbt per tap', () => {
  for (const src of CORPUS) {
    if (!renders(src)) continue;
    const { tex, taps } = T.scan(src);
    assert.ok(tex, src);
    let html;
    assert.doesNotThrow(() => { html = render(tex, true); }, `wrapped: ${src} -> ${tex}`);
    const n = (html.match(/data-mbt="\d+"/g) || []).length;
    assert.equal(n, taps.length, `taps of ${src}`);
  }
});

test('scan leaves names, text, delimiters and environment arguments alone', () => {
  const t = src => T.scan(src).taps.map(x => x.t).join(' ');
  assert.equal(t('\\text{abc} + 1'), '+ 1');
  assert.equal(t('\\begin{bmatrix} 1 & 2 \\end{bmatrix}'), '1 2');
  assert.equal(t('\\left( x \\right)'), 'x');
  assert.equal(t('\\mathrm{d}x'), 'x');
  assert.equal(t('\\frac{\\partial L}{\\partial w}'), '\\partial L \\partial w');
  assert.equal(t('\\sum_{i=1}^{n}'), 'i = 1 n');
  assert.equal(t('\\sqrt[3]{8}'), '8');
  assert.equal(T.scan('\\htmlData{a=1}{x}').tex, null, 'already has \\html: not wrapped');
});

test('scan braces arguments that were written bare', () => {
  assert.equal(T.scan('x^2').tex, 'x^{\\htmlData{mbt=1}{2}}'.replace('x^', '\\htmlData{mbt=0}{x}^'));
  assert.ok(T.scan('\\frac12').tex.startsWith('\\frac{\\htmlData{mbt=0}{1}}{\\htmlData{mbt=1}{2}}'));
});

test('replaceAt keeps control words apart from letters', () => {
  const src = '2x + \\cdot2';
  const taps = T.taps(src);
  assert.equal(T.replaceAt(src, taps[0], '\\infty'), '\\infty x + \\cdot2');
  assert.equal(taps[3].t, '\\cdot');
  assert.equal(T.replaceAt(src, taps[4], 'z'), '2x + \\cdot z');
  assert.equal(T.replaceAt('2 + \\infty', T.taps('2 + \\infty')[2], '2'), '2 + 2');
});

test('look-alikes: the pairs from the debug images, both ways', () => {
  const pairs = [['2', 'z'], ['2', 'x'], ['2', '\\alpha'], ['2', '\\infty'], ['\\infty', '\\alpha'], ['\\infty', '\\propto'],
    ['\\alpha', '\\propto'], ['1', 'l'], ['1', '|'], ['l', '|'], ['0', 'o'], ['0', 'O'], ['o', 'O'], ['d', '\\partial'],
    ['x', '\\times'], ['\\partial', '\\delta'], ['\\sigma', '\\delta']];
  for (const [a, b] of pairs) {
    assert.ok(T.alternatives(a).includes(b), `${a} offers ${b}`);
    assert.ok(T.alternatives(b).includes(a), `${b} offers ${a}`);
  }
});

test('look-alikes are tappable, render, never include the symbol itself, at most 6 plus the case swap', () => {
  for (const [tok, list] of Object.entries(T.LOOKALIKES)) {
    assert.ok(T.isTappable(tok), `key ${tok}`);
    const alts = T.alternatives(tok);
    assert.ok(alts.length <= 6 && !alts.includes(tok), tok);
    for (const a of list) {
      assert.ok(T.isTappable(a), `${tok} -> ${a} is tappable`);
      assert.ok(renders(a), `${a} renders`);
    }
  }
  assert.ok(T.alternatives('q').includes('Q'), 'case swap for letters');
  assert.deepEqual(T.alternatives('\\oint'), []);
});

test('locks survive a re-recognition of the same ink, and of more ink', () => {
  // the model read 2 + 2 as 2 + \infty; the user picked 2
  const before = '2 + \\infty';
  const k = 2;
  const lock = T.makeLock(before, k, '2');
  assert.deepEqual(lock, { from: '\\infty', to: '2', pos: 2, n: 3, l: '+', r: null });
  assert.equal(T.applyLocks('2 + \\infty', [lock]), '2 + 2');
  // more ink on the right: 2 + \infty = 4
  assert.equal(T.applyLocks('2 + \\infty = 4', [lock]), '2 + 2 = 4');
  // more ink on the left: 1 + 2 + \infty
  assert.equal(T.applyLocks('1 + 2 + \\infty', [lock]), '1 + 2 + 2');
  // the model now reads it right: nothing to do
  assert.equal(T.applyLocks('2 + 2 = 4', [lock]), '2 + 2 = 4');
  // a different symbol there now: left alone
  assert.equal(T.applyLocks('2 + y', [lock]), '2 + y');
});

test('a lock does not rewrite a real occurrence of the symbol elsewhere', () => {
  // "2x" was read "xx"; the first x was picked as 2
  const lock = T.makeLock('xx', 0, '2');
  assert.equal(T.applyLocks('xx + 1', [lock]), '2x + 1');
  assert.equal(T.applyLocks('2x + 1', [lock]), '2x + 1');
});

test('isEmptyReading: NONE and empty matrices are empty', () => {
  for (const s of ['', '  ', 'NONE', '\\text{NONE}', '\\begin{bmatrix} \\end{bmatrix}', '\\begin{bmatrix}\\end{bmatrix} =',
    '\\begin{bmatrix} \\text{ } \\\\ \\end{bmatrix}', '\\begin{pmatrix} & \\\\ & \\end{pmatrix}', '\\begin{bmatrix} \\quad \\end{bmatrix}']) {
    assert.ok(T.isEmptyReading(s), JSON.stringify(s));
  }
  for (const s of ['x \\geq \\begin{bmatrix} \\end{bmatrix}', '\\begin{bmatrix} 1 \\end{bmatrix}', '0', 'None of these',
    '\\begin{bmatrix} \\text{a} \\end{bmatrix}', '\\begin{bmatrix} \\quadx \\end{bmatrix}']) {
    assert.ok(!T.isEmptyReading(s), JSON.stringify(s));
  }
});

test('reading() accepts the old reply and the new one', () => {
  const old = T.reading({ latex: 'x^2', raw: 'x^2', ms: 500, model: 'qwen3-vl:8b-instruct' });
  assert.equal(old.latex, 'x^2');
  assert.equal(old.empty, false);
  assert.equal(old.agree, null);
  assert.equal(old.differ, false);
  assert.deepEqual(old.cands, []);
  assert.equal(old.promptTokens, null);

  assert.equal(T.reading({ latex: '', empty: true }).empty, true);
  assert.equal(T.reading({ latex: 'NONE' }).empty, true);
  assert.equal(T.reading({ latex: '\\begin{bmatrix} \\end{bmatrix}' }).empty, true);
  assert.equal(T.reading(null).empty, true);

  // the ensemble disagrees: latex is Uni-MuMER's, candidates come qwen first; the server's pick leads
  const both = T.reading({
    latex: '2 + 2', model: 'uni-mumer-2b', backend: 'ensemble', agree: false, prompt_tokens: 107, empty: false,
    candidates: [{ backend: 'qwen', model: 'qwen3-vl', latex: '2 + \\infty', ms: 400 }, { backend: 'unimumer', model: 'uni-mumer-2b', latex: '2 + 2', ms: 100 }],
  });
  assert.equal(both.differ, true);
  assert.equal(both.sure, false);
  assert.deepEqual(both.cands.map(c => c.backend), ['unimumer', 'qwen']);
  assert.equal(both.by, 'unimumer');
  assert.equal(both.promptTokens, 107);

  // agreeing after normalization: no choice to make
  const same = T.reading({ latex: 'x^{2}', agree: true, candidates: [{ backend: 'qwen', latex: 'x^{2}' }, { backend: 'unimumer', latex: 'x^2' }] });
  assert.equal(same.differ, false);
  assert.equal(same.sure, true);
  // empty is about latex only: the other candidate's text is offered, and not committed on its own
  const half = T.reading({ latex: '', empty: true, agree: false, candidates: [{ backend: 'qwen', latex: '' }, { backend: 'unimumer', latex: '\\uparrow' }] });
  assert.equal(half.latex, '\\uparrow');
  assert.equal(half.empty, false);
  assert.equal(half.differ, false);
  assert.equal(half.sure, false);
  assert.equal(half.by, 'unimumer');
  // one member failed: one candidate, agree false, errors
  const failed = T.reading({ latex: 'w', agree: false, errors: { qwen: 'timeout' }, candidates: [{ backend: 'unimumer', latex: 'w' }] });
  assert.equal(failed.sure, false);
  assert.equal(failed.differ, false);
  assert.deepEqual(failed.errors, { qwen: 'timeout' });
  // a single backend
  assert.equal(T.reading({ latex: 'w', agree: null, backend: 'qwen', candidates: [{ backend: 'qwen', latex: 'w' }] }).by, 'qwen');
});

test('stackLines stacks top-level lines, not matrix rows', () => {
  assert.equal(T.stackLines('x + 1'), null);
  assert.equal(T.stackLines('\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}'), null);
  const two = T.stackLines('A x = b\nA = \\begin{bmatrix} 1 & 0 \\\\ 4 & 9 \\end{bmatrix}');
  assert.ok(two.startsWith('\\begin{gathered}A x = b \\\\ A = \\begin{bmatrix}'), two);
  assert.ok(renders(two));
  assert.equal(T.stackLines('x = 1 \\\\ y = 2'), '\\begin{aligned}x &= 1 \\\\ y &= 2\\end{aligned}');
  assert.ok(renders(T.stackLines('a + b \\\\ c')));
});
