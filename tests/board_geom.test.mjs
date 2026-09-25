import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../static/board/geom.js';

const G = globalThis.mathboardBoard.geom;
const S = 75; // the user's mouse handwriting: symbols about 75 px tall

// A glyph box of height h at (x, y) (top-left), width w.
const box = (x, y, w, h = S) => [x, y, x + w, y + h];
const group = (id, b, bars = []) => ({ id, box: b, bars });

test('writingSize is the median glyph height, skipping dots and bars', () => {
  const boxes = [box(0, 0, 50, 70), box(80, 0, 50, 80), box(160, 0, 50, 76), [240, 30, 244, 34], [300, 40, 460, 48]];
  assert.equal(G.writingSize(boxes, 48), 76);
  assert.equal(G.writingSize(boxes.slice(0, 2), 48), 48, 'fewer than 3 glyphs: the fallback');
  // the two crossing strokes of an x are one glyph
  assert.equal(G.glyphBoxes([box(0, 0, 60, 60), box(2, 1, 58, 60)]).length, 1);
  // an exponent that only touches its base's corner stays its own glyph
  assert.equal(G.glyphBoxes([box(0, 20, 50, 60), box(48, 0, 25, 30)]).length, 2);
  assert.equal(G.writingSize([box(0, 0, 50, 400), box(100, 0, 50, 400), box(200, 0, 50, 400)], 48), 180, 'clamped');
});

test('reach scales with the writing size', () => {
  const g = [group('a', box(0, 0, 200))];
  // a symbol 0.8 S to the right joins; 1.2 S does not
  assert.deepEqual(G.joinTargets(g, box(200 + 0.8 * S, 0, 50), S), ['a']);
  assert.deepEqual(G.joinTargets(g, box(200 + 1.2 * S, 0, 50), S), []);
  // at the default size of a pen writer the same gap is too far
  assert.deepEqual(G.joinTargets(g, box(200 + 0.8 * S, 0, 50), 44), []);
  // the Grouping reach slider still multiplies it
  assert.deepEqual(G.joinTargets(g, box(200 + 1.2 * S, 0, 50), S, 1.5), ['a']);
});

test('an exponent joins its base', () => {
  const g = [group('x', box(0, 40, 55))];
  assert.deepEqual(G.joinTargets(g, box(62, 0, 35, 40), S), ['x']);
});

test('fraction: numerator, bar and denominator end up together', () => {
  const num = group('num', box(40, 0, 60));           // d
  const bar = [10, S + 14, 170, S + 24];              // 160 px bar, 10 px tall
  assert.ok(G.isBar(bar, S));
  // the bar reaches up to the numerator, which is out of the plain vertical reach
  const far = [10, S + 40, 170, S + 50];
  assert.ok(far[1] - num.box[3] > G.REACH_Y * S);
  assert.deepEqual(G.joinTargets([num], far, S), ['num']);
  assert.deepEqual(G.joinTargets([num], bar, S), ['num']);
  // a denominator written later lands in the bar's zone
  const withBar = group('num', G.unionBox([num.box, bar]), [bar]);
  const den = box(30, bar[3] + 30, 110);
  assert.deepEqual(G.joinTargets([withBar], den, S), ['num']);
  // a denominator written before the bar is gathered too, and both merge
  const lone = group('den', den);
  assert.deepEqual(G.joinTargets([num, lone], bar, S), ['num', 'den']);
});

test('an = sign is not a fraction bar, and a neighbour on the bar line is not pulled in', () => {
  const eqStroke = [0, 0, 70, 6];
  assert.ok(!G.isBar(eqStroke, S));
  const bar = [0, 100, 160, 108];
  // a group to the right of the bar, on the bar's line, well past its 15% slack
  const right = group('r', box(260, 70, 60));
  assert.deepEqual(G.joinTargets([right], bar, S), []);
});

test('a line far above a fraction bar is not gathered', () => {
  const lineAbove = group('above', box(0, 0, 300));
  const bar = [20, S + 1.4 * S + 10, 200, S + 1.4 * S + 18];
  assert.deepEqual(G.joinTargets([lineAbove], bar, S), []);
});

test('new-line guard: starting left of a group and below it starts a new group', () => {
  const line1 = group('l1', box(100, 0, 300));
  // 20 px below line 1, within the vertical reach, but starting left of it: a new line
  assert.deepEqual(G.joinTargets([line1], box(80, S + 20, 50), S), []);
  // the same gap, but starting inside the line's span: still joins (a subscript, a second matrix row)
  assert.deepEqual(G.joinTargets([line1], box(140, S + 20, 50), S), ['l1']);
});

test('new-line guard for the whole line: a later symbol that reaches both lines stays on the new one', () => {
  // line 2: A = [...] from x = 200; line 3 starts at x = 150 with b, then = and 1 reach up to line 2
  const line2 = group('l2', [200, 305, 640, 410]);
  const line3 = group('l3', [159, 430, 262, 505]); // b =
  assert.deepEqual(G.joinTargets([line2, line3], box(290, 430, 25), S), ['l3']);
  // lines that share a left margin count as new lines too
  assert.deepEqual(G.joinTargets([group('l1', box(200, 0, 300))], box(205, S + 15, 45), S), []);
});

test('a small exponent floating up-right of its base joins it', () => {
  const x = group('x', [704, 196, 742, 246]), s = 65;
  const exp = [753, 128, 780, 176];      // 20 px above the x, out of the vertical reach
  assert.ok(x.box[1] - exp[3] > G.REACH_Y * s * 0.8);
  assert.deepEqual(G.joinTargets([x], exp, s * 0.8), [], 'a bigger stroke: plain reach only');
  assert.deepEqual(G.joinTargets([x], exp, s), ['x']);
  // a full-size symbol up there is a new expression, not an exponent
  assert.deepEqual(G.joinTargets([x], [753, 40, 800, 150], s), []);
});

test('a matrix written with wide gaps stays one group', () => {
  // [ 1   2   3 ] with 60 px gaps between 50 px wide symbols
  let groups = [group('m', box(0, 0, 25, 100))];
  let u = groups[0].box;
  for (const x of [85, 195, 305]) {
    const s = box(x, 12, 50);
    assert.deepEqual(G.joinTargets(groups, s, S), ['m'], `symbol at ${x}`);
    u = G.unionBox([u, s]);
    groups = [group('m', u)];
  }
});

test('burst: the group you just wrote in reaches twice as far sideways', () => {
  const g = [group('a', box(0, 0, 200))];
  const s = box(200 + 1.5 * S, 0, 50);
  assert.deepEqual(G.joinTargets(g, s, S), []);
  assert.deepEqual(G.joinTargets(g, s, S, 1, { burst: 'a' }), ['a']);
});

test('nearZones cover the reach and the bar zones', () => {
  const g = group('a', box(0, 0, 200), [[0, 90, 200, 98]]);
  assert.ok(G.isNear(g, 250, 30, S));
  assert.ok(!G.isNear(g, 400, 30, S));
  assert.ok(G.isNear(g, 100, 98 + S, S), 'under the bar: the denominator area');
  assert.ok(G.isNear(g, 330, 30, S, 1, true), 'burst widens it');
});

test('profiles switch timing and thresholds by pointer type', () => {
  assert.equal(G.profile('mouse').size, 72);
  assert.ok(G.profile('pen').idle < G.profile('mouse').idle);
  assert.equal(G.profile('touch').ceiling, 0, 'touch has no hover');
  assert.equal(G.profile('unknown'), G.profile('mouse'));
});

// ------------------------------------------------------------ scratch-out
const zigzag = (x, y, w, h, n) => Array.from({ length: n * 2 + 1 }, (_, i) => [x + (i % 2 ? w : 0), y + (i * h) / (n * 2)]);

test('looksLikeScratch: a zig-zag yes, letters no', () => {
  assert.ok(G.looksLikeScratch(zigzag(0, 0, 100, 30, 3), 3));
  assert.ok(!G.looksLikeScratch(zigzag(0, 0, 100, 30, 1), 3), 'a z is two reversals');
  // an m: up and down four times while moving right
  const m = [[0, 60], [0, 0], [15, 0], [20, 60], [25, 0], [45, 0], [50, 60], [55, 0], [70, 0], [75, 60]];
  assert.ok(!G.looksLikeScratch(m, 3));
  assert.ok(!G.looksLikeScratch([[0, 0], [5, 5], [3, 4]], 3), 'too short');
  // mouse jitter inside a straight line is not a reversal
  const line = Array.from({ length: 40 }, (_, i) => [i * 5, (i % 2) * 2]);
  assert.ok(!G.looksLikeScratch(line, 3));
});

test('crossings counts where two polylines cross', () => {
  const vertical = [[50, -10], [50, 40]];
  assert.equal(G.crossings(zigzag(0, 0, 100, 30, 3), vertical), 6);
  assert.equal(G.crossings([[0, 0], [10, 0]], [[20, -5], [20, 5]]), 0);
});

// ------------------------------------------------------------ lasso
test('lasso hit tests', () => {
  const poly = [[0, 0], [100, 0], [100, 100], [0, 100]];
  assert.ok(G.pointInPoly(50, 50, poly));
  assert.ok(!G.pointInPoly(150, 50, poly));
  assert.equal(G.fracInside([[10, 10], [20, 20], [150, 20], [160, 20]], poly), 0.5);
  assert.equal(G.rectInside([10, 10, 90, 90], poly), 1);
  assert.equal(G.rectInside([50, 0, 150, 100], poly), 0.4);
  assert.equal(G.coverage([0, 0, 10, 10], [5, 0, 20, 10]), 0.5);
});
