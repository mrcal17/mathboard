// Presets for the 3D tab: ready-made scenes for ML and linear-algebra lectures.
//   The Presets button in the sidebar header opens a menu of them in sections. Its search field
//   filters on the label, section, description and rows (every word must match somewhere); ↑ ↓
//   (or Tab) move, ← → change column while the field is empty, Enter loads, Esc closes.
//   Loading one replaces the rows and sets the camera: 2D or 3D, the axis extent and a framing.
//   The rows and camera it replaced go on an undo stack (kept in localStorage), which the toast's
//   Undo and the menu's "Back to your rows" pop.
//   api.presets = { list, load(key), undo(), canUndo } for scripts and tests.
// PRESETS is plain data, and this module loads no DOM or scene code at the top level:
// tests/presets.test.mjs evaluates every preset's rows in node.

// Row colours are indices into grapher.js PALETTE.
const R = 0, B = 1, G = 2, P = 3, O = 4, T = 5, N = 7;

// rows: [src, colour, {min, max, hidden}] (a row without a colour is a comment).
// camera: {flat: true, extent, frame: [x0, x1, y0, y1]}  2D, that box fitted to the view
//         {extent, dir, target, zoom}                     3D, looking along dir at target; zoom 1
//                                                         frames about +-1.25 extent, as Fit does
export const PRESETS = [
  // ---------------------------------------------------------------- ML: losses
  {
    key: 'nll-prob', group: 'ML: losses', label: 'NLL vs 0/1 error by probability',
    note: 'The loss we train, −ln p, against the 0/1 error it stands in for, where p is the probability the model gives the true class. Drag p.',
    rows: [
      ['# x is p, the probability of the true class'],
      ['y = -ln(x) {0 < x <= 1}  # NLL', R],
      ['y = heaviside(0.5 - x) {0 < x <= 1}  # 0/1 error', B],
      ['p = 0.25  # one prediction', O, { min: 0.1, max: 1 }],
      ['point(p, -ln(p))', R],
      ['point(p, heaviside(0.5 - p))', B],
    ],
    camera: { flat: true, extent: 1.5, frame: [-0.1, 1.1, -0.15, 1.95] },
  },
  {
    key: 'nll-margin', group: 'ML: losses', label: 'NLL vs error by margin',
    note: 'Logistic loss by margin m = y · score. In bits it lies above the 0/1 error, and its gradient σ(m) − 1 never quite reaches 0.',
    rows: [
      ['# x is the margin m = y · score: right when m > 0'],
      ['nll(m) = -ln(sigmoid(m))  # NLL in nats', O, { hidden: true }],
      ['y = nll(x) / ln(2)  # in bits: above the error', R],
      ['y = heaviside(-x)  # 0/1 error', B],
      ["nll'  # its gradient σ(m) − 1", G],
    ],
    camera: { flat: true, extent: 3, frame: [-3.1, 3.1, -1.2, 3.6] },
  },
  {
    key: 'ce-surface', group: 'ML: losses', label: 'Cross-entropy surface',
    note: 'The loss −ln softmax(z)₁ over logits z = (x, y, 0) when class 1 is right: flat where x wins, a ramp where it loses. The point is one prediction.',
    rows: [
      ['# logits (x, y, 0); class 1 is the right one'],
      ['z = -ln(softmax((x, y, 0)).x)  # cross-entropy', T],
      ['a = 0.5  # logit of class 1', O, { min: -3, max: 3 }],
      ['b = 2  # logit of class 2', O, { min: -3, max: 3 }],
      ['p = softmax((a, b, 0))  # the prediction', O, { hidden: true }],
      ['L = -ln(p.x)  # its loss', O],
      ['P = point(a, b, L)  # at the height of its loss', O],
    ],
    camera: { extent: 4, dir: [10, -7, 7], target: [0, 0, 1], zoom: 0.8 },
  },
  {
    key: 'surrogates', group: 'ML: losses', label: 'Surrogate losses by margin',
    note: 'Squared error, hinge and logistic loss next to the 0/1 error they stand in for. Each lies above it and has a gradient to follow.',
    rows: [
      ['# x is the margin m = y · score'],
      ['y = heaviside(-x)  # 0/1 error', B],
      ['y = -ln(sigmoid(x)) / ln(2)  # logistic, in bits', R],
      ['y = max(0, 1 - x)  # hinge (SVM)', G],
      ['y = (1 - x)^2  # squared: punishes m > 1 too', P],
      ['y = exp(-x)  # exponential (boosting)', O, { hidden: true }],
    ],
    camera: { flat: true, extent: 3, frame: [-2.3, 3.1, -0.3, 3.5] },
  },
  {
    key: 'kl', group: 'ML: losses', label: 'Cross-entropy, entropy and KL',
    note: 'A coin with P(heads) = q, modelled as p = x: the cross-entropy is lowest at p = q, where it equals the entropy H(q). The gap above is KL(q‖p).',
    rows: [
      ['q = 0.3  # the true P(heads)', R, { min: 0.02, max: 0.98 }],
      ['H(p) = -(p ln(p) + (1 - p) ln(1 - p))  # entropy', P],
      ['y = -(q ln(x) + (1 - q) ln(1 - x))  # cross-entropy', R],
      ['y = H(q) {0 < x < 1}  # its minimum, H(q)', N],
      ['point(q, H(q))', R],
    ],
    camera: { flat: true, extent: 1.5, frame: [-0.1, 1.1, -0.15, 1.95] },
  },

  // ---------------------------------------------------------------- ML: softmax and attention
  {
    key: 'softmax-temp', group: 'ML: softmax and attention', label: 'Softmax temperature',
    note: 'Softmax squashes the grid of logits onto the probability triangle; the temperature T sets how hard. Play T: as T → 0 everything crowds into the corners.',
    rows: [
      ['T = 1  # temperature: play it', P, { min: 0.1, max: 3 }],
      ['softmax(T)  # the logit grid on the triangle', P],
      ['p = softmax((2, 1, 0.5), T)  # one set of logits', O],
    ],
    camera: { extent: 1.5, dir: [10, 5, 6], target: [0.33, 0.33, 0.33], zoom: 1.5 },
  },
  {
    key: 'attention', group: 'ML: softmax and attention', label: 'Attention weights on the simplex',
    note: 'A query turns past three keys. Softmax of the scores q · kᵢ is a point on the probability triangle that swings toward the best-matching key.',
    rows: [
      ['theta = 0.6  # the query turns: play it', O, { min: 0, max: 6.2832 }],
      ['k1 = (1, 0, 0)  # three keys', R],
      ['k2 = (-0.5, 0.87, 0)', B],
      ['k3 = (-0.5, -0.87, 0)', G],
      ['q = 2 (cos(theta), sin(theta), 0)  # the query', O],
      ['softmax @ (0, 2.2, 0)  # the weights live on this triangle', P],
      ['a = trail(softmax((dot(q, k1), dot(q, k2), dot(q, k3)))) @ (0, 2.2, 0)', O],
    ],
    camera: { extent: 3, dir: [7, 2, 10], target: [0.2, 1.3, 0.2], zoom: 1.4 },
  },
  {
    key: 'smooth-max', group: 'ML: softmax and attention', label: 'Smooth max: logsumexp',
    note: 'logsumexp is a smooth max: it rounds off the creases of max(x, y, 0) and sits at most T ln 3 above it. Lower T to sharpen it, or show the gap row to see where they differ.',
    rows: [
      ['T = 1  # temperature: T → 0 gives max', B, { min: 0.05, max: 2 }],
      ['z = T logsumexp((x, y, 0) / T)  # smooth max', B],
      ['z = max(x, y, 0)  # the max itself', R],
      ['z = T logsumexp((x, y, 0) / T) - max(x, y, 0)  # the gap', P, { hidden: true }],
    ],
    camera: { extent: 3, dir: [9, -7, 5], target: [0, 0, 0.6] },
  },

  // ---------------------------------------------------------------- ML: activations and gradients
  {
    key: 'activations', group: 'ML: activations and gradients', label: 'Activation gallery',
    note: 'Sigmoid, tanh, ReLU, GELU and SiLU, each row showing its formula.',
    rows: [
      ['sigmoid  # squashes to (0, 1)', R],
      ['tanh  # (−1, 1), centred on 0', B],
      ['relu  # no ceiling; flat below 0', G],
      ['gelu  # a smooth ReLU (BERT, GPT)', P],
      ['silu  # x σ(x), also called swish', O],
    ],
    camera: { flat: true, extent: 3, frame: [-3.2, 3.2, -1.3, 3.2] },
  },
  {
    key: 'vanishing', group: 'ML: activations and gradients', label: 'Vanishing gradients',
    note: 'The slopes of sigmoid, tanh and ReLU. Through a chain of n sigmoids the gradient is a product of n factors of at most 1/4, so it shrinks like (1/4)ⁿ. Play n.',
    rows: [
      ["sigmoid'  # at most 1/4", R],
      ["tanh'  # at most 1", B],
      ["relu'  # exactly 1 for x > 0", G],
      ['n = 2  # depth: play it', P, { min: 1, max: 10 }],
      ["y = sigmoid'(x)^n  # through n sigmoids", P],
    ],
    camera: { flat: true, extent: 2, frame: [-2.1, 2.1, -0.2, 1.2] },
  },
  {
    key: 'relu-net', group: 'ML: activations and gradients', label: 'A ReLU network is piecewise linear',
    note: 'Three ReLU units, each a ridge that switches on across a line, summed by the output unit: flat pieces meeting at creases. Show the row of a unit to see its ridge.',
    rows: [
      ['h1(x, y) = relu(x + y - 1)  # three hidden units', R, { hidden: true }],
      ['h2(x, y) = relu(x - 2 y)', B, { hidden: true }],
      ['h3(x, y) = relu(-2 x - y - 3)', G, { hidden: true }],
      ['z = 0.4 h1(x, y) - 0.3 h2(x, y) + 0.4 h3(x, y)  # the output: flat pieces', T],
    ],
    camera: { extent: 3, dir: [9, 5, 7], target: [0, 0, 0], zoom: 0.85 },
  },

  // ---------------------------------------------------------------- ML: optimization and models
  {
    key: 'gd-lr', group: 'ML: optimization and models', label: 'Gradient descent and the learning rate',
    note: 'Gradient descent on the bowl a x² + b y², in closed form. Play k to take steps: past η = 1/(2a) it zigzags across the valley, past 1/a it diverges.',
    rows: [
      ['z = a x^2 + b y^2  # the loss: a bowl', G],
      ['a = 0.5  # curvature the steep way', G, { min: 0.1, max: 1 }],
      ['b = 0.1  # and the shallow way', G, { min: 0.02, max: 1 }],
      ['eta = 0.6  # learning rate', O, { min: 0.05, max: 2.2 }],
      ['k = 0  # steps taken: play it', O, { min: 0, max: 20 }],
      ['X = 2 (1 - 2 eta a)^floor(k)  # each step: x times 1 − 2ηa', O],
      ['Y = 2.5 (1 - 2 eta b)^floor(k)', O],
      ['L = a X^2 + b Y^2  # the loss there', O],
      ['w = trail(point(X, Y, L))  # the weights, on the bowl', O],
    ],
    camera: { extent: 3, dir: [4, 10, 8], target: [0, 0, 0.8], zoom: 0.95 },
  },
  {
    key: 'mse', group: 'ML: optimization and models', label: 'Linear regression: the MSE surface',
    note: 'Mean squared error of the line y = w x + b on three data points, over (w, b): a bowl whose lowest point is the least-squares fit. Move the data with the sliders.',
    rows: [
      ['# data (−1, y1), (0, y2), (1, y3); axes w, b, loss'],
      ['y1 = 0', B, { min: -3, max: 3 }],
      ['y2 = 2', B, { min: -3, max: 3 }],
      ['y3 = 1', B, { min: -3, max: 3 }],
      ['mse(w, b) = ((b - w - y1)^2 + (b - y2)^2 + (b + w - y3)^2) / 3', T],
      ['w_ls = (y3 - y1) / 2  # least squares', O],
      ['b_ls = (y1 + y2 + y3) / 3', O],
      ['best = point(w_ls, b_ls, mse(w_ls, b_ls))  # the lowest point', O],
    ],
    camera: { extent: 3, dir: [7, 5, 9], target: [0.4, 0.8, 0.8], zoom: 1.05 },
  },
  {
    key: 'logreg', group: 'ML: optimization and models', label: 'Logistic regression',
    note: 'P(class 1) = σ(w₁x + w₂y + b) over the plane. The decision boundary, where it crosses 1/2, is drawn on the surface and in the floor. The weights tilt it and sharpen it.',
    rows: [
      ['w1 = 2  # weights', O, { min: -4, max: 4 }],
      ['w2 = 1.5', O, { min: 0.25, max: 4 }],
      ['b = 0.5  # bias', O, { min: -3, max: 3 }],
      ['z = sigmoid(w1 x + w2 y + b)  # P(class 1)', T],
      ['y = -(w1 x + b) / w2 {|y| <= 2}  # decision boundary', O],
      ['y = -(w1 x + b) / w2 {|y| <= 2} @ (0, 0, 0.5)  # at 1/2', O],
      ['z = 0.5 {|x| <= 2, |y| <= 2}  # the level 1/2', N, { hidden: true }],
    ],
    camera: { extent: 2, dir: [7, -9, 6], target: [0, 0, 0.4], zoom: 1.1 },
  },
  {
    key: 'l1-l2', group: 'ML: optimization and models', label: 'L1 vs L2 regularization',
    note: 'The L1 penalty |x| + |y| is a pyramid with its corners on the axes; the L2 penalty x² + y² is a round bowl. Those corners are why L1 sets weights to exactly 0.',
    rows: [
      ['z = |x| + |y|  # L1: corners on the axes', R],
      ['z = x^2 + y^2  # L2: round', B],
    ],
    camera: { extent: 2, dir: [10, 5, 6], target: [0, 0, 0.6] },
  },
  {
    key: 'gaussian', group: 'ML: optimization and models', label: 'Gaussian',
    note: 'The normal density in 2D (the bump) and in 1D (on the back wall), with standard deviation s. Make s smaller: both get narrower and taller, keeping an area of 1.',
    rows: [
      ['s = 0.45  # standard deviation', P, { min: 0.35, max: 1.5 }],
      ['z = exp(-(x^2 + y^2) / (2 s^2)) / (2 pi s^2)  # 2D', P],
      ['z = exp(-y^2 / (2 s^2)) / (s sqrt(2 pi)) @ (-1.5, 0, 0)  # 1D, on the wall', B],
    ],
    camera: { extent: 1.5, dir: [10, 5, 6], target: [-0.3, 0, 0.3], zoom: 1 },
  },

  // ---------------------------------------------------------------- Linear algebra
  {
    key: 'span', group: 'Linear algebra', label: 'Span and linear combinations',
    note: 'a u + b v laid tip to tail for two sliders, inside span(u, v): the plane of every combination. Play a and b.',
    rows: [
      ['u = (1, 2, 0)', R],
      ['v = (-1, 1, 2)', B],
      ['a = 1.5  # sliders: play them', R, { min: -2, max: 2 }],
      ['b = 1', B, { min: -2, max: 2 }],
      ['a u', R],
      ['b v @ a u  # tip to tail', B],
      ['w = a u + b v', G],
      ['span(u, v)  # every combination', P],
    ],
    camera: { extent: 4, dir: [10, 5, 6], target: [0, 1, 1] },
  },
  {
    key: 'transform', group: 'Linear algebra', label: 'Transformation and eigenvectors',
    note: 'transform(A, t) moves space from I to A as t plays from 0 to 1; eigen(A) draws the lines A only stretches. u lies on one and stays there; v turns.',
    rows: [
      ['A = [1.5 0.5 0; 0.5 1.5 0; 0 0 0.75]', N],
      ['t = 1  # play it from 0', O, { min: 0, max: 1 }],
      ['transform(A, t)', O],
      ['eigen(A)  # lines A only stretches', P],
      ['u = (1, 1, 0)  # on an eigenline', R],
      ['v = (1, -0.5, 1)  # off them', B],
    ],
    camera: { extent: 3, dir: [10, 5, 6], target: [0, 0, 0.5] },
  },
  {
    key: 'subspaces', group: 'Linear algebra', label: 'The four fundamental subspaces',
    note: 'A rank-1 2×3 matrix: its row space and null space in ℝ³, its column space and left null space in ℝ², each pair at right angles.',
    rows: [
      ['A = [1 2 3; 2 4 6]  # rank 1: row 2 is twice row 1', N],
      ['subspaces(A)', P],
    ],
    camera: { extent: 6, dir: [10, 5, 6], target: [0, 0, 0] },
  },
  {
    key: 'elimination', group: 'Linear algebra', label: 'Gauss-Jordan elimination',
    note: 'The row picture of a 3×3 system, one elimination step at a time: the planes turn but keep meeting at the solution (2, 3, −1). Play k.',
    rows: [
      ['A = [2 1 -1; -3 -1 2; -2 1 2]', N],
      ['b = (8, -11, -3)', N, { hidden: true }],
      ['k = 0  # step: play it', O],
      ['eliminate(A, b, k)', O],
    ],
    camera: { extent: 5, dir: [6, 10, 5], target: [0.5, 1, -0.5], zoom: 0.8 },
  },
  {
    key: 'svd', group: 'Linear algebra', label: 'SVD: sphere to ellipsoid',
    note: 'A takes the unit sphere to an ellipsoid in three moves: rotate by Vᵀ, stretch by Σ, rotate by U. Play t from 0 to 3.',
    rows: [
      ['A = [2 1 0; 0 1.5 0.5; 0.5 0 1]', N],
      ['t = 0  # 0 to 3: rotate, stretch, rotate', O, { min: 0, max: 3 }],
      ['svdview(A, t)', O],
    ],
    camera: { extent: 3, dir: [10, 5, 6], target: [0, 0, 0], zoom: 1.25 },
  },
  {
    key: 'gram-schmidt', group: 'Linear algebra', label: 'Gram-Schmidt',
    note: 'Three vectors made orthonormal one step at a time: each loses its projections onto the ones before it, then is scaled to length 1. Play k.',
    rows: [
      ['u = (2, 1, 0)', R],
      ['v = (1, 2, 1)', B],
      ['w = (0, 1, 2)', G],
      ['k = 0  # step: play it', O],
      ['gramschmidt(u, v, w, k)', O],
    ],
    camera: { extent: 3, dir: [10, -4, 5], target: [0.3, 0.5, 0.4], zoom: 1.2 },
  },
  {
    key: 'lstsq', group: 'Linear algebra', label: 'Least squares is a projection',
    note: 'Fitting a line through three points is projecting b onto the column space of A: p = A x̂ is the closest reachable point, and the residual meets C(A) at a right angle.',
    rows: [
      ['# fit b ≈ c + d t through (0, 1), (1, 3), (2, 2)'],
      ['A = [1 0; 1 1; 1 2]  # columns: 1 and t', N],
      ['b = (1, 3, 2)', R, { hidden: true }],
      ['lstsq(A, b)', B],
    ],
    camera: { extent: 4, dir: [10, 2, 4], target: [0.5, 1, 1], zoom: 1.1 },
  },
];

// ================================================================ pure helpers (tested)

// A preset's rows in grapher form, {src, color, min, max, hidden}.
export function presetRows(p, palette) {
  return p.rows.map(([src, c, opts = {}]) => ({ src, color: palette[c ?? N], ...opts }));
}

const fold = s => s.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase(); // é -> e
const hays = new WeakMap();
// Does a preset match the search text? Every word must appear in its label, section, note or rows.
export function matches(p, text) {
  const words = fold(text).split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  if (!hays.has(p)) hays.set(p, fold(`${p.label} ${p.key} ${p.group} ${p.note} ${p.rows.map(r => r[0]).join(' ')}`));
  return words.every(w => hays.get(p).includes(w));
}

const FOV = 40; // scene.js
// The 3D pose for a camera spec, for a view of the given aspect (width / height).
export function pose3D(cam, aspect) {
  const d = cam.dir ?? [10, 5, 6], L = Math.hypot(...d), t = cam.target ?? [0, 0, 0];
  const dist = (cam.extent * 1.25) / Math.tan((FOV * Math.PI) / 360) / Math.min(1, aspect || 1) / (cam.zoom ?? 1);
  return { position: t.map((x, i) => x + (d[i] / L) * dist), target: t.slice(), zoom: 1, ortho: false };
}
// The 2D (top, orthographic) pose showing frame [x0, x1, y0, y1], for an orthographic camera whose
// half-height at zoom 1 is halfH.
export function pose2D(cam, aspect, halfH) {
  const [x0, x1, y0, y1] = cam.frame, target = [(x0 + x1) / 2, (y0 + y1) / 2, 0];
  const zoom = Math.min(halfH / ((y1 - y0) / 2), (halfH * aspect) / ((x1 - x0) / 2));
  const dist = (cam.extent * 1.25) / Math.tan((FOV * Math.PI) / 360), n = Math.hypot(1e-3, 1);
  return { position: [target[0], target[1] - (1e-3 / n) * dist, dist / n], target, zoom, ortho: true };
}

// ================================================================ UI
const UNDO_KEY = 'mathboard.presets.undo';
const UNDO_MAX = 12, MS = 700;

const CSS = `
#g-panel header .gp-open { margin-right: var(--sp-4); }
#g-panel header .gp-caret { display: inline-flex; margin: 0 -3px 0 1px; color: var(--text-3); transition: transform var(--dur-2) var(--ease); }
#g-panel header .gp-caret .ui-icon { width: 12px; height: 12px; }
#g-panel header .gp-open[aria-expanded="true"] .gp-caret { transform: rotate(180deg); }
.gp-pop {
  position: fixed; z-index: 20; display: flex; flex-direction: column; overflow: hidden;
  width: min(620px, calc(100vw - 24px)); padding: 0;
  animation: gp-in var(--dur-2) var(--ease-out);
}
.gp-pop[hidden], .gp-pop [hidden], body.clean .gp-pop, body.lec-audience .gp-pop { display: none; }
@keyframes gp-in { from { opacity: 0; transform: translateY(-4px); } }
.gp-top { display: flex; align-items: center; gap: 16px; padding: 10px 12px; border-bottom: 1px solid var(--line-1); }
.gp-field { position: relative; flex: 1; min-width: 0; display: flex; color: var(--text-3); }
.gp-field .ui-icon { position: absolute; left: 10px; top: 8px; pointer-events: none; }
.gp-find { flex: 1; height: 32px; padding-left: 34px; font-size: var(--fs-md); }
.gp-keys { flex: none; display: flex; gap: 12px; font-size: var(--fs-sm); color: var(--text-3); white-space: nowrap; }
.gp-keys kbd + kbd { margin-left: 3px; }
.gp-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 0 8px 8px; }
.gp-cols { columns: 270px; column-gap: 12px; }
.gp-sec { break-inside: avoid; padding-top: 2px; }
.gp-sec h4 { margin: 12px 10px 4px; }
.gp-pop .gp-item { min-height: 30px; padding: 6px 10px; }
.gp-pop .gp-item:hover { background: transparent; }
.gp-pop .gp-item.act { background: var(--raise); }
.gp-none { margin: 16px 10px 8px; color: var(--text-3); }
.gp-foot { display: flex; align-items: flex-start; gap: 12px; padding: 10px 12px 12px 14px; border-top: 1px solid var(--line-1); }
.gp-note { flex: 1; margin: 0; min-height: calc(2 * 1.45em); font-size: 12.5px; line-height: 1.45; color: var(--text-3); }
.gp-note b { margin-right: 4px; font-weight: var(--fw-strong); color: var(--text-1); }
.gp-back { flex: none; margin-top: -3px; }
#toast .gp-undo { margin: -3px -6px -3px 12px; vertical-align: baseline; }
`;

const esc = s => String(s).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);

export function install(api) {
  const header = api.panelEl?.querySelector('header');
  if (!header || api.params.has('audience')) return; // an audience window only mirrors
  api.addStyles(CSS);
  const ic = (name, size) => window.mathboardIcons?.svg(name, { size }) || api.icon?.(name) || '';
  const byKey = new Map(PRESETS.map(p => [p.key, p]));

  // ---------------------------------------------------------------- the undo stack
  let stack = [];
  try { stack = JSON.parse(localStorage.getItem(UNDO_KEY)) || []; } catch { /* nothing to go back to */ }
  stack = Array.isArray(stack) ? stack.filter(s => Array.isArray(s?.rows)) : [];
  const saveStack = () => {
    if (!api.persist) return;
    try { localStorage.setItem(UNDO_KEY, JSON.stringify(stack)); } catch { /* storage full: undo still works until a reload */ }
  };

  // An orthographic zoom only means something with the frustum it was taken with (scene.js sets
  // that from the camera distance when it switches), so keep its height too.
  const sceneState = () => {
    const sc = api.scene;
    return sc ? { pose: sc.getPose(), flat: sc.is2D, spin: sc.autoRotate, orthoH: sc.ortho ? sc.camera.top : null } : {};
  };
  function restore(s) {
    api.setRows(s.rows);
    const sc = api.scene;
    if (!sc) return;
    if (s.flat != null && sc.is2D !== s.flat) sc.set2D(s.flat);
    sc.setAutoRotate(!!s.spin);
    if (s.pose) {
      const pose = { ...s.pose };
      if (pose.ortho) {
        sc.setOrtho(true);
        if (s.orthoH > 0) pose.zoom = (pose.zoom ?? 1) * (sc.camera.top / s.orthoH);
      }
      sc.setPose(pose, MS);
    }
    api.recompute(); // saves the extent with the rows
  }

  function setCamera(cam) {
    const sc = api.scene;
    if (!sc) return;
    const el = api.viewEl, aspect = el.clientWidth / Math.max(1, el.clientHeight);
    sc.setAutoRotate(false);
    if (cam.flat) {
      if (!sc.is2D) sc.set2D(true);
      sc.setExtent(cam.extent);
      sc.setPose(pose2D(cam, aspect, sc.camera.top), MS);
    } else {
      if (sc.is2D) sc.set2D(false);
      sc.setExtent(cam.extent);
      sc.setPose(pose3D(cam, aspect), MS);
    }
  }

  function load(key) {
    const p = byKey.get(key);
    if (!p) return false;
    if (api.view !== 'graph') api.setView('graph');
    stack.push({ label: p.label, rows: api.getState().rows, ...sceneState() });
    if (stack.length > UNDO_MAX) stack.shift();
    saveStack();
    api.setRows(presetRows(p, api.PALETTE));
    setCamera(p.camera);
    api.recompute();
    undoToast(p.label);
    paintBack();
    return true;
  }

  function undo() {
    const s = stack.pop();
    if (!s) return false;
    saveStack();
    restore(s);
    paintBack();
    api.toast(s.label ? `Back to your rows from before ${s.label}` : 'Back to your rows', 1800);
    return true;
  }

  // The shared toast, with an Undo button after the text (api.toast replaces the text, and its
  // timer hides it, so a later toast simply takes over).
  function undoToast(label) {
    api.toast(`Loaded ${label}.`, 8000);
    const t = document.getElementById('toast');
    if (!t) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'gp-undo ui-btn xs soft';
    b.textContent = 'Undo';
    b.title = 'Bring back the rows and camera from before this preset';
    b.onclick = () => { t.hidden = true; undo(); };
    t.appendChild(b);
  }

  // ---------------------------------------------------------------- the button and the menu
  const btn = document.createElement('button');
  btn.className = 'gp-open ui-btn sm';
  btn.title = 'Presets: ready-made scenes for ML and linear-algebra lectures';
  btn.setAttribute('aria-haspopup', 'dialog');
  btn.setAttribute('aria-expanded', 'false');
  btn.innerHTML = `${ic('sparkle')}<span>Presets</span><span class="gp-caret" aria-hidden="true">${ic('chevron-down', 12)}</span>`;
  const add = header.querySelector('#g-add');
  if (add) add.after(btn);
  else header.prepend(btn);

  const groups = new Map();
  for (const p of PRESETS) {
    if (!groups.has(p.group)) groups.set(p.group, []);
    groups.get(p.group).push(p);
  }
  const pop = document.createElement('div');
  pop.className = 'gp-pop ui-menu';
  pop.hidden = true;
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Presets');
  pop.innerHTML = `
    <div class="gp-top">
      <label class="gp-field">${ic('search')}<input class="gp-find ui-field" type="text" spellcheck="false" autocomplete="off"
        placeholder="Search ${PRESETS.length} presets" role="combobox" aria-label="Search the presets" aria-controls="gp-list"
        aria-expanded="true" aria-autocomplete="list"></label>
      <span class="gp-keys"><span><kbd>&uarr;</kbd><kbd>&darr;</kbd><kbd>&larr;</kbd><kbd>&rarr;</kbd> choose</span>
        <span><kbd>Enter</kbd> load</span> <span><kbd>Esc</kbd> close</span></span>
    </div>
    <div class="gp-scroll">
      <div class="gp-cols" id="gp-list" role="listbox" aria-label="Presets">${[...groups].map(([g, list], i) => `
        <section class="gp-sec" role="group" aria-labelledby="gp-h${i}"><h4 class="ui-overline" id="gp-h${i}">${esc(g)}</h4>${list.map(p => `
          <button type="button" class="gp-item ui-menu-item" role="option" aria-selected="false" id="gp-${p.key}" data-key="${p.key}"
            title="${esc(p.note)}">${esc(p.label)}</button>`).join('')}
        </section>`).join('')}
      </div>
      <p class="gp-none" hidden></p>
    </div>
    <div class="gp-foot">
      <p class="gp-note"></p>
      <button type="button" class="gp-back ui-btn sm soft" hidden>${ic('undo')}<span>Back to your rows</span></button>
    </div>`;
  // In the page's own layer (DESIGN.md: panels are 20), above #graph: the view's labels take
  // z-indexes up to their count inside it.
  document.body.appendChild(pop);
  const q = sel => pop.querySelector(sel);
  const find = q('.gp-find'), scroll = q('.gp-scroll'), none = q('.gp-none'), note = q('.gp-note'), back = q('.gp-back');
  const secs = [...pop.querySelectorAll('.gp-sec')];
  const items = [...pop.querySelectorAll('.gp-item')].map(el => ({ p: byKey.get(el.dataset.key), el })); // screen order
  const HINT = 'Loading a preset replaces the rows and the camera. Undo, in the toast or here, brings yours back.';

  function paintBack() {
    const s = stack.at(-1);
    back.hidden = !s;
    if (s) back.title = `Bring back the rows and camera from before ${s.label}`;
  }

  let act = null;
  function setActive(it, scrollTo = true) {
    if (act) { act.el.classList.remove('act'); act.el.setAttribute('aria-selected', 'false'); }
    act = it || null;
    if (act) {
      act.el.classList.add('act');
      act.el.setAttribute('aria-selected', 'true');
      find.setAttribute('aria-activedescendant', act.el.id);
      if (scrollTo) act.el.scrollIntoView({ block: 'nearest' });
    } else find.removeAttribute('aria-activedescendant');
    note.innerHTML = act ? `<b>${esc(act.p.label)}</b> ${esc(act.p.note)}` : esc(HINT);
  }
  function filter() {
    const text = find.value;
    let first = null, named = null;
    const words = fold(text).split(/\s+/).filter(Boolean);
    for (const it of items) {
      const hit = matches(it.p, text);
      it.el.hidden = !hit;
      if (!hit) continue;
      first ||= it;
      if (!named && words.every(w => fold(it.p.label).includes(w))) named = it;
    }
    for (const s of secs) s.hidden = !s.querySelector('.gp-item:not([hidden])');
    none.hidden = !!first;
    none.textContent = first ? '' : `No preset matches "${text.trim()}"`;
    setActive(words.length ? named || first : null);
  }
  const shown = () => items.filter(it => !it.el.hidden);
  function step(d) {
    const list = shown(), i = list.indexOf(act);
    if (list.length) setActive(list[i < 0 ? (d > 0 ? 0 : list.length - 1) : (i + d + list.length) % list.length]);
  }
  // ← →: the item in the nearest column that way, at about the same height.
  function stepColumn(d) {
    const list = shown();
    if (!act) return setActive(list[0]);
    const r = act.el.getBoundingClientRect();
    let best = null, bestD = Infinity;
    for (const it of list) {
      const b = it.el.getBoundingClientRect(), dx = (b.left - r.left) * d;
      if (dx < 20) continue;
      const dist = dx * 1e3 + Math.abs(b.top - r.top);
      if (dist < bestD) { best = it; bestD = dist; }
    }
    if (best) setActive(best);
  }

  // Under the button, inside the window.
  function place() {
    const b = btn.getBoundingClientRect(), w = pop.offsetWidth, top = Math.round(b.bottom + 6);
    pop.style.top = `${top}px`;
    pop.style.left = `${Math.round(Math.max(12, Math.min(b.left - 4, innerWidth - w - 12)))}px`;
    pop.style.maxHeight = `${Math.max(240, innerHeight - top - 12)}px`;
  }
  const isOpen = () => !pop.hidden;
  function open(on) {
    if (on === isOpen()) return;
    pop.hidden = !on;
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-expanded', String(on));
    if (on) {
      find.value = '';
      filter();
      paintBack();
      place();
      scroll.scrollTop = 0;
      find.focus({ preventScroll: true });
    } else if (pop.contains(document.activeElement)) document.activeElement.blur();
  }
  function choose(key) {
    open(false);
    load(key);
  }

  btn.onclick = () => open(!isOpen());
  find.addEventListener('input', filter);
  pop.addEventListener('pointermove', e => { // not pointerover: keys that scroll the list under a still mouse keep their pick
    const el = e.target.closest?.('.gp-item');
    if (el && el !== act?.el) setActive(items.find(it => it.el === el), false);
  });
  pop.addEventListener('click', e => {
    const el = e.target.closest?.('.gp-item');
    if (el) choose(el.dataset.key);
  });
  back.onclick = () => { open(false); undo(); };
  document.addEventListener('pointerdown', e => {
    if (isOpen() && !pop.contains(e.target) && !btn.contains(e.target)) open(false);
  }, true);
  // While open the menu has the keys first (capture phase): Esc closes it, the arrows and Enter
  // choose, and any other key goes to the search field, so no 3D-tab shortcut fires. Alt+1..3 close
  // it and switch tabs as usual.
  window.addEventListener('keydown', e => {
    if (!isOpen() || ['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) return;
    if (e.altKey) { open(false); return; }
    const k = e.key;
    if (k === 'Escape') { open(false); btn.focus({ preventScroll: true }); }
    else if (k === 'ArrowDown' || k === 'ArrowUp' || k === 'Tab') step(k === 'ArrowUp' || (k === 'Tab' && e.shiftKey) ? -1 : 1);
    else if ((k === 'ArrowLeft' || k === 'ArrowRight') && !find.value) stepColumn(k === 'ArrowRight' ? 1 : -1);
    else if (k === 'Enter') { if (act) choose(act.p.key); }
    else {
      if (document.activeElement !== find) find.focus({ preventScroll: true }); // the key then types into it
      e.stopImmediatePropagation();
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  window.addEventListener('resize', () => { if (isOpen()) place(); });
  api.onViewChange(() => open(false));

  document.getElementById('g-help')?.insertAdjacentHTML('beforeend', `<h4 class="ui-overline">Presets</h4>
<p><b>Presets</b> in the header loads a ready-made scene: losses, softmax and attention, activations, optimization,
and linear algebra. It replaces the rows and camera; <b>Undo</b> in its toast, or <b>Back to your rows</b> in the menu, brings yours back.</p>`);

  api.presets = { list: PRESETS, load, undo, get canUndo() { return stack.length > 0; } };
  paintBack();
}
