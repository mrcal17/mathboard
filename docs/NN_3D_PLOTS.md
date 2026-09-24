# Net tab: the 3D plots panel (contract)

This extends docs/NN_CONTRACT.md. It adds one store state key, `s3d`, carried to the audience
window, and one module, `surf3d.js` with `surf3d.css`, installed after `view3d`. Toolbar button:
**3D plots** (group `surf3d`). Keys: **P** opens or closes the panel, **Shift+P** switches to the
next plot. Unit tests for its pure helpers: `tests/surf3d.test.mjs`.

The panel floats over `#nn-stage` like the Train and Attention panels. It draws with Three.js
(`three`, `three/addons/controls/OrbitControls.js`, `three/addons/renderers/CSS2DRenderer.js`
from the import map), imported the first time the panel opens, so a closed panel costs nothing.
It has its own WebGL context. Labels are KaTeX in a CSS2D layer.

| plot | shows | needs |
|---|---|---|
| surface | one neuron's value as a height over the input plane, with the training points | 2 inputs |
| landscape | the dataset loss on a plane through the weights, with the training path | a dataset that fits the net, 2 to 5000 parameters |
| space | the dataset in each layer's activation space, morphing from the inputs to the outputs | a dataset with the net's inputs |
| simplex | a 3-class softmax: the predicted probabilities on the triangle ŷ₁ + ŷ₂ + ŷ₃ = 1 | a softmax output of 3 neurons (not a token layer) |

A plot the net can't have shows why in the view (with presets to try), and its mode button is dimmed.

## `state.s3d` (owner: surf3d.js)

`null` = the panel is closed. The panel is open exactly when it is set, so the audience mirror
drives it through the store alone. Always set a complete object (`ctx.surf3d.show` fills one in).

```js
s3d = null | {
  mode: 'surface' | 'landscape' | 'space' | 'simplex',
  // surface
  neuron: 'sel' | 'out' | nodeId,   // 'sel': follow the selection (a node; an edge's target; a layer's first
                                    //   neuron), else the output. 'out': the output; with 2+ outputs "the winning class"
  pre: bool,                        // true: z (the input sum W a + b), false: a = f(z). Ignored on inputs and attention
  // landscape
  dirs: null | 'random' | 'pca' | 'weights',   // null: 'weights' for a net with 2 parameters, else 'random'
  seed: int,                        // random directions (↻ adds 1)
  wa, wb: paramKey | null,          // 'weights': the two parameters (null: the selected edge's, else the first two)
  range: number,                    // half-width of the plane; 0 = auto
  log: bool,                        // height by log₁₀ L
  basis: null | {                   // the plane, written by the presenter (the audience draws what it says)
    id, sig, data,                  //   sig: the parameter keys joined; data: the dataset key. A mismatch = stale
    kind: 'random' | 'pca' | 'weights', wa, wb,
    theta0: number[], d1: number[], d2: number[],   // θ₀ and the two directions, 10 significant digits
    span: number,                   //   the half-width in use (auto grows it)
    share: number | null,           //   pca: the share of the path's variance in the plane
    note: string,                   //   e.g. why PCA fell back to random directions
  },
  trail: [[α, β, L], ...],          // the training path projected onto the plane, at its true loss (5 digits)
  // space
  stage: number | null,             // 0 .. stages - 1, fractional while morphing; null = the default stage
  cam: null | { p: [x, y, z], t: [x, y, z] },   // camera position and orbit target; null = the plot's default view
}
```

- The presenter writes `cam` at most every 120 ms while orbiting, and `stage` every frame while the
  morph plays. Changing only `stage` or `cam` re-places the drawing without new forward passes.
- A new `meta.title` (a preset, an import) resets `neuron`, `pre`, `stage`, `basis`, `trail`, `wa`,
  `wb` and `dirs`, and keeps the mode and camera. A neuron that is gone falls back to `'sel'`.
- `ctx.surf3d.show(mode, patch)` merges `patch` into the state (switching mode sets `cam` null
  unless the patch has one). `hide()` sets null.

**UI state** (not in the store): localStorage `mathboard.nn.s3d` = `{ x, y, w, h, mode }`: the
dragged spot (null = top left, or right of an open Attention panel there), the panel width
(default 460, at least 320), the view's height (default 340, at least 180) and the last mode, which
P and the button open. The audience reads the presenter's through `storage` events, as for the
Attention panel. The view's height shrinks to fit the stage below the panel's top.

## The panel

- **Header**: the four mode buttons, ⌂ (the default view) and × (P). Drag it to move the panel,
  double-click it to put it back; the corner grip sets the width and the view's height.
- **View**: drag to orbit, right-drag (or Shift-drag) to pan, wheel to zoom (OrbitControls with
  damping). Pointer and wheel input stay in the panel.
- **Caption**: what is plotted, in KaTeX with live numbers, and a note line.
- The audience window and the clean view (H) show the plot and its caption with only the active
  mode's button: no controls row, ⌂, × or grip, and the audience can't orbit (it follows `cam`).

## Surface

For a 2-input net, one neuron's value over a 49 × 49 grid of the input plane. The domain is the
Train plot's: the square box around the dataset's points, 10% larger (with no 2-D dataset,
±max(1.5, |x| + 0.5) around the inputs).

- **Which neuron**: `neuron`. Clicking a neuron on the canvas selects it, so with "follow the
  selection" the lecturer clicks through the layers and the surface follows. The output of a net
  with 2+ outputs is "the winning class": colour = argmax ŷ, height = max ŷ (its margin over the
  runner-up sets the colour's strength).
- **Height**: `z` or `a = f(z)`, each from one batched forward pass (the dense path is train.js's
  `forwardMany` keeping z too; a net with attention goes through `model.predict`, which has no z).
  sigmoid, softmax and tanh values use their fixed range; anything else the grid's range (with 0
  for z, ReLU and identity, and the targets for the output), which grows at once and shrinks slowly
  while training.
- **Colour**: `colorFor`'s blue and orange by the sign around the range's centre, on a neutral grey.
- **The flat plane**: the level where the neuron changes its answer: 0.5 for sigmoid (on the output,
  the decision boundary), 0 for tanh and for z. Its crossing with the surface is drawn in HI.
- **Points**: the output shows the training points at their targets, a hidden neuron at its own
  value, the winning class on the floor. The ● (HI) is the net's current input.
- **Caption**: a first-layer neuron gets its formula with the live weights, e.g.
  `h⁽¹⁾₂ = tanh(0.83 x₁ − 1.20 x₂ + 0.10)`: the plane z, bent by tanh. Deeper neurons get the
  general form.
- Live while training: at most one refresh per 90 ms.

## Landscape

The loss `L(θ₀ + α δ₁ + β δ₂)` over the plane through the weights θ₀ spanned by two directions.

- **Parameters**: exactly what `model.trainStep` moves, in reading order (layer by layer, its
  weights row by row, then its biases). A tie group is one parameter; fixed edges and attention
  biases are not parameters (`paramList`).
- **The loss**: train.js's `datasetLoss` with the net's loss (`meta.loss`), over the Train panel's
  dataset (at most 500 samples, evenly spaced, when it has more; the caption says so). At the centre
  it is the Train panel's loss.
- **Directions** (`dirs`):
  - `random`: two Gaussian directions, filter-normalised as in Li et al., "Visualizing the Loss
    Landscape of Neural Nets" (2018): each filter's slice of δ is scaled to the norm of the same
    filter of θ₀. A filter here is one neuron's incoming weights and its bias (a tied matrix
    `W:i,j`: its column j). A filter whose weights are all 0 (the W = 0 presets) gets the RMS of
    the other filters' norms, or 1, so the plane is never flat there. In a small net two draws can
    be nearly parallel: δ₂ is redrawn (up to 12 times) until |cos(δ₁, δ₂)| ≤ 0.5. α = ±1 moves every
    neuron by its own weights' norm.
  - `pca`: the top two principal directions (unit vectors) of the recorded path plus θ₀. With
    fewer than 3 path points it falls back to random directions and says so. Random directions
    catch little of a training path (it runs mostly across them); PCA puts it in the plane.
  - `weights`: two chosen parameters as unit directions, every other parameter held at θ₀; the
    axes then read the weights' own values. A net with 2 parameters (linreg: w and b) starts here,
    and the slice is the whole loss surface.
- **θ₀** is the weights when the plane was made: on opening, Re-center, a change of directions,
  seed, weights or range, a structural or dataset change, and a Reset in the Train panel (step 0
  with new weights). It stays put while training, so the ● rolls across a fixed surface.
- **Range**: `range`, or auto: 1 (random, weights) or 0 (pca) at least, grown to 1.15 × the path's
  and the current point's extent, rounded up (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 × 10ᵏ). When the ●
  passes 98% of it, auto widens it to 1.3 × and the grid is recomputed.
- **Grid**: an 11 × 11 preview, then 31 × 31 (odd, so the centre is a vertex: α = β = 0 exactly),
  computed a few milliseconds per frame. A net without attention patches the entries of
  `model.matrices` in place and runs `forwardMany`; a net with attention writes a scratch copy
  and runs `model.predict`.
- **Height**: from the grid's minimum to min(max, min + 1.3 × (max(85th percentile, the path's
  highest loss, the current loss) − min)); anything higher is drawn flat at the top and the note
  says where the top was cut. `log` maps log₁₀ L instead. Viridis by height, 8 contour lines on the
  surface and faintly on the floor.
- **The path**: recorded by the presenter from the Train panel's `train` events whatever the panel
  shows (at most every 60 ms while running, and at each pause or step; 400 points, halved when
  full): the full parameter vector and its loss. Undoing training cuts it back to the current step;
  a Reset starts it again. Its least-squares coordinates in the plane go into `trail`. It is drawn
  as a tube at its **true loss**, so where the net left the plane it floats above or sinks below the
  slice; a faint copy shows through the surface, and its shadow runs on the floor. The note gives
  the current point's loss and its distance from the plane (0 when it is on it).
- The ● is the current weights, projected, at their true loss.

## Space and simplex

The dataset's points (in their class colour; a regression by its target) at each **stage**: the
input `x`, then for each layer `z⁽ˡ⁾ = W a + b` and `h⁽ˡ⁾ = f(z⁽ˡ⁾)` (one stage for an identity layer;
`ŷ` for the output; a net with attention has no z stages). The stage chips under the slider jump
to a stage; ▶ plays the morph from the current stage to the output, slowing near each stage.

**Mismatched sizes.** Every stage is placed in the same 3-D box:
- d ≤ 3 coordinates are the first d axes and the rest are 0 (a 2-D input is the plane z = 0 in the
  middle of the box, a 1-D layer the x axis);
- d > 3 is projected onto its top three principal components (PCA of that stage's points; the
  projection is not centred, so the origin stays at the centre), with each component's sign chosen
  to agree with the previous stage so the morph doesn't flip. The caption says how much of the
  spread the three hold, and that a PCA view is a shadow.

Between two stages each point moves on a straight line from its place in one to its place in the
next. From `a⁽ˡ⁻¹⁾` to `z⁽ˡ⁾` that is the affine map (lines stay lines, parallel lines stay
parallel, in the drawing too when both sides are 3-D or less); from `z⁽ˡ⁾` to `h⁽ˡ⁾` it is the
activation acting on each axis. Each stage is scaled to fill the box (the ticks at the axis ends
give ±that scale), except a tanh, sigmoid or softmax stage with ≤ 3 neurons, whose box is its range:
saturated points sit on its faces. For a 2-input net the input grid lines are carried through
every stage, so the plane visibly bends and folds.

**Simplex** (a 3-class softmax output): the output stage alone, as probabilities in the unit cube
(world = 2p − 1), seen along (1, 1, 1): the triangle ŷ₁ + ŷ₂ + ŷ₃ = 1, its corners labelled, and the
three lines from its centre to the edge midpoints where two classes tie. In Space mode the same
triangle is drawn at a 3-neuron softmax output stage.

## Handle (tests, other modules)

```js
ctx.surf3d = panel.nnS3d = {
  MODES,                           // ['surface', 'landscape', 'space', 'simplex']
  open, mode,                      // getters
  show(mode?, patch?) -> bool,     // false in the audience
  hide(), toggle(mode?), cycle(dir = 1),
  recenter(),                      // landscape: a new plane through the current weights
  clearPath(),                     // landscape: forget the path (keeps the current point)
  available() -> { surface, landscape, space, simplex },   // '' or why not
  probe() -> { mode, gl, error, msg, busy, stats, surface?, landscape?, space? },
  el,                              // the panel
}
```

`probe()` reports what the last drawing used: `surface` = `{ G, dom, lo, hi, zh, neuron, xs, ys,
values, meshZ }` (the plotted value at every vertex and its mesh height), `landscape` = `{ G, span,
kind, center, current: { a, b, r, loss }, trail, done, lo, hi, n, basis }`, `space` = `{ stage,
stages: [{ l, part, d, e, off, pca, share }], coords (the first 5 points of the nearest stage, in the
box), X, n }`. `busy` is true while a grid is computing, a refresh is pending or the morph plays.
The pure helpers (`paramList`, `readParams`, `writeParams`, `filterNormalize`, `randomDirections`,
`project`, `symEig`, `pathDirections`, `isoLines`, `niceTicks`, `niceUp`, `plainLabel`, `forwardZA`,
`makeEvaluator`) are named exports; the module imports Three.js only when the panel opens, so
Node can import it.

## Hooks this module needs from others

- **View fit (view.js)**: `FLOATS` should include `.nn-s3d`, so `fit()` and `outOfView()` keep the
  net clear of this panel and refit when it opens, closes, resizes or is dragged (view.js already
  watches every stage child's size and a pointerup in the stage). Until then the view ignores it.
- **Inspector cards** (inspector.js) avoid `.nn-train` when placing a card; `.nn-s3d` could join it.
- Explain's caption card already counts every panel over the stage.

## Mirror

nn.js carries `s3d` in `mirrorState()`, calls the `onMirror` listeners on its store event, and
`applyMirror` sets it when it differs. The audience never writes it: it computes the surface, the
landscape grid (from `basis`) and the stages from its own mirrored net, draws the presenter's
`trail`, and eases its camera toward `cam`.

## Testing

`node --test tests/surf3d.test.mjs`: the parameter list holds exactly what `trainStep` moves (ties
and fixed edges included), the evaluator's centre is the dataset loss through `model.predict` (an
attention net too), filter normalisation, the projection, the path's principal directions,
`symEig`, `isoLines`, and `forwardZA` against `model.forward`. In a browser (Python Playwright, the
recipe in docs/FEATURE_GUIDE.md) `ctx.surf3d.show(mode)`, then wait for `!probe().busy`.
