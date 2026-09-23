# Net tab: shared contract

The third tab, **Net** (Alt+3), is an editable neural-network visualizer. You can place neurons
anywhere and click into a node, edge or layer to see and edit its parameters. The network is
also shown live as matrix multiplication, `z = W a + b`, and can be trained on toy datasets.

The tab uses `body[data-view="nn"]`, with its root at `<section id="nn" hidden>` (grapher.js's
`setView` toggles `hidden`). All code lives in `static/nn/`. It is plain ES modules: no build
step and no new dependencies. KaTeX is available as the global `katex`, and Three.js is available
through the import map.

## Files

Modules talk through the store, `ctx` and the handles listed below.

| file | role |
|---|---|
| `store.js` | shared state, undo, events, colours |
| `model.js` (tests: `tests/nn_model.test.mjs`) | network JSON, edits, maths, presets, datasets. Pure: no DOM |
| `nn.js`, `nn.css` | shell: layout, splitter, toolbar, shell keys, persistence, module loading, audience mirror API. Tab wiring: `index.html`, `graph/grapher.js`, `app.js`, `style.css`, `graph/features/lecture.js` |
| `view.js`, `view.css` | the SVG canvas in `#nn-stage` |
| `inspector.js`, `inspector.css` | floating cards for the selected node, edge or layer |
| `matrix.js`, `matrix.css` | the matrix panel in `#nn-matrix`, and step-through (`state.anim`) |
| `train.js`, `train.css` | the floating Train panel: datasets, training loop, plots, per-neuron maps |

## Network JSON (`model.js`)

```js
net = {
  v: 1,
  layers: [{ id, name, act }],   // array order = layer index. layers[0] = inputs, last = outputs
  nodes:  [{ id, layer, x, y, label, bias, value, target, params }],
  edges:  [{ id, from, to, w }],
  meta:   { title, loss: 'mse' | 'xent', nextId, train: {...} },   // meta.train: see below
}
```

A valid net always has at least 2 layers. `emptyNet()` is an empty Input and an empty Output layer.

**Ids and order**
- Ids are unique strings and are never reused. Get one from `model.uid(net, prefix)`.
- A node's order within its layer is its order in `net.nodes` (filtered by layer). That is its
  row in `W` and its entry in the layer vector. Only `moveNode` (and `addNode`'s `index`) set it.
- `x, y` is a free world position (CSS px at zoom 1). Dragging a node never changes its order.

**Node fields**
- `label` is KaTeX source. Defaults, 1-based: `x_{1}` for inputs, `h^{(1)}_{2}` for hidden
  nodes, `\hat y_{1}` for outputs. Structural edits renumber labels that still look like
  defaults (or are empty); custom labels are never touched.
- `bias` is ignored on layer 0.
- `value` is the input value; it only matters on layer 0.
- `target` is `number | null` and only matters on the output layer. When every output node has
  a numeric target, the store also runs the backward pass.
- `params` is a user-defined `{ key: string | number }` map shown in the inspector. The maths
  ignores it.

**Edges**
- Edges only go forward: `layerIndex(from) < layerIndex(to)`.
- A missing edge is a fixed 0 entry (masked).
- Skip edges (l-2 to l and so on) are allowed. They add an extra term `W^{(l,k)} a^{(k)}`.

**Activations**
- `act` is one of `identity relu leaky sigmoid tanh softmax`.
- `softmax` acts on the whole layer.

## `meta.train` (train.js)

The Train panel's settings and progress live in the net, so export, undo and the audience mirror
carry them. A preset or import may have `meta.train = {}`: always read it through
`readSettings(net, model)` (exported by `train.js`), which fills defaults and repairs junk without
mutating the net. train.js writes the full object back on the first setting change, step, reset,
adapt or play frame.

| key | type | default | meaning |
|---|---|---|---|
| `dataset` | string | see below | a `DATASETS` key |
| `n` | int 4..5000 | 200 | number of points |
| `noise` | number 0..10 | 0.1 | `make()` noise |
| `seed` | int | 1 | data seed |
| `lr` | number 1e-7..1000 | 0.1 | learning rate (the UI lists 0.0001..10) |
| `batch` | int >= 0 | 10 | mini-batch size; 0 = the whole dataset |
| `speed` | int 1..1000 | 5 | max `trainStep` calls per animation frame (the UI lists 1..100) |
| `initSeed` | int | 1 | seed Reset passes to `model.randomize` |
| `seen` | int >= 0 | 0 | samples consumed; epoch = `floor(seen / n)` |
| `steps` | int >= 0 | 0 | gradient steps taken |
| `hist` | number[] | `[]` | full-dataset loss (4 significant digits), one entry per `every` epochs; entry 0 is before training |
| `every` | int >= 1 | 1 | epochs per `hist` entry; past 400 entries `hist` is halved and `every` doubled |
| `maps` | bool | true | per-neuron maps on |
| `space` | string | `''` | plot: `''` = input space, or the id of a hidden layer with exactly 2 neurons |

- Default `dataset`: the preset's `PRESETS[k].dataset` (a net is matched to its preset by
  `meta.title`) if its shape fits, else the first dataset whose inputs and outputs match the net,
  else `xor`.
- Changing `dataset`, `n`, `noise` or `seed` resets `seen`, `steps`, `hist` and `every`.
- The loss choice stays in `meta.loss`.
- The panel's own UI state is not in the net: localStorage `mathboard.nn.train` =
  `{ open, fold, x, y }` (`x`/`y` null = the default top-right spot). The audience window never
  reads or writes it.

## `model.js` exports (pure; no DOM)

```js
ACTS            // { name: { label, tex, f(z), df(z, a), vector?, slope? } }. softmax is { vector: true }:
                //   forward/backward special-case it; its scalar f is sigmoid (softmax([z, 0])_1), for plots only.
                //   leaky.slope = 0.1
activate(act, z[], out?) -> a[]                  // a whole layer, softmax included
uid(net, prefix)
emptyNet(), clone(net), validate(net) -> string[], normalize(net) -> net   // normalize repairs loaded JSON
relabel(net, { onlyEmpty }), defaultLabel(net, nodeId)
PRESETS         // { key: { label, group, note, dataset: DATASETS key | null, build(seed = 1) -> net } }; build() also
                //   records the dataset in meta.train.dataset. Listed in menu order: the shell's New net picker has
                //   one <optgroup> per group (in first-seen order), shows note as the option's tooltip and toasts
                //   it for 5 s when the preset loads from the picker (not on a #nn= preload). note is one line on
                //   what to notice, matrix panel first. Hand-wired presets ignore the seed. meta.title is unique per preset.
                //   Basics: gates, xor_gates, xor_relu (hand-set), xor, perceptron, logreg, linreg, softmax_reg
                //     (the last three start at W = 0), linear (all identity: demo of collapse)
                //   MLPs: mlp, deep, classifier (2-4-3 softmax), wide, narrow_deep (49 parameters each), funnel,
                //     uat (hand-built ReLU hinges on the sine)
                //   Skip connections: residual, bottleneck, ffn, densenet, unet, wide_deep
                //   Structure in W: conv1d, conv1d_s2, avgpool, maxpool (b + ReLU(a - b)), lenet, gnn (hand-set),
                //     towers (block-diagonal), multitask (shared trunk, block-diagonal heads)
                //   Sequences: rnn (unrolled, same initial weights per step), wavenet (dilated causal conv)
                //   Embeddings & autoencoders: autoencoder, embedding (one-hot lookup), pca_ae (linear, on cloud)
                //   Teaching demos: vanishing (sigmoid chain), gan (D(G(z)) as one net)
layerIndex(net, layerId), nodeLayerIndex(net, nodeId), nodesIn(net, layerIdOrIndex) -> nodes[]
node(net, id), edge(net, id), edgeBetween(net, a, b)   // edgeBetween matches either direction

// edits: mutate net in place and keep validate(net) empty (removing a node removes its edges, a layer its nodes)
addLayer(net, at, { name, act, size = 2, dense = false, seed }) -> id   // nodes in a column between the neighbours;
                                                                        //   dense: also wire it to both neighbours
removeLayer(net, id, { bridge, seed }) -> bool  // refuses (false) at 2 layers. bridge: wire the two
                                                //   neighbours densely if nothing joins them (the shell's Delete uses it)
setLayer(net, id, { name, act }) -> bool
addNode(net, layer, { x, y, label, bias, value, target, params, index, connect, seed }) -> id | null
                // index = order within the layer; connect: wire it to every node of both neighbour layers
                //   (seeded Xavier). A new output gets target 0 when every other output has a target
removeNode(net, id), moveNode(net, id, toIndex)
setNode(net, id, patch) -> bool                 // x y label bias value target params layer. Numeric strings are
                                                //   accepted, bad values ignored; a new layer drops sideways/backward edges
connect(net, from, to, w?) -> id | null         // reuses an existing edge, swaps a backward pair, null within a layer;
                                                //   no w: a seeded random weight in (-1, 1)
disconnect(net, edgeId), setWeight(net, edgeId, w)
connectDense(net, fromLayer, toLayer, { seed, scheme = 'xavier', w }) -> edgeId[]   // existing edges keep their w
randomize(net, { seed, scheme: 'xavier' | 'he' | 'small', biases: 'zero' | 'small' | 'keep' })   // biases default 'zero'
autoLayout(net, { width, height })              // nodes in evenly spaced columns, centred

// maths
matrices(net) -> [ per layer l >= 1: { l, id, act, rows: nodeId[], b: number[],
                   terms: [{ k, cols: nodeId[], W: number[][], edge: (edgeId|null)[][] }] } ]
                 // matrices()[l - 1].l === l. The k = l-1 term always exists; other k only when a skip edge exists. Sorted k desc
forward(net, x?) -> { z: (number[]|null)[], a: number[][], node: { [id]: { z, a } } }   // x defaults to layer-0 values
backward(net, fwd, y, loss) -> { loss, note, dA: number[][], dZ: number[][], dW, db: number[][],
                                 node: { [id]: { da, dz } }, edge: { [id]: dw } }
trainStep(net, { X, Y }, { lr = 0.1, loss }) -> mean loss   // one step over the given batch (gradients averaged);
                                                            //   the loss is from before the update. Non-finite: no update
predict(net, X, { layer }) -> number[][]       // outputs; layer = index | id (that layer's a) | 'all' (every layer per sample)
collapse(net) -> { W, b, rows, cols } | null   // the affine map y = W x + b when every non-input layer is identity
DATASETS        // { key: { label, inputs, outputs, kind: 'class' | 'reg', make(n = 200, seed = 1, noise = 0) -> { X, Y } } }
                //   xor, circles, spiral, blobs, moons (2 -> 1), three (2 -> 3 one-hot), line, sine (1 -> 1),
                //   cloud (3 -> 3 regression, target = input: a flat 3-D cloud, for pca_ae).
                //   noise is Gaussian, on the inputs (class) or the targets (reg)
rng(seed) -> () => [0, 1)                      // mulberry32; no seed = a random one
fmt(x, digits = 2) -> string                   // ASCII minus, never -0.00, 'NaN' / 'inf' / '-inf'. KaTeX-safe
```

**Indexing**
- Every per-layer array from `forward` / `backward` is indexed by layer index l (0 = inputs).
  Layer-0 slots are placeholders: `fwd.z[0] = null`, `bwd.dZ[0] = bwd.db[0] = bwd.dW[0] = []`.
  `bwd.dA[0]` is real (dL/dx).
- `bwd.dW[l][t]` is parallel to `matrices()[l - 1].terms[t].W`. Masked entries are 0 in `dW`;
  `bwd.edge` only has existing edges.
- `bwd.node[id].dz` is null on the input layer. `bwd.note` is a string or null.

**Losses**
- `mse`: `½·mean over outputs of (a - y)²`.
- `xent` with a softmax output: `-Σ y_i log p_i`; the gradient is `p·Σy - y` (`p - y` for a
  proper distribution).
- `xent` with sigmoid outputs: binary cross-entropy, mean over outputs.
- `xent` on any other output layer: fall back to mse and say so in `backward().note`.

## `store.js`

```js
import { createStore, colorFor, POS, NEG, HI } from './store.js';
store.net       // the live net. Read it fresh every time; its identity never changes (undo/load mutate in place)
store.state     // { sel, hover, anim, fwd, bwd }. fwd / bwd are null when the recompute threw; bwd also without targets
store.model     // the model.js namespace
store.on(evt, fn(payload, store)) -> off      store.emit(evt, payload)      store.set(key, value)   // sets state[key], emits key
store.commit(label)           // after an edit: undo snapshot + recompute + emit 'net' then 'values'. store.lastLabel = label
store.touch()                 // continuous change (slider, drag, training tick): recompute + emit, no undo entry.
                              //   Call commit() when the gesture ends
store.layout()                // positions only changed: emits 'layout' (no recompute, no undo entry)
store.undo(), store.redo() -> bool, store.canUndo, store.canRedo
store.load(net | json, { history = true })   // swap in a whole net (normalized). history: false also clears undo/redo
```

**Events**
- `net` `{ structural }`. `structural` is true when ids or order changed (nodes, edges or
  layers added, removed or reordered). A structural change also clears `sel` / `hover` that
  point at something gone.
- `values`: `state.fwd` / `state.bwd` were recomputed. Always follows `net`.
- `layout`: node positions changed.
- `sel`: `null | { kind: 'node' | 'edge' | 'layer', id }`.
- `hover`: one of
  - `null`;
  - `{ kind: 'node' | 'edge' | 'layer' | 'bias', id }`;
  - `{ kind: 'pair', from, to }` (a masked matrix entry);
  - `{ kind: 'row', layer, i }` / `{ kind: 'col', layer, k, j }` (whole matrix row/column).
    `layer` and `k` are layer indices (the view also accepts ids), and `i` / `j` are 0-based.
  The view emits node, edge and layer; the inspector and the matrix panel emit every kind.
- `anim`: see below.
- `train`: `{ epoch, loss, running }`, emitted by `train.js` on play, pause, step, reset and every
  frame while training. `loss` is the full-dataset loss, or null. Nothing listens to it yet.

**`state.anim`** (step-through, owned by `matrix.js`)
- `null | { dir: 'fwd' | 'bwd', l, i, phase }`. `l` is a layer index >= 1 (the view also accepts
  a layer id) and `i` the 0-based row in that layer.
- `phase` is `'dot'` on every forward step (row i of W times the input, plus b_i, then the
  activation) and `'delta'` on every backward step (δ_i, row i of ∂L/∂W, ∂L/∂b_i). matrix.js
  sets no other values.
- Order: every row of layers 1..L forward, then, only when `state.bwd` exists, every row of
  layers L..1 backward. From null, S starts at the first forward row (the first backward row
  when Backward is on) and Shift+S at the last step. Stepping past either end, or ■, sets null.
- A step switches Batch off; a bwd step switches Backward on. matrix.js resets `anim` to null
  when it no longer fits the net (row gone, or `bwd` without `state.bwd`), when Batch is turned
  on, and when Backward is turned off during a bwd step.
- view.js lights neuron i of layer l (class `lit`). For fwd it also lights its incoming edges and
  their sources, for bwd its outgoing edges and their targets; everything else dims, and a pulse
  runs along the lit edges (source to target for fwd, reversed for bwd). `phase` only restarts
  the pulse when it changes.
- matrix.js highlights row i of W, b_i and (z_i, a_i) for fwd, or the backward row for bwd, plus
  the input vector. It shows the step's arithmetic in a box under layer l, and its toolbar reads
  `forward · layer l · row i/n`.

**Performance**
- `net` / `values` can fire every animation frame while training.
- On non-structural changes, update attributes and text in place. Rebuild DOM only on
  `structural` (matrix.js also rebuilds when its toggles, labels or loss case change).

**Colours**
- `colorFor(v, max = 1, theme = 'dark')` gives a diverging `rgba()`: `POS` blue for v >= 0,
  `NEG` orange for v < 0, alpha `0.12 + 0.88·min(1, |v|/max)`. `max` 0 counts as 1; a
  non-finite v is grey. The light theme uses darker RGB (31,111,209 / 216,82,44).
- `POS #4aa3ff`, `NEG #ff7a59` and `HI #ffd54a` are the dark-theme values. `HI` is the colour for
  hover, selection, the lit step and the current sample (view.css's `--nnv-hi` is `#e8a800` on
  the light theme).
- Shared scales, each `|| 1`, so an edge, its matrix cell and its inspector slider always match:

| scale | max over | used by |
|---|---|---|
| weights | \|w\| of every edge | view edge colour and width (1.2 + 5·\|w\|/max px); inspector weight sliders; matrix W cells (also in `[W \| b]` and the Collapse chain) |
| biases | \|b\| of every node | inspector bias sliders; matrix b cells (and the b column of `[W \| b]`). The view doesn't colour biases |
| activations | every entry of `fwd.a` and `fwd.z` | view neuron fill; matrix x, a, z, ŷ and y cells. Batch cells keep the live sample's scale and saturate |
| gradients | every entry of `bwd.dZ` and `bwd.dA`, and max\|dZ[l]\|·max\|a[k]\| for every term k of every layer l | view δ ring (colour, width 1.5 + 4·\|δ\|/max); matrix δ, ∂L/∂a, ∂L/∂W and ∂L/∂b cells |

- Other scales: matrix σ'(z) cells use max 1 and W_eff / b_eff their own max; inspector
  input-value sliders use the activations scale. The inspector's `solid(v)` is the full-strength sign colour
  (weights in its KaTeX, slider accents).
- train.js takes the POS / NEG RGB from `colorFor(±1, 1, theme)` and blends onto `--bg`. Output
  field: one output is coloured by the sign of `(out - mid) / half` (mid and half from the
  dataset's target range), alpha `0.05 + 0.62·|v|`; K outputs by the argmax class (class 0 NEG,
  1 POS, then 5 extra colours), alpha by the margin over the runner-up. Per-neuron maps: sigmoid
  and softmax on [0, 1], tanh on [-1, 1], anything else on ± that neuron's max |a| over the
  grid, alpha as `colorFor`. `HI` rings the current sample.

## DOM and module interface (shell)

```html
<section id="nn" hidden>
  <header id="nn-bar"> shell toolbar ... </header>
  <div id="nn-main">
    <div id="nn-stage"></div>   <!-- view mounts the SVG; inspector and train float over it -->
    <div id="nn-split"></div>   <!-- drag to resize, double-click to hide / show the matrix panel -->
    <aside id="nn-matrix"></aside>
  </div>
</section>
```

`nn.js` imports `model.js` and `store.js` first (if either fails, the stage shows an error and
`body.dataset.nnReady = 'error'`). It then imports `view`, `inspector`, `matrix` and `train` in
parallel and installs them in that order, awaiting each `export function install(ctx)`. A module
that fails to import or install is logged and skipped.

```js
ctx = {
  store, model, el: { root, bar, stage, matrix },
  addButton({ label, title, onClick, group = 'modules' }) -> <button>,   // into #nn-bar; label is HTML
  toast(msg, ms?), theme() -> 'dark' | 'light', onTheme(fn(theme)) -> off, onShow(fn(visible)) -> off,
  active(e?) -> bool,     // Net tab visible, not the audience window, and e (if given) is not typing into a text
                          //   input, textarea, select or contenteditable (range / checkbox / radio / button inputs don't count)
  audience: bool,         // read-only mirror window: render only, never edit or persist
  graph,                  // getter: window.mathboardGraph or null (3D tab API: rows, rowByName, addRow, setRowSource,
                          //   removeRow, setRows, setView, toast, audience: { open(), isOpen } from lecture.js, ...)
  view,                   // set by view.js during its install (see below)
  inspector,              // set by inspector.js
  matrix,                 // set by matrix.js (test / debug handle)
}
ctx.view = {
  svg, worldToScreen(x, y) -> { x, y },    // world -> px relative to #nn-stage
  screenToWorld(x, y) -> { x, y },         // px relative to the SVG (it fills the stage) -> world
  nodeRect(id) -> { x, y, w, h } | null,   // stage px, for placing the inspector
  fit(ms?),                                // frames the net, beside a tall panel floating at one side (Train).
                                           //   The view refits by itself when that panel opens, folds or closes,
                                           //   unless the user has panned or zoomed since the last fit
  outOfView() -> bool,                     // part of the net is off the stage or under that panel
  contentRect() -> { x, y, w, h },         // stage px box of the whole net (lanes and headers included)
  setNodeImage(id, url | null),            // train's per-neuron maps, clipped inside the node; kept across rebuilds
  png(scale = 2) -> Promise<dataURL>,      // the net on the theme background, without hover, selection or handles
  weights,                                 // get / set: the W labels toggle
}
ctx.inspector = {
  open(target, { pin = true }) -> card element | null,   // a target that has a card reuses it; pin: false only selects
  closeAll(), cards -> [{ target, pinned, el }],
}
ctx.matrix = { step(±1), toggle(key), opt, render(), update() }   // not a contract: for tests and the console.
               // opt = { mode: 'fwd' | 'bwd', bias, batch, collapse, labels }; render() forces a rebuild
```

- Toolbar: the shell's groups `net` (New net…, + Layer, Layout, Fit, Randomize), `edit` (↶ ↷) and
  `file` (Export, Import, PNG, To board) come first, then module groups in the order they are
  first used (view's `view`: Weights; train's `train`: Train), then a spacer, **Audience** and
  `?`. Audience calls `window.mathboardGraph.audience.open()` (set by
  `graph/features/lecture.js`), the same window the 3D tab's Audience button opens; it is lit
  while that window is open (`audience.isOpen`, checked every second while the tab is shown).
  Toolbar buttons never take focus, so Space stays with training.
- train.js sets no `ctx` field. Its named exports are `readSettings(net, model)`, `netShape`,
  `defaultDataset`, `forwardMany(net, model, X, n, from, M)`, `datasetLoss` and
  `adaptNet(net, model, ds, { seed })`; matrix.js imports `readSettings` for its Batch view. Test
  handle: `document.querySelector('.nn-train').nnTrain` =
  `{ play, pause, step, reset, adapt, loadSample(i), running, eval, point(i) }`.

**Keys in the Net tab** (only when `ctx.active(e)`; never while typing)

| owner | keys |
|---|---|
| shell | Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) undo/redo, Delete/Backspace remove selection, Esc close the cheat sheet or deselect, F fit, H hide UI, ? cheat sheet |
| view | W: weight labels on edges |
| matrix | S: step, Shift+S: step back, B: bias-trick toggle |
| train | Space: play/pause training, T: single training step |

Alt+1 / Alt+2 / Alt+3 switch tabs (grapher.js, always).

**CSS loading**
- `index.html` links `nn/nn.css`.
- `nn.js` links `nn/<module>.css` only after `<module>.js` has imported, just before its install,
  and never twice. inspector.js also links its own CSS if it is missing.
- `view.js` `png()` draws a clone of the SVG as an image, so it fetches `view.css` again and the
  KaTeX CSS with its woff2 fonts inlined as data URLs, and resolves the `--nnv-*` variables inline.

**Clean view (H)**

H toggles `body.clean`, the same clean view as the board and the 3D tab, and in the Net tab it
matches what the audience window shows. It hides the toolbar, the cheat sheet, the matrix
panel's toolbar (and its → 3D buttons), the view's connect handle and the empty-net hint, and
freezes the splitter. Inspector cards and the Train panel stay as demo content without their
chrome: cards lose the pin, close and param-delete buttons, "+ add" and the drag cursor (they
still work); the Train panel keeps its readout, loss chart, warning text and plots and hides its
settings and buttons (as `.nn-train.ro`). These rules live in `nn.css`, under `body.clean #nn`.
`#nn-main` sits in grid row 2, so the stage keeps the full height when the toolbar is hidden.

**Audience mirror**

nn.js publishes `window.mathboardNet = { store, ctx, audience, ready, mirrorState(), applyMirror(m),
onMirror(fn) -> off }`. `ready` turns true, and `window` gets the `mathboard:nn-ready` event, once
every module has installed.
- `mirrorState()` -> `{ net, sel, hover, anim, split, matrixHidden, matrix, weights }`, where
  `matrix` is a copy of `ctx.matrix.opt` (or null) and `weights` the W labels toggle.
- `onMirror(fn)` fires on the store's `net`, `layout`, `sel`, `hover` and `anim` events, when
  the split changes, and one frame after any click in the toolbar or the matrix panel or a key
  up in the tab (the matrix toggles and W have no store event).
- `applyMirror(m)` does nothing outside an audience window. It loads `m.net` with
  `{ history: false }` when its JSON changed, copies `m.matrix` into `ctx.matrix.opt` and
  re-renders the panel when a toggle differs, sets W, sets `sel` / `hover` / `anim` when they
  differ, and applies the split.
- `graph/features/lecture.js` carries it over its BroadcastChannel. The presenter posts
  `{ nn: mirrorState() }` at most once per frame, leaving out `net` when it hasn't changed, and
  also sends a full snapshot (including `view` and the net) every second. The audience switches
  to the Net tab when the presenter's `view` is `nn` (anything else shows the 3D view) and passes
  `nn` to `applyMirror`, buffering it until the Net tab is ready.
- Only that state is mirrored. Each window keeps its own pan / zoom (the audience view refits
  when the net leaves the frame), pinned cards and Train panel layout (the audience always shows
  the Train panel, read-only).

**Persistence and test preload**
- localStorage `mathboard.nn` = `{ v: 1, net, split, matrixHidden }`, owned by the shell, written
  500 ms after a `net` or `layout` event and on `pagehide`. `mathboard.nn.train` belongs to
  train.js (see `meta.train`).
- `index.html#nn=<preset key>`, `#nn=<base64url JSON>` or `#nn=<URI-encoded JSON>` opens the Net
  tab with that net (presets built with seed 1) and never saves. Use this for Playwright tests.
- Audience windows (`?audience`) never save.
- The shell sets `document.body.dataset.nnReady = '1'` once every module has installed.

## Testing

- `node --test "tests/*.test.mjs"` must stay green.
- For UI, use Python Playwright:
  - headless Chromium with `--use-angle=swiftshader --enable-unsafe-swiftshader`;
  - serve `static/` with `python -m http.server 8821` (any free port);
  - **never start `server.py`** (it loads a GPU model).
- Keep throwaway scripts outside the repo (e.g. under `%TEMP%`) and delete them when you're done.
