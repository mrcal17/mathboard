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
| `focus.js` (tests: `tests/nn_focus.test.mjs`) | the lens rules: what a lens emphasizes, dims and hides; token names; stages. Pure: no DOM (docs/NN_LENS.md) |
| `lens.js`, `lens.css` | the lens bar at the bottom of `#nn-stage`; owns `state.lens` (docs/NN_LENS.md) |
| `attnviz.js`, `attnviz.css` | the floating attention panel (arcs, dots, mix, heat); owns `state.viz` (docs/NN_LENS.md) |
| `tour.js`, `tour.css` | Explain, the guided walkthrough and its caption card; owns `state.tour` (docs/NN_LENS.md) |
| `view3d.js`, `view3d.css` | the 3D net view over `#nn-stage` (stack, heads, tensor); owns `state.v3d` (docs/NN_3D.md) |
| `surf3d.js`, `surf3d.css` | the floating 3D plots panel (surface, landscape, space, simplex); owns `state.s3d` (docs/NN_3D_PLOTS.md) |
| `flow.js`, `flow.css` (tests: `tests/nn_flow.test.mjs`) | the Flow view over `#nn-stage`: the whole forward pass as matrix tiles, stage by stage; owns `state.flow` (docs/NN_FLOW.md) |

## Network JSON (`model.js`)

```js
net = {
  v: 1,
  layers: [{ id, name, act, tokens?, groups?, kind?, heads?, causal?, scale?, window?, pos?, linear? }],   // array order =
                                 //   layer index. layers[0] = inputs, last = outputs. Optional fields: docs/NN_ATTENTION.md,
                                 //   window / pos / linear: docs/NN_FLOW.md, Variants
  nodes:  [{ id, layer, x, y, label, bias, value, target, params, tie?, fixed? }],
  edges:  [{ id, from, to, w, tie?, fixed? }],
  meta:   { title, loss: 'mse' | 'xent', nextId, train: {...} },   // meta.train: see below
}
```

The optional fields (all absent in older nets, which load unchanged):
- `layer.tokens` (int >= 1) and `layer.groups` (distinct strings, e.g. `['Q','K','V']`) shape a layer
  as token rows: node k is group `floor(k / (tokens·d))`, token `floor(k / d) % tokens`, feature
  `k % d`. `layer.kind` is `'dense'` (the default) or `'attention'`, with `heads`, `causal` and `scale`.
- `edge.tie` (string or null): edges with the same tie share one weight. `edge.fixed` (bool): never
  trained, randomized or re-weighted. `node.tie` (string or null): nodes with the same tie share one
  bias (not on the input layer or an attention layer). `node.fixed` (bool): the node's bias is never
  trained, randomized or set (its gradient is still reported); fixed beats tie, as on edges.
- An attention layer's `window` (int >= 1: query i reads keys j with |i - j| < window), `pos`
  (`'rope'` | `'alibi'`) and `linear` (true: no softmax): docs/NN_FLOW.md, Variants.
- `meta.vocab` (string[]): names the one-hot slots of a token input and a softmax output of that
  width; `meta.flow` (true): the net opens in the Flow view when it loads (docs/NN_FLOW.md).
  `normalize` keeps them, as any unknown meta field.

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
- `tie` (optional string, e.g. `b_Q:2`): nodes with the same `tie` share one bias. `setNode`'s
  `bias` sets the whole group, `randomize` draws once per group and `trainStep` steps it by the
  summed gradient (`bwd.tie[tie]`). Never on the input layer or an attention layer.

**Edges**
- Edges only go forward: `layerIndex(from) < layerIndex(to)`.
- A missing edge is a fixed 0 entry (masked).
- Skip edges (l-2 to l and so on) are allowed. They add an extra term `W^{(l,k)} a^{(k)}`.

**Activations**
- `act` is one of `identity relu leaky sigmoid tanh softmax`, or one of the transformer variants'
  `gelu layernorm rmsnorm swiglu` (docs/NN_FLOW.md, Variants).
- `softmax` acts on the whole layer, or per token (and group) on a token layer; so do `layernorm`
  ((z - mean) / sqrt(variance + ε)) and `rmsnorm` (z / sqrt(mean(z²) + ε)), ε = `NORM_EPS` = 1e-5, with no
  learned gain or shift. `swiglu` gates the layer's second half by its first: a = [silu(g) ⊙ u, u]
  (with groups G, U, the groups). Softmax and these three are `vector: true` in `ACTS` (layer-wide);
  `gelu` (the tanh form) is scalar.
- An attention layer's `act` is always `identity`.

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
| `maps` | bool | true | per-neuron maps on (always off with a `seq` dataset; the setting comes back with a plain one) |
| `space` | string | `''` | plot: `''` = input space, or the id of a hidden layer with exactly 2 neurons. With a `seq` dataset: `'<attention layer id>#<head>'`, the attention matrix of that layer and head (0-based; `''` or an unknown value picks the first) |
| `init` | `{ name: scheme }` | absent | optional: how Reset draws the shared matrices with that name (`model.randomize`'s `init`, schemes `INIT_SCHEMES`); entries with an unknown scheme are dropped. The word presets record `{ W_Q: 'small', W_K: 'small', W_V: 'identity' }` |

- Default `dataset`: the preset's `PRESETS[k].dataset` (a net is matched to its preset by
  `meta.title`) if its shape fits, else the first dataset whose inputs and outputs match the net,
  else `xor`.
- A preset may also record `lr` (`PRESETS[k].lr`, one of the listed rates): the attention presets
  need more than the default 0.1 to learn their pattern in a few thousand steps. It may record
  `noise` the same way (`PRESETS[k].noise`): the word presets train on the exact word vectors
  (noise 0).
- Changing `dataset`, `n`, `noise` or `seed` resets `seen`, `steps`, `hist` and `every`.
- The loss choice stays in `meta.loss`.
- The panel's own UI state is not in the net: localStorage `mathboard.nn.train` =
  `{ open, fold, settings, x, y }` (`settings`: the Settings section, which holds the
  hyperparameters, is unfolded; it starts folded. `x`/`y` null = the default top-right spot). The audience window never
  writes it; it follows the presenter's `open` and `fold` through `storage` events and keeps the
  default spot.

## `model.js` exports (pure; no DOM)

```js
ACTS            // { name: { label, tex, f(z), df(z, a), vector?, slope? } }. softmax is { vector: true }:
                //   forward/backward special-case it; its scalar f is sigmoid (softmax([z, 0])_1), for plots only.
                //   leaky.slope = 0.1
activate(act, z[], out?, seg?) -> a[]            // a whole layer, softmax included; seg: softmax per run of seg entries
uid(net, prefix)
emptyNet(), clone(net), validate(net) -> string[], normalize(net) -> net   // normalize repairs loaded JSON, token /
                //   tie / attention fields included: bad fields dropped, tie groups equalized (first member wins),
                //   fixed beats tie, edges into an attention layer dropped and its biases zeroed, a layer whose size
                //   doesn't split into tokens x groups made plain, a broken attention layer made dense
relabel(net, { onlyEmpty }), defaultLabel(net, nodeId)
PRESETS         // { key: { label, group, note, dataset: DATASETS key | null, lr: number | null, noise: number | null,
                //   build(seed = 1) -> net } };
                //   build() also records the dataset in meta.train.dataset, and lr and noise (when set) in meta.train.
                //   Listed in menu order: the shell's New net picker has
                //   one <optgroup> per group (in first-seen order), shows note as the option's tooltip and toasts
                //   it for 5 s when the preset loads from the picker (not on a #nn= preload). note is one line on
                //   what to notice, matrix panel first. Presets with hand-set weights ignore the seed. meta.title is unique per preset.
                //   Basics: gates, xor_gates, xor_relu (hand-set), xor, perceptron, logreg, linreg, softmax_reg
                //     (the last three start at W = 0), linear (all identity: demo of collapse)
                //   MLPs: mlp, deep, classifier (2-4-3 softmax), wide, narrow_deep (49 parameters each), funnel,
                //     uat (hand-built ReLU hinges on the sine)
                //   Skip connections: residual, bottleneck, ffn, densenet, unet, wide_deep
                //   Structure in W: conv1d, conv1d_s2 (tied kernel k:1,o and bias b:1), avgpool (fixed edges),
                //     maxpool (b + ReLU(a - b)), lenet (tied conv, fixed pool), gnn (hand-set),
                //     towers (block-diagonal), multitask (shared trunk, block-diagonal heads)
                //   Sequences: rnn (unrolled; tied W_{hh}, w_x, b_h), wavenet (dilated causal conv, a tied kernel k^{(l)} per layer)
                //   Attention (token nets, mse, see docs/NN_ATTENTION.md): words (hand-set "the cat sat": one-hot
                //     det / noun / verb -> tied QKV -> Z, no dataset, sets meta.tokenNames, see docs/NN_LENS.md),
                //     pronouns and agreement (3 words x 2, their WORDS vectors -> tied QKV -> Z = the output; nl_pronoun
                //     and nl_agree, lr 0.3, noise 0; W_V starts as I and b_V at 0, W_Q and W_K small, recorded as
                //     meta.train.init; the first sentence names the tokens),
                //     attention (3 tokens x 2 -> tied QKV -> Z, seq_max, lr 0.3), causal (seq_prev, lr 0.3), causal_rot
                //     (hand-set solution of causal's task: W_Q turns the positions back 120°; seq_prev), multihead
                //     (2 heads, seq_minmax, lr 1), transformer (2 tokens: QKV, Z, H = X + Z W_O, ReLU FFN d -> 2d,
                //     Y = H + FFN; seq_addmax, lr 0.3), tiny_lm (the tiny language model: 5 one-hot words -> X = O W_E + P
                //     -> Q, K, V -> 2 causal heads -> H = X + Z W_O -> ReLU FFN d -> 4d (8 -> 32) -> Y = H + F W_2 -> softmax
                //     over 23 words per position, xent; nl_lm, lr 0.2, noise 0, 300 points; its inputs are "the cat sat on
                //     the"; sets meta.vocab and meta.flow; 670 nodes, over the menu's 40 like its variants, docs/NN_FLOW.md).
                //   Tiny LM variants: tiny_lm_nope, _sin, _rope, _alibi, _mqa, _gqa, _window, _linear, _nomask, _prenorm,
                //     _postnorm, _rmsnorm, _gelu, _swiglu, _tied, _2layer, _window2: tiny_lm with one change each, on nl_lm
                //     (up to 1270 nodes). They and tiny_lm have family: 'tiny_lm', axis, short and title (the meta.title
                //     their nets carry), which the Flow view's variant picker reads (docs/NN_FLOW.md, Variants)
                //     The 3-token presets
                //     lay X and Q, K, V out as tokens x d grids (a row per token, a column per feature, groups stacked,
                //     X between the Q and K blocks), so they fit a 1600 x 900 window at 0.63 to 0.73 zoom; later layers are columns
                //   Embeddings & autoencoders: autoencoder, embedding (one-hot lookup), pca_ae (linear, on cloud)
                //   Teaching demos: vanishing (sigmoid chain), gan (D(G(z)) as one net)
layerIndex(net, layerId), nodeLayerIndex(net, nodeId), nodesIn(net, layerIdOrIndex) -> nodes[]
node(net, id), edge(net, id), edgeBetween(net, a, b)   // edgeBetween matches either direction

// edits: mutate net in place and keep validate(net) empty (removing a node removes its edges, a layer its nodes)
//   Token nets: edits never refuse on token grounds. A layer whose node count no longer splits into its tokens x
//   groups loses tokens / groups, and an attention layer whose Q, K, V input breaks (a node added to or removed
//   from Q, K, V, a layer inserted between them, Q, K, V removed) becomes a plain dense layer. Only wiring INTO
//   an attention layer refuses (connect -> null, connectDense -> []).
addLayer(net, at, { name, act, size = 2, dense = false, seed }) -> id   // nodes in a column between the neighbours;
                                                                        //   dense: also wire it to both neighbours
removeLayer(net, id, { bridge, seed }) -> bool  // refuses (false) at 2 layers. bridge: wire the two
                                                //   neighbours densely if nothing joins them (the shell's Delete uses it)
setLayer(net, id, { name, act, causal, heads, scale, window, pos, linear }) -> bool   // an attention layer's act stays
                                                //   identity; heads only if it divides d; scale: a number, or null for
                                                //   1/sqrt(d_k / heads); window, pos: null removes them, linear: false
addNode(net, layer, { x, y, label, bias, value, target, params, index, connect, seed }) -> id | null
                // index = order within the layer; connect: wire it to every node of both neighbour layers
                //   (seeded Xavier). A new output gets target 0 when every other output has a target
removeNode(net, id), moveNode(net, id, toIndex)
setNode(net, id, patch) -> bool                 // x y label bias value target params layer. Numeric strings are
                                                //   accepted, bad values ignored; a new layer drops sideways/backward edges.
                                                //   bias sets the node's whole bias tie group; ignored on an attention layer
connect(net, from, to, w?) -> id | null         // reuses an existing edge, swaps a backward pair, null within a layer
                                                //   or into an attention layer; no w: a seeded random weight in (-1, 1).
                                                //   w on an existing tied edge sets its group; a fixed edge keeps its w
disconnect(net, edgeId), setWeight(net, edgeId, w) -> bool   // setWeight: a tied edge sets its group; false on a fixed edge
connectDense(net, fromLayer, toLayer, { seed, scheme = 'xavier', w }) -> edgeId[]   // existing edges keep their w
randomize(net, { seed, scheme: 'xavier' | 'he' | 'small', biases: 'zero' | 'small' | 'keep', init })   // biases default 'zero'.
                                                //   Fixed edges and fixed biases kept, one draw per tie group, attention biases stay 0.
                                                //   init: { <name>: scheme } for the edges tied as '<name>:<i>,<j>', a scheme of
                                                //   INIT_SCHEMES = ['xavier', 'he', 'small', 'identity', 'zero'] ('identity': 1 where
                                                //   i = j, else 0); other names and unknown schemes use scheme
autoLayout(net, { width, height })              // nodes in evenly spaced columns, centred; token layers leave an
                                                //   extra quarter row between tokens and another between groups

// tokens, ties, attention (docs/NN_ATTENTION.md)
tokenShape(net, layer) -> { tokens, d, groups: string[] | null }   // a plain layer is 1 token of d = its size
tokenPos(net, nodeId) -> { l, index, g, group, token, feature } | null   // 0-based; group null without groups
reshape(net, layer, vec) -> { [group | 'X']: number[][] }        // tokens x d matrices of any layer vector
                                                                 //   (a, z, b, dA...); missing entries read 0
attnSpec(net, layer) -> { l, tokens, d, heads, dh, scale, causal } | null   // a working attention layer, defaults
                                                                 //   filled (d = d_k = d_v, dh = d / heads); window, pos, linear too when set
attnVisible(spec, i, j) -> bool                                  // query i may read key j: the causal mask and the window
attend(x, spec) -> { heads, tokens, dk, scale, causal, ..., out } // one attention layer from its Q, K, V activations x,
                                                                 //   as forward computes it (fwd.attn[l]'s report, and out)
ropeFreq(p, dh), ROPE_BASE = 10000, alibiSlope(h), NORM_EPS     // the variants' constants (docs/NN_FLOW.md, Variants)
tiedMatrices(net, l) -> [{ name, W, ties, k, fromGroup, toGroup, edges, tokenwise }]   // [] unless every non-fixed
                // edge into l is tied as '<name>:<i>,<j>'. W is in the X W convention (rows = input feature): Q = X W_Q,
                // an edge from feature i to feature j has weight W[i-1][j-1], the per-token receiving-row matrix is Wᵀ.
                // ties[i][j] = tie id | null; k = source layer (null if mixed); tokenwise: the layer matrix is exactly
                // I_tokens ⊗ Wᵀ. Fixed (residual) edges are left out: find them by edge.fixed. A shared matrix whose
                // every edge here reads entry (i, j) the other way round (tied embeddings: the logits' W_E) comes back
                // in this layer's X W convention with transposed: true (and tokenwise)

// maths
matrices(net) -> [ per layer l >= 1: { l, id, kind: 'dense', act, rows: nodeId[], b: number[],
                   terms: [{ k, cols: nodeId[], W: number[][], edge: (edgeId|null)[][] }] } ]
                 // matrices()[l - 1].l === l. The k = l-1 term always exists; other k only when a skip edge exists. Sorted k desc.
                 // Tied entries simply repeat values. An attention layer: { l, id, kind: 'attention', act: 'identity',
                 //   rows, b: all 0, terms: [] } (its z is not b + Σ W a: read fwd.attn)
forward(net, x?) -> { z: (number[]|null)[], a: number[][], node: { [id]: { z, a } }, attn }   // x defaults to layer-0 values
                 // attn[l] (null except on attention layers) = { heads: [{ Q, K, V, Z (tokens x dh), S, A (tokens x tokens) }],
                 //   tokens, dk: dh, scale, causal }. S is the scaled score QKᵀ·scale with masked cells -Infinity; A = softmax(S).
                 //   The variants add window / pos / linear, and per head Qr, Kr (RoPE), Qf, Kf (linear), B (ALiBi):
                 //   docs/NN_FLOW.md, Variants
backward(net, fwd, y, loss) -> { loss, note, dA: number[][], dZ: number[][], dW, db: number[][],
                                 node: { [id]: { da, dz } }, edge: { [id]: dw }, attn, tie }
                 // attn[l] = { heads: [{ dZ, dQ, dK, dV (tokens x dh), dA, dS (tokens x tokens) }] }: dV = Aᵀ dZ, dA = dZ Vᵀ,
                 //   dS = A ⊙ (dA - rowsum(dA ⊙ A)), dQ = dS K · scale, dK = dSᵀ Q · scale (this layer's own share of
                 //   the Q, K, V layer's dA). tie[tieId] = a shared weight's or bias's gradient, the sum of its members'
                 //   (edge[id] and node[id].dz stay per member). db is all 0 on an attention layer; dW[l] is []
trainStep(net, { X, Y }, { lr = 0.1, loss }) -> mean loss   // one step over the given batch (gradients averaged);
                                                            //   the loss is from before the update. Non-finite: no update.
                                                            //   A tie group steps by its summed gradient (members stay
                                                            //   equal); fixed edges, fixed biases and attention biases never move
predict(net, X, { layer }) -> number[][]       // outputs; layer = index | id (that layer's a) | 'all' (every layer per sample)
collapse(net) -> { W, b, rows, cols } | null   // the affine map y = W x + b when every non-input layer is identity
                                               //   (null with an attention layer)
DATASETS        // { key: { label, inputs, outputs, kind: 'class' | 'reg' | 'seq', tokens?, vocab?, decode?,
                //   make(n = 200, seed = 1, noise = 0) -> { X, Y, words?, targetWords? } } }
                //   xor, circles, spiral, blobs, moons (2 -> 1), three (2 -> 3 one-hot), line, sine (1 -> 1),
                //   cloud (3 -> 3 regression, target = input: a flat 3-D cloud, for pca_ae).
                //   noise is Gaussian, on the inputs (class) or the targets (reg).
                //   Sequence tasks (kind 'seq', regression, mse): X and Y rows are token-major, inputs = tokens · d_in,
                //   outputs = tokens · d_out; noise goes on the token features and the targets follow the noisy tokens.
                //     seq_max (3 tokens x 2 -> 3 x 2): every position outputs the token with the largest x₁
                //     seq_minmax (3 x 2 -> 3 x 2): every position outputs (max x₁, min x₁)
                //     seq_prev (3 x 3 -> 3 x 1): tokens (c, cos θ, sin θ), θ = 0, 120°, 240°; y_i = c_{i-1} (y_1 = c_1)
                //     seq_addmax (2 x 2 -> 2 x 2): y_i = ReLU(x_i + the token with the largest x₁)
                //   In seq_max / seq_minmax / seq_addmax the x₁ are uniform on [-0.9, 0.9], at least 0.3 apart
                //   Word tasks (kind 'seq', 3 tokens x 2 -> 3 x 2, mse): each token is a word's WORDS vector, and each
                //   target the vector of a word in the same sentence (a noisy copy when noise > 0). They add
                //     vocab: { word: [x, y] }, the WORDS entries the task uses;
                //     decode(y) -> (string | null)[]: each token of an output row named by its nearest vocab word
                //       (null where a coordinate is not a finite number);
                //     make() also returns words (per sample, the 3 words) and targetWords (the word each token
                //       should output). The noise has its own random stream: a seed gives the same sentences at any noise.
                //     nl_pronoun: "dog sees itself": noun, verb, reflexive, all agreeing in number (8 sentences).
                //       Targets: the noun, the verb, the noun (the reflexive outputs the noun it refers to)
                //     nl_agree: "dog chases cats": subject, verb agreeing with it, an object of the other number
                //       (16 sentences). Targets: the subject, the subject (the verb outputs its subject), the object
                //   Next-word tasks (kind 'seq', xent, docs/NN_FLOW.md): one-hot words in, the next word at each
                //     position out. vocab: { word: slot }; decode names each token by its most likely word; make()
                //     returns words and targetWords; noise on the inputs only.
                //     nl_lm (5 tokens x 23 -> 5 x 23): "the SUBJECT VERB PREPOSITION the OBJECT" from a small grammar
                //       (20 sentences, "the cat sat on the mat"); the object is fixed by the subject and the
                //       preposition together. Inputs the first 5 words, targets the last 5 (the tiny language model's)
                //     nl_next (3 tokens x 7 -> 3 x 7): one-hot over '.', dog, cat, dogs, cats, chases, chase. Inputs
                //       '. subject verb', targets 'subject verb object' (the verb agrees, the object is the other
                //       animal in the other number: 4 sentences). Kept for nets saved with the first tiny language model
WORDS           // frozen { word: [x, y] }: dog (1.2, 0.6), cat (0.6, 0.6), itself (0, 0.6), sees (-0.6, 0.6),
                //   chases (-1.2, 0.6), and the plurals dogs, cats, themselves, see, chase at y = -0.6.
                //   x is the kind of word (nouns right, verbs left), y the number
rng(seed) -> () => [0, 1)                      // mulberry32; no seed = a random one
fmt(x, digits = 2) -> string                   // ASCII minus, never -0.00, 'NaN' / 'inf' / '-inf'. KaTeX-safe
fmtg(x, digits = 2) -> string                  // for gradients: fmt from |x| >= 0.01; below, two significant figures:
                                               //   '0.0034' down to 0.001, then '3.4e-4'; |x| < 1e-9 (rounding noise) reads fmt(0).
                                               //   In KaTeX write the exponent as \mathrm{e}{-4}
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
  proper distribution). On a token layer (softmax per token): the mean over its tokens (and groups)
  of each one's `-Σ y log p`, so the gradient is divided by that count.
- `xent` with sigmoid outputs: binary cross-entropy, mean over outputs.
- `xent` on any other output layer: fall back to mse and say so in `backward().note`.

## `store.js`

```js
import { createStore, colorFor, POS, NEG, HI } from './store.js';
store.net       // the live net. Read it fresh every time; its identity never changes (undo/load mutate in place)
store.state     // { sel, hover, anim, fwd, bwd, lens, viz, tour }. fwd / bwd are null when the recompute threw; bwd also
                //   without targets. lens / viz / tour start null: docs/NN_LENS.md
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
  - `{ kind: 'token', layer, t, g?, h? }`: token `t` (0-based) of a token layer, `layer` an index
    (the view, the matrix panel and the inspector also accept an id). `g` limits it to one group
    (0 = Q, 1 = K, 2 = V on a Q, K, V layer); `h` to one head, on an attention layer or on the
    Q, K, V layer before it. Without them it means the whole token. The view lights the token's
    box, neurons and edges; on an attention layer (or a query token) it also shows the token's
    row of A, and for a key or value token its column. The matrix panel lights the token's cells
    and that row or column of S and A; the inspector lights the matching row, column or bar of
    an attention layer's or neuron's card.
  The view emits node, edge, layer and token (a token box, and a heatmap cell or attention edge
  as its row of A); the inspector and the matrix panel emit every kind (the matrix panel's `t_i`
  headers and S / A cells, and the inspector's A tables, emit token); the attention panel emits
  token (docs/NN_LENS.md).
- `anim`: see below.
- `train`: `{ epoch, loss, running }`, emitted by `train.js` on play, pause, step, reset and every
  frame while training. `loss` is the full-dataset loss, or null (while playing, the last one the
  panel worked out: see Performance). The inspector listens to `running` (it brings every card up
  to date on pause).
- `lens`, `viz`, `tour`: see docs/NN_LENS.md.

**`state.anim`** (step-through, owned by `matrix.js`)
- `null | { dir: 'fwd' | 'bwd', l, i, phase }`. `l` is a layer index >= 1 (the view also accepts
  a layer id) and `i` the 0-based row in that layer.
- `phase` is one of `'dot' | 'scores' | 'softmax' | 'sum' | 'delta'`. On a dense or token layer
  every forward step is `'dot'` (row i of W times the input, plus b_i, then the activation) and
  every backward step `'delta'` (δ_i, row i of ∂L/∂W, ∂L/∂b_i). A working attention layer steps
  a token at a time, and `i` is that token's first neuron: forward in three phases, `'scores'`
  (q_i · k_j for every j), `'softmax'` (row i of A) and `'sum'` (z_i = Σ_j A_ij v_j), then one
  `'delta'` step per token backward (∂z_i, ∂A, ∂S, ∂q_i, ∂k_i, ∂v_i). matrix.js sets no other values.
- Order: every row (or attention token and phase) of layers 1..L forward, then, only when
  `state.bwd` exists, every row (or attention token) of layers L..1 backward. From null, S starts
  at the first forward step (the first backward step when Backward is on) and Shift+S at the last
  step. Stepping past either end, or ■, sets null.
- A step switches Batch off; a bwd step switches Backward on. matrix.js resets `anim` to null
  when it no longer fits the net (row gone, or `bwd` without `state.bwd`), when Batch is turned
  on, and when Backward is turned off during a bwd step.
- view.js lights neuron i of layer l (class `lit`). For fwd it also lights its incoming edges and
  their sources, for bwd its outgoing edges and their targets; everything else dims, and a pulse
  runs along the lit edges (source to target for fwd, reversed for bwd). `phase` only restarts
  the pulse when it changes. On an attention layer it lights the token's Z neurons: `'scores'`
  its query and the keys it may see, `'softmax'` its row of attention edges with their A_ij,
  `'sum'` the same edges with pulses from the values; backward, all of them plus the token's
  outgoing edges.
- matrix.js highlights row i of W, b_i and (z_i, a_i) for fwd, or the backward row for bwd, plus
  the input vector (on an attention layer: the token's row of Q, S, A or Z, by phase). It shows
  the step's arithmetic in a box under layer l, and its toolbar reads `forward · layer l · row i/n`
  (`forward · layer l · token i/n · scores`, and so on, on an attention layer). When the toolbar
  has no room for that readout, or is hidden (H, the audience window), it is the step box's title.

**Performance**
- `net` / `values` can fire every animation frame while training.
- On non-structural changes, update attributes and text in place. Rebuild DOM only on
  `structural` (matrix.js also rebuilds when its toggles, labels or loss case change).
- Work over the whole dataset gets a share of the time while training, measured as it runs (the
  median of the last few costs), so a big net's frames go to training and a small net's are as
  before: the Train panel's full redraw (the dataset's loss, the readout, the chart, the plot) waits
  until it would take at most 20% of the time; one that costs under a frame's share (about 4 ms) is
  still done every frame. Pause, a step and any other change redraw at once, so what shows after
  them is exact. The audience window spaces the presenter's nets the same way and always draws the
  last one. surf3d.js spaces its trail's points (docs/NN_3D_PLOTS.md) and nn.js the mirror posts
  (Audience mirror) by the same rule.
- What nobody can see waits. The matrix panel, hidden (`#nn-main.matrix-hidden`: the divider, or
  the Flow view's `matrixAway`), skips its scheduled renders (net, values, toggles) but still
  builds after a structural change and answers `lens` and `anim` at once. The canvas, covered
  (the Flow or 3D view sets the SVG's `visibility: hidden`), puts off its value paint and meta
  check; hover, lens, layout, structure and fits go on. Both catch up in a microtask when shown
  again, before the next frame draws, with their transitions off for two frames (`.nm-snap`,
  `.nnv-snap` zero `--dur-1` and `--dur-2`), or in the background once 300 ms pass without
  training. The canvas's `contentBox()` (fit, outOfView, contentRect, png) flushes in full.

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
- Matrix panel cells take `colorFor` with its alpha eased above 0.5 to at most 0.72, so their
  digits are always the text colour; a value that reads as zero has no fill, and a fixed edge's
  cell none either (fixed edges are neutral, as on the canvas). Attention weights A_ij are filled
  with `--att` at alpha `0.07 + 0.93·A` (eased the same way), as the canvas heatmap.
- train.js takes the POS / NEG RGB from `colorFor(±1, 1, theme)` and blends onto `--bg`. Output
  field: one output is coloured by the sign of `(out - mid) / half` (mid and half from the
  dataset's target range), alpha `0.05 + 0.62·|v|`; K outputs by the argmax class (class 0 NEG,
  1 POS, then 5 extra colours), alpha by the margin over the runner-up. Per-neuron maps: sigmoid
  and softmax on [0, 1], tanh on [-1, 1], anything else on ± that neuron's max |a| over the
  grid, alpha as `colorFor`. `HI` rings the current sample.

**Numbers**
- Values show `model.fmt(x, 2)` (the inspector's gradient lines 3 decimals), with a Unicode minus
  in HTML. Gradients (δ, ∂L/∂a, ∂L/∂W, ∂L/∂b, attention's ∂ matrices and the step-through's
  backward numbers) use `model.fmtg`, the same way in the matrix panel, the cards and the view's
  δ and ∂L/∂w labels, so a small gradient reads 0.0034 or 3.4e−4 rather than 0.00.

## DOM and module interface (shell)

```html
<section id="nn" hidden>
  <header id="nn-bar"> shell toolbar ... </header>
  <div id="nn-main">
    <div id="nn-stage"></div>   <!-- view mounts the SVG; the inspector's cards, the Train and attention
                                     panels, the lens bar and the Explain card float over it -->
    <div id="nn-split"></div>   <!-- drag to resize, double-click to hide / show the matrix panel -->
    <aside id="nn-matrix"></aside>
  </div>
</section>
```

`nn.js` imports `model.js` and `store.js` first (if either fails, the stage shows an error and
`body.dataset.nnReady = 'error'`). It then imports `view`, `inspector`, `matrix`, `train`, `lens`,
`attnviz` and `tour` in parallel and installs them in that order, awaiting each
`export function install(ctx)`. A module that fails to import or install is logged and skipped.
`focus.js` is not installed: the modules import it (docs/NN_LENS.md).

```js
ctx = {
  store, model, el: { root, bar, stage, matrix },
  addButton({ label, title, onClick, group = 'modules', icon }) -> <button>,   // into #nn-bar (see Toolbar below);
                          //   label is HTML; icon (before the label) is a static/icons.js name, an <svg>
                          //   string or glyph HTML (a known label, e.g. Train, gets its line icon instead of a
                          //   glyph); no label makes an icon-only button, its title also its aria-label
  toast(msg, ms?), theme() -> 'dark' | 'light', onTheme(fn(theme)) -> off, onShow(fn(visible)) -> off,
  active(e?) -> bool,     // Net tab visible, not the audience window, and e (if given) is not typing into a text
                          //   input, textarea, select or contenteditable (range / checkbox / radio / button inputs don't count)
  audience: bool,         // read-only mirror window: render only, never edit or persist
  matrixAway(on),         // hide the matrix panel for a while (the Flow view shows the matrices itself); false
                          //   brings back the user's state. Not saved; the divider (drag, double-click) ends it
  graph,                  // getter: window.mathboardGraph or null (3D tab API: rows, rowByName, addRow, setRowSource,
                          //   removeRow, setRows, setView, toast, audience: { open(), isOpen } from lecture.js, ...)
  view,                   // set by view.js during its install (see below)
  inspector,              // set by inspector.js
  matrix,                 // set by matrix.js (test / debug handle)
  train,                  // set by train.js: its test handle (below)
  attnviz, tour,          // set by attnviz.js and tour.js: docs/NN_LENS.md
  view3d, flow,           // set by view3d.js (docs/NN_3D.md) and flow.js (docs/NN_FLOW.md)
}
ctx.view = {
  svg, worldToScreen(x, y) -> { x, y },    // world -> px relative to #nn-stage
  screenToWorld(x, y) -> { x, y },         // px relative to the SVG (it fills the stage) -> world
  nodeRect(id) -> { x, y, w, h } | null,   // stage px, for placing the inspector
  fit(ms?),                                // frames the net in the free part of the stage: above the lens bar,
                                           //   clear of the Train and attention panels (docs/NN_LENS.md, View fit).
                                           //   The view refits by itself when a panel opens, folds, resizes or
                                           //   closes, unless the user has panned or zoomed since the last fit
  outOfView() -> bool,                     // part of the net is outside that free area
  contentRect() -> { x, y, w, h },         // stage px box of the whole net (lanes and headers included)
  setNodeImage(id, url | null),            // train's per-neuron maps, clipped inside the node; kept across rebuilds
  png(scale = 2) -> Promise<dataURL>,      // the net on the theme background, without hover, selection or handles
  weights,                                 // get / set: the W labels toggle
}
ctx.inspector = {
  open(target, { pin = true }) -> card element | null,   // a target that has a card reuses it; pin: false only selects
  closeAll(), cards -> [{ target, pinned, el }],
}
ctx.matrix = { step(±1), toggle(key), opt, render(), update(), reveal(layer, part?, opts?), parts(layer) }
               // not a contract except reveal / parts (for tour.js, docs/NN_LENS.md): for tests and the console.
               // opt = { mode: 'fwd' | 'bwd', bias, batch, collapse, labels, expand }; render() forces a rebuild.
               //   expand: a comma-separated string of the token layer ids whose "Flattened: z = W a" form is
               //   open (a string, so the audience mirror compares it by value)
```

- Toolbar: three rounded clusters in the board toolbar's style, level with the Board / 3D / Net
  tabs, each made of divider-separated sections (`BAR` in nn.js). **build**: `new` (New net),
  `net` (+ Layer, Layout, Fit, Randomize), `edit` (Undo, Redo: icons). **show**: `view` (Weights, lens.js's Lens,
  view3d's 3D, flow's Flow), `panels` (the groups `train`, `attnviz` and `surf3d`: Train, Attention, 3D plots),
  `tour` (Explain). **tail**, at the right end: `file` (the File menu), `tail` (**Audience** and `?`,
  the cheat sheet: the keys below and the toolbar buttons). A group joins the section of the same
  name (or `panels`, as above) whatever the install order; any other group gets a section of its
  own at the end of show, in the order first used. Buttons carry static/icons.js line icons (`?`,
  Undo and Redo are icon-only). Short of room the bar first tightens its spacing (`.nn-snug`, icons
  kept: 1600 px wide fits), then drops the icons (`.nn-compact`; icon-only buttons and + Layer's
  plus stay), then some padding (`.nn-tight`: 1280 px fits), and only then wraps (below about
  1260 px).
  Audience calls `window.mathboardGraph.audience.open()` (set by `graph/features/lecture.js`), the
  same window the 3D tab's Audience button opens; it is lit while that window is open
  (`audience.isOpen`, checked every second while the tab is shown). Toolbar buttons never take
  focus, so Space stays with training.
- Menus: **New net** (also **N**) and **File** open a popover under their button, inside `#nn-bar`
  (so H and the audience window hide it too); one at a time, closed by Esc, a click outside or on
  the button, or leaving the tab. New net lists `PRESETS` in sections by `p.group` (in `PRESETS`
  order), in columns, then **Blank net**; a search field filters on label, key, group and note
  (every word must match), and the line at the bottom shows the active preset's note (also its
  tooltip). Typing filters, ↑ ↓ (or Tab) move, ← → change column while the field is empty, Enter
  opens the preset (Ctrl+Z goes back), and while it is open no other shortcut sees a key. File holds
  Export, Import, PNG and To board (↑ ↓ Enter; any other key closes it and goes on as usual).
- `ctx.toast` in this tab is centred over the stage, not the window, and lifted above the lens bar
  and the 3D view's bar when they are under it (nn.js sets `--nn-toast-*` on body; the rule is
  `body[data-view="nn"] #toast` in nn.css).
- train.js sets `ctx.train` (its test handle, below). Its named exports are
  `readSettings(net, model)`, `netShape`, `defaultDataset`, `forwardMany(net, model, X, n, from, M, outOnly)`,
  `datasetLoss(P, Y, n, K, loss, outAct, segments = 1)`, `wordAccuracy(P, targetWords, n, K, ds)` and
  `adaptNet(net, model, ds, { seed })`; matrix.js and inspector.js import `readSettings`.
  `forwardMany`'s `outOnly` (default false) is for a caller that reads only the output layer (a
  loss): a net with attention then fills only that layer (and layer 0), with the same numbers.
  `wordAccuracy` is the share of output tokens whose nearest word (`ds.decode`) is the target word;
  the readout shows it as **words** (classification shows **acc**). `datasetLoss`'s `segments` is the number of
  softmax blocks of a token output layer (tokens × groups): its cross-entropy is their mean, as in
  `backward`. `adaptNet` returns a sentence saying what it did, or throws an `Error` whose message
  says why it can't (a token net on a plain dataset, another token count, an attention output of
  the wrong width, shared or fixed weights laid out for other inputs), leaving the net untouched.
  Test handle: `ctx.train` = `document.querySelector('.nn-train').nnTrain` =
  `{ play, pause, step, reset, adapt, loadSample(i), stepSample(±1), fold(on?), running, eval, open,
  folded, point(i) }`. `stepSample` loads the next or previous dataset sample (from the one the
  inputs hold, else the first or last), as the ◀ ▶ buttons under a sequence plot do. `loadSample`
  on a word dataset also sets `meta.tokenNames` to the sample's words, in the same commit; a sample
  of a dataset without words drops names that are all `WORDS` (left by a word dataset). `fold(on)`
  folds the panel to its header or unfolds it (no argument toggles), saved like the button;
  Explain uses it (docs/NN_LENS.md).
- The shell's **Delete** and **+ Layer** keep token layers whole, by view.js's rules (its
  double-click and wiring code), and toast why instead of editing: Delete refuses one neuron of a
  token, Q, K, V or attention layer, a fixed edge, and the Q, K, V layer an attention layer reads;
  + Layer refuses to go right before an attention layer or next to a token layer.

**Keys in the Net tab** (only when `ctx.active(e)`; never while typing)

| owner | keys |
|---|---|
| shell | Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) undo/redo, Delete/Backspace remove selection, Esc close the cheat sheet or deselect, F fit, N the New net menu, H hide UI, ? cheat sheet. An open toolbar menu takes the keys first (capture phase) |
| view | W: weight labels on edges |
| matrix | S: step, Shift+S: step back, B: bias-trick toggle |
| train | Space: play/pause training, T: single training step |
| lens | 1–9 / 0: follow token n / every token, `[` / `]`: previous / next stage, L: lens bar |
| attnviz | A: attention panel, M / Shift+M: next / previous mode |
| tour | E: Explain; while it runs → / PageDown, ← / PageUp, Esc (caught first) |
| view3d | D: 3D view, Shift+D: next view, ← / → in the tensor view (Explain catches them first) |
| surf3d | P: 3D plots panel, Shift+P: next plot |
| flow | G: Flow view, Shift+G: play its stages, ← / → its stages while it is on (Explain catches them first) |

Details for the last three: docs/NN_LENS.md.

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
chrome: cards lose the rename, pin, close and param-delete buttons, "+ add", the Remove edge
foot (`.ui-chrome`) and the drag cursor (they still work); the Train panel keeps its readout, loss chart, warning text and plots and hides its
settings and buttons (as `.nn-train.ro`). These rules live in `nn.css`, under `body.clean #nn`.
The lens bar hides; the attention panel and the Explain card stay without their chrome
(docs/NN_LENS.md; rules in `lens.css`, `attnviz.css`, `tour.css`).
`#nn-main` sits in grid row 2, so the stage keeps the full height when the toolbar is hidden.

**Audience mirror**

nn.js publishes `window.mathboardNet = { store, ctx, audience, ready, mirrorState(), applyMirror(m),
onMirror(fn) -> off }`. `ready` turns true, and `window` gets the `mathboard:nn-ready` event, once
every module has installed.
- `mirrorState()` -> `{ net, sel, hover, anim, split, matrixHidden, lens, viz, tour, v3d, s3d, flow, matrix, weights }`
  (`matrixHidden` as shown, `matrixAway` included), where
  `matrix` is a copy of `ctx.matrix.opt` (or null, `expand` included) and `weights` the W labels
  toggle. `hover` carries token hovers too, so the audience sees the token and attention row the
  presenter points at.
- `onMirror(fn)` fires on the store's `net`, `layout`, `sel`, `hover`, `anim`, `lens`, `viz`, `tour`, `v3d`, `s3d` and `flow` events, when
  the split changes, and one frame after any click in the toolbar or the matrix panel or a key
  up in the tab (the matrix toggles and W have no store event). While training, the `net` events'
  calls are spaced so the posts they cause take at most 15% of the time (mirrorState() times each
  post; a small net's still go every frame); the pause's commit and every other event call at once.
- `applyMirror(m)` does nothing outside an audience window. It loads `m.net` with
  `{ history: false }` when its JSON changed, copies `m.matrix` into `ctx.matrix.opt` and
  re-renders the panel when a toggle differs, sets W, sets `sel` / `hover` / `anim` / `lens` /
  `viz` / `tour` / `v3d` / `s3d` / `flow` when they differ, and applies the split.
- `graph/features/lecture.js` carries it over its BroadcastChannel. The presenter posts
  `{ nn: mirrorState() }` at most once per frame, leaving out `net` when it hasn't changed, and
  also sends a full snapshot (including `view` and the net) every second. The audience switches
  to the Net tab when the presenter's `view` is `nn` (anything else shows the 3D view) and passes
  `nn` to `applyMirror`, buffering it until the Net tab is ready.
- Only that state is mirrored. Each window keeps its own pan / zoom (the audience view refits
  when the net leaves the frame) and pinned cards. Panel layout goes through localStorage
  instead (both windows share an origin): the audience's read-only Train panel follows the
  presenter's `open` and `fold` from `mathboard.nn.train` (its spot stays the default), and its
  attention panel the presenter's spot, width and mode from `mathboard.nn.attnviz`, both through
  `storage` events. Its cards fold and unfold their sections with the presenter's
  (`mathboard.nn.inspector`), through the same events.

**Persistence and test preload**
- localStorage `mathboard.nn` = `{ v: 1, net, split, matrixHidden }`, owned by the shell, written
  500 ms after a `net` or `layout` event and on `pagehide`. `mathboard.nn.train` belongs to
  train.js (see `meta.train`), `mathboard.nn.lens` to lens.js and `mathboard.nn.attnviz` to
  attnviz.js (docs/NN_LENS.md). `mathboard.nn.inspector` = `{ folds: { <section key>: folded } }`
  belongs to inspector.js: the sections the user folded or opened, per kind (`node.grad`,
  `layer.act`, ...). A key that is not there takes its default (folded).
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
