# Net tab: the Flow view and the tiny language model (contract)

This extends docs/NN_CONTRACT.md, NN_ATTENTION.md and NN_LENS.md. One module, `static/nn/flow.js`
(with `flow.css`), draws one whole forward pass of the current net left to right, as live matrix
tiles with their shapes, down to the softmax over the next word. Like the 3D view, it covers
`#nn-stage` and replaces the SVG canvas while it is on. The matrix panel steps aside while it is on
(the flow shows the matrices itself: `ctx.matrixAway`, not saved, and dragging the divider brings it
back); the cards, the Train and Attention panels, the lens bar and Explain keep working over it and
stay linked through the store.
It adds one store key, `flow`, which the audience window mirrors, and one preset made for it:
**Tiny language model** (`tiny_lm`) on the next-word dataset `nl_next`.

Toggle it with the toolbar's **Flow** button (group `view`, beside 3D) or **G**. Shift+G plays the
stages. The Flow view and the 3D view replace the canvas one at a time: opening one closes the other.

## The tiny language model (`tiny_lm`, model.js)

A decoder-only transformer small enough to read, trainable end to end:

| layer | shape | computes |
|---|---|---|
| 0 Words (one-hot) | 3 × 7 | position t holds its word as a one-hot row over the 7-word vocabulary |
| 1 X = E[w] + P | 3 × 4 | `X = O W_E + P`: the tied embedding W_E (7 × 4); P, a learned vector per position, is this layer's own untied biases |
| 2 Q, K, V | 3 × 4 each | `Q = X W_Q + b_Q` and so on, tied 4 × 4 matrices, biases shared per feature |
| 3 Attention Z | 3 × 4 | 2 causal heads of d_h = 2, scale 1/√2; `Z = [Z_1 Z_2]` |
| 4 H = X + Z W_O | 3 × 4 | tied W_O (4 × 4), + X on fixed residual edges, + b_O |
| 5 FFN | 3 × 16 | `F = ReLU(H W_1 + b_1)`, d → 4d |
| 6 Y = H + FFN | 3 × 4 | `Y = H + F W_2 + b_2`, + H on fixed residual edges |
| 7 Next word | 3 × 7 | `p = softmax(Y W_U + b_U)` at each position (softmax per token), cross-entropy loss |

174 neurons and 768 edges, so it is the one preset exempt from the menu's 40-neuron limit (the
test says so): the canvas draws it as a wide strip of token grids, and the Flow view is where it
reads. `meta.flow = true` opens the Flow view when the preset loads (and on a `#nn=tiny_lm`
preload). `meta.vocab` names the 7 one-hot slots: `['.', 'dog', 'cat', 'dogs', 'cats', 'chases',
'chase']`. `normalize` keeps both, as it keeps any unknown meta field.

**Initialisation.** The weights are drawn by `randomize` with the Train panel's Reset recipe (He,
as the net has a ReLU, biases 0, so P starts at 0), except W_Q, W_K, W_O and W_2, which start small
(`meta.train.init = { W_Q: 'small', W_K: 'small', W_O: 'small', W_2: 'small' }`): every position
first reads its past about evenly, and the block starts near Y = X. Reset with init seed 1 gives
back the preset's exact weights. `PRESETS.tiny_lm.lr` is 0.1 and `noise` 0.

**No LayerNorm.** model.js has no per-token normalization, and adding one cleanly means a new
vector activation whose Jacobian the matrix panel and the cards would also have to show (they
special-case softmax as the only vector activation). With one block, d = 4 and the small init
above, training is stable without it: the loss reaches its floor in about 500 steps of batch 10.
Pre-norm would slot in as a normalization before Q, K, V and before the FFN; the Flow view would
then show it as one more stage per branch.

### The next-word dataset (`nl_next`)

Sentences "subject verb object" over `WORDS`, read after a start token `.` (as in makemore): the
inputs are `. subject verb` and the targets `subject verb object`, so position t reads the words
up to t (causal) and should output word t + 1. The verb agrees with its subject (chases / chase)
and the object is the other animal in the other number (dog chases cats, cats chase dog), so
there are 4 sentences. After the verb, only the subject two positions back says which animal
comes next: the last position has to attend to it.

- `{ label, inputs: 21, outputs: 21, kind: 'seq', tokens: 3, vocab, decode, make }`. `vocab` maps
  each word to its one-hot slot (the WORDS tasks map words to vectors instead); `decode(y)` names
  each position by its most likely word (null where a value is not finite). `make()` also returns
  `words` (the inputs, `.` first) and `targetWords`, so the Train panel's stepper, sample names,
  **ŷ → word** column and **words** accuracy work as for the word datasets.
- Noise, when set, goes on the one-hot inputs only (from its own random stream); the targets stay
  one-hot, as cross-entropy needs.
- The first word is a four-way guess: nothing comes before it, so the best prediction there is
  1/4 on each noun and the loss floor is ln 4 / 3 ≈ 0.462. Word accuracy therefore tops out near
  75% (the first position is right 1 time in 4).
- Trained from the preset (lr 0.1, batch 10), p(cats | . dog chases) goes from 0.26 to 0.98 in
  500 steps and past 0.99 by 1000, and the verb and object positions to about 1.0 for every
  sentence. The two heads end up sharing the job: knocking either one out still leaves cats at
  0.94 or more, so the knock-out shows how redundant they are rather than one head doing it all.

## The flow

The flow is built from the net's structure by `buildFlow` (pure), stage by stage, left to right;
the stages wrap onto the next line when the free area is narrow. A **stage** is a column of tiles
with a numbered header; a **tile** is one matrix, rows = tokens, with its symbol and shape
(`Q₁ 3×2`). Cells are coloured by `colorFor` on the shared activations scale (attention weights
and probabilities on 0..1), with the fill's alpha capped at 0.72 so the digits can always be
`--text-1` (docs/DESIGN.md), zero cells unfilled and masked cells hatched.

On the tiny language model, the 14 stages are:

| # | stage | tiles | focus on click (lens) |
|---|---|---|---|
| 1 | Words | O, the one-hot rows, the vocabulary as column headers | layer 0 |
| 2 | Embed + position | O W_E + P = X | layer 1 |
| 3 | Q, K, V per head | Q_h, K_h, V_h for each head, a row per head in its colour | layer 2 |
| 4 | Scores | S_h = Q_h K_hᵀ/√d_h, causally masked cells −∞ | attention layer, part `scores` |
| 5 | Attention A | A_h = softmax(S_h) | part `softmax` |
| 6 | A V | Z_h = A_h V_h | part `mix` |
| 7 | Concat | Z = [Z_1 Z_2], a head-colour rule over each head's columns | attention layer |
| 8 | · W_O | Z W_O | layer 4 |
| 9 | + residual | H = X + Z W_O + b_O | layer 4 |
| 10 | FFN | F = ReLU(H W_1 + b_1) | layer 5 |
| 11 | · W_2 | F W_2 | layer 6 |
| 12 | + residual | Y = H + F W_2 + b_2 | layer 6 |
| 13 | Logits | ℓ = Y W_U + b_U, the vocabulary as column headers | layer 7 |
| 14 | Next word | p = softmax(ℓ): a bar per word at each position, captioned with the words it may see (`after . dog`); the most likely bar is solid, the target ringed in HI | layer 7 |

Above the stages, the input sentence and the predicted next word (the last position's most likely
word and its probability, with ✓ or ✗ against the target); the words are the one-hot input read
through `meta.vocab`, else the token names.

**Building rules**, for any net:
- Layer 0 is one tile (a one-hot input over `meta.vocab` shows its words).
- A Q, K, V layer that feeds an attention layer: one stage, Q_h, K_h, V_h per head (from
  `fwd.attn`). An attention layer: scores, softmax and A V per head, then concat with 2+ heads.
- A **tokenwise tied** layer (`tiedMatrices(net, l)` all tokenwise, and every fixed edge an
  identity copy of an earlier layer, slot for slot):
  - with a residual: a stage with the products (`Z W_O`), then a stage with the sum;
  - with a bias per position (no bias ties on a token layer): products + P = the layer, one stage;
  - with a softmax: a logits stage and a probabilities stage (bars when the width matches
    `meta.vocab`);
  - otherwise: one stage with the layer (`FFN` when it widens with a ReLU).
- **Anything else** (a plain dense layer, untied or grouped weights): one tile per layer, its
  activations as a row, each neuron traced to its incoming edges. A net without an attention
  layer says so above the flow, with a button that opens the tiny language model.

## Interaction

- **Stepping.** ◀ ▶ (or ← → while the view is on and Explain is not running) light one stage at a
  time: it gets an HI frame and its cells fill in, diagonal by diagonal; the stages not computed
  yet fade to 20%. Past either end, or ■, shows the whole pass again (no stage lit). ▶ (Shift+G)
  plays one stage every 1.6 s from the lit one (from the start when none is lit or the last one
  is) and stops at the end. The numbered chips jump to a stage. The lit stage scrolls into view
  when the flow is taller than its box.
- **Tracing.** Hovering a cell frames the cells it was computed from, one step back (a score: its
  query row and key row; a weight: its row of scores; A V: its row of A and the value column; a
  sum: its parts; a layer neuron: its incoming edges' neurons) and dims the rest. A tip above the
  tile does the arithmetic with the live numbers (`S₁[3,2] = q₃·k₂ × 0.71 = (0.21·0.29 + …) ×
  0.71 = …`). The hovered neuron, or the query token of a score or weight, becomes the shared
  hover, so the matrix panel and the cards light it too; a neuron hovered there frames its cell
  here, and a hovered token its rows.
- **Clicking** a stage's header lights it, sets the lens focus to its layer (and part: Q, K, V
  parts are not stages, so the Q, K, V stage focuses the whole layer) and reveals it in the
  matrix panel (`ctx.matrix.reveal`). Clicking a cell selects its neuron (`sel`), so the inspector
  opens its card beside the cell; a score or weight cell selects the attention layer. Clicking a
  word above the flow follows that token (lens `token`, again to stop); empty space deselects.
- **Heads.** With 2+ heads, a chip per head (in its colour) knocks that head out and back in. A
  knocked-out head's S, A and Z tiles fade with an "off" badge, its columns of the concat are 0
  (hatched), and every later stage is recomputed without it (`ablate`); the prediction above the
  flow says "head 2 off". The net itself is not changed, so the canvas, the matrix panel, the
  cards and training keep the full model. At least one head stays on.
- **1.2** shows or hides the numbers in the cells (on by default; narrow tiles, such as the FFN,
  the one-hot words and the logits, show them on hover only). **‹ sample ›** steps through the
  Train panel's samples (`ctx.train.stepSample`).
- **The lens** is read too: the focused layer's stages (or its part's) get a frame, a followed token
  its row in every tile, and a kept head dims the other heads' rows.
- **Live.** The tiles repaint on every `values` event, so they follow the Train panel as it
  trains, the samples it loads and every edit.

## `state.flow` (owner: flow.js)

```js
flow = null | {
  stage: null | int >= 0,     // the lit stage (0-based), null = the whole pass
  play: bool,                 // playing (only with a stage)
  off: int[],                 // knocked-out heads (0-based, sorted), of every attention layer that has them
  nums: bool,                 // numbers in the cells
  hover: null | { t: tileId, i, j },   // the cell the presenter points at (mirrored: its trace and tip)
}
```

- It starts null (off). Set it with `store.set('flow', …)`; `cleanFlow(v)` completes and repairs
  any object (null stays null). The presenter's play timer advances `stage`; the audience only
  follows it.
- **Mirror.** nn.js carries `flow` in `mirrorState()`, fires `onMirror` on its event and
  `applyMirror` sets it. The audience window draws the same stage, knocked-out heads, numbers and
  hovered cell with its trace and tip, read-only (its bar keeps only the caption).
- It is not saved; `meta.flow` in a net is what opens the view when that net loads.

## While the view is on

- The SVG is hidden (`visibility`), not removed. `ctx.view.nodeRect(id)` returns the rect of the
  neuron's cell (stage px, so the cards open beside it), `ctx.view.contentRect()` the flow's box,
  and `ctx.view.fit(ms)` refits the 2D canvas behind and the flow (F). The originals come back
  when the view closes.
- **Framing.** The flow sits in the roomiest free rectangle of the stage (at least 320 × 200), clear
  of the Train, Attention and 3D plots panels and above its bar and the lens bar, as view.js's
  fit does. It takes the largest zoom from 1.4 down to 0.72 at which the whole flow fits (the
  steps, then halving between the fitting one and the one above it); below that the text would be
  too small, so the box scrolls instead. A stage and the arrow after it wrap together. It refits when a panel opens,
  folds, moves or closes, and when the structure changes.
- **The bar** runs along the bottom of the stage, between the floating panels that reach down
  there, lifted above the lens bar when that one is under it (`.ui-float`). Its controls (Flow,
  ■ ◀ ▶ ▶|, the stage chips, the head chips, 1.2, ‹ sample ›) are `.ui-chrome`, so the clean view
  (H) and the audience keep only the caption: the lit stage's number, title, formula (KaTeX) and a
  sentence with the live numbers, or a line on the whole pass.
- **Rendering.** Plain DOM (a CSS grid per tile). The tiles are rebuilt only when the structure
  key changes (`flowKey`: the stages, the tiles' shapes and labels, the knocked-out heads); every
  `values` event repaints the cells in place, skipping unchanged colours and text. On the tiny
  language model the view costs less per frame than the SVG canvas it replaces.

`ctx.flow` (test and console handle) = `{ on, toggle(on?), open(patch?), step(±1), play(on?),
head(h), fit(), info() }`; `info()` reports the stage keys, the lit stage, the zoom, the free
area, the tile count, `why` and the prediction.

## Keys (when `ctx.active(e)`; never in the audience window)

| key | does |
|---|---|
| G | Flow view on / off |
| Shift+G | play / pause the stages (opens the view when it is off) |
| ← / → | Flow view: previous / next stage (Explain catches them first while it runs) |

## Pure exports (tests: `tests/nn_flow.test.mjs`)

```js
cleanFlow(v) -> flow | null
attendHeads(x, spec, off = Set) -> { heads: [{ Q, K, V, S, A, Z }], out }   // model.js's attention, heads in off
                                                                            //   left out of out (their columns 0)
propagate(net, a, from = 1, { off }) -> { a, z, attn }   // the forward pass again from layer `from`
ablate(net, fwd, off) -> { a, z, attn, from }            // fwd with those heads knocked out (from = -1: unchanged)
buildFlow(net, { fwd, off }) -> { stages, tiles, nodeCell, words, labels, next, attention, why, max, heads, T }
flowKey(F) -> string                                     // the structure key
texHtml(tex) -> html                                     // W_{out} -> W<sub>out</sub>, for tips and titles
```

The tests check that `attendHeads` gives `fwd.attn` exactly and `propagate` reproduces
`model.forward` from any layer on eight presets; that knocking head h out equals zeroing the rows
of W_O that read its columns of Z; the tiny language model's 14 stages, tile shapes, masks and
words; that every tile holds the forward pass and every sum adds up (X = O W_E + P, H = X + Z W_O
+ b_O, Y = H + F W_2 + b_2); that on every preset each neuron is drawn once, where its value is,
and every source is a drawn cell; the traces one step back; the knock-out tiles; that the
structure key ignores new values; and the fallback without attention. `tests/nn_model.test.mjs`
checks `nl_next` (one-hot rows, the grammar, decode, noise), the preset's architecture and Reset,
every gradient of the whole model against finite differences (every edge, tie group, bias and
input, with cross-entropy over the vocabulary at each position), and that training learns the
grammar.

## Limitations

- PNG and To board picture the 2D canvas, not the Flow view.
- A knocked-out head is a view of the forward pass only: the backward pass, the matrix panel and
  training keep every head.
- The trace goes one step back; a cell's full ancestry (everything before it through attention)
  is most of the flow.
- The fallback for layers that are not tokenwise tied shows each layer as one row of numbers and
  its weights only in the tip.

## Testing

`node --test "tests/*.test.mjs"`. In the browser (docs/FEATURE_GUIDE.md's recipe, with
`--use-angle=swiftshader --enable-unsafe-swiftshader`): `index.html#nn=tiny_lm` opens straight
into the Flow view; `mathboardNet.ctx.train.play()` / `pause()` train it, `ctx.flow.step(1)`,
`ctx.flow.head(1)` and `ctx.flow.info()` drive and read it, and the cells are
`.nnf-c[data-t="<tile>"][data-i="<row>"][data-j="<col>"]` (bars `.nnf-b`) to point at. For
`docs/media/net-flow.png` the preset was trained (1500 steps of batch 10 at lr 0.1), the sample
". dog chases" loaded, the Train panel folded, the lens bar closed (L), the Attention A stage
clicked (lens focus, matrix panel revealed) then ■, and the pointer put on A₁[3,2].
