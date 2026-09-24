# Net tab: sequences, weight tying and attention (contract extension)

This extends docs/NN_CONTRACT.md so the Net tab can show a real transformer block as matrix
multiplication. Everything here is optional in the JSON, so older nets load unchanged.

**Status:** implemented: model.js (tests: `tests/nn_model.test.mjs`), the canvas, the matrix
panel, the cards, the Train panel and the shell. The spec below is kept as it was written.
"model.js as implemented" and "Display as implemented" list where the code differs from it or
goes further. The binding interface is docs/NN_CONTRACT.md.

## Why

The base model is a feed-forward DAG of scalar neurons. Attention needs two things it lacks:
parameters shared across tokens (`W_Q` is applied to every token), and products of
activations (`q_i · k_j`, `A V`). This extension adds weight tying, fixed edges,
token-shaped layers and an attention layer.

## Network JSON additions

**Edges**
- `edge.tie: string | null`. Edges with the same `tie` share one parameter.
  - The id format is `<name>:<i>,<j>`, e.g. `W_Q:1,2`, meaning entry (i, j) (1-based) of a small
    named matrix. `<name>` is KaTeX source.
  - Setting the weight of any tied edge sets every edge in the group.
  - `randomize` draws one value per group.
  - `trainStep` updates the group with the sum of its edges' gradients.
- `edge.fixed: boolean`. Never changed by `trainStep`, `randomize` or `connectDense`. Used for
  residual identity edges and fixed pooling weights. The UI shows these read-only.

**Layers**
- `layer.tokens: int >= 1` (absent means 1, a plain vector). The layer's nodes form `tokens`
  rows of `d` features each.
- `layer.groups: string[]`, optional. The nodes split into equal named groups, e.g.
  `['Q','K','V']`, each `tokens × d`.
- Node order is group-major, then token-major, then feature. So node k of a grouped layer is
  group `floor(k / (tokens·d))`, token `floor(k / d) % tokens` and feature `k % d`. The size must
  divide evenly; `validate` reports it otherwise.
- `layer.kind: 'dense' | 'attention'` (absent means `'dense'`).
- An attention layer `{ kind: 'attention', tokens, heads = 1, causal = false, scale }`:
  - **Input:** the previous layer, which must have `groups: ['Q','K','V']` and the same
    `tokens`. Q and K share `d_k`; V has `d_v`. `heads` must divide `d_k` and `d_v`.
  - **Output:** `Z = softmax(Q Kᵀ · scale + M) V` per head, where `M` is the causal mask
    (−∞ above the diagonal). Heads are concatenated, and the result is flattened token-major.
    The layer size is `tokens · d_v`.
  - `scale` defaults to `1/√(d_k / heads)`.
  - The layer has no incoming edges and no biases, and its `act` is `identity`. `normalize`
    enforces this by dropping edges into it and zeroing its biases.
  - Its nodes may have outgoing edges (dense, tied or fixed) like any other layer.
- **`softmax` on a token layer** applies per token (and per group), not over the whole layer.
- **LayerNorm is deliberately absent.** With `d = 2` it maps every token to (±1, ∓1). Presets
  say so in their note.

## model.js additions

```js
tokenShape(net, layer) -> { tokens, d, groups: string[] | null }
reshape(net, layer, vec) -> { [group | 'X']: number[][] }        // tokens × d matrices
tiedMatrices(net, l) -> [{ name, W: number[][], fromGroup, toGroup, edges }]
    // If every edge into layer l is tied, returns the small shared matrices, written in the
    //   X W convention (rows = input feature, cols = output feature) so that Q = X W_Q.
    //   Edge-level (receiving-row) W for one token = W_Qᵀ; the full layer matrix is I_n ⊗ W_Qᵀ.
forward(...)   // also fwd.attn[l] = { heads: [{ Q, K, V, S, A, Z }] } for attention layers
backward(...)  // attention gradients:
               //   dV = Aᵀ dZ, dA = dZ Vᵀ, dS = A ⊙ (dA − rowsum(dA ⊙ A)),
               //   dQ = dS K · scale, dK = dSᵀ Q · scale.
               // Also bwd.attn[l] = { heads: [{ dZ, dA, dS, dQ, dK, dV }] } and
               //   bwd.tie[tieId] = the summed gradient.
               // bwd.edge[id] stays the edge's own contribution δ_i a_j.
DATASETS       // add sequence tasks with kind 'seq' and a tokens field. At least:
               //   one content-based task (e.g. every position outputs the token with the
               //   largest first feature), and
               //   one position-based task (e.g. previous token, causal, with positional
               //   features in the input).
               // Both must be learnable by the presets below within a few thousand steps.
PRESETS        // group 'Attention':
               //   attention: X (3 tokens × 2) -> tied QKV -> Z.
               //   transformer: one block: X -> tied QKV -> attention Z -> tied W_O
               //     + fixed residual from X -> tied FFN (d -> 2d or 4d, ReLU)
               //     + fixed residual -> output.
               //   causal: masked attention on the position task.
               //   multihead (optional): 2 heads.
               // Retrofit the existing conv and unrolled-RNN presets with ties, so training
               //   keeps the kernel / recurrent weights shared. Drop their "training unties
               //   them" notes.
```

`matrices(net)` keeps working for every dense layer; tied entries simply repeat values. For an
attention layer it returns `{ l, id, kind: 'attention', rows, terms: [] }`.

### model.js as implemented

Differences from the spec above:
- **Equal groups, so `d_k = d_v`.** Groups split a layer evenly, so Q, K and V all have the same
  `d`. `heads` must divide it.
- **Shared biases: `node.tie`.** Not in the spec, but needed, because otherwise training moves
  each token's bias on its own. Nodes with the same `tie` (format `<name>:<j>`, e.g. `b_Q:2`) share
  one bias. `setNode(..., { bias })` sets the group, `randomize` draws once, and `trainStep` steps
  by the summed gradient. `bwd.tie[id]` covers bias ties too. There are no bias ties on the input
  layer or on an attention layer (`validate` reports them, and edits and `normalize` drop them).
  The presets tie every token layer's biases.
- **`tiedMatrices(net, l)`** returns matrices whenever every non-fixed edge into `l` is tied, so
  `H = X + Z W_O` gives `[W_O]` (the fixed residual is left out: find it by `edge.fixed`). Each
  entry also has `ties` (a tie id per cell), `k` (the source layer) and `tokenwise` (the layer
  matrix is exactly `I ⊗ Wᵀ`). `edges` lists the member edges into this layer only.
- **`matrices()`** attention entries also carry `act: 'identity'` and `b` (all 0). Every other
  entry has `kind: 'dense'`.
- **`fwd.attn` and `bwd.attn`** are per-layer arrays, `null` except on attention layers.
  `fwd.attn[l]` also has `tokens`, `dk` (per head), `scale` and `causal`. `bwd.attn[l].heads[h].dQ`,
  `dK` and `dV` are this layer's own contribution to the Q, K, V layer's `dA` (skip edges from
  Q, K, V add theirs). `bwd.db` is all 0 on an attention layer, and `bwd.dW[l]` is `[]`.
- **Softmax xent on a token layer** is the mean over its tokens (and groups) of each one's
  `-Σ y log p`.
- **Setting weights.** `setWeight` returns false on a fixed edge. `connect(a, b, w)` on an
  existing tied edge sets its group, and leaves a fixed edge alone.
- **Wiring into an attention layer.** `connect` returns null and `connectDense` returns `[]`.
- **Edits degrade instead of refusing.** A layer whose size no longer splits into tokens × groups
  loses those fields. An attention layer whose Q, K, V input breaks becomes a dense layer: a node
  added to or removed from Q, K, V, a layer inserted between them, or Q, K, V removed. The
  reason: train.js's `resizeLayer` loops on `addNode` / `removeNode`, so a refusal there would
  hang it.
- **`setLayer`** also takes `{ causal, heads, scale }` on an attention layer. `heads` only
  applies if it divides `d`, `scale: null` restores the default, and `act` stays `identity`.
- **New exports:** `tokenPos(net, nodeId)` gives `{ l, index, g, group, token, feature }`, and
  `attnSpec(net, l)` gives `{ l, tokens, d, heads, dh, scale, causal }`. `activate(act, z, out, seg)`
  runs softmax per `seg` entries.
- **Datasets:** `seq_max` (the content task, 3 tokens × 2), `seq_prev` (the position task,
  causal, tokens `(c, cos θ, sin θ)`), `seq_minmax` (two heads: `(max x₁, min x₁)`), and
  `seq_addmax` (the transformer's task, 2 tokens × 2: `y_i = ReLU(x_i + x_max)`). All are
  `kind: 'seq'` with `tokens`, and all use mse.
- **Word datasets:** `nl_pronoun` and `nl_agree`, 3-word sentences over a 10-word vocabulary
  whose fixed 2-D vectors are the export `WORDS`. The first coordinate is the kind of word (dog 1.2,
  cat 0.6, itself 0, sees −0.6, chases −1.2), the second the number (0.6 singular, −0.6 plural:
  dogs, cats, themselves, see, chase). Articles are left out. Each token's target is the vector of a
  word in its sentence, so an attention layer whose values are the words can copy it, and every
  position has a target. No positions are needed, and none are in the input.
  - `nl_pronoun`, "dog sees itself": noun, verb, reflexive, agreeing in number (8 sentences). The
    reflexive outputs the noun it refers to; the noun and the verb output themselves. It uses the
    reflexive because a plain pronoun, as in "dog chases it", can't refer to the subject of its
    own clause.
  - `nl_agree`, "dog chases cats": subject, verb agreeing with it, object (16 sentences). The verb
    outputs its subject; the nouns output themselves. The object always has the other number, so
    the verb can find its subject by number alone (without word positions it has no other way).
  - Besides `X` and `Y`, `make()` returns `words` and `targetWords` (a sentence per sample), and the
    dataset has `vocab` (its words and vectors) and `decode(y)` (each token's nearest word). Noise
    is drawn from its own random stream, so a seed gives the same sentences at any noise level.
  - A causal next-word task was left out on purpose: with mse on word vectors, positions whose next
    word the grammar leaves open would need "don't care" targets (and so no backward pass for any
    sample), a softmax over the vocabulary doesn't fit the 40-node limit, and a grammar that fixes
    every next word has only a handful of sentences.
- **Presets** (group Attention, after Sequences): `words` (hand-set), `pronouns`, `agreement`,
  `attention`, `causal`, `causal_rot`, `multihead` and `transformer`.
  - The **transformer has 2 tokens**, because 3 tokens would need 54 nodes against the menu's
    40-node limit. Its `W_O` and `W_2` start small, so the block starts near `Y = X`.
  - Each attention preset records an `lr` in `meta.train.lr` (`PRESETS[k].lr`): 0.3 for
    `attention`, `causal` and `transformer`, 1 for `multihead`. At the panel's batch 10 they learn
    their pattern in 3000 steps.
  - **`pronouns` and `agreement`** are `attention`'s net (30 nodes) on the word datasets, with
    `Z` as the output, lr 0.3 and noise 0 (`PRESETS[k].noise`). `W_V` starts as the identity and
    `b_V` at 0, so each value is its word's own vector, and `W_Q`, `W_K` start small (the `small`
    scheme), so every word first reads all three about evenly. `meta.train.init` records that
    recipe (`{ W_Q: 'small', W_K: 'small', W_V: 'identity' }`, `randomize`'s `init` option), and
    the Train panel's Reset uses it, so a new init seed trains like the preset (init seeds 1 to 40
    all reached 100% in a trial run). From a fully random start many runs lock into a swap instead:
    rows read the wrong word and `W_V` learns to map it back (a local minimum;
    `tests/nn_model.test.mjs` has one). Trained from seed 1 for 3000 steps, both reach 100% word
    accuracy on held-out sentences. In `pronouns` the reflexive's row of A peaks on its noun in
    every sentence (mean weight 0.95) and the other rows on themselves; in `agreement` the verb's
    row peaks on its subject in every sentence (mean 0.96, about 0.01 on the object).
- **Retrofits:**
  - `conv1d`, `conv1d_s2` and `lenet`: kernel ties `k:1,<tap>` and one bias `b:1`.
  - `wavenet`: `k^{(l)}:1,1` (reads `x_{t-d}`) and `k^{(l)}:1,2` (reads `x_t`), plus `b^{(l)}:1`.
  - `rnn`: `W_{hh}:<from>,<to>`, `w_x:1,<to>` and `b_h:<to>`, in the X W convention.
  - `lenet`'s pool and `avgpool` are fixed edges.
  - The residual presets' identity shortcuts are unchanged (still trainable).

## Display expectations

**View**
- Token layers draw their nodes as token blocks: a rounded box per token labelled t₁…tₙ, with
  group bands such as Q / K / V.
- An attention layer draws data-dependent edges from V tokens to Z tokens, with width and
  opacity set by `A[i][j]`, plus a small n×n heatmap of `A` in its header.
- Hovering a Z token shows its attention row.
- Hovering or selecting a tied edge lights every edge in its tie group.
- Fixed edges are dashed and can't be edited.

**Matrix panel**
- Token layers show activations as `tokens × d` matrices (X, Q, K, V, Z, H) instead of flat
  vectors.
- Tied tokenwise layers show compactly as `Q = X W_Q`. An expander shows the flattened
  `I ⊗ W_Qᵀ` block-diagonal form, tying it back to `z = W a`.
- An attention layer shows `S = QKᵀ/√d_k`, then `A = softmax(S)` (a heatmap, with masked
  cells as −∞), then `Z = A V`. Residuals show as `+ X`.
- Step-through for attention works row by row: the scores `q_i · k_j` with numbers, the
  softmax, then the weighted sum of `v_j`.
- Backward mode shows the attention gradient equations above with numbers.

**Inspector**
- An attention neuron's card shows `z = Σ_j A_ij V_j,f` with numbers, the attention row as
  bars, and the scores.
- A tied edge's card says "shared parameter W_Q(1,2), used by n edges". Its slider moves all of
  them, and its gradient is shown as the sum of the per-token terms.
- A fixed edge's card is read-only.
- An attention layer's card shows tokens, heads, a causal toggle and the scale.

**Train**
- For `seq` datasets, the plot shows the attention matrix of the current sample, plus outputs
  against targets per token. A sample stepper replaces click-a-point.
- Neuron maps are off for `seq` datasets.

### Display as implemented

Everything above is in place. Differences and additions:
- **Layout.** The three 3-token presets (`attention`, `causal`, `multihead`) draw X and the Q, K, V
  layer as tokens × d grids, like the matrix panel: a row per token, a column per feature, the
  groups stacked, with X halfway between the Q and K blocks so no edge runs level through the
  next neuron of a row. Q, K, V is then 9 rows high instead of an 18-neuron column, and in a
  1600 × 900 window with the Train panel and a 40% matrix panel these presets fit at 0.63 to 0.73
  zoom instead of about 0.45. Later layers, and every other preset, stay columns; **Layout** makes
  columns. A layer of one token draws no token boxes: its group band is enough.
- **Heatmap and attention edges.** The heatmap sits just above the attention layer's header, one
  n×n block per head. Attention edges run V_j,f -> Z_i,f, one per feature; causally masked pairs
  have none, and their heatmap cells are hatched.
- **Hover.** Hovering a token box, a heatmap cell or an attention edge sets the shared hover
  `{ kind: 'token', layer, t, g?, h? }` (docs/NN_CONTRACT.md), so the matrix panel, the cards and
  the audience window follow it. A Z token or a query token shows its row of A, a key or value
  token its column; a Z or V neuron, the step-through's token and the selected neuron do too.
  The matrix panel's t_i headers and S / A cells, and the attention layer card's A tables, hover
  tokens the same way.
- **Edits keep token layers whole.** A double-click on a token layer adds a feature to every token
  (with a new shared entry for each tied matrix), or says why it can't; wiring two tokenwise-tied
  layers adds a new shared entry for every token. The shell's Delete refuses one neuron of a
  token, Q, K, V or attention layer, a fixed edge, and the Q, K, V layer an attention layer
  reads; + Layer refuses to go right before an attention layer or next to a token layer. Each
  refusal toasts why. The model underneath still degrades instead of refusing (see above), so
  imported or scripted edits keep working.
- **Matrix panel.** The attention stages are numbered (1 · scores, 2 · softmax, 3 · weighted sum),
  per head with multihead, then the heads concatenated. The formula on top writes a whole token
  net (every layer compact or attention) as one line, naming layers used twice in "where" lines.
  A tied layer's gradient is Xᵀδ, shown as the sum of one outer product per token, and a shared
  bias row as 1 bᵀ. The Batch view keeps token layers flat and shows attention's Z per sample
  only (A changes with every input, so there is no fixed W). Collapse says why a net with
  attention can't be one matrix.
- **Step-through.** An attention layer steps a token at a time: forward in three phases
  (`anim.phase` `'scores'`, `'softmax'`, `'sum'`), backward once per token (`'delta'`).
- **Numbers.** Gradients below 0.01 keep two significant figures (0.0034, 3.4e−4) everywhere, so
  the small gradients of these presets' default samples don't read 0.00.
- **Train.** The plot picks the attention layer and head (`meta.train.space = '<layer id>#<head>'`).
  **Adapt network** resizes a token net token by token and refuses (with the reason) what it can't
  rewire.
- **Word datasets in the Train panel.** The stepper reads `sample 3 of 200: “cat sees itself”`, and
  loading a sample (◀ ▶, `loadSample`) sets `meta.tokenNames` to its words in the same commit, so
  the canvas, the matrix panel, the lens bar and the attention panel name the tokens. The plot's
  output column, **ŷ → word**, names each token's output by its nearest word, marks it ✓ or ✗
  against the target word, and shows the target below it; hovering it gives the numbers. The
  readout adds **words**, the word accuracy over the whole dataset (`wordAccuracy`).
