# Net tab: the Flow view and the tiny language model (contract)

This extends docs/NN_CONTRACT.md, NN_ATTENTION.md and NN_LENS.md. One module, `static/nn/flow.js`
(with `flow.css`), draws one whole forward pass of the current net left to right, as live matrix
tiles with their shapes, down to the softmax over the next word. Like the 3D view, it covers
`#nn-stage` and replaces the SVG canvas while it is on. The matrix panel steps aside while it is on
(the flow shows the matrices itself: `ctx.matrixAway`, not saved, and dragging the divider brings it
back); the cards, the Train and Attention panels, the lens bar and Explain keep working over it and
stay linked through the store.
It adds one store key, `flow`, which the audience window mirrors, and one preset made for it:
**Tiny language model** (`tiny_lm`) on the next-word dataset `nl_lm`, with 17 variants of it that
each change one thing (Variants, below), reachable from a picker in the Flow bar.

Toggle it with the toolbar's **Flow** button (group `view`, beside 3D) or **G**. Shift+G plays the
stages. The Flow view and the 3D view replace the canvas one at a time: opening one closes the other.

## The tiny language model (`tiny_lm`, model.js)

A decoder-only transformer small enough to read, trainable end to end:

| layer | shape | computes |
|---|---|---|
| 0 Words (one-hot) | 5 × 23 | position t holds its word as a one-hot row over the 23-word vocabulary |
| 1 X = E[w] + P | 5 × 8 | `X = O W_E + P`: the tied embedding W_E (23 × 8); P, a learned vector per position, is this layer's own untied biases |
| 2 Q, K, V | 5 × 8 each | `Q = X W_Q + b_Q` and so on, tied 8 × 8 matrices, biases shared per feature |
| 3 Attention Z | 5 × 8 | 2 causal heads of d_h = 4, scale 1/2; `Z = [Z_1 Z_2]` |
| 4 H = X + Z W_O | 5 × 8 | tied W_O (8 × 8), + X on fixed residual edges, + b_O |
| 5 FFN | 5 × 32 | `F = ReLU(H W_1 + b_1)`, d → 4d |
| 6 Y = H + FFN | 5 × 8 | `Y = H + F W_2 + b_2`, + H on fixed residual edges |
| 7 Next word | 5 × 23 | `p = softmax(Y W_U + b_U)` at each position (softmax per token), cross-entropy loss |

**Widths.** d_model is 8. X, H and Y are the residual stream: `H = X + Z W_O` and `Y = H + F W_2`
add cell by cell, so each branch has to come back to the stream's 8 columns and the stream keeps
one width all the way through. The FFN widens to 4 d_model = 32 inside, and the words set the
input's and the output's width, 23. The Flow view shows all three (see Widths below).

670 neurons and 5760 edges, so it is the one preset exempt from the menu's 40-neuron limit (the
test says so): the canvas draws it as a wide strip of token grids, and the Flow view is where it
reads. `meta.flow = true` opens the Flow view when the preset loads (and on a `#nn=tiny_lm`
preload). `meta.vocab` names the 23 one-hot slots, by role: `the`; `cat dog bird`; `sat slept
jumped ran flew`; `on in over to`; `mat box wall rug yard fence park branch nest tree`. `normalize`
keeps both, as it keeps any unknown meta field. The inputs hold the headline sentence, "the cat
sat on the", with the targets "cat sat on the mat", so the view opens on it; `meta.tokenNames`
names the tokens after it. `meta.train.n` is 300, so the Train panel draws 300 sentences.

**Initialisation.** The weights are drawn by `randomize` with the Train panel's Reset recipe (He,
as the net has a ReLU, biases 0, so P starts at 0), except W_Q, W_K, W_O and W_2, which start small
(`meta.train.init = { W_Q: 'small', W_K: 'small', W_O: 'small', W_2: 'small' }`): every position
first reads its past about evenly, and the block starts near Y = X. Reset with init seed 1 gives
back the preset's exact weights. `PRESETS.tiny_lm.lr` is 0.2 and `noise` 0.

**No LayerNorm.** With one block, d = 8 and the small init above, training is stable without it,
so tiny_lm has none. The norm variants add it (and RMSNorm) as layer-wide activations: pre-norm as
a normalization before Q, K, V, before the FFN and before the logits, each one more stage in the
Flow view, and post-norm on the two sums (Variants).

### The next-word datasets (`nl_lm`, and `nl_next`)

Both are built by one helper: a sentence of T + 1 one-hot words, the first T as the inputs and the
last T as the targets, so position t reads the words up to t (causal) and should output word t + 1.

- `{ label, inputs, outputs, kind: 'seq', tokens, loss: 'xent', vocab, decode, make }`. `vocab` maps
  each word to its one-hot slot (the WORDS tasks map words to vectors instead); `decode(y)` names
  each position by its most likely word (null where a value is not finite). `make()` also returns
  `words` (the inputs) and `targetWords`, so the Train panel's stepper, sample names,
  **ŷ → word** column and **words** accuracy work as for the word datasets.
- Noise, when set, goes on the one-hot inputs only (from its own random stream); the targets stay
  one-hot, as cross-entropy needs.

**`nl_lm`** (5 positions × 23 words, the tiny language model's): "the SUBJECT VERB PREPOSITION the
OBJECT", six words from a small grammar, of which the model reads the first five. The subject picks
its verbs, the verb its prepositions, and the object is fixed by the subject and the preposition
together:

| subject | verbs | on | in | over | to |
|---|---|---|---|---|---|
| cat | sat, slept, jumped | mat | box | wall | |
| dog | sat, slept, jumped, ran | rug | yard | fence | park |
| bird | sat, slept, flew | branch | nest | tree | nest |

with sat and slept taking on or in, jumped on or over, ran to or in, and flew to or over. That makes
20 sentences, such as "the cat sat on the mat", "the dog ran to the park" and "the bird flew over
the tree". Every choice is uniform (a subject 1/3, then its verbs, then the verb's prepositions).
- **Why attention.** The last position reads "the", like the first, so its own word says nothing
  about what comes next. "sat on the" ends in mat, rug or branch, depending on the subject three
  words back, and the same subject ends differently after on, in and over: the last position has to
  look back at both.
- **What can be learned.** The first three positions stay guesses (1/3 on each subject; the
  subject's verbs; on or in), after the preposition comes "the", and the last word is sure. So the
  loss floor is about 0.60 (ln 3, the verbs' mean entropy and ln 2, over 5 positions) and word
  accuracy tops out near 64%.
- **Trained from the preset** through the Train panel (lr 0.2, batch 10, 300 points),
  p(mat | the cat sat on the) is 0.95 after 300 steps, 0.99 after 400, 0.998 after 600 and 0.999
  after 1000, when the loss is about 0.62. At 1000 steps every sentence's last word gets 0.97 or
  more, and row 5 of A puts about half its weight on the preposition in both heads, with head 2
  also reading the subject (0.24 on "cat"). Knocking head 2 out turns both "the cat sat on the"
  and "the dog sat on the" into wall (0.42 and 0.45): without it the subject and the preposition no
  longer come together. Knocking head 1 out leaves mat at 0.98 there, but turns "the bird sat in
  the" into branch, the bird's word for on. How the heads share the job changes from run to run
  (another init seed or batch order).

**`nl_next`** (3 positions × 7 words) is kept for nets saved with the first tiny language model:
"subject verb object" over `WORDS`, read after a start token `.`, so `. dog chases` has the targets
`dog chases cats`. The verb agrees with its subject and the object is the other animal in the other
number (4 sentences). The first word is a four-way guess (loss floor ln 4 / 3 ≈ 0.462). Such a net
still opens in the Flow view, with the same numbered sentence, labels, widths and ending.

## The flow

The flow is built from the net's structure by `buildFlow` (pure), stage by stage, left to right;
the stages wrap onto the next line when the free area is narrow. A **stage** is a column of tiles
with a numbered header; a **tile** is one matrix, rows = tokens, with its symbol and shape
(`Q₁ 5×4`). Cells are coloured by `colorFor` on the shared activations scale (attention weights
and probabilities on 0..1), with the fill's alpha capped at 0.72 so the digits can always be
`--text-1` (docs/DESIGN.md), zero cells unfilled and masked cells hatched.

On the tiny language model, the 14 stages are:

| # | stage | tiles | focus on click (lens) |
|---|---|---|---|
| 1 | Words | O, the one-hot rows, the vocabulary as column headers | layer 0 |
| 2 | Embed + position | O W_E + P = X | layer 1 |
| 3 | Q, K, V per head | Q_h, K_h, V_h for each head, a row per head in its colour | layer 2 |
| 4 | Scores | S_h = Q_h K_hᵀ/√d_h, causally masked cells −∞, the keys numbered by position | attention layer, part `scores` |
| 5 | Attention A | A_h = softmax(S_h) | part `softmax` |
| 6 | A V | Z_h = A_h V_h | part `mix` |
| 7 | Concat | Z = [Z_1 Z_2], a head-colour rule over each head's columns | attention layer |
| 8 | · W_O | Z W_O | layer 4 |
| 9 | + residual | H = X + Z W_O + b_O | layer 4 |
| 10 | FFN | F = ReLU(H W_1 + b_1) | layer 5 |
| 11 | · W_2 | F W_2 | layer 6 |
| 12 | + residual | Y = H + F W_2 + b_2 | layer 6 |
| 13 | Logits | ℓ₅ = y₅ W_U + b_U, the last position's scores, the vocabulary as column headers | layer 7 |
| 14 | Next word | p₅ = softmax(ℓ₅): the next word's distribution as a bar chart over the vocabulary | layer 7 |

**The beginning.** Above the stages, the input sentence in order as numbered chips (① the ② cat
③ sat ④ on ⑤ the), an arrow, and the next word: ⑥ ? while a stage before the last is lit, then
the most likely word with its probability, ✓ or ✗ and "true next word: mat" once the pass has
reached the end (the whole pass or the last stage). The words are the one-hot input read through
`meta.vocab`, else the token names. A plain net shows no chips.

**Rows.** Every matrix names its rows: the first tile of each stage row carries the labels, and
the tiles beside it share its rows (they sit on the same grid). With words as tokens a row reads
①the, its position's number and word; otherwise the token's label (t1, t2 or its name). The scores
and weights number their key columns by position (1 2 3 4 5), and the tips name positions the same
way ("2 “cat” can't see the later 4 “on”").

**Widths.** One cell width for the whole flow, so a tile's width shows how many columns it has:
34 px with the numbers, or, when some tile has more than 10 columns (the tiny language model),
18 px squares 1 px apart with short numbers (.12, −.34, 1.2) and zeros left blank. A bracket under a
tile gives its width: d_model = 8 under the residual stream (X, H, Y and the Z W_O, F W_2 they add),
4 d_model = 32 under the FFN (d_ff when it is not 4×), vocab = 23 under the one-hot words, the
logits and the chart. A line under the sentence says why: "Residual additions (X + attention, H +
FFN) add cell by cell, so the residual stream X, H and Y keeps one width, d_model = 8. The FFN
widens to 32 and back; the logits have one column per word, 23." The + residual and FFN captions say
the same for their stage.

**The ending.** A language model's head (a softmax over `meta.vocab` on the last layer, at 2+
positions) shows only what generating reads, the last position: the logits stage has one row (ℓ₅,
from row 5 of Y; the other rows of Y fade, since nothing after reads them) and the Next word stage
is one bar chart over the vocabulary, the words under the bars, the most likely bar solid with its
word in `--text-1`, the true next word ringed in HI with its word in `--hi-text`, and a key. The
stage's **last position | every position (how it's trained)** toggle (`.ui-seg`, `.ui-chrome`)
switches to every position: the logits get all 5 rows and the ending a row of bars per position,
captioned with what that position has seen and its guess ("after the cat → sat .34"), the true next
word ringed, and the note "Training scores every position against its true next word, and the loss
is the mean over the 5; generating reads only the last one." The stage is then called "Next word,
every position". Any other softmax output keeps every row.

**Building rules**, for any net:
- Layer 0 is one tile (a one-hot input over `meta.vocab` shows its words).
- A Q, K, V layer that feeds an attention layer: one stage, Q_h, K_h, V_h per head (from
  `fwd.attn`). An attention layer: scores, softmax and A V per head, then concat with 2+ heads.
- A **tokenwise tied** layer (`tiedMatrices(net, l)` all tokenwise, and every fixed edge an
  identity copy of an earlier layer, slot for slot):
  - with a residual: a stage with the products (`Z W_O`), then a stage with the sum; the sum is
    recorded (`F.residual`) for the widths;
  - with a bias per position (no bias ties on a token layer): products + P = the layer, one stage;
  - with a softmax: a logits stage and a probabilities stage (bars when the width matches
    `meta.vocab`; a language model's head as The ending above);
  - otherwise: one stage with the layer (`FFN` when it widens with a ReLU).
- **Anything else** (a plain dense layer, untied or grouped weights): one tile per layer, its
  activations as a row, each neuron traced to its incoming edges. A net without an attention
  layer says so above the flow, with a button that opens the tiny language model.

## Interaction

- **Stepping.** ◀ ▶ (or ← → while the view is on and Explain is not running) light one stage at a
  time: it gets an HI frame and its cells fill in, diagonal by diagonal; the stages not computed
  yet fade to 20%, and the next word above the flow reads ? until the last stage. Past either end,
  or ■, shows the whole pass again (no stage lit). ▶ (Shift+G) plays one stage every 1.6 s from the
  lit one (from the start when none is lit or the last one is) and stops at the end. The numbered
  chips jump to a stage. The lit stage scrolls into view when the flow is taller than its box.
- **Tracing.** Hovering a cell frames the cells it was computed from, one step back (a score: its
  query row and key row; a weight: its row of scores; A V: its row of A and the value column; a
  sum: its parts; the last position's logits: row 5 of Y; a layer neuron: its incoming edges'
  neurons) and dims the rest. A tip above the tile does the arithmetic with the live numbers
  (`S₁[3,2] = q₃·k₂ × 0.50 = (0.21·0.29 + …) × 0.50 = …`, `p(mat | the cat sat on the) = e^ℓ / Σ e^ℓ
  = 1.00, the most likely word, the true next word`). The hovered neuron, or the query token of a
  score or weight, becomes the shared hover, so the matrix panel and the cards light it too; a
  neuron hovered there frames its cell here, and a hovered token its rows.
- **Clicking** a stage's header lights it, sets the lens focus to its layer (and part: Q, K, V
  parts are not stages, so the Q, K, V stage focuses the whole layer) and reveals it in the
  matrix panel (`ctx.matrix.reveal`). Clicking a cell selects its neuron (`sel`), so the inspector
  opens its card beside the cell; a score or weight cell selects the attention layer. Clicking a
  word above the flow follows that token (lens `token`, again to stop); empty space deselects.
  The ending's toggle sets `flow.every`.
- **Heads.** With 2+ heads, a chip per head (in its colour) knocks that head out and back in. A
  knocked-out head's S, A and Z tiles fade with an "off" badge, its columns of the concat are 0
  (hatched), and every later stage is recomputed without it (`ablate`); the prediction above the
  flow says "head 2 off". The net itself is not changed, so the canvas, the matrix panel, the
  cards and training keep the full model. At least one head stays on.
- **1.2** shows or hides the numbers in the cells (on by default; the short ones in narrow cells).
  **‹ sample ›** steps through the Train panel's samples (`ctx.train.stepSample`).
- **The lens** is read too: the focused layer's stages (or its part's) get a frame, a followed token
  its row in every tile (the last-position tiles when it is the last), and a kept head dims the
  other heads' rows.
- **Live.** The tiles repaint on every `values` event, so they follow the Train panel as it
  trains, the samples it loads and every edit.

## Variants (the tiny language model's family)

Seventeen presets, each the tiny language model with one change, all on `nl_lm` with the same
sizes (d_model 8, 5 positions, 23 words, an FFN of 32), the same recipe (He, biases 0, W_Q, W_K,
W_O and W_2 small, recorded as `meta.train.init`, so Reset with init seed 1 gives the preset back)
and the same headline sentence. They make the New net menu's **Tiny LM variants** section. In
`PRESETS` each (tiny_lm too) has `family: 'tiny_lm'`, an `axis` (Baseline, Positions, Attention,
Norm, FFN, Other), a `short` name and the `title` its nets carry.

**The picker.** While the flow shows a net of the family, its bar starts with a select
(`.ui-field.sm`, inside the bar's `.ui-chrome`, so H and the audience hide it) listing the whole
family by axis. Picking one builds it (`store.load`, so Ctrl+Z comes back; `ctx.flow.variant(key)`
does the same) on the same data: the Train panel's points, noise, data seed, batch, speed and init
seed carry over (the net is built with that init seed, as Reset would draw it), and so does the
sentence in the inputs, with its targets and token names. The variant keeps its own recorded
learning rate. The lit stage and knocked-out heads reset, and the note toasts. Under the sentence
a line says what the variant changes (its note).

| key | axis | what changes | what the flow shows | lr |
|---|---|---|---|---|
| `tiny_lm_nope` | Positions | no position vector: X's biases fixed at 0 (`node.fixed`) | "Embed", X = O W_E as one tile; the caption says the order reaches the model only through the causal mask | 0.2 |
| `tiny_lm_sin` | Positions | P fixed: sin(t / 10000^(2i/8)) in column 2i + 1, cos in 2i + 2 (t the position, 0-based), as fixed biases | "Embed + position (fixed)", P beside O W_E, each cell's tip its formula | 0.2 |
| `tiny_lm_rope` | Positions | no P; the attention layer's `pos: 'rope'` | "Rotate q, k (RoPE)" before the scores: Q̃_h, K̃_h per head, the tip the rotation with numbers; the scores read them | 0.2 |
| `tiny_lm_alibi` | Positions | no P; `pos: 'alibi'` | the scores as QKᵀ/√d + B = S per head, B the fixed penalty on its own colour scale | 0.2 |
| `tiny_lm_mqa` | Attention | one K and one V for both heads: K and V tied across heads, so W_K and W_V are 8 × 4 | head 2's K and V lighter, with a "= K₁" badge; the Q, K, V caption says multi-query and the sizes | 0.2 |
| `tiny_lm_gqa` | Attention | 4 heads of d_h = 2, 2 K, V heads (heads 1, 2 and 3, 4 share) | the badges on heads 2 and 4; four head chips | 0.2 |
| `tiny_lm_window` | Attention | `window: 3`: a position reads itself and the 2 before it | the band mask in S and A; a masked tip says "outside the window of 3" | 0.2 |
| `tiny_lm_linear` | Attention | `linear: true`: no softmax | "Feature map φ" (φ(Q_h), φ(K_h)), the scores φ(q)·φ(k) with no scale and the masked cells blank, then "Weights A" = S over its row sum | 0.2 |
| `tiny_lm_nomask` | Attention | `causal: false` | no hatching; the captions say every position sees its own next word | 0.2 |
| `tiny_lm_prenorm` | Norm | N₁ = LN(X), N₂ = LN(H), N₃ = LN(Y) before Q, K, V, the FFN and the logits (fixed identity edges, fixed zero biases); the residuals skip them | a LayerNorm stage each, the tip (x − μ)/σ with numbers | 0.5 |
| `tiny_lm_postnorm` | Norm | H = LN(X + Z W_O + b_O), Y = LN(H + F W_2 + b_2) | "+ residual, LayerNorm" sums, LN(…) in the formulas and the tips | 0.5 |
| `tiny_lm_rmsnorm` | Norm | pre-norm with RMSNorm | RMSNorm stages, the tip x / rms | 0.5 |
| `tiny_lm_gelu` | FFN | the FFN's GELU | "FFN" with GELU in its caption | 0.2 |
| `tiny_lm_swiglu` | FFN | the FFN layer as groups G, U (5 × 32 each), act `swiglu`: W_1 into G, W_3 into U, W_2 reading G | "Gate and up" (G before silu, and U), then "SwiGLU" (F = silu(G) ⊙ U), which F W_2 reads | 0.2 |
| `tiny_lm_tied` | Other | no W_U: the logits' edges tie to W_E's entries | the logits read W_Eᵀ, and the caption says why | 0.2 |
| `tiny_lm_2layer` | Other | two pre-norm blocks, each with its own matrices (W_Q^(1), W_Q^(2), ...) | 29 stages; the stream is X, H₁, Y₁, H₂, Y₂ | 0.5 |
| `tiny_lm_window2` | Other | the same with a window of 3 in both blocks | the band mask in both attention layers | 0.5 |

**The model's side** (model.js, listed in docs/NN_CONTRACT.md):
- Activations `gelu` (the tanh form), `layernorm` and `rmsnorm` (per token on a token layer, as
  softmax; no learned gain or shift; ε = 1e-5 under the root) and `swiglu` (the layer's first half
  gates its second: a = [silu(g) ⊙ u, u]). The last three are layer-wide (`vector: true`), with
  their Jacobians in the backward pass.
- `node.fixed`: a bias that is never trained, randomized or set (X's in NoPE, RoPE and ALiBi, the
  sinusoidal P, the norm layers' zeros). Its gradient is still reported.
- Attention layers take `window`, `pos` (`'rope'` | `'alibi'`) and `linear`. RoPE turns pair p of
  a head's columns (2p + 1, 2p + 2) by t · 10000^(−2p/d_h) at position t (0-based): with d_h = 4,
  57.3° and 0.57° per position. ALiBi's slopes are 2^−(h+1) (½, ¼, …), the paper's first ones for
  8 heads; its 2-head slopes (1/16, 1/256) would barely show over 5 positions. Linear attention
  uses φ = elu + 1: S = φ(q)·φ(k) on the visible cells and A = S over its row's sum. `fwd.attn`'s
  heads then also carry Qr, Kr (RoPE's turned q, k), Qf, Kf (φ of them) and B (ALiBi's bias); Q
  and K stay the Q, K, V layer's own.
- Multi-query and grouped-query attention need no new maths: the ties share K and V. Tied
  embeddings neither; `tiedMatrices` reads the logits' matrix as W_E transposed (`transposed`).
- `attend(x, spec)` computes one attention layer as `forward` does (the flow's knock-outs use it);
  `attnVisible(spec, i, j)`, `ropeFreq(p, d_h)`, `alibiSlope(h)` and `NORM_EPS` are exported.

**Learning rates.** 0.2, as tiny_lm, except the normalized variants (the three norms and the two
blocks): 0.5, where they train best. At 0.5 NoPE, the sinusoidal P and multi-query diverge. The
two-block variants are pre-norm because without the norms two blocks diverge even at 0.1 (at
0.05 they train, slowly: p(mat) 0.84 after 600 steps).

**What they show**, trained from the preset in node (batch 10 from 300 sentences, the recorded
rate, 600 steps; the loss over the 300, and how many of the 20 sentences get more than 0.8 on
their last word):

| variant | loss | p(mat \| the cat sat on the) | last word > 0.8 |
|---|---|---|---|
| tiny_lm | 0.631 | 0.998 | 20 / 20 |
| no positions, sinusoidal, RoPE, ALiBi | 0.635, 0.643, 0.636, 0.634 | 0.997, 0.988, 0.996, 0.996 | 20 / 20 each |
| multi-query, grouped-query | 0.641, 0.635 | 0.997, 0.998 | 20 / 20 each |
| window 3 | 0.811 | 0.406 | 1 / 20 |
| linear attention | 0.643 | 0.992 | 19 / 20 |
| no mask | 0.002 | 0.999 | 20 / 20 |
| pre-norm LN, post-norm LN, pre-norm RMSNorm | 0.630, 0.635, 0.626 | 0.980, 0.976, 0.983 | 20, 18, 20 / 20 |
| GELU, SwiGLU | 0.633, 0.615 | 0.998, 0.997 | 20 / 20 each |
| tied embeddings | 0.625 | 0.990 | 20 / 20 |
| 2 layers, 2 layers with window 3 | 0.632, 0.637 | 0.973, 0.951 | 18, 19 / 20 |

- **Positions don't matter here.** The task is about content (which subject, which preposition),
  and the causal mask alone tells position 1 from position 5 (one sees one word, the other five),
  so no position vector, a fixed one, RoPE and ALiBi all end where tiny_lm does.
- **The window.** The last position sees only "sat on the", so cat, dog and bird get exactly the
  same distribution there (mat 0.41, rug 0.31, branch 0.28), while "the dog ran to the" still gets
  park (only the dog runs). Two blocks with the same window get it back (0.95): in the first the
  verb and the preposition read the subject, in the second the last position reads them.
- **No mask.** Positions 1 to 4 copy their next word from the position after them (after "the",
  dog gets more than 0.95 when the sentence is about a dog), so the loss falls far below the causal
  floor of about 0.60; the last position has nothing to copy and learns as before.
- **Linear attention** does nearly as well with flatter weights: the last row's entropy is 1.52
  against softmax's 1.30 (uniform over 5 is 1.61).
- **The norms** start higher (loss 3.65 against 3.06: every row normalized makes larger logits)
  and train as well at 0.5.

**Building rules they add** (`buildFlow`, for any net):
- A layer whose biases are all fixed at 0 has no bias term; a token layer's fixed non-zero biases
  are a fixed P.
- A norm layer (layernorm or rmsnorm, fed only by fixed identity edges from one earlier layer,
  fixed zero biases): one stage (`norm<l>`), its tile marked d_model. A sum with a norm activation
  is the "+ residual, LayerNorm" stage (post-norm).
- A SwiGLU layer (groups G, U from one source): `gate<l>` (tiles `gate<l>`, the gate before silu,
  and `L<l>.U`) and `glu<l>` (`L<l>.G`, F, the FFN for the widths); the next layer's product reads F.
  GELU widens like ReLU (the FFN).
- A shared matrix read transposed is named `W^⊤`.
- RoPE adds `rope<l>` (tiles `rq<l>.<h>`, `rk<l>.<h>`), linear attention `phi<l>` (`fq`, `fk`) and
  the weights stage "Weights A"; ALiBi makes each head's scores row `qk<l>.<h>` + `ab<l>.<h>` =
  `s<l>.<h>`, the B tiles on scale `'bias'` (`F.max.bias`). The mask comes from `attnVisible`.
- Heads whose K (V) columns carry the same tie ids share it: their K, V tiles get `same` (the
  first such head's name), drawn lighter with a badge, and the Q, K, V caption names the kind.
- A layer named `H₂ = …` is H_{2}; a row of a symbol with a subscript reads (n_{3})_{5}.
- `F.variant` = `variantOf(net)` (`{ key, axis, short, note }` or null, matched by title) and
  `variantMenu()` the picker's list; a variant's structure key includes its key and masks.

## `state.flow` (owner: flow.js)

```js
flow = null | {
  stage: null | int >= 0,     // the lit stage (0-based), null = the whole pass
  play: bool,                 // playing (only with a stage)
  off: int[],                 // knocked-out heads (0-based, sorted), of every attention layer that has them
  nums: bool,                 // numbers in the cells
  every: bool,                // a language model's ending at every position (how it's trained), else the last
  hover: null | { t: tileId, i, j },   // the cell the presenter points at (mirrored: its trace and tip)
}
```

- It starts null (off). Set it with `store.set('flow', …)`; `cleanFlow(v)` completes and repairs
  any object (null stays null). The presenter's play timer advances `stage`; the audience only
  follows it.
- **Mirror.** nn.js carries `flow` in `mirrorState()`, fires `onMirror` on its event and
  `applyMirror` sets it. The audience window draws the same stage, knocked-out heads, numbers,
  ending and hovered cell with its trace and tip, read-only (its bar keeps only the caption).
- It is not saved; `meta.flow` in a net is what opens the view when that net loads.

## While the view is on

- The SVG is hidden (`visibility`), not removed. `ctx.view.nodeRect(id)` returns the rect of the
  neuron's cell (stage px, so the cards open beside it; null for an output neuron the ending does
  not draw), `ctx.view.contentRect()` the flow's box, and `ctx.view.fit(ms)` refits the 2D canvas
  behind and the flow (F). The originals come back when the view closes.
- **Framing.** The flow sits in the roomiest free rectangle of the stage (at least 320 × 200), clear
  of the Train, Attention and 3D plots panels and above its bar and the lens bar, as view.js's
  fit does. It takes the largest zoom from 1.4 down to 0.72 at which the whole flow fits (the
  steps, then halving between the fitting one and the one above it); below that the text would be
  too small, so the box scrolls instead. A stage and the arrow after it wrap together. It refits when a panel opens,
  folds, moves or closes, and when the structure changes. The tiny language model fits at 0.745 in
  a 1600 × 900 window with the Train panel open and at 0.81 with it folded; at 1280 × 800 it scrolls.
- **The bar** runs along the bottom of the stage, between the floating panels that reach down
  there, lifted above the lens bar when that one is under it (`.ui-float`). Its controls (Flow,
  ■ ◀ ▶ ▶|, the stage chips, the head chips, 1.2, ‹ sample ›) are `.ui-chrome`, so the clean view
  (H) and the audience keep only the caption: the lit stage's number, title, formula (KaTeX) and a
  sentence with the live numbers, or a line on the whole pass.
- **Rendering.** Plain DOM (a CSS grid per tile). The tiles are rebuilt only when the structure
  key changes (`flowKey`: the stages, the tiles' shapes, labels, brackets and faded rows, the
  knocked-out heads, the ending); every `values` event repaints the cells in place, skipping
  unchanged colours and text.

`ctx.flow` (test and console handle) = `{ on, toggle(on?), open(patch?), step(±1), play(on?),
every(on?), head(h), variant(key), fit(), info() }`; `variant(key)` builds another variant of the
tiny language model on the same data, as the picker does; `info()` reports the stage keys, the lit
stage, the zoom, the free area, the tile count, `why`, the prediction, `lm`, `every`, the cell
width and the `variant` key.

## Keys (when `ctx.active(e)`; never in the audience window)

| key | does |
|---|---|
| G | Flow view on / off |
| Shift+G | play / pause the stages (opens the view when it is off) |
| ← / → | Flow view: previous / next stage (Explain catches them first while it runs) |

## Pure exports (tests: `tests/nn_flow.test.mjs`)

```js
cleanFlow(v) -> flow | null
attendHeads(x, spec, off = Set) -> { heads: [{ Q, K, V, S, A, Z, ... }], out }   // model.attend (the variants'
                                                                            //   fields too), heads in off left out of out
propagate(net, a, from = 1, { off }) -> { a, z, attn }   // the forward pass again from layer `from`
ablate(net, fwd, off) -> { a, z, attn, from }            // fwd with those heads knocked out (from = -1: unchanged)
buildFlow(net, { fwd, off, every }) -> { stages, tiles, nodeCell, words, labels, numbered, next, lm, every,
                                         residual, ffn, dims, dimsText, attention, why, max: { act, bias },
                                         heads, T, norms, variant }
flowKey(F) -> string                                     // the structure key
texHtml(tex) -> html                                     // W_{out} -> W<sub>out</sub>, for tips and titles
variantOf(net) -> { key, axis, short, note } | null      // the tiny language model's family, by meta.title
variantMenu() -> [[axis, [{ key, short }]]]              // the picker's list, in menu order
```

A tile is `{ id, l, tex, kind: 'mat' | 'bars' | 'dist', rows, cols, v, pos, rowLab, rowNo, colLab,
head, off, mask, scale, onehot, target, dim, dimRows, note, node, src, tip }`: `pos` maps its rows to
positions (the last-position tiles have `[4]`), `rowNo` numbers them, `dim` is its bracket
(`{ kind: 'model' | 'ffn' | 'vocab', html }`) and `dimRows` the rows that fade. `scale` is `'act'`,
`'attn'`, `'prob'` or `'bias'` (ALiBi's B); a K or V tile of a head that shares it has `same` (the
name of the one it equals), and a linear attention's scores `maskBlank` (masked cells are left out,
not −∞).

The tests check that `attendHeads` gives `fwd.attn` exactly and `propagate` reproduces
`model.forward` from any layer on eight presets; that knocking head h out equals zeroing the rows
of W_O that read its columns of Z; the tiny language model's 14 stages, tile shapes, masks and
words; that every tile holds the forward pass and every sum adds up (X = O W_E + P, H = X + Z W_O
+ b_O, Y = H + F W_2 + b_2); that on every preset (with `every`) each neuron is drawn once, where
its value is, and every source is a drawn cell; the traces one step back; the ending (only position
5 drawn by default, Y's other rows faded, the chart's context and true next word, every position's
rows, targets and note); the row labels and numbered keys; the widths (the brackets, `dims` and the
caption); the knock-out tiles; that the structure key ignores new values but not `every`; the
transformer block (not a language model: rows by token, d_model 2, d_ff 4); and the fallback
without attention. `tests/nn_model.test.mjs` checks `nl_lm` (one-hot rows, the 20 sentences of the
grammar and that they all come up, that on, in and over each end three ways, decode, noise),
`nl_next`, the preset's architecture, headline sentence and Reset, the gradients against finite
differences (every shared matrix entry is summed from its edges; every shared bias, position bias
and a sample of the shared entries, inputs, edges and biases against finite differences, with
cross-entropy over the vocabulary at each position), that training puts more than 0.95 on mat after
"the cat sat on the" (and more than 0.8 on every sentence's last word, with the open positions
still spread), and Adapt.

For the variants, `tests/nn_model.test.mjs` checks the new activations' values and their backward
pass against finite differences (as hidden and output layers, and per token), the attention
variants against a formula-by-formula reference (window, RoPE, ALiBi, linear, and together, with
`attend` equal to `forward`) and every one of their gradients against finite differences, that RoPE's
scores depend only on the distance, the new fields in validate, normalize and setLayer,
`node.fixed`, the transposed tie, each variant's architecture, Reset and a sampled finite-difference
check of its whole backward pass (every untied and shared bias, and a sample of shared entries,
edges and inputs), and what training shows (four variants learn as tiny_lm does; the window's last
position gives every subject exactly the same guess while two windows recover it; no mask falls
below a loss of 0.05). `tests/nn_flow.test.mjs` checks `variantOf` and `variantMenu`, that
`attendHeads` and `propagate` give `forward` exactly on every variant, the knock-out equivalence on
five of them, and each variant's stages and tiles (the turned Q and K, S = QKᵀ/√d + B, the band
mask, φ and A = S over its row sum, the shared K and V, the norm rows, F = silu(G) ⊙ U, W_Eᵀ and the
block symbols); the every-preset tests cover them too.

## Limitations

- PNG and To board picture the 2D canvas, not the Flow view.
- A knocked-out head is a view of the forward pass only: the backward pass, the matrix panel and
  training keep every head.
- The trace goes one step back; a cell's full ancestry (everything before it through attention)
  is most of the flow.
- The fallback for layers that are not tokenwise tied shows each layer as one row of numbers and
  its weights only in the tip.
- **The variants elsewhere.** Only the Flow view draws what a variant changes. The canvas, the
  matrix panel, the cards, the Attention panel and the 3D view show the variants' true numbers (S, A
  and Z come from the model, and so do the norms' values), but write attention as
  softmax(QKᵀ/√d_h), without RoPE's turn, ALiBi's B or linear attention's φ, and the canvas and the
  3D view hatch only causal cells, so a window's older keys show as zero weights instead of masked.
  The matrix panel and the cards give LayerNorm, RMSNorm and SwiGLU's backward step as the
  Jacobian product δ = Jᵀ ∂L/∂a, with its value but not its entries.
- LayerNorm and RMSNorm have no learned gain or shift. Right before a matrix (pre-norm) those would
  fold into it; in post-norm they would not, so there the residual stream stays exactly normalized
  where the first transformer's γ and β could rescale it.
- Multi-query and grouped-query attention share K and V by ties, so the Q, K, V layer still holds a
  K column per query column (the shared ones repeated): the saving they bring when generating, a
  smaller cache of keys and values, is stated in the caption, not drawn.
- The two-block variants change two things, depth and the pre-norm that lets two blocks train.
- SwiGLU keeps the FFN at 32 wide, so it has 50% more FFN weights than ReLU; LLaMA shrinks it to
  about ⅔ to match (a width of 21 here).
- **Training speed.** The tiny language model is 7.5 times the first one's size (5760 edges against
  768), and the other panels' per-frame work grows with it: the Train panel's evaluation of the
  whole dataset and its sequence plot, the matrix panel, the hidden SVG canvas and the audience
  mirror. In the headless test browser it trains about 27 steps a second at 7 frames a second
  (the first one: 206 steps at 41), so 400 steps take about 15 s there. The Flow view itself is
  about 3% of the frame. Measured again when the variants came in, on a machine busy with other
  work (single runs ranged from 5 to 34 steps a second), interleaving the code before and after:
  over 23 runs each, a median of 13.9 steps a second before and 15.0 after, the best 31.8 and
  34.3. In node, best of 24, a batch-10 step costs 1.18 to 1.25 ms either way, the Train panel's
  pass over the 300 sentences 5.1 ms and the Flow view's `buildFlow` 4.5 to 5.5 ms. The one-block
  variants cost the same per step as tiny_lm (1.14 to 1.22 ms) except SwiGLU (1.42 ms, its second
  matrix up); the two-block ones twice (2.05 ms).

## Testing

`node --test "tests/*.test.mjs"`. In the browser (docs/FEATURE_GUIDE.md's recipe, with
`--use-angle=swiftshader --enable-unsafe-swiftshader`): `index.html#nn=tiny_lm` opens straight
into the Flow view; `mathboardNet.ctx.train.play()` / `pause()` train it, `ctx.flow.step(1)`,
`ctx.flow.every(true)`, `ctx.flow.head(1)` and `ctx.flow.info()` drive and read it, and the cells are
`.nnf-c[data-t="<tile>"][data-i="<row>"][data-j="<col>"]` (bars `.nnf-b`) to point at. For
`docs/media/net-flow.png` the preset was trained through the Train panel (1000 steps of batch 10 at
lr 0.2, speed 5), the Train panel folded (`ctx.train.fold(true)`), and the pointer put on A₂[5,2].
A variant opens the same way (`#nn=tiny_lm_rope`), or from the open view with
`ctx.flow.variant('tiny_lm_rope')` or the picker (`select.nnf-varsel`).
