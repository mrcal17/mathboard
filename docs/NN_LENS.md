# Net tab: lens, attention panel and Explain (contract)

This extends docs/NN_CONTRACT.md and docs/NN_ATTENTION.md. It adds three store state keys,
each carried to the audience window:

- `lens` (owner: lens.js): filtering and focus.
- `viz` (owner: attnviz.js): the attention panel.
- `tour` (owner: tour.js): Explain, the guided walkthrough.

It also adds one net field, `meta.tokenNames`, and four modules: `focus.js` (pure rules, no
install), `lens.js`, `attnviz.js` and `tour.js`. The shell installs `view`, `inspector`,
`matrix`, `train`, `lens`, `attnviz`, `tour` in that order. Every module that uses focus.js
(view, inspector, matrix, train, lens, attnviz, tour) imports it statically, with no fallback.
Toolbar buttons: **Lens** (group `view`), **Attention** (group `attnviz`) and **Explain** (group
`tour`); the `?` cheat sheet lists them and their keys.

## Token names

`net.meta.tokenNames: (string | number)[] | undefined` holds one plain-text name per token (slot
i = token i, 0-based), used for that token in every token layer, e.g. `['the', 'cat', 'sat']` in
the `words` preset. `normalize` keeps it, as it keeps any unknown meta field.

- **One source.** focus.js's `tokenNames(net)` returns one entry per slot: the trimmed name, or
  `null` for an unnamed slot (`[]` when the field is not an array). A slot is named when it holds
  a string or a finite number whose trimmed text is not empty and is not that slot's own default
  `t{i+1}` (1-based: `'t2'` in slot 1 is unnamed, `'t2'` in slot 0 is a name).
  `tokenLabel(net, t)` is the name, else `t{t+1}`.
- **Every reader goes through these two**: the canvas token boxes (cut to 8 characters, the full
  name in the tooltip), the matrix panel's headers and trace card, the inspector cards, the Train
  plot's row labels, the attention panel, the lens bar's token chips and the Explain captions.
  In KaTeX a name is `\text{…}`; an unnamed token stays `t_{i}`.
- **Renaming** happens in the attention panel only. `ctx.attnviz.rename(i, name)` collapses
  whitespace, trims, keeps at most 16 characters, and treats an empty name as unnamed. It fills
  the unnamed slots with their defaults `t{k+1}`, so the stored list never has a blank before a
  named slot, and deletes `meta.tokenNames` when no slot is named. Each rename is one
  `store.commit`, so undo, export and the audience carry it.

## `state.lens` (owner: lens.js)

```js
lens = {
  focus: null | { layer: layerId, part?: string },  // part: one of the layer's groups ('Q', 'K', 'V'),
                                                    //   or 'scores' | 'softmax' | 'mix' on an attention layer
  token: null | int,     // follow one token (0-based)
  head:  null | int,     // keep one head (0-based) of every attention layer and of the Q, K, V layer it reads
  show:  { weights: true, attention: true, fixed: true },   // edge types drawn
  minW: 0,               // hide non-fixed edges with |w| < minW
  minA: 0,               // hide attention edges with A_ij < minA (0..1)
}
```

- `state.lens` starts null, which reads as the default. Set it with `store.set('lens', next)`,
  always a complete object. `lens.js` exports `DEFAULT_LENS` (frozen, from focus.js) and
  `lensOf(store)`: the current lens, cleaned for `store.net`, as a fresh copy you may change and set.
- **Always valid.** lens.js runs `cleanLens` after every `net` event and on every lens another
  module sets (attention panel, matrix panel, Explain). A structural change that also changes
  `meta.title` (a preset, an import) resets it to the default. In the audience window lens.js
  returns from `install` at once: that window shows the presenter's lens as mirrored.
- **Dimmed vs hidden.** Outside the emphasis is dimmed; removed by `show` or a threshold is hidden.
  Hover, selection and the step-through still light what they point at.

| | dimmed (emphasis v, 0..1) | hidden |
|---|---|---|
| canvas | opacity 0.1 + 0.9·v; numbers dropped below v = 0.25 | not drawn |
| matrix panel | opacity 0.16 + 0.84·v | W and A cells under minW / minA dimmed; `show` alone changes nothing |
| cards | opacity 0.3 + 0.7·v (never the card's own target) | opacity 0.16 |
| attention panel (arcs) | line opacity × 0.13 | no line |

## `focus.js` (pure, no DOM; tests: `tests/nn_focus.test.mjs`)

```js
DEFAULT_LENS                       // frozen default lens
ATTN_PARTS                         // ['scores', 'softmax', 'mix']: an attention layer's parts, in the matrix panel's order
copyLens(lens) -> lens             // a complete, independent copy (missing fields defaulted; not validated)
tokenNames(net) -> (string | null)[],  tokenLabel(net, t) -> string   // see Token names
stages(net) -> [{ layer, part? }]  // what [ / ] and Explain walk: every layer after the input, an attention layer
                                   //   as its three parts (Q, K, V parts are not stages)
lensInfo(net) -> { tokens, heads, names, layers, stages, hasWeights, hasFixed, hasAttention, maxW }
                 // tokens: the most tokens of any token layer (0: none); heads: the most heads of any attention
                 //   layer (1: nothing to pick); names: tokenNames(net); layers: every layer, the input included,
                 //   as { id, index, name, kind: 'dense' | 'qkv' | 'attention', parts } (parts: its group names,
                 //   ATTN_PARTS, or []); maxW: the largest |w| of a non-fixed edge
cleanLens(net, lens) -> lens       // the same object when nothing needs fixing; null or junk -> DEFAULT_LENS
stepStage(net, focus, dir) -> focus | null
emphasis(net, fwd, lens) -> E
```

- **cleanLens** drops a focus on a missing layer (a layer index becomes its id), a part the layer
  doesn't have, a token outside `lensInfo(net).tokens`, and a head outside the heads or in a net
  without multi-head attention. `minW` below 0 becomes 0; `minA` is clamped to [0, 1]. So `token`
  is always null in a net without token layers.
- **stepStage**: from no focus, the first stage (dir 1) or the last (dir -1); past either end,
  null. A focus that is not a stage steps from where it sits: a Q, K or V part as just after its
  layer, a whole attention layer as just before its `scores`, the input layer as just before the
  first stage (so `]` gives the first stage and `[` gives null).

```js
E = emphasis(net, fwd, lens) = {
  any: bool,                     // something is dimmed (false: the lens emphasizes everything)
  hides: bool,                   // show or a threshold is off its default (something may be hidden)
  lens,                          // the cleaned lens E answers for
  node(id) -> 0..1,              // 1 = emphasized, 0 = dimmed; in between for attention-weighted keys and values
  edge(id) -> 0..1,
  attn(l, i, j, h) -> 0 | 1,     // attention edge V token j -> Z token i, head h, in attention layer l
  layer(l) -> 0..1,              // the most any of its neurons gets (an empty layer: 1 when the focus keeps it)
  hidden: { edge(id) -> bool, attn(l, i, j, h) -> bool },
  rows(l) -> null | Set<int>,    // token rows to emphasize in l's matrices; null = all, empty = the layer is dimmed
  heads(l) -> null | Set<int>,   // heads kept in l (an attention layer or its Q, K, V layer)
  groups(l) -> null | Set<int>,  // group indices a focus part keeps in l; empty = the layer is dimmed
}
```

Layer arguments are indices or ids; an unknown layer gives 1 or null. With no focus, token or
head: `any` is false, every weight 1 and `rows` / `heads` / `groups` null.

**Rules**
- focus, token and head intersect: a neuron gets the minimum of the three; an edge the minimum
  over the focus and both its ends.
- **focus.layer = L** (not attention) emphasizes L, every layer with an edge into L, and the edges
  into L; other weight edges dim. A `part` (a group) keeps only that group of L, the edges into it
  and the layers feeding it. A focus on the input layer keeps only the input neurons.
- **focus on an attention layer** emphasizes it (Z) and its Q, K, V layer and dims every weight
  edge. `scores` and `softmax` keep Q and K and dim the attention edges; `mix` keeps V and the
  attention edges; no part keeps all of them.
- **token = t** emphasizes token t in every token layer with more than t tokens. In a Q, K, V
  layer, key and value token j get A_tj (of their feature's head; 1 before a forward pass), so
  what t attends to stays visible. In an attention layer only the edges into Z token t stay.
  Plain layers are unaffected.
- **head = h** keeps features `[h·dh, (h+1)·dh)` of each Q, K, V group and of Z, and head h's
  attention edges.
- **hidden**: `show.fixed` off hides fixed edges, `show.weights` off every other weight edge,
  and `minW` non-fixed edges with |w| < minW. `show.attention` off hides every attention edge and
  `minA` those with A_ij < minA. `hidden.attn` is also true for causally masked pairs (j > i),
  whatever the lens.

## The lens bar (lens.js)

The bar floats at the bottom left of `#nn-stage` and shows only the controls that apply:
- **Focus** menu: Whole net, then every layer, the input included, each followed by its parts
  (`Q`, `K`, `V`; `1 scores`, `2 softmax`, `3 mix`). `[` and `]` skip the input.
- **Token** (2+ tokens): `all`, then a chip per token (its name cut to 10 characters, or t_n).
- **Head** (2+ heads): `all`, 1, 2, ...
- **Edges**: a show / hide toggle per edge type when the net has two or more (W, A, fixed), and
  threshold sliders: |w| from 0 to the largest |w| rounded up to 0.1, A from 0 to 1. A slider is
  disabled while its edge type is hidden.
- **Clear**: the default lens.

Clicking the active token or head chip turns it off. The toolbar's **Lens** button shows or hides
the bar and carries a dot while the lens is not the default. The bar is hidden in the clean view
(H) and never built in the audience window; while it is not showing, its keys toast what they
did. UI state: localStorage `mathboard.nn.lens` = `{ open }`.

## How the other modules draw the lens

- **Canvas (view.js)** dims and hides as in the table above and outlines the focused layer's
  header and band. When nothing is hovered or stepped, a followed token shows its row of A on the
  attention edges (only head h's when `head` is set), as hovering it would.
- **Matrix panel (matrix.js)** applies the lens in place, without a rebuild. focus folds every
  other layer to a one-line summary (click a summary to focus that layer), scrolls the focused
  layer to the top, marks its part and fades its other parts; the step-through's layer and layers
  opened by `reveal` stay open, and a focus on the input opens every layer that reads it. token
  outlines row t in every token matrix, dims the other rows and opens the trace card. head hides
  the other heads' blocks and trace lines.
- **Token trace card** (top of the matrix panel): shown while `lens.token` is set and some token
  layer has that token. It reads "Following token n" (plus “name” when named) and holds token t's
  live row through every token layer: X, then Q, K, V, per head S and A, Z, and the later layers
  up to Ŷ. Hovering an entry hovers its neuron or token; clicking a line reveals that matrix
  (`reveal(layer, part, { token })`); × stops following (not in the audience).
- **Cards (inspector.js)** fade weight rows, sliders, attention bars and heatmap cells by the same
  emphasis.

`ctx.matrix` has two more handles, for tour.js:

```js
ctx.matrix.reveal(layer, part = null, { token = null, flash = true, smooth = true }) -> bool
    // layer: id or index. Opens it if the focus folds it (until the focus moves), scrolls it, or its part
    //   ('Q' | 'K' | 'V' | 'scores' | 'softmax' | 'mix'), to the top and flashes it. With a token and a part
    //   it centres that token's outlined row in the part instead. Layer 0 reveals the first layer that
    //   reads the input. false when the layer has no section.
ctx.matrix.parts(layer) -> string[]   // the parts that layer's section shows, in order, e.g. ['Q', 'K', 'V'] or ATTN_PARTS
```

## `state.viz` (owner: attnviz.js)

```js
viz = null | { mode: 'arcs' | 'dots' | 'mix' | 'heat', layer: attentionLayerId }
```

- The attention panel floats over `#nn-stage` and is open exactly when `viz` is set, so Explain
  and the mirror drive it through the store alone. After an edit attnviz points `layer` at a
  working attention layer (the first one when its own broke).
- Its query token and head are `lens.token` and `lens.head`; picking either in the panel sets the
  lens, so the canvas and the matrix panel follow. Dots and mix need one query and one head: with
  none followed they show the query that attends most decisively (the row holding the largest
  A_ij, rows that see a single key skipped) and head 1, on a dashed chip.

```js
ctx.attnviz = {          // also the panel element's .nnAttn
  MODES,                 // ['arcs', 'dots', 'mix', 'heat']
  open, mode, layer,     // getters: panel open; its mode (null when closed); the attention layer it shows or would show
  layers() -> id[],      // the working attention layers
  show(mode?, { layer }?) -> bool,   // mode defaults to the current one, else the last used; false in the audience,
                                     //   or with no attention layer (it toasts)
  hide(), toggle(mode?), cycle(dir = 1),   // cycle opens the panel in the last used mode when closed
  send3d() -> bool, rename(i, name) -> bool,   // rename: true when it committed
  geometry(),            // what the last drawing placed where (for tests)
  el,                    // the panel
}
```

| mode | shows |
|---|---|
| arcs | who reads whom: queries left, keys right, a line per pair as thick as A_ij, a colour per head (up to three heads together; with more, head 1 unless one is picked). Masked pairs and pairs the lens hides get no line; pairs outside its emphasis fade. Hover a token to see its lines alone with their A_ij; click it to follow it. |
| dots | why: q_i and every k_j as arrows from the origin (a number line when d_h = 1, a chosen pair of dimensions when d_h > 2), q's line with each key's shadow on it, and a table of q·k, s = q·k × scale and A = softmax(s) as bars. Masked keys are dashed. Holds the scale slider. |
| mix | what comes out: the v_j, their shaded convex hull, the A_ij v_j tip-to-tail and the resulting z_i (a staircase on a number line when d_h = 1). Holds Send to 3D. |
| heat | the numbers: S and A of each head as heatmaps (with numbers when the cells are big enough), masked cells hatched, the followed query's row ringed. Hover a cell for what it means; click a row to follow that query. |

- **Header**: mode buttons, a layer menu (2+ attention layers) and ×. Drag it to move the panel,
  double-click it to put the panel back; the corner grip sets the width. Under the drawing: the
  mode's formula in KaTeX with live numbers, and a one-line note.
- **Picks row**: query chips (with `all` in arcs and heat) and, with 2+ heads, head chips in
  their colours.
- **Scale slider** (dots; the temperature control): sets the layer's `scale` on a log scale from
  2⁻⁴ to 2⁴ times the default 1/√d_h. Larger sharpens A, smaller flattens it. Dragging calls
  `store.touch()`, releasing commits; within 0.03 (in log₂) of the default it stores null (the
  default). The 1/√d button resets it.
- **Send to 3D** (mix) writes the shown query's construction (the picked head, else head 1) into
  the 3D tab: a row `v{j} = (…)` per value, a 0..1 slider `A{i}{j}` (`A{i}_{j}` from 10 tokens) per
  key it may see, the terms `A{i}{j} v{j} @ …` tip-to-tail, and `z{i} = …`. Rows from an earlier
  send are replaced. d_h = 1 sends (v, 0); d_h > 3 sends the plane that is drawn. It then switches
  to the 3D view, sets the extent and, unless d_h = 3, looks from the top.
- **Renaming**: click the followed token's query chip again, or double-click any query chip;
  Enter keeps, Esc cancels, empty unnames (rules under Token names).
- **Hover** goes through the store as `{ kind: 'token' }` (a query: the attention layer; a key or
  value: the Q, K, V layer with g = 1 or 2; a heatmap cell: its query row and head), so the canvas
  lights it. A token hovered on the canvas or in the matrix panel isolates its lines here.
- **UI state** (not in the store): localStorage `mathboard.nn.attnviz` = `{ x, y, w, mode }`: the
  dragged spot (null = top left), the width (default 400, at least 300) and the last used mode,
  which A and M open with. The audience window reads the presenter's through `storage` events. Its
  panel is read-only: only the active mode button, no ×, grip, Send to 3D or reset, and the scale
  slider disabled. The clean view (H) hides ×, the grip, Send to 3D and reset.

## `state.tour` (owner: tour.js)

```js
tour = null | { i, n, title, text }   // title and text: plain text with $…$ KaTeX
```

- The caption card renders `tour` alone, so an audience window only needs the state. It sits along
  the bottom or the top edge of the stage where it hides the least, trying three widths. Costs:
  each neuron it covers (4 on the focused layer and the one before it, else 1), each layer header
  (2, the focused layer's 8), each floating panel by the share covered (the attention panel 12, a
  folded Train panel 1, others 6), the attention panel's header 3 more, then the net's box. In the clean view and the audience its
  buttons are hidden.
- `buildSteps(net, M, fwd?, { stages }?) -> step[]` is a pure named export (`stages` defaults to
  focus.js's). A step is `{ key, title(env), text(env), lens: { focus?, token?, head? }, viz?,
  anim?, reveal?: { layer, part?, token? }, collapse? }` with `env = { net, fwd, bwd }`; captions
  quote the live numbers.
- **Stories**, in the order of `stages(net)`:
  - a token net with attention: the tokens (focus on X), Q, K and V (focus on each group), scores
    (viz dots), softmax (heat), mix (mix), each further head (heat, `lens.head`) and the heads side
    by side, one step per later layer (residual, FFN, output), then one token followed
    (`lens.token`, arcs, the step-through on its `sum` phase) and on through the rest of the block;
  - a plain net: the input (focus on layer 0), each layer, one row (the step-through on neuron 1
    of the first layer with inputs), then either "linear layers collapse" (every layer identity:
    the matrix panel's Collapse on) or "why the activations matter".
- **A step** sets `lens` to the lens from before the tour (its `show` and thresholds kept) with the
  step's focus, token and head (null when missing); `viz` to the step's (null closes the panel);
  Collapse on for a `collapse` step and off again after it; Batch off for a step with `anim`;
  `anim` to the step's (or null). It then calls `ctx.matrix.reveal(reveal.layer, reveal.part,
  { token })` and sets `tour`. Captions are re-read at most every 150 ms on `values`, so a
  training net keeps them current.
- **Train panel**: starting Explain folds an open Train panel to its header and refits the view;
  the end unfolds it, unless the user changed the fold meanwhile. The Train panel handle
  (`.nn-train`'s `nnTrain`) has `fold(on?)` (no argument toggles), `folded` and `open` for this,
  and the same object is `ctx.train`. The fold is Train's saved UI state, so the audience's panel
  follows it.
- **Ending** (Done, ×, Esc, E, or → on the last step) puts back lens, viz (if its layer still
  exists), anim, Collapse and Batch, and sets `tour` null. A structural change ends the tour
  without putting the step-through back; after a new `meta.title` the lens and viz are not put
  back either (lens.js and attnviz.js have reset them for the new net). A new story on the same
  structure (a layer's kind, act, heads, causal, tokens or groups changed) rebuilds the steps and
  stays on the same step number.

```js
ctx.tour = {             // test handle
  buildSteps() -> step[],                     // the steps for store.net and the current forward pass
  running,                                    // getter
  start(), go(±1),                            // start: a no-op in the audience; go past the last step ends the tour
  end({ anim = true, fresh = false }?),       // anim: false leaves the step-through off; fresh: lens and viz not put back
  place(),                                    // re-place the caption card
}
```

## View fit (view.js)

`ctx.view.fit(ms?)` frames the net in the largest free rectangle of `#nn-stage`: above the lens
bar (when the bar is shown at the bottom; the area always keeps at least the top half of the
stage) and clear of the Train and Attention panels (`.nn-train`, `.nn-attnviz`, with 6 px of
air), wherever they have been dragged. Of the free rectangles at least 160 px each way, those in
which the net comes out within 90% of its largest possible zoom (itself capped at the fit limit,
1.6) qualify, and the roomiest of them wins, so a folded panel's header does not push the net into
a corner. With none left it falls back to the whole stage above the lens bar. `outOfView()` tests
against the same area. The view refits by itself when a panel or the lens bar opens, folds,
resizes or closes, and when a panel drag ends (a pointerup in the stage after which a panel's rect
changed), unless the user has panned or zoomed since the last fit.

## Keys (Net tab, when `ctx.active(e)`; never in the audience window)

| owner | keys |
|---|---|
| lens | 1–9: follow token n (nets with 2+ tokens); 0: every token; `[` / `]`: previous / next stage focus; L: show / hide the lens bar |
| attnviz | A: open / close the attention panel; M / Shift+M: next / previous mode (opens the panel in its last mode when closed) |
| tour | E: start / end Explain; while a tour runs, → / PageDown next (past the last step ends it), ← / PageUp back, Esc ends it |

The existing keys (S, B, W, T, Space, F, H, Delete, Esc, ?) are unchanged. Explain's keys are
caught in the capture phase, so while a tour runs Esc ends it before the shell's Esc deselects.

## Mirror

`nn.js` carries `lens`, `viz` and `tour` in `mirrorState()`, calls the `onMirror` listeners on
their store events, and `applyMirror` sets each one that differs. The audience window only renders
them: lens.js does nothing there, the attention panel is read-only and the caption card has no
buttons. Panel layout is not in the mirror: the attention panel's spot, width and last mode, and
the Train panel's open and fold, come from the presenter's localStorage through `storage` events
(both windows share one origin).
