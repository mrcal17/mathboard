# Net tab: the 3D net view (contract)

This extends docs/NN_CONTRACT.md, NN_ATTENTION.md and NN_LENS.md. One module, `static/nn/view3d.js`
(with `view3d.css`), draws the current net in 3D with Three.js. While it is on, a WebGL stage covers
`#nn-stage` and replaces the SVG canvas; the matrix panel, the cards, the Train, Attention and 3D
plots panels, the lens bar and Explain keep working over it and stay linked through the store. It
adds one store key, `v3d`, which the audience window mirrors. Three.js (and its OrbitControls and
CSS2DRenderer addons) is imported the first time the view opens, so the module costs nothing while
it is off.

Toggle it with the toolbar's **3D** button (group `view`) or **D**. Shift+D steps through the three
views; the bar at the bottom right switches them too.

## The three views

### Stack: any net, layers in depth

- **Layout.** One sheet per layer along x. A sheet stands in a y-z plane: features run down (y), as
  in the canvas's columns, with groups (Q, K, V) stacked and a gap between them; tokens go into depth
  (z, token 1 in front). So a tokenwise (tied) layer stays in its token's plane and only attention
  edges cross between planes. A dense layer is a column up to 4 neurons, and past that a grid with
  its columns in depth (row-major: neuron k is row ⌊k / c⌋, column k mod c).
- **Neurons** are spheres filled like the canvas: `colorFor(a, max)` over the node colour, on the
  shared activations scale (every entry of `fwd.a` and `fwd.z`). **Weight edges** are cylinders
  coloured by `colorFor(w, maxW)` (the alpha blended onto the background) and 0.013 + 0.05·|w|/maxW
  thick; skip edges arc over the sheets they skip (the headers move up to clear them). Fixed edges
  (a residual, a pooling weight) are not parameters: thin, dashed and in the muted text colour
  (--text-3) at 45%, as on the canvas, at full strength while hovered, selected or lit.
  **Attention edges** run V_j,f → Z_i,f in the canvas's violet, as thick and as opaque as A_ij;
  causally masked pairs have none. Each attention sheet has an n × n tile of A per head in front of
  it, as bars as tall as A_ij, masked cells flat.
- **Labels** (CSS2D): text with one halo, as on the canvas. A header per layer (name, and shape or
  activation · count), plain text, with a pill and a HI ring while the layer is hovered or selected
  and a neutral ring for the lens's focus (neighbours that would touch on screen drop the second
  line, then every other one moves up); Q / K / V beside their
  groups, token names (or t₁…) under the first token layer, an A beside each tile. The 1.2 button
  puts values under the neurons (off by default in this view). The hovered, else the selected,
  neuron shows its label and value in a tip; the shown row of A gets its A_ij on the edges.
- **Hover, selection, step-through, lens**, by the canvas's rules (view.js `resolve`,
  `resolveAnim`, `attFoci`): the selected neuron gets a thick ring, the hovered one a thin ring, the
  step-through's a pulsing one and pulses along its lit edges (reversed for backward). While
  something is hovered or stepped, unrelated neurons fade to 0.42, edges to 0.13 and attention edges
  to 0.1. The lens (focus.js `emphasis`) sets opacity 0.1 + 0.9·v, drops numbers below v = 0.25 and
  hides what `show` and the thresholds remove; a followed token shows its row of A on the tile.
- **Input.** Moving over a neuron hovers it (`{ kind: 'node' }`), over an edge the edge, over an
  attention edge or an A cell its row (`{ kind: 'token', layer, t, h? }`). A click selects the
  neuron or edge (so the inspector opens its card), an A cell selects the attention layer, empty
  space clears the selection. Drag orbits, right-drag or Shift-drag pans, the wheel zooms.

### Heads: one attention layer, a slab per head

Cells are cubes in x-y planes, rows = tokens, coloured like the matrix panel (value colour on the
front face, the other faces shaded). From the top:

1. **Q | K | V** as whole T × d matrices from `fwd.a` of the Q, K, V layer. Each head's column chunk
   sits on a plate in that head's colour (the attention panel's: violet, green, amber, then more).
2. **split by columns**: a line per head and matrix runs from its chunk to its slab.
3. **One slab per head**, each a plate in the head's colour, stepping down and back in depth:
   `Q_h K_h → A_h · V_h = Z_h`, with Q_h, K_h, V_h, A_h and Z_h the real `fwd.attn[l].heads[h]`
   numbers. A_h is a T × T tile of bars as tall as A_ij (violet; masked cells flat).
4. **concat**: a line per head from Z_h to its chunk of `Z = [Z_1 … Z_h]` (the attention layer's
   `fwd.a`), right of the slabs; then, when the next layer is a tied tokenwise matrix from this one
   (`projOf`: the transformer's W_O), `× W_O = Z W_O` (the projection alone: the residual and b_O of
   H = X + Z W_O + b_O are left out).

The caption in the bar gives `Z_h = softmax(Q_h K_hᵀ/√d_h) V_h` and `Z = [Z_1 … Z_h] → Z W_O` for
this layer. The **heads** chips set the attention layer's `heads` (any divisor of d; one
`store.commit`, so Ctrl+Z undoes it). With 2+ attention layers a menu picks one.

- **Hover in**: a cell of Q, K or V hovers that token of the Q, K, V layer (`g`, and `h` with 2+
  heads); Z_h hovers the attention layer's token t (with `h`); a bar hovers its query row; a whole
  matrix's cell hovers its neuron; a W_O cell hovers one of its tied edges (the canvas lights the tie
  group). A click selects the neuron (the whole-matrix cell stands for it), W_O's edge, or for a bar
  the attention layer.
- **Hover out**: a token of the attention layer lights row t of Q_h, A_h, Z_h, Z and Z W_O (one head
  or all); a query token of the Q, K, V layer lights its rows and its row of A, a key or value
  token its rows and its column of A; a neuron lights its cells; a tied W_O edge its cell. Lit cells
  get a frame in the hover colour, the rest fades to 0.42.
- **Step-through** on the attention layer: `scores` lights q_i and the keys, `softmax` row i of A,
  `sum` row i of A with V and z_i; backward all of them. On another layer, the neuron's cells.
- **Lens**: E.node for cells that stand for a neuron, E.attn for bars, E.edge for W_O; a kept head
  fades the other heads' plates and lines.
- **Numbers** (the 1.2 button, off by default here): on a lit or hovered cell or bar always; with
  the button on, on every cell and bar the lens keeps, at full size once its face is at least 28 px
  wide on screen (`NUM_PX`, measured as the camera moves) and below that without the leading 0 and
  shrunk to fit the face (7.5 px at the least), so turning them on always shows them.

### Tensor: the reshape, as moving cubes

The step people get wrong implementing multi-head attention. Q and K sit side by side on top, V
below; every cube is one number and moves to where the step puts it (1.1 s, staggered by head when
the shape changes; cubes that move between width and depth rise a little on the way).

| step | form | code (as written in the caption) | what moves |
|---|---|---|---|
| 1 | `[T, d]` | `Q = X @ W_Q` | rows are tokens |
| 2 | `[T, h, d/h]` | `Q = Q.view(T, h, d_h)` | each row's features split into h chunks (gaps appear); same memory order |
| 3 | `[h, T, d/h]` | `Q = Q.transpose(0, 1)` | chunk h of every token slides into slab h, in depth |
| 4 | `[h, T, d/h]` | `Z = softmax(Q @ K.mT * s) @ V` | Q and K are used up: A_h appears as bars between them; V's cubes turn into Z_h |
| 5 | `[T, h, d/h]` | `Z = Z.transpose(0, 1)` | the slabs come back beside each other |
| 6 | `[T, d]` | `Z = Z.reshape(T, d)` | the gaps close: concat(Z_1, …, Z_h) |
| 7 | `[T, d]` | `out = Z @ W_O` | Z moves up, W_O appears beside it, Z W_O below |

Step 7 is there when the source has a d × d W_O. **the bug** replaces steps 2–6 with
`Q.view(h, T, d_h)` (no transpose: slab k is the k-th run of T·d/h numbers in memory order), the same
attention on those slabs, and `Z.view(T, d)`: the cubes go back where they started with the wrong
numbers, and the caption says by how much. Turning the bug on colours by tokens, which shows at a
glance that a buggy "head" holds pieces of one or two tokens rather than a chunk of each.

- **Source**: `net` (the attention layer's Q, K, V from `fwd`, its scale, mask and heads; the caption
  confirms that step 6 matches the layer's Z) or the fixed **4×6 example** (T = 4, d = 6, seeded
  numbers, h = 1, 2, 3 or 6, scale 1/√(d/h), and a random W_O), which works for any net, attention
  or not. The default is `net` when the net has an attention layer with 2+ heads.
- **Colour**: `values` (colorFor over the node colour, one scale for Q, K, V, Z and Z W_O; W_O on its
  own) or `tokens` (the token each number came from; numbers hidden). Shape labels under each
  block, head labels over each chunk or slab.
- **Numbers** (the 1.2 button, on by default here, `values` only): only on the step's matrices
  (the cubes it shows, not those shrinking away): on every one of them, sized as in heads (full size
  from a 28 px face, `NUM_PX`, and shrunk to the face below that). With the button off, only on a
  lit cube. The cube under the mouse (either source) is framed and shows its number.
- **Playback**: the previous / next arrows and the step chips, ← → while the view shows, play runs
  one step every 2.6 s (presenter only; the audience follows `step`).
- **Hover** (net source): hovering a cube hovers its neuron (Q, K, V, or Z from step 4); a hovered
  or selected neuron frames its cube.

## `state.v3d` (owner: view3d.js)

```js
v3d = null | {
  mode: 'stack' | 'heads' | 'tensor',
  camera: null | { p: [x, y, z], t: [x, y, z], k },   // position, orbit target, distance / fit distance;
                                                       //   null = the view's own framing
  layer: null | layerId,   // heads, tensor: the attention layer (null or gone: the first working one)
  step: int >= 0,          // tensor: the step shown (clamped to the steps there are)
  src: null | 'net' | 'example',   // tensor: null = net when it has 2+ heads, else the example
  h: 1 | 2 | 3 | 6,        // tensor example: heads (divisors of EXAMPLE.d = 6), default 3
  bug: bool,               // tensor: the no-transpose bug
  color: 'value' | 'token',// tensor
  nums: bool,              // numbers on neurons and cells (on by default in tensor only)
}
```

- It starts null (off). Set it with `store.set('v3d', …)`; `cleanV3d(v)` completes and repairs any
  object (null stays null). Switching the view resets `camera` (null), `step` and `nums`.
- **Camera.** The presenter publishes its camera at most every 120 ms while it moves and once when
  it settles. `k` is the camera's distance over the fit distance for its direction in this window's
  free area, so an audience window of another size frames the same view: it keeps the direction and
  target and scales the distance by its own fit, then eases toward it. A camera set by another
  module is flown to (350 ms).
- **Mirror.** nn.js carries `v3d` in `mirrorState()`, fires `onMirror` on its event and `applyMirror`
  sets it; the audience window only renders (no orbiting, hover, clicks or bar buttons).
- **UI state** (not in the store): localStorage `mathboard.nn.view3d` = `{ mode }`, the view D opens
  in. The audience never writes it.

## While the view is on

- `ctx.view.nodeRect(id)` returns the projected rect of the neuron's sphere or cube (in heads mode
  the whole-matrix cell, in tensor mode its cube when shown; null for neurons the view doesn't
  draw), `ctx.view.contentRect()` the projected box of the whole 3D content, and `ctx.view.fit(ms)`
  refits the 2D canvas behind and the 3D camera (F). The originals come back when the view closes.
  The SVG is hidden (`visibility`), not removed; the 2D view keeps its pan and zoom.
- **Framing** follows view.js's fit: the largest free rectangle of the stage clear of the Train,
  Attention and other floating panels and of this bar, above the lens bar; the orbit target sits at
  its centre (a camera view offset) and the distance fits every drawn point. It refits when a panel
  opens, folds, moves or closes, unless the user has orbited, panned or zoomed since.
- The bar sits at the bottom right (a strip along the bottom in heads and tensor views, with the
  caption), lifted above the lens bar when that one reaches under it. The clean view (H) and the
  audience keep only the caption.

`ctx.view3d` (test and console handle) = `{ MODES, on, mode, ready, toggle(on?), open(mode?),
setMode(mode), cycle(dir), step(±1), fit(ms), info() }`; `info()` reports the mode, the camera, the
tensor step and counters of builds and paints.

## Keys (when `ctx.active(e)`; never in the audience window)

| key | does |
|---|---|
| D | 3D view on / off |
| Shift+D | next view: stack, heads (with an attention layer), tensor |
| ← / → | tensor view: previous / next step (Explain catches them first while it runs) |

F (fit), H (clean view), the lens keys, S (step-through), Space / T (training) and the rest work as
before.

## Rendering

- Instanced meshes throughout (spheres, cubes, bars, cylinder edges, plates, rings, pulses), with
  the shading baked into vertex colours on unlit materials, so a value's colour is exact on the
  face turned to the camera, in both themes. Dimming blends toward the page background instead of
  using transparency.
- A structural change, a new layer shape (tokens, groups, heads, causal, names) or another view or
  source rebuilds; everything else (training at full speed, hover, lens, theme) repaints colours and
  matrices in place. The loop renders only when something changed or moves, and stops while the
  Net tab is hidden. Turning the view off disposes the content, geometries, materials, the
  renderer (its WebGL context is released) and the DOM.

## Pure exports (tests: `tests/nn_view3d.test.mjs`)

```js
MODES, EXAMPLE = { T: 4, d: 6, seed }, divisors(n), cleanV3d(v)
slotOf(form, t, f, { T, d, h }) -> { head, row, col }   // form 'TD' | 'THD' | 'HTD' | 'BUG'
toHeads(X, form, h) -> h × T × dh,  fromHeads(H, form, T, d) -> T × d
attend(Qh, Kh, Vh, { scale, causal }) -> [{ S, A, Z }] per head,  matmul(A, B)
tensorSteps({ bug, proj }) -> [{ key, form, show }]
tensorStory({ Q, K, V, h, scale, causal, WO, bug }) -> { steps, Qh, Kh, Vh, heads, Z, out, ... }
exampleQKV() -> { Q, K, V, WO },  projOf(net, l) -> W_O into l + 1 or null,  attnLayers(net)
```

The tests check that the correct story reproduces `fwd.attn` and the attention layer's Z on the
attention presets, that the bug changes Z whenever h > 1, and the slot rules.

## Limitations

- PNG and To board picture the 2D canvas, not the 3D view. Train's per-neuron maps are 2D only.
- The inspector re-places its cards on pointer and wheel input in the tab. When the camera moves by
  itself (a fit, the audience following), view3d.js sends the tab a no-op wheel event so the cards
  follow; a `ctx.inspector` hook for this would be cleaner.
- Editing (dragging neurons, wiring, double-click) stays on the 2D canvas; the only edit here is
  heads mode's heads chips.

## Testing

`node --test "tests/*.test.mjs"`. In the browser (docs/FEATURE_GUIDE.md's recipe, with
`--use-angle=swiftshader --enable-unsafe-swiftshader`): `mathboardNet.ctx.view3d.setMode('heads')`,
`store.set('v3d', { ...store.state.v3d, bug: true, step: 1 })`, then `ctx.view.nodeRect(id)` for
a point to move the mouse to. For `docs/media/net-3d-heads.png` the multihead preset was trained
(speed 100, about 5000 steps), then `loadSample(0)`, the Train panel folded, the lens bar closed
(L), heads view, and `store.set('hover', { kind: 'token', layer: 2, t: 0 })`.
`net-3d-reshape.png` puts two crops of the stage side by side: the 4×6 example with h = 3 in token
colours at step 3, and the bug at its step 2.
