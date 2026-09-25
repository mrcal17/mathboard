# Board diagnosis: why the handwriting board feels finicky

Diagnosis of the current Math-pen pipeline (`static/app.js` grouping and rasterizing, `server.py` prompt and
`clean_latex`, Ollama `qwen3-vl:8b-instruct`), written 2026-09-25. No app code was changed. Outside options
(other recognizers, pen UX from other products) are in [BOARD_RESEARCH.md](BOARD_RESEARCH.md); this note is about
what the current pipeline does wrong and what to change first.

The complaints were: "My 2s end up as x or infinity a lot, things that should be bounded together aren't, there's
not a lot of degrees of freedom while writing."

**Update, same day.** The server-side parts of fixes 2, 5 and 8 are now in `server.py`: crop normalization, the
`clean_latex` repairs, prompt token counts, and a permanent replay eval (`tools/eval_board.py`). On the same 37
images, qwen3-vl went from 16/37 to 26/37, and Uni-MuMER, the new second backend, scored 29/37. See
[BOARD_BACKENDS.md](BOARD_BACKENDS.md).

## Summary

**Top causes, with evidence**

1. **Related strokes are split into separate groups, and a split group is unreadable.** The reach is fixed in
   screen pixels (16 px vertical, 56 px sideways) while this user's symbols are about 40 to 140 px tall (median
   about 80 px). A fraction's numerator, bar and denominator land in three groups. 11 of the 40 debug images are
   such fragments (three fractions in three pieces each, one matrix in two). **None of the 6 numerator and
   denominator pieces was read correctly (the 3 bars came back as NONE or junk). The same strokes stacked into one
   image were read correctly 9 out of 9 times** (`\frac{d}{dx}` twice, `\frac{\partial L}{\partial w}` once,
   each at three gap sizes).
2. **The model sees every symbol hugely magnified.** Ollama (llama.cpp under it) resizes every image up to about
   1024 visual tokens, about 1 megapixel at 32 px per token. The board sends crops of 0.02 to 0.43 MP where the
   ink fills about 80% of the frame, so each crop is upscaled 1.6x to 5.8x. At the model's input the median
   symbol is about 415 px (13 tokens) tall, with 12 to 37 px strokes. Re-rendering the same debug images so a
   symbol is about 64 px at the model's input raised accuracy from **15/37 to 23/37**. Only padding the original
   image, with no re-stroking, gave **24/37**. The noise floor is about 2 (see below). This fixes `2 + \infty`
   (it becomes `2 + 2`), `- \infty` read as NONE, `\text{d} \times` for `dx`, and the words.
3. **Partial expressions are converted while you are still writing them.** After 700 ms of pen idle, which is
   short with a mouse, a half-written group is sent. Three debug images are half-written matrices, and the replies
   dropped the entry (`\begin{bmatrix} \end{bmatrix}` for `[ 1`) or invented an `=`. An empty matrix is valid
   KaTeX, so it is shown and it hides the ink of the `1`.
4. **Grouping works on geometry you can't see.** After conversion the ink is hidden but its bounding box still
   decides grouping. The typeset is capped at 56 px font (digits about 36 px tall) against 80 to 120 px
   handwriting, and it is left-aligned in the ink box. That leaves a large invisible region that still pulls new
   strokes in.
5. **Corrections don't stick.** A LaTeX edit made in the editor is overwritten the next time a stroke joins that
   group. Re-recognize sends the same image at temperature 0, and the replay reproduced 33 of 37 replies byte for
   byte, so it rarely changes anything. There is no way to merge or split groups by hand.

The prompt and the sampling settings are not the problem. Two prompt rewrites scored 15/37 and 17/37 against
15/37, within noise. Token logprobs show the misreads are confident (`\infty` 0.99, `\alpha` 0.99, `x` 0.99), so
decoding changes, n-best lists or confidence gating would not catch them.

**Fixes ranked by impact** (details in [section 5](#5-prioritised-plan))

| # | Fix | Files | Measured or expected effect |
|---|---|---|---|
| 1 | Size-relative, 2D-aware grouping (fraction bars, stacked rows) | `static/app.js` | fraction pieces 0/6 correct, grouped 9/9 |
| 2 | Rasterize at a fixed symbol scale in a ~1 MP canvas | `static/app.js` | 15/37 to 23-24/37 on the same images, no latency change |
| 3 | Convert when you move on, not on a short idle timer | `static/app.js` | removes the mid-write matrix misreads (3 of 40 images) |
| 4 | Group against what is visible; size typeset like the ink | `static/app.js` | removes the invisible sticky zone |
| 5 | `clean_latex` repairs: line breaks, empty matrices, `\text{---}` | `server.py`, `static/app.js` | fixes the two-line image; stops empty brackets hiding ink |
| 6 | Keep manual edits; make Re-recognize change its input | `static/app.js` | corrections survive later strokes |
| 7 | Regroup tools: lasso Group/Split, Convert now, Hold, join preview | `static/app.js`, `index.html`, `style.css` | user can repair any grouping mistake |
| 8 | Log strokes and prompt tokens with each debug image; keep a replay eval | `static/app.js`, `server.py` | exact replays of grouping and rendering later |
| 9 | Leave the prompt alone (drop the "thin" line after fix 2) | `server.py` | none measured; listed so nobody spends time here |

## 0. How this was measured

- **Data.** The 40 images in `debug/` with the raw replies are what the model actually received (`save_debug`
  writes the decoded upload). I labelled 37 of them from the images and left 3 out: one that could be an alpha
  or a squashed 2, two stray bars with no clear intent, and one half-erased word. Words are scored as written
  (the word as written, not a corrected spelling). Isolated loop glyphs accept `\partial`, `\sigma` or `\delta`. A lone bar accepts `-` or
  NONE. An arrow accepts NONE or an arrow command.
- **Model.** Only the already loaded `qwen3-vl:8b-instruct` (Q4_K_M, 5.8 GB resident) on the running Ollama
  0.30.9. I checked `nvidia-smi` first (9.1 GB of 16.3 GB in use, no other model loaded). All requests were
  sequential, about 360 in total, with the same system prompt, user prompt, `temperature: 0` and
  `num_predict: 512` as `server.py` (the harness imports them from `server.py`). server.py was not touched.
- **Noise.** Replaying the 37 images unchanged reproduced 33 replies exactly. The 4 that changed kept the score at
  15/37, so a difference of 1 or 2 images between two variants is noise.
- **Variants** were rendered from the rasters, since strokes are not saved. "Re-render" skeletonizes the ink and
  redraws it with a chosen symbol height and stroke width, antialiased, in a canvas of at least 1 MP. "Pad-only"
  keeps the original pixels and only adds white border.
- Scratch harness, outside the repo: `C:\Users\landa\AppData\Local\Temp\mbdiag\` (`harness.py` has the labels and
  the Ollama cache, `render.py`, `run_*.py`, `compose.py`). Nothing left the machine.

## 1. Recognition

### 1.1 What the model receives

| Property | Today | Notes |
|---|---|---|
| Canvas | 115 to 768 px per side, 0.02 to 0.43 MP | `RASTER_MAX` 768, scale `k` capped at 2 |
| Padding | `max(14, 0.1 * longest side)` board px | ink fills about 80% of the frame |
| Stroke | uniform, `max(5, 0.009 * canvas)` px, round caps | 5.2 to 8.8 px in the debug images; pressure ignored (good) |
| Smoothing | quadratic curves through midpoints, same as the screen | mouse jitter is visible but did not matter (below) |
| Antialiasing | Chrome gives 4 grey levels (0, 64, 127, 191, 255) | coarse but harmless |
| Background | opaque white RGBA, black ink | fine |
| **Server resize** | **every image grows to about 1024 tokens (about 1 MP)** | measured, see below |
| Upscale factor | 1.56x to 5.83x, median 2.13x | small crops (single glyphs) get the most |
| Stroke at model input | 12 to 37 px | |
| Symbol at model input | 89 to 1347 px, **median 415 px, about 13 tokens tall** | |

How the resize was measured: square blank PNGs from 64x64 up to 1024x1024, sent with a short prompt, all gave
exactly 1073 prompt tokens (about 49 text plus about 1024 image tokens). Non-square ones under 1 MP (166x186,
768x384) gave 1103 and 1107, from rounding to the 32 px grid. 1536x768 gave 1201, which is 48 x 24 tokens of
32 px plus text. So anything under about 1 MP is upscaled to about 1024 tokens, and the 768 px cap and the `k <= 2`
cap in `rasterize()` give no control over resolution. The only thing the board controls is how large a symbol is
relative to the frame. Replays of the board's own requests used 1387 to 1462 prompt tokens.

The runtime context is 4096 tokens (`ollama ps`). In one experiment a 1.8 MP canvas worked but a 3.0 MP canvas
(2497x1195) returned HTTP 400. Any new renderer should cap the canvas at about 1.5 MP.

### 1.2 Failure classes in `debug/`

Live replies (as recorded): **15/37 correct**. The classes overlap (a `d` that is also a fraction piece is counted
in both).

| Class | Images | Live result | Main cause |
|---|---|---|---|
| Fraction pieces (numerator, bar, denominator sent alone) | 9 | 0/6 symbol pieces; 2 of 3 bars NONE (accepted) | grouping |
| Looped 2 read as something else | 3 of 5 images (3 of 8 handwritten 2s) | `\alpha`, `x \geq`, `2 + \infty` | scale, plus isolation for 2 of them |
| Half-written matrix | 3 | 0/3 (entry dropped, `=` invented) | conversion timing |
| Math read as NONE | 5 (`-\infty`, two `d`, two loop glyphs) | 0/5 | scale and isolation |
| `d` and `\partial` confusions | (inside the fraction pieces) | `al`, `\overline{a} w`, `\text{d}\downarrow X` | isolation |
| Words | 5 | 2/5 (one short word came back with a subscript, another as a different real word) | scale |
| Two lines in one group | 1 | content right, renders on one line | `clean_latex` |
| Overlapping row and column vector | 1 | wrong | messy ink, probably out of reach for any fix |
| Plain lines, digits, single letters, full matrix | 10 | 10/10 | |

The user's 2 has a small loop at the bottom left where the stroke crosses itself. In a big, thick, isolated
rendering that crossing looks like an alpha, a figure eight or an x. In context and at a normal scale it is read as
2 (see 1.3).

### 1.3 Experiments (same 37 images each time)

| Variant | Correct | Change vs live |
|---|---|---|
| Live replies (recorded) | 15/37 | |
| Replay, unchanged | 15/37 | 33/37 replies identical |
| Prompt P1: handwriting hints (looped 2, d vs `\partial`, x vs `\times`), softer NONE rule | 15/37 | 0 |
| Prompt P2: two-sentence minimal prompt | 17/37 | +2 (noise) |
| Re-stroke only, same scale and width (control) | 14/37 | -1 (noise); `2 + 2 = 4` became `x + x = 4` |
| Re-render, symbol 128 px, stroke 9 px | 21/37 | +6 |
| **Re-render, symbol 64 px, stroke 5 px** | **23/37** | **+8** |
| Re-render, symbol 64 px, stroke 2.6 px | 22/37 | +7 |
| Re-render 64 px with prompt P1 | 21/37 | +6 |
| **Pad-only: original pixels, border so a symbol is ~64 px after the server resize** | **24/37** | **+9** |

- 18 images were right in all three scale-normalized re-renders, against 12 right in all three unnormalized
  runs.
- Handwritten 2s read as 2: 5/8 live, 6/8 to 7/8 with a normalized scale. `2 + \infty` became `2 + 2` in all five
  normalized runs. The 2s that still fail are one isolated 2 (`\alpha`, `\lambda` or `x` in every variant but
  one) and the fragment `2 3 ]`: both are context problems (fix 1).
- What changed with scale: `- \infty` stopped being NONE, `dx` stopped being `\text{d} \times`, words were copied
  instead of turned into subscripts, and the half-matrix `[ 1` kept its 1.
- Stroke width in the 2.6 to 5 px range at 64 px symbols did not matter (22 vs 23). Re-stroking and antialiasing
  did not matter (control 14 vs 15, pad-only 24 vs re-render 23). **Symbol scale relative to the frame is the
  lever.**
- Regressions from normalizing: a long word lost a letter in every 64 px run (pad-only included), and the
  re-renders read the lone arrow as `1` (pad-only kept NONE). Pad-only got `2 3 ]` wrong (`e_3`) where the 64 px
  re-render got it right. These are single images and within noise.
- Latency does not change: pad-only requests used 1383 to 1462 prompt tokens, the same as today, because the
  server was already scaling everything to about 1 MP.

**Grouping test.** I stacked the three pieces of each split fraction (they share the same scale, `k = 2`) with
12, 24 and 40 board-px gaps and sent each stack with today's rendering.

| Fraction | Pieces sent alone (live) | Stacked, 3 gap sizes |
|---|---|---|
| d/dx (first) | NONE, NONE, `\text{d} \times` | `\frac{d}{dx}` 3/3 |
| d/dx (second) | NONE, NONE, `\text{d} \downarrow \quad X` | `\frac{d}{dx}` 3/3 |
| dL/dw | `al`, `\text{0}`, `\overline{a} w` | `\frac{\partial L}{\partial w}` 3/3 |

The 64 px re-render of the stacks was also 3/3. Context fixes the `d` and `\partial` problems completely, even
at today's scale.

### 1.4 Prompt, sampling and `clean_latex`

- **Sampling is fine.** `temperature: 0`, `num_predict: 512`. The model has no `thinking` capability
  (`/api/show`: completion, vision, tools), so `think` is never sent and not needed. No constrained output,
  which is fine.
- **Logprobs are available** on this Ollama (`logprobs`, `top_logprobs`) but they don't help here. The wrong token
  had probability 0.99 for `\infty` in `2 + \infty`, 0.99 for `\alpha`, and 0.99 for `x` in `x \geq`. The model is
  confidently wrong, because the input is wrong. An alternatives menu built from logprobs would not offer 2.
- **Prompt.** Hints did not help, and moved errors around (with hints, the `d`s were read as `\partial`). A
  minimal prompt scored the same, but it loses the matrix and line formatting rules (`[1 2 3]` came back as plain
  text). Keep the current prompt.
- **`clean_latex` gaps seen in the replies:**
  - A raw newline between lines (`A x = b\nA = \begin{bmatrix}...`). KaTeX treats the newline as a space. With the
    vendored KaTeX I checked that this renders as one line, while a `gathered` version stacks it. The `gathered`
    fallback in `toHTML` doesn't fire, because it requires `\\` and no `\begin`.
  - `\begin{bmatrix} \end{bmatrix}` (3 of 40 live replies) renders fine, so the block is marked done and the ink of
    the `1` inside it disappears.
  - `\text{---}` for a lone bar (1 unchanged replay, 3 more under prompt P1) and `\text{NONE}` (1 unchanged
    replay) slip through the NONE check.
  - The `CELL_FIX` matrix repairs never triggered in this set.

## 2. Grouping

### 2.1 How it works today (`commitMath`)

A finished Math stroke joins every block whose ink box, grown by `REACH_X * reach` sideways and
`REACH_Y * reach` vertically (56 px and 16 px at the default reach of 1.0), overlaps the stroke's box. If none
matches and the last Math stroke was under 1.5 s ago, the sideways reach to that last block doubles (vertical
stays 16). A stroke that touches several blocks merges them and clears the typeset. Each change sets
`due = now + delay` (700 ms). `pump()` sends the oldest due block once the pen is up.

### 2.2 Why related strokes split

- **The reach is absolute, the handwriting is not.** From the 21 debug images whose scale is known (`k = 2`), this
  user's symbols are about 35 to 140 board px tall, median about 75 to 80 px (bars left out). 16 px of vertical
  reach is about 0.2 of a
  symbol, and 56 px sideways is about 0.7. Mouse handwriting is big, so the normal gaps are bigger than the reach.
- **Fractions.** The bars here are 160 to 240 px wide and 8 to 12 px tall. With 16 px of vertical reach the bar
  joins neither the numerator above it nor the denominator below it unless they almost touch. All three fractions
  in `debug/` split into three groups.
- **Matrices written entries-first.** A row below needs to be within 16 px, so rows split until a bracket bridges
  them. The bridge then merges them (that part works: `[1 ; 4 8]` became one group). Wide column gaps split
  sideways. `[1 2 3]` split into `[ 1` and `2 3 ]`, read as `\begin{bmatrix} \end{bmatrix} =` and
  `x \geq \begin{bmatrix} \end{bmatrix}`: that is where one of the "2 as x" cases came from.
- **Exponents and subscripts** usually work, because the small symbol overlaps the base's vertical range.
- **Separate lines** stay apart only while they are more than 16 px apart. `A x = b` and `A = [...]` were close
  enough to merge (debug 194001), then came back as one line (see 1.4).
- **The Grouping reach slider** scales both directions by the same factor, so reaching a denominator (x2.5 gives
  40 px) also merges stacked lines and neighbours. It can't fix fractions without breaking lines.
- **The fast-writing rule** only widens the sideways reach, only to the last block, and only for 1.5 s, which is
  short when moving a mouse to the next symbol.
- **NONE blocks still attract strokes.** An arrow or a lone glyph read as NONE stays a block with a box, so a later
  stroke near it joins it.

### 2.3 When a group is sent

After 700 ms of idle following the last stroke in that group. With a mouse, moving to the next symbol and starting
it often takes longer, so a half-written group converts, its ink is hidden, and the new typeset appears at a
different size. The next stroke then joins, and the whole group converts again. This produced the three
half-written matrix images. The model reads a fragment worse than the whole (above), so each partial conversion is
a chance for a visible wrong result, even if the final one is right.

### 2.4 Adding a stroke to a group that is already typeset

- The hit test uses the group's ink box (`b.box`), which is hidden. The visible typeset is at most 56 px font
  (digits about 36 px tall), left-aligned in that box and vertically centred. With 80 to 120 px handwriting, most
  of the ink box is invisible empty space that still captures strokes.
- `touch(p, b, keepTs = true)` keeps the old typeset dimmed and the old ink hidden, and shows the new stroke's ink.
  After the delay the whole group (old strokes included) is re-rasterized and re-read. The new reply replaces
  `b.latex`, **including a LaTeX edit you made by hand**, and the typeset is refit to the new box, so it can jump
  and change size.
- If the stroke touches two groups, they merge: the typeset is cleared, all the ink comes back, and the merged
  group is read again.
- Ctrl+Z restores the previous reading (the page snapshot has the LaTeX), but it also removes the stroke.

## 3. Writing freedom and correction

**What you can do today**

- Select (S), tap a group: edit its LaTeX, Re-recognize, Copy, Keep as ink, Delete, Send to 3D. Drag moves a whole
  group.
- Eraser: touching typeset reveals the ink; erasing strokes re-queues the group. Touching typeset also queues a
  re-read, even if you only wanted to look.
- Ctrl+Z / Ctrl+Y per page.
- Settings: Convert after (250 to 2500 ms), Grouping reach (0.5x to 2.5x, both axes), Max typeset size (20 to
  120 px), ink hidden or faint after conversion.
- Draw pen (D) for anything that should never convert.

**What is missing**

- **Regrouping.** No way to merge two groups or split one apart except erasing and rewriting further away. No
  lasso or multi-select, and individual strokes of a Math group can't be selected.
- **Pacing.** No "convert now" and no "hold": the only control is the global delay, which slows every conversion.
- **Feedback while writing.** You can't see which group a stroke will join until it is converted. The dashed
  outline shows only pending groups.
- **Sticky corrections.** A manual edit is lost when a stroke joins. Re-recognize is almost always a no-op
  (deterministic replay).
- **Size.** The typeset shrinks your expression to about half its handwritten height (56 px cap), so the layout
  you wrote moves around.
- **Choice after the fact.** Changing Math ink to Draw ink is only possible after conversion (Keep as ink). The
  reverse, Draw to Math, is not possible.

## 4. Supporting numbers

- User symbol height: about 35 to 140 board px, median about 75 to 80 (21 images at `k = 2`, bars left out).
- Model input today: upscale 1.56x to 5.83x; symbol median 415 px (13 tokens); stroke 12 to 37 px.
- Replay determinism at temperature 0: 33/37 identical.
- Replay latency: 0.39 to 0.87 s per request (prompt about 1400 tokens, mostly image).
- Requests used for this note: about 360, all sequential, one model.

## 5. Prioritised plan

### Fix 1. Size-relative, 2D-aware grouping

**Change.** In `commitMath`, replace the fixed reach with one derived from the writing size, and add a
fraction-bar rule.

- Writing size `S`: the median height of the page's last 30 Math glyphs. Merge overlapping stroke boxes into
  glyphs, skip dots (under 6 px) and bars (width over 4x height). Default 48 px until there are 3 glyphs.
- Reach: sideways `0.9 * S * reach`, vertical `0.35 * S * reach`. For this user that is about 72 px and 28 px
  instead of 56 and 16.
- Fraction bar: a stroke at least `1.3 * S` wide and 4x wider than tall reaches `1.2 * S` above and below,
  within its own x-span plus 15%. It joins every group in that zone. Later strokes that land in an existing bar's
  zone join the bar's group. The width threshold keeps an `=` sign from acting as a fraction bar.
- New-line guard: a stroke that starts left of a group's left edge and entirely below its bottom starts a new group,
  so the larger vertical reach doesn't merge stacked lines.

**Files.** `static/app.js` (`commitMath`, `REACH_X` and `REACH_Y`, new helpers `writingSize`, `isBar`,
`barZone`). The reach slider keeps working as a multiplier.

**Expected effect.** The three split fractions in `debug/` become single groups, which read 9/9 correctly in the
stacking test (their 6 symbol pieces read 0/6 alone). `[1 2 3]` written with this user's spacing stays one group. Fewer isolated glyphs overall, and
isolated glyphs are the worst class (2 of 8 correct live). Risk: some neighbouring expressions merge. Fix 7's join
preview and Split make that visible and repairable.

### Fix 2. Rasterize at a fixed symbol scale

**Change.** In `rasterize()`, choose the scale from the glyph size instead of the box size, and send a canvas that
is already about 1 MP so the server doesn't resize it.

```js
const SYM_PX = 64, LINE_PX = 5, MIN_AREA = 1 << 20, MAX_AREA = 1.5 * (1 << 20);
// S = median glyph height of this group in board px (same helper as Fix 1)
let k = SYM_PX / S;                                        // canvas px per board px
let w = bw * k + 1.5 * SYM_PX, h = bh * k + 1.5 * SYM_PX;  // content plus margins
if (w * h > MAX_AREA) { const f = Math.sqrt(MAX_AREA / (w * h)); k *= f; w *= f; h *= f; }
const g = Math.max(1, Math.sqrt(MIN_AREA / (w * h)));      // pad, don't scale, up to ~1 MP
// canvas = round(w * g) x round(h * g), content centred, lineWidth = LINE_PX / k
```

Keep black on white, uniform width and round caps. Drop `RASTER_MAX` and the `k <= 2` cap. Put the 1 MP figure in
one constant with a comment: it comes from the server's minimum image tokens, which could change with an Ollama
update. Have `server.py` return `prompt_eval_count` so it can be checked.

**Files.** `static/app.js` (`rasterize`, constants). `server.py` (add `prompt_eval_count` to `timings`).

**Expected effect.** 15/37 to 23/37 (re-render) or 24/37 (pad-only) on the debug set, for the same latency.
Fixes `2 + \infty`, `-\infty` read as NONE, `dx` read as `\text{d} \times`, and words turned into subscripts. The
MAX_AREA cap keeps long expressions inside the 4096-token context.

### Fix 3. Convert when you move on, not while you're writing

**Change.**

- When a Math stroke starts outside a pending group's reach, set that group's `due` to now: you've moved on, so
  convert it at once.
- For a group you're still near, wait longer: default idle 1200 ms for mouse input, and don't count time while the
  pointer hovers inside the group's reach zone, with a 3 s ceiling.
- Treat an empty matrix reply (Fix 5) as "not yet" and keep the ink.

**Files.** `static/app.js` (`onDown`, `onMove` hover tracking, `touch`, `pump`, `DEFAULTS.delay`).

**Expected effect.** No conversions of half-written matrices (3 of 40 debug images). Moving on converts sooner than
today's 700 ms. Less typeset churn while writing a 2D expression.

### Fix 4. Group against what's visible; size typeset like the ink

**Change.**

- In the hit test, use the typeset rectangle (`b.tsRect`) plus the boxes of any strokes not yet converted, when the
  typeset is showing. Otherwise use the ink box.
- Raise the default `maxFont` from 56 to about 110, or fit the typeset to the ink height without a cap, so the
  typeset covers roughly the area of the ink it replaces.

**Files.** `static/app.js` (`commitMath`, `paintTypeset`, `DEFAULTS.maxFont`).

**Expected effect.** What you see is what groups. No invisible zone to the right of or below shrunken typeset
pulling strokes in, and no layout jumping to half size on conversion.

### Fix 5. `clean_latex` repairs

**Change.**

- Newlines outside environments: join the lines with ` \\ ` and wrap them in `aligned` if every line has an `=`,
  otherwise in `gathered`. In `toHTML`, also allow the `gathered` fallback when the source has `\begin`.
- A matrix environment with no cells (`\begin{bmatrix}\s*\end{bmatrix}`, possibly with `=` or `\geq` around it):
  return an empty string, so the group stays as ink and is retried when it changes.
- `\text{---}`, `\text{--}` or `\text{-}` alone becomes `-`. `\text{NONE}` becomes NONE.

**Files.** `server.py` (`clean_latex`), `static/app.js` (`toHTML`).

**Expected effect.** The two-line image renders stacked (checked with the vendored KaTeX). Empty brackets no longer
hide a written `1` (3 of 40 live replies).

### Fix 6. Keep corrections; make Re-recognize meaningful

**Change.**

- `applyEdit` sets `b.manual = true`. A stroke that joins a manual group keeps the manual LaTeX, and the model's
  new reading appears as a one-tap suggestion ("Use: ...") instead of overwriting.
- Alternatively, don't let new strokes join a manual group unless they touch its typeset.
- Re-recognize changes the input: render at a second scale (for example 96 px symbols), or include the
  neighbouring groups' ink. Same input at temperature 0 gives the same answer 33 times out of 37.

**Files.** `static/app.js` (`applyEdit`, `commitMath`, `applyResult`, `ed-rerun` handler), `index.html` and
`style.css` (suggestion chip).

**Expected effect.** A fix made once stays fixed. Re-recognize becomes a real second opinion.

### Fix 7. Regroup tools and pacing controls

**Change.**

- **Lasso.** In Select mode, dragging on empty board draws a lasso that selects strokes and groups. The actions
  are Group (merge into one group and convert), Split (move the selected strokes into their own group), Convert
  now, Keep as ink, To math (turns Draw strokes into Math), and Delete.
- **Join preview.** While a Math stroke is being drawn, outline the group it will join, updated every few
  pointer moves, so a bad join is visible before the pen lifts.
- **Keys.** Enter or Space converts all pending groups now. Holding a key (or a Hold toggle) pauses conversion,
  for writing a big 2D expression slowly.

**Files.** `static/app.js` (select branch of `onDown`, `onMove` and `onUp`; `pickAt`; selection state; key map),
`index.html` (selection toolbar), `style.css`.

**Expected effect.** Every grouping mistake left by Fixes 1 and 4 can be repaired in one gesture instead of by
erasing and rewriting. Slow writers get control over timing without a global slow delay.

### Fix 8. Better debug data and a replay eval

**Change.**

- Send the group's strokes (board coordinates) with the image, and save them next to each debug PNG. Save the
  `prompt_eval_count` too.
- Keep a small labelled replay set and a script that reports accuracy for a renderer or prompt change (the scratch
  harness above does this for rasters).

**Files.** `static/app.js` (`recognize`), `server.py` (`save_debug`, `recognize`), an optional script under
`tests/`.

**Expected effect.** Grouping and rendering changes can be replayed exactly from strokes, and every change gets a
before and after number instead of an impression.

### Fix 9. Prompt: leave it

**Change.** None, except removing "so strokes are shaky and thin" once Fix 2 makes the rendering uniform. Don't
add handwriting hints: they scored 15/37 against 15/37, and 21/37 against 23/37 on top of the 64 px render. They
also shifted `d` into `\partial`. Don't build a logprob alternatives menu: the misreads have 0.99 confidence.

**Files.** `server.py` (`USER_PROMPT`), optional.

## Appendix: per-image results

The per-image table (what each capture showed and what each variant read) stays on the machine that
made it, in the gitignored `evalset/diagnosis_per_image.md`, since it transcribes the user's own board.
