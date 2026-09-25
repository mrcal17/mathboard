# Mathboard design system

This file is the visual spec for all three tabs. It has the principles, the tokens and shared
components (implemented in `static/style.css` sections 1 to 4 and `static/icons.js`), and a
per-area plan for the redesign. Each area is owned by one person or agent (section 8).

In short: **quiet, neutral chrome; colour only for data; detail on demand.** The canvases, the
matrices and the plots are the product. Everything around them should step back until you
point at it.

## 1. Principles

1. **Colour means data.** The chrome is neutral: slate greys taken from the chalkboard, and
   near-white or near-black text. There is no blue UI accent any more, because blue already
   means *positive*. A toggle that is on, a selected segment and a focused field are shown with
   tone (a lighter fill, a ring), never with a hue. Hue is kept for positive / negative values,
   the highlight (HI), attention, and the series colours of the 3D tab and the board inks.
2. **Quiet by default, detail on demand.** Show what a teacher needs to read from the back of
   the room. Everything else appears on hover, on selection, or behind a disclosure (a folded
   section, a menu, a tooltip). Each item in section 7 says what shows by default and where the
   moved detail now lives. Nothing is removed: every control and every key still works.
3. **One surface per thing.** Don't nest boxes. Separate things with space and tone before
   lines. A floating surface has one hairline and one shadow. Inside it, groups are separated by
   8 to 12 px of space. Hairlines (`--line-1`) are for the few places where space isn't enough.
4. **A strict text hierarchy.** There are four text levels (`--text-1` to `--text-4`) and six
   sizes. Numbers are tabular. Values are written in text colours, with the colour on the mark
   beside them (a swatch, a cell fill, the edge itself), not on the digits.
5. **Legible on a projector.** Canvas text scales with zoom (`--nnv-ts`). Body text is at least
   4.5:1 against its surface, and HI stays visible on both themes.
6. **Cheap to draw.** No `backdrop-filter` (it re-blurs the live SVG or WebGL canvas under it
   every frame). No filters or animated shadows on canvas elements. Transitions only on opacity,
   transform and colour. `prefers-reduced-motion` turns them off (base rule in style.css).
7. **One component, everywhere.** There is one button, one toggle state, one segmented control,
   one field and one floating panel. Module CSS may position and size them, but it should not
   invent new looks for them.

## 2. Tokens

All tokens live on `:root` (dark, the chalkboard) and `[data-theme="light"]` (the
whiteboard) in `style.css` section 1. `color-scheme` follows the theme, so native selects,
checkboxes, number spinners and scrollbars match.

**JS reads some of these with `getComputedStyle`**: `--bg` (as hex in view3d.js; nn.js, train.js,
surf3d.js and bridge.js too), `--fg`, `--ui-fg`, `--ui-muted`, `--ui-line` and `--accent`. They
must stay literal hex or rgba values, never `color-mix()`. `graph/scene.js` hardcodes the WebGL
clear colours to equal `--bg` (`0x1d2327` and `0xfbfbf8`): change both together.

### Surfaces (lowest to highest)

| token | dark | light | use |
|---|---|---|---|
| `--bg` | `#1d2327` | `#fbfbf8` | the page and every canvas |
| `--dock` | `#20272c` | `#f6f6f2` | docked panels: the matrix panel, the 3D sidebar |
| `--dock-act` | `#2f383e` | `#ffffff` | on a dock: the active row of a list (the 3D tab's row being edited), stronger than `--hover` |
| `--float` | `#262e33` | `#ffffff` | floating: toolbar clusters, cards, panels, menus, toasts |
| `--raise` | `#2f383e` | `#f0f0ec` | on a float: the hovered or active menu row, a hovered chip |
| `--well` | `#1b2125` | `#f1f1ed` | recessed: fields, segmented tracks, chart, plot and code backgrounds |
| `--thumb` | `#3a444b` | `#ffffff` | the selected segment (with `--shadow-1`) |
| `--hover` / `--press` | white 6% / 10% | black 4.5% / 8% | a transparent button being hovered or pressed |
| `--soft` / `--soft-hover` | white 7% / 11% | black 5% / 8% | the soft (secondary) button |
| `--on` / `--on-line` | white 11% / 14% | black 6.5% / 10% | a toggle that is on: fill plus an inset 1 px ring |
| `--float-line` | white 8% | black 8% | the hairline round a floating surface |
| `--scrim` | | | behind a modal (none yet) |

### Text and lines

| token | dark | light | use |
|---|---|---|---|
| `--text-1` | `#ecebe4` | `#1d1d1f` | values, titles, labels you read, active controls |
| `--text-2` | `#bcc3c7` | `#45494d` | body copy, idle controls |
| `--text-3` | `#939ca2` | `#6b7075` | captions, units, section heads, hints (4.5:1 or better on `--float`) |
| `--text-4` | `#687278` | `#a0a5a9` | disabled, placeholders, decoration |
| `--line-1` | white 6% | black 6% | dividers inside a surface |
| `--line-2` | white 10% | black 10% | control outlines, matrix brackets |
| `--line-3` | white 18% | black 20% | hovered and focused outlines |

### Controls

`--control` tints native checkboxes, radios and ranges (`accent-color`). `--primary-bg`,
`--primary-fg` and `--primary-hover` make the primary button (inverted: near-white on the
chalkboard, near-black on the whiteboard). `--focus` is the focus outline and `--focus-soft` the
3 px focus halo of `.ui-field`. `--scroll` is the scrollbar thumb. `--tip-bg` and `--tip-fg` are
for tooltips (dark on both themes).

### Legacy names

`--fg`, `--ui-bg` (= `--float`), `--ui-fg` (= `--text-1`), `--ui-muted` (= `--text-3`),
`--ui-line` (= `--line-2`) and `--accent` still work, so nothing breaks before an area moves
over. `--accent` is now a neutral slate (`#5b6870`), because a colourful accent would compete
with the data. Old rules like `background: var(--accent); color: #fff` therefore render as
slate buttons. **New code should not use `--accent`.** Use `.on` / `--on` for selected states,
`.primary` for the one main action, and `--text-1` for curves or indicators that are UI rather
than data.

## 3. Data palette

| token | dark | light | meaning |
|---|---|---|---|
| `--pos` | `#4aa3ff` | `#1f6fd1` | value ≥ 0 (store.js `POS`, `colorFor`) |
| `--neg` | `#ff7a59` | `#d8522c` | value < 0 (`NEG`) |
| `--hi` | `#ffd54a` | `#e8a800` | hover, selection, the lit step, the current sample (`HI`; view.css used `#e8a800` for light) |
| `--hi-soft` | HI 14% | HI 13% | the band behind a highlighted row, a flash |
| `--hi-text` | `#ffd54a` | `#9a6700` | HI used as text (the light HI is too pale for text) |
| `--att` | `#b794ff` | `#7442d6` | attention weights A_ij (computed, not a parameter); head 1 |
| `--head-2`, `--head-3` | `#199e70`, `#c98500` | `#1baf7a`, `#eda100` | attention heads 2 and 3 |
| `--head-4` to `--head-6` | `#e0629a`, `#4fb3d9`, `#9ccc3d` | `#c2185b`, `#1f86b8`, `#6a9a17` | heads 4 to 6 (literal hex: view3d.js reads them). Every panel cycles the six from `--att` |
| `--nan` | grey 40% | | a non-finite value |

- `store.js` is unchanged (`POS`, `NEG`, `HI`, `colorFor`). It stays the source for SVG, canvas and
  WebGL. The CSS tokens mirror it. The blue / orange pair passes the colour-vision checks on both
  themes (worst protan ΔE 23). **Diverging values fade to transparent over the surface**, so the
  midpoint is the neutral surface, never a hue.
- The light theme's HI was `#e8a800` in view.css, view3d.css, attnviz.css and surf3d.js, and
  `#d99a00` in matrix.css. It is now `--hi` = `#e8a800` everywhere, and `--hi-text` for text.
- **Text wears text tokens.** A number next to a coloured cell is `--text-1`. Colour the cell,
  the swatch or the edge, not the digits. (The inspector's derivations colour a weight's digits
  by sign to tie them to the slider. That is the one allowed exception, and it keeps
  `solid(v)`.)
- **Colour follows the thing.** A token keeps its colour in every view, and a head keeps its
  colour in every panel.
- **Status colours are separate from data.** `--ok`, `--warn` (amber, apart from HI's yellow)
  and `--danger` (with `--danger-soft`) are for state. Always pair them with a word or an icon.
- Series palettes stay where they are, because they are data: `graph/grapher.js PALETTE`, the
  features' basis and subspace colours, the board inks in `app.js COLORS`, and the Train
  classes (`train.js EXTRA`).

## 4. Type, space, radius, elevation, motion, focus

**Type.** Use `--font-ui` everywhere in the chrome: Segoe UI Variable Text, then Segoe UI and
the system stack. Chromium resolves the Windows 11 variable family by name, so no font is
vendored. A vendored font would also be missing from `view.js png()` exports, which draw the
SVG as an image. `--font-display` (Segoe UI Variable Display) is for titles of 16 px and up,
`--font-mono` (Cascadia Mono, Consolas) for code, LaTeX source and shapes, and `--font-math`
(KaTeX_Math) for italic symbols drawn outside KaTeX.

| token | size | use |
|---|---|---|
| `--fs-xs` | 11 px | overlines, tick labels, matrix row and column headers, kbd |
| `--fs-sm` | 12 px | captions, hints, chips, secondary values |
| `--fs-md` | 13 px | panel body and controls (12.5 px inside `.sm` controls) |
| `--fs-bar` | 13.5 px | toolbars and the tab switcher |
| `--fs-lg` | 14 px | card titles, readouts (`.ui-stat`) |
| `--fs-xl` | 16 px | Explain title and text, big titles |
| `--fs-2xl` | 20 px | the board editor preview, hero numbers |

Line heights are `--lh-tight` 1.2, `--lh` 1.4 and `--lh-text` 1.55 (running text). Use two
weights, `--fw-medium` 500 and `--fw-strong` 600, and never bold on hover (it shifts the
width). `.ui-overline` (11 px, 600, uppercase, 0.06em) is only for menu group heads and the
cheat sheet. Section heads inside cards are sentence case (`.ui-sec-h`). Canvas text keeps its
own sizes in view.css, multiplied by `--nnv-ts`.

**Space.** A 4 px grid: `--sp-2`, `--sp-4`, `--sp-6`, `--sp-8`, `--sp-12`, `--sp-16`, `--sp-24`
and `--sp-32`. Panels are padded 12 px. Rows inside panels are 8 px apart and groups 12 px.
Controls are 30 (`.ui-btn`), 26 (`.sm`) or 22 (`.xs`) px tall. Toolbar buttons and tabs are 31 px
(nn.js fits the bar around that).

**Radius.** `--r-xs` 4 (cells, kbd), `--r-sm` 6 (fields, segments, xs buttons), `--r-md` 8
(buttons, tracks, menu rows, inner boxes), `--r-lg` 12 (floating surfaces) and `--r-pill`.
Nested corners step down by the padding between them: a 12 px panel with 4 px padding holds
8 px rows.

**Elevation.** `--shadow-1` is for thumbs and knobs, `--shadow-2` for docked-over floats
(toolbar clusters, cards, panels) and `--shadow-3` for transient layers (menus, popovers, toasts,
the cheat sheet, Explain). A floating surface is `--float` + `1px solid --float-line` + a
shadow. It is opaque, with no backdrop blur.

**Motion.** `--dur-1` 90 ms (hover, press), `--dur-2` 160 ms (state, fades, toasts), `--dur-3`
240 ms (panels, layout). The easings are `--ease` (standard) and `--ease-out` (entering). Only
opacity, transform and colours animate. The step pulse and the breathing ring on the canvas are
data animations and stay.

**Focus.** A global `:focus-visible` gives a 2 px `--focus` outline, offset 2 px. Text fields get a
1 px `--line-3` outline, or with `.ui-field` a `--line-3` border and a 3 px `--focus-soft` halo.
Toolbar buttons that never take focus (the Net bar) are unaffected.

**Layers (z-index).** On the board: typeset 1, ink 2, fx 3. `#graph` and `#nn` are 5, the board
toolbar 10, the tabs 15, panels 20 and the toast 30. Inside `#nn-stage`: the 3D view 4, the
Train / Attention / 3D plots panels 6, the lens bar 7, cards 20, Explain 25, and `#nn-bar` 30
(its menus).

## 5. Components

Each is a `ui-*` class in style.css section 3, plus element defaults written with `:where()`,
so any module rule wins over them. Add the class to the element in JS where the DOM is built.
If a module has to keep its own class, copy the recipe using the tokens. Don't invent a
variant. If one is missing, say so in your report.

| component | class | recipe |
|---|---|---|
| button (ghost) | any `<button>` | transparent, radius 8, `--hover` / `--press`; disabled at 40%. `.ui-btn` adds the sized version: 30 px, 0 10 px padding, 13 px, `--text-2`, turning `--text-1` on hover |
| toggle on | `button.on` | `--on` fill, inset 1 px `--on-line` ring, `--text-1`. No colour, no border change |
| soft (secondary) | `.ui-btn.soft` | `--soft` fill, `--text-1`. Used for Step, Reset, connect previous |
| primary | `.primary` / `.ui-btn.primary` | inverted `--primary-bg` / `--primary-fg`, 600. **One per surface** (Play, Apply) |
| danger | `.ui-btn.danger` | ghost that turns `--danger` on `--danger-soft` on hover (remove edge, Delete, Erase all) |
| icon button | `.ui-btn.icon` (+ `.sm` / `.xs`) | square, a 16 / 14 px icon, `title` required |
| sizes | `.ui-btn.sm`, `.ui-btn.xs` | 26 px / 12.5 px text; 22 px / 12 px text |
| segmented | `.ui-seg` (+ `.sm`) | a `--well` track (radius 8, 2 px padding, 2 px gap); segments 24 px, `--text-3`; the selected one is `.on`: `--thumb` + `--shadow-1` + `--text-1`. For one choice out of a few (modes, views, head count) |
| text / number field | `.ui-field` (+ `.num`, `.mono`, `.sm`) | `--well`, transparent 1 px border, radius 6, 26 px, 12.5 px; hover `--line-2` border; focus `--line-3` border + `--focus-soft` halo; `.num` is tabular, right-aligned, no spinners |
| select | `select.ui-field` or `.ui-select` | a field with `appearance: none` and a 12 px chevron. `.on` while it is set (the lens bar's Focus): the toggle's `--on` fill and ring, the chevron kept |
| checkbox | `.ui-check` in a `.ui-check-row` label | native, 14 px, `--control` tint |
| switch | `input[type=checkbox].ui-switch` | 28 × 16 pill, `--line-3` track, `--primary-bg` when on, 12 px knob. For settings that are on or off (faint ink, neuron maps) |
| slider | `.ui-range` | native range, `--control` tint, 16 px tall. Data sliders set `accentColor` inline to the sign colour, as the inspector does |
| floating panel | `.ui-float` > `.ui-float-head` (`.drag`) + `.ui-float-body` + `.ui-float-foot` | `--float`, `--float-line`, radius 12, `--shadow-2`. The head is 38 px: `.ui-float-title` (13 px, 600) + `.ui-float-meta` (12 px, `--text-3`, tabular) + `.ui-float-sp` + `.ui-btn.sm.icon` actions (`--text-3`). **No rule under the head.** The body is padded 0 12 12 and scrolls; the foot has a `--line-1` top rule and right-aligned actions. `.folded` hides the body and foot. The resize corner is `.ui-grip` |
| board panels, cheat sheet | `.panel` | `--float`, radius 12, `--shadow-3`, padding 12, selectable text |
| popover / menu | `.ui-menu` > `.ui-menu-item` (`.act`), `.ui-menu-sep`, `.ui-overline.ui-menu-head` | `--float`, radius 12, `--shadow-3`, 4 px padding; rows 30 px, radius 8, `--raise` when hovered or active, an optional icon (`--text-3`) and a `<small>` second line |
| tooltip | `.ui-tip` | `--tip-bg` / `--tip-fg`, radius 6, 12 px, tabular, `<small>` second line at 66%. For hover readouts drawn by the app. Native `title` stays for button hints |
| kbd | `<kbd>` or `.ui-kbd` (+ `.lg`) | 18 px tall, mono 11 px, `--well`, `--line-2` border plus a 1 px bottom inset. `.lg` is 20 px and 12 px, for running text of 15 px and up (the empty-net hint) |
| chip | `.ui-chip` (+ `.on`, `.hi`) with an optional `.ui-sw` swatch | a 22 px pill in `--well`, 12 px, `--text-2`; `.on` works like a toggle; `.hi` adds a 1.5 px `--hi` ring (the followed token). For a pick among many: tokens, heads, filters |
| swatch | `.ui-sw` | an 8 px dot, the colour set inline (`style="background: var(--head-2)"`): the colour of what a label names (a head, the ball's HI), on the mark, never on the text. Stands alone (matrix head titles, notes) or leads a chip |
| folding section | `.ui-sec` > `button.ui-sec-h` (+ `.ui-sec-sum`) + `.ui-sec-b`; `.folded` | the head is 26 px, 12 px 600 `--text-3` sentence case with a chevron, `--text-1` on hover; `.ui-sec-sum` is a right-aligned summary (`--text-4`, tabular, ellipsed, `--text-3` on hover) shown only while folded. Sections are 8 px apart, with no dividers |
| text link | `.ui-link` | a button that reads as words ("more" after a folded note): the line's font, `--text-2`, a `--line-3` underline 2 px below; `--text-1` and a full underline on hover |
| help icon | `.ui-help` on a `.ui-btn.sm.icon` (`tabindex="-1"`, an `aria-label`) or a span | the how-to lives in its `title`. Not a control: no fill on hover, `cursor: help`, `--text-3` turning `--text-1`. Mark it `.ui-chrome` (the audience has no use for it) |
| readout | `.ui-stat` > `b` + `span` | the value is 14 px 600 `--text-1` tabular; the label 11 px `--text-3` |
| callout | `.ui-callout` (+ `.warn`, `.danger`) | a `--well` box, radius 8, 12.5 px `--text-2`; `.warn` and `.danger` add a 2 px status rule at the left. Always with words |
| text roles | `.ui-overline`, `.ui-caption`, `.ui-num`, `.ui-divider`, `.ui-sr` | as named |
| icon | `mathboardIcons.svg(name)` or `<span data-icon="name">` | section 6 |
| toast | `#toast` | `--float`, radius 12, `--shadow-3`, 13 px, centred 36 px from the bottom, max 560 px, and a 160 ms rise when shown |
| clean-view chrome | `.ui-chrome` | hidden by H (`body.clean`) and in audience windows (`body.lec-audience`, `#nn.nn-audience`). Mark new chrome with it instead of writing more clean-mode rules |
| tab switcher | `#tabs` | a float cluster; the buttons are 31 px, 13.5 px 500, `--text-3`, with a 15 px icon; the active one is `--thumb` + `--shadow-1` (light theme: `--well` + a `--line-1` ring) |

**Scrollbars** are thin, with a `--scroll` thumb on a transparent track (`scrollbar-width` and
`scrollbar-color` on `*`).

## 6. Icons

`static/icons.js` is a plain script loaded before `app.js`. It gives
`window.mathboardIcons = { svg(name, { size, cls }), hydrate(root), names }`. The icons are 24 × 24
line drawings stroked with `currentColor` (1.8 wide, round caps and joins), so they take the
button's text colour. Static markup uses `<span data-icon="name"></span>`, which is filled when
the script loads. Call `hydrate()` again for markup built later, or insert `svg()` strings
directly (nn.js `addButton({ icon })` takes HTML).

The current names: board, cube, net, plus, minus, close, undo, redo, trash, sparkle, layout,
fit, dice, home, search, more, pin, pencil (also `edit`: rename; board is the tab's pencil),
settings, chevron-down/up/left/right, play, pause, stop, step-forward, step-back, reset, record,
weights, lens, layers, flow (a matrix into the next-word bars), train, attention, surface,
explain, file, audience, help, sigma, draw, eraser, pointer, laser, download, upload, image,
link and right-angle (for the 3D tab's Link cameras and Right angles; those toggles stay text-only
while the display row has to fit beside the output icons).
The glyph icons in use today (✦ ⇉ ◎ ⬡ ∇ ⌒ ≋ ▷ ⧉ ↶ ↷ ⚙ Σ ✎ ⌫ ⬚ ◉ ‹ › « ») should all move to
these. Adding an icon is a foundation change: ask for it in your report and use the closest
existing one in the meantime.

## 7. Per-area plan

Items are in priority order: P1 is the redesign, P2 polish, P3 nice to have. "Default" is what
the teacher sees without doing anything, and "Detail" is where the moved information now lives.

### A. Net canvas and inspector (`view.js`, `view.css`, `inspector.js`, `inspector.css`)

Canvas, P1 (the biggest source of noise: nested boxes, repeated labels, a loud residual path):
- **A1 Lanes.** Default: no lane rectangles. The neurons stand on the dot grid in columns.
  `.nnv-band` gets no fill or stroke by default. Detail: a `--hover`-strength tint when the
  layer's header is hovered, a 1.5 px `--hi` outline when the layer is selected, and a `--text-3`
  outline for the lens-focused stage. Empty layers keep a dashed `--line-2` outline, because it
  is where you double-click to add a neuron.
- **A2 Headers.** Default: text only, with no box. The name is 14 px 600 `--text-1`
  (× `--nnv-ts`); the sub line (act · size) is 12 px `--text-3`. A header may be wider than its
  lane, so "H = X + Z W_O" no longer truncates. Where two headers would collide, drop the
  narrower one's sub line. Detail: hovering shows a `--hover` pill (rx 8) behind the text, and
  selecting shows the same pill with a 1.5 px `--hi` ring. The pill rect stays as the hit area
  (transparent by default), so dragging and clicking headers work as before.
- **A3 Token nets.** Default: remove the dashed Q / K / V group boxes (`.nnv-grp`) and keep the
  group letters (`--text-3`, italic). Token boxes get a `--nnv-tok` fill and no stroke. Token
  labels (t₁, or the token's word) appear **once per row**, in a left gutter beside the first
  token layer, instead of in every layer where they collide with the values. Detail: hovering a
  token box shows its label in that layer and a `--line-3` stroke. The followed token keeps its
  `--att` ring.
- **A4 Fixed edges** (residual identity, fixed pooling). Default: neutral, not weight-coloured:
  1.25 px `--text-3` at 35% opacity, dashed 2 5. They are not parameters, so they get no data
  colour (today they are the loudest thing on the transformer canvas). Detail: full strength
  plus the HI outline when hovered, selected or lit. The edge card says "fixed".
- **A5 Weight labels (W).** Default with W on: the value only (12 px 600, with the `--bg`
  halo), placed 40% along the edge from the source and staggered per target so neighbours don't
  collide. Detail: the ∂L/∂w line shows only on the hovered or selected edge and its tie group.
- **A6 Hover and selection.** Replace the 14 px translucent HI halo on edges with a crisp
  outline: a second path 3 px wider than the line, under it, in `--hi` at 50% (hover) or 90%
  (selected). Neurons: a 2 px `--hi` ring at 70% with a 3 px gap when hovered, and 2.5 px at full
  strength when selected. The step-through keeps its breathing ring and pulse.
- **A7 Neuron chrome.** The rim is 1.25 px at 28% (dark) or 22% (light). The KaTeX label gets a
  single halo (`0 0 3px var(--nnv-bg)`) instead of three. Value labels are 12.5 px 500 `--text-2`,
  turning `--text-1` when the neuron is hovered, selected or lit. δ labels show only on hover,
  on selection or with W on. The target label ("y = 1.00") is 12 px `--text-3`.

Canvas, P2:
- **A8 Attention heatmap in the header.** It is data, so it stays. Remove its frame and the "A"
  caption (the header already says attention), put it inline to the left of the name at cap
  height, and use 1 px gaps between cells.
- **A9 Dot grid.** `--nnv-dot` at 4.5% (dark) or 6% (light), hidden when zoomed out
  (`--nnv-ts` > 1.3).
- **A10 Connect handle.** 5 px, `--nnv-node` fill with a 2 px `--hi` stroke, shown only while that
  neuron is hovered.
- **A11 Tokens.** Map the `--nnv-*` variables onto the tokens (`--nnv-text: var(--text-1)`,
  `--nnv-muted: var(--text-3)`, `--nnv-hi: var(--hi)`, `--nnv-att: var(--att)`, `--nnv-bad:
  var(--danger)`, and so on). Keep `VARS` in view.js in step, so `png()` still resolves them.
  `.nnv` keeps a literal font stack. It may use the `--font-ui` family names, but not the
  variable, because png() doesn't resolve it.
- **A12 Empty-net hint.** 15 px `--text-3` on one line, with `<kbd>`-like chips for N and the
  double-click.

#### Net 2D view (the canvas overhaul)

Data first, structure recedes. Where this differs from A2, A4 to A7 and A9, this wins.
- **Edges.** Screen px strokes (world px set as px ÷ zoom; not `vector-effect: non-scaling-stroke`,
  which costs Chromium a fresh stroke of every path each paint), 1 to 2.25 px by |w| / max|w|
  (attention 1 to 3 px by A_ij). Opacity follows the same ratio on a perceptual curve: 16 levels
  (attention 12) with equal ΔE steps against `--bg`, the strongest at 0.9. Below 3% of the largest
  weight an edge is a neutral hairline (`--nnv-zero`, 10%): structure, not data. An attention edge
  under A = 0.02 is not drawn. Fixed edges are dotted `--text-3` at 45%.
- **Batched.** Edges have no DOM of their own: one `<path>` per colour, level and lens level
  (`.nnv-b` in `.nnv-eb` groups), hit-tested in view.js. What is hovered, selected or stepped, a tie
  group, and the edges of a hovered or selected neuron or layer come forward as their own paths
  (`.nnv-fe`) at a stronger opacity (a floor of 0.3), with the HI outline under the marked ones. Hover
  dims the rest to 13% (attention 10%), selection does not.
- **Neurons.** One circle: the activation as fill (colorFor over `--nnv-node`) and a 1 px rim at 20%
  (16% light). The δ ring is detail, like the δ numbers.
- **Level of detail by zoom.** Below 0.42: fills only (no labels, numbers or δ rings; W labels
  wait; edges under a pixel); below 0.3 the token row labels and group letters go too. Below 1.1:
  the input and output layers' labels, numbers and targets. From 1.1: all. The hovered, selected and stepped neurons, and a hovered neuron's
  neighbours, always show theirs. KaTeX labels render the first time they can show.
- **Headers.** One label per layer: the name (13 px 600 `--text-2`) and the shape (12 px `--text-3`),
  tabular, on one line; stacked, then the name alone (shape in the tooltip) where a neighbour is
  close. They grow up to 10 times when zoomed out, as far as the neighbours allow.
- **No boxes.** Token boxes are hit areas: a `--nnv-tok` tint on hover, the `--att` ring when
  followed. The lens-focused lane is a `--hover` tint, not an outline.
- **Motion.** Overlay edges fade in and out over `--dur-2`; a lens change cross-fades the edge layers
  (so Explain steps glide); nothing animates per training frame.

Inspector, P1 (today a node card is a full screen tall):
- **A13 Card chrome.** Use the `.ui-float` recipe. The 38 px head holds the title (KaTeX 15 px)
  and the kind (12 px `--text-3`). Pin and close are `.ui-btn.xs.icon` (icons `pin`, `close`),
  in `--text-4` until the card is hovered. There is no rule under the head and no yellow border
  on `.sel` (the canvas already shows the selection). Pinned is shown only by the pin icon in
  `--text-1`. `flash` becomes a 600 ms `--hi-soft` fade of the head, not a ring.
- **A14 Progressive disclosure.** Every section starts folded, with its summary on the right
  (inspector.js `OPEN_BY_DEFAULT` is empty); what stays open is the head, the where-it-sits line,
  a parameter's slider and the one-line computation. Opening or folding a section is remembered
  per section kind (`node.grad`, `layer.act`, ...) for every card and across sessions
  (localStorage `mathboard.nn.inspector`), and the audience window follows it. The lists below
  were the first plan; each of their "Default" sections is now one click away.
  - Node. Default: a value strip under the head (`.ui-stat` pairs: z, a, and δ when backward is
    on) and **Weights in + bias**. Folded, each with a summary on the right: Forward
    ("z 0.85 → a 0.69"), Gradients ("δ 0.058"), Outgoing weights ("3 edges") and Params ("2").
  - Edge. Default: Weight and Contribution. The Edge section merges into the head's kind line
    ("row 1, column 2 of W⁽¹⁾"). Folded: Gradient.
  - Layer. Default: Layer (name, size, the shape line) and Activation. Folded: Wiring and
    Biases / Inputs. For an attention layer, Attention is open.
- **A15 Label editing.** Default: the label field leaves the body. Detail: double-click the title,
  or click the pencil that shows while the head is hovered, to edit the KaTeX label inline (a
  `.ui-field.mono` in place of the title; Enter commits, Esc cancels). The "row 1 of W, column 1
  of ..." line and the layer chip become one `--text-3` line under the head.
- **A16 Sections.** Use `.ui-sec` (sentence case, a chevron, 8 px apart, no dividers and no
  uppercase). The body is 13 px.
- **A17 Slider rows.** 28 px tall: an 8 px swatch, the KaTeX label at 13 px, the range with
  `accentColor = solid(v)` as now, and a 58 px `.ui-field.sm.num`. Hovering a row gives it a
  `--hover` fill. A linked row (`.hi`) gets a `--hi-soft` fill and no inset ring.
- **A18 Controls.** Connect previous / next and randomize become `.ui-btn.sm.soft`. Remove edge
  becomes `.ui-btn.sm.danger` in a `.ui-float-foot`. The "+ add" param button becomes
  `.ui-btn.xs` with the plus icon, and selects become `.ui-field.sm`.

Inspector, P2:
- **A19 Plots.** The activation curve is 1.75 px `--text-1` (it was `--accent`), with `--line-2`
  axes, and the current sample's dot in `--hi`. 64 px tall.
- **A20 Attention in cards.** A-row bars are `--att` on `--well` (they were `--accent`), with
  values in `--text-2` tabular. Heatmap cells use `colorFor` with 2 px gaps and no borders.
- **A21 Placement.** A new card avoids covering an open floating panel when it can: use the same
  free-area rects as `view.fit`. Its max height is min(70% of the stage, 640 px).

### B. Matrix panel (`matrix.js`, `matrix.css`)

P1:
- **B1 Surface.** The panel background is `--dock`, and the splitter hairline is the only
  divider. Padding 12 16.
- **B2 Toolbar.** One 40 px row, sticky, on `--dock`, with no bottom rule. Left: step back (a
  `.ui-btn.sm.icon`), Step (`.ui-btn.sm.soft` with the step-forward icon) and stop (an icon).
  Next to them, the step readout in `--hi-text`, only while stepping. Right: Backward, [W | b],
  Batch, Collapse and Labels as `.ui-btn.sm` toggles with `.on` (neutral, `--text-3` when off,
  no outlines), then → 3D as a ghost `.ui-btn.sm`. When the panel is narrower than the row, the
  toggles move into a "View" `.ui-menu`. The keys (S, Shift+S, B) still work.
- **B3 Cells** (the heaviest part today). 22 px tall, radius 4, min-width 3.2em, 2 px gaps, UI
  font 12.5 px tabular (not monospace). The fill is `colorFor` with its alpha capped at 0.72, so
  the text can always be `--text-1`: drop the dark-text flip (`.nm-strong`). Zero cells get
  `--text-3` and no fill. Masked cells are transparent with a `--line-1` hatch. Brackets are
  1.25 px `--text-3`.
- **B4 Layer blocks.** The head line reads "Layer 1  Hidden · tanh" (13 px 600 `--text-1` plus
  `--text-3`). The shape chain (4×2 · 2×1 + 4×1 → 4×1) is in `--text-4` mono 11 px and shows only
  while the block is hovered, or always with Labels on. Blocks are 20 px apart with a `--line-1`
  hairline. There is no left border unless the block is hovered or selected, which gives it a
  2 px `--hi` left rule and no wash.
- **B5 Highlights.** A hovered cell gets a 1.5 px `--hi` inset ring. The step's row becomes one
  `--hi-soft` band behind the row instead of an outline per cell. Dashed secondary outlines
  (`nm-hi2`, `nm-an2`) become `--hi-soft` fills. The selection keeps its ring and loses its glow.
- **B6 Step box and token trace.** A `--well` card, radius 8, with a 2 px `--hi` left rule; its
  title is 12 px 600 `--hi-text`. No yellow wash and no yellow border.

P2:
- **B7 Headline.** The formula is at 15 px. The description under it is a `.ui-caption` clamped to
  two lines, which expands when clicked.
- **B8 Row and column headers** (h₁, x₁, t_i) are in `--text-4` until their row or column is
  hovered. They keep emitting row, column and token hovers.
- **B9 Flattened.** The "Flattened: z = W a" toggle becomes a ghost `.ui-btn.xs` with a chevron.
  Its open block is indented 12 px behind a `--line-1` rule.
- **B10 Lens-folded layers.** A single summary line in `--text-3`, turning `--text-1` on hover.
- **B11** Use `--hi` / `--hi-soft` / `--hi-text` instead of `--nm-hi` (the light theme becomes
  `#e8a800`).

### C. Net floating panels (`train`, `attnviz`, `surf3d`, `lens`, `tour`)

Shared, P1: each panel uses the `.ui-float` recipe with a `.ui-float-head` (dragging stays on
the head), `.ui-seg` for modes, `.ui-field` / `.ui-select` for inputs, `.ui-grip` for the resize
corner, and no rule under the head. Italic hint lines become `.ui-caption`, not italic. The
"how to use it" hints move into a help icon's `title` in the head. Widths: Train 300, Attention
400, 3D plots 460.

Train, P1:
- **C1 Head.** "Train" + meta ("epoch 12 · loss 0.183") + a play / pause icon + a fold chevron.
- **C2 Default body.** The dataset as a full-width `.ui-select`. An action row: Play
  (`.ui-btn.sm.primary` with the play or pause icon), Step (soft) and Reset (ghost), with the
  init seed as a small field on the right next to a reset icon. The readout as a `.ui-stat` row
  (epoch, step, loss, acc or words). The loss chart on `--well` with no border, 56 px. The plot
  select plus neuron maps (`.ui-check-row`), and the plot with radius 8 and no border. The sample
  ◀ ▶ are `.ui-btn.xs.icon`.
- **C3 Hyperparameters.** Default: points, noise, seed, loss, rate, batch and speed go into a
  folded `.ui-sec` called "Settings". Its summary reads "200 pts · noise 0.1 · lr 0.1 · batch 10
  · 5×". Detail: unfold it. The clean view and audience (`.ro`) still hide all of it.
- **C4 Warning.** `.nt-warn` becomes a `.ui-callout.warn`, and "Adapt network" a
  `.ui-btn.sm.soft`.
- **C5 Canvas colours.** The loss curve is `--text-1` (it was `--accent`) and the grid `--line-1`.

Attention, P1:
- **C6 Head.** The title + a `.ui-seg` (Arcs, Dots, Mix, Heat) + the layer as a `.ui-select.sm` +
  close. The query and head picks are `.ui-chip`s: the followed one is `.hi`, and "auto" is a
  `--text-4` chip with a title.
- **C7 Drawings.** Token boxes get a `--well` fill and no stroke. Hot and hovered ones get a
  `--line-3` stroke (not `--accent`). Arcs are `--att`, and heads use `--att`, `--head-2` and
  `--head-3`. Numbers use text tokens. Plot backgrounds are `--well` with no stroke.
- **C8 Hints.** The hints move behind the help icon. The formula caption (KaTeX, 14 px) stays.
- **C9 Scale row** (Dots and Mix only): a `--text-3` label, a `.ui-range`, and the readout in
  `--text-2` tabular. The "1/√d" reset is a `.ui-btn.xs`.

3D plots, P1:
- **C10 Head.** The same pattern, with a `.ui-seg` of Surface, Landscape, Space and Simplex (a
  plot that doesn't apply looks disabled), and home and close as icons.
- **C11 Controls.** `.ui-select.sm`, `.ui-check-row`, and `.ui-btn.sm.soft` for Re-center and
  Clear path. The stops are a `.ui-seg.sm`, and play an icon button.
- **C12 View.** Radius 8, no border, `--bg`. CSS2D labels are 12 px `--text-3` with one halo. The
  caption is KaTeX at 14 px. The note is a `.ui-caption`, folded to one line with a "more" link.

Lens bar:
- **C13 (P1) Default: hidden** (see the open decisions). L and the toolbar's Lens button show it,
  and the button's dot says when a lens is active.
- **C14 (P1) When shown.** A single-row `.ui-float` pill, 40 px tall, radius 12. Groups are 12 px
  apart, with no dividers. Focus is a `.ui-select.sm`, and tokens are `.ui-chip`s (a select when
  there are more than five). The edge thresholds are a label + a 72 px `.ui-range` + a readout,
  and Clear is a `.ui-btn.xs`. Below 1300 px of stage width, the threshold groups fold into a
  "more" menu, so the bar never wraps over the Train panel (as it does at 1280 today).

Explain, P2:
- **C15 Card.** The `.ui-float` recipe with `--shadow-3`, min(600 px) wide, padding 14 16. The
  count ("4 / 12") is a neutral `.ui-chip`, tabular. The title is 16 px 600 `--font-display`, and
  the text 15.5 px / 1.55. Navigation is `.ui-btn.sm.icon` (chevron-left, chevron-right, close).
  The progress bar is 2 px of `--text-3` on `--line-1`. Its placement avoids the lens bar and the
  panels (tour.js `place()`).

### D. Net shell and 3D view (`nn.js`, `nn.css`, `view3d.js`, `view3d.css`)

P1:
- **D1 Toolbar** (keep the clusters and the popover structure: the user likes them). Use the
  tokens: `--float`, `--float-line`, `--shadow-2`. Buttons are `--text-2`, turning `--text-1` with
  `--hover`. The on state is the shared one. Icons come from `mathboardIcons.svg` (sparkle, plus,
  layout, fit, dice, undo, redo, weights, lens, layers, train, attention, surface, explain, file,
  audience, help), 16 px, 6 px from the label. Undo, redo and ? become icon-only 31 px buttons.
  The `.nn-compact` / `.nn-tight` fitting stays as it is.
- **D2 Menus.** `.nn-pop` takes the `.ui-menu` look: no `backdrop-filter`, `--raise` for the
  active row, the search field as a 32 px `.ui-field`, kbd hints, and `.ui-overline` section
  heads. The layout and the keys stay the same.
- **D3 Cheat sheet.** An opaque `.panel`, 560 px wide, in two columns (the keys as kbd chips, and
  what they do). Sections get `.ui-overline` heads and a close icon.
- **D4 Splitter.** 1 px `--line-1`, becoming 2 px `--line-3` on hover or drag (not `--accent`).
  With the matrix panel hidden, a small pill grip shows at the right edge on hover.
- **D5 Toast.** In the Net tab it is centred over the stage (not the window) and sits above the
  lens bar.
- **D6 3D view bar.** The `.ui-float` recipe. The mode is a `.ui-seg` (Stack, Heads, Tensor), the
  steps a `.ui-seg.sm`, selects are `.ui-select.sm` and buttons `.ui-btn.sm`. The caption is KaTeX
  at 14 px, and code sits on a `--well`.

P2:
- **D7 3D labels** (CSS2D). Headers are plain text, as on the 2D canvas (no pill), with one
  halo. Tensor mode shows the numbers only on the current step's matrices. With 1.2 on, every
  cell shows its number, full size from a 28 px face and shrunk to fit below that; with it off,
  only the hovered or lit cell (decision 10).
- **D8 Clean view.** The rules stay in nn.css. New chrome uses `.ui-chrome`.

### E. Board tab (`app.js`, style.css section 5, the board markup in `index.html`)

P1:
- **E1 Toolbar.** Tokens (`--float`, `--float-line`, `--shadow-2`) and buttons in `--text-2`
  with the shared on state for the current tool. Icons via `data-icon`: sigma, draw, eraser,
  pointer, laser, undo, redo, chevron-left, chevron-right, plus, trash, download and settings.
  The tools keep their labels. Clear and Export may go icon-only, with titles. Group dividers
  become 1 px `--line-1` lines 20 px tall, not the full height.
- **E2 Swatches.** 18 px dots with a 2 px gap. The current one gets a 2 px `--text-1` ring at a
  2 px offset (it was `--accent`).
- **E3 Pages.** "1 / 1" in `--text-3` tabular, with prev and next as icon buttons.
- **E4 Status pill.** When ready, it shrinks to the dot, and the text shows on hover. Busy and
  error keep their text (error in `--danger`).
- **E5 Settings panel.** A `.panel` 320 px wide: a 14 px 600 title with a close icon. Each row has
  its label in `--text-2` and its value in `--text-1` tabular, with a `.ui-range`. Selects are
  `.ui-select`, and "Erase all pages…" is a `.ui-btn.sm.danger`. The key hints go in a
  `.ui-caption` footer with kbd.
- **E6 Editor popover.** The textarea is a `.ui-field.mono`, and the preview is 20 px. The actions
  fit on one row: Apply (`.primary`, Enter), then Re-recognize, Copy and Send to 3D as
  `.ui-btn.sm.soft`, and Keep as ink and Delete behind a "more" `.ui-menu`, or Delete as a danger
  ghost at the right end. 480 px wide.

P2:
- **E7** The image outline for selection or editing is 2 px dashed `--text-3` (it was
  `--accent`). The laser stays red (it is a pointer, not data).

### F. 3D tab (`graph/*`, style.css section 6, the `#graph` markup in `index.html`)

P1:
- **F1 Side panel.** `--dock`, with a `--line-1` right border. Its padding-top still leaves room
  for the tabs (58 px).
- **F2 Header.** Buttons become `.ui-btn.sm` with icons: plus "Add", fit, home "Reset view",
  help, and chevron-left to collapse.
- **F3 Feature buttons** (`#g-tools`, 11 buttons over 3 rows today). Group them by meaning:
  - camera: a `.ui-seg.sm` (Iso, Top, Front, Side) plus Ortho and 2D as toggles;
  - display toggles: Right angles, TeX, Steps and Rotate as `.ui-btn.sm` toggles;
  - output: PNG, Rec, To board, Audience and Clear trails, as icon buttons or behind a "more"
    `.ui-menu`.

  This needs a `group` option on `api.addToolbarButton` in grapher.js, with the features passing
  their group. The keys stay the same.
- **F4 Rows.** A 12 px dot (it was 20), with a 2 px ring when the row is hidden. Separators are
  `--line-1`. Hovering gives `--hover`, and a focused row gets a neutral tint (not `--accent`).
  The output line is 13 px `--text-3`, and × on hover becomes the close icon.
- **F5 Sliders.** Min and max as borderless `.ui-field.sm.num`, `.ui-range`, a play icon button,
  and the speed in `--text-3`.

P2:
- **F6 Help.** `code` becomes a `--well` chip (`--font-mono` 12 px), with paragraphs at 12.5 px
  `--text-2`, grouped under `.ui-overline` heads.
- **F7 View overlays.** dual-head, lec-overlay, the drag readout (a `.ui-tip`) and pp-rec (the
  `--danger` dot) take the tokens, and labels get one halo.
- **F8** `#g-expand` becomes a float `.ui-btn.icon`. The series palette stays: it is data.

## 8. File ownership and working rules

| area | owns | may read |
|---|---|---|
| foundation (done) | style.css sections 1 to 4, `static/icons.js`, `#tabs` / `#toast` / the head and scripts of `index.html`, this file | everything |
| **A** Net canvas + inspector | `nn/view.js`, `nn/view.css`, `nn/inspector.js`, `nn/inspector.css` | store.js, focus.js, model.js |
| **B** matrix panel | `nn/matrix.js`, `nn/matrix.css` | the same |
| **C** Net floating panels | `nn/train.*`, `nn/attnviz.*`, `nn/surf3d.*`, `nn/lens.*`, `nn/tour.*` | the same |
| **D** Net shell + 3D view | `nn/nn.js`, `nn/nn.css` (including its `body.clean #nn` rules), `nn/view3d.*`, the `#nn` markup | the same |
| **E** Board tab | `app.js`, style.css section 5, the `#toolbar` / `#status` / `#editor` / `#settings` markup | |
| **F** 3D tab | `graph/grapher.js`, `graph/scene.js`, `graph/features/*.js`, style.css section 6, the `#graph` markup | |

Working rules:
- **Edit only your own files.** In style.css and index.html, edit only your own section, with
  exact-string edits: never rewrite the whole file, because others edit it at the same time.
- The foundation (tokens, `ui-*`, icons) is read-only for A to F. If something is missing, use
  the closest component and ask for it in your report. Don't fork a local variant.
- **Don't rename classes that other files style.** nn.css's clean-view rules target
  `.nn-insp-btn`, `.nn-add`, `.nn-insp-head`, `.nn-insp.pinned`, `.nn-train` and `.nt-head`,
  `.nt-ctl`, `.nt-warn button`, `.nt-go-mini`, `.nt-grid`, `.nt-hint`, `.nnv-handle` and `.nnv-hint`.
  Keep them, adding `ui-*` classes next to them. New chrome that must hide under H gets
  `.ui-chrome`.
- `.nm-audience`, `.ro`, `#nn.nn-audience` and `body.clean` must look right: the audience sees
  the content without the chrome.
- Keep docs/NN_CONTRACT.md (and NN_LENS.md, NN_3D*.md) true for what they state (the toolbar,
  clean view, keys, CSS loading, colours). The owner of a change updates those lines.
- Remove nothing. Every control stays reachable, every key works, and the audience mirror and H
  keep working.
- No new dependencies and no build step. Match the code style around you. No em-dashes.

**Definition of done for each area.** Check at 1600 × 900 and 1280 × 800, in both themes, with H,
and in an `?audience` window. `node --test "tests/*.test.mjs"` stays green. There is no
`backdrop-filter` and no new per-frame work on the canvases. The browser checks use Python
Playwright with Chromium args `--use-angle=swiftshader --enable-unsafe-swiftshader`, against
`python -m http.server <spare port>` on `static/`. Never use server.py or port 8791. Before and
after screenshots go under `%TEMP%\mathboard-design\`.

## 9. Audit notes (before)

The before screenshots are in `%TEMP%\mathboard-design\before\` (`d-` dark, `l-` light):
board, 3D, Net xor / mlp / transformer / words, cards, panels, menus, clean and audience. The
recurring problems:
- **Blue everywhere.** The UI accent (`#5ac8fa`) sat next to positive-blue data: "Labels" on,
  Play, the Arcs segment, the 3D tab's toggles and the matrix's blue cells all used it.
- **On states in five styles.** Accent border plus tint (toolbar), solid accent with white text
  (matrix bar, segmented controls), bold with a tint (3D plots stops), a yellow ring (chips), and
  an accent ring (the pinned card).
- **Boxes in boxes on the canvas.** Lanes, then header pills, then dashed group boxes, then token
  boxes, then neurons. The transformer's residual edges are 6 px blue dashed arcs. t₁ / t₂ labels
  repeat in every layer and collide with the values, and headers truncate ("H = X + Z W_O").
- **Cards a screen tall.** A node card opens every section (derivation, gradients, params), and
  covers the Train panel.
- **Heavy matrices.** Saturated blocks in monospace with a dark-text flip, outlined buttons in the
  toolbar, a full-height yellow wash on the step and trace cards, and per-cell outlines.
- **Dense panels.** The Train panel shows nine settings all the time. The lens bar is always
  shown and wraps over the Train panel at 1280. See-through surfaces (the cheat sheet, with the
  matrix panel showing through).
- **Mixed sizes and icons.** A dozen font sizes (10 to 17 px), glyph icons from assorted fonts,
  and native focus rings.

## 10. Decisions

All are implemented. Each one is a single change to reverse.
1. **Neutral chrome (no blue accent).** This is in the foundation. To bring back a coloured
   accent, change `--on`, `--on-line`, `--thumb`, `--focus` and `--accent` in style.css section 1.
2. **Quieter first run.** The lens bar is hidden by default (C13). The Train panel opens with its
   body, but its hyperparameters are folded into Settings (C3); it is not folded to its head. Both
   are UI state in localStorage, not in the net.
3. **Inspector defaults** (A14): every section starts folded, remembered per section kind.
4. **Matrix cells** (B3): the UI font instead of monospace, and the alpha cap of 0.72 (the most
   saturated cells get a little lighter).
5. **Fixed edges go neutral** (A4). They have no weight colour (w = 1 drew them full blue).
6. **Token labels once per row** (A3), rather than in every layer.
7. **The 3D tab toolbar groups** (F3), with a `group` option on `addToolbarButton`.
8. **The board status pill** shrinks to its dot when ready (E4).
9. **The Flow view takes the matrix panel's place** while it is open (it draws the matrices
   itself); closing it brings back the panel as the user had it (`ctx.matrixAway`, not saved).
10. **3D view numbers** (D7): the 1.2 button always shows them when on, shrunk to fit a face
   narrower than 28 px (without the leading 0, 7.5 px at the least). It starts on in Tensor, where
   the numbers are the story, and off in Heads and Stack; a hovered or lit cell shows its number
   either way.
