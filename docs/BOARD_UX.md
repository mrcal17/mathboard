# Board UX: grouping, timing, preview and corrections

How the handwriting board decides what belongs together, when it reads it, how a reading becomes
typeset, and how you fix it. Written 2026-09-25 for the upgrade that follows
[BOARD_DIAGNOSIS.md](BOARD_DIAGNOSIS.md) (the measured causes) and [BOARD_RESEARCH.md](BOARD_RESEARCH.md)
(the patterns). The server side is in [BOARD_BACKENDS.md](BOARD_BACKENDS.md).

Code: `static/app.js` (state, rendering, input), `static/board/geom.js` (grouping, scratch-out, lasso;
tested in `tests/board_geom.test.mjs`) and `static/board/tex.js` (symbol mapping, look-alikes, locks,
replies; tested in `tests/board_tex.test.mjs` against the vendored KaTeX). Both modules are classic
scripts loaded before `app.js`; they register `globalThis.mathboardBoard.geom` and `.tex`.

## 1. Grouping

All distances are in writing sizes **S**: the median height of the page's last 30 Math glyphs.
Strokes in writing order become glyphs (a stroke that overlaps the previous glyph by a quarter of
the smaller box joins it, like the two strokes of an x); dots under 6 px (4 px for a pen) and flat
strokes (more than 4 times wider than tall: bars, minus, the halves of =) are skipped. Until a page
has 3 glyphs, S is the last value measured with the same pointer type, else the profile's default
(mouse 72 px, pen 44 px, touch 56 px). S is clamped to 20 to 180 px and is measured before the new
stroke counts. For this user's mouse writing (symbols about 75 px) the rules below give:

| Rule | Size | At S = 75 px |
|---|---|---|
| Sideways reach from an expression's visible box | 0.9 S | 68 px |
| Vertical reach | 0.35 S | 26 px |
| Burst: the expression you wrote in less than 2.5 s ago (pen 1.5 s, touch 1.8 s) reaches further sideways | 2 x 0.9 S | 135 px |
| Script zone: a stroke at most 0.8 S tall, right of an expression's right end (from 0.5 S inside it), may float above or below it | 0.6 S | 45 px |
| Fraction bar: at least 1.3 S wide and 4 times wider than tall | 1.3 S | 98 px |
| A bar gathers every expression whose middle is within 1.2 S above or below it and which lies at least 30% over its span (plus 15% of its width on each side) | 1.2 S | 90 px |
| A later stroke whose centre lands in a bar's zone joins the bar's expression | | |
| New line: a stroke entirely below an expression that starts left of it, or within 0.3 S of its left edge, doesn't join it by reach | 0.3 S | 23 px |

- The new-line guard also holds for the line you're on: a stroke that reaches both a line and an
  expression that started as a new line under it stays with the new one (`b = 1` under `A = [...]`).
- Bars win over the guard, so a denominator that starts left of the numerator still joins.
- The Grouping reach slider multiplies every reach, including the bar zones.
- Grouping uses what you see: a converted expression's typeset box (plus any ink not yet converted
  into it), not its hidden ink box. The typeset is sized like the ink: the KaTeX glyph area (the
  struts, not the line height) is fitted to the ink's height (at least 0.75 S, so a lone = isn't
  tiny) and at most 1.1 times its width, capped by *Max typeset size* (default 110 px, was 56). It
  starts at the ink's left edge and is centred on it vertically.
- **Live outline.** While a Math stroke is drawn, every 4th point re-runs the join rule on the
  stroke so far, and a dashed `--text-3` outline surrounds the expression(s) it will join,
  together with the stroke. Nothing is outlined when it will start a new expression. Hidden by H.

## 2. When an expression is read

| Event | What happens |
|---|---|
| Pen-down outside a waiting expression's near zone (its reach, script zone and bar zones; the burst zone for the one you just wrote in) | "Moved on": it is read at once, even while you draw the new stroke |
| Pen idle on it | read after *Convert after* (mouse 1200 ms by default, was 700; pen 75% = 900 ms; touch 85% = 1020 ms) |
| Pointer hovering in its near zone | the idle clock stops, up to a ceiling after its last change: mouse 3 s, pen 2.4 s (two thirds of that in Preview: 2 s and 1.6 s); touch can't hover |
| **Enter** (or the sparkle button) | every waiting expression on the page is read now and committed |
| **Hold** (P, the pause button) | nothing is read until you release it (it then converts within 0.25 s) or press Enter. The status pill says Hold |
| Settings saved with the old defaults (700 ms, 56 px) | move to the new ones once; values you had changed are kept |

The pointer type of the last pen-down (`pointerType` mouse, pen or touch) picks the profile in
`geom.PROFILES`: default size, idle factor, hover ceiling, burst time, dot size, scratch-out
reversals and tap tolerance.

**The request** (`POST /api/recognize`, docs/BOARD_BACKENDS.md section 5): the image as before, plus
`symbol_px` (the expression's median glyph height times the raster scale), `strokes` (each point
`[x, y, t_ms]` in image pixels, times from the expression's first stroke; points now store their
time as a 4th number, older boards send 0), and `backend` when you picked one in *Settings →
Recognizer*. The raster scale also keeps symbols at least 72 px tall, within 1.5 MP; lines are 8%
of a symbol (5 px at 64 px). The server normalizes the rest.

**The reply** is read by `tex.reading()`, which accepts the old shape (`{ latex, raw, ms, model }`) and
the new one. `empty` covers `latex` only: in ensemble mode the other candidate's text is offered.
An empty matrix or NONE counts as empty too (the old server sends those). Nothing recognized keeps
the ink (and the typeset an extended expression already had).

## 3. Preview, Auto and Off

*Settings → Convert*: **Preview** (the default), **Auto** (the old replace-in-place) or **Off**.

- **Preview.** A reading appears in a chip under the ink (`--float`, 0.4 S font, 16 to 30 px); the
  ink stays. Commit it by clicking the chip, pressing **Enter**, or, with *Commit when you write
  elsewhere* on (default), by starting a stroke outside it: a waiting reading commits at pen-up, one
  still on its way commits when it arrives. **Esc** drops the reading and keeps the ink; the
  expression then stays ink until it changes (Ctrl+Z brings the reading back).
- **Auto.** A sure reading replaces the ink at once, as before.
- **Off.** Expressions are grouped (with their dashed outline) but only read on Enter, a lasso
  Group or Re-recognize, and then committed.

In every mode a reading waits for you, and never commits on its own, when:

- **the ensemble disagreed** (`agree: false` with two readings): the chip says *Pick one* and shows
  both, the server's pick first (Uni-MuMER in ensemble mode), each tagged with its recognizer and
  number. Click one or press **1** / **2**. Enter and moving on leave it alone;
- **only one recognizer answered** (`agree: false`, one candidate: the other saw nothing or
  failed): *Unsure*, with the recognizer's name;
- **the expression has a hand edit**: the new reading is a *Use* suggestion. Your LaTeX stays until
  you click it.

The choice chips have an edit button for typing it yourself. Esc and the number keys act on the
chip under the pointer, else the reading nearest the pointer, else the newest. Chips are
`.ui-chrome`: hidden by H and in the audience window, like the waiting outlines. Export includes
readings still waiting under their ink.

## 4. Corrections

- **Hand edits stick.** Apply in the editor marks the expression manual: strokes added later bring
  the model's reading as a suggestion only. Revealing the ink (the eraser on typeset, erasing or
  splitting strokes) drops the hand edit, since the ink changed.
- **Re-recognize** changes its input. When the server offers two single recognizers it asks the
  other one than the one that produced the current reading (`backend` override). Otherwise it
  alternates two renderings: symbols 1.5x larger with 1.4x lines, and 0.7x with 0.8x lines. The
  `symbol_px` sent stays the plain value, so a normalizing server renders the symbols 1.5x or 0.7x
  its target. The same image at temperature 0 gives the same answer, which is why. If the reading
  comes back the same, a toast says so.
- **Look-alikes.** Tap a typeset symbol with the Math pen (a press that moves at most 4 px and
  lasts at most 350 ms with a mouse; pen 6 px / 300 ms, touch 10 px / 300 ms) and a menu offers its
  look-alikes from `tex.LOOKALIKES`: 2 / z / x / alpha / lambda / infinity, infinity / alpha /
  propto, 1 / l / | / I, 0 / o / O / theta, d / partial / delta, and so on, up to 6 plus the other case
  of a letter. **1** to **9** or a click picks one; **Edit LaTeX** opens the editor. Select-tapping
  an expression still opens the editor, as before.
- **How a tap finds its token.** `tex.scan()` wraps each tappable token (single letters and digits,
  `+ - = < > ( ) | / !`, Greek letters and single-glyph symbols such as `\infty`, `\partial`,
  `\times`) in `\htmlData{mbt=K}{...}`, adding braces where the token is a command argument or a
  script (`x^2`, `\frac12`). Names, text, environments, delimiters after `\left`, fonts and colours
  are left alone, and sources that already use `\html...` are not wrapped. KaTeX renders that with
  `trust` limited to `\htmlData`, so each glyph carries `data-mbt="K"`; the smallest one under the
  tap is the token. If the wrapped source doesn't render, the typeset is simply not tappable.
- **Locks.** A pick is stored on the expression as `{ from, to, pos, n, l, r }`: the model's token,
  your pick, its position among the tappable tokens, their count and its neighbours. Every new
  reading (also each candidate of a choice) gets the locks applied first: where the reading has
  `from` at the same position, or shifted by the change in length when strokes were added in front,
  it becomes `to`; the best-placed match wins, and a reading that already has `to` there is left
  alone. Locks are dropped when two expressions merge or you apply a hand edit.

## 5. Regrouping and erasing

- **Lasso.** With Select, drag on empty board. Ink with at least 60% of its points inside is
  selected, and a converted expression whose typeset is at least half inside, with all its strokes.
  The selection gets an `--hi` halo (a tint over typeset) and a 2 px dashed `--text-3` box, and
  a bar under it:
  - **Group**: the selected math becomes one new expression, read now;
  - **Split**: from each expression that is only partly selected, the selected strokes become their own;
  - **To math**: selected Draw-pen ink becomes math and one expression with any selected math;
  - **Keep as ink**: the selected math becomes Draw-pen ink;
  - the bin, or **Delete** / **Backspace**: delete.

  The expressions that lost strokes show their ink and are read again. Drag inside the selection
  to move it (whole expressions keep their typeset). Esc, a click elsewhere, a page change or H
  drops the selection.
- **Scratch-out.** A Math stroke that zig-zags (at least 3 sideways reversals for a mouse, 4 for a
  pen or touch, each a quarter of the width, with a path 2.5 times the width; up-and-down
  zig-zags need 2 more and must be packed closer than a fifth of their height each, so m, w and M
  don't count) erases the ink it crosses: strokes it crosses twice, or once when they are mostly
  inside it, dots inside it, and converted expressions whose typeset it covers by half. When it
  crosses nothing it is an ordinary stroke. Ctrl+Z brings the ink back. The Draw pen never
  scratches out, so hatching stays hatching.

## 6. Keys and gestures

| Input | Board |
|---|---|
| Enter | read every waiting expression now; commit the sure readings waiting under the ink |
| P | Hold on / off |
| Esc | close the open panel or menu, else drop the lasso selection, else keep the ink of a reading |
| 1 to 9 | pick a look-alike in the open menu, else reading n of the targeted choice |
| Delete, Backspace | delete the lasso selection |
| Math pen tap on typeset | look-alike menu |
| Math pen zig-zag over ink | erase it |
| Select tap | editor (as before) |
| Select drag on empty board | lasso |
| Select drag inside a selection | move it |
| Click a reading chip | commit it (or that reading of a choice) |

All earlier keys (M, D, E, S, L, H, T, N, arrows, Ctrl+Z / Y) are unchanged.

## 7. Data and compatibility

New block fields, all optional: `pv` (a waiting reading `{ latex, cands, suggest, unsure, by, at }`),
`manual`, `locks`, `by` (the backend of the last reading), `t0`, `movedOn`, `force`, `forceCommit`,
`commitOnResult`, `rv` / `rb` / `reruns` (Re-recognize). Strokes gain `t0` and a 4th number per
point. Saved boards and undo snapshots from before load unchanged: blocks without these fields
behave as before. Settings gain `mode`, `commitOnMove`, `backend` and `v`.

## 8. Not verified

- Everything above was checked in Chromium with the mouse, first against a fake `/api/*`
  (`page.route`) and then against a real `server.py` in ensemble mode: grouping of fractions,
  exponents, matrices and multi-line writing, the timing, Hold and Enter, the old and the new reply
  shapes, choices, Esc, the lasso actions, the live outline, scratch-out, look-alikes and locks, both
  themes, H and the audience window. An idle chip appears about 2.3 to 2.6 s after the last pen-up
  (1.35 to 1.44 s of waiting, then the ensemble's 0.9 to 1.4 s). Mouse strokes drawn by a script
  say nothing about accuracy on real handwriting.
- The real run fixed: a stroke that starts on a reading chip is handed to the board (a tap still
  clicks the chip); a stale chip no longer commits over newer strokes; Esc drops a reading that is
  still in flight; typeset is sized from the glyphs, not KaTeX's line box, so it matches the ink.
- Chips can overlap neighbouring typeset, and the status pill names the default mode's model even
  when Settings picks one backend.
- Pen and touch profiles were only exercised with synthetic pointer events; the numbers for them
  are estimates until someone writes with a tablet.
- The grouping numbers come from the diagnosis (this user's mouse writing) and the synthetic
  handwriting of the checks. Sums with limits written under the sign (`i=1` under a Σ that starts
  left of it) are split by the new-line guard; lasso Group joins them.
