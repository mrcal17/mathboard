# Mathboard 3D tab: guide for feature authors

Mathboard is a local lecture tool for teaching linear algebra and neural networks.
`static/index.html` has three tabs: a handwriting board (`static/app.js`), a Desmos-style 3D vector
grapher (`static/graph/`) and a neural-network visualizer (`static/nn/`, whose module contract is
`docs/NN_CONTRACT.md`). This guide is for adding one feature module to the 3D tab. Read the
README's "3D tab" section, then skim the headers of the files below.

| File | Role | Who edits |
|---|---|---|
| `static/graph/lang.js` | parser + evaluator; `registerFunction`, `registerType`, `registerConstant`, `values` helpers, `parseLine` (AST) | core: not in a feature change |
| `static/graph/linalg.js` | numeric core: rref, eliminationSteps, rank, nullspace/colspace/rowspace/leftNullspace, solve, lstsq, gramSchmidt, qr, lu, charpoly, eig, svd, det, inv, matmul, matvec... | core: not in a feature change |
| `static/graph/scene.js` | Three.js view; `registerRenderer(kind, fn)`, ctx helpers, camera API | core: not in a feature change |
| `static/graph/grapher.js` | expression list UI; builds the `api` passed to features | core: not in a feature change |
| `static/graph/features/<name>.js` | **your module** | you |
| `tests/<name>.test.mjs` | node:test tests for your pure logic | you |

**Only create or edit the files your task names.** Never edit the shared files, `index.html`,
`style.css`, or another feature's files. If an API you need is missing, work around it inside
your own module (e.g. build DOM yourself, inject CSS with `api.addStyles`) and list the gap in
your report. Match the house style: 2-space indent, single quotes, semicolons, sparse comments.

## Module contract

```js
// static/graph/features/<name>.js
import { registerFunction, registerType, values } from '../lang.js';
import { registerRenderer, THREE } from '../scene.js';
import * as la from '../linalg.js';

export function install(api) { /* hooks, buttons, keyboard, ... */ }
```

Language functions/types should be registered at module top level or in `install` (both run
before the first evaluation). Features load in this order and failures are isolated:
`plots, transform, fields, combos, systems, dual, lecture, present, bridge, drag`.

### Language (`lang.js`)
- Values: numbers; `{type:'vec', v:[x,y,z]}`, `{type:'point', v}`, `{type:'mat', m: rows}`,
  `{type:'span', vecs}` (independent vectors only), `plane{normal}`, `parallelogram{u,v}`,
  `parallelepiped{u,v,w}`. Build them with `values.vec([..])`, `values.mat(rows)` etc.
- `registerFunction(name, { n: count | [min, max], kind?: 'num'|'vec'|'mat'|..., f: (args, name) => value })`.
  `kind` checks every argument; omit it and check yourself for mixed arguments. Throw `Error`
  with a short human message on bad input; it is shown under the row.
- `registerType(type, { describe: 'a transformation', format(v) -> text, latex(v) -> KaTeX for the
  row readout (shown after "= "), readout(v) -> {latex} or {text} shown as is, drawable: false for
  readout-only values (or a function of the value), numbers(v) -> number[] (NaN/inf check) })`.
- **Graphs.** A row that uses `x`, `y` or `z` without defining them evaluates to
  `{type: 'graph', mode: 'curve' | 'surface' | 'param', ins, dep, at(env)}` (`y = x^2` has
  `ins: ['x']`, `dep: 'y'`; `at({x: 2})` is 4). `f(x) = ...` rows are graphs with `callable`,
  `params` and `call(args)`; a bare `softmax` is `{type: 'softmaxmap', T}`. Rows are compiled once
  into closures, so sampling `at` thousands of times per frame is cheap. `features/plots.js` draws
  both kinds; `transform.js` carries them by adding the matrix as `item.M`. `parseLine(line, userFns)`
  takes the set of `f(x) = ...` names so `f(2)` parses as a call (without it, as `f * 2`).
- A row is `[name =] expr [@ origin]`. `@` only moves where the value is drawn (`res.origin`).
- A literal number row (`t = 0.5`) becomes a slider with a play button (ping-pong over its range;
  users can edit min/max). **Use slider variables for animation parameters and steps**, e.g.
  `transform(A, t)` or `eliminate(A, b, k)` (floor `k` for discrete steps).

### Scene (`scene.js`)
Every drawable row becomes an item `{...value, kind: value.type, o: origin, color, label: name, index, rowId}`.
`registerRenderer(kind, (item, ctx) => ...)` draws it. `ctx` gives:
`THREE, E (axis half-extent), s (= E/6, size unit for thickness), theme, colors, camera,
add(...objs), own(disposable), v3([x,y,z]), mat(kind, color, opacity)` with kinds
`solid | glass | ghost | surface | line | edge`,
`arrow(o, v, color, {opacity, thickness, head})`, `dot(pos, color, r, opacity)`,
`lines(pointPairs, color, {opacity, dashed})`, `polyline(points, color, {opacity, dashed})`,
`planePatch(o, e1, e2, color, lattice|null, {size, opacity})`, `label(pos, latex, color, cls)`,
`orthoBasis(vecs)`, `placeAlong(mesh, from, dir, len, radius)` (for GEO.cyl / GEO.cone),
`GEO` (cyl, cone, sphere, box, boxEdges), `onFrame(fn(dt, t))` (per-frame, cleared on rebuild).
Positions are `THREE.Vector3`. Objects you add are disposed on rebuild if you `own()` their
geometry/material (ctx helpers already do). Content is rebuilt on every recompute (every keystroke
and every slider frame), so renderers must be cheap and stateless; keep cross-frame state
(e.g. trails) in your module keyed by `item.rowId`.

Scene instance (via `api.scene` / `api.onSceneReady`): `getPose()`, `setPose(pose, ms)`, `flyTo`,
`viewPreset('iso'|'top'|'front'|'side')`, `setOrtho(bool)`, `set2D(bool)`, `setAutoRotate(bool)`,
`setExtent(x)`, `fit()`, `onFrame(fn)` (persistent; returns unsubscribe), `pick(clientX, clientY, filter?)`
-> `{item, object, point}` (first hit passing `filter(item, object)`), `snapshot()` (PNG data URL, WebGL only; labels are DOM in
`scene.labelLayer`), `render()`, `canvas`, `controls` (OrbitControls), `camera`, `renderer`, `scene`.

### Feature API (`api`, also `window.mathboardGraph`)
`rows` (each has `src, color, hidden, min, max, id, el: {li, dot, src, out, slider, main, ...}`),
`results` (per row `{name, value, origin, error, slider}`), `scene`, `view`, `lang`, `PALETTE`,
`params` (query string), `hashParams`,
`addRow(src, {after, color, hidden, focus})`, `removeRow(row)`, `setRowSource(row, src)`,
`setRows(list)`, `recompute()`, `rowByName(name)`, `getState()` / `setState(state, {cameraMs})`,
`onRecompute(fn(results, rows))`, `addItemsHook(fn(items, {rows, results}) -> items)` (runs in
feature order), `addRowDecorator(fn(row, result, el))`, `onSceneReady(fn(scene))`,
`onViewChange(fn(view))`, `addToolbarButton({label, title, onClick, group, icon, seg})` (see below),
`addOverlay(el)` (absolute inside the 3D view), `addStyles(css)`, `setView`, `setCollapsed`,
`toast(msg)`, `icon(name)` (an SVG string from `static/icons.js`, `''` for an unknown name),
`panelEl`, `viewEl`.

**Toolbar buttons.** `addToolbarButton` returns the `<button>`: toggle it with `.on`, hide it with
`hidden` (a group with nothing shown folds away), and keep a key in its `title`. Pick a `group` by
meaning:

| group | for | looks like |
|---|---|---|
| `'camera'` | view presets, projection, linking views | its own line, first |
| `'display'` | what is drawn, and panels that open in the sidebar | text toggles (`.ui-btn.sm`) |
| `'output'` | exports and windows (PNG, recording, audience) | icon buttons at the right: give an `icon` |
| `'more'` | rarer actions | a row of the `...` menu after the output icons: the label, the icon and the title as a second line |
| any other name | anything else (the default is `'other'`) | its own group after these |

`icon` is a name from `static/icons.js` (docs/DESIGN.md, Icons); outside `'output'` it goes before
the label. `seg: 'name'` puts buttons of one group into one segmented track (`.ui-seg`), which
leads its group: mark the current choice with `.on`, as lecture.js does for Iso / Top / Front /
Side. For example:

```js
const b = api.addToolbarButton({ label: 'Trails', title: 'Show trails (T)', group: 'display', onClick: toggle });
api.addToolbarButton({ label: 'SVG', title: 'Download an SVG', group: 'more', icon: 'download', onClick: save });
```

**Help.** Append a section to `#g-help` that starts with its own head,
`<h4 class="ui-overline">Your feature</h4>`, followed by `<p>` lines with `<code>` for syntax.
Mouse and key hints go in `#g-help-keys` (with `<kbd>`), which stays last.

Keyboard: the board's shortcuts are off while the 3D tab is shown. Ignore keys typed into inputs
(`e.target.closest('input, textarea, select')`), and only act when `api.view === 'graph'`.
Colours: use the row colour (`item.color`) or `api.PALETTE`; work in both themes
(`document.documentElement.dataset.theme` is `dark` or `light`).

## Testing

1. Pure logic: `node --test tests/<name>.test.mjs` from the project dir.
2. In a browser: serve the static folder on a free port (don't start `server.py` for this: it
   loads a GPU model), then drive a private headless Chromium with Python Playwright (tested
   with 1.59). Run the script from the project root.

```python
# python test_x.py   (keep scratch scripts in your temp folder, not in the repo)
import asyncio, os, subprocess, sys, tempfile, urllib.parse
from playwright.async_api import async_playwright
PORT = 8840
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1',
                        '-d', 'static'], stderr=subprocess.DEVNULL)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await b.new_page(viewport={'width': 1400, 'height': 850})
        pg.on('pageerror', lambda e: print('PAGEERROR', e))
        pg.on('console', lambda m: m.type == 'error' and print('CONSOLE', m.text))
        rows = urllib.parse.quote('A = [[1,1,0],[0,1,0],[0,0,1]]\nt = 0.5\ntransform(A, t)')
        await pg.goto(f'http://127.0.0.1:{PORT}/index.html#graph={rows}&view=iso')
        await pg.wait_for_function("document.body.dataset.graphReady === '1'")
        await pg.wait_for_timeout(400)
        print(await pg.evaluate('mathboardGraph.results.map(r => r.error || r.name)'))
        await pg.screenshot(path=os.path.join(tempfile.gettempdir(), 'mb_xx.png'))  # then look at it
        await b.close()
try: asyncio.run(main())
finally: srv.terminate()
```

The `#graph=` hash preloads rows (newline-separated) without touching the saved list;
`&view=top|front|side|iso` sets the camera. Look at your screenshots: check that things are
visible, sized sensibly at the default extent (E = 6), labelled, and readable in both themes
(toggle with `document.documentElement.dataset.theme = 'light'`). 404s for other features'
modules that don't exist yet are expected; any other console error is yours to fix.

### README screenshots and the GIF

The images in `docs/media/` were made with the same setup at a 1600×900 viewport. Things that
help:

- **Seeding local storage.** Open any same-origin URL first (e.g. `/style.css`), set the keys,
  then load `index.html`. Going from `index.html` to `index.html#...` only changes the hash and
  doesn't reload the app. The board reads `mathboard.board` (`{pages, pageIdx, media}`; see the
  "board state" comment in `static/app.js`), and the theme lives in `mathboard.settings`
  (`{"theme": "dark" | "light"}`).
- **Status chip.** With `http.server` there is no `/api/status`, so the chip turns red. Answer it
  with `page.route('**/api/status', ...)` and a JSON body like the one `server.py` returns
  (`{ready, state, model, error, last_ms}`).
- **Framing the 3D view.** `#graph=...&view=iso` loads the rows; then
  `mathboardGraph.scene.setExtent(e)` and a `getPose()` / `setPose(pose, 0)` with a larger `zoom`.
- **Net tab.** `#nn=<preset>` loads a preset. `mathboardNet.store` and `mathboardNet.ctx.view.nodeRect(id)`
  give you nodes to click; `store.set('hover', {kind: 'edge', id})` lights an edge everywhere, and
  `{kind: 'token', layer, t}` a token (on an attention layer, with its row of A). For
  `docs/media/net-transformer.png` the transformer preset was trained with the Train panel's handle
  (`document.querySelector('.nn-train').nnTrain.play()`, speed 100, about 6000 steps), then
  `loadSample(0)`, a fit, and the matrix panel scrolled to the attention layer.
- **Animated GIF.** With the flags above, screenshots of the Net tab took about 2 s each. Adding
  `--disable-gpu-compositing` brings that under 0.1 s, fast enough to grab a frame every
  100 ms while training runs. Then:
  `ffmpeg -framerate 10 -i %04d.png -vf "split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" out.gif`.
  Lossless PNG frames keep the GIF small. The same clip made from Playwright's WebM recording
  came out about three times larger, because compression noise changes every frame.
- **Lens, Attention panel and Explain.** `docs/media/net-lens.png` is the transformer trained as
  above (then `store.net` saved as JSON and reloaded with `#nn=<URI-encoded JSON>`), the Train
  panel folded (`mathboardNet.ctx.train.fold(true)`) and key 2 pressed. `net-attention.png` is
  `words` with `mathboardNet.ctx.attnviz.show('mix')` and key 3. For `net-explain.gif`, press E
  and grab eight frames 100 ms apart after each → (the refit and the caption card settle), then
  hold the last one: an ffmpeg concat list (`file '0007.png'` / `duration 3`) with `-fps_mode vfr`
  and the palette filter above keeps the held frames free. `net-explain.png` is its last step.
- **Word presets.** `docs/media/net-words.png` is `#nn=agreement` trained with the Train panel's
  handle (speed 100, 3000 steps), then `loadSample(i)` for the sample whose `words` are
  "dog chases cats" (sample 10 at the preset's settings), the matrix panel hidden (a double-click
  on `#nn-split`), `mathboardNet.ctx.attnviz.show('arcs')`, key 2 and a fit.

## When you're done
Note what syntax/UI you added (with a 3-6 line example a lecturer would type), the files you
created, test results, and any shared-API gaps or known limitations. Keep it short.

## Reserved function names (don't register another feature's names)

| Feature | Names |
|---|---|
| core (lang.js) | dot cross norm length mag unit normalize proj angle deg rad sin cos tan asin acos atan sqrt abs exp ln log min max det inv transpose matrix point span plane parallelogram parallelepiped sinh cosh tanh sigmoid σ relu leakyrelu leaky_relu elu gelu softplus silu swish mish erf heaviside sign floor ceil round mod softmax logsumexp |
| core coordinates | x y z are free in a row that doesn't define them (the row becomes a graph) |
| combos | explain chain trail target arc shadow components crossview line plane3 intersect distance |
| transform | transform eigen svdview fixed |
| fields | flow iterate power quadric |
| systems | rowpicture colpicture eliminate lstsq subspaces gramschmidt basis coords rref rank nullspace colspace rowspace leftnull eig svd qr lu charpoly tr solve |
| dual | map |

Need another name? Pick one not in this table and mention it in your report.
