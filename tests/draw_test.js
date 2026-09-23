// End-to-end check for the Playwright MCP `browser_run_code` tool: draws crude "handwriting"
// with real mouse events, waits for conversion, and returns what each expression became.
async (page) => {
  const arc = (cx, cy, r, a0, a1, n = 14) => Array.from({ length: n + 1 }, (_, i) => {
    const a = ((a0 + ((a1 - a0) * i) / n) * Math.PI) / 180;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  });
  const G = { // unit-box polylines, y down
    '1': [[[0.25, 0.22], [0.55, 0], [0.55, 1]]],
    '2': [[...arc(0.5, 0.3, 0.3, 200, 380), [0.12, 1], [0.9, 1]]],
    '3': [[...arc(0.45, 0.25, 0.25, 200, 450), ...arc(0.45, 0.73, 0.27, 270, 520)]],
    '4': [[[0.65, 1], [0.65, 0], [0.05, 0.7], [0.95, 0.7]]],
    'x': [[[0.1, 0.35], [0.9, 1]], [[0.9, 0.35], [0.1, 1]]],
    '+': [[[0.5, 0.3], [0.5, 0.95]], [[0.18, 0.62], [0.82, 0.62]]],
    '=': [[[0.12, 0.5], [0.88, 0.5]], [[0.12, 0.78], [0.88, 0.78]]],
    '[': [[[0.75, 0], [0.3, 0], [0.3, 1], [0.75, 1]]],
    ']': [[[0.25, 0], [0.7, 0], [0.7, 1], [0.25, 1]]],
    '-': [[[0, 0.5], [1, 0.5]]],
  };
  const jit = () => (Math.random() - 0.5) * 2.5;
  async function stroke(pts) {
    await page.mouse.move(pts[0][0], pts[0][1]);
    await page.mouse.down();
    for (let i = 1; i < pts.length; i++) await page.mouse.move(pts[i][0], pts[i][1], { steps: 3 });
    await page.mouse.up();
  }
  async function glyph(ch, x, y, w, h) {
    for (const s of G[ch]) await stroke(s.map(([u, v]) => [x + u * w + jit(), y + v * h + jit()]));
  }
  async function text(str, x, y, size) { // '^' makes the next char a superscript
    let cx = x;
    for (let i = 0; i < str.length; i++) {
      let ch = str[i], sup = false;
      if (ch === '^') { sup = true; ch = str[++i]; }
      const w = size * 0.6 * (sup ? 0.6 : 1), h = size * (sup ? 0.55 : 1);
      await glyph(ch, cx, sup ? y - size * 0.3 : y, w, h);
      cx += w + size * 0.22;
    }
  }

  await page.evaluate(() => { localStorage.clear(); });
  await page.reload();
  await page.waitForFunction(() => window.mathboard && window.mathboard.server.ready, null, { timeout: 120000 });
  await page.keyboard.press('m');

  const t0 = Date.now();
  await text('x^2+1=3', 120, 150, 56);                 // superscript
  await glyph('1', 720, 110, 28, 45);                  // fraction 1/2
  await glyph('-', 700, 160, 72, 20);
  await glyph('2', 720, 187, 28, 45);
  await glyph('1', 190, 350, 30, 50);                  // matrix: entries first...
  await glyph('2', 260, 350, 30, 50);
  await glyph('3', 190, 430, 30, 50);
  await glyph('4', 260, 430, 30, 50);
  await glyph('[', 150, 335, 30, 160);                 // ...then brackets merge the rows
  await glyph(']', 310, 335, 30, 160);
  const drawn = Date.now();

  await page.waitForFunction(() => {
    const bs = window.mathboard.pages[0].blocks;
    return bs.length && bs.every(b => b.status !== 'pending' && b.status !== 'busy');
  }, null, { timeout: 60000, polling: 100 });
  const done = Date.now();

  const blocks = await page.evaluate(() => window.mathboard.pages[0].blocks.map(b => ({
    status: b.status, latex: b.latex, box: b.box.map(Math.round),
  })));
  return { drawMs: drawn - t0, settleMs: done - drawn, blocks };
}
