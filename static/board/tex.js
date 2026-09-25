'use strict';
// LaTeX helpers for the board, shared by app.js and the node tests (tests/board_tex.test.mjs):
// - scan(): finds the symbols you can tap in typeset maths and wraps each one in
//   \htmlData{mbt=K}{...}, so the rendered glyph carries data-mbt="K" (KaTeX is given
//   trust only for \htmlData);
// - the look-alike table behind the tap-a-symbol menu (2 / z / x, infinity / alpha / propto, ...);
// - locks, which keep a symbol you picked through later re-recognition;
// - reading(), which turns a /api/recognize reply (old or new server) into one shape.
// A classic script loaded before app.js; it registers globalThis.mathboardBoard.tex.
(() => {
  // ------------------------------------------------------------ tokens
  // [{ t, i, j }]: control words (\alpha), control symbols (\, \\), and single characters.
  // Whitespace is skipped; i / j are the token's start and end in the source.
  function tokenize(src) {
    const out = [];
    let i = 0;
    while (i < src.length) {
      const c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      let j = i + 1;
      if (c === '\\') {
        if (/[A-Za-z]/.test(src[i + 1] || '')) { j = i + 2; while (j < src.length && /[A-Za-z]/.test(src[j])) j++; }
        else j = Math.min(src.length, i + 2);
      }
      out.push({ t: src.slice(i, j), i, j });
      i = j;
    }
    return out;
  }

  const GREEK = 'alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega';
  const SYMBOLS = 'infty partial nabla propto times cdot pm mp div leq geq le ge neq ne approx sim simeq equiv to rightarrow leftarrow ell hbar emptyset varnothing in notin subset subseteq supset supseteq cup cap wedge vee forall exists perp parallel circ star ast top bot mapsto';
  // Tappable command tokens: one glyph each, no arguments.
  const SYM_CMDS = new Set(`${GREEK} ${SYMBOLS}`.split(' ').map(n => '\\' + n));
  const SYM_CHAR = /^[A-Za-z0-9+\-=<>()|/!]$/;
  // Commands whose arguments are ordinary maths: an unbraced tappable argument gets braces.
  const ARITY = {
    '\\frac': 2, '\\dfrac': 2, '\\tfrac': 2, '\\cfrac': 2, '\\binom': 2, '\\dbinom': 2, '\\tbinom': 2,
    '\\overset': 2, '\\underset': 2, '\\stackrel': 2,
    '\\sqrt': 1, '\\overline': 1, '\\underline': 1, '\\hat': 1, '\\widehat': 1, '\\bar': 1, '\\vec': 1,
    '\\dot': 1, '\\ddot': 1, '\\tilde': 1, '\\widetilde': 1, '\\check': 1, '\\widecheck': 1, '\\breve': 1,
    '\\acute': 1, '\\grave': 1, '\\mathring': 1, '\\overrightarrow': 1, '\\overleftarrow': 1,
    '\\overleftrightarrow': 1, '\\underbrace': 1, '\\overbrace': 1, '\\boxed': 1, '\\cancel': 1,
    '\\bcancel': 1, '\\xcancel': 1, '\\boldsymbol': 1, '\\bm': 1, '\\pmb': 1, '\\xrightarrow': 1,
    '\\xleftarrow': 1, '\\phantom': 1, '\\hphantom': 1, '\\vphantom': 1, '\\smash': 1,
  };
  // Commands whose (first) argument is copied as is: names, text, colours, fonts.
  const RAW = new Set(['\\begin', '\\end', '\\text', '\\textrm', '\\textit', '\\textbf', '\\textsf', '\\texttt',
    '\\textnormal', '\\textup', '\\emph', '\\mathrm', '\\mathit', '\\mathbf', '\\mathsf', '\\mathtt', '\\mathcal',
    '\\mathbb', '\\mathfrak', '\\mathscr', '\\operatorname', '\\color', '\\textcolor', '\\colorbox', '\\label',
    '\\tag', '\\hspace', '\\vspace', '\\href', '\\url', '\\mbox', '\\hbox', '\\ce', '\\pu']);
  // The token after these is a delimiter and must stay bare.
  const DELIM = new Set(['\\left', '\\right', '\\middle', '\\big', '\\Big', '\\bigg', '\\Bigg', '\\bigl', '\\bigr',
    '\\Bigl', '\\Bigr', '\\biggl', '\\biggr', '\\Biggl', '\\Biggr', '\\bigm', '\\Bigm', '\\biggm', '\\Biggm']);
  const TWO_ARG_ENVS = /^(array|darray|subarray|alignat\*?|alignedat)$/;
  const isTappable = t => SYM_CMDS.has(t) || SYM_CHAR.test(t);

  // Walk the source once: taps are the tappable tokens in order, tex the source with each of them
  // wrapped in \htmlData{mbt=K}{...}. tex is null when the source already uses \html... or \verb.
  function scan(src) {
    src = String(src || '');
    const toks = tokenize(src), taps = [];
    const bail = /\\(html|verb|href|url|includegraphics)/.test(src);
    let out = '', prev = 0, k = 0;
    const stack = [{ need: 0 }];
    const top = () => stack[stack.length - 1];
    const emit = (tok, text = tok.t) => { out += src.slice(prev, tok.i) + text; prev = tok.j; };
    const slot = () => { const f = top(); if (f.need > 0) { f.need--; return true; } return false; };
    // Copy the next unit (a braced group or one token) as it is; returns the index after it.
    function raw(at) {
      if (at >= toks.length) return at;
      if (toks[at].t !== '{') { emit(toks[at]); return at + 1; }
      let depth = 0, e = at;
      for (; e < toks.length; e++) {
        if (toks[e].t === '{') depth++;
        else if (toks[e].t === '}' && --depth === 0) break;
      }
      e = Math.min(e, toks.length - 1);
      out += src.slice(prev, toks[e].j);
      prev = toks[e].j;
      return e + 1;
    }
    // An optional [..] argument (\sqrt[3], \\[4pt]) is copied as it is.
    function optional(at) {
      if (toks[at]?.t !== '[') return at;
      let e = at;
      while (e < toks.length && toks[e].t !== ']') e++;
      e = Math.min(e, toks.length - 1);
      out += src.slice(prev, toks[e].j);
      prev = toks[e].j;
      return e + 1;
    }
    for (let n = 0; n < toks.length;) {
      const tok = toks[n], t = tok.t;
      if (t === '{') { slot(); stack.push({ need: 0 }); emit(tok); n++; continue; }
      if (t === '}') { if (stack.length > 1) stack.pop(); emit(tok); n++; continue; }
      if (t === '^' || t === '_') { emit(tok); top().need++; n++; continue; }
      if (DELIM.has(t)) { slot(); emit(tok); n = raw(n + 1); continue; }
      if (RAW.has(t)) {
        slot();
        emit(tok);
        n++;
        if (toks[n]?.t === '*') { emit(toks[n]); n++; }
        const env = t === '\\begin' ? src.slice(toks[n]?.j ?? 0, (toks.find((x, i) => i > n && x.t === '}') || {}).i ?? 0) : '';
        n = raw(n);
        if (TWO_ARG_ENVS.test(env)) n = raw(n);
        if (t === '\\textcolor') top().need++;
        continue;
      }
      if (t in ARITY) { slot(); emit(tok); n = optional(n + 1); top().need += ARITY[t]; continue; }
      if (t === '\\\\') { slot(); emit(tok); n = optional(n + 1); continue; }
      if (isTappable(t)) {
        const arg = slot();
        taps.push({ t, i: tok.i, j: tok.j });
        const w = `\\htmlData{mbt=${k++}}{${t}}`;
        emit(tok, arg ? `{${w}}` : w);
        n++;
        continue;
      }
      slot();
      emit(tok);
      n++;
    }
    out += src.slice(prev);
    return { tex: bail ? null : out, taps };
  }
  const taps = src => scan(src).taps;

  // Put `to` in place of one tapped token, with a space where a control word would run into a
  // letter (\cdot2 -> \cdot z, 2x -> \infty x).
  function replaceAt(src, tap, to) {
    let before = src.slice(0, tap.i), after = src.slice(tap.j);
    if (/^[A-Za-z]/.test(to) && /\\[A-Za-z]+$/.test(before)) before += ' ';
    if (/^\\[A-Za-z]+$/.test(to) && /^[A-Za-z]/.test(after)) after = ' ' + after;
    return before + to + after;
  }

  // ------------------------------------------------------------ look-alikes
  // Ordered by how often the pair is confused in handwriting (the first entries are the ones seen
  // in the board's own debug images: a looped 2 read as alpha, x, lambda or infinity; d read as a
  // partial). Every entry is itself a tappable token, so a pick can be picked again.
  const LOOKALIKES = {
    '2': ['z', 'x', '\\alpha', 'Z', '\\lambda', '\\infty'],
    'z': ['2', 'Z', '3', '\\zeta'],
    'Z': ['2', 'z', '\\zeta'],
    'x': ['2', '\\times', 'X', '\\chi', 'y', 'k'],
    'X': ['x', '\\times', '\\chi', 'K'],
    '\\times': ['x', 'X', '+'],
    '\\infty': ['2', '\\alpha', '\\propto', '8', '\\partial'],
    '\\alpha': ['2', 'a', '\\infty', '\\propto', '\\partial'],
    '\\propto': ['\\alpha', '\\infty', '2'],
    '\\lambda': ['2', '\\alpha', 'h', 'A'],
    '1': ['l', '|', 'I', '7', '/'],
    'l': ['1', '|', 'I', '\\ell', 'e'],
    '|': ['1', 'l', 'I', '/'],
    'I': ['1', 'l', '|', 'T'],
    '/': ['1', '|', 'l'],
    '\\ell': ['l', '1', 'e'],
    '7': ['1', 'T', '>'],
    '0': ['o', 'O', '\\theta', 'D', '6'],
    'o': ['0', 'O', 'a', '\\circ', '\\sigma'],
    'O': ['0', 'o', 'D', 'Q', '\\Theta'],
    '\\circ': ['o', '0', '\\cdot'],
    '\\theta': ['0', 'O', '\\Theta', '\\phi'],
    '\\Theta': ['\\theta', 'O', '0'],
    'd': ['\\partial', '\\delta', 'a', '6', 'o'],
    '\\partial': ['d', '\\delta', '\\sigma', '2', '6', 'o'],
    '\\delta': ['\\partial', '\\sigma', 'd', '8', '6'],
    '\\sigma': ['\\delta', '\\partial', '6', 'o', 'b'],
    'a': ['\\alpha', 'o', 'd', 'q', '9'],
    '6': ['b', '\\delta', '\\sigma', 'G', '0'],
    'b': ['6', 'h', '\\beta', 'D'],
    '\\beta': ['B', 'b', '\\partial', '8'],
    'B': ['8', '\\beta', 'R', '3'],
    '8': ['B', '\\infty', '3', '\\delta', 'g'],
    '3': ['8', 'B', 'z'],
    '5': ['s', 'S'],
    's': ['5', 'S'],
    'S': ['5', 's'],
    '9': ['g', 'q', '4', 'a'],
    'g': ['9', 'q', 'y'],
    'q': ['9', 'g', 'a'],
    '4': ['y', '9', 'A', '\\psi'],
    'y': ['4', 'g', '\\gamma', 'x'],
    '\\gamma': ['y', 'r', '\\nu'],
    't': ['+', 'f', '\\tau'],
    '+': ['t', '\\pm', '\\times'],
    '=': ['\\equiv', '\\approx', '-'],
    '-': ['=', '\\sim'],
    '<': ['\\leq', '(', '\\in'],
    '>': ['\\geq', ')', '7'],
    '\\leq': ['<', '\\subseteq'],
    '\\geq': ['>', '\\supseteq'],
    '(': ['1', 'C', 'c', '<'],
    ')': ['1', '>'],
    'u': ['v', '\\mu', 'n', 'U'],
    'v': ['u', '\\nu', 'r', 'V'],
    '\\nu': ['v', 'u', '\\gamma'],
    '\\mu': ['u', 'n', 'm'],
    'n': ['u', 'h', '\\eta', 'r', '\\pi', 'm'],
    'h': ['n', 'b', 'k', '\\lambda'],
    'r': ['v', '\\gamma', 'n', '\\tau'],
    'w': ['\\omega', 'W', '\\psi', 'v'],
    'W': ['w', '\\omega', 'M'],
    '\\omega': ['w', 'W', '\\varpi'],
    'm': ['n', 'M', '\\mu'],
    'M': ['m', 'N', 'W'],
    'N': ['M', 'n', 'H'],
    'p': ['\\rho', 'P', 'D'],
    'P': ['p', 'D', '\\rho', 'R'],
    '\\rho': ['p', 'P'],
    'e': ['\\epsilon', '\\varepsilon', 'c', 'l'],
    '\\epsilon': ['e', '\\in', '\\varepsilon'],
    '\\varepsilon': ['e', '\\epsilon', '3'],
    '\\in': ['\\epsilon', 'e', 'E'],
    'E': ['\\Sigma', '\\in', 'F', '\\epsilon'],
    'c': ['C', 'e', '(', '\\subset'],
    'C': ['c', '(', 'G', '\\subset'],
    'k': ['K', '\\kappa', 'h', 'x'],
    'K': ['k', '\\kappa', 'X'],
    '\\kappa': ['k', 'x', 'K'],
    'i': ['j', '1', 'l', '\\iota'],
    'j': ['i', 'y', 'J'],
    'J': ['j', 'I', 'T'],
    'A': ['4', '\\Lambda', '\\Delta', 'H'],
    'H': ['A', 'K', 'N'],
    '\\Delta': ['A', '\\delta', '4', '\\nabla'],
    '\\nabla': ['V', '\\Delta', 'v'],
    'V': ['v', 'U', '\\nabla', '\\vee'],
    'U': ['u', 'V', '\\cup'],
    'D': ['0', 'O', 'P'],
    'Q': ['O', '0', 'a'],
    'R': ['B', 'P', 'K'],
    'G': ['6', 'C'],
    'T': ['7', '\\tau', 'I', '+'],
    '\\tau': ['T', 't', 'r'],
    'f': ['t', 'F'],
    'F': ['E', 'f', 'T'],
    '\\phi': ['\\varphi', '\\theta', '\\psi', '\\emptyset', 'o'],
    '\\varphi': ['\\phi', '\\psi'],
    '\\psi': ['\\phi', 'y', '4', '\\Psi'],
    '\\pi': ['n', '\\Pi', 'T'],
    '\\Pi': ['\\pi', 'n', 'T'],
    '\\Sigma': ['E', '\\in'],
    '\\eta': ['n', '\\mu'],
    '\\chi': ['x', 'X', '\\kappa'],
    '\\zeta': ['z', '2', '\\xi'],
    '\\xi': ['\\zeta', '3', 'E'],
    'Y': ['y', '\\gamma', '4', 'V'],
    '\\cdot': ['\\times', '\\circ'],
  };
  // Up to 6 look-alikes, then the other case of a letter.
  function alternatives(tok) {
    const list = [...(LOOKALIKES[tok] || [])];
    if (/^[A-Za-z]$/.test(tok)) list.push(tok === tok.toLowerCase() ? tok.toUpperCase() : tok.toLowerCase());
    return [...new Set(list)].filter(t => t !== tok).slice(0, 6);
  }

  // ------------------------------------------------------------ locks
  // A pick on tapped token k: remembered by its position among the tappable tokens, the total
  // count, and its neighbours, so a later reading of the same ink gets the same fix.
  function makeLock(src, k, to) {
    const T = taps(src), tok = T[k];
    if (!tok) return null;
    return { from: tok.t, to, pos: k, n: T.length, l: T[k - 1]?.t ?? null, r: T[k + 1]?.t ?? null };
  }
  // Re-apply locks to a new reading. Where the reading has the lock's `from` symbol at the same
  // place (or shifted by the change in length, for strokes added in front), it becomes `to`.
  // A reading that already has `to` there, or something else entirely, is left alone.
  function applyLocks(src, locks) {
    let out = String(src || '');
    for (const L of locks || []) {
      if (!L || !L.from || !L.to) continue;
      const T = taps(out);
      const at = [...new Set([L.pos, L.pos + (T.length - L.n)])].filter(k => k >= 0 && k < T.length);
      // The best-placed match wins: `from` there gets fixed, `to` there means it is already right.
      let best = null, score = -1;
      for (const k of at) {
        const t = T[k].t;
        if (t !== L.from && t !== L.to) continue;
        const s = ((T[k - 1]?.t ?? null) === L.l) + ((T[k + 1]?.t ?? null) === L.r) + (k === L.pos ? 0.5 : 0);
        if (s > score || (s === score && t === L.to)) { score = s; best = { k, fix: t === L.from }; }
      }
      if (best?.fix) out = replaceAt(out, T[best.k], L.to);
    }
    return out;
  }

  // ------------------------------------------------------------ readings
  // Nothing to show: empty, NONE, or only empty matrices (\begin{bmatrix} \end{bmatrix} =), which
  // KaTeX would happily render over the ink of the 1 you were writing. A matrix of only spacing,
  // & and row breaks is empty too (qwen answers a doodle with \begin{bmatrix} \text{ } \\ \end{bmatrix}).
  function isEmptyReading(latex) {
    const t = String(latex || '').trim();
    if (!t || /^(\\text\{\s*)?NONE\s*\}?\.?$/i.test(t)) return true;
    const re = /\\begin\{([pbvBV]?matrix|array)\}(\{[^}]*\})?(?:\s|&|~|\\\\|\\[,;:! ]|\\q?quad(?![A-Za-z])|\\(?:text|mathrm)\{\s*\})*\\end\{\1\}/g;
    if (!re.test(t)) return false;
    return !t.replace(re, '').replace(/\\(geq|leq|ge|le|neq|quad|qquad|,|;|!)|[=<>\s&]/g, '');
  }
  // For comparing two readings: no spaces, no braces round one character after ^ or _.
  const canon = s => String(s || '').replace(/\s+/g, '').replace(/([\^_])\{([^{}\\])\}/g, '$1$2');
  const sameReading = (a, b) => canon(a) === canon(b);

  // One shape for every /api/recognize reply (docs/BOARD_BACKENDS.md section 5). The old server
  // sends { latex, raw, ms, model }; the new one adds empty (about latex only), candidates
  // ([{ backend, model, latex, ms, ... }], qwen first), agree (true / false in ensemble mode, else
  // null), backend (the mode that ran), prompt_tokens and, when one ensemble member failed, errors
  // with a single candidate and agree false.
  //   latex: the reading to show: the server's best, or the other candidate's when the best is empty
  //   cands: the distinct non-empty readings, the best first; differ: there are two to choose from
  //   sure: false when the ensemble disagreed or half of it failed, so nothing commits on its own
  //   by: the backend that produced latex (Re-recognize asks another one)
  function reading(d) {
    d = d && typeof d === 'object' ? d : {};
    const agree = d.agree === true ? true : d.agree === false ? false : null;
    let latex = typeof d.latex === 'string' ? d.latex.trim() : '';
    if (isEmptyReading(latex)) latex = '';
    const cands = [];
    for (const c of Array.isArray(d.candidates) ? d.candidates : []) {
      const l = typeof c?.latex === 'string' ? c.latex.trim() : '';
      if (!l || isEmptyReading(l) || cands.some(x => sameReading(x.latex, l))) continue;
      cands.push({ backend: String(c.backend || ''), model: String(c.model || ''), latex: l, ms: Number.isFinite(c.ms) ? c.ms : null });
    }
    if (!latex && agree === false && cands.length) latex = cands[0].latex;
    const best = cands.findIndex(c => sameReading(c.latex, latex));
    if (best > 0) cands.unshift(...cands.splice(best, 1)); // the server's pick first: it is preselected
    else if (latex && best < 0) cands.unshift({ backend: String(d.backend || ''), model: String(d.model || ''), latex, ms: null });
    const differ = agree === false && cands.length >= 2;
    const by = latex ? (cands[0]?.backend || String(d.backend || '')) : '';
    return {
      latex, empty: !latex, agree, differ, sure: agree !== false, by, cands: differ ? cands : [],
      model: String(d.model || ''), backend: String(d.backend || ''), ms: Number.isFinite(d.ms) ? d.ms : null,
      promptTokens: Number.isFinite(d.prompt_tokens) ? d.prompt_tokens : null,
      errors: d.errors && typeof d.errors === 'object' ? d.errors : null,
    };
  }

  // Lines broken with \\ outside any environment render on one line in KaTeX's display mode, and a
  // raw newline is just a space: stack them in gathered (aligned when every line has an =).
  function stackLines(src) {
    const s = String(src || '').trim();
    let depth = 0, prev = 0;
    const cuts = [], newlines = []; // [start, end] of each \\ and each newline outside environments
    for (const tok of tokenize(s)) {
      if (!depth && s.slice(prev, tok.i).includes('\n')) newlines.push([prev, tok.i]);
      if (tok.t === '\\begin') depth++;
      else if (tok.t === '\\end') depth = Math.max(0, depth - 1);
      else if (tok.t === '\\\\' && !depth) cuts.push([tok.i, tok.j]);
      prev = tok.j;
    }
    const at = cuts.length ? cuts : newlines;
    if (!at.length) return null;
    let lines = [], from = 0;
    for (const [a, b] of at) { lines.push(s.slice(from, a)); from = b; }
    lines.push(s.slice(from));
    lines = lines.map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) return null;
    if (lines.every(l => /^[^=]*[^\\]=/.test(l) && !/&/.test(l))) {
      return `\\begin{aligned}${lines.map(l => l.replace(/=/, '&=')).join(' \\\\ ')}\\end{aligned}`;
    }
    return `\\begin{gathered}${lines.join(' \\\\ ')}\\end{gathered}`;
  }

  const tex = {
    tokenize, scan, taps, isTappable, replaceAt, LOOKALIKES, alternatives, makeLock, applyLocks,
    isEmptyReading, canon, sameReading, reading, stackLines,
  };
  globalThis.mathboardBoard = Object.assign(globalThis.mathboardBoard || {}, { tex });
})();
