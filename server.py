#!/usr/bin/env python3
"""Mathboard: a live whiteboard that turns handwritten math into typeset LaTeX.

Serves the board UI from ./static and forwards recognition requests to a local
vision model: qwen3-vl through Ollama (the default), Uni-MuMER through llama.cpp's
llama-server, or both at once ("ensemble"). Standard library only; if Pillow is
installed, crops are rescaled for the model first (docs/BOARD_BACKENDS.md).

    python server.py                       # opens the board in an app window
    python server.py --model NAME          # use a different Ollama vision model
    python server.py --backend ensemble    # qwen3-vl and Uni-MuMER side by side
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import zlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:  # optional: without Pillow, images go to the model exactly as the board sent them
    from PIL import Image
except ImportError:
    Image = None

STATIC = Path(__file__).resolve().parent / "static"
# The last few images the model saw + its raw replies. MATHBOARD_DEBUG_DIR moves them (a test server
# must not rotate your own handwriting out of debug/).
DEBUG_DIR = Path(os.environ.get("MATHBOARD_DEBUG_DIR") or Path(__file__).resolve().parent / "debug")
DEBUG_KEEP = 40
DEFAULT_MODEL = os.environ.get("MATHBOARD_MODEL", "qwen3-vl:8b-instruct")
DEFAULT_OLLAMA = os.environ.get("MATHBOARD_OLLAMA", "http://127.0.0.1:11434")
DEFAULT_BACKEND = os.environ.get("MATHBOARD_BACKEND", "qwen")
DEFAULT_UNIMUMER = os.environ.get("MATHBOARD_UNIMUMER", "")  # llama-server URL, e.g. http://127.0.0.1:8792

MODES = ("qwen", "unimumer", "ensemble")
# In ensemble mode both models read the image; when they disagree, `latex` is this one's answer.
# Uni-MuMER was right more often on the disagreements in the eval set (docs/BOARD_BACKENDS.md).
ENSEMBLE_PRIMARY = "unimumer"

SYSTEM_PROMPT = (
    "You transcribe handwritten mathematics from images into LaTeX for a live "
    "lecture whiteboard. You are a transcriber, not a solver."
)

USER_PROMPT = r"""Transcribe the handwriting in this image into LaTeX.
Rules:
- Output only the LaTeX. No $ signs, no \[ \], no code fences, no commentary.
- Copy exactly what is written. Never solve, simplify, reorder or finish anything. If it ends with "=", your output ends with "=".
- Matrices: pmatrix for ( ), bmatrix for [ ], vmatrix for | |, matrix when there are no brackets.
  Always write the full environment, even for a single row: \begin{bmatrix} a & b & c \end{bmatrix}.
- Several stacked lines: \begin{aligned} ... \end{aligned}, one row per written line, rows separated by \\, with & placed before each line's first = sign.
- Written words go in \text{...}.
- If there is no math or writing (a doodle, arrow or shape), output exactly: NONE

Context: this is live handwriting from a math lecture, often drawn with a mouse, so strokes are shaky and thin.
- Everything drawn is math notation: digits, letters, operators, brackets. Never output shape commands (\bigcirc, \circ, \square, \Box, \triangle, \diamond).
- A small closed loop is 0 (or the letter o in words). A short vertical or slightly curved stroke is 1.
- For a matrix, count its rows and columns first and include every entry, including zeros.
- Inside a matrix, an entry that looks like ( or ) is a curved 1."""

# Uni-MuMER's training prompt, verbatim (phxember/Uni-MuMER-Data). In training the image came first,
# with no system message, and that is how it is sent here.
UNIMUMER_PROMPT = (
    "I have an image of a handwritten mathematical expression. "
    "Please write out the expression of the formula in the image using LaTeX format."
)

# What each backend's vision encoder should get. Symbols read best at about `symbol` px tall
# (docs/BOARD_DIAGNOSIS.md, 1.3); `margin` is white space around the ink, in symbols.
# qwen: Ollama starts qwen3-vl with --image-min-tokens 1024 (32 x 32 px per token), so anything under
#   about 1 MP is enlarged; pad up to that instead. Its 4096-token context failed at 3 MP, so cap at 1.5 MP.
# unimumer: trained on 65,536 to 262,144 px images (llama-server gets --image-min-tokens 64
#   --image-max-tokens 256) with tight crops and symbols around 80 px (median of its training renders).
NORM = {
    "qwen": {"symbol": 64, "margin": 0.75, "min_area": 1 << 20, "max_area": 3 << 19},
    "unimumer": {"symbol": 80, "margin": 0.25, "min_area": 1 << 16, "max_area": 1 << 18},
}

# Matrix cells that can only be misreads of a hand-drawn 1 or 0.
MATRIX_RE = re.compile(r"(\\begin\{([pbvBV]?matrix)\})(.*?)(\\end\{\2\})", re.S)
CELL_FIX = {"(": "1", ")": "1", "|": "1", r"\bigcirc": "0", r"\circ": "0", "o": "0", "O": "0"}
# A matrix with no entries: valid KaTeX, but it would hide the ink of the half-written matrix it came from.
EMPTY_MATRIX_RE = re.compile(r"\\begin\{([pbvBV]?matrix)\}[\s&\\]*\\end\{\1\}")
# A whole reading that is one bracketless matrix with a single cell: Uni-MuMER wraps a lone fraction
# like that, which renders the same as the cell and made the ensemble disagree with qwen's bare \frac.
LONE_CELL_RE = re.compile(r"\\begin\{matrix\}(.*)\\end\{matrix\}", re.S)
# A lone bar the model boxed up as text, and a NONE it wrapped.
TEXT_DASH_RE = re.compile(r"\\(?:text|mathrm)\{\s*(?:-{1,3}|\u2013|\u2014)\s*\}")
TEXT_NONE_RE = re.compile(r"\\(?:text|mathrm|textbf|texttt)\{\s*NONE\s*\.?\s*\}", re.I)
# Spacing and wrappers that are not content on their own.
FILLER_RE = re.compile(r"\\(?:[,;:! ]|quad|qquad|displaystyle|textstyle|text|mathrm|boxed)(?![A-Za-z])|[{}\s~&]")


# Shorthand the model sometimes emits for one-row matrices: bmatrix{1 & 2 & 3} or \pmatrix{...}.
SHORT_MATRIX_RE = re.compile(r"(?<![A-Za-z{])\\?([pbvBV]?matrix)\s*\{")


def _expand_short_matrices(t: str) -> str:
    out, i = [], 0
    for m in SHORT_MATRIX_RE.finditer(t):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while j < len(t) and depth:
            if t[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        if depth:
            break
        env = m.group(1)
        out += [t[i:m.start()], rf"\begin{{{env}}} {t[m.end():j - 1].strip()} \end{{{env}}}"]
        i = j
    return "".join(out) + t[i:]


def _fix_matrix_cells(m: re.Match) -> str:
    rows = []
    for row in m.group(3).split("\\\\"):
        cells = [c.replace(c.strip(), CELL_FIX[c.strip()]) if c.strip() in CELL_FIX else c for c in row.split("&")]
        rows.append("&".join(cells))
    return m.group(1) + "\\\\".join(rows) + m.group(4)


# Things that open or close a nesting level, plus the characters the line helpers look for.
NEST_RE = re.compile(r"\\begin\{[^}]*\}|\\end\{[^}]*\}|\\[{}&=\\]|[{}]|\n|&|=")


def _top_level(t: str, char: str) -> list[int]:
    """Positions of `char` (a newline, & or =) outside braces and environments."""
    depth, out = 0, []
    for m in NEST_RE.finditer(t):
        s = m.group()
        if s == "{" or s.startswith("\\begin"):
            depth += 1
        elif s == "}" or s.startswith("\\end"):
            depth -= 1
        elif s == char and depth <= 0:
            out.append(m.start())
    return out


def _stack_lines(t: str) -> str:
    """Raw newlines between written lines: KaTeX would run them together, so stack them.

    Every line has an = sign: aligned, with & before each first =. Otherwise gathered."""
    cuts = _top_level(t, "\n")
    if not cuts:
        return t
    lines = [t[a + 1:b] for a, b in zip([-1] + cuts, cuts + [len(t)])]
    lines = [re.sub(r"\\\\\s*$", "", ln.strip()).strip() for ln in lines]  # a trailing \\ is the row break added below
    lines = [ln for ln in lines if ln]
    if len(lines) < 2:
        return " ".join(lines)
    eqs = [_top_level(ln, "=") for ln in lines]
    if any(_top_level(ln, "&") for ln in lines):  # the model already placed the & signs
        return r"\begin{aligned} " + r" \\ ".join(lines) + r" \end{aligned}"
    if all(eqs):
        rows = [f"{ln[:e[0]].rstrip()} &{ln[e[0]:]}" for ln, e in zip(lines, eqs)]
        return r"\begin{aligned} " + r" \\ ".join(rows) + r" \end{aligned}"
    return r"\begin{gathered} " + r" \\ ".join(lines) + r" \end{gathered}"


def clean_latex(text: str) -> str:
    """Strip the wrappers models like to add; return '' for 'nothing to transcribe'."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```[A-Za-z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    t = re.sub(r"^(latex|LaTeX)\s*:\s*", "", t)
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        inner = t[len(a):-len(b)]
        if t.startswith(a) and t.endswith(b) and inner.strip() and a not in inner and b not in inner:
            t = inner.strip()
            break
    m = re.fullmatch(r"\\begin\{(equation\*?|displaymath|math)\}(.*)\\end\{\1\}", t, flags=re.S)
    if m:
        t = m.group(2).strip()
    if t.strip(" .").upper() == "NONE" or TEXT_NONE_RE.fullmatch(t.strip()):
        return ""
    t = re.sub(r"\\(begin|end)\s+\{", r"\\\1{", t)  # Uni-MuMER writes \begin {pmatrix}
    t = MATRIX_RE.sub(_fix_matrix_cells, _expand_short_matrices(t))
    if EMPTY_MATRIX_RE.search(t) or any(not FILLER_RE.sub("", c.group(3).replace("\\\\", "")) for c in MATRIX_RE.finditer(t)):
        return ""  # also a matrix of only spacing, like qwen's \begin{bmatrix} \text{ } \\ \end{bmatrix} for a doodle
    m = LONE_CELL_RE.fullmatch(t.strip())
    if m and "\\end{matrix}" not in m.group(1) and not _top_level(m.group(1), "&") and not _top_level(m.group(1), "\\\\"):
        t = m.group(1).strip()
    t = TEXT_DASH_RE.sub("-", _stack_lines(t))
    return t if FILLER_RE.sub("", t.replace("\\\\", "")) else ""


def tidy_spacing(text: str) -> str:
    """Uni-MuMER writes CROHME-style spaced tokens (x ^ { 2 }); make them read like hand-typed LaTeX."""
    out = []
    for tok in re.findall(r"\\[A-Za-z]+|\\.|\s+|.", text.strip(), flags=re.S):
        if tok.isspace():
            continue
        if out and re.fullmatch(r"\\[A-Za-z]+", out[-1]) and tok[0].isalpha():
            out.append(" ")  # \alpha x must not become \alphax
        if tok in ("&", "\\\\"):  # keep matrix cells readable in the LaTeX editor
            tok = f" {tok} "
        out.append(tok)
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def postprocess(backend: str, raw: str) -> str:
    return clean_latex(tidy_spacing(raw) if backend == "unimumer" else raw)


def canonical(latex: str) -> str:
    """A spelling-insensitive form of cleaned LaTeX, for comparing two readings (ensemble, eval).

    x^{2} and x^2, \\text{Word} and Word, \\left( and (, \\dfrac and \\frac, aligned and plain stacked
    rows, spacing: all compare equal. Anything that renders differently stays different."""
    s = latex.strip().replace("\\\\", "\x01")  # a row break is one token; its second \ must not start a command
    s = re.sub(r"\\(?:text|mathrm|textit|mathit|textrm|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)(?![A-Za-z])", "", s)
    s = re.sub(r"\\(?:[,;:! ]|quad|qquad|displaystyle|textstyle)(?![A-Za-z])|~", "", s)
    s = re.sub(r"\\[dt]frac(?![A-Za-z])", r"\\frac", s)
    s = re.sub(r"\\(?:ldots|cdots|dots)(?![A-Za-z])", r"\\dots", s)
    s = re.sub(r"\\(begin|end)\s*\{(aligned|gathered|align\*?|gather\*?)\}", "", s)
    s = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z])", lambda m: m.group(1) + "\x00", s)  # keep the space ending a command word
    s = re.sub(r"\s+", "", s).replace("\x00", " ")
    s = s.replace("&=", "=")
    for _ in range(2):  # ^{2} -> ^2, _{\alpha} -> _\alpha, {} -> nothing
        s = re.sub(r"([\^_])\{(\\[A-Za-z]+|[^{}\\])\}", r"\1\2", s)
        s = s.replace("{}", "")
    for env, (a, b) in {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"), "vmatrix": ("|", "|")}.items():
        s = re.sub(rf"{re.escape(a)}\\begin\{{matrix\}}(.*?)\\end\{{matrix\}}{re.escape(b)}",
                   rf"\\begin{{{env}}}\1\\end{{{env}}}", s)
    return s.replace("\x01", "\\\\")


# ------------------------------------------------------------------ image normalization
def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def estimate_symbol_px(gray) -> float | None:
    """Median symbol height, in px, of a black-on-white grayscale image; None when there is no ink.

    Connected ink blobs stand in for symbols. Bars (one stroke thick: a minus, a fraction bar, the halves
    of =) and dots are left out; when only bars are left, a bar counts as 0.6 of its width."""
    r = max(1, math.ceil(max(gray.size) / 320))  # work on a reduced copy: the blob walk is plain Python
    small = gray.reduce(r) if r > 1 else gray
    w, h = small.size
    ink = small.point(lambda v: 255 if v < 200 else 0).tobytes()  # thin strokes turn grey when reduced
    runs = [len(m) for y in range(h) for m in re.findall(rb"\xff+", ink[y * w:(y + 1) * w])]
    if not runs:
        return None
    stroke = _median(runs) * r
    seen, n, blobs = bytearray(w * h), w * h, []
    i = ink.find(255)
    while i != -1:
        if not seen[i]:
            seen[i] = 1
            stack, x0, x1, y0, y1 = [i], w, 0, h, 0
            while stack:
                j = stack.pop()
                y, x = divmod(j, w)
                x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
                for k in (j - w - 1, j - w, j - w + 1, j - 1, j + 1, j + w - 1, j + w, j + w + 1):
                    if 0 <= k < n and ink[k] and not seen[k] and abs(k % w - x) <= 1:
                        seen[k] = 1
                        stack.append(k)
            blobs.append(((x1 - x0 + 1) * r, (y1 - y0 + 1) * r))
        i = ink.find(255, i + 1)
    blobs = [(bw, bh) for bw, bh in blobs if max(bw, bh) > max(8, 2 * stroke)]  # dots and specks
    tall = [bh for bw, bh in blobs if not (bh <= 3 * stroke and bw >= 1.5 * bh)]
    if tall:
        return _median(tall)
    return _median([0.6 * bw for bw, bh in blobs]) if blobs else None


def load_gray(png: bytes):
    """The upload as grayscale on white (the board sends opaque RGBA; transparency would read as black)."""
    im = Image.open(io.BytesIO(png))
    im.load()
    if im.mode in ("RGBA", "LA", "PA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        im = bg
    return im.convert("L")


def normalize_image(png: bytes, profile: dict, symbol_px: float | None = None) -> tuple[bytes, dict]:
    """Crop to the ink, scale so symbols are about profile['symbol'] px tall, pad to the model's pixel budget.

    symbol_px is the client's median symbol height in this image's pixels; without it, it is estimated.
    Returns the PNG to send and a note of what was done."""
    if Image is None:
        return png, {"applied": False, "reason": "Pillow is not installed"}
    t0 = time.perf_counter()
    try:
        gray = load_gray(png)
    except Exception as e:  # noqa: BLE001  (Pillow raises many types for bad input)
        return png, {"applied": False, "reason": f"unreadable image ({e})"}
    source = "hint"
    if not symbol_px:
        symbol_px, source = estimate_symbol_px(gray), "estimate"
    box = gray.point(lambda v: 255 if v < 200 else 0).getbbox()
    if not symbol_px or not box:
        return png, {"applied": False, "reason": "no ink"}
    crop = gray.crop(box)
    f = profile["symbol"] / symbol_px
    pad = profile["margin"] * profile["symbol"]
    area = (crop.width * f + 2 * pad) * (crop.height * f + 2 * pad)
    if area > profile["max_area"]:
        g = math.sqrt(profile["max_area"] / area)
        f, pad = f * g, pad * g
    size = (max(1, round(crop.width * f)), max(1, round(crop.height * f)))
    ink = crop.resize(size, Image.LANCZOS if f < 1 else Image.BICUBIC)
    cw, ch = size[0] + 2 * pad, size[1] + 2 * pad
    g = max(1.0, math.sqrt(profile["min_area"] / (cw * ch)))  # pad, don't scale, up to the floor
    cw, ch = math.ceil(cw * g), math.ceil(ch * g)
    canvas = Image.new("L", (cw, ch), 255)
    canvas.paste(ink, ((cw - size[0]) // 2, (ch - size[1]) // 2))
    out = io.BytesIO()
    canvas.convert("RGB").save(out, "PNG", compress_level=1)
    return out.getvalue(), {
        "applied": True, "symbol_px": round(symbol_px, 1), "source": source, "scale": round(f, 3),
        "size": [cw, ch], "ms": round((time.perf_counter() - t0) * 1000),
    }


# ------------------------------------------------------------------ backends
class BackendError(RuntimeError):
    pass


class OllamaError(BackendError):
    pass


def blank_png(w: int = 64, h: int = 64) -> str:
    """A white PNG as base64, used to load the model before the first real stroke."""
    rows = b"".join(b"\x00" + b"\xff" * (w * 3) for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _http_json(url: str, payload: dict | None, timeout: float, who: str, err=BackendError):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
            if isinstance(detail, dict):  # llama-server: {"error": {"message": ...}}
                detail = detail.get("message", detail)
        except (ValueError, AttributeError):
            pass
        raise err(f"{who} {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise err(f"can't reach {who} at {url.split('/', 3)[2]} ({getattr(e, 'reason', e)})") from None


class OllamaBackend:
    """qwen3-vl (or any Ollama vision model): owns the Ollama connection and the active model."""

    name = "qwen"

    def __init__(self, url: str, model: str, keep_alive: str):
        self.url = url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.state = "starting"  # starting | downloading | loading | ready | error
        self.error = ""
        self.last_ms: int | None = None
        self._think_flag = False
        self._caps: dict[str, list[str]] = {}
        self._rec_lock = threading.Lock()  # one recognition at a time
        self._activating = False
        self._last_attempt = 0.0
        self._last_used = 0.0

    # -- plumbing -----------------------------------------------------------
    def _call(self, path: str, payload: dict | None = None, timeout: float = 300):
        return _http_json(self.url + path, payload, timeout, "Ollama", OllamaError)

    def capabilities(self, model: str) -> list[str]:
        if model not in self._caps:
            info = self._call("/api/show", {"model": model}, timeout=30)
            self._caps[model] = info.get("capabilities") or []
        return self._caps[model]

    def vision_models(self) -> list[str]:
        names = [m["name"] for m in self._call("/api/tags", timeout=10).get("models", [])]
        out = []
        for name in names:
            try:
                if "vision" in self.capabilities(name):
                    out.append(name)
            except OllamaError:
                pass
        return out

    def _pull(self, model: str) -> None:
        self.state, self.error = "downloading", f"downloading {model}"
        req = urllib.request.Request(
            self.url + "/api/pull", data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                msg = json.loads(line)
                if msg.get("error"):
                    raise OllamaError(msg["error"])
                total, done = msg.get("total"), msg.get("completed")
                if total and done:
                    self.error = f"downloading {model}: {100 * done // total}%"
        self._caps.pop(model, None)

    # -- lifecycle ----------------------------------------------------------
    def activate(self, model: str | None = None) -> None:
        """Switch to `model`, downloading it if needed, and load it into VRAM."""
        model = model or self.model
        self._activating, self._last_attempt = True, time.time()
        self.model, self.state, self.error = model, "loading", ""
        try:
            try:
                caps = self.capabilities(model)
            except OllamaError as e:
                if "not found" not in str(e):
                    raise
                self._pull(model)
                self.state, self.error = "loading", ""
                caps = self.capabilities(model)
            if "vision" not in caps:
                raise OllamaError(f"{model} can't read images (capabilities: {', '.join(caps) or 'none'})")
            self._think_flag = "thinking" in caps
            self.recognize(blank_png())  # loads weights + vision encoder
            self.state = "ready"
            print(f"[mathboard] {model} loaded")
        except OllamaError as e:
            self.state, self.error = "error", str(e)
            print(f"[mathboard] {e}")
        finally:
            self._activating = False

    def heartbeat(self) -> None:
        """Called on every status poll: retry a failed start, keep the model resident."""
        now = time.time()
        if self.state == "error" and not self._activating and now - self._last_attempt > 5:
            threading.Thread(target=self.activate, args=(self.model,), daemon=True).start()
        elif self.state == "ready" and now - self._last_used > 240:
            self._last_used = now
            threading.Thread(target=self._refresh_keep_alive, daemon=True).start()

    def _refresh_keep_alive(self) -> None:
        try:
            self._call("/api/generate", {"model": self.model, "keep_alive": self.keep_alive}, timeout=120)
        except OllamaError:
            pass

    def unload(self) -> None:
        try:
            self._call("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=10)
        except OllamaError:
            pass

    def status(self) -> dict:
        return {
            "ready": self.state == "ready", "state": self.state, "model": self.model,
            "error": self.error, "last_ms": self.last_ms,
        }

    # -- recognition --------------------------------------------------------
    def recognize(self, image_b64: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT, "images": [image_b64]},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": 512},
        }
        if self._think_flag:
            payload["think"] = False
        with self._rec_lock:
            t0 = time.perf_counter()
            data = self._call("/api/chat", payload)
            ms = round((time.perf_counter() - t0) * 1000)
        self._last_used = time.time()
        self.last_ms = ms
        raw = (data.get("message") or {}).get("content", "")
        ns = lambda k: round((data.get(k) or 0) / 1e6)  # noqa: E731
        return {
            "raw": raw, "ms": ms, "model": self.model, "prompt_tokens": data.get("prompt_eval_count"),
            "timings": {"load": ns("load_duration"), "prompt": ns("prompt_eval_duration"),
                        "decode": ns("eval_duration"), "tokens": data.get("eval_count"),
                        "prompt_tokens": data.get("prompt_eval_count")},
        }


class LlamaServerBackend:
    """Uni-MuMER on llama.cpp's llama-server (OpenAI-style API). start_mathboard.bat starts that process."""

    name = "unimumer"

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.model = "Uni-MuMER"
        self.state = "starting"  # starting | loading | ready | error
        self.error = ""
        self.last_ms: int | None = None
        self._rec_lock = threading.Lock()
        self._activating = False
        self._last_attempt = 0.0

    def _call(self, path: str, payload: dict | None = None, timeout: float = 300):
        return _http_json(self.url + path, payload, timeout, "llama-server")

    def activate(self, model: str | None = None) -> None:
        """Wait for llama-server to finish loading, then run one image through it to warm up the encoder."""
        self._activating, self._last_attempt = True, time.time()
        self.state, self.error = "loading", ""
        try:
            self._call("/health", timeout=10)  # 503 while the model is still loading
            ids = [m.get("id", "") for m in self._call("/v1/models", timeout=10).get("data", [])]
            if ids:
                self.model = re.sub(r"\.gguf$", "", re.split(r"[\\/]", ids[0])[-1])
            self.recognize(blank_png())
            self.state = "ready"
            print(f"[mathboard] {self.model} ready on llama-server {self.url}")
        except BackendError as e:
            self.state, self.error = "error", str(e)
            print(f"[mathboard] {e}")
        finally:
            self._activating = False

    def heartbeat(self) -> None:
        if self.state == "error" and not self._activating and time.time() - self._last_attempt > 5:
            threading.Thread(target=self.activate, daemon=True).start()

    def unload(self) -> None:
        pass  # llama-server belongs to start_mathboard.bat, which stops it

    def status(self) -> dict:
        return {
            "ready": self.state == "ready", "state": self.state, "model": self.model,
            "error": self.error, "last_ms": self.last_ms,
        }

    def recognize(self, image_b64: str) -> dict:
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                {"type": "text", "text": UNIMUMER_PROMPT},
            ]}],
            "stream": False, "temperature": 0, "max_tokens": 512,
        }
        with self._rec_lock:
            t0 = time.perf_counter()
            data = self._call("/v1/chat/completions", payload)
            ms = round((time.perf_counter() - t0) * 1000)
        self.last_ms = ms
        raw = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        tm = data.get("timings") or {}
        prompt_tokens = (data.get("usage") or {}).get("prompt_tokens")
        return {
            "raw": raw, "ms": ms, "model": self.model, "prompt_tokens": prompt_tokens,
            "timings": {"load": 0, "prompt": round(tm.get("prompt_ms") or 0), "decode": round(tm.get("predicted_ms") or 0),
                        "tokens": tm.get("predicted_n"), "prompt_tokens": prompt_tokens},
        }


class Engine:
    """Routes each recognition to one backend, or to both at once in ensemble mode."""

    LABELS = {"qwen": "qwen3-vl (Ollama)", "unimumer": "Uni-MuMER (llama-server)", "ensemble": "both, compared"}

    def __init__(self, backends: dict, default: str, normalize: bool = True, primary: str = ENSEMBLE_PRIMARY):
        self.backends = backends  # name -> OllamaBackend | LlamaServerBackend
        if default not in self.available():
            raise ValueError(f"backend {default!r} is not configured (have: {', '.join(self.available())})")
        self.default = default
        self.normalize = normalize
        self.primary = primary
        self.last_ms: int | None = None
        self._active = set(self.members(default))  # backends kept loaded and polled
        for name, b in backends.items():
            if name not in self._active:
                b.state = "idle"  # loaded on its first request

    def available(self) -> list[str]:
        return [m for m in MODES if all(n in self.backends for n in self.members(m))]

    @staticmethod
    def members(mode: str) -> list[str]:
        return ["qwen", "unimumer"] if mode == "ensemble" else [mode]

    # -- lifecycle ----------------------------------------------------------
    def activate(self) -> None:
        for name in self._active:
            threading.Thread(target=self.backends[name].activate, daemon=True).start()

    def heartbeat(self) -> None:
        for name in self._active:
            self.backends[name].heartbeat()

    def unload(self) -> None:
        for name in self._active:
            self.backends[name].unload()

    def status(self) -> dict:
        sts = {n: b.status() for n, b in self.backends.items()}

        def combined(mode: str) -> dict:
            parts = [sts[n] for n in self.members(mode)]
            bad = next((p for p in parts if not p["ready"]), None)
            return {
                "ready": bad is None, "state": bad["state"] if bad else "ready",
                "model": " + ".join(p["model"] for p in parts),
                "error": "; ".join(p["error"] for p in parts if p["error"]),
            }

        out = {**combined(self.default), "last_ms": self.last_ms, "backend": self.default,
               "normalize": self.normalize and Image is not None}
        out["backends"] = [{"name": m, "label": self.LABELS[m], "default": m == self.default, **combined(m)}
                           for m in self.available()]
        return out

    # -- recognition --------------------------------------------------------
    def _run_one(self, name: str, png: bytes, image_b64: str, symbol_px: float | None) -> dict:
        backend, norm = self.backends[name], {"applied": False, "reason": "off"}
        if self.normalize:
            png2, norm = normalize_image(png, NORM[name], symbol_px)
            if norm["applied"]:
                image_b64 = base64.b64encode(png2).decode()
        r = backend.recognize(image_b64)
        return {"backend": name, **r, "latex": postprocess(name, r["raw"]), "norm": norm}

    def recognize(self, image_b64: str, symbol_px: float | None = None, backend: str | None = None) -> dict:
        mode = backend or self.default
        if mode not in self.available():
            raise BackendError(f"backend {mode!r} is not available (have: {', '.join(self.available())})")
        names = self.members(mode)
        for n in names:
            if n not in self._active:  # first use of a backend outside the default mode: load it, keep it warm
                self._active.add(n)
                self.backends[n].activate()
        png = base64.b64decode(image_b64)
        t0 = time.perf_counter()
        if self.normalize and not symbol_px and len(names) > 1 and Image is not None:
            try:  # estimate once for both
                symbol_px = estimate_symbol_px(load_gray(png))
            except Exception:  # noqa: BLE001  (normalize_image reports unreadable images)
                symbol_px = None
        results, errors = {}, {}

        def run(n: str) -> None:
            try:
                results[n] = self._run_one(n, png, image_b64, symbol_px)
            except BackendError as e:
                errors[n] = str(e)

        threads = [threading.Thread(target=run, args=(n,)) for n in names[1:]]
        for th in threads:
            th.start()
        run(names[0])
        for th in threads:
            th.join()
        ms = round((time.perf_counter() - t0) * 1000)
        if not results:
            raise BackendError("; ".join(errors.values()))
        cands = [results[n] for n in names if n in results]
        agree = None
        if mode == "ensemble":
            agree = len(cands) == 2 and canonical(cands[0]["latex"]) == canonical(cands[1]["latex"])
        best = results.get(self.primary) if mode == "ensemble" and self.primary in results else cands[0]
        self.last_ms = ms
        out = {
            "latex": best["latex"], "raw": best["raw"], "ms": ms, "model": best["model"],
            "prompt_tokens": best["prompt_tokens"], "empty": not best["latex"],
            "candidates": [{k: c[k] for k in ("backend", "model", "latex", "raw", "ms", "prompt_tokens", "norm")}
                           for c in cands],
            "agree": agree, "backend": mode, "timings": best["timings"], "norm": best["norm"],
        }
        if errors:
            out["errors"] = errors
        return out


def save_debug(image_b64: str, result: dict, extra: dict | None = None) -> None:
    """Keep the last DEBUG_KEEP recognitions (PNG + JSON) for diagnosing misreads."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        (DEBUG_DIR / f"{stamp}.png").write_bytes(base64.b64decode(image_b64))
        info = {k: result[k] for k in ("latex", "raw", "ms", "model") if k in result}
        info.update({k: result[k] for k in ("backend", "prompt_tokens", "agree", "norm") if k in result})
        if len(result.get("candidates") or []) > 1:
            info["candidates"] = [{k: c[k] for k in ("backend", "model", "latex", "raw", "ms")} for c in result["candidates"]]
        info.update({k: v for k, v in (extra or {}).items() if v is not None})
        (DEBUG_DIR / f"{stamp}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        for old in sorted(DEBUG_DIR.glob("*.png"))[:-DEBUG_KEEP]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


class Handler(SimpleHTTPRequestHandler):
    engine: Engine
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".css": "text/css", ".html": "text/html",
        ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".svg": "image/svg+xml",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):  # keep the console for recognition logs
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        eng = self.engine
        if self.path == "/api/status":
            eng.heartbeat()
            return self._json(200, eng.status())
        if self.path == "/api/models":
            if "qwen" not in eng.members(eng.default):
                llama = eng.backends["unimumer"]
                return self._json(200, {"models": [llama.model], "active": llama.model})
            try:
                ollama = eng.backends["qwen"]
                return self._json(200, {"models": ollama.vision_models(), "active": ollama.model})
            except BackendError as e:
                return self._json(502, {"error": str(e)})
        return super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad JSON"})

        if self.path == "/api/recognize":
            image = str(body.get("image", "")).split(",")[-1]
            if not image:
                return self._json(400, {"error": "no image"})
            try:
                symbol_px = float(body.get("symbol_px") or 0) or None
            except (TypeError, ValueError):
                symbol_px = None
            if symbol_px is not None and not 4 <= symbol_px <= 4096:
                symbol_px = None
            backend = body.get("backend") or None
            if backend is not None and backend not in self.engine.available():
                return self._json(400, {"error": f"backend {backend!r} is not available "
                                                 f"(have: {', '.join(self.engine.available())})"})
            try:
                result = self.engine.recognize(image, symbol_px=symbol_px, backend=backend)
            except BackendError as e:
                print(f"[rec] error: {e}")
                return self._json(502, {"error": str(e)})
            except ValueError as e:  # undecodable base64
                return self._json(400, {"error": f"bad image ({e})"})
            flag = "" if result["agree"] is None else ("  (agree)" if result["agree"] else "  (DISAGREE)")
            print(f"[rec] {result['ms']:>5} ms  {result['latex'] or '(none)'}{flag}")
            strokes = body.get("strokes")
            save_debug(image, result, {"symbol_px_hint": symbol_px,
                                       "strokes": strokes if isinstance(strokes, list) else None})
            return self._json(200, result)

        if self.path == "/api/model":
            model = str(body.get("model", "")).strip()
            if not model:
                return self._json(400, {"error": "no model"})
            llama = self.engine.backends.get("unimumer")
            if llama and model == llama.model:  # the only entry /api/models lists in unimumer mode
                return self._json(200, {"ok": True})
            threading.Thread(target=self.engine.backends["qwen"].activate, args=(model,), daemon=True).start()
            return self._json(200, {"ok": True})

        return self._json(404, {"error": "not found"})


def open_board(url: str, app_window: bool) -> None:
    """Prefer a chromeless app window: cleaner for screen sharing / OBS window capture."""
    if app_window:
        candidates = [
            shutil.which("chrome"),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for exe in candidates:
            if exe and Path(exe).exists():
                subprocess.Popen([exe, f"--app={url}", "--window-size=1600,900"])
                return
    webbrowser.open(url)


def build_engine(args) -> Engine:
    backends = {"qwen": OllamaBackend(args.ollama, args.model, args.keep_alive)}
    unimumer = args.unimumer or ("http://127.0.0.1:8792" if args.backend in ("unimumer", "ensemble") else "")
    if unimumer:
        backends["unimumer"] = LlamaServerBackend(unimumer)
    return Engine(backends, args.backend, normalize=not args.no_normalize)


class BoardServer(ThreadingHTTPServer):
    # http.server sets SO_REUSEADDR, which on Windows lets a second server bind a port that is
    # already in use: "port busy" never fired and every launch stacked another server on 8791.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mathboard: handwriting -> LaTeX whiteboard")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=MODES,
                    help=f"recognizer: qwen (Ollama), unimumer (llama-server) or ensemble (default {DEFAULT_BACKEND})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama vision model (default {DEFAULT_MODEL})")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA, help="Ollama base URL")
    ap.add_argument("--unimumer", default=DEFAULT_UNIMUMER,
                    help="llama-server URL for Uni-MuMER (default http://127.0.0.1:8792 when the backend needs it)")
    ap.add_argument("--keep-alive", default="30m", help="how long Ollama keeps the model loaded when idle")
    ap.add_argument("--no-normalize", action="store_true", help="send crops as drawn, without rescaling them")
    ap.add_argument("--no-browser", action="store_true", help="don't open the board")
    ap.add_argument("--tab", action="store_true", help="open a normal browser tab instead of an app window")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    engine = build_engine(args)
    Handler.engine = engine
    try:
        server = BoardServer(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"[mathboard] port {args.port} is busy - is Mathboard already running? Opening {url}")
        if not args.no_browser:
            open_board(url, app_window=not args.tab)
        return

    engine.activate()
    if Image is None and not args.no_normalize:
        print("[mathboard] Pillow is not installed: crops go to the model unscaled (pip install pillow)")
    print(f"[mathboard] {url}  backend={args.backend}  (Ctrl+C to stop)")
    if not args.no_browser:
        open_board(url, app_window=not args.tab)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("[mathboard] stopping, unloading model")
        engine.unload()


if __name__ == "__main__":
    main()
