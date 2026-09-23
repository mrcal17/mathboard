#!/usr/bin/env python3
"""Mathboard: a live whiteboard that turns handwritten math into typeset LaTeX.

Serves the board UI from ./static and forwards recognition requests to a local
Ollama vision model. Standard library only.

    python server.py                  # opens the board in an app window
    python server.py --model NAME     # use a different Ollama vision model
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
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

STATIC = Path(__file__).resolve().parent / "static"
DEBUG_DIR = Path(__file__).resolve().parent / "debug"  # last few images the model saw + its raw replies
DEBUG_KEEP = 40
DEFAULT_MODEL = os.environ.get("MATHBOARD_MODEL", "qwen3-vl:8b-instruct")
DEFAULT_OLLAMA = os.environ.get("MATHBOARD_OLLAMA", "http://127.0.0.1:11434")

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

# Matrix cells that can only be misreads of a hand-drawn 1 or 0.
MATRIX_RE = re.compile(r"(\\begin\{([pbvBV]?matrix)\})(.*?)(\\end\{\2\})", re.S)
CELL_FIX = {"(": "1", ")": "1", "|": "1", r"\bigcirc": "0", r"\circ": "0", "o": "0", "O": "0"}


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


class OllamaError(RuntimeError):
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
    if t.strip(" .").upper() == "NONE":
        return ""
    return MATRIX_RE.sub(_fix_matrix_cells, _expand_short_matrices(t))


def save_debug(image_b64: str, result: dict) -> None:
    """Keep the last DEBUG_KEEP recognitions (PNG + JSON) for diagnosing misreads."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        (DEBUG_DIR / f"{stamp}.png").write_bytes(base64.b64decode(image_b64))
        info = {k: result[k] for k in ("latex", "raw", "ms", "model")}
        (DEBUG_DIR / f"{stamp}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        for old in sorted(DEBUG_DIR.glob("*.png"))[:-DEBUG_KEEP]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


class Engine:
    """Owns the Ollama connection and the active model."""

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
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise OllamaError(f"Ollama {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise OllamaError(f"can't reach Ollama at {self.url} ({getattr(e, 'reason', e)})") from None

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
    def activate(self, model: str) -> None:
        """Switch to `model`, downloading it if needed, and load it into VRAM."""
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
            "latex": clean_latex(raw), "raw": raw, "ms": ms, "model": self.model,
            "timings": {"load": ns("load_duration"), "prompt": ns("prompt_eval_duration"),
                        "decode": ns("eval_duration"), "tokens": data.get("eval_count")},
        }


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
        if self.path == "/api/status":
            self.engine.heartbeat()
            return self._json(200, self.engine.status())
        if self.path == "/api/models":
            try:
                return self._json(200, {"models": self.engine.vision_models(), "active": self.engine.model})
            except OllamaError as e:
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
                result = self.engine.recognize(image)
            except OllamaError as e:
                print(f"[rec] error: {e}")
                return self._json(502, {"error": str(e)})
            print(f"[rec] {result['ms']:>5} ms  {result['latex'] or '(none)'}")
            save_debug(image, result)
            return self._json(200, result)

        if self.path == "/api/model":
            model = str(body.get("model", "")).strip()
            if not model:
                return self._json(400, {"error": "no model"})
            threading.Thread(target=self.engine.activate, args=(model,), daemon=True).start()
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Mathboard: handwriting -> LaTeX whiteboard")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama vision model (default {DEFAULT_MODEL})")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA, help="Ollama base URL")
    ap.add_argument("--keep-alive", default="30m", help="how long Ollama keeps the model loaded when idle")
    ap.add_argument("--no-browser", action="store_true", help="don't open the board")
    ap.add_argument("--tab", action="store_true", help="open a normal browser tab instead of an app window")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    engine = Engine(args.ollama, args.model, args.keep_alive)
    Handler.engine = engine
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"[mathboard] port {args.port} is busy - is Mathboard already running? Opening {url}")
        if not args.no_browser:
            open_board(url, app_window=not args.tab)
        return

    threading.Thread(target=engine.activate, args=(args.model,), daemon=True).start()
    print(f"[mathboard] {url}  model={args.model}  (Ctrl+C to stop)")
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
