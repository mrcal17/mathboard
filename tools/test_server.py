#!/usr/bin/env python3
"""Unit tests for server.py's text and image helpers. No model, no network, no running server.

    python tools/test_server.py
"""
from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server as S  # noqa: E402  (definitions only; main() is not run)


class CleanLatex(unittest.TestCase):
    def test_wrappers_and_none(self):
        self.assertEqual(S.clean_latex("$x^2$"), "x^2")
        self.assertEqual(S.clean_latex("```latex\nx+1\n```"), "x+1")
        self.assertEqual(S.clean_latex("NONE"), "")
        self.assertEqual(S.clean_latex("NONE."), "")
        self.assertEqual(S.clean_latex(r"\text{NONE}"), "")
        self.assertEqual(S.clean_latex(r"\mathrm{None}"), "")

    def test_text_dash_is_a_minus(self):
        self.assertEqual(S.clean_latex(r"\text{---}"), "-")
        self.assertEqual(S.clean_latex(r"\text{--}"), "-")
        self.assertEqual(S.clean_latex(r"\text{-}"), "-")
        self.assertEqual(S.clean_latex(r"\text{--} x"), "- x")
        self.assertEqual(S.clean_latex(r"\text{Word}"), r"\text{Word}")

    def test_empty_results(self):
        for t in (r"\begin{bmatrix} \end{bmatrix}", r"\begin{bmatrix} \end{bmatrix} =",
                  r"x \geq \begin{bmatrix} \end{bmatrix}", r"\begin{pmatrix} & \\ & \end{pmatrix}",
                  r"\text{ }", "{}", r"\quad", "\\\\", "&", "  "):
            self.assertEqual(S.clean_latex(t), "", t)
        self.assertEqual(S.clean_latex(r"\begin{bmatrix} 1 \end{bmatrix}"), r"\begin{bmatrix} 1 \end{bmatrix}")
        self.assertEqual(S.clean_latex("="), "=")

    def test_newlines_stack(self):
        two = "A x = b\n" + r"A = \begin{bmatrix} 1 & 0 \\ 4 & 9 \end{bmatrix}"
        self.assertEqual(S.clean_latex(two),
                         r"\begin{aligned} A x &= b \\ A &= \begin{bmatrix} 1 & 0 \\ 4 & 9 \end{bmatrix} \end{aligned}")
        self.assertEqual(S.clean_latex("x + 1\ny = 2"), r"\begin{gathered} x + 1 \\ y = 2 \end{gathered}")
        self.assertEqual(S.clean_latex("a = b \\\\\nc = d"), r"\begin{aligned} a &= b \\ c &= d \end{aligned}")
        self.assertEqual(S.clean_latex("x\n\n"), "x")
        self.assertEqual(S.clean_latex("a &= b\nc &= d"), r"\begin{aligned} a &= b \\ c &= d \end{aligned}")

    def test_newlines_inside_environments_stay(self):
        m = "\\begin{bmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{bmatrix}"
        self.assertEqual(S.clean_latex(m), m)
        a = "\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}"
        self.assertEqual(S.clean_latex(a), a)

    def test_matrix_repairs(self):
        self.assertEqual(S.clean_latex(r"bmatrix{1 & 2 & 3}"), r"\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}")
        self.assertEqual(S.clean_latex(r"\begin{bmatrix} ( & 0 \end{bmatrix}"), r"\begin{bmatrix} 1 & 0 \end{bmatrix}")
        self.assertEqual(S.clean_latex(r"\begin {pmatrix} x \end {pmatrix}"), r"\begin{pmatrix} x \end{pmatrix}")

    def test_matrix_of_spacing_is_empty(self):
        for t in (r"\begin{bmatrix} \text{ } \\ \end{bmatrix}", r"\begin{pmatrix} \quad & \, \end{pmatrix} = x"):
            self.assertEqual(S.clean_latex(t), "", t)
        self.assertEqual(S.clean_latex(r"\begin{bmatrix} \text{a} \end{bmatrix}"), r"\begin{bmatrix} \text{a} \end{bmatrix}")

    def test_lone_cell_matrix_unwraps(self):
        self.assertEqual(S.postprocess("unimumer", r"\begin{matrix} \frac { 1 } { 2 } \end{matrix}"), r"\frac{1}{2}")
        for keep in (r"\begin{matrix} 1 & 2 \end{matrix}", r"\begin{matrix} 1 \\ 2 \end{matrix}",
                     r"\begin{matrix} a \end{matrix} + \begin{matrix} b \end{matrix}", r"\begin{bmatrix} 1 \end{bmatrix}"):
            self.assertEqual(S.clean_latex(keep), keep)
        self.assertEqual(S.clean_latex(r"\begin{matrix} \begin{pmatrix} 1 & 0 \end{pmatrix} \end{matrix}"),
                         r"\begin{pmatrix} 1 & 0 \end{pmatrix}")


class UniMuMerOutput(unittest.TestCase):
    def test_tidy_spacing(self):
        self.assertEqual(S.tidy_spacing(r"\frac { \partial L } { \partial w }"), r"\frac{\partial L}{\partial w}")
        self.assertEqual(S.tidy_spacing(r"\alpha x + \beta"), r"\alpha x+\beta")
        self.assertEqual(S.tidy_spacing("W o r d +"), "Word+")
        self.assertEqual(S.tidy_spacing(r"x ^ { 2 } = 4"), "x^{2}=4")

    def test_postprocess(self):
        self.assertEqual(S.postprocess("unimumer", r"\begin {bmatrix} 1 & 2 \\ 3 & 4 \end {bmatrix}"),
                         r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}")
        self.assertEqual(S.postprocess("qwen", "2 + 2 = 4"), "2 + 2 = 4")


class Canonical(unittest.TestCase):
    def same(self, a, b):
        self.assertEqual(S.canonical(a), S.canonical(b), (a, b))

    def differ(self, a, b):
        self.assertNotEqual(S.canonical(a), S.canonical(b), (a, b))

    def test_spelling_variants_match(self):
        self.same("x^{2}+1", "x ^ 2 + 1")
        self.same(r"\text{Word}+84", "Word + 84")
        self.same(r"\left( a \right)", "(a)")
        self.same(r"\dfrac{d}{dx}", r"\frac { d } { d x }")
        self.same(r"\alpha x", r"\alpha  x")
        self.same(r"\begin{aligned} A x &= b \\ A &= 1 \end{aligned}", r"Ax=b \\ A=1")
        self.same(r"\left[\begin{matrix}1&2\end{matrix}\right]", r"\begin{bmatrix} 1 & 2 \end{bmatrix}")
        self.same(r"\begin{bmatrix} 1 & 0 \\ 4 & 9 \end{bmatrix}", r"\begin{bmatrix}1&0\\4&9\end{bmatrix}")

    def test_different_math_differs(self):
        self.differ("2+2", "2+\\infty")
        self.differ(r"\begin{bmatrix}1&0\\4&9\end{bmatrix}", r"\begin{bmatrix}1&04&9\end{bmatrix}")
        self.differ(r"\alpha x", r"\alphax")
        self.differ("x_2", "x^2")
        self.differ(r"\partial", "d")
        self.differ(r"Ax=b \\ A=1", "Ax=bA=1")

    def test_row_break_survives(self):
        self.assertIn("\\\\", S.canonical(r"a \\ b"))


@unittest.skipIf(S.Image is None, "Pillow is not installed")
class Normalize(unittest.TestCase):
    @staticmethod
    def page(symbols=3, sym=100, stroke=6, size=(700, 300)):
        """A white image with `symbols` vertical bars (like 1s) `sym` px tall, plus a minus sign."""
        from PIL import ImageDraw
        im = S.Image.new("RGB", size, "white")
        d = ImageDraw.Draw(im)
        for i in range(symbols):
            x = 60 + i * 120
            d.line([(x, 100), (x + 10, 100 + sym)], fill="black", width=stroke)
        d.line([(60 + symbols * 120, 150), (60 + symbols * 120 + 80, 150)], fill="black", width=stroke)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def test_estimate_ignores_bars(self):
        png = self.page(sym=100)
        est = S.estimate_symbol_px(S.Image.open(io.BytesIO(png)).convert("L"))
        self.assertAlmostEqual(est, 100, delta=12)

    def test_qwen_profile_pads_to_a_megapixel(self):
        out, info = S.normalize_image(self.page(sym=150), S.NORM["qwen"])
        self.assertTrue(info["applied"])
        self.assertEqual(info["source"], "estimate")
        w, h = S.Image.open(io.BytesIO(out)).size
        self.assertGreaterEqual(w * h, S.NORM["qwen"]["min_area"])
        self.assertLessEqual(w * h, S.NORM["qwen"]["max_area"] * 1.05)
        self.assertAlmostEqual(info["scale"], 64 / 150, delta=0.06)

    def test_hint_wins_and_unimumer_budget(self):
        out, info = S.normalize_image(self.page(sym=150), S.NORM["unimumer"], symbol_px=40)
        self.assertEqual(info["source"], "hint")
        w, h = S.Image.open(io.BytesIO(out)).size
        self.assertGreaterEqual(w * h, S.NORM["unimumer"]["min_area"])
        self.assertLessEqual(w * h, S.NORM["unimumer"]["max_area"] * 1.05)

    def test_blank_is_left_alone(self):
        png = base64.b64decode(S.blank_png())
        out, info = S.normalize_image(png, S.NORM["qwen"])
        self.assertFalse(info["applied"])
        self.assertEqual(out, png)


class EngineRouting(unittest.TestCase):
    class Fake:
        def __init__(self, name, raw):
            self.name, self.raw, self.calls, self.state = name, raw, 0, "ready"

        def activate(self):
            self.state = "ready"

        def recognize(self, image_b64):
            self.calls += 1
            return {"raw": self.raw, "ms": 5, "model": self.name + "-model", "prompt_tokens": 100, "timings": {}}

        def status(self):
            return {"ready": self.state == "ready", "state": self.state, "model": self.name + "-model",
                    "error": "", "last_ms": None}

    def engine(self, qwen_raw, uni_raw, default="ensemble"):
        return S.Engine({"qwen": self.Fake("qwen", qwen_raw), "unimumer": self.Fake("unimumer", uni_raw)},
                        default, normalize=False)

    def test_ensemble_agreement(self):
        eng = self.engine("2 + 2 = 4", "2 + 2 = 4")
        r = eng.recognize(S.blank_png())
        self.assertTrue(r["agree"])
        self.assertEqual(len(r["candidates"]), 2)
        self.assertEqual(r["latex"], S.postprocess(eng.primary, "2 + 2 = 4"))
        self.assertFalse(r["empty"])
        self.assertEqual(r["prompt_tokens"], 100)

    def test_ensemble_disagreement_uses_primary(self):
        eng = self.engine("2 + \\infty", "2 + 2")
        r = eng.recognize(S.blank_png())
        self.assertFalse(r["agree"])
        self.assertEqual(r["latex"], S.postprocess(eng.primary, "2 + \\infty" if eng.primary == "qwen" else "2 + 2"))

    def test_single_mode_and_override(self):
        eng = self.engine("NONE", "x", default="qwen")
        self.assertEqual(eng.backends["unimumer"].state, "idle")  # outside the default mode: not loaded yet
        r = eng.recognize(S.blank_png())
        self.assertIsNone(r["agree"])
        self.assertTrue(r["empty"])
        self.assertEqual(r["latex"], "")
        self.assertEqual([c["backend"] for c in r["candidates"]], ["qwen"])
        r = eng.recognize(S.blank_png(), backend="unimumer")
        self.assertEqual(r["latex"], "x")
        self.assertEqual(eng.backends["unimumer"].state, "ready")
        with self.assertRaises(S.BackendError):
            S.Engine({"qwen": self.Fake("qwen", "x")}, "qwen").recognize(S.blank_png(), backend="ensemble")

    def test_status_lists_backends(self):
        st = self.engine("x", "x", default="qwen").status()
        self.assertEqual(st["backend"], "qwen")
        self.assertTrue(st["ready"])
        self.assertEqual([b["name"] for b in st["backends"]], ["qwen", "unimumer", "ensemble"])
        self.assertEqual([b["default"] for b in st["backends"]], [True, False, False])
        self.assertEqual([b["state"] for b in st["backends"]], ["ready", "idle", "idle"])


class RemoteAccess(unittest.TestCase):
    """MATHBOARD_TOKEN: requests through a tunnel need the token; requests on this machine don't."""

    @classmethod
    def setUpClass(cls):
        import http.client
        import threading
        cls.http = http.client
        cls.saved = S.ACCESS_TOKEN
        S.ACCESS_TOKEN = "s3cret-token"
        S.Handler.engine = None  # only static files are fetched here
        cls.server = S.BoardServer(("127.0.0.1", 0), S.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        S.ACCESS_TOKEN = cls.saved

    def get(self, path, headers=None, method="GET", body=None):
        c = self.http.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        r.read()
        c.close()
        return r

    def test_local_requests_need_nothing(self):
        self.assertEqual(self.get("/index.html").status, 200)

    def test_tunnel_requests_need_the_token(self):
        tunnel = {"Host": "board.example.com", "Cf-Connecting-Ip": "203.0.113.7"}
        self.assertEqual(self.get("/index.html", tunnel).status, 401)
        self.assertEqual(self.get("/?token=wrong", tunnel).status, 401)
        self.assertEqual(self.get("/api/recognize", tunnel, "POST", b"{}").status, 401)
        r = self.get("/?token=s3cret-token", tunnel)
        self.assertEqual(r.status, 303)
        self.assertEqual(r.getheader("Location"), "/")
        self.assertIn("mb_token=s3cret-token", r.getheader("Set-Cookie"))
        self.assertIn("HttpOnly", r.getheader("Set-Cookie"))
        self.assertEqual(self.get("/index.html", {**tunnel, "Cookie": "mb_token=s3cret-token"}).status, 200)
        self.assertEqual(self.get("/index.html", {**tunnel, "Cookie": "mb_token=nope"}).status, 401)
        # a forwarded request is remote even if it claims a local Host
        self.assertEqual(self.get("/index.html", {"Host": "127.0.0.1", "X-Forwarded-For": "203.0.113.7"}).status, 401)


if __name__ == "__main__":
    unittest.main(verbosity=1)
