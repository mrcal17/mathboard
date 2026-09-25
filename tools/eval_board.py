#!/usr/bin/env python3
"""Replay a labelled set of board crops against a recognition backend and score it.

Uses server.py's own code path (normalization, prompts, clean-up, ensemble), so a score here is what
the board would get. Requests run one at a time. Nothing is started or stopped: Ollama and, for
unimumer / ensemble, llama-server must already be running (docs/BOARD_BACKENDS.md).

The set is a folder with labels.json and images/<name>.png. The default, evalset/, is gitignored:
it holds your handwriting. Results go to <set>/results/<tag>.json.

    python tools/eval_board.py                                  # qwen3-vl, normalized
    python tools/eval_board.py --no-normalize --tag qwen-raw    # as the board sends it today
    python tools/eval_board.py --backend unimumer --tag uni4b
    python tools/eval_board.py --backend ensemble --tag ens
    python tools/eval_board.py --import-debug --tag live        # score the replies saved next to each image
    python tools/eval_board.py --rescore evalset/results/uni4b.json
    python tools/eval_board.py --compare evalset/results/qwen-raw.json evalset/results/qwen.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server as S  # noqa: E402  (definitions only; main() is not run)


# ------------------------------------------------------------------ labels and scoring
def load_set(folder: Path) -> dict:
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    for lab in labels["images"].values():
        lab["_accept"] = {S.canonical(a) for a in lab.get("accept", [])}
    return labels


def is_correct(label: dict, latex: str) -> bool:
    c = S.canonical(latex)
    if c in label["_accept"]:
        return True
    rule = label.get("rule")
    if not rule:
        return False
    if "digits" in rule and re.sub(r"[^0-9]", "", c) != rule["digits"]:
        return False
    body = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", c)
    return (all(S.canonical(x) in c for x in rule.get("require", []))
            and not any(S.canonical(x) in body for x in rule.get("forbid", [])))


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else 0


def rescore(items: dict, primary: str | None = None) -> None:
    """Re-derive latex from the stored raw replies, so clean-up changes show without new requests."""
    for it in items.values():
        it["primary"] = primary or it.get("primary")
        for c in it.get("candidates", []):
            c["latex"] = S.postprocess(c["backend"], c["raw"])
        cands = it.get("candidates", [])
        if len(cands) == 2:
            it["agree"] = S.canonical(cands[0]["latex"]) == S.canonical(cands[1]["latex"])
        if cands:
            best = next((c for c in cands if c["backend"] == it.get("primary")), cands[0])
            it["latex"] = best["latex"]


def report(labels: dict, run: dict, show: str) -> dict:
    items = run["items"]
    names = [n for n in labels["images"] if n in items]
    ok = {n: is_correct(labels["images"][n], items[n]["latex"]) for n in names}
    ens = any(len(items[n].get("candidates", [])) == 2 for n in names)
    if show != "none":
        for n in names:
            it = items[n]
            if show == "wrong" and ok[n]:
                continue
            lab = labels["images"][n]
            cells = [n, f"{lab['category'][:14]:<14}", "ok" if ok[n] else " x", f"{it.get('ms') or 0:>5}",
                     f"{it.get('prompt_tokens') or 0:>5}"]
            if ens:
                q, u = (next((c for c in it["candidates"] if c["backend"] == b), {}) for b in ("qwen", "unimumer"))
                cells += ["=" if it.get("agree") else "!", f"q:{q.get('latex', '?')[:38]:<38}", f"u:{u.get('latex', '?')[:38]}"]
            else:
                cells.append(it["latex"][:80] or "(none)")
            print("  ".join(cells))
    total = sum(ok.values())
    print(f"\n{run['meta'].get('tag', '')}: {total}/{len(names)} correct  "
          f"({run['meta'].get('backend')}, normalize={run['meta'].get('normalize')})")
    cats: dict[str, list[int]] = {}
    for n in names:
        c = cats.setdefault(labels["images"][n]["category"], [0, 0])
        c[0] += ok[n]
        c[1] += 1
    print("  by category: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(cats.items())))
    ms = [items[n]["ms"] for n in names if items[n].get("ms")]
    if ms:
        print(f"  latency ms: mean {sum(ms) / len(ms):.0f}, p50 {pct(ms, 50):.0f}, p95 {pct(ms, 95):.0f}, max {max(ms)}")
    toks = [items[n]["prompt_tokens"] for n in names if items[n].get("prompt_tokens")]
    if toks:
        print(f"  prompt tokens: min {min(toks)}, median {pct(toks, 50)}, max {max(toks)}")
    norm_ms = [c["norm"]["ms"] for n in names for c in items[n].get("candidates", []) if c.get("norm", {}).get("ms") is not None]
    if norm_ms:
        print(f"  normalization ms: mean {sum(norm_ms) / len(norm_ms):.0f}, max {max(norm_ms)}")
    out = {"correct": total, "total": len(names)}
    if ens:
        agree = [n for n in names if items[n].get("agree")]
        dis = [n for n in names if not items[n].get("agree")]

        def right(n: str, b: str) -> bool:
            c = next((c for c in items[n]["candidates"] if c["backend"] == b), None)
            return bool(c) and is_correct(labels["images"][n], c["latex"])

        print(f"  agree on {len(agree)}/{len(names)}; right when they agree: {sum(ok[n] for n in agree)}/{len(agree)}")
        print(f"  disagree on {len(dis)}: qwen right {sum(right(n, 'qwen') for n in dis)}, "
              f"unimumer right {sum(right(n, 'unimumer') for n in dis)}, "
              f"either right {sum(right(n, 'qwen') or right(n, 'unimumer') for n in dis)}")
        out.update(agree=len(agree), agree_right=sum(ok[n] for n in agree), disagree=len(dis))
    return out


# ------------------------------------------------------------------ running
def make_engine(args) -> S.Engine:
    backends = {}
    if args.backend in ("qwen", "ensemble"):
        backends["qwen"] = S.OllamaBackend(args.ollama, args.model, "30m")
    if args.backend in ("unimumer", "ensemble"):
        backends["unimumer"] = S.LlamaServerBackend(args.unimumer)
    eng = S.Engine(backends, args.backend, normalize=not args.no_normalize, primary=args.primary)
    for name, b in backends.items():
        b.activate()  # checks it is up, then one blank warm-up image (not timed)
        if b.state != "ready":
            sys.exit(f"{name} is not ready: {b.error}")
    return eng


def run_set(args, labels: dict, folder: Path) -> dict:
    eng = make_engine(args)
    names = [n for n in labels["images"] if not args.only or n in args.only.split(",")]
    items = {}
    for i, n in enumerate(names, 1):
        b64 = base64.b64encode((folder / "images" / f"{n}.png").read_bytes()).decode()
        r = eng.recognize(b64)
        items[n] = {k: r[k] for k in ("latex", "raw", "ms", "model", "prompt_tokens", "agree", "candidates")}
        items[n]["primary"] = eng.primary
        print(f"[{i}/{len(names)}] {n} {r['ms']} ms  {r['latex'] or '(none)'}", file=sys.stderr)
    models = {n: b.model for n, b in eng.backends.items()}
    return {"meta": {"tag": args.tag, "backend": args.backend, "models": models, "normalize": eng.normalize,
                     "norm": {n: S.NORM[n] for n in eng.backends} if eng.normalize else None,
                     "primary": eng.primary, "date": time.strftime("%Y-%m-%d %H:%M")},
            "items": items}


def import_debug(labels: dict, folder: Path, tag: str) -> dict:
    """The live replies server.py saved next to each image (images/<name>.json), as a run."""
    items = {}
    for n in labels["images"]:
        p = folder / "images" / f"{n}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            cand = {"backend": d.get("backend", "qwen"), "model": d.get("model"), "raw": d.get("raw", ""),
                    "ms": d.get("ms"), "prompt_tokens": d.get("prompt_tokens")}
            items[n] = {**cand, "latex": "", "agree": None, "candidates": [cand]}
    run = {"meta": {"tag": tag, "backend": "live replies", "normalize": None}, "items": items}
    rescore(items)
    return run


def compare(labels: dict, a: dict, b: dict) -> None:
    for n in labels["images"]:
        if n in a["items"] and n in b["items"]:
            ra, rb = (is_correct(labels["images"][n], r["items"][n]["latex"]) for r in (a, b))
            if ra != rb:
                print(f"{'+' if rb else '-'} {n} {labels['images'][n]['category']:<14} "
                      f"{a['items'][n]['latex'] or '(none)'!r} -> {b['items'][n]['latex'] or '(none)'!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a recognition backend on a labelled set of board crops")
    ap.add_argument("--set", default=str(ROOT / "evalset"), help="folder with labels.json and images/")
    ap.add_argument("--backend", default="qwen", choices=S.MODES)
    ap.add_argument("--model", default=S.DEFAULT_MODEL, help="Ollama model for the qwen backend")
    ap.add_argument("--ollama", default=S.DEFAULT_OLLAMA)
    ap.add_argument("--unimumer", default=S.DEFAULT_UNIMUMER or "http://127.0.0.1:8792", help="llama-server URL")
    ap.add_argument("--primary", choices=("qwen", "unimumer"),
                    help=f"ensemble: whose answer is `latex` when they disagree (default {S.ENSEMBLE_PRIMARY}; "
                         "with --rescore, the one the run used)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--symbol-target", type=int, default=0,
                    help="override the symbol height (px) the normalization aims for, for every backend in the run")
    ap.add_argument("--only", default="", help="comma-separated image names")
    ap.add_argument("--tag", default="", help="results file name (default: backend and normalization)")
    ap.add_argument("--show", default="all", choices=("all", "wrong", "none"))
    ap.add_argument("--rescore", metavar="RESULTS", help="score a saved run again (clean-up and labels as they are now)")
    ap.add_argument("--import-debug", action="store_true", help="score the replies saved with each image")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="list images whose score differs between two runs")
    args = ap.parse_args()

    folder = Path(args.set)
    labels = load_set(folder)
    if args.compare:
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        for r in (a, b):
            rescore(r["items"])
        report(labels, a, "none")
        report(labels, b, "none")
        return compare(labels, a, b)
    if args.rescore:
        run = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        rescore(run["items"], args.primary)
        report(labels, run, args.show)
        return
    args.primary = args.primary or S.ENSEMBLE_PRIMARY
    if args.symbol_target:
        for prof in S.NORM.values():
            prof["symbol"] = args.symbol_target
    args.tag = args.tag or ("live" if args.import_debug else f"{args.backend}{'-raw' if args.no_normalize else ''}")
    run = import_debug(labels, folder, args.tag) if args.import_debug else run_set(args, labels, folder)
    run["summary"] = report(labels, run, args.show)
    out = folder / "results" / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
