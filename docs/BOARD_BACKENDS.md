# Board backends: qwen3-vl, Uni-MuMER and the ensemble

How `server.py` reads handwriting since 2026-09-25, what each backend scored on the labelled debug images, how to
set up Uni-MuMER, and the API the board client builds against. Background: [BOARD_DIAGNOSIS.md](BOARD_DIAGNOSIS.md)
(why the board misreads) and [BOARD_RESEARCH.md](BOARD_RESEARCH.md) (why Uni-MuMER).

## Summary

`server.py` has three recognizers, chosen with `--backend` or `MATHBOARD_BACKEND`:

| Backend | What runs | Default |
|---|---|---|
| `qwen` | qwen3-vl:8b-instruct in Ollama, the existing prompt | yes, for now |
| `unimumer` | Uni-MuMER (Qwen3-VL 2B or 4B fine-tuned for handwritten math) in llama.cpp's `llama-server` | |
| `ensemble` | both at once; the reply says whether they agree | |

Before either model sees a crop, the server now rescales it so symbols are about 64 px (qwen) or 80 px (Uni-MuMER)
tall ([section 4](#4-image-normalization)). This needs Pillow and can be turned off with `--no-normalize`.

**Results on the 37 labelled debug images** (exact match after LaTeX normalization, [section 6](#6-eval-harness)):

| Run | Correct | Latency p50 | Prompt tokens (median) |
|---|---|---|---|
| Live replies as recorded (scored with today's clean-up) | 16/37 | about 1.0 s | |
| qwen3-vl, crops as the board sends them | 16/37 | 1.07 s | 1416 |
| **qwen3-vl, normalized** | **26/37** (twice) | 0.93 to 1.09 s | 1412 |
| Uni-MuMER 4B Q8_0, as sent | 30/37 | 0.13 s | 251 |
| **Uni-MuMER 4B Q8_0, normalized** | **29/37** | 0.10 to 0.16 s | 107 |
| Uni-MuMER 2B Q8_0, as sent | 26/37 | 0.09 s | 251 |
| **Uni-MuMER 2B Q8_0, normalized** | **29/37** (twice) | 0.08 to 0.14 s | 107 |
| Ensemble qwen + 4B (Uni-MuMER's answer as `latex`) | 29/37 | 0.93 s | |
| Ensemble qwen + 2B (Uni-MuMER's answer as `latex`) | 29/37 | 0.46 s | |

**Ensemble agreement** (both normalized):

| Pair | They agree | Right when they agree | They disagree | qwen right | Uni-MuMER right | Either right |
|---|---|---|---|---|---|---|
| qwen + 4B | 17/37 | 16/17 | 20 | 10 | 13 | 18 |
| qwen + 2B | 18/37 | 18/18 | 19 | 8 | 11 | 15 |

The one agreeing miss is a long written word: both models dropped the same letter.

**Recommendation**

- **With today's client**, which only uses `latex`: `unimumer` with the 2B. It gets 29/37 against 26/37, answers in
  about 0.1 s instead of about 1 s, and needs 3.1 GB of VRAM instead of 5.8 GB. It is weaker on written words and on
  two lines in one group (see [section 1](#1-measurements)).
- **Once the client does preview-then-commit**: `ensemble` with the 2B. `latex` is Uni-MuMER's answer (the same 29/37),
  and `agree` is the auto-commit signal: it was right 18 times out of 18, on about half of the groups. When they
  disagree, the client shows both answers, and one of them was right 15 times out of 19. This costs qwen's latency
  (0.5 to 1 s) and 12.8 GB of VRAM in total.
- **2B over 4B.** Normalized, they tie at 29/37. The 4B coped better with unnormalized crops (30 against 26). It
  needs 2.4 GB more VRAM, and next to qwen3-vl it fills the card to 15.3 of 16.3 GB, which leaves no room for OBS or
  anything else. The 4B is a reasonable choice for `unimumer` alone (5.5 GB).
- **Keep normalization on.** qwen went from 16/37 to 26/37 with no latency cost. For Uni-MuMER it made no measurable
  difference (4B 29 against 30, 2B 29 against 26). It stays on so that Uni-MuMER sees crops in its training
  range whatever the client's raster does.

The code default stays `qwen`. Switching is one line in `mathboard.local.bat` ([section 2](#2-setup)).

## 1. Measurements

Machine: RTX 5070 Ti 16 GB (WDDM, driver 610.88), Ollama 0.30.9 (qwen3-vl:8b-instruct Q4_K_M, 5.8 GB resident),
llama.cpp b11178. All requests were sequential, temperature 0. The set is the 37 labelled images of
BOARD_DIAGNOSIS.md, so a difference of 1 or 2 images is noise.

**By category** (normalized runs):

| Category | Images | qwen3-vl | Uni-MuMER 4B | Uni-MuMER 2B |
|---|---|---|---|---|
| looped 2 | 5 | 3 | 5 | 4 |
| lone d / dx | 4 | 2 | 3 | 4 |
| sigma or partial, alone | 2 | 0 | 2 | 2 |
| fraction pieces with a partial | 3 | 1 | 1 | 2 |
| half-written matrix | 3 | 2 | 3 | 2 |
| words | 5 | 4 | 2 | 2 |
| two lines in one group | 1 | 1 | 0 | 0 |
| matrices (one clean, one with overlapping strokes) | 2 | 1 | 1 | 1 |
| lines, digits, letters, bars, arrow, minus infinity | 12 | 12 | 12 | 12 |

- **What Uni-MuMER fixes.** Every looped 2 in the 4B run (`2 + 2`, the isolated 2 that qwen reads as `x` or `\alpha`,
  and `2 3 ]`), lone `d`s that qwen answers NONE, and the loop glyphs (`\sigma`). This is the "2s become x or infinity"
  complaint.
- **What it can't do.** It was never trained to write `\text` or stacked lines. "Word" comes back as `W_{ord}`, and
  `Ax=b` over `A=[...]` comes back as one line. It never answers NONE: the arrow became `\uparrow` (accepted), and a
  real doodle would get some LaTeX. In the ensemble all of these show up as disagreements, so they are not
  auto-committed.
- **Determinism.** Repeat runs gave identical replies on all 37 images for qwen3-vl (normalized) and the 2B.
- **Symbol target for Uni-MuMER.** 64, 80 and 110 px all scored 29/37 with the 4B, so the 80 px default is not
  critical. It comes from 76 sampled training images, measured at the model's input: median 77 px overall and
  83 px for MathWriting.

**Latency.** Per request, from the harness, including HTTP and normalization:

- qwen3-vl: p50 0.43 to 1.06 s for the model call, depending on the run, with single requests from 0.37 to 2.5 s.
  The GPU and CPU are shared with the desktop, and the spread comes from that, not from the settings: the same
  images with the same settings, run twice, gave p50 1.09 and 0.93 s.
- Uni-MuMER: p50 0.08 to 0.16 s. Prompts are about 100 tokens (64 image tokens plus the text), against about 1400
  for qwen3-vl.
- The ensemble takes as long as the slower of the two, which is qwen3-vl. The two requests run in parallel.
- Normalization: p50 7 to 11 ms for Uni-MuMER's small images, and p50 15 to 44 ms for qwen's 1 MP canvas, where
  the PNG encode dominates. The slowest was 0.3 s, on a busy CPU.

**VRAM** (`nvidia-smi` MiB divided by 1000, as in the diagnosis; whole card, desktop apps included):

| Loaded | Used of 16.3 GB |
|---|---|
| qwen3-vl in Ollama (baseline) | 9.3 to 9.6 GB |
| + Uni-MuMER 2B (Q8_0 + f16 mmproj, 2048 context) | 12.7 GB idle, 12.8 GB during the ensemble |
| + Uni-MuMER 4B (Q8_0 + f16 mmproj, 2048 context) | 15.1 GB idle, 15.3 GB during the ensemble |

Power reached 170 to 218 W during the 4B ensemble run, and latency stayed normal, so nothing spilled into system RAM.
If the card is ever close to full and replies get slow while the power stays low, the spill has started: use the 2B
or stop one model.

## 2. Setup

Nothing here goes into git. Binaries and models live outside the repo.

**llama.cpp.** Release b11178 (2026-09-25) from <https://github.com/ggml-org/llama.cpp/releases>: the two files
`llama-b11178-bin-win-cuda-13.4-x64.zip` and `cudart-llama-bin-win-cuda-13.4-x64.zip`, unzipped together into
`C:\Users\landa\tools\llama.cpp\` (`VERSION.txt` there records the build). Use a CUDA 13 build: the CUDA 12.4 build
is compiled without Blackwell (sm_120) kernels. This one loads the 4B in about 2.5 s on the 5070 Ti.

**Models.** From mradermacher's GGUF conversions of phxember's Uni-MuMER weights, in `C:\Users\landa\models\uni-mumer\`
(`sha256.txt` there matches the Hugging Face LFS hashes):

| File | Size | Source |
|---|---|---|
| `Uni-MuMER-Qwen3-VL-2B.Q8_0.gguf` | 1.83 GB | [Uni-MuMER-Qwen3-VL-2B-GGUF](https://huggingface.co/mradermacher/Uni-MuMER-Qwen3-VL-2B-GGUF) |
| `Uni-MuMER-Qwen3-VL-2B.mmproj-f16.gguf` | 0.82 GB | same |
| `Uni-MuMER-Qwen3-VL-4B.Q8_0.gguf` | 4.28 GB | [Uni-MuMER-Qwen3-VL-4B-GGUF](https://huggingface.co/mradermacher/Uni-MuMER-Qwen3-VL-4B-GGUF) |
| `Uni-MuMER-Qwen3-VL-4B.mmproj-f16.gguf` | 0.84 GB | same |

The f16 vision projector was chosen over Q8_0 to keep the image encoder at full precision. It costs about 0.4 GB.

**Starting it.** `start_mathboard.bat` reads `MATHBOARD_BACKEND`. For `unimumer` or `ensemble`, it starts
`llama-server` in the same console, points `server.py` at it, and stops it by PID when the board stops. Closing the
console or pressing Ctrl+C stops it as well. If llama-server or a model file is missing, it says so and falls back to
`qwen`. If something is already listening on the port, it is reused. The log goes to
`%TEMP%\mathboard-llama-server.log`.

Settings for this machine go in `mathboard.local.bat` next to the launcher. That file is gitignored. For example:

```bat
rem Mathboard launcher settings for this PC.
set MATHBOARD_BACKEND=unimumer
rem The 4B instead of the default 2B:
rem set "UNIMUMER_MODEL=%USERPROFILE%\models\uni-mumer\Uni-MuMER-Qwen3-VL-4B.Q8_0.gguf"
rem set "UNIMUMER_MMPROJ=%USERPROFILE%\models\uni-mumer\Uni-MuMER-Qwen3-VL-4B.mmproj-f16.gguf"
rem Other overrides: LLAMA_SERVER (path to llama-server.exe), UNIMUMER_PORT (default 8792).
```

**By hand**, the same thing the launcher runs:

```bat
llama-server -m Uni-MuMER-Qwen3-VL-2B.Q8_0.gguf --mmproj Uni-MuMER-Qwen3-VL-2B.mmproj-f16.gguf ^
  --host 127.0.0.1 --port 8792 -ngl 999 -c 2048 -np 1 -fit off --cache-ram 0 ^
  --image-min-tokens 64 --image-max-tokens 256
python server.py --backend ensemble --unimumer http://127.0.0.1:8792
```

| Flag | Why |
|---|---|
| `--image-min-tokens 64 --image-max-tokens 256` | Uni-MuMER's training range: images capped at 262,144 px (LLaMA-Factory `image_max_pixels`) and raised to at least 65,536 px by the Qwen3-VL processor; one token is 32 x 32 px. llama-server warns that Qwen-VL needs 1024 tokens "on grounding tasks": that is about boxes, not transcription. |
| `-c 2048` | the training cutoff; a prompt here is 100 to 300 tokens and a reply is capped at 512 |
| `-ngl 999 -fit off` | everything on the GPU, with these settings kept as given |
| `-np 1 --cache-ram 0` | one slot, since `server.py` sends one request at a time per backend; no host-RAM prompt cache, since every prompt starts with a new image |

**Prompt.** Uni-MuMER gets its training prompt verbatim (`I have an image of a handwritten mathematical expression.
Please write out the expression of the formula in the image using LaTeX format.`), as a user message with the image
first and no system message. That is how the training rows are laid out (`<image>` then the text, LLaMA-Factory's
`qwen3_vl_nothink` template, no default system prompt), and llama-server's `/apply-template` confirms the model's
own chat template renders it the same way. Decoding is greedy (temperature 0); the authors' eval used 0.2. Its
spaced CROHME-style output (`x ^ { 2 }`) is tidied to `x^{2}` before the usual clean-up.

## 3. Server options

| Flag | Default | Meaning |
|---|---|---|
| `--backend NAME` | `qwen` (or `$MATHBOARD_BACKEND`) | `qwen`, `unimumer` or `ensemble` |
| `--unimumer URL` | `$MATHBOARD_UNIMUMER`, else `http://127.0.0.1:8792` when the backend needs it | llama-server base URL; set it with `--backend qwen` to allow per-request `unimumer` and `ensemble` |
| `--no-normalize` | normalization on | send crops exactly as the board drew them |
| `--model`, `--ollama`, `--keep-alive` | as before | the Ollama side |

A backend that is configured but not part of the default mode shows as `idle` in `/api/status`. The first request
that asks for it loads it, which is slow once.

In ensemble mode, `ENSEMBLE_PRIMARY` in `server.py` (`unimumer`) decides whose answer is `latex` when the two
disagree.

`MATHBOARD_DEBUG_DIR` moves the debug captures (default `debug/`), so a test server doesn't rotate the user's
last 40 images out. On Windows the server binds its port exclusively: a second launch sees "port busy" and
opens the running board instead of stacking another server on the same port.

Two clean-up rules came from the first run against real models: a reading that is one bracketless 1×1
`matrix` is unwrapped (Uni-MuMER wraps a lone `\frac{1}{2}` that way, which made every fraction disagree),
and a matrix holding only spacing, `&` or row breaks counts as empty.

## 4. Image normalization

`normalize_image()` runs for each backend before its request:

1. Composite onto white and take the grayscale.
2. **Symbol height.** Use the client's `symbol_px` hint if it was sent. Otherwise estimate it: the median height of
   the connected ink blobs, skipping dots and bars (a blob at most 3 strokes thick and at least 1.5 times as wide
   as tall: minus signs, fraction bars, the halves of `=`). If only bars are left, 0.6 of a bar's width counts. On
   the eval images this matches the diagnosis's estimator except where bars pulled that one down.
3. Crop to the ink. Scale it so the median symbol is `symbol` px tall, with a `margin` of white around it.
4. Pad, without scaling, up to the backend's pixel floor, keeping the aspect ratio. Shrink if the result would exceed the cap.

| Profile | Symbol | Margin | Floor | Cap | Why |
|---|---|---|---|---|---|
| `qwen` | 64 px | 0.75 symbol | 1 MP | 1.5 MP | Ollama enlarges anything under about 1 MP (`--image-min-tokens 1024`); 64 px was the best render in the diagnosis; a 3 MP canvas overflowed the 4096-token context |
| `unimumer` | 80 px | 0.25 symbol | 65,536 px | 262,144 px | its training range, tight crops like its training images, symbols at their median size |

Without Pillow, `server.py` still runs. Crops are then sent unscaled, and `/api/status` reports `normalize: false`.

## 5. API for the client

### `POST /api/recognize`

Request JSON:

| Field | Required | Meaning |
|---|---|---|
| `image` | yes | PNG as a data URL or bare base64, black ink on opaque white, as today |
| `symbol_px` | no | median handwritten symbol height in the image's own pixels (board px times the raster scale), skipping dots and bars (strokes over 4 times wider than tall). Values outside 4 to 4096 are ignored. Without it the server estimates it. |
| `strokes` | no | `[[[x, y, t_ms], ...], ...]` in image pixels. Only saved next to the debug image, for replays; never used for recognition. |
| `backend` | no | `"qwen"`, `"unimumer"` or `"ensemble"` for this request only. It must be one of the names in `/api/status` `backends`, otherwise the reply is 400. |

Response 200:

| Field | Meaning |
|---|---|
| `latex` | the best answer, cleaned. `""` when nothing was recognized. In ensemble mode, Uni-MuMER's answer (`ENSEMBLE_PRIMARY`). |
| `empty` | `true` when `latex` is `""`: NONE, an empty matrix, only spacing, or an empty reply. Keep the ink. |
| `raw`, `model`, `prompt_tokens` | the best answer's raw reply, model name and prompt token count |
| `ms` | wall time of the whole request, including normalization and, in ensemble mode, both models |
| `candidates` | `[{backend, model, latex, raw, ms, prompt_tokens, norm}]`, one per backend that answered, in the order qwen, unimumer |
| `agree` | ensemble: `true` when both answers are equal after `canonical()` (spacing, `^{2}` against `^2`, `\text{}`, `\left`, `aligned` wrappers ignored); both empty counts as agreeing. Single backend: `null`. |
| `backend` | the mode that ran |
| `timings`, `norm` | the best answer's timing split and what normalization did (`applied`, `symbol_px`, `source`: hint or estimate, `scale`, `size`) |
| `errors` | only when one ensemble member failed: `{backend: message}`. Then `candidates` has one entry and `agree` is `false`. |

Errors stay as before: a non-200 status with `{error}`. 400 means bad JSON, no image, undecodable base64 or an
unavailable backend. 502 means a backend failed or is unreachable (both of them, in ensemble mode).

### `GET /api/status`

The old fields (`ready`, `state`, `model`, `error`, `last_ms`) describe the default mode. In ensemble mode they cover
both models, and `ready` is true only when both are loaded. New fields:

- `backend`: the default mode;
- `normalize`: whether crops are rescaled;
- `backends`: `[{name, label, default, ready, state, model, error}]`, one per usable mode, in the order qwen,
  unimumer, ensemble. `state` is `ready`, `loading`, `idle` (configured, loads on first use), `downloading`,
  `starting` or `error`.

`GET /api/models` and `POST /api/model` still switch the Ollama model. In `unimumer` mode, `/api/models` lists the
llama-server model only.

### What the client should do

- **Preview-then-commit.** Keep recognizing after idle as today, but show `latex` as a preview near the ink and keep
  the ink. With `ensemble`, auto-commit only when `agree` is `true`. When it is `false`, offer
  `candidates[*].latex` as choices, with `latex` preselected, plus Edit and Write again. `empty` refers to `latex`
  only: in ensemble mode the other candidate may still have text, and it is worth offering.
- **One request in flight.** Each backend handles one request at a time. Keep the current rule: one request at a time,
  and drop replies for a group that changed since it was sent.
- **Symbol-size hint.** Send `symbol_px` once the grouping code measures glyph sizes (the writing size of Fix 1 in
  BOARD_DIAGNOSIS.md, times the raster scale). The numbers above were measured without hints, using the estimate.
  A hint matters most for a pen tablet: the finer pen changes stroke width and letter size, and normalization
  rescales by symbol height, not by stroke width.
- **Raster.** No change is needed. The server rescales for each backend, so the board can keep sending
  768 px-capped crops. If Fix 2 (fixed symbol scale in a 1 MP canvas) lands anyway, send `symbol_px` with it.
- **Second opinion.** A Re-recognize button can send `backend: "qwen"` or `"unimumer"`: a different model, rather
  than the same input at temperature 0, which returns the same answer.
- **Lines and words.** Uni-MuMER runs lines together and turns words into subscripts. Splitting a group into lines
  before recognizing (stage 3 in BOARD_RESEARCH.md) would help both models. Words written with the Math pen are
  better served by qwen.

## 6. Eval harness

`tools/eval_board.py` replays a labelled set through `server.py`'s own code path: normalization, prompts, clean-up
and ensemble. Requests run one at a time. The script doesn't start or stop anything: Ollama and, for `unimumer` or
`ensemble`, llama-server must already be running.

```bat
python tools/eval_board.py                                 :: qwen3-vl, normalized
python tools/eval_board.py --no-normalize --tag qwen-raw
python tools/eval_board.py --backend unimumer --tag uni2b
python tools/eval_board.py --backend ensemble --tag ens2b --show wrong
python tools/eval_board.py --import-debug --tag live      :: score the replies saved next to each image
python tools/eval_board.py --rescore evalset/results/ens2b.json --primary qwen
python tools/eval_board.py --compare evalset/results/qwen-raw.json evalset/results/qwen.json
```

It prints one line per image (category, right or wrong, ms, prompt tokens, answer; both answers and `=` or `!` in
ensemble mode). Then totals by category, latency (mean, p50, p95, max), prompt tokens, normalization time and, for
the ensemble, the agreement numbers. Every run is saved to `evalset/results/<tag>.json` with the raw replies.
`--rescore` recomputes the answers from those, so clean-up and label changes need no new requests. `--symbol-target N`
overrides the normalization target, for trying sizes.

**The set lives in `evalset/`, which is gitignored**, because it is your handwriting:

- `evalset/images/<name>.png`, plus `<name>.json`: the image and reply pairs copied from `debug/`. `debug/` keeps only
  the last 40, so copy new ones before they rotate out.
- `evalset/labels.json`:
  `{"images": {"<name>": {"category": ..., "intended": ..., "accept": [...]}}, "excluded": {...}}`. A reply counts as
  right when `canonical(reply)` equals the `canonical()` of one `accept` string, or when it passes the `rule`:
  `digits` (the reply's digits, in order), `require` and `forbid` (substrings; `forbid` is checked with
  `\begin{..}` and `\end{..}` removed). The rules are for half-written matrices, where any reasonable bracket
  spelling is fine. `""` in `accept` means NONE is right.

`tools/test_server.py` holds unit tests for the clean-up, `canonical()`, the normalization and the backend routing,
with no model or network: `python tools/test_server.py`.

## 7. Not verified

- **llama.cpp against transformers.** Uni-MuMER was only run through llama.cpp. The open report of a noisier Qwen3-VL
  vision path in llama.cpp ([#29251](https://github.com/ggml-org/llama.cpp/issues/29251)) was not checked against
  the reference transformers or vLLM implementation.
- **Ollama import** of the Uni-MuMER GGUF was not tried: llama-server worked and needs no import.
- **Pen tablet.** There is no tablet data. Everything above is mouse writing.
- **Set size.** 37 images, all from two sessions. Ties like 4B against 2B need a larger set, and so does every number
  in the tables above. Add images to `evalset/` as they come.
- **Launcher.** The llama-server start, the PID stop, the missing-file fallback and the environment hand-over were
  tested with a copy of `start_mathboard.bat` whose `python server.py` line was replaced by a health probe. A full run
  with `server.py`, and the Ctrl+C or window-close path, were not tested.
- **End to end.** The board and a real `server.py` in ensemble mode (with its own llama-server) were run together
  with mouse strokes in Playwright: request p50 917 ms (qwen 898, Uni-MuMER 62), unimumer alone 91 ms, VRAM
  12.8 GB of 16 during reads. Synthetic strokes say nothing about accuracy.
- **A dead ensemble member** still shows as ready in `/api/status`; the reply's `errors` field and a one-candidate
  "Unsure" chip are the only signs. A separate "degraded" state would need its own status field.
