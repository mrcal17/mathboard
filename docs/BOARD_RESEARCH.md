# Board research: reliable, pleasant handwritten math

Research notes for the Mathboard board tab (Math pen, stroke grouping, recognition, KaTeX). Written 2026-09-25.
This is research only: no model was downloaded, nothing was benchmarked on the GPU, and no code was changed.
Where a number is quoted it comes from the linked source, not from a measurement on this machine.
A separate note covers the diagnosis of the current pipeline from the debug images, so this file is about outside options.

Today the board works like this: strokes are grouped by bounding-box proximity (`REACH_X` 56 px, `REACH_Y` 16 px, with a
wider sideways reach for 1.5 s after the last stroke). After 700 ms idle, the group is rasterized (black on white, up to
768 px on the long side) and sent through `server.py` to Ollama `qwen3-vl:8b-instruct` at temperature 0. The typeset
result then replaces the ink in place.

## Summary

**Top recognition options**

| # | Option | Published handwriting accuracy | Size | Licence | Fit for this machine |
|---|---|---|---|---|---|
| 1 | **Uni-MuMER** (Qwen3-VL-2B/4B or Qwen3.5-2B/4B fine-tuned for handwritten math) | CROHME 2014 82-84 %, CROHME 2019 76-80 %, HME100K 69-70 %, MathWriting test 51-54 % exact match (69-72 % CDM) | 2.1-4.5 B; Q8_0 GGUF 1.8-4.3 GB plus a 0.45 GB mmproj | Apache-2.0 | GGUF with mmproj already on Hugging Face; llama.cpp route is the safer one, Ollama import of Qwen3-VL fine-tunes is buggy |
| 2 | **TexTeller 3** (ViT + TrOCR decoder) | Claimed CROHME 2014 88.0 %, HME100K 90.7 %, but the paper is withdrawn and test-set contamination is possible, so treat as unverified | 298 M | Apache-2.0 | `pip install texteller`, torch 2.6+ (cu128 fine), ONNX, `--num-beams` gives n-best |
| 3 | **Keep a general VLM** (the current qwen3-vl:8b), used as a second opinion | No CROHME exact-match number for Qwen3-VL-8B. On OmniHandwritingOCR it scores formula F1 79.9 single-line, 86.9 easy multi-line, 70.5 hard multi-line (a softer metric). Qwen2.5-VL-7B zero-shot gets 52.2 % CROHME average | 8 B | Apache-2.0 | Already running |

The jump from a zero-shot VLM to a handwriting fine-tune of the same family is the largest single gain in the literature
(Qwen2.5-VL-7B zero-shot 56.0 % on CROHME 2014 against 82-84 % for the Uni-MuMER fine-tunes). Model size does not close
that gap: Qwen2.5-VL-72B zero-shot only reaches 59.7 %.

**Top UX patterns worth copying**

1. **Recognize continuously, commit only on intent.**
   - Keep the ink and show the result as a small preview under it, not typeset in place. Typeset in Place was the slowest style in a controlled study.
   - None of the mature products surveyed replaces ink automatically on an idle timer.
2. **Mark only the ambiguous symbols, tap to pick an alternative.**
   - Sources: Apple Math Notes' blue dotted line, Math Input Panel, OneNote "Fix it".
   - Lock the choice so re-recognition does not undo it (MathBrush).
   - Token logprobs make this possible here.
3. **Grouping that knows about math** (MathPaper): per-symbol inflated boxes, operators never left alone, and re-grouping after every stroke. Add time-burst grouping and visible group outlines.
4. **Scratch-out to erase, available while writing** (MyScript, Mathpix, Apple, MathPad2). Count it only when it crosses ink.
5. **Lasso to regroup, merge or split** (OneNote, GoodNotes, Nebo).
6. **User choice of real-time or batch**, like Apple's Insert / Suggest / Off.
7. **For hard 2D input, make structure visible** with boxes for fractions and scripts (Math Boxes).

**Staged path**

- **Stage 0.** Build an evaluation set from the debug images and your own corrections.
- **Stage 1.** Changes that need no new model:
  - preview-then-commit;
  - tap-to-fix alternatives from logprobs, with locks;
  - scratch-out and lasso regroup;
  - math-aware, time-aware grouping;
  - a check of how much Ollama upscales each crop. It forces Qwen-VL images to at least 1024 visual tokens.
- **Stage 2.** A/B test Uni-MuMER-Qwen3-VL-4B (and 2B) behind a backend switch. Keep qwen3-vl:8b or TexTeller as a verifier, and auto-commit only when the two agree.
- **Stage 3.** The app owns coarse layout: lines become `aligned`, bracket-detected matrices become cells, plus structural boxes.
- **Stage 4.** A LoRA fine-tune on your own corrections plus MathWriting, and optionally stroke-order features.

---

## 1. Recognition

### 1.1 Reading the numbers

- **ExpRate** is the exact match of the whole expression after the paper's own LaTeX normalization. **CDM** is a visual
  match: it renders both LaTeX strings and compares the images, so it forgives `x^{2}` against `x^2` spacing and brace
  differences. **CER** is the token error rate. The numbers are not comparable across papers because each normalizes differently.
- **CROHME** (2014/2016/2019 test sets, about 1,000 expressions each) has only 101 symbol classes
  ([CROHME tasks](https://www.isical.ac.in/~crohme/CROHME_tasks1.html)). Models trained only on it have vocabularies of
  about 112 tokens and do not know `\infty`-heavy or matrix-heavy lecture notation well.
- **HME100K** is photographed paper handwriting (245 symbols). **MathWriting** (Google, 2024) is 230k human plus 400k
  synthetic expressions written on touch screens, stored as online ink (x, y, t per point) with LaTeX labels
  ([paper](https://arxiv.org/html/2404.10690), [repo](https://github.com/google-research/google-research/blob/master/mathwriting/README.md)).
  Data licence: CC BY-NC-SA 4.0. **MathWriting is the closest public match to Mathboard input**, and it is the hardest
  benchmark: Uni-MuMER drops from about 80 % on CROHME to 51-54 % exact match on MathWriting.

### 1.2 Comparison

ExpRate in % unless noted. C14/C16/C19 = CROHME test years. HW = handwriting.

| Option | Type | Published HW accuracy | Params / files | Licence | Windows + sm_120 practicality | Maturity | n-best / confidence |
|---|---|---|---|---|---|---|---|
| **qwen3-vl:8b-instruct** (current) | general VLM, zero-shot | no CROHME exact match published. OmniHandwritingOCR formula F1: 79.88 single-line, 86.92 easy multi-line, 70.45 hard multi-line ([paper, Tables 6-7](https://arxiv.org/html/2608.18586)). Proxy on CROHME: Qwen2.5-VL-7B zero-shot C14 55.98, C16 50.92, C19 49.62, HME100K 54.57; Qwen2.5-VL-3B C14 38.64; Qwen2.5-VL-72B C14 59.74; GPT-4o C14 50.61; Gemini 2.5 Flash C14 58.01 ([Uni-MuMER paper, Table 1](https://arxiv.org/html/2505.23566)) | 8 B | Apache-2.0 | running today in Ollama | mature | Ollama `logprobs` / `top_logprobs` since v0.12.11 ([release](https://github.com/ollama/ollama/releases/tag/v0.12.11), [API](https://docs.ollama.com/api/generate)) |
| **Uni-MuMER** (NeurIPS 2025 spotlight; Qwen3-VL and Qwen3.5 variants released 2026-04-13) | VLM fully fine-tuned for HMER | Qwen3-VL-4B: C14 82.35, C16 79.34, C19 78.98, CROHME23 69.17, HME100K 69.79, **MathWriting test 53.15 (CDM 71.7)**. Qwen3-VL-2B: C14 83.27, C19 79.40, CROHME23 70.96, MathWriting 50.66. Qwen3.5-4B: MathWriting 54.32 ([4B card](https://huggingface.co/phxember/Uni-MuMER-Qwen3-VL-4B), [repo](https://github.com/BFlameSwift/Uni-MuMER)) | 2.1 / 4.4 B (Qwen3-VL), 2.2 / 4.5 B (Qwen3.5). 4B Q8_0 GGUF 4.28 GB + mmproj Q8_0 0.45 GB ([GGUF](https://huggingface.co/mradermacher/Uni-MuMER-Qwen3-VL-4B-GGUF)); 2B Q8_0 1.83 GB + mmproj ([GGUF](https://huggingface.co/mradermacher/Uni-MuMER-Qwen3-VL-2B-GGUF); the `cstr` GGUFs target the CrispEmbed runtime instead) | Apache-2.0 weights; training data repo MIT, but includes MathWriting (NC-SA) | Reference runtime is vLLM (no native Windows). On Windows: llama.cpp `llama-server --mmproj` (Qwen3-VL supported, [Qwen GGUF card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)) or transformers on torch cu128. Importing a Qwen3-VL fine-tune GGUF into Ollama is reported broken ([#13101](https://github.com/ollama/ollama/issues/13101), [#16264](https://github.com/ollama/ollama/issues/16264)); retest on Ollama 0.30.x, which now runs llama-server underneath. There is an open report that llama.cpp's Qwen3-VL vision path is noisier than transformers ([#29251](https://github.com/ggml-org/llama.cpp/issues/29251), not math-specific). Because Ollama 0.30+ is llama.cpp inside, that report applies to the current qwen3-vl:8b setup too | research code, active, GGUFs widely downloaded | LLM logprobs (llama-server and Ollama); eval uses temp 0.2 |
| **TexTeller 3.0** | ViT encoder + TrOCR decoder | paper claims C14 88.0, C16 85.9, C19 85.8, HME100K 90.7 ([v1](https://arxiv.org/html/2508.09220v1)). The paper was withdrawn twice (last 2026-01-10, [abs](https://arxiv.org/abs/2508.09220v4)), and the repo says its handwritten subset was collected "including both training and test sets" of open datasets ([README](https://github.com/OleehyO/TexTeller)). **Unverified.** | 298 M (1.19 GB fp32) ([HF](https://huggingface.co/OleehyO/TexTeller)) | Apache-2.0 | `uv pip install texteller`, torch 2.6+ so cu128 works; ONNX export; Ray Serve server (`texteller launch`) | maintained app, last model 2024-06 | `--num-beams`; HF `generate` can return n sequences with scores |
| **UniMERNet** T/S/B | Donut-Swin encoder-decoder | C14 67.4, C16 68.4, C19 65.4, HME100K 68.0 (as quoted in the [TexTeller paper](https://arxiv.org/html/2508.09220v1)); UniMER-Test HWE BLEU 0.883 / 0.889 / 0.895 ([paper](https://arxiv.org/html/2404.15254)) | 107 / 202 / 325 M | Apache-2.0 | PyTorch, torch 2.2+ | maintained (update 2025-09) | beam (not verified) |
| **PP-FormulaNet_plus** S/M/L | PaddleOCR formula model | no handwriting-only numbers; plus-L En-BLEU 92.22 on mostly printed sets ([card](https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-L)) | S 58 M, L 179 M | Apache-2.0 | Paddle on Windows + RTX 50 needs special wheels with known issues ([PaddleX install](https://paddlepaddle.github.io/PaddleX/3.3/en/installation/paddlepaddle_install.html)) | mature, printed focus | not exposed |
| **PaddleOCR-VL** 0.9B | document VLM | trained with CROHME and MathWriting ([paper](https://arxiv.org/html/2510.14528v1)), yet only 53.46 single-line formula F1 on OmniHandwritingOCR ([paper](https://arxiv.org/html/2608.18586)) and 51.75 formula CDM (v1.6) on WildHandBench ([paper](https://arxiv.org/html/2608.22959)) | 0.96 B | Apache-2.0 ([HF](https://hf.co/PaddlePaddle/PaddleOCR-VL)) | Paddle or vLLM | mature for documents | LLM logprobs |
| **Other VLMs, zero-shot** | general and document VLMs | OmniHandwritingOCR formula F1 (single / easy multi / hard multi): Kimi-VL-A3B-Instruct 93.33 / 91.99 / 62.46, InternVL3-78B 84.08 / 85.13 / 66.32, GOT-OCR2 78.51 / 56.46 / 39.53, DeepSeek-OCR 66.01 / 79.65 / 61.58, Gemma-3-27B 53.74 / 75.96 / 58.05 ([paper](https://arxiv.org/html/2608.18586)). The single-line set is built from CROHME, HME100K and MLHME, so models trained on those may be inflated. WildHandBench formula CDM: Qwen3-VL (large) 69.06, InternVL3.5 64.00, GLM-OCR 59.01, dots.ocr 53.17, MinerU2.5 36.56, DeepSeek-OCR2 18.33 ([paper](https://arxiv.org/html/2608.22959)) | Kimi-VL is 16.4 B MoE (3 B active) | mostly MIT / Apache | Kimi-VL is not in the Ollama library; GLM-OCR and DeepSeek-OCR are | varies | LLM logprobs |
| **pix2tex / LaTeX-OCR** | ViT | UniMER-Test HWE BLEU **0.012** ([UniMERNet paper, Table 5](https://arxiv.org/html/2404.15254)); printed only, handwriting is an open TODO ([repo](https://github.com/lukas-blecher/LaTeX-OCR)) | small | MIT | easy | stale | temperature only |
| **Pix2Text MFR 1.5** | TrOCR | no handwriting numbers published ([card](https://huggingface.co/breezedeus/pix2text-mfr-1.5)) | small | MIT | ONNX Runtime CPU | maintained | HF generate |
| **CROHME specialists** (CoMER, PosFormer, ICAL, TAMER, SSAN, BTTR, CAN) | small DenseNet + attention decoders | CROHME average 53-61; HME100K 64-70 ([Uni-MuMER paper](https://arxiv.org/html/2505.23566)); PosFormer C14 62.68 ([paper](https://arxiv.org/html/2407.07764v1)) | about 6-7 M | mixed; PosFormer "academic research only" ([repo](https://github.com/SJTU-DeepVisionLab/PosFormer)); several have no licence file | pinned to torch 1.8.1 / CUDA 11.1 / Lightning 1.4.9 ([ICAL](https://github.com/qingzhenduyu/ICAL)), which does not run on sm_120 without porting; CPU would be fine for their size | research code | beam 10 inside |
| **Structural online HMER** ("The Return of Structural HMER", 2025) | strokes, then segmentation, symbol classes and relations | CROHME 2023 expression accuracy 74.14 %; outputs a stroke-label graph, so every symbol maps back to its strokes ([paper](https://arxiv.org/html/2508.19773v1)) | small | paper CC BY 4.0; annotated CROHME+ / MathWriting+ released; code not clearly released | n/a | research | per-symbol |
| **GryphOne** (2026, masked diffusion) | offline | MathWriting test CER 5.51, EM 59.9; C14 65.2 ([paper](https://arxiv.org/html/2602.03370)) | not stated | none found | no weights found | research | n/a |
| **MathWriting baselines** (Google) | online and hybrid | test CER / EM: CTC Transformer 35 M 5.49 / 60 %, PaLI 700 M 5.95 / 64 %, PaLIGemma 3 B 5.97 / 69 % ([paper, Table 6](https://arxiv.org/html/2404.10690)) | 35 M-3 B | n/a | **no weights released** | n/a | n/a |
| **Seshat** | online grammar parser (CROHME 2014 best system trained on competition data) | CROHME 2014 era | small | GPL-3.0 ([repo](https://github.com/falvaro/seshat)) | C++, last push 2020 | stale | parse alternatives |
| **MyScript iink SDK** (native) | commercial online recognizer (Nebo, MyScript Math) | 250 math symbols, matrices, several equations per block ([MyScript Math](https://www.myscript.com/math/)); no public benchmark | n/a | commercial, needs a certificate; pricing is not public ([vendor listing](https://idp-software.com/vendors/myscript/)) | native SDK runs offline and has Windows WPF/UWP examples ([WPF examples](https://github.com/MyScript/interactive-ink-examples-wpf), [offline answer](https://developer-support.myscript.com/support/discussions/topics/16000032257)); the web SDK uses their cloud | product grade | candidates per symbol |
| **Windows Math Recognizer** (`micaut.dll`, Math Input Control) | legacy online recognizer | none published | n/a | ships with Windows | present on this PC (`C:\Program Files\Common Files\microsoft shared\ink\micaut.dll`). It is a COM control with its own window that returns MathML ([IMathInputControl](https://learn.microsoft.com/en-us/windows/win32/api/micaut/nn-micaut-imathinputcontrol)); standalone Math Input Panel was removed in Windows 11 ([summary](https://helpdeskgeek.com/math-input-panel-windows-11-10/)) | 2009-era | alternatives in its UI |

Notes:

- **Why Uni-MuMER tops the list.** It is the only open, permissively licensed model that is trained on MathWriting and has
  published MathWriting numbers. It has GGUF builds in the same Qwen3-VL family already in use, and its best version beats
  the strongest specialists by about 16-19 points on CROHME average (79.74 against 60.9-61.2). Its own ablation shows the 3B vanilla fine-tune (no extra tasks) already
  reaches 68.6 % CROHME average against 38.0 % zero-shot, and the extra tasks (tree-CoT, error-driven learning, symbol
  counting) add about 5 more ([paper](https://arxiv.org/html/2505.23566)).
  It expects its training prompt: `I have an image of a handwritten mathematical expression. Please write out the
  expression of the formula in the image using LaTeX format.` ([dataset rows](https://huggingface.co/datasets/phxember/Uni-MuMER-Data)).
  It was trained with images capped at 262,144 pixels, a 512 x 512 equivalent ([card](https://huggingface.co/phxember/Uni-MuMER-Qwen3-VL-4B)).
  Its output is CROHME-style spaced tokens (`x ^ { 2 }`) and it includes `bmatrix` cells from MathWriting. It is not
  trained to emit `\text{}` or `aligned`, so the current prompt rules for those would move into app logic (section 3).
- **What "2 read as x or infinity" means in these terms.** These are symbol-level confusions between shapes that differ
  mainly in stroke order and closure.
  - The MathWriting authors name this failure: "Two of the main causes of mistakes are confusing similar-looking
    characters like "z" and "2", and errors in the structural arrangement of the characters" ([paper](https://arxiv.org/html/2404.10690)).
  - WildHandBench finds formula errors "are almost entirely prior-driven across all models (87-98%)". The model's LaTeX
    expectations override what it sees, "likely because LaTeX syntax is inherently low-perplexity"
    ([paper](https://arxiv.org/html/2608.22959)). Prompt rules do little against that. Training on handwriting does:
    Uni-MuMER's error-driven-learning task exists "for reducing confusion among visually similar characters"
    ([paper](https://arxiv.org/abs/2505.23566)).
  - Stroke order also separates these shapes well (section 1.3).
- **Latency and VRAM.** No single-image latency on a 5070 Ti-class card is published for any of these. The published
  figures are batch throughput on A800/A40 cards, for example Uni-MuMER's 43.4 frames per second with vLLM batching.
  All of the top three fit in 16 GB together. The 2B and 4B fine-tunes are smaller than the current 8B and would get a
  shorter prompt, but the speed difference has to be measured, not assumed.

### 1.3 Online (strokes) versus offline (image)

- **Stroke order helps a lot when a model is trained for it.** On MathWriting with the same PaLI model, CER was 8.07 %
  from the image alone, 4.64 % from ink tokens and 4.55 % from both. Time and distance encoded in the colour channels of
  the rendered image both helped ([Representing Online Handwriting for Recognition in Large VLMs](https://arxiv.org/html/2402.15307)).
  The MathWriting paper's best baselines are also online or hybrid ([paper](https://arxiv.org/html/2404.10690)).
- **There is no mature open online HMER model with downloadable weights.**
  - Google released the data but not the models. That includes InkFM (2025), an ink foundation model that reports state-of-the-art MathWriting recognition ([paper](https://arxiv.org/abs/2503.23081)); no weights were found.
  - Seshat is GPL and from the CROHME 2014 era.
  - The structural 2025 system has no clear code release.
  - The good online recognizers (MyScript, Windows Ink, Apple) are closed.
- **InkSight** goes the other way: it turns photos of handwriting into digital ink, with Apache-2.0 code and small
  weights ([repo](https://github.com/google-research/inksight), [paper](https://arxiv.org/abs/2402.05804)). Mathboard
  already has the ink, so InkSight is not needed for recognition.
- **Recommendation.** Stay offline for the recognizer now, but use the stroke data you already capture for everything
  around it:
  - grouping, with time as well as space;
  - symbol segmentation and stroke-to-token alignment for tap-to-fix;
  - scratch-out detection;
  - later, as an extra input in a fine-tune. One trick needs no architecture change: render stroke time as colour, as in the paper above. It only pays off after fine-tuning, because an off-the-shelf model has never seen that encoding.

### 1.4 Cheap wins around any recognizer

| Technique | What it fixes | Cost | Evidence |
|---|---|---|---|
| **Swap in a handwriting fine-tune** (Uni-MuMER) with its exact training prompt and training-size renders | symbol confusions, dropped terms | a second backend (llama-server) or a working Ollama import; A/B testing | Qwen2.5-VL-3B went from 37.95 % zero-shot to 79.74 % CROHME average after the fine-tune ([paper](https://arxiv.org/html/2505.23566)) |
| **Token confidence from logprobs** (`logprobs: true, top_logprobs: 5`) | shows which symbol is uncertain; gives alternatives for that position | one request flag; parsing | supported in Ollama v0.12.11+ and llama-server. Caveat: an alternative token only tells you the first token of a divergent path (`\infty` may be several tokens); to show a full alternative, re-decode with the chosen prefix |
| **n-best by beam or sampling** | offers whole-expression alternatives (for example `x^2`, `x_2`, `2^2`) | TexTeller `--num-beams k` is almost free; with a VLM, k samples at different seeds cost k decodes | standard; no HMER-specific gain numbers found |
| **Two-model agreement** (verifier) | auto-commit only when a fine-tune and a second model agree after normalization; otherwise show both as choices | a second model resident (both fit in 16 GB); run both in parallel, so latency is the slower of the two | CE-OCR (Consensus Entropy) builds OCR verification on exactly this: "Correct predictions converge in output space, while errors diverge" across different VLMs ([paper](https://arxiv.org/abs/2504.11101)). Its formula-recognition gain, and a report that resampling one model does worse than greedy decoding, came through a summary and were not re-checked |
| **Constrained decoding** | forbids outputs the board never wants (`\bigcirc`, `\square`, prose) | llama.cpp GBNF grammar ([grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)); Ollama only offers JSON-schema `format`, which does not constrain LaTeX content | a full LaTeX grammar is hard; a whitelist of commands and characters is easy. Today the prompt tries to do this with rules |
| **Validity check and retry** | KaTeX parse failures | already partly done (`toHTML` repairs) | n/a |
| **Few-shot with your own corrected crops** | your personal allographs (your 2, your x) | 2-6 extra images per request. Under Ollama each Qwen-VL image is at least 1024 tokens (see the pixel-budget row), and Ollama's default context is 4k below 24 GiB VRAM ([docs](https://docs.ollama.com/context-length)), so `num_ctx` has to be raised and latency grows | modest and task-dependent: +7 % to +20 % relative CER for a custom 8M-param model trained for in-context use ([paper](https://arxiv.org/html/2603.29450)); under 4 points for general VLMs on a font task ([paper](https://arxiv.org/pdf/2503.23768)). Needs measuring on this task |
| **Per-user fine-tune (LoRA)** on corrections plus MathWriting | the same, permanently, with no prompt cost | a training run on the local GPU; data collection | for handwritten text, fine-tuning on a new writer cut CER by 25 % relative with 16 lines and 50 % with 256 lines ([paper](https://arxiv.org/abs/2302.06308)); Unsloth supports Qwen3-VL fine-tuning from 2B up and ships free Colab notebooks for 8B, which run on 16 GB T4 cards ([docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)) |
| **Control the pixel budget the model actually sees** | blurry, upscaled crops | trivial to check; a server flag with llama-server | Ollama 0.30+ runs GGUF models through upstream llama-server ([PR #16031](https://github.com/ollama/ollama/pull/16031)); 0.30.9 is installed here. For `qwen2vl/qwen25vl/qwen3vl` it passes `--image-min-tokens 1024` ([llama_server.go](https://github.com/ollama/ollama/blob/main/llm/llama_server.go)). One Qwen3-VL token covers 32 x 32 px (16 px patches merged 2 x 2), so every crop is enlarged to at least about 1 megapixel. A 768 x 200 crop (0.15 MP) grows about 7x in area. Whether this hurts has not been tested. For Uni-MuMER, which was trained at up to 256 tokens (262,144 px), set `--image-min-tokens` / `--image-max-tokens` on llama-server to match |
| **Render like the training data** | domain gap | trivial | Uni-MuMER: black ink on white, images capped at 262,144 pixels (512 x 512 equivalent); its MathWriting training renders are about 700-770 px wide in the rows checked ([dataset](https://huggingface.co/datasets/phxember/Uni-MuMER-Data)); TexTeller 448 x 448 grayscale |

---

## 2. Pen UX patterns from mature products

### 2.1 Product by product

| Product | Grouping | Commit timing and preview | Correction | 2D layout |
|---|---|---|---|---|
| **MyScript Nebo / MyScript Math / iink SDK** | In documents, math goes in an explicit math block; in freeform pages you lasso and choose Convert, Math ([Nebo help](https://help.myscript.com/nebo/create-content/math/)) | The web SDK recognizes on each pen-up (`POINTER_UP` trigger; a `QUIET_PERIOD` trigger defaults to 2000 ms) ([iinkJS defaults](https://myscript.github.io/iinkJS/docs/configuration_DefaultConfiguration.js.html)). A typeset preview updates while ink stays ink. Converting is a separate action (double tap in a math block) | scratch-out or strike-through erases a symbol or a whole formula ([gestures](https://developer.myscript.com/doc/interactive-ink/3.0/concepts/editing-gestures/)); overwrite; LaTeX copy from the preview menu | matrices and several equations in one block; 250 symbols |
| **Windows Math Input Panel / Ink Equation** (Word, PowerPoint) | a dedicated writing area | preview pane above the ink updates as you write ("the preview should change as the app recognizes more"); explicit Insert commits ([guide](https://helpdeskgeek.com/math-input-panel-windows-11-10/)); the control has a settable preview height ([API](https://learn.microsoft.com/en-us/windows/win32/api/micaut/nf-micaut-imathinputcontrol-setpreviewheight)) | Select and Correct: tap a misread symbol and pick from a list of alternatives; erase by dragging; advice to finish writing first "because the extra context improves recognition" | full 2D |
| **OneNote Ink to Math** | lasso the equation | Math pane shows the recognized result for confirmation; Ink to Math converts on demand ([support](https://support.microsoft.com/en-us/topic/create-math-equations-using-ink-or-text-with-math-assistant-in-onenote-dc818fad-60e0-432d-8cae-b61f9febf874)) | Fix it mode: lasso a wrong symbol or part of the equation, then pick from alternatives | full 2D |
| **GoodNotes 5/6** | lasso or an ink gesture to circle it. For Math Assist the fix for grouping is spatial: "try moving them so that they are highlighted together" ([support](https://support.goodnotes.com/hc/en-us/articles/10779297639567-Why-isn-t-my-math-being-recognized-by-Math-Assist)) | convert on demand ([support](https://support.goodnotes.com/hc/en-us/articles/7443597657231-Convert-your-Handwritten-Math-to-Rendered-Math)). Math Assist (2024) recognizes in the background: a recognized equation "will glow blue", tapping the glow gives a menu, and "the answer will appear in handwriting" ([support](https://support.goodnotes.com/hc/en-us/articles/10779567357199-How-to-use-Math-Assist)) | wrong lines are underlined; the fix is edit LaTeX | n/a |
| **Apple Math Notes** (iPadOS 18+) | automatic, continuous | Ink never converts. Writing `=` (or a line under a column of numbers) triggers evaluation, and the answer is drawn in synthesized handwriting that matches yours. Three modes: Insert Results, Suggest Results (a Solve button appears), Off ([iPad guide](https://support.apple.com/guide/ipad/solve-math-with-math-notes-ipadeb38d0f8/ipados), [newsroom](https://www.apple.com/newsroom/2024/06/ipados-18-introduces-powerful-intelligence-features-and-apps-for-apple-pencil/)) | **Only ambiguous characters are marked**: "A blue dotted line indicates that there is an ambiguous character in the equation". You tap it and pick the intended character. A red dotted line marks unrecognized or unsolvable math ([support](https://support.apple.com/en-us/120852)). Scratch-out deletes | variables, live recompute when you edit a number |
| **Notability** | Select, circle the ink, Convert, Math; a separate Group command makes ink one element | result is an image you can edit through its LaTeX ([help](https://intercom.help/notability/en-us/articles/16300114-handwriting-and-math-conversion)) | edit LaTeX | fractions, sub/superscripts, vertical pairs, matrices, fences, roots |
| **Samsung Notes** | automatic | writing an expression that ends in `=` makes the answer "appear automatically" ([Samsung](https://www.samsung.com/in/support/mobile-devices/how-to-use-maths-solver-feature-in-notes-on-galaxy-devices/)) | no correction UI documented | n/a |
| **Mathpix digital ink** | the canvas | optional live rendering as you draw so users "check their work as they go" ([digital ink](https://mathpix.com/digital-ink)) | scribble to erase, strike-through to delete | full |
| **MathBrush** (Waterloo, research) | incremental | the best guess is updated after each new stroke; recognition timing was a user preference (after each stroke, after a pause, or on request) ([2006](https://www.scg.uwaterloo.ca/mathbrush/publications/dags2006.pdf)) | select any part, at any time, and choose an alternative from a drop-down ([DAS 2008](https://www.scg.uwaterloo.ca/mathbrush/publications/DAS2008.pdf)). **A choice becomes a lock** that is kept as the user keeps writing: an expression lock fixes a symbol, a semantic lock forces a structure ([MacLean PhD 2014](https://www.scg.uwaterloo.ca/mathbrush/publications/MacLeanPhD2014.pdf)) | full, CAS back end |
| **MathPaper** (Brown/UCF, research) | **automatic range segmentation**: each recognized symbol gets a bounding box inflated "based on the size, shape, and identity of the symbol". Commas and `+ / = < >` "are prevented from becoming isolated ranges". Segmentation re-runs after every stroke, delete or drag, so ranges merge and split on their own ([paper](https://www.eecs.ucf.edu/~jjl/pubs/MathPaper_SG.pdf)) | real-time typeset feedback under the ink. Batch feedback is argued to "introduce a heavyweight cognitive step", while live feedback lets users "amortize the inspection" | a stroke counts as delete only if it crosses other strokes and as a lasso only if it encloses strokes; lasso-drag re-parses live; hover widgets instead of toolbars; red squiggle for syntax errors | full; missing values shown as `??` |
| **MathPad2** (Brown, research) | lasso, then tap. A "recognize all" button failed because line detection was unreliable ([paper](https://www.eecs.ucf.edu/isuelab/publications/pubs/mathpad.pdf)) | explicit | scribble-erase became scribble + tap to avoid false positives; a hover button opens an alternatives menu; users liked scribble-erase | n/a |
| **Hands-On Math** (UIST 2010) | the page is the scope | n/a | pen-only lasso and scribble gestures "conflict with regular inking", so commands use pen plus touch ([report](https://www.microsoft.com/en-us/research/wp-content/uploads/2010/12/bts_brown_handsonmath_finalreport.pdf)) | CAS manipulation |
| **Math Boxes** (UCF, research) | boxes auto-created around scripts and fractions as they are detected | boxes grow as you write into them; characters morph into a cleaned-up ink font as feedback ([paper](https://www.cs.ucf.edu/icerc/isuelab/publications/pubs/iui2015.pdf)) | write into the right box | explicit structure |
| **Detexify** | one symbol at a time | classifies the drawn symbol | returns a ranked list of candidates (k-nearest-neighbour) ([author's notes](https://gist.github.com/kirel/149896)) | single symbols only |
| **Desmos, Microsoft Whiteboard** | no native ink-to-math found. Whiteboard only beautifies ink; old "Desmos handwriting" demos were MyScript feeding Desmos ([post](https://colleenyoung.org/2013/12/21/desmos-wolframalpha-handwriting-recognition/)) | n/a | n/a | n/a |

None of the mature products above replaces ink with typeset automatically on an idle timer. They all keep the ink and
show feedback beside it (Math Input Panel, Mathpix, Apple, GoodNotes Math Assist), or they convert only on an explicit
action (MyScript, OneNote, GoodNotes convert, Notability).

### 2.2 HCI evidence

- **Typeset in place is the slowest feedback style.** LaViola, Leal, Miller and Zeleznik compared Typeset in Place, Adjusted Ink, Large Offset and Small Offset with 24 subjects. They found "subjects took significantly longer to complete the recognition task with Typeset in Place and generally preferred Adjusted Ink or Small Offset" ([GI 2008](https://www.cs.ucf.edu/~jjl/pubs/laviolaGI2008.pdf)). The paper also notes that many people instinctively try to edit the typeset instead of the ink. **Typeset in Place is what Mathboard does now.**
- **Real time or batch should be the user's choice.** Users preferred real-time recognition when writing several expressions (105 of 144 tasks). At high accuracy they had no preference, and "the choice of recognition mode is better left up to the user". Users also changed how they wrote in response to feedback ([Bott, Gabriele, LaViola, SBIM 2011](https://www.cs.ucf.edu/~jjl/pubs/bott2011.pdf)).
- **Users prefer to trigger recognition after they finish, even when that means errors.**
  - They prefer separating diagram from annotation at creation time (the Math pen and Draw pen already do this).
  - They want feedback to "transform and clutter their sketch as little as possible", and they prefer errors that are predictable.
  - A reliable button was the favourite trigger, and a 4 s pause trigger was called too long or disruptive by some users.
  - This was a small study (5 users) on circuit diagrams ([Wais, Wolin, Alvarado, SBIM 2007](http://www.cs.ucf.edu/courses/cap6938/fall2008/penui/readings/userStudy-SBIM07.pdf)).
- **Repair needs two routes: choice and repetition.**
  - Natural repair falls into scratch-out, insertion and overwriting. Showing users how a repair was interpreted raised accuracy on repaired words from 37 % to 65 % ([Huerst et al., CHI 1998](https://isl.iar.kit.edu/downloads/Interactive_Error_Repair_1998_acmchi.pdf)).
  - Correction is either repetition (rewrite) or choice (an n-best list). A choice list always needs an escape to repetition, because the right answer may not be in it ([Mankoff et al., UIST 2000](https://sites.cc.gatech.edu/fce/errata/publications/uist-oops00.pdf)).
- **Handwriting is worth the effort.** It was faster than keyboard equation editing, and the gap grew with equation complexity, measured with recognition taken out of the loop ([Anthony, Yang, Koedinger 2005](http://pact.cs.cmu.edu/pubs/Anthony,%20Yang,%20Koedinger-05.pdf)).
- **The best feedback depends on difficulty.** "The fluidness of the offset method is preferred for simple expressions but as difficulty increases, our math boxes method is overwhelmingly preferred" ([Math Boxes, IUI 2015](https://www.cs.ucf.edu/icerc/isuelab/publications/pubs/iui2015.pdf)).
- **Correction can matter more than raw accuracy.** MathBrush's defining feature is "at any point request alternative interpretations of any portion of their input", rather than one possibly wrong result ([MathBrush](https://www.scg.uwaterloo.ca/mathbrush/publications/DAS2008.pdf)).

### 2.3 Patterns worth copying, in order of value for Mathboard

1. **Split recognizing from committing.**
   - Recognize in the background after idle, as now.
   - Show the result as a small offset preview under the group, and keep the ink.
   - Replace the ink with typeset only on intent. Intent could be a tap on the preview, starting a new expression elsewhere, a page change, or an explicit "convert all".
   - A setting chooses real-time (auto-commit after N seconds) or batch.
   - This alone removes most of the "no freedom while writing" feeling, because nothing moves under the pen.
2. **Mark only the doubtful symbols, and let a tap fix them** (Apple's blue dotted line, Math Input Panel Select and Correct, OneNote Fix it).
   - Use token logprobs to underline only the low-confidence symbols in the preview.
   - Tapping one shows the top alternatives, for example `2 | x | z | \infty`.
   - Always offer the escape to repetition: "write again" and "edit LaTeX".
   - Every pick is saved as a labelled example (see stage 4).
3. **Lock the choices** (MathBrush). A picked alternative survives later re-recognition of the same group. Mathboard re-converts a whole expression when you add to it, so without locks a fixed `2` can flip back to `x`.
4. **Scratch-out to erase** with the Math pen, counted only when the zig-zag crosses existing strokes (MathPaper's rule). This is standard in MyScript, Mathpix, Apple and MathPad2.
5. **Grouping that understands math** (MathPaper).
   - Inflate each symbol's box by what it is. An operator, `=`, a comma or a fraction bar reaches further than a digit, and those never stand alone as their own expression.
   - Re-run grouping after every stroke, delete and move, so groups merge and split by themselves.
   - Use time as well as space. Strokes written in one burst belong together even across larger gaps: the fraction bar written last, the second bar of `=`, the dot of an `i`, an exponent written after the base. A long pause followed by a pen-down far away starts a new group. For scale, Evernote's grouping patent "dries" a burst of wet ink after 1200 ms without a new stroke ([US9007390B2](https://patents.google.com/patent/US9007390B2/en)).
   - A late stroke that overlaps an existing group (a fraction bar, an exponent, a bracket) joins that group whatever the time gap.
   - Keep group outlines visible while writing, like the GoodNotes glow or the MathPad2 box, so grouping mistakes show before recognition.
6. **Lasso regroup** as the explicit override.
   - With Select, draw a loop to make exactly those strokes one expression.
   - Drag a stroke out of a group to split it.
   - Lasso across two groups to merge them.
   - Dragging a lasso selection re-recognizes it.
7. **An optional structured mode for hard input.** A math-block or Math-Boxes-style mode shows boxes for fractions, scripts and matrix cells, for long derivations and matrices.
8. **Keep the existing escape hatches:** edit as LaTeX, re-recognize, keep as ink, undo. These are already in Mathboard.

---

## 3. Recommendation: a staged path

Each stage is a separate, reversible step with its own switch. New behaviour defaults off until it is measured.

### Stage 0: an evaluation set first

- Save each recognition as (PNG, strokes JSON, model output, your final LaTeX). The debug folder already keeps the last 40 images and replies.
- Collect 100-300 of your own expressions, rich in the failure cases (`2`, `x`, `z`, `\infty`, matrices).
- Score normalized exact match plus a KaTeX-render compare, and log p50/p95 latency.
- Everything after this is judged against that set, not against benchmark tables.
- Cost: a small logging and scoring script. No model changes.

### Stage 1: UX that needs no new model

- Preview-then-commit with a small offset preview (pattern 1), plus a real-time or batch setting.
- Token confidence and tap-to-fix alternatives from `logprobs` / `top_logprobs`, with locks on the picks (patterns 2 and 3). This needs one request flag in `server.py` and returns tokens with scores to the client.
- Scratch-out gesture (pattern 4) and lasso regroup, merge and split (pattern 6).
- Math-aware, time-aware grouping with visible group outlines (pattern 5).
- Check the image-token count the server logs for a typical crop, given Ollama's 1024-token floor (section 1.4). Render at the size the model will actually see, or pad small crops rather than letting the server upscale them. A/B this on the stage 0 set.
- Keep using Ollama's native `/api/chat` for logprobs. One report says the OpenAI-compatible endpoint drops those fields ([#16117](https://github.com/ollama/ollama/issues/16117)).
- **Trade-off:** more client code in `app.js`, but no GPU or model risk. It directly targets "little freedom while writing" and "no easy fix". It does not by itself make a `2` read as `2`.

### Stage 2: a better recognizer, A/B against the current one

- Add a backend switch in `server.py`: Ollama (current) or an OpenAI-compatible endpoint such as llama.cpp `llama-server` with `--mmproj`.
- Try `Uni-MuMER-Qwen3-VL-4B` Q8_0 first, then 2B.
  - Use the exact training prompt and render at about the training size (262,144 pixels, 512 x 512 equivalent). On llama-server, pin `--image-min-tokens` / `--image-max-tokens` near 256 so the server does not rescale it.
  - Try importing into Ollama too, but expect the known Qwen3-VL fine-tune import bugs.
  - Note the open llama.cpp precision report. If the fine-tune scores worse under llama.cpp than its card suggests, run it under transformers on torch cu128 as a small Python service.
- Optionally add **TexTeller** as a second opinion with beam n-best. It is small and runs on GPU (torch cu128) or on CPU through ONNX.
- **Agreement rule:** if the two normalized outputs match, commit. If not, show both as choices in the preview.
- **Trade-offs:**
  - a second runtime to install and keep running, and a second model in VRAM;
  - the fine-tune's output style differs (spaced tokens, no `\text`/`aligned`), so post-processing changes;
  - MathWriting-derived weights carry non-commercial data terms, which is fine for personal use.

### Stage 3: the app owns coarse 2D layout

- Detect lines inside a group (a horizontal gap in the stroke projection) and recognize each line alone, then assemble `\begin{aligned}...\end{aligned}`. The HMER training sets are mostly single expressions, so multi-line crops are likely to be a weak spot (check this on the stage 0 set).
- Detect matrices from tall bracket strokes on both sides. Recognize cells by grid clustering, or send the whole crop and check the cell count.
- Map tokens back to strokes for tap-a-symbol. Two ways:
  - approximate: order stroke clusters left to right by bounding box and align them with the token sequence;
  - exact: a stroke-level structural recognizer, which does not exist as open code yet.
- Math-Boxes-style structure boxes as an optional mode.
- **Trade-off:** layout heuristics are fiddly and need the stage 0 set to tune. They also make the recognizer's job smaller, which helps every model.

### Stage 4: personal adaptation (ambitious)

- **Fine-tune.** LoRA on Uni-MuMER-Qwen3-VL-4B (or 2B) using your logged corrections, mixed with MathWriting so it does not forget. Evaluate on a held-out slice of the stage 0 set. Keep the base model as the fallback.
- **Stroke-order input.** A second fine-tune could add stroke-time colour rendering or ink tokens, which is where the online-versus-offline gains in section 1.3 come from.
- **Longer term.** A stroke-level structural recognizer (the 2025 paper's released CROHME+ and MathWriting+ stroke-label data make training one possible) would give exact symbol-to-stroke mapping for corrections.
- **Trade-offs:**
  - a training pipeline and the Windows Unsloth install gotchas;
  - a retrain whenever enough new corrections accumulate;
  - risk of overfitting to one writer, which is the goal for a personal board but worth watching for guest lecturers.

### What not to do

- Do not reach for a bigger general VLM (32B or 72B) to fix symbol confusions. The zero-shot 72B is only 3.8 points better than 7B on CROHME 2014 and worse on HME100K ([paper](https://arxiv.org/html/2505.23566)), and it would not fit alongside anything else.
- Do not adopt pix2tex or the CROHME-only specialists for live handwriting. They are printed-only, or have a roughly 112-token vocabulary on a torch 1.8 stack.
- Do not replace ink with typeset in place as the default commit feedback (section 2.2).

## Could not verify

- Single-image latency on this GPU for any model. This needs a benchmark run, which was out of scope.
- Qwen3-VL-8B zero-shot exact match on CROHME or MathWriting. Only the OmniHandwritingOCR F1 was found.
- Whether Ollama's 1024-token upscaling of small crops hurts accuracy. This is inferred from the code and needs an A/B test.
- The CE-OCR ensemble gain and the "resampling one model is worse than greedy" result.
- Whether `ollama run hf.co/<repo>:Q8_0` works for the Uni-MuMER GGUF repos. It works for the official Qwen3-VL GGUF ([card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)), but fine-tune imports are reported broken.
- TexTeller's real handwriting accuracy (paper withdrawn; possible test contamination).
- Whether onnxruntime-gpu wheels include sm_120 kernels. A community report says they do not, so assume CPU ONNX for small models.
- MyScript's current licence price for hobby use.
