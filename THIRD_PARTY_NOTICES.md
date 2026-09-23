# Third-party notices

Mathboard vendors two libraries under `static/vendor/` so it runs without a network connection.
Both are MIT-licensed, and each keeps its license file next to the code.

| Library | Version | Files | License |
|---|---|---|---|
| [KaTeX](https://katex.org/) | 0.16.47 | `static/vendor/katex/` (library, CSS, fonts, contrib scripts) | MIT, [`static/vendor/katex/LICENSE`](static/vendor/katex/LICENSE) |
| [three.js](https://threejs.org/) | r186 | `static/vendor/three/` (`three.module.js`, `three.core.js`, `addons/controls/OrbitControls.js`, `addons/renderers/CSS2DRenderer.js`) | MIT, [`static/vendor/three/LICENSE`](static/vendor/three/LICENSE) |

## KaTeX

Copyright (c) 2013-2020 Khan Academy and other contributors.

The KaTeX fonts in `static/vendor/katex/fonts/` come from the KaTeX project
([KaTeX/katex-fonts](https://github.com/KaTeX/katex-fonts)), also under the MIT License,
Copyright (c) 2018 Khan Academy.

## three.js

Copyright (c) 2010-2026 three.js authors.

## Models

Mathboard does not include any model weights. Handwriting recognition runs through a local
[Ollama](https://ollama.com/) install, and the model you pull (by default
`qwen3-vl:8b-instruct`) comes with its own license from its publisher.
