# `models/llm/` — Desktop-only generative model

Download any GGUF model here (~3B default at Q4, e.g. Qwen 2.5 3B or Llama 3.2 3B; up to ~7B on a 16GB machine). Runs via **llama.cpp on the desktop only** — the dream pass and `verify.py`'s interpretive analysis.

**The field station runs no LLM/VLM at all.** The Pi-side QC/consolidation step is a deterministic predict→compare→correct engine (decision #51) — there is nothing to download or configure here for field operation.

`.gguf` files are not bundled in the repository due to size — see `.gitignore`. The user downloads their chosen model directly.
