# `audtheia/analysis/` — Per-detection QC, verification, and longitudinal pattern discovery

| File | Builds in | Role |
|---|---|---|
| `observation.py` | Session 7 | Pi CPU. Deterministic predict→compare→correct QC engine — **not an LLM**. Validates, completes (controlled-vocabulary status), consolidates. Runs `field`-tier (`deterministic_flag`) skills only. Routes structural novelty to `qc_deferred`. |
| `verify.py` | Session 8 | Desktop. RF-DETR re-verification (ONNX Runtime); owns interpretive points (ecological role, rarity, cross-modal attribution), tagged `llm_inferred`; computes authoritative salience; sets the `verified` flag (the occipital gate). |
| `dream.py` | Session 9 | Desktop. Longitudinal NREM→REM pass; unsupervised pattern discovery; writes candidate hypotheses tagged `dream` with effect size + data span. |

The measured-vs-inferred firewall is structural, not a convention: `observation.py` never writes to a desktop-owned table. See `audtheia-v2-decisions-log.md` #24, #30, #45, #47, #51.
