# models/llm

The desktop-only generative model, and it is optional. Download any GGUF model into this folder (a model of roughly three billion parameters is a good default, and up to about seven billion runs on a 16 GB machine). It runs through llama.cpp on the desktop hub, where it enriches the longitudinal pass with narrated patterns and adds the interpretive analysis on the desktop.

It is optional because the pipeline runs without it: the longitudinal dream pass still discovers and records its candidate patterns with no language model present, and verification still opens the analysis gate. The language model adds richer, human-readable interpretation on top; it never gates the pipeline.

The field station runs no generative model at all. Its quality-control and consolidation step is a deterministic engine, so there is nothing to download or configure here for field operation.

Model files are not committed to the repository because of their size (see .gitignore). Download your chosen model directly into this folder and set its path in the settings file.
