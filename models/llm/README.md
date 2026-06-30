# models/llm

The desktop-only generative model. Download any GGUF model into this folder (a
model of roughly three billion parameters is a good default, and up to about
seven billion runs on a 16 GB machine). It runs through llama.cpp on the desktop
hub, where it supports the longitudinal pass and the interpretive analysis.

The field station runs no generative model at all. Its quality-control and
consolidation step is a deterministic engine, so there is nothing to download or
configure here for field operation.

Model files are not bundled in the repository because of their size (see
.gitignore). Download your chosen model directly into this folder.
