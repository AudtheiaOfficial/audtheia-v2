# Perch 2.0 class labels

`labels.csv` is the list of the 14,795 classes Perch 2.0 scores, in the order of
the model's `label` output: 14,597 species from the iNaturalist 2024 taxonomy and
198 general sound classes from FSD50K. The first row is a header.

It is copied unchanged from the `assets` folder of Google's official Perch 2.0
release on Kaggle (model `google/bird-vocalization-classifier`, variation
`perch_v2_cpu`), which is published under the Apache License 2.0. It ships here
because the official release cannot be downloaded without signing in, while the
model itself can; the model is fetched on request from the ONNX conversion that
Google's model card links to.

SHA-256: `e4d5c0397d8fb08bf90c6b13a34810af53504faad927e472fcc567793c9de057`

Citation: van Merrienboer, Dumoulin, Hamer, Harrell, Burns and Denton, Perch 2.0:
The Bittern Lesson for Bioacoustics, arXiv:2508.04665, 2025.
