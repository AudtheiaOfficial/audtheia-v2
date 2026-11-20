"""Prepare BioCLIP 2 for desktop verification: export its image encoder and name embeddings.

Path: audtheia/training/bioclip_export.py

BioCLIP 2 (Gu et al., NeurIPS 2025; MIT licence) is a CLIP model trained on 214
million images of organisms. It names the organism in a picture by comparing
the picture's embedding with the embeddings of candidate species names. Its
authors publish it for PyTorch only, so this script runs once, inside the Train
tab's image environment, which already has PyTorch, and writes what the desktop
needs to run it with ONNX Runtime alone:

  bioclip2_visual.onnx   the image encoder, returning unit-length embeddings
  text_embeddings.npy    one unit-length embedding per candidate name
  labels.json            the candidate names, in the order of the embeddings
  bioclip2.json          the preprocessing the model was trained with, read from
                         the model itself, its logit scale, and the check below

Before anything is kept, the exported encoder is run on a test image beside the
PyTorch model and the two embeddings must agree; a failed check writes nothing.
Each name is embedded as "a photo of <scientific name>." The weights are fetched
once from the authors' Hugging Face repository, on this computer.

Run as: python -m audtheia.training.bioclip_export --out DIR --names NAMES.json
[--model hf-hub:imageomics/bioclip-2]. Progress lines start with "PROGRESS".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "hf-hub:imageomics/bioclip-2"
TEMPLATE = "a photo of {}."
PARITY_TOLERANCE = 1e-3
TEXT_BATCH = 256


def say(text: str) -> None:
    print(text, flush=True)


def export(out_dir: Path, names: list, model_id: str = DEFAULT_MODEL, *, pretrained=None) -> dict:
    """Export the encoder and the name embeddings into out_dir. Returns the sidecar."""
    import numpy as np
    import open_clip
    import torch
    import onnxruntime as ort

    torch.set_grad_enabled(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    say("PROGRESS loading the model (the first run downloads it)")
    if model_id.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_id)
    model.eval()

    # The preprocessing the model was trained with, read from the model itself.
    getter = getattr(open_clip, "get_model_preprocess_cfg", None)
    cfg = dict(getter(model) if getter else (getattr(model.visual, "preprocess_cfg", None) or {}))
    size = cfg.get("size", getattr(model.visual, "image_size", 224))
    size = int(size[0] if isinstance(size, (list, tuple)) else size)
    mean = [float(x) for x in cfg.get("mean", open_clip.OPENAI_DATASET_MEAN)]
    std = [float(x) for x in cfg.get("std", open_clip.OPENAI_DATASET_STD)]

    class Encoder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixels):
            feats = self.inner.encode_image(pixels)
            return feats / feats.norm(dim=-1, keepdim=True)

    encoder = Encoder(model).eval()
    onnx_path = out_dir / "bioclip2_visual.onnx.part"
    say("PROGRESS exporting the image encoder")
    dummy = torch.randn(1, 3, size, size)
    try:
        torch.onnx.export(encoder, (dummy,), str(onnx_path), input_names=["pixels"], output_names=["embedding"],
                          dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
                          dynamo=False)
    except Exception as exc:  # noqa: BLE001 - newer PyTorch may offer only the newer exporter
        say(f"PROGRESS the classic exporter was unavailable ({exc}); using the newer one")
        batch = torch.export.Dim("batch", min=1, max=64)
        program = torch.onnx.export(encoder, (dummy,), input_names=["pixels"], output_names=["embedding"],
                                    dynamic_shapes={"pixels": {0: batch}}, opset_version=18, dynamo=True,
                                    external_data=False)
        program.save(str(onnx_path))

    say("PROGRESS checking the exported encoder against PyTorch")
    probe = torch.rand(2, 3, size, size)
    probe = (probe - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    expected = encoder(probe).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["embedding"], {"pixels": probe.numpy().astype(np.float32)})[0]
    diff = float(np.max(np.abs(expected - got)))
    del session
    if diff > PARITY_TOLERANCE:
        onnx_path.unlink(missing_ok=True)
        raise RuntimeError(f"the exported encoder differs from PyTorch by {diff:.2e}; nothing was kept")

    unique = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
    say(f"PROGRESS embedding {len(unique)} species names")
    rows = []
    started = time.time()
    for start in range(0, len(unique), TEXT_BATCH):
        batch = unique[start:start + TEXT_BATCH]
        tokens = tokenizer([TEMPLATE.format(n) for n in batch])
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        rows.append(feats.numpy().astype(np.float32))
        say(f"PROGRESS names {min(start + TEXT_BATCH, len(unique))} of {len(unique)}")
    embeddings = np.concatenate(rows, axis=0) if rows else np.zeros((0, got.shape[1]), dtype=np.float32)

    sidecar = {
        "model": model_id,
        "open_clip_version": getattr(open_clip, "__version__", None),
        "torch_version": torch.__version__,
        "image_size": size,
        "mean": mean,
        "std": std,
        "resize": "shortest side, bicubic, then centre crop",
        "logit_scale": float(model.logit_scale.exp().item()),
        "template": TEMPLATE,
        "names": len(unique),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else int(got.shape[1]),
        "parity_max_abs_diff": diff,
        "seconds_embedding_names": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    np.save(out_dir / "text_embeddings.npy", embeddings)
    (out_dir / "labels.json").write_text(json.dumps(unique, ensure_ascii=False, indent=0), encoding="utf-8")
    (out_dir / "bioclip2.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    onnx_path.replace(out_dir / "bioclip2_visual.onnx")
    say("PROGRESS done")
    return sidecar


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    names = json.loads(Path(args.names).read_text(encoding="utf-8"))
    try:
        sidecar = export(Path(args.out), names, args.model)
    except Exception as exc:  # noqa: BLE001 - reported to the application through the log
        say(f"FAILED {exc}")
        return 1
    say("RESULT " + json.dumps(sidecar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
