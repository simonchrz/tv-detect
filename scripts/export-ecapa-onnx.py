#!/usr/bin/env python3
"""Export speechbrain ECAPA-TDNN to ONNX for CoreML inference.

Replaces the PyTorch CPU pipeline (3× realtime) with onnxruntime +
CoreMLExecutionProvider on Apple Silicon (~5-10× realtime expected).

The exported graph wraps the full encode_batch pipeline:
    audio (B, T) → compute_features → mean_var_norm → embedding_model
                → embedding (B, 1, 192)

Usage:
    export-ecapa-onnx.py <output.onnx>
                          [--window-s 2.0]    # input shape hint for export
                          [--sr 16000]
                          [--validate]        # parity check vs PyTorch
"""

import argparse
import sys

import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier


class ECAPAEncodeWrap(nn.Module):
    """Wraps the full encode_batch pipeline as a single export-able graph.

    SpeechBrain's encode_batch internally splits compute_features +
    mean_var_norm + embedding_model + (optional embedding norm). The
    embedding norm requires running stats from training; we skip it here
    since the downstream cosine similarity is invariant to L2-norm of
    the embedding (and we re-normalise in extract anyway).
    """
    def __init__(self, classifier: EncoderClassifier):
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wavs):
        # wavs: (B, T) float32 in [-1, 1]
        # Pass wav_lens=None throughout — works for fixed-length batches.
        feats = self.compute_features(wavs)
        # mean_var_norm in InputNormalization mode needs lengths;
        # passing all-1 tensor matches "use full length".
        lens = torch.ones(wavs.shape[0], device=wavs.device)
        feats = self.mean_var_norm(feats, lens)
        emb = self.embedding_model(feats, lens)
        return emb  # (B, 1, 192)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32,
                    help="fixed batch size baked into the graph. Caller "
                         "must pad partial batches to this size.")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    print("loading speechbrain ECAPA-TDNN", file=sys.stderr)
    cls = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    cls.eval()
    wrap = ECAPAEncodeWrap(cls).eval()

    win_samples = int(args.sr * args.window_s)
    dummy = torch.randn(args.batch, win_samples)

    print(f"exporting to {args.output} (input shape ({args.batch}, {win_samples}))",
          file=sys.stderr)
    # Use the dynamo exporter (default in PyTorch 2.5+) — handles modern
    # ops better than the legacy torchscript path.
    # NOTE: dynamic_axes intentionally OMITTED — fixed batch dim lets the
    # CoreML MLProgram backend allocate static MPS graphs (otherwise MPS
    # rejects 'unbounded dimension' inputs). Caller must pad the final
    # partial batch to BATCH; extra outputs get discarded.
    torch.onnx.export(
        wrap,
        (dummy,),
        args.output,
        input_names=["wavs"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  wrote {args.output}", file=sys.stderr)

    # Inline external weights into the .onnx file so the model is a single
    # self-contained artifact (easier to deploy + cache). ECAPA is ~80MB,
    # well under the 2GB protobuf limit.
    import onnx
    m = onnx.load(args.output, load_external_data=True)
    onnx.save(m, args.output, save_as_external_data=False)
    print(f"  inlined weights → single-file .onnx", file=sys.stderr)
    import os as _os
    sidecar = args.output + ".data"
    if _os.path.exists(sidecar):
        _os.remove(sidecar)

    if args.validate:
        import onnxruntime as ort
        import numpy as np
        print("validating: PyTorch vs onnxruntime parity", file=sys.stderr)
        providers = []
        avail = ort.get_available_providers()
        if "CoreMLExecutionProvider" in avail:
            providers.append((
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            ))
        providers.append("CPUExecutionProvider")
        sess = ort.InferenceSession(args.output, providers=providers)
        print(f"  ort providers: {sess.get_providers()}", file=sys.stderr)

        with torch.no_grad():
            ref = wrap(dummy).numpy()
        out = sess.run(["embedding"], {"wavs": dummy.numpy()})[0]
        diff = np.abs(ref - out)
        print(f"  shape: torch={ref.shape}  ort={out.shape}", file=sys.stderr)
        print(f"  abs-diff: max={diff.max():.6f}  mean={diff.mean():.6f}", file=sys.stderr)
        # cosine similarity between flattened embeddings
        r = ref.flatten() / (np.linalg.norm(ref) or 1)
        o = out.flatten() / (np.linalg.norm(out) or 1)
        cos = float(r @ o)
        print(f"  cosine similarity: {cos:.6f}", file=sys.stderr)
        if cos < 0.999:
            print(f"  WARNING: cos < 0.999, exported graph drifts from PyTorch",
                  file=sys.stderr)
        else:
            print(f"  ✓ parity OK", file=sys.stderr)


if __name__ == "__main__":
    main()
