#!/usr/bin/env python3
"""Export torchvision MobileNetV2's feature extractor to ONNX.

The backbone is what tv-detect runs per-frame to get a 1280-dim
representation; the trainable head (Linear 1280→1) is shipped
separately so online-learning can update just the head without
re-exporting the heavy ONNX file.

Usage:
    python3 scripts/export-backbone.py --output tvd-backbone.onnx

Default output is /Users/simon/mnt/pi-tv/hls/.tvd-models/backbone.onnx
so both Mac and Pi see it via the SMB share.
"""
import argparse
import os
import sys

import torch
from torchvision import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.expanduser(
        "~/mnt/pi-tv/hls/.tvd-models/backbone.onnx"))
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # MobileNetV2 with ImageNet pretrained weights, classifier replaced
    # with Identity so the forward pass returns the 1280-dim feature
    # vector directly (mirrors the multichannel notebook's frozen-
    # backbone setup).
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier = torch.nn.Identity()
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        out = model(dummy)
    feat_dim = out.shape[-1]
    print(f"sanity: forward pass → {tuple(out.shape)} (feature dim {feat_dim})")

    # Use the legacy (TorchScript-based) exporter — the new dynamo
    # path produces an external-data layout that ORT 1.25 chokes on
    # ("model_path must not be empty"). Single-file output, ~8.5 MB,
    # works in ORT both Python and Go.
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["frame"],
        output_names=["features"],
        dynamic_axes={"frame": {0: "batch"}, "features": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(args.output)
    print(f"wrote {args.output}  ({sz/1024/1024:.1f} MB, "
          f"feat_dim={feat_dim}, opset={args.opset})")


if __name__ == "__main__":
    sys.exit(main())
