#!/usr/bin/env python3
"""Per-channel binary CNN classifier: "logo present in this frame Y/N".

Input: 64x64 RGB crops from extract_logo_dataset.py output.
Architecture: tiny 3-conv CNN (~30K params), trains in <1 min per
channel on M5 Pro. Saves a .pt + ONNX per channel for Go integration.

Trained model targets the logo-region specifically (= what
extract_logo_dataset cropped). Inference at detect-time crops the
same bbox+margin region from each frame and runs this model.

Usage:
  train_logo_cnn.py [--channel <slug>]      # all channels by default
  train_logo_cnn.py --channel vox            # train just one
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

DATA_ROOT = Path.home() / ".cache" / "logo-cnn-data"
OUT_ROOT = Path.home() / ".cache" / "logo-cnn-models"
INPUT_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cpu")


class LogoDataset(Dataset):
    def __init__(self, slug, train, val_frac=0.20, seed=42):
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ])
        # 0 = ad (= no logo), 1 = show (= logo present)
        ad_files   = sorted((DATA_ROOT / slug / "ad").glob("*.png"))
        show_files = sorted((DATA_ROOT / slug / "show").glob("*.png"))
        all_items = [(p, 0) for p in ad_files] + [(p, 1) for p in show_files]
        # Deterministic split by hash of filename
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(all_items), generator=rng)
        n_val = int(len(all_items) * val_frac)
        val_idx = set(idx[:n_val].tolist())
        if train:
            self.items = [it for i, it in enumerate(all_items)
                          if i not in val_idx]
        else:
            self.items = [it for i, it in enumerate(all_items)
                          if i in val_idx]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


class LogoCNN(nn.Module):
    """Tiny 3-conv CNN for binary logo-presence classification."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)  # 16x32x32
        x = F.max_pool2d(F.relu(self.c2(x)), 2)  # 32x16x16
        x = F.max_pool2d(F.relu(self.c3(x)), 2)  # 64x8x8
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)  # logits


def train_channel(slug):
    ad_n = len(list((DATA_ROOT / slug / "ad").glob("*.png")))
    show_n = len(list((DATA_ROOT / slug / "show").glob("*.png")))
    if ad_n < 30 or show_n < 30:
        print(f"  {slug}: SKIP (too few samples: ad={ad_n} show={show_n})")
        return None
    train_ds = LogoDataset(slug, train=True)
    val_ds   = LogoDataset(slug, train=False)
    # Class-weighted loss: rebalance ad-poor channels
    n_ad   = sum(1 for _, l in train_ds.items if l == 0)
    n_show = sum(1 for _, l in train_ds.items if l == 1)
    pos_weight = torch.tensor([n_ad / max(1, n_show)], device=DEVICE)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=2)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=2)
    model = LogoCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_acc = 0.0
    best_state = None
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = (torch.sigmoid(model(x)) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
                tp += ((pred == 1) & (y == 1)).sum().item()
                fp += ((pred == 1) & (y == 0)).sum().item()
                fn += ((pred == 0) & (y == 1)).sum().item()
                tn += ((pred == 0) & (y == 0)).sum().item()
        acc = correct / total if total else 0
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or ep == EPOCHS - 1 or ep % 10 == 9:
            ad_recall   = tn / max(1, tn + fp)  # correctly id'd "no logo"
            show_recall = tp / max(1, tp + fn)  # correctly id'd "logo"
            print(f"  {slug:14s} ep{ep+1:>2d}/{EPOCHS}  acc={acc*100:.1f}%  "
                  f"ad-recall={ad_recall*100:.1f}%  show-recall={show_recall*100:.1f}%")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    pt_path = OUT_ROOT / f"{slug}.logo-cnn.pt"
    onnx_path = OUT_ROOT / f"{slug}.logo-cnn.onnx"
    torch.save(model.state_dict(), pt_path)
    # Export to ONNX for Go integration. Use the LEGACY torch.onnx
    # exporter (= dynamo=False) so weights stay INLINE in the single
    # .onnx file — the new dynamo exporter shards weights to a sibling
    # .onnx.data which complicates Go deployment (= would need to
    # serve+cache two files per channel). Tiny model (~30K params,
    # ~150 KB) doesn't benefit from external storage anyway.
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model.cpu(), dummy, str(onnx_path),
        input_names=["frame"], output_names=["logit"],
        dynamic_axes={"frame": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Clean up the dynamo-exporter's stale external data file if a
    # prior run left one behind.
    data_path = onnx_path.with_suffix(".onnx.data")
    if data_path.exists():
        data_path.unlink()
    print(f"  {slug}: best val-acc {best_acc*100:.2f}% → {onnx_path.name}")
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="single channel slug; default = all")
    args = ap.parse_args()

    print(f"device: {DEVICE}", flush=True)
    if args.channel:
        slugs = [args.channel]
    else:
        slugs = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())
    results = {}
    for slug in slugs:
        print(f"\n=== {slug} ===")
        acc = train_channel(slug)
        if acc is not None:
            results[slug] = acc

    print(f"\n{'channel':14s} {'val-acc':>8s}")
    print("-" * 30)
    for slug, acc in sorted(results.items()):
        print(f"{slug:14s} {acc*100:>7.2f}%")


if __name__ == "__main__":
    main()
