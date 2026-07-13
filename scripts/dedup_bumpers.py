#!/usr/bin/env python3
"""Dedup near-duplicate bumper PNGs per channel via dHash + Hamming
clustering, AND quarantine near-empty ("sparse") templates. RTL/ProSieben/
VOX accumulate 50+ templates each over time as the user captures bumpers
from new airings; many are pixel-identical or off-by-a-few-bits (= same
bumper, different I-frame). Each template adds per-frame correlation
work in tv-detect, so trimming the set is direct production-time savings.

Production drain measurement (= 2026-05-04 sample of ~3 detects):
  bumper phase ≈ 17 % of total wall time avg, up to 20 % on
  RTL Let's Dance (= 99 templates). Halving the template count
  via dedup cuts ~8 % of per-detect wall time on bumper-heavy
  channels.

Sparse-template quarantine (added 2026-07-13, root-caused via Let's
Dance false positives): tv-detect's bumper match is IoU over a
luma-thresholded binary mask (internal/signals/bumper.go). A template
that's mostly BLACK (a bad capture that landed on a blackout/transition
frame instead of the real bumper card) has very few "white" bits — with
so little structure, near-black VIDEO frames (dark scene transitions,
letterbox bars, dimmed-lighting shots) coincidentally overlap with it at
high IoU purely by chance, not because the bumper is actually on screen.
Confirmed 2026-07-13: RTL's bumper-4eb8af6f-3742-05.png (0.2% white
pixels) matched a Let's Dance intro frame at IoU 0.998, ~1400s before any
real ad break — and the same pathology exists on kabel-eins (15 sparse
templates), ProSieben (11), sat-1 (9). A minimum white-pixel-fraction
gate (checked against a fixed 720x576 reference resample — the FRACTION
is resample-resolution-invariant, so this is correct regardless of
whether a given channel's live decode is native or downscaled) catches
these without touching legitimately sparse-but-real bumpers (the real
template population clusters around 54% white; the false-positive
culprits found so far were all <2%).

Algorithm:
  1. For each channel + each direction (start/end) separately:
     a. Compute dHash of every PNG (= 9x8 grayscale, diff adjacent
        pixels horizontally → 64-bit hash) AND its white-pixel fraction
        (luma > 80 over a 720x576 resample, same threshold tv-detect
        uses at inference).
     b. Sparse check FIRST: white fraction < --sparse-threshold →
        quarantine, skip the dedup comparison entirely (no point
        deduping a template that shouldn't be kept regardless).
     c. Greedy clustering on the survivors: walk PNGs in mtime order;
        keep one only if no kept PNG within Hamming distance ≤
        HAMMING_THRESHOLD.
     d. Move rejected PNGs to <dir>/dedup-archive/ (NOT delete; safe
        rollback if a kept template turns out to miss a regression).

Usage:
  ./dedup-bumpers.py /mnt/tv/hls/.tvd-bumpers           # all channels, dry-run
  ./dedup-bumpers.py --apply /mnt/tv/hls/.tvd-bumpers   # actually move files
  ./dedup-bumpers.py --apply --slug rtl /path            # one channel only
  ./dedup-bumpers.py --sparse-threshold 0 /path          # disable sparse check
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

HAMMING_THRESHOLD = 6      # bits-different ≤ this = same template
SPARSE_THRESHOLD = 0.03    # white-pixel fraction below this = quarantine
REF_W, REF_H = 720, 576    # reference resample for the sparse check
LUMA_TH = 80                # matches internal/signals/bumper.go loadBumperTemplate's default


def dhash(path: Path) -> int:
    """9x8 horizontal-difference dHash → 64-bit int. Robust to small
    pixel shifts; near-identical bumpers hash within 0-3 bits."""
    img = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    px = img.load()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (1 if px[x, y] > px[x + 1, y] else 0)
    return bits


def white_fraction(path: Path) -> float:
    """Fraction of pixels with luma > LUMA_TH, resampled to a fixed
    REF_W x REF_H reference — mirrors tv-detect's own luma-threshold
    binary mask (internal/signals/bumper.go loadBumperTemplate). The
    FRACTION is invariant to the reference resolution chosen (nearest-
    neighbor resample preserves area ratios), so this one fixed size
    is valid regardless of which decode resolution a given channel
    actually runs at in production."""
    img = Image.open(path).convert("RGB").resize((REF_W, REF_H), Image.NEAREST)
    arr = np.asarray(img, dtype=np.int32)
    luma = (77 * arr[:, :, 0] + 150 * arr[:, :, 1] + 29 * arr[:, :, 2]) >> 8
    return float((luma > LUMA_TH).mean())


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_dir(d: Path, threshold: int, sparse_th: float,
             apply: bool) -> tuple[int, int]:
    """Returns (kept, dropped) counts for the directory."""
    pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
    if not pngs:
        return 0, 0
    archive = d / "dedup-archive"
    if apply:
        archive.mkdir(exist_ok=True)
    keepers: list[tuple[Path, int]] = []
    dropped = 0
    for p in pngs:
        if sparse_th > 0:
            try:
                frac = white_fraction(p)
            except Exception as e:
                frac = None
                print(f"    {p.name}: white-fraction err {e} -- "
                      f"skipping sparse check")
            if frac is not None and frac < sparse_th:
                dropped += 1
                arrow = "->" if apply else "would ->"
                print(f"    {p.name}  {arrow} dedup-archive/  "
                      f"(sparse, {100*frac:.2f}% white < "
                      f"{100*sparse_th:.0f}% threshold)")
                if apply:
                    shutil.move(str(p), str(archive / p.name))
                continue
        try:
            h = dhash(p)
        except Exception as e:
            print(f"    {p.name}: dhash err {e} -- keeping conservatively")
            keepers.append((p, -1))
            continue
        # near-duplicate of any keeper?
        dup_of = None
        for kp, kh in keepers:
            if kh < 0:
                continue
            if hamming(h, kh) <= threshold:
                dup_of = kp.name
                break
        if dup_of:
            dropped += 1
            arrow = "->" if apply else "would ->"
            print(f"    {p.name}  {arrow} dedup-archive/  "
                  f"(dup of {dup_of})")
            if apply:
                shutil.move(str(p), str(archive / p.name))
        else:
            keepers.append((p, h))
    return len(keepers), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="bumper root, e.g. /mnt/tv/hls/.tvd-bumpers")
    ap.add_argument("--apply", action="store_true",
                    help="actually move duplicates (default = dry-run)")
    ap.add_argument("--slug", default="",
                    help="restrict to one channel slug")
    ap.add_argument("--threshold", type=int, default=HAMMING_THRESHOLD,
                    help="Hamming distance under which two PNGs are "
                         "considered duplicates (default 6/64 bits)")
    ap.add_argument("--sparse-threshold", type=float, default=SPARSE_THRESHOLD,
                    help="white-pixel fraction (0..1) below which a "
                         "template is quarantined as near-empty/false-"
                         "positive-prone (default 0.03). 0 disables.")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"{args.root} not a directory")
    total_kept = total_dropped = 0
    for chan_dir in sorted(args.root.iterdir()):
        if not chan_dir.is_dir():
            continue
        if args.slug and chan_dir.name != args.slug:
            continue
        print(f"\n=== {chan_dir.name} ===")
        for kind in ("end", "start"):
            sub = chan_dir / kind
            if not sub.is_dir():
                continue
            print(f"  {kind}/:")
            kept, dropped = dedup_dir(sub, args.threshold,
                                      args.sparse_threshold, args.apply)
            total_kept += kept
            total_dropped += dropped
            print(f"    {kept} kept, {dropped} dropped")
    print()
    print(f"=== Total: {total_kept} kept, {total_dropped} dropped "
          f"({100*total_dropped/(total_kept+total_dropped):.0f}%) ===")
    if not args.apply:
        print("(dry-run -- re-run with --apply to actually move files)")


if __name__ == "__main__":
    main()
