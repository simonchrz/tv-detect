#!/usr/bin/env python3
"""Dedup near-duplicate bumper PNGs per channel via dHash + Hamming
clustering. RTL/ProSieben/VOX accumulate 50+ templates each over time
as the user captures bumpers from new airings; many are pixel-identical
or off-by-a-few-bits (= same bumper, different I-frame). Each template
adds per-frame correlation work in tv-detect, so trimming the set is
direct production-time savings.

Production drain measurement (= 2026-05-04 sample of ~3 detects):
  bumper phase ≈ 17 % of total wall time avg, up to 20 % on
  RTL Let's Dance (= 99 templates). Halving the template count
  via dedup cuts ~8 % of per-detect wall time on bumper-heavy
  channels.

Algorithm:
  1. For each channel + each direction (start/end) separately:
     a. Compute dHash of every PNG (= 9x8 grayscale, diff adjacent
        pixels horizontally → 64-bit hash).
     b. Greedy clustering: walk PNGs in mtime order; keep one only if
        no kept PNG within Hamming distance ≤ HAMMING_THRESHOLD.
     c. Move rejected PNGs to <dir>/dedup-archive/ (NOT delete; safe
        rollback if a kept template turns out to miss a regression).

Usage:
  ./dedup-bumpers.py /mnt/tv/hls/.tvd-bumpers           # all channels, dry-run
  ./dedup-bumpers.py --apply /mnt/tv/hls/.tvd-bumpers   # actually move files
  ./dedup-bumpers.py --apply --slug rtl /path           # one channel only
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from PIL import Image

HAMMING_THRESHOLD = 6  # bits-different ≤ this = same template


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


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_dir(d: Path, threshold: int, apply: bool) -> tuple[int, int]:
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
        try:
            h = dhash(p)
        except Exception as e:
            print(f"    {p.name}: dhash err {e} — keeping conservatively")
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
            kept, dropped = dedup_dir(sub, args.threshold, args.apply)
            total_kept += kept
            total_dropped += dropped
            print(f"    {kept} kept, {dropped} dropped")
    print()
    print(f"=== Total: {total_kept} kept, {total_dropped} dropped "
          f"({100*total_dropped/(total_kept+total_dropped):.0f}%) ===")
    if not args.apply:
        print("(dry-run — re-run with --apply to actually move files)")


if __name__ == "__main__":
    main()
