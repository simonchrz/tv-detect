#!/usr/bin/env python3
"""Per-channel + per-kind perceptual-hash dedup of bumper templates.

Computes dHash (8x8 grayscale gradient to 64-bit hash) per PNG, groups
templates with hamming distance < 5 (= near-identical), keeps the first
per group + MOVES the rest into <slug>/dedup-archive/<kind>/.

Designed to fix the 2026-05-11 bumper-batch fallout: after auto-capture
added 4057 templates without dedup, ProSieben had 1182+1185 templates
which inflated tv-detect's bumper-matching to 60-70 % of pipeline-time
+ caused many '0 blocks' detections from over-snapping.

Run on Pi (= where /mnt/tv/hls/.tvd-bumpers lives, has PIL installed).
Estimated dedup ratio: prosieben ~10-15× (= 1500 to 100-150), kabel-eins
similar, sat-1 maybe 3-5×.
"""
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/mnt/tv/hls/.tvd-bumpers")
HAMMING_THRESHOLD = 5  # <=5 = near-duplicate, group together


def dhash(img_path, size=8):
    """8x8 dHash: gradient between adjacent pixels, returns 64-bit int."""
    try:
        with Image.open(img_path) as img:
            small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
            pixels = list(small.getdata())
        bits = 0
        for row in range(size):
            for col in range(size):
                left = pixels[row * (size + 1) + col]
                right = pixels[row * (size + 1) + col + 1]
                bits = (bits << 1) | (1 if left > right else 0)
        return bits
    except Exception as e:
        print(f"  ERR {img_path.name}: {e}", flush=True)
        return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def dedup_dir(d, archive_root):
    """Cluster files by dHash, keep first per cluster, move rest."""
    files = sorted(d.glob("*.png"))
    if not files:
        return 0, 0
    hashes = []
    for f in files:
        h = dhash(f)
        if h is not None:
            hashes.append((f, h))
    # Greedy clustering: walk files, assign each to first cluster within
    # threshold or open new cluster
    clusters = []  # list of (rep_hash, [files])
    for f, h in hashes:
        placed = False
        for i, (rh, members) in enumerate(clusters):
            if hamming(h, rh) <= HAMMING_THRESHOLD:
                members.append(f)
                placed = True
                break
        if not placed:
            clusters.append((h, [f]))
    # Move all but first per cluster
    archive_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for _, members in clusters:
        for extra in members[1:]:
            try:
                target = archive_root / extra.name
                shutil.move(str(extra), str(target))
                moved += 1
            except Exception as e:
                print(f"  move-err {extra.name}: {e}", flush=True)
    return len(files), len(clusters)


def main():
    if not ROOT.is_dir():
        sys.exit(f"missing {ROOT}")
    print(f"Bumper-dedup with hamming threshold <= {HAMMING_THRESHOLD}\n")
    print(f"{'channel':16}  {'end beforetoafter':>18}  {'start beforetoafter':>20}  saved")
    print("-" * 70)
    total_before = total_after = 0
    for slug_dir in sorted(ROOT.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        archive = slug_dir / "dedup-archive"
        n_e_before = n_e_after = 0
        n_s_before = n_s_after = 0
        end_dir = slug_dir / "end"
        start_dir = slug_dir / "start"
        if end_dir.is_dir():
            n_e_before, n_e_after = dedup_dir(end_dir, archive / "end")
        if start_dir.is_dir():
            n_s_before, n_s_after = dedup_dir(start_dir, archive / "start")
        before = n_e_before + n_s_before
        after = n_e_after + n_s_after
        total_before += before
        total_after += after
        if before == 0:
            continue
        print(f"{slug:16}  {n_e_before:>4} to {n_e_after:<11}  "
              f"{n_s_before:>5} to {n_s_after:<12}  "
              f"-{before - after:>4} ({(before-after)/max(1,before)*100:>3.0f}%)")
    print("-" * 70)
    saved = total_before - total_after
    pct = saved / max(1, total_before) * 100
    print(f"  TOTAL: {total_before} to {total_after}  (saved {saved} = {pct:.0f}%)")


if __name__ == "__main__":
    main()
