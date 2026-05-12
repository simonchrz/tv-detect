#!/usr/bin/env python3
"""Cap each channel+kind to MAX_PER_KIND bumper templates by progressively
increasing hamming threshold until cluster count fits the cap.

Diff vs dedup_bumpers.py: dedup uses fixed threshold 5; cap auto-tunes
per channel to enforce a hard ceiling. Remaining templates are still
visually distinct (= just merged with broader similarity).

Run on Pi (PIL available, /mnt/tv/hls/.tvd-bumpers exists). Re-runnable:
moves excess into <slug>/dedup-archive/<kind>/ same as dedup_bumpers.py.
"""
import shutil
import sys
from pathlib import Path
from PIL import Image

ROOT = Path("/mnt/tv/hls/.tvd-bumpers")
MAX_PER_KIND = 100


def dhash(p, size=8):
    try:
        with Image.open(p) as img:
            small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
            px = list(small.getdata())
        bits = 0
        for r in range(size):
            for c in range(size):
                bits = (bits << 1) | (1 if px[r*(size+1)+c] > px[r*(size+1)+c+1] else 0)
        return bits
    except Exception as e:
        print(f"  ERR {p.name}: {e}", flush=True)
        return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def cluster_at(hashes, threshold):
    """Greedy clustering at a given hamming threshold. Returns list of
    clusters, each [(file, hash), ...]."""
    clusters = []
    for f, h in hashes:
        placed = False
        for cluster in clusters:
            if hamming(h, cluster[0][1]) <= threshold:
                cluster.append((f, h))
                placed = True
                break
        if not placed:
            clusters.append([(f, h)])
    return clusters


def cap_dir(d, archive_root, max_n):
    files = sorted(d.glob("*.png"))
    if len(files) <= max_n:
        return len(files), len(files)
    hashes = [(f, dhash(f)) for f in files]
    hashes = [(f, h) for f, h in hashes if h is not None]
    if len(hashes) <= max_n:
        return len(hashes), len(hashes)
    # Auto-scale threshold: start at 5, increment until cluster count <= max_n
    threshold = 5
    while threshold <= 30:
        clusters = cluster_at(hashes, threshold)
        if len(clusters) <= max_n:
            break
        threshold += 1
    # Move all-but-first per cluster
    archive_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for cluster in clusters:
        for f, _ in cluster[1:]:
            try:
                shutil.move(str(f), str(archive_root / f.name))
                moved += 1
            except Exception as e:
                print(f"  move-err {f.name}: {e}", flush=True)
    print(f"    threshold={threshold} → {len(clusters)} clusters "
          f"(moved {moved})", flush=True)
    return len(files), len(clusters)


def main():
    if not ROOT.is_dir():
        sys.exit(f"missing {ROOT}")
    print(f"Bumper cap: max {MAX_PER_KIND} per kind per channel\n")
    print(f"{'channel':16}  {'end before to after':>20}  {'start before to after':>22}")
    print("-" * 70)
    total_before = total_after = 0
    for slug_dir in sorted(ROOT.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        archive = slug_dir / "dedup-archive"
        end_dir = slug_dir / "end"
        start_dir = slug_dir / "start"
        n_e_before = n_e_after = 0
        n_s_before = n_s_after = 0
        if end_dir.is_dir():
            n_e_before, n_e_after = cap_dir(end_dir, archive / "end", MAX_PER_KIND)
        if start_dir.is_dir():
            n_s_before, n_s_after = cap_dir(start_dir, archive / "start", MAX_PER_KIND)
        before = n_e_before + n_s_before
        after = n_e_after + n_s_after
        total_before += before
        total_after += after
        if before == 0:
            continue
        print(f"{slug:16}  {n_e_before:>5} to {n_e_after:<13}  "
              f"{n_s_before:>5} to {n_s_after:<13}")
    print("-" * 70)
    print(f"  TOTAL: {total_before} to {total_after} "
          f"(saved {total_before - total_after})")


if __name__ == "__main__":
    main()
