#!/usr/bin/env python3
"""Quick MVP validator for speaker-fingerprint separation.

Takes one recording's pre-extracted embeddings and its user-confirmed ad
blocks. Splits embeddings into "show" vs "ad" windows, builds a centroid
from the show windows, then reports average cosine distance from each set
to the centroid. If the ad windows are much further from the centroid
than the show windows, speaker fingerprinting is a useful signal.

This is a single-recording sanity check. Real per-show centroids will be
built from multiple confirmed episodes (build-show-centroid.py).

Usage:
    analyze-speaker-separation.py <recording.npz> <ads.json>

ads.json format: {"ads": [[start, end], ...], "duration_s": ...}
"""

import argparse
import json
import sys

import numpy as np


def cos_dist(emb_set: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos_sim) per row. Embeddings + centroid must be L2-normed."""
    return 1.0 - emb_set @ centroid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("ads_json")
    ap.add_argument("--require-speech", action="store_true",
                    help="restrict centroid + analysis to has_speech=True windows")
    args = ap.parse_args()

    z = np.load(args.npz)
    times = z["times"]
    emb = z["embeddings"]
    has_speech = z["has_speech"]

    with open(args.ads_json) as f:
        ads_data = json.load(f)
    ad_blocks = ads_data["user"] or ads_data["ads"]
    dur = ads_data["duration_s"]
    if not ad_blocks:
        print("no ad blocks in ground truth — can't separate", file=sys.stderr)
        sys.exit(1)

    is_ad = np.zeros(len(times), dtype=bool)
    for s, e in ad_blocks:
        is_ad |= (times >= s) & (times < e)
    is_show = ~is_ad

    if args.require_speech:
        is_show &= has_speech
        is_ad &= has_speech

    n_show = int(is_show.sum())
    n_ad = int(is_ad.sum())
    print(f"windows: {len(times)} total ({100*has_speech.mean():.0f}% speech)")
    print(f"  show: {n_show} ({100*n_show/len(times):.0f}%)")
    print(f"  ad:   {n_ad} ({100*n_ad/len(times):.0f}%)")

    if n_show < 10:
        print(f"too few show windows for a centroid", file=sys.stderr)
        sys.exit(1)

    centroid = emb[is_show].mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1
    show_dists = cos_dist(emb[is_show], centroid)
    ad_dists = cos_dist(emb[is_ad], centroid) if n_ad else np.array([])

    print(f"\ncentroid (l2-mean of {n_show} show windows):")
    print(f"  show distance:  mean={show_dists.mean():.3f}  median={np.median(show_dists):.3f}  p90={np.percentile(show_dists, 90):.3f}")
    if n_ad:
        print(f"  ad distance:    mean={ad_dists.mean():.3f}  median={np.median(ad_dists):.3f}  p90={np.percentile(ad_dists, 90):.3f}")
        sep = ad_dists.mean() - show_dists.mean()
        print(f"\n  separation:     {sep:+.3f} (positive = ads further from centroid = useful signal)")

        # Naive classifier: threshold halfway between the two means
        thresh = (show_dists.mean() + ad_dists.mean()) / 2
        show_pred_show = (show_dists < thresh).mean()
        ad_pred_ad = (ad_dists >= thresh).mean()
        print(f"  naive @threshold {thresh:.3f}:")
        print(f"    show classified-as-show: {100*show_pred_show:.0f}%")
        print(f"    ad   classified-as-ad:   {100*ad_pred_ad:.0f}%")


if __name__ == "__main__":
    main()
