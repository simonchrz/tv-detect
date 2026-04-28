#!/usr/bin/env python3
"""Build a per-show speaker centroid from multiple confirmed episodes.

Reads pre-extracted embeddings (.npz from extract-speaker-embeddings.py) for
N recordings of the same show, slices each by user-confirmed ad blocks to
keep only "show" windows, aggregates them into a single L2-normalised
centroid.

Optionally also stores covariance for Mahalanobis distance instead of plain
cosine — Mahalanobis weights down dimensions where the show's speaker pool
varies a lot (e.g. talk shows with rotating guests).

Output: <show-slug>.npz with
    centroid:     (192,) float32 — l2-normalised mean
    cov_inv:      (192, 192) float32 — inverse covariance (optional)
    n_windows:    int — total show windows aggregated
    episodes:     list[str] — UUIDs used
    show_title:   str

Usage:
    build-show-centroid.py <show-slug> <output.npz> \\
        --episode <uuid>:<embeddings.npz>:<ads.json> ...
        [--with-cov]
"""

import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("show_slug")
    ap.add_argument("output")
    ap.add_argument("--episode", action="append", required=True,
                    help="format: uuid:embeddings.npz:ads.json (repeatable)")
    ap.add_argument("--with-cov", action="store_true",
                    help="also store inverse covariance for Mahalanobis distance")
    ap.add_argument("--show-title", default="",
                    help="display name (just stored as metadata)")
    args = ap.parse_args()

    all_show_emb = []
    episodes = []
    for spec in args.episode:
        uuid, npz_path, ads_path = spec.split(":")
        z = np.load(npz_path)
        times = z["times"]
        emb = z["embeddings"]
        has_speech = z["has_speech"]
        with open(ads_path) as f:
            ads_data = json.load(f)
        ad_blocks = ads_data.get("user") or ads_data.get("ads") or []
        is_ad = np.zeros(len(times), dtype=bool)
        for s, e in ad_blocks:
            is_ad |= (times >= s) & (times < e)
        is_show = (~is_ad) & has_speech
        n = int(is_show.sum())
        print(f"  {uuid[:8]}: {n} show windows ({100*n/len(times):.0f}% of {len(times)})",
              file=sys.stderr, flush=True)
        if n == 0:
            continue
        all_show_emb.append(emb[is_show])
        episodes.append(uuid)

    if not all_show_emb:
        print("no show windows aggregated", file=sys.stderr)
        sys.exit(1)

    show_emb = np.concatenate(all_show_emb, axis=0)
    print(f"\ntotal: {len(show_emb)} show windows from {len(episodes)} episodes",
          file=sys.stderr, flush=True)

    centroid = show_emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1

    out_data = {
        "centroid": centroid.astype(np.float32),
        "n_windows": len(show_emb),
        "episodes": np.array(episodes),
        "show_title": np.array(args.show_title or args.show_slug),
    }

    if args.with_cov:
        # Shrinkage cov: regularised toward identity to avoid singular
        # matrices when n_windows < 192.
        cov = np.cov(show_emb, rowvar=False)
        n = len(show_emb)
        shrink = max(0.1, 1.0 - n / 1000.0)  # heuristic: less shrinkage with more data
        cov_reg = (1 - shrink) * cov + shrink * np.eye(192) * np.trace(cov) / 192
        cov_inv = np.linalg.inv(cov_reg).astype(np.float32)
        out_data["cov_inv"] = cov_inv
        print(f"  cov shrinkage={shrink:.2f}, cov_inv stored", file=sys.stderr)

    np.savez_compressed(args.output, **out_data)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
