#!/usr/bin/env python3
"""Compute per-window speaker confidence (= cosine similarity to show centroid).

Input: embeddings.npz from extract-speaker-embeddings.py + centroid.npz from
build-show-centroid.py.

Output: CSV ready for tv-detect's --speaker-csv flag:
    time_s,speaker_conf,has_speech
    0.000,0.823,1
    1.000,0.875,1
    2.000,0.142,1   <- a window with very different speaker → low conf
    3.000,0.500,0   <- silence/no-speech window → has_speech=0

speaker_conf = (1 + cos_sim) / 2 — re-mapped to [0, 1] where 1 = identical
to centroid, 0 = orthogonal-or-opposite. Re-mapping (rather than raw cosine
distance) keeps the signal in the same [0,1] range as logo/NN confidences,
so blending in tv-detect's blocks.Form is uniform.

Usage:
    compute-speaker-confs.py <embeddings.npz> <centroid.npz> <output.csv>
"""

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings")
    ap.add_argument("centroid")
    ap.add_argument("output")
    args = ap.parse_args()

    z = np.load(args.embeddings)
    times = z["times"]
    emb = z["embeddings"]
    has_speech = z["has_speech"]

    c = np.load(args.centroid)
    centroid = c["centroid"]
    centroid = centroid / (np.linalg.norm(centroid) or 1)

    # Embeddings are already L2-normalised in extract step, so cos_sim = dot.
    cos_sim = emb @ centroid                # range [-1, 1]
    speaker_conf = (1 + cos_sim) / 2        # remap to [0, 1]

    n_speech = int(has_speech.sum())
    print(f"{len(times)} windows, {n_speech} with speech ({100*n_speech/len(times):.0f}%)",
          file=sys.stderr)
    print(f"speaker_conf: mean={speaker_conf.mean():.3f}  "
          f"speech-only mean={speaker_conf[has_speech].mean():.3f}",
          file=sys.stderr)

    with open(args.output, "w") as f:
        f.write("time_s,speaker_conf,has_speech\n")
        for i in range(len(times)):
            f.write(f"{times[i]:.3f},{speaker_conf[i]:.4f},"
                    f"{1 if has_speech[i] else 0}\n")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
