#!/usr/bin/env python3
"""Does the ground truth contradict the raw signal?

Every IoU comparison in this repo assumes the archive labels are right. When
they are not, the measurement inverts: a decoder that finds a real ad break the
labels missed gets PUNISHED for being correct. That is not hypothetical — the
truncated-VOD era left labels that stop partway through a recording, and
dvr-rtl-1785073500 carries 2 ad blocks ending at 1937 s for a 7428 s recording.
Measured against those labels, production reads 0.157.

This check is deliberately DECODER-INDEPENDENT. It does not ask whether
blocks.Form or the HSMM agrees with the labels — both could be wrong the same
way. It asks whether the per-frame NN probability, the input both of them
consume, agrees with the labels:

  HOLE    a stretch where the NN is confidently AD (p > hi) for longer than a
          real break's minimum, and the labels say show. Either the labels
          missed a break, or the NN is confidently wrong for minutes on end.
  PHANTOM the reverse: labels say ad, NN is confidently SHOW throughout.

Neither proves the labels wrong on its own. What they do is separate "this
recording is a hard case" from "this recording should not be in the measurement
set at all", which is a distinction the IoU number alone cannot make.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ARCH = os.path.expanduser("~/.cache/tvd-train-archive")
DUMPS = os.path.expanduser("~/.cache/tv-detect-daemon/emit-signals")


def to_seconds(nn, fps):
    """Per-frame confidences -> per-second means (same reduction the harness
    uses, so the numbers are comparable)."""
    n = int(len(nn) / fps)
    if n <= 0:
        return np.zeros(0)
    idx = (np.arange(len(nn)) / fps).astype(int)
    idx = np.clip(idx, 0, n - 1)
    out = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(out, idx, nn)
    np.add.at(cnt, idx, 1.0)
    return out / np.maximum(cnt, 1.0)


def runs(mask, min_len):
    """Contiguous True runs of at least min_len samples, as (start, end)."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=DUMPS)
    ap.add_argument("--hi", type=float, default=0.80,
                    help="p above this counts as confidently AD")
    ap.add_argument("--lo", type=float, default=0.20,
                    help="p below this counts as confidently SHOW")
    ap.add_argument("--min-run", type=int, default=90,
                    help="seconds; shorter than a real break is not evidence")
    ap.add_argument("--smooth", type=int, default=15,
                    help="seconds of moving average before thresholding")
    args = ap.parse_args()

    print(f"{'uuid':30} {'dauer':>6} {'holes':>6} {'phantom':>8} {'which':>6}"
          f"  Titel")
    flagged = []
    for f in sorted(glob.glob(os.path.join(args.dumps, "dvr-*.json"))):
        u = os.path.basename(f)[:-5]
        npz = os.path.join(ARCH, f"{u}.npz")
        if not os.path.exists(npz):
            continue
        m = json.loads(str(np.load(npz, allow_pickle=True)["meta"]))
        d = json.load(open(f))
        ps = to_seconds(np.array(d["nn_confs"], dtype=np.float64), d["fps"])
        if len(ps) == 0:
            continue
        k = max(1, args.smooth)
        ps = np.convolve(ps, np.ones(k) / k, mode="same")
        n = len(ps)

        gt = np.zeros(n, dtype=bool)
        for a, b in (m.get("ads") or []):
            gt[int(max(0, a)):int(min(n, b))] = True

        hole = sum(b - a for a, b in runs((ps > args.hi) & ~gt, args.min_run))
        phan = sum(b - a for a, b in runs((ps < args.lo) & gt, args.min_run))
        w = m.get("which", "?")
        mark = ""
        # A hole longer than two full breaks means the labels are missing a
        # chunk of the recording, not just an edge.
        if hole > 2 * args.min_run * 2:
            mark = "  <-- Labels uebersehen Werbung"
            flagged.append((u, "hole", hole))
        elif phan > 2 * args.min_run:
            mark = "  <-- Labels behaupten Werbung ohne Signal"
            flagged.append((u, "phantom", phan))
        print(f"{u:30} {n:6d} {hole:6d} {phan:8d} {w:>6}  "
              f"{str(m.get('title'))[:20]}{mark}")

    print(f"\n{len(flagged)} von den geprueften Aufnahmen sind als Messgrundlage"
          f" fragwuerdig:")
    for u, kind, secs in sorted(flagged, key=lambda x: -x[2]):
        print(f"  {u:30} {kind:8s} {secs:5d} s")
    if not flagged:
        print("  (keine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
