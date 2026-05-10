#!/usr/bin/env python3
"""Shadow-evaluation: would snap-to-boundary-peak improve Block-IoU?

Loads each test-set recording's existing ads.json (= what tv-detect
emitted via the production pipeline), the boundary_head.bin scores
on the same cached embeddings, and the user truth ads_user.json.
For each detected block, snap start/end to the nearest local-max in
the boundary-prob signal within ±SNAP_WINDOW_S, optionally gated by
an ad/show transition check via the existing per-frame classifier
output (= we recompute it from the ad-head feature too).

Reports per-recording Block-IoU before vs after snap, plus an
overall delta. If positive → wire into Go pipeline (Phase 4 proper).
If neutral/negative → iterate on the head or feature design first.

Cross-filter logic: a candidate peak only counts as a transition if
the average per-frame ad/show probability differs across it (= mean
ad-prob in ±3s before vs after the peak shows a sign-flip across 0.5).
This kills intra-ad-spot transitions where the boundary head fires
but neither side is "show".
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


WIN_OFFSETS = (-1, 0, 1)
BACKBONE_DIM = 1280


def is_test_uuid(uuid, frac=0.20):
    h = int(hashlib.sha256(uuid.encode()).hexdigest()[:8], 16)
    return (h % 100) / 100.0 < frac


def load_bndr_head(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"BNDR"
    version, in_dim, hidden = struct.unpack("<III", raw[4:16])
    assert version == 1
    off = 16
    W1 = np.frombuffer(raw[off:off + in_dim * hidden * 4],
                       dtype=np.float32).reshape(in_dim, hidden)
    off += in_dim * hidden * 4
    b1 = np.frombuffer(raw[off:off + hidden * 4], dtype=np.float32)
    off += hidden * 4
    W2 = np.frombuffer(raw[off:off + hidden * 4], dtype=np.float32)
    off += hidden * 4
    b2 = float(np.frombuffer(raw[off:off + 4], dtype=np.float32)[0])
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "in_dim": in_dim, "hidden": hidden}


def boundary_scores(embeds, head):
    n, dim = embeds.shape
    X = np.zeros((n, len(WIN_OFFSETS) * dim), dtype=np.float32)
    for i in range(n):
        for w, off in enumerate(WIN_OFFSETS):
            j = max(0, min(n - 1, i + off))
            X[i, w*dim:(w+1)*dim] = embeds[j]
    H = np.maximum(0, X @ head["W1"] + head["b1"])
    logit = H @ head["W2"] + head["b2"]
    return 1.0 / (1.0 + np.exp(-logit))


def find_peak(scores, target_frame, window):
    """Find the highest-score local maximum within ±window frames of
    target_frame. Returns (peak_frame, peak_score) or (target_frame,
    score_at_target) if no peak above 0.3 found."""
    lo = max(0, target_frame - window)
    hi = min(len(scores) - 1, target_frame + window)
    if hi - lo < 2:
        return target_frame, scores[target_frame]
    sub = scores[lo:hi+1]
    candidates = []
    for i in range(1, len(sub) - 1):
        if sub[i] > 0.3 and sub[i] >= sub[i-1] and sub[i] >= sub[i+1]:
            candidates.append((lo + i, float(sub[i])))
    if not candidates:
        return target_frame, float(scores[target_frame])
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


def transition_check(ad_prob, peak, side, fps, side_window_s=3.0):
    """Return True if the per-frame ad probability has a transition
    across the peak frame in the expected direction.

    side='start' → expect ad-prob LOW before, HIGH after (show→ad)
    side='end'   → expect ad-prob HIGH before, LOW after (ad→show)

    A 'transition' here is a 0.2+ delta in mean ad-prob across the
    peak. Smaller delta → likely intra-ad-spot transition, don't snap."""
    w = int(side_window_s * fps)
    n = len(ad_prob)
    pre_lo = max(0, peak - w)
    pre_hi = max(0, peak)
    post_lo = min(n, peak + 1)
    post_hi = min(n, peak + 1 + w)
    if pre_hi <= pre_lo or post_hi <= post_lo:
        return False
    pre_mean = float(np.mean(ad_prob[pre_lo:pre_hi]))
    post_mean = float(np.mean(ad_prob[post_lo:post_hi]))
    delta = post_mean - pre_mean
    if side == "start":
        return delta > 0.2  # show → ad: ad-prob rises
    else:
        return delta < -0.2  # ad → show: ad-prob falls


def block_iou(pred_blocks, true_blocks, total_s):
    """Per-frame Block-IoU. Same convention as train-head.py."""
    fps_eval = 1.0
    n = max(int(total_s * fps_eval), 1)
    pred_mask = np.zeros(n, dtype=bool)
    true_mask = np.zeros(n, dtype=bool)
    for s, e in pred_blocks:
        a = max(0, int(s * fps_eval))
        b = min(n, int(e * fps_eval))
        if b > a:
            pred_mask[a:b] = True
    for s, e in true_blocks:
        a = max(0, int(s * fps_eval))
        b = min(n, int(e * fps_eval))
        if b > a:
            true_mask[a:b] = True
    inter = int((pred_mask & true_mask).sum())
    union = int((pred_mask | true_mask).sum())
    return inter / union if union > 0 else 1.0


def find_latest_cache(cache_dir, uuid):
    candidates = sorted(cache_dir.glob(f"{uuid}-*.npy"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_user_ads(rec_dir):
    user = rec_dir / "ads_user.json"
    if not user.exists():
        return None
    raw = json.loads(user.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("ads") or []
    return None


def load_auto_ads(rec_dir):
    auto = rec_dir / "ads.json"
    if not auto.exists():
        return None
    try:
        return json.loads(auto.read_text())
    except Exception:
        return None


def synthetic_ad_prob(boundary_scores, blocks, n_frames, fps):
    """Without re-running the ad-classifier we approximate per-frame
    ad-probability by 'inside any ads.json block = 1.0, else = 0.0'.
    This is the auto pipeline's best estimate at the time it wrote
    ads.json. Coarse but adequate for the transition-check gate."""
    p = np.zeros(n_frames, dtype=np.float32)
    for s, e in blocks:
        a = max(0, int(s * fps))
        b = min(n_frames, int(e * fps))
        p[a:b] = 1.0
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls-root", default="/tmp/tv-train-snapshot")
    ap.add_argument("--feature-cache",
                    default="/Users/simon/.cache/tvd-features")
    ap.add_argument("--head", default="/tmp/boundary_head.bin")
    ap.add_argument("--snap-window-s", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--no-cross-filter", action="store_true",
                    help="skip the ad-prob transition check (= snap to "
                         "ANY peak in window, even intra-ad)")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate at most N recordings (0 = all test set)")
    args = ap.parse_args()

    head = load_bndr_head(args.head)
    print(f"loaded BNDR v1 head from {args.head} "
          f"({head['in_dim']}→{head['hidden']}→1)")
    cache_dir = Path(args.feature_cache)
    hls_root = Path(args.hls_root)

    delta_iou = []
    n_evaluated = 0
    n_snapped = 0
    n_skipped_filter = 0
    print()
    print(f"  {'uuid':14}  {'iou_before':>10}  {'iou_after':>10}  "
          f"{'delta':>7}  blocks  snapped/filtered")
    for rec_dir in sorted(hls_root.glob("_rec_*")):
        uuid = rec_dir.name[5:]
        if not is_test_uuid(uuid):
            continue
        true_ads = load_user_ads(rec_dir)
        if not true_ads:
            continue
        auto_ads = load_auto_ads(rec_dir)
        if auto_ads is None:
            continue
        cache_path = find_latest_cache(cache_dir, uuid)
        if not cache_path:
            continue
        feats = np.load(cache_path)
        if feats.shape[0] < 3 or feats.shape[1] < BACKBONE_DIM:
            continue
        embeds = feats[:, :BACKBONE_DIM].astype(np.float32)
        n_frames = embeds.shape[0]
        scores = boundary_scores(embeds, head)
        ad_prob = synthetic_ad_prob(scores, auto_ads, n_frames, args.fps)
        snap_w = int(args.snap_window_s * args.fps)
        new_blocks = []
        snapped_local = 0
        filtered_local = 0
        for s, e in auto_ads:
            sf = int(round(s * args.fps))
            ef = int(round(e * args.fps))
            new_sf, new_ef = sf, ef
            for tag, target, want_side in (("start", sf, "start"),
                                             ("end", ef, "end")):
                pk_frame, pk_score = find_peak(scores, target, snap_w)
                if pk_frame == target:
                    continue
                if not args.no_cross_filter:
                    if not transition_check(ad_prob, pk_frame, want_side,
                                              args.fps):
                        filtered_local += 1
                        continue
                if tag == "start":
                    new_sf = pk_frame
                else:
                    new_ef = pk_frame
                snapped_local += 1
            new_blocks.append([new_sf / args.fps, new_ef / args.fps])
        total_s = n_frames / args.fps
        iou_before = block_iou(auto_ads, true_ads, total_s)
        iou_after = block_iou(new_blocks, true_ads, total_s)
        delta = iou_after - iou_before
        delta_iou.append(delta)
        n_evaluated += 1
        n_snapped += snapped_local
        n_skipped_filter += filtered_local
        print(f"  {uuid[:14]}  {iou_before:>10.3f}  {iou_after:>10.3f}  "
              f"{delta:+7.3f}  {len(auto_ads):2d}     "
              f"{snapped_local}/{filtered_local}")
        if args.limit and n_evaluated >= args.limit:
            break

    if not delta_iou:
        print("\n(no recordings evaluated)")
        return 1
    arr = np.array(delta_iou)
    print(f"\n=== Summary ({n_evaluated} test recordings) ===")
    print(f"  mean ΔIoU:       {arr.mean():+.4f}")
    print(f"  median ΔIoU:     {np.median(arr):+.4f}")
    print(f"  worst ΔIoU:      {arr.min():+.4f}")
    print(f"  best ΔIoU:       {arr.max():+.4f}")
    print(f"  improved:        {int((arr > 0.001).sum())} recs")
    print(f"  unchanged:       {int((np.abs(arr) <= 0.001).sum())} recs")
    print(f"  regressed:       {int((arr < -0.001).sum())} recs")
    print(f"  total snapped:   {n_snapped} boundaries")
    print(f"  filtered out:    {n_skipped_filter} candidate peaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
