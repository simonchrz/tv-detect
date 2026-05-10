#!/usr/bin/env python3
"""Train a boundary-detection head — separate from the ad/show classifier.

The production head.bin answers "is this frame ad or show?" per frame.
Boundaries (= ad-start, ad-end transitions) are derived from runs in
that per-frame stream. /learning's failure-mode analysis says the
dominant failure mode is `missed_bumper` — i.e. the rough block
position is right but the precise start/end frame is off by several
seconds, OR a bumper that signals a transition gets missed entirely.

This trains a SECOND head whose only job is "is this frame an ad
boundary?". Per-frame input is a temporal context window:

  x = concat(embed[t-1s], embed[t], embed[t+1s])   # 3 × 1280 = 3840 dims

The MLP learns to spot the visual derivative around real boundaries.
At inference time the Go side computes a boundary-probability map
across the recording; the block-formation state machine consumes it
either to snap existing boundaries (boundary_drift case) or to open a
new block where it sees a high-confidence peak that's not inside any
existing block (missed_bumper case).

Bootstrap labels: every (start, end) pair in ads_user.json gives two
boundary frames. With ~300 reviewed recordings × ~2 ad blocks each =
~1200 positives. Negatives = sub-sampled "deep" frames (>10s from any
boundary), 5x positive count by default.

This script REQUIRES the feature cache from train-head.py to already
exist for the recordings it processes. Skips any recording where the
cached .npy can't be found — no re-extraction. Run train-head.py first
to populate the cache (the nightly run does this).

Output:
  ~/.../boundary_head.bin       — trained head (BNDR magic + W1+b1+W2+b2)
  ~/.../boundary_head.history.json — train metrics
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


# ── boundary_head.bin file format (BNDR v1) ─────────────────────────
#
# Single-hidden-layer MLP for boundary scoring. Same architecture
# pattern as the ad-head MLP1 v1 but stripped of logo/audio/channel/
# whisper extras — the only input is the temporal-context backbone
# embedding stack.
#
# Header (16 bytes, all little-endian):
#   0..3   magic           "BNDR"
#   4..7   version         uint32 = 1
#   8..11  input_dim       uint32 (= window_size * backbone_dim, e.g. 3840)
#   12..15 hidden_dim      uint32 (e.g. 64)
#
# Weights (float32 LE, in this order):
#   W1: (input_dim, hidden_dim) row-major
#   b1: (hidden_dim,)
#   W2: (hidden_dim, 1) row-major
#   b2: (1,)
#
# Inference: σ(W2·ReLU(W1·x + b1) + b2). x = concat over the 3-frame
# window of backbone embeddings. Window centre is the frame being
# scored; t-1s / t / t+1s pad with zeros at recording start/end.
def write_boundary_head_v1(path, mlp, *, input_dim, hidden_dim):
    if mlp.coefs_[0].shape != (input_dim, hidden_dim):
        raise ValueError(
            f"W1 shape {mlp.coefs_[0].shape} != ({input_dim},{hidden_dim})")
    with open(path, "wb") as f:
        f.write(b"BNDR")
        f.write(struct.pack("<III", 1, input_dim, hidden_dim))
        f.write(mlp.coefs_[0].astype(np.float32).tobytes())
        f.write(mlp.intercepts_[0].astype(np.float32).tobytes())
        f.write(mlp.coefs_[1].astype(np.float32).tobytes())
        f.write(mlp.intercepts_[1].astype(np.float32).tobytes())


WINDOW_OFFSETS = (-1, 0, 1)  # ±1 frame window (= ±1s at fps=1)
BACKBONE_DIM = 1280


def build_temporal_features(embeds):
    """embeds: (n_frames, BACKBONE_DIM). Returns (n_frames, n_window*BACKBONE_DIM).

    For each frame i, concat [embeds[i+off] for off in WINDOW_OFFSETS].
    Out-of-range neighbours pad with the nearest in-range frame (= mirror
    the boundary, NOT zero-pad — keeps statistics consistent for frames
    near the recording start/end)."""
    n = embeds.shape[0]
    n_win = len(WINDOW_OFFSETS)
    out = np.empty((n, n_win * BACKBONE_DIM), dtype=np.float32)
    for i in range(n):
        slots = []
        for off in WINDOW_OFFSETS:
            j = max(0, min(n - 1, i + off))
            slots.append(embeds[j])
        out[i] = np.concatenate(slots)
    return out


def boundary_frames_from_ads(ads, fps):
    """ads: list of [start_s, end_s]. fps: float. Returns sorted list
    of boundary frame indices (= start frames + end frames, one frame
    per boundary)."""
    out = []
    for s, e in ads:
        out.append(int(round(s * fps)))
        out.append(int(round(e * fps)))
    return sorted(set(out))


def labels_with_tolerance(n_frames, boundary_frames, tol):
    """Return uint8 array, 1 if frame is within ±tol of any boundary."""
    y = np.zeros(n_frames, dtype=np.uint8)
    for bf in boundary_frames:
        lo = max(0, bf - tol)
        hi = min(n_frames - 1, bf + tol)
        y[lo:hi+1] = 1
    return y


def find_latest_cache(cache_dir, uuid):
    """Find the most-recently-modified .npy for this uuid (= matches
    current source mtime). Returns Path or None."""
    candidates = sorted(cache_dir.glob(f"{uuid}-*.npy"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    return candidates[0] if candidates else None


def is_test_uuid(uuid, test_frac=0.20):
    """Same uuid-hash split as train-head.py — deterministic across runs."""
    h = int(hashlib.sha256(uuid.encode()).hexdigest()[:8], 16)
    return (h % 100) / 100.0 < test_frac


def load_recording(rec_dir, cache_dir, fps):
    """Returns (uuid, embeds, boundary_frames, n_frames) or None.

    Recordings with `ads_user.json` containing a non-empty `ads` list
    are positive-bearing (boundary frames at start/end of each block).
    Recordings with `ads_user.json` AND empty `ads` are confirmed
    no-boundary content (= user reviewed ARD/ZDF/music-show etc.) and
    return with `boundary_frames=[]` — caller treats them as
    negative-only contributors. Recordings with no ads_user.json or
    no cached features are skipped entirely (cannot infer either way)."""
    uuid = rec_dir.name[len("_rec_"):]
    user_path = rec_dir / "ads_user.json"
    if not user_path.exists():
        return None
    try:
        raw = json.loads(user_path.read_text())
    except Exception:
        return None
    if isinstance(raw, list):
        ads = raw
    elif isinstance(raw, dict):
        ads = raw.get("ads") or []
    else:
        return None
    cache_path = find_latest_cache(cache_dir, uuid)
    if not cache_path:
        return None
    try:
        feats = np.load(cache_path)
    except Exception:
        return None
    if feats.shape[0] < 3 or feats.shape[1] < BACKBONE_DIM:
        return None
    embeds = feats[:, :BACKBONE_DIM].astype(np.float32)
    n_frames = embeds.shape[0]
    bf = boundary_frames_from_ads(ads, fps)
    bf = [b for b in bf if 0 <= b < n_frames]
    return uuid, embeds, bf, n_frames


def subsample_negatives(y_bound, neg_ratio, rng):
    """Returns boolean keep-mask: all positives + neg_ratio × random
    negatives (uniformly chosen from positions that are not within the
    tolerance window of any boundary)."""
    pos_idx = np.flatnonzero(y_bound == 1)
    neg_idx = np.flatnonzero(y_bound == 0)
    n_keep = min(len(neg_idx), len(pos_idx) * neg_ratio)
    if n_keep == 0:
        return None
    chosen_neg = rng.choice(neg_idx, size=n_keep, replace=False)
    keep = np.zeros(len(y_bound), dtype=bool)
    keep[pos_idx] = True
    keep[chosen_neg] = True
    return keep


def per_recording_eval(mlp, per_rec_test, fps, tol):
    """For each test recording: predict per-frame, find local-maxima
    in the predicted boundary-prob signal, match each true boundary
    to the nearest predicted peak within ±tol*5 frames, report:
      - n_true: number of true boundaries
      - n_matched: how many true boundaries had a prediction within window
      - mean_dist: mean |true - matched_pred| in frames (matched only)
      - max_score: max prediction score across the recording (= sanity)
    Returns dict with overall + per-rec breakdown."""
    rows = []
    for r in per_rec_test:
        uuid = r["uuid"]
        X_full = r["X_full"]
        true_b = r["boundary_frames"]
        n_frames = X_full.shape[0]
        # Predict probabilities frame-by-frame
        p = mlp.predict_proba(X_full)[:, 1]
        # Find candidate peaks: local maxima above 0.3, separated by >5 frames
        peaks = []
        for i in range(1, n_frames - 1):
            if p[i] > 0.3 and p[i] >= p[i-1] and p[i] >= p[i+1]:
                if not peaks or i - peaks[-1] > 5:
                    peaks.append(i)
                elif p[i] > p[peaks[-1]]:
                    peaks[-1] = i
        match_window = tol * 5  # ±10 frames default → ±10s
        matched = 0
        dists = []
        for tb in true_b:
            best = None
            for pk in peaks:
                d = abs(pk - tb)
                if d <= match_window and (best is None or d < best):
                    best = d
            if best is not None:
                matched += 1
                dists.append(best)
        rows.append({
            "uuid": uuid[:12],
            "n_true": len(true_b),
            "n_peaks": len(peaks),
            "n_matched": matched,
            "mean_dist": float(np.mean(dists)) if dists else None,
            "max_score": float(p.max()),
        })
    n_true = sum(r["n_true"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    all_dists = []
    for r in rows:
        if r["mean_dist"] is not None:
            all_dists.extend([r["mean_dist"]] * r["n_matched"])
    return {
        "n_recs": len(rows),
        "n_true_boundaries": n_true,
        "n_matched": n_matched,
        "recall": n_matched / n_true if n_true else 0,
        "mean_match_dist_frames": float(np.mean(all_dists)) if all_dists else None,
        "per_rec": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls-root", default="/tmp/tv-train-snapshot",
                    help="dir containing _rec_<uuid>/ads_user.json files "
                         "(= the snapshot mirror that train-head.py uses)")
    ap.add_argument("--feature-cache",
                    default=os.path.expanduser("~/.cache/tvd-features"),
                    help="dir of .npy feature caches (= written by "
                         "train-head.py's feature extractor)")
    ap.add_argument("--fps-extract", type=float, default=1.0,
                    help="must match the fps the cache was extracted at")
    ap.add_argument("--boundary-tol", type=int, default=2,
                    help="positive label window: frames within ±tol of a "
                         "user-confirmed boundary are positives. fps=1 → "
                         "tol=2 means ±2s tolerance.")
    ap.add_argument("--neg-ratio", type=int, default=5,
                    help="subsample negatives to neg_ratio × positives "
                         "per recording. 5 keeps training balanced "
                         "without throwing away too much non-boundary signal.")
    ap.add_argument("--empty-ads-negs", type=int, default=50,
                    help="for recordings with ads_user.json + empty ads "
                         "(= user-confirmed no-boundary content like "
                         "ÖR shows), sample N random frames per recording "
                         "as additional definite-negative training data.")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--output", default=os.path.expanduser(
                    "~/mnt/pi-tv/hls/.tvd-models/boundary_head.bin"))
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing boundary_head.bin (= eval-only run)")
    args = ap.parse_args()

    cache_dir = Path(args.feature_cache)
    if not cache_dir.is_dir():
        sys.exit(f"feature cache not found: {cache_dir}")
    hls_root = Path(args.hls_root)
    if not hls_root.is_dir():
        sys.exit(f"hls root not found: {hls_root}")

    print(f"loading recordings from {hls_root}")
    print(f"feature cache: {cache_dir} ({len(list(cache_dir.glob('*.npy')))} files)")
    print(f"window offsets: {WINDOW_OFFSETS} (= input_dim "
          f"{len(WINDOW_OFFSETS)*BACKBONE_DIM})")
    print(f"boundary tolerance: ±{args.boundary_tol} frames "
          f"(±{args.boundary_tol/args.fps_extract:.1f}s at fps={args.fps_extract})")
    print(f"negative ratio: {args.neg_ratio}× positives per recording")

    rng = np.random.default_rng(42)
    per_rec_train = []
    per_rec_test = []
    per_rec_empty_train = []  # confirmed no-boundary, neg-only
    n_skipped = 0
    n_loaded = 0
    n_empty = 0
    for rec_dir in sorted(hls_root.glob("_rec_*")):
        loaded = load_recording(rec_dir, cache_dir, args.fps_extract)
        if loaded is None:
            n_skipped += 1
            continue
        uuid, embeds, bf, n_frames = loaded
        n_loaded += 1
        X_full = build_temporal_features(embeds)
        y_full = labels_with_tolerance(n_frames, bf, args.boundary_tol)
        rec_data = {
            "uuid": uuid,
            "X_full": X_full,
            "y_full": y_full,
            "boundary_frames": bf,
            "n_frames": n_frames,
        }
        if not bf:
            n_empty += 1
            # Empty-ads recordings only contribute to training (= no
            # boundaries to evaluate against). The test set is held
            # out for boundary-recall measurement, where empty-ads
            # would just add zero true positives.
            if not is_test_uuid(uuid):
                per_rec_empty_train.append(rec_data)
            continue
        if is_test_uuid(uuid):
            per_rec_test.append(rec_data)
        else:
            per_rec_train.append(rec_data)
    print(f"loaded {n_loaded} recordings "
          f"({len(per_rec_train)} train+pos, "
          f"{len(per_rec_empty_train)} train empty-ads neg-only, "
          f"{len(per_rec_test)} test); skipped {n_skipped}")
    if not per_rec_train:
        sys.exit("no train recordings with boundaries — cannot fit")

    # Pool training data with per-recording subsampling so each
    # recording contributes proportionally (= no single long recording
    # with many boundaries dominates).
    X_train_chunks = []
    y_train_chunks = []
    n_pos_total = 0
    n_neg_total = 0
    for r in per_rec_train:
        keep = subsample_negatives(r["y_full"], args.neg_ratio, rng)
        if keep is None or keep.sum() == 0:
            continue
        X_train_chunks.append(r["X_full"][keep])
        y_train_chunks.append(r["y_full"][keep])
        n_pos_total += int((r["y_full"][keep] == 1).sum())
        n_neg_total += int((r["y_full"][keep] == 0).sum())
    # Empty-ads recordings: random N negatives per recording so a long
    # ÖR documentary doesn't dominate. These are guaranteed-negative
    # frames (= user reviewed and confirmed no boundaries exist).
    n_extra_neg = 0
    for r in per_rec_empty_train:
        n = r["n_frames"]
        k = min(args.empty_ads_negs, n)
        if k <= 0:
            continue
        idx = rng.choice(n, size=k, replace=False)
        X_train_chunks.append(r["X_full"][idx])
        y_train_chunks.append(np.zeros(k, dtype=np.uint8))
        n_extra_neg += k
    n_neg_total += n_extra_neg
    X_train = np.vstack(X_train_chunks)
    y_train = np.concatenate(y_train_chunks)
    print(f"\ntrain pool: {X_train.shape[0]} frames "
          f"({n_pos_total} pos, {n_neg_total} neg "
          f"[{n_extra_neg} from empty-ads], "
          f"{n_pos_total/(n_pos_total+n_neg_total)*100:.1f}% pos)")

    print(f"\nfitting MLP({X_train.shape[1]} → {args.hidden_dim} → 1)...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(args.hidden_dim,),
                        activation="relu",
                        solver="adam",
                        max_iter=args.max_iter,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                        verbose=False)
    mlp.fit(X_train, y_train)
    print(f"  fit done in {time.time()-t0:.1f}s, "
          f"{mlp.n_iter_} epochs, train_loss {mlp.loss_:.4f}")

    # In-domain train metrics for sanity (same balanced subset).
    yhat_train = mlp.predict(X_train)
    print(f"\n=== Train (balanced subset) ===")
    print(f"  precision: {precision_score(y_train, yhat_train):.3f}")
    print(f"  recall:    {recall_score(y_train, yhat_train):.3f}")
    print(f"  f1:        {f1_score(y_train, yhat_train):.3f}")

    # The real metric: per-recording boundary recall on the test set.
    # Trained on balanced data but evaluated against the full natural
    # frame distribution (= 99.8% negative) per recording, asking
    # "did the boundary actually get matched by a peak in the prob
    # curve, and how close?"
    print(f"\n=== Test ({len(per_rec_test)} recordings, full per-frame eval) ===")
    eval_res = per_recording_eval(mlp, per_rec_test, args.fps_extract,
                                    args.boundary_tol)
    print(f"  true boundaries:    {eval_res['n_true_boundaries']}")
    print(f"  matched (within ±{args.boundary_tol*5} frames): "
          f"{eval_res['n_matched']}")
    print(f"  recall:             {eval_res['recall']*100:.1f}%")
    if eval_res["mean_match_dist_frames"] is not None:
        print(f"  mean match dist:    "
              f"{eval_res['mean_match_dist_frames']:.2f} frames "
              f"(={eval_res['mean_match_dist_frames']/args.fps_extract:.2f}s)")

    # Worst 5 (= candidates for active-learning surface in Phase 1)
    sorted_recs = sorted(eval_res["per_rec"],
                         key=lambda r: r["n_matched"] / max(1, r["n_true"]))
    print(f"\n  worst 5 recordings by recall:")
    for r in sorted_recs[:5]:
        rate = r["n_matched"] / max(1, r["n_true"])
        print(f"    {r['uuid']}  matched {r['n_matched']}/{r['n_true']} "
              f"({rate*100:.0f}%)  max_score={r['max_score']:.2f}")

    if args.no_write:
        print("\n--no-write set, skipping boundary_head.bin output")
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_boundary_head_v1(out,
                            mlp,
                            input_dim=X_train.shape[1],
                            hidden_dim=args.hidden_dim)
    sz = out.stat().st_size
    print(f"\nwrote {out} ({sz} bytes, BNDR v1)")

    hist_path = out.with_suffix(".history.json")
    hist = []
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
        except Exception:
            hist = []
    hist.append({
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "input_dim": X_train.shape[1],
        "hidden_dim": args.hidden_dim,
        "window_offsets": list(WINDOW_OFFSETS),
        "boundary_tol": args.boundary_tol,
        "neg_ratio": args.neg_ratio,
        "n_train_recs": len(per_rec_train),
        "n_test_recs": len(per_rec_test),
        "n_train_frames": X_train.shape[0],
        "n_train_pos": n_pos_total,
        "test_recall": eval_res["recall"],
        "test_mean_match_dist_frames": eval_res["mean_match_dist_frames"],
    })
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"  history: {hist_path.name} ({len(hist)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
