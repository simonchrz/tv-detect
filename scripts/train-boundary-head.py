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


BACKBONE_DIM = 1280

# Cached feature column layout (= written by train-head.py
# featurize_recording, post-2026-04 letterbox-v2 + audio-v1 era):
#   [0..1280)  backbone embedding
#   [1280]     logo conf  (NaN sentinel for missing/letterbox-broken)
#   [1281]     audio RMS  ([0,1] normalised, present always since audio-v1)
LOGO_COL = 1280
AUDIO_COL = 1281

SUMMARY_HALF_S = 10  # ±10s for mean(embed) summary feature
AUDIO_VAR_HALF_S = 5  # ±5s for audio RMS variance feature


def load_main_nn(head_path, channel_map_path):
    """Load the production main NN classifier (MLP1 v1 OR MLP2 v2) so
    we can compute per-frame ad-prob as a feature for the boundary head.
    Returns dict with weights + chan_map. None on any error (= caller
    falls back to omitting the feature)."""
    try:
        raw = Path(head_path).read_bytes()
        magic = raw[:4]
        if magic == b"MLP1":
            header_len = 36
            ver, in_dim, hid, out_dim, bb_dim, n_logo, n_audio, n_chan = \
                struct.unpack("<IIIIIIII", raw[4:header_len])
            n_whisper = 0
        elif magic == b"MLP2":
            header_len = 40
            ver, in_dim, hid, out_dim, bb_dim, n_logo, n_audio, n_chan, n_whisper = \
                struct.unpack("<IIIIIIIII", raw[4:header_len])
        else:
            return None
        off = header_len
        W1 = np.frombuffer(raw[off:off+in_dim*hid*4],
                            dtype=np.float32).reshape(in_dim, hid)
        off += in_dim * hid * 4
        b1 = np.frombuffer(raw[off:off+hid*4], dtype=np.float32)
        off += hid * 4
        W2 = np.frombuffer(raw[off:off+hid*out_dim*4],
                            dtype=np.float32).reshape(hid, out_dim)
        off += hid * out_dim * 4
        b2 = np.frombuffer(raw[off:off+out_dim*4], dtype=np.float32)
        cm = json.loads(Path(channel_map_path).read_text())
        return {
            "W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "in_dim": in_dim, "hid": hid,
            "n_logo": n_logo, "n_audio": n_audio,
            "n_chan": n_chan, "n_whisper": n_whisper,
            "slug_to_idx": {s: i for i, s in enumerate(cm.get("slugs", []))},
        }
    except Exception as e:
        print(f"  load_main_nn err: {e}")
        return None


def main_nn_ad_probs(main_nn, embeds, logo_per_frame, audio_per_frame, slug):
    """Run the main NN forward pass to get per-frame ad-prob. embeds is
    (n, 1280). logo and audio are (n,) per-frame. slug → channel one-hot
    via main_nn.slug_to_idx (-1 = unknown → all-zero one-hot).
    Whisper slot (if MLP2) gets neutral 0.5 since we don't have whisper
    data at boundary-head training time."""
    n = embeds.shape[0]
    in_dim = main_nn["in_dim"]
    n_logo = main_nn["n_logo"]
    n_audio = main_nn["n_audio"]
    n_chan = main_nn["n_chan"]
    n_whisper = main_nn["n_whisper"]
    slug_idx = main_nn["slug_to_idx"].get(slug, -1)
    X = np.zeros((n, in_dim), dtype=np.float32)
    X[:, :BACKBONE_DIM] = embeds
    off = BACKBONE_DIM
    if n_logo:
        # Substitute 0.5 for NaN sentinels (= same as inference path).
        l = np.where(np.isnan(logo_per_frame), 0.5, logo_per_frame)
        X[:, off] = l
        off += 1
    if n_audio:
        X[:, off] = audio_per_frame
        off += 1
    if n_chan and slug_idx >= 0:
        X[:, off + slug_idx] = 1.0
    off += n_chan
    if n_whisper:
        X[:, off] = 0.5
    H = np.maximum(0, X @ main_nn["W1"] + main_nn["b1"])
    logits = H @ main_nn["W2"] + main_nn["b2"]
    return 1.0 / (1.0 + np.exp(-logits[:, 0]))


def fetch_uuid_to_slug():
    """Pull uuid → channel_slug mapping from gateway. Cached locally
    so we don't hit the gateway every run."""
    cache = Path.home() / ".cache" / "tvd-boundary-uuid-slug.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    out = {}
    try:
        import urllib.request
        with urllib.request.urlopen(
                "http://raspberrypi5lan:8080/api/internal/training-snapshot",
                timeout=15) as r:
            data = json.loads(r.read())
        for rec in data.get("recordings", []):
            uuid = rec.get("uuid")
            slug = rec.get("channel_slug") or ""
            if uuid:
                out[uuid] = slug
        cache.write_text(json.dumps(out))
    except Exception as e:
        print(f"  uuid→slug fetch err: {e}")
    return out

# Window half-width controls temporal context. Default 1 = (-1, 0, +1)
# = 3-frame window = ±1s at fps=1. With half=3 = (-3..-1, 0, +1..+3) =
# 7-frame window = ±3s — wider context lets the head distinguish
# ad/show boundaries from intra-ad spot transitions (= the latter look
# like local "visual change" within ±1s but show consistent ad-like
# content within ±3s).
def make_window_offsets(half):
    return tuple(range(-half, half + 1))


def _windowed_concat(arr, window_offsets):
    """arr: (n,) or (n, d). Returns (n, |W|) or (n, |W|*d) — each
    frame i concat'd with neighbours at offsets, mirror-padded."""
    n = arr.shape[0]
    n_win = len(window_offsets)
    if arr.ndim == 1:
        out = np.empty((n, n_win), dtype=np.float32)
        for i in range(n):
            for w, off in enumerate(window_offsets):
                out[i, w] = arr[max(0, min(n-1, i+off))]
    else:
        d = arr.shape[1]
        out = np.empty((n, n_win * d), dtype=np.float32)
        for i in range(n):
            for w, off in enumerate(window_offsets):
                j = max(0, min(n-1, i+off))
                out[i, w*d:(w+1)*d] = arr[j]
    return out


def _windowed_mean(arr, half):
    """Centered mean over [i-half..i+half]. Mirror-padded at edges."""
    n = arr.shape[0]
    if arr.ndim == 1:
        out = np.empty(n, dtype=np.float32)
    else:
        out = np.empty((n, arr.shape[1]), dtype=np.float32)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def _windowed_var(arr, half):
    """Centered variance over [i-half..i+half] for 1D arr."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.var(arr[lo:hi]))
    return out


def build_temporal_features(embeds, window_offsets,
                              *, main_nn_probs=None,
                              audio_per_frame=None,
                              logo_per_frame=None,
                              channel_onehot=None,
                              with_summary=False,
                              audio_var_half=AUDIO_VAR_HALF_S):
    """Composite per-frame feature builder.

    Always-on:
      - backbone window: |W| × 1280 (= 8960 at half=3, 3840 at half=1)

    Optional (each adds columns when its source array is provided):
      - main_nn_probs (n,): adds |W| columns (windowed ad-prob)
      - audio_per_frame (n,): adds |W| columns + 1 variance summary
      - logo_per_frame (n,): adds 1 column at center frame only
      - channel_onehot (k,): repeated k columns per frame (constant
        across frames within a recording — but the trained head can
        weight it per-channel)
      - with_summary=True: adds 1280 columns (mean(embed) ±10s)
    """
    n = embeds.shape[0]
    chunks = [_windowed_concat(embeds, window_offsets)]
    if main_nn_probs is not None:
        chunks.append(_windowed_concat(main_nn_probs, window_offsets))
    if audio_per_frame is not None:
        chunks.append(_windowed_concat(audio_per_frame, window_offsets))
        chunks.append(_windowed_var(audio_per_frame, audio_var_half)
                      .reshape(-1, 1))
    if logo_per_frame is not None:
        # Center-only: NaN → 0.5 substitute.
        center = np.where(np.isnan(logo_per_frame), 0.5, logo_per_frame)
        chunks.append(center.reshape(-1, 1))
    if with_summary:
        chunks.append(_windowed_mean(embeds, SUMMARY_HALF_S))
    if channel_onehot is not None:
        # Constant per-frame block (= same one-hot all rows). Tile.
        k = len(channel_onehot)
        block = np.tile(channel_onehot.astype(np.float32), (n, 1))
        chunks.append(block)
    return np.concatenate(chunks, axis=1)


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
    """Returns (uuid, embeds, logo_per_frame, audio_per_frame, ads,
    boundary_frames, n_frames) or None. Now also returns the per-frame
    logo + audio columns from the cached features (needed for both
    the main-NN forward pass and the boundary-head extras).

    Recordings with `ads_user.json` containing a non-empty `ads` list
    are positive-bearing. Empty-ads recordings return with bf=[] and
    are used as negative-only contributors. Skips recordings without
    ads_user.json or matching cached features."""
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
    logo_per_frame = (feats[:, LOGO_COL].astype(np.float32)
                      if feats.shape[1] > LOGO_COL else None)
    audio_per_frame = (feats[:, AUDIO_COL].astype(np.float32)
                       if feats.shape[1] > AUDIO_COL else None)
    bf = boundary_frames_from_ads(ads, fps)
    bf = [b for b in bf if 0 <= b < n_frames]
    return uuid, embeds, logo_per_frame, audio_per_frame, ads, bf, n_frames


def subsample_negatives(y_bound, neg_ratio, rng,
                          *, intra_ad_mask=None, intra_ratio=0.0):
    """Returns boolean keep-mask: all positives + neg_ratio × negatives.

    With intra_ratio>0, that fraction of negatives is preferentially
    drawn from `intra_ad_mask==True` positions (= frames inside an ad
    block but NOT within ±tol of a boundary). Teaches the head that
    visual changes inside an ad are NOT block boundaries — direct
    fix for the missed_bumper false-positive flood."""
    pos_idx = np.flatnonzero(y_bound == 1)
    neg_idx = np.flatnonzero(y_bound == 0)
    n_total_neg = min(len(neg_idx), len(pos_idx) * neg_ratio)
    if n_total_neg == 0:
        return None
    if intra_ad_mask is not None and intra_ratio > 0:
        intra_idx = np.flatnonzero(intra_ad_mask & (y_bound == 0))
        n_intra = min(len(intra_idx), int(n_total_neg * intra_ratio))
        n_random = n_total_neg - n_intra
        chosen_intra = rng.choice(intra_idx, size=n_intra, replace=False) \
            if n_intra > 0 else np.array([], dtype=int)
        # Random pool: negatives not already in chosen_intra
        chosen_set = set(chosen_intra.tolist())
        random_pool = np.array([i for i in neg_idx if i not in chosen_set])
        if len(random_pool) < n_random:
            n_random = len(random_pool)
        chosen_random = rng.choice(random_pool, size=n_random, replace=False) \
            if n_random > 0 else np.array([], dtype=int)
        chosen_neg = np.concatenate([chosen_intra, chosen_random])
    else:
        chosen_neg = rng.choice(neg_idx, size=n_total_neg, replace=False)
    keep = np.zeros(len(y_bound), dtype=bool)
    keep[pos_idx] = True
    keep[chosen_neg] = True
    return keep


def intra_ad_mask_from_ads(ads, n_frames, fps, boundary_tol):
    """Mask: True for frames INSIDE any ad block but OUTSIDE the
    boundary tolerance window. These are the "intra-ad spot
    transition" candidates for explicit-negative sampling."""
    inside = np.zeros(n_frames, dtype=bool)
    near_boundary = np.zeros(n_frames, dtype=bool)
    for s, e in ads:
        sf = max(0, int(round(s * fps)))
        ef = min(n_frames - 1, int(round(e * fps)))
        if ef > sf:
            inside[sf:ef+1] = True
        for bf_pos in (sf, ef):
            lo = max(0, bf_pos - boundary_tol)
            hi = min(n_frames - 1, bf_pos + boundary_tol)
            near_boundary[lo:hi+1] = True
    return inside & ~near_boundary


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
    ap.add_argument("--window-half", type=int, default=1,
                    help="temporal context half-width in frames. "
                         "Default 1 → ±1s window (3 frames concat = "
                         "3840 input dim). 3 → ±3s window (7 frames "
                         "concat = 8960 input dim) — wider context "
                         "helps distinguish ad/show transitions from "
                         "intra-ad spot transitions, at ~2× train + "
                         "inference cost.")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=200)
    # Optimisation flags (= the 5 paths from the boundary-head review):
    ap.add_argument("--with-main-nn", action="store_true",
                    help="#4: per-frame main-NN ad-prob added as feature "
                         "column (windowed). Direct fix for intra-ad-vs-"
                         "boundary discrimination — main NN tells the "
                         "head 'this region is ad-like' so a peak in a "
                         "uniform-ad region won't fire.")
    ap.add_argument("--with-summary", action="store_true",
                    help="#6: mean(embed) ±10s as 1280 summary cols. "
                         "Gives the head 'broader content type' info.")
    ap.add_argument("--with-channel", action="store_true",
                    help="#5: channel one-hot per frame (constant within "
                         "recording but lets the head learn per-channel "
                         "bumper styles).")
    ap.add_argument("--with-audio", action="store_true",
                    help="#7: audio RMS windowed cols + variance summary. "
                         "Boundaries often have brief audio transitions.")
    ap.add_argument("--with-logo-col", action="store_true",
                    help="append a center-frame logo column (+1 dim). OFF by "
                         "default: the Go BNDR loader reads exactly window*1280 "
                         "dims and has no logo input, so enabling this without a "
                         "matching Go change produces an unloadable head "
                         "(input_dim 3841 != 3840).")
    ap.add_argument("--intra-ad-negs-ratio", type=float, default=0.0,
                    help="#1: bias negative sampling toward in-ad frames. "
                         "0.0 = random sampling (default). 0.5 = half of "
                         "negatives drawn from inside ad blocks specifically "
                         "(= explicit intra-ad negatives, teaches the head "
                         "'visual change inside ad ≠ boundary'). Recommend "
                         "0.5-0.7.")
    ap.add_argument("--main-head", default="/tmp/main_head.bin",
                    help="path to the deployed MLP1/MLP2 head.bin")
    ap.add_argument("--main-channel-map", default="/tmp/main_channel_map.json",
                    help="path to the head.channel-map.json sidecar")
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
    window_offsets = make_window_offsets(args.window_half)

    print(f"loading recordings from {hls_root}")
    print(f"feature cache: {cache_dir} ({len(list(cache_dir.glob('*.npy')))} files)")
    print(f"window offsets: {window_offsets} (= input_dim "
          f"{len(window_offsets)*BACKBONE_DIM})")
    print(f"boundary tolerance: ±{args.boundary_tol} frames "
          f"(±{args.boundary_tol/args.fps_extract:.1f}s at fps={args.fps_extract})")
    print(f"negative ratio: {args.neg_ratio}× positives per recording")

    main_nn = None
    uuid_to_slug = {}
    if args.with_main_nn:
        main_nn = load_main_nn(args.main_head, args.main_channel_map)
        if main_nn is None:
            sys.exit(f"--with-main-nn requires {args.main_head} + sidecar")
        print(f"  main NN: in_dim={main_nn['in_dim']} hid={main_nn['hid']} "
              f"n_chan={main_nn['n_chan']}")
    if args.with_main_nn or args.with_channel:
        uuid_to_slug = fetch_uuid_to_slug()
        print(f"  uuid→slug: {len(uuid_to_slug)} entries")

    n_chan = (main_nn["n_chan"] if main_nn
              else (9 if args.with_channel else 0))
    slug_list = (list(main_nn["slug_to_idx"].keys()) if main_nn
                 else ["comedy-central", "kabel-eins", "nick", "prosieben",
                       "rtl", "rtlzwei", "sat-1", "sixx", "vox"])

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
        uuid, embeds, logo_pf, audio_pf, ads, bf, n_frames = loaded
        n_loaded += 1
        slug = uuid_to_slug.get(uuid, "")
        # Compute main NN ad-prob if requested + sources available
        nn_probs = None
        if args.with_main_nn and main_nn is not None and \
                logo_pf is not None and audio_pf is not None:
            nn_probs = main_nn_ad_probs(
                main_nn, embeds, logo_pf, audio_pf, slug)
        # Channel one-hot vector
        chan_vec = None
        if args.with_channel:
            chan_vec = np.zeros(n_chan, dtype=np.float32)
            if slug in slug_list:
                chan_vec[slug_list.index(slug)] = 1.0
        X_full = build_temporal_features(
            embeds, window_offsets,
            main_nn_probs=nn_probs,
            audio_per_frame=audio_pf if args.with_audio else None,
            # Logo center column is OFF by default: the Go BNDR loader
            # (internal/signals/boundary.go) reads exactly window*1280 dims
            # and has no logo input, so a logo column desyncs the head from
            # the consumer (input_dim 3841 != 3840). Kept behind a flag for a
            # future coordinated both-sides extension. Logo is already the
            # primary block-formation signal downstream, so dropping it here
            # costs the boundary head little.
            logo_per_frame=(logo_pf if args.with_logo_col else None),
            channel_onehot=chan_vec,
            with_summary=args.with_summary,
        )
        y_full = labels_with_tolerance(n_frames, bf, args.boundary_tol)
        intra_mask = (intra_ad_mask_from_ads(ads, n_frames, args.fps_extract,
                                               args.boundary_tol)
                      if (bf and args.intra_ad_negs_ratio > 0) else None)
        rec_data = {
            "uuid": uuid,
            "X_full": X_full,
            "y_full": y_full,
            "boundary_frames": bf,
            "n_frames": n_frames,
            "intra_mask": intra_mask,
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
        keep = subsample_negatives(
            r["y_full"], args.neg_ratio, rng,
            intra_ad_mask=r["intra_mask"],
            intra_ratio=args.intra_ad_negs_ratio,
        )
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
        "window_offsets": list(window_offsets),
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
