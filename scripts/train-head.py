#!/usr/bin/env python3
"""Train tv-detect's linear head on accumulated user-edited cutlists.

Workflow:
  1. Walk every recording dir under HLS_ROOT for an ads_user.json
     (truth) or ads.json (auto, lower-quality fallback).
  2. For each labelled recording, extract 1 frame/sec via ffmpeg and
     push it through the ONNX backbone to get 1280-dim features.
  3. Pool all features + labels, fit a logistic regression
     (Linear(1280) + sigmoid) — one big closed-form-ish step on
     scikit-learn.
  4. Write head.bin: 1280 weights × float32, then 1 bias × float32 LE.

Inputs:
  --backbone   ONNX backbone (default: ~/mnt/pi-tv/hls/.tvd-models/backbone.onnx)
  --output     head.bin destination (same dir, default head.bin)
  --hls-root   recordings dir (default ~/mnt/pi-tv/hls)
  --feature-cache   cached features dir (default ~/.cache/tvd-features)

The feature cache is keyed by recording uuid + source mtime, so
re-running on the same set is cheap (only new recordings get
re-extracted).
"""
import argparse
import concurrent.futures as cf
import gc
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


# ── MLP head.bin file format (v1) ────────────────────────────────
# A single-hidden-layer MLP (1290→32→1 typical) serialised as a flat
# byte stream that the Go inference path can mmap. Distinct from the
# legacy LogReg head.bin (5128/5132/5148/5152/5156 B size-detected)
# via a 4-byte magic prefix — Go discriminates by magic, not by size,
# so MLP heads can have arbitrary widths without breaking detection.
#
# Layout, all integers little-endian uint32, all weights little-endian
# float32, packed back-to-back:
#
#   bytes  field           value
#   0..3   magic           "MLP1" (= 0x4D 0x4C 0x50 0x31)
#   4..7   version         1
#   8..11  input_dim       D    (e.g. 1290 = 1280 backbone+1 logo+1 audio+8 chan)
#   12..15 hidden_dim      H    (e.g. 32)
#   16..19 output_dim      O    (= 1; sigmoid'd at inference, kept for fwd-compat)
#   20..23 backbone_dim    1280 (sanity check vs. ONNX backbone output)
#   24..27 n_logo          0 or 1  (logo-conf input present?)
#   28..31 n_audio         0 or 1  (audio-RMS input present?)
#   32..35 n_channel       0..N    (size of channel one-hot block; slug→idx
#                                   resolution lives in head.channel-map.json)
#   36..(36+D*H*4)         W1   (D*H float32, ROW-MAJOR: input i, hidden j → idx i*H+j)
#   ...                    b1   (H float32)
#   ...                    W2   (H*O float32, row-major)
#   ...                    b2   (O float32)
#
# Inference (per frame, after building x = [backbone(1280), logo?, audio?, chan_onehot?]):
#   h     = max(0, x @ W1 + b1)         # ReLU activation
#   logit = h @ W2 + b2                 # shape (O,) — for O=1, scalar
#   prob  = 1 / (1 + exp(-logit))       # sigmoid (for O=1)
#
# Total size: 36 + (D*H + H + H*O + O) * 4 bytes. For 1290/32/1: 165 416 B.
#
# Channel-one-hot at inference: load head.channel-map.json sidecar,
# look up the recording's channel slug → idx; one-hot column at idx
# equals 1.0, all others 0.0. Unknown slug → all-zero one-hot
# (model degrades gracefully to "channel-agnostic" prediction).
def write_mlp_head_v1(path, mlp, *, input_dim, hidden_dim,
                      backbone_dim=1280, n_logo=0, n_audio=0,
                      n_channel=0):
    """Serialise a trained sklearn MLPClassifier (single hidden layer,
    ReLU, logistic output) to the v1 MLP head.bin format. The MLP must
    have hidden_layer_sizes=(hidden_dim,) and binary output.

    Writes path atomically (= path.tmp + rename) so a crash mid-write
    can't leave Go consumers reading a partial file."""
    import struct
    if len(mlp.coefs_) != 2 or len(mlp.intercepts_) != 2:
        raise ValueError(f"expected single-hidden-layer MLP; got "
                         f"{len(mlp.coefs_)} coef matrices")
    W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
    W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
    if W1.shape != (input_dim, hidden_dim):
        raise ValueError(f"W1 shape {W1.shape} != ({input_dim}, {hidden_dim})")
    if b1.shape != (hidden_dim,):
        raise ValueError(f"b1 shape {b1.shape} != ({hidden_dim},)")
    output_dim = b2.shape[0]
    if W2.shape != (hidden_dim, output_dim):
        raise ValueError(f"W2 shape {W2.shape} != ({hidden_dim}, {output_dim})")
    if backbone_dim + n_logo + n_audio + n_channel != input_dim:
        raise ValueError(
            f"input_dim {input_dim} != backbone {backbone_dim} + "
            f"logo {n_logo} + audio {n_audio} + chan {n_channel}")
    header = struct.pack("<8I",
                         0x31504C4D,  # "MLP1" little-endian = M(4D)L(4C)P(50)1(31)
                         1, input_dim, hidden_dim, output_dim,
                         backbone_dim, n_logo, n_audio)
    header += struct.pack("<I", n_channel)
    body = (W1.tobytes() + b1.tobytes()
            + W2.tobytes() + b2.tobytes())
    from pathlib import Path as _P
    p = _P(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(header + body)
    tmp.replace(p)
    return p.stat().st_size


# ── MLP head.bin file format (v2 — adds whisper-prob input) ──────
# Identical to v1 except:
#   - magic is "MLP2" (= 0x32504C4D LE) — Go discriminates by magic,
#     so an old MLP1-only binary on a v2 file falls through to legacy
#     LogReg detection which fails clean (= 0.5 fallback) instead of
#     mis-parsing v2 weights against the wrong header layout.
#   - header grows from 36 → 40 bytes: appends a 10th uint32 LE field
#     `n_whisper` (0 or 1) AFTER `n_channel`.
#   - input_dim contract becomes
#         backbone + n_logo + n_audio + n_channel + n_whisper == input_dim
#   - input vector layout at inference becomes
#         [backbone(1280), logo?, audio?, chan_onehot?, whisper_prob?]
#     (= whisper appended LAST so old chan-one-hot indices stay aligned).
#
# Writer below is a standalone function so v1 and v2 stay independent;
# rolling forward to v2 means flipping the head-arch flag without
# touching write_mlp_head_v1, and rolling back is the same flip.
def write_mlp_head_v2(path, mlp, *, input_dim, hidden_dim,
                      backbone_dim=1280, n_logo=0, n_audio=0,
                      n_channel=0, n_whisper=0):
    """Serialise an sklearn MLPClassifier to the v2 MLP head.bin
    format. Same atomic write + shape validation as v1, plus the
    extra whisper input slot. Returns bytes written."""
    import struct
    if len(mlp.coefs_) != 2 or len(mlp.intercepts_) != 2:
        raise ValueError(f"expected single-hidden-layer MLP; got "
                         f"{len(mlp.coefs_)} coef matrices")
    W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
    W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
    if W1.shape != (input_dim, hidden_dim):
        raise ValueError(f"W1 shape {W1.shape} != ({input_dim}, {hidden_dim})")
    if b1.shape != (hidden_dim,):
        raise ValueError(f"b1 shape {b1.shape} != ({hidden_dim},)")
    output_dim = b2.shape[0]
    if W2.shape != (hidden_dim, output_dim):
        raise ValueError(f"W2 shape {W2.shape} != ({hidden_dim}, {output_dim})")
    if backbone_dim + n_logo + n_audio + n_channel + n_whisper != input_dim:
        raise ValueError(
            f"input_dim {input_dim} != backbone {backbone_dim} + "
            f"logo {n_logo} + audio {n_audio} + chan {n_channel} + "
            f"whisper {n_whisper}")
    header = struct.pack("<10I",
                         0x32504C4D,  # "MLP2" little-endian
                         2, input_dim, hidden_dim, output_dim,
                         backbone_dim, n_logo, n_audio, n_channel,
                         n_whisper)
    body = (W1.tobytes() + b1.tobytes()
            + W2.tobytes() + b2.tobytes())
    from pathlib import Path as _P
    p = _P(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(header + body)
    tmp.replace(p)
    return p.stat().st_size


# ── MLP head.bin file format (v3 — adds temporal-delta input) ────
# Identical to v2 except:
#   - magic is "MLP3" (= 0x33504C4D LE) — a v2-only Go binary falls
#     through to legacy LogReg detection on a v3 file (fails clean,
#     0.5 fallback) instead of mis-parsing v3 weights against the v2
#     40-byte header.
#   - header grows from 40 → 44 bytes: appends an 11th uint32 LE
#     field `n_temporal` (0 or 2 — always both-or-neither, the L2
#     distance to the previous AND next frame) AFTER `n_whisper`.
#   - input_dim contract becomes
#         backbone + n_logo + n_audio + n_channel + n_whisper
#         + n_temporal == input_dim
#   - input vector layout at inference becomes
#         [backbone(1280), logo?, audio?, chan_onehot?, whisper_prob?,
#          l2_dist_prev?, l2_dist_next?]
#     (= temporal appended LAST so v1/v2 column order stays a prefix).
#     The L2 distance is computed over [backbone, logo?, audio?] of
#     the adjacent frame (NOT the channel-onehot/whisper columns,
#     which don't vary meaningfully frame-to-frame) — see nn.go
#     confidenceMLP for the matching Go-side computation.
# Migrated to production 2026-07-12 after a 7-night --shadow-eval
# series (6/7 nights positive vs mlp32-channel-whisper).
def write_mlp_head_v3(path, mlp, *, input_dim, hidden_dim,
                      backbone_dim=1280, n_logo=0, n_audio=0,
                      n_channel=0, n_whisper=0, n_temporal=0):
    """Serialise an sklearn-compatible MLP to the v3 MLP head.bin
    format. Same atomic write + shape validation as v2, plus the
    2 temporal-delta input slots. Returns bytes written."""
    import struct
    if len(mlp.coefs_) != 2 or len(mlp.intercepts_) != 2:
        raise ValueError(f"expected single-hidden-layer MLP; got "
                         f"{len(mlp.coefs_)} coef matrices")
    W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
    W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
    if W1.shape != (input_dim, hidden_dim):
        raise ValueError(f"W1 shape {W1.shape} != ({input_dim}, {hidden_dim})")
    if b1.shape != (hidden_dim,):
        raise ValueError(f"b1 shape {b1.shape} != ({hidden_dim},)")
    output_dim = b2.shape[0]
    if W2.shape != (hidden_dim, output_dim):
        raise ValueError(f"W2 shape {W2.shape} != ({hidden_dim}, {output_dim})")
    if (backbone_dim + n_logo + n_audio + n_channel + n_whisper + n_temporal
            != input_dim):
        raise ValueError(
            f"input_dim {input_dim} != backbone {backbone_dim} + "
            f"logo {n_logo} + audio {n_audio} + chan {n_channel} + "
            f"whisper {n_whisper} + temporal {n_temporal}")
    header = struct.pack("<11I",
                         0x33504C4D,  # "MLP3" little-endian
                         3, input_dim, hidden_dim, output_dim,
                         backbone_dim, n_logo, n_audio, n_channel,
                         n_whisper, n_temporal)
    body = (W1.tobytes() + b1.tobytes()
            + W2.tobytes() + b2.tobytes())
    from pathlib import Path as _P
    p = _P(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(header + body)
    tmp.replace(p)
    return p.stat().st_size


# ── MLP head.bin file format (v4 — adds minute-of-hour prior input) ─
# Identical to v3 except:
#   - magic is "MLP4" (= 0x34504C4D LE) — a v3-only Go binary fails
#     clean on a v4 file (0.5 fallback) instead of mis-parsing.
#   - header grows from 44 → 48 bytes: appends a 12th uint32 LE field
#     `n_minuteprior` (0 or 1) AFTER `n_temporal`.
#   - input_dim contract becomes
#         backbone + n_logo + n_audio + n_channel + n_whisper
#         + n_temporal + n_minuteprior == input_dim
#   - input vector layout at inference becomes
#         [backbone(1280), logo?, audio?, chan_onehot?, whisper_prob?,
#          l2_dist_prev?, l2_dist_next?, minute_prior?]
#     (= appended LAST so v1/v2/v3 column order stays a prefix).
#     minute_prior = P(ad | minute_of_hour) from the per-channel
#     histogram sidecar head.minute-prior.json (built nightly from all
#     labelled recordings); frame → minute via the recording's wall-
#     clock start_ts + frame offset. Unknown slug / missing start_ts →
#     the sidecar's corpus-wide neutral value. See nn.go
#     loadMLPHeadV4 + minutePriorAt for the matching Go side.
# Migrated to production 2026-07-22 after a 3-night --shadow-eval
# series (+0.021/+0.021/+0.016 vs the cwt production replica).
def write_mlp_head_v4(path, mlp, *, input_dim, hidden_dim,
                      backbone_dim=1280, n_logo=0, n_audio=0,
                      n_channel=0, n_whisper=0, n_temporal=0,
                      n_minuteprior=0):
    """Serialise an sklearn-compatible MLP to the v4 MLP head.bin
    format. Same atomic write + shape validation as v3, plus the
    minute-prior input slot. Returns bytes written."""
    import struct
    if len(mlp.coefs_) != 2 or len(mlp.intercepts_) != 2:
        raise ValueError(f"expected single-hidden-layer MLP; got "
                         f"{len(mlp.coefs_)} coef matrices")
    W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
    W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
    if W1.shape != (input_dim, hidden_dim):
        raise ValueError(f"W1 shape {W1.shape} != ({input_dim}, {hidden_dim})")
    if b1.shape != (hidden_dim,):
        raise ValueError(f"b1 shape {b1.shape} != ({hidden_dim},)")
    output_dim = b2.shape[0]
    if W2.shape != (hidden_dim, output_dim):
        raise ValueError(f"W2 shape {W2.shape} != ({hidden_dim}, {output_dim})")
    if (backbone_dim + n_logo + n_audio + n_channel + n_whisper + n_temporal
            + n_minuteprior != input_dim):
        raise ValueError(
            f"input_dim {input_dim} != backbone {backbone_dim} + "
            f"logo {n_logo} + audio {n_audio} + chan {n_channel} + "
            f"whisper {n_whisper} + temporal {n_temporal} + "
            f"minuteprior {n_minuteprior}")
    header = struct.pack("<12I",
                         0x34504C4D,  # "MLP4" little-endian
                         4, input_dim, hidden_dim, output_dim,
                         backbone_dim, n_logo, n_audio, n_channel,
                         n_whisper, n_temporal, n_minuteprior)
    body = (W1.tobytes() + b1.tobytes()
            + W2.tobytes() + b2.tobytes())
    from pathlib import Path as _P
    p = _P(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(header + body)
    tmp.replace(p)
    return p.stat().st_size


# ---- v5 ("MLP5") -----------------------------------------------------
#   - magic is "MLP5" (= 0x35504C4D LE) — ein v4-Binary scheitert sauber
#     auf einer v5-Datei (0.5-Rueckfall) statt sie falsch zu lesen.
#   - Header waechst 48 → 52 Byte: haengt ein 13. uint32 LE `n_whispermask`
#     (0 oder 1) HINTER `n_minuteprior`.
#   - input_dim-Vertrag:
#         backbone + n_logo + n_audio + n_channel + n_whisper
#         + n_temporal + n_minuteprior + n_whispermask == input_dim
#   - Spaltenreihenfolge bei der Inferenz:
#         [backbone(1280), logo?, audio?, chan_onehot?, whisper_prob?,
#          l2_dist_prev?, l2_dist_next?, minute_prior?, whisper_mask?]
#     (= wieder GANZ HINTEN, damit v1..v4 ein Praefix bleiben).
#     whisper_mask = 1.0 wenn fuer die Aufnahme Whisper-Daten vorliegen,
#     sonst 0.0. Siehe _whisper_present fuer das Warum und nn.go
#     loadMLPHeadV5 fuer die Go-Seite.
def write_mlp_head_v5(path, mlp, *, input_dim, hidden_dim,
                      backbone_dim=1280, n_logo=0, n_audio=0,
                      n_channel=0, n_whisper=0, n_temporal=0,
                      n_minuteprior=0, n_whispermask=0):
    """Serialise an sklearn-compatible MLP to the v5 MLP head.bin format.
    Same atomic write + shape validation as v4, plus the whisper-presence
    slot. Returns bytes written."""
    import struct
    if len(mlp.coefs_) != 2 or len(mlp.intercepts_) != 2:
        raise ValueError(f"expected single-hidden-layer MLP; got "
                         f"{len(mlp.coefs_)} coef matrices")
    W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
    W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
    if W1.shape != (input_dim, hidden_dim):
        raise ValueError(f"W1 shape {W1.shape} != ({input_dim}, {hidden_dim})")
    if b1.shape != (hidden_dim,):
        raise ValueError(f"b1 shape {b1.shape} != ({hidden_dim},)")
    output_dim = b2.shape[0]
    if W2.shape != (hidden_dim, output_dim):
        raise ValueError(f"W2 shape {W2.shape} != ({hidden_dim}, {output_dim})")
    if (backbone_dim + n_logo + n_audio + n_channel + n_whisper + n_temporal
            + n_minuteprior + n_whispermask != input_dim):
        raise ValueError(
            f"input_dim {input_dim} != backbone {backbone_dim} + "
            f"logo {n_logo} + audio {n_audio} + chan {n_channel} + "
            f"whisper {n_whisper} + temporal {n_temporal} + "
            f"minuteprior {n_minuteprior} + whispermask {n_whispermask}")
    header = struct.pack("<13I",
                         0x35504C4D,  # "MLP5" little-endian
                         5, input_dim, hidden_dim, output_dim,
                         backbone_dim, n_logo, n_audio, n_channel,
                         n_whisper, n_temporal, n_minuteprior,
                         n_whispermask)
    body = (W1.tobytes() + b1.tobytes()
            + W2.tobytes() + b2.tobytes())
    from pathlib import Path as _P
    p = _P(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(header + body)
    tmp.replace(p)
    return p.stat().st_size


class _DeployedMLP:
    """Reconstructs a v2 ('MLP2') head.bin as a predict_proba-compatible object
    so the deploy gate can re-score the CURRENTLY-DEPLOYED head on the new test
    set — an apples-to-apples head-to-head that is robust to test-set
    composition changes (no historical IoU floor). Matches sklearn's
    MLPClassifier(activation='relu') binary forward: relu hidden + sigmoid out."""

    def __init__(self, W1, b1, W2, b2, input_dim):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.input_dim = input_dim

    def predict_proba(self, X):
        h = np.maximum(X.astype(np.float64) @ self.W1 + self.b1, 0.0)  # relu
        o = (h @ self.W2 + self.b2).ravel()
        p = 1.0 / (1.0 + np.exp(-o))  # sigmoid (binary head)
        return np.column_stack([1.0 - p, p])


def load_deployed_mlp(path):
    """Parse a v2 ('MLP2') or v3 ('MLP3') head.bin into a _DeployedMLP, or
    None if it isn't one (legacy logreg head / missing / corrupt). Used for
    the head-to-head deploy gate; the caller must check .input_dim matches
    the candidate's feature dim."""
    import struct
    try:
        raw = Path(path).read_bytes()
    except Exception:
        return None
    if len(raw) < 40:
        return None
    magic = struct.unpack("<I", raw[:4])[0]
    if magic == 0x35504C4D:  # "MLP5", version 5, 52-byte header
        if len(raw) < 52:
            return None
        hdr = struct.unpack("<13I", raw[:52])
        if hdr[1] != 5:
            return None
        input_dim, hidden_dim, output_dim = hdr[2], hdr[3], hdr[4]
        off = 52
    elif magic == 0x34504C4D:  # "MLP4", version 4, 48-byte header
        if len(raw) < 48:
            return None
        hdr = struct.unpack("<12I", raw[:48])
        if hdr[1] != 4:
            return None
        input_dim, hidden_dim, output_dim = hdr[2], hdr[3], hdr[4]
        off = 48
    elif magic == 0x33504C4D:  # "MLP3", version 3, 44-byte header
        if len(raw) < 44:
            return None
        hdr = struct.unpack("<11I", raw[:44])
        if hdr[1] != 3:
            return None
        input_dim, hidden_dim, output_dim = hdr[2], hdr[3], hdr[4]
        off = 44
    elif magic == 0x32504C4D:  # "MLP2", version 2, 40-byte header
        hdr = struct.unpack("<10I", raw[:40])
        if hdr[1] != 2:
            return None
        input_dim, hidden_dim, output_dim = hdr[2], hdr[3], hdr[4]
        off = 40
    else:
        return None

    def take(n):
        nonlocal off
        a = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).astype(np.float64)
        off += n * 4
        return a
    try:
        W1 = take(input_dim * hidden_dim).reshape(input_dim, hidden_dim)
        b1 = take(hidden_dim)
        W2 = take(hidden_dim * output_dim).reshape(hidden_dim, output_dim)
        b2 = take(output_dim)
    except Exception:
        return None
    return _DeployedMLP(W1, b1, W2, b2, input_dim)


class WeightedMLP:
    """Single-hidden-layer MLP (relu → sigmoid) with true fractional
    sample_weight — the drop-in replacement for sklearn's MLPClassifier
    in all head fits (2026-07-07).

    Why not MLPClassifier: it has no sample_weight, so weights were
    approximated by integer-rounded row duplication (max(round(sw),1)).
    That quantization silently erased the whole designed weighting —
    pseudo 0.3→1 (3× too strong), age-decay 0.5..0.99→1, bumper-boost
    1.4→1, NaN-logo 0→1 — only the user 2× survived. It also cost two
    full oversampled matrix copies (~29 GB peak) per fit.

    Semantics kept from MLPClassifier: Glorot-uniform init, Adam
    (lr 1e-3, β 0.9/0.999), L2 alpha 1e-4, early stopping on a 10%
    validation split with patience 10 / tol 1e-4, best-epoch weights
    restored. Deliberate differences: batch 512 (was min(200,n)) for
    BLAS efficiency, validation criterion = weighted log-loss (was
    accuracy), rows with weight<=0 are dropped up front. Deterministic
    per (data, random_state) like before. Exposes coefs_/intercepts_/
    n_iter_/loss_/predict/predict_proba, so write_mlp_head_v1/v2 and
    every consumer stay unchanged."""

    def __init__(self, hidden_dim=32, max_iter=80, random_state=0,
                 alpha=1e-4, batch_size=512, lr=1e-3,
                 validation_fraction=0.1, n_iter_no_change=10, tol=1e-4):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.coefs_ = None
        self.intercepts_ = None
        self.n_iter_ = 0
        self.loss_ = float("nan")

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        w = (np.ones(len(y), dtype=np.float32) if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float32).ravel())
        keep = w > 0
        if not keep.all():
            X, y, w = X[keep], y[keep], w[keep]
        n = len(y)
        if n == 0:
            raise ValueError("WeightedMLP.fit: no rows with weight > 0")
        # One shuffled copy; train/val are then contiguous slices (views),
        # so peak memory is input + this copy — no oversample, no second
        # sklearn-internal split copy.
        perm = rng.permutation(n)
        X, y, w = X[perm], y[perm], w[perm]
        n_val = max(1, int(n * self.validation_fraction)) if n >= 10 else 0
        Xt, yt, wt = X[n_val:], y[n_val:], w[n_val:]
        Xv, yv, wv = X[:n_val], y[:n_val], w[:n_val]

        d, h = X.shape[1], self.hidden_dim
        bound1 = np.sqrt(6.0 / (d + h))
        bound2 = np.sqrt(6.0 / (h + 1))
        W1 = rng.uniform(-bound1, bound1, (d, h)).astype(np.float32)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = rng.uniform(-bound2, bound2, (h, 1)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        def _val_loss():
            if n_val == 0:
                return float("nan")
            p = self._forward_params(Xv, W1, b1, W2, b2)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            bce = -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            return float((bce * wv).sum() / wv.sum())

        best_val = np.inf
        best_params = None
        stale = 0
        nt = len(yt)
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(nt)
            epoch_loss = 0.0
            epoch_wsum = 0.0
            for lo in range(0, nt, self.batch_size):
                idx = order[lo:lo + self.batch_size]
                xb, yb, wb = Xt[idx], yt[idx], wt[idx]
                nb = len(idx)
                z1 = xb @ W1 + b1
                a1 = np.maximum(z1, 0.0)
                z2 = (a1 @ W2).ravel() + b2[0]
                p = 1.0 / (1.0 + np.exp(-z2))
                pc = np.clip(p, 1e-7, 1 - 1e-7)
                wsum = wb.sum()
                bce = -(yb * np.log(pc) + (1 - yb) * np.log(1 - pc))
                epoch_loss += float((bce * wb).sum())
                epoch_wsum += float(wsum)
                # weighted BCE gradient + sklearn-style L2 (alpha/batch)
                dz2 = ((p - yb) * wb / wsum).astype(np.float32)
                gW2 = a1.T @ dz2[:, None] + (self.alpha / nb) * W2
                gb2 = np.array([dz2.sum()], dtype=np.float32)
                da1 = dz2[:, None] @ W2.T
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1 + (self.alpha / nb) * W1
                gb1 = dz1.sum(axis=0)
                t_step += 1
                for pi, gi in zip(range(4), (gW1, gb1, gW2, gb2)):
                    m[pi] = beta1 * m[pi] + (1 - beta1) * gi
                    v[pi] = beta2 * v[pi] + (1 - beta2) * gi * gi
                    mh = m[pi] / (1 - beta1 ** t_step)
                    vh = v[pi] / (1 - beta2 ** t_step)
                    params[pi] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
                W1, b1, W2, b2 = params
            self.n_iter_ = epoch
            self.loss_ = epoch_loss / max(epoch_wsum, 1e-9)
            vl = _val_loss()
            if n_val:
                if vl < best_val - self.tol:
                    best_val = vl
                    best_params = [p.copy() for p in params]
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.n_iter_no_change:
                        break
        if best_params is not None:
            W1, b1, W2, b2 = best_params
        self.coefs_ = [W1.astype(np.float64), W2.astype(np.float64)]
        self.intercepts_ = [b1.astype(np.float64), b2.astype(np.float64)]
        return self

    @staticmethod
    def _forward_params(X, W1, b1, W2, b2, chunk=1 << 18):
        out = np.empty(len(X), dtype=np.float64)
        for lo in range(0, len(X), chunk):
            xb = np.asarray(X[lo:lo + chunk], dtype=np.float32)
            a1 = np.maximum(xb @ W1 + b1, 0.0)
            z2 = (a1 @ W2).ravel() + b2[0]
            out[lo:lo + chunk] = 1.0 / (1.0 + np.exp(-z2.astype(np.float64)))
        return out

    def predict_proba(self, X):
        p = self._forward_params(X, self.coefs_[0].astype(np.float32),
                                 self.intercepts_[0].astype(np.float32),
                                 self.coefs_[1].astype(np.float32),
                                 self.intercepts_[1].astype(np.float32))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


WHISPER_CACHE = Path.home() / ".cache" / "tv-whisper"


def _whisper_present(uuid):
    """Gibt es fuer diese Aufnahme ueberhaupt Whisper-Daten?

    ⚠️ Das ist NICHT dasselbe wie "die Wahrscheinlichkeit ist 0.5".
    `_load_whisper_per_sec` fuellt fehlende Dateien mit neutralen 0.5 —
    und am 2026-08-06 hatten **300 von 557** archivierten Aufnahmen keine
    Whisper-Datei (historischer Einbruch im Juni, Quellen inzwischen
    geloescht, also nicht nachholbar; aktuelle Aufnahmen liegen bei 93 %).

    Der Kopf konnte "kein Ton-Signal" und "Ton sagt 50/50" damit nicht
    unterscheiden und hat gelernt, der Spalte nur halb zu trauen —
    obwohl er sie stark gewichtet (Norm 15.5, oberstes Perzentil) und sie
    im Betrieb fast immer echt ist. Gemessen auf dem Test-Satz des Laufs
    20260806T041055: Aufnahmen ohne Whisper liegen 0.049 IoU tiefer
    (p=0.034), und dieselbe Aufnahme mit/ohne Whisper-Feed schwankt um
    bis zu 0.6.

    Diese Funktion liefert die Indikatorspalte, die beides trennt.

    ⚠️ Die Bedingung muss der Go-Seite EXAKT entsprechen, sonst sieht der
    Kopf im Betrieb eine andere Spalte als im Training. Go setzt die Spur
    nur, wenn `signals.LoadWhisperPerSecond` durchlaeuft — und die
    verlangt: Datei lesbar, JSON parsebar, `windows` nicht leer. Eine
    blosse Existenzpruefung waere schon bei einer abgeschnittenen Datei
    auseinandergelaufen, und zwar lautlos.
    """
    p = WHISPER_CACHE / f"{uuid}.whisper.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return bool(d.get("windows"))


def _load_whisper_per_sec(uuid, n_seconds):
    """Per-second whisper-prob array (length n_seconds) from
    ~/.cache/tv-whisper/<uuid>.whisper.json. Each second is averaged
    across the windows that contain it (windows are 60 s long at
    30 s stride → 2× overlap typical). File missing or malformed →
    all-0.5 fallback (= neutral; lets the MLP treat the recording
    as whisper-uninformative without crashing the index path)."""
    p = WHISPER_CACHE / f"{uuid}.whisper.json"
    if not p.is_file():
        return np.full(n_seconds, 0.5, dtype=np.float32)
    try:
        d = json.loads(p.read_text())
    except Exception:
        return np.full(n_seconds, 0.5, dtype=np.float32)
    windows = d.get("windows", [])
    ws = int(d.get("window_s", 60))
    sums = np.zeros(n_seconds, dtype=np.float32)
    counts = np.zeros(n_seconds, dtype=np.int32)
    for w in windows:
        t0 = int(w.get("t", 0))
        prob = float(w.get("prob", 0.5))
        lo = max(0, t0)
        hi = min(n_seconds, t0 + ws)
        if hi > lo:
            sums[lo:hi] += prob
            counts[lo:hi] += 1
    out = np.full(n_seconds, 0.5, dtype=np.float32)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def slugify(name):
    s = name.lower()
    for k, v in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def build_onnx_session(backbone_path):
    # No explicit SessionOptions — that path triggers an empty
    # model_path check failure when the model has external-data
    # sidecars (PyTorch's exporter writes <name>.data alongside
    # <name>.onnx for any nontrivial model). Just pass the file path.
    avail = ort.get_available_providers()
    providers = [p for p in ("CoreMLExecutionProvider",
                              "CPUExecutionProvider") if p in avail]
    return ort.InferenceSession(backbone_path, providers=providers)


def preprocess_one(rgb, w, h):
    # rgb is bytes len 3*w*h. Output 1×3×224×224 float32.
    arr = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
    # bilinear resize via PIL is overkill; cv2 is faster but extra dep.
    # numpy roughly: nearest is fine for backbone training.
    sy = np.linspace(0, h - 1, 224).astype(np.int32)
    sx = np.linspace(0, w - 1, 224).astype(np.int32)
    small = arr[sy[:, None], sx[None, :]]
    f = small.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f = (f - mean) / std
    # HWC → CHW → NCHW
    return f.transpose(2, 0, 1)[None, ...]


def extract_frames_via_ffmpeg(src, w, h, fps=1.0):
    """Yield raw rgb24 bytes (one frame at a time) from ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", src,
        "-map", "0:v:0",
        "-vf", f"fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunk = 3 * w * h
    while True:
        buf = p.stdout.read(chunk)
        if len(buf) < chunk:
            break
        yield buf
    p.wait()


def probe_dims(src):
    # check_output only captures stdout; stderr leaks to parent → log
    # file when tv-train-head.sh runs us. Some DVB recordings have
    # damaged sequence headers that emit "[mpeg2video] Invalid frame
    # dimensions 0x0" at codec-init time, BELOW the `-v error` filter.
    # Pipe stderr to /dev/null so those warnings stay out of the log.
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", src,
    ], stderr=subprocess.DEVNULL).decode().strip()
    # MPEG-TS sometimes lists multiple video streams (e.g. 0x0 SDT
    # noise + the real one); take the first non-zero pair.
    for line in out.splitlines():
        parts = [p for p in line.strip().split(",") if p]
        if len(parts) >= 2:
            try:
                w, h = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if w > 0 and h > 0:
                return w, h
    raise RuntimeError(f"no usable video stream in {src}: {out!r}")


def featurize_recording(sess, src, fps_extract):
    """Stream frames from ffmpeg, push each through the ONNX backbone.
    Per-frame Run() — batching was tested and made things slower on
    M5 Pro because the ffmpeg pipe (not the backbone) is the
    bottleneck, and the np.stack copy adds overhead."""
    w, h = probe_dims(src)
    feats = []
    for buf in extract_frames_via_ffmpeg(src, w, h, fps_extract):
        x = preprocess_one(buf, w, h)
        out = sess.run(["features"], {"frame": x})[0]  # (1, 1280)
        feats.append(out[0])
    return np.stack(feats) if feats else np.zeros((0, 1280), np.float32)


_YAMNET = None  # lazy-loaded once per process

def _load_yamnet():
    """Load Google YAMNet from TF Hub. Frozen 1024-dim audio embedding
    extractor — trained on AudioSet (2 M YouTube clips, 521 classes).
    The embedding captures music vs speech vs noise patterns that
    raw RMS misses. Cached process-globally because TF graph load
    is ~3 s, and the worker pool calls into it dozens of times."""
    global _YAMNET
    if _YAMNET is not None:
        return _YAMNET
    import tensorflow_hub as hub
    _YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
    return _YAMNET


def extract_audio_yamnet_per_second(src, n_seconds, target_sr=16000):
    """Extract YAMNet 1024-dim audio embedding per second of source.

    Pipeline: ffmpeg → mono 16 kHz f32 PCM → YAMNet → embeddings.
    YAMNet's native frame rate is one embedding per ~0.48 s; we
    average pairs of consecutive embeddings to land at one vector
    per second, matching the per-frame backbone cadence.

    Returns (n_seconds, 1024) float32. Falls back to zeros on any
    failure (silent corruption beats crashing the whole training
    pipeline)."""
    import numpy as np
    import io
    neutral = np.zeros((n_seconds, 1024), dtype=np.float32)
    try:
        proc = subprocess.run([
            "ffmpeg", "-nostdin", "-nostats", "-loglevel", "error",
            "-i", str(src),
            "-map", "0:a:0", "-ac", "1", "-ar", str(target_sr),
            "-f", "f32le", "-"
        ], capture_output=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return neutral
    if proc.returncode != 0 or not proc.stdout:
        return neutral
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    if wav.size < target_sr:
        return neutral
    yam = _load_yamnet()
    # YAMNet returns (N_frames_0.48s, 521 scores), (N, 1024 emb), spectrogram
    _scores, embeddings, _spec = yam(wav)
    emb = embeddings.numpy()  # (N, 1024)
    # Pair-average to ~1 Hz (YAMNet ≈ 0.48 s/frame → 2 frames ≈ 1 s).
    pairs = emb.shape[0] // 2
    if pairs == 0:
        return neutral
    emb_1hz = emb[:pairs*2].reshape(pairs, 2, 1024).mean(axis=1)
    if pairs < n_seconds:
        result = neutral.copy()
        result[:pairs] = emb_1hz.astype(np.float32)
        return result
    return emb_1hz[:n_seconds].astype(np.float32)


def extract_audio_rms_per_second(src, n_seconds, sample_rate=48000):
    """Extract per-second RMS loudness via ffmpeg astats. Returns a
    (n_seconds,) float32 array normalised so very quiet (≤ -60 dB) → 0
    and full-scale (0 dB) → 1.

    German private TV consistently runs ads ~6-10 dB hotter than
    show content despite EU loudness regulation. The orthogonal
    information vs the visual backbone is what makes this worth
    adding — when the model sees a frame that LOOKS show-like but
    audio is at 0.9 normalised loudness, the combination is
    strictly more informative than either alone.

    Falls back to a neutral 0.5 array on any failure."""
    neutral = np.full(n_seconds, 0.5, dtype=np.float32)
    try:
        # ametadata defaults to writing each frame's metadata to
        # stderr at INFO level. We can't use `-loglevel error` to
        # suppress the [mpeg2video] codec warnings without losing
        # those metadata lines. Workaround: ametadata `file=-` →
        # writes to stdout instead, leaving stderr free to be quieted.
        # `fatal` level suppresses the warnings but keeps real fatal
        # errors visible (= empty stdout falls back to neutral, so
        # we'd notice missing data anyway).
        out = subprocess.run([
            "ffmpeg", "-nostdin", "-nostats", "-loglevel", "fatal",
            "-i", str(src),
            "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
            "-af", (f"asetnsamples=n={sample_rate},"
                    f"astats=metadata=1:reset=1,"
                    f"ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-"),
            "-f", "null", "/dev/null"
        ], capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return neutral
    rms_db_seq = []
    for line in out.stdout.splitlines():
        if "RMS_level=" in line:
            v = line.split("=", 1)[-1].strip()
            try:
                rms_db_seq.append(float(v))
            except ValueError:
                rms_db_seq.append(-90.0)  # ffmpeg writes "-inf" for silence
    if not rms_db_seq:
        return neutral
    rms_arr = np.array(rms_db_seq, dtype=np.float32)
    norm = np.clip((rms_arr + 60.0) / 60.0, 0.0, 1.0).astype(np.float32)
    if len(norm) < n_seconds:
        result = neutral.copy()
        result[:len(norm)] = norm
        return result
    return norm[:n_seconds]


def extract_uniformity_per_second(src, n_seconds):
    """Per-second luma spread (YHIGH-YLOW from ffmpeg signalstats),
    normalised to 0..1. High spread = textured (typical show), low
    spread = uniform (typical ad: text overlays, logo cards, branding
    backgrounds). Comskip's `non_uniformity` in the same vein.

    Why YHIGH-YLOW and not YMAX-YMIN: percentile-style spread is
    immune to a single outlier pixel, full-range is dominated by
    speculars/noise. signalstats has no YDEV/YSTDEV in the ffmpeg
    builds we ship (build flags don't include it).

    Returns (n_seconds,) float32 normalised so 0..200 spread → 0..1.
    Falls back to a neutral 0.5 array on any failure."""
    neutral = np.full(n_seconds, 0.5, dtype=np.float32)
    try:
        out = subprocess.run([
            "ffmpeg", "-nostdin", "-nostats", "-loglevel", "info",
            "-i", str(src),
            "-vf", "fps=1,signalstats,metadata=mode=print:file=-",
            "-an", "-f", "null", "-"
        ], capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return neutral
    # signalstats prints two lines per frame we care about — pair them.
    cur_low = cur_high = None
    spreads = []
    for line in out.stdout.splitlines():
        if "lavfi.signalstats.YLOW=" in line:
            try: cur_low = float(line.split("=", 1)[1].strip())
            except ValueError: cur_low = None
        elif "lavfi.signalstats.YHIGH=" in line:
            try: cur_high = float(line.split("=", 1)[1].strip())
            except ValueError: cur_high = None
            if cur_low is not None and cur_high is not None:
                spreads.append(cur_high - cur_low)
                cur_low = cur_high = None
    if not spreads:
        return neutral
    arr = np.array(spreads, dtype=np.float32)
    norm = np.clip(arr / 200.0, 0.0, 1.0)
    if len(norm) < n_seconds:
        result = neutral.copy()
        result[:len(norm)] = norm
        return result
    return norm[:n_seconds]


# Keep in sync with the daemon's detect_letterbox_offset() in
# ~/bin/tv-thumbs-daemon.py — both pipelines must apply the same
# y-offset to a given recording, otherwise cached training features
# (logoConf=0 in the black bar) won't match Mac-side inference values
# (logoConf>0 with offset applied).
LETTERBOX_LOGO_OVERHANG = 20


def detect_letterbox_offset(src):
    """Return recommended --logo-y-offset for `src`, or 0 if no
    meaningful letterbox. Runs a 5s cropdetect pass at the 60s mark
    (skips intros/promos)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "info",
             "-ss", "60", "-t", "5", "-i", str(src),
             "-vf", "cropdetect=24:16:0",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
    except Exception:
        return 0
    ys = re.findall(r"crop=\d+:\d+:\d+:(\d+)", r.stderr)
    if not ys:
        return 0
    y = int(ys[-1])
    if y < 8:
        return 0
    return max(0, y - LETTERBOX_LOGO_OVERHANG)


def extract_logo_per_second(src, logo_path, n_seconds, tv_detect, y_offset=0):
    """Run tv-detect --emit-logo-csv against `src` with `logo_path` as
    the channel template, then downsample the per-frame confidences
    to one value per second by mean. Returns a (n_seconds,) float32
    array. Used as the 1281st input feature to the WITH-LOGO head —
    lets the head learn channel-specific "trust the logo template"
    patterns instead of relying on post-hoc NNWeight blending.

    Sentinel for "not measured" is **NaN** (= train-head loaders detect
    this and substitute 0.5 right before fit, BUT also log per-recording
    miss rates so corruption-driven extraction failures stand out
    instead of being masked as plain "neutral" 0.5s in the cache). The
    earlier behaviour of writing a flat 0.5 array on failure made
    catastrophic decode failures (= h264 PPS errors mid-stream killing
    chunks 2+ silently) look identical to "logo present but ambiguous"
    — block-formation could not distinguish, IoU collapsed.

    NaN sources, in order of severity:
      - missing template / file → all-NaN (= channel has no template at all)
      - subprocess timeout / OSError → all-NaN
      - subprocess returncode != 0 → keep partial CSV that DID parse;
        mark untouched seconds NaN (= chunked decoder may produce
        chunk 0+1 frames before chunk 2 corruption aborts the run)
      - timestamps not present in CSV → NaN for those exact seconds"""
    nan_arr = np.full(n_seconds, np.nan, dtype=np.float32)
    if not logo_path or not Path(logo_path).exists():
        return nan_arr
    # --workers 2 (not 4): this subprocess runs INSIDE a worker of the
    # ProcessPoolExecutor that drives extraction (args.workers, default
    # 4). 4 outer workers × 4 inner decode threads = 16-way oversubscribe
    # on a machine that's also running 4 concurrent CoreML backbone
    # extractions — which starved/killed the logo subprocess for long
    # recordings, surfacing as whole-recording 100% NaN ("corrupt
    # stream" in the post-load summary) that re-queued and re-failed
    # every night without healing. 2 inner threads keeps the logo decode
    # alive under that load; solo it's only ~40 s for a 75-min recording
    # so the throughput loss is negligible.
    cmd = [tv_detect, "--quiet", "--workers", "2",
           "--logo", str(logo_path)]
    if y_offset > 0:
        cmd += ["--logo-y-offset", str(y_offset)]
    cmd += ["--emit-logo-csv", str(src)]
    # On timeout, SALVAGE the partial stdout the subprocess buffered
    # before the deadline (TimeoutExpired.stdout) instead of discarding
    # everything — a timed-out long recording then yields measured
    # seconds for the chunks that did decode, with only the tail NaN,
    # rather than an all-NaN row that zeroes the whole recording's
    # training weight. OSError (spawn/resource failure under contention)
    # has no output; retry once, then give up to all-NaN.
    stdout_text = ""
    for attempt in range(2):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=900)
            stdout_text = out.stdout or ""
            break
        except subprocess.TimeoutExpired as e:
            stdout_text = e.stdout or ""  # partial CSV decoded so far
            break
        except OSError:
            if attempt == 0:
                time.sleep(5)
                continue
            return nan_arr
    # NOTE: do NOT bail on returncode != 0 — partial CSV (= what chunks
    # successfully decoded before the failed one aborted) is still
    # useful. Untouched seconds stay NaN.
    sums = np.zeros(n_seconds, dtype=np.float64)
    counts = np.zeros(n_seconds, dtype=np.int32)
    for line in stdout_text.splitlines():
        if not line or line.startswith("idx"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            t = float(parts[1]); c = float(parts[2])
        except ValueError:
            continue
        s = int(t)
        if 0 <= s < n_seconds:
            sums[s] += c
            counts[s] += 1
    result = nan_arr.copy()
    mask = counts > 0
    result[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return result


# Channel one-hot ordering — MUST stay in sync with Go's nnChannels
# in internal/signals/nn.go. Append-only: never re-order or insert,
# or every previously trained head's channel weights map to wrong
# channels at inference.
CHANNELS = ["kabel-eins", "prosieben", "rtl", "sat-1", "sixx", "vox"]


def channel_one_hot(slug, n_seconds):
    """Return (n_seconds, len(CHANNELS)) sparse one-hot. Unknown
    slug = all zeros (matches Go-side channelIdx=-1 fallback)."""
    arr = np.zeros((n_seconds, len(CHANNELS)), dtype=np.float32)
    if slug in CHANNELS:
        arr[:, CHANNELS.index(slug)] = 1.0
    return arr


def smooth_mean(x, half_w):
    """Centered rolling mean. Same logic as Go-side smoothMean()
    in internal/blocks/blocks.go — ensures eval matches deployment."""
    if half_w <= 0 or len(x) == 0:
        return x
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


# Fold the Unicode dash variants EPG sources mix for the same show onto a plain
# "-", so a recording filename with an EN DASH ("Charmed – Zauberhafte Hexen")
# groups into the SAME per-show cohort as the hyphen airings ("Charmed - …")
# instead of fragmenting (which halved Charmed's per-show stats + split its
# training cluster). Mirrors tv-recorder's showDashReplacer + epg.go. 2026-06-05.
_DASH_FOLD = {ord(c): "-" for c in "‐‑‒–—―"}


def fold_show_title(t):
    """Canonical per-show key: dash-folded show title (no lowercasing — display
    casing stays intact)."""
    return t.translate(_DASH_FOLD)


# --- realistic eval: replay through the REAL blocks.Form() pipeline ------
#
# eval_split() used to threshold clf.predict_proba() at 0.5 and group
# contiguous runs (to_blocks below) — never exercising the production
# refinement blocks.Form() applies (logo blend, nn-gate, bumper/letterbox/
# I-frame/scene-cut snap, start/end-extend). Root-caused 2026-07-14 while
# investigating a training rejection: this meant BOTH the printed IoU
# numbers AND the head-to-head deploy gate were scoring a materially
# simpler pipeline than what actually ships, so absolute IoU was
# understated and (worse) the gate's deploy/reject calls weren't judging
# what production would actually do with a candidate head.
#
# Fix: tv-detect gained --emit-signals-json (dump every raw per-frame/event
# signal Form() consumes, once, right after a normal detect run) and
# --replay-signals (+--replay-nn-csv) to re-run ONLY block formation
# against a cached dump with a fresh NN confidence stream — fast (no
# ffmpeg), so a training run can call the real Form() once per candidate
# per test recording. See tv-detect/cmd/tv-detect/replay.go.
#
# Coverage is opportunistic, not guaranteed: replay needs (a) the
# recording's source .ts still in the local cache (LRU-evicted after
# ~60-1700 GB, dual-copy rule keeps the VOD but not necessarily the raw
# .ts) and (b) a cached logo template for its channel. Recordings without
# a cache fall back to the old to_blocks() threshold grouper — see the
# "realistic eval" coverage line eval_split() prints. Per-show/per-channel
# Form() overrides (e.g. RTL "Let's Dance" nn_gate=0/nn_weight=1.0, see
# memory tv_detect_letsdance_logo_hiding_fix) ARE applied — _replay_blocks
# fetches GET /api/internal/detect-config/<uuid> (memoized per uuid) and
# builds the CLI flags the same way tv-thumbs-daemon.py's process_detect
# does, not tv-detect's bare CLI defaults (which differ from production
# even with no override: logo-smooth 0 vs 5, bumper-snap 90 vs 10).
TVD_BIN = Path(__file__).resolve().parent.parent / "build" / "tv-detect"
MODEL_CACHE = Path.home() / ".cache" / "tv-detect-daemon"
SOURCE_CACHE = MODEL_CACHE / "source"
SIGNALS_CACHE = Path.home() / ".cache" / "tvd-eval-signals"
SIGNALS_CACHE.mkdir(parents=True, exist_ok=True)
MAX_NEW_SIGNALS_PER_RUN = 60  # cap first-time decode cost per training run
# (30→60 on 2026-07-14: at ~20s per build the worst case is ~20 min,
# and full test-set coverage in 1-2 nights unlocks both the realistic
# eval on all ~102 test recs and re-running the Form() param sweep on
# a broad basis before applying its per-channel suggestions.)


def _bumper_templates(slug):
    bdir = MODEL_CACHE / "bumpers" / slug
    if not bdir.is_dir():
        return [], []
    end = sorted(str(p) for p in bdir.glob("*.png"))
    if (bdir / "end").is_dir():
        end += sorted(str(p) for p in (bdir / "end").glob("*.png"))
    start = []
    if (bdir / "start").is_dir():
        start = sorted(str(p) for p in (bdir / "start").glob("*.png"))
    return end, start


def _ensure_signals_cache(uuid, slug, budget):
    """Build (once) or reuse the --emit-signals-json cache for uuid.
    Returns the cache path, or None (caller falls back to to_blocks()).
    budget is a [int] 1-elem list of remaining new builds this run —
    decremented on an actual decode, shared across all callers so the
    MAX_NEW_SIGNALS_PER_RUN cap applies process-wide."""
    cache_path = SIGNALS_CACHE / f"{uuid}.json"
    if cache_path.is_file():
        return cache_path
    if cache_path.with_suffix(".stale").is_file():
        # Tombstoned by _replay_blocks: the current source is known to
        # mismatch this recording's frozen features — rebuilding would
        # just re-create the same doomed cache. Cleared naturally if
        # the features ever get re-extracted (different filename) or
        # manually by deleting the .stale file.
        return None
    if budget[0] <= 0 or not TVD_BIN.is_file():
        return None
    src = SOURCE_CACHE / f"{uuid}.ts"
    if not src.is_file():
        return None
    logo_path = MODEL_CACHE / "logos" / f"{slug}.logo.txt"
    if not logo_path.is_file():
        return None
    end_bumpers, start_bumpers = _bumper_templates(slug)
    # Go's flag package stops parsing at the first non-flag token — the
    # positional <input> must come LAST, after every --flag.
    cmd = [str(TVD_BIN), "--quiet", "--logo", str(logo_path),
           "--emit-signals-json", str(cache_path), "--output", "summary"]
    if end_bumpers:
        cmd += ["--bumper-templates", ",".join(end_bumpers)]
    if start_bumpers:
        cmd += ["--bumper-templates-start", ",".join(start_bumpers)]
    cmd.append(str(src))
    budget[0] -= 1
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                        text=True, timeout=1800)
    except Exception as e:
        print(f"  signals-cache: {uuid[:8]} build failed: {e}", flush=True)
        return None
    return cache_path if cache_path.is_file() else None


_signals_header_cache = {}  # cache_path -> (fps, frame_count), avoids re-parsing the JSON per candidate eval


def _signals_header(cache_path):
    key = str(cache_path)
    if key not in _signals_header_cache:
        with open(cache_path) as f:
            d = json.load(f)
        _signals_header_cache[key] = (d["fps"], d["frame_count"])
    return _signals_header_cache[key]


def _signals_cache_path(uuid):
    """Read-only lookup for eval_split() — never builds (that only happens
    in the one-time pre-pass in main(), which has the uuid->slug map;
    building requires a real decode and has no business running inside a
    metric function called many times per training run)."""
    p = SIGNALS_CACHE / f"{uuid}.json"
    return p if p.is_file() else None


_detect_config_cache = {}  # uuid -> gateway /api/internal/detect-config response (or None), memoized


def _fetch_detect_config(uuid):
    """GET /api/internal/detect-config/<uuid> — the exact per-show/per-channel
    Form() overrides (nn_gate/nn_weight/bumper_threshold/logo_smooth_s/
    start_extend_s/end_extend_s/min_block_s/max_block_s) production's Mac
    daemon (tv-thumbs-daemon.py) applies for this recording — e.g. RTL
    "Let's Dance" nn_gate=0/nn_weight=1.0, see memory
    tv_detect_letsdance_logo_hiding_fix. Memoized per uuid (fixed for a
    given recording's lifetime, called once per candidate eval otherwise)."""
    if uuid in _detect_config_cache:
        return _detect_config_cache[uuid]
    cfg = None
    try:
        # /api/internal/* is Caddy/:8443-only (not plain :9983, unlike
        # /api/dvr/* above) — the same host:port tv-thumbs-daemon.py's
        # GATEWAY constant uses for this exact endpoint.
        import urllib.request, ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(
                f"https://raspberrypi5lan:8443/api/internal/detect-config/{uuid}",
                timeout=10, context=ctx) as r:
            cfg = json.loads(r.read())
    except Exception:
        cfg = None
    _detect_config_cache[uuid] = cfg
    return cfg


# Block-forming decoder this evaluation uses. MUST track what the Mac daemon
# actually runs (tv-thumbs-daemon.py process_detect), because everything
# downstream — the head-to-head deploy gate, the golden trend, the
# active-learning cohort weights — is computed from the per-rec IoU this
# produces.
#
# ⚠️ Until 2026-08-05 this was not passed at all. tv-detect's --decoder
# defaults to "form", while PRODUCTION has run "hsmm --hsmm-dur-w 15" since
# 2026-07-29. For a week the nightly selected heads by a decoder that no
# longer cut a single recording.
#
# That is not a cosmetic mismatch, because Form is structurally almost blind
# to the head: its per-frame score is a logo/NN blend at nn_weight 0.3, so
# the NN carries 30% of the decision, while the HSMM consumes the raw NN
# probability as 100% of its emission. Measured 2026-08-05 on 14 golden
# recordings with this function's own flags, swapping ONLY the head that
# produced the probabilities (deployed 07-29 champion vs the 08-03 head):
#
#     under form (what the gate saw):     0.773 vs 0.808   -> +0.035
#     under hsmm (what production does):  0.828 vs 0.939   -> +0.111
#
# and 11 of the 14 scored BYTE-IDENTICAL under form for the two heads. The
# gate could barely see a difference that costs production 11 IoU points.
#
# ⚠️ Changing this makes every golden/per-rec number before 2026-08-05
# incomparable to every number after — they measure different block formers.
# `decoder` is therefore persisted into golden-trend.jsonl; compare only
# within one decoder.
EVAL_DECODER = ["--decoder", "hsmm", "--hsmm-dur-w", "15"]


def _replay_blocks(cache_path, proba, fps_extract, uuid, default_min_block_s=60):
    """Feeds one candidate's per-frame (fps_extract-rate) ad-probability
    through the real production block former via `tv-detect
    --replay-signals` against the cached decode signals. Builds the CLI
    flags the exact same way tv-thumbs-daemon.py's process_detect does
    (same always-pass-nn-weight/logo-smooth, only-pass-if-set nn-gate/
    start-extend/end-extend, hardcoded bumper-snap=90 semantics, and
    EVAL_DECODER) — so a per-show/per-channel override actually changes
    what eval measures, instead of silently falling back to tv-detect's
    bare CLI defaults (which differ from production's: logo-smooth
    defaults to 0 in production vs 5 in the CLI, bumper-snap to 90 vs 10,
    and the decoder to form vs hsmm). Returns a [(start_s,end_s), ...]
    list, or None on any failure (caller falls back to to_blocks())."""
    try:
        fps, frame_count = _signals_header(cache_path)
    except Exception:
        return None
    # Staleness guard (2026-07-16, found via a user TRIM): the signals
    # cache is a decode snapshot — when the recording's source changes
    # (trim replaces the .ts; the daemon's freshness sweep invalidates
    # features + archive but doesn't know about THIS cache), the cached
    # duration no longer matches the current features. Replaying proba
    # of one length against signals of another silently mis-scores the
    # recording. Compare durations; >10s apart → drop the cache (the
    # nightly pre-pass rebuilds it from the new source) and fall back.
    if abs(frame_count / fps - len(proba) / fps_extract) > 10:
        try:
            Path(cache_path).unlink()
            _signals_header_cache.pop(str(cache_path), None)
            # Tombstone: when the CURRENT source itself no longer matches
            # the frozen features (source truncated after extraction, rec
            # dead → features never re-extracted), a rebuild produces the
            # exact same mismatch — without this marker the pre-pass
            # rebuilt and the eval re-dropped the cache every single
            # night (observed 07-17/07-18 on a dead Let's Dance rec).
            Path(cache_path).with_suffix(".stale").write_text(
                f"cache {frame_count / fps:.0f}s vs features "
                f"{len(proba) / fps_extract:.0f}s")
            print(f"  signals-cache: {uuid[:8]} stale "
                  f"(cache {frame_count / fps:.0f}s vs features "
                  f"{len(proba) / fps_extract:.0f}s — source changed?), "
                  f"dropped + tombstoned", flush=True)
        except OSError:
            pass
        return None
    cfg = _fetch_detect_config(uuid) or {}
    nn_weight = cfg.get("nn_weight", -1)
    nn_weight = nn_weight if nn_weight is not None and nn_weight >= 0 else 0.3
    logo_smooth = cfg.get("logo_smooth_s") or 0
    nn_gate = cfg.get("nn_gate", -1)
    min_block_s, max_block_s = default_min_block_s, None
    if cfg.get("min_block_s") and cfg.get("max_block_s"):
        min_block_s, max_block_s = cfg["min_block_s"], cfg["max_block_s"]
    nn_csv = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False) as f:
            nn_csv = f.name
            f.write("idx,time_s,nn_confidence\n")
            for i in range(frame_count):
                src_i = min(len(proba) - 1, int(i / fps * fps_extract))
                f.write(f"{i},{i / fps:.3f},{proba[src_i]:.4f}\n")
        cmd = [str(TVD_BIN), "--quiet", "--replay-signals", str(cache_path),
               "--replay-nn-csv", nn_csv, "--output", "summary",
               "--min-block-sec", str(min_block_s),
               "--nn-weight", str(nn_weight),
               "--logo-smooth", str(logo_smooth),
               # Hardcoded to match process_detect's own bumper-snap window
               # (wider than tv-detect's own --bumper-snap default of 10) —
               # harmless no-op when the cached signals carry no bumper conf
               # (empty templates at emit-signals-json time).
               "--bumper-snap", "90",
               "--bumper-threshold", str(cfg.get("bumper_threshold", 0.75))]
        if max_block_s:
            cmd += ["--max-block-sec", str(max_block_s)]
        if nn_gate is not None and nn_gate >= 0:
            cmd += ["--nn-gate", str(nn_gate)]
        nn_smooth = cfg.get("nn_smooth", -1)
        if nn_smooth is not None and nn_smooth >= 0:
            cmd += ["--nn-smooth", str(nn_smooth)]
        if cfg.get("start_extend_s", 0):
            cmd += ["--start-extend", str(cfg["start_extend_s"])]
        if cfg.get("end_extend_s", 0):
            cmd += ["--end-extend", str(cfg["end_extend_s"])]
        # ⚠️ Zuletzt, damit klar bleibt: OHNE das hier misst die ganze
        # Bewertung "form", und die Produktion schneidet mit "hsmm".
        cmd += EVAL_DECODER
        cmd.append("dummy")
        r = subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=60)
        out = json.loads(r.stdout)
        return [(float(b[0]), float(b[1])) for b in out.get("blocks", [])]
    except Exception:
        return None
    finally:
        if nn_csv:
            try:
                os.unlink(nn_csv)
            except OSError:
                pass


def to_blocks(preds, fps=1.0, min_block_s=30):
    """Convert per-frame ad/show predictions to a list of contiguous
    [start_s, end_s] blocks. Mimics the deployed state machine's
    minimum-block filter — anything shorter than min_block_s gets
    dropped, since the production blocks.Form() does the same."""
    blocks = []
    in_block = False
    start = 0
    for i, p in enumerate(preds):
        if p and not in_block:
            start = i
            in_block = True
        elif (not p) and in_block:
            blocks.append((start / fps, i / fps))
            in_block = False
    if in_block:
        blocks.append((start / fps, len(preds) / fps))
    return [b for b in blocks if b[1] - b[0] >= min_block_s]


def block_iou(pred_blocks, gt_blocks):
    """Mean IoU across ground-truth blocks: for each GT block, find
    the predicted block with the highest overlap and use its IoU.
    Returns 1.0 if both are empty, 0.0 if exactly one is empty."""
    if not gt_blocks and not pred_blocks:
        return 1.0
    if not gt_blocks or not pred_blocks:
        return 0.0
    out = []
    for gt in gt_blocks:
        best = 0.0
        for pr in pred_blocks:
            inter = max(0, min(pr[1], gt[1]) - max(pr[0], gt[0]))
            union = max(pr[1], gt[1]) - min(pr[0], gt[0])
            if union > 0:
                best = max(best, inter / union)
        out.append(best)
    return sum(out) / len(out)


def confusion_analysis(clf, recs, fps_extract, smooth_s, output_path):
    """Per-test-recording forensic dump: where exactly does the model
    fail? Classifies each error-frame run by its position relative
    to GT block edges, plus a block-level matched/missed/extra view.

    Boundary errors (within ±10s of a GT edge) are usually harmless
    label-noise — auto-detection rounds boundaries by a few seconds,
    smoothing shifts them another 2-5s, the human label might be off
    too. They drag IoU but can't be improved without sub-second
    ground truth.

    Intra-block errors (>10s deep into ad or show) point to genuine
    confusion: bumpers/promos that look like ads, sponsor cards
    inside the show, etc. Those are the targets for active labelling
    or feature improvements."""
    half_w = int(smooth_s * fps_extract / 2) if smooth_s > 0 else 0
    written = 0
    with open(output_path, "w") as f:
        f.write(f"# tv-detect confusion analysis (smooth_s={smooth_s})\n")
        f.write(f"# Generated by train-head.py against the held-out test set.\n")
        f.write(f"# Boundary error: error frame within ±10s of any GT block edge.\n")
        f.write(f"# Intra error: error frame >10s from any GT edge.\n")
        f.write(f"# Multi-frame runs (≥5) listed; isolated single-frame errors omitted.\n\n")

        for uuid, title, ads, X, y, *_ in recs:
            proba = clf.predict_proba(X)[:, 1]
            if half_w > 0:
                proba = smooth_mean(proba, half_w)
            pred = (proba >= 0.5).astype(np.int32)
            errors = (pred != y)
            n_err = int(errors.sum())

            edges = sorted([float(s) for s, _ in ads] +
                           [float(e) for _, e in ads])

            def near_edge(t):
                if not edges:
                    return False
                return min(abs(t - e) for e in edges) < 10.0

            boundary_n = intra_n = 0
            error_runs = []  # (t_start, t_end, length, type, FN/FP)
            in_run = False; run_start = 0
            for i, is_err in enumerate(errors):
                if is_err and not in_run:
                    in_run = True; run_start = i
                elif (not is_err) and in_run:
                    in_run = False
                    length = i - run_start
                    t = run_start / fps_extract
                    t_end = i / fps_extract
                    is_boundary = near_edge(t) or near_edge(t_end)
                    fnfp = "FN" if y[run_start] == 1 else "FP"
                    if is_boundary:
                        boundary_n += length
                        etype = "boundary"
                    else:
                        intra_n += length
                        etype = "intra"
                    if length >= 5:
                        error_runs.append((t, t_end, length, etype, fnfp))
            if in_run:
                length = len(errors) - run_start
                t = run_start / fps_extract; t_end = len(errors) / fps_extract
                is_boundary = near_edge(t) or near_edge(t_end)
                fnfp = "FN" if y[run_start] == 1 else "FP"
                if is_boundary: boundary_n += length
                else: intra_n += length
                if length >= 5:
                    error_runs.append((t, t_end, length,
                                       "boundary" if is_boundary else "intra",
                                       fnfp))

            # Block-level analysis
            pred_blocks = to_blocks(pred, fps=fps_extract)
            gt_blocks = [(float(s), float(e)) for s, e in ads]
            missed, extra, matched = [], [], []
            for gs, ge in gt_blocks:
                ovl = [(ps, pe) for ps, pe in pred_blocks
                       if ps < ge and pe > gs]
                if not ovl:
                    missed.append((gs, ge))
                else:
                    for ps, pe in ovl:
                        inter = max(0, min(pe, ge) - max(ps, gs))
                        union = max(pe, ge) - min(ps, gs)
                        iou_b = inter / union if union > 0 else 0
                        matched.append((gs, ge, ps, pe, iou_b))
            for ps, pe in pred_blocks:
                if not any(gs < pe and ge > ps for gs, ge in gt_blocks):
                    extra.append((ps, pe))

            f.write(f"## {title}  ({uuid[:8]})\n")
            f.write(f"  frames:  total={len(y)}  errors={n_err} "
                    f"({100*n_err/len(y):.1f}%)\n")
            f.write(f"  errors:  boundary={boundary_n}  intra={intra_n}\n")
            f.write(f"  blocks:  GT={len(gt_blocks)}  pred={len(pred_blocks)}  "
                    f"matched={len(matched)}  missed={len(missed)}  "
                    f"extra={len(extra)}\n")
            for gs, ge, ps, pe, iou_b in matched:
                f.write(f"    matched  GT[{gs:5.0f},{ge:5.0f}]  "
                        f"pred[{ps:5.0f},{pe:5.0f}]  IoU={iou_b:.2f}\n")
            for gs, ge in missed:
                f.write(f"    MISSED   GT[{gs:5.0f},{ge:5.0f}]  ({ge-gs:.0f}s)\n")
            for ps, pe in extra:
                f.write(f"    EXTRA  pred[{ps:5.0f},{pe:5.0f}]  ({pe-ps:.0f}s)\n")
            for t, t_end, length, etype, fnfp in error_runs[:30]:
                f.write(f"    err  t={t:6.0f}-{t_end:6.0f}s  {fnfp}  "
                        f"{etype:8s}  {length} frames\n")
            f.write("\n")
            written += 1
    print(f"\nconfusion: {written} test recordings analysed → {output_path}")


def _is_likely_movie(title, n_recs, total_frames, fps_extract):
    """Pragmatic movie-vs-TV split for eval reporting. Mirrors the
    /bibliothek heuristic loosely: long broadcast + single recording
    + no series-style subtitle in the title. Recordings that fail any
    check default to TV-class.

    Heuristic only — there will be edge cases (Staying Alive 157 min
    has " - Stars singen mit Legenden" subtitle → correctly TV; a true
    movie rebroadcast without subtitle stays movie). Goal is not
    perfection but stopping a single ~150 min Moonfall-like outlier
    from dominating the headline OVERALL IoU."""
    if n_recs > 1:
        return False
    avg_duration_s = total_frames / fps_extract
    if avg_duration_s < 80 * 60:
        return False
    if " - " in title or " — " in title:
        return False
    return True


def eval_split(clf, recs, fps_extract, smooth_s=0):
    """Per-frame + block-level evaluation of `clf` on the held-out
    recordings, broken down by show title (proxy for channel since
    the channel slug isn't in the cache index).

    smooth_s > 0: apply the same rolling-mean smoothing the Go
    pipeline uses (NNSmoothS) so the eval reflects deployment."""
    suffix = f" (smooth={smooth_s}s)" if smooth_s > 0 else ""
    print(f"\n=== held-out evaluation{suffix} ===")
    by_show = {}  # title -> {frames, correct, tp, fp, fn, ious[], n_recs}
    per_rec_iou = {}  # uuid -> IoU, for the paired head-to-head gate
    overall_frames = overall_correct = 0
    half_w = int(smooth_s * fps_extract / 2) if smooth_s > 0 else 0
    n_realistic = n_fallback = 0
    per_rec_stats = []  # (uuid, title, n_frames, n_errors) for the GT-outlier guard
    for uuid, title, ads, X, y, *_rest in recs:
        has_user = bool(_rest[0]) if _rest else False
        proba = clf.predict_proba(X)[:, 1]
        if half_w > 0:
            proba = smooth_mean(proba, half_w)
        pred = (proba >= 0.5).astype(np.int32)
        n = len(y)
        correct = int((pred == y).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        # Realistic path: replay the RAW (unsmoothed/unthresholded) proba
        # through the real blocks.Form() pipeline (logo blend, nn-gate,
        # bumper/letterbox/I-frame snap) when a decode-signals cache exists
        # for this recording — Form applies its own NNSmoothS, so feeding
        # smoothed proba here would double-smooth. Falls back to the naive
        # threshold+contiguous-run grouper otherwise.
        cache_path = _signals_cache_path(uuid)
        pred_blocks = None
        if cache_path is not None:
            pred_blocks = _replay_blocks(cache_path, clf.predict_proba(X)[:, 1],
                                          fps_extract, uuid)
        if pred_blocks is not None:
            n_realistic += 1
        else:
            n_fallback += 1
            pred_blocks = to_blocks(pred, fps=fps_extract)
        # ads is already in seconds (start, end pairs).
        gt_blocks = [(float(a[0]), float(a[1])) for a in ads]
        iou = block_iou(pred_blocks, gt_blocks)
        per_rec_iou[uuid] = iou
        b = by_show.setdefault(title, {"frames": 0, "correct": 0,
                                       "tp": 0, "fp": 0, "fn": 0,
                                       "ious": [], "n_recs": 0})
        b["frames"] += n
        b["correct"] += correct
        b["tp"] += tp; b["fp"] += fp; b["fn"] += fn
        b["ious"].append(iou)
        b["n_recs"] += 1
        overall_frames += n
        overall_correct += correct
        per_rec_stats.append((uuid, title, n, n - correct, has_user))
    # GT-outlier guard (2026-07-15): a healthy head doesn't fail at
    # 47-65% on individual recordings while scoring ~96% overall — when
    # it does, the recording's frozen ground truth is the likely
    # culprit, not the model (found twice this week: "Reisen mit
    # Kreta.de", then "Abenteuer Leben täglich"+"Von Hecke zu Hecke" =
    # 50% of ALL measured test errors in two dead recs with
    # incomplete old auto-era cutlists). Print suspects so the next
    # broken-GT recording surfaces in the nightly log immediately
    # instead of silently dragging metrics for weeks. Heuristic:
    # per-rec error rate >= 25% AND >= 5x the corpus-wide rate.
    if overall_frames:
        _corpus_err = 1.0 - overall_correct / overall_frames
        # User-reviewed recordings are EXCLUDED from the suspect list: their
        # GT is verified ground truth, so a high frame-error there is a model
        # weakness (or a frame-vs-block scoring quirk), NOT bad GT — suggesting
        # TEST_SET_EXCLUDE for them would drop a good, hand-checked test rec.
        # 2026-07-24: Rubble & Crew (dvr-toggo-plus-1780640400) kept re-flagging
        # at err 28-70% for two nights AFTER the user reviewed it, though its
        # block-IoU is a healthy 0.79 — pure false alarm. The heuristic exists
        # to surface UNreviewed recs with likely-broken frozen auto-GT.
        _suspects = [(u, t, e / n) for u, t, n, e, rev in per_rec_stats
                     if n > 0 and not rev and e / n >= 0.25
                     and e / n >= 5 * max(_corpus_err, 0.01)]
        if _suspects:
            print(f"  ⚠ GT-outlier suspects (unreviewed, err-rate >=25% and >=5x "
                  f"corpus mean {100*_corpus_err:.1f}% — check the "
                  f"recording's frozen ads list before blaming the "
                  f"model; candidates for TEST_SET_EXCLUDE):")
            for u, t, r in sorted(_suspects, key=lambda x: -x[2]):
                print(f"    {u}  {t[:40]:40s} err={100*r:.0f}%")
    if n_realistic or n_fallback:
        print(f"  realistic eval: {n_realistic}/{n_realistic + n_fallback} "
              f"test recs via full production pipeline (blocks.Form()), "
              f"{n_fallback} via naive threshold fallback (no cached "
              f"decode signals for that recording)")
    # Per-show table.
    print(f"{'show':40s} {'recs':>4} {'frames':>7} {'acc':>6} "
          f"{'F1':>5} {'IoU':>5}")
    for title in sorted(by_show.keys()):
        b = by_show[title]
        acc = b["correct"] / b["frames"] if b["frames"] else 0
        prec = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0
        rec  = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        iou = sum(b["ious"]) / len(b["ious"]) if b["ious"] else 0
        print(f"{title[:40]:40s} {b['n_recs']:>4} {b['frames']:>7} "
              f"{acc*100:>5.1f}% {f1:>5.2f} {iou:>5.2f}")
    overall_acc = overall_correct / overall_frames if overall_frames else 0
    all_ious = [i for b in by_show.values() for i in b["ious"]]
    overall_iou = sum(all_ious) / len(all_ious) if all_ious else 0
    # Median IoU as a 2nd metric robust to per-recording outliers
    # (= a single Moonfall at 0.56 IoU drags 37-recording mean by
    # 1.5 pp on its own; median ignores it). Used by the deploy
    # gate downstream — mean-IoU still printed for continuity.
    overall_iou_median = (float(np.median(all_ious))
                          if all_ious else 0)
    tp = sum(b["tp"] for b in by_show.values())
    fp = sum(b["fp"] for b in by_show.values())
    fn = sum(b["fn"] for b in by_show.values())
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    # Movies vs TV split — movies have fundamentally different ad
    # structure (no periodic breaks) and a single 150-min film with
    # 0.40 IoU dominates the mean. Report both classes separately
    # so the headline OVERALL is still comparable run-to-run, but
    # movie-class regressions can also be tracked. Headline OVERALL
    # remains unified for backward compat with display; the deploy-
    # gate downstream prefers iou_tv_median (TV-class only) so a
    # rebroadcast of one tricky film doesn't block a TV deployment.
    movie_titles = {t for t, b in by_show.items()
                    if _is_likely_movie(t, b["n_recs"], b["frames"],
                                         fps_extract)}
    iou_tv = iou_tv_median = None
    for label, titles in (("TV", set(by_show.keys()) - movie_titles),
                          ("movies", movie_titles)):
        if not titles:
            continue
        sub_b = [by_show[t] for t in titles]
        sub_frames = sum(b["frames"] for b in sub_b)
        sub_correct = sum(b["correct"] for b in sub_b)
        sub_ious = [i for b in sub_b for i in b["ious"]]
        sub_n_recs = sum(b["n_recs"] for b in sub_b)
        sub_acc = sub_correct / sub_frames if sub_frames else 0
        sub_iou = sum(sub_ious) / len(sub_ious) if sub_ious else 0
        sub_med = float(np.median(sub_ious)) if sub_ious else 0
        sub_tp = sum(b["tp"] for b in sub_b)
        sub_fp = sum(b["fp"] for b in sub_b)
        sub_fn = sum(b["fn"] for b in sub_b)
        sub_p = sub_tp/(sub_tp+sub_fp) if (sub_tp+sub_fp) else 0
        sub_r = sub_tp/(sub_tp+sub_fn) if (sub_tp+sub_fn) else 0
        sub_f1 = 2*sub_p*sub_r/(sub_p+sub_r) if (sub_p+sub_r) else 0
        if label == "TV":
            iou_tv = sub_iou
            iou_tv_median = sub_med
        print(f"{'OVERALL (' + label + ')':40s} {sub_n_recs:>4} "
              f"{sub_frames:>7} {sub_acc*100:>5.1f}% {sub_f1:>5.2f} "
              f"{sub_iou:>5.2f} (median {sub_med:.2f})")
    print(f"{'OVERALL':40s} {len(recs):>4} {overall_frames:>7} "
          f"{overall_acc*100:>5.1f}% {f1:>5.2f} {overall_iou:>5.2f} "
          f"(median {overall_iou_median:.2f})")
    return {"acc": overall_acc, "f1": f1, "iou": overall_iou,
            "iou_median": overall_iou_median,
            "iou_tv": iou_tv if iou_tv is not None else overall_iou,
            "iou_tv_median": (iou_tv_median if iou_tv_median is not None
                              else overall_iou_median),
            "n_recs": len(recs), "n_frames": overall_frames,
            "per_rec_iou": per_rec_iou}


def labels_for(seconds, ad_blocks):
    """seconds is a list of frame timestamps (1 per fps_extract step).
    Returns 0/1 per timestamp; 1 if t falls inside any (s,e)."""
    out = np.zeros(len(seconds), dtype=np.float32)
    for i, t in enumerate(seconds):
        for s, e in ad_blocks:
            if s <= t <= e:
                out[i] = 1
                break
    return out


# Per-worker ONNX session for ProcessPool. Each subprocess builds its
# own session at startup; sharing across processes isn't safe.
_WORKER_SESS = None

def _worker_init(backbone_path):
    global _WORKER_SESS
    _WORKER_SESS = build_onnx_session(backbone_path)

def _worker_extract(args):
    """Subprocess entry. Returns (cache_path, features). Caller
    persists to disk to avoid every worker writing concurrently to
    shared state.

    Feature column layout (any subset, in order):
      [0..1280)        backbone embedding
      logo conf        (1 col, if logo_path given)
      channel one-hot  (6 cols, if chan_slug given)
      audio rms        (1 col, if with_audio)
      yamnet emb       (1024 cols, if with_yamnet)
      uniformity       (1 col, if with_uniformity)
    Both Go inference and the cache key encoding rely on this exact
    order — keep concat operations in the same sequence."""
    (src, fps_extract, cache_path, logo_path, tv_detect_bin,
     chan_slug, with_audio, with_yamnet, with_uniformity) = args
    feats = featurize_recording(_WORKER_SESS, src, fps_extract)
    if logo_path:
        # Letterbox-aware: shift template y-coords down by N pixels for
        # 16:9-in-4:3 broadcasts. Without this, cached features for
        # affected recordings have logoConf=0 throughout, while Mac-side
        # inference now gets non-zero values — model trained on stale
        # features would mispredict at inference time.
        y_off = detect_letterbox_offset(src)
        logo_arr = extract_logo_per_second(
            src, logo_path, n_seconds=feats.shape[0],
            tv_detect=tv_detect_bin, y_offset=y_off)
        feats = np.concatenate(
            [feats, logo_arr.reshape(-1, 1).astype(np.float32)], axis=1)
    if chan_slug:
        chan = channel_one_hot(chan_slug, feats.shape[0])
        feats = np.concatenate([feats, chan], axis=1)
    if with_audio:
        rms = extract_audio_rms_per_second(src, n_seconds=feats.shape[0])
        feats = np.concatenate(
            [feats, rms.reshape(-1, 1).astype(np.float32)], axis=1)
    if with_yamnet:
        yam = extract_audio_yamnet_per_second(src, n_seconds=feats.shape[0])
        feats = np.concatenate([feats, yam], axis=1)
    if with_uniformity:
        u = extract_uniformity_per_second(src, n_seconds=feats.shape[0])
        feats = np.concatenate(
            [feats, u.reshape(-1, 1).astype(np.float32)], axis=1)
    return cache_path, feats


def _churn_col(X, fenster=61):
    """Bildunruhe-Niveau: der 1s-L2-Delta, über `fenster` Sekunden gemittelt.

    Warum das gebraucht wird. Die beiden Produktionsspalten sind der Abstand
    zur Nachbarsekunde — ein Einzelsprung, und der ist verrauscht. Gemessen
    über den Golden-Satz (Rang-AUC gegen die Labels, 2026-08-06):

        Delta über 1s (Produktion)    0.637
        1s-Delta, 31s gemittelt       0.880
        1s-Delta, 61s gemittelt       0.932

    Und es ist NEUE Information, keine Wiederholung dessen, was der Kopf
    schon hat: die Korrelation mit seiner Ausgabe liegt bei 0.625, und in
    genau den Sekunden, in denen er unsicher ist (0.2<p<0.8, im Median 248
    je Aufnahme), trennt das Merkmal noch mit 0.784.

    ⚠️ 61 Sekunden. Zuerst stand hier 31, aus Sorge um die Chunk-Grenzen
    (Go rechnet den Kopf chunkweise, ein Fenster am Rand sieht die
    Nachbarschaft nicht). Die Sorge war UNBEGRÜNDET — am 2026-08-07
    nachgemessen, Unruhe global gegen chunkweise bei sonst identischem
    Kopf und Decoder:

        Fenster   global   4 Chunks    Delta   betroffen
          31 s     0.936     0.936    -0.000      0/37
          61 s     0.931     0.930    -0.000      1/37
         121 s     0.934     0.934    +0.000      0/37

    Die Unruhe ist eine geglättete Größe, und der Dauer-Prior des Decoders
    bügelt lokale Störungen weg. Bei 61 s trennt das Merkmal deutlich
    besser (Rang-AUC 0.932 gegen 0.880).

    ⚠️ Die Breite ändert input_dim NICHT — Absicht, damit das nächtliche
    Head-to-Head funktionsfähig bleibt und die Änderung sofort beurteilen
    kann.

    ⚠️ Am Rand wird auf die TATSÄCHLICH vorhandenen Werte normiert, nicht
    mit Nullen aufgefüllt. Nullen zögen die Unruhe am Rand nach unten, was
    wie „Sendung" aussieht — eine gerichtete Verzerrung. Der Teilfenster-
    Mittelwert ist nur verrauschter. Die Go-Seite muss das exakt so machen.

    ⚠️ NICHT mit np.convolve(mode="same") gebaut. Das gibt die Laenge des
    LAENGEREN Arrays zurueck — bei einer Aufnahme mit 7 Sekunden und einem
    31er-Fenster also 31 Werte statt 7, und das Training stirbt beim
    column_stack (2026-08-06 genau so passiert). Das Praefixsummen-Fenster
    unten ist fuer jede Laenge korrekt und entspricht Zeile fuer Zeile der
    Schleife in nn.go confidenceMLPChunk.
    """
    T = X.shape[0]
    d = np.zeros(T, dtype=np.float32)
    if T > 1:
        d[1:] = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
    halb = fenster // 2
    cs = np.concatenate([[0.0], np.cumsum(d, dtype=np.float64)])
    i = np.arange(T)
    lo = np.maximum(i - halb, 0)
    hi = np.minimum(i + halb + 1, T)
    summe = cs[hi] - cs[lo]
    anzahl = (hi - lo).astype(np.float64)
    return (summe / np.maximum(anzahl, 1.0)).astype(np.float32).reshape(-1, 1)


# ---- DER ZUSATZBLOCK — eine einzige Definition -----------------------------
#
# Alles, was hinter die Backbone-Merkmale gehaengt wird, entsteht hier und
# NUR hier. Spaltenreihenfolge (= der Header-Vertrag, den nn.go liest):
#
#     kanal-one-hot | whisper | dp | dn | unruhe | minute-prior | maske
#
# ⚠️ Warum das eine eigene Funktion sein MUSS. Bis 2026-08-07 stand dieselbe
# Rechnung an SECHS Stellen: _augment_teacher_feats (Hygiene-Lehrer), der
# blockweise Produktions-Fit, _aug_test, _augment_channel_whisper_temporal
# (+ _augment_cwt_minuteprior), der All-Data-Refit und build_X in
# corpus-label-audit.py. Eine neue Spalte hiess: fuenf bis sechs Stellen
# aendern, und jede vergessene Stelle ist ein STILLER Bruch — der Kopf sieht
# dann in einem Pfad eine andere Eingabe als in den anderen, und das
# aeussert sich als "das Modell ist etwas schlechter geworden", nicht als
# Fehler. Genau so ist die Unruhe-Spalte am 2026-08-06 eingezogen (fuenf
# Edits), und dabei ist die Lehrer-Erkennung still ausgefallen.
#
# ⚠️ Die Reihenfolge ist ein PRAEFIX-Vertrag: neue Spalten kommen HINTEN
# dazu, damit die Spaltenlage aelterer Koepfe gueltig bleibt und eine
# Migration ein Header-Bump ist, kein Umsortieren.
#
# Die Go-Seite (internal/signals/nn.go) baut denselben Block beim Ableiten
# noch einmal — das laesst sich nicht zusammenlegen (Begruendung dort),
# deshalb bindet scripts/gen-augment-parity.py die beiden mit
# Goldwert-Vektoren aneinander, wie bei hsmm.
def zusatzspalten(X, uuid, slug, chan_idx, n_chan=None, *,
                  kanal=True, whisper=False, temporal=False, churn=False,
                  mp_col=None, maske=False):
    """Der Zusatzblock fuer EINE Aufnahme, Form (T, k).

    `X` sind die rohen Backbone-Merkmale der Aufnahme — zusammenhaengend und
    VOR jeder Hygiene-Maske, denn dp/dn/unruhe sind Nachbarschafts-Groessen:
    auf einer bereits geloecherten Matrix wuerde eine weggeworfene Sekunde
    einen Szenenwechsel vortaeuschen. Der Aufrufer maskiert danach Block und
    Basis gemeinsam.

    `chan_idx` ist die Zuordnung slug→Spalte des Kopfes, um den es geht — der
    Lehrer braucht SEINE eigene Karte, nicht die des laufenden Durchgangs.

    `mp_col` ist eine Funktion (uuid, T) → (T, 1); None heisst "keine
    Minute-Prior-Spalte".
    """
    T = X.shape[0]
    if n_chan is None:
        n_chan = len(chan_idx)
    teile = []
    if kanal:
        oh = np.zeros((T, n_chan), dtype=np.float32)
        if slug in chan_idx:
            oh[:, chan_idx[slug]] = 1.0
        teile.append(oh)
    if whisper:
        teile.append(_load_whisper_per_sec(uuid, T).reshape(-1, 1))
    if temporal:
        dp = np.zeros((T, 1), dtype=np.float32)
        dn = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            d = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
            dp[1:, 0] = d
            dn[:-1, 0] = d
        teile.append(dp)
        teile.append(dn)
        # ⚠️ Die Unruhe haengt am Temporal-Block. Ohne ihn gibt es sie nicht —
        # sonst bekaeme ein v4-Kopf (n_temporal=2) eine Spalte zu viel, fiele
        # durch die Breitenpruefung, und der betroffene Pfad liefe still ohne
        # ihn weiter.
        if churn:
            teile.append(_churn_col(X))
    if mp_col is not None:
        teile.append(mp_col(uuid, T))
    if maske:
        teile.append(np.full((T, 1),
                             1.0 if _whisper_present(uuid) else 0.0,
                             dtype=np.float32))
    if not teile:
        return np.zeros((T, 0), dtype=np.float32)
    return np.hstack(teile).astype(np.float32)


def mit_zusatz(X, uuid, slug, chan_idx, n_chan=None, **kw):
    """Basis + Zusatzblock — die uebliche Form fuer Pfade, die pro Aufnahme
    eine fertige Matrix wollen."""
    z = zusatzspalten(X, uuid, slug, chan_idx, n_chan, **kw)
    if z.shape[1] == 0:
        return X.astype(np.float32)
    return np.hstack([X, z]).astype(np.float32)


def _augment_teacher_feats(X, slug, chan_idx, uuid, wants_whisper,
                           wants_temporal=False, mp_col=None,
                           wants_churn=False, wants_mask=False):
    """Rebuild the channel-one-hot(+whisper)(+temporal) augmented feature
    matrix a v2/v3 MLP teacher was trained on, so it scores identically in
    the label-hygiene pass.

    Column order MUST mirror main()'s _aug_test EXACTLY: [base, channel-one-hot
    (by chan_idx), whisper?, temporal?]. `chan_idx` maps slug→column from the
    TEACHER's head.channel-map.json — the one-hot order is run-specific, so
    the teacher must be scored with ITS OWN map, not the new run's. A
    misaligned column here would feed the teacher garbage and drop correct
    frames, so the caller also keeps the per-recording drop-rate cap as a
    backstop.

    Die Rechnung selbst steht in zusatzspalten() — der Lehrer unterscheidet
    sich vom laufenden Durchgang NUR durch seine eigene Kanalkarte und seine
    eigene Minute-Prior-Beilage (die lebende Tabelle driftet naechtlich),
    nicht durch die Spaltenlage."""
    return mit_zusatz(X, uuid, slug, chan_idx,
                      whisper=wants_whisper, temporal=wants_temporal,
                      churn=wants_churn, mp_col=mp_col, maske=wants_mask)



# ---- GOLDEN-BODEN (Sperrklinke gegen langsamen Drift) ----------------------
#
# Das paarweise Gate ist ein RELATIVER Test: Kandidat gegen den AKTUELLEN
# Champion, abgelehnt wird nur, was BELASTBAR schlechter ist. Ein bisschen
# schlechter deployt — und ist ab dann der Massstab, gegen den die naechste
# Nacht vergleicht. Das ist eine Ratsche ohne Sperrklinke: nach oben zieht
# nichts zurueck.
#
# ⚠️ Genau so gemessen am 2026-08-03. Vier Deploys in Folge, jeder einzelne
# mit "median Δ +0.000, keine belastbare Regression":
#
#     Golden-Median  0.915 → 0.904 → 0.901 → 0.896
#     Golden-Mean    0.886 → 0.870 → 0.850 → 0.846
#
# Kein Schritt war fuer sich ablehnungswuerdig, der Korpus lag stabil bei
# 270–277 Aufnahmen. Der Golden-Wert wurde die ganze Zeit berechnet,
# protokolliert und mit "compare night-to-night, this is the real trend"
# beschriftet — aber nirgends ausgewertet. Reine Beobachtung.
#
# Der Boden ist ABSOLUT (gegen den besten je deployten Wert), nicht relativ —
# sonst waere er dieselbe Ratsche eine Ebene hoeher.
#
# Kein Deadlock: ein Kandidat, der den Champion SCHLAEGT, kommt immer durch,
# auch wenn er noch unter dem Bestwert liegt. Nur so klettert der Stand nach
# einem Absacker wieder hoch. Blockiert wird ausschliesslich, wer unter dem
# Bestwert liegt UND den Champion nicht verbessert.
def golden_bestwert(trend_pfad, set_hash):
    """Bester je DEPLOYTER Golden-Median fuer genau diesen Golden-Satz.

    ⚠️ Nur Eintraege mit demselben set_hash. Ueber eine Satz-Aenderung hinweg
    zu vergleichen waere genau der Fehler, den der Hash verhindern soll — und
    Eintraege mit `missing` sind nicht komposition-konstant, zaehlen also
    ebenfalls nicht.

    ⚠️ Und nur Eintraege mit DEMSELBEN DECODER. Bis 2026-08-05 wurde der
    Golden-Wert mit `form` gemessen, obwohl die Produktion seit 07-29 `hsmm`
    faehrt (s. EVAL_DECODER). Das ist derselbe Fehler eine Ebene tiefer: eine
    stille Neudefinition dessen, was die Zahl bedeutet. hsmm liegt auf diesem
    Satz systematisch HOEHER als form (gemessen +0.14 Mittel), ein
    form-Bestwert waere als Boden also wirkungslos — die Sperrklinke haette
    genau das nicht mehr getan, wofuer es sie gibt.

    Folge: nach der Umstellung ist der Boden zunaechst leer und baut sich ab
    der ersten hsmm-Nacht neu auf. Das ist gewollt; ein geerbter Boden waere
    schlimmer als keiner.
    """
    best, best_ts = None, None
    if not trend_pfad or not Path(trend_pfad).exists():
        return None, None
    jetzt = " ".join(EVAL_DECODER) or "form"
    for ln in Path(trend_pfad).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if not e.get("deployed") or e.get("missing"):
            continue
        if e.get("set_hash") != set_hash:
            continue
        if (e.get("decoder") or "form") != jetzt:
            continue
        v = e.get("golden_median")
        if v is not None and (best is None or float(v) > best):
            best, best_ts = float(v), e.get("ts")
    return best, best_ts


def golden_stau(trend_pfad, set_hash):
    """Wie viele der juengsten Laeufe in Folge NICHT deployt haben.

    Dient nur der Sichtbarkeit: ein Gate, das dauerhaft blockt, sieht in der
    Logzeile genauso aus wie eines, das einmal blockt.
    """
    if not trend_pfad or not Path(trend_pfad).exists():
        return 0
    eintraege = []
    for ln in Path(trend_pfad).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("set_hash") != set_hash:
            continue
        # Gleiche Begruendung wie in golden_bestwert: ein Stau, der aus
        # form-Naechten stammt, sagt ueber die hsmm-Reihe nichts.
        if (e.get("decoder") or "form") != (" ".join(EVAL_DECODER) or "form"):
            continue
        eintraege.append(e)
    n = 0
    for e in reversed(eintraege):
        if e.get("deployed"):
            break
        n += 1
    return n


def golden_boden(deploy, reason, *, golden_floor, train_archive,
                 cand_pr, champ_pr, melde=print):
    if not deploy or golden_floor <= 0:
        return deploy, reason
    try:
        gpath = (Path(train_archive) / "golden-eval-set.json"
                 if train_archive else None)
        if not (gpath and gpath.exists()):
            return deploy, reason
        meta = json.loads(gpath.read_text())
        golden = set(meta.get("uuids", []))
        if not golden:
            return deploy, reason

        # ⚠️ Komposition-Konstanz ist die ganze Geschaeftsgrundlage dieser
        # Zahl. Fehlt auch nur eine gepinnte Aufnahme, ist der Median mit
        # frueheren Naechten NICHT vergleichbar — dann darf er auch nicht
        # gaten. Lieber nicht pruefen als falsch pruefen.
        fehlend = sorted(golden - set(cand_pr))
        if fehlend:
            melde(f"  Golden-Boden: uebersprungen — {len(fehlend)} gepinnte "
                  f"Aufnahme(n) fehlen heute, Median nicht komposition-konstant")
            return deploy, reason

        g_cand = float(np.median([cand_pr[u] for u in golden]))
        # Champion im GLEICHEN Lauf auf denselben Aufnahmen — ehrlicher als
        # der Wert aus der Historie, weil identische Bedingungen.
        gemeinsam = [u for u in golden if u in champ_pr]
        g_champ = (float(np.median([champ_pr[u] for u in gemeinsam]))
                   if len(gemeinsam) == len(golden) else None)

        best, best_ts = golden_bestwert(
            Path(train_archive) / "golden-trend.jsonl" if train_archive else None,
            meta.get("set_hash"))
        if best is None:
            # ⚠️ Diese Zeile erscheint auch in der ERSTEN Nacht nach einem
            # Decoder-Wechsel, nicht nur bei einem neuen Golden-Satz. Ohne den
            # Hinweis liest sie sich wie ein Satz-Wechsel, und der Boden sieht
            # aus, als waere er verloren gegangen — er wird nur neu aufgebaut.
            melde(f"  Golden-Boden: noch kein Bestwert fuer diesen Satz "
                  f"(set_hash {str(meta.get('set_hash'))[:8]}, decoder "
                  f"{' '.join(EVAL_DECODER) or 'form'}) — heutiger Wert "
                  f"{g_cand:.3f} wird der erste")
            return deploy, reason

        abstand = g_cand - best
        schlaegt_champ = g_champ is not None and g_cand > g_champ
        if abstand < -golden_floor:
            if schlaegt_champ:
                melde(f"  Golden-Boden: {g_cand:.3f} liegt zwar {-abstand:.3f} "
                      f"unter dem Bestwert {best:.3f}, schlaegt aber den "
                      f"Champion ({g_champ:.3f}) — Aufstieg, deploy")
                return deploy, reason
            champ_txt = f", Champion {g_champ:.3f}" if g_champ is not None else ""
            # ⚠️ Ein Boden, der Nacht fuer Nacht blockt, ist kein Schutz mehr,
            # sondern ein Stillstand: der Champion friert ein und keine neuen
            # Daten kommen mehr in Produktion. Das ist richtiger als still
            # abzurutschen, darf aber nicht selbst still passieren.
            _stau = golden_stau(
                Path(train_archive) / "golden-trend.jsonl" if train_archive else None,
                meta.get("set_hash"))
            if _stau >= 3:
                melde(f"  ⚠ Golden-Boden blockt die {_stau + 1}. Nacht in Folge — "
                      f"der Champion steht seit {_stau + 1} Laeufen. Entweder ist "
                      f"der Bestwert {best:.3f} nicht mehr erreichbar (dann "
                      f"gehoert er ueberprueft), oder es gibt ein echtes "
                      f"Datenproblem. NICHT einfach --golden-floor hochsetzen.")
            return False, (reason + f" — ABER GOLDEN-BODEN: {g_cand:.3f} liegt "
                           f"{-abstand:.3f} unter dem besten je deployten Wert "
                           f"{best:.3f} ({best_ts}){champ_txt} und verbessert "
                           f"den Champion nicht — langsamer Drift, Champion bleibt")
        melde(f"  Golden-Boden: {g_cand:.3f} vs Bestwert {best:.3f} "
              f"({abstand:+.3f}) — passiert")
    except Exception as e:
        # Ein kaputter Boden darf keinen Lauf verhindern; er darf nur nicht
        # stillschweigend nichts tun.
        melde(f"  Golden-Boden: nicht auswertbar ({e}) — Gate unveraendert")
    return deploy, reason


def main():
    ap = argparse.ArgumentParser()
    # Defaults are the post-SMB-migration locations: backbone +
    # logo templates live in the Mac-side daemon cache (refreshed
    # via gateway HTTP); output + hls-root are the wrapper's tmp
    # paths (snapshot fetched + uploaded via HTTP). Standalone
    # invocations work as long as tv-thumbs-daemon has populated
    # the caches and the snapshot dir exists.
    ap.add_argument("--backbone", default=os.path.expanduser(
        "~/.cache/tv-detect-daemon/backbone.onnx"))
    ap.add_argument("--output",
                    default=os.path.expanduser(
                        "~/.cache/tv-train-head-out/head.bin"))
    ap.add_argument("--hls-root", default="/tmp/tv-train-snapshot")
    ap.add_argument("--feature-cache", default=os.path.expanduser(
        "~/.cache/tvd-features"))
    ap.add_argument("--train-archive", default=os.path.expanduser(
        "~/.cache/tvd-train-archive"),
        help="deletion-safe corpus archive: freeze each trustworthy-labelled "
             "recording's label (+ a pointer to its cached features) so it "
             "keeps training the head after its .ts is deleted/dedup'd. Empty "
             "string disables.")
    ap.add_argument("--fps-extract", type=float, default=1.0)
    ap.add_argument("--reextract-logo-nan-pct", type=float, default=10.0,
                    help="re-extract a cached .npy if its logo column has "
                         "this percent or more NaN-sentinels. Catches stale "
                         "features from older buggy extractors. Default 10. "
                         "Set to 100 to disable (= trust any cached file).")
    ap.add_argument("--prefer", choices=["user", "auto", "any"], default="any",
                    help="user = only ads_user.json; auto = only ads.json; "
                         "any = user where present, else auto")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel feature-extraction workers (each loads "
                         "its own ONNX session, ~100 MB resident)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of recordings (deterministically hashed "
                         "by uuid) held out for evaluation. 0 = no split, "
                         "trains on everything (no validation reported)")
    ap.add_argument("--final-on-all", action="store_true", default=True,
                    help="after train/test eval, refit the head on ALL "
                         "recordings before writing head.bin (standard "
                         "practice — validation tells you it works, then "
                         "you ship the full-data model). Disable with "
                         "--no-final-on-all.")
    ap.add_argument("--no-final-on-all", dest="final_on_all",
                    action="store_false")
    ap.add_argument("--max-ad-rate", type=float, default=0.50,
                    help="drop recordings whose final (smart-merged) "
                         "label set marks more than this fraction of "
                         "frames as ad. Catches broken-template runs "
                         "where a whole recording was wrongly tagged "
                         "100%% ad. Real content never exceeds ~40%%.")
    ap.add_argument("--user-weight", type=float, default=2.0,
                    help="sample-weight multiplier applied to frames "
                         "from recordings that have an ads_user.json. "
                         "User-confirmed labels are higher-quality "
                         "than auto-only — train pulls toward them. "
                         "1.0 = no preference.")
    ap.add_argument("--surface-uncertain", type=int, default=0,
                    help="for each recording, list the N timestamps "
                         "where the trained head is least confident "
                         "(|p - 0.5| smallest). Surfaces the highest-"
                         "value frames for manual labelling. Output "
                         "goes to <output>.uncertain.txt next to head.bin.")
    ap.add_argument("--emit-confusion", action="store_true",
                    help="write a per-recording confusion analysis to "
                         "<output>.confusion.txt: classifies error "
                         "frames as boundary (within ±10s of a GT "
                         "block edge — typical IoU drag) vs intra "
                         "(content-confusion deep in/out of a block). "
                         "Also: GT blocks fully missed by predictions, "
                         "and pred blocks with no GT overlap. Use to "
                         "decide whether to fix labels, training data, "
                         "or model architecture.")
    ap.add_argument("--with-logo", action="store_true",
                    help="extract per-frame logo confidence as a 1281st "
                         "input feature, train a WITH-LOGO head (1281 "
                         "weights + bias = 5128 B). Forces a cache "
                         "rebuild (key bumped) but nothing else changes "
                         "downstream — Go-side reloadHead auto-detects "
                         "the format by file size. Recommended once "
                         "the slug map covers all channels.")
    ap.add_argument("--with-audio", action="store_true",
                    help="append per-second audio RMS loudness as an "
                         "additional input feature. EXPERIMENTAL — Go "
                         "inference doesn't yet read audio-format heads, "
                         "so use --output to a non-default path while "
                         "evaluating; production head.bin must stay in "
                         "a Go-loadable format until the inference path "
                         "is extended.")
    ap.add_argument("--with-self-training", action="store_true",
                    help="Phase A — validate self-training pseudo-label "
                         "viability: for each test recording predict, then "
                         "filter to frames where the head is highly confident "
                         "(p>0.97 or p<0.03) AND the wall-clock minute-prior "
                         "agrees (same side of 0.5). Report accuracy of those "
                         "candidate pseudo-labels vs user truth. Requires "
                         "--with-minute-prior. Phase B (write pseudo-labels "
                         "to disk + mix into training) is a separate flag.")
    ap.add_argument("--self-train-conf", type=float, default=0.97,
                    help="confidence threshold for self-training pseudo-labels "
                         "(symmetric: p>X for ad, p<1-X for show). Default 0.97 "
                         "= keep only the most confident 6%% of frames roughly.")
    ap.add_argument("--write-pseudo-labels", action="store_true",
                    help="Phase B — after self-training validation, write "
                         "pseudo_labels.json into every unreviewed recording's "
                         "rec_dir. Next training run picks them up as low-"
                         "weight (0.3×) training data. Champion-Challenger gate "
                         "catches regressions. Implies --with-self-training. "
                         "Files are auto-deleted on review (when ads_user.json "
                         "appears) — pseudo never overrides user labels.")
    ap.add_argument("--pseudo-weight", type=float, default=0.3,
                    help="sample-weight for pseudo-labelled frames. 0.3× = "
                         "60%% of an auto-only frame (1.0×), 15%% of a user-"
                         "confirmed frame (2.0×). Conservative — pseudo labels "
                         "are very accurate (~99%%) but uniform low weight "
                         "limits damage from any rare wrong ones.")
    ap.add_argument("--co-train", action="store_true",
                    help="train two extra heads alongside the production "
                         "one: head_logo (visual: backbone+logo+channel) "
                         "and head_audio (acoustic: rms+channel). Reports "
                         "agreement statistics on the test set — frames "
                         "where both confident-and-agree are pseudo-label "
                         "candidates (Phase 2, not auto-applied yet); "
                         "where they disagree are active-learning targets. "
                         "Forces --with-logo --with-audio --with-channel.")
    ap.add_argument("--with-minute-prior", action="store_true",
                    help="empirically build per-channel P(ad | minute_of_hour) "
                         "from the labelled recordings and use it as an "
                         "active-learning diagnostic. Frames where the "
                         "trained head's prediction strongly disagrees "
                         "with the wall-clock prior become high-value "
                         "labelling targets — surfaces frames the prior "
                         "knows but the head doesn't (or vice versa). "
                         "Prior cached as <hls-root>/.minute_prior_by_channel.json.")
    ap.add_argument("--with-bumpers", action="store_true",
                    help="use ffmpeg blackdetect+silencedetect (bumper "
                         "candidates) as a teacher signal: ads.json "
                         "boundaries that align with a bumper (±2 s) "
                         "get a sample-weight boost on their ±2 s frame "
                         "window. Independent of the head's features; "
                         "purely shifts how much the head trusts each "
                         "boundary. Bumpers cached as <rec>/bumpers.json.")
    ap.add_argument("--bumper-boost", type=float, default=1.4,
                    help="sample-weight multiplier for frames within ±2 s "
                         "of an ads.json boundary that has a bumper "
                         "within ±2 s. Off when --with-bumpers is unset. "
                         "1.4× sits between confirmed_show (1.2×) and "
                         "confirmed_ad_skip (1.5×).")
    ap.add_argument("--bumper-detect", default=str(
        Path(__file__).parent / "bumper-detect.py"),
        help="path to bumper-detect.py helper (default: same dir as "
             "this script).")
    ap.add_argument("--with-yamnet", action="store_true",
                    help="append per-second YAMNet 1024-dim audio "
                         "embedding (Google's frozen AudioSet model). "
                         "Captures music vs speech vs noise patterns "
                         "that raw RMS misses. Adds 1024 input dims, "
                         "head grows to 2305 (with-logo) or 2304 "
                         "(audio-only). EXPERIMENTAL: Go inference "
                         "can't read this format — use --output /tmp/.")
    ap.add_argument("--with-uniformity", action="store_true",
                    help="append per-second luma std-dev (ffmpeg "
                         "signalstats YDEV, mean per second, normalised "
                         "to 0..1). Comskip's uniformity signal: ads "
                         "tend to have uniform color blocks (text "
                         "overlays, brand cards), shows are textured. "
                         "1 extra input dim. EXPERIMENTAL: Go-side head "
                         "loader needs new size before deploying.")
    ap.add_argument("--head-arch",
                    choices=["logreg", "mlp32-channel",
                             "mlp32-channel-whisper",
                             "mlp32-channel-whisper-temporal",
                             "mlp32-channel-whisper-temporal-mp",
                             "mlp32-channel-whisper-temporal-mp-wm"],
                    default="logreg",
                    help="head architecture to deploy. "
                         "'logreg' = legacy linear head (size-detected). "
                         "'mlp32-channel' = MLP1 v1 = 1290→32→1 with N-dim "
                         "channel one-hot. Production cutover 2026-05-03; "
                         "shadow IoU +0.22 vs LogReg. "
                         "'mlp32-channel-whisper' = MLP2 v2 = adds a 6th "
                         "input column carrying per-second whisper-prob "
                         "(= 1291→32→1 with one extra slot). Shadow IoU "
                         "+0.27 vs LogReg / +0.075 vs mlp32-channel; the "
                         "MLP non-linearity absorbs the whisper signal "
                         "without the L2-balance fragility a LogReg "
                         "column-add would suffer. Requires Mac daemon "
                         "to pass --nn-whisper-json to tv-detect at "
                         "inference (deployed in tandem). The feature-dim "
                         "change trips the deploy-decision's 'feature "
                         "dim changed → reset baseline' branch; a one-time "
                         "forced deploy happens + history.json baseline "
                         "restarts. "
                         "'mlp32-channel-whisper-temporal' = MLP3 v3 = adds "
                         "2 more input columns (L2 distance to the previous/"
                         "next frame's [backbone+logo+audio] vector) APPENDED "
                         "LAST after whisper. Migration decided 2026-07-12 "
                         "after a 7-night --shadow-eval series (6/7 nights "
                         "positive, median Δ ≈ +0.02..+0.07 vs "
                         "mlp32-channel-whisper). Requires the Go inference "
                         "side (internal/signals/nn.go) to compute the same "
                         "deltas at serve time — deployed in tandem, see "
                         "nn.go's v3 header support. "
                         "'mlp32-channel-whisper-temporal-mp' = MLP4 v4 = adds "
                         "1 more column: P(ad | minute-of-hour) from the "
                         "per-channel wall-clock prior histogram (requires "
                         "--with-minute-prior so the table exists). Migration "
                         "2026-07-22 after a 3-night shadow series "
                         "(+0.021/+0.021/+0.016 vs the cwt replica). Ships a "
                         "head.minute-prior.json sidecar; the Go side "
                         "(nn.go v4) + the daemon's --start-ts flag look up "
                         "the same prior at serve time — deployed in tandem.")
    ap.add_argument("--seed-sweep", type=int, default=0, metavar="N",
                    help="N-mal dieselbe Architektur auf denselben Daten "
                         "fitten, nur mit anderem Init-Seed, und die "
                         "Streuung von Golden- und Test-Median melden. "
                         "Trennt Fit-Zufall von Korpus-Drift — die Zahl, "
                         "an der sich jede Migrations-Entscheidung messen "
                         "lassen muss. Braucht --shadow-eval (nutzt dessen "
                         "Fit/Eval-Pfad); nicht im Nightly.")
    ap.add_argument("--shadow-eval", action="store_true",
                    help="after the production LogReg fit + eval, also "
                         "train + evaluate three architectural variants "
                         "(MLP-32, MLP-32 + 24-channel one-hot, MLP-32 + "
                         "temporal L2 deltas) on the same train/test split "
                         "and print a comparison table. Pure measurement "
                         "— shadow heads are NOT written or deployed; the "
                         "LogReg path proceeds normally. Use this to "
                         "decide whether a structural head change is "
                         "worth a full migration.")
    ap.add_argument("--with-channel", action="store_true",
                    help="append a 6-dim sparse one-hot of the channel "
                         "(kabel-eins/prosieben/rtl/sat-1/sixx/vox) as "
                         "extra input features so the head can learn "
                         "channel-specific bias. Channel order is "
                         "fixed (alphabetical) — never re-order or "
                         "insert, only append, or every previously "
                         "trained head breaks. Combinable with "
                         "--with-logo for a 1287-feature head (5152 B).")
    ap.add_argument("--logo-dir", default=os.path.expanduser(
                        "~/.cache/tv-detect-daemon/logos"),
                    help="directory of channel-keyed cached logo "
                         "templates (used by --with-logo extraction). "
                         "tv-thumbs-daemon keeps this dir fresh from "
                         "the gateway's /api/internal/detect-logo/<slug> "
                         "endpoint.")
    ap.add_argument("--tv-detect", default=os.path.expanduser(
                        "~/.local/bin/tv-detect"),
                    help="path to tv-detect binary, used by --with-logo "
                         "to compute per-frame logo confidence via "
                         "--emit-logo-csv subprocess.")
    ap.add_argument("--rollback-iou-drop", type=float, default=0.05,
                    help="champion-challenger: if the new model's "
                         "test Block-IoU drops by more than this vs "
                         "the previous successful run, REJECT the new "
                         "head.bin and keep the previous one. "
                         "Set to 1.0 to disable (always deploy).")
    ap.add_argument("--reset-baseline", action="store_true",
                    help="one-time: skip the champion-challenger IoU floor and "
                         "deploy if the head clears --reset-baseline-floor. Use "
                         "after the test-set POPULATION fundamentally changed "
                         "(e.g. the corpus-fix that recovered disk-pruned recs) "
                         "so the historical floor — anchored to the old, "
                         "non-representative test set — is no longer comparable. "
                         "Future runs then compare against this new baseline.")
    ap.add_argument("--reset-baseline-floor", type=float, default=0.5,
                    help="absolute median-IoU sanity floor for --reset-baseline "
                         "(default 0.50) so a reset never re-anchors on garbage.")
    ap.add_argument("--rollback-acc-drop", type=float, default=0.03,
                    help="same as --rollback-iou-drop but for per-frame "
                         "test accuracy. Either trigger fires rejection.")
    ap.add_argument("--reviewed-regression-veto", type=float, default=0.30,
                    help="paired head-to-head guard: if ANY user-REVIEWED "
                         "(ground-truth) test rec drops by more than this vs "
                         "the champion AND falls below --reviewed-regression-floor "
                         "in absolute terms, block the deploy — even when the "
                         "median delta is a tie. The median gate deliberately "
                         "ignores the asymmetric negative tail; this catches a "
                         "silent big loss (e.g. −0.50 IoU) on a rec we KNOW is "
                         "labelled correctly. Only reviewed recs count (auto-labels "
                         "can themselves be wrong, so a drop there isn't trusted). "
                         "Set to 1.0 to disable.")
    ap.add_argument("--reviewed-regression-floor", type=float, default=0.50,
                    help="absolute-IoU floor for the reviewed-regression veto: a "
                         "big drop only vetoes if the candidate ALSO lands below "
                         "this (i.e. the rec became genuinely bad, not just "
                         "relatively worse). Stops boundary-jitter on already-good "
                         "recs (0.95→0.60) from re-creating the 2026-07-02..06 "
                         "rejection deadlock, while still catching real breakage "
                         "(0.35→0.00, 0.75→0.20).")
    ap.add_argument("--golden-floor", type=float, default=0.010,
                    help="absolute Sperrklinke gegen langsamen Drift. Das "
                         "paarweise Gate vergleicht immer gegen den AKTUELLEN "
                         "Champion und laesst alles durch, was nicht belastbar "
                         "schlechter ist — der leicht schlechtere Kandidat wird "
                         "damit selbst zum neuen Massstab. Gemessen 2026-08-03: "
                         "vier Deploys in Folge, jeder fuer sich 'keine "
                         "belastbare Regression', zusammen Golden-Median 0.915 "
                         "-> 0.896. Diese Schwelle blockt einen Kandidaten, der "
                         "mehr als so viel unter dem BESTEN je deployten "
                         "Golden-Median liegt UND den Champion nicht schlaegt. "
                         "0 = aus.")
    ap.add_argument("--stability-window", type=int, default=10,
                    help="how many recent per-rec-iou.jsonl runs the veto's "
                         "bistability check looks back over. A rec whose own IoU "
                         "spans more than --reviewed-regression-veto across this "
                         "window is flip-flopping between block-boundary "
                         "attractors, so a single-run drop on it says nothing "
                         "about the model and must not veto a deploy.")
    ap.add_argument("--reviewed-regression-severe", type=float, default=0.45,
                    help="a SINGLE reviewed rec dropping by at least this much (and "
                         "below the floor) is enough to veto on its own. Below this, "
                         "it takes a PATTERN (≥2 reviewed regressions) to block — so "
                         "one moderate drop (possibly a degenerate/stale-label rec) "
                         "is logged loudly but still deploys, avoiding a single-rec "
                         "deadlock.")
    ap.add_argument("--both-cold-floor", type=float, default=0.40,
                    help="both-heads-cold report threshold: test recs where BOTH "
                         "the candidate AND the champion score below this IoU are "
                         "systematic blind spots the paired gate can never see "
                         "(champion equally bad → delta ≈ 0). Written to "
                         "<output>.both-cold.jsonl for the review/feature queue.")
    ap.add_argument("--ablate-minute-prior", action="store_true",
                    help="diagnostic: after the test-set eval, re-score the "
                         "deployed head with the minute-of-hour prior column "
                         "neutralised (set to its per-channel mean) and print the "
                         "IoU delta. Confirms the MLP4 minute column actually "
                         "earns its complexity. Does not affect the deploy.")
    ap.add_argument("--hygiene-disagree-conf", type=float, default=0.9,
                    help="if the existing head.bin (used as 'teacher') "
                         "predicts the OPPOSITE label with confidence "
                         "> this, drop that frame from training. "
                         "0 = off. The drop-rate per recording is "
                         "capped at 30 %% so a busted teacher can't "
                         "wipe out a recording's labels.")
    ap.add_argument("--hygiene-max-drop-rate", type=float, default=0.30,
                    help="upper bound on what fraction of a recording's "
                         "frames the hygiene filter is allowed to drop. "
                         "Above this, the recording is left untouched "
                         "(teacher more likely wrong than labels).")
    ap.add_argument("--daemon-cache",
                    default=str(Path.home() / ".cache/tv-detect-daemon/source"),
                    help="UUID-keyed .ts cache populated by tv-thumbs-daemon "
                         "(detect + prefetch). Checked before SMB fallback.")
    args = ap.parse_args()

    # --co-train forces the feature flags it needs (otherwise the
    # column slicing below points at non-existent columns).
    if args.co_train:
        args.with_logo = True
        args.with_audio = True
        args.with_channel = True

    # EIN Zeitstempel fuer den ganzen Lauf. Er benennt das Kopf-Archiv, die
    # Zeile in golden-trend.jsonl und die Zeilen in shadow-trend.jsonl — nur
    # so laesst sich spaeter "welche Schattenvariante gehoerte zu welchem
    # deployten Kopf" ueber die Dateien hinweg verbinden. Vor 2026-08-09
    # entstand er erst im Deploy-Block, also nach den Schattenlaeufen.
    ts = time.strftime("%Y%m%dT%H%M%S")
    cache_dir = Path(args.feature_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.train_archive).expanduser() if args.train_archive else None
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
    sess = build_onnx_session(args.backbone)
    print(f"providers: {sess.get_providers()}")

    # uuid → channel slug map. Used by --with-logo for logo template
    # lookup AND by --with-minute-prior for per-channel histograms.
    # uuid_start carries unix start_time (wall-clock) needed by the
    # minute-prior path to map frame offsets → minute-of-hour buckets.
    # Failure is non-fatal: missing entries fall back to neutral defaults.
    uuid_slug = {}
    uuid_start = {}
    uuid_cohort = {}  # uuid → (title, channel_name) for cohort-trust gate
    if args.with_logo or args.with_minute_prior:
        try:
            import urllib.request, ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
            chans = json.loads(urllib.request.urlopen(
                "https://raspberrypi5lan:8443/api/channels",
                timeout=10, context=ctx).read())
            chname_to_slug = {c["name"]: c["slug"]
                              for c in chans.get("channels", [])
                              if c.get("name") and c.get("slug")}
            entries = json.loads(urllib.request.urlopen(
                "http://raspberrypi5lan:9983/api/dvr/entry/grid?limit=2000",
                timeout=10).read())
            for e in entries.get("entries", []):
                u = e.get("uuid"); cn = e.get("channelname", "")
                if u and cn in chname_to_slug:
                    uuid_slug[u] = chname_to_slug[cn]
                if u and e.get("start_real"):
                    uuid_start[u] = int(e["start_real"])
                elif u and e.get("start"):
                    uuid_start[u] = int(e["start"])
                if u:
                    uuid_cohort[u] = (
                        (e.get("disp_title") or "").strip(),
                        cn or "")
            print(f"slug map: {len(uuid_slug)} uuid→slug entries from gateway")
        except Exception as ex:
            print(f"slug map: gateway unreachable ({ex}) — "
                  f"all logo confs will fall back to 0.5", flush=True)

    # Cohort-trust scan for auto_confirmed_no_ads gating.
    # Auto-confirm pipeline currently produces false-positives for
    # shows on under-bumper-templated channels (e.g. Nick SpongeBob:
    # 91 of 118 auto-confirmed "no ads" but 11 user-reviewed have
    # ad blocks → the 91 are likely also wrong, just never reviewed).
    # If we feed those 91 into training as "all-show" we poison the
    # model. Build a set of (title, channel) cohorts that have at
    # least one user-confirmed-with-ads recording — for any auto-
    # confirm-no-ads in those cohorts, fall back to bootstrap rather
    # than trusting the empty signal.
    suspect_cohorts = set()
    for rec_dir in Path(args.hls_root).glob("_rec_*"):
        u = rec_dir.name[5:]
        cohort = uuid_cohort.get(u)
        if not cohort or not cohort[0]:
            continue
        user_p = rec_dir / "ads_user.json"
        if not user_p.is_file(): continue
        try:
            d = json.loads(user_p.read_text())
            if not isinstance(d, dict): continue
            ads = d.get("ads") or []
            if ads:
                suspect_cohorts.add(cohort)
        except Exception:
            pass
    # Also recover trust anchors from the deletion-safe archive. DVR
    # series-retention (5 newest + >14d, dvr_series_retention) prunes old
    # episodes on a schedule unrelated to training — a user-reviewed episode
    # that was the SOLE cohort-trust anchor can age out of the _rec_* walk
    # above on any given night, silently flipping its whole cohort from
    # trusted to bootstrap and shifting that night's training distribution
    # (root-caused 2026-07-09: "Bella Italia" RTLZWEI lost its anchor this
    # way, cohort-trust 45->44, concentrated regression on the shared head).
    # The frozen archive entry still carries which/ads, so recheck it here
    # instead of only the live directory.
    if archive_dir is not None:
        for npz_path in archive_dir.glob("*.npz"):
            u = npz_path.stem
            try:
                z = np.load(npz_path, allow_pickle=False)
                a_meta = json.loads(str(z["meta"]))
            except Exception:
                continue
            # Frozen cohort (set at write-time, see the archive-write block)
            # survives even after the DVR entry itself is deleted by series-
            # retention. Fall back to a live lookup for older archive entries
            # written before this field existed.
            cohort = tuple(a_meta["cohort"]) if a_meta.get("cohort") else uuid_cohort.get(u)
            if not cohort or not cohort[0] or cohort in suspect_cohorts:
                continue
            if a_meta.get("which") in ("user", "merged") and a_meta.get("ads"):
                suspect_cohorts.add(cohort)
    print(f"cohort-trust: {len(suspect_cohorts)} (title,channel) cohorts "
          f"have ≥1 user-confirmed-with-ads — auto-confirm-empty in those "
          f"cohorts will be treated as bootstrap (not labels)")
    if archive_dir is not None:
        try:
            snap_path = archive_dir / "cohort-trust.jsonl"
            with open(snap_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y%m%dT%H%M%S"),
                    "n": len(suspect_cohorts),
                    "cohorts": sorted(f"{t} / {c}" for t, c in suspect_cohorts),
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"cohort-trust: snapshot write failed: {e}")

    # Pass 1 — discover labelled recordings, separate cached from
    # uncached. Cached ones we just load synchronously; uncached
    # ones go to the worker pool.
    cached, todo = [], []  # cached: (rec_info, cache_path); todo: (rec_info, src, cache_path)
    corpus_no_ts = 0       # recovered from feature cache because the .ts was dedup'd
    resurrect_seen = []    # (uuid, n) — auto blocks kept alongside a user list
    resurrect_vetoed = []  # (uuid, n) — of those, dropped by confirmed_show
    for rec_dir in sorted(Path(args.hls_root).glob("_rec_*")):
        uuid = rec_dir.name[5:]
        user = rec_dir / "ads_user.json"
        auto = rec_dir / "ads.json"

        # Read both. ads_user.json may be the legacy list-of-pairs
        # format or the dict {"ads":[…], "deleted":[…]} written since
        # the gateway's smart-merge rewrite. Either way we want the
        # merged view (auto-detected ∪ user-refined, minus user-
        # deleted false positives) as the training label set —
        # otherwise refining one boundary in a 3-block recording
        # would drop the other 2 blocks from training.
        def _load(path):
            try: return json.loads(path.read_text())
            except Exception: return None

        user_raw = _load(user) if user.exists() else None
        confirmed_show = []  # explicit "this frame is show" labels from /mark-reviewed
        confirmed_ad_skips = []  # implicit "user pressed skip here = ad" labels
        # auto-confirmed: gateway tagged this recording's ads.json as
        # "high-confidence 0 ad blocks" via the auto-confirm pipeline,
        # then snapshotted that into ads_user.json with empty ads. The
        # informational signal is "no ads in this recording" — every
        # frame is show. Pre-2026-05-14 the bootstrap branch dropped
        # these from training (treated as "no labels"), losing ~150
        # recordings of negative-class signal per cron run.
        auto_confirmed_no_ads = False
        if isinstance(user_raw, list):
            user_ads, deleted = user_raw, []
        elif isinstance(user_raw, dict):
            user_ads = user_raw.get("ads") or []
            deleted = user_raw.get("deleted") or []
            confirmed_show = [float(x) for x in
                              user_raw.get("confirmed_show", []) or []]
            confirmed_ad_skips = [float(x) for x in
                                   user_raw.get("confirmed_ad_skips", []) or []]
            if (user_raw.get("auto_confirmed_at")
                    and (user_raw.get("auto_confirm_n_blocks") or 0) == 0
                    and (user_raw.get("auto_confirm_score") or 0) >= 0.9):
                auto_confirmed_no_ads = True
        else:
            user_ads, deleted = [], []

        auto_ads = _load(auto) if auto.exists() else None
        if not isinstance(auto_ads, list):
            auto_ads = []

        if args.prefer == "user":
            ads = user_ads
            which = "user" if user.exists() else ""
        elif args.prefer == "auto":
            ads = auto_ads
            which = "auto" if auto.exists() else ""
        else:  # "any" — smart-merge auto + user
            def _overlaps(a, b): return a[0] < b[1] and b[0] < a[1]
            surviving = [a for a in auto_ads
                         if not any(_overlaps(a, x) for x in user_ads)
                         and not any(_overlaps(a, d) for d in deleted)]
            # An auto block the reviewer removed only stays out if the removal
            # was recorded in "deleted"; a client that just omits the block lets
            # it survive here and become a label the human already rejected.
            # tv-recorder derives the missing entry since 2026-07-26, but an
            # explicit confirmed_show inside a surviving block is proof on its
            # own — dvr-rtl-1781909700 scored IoU 0.000 on both heads because
            # the model correctly followed the review and the label did not.
            if confirmed_show and surviving:
                vetoed = [a for a in surviving
                          if any(a[0] <= t <= a[1] for t in confirmed_show)]
                if vetoed:
                    surviving = [a for a in surviving if a not in vetoed]
                    resurrect_vetoed.append((uuid, len(vetoed)))
            if surviving and user_raw is not None:
                resurrect_seen.append((uuid, len(surviving)))
            ads = sorted(surviving + list(user_ads), key=lambda b: b[0])
            which = ("merged" if user.exists() and auto.exists()
                     else "user" if user.exists()
                     else "auto" if auto.exists() else "")
        # Pseudo-label fallback: if no user OR auto labels, but a
        # pseudo_labels.json exists from a previous self-training run,
        # use those per-frame labels (filtered by confidence + minute-
        # prior agreement; lower training weight than auto).
        pseudo_path = rec_dir / "pseudo_labels.json"
        pseudo_data = None
        if not ads and pseudo_path.is_file():
            try:
                pseudo_data = json.loads(pseudo_path.read_text())
                which = "pseudo"
            except Exception:
                pseudo_data = None
        # Bootstrap path for --write-pseudo-labels: an unreviewed
        # recording with no labels at all must still be feature-extracted
        # so Phase B can predict on it and write pseudo_labels.json.
        # Marked as "bootstrap" — contributes zero training frames
        # this run (frame_mask = all False) but seeds the next run.
        is_bootstrap = False
        if not ads and pseudo_data is None:
            cohort = uuid_cohort.get(uuid)
            cohort_suspect = cohort in suspect_cohorts if cohort else False
            if auto_confirmed_no_ads and not cohort_suspect:
                # Trusted "no ads" signal: empty ads_user.json from
                # the auto-confirm pipeline AND no other recording in
                # this (title, channel) cohort has user-confirmed ads
                # → confidence the recording really has no ads is high.
                # labels_for(seconds, []) returns all-zero per-frame
                # labels — the signal the model needs to learn "this
                # whole recording is show".
                which = "auto-confirm"
            elif args.write_pseudo_labels:
                is_bootstrap = True
                which = ("bootstrap-cohort-suspect" if cohort_suspect
                         and auto_confirmed_no_ads else "bootstrap")
            else:
                continue

        # NOTE: we used to require a playable HLS-VOD here (index.m3u8 marker,
        # written by the snapshot fetch only when has_index_m3u8). That dropped
        # every recording whose VOD was disk-pruned — including ~100 *reviewed*
        # ones whose ground-truth labels + cached features still exist. Training
        # needs features + labels, NOT a VOD, so the gate is gone; the feature
        # gate below (.ts or cached .npy) is the real "can we featurize it?"
        # check and drops anything genuinely un-featurizable.
        base_txts = [p for p in rec_dir.glob("*.txt") if not any(
            p.name.endswith(s) for s in (".logo.txt", ".cskp.txt",
                                          ".tvd.txt", ".trained.logo.txt"))]
        if not base_txts:
            base_txts = [p for p in rec_dir.glob("*.cskp.txt")]
        if not base_txts:
            continue
        base = base_txts[0].stem.replace(".cskp", "")
        title = fold_show_title(base.split(" $")[0])
        # Cache-key suffix: -l2 logo (cropdetect y-offset), -c1 channel
        # one-hot, -a1 audio rms, -y1 yamnet, -u1 uniformity. The suffix is
        # part of the cache filename, so a flag flip never reuses wrong-shape
        # features.
        suffix = ""
        if args.with_logo:    suffix += "-l2"
        if args.with_channel: suffix += "-c1"
        if args.with_audio:   suffix += "-a1"
        if args.with_yamnet:  suffix += "-y1"
        if args.with_uniformity: suffix += "-u1"
        fps_tag = f"-fps{int(args.fps_extract*100)}"
        slug = uuid_slug.get(uuid, "")
        # The source .ts lives in the daemon's T7 cache (UUID-keyed). It's
        # preferred — it lets us key the feature cache by the .ts mtime and
        # re-extract on demand. BUT after Pi-dedup the .ts is frequently gone
        # while the extracted features survive in the (never-evicted) feature
        # cache. A missing .ts must NOT drop the recording: fall back to the
        # newest matching cached .npy and keep it in the corpus on its features
        # alone. Only when BOTH are gone do we skip (the daemon's prefetch loop
        # may restore the .ts before the next run).
        cand = Path(args.daemon_cache) / f"{uuid}.ts"
        if cand.exists():
            src = cand
            src_mt = int(src.stat().st_mtime)
            cache_path = cache_dir / f"{uuid}-{src_mt}{fps_tag}{suffix}.npy"
            rec_info = (uuid, title, ads, which, slug, str(rec_dir), str(src),
                         pseudo_data, is_bootstrap,
                         confirmed_show, confirmed_ad_skips)
            if cache_path.exists():
                # Re-extract if the cached features have a high NaN-rate in the
                # logo column. Stale .npy from the pre-2026-05-23 tv-detect
                # (interlaced-PTS bug, back-half rows NaN) live in the cache
                # until the .ts mtime changes, which never happens on finalized
                # recordings. mmap-peek is cheap (~50 ms per file).
                reextract = False
                if args.with_logo:
                    try:
                        arr = np.load(cache_path, mmap_mode="r")
                        if arr.shape[1] > 1280 and len(arr) > 0:
                            nan_pct = (100.0 * np.isnan(arr[:, 1280]).sum()
                                       / len(arr))
                            if nan_pct >= args.reextract_logo_nan_pct:
                                reextract = True
                    except Exception:
                        reextract = True
                if reextract:
                    todo.append((rec_info, str(src), cache_path))
                else:
                    cached.append((rec_info, cache_path))
            else:
                todo.append((rec_info, str(src), cache_path))
        else:
            # No .ts (dedup'd) — recover from the feature cache. Newest .npy
            # that matches the current fps+suffix (anchored on the literal
            # tail so a different feature set never loads wrong-shape data).
            hits = sorted(cache_dir.glob(f"{uuid}-*{fps_tag}{suffix}.npy"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if not hits:
                continue
            cache_path = hits[0]
            src_mt = int(cache_path.stat().st_mtime)  # proxy for rec_age_days
            rec_info = (uuid, title, ads, which, slug, str(rec_dir), "",
                         pseudo_data, is_bootstrap,
                         confirmed_show, confirmed_ad_skips)
            cached.append((rec_info, cache_path))
            corpus_no_ts += 1

    if corpus_no_ts:
        print(f"corpus: recovered {corpus_no_ts} recording(s) from the feature "
              f"cache whose .ts was dedup'd (kept in corpus, no re-extract)")

    if resurrect_vetoed:
        n = sum(k for _, k in resurrect_vetoed)
        print(f"label-merge: dropped {n} auto block(s) across "
              f"{len(resurrect_vetoed)} recording(s) that a confirmed_show "
              f"timestamp falls inside (reviewer rejected them; the client "
              f"omitted the 'deleted' entry)")
        for u, k in resurrect_vetoed[:10]:
            print(f"    {u}  ({k})")
    if resurrect_seen:
        n = sum(k for _, k in resurrect_seen)
        print(f"label-merge: {n} auto block(s) across {len(resurrect_seen)} "
              f"reviewed recording(s) survive with no user counterpart — "
              f"intended as un-refined blocks, but an unrecorded removal looks "
              f"identical. Watch this count: it should not grow.")
        for u, k in resurrect_seen[:10]:
            print(f"    {u}  ({k})")

    # Pass 2 — extract uncached features in parallel. Each worker loads
    # its own ONNX session at init (~100 MB resident); 4 workers × that
    # is fine on M5 Pro.
    if todo:
        flags = []
        if args.with_logo: flags.append("logo")
        if args.with_channel: flags.append("chan")
        if args.with_audio: flags.append("audio")
        if args.with_yamnet: flags.append("yamnet")
        if args.with_uniformity: flags.append("uniformity")
        flagstr = f" (+{'+'.join(flags)})" if flags else ""
        print(f"extracting {len(todo)} new recording(s) on {args.workers} workers{flagstr}...")
        t0 = time.time()
        logo_dir = Path(args.logo_dir)
        with cf.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_worker_init,
                initargs=(args.backbone,)) as ex:
            future_map = {}
            for rec_info, src, cache_path in todo:
                slug = rec_info[4]
                logo_path = ""
                if args.with_logo and slug:
                    cand = logo_dir / f"{slug}.logo.txt"
                    if cand.is_file() and cand.stat().st_size > 0:
                        logo_path = str(cand)
                chan_slug = slug if args.with_channel else ""
                future_map[ex.submit(
                    _worker_extract,
                    (src, args.fps_extract, str(cache_path),
                     logo_path, args.tv_detect, chan_slug,
                     args.with_audio, args.with_yamnet,
                     args.with_uniformity))] = rec_info
            done = 0; skipped = 0
            for fut in cf.as_completed(future_map):
                rec_info = future_map[fut]
                try:
                    cache_path_str, feats = fut.result()
                except Exception as e:
                    # ffprobe / ffmpeg failures (= source missing,
                    # corrupt, or evicted from T7 cache between snapshot
                    # fetch and worker run) used to crash the whole
                    # train script. Skip the recording, keep going.
                    skipped += 1
                    print(f"  [SKIP] {rec_info[0][:8]} {rec_info[1][:35]} — {type(e).__name__}: {str(e)[:120]}",
                          flush=True)
                    continue
                np.save(cache_path_str, feats)
                done += 1
                print(f"  [{done}/{len(todo)}] {rec_info[0][:8]} {rec_info[1][:35]} → {feats.shape}",
                      flush=True)
            if skipped:
                print(f"  ⚠ skipped {skipped} recording(s) due to extract errors",
                      flush=True)
        print(f"  parallel extract: {time.time()-t0:.1f}s for {len(todo)} recordings")

    # Pass 2.6 — SERIAL logo-NaN salvage. The parallel Pass 2 runs `args.workers`
    # outer workers, each driving a CoreML backbone pass AND a logo subprocess;
    # on LONG recordings the logo subprocess gets starved/killed under that load
    # and the whole recording comes back 100% NaN (the .ts is fine — solo
    # re-extraction yields full valid confidences). The cached-NaN guard above
    # only re-queues such recordings into the SAME contended pool next run, so
    # they re-fail and never heal (re-confirmed 2026-06-15: 9 long garden/Galileo
    # recs, all 100% NaN in the cron, all 0% NaN extracted solo). Here we
    # re-extract the logo column ALONE — no ProcessPoolExecutor, no GPU
    # contention — and patch it back into the .npy. Solo is ~40 s even for a 2 h
    # recording, and this only runs for the offenders that Pass 2 left NaN.
    if args.with_logo:
        salvage_logo_dir = Path(args.logo_dir)
        salvage = []
        seen_cp = set()
        for rec_info, cache_path in (
                [(ri, cp) for ri, _, cp in todo] +
                [(ri, cp) for ri, cp in cached]):
            cp = str(cache_path)
            if cp in seen_cp:
                continue
            src = rec_info[6]; slug = rec_info[4]
            if not src or not Path(src).exists() or not slug:
                continue  # dedup'd (no .ts) or no slug → can't re-extract
            cand = salvage_logo_dir / f"{slug}.logo.txt"
            if not (cand.is_file() and cand.stat().st_size > 0):
                continue
            try:
                arr = np.load(cp, mmap_mode="r")
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] <= 1280 or len(arr) == 0:
                continue
            nan_pct = 100.0 * np.isnan(arr[:, 1280]).sum() / len(arr)
            if nan_pct >= args.reextract_logo_nan_pct:
                salvage.append((rec_info, cp, str(cand), nan_pct))
                seen_cp.add(cp)
        if salvage:
            print(f"logo-salvage: {len(salvage)} recording(s) ≥"
                  f"{args.reextract_logo_nan_pct:.0f}% logo-NaN after parallel "
                  f"extract — re-extracting SOLO (no contention)...", flush=True)
            t0 = time.time(); healed = 0
            for rec_info, cp, logo_path, before in salvage:
                src = rec_info[6]
                try:
                    feats = np.load(cp)
                    y_off = detect_letterbox_offset(src)
                    logo_arr = extract_logo_per_second(
                        src, logo_path, n_seconds=feats.shape[0],
                        tv_detect=args.tv_detect, y_offset=y_off)
                    after = 100.0 * np.isnan(logo_arr).sum() / max(len(logo_arr), 1)
                    if after < before:
                        feats[:, 1280] = logo_arr.astype(np.float32)
                        np.save(cp, feats)
                        healed += 1
                    print(f"  {'✓' if after < before else '·'} "
                          f"{rec_info[0][:8]} {rec_info[1][:35]} "
                          f"{before:.0f}%→{after:.0f}% NaN", flush=True)
                except Exception as e:
                    print(f"  ✗ {rec_info[0][:8]} {rec_info[1][:35]} — "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            print(f"  logo-salvage: healed {healed}/{len(salvage)} in "
                  f"{time.time()-t0:.1f}s", flush=True)

    # Pass 2.5 — generate bumpers.json for any recording missing one,
    # in parallel. Cheap (ffmpeg-only, no ML, ~10 s per recording on
    # M5 Pro) and the cache is just a JSON file next to the .ts.
    if args.with_bumpers:
        all_rec_infos = ([ri for ri, _ in cached] +
                          [ri for ri, _, _ in todo])
        bumper_todo = []
        for ri in all_rec_infos:
            rec_dir = Path(ri[5])
            src = ri[6]
            bj = rec_dir / "bumpers.json"
            if not bj.exists() and src and Path(src).exists():
                bumper_todo.append((src, str(bj)))
        if bumper_todo:
            print(f"bumpers: extracting for {len(bumper_todo)} recording(s) "
                  f"on {args.workers} workers...")
            t0 = time.time()
            def _run_bumper(args_tuple):
                src, out = args_tuple
                subprocess.run([sys.executable, args.bumper_detect, src,
                                "--out", out, "--quiet"],
                                check=False, capture_output=True, timeout=600)
                return out
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                done = 0
                for _ in ex.map(_run_bumper, bumper_todo):
                    done += 1
                    if done % 5 == 0 or done == len(bumper_todo):
                        print(f"  [{done}/{len(bumper_todo)}]", flush=True)
            print(f"  parallel bumper-detect: {time.time()-t0:.1f}s")

    # Pass 3 — load all cached features + assemble per-recording arrays
    # (kept separate so we can split train/test BY RECORDING, not by
    # frame — a random per-frame split leaks show identity since
    # adjacent frames are highly correlated).
    # _touch_atime: mark a feature .npy as in-use for the wrapper's
    # 60-day cache prune. APFS does NOT reliably update atime on reads
    # (verified 2026-07-15: files np.load'ed the same morning still
    # showed weeks-old atimes), so without an explicit bump the prune
    # would eventually delete features the deletion-safe archive still
    # references — mass-orphaning frozen corpus entries. mtime is
    # deliberately preserved (it serves as the rec-age proxy for the
    # age-decay weighting of cache-recovered recordings).
    def _touch_atime(p):
        try:
            st = os.stat(p)
            os.utime(p, (time.time(), st.st_mtime))
        except OSError:
            pass

    per_rec = []  # list of (uuid, title, ads, X, y, has_user)
    dropped_high = []
    logo_nan_offenders = []  # (uuid, title, miss_pct) for log summary
    logo_nan_mask_by_uuid = {}  # uuid → bool array, True where logo was NaN
    for rec_info, cache_path in cached + [(ri, cp) for ri, _, cp in todo]:
        if not Path(cache_path).exists():
            continue
        feats = np.load(cache_path)
        _touch_atime(cache_path)
        if feats.shape[0] == 0:
            continue
        # NaN sentinel handling: extract_logo_per_second writes NaN
        # for unmeasurable seconds (= corrupt stream chunk, missing
        # template). For the X matrix we substitute 0.5 to keep
        # arithmetic valid, but the per-frame NaN mask is preserved
        # in logo_nan_mask_by_uuid so the sample-weight builder can
        # zero out those frames in fit (= "head learns missing == skip"
        # without coupling to a Go-side schema change). Per-recording
        # miss rates ≥10% are surfaced in the post-load summary.
        if args.with_logo and feats.shape[1] > 1280:
            nan_mask = np.isnan(feats[:, 1280])
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                miss_pct = 100.0 * n_nan / len(nan_mask)
                if miss_pct >= 10.0:
                    logo_nan_offenders.append(
                        (rec_info[0][:8], rec_info[1][:30], miss_pct))
                feats = feats.copy()
                feats[nan_mask, 1280] = 0.5
                logo_nan_mask_by_uuid[rec_info[0]] = nan_mask
        uuid, title, ads, which, *rest = rec_info
        rec_dir_path = Path(rest[1]) if len(rest) > 1 else None
        pseudo_data = rest[3] if len(rest) > 3 else None
        is_bootstrap = rest[4] if len(rest) > 4 else False
        # Carried through rec_info since 2026-07-26. Before that this loop read
        # the pass-1 loop variables, which by now hold the LAST recording's
        # values — so every recording in the corpus was assigned one arbitrary
        # confirmed_show set (233 of 591 archive entries share the identical
        # [22.0, 1281.0]) and all 54 real ones were ignored. confirmed_show only
        # bumps a frame's weight to 1.2×, but confirmed_ad_skips FORCES label=1:
        # had a recording with skip presses been last in the glob order, those
        # timestamps would have become ad labels in every recording.
        confirmed_show = rest[5] if len(rest) > 5 else []
        confirmed_ad_skips = rest[6] if len(rest) > 6 else []
        bumpers = []
        if args.with_bumpers and rec_dir_path is not None:
            bj = rec_dir_path / "bumpers.json"
            if bj.exists():
                try: bumpers = json.loads(bj.read_text())
                except Exception: bumpers = []
        seconds = [i / args.fps_extract for i in range(feats.shape[0])]
        # Pseudo path: build a per-frame label array from the
        # {frames, labels} arrays in pseudo_labels.json + a frame_mask
        # marking which frames have opinions. Frames without pseudo-
        # labels are excluded from training via the mask (no opinion =
        # not training data, NOT a default-show prediction).
        if pseudo_data:
            n_frames = feats.shape[0]
            labels = np.zeros(n_frames, dtype=np.int8)
            frame_mask = np.zeros(n_frames, dtype=bool)
            for fi, ll in zip(pseudo_data.get("frames", []),
                                pseudo_data.get("labels", [])):
                if 0 <= fi < n_frames:
                    labels[fi] = int(ll)
                    frame_mask[fi] = True
        elif is_bootstrap:
            # Zero training contribution: empty labels, all-False mask.
            # Still in per_rec so Phase B can predict on these features.
            labels = np.zeros(feats.shape[0], dtype=np.int8)
            frame_mask = np.zeros(feats.shape[0], dtype=bool)
        else:
            labels = labels_for(seconds, ads)
            frame_mask = None
        ad_rate = float(labels.mean()) if len(labels) else 0.0
        # Hygiene filter: drop obviously-broken recordings. The auto
        # detection on a recording with a bad logo template can mark
        # 80-100% of frames as ad (e.g. RTL with the pre-cap 74304-px²
        # template we found earlier today). Such labels poison the
        # head — train acc looks fine but we're memorising garbage.
        # Skip the filter for pseudo recordings — their ad_rate
        # reflects only the kept high-confidence frames, not whole-
        # recording structure.
        if pseudo_data is None and ad_rate > args.max_ad_rate:
            dropped_high.append((uuid[:8], title[:30], ad_rate))
            continue
        has_user = which in ("user", "merged")
        rec_age_days = (time.time() - src_mt) / 86400.0
        # Cluster-anchored ad spots from the gateway snapshot (= spots
        # whose audio+visual fingerprint matches a known ≥3-member
        # family). For reviewed recordings: redundant with ads_user
        # but bumps weight at high-confidence frames. For unreviewed:
        # the only high-confidence ad signal we have without manual
        # review.
        cluster_anchored = []
        ca_path = rec_dir / "cluster_anchored.json"
        if ca_path.is_file():
            try:
                cluster_anchored = json.loads(ca_path.read_text()) or []
            except Exception:
                cluster_anchored = []
        per_rec.append((uuid, title, ads, feats, labels, has_user,
                        confirmed_show, confirmed_ad_skips, rec_age_days,
                        bumpers, frame_mask, which == "pseudo",
                        is_bootstrap, cluster_anchored))
        # Deletion-safe training archive: freeze a trustworthy-labelled
        # recording's label (+ a pointer to its cached features) so it stays
        # in the corpus even after its .ts is deleted/dedup'd. Only reviewed
        # (user/merged), auto-confirmed, or cluster-anchored labels — never
        # pseudo/bootstrap (those must stay live + refreshable). Features
        # aren't duplicated (the never-evicted feature cache keeps the .npy);
        # we store its path. Overwritten each run so re-reviews refresh;
        # persists unchanged once the recording is gone.
        if (archive_dir is not None and frame_mask is None and not is_bootstrap
                and (which in ("user", "merged", "auto-confirm")
                     or bool(cluster_anchored))):
            try:
                np.savez(archive_dir / f"{uuid}.npz",
                         labels=labels,
                         meta=json.dumps({
                             "uuid": uuid, "title": title,
                             "slug": uuid_slug.get(uuid, ""),
                             "start_ts": uuid_start.get(uuid, 0),
                             "ads": ads, "which": which,
                             "confirmed_show": confirmed_show,
                             "confirmed_ad_skips": confirmed_ad_skips,
                             "cluster_anchored": cluster_anchored,
                             "feature_npy": str(cache_path),
                             # Frozen while the uuid is still live in
                             # uuid_cohort (= gateway dvr/entry/grid). Series-
                             # retention (dvr_series_retention) can delete the
                             # DVR entry itself, not just the .ts — at that
                             # point the gateway no longer knows the cohort,
                             # so it must be recovered from here, not re-
                             # looked-up (root-caused 2026-07-09).
                             "cohort": list(uuid_cohort.get(uuid, ("", ""))),
                         }))
            except Exception as e:
                print(f"  train-archive: write {uuid[:8]} failed: {e}", flush=True)

    # Deletion-safe training archive (read side): admit recordings no longer
    # live (deleted, or .ts dedup'd → dropped from the walk above) that have a
    # frozen entry. Features come from the never-evicted feature cache via the
    # stored path; labels from the archive. Keeps the corpus + sticky test
    # split stable across deletions/dedup. Shape-mismatch (fps/feature-set
    # drift) → skip rather than misalign labels.
    # Dead recordings with pure-machine labels (which=auto/auto-confirm,
    # never human-touched) are collected here during archive injection so
    # the split step can RETIRE them from the TEST set (ledger flip
    # test→train — leak-safe in that direction, the reverse is forbidden).
    # Rationale (2026-07-15): 65 of 103 test recs were dead+frozen, 27 of
    # them labelled only by old detection heads — the gate was scoring
    # candidates against stale machine output that can never be
    # re-verified (sources gone from Pi, VOD, and Mac cache alike) and
    # that evaluates the whisper feature at neutral 0.5 (transcripts
    # predate WHISPER_TRANSCRIBE). Human-touched dead recs (user/merged)
    # stay: imperfect but anchored ground truth. Standing rule, not a
    # one-shot migration — recordings that die later get caught too.
    dead_machine_labelled = set()

    if archive_dir is not None:
        live_uuids = ({ri[0] for ri, _ in cached}
                      | {ri[0] for ri, _, _ in todo})
        injected = 0
        for npz_path in sorted(archive_dir.glob("*.npz")):
            u = npz_path.stem
            if u in live_uuids:
                continue
            try:
                z = np.load(npz_path, allow_pickle=False)
                a_labels = z["labels"]
                a_meta = json.loads(str(z["meta"]))
            except Exception:
                continue
            if a_meta.get("which", "") in ("auto", "auto-confirm"):
                dead_machine_labelled.add(u)
            fnpy = Path(a_meta.get("feature_npy", ""))
            if not fnpy.exists():
                continue  # features cleared → can't reconstruct; skip
            try:
                a_feats = np.load(fnpy)
                _touch_atime(fnpy)
            except Exception:
                continue
            if a_feats.shape[0] == 0 or a_feats.shape[0] != len(a_labels):
                continue
            if args.with_logo and a_feats.shape[1] > 1280:
                nm = np.isnan(a_feats[:, 1280])
                if nm.any():
                    a_feats = a_feats.copy()
                    a_feats[nm, 1280] = 0.5
                    logo_nan_mask_by_uuid[u] = nm
            a_which = a_meta.get("which", "")
            a_start = a_meta.get("start_ts", 0)
            a_age = (time.time() - a_start) / 86400.0 if a_start else 0.0
            # Dead recs aren't in the gateway grid, so uuid_start (used
            # by the minute-prior feature paths) would miss them — the
            # frozen start_ts serves the same purpose.
            if a_start and u not in uuid_start:
                uuid_start[u] = int(a_start)
            # Re-apply the same suspect-cohort downgrade the LIVE walk does
            # (~line 1781) to frozen archive entries — root-caused 2026-07-13
            # via SpongeBob Schwammkopf: two archive entries written BEFORE
            # the 2026-07-09 cohort-trust fix (no "cohort" key at all, uuid
            # long gone from the live DVR grid so uuid_cohort can't recover
            # it either) sat frozen as which="auto-confirm" (0% ad rate,
            # never reviewed) despite their own (title,channel) cohort
            # having user-confirmed episodes WITH real ads elsewhere — the
            # exact "Nick SpongeBob 91/118 auto-confirmed-empty, 11 reviewed
            # have ads" pattern this mechanism exists to catch. Without this,
            # a stale pre-fix archive entry stays untouchably "trusted" as
            # TEST ground truth forever, since injection only happens once
            # per uuid and the live downgrade logic never re-runs on it.
            a_cohort = (tuple(a_meta["cohort"]) if a_meta.get("cohort")
                       else uuid_cohort.get(u))
            a_cohort_suspect = a_cohort in suspect_cohorts if a_cohort else False
            a_is_bootstrap = a_which == "auto-confirm" and a_cohort_suspect
            # Frozen cluster_anchored is NEVER re-injected for archive-only
            # (non-live) recordings — root-caused 2026-07-14: the spot-
            # fingerprint family DB is re-clustered (rebuildFamilies, full
            # renumber) as new recordings get fingerprinted every night, so
            # a family_id/window_start_s snapshot frozen once at archive-
            # write time silently rots. Found via a training-rejection
            # investigation: 419 recordings, grouped into 5 batches each
            # written within a single archiving run (weeks apart, mostly
            # ~1 night wide), carried BYTE-IDENTICAL cluster_anchored
            # payloads despite being completely unrelated shows/channels
            # — direct sqlite inspection on the Pi
            # confirmed those same fingerprint rows, where they still exist
            # at all, no longer share any family in the CURRENT live table
            # (each now sits alone, family_size=1). Since cluster_anchored
            # exists specifically to catch cases a live re-scan could
            # otherwise verify (comment above fingerprintRescue/the "only
            # high-confidence signal for unreviewed recordings" rationale
            # in the live-walk cluster_anchored read above), and an
            # archive-only uuid can by definition never be re-verified
            # (it dropped out of the live grid, same "gap" as the
            # 2026-07-09/07-13 cohort-trust fix — see
            # tv_detect_cohort_trust_archive_gap), the frozen value is
            # unlike `cohort`/`ads`/labels (durable facts, correctly kept)
            # — it is a live-DB-state cache that has no business outliving
            # the DB state it was read from. Drop it rather than
            # re-injecting a month-old (possibly wrong) label=1 anchor
            # into every training run indefinitely.
            per_rec.append((u, a_meta.get("title", ""), a_meta.get("ads", []),
                            a_feats, a_labels, a_which in ("user", "merged"),
                            a_meta.get("confirmed_show", []),
                            a_meta.get("confirmed_ad_skips", []), a_age,
                            [], None, False, a_is_bootstrap, []))
            injected += 1
        if injected:
            print(f"train-archive: injected {injected} deleted/dedup'd "
                  f"recording(s) (trained via frozen label + cached features)")

    # Right-pad all per_rec feature matrices to the widest column count
    # we have. Cached .npy files from older runs (extracted before
    # the slug→logo lookup landed) can be 1 column narrower than the
    # current full-feature pipeline. Padding with neutral 0.5 keeps
    # train/eval/concat happy without forcing a full re-extraction
    # of every old recording.
    if per_rec:
        target_dim = max(r[3].shape[1] for r in per_rec)
        for i, r in enumerate(per_rec):
            f = r[3]
            if f.shape[1] < target_dim:
                pad = np.full((f.shape[0], target_dim - f.shape[1]),
                              0.5, dtype=f.dtype)
                f = np.concatenate([f, pad], axis=1)
                per_rec[i] = (r[0], r[1], r[2], f, r[4], r[5],
                              r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                              r[13] if len(r) > 13 else [])
    if dropped_high:
        print(f"hygiene: dropped {len(dropped_high)} recording(s) with "
              f"ad-rate > {args.max_ad_rate*100:.0f}% "
              f"(suspect broken auto-labels):")
        for u, t, r in dropped_high:
            print(f"  {u} {t:30s} {r*100:.0f}%")
    if logo_nan_offenders:
        print(f"logo: {len(logo_nan_offenders)} recording(s) with "
              f">=10% unmeasurable seconds (= NaN-sentinel from corrupt "
              f"stream chunks; substituted with 0.5 for fit, but flagged "
              f"because block-formation can't see logo signal there):")
        for u, t, p in sorted(logo_nan_offenders, key=lambda x: -x[2])[:20]:
            print(f"  {u} {t:30s} {p:.0f}% NaN")

    if not per_rec:
        print("no labelled recordings found", file=sys.stderr)
        sys.exit(1)

    # Per-channel wall-clock minute-of-hour prior: empirical
    # P(ad | minute_of_hour) histogram aggregated across all labelled
    # recordings for each channel. Privates (RTL/Pro7/SAT.1/sixx/VOX)
    # slot ad blocks at fixed minute offsets (regulated to 12 min/h
    # max, mediabuying favours predictability), so the histogram is
    # often sharply peaked at 3-4 minute ranges per hour. Used by
    # the active-learning surfacer below — frames where head's p
    # diverges strongly from prior(minute) become high-value targets.
    minute_prior = {}  # slug -> [60-element list of P(ad)]
    if args.with_minute_prior:
        from collections import defaultdict
        bucket_pos = defaultdict(lambda: np.zeros(60, dtype=np.float64))
        bucket_n = defaultdict(lambda: np.zeros(60, dtype=np.float64))
        for r in per_rec:
            uuid = r[0]
            slug = uuid_slug.get(uuid, "")
            start = uuid_start.get(uuid, 0)
            if not slug or not start:
                continue
            labels = r[4]
            n = len(labels)
            for i in range(n):
                wall_s = start + i / args.fps_extract
                m = int(wall_s // 60) % 60
                bucket_pos[slug][m] += float(labels[i])
                bucket_n[slug][m] += 1.0
        # Bayesian smoothing: add a virtual 5 frames at the channel
        # average rate to each bucket so under-observed minutes don't
        # collapse to 0 or 1 from a single example.
        for slug, n_arr in bucket_n.items():
            ch_mean = bucket_pos[slug].sum() / max(1.0, n_arr.sum())
            smoothed = ((bucket_pos[slug] + 5.0 * ch_mean) /
                        (n_arr + 5.0))
            minute_prior[slug] = smoothed.round(3).tolist()
        if minute_prior:
            prior_path = Path(args.hls_root) / ".minute_prior_by_channel.json"
            prior_path.write_text(json.dumps(minute_prior, indent=1))
            # Print sharpness summary so the user sees whether the prior
            # actually carries signal (peaked = useful, flat = useless).
            print(f"minute-prior: built for {len(minute_prior)} channel(s) "
                  f"→ {prior_path.name}")
            for slug, p in sorted(minute_prior.items()):
                arr = np.array(p)
                peaks = int((arr > arr.mean() + arr.std()).sum())
                print(f"  {slug:12s} mean={arr.mean():.2f}  "
                      f"min={arr.min():.2f}  max={arr.max():.2f}  "
                      f"peaks={peaks}/60")

    # Deterministic train/test split by recording uuid. Same uuid →
    # same bucket across runs, so adding new recordings doesn't shuffle
    # the existing split.
    #
    # TEST_SET_EXCLUDE — known-bad ground truth, never trust as a test
    # target (still eligible for train_recs at its normal which=auto
    # weight; only the SCORING role is revoked). Root-caused 2026-07-13:
    # "Reisen mit Kreta.de" sat at IoU 0.00 for the whole shadow-eval
    # week, dragging the movies/niche median down. Its "ads" field
    # (which=auto, never user-reviewed) has exactly ONE block
    # (414-582s), but the SAME recording's own cluster_anchored list
    # independently flags 9 more high-confidence ad spots (family sizes
    # 3-59, i.e. matched against 3-59 other airings) between 618s and
    # 1029s that never made it into "ads" — the auto-cutlist truth is
    # badly incomplete, so the model's low IoU against it was punishing
    # correct detections, not revealing a real weakness. Flagged in the
    # 2026-07-07 optimization backlog review; add future confirmed-bad
    # test recs here rather than re-litigating per rejection.
    TEST_SET_EXCLUDE = {
        "dvr-anixe-1781518500",
        # 2026-07-30 label audit: frozen-archive entry (source long gone,
        # serien-retention) with a 361s hole on user labels — NN sure-ad
        # inside labelled show, unverifiable forever. npz quarantined like
        # the 07-28 fossils.
        "dvr-rtl-1781444700",          # Die Beet-Brüder
        # 2026-07-31 both-cold triage: two more dead merged-label fossils,
        # both heads <0.40 for days, no source anywhere to verify.
        # 989a0bea has NO channel slug (tvh-era) and a 0-120s stub label.
        "989a0bea63b249d1a6243d5f3f27e0ed",  # SpongeBob (tvh-era)
        "dvr-rtl-1780224000",          # Die Beet-Brüder (2. Fossil)
        # 2026-07-15 FP-concentration analysis on the deployed MLP3:
        # these two DEAD which=merged recordings alone carried 50% of
        # ALL measured test-frame errors (4210 of 8385) — the model
        # "false-positives" on 47-65% of their runtime, against a GT
        # that marks only ~30% ad, has ZERO confirmed_show points, and
        # can never be re-verified (no source/VOD/cache anywhere). A
        # 95.6%-acc model doesn't fail at 65% on one recording; the
        # frozen old auto-era cutlist is what's wrong (same class as
        # Kreta.de above, but which=merged so the dead-machine-label
        # retirement rule spared them).
        "dvr-kabel-eins-1779980100",   # Abenteuer Leben täglich
        "dvr-rtlzwei-1780226100",      # Von Hecke zu Hecke
        # 2026-07-20 Form()-sweep triage: these six scored IoU 0.000
        # across ALL 217 grid combos (features↔signals durations verified
        # aligned, so not a cache artifact) — the GT itself is the outlier,
        # not the params. No user labels, no archive npz. Four carry an
        # empty machine cutlist on 63-82-min recordings (n_blocks=0-as-
        # labels class), two a single tail block running exactly to the
        # recording end (captured-neighbour signature).
        "dvr-prosieben-1783011902",    # 72min, zero-block GT
        "dvr-prosieben-1783271066",    # 82min, zero-block GT
        "dvr-vox-1783008000",          # 63min, zero-block GT
        "dvr-vox-1783357200",          # 78min, zero-block GT
        "dvr-prosieben-1780544100",    # Call Me Kat — GT=[1320,1920.56] to EOF
        "dvr-prosieben-1781406105",    # Die Goldbergs — GT=[1291,1592] to EOF
        # 2026-07-22: Let's Dance 05-22 — dead rec whose cached source is
        # truncated (12570s) vs its frozen features (14056s), so the
        # signals cache is permanently tombstoned and eval falls back to
        # the naive threshold path — which ignores the per-show nn-heavy
        # override this logo-hiding show REQUIRES. Scores IoU/F1 0.00
        # forever (production would cut it fine), and its title classifies
        # as "movie", so it alone floored OVERALL(movies) to 0.32. No user
        # labels; can never be re-verified.
        "dvr-rtl-1779473700",          # Let's Dance — truncated source, naive-fallback-only
        # 2026-07-24 label audit (prosieben-1779878100 was 0.00/0.00 on BOTH
        # heads in the both-heads-cold report): a whole BATCH of archive npz
        # written 06-02 07:41 — i.e. BEFORE the 06-04 source-cache truncation
        # guard ([[source_cache_truncation_silent]]) — were extracted from a
        # source truncated at ~40-45min. Signature (swept across all archives):
        # the meta "ads" list carries a block whose END is BEYOND the label
        # horizon while labels are all-zero (ad_frac 0.000).
        #
        # These entries are now BELT-AND-SUSPENDERS: the 11 npz themselves were
        # QUARANTINED out of the corpus (~/.cache/tvd-train-archive-quarantine/)
        # because — contrary to the first read of this audit — the all-zero
        # labels are POISON in TRAIN, not merely incomplete negatives. Proof:
        # moving prosieben-1779878100 test→train in the 07-24 15:37 run (via an
        # earlier TEST_SET_EXCLUDE-only fix that kept them train-eligible)
        # crashed the sibling reviewed BBT dvr-prosieben-1782482508 from a stable
        # 0.88 (3 prior runs) to 0.45 — the deterministic MLP (random_state=0)
        # learned "BBT = no ad" from the mislabelled rec. A truncated sitcom with
        # its ad-break cut off and labelled all-show is an unrepresentative
        # negative that biases the whole show's prior. Unfixable (no source to
        # re-detect), so they're removed from BOTH roles. If a quarantined npz is
        # ever restored, these keep it out of TEST too. 11 recs, 2.5-Men / Big
        # Bang / Charmed midday reruns. (Galileo dvr-prosieben-1780506300 hit the
        # truncation signature too but is a real 65min rec, ad_frac 0.311, minor
        # tail overrun — NOT truncated, left in the corpus.)
        "dvr-prosieben-1779870300",    # Two and a Half Men — trunc 40min, quarantined
        "dvr-prosieben-1779871800",    # Two and a Half Men — trunc 45min, quarantined
        "dvr-prosieben-1779873600",    # Two and a Half Men — trunc 40min, quarantined
        "dvr-prosieben-1779875100",    # The Big Bang Theory — trunc 40min, quarantined
        "dvr-prosieben-1779878100",    # The Big Bang Theory — trunc 40min, quarantined (crashed sibling)
        "dvr-prosieben-1779884400",    # Two and a Half Men — trunc 40min, quarantined
        "dvr-prosieben-1779885900",    # Two and a Half Men — trunc 45min, quarantined
        "dvr-prosieben-1779889301",    # The Big Bang Theory — trunc 42min, quarantined
        "dvr-sixx-1779894000",         # Charmed — trunc 40min, quarantined
        "dvr-sixx-1779980700",         # Charmed — trunc 40min, quarantined
        "dvr-sixx-1780069500",         # Charmed — trunc 45min, quarantined
        # 2026-07-27 corpus-wide label audit (scripts/corpus-label-audit.py):
        # 16 of 486 archived recordings carry labels that contradict their OWN
        # per-second NN signal — the head is confidently "ad" for minutes in a
        # stretch the labels call show, or the reverse. Same poison class as the
        # 07-24 batch above, found systematically rather than by chance: the
        # audit recomputes the deployed head over the archive's own features, so
        # it needs no decode and covers every recording, not just the ones with
        # a signals dump. Verified before use — its verdict matched the real
        # dump on 35 of 35 recordings that had both.
        #
        # These seven are DEAD (no source, no VOD) and machine-labelled, so the
        # labels can never be re-derived. Their npz are in
        # ~/.cache/tvd-train-archive-quarantine/ ; these entries keep them out
        # of TEST too should an npz ever be restored. Two of them carry ZERO ad
        # blocks against 1254 s and 785 s of confident ad signal — the
        # n_blocks=0-as-labels class that provably crashed a sibling recording
        # from 0.88 to 0.45 on 07-24.
        "dvr-sixx-1780040100",         # Charmed — 0 blocks vs 1254s ad signal
        "dvr-sixx-1779954000",         # Charmed — 0 blocks vs 785s ad signal
        "dvr-rtl-1780832400",          # Die Beet-Brüder — 826s ad outside labels
        "dvr-sixx-1779951600",         # Charmed — 493s ad outside labels
        "dvr-nick-1781033400",         # Futurama — 411s ad outside labels
        "dvr-nick-1781031900",         # Futurama — 382s ad outside labels
        "dvr-sixx-1780291500",         # Charmed — 374s ad outside labels
        # 2026-07-28, second pass: these three carry HUMAN labels in the
        # archive but none in the live system — ads.json is empty for all
        # three, two are gone from the DVR grid entirely, and their npz are
        # frozen at 25 June / 18 July because no npz is written for a
        # label-less recording. So the archive is the ONLY place these labels
        # still exist, and the head contradicts them over 250-460 s. Removing
        # them is not overruling a reviewer: the reviewer's verdict is already
        # gone everywhere else. dvr-kabel-eins-1779980100 — the same show, the
        # same signature — was quarantined for exactly this on 07-15.
        "dvr-vox-1781036100",          # Hot oder Schrott — 463s hole, dead
        "dvr-kabel-eins-1780325700",   # Abenteuer Leben täglich — 391s hole, dead
        "dvr-nick-1778954400",         # SpongeBob — 257s phantom, live ads empty
    }

    # ── Sticky, channel-stratified split (2026-07-14) ───────────────
    # The pure hash<test_frac rule left per-channel test coverage to
    # chance — Disney Channel ended up with n=2-4 test recs and single-
    # handedly drove the 07-14 03:30 rejection's cohort table. Naive
    # re-stratification (rank-within-channel) would MOVE existing
    # memberships as the pool changes, and any train→test move is a
    # leak (the rec was already trained on — see memory
    # test_set_stickiness). Design that keeps both properties:
    #   * a persistent ledger (split-ledger.json in the train-archive
    #     dir) freezes every uuid's bucket forever at first sight;
    #   * uuids ALREADY KNOWN (seeded on the ledger's first run) keep
    #     exactly the old hash-rule bucket → zero-cutover, the gate's
    #     paired comparison is unaffected;
    #   * only NEW arrivals get a channel-adaptive test probability
    #     p = clamp(frac + (frac − channel_share), 0.05, 0.5) — an
    #     under-tested channel admits new recs to test more readily,
    #     an over-tested one less, converging each channel toward
    #     test_frac without ever reassigning an existing recording.
    import hashlib
    _ledger_path = (Path(args.train_archive) / "split-ledger.json"
                    if args.train_archive else None)
    _ledger = {}
    if _ledger_path and _ledger_path.is_file():
        try:
            _ledger = json.loads(_ledger_path.read_text())
        except Exception:
            _ledger = {}
    # Captured at load: on the very first run EVERY uuid must get the
    # legacy hash rule (seeding), not just the first one processed —
    # `not _ledger` would flip to False after the first assignment.
    _ledger_seeding = not _ledger
    _ledger_dirty = [False]
    # GOLDEN-EVAL pin: a frozen set of held-out recs (golden-eval-set.json) that
    # are FORCED to test every run so the golden-median trend is both leakage-free
    # AND composition-constant (see the GOLDEN-EVAL print near the deploy line).
    # Pinning is free here — the set was built from recs already reliably in test.
    _golden_pin = set()
    if args.train_archive:
        _gp = Path(args.train_archive) / "golden-eval-set.json"
        if _gp.is_file():
            try:
                _golden_pin = set(json.loads(_gp.read_text()).get("uuids", []))
            except Exception:
                _golden_pin = set()
    # Retire dead machine-labelled recordings from TEST (see the
    # dead_machine_labelled comment at the archive-injection site).
    # Runs against the ledger only — on the seeding run the set is
    # applied right after seeding via the same flip below.
    _retired = 0
    for _u in dead_machine_labelled:
        if _ledger.get(_u) == "test":
            _ledger[_u] = "train"
            _ledger_dirty[0] = True
            _retired += 1
    if _retired:
        print(f"split ledger: retired {_retired} dead machine-labelled "
              f"recording(s) from test → train (stale, never "
              f"re-verifiable ground truth)", flush=True)
    _uuid_slug_for_split = dict(uuid_slug)  # live grid
    for r in per_rec:  # archive-injected recs: recover slug from uuid
        u = r[0]
        if u not in _uuid_slug_for_split and u.startswith("dvr-"):
            _uuid_slug_for_split[u] = "-".join(u.split("-")[1:-1])

    def _hash_frac(uuid_str):
        h = int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)
        return h / 2**128

    def _is_test(uuid_str):
        if uuid_str in TEST_SET_EXCLUDE:
            return False
        if uuid_str in _golden_pin:
            # Frozen golden-eval rec: always test (never train → leakage-free,
            # never churned out → the golden-median stays comparable).
            if _ledger.get(uuid_str) != "test":
                _ledger[uuid_str] = "test"
                _ledger_dirty[0] = True
            return True
        if uuid_str in _ledger:
            return _ledger[uuid_str] == "test"
        if uuid_str in dead_machine_labelled:
            # never admit a dead machine-labelled recording to test —
            # covers the fresh-ledger/seeding path where the retirement
            # flip above found no existing entry to act on.
            _ledger[uuid_str] = "train"
            _ledger_dirty[0] = True
            return False
        if _ledger_seeding:
            # First-ever run (no ledger file yet): seed with the legacy
            # hash rule so every existing membership is preserved
            # verbatim — zero cutover.
            verdict = _hash_frac(uuid_str) < args.test_frac
        else:
            slug = _uuid_slug_for_split.get(uuid_str, "")
            n_test = n_all = 0
            for u, bucket in _ledger.items():
                if _uuid_slug_for_split.get(u, "") == slug:
                    n_all += 1
                    if bucket == "test":
                        n_test += 1
            share = (n_test / n_all) if n_all else args.test_frac
            p = min(0.5, max(0.05,
                             args.test_frac + (args.test_frac - share)))
            verdict = _hash_frac(uuid_str) < p
        _ledger[uuid_str] = "test" if verdict else "train"
        _ledger_dirty[0] = True
        return verdict
    # Bootstrap recordings (no labels yet, only present so Phase B can
    # predict on their features) are excluded from train AND test —
    # they have nothing to validate against.
    def _is_bootstrap(r): return len(r) > 12 and r[12]
    def _is_pseudo(r): return len(r) > 11 and r[11]
    train_recs = [r for r in per_rec
                  if not _is_test(r[0]) and not _is_bootstrap(r)]
    # Pseudo-labelled recordings are excluded from the test set:
    # eval against pseudo-labels is circular (model graded against its
    # own predictions), and the gaps between pseudo-labelled frames
    # default to label=0 which produces false-positive misses where
    # the model predicts ad for a "no-opinion" frame. Train-side they
    # contribute frames via frame_mask filtering at lower weight.
    test_recs  = [r for r in per_rec
                  if _is_test(r[0]) and not _is_bootstrap(r)
                                    and not _is_pseudo(r)]

    # Golden pins that did NOT make it into test_recs, with the reason.
    #
    # The golden set is pinned to test precisely so its membership cannot
    # drift, but the pin only forces _is_test — a pinned recording can still
    # fall out via the bootstrap/pseudo filters above, or by never reaching
    # per_rec at all (no archive entry, no features, dropped by an earlier
    # hygiene rule). When that happens the golden median is quietly computed
    # over fewer recordings.
    #
    # This exists because dvr-rtl-1781909700 (CSI: Miami) was absent for two
    # nights and nothing said so. Root cause, found 2026-07-28: its labels are
    # EMPTY. Its only ad block was one the reviewer had removed and that
    # survived the merge; the confirmed_show veto cleared it on 07-26, leaving
    # nothing — and a recording with no labels counts as bootstrap, so it drops
    # out of train AND test. Its archive npz still carries the pre-veto
    # ads=[[0,99.68]] because no npz is written for a label-less recording,
    # which is also why it read IoU 0.000 on both heads: the model followed the
    # review correctly and the frozen label did not. It has been removed from
    # the golden set. The check stays so the next one is named on the spot.
    if _golden_pin:
        _in_test = {r[0] for r in test_recs}
        _in_per_rec = {r[0] for r in per_rec}
        _lost = []
        for _u in sorted(_golden_pin - _in_test):
            if _u not in _in_per_rec:
                _why = "not in per_rec (no archive entry / features, or dropped earlier)"
            else:
                _r = next(r for r in per_rec if r[0] == _u)
                if _is_bootstrap(_r):
                    _why = "bootstrap (no labels yet)"
                elif _is_pseudo(_r):
                    _why = "pseudo-labelled (excluded from test to avoid circular eval)"
                else:
                    _why = "in per_rec and neither bootstrap nor pseudo — UNEXPLAINED"
            _lost.append((_u, _why))
        if _lost:
            print(f"golden-pin: {len(_lost)} of {len(_golden_pin)} pinned rec(s) "
                  f"are NOT in test_recs — the golden median will be short:")
            for _u, _why in _lost:
                print(f"  {_u}  {_why}")

    # Persist the split ledger + report per-channel test coverage so
    # stratification drift is visible run-to-run.
    if _ledger_path and _ledger_dirty[0]:
        try:
            tmp = _ledger_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_ledger, indent=0, sort_keys=True))
            tmp.rename(_ledger_path)
        except Exception as e:
            print(f"split-ledger: persist failed: {e}", flush=True)
    if _ledger:
        _cov = {}
        for u, bucket in _ledger.items():
            s = _uuid_slug_for_split.get(u, "?")
            d = _cov.setdefault(s, [0, 0])
            d[1] += 1
            if bucket == "test":
                d[0] += 1
        line = "  ".join(
            f"{s}:{t}/{n}" for s, (t, n) in sorted(_cov.items())
            if n >= 5)
        print(f"split ledger: {len(_ledger)} uuids"
              f"{' (seeded from legacy hash rule)' if _ledger_seeding else ''}"
              f" — test share per channel (n≥5): {line}", flush=True)

    # One-time pre-pass: build any missing --emit-signals-json decode-signal
    # caches for the test set, so eval_split()'s realistic-eval path (see
    # comment above _replay_blocks) has something to replay against. Runs
    # here, not lazily inside eval_split (called many times per training
    # run) — capped per run since a cache miss means a real decode.
    _signals_budget = [MAX_NEW_SIGNALS_PER_RUN]
    _signals_built = 0
    for r in test_recs:
        before = _signals_budget[0]
        if _ensure_signals_cache(r[0], uuid_slug.get(r[0], ""), _signals_budget) is not None \
                and _signals_budget[0] < before:
            _signals_built += 1
    if _signals_built or _signals_budget[0] < MAX_NEW_SIGNALS_PER_RUN:
        print(f"signals-cache: built {_signals_built} new decode-signal "
              f"cache(s) this run (budget {MAX_NEW_SIGNALS_PER_RUN}/run); "
              f"{sum(1 for r in test_recs if _signals_cache_path(r[0]))}/"
              f"{len(test_recs)} test recs now have one", flush=True)

    # Label-hygiene pass (Stufe 2): use the existing head.bin as a
    # teacher to drop frames where labels and teacher strongly
    # disagree. Frames likely-mislabelled (auto-detect boundary off,
    # ROI smear, etc.) get masked out instead of poisoning the next
    # head. Capped per-recording so a broken teacher can't nuke
    # everything.
    teacher_w = teacher_b = None
    teacher_mlp = None        # v2/v3/v4 MLP teacher (predict_proba) when available
    teacher_chan_idx = None   # slug→col from the TEACHER's own channel-map
    teacher_whisper = False
    teacher_temporal = False
    teacher_churn = False
    teacher_mask = False
    teacher_mp_col = None     # v4 teacher: minute-prior closure from ITS sidecar
    feat_dim = per_rec[0][3].shape[1] if per_rec else 0
    if args.hygiene_disagree_conf > 0 and Path(args.output).exists():
        try:
            raw = Path(args.output).read_bytes()
            mlp = load_deployed_mlp(args.output)
            if mlp is not None:
                # v2/v3 MLP teacher. It scores on channel-one-hot(+whisper)
                # (+temporal) augmented features, so we rebuild them with the
                # TEACHER's OWN channel-map (one-hot order is run-specific) —
                # from the head.channel-map.json sidecar next to the head.
                # Derive whisper/temporal presence from the dim budget; ANY
                # mismatch (changed logo/audio flags, channel set, or a
                # missing map) → skip cleanly rather than feed a misaligned
                # vector.
                cmap_path = Path(args.output).with_name(
                    Path(args.output).stem + ".channel-map.json")
                slugs = (json.loads(cmap_path.read_text()).get("slugs", [])
                         if cmap_path.exists() else [])
                n_chan = len(slugs)
                # ⚠️ Die Zusatzspalten werden GERECHNET, nicht gegen eine
                # Liste bekannter Formen geprueft. Die Liste war der Fehler:
                # als der v5-Kopf (Whisper-Maske) dazukam, passte keine
                # Form mehr, der Lehrer fiel auf "unalignable" und der
                # Hygiene-Durchlauf lief still ohne ihn weiter — kein
                # Absturz, nur eine stumm abgeschaltete Pruefung.
                #
                # Reihenfolge der Zusatzspalten (= Header-Vertrag):
                #   whisper(1) temporal(2 oder 3) minuteprior(1) maske(1)
                _extra = mlp.input_dim - (feat_dim + n_chan)
                _bekannt = {
                    0: (False, False, False, False),
                    1: (True, False, False, False),   # v2
                    3: (True, True, False, False),    # v3
                    4: (True, True, True, False),     # v4
                    5: (True, True, True, True),      # v5, temporal=2
                    6: (True, True, True, True),      # v5, temporal=3
                }
                if _extra not in _bekannt:
                    mlp = None  # dim budget doesn't add up → unalignable
                if mlp is not None:
                    (teacher_whisper, teacher_temporal,
                     _t_mp, teacher_mask) = _bekannt[_extra]
                    teacher_churn = (_extra == 6)
                if mlp is not None and _t_mp:
                    # minute-prior column from the teacher's OWN sidecar
                    # (deployed alongside its head.bin).
                    _mp_side = Path(args.output).with_suffix(
                        ".minute-prior.json")
                    _tpriors, _tneutral = {}, 0.25
                    try:
                        _side = json.loads(_mp_side.read_text())
                        _tpriors = {k: np.array(v, dtype=np.float32)
                                    for k, v in
                                    (_side.get("priors") or {}).items()}
                        _tneutral = float(_side.get("neutral", 0.25))
                    except Exception:
                        pass

                    def teacher_mp_col(uuid, T, _p=_tpriors, _n=_tneutral):
                        slug = uuid_slug.get(uuid, "")
                        start = uuid_start.get(uuid, 0)
                        if start and slug in _p:
                            minutes = ((start + np.arange(T)
                                        / args.fps_extract)
                                       // 60 % 60).astype(int)
                            return _p[slug][minutes].reshape(-1, 1)
                        return np.full((T, 1), _n, dtype=np.float32)
                if mlp is not None:
                    teacher_mlp = mlp
                    teacher_chan_idx = {s: i for i, s in enumerate(slugs)}
                    print(f"label-hygiene: v2/v3 MLP teacher loaded "
                          f"(input_dim={mlp.input_dim}, n_chan={n_chan}, "
                          f"whisper={teacher_whisper}, "
                          f"temporal={teacher_temporal}"
                          f"{', churn' if teacher_churn else ''}"
                          f"{', maske' if teacher_mask else ''})")
                else:
                    print(f"label-hygiene: v2/v3 MLP teacher unalignable "
                          f"(input_dim={load_deployed_mlp(args.output).input_dim}, "
                          f"feat_dim={feat_dim}, n_chan={n_chan}, "
                          f"map={'present' if cmap_path.exists() else 'MISSING'}) "
                          f"— skipping")
            else:
                # Legacy linear (logreg) teacher: size-detected flat weights.
                cand_dims = {1280 + L + C + A
                             for L in (0, 1)
                             for C in (0, len(CHANNELS))
                             for A in (0, 1)}
                for cand_dim in sorted(cand_dims):
                    if len(raw) == (cand_dim + 1) * 4 and cand_dim == feat_dim:
                        teacher_w = np.frombuffer(raw[:cand_dim*4],
                                                  dtype=np.float32)
                        teacher_b = struct.unpack("<f", raw[cand_dim*4:])[0]
                        break
                if teacher_w is None:
                    print(f"label-hygiene: teacher {len(raw)}B incompatible "
                          f"with current feat_dim={feat_dim} — skipping")
        except Exception:
            teacher_mlp = teacher_w = None
    keep_masks = []
    drops_total = drops_kept = 0
    user_veto_total = user_veto_recs = 0
    for r in train_recs:
        n = len(r[4])
        # Pseudo-labelled recordings: their frame_mask (r[10]) marks
        # which frames carry an opinion. Frames without pseudo labels
        # must be excluded — they have no truth to learn from. Skip
        # the teacher-disagree hygiene filter for pseudo recordings;
        # the conf+prior filter at write-time already gates them.
        is_pseudo = len(r) > 11 and r[11]
        if is_pseudo:
            mask = r[10] if r[10] is not None else np.ones(n, dtype=bool)
        elif teacher_mlp is not None or teacher_w is not None:
            if teacher_mlp is not None:
                Xa = _augment_teacher_feats(r[3], uuid_slug.get(r[0], ""),
                                            teacher_chan_idx, r[0], teacher_whisper,
                                            teacher_temporal, teacher_mp_col,
                                            teacher_churn, teacher_mask)
                proba = teacher_mlp.predict_proba(Xa)[:, 1]
            else:
                logits = r[3] @ teacher_w + teacher_b
                proba = 1.0 / (1.0 + np.exp(-logits))
            # disagreement: label=1 but proba<(1-conf), or label=0 but proba>conf
            disagree = (((r[4] == 1) & (proba < 1 - args.hygiene_disagree_conf)) |
                        ((r[4] == 0) & (proba >     args.hygiene_disagree_conf)))
            drop_rate = disagree.mean()
            if r[5]:
                # User-reviewed recordings are exempt: their labels are
                # ground truth (same trust hierarchy as the 2× user-weight
                # below). Reviews happen precisely where the deployed head
                # was confidently wrong, so a champion-teacher veto here
                # censors exactly the corrections the challenger needs —
                # the challenger then can't outlearn the champion and the
                # head-to-head gate deadlocks (2026-07-02..06: 5 straight
                # rejections after 4 reviews landed).
                mask = np.ones(n, dtype=bool)
                if disagree.any():
                    user_veto_total += int(disagree.sum())
                    user_veto_recs += 1
            elif drop_rate > args.hygiene_max_drop_rate:
                # teacher likely wrong, not labels — keep everything
                mask = np.ones(n, dtype=bool)
            else:
                mask = ~disagree
                drops_total += int(disagree.sum())
                drops_kept += 1
        else:
            mask = np.ones(n, dtype=bool)
        keep_masks.append(mask)
    if (teacher_mlp is not None or teacher_w is not None) and drops_kept > 0:
        print(f"label-hygiene: dropped {drops_total} frames across "
              f"{drops_kept} recordings (teacher disagreed at conf "
              f">{args.hygiene_disagree_conf})")
    if user_veto_total > 0:
        print(f"label-hygiene: kept {user_veto_total} teacher-disputed frames "
              f"across {user_veto_recs} user-reviewed recordings (user labels win)")

    X_train_parts, y_train_parts, sw_train_parts = [], [], []
    confirmed_extra_w = 0
    bumper_boost_total = 0
    bumper_boost_recs = set()
    for r, mask in zip(train_recs, keep_masks):
        X_train_parts.append(r[3][mask])
        y_train_parts.append(r[4][mask])
        # Per-frame sample weights: frames from user-confirmed
        # recordings carry --user-weight× the influence of frames from
        # auto-only recordings (default 2×). User-eyeballed labels are
        # higher quality; auto labels are noisy at boundaries and
        # prone to template mishaps. Pseudo-labelled frames sit
        # below auto at args.pseudo_weight (default 0.3×) — high-acc
        # but low-volume signal that should not dominate.
        is_pseudo = len(r) > 11 and r[11]
        if is_pseudo:
            base_w = args.pseudo_weight
        else:
            base_w = args.user_weight if r[5] else 1.0
        # Age decay: recent recordings reflect current channel ad
        # patterns (banner styles, sponsor slates change over months).
        # Linear ramp 1.0 → 0.5 over [0, 90] days, then 0.5 → 0 over
        # [90, 180]. Beyond 180d the recording contributes nothing.
        age_d = r[8] if len(r) > 8 else 0
        if age_d > 180:
            age_mult = 0.0
        elif age_d > 90:
            age_mult = 0.5 * (180 - age_d) / 90.0
        else:
            age_mult = 1.0 - 0.5 * age_d / 90.0
        base_w *= age_mult
        if base_w <= 0:
            # Skip entirely — adds rows with weight 0 confuses sklearn
            sw_train_parts.append(np.empty(0, dtype=np.float32))
            X_train_parts[-1] = X_train_parts[-1][:0]
            y_train_parts[-1] = y_train_parts[-1][:0]
            continue
        # confirmed_show frames (set by /api/recording/<uuid>/mark-reviewed)
        # are explicit "this is show, model was wrong to be unsure"
        # negative labels — bonus weight on top of the recording's
        # base weight. These directly target the active-learning
        # uncertainty that surfaced them, so they have outsized
        # impact per sample.
        confirmed = r[6] if len(r) > 6 else []
        skip_confirms = r[7] if len(r) > 7 else []
        full_n = len(r[4])
        sw_full = np.full(full_n, base_w, dtype=np.float32)
        if confirmed:
            for t in confirmed:
                idx = int(round(t * args.fps_extract))
                if 0 <= idx < full_n:
                    sw_full[idx] = base_w * 1.2
                    confirmed_extra_w += 1
        # Skip-press signals: force label=1 at that frame (user
        # confirmed an ad block was real by skipping it) and bump
        # weight to 1.5× — slightly stronger than confirmed_show
        # because skip is a more deliberate user action.
        if skip_confirms:
            yslice = r[4]
            for t in skip_confirms:
                idx = int(round(t * args.fps_extract))
                if 0 <= idx < full_n:
                    yslice[idx] = 1
                    sw_full[idx] = base_w * 1.5
        # Bumper-confirmed boundaries: if an ads.json boundary has a
        # bumper within ±2 s, the ±2 s frame window around the boundary
        # gets bumper_boost× weight. Independent positive evidence that
        # the boundary is real.
        bumpers = r[9] if len(r) > 9 else []
        if bumpers and args.bumper_boost > 1.0:
            bumper_ts = [b["t"] for b in bumpers]
            radius_s = 2.0
            radius_f = int(round(radius_s * args.fps_extract))
            for s, e in r[2]:
                for edge in (s, e):
                    if any(abs(edge - bt) <= radius_s for bt in bumper_ts):
                        i0 = max(0, int(round(edge * args.fps_extract)) - radius_f)
                        i1 = min(full_n, int(round(edge * args.fps_extract)) + radius_f + 1)
                        sw_full[i0:i1] = np.maximum(
                            sw_full[i0:i1], base_w * args.bumper_boost)
                        bumper_boost_total += (i1 - i0)
                        bumper_boost_recs.add(r[0])
        # Cluster-anchored ad spots: each entry is a time span whose
        # audio+visual fingerprint matched a known ≥3-member family.
        # Force label=1 in that span and boost weight to 1.5× — same
        # treatment as user skip-presses. For unreviewed recordings
        # (= no ads_user.json) this is the only high-confidence ad
        # signal we have.
        cluster_anchored = r[13] if len(r) > 13 else []
        cluster_anchor_frames = 0
        if cluster_anchored:
            yslice = r[4]
            for spot in cluster_anchored:
                try:
                    s_t = float(spot.get("window_start_s") or 0)
                    e_t = float(spot.get("end_s") or s_t)
                except Exception:
                    continue
                i0 = max(0, int(round(s_t * args.fps_extract)))
                i1 = min(full_n, int(round(e_t * args.fps_extract)) + 1)
                if i1 <= i0:
                    continue
                yslice[i0:i1] = 1
                sw_full[i0:i1] = np.maximum(sw_full[i0:i1], base_w * 1.5)
                cluster_anchor_frames += (i1 - i0)
        if cluster_anchor_frames:
            globals().setdefault("_cluster_anchor_total", [0])[0] = (
                globals().get("_cluster_anchor_total", [0])[0]
                + cluster_anchor_frames)
        # NaN-logo skip: frames where logo extraction failed (= NaN
        # before substitute-with-0.5) contribute weight 0 to the fit.
        # Without this the head sees "real 0.5" and "fallback 0.5" as
        # identical, learns wrong patterns from the corrupt-stream
        # frames. Trade-off: backbone signal for those frames is also
        # discarded — acceptable because the recording's measured
        # frames still cover most of the show / ad supervision.
        nan_mask = logo_nan_mask_by_uuid.get(r[0])
        if nan_mask is not None and len(nan_mask) == full_n:
            sw_full[nan_mask] = 0.0
        sw_train_parts.append(sw_full[mask])
    if confirmed_extra_w:
        print(f"confirmed-show: upweighted {confirmed_extra_w} frame(s) "
              f"from /mark-reviewed (1.2× over base weight)")
    _ca_total = globals().get("_cluster_anchor_total", [0])[0]
    if _ca_total:
        print(f"cluster-anchored: forced ad + 1.5× weight on "
              f"{_ca_total} frame(s)")
    n_skip = sum(len(r[7]) if len(r) > 7 else 0 for r in train_recs)
    if n_skip:
        print(f"skip-press signals: {n_skip} confirmed-ad frame(s) "
              f"(label=1, 1.5× weight)")
    if bumper_boost_total:
        print(f"bumper-confirmed boundaries: {bumper_boost_total} frame(s) "
              f"boosted across {len(bumper_boost_recs)} recording(s) "
              f"({args.bumper_boost}× weight)")
    X_train = np.concatenate(X_train_parts) if X_train_parts else np.empty((0, per_rec[0][3].shape[1]))
    y_train = np.concatenate(y_train_parts) if y_train_parts else np.empty(0)
    sw_train = np.concatenate(sw_train_parts) if sw_train_parts else np.empty(0)
    n_user = sum(1 for r in train_recs if r[5])
    print(f"\nsplit: {len(train_recs)} train recs ({len(y_train)} frames, "
          f"{100*y_train.mean():.1f}% ad, {n_user} user-confirmed @ "
          f"weight {args.user_weight}×), "
          f"{len(test_recs)} test recs")

    from sklearn.linear_model import LogisticRegression
    # No class_weight: tested 'balanced' which made the model MORE
    # eager to call ad. False-positive penalisation is handled at
    # inference time (NNWeight, NNGate) instead.
    clf = LogisticRegression(max_iter=2000, C=1.0, verbose=0)
    clf.fit(X_train, y_train, sample_weight=sw_train)
    train_pred = clf.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    # NOT the deployed model's accuracy — this is the internal LogReg
    # baseline, fit unconditionally regardless of --head-arch purely as
    # a reference point for the shadow table / historical continuity.
    # It's also train-set (not held-out) accuracy, so it doesn't even
    # measure the same thing as the metrics that matter. Look for
    # "PRODUCTION METRIC" near the DEPLOYED/REJECTED line instead.
    print(f"LogReg baseline (internal reference, NOT deployed) train acc "
          f"{train_acc*100:.1f}%  L2={float(np.linalg.norm(clf.coef_)):.2f}  "
          f"bias={float(clf.intercept_[0]):+.3f}")

    # ── Platt calibration ────────────────────────────────────────
    # Fit σ(A * logit + B) on the HOLD-OUT test set against true
    # labels. Logistic-regression probabilities tend to be
    # over-confident: a frame the model says is 95 % ad is wrong
    # more than 5 % of the time. Calibrating maps the raw logits
    # to probabilities that actually equal the empirical hit-rate.
    # The active-learning step downstream uses these calibrated
    # probs to surface frames where the model is GENUINELY unsure
    # (not just close to 0.5 by accident of the linear-head's
    # over-confidence). Saved as head.calibration.json sidecar
    # next to head.bin — Go detector reads weights as before, the
    # sidecar is opt-in for code paths that want calibrated output.
    calibration = None
    if test_recs:
        X_test_parts = []; y_test_parts = []
        for uuid, title, ads, X, y, *_ in test_recs:
            if X is None or len(X) == 0:
                continue
            expected_dim = clf.coef_.shape[1]
            if X.shape[1] < expected_dim:
                X = np.concatenate([X, np.full(
                    (X.shape[0], expected_dim - X.shape[1]),
                    0.5, dtype=X.dtype)], axis=1)
            X_test_parts.append(X)
            y_test_parts.append(y)
        if X_test_parts:
            X_test = np.concatenate(X_test_parts)
            y_test = np.concatenate(y_test_parts)
            logits_test = clf.decision_function(X_test)
            # Single-feature LR fits A and B in σ(A*logit + B).
            # Reshape (n,) → (n,1) so sklearn treats it as 1-D feature.
            cal_clf = LogisticRegression(max_iter=2000, C=1e6, verbose=0)
            cal_clf.fit(logits_test.reshape(-1, 1), y_test)
            A = float(cal_clf.coef_[0][0])
            B = float(cal_clf.intercept_[0])
            # Quality: Brier score before vs after. Brier = mean
            # squared error of predicted prob vs label; calibration
            # SHOULD improve it (lower = better).
            raw_proba = 1.0 / (1.0 + np.exp(-logits_test))
            cal_proba = 1.0 / (1.0 + np.exp(-(A * logits_test + B)))
            brier_raw = float(np.mean((raw_proba - y_test) ** 2))
            brier_cal = float(np.mean((cal_proba - y_test) ** 2))
            calibration = {
                "method": "platt",
                "head_arch": "logreg",
                "A": A, "B": B,
                "n_calibration_frames": int(len(y_test)),
                "n_calibration_recs": len(test_recs),
                "brier_raw": brier_raw,
                "brier_calibrated": brier_cal,
                "improvement": brier_raw - brier_cal,
                # applied: whether consumers should actually USE A/B.
                # Root-caused 2026-07-14: Platt fits log-loss, not Brier
                # — on an already-well-calibrated head it consistently
                # SOFTENS confidences (A≈0.7) and worsens Brier (✗ in
                # 5/5 recent nightly runs, both the LogReg and the MLP
                # variant). A calibration that measurably degrades the
                # probability quality must not be applied; keep the fit
                # in the sidecar for diagnostics/tracking only.
                "applied": brier_cal < brier_raw,
            }
            arrow = "✓" if brier_cal < brier_raw else "✗"
            print(f"calibration: Platt A={A:+.3f} B={B:+.3f}  "
                  f"Brier {brier_raw:.4f} → {brier_cal:.4f} {arrow}"
                  + ("" if brier_cal < brier_raw else
                     "  (not applied — raw probs are better calibrated)"))

    # Freeze the LOGREG calibration for calibrated_proba NOW — the
    # `calibration` name gets overwritten later by Phase D with
    # MLP-fitted constants (late-binding closure bug found 2026-07-14:
    # calibrated_proba applied MLP A/B to LogReg logits). Only frozen
    # when it actually improved Brier, else None → raw probabilities.
    _logreg_cal = (dict(calibration)
                   if calibration and calibration["applied"] else None)

    def calibrated_proba(X):
        """Apply Platt scaling on top of clf (LogReg) logits when the
        LogReg-fitted calibration is available AND improved Brier;
        fall back to raw predict_proba otherwise. Used by the
        active-learning surface step so 'uncertainty' reflects the
        model's true uncertainty."""
        if _logreg_cal is None:
            return clf.predict_proba(X)[:, 1]
        z = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-(_logreg_cal["A"] * z + _logreg_cal["B"])))

    # Evaluate on held-out recordings — both raw (matches a deploy
    # without --nn-smooth) and 10s-smoothed (matches the new default).
    metrics_smooth = None
    deployed_test_metrics = None  # deployed head re-scored on this test set (head-to-head gate)
    if test_recs:
        eval_split(clf, test_recs, args.fps_extract, smooth_s=0)
        metrics_smooth = eval_split(clf, test_recs, args.fps_extract,
                                    smooth_s=10)
        if args.emit_confusion:
            confusion_analysis(clf, test_recs, args.fps_extract,
                               smooth_s=10,
                               output_path=Path(args.output).with_suffix(".confusion.txt"))

    # ── Production MLP+channel fit (--head-arch mlp32-channel) ────
    # Fits the WeightedMLP that will replace the LogReg head.bin if
    # --head-arch=mlp32-channel. Reuses the production X_train +
    # y_train + sw_train (= already hygiene-masked + age-decayed +
    # bumper-boosted) and appends a channel one-hot block post-hoc.
    # sw_train is applied as TRUE fractional sample_weight (until
    # 2026-07-07 it was integer-rounded row duplication, which erased
    # pseudo-0.3/age-decay/bumper-boost — see WeightedMLP docstring).
    # Eval on the channel-augmented test set; the resulting metrics
    # override metrics_smooth so the deploy-decision branch downstream
    # operates on what we'll actually ship, not on the LogReg baseline.
    mlp_prod_clf = None
    mlp_gate_clf = None  # train-only head snapshot for the honest head-to-head gate
    mlp_prod_chan_slugs = None
    mlp_prod_in_dim = 0
    wants_mlp = args.head_arch in ("mlp32-channel",
                                    "mlp32-channel-whisper",
                                    "mlp32-channel-whisper-temporal",
                                    "mlp32-channel-whisper-temporal-mp",
                                    "mlp32-channel-whisper-temporal-mp-wm")
    wants_whisper = args.head_arch in ("mlp32-channel-whisper",
                                       "mlp32-channel-whisper-temporal",
                                       "mlp32-channel-whisper-temporal-mp",
                                       "mlp32-channel-whisper-temporal-mp-wm")
    wants_temporal = args.head_arch in ("mlp32-channel-whisper-temporal",
                                        "mlp32-channel-whisper-temporal-mp",
                                        "mlp32-channel-whisper-temporal-mp-wm")
    wants_minuteprior = args.head_arch in ("mlp32-channel-whisper-temporal-mp",
                                          "mlp32-channel-whisper-temporal-mp-wm")
    # ⚠️ Die Whisper-Indikatorspalte. Siehe _whisper_present: sie trennt
    # "keine Ton-Daten vorhanden" von "Ton sagt 50/50" — ohne sie sind
    # beide konstant 0.5, und das betraf am 2026-08-06 die Haelfte des
    # Korpus.
    wants_whispermask = args.head_arch == "mlp32-channel-whisper-temporal-mp-wm"
    # Die Unruhe-Spalte haengt an DIESER Architektur, nicht an wants_temporal:
    # die aelteren Archs schreiben n_temporal=2 in den Header, bekaemen aber
    # sonst drei Spalten — der Header-Vertrag flöge auf.
    wants_churn = wants_whispermask
    # Corpus-wide neutral fill for the minute-prior column (recordings
    # with no start_ts / channels with no histogram): mean of all prior
    # buckets ≈ base ad rate, so the column carries no signal instead of
    # a false one. Persisted in the sidecar so Go uses the SAME value.
    #
    # ⚠️ Wird berechnet, sobald eine Prior-Tabelle da IST — nicht erst, wenn
    # der laufende --head-arch die Spalte will. Die Schatten-Sonde
    # _augment_cwt_minuteprior benutzt sie naemlich auch dann, und bis
    # 2026-08-07 hatte sie dafuer einen eigenen, anders berechneten
    # Neutralwert. Zwei Definitionen desselben Fuellwerts sind genau die
    # Sorte Naht, die spaeter still auseinanderlaeuft.
    mp_neutral = 0.25
    if wants_minuteprior and not minute_prior:
        raise SystemExit("--head-arch mlp32-channel-whisper-temporal-mp "
                         "requires --with-minute-prior (the prior table "
                         "is the feature source)")
    if minute_prior:
        _all_p = [v for arr in minute_prior.values() for v in arr]
        if _all_p:
            mp_neutral = float(sum(_all_p) / len(_all_p))

    def _minuteprior_col(uuid, T):
        """Per-frame P(ad | minute-of-hour) column for one recording —
        THE production feature definition; every consumer (train fit,
        eval augment, all-data refit, shadow, teacher) must go through
        here so train/serve columns can't drift apart."""
        slug = uuid_slug.get(uuid, "")
        start = uuid_start.get(uuid, 0)
        if start and slug in minute_prior:
            prior_arr = np.array(minute_prior[slug], dtype=np.float32)
            minutes = ((start + np.arange(T) / args.fps_extract)
                       // 60 % 60).astype(int)
            return prior_arr[minutes].reshape(-1, 1)
        return np.full((T, 1), mp_neutral, dtype=np.float32)
    if wants_mlp and test_recs:
        prod_chan_slugs = sorted({uuid_slug.get(r[0], "")
                                   for r in train_recs + test_recs} - {""})
        prod_chan_idx = {s: i for i, s in enumerate(prod_chan_slugs)}
        n_chan = len(prod_chan_slugs)
        # Per-rec frame counts in the post-hygiene X_train/y_train.
        # The X_train_parts loop above appended (or zeroed) for each
        # entry of train_recs in order, so the parallel zip stays
        # consistent here.
        rec_lengths_train = [len(p) for p in y_train_parts]

        def _prod_zusatz(X, uuid, chan_idx=None, n=None):
            """Der Zusatzblock in der Auspraegung des laufenden --head-arch.
            EINE Stelle fuer Fit, Auswertung, Ablation und All-Data-Refit —
            damit keiner der vier Pfade eine andere Eingabe sieht als die
            anderen."""
            return zusatzspalten(
                X, uuid, uuid_slug.get(uuid, ""),
                prod_chan_idx if chan_idx is None else chan_idx,
                n_chan if n is None else n,
                whisper=wants_whisper, temporal=wants_temporal,
                churn=wants_churn,
                mp_col=_minuteprior_col if wants_minuteprior else None,
                maske=wants_whispermask)

        # Der komplette Zusatzblock, pro Aufnahme auf dem ROHEN X gebaut
        # und danach mit derselben Hygiene-Maske gesiebt wie die Basis.
        #
        # ⚠️ Vor dem 2026-08-07 stand hier ein Block je SPALTE, ueber alle
        # Aufnahmen verkettet. Fuer die Kanal-, Temporal-, Minute-Prior- und
        # Masken-Spalten war das gleichwertig — fuer die Whisper-Spalte NICHT:
        # sie wurde mit der NACH-Masken-Frame-Zahl geholt und ungefiltert in
        # den maskierten Bereich geschrieben. _load_whisper_per_sec indiziert
        # aber nach ABSOLUTER Sekunde, also war die Spalte um die Zahl der bis
        # dahin verworfenen Frames verschoben und das Ende der Aufnahme fehlte
        # ganz. Betroffen: jede Aufnahme mit loechriger Maske — im Lauf vom
        # 2026-08-07 waren das 168 von 283. Die Auswertung (_aug_test) und die
        # Go-Seite hatten die Spalte immer richtig; der Fit lernte also auf
        # einer anderen Eingabe, als er spaeter bewertet wurde.
        # Test: scripts/test_zusatzspalten.py, Klasse MaskenVersatz.
        zusatz_parts = []
        for r, mask, n in zip(train_recs, keep_masks, rec_lengths_train):
            if n <= 0:
                continue
            zusatz_parts.append(_prod_zusatz(r[3], r[0])[mask])
        zusatz_train = (np.concatenate(zusatz_parts) if zusatz_parts
                        else np.zeros((len(X_train), 0), dtype=np.float32))
        X_train_ch = np.hstack([X_train, zusatz_train])
        print(f"\n=== --head-arch {args.head_arch}: production fit ===")
        print(f"  base train dim: {X_train_ch.shape[1]} "
              f"({n_chan} channels"
              f"{', +whisper' if wants_whisper else ''}"
              f"{', +temporal' if wants_temporal else ''})")
        print(f"  base train frames: {len(X_train_ch)} "
              f"(true sample_weight, no oversample)")
        mlp_prod_clf = WeightedMLP(hidden_dim=32, max_iter=200,
                                   random_state=0)
        mlp_prod_clf.fit(X_train_ch, y_train, sw_train)
        print(f"  fit done in {mlp_prod_clf.n_iter_} epochs, "
              f"loss={mlp_prod_clf.loss_:.4f}")

        # Augment test_recs with the same channel one-hot (and
        # whisper/temporal if active) for eval.
        def _aug_test(recs):
            out = []
            for r in recs:
                X_new = np.hstack([r[3], _prod_zusatz(r[3], r[0])]
                                  ).astype(np.float32)
                out.append((r[0], r[1], r[2], X_new, r[4]) + tuple(r[5:]))
            return out
        test_recs_ch = _aug_test(test_recs)
        print(f"\n=== {args.head_arch} held-out evaluation (smooth=10s) ===")
        eval_split(mlp_prod_clf, test_recs_ch, args.fps_extract, smooth_s=0)
        metrics_smooth_mlp = eval_split(mlp_prod_clf, test_recs_ch,
                                         args.fps_extract, smooth_s=10)
        # Override the deploy-decision metric — the deploy block
        # downstream uses metrics_smooth to compare against the last
        # deployed run; what gets compared must match what gets
        # written to head.bin.
        metrics_smooth = metrics_smooth_mlp
        # Minute-prior ablation diagnostic: re-score the SAME candidate head on a
        # copy of the test set where the minute-of-hour column (appended LAST by
        # _aug_test) is flattened to its neutral fill. The IoU delta is the column's
        # marginal contribution — confirms the MLP4 migration earns its complexity
        # (or flags it as dead weight). Read-only; never touches the deploy.
        if args.ablate_minute_prior and wants_minuteprior:
            def _med_iou(m):
                return m.get("iou_tv_median",
                             m.get("iou_median", m.get("iou", 0.0)))
            # ⚠️ Die Spalte wird GERECHNET, nicht als "die letzte" angenommen.
            # Bis 2026-08-07 stand hier Xa[:, -1]. Das stimmte fuer MLP4 —
            # aber seit dem v5-Kopf (2026-08-06) sitzt GANZ HINTEN die
            # Whisper-Maske, und der Minute-Prior davor. Die Ablation hat
            # also eine Nacht lang die Maske flachgelegt und das Ergebnis
            # als "minute-prior ABLATION" ueberschrieben. Die aelteren
            # Zahlen der 8-Naechte-Serie (Δ≈0) sind davon nicht betroffen,
            # die liefen alle vor v5.
            mp_idx = -2 if wants_whispermask else -1
            ablated = []
            for r in test_recs_ch:
                Xa = r[3].copy()
                Xa[:, mp_idx] = mp_neutral
                ablated.append((r[0], r[1], r[2], Xa, r[4]) + tuple(r[5:]))
            print("\n=== minute-prior ABLATION (candidate, column neutralised) ===")
            m_abl = eval_split(mlp_prod_clf, ablated, args.fps_extract, smooth_s=10)
            _with, _without = _med_iou(metrics_smooth_mlp), _med_iou(m_abl)
            # Per-rec: how many recs the column actually moves, and by how much.
            _pw = metrics_smooth_mlp.get("per_rec_iou") or {}
            _po = m_abl.get("per_rec_iou") or {}
            _sh = [u for u in _pw if u in _po]
            _moved = sorted(((_pw[u] - _po[u], u) for u in _sh
                             if abs(_pw[u] - _po[u]) > 0.02),
                            key=lambda x: -abs(x[0]))
            print(f"  median-IoU with minute-prior {_with:.3f}  vs neutralised "
                  f"{_without:.3f}  → Δ {_with - _without:+.3f}")
            print(f"  recs moved >0.02 by the column: {len(_moved)}/{len(_sh)}")
            for d, u in _moved[:8]:
                t, c = uuid_cohort.get(u, ("", ""))
                lbl = f"{t} / {c}" if t else u
                print(f"    {d:+.3f}  {lbl}  ({u})")
        mlp_prod_chan_slugs = prod_chan_slugs
        mlp_prod_in_dim = X_train_ch.shape[1]
        # Snapshot the TRAIN-ONLY head now, before --final-on-all rebinds
        # mlp_prod_clf to the all-data refit. This honest (held-out) head is
        # written next to head.bin as head.gate.bin and is what the NEXT run's
        # head-to-head re-scores — so the gate compares train-only-candidate vs
        # train-only-champion (both honest) instead of train-only-candidate vs
        # all-data-champion (which trained on the test set → inflated, the
        # ~0.04 memorisation bias that kept rejecting good candidates).
        mlp_gate_clf = mlp_prod_clf

        # Head-to-head: re-score the CURRENTLY-DEPLOYED head (args.output still
        # holds it — the candidate is written only after the gate) on this exact
        # augmented test set, so the deploy gate can compare both heads on
        # identical data instead of against a historical IoU floor anchored to a
        # different test set. Requires a same-dim v2 MLP AND the same channel
        # one-hot layout — a same-COUNT but reordered channel-map (one channel
        # gains labels while another loses them) would misalign the columns and
        # make the deployed re-eval garbage, so verify the slug list matches
        # (head.channel-map.json sidecar) before trusting it; otherwise skip and
        # fall back to the floor logic below.
        if test_recs_ch:
            # Re-score the deployed champion's TRAIN-ONLY gate head (head.gate.bin)
            # when present — it never trained on the test set, so its score is
            # honest and comparable to the (also train-only) candidate. Fall back
            # to head.bin (the all-data, test-inflated head) only until the first
            # deploy under this scheme has written a head.gate.bin.
            _gate_path = Path(args.output).with_suffix(".gate.bin")
            _dep_src = _gate_path if _gate_path.exists() else Path(args.output)
            _dep = load_deployed_mlp(_dep_src)
            print(f"  head-to-head champion source: {_dep_src.name}")
            _dep_slugs = None
            _cm = Path(args.output).with_suffix(".channel-map.json")
            if _cm.exists():
                try:
                    _dep_slugs = json.loads(_cm.read_text()).get("slugs")
                except Exception:
                    _dep_slugs = None
            if (_dep is not None and _dep.input_dim == mlp_prod_in_dim
                    and _dep_slugs == prod_chan_slugs):
                print("\n=== deployed-head re-eval (head-to-head, smooth=10s) ===")
                deployed_test_metrics = eval_split(_dep, test_recs_ch,
                                                   args.fps_extract, smooth_s=10)
            elif _dep is not None and _dep.input_dim == mlp_prod_in_dim:
                print("  head-to-head skipped: channel-map differs from the "
                      "deployed head — falling back to the historical floor")
            elif _dep is not None:
                # ⚠️ Dieser Fall war bis 2026-08-06 STILL. Er tritt bei jedem
                # Architekturwechsel ein (input_dim des Kandidaten passt nicht
                # zum deployten Kopf) — also genau in der Nacht, in der man am
                # ehesten wissen will, dass der paarweise Schutz gerade nicht
                # greift. Ohne Meldung sah das Log aus wie ein normaler Lauf.
                print(f"  head-to-head SKIPPED: deployed head has input_dim "
                      f"{_dep.input_dim}, candidate {mlp_prod_in_dim} — "
                      f"Architekturwechsel. Es schuetzen nur noch der "
                      f"historische IoU-Boden und der Golden-Boden.")

        # --- Holdout overfit diagnostic (env HOLDOUT_REPORT_UUIDS) -----------
        # Read-only: score the deployed champion vs this run's candidate on a
        # set of recordings the CHAMPION never trained on (e.g. reviews added
        # after it was anchored). Answers "is the champion genuinely better, or
        # overfit to its frozen split?" Prints only — changes no deploy decision.
        _ho = os.environ.get("HOLDOUT_REPORT_UUIDS", "").strip()
        if _ho and test_recs_ch:
            ho_uuids = {u.strip() for u in _ho.split(",") if u.strip()}
            _depH = load_deployed_mlp(args.output)
            if _depH is not None and _depH.input_dim == mlp_prod_in_dim:
                test_uuids = {r[0] for r in test_recs}
                ho_aug = _aug_test([r for r in per_rec if r[0] in ho_uuids])
                hw = int(10 * args.fps_extract / 2)

                def _iou_of(head, r):
                    proba = head.predict_proba(r[3])[:, 1]
                    proba = smooth_mean(proba, hw)
                    pred = (proba >= 0.5).astype(np.int32)
                    gt = [(float(a[0]), float(a[1])) for a in r[2]]
                    return block_iou(to_blocks(pred, fps=args.fps_extract), gt)

                print("\n=== HOLDOUT overfit report (champion never trained "
                      "on these) ===")
                print(f"{'uuid':18}{'split':6}{'champ':>6}{'cand':>6}  show")
                rows = []
                for r in ho_aug:
                    cj, kj = _iou_of(_depH, r), _iou_of(mlp_prod_clf, r)
                    insplit = "test" if r[0] in test_uuids else "train"
                    rows.append((insplit, cj, kj))
                    print(f"{r[0][:18]:18}{insplit:6}{cj:6.2f}{kj:6.2f}  "
                          f"{r[1][:32]}")
                champ_all = [c for _, c, _ in rows]
                tt = [(c, k) for s, c, k in rows if s == "test"]
                if champ_all:
                    print(f"\nchampion median over ALL {len(champ_all)} holdout "
                          f"recs: {float(np.median(champ_all)):.3f}  "
                          f"(vs its frozen-split ~0.858)")
                if tt:
                    print(f"holdout∩test ({len(tt)} recs, NEITHER head trained): "
                          f"champion {float(np.median([c for c, _ in tt])):.3f} "
                          f"vs candidate "
                          f"{float(np.median([k for _, k in tt])):.3f}")
                print("read: champion holds ~0.85 on never-seen → genuine "
                      "(keep); drops a lot → overfit to frozen split "
                      "(reset-baseline justified)")

        # Phase D: Platt calibration on MLP logits. MLPClassifier
        # exposes predict_proba (= already sigmoid'd); recover logits
        # via the inverse: logit = log(p / (1-p)). Then fit the same
        # σ(A*logit + B) Platt model the LogReg path uses, override
        # the `calibration` dict so the existing sidecar-write block
        # downstream emits MLP-appropriate values. The gateway's
        # active-learning surface + future Go-side calibration paths
        # consume head.calibration.json without caring about the
        # underlying head architecture — same shape, MLP numbers.
        from sklearn.linear_model import LogisticRegression as _LR_for_cal
        X_test_concat = np.concatenate(
            [r[3] for r in test_recs_ch
             if r[3] is not None and len(r[3]) > 0])
        y_test_concat = np.concatenate(
            [r[4] for r in test_recs_ch
             if r[3] is not None and len(r[3]) > 0])
        if len(y_test_concat) > 0:
            proba_mlp = mlp_prod_clf.predict_proba(X_test_concat)[:, 1]
            # Clamp to (eps, 1-eps) so log(p/(1-p)) is finite even
            # when MLP saturates. eps=1e-6 keeps the tail of the
            # logit distribution sane (= log(1e6/1) ≈ 13.8) while
            # not drowning out genuine confidence with truncation.
            eps = 1e-6
            p_clip = np.clip(proba_mlp, eps, 1.0 - eps)
            logits_mlp = np.log(p_clip / (1.0 - p_clip))
            cal_mlp = _LR_for_cal(max_iter=2000, C=1e6, verbose=0)
            cal_mlp.fit(logits_mlp.reshape(-1, 1), y_test_concat)
            A_mlp = float(cal_mlp.coef_[0][0])
            B_mlp = float(cal_mlp.intercept_[0])
            raw_proba_mlp = proba_mlp
            cal_proba_mlp = 1.0 / (1.0 + np.exp(-(A_mlp * logits_mlp + B_mlp)))
            brier_raw_mlp = float(np.mean((raw_proba_mlp - y_test_concat) ** 2))
            brier_cal_mlp = float(np.mean((cal_proba_mlp - y_test_concat) ** 2))
            calibration = {
                "method": "platt",
                "head_arch": args.head_arch,
                "A": A_mlp, "B": B_mlp,
                "n_calibration_frames": int(len(y_test_concat)),
                "n_calibration_recs": len(test_recs_ch),
                "brier_raw": brier_raw_mlp,
                "brier_calibrated": brier_cal_mlp,
                "improvement": brier_raw_mlp - brier_cal_mlp,
                # See the LogReg calibration block for the rationale —
                # a fit that worsens Brier is diagnostics-only.
                "applied": brier_cal_mlp < brier_raw_mlp,
            }
            arrow = "✓" if brier_cal_mlp < brier_raw_mlp else "✗"
            print(f"\nMLP calibration: Platt A={A_mlp:+.3f} "
                  f"B={B_mlp:+.3f}  Brier "
                  f"{brier_raw_mlp:.4f} → {brier_cal_mlp:.4f} {arrow}"
                  + ("" if brier_cal_mlp < brier_raw_mlp else
                     "  (not applied — raw probs are better calibrated)"))

    # ── Shadow architecture comparison (--shadow-eval) ────────────
    # Compare 3 alternative head architectures against the production
    # LogReg on the SAME train/test split. Pure measurement — shadow
    # heads are NOT written, archived, deployed, or fed back into
    # history.json. Use the printed table to decide if a structural
    # head migration is worth pursuing.
    #
    # Simplification vs production fit: shadow ignores the hygiene
    # frame-mask + age-decay weighting + bumper boosts + skip-press
    # signals. Captures only `user_weight×` for ads_user-confirmed
    # recordings (the dominant signal). All three variants share the
    # same simplification → relative ordering between variants is
    # apples-to-apples; comparison vs the LogReg baseline carries
    # this caveat (LogReg used the full pipeline).
    if args.shadow_eval and test_recs and metrics_smooth:

        # Channel slug → idx (alphabetical for determinism, only slugs
        # actually present in train+test get a column — keeps the
        # one-hot tight at the corpus's true channel count).
        chan_slugs = sorted({uuid_slug.get(r[0], "")
                             for r in train_recs + test_recs} - {""})
        chan_idx = {s: i for i, s in enumerate(chan_slugs)}
        n_chan = len(chan_slugs)

        # Whisper-prob loader at module scope (_load_whisper_per_sec)
        # — both the shadow-eval variants here and the production
        # mlp32-channel-whisper fit downstream call it.

        # Die Schatten-Sonden. Jede ist eine Auspraegung desselben
        # Zusatzblocks — die Rechnung steht in zusatzspalten(), hier nur
        # noch welche Spalten die Sonde sehen soll.
        def _augment_channel(X, slug, _uuid=None):
            return mit_zusatz(X, _uuid, slug, chan_idx, n_chan)

        def _augment_temporal(X, slug, _uuid=None):
            # |X[t] - X[t-1]| und |X[t] - X[t+1]| als 2 Zusatzspalten,
            # OHNE Kanal-one-hot: Szenenwechsel-Signal, das die
            # Backbone-Merkmale je Sekunde nicht sehen koennen.
            return mit_zusatz(X, _uuid, slug, chan_idx, n_chan,
                              kanal=False, temporal=True)

        def _augment_channel_whisper(X, slug, uuid):
            # Whisper Stage 4: die Wahrscheinlichkeit je Sekunde direkt als
            # Eingabespalte statt als Nachbearbeitungsregel.
            return mit_zusatz(X, uuid, slug, chan_idx, n_chan, whisper=True)

        def _augment_channel_whisper_temporal(X, slug, uuid):
            # Produktions-Zwilling (2026-07-07 migriert). Die Deltas rechnen
            # auf dem rohen X der Aufnahme (zusammenhaengend, vor der Maske),
            # damit ein weggeworfener Frame keinen Szenenwechsel vortaeuscht.
            return mit_zusatz(X, uuid, slug, chan_idx, n_chan,
                              whisper=True, temporal=True, churn=wants_churn)

        def _augment_cwt_minuteprior(X, slug, uuid):
            # Produktions-Zwilling + P(Werbung | Minute der Stunde) als EINE
            # Spalte ganz hinten. Motivation (2026-07-15): 77 % der Fehler auf
            # dem Testsatz sind falsche "Werbung"-Rufe mitten in der Sendung;
            # deutsche Privatsender legen Werbung auf feste Minutenoffsets.
            #
            # ⚠️ Nutzt _minuteprior_col — dieselbe Definition wie der
            # Produktions-Fit. Bis 2026-08-07 stand hier eine zweite Kopie
            # mit einem EIGENEN Neutralwert (_mp_neutral), der anders
            # berechnet wurde als mp_neutral: die Sonde haette gegen eine
            # andere Spalte gemessen als die Produktion, sobald der laufende
            # --head-arch die Minute-Prior-Spalte nicht selbst wollte.
            return mit_zusatz(X, uuid, slug, chan_idx, n_chan,
                              whisper=True, temporal=True, churn=wants_churn,
                              mp_col=_minuteprior_col)

        def _augment_ct_minuteprior(X, slug, uuid):
            # DIESELBE Auspraegung wie _augment_cwt_minuteprior, nur OHNE die
            # Whisper-Spalte (1299 statt 1300). Das Paar der beiden isoliert
            # genau eine Spalte — alles andere (Kanal, dp/dn, Unruhe,
            # Minute-Prior, Gewichte, Testsatz) ist identisch.
            #
            # Anlass (2026-08-09): der Schattenvergleich zeigt zwei Naechte in
            # Folge, dass "+ channel + whisper" unter "+ channel" liegt
            # (-0.027 / -0.020) und dass die schlankste Variante (nur
            # Zeitdeltas) mit 0.946 ueber allem steht. Die Sonde
            # channel+whisper misst aber gegen eine Variante OHNE Zeitdeltas
            # und Minute-Prior — sie beantwortet nicht, ob die Spalte in der
            # HEUTIGEN Architektur noch traegt. Diese hier tut es.
            #
            # Die Whisper-MASKE faellt mit weg: sie sagt "Whisper-Daten
            # vorhanden ja/nein" und hat ohne die Wahrscheinlichkeitsspalte
            # keinen Bezug mehr (s. Memory whisper_luecke_und_indikatorspalte).
            return mit_zusatz(X, uuid, slug, chan_idx, n_chan,
                              temporal=True, churn=wants_churn,
                              mp_col=_minuteprior_col)

        def _build_train(recs, augment):
            # recs must be train_recs: rows are taken via the parallel
            # keep_masks / sw_train_parts arrays so the shadow variants
            # train with the SAME semantics as the production fit —
            # hygiene masks, pseudo frame_mask + pseudo_weight, age
            # decay, confirmed/skip/bumper/cluster boosts, NaN-logo=0.
            # (Until 2026-07-07 this rebuilt simplified weights itself:
            # pseudo recs entered unmasked at full weight and only
            # user-2× survived — the variant table compared apples to
            # slightly rotten apples.)
            assert recs is train_recs, "_build_train is keyed to train_recs"
            Xs, ys, sws = [], [], []
            for r, mask, sw in zip(recs, keep_masks, sw_train_parts):
                if len(sw) == 0:
                    continue  # age>180d — skipped entirely upstream
                slug = uuid_slug.get(r[0], "")
                X_aug = augment(r[3], slug, r[0])[mask]
                Xs.append(X_aug)
                ys.append(r[4][mask])
                sws.append(sw)
            return (np.concatenate(Xs), np.concatenate(ys),
                    np.concatenate(sws))

        def _augment_test_recs(recs, augment):
            out = []
            for r in recs:
                slug = uuid_slug.get(r[0], "")
                X_aug = augment(r[3], slug, r[0])
                out.append((r[0], r[1], r[2], X_aug, r[4]) + tuple(r[5:]))
            return out

        def _fit_eval(name, augment, hidden=(32,), seed=0):
            X_aug, y_aug, sw_aug = _build_train(train_recs, augment)
            test_aug = _augment_test_recs(test_recs, augment)
            in_dim, n_train_frames = X_aug.shape[1], len(X_aug)
            print(f"\n=== Shadow variant: {name} ===")
            print(f"  feature dim: {in_dim}, "
                  f"train frames: {n_train_frames} (weighted, no oversample), "
                  f"hidden: {hidden}")
            mlp = WeightedMLP(hidden_dim=hidden[0], max_iter=80,
                              random_state=seed)
            mlp.fit(X_aug, y_aug, sw_aug)
            print(f"  fit done in {mlp.n_iter_} epochs, "
                  f"loss={mlp.loss_:.4f}")
            del X_aug, y_aug, sw_aug
            gc.collect()
            metrics = eval_split(mlp, test_aug, args.fps_extract, smooth_s=10)
            return metrics, mlp, in_dim

        # ── Seed je Nacht, nicht je Sonde ────────────────────────────
        # Bis 2026-08-09 fitteten alle Schattenvarianten jede Nacht mit
        # random_state=0. Bei ueber 99 % Korpus-Ueberlappung zwischen zwei
        # Naechten ist eine Serie daraus nicht N Stichproben, sondern
        # naeherungsweise EINE Messung N-mal wiederholt: der Vorzeichentest
        # ueber die Naechte misst dann, wie reproduzierbar derselbe Seed
        # ist, nicht wie stabil der Effekt.
        #
        # Der Seed-Sweep vom selben Tag hat gezeigt, wie viel daran haengt:
        # identische Daten, identische Architektur, nur anderer Seed →
        # Golden-Median 0.901 bis 0.924 (Std 0.008). Also innerhalb einer
        # Nacht fuer ALLE Sonden derselbe Seed (die gepaarten Vergleiche
        # bleiben sauber), aber von Nacht zu Nacht ein anderer, damit ein
        # Median ueber die Serie auch das Seed-Glueck mit ausmittelt.
        #
        # Der Wert steht in jeder jsonl-Zeile — ohne ihn ist im Nachhinein
        # nicht pruefbar, ob zwei Zeilen ueberhaupt vergleichbar sind.
        # Die Produktion bleibt bewusst bei 0: ihr Seed zu bewegen wuerde
        # den ausgelieferten Kopf veraendern und mit dem Gate wechselwirken.
        nacht_seed = int(hashlib.sha256(ts.encode()).hexdigest()[:8], 16) % 10000

        print("\n" + "=" * 70)
        print(f"SHADOW EVAL — {n_chan} channel slugs in corpus, "
              f"Seed dieser Nacht: {nacht_seed}")
        print("=" * 70)
        m_v1, mlp_v1, _ = _fit_eval(
            "MLP-32 (1282→32→1)", lambda X, _s, _u=None: X, seed=nacht_seed)
        m_v2, mlp_v2, in_dim_v2 = _fit_eval(
            f"MLP-32 + channel one-hot (+{n_chan} dim)", _augment_channel,
            seed=nacht_seed)
        m_v3, mlp_v3, _ = _fit_eval(
            "MLP-32 + temporal L2 deltas (+2 dim)", _augment_temporal,
            seed=nacht_seed)
        # Whisper Stage 4 prototype — per-second whisper-prob as a
        # direct MLP input column instead of the post-processor rules.
        # n=149 reviews + MLP non-linearity should absorb the new
        # column without the L2-balance fragility of a LogReg add.
        # If +0.03 IoU vs MLP+channel → migrate; if neutral → keep
        # the post-processor rules (= they already gain +5.4 % on
        # n=9 and the rules are simpler to reason about).
        m_v4, mlp_v4, _ = _fit_eval(
            f"MLP-32 + channel + whisper-prob (+{n_chan + 1} dim)",
            _augment_channel_whisper, seed=nacht_seed)
        # MLP-64 capacity probe (m_v5) and the +temporal combo (m_v6)
        # were retired 2026-07-12: a 7-night --shadow-eval series
        # settled both questions (MLP-64 = noise, keep 32; +temporal =
        # 6/7 nights positive, migrated into production as
        # mlp32-channel-whisper-temporal / MLP3 v3). See
        # _augment_channel_whisper_temporal above if a future capacity
        # or feature probe needs the same pattern again.
        #
        # Minute-prior probe (started 2026-07-15): production replica
        # (cwt, = shadow-semantics twin of the deployed arch) vs the
        # same + P(ad|minute) column. Decision rule as with temporal:
        # consistently positive median Δ across several nights →
        # migrate (header bump + Go nn.go parity + daemon start_ts
        # pass-through); noise → drop the column, keep minute-prior
        # in Phase A pseudo-labelling only.
        m_v6, mlp_v6, _ = _fit_eval(
            f"MLP-32 + chan + whisper + temporal (prod replica)",
            _augment_channel_whisper_temporal, seed=nacht_seed)
        m_v7, mlp_v7, _ = _fit_eval(
            f"MLP-32 + cwt + minute-prior (+1 dim)",
            _augment_cwt_minuteprior, seed=nacht_seed)
        # Die Whisper-Sonde in der heutigen Architektur: identisch zu v7,
        # nur ohne die eine Spalte. Δ(v7−v8) ist der Beitrag von Whisper.
        m_v8, mlp_v8, _ = _fit_eval(
            f"MLP-32 + ct + minute-prior (OHNE whisper)",
            _augment_ct_minuteprior, seed=nacht_seed)

        # ── Rauschboden (--seed-sweep N) ─────────────────────────────
        # Dieselben Daten, dieselbe Architektur, nur ein anderer
        # Initialisierungs-Seed. Die Streuung DARAUS ist reiner Fit-Zufall;
        # alles was ueber die Naechte darueber hinaus schwankt, kommt vom
        # wandernden Korpus. Ohne diese Trennung ist keine
        # Entscheidungsregel kalibrierbar: am 2026-08-09 schwankte dieselbe
        # Sondenzelle ueber drei Laeufe um 0.028, und ich konnte nicht
        # sagen, ob das der Fit war oder die Daten.
        #
        # Nicht im Nightly (ein Fit+Eval je Seed). Von Hand aufrufen.
        seed_rows = []
        if args.seed_sweep > 1:
            def _augment_prod(X, slug, uuid):
                return mit_zusatz(
                    X, uuid, slug, chan_idx, n_chan,
                    whisper=wants_whisper, temporal=wants_temporal,
                    churn=wants_churn,
                    mp_col=_minuteprior_col if wants_minuteprior else None,
                    maske=wants_whispermask)
            for _s in range(args.seed_sweep):
                _m, _, _dim = _fit_eval(
                    f"SEED {_s} — Produktions-Architektur", _augment_prod,
                    seed=_s)
                seed_rows.append((_s, _m, _dim))

        # Persist the MLP+channel variant in v1-MLP head.bin format
        # (= what the Go forward-pass loader will consume). Sidecar
        # head.channel-map.json is co-written by the deploy block
        # below; this dumps to a parallel tree so it doesn't collide
        # with a production deploy. Use this file to bootstrap Go-
        # side regression tests before the production switch.
        try:
            shadow_dir = Path(args.output).parent / "shadow"
            shadow_dir.mkdir(parents=True, exist_ok=True)
            mlp_path = shadow_dir / "head.mlp32-channel.bin"
            n_audio_used = 1 if args.with_audio else 0
            n_logo_used = 1 if args.with_logo else 0
            sz = write_mlp_head_v1(
                mlp_path, mlp_v2,
                input_dim=in_dim_v2, hidden_dim=32,
                backbone_dim=1280, n_logo=n_logo_used,
                n_audio=n_audio_used, n_channel=n_chan)
            # Sidecar with the slug list — deterministic alphabetical
            # order matching chan_slugs above; Go reads this to map
            # an inference-time slug to the one-hot column.
            cm_path = shadow_dir / "head.mlp32-channel.channel-map.json"
            cm_path.write_text(json.dumps({
                "version": 1, "n": n_chan,
                "slugs": chan_slugs,
            }, indent=2))
            print(f"\n  shadow MLP head written: {mlp_path} ({sz} B)")
            print(f"  shadow channel-map:      {cm_path.name}")
        except Exception as e:
            print(f"\n  shadow MLP write err: {e}")

        # Variant comparisons use MEDIAN IoU (like the deploy gate) —
        # the mean is outlier-fragile: 2026-07-07, two same-day runs
        # swung v4's mean 0.720→0.684 and flipped the MLP-64 verdict
        # from "overfits" to "capacity helps" on a handful of block-
        # formation flips in movie-length recs. Mean stays as a second
        # column for continuity with the pre-07-07 tables.
        def _vmed(m):
            return m.get("iou_tv_median", m.get("iou_median", m.get("iou", 0.0)))
        base_iou = metrics_smooth["iou"]
        base_med = _vmed(metrics_smooth)
        base_acc = metrics_smooth["acc"]

        # Golden-Median je Variante. Der Testsatz wechselt seine
        # Zusammensetzung mit dem Korpus, der Golden-Satz nicht — nur DIESE
        # Spalte ist mit dem Nacht-zu-Nacht-Trend und dem Golden-Boden
        # vergleichbar. Gleiche Rechnung wie golden_boden(): Median der
        # per-rec-IoU ueber die gepinnten uuids, mit derselben
        # Vollstaendigkeitspflicht (fehlt eine, ist der Wert nicht
        # zusammensetzungs-konstant und wird nicht gezeigt).
        _golden_uuids = set()
        try:
            if args.train_archive:
                _gp = Path(args.train_archive) / "golden-eval-set.json"
                if _gp.exists():
                    _golden_uuids = set(
                        json.loads(_gp.read_text()).get("uuids") or [])
        except Exception as e:
            print(f"  Golden-Satz nicht lesbar ({e}) — Spalte entfaellt")

        def _gmed(m):
            pr = (m or {}).get("per_rec_iou") or {}
            if not _golden_uuids or not _golden_uuids <= set(pr):
                return None
            return float(np.median([pr[u] for u in _golden_uuids]))

        def _gtxt(m):
            g = _gmed(m)
            return f"{g:>6.3f}" if g is not None else f"{'—':>6}"

        print("\n" + "=" * 70)
        print("SHADOW COMPARISON (smooth=10s, test set)")
        print("=" * 70)
        print(f"  {'arch':35s}  {'medIoU':>6}  {'Δ vs LR':>9}  {'mean':>6}  "
              f"{'acc':>6}  {'golden':>6}")
        print(f"  {'baseline (' + args.head_arch + ')':35s}  "
              f"{base_med:>6.3f}  {'(ref)':>9}  {base_iou:>6.3f}  "
              f"{base_acc*100:>5.1f}%  {_gtxt(metrics_smooth)}")
        for name, m in [("MLP-32", m_v1),
                        (f"MLP-32 + channel ({n_chan})", m_v2),
                        ("MLP-32 + temporal", m_v3),
                        (f"MLP-32 + channel + whisper", m_v4),
                        ("MLP-32 + cwt (prod replica)", m_v6),
                        ("MLP-32 + cwt + minute-prior", m_v7),
                        ("MLP-32 + ct + mp (OHNE whisper)", m_v8)]:
            d = _vmed(m) - base_med
            mark = "  ↑" if d > 0.01 else ("  ↓" if d < -0.01 else "")
            print(f"  {name:35s}  {_vmed(m):>6.3f}  "
                  f"{d:>+9.3f}  {m['iou']:>6.3f}  {m['acc']*100:>5.1f}%  "
                  f"{_gtxt(m)}{mark}")
        # The decision number for the minute-prior probe: Δ vs the
        # production replica under identical shadow semantics.
        d_mp = _vmed(m_v7) - _vmed(m_v6)
        print(f"\n  minute-prior probe: cwt {_vmed(m_v6):.3f} → "
              f"cwt+mp {_vmed(m_v7):.3f}  (Δ {d_mp:+.3f})")
        # Die Whisper-Spalte, isoliert in der heutigen Architektur:
        # v7 und v8 unterscheiden sich in genau dieser einen Spalte.
        d_wh = _vmed(m_v7) - _vmed(m_v8)
        g7, g8 = _gmed(m_v7), _gmed(m_v8)
        g_txt = (f", golden {g8:.3f} → {g7:.3f} (Δ {g7 - g8:+.3f})"
                 if g7 is not None and g8 is not None else "")
        print(f"  whisper probe:      ct+mp {_vmed(m_v8):.3f} → "
              f"cwt+mp {_vmed(m_v7):.3f}  (Δ {d_wh:+.3f}){g_txt}")
        # Stage-4 specific Δ vs Stage-3 baseline (= MLP+channel).
        # Tells you "is whisper as an MLP input column better than
        # whisper as a post-processor rule set". Migration breaks even
        # somewhere around +0.03 IoU.
        d_stage4 = _vmed(m_v4) - _vmed(m_v2)
        mark4 = ("  ↑ migrate" if d_stage4 > 0.03 else
                 ("  ↓ keep post-processor" if d_stage4 < -0.01 else
                  "  ≈ neutral, keep post-processor"))
        print()
        print(f"  Δ vs Stage-3 baseline (MLP+channel): "
              f"{d_stage4:+.3f}{mark4}")
        print()

        if seed_rows:
            _gs = [_gmed(m) for _, m, _ in seed_rows]
            _ts_ = [_vmed(m) for _, m, _ in seed_rows]
            _gs = [g for g in _gs if g is not None]
            print("\n" + "=" * 70)
            print(f"RAUSCHBODEN — {len(seed_rows)} Fits, identische Daten, "
                  f"nur Init-Seed verschieden")
            print("=" * 70)
            print(f"  {'seed':>4}  {'medIoU':>6}  {'golden':>6}")
            for _s, _m, _ in seed_rows:
                _g = _gmed(_m)
                print(f"  {_s:>4}  {_vmed(_m):>6.3f}  "
                      f"{(f'{_g:.3f}' if _g is not None else '—'):>6}")
            def _spanne(v, label):
                if len(v) < 2:
                    return
                print(f"  {label}: Spanne {max(v) - min(v):.3f} "
                      f"({min(v):.3f}–{max(v):.3f}), "
                      f"Std {float(np.std(v)):.3f}")
            _spanne(_ts_, "test  ")
            _spanne(_gs, "golden")
            print("\n  Lesart: was hier steht, ist NUR Fit-Zufall. Eine "
                  "Nacht-zu-Nacht-Differenz unterhalb dieser Spanne ist "
                  "kein Ergebnis — egal wie plausibel die Geschichte dazu "
                  "klingt.")
            print()

        # ── Die Serie festhalten ─────────────────────────────────────
        # Eine Zeile je Nacht UND Variante. Grund (2026-08-09): die
        # Migrations-Entscheidungen dieses Repos (temporal nach 7 Naechten,
        # Minute-Prior nach 3) haengen an einer Serie, die bisher nur als
        # Text in einem 4-MB-Log existierte. Wer sie auswerten wollte,
        # musste Prosa parsen — und hat dabei zwei zufaellig gleiche
        # Zahlen (0.946 an zwei Abenden) fuer Reproduzierbarkeit gehalten,
        # obwohl dieselbe Zelle ueber drei Laeufe um 0.028 schwankt.
        #
        # set_hash + decoder + n_test stehen in jeder Zeile, weil ein
        # Median ohne sie nicht vergleichbar ist: der Golden-Satz hat
        # schon einmal still seine Zusammensetzung gewechselt, und der
        # Decoder-Wechsel form→hsmm hat jede Zahl davor entwertet.
        try:
            if args.train_archive:
                _gmeta = {}
                if _golden_uuids:
                    _gmeta = json.loads(
                        (Path(args.train_archive)
                         / "golden-eval-set.json").read_text())
                _zeilen = [("baseline", args.head_arch, metrics_smooth)]
                _zeilen += [("shadow", n, m) for n, m in [
                    ("mlp32", m_v1),
                    ("mlp32-channel", m_v2),
                    ("mlp32-temporal", m_v3),
                    ("mlp32-channel-whisper", m_v4),
                    ("mlp32-cwt", m_v6),
                    ("mlp32-cwt-mp", m_v7),
                    ("mlp32-ct-mp", m_v8)]]
                with open(Path(args.train_archive) / "shadow-trend.jsonl",
                          "a") as _sf:
                    for _rolle, _name, _m in _zeilen:
                        _g = _gmed(_m)
                        _pr = (_m or {}).get("per_rec_iou") or {}
                        _gm = ([_pr[u] for u in _golden_uuids]
                               if _g is not None else [])
                        _sf.write(json.dumps({
                            "ts": ts, "rolle": _rolle, "arch": _name,
                            # Produktion fittet fest mit 0, die Sonden mit dem
                            # Seed dieser Nacht. Ohne das Feld sieht ein
                            # spaeterer Leser nicht, dass baseline- und
                            # shadow-Zeilen NICHT seed-gleich sind — und
                            # vergleicht sie arglos.
                            "seed": 0 if _rolle == "baseline" else nacht_seed,
                            "golden_median": (round(_g, 4)
                                              if _g is not None else None),
                            "golden_mean": (round(float(np.mean(_gm)), 4)
                                            if _gm else None),
                            "golden_n": len(_gm),
                            "test_median": round(_vmed(_m), 4),
                            "test_mean": round(float(_m["iou"]), 4),
                            "acc": round(float(_m["acc"]), 4),
                            "n_test": int(_m.get("n_recs") or 0),
                            "n_train_recs": len(train_recs),
                            "set_version": _gmeta.get("version"),
                            "set_hash": _gmeta.get("set_hash"),
                            "decoder": " ".join(EVAL_DECODER) or "form",
                        }) + "\n")
                print(f"  Serie fortgeschrieben → shadow-trend.jsonl "
                      f"({len(_zeilen)} Zeilen, ts={ts})")
        except Exception as e:
            print(f"  shadow-trend.jsonl nicht geschrieben: {e}")
        print()

    # ── Self-Training (Phase A, validation only) ─────────────────
    # Test how reliable our pseudo-labels would be: for each TEST
    # recording the head was never trained on, predict probabilities,
    # filter to frames where (a) the head is highly confident AND
    # (b) the wall-clock minute-prior agrees with the prediction
    # (independent sanity check). Compare those filtered predictions
    # against user truth — if they're ≥95% accurate, Phase B (write
    # pseudo-labels for unreviewed recordings, mix into next training
    # round at reduced weight) is safe to enable.
    if args.with_self_training and test_recs:
        if not minute_prior:
            print("\nself-training: --with-minute-prior is required "
                  "(skipping validation)")
        else:
            conf_th = args.self_train_conf
            # Per-channel confidence gate (2026-07-13, optimization
            # backlog #9): self-training accuracy at the GLOBAL
            # threshold streaks wildly by channel (sat-1 100%, rtl
            # ~92%) — a single conf_th either wastes headroom on clean
            # channels or risks accuracy on noisy ones. Sweep a few
            # candidates per channel at Phase-A time (cheap: proba/
            # p_prior are already computed once per rec, only the
            # mask/threshold comparison repeats) and let each channel
            # use the LOOSEST candidate that still clears the existing
            # 95%-SAFE bar — never looser than validated, never
            # stuck at the global default if a channel can support more.
            CONF_CANDIDATES = sorted({conf_th, 0.99, 0.97, 0.95, 0.93},
                                     reverse=True)
            n_total = n_pseudo = n_correct = 0
            per_chan_stats = {}  # slug -> [n_pseudo, n_correct] @ conf_th (reporting only)
            # slug -> cand_th -> [n_pseudo, n_correct]
            per_chan_cand = {}
            for r in test_recs:
                uuid = r[0]
                slug = uuid_slug.get(uuid, "")
                start = uuid_start.get(uuid, 0)
                if not slug or not start or slug not in minute_prior:
                    continue
                proba = clf.predict_proba(r[3])[:, 1]
                y_truth = r[4]
                n = len(proba)
                n_total += n
                prior_arr = np.array(minute_prior[slug])
                minutes = ((start + np.arange(n) / args.fps_extract)
                           // 60 % 60).astype(int)
                p_prior = prior_arr[minutes]
                # Confidence + agreement filter (reporting @ global conf_th)
                conf_ad = (proba >= conf_th) & (p_prior >= 0.5)
                conf_show = (proba <= 1 - conf_th) & (p_prior < 0.5)
                pseudo_mask = conf_ad | conf_show
                pseudo_label = np.where(conf_ad, 1, 0)
                correct = (pseudo_label[pseudo_mask] == y_truth[pseudo_mask]).sum()
                n_pseudo += int(pseudo_mask.sum())
                n_correct += int(correct)
                per_chan_stats.setdefault(slug, [0, 0])
                per_chan_stats[slug][0] += int(pseudo_mask.sum())
                per_chan_stats[slug][1] += int(correct)
                # Per-candidate sweep for the per-channel gate.
                cand_stats = per_chan_cand.setdefault(slug, {})
                for cth in CONF_CANDIDATES:
                    c_ad = (proba >= cth) & (p_prior >= 0.5)
                    c_show = (proba <= 1 - cth) & (p_prior < 0.5)
                    c_mask = c_ad | c_show
                    c_label = np.where(c_ad, 1, 0)
                    c_correct = (c_label[c_mask] == y_truth[c_mask]).sum()
                    npi, nci = cand_stats.get(cth, (0, 0))
                    cand_stats[cth] = (npi + int(c_mask.sum()),
                                       nci + int(c_correct))
            # Pick the loosest (lowest) candidate per channel that
            # clears 95% accuracy on ≥200 candidate frames (small-n
            # channels keep the global default — not enough evidence
            # to trust a looser threshold). Falls back to conf_th.
            PER_CHAN_MIN_FRAMES = 200
            per_chan_conf = {}
            for slug, cand_stats in per_chan_cand.items():
                best = conf_th
                for cth in sorted(CONF_CANDIDATES):  # loosest first
                    npi, nci = cand_stats.get(cth, (0, 0))
                    if npi >= PER_CHAN_MIN_FRAMES and nci / npi >= 0.95:
                        best = cth
                        break
                per_chan_conf[slug] = best
            print(f"\n=== Self-Training validation (test set, {n_total} frames) ===")
            print(f"  threshold p>{conf_th} or p<{1-conf_th:.2f} + minute-prior agrees")
            if n_pseudo == 0:
                print(f"  no frames passed the filter — threshold too tight "
                      f"or prior coverage too sparse")
            else:
                acc = 100 * n_correct / n_pseudo
                kept = 100 * n_pseudo / n_total
                print(f"  candidates: {n_pseudo}/{n_total} ({kept:.1f}% of frames)")
                print(f"  accuracy:   {n_correct}/{n_pseudo} ({acc:.2f}%)")
                verdict = ("SAFE" if acc >= 95.0 else
                            "RISKY" if acc >= 90.0 else "UNSAFE")
                print(f"  → Phase B viability: {verdict} "
                      f"(≥95% safe, 90-95% risky w/ low weight, <90% don't)")
                print(f"\n  per-channel breakdown (@ global threshold "
                      f"{conf_th}) + gated threshold used in Phase B:")
                for slug, (npi, nci) in sorted(per_chan_stats.items()):
                    gated = per_chan_conf.get(slug, conf_th)
                    tag = " (loosened)" if gated < conf_th else ""
                    if npi == 0:
                        print(f"    {slug:14s}  no candidates  "
                              f"gate={gated}{tag}")
                    else:
                        print(f"    {slug:14s}  {nci:>5}/{npi:<5}  "
                              f"acc {100*nci/npi:5.1f}%  gate={gated}{tag}")

            # ── Phase B: write pseudo_labels.json for unreviewed
            # recordings. Walks ALL recordings (not just per_rec — those
            # are skipped without ads.json), predicts with the current
            # production head + minute-prior agreement filter, writes a
            # pseudo_labels.json next to ads.json. Cleared automatically
            # when ads_user.json appears (loader checks user-first).
            if args.write_pseudo_labels:
                head_ts = time.strftime("%Y%m%dT%H%M%S")
                n_written = n_skipped = 0
                for d in sorted(Path(args.hls_root).glob("_rec_*")):
                    uuid = d.name[5:]
                    user = d / "ads_user.json"
                    pseudo_path = d / "pseudo_labels.json"
                    # Stale pseudo-labels for now-reviewed recordings are
                    # superseded by the user file — clean up.
                    if user.exists():
                        if pseudo_path.exists():
                            try: pseudo_path.unlink()
                            except Exception: pass
                        n_skipped += 1
                        continue
                    slug = uuid_slug.get(uuid, "")
                    start = uuid_start.get(uuid, 0)
                    if not slug or not start or slug not in minute_prior:
                        n_skipped += 1; continue
                    # Need cached features; if missing, this recording
                    # was not in per_rec — skip until next training run
                    # has processed it.
                    matching = [r for r in per_rec if r[0] == uuid]
                    if not matching:
                        n_skipped += 1; continue
                    feats = matching[0][3]
                    # Defensive pad: bootstrap recordings extracted
                    # without a known channel slug skip the optional
                    # feature columns (logo / channel / audio), so
                    # their cached array can be narrower than what
                    # clf was fit on. Right-pad with neutral 0.5 so
                    # predict_proba doesn't throw — neutral values
                    # mean these recordings just don't contribute
                    # the channel-specific signal but the rest works.
                    expected_dim = clf.coef_.shape[1]
                    if feats.shape[1] < expected_dim:
                        pad = np.full(
                            (feats.shape[0], expected_dim - feats.shape[1]),
                            0.5, dtype=feats.dtype)
                        feats = np.concatenate([feats, pad], axis=1)
                    proba = clf.predict_proba(feats)[:, 1]
                    n = len(proba)
                    prior_arr = np.array(minute_prior[slug])
                    minutes = ((start + np.arange(n) / args.fps_extract)
                               // 60 % 60).astype(int)
                    p_prior = prior_arr[minutes]
                    # Per-channel gated threshold (see Phase A sweep
                    # above) — falls back to the global default for
                    # channels with no/thin test-set representation.
                    rec_conf_th = per_chan_conf.get(slug, conf_th)
                    conf_ad = (proba >= rec_conf_th) & (p_prior >= 0.5)
                    conf_show = (proba <= 1 - rec_conf_th) & (p_prior < 0.5)
                    pseudo_mask = conf_ad | conf_show
                    if not pseudo_mask.any():
                        if pseudo_path.exists():
                            try: pseudo_path.unlink()
                            except Exception: pass
                        n_skipped += 1; continue
                    frames = np.where(pseudo_mask)[0].tolist()
                    labels = np.where(conf_ad, 1, 0)[pseudo_mask].astype(int).tolist()
                    pseudo_path.write_text(json.dumps({
                        "version": 1,
                        "head_ts": head_ts,
                        "threshold": rec_conf_th,
                        "fps": args.fps_extract,
                        "n_frames": int(pseudo_mask.sum()),
                        "n_total": n,
                        "frames": frames,
                        "labels": labels,
                    }))
                    n_written += 1
                print(f"\nself-training Phase B: wrote {n_written} pseudo_labels.json file(s) "
                      f"(skipped {n_skipped} — already reviewed or no candidates)")

    # ── Co-Training (Phase 1, analysis only) ─────────────────────
    # Train two extra heads with disjoint discriminative feature
    # views and report how often they agree. Foundation for Phase 2
    # (use agreement on UNLABELLED frames as pseudo-labels) once we
    # know the audio head carries enough signal to be a co-teacher.
    # Per Blum & Mitchell (1998), co-training is mathematically sound
    # when the two views are conditionally independent given the
    # label — visual (backbone+logo) and acoustic (rms) approximately
    # satisfy that for ad/show classification.
    if args.co_train and test_recs:
        # Compute column-slice indices dynamically from the flag set.
        # Order MUST match _worker_extract: backbone, logo, channel,
        # rms, yamnet (each section optional). Drift here = wrong
        # features fed to the wrong head = silently broken results.
        col = 1280
        backbone_cols = list(range(0, 1280))
        logo_col = -1
        if args.with_logo:
            logo_col = col; col += 1
        channel_cols = []
        if args.with_channel:
            channel_cols = list(range(col, col + 6)); col += 6
        audio_col = -1
        if args.with_audio:
            audio_col = col; col += 1
        yamnet_cols = []
        if args.with_yamnet:
            yamnet_cols = list(range(col, col + 1024)); col += 1024
        uniformity_col = -1
        if args.with_uniformity:
            uniformity_col = col; col += 1

        # head_logo: visual signal — backbone + logo + channel + uniformity
        logo_view_cols = backbone_cols + (
            [logo_col] if logo_col >= 0 else []) + channel_cols + (
            [uniformity_col] if uniformity_col >= 0 else [])
        # head_audio: acoustic signal — yamnet (if present) + rms +
        # channel. NO backbone (that's the visual view; including it
        # would defeat conditional-independence for co-training).
        audio_view_cols = yamnet_cols + (
            [audio_col] if audio_col >= 0 else []) + channel_cols
        audio_desc = "+".join(filter(None, [
            "yamnet" if yamnet_cols else "",
            "rms" if audio_col >= 0 else "",
            "chan" if channel_cols else ""]))

        from sklearn.linear_model import LogisticRegression as LR
        clf_logo = LR(max_iter=2000, C=1.0)
        clf_audio = LR(max_iter=2000, C=1.0)
        clf_logo.fit(X_train[:, logo_view_cols], y_train,
                     sample_weight=sw_train)
        clf_audio.fit(X_train[:, audio_view_cols], y_train,
                      sample_weight=sw_train)

        # Evaluate each sub-head on test set (unsmoothed, raw frame acc)
        X_test_parts, y_test_parts = [], []
        for r in test_recs:
            X_test_parts.append(r[3])
            y_test_parts.append(r[4])
        X_test = np.concatenate(X_test_parts) if X_test_parts else np.empty((0, X_train.shape[1]))
        y_test = np.concatenate(y_test_parts) if y_test_parts else np.empty(0)

        p_main = clf.predict_proba(X_test)[:, 1]
        p_logo = clf_logo.predict_proba(X_test[:, logo_view_cols])[:, 1]
        p_audio = clf_audio.predict_proba(X_test[:, audio_view_cols])[:, 1]

        acc_main = ((p_main >= 0.5) == y_test).mean()
        acc_logo = ((p_logo >= 0.5) == y_test).mean()
        acc_audio = ((p_audio >= 0.5) == y_test).mean()

        # Agreement matrix at confidence threshold 0.7
        conf = 0.7
        logo_ad = p_logo > conf
        logo_show = p_logo < (1 - conf)
        audio_ad = p_audio > conf
        audio_show = p_audio < (1 - conf)
        n = len(y_test)
        agree_ad = (logo_ad & audio_ad).sum()
        agree_show = (logo_show & audio_show).sum()
        disagree_ad = (logo_ad & audio_show).sum()
        disagree_show = (logo_show & audio_ad).sum()
        either_unsure = n - agree_ad - agree_show - disagree_ad - disagree_show

        # When both heads agree confidently, how often do they agree
        # WITH THE TRUTH? If high, agreement is a reliable pseudo-label
        # signal. If low, even agreement is noisy and pseudo-labels
        # would inject errors.
        agree_mask = (logo_ad & audio_ad) | (logo_show & audio_show)
        if agree_mask.sum() > 0:
            agree_pred = np.where(logo_ad & audio_ad, 1, 0)[agree_mask]
            agree_truth = y_test[agree_mask]
            agree_acc = (agree_pred == agree_truth).mean()
        else:
            agree_acc = 0.0

        print(f"\n=== Co-Training analysis (test set, {n} frames) ===")
        print(f"head_main  acc {acc_main*100:.1f}%  ({X_train.shape[1]} dims)")
        print(f"head_logo  acc {acc_logo*100:.1f}%  ({len(logo_view_cols)} dims, backbone+logo+chan)")
        print(f"head_audio acc {acc_audio*100:.1f}%  ({len(audio_view_cols)} dims, {audio_desc})")
        print(f"\nAgreement (both confident at p>{conf}):")
        print(f"  agree-AD       {agree_ad:>6}  ({100*agree_ad/n:.1f}%)")
        print(f"  agree-SHOW     {agree_show:>6}  ({100*agree_show/n:.1f}%)")
        print(f"  disagree (logo=AD, audio=SHOW)  {disagree_ad:>6}  ({100*disagree_ad/n:.1f}%)")
        print(f"  disagree (logo=SHOW, audio=AD)  {disagree_show:>6}  ({100*disagree_show/n:.1f}%)")
        print(f"  either unsure  {either_unsure:>6}  ({100*either_unsure/n:.1f}%)")
        print(f"\nWhen both agree confidently, accuracy vs truth: {agree_acc*100:.1f}%")
        print(f"  → Pseudo-label viability: "
              f"{'GOOD' if agree_acc >= 0.95 else ('OK' if agree_acc >= 0.85 else 'POOR')} "
              f"(>=0.95 → safe, 0.85-0.95 → with reduced weight, <0.85 → don't)")

        # Save sub-heads for inspection (NOT deployed — Go inference
        # only loads head.bin in the production format).
        for name, sub_clf, sub_dim in [
                ("head_logo", clf_logo, len(logo_view_cols)),
                ("head_audio", clf_audio, len(audio_view_cols))]:
            sub_path = Path(args.output).with_suffix(f".{name}.bin")
            with open(sub_path, "wb") as f:
                for w in sub_clf.coef_.ravel():
                    f.write(struct.pack("<f", float(w)))
                f.write(struct.pack("<f", float(sub_clf.intercept_[0])))
            print(f"saved {sub_path.name} ({sub_dim} weights + bias)")

    # Refit on ALL data before writing head.bin (validation told us
    # it works; ship the full-data model).
    if args.final_on_all and test_recs:
        print("\nrefitting on all data for production head...")
        # Bootstrap recordings (no slug at extract → optional feature
        # columns absent) have narrower X than full-feature recordings
        # → np.concatenate fails. Drop them from the final fit since
        # they have no labels anyway (frame_mask=all-False).
        target_dim = max(r[3].shape[1] for r in per_rec)
        keep = [r for r in per_rec if r[3].shape[1] == target_dim
                                    and not (len(r) > 12 and r[12])]
        X_all = np.concatenate([r[3] for r in keep])
        y_all = np.concatenate([r[4] for r in keep])
        clf = LogisticRegression(max_iter=2000, C=1.0, verbose=0)
        clf.fit(X_all, y_all)
        full_acc = (clf.predict(X_all) == y_all).mean()
        print(f"full-data fit acc {full_acc*100:.1f}% "
              f"({len(keep)}/{len(per_rec)} recs, "
              f"{X_all.shape[0]} frames)")
    weights = clf.coef_.ravel().astype(np.float32)  # (1280,)
    bias = float(clf.intercept_[0])

    # MLP refit-on-all (mlp32-channel mode) — analogous to the LogReg
    # final_on_all above, but for the MLP that will be written to
    # head.bin. Builds a channel-augmented X_all from per_rec (skipping
    # bootstrap recordings whose narrower X breaks np.concatenate),
    # then fits the production MLP on the union of train + test.
    # mlp_prod_clf is rebound to the all-data version; held-out IoU
    # in metrics_smooth was already captured against the earlier
    # train-only fit, so the deploy-decision logic is unaffected.
    if (wants_mlp
            and args.final_on_all and test_recs
            and mlp_prod_clf is not None):
        target_dim_all = max(r[3].shape[1] for r in per_rec)
        keep_all = [r for r in per_rec
                    if r[3].shape[1] == target_dim_all
                       and not (len(r) > 12 and r[12])]
        n_chan_prod = len(mlp_prod_chan_slugs)
        prod_chan_idx_all = {s: i for i, s in enumerate(mlp_prod_chan_slugs)}
        # Zusatzblock je Aufnahme, in derselben Reihenfolge wie die
        # X-Verkettung. Eigene Kanalkarte: der Refit laeuft ueber ALLE
        # Aufnahmen, nicht nur ueber train+test, deshalb prod_chan_idx_all.
        X_parts = []
        y_parts = []
        for r in keep_all:
            X_parts.append(np.hstack([
                r[3], _prod_zusatz(r[3], r[0], prod_chan_idx_all,
                                   n_chan_prod)]).astype(np.float32))
            y_parts.append(r[4])
        X_all_ch = np.concatenate(X_parts) if X_parts else np.empty((0, 0))
        y_all_ch = np.concatenate(y_parts) if y_parts else np.empty(0)
        # The per-rec hstack parts are a second full corpus copy (~10 GB)
        # that would otherwise stay referenced through the fit below.
        del X_parts, y_parts
        gc.collect()
        # Reconstruct per-frame sample weights from per-rec metadata
        # (= same logic as the train-only fit: user_weight × age decay,
        # pseudo_weight for pseudo-labelled, base 1.0 otherwise). Skip
        # bumper boost + skip-press signals — those live on per-frame
        # masks not retained here. Acceptable simplification: the
        # all-data refit's role is "more data, mostly the same
        # gradient", not "perfect weighting reproducibility".
        sw_parts = []
        for r in keep_all:
            T = len(r[4])
            is_pseudo = len(r) > 11 and r[11]
            if is_pseudo:
                bw = args.pseudo_weight
            else:
                bw = args.user_weight if r[5] else 1.0
            age_d = r[8] if len(r) > 8 else 0
            if age_d > 180:
                age_mult = 0.0
            elif age_d > 90:
                age_mult = 0.5 * (180 - age_d) / 90.0
            else:
                age_mult = 1.0 - 0.5 * age_d / 90.0
            bw *= age_mult
            sw_arr = np.full(T, max(bw, 0.0), dtype=np.float32)
            # NaN-logo skip in the all-data refit too — symmetric with
            # the train-only fit so the production head doesn't learn
            # corruption-driven 0.5s as legitimate signal.
            nan_mask = logo_nan_mask_by_uuid.get(r[0])
            if nan_mask is not None and len(nan_mask) == T:
                sw_arr[nan_mask] = 0.0
            sw_parts.append(sw_arr)
        sw_all_ch = np.concatenate(sw_parts) if sw_parts else np.empty(0)
        # Drop weight-0 rows (= rec older than 180 d); they'd just dilute.
        nz = sw_all_ch > 0
        X_all_ch = X_all_ch[nz]
        y_all_ch = y_all_ch[nz]
        sw_all_ch = sw_all_ch[nz]
        print(f"\nrefitting MLP on all data for production head...")
        print(f"  base: {len(X_all_ch)} frames "
              f"({len(keep_all)}/{len(per_rec)} recs), "
              f"true sample_weight (no oversample)")
        mlp_prod_clf = WeightedMLP(hidden_dim=32, max_iter=200,
                                   random_state=0)
        mlp_prod_clf.fit(X_all_ch, y_all_ch, sw_all_ch)
        full_acc = (mlp_prod_clf.predict(X_all_ch) == y_all_ch).mean()
        print(f"  full-data fit acc {full_acc*100:.1f}%, "
              f"epochs {mlp_prod_clf.n_iter_}, "
              f"loss {mlp_prod_clf.loss_:.4f}")
        del X_all_ch, y_all_ch, sw_all_ch
        gc.collect()
        # BIAS-CHECK (env BIAS_CHECK): this all-data refit IS what becomes
        # head.bin / the deployed champion — and it just trained on the TEST set
        # too. Scoring it on the test set is scoring on TRAINING data → inflated.
        # The deployed-head re-eval in the gate does exactly this every night, so
        # the champion gets a memorisation advantage over the honest (train-only)
        # candidate. This prints the magnitude: all-data-on-test vs the train-only
        # held-out the gate uses for the candidate.
        if os.environ.get("BIAS_CHECK") and test_recs_ch:
            _m_all = eval_split(mlp_prod_clf, test_recs_ch, args.fps_extract, smooth_s=10)
            print(f"  [BIAS-CHECK] all-data refit (=head.bin) re-scored on the "
                  f"TEST set it TRAINED on: tv-median={_m_all.get('iou_tv_median'):.3f}  "
                  f"|  honest train-only held-out (gate's candidate metric)="
                  f"{metrics_smooth.get('iou_tv_median'):.3f}")

    # Active-learning surface: pick the N frames per recording where
    # the trained head is least confident. These are the frames worth
    # the user's labelling time — high-confidence frames are already
    # right and don't move the model.
    if args.surface_uncertain > 0:
        out_path = Path(args.output).with_suffix(".uncertain.txt")
        # Two-bucket surfacing: half the slots go to high-uncertainty
        # frames (model unsure, p≈0.5), half to high-divergence frames
        # (model confident BUT wall-clock prior strongly disagrees).
        # The two signals catch different failure modes — uncertainty
        # finds boundary frames, divergence finds wrong-but-confident
        # predictions. Combining via max() doesn't work: any p≈0.5
        # frame trivially wins because unc is in [0,1] while div ∈
        # [0, ~0.6], so divergence cases never surface.
        n_unc = max(1, args.surface_uncertain // 2)
        n_div = args.surface_uncertain - n_unc
        # Cohort-aware budget: the held-out eval tells us which shows the model
        # is WEAK on (low Block-IoU = confidently-wrong, exactly the failure the
        # global uncertainty sampler misses — those frames aren't near p≈0.5, so
        # pure |p-0.5| ranking never surfaces them). Bias more labelling budget
        # onto recordings whose show (fallback: channel) scored low on the test
        # set, so human review concentrates where it actually moves the model.
        # Uniform fallback when no eval IoU is available.
        from statistics import median as _median
        # Die Kohorten-Gewichtung folgt dem CHAMPION, nicht dem Kandidaten.
        #
        # Begründung ist ein Prinzip, KEINE gemessene Verbesserung:
        # metrics_smooth ist die Auswertung des frisch trainierten
        # Kandidaten, und der wird oft ABGELEHNT (Golden-Boden). Label-
        # Budget soll dorthin fließen, wo das AUSGELIEFERTE Modell schwach
        # ist — das ist es, was der Nutzer erlebt. Ein abgelehnter Kandidat
        # beschreibt einen Zustand, den nie jemand zu sehen bekommt.
        #
        # ⚠️ EHRLICH ZUR WIRKUNG: am Lauf 20260804T054558 nachgerechnet
        # (91 Recs mit beiden Werten) ändert das KEINEN einzigen
        # Multiplikator — Kandidat und Champion urteilten dort fast gleich
        # (größte Abweichung 0.34, comedy-central bei BEIDEN 0.2977). Der
        # Sprung auf 0.991 dort kam vom Rollback auf den 0.915-Kopf, nicht
        # von dieser Unterscheidung. Die Änderung ist also Absicherung
        # gegen einen Fall, der auftreten KANN (Kandidat bricht ein, wo der
        # Champion trägt), nicht die Behebung eines gemessenen Schadens.
        # Wer sie später bewertet: nicht mehr erwarten, als hier steht.
        #
        # deployed_test_metrics ist die Head-to-Head-Auswertung des
        # DEPLOYTEN Kopfs auf demselben Testsatz. Nur wenn sie fehlt
        # (Kanal-Karte weicht ab, kein Champion ladbar), bleibt der
        # Kandidat die Notlösung.
        _pri_quelle = "champion (head-to-head)"
        try:
            _pri = (deployed_test_metrics or {}).get("per_rec_iou") or {}
        except Exception:
            _pri = {}
        if not _pri:
            _pri_quelle = "kandidat (kein head-to-head verfuegbar)"
            try:
                _pri = metrics_smooth.get("per_rec_iou") or {}
            except Exception:
                _pri = {}
        _uuid_title = {r[0]: r[1] for r in per_rec}
        _title_iou, _slug_iou = {}, {}
        for _u, _io in _pri.items():
            _t = _uuid_title.get(_u, "")
            if _t:
                _title_iou.setdefault(_t, []).append(_io)
            _s = uuid_slug.get(_u, "")
            if _s:
                _slug_iou.setdefault(_s, []).append(_io)
        _title_iou = {k: _median(v) for k, v in _title_iou.items()}
        _slug_iou = {k: _median(v) for k, v in _slug_iou.items()}
        _global_iou = _median(list(_pri.values())) if _pri else 1.0

        def _cohort_mult(title, slug):
            io = _title_iou.get(title)
            if io is None:
                io = _slug_iou.get(slug, _global_iou)
            if io < 0.30:
                return 3
            if io < 0.50:
                return 2
            return 1

        _mult_hist = {1: 0, 2: 0, 3: 0}
        _budget = 0
        skipped_logo = 0
        skipped_whisper = 0
        emitted = 0
        with open(out_path, "w") as f:
            f.write("# uuid\ttime_s\tprobability\ttitle\tsource\n")
            for uuid, title, ads, X, y, *_ in per_rec:
                # Cohort-weighted per-recording budget (weak shows get more).
                _slug = uuid_slug.get(uuid, "")
                mult = _cohort_mult(title, _slug)
                _mult_hist[mult] = _mult_hist.get(mult, 0) + 1
                k_unc, k_div = n_unc * mult, n_div * mult
                _budget += k_unc + k_div
                # Pad bootstrap recordings (no slug → no logo column
                # at extract time) up to clf's expected dim so
                # predict_proba doesn't throw. Same defensive pattern
                # as the Phase B inference site below.
                expected_dim = clf.coef_.shape[1]
                if X.shape[1] < expected_dim:
                    X = np.concatenate([X, np.full(
                        (X.shape[0], expected_dim - X.shape[1]),
                        0.5, dtype=X.dtype)], axis=1)
                # Use calibrated probabilities so "uncertainty"
                # reflects the model's true confidence, not the
                # over-confidence of an uncalibrated logistic head.
                proba = calibrated_proba(X)
                n = len(proba)
                # Filter 1 — logo-sentinel strip. Frames whose logo
                # column was the NaN sentinel (= extract_logo silently
                # failed on a corrupt stream chunk) get a calibrated
                # proba near 0.5 because the head sees the substituted
                # neutral input. They're not real uncertainty — they're
                # missing data. Surfacing them wastes review time.
                skip_mask = np.zeros(n, dtype=bool)
                logo_nan = logo_nan_mask_by_uuid.get(uuid)
                if logo_nan is not None and len(logo_nan) == n:
                    skip_mask |= logo_nan
                # Filter 2 — whisper-agreement strip. If the per-second
                # whisper ad-classifier (F1=0.94) is highly confident
                # AND agrees with the head's lean, the frame is
                # effectively already labelled — the whisper feature
                # block carries that signal into the next retrain, so a
                # manual review adds nothing. Disagreements stay (they
                # are exactly the frames worth labelling).
                whisper_ps = _load_whisper_per_sec(uuid, n)
                whisper_conf = np.abs(whisper_ps - 0.5) > 0.35
                whisper_agrees = (whisper_ps > 0.5) == (proba > 0.5)
                whisper_skip = whisper_conf & whisper_agrees
                pre_logo = int(skip_mask.sum())
                skip_mask |= whisper_skip
                skipped_logo += pre_logo
                skipped_whisper += int(skip_mask.sum()) - pre_logo
                unc = 1.0 - 2.0 * np.abs(proba - 0.5)
                unc_masked = np.where(skip_mask, -1.0, unc)
                top_unc_idx = np.argsort(-unc_masked)[:k_unc]
                top_unc = set(int(i) for i in top_unc_idx
                              if unc_masked[i] >= 0)
                top_div = set()
                slug = _slug
                start = uuid_start.get(uuid, 0)
                if minute_prior.get(slug) and start and k_div > 0:
                    prior_arr = np.array(minute_prior[slug])
                    minutes = ((start + np.arange(n) / args.fps_extract)
                               // 60 % 60).astype(int)
                    p_prior = prior_arr[minutes]
                    div = np.abs(proba - p_prior)
                    # Only surface divergence frames where the head is
                    # actually CONFIDENT (|p-0.5| > 0.3) — otherwise
                    # they overlap with the unc bucket and add nothing.
                    confident_mask = np.abs(proba - 0.5) > 0.3
                    div_masked = np.where(confident_mask & ~skip_mask,
                                          div, -1.0)
                    top_div_idx = np.argsort(-div_masked)[:k_div]
                    top_div = set(int(i) for i in top_div_idx
                                  if div_masked[i] >= 0)
                # dedupe + chronological order; tag source for the UI
                rows = []
                for i in sorted(top_unc | top_div):
                    src = "div" if (i in top_div and i not in top_unc) else (
                          "both" if (i in top_unc and i in top_div) else "unc")
                    t = i / args.fps_extract
                    rows.append((t, proba[i], src))
                for t, p, src in rows:
                    f.write(f"{uuid}\t{t:.1f}\t{p:.3f}\t{title[:35]}\t{src}\n")
                    emitted += 1
        cap = _budget
        print(f"\nactive-learning: base top-{n_unc} uncertain + top-{n_div} "
              f"divergent per recording, cohort-weighted → {out_path}")
        # ⚠️ Quelle mitloggen: fällt die Gewichtung still auf den Kandidaten
        # zurück (fehlendes head-to-head), lenkt sie das Label-Budget wieder
        # nach den Schwächen eines Modells, das nie ausgeliefert wird.
        print(f"  cohort-IoU-Quelle: {_pri_quelle} ({len(_pri)} recs)")
        print(f"  cohort budget: {_mult_hist.get(3, 0)} recs ×3 (IoU<0.30), "
              f"{_mult_hist.get(2, 0)} ×2 (IoU<0.50), {_mult_hist.get(1, 0)} ×1")
        print(f"  emitted {emitted} (cap {cap}, "
              f"−{cap - emitted} from filters)")
        print(f"  corpus filter populations: {skipped_logo} logo-sentinel, "
              f"{skipped_whisper} whisper-agreement")

    # ── Champion-challenger gate ──────────────────────────────────
    # Don't ship a head that regressed against the last successful
    # one. Keeps Stufe 2 (pseudo-labels) and any future label-quality
    # experiment from silently degrading inference. Decision uses
    # the 10s-smoothed test metrics — that's what the deployed
    # blocks.Form() actually consumes (NNSmoothS=10 default).
    history_path = Path(args.output).with_suffix(".history.json")
    archive_dir = Path(args.output).parent / "archive"
    archive_dir.mkdir(exist_ok=True)
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []
    last_deployed = next((h for h in reversed(history)
                          if h.get("deployed")), None)

    deploy = True
    reason = "first run" if last_deployed is None else "no regression"
    if args.reset_baseline and metrics_smooth:
        # Operator-forced one-time baseline reset: the test-set population
        # fundamentally changed (e.g. the 2026-06-02 corpus fix recovered ~100
        # disk-pruned reviewed recordings), so the historical IoU floor —
        # anchored to the old, smaller, non-representative test set — is not a
        # valid comparison. Skip the floor; deploy only if the head clears an
        # ABSOLUTE sanity bar so a reset never re-anchors on garbage. Future
        # runs compare against THIS baseline.
        cur_iou_med = metrics_smooth.get(
            "iou_tv_median",
            metrics_smooth.get("iou_median", metrics_smooth["iou"]))
        if cur_iou_med >= args.reset_baseline_floor:
            deploy = True
            reason = (f"baseline reset (--reset-baseline): median-IoU "
                      f"{cur_iou_med:.3f} on {metrics_smooth['n_recs']} test "
                      f"recs >= {args.reset_baseline_floor:.2f} sanity floor — "
                      f"re-anchoring after test-set population change")
        else:
            deploy = False
            reason = (f"baseline reset refused: median-IoU {cur_iou_med:.3f} < "
                      f"{args.reset_baseline_floor:.2f} absolute sanity floor")
    elif last_deployed and metrics_smooth:
        prev_n = last_deployed.get("n_test_recs", 0)
        cur_n  = metrics_smooth["n_recs"]
        prev_feat = last_deployed.get("n_features", 0)
        # MLP modes deploy a head whose input_dim includes the channel
        # one-hot (and whisper) blocks appended on top of X_train. The
        # history-comparison must use the SAME effective dim or the
        # delta is reported wrong (= a channel-only → channel+whisper
        # MLP cutover would log "1291→1282" instead of "1291→1292"
        # because X_train.shape[1] is the LogReg base dim 1282).
        # is_mlp_write is computed below the deploy block so we
        # inline-recompute the equivalent (= same boolean) here.
        if wants_mlp and mlp_prod_clf is not None:
            cur_feat = mlp_prod_in_dim
        else:
            cur_feat = X_train.shape[1] if hasattr(X_train, "shape") else 0
        # Fallback when history.json lacks n_features (legacy entries
        # written before that field was added): infer from head.bin
        # file size — each weight is 4 B float32, plus 4 B bias.
        if not prev_feat and Path(args.output).exists():
            try:
                sz = Path(args.output).stat().st_size
                if sz > 4 and sz % 4 == 0:
                    prev_feat = (sz - 4) // 4
            except Exception:
                pass
        # Only a LARGE feature-dim change is a genuine architecture switch
        # (new feature TYPE: whisper/audio block added, head re-shaped) where
        # the old baseline is incomparable and a reset is warranted. A SMALL
        # change is just the channel-one-hot block growing/shrinking as the
        # channel-map tracks whichever channels have labels this cycle — that
        # is NOT an architecture change and must NOT waive the IoU floor.
        # 2026-05-30 regression: a 10→7 channel-map shrink (dim 1293→1290)
        # took the bypass and deployed a median-IoU 0.85 head that the floor
        # (0.863) would have rejected — and DID reject for two sibling runs
        # the same night. Channel-column changes now fall through to the
        # floor/regression checks below.
        CHANNEL_DIM_TOL = 32  # channel-one-hot block is <= ~20 slugs
        if (prev_feat and cur_feat and prev_feat != cur_feat
                and abs(cur_feat - prev_feat) > CHANNEL_DIM_TOL):
            reason = (f"feature dim changed ({prev_feat}→{cur_feat}) — "
                      f"architecture switch, deploying & resetting baseline")
        elif deployed_test_metrics is not None:
            # Head-to-head: candidate AND the currently-deployed head were both
            # scored on THIS exact test set, so compare them directly — apples-
            # to-apples, robust to test-set composition changes (no historical
            # IoU floor that false-rejects when the corpus shifts; replaces the
            # old --reset-baseline dance). Deploy unless the candidate is a real
            # regression vs the deployed head on the same data.
            def _med(m):
                return m.get("iou_tv_median", m.get("iou_median", m.get("iou", 0.0)))
            # PAIRED comparison: both heads scored the SAME recs, so compare PER
            # REC (candidate_iou − champion_iou) instead of two independent
            # medians. Per-rec difficulty cancels in the delta — a hard rec drags
            # both heads equally — so the ±0.05 test-set noise that made a true
            # tie look like a 0.05 "regression" (2026-06-05: same head scored
            # 0.802 then 0.757 on different samples) is gone. A bootstrap CI on
            # the median delta then decides significance, so a noisy near-tie
            # can't flip the gate. Keep the champion ONLY on a confident
            # regression; otherwise deploy the fresher candidate.
            cand_pr = metrics_smooth.get("per_rec_iou") or {}
            dep_pr = deployed_test_metrics.get("per_rec_iou") or {}
            shared = [u for u in cand_pr if u in dep_pr]
            if len(shared) >= 10:
                deltas = np.array([cand_pr[u] - dep_pr[u] for u in shared])
                med_d = float(np.median(deltas))
                n_better = int((deltas > 0.02).sum())
                n_worse = int((deltas < -0.02).sum())
                rng = np.random.default_rng(0)  # deterministic CI
                boot = np.array([np.median(rng.choice(deltas, len(deltas), replace=True))
                                 for _ in range(2000)])
                lo, hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
                PAIRED_SLACK = 0.005
                base = (f"head-to-head PAIRED on {len(shared)} recs: median Δ "
                        f"{med_d:+.3f} (90% CI [{lo:+.3f},{hi:+.3f}]), "
                        f"{n_better} better / {n_worse} worse")
                # Broad-flat regression guard: the median delta is robust to
                # tails, so a candidate slightly worse on MANY recs but unchanged
                # at the MEDIAN rec sails through hi<-SLACK. 2026-06-09: median Δ
                # -0.002 (passed) yet 19 worse / 8 better quietly dropped the
                # overall TV-median 0.80→0.70 and tripped the dashboard's -3.35pp
                # 7-day-drift flag. n_better/n_worse already count only MEANINGFUL
                # changes (|Δ|>0.02), so a ≥2× worse-lean with a clear margin is a
                # real broad regression the robust median misses — keep champion.
                broad_margin = max(5, round(0.15 * len(shared)))
                # Reviewed-regression veto: the median gate is robust to the
                # asymmetric negative tail BY DESIGN, so a candidate can silently
                # lose a big chunk of IoU on a handful of recs and still deploy on
                # median Δ 0 (2026-07-24: −0.50 / −0.35 / −0.32 movers, all sailed
                # through). That is acceptable when the loss is on AUTO-labelled
                # recs (their ground truth is itself model output — a "drop" may
                # just be the labels changing). It is NOT acceptable on a rec the
                # user REVIEWED: there the ground truth is trusted, so a >veto drop
                # is a genuine local regression the median must not paper over.
                reviewed = {r[0] for r in per_rec if len(r) > 5 and r[5]}
                # BISTABLE recs are excluded as veto sources. Some recordings sit
                # exactly on a block-boundary decision edge and flip between two
                # attractors run to run — 2ff4df28 alternates between ~0.48 and
                # ~0.97 with nothing in between, 0a9ddd3e between ~0.47 and ~0.99.
                # A veto that trusts a single run's drop fires whenever the
                # champion landed on the good side and the candidate on the bad
                # one: on 2026-07-25 that blocked a candidate whose overall metric
                # was BETTER (mean 0.81 vs 0.80), and the same pattern would have
                # fired on 07-20 and 07-24 — roughly every third night, i.e. the
                # deadlock this veto was explicitly designed to avoid. Read the
                # per-rec history we already persist and drop any rec whose OWN
                # spread across recent runs exceeds the veto threshold: for such a
                # rec a large delta carries no information about the model.
                unstable = set()
                try:
                    if args.train_archive:
                        _h = Path(args.train_archive) / "per-rec-iou.jsonl"
                        if _h.exists():
                            hist = {}
                            for _ln in _h.read_text().splitlines()[-args.stability_window:]:
                                if not _ln.strip():
                                    continue
                                for _u, _v in (json.loads(_ln).get("candidate") or {}).items():
                                    hist.setdefault(_u, []).append(_v)
                            for _u, _vs in hist.items():
                                if len(_vs) >= 3 and (max(_vs) - min(_vs)) > args.reviewed_regression_veto:
                                    unstable.add(_u)
                except Exception as _e:
                    print(f"  veto stability check unavailable ({_e}) — "
                          f"treating all recs as stable")
                rev_regr = sorted(
                    ((cand_pr[u] - dep_pr[u], u) for u in shared
                     if u in reviewed and u not in unstable
                     and dep_pr[u] - cand_pr[u] > args.reviewed_regression_veto
                     and cand_pr[u] < args.reviewed_regression_floor),
                    key=lambda x: x[0])
                _skipped = [u for u in shared if u in reviewed and u in unstable
                            and dep_pr[u] - cand_pr[u] > args.reviewed_regression_veto]
                if _skipped:
                    print(f"  veto: ignoring {len(_skipped)} BISTABLE rec(s) whose "
                          f"own history spans >{args.reviewed_regression_veto:.2f} "
                          f"(flip-flop, not regression):")
                    for u in _skipped:
                        t, c = uuid_cohort.get(u, ("", ""))
                        print(f"    {cand_pr[u] - dep_pr[u]:+.3f}  {t or u} / {c}")
                # Fire on a PATTERN (≥2 reviewed regressions) or a single
                # CATASTROPHIC one (≥ severe). A lone moderate drop is logged but
                # deploys — it may be a degenerate no-ad / stale-label rec, and
                # blocking on one rec risks the single-rec deadlock.
                severe = [(d, u) for d, u in rev_regr
                          if -d >= args.reviewed_regression_severe]
                veto_fires = (args.reviewed_regression_veto < 1.0
                              and (len(rev_regr) >= 2 or bool(severe)))
                if hi < -PAIRED_SLACK:  # 95%-confident the candidate is worse
                    deploy = False
                    reason = base + " — confident regression, keeping current head"
                elif (n_worse >= 2 * max(n_better, 1)
                      and n_worse - n_better >= broad_margin):
                    deploy = False
                    reason = base + (f" — broad regression ({n_worse}≥2×{n_better}, "
                                     f"margin≥{broad_margin}), keeping current head")
                elif veto_fires:
                    # Median gate would deploy, but a trusted-ground-truth
                    # regression pattern/catastrophe overrides it.
                    deploy = False
                    worst_d, worst_u = rev_regr[0]
                    wt, wc = uuid_cohort.get(worst_u, ("", ""))
                    wlabel = f"{wt} / {wc}" if wt else worst_u
                    trig = ("catastrophic" if severe and len(rev_regr) < 2
                            else f"pattern ({len(rev_regr)} recs)")
                    reason = base + (
                        f" — REVIEWED-REGRESSION VETO [{trig}]: reviewed rec(s) "
                        f"dropped >{args.reviewed_regression_veto:.2f} vs champion "
                        f"AND below {args.reviewed_regression_floor:.2f} "
                        f"(worst {worst_d:+.2f} on {wlabel}), keeping current head")
                    print("  REVIEWED-REGRESSION VETO — trusted ground-truth recs "
                          "that regressed:")
                    for d, u in rev_regr:
                        t, c = uuid_cohort.get(u, ("", ""))
                        lbl = f"{t} / {c}" if t else u
                        print(f"    {d:+.3f}  cand={cand_pr[u]:.3f} "
                              f"champ={dep_pr[u]:.3f}  {lbl}  ({u})")
                else:
                    reason = base + " — not a confident regression, deploy"
                    if rev_regr:
                        # Moderate lone regression: logged, not blocking.
                        print("  note: 1 moderate reviewed regression (not a "
                              "pattern, not catastrophic) — logged, deploying:")
                        for d, u in rev_regr:
                            t, c = uuid_cohort.get(u, ("", ""))
                            lbl = f"{t} / {c}" if t else u
                            print(f"    {d:+.3f}  cand={cand_pr[u]:.3f} "
                                  f"champ={dep_pr[u]:.3f}  {lbl}  ({u})")

                # Both-heads-cold report: recs where BOTH heads score below the
                # floor are systematic blind spots — the paired gate is structurally
                # blind to them (champion equally bad → Δ≈0), so they never improve
                # on their own. Surface them for label review / feature work.
                both_cold = sorted(
                    ((min(cand_pr[u], dep_pr[u]), u) for u in shared
                     if cand_pr[u] < args.both_cold_floor
                     and dep_pr[u] < args.both_cold_floor),
                    key=lambda x: x[0])
                if both_cold:
                    print(f"  BOTH-HEADS-COLD ({len(both_cold)} recs < "
                          f"{args.both_cold_floor:.2f} on BOTH heads — systematic "
                          f"blind spots invisible to the paired gate):")
                    for io, u in both_cold[:12]:
                        t, c = uuid_cohort.get(u, ("", ""))
                        lbl = f"{t} / {c}" if t else u
                        rev = " [reviewed]" if u in reviewed else ""
                        print(f"    cand={cand_pr[u]:.3f} champ={dep_pr[u]:.3f}  "
                              f"{lbl}{rev}  ({u})")
                    try:
                        _bc = Path(args.output).with_suffix(".both-cold.jsonl")
                        with open(_bc, "w") as f:
                            for io, u in both_cold:
                                t, c = uuid_cohort.get(u, ("", ""))
                                f.write(json.dumps({
                                    "uuid": u, "title": t, "channel": c,
                                    "cand_iou": round(cand_pr[u], 4),
                                    "champ_iou": round(dep_pr[u], 4),
                                    "reviewed": u in reviewed}) + "\n")
                    except Exception as e:
                        print(f"  both-cold persist failed: {e}")
                # Rejection diagnostic — auto-surface WHERE a regression
                # concentrates instead of leaving that to manual per-rec-iou
                # archaeology (2026-07-09 cohort-trust incident took ~20min
                # of log/archive spelunking to trace to a single show/channel
                # cohort). Worst individual recs + aggregated by (title,
                # channel) + by channel alone, so a systemic show/channel
                # cluster (like that incident) is visible at a glance instead
                # of only the median-Δ summary line.
                if not deploy:
                    order = np.argsort(deltas)
                    print("  REJECTION DIAGNOSTIC — worst-regressed recordings:")
                    for i in order[:10]:
                        u = shared[i]
                        title, chan = uuid_cohort.get(u, ("", ""))
                        label = f"{title} / {chan}" if title else u
                        print(f"    {deltas[i]:+.3f}  cand={cand_pr[u]:.3f} "
                              f"champ={dep_pr[u]:.3f}  {label}  ({u})")
                    by_cohort, by_chan = {}, {}
                    for i, u in enumerate(shared):
                        title, chan = uuid_cohort.get(u, ("", ""))
                        if title:
                            by_cohort.setdefault((title, chan), []).append(deltas[i])
                        if chan:
                            by_chan.setdefault(chan, []).append(deltas[i])
                    cohort_avg = sorted(
                        ((k, float(np.mean(v)), len(v)) for k, v in by_cohort.items()
                         if len(v) >= 2),
                        key=lambda x: x[1])[:5]
                    if cohort_avg:
                        print("  REJECTION DIAGNOSTIC — worst (title,channel) cohorts "
                              "(avg Δ, n≥2 recs):")
                        for (title, chan), avg, n in cohort_avg:
                            print(f"    {avg:+.3f} (n={n})  {title} / {chan}")
                    chan_avg = sorted(
                        ((k, float(np.mean(v)), len(v)) for k, v in by_chan.items()),
                        key=lambda x: x[1])[:5]
                    if chan_avg:
                        print("  REJECTION DIAGNOSTIC — worst channels (avg Δ):")
                        for chan, avg, n in chan_avg:
                            print(f"    {avg:+.3f} (n={n})  {chan}")
                # Persist the per-rec pairing — the gate's raw evidence.
                # Until 2026-07-07 only the aggregate reason survived, so
                # diagnosing a rejection streak (which recs drag? one
                # channel? one show?) required re-running training. One
                # jsonl line per run; lives in the train-archive dir
                # because args.output is /tmp (volatile across reboots).
                try:
                    if args.train_archive:
                        _prr = Path(args.train_archive) / "per-rec-iou.jsonl"
                        _prr.parent.mkdir(parents=True, exist_ok=True)
                        with open(_prr, "a") as f:
                            f.write(json.dumps({
                                "ts": time.strftime("%Y%m%dT%H%M%S"),
                                "n_shared": len(shared),
                                "median_delta": round(med_d, 4),
                                "deploy": deploy,
                                "champion_src": _dep_src.name,
                                "n_reviewed_regr": len(rev_regr),
                                "n_both_cold": len(both_cold),
                                "candidate": {u: round(cand_pr[u], 4) for u in shared},
                                "champion": {u: round(dep_pr[u], 4) for u in shared},
                            }) + "\n")
                except Exception as e:
                    print(f"  per-rec-iou persist failed: {e}")
            else:
                # too few shared recs (or old metrics w/o per_rec_iou) → fall back
                # to the independent-median comparison.
                cand, dep = _med(metrics_smooth), _med(deployed_test_metrics)
                if cand < dep - 0.02:
                    deploy = False
                    reason = (f"head-to-head on {cur_n} test recs: candidate "
                              f"TV-median-IoU {cand:.3f} < deployed {dep:.3f} − 0.02 "
                              f"— regression, keeping current head (unpaired fallback)")
                else:
                    reason = (f"head-to-head on {cur_n} test recs: candidate "
                              f"TV-median-IoU {cand:.3f} ≥ deployed {dep:.3f} − 0.02 "
                              f"— deploy (unpaired fallback)")
        elif prev_n != cur_n:
            # Test set changed → mean-IoU comparison is apples-to-
            # oranges (a single Moonfall-style outlier shifts mean
            # 5+ pp). Use MEDIAN-IoU as the gate metric — robust to
            # outliers, captures actual model quality. Plus require
            # the train-corpus to not have collapsed (15.05: cron
            # corpus shrank 216→172 trains over 3 days from cohort-
            # gate dropping auto-confirms; no labels = no training).
            # Prefer TV-class median: movies (rebroadcast of Moonfall
            # etc) have fundamentally different ad structure and a
            # single new film at 0.40 IoU can pull unified median
            # under the floor and reject a deployment that improves
            # TV-class performance. Fallback chain: tv_median →
            # unified median → mean — old history entries that don't
            # have tv_median fall through gracefully.
            cur_iou_med = metrics_smooth.get(
                "iou_tv_median",
                metrics_smooth.get("iou_median", metrics_smooth["iou"]))
            cur_train_n = len(train_recs)
            cur_total_n = cur_train_n + cur_n  # cur_n is current test
            # 3-run window (was 5) so a baseline-shift event (e.g.
            # new channel column → arch-reset deploy with lower IoU)
            # rolls out of the floor calc within 3 cron days. With 5
            # the floor stayed anchored to pre-shift heads for ~5 days,
            # rejecting candidates that were strictly better than the
            # last deployed (2026-05-20: 0.874 rejected vs 0.849 last
            # deployed, but median(last 5) still 0.911 from pre-disney
            # heads).
            recent_deployed = [h for h in reversed(history)
                               if h.get("deployed")][:3]
            recent_ious_med = [h.get("test_iou_tv_median",
                                     h.get("test_iou_median",
                                           h.get("test_iou", 0)))
                               for h in recent_deployed
                               if h.get("test_iou") is not None]
            # Corpus shrinkage gate: check TOTAL corpus (train + test),
            # not train alone. When test grows via UUID-hash split moving
            # a few recordings from train→test, train shrinks but total
            # stays the same — that's not a real corpus collapse, just
            # rebalancing. The original 15.05 incident the gate was
            # added for was a real total-corpus collapse from cohort-
            # gate dropping auto-confirms, which DOES show up in total.
            recent_total_ns = [h.get("n_train_recs", 0) + h.get("n_test_recs", 0)
                               for h in recent_deployed
                               if h.get("n_train_recs", 0)]
            MEDIAN_FLOOR_DROP = 0.03  # 3 pp slack vs median (was 4)
            CORPUS_FLOOR_RATIO = 0.85  # require >=85% of recent total-N
            floor_med = (sorted(recent_ious_med)[len(recent_ious_med)//2]
                         - MEDIAN_FLOOR_DROP) if recent_ious_med else 0
            floor_total_n = int(
                (sorted(recent_total_ns)[len(recent_total_ns)//2]
                 if recent_total_ns else 0) * CORPUS_FLOOR_RATIO)
            if recent_ious_med and cur_iou_med < floor_med:
                deploy = False
                med = sorted(recent_ious_med)[len(recent_ious_med)//2]
                reason = (f"test-set composition changed "
                          f"({prev_n}→{cur_n}), but TV-class median-IoU "
                          f"{cur_iou_med:.3f} < median(last "
                          f"{len(recent_ious_med)})={med:.3f} - "
                          f"{MEDIAN_FLOOR_DROP*100:.0f}pp floor — "
                          f"refusing deploy of likely-regression head")
            elif recent_total_ns and cur_total_n < floor_total_n:
                # Bypass corpus-shrinkage gate when low-prio invalidation
                # drain is still in progress — the missing recordings
                # aren't lost forever, they're awaiting re-detect with
                # the just-deployed head. Cron fires mid-drain look like
                # corpus collapse but resolve naturally. Threshold: if
                # >30 low-prio markers pending, the corpus IS in drain
                # mode (= post-deploy invalidation phase) and the
                # comparison is invalid. 2026-05-17 incident.
                invalidation_pending = 0
                try:
                    import urllib.request
                    invalidation_pending = len(json.loads(
                        urllib.request.urlopen(
                            "http://raspberrypi5lan:8080/api/internal/detect-pending-low",
                            timeout=5).read()).get("pending", []))
                except Exception:
                    pass
                if invalidation_pending > 30:
                    med_n = sorted(recent_total_ns)[len(recent_total_ns)//2]
                    reason = (f"test-set composition changed "
                              f"({prev_n}→{cur_n}), total-corpus "
                              f"{cur_total_n} < median {med_n} but "
                              f"{invalidation_pending} re-detects pending "
                              f"(post-deploy drain) — deploying anyway "
                              f"(median-IoU {cur_iou_med:.3f})")
                else:
                    deploy = False
                    med_n = sorted(recent_total_ns)[len(recent_total_ns)//2]
                    reason = (f"test-set composition changed "
                              f"({prev_n}→{cur_n}), but total-corpus "
                              f"{cur_total_n} (train {cur_train_n}+test {cur_n}) < "
                              f"{int(CORPUS_FLOOR_RATIO*100)}% of "
                              f"median-recent={med_n} (={floor_total_n}) — "
                              f"refusing deploy on shrunk corpus")
            else:
                reason = (f"test-set composition changed ({prev_n}→{cur_n} "
                          f"recordings) — comparison invalidated, deploying "
                          f"(median-IoU {cur_iou_med:.3f}, "
                          f"train+test {cur_train_n}+{cur_n}={cur_total_n})")
        else:
            d_iou = last_deployed["test_iou"] - metrics_smooth["iou"]
            d_acc = last_deployed["test_acc"] - metrics_smooth["acc"]
            if d_iou > args.rollback_iou_drop:
                deploy = False
                reason = (f"test IoU regression {d_iou*100:+.1f} pp "
                          f"(prev {last_deployed['test_iou']:.2f} → "
                          f"{metrics_smooth['iou']:.2f}) > "
                          f"{args.rollback_iou_drop*100:.0f} pp threshold")
            elif d_acc > args.rollback_acc_drop:
                deploy = False
                reason = (f"test acc regression {d_acc*100:+.1f} pp "
                          f"(prev {last_deployed['test_acc']*100:.1f}% → "
                          f"{metrics_smooth['acc']*100:.1f}%) > "
                          f"{args.rollback_acc_drop*100:.0f} pp threshold")
            else:
                reason = (f"test IoU {metrics_smooth['iou']:.2f} (prev "
                          f"{last_deployed['test_iou']:.2f}), "
                          f"test acc {metrics_smooth['acc']*100:.1f}% "
                          f"(prev {last_deployed['test_acc']*100:.1f}%)")

    # Always write the candidate to the archive — useful for manual
    # inspection / rollback even when not deployed. Writer branches
    # on --head-arch:
    #   logreg                  → packed float32 weights + bias
    #   mlp32-channel           → MLP1 v1 (magic-prefixed)
    #   mlp32-channel-whisper   → MLP2 v2 (= v1 + n_whisper header field)
    archive_path = archive_dir / f"head.{ts}.bin"
    is_mlp_write = wants_mlp and mlp_prod_clf is not None

    def _write_head(path, clf=None):
        clf = clf if clf is not None else mlp_prod_clf
        n_logo_used = 1 if args.with_logo else 0
        n_audio_used = 1 if args.with_audio else 0
        n_chan_used = len(mlp_prod_chan_slugs)
        if wants_whispermask:
            write_mlp_head_v5(path, clf,
                              input_dim=mlp_prod_in_dim,
                              hidden_dim=32, backbone_dim=1280,
                              n_logo=n_logo_used,
                              n_audio=n_audio_used,
                              n_channel=n_chan_used,
                              n_whisper=1,
                              n_temporal=3 if wants_churn else 2,
                              n_minuteprior=1, n_whispermask=1)
        elif wants_minuteprior:
            write_mlp_head_v4(path, clf,
                              input_dim=mlp_prod_in_dim,
                              hidden_dim=32, backbone_dim=1280,
                              n_logo=n_logo_used,
                              n_audio=n_audio_used,
                              n_channel=n_chan_used,
                              n_whisper=1, n_temporal=2,
                              n_minuteprior=1)
        elif wants_temporal:
            write_mlp_head_v3(path, clf,
                              input_dim=mlp_prod_in_dim,
                              hidden_dim=32, backbone_dim=1280,
                              n_logo=n_logo_used,
                              n_audio=n_audio_used,
                              n_channel=n_chan_used,
                              n_whisper=1, n_temporal=2)
        elif wants_whisper:
            write_mlp_head_v2(path, clf,
                              input_dim=mlp_prod_in_dim,
                              hidden_dim=32, backbone_dim=1280,
                              n_logo=n_logo_used,
                              n_audio=n_audio_used,
                              n_channel=n_chan_used,
                              n_whisper=1)
        else:
            write_mlp_head_v1(path, clf,
                              input_dim=mlp_prod_in_dim,
                              hidden_dim=32, backbone_dim=1280,
                              n_logo=n_logo_used,
                              n_audio=n_audio_used,
                              n_channel=n_chan_used)

    if is_mlp_write:
        _write_head(archive_path)
    else:
        with open(archive_path, "wb") as f:
            for w in weights:
                f.write(struct.pack("<f", float(w)))
            f.write(struct.pack("<f", bias))

    # GOLDEN-EVAL: composition-CONSTANT median over a frozen set of pinned
    # held-out recs (golden-eval-set.json). The PRODUCTION METRIC printed below is
    # over WHICHEVER recs the sticky split put in test tonight — its median swings
    # ±3-5pp on composition alone (n=68↔104 over July), so it is unreadable as a
    # trend (2026-07-24: the curve looked flat because the noise buried a ~+6pp
    # real climb). This number is over the SAME recs every night, so its movement
    # is the model's, not the sample's. Pinned to test in _is_test → leakage-free.
    #
    # Defined here and called from BOTH the deploy and the reject branch: it lived
    # inside `if deploy:` on its first night (2026-07-25) and that run was
    # rejected, so the trend recorded nothing — a gap exactly on the nights where
    # you most want to know whether the candidate really was worse on the stable
    # set. The persisted `deployed` flag distinguishes "this shipped" from "this is
    # what the rejected candidate would have scored".
    def report_golden_eval():
        try:
            _gpath = (Path(args.train_archive) / "golden-eval-set.json"
                      if args.train_archive else None)
            _gpr = (metrics_smooth or {}).get("per_rec_iou") or {}
            if not (_gpath and _gpath.exists() and _gpr):
                return
            _golden = set(json.loads(_gpath.read_text()).get("uuids", []))
            _gvals = [_gpr[u] for u in _golden if u in _gpr]
            if not _gvals:
                return
            _gmed = float(np.median(_gvals))
            _gmean = float(np.mean(_gvals))
            _tag = "this model" if deploy else "REJECTED candidate"
            print(f"  GOLDEN-EVAL ({_tag}, composition-constant, "
                  f"{len(_gvals)}/{len(_golden)} pinned recs): "
                  f"median {_gmed:.3f}  mean {_gmean:.3f}"
                  + ("  ← compare night-to-night, this is the real trend"
                     if deploy else "  (production keeps its previous value)"))
            # Record WHICH pinned recs were missing, not just how many. On
            # 2026-07-27 the run scored 59/60 and the median moved 0.891 →
            # 0.905; with only `n` persisted there was no way to tell whether
            # the model improved or a hard recording had dropped out mid-cron
            # (the known invalidation-drain race). Composition-constancy is the
            # entire point of this metric, so a night where composition did
            # move has to name the difference instead of hiding it in a count.
            _missing = sorted(_golden - set(_gpr))
            # Persist WHICH set produced this number. The v1 set silently
            # decayed — 35 of its 60 members no longer existed as recordings
            # and 37 had no signals cache, so they scored through the naive
            # threshold grouper instead of blocks.Form. Every median from that
            # era is a mix of two different measurements and nothing in the
            # trend file said so. A hash makes any future change to the set
            # visible at the point of comparison instead of a year later.
            _gmeta = json.loads(_gpath.read_text())
            with open(Path(args.train_archive) / "golden-trend.jsonl", "a") as _gf:
                _gf.write(json.dumps({
                    "ts": ts, "n": len(_gvals),
                    "golden_median": round(_gmed, 4),
                    "golden_mean": round(_gmean, 4),
                    "set_version": _gmeta.get("version", 1),
                    "set_hash": _gmeta.get("set_hash"),
                    # Which block former produced this number. Entries
                    # before 2026-08-05 have no key and were all "form",
                    # while production ran "hsmm" from 07-29 — those
                    # medians are NOT comparable to later ones. Without
                    # this field the discontinuity would look like a
                    # model jump, which is exactly the kind of silent
                    # re-definition the set_hash was added to prevent.
                    "decoder": " ".join(EVAL_DECODER) or "form",
                    "missing": _missing,
                    "deployed": deploy}) + "\n")
            if _missing:
                print(f"  GOLDEN-EVAL WARNING: {len(_missing)} pinned rec(s) "
                      f"absent — tonight's median is NOT composition-constant "
                      f"and must not be compared to previous nights as-is: "
                      f"{', '.join(_missing)}")
        except Exception as _ge:
            print(f"  golden-eval failed: {_ge}")

    deploy, reason = golden_boden(
        deploy, reason,
        golden_floor=args.golden_floor,
        train_archive=args.train_archive,
        cand_pr=(metrics_smooth or {}).get("per_rec_iou") or {},
        champ_pr=(deployed_test_metrics or {}).get("per_rec_iou") or {})

    if deploy:
        if is_mlp_write:
            _write_head(args.output)
            # Honest gate head: the TRAIN-ONLY snapshot, written next to head.bin
            # so the NEXT run's head-to-head compares train-only-vs-train-only.
            # Stays local (Mac /tmp) like head.bin — not shipped to the Pi (the
            # detector uses head.bin; only train-head's gate reads head.gate.bin).
            if mlp_gate_clf is not None:
                _write_head(Path(args.output).with_suffix(".gate.bin"), mlp_gate_clf)
                print(f"  gate head: {Path(args.output).with_suffix('.gate.bin').name} "
                      f"(train-only, honest)")
        else:
            with open(args.output, "wb") as f:
                for w in weights:
                    f.write(struct.pack("<f", float(w)))
                f.write(struct.pack("<f", bias))
        sz = os.path.getsize(args.output)
        if is_mlp_write:
            fmt = ("MLP5 v5" if wants_whispermask else
                   "MLP4 v4" if wants_minuteprior else
                   "MLP3 v3" if wants_temporal else
                   "MLP2 v2" if wants_whisper else "MLP1 v1")
        else:
            fmt = "LogReg packed"
        print(f"\nDEPLOYED → {args.output} ({sz} B, {fmt})")
        print(f"  archive: {archive_path.name}")
        print(f"  reason: {reason}")
        # The headline "train acc NN.N%" printed earlier is the internal
        # LogReg baseline (fit unconditionally for comparison, never
        # deployed) — easy to mistake for THE number since it's the
        # first accuracy figure in the log. This is the one that
        # actually matters: the just-deployed model's own held-out
        # test performance. (2026-07-13: user was watching the LogReg
        # line plateau at 91.4% and reasonably asked why the real
        # metric wasn't visible — it was, just 100+ lines earlier
        # under a per-show table, not labelled as "this is production".)
        if metrics_smooth:
            _prod_med = metrics_smooth.get(
                "iou_tv_median", metrics_smooth.get("iou_median",
                                                     metrics_smooth.get("iou", 0.0)))
            print(f"  PRODUCTION METRIC (this deployed model, held-out "
                  f"test, smooth=10s): acc {metrics_smooth.get('acc', 0)*100:.1f}%  "
                  f"IoU mean {metrics_smooth.get('iou', 0):.2f}  "
                  f"IoU median {_prod_med:.2f}")
            report_golden_eval()
        # Calibration sidecar: read by the gateway's active-learning
        # endpoints and (eventually) by the Go detector. Written
        # next to head.bin so it stays version-locked with the head
        # that produced the calibration. No-op when test_recs was
        # empty or refit dimensions differ.
        if calibration is not None:
            calib_path = Path(args.output).with_suffix(".calibration.json")
            calib_out = dict(calibration)
            calib_out["ts"] = ts
            calib_path.write_text(json.dumps(calib_out, indent=2))
            print(f"  calibration sidecar: {calib_path.name}")
        # Test-set UUIDs sidecar — read by the gateway's prewarm loop
        # to scope post-deploy bulk re-detect to ONLY the test
        # recordings (= what the per-show IoU snapshot needs to be
        # accurate). The remaining ~85 % of train + unreviewed
        # recordings get lazy-regenerated by /recording/<uuid>/ads
        # on next view. Cuts total post-deploy compute from ~3 h to
        # ~25 min on this setup.
        try:
            test_uuids = [r[0] for r in test_recs if r and r[0]]
            ts_path = Path(args.output).with_suffix(".test-set.json")
            ts_path.write_text(json.dumps({
                "ts": ts,
                "n": len(test_uuids),
                "uuids": test_uuids,
            }, indent=2))
            print(f"  test-set sidecar: {ts_path.name} ({len(test_uuids)} uuids)")
        except Exception as e:
            print(f"  test-set sidecar err: {e}")
        # Channel-map sidecar — alphabetically-sorted list of channel
        # slugs present in the (train + test) corpus. Index in the list
        # IS the one-hot column the MLP+channel head would use. Written
        # unconditionally (independent of --shadow-eval / --with-channel)
        # so an MLP head migration in Go has the slug→idx map already
        # version-locked next to head.bin. Empty slugs (= recordings
        # without a known channel) excluded; deploy-time fallback for an
        # unknown slug is "all-zero one-hot" handled by the Go loader.
        try:
            chan_slugs = sorted({uuid_slug.get(r[0], "")
                                 for r in train_recs + test_recs} - {""})
            cm_path = Path(args.output).with_suffix(".channel-map.json")
            cm_path.write_text(json.dumps({
                "ts": ts,
                "version": 1,
                "n": len(chan_slugs),
                "slugs": chan_slugs,
            }, indent=2))
            print(f"  channel-map sidecar: {cm_path.name} "
                  f"({len(chan_slugs)} slugs)")
        except Exception as e:
            print(f"  channel-map sidecar err: {e}")
        # Minute-prior sidecar (v4 heads): the per-channel P(ad | minute-
        # of-hour) table + the neutral fill, version-locked next to
        # head.bin so Go inference looks up the SAME prior the head was
        # trained with (the live table drifts nightly). Written whenever
        # the table exists so a later arch flip finds it in place.
        if minute_prior:
            try:
                mp_path = Path(args.output).with_suffix(".minute-prior.json")
                mp_path.write_text(json.dumps({
                    "ts": ts,
                    "version": 1,
                    "neutral": round(mp_neutral, 4),
                    "priors": minute_prior,
                }, indent=2))
                print(f"  minute-prior sidecar: {mp_path.name} "
                      f"({len(minute_prior)} slugs, neutral {mp_neutral:.3f})")
            except Exception as e:
                print(f"  minute-prior sidecar err: {e}")
        # Per-rec IoU sidecar: the deployed candidate's (and, when the
        # head-to-head ran, the champion's) per-recording IoU + show
        # titles. Turns the "why did the mean move?" morning
        # decomposition into a two-file diff instead of an hour of log
        # archaeology (2026-07-21: the 0.81→0.77 drop took exactly
        # that hour to attribute to GT churn + composition).
        try:
            pr_path = Path(args.output).with_suffix(".per-rec-iou.json")
            _round = lambda d: {u: round(float(v), 4)
                                for u, v in (d or {}).items()}
            pr_path.write_text(json.dumps({
                "ts": ts,
                "candidate": _round(metrics_smooth.get("per_rec_iou")
                                    if metrics_smooth else {}),
                "champion": _round(
                    deployed_test_metrics.get("per_rec_iou")
                    if deployed_test_metrics else {}),
                "titles": {r[0]: r[1] for r in (test_recs or [])},
            }, indent=2))
            print(f"  per-rec-iou sidecar: {pr_path.name} "
                  f"({len(metrics_smooth.get('per_rec_iou') or {})} recs)")
            # Compact log companion: the biggest per-rec movers vs the
            # champion (when the head-to-head ran) — the first thing a
            # "why did the metric move?" morning look needs.
            cand_pri = metrics_smooth.get("per_rec_iou") or {}
            champ_pri = (deployed_test_metrics or {}).get("per_rec_iou") or {}
            movers = sorted(((u, cand_pri[u] - champ_pri[u])
                             for u in cand_pri if u in champ_pri),
                            key=lambda x: abs(x[1]), reverse=True)[:8]
            _titles = {r[0]: r[1] for r in (test_recs or [])}
            if movers and abs(movers[0][1]) >= 0.005:
                print("  top per-rec movers (candidate − champion):")
                for u, dlt in movers:
                    if abs(dlt) < 0.005:
                        break
                    print(f"    {dlt:+.2f}  {u}  "
                          f"{_titles.get(u, '')[:36]}")
        except Exception as e:
            print(f"  per-rec-iou sidecar err: {e}")
        # Archive the FULL bundle (head + its sidecars) under the same ts, so
        # any past champion is restorable as a unit. The live sidecars next to
        # head.bin get overwritten on every deploy, so an archived head.<ts>.bin
        # alone is useless for rollback without its matching channel-map /
        # calibration. 2026-05-30: a regression deployed via the channel-dim
        # bypass turned out un-rollbackable for exactly this reason — only
        # head.bin was archived, the champion's 10-slug channel-map was gone.
        try:
            for suffix in (".calibration.json", ".test-set.json",
                           ".channel-map.json", ".minute-prior.json",
                           ".per-rec-iou.json"):
                live = Path(args.output).with_suffix(suffix)
                if live.exists():
                    (archive_dir / f"head.{ts}{suffix}").write_text(
                        live.read_text())
            print(f"  archived full bundle → head.{ts}.* "
                  f"(rollback-restorable)")
        except Exception as e:
            print(f"  bundle archive err: {e}")
    else:
        print(f"\nREJECTED — kept previous {args.output}")
        print(f"  candidate archived as {archive_path.name}")
        print(f"  reason: {reason}")
        if metrics_smooth:
            _cand_med = metrics_smooth.get(
                "iou_tv_median", metrics_smooth.get("iou_median",
                                                     metrics_smooth.get("iou", 0.0)))
            print(f"  candidate's own held-out metric (NOT deployed — "
                  f"production keeps its current numbers): acc "
                  f"{metrics_smooth.get('acc', 0)*100:.1f}%  "
                  f"IoU mean {metrics_smooth.get('iou', 0):.2f}  "
                  f"IoU median {_cand_med:.2f}")
            report_golden_eval()

    # n_features for the history entry: in MLP1 mode the input dim
    # includes the channel one-hot block (= mlp_prod_in_dim = 1290
    # typical). LogReg mode falls back to X_train's column count.
    if is_mlp_write:
        hist_n_features = mlp_prod_in_dim
    else:
        hist_n_features = (int(X_train.shape[1])
                           if hasattr(X_train, "shape") else 0)
    history.append({
        "ts": ts,
        "n_train_recs": len(train_recs),
        "n_test_recs": len(test_recs),
        "n_features": hist_n_features,
        "arch": args.head_arch,
        "train_acc": float(train_acc),
        "test_acc": float(metrics_smooth["acc"]) if metrics_smooth else None,
        "test_iou": float(metrics_smooth["iou"]) if metrics_smooth else None,
        "test_iou_median": float(metrics_smooth.get("iou_median",
                                                     metrics_smooth["iou"]))
                            if metrics_smooth else None,
        # TV-class metrics (= excludes single-rec movies). Deploy-gate
        # prefers test_iou_tv_median; iou_median stays for display.
        "test_iou_tv": (float(metrics_smooth.get("iou_tv",
                                                   metrics_smooth["iou"]))
                        if metrics_smooth else None),
        "test_iou_tv_median": (float(metrics_smooth.get(
                                       "iou_tv_median",
                                       metrics_smooth.get("iou_median",
                                                            metrics_smooth["iou"])))
                                if metrics_smooth else None),
        "test_f1":  float(metrics_smooth["f1"])  if metrics_smooth else None,
        "deployed": deploy,
        "reason":   reason,
        "archive":  archive_path.name,
    })
    # Cap history at 200 entries — that's ~6 months of nightly runs.
    history = history[-200:]
    history_path.write_text(json.dumps(history, indent=2))

    # Signal the sh-wrapper whether this run actually deployed. Wrapper
    # uses rc=3 to skip the bundle/upload step — without that signal,
    # a rejected candidate's previous-head local file (which can be
    # stale vs Pi after a manual rollback) gets bundled + pushed,
    # silently overwriting Pi's current head with the local stale
    # version (= 14.05: rejected today's 0.87 candidate, but bundle
    # still uploaded the morning's 0.83 regression head that was
    # left in /tmp from before the manual rollback). Pi-state is
    # source of truth when no new head is being shipped.
    return 0 if deploy else 3


if __name__ == "__main__":
    sys.exit(main())
