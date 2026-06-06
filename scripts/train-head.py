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
import json
import os
import re
import struct
import subprocess
import sys
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
    """Parse a v2 MLP head.bin into a _DeployedMLP, or None if it isn't one
    (legacy logreg head / missing / corrupt). Used for the head-to-head deploy
    gate; the caller must check .input_dim matches the candidate's feature dim."""
    import struct
    try:
        raw = Path(path).read_bytes()
    except Exception:
        return None
    if len(raw) < 40:
        return None
    hdr = struct.unpack("<10I", raw[:40])
    if hdr[0] != 0x32504C4D or hdr[1] != 2:  # magic "MLP2", version 2
        return None
    input_dim, hidden_dim, output_dim = hdr[2], hdr[3], hdr[4]
    off = 40

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


WHISPER_CACHE = Path.home() / ".cache" / "tv-whisper"


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
    cmd = [tv_detect, "--quiet", "--workers", "4",
           "--logo", str(logo_path)]
    if y_offset > 0:
        cmd += ["--logo-y-offset", str(y_offset)]
    cmd += ["--emit-logo-csv", str(src)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return nan_arr
    # NOTE: do NOT bail on returncode != 0 — partial CSV (= what chunks
    # successfully decoded before the failed one aborted) is still
    # useful. Untouched seconds stay NaN.
    sums = np.zeros(n_seconds, dtype=np.float64)
    counts = np.zeros(n_seconds, dtype=np.int32)
    for line in out.stdout.splitlines():
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
    for uuid, title, ads, X, y, *_ in recs:
        proba = clf.predict_proba(X)[:, 1]
        if half_w > 0:
            proba = smooth_mean(proba, half_w)
        pred = (proba >= 0.5).astype(np.int32)
        n = len(y)
        correct = int((pred == y).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
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
    ap.add_argument("--output", default="/tmp/tv-train-head-out/head.bin")
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
                             "mlp32-channel-whisper"],
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
                         "restarts.")
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
    print(f"cohort-trust: {len(suspect_cohorts)} (title,channel) cohorts "
          f"have ≥1 user-confirmed-with-ads — auto-confirm-empty in those "
          f"cohorts will be treated as bootstrap (not labels)")

    # Pass 1 — discover labelled recordings, separate cached from
    # uncached. Cached ones we just load synchronously; uncached
    # ones go to the worker pool.
    cached, todo = [], []  # cached: (rec_info, cache_path); todo: (rec_info, src, cache_path)
    corpus_no_ts = 0       # recovered from feature cache because the .ts was dedup'd
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
                         pseudo_data, is_bootstrap)
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
                         pseudo_data, is_bootstrap)
            cached.append((rec_info, cache_path))
            corpus_no_ts += 1

    if corpus_no_ts:
        print(f"corpus: recovered {corpus_no_ts} recording(s) from the feature "
              f"cache whose .ts was dedup'd (kept in corpus, no re-extract)")

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
    per_rec = []  # list of (uuid, title, ads, X, y, has_user)
    dropped_high = []
    logo_nan_offenders = []  # (uuid, title, miss_pct) for log summary
    logo_nan_mask_by_uuid = {}  # uuid → bool array, True where logo was NaN
    for rec_info, cache_path in cached + [(ri, cp) for ri, _, cp in todo]:
        if not Path(cache_path).exists():
            continue
        feats = np.load(cache_path)
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
                         }))
            except Exception as e:
                print(f"  train-archive: write {uuid[:8]} failed: {e}", flush=True)

    # Deletion-safe training archive (read side): admit recordings no longer
    # live (deleted, or .ts dedup'd → dropped from the walk above) that have a
    # frozen entry. Features come from the never-evicted feature cache via the
    # stored path; labels from the archive. Keeps the corpus + sticky test
    # split stable across deletions/dedup. Shape-mismatch (fps/feature-set
    # drift) → skip rather than misalign labels.
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
            fnpy = Path(a_meta.get("feature_npy", ""))
            if not fnpy.exists():
                continue  # features cleared → can't reconstruct; skip
            try:
                a_feats = np.load(fnpy)
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
            per_rec.append((u, a_meta.get("title", ""), a_meta.get("ads", []),
                            a_feats, a_labels, a_which in ("user", "merged"),
                            a_meta.get("confirmed_show", []),
                            a_meta.get("confirmed_ad_skips", []), a_age,
                            [], None, False, False,
                            a_meta.get("cluster_anchored", [])))
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
                              r[6], r[7], r[8], r[9], r[10], r[11], r[12])
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
    import hashlib
    def _is_test(uuid_str):
        h = int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)
        return (h / 2**128) < args.test_frac
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

    # Label-hygiene pass (Stufe 2): use the existing head.bin as a
    # teacher to drop frames where labels and teacher strongly
    # disagree. Frames likely-mislabelled (auto-detect boundary off,
    # ROI smear, etc.) get masked out instead of poisoning the next
    # head. Capped per-recording so a broken teacher can't nuke
    # everything.
    teacher_w = teacher_b = None
    feat_dim = per_rec[0][3].shape[1] if per_rec else 0
    if args.hygiene_disagree_conf > 0 and Path(args.output).exists():
        try:
            raw = Path(args.output).read_bytes()
            # Auto-detect head format by raw size; require that the
            # teacher's input dim matches the current feature dim or
            # we'd matmul-mismatch (e.g. switching --with-logo on for
            # the first time produces 1281-dim features but the on-disk
            # teacher is still 1280-weight from the previous training).
            # Try every supported feature-dim (legacy, +logo, +chan,
            # +audio, and combinations). Audio is +1 column, logo is
            # +1, channel is +len(CHANNELS).
            cand_dims = set()
            for L in (0, 1):                 # logo
                for C in (0, len(CHANNELS)): # channel
                    for A in (0, 1):         # audio
                        cand_dims.add(1280 + L + C + A)
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
            teacher_w = None
    keep_masks = []
    drops_total = drops_kept = 0
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
        elif teacher_w is not None:
            logits = r[3] @ teacher_w + teacher_b
            proba = 1.0 / (1.0 + np.exp(-logits))
            # disagreement: label=1 but proba<(1-conf), or label=0 but proba>conf
            disagree = (((r[4] == 1) & (proba < 1 - args.hygiene_disagree_conf)) |
                        ((r[4] == 0) & (proba >     args.hygiene_disagree_conf)))
            drop_rate = disagree.mean()
            if drop_rate > args.hygiene_max_drop_rate:
                # teacher likely wrong, not labels — keep everything
                mask = np.ones(n, dtype=bool)
            else:
                mask = ~disagree
                drops_total += int(disagree.sum())
                drops_kept += 1
        else:
            mask = np.ones(n, dtype=bool)
        keep_masks.append(mask)
    if teacher_w is not None and drops_kept > 0:
        print(f"label-hygiene: dropped {drops_total} frames across "
              f"{drops_kept} recordings (teacher disagreed at conf "
              f">{args.hygiene_disagree_conf})")

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
    print(f"train acc {train_acc*100:.1f}%  L2={float(np.linalg.norm(clf.coef_)):.2f}  "
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
            }
            arrow = "✓" if brier_cal < brier_raw else "✗"
            print(f"calibration: Platt A={A:+.3f} B={B:+.3f}  "
                  f"Brier {brier_raw:.4f} → {brier_cal:.4f} {arrow}")

    def calibrated_proba(X):
        """Apply Platt scaling on top of clf logits when calibration
        is available; fall back to raw predict_proba otherwise. Used
        by the active-learning surface step so 'uncertainty' reflects
        the model's true uncertainty, not the logistic-head's
        over-confidence."""
        if calibration is None:
            return clf.predict_proba(X)[:, 1]
        z = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-(calibration["A"] * z + calibration["B"])))

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
    # Fits the MLPClassifier that will replace the LogReg head.bin if
    # --head-arch=mlp32-channel. Reuses the production X_train +
    # y_train + sw_train (= already hygiene-masked + age-decayed +
    # bumper-boosted) and appends a channel one-hot block post-hoc.
    # Sample weights → integer-rounded oversampling because
    # MLPClassifier doesn't accept sample_weight. Eval on the
    # channel-augmented test set; the resulting metrics override
    # metrics_smooth so the deploy-decision branch downstream operates
    # on what we'll actually ship, not on the LogReg baseline.
    mlp_prod_clf = None
    mlp_gate_clf = None  # train-only head snapshot for the honest head-to-head gate
    mlp_prod_chan_slugs = None
    mlp_prod_in_dim = 0
    wants_mlp = args.head_arch in ("mlp32-channel",
                                    "mlp32-channel-whisper")
    wants_whisper = args.head_arch == "mlp32-channel-whisper"
    if wants_mlp and test_recs:
        from sklearn.neural_network import MLPClassifier
        prod_chan_slugs = sorted({uuid_slug.get(r[0], "")
                                   for r in train_recs + test_recs} - {""})
        prod_chan_idx = {s: i for i, s in enumerate(prod_chan_slugs)}
        n_chan = len(prod_chan_slugs)
        # Per-rec frame counts in the post-hygiene X_train/y_train.
        # The X_train_parts loop above appended (or zeroed) for each
        # entry of train_recs in order, so the parallel zip stays
        # consistent here.
        rec_lengths_train = [len(p) for p in y_train_parts]
        chan_oh_train = np.zeros((len(X_train), n_chan), dtype=np.float32)
        offset = 0
        for r, n in zip(train_recs, rec_lengths_train):
            slug = uuid_slug.get(r[0], "")
            if slug in prod_chan_idx and n > 0:
                chan_oh_train[offset:offset + n,
                              prod_chan_idx[slug]] = 1.0
            offset += n
        # Optional per-frame whisper-prob column (mlp32-channel-whisper
        # only). Aligned by walking train_recs in the same order as
        # rec_lengths_train; per-rec slice of X_train gets a 1-dim
        # column from the recording's whisper.json. Missing whisper
        # → 0.5 fallback per the loader spec.
        if wants_whisper:
            whisper_train = np.full(len(X_train), 0.5, dtype=np.float32)
            offset = 0
            for r, n in zip(train_recs, rec_lengths_train):
                if n <= 0:
                    continue
                wp = _load_whisper_per_sec(r[0], n)
                whisper_train[offset:offset + n] = wp
                offset += n
            X_train_ch = np.hstack(
                [X_train, chan_oh_train,
                 whisper_train.reshape(-1, 1)])
        else:
            X_train_ch = np.hstack([X_train, chan_oh_train])
        # Oversample by integer-rounded sw_train so user-confirmed +
        # bumper-boosted frames retain their elevated influence in
        # the MLP fit (MLPClassifier ignores sample_weight).
        w_int = np.maximum(np.round(sw_train).astype(np.int32), 1)
        sample_idx = np.repeat(np.arange(len(X_train_ch)), w_int)
        X_train_os = X_train_ch[sample_idx]
        y_train_os = y_train[sample_idx]
        print(f"\n=== --head-arch {args.head_arch}: production fit ===")
        print(f"  base train dim: {X_train_ch.shape[1]} "
              f"({n_chan} channels"
              f"{', +whisper' if wants_whisper else ''})")
        print(f"  base train frames: {len(X_train_ch)}")
        print(f"  oversampled frames: {len(X_train_os)}")
        mlp_prod_clf = MLPClassifier(hidden_layer_sizes=(32,),
                                      max_iter=200, random_state=0,
                                      early_stopping=True,
                                      validation_fraction=0.1,
                                      verbose=False)
        mlp_prod_clf.fit(X_train_os, y_train_os)
        print(f"  fit done in {mlp_prod_clf.n_iter_} epochs, "
              f"loss={mlp_prod_clf.loss_:.4f}")

        # Augment test_recs with the same channel one-hot (and
        # whisper if active) for eval.
        def _aug_test(recs):
            out = []
            for r in recs:
                X = r[3]
                T = X.shape[0]
                slug = uuid_slug.get(r[0], "")
                oh = np.zeros((T, n_chan), dtype=np.float32)
                if slug in prod_chan_idx:
                    oh[:, prod_chan_idx[slug]] = 1.0
                if wants_whisper:
                    wp = _load_whisper_per_sec(r[0], T).reshape(-1, 1)
                    X_new = np.hstack([X, oh, wp]).astype(np.float32)
                else:
                    X_new = np.hstack([X, oh]).astype(np.float32)
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
            }
            arrow = "✓" if brier_cal_mlp < brier_raw_mlp else "✗"
            print(f"\nMLP calibration: Platt A={A_mlp:+.3f} "
                  f"B={B_mlp:+.3f}  Brier "
                  f"{brier_raw_mlp:.4f} → {brier_cal_mlp:.4f} {arrow}")

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
        from sklearn.neural_network import MLPClassifier

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

        def _augment_channel(X, slug, _uuid=None):
            T = X.shape[0]
            oh = np.zeros((T, n_chan), dtype=np.float32)
            if slug in chan_idx:
                oh[:, chan_idx[slug]] = 1.0
            return np.hstack([X, oh])

        def _augment_temporal(X, slug, _uuid=None):
            # |X[t] - X[t-1]| and |X[t] - X[t+1]| as 2 extra dims.
            # Captures scene-change signal that per-frame backbone
            # embeddings can't see — boundary detection lever.
            T = X.shape[0]
            dp = np.zeros(T, dtype=np.float32)
            dn = np.zeros(T, dtype=np.float32)
            if T > 1:
                d = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
                dp[1:] = d
                dn[:-1] = d
            return np.column_stack([X, dp, dn]).astype(np.float32)

        def _augment_channel_whisper(X, slug, uuid):
            # Channel one-hot + per-frame whisper prob (1 extra dim).
            # Whisper Stage 4 prototype: instead of running whisper
            # as a post-processor (FP-killer / boundary-extend /
            # missed-adder rules with hand-tuned thresholds), feed
            # the per-second prob directly to the MLP and let it
            # learn the right combination with the other features.
            T = X.shape[0]
            oh = np.zeros((T, n_chan), dtype=np.float32)
            if slug in chan_idx:
                oh[:, chan_idx[slug]] = 1.0
            wp = _load_whisper_per_sec(uuid, T).reshape(-1, 1)
            return np.hstack([X, oh, wp]).astype(np.float32)

        def _build_train(recs, augment):
            Xs, ys, sws = [], [], []
            for r in recs:
                slug = uuid_slug.get(r[0], "")
                X_aug = augment(r[3], slug, r[0])
                y = r[4]
                base_w = args.user_weight if r[5] else 1.0
                sw = np.full(len(y), base_w, dtype=np.float32)
                # NaN-logo skip (= same rationale as the LogReg sw
                # path): zero out frames whose logo column was
                # NaN-substituted so they don't poison the MLP fit.
                nan_mask = logo_nan_mask_by_uuid.get(r[0])
                if nan_mask is not None and len(nan_mask) == len(y):
                    sw[nan_mask] = 0.0
                Xs.append(X_aug); ys.append(y); sws.append(sw)
            return (np.concatenate(Xs), np.concatenate(ys),
                    np.concatenate(sws))

        def _augment_test_recs(recs, augment):
            out = []
            for r in recs:
                slug = uuid_slug.get(r[0], "")
                X_aug = augment(r[3], slug, r[0])
                out.append((r[0], r[1], r[2], X_aug, r[4]) + tuple(r[5:]))
            return out

        def _oversample(X, y, sw):
            # MLPClassifier doesn't accept sample_weight; replicate
            # rows by their integer weight (user_weight=2 → user
            # frames appear 2×). Cheap + faithful approximation.
            w = np.maximum(np.round(sw).astype(np.int32), 1)
            idx = np.repeat(np.arange(len(X)), w)
            return X[idx], y[idx]

        def _fit_eval(name, augment, hidden=(32,)):
            X_aug, y_aug, sw_aug = _build_train(train_recs, augment)
            test_aug = _augment_test_recs(test_recs, augment)
            X_os, y_os = _oversample(X_aug, y_aug, sw_aug)
            print(f"\n=== Shadow variant: {name} ===")
            print(f"  feature dim: {X_aug.shape[1]}, "
                  f"train frames: {len(X_aug)} → {len(X_os)} oversampled, "
                  f"hidden: {hidden}")
            mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=80,
                                random_state=0, early_stopping=True,
                                validation_fraction=0.1, verbose=False)
            mlp.fit(X_os, y_os)
            print(f"  fit done in {mlp.n_iter_} epochs, "
                  f"loss={mlp.loss_:.4f}")
            metrics = eval_split(mlp, test_aug, args.fps_extract, smooth_s=10)
            return metrics, mlp, X_aug.shape[1]

        print("\n" + "=" * 70)
        print(f"SHADOW EVAL — {n_chan} channel slugs in corpus")
        print("=" * 70)
        m_v1, mlp_v1, _ = _fit_eval(
            "MLP-32 (1282→32→1)", lambda X, _s, _u=None: X)
        m_v2, mlp_v2, in_dim_v2 = _fit_eval(
            f"MLP-32 + channel one-hot (+{n_chan} dim)", _augment_channel)
        m_v3, mlp_v3, _ = _fit_eval(
            "MLP-32 + temporal L2 deltas (+2 dim)", _augment_temporal)
        # Whisper Stage 4 prototype — per-second whisper-prob as a
        # direct MLP input column instead of the post-processor rules.
        # n=149 reviews + MLP non-linearity should absorb the new
        # column without the L2-balance fragility of a LogReg add.
        # If +0.03 IoU vs MLP+channel → migrate; if neutral → keep
        # the post-processor rules (= they already gain +5.4 % on
        # n=9 and the rules are simpler to reason about).
        m_v4, mlp_v4, _ = _fit_eval(
            f"MLP-32 + channel + whisper-prob (+{n_chan + 1} dim)",
            _augment_channel_whisper)

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

        base_iou = metrics_smooth["iou"]
        base_acc = metrics_smooth["acc"]
        print("\n" + "=" * 70)
        print("SHADOW COMPARISON (smooth=10s, test set)")
        print("=" * 70)
        print(f"  {'arch':35s}  {'IoU':>6}  {'Δ vs LR':>9}  {'acc':>6}")
        print(f"  {'baseline LogReg (production)':35s}  "
              f"{base_iou:>6.3f}  {'(ref)':>9}  {base_acc*100:>5.1f}%")
        for name, m in [("MLP-32", m_v1),
                        (f"MLP-32 + channel ({n_chan})", m_v2),
                        ("MLP-32 + temporal", m_v3),
                        (f"MLP-32 + channel + whisper", m_v4)]:
            d = m["iou"] - base_iou
            mark = "  ↑" if d > 0.01 else ("  ↓" if d < -0.01 else "")
            print(f"  {name:35s}  {m['iou']:>6.3f}  "
                  f"{d:>+9.3f}  {m['acc']*100:>5.1f}%{mark}")
        # Stage-4 specific Δ vs Stage-3 baseline (= MLP+channel,
        # the current production architecture). The Δ-vs-LogReg
        # column above tells you "is the structural change worth
        # the format migration"; this Δ tells you "is whisper as
        # an MLP input column better than whisper as a post-
        # processor rule set". Migration breaks even somewhere
        # around +0.03 IoU.
        d_stage4 = m_v4["iou"] - m_v2["iou"]
        mark4 = ("  ↑ migrate" if d_stage4 > 0.03 else
                 ("  ↓ keep post-processor" if d_stage4 < -0.01 else
                  "  ≈ neutral, keep post-processor"))
        print()
        print(f"  Δ vs Stage-3 baseline (MLP+channel): "
              f"{d_stage4:+.3f}{mark4}")
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
            n_total = n_pseudo = n_correct = 0
            per_chan_stats = {}  # slug -> [n_pseudo, n_correct]
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
                # Confidence + agreement filter
                conf_ad = (proba >= conf_th) & (p_prior >= 0.5)
                conf_show = (proba <= 1 - conf_th) & (p_prior < 0.5)
                pseudo_mask = conf_ad | conf_show
                pseudo_label = np.where(conf_ad, 1, 0)
                # Accuracy on the kept-frames
                correct = (pseudo_label[pseudo_mask] == y_truth[pseudo_mask]).sum()
                n_pseudo += int(pseudo_mask.sum())
                n_correct += int(correct)
                per_chan_stats.setdefault(slug, [0, 0])
                per_chan_stats[slug][0] += int(pseudo_mask.sum())
                per_chan_stats[slug][1] += int(correct)
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
                print(f"\n  per-channel breakdown:")
                for slug, (npi, nci) in sorted(per_chan_stats.items()):
                    if npi == 0:
                        print(f"    {slug:14s}  no candidates")
                    else:
                        print(f"    {slug:14s}  {nci:>5}/{npi:<5}  "
                              f"acc {100*nci/npi:5.1f}%")

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
                    conf_ad = (proba >= conf_th) & (p_prior >= 0.5)
                    conf_show = (proba <= 1 - conf_th) & (p_prior < 0.5)
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
                        "threshold": conf_th,
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
        from sklearn.neural_network import MLPClassifier
        target_dim_all = max(r[3].shape[1] for r in per_rec)
        keep_all = [r for r in per_rec
                    if r[3].shape[1] == target_dim_all
                       and not (len(r) > 12 and r[12])]
        n_chan_prod = len(mlp_prod_chan_slugs)
        prod_chan_idx_all = {s: i for i, s in enumerate(mlp_prod_chan_slugs)}
        # Per-rec channel one-hot block (+ whisper if active), row-by-
        # row to match the per-rec X concatenation order.
        X_parts = []
        y_parts = []
        for r in keep_all:
            X = r[3]
            T = X.shape[0]
            oh = np.zeros((T, n_chan_prod), dtype=np.float32)
            slug = uuid_slug.get(r[0], "")
            if slug in prod_chan_idx_all and T > 0:
                oh[:, prod_chan_idx_all[slug]] = 1.0
            if wants_whisper:
                wp = _load_whisper_per_sec(r[0], T).reshape(-1, 1)
                X_parts.append(np.hstack([X, oh, wp]).astype(np.float32))
            else:
                X_parts.append(np.hstack([X, oh]).astype(np.float32))
            y_parts.append(r[4])
        X_all_ch = np.concatenate(X_parts) if X_parts else np.empty((0, 0))
        y_all_ch = np.concatenate(y_parts) if y_parts else np.empty(0)
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
        w_int = np.maximum(np.round(sw_all_ch).astype(np.int32), 1)
        sample_idx = np.repeat(np.arange(len(X_all_ch)), w_int)
        X_all_os = X_all_ch[sample_idx]
        y_all_os = y_all_ch[sample_idx]
        print(f"\nrefitting MLP on all data for production head...")
        print(f"  base: {len(X_all_ch)} frames "
              f"({len(keep_all)}/{len(per_rec)} recs), "
              f"oversampled: {len(X_all_os)} frames")
        mlp_prod_clf = MLPClassifier(hidden_layer_sizes=(32,),
                                      max_iter=200, random_state=0,
                                      early_stopping=True,
                                      validation_fraction=0.1,
                                      verbose=False)
        mlp_prod_clf.fit(X_all_os, y_all_os)
        full_acc = (mlp_prod_clf.predict(X_all_os) == y_all_os).mean()
        print(f"  full-data fit acc {full_acc*100:.1f}%, "
              f"epochs {mlp_prod_clf.n_iter_}, "
              f"loss {mlp_prod_clf.loss_:.4f}")
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
        skipped_logo = 0
        skipped_whisper = 0
        emitted = 0
        with open(out_path, "w") as f:
            f.write("# uuid\ttime_s\tprobability\ttitle\tsource\n")
            for uuid, title, ads, X, y, *_ in per_rec:
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
                top_unc_idx = np.argsort(-unc_masked)[:n_unc]
                top_unc = set(int(i) for i in top_unc_idx
                              if unc_masked[i] >= 0)
                top_div = set()
                slug = uuid_slug.get(uuid, "")
                start = uuid_start.get(uuid, 0)
                if minute_prior.get(slug) and start and n_div > 0:
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
                    top_div_idx = np.argsort(-div_masked)[:n_div]
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
        cap = (n_unc + n_div) * len(per_rec)
        print(f"\nactive-learning: top-{n_unc} uncertain + top-{n_div} divergent "
              f"frames per recording → {out_path}")
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
                if hi < -PAIRED_SLACK:  # 95%-confident the candidate is worse
                    deploy = False
                    reason = base + " — confident regression, keeping current head"
                else:
                    reason = base + " — not a confident regression, deploy"
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
    ts = time.strftime("%Y%m%dT%H%M%S")
    archive_path = archive_dir / f"head.{ts}.bin"
    is_mlp_write = wants_mlp and mlp_prod_clf is not None

    def _write_head(path, clf=None):
        clf = clf if clf is not None else mlp_prod_clf
        n_logo_used = 1 if args.with_logo else 0
        n_audio_used = 1 if args.with_audio else 0
        n_chan_used = len(mlp_prod_chan_slugs)
        if wants_whisper:
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
            fmt = "MLP2 v2" if wants_whisper else "MLP1 v1"
        else:
            fmt = "LogReg packed"
        print(f"\nDEPLOYED → {args.output} ({sz} B, {fmt})")
        print(f"  archive: {archive_path.name}")
        print(f"  reason: {reason}")
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
        # Archive the FULL bundle (head + its sidecars) under the same ts, so
        # any past champion is restorable as a unit. The live sidecars next to
        # head.bin get overwritten on every deploy, so an archived head.<ts>.bin
        # alone is useless for rollback without its matching channel-map /
        # calibration. 2026-05-30: a regression deployed via the channel-dim
        # bypass turned out un-rollbackable for exactly this reason — only
        # head.bin was archived, the champion's 10-slug channel-map was gone.
        try:
            for suffix in (".calibration.json", ".test-set.json",
                           ".channel-map.json"):
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
