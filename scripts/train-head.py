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
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", src,
    ]).decode().strip()
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
        out = subprocess.run([
            "ffmpeg", "-nostdin", "-nostats", "-i", str(src),
            "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
            "-af", (f"asetnsamples=n={sample_rate},"
                    f"astats=metadata=1:reset=1,"
                    f"ametadata=mode=print:key=lavfi.astats.Overall.RMS_level"),
            "-f", "null", "/dev/null"
        ], capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return neutral
    rms_db_seq = []
    for line in out.stderr.splitlines():
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

    Falls back to a constant 0.5 array on any failure (missing
    template, tv-detect crash, parse glitch) so training doesn't die
    just because one recording is missing a logo."""
    neutral = np.full(n_seconds, 0.5, dtype=np.float32)
    if not logo_path or not Path(logo_path).exists():
        return neutral
    cmd = [tv_detect, "--quiet", "--workers", "4",
           "--logo", str(logo_path)]
    if y_offset > 0:
        cmd += ["--logo-y-offset", str(y_offset)]
    cmd += ["--emit-logo-csv", str(src)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return neutral
    if out.returncode != 0:
        return neutral
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
    result = neutral.copy()
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


def eval_split(clf, recs, fps_extract, smooth_s=0):
    """Per-frame + block-level evaluation of `clf` on the held-out
    recordings, broken down by show title (proxy for channel since
    the channel slug isn't in the cache index).

    smooth_s > 0: apply the same rolling-mean smoothing the Go
    pipeline uses (NNSmoothS) so the eval reflects deployment."""
    suffix = f" (smooth={smooth_s}s)" if smooth_s > 0 else ""
    print(f"\n=== held-out evaluation{suffix} ===")
    by_show = {}  # title -> {frames, correct, tp, fp, fn, ious[], n_recs}
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
    tp = sum(b["tp"] for b in by_show.values())
    fp = sum(b["fp"] for b in by_show.values())
    fn = sum(b["fn"] for b in by_show.values())
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(f"{'OVERALL':40s} {len(recs):>4} {overall_frames:>7} "
          f"{overall_acc*100:>5.1f}% {f1:>5.2f} {overall_iou:>5.2f}")
    return {"acc": overall_acc, "f1": f1, "iou": overall_iou,
            "n_recs": len(recs), "n_frames": overall_frames}


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
    Both Go inference and the cache key encoding rely on this exact
    order — keep concat operations in the same sequence."""
    (src, fps_extract, cache_path, logo_path, tv_detect_bin,
     chan_slug, with_audio, with_yamnet) = args
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
    return cache_path, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=os.path.expanduser(
        "~/mnt/pi-tv/hls/.tvd-models/backbone.onnx"))
    ap.add_argument("--output", default=os.path.expanduser(
        "~/mnt/pi-tv/hls/.tvd-models/head.bin"))
    ap.add_argument("--hls-root", default=os.path.expanduser(
        "~/mnt/pi-tv/hls"))
    ap.add_argument("--feature-cache", default=os.path.expanduser(
        "~/.cache/tvd-features"))
    ap.add_argument("--fps-extract", type=float, default=1.0)
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
                         "are very accurate (~99%) but uniform low weight "
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
                        "~/mnt/pi-tv/hls/.tvd-logos"),
                    help="directory of channel-keyed cached logo "
                         "templates (used by --with-logo extraction).")
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
    ap.add_argument("--source-root", default="",
                    help="alternative root for .ts files (e.g. local SSD "
                         "cache). Falls back to <hls-root>/.. if a "
                         "particular .ts isn't found there.")
    args = ap.parse_args()

    # --co-train forces the feature flags it needs (otherwise the
    # column slicing below points at non-existent columns).
    if args.co_train:
        args.with_logo = True
        args.with_audio = True
        args.with_channel = True

    cache_dir = Path(args.feature_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = build_onnx_session(args.backbone)
    print(f"providers: {sess.get_providers()}")

    # uuid → channel slug map. Used by --with-logo for logo template
    # lookup AND by --with-minute-prior for per-channel histograms.
    # uuid_start carries unix start_time (wall-clock) needed by the
    # minute-prior path to map frame offsets → minute-of-hour buckets.
    # Failure is non-fatal: missing entries fall back to neutral defaults.
    uuid_slug = {}
    uuid_start = {}
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
                "http://raspberrypi5lan:9981/api/dvr/entry/grid?limit=500",
                timeout=10).read())
            for e in entries.get("entries", []):
                u = e.get("uuid"); cn = e.get("channelname", "")
                if u and cn in chname_to_slug:
                    uuid_slug[u] = chname_to_slug[cn]
                if u and e.get("start_real"):
                    uuid_start[u] = int(e["start_real"])
                elif u and e.get("start"):
                    uuid_start[u] = int(e["start"])
            print(f"slug map: {len(uuid_slug)} uuid→slug entries from gateway")
        except Exception as ex:
            print(f"slug map: gateway unreachable ({ex}) — "
                  f"all logo confs will fall back to 0.5", flush=True)

    # Pass 1 — discover labelled recordings, separate cached from
    # uncached. Cached ones we just load synchronously; uncached
    # ones go to the worker pool.
    cached, todo = [], []  # cached: (rec_info, cache_path); todo: (rec_info, src, cache_path)
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
        if isinstance(user_raw, list):
            user_ads, deleted = user_raw, []
        elif isinstance(user_raw, dict):
            user_ads = user_raw.get("ads") or []
            deleted = user_raw.get("deleted") or []
            confirmed_show = [float(x) for x in
                              user_raw.get("confirmed_show", []) or []]
            confirmed_ad_skips = [float(x) for x in
                                   user_raw.get("confirmed_ad_skips", []) or []]
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
            if args.write_pseudo_labels:
                is_bootstrap = True
                which = "bootstrap"
            else:
                continue

        index = rec_dir / "index.m3u8"
        if not index.exists():
            continue
        base_txts = [p for p in rec_dir.glob("*.txt") if not any(
            p.name.endswith(s) for s in (".logo.txt", ".cskp.txt",
                                          ".tvd.txt", ".trained.logo.txt"))]
        if not base_txts:
            base_txts = [p for p in rec_dir.glob("*.cskp.txt")]
        if not base_txts:
            continue
        base = base_txts[0].stem.replace(".cskp", "")
        title = base.split(" $")[0]
        src = None
        if args.source_root:
            cand = Path(args.source_root) / title / f"{base}.ts"
            if cand.exists():
                src = cand
        if src is None:
            cand = Path(args.hls_root).parent / title / f"{base}.ts"
            if cand.exists():
                src = cand
        if src is None:
            continue

        src_mt = int(src.stat().st_mtime)
        # Cache-key suffix: -l1 = logo feature included; -c1 = channel
        # one-hot included; -a1 = audio rms included. Cached file shape
        # depends on the suffix combination; flipping any flag forces a
        # rebuild for the affected entries (old caches stay on disk but
        # become unused).
        suffix = ""
        # -l2 bumps -l1: l2 applies a per-recording cropdetect-derived
        # y-offset to the logo template (letterbox compensation), so
        # cached logoConf values differ from the unshifted -l1 era.
        if args.with_logo:    suffix += "-l2"
        if args.with_channel: suffix += "-c1"
        if args.with_audio:   suffix += "-a1"
        if args.with_yamnet:  suffix += "-y1"
        cache_path = cache_dir / f"{uuid}-{src_mt}-fps{int(args.fps_extract*100)}{suffix}.npy"
        slug = uuid_slug.get(uuid, "")
        rec_info = (uuid, title, ads, which, slug, str(rec_dir), str(src),
                     pseudo_data, is_bootstrap)
        if cache_path.exists():
            cached.append((rec_info, cache_path))
        else:
            todo.append((rec_info, str(src), cache_path))

    # Pass 2 — extract uncached features in parallel. Each worker loads
    # its own ONNX session at init (~100 MB resident); 4 workers × that
    # is fine on M5 Pro.
    if todo:
        flags = []
        if args.with_logo: flags.append("logo")
        if args.with_channel: flags.append("chan")
        if args.with_audio: flags.append("audio")
        if args.with_yamnet: flags.append("yamnet")
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
                     args.with_audio, args.with_yamnet))] = rec_info
            done = 0
            for fut in cf.as_completed(future_map):
                rec_info = future_map[fut]
                cache_path_str, feats = fut.result()
                np.save(cache_path_str, feats)
                done += 1
                print(f"  [{done}/{len(todo)}] {rec_info[0][:8]} {rec_info[1][:35]} → {feats.shape}",
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
    for rec_info, cache_path in cached + [(ri, cp) for ri, _, cp in todo]:
        if not Path(cache_path).exists():
            continue
        feats = np.load(cache_path)
        if feats.shape[0] == 0:
            continue
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
        per_rec.append((uuid, title, ads, feats, labels, has_user,
                        confirmed_show, confirmed_ad_skips, rec_age_days,
                        bumpers, frame_mask, which == "pseudo",
                        is_bootstrap))
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
        sw_train_parts.append(sw_full[mask])
    if confirmed_extra_w:
        print(f"confirmed-show: upweighted {confirmed_extra_w} frame(s) "
              f"from /mark-reviewed (1.2× over base weight)")
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

    # Evaluate on held-out recordings — both raw (matches a deploy
    # without --nn-smooth) and 10s-smoothed (matches the new default).
    metrics_smooth = None
    if test_recs:
        eval_split(clf, test_recs, args.fps_extract, smooth_s=0)
        metrics_smooth = eval_split(clf, test_recs, args.fps_extract,
                                    smooth_s=10)
        if args.emit_confusion:
            confusion_analysis(clf, test_recs, args.fps_extract,
                               smooth_s=10,
                               output_path=Path(args.output).with_suffix(".confusion.txt"))

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

        # head_logo: visual signal — backbone + logo + channel
        logo_view_cols = backbone_cols + (
            [logo_col] if logo_col >= 0 else []) + channel_cols
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
                proba = clf.predict_proba(X)[:, 1]
                unc = 1.0 - 2.0 * np.abs(proba - 0.5)
                top_unc = set(np.argsort(-unc)[:n_unc].tolist())
                top_div = set()
                slug = uuid_slug.get(uuid, "")
                start = uuid_start.get(uuid, 0)
                if minute_prior.get(slug) and start and n_div > 0:
                    prior_arr = np.array(minute_prior[slug])
                    n = len(proba)
                    minutes = ((start + np.arange(n) / args.fps_extract)
                               // 60 % 60).astype(int)
                    p_prior = prior_arr[minutes]
                    div = np.abs(proba - p_prior)
                    # Only surface divergence frames where the head is
                    # actually CONFIDENT (|p-0.5| > 0.3) — otherwise
                    # they overlap with the unc bucket and add nothing.
                    confident_mask = np.abs(proba - 0.5) > 0.3
                    div_masked = np.where(confident_mask, div, -1.0)
                    top_div = set(np.argsort(-div_masked)[:n_div].tolist())
                # dedupe + chronological order; tag source for the UI
                rows = []
                for i in sorted(top_unc | top_div):
                    src = "div" if (i in top_div and i not in top_unc) else (
                          "both" if (i in top_unc and i in top_div) else "unc")
                    t = i / args.fps_extract
                    rows.append((t, proba[i], src))
                for t, p, src in rows:
                    f.write(f"{uuid}\t{t:.1f}\t{p:.3f}\t{title[:35]}\t{src}\n")
        print(f"\nactive-learning: top-{n_unc} uncertain + top-{n_div} divergent "
              f"frames per recording → {out_path}")

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
    if last_deployed and metrics_smooth:
        prev_n = last_deployed.get("n_test_recs", 0)
        cur_n  = metrics_smooth["n_recs"]
        prev_feat = last_deployed.get("n_features", 0)
        cur_feat  = X_train.shape[1] if hasattr(X_train, "shape") else 0
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
        if prev_feat and cur_feat and prev_feat != cur_feat:
            # Architecture changed (e.g. --with-channel toggled, head
            # grew from 1281 → 1287 weights). Direct IoU comparison
            # against the old architecture is meaningless — the new
            # one might be objectively better even if the absolute
            # number on the test set looks worse during the warm-up.
            reason = (f"feature dim changed ({prev_feat}→{cur_feat}) — "
                      f"architecture switch, deploying & resetting baseline")
        elif prev_n != cur_n:
            # Test set changed (e.g. recording added/dropped/relabelled
            # turned from skip→include). Direct metric comparison is
            # apples-to-oranges — a single hard recording can shift the
            # mean by 10+ pp without the model getting worse. Deploy
            # and reset the baseline.
            reason = (f"test-set composition changed ({prev_n}→{cur_n} "
                      f"recordings) — comparison invalidated, deploying")
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
    # inspection / rollback even when not deployed.
    ts = time.strftime("%Y%m%dT%H%M%S")
    archive_path = archive_dir / f"head.{ts}.bin"
    with open(archive_path, "wb") as f:
        for w in weights:
            f.write(struct.pack("<f", float(w)))
        f.write(struct.pack("<f", bias))

    if deploy:
        with open(args.output, "wb") as f:
            for w in weights:
                f.write(struct.pack("<f", float(w)))
            f.write(struct.pack("<f", bias))
        print(f"\nDEPLOYED → {args.output} ({os.path.getsize(args.output)} B)")
        print(f"  archive: {archive_path.name}")
        print(f"  reason: {reason}")
    else:
        print(f"\nREJECTED — kept previous {args.output}")
        print(f"  candidate archived as {archive_path.name}")
        print(f"  reason: {reason}")

    history.append({
        "ts": ts,
        "n_train_recs": len(train_recs),
        "n_test_recs": len(test_recs),
        "n_features": int(X_train.shape[1]) if hasattr(X_train, "shape") else 0,
        "train_acc": float(train_acc),
        "test_acc": float(metrics_smooth["acc"]) if metrics_smooth else None,
        "test_iou": float(metrics_smooth["iou"]) if metrics_smooth else None,
        "test_f1":  float(metrics_smooth["f1"])  if metrics_smooth else None,
        "deployed": deploy,
        "reason":   reason,
        "archive":  archive_path.name,
    })
    # Cap history at 200 entries — that's ~6 months of nightly runs.
    history = history[-200:]
    history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    sys.exit(main())
