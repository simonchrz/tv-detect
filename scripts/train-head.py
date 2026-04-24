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
    w, h = probe_dims(src)
    feats = []
    for buf in extract_frames_via_ffmpeg(src, w, h, fps_extract):
        x = preprocess_one(buf, w, h)
        out = sess.run(["features"], {"frame": x})[0]  # (1, 1280)
        feats.append(out[0])
    return np.stack(feats) if feats else np.zeros((0, 1280), np.float32)


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
    args = ap.parse_args()

    cache_dir = Path(args.feature_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = build_onnx_session(args.backbone)
    print(f"providers: {sess.get_providers()}")

    X_all, y_all, n_recs = [], [], 0
    for rec_dir in sorted(Path(args.hls_root).glob("_rec_*")):
        uuid = rec_dir.name[5:]
        # Find ads source
        user = rec_dir / "ads_user.json"
        auto = rec_dir / "ads.json"
        ads = None
        which = ""
        if user.exists() and args.prefer in ("user", "any"):
            try: ads = json.loads(user.read_text()); which = "user"
            except Exception: pass
        if ads is None and auto.exists() and args.prefer in ("auto", "any"):
            try: ads = json.loads(auto.read_text()); which = "auto"
            except Exception: pass
        if ads is None:
            continue

        # Find source file
        index = rec_dir / "index.m3u8"
        if not index.exists():
            continue
        # Locate via tvheadend API or scan parent — simpler: assume the
        # caller has /recordings mounted at HLS_ROOT/.. (same convention
        # as tv-comskip.sh). We use the cskp.txt or current .txt
        # filename to derive the source basename, then look it up.
        base_txts = [p for p in rec_dir.glob("*.txt") if not any(
            p.name.endswith(s) for s in (".logo.txt", ".cskp.txt",
                                          ".tvd.txt", ".trained.logo.txt"))]
        if not base_txts:
            base_txts = [p for p in rec_dir.glob("*.cskp.txt")]
        if not base_txts:
            continue
        base = base_txts[0].stem.replace(".cskp", "")
        # The recording's source TS lives under <hls-root>/../<title>/<base>.ts
        title = base.split(" $")[0]
        src = Path(args.hls_root).parent / title / f"{base}.ts"
        if not src.exists():
            continue

        # Cache key: uuid + src mtime
        src_mt = int(src.stat().st_mtime)
        cache_path = cache_dir / f"{uuid}-{src_mt}-fps{int(args.fps_extract*100)}.npy"
        if cache_path.exists():
            feats = np.load(cache_path)
        else:
            print(f"extract {uuid[:8]} {title[:40]} ({which})...", flush=True)
            t0 = time.time()
            feats = featurize_recording(sess, str(src), args.fps_extract)
            print(f"  → {feats.shape} in {time.time()-t0:.1f}s")
            np.save(cache_path, feats)
        if feats.shape[0] == 0:
            continue

        seconds = [i / args.fps_extract for i in range(feats.shape[0])]
        labels = labels_for(seconds, ads)
        X_all.append(feats)
        y_all.append(labels)
        n_recs += 1

    if not X_all:
        print("no labelled recordings found", file=sys.stderr)
        sys.exit(1)
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    print(f"\ntraining on {n_recs} recordings, "
          f"{len(y)} frames, {int(y.sum())} ad ({100*y.mean():.1f}%)")

    # Logistic regression: Linear(1280) + sigmoid + BCE.
    # Solve via sklearn so we don't need PyTorch in this script at all.
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, verbose=0)
    clf.fit(X, y)
    weights = clf.coef_.ravel().astype(np.float32)  # (1280,)
    bias = float(clf.intercept_[0])
    pred = clf.predict(X)
    acc = (pred == y).mean()
    print(f"trained: train acc {acc*100:.1f}%  weights L2={float(np.linalg.norm(weights)):.2f}  bias={bias:+.3f}")

    # Write head.bin: 1280×float32 weights then 1×float32 bias, LE.
    with open(args.output, "wb") as f:
        for w in weights:
            f.write(struct.pack("<f", float(w)))
        f.write(struct.pack("<f", bias))
    print(f"wrote {args.output} ({os.path.getsize(args.output)} B)")


if __name__ == "__main__":
    sys.exit(main())
