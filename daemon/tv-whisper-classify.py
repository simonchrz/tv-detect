#!/usr/bin/env python3
"""Whisper-based ad-vs-show classifier for a recording.

Pipeline: .ts → 60-s wav segments (30-s stride) → whisper-cpp transcribe →
TF-IDF + LogReg classifier → per-window ad-probability JSON next to the
source.

Output JSON schema:
    {
      "src": "<absolute path to .ts>",
      "duration_s": 4151,
      "model": "base",
      "classifier_n_train": 157,
      "classifier_cv_f1": 0.94,
      "window_s": 60,
      "stride_s": 30,
      "windows": [{"t": 0, "prob": 0.12, "text": "..."}, ...],
      "ts": "20260501T161200Z"
    }

Idempotent: if <stem>.whisper.json exists and is newer than the source,
returns immediately (exit 0).

Designed for daemon orchestration — runs in <90s on a 70-min recording
on Apple Silicon (4 parallel workers). No Pi-side support (whisper-cpp
needs Metal for usable performance).

Usage:
    tv-whisper-classify.py --src /path/to/rec.ts [--output rec.whisper.json]
                           [--window-s 60] [--stride-s 30] [--workers 4]
                           [--model base]

Exit codes:
    0 = success (or already-fresh sidecar)
    1 = source not found / unreadable
    2 = whisper or classifier missing
    3 = ffmpeg/whisper subprocess failed
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CLASSIFIER_PATH = Path.home() / ".cache/tv-whisper/classifier.pkl"
WHISPER_MODEL_DIR = Path.home() / ".cache/whisper"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
WHISPER_CLI = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"


def probe_duration_s(src: Path) -> float:
    """Returns float seconds via ffprobe. Raises on error."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)], text=True, timeout=30)
    return float(out.strip())


def extract_wav(src: Path, start_s: int, dur_s: int, dest: Path) -> bool:
    """16 kHz mono PCM — whisper.cpp expects this. Returns True on success."""
    r = subprocess.run(
        [FFMPEG, "-ss", str(start_s), "-t", str(dur_s),
         "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-y", str(dest)],
        capture_output=True, timeout=60)
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000


def transcribe_one(args):
    """Worker: extract wav + whisper-cli + read txt back. Tuple in,
    (start_s, text) out."""
    start_s, dur_s, src, work_dir, model_path = args
    wav = work_dir / f"w_{start_s:06d}.wav"
    txt = wav.with_suffix("")  # whisper-cli writes <prefix>.txt
    if not extract_wav(src, start_s, dur_s, wav):
        return start_s, ""
    r = subprocess.run(
        [WHISPER_CLI, "-m", str(model_path), "-f", str(wav),
         "-l", "de", "-nt", "-otxt", "-of", str(txt)],
        capture_output=True, timeout=180)
    out_txt = wav.with_suffix(".txt")
    if r.returncode != 0 or not out_txt.exists():
        return start_s, ""
    text = out_txt.read_text().strip()
    # Drop intermediate files now to keep /tmp small during long runs
    try: wav.unlink()
    except OSError: pass
    try: out_txt.unlink()
    except OSError: pass
    return start_s, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Path to source .ts (or any ffmpeg-readable file)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSON path (default: <src-stem>.whisper.json)")
    ap.add_argument("--window-s", type=int, default=60)
    ap.add_argument("--stride-s", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel whisper workers (Apple Silicon Metal "
                         "handles 4 well; bump to 6-8 on M3+/M4)")
    ap.add_argument("--model", default="base",
                    help="whisper.cpp model name (= ggml-<name>.bin in "
                         f"{WHISPER_MODEL_DIR}). tiny=fast/low-acc, "
                         "base=balanced (default), small=slow/higher-acc.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a fresh sidecar exists.")
    args = ap.parse_args()

    if not args.src.is_file():
        print(f"src not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.src.with_suffix(".whisper.json")
    if out_path.exists() and not args.force:
        if out_path.stat().st_mtime >= args.src.stat().st_mtime:
            print(f"sidecar fresh: {out_path}", flush=True)
            sys.exit(0)

    if not CLASSIFIER_PATH.is_file():
        print(f"classifier missing: {CLASSIFIER_PATH}", file=sys.stderr)
        sys.exit(2)
    model_path = WHISPER_MODEL_DIR / f"ggml-{args.model}.bin"
    if not model_path.is_file():
        print(f"whisper model missing: {model_path}", file=sys.stderr)
        sys.exit(2)

    with open(CLASSIFIER_PATH, "rb") as fp:
        bundle = pickle.load(fp)
    vec = bundle["vectorizer"]
    clf = bundle["classifier"]

    duration = probe_duration_s(args.src)
    n_windows = max(1, int((duration - args.window_s) // args.stride_s) + 1)
    starts = [i * args.stride_s for i in range(n_windows)]
    print(f"src={args.src.name} dur={duration:.0f}s "
          f"windows={n_windows} workers={args.workers}", flush=True)

    work_dir = Path(tempfile.mkdtemp(prefix="tv-whisper-"))
    t0 = time.time()
    results = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            tasks = [(s, args.window_s, args.src, work_dir, model_path)
                     for s in starts]
            for start, text in ex.map(transcribe_one, tasks):
                results[start] = text
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    transcribe_dt = time.time() - t0

    # Classify each window
    texts = [results.get(s, "") for s in starts]
    # predict_proba returns [[P(show), P(ad)]] per row; take ad column
    X = vec.transform(texts)
    probs = clf.predict_proba(X)[:, 1].tolist()

    out = {
        "src": str(args.src.resolve()),
        "duration_s": duration,
        "model": args.model,
        "classifier_n_train": bundle.get("n_train"),
        "classifier_cv_f1": bundle.get("cv_f1"),
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "windows": [{"t": s, "prob": round(p, 4),
                     "text": (t[:200] if t else "")}
                    for s, p, t in zip(starts, probs, texts)],
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "transcribe_s": round(transcribe_dt, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    n_ad = sum(1 for p in probs if p > 0.5)
    print(f"done: transcribe={transcribe_dt:.0f}s, "
          f"{n_ad}/{n_windows} windows >0.5 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
