#!/usr/bin/env python3
"""Parallel backfill: extract speaker embeddings for all cached recordings
that don't already have a .npz in the embeddings cache.

Designed to run AFTER a bulk-redetect cycle, when sources are already
cached locally and the daemon's main detect loop is idle. Uses nice +
configurable parallelism so it can run alongside other work without
starving the box.

Usage:
    backfill-speaker-embeddings.py [--parallel N] [--nice 15]
                                   [--src-dir <path>] [--emb-dir <path>]
                                   [--limit N]
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_SRC = Path.home() / ".cache" / "tv-detect-daemon" / "source"
DEFAULT_EMB = Path.home() / ".cache" / "tv-detect-daemon" / "embeddings"
DEFAULT_PYTHON = "/Users/simon/ml/tv-classifier/.venv/bin/python"
DEFAULT_SCRIPT = Path(__file__).parent / "extract-speaker-embeddings.py"


def extract_one(args):
    src, out, python, script, niceness = args
    if out.exists():
        return (src.stem[:8], "skip", 0)
    cmd = ["nice", "-n", str(niceness),
           python, str(script), str(src), str(out)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        dt = time.time() - t0
        if r.returncode != 0:
            return (src.stem[:8], "FAIL", dt, r.stderr[-200:])
        return (src.stem[:8], "ok", dt)
    except Exception as e:
        return (src.stem[:8], "ERR", time.time() - t0, str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent worker processes (each ~500MB RAM "
                         "for ECAPA-TDNN model)")
    ap.add_argument("--nice", type=int, default=15,
                    help="niceness (0=normal, 19=idlest); 15 = polite "
                         "alongside daemon detect work")
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N files (0 = all)")
    args = ap.parse_args()

    args.emb_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.src_dir.glob("*.ts"))
    if not sources:
        print(f"no .ts files in {args.src_dir}", file=sys.stderr)
        sys.exit(1)

    pending = []
    skipped = 0
    for s in sources:
        out = args.emb_dir / f"{s.stem}.npz"
        if out.exists():
            skipped += 1
            continue
        pending.append((s, out, args.python, args.script, args.nice))
    if args.limit:
        pending = pending[: args.limit]

    print(f"sources: {len(sources)}  already-extracted: {skipped}  pending: {len(pending)}",
          file=sys.stderr)
    if not pending:
        print("nothing to do", file=sys.stderr)
        return

    print(f"running {args.parallel} workers (nice={args.nice})", file=sys.stderr)
    t0 = time.time()
    n_done = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(extract_one, p): p for p in pending}
        for fut in as_completed(futs):
            res = fut.result()
            uuid, status, dt = res[:3]
            n_done += 1
            elapsed = time.time() - t0
            if status == "ok":
                print(f"  [{n_done}/{len(pending)}] {uuid} ok ({dt:.0f}s) "
                      f"— elapsed {elapsed/60:.1f}min",
                      file=sys.stderr, flush=True)
            elif status == "skip":
                pass
            else:
                n_fail += 1
                err = res[3] if len(res) > 3 else ""
                print(f"  [{n_done}/{len(pending)}] {uuid} {status} "
                      f"({dt:.0f}s): {err}", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"\ndone: {n_done - n_fail} ok / {n_fail} failed in "
          f"{elapsed/60:.1f}min", file=sys.stderr)


if __name__ == "__main__":
    main()
