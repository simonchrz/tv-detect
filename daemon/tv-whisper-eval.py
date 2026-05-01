#!/usr/bin/env python3
"""Offline evaluator for the Whisper post-processor.

Computes Block-IoU before vs after applying post-processing rules to
the raw detect output, against ads_user.json ground truth. Drives the
go/no-go decision for daemon-side post-processor integration.

Inputs per recording:
    ads_user.json  — ground truth (user-confirmed)
    ads.json       — raw detect output (= "before")
    whisper.json   — sliding-window ad probabilities

Output: per-recording IoU before/after + corpus-level mean delta.

Rules under test:
    R1 (FP-Killer): if a detected block has ≥50% of overlapping
       whisper-windows with prob<0.30, remove that block.
    R2 (Missed-Adder): if ≥3 contiguous whisper-windows (≥120s span)
       all have prob>0.60, AND no detected block overlaps the cluster,
       AND cluster duration is in [60, 600]s → add as new block.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def load_ads(path: Path) -> list[list[float]]:
    if not path.is_file(): return []
    try:
        d = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(d, list):  # legacy list format
        return [a for a in d if isinstance(a, list) and len(a) >= 2]
    if isinstance(d, dict):
        return [a for a in d.get("ads", []) if isinstance(a, list) and len(a) >= 2]
    return []


def load_whisper(path: Path) -> tuple[list[dict], int, int]:
    """Returns (windows, window_s, stride_s)."""
    d = json.loads(path.read_text())
    return d["windows"], d["window_s"], d["stride_s"]


def windows_overlapping(start: float, end: float, windows: list[dict],
                          window_s: int) -> list[dict]:
    """Whisper windows that overlap the [start, end] interval."""
    out = []
    for w in windows:
        ws, we = w["t"], w["t"] + window_s
        if ws < end and we > start:
            out.append(w)
    return out


def fp_killer(ads: list[list[float]], windows: list[dict],
              window_s: int, mean_thr: float = 0.45) -> tuple[list[list[float]], list]:
    """Remove blocks whose mean whisper-prob over overlapping windows
    is below `mean_thr`. Mean-based instead of majority-thresholded
    because empirical n=9 showed even "mostly show" blocks rarely had
    >50% windows below 0.3 — Whisper sits in 0.3-0.5 ambiguous range.
    Mean<0.45 is the empirically-tuned cut that fires on the Galileo
    drone-segment FP without touching any true ads."""
    kept, removed = [], []
    for block in ads:
        s, e = block[0], block[1]
        ovlp = windows_overlapping(s, e, windows, window_s)
        if not ovlp:
            kept.append(block); continue
        mean_p = sum(w["prob"] for w in ovlp) / len(ovlp)
        if mean_p < mean_thr:
            removed.append((list(block), round(mean_p, 3)))
        else:
            kept.append(block)
    return kept, removed


def boundary_extend(ads: list[list[float]], windows: list[dict],
                    window_s: int, stride_s: int,
                    prob_thr: float = 0.55,
                    max_extend_s: int = 120) -> tuple[list[list[float]], list]:
    """For each block, walk forward through whisper windows starting at
    block-end. While next contiguous window has prob > `prob_thr`,
    extend block end (capped at `max_extend_s` total extension).
    Catches truncated ad-block ends (= the dominant low-IoU pattern in
    the n=9 corpus). Returns (extended_blocks, [(orig_end, new_end)…])."""
    by_t = {w["t"]: w["prob"] for w in windows}
    starts_sorted = sorted(by_t.keys())
    out, changes = [], []
    for block in ads:
        s, e = block[0], block[1]
        candidates = [t for t in starts_sorted if e <= t < e + max_extend_s]
        new_e = e
        for t in candidates:
            if by_t[t] > prob_thr:
                new_e = min(t + window_s, e + max_extend_s)
            else:
                break  # don't jump gaps; first non-ad window stops the walk
        if new_e != e:
            changes.append((round(e, 1), round(new_e, 1)))
        out.append([s, new_e])
    return out, changes


def missed_adder(ads: list[list[float]], windows: list[dict],
                  window_s: int, stride_s: int,
                  prob_thr: float = 0.60, min_run: int = 3,
                  min_span_s: int = 120, min_block_s: int = 60,
                  max_block_s: int = 600) -> tuple[list[list[float]], list]:
    """Add new blocks for whisper-clusters not covered by any existing
    block. Returns (added_blocks, full_ads_list)."""
    added = []
    # Find runs of consecutive high-prob windows
    run_start = None
    for i, w in enumerate(windows):
        if w["prob"] > prob_thr:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                w0, w1 = windows[run_start], windows[i - 1]
                cluster_s = w0["t"]
                cluster_e = w1["t"] + window_s
                # Check if existing block overlaps
                covered = any(b[0] < cluster_e and b[1] > cluster_s
                              for b in ads)
                span = cluster_e - cluster_s
                if (not covered and span >= min_span_s
                        and min_block_s <= span <= max_block_s):
                    added.append([float(cluster_s), float(cluster_e)])
            run_start = None
    # tail
    if run_start is not None and len(windows) - run_start >= min_run:
        w0, w1 = windows[run_start], windows[-1]
        cluster_s = w0["t"]; cluster_e = w1["t"] + window_s
        covered = any(b[0] < cluster_e and b[1] > cluster_s for b in ads)
        span = cluster_e - cluster_s
        if (not covered and span >= min_span_s
                and min_block_s <= span <= max_block_s):
            added.append([float(cluster_s), float(cluster_e)])
    return added, sorted(ads + added)


def block_iou(p, g):
    inter = max(0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union else 0


def mean_iou(pred: list, gt: list) -> float:
    """Per-GT-block mean IoU (recall-only; FPs don't reduce score)."""
    if not gt: return 1.0 if not pred else 0.0
    return sum(max((block_iou(p, g) for p in pred), default=0)
               for g in gt) / len(gt)


def _to_seconds(blocks):
    s = set()
    for b in blocks:
        for i in range(int(b[0]), int(b[1]) + 1):
            s.add(i)
    return s


def sym_iou(pred: list, gt: list) -> float:
    """Symmetric IoU = total intersection seconds / total union seconds.
    UNLIKE mean_iou, this DOES penalise false-positive blocks. Required
    to gauge the value of FP-killer rules. Approximates the user-perceived
    "how much wrong ad time" metric."""
    p = _to_seconds(pred); g = _to_seconds(gt)
    if not p and not g: return 1.0
    inter = len(p & g); union = len(p | g)
    return inter / union if union else 0


def apply_postprocess(raw: list, windows: list, window_s: int,
                       stride_s: int) -> tuple[list, dict]:
    """Apply the full post-processor pipeline (FP-killer → boundary-extend
    → missed-adder) to raw detection output. Returns (refined_blocks,
    diagnostic_dict)."""
    after_fp, removed = fp_killer(raw, windows, window_s)
    after_ext, extended = boundary_extend(after_fp, windows, window_s, stride_s)
    added, after_full = missed_adder(after_ext, windows, window_s, stride_s)
    return after_full, {
        "removed": removed, "extended": extended, "added": added,
    }


def evaluate_one(rec_dir: Path, whisper_dir: Path) -> dict | None:
    uuid = rec_dir.name[5:]
    user_p = rec_dir / "ads_user.json"
    auto_p = rec_dir / "ads.json"
    whisper_p = whisper_dir / f"{uuid}.whisper.json"
    if not (user_p.is_file() and auto_p.is_file() and whisper_p.is_file()):
        return None
    gt = load_ads(user_p)
    raw = load_ads(auto_p)
    if not gt: return None
    windows, win_s, stride_s = load_whisper(whisper_p)
    refined, diag = apply_postprocess(raw, windows, win_s, stride_s)
    return {
        "uuid": uuid[:8],
        "n_gt": len(gt), "n_raw": len(raw), "n_refined": len(refined),
        "iou_raw": sym_iou(raw, gt),
        "iou_refined": sym_iou(refined, gt),
        "delta": sym_iou(refined, gt) - sym_iou(raw, gt),
        **diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hls-root", type=Path,
                    default=Path("/Users/simon/mnt/pi-tv/hls"))
    ap.add_argument("--whisper-dir", type=Path,
                    default=Path.home() / ".cache/tv-whisper")
    ap.add_argument("--uuids-file", type=Path,
                    help="optional: limit to UUIDs in this file")
    args = ap.parse_args()

    if args.uuids_file:
        wanted = set(args.uuids_file.read_text().split())
        rec_dirs = [args.hls_root / f"_rec_{u}" for u in wanted]
    else:
        rec_dirs = list(args.hls_root.glob("_rec_*"))

    results = []
    for d in sorted(rec_dirs):
        r = evaluate_one(d, args.whisper_dir)
        if r: results.append(r)

    if not results:
        print("no recordings with all three sidecars (ads_user + ads + whisper)")
        sys.exit(1)

    print(f"{'uuid':10} {'gt':3} {'raw':4} {'ref':4} {'iou_raw':8} "
          f"{'iou_ref':8} {'delta':6}  changes")
    for r in results:
        ch = []
        if r["removed"]: ch.append(f"−{len(r['removed'])}fp")
        if r["extended"]: ch.append(f"⤇{len(r['extended'])}ext")
        if r["added"]: ch.append(f"+{len(r['added'])}new")
        ch_str = ",".join(ch) if ch else "—"
        print(f"{r['uuid']:10} {r['n_gt']:3} {r['n_raw']:4} {r['n_refined']:4} "
              f"{r['iou_raw']:.3f}    {r['iou_refined']:.3f}    "
              f"{r['delta']:+.3f}   {ch_str}")

    n = len(results)
    mean_raw = sum(r["iou_raw"] for r in results) / n
    mean_ref = sum(r["iou_refined"] for r in results) / n
    n_helped = sum(1 for r in results if r["delta"] > 0.001)
    n_hurt = sum(1 for r in results if r["delta"] < -0.001)
    print(f"\nCorpus (n={n}):  symmetric IoU {mean_raw:.3f} → {mean_ref:.3f}  "
          f"Δ={mean_ref-mean_raw:+.3f}")
    print(f"  helped: {n_helped}, hurt: {n_hurt}, unchanged: {n-n_helped-n_hurt}")
    if mean_ref > mean_raw + 0.02:
        print(f"\n→ POSITIVE: post-processor gains ≥0.02 mean IoU; integrate")
    elif mean_ref > mean_raw - 0.02:
        print(f"\n→ NEUTRAL: post-processor within noise")
    else:
        print(f"\n→ NEGATIVE: post-processor LOSES IoU; do NOT integrate")


if __name__ == "__main__":
    main()
