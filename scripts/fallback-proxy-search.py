#!/usr/bin/env python3
"""Is there a LABEL-FREE signal that says "block forming failed here"?

Background. Structured decoding (HSMM over the same per-frame probabilities
blocks.Form consumes) is not uniformly better or worse — it wins where
production is weak and loses where production is already right. Measured over
23 faithful dumps:

    PROD < 0.90   n=16   HSMM  +0.090   12/16 wins
    PROD >= 0.90  n= 7   HSMM  -0.117    1/7

That split is real but useless as a switch: it is keyed on the production IoU,
which needs the ground truth. Knowing it means already having the answer.

The per-show switch is dead too — same show, opposite directions:

    Galileo        0.762 -> 0.797   +0.035
    Galileo 360    0.976 -> 0.769   -0.207

So the only way structured decoding ships is a proxy computable at DETECT time,
from the signals alone, that separates the recordings where production fails
from the ones where it succeeds. This script measures candidate proxies against
the known deltas and reports whether any of them beats "always production".

The bar is deliberately high and stated up front, before looking:

  1. The gated policy must beat BOTH always-prod and always-hsmm on mean IoU.
  2. It must do so with a threshold that is not tuned to the last decimal —
     the win has to survive moving the cut around (reported as a sweep).
  3. n=23 is small. A proxy that only works at one exact threshold is noise.

If nothing clears that, the honest output is "no proxy", and the Go port of
Viterbi does not get written.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "hmmproto", os.path.join(HERE, "hmm-decode-proto.py"))
P = importlib.util.module_from_spec(spec)
sys.modules["hmmproto"] = P
spec.loader.exec_module(P)

CACHE = os.path.expanduser("~/.cache/tvd-fallback-proxy.json")


def blocks_dur(bs):
    return sum(b - a for a, b in bs)


def iou_pair(a, b):
    """Block-IoU between two block lists (same metric as vs ground truth)."""
    return P.block_iou(a, b)


def collect():
    """Per recording: production blocks, hsmm blocks, ground truth, signals."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    files = []
    seen = set()
    for dd in P.DUMP_DIRS:
        import glob
        for f in sorted(glob.glob(os.path.join(dd, "dvr-*.json"))):
            u = os.path.basename(f)[:-5]
            if u not in seen:
                seen.add(u)
                files.append(f)
    out = []
    for f in files:
        u = os.path.basename(f)[:-5]
        gt = P.gt_of(u)
        if not gt:
            continue
        d = json.load(open(f))
        nn = np.array(d["nn_confs"], dtype=np.float64)
        fps = d["fps"]
        ps = P.to_seconds(nn, fps)
        # Key on the dump's identity, not just the uuid. The same recording can
        # have dumps from different heads (/tmp/faithful-emit vs the daemon's
        # dir), and a uuid-only key silently serves the stale one — which is
        # exactly the head-mixing the harness docstring forbids.
        st = os.stat(f)
        key = f"{u}:{st.st_size}:{int(st.st_mtime)}"
        if key in cache:
            rec = cache[key]
        else:
            try:
                cfg = P.detect_cfg(u)
            except Exception as e:
                print(f"{u:29} uebersprungen — keine detect-config ({e})",
                      flush=True)
                continue
            prod = P.production_replay(f, cfg)
            hsmm = P.viterbi_hsmm(ps, 4 * 60, 0.55, 12 * 60, 0.9,
                                  1.0, 60.0, 60, 15 * 60)
            rec = {"prod": prod, "hsmm": hsmm,
                   "title": (cfg.get("show_title") or "?")[:26]}
            cache[key] = rec
            json.dump(cache, open(CACHE, "w"))
            print(f"  gerechnet: {u}", flush=True)
        prod = [tuple(b) for b in rec["prod"]]
        hsmm = [tuple(b) for b in rec["hsmm"]]
        out.append({
            "uuid": u, "title": rec["title"],
            "prod": prod, "hsmm": hsmm, "gt": gt, "ps": ps,
            "dur": len(ps),
            "iou_prod": P.block_iou(prod, gt),
            "iou_hsmm": P.block_iou(hsmm, gt),
        })
    return out


def proxies(r):
    """Every value here must be computable at detect time — signals + blocks
    only, never ground truth."""
    ps, prod, hsmm, dur = r["ps"], r["prod"], r["hsmm"], r["dur"]
    inside = np.zeros(dur, dtype=bool)
    for a, b in prod:
        inside[int(max(0, a)):int(min(dur, b))] = True
    band = ((ps > 0.3) & (ps < 0.7)).mean()
    durs = [b - a for a, b in prod] or [0.0]
    return {
        # How far the two decoders disagree on the same input.
        "disagree": 1.0 - iou_pair(prod, hsmm),
        # Fraction of the recording the NN is unsure about.
        "band": float(band),
        # Mean NN probability inside what production called an ad. If block
        # forming overshot, the tail of the block sits in low-probability
        # territory and drags this down.
        "p_in": float(ps[inside].mean()) if inside.any() else 0.0,
        # Ad share. German commercial TV runs ~15-25% per hour; far off means
        # something went wrong, in either direction.
        "adfrac": blocks_dur(prod) / max(1.0, dur),
        # Longest block. Real ad breaks cap out around 5-6 minutes.
        "maxblk": float(max(durs)),
        "nblk": float(len(prod)),
    }


def evaluate(rows, key, invert=False):
    """Sweep a threshold over one proxy; report the best gated policy and how
    wide the winning region is. A single lucky cut is not a result."""
    vals = sorted({round(r["px"][key], 4) for r in rows})
    always_prod = np.mean([r["iou_prod"] for r in rows])
    always_hsmm = np.mean([r["iou_hsmm"] for r in rows])
    best = None
    wins = []
    for t in vals:
        gated = []
        for r in rows:
            on = (r["px"][key] < t) if invert else (r["px"][key] >= t)
            gated.append(r["iou_hsmm"] if on else r["iou_prod"])
        m = float(np.mean(gated))
        n_on = sum(1 for r in rows
                   if ((r["px"][key] < t) if invert else (r["px"][key] >= t)))
        if m > max(always_prod, always_hsmm) + 1e-9:
            wins.append(t)
        if best is None or m > best[1]:
            best = (t, m, n_on)
    return always_prod, always_hsmm, best, len(wins), len(vals)


def main():
    rows = collect()
    for r in rows:
        r["px"] = proxies(r)
    rows.sort(key=lambda r: r["iou_prod"])

    print(f"\nn={len(rows)}   "
          f"immer PROD {np.mean([r['iou_prod'] for r in rows]):.3f}   "
          f"immer HSMM {np.mean([r['iou_hsmm'] for r in rows]):.3f}\n")

    keys = ["disagree", "band", "p_in", "adfrac", "maxblk", "nblk"]
    hdr = "  ".join(f"{k:>8}" for k in keys)
    print(f"{'PROD':>6} {'HSMM':>6} {'delta':>7}  {hdr}  Sendung")
    for r in rows:
        d = r["iou_hsmm"] - r["iou_prod"]
        vals = "  ".join(f"{r['px'][k]:8.3f}" for k in keys)
        print(f"{r['iou_prod']:6.3f} {r['iou_hsmm']:6.3f} {d:+7.3f}  "
              f"{vals}  {r['title']}")

    print("\n=== Korrelation Proxy <-> delta ===")
    d = np.array([r["iou_hsmm"] - r["iou_prod"] for r in rows])
    for k in keys:
        v = np.array([r["px"][k] for r in rows])
        c = float(np.corrcoef(v, d)[0, 1]) if v.std() > 0 else float("nan")
        print(f"  {k:10s} r={c:+.3f}")

    print("\n=== Gated policy (HSMM nur wenn Proxy die Schwelle reisst) ===")
    print("   Ein Gewinn zaehlt nur, wenn er BEIDE Baselines schlaegt und die")
    print("   Gewinn-Region breit ist. 'breite 1/23' ist Rauschen.\n")
    for k in keys:
        for inv in (False, True):
            ap, ah, best, nwin, ntot = evaluate(rows, k, inv)
            t, m, n_on = best
            arrow = "<" if inv else ">="
            mark = "  <-- schlaegt beide" if m > max(ap, ah) + 1e-9 else ""
            print(f"  {k:10s} {arrow} {t:8.3f}  ->  {m:.3f} "
                  f"(HSMM an bei {n_on}/{len(rows)}, Gewinn-Region "
                  f"{nwin}/{ntot}){mark}")


if __name__ == "__main__":
    main()


def loocv(rows, key, invert):
    """Leave-one-out: pick the threshold on the other n-1, apply to the held-out
    one. Sweeping a threshold on n=23 and reporting the best value is not a
    result — with ~23 candidate cuts per proxy and 6 proxies, SOME cut beats
    both baselines by chance. This is the only version of the number that
    means anything."""
    got = []
    for i in range(len(rows)):
        tr = rows[:i] + rows[i + 1:]
        _, _, best, _, _ = evaluate(tr, key, invert)
        t = best[0]
        r = rows[i]
        on = (r["px"][key] < t) if invert else (r["px"][key] >= t)
        got.append(r["iou_hsmm"] if on else r["iou_prod"])
    return float(np.mean(got))
