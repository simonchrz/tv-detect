#!/usr/bin/env python3
"""HMM/Viterbi structured decoding — research prototype + measurement harness.

Question: does replacing blocks.Form's hand-tuned state machine (hysteresis +
rolling-mean smoothing + snap windows) with structured decoding over the same
per-frame probabilities improve block IoU?

Run against FAITHFULLY emitted signal dumps only (full production command:
logo-cnn + bumper templates + whisper + start-ts). Dumps emitted without them
measure a pipeline that does not exist — see the 2026-07-24 boundary-head
incident where non-faithful dumps read +0.016 and faithful read -0.015.

FINDINGS (2026-07-24, user-reviewed GT, faithful dumps, learned per-channel
drift applied to the production baseline — omitting it understates production):

  Channel     n   PRODUCTION   HMM(pure NN)   delta
  prosieben   8       0.822        0.912      +0.091
  rtl         3       0.750        0.901      +0.151
  vox         3       0.840        0.833      -0.006

  KEY: feed the HMM the RAW NN probability, not blocks.Form's gated blend.
  rtl's nn_gate=0.3 routes 10.7% of frames to a logo fallback, and its logo is
  weak (AUC 0.827 vs NN 0.973) — the blend degrades a good signal. Switching
  the emission from blended score to pure NN takes rtl from 0.831 to 0.901.
  prosieben (gate=0) is unaffected because nothing falls back.

  vox is neutral and NOT explained by: gate (1.1% fallback), NN quality (AUC
  0.993, same as prosieben), block structure (2.0 blocks/rec, same lengths),
  or min_block (0.833 flat over 30-120s). Its HMM blocks sit correctly; only
  edges differ by 20-90s in both directions. vox is simply the channel where
  the existing pipeline already performs at HMM level.

  A first hypothesis — "HMM wins wherever the emission is the calibrated head
  probability (nn_weight=1.0)" — was DISPROVEN by vox (also nn_weight=1.0, no
  win). Do not resurrect it. The gate/blend explanation above is what actually
  separates rtl, and it is measured, not assumed.

  SIDE FINDING, suspicion only (n=11): the learned per-channel drift makes
  things WORSE on user-labelled recordings (prosieben 0.856 -> 0.822, rtl
  0.840 -> 0.750 when --start-extend/--end-extend are applied). Counter-
  intuitive since the drift is learned FROM user corrections. Either stale,
  overshooting, or encoding a preference the GT does not carry. Needs its own
  investigation — potentially a bigger lever than anything else here because it
  affects every channel that has drift.

  Ablation on prosieben, all on the SAME input:
    state machine raw ......... 0.660   (precision 0.975 / recall 0.659:
                                         systematically cuts blocks short)
    naive threshold @0.5 ...... 0.663   (= the state machine's hysteresis
                                         buys nothing over thresholding)
    + 30s gap merge ........... 0.789
    + 10s smoothing ........... 0.881   (smoothing is the biggest single win)
    HMM Viterbi ............... 0.916-0.927

  Two more hypotheses the prototype disproved:
  1. Duration priors do NOT matter (constant 0.916 across mean-ad 2-8min x
     mean-show 8-25min). The gain is the transition penalty suppressing state
     flicker, not learned block lengths.
  2. The deterministic snaps are COMPENSATION for weak block forming, not
     independent value: on HMM edges with production windows they cost -0.158.
     With tight windows they are merely neutral (scene +-2s: 0.928 vs 0.927).
"""

import json, glob, os, subprocess, sys
import numpy as np

BIN = "/Users/simon/src/tv-detect/build/tv-detect"
ARCH = os.path.expanduser("~/.cache/tvd-train-archive")


def block_iou(pred, gt):
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    tot = 0.0
    for g in gt:
        best = 0.0
        for p in pred:
            inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
            uni = max(p[1], g[1]) - min(p[0], g[0])
            if uni > 0:
                best = max(best, inter / uni)
        tot += best
    return tot / len(gt)


def gt_of(u):
    z = np.load(f"{ARCH}/{u}.npz", allow_pickle=True)
    return [tuple(b) for b in json.loads(str(z["meta"])).get("ads", [])]


def to_seconds(nn, fps):
    """Per-second mean of the 25fps NN stream (= the rate the head trained at)."""
    n = int(len(nn) // fps)
    return np.array([nn[int(i * fps):int((i + 1) * fps)].mean() for i in range(n)])


def runs_to_blocks(states, min_len):
    out, i, n = [], 0, len(states)
    while i < n:
        if states[i] == 1:
            j = i
            while j + 1 < n and states[j + 1] == 1:
                j += 1
            if (j + 1 - i) >= min_len:
                out.append((float(i), float(j + 1)))
            i = j + 1
        else:
            i += 1
    return out


def viterbi_hmm(p, mean_show_s, mean_ad_s, emit_w, min_block_s):
    """Plain 2-state HMM, geometric durations from expected run lengths."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    e = np.stack([np.log(1 - p), np.log(p)]) * emit_w        # (2, T)
    a_ss = np.log(1 - 1 / mean_show_s); a_sa = np.log(1 / mean_show_s)
    a_aa = np.log(1 - 1 / mean_ad_s);   a_as = np.log(1 / mean_ad_s)
    T = p.shape[0]
    dp = np.full((2, T), -np.inf); bp = np.zeros((2, T), dtype=np.int8)
    dp[0, 0], dp[1, 0] = e[0, 0], e[1, 0] + np.log(0.25)
    for t in range(1, T):
        c0 = (dp[0, t - 1] + a_ss, dp[1, t - 1] + a_as)
        c1 = (dp[0, t - 1] + a_sa, dp[1, t - 1] + a_aa)
        bp[0, t] = int(c0[1] > c0[0]); dp[0, t] = max(c0) + e[0, t]
        bp[1, t] = int(c1[1] > c1[0]); dp[1, t] = max(c1) + e[1, t]
    s = int(dp[1, T - 1] > dp[0, T - 1])
    states = np.zeros(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = bp[s, t]
    return runs_to_blocks(states, min_block_s)


def viterbi_hsmm(p, ad_mu, ad_sd, show_mu, show_sd, emit_w, dur_w,
                 min_block_s, max_block_s):
    """Explicit-duration HMM: a segment's score = summed emissions + a
    log-normal-ish duration prior, so block LENGTH is learned structure
    instead of the hand-set MinBlockS/MaxBlockS constants."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    T = len(p)
    la, ls = np.log(p) * emit_w, np.log(1 - p) * emit_w
    ca = np.concatenate([[0.0], np.cumsum(la)])
    cs = np.concatenate([[0.0], np.cumsum(ls)])

    def dur_lp(d, mu, sd):
        return -0.5 * ((np.log(d) - np.log(mu)) / sd) ** 2 * dur_w

    NEG = -1e18
    # dp[t][k]: best score for prefix of length t ending in state k (0 show, 1 ad)
    dp = np.full((T + 1, 2), NEG); dp[0, :] = 0.0
    bk = np.zeros((T + 1, 2), dtype=np.int64)
    dmin_a, dmax_a = int(min_block_s), int(max_block_s)
    dmin_s, dmax_s = 30, int(45 * 60)
    for t in range(1, T + 1):
        for k, (dmin, dmax, cum, mu, sd) in enumerate((
                (dmin_s, dmax_s, cs, show_mu, show_sd),
                (dmin_a, dmax_a, ca, ad_mu, ad_sd))):
            prev = 1 - k
            best, bd = NEG, 0
            lo = max(0, t - dmax)
            for st in range(t - dmin, lo - 1, -1):
                if st < 0:
                    break
                base = 0.0 if st == 0 else dp[st, prev]
                if base <= NEG / 2:
                    continue
                d = t - st
                sc = base + (cum[t] - cum[st]) + dur_lp(d, mu, sd)
                if sc > best:
                    best, bd = sc, st
            dp[t, k], bk[t, k] = best, bd
    k = int(dp[T, 1] > dp[T, 0])
    if dp[T, k] <= NEG / 2:
        return []
    segs, t = [], T
    while t > 0:
        st = bk[t, k]
        segs.append((st, t, k))
        t, k = st, 1 - k
    return [(float(a), float(b)) for a, b, kk in reversed(segs) if kk == 1
            and (b - a) >= min_block_s]


def statemachine_raw(dump_path):
    """Current pipeline's block formation with every deterministic snap OFF —
    isolates state machine + smoothing, the part the HMM replaces."""
    cmd = [BIN, "--quiet", "--replay-signals", dump_path, "--output", "summary",
           "--nn-gate", "0", "--nn-weight", "1.0", "--logo-smooth", "5",
           "--nn-smooth", "10", "--min-block-sec", "60",
           "--bumper-snap", "0", "--iframe-snap", "0", "--scene-cut-snap", "0",
           "--letterbox-snap", "0", "--logo-cross-refine", "0",
           "--boundary-snap", "0", "dummy"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    return [(float(b[0]), float(b[1])) for b in json.loads(r.stdout).get("blocks", [])]


def main():
    files = sorted(glob.glob("/tmp/faithful-emit/dvr-*.json"))
    rows = []
    for f in files:
        u = os.path.basename(f)[:-5]
        gt = gt_of(u)
        if not gt:
            continue
        d = json.load(open(f))
        nn = np.array(d["nn_confs"], dtype=np.float64)
        fps = d["fps"]
        ps = to_seconds(nn, fps)
        base = block_iou(statemachine_raw(f), gt)
        hmm = block_iou(viterbi_hmm(ps, 12 * 60, 4 * 60, 1.0, 60), gt)
        hsmm = block_iou(viterbi_hsmm(ps, 4 * 60, 0.55, 12 * 60, 0.9,
                                      1.0, 60.0, 60, 15 * 60), gt)
        rows.append((u, base, hmm, hsmm))
        print(f"{u:30} statemachine {base:.3f} | HMM {hmm:.3f} ({hmm-base:+.3f}) "
              f"| HSMM {hsmm:.3f} ({hsmm-base:+.3f})", flush=True)
    if rows:
        b = np.mean([r[1] for r in rows]); h = np.mean([r[2] for r in rows])
        s = np.mean([r[3] for r in rows])
        print(f"\nn={len(rows)}  statemachine {b:.3f} | HMM {h:.3f} ({h-b:+.3f}) "
              f"| HSMM {s:.3f} ({s-b:+.3f})")


if __name__ == "__main__":
    main()
