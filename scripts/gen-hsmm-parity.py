#!/usr/bin/env python3
"""Generate golden vectors for the Go HSMM parity test.

The Go decoder in internal/blocks/hsmm.go must reproduce
scripts/hmm-decode-proto.py's viterbi_hsmm EXACTLY — same blocks, not
"similar" blocks. Every published number was produced by the Python
implementation; a Go port that quietly differs ships something that was never
measured.

Cases are chosen to hit the parts where a port usually drifts:

  flat-show / flat-ad   degenerate inputs, no transition at all
  two-breaks            the ordinary case
  short-break           a break just under MinBlockS — must be dropped
  exact-min             a break exactly at MinBlockS — must be kept
  ramp                  no confident region anywhere; forces the duration
                        prior to decide alone
  noisy                 seeded pseudo-random, the case where a tie-break
                        direction or an off-by-one in the descending start
                        loop shows up as a shifted edge
  adfree-*              longer than the old 45-min show cap with no ads at
                        all — the 2026-07-27 phantom-block regression
  long-with-break       same length, but WITH a real break, so the fix cannot
                        be "emit nothing when long"
  real-*                genuine per-second probabilities from faithful dumps,
                        truncated to keep testdata small

Real cases matter because synthetic ones are smooth: a real signal has the
sub-second jitter that makes near-ties common, which is exactly where an
inverted tie-break hides.
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

# The exact parameters every measurement used.
ARGS = dict(ad_mu=4 * 60, ad_sd=0.55, show_mu=12 * 60, show_sd=0.9,
            emit_w=1.0, dur_w=60.0, min_block_s=60, max_block_s=15 * 60)
DUMPS = os.path.expanduser("~/.cache/tv-detect-daemon/emit-signals")
OUT = os.path.join(HERE, "..", "internal", "blocks", "testdata",
                   "hsmm_parity.json")


def seg(n, v):
    return [v] * n


def build_cases():
    rng = np.random.default_rng(20260727)
    cases = []

    cases.append(("flat-show", seg(1200, 0.02)))
    cases.append(("flat-ad", seg(1200, 0.97)))
    cases.append(("two-breaks",
                  seg(600, 0.03) + seg(200, 0.95) + seg(700, 0.03)
                  + seg(180, 0.96) + seg(500, 0.02)))
    # 59 s of ad: below MinBlockS=60, must not survive.
    cases.append(("short-break", seg(600, 0.03) + seg(59, 0.95) + seg(600, 0.03)))
    # Exactly 60 s: the boundary case of the >= filter.
    cases.append(("exact-min", seg(600, 0.03) + seg(60, 0.95) + seg(600, 0.03)))
    cases.append(("ramp", list(np.linspace(0.05, 0.95, 1500))))
    cases.append(("noisy", list(np.clip(
        rng.normal(0.5, 0.28, 1400), 0.001, 0.999))))
    # REGRESSION GUARD for the 2026-07-27 phantom-block defect: an ad-free
    # recording LONGER than the old 45-minute show cap. With that cap in place
    # the strictly alternating states could not cover it with show segments
    # alone and the decoder invented a 60 s ad block. Every case above is
    # shorter than 45 min, which is exactly why the parity suite did not catch
    # it — the bug needed a length no test had.
    cases.append(("adfree-55min", seg(3300, 0.02)))
    cases.append(("adfree-90min", seg(5400, 0.01)))
    # And the mirror image: a real break in a recording past the old cap must
    # still be found, so the fix cannot be "never emit anything when long".
    cases.append(("long-with-break",
                  seg(1800, 0.03) + seg(240, 0.95) + seg(1800, 0.03)))

    # Two adjacent breaks with a show gap far shorter than hsmmShowMinS=30 —
    # the decoder cannot place a show segment there and must choose.
    cases.append(("tiny-show-gap",
                  seg(400, 0.03) + seg(150, 0.95) + seg(12, 0.04)
                  + seg(150, 0.95) + seg(400, 0.03)))

    for uuid, keep in (("dvr-rtl-1782090300", 3000),
                       ("dvr-vox-1778860800", 2500),
                       ("dvr-prosieben-1778559566", 1400)):
        p = os.path.join(DUMPS, f"{uuid}.json")
        if not os.path.exists(p):
            print(f"  fehlt, uebersprungen: {uuid}")
            continue
        d = json.load(open(p))
        ps = P.to_seconds(np.array(d["nn_confs"], dtype=np.float64), d["fps"])
        cases.append((f"real-{uuid}", [float(x) for x in ps[:keep]]))
    return cases


def main():
    out = []
    for name, p in build_cases():
        arr = np.asarray(p, dtype=np.float64)
        blocks = P.viterbi_hsmm(arr, ARGS["ad_mu"], ARGS["ad_sd"],
                                ARGS["show_mu"], ARGS["show_sd"],
                                ARGS["emit_w"], ARGS["dur_w"],
                                ARGS["min_block_s"], ARGS["max_block_s"])
        out.append({"name": name,
                    "p": [float(x) for x in arr],
                    "blocks": [[float(a), float(b)] for a, b in blocks]})
        print(f"  {name:34} n={len(arr):6d}  blocks={len(blocks)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"params": ARGS, "cases": out}, f)
    print(f"\n-> {os.path.normpath(OUT)}  "
          f"({os.path.getsize(OUT) / 1024:.0f} KB, {len(out)} Faelle)")


if __name__ == "__main__":
    main()
