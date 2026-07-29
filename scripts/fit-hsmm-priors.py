#!/usr/bin/env python3
"""Fit per-channel HSMM duration priors from the corpus' human-touched labels.

Stage 4 of the replace-the-state-machine plan (2026-07-28). The HSMM currently
uses one global log-normal prior for everything: ad blocks median 4 min
(sd 0.55), show segments median 12 min (sd 0.9). Real channels differ — sat-1's
long midday breaks are nothing like nick's short kids-slot interruptions — and
every published HSMM number was measured with the global constants.

Provenance rules:
  * which in (user, merged) only. Auto labels are the decoder's own output;
    fitting priors on them would teach the HSMM to imitate the state machine,
    including its faults.
  * show-segment lengths are measured BETWEEN consecutive ad blocks only —
    never from recording start to first block or last block to recording end,
    because those are truncated by the recording window, not by the
    broadcaster's schedule (a classic censoring bias: including them drags the
    show median down hard on short recordings).
  * channels with fewer than MIN_SAMPLES intervals keep the global prior
    (a median over 4 values is noise — see the sat-1 budget artefact in the
    surfacing analysis, same failure shape).

RE-RUN THIS AFTER THE CORPUS CONVERGES to the trailer-start convention
(2026-07-28 bulk redetect + nightly rebuild): ad-block lengths shrink by the
trailer duration (~60-90 s), so priors fitted before convergence are biased
long. The output records which labels it saw so the two runs can be compared.

Output: hsmm-priors.json next to the train archive — consumed by nothing yet;
wiring it into the decoder is part of the stage-1 re-measurement, not before.
"""
import json
import math
import os
import sys
import glob

import numpy as np

ARCH = os.path.expanduser("~/.cache/tvd-train-archive")
OUT = os.path.join(ARCH, "hsmm-priors.json")
MIN_SAMPLES = 12


def lognorm_fit(vals):
    """Median + log-space sd — matches viterbi_hsmm's dur_lp shape."""
    logs = np.log(np.asarray(vals, dtype=np.float64))
    return float(np.exp(np.median(logs))), float(logs.std(ddof=1))


def main():
    per_ch_ads, per_ch_shows = {}, {}
    n_recs = 0
    for f in sorted(glob.glob(f"{ARCH}/dvr-*.npz")):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        if m.get("which") not in ("user", "merged"):
            continue
        ads = sorted((float(a), float(b)) for a, b in (m.get("ads") or []))
        if not ads:
            continue
        slug = m.get("slug") or "?"
        n_recs += 1
        for a, b in ads:
            if b - a >= 30:
                per_ch_ads.setdefault(slug, []).append(b - a)
        # Interior show segments only (censoring — see module docstring).
        for (_, b1), (a2, _) in zip(ads, ads[1:]):
            if a2 - b1 >= 60:
                per_ch_shows.setdefault(slug, []).append(a2 - b1)

    out = {"fitted_from": {"recordings": n_recs, "which": ["user", "merged"]},
           "global": {"ad_mu_s": 240, "ad_sd": 0.55,
                      "show_mu_s": 720, "show_sd": 0.9},
           "channels": {}}
    print(f"{n_recs} Aufnahmen, {len(per_ch_ads)} Kanäle")
    print(f"{'Kanal':16} {'n_ad':>5} {'ad_mu':>7} {'ad_sd':>6} "
          f"{'n_show':>6} {'show_mu':>8} {'show_sd':>8}")
    for ch in sorted(per_ch_ads):
        ads, shows = per_ch_ads[ch], per_ch_shows.get(ch, [])
        row = {}
        if len(ads) >= MIN_SAMPLES:
            mu, sd = lognorm_fit(ads)
            row["ad_mu_s"], row["ad_sd"] = round(mu, 1), round(sd, 3)
        if len(shows) >= MIN_SAMPLES:
            mu, sd = lognorm_fit(shows)
            row["show_mu_s"], row["show_sd"] = round(mu, 1), round(sd, 3)
        marker = "" if row else "   (zu wenig → global)"
        print(f"{ch:16} {len(ads):5d} {row.get('ad_mu_s','—'):>7} "
              f"{row.get('ad_sd','—'):>6} {len(shows):6d} "
              f"{row.get('show_mu_s','—'):>8} {row.get('show_sd','—'):>8}"
              f"{marker}")
        if row:
            out["channels"][ch] = row
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
