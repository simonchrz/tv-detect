# HSMM structured decoding — pre-registration of the holdout test

**Written 2026-07-27, BEFORE collecting the holdout dumps.** The point of
writing it first is that everything below can be checked against what actually
happens, instead of being reshaped afterwards to fit it. This repo has already
paid for the alternative twice: the 07-24 boundary-head result read +0.016 on
non-faithful dumps and −0.015 on faithful ones, and the first HMM result read
+0.242 against a mislabelled baseline and −0.004 against the real one.

## What is being tested

`blocks.Form` (hysteresis + rolling mean + snap windows) versus HSMM Viterbi
decoding over the *same* per-frame probabilities, with explicit state-duration
distributions.

## What is already measured (the training sample, n=22)

Faithful dumps, current head, production baseline via `production_replay()`,
one recording (`dvr-rtl-1785073500`) excluded because `gt-hygiene.py` shows its
labels are wrong in both directions — 1248 s of confident ad outside them and
847 s of confident show inside them.

```
  immer PROD   0.839
  immer HSMM   0.862   (+0.023)

  gate band < 0.049    0.886 in-sample, 0.881 leave-one-out
```

`band` = fraction of seconds where the smoothed NN probability lies in
(0.3, 0.7).

## Hypotheses, with numbers, before the data

**H1 — always-HSMM.** On ≥18 unseen recordings, HSMM beats production by
**≥ +0.010** mean block IoU.

*Confidence: moderate.* +0.023 on n=22 is not nothing, but it is carried by a
handful of large positive deltas (+0.322, +0.281, +0.253, +0.241) against a few
large negative ones (−0.365, −0.216, −0.207). A sample that happens to contain
one more catastrophic loser than winner flips the sign. I expect the holdout to
come out positive but smaller than +0.023.

**H2 — the band gate.** The gate `band < 0.049` beats always-HSMM by
**≥ +0.010** on the same unseen recordings, with the threshold FROZEN at 0.049.
No re-tuning. A gate that needs a new threshold on the holdout has failed.

*Confidence: low.* Three reasons to doubt it, all visible before the test:

1. The linear correlation of `band` to the delta is r = −0.09. There is no
   monotone relationship; the gate wins by cutting a three-recording tail.
2. It excludes the single largest win in the sample (CSI Vegas, band 0.127,
   +0.322) and keeps four losers.
3. Twelve proxy/direction combinations were swept. Leave-one-out protects
   against threshold overfitting, not against picking the luckiest of twelve.

I expect H2 to fail. If it passes anyway, that is informative and the mechanism
then needs explaining before anything ships.

## What each outcome means — decided now

| H1 | H2 | Consequence |
|---|---|---|
| pass | pass | Port Viterbi to Go, gated. Explain the mechanism first. |
| pass | fail | Port Viterbi to Go, **ungated**. Simpler and it is what held. |
| fail | pass | Do NOT ship. A gate that beats a baseline that itself lost is a curve. |
| fail | fail | Structured decoding is closed for good. Record it and stop. |

Whatever comes out, the golden-60 eval is the final gate before deploy — this
holdout only decides whether the Go port gets written at all.

## Holdout selection rules — fixed before looking at any result

Selection must not touch the outcome. A recording qualifies iff:

1. It is **not** among the 23 already measured.
2. It has a DVR grid entry (so `detect-config` resolves and the production
   replay is faithful).
3. Its source `.ts` is cached locally (no re-fetch, no truncation risk).
4. Its archive labels are `which == "merged"` — human-touched, not pure
   auto-confirm.
5. It has ≥1 ad block in the archive.
6. `gt-hygiene.py` does not flag it.

Ordering is by uuid; the first N per channel are taken, capped at 3 per show so
no single series dominates. **Production IoU is never consulted** — that is the
quantity under test.

### Amendment, same day, before any result was computed

The rule above produced a channel-skewed set: sorting by uuid let the
alphabetically early channels (comedy-central, disney-channel, kabel-eins,
nick) fill all 20 slots, with rtl, rtlzwei, vox and sat-1 absent entirely.
Structured decoding behaves differently per channel — the training sample's
per-channel deltas run from −0.089 to +0.161 — so a set missing the four
largest commercial channels would not test the hypotheses it claims to.

Replaced by **round-robin across channels**, max 2 per show, which is equally
blind to the outcome. Recorded here rather than quietly swapped, because a
selection rule edited after seeing results is how a holdout stops being one.
This edit was made before any IoU on the holdout was computed.
