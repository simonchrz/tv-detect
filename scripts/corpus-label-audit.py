#!/usr/bin/env python3
"""Audit the WHOLE training corpus for labels that contradict the signal.

Why. `gt-hygiene.py` checks recordings for which a faithful signal dump exists —
23 of them. One (dvr-rtl-1785073500) turned out to have labels wrong in both
directions: 1248 s of confident ad outside them, 847 s of confident show inside
them. Measured against those labels production reads 0.157, and that number
went into a comparison as if it meant something.

If that rate holds corpus-wide, the nightly training is fitting broken labels
and every gate downstream inherits them. But producing a signal dump means
running a full detect (video decode, minutes per recording), so 400 of them is
not a check, it is an overnight job.

The shortcut: the train archive already stores per-second backbone features
(1282-dim) next to the per-second labels, and the deployed head is a 32-hidden
MLP over 1299 inputs. The missing 17 columns — channel one-hot, whisper,
temporal L2 deltas, minute prior — are pure augmentation, reproducible from
sidecars that are all on disk. So the head's per-second probability can be
recomputed for every archived recording with no decode at all.

The reconstruction is NOT frame-exact, and cannot be. Detect runs the head per
FRAME at 25 fps — the dumped confidences vary within a single second by up to
sigma 0.26 — while the archive stores one pooled 1282-dim feature vector per
second. The head is a relu MLP, so head(pooled) != mean(head(per-frame)). A
first attempt to verify probability-by-probability read r = 0.79..0.995 and
blamed the archive; the mismatch was the comparison, not the data.

So --verify checks the thing this script actually claims. It runs the audit
verdict on the reconstruction AND on the real dump for every recording that has
both, and reports whether they agree on which recordings are flagged. Agreement
on the verdict is what makes the corpus sweep meaningful; agreement on the third
decimal was never available. Do not trust an audit whose verification did not
pass.

HOLE AND PHANTOM ARE NOT EQUALLY STRONG EVIDENCE. A hole (NN confidently AD
where the labels say show) is hard evidence of a missed break: nothing in the
pipeline suppresses a stretch the head calls ad for minutes on end. A phantom
(labels say AD where the NN says show) is much weaker, because production forms
blocks from a BLEND — nn_weight defaults to 0.3, so on a channel where the logo
carries the decision a block can be entirely logo-driven while the NN quietly
disagrees. For a which=auto recording, whose labels ARE that block output, a
phantom says "logo and NN disagree here", not "the label is wrong".

WHAT A PHANTOM ACTUALLY IS, established 2026-07-28 by looking at frames on
three recordings (contact sheets every 12 s across the disputed stretch):
PROGRAMME TRAILERS. German commercial breaks end with promos for the channel's
own shows — real programme footage, often carrying the promoted show's own
on-screen bug — so the head reads them as "show", which is what a phantom is.
They are part of the break, not of the programme.

    recording              show resumes   label ends      NN edge
    rtlzwei-1781943300        ~54:06      54:35 (+29s)   53:11 (-55s)
    rtlzwei-1781946900        ~34:42      35:27 (+45s)   34:06 (-36s)
    kabel-eins-1779379007     ~33:00      33:49 (+49s)   32:28 (-32s)

So BOTH sides are wrong and in opposite directions: labels run 30-50 s past the
show's return, the NN edge sits 30-55 s before it. Trimming a block to the NN
edge — which is what "fix the phantom" naively means, and what was actually
done to five recordings before this was understood — inherits the second error.
Measured on the three above: one clearly worse, one neutral, one better. All
five were reverted.

USER DECISION 2026-07-28: programme trailers count as SHOW — they must not be
cut. That settles which side is wrong: the LABELS are, twice over. They cut
trailers the user wants to keep AND overshoot 30-50 s into the programme, so
the app's ad-skip jumps past the show's first half minute. The NN edge sits
approximately at trailer start and is therefore the better boundary under this
preference (residual error ~30 s of trailer, versus ~90 s of trailer+show).
The five reverted trims were re-applied on that basis.

Open detector-side: fresh detects still END blocks past the show's return
(the ident/bumper sits between trailers and show and the end-snap lands on
it). Fixing that at the source is a separate, measured change — the trim
sweep only corrects existing labels.

The earlier note below stands but was reached for the wrong reason.

ALSO SETTLED BY REVIEW 2026-07-28, and it cost a wrong diagnosis first.
dvr-disney-channel-1781583000 was flagged with 304 s of phantom; the NN called
the disputed stretch "show" for six minutes, and enabling logo smoothing
removed the block cleanly — a tidy fix for an apparent defect. Watched by a
human, the block is REAL advertising: "Micky Maus Wunderhaus+" is a kids'
strand of several short cartoons, and a different cartoon starts right after
it. The logo was right, the NN was wrong, and the "fix" would have deleted a
correct block (it also scored 0.102 WORSE on the one neighbouring recording
with human labels — Disney Channel's logo_smooth_s=0 is deliberate).

So phantoms are now counted ONLY where a human put the label. Every recording
quarantined on 2026-07-27 was flagged for a HOLE with phantom exactly 0, so
that decision is unaffected.

Column layout, from train-head.py `_augment_cwt_minuteprior`:

    [ X(1282) | channel one-hot(13) | whisper(1) | dp(1) | dn(1) | prior(1) ]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

ARCH = os.path.expanduser("~/.cache/tvd-train-archive")
MODELS = os.path.expanduser("~/.cache/tv-detect-daemon")
WHISPER = os.path.expanduser("~/.cache/tv-whisper")
DUMPS = os.path.expanduser("~/.cache/tv-detect-daemon/emit-signals")


def load_head(path):
    raw = open(path, "rb").read()
    name = raw[:4].decode("latin1")
    nh = {"MLP4": 12, "MLP3": 11, "MLP2": 10}.get(name)
    if nh is None:
        raise SystemExit(f"unbekannter head-magic {name!r}")
    hdr = struct.unpack(f"<{nh}I", raw[:nh * 4])
    idim, hdim, odim = hdr[2], hdr[3], hdr[4]
    off = nh * 4

    def take(n):
        nonlocal off
        a = np.frombuffer(raw, np.float32, count=n, offset=off).astype(np.float64)
        off += n * 4
        return a
    W1 = take(idim * hdim).reshape(idim, hdim)
    b1 = take(hdim)
    W2 = take(hdim * odim).reshape(hdim, odim)
    b2 = take(odim)
    return idim, (W1, b1, W2, b2)


def head_prob(X, p):
    W1, b1, W2, b2 = p
    h = np.maximum(X.astype(np.float64) @ W1 + b1, 0.0)
    o = (h @ W2 + b2).ravel()
    return 1.0 / (1.0 + np.exp(-o))


def whisper_per_sec(uuid, n):
    p = os.path.join(WHISPER, f"{uuid}.whisper.json")
    if not os.path.isfile(p):
        return np.full(n, 0.5, np.float32)
    try:
        d = json.loads(open(p).read())
    except Exception:
        return np.full(n, 0.5, np.float32)
    ws = int(d.get("window_s", 60))
    s = np.zeros(n, np.float32)
    c = np.zeros(n, np.int32)
    for w in d.get("windows", []):
        lo = max(0, int(w.get("t", 0)))
        hi = min(n, lo + ws)
        if hi > lo:
            s[lo:hi] += float(w.get("prob", 0.5))
            c[lo:hi] += 1
    out = np.full(n, 0.5, np.float32)
    m = c > 0
    out[m] = s[m] / c[m]
    return out


def build_X(feat, slug, uuid, start_ts, chan_idx, n_chan, prior, neutral):
    T = feat.shape[0]
    oh = np.zeros((T, n_chan), np.float32)
    if slug in chan_idx:
        oh[:, chan_idx[slug]] = 1.0
    wp = whisper_per_sec(uuid, T).reshape(-1, 1)
    dp = np.zeros((T, 1), np.float32)
    dn = np.zeros((T, 1), np.float32)
    if T > 1:
        d = np.linalg.norm(feat[1:] - feat[:-1], axis=1).astype(np.float32)
        dp[1:, 0] = d
        dn[:-1, 0] = d
    if start_ts and slug in prior:
        arr = np.array(prior[slug], np.float32)
        minutes = ((start_ts + np.arange(T)) // 60 % 60).astype(int)
        mp = arr[minutes].reshape(-1, 1)
    else:
        mp = np.full((T, 1), neutral, np.float32)
    return np.hstack([feat, oh, wp, dp, dn, mp]).astype(np.float32)



def which_of(meta):
    """Label provenance: "user"/"merged" mean a human touched it."""
    return meta.get("which", "?")

def runs(mask, min_len):
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare reconstruction against real dumps and exit")
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--min-run", type=int, default=90)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    idim, params = load_head(os.path.join(MODELS, "head.bin"))
    # Sidecar is {"ts","version","n","slugs":[...]}; one-hot index is the
    # position in "slugs", which is the order train-head.py built it in.
    cmap = json.loads(open(os.path.join(MODELS, "head.channel-map.json")).read())
    slugs = cmap["slugs"] if isinstance(cmap, dict) else list(cmap)
    chan_idx = {s: i for i, s in enumerate(slugs)}
    # Sidecar is {"ts","version","neutral","priors":{slug:[60 floats]}}. Take
    # the stored neutral rather than recomputing the mean — that is the value
    # the deployed head was trained against.
    prior_path = os.path.join(MODELS, "head.minute-prior.json")
    pj = json.loads(open(prior_path).read()) if os.path.exists(prior_path) else {}
    prior = pj.get("priors", pj if "priors" not in pj else {})
    neutral = float(pj.get("neutral", 0.25))
    n_chan = idim - 1282 - 4
    print(f"head input_dim={idim}  n_chan={n_chan}  "
          f"channel-map={len(chan_idx)}  prior-slugs={len(prior)}  "
          f"neutral={neutral:.3f}")
    if n_chan != len(chan_idx):
        print(f"  WARNUNG: n_chan {n_chan} != channel-map {len(chan_idx)}")

    files = sorted(glob.glob(os.path.join(ARCH, "dvr-*.npz")))
    if args.limit:
        files = files[:args.limit]
    newest_npz = max((os.path.getmtime(f) for f in files), default=0.0)

    def verdict(ps, ads, n):
        """The audit's decision for one recording: (hole_s, phantom_s, flagged)."""
        k = max(1, args.smooth)
        ps = np.convolve(ps, np.ones(k) / k, mode="same")
        gt = np.zeros(n, bool)
        for a, b in (ads or []):
            gt[int(max(0, a)):int(min(n, b))] = True
        hole = sum(b - a for a, b in runs((ps[:n] > args.hi) & ~gt, args.min_run))
        phan = sum(b - a for a, b in runs((ps[:n] < args.lo) & gt, args.min_run))
        return hole, phan, (hole > 4 * args.min_run or phan > 2 * args.min_run)

    if args.verify:
        print("\n=== Verifikation: stimmt das URTEIL ueberein? ===")
        print("   (Wahrscheinlichkeiten koennen es nicht — Detect rechnet pro")
        print("    Frame bei 25 fps, das Archiv speichert Sekunden-Features.)\n")
        print(f"{'uuid':30} {'korr':>7} | {'rekonstruiert':>22} | {'echter Dump':>22}")
        agree = disagree = 0
        for f in files:
            u = os.path.basename(f)[:-4]
            dp_ = os.path.join(DUMPS, f"{u}.json")
            if not os.path.exists(dp_):
                continue
            m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
            fp = m.get("feature_npy", "")
            if not os.path.exists(fp):
                continue
            feat = np.load(fp)
            X = build_X(feat, m.get("slug", ""), u, int(m.get("start_ts") or 0),
                        chan_idx, n_chan, prior, neutral)
            if X.shape[1] != idim:
                # Older extraction with a different feature width — not a
                # verification failure, just not comparable.
                print(f"{u:30} uebersprungen: Feature-Breite {feat.shape[1]} "
                      f"ergibt {X.shape[1]}, Head will {idim}")
                continue
            mine = head_prob(X, params)
            d = json.loads(open(dp_).read())
            nn = np.array(d["nn_confs"], np.float64)
            fps = d["fps"]
            # dump is per-FRAME, features are per-SECOND: reduce the dump the
            # same way the harness does before comparing.
            n = min(len(mine), int(len(nn) / fps))
            idx = np.clip((np.arange(len(nn)) / fps).astype(int), 0, n - 1)
            s = np.zeros(n)
            c = np.zeros(n)
            np.add.at(s, idx, nn)
            np.add.at(c, idx, 1.0)
            theirs = s / np.maximum(c, 1.0)
            corr = (float(np.corrcoef(mine[:n], theirs[:n])[0, 1])
                    if n > 2 else float("nan"))
            ads = m.get("ads") or []
            h1, p1, f1 = verdict(mine, ads, n)
            h2, p2, f2 = verdict(theirs, ads, n)
            same = f1 == f2
            agree += same
            disagree += (not same)
            mark = "" if same else "   <-- URTEILE WEICHEN AB"
            print(f"{u:30} {corr:7.4f} | hole {h1:5d} phan {p1:5d} "
                  f"{'FLAG' if f1 else '  ok'} | hole {h2:5d} phan {p2:5d} "
                  f"{'FLAG' if f2 else '  ok'}{mark}")
        print(f"\n  {agree} Urteile gleich, {disagree} verschieden")
        if disagree:
            print("  Rekonstruktion NICHT vertrauenswuerdig — Audit waere Unsinn.")
            return 1
        print("  Rekonstruktion bestaetigt — der Audit darf laufen.")
        return 0

    print(f"\n=== Audit ueber {len(files)} archivierte Aufnahmen ===")
    flagged, checked, skipped = [], 0, 0
    for f in files:
        u = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        fp = m.get("feature_npy", "")
        if not os.path.exists(fp):
            skipped += 1
            continue
        feat = np.load(fp, mmap_mode="r")
        X = build_X(np.asarray(feat), m.get("slug", ""), u,
                    int(m.get("start_ts") or 0), chan_idx, n_chan, prior, neutral)
        if X.shape[1] != idim:
            skipped += 1
            continue
        ps = head_prob(X, params)
        k = max(1, args.smooth)
        ps = np.convolve(ps, np.ones(k) / k, mode="same")
        n = len(ps)
        gt = np.zeros(n, bool)
        for a, b in (m.get("ads") or []):
            gt[int(max(0, a)):int(min(n, b))] = True
        hole = sum(b - a for a, b in runs((ps > args.hi) & ~gt, args.min_run))
        phan = sum(b - a for a, b in runs((ps < args.lo) & gt, args.min_run))
        checked += 1
        # A phantom only counts when a HUMAN put the label there. On a
        # machine-labelled recording the label IS the block output, which on a
        # logo-driven channel can be formed almost entirely from the logo while
        # the NN disagrees — so the "contradiction" is just the blend's two
        # inputs disagreeing, which is neither news nor an error.
        #
        # Settled by review on 2026-07-28: dvr-disney-channel-1781583000 was
        # flagged with 304 s of phantom, the NN called it show for six minutes,
        # and the block is REAL advertising — "Micky Maus Wunderhaus+" is a
        # kids' strand of several short cartoons with breaks between them, and
        # a different cartoon starts right after the disputed block. The logo
        # was right and the NN was wrong. Counting that as a label defect
        # produced a false alarm and nearly a wrong config change.
        human = which_of(m) in ("user", "merged")
        if not human:
            phan = 0
        if hole > 4 * args.min_run or phan > 2 * args.min_run:
            # A recording whose npz the last training run did NOT rewrite was
            # not evaluated by it — for a labelled recording that means its
            # labels went empty (a label-less recording counts as bootstrap and
            # drops out of train AND test, and no npz is written for it). Its
            # archive entry is then frozen at the last state that HAD labels,
            # and the audit would keep reporting a contradiction that no longer
            # exists anywhere but in that stale file. Seen on
            # dvr-nick-1778954400 and dvr-rtl-1781909700, the latter of which
            # also silently shortened the golden median for two nights.
            frozen = (newest_npz - os.path.getmtime(f)) > 36 * 3600
            flagged.append((u, m.get("which", "?"), str(m.get("title"))[:26],
                            n, hole, phan, frozen))
    print(f"  geprueft {checked}, uebersprungen {skipped} (keine Features)")
    print(f"\n{len(flagged)} Aufnahmen widersprechen ihrem eigenen Signal:")
    print(f"  {'uuid':30} {'which':>7} {'dauer':>6} {'hole':>6} {'phantom':>8}"
          f"  Titel")
    for u, w, t, n, hole, phan, frozen in sorted(flagged,
                                                 key=lambda r: -(r[4] + r[5])):
        tag = "  [ARCHIV EINGEFROREN — Labels evtl. schon leer]" if frozen else ""
        print(f"  {u:30} {w:>7} {n:6d} {hole:6d} {phan:8d}  {t}{tag}")
    if flagged:
        json.dump([r[0] for r in flagged], open("/tmp/label-audit-flagged.json", "w"))
        print("\n  uuids -> /tmp/label-audit-flagged.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
