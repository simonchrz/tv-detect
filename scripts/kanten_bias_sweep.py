#!/usr/bin/env python3
"""Sweep über --hsmm-ad-bias: was kauft Ad-Aversion an den Kanten?

Fährt den NACKTEN Kandidaten-Kopf (Probelauf 2026-08-13) über den echten
Replay-Pfad (build/tv-detect --replay-signals, EVAL_DECODER) mit
verschiedenen Ad-Bias-Werten über den Golden-Satz und misst je Bias:

  * Sendung-geschnitten-Sekunden (das Ziel — runter)
  * Werbung-gelassen-Sekunden (der Preis — darf rauf)
  * Kanten exakt (|Δ| ≤ 2 s) und Block-IoU-Median (die Wache — nicht kippen)

Exploratorische MESSUNG, kein Urteil: sie liefert die Kurve, aus der eine
Registrierung Schwelle und Zielwert VOR der Serie festlegt.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
SC = Path("/private/tmp/claude-501/-Users-simon-src-Comskip/"
          "78f3aae5-6661-4a6a-80a9-6e1ff6bb06c3/scratchpad")
HEAD = SC / "bare-probelauf/out/head.bin"
ARCHIV = Path.home() / ".cache/tvd-train-archive"


def lade_kopf(pfad):
    raw = pfad.read_bytes()
    magic, ver, ind, hid, out, bb, nl, na, nc = struct.unpack("<9I", raw[:36])
    assert magic == 0x31504C4D and nc == 0, "kein bare MLP1"
    off = 36
    def arr(n):
        nonlocal off
        a = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
        off += n * 4
        return a
    W1 = arr(ind * hid).reshape(ind, hid)
    b1 = arr(hid)
    W2 = arr(hid * out).reshape(hid, out)
    b2 = arr(out)
    return W1, b1, W2, b2, ind


def proba(X, kopf):
    W1, b1, W2, b2, ind = kopf
    assert X.shape[1] == ind, (X.shape, ind)
    h = np.maximum(X.astype(np.float32) @ W1 + b1, 0)
    logit = (h @ W2 + b2).ravel().astype(np.float64)
    return 1.0 / (1.0 + np.exp(-logit))


def iou(a, b):
    def summe(bl):
        return sum(e - s for s, e in bl)
    schnitt = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            schnitt += max(0.0, min(e1, e2) - max(s1, s2))
    u = summe(a) + summe(b) - schnitt
    return schnitt / u if u > 0 else 1.0


def main():
    spec = importlib.util.spec_from_file_location("th", REPO / "train-head.py")
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    spec2 = importlib.util.spec_from_file_location("fm", REPO / "fehlermoden.py")
    fm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fm)

    kopf = lade_kopf(HEAD)
    golden = json.loads((ARCHIV / "golden-eval-set.json").read_text())["uuids"]
    basis_decoder = list(th.EVAL_DECODER)

    faelle = []
    import glob
    for u in golden:
        cache = th.SIGNALS_CACHE / f"{u}.json"
        feats = sorted(glob.glob(
            str(Path.home() / f".cache/tvd-features/{u}-*fps100-l2-a1.npy")))
        d = fm.SNAPSHOT / f"_rec_{u}"
        user = fm.bloecke(d / "ads_user.json")
        if not cache.exists() or not feats or not user:
            continue
        X = np.load(feats[-1])
        if X.shape[1] != kopf[4]:
            # Alte Feature-Datei mit anderem Spaltensatz — ehrlich
            # auslassen statt heimlich zu polstern.
            print(f"  uebersprungen {u}: dim {X.shape[1]} != {kopf[4]}")
            continue
        faelle.append((u, cache, proba(X, kopf), user))
    print(f"{len(faelle)}/{len(golden)} Golden-Aufnahmen mit Signalen, "
          f"Features und Labels\n")
    print(f"{'bias':>6} {'SendGeschn':>10} {'WerbGelass':>10} "
          f"{'Kanten<=2s':>10} {'IoU-Median':>10} {'Bloecke':>7}")

    for bias in (0.0, -0.2, -0.4, -0.7, -1.0, -1.5, 0.3):
        th.EVAL_DECODER = basis_decoder + (
            ["--hsmm-ad-bias", str(bias)] if bias else [])
        geschn = gelass = 0.0
        exakt = kanten = nbloecke = 0
        ious = []
        for u, cache, p, user in faelle:
            bl = th._replay_blocks(cache, p, 1.0, u)
            if bl is None:
                continue
            auto = [(float(a), float(b)) for a, b in bl]
            z = fm.zerlege(auto, user)
            geschn += z["sendung_geschnitten_s"]
            gelass += z["werbung_gelassen_s"]
            nbloecke += len(auto)
            ious.append(iou(auto, user))
            for ub, ab, ds, de in z["grenze"]:
                pass
            for ub in user:
                # Kanten-Exaktheit gegen den besten Partner
                best = min(auto, key=lambda a: abs(a[0]-ub[0])+abs(a[1]-ub[1]),
                           default=None)
                if best:
                    for dv in (best[0]-ub[0], best[1]-ub[1]):
                        kanten += 1
                        if abs(dv) <= 2.0:
                            exakt += 1
        print(f"{bias:>6} {geschn:>10.0f} {gelass:>10.0f} "
              f"{f'{exakt}/{kanten}':>10} "
              f"{np.median(ious):>10.3f} {nbloecke:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
