#!/usr/bin/env python3
"""Welche Eingabespalten benutzt der DEPLOYTE Kopf wirklich?

DIE FRAGE
---------
Der Kopf sieht 1282 Spalten: 1280 Backbone, eine Logo-, eine Audio-Spalte,
dazu je nach Kopf-Header Kanal-, Temporal-, Whisper- und Minute-Prior-
Spalten. Welche davon tragen die Entscheidung? Wer das nicht weiss, kann
weder sagen, ob ein Signal FEHLT, noch ob eines still TOT ist.

Und tot war hier schon mehr als einmal etwas: der Minute-Prior lag acht
Naechte inert daneben (Δ≈-0.001), die Zusatzspalten waren durch einen
Auffueller um eine Position verschoben, und der Logo-Sentinel maskierte
einen stillen Extraktionsfehler. Alle drei fielen erst spaet auf.

DIE METHODE
-----------
Permutations-Wichtigkeit: eine Spalte über alle Zeilen durchmischen und
messen, wie viel Leistung verlorengeht. Ist der Verlust null, benutzt der
Kopf die Spalte nicht. Gemessen wird am EINGEFRORENEN Kopf — nichts wird
trainiert, nichts deployt.

⚠️ Die 1280 Backbone-Spalten werden als GRUPPE gemischt, nicht einzeln.
Einzeln misst man nur, wie redundant eine Einbettung ist (jede Spalte
sieht wertlos aus, weil 1279 andere dasselbe sagen) — die typische Falle
der Permutations-Wichtigkeit bei korrelierten Merkmalen. Die Frage lautet
hier: was traegt der Backbone GEGEN Logo, Audio und die Zusatzspalten.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
ARCH = Path.home() / ".cache" / "tvd-train-archive"


def _audit():
    spec = importlib.util.spec_from_file_location("cla", _HIER / "corpus-label-audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kopf", default=str(Path.home() / ".cache/tv-train-head-out/head.bin"))
    ap.add_argument("--limit", type=int, default=60, help="Aufnahmen aus dem Messsatz")
    ap.add_argument("--wiederholungen", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()

    cla = _audit()
    idim, params, felder = cla.load_head(a.kopf)
    print(f"Kopf {a.kopf}")
    print(f"  Eingabebreite {idim}, Zusatzspalten {felder}")

    ms = json.loads((ARCH / "messsatz-2026-09-07.json").read_text())
    uuids = ms["uuids"][:a.limit]
    import importlib.util as iu
    spec = iu.spec_from_file_location("bs", _HIER / "backbone-sonde.py")
    bs = iu.module_from_spec(spec); spec.loader.exec_module(bs)
    ub = bs.menschlabels()

    Xs, ys = [], []
    for u in uuids:
        f = ARCH / f"{u}.npz"
        if not f.is_file():
            continue
        try:
            m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        except Exception:
            continue
        fp = m.get("feature_npy", "")
        truth = ub.get(u)
        if not fp or not os.path.exists(fp) or not truth:
            continue
        feat = np.asarray(np.load(fp, mmap_mode="r"), np.float32)
        X = cla.build_X(feat, m.get("slug", ""), u, int(m.get("start_ts") or 0),
                        {}, felder.get("channel", 0), None, 0.175,
                        n_temporal=felder.get("temporal", 0) and 2 or 0,
                        with_whisper=bool(felder.get("whisper", 0)),
                        with_prior=bool(felder.get("minuteprior", 0)))
        if X.shape[1] != idim:
            continue
        n = X.shape[0]
        y = np.zeros(n, np.int8)
        for s, e in truth:
            y[max(0, int(s)):min(n, int(e))] = 1
        Xs.append(X[::3]); ys.append(y[::3])
    if not Xs:
        print("keine Daten"); return 1
    X = np.vstack(Xs); y = np.concatenate(ys)
    print(f"  {X.shape[0]} Zeilen aus {len(Xs)} Aufnahmen, {100*y.mean():.1f} % Werbung\n")

    def guete(Xa):
        p = cla.head_prob(Xa, params)
        pred = (p > 0.5).astype(np.int8)
        tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

    basis = guete(X)
    print(f"  F1 unveraendert: {basis:.4f}\n")

    gruppen = {"Backbone (0-1279)": list(range(0, 1280)),
               "Logo (1280)": [1280], "Audio (1281)": [1281]}
    for i in range(1282, idim):
        gruppen[f"Zusatzspalte {i}"] = [i]

    rng = np.random.default_rng(0)
    erg = {}
    print(f"{'Gruppe':<26}{'F1 gemischt':>13}{'Verlust':>10}")
    for name, idx in gruppen.items():
        verluste = []
        for _ in range(a.wiederholungen):
            Xp = X.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, idx] = Xp[perm][:, idx]
            verluste.append(basis - guete(Xp))
        v = float(np.median(verluste))
        erg[name] = round(v, 4)
        marke = "  <- unbenutzt" if abs(v) < 0.002 else ""
        print(f"  {name:<24}{basis-v:>13.4f}{v:>10.4f}{marke}")
    if a.json:
        Path(a.json).write_text(json.dumps({"basis": basis, "verlust": erg}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
