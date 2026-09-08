#!/usr/bin/env python3
"""O22 — hilft die Lautheits-SCHWANKUNG, wo die Lautheit selbst nichts bringt?

DIE FRAGE
---------
Die Spalten-Wichtigkeit am deployten Kopf (2026-09-08) zeigt: die
Audio-Spalte ist praktisch tot, Permutationsverlust 0.0021 gegen 0.2954
beim Logo. Die Spalte ist aber NICHT kaputt — sie enthaelt echte,
schwankende Werte. Kaputt ist ihre Praemisse.

Der Docstring von `extract_audio_rms_per_second` begruendet die Spalte mit
„German private TV consistently runs ads ~6-10 dB hotter than show
content". Gemessen an 293576 Sekunden aus 98 Aufnahmen: **1.23 dB.** Die
EU-Lautheitsregulierung hat den alten Trick erledigt; die Erinnerung
`audio_hardcut_signals_weak` hielt das schon fest.

WAS STATTDESSEN TRAEGT
----------------------
Dieselben Rohdaten, andere Statistik. Werbung ist stark komprimiert, ihre
Lautheit schwankt kaum; Sendung hat Dialog, Musik und Stille. Die
SCHWANKUNG ueber 30 Sekunden trennt deshalb besser als der Pegel selbst.
AUC innerhalb jeder Aufnahme, Median ueber 98 Aufnahmen:

    Lautheit (wie heute)        0.592   nuetzlich in 43 % der Aufnahmen
    Schwankung ueber 10 s       0.695   nuetzlich in 80 %
    Schwankung ueber 30 s       0.726   nuetzlich in 81 %

⚠️ Je Aufnahme gerechnet, nicht gepoolt. Gepoolt ueber alle Aufnahmen
misst man zum Teil Unterschiede ZWISCHEN Sendern und Sendungen statt
innerhalb — dieselbe Haeufungsfalle, die am 2026-09-07 aus einem
Faktor 1.8 einen Faktor 3.3 gemacht hat.

WAS DAS NOCH NICHT HEISST
-------------------------
AUC 0.73 allein sagt nichts darueber, ob die Spalte dem Kopf etwas
HINZUFUEGT. Das Backbone koennte dieselbe Information schon tragen
(laute, statische Bilder). Genau das misst dieser Lauf.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent


def _o20():
    spec = importlib.util.spec_from_file_location("o20", _HIER / "o20-klassen-split.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rollsd(a, k):
    """Gleitende Standardabweichung, ueber Praefixsummen statt Schleife."""
    a = a.astype(np.float64)
    c1 = np.cumsum(np.concatenate([[0.0], a]))
    c2 = np.cumsum(np.concatenate([[0.0], a * a]))
    n = len(a)
    lo = np.maximum(0, np.arange(n) - k // 2)
    hi = np.minimum(n, np.arange(n) + k // 2 + 1)
    m = (hi - lo).astype(np.float64)
    mu = (c1[hi] - c1[lo]) / m
    var = np.maximum(0.0, (c2[hi] - c2[lo]) / m - mu * mu)
    return np.sqrt(var).astype(np.float32)


def dynamik_spalte(X, rec, k=30):
    """Je AUFNAHME getrennt rechnen — ueber Aufnahmegrenzen hinweg zu
    glaetten wuerde am Rand Unsinn erzeugen."""
    aus = np.empty(X.shape[0], np.float32)
    for r in np.unique(rec):
        m = rec == r
        aus[m] = rollsd(X[m, 1281], k)
    return aus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--fenster", type=int, default=30)
    ap.add_argument("--nur-kontrollarm", action="store_true", dest="nur_kontroll")
    ap.add_argument("--json")
    a = ap.parse_args()

    o = _o20()
    print("Lade train …", flush=True)
    Xtr, ytr, rec_tr = o.lade("train", None, a.schritt)
    print("Lade test …", flush=True)
    Xte, yte, rec_te = o.lade("test", None, 1)

    # ⚠️ Die Dynamik-Spalte VOR der Standardisierung anhaengen, damit sie
    # dieselbe Behandlung bekommt wie jede andere. Und je Aufnahme
    # gerechnet — beim train-Satz mit Schrittweite 4, also entspricht das
    # Fenster dort 30 Zeilen = 120 Sekunden. Deshalb wird das Fenster in
    # ZEILEN gerechnet und fuer train durch die Schrittweite geteilt.
    dtr = dynamik_spalte(Xtr, rec_tr, max(2, a.fenster // a.schritt))
    dte = dynamik_spalte(Xte, rec_te, a.fenster)
    Xtr_p = np.concatenate([Xtr, dtr[:, None]], axis=1)
    Xte_p = np.concatenate([Xte, dte[:, None]], axis=1)
    print(f"train {Xtr.shape[0]} Zeilen, test {Xte.shape[0]}; "
          f"Dynamik-Fenster {a.fenster} s")

    Xtr_s, Xte_s = o.standardisieren(Xtr, Xte)
    Xtr_ps, Xte_ps = o.standardisieren(Xtr_p, Xte_p)
    wahr = (yte > 0).astype(np.int64)

    erg = {"ohne": [], "mit": []}
    arme = ["ohne"] if a.nur_kontroll else ["ohne", "mit"]
    for seed in range(a.seeds):
        for arm in arme:
            A, B = (Xtr_s, Xte_s) if arm == "ohne" else (Xtr_ps, Xte_ps)
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            s = o.f1((o.glaetten(p, rec_te) > 0.5).astype(np.int64), wahr)
            erg[arm].append(s)
            print(f"  Seed {seed}  {arm:<5} F1 {s:.4f}", flush=True)
    for arm in arme:
        v = np.array(erg[arm])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"\n{arm}: Median {np.median(v):.4f}  Mittel {v.mean():.4f}  sd {sd:.4f}")
    if not a.nur_kontroll:
        d = np.array(erg["mit"]) - np.array(erg["ohne"])
        print(f"\nDelta (mit minus ohne), gepaart: Median {np.median(d):+.4f}  "
              f"positiv in {int((d>0).sum())} von {len(d)} Seeds")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
