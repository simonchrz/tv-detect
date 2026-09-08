#!/usr/bin/env python3
"""Lernkurve: bringen mehr Labels überhaupt noch etwas?

DIE FRAGE
---------
Mehrere Vorschläge dieser Sitzung laufen auf „mehr oder feinere Labels
beschaffen" hinaus — Unterklassen für Trailer und Ident (Idee 5), echte
Anker-Labels, mehr reviewte Aufnahmen. Alle sind teuer, und keiner wurde
je gegen die einfachste Gegenfrage gehalten: **ist der Korpus überhaupt
noch hungrig?**

Die Lernkurve beantwortet das ohne ein einziges neues Label. Derselbe
Kopf, trainiert auf 25 / 50 / 75 / 100 % der Aufnahmen, gemessen auf
demselben Testsatz. Flacht die Kurve zwischen 75 und 100 % ab, ist der
Korpus gesättigt und jede Beschaffungsidee ist erledigt. Steigt sie noch,
sagt die Steigung, was ein Verdoppeln brächte.

⚠️ GEZOGEN WIRD NACH AUFNAHME, NICHT NACH ZEILE. Sekunden derselben
Aufnahme sind hochgradig abhängig; wer zeilenweise zieht, hat bei 25 %
immer noch fast alle Aufnahmen im Satz und misst die Sättigung viel zu
früh. Der Split-Ledger zieht aus demselben Grund nach uuid.

Die Anteile bauen aufeinander auf (25 % ⊂ 50 % ⊂ 75 %), damit die Kurve
nicht zusätzlich von der Zusammensetzung springt.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--anteile", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--json")
    a = ap.parse_args()

    o = _o20()
    print("Lade train …", flush=True)
    Xtr, ytr, rec_tr = o.lade("train", None, a.schritt)
    print("Lade test …", flush=True)
    Xte, yte, rec_te = o.lade("test", None, 1)
    Xtr, Xte = o.standardisieren(Xtr, Xte)
    wahr = (yte > 0).astype(np.int64)
    n_rec = int(rec_tr.max()) + 1
    print(f"train {Xtr.shape[0]} Zeilen aus {n_rec} Aufnahmen, "
          f"test {Xte.shape[0]} Zeilen")

    anteile = [float(x) for x in a.anteile.split(",")]
    erg = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        reihenfolge = rng.permutation(n_rec)     # je Seed eine andere Ziehung
        for p in anteile:
            k = max(1, int(round(p * n_rec)))
            nimm = set(reihenfolge[:k].tolist())   # geschachtelt: 25 % ⊂ 50 %
            m = np.isin(rec_tr, list(nimm))
            pr = o.fit_und_werte(Xtr[m], ytr[m], Xte, yte, rec_te, 2, seed,
                                 a.epochen, a.hidden)
            f1 = o.f1((o.glaetten(pr, rec_te) > 0.5).astype(np.int64), wahr)
            erg.setdefault(p, []).append(f1)
            print(f"  Seed {seed}  {100*p:>5.0f} % = {k:>3} Aufnahmen, "
                  f"{int(m.sum()):>7} Zeilen  F1 {f1:.4f}", flush=True)

    print(f"\n{'Anteil':>8}{'Aufnahmen':>11}{'F1 Median':>11}{'Zuwachs':>10}")
    vor = None
    for p in anteile:
        v = np.array(erg[p])
        med = float(np.median(v))
        k = max(1, int(round(p * n_rec)))
        zu = f"{med-vor:+.4f}" if vor is not None else "—"
        print(f"{100*p:>7.0f}%{k:>11}{med:>11.4f}{zu:>10}")
        vor = med
    letzte = float(np.median(erg[anteile[-1]])) - float(np.median(erg[anteile[-2]]))
    print(f"\n  Zuwachs im letzten Viertel: {letzte:+.4f}")
    print("  Ist er kleiner als das Seed-Rauschen (~0.0045), ist der Korpus")
    print("  gesaettigt und mehr Labels derselben Art bringen nichts.")
    if a.json:
        Path(a.json).write_text(json.dumps({str(k): v for k, v in erg.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
