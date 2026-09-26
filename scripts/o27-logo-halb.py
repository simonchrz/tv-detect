#!/usr/bin/env python3
"""O27 — Hilft es, die Logo-Spalte bei der Produktions-Aufloesung zu trainieren?

Registrierung: docs/o27-logo-halbe-aufloesung-preregistration.md.

Arme (gleiche Zeilen, Seeds, Architektur, Standardisierung aus train):
  kontrolle  Logo-Spalte (Index 1280) aus dem Archiv, also VOLLE Aufloesung
  versuch    Logo-Spalte bei HALBER Aufloesung, wo die Quelle noch lag
             (~/.cache/tvd-o27-logo-halb, erzeugt mit o27-vormessung.py /
             derselben Funktion wie das Training); sonst Archiv
Primaer: F1 auf menschlich gelabelten test-Aufnahmen mit Quelle.
Neben:   F1 auf allen test-Aufnahmen mit halber Spalte.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent


def _vm():
    spec = importlib.util.spec_from_file_location("o27vm", HIER / "o27-vormessung.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["o27vm"] = m
    spec.loader.exec_module(m)
    return m


def halbe_spalte(X, rec, uuids, schritt, cache):
    """Ersetzt Spalte 1280 zeilengenau (Sekunden 0, schritt, 2*schritt …)
    durch die halbe Aufloesung, wo sie vorliegt. Liefert (X', Menge der
    ersetzten Aufnahme-Indizes)."""
    X = X.copy()
    ersetzt = set()
    for i, u in enumerate(uuids):
        c = cache / f"{u}.npy"
        if not c.is_file():
            continue
        v = np.load(c)
        m = rec == i
        k = int(m.sum())
        w = v[::schritt][:k]
        if len(w) != k:
            continue
        X[m, 1280] = np.where(np.isnan(w), 0.5, w).astype(np.float32)
        ersetzt.add(i)
    return X, ersetzt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()

    vm = _vm()
    o = vm._lade("o20_o27l", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_o27l", HIER / "label_herkunft.py")

    def mensch(u):
        f = vm.SNAPSHOT / f"_rec_{u}" / "ads_user.json"
        try:
            return lh.mensch_aus_markern(json.loads(f.read_text())) is True
        except Exception:
            return False

    print("Lade train …", flush=True)
    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    print("Lade test …", flush=True)
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)

    Xtr_h, tr_ers = halbe_spalte(Xtr, rec_tr, u_tr, a.schritt, vm.HALB_CACHE)
    Xte_h, te_ers = halbe_spalte(Xte, rec_te, u_te, 1, vm.HALB_CACHE)
    primaer_rec = np.array([i in te_ers and mensch(u) for i, u in enumerate(u_te)])
    neben_rec = np.array([i in te_ers for i in range(len(u_te))])
    print(f"train {len(u_tr)} Aufnahmen, davon {len(tr_ers)} mit halber Spalte; "
          f"test mit halber Spalte {int(neben_rec.sum())}, davon menschlich {int(primaer_rec.sum())}",
          flush=True)
    m_prim, m_neben = primaer_rec[rec_te], neben_rec[rec_te]

    Xtr_s, Xte_s = o.standardisieren(Xtr, Xte)
    Xtr_hs, Xte_hs = o.standardisieren(Xtr_h, Xte_h)
    wahr = (yte > 0).astype(np.int64)

    def werte(p):
        pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
        return o.f1(pred[m_prim], wahr[m_prim]), o.f1(pred[m_neben], wahr[m_neben])

    erg = {"kontrolle": [], "versuch": []}
    for seed in range(a.seeds):
        for arm, (A, B) in (("kontrolle", (Xtr_s, Xte_s)), ("versuch", (Xtr_hs, Xte_hs))):
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            prim, neben = werte(p)
            erg[arm].append([prim, neben])
            print(f"  Seed {seed}  {arm:<9} F1 primaer {prim:.4f}  neben {neben:.4f}", flush=True)
    for arm in erg:
        v = np.array(erg[arm])[:, 0]
        print(f"\n{arm}: primaer Median {np.median(v):.4f}  sd {v.std(ddof=1):.4f}")
    d = np.array(erg["versuch"])[:, 0] - np.array(erg["kontrolle"])[:, 0]
    d2 = np.array(erg["versuch"])[:, 1] - np.array(erg["kontrolle"])[:, 1]
    print(f"\nDelta primaer (versuch - kontrolle), gepaart: Median {np.median(d):+.4f}  "
          f"positiv in {int((d > 0).sum())} von {len(d)} Seeds")
    print(f"Delta Nebenwert: Median {np.median(d2):+.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
