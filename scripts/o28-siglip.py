#!/usr/bin/env python3
"""O28 — Hebt ein text-faehiger Bild-Encoder (SigLIP 2) als Zusatzblock die Leistung?

Registrierung: docs/o28-siglip2-preregistration.md.

Arme (gleiche Zeilen, Seeds, Architektur, Standardisierung aus train):
  kontrolle  PRODUKTIONSZUSTAND: Archiv-Merkmale, test-Logo-Spalte halb
             (wie der Detect; Lehre aus O27)
  versuch    kontrolle + 64 SigLIP-2-Hauptkomponenten + siglip_da.
             PCA NUR auf train-Zeilen mit Merkmalen; ohne Merkmale Nullen
             und siglip_da=0.
Primaer: F1 auf menschlich gelabelten test-Aufnahmen mit halber Logo-Spalte
UND SigLIP-Merkmalen. Neben: alle test-Aufnahmen mit beidem.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
SIGLIP_CACHE = Path.home() / ".cache/tvd-siglip2"
KOMPONENTEN = 64


def _lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def siglip_zeilen(rec, uuids, schritt, cache):
    """Roh-Merkmale (n, 768) zeilengenau wie halbe_spalte (Sekunden 0,
    schritt, …). Liefert (S, da, Menge der Aufnahme-Indizes mit Merkmalen)."""
    n = len(rec)
    S = np.zeros((n, 768), np.float32)
    da = np.zeros(n, np.float32)
    mit = set()
    for i, u in enumerate(uuids):
        c = cache / f"{u}.npy"
        if not c.is_file():
            continue
        m = rec == i
        k = int(m.sum())
        w = np.load(c)[::schritt][:k]
        if len(w) != k:
            continue
        S[m] = w.astype(np.float32)
        da[m] = 1.0
        mit.add(i)
    return S, da, mit


def pca_block(S_tr, da_tr, S_te, da_te, k=KOMPONENTEN):
    """PCA NUR aus train-Zeilen mit Merkmalen; Zeilen ohne Merkmale bleiben
    0, dazu die Indikatorspalte. Liefert (B_tr, B_te) mit k+1 Spalten."""
    t = da_tr > 0
    mu = S_tr[t].mean(0, keepdims=True)
    cov = np.cov((S_tr[t] - mu).T)
    w, V = np.linalg.eigh(cov)
    V = V[:, np.argsort(w)[::-1][:k]].astype(np.float32)

    def proj(S, da):
        P = ((S - mu) @ V) * da[:, None]
        return np.concatenate([P, da[:, None]], 1).astype(np.float32)

    return proj(S_tr, da_tr), proj(S_te, da_te)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()

    vm = _lade("o27vm_o28", HIER / "o27-vormessung.py")
    o27 = _lade("o27_o28", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_o28", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_o28", HIER / "label_herkunft.py")

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

    # Produktionszustand: test-Logo-Spalte halb (train bleibt Archiv, wie im Nightly)
    Xte_p, te_halb = o27.halbe_spalte(Xte, rec_te, u_te, 1, vm.HALB_CACHE)
    S_tr, da_tr, tr_mit = siglip_zeilen(rec_tr, u_tr, a.schritt, SIGLIP_CACHE)
    S_te, da_te, te_mit = siglip_zeilen(rec_te, u_te, 1, SIGLIP_CACHE)
    beides = te_halb & te_mit
    primaer_rec = np.array([i in beides and mensch(u) for i, u in enumerate(u_te)])
    neben_rec = np.array([i in beides for i in range(len(u_te))])
    print(f"train {len(u_tr)} Aufnahmen, davon {len(tr_mit)} mit SigLIP; test mit halber "
          f"Spalte UND SigLIP {int(neben_rec.sum())}, davon menschlich {int(primaer_rec.sum())}",
          flush=True)
    m_prim, m_neben = primaer_rec[rec_te], neben_rec[rec_te]

    B_tr, B_te = pca_block(S_tr, da_tr, S_te, da_te)
    del S_tr, S_te
    Xtr_k, Xte_k = o.standardisieren(Xtr, Xte_p)
    Xtr_v, Xte_v = o.standardisieren(np.concatenate([Xtr, B_tr], 1),
                                     np.concatenate([Xte_p, B_te], 1))
    wahr = (yte > 0).astype(np.int64)

    def werte(p):
        pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
        return o.f1(pred[m_prim], wahr[m_prim]), o.f1(pred[m_neben], wahr[m_neben])

    erg = {"kontrolle": [], "versuch": []}
    for seed in range(a.seeds):
        for arm, (A, B) in (("kontrolle", (Xtr_k, Xte_k)), ("versuch", (Xtr_v, Xte_v))):
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
