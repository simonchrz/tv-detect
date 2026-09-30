#!/usr/bin/env python3
"""O30 — Ersetzt SigLIP den alten Backbone, und tragen mehr Komponenten?

Registrierung: docs/o30-backbone-ersatz-preregistration.md.

Arme (gleiche Zeilen, Seeds, Architektur, Standardisierung aus train):
  K  PRODUKTIONSZUSTAND (MLP7): Backbone 1280 + Logo (test halb) + Audio
     + 3 OCR-Spalten + SigLIP PCA-64 + siglip_da
  A  OHNE Backbone: Logo + Audio + OCR + SigLIP-64 + da
  B  K mit SigLIP-128
  C  K mit SigLIP-256
  D  OHNE Backbone, SigLIP-256
Primaer: F1 auf menschlich gelabelten test-Aufnahmen mit halber Logo-Spalte
UND SigLIP-Spur. --nur-kontrolle misst nur K (Rauschen VOR der
Registrierung).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent


def _lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--nur-kontrolle", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    o28 = _lade("o28_o30", HIER / "o28-siglip.py")
    o26 = _lade("o26_o30", HIER / "o26-ocr-spalte.py")
    vm = _lade("o27vm_o30", HIER / "o27-vormessung.py")
    o27 = _lade("o27_o30", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_o30", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_o30", HIER / "label_herkunft.py")

    def mensch(u):
        try:
            return lh.mensch_aus_markern(json.loads(
                (vm.SNAPSHOT / f"_rec_{u}" / "ads_user.json").read_text())) is True
        except Exception:
            return False

    print("Lade …", flush=True)
    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)
    Xte_p, te_halb = o27.halbe_spalte(Xte, rec_te, u_te, 1, vm.HALB_CACHE)
    Otr = o26.spalten_fuer(rec_tr, u_tr, a.schritt)
    Ote = o26.spalten_fuer(rec_te, u_te, 1)
    S_tr, da_tr, tr_mit = o28.siglip_zeilen(rec_tr, u_tr, a.schritt, o28.SIGLIP_CACHE)
    S_te, da_te, te_mit = o28.siglip_zeilen(rec_te, u_te, 1, o28.SIGLIP_CACHE)
    prim = np.array([i in (te_halb & te_mit) and mensch(u) for i, u in enumerate(u_te)])
    print(f"train {len(u_tr)} Aufnahmen, davon {len(tr_mit)} mit SigLIP; primaer {int(prim.sum())} "
          f"test-Aufnahmen", flush=True)
    m_prim = prim[rec_te]
    # Logo + Audio = die beiden Spalten hinter dem Backbone (Index 1280, 1281)
    LAtr, LAte = Xtr[:, 1280:1282], Xte_p[:, 1280:1282]

    bloecke = {64: o28.pca_block(S_tr, da_tr, S_te, da_te, k=64)}
    if not a.nur_kontrolle:
        for k in (128, 256):
            bloecke[k] = o28.pca_block(S_tr, da_tr, S_te, da_te, k=k)
    del S_tr, S_te

    def arm(mit_backbone, k):
        Btr, Bte = bloecke[k]
        basis_tr = Xtr if mit_backbone else LAtr
        basis_te = Xte_p if mit_backbone else LAte
        return (np.concatenate([basis_tr, Otr, Btr], 1),
                np.concatenate([basis_te, Ote, Bte], 1))

    arme = {"K": (True, 64)}
    if not a.nur_kontrolle:
        arme.update({"A": (False, 64), "B": (True, 128), "C": (True, 256), "D": (False, 256)})
    wahr = (yte > 0).astype(np.int64)
    erg = {k: [] for k in arme}
    for name, (bb, k) in arme.items():
        A, B = o.standardisieren(*arm(bb, k))
        print(f"  Arm {name}: {A.shape[1]} Spalten", flush=True)
        for seed in range(a.seeds):
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
            erg[name].append(o.f1(pred[m_prim], wahr[m_prim]))
            print(f"  {name}  Seed {seed}  F1 {erg[name][-1]:.4f}", flush=True)
        del A, B
    print()
    for name, v in erg.items():
        v = np.array(v)
        d = v - np.array(erg["K"])
        print(f"{name}: Median {np.median(v):.4f}  sd {v.std(ddof=1):.4f}  "
              f"Delta zu K {np.median(d):+.4f}  positiv {int((d > 0).sum())}/{len(d)}  "
              f"unter -0.005: {int((d < -0.005).sum())}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
