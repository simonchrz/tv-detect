#!/usr/bin/env python3
"""O28-Gegenprobe (Diagnose, NACH dem Urteil, aendert es nicht).

Woher kommt der Gewinn des SigLIP-Blocks? Zwei Placebo-Arme neben
kontrolle/versuch, gleiche Seeds:
  nur_indikator  nur siglip_da, keine Komponenten — traegt schon die Frage
                 "welche Aufnahme hat eine Quelle"?
  zeit_gemischt  SigLIP-Zeilen INNERHALB jeder Aufnahme permutiert (train
                 und test): Sendungs-/Kanal-Identitaet bleibt, der Bezug
                 zur Sekunde ist weg. Haelt der Gewinn, erkennt SigLIP die
                 Sendung, nicht die Werbung.
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


def zeit_mischen(S, rec, rng):
    S = S.copy()
    for i in np.unique(rec):
        idx = np.flatnonzero(rec == i)
        S[idx] = S[rng.permutation(idx)]
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--mit-ocr", action="store_true",
                    help="Kontrolle = K aus O29 (Produktionszustand inkl. 3 OCR-Spalten)")
    ap.add_argument("--json")
    a = ap.parse_args()

    o28 = _lade("o28_gp", HIER / "o28-siglip.py")
    vm = _lade("o27vm_gp", HIER / "o27-vormessung.py")
    o27 = _lade("o27_gp", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_gp", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_gp", HIER / "label_herkunft.py")

    def mensch(u):
        try:
            return lh.mensch_aus_markern(json.loads(
                (vm.SNAPSHOT / f"_rec_{u}" / "ads_user.json").read_text())) is True
        except Exception:
            return False

    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)
    Xte_p, te_halb = o27.halbe_spalte(Xte, rec_te, u_te, 1, vm.HALB_CACHE)
    if a.mit_ocr:
        o26 = _lade("o26_gp", HIER / "o26-ocr-spalte.py")
        Xtr = np.concatenate([Xtr, o26.spalten_fuer(rec_tr, u_tr, a.schritt)], 1)
        Xte_p = np.concatenate([Xte_p, o26.spalten_fuer(rec_te, u_te, 1)], 1)
    S_tr, da_tr, _ = o28.siglip_zeilen(rec_tr, u_tr, a.schritt, o28.SIGLIP_CACHE)
    S_te, da_te, te_mit = o28.siglip_zeilen(rec_te, u_te, 1, o28.SIGLIP_CACHE)
    prim = np.array([i in (te_halb & te_mit) and mensch(u) for i, u in enumerate(u_te)])[rec_te]
    print(f"primaer: {int(prim.sum())} Sekunden", flush=True)

    rng = np.random.default_rng(0)
    B_tr, B_te = o28.pca_block(S_tr, da_tr, S_te, da_te)
    G_tr, G_te = o28.pca_block(zeit_mischen(S_tr, rec_tr, rng), da_tr,
                               zeit_mischen(S_te, rec_te, rng), da_te)
    del S_tr, S_te
    arme = {
        "kontrolle": (Xtr, Xte_p),
        "versuch": (np.concatenate([Xtr, B_tr], 1), np.concatenate([Xte_p, B_te], 1)),
        "nur_indikator": (np.concatenate([Xtr, da_tr[:, None]], 1),
                          np.concatenate([Xte_p, da_te[:, None]], 1)),
        "zeit_gemischt": (np.concatenate([Xtr, G_tr], 1), np.concatenate([Xte_p, G_te], 1)),
    }
    wahr = (yte > 0).astype(np.int64)
    erg = {k: [] for k in arme}
    for k, (A, B) in arme.items():
        A, B = o.standardisieren(A, B)
        for seed in range(a.seeds):
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
            erg[k].append(o.f1(pred[prim], wahr[prim]))
            print(f"  {k:<14} Seed {seed}  F1 {erg[k][-1]:.4f}", flush=True)
    print()
    for k, v in erg.items():
        d = np.array(v) - np.array(erg["kontrolle"])
        print(f"{k:<14} Median {np.median(v):.4f}  Delta {np.median(d):+.4f}  positiv {int((d > 0).sum())}/{len(d)}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
