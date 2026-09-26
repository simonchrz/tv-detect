#!/usr/bin/env python3
"""O29 — Traegt ein billiger Kontext-Traeger den O28-Gewinn?

Registrierung: docs/o29-kontext-traeger-preregistration.md.

O28 zeigte: rund zwei Drittel des SigLIP-Gewinns ueberleben, wenn man die
Zeilen je Aufnahme zeitlich mischt — es ist Kontext auf Aufnahme-Ebene.
Arme (gleiche Zeilen, Seeds, Architektur, Standardisierung aus train):
  K  PRODUKTIONSZUSTAND: Archiv-Merkmale, test-Logo-Spalte halb, dazu die
     drei OCR-Spalten wie MLP6 (O28 trainierte noch ohne sie)
  M  K + SigLIP-MITTEL je Aufnahme (PCA-64 ueber train-Aufnahmen, je
     Aufnahme konstant) + siglip_da
  T  K + Sendungs-Kennung: one-hot der Titel mit >= 3 train-Aufnahmen,
     sonst "sonstige"
  S  K + SigLIP sekundengenau (O28-Versuch), als Referenz
Primaer: F1 auf menschlich gelabelten test-Aufnahmen mit halber Logo-Spalte
UND SigLIP-Merkmalen. --nur-kontrolle misst nur K (Rauschen VOR der
Registrierung).
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
ARCH = Path.home() / ".cache/tvd-train-archive"
MIN_FOLGEN = 3


def _lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def titel(u):
    try:
        return json.loads(str(np.load(ARCH / f"{u}.npz", allow_pickle=True)["meta"])).get("title") or ""
    except Exception:
        return ""


def titel_block(rec_tr, u_tr, rec_te, u_te, titel_von=titel):
    """One-hot der Titel mit >= MIN_FOLGEN train-Aufnahmen + Spalte sonstige.
    Die Titelliste kommt NUR aus train."""
    t_tr = [titel_von(u) for u in u_tr]
    t_te = [titel_von(u) for u in u_te]
    haeufig = sorted(t for t, n in Counter(t_tr).items() if t and n >= MIN_FOLGEN)
    idx = {t: j for j, t in enumerate(haeufig)}

    def block(rec, tt):
        B = np.zeros((len(rec), len(haeufig) + 1), np.float32)
        spalte = np.array([idx.get(t, len(haeufig)) for t in tt])
        B[np.arange(len(rec)), spalte[rec]] = 1.0
        return B

    return block(rec_tr, t_tr), block(rec_te, t_te), haeufig


def mittel_block(rec_tr, u_tr, rec_te, u_te, cache, k=64):
    """SigLIP-Mittel je Aufnahme (ueber ALLE Sekunden der Datei), PCA nur aus
    train-Aufnahmen, auf jede Zeile der Aufnahme gelegt; + siglip_da."""
    def mittel(uu):
        M = np.zeros((len(uu), 768), np.float32)
        da = np.zeros(len(uu), np.float32)
        for i, u in enumerate(uu):
            f = cache / f"{u}.npy"
            if f.is_file():
                M[i] = np.load(f).astype(np.float32).mean(0)
                da[i] = 1.0
        return M, da

    M_tr, da_tr = mittel(u_tr)
    M_te, da_te = mittel(u_te)
    t = da_tr > 0
    mu = M_tr[t].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M_tr[t] - mu, full_matrices=False)
    V = Vt[:k].T

    def proj(M, da, rec):
        P = ((M - mu) @ V) * da[:, None]
        return np.concatenate([P, da[:, None]], 1).astype(np.float32)[rec]

    return proj(M_tr, da_tr, rec_tr), proj(M_te, da_te, rec_te)


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

    o28 = _lade("o28_o29", HIER / "o28-siglip.py")
    o26 = _lade("o26_o29", HIER / "o26-ocr-spalte.py")
    vm = _lade("o27vm_o29", HIER / "o27-vormessung.py")
    o27 = _lade("o27_o29", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_o29", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_o29", HIER / "label_herkunft.py")

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
    # Produktionszustand = MLP6: die drei OCR-Spalten gehoeren in JEDEN Arm
    Ktr = np.concatenate([Xtr, o26.spalten_fuer(rec_tr, u_tr, a.schritt)], 1)
    Kte = np.concatenate([Xte_p, o26.spalten_fuer(rec_te, u_te, 1)], 1)
    te_mit = {i for i, u in enumerate(u_te) if (o28.SIGLIP_CACHE / f"{u}.npy").is_file()}
    prim = np.array([i in (te_halb & te_mit) and mensch(u) for i, u in enumerate(u_te)])
    print(f"primaer: {int(prim.sum())} test-Aufnahmen; OCR-Spur train "
          f"{Ktr[:, -1].mean():.2f}, test {Kte[:, -1].mean():.2f} der Zeilen", flush=True)
    m_prim = prim[rec_te]

    arme = {"K": (Ktr, Kte)}
    if not a.nur_kontrolle:
        Mtr, Mte = mittel_block(rec_tr, u_tr, rec_te, u_te, o28.SIGLIP_CACHE)
        Ttr, Tte, haeufig = titel_block(rec_tr, u_tr, rec_te, u_te)
        print(f"Titel-Spalten: {len(haeufig)} + sonstige; test-Zeilen unter sonstige "
              f"{Tte[:, -1].mean():.2f}", flush=True)
        S_tr, da_tr, _ = o28.siglip_zeilen(rec_tr, u_tr, a.schritt, o28.SIGLIP_CACHE)
        S_te, da_te, _ = o28.siglip_zeilen(rec_te, u_te, 1, o28.SIGLIP_CACHE)
        Btr, Bte = o28.pca_block(S_tr, da_tr, S_te, da_te)
        del S_tr, S_te
        arme.update({
            "M": (np.concatenate([Ktr, Mtr], 1), np.concatenate([Kte, Mte], 1)),
            "T": (np.concatenate([Ktr, Ttr], 1), np.concatenate([Kte, Tte], 1)),
            "S": (np.concatenate([Ktr, Btr], 1), np.concatenate([Kte, Bte], 1)),
        })
    wahr = (yte > 0).astype(np.int64)
    erg = {k: [] for k in arme}
    for k in list(arme):
        A, B = o.standardisieren(*arme.pop(k))
        for seed in range(a.seeds):
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
            erg[k].append(o.f1(pred[m_prim], wahr[m_prim]))
            print(f"  {k}  Seed {seed}  F1 {erg[k][-1]:.4f}", flush=True)
        del A, B
    print()
    for k, v in erg.items():
        v = np.array(v)
        d = v - np.array(erg["K"])
        print(f"{k}: Median {np.median(v):.4f}  sd {v.std(ddof=1):.4f}  "
              f"Delta zu K {np.median(d):+.4f}  positiv {int((d > 0).sum())}/{len(d)}")
    if "S" in erg:
        for ref in ("M", "T"):
            d = np.array(erg["S"]) - np.array(erg[ref])
            print(f"S gegen {ref}: Median {np.median(d):+.4f}  positiv {int((d > 0).sum())}/{len(d)}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
