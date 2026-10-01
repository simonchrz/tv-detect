#!/usr/bin/env python3
"""O31 — Bindet ein Stabilitaets-Ziel den neuen Kopf an den alten, ohne Qualitaet zu kosten?

Registrierung: docs/o31-stabilitaet-preregistration.md.

Simuliert zwei Naechte auf denselben Zeilen wie O30 (Produktionszustand
MLP7: Backbone + Logo (test halb) + Audio + OCR + SigLIP-64 + da):
  Champion   trainiert OHNE die neuesten --neu-anteil Aufnahmen (eigener Seed)
  Kandidat   trainiert auf ALLEN train-Aufnahmen
Arme fuer den Kandidaten (gleiche Seeds, gleiche Architektur):
  K    harte Labels (heutiges Nightly)
  S30  auf Aufnahmen, die der Champion schon kannte: Ziel 0.7*y + 0.3*p_champ
  S50  dito mit 0.5; neue Aufnahmen behalten in allen Armen das harte Label
Primaer (wie O30): menschlich gelabelte test-Aufnahmen mit halber
Logo-Spalte UND SigLIP-Spur.
  kipp  Anteil der Testframes, deren geglaettete Entscheidung (p>0.5)
        sich zwischen Champion und Kandidat unterscheidet
  F1    gegen die Labels
--nur-kontrolle misst nur K (Rauschen VOR der Registrierung).
"""
import argparse
import importlib.util
import json
import re
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


def zeitpunkt(u):
    """Aufnahmebeginn aus der uuid (dvr-<slug>-<epoch>); Hash-uuids = 0 (alt)."""
    m = re.search(r"-(\d{9,10})$", u)
    return int(m.group(1)) if m else 0


def fit(Xtr, ziel, hart, seed, epochen, hidden, Xpred):
    """2-Klassen-MLP wie o20.fit_und_werte, aber mit WEICHEM Ziel.

    ziel: P(Werbung) je Zeile (hart = 0/1). Klassengewichte nach dem HARTEN
    Label (N/(2*n_k)) — dieselbe Regel wie O20/O30, damit der Stabilitaets-
    Arm nicht nebenbei eine andere Schieflagen-Korrektur bekommt (O20-Lehre).
    Liefert P(Werbung) fuer jede Matrix in Xpred.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.from_numpy(Xtr).to(dev)
    zt = torch.from_numpy(np.stack([1 - ziel, ziel], 1).astype(np.float32)).to(dev)
    zaehl = np.bincount(hart, minlength=2).astype(np.float64)
    wk = len(hart) / (2 * np.maximum(zaehl, 1))
    wz = torch.from_numpy(wk[hart].astype(np.float32)).to(dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 8192
    idx = np.arange(len(ziel))
    for _ in range(epochen):
        np.random.shuffle(idx)
        for s in range(0, len(idx), B):
            j = torch.from_numpy(idx[s:s + B]).to(dev)
            opt.zero_grad()
            logp = torch.log_softmax(net(Xt[j]), dim=1)
            # gewichtetes Mittel wie CrossEntropyLoss(weight=...)
            l = -(zt[j] * logp).sum(1)
            loss = (l * wz[j]).sum() / wz[j].sum()
            loss.backward()
            opt.step()
    out = []
    with torch.no_grad():
        for X in Xpred:
            p = []
            for s in range(0, len(X), 65536):
                p.append(torch.softmax(net(torch.from_numpy(X[s:s + 65536]).to(dev)),
                                       dim=1)[:, 1].cpu().numpy())
            out.append(np.concatenate(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--neu-anteil", type=float, default=0.10)
    ap.add_argument("--nur-kontrolle", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    o28 = _lade("o28_o31", HIER / "o28-siglip.py")
    o26 = _lade("o26_o31", HIER / "o26-ocr-spalte.py")
    vm = _lade("o27vm_o31", HIER / "o27-vormessung.py")
    o27 = _lade("o27_o31", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_o31", HIER / "o20-klassen-split.py")
    lh = vm._lade("lh_o31", HIER / "label_herkunft.py")

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
    m_prim = prim[rec_te]
    Btr, Bte = o28.pca_block(S_tr, da_tr, S_te, da_te, k=64)
    del S_tr, S_te
    A, B = o.standardisieren(np.concatenate([Xtr, Otr, Btr], 1),
                             np.concatenate([Xte_p, Ote, Bte], 1))
    del Xtr, Xte, Xte_p
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # "Neu" = die juengsten Aufnahmen nach Aufnahmebeginn; der Champion
    # kennt sie nicht.
    zp = np.array([zeitpunkt(u) for u in u_tr])
    grenze = np.quantile(zp[zp > 0], 1 - a.neu_anteil)
    neu_rec = zp >= grenze
    neu = neu_rec[rec_tr]
    hart = (ytr > 0).astype(np.int64)
    print(f"train {len(u_tr)} Aufnahmen ({int(neu_rec.sum())} neu), {A.shape[1]} Spalten; "
          f"primaer {int(prim.sum())} test-Aufnahmen", flush=True)

    wahr = (yte > 0).astype(np.int64)
    arme = {"K": 0.0} if a.nur_kontrolle else {"K": 0.0, "S30": 0.3, "S50": 0.5}
    erg = {k: {"kipp": [], "f1": []} for k in arme}
    erg["champion_f1"] = []
    for seed in range(a.seeds):
        alt = ~neu
        p_ch_tr, p_ch_te = fit(A[alt], hart[alt].astype(np.float32), hart[alt],
                               1000 + seed, a.epochen, a.hidden, [A, B])
        ent_ch = o.glaetten(p_ch_te, rec_te) > 0.5
        erg["champion_f1"].append(o.f1(ent_ch[m_prim].astype(np.int64), wahr[m_prim]))
        for name, lam in arme.items():
            ziel = hart.astype(np.float32)
            if lam > 0:
                ziel = np.where(alt, (1 - lam) * ziel + lam * p_ch_tr, ziel).astype(np.float32)
            (p_te,) = fit(A, ziel, hart, seed, a.epochen, a.hidden, [B])
            ent = o.glaetten(p_te, rec_te) > 0.5
            kipp = float((ent[m_prim] != ent_ch[m_prim]).mean())
            f1 = o.f1(ent[m_prim].astype(np.int64), wahr[m_prim])
            erg[name]["kipp"].append(kipp)
            erg[name]["f1"].append(f1)
            print(f"  Seed {seed}  {name:4s} kipp {kipp:.4f}  F1 {f1:.4f}  "
                  f"(Champion F1 {erg['champion_f1'][-1]:.4f})", flush=True)
    print()
    K = erg["K"]
    for name in arme:
        kv, fv = np.array(erg[name]["kipp"]), np.array(erg[name]["f1"])
        rel = kv / np.array(K["kipp"])
        df = fv - np.array(K["f1"])
        print(f"{name}: kipp Median {np.median(kv):.4f} sd {kv.std(ddof=1):.4f}  "
              f"Verhaeltnis zu K {np.median(rel):.3f} (alle < 1: {bool((rel < 1).all())})  "
              f"F1 Median {np.median(fv):.4f} sd {fv.std(ddof=1):.4f}  "
              f"ΔF1 {np.median(df):+.4f} (min {df.min():+.4f})")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
