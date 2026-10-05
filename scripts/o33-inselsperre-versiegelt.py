#!/usr/bin/env python3
"""O33 — Inselsperre (90 s) auf dem versiegelten Satz (docs/o33-inselsperre-versiegelt-preregistration.md).

Abgeleitet von o32-inselsperre.py: EIN Kopf je Seed (alle train-Aufnahmen),
ausgewertet auf dem Eimer "versiegelt"; primaer nur nicht-maschinelle Labels
(massstab-audit). Unten der O32-Text zur Bauart.

O32:

Zwei simulierte Naechte je Seed, gebaut wie O31 (gleiche Zeilen, Spalten,
Architektur; Produktionszustand MLP7):
  Champion  trainiert OHNE die juengsten --neu-anteil train-Aufnahmen (Seed 1000+s)
  Kandidat  trainiert auf ALLEN train-Aufnahmen (Seed s)
Beide werden auf ALLEN test-Aufnahmen durch den Gate-Dekoder geschickt
(train-head._replay_blocks; ohne Decode-Spur _replay_ohne_spur, wie das Gate
seit ee713ae) und mit block_iou gegen die Archiv-Labels gemessen. Einziger
Unterschied je Arm: --hsmm-inner-show-min.

  kipp  Aufnahmen mit |IoU_Kandidat - IoU_Champion| > 0.1 (= Kippzahl des Gates)
  IoU   Mittel des Kandidaten ueber die test-Aufnahmen

--nur-k  Vormessung (nur Arm K), VOR der Registrierung.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
ARCH = Path.home() / ".cache" / "tvd-train-archive"
ARME = {"K": 0, "I90": 90}


def _lade(name, pfad):
    argv, sys.argv = sys.argv, [name]
    try:
        spec = importlib.util.spec_from_file_location(name, pfad)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def innere_luecken(ads):
    a = sorted((float(x), float(y)) for x, y in ads)
    return [b0 - a1 for (_, a1), (b0, _) in zip(a, a[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="tv-detect mit --hsmm-inner-show-min")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--neu-anteil", type=float, default=0.10)
    a = ap.parse_args()

    o31 = _lade("o31_o32", HIER / "o31-stabilitaet.py")
    o28 = _lade("o28_o32", HIER / "o28-siglip.py")
    o26 = _lade("o26_o32", HIER / "o26-ocr-spalte.py")
    vm = _lade("o27vm_o32", HIER / "o27-vormessung.py")
    o27 = _lade("o27_o32", HIER / "o27-logo-halb.py")
    o = vm._lade("o20_o32", HIER / "o20-klassen-split.py")
    th = _lade("th_o32", HIER / "train-head.py")
    th.TVD_BIN = Path(a.bin)

    # ── Daten wie O31 ────────────────────────────────────────────────
    print("Lade …", flush=True)
    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    Xte, yte, rec_te, u_te = o.lade("versiegelt", None, 1, mit_uuids=True)
    Xte_p, _ = o27.halbe_spalte(Xte, rec_te, u_te, 1, vm.HALB_CACHE)
    Otr = o26.spalten_fuer(rec_tr, u_tr, a.schritt)
    Ote = o26.spalten_fuer(rec_te, u_te, 1)
    S_tr, da_tr, _ = o28.siglip_zeilen(rec_tr, u_tr, a.schritt, o28.SIGLIP_CACHE)
    S_te, da_te, _ = o28.siglip_zeilen(rec_te, u_te, 1, o28.SIGLIP_CACHE)
    Btr, Bte = o28.pca_block(S_tr, da_tr, S_te, da_te, k=64)
    del S_tr, S_te
    A, B = o.standardisieren(np.concatenate([Xtr, Otr, Btr], 1),
                             np.concatenate([Xte_p, Ote, Bte], 1))
    del Xtr, Xte, Xte_p
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    zp = np.array([o31.zeitpunkt(u) for u in u_tr])
    grenze = np.quantile(zp[zp > 0], 1 - a.neu_anteil)
    alt = ~(zp >= grenze)[rec_tr]
    hart = (ytr > 0).astype(np.int64)

    # ── Testaufnahmen: Labels aus dem Archiv (wie das Gate), Spur ja/nein ──
    test = []
    for i, u in enumerate(u_te):
        npz = ARCH / f"{u}.npz"
        if not npz.exists():
            continue
        m = json.loads(str(np.load(npz, allow_pickle=True)["meta"]))
        zeilen = np.flatnonzero(rec_te == i)
        cp = th._signals_cache_path(u)
        if cp is not None:
            fps, fc = th._signals_header(cp)
            if abs(fc / fps - len(zeilen)) > 10:
                cp = None  # das Gate wuerde die Spur verwerfen; hier nur nicht nutzen
        gt = [(float(x), float(y)) for x, y in (m.get("ads") or [])]
        test.append((u, zeilen, cp, gt))
    n_spur = sum(t[2] is not None for t in test)
    print(f"train {len(u_tr)} Aufnahmen, {A.shape[1]} Spalten; test {len(test)} "
          f"Aufnahmen ({n_spur} mit Decode-Spur)", flush=True)

    def bewerten(p_te, w):
        th.EVAL_DECODER = ["--decoder", "hsmm", "--hsmm-dur-w", "15"] + (
            ["--hsmm-inner-show-min", str(w)] if w else [])
        out = {}
        for u, zeilen, cp, gt in test:
            p = p_te[zeilen]
            b = (th._replay_blocks(cp, p, 1.0, u) if cp is not None
                 else th._replay_ohne_spur(p, 1.0, u))
            if b is not None:
                out[u] = th.block_iou(b, gt)
        return out

    ma = _lade("massstab_o33", HIER / "massstab-audit.py")
    meta = ma.archiv_meta()
    herk = {t[0]: ma.herkunft(t[0], meta)[0] for t in test}
    primaer = {u for u, h in herk.items() if h != "maschine"}
    print(f"Herkunft: {sum(h == 'mensch' for h in herk.values())} Mensch, "
          f"{sum(h == 'maschine' for h in herk.values())} Maschine, "
          f"{sum(h == 'unbekannt' for h in herk.values())} unbekannt; "
          f"primaer {len(primaer)}", flush=True)

    erg = {"seeds": a.seeds, "n": len(test), "n_primaer": len(primaer), "n_spur": n_spur,
           "herkunft": herk, "je_seed": []}
    for seed in range(a.seeds):
        (p_ka,) = o31.fit(A, hart.astype(np.float32), hart, seed, a.epochen, a.hidden, [B])
        ik, ii = bewerten(p_ka, 0), bewerten(p_ka, 90)
        gem = sorted(set(ik) & set(ii))
        gp = [u for u in gem if u in primaer]
        z = {"K": ik, "I90": ii,
             "d_primaer": float(np.mean([ii[u] - ik[u] for u in gp])),
             "d_alle": float(np.mean([ii[u] - ik[u] for u in gem])),
             "iou_K_primaer": float(np.mean([ik[u] for u in gp]))}
        erg["je_seed"].append(z)
        print(f"  Seed {seed}  primaer n={len(gp)}  IoU K {z['iou_K_primaer']:.4f}  "
              f"Δ primaer {z['d_primaer']:+.4f}  Δ alle {z['d_alle']:+.4f}", flush=True)

    dp = np.array([z["d_primaer"] for z in erg["je_seed"]])
    da = np.array([z["d_alle"] for z in erg["je_seed"]])
    verl, ausg = [], []
    for s_, z in enumerate(erg["je_seed"]):
        for u in primaer:
            if u in z["K"] and u in z["I90"] and z["I90"][u] - z["K"][u] < -0.10:
                gt = next(t[3] for t in test if t[0] == u)
                (ausg if any(g < 90 for g in innere_luecken(gt)) else verl).append(
                    (s_, u, round(z["I90"][u] - z["K"][u], 3)))
    r1 = bool(np.median(dp) >= 0.002 and (dp > 0).sum() >= 4)
    r2 = len(verl) <= 2
    erg.update({"median_d_primaer": float(np.median(dp)), "seeds_positiv": int((dp > 0).sum()),
                "median_d_alle": float(np.median(da)), "verluste": verl,
                "verluste_ausgenommen": ausg, "R1": r1, "R2": r2,
                "urteil": "ERFUELLT" if r1 and r2 else "VERFEHLT"})
    print()
    print(f"primaer: Δ je Seed {[round(x, 4) for x in dp]}  Median {np.median(dp):+.4f}  "
          f"positiv {int((dp > 0).sum())}/{len(dp)}  → R1 {'haelt' if r1 else 'verfehlt'}")
    print(f"Verluste > 0.10: {len(verl)} (+{len(ausg)} Label-Luecke < 90 s) → R2 "
          f"{'haelt' if r2 else 'verfehlt'}")
    for v in verl + ausg:
        print(f"      Seed {v[0]}  {v[1]}  {v[2]:+.3f}")
    print(f"sekundaer (alle {len(test)}): Median Δ {np.median(da):+.4f}")
    print(f"URTEIL O33: {erg['urteil']}")
    out = ARCH / "o33-ergebnis.json"
    out.write_text(json.dumps(erg, indent=1))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
