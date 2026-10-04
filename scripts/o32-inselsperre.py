#!/usr/bin/env python3
"""O32 — Sendungsinseln im Werbeblock verbieten (docs/o32-inselsperre-preregistration.md).

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
ARME = {"K": 0, "I60": 60, "I90": 90}


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
    ap.add_argument("--nur-k", action="store_true", dest="nur_k")
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
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)
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

    arme = {"K": 0} if a.nur_k else ARME

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

    erg = {"seeds": a.seeds, "n_test": len(test), "n_spur": n_spur,
           "arme": {k: {"kipp": [], "iou": [], "iou_champion": []} for k in arme},
           "je_aufnahme": {k: [] for k in arme}}
    for seed in range(a.seeds):
        (p_ch,) = o31.fit(A[alt], hart[alt].astype(np.float32), hart[alt],
                          1000 + seed, a.epochen, a.hidden, [B])
        (p_ka,) = o31.fit(A, hart.astype(np.float32), hart,
                          seed, a.epochen, a.hidden, [B])
        for name, w in arme.items():
            ich, ika = bewerten(p_ch, w), bewerten(p_ka, w)
            gem = sorted(set(ich) & set(ika))
            kipp = int(sum(abs(ika[u] - ich[u]) > 0.1 for u in gem))
            r = erg["arme"][name]
            r["kipp"].append(kipp)
            r["iou"].append(float(np.mean([ika[u] for u in gem])))
            r["iou_champion"].append(float(np.mean([ich[u] for u in gem])))
            erg["je_aufnahme"][name].append({"champion": ich, "kandidat": ika})
            print(f"  Seed {seed}  {name:4s} kipp {kipp:3d}  IoU Kandidat "
                  f"{r['iou'][-1]:.4f}  Champion {r['iou_champion'][-1]:.4f}  "
                  f"(n={len(gem)})", flush=True)

    print()
    K = erg["arme"]["K"]
    for name in arme:
        r = erg["arme"][name]
        kv, iv = np.array(r["kipp"]), np.array(r["iou"])
        zeile = (f"{name}: kipp {kv.tolist()} (Summe {kv.sum()})  "
                 f"IoU Median {np.median(iv):.4f}")
        if name != "K":
            kk = np.array(K["kipp"])
            d = iv - np.array(K["iou"])
            zeile += (f"  | kipp_I/kipp_K (Summe) {kv.sum() / max(kk.sum(), 1):.3f}, "
                      f"kipp_I <= kipp_K in {int((kv <= kk).sum())}/{len(kv)}  "
                      f"ΔIoU Median {np.median(d):+.4f} (min {d.min():+.4f})")
            # R3: Einzelverluste des Kandidaten gegen K, ueber alle Seeds
            w = ARME[name]
            verl, ausgenommen = [], []
            for s in range(a.seeds):
                ik = erg["je_aufnahme"]["K"][s]["kandidat"]
                ia = erg["je_aufnahme"][name][s]["kandidat"]
                for u in set(ik) & set(ia):
                    if ia[u] - ik[u] < -0.10:
                        gt = next(t[3] for t in test if t[0] == u)
                        ziel = (ausgenommen if any(g < w for g in innere_luecken(gt))
                                else verl)
                        ziel.append((s, u, round(ia[u] - ik[u], 3)))
            r["verluste"], r["verluste_ausgenommen"] = verl, ausgenommen
            zeile += f"  | Verluste >0.10: {len(verl)} (+{len(ausgenommen)} Label-Luecke < {w}s)"
        print(zeile)
        for v in r.get("verluste", []) + r.get("verluste_ausgenommen", []):
            print(f"      Seed {v[0]}  {v[1]}  {v[2]:+.3f}")
    out = ARCH / ("o32-vormessung.json" if a.nur_k else "o32-ergebnis.json")
    out.write_text(json.dumps(erg, indent=1))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
