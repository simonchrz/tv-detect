#!/usr/bin/env python3
"""Sitzen die Fehler des Kopfes dort, wo das Backbone verwechselt?

DIE FRAGE
---------
Idee 4 waere, das Backbone anzutrainieren statt nur den Kopf darauf zu
setzen. Das kostet 72 % des Korpus, weil die Merkmale neu extrahiert
werden muessten und 737 von 1028 Aufnahmen keine Quelle mehr haben. Bevor
irgendjemand das bezahlt, gehoert geprueft, ob die Repraesentation
ueberhaupt der Engpass ist.

Indiz gibt es: die Wiederholungssuche zeigt, dass das Backbone Simpsons
mit Futurama und Batman v Superman mit Predator verwechselt. Ein Indiz
ist aber kein Beweis, dass es DESHALB Werbung schlechter erkennt.

DER TEST
--------
Fuer jede Sekunde mit menschlichem Label werden die k aehnlichsten
Sekunden in ANDEREN Aufnahmen gesucht. Der Anteil dieser Nachbarn mit
dem GEGENTEILIGEN Label ist die Verwechslungsrate der Sekunde: hoch
heisst, das Backbone legt Werbung und Sendung an dieselbe Stelle.

Dann wird gefragt, ob der Kopf genau dort irrt.

  * Steigt seine Fehlerrate mit der Verwechslungsrate steil an, ist die
    Repraesentation der Engpass und Idee 4 ist begruendet.
  * Bleibt sie flach, hat der Kopf mehr gelernt als ein
    Naechster-Nachbar-Klassifikator, und ein neues Backbone kauft nichts.

WAS BEWUSST AUSGESCHLOSSEN WIRD
-------------------------------
Sekunden nahe einer Blockgrenze. Dort irrt der Kopf aus einem anderen
Grund (die Grenze liegt eine Sekunde daneben), und Weg 2 hat am
2026-09-07 gezeigt, dass seine Kanten so gut sind wie die des Menschen.
Gefragt ist das INNERE der Bloecke, wo Erkennen zaehlt, nicht Verorten.

Nur menschlich gelabelte Aufnahmen (label_herkunft.py). Ein maschinelles
Label ist die frueherer Ausgabe desselben Kopfes; dagegen zu messen
hiesse zu fragen, ob der Kopf mit sich selbst uebereinstimmt.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

FEATS = Path.home() / ".cache" / "tvd-features"
DUMPS = Path.home() / ".cache" / "tv-detect-daemon" / "emit-signals"
SNAPSHOT = Path("/tmp/tv-train-snapshot")
_HIER = Path(__file__).resolve().parent


def _lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, _HIER / datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def menschlabels():
    lh = _lade("lh", "label_herkunft.py")
    aus = {}
    for d in sorted(SNAPSHOT.glob("_rec_*")):
        f = d / "ads_user.json"
        if not f.is_file():
            continue
        try:
            roh = json.loads(f.read_text())
        except Exception:
            continue
        if lh.mensch_aus_markern(roh) is not True:
            continue
        bl = [(float(a), float(b)) for a, b in (roh.get("ads") or [])
              if float(b) > float(a)]
        if bl:
            aus[d.name[5:]] = sorted(bl)
    return aus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=20, help="Nachbarn je Sekunde")
    ap.add_argument("--rand", type=int, default=15,
                    help="Sekunden um jede Blockgrenze, die ausgelassen werden")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--json")
    ap.add_argument("--npz", help="Arrays hierhin, damit Nachfragen nichts neu rechnen")
    a = ap.parse_args()

    import torch
    ub = menschlabels()
    dateien = {}
    for p in sorted(FEATS.glob("*.npy")):
        dateien[p.name.rsplit("-", 4)[0]] = p
    uuids = [u for u in sorted(ub) if u in dateien and (DUMPS / f"{u}.json").is_file()]
    print(f"{len(ub)} menschlich gelabelte Aufnahmen, davon {len(uuids)} "
          f"mit Merkmalen UND Signal-Dump")

    # --- Merkmale: zentriert, normiert, auf DIM projiziert -------------
    def roh(u):
        arr = np.load(dateien[u], mmap_mode="r")
        return np.array(arr[:, :1280], dtype=np.float32, copy=True)

    acc = np.zeros(1280, np.float64)
    n = 0
    for u in uuids:
        X = roh(u)
        acc += X.sum(0)
        n += X.shape[0]
    mu = (acc / n).astype(np.float32)
    stich = np.vstack([(roh(u) - mu)[::23] for u in uuids])
    stich /= np.linalg.norm(stich, axis=1, keepdims=True) + 1e-6
    _, _, Vt = np.linalg.svd(stich - stich.mean(0), full_matrices=False)
    P = Vt[:a.dim].astype(np.float32).T

    X, rec, sek = [], [], []
    for i, u in enumerate(uuids):
        F = roh(u) - mu
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        F = F @ P
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        X.append(F.astype(np.float32))
        rec.append(np.full(F.shape[0], i, np.int32))
        sek.append(np.arange(F.shape[0], dtype=np.int32))
    X = np.vstack(X)
    rec = np.concatenate(rec)
    sek = np.concatenate(sek)
    print(f"{X.shape[0]} Sekunden im Index")

    # --- Wahrheit und Kopf-Ausgabe je Sekunde --------------------------
    label = np.zeros(X.shape[0], np.int8)
    randnah = np.zeros(X.shape[0], bool)
    kopf = np.full(X.shape[0], np.nan, np.float32)
    pos = 0
    for i, u in enumerate(uuids):
        m = rec == i
        nsec = int(m.sum())
        bl = ub[u]
        lab = np.zeros(nsec, np.int8)
        nah = np.zeros(nsec, bool)
        for (s, e) in bl:
            lab[max(0, int(s)):min(nsec, int(e))] = 1
            for g in (s, e):
                lo, hi = max(0, int(g) - a.rand), min(nsec, int(g) + a.rand)
                nah[lo:hi] = True
        label[m] = lab
        randnah[m] = nah
        try:
            d = json.loads((DUMPS / f"{u}.json").read_text())
            nn = np.asarray(d.get("nn_confs") or [], np.float32)
            fps = float(d.get("fps") or 25)
            if len(nn) > 100:
                ende = min(nsec, int(len(nn) / fps))
                idx = np.flatnonzero(m)
                for t in range(ende):
                    kopf[idx[t]] = nn[int(t * fps):int((t + 1) * fps)].mean()
        except Exception:
            pass
        pos += nsec

    gilt = (~randnah) & np.isfinite(kopf)
    print(f"{int(gilt.sum())} Sekunden im Blockinneren mit Kopf-Ausgabe "
          f"(Rand ±{a.rand}s ausgelassen), davon {int(label[gilt].sum())} Werbung")

    # --- Nachbarn auf der GPU ------------------------------------------
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    T = torch.from_numpy(X).to(dev)
    R = torch.from_numpy(rec.astype(np.int32)).to(dev)
    L = torch.from_numpy(label.astype(np.float32)).to(dev)
    gegen = np.zeros(X.shape[0], np.float32)
    B = 2048
    for s in range(0, X.shape[0], B):
        e = min(s + B, X.shape[0])
        S = T[s:e] @ T.T
        S[R[s:e].unsqueeze(1) == R.unsqueeze(0)] = -2.0   # eigene Aufnahme raus
        _, idx = torch.topk(S, k=a.k, dim=1)
        nlab = L[idx]                                     # Labels der Nachbarn
        eig = L[s:e].unsqueeze(1)
        gegen[s:e] = (nlab != eig).float().mean(1).cpu().numpy()
        del S
    print("Nachbarn gezaehlt.")

    # --- Auswertung ----------------------------------------------------
    g = gegen[gilt]
    lab = label[gilt]
    kf = kopf[gilt]
    falsch = ((kf > 0.5).astype(np.int8) != lab)
    np.savez(Path(a.npz or "/tmp/backbone-sonde.npz"),
             gegen=g, label=lab, kopf=kf, falsch=falsch,
             rec=rec[gilt], sek=sek[gilt], uuids=np.array(uuids, object))

    kanten = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.01]

    # ⚠️ DER STOERFAKTOR. In den hohen Verwechslungs-Klassen steigt auch
    # der Werbeanteil (17.8 % auf 52.2 %). Werbung ist die
    # Minderheitsklasse und die schwierigere; ein Anstieg der Fehlerrate
    # koennte also allein von der Zusammensetzung kommen, nicht von der
    # Verwechslung. Deshalb wird GETRENNT NACH KLASSE gerechnet. Nur wenn
    # die Rate INNERHALB beider Klassen steigt, traegt der Befund.
    print(f"\n{'Nachbarn gegen':<18}{'Sekunden':>9}{'falsch':>9}"
          f"{'| Werbung: n':>14}{'falsch':>9}{'| Sendung: n':>14}{'falsch':>9}")
    zeilen = []
    for i in range(len(kanten) - 1):
        m = (g >= kanten[i]) & (g < kanten[i + 1])
        if not m.any():
            continue
        z = {"von": kanten[i], "bis": kanten[i + 1], "n": int(m.sum()),
             "falsch_pct": round(float(100 * falsch[m].mean()), 2)}
        teil = []
        for kl, name in ((1, "werbung"), (0, "sendung")):
            mk = m & (lab == kl)
            if mk.any():
                z[f"n_{name}"] = int(mk.sum())
                z[f"falsch_{name}_pct"] = round(float(100 * falsch[mk].mean()), 2)
                teil.append((int(mk.sum()), 100 * falsch[mk].mean()))
            else:
                teil.append((0, float("nan")))
        print(f"  {kanten[i]:.2f}-{kanten[i+1]:.2f}{'':<6}{int(m.sum()):>9}"
              f"{100*falsch[m].mean():>8.1f}%"
              f"{teil[0][0]:>14}{teil[0][1]:>8.1f}%"
              f"{teil[1][0]:>14}{teil[1][1]:>8.1f}%")
        zeilen.append(z)
    print(f"\n  insgesamt: {100*falsch.mean():.1f}% falsch bei "
          f"{100*lab.mean():.1f}% Werbeanteil")

    # Wo LIEGEN die Fehler? Das ist die Zahl, die ueber Idee 4 entscheidet:
    # eine steile Rate in einer winzigen Ecke ist folgenlos.
    hoch = g >= 0.50
    print(f"  Sekunden mit >=50 % Gegen-Nachbarn: {100*hoch.mean():.1f}% aller "
          f"Sekunden, aber {100*falsch[hoch].sum()/max(falsch.sum(),1):.1f}% aller Fehler")
    if len(g) > 10:
        r = float(np.corrcoef(g, falsch.astype(np.float32))[0, 1])
        print(f"  Korrelation Verwechslung <-> Fehler: r = {r:+.3f}")
    # Welche Sendungen stellen die verwechselten Sekunden? Wenn es die
    # Zeichentrick- und Film-Ecke ist, deckt sich das mit dem, was die
    # Wiederholungssuche unabhaengig gefunden hat -- dann ist es ein
    # STIL-Problem der Repraesentation und ein antrainiertes Backbone
    # koennte es loesen. Verteilt es sich ueber alles, ist es eher echte
    # Mehrdeutigkeit, und ein neues Backbone kauft nichts.
    import collections
    meta = _lade("w", "wiederholung.py").metadaten()
    rg = rec[gilt]
    top = collections.Counter()
    ges = collections.Counter()
    for i in np.flatnonzero(hoch & falsch):
        top[uuids[int(rg[i])]] += 1
    for i in np.flatnonzero(gilt):
        pass
    for i, u in enumerate(uuids):
        ges[u] = int((rg == i).sum())
    print(f"\n  Aufnahmen mit den meisten verwechselten Fehlern:")
    for u, n in top.most_common(10):
        t = meta.get(u, ("", ""))[0] or "?"
        print(f"    {n:>5} von {ges[u]:>5} Sekunden  {t[:38]:<40} {u}")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"k": a.k, "rand": a.rand, "dim": a.dim, "zeilen": zeilen,
             "gesamt_falsch_pct": round(float(100 * falsch.mean()), 2),
             "anteil_sekunden_hoch": round(float(100 * hoch.mean()), 2),
             "anteil_fehler_hoch": round(float(100 * falsch[hoch].sum() / max(falsch.sum(), 1)), 2),
             "r": round(r, 4) if len(g) > 10 else None}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
