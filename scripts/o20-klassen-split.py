#!/usr/bin/env python3
"""O20 — hilft es dem Kopf, wenn er zwei Sorten Werbung getrennt lernen darf?

DIE FRAGE
---------
Am 2026-09-07 gemessen: innerhalb der EINEN Zielklasse „Werbung" irrt der
Kopf auf dem Teil, der kein nachweislich wiederholter Spot ist, drei- bis
sechsmal so oft. Der Abstand bleibt, wenn man nach Verwechslungsrate im
Einbettungsraum schichtet (also nicht die Repraesentation), und wenn man
nach Abstand zur Blockkante schichtet (also nicht der Kontext). Es bleibt
die Zielklasse: „Produktspot ODER Programmvorschau ODER Ident" ist ein
Begriff, den der Kopf als einen lernen muss.

WARUM DIE ANKER-AUFTEILUNG HIER TROTZDEM TAUGT
-----------------------------------------------
Als TRAININGSZIEL waere sie falsch: sie sagt „wurde fingerprintet", nicht
„ist ein Produktspot", und ein Spot der nur einmal lief hat keinen Anker.
Fuer DIESE Frage stoert das nicht. Die dritte Klasse ist ein Hilfssignal
waehrend des Trainings; gelesen wird nur die BINAERE Ausgabe, also
P(Werbung) = P(Klasse 1) + P(Klasse 2). Wenn das Aufteilen des Ziels die
binaere Leistung hebt, ist die Heterogenitaet bestaetigt und echte
Unterklassen-Labels lohnen sich. Wenn nicht, ist die Idee billig erledigt.

WAS HIER NICHT PASSIERT
-----------------------
Nichts wird deployt, kein Nightly angefasst, kein Label geschrieben. Der
Lauf liest das Trainings-Archiv und den Split-Ledger und rechnet in
seinem eigenen Prozess (L4). Der versiegelte Satz wird nie angefasst.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ARCH = Path.home() / ".cache" / "tvd-train-archive"
BILD = Path.home() / ".cache" / "tvd-wiederholung" / "korpus"
LEDGER = ARCH / "split-ledger.json"


def anker_maske(u, n):
    """Sekunden, die ein Bild-Anker deckt. Praezision 96.8 % gegen
    Menschenlabel, geeicht am 2026-09-07."""
    p = BILD / f"{u}.json"
    m = np.zeros(n, bool)
    if not p.is_file():
        return m
    try:
        for a in json.loads(p.read_text())["anchored"]:
            s, e = int(a["window_start_s"]), int(a["end_s"]) + 1
            m[max(0, s):min(n, e)] = True
    except Exception:
        pass
    return m


def lade(eimer, limit=None, schritt=1):
    """(X, y3, uuid-Index) fuer einen Split-Eimer.

    y3: 0 = Sendung, 1 = Werbung MIT Anker, 2 = Werbung OHNE Anker.
    Der binaere Arm bildet daraus spaeter y>0.
    """
    led = json.loads(LEDGER.read_text())
    ziel = sorted(u for u, v in led.items() if v == eimer)
    if limit:
        ziel = ziel[:limit]
    Xs, ys, recs = [], [], []
    for i, u in enumerate(ziel):
        f = ARCH / f"{u}.npz"
        if not f.is_file():
            continue
        try:
            m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        except Exception:
            continue
        fp = m.get("feature_npy", "")
        ads = m.get("ads") or []
        if not fp or not os.path.exists(fp) or not ads:
            continue
        F = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float32)
        n = F.shape[0]
        # Breite vereinheitlichen wie die Produktion: Sentinel 0.5 an
        # Index 1280 EINSETZEN, nicht anhaengen (merkmalsspalte_verschoben).
        if F.shape[1] == 1281:
            F = np.concatenate([F[:, :1280],
                                np.full((n, 1), 0.5, np.float32),
                                F[:, 1280:]], axis=1)
        if F.shape[1] != 1282:
            continue
        # ⚠️ NaN in der Logo-Spalte. extract_logo scheitert still auf
        # einem kaputten Stream-Stueck und hinterlaesst NaN (Memory
        # logo_nan_is_contention_not_corruption); die Produktion setzt
        # dort denselben neutralen Sentinel 0.5 ein. Ohne das vergiftet
        # EIN NaN ueber den Spaltenmittelwert die ganze Spalte und damit
        # jede Zeile: der Kopf gibt NaN aus, sagt auf alles "Werbung" und
        # landet bei F1 0.3771 — exakt der Wert fuer 23 % Werbeanteil.
        # Drei Seeds lieferten dieselbe Zahl auf vier Stellen, und DAS
        # war der Hinweis: kein Modell, ein Kollaps.
        if np.isnan(F).any():
            F = np.nan_to_num(F, nan=0.5)
        y = np.zeros(n, np.int64)
        for a, b in ads:
            y[max(0, int(a)):min(n, int(b))] = 1
        ank = anker_maske(u, n)
        y[(y == 1) & ~ank] = 2
        # Unterabtastung: der volle train-Eimer waeren 3 Mio Sekunden zu
        # je 1282 float32, also 15 GB. Jede k-te Sekunde haelt ALLE
        # Aufnahmen im Satz und passt in den Speicher. Beide Arme sehen
        # exakt dieselben Zeilen.
        if schritt > 1:
            F = F[::schritt]; y = y[::schritt]
        Xs.append(F)
        ys.append(y)
        recs.append(np.full(F.shape[0], len(recs), np.int32))
    if not Xs:
        return None, None, None
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(recs)


def standardisieren(Xtr, Xte):
    """Spaltenweise zentrieren und skalieren, Kennwerte NUR aus train.

    Ohne das kollabiert der Kopf: die Backbone-Spalten haben Norm ~11,
    die Logo- und Audio-Spalte liegen zwischen 0 und 1. Der erste Lauf
    sagte auf ALLES "Werbung" und landete bei F1 0.377 — genau der Wert,
    den man bei 23 % Werbeanteil fuer eine Alles-ist-Werbung-Vorhersage
    bekommt. Drei Seeds lieferten exakt dieselbe Zahl, was der Hinweis
    war: das ist kein Modell, das ist ein Kollaps.
    """
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_und_werte(Xtr, ytr, Xte, yte, rec_te, klassen, seed, epochen, hidden,
                  faire_gewichte=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr if klassen == 3 else (ytr > 0).astype(np.int64)).to(dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, klassen)).to(dev)
    # Klassengewichte gegen die Schieflage (ca. 23 % Werbung). Fuer beide
    # Arme nach DERSELBEN Regel: Gewicht = N / (K * n_k).
    zaehl = torch.bincount(yt, minlength=klassen).float()
    w = (len(yt) / (klassen * zaehl.clamp(min=1))).to(dev)
    # ⚠️ NACHTRAEGLICHE DIAGNOSE, nicht Teil der Registrierung.
    # Die Regel N/(K*n_k) gibt bei drei Klassen der Werbung insgesamt das
    # 6.3-fache Gewicht gegenueber Sendung, bei zwei Klassen nur das
    # 3.15-fache — der Versuchsarm bekam also nebenbei eine ganz andere
    # Schieflagen-Korrektur. Das ist ein Konstruktionsfehler in O20: die
    # Zielaenderung war mit einer Gewichtsaenderung vermengt. Mit
    # --faire-gewichte tragen Klasse 1 und 2 zusammen genau so viel wie
    # die eine Werbeklasse im Kontrollarm.
    if faire_gewichte and klassen == 3:
        n_show = float(zaehl[0]); n_ad = float(zaehl[1] + zaehl[2])
        w_show = len(yt) / (2 * max(n_show, 1))
        w_ad = len(yt) / (2 * max(n_ad, 1))
        w = torch.tensor([w_show, w_ad, w_ad], dtype=torch.float32).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss(weight=w)
    B = 8192
    idx = np.arange(len(yt))
    for ep in range(epochen):
        np.random.shuffle(idx)
        for s in range(0, len(idx), B):
            j = torch.from_numpy(idx[s:s + B]).to(dev)
            opt.zero_grad()
            loss = lf(net(Xt[j]), yt[j])
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy(Xte).to(dev)), dim=1).cpu().numpy()
    # BINAER lesen: bei drei Klassen ist Werbung = Klasse 1 + Klasse 2.
    pad = p[:, 1] if klassen == 2 else p[:, 1] + p[:, 2]
    return pad


def glaetten(p, rec, k=10):
    """Gleitendes Mittel je Aufnahme — dieselbe Idee wie smooth=10s im
    Nightly. Ueber Aufnahmegrenzen hinweg zu glaetten waere ein Fehler."""
    out = np.empty_like(p)
    for r in np.unique(rec):
        m = rec == r
        v = p[m]
        out[m] = np.convolve(v, np.ones(k) / k, mode="same")
    return out


def f1(pred, wahr):
    tp = float(((pred == 1) & (wahr == 1)).sum())
    fp = float(((pred == 1) & (wahr == 0)).sum())
    fn = float(((pred == 0) & (wahr == 1)).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--limit", type=int, help="nur N Aufnahmen je Eimer (Probelauf)")
    ap.add_argument("--schritt", type=int, default=4,
                    help="jede k-te Sekunde im TRAINING (Speicher)")
    ap.add_argument("--nur-kontrollarm", action="store_true", dest="nur_kontroll",
                    help="nur der 2-Klassen-Arm — fuer die Rauschmessung VOR "
                         "der Schwellenfestlegung")
    ap.add_argument("--faire-gewichte", action="store_true",
                    dest="faire_gewichte",
                    help="NACHTRAEGLICHE Diagnose: Klasse 1+2 zusammen so\n                         schwer wie die Werbeklasse im Kontrollarm")
    ap.add_argument("--json")
    a = ap.parse_args()

    print("Lade train …", flush=True)
    Xtr, ytr, _ = lade("train", a.limit, a.schritt)
    print("Lade test …", flush=True)
    Xte, yte, rec_te = lade("test", a.limit, 1)
    if Xtr is None or Xte is None:
        print("keine Daten"); return 1
    wahr = (yte > 0).astype(np.int64)
    print(f"train {Xtr.shape[0]} Sekunden ({100*(ytr>0).mean():.1f} % Werbung, "
          f"davon {100*(ytr==1).sum()/max((ytr>0).sum(),1):.0f} % mit Anker)")
    print(f"test  {Xte.shape[0]} Sekunden ({100*wahr.mean():.1f} % Werbung)")
    Xtr, Xte = standardisieren(Xtr, Xte)
    print("standardisiert (Kennwerte nur aus train)")

    erg = {"2": [], "3": []}
    arme = [2] if a.nur_kontroll else [2, 3]
    for seed in range(a.seeds):
        for k in arme:
            p = fit_und_werte(Xtr, ytr, Xte, yte, rec_te, k, seed,
                              a.epochen, a.hidden, a.faire_gewichte)
            ps = glaetten(p, rec_te)
            s = f1((ps > 0.5).astype(np.int64), wahr)
            erg[str(k)].append(s)
            print(f"  Seed {seed}  {k} Klassen  F1 {s:.4f}", flush=True)
    for k in arme:
        v = np.array(erg[str(k)])
        print(f"\n{k} Klassen: Median {np.median(v):.4f}  "
              f"Mittel {v.mean():.4f}  Streuung(sd) {v.std(ddof=1):.4f}  "
              f"Spanne {v.max()-v.min():.4f}")
    if not a.nur_kontroll:
        d = np.array(erg["3"]) - np.array(erg["2"])
        print(f"\nDelta (3 minus 2), gepaart je Seed: Median {np.median(d):+.4f}  "
              f"positiv in {int((d>0).sum())} von {len(d)} Seeds")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
