#!/usr/bin/env python3
"""Fehlerbudget: WO geht IoU verloren, in Sekunden, mit einer Ursache je Sekunde.

WARUM
-----
Der 2026-09-07 lief als Folge von Ideen: vorschlagen, bauen, messen,
zuruecknehmen. Fuenf Aussagen wurden am selben Tag widerrufen. Nicht weil
die Messungen falsch waren, sondern weil es keinen gemeinsamen Massstab
gab, an dem sich ordnen liess, was sich lohnt. Dieses Skript ist der
Massstab.

DAS PRINZIP
-----------
Bevor irgendetwas verbessert wird, wird der Verlust zerlegt. Jede Sekunde,
in der Modell und Wahrheit auseinanderliegen, bekommt GENAU EINE Ursache.
Dazu zwei Orakel-Laeufe, die sagen, was maximal zu holen waere, ohne
irgendetwas zu bauen:

  * Orakel-NN -> echter Dekoder: die Labels werden als NN-Ausgabe
    eingespielt. Was dann noch fehlt, ist der DEKODER-Deckel.
  * echtes NN -> kein Dekoder: das geglaettete NN wird nur geschwellt.
    Der Abstand zur Produktion zeigt, was der Dekoder heute beitraegt.

MESSSATZ
--------
messsatz-2026-09-07.json: 98 Aufnahmen mit menschlichem Label, geprueft
gleich langer Zeitachse und Signal-Dump. Eingefroren mit Hash. 76 davon
liegen im train-Eimer; fuer DEKODER-Fragen ist das egal (der Kopf ist
eingefroren), fuer KOPF-Fragen zaehlen nur die 22 test-Aufnahmen.

DIE URSACHEN
------------
Fuer jede Sekunde mit Modell != Wahrheit:
  Kante             innerhalb +-15 s einer Wahrheits-Kante
  NN erfindet       Modell sagt Werbung, Wahrheit Sendung, NN > 0.5
  Dekoder erfindet  dito, aber NN <= 0.5 -- der Dekoder hat es gebaut
  NN verpasst       Modell sagt Sendung, Wahrheit Werbung, NN <= 0.5
  Dekoder verpasst  dito, aber NN > 0.5 -- das NN war da, der Dekoder nicht
Dazu getrennt, weil es KEIN Modellfehler ist:
  Label/Konvention  Wiederholungs-Anker (96.8 % Praezision) sagen Werbung,
                    das Label sagt Sendung; und Endbloecke am Aufnahmeende
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MESSSATZ = Path.home() / ".cache/tvd-train-archive/messsatz-2026-09-07.json"
DUMPS = Path.home() / ".cache/tv-detect-daemon/emit-signals"
BILD = Path.home() / ".cache/tvd-wiederholung/korpus"
SNAPSHOT = Path("/tmp/tv-train-snapshot")
BIN = Path.home() / ".local/bin/tv-detect"
_HIER = Path(__file__).resolve().parent


def _lade(name, datei):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _HIER / datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def block_iou(pred, gt):
    """Definition aus eval_production_cutlists.py = train-head.py."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tot = 0.0
    for gs, ge in gt:
        best = 0.0
        for ps, pe in pred:
            inter = max(0.0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            if union > 0:
                best = max(best, inter / union)
        tot += best
    return tot / len(gt)


def _bl(x):
    return [(float(a), float(b)) for a, b in (x or []) if float(b) > float(a)]


def replay(dump_pfad):
    r = subprocess.run([str(BIN), "--replay-signals", str(dump_pfad),
                        "--decoder", "hsmm", "--hsmm-dur-w", "15"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return _bl(json.loads(r.stdout).get("blocks"))
    except Exception:
        return None


def orakel_dump(dump, truth, tmp, uuid, hoch=0.98, tief=0.02):
    """Dump mit den LABELS als NN-Ausgabe. 0.98/0.02 statt 1/0, damit
    kein log(0) im Dekoder entsteht."""
    fps = float(dump.get("fps") or 25)
    n = len(dump["nn_confs"])
    nn = np.full(n, tief, np.float32)
    for s, e in truth:
        nn[int(s * fps):min(n, int(e * fps))] = hoch
    d = dict(dump)
    d["nn_confs"] = nn.tolist()
    p = tmp / f"{uuid}-orakel.json"
    p.write_text(json.dumps(d))
    return p


def nn_nur(nn, fps, k=10):
    """Geglaettetes NN geschwellt, ohne Dekoder. Laeufe >= 30 s."""
    sek = np.array([nn[int(t * fps):int((t + 1) * fps)].mean()
                    for t in range(int(len(nn) / fps))])
    g = np.convolve(sek, np.ones(k) / k, mode="same")
    m = g > 0.5
    aus, s = [], None
    for t, v in enumerate(m):
        if v and s is None:
            s = t
        if not v and s is not None:
            if t - s >= 30:
                aus.append((float(s), float(t)))
            s = None
    if s is not None and len(m) - s >= 30:
        aus.append((float(s), float(len(m))))
    return aus


def maske(bl, n):
    m = np.zeros(n, bool)
    for s, e in bl:
        m[max(0, int(s)):min(n, int(e))] = True
    return m


def ursachen(pred, truth, nn_sek, n, rand=15):
    """Je Sekunde mit pred != truth genau eine Ursache."""
    P, T = maske(pred, n), maske(truth, n)
    kanten = np.array([x for b in truth for x in b]) if truth else np.zeros(0)
    c = collections.Counter()
    for t in np.flatnonzero(P != T):
        nahe = len(kanten) and np.min(np.abs(kanten - t)) <= rand
        if nahe:
            c["Kante"] += 1
        elif P[t] and not T[t]:
            c["NN erfindet" if nn_sek[t] > 0.5 else "Dekoder erfindet"] += 1
        else:
            c["Dekoder verpasst" if nn_sek[t] > 0.5 else "NN verpasst"] += 1
    return c


def label_seite(truth, n, uuid):
    """Sekunden, die das LABEL vermutlich falsch hat -- kein Modellfehler."""
    c = collections.Counter()
    T = maske(truth, n)
    p = BILD / f"{uuid}.json"
    if p.is_file():
        A = maske([(a["window_start_s"], a["end_s"])
                   for a in json.loads(p.read_text())["anchored"]], n)
        c["Anker sagt Werbung, Label Sendung"] = int((A & ~T).sum())
    for s, e in truth:
        if n - e <= 3 and e - s >= 60:
            c["Endblock am Aufnahmeende"] += int(e - s)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    ms = json.loads(MESSSATZ.read_text())
    uuids = ms["uuids"][:a.limit] if a.limit else ms["uuids"]
    print(f"Messsatz {ms['name']} (hash {ms['hash']}), {len(uuids)} Aufnahmen")
    ub = _lade("bs", "backbone-sonde.py").menschlabels()
    led = json.loads((Path.home() / ".cache/tvd-train-archive/split-ledger.json").read_text())

    tmp = Path(tempfile.mkdtemp(prefix="fehlerbudget-"))
    zeilen = []
    ges = collections.Counter()
    ges_label = collections.Counter()
    for i, u in enumerate(uuids, 1):
        truth = ub.get(u)
        dp = DUMPS / f"{u}.json"
        if not truth or not dp.is_file():
            continue
        dump = json.loads(dp.read_text())
        fps = float(dump.get("fps") or 25)
        nn = np.asarray(dump["nn_confs"], np.float32)
        n = int(len(nn) / fps)
        nn_sek = np.array([nn[int(t * fps):int((t + 1) * fps)].mean() for t in range(n)])

        prod = replay(dp)
        orak = replay(orakel_dump(dump, truth, tmp, u))
        nur = nn_nur(nn, fps)
        if prod is None or orak is None:
            print(f"  {u}: Replay fehlgeschlagen", file=sys.stderr)
            continue
        z = {"uuid": u, "eimer": led.get(u, "?"), "n": n,
             "iou_prod": block_iou(prod, truth),
             "iou_orakel_nn": block_iou(orak, truth),
             "iou_nn_nur": block_iou(nur, truth)}
        c = ursachen(prod, truth, nn_sek, n)
        cl = label_seite(truth, n, u)
        z["ursachen"] = dict(c)
        z["label"] = dict(cl)
        ges.update(c)
        ges_label.update(cl)
        zeilen.append(z)
        if i % 10 == 0:
            print(f"  {i}/{len(uuids)}", flush=True)

    if not zeilen:
        print("nichts gemessen"); return 1

    def med(k, filt=None):
        v = [z[k] for z in zeilen if (filt is None or z["eimer"] == filt)]
        return float(np.median(v)) if v else float("nan")

    print(f"\n=== Block-IoU (Median), {len(zeilen)} Aufnahmen ===")
    print(f"{'':<34}{'alle':>8}{'nur test':>10}")
    for k, name in (("iou_prod", "Produktion (echtes NN, HSMM)"),
                    ("iou_orakel_nn", "Orakel-NN -> HSMM"),
                    ("iou_nn_nur", "echtes NN, KEIN Dekoder")):
        print(f"  {name:<32}{med(k):>8.3f}{med(k,'test'):>10.3f}")
    print(f"\n  Dekoder-Deckel  (1 - Orakel-NN):     {1-med('iou_orakel_nn'):.3f}"
          f"  <- so viel geht verloren, selbst wenn das NN perfekt waere")
    print(f"  NN-Deckel       (Orakel - Produktion): {med('iou_orakel_nn')-med('iou_prod'):.3f}"
          f"  <- so viel waere mit perfektem NN zu holen")
    print(f"  Dekoder-Beitrag (Produktion - NN nur): {med('iou_prod')-med('iou_nn_nur'):+.3f}"
          f"  <- was der HSMM heute gegenueber blossem Schwellen bringt")

    tot = sum(ges.values())
    print(f"\n=== Verlust-Sekunden nach Ursache (Modell gegen Wahrheit) ===")
    print(f"{'Ursache':<20}{'Sekunden':>10}{'Anteil':>8}")
    for k, v in ges.most_common():
        print(f"  {k:<18}{v:>10}{100*v/max(tot,1):>7.1f}%")
    print(f"  {'SUMME':<18}{tot:>10}")

    tl = sum(ges_label.values())
    print(f"\n=== Label-Seite (kein Modellfehler, getrennt gezaehlt) ===")
    for k, v in ges_label.most_common():
        print(f"  {k:<40}{v:>8}s")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"messsatz": ms["hash"], "zeilen": zeilen,
             "ursachen": dict(ges), "label": dict(ges_label)}, indent=1))
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
