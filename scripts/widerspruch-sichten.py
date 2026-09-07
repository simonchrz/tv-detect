#!/usr/bin/env python3
"""Ordnet die langen Widersprüche zwischen Kopf und Menschenlabel ein.

WOHER DIE LISTE KOMMT
---------------------
`backbone-sonde.py` schreibt je Sekunde, ob der Kopf gegen das
menschliche Label steht. Der Median eines solchen Fehlerlaufs ist ZWEI
Sekunden — Flackern, das der HSMM-Dauer-Prior ohnehin verschluckt, weil
ein Werbeblock Minuten lang ist. Nur Läufe ab 30 s werden zu erfundenen
oder verpassten Blöcken und kosten IoU. Am 2026-09-07 waren das **26
Läufe in 12 von 99 Aufnahmen**, und alle 26 lagen im train-Eimer, keiner
im test-Eimer.

WAS DIE SICHTUNG AM 2026-09-07 ERGAB
-------------------------------------
Fünf Läufe wurden mit den Augen geprüft (Bilder aus der Quelle), der Rest
nach den daran geeichten Regeln eingeordnet:

  Label verpasst Werbung                11 Läufe, 646 s, 7 Aufnahmen
  Split-Screen-Werbung (Konvention)      7 Läufe, 477 s, 1 Aufnahme
  Label zu breit (Sendung als Werbung)   6 Läufe, 243 s, 3 Aufnahmen
  Nachlauf (Folgesendung, Konvention)    2 Läufe, 307 s, 2 Aufnahmen

**Zwei Drittel sind kaputte Labels, kein Modellfehler.** Belegt an:

  * `dvr-prosieben-1778691925` (Galileo): Label `[3316,3972]` verschluckt
    vier Minuten Sendung und endet dann VOR dem Ende der echten Werbung.
    Bei 3740–3900 s ist Galileo zu sehen, bei 3987 s Champagner- und
    ABOUT-YOU-Spots. Das Modell hat in beide Richtungen recht.
  * `dvr-nick-1778860200` (SpongeBob): der Lauf beginnt mit einer
    „Werbung"-Tafel, danach Netflix-Promo und Zeitschriftenanzeige.
  * `dvr-rtl-1781545200` (GZSZ): ein „Undercover Boss"-Trailer, per
    Konvention Programmvorschau und damit Werbung.

DIE ZWEI KONVENTIONEN, DIE KEINE FEHLER SIND
---------------------------------------------
**Nachlauf.** Läuft die Aufnahme über ihr Ende hinaus, beginnt die
Folgesendung, und dieser Schwanz wird absichtlich als überspringbar
markiert (`overrun` in der /ads-Antwort). Bei
`dvr-kabel-eins-1780856070` sind das 267 s „Yes we camp!" mit
Sendungs- UND Senderlogo — genau die Aufnahme, die das nächtliche
Label-Audit seit Tagen als widersprüchlich meldet. Sie ist es nicht.
Der Nachlauf wird deshalb ausdrücklich ausgeschlossen.

**Split-Screen-Werbung.** Bei `dvr-rtl-1780078500` (Let's Dance) läuft
der Spot groß, die Show weiter im kleinen Fenster, und der Bildschirm
zeigt selbst das Wort „Werbung". Der Kopf sagt Werbung (Logo im großen
Bild fehlt), das Label sagt Sendung (die Show läuft ja). Beide haben auf
ihre Art recht — das ist eine Klasse, die es im binären Ziel nicht gibt,
und der stärkste Einzelbeleg für Idee 5 (mehr Klassen). EINE Aufnahme
stellt damit 7 der 26 Läufe.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
DUMPS = Path.home() / ".cache" / "tv-detect-daemon" / "emit-signals"
BILD = Path.home() / ".cache" / "tvd-wiederholung" / "korpus"


def _lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, _HIER / datei)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def laeufe_aus_sonde(npz, min_dauer):
    z = np.load(npz, allow_pickle=True)
    lab, falsch, rec, sek = z["label"], z["falsch"], z["rec"], z["sek"]
    uu = [str(x) for x in z["uuids"]]
    aus = []
    for i in range(len(uu)):
        m = rec == i
        t, f, l = sek[m], falsch[m], lab[m]
        o = np.argsort(t)
        t, f, l = t[o], f[o], l[o]
        lauf, art, start = 0, None, None
        for k in range(len(t)):
            a = ("S->W" if l[k] == 0 else "W->S") if f[k] else None
            if a and k > 0 and t[k] - t[k - 1] == 1 and art == a:
                lauf += 1
            else:
                if art and lauf >= min_dauer:
                    aus.append({"uuid": uu[i], "start": int(start),
                                "dauer": int(lauf), "art": art})
                lauf, art, start = (1 if a else 0), a, int(t[k])
        if art and lauf >= min_dauer:
            aus.append({"uuid": uu[i], "start": int(start), "dauer": int(lauf), "art": art})
    return aus


def anreichern(r, ads_dir):
    u, s, d = r["uuid"], r["start"], r["dauer"]
    e = s + d
    r["kopf"] = r["logo"] = float("nan")
    p = DUMPS / f"{u}.json"
    if p.is_file():
        try:
            sig = json.loads(p.read_text())
            fps = float(sig.get("fps") or 25)
            nn = np.asarray(sig.get("nn_confs") or [], np.float32)
            lg = np.asarray(sig.get("logo_confs") or [], np.float32)
            a_, b_ = int(s * fps), int(e * fps)
            if b_ <= len(nn):
                r["kopf"] = round(float(nn[a_:b_].mean()), 3)
            if b_ <= len(lg):
                r["logo"] = round(float(np.nanmean(lg[a_:b_])), 3)
        except Exception:
            pass
    r["bild"] = 0.0
    bp = BILD / f"{u}.json"
    if bp.is_file():
        try:
            iv = [(a["window_start_s"], a["end_s"])
                  for a in json.loads(bp.read_text())["anchored"]]
            r["bild"] = round(100 * sum(max(0, min(e, b) - max(s, a))
                                        for a, b in iv) / d, 0)
        except Exception:
            pass
    # Nachlauf: die /ads-Antwort nennt ihn ausdruecklich.
    r["im_nachlauf"] = False
    ap = Path(ads_dir) / f"{u}.json"
    if ap.is_file():
        try:
            o = json.loads(ap.read_text()).get("overrun")
            if o and len(o) >= 2:
                lo, hi = float(o[0]), float(o[1])
                r["im_nachlauf"] = (min(e, hi) - max(s, lo)) > 0.5 * d
        except Exception:
            pass
    return r


def urteil(r, titel):
    """Geeicht an fuenf mit den Augen geprueften Laeufen (siehe Modulkopf).

    Die Reihenfolge ist Absicht: Konventionen zuerst, sonst wuerden sie
    als Labelfehler gezaehlt und die Zahl waere zu gross.
    """
    if r["im_nachlauf"]:
        return "Nachlauf (Folgesendung, Konvention)"
    if titel.startswith("Let's Dance"):
        return "Split-Screen-Werbung (Konventionsluecke)"
    if r["art"] == "W->S" and r["logo"] >= 0.80 and r["kopf"] <= 0.30:
        return "Label zu breit (Sendung als Werbung)"
    if r["art"] == "S->W" and r["logo"] <= 0.40 and r["kopf"] >= 0.70:
        return "Label verpasst Werbung"
    return "unklar, ansehen"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="/tmp/backbone-sonde.npz")
    ap.add_argument("--ads", default="/tmp/ads",
                    help="Verzeichnis mit /recording/<uuid>/ads-Antworten")
    ap.add_argument("--min-dauer", type=int, default=30, dest="min_dauer",
                    help="kuerzere Laeufe verschluckt der HSMM ohnehin")
    ap.add_argument("--json")
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        print(f"{a.npz} fehlt — erst backbone-sonde.py --npz laufen lassen")
        return 1
    meta = _lade("w", "wiederholung.py").metadaten()
    runs = [anreichern(r, a.ads) for r in laeufe_aus_sonde(a.npz, a.min_dauer)]
    for r in runs:
        r["titel"] = meta.get(r["uuid"], ("", ""))[0] or "?"
        r["urteil"] = urteil(r, r["titel"])
    runs.sort(key=lambda r: -r["dauer"])

    c = collections.Counter(r["urteil"] for r in runs)
    sek = collections.Counter()
    aufn = collections.defaultdict(set)
    for r in runs:
        sek[r["urteil"]] += r["dauer"]
        aufn[r["urteil"]].add(r["uuid"])
    print(f"{len(runs)} Laeufe ab {a.min_dauer}s in "
          f"{len({r['uuid'] for r in runs})} Aufnahmen\n")
    print(f"{'Urteil':<42}{'Laeufe':>7}{'Sek.':>7}{'Aufn.':>7}")
    for k, n in c.most_common():
        print(f"  {k:<40}{n:>7}{sek[k]:>7}{len(aufn[k]):>7}")
    kaputt = sum(n for k, n in c.items() if k.startswith("Label"))
    print(f"\n  kaputte Labels: {kaputt} von {len(runs)} Laeufen "
          f"({100*kaputt/max(len(runs),1):.0f} %)")
    print(f"\n{'Dauer':>6}{'Art':>6}{'Kopf':>7}{'Logo':>7}{'Bild':>7}  Urteil / Titel / Sekunde")
    for r in runs:
        print(f"{r['dauer']:>6}{r['art']:>6}{r['kopf']:>7.2f}{r['logo']:>7.2f}"
              f"{r['bild']:>6.0f}%  {r['urteil'][:34]:<36}{r['titel'][:22]:<24}{r['start']}s")
    if a.json:
        Path(a.json).write_text(json.dumps(runs, indent=1))
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
