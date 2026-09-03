#!/usr/bin/env python3
"""Agenten-Urteile einsammeln und den Kandidatenbericht bauen.

Erwartet je Kante eine Datei <urteile>/e<NNNN>.txt mit 15 Zeichen W/S in
zeitlicher Reihenfolge (-24 s bis +4 s in 2-s-Schritten).

⚠️ Schreibt KEIN Label. Ausgabe ist eine Liste zum Vorlegen — Labels sind
Eingabe, nicht Stellschraube (Leitplanke L2).
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "va", Path(__file__).with_name("vorlauf_auswerten.py"))
va = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(va)

I_START = 12          # Index des Bildes AM Blockanfang
SCHRITT_S = 2.0


def bericht(kanten, urteile_dir):
    zu, ok, unklar, fehlt = [], 0, 0, 0
    for a in kanten:
        p = urteile_dir / ("e%04d.txt" % a["nr"])
        if not p.is_file():
            fehlt += 1
            continue
        folge = p.read_text().strip()
        if len(folge) != 15 or set(folge) - {"W", "S"}:
            fehlt += 1
            continue
        art, n = va.urteil(folge, I_START)
        if art == "zu_spaet":
            zu.append({**a, "sekunden": SCHRITT_S * n, "gedeckelt": n == I_START})
        elif art == "ok":
            ok += 1
        else:
            unklar += 1
    return zu, ok, unklar, fehlt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--urteile", type=Path, required=True)
    args = ap.parse_args()

    kanten = json.loads((args.sweep / "kanten.json").read_text())
    zu, ok, unklar, fehlt = bericht(kanten, args.urteile)
    n_aus = len(zu) + ok
    print("Blockanfaenge: %d   ausgewertet: %d   fehlend: %d" % (len(kanten), n_aus + unklar, fehlt))
    print("zu spaet: %d   sauber: %d   unklar: %d" % (len(zu), ok, unklar))
    if n_aus:
        print("Rate: %d/%d = %.0f %%" % (len(zu), n_aus, 100 * len(zu) / n_aus))
    print()
    print("nach Kanal (zu spaet / auswertbar):")
    proK = Counter(a["kanal"] for a in zu)
    alleK = Counter(a["kanal"] for a in kanten)
    for k in sorted(alleK, key=lambda k: -proK.get(k, 0)):
        if proK.get(k):
            print("  %-16s %2d / %2d" % (k, proK[k], alleK[k]))
    print()
    # ⚠️ Die Einheit ist die AUFNAHME, nicht die Kante. Beim Probelauf fielen
    # bei zwei Aufnahmen je ZWEI Anfaenge mit aehnlichem Versatz auf — das
    # sind keine Einzelausrutscher, sondern eine Aufnahme, die durchgehend
    # nach einer anderen Regel gelabelt wurde. Wer je Kante entscheidet,
    # flickt Symptome und laesst dieselbe Aufnahme halb falsch stehen.
    je_rec = {}
    for a in zu:
        je_rec.setdefault(a["uuid"], []).append(a)
    kanten_je_rec = {}
    for a in kanten:
        kanten_je_rec[a["uuid"]] = kanten_je_rec.get(a["uuid"], 0) + 1

    print("KANDIDATEN je Aufnahme (nichts geschrieben, L2):")
    for u, xs in sorted(je_rec.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        x0 = xs[0]
        print("  %-32s %-14s %s   %d von %d Anfaengen" % (
            u, x0["kanal"], x0["label_datum"], len(xs), kanten_je_rec[u]))
        for a in sorted(xs, key=lambda a: a["start"]):
            print("      Anfang %9.1f  -> ab ca. %9.1f   (%s%.0f s)" % (
                a["start"], a["start"] - a["sekunden"],
                ">=" if a["gedeckelt"] else "", a["sekunden"]))
    ganz = [u for u, xs in je_rec.items() if len(xs) == kanten_je_rec[u] and len(xs) > 1]
    if ganz:
        print()
        print("davon DURCHGEHEND betroffen (jeder geprüfte Anfang zu spaet) — "
              "diese Aufnahmen folgen einer anderen Konvention, nicht einem "
              "Ausrutscher:")
        for u in sorted(ganz):
            print("  " + u)
    return 0


if __name__ == "__main__":
    sys.exit(main())
