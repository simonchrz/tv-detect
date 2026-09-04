#!/usr/bin/env python3
"""Hat der Agent die Bilder wirklich angesehen?

⚠️ Warum das nötig ist (2026-09-04): drei von zwanzig Klassifikations-
Agenten lieferten wohlgeformte, plausible Urteile — ohne die Bilder gelesen
zu haben. Einer reproduzierte vier Sequenzen ZEICHENGLEICH zum Vortag. Am
Ergebnis war nichts zu sehen; verraten hat es allein die Zahl der
Tool-Aufrufe, und die kommt vom Harness, nicht vom Agenten.

Die Prüfung zählt im Agenten-Protokoll, wie viele der erwarteten Bilder
tatsächlich gelesen wurden. Ein Auftrag ohne vollständige Lesevorgänge wird
VERWORFEN, nicht bewertet.

⚠️ NUR auf abgeschlossene Protokolle anwenden. Ein noch laufender Agent hat
ein unfertiges Protokoll; wer da zählt, hält gute Läufe für Fälschungen —
genau dieser Fehler ist mir am 04.09. unterlaufen und hat die Zahl der
betroffenen Kanten verdoppelt.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def bilder_gelesen(protokoll, muster):
    """Zahl der Bild-Lesevorgänge im Protokoll — ohne es in den Speicher zu
    laden (die Dinger sind hunderte MB)."""
    p = subprocess.run(["grep", "-o", muster, str(protokoll)],
                       capture_output=True, text=True)
    return len(set(p.stdout.split()))


def urteil_gueltig(text, laenge):
    t = text.strip()
    return len(t) == laenge and not (set(t) - {"W", "S"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protokolle", type=Path, required=True)
    ap.add_argument("--urteile", type=Path, required=True)
    ap.add_argument("--marke", required=True,
                    help="Pfadstück, das die Zieldateien im Auftrag kennzeichnet")
    ap.add_argument("--bildmuster", default=r"sweep/e[0-9]\{4\}/bild[0-9]\{2\}\.png")
    ap.add_argument("--je-kante", type=int, default=15)
    args = ap.parse_args()

    verdaechtig, geprueft = [], 0
    for prot in sorted(args.protokolle.glob("a*.output")):
        kopf = subprocess.run(["grep", "-o", args.marke + "/e[0-9]\\{4\\}",
                               str(prot)], capture_output=True, text=True).stdout
        ziele = sorted(set(re.findall(r"e\d{4}", kopf)))
        if not ziele:
            continue
        geprueft += 1
        n = bilder_gelesen(prot, args.bildmuster)
        if n < args.je_kante * len(ziele):
            verdaechtig.append((ziele, n, args.je_kante * len(ziele)))

    print("Auftraege geprueft: %d   mit Luecke: %d" % (geprueft, len(verdaechtig)))
    weg = []
    for ziele, n, soll in verdaechtig:
        print("  %s … %s: %d von %d Bildern gelesen" % (ziele[0], ziele[-1], n, soll))
        weg += ziele

    formfehler = []
    for p in sorted(args.urteile.glob("e*.txt")):
        if not urteil_gueltig(p.read_text(), args.je_kante):
            formfehler.append(p.stem)
    if formfehler:
        print("Formfehler im Urteil:", " ".join(formfehler))

    zu_verwerfen = sorted(set(weg) | set(formfehler))
    if zu_verwerfen:
        print("\nZU VERWERFEN (%d): %s" % (len(zu_verwerfen), " ".join(zu_verwerfen)))
        return 1
    print("\nalle Urteile stammen von Agenten mit vollstaendigen Lesevorgaengen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
