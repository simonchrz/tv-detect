#!/usr/bin/env python3
"""Ist die Zeit->Bild-Zuordnung dieser Aufnahme reproduzierbar?

⚠️ 2026-09-04: aus zwei Fenstern um dieselbe Kante kamen fuer dieselbe
Sekundenangabe VERSCHIEDENE Bilder — pixelweise ueber die ganze Flaeche,
nicht um ein Einzelbild versetzt. Auch eine direkte Suche auf dieselbe
Sekunde traf ein drittes Bild. Betroffen sind DVB-Mitschnitte mit
kaputtem Zeitstempelstrom (PTS-Wrap, fehlende Referenzbilder); die Suche
landet dort woanders, als sie soll.

Diese Pruefung klaert nicht die Ursache, sondern die Eigenschaft, auf die
es ankommt: dieselbe Sekunde zweimal aus verschiedenen Startpunkten holen
und die Bilder vergleichen. Weichen sie ab, ist jede Sekundenangabe aus
dieser Aufnahme eine Annahme — und Messungen daran sind wertlos.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def bild(quelle, ss, ziel, breite=320):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(ss), "-i", str(quelle),
                    "-vf", f"scale={breite}:-2", "-frames:v", "1", "-y", str(ziel)],
                   capture_output=True)
    return ziel.exists() and ziel.stat().st_size > 0


def abweichung(a, b):
    from PIL import Image, ImageChops
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    if ia.size != ib.size:
        return 255.0
    d = ImageChops.difference(ia, ib).convert("L")
    px = d.get_flattened_data() if hasattr(d, "get_flattened_data") else d.getdata()
    px = list(px)
    return sum(px) / len(px)


def reproduzierbar(quelle, sekunde, schwelle=8.0):
    """Dieselbe Sekunde direkt und aus einem 20 s frueheren Fenster."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a, b = td / "a.png", td / "b.png"
        if not bild(quelle, sekunde, a):
            return None, None
        # zweiter Weg: 20 s frueher ansetzen und 20 s weiterlaufen lassen
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(sekunde - 20),
                        "-i", str(quelle), "-vf", "scale=320:-2,fps=1/20",
                        "-frames:v", "2", "-y", str(td / "f%02d.png")],
                       capture_output=True)
        zwei = td / "f02.png"
        if not zwei.exists():
            return None, None
        d = abweichung(a, zwei)
        return d <= schwelle, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quellen", nargs="+", type=Path)
    ap.add_argument("--sekunde", type=float, default=600.0)
    args = ap.parse_args()
    schlecht = 0
    for q in args.quellen:
        ok, d = reproduzierbar(q, args.sekunde)
        if ok is None:
            print("%-34s  nicht pruefbar" % q.stem[:33]); continue
        if not ok:
            schlecht += 1
        print("%-34s  Abweichung %6.1f  %s" % (q.stem[:33], d, "ok" if ok else "<<< NICHT REPRODUZIERBAR"))
    print("\nnicht reproduzierbar: %d von %d" % (schlecht, len(args.quellen)))
    return 1 if schlecht else 0


if __name__ == "__main__":
    sys.exit(main())
