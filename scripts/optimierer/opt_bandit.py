#!/usr/bin/env python3
"""Welche Ideenklasse ist als Naechstes dran? Thompson-Sampling ueber die
Bilanz aller Laeufe.

Belohnung je Lauf (0..1), nur aus dem Urteil — nie aus einer Selbstauskunft:
  r = clip((Median-Delta Entwicklung + 0.003) / 0.006, 0, 1)
      0 bei -0.003 oder schlechter, 0.5 bei Null, 1 ab +0.003 (Schwelle)
  halbiert, wenn der Aufnahme-Median sank (Mittel auf Kosten der Typischen)
  1.0 bei Status kandidat, 0.0 bei nicht_bestaetigt (Entwicklungsglueck)
Je Klasse Beta(1 + Summe r, 1 + Summe (1 - r)); gezogen wird die Klasse mit
der hoechsten Stichprobe. Unerprobte Klassen haben Beta(1, 1) und kommen
dadurch von selbst dran; tote Klassen trocknen aus, ohne ganz zu verschwinden.

Ausgabe: Tabelle und eine letzte Zeile `KLASSE <name>`.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

LAEUFE = Path(__file__).resolve().parent.parent.parent / "docs/optimierer/laeufe"
KLASSEN = ["merkmale", "training", "nachbearbeitung", "daten", "labels"]


def belohnung(u):
    st = u.get("status")
    if st == "kandidat":
        return 1.0
    if st == "nicht_bestaetigt":
        return 0.0
    e = u.get("entwicklung")
    if not e:
        return None
    r = min(1.0, max(0.0, (e["median_delta"] + 0.003) / 0.006))
    if e.get("delta_aufnahme_median", 0) < 0:
        r *= 0.5
    return r


def bilanz():
    b = {k: [] for k in KLASSEN}
    for d in sorted(LAEUFE.glob("[0-9][0-9][0-9]-*")):
        v = json.loads((d / "vorschlag.json").read_text())
        u = json.loads((d / "urteil.json").read_text()) if (d / "urteil.json").is_file() else {}
        r = belohnung(u)
        if r is not None and v.get("klasse") in b:
            b[v["klasse"]].append(r)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="nur fuer Tests")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    b = bilanz()
    zug = {}
    print(f"{'Klasse':<16}{'Laeufe':>7}{'mittl. r':>10}{'Beta':>14}{'Zug':>8}")
    for k in KLASSEN:
        s = sum(b[k])
        al, be = 1 + s, 1 + len(b[k]) - s
        zug[k] = rng.betavariate(al, be)
        print(f"{k:<16}{len(b[k]):>7}{(np.mean(b[k]) if b[k] else float('nan')):>10.2f}"
              f"{f'({al:.1f}, {be:.1f})':>14}{zug[k]:>8.2f}")
    print(f"KLASSE {max(zug, key=zug.get)}")


if __name__ == "__main__":
    main()
