#!/usr/bin/env python3
"""Die Blockreihenfolge der Trainingsmatrix muss festgenagelt sein.

Gefunden 2026-09-11 bei der Suche nach der Ursache von O18: zwei
identische Laeufe streuten um Median |Δ| 0.0073, waehrend der zu
messende Effekt 0.0032 war. Die beiden im Ledger genannten Verdaechtigen
trugen nicht -- der Fit-Kern ist prozessuebergreifend bit-gleich, und an
den Alters-Kanten liegt in zwei Stunden keine Aufnahme.

Es war die Reihenfolge: `per_rec` entsteht als `cached + todo`, und
welche Aufnahme in welcher Liste landet, haengt nur davon ab, ob ihre
Merkmalsdatei schon existierte. Gemessen an 60 synthetischen Aufnahmen
verschiebt allein das Verschieben von zwoelf Bloecken ans Ende den
Verlust um bis zu 0.0072.
"""
import re
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
QUELL = (HIER / "train-head.py").read_text()


class ReihenfolgeStabil(unittest.TestCase):
    def test_per_rec_wird_sortiert(self):
        self.assertIn("per_rec.sort(key=lambda r: r[0])", QUELL,
                      "per_rec muss nach uuid sortiert werden, bevor daraus "
                      "eine Matrix wird")

    def test_sortierung_steht_vor_der_matrix(self):
        i = QUELL.index("per_rec.sort(")
        j = QUELL.index("target_dim = max(r[3].shape[1] for r in per_rec)")
        self.assertLess(i, j, "erst sortieren, dann die Matrix bauen")

    def test_sortiert_nach_uuid_nicht_nach_titel_oder_zeit(self):
        # uuid ist r[0] und das einzige stabile Feld: der Titel kann sich
        # aendern, die Cache-mtime wandert bei jeder Neu-Extraktion.
        i = QUELL.index("per_rec.sort(")
        self.assertIn("r[0]", QUELL[i:i + 60])
        for verboten in ("r[1]", "mtime", "time.time"):
            self.assertNotIn(verboten, QUELL[i:i + 60],
                             f"nicht nach {verboten} sortieren")

    def test_begruendung_steht_dabei(self):
        # Ohne die Zahlen wird die Zeile beim naechsten Umbau als
        # ueberfluessig geloescht.
        i = QUELL.index("per_rec.sort(")
        davor = QUELL[max(0, i - 1800):i]
        self.assertIn("0.0073", davor, "die gemessene Streuung gehoert dazu")
        self.assertIn("O18", davor, "der Anlass gehoert dazu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
