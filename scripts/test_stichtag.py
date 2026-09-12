#!/usr/bin/env python3
"""Zwei Laeufe mit demselben Stichtag muessen bit-gleiche Gewichte rechnen.

Gefunden 2026-09-12 mit dem Matrix-Fingerabdruck: zwei Laeufe 20 Minuten
auseinander hatten BIT-GLEICHE Merkmale (X) und Labels (y), aber bei 602
von 681 Aufnahmen abweichende Gewichte — alle minimal, alle negativ. Das
ist die Altersrampe: `age_mult` sinkt stetig mit der Zeit. Ueber die
Golden-Auswertung schlug das mit 0.0116 durch und hat O18 blockiert.
"""
import re
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
QUELL = (HIER / "train-head.py").read_text()


class Stichtag(unittest.TestCase):
    def test_flagge_existiert(self):
        self.assertIn('"--stichtag"', QUELL)

    def test_keine_uhr_mehr_in_den_altersrechnungen(self):
        # Beide Stellen muessen ueber die Referenzzeit gehen.
        self.assertIn("rec_age_days = (args._jetzt - src_mt)", QUELL)
        self.assertIn("a_age = (args._jetzt - a_start)", QUELL)
        for treffer in re.findall(r".*_age.*= \(time\.time\(\).*", QUELL):
            self.fail(f"Altersrechnung liest noch die Uhr: {treffer.strip()}")

    def test_referenzzeit_wird_einmal_bestimmt(self):
        self.assertIn("args._jetzt = JETZT", QUELL)
        self.assertEqual(QUELL.count("args._jetzt = JETZT"), 1,
                         "die Referenzzeit darf nur an EINER Stelle entstehen")

    def test_der_lauf_sagt_welche_zeit_gilt(self):
        # ⚠️ Ohne Ausgabe ist ein gesetzter Stichtag von einem vergessenen
        # nicht zu unterscheiden — und genau diese Ununterscheidbarkeit hat
        # zwei Tage gekostet.
        i = QUELL.index("args._jetzt = JETZT")
        davor = QUELL[max(0, i - 1400):i]
        self.assertIn("stichtag:", davor)
        self.assertIn("NICHT vergleichbar", davor,
                      "ohne Stichtag muss der Lauf sagen, dass er nicht "
                      "vergleichbar ist")

    def test_begruendung_steht_dabei(self):
        i = QUELL.index("args._jetzt = JETZT")
        davor = QUELL[max(0, i - 1400):i]
        self.assertIn("0.0116", davor, "die gemessene Wirkung gehoert dazu")
        self.assertIn("602", davor, "wie viele Aufnahmen betroffen waren")

    def test_fingerabdruck_bleibt(self):
        # Er hat den Fund ermoeglicht und muss die naechste Suche fuehren.
        self.assertIn("matrix-fingerprint:", QUELL)
        self.assertIn("Gewichte je Aufnahme", QUELL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
