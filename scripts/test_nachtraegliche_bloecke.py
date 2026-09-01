#!/usr/bin/env python3
"""Tests fuer die Regel "nach dem Abschluss dazugekommen".

⚠️ Der Anlass, 2026-09-01: dvr-rtl-1780683300 (Let's Dance) wurde am 31.07.
mit 7 Bloecken abgeschlossen; am 17.08. schrieb ein Config-Fingerprint-
Re-Detect ads.json mit 13 Bloecken neu. Die 6 neuen hat nie jemand geprueft —
weder ein Mensch noch die Auto-Bestaetigung — und sie standen ueber den Merge
im Trainingslabel. Mindestens zwei zeigen nachweislich die Sendung.

Ausfuehren: python3 scripts/test_nachtraegliche_bloecke.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "th", Path(__file__).resolve().parent / "train-head.py")
th = importlib.util.module_from_spec(_spec)
sys.modules["th"] = th
try:
    _spec.loader.exec_module(th)
except SystemExit:
    pass

nachtraeglich = th.nachtraegliche_bloecke


class TestNachtraeglicheBloecke(unittest.TestCase):

    def test_ohne_schnappschuss_wird_nichts_verworfen(self):
        """Kein auto_at_review heisst "unbekannt", nicht "alles neu".

        Bei Aufnahmen von vor 08-2026 fehlt der Schnappschuss. Wer daraus
        "also alles nachtraeglich" macht, loescht das halbe Korpus.
        """
        self.assertEqual(nachtraeglich([[10, 20], [30, 40]], None), [])

    def test_bekannte_bloecke_bleiben(self):
        schnapp = [[10, 20], [30, 40]]
        self.assertEqual(nachtraeglich([[10, 20], [30, 40]], schnapp), [])

    def test_neuer_block_wird_erkannt(self):
        schnapp = [[10, 20]]
        self.assertEqual(nachtraeglich([[10, 20], [300, 400]], schnapp),
                         [[300, 400]])

    def test_verschobene_kante_ist_derselbe_block(self):
        """⚠️ Der wichtigste Fall. Ein Re-Detect verschiebt Kanten um
        Sekunden. Wer auf GLEICHHEIT prueft statt auf Ueberlappung, haelt
        jeden nachjustierten Block fuer neu und verwirft die ganze Liste —
        aus einer Schutzmassnahme wird ein Korpus-Schredder."""
        schnapp = [[426, 551]]
        self.assertEqual(nachtraeglich([[428, 549]], schnapp), [])
        self.assertEqual(nachtraeglich([[420, 560]], schnapp), [])

    def test_beruehrung_zaehlt_nicht_als_ueberlappung(self):
        """Ende==Anfang ist keine Ueberlappung: zwei Bloecke, die sich nur
        beruehren, sind verschiedene Bloecke."""
        self.assertEqual(nachtraeglich([[20, 30]], [[10, 20]]), [[20, 30]])

    def test_der_echte_fall_letsdance(self):
        """Die 7 abgeschlossenen vs. die 13 nach dem Re-Detect."""
        beim_abschluss = [[1250, 1771], [3263, 3750], [5305, 5798],
                          [7172, 7679], [9060, 9556], [10830, 11340],
                          [13108, 13580]]
        # Was nach dem Merge ueberlebt: die Auto-Bloecke ohne User-Gegenstueck.
        ueberlebende = [[426, 551], [2985, 3136], [6662, 6802],
                        [7926, 8062], [8385, 8493], [11914, 12068]]
        neu = nachtraeglich(ueberlebende, beim_abschluss)
        self.assertEqual(len(neu), 6, "alle 6 sind nach dem Abschluss entstanden")
        # Die beiden, die ich an Einzelbildern als Sendung belegt habe.
        self.assertIn([426, 551], neu)
        self.assertIn([6662, 6802], neu)

    def test_leerer_schnappschuss_ist_nicht_None(self):
        """[] heisst "die Maschine schlug beim Abschluss NICHTS vor" — dann
        ist jeder heutige Block nachtraeglich. Das ist etwas anderes als
        None ("wir wissen es nicht") und darf nicht damit verschmelzen."""
        self.assertEqual(nachtraeglich([[10, 20]], []), [[10, 20]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
