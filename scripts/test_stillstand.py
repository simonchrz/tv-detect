#!/usr/bin/env python3
"""Tests fuer die Stillstandserkennung der OCR-Erhebung (kanten-schatten.py).

⚠️ Anlass 2026-09-01: O13 meldete Nacht fuer Nacht "noch nicht auswertbar",
als waere Geduld gefragt. Tatsaechlich war die OCR-Erhebung seit 16 Tagen
stehengeblieben — 58 Dumps am 15.08., danach 123 ohne, weil --ocr-marker
nicht mehr gesetzt wird.

Drei Fassungen der Pruefung bestanden nacheinander NICHT:
  "gibt es je OCR-Dumps?"       -> ja, 58 von 199
  "welche seit dem Schnitt?"    -> ja, alle 58 liegen 27 min danach
  "gibt es NOCH welche?"        -> nein. Erst das deckt es auf.

Ausfuehren: python3 scripts/test_stillstand.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ks", Path(__file__).resolve().parent / "kanten-schatten.py")
ks = importlib.util.module_from_spec(_spec)
sys.modules["ks"] = ks
try:
    _spec.loader.exec_module(ks)
except SystemExit:
    pass

TAG = 86400


class TestStillstand(unittest.TestCase):

    def test_laufende_erhebung_meldet_nichts(self):
        self.assertIsNone(ks.stillstand_tage([100 * TAG, 101 * TAG],
                                             [100 * TAG, 101 * TAG]))

    def test_kurze_luecke_meldet_nichts(self):
        """Zwei Tage ohne OCR sind Betrieb, kein Stillstand."""
        self.assertIsNone(ks.stillstand_tage([103 * TAG], [101 * TAG]))

    def test_der_echte_fall(self):
        """16 Tage Abstand — genau die Lage am 01.09.2026."""
        got = ks.stillstand_tage([116 * TAG], [100 * TAG])
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, 16.0, places=1)

    def test_ohne_ocr_dumps_kein_urteil(self):
        """Kein einziger OCR-Dump ist ein ANDERER Fall (eigene Meldung) und
        darf hier nicht als Stillstand durchgehen."""
        self.assertIsNone(ks.stillstand_tage([100 * TAG], []))

    def test_ohne_dumps_kein_urteil(self):
        self.assertIsNone(ks.stillstand_tage([], [100 * TAG]))

    def test_gemessen_wird_gegen_den_neuesten_dump_nicht_gegen_jetzt(self):
        """⚠️ Der Fallstrick: gegen die aktuelle Uhrzeit zu messen wuerde
        JEDE Betriebspause als Stillstand melden. Laeuft der Detect gar
        nicht, faellt auch nichts aus — hier stehen beide gleich alt da,
        und das ist kein Stillstand, egal wie lange es her ist."""
        self.assertIsNone(ks.stillstand_tage([50 * TAG], [50 * TAG]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
