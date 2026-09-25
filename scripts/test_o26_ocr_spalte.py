#!/usr/bin/env python3
"""O26: die OCR-Spalten muessen zeilengenau zu X passen.

Ein Versatz zwischen Spur-Sekunde und Merkmalszeile waere still: die Spalte
saehe informativ aus und laege neben den Stellen, die sie markiert (dieselbe
Klasse wie frames_tragen_erwartete_zeit).
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("o26", HIER / "o26-ocr-spalte.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class OcrSpalten(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = M.SPUR
        M.SPUR = self.tmp
        (self.tmp / "u1.json").write_text(json.dumps({
            "abgetastet": [{"von": 0, "bis": 180}],
            "funde": [{"time_s": 100, "hinweis": True, "werbemarker": False},
                      {"time_s": 150, "hinweis": False, "werbemarker": True}]}))

    def tearDown(self):
        M.SPUR = self.alt

    def test_fenster_und_abdeckung(self):
        s = M.ocr_spalten("u1", 200)
        self.assertEqual(s[90:111, 0].tolist(), [1] * 21)   # +-10 s um 100
        self.assertEqual(s[89, 0], 0)
        self.assertEqual(s[111, 0], 0)
        self.assertEqual(s[140:161, 1].sum(), 21)
        self.assertEqual(s[:180, 2].sum(), 180)             # abgetastet
        self.assertEqual(s[180:, 2].sum(), 0)               # dahinter nicht

    def test_ohne_spur_alles_null(self):
        self.assertEqual(M.ocr_spalten("gibtsnicht", 50).sum(), 0)

    def test_schrittweite_zeilengenau(self):
        # lade() nimmt Sekunden 0, 4, 8, ... -> Zeile 25 = Sekunde 100
        rec = np.zeros(50, np.int32)
        s = M.spalten_fuer(rec, ["u1"], 4)
        self.assertEqual(s[25, 0], 1)

    def test_schrittweite_rand(self):
        rec = np.zeros(50, np.int32)
        s = M.spalten_fuer(rec, ["u1"], 4)
        self.assertEqual(s[22, 0], 0)   # Sekunde 88 < 90
        self.assertEqual(s[23, 0], 1)   # Sekunde 92 im Fenster
        self.assertEqual(s[28, 0], 0)   # Sekunde 112 > 110


if __name__ == "__main__":
    unittest.main()
