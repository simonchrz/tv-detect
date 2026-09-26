#!/usr/bin/env python3
"""O27: die halbe Logo-Spalte muss zeilengenau zur unterabgetasteten Matrix
passen (Sekunden 0, schritt, 2*schritt …) — ein Versatz waere ein stiller
Fehler, der den Versuchsarm schlechter aussehen liesse."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("o27l", HIER / "o27-logo-halb.py")
M = importlib.util.module_from_spec(spec)
sys.modules["o27l"] = M
spec.loader.exec_module(M)


class HalbeSpalte(unittest.TestCase):
    def test_zeilengenau_mit_schritt(self):
        cache = Path(tempfile.mkdtemp())
        np.save(cache / "a.npy", np.arange(40, dtype=np.float32) / 100)  # 40 s
        v = np.full(20, np.nan, np.float32); v[4] = 0.9
        np.save(cache / "b.npy", v)
        # Aufnahme a: 10 Zeilen (schritt 4), b: 5 Zeilen; c ohne Cache
        rec = np.array([0] * 10 + [1] * 5 + [2] * 3)
        X = np.zeros((18, 1282), np.float32)
        X2, ers = M.halbe_spalte(X, rec, ["a", "b", "c"], 4, cache)
        self.assertEqual(ers, {0, 1})
        np.testing.assert_allclose(X2[:10, 1280], np.arange(0, 40, 4) / 100)
        self.assertAlmostEqual(float(X2[11, 1280]), 0.9, places=5)  # b: Sekunde 4 = Zeile 1
        self.assertEqual(float(X2[10, 1280]), 0.5)   # NaN -> 0.5
        self.assertTrue((X2[15:, 1280] == 0).all())  # c unveraendert
        self.assertTrue((X[:, 1280] == 0).all())     # Original unberuehrt


if __name__ == "__main__":
    unittest.main()
