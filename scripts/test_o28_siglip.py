#!/usr/bin/env python3
"""Tests fuer o28-siglip.py: Zeilen-Ausrichtung und PCA nur aus train."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("o28", HIER / "o28-siglip.py")
o28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o28)


class Tests(unittest.TestCase):
    def test_zeilen_mit_schritt_und_fehlender_aufnahme(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            v = np.arange(10 * 768, dtype=np.float16).reshape(10, 768) % 7
            np.save(d / "a.npy", v)
            rec = np.array([0, 0, 0, 1, 1])
            S, da, mit = o28.siglip_zeilen(rec, ["a", "b"], 4, d)
            self.assertEqual(mit, {0})
            np.testing.assert_array_equal(da, [1, 1, 1, 0, 0])
            np.testing.assert_array_equal(S[:3], v[::4][:3].astype(np.float32))
            self.assertFalse(S[3:].any())

    def test_zu_kurz_zaehlt_nicht(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "a.npy", np.ones((2, 768), np.float16))
            S, da, mit = o28.siglip_zeilen(np.zeros(5, int), ["a"], 1, d)
            self.assertEqual(mit, set())
            self.assertFalse(da.any())

    def test_pca_ohne_merkmale_null_und_indikator(self):
        rng = np.random.default_rng(0)
        S_tr = rng.normal(size=(200, 768)).astype(np.float32)
        da_tr = np.r_[np.ones(150), np.zeros(50)].astype(np.float32)
        S_tr[da_tr == 0] = 0
        S_te = rng.normal(size=(20, 768)).astype(np.float32)
        da_te = np.r_[np.ones(10), np.zeros(10)].astype(np.float32)
        B_tr, B_te = o28.pca_block(S_tr, da_tr, S_te, da_te, k=8)
        self.assertEqual(B_tr.shape, (200, 9))
        self.assertFalse(B_te[10:, :8].any())
        np.testing.assert_array_equal(B_te[:, 8], da_te)
        # Zentrierung aus train-Zeilen MIT Merkmalen
        self.assertLess(np.abs(B_tr[:150, :8].mean(0)).max(), 1e-4)

    def test_pca_haengt_nicht_von_test_ab(self):
        rng = np.random.default_rng(1)
        S_tr = rng.normal(size=(100, 768)).astype(np.float32)
        da_tr = np.ones(100, np.float32)
        a, _ = o28.pca_block(S_tr, da_tr, rng.normal(size=(5, 768)).astype(np.float32),
                             np.ones(5, np.float32), k=4)
        b, _ = o28.pca_block(S_tr, da_tr, 100 * rng.normal(size=(5, 768)).astype(np.float32),
                             np.ones(5, np.float32), k=4)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
