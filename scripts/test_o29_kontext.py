#!/usr/bin/env python3
"""Tests fuer o29-kontext.py: Titelliste nur aus train, Mittel konstant je Aufnahme."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("o29", HIER / "o29-kontext.py")
o29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o29)


class Tests(unittest.TestCase):
    def test_titel_nur_aus_train_und_sonstige(self):
        tt = {"a": "X", "b": "X", "c": "X", "d": "Y", "e": "Y", "f": "Z"}
        rec_tr = np.array([0, 1, 2, 3])
        B_tr, B_te, haeufig = o29.titel_block(rec_tr, ["a", "b", "c", "d"], np.array([0, 0, 1]),
                                              ["f", "e"], titel_von=tt.get)
        self.assertEqual(haeufig, ["X"])
        np.testing.assert_array_equal(B_tr.argmax(1), [0, 0, 0, 1])
        np.testing.assert_array_equal(B_te, [[0, 1], [0, 1], [0, 1]])

    def test_mittel_konstant_je_aufnahme(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in range(5):
                np.save(d / f"u{i}.npy", rng.normal(size=(20, 768)).astype(np.float16))
            rec_tr = np.repeat(np.arange(5), 3)
            B_tr, B_te = o29.mittel_block(rec_tr, [f"u{i}" for i in range(5)],
                                          np.array([0, 0, 1]), ["u0", "fehlt"], d, k=3)
            self.assertEqual(B_tr.shape, (15, 4))
            for i in range(5):
                self.assertTrue((B_tr[rec_tr == i] == B_tr[rec_tr == i][0]).all())
            np.testing.assert_array_equal(B_te[2], [0, 0, 0, 0])
            np.testing.assert_array_equal(B_te[:2, 3], [1, 1])


if __name__ == "__main__":
    unittest.main()
