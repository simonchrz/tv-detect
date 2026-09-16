#!/usr/bin/env python3
"""Seed-Phase ohne Kopien: float64-Baseline-Matrix, gleicher Fingerabdruck,
Copy-on-Write-Archiv, Unsicherheit aus dem deployten MLP.

Umbauten vom 2026-09-16. Was sich lokal beweisen laesst, steht hier; die
Reihenfolge des Matrixbaus ist Ende-zu-Ende belegt (alter und neuer Code,
gleicher Stichtag, Fingerabdruck/Baseline/Seeds/Golden identisch).
"""
import hashlib
import re
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
SRC = (HIER / "train-head.py").read_text()
CODE = "\n".join(z for z in SRC.splitlines() if not z.lstrip().startswith("#"))


def _fp(a, dtype=None):
    """Woertlich die Rechenvorschrift aus train-head.py (verschachtelte def)."""
    h = hashlib.sha1()
    for lo in range(0, len(a), 65536):
        h.update(np.ascontiguousarray(a[lo:lo + 65536], dtype=dtype))
    return h.hexdigest()[:12]


class FingerabdruckBleibtGleich(unittest.TestCase):
    def test_float64_matrix_nach_float32_gehasht(self):
        r = np.random.default_rng(1)
        x32 = r.normal(size=(70001, 37)).astype(np.float32)
        x64 = x32.astype(np.float64)                    # exakt, wie _gestapelt
        alt = hashlib.sha1(np.ascontiguousarray(x32).tobytes()).hexdigest()[:12]
        self.assertEqual(_fp(x64, np.float32), alt)
        self.assertEqual(_fp(x32), alt)

    def test_vektoren_unveraendert(self):
        y = (np.random.default_rng(2).normal(size=200003) > 0).astype(np.float32)
        self.assertEqual(_fp(y), hashlib.sha1(y.tobytes()).hexdigest()[:12])

    def test_train_head_hasht_x_als_float32(self):
        self.assertIn("_fp(X_train, np.float32)", CODE)
        i = CODE.index("def _fp(a, dtype=None):")
        self.assertIn("np.ascontiguousarray(a[lo:lo + 65536], dtype=dtype)", CODE[i:i + 600])


class CopyOnWriteMapping(unittest.TestCase):
    def test_schreiben_trifft_nur_die_kopie(self):
        import tempfile
        d = Path(tempfile.mkdtemp()); f = d / "x.npy"
        a = np.random.default_rng(3).normal(size=(500, 1282)).astype(np.float32)
        a[10:20, 1280] = np.nan
        np.save(f, a)
        m = np.load(f, mmap_mode="c")
        nm = np.isnan(m[:, 1280])
        m[nm, 1280] = 0.5                                 # wie im Archiv-Pfad
        self.assertTrue(np.all(m[10:20, 1280] == 0.5))
        self.assertTrue(np.isnan(np.load(f)[10:20, 1280]).all(), "Datei muss unberuehrt bleiben")
        mask = np.zeros(500, bool); mask[::3] = True
        self.assertTrue(np.array_equal(m[mask], np.where(np.isnan(a), 0.5, a)[mask]))

    def test_archiv_laedt_gemappt(self):
        self.assertIn('a_feats = np.load(fnpy, mmap_mode="c")', CODE)
        self.assertNotIn("a_feats = a_feats.copy()", CODE)


class BaselineUndSeeds(unittest.TestCase):
    def test_baseline_matrix_float64_und_wieder_frei(self):
        self.assertIn("_gestapelt(X_train_parts, np.float64)", CODE)
        self.assertIn("train_pred = _vorhersage_blockweise(clf, X_train)", CODE)
        i = CODE.index("train_pred = _vorhersage_blockweise(clf, X_train)")
        self.assertIn("X_train = np.empty((0, _base_dim), dtype=np.float32)", CODE[i:i + 500])

    def test_seed_matrix_aus_rohdaten_ohne_hstack(self):
        self.assertNotIn("X_train_ch = np.hstack", CODE)
        self.assertIn("_gestapelt(_basis_parts, np.float32, zusatz_parts)", CODE)
        self.assertIn("len(X_train_ch) != len(y_train)", CODE, "Zeilen-Wache fehlt")


class UnsicherheitAusDemMLP(unittest.TestCase):
    def test_logreg_voll_fit_nur_fuer_logreg_produktion(self):
        self.assertIn("_logreg_ist_produktion = not (wants_mlp and mlp_prod_clf is not None)", CODE)
        self.assertIn("if args.final_on_all and test_recs and _logreg_ist_produktion:", CODE)

    def test_bericht_nutzt_deployten_kopf(self):
        self.assertIn("proba = mlp_prod_clf.predict_proba(Xa)[:, 1]", CODE)
        # Kalibrierung genau wie head.calibration.json: nur wenn "applied".
        self.assertIn('calibration.get("applied")', CODE)
        # und derselbe Kanal-Index wie der ausgelieferte Kopf
        self.assertIn("enumerate(mlp_prod_chan_slugs)", CODE)


if __name__ == "__main__":
    unittest.main()
