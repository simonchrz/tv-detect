#!/usr/bin/env python3
"""Die Schlussphase (Voll-Anpassungen) muss ohne Matrix-Kopien dasselbe rechnen.

Drei Umbauten vom 2026-09-16, jeder hier gegen die alte Rechnung gestellt:
  * `_gestapelt`: vorbelegte Matrix statt Teile-Liste + concatenate/hstack.
  * `_vorhersage_blockweise`: sklearn-predict ueber Zeilenbloecke.
  * Gewicht-0-Zeilen bleiben in der Matrix; WeightedMLP.fit filtert selbst
    (frueher `X_all_ch = X_all_ch[nz]`, eine 16-GB-Kopie).
"""
import re
import tracemalloc
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HIER = Path(__file__).resolve().parent
SRC = (HIER / "train-head.py").read_text()


def _funktion(name):
    m = re.search(rf"(?m)^def {name}\(", SRC)
    e = re.search(r"(?m)^(?:def |class )", SRC[m.start() + 1:])
    ns = {"np": np}
    exec(SRC[m.start():m.start() + 1 + e.start()], ns)
    return ns[name]


def _klasse():
    m = re.search(r"(?m)^class WeightedMLP\b", SRC)
    e = re.search(r"(?m)^(?:def |class )", SRC[m.start() + 1:])
    ns = {"np": np}
    exec(SRC[m.start():m.start() + 1 + e.start()], ns)
    return ns["WeightedMLP"]


gestapelt = _funktion("_gestapelt")
vorhersage_blockweise = _funktion("_vorhersage_blockweise")
MLP = _klasse()


class Gestapelt(unittest.TestCase):
    def _bloecke(self, dtype):
        r = np.random.default_rng(1)
        return [r.normal(size=(n, 7)).astype(dtype) for n in (5, 1, 300, 42)]

    def test_wie_concatenate(self):
        b = self._bloecke(np.float32)
        self.assertTrue(np.array_equal(gestapelt(b, np.float32), np.concatenate(b)))

    def test_float32_nach_float64_exakt(self):
        b = self._bloecke(np.float32)
        self.assertTrue(np.array_equal(gestapelt(b, np.float64),
                                       np.concatenate(b).astype(np.float64)))

    def test_float64_nach_float32_wie_astype(self):
        b = self._bloecke(np.float64)
        self.assertTrue(np.array_equal(gestapelt(b, np.float32),
                                       np.concatenate(b).astype(np.float32)))

    def test_mit_zusatzbloecken_wie_hstack(self):
        b = self._bloecke(np.float32)
        r = np.random.default_rng(2)
        z = [r.normal(size=(len(x), 3)).astype(np.float32) for x in b]
        alt = np.concatenate([np.hstack([x, zz]).astype(np.float32) for x, zz in zip(b, z)])
        self.assertTrue(np.array_equal(gestapelt(b, np.float32, z), alt))

    def test_leer(self):
        self.assertEqual(gestapelt([], np.float32).shape, (0, 0))


class VorhersageBlockweise(unittest.TestCase):
    def test_gleich_wie_am_stueck(self):
        r = np.random.default_rng(3)
        X = r.normal(size=(70001, 12)).astype(np.float64)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = LogisticRegression(max_iter=100).fit(X, y)
        self.assertTrue(np.array_equal(vorhersage_blockweise(clf, X, block=1000), clf.predict(X)))

    def test_sklearn_kopiert_float64_nicht(self):
        # Die Begruendung fuer das float64-Vorbelegen: eine C-zusammenhaengende
        # float64-Matrix uebernimmt sklearn ohne eigene Kopie. Waere das
        # falsch, laege der Gipfel bei ~2x X.
        r = np.random.default_rng(4)
        X = r.normal(size=(150000, 64)).astype(np.float64)
        y = (X[:, 0] > 0).astype(int)
        tracemalloc.start()
        LogisticRegression(max_iter=20).fit(X, y)
        _, spitze = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(spitze, 0.5 * X.nbytes,
                        f"sklearn hat offenbar kopiert: Spitze {spitze/2**20:.0f} MB bei X={X.nbytes/2**20:.0f} MB")


class NullgewichteBleibenInDerMatrix(unittest.TestCase):
    def test_fit_filtert_identisch(self):
        r = np.random.default_rng(5)
        X = r.normal(size=(5000, 30)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        w = r.uniform(0.2, 2.0, 5000).astype(np.float32)
        w[r.choice(5000, 400, replace=False)] = 0.0
        nz = w > 0
        a = MLP(hidden_dim=8, random_state=2, max_iter=10).fit(X, y, w)
        b = MLP(hidden_dim=8, random_state=2, max_iter=10).fit(X[nz], y[nz], w[nz])
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(a.coefs_, b.coefs_)))
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(a.intercepts_, b.intercepts_)))
        self.assertEqual((a.n_iter_, a.loss_), (b.n_iter_, b.loss_))
        # und die gemeldete Genauigkeit ueber die gewichteten Zeilen ist gleich
        self.assertEqual((a.predict(X)[nz] == y[nz]).mean(), (b.predict(X[nz]) == y[nz]).mean())


class QuelltextWache(unittest.TestCase):
    # Nur Code zaehlt: die Kommentare nennen die entfernten Zeilen woertlich.
    CODE = "\n".join(z for z in SRC.splitlines() if not z.lstrip().startswith("#"))

    def test_kopien_sind_weg(self):
        self.assertNotIn("X_all_ch = X_all_ch[nz]", self.CODE)
        self.assertNotIn("X_all = np.concatenate([r[3] for r in keep])", self.CODE)
        self.assertNotIn("full_acc = (clf.predict(X_all) == y_all).mean()", self.CODE)
        self.assertEqual(self.CODE.count("_gestapelt("), 3, "zwei Aufrufe + Definition")


if __name__ == "__main__":
    unittest.main()
