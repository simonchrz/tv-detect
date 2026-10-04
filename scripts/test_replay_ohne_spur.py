#!/usr/bin/env python3
"""Aufnahmen ohne Decode-Spur: Gate misst mit dem Produktions-Dekoder (hsmm),
nicht mit der Schwelle (2026-10-04)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("th_ohne_spur", HIER / "train-head.py")
th = importlib.util.module_from_spec(spec)
sys.modules["th_ohne_spur"] = th
spec.loader.exec_module(th)


def _signal():
    p = np.full(1800, 0.03)
    p[600:840] = 0.97
    p[700:715] = 0.45  # kurzer Einbruch nahe der Schwelle
    return p


class Tests(unittest.TestCase):
    def test_nur_fuer_hsmm(self):
        alt = th.EVAL_DECODER
        th.EVAL_DECODER = []
        try:
            self.assertIsNone(th._replay_ohne_spur(_signal(), 1.0, "x"))
        finally:
            th.EVAL_DECODER = alt

    @unittest.skipUnless(th.TVD_BIN.exists(), "tv-detect nicht gebaut")
    def test_hsmm_ueberbrueckt_einbruch(self):
        b = th._replay_ohne_spur(_signal(), 1.0, "nicht-im-grid")
        self.assertEqual(len(b), 1, b)
        self.assertAlmostEqual(b[0][0], 600, delta=3)
        self.assertAlmostEqual(b[0][1], 840, delta=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
