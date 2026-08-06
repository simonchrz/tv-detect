#!/usr/bin/env python3
"""Tests fuer _churn_col — die Unruhe-Spalte des Temporal-Blocks.

⚠️ Der wichtigste Test hier ist der auf KURZE Aufnahmen. Die erste Fassung
benutzte np.convolve(mode="same"), und das gibt die Laenge des LAENGEREN
Arrays zurueck: bei einer Aufnahme mit 7 Sekunden und einem 31er-Fenster
also 31 Werte statt 7. Das Training starb erst Minuten spaeter beim
column_stack, mit einer Fehlermeldung, die auf die Ursache nicht hinwies
(2026-08-06, ein verlorener Trainingslauf).

Der zweite Test sichert die Paritaet zur Go-Seite: dort laeuft in
nn.go confidenceMLPChunk eine geclippte Fensterschleife. Laufen die beiden
auseinander, sieht der Kopf im Betrieb eine andere Spalte als im Training —
und das sieht in der Produktion aus wie ein leicht schlechteres Modell,
nicht wie ein Fehler.

Ausfuehren: python3 scripts/test_churn_col.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "th", str(Path(__file__).resolve().parent / "train-head.py"))
th = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ["train-head"]
try:
    _spec.loader.exec_module(th)
except SystemExit:
    pass
finally:
    sys.argv = _argv


def _rampe(T, breite=4):
    """Features, deren 1s-Delta konstant ist — macht die Sollwerte
    nachrechenbar."""
    return np.cumsum(np.ones((T, breite), np.float32), axis=0)


class ChurnLaenge(unittest.TestCase):
    def test_laenge_stimmt_fuer_jede_aufnahme(self):
        """Auch wenn die Aufnahme KUERZER ist als das Fenster."""
        for T in (1, 2, 5, 7, 15, 30, 31, 32, 60, 200, 3900):
            with self.subTest(T=T):
                c = th._churn_col(_rampe(T))
                self.assertEqual(
                    c.shape, (T, 1),
                    f"Aufnahme mit {T} Sekunden ergibt {c.shape[0]} Werte — "
                    f"np.convolve(mode='same') zurueck im Spiel?")

    def test_eine_einzige_sekunde(self):
        c = th._churn_col(_rampe(1))
        self.assertEqual(c.shape, (1, 1))
        self.assertEqual(float(c[0, 0]), 0.0)  # kein Vorgaenger, kein Delta


class ChurnParitaet(unittest.TestCase):
    def test_gegen_die_go_schleife(self):
        """Dieselben Werte wie die geclippte Fensterschleife in nn.go."""
        werte = np.array([0, 1, 3, 6, 10, 15, 21, 28],
                         np.float32).reshape(-1, 1)
        c = th._churn_col(werte, fenster=5)[:, 0]
        dp = np.concatenate([[0.0], np.diff(werte[:, 0])])
        erwartet = []
        for i in range(len(dp)):
            fenster = [dp[j] for j in range(i - 2, i + 3) if 0 <= j < len(dp)]
            erwartet.append(sum(fenster) / len(fenster))
        np.testing.assert_allclose(c, erwartet, atol=1e-6)

    def test_rand_wird_normiert_nicht_genullt(self):
        """Am Rand durch die VORHANDENEN Werte teilen, nicht durch die
        Fensterbreite.

        Nullauffuellung zoege die Unruhe am Rand nach unten — und zu wenig
        Unruhe liest der Kopf als Sendung. Das waere eine gerichtete
        Verzerrung, kein blosses Rauschen."""
        # breite=1, damit der L2-Abstand gleich der skalaren Differenz ist
        # (bei 4 Spalten waere er sqrt(4)=2 — daran ist die erste Fassung
        # dieses Tests gescheitert, nicht am Code).
        c = th._churn_col(_rampe(100, breite=1), fenster=31)[:, 0]
        # Deltas sind konstant 1.0 ab Sekunde 1, Sekunde 0 ist 0.
        # In der Mitte also exakt 1.0.
        self.assertAlmostEqual(float(c[50]), 1.0, places=5)
        # Am linken Rand: Fenster [0..15], enthaelt die eine Null.
        self.assertAlmostEqual(float(c[0]), 15.0 / 16.0, places=5)
        # Mit Nullauffuellung waere es 15/31 ≈ 0.484 — deutlich daneben.
        self.assertGreater(float(c[0]), 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
