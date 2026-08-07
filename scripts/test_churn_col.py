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
import re
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


class TeacherBreiten(unittest.TestCase):
    """_augment_teacher_feats muss fuer JEDE bekannte Kopfform exakt die
    Breite liefern, die der Header-Vertrag vorsieht.

    ⚠️ Genau hier ist am 2026-08-06 etwas still kaputtgegangen: die
    Lehrer-Erkennung matchte input_dim gegen eine LISTE bekannter Formen,
    und als der v5-Kopf dazukam, passte keine mehr. Der Lehrer fiel auf
    "unalignable", der Hygiene-Durchlauf lief ohne ihn weiter — kein
    Absturz, nur eine stumm abgeschaltete Pruefung. Deshalb wird die Breite
    jetzt gerechnet, und dieser Test haelt die Rechnung fest.

    Spaltenreihenfolge (Header-Vertrag):
      basis | kanal | whisper | temporal(2|3) | minuteprior | maske
    """

    def _breite(self, **kw):
        T, basis, n_chan = 40, 6, 3
        X = _rampe(T, breite=basis)
        chan_idx = {"alpha": 0, "beta": 1, "gamma": 2}
        Xa = th._augment_teacher_feats(X, "alpha", chan_idx, "dvr-x-1", **kw)
        self.assertEqual(Xa.shape[0], T)
        return Xa.shape[1] - basis - n_chan   # die Zusatzspalten

    def test_v2_nur_whisper(self):
        self.assertEqual(self._breite(wants_whisper=True), 1)

    def test_v3_whisper_temporal(self):
        self.assertEqual(
            self._breite(wants_whisper=True, wants_temporal=True), 3)

    def test_v4_plus_minuteprior(self):
        mp = lambda uuid, T: np.zeros((T, 1), np.float32)
        self.assertEqual(
            self._breite(wants_whisper=True, wants_temporal=True,
                         mp_col=mp), 4)

    def test_v5_temporal2_mit_maske(self):
        mp = lambda uuid, T: np.zeros((T, 1), np.float32)
        self.assertEqual(
            self._breite(wants_whisper=True, wants_temporal=True,
                         mp_col=mp, wants_mask=True), 5)

    def test_v5_temporal3_mit_churn_und_maske(self):
        mp = lambda uuid, T: np.zeros((T, 1), np.float32)
        self.assertEqual(
            self._breite(wants_whisper=True, wants_temporal=True,
                         mp_col=mp, wants_churn=True, wants_mask=True), 6)

    def test_churn_ohne_temporal_ist_wirkungslos(self):
        """Die Unruhe haengt am Temporal-Block; ohne den gibt es sie nicht."""
        self.assertEqual(
            self._breite(wants_whisper=True, wants_churn=True), 1)


class FensterbreiteStimmtUeberall(unittest.TestCase):
    """Die Fensterbreite steht noch an ZWEI Stellen — einmal je Sprache.

    ⚠️ Bis zum 2026-08-07 waren es drei: corpus-label-audit.py hatte eine
    eigene Kopie. Die haengt jetzt an train-head.py (s. test_audit_hat_keine_
    eigene_kopie), es bleibt also nur die Sprachgrenze. Laufen die beiden
    auseinander, sieht der Kopf im Betrieb eine andere Spalte als im
    Training — und das aeussert sich als leicht schlechteres Modell, nicht
    als Fehler. Der Test liest die Go-Konstante aus dem Quelltext, weil es
    keinen anderen Weg gibt, die beiden Sprachen aneinander zu binden."""

    def _go_konstante(self):
        p = Path(__file__).resolve().parent.parent / "internal/signals/nn.go"
        m = re.search(r"^const churnWindowS = (\d+)$", p.read_text(), re.M)
        self.assertIsNotNone(m, "churnWindowS nicht in nn.go gefunden — "
                                "umbenannt? Dann hier nachziehen.")
        return int(m.group(1))

    def test_go_und_python_gleich(self):
        import inspect
        vorgabe = inspect.signature(th._churn_col).parameters["fenster"].default
        self.assertEqual(
            vorgabe, self._go_konstante(),
            "train-head.py _churn_col und nn.go churnWindowS unterscheiden "
            "sich — stiller Train/Serve-Bruch")

    def test_audit_hat_keine_eigene_kopie(self):
        """Das Audit darf die Spalten NICHT selbst bauen. Tat es bis
        2026-08-07, und lief dabei zweimal von der Produktion weg."""
        p = Path(__file__).resolve().parent / "corpus-label-audit.py"
        quelle = p.read_text()
        self.assertNotIn(
            "def _churn(", quelle,
            "corpus-label-audit.py hat wieder eine eigene Unruhe-Spalte")
        self.assertNotIn(
            "def whisper_per_sec(", quelle,
            "corpus-label-audit.py hat wieder einen eigenen Whisper-Lader")
        self.assertIn(
            "TH.mit_zusatz(", quelle,
            "corpus-label-audit.py baut die Eingabe nicht mehr ueber "
            "train-head.py — dann urteilt es mit einer anderen Eingabe "
            "als die Produktion")


if __name__ == "__main__":
    unittest.main(verbosity=2)
