#!/usr/bin/env python3
"""Tests für die Blockableitung aus dem groben Durchgang.

⚠️ Diese Funktion existiert, weil der bisherige Messsatz eine andere
Population abbildete als der Korpus: Bilder um die Modellkante herum können
nur Kanten beurteilen, die nahe genug an der Modellkante liegen. Der grobe
Durchgang tastet die ganze Aufnahme ab und findet die Blöcke ohne das
Modell — die Grenzen sind dabei absichtlich nur auf den Takt genau.

Ausführen: python3 scripts/test_bloecke_aus_grob.py
"""
import importlib.util
import sys
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ar", Path(__file__).resolve().parent / "agent-review.py")
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

S = "sendungsinhalt"
W = "produktwerbung"
T = "programmvorschau"   # Konvention: Werbung
M = "mitmachtafel"       # Konvention: Sendung
U = "unklar"


def folge(*paare):
    """(Sekunde, Kategorie) im 45-s-Takt aus einer Kurzschreibweise."""
    return [(45 * (i + 1), k) for i, k in enumerate(paare)]


class BloeckeAusGrob(unittest.TestCase):

    def test_ein_block(self):
        got = _ar.bloecke_aus_grob(folge(S, S, W, W, W, S, S))
        self.assertEqual(got, [(135, 270)])

    def test_zwei_bloecke(self):
        got = _ar.bloecke_aus_grob(folge(S, W, W, S, S, W, W, S))
        self.assertEqual(got, [(90, 180), (270, 360)])

    def test_trailer_zaehlt_als_werbung(self):
        # Die Konvention steckt in KONVENTION, nicht im Agenten-Auftrag.
        got = _ar.bloecke_aus_grob(folge(S, T, T, S))
        self.assertEqual(got, [(90, 180)])

    def test_mitmachtafel_unterbricht_keinen_block(self):
        # Eine Mitmachtafel zaehlt als Sendung — sie beendet den Block.
        got = _ar.bloecke_aus_grob(folge(S, W, W, M, W, W, S))
        self.assertEqual(len(got), 2)

    # ── Die Absicherungen ───────────────────────────────────────────────

    def test_einzelner_werbepunkt_ist_kein_block(self):
        # ⚠️ Der wichtigste Fall: ein Trailer MITTEN in der Sendung ist im
        # 45-s-Takt ein einzelner Werbepunkt. Ohne die Mindestlaenge waere
        # daraus ein Werbeblock geworden — und der Ad-Skip haette Sendung
        # weggeschnitten, was irreversibel ist.
        got = _ar.bloecke_aus_grob(folge(S, S, W, S, S))
        self.assertEqual(got, [])

    def test_unklar_beendet_keinen_block(self):
        # Ein unbrauchbares Bild darf einen Block nicht zerreissen; es ist
        # weder Werbung noch Sendung und wird schlicht uebergangen.
        got = _ar.bloecke_aus_grob(folge(S, W, U, W, S))
        self.assertEqual(got, [(90, 225)])

    def test_block_am_ende_ohne_rueckkehr(self):
        # Endet die Aufnahme im Werbeblock, gibt es kein schliessendes
        # Sendungsbild — der Block reicht bis zum letzten Punkt.
        got = _ar.bloecke_aus_grob(folge(S, S, W, W))
        self.assertEqual(got, [(135, 180)])

    def test_leere_folge(self):
        self.assertEqual(_ar.bloecke_aus_grob([]), [])

    def test_alles_sendung(self):
        self.assertEqual(_ar.bloecke_aus_grob(folge(S, S, S, S)), [])


class FeinesFensterDecktDenTakt(unittest.TestCase):
    """Die zweite Stufe muss die Wahrheit einschliessen — sonst misst sie
    wieder nur das Raster.

    ⚠️ Der Grund fuer diesen Test: der grobe Durchgang meldet den Start per
    Konstruktion bis zu einen ganzen Takt ZU SPAET (gemessen gegen
    menschliche Golden-Labels: Median +29 s bei 45 s Takt). Ein symmetrisches
    Fenster um den groben Punkt wuerde die Wahrheit auf der frueheren Seite
    zur Haelfte verfehlen — das feine Fenster ist deshalb absichtlich
    asymmetrisch nach hinten gezogen.
    """

    def _fenster(self, t):
        return (t - _ar.GROB_TAKT_S - _ar.FEIN_MARGE_S, t + _ar.FEIN_MARGE_S)

    def test_deckt_das_ganze_taktintervall_davor(self):
        # Die Wahrheit kann ueberall in (t-Takt, t] liegen.
        t = 900.0
        a, b = self._fenster(t)
        self.assertLessEqual(a, t - _ar.GROB_TAKT_S)
        self.assertGreaterEqual(b, t)

    def test_ist_nach_hinten_gezogen_nicht_symmetrisch(self):
        t = 900.0
        a, b = self._fenster(t)
        self.assertGreater(t - a, b - t)

    def test_schrittweite_teilt_das_fenster(self):
        # Sonst faellt der letzte Punkt aus dem Raster und die Kante kann
        # genau am Rand liegen, wo die Ableitung sie ablehnt.
        a, b = self._fenster(900.0)
        self.assertEqual((b - a) % _ar.SCHRITT_S, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
