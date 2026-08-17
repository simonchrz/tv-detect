#!/usr/bin/env python3
"""Tests für die Kantenableitung aus Einzelbild-Urteilen.

⚠️ Diese Funktion ist der Ersatz für eine Frage, die drei von drei
Fehlurteilen erzeugt hat („bei welcher Sekunde ist der Übergang" — sie hat
immer eine Antwort, auch wenn die Grenze gar nicht im Bild ist). Der Wert
der Ablösung liegt ausschließlich darin, dass sie NEIN sagen kann. Die
Tests prüfen deshalb vor allem die Ablehnungen; dass sie bei einer sauberen
Folge die richtige Zahl liefert, ist der einfachere Teil.

Ausführen: python3 scripts/test_kante_aus_folge.py
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
T = "programmvorschau"   # zählt per Konvention als Werbung
M = "mitmachtafel"       # zählt per Konvention als Sendung
U = "unklar"


class KanteAusFolge(unittest.TestCase):

    def test_sauberer_blockstart(self):
        # Sendung, Sendung, dann Werbung → Kante beim ersten Werbebild.
        folge = [(100, S), (104, S), (108, W), (112, W), (116, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(grund)
        self.assertEqual(kante, 108)

    def test_sauberes_blockende(self):
        folge = [(200, W), (204, W), (208, S), (212, S), (216, S)]
        kante, grund = _ar.kante_aus_folge(folge, "ende")
        self.assertIsNone(grund)
        self.assertEqual(kante, 208)

    def test_trailer_zaehlt_als_werbung(self):
        # Die Konvention steckt im Code, nicht im Auftrag: ein Trailer vor
        # dem Sendungsbeginn verschiebt die Kante nach hinten.
        folge = [(200, W), (204, T), (208, T), (212, S), (216, S)]
        kante, grund = _ar.kante_aus_folge(folge, "ende")
        self.assertIsNone(grund)
        self.assertEqual(kante, 212)

    def test_mitmachtafel_zaehlt_als_werbung(self):
        # ⚠️ Bis zum 2026-08-17 stand hier „zaehlt als Sendung" und die Kante
        # lag auf 108. Simon hat die Konvention an diesem Tag
        # zurueckgenommen — der Gewinnspiel-Insert ist Werbung, die Kante
        # liegt damit VOR ihm. Genau diese Zuordnung hatte 12 von 22
        # bearbeiteten Agenten-Bloecken um ~30 s verschoben (Ledger §3am).
        folge = [(92, S), (96, S), (100, M), (104, M), (108, W), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(grund)
        self.assertEqual(kante, 100)

    # ── Die Ablehnungen: der eigentliche Zweck ──────────────────────────

    def test_kein_wechsel_im_fenster(self):
        # Genau der Fall, in dem der Agent frueher trotzdem antwortete:
        # alles Werbung, die echte Grenze liegt ausserhalb.
        folge = [(100, W), (104, W), (108, W), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        self.assertIn("kein Wechsel", grund)

    def test_wechsel_am_fensterrand_zaehlt_nicht(self):
        # Wechsel beim letzten Bild: die Grenze koennte genauso gut spaeter
        # liegen. Lieber keine Kante als eine geratene.
        folge = [(100, S), (104, S), (108, S), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        self.assertIn("Fensterrand", grund)

    def test_hin_und_her_wird_verworfen(self):
        folge = [(100, S), (104, W), (108, S), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        self.assertIn("Wechsel", grund)

    def test_falsche_richtung(self):
        # Bei einem Blockstart muss Sendung → Werbung stehen, nicht umgekehrt.
        folge = [(100, W), (104, W), (108, S), (112, S)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        self.assertIn("Richtung", grund)

    def test_unklar_am_uebergang_blockiert(self):
        # Ein "unklar" genau an der Grenze macht sie unbestimmt.
        folge = [(100, S), (104, U), (108, W), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        # ⚠️ Der GRUND muss "unklar am Uebergang" sein und nicht "kein
        # Wechsel": davon haengt ab, ob nachgefasst wird mit weiterem
        # Fenster (Grenze liegt draussen) oder mit feinerer Abtastung
        # (Grenze liegt hier, nur zu grob getroffen). Am 2026-08-16 lief
        # dieser Fall in die falsche Abhilfe.
        self.assertIn("unklar", grund)

    def test_unklar_ausserhalb_des_uebergangs_meldet_kein_wechsel(self):
        # Alles Werbung, ein unbrauchbares Bild dazwischen: die Grenze liegt
        # wirklich ausserhalb, hier hilft nur ein weiteres Fenster.
        folge = [(100, W), (104, U), (108, W), (112, W)]
        kante, grund = _ar.kante_aus_folge(folge, "start")
        self.assertIsNone(kante)
        self.assertIn("kein Wechsel", grund)

    def test_zu_wenige_bilder(self):
        kante, grund = _ar.kante_aus_folge([(100, S), (104, W)], "start")
        self.assertIsNone(kante)
        self.assertIn("zu wenige", grund)


if __name__ == "__main__":
    unittest.main(verbosity=2)
