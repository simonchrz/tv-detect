#!/usr/bin/env python3
"""Tests für den Nachlauf-Filter im Korpus-Label-Audit.

WARUM DIESER FILTER EINEN TEST BRAUCHT
--------------------------------------
Er UNTERDRÜCKT eine Meldung. Ein Filter, der zu viel wegnimmt, macht den
Wächter still, und ein stiller Wächter fällt niemandem auf — anders als
ein lauter Fehlalarm. Genau deshalb muss die Bedingung eng sein und
geprüft werden.

Der Nachlauf ist die Folgesendung am Aufnahmeende: läuft eine Aufnahme
über ihr geplantes Ende hinaus, hängt tv-recorder den Schwanz als Block
an, damit der Spieler ihn überspringt (`overrunBlock`, ads.go:273,
Ergebnis `[stop - start_real, duration]`). Der Kopf sagt dort zu Recht
„Sendung", das Label sagt „überspringen", beide haben recht.

Erkannt wird er an der Form, weil das Archiv weder `stop` noch
`start_real` kennt: **Block endet am Aufnahmeende UND ist mindestens 60 s
lang** — dieselbe Mindestlänge, die tv-recorder verlangt.

Ausführen: python3 scripts/test_nachlauf_maske.py
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cla", _HIER / "corpus-label-audit.py")
_cla = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cla)
maske = _cla.nachlauf_maske


class Nachlauf(unittest.TestCase):
    def test_block_am_ende_wird_erkannt(self):
        # 4106 s Aufnahme, Block [3808, 4106) -- der reale Fall
        # dvr-kabel-eins-1780856070.
        m = maske([[1266, 1729], [2812, 3295], [3808, 4106]], 4106)
        self.assertTrue(m[3900], "Nachlauf muss erkannt werden")
        self.assertFalse(m[1500], "fruehere Bloecke duerfen nicht mitgenommen werden")
        self.assertFalse(m[3000], "auch der zweite Block nicht")
        self.assertEqual(int(m.sum()), 4106 - 3808)

    def test_kurzer_endblock_zaehlt_nicht(self):
        # 40 s ist kein Nachlauf -- tv-recorder verlangt >= 60 s. Ein
        # kurzer Werbeblock am Aufnahmeende bleibt also messbar.
        m = maske([[4066, 4106]], 4106)
        self.assertFalse(m.any(), "unter 60 s ist kein Nachlauf")

    def test_block_der_nicht_ans_ende_reicht(self):
        # Endet 100 s vor Schluss: normaler Werbeblock, kein Nachlauf.
        m = maske([[3800, 4006]], 4106)
        self.assertFalse(m.any(), "nur ein Block AM Ende ist der Nachlauf")

    def test_kleine_rundungsluecke_ist_erlaubt(self):
        # Die Blockdauer kommt aus der Playlist, die Sekundenzahl aus den
        # Merkmalen; ein bis zwei Sekunden Versatz sind normal.
        m = maske([[3808, 4104]], 4106)
        self.assertTrue(m[4000], "2 s Rundungsluecke darf den Nachlauf nicht verstecken")

    def test_leer_und_none(self):
        self.assertFalse(maske([], 100).any())
        self.assertFalse(maske(None, 100).any())

    def test_ueberlange_werte_klemmen_nicht_ab(self):
        # Ein Block, der ueber die Merkmalslaenge hinausragt, darf keinen
        # IndexError werfen.
        m = maske([[50, 200]], 100)
        self.assertEqual(len(m), 100)
        self.assertTrue(m[60])

    def test_maske_steuert_nur_die_flagge_nicht_den_zaehler(self):
        """Die Maske darf NICHTS verschwinden lassen.

        Von vier Aufnahmen mit Endblock war eine
        (dvr-kabel-eins-1781539018, 742 s) nach den Bild-Ankern zu 43 %
        mit wiederholten Spots belegt — echte Werbung, kein Nachlauf. Die
        Form allein trennt das nicht. Deshalb wird der Phantom-Zaehler
        VOLL gefuehrt und nur die MELDESCHWELLE ohne den Endblock
        gerechnet. Ein stiller Filter waere schlimmer als der Fehlalarm,
        den er behebt.

        Geprueft wird strukturell, weil ein Rueckfall auf "phan" in der
        Flaggenzeile keine Ausnahme wirft, sondern nur den Fehlalarm
        zurueckbringt.
        """
        quelle = (_HIER / "corpus-label-audit.py").read_text()
        zeilen = [z.strip() for z in quelle.splitlines()]
        hole = [z for z in zeilen if z.startswith("hole = sum(")]
        phan = [z for z in zeilen if z.startswith("phan = sum(")]
        flag = [z for z in zeilen if z.startswith("phan_flag = sum(")]
        self.assertTrue(hole and phan and flag, "alle drei Zaehler muessen da sein")
        for z in hole:
            self.assertNotIn("gt_flag", z,
                             "hole darf die Endblock-Maske nicht benutzen")
        for z in phan:
            self.assertNotIn("gt_flag", z,
                             "der Phantom-ZAEHLER muss voll bleiben")
        for z in flag:
            self.assertIn("gt_flag", z,
                          "nur phan_flag laeuft ueber die beschnittene Maske")
        entscheid = [z for z in zeilen
                     if "> 2 * args.min_run" in z and "hole >" in z]
        self.assertTrue(entscheid, "Entscheidungszeile nicht gefunden")
        for z in entscheid:
            self.assertIn("phan_flag >", z,
                          "die MELDUNG muss an phan_flag haengen, nicht an phan")


class MenschenRegel(unittest.TestCase):
    """Phantom zaehlt nur, wo ein MENSCH das Label gesetzt hat.

    Auf einer maschinell gelabelten Aufnahme IST das Label die Ausgabe
    des Blockbildners; ein Widerspruch zwischen Logo und NN ist dort
    weder neu noch ein Fehler (siehe den langen Kommentar an der Stelle).
    Beim Einbau des Endblock-Filters wurde `phan_flag` von dieser Regel
    ausgenommen — und weil die MELDUNG daran haengt, meldete der Waechter
    ploetzlich maschinelle Aufnahmen mit "0" in der Phantom-Spalte.
    """

    def test_beide_zaehler_werden_genullt(self):
        quelle = (_HIER / "corpus-label-audit.py").read_text()
        i = quelle.index("if not human:")
        block = quelle[i:i + 200]
        self.assertIn("phan = 0", block, "der angezeigte Zaehler muss genullt werden")
        self.assertIn("phan_flag = 0", block,
                      "der Zaehler, an dem die MELDUNG haengt, ebenso — "
                      "sonst meldet der Waechter maschinelle Labels")


if __name__ == "__main__":
    unittest.main(verbosity=2)
