#!/usr/bin/env python3
"""Die Ausschlussliste hat EINE Definition, und alle drei Leser benutzen sie.

Anlass 2026-09-14: TEST_SET_EXCLUDE lebte nur in train-head.py. Die beiden
Audits leiteten den Test-Eimer aus dem Split-Ledger ab und zaehlten 12
ausgeschlossene oder quarantaenierte Aufnahmen als Massstab -- der
Review-Hebel bot eine am 07.09. quarantaenierte Let's-Dance-Aufnahme als
Golden-Kandidaten an.
"""
import importlib.util
import re
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent


def quelle(name):
    return (HIER / name).read_text()


class EineDefinition(unittest.TestCase):
    def test_liste_nur_im_modul(self):
        for name in ("train-head.py", "golden_v3_vorschlag.py", "massstab-audit.py"):
            self.assertNotRegex(quelle(name), r"TEST_SET_EXCLUDE\s*=\s*\{",
                                f"{name} definiert die Liste selbst — zwei Wahrheiten")
        # ⚠️ MULTILINE: ohne das ankert ^ am Anfang der ganzen Datei, und
        # der Test schlaegt fehl, obwohl die Liste auf Modulebene steht.
        self.assertRegex(quelle("test_ausschluss.py"),
                         re.compile(r"^TEST_SET_EXCLUDE\s*=\s*\{", re.MULTILINE),
                         "das Modul muss die Liste auf Modulebene tragen")

    def test_alle_drei_leser_laden_das_modul(self):
        for name in ("train-head.py", "golden_v3_vorschlag.py", "massstab-audit.py"):
            self.assertIn('"test_ausschluss.py"', quelle(name), name)

    def test_audits_ziehen_die_liste_ab(self):
        self.assertIn("_ta.ausgeschlossen(ARCHIV)", quelle("golden_v3_vorschlag.py"))
        self.assertIn("_ta.ausgeschlossen(ARCHIV)", quelle("massstab-audit.py"))
        self.assertIn("and u not in _raus", quelle("massstab-audit.py"))

    def test_trainer_benutzt_das_modul(self):
        self.assertIn("TEST_SET_EXCLUDE = _ta.TEST_SET_EXCLUDE", quelle("train-head.py"))


class ModulVerhalten(unittest.TestCase):
    def setUp(self):
        s = importlib.util.spec_from_file_location("ta", HIER / "test_ausschluss.py")
        self.m = importlib.util.module_from_spec(s); s.loader.exec_module(self.m)

    def test_quarantaene_ohne_ordner_ist_leer(self):
        self.assertEqual(self.m.quarantaene(Path("/nonexistent/archiv")), set())

    def test_ausgeschlossen_ist_obermenge_der_liste(self):
        a = self.m.ausgeschlossen(Path("/nonexistent/archiv"))
        self.assertTrue(set(self.m.TEST_SET_EXCLUDE) <= a)

    def test_lets_dance_steht_drauf(self):
        # Simons Quarantaene-Entscheid vom 07.09. — Guertel und Hosentraeger.
        self.assertIn("dvr-rtl-1779473700", self.m.TEST_SET_EXCLUDE)


if __name__ == "__main__":
    unittest.main()
