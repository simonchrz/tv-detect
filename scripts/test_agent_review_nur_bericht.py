#!/usr/bin/env python3
"""Der Berichtsmodus darf Test-Aufnahmen klassifizieren, aber NIE beschreiben.

Anlass 2026-09-14: 29 reviewbare Test-Aufnahmen ohne menschliches Review
sind die einzige Stellschraube fuer den Golden-Satz. Agenten duerfen dort
nicht labeln (Schutzkette vom 06.09.), aber sie duerfen dem Menschen zeigen,
wo ein Block vermutlich falsch liegt. Dafuer gibt es --nur-bericht — und
zwei Sperren, damit daraus nie ein Label wird.
"""
import re
import unittest
from pathlib import Path

QUELL = (Path(__file__).resolve().parent / "agent-review.py").read_text()


def _funktion(name):
    i = QUELL.index(f"def {name}(")
    j = QUELL.find("\ndef ", i + 1)
    return QUELL[i:j if j > 0 else None]


class ZweiSperren(unittest.TestCase):
    def test_vorbereiten_markiert_den_ordner(self):
        v = _funktion("vorbereiten")
        self.assertIn('(ziel / "NUR-BERICHT").write_text(', v)

    def test_anwenden_verweigert_markierte_ordner(self):
        a = _funktion("anwenden")
        self.assertIn('(d / "NUR-BERICHT").is_file()', a)
        # Die Sperre muss VOR jedem Schreibpfad stehen — also vor veraltet()
        # und dem POST.
        self.assertLess(a.index('"NUR-BERICHT"'), a.index("veraltet("))

    def test_train_sperre_bleibt_zusaetzlich(self):
        # Die Markierung ersetzt die Ledger-Pruefung nicht, sie ergaenzt sie.
        a = _funktion("anwenden")
        self.assertIn("not in erlaubt", a)
        self.assertLess(a.index("not in erlaubt"), a.index('"NUR-BERICHT"'))


class KeineAutomatischeAuswahl(unittest.TestCase):
    def test_nur_mit_uuid_liste(self):
        v = _funktion("vorbereiten")
        i = v.index("if nur_bericht:")
        self.assertIn("if not uuids:", v[i:i + 1200],
                      "ohne uuid-Liste waere das die Tuer vom 06.09.")

    def test_golden_bleibt_auch_im_berichtsmodus_tabu(self):
        v = _funktion("vorbereiten")
        i = v.index("if nur_bericht:")
        self.assertIn("golden_uuids()", v[i:i + 1500])

    def test_nur_fuer_kantenfenster(self):
        self.assertIn("a.nur_bericht and (a.grob or a.fein or a.anwenden", QUELL)

    def test_ledger_unlesbar_bleibt_fail_closed(self):
        # Auch im Berichtsmodus: kein Ledger, kein Lauf.
        v = _funktion("vorbereiten")
        self.assertLess(v.index("if _erlaubt is None:"), v.index("if nur_bericht:"))


if __name__ == "__main__":
    unittest.main()
