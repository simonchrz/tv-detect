#!/usr/bin/env python3
"""Auto-Confirm-Filter von review-effort.py: live vom Pi, Spiegel als Rueckfall.

Befund 2026-09-05: der Spiegel (04:32) kannte die Fingerprint-Bestaetigungen
von 08:00 nicht, der Tagesdurchgang (08:07) zaehlte sie als Mensch mit 0 s/h.
"""
import importlib.util
import unittest
from pathlib import Path

_p = Path(__file__).with_name("review-effort.py")
_spec = importlib.util.spec_from_file_location("review_effort", _p)
re_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(re_)


class Vereinigung(unittest.TestCase):
    def test_live_und_spiegel_werden_vereinigt(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt({"neu-0800"}, {"alt-0432"})
        self.assertEqual(uuids, {"neu-0800", "alt-0432"})
        self.assertEqual(quelle, "pi+spiegel")

    def test_leeres_live_ist_ein_ergebnis(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt(set(), {"alt"})
        self.assertEqual(uuids, {"alt"})
        self.assertEqual(quelle, "pi+spiegel")

    def test_ohne_pi_nur_spiegel_und_gesagt(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt(None, {"alt"})
        self.assertEqual(uuids, {"alt"})
        self.assertEqual(quelle, "spiegel")

    def test_live_liefert_none_wenn_ssh_fehlt(self):
        self.assertIsNone(re_.auto_bestaetigte_live(host="kein.host.invalid", timeout=5))


if __name__ == "__main__":
    unittest.main()
