#!/usr/bin/env python3
"""ocr-spur-nachrechnen.py: wann eine Spur neu gerechnet wird, und in welcher
Reihenfolge.

Die Staleness-Pruefung ist der Kern: eine neu geholte oder getrimmte Quelle
hat eine andere Zeitachse, und eine Spur darauf waere still falsch —
dieselbe Klasse wie die veralteten Fingerprints (CSI, South Park, 22.09.).
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ocr_spur", HIER / "ocr-spur-nachrechnen.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Staleness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.q = self.tmp / "a.ts"
        self.q.write_bytes(b"x" * 100)
        os.utime(self.q, (1_700_000_000, 1_700_000_000))

    def _spur(self, **felder):
        s = {"quelle_bytes": 100, "quelle_mtime": 1_700_000_000, "fehlgeschlagen": []}
        s.update(felder)
        (self.tmp / "a.json").write_text(json.dumps(s))

    def test_fehlt(self):
        self.assertEqual(M.veraltet_oder_fehlt(self.tmp, "a", self.q), (True, "fehlt"))

    def test_passt(self):
        self._spur()
        self.assertEqual(M.veraltet_oder_fehlt(self.tmp, "a", self.q), (False, ""))

    def test_quelle_groesse_gewechselt(self):
        self._spur(quelle_bytes=99)
        self.assertEqual(M.veraltet_oder_fehlt(self.tmp, "a", self.q)[1], "Quelle gewechselt")

    def test_quelle_mtime_gewechselt(self):
        self._spur()
        os.utime(self.q, (1_700_000_100, 1_700_000_100))
        self.assertEqual(M.veraltet_oder_fehlt(self.tmp, "a", self.q)[1], "Quelle gewechselt")

    def test_luecken_werden_nachgeholt(self):
        self._spur(fehlgeschlagen=[{"von": 0, "bis": 180}])
        self.assertEqual(M.veraltet_oder_fehlt(self.tmp, "a", self.q)[1], "Luecken")


class Reihenfolge(unittest.TestCase):
    def test_messsatz_und_golden_zuerst(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "m.json").write_text(json.dumps({"uuids": ["z-mess", "fehlt-hier"]}))
        (tmp / "g.json").write_text(json.dumps({"uuids": ["y-gold", "z-mess"]}))
        alt_m, alt_g = M.MESSSATZ, M.GOLDEN
        M.MESSSATZ, M.GOLDEN = tmp / "m.json", tmp / "g.json"
        try:
            r = M.reihenfolge({"a-rest", "b-rest", "y-gold", "z-mess"})
        finally:
            M.MESSSATZ, M.GOLDEN = alt_m, alt_g
        self.assertEqual(r, ["z-mess", "y-gold", "a-rest", "b-rest"])

    def test_golden_als_liste_von_objekten(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "g.json").write_text(json.dumps({"recs": [{"uuid": "u1"}]}))
        self.assertEqual(M._uuids(tmp / "g.json"), ["u1"])
        (tmp / "l.json").write_text(json.dumps(["u2"]))
        self.assertEqual(M._uuids(tmp / "l.json"), ["u2"])


if __name__ == "__main__":
    unittest.main()
