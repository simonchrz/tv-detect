#!/usr/bin/env python3
"""train-head: Wisch-Punkte aus wisch.json (nur Aufnahmen ohne ads_user.json)."""
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("th_wisch", HIER / "train-head.py")
th = importlib.util.module_from_spec(spec)
sys.modules["th_wisch"] = th
spec.loader.exec_module(th)


class Tests(unittest.TestCase):
    def schreibe(self, marken):
        d = Path(tempfile.mkdtemp())
        (d / "wisch.json").write_text(json.dumps({"marken": marken}))
        return d

    def test_urteile(self):
        d = self.schreibe([
            {"t": 798, "urteil": "trailer", "ts": 1},
            {"t": 100, "urteil": "sendung", "ts": 1},
            {"t": 200, "urteil": "werbung", "ts": 1},
            {"t": 300, "urteil": "unklar", "ts": 1},
        ])
        self.assertEqual(th.wisch_punkte(d), [(100.0, 0), (200.0, 1), (798.0, 1)])

    def test_letzte_marke_gilt(self):
        d = self.schreibe([
            {"t": 50, "urteil": "sendung", "ts": 2},
            {"t": 50, "urteil": "werbung", "ts": 1},
            {"t": 60, "urteil": "werbung", "ts": 1},
            {"t": 60, "urteil": "unklar", "ts": 2},
        ])
        self.assertEqual(th.wisch_punkte(d), [(50.0, 0)])

    def test_ohne_datei(self):
        self.assertEqual(th.wisch_punkte(Path(tempfile.mkdtemp())), [])
        d = Path(tempfile.mkdtemp())
        (d / "wisch.json").write_text("kaputt")
        self.assertEqual(th.wisch_punkte(d), [])

    def test_nur_ohne_menschen_label(self):
        # Die Quelle darf nur gelesen werden, wenn ads_user.json fehlt —
        # sonst spiegelt der tv-recorder schon, und ein Menschen-Block
        # schlaegt einen Einzelpunkt.
        src = (HIER / "train-head.py").read_text()
        self.assertIn("wisch = wisch_punkte(rec_dir) if user_raw is None else []", src)

    def test_per_rec_umbau_behaelt_feld(self):
        # Der Breiten-Umbau baut per_rec-Tupel neu; Index 14 darf nicht fallen.
        src = (HIER / "train-head.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
                    and ast.unparse(node.targets[0]) == "per_rec[i]"):
                self.assertIn("r[14]", ast.unparse(node.value))
                break
        else:
            self.fail("per_rec[i]-Umbau nicht gefunden")


if __name__ == "__main__":
    unittest.main()
