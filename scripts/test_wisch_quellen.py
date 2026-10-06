"""wisch-quellen.py: Kartenpositionen fuer Anker-Widersprueche und Massstab."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location("wq", Path(__file__).resolve().parent / "wisch-quellen.py")
wq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wq)


class WischQuellen(unittest.TestCase):
    def test_massstab_mitte_und_kanten(self):
        k = [t for _, t in wq.massstab_karten("u", [(600.0, 900.0)])]
        self.assertEqual(sorted(k), [580, 620, 750, 880, 920])

    def test_massstab_kurzer_block_und_nachbar(self):
        # kurzer Block: nur Mitte + aussen; aussen-Punkt im Nachbarblock entfaellt
        k = sorted(t for _, t in wq.massstab_karten("u", [(100.0, 140.0), (150.0, 400.0)]))
        self.assertIn(120, k)
        self.assertNotIn(160, k)

    def test_anker_widerspruch(self):
        with tempfile.TemporaryDirectory() as d:
            alt, wq.BILD = wq.BILD, Path(d)
            try:
                (Path(d) / "u.json").write_text(json.dumps({"anchored": [
                    {"window_start_s": 300, "end_s": 330}, {"window_start_s": 600, "end_s": 900},
                    {"window_start_s": 1000, "end_s": 1005}]}))
                k = wq.anker_karten("u", [(600.0, 900.0)], min_s=10)
                self.assertEqual([(z[0], z[2], z[3]) for z in k], [(30, 315, "anker")])
                lang = wq.anker_karten("u", [], min_s=10)
                self.assertEqual(len([z for z in lang if z[0] == 300]), 3)
            finally:
                wq.BILD = alt


if __name__ == "__main__":
    unittest.main()
