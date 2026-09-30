#!/usr/bin/env python3
"""zeitachsen-check: Merkmals-Zeilen gegen Quell-Dauer (2026-09-30)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("zc", HIER / "zeitachsen-check.py")
zc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zc)


class Tests(unittest.TestCase):
    def test_sprung_wird_erkannt_auch_ohne_menschenlabel(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); arch = d / "archiv"; snap = d / "snap"; arch.mkdir(); snap.mkdir()
            for u, zeilen, sek in (("gut", 1800, 1800.4), ("sprung", 1400, 1000.0), ("kurz", 1790, 1800.0)):
                np.save(d / f"{u}.npy", np.zeros((zeilen, 2), np.float32))
                np.savez(arch / f"{u}.npz", meta=json.dumps({"feature_npy": str(d / f"{u}.npy")}))
                r = snap / f"_rec_{u}"; r.mkdir()
                # KEIN ads_user.json: automatisch gelabelt
                (r / "x.txt").write_text(f"FILE PROCESSING COMPLETE {int(sek * 25)} FRAMES AT 2500\n")
            alt, zc.ARCHIV = zc.ARCHIV, arch
            try:
                aus = zc.zeilen_versatz(sorted(snap.glob("_rec_*")), 15.0)
            finally:
                zc.ARCHIV = alt
            self.assertEqual(set(aus), {"sprung"})
            self.assertAlmostEqual(aus["sprung"], 400.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
