#!/usr/bin/env python3
"""train-head --siglip-spalten: MLP7 lesen/schreiben, Spalten, Projektion nur aus train."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("th_sig", HIER / "train-head.py")
th = importlib.util.module_from_spec(spec)
sys.modules["th_sig"] = th
spec.loader.exec_module(th)
TD = HIER.parent / "internal/signals/testdata"


class Tests(unittest.TestCase):
    def test_mlp7_fixture_lesbar(self):
        m = th.load_deployed_mlp(TD / "mlp7-siglip.bin")
        self.assertIsNotNone(m)
        self.assertEqual((m.n_ocr, m.n_siglip), (3, 65))
        self.assertEqual(m.siglip_V.shape, (768, 64))
        fx = json.loads((TD / "mlp7-siglip-parity.json").read_text())
        sl = th._siglip_modul()
        E = np.load(TD / "mlp7-siglip-spur.npy")
        n, bb = fx["n"], fx["backbone"]
        X = np.concatenate([np.asarray(fx["embeds"], np.float32).reshape(n, bb),
                            np.asarray(fx["logo"], np.float32)[:, None],
                            np.asarray(fx["rms"], np.float32)[:, None],
                            np.zeros((n, 3), np.float32),
                            sl.spalten(E, n, m.siglip_mu, m.siglip_V)], 1)
        p = m.predict_proba(X)[:, 1]
        np.testing.assert_allclose(p, fx["erwartet"], atol=1e-5)

    def test_mlp6_hat_kein_siglip(self):
        m = th.load_deployed_mlp(TD / "mlp6-ocr.bin")
        self.assertEqual(m.n_siglip, 0)

    def test_zusatzspalten_haengt_siglip_hinten_an(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "u.npy", np.ones((5, 768), np.float16))
            (d / "u.json").write_text("{}")
            alt = th.SIGLIP_SPUR_DIR
            th.SIGLIP_SPUR_DIR = d
            try:
                mu, V = np.zeros(768, np.float32), np.eye(768, 64, dtype=np.float32)
                z = th.zusatzspalten(np.zeros((8, 1282), np.float32), "u", "", {}, 0,
                                     kanal=False, ocr=False, siglip=(mu, V))
            finally:
                th.SIGLIP_SPUR_DIR = alt
            self.assertEqual(z.shape, (8, 65))
            np.testing.assert_array_equal(z[:5, 64], 1)
            self.assertFalse(z[5:].any(), "hinter der Spur alles 0")

    def test_projektion_nur_aus_train(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(0)
            led = {}
            for i in range(25):
                np.save(d / f"t{i}.npy", rng.normal(size=(40, 768)).astype(np.float16))
                (d / f"t{i}.json").write_text("{}"); led[f"t{i}"] = "train"
            np.save(d / "x.npy", np.full((40, 768), 1e3, np.float16))
            (d / "x.json").write_text("{}"); led["x"] = "test"
            mu, V, n = th.siglip_projektion({"eimer": led}, schritt=1, spur_dir=d)
            self.assertEqual(n, 25)
            self.assertLess(np.abs(mu).max(), 1.0, "test-Aufnahme floss in die Projektion")


if __name__ == "__main__":
    unittest.main()
