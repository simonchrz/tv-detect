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

    def test_projektion_vorzeichen_stabil(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(2000, 768)).astype(np.float32) * np.linspace(3, 0.1, 768)
        sl = th._siglip_modul()
        _, V1 = sl.projektion_anpassen(Z)
        _, V2 = sl.projektion_anpassen(-Z[::-1] + 0)  # gleiche Kovarianz, andere Daten-Lage
        g = np.abs(V1).argmax(0)
        self.assertTrue((V1[g, np.arange(V1.shape[1])] > 0).all())
        np.testing.assert_allclose(V1, V2, atol=1e-3)

    def test_champion_bekommt_eigene_projektion(self):
        sl = th._siglip_modul()
        E = np.load(TD / "mlp7-siglip-spur.npy")
        n = 5
        rng = np.random.default_rng(1)
        V_eigen = rng.normal(size=(768, 64)).astype(np.float32)
        mu = np.zeros(768, np.float32)
        kopf = th._DeployedMLP(None, None, None, None, 10 + 65, 0, 65, mu, V_eigen)
        X = np.zeros((n, 10 + 65), np.float32)
        X[:, -65:] = sl.spalten(E, n, mu, -V_eigen)  # Kandidat: gespiegelt
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "u1.npy", E)
            (Path(d) / "u1.json").write_text("{}")
            alt, th.SIGLIP_SPUR_DIR = th.SIGLIP_SPUR_DIR, Path(d)
            try:
                (r,) = list(th._MitEigenerSiglip([("u1", "t", [], X, None)], kopf))
            finally:
                th.SIGLIP_SPUR_DIR = alt
        np.testing.assert_allclose(r[3][:, -65:], sl.spalten(E, n, mu, V_eigen), atol=1e-5)
        np.testing.assert_array_equal(r[3][:, :10], X[:, :10])
        self.assertTrue((X[:, -65:-1] == sl.spalten(E, n, mu, -V_eigen)[:, :-1]).all(),
                        "Original darf nicht veraendert werden")

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


    def test_audit_uebergibt_siglip_an_jedem_build_x(self):
        """corpus-label-audit hat ZWEI build_X-Aufrufe; am 2026-09-29 bekam nur
        einer die Projektion, und das Audit uebersprang alle 930 Aufnahmen
        (1285 statt 1350 Spalten)."""
        import ast
        baum = ast.parse((HIER / "corpus-label-audit.py").read_text())
        aufrufe = [n for n in ast.walk(baum) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "build_X"]
        self.assertGreaterEqual(len(aufrufe), 2)
        for c in aufrufe:
            hat = len(c.args) >= 14 or any(k.arg == "siglip" for k in c.keywords)
            self.assertTrue(hat, f"build_X in Zeile {c.lineno} ohne siglip")


if __name__ == "__main__":
    unittest.main()
