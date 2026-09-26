#!/usr/bin/env python3
"""Daemon: wann die SigLIP-Spur vor dem Detect erzeugt wird (siglip-spur-design.md).

Dieselben Zusagen wie test_daemon_ocr_spur.py:
  * ein Kopf ohne SigLIP (MLP1, MLP6) loest KEINE Erzeugung aus;
  * eine veraltete Spur (oder eine npy ohne Beilage) wird NIE mitgegeben;
  * scheitert die Erzeugung, laeuft der Detect ohne Spur weiter (None).
Dazu: ein MLP7-Kopf bekommt auch seine OCR-Spur (bis 2026-09-26 pruefte
_kopf_braucht_ocr nur b"MLP6"), und der Re-Filter raeumt die Spur.
Kopf-Erkennung gegen die echten Fixtures aus internal/signals/testdata.
"""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvthumbs_sig", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
TD = REPO / "internal/signals/testdata"


class KopfErkennung(unittest.TestCase):
    def test_mlp7(self):
        self.assertTrue(D._kopf_braucht_siglip(TD / "mlp7-siglip.bin"))
        self.assertTrue(D._kopf_braucht_ocr(TD / "mlp7-siglip.bin"),
                        "v7-Kopf muss seine OCR-Spur weiter bekommen")

    def test_aeltere_brauchen_kein_siglip(self):
        self.assertFalse(D._kopf_braucht_siglip(TD / "mlp6-ocr.bin"))
        self.assertFalse(D._kopf_braucht_siglip(TD / "mlp1-bare.bin"))
        self.assertFalse(D._kopf_braucht_siglip("/gibt/es/nicht.bin"))
        self.assertTrue(D._kopf_braucht_ocr(TD / "mlp6-ocr.bin"))


class SpurFuerDetect(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.q = self.tmp / "u.ts"
        self.q.write_bytes(b"x" * 100)
        os.utime(self.q, (1_700_000_000, 1_700_000_000))
        self.alt = (D.SIGLIP_SPUR_DIR, D.SIGLIP_PY, D.SIGLIP_SKRIPT, D.subprocess.run)
        D.SIGLIP_SPUR_DIR = self.tmp / "spur"
        D.SIGLIP_SPUR_DIR.mkdir()
        D.SIGLIP_PY = "/bin/echo"
        D.SIGLIP_SKRIPT = self.q  # existiert; Aufruf wird ersetzt
        self.aufrufe = 0

    def tearDown(self):
        D.SIGLIP_SPUR_DIR, D.SIGLIP_PY, D.SIGLIP_SKRIPT, D.subprocess.run = self.alt

    def _spur(self, mit_json=True, **f):
        (D.SIGLIP_SPUR_DIR / "u.npy").write_bytes(b"npy")
        if mit_json:
            s = {"quelle_bytes": 100, "quelle_mtime": 1_700_000_000, "leer": []}
            s.update(f)
            (D.SIGLIP_SPUR_DIR / "u.json").write_text(json.dumps(s))

    def _run(self, rc=0):
        def run(cmd, **kw):
            self.aufrufe += 1
            if rc == 0:
                self._spur()
            return subprocess.CompletedProcess(cmd, rc, "", "kaputt" if rc else "")
        D.subprocess.run = run

    def test_frische_spur_ohne_erzeugung(self):
        self._spur(); self._run()
        self.assertEqual(D._siglip_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"),
                         D.SIGLIP_SPUR_DIR / "u.npy")
        self.assertEqual(self.aufrufe, 0)

    def test_v6_erzeugt_nicht(self):
        self._run()
        self.assertIsNone(D._siglip_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"))
        self.assertEqual(self.aufrufe, 0, "v6-Kopf darf den Detect nicht verlangsamen")

    def test_veraltet_oder_ohne_beilage_nie(self):
        self._spur(quelle_bytes=99); self._run()
        self.assertIsNone(D._siglip_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"))
        (D.SIGLIP_SPUR_DIR / "u.json").unlink()
        self.assertIsNone(D._siglip_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"),
                          "npy ohne Beilage (O28-Altlast) gilt als fehlend")

    def test_leere_kacheln_nie(self):
        self._spur(leer=[[0, 180]]); self._run()
        self.assertIsNone(D._siglip_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"))

    def test_v7_erzeugt_bei_fehlender_spur(self):
        self._run()
        self.assertEqual(D._siglip_spur_fuer("u", self.q, TD / "mlp7-siglip.bin"),
                         D.SIGLIP_SPUR_DIR / "u.npy")
        self.assertEqual(self.aufrufe, 1)

    def test_gescheiterte_erzeugung_detect_laeuft_weiter(self):
        self._run(rc=1)
        self.assertIsNone(D._siglip_spur_fuer("u", self.q, TD / "mlp7-siglip.bin"))


class Invalidierung(unittest.TestCase):
    def test_refilter_raeumt_siglip(self):
        tmp = Path(tempfile.mkdtemp())
        namen = ("OCR_SPUR_DIR", "SIGLIP_SPUR_DIR", "EMB_CACHE", "SPK_CSV_CACHE",
                 "TVD_FEATURES", "TVD_ARCHIVE", "WHISPER_CACHE")
        alt = {n: getattr(D, n) for n in namen}
        try:
            for n in namen:
                setattr(D, n, tmp / n)
                (tmp / n).mkdir()
            dateien = [tmp / "SIGLIP_SPUR_DIR" / "u.npy", tmp / "SIGLIP_SPUR_DIR" / "u.json"]
            for f in dateien:
                f.write_text("x")
            D._invalidate_derived("u")
            for f in dateien:
                self.assertFalse(f.exists(), f"{f.name} ueberlebt den Re-Filter")
        finally:
            for n, v in alt.items():
                setattr(D, n, v)


if __name__ == "__main__":
    unittest.main()
