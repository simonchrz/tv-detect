#!/usr/bin/env python3
"""Daemon (O26): wann die OCR-Spur vor dem Detect erzeugt wird.

Drei Zusagen, die still brechen koennten:
  * ein MLP1-Kopf loest KEINE Erzeugung aus (der Detect wird nicht langsamer);
  * eine veraltete Spur wird NIE mitgegeben (andere Zeitachse);
  * scheitert die Erzeugung, laeuft der Detect ohne Spur weiter (None).
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
spec = importlib.util.spec_from_file_location("tvthumbs", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
TD = REPO / "internal/signals/testdata"


class KopfErkennung(unittest.TestCase):
    def test_mlp6_braucht_ocr(self):
        self.assertTrue(D._kopf_braucht_ocr(TD / "mlp6-ocr.bin"))

    def test_mlp1_braucht_keine(self):
        self.assertFalse(D._kopf_braucht_ocr(TD / "mlp1-bare.bin"))

    def test_fehlender_kopf(self):
        self.assertFalse(D._kopf_braucht_ocr("/gibt/es/nicht.bin"))


class SpurFuerDetect(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.q = self.tmp / "u.ts"
        self.q.write_bytes(b"x" * 100)
        os.utime(self.q, (1_700_000_000, 1_700_000_000))
        self.alt = (D.OCR_SPUR_DIR, D.OCR_SPUR_BIN, D.subprocess.run)
        D.OCR_SPUR_DIR = self.tmp / "spur"
        D.OCR_SPUR_DIR.mkdir()
        D.OCR_SPUR_BIN = "/bin/echo"  # existiert; Aufruf wird ersetzt
        self.aufrufe = 0

    def tearDown(self):
        D.OCR_SPUR_DIR, D.OCR_SPUR_BIN, D.subprocess.run = self.alt

    def _spur(self, **f):
        s = {"quelle_bytes": 100, "quelle_mtime": 1_700_000_000, "fehlgeschlagen": []}
        s.update(f)
        (D.OCR_SPUR_DIR / "u.json").write_text(json.dumps(s))

    def _run_erzeugt(self, rc=0):
        def run(cmd, **kw):
            self.aufrufe += 1
            if rc == 0:
                self._spur()
            return subprocess.CompletedProcess(cmd, rc, "", "kaputt" if rc else "")
        D.subprocess.run = run

    def test_frische_spur_ohne_erzeugung(self):
        self._spur()
        self._run_erzeugt()
        p = D._ocr_spur_fuer("u", self.q, TD / "mlp6-ocr.bin")
        self.assertEqual(p, D.OCR_SPUR_DIR / "u.json")
        self.assertEqual(self.aufrufe, 0)

    def test_mlp1_erzeugt_nicht(self):
        self._run_erzeugt()
        self.assertIsNone(D._ocr_spur_fuer("u", self.q, TD / "mlp1-bare.bin"))
        self.assertEqual(self.aufrufe, 0, "MLP1-Kopf darf den Detect nicht verlangsamen")

    def test_veraltete_spur_nie_mitgeben(self):
        self._spur(quelle_bytes=99)
        self._run_erzeugt()
        self.assertIsNone(D._ocr_spur_fuer("u", self.q, TD / "mlp1-bare.bin"))

    def test_mlp6_erzeugt_bei_fehlender_spur(self):
        self._run_erzeugt()
        p = D._ocr_spur_fuer("u", self.q, TD / "mlp6-ocr.bin")
        self.assertEqual(p, D.OCR_SPUR_DIR / "u.json")
        self.assertEqual(self.aufrufe, 1)

    def test_gescheiterte_erzeugung_detect_laeuft_weiter(self):
        self._run_erzeugt(rc=1)
        self.assertIsNone(D._ocr_spur_fuer("u", self.q, TD / "mlp6-ocr.bin"))



class Invalidierung(unittest.TestCase):
    """Re-Filter: alle abgeleiteten Artefakte der ALTEN Quelle muessen weg."""

    def test_raeumt_ocr_und_sprecher(self):
        tmp = Path(tempfile.mkdtemp())
        namen = ("OCR_SPUR_DIR", "EMB_CACHE", "SPK_CSV_CACHE",
                 "TVD_FEATURES", "TVD_ARCHIVE", "WHISPER_CACHE")
        alt = {n: getattr(D, n) for n in namen}
        try:
            for n in namen:
                setattr(D, n, tmp / n)
                (tmp / n).mkdir()
            dateien = [tmp / "OCR_SPUR_DIR" / "u.json",
                       tmp / "EMB_CACHE" / "u.npz",
                       tmp / "SPK_CSV_CACHE" / "u.speaker.csv"]
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
