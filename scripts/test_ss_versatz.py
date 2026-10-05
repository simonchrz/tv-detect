#!/usr/bin/env python3
"""daemon/ss_versatz.py: gleiche Regel wie tv-detect decode.Info.SeekOffsetS."""
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ss_versatz", REPO / "daemon" / "ss_versatz.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "kein ffmpeg")
class Tests(unittest.TestCase):
    def _ff(self, *a):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)

    def test_zwei_programme_versatz_10s(self):
        # Wie internal/decode/versatz_test.go: Ton-Programm ab ~1.4 s,
        # Video-Programm (eigene PIDs) ab ~11.4 s, hintereinandergehaengt.
        with tempfile.TemporaryDirectory() as d:
            v, a, f = Path(d, "v.ts"), Path(d, "a.ts"), Path(d, "versatz.ts")
            self._ff("-f", "lavfi", "-i", "nullsrc=s=32x18:r=25:d=5", "-c:v", "mpeg2video",
                     "-output_ts_offset", "10", "-mpegts_start_pid", "0x200",
                     "-mpegts_service_id", "2", str(v))
            self._ff("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=1", "-c:a", "mp2", str(a))
            f.write_bytes(a.read_bytes() + v.read_bytes())
            self.assertAlmostEqual(V.versatz(f), 10, delta=1)
            self.assertEqual(V.versatz(v), 0.0)  # normale Datei: unangetastet

    def test_unlesbar_ist_null(self):
        self.assertEqual(V.versatz("/gibt/es/nicht.ts"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
