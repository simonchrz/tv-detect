#!/usr/bin/env python3
"""Spot-Fingerprints (tv-spot-extract.py + Daemon-Start), Sweep 2026-09-25.

  * haengt fpcalc oder ffmpeg, wird ffmpeg getoetet und abgeraeumt;
  * Aufnahmen ohne Spots werden lokal gemerkt und verbrauchen kein --limit
    mehr (der Pi kann sie nicht als erledigt fuehren, s. SPOT_LEER_FILE);
    ein Extraktionsfehler ist dagegen kein Befund und wird nicht gemerkt;
  * der Daemon startet keinen zweiten Lauf, solange der erste laeuft.
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _laden(name, pfad):
    spec = importlib.util.spec_from_file_location(name, REPO / pfad)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _laden("tvspot", "daemon/tv-spot-extract.py")
D = _laden("tvthumbs", "daemon/tv-thumbs-daemon.py")


class FakeFF:
    def __init__(self, wait_haengt):
        self.wait_haengt = wait_haengt
        self.killed = False
        self.stdout = self

    def close(self):
        pass

    def poll(self):
        return None

    def wait(self, timeout=None):
        if self.wait_haengt and not self.killed and timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return -9 if self.killed else 0

    def kill(self):
        self.killed = True


class FfmpegWirdAbgeraeumt(unittest.TestCase):
    def setUp(self):
        self.alt = (S.subprocess.Popen, S.subprocess.run)
        self.ff = FakeFF(wait_haengt=True)
        S.subprocess.Popen = lambda *a, **kw: self.ff

    def tearDown(self):
        S.subprocess.Popen, S.subprocess.run = self.alt

    def test_fpcalc_timeout(self):
        def run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 30)
        S.subprocess.run = run
        self.assertIsNone(S.extract_chromaprint("/x.ts", 10.0, 30.0))
        self.assertTrue(self.ff.killed, "ffmpeg bleibt nach fpcalc-Timeout stehen")

    def test_wait_timeout(self):
        S.subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, b"FINGERPRINT=" + b",".join([b"1"] * 60), b"")
        S.extract_chromaprint("/x.ts", 10.0, 30.0)
        self.assertTrue(self.ff.killed, "ffmpeg bleibt nach wait-Timeout stehen")


class OhneSpots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = {n: getattr(S, n) for n in (
            "SPOT_LEER_FILE", "get_local_ts", "fetch_ads_user",
            "silence_intervals", "extract_chromaprint", "http_json",
            "process_uuid")}
        S.SPOT_LEER_FILE = self.tmp / "leer.json"
        S.get_local_ts = lambda u: self.tmp / f"{u}.ts"
        S.fetch_ads_user = lambda u: [[0.0, 30.0]]
        S.silence_intervals = lambda *a: []

    def tearDown(self):
        for n, v in self.alt.items():
            setattr(S, n, v)

    def test_null_spots_werden_gemerkt(self):
        S.extract_chromaprint = lambda *a: None   # zu wenig Fingerprint
        self.assertEqual(S.process_uuid("a"), (0, True))
        self.assertTrue(S.leer_bekannt("a"))

    def test_fehler_ist_kein_befund(self):
        def kaputt(*a):
            S._chromaprint_fehler += 1
            return None
        S.extract_chromaprint = kaputt
        S.process_uuid("a")
        self.assertFalse(S.leer_bekannt("a"))

    def test_gemerkte_kosten_kein_budget(self):
        S.leer_merken("a")
        S.http_json = lambda url: {"uuids": ["a", "b"]}
        gesehen = []
        S.process_uuid = lambda u: gesehen.append(u) or (1, True)
        alt_argv = sys.argv
        sys.argv = ["tv-spot-extract.py", "--queue", "--limit", "1"]
        try:
            S.main()
        finally:
            sys.argv = alt_argv
        self.assertEqual(gesehen, ["b"])

    def test_ttl_laeuft_ab(self):
        S.leer_merken("a")
        alt = S.SPOT_LEER_TTL_S
        S.SPOT_LEER_TTL_S = -1
        try:
            self.assertFalse(S.leer_bekannt("a"))
        finally:
            S.SPOT_LEER_TTL_S = alt


class DaemonStartetNurEinen(unittest.TestCase):
    def setUp(self):
        self.alt = (D.subprocess.Popen, D._spot_proc)
        self.starts = []

        class Laeuft:
            rc = None

            def poll(self):
                return self.rc
        self.Laeuft = Laeuft

        def popen(*a, **kw):
            p = Laeuft()
            self.starts.append(p)
            return p
        D.subprocess.Popen = popen
        D._spot_proc = None

    def tearDown(self):
        D.subprocess.Popen, D._spot_proc = self.alt

    def test_kein_zweiter_lauf(self):
        self.assertTrue(D._spot_extract_starten(10))
        self.assertFalse(D._spot_extract_starten(10))
        self.assertEqual(len(self.starts), 1)

    def test_nach_ende_wieder(self):
        D._spot_extract_starten(10)
        self.starts[0].rc = 0
        self.assertTrue(D._spot_extract_starten(10))
        self.assertEqual(len(self.starts), 2)


if __name__ == "__main__":
    unittest.main()
