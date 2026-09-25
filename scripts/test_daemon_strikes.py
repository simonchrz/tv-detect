#!/usr/bin/env python3
"""Daemon: der Strike-Zaehler detect-retries.json zaehlt richtig.

  * ein Watchdog-Kill ist EIN Strike, auch wenn der getoetete Lauf danach
    mit rc=-9 zurueckkehrt und seinerseits einen Fehler meldet;
  * parallele Detect-Threads (DETECT_PARALLEL=3) verlieren keine Strikes
    durch Lesen-Aendern-Schreiben ohne Sperre.
"""
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvthumbs", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class Basis(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = (D.DETECT_RETRY_FILE, D._detect_src_size,
                    D.MAX_DETECT_RETRIES, D._load_detect_retries)
        D.DETECT_RETRY_FILE = self.tmp / "detect-retries.json"
        D._detect_src_size = lambda u: 123
        D.MAX_DETECT_RETRIES = 99  # kein give-up-POST im Test
        D._strike_schon_gezaehlt.clear()

    def tearDown(self):
        (D.DETECT_RETRY_FILE, D._detect_src_size,
         D.MAX_DETECT_RETRIES, D._load_detect_retries) = self.alt
        D._strike_schon_gezaehlt.clear()

    def _n(self, u):
        return json.loads(D.DETECT_RETRY_FILE.read_text())[u]["n"]


class WatchdogEinStrike(Basis):
    def test_kill_plus_rc9_ist_ein_strike(self):
        D._watchdog_strike("u")
        D._record_detect_failure("u")      # rc=-9 des getoeteten Laufs
        self.assertEqual(self._n("u"), 1)

    def test_naechster_echter_fehler_zaehlt_wieder(self):
        D._watchdog_strike("u")
        D._record_detect_failure("u")      # verschluckt
        D._record_detect_failure("u")      # neuer Lauf, neuer Fehler
        self.assertEqual(self._n("u"), 2)

    def test_erfolg_raeumt_markierung(self):
        D._watchdog_strike("u")
        D._record_detect_success("u")
        D._record_detect_failure("u")
        self.assertEqual(self._n("u"), 1)


class Parallel(Basis):
    def test_keine_verlorenen_strikes(self):
        echt = self.alt[3]

        def langsam():
            d = echt()
            time.sleep(0.05)  # Lesen-Aendern-Schreiben-Fenster aufreissen
            return d
        D._load_detect_retries = langsam
        ts = [threading.Thread(target=D._record_detect_failure, args=(f"u{i}",))
              for i in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        d = json.loads(D.DETECT_RETRY_FILE.read_text())
        self.assertEqual(sorted(d), ["u0", "u1", "u2"])


if __name__ == "__main__":
    unittest.main()
