#!/usr/bin/env python3
"""Daemon: nur eine WIRKLICH fehlende Quelle gibt einen Detect auf.

get_source() liefert None aus fuenf Gruenden; process_detect las jeden als
"HLS-VOD-only orphan" und gab sofort auf (force-Strike, detect-give-up,
Marker weg). Hier festgenagelt:
  * get_source_mit_grund unterscheidet 404 / 425 / 5xx / Netz;
  * nur QUELLE_FEHLT zaehlt als (erzwungener) Strike;
  * ein Fehler beim Holen der detect-config ist vorlaeufig, kein Strike.
"""
import importlib.util
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvthumbs", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _http_fehler(code):
    def urlopen(req, *a, **kw):
        raise urllib.error.HTTPError(str(req), code, "x", {}, io.BytesIO(b""))
    return urlopen


class Grund(unittest.TestCase):
    def setUp(self):
        self.alt = (D.SOURCE_CACHE, D.urllib.request.urlopen)
        D.SOURCE_CACHE = Path(tempfile.mkdtemp())

    def tearDown(self):
        D.SOURCE_CACHE, D.urllib.request.urlopen = self.alt
        D._failed_until.clear()

    def _grund(self, urlopen):
        D.urllib.request.urlopen = urlopen
        p, g = D.get_source_mit_grund("dvr-x-1")
        self.assertIsNone(p)
        self.assertIsNone(D.get_source("dvr-x-1"))
        return g

    def test_404_fehlt(self):
        self.assertEqual(self._grund(_http_fehler(404)), D.QUELLE_FEHLT)

    def test_425_laeuft(self):
        self.assertEqual(self._grund(_http_fehler(425)), D.QUELLE_LAEUFT)

    def test_5xx_fehler(self):
        self.assertEqual(self._grund(_http_fehler(503)), D.QUELLE_FEHLER)

    def test_netz_fehler(self):
        def urlopen(*a, **kw):
            raise urllib.error.URLError("weg")
        self.assertEqual(self._grund(urlopen), D.QUELLE_FEHLER)


class ProcessDetect(unittest.TestCase):
    def setUp(self):
        self.alt = (D.http_get_json, D.get_source_mit_grund,
                    D._record_detect_failure)
        self.strikes = []
        D.http_get_json = lambda url: {}
        D._record_detect_failure = (
            lambda uuid, force=False: self.strikes.append(force) or 1)

    def tearDown(self):
        (D.http_get_json, D.get_source_mit_grund,
         D._record_detect_failure) = self.alt
        D._failed_until.clear()

    def _lauf(self, grund):
        D.get_source_mit_grund = lambda u: (None, grund)
        self.assertFalse(D.process_detect("dvr-x-1"))

    def test_vorlaeufig_kein_strike(self):
        for g in (D.QUELLE_LAEUFT, D.QUELLE_FEHLER):
            self.strikes.clear()
            D._failed_until.clear()
            self._lauf(g)
            self.assertEqual(self.strikes, [], f"{g} darf nicht aufgeben")
            self.assertIn("dvr-x-1", D._failed_until, "Cooldown fehlt")

    def test_404_gibt_auf(self):
        self._lauf(D.QUELLE_FEHLT)
        self.assertEqual(self.strikes, [True])

    def test_nur_dump_gibt_nie_auf(self):
        D.get_source_mit_grund = lambda u: (None, D.QUELLE_FEHLT)
        D.process_detect("dvr-x-1", nur_dump=True)
        self.assertEqual(self.strikes, [])

    def test_detect_config_fehler_kein_strike(self):
        def wirft(url):
            raise urllib.error.URLError("pi neu gestartet")
        D.http_get_json = wirft
        D.get_source_mit_grund = lambda u: self.fail("darf nicht laufen")
        self.assertFalse(D.process_detect("dvr-x-1"))
        self.assertEqual(self.strikes, [])
        self.assertIn("dvr-x-1", D._failed_until)


if __name__ == "__main__":
    unittest.main()
