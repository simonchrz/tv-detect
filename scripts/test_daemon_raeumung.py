#!/usr/bin/env python3
"""Daemon: Quellen-Cache-Raeumung und get_source (Sweep 2026-09-25).

Zwei Zusagen, die beide STILL brechen:
  * nur ein 404 des Pi macht eine Kopie zur "einzigen" — ein 5xx/400/425
    darf NIE als "Pi hat sie" gelesen werden (sonst raeumt Stufe 1 die
    einzige Kopie in dem Glauben, es sei ein Duplikat);
  * ein erfolgreicher Download bleibt erfolgreich, auch wenn die Raeumung
    danach stolpert (parallel geloeschte Datei → stat() wirft).
"""
import importlib.util
import io
import os
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
        raise urllib.error.HTTPError(getattr(req, "full_url", str(req)),
                                     code, "x", {}, io.BytesIO(b""))
    return urlopen


class Antwort:
    def __init__(self, status=200, body=b"", headers=None):
        self.status = status
        self._b = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Raeumung(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = (D.SOURCE_CACHE, D._freier_platz_gb, D.geschuetzte_uuids,
                    D.urllib.request.urlopen)
        D.SOURCE_CACHE = self.tmp
        # zwischen HARD und MIN: Stufe 1, nur Duplikate duerfen weg
        frei = (D.SOURCE_CACHE_MIN_FREE_GB + D.SOURCE_CACHE_HARD_FREE_GB) / 2
        D._freier_platz_gb = lambda p: frei
        D.geschuetzte_uuids = lambda: set()
        self.f = self.tmp / "dvr-x-1.ts"
        self.f.write_bytes(b"x" * 10)

    def tearDown(self):
        (D.SOURCE_CACHE, D._freier_platz_gb, D.geschuetzte_uuids,
         D.urllib.request.urlopen) = self.alt

    def test_5xx_raeumt_nicht(self):
        for code in (502, 503, 500, 400, 425):
            D.urllib.request.urlopen = _http_fehler(code)
            D._maybe_evict_source_cache()
            self.assertTrue(self.f.exists(),
                            f"HTTP {code} darf nicht als 'Pi hat sie' gelten")

    def test_404_schuetzt_in_stufe_1(self):
        D.urllib.request.urlopen = _http_fehler(404)
        D._maybe_evict_source_cache()
        self.assertTrue(self.f.exists())

    def test_200_raeumt_duplikat(self):
        D.urllib.request.urlopen = lambda *a, **kw: Antwort(200)
        D._maybe_evict_source_cache()
        self.assertFalse(self.f.exists())

    def test_204_raeumt_nicht(self):
        D.urllib.request.urlopen = lambda *a, **kw: Antwort(204)
        D._maybe_evict_source_cache()
        self.assertTrue(self.f.exists())


class DownloadBleibtErfolg(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = (D.SOURCE_CACHE, D._maybe_evict_source_cache,
                    D._drop_pi_source, D.urllib.request.urlopen)
        D.SOURCE_CACHE = self.tmp
        D._drop_pi_source = lambda u: "done"
        body = b"y" * 5000
        D.urllib.request.urlopen = lambda *a, **kw: Antwort(
            200, body, {"Content-Length": str(len(body))})

    def tearDown(self):
        (D.SOURCE_CACHE, D._maybe_evict_source_cache,
         D._drop_pi_source, D.urllib.request.urlopen) = self.alt

    def test_raeumungsfehler_kippt_download_nicht(self):
        def wirft():
            raise FileNotFoundError("parallel geloescht")
        D._maybe_evict_source_cache = wirft
        p = D.get_source("dvr-x-2")
        self.assertEqual(p, self.tmp / "dvr-x-2.ts")
        self.assertTrue(p.exists())

    def test_raeumung_uebersteht_verschwundene_datei(self):
        D._maybe_evict_source_cache = self.alt[1]
        # kaputter Symlink: glob() listet ihn, stat() wirft
        os.symlink(self.tmp / "gibt-es-nicht", self.tmp / "dvr-weg-1.ts")
        alt_frei = D._freier_platz_gb
        D._freier_platz_gb = lambda p: 0.0
        try:
            D._maybe_evict_source_cache()   # darf nicht werfen
        finally:
            D._freier_platz_gb = alt_frei


if __name__ == "__main__":
    unittest.main()
