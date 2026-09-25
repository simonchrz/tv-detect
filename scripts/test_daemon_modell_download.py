#!/usr/bin/env python3
"""Daemon: Modell-Download im Detect (Sweep 2026-09-25).

  * scheitert der Download von head.bin/backbone, bekommt die uuid einen
    Cooldown (sonst alle 5 s ein neuer Versuch samt neuem Sprecher-Thread);
  * je uuid laeuft hoechstens EINE Sprecher-Extraktion zugleich;
  * http_download ersetzt die Datei atomar (neue Inode): ein tv-detect, der
    head.bin gerade liest, sieht nie einen halb geschriebenen Kopf.
"""
import importlib.util
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvthumbs", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class Antwort:
    def __init__(self, body=b"", headers=None):
        self._b = io.BytesIO(body)
        self.headers = headers or {}
        self.status = 200

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Cooldown(unittest.TestCase):
    def setUp(self):
        self.alt = (D.http_get_json, D.get_source_mit_grund,
                    D._ensure_speaker_artifacts, D.http_download,
                    D._record_detect_failure)
        q = Path(tempfile.mkdtemp()) / "u.ts"
        q.write_bytes(b"x")
        D.http_get_json = lambda url: {"head_url": "/h", "backbone_url": "/b"}
        D.get_source_mit_grund = lambda u: (q, D.QUELLE_OK)
        D._ensure_speaker_artifacts = lambda *a: None
        self.strikes = []
        D._record_detect_failure = lambda u, force=False: self.strikes.append(u)

        def wirft(url, dest):
            raise OSError("pi weg")
        D.http_download = wirft

    def tearDown(self):
        (D.http_get_json, D.get_source_mit_grund,
         D._ensure_speaker_artifacts, D.http_download,
         D._record_detect_failure) = self.alt
        D._failed_until.clear()

    def test_modellfehler_setzt_cooldown(self):
        self.assertFalse(D.process_detect("dvr-x-1"))
        self.assertIn("dvr-x-1", D._failed_until)
        self.assertEqual(self.strikes, [], "vorlaeufig, kein Strike")


class SprecherSperre(unittest.TestCase):
    def setUp(self):
        self.alt = D._ensure_speaker_artifacts_ungesperrt
        self.gleichzeitig = self.max = 0
        self.lock = threading.Lock()

        def langsam(uuid, src, show):
            with self.lock:
                self.gleichzeitig += 1
                self.max = max(self.max, self.gleichzeitig)
            time.sleep(0.1)
            with self.lock:
                self.gleichzeitig -= 1
            return None
        D._ensure_speaker_artifacts_ungesperrt = langsam

    def tearDown(self):
        D._ensure_speaker_artifacts_ungesperrt = self.alt

    def test_eine_extraktion_je_uuid(self):
        ts = [threading.Thread(target=D._ensure_speaker_artifacts,
                               args=("u", "/x.ts", "Show")) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(self.max, 1)

    def test_verschiedene_uuids_laufen_parallel(self):
        ts = [threading.Thread(target=D._ensure_speaker_artifacts,
                               args=(f"u{i}", "/x.ts", "Show")) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(self.max, 2)


class AtomarerDownload(unittest.TestCase):
    def setUp(self):
        self.alt = D.urllib.request.urlopen
        self.dest = Path(tempfile.mkdtemp()) / "head.bin"
        self.dest.write_bytes(b"alt" * 100)

        def urlopen(req, *a, **kw):
            head = getattr(req, "method", None) == "HEAD"
            return Antwort(b"" if head else b"neu" * 100,
                           {"Last-Modified": "Fri, 25 Sep 2026 00:00:00 GMT",
                            "Content-Length": "300"})
        D.urllib.request.urlopen = urlopen

    def tearDown(self):
        D.urllib.request.urlopen = self.alt

    def test_neue_inode_und_inhalt(self):
        ino = os.stat(self.dest).st_ino
        # ein Leser, der die alte Datei offen haelt, muss sie unversehrt sehen
        with open(self.dest, "rb") as leser:
            D.http_download("https://pi/h", self.dest)
            self.assertEqual(leser.read(), b"alt" * 100)
        self.assertNotEqual(os.stat(self.dest).st_ino, ino)
        self.assertEqual(self.dest.read_bytes(), b"neu" * 100)
        self.assertEqual(list(self.dest.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
