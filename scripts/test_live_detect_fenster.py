#!/usr/bin/env python3
"""tv-live-detect: ein Fenster wird nur ausgewertet, wenn die Aussage stimmt.

  * scheitert tv-detect (rc!=0 oder Timeout), bleibt der alte Stand stehen
    und nichts wird gePOSTet — vorher wurden alle Bloecke des Fensters
    verworfen und "keine Werbung" hochgeladen;
  * fehlt ein Segment, wird nur der lueckenlose Schwanz nach der Luecke
    ausgewertet (exakte Zeitachse); ist der zu kurz, gar nichts. Vorher
    kamen alle Bloecke nach der Luecke um deren Laenge zu frueh heraus.
"""
import importlib.util
import io
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvlive", REPO / "daemon/tv-live-detect.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

N_SEGS = 300
# Block bei Sekunde 10..40 des ausgewerteten merged.ts (25 fps)
CUTLIST = ("FILE PROCESSING COMPLETE 7500 FRAMES AT 2500\n"
           "-------------------------\n250\t1000\n")


class Antwort:
    def __init__(self, body):
        self._b = io.BytesIO(body)
        self.status = 200

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Fenster(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        namen = ("urlopen", "WORK_DIR", "cached_logo_path",
                 "channel_logo_smooth_s", "_live_config", "fetch_segment_http",
                 "blackframe_extend_ads", "silence_extend_ads", "save_live_ads")
        self.alt = {n: (L.urllib.request.urlopen if n == "urlopen"
                        else getattr(L, n)) for n in namen}
        self.alt_run = L.subprocess.run
        self.pdt = int(time.time()) - 600
        iso = datetime.fromtimestamp(self.pdt, timezone.utc).isoformat()
        m3u8 = ("#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:" + iso + "\n" +
                "".join(f"#EXTINF:1.0,\nseg_{i:05d}.ts\n" for i in range(N_SEGS)))
        L.urllib.request.urlopen = lambda *a, **kw: Antwort(m3u8.encode())
        L.WORK_DIR = self.tmp
        L.cached_logo_path = lambda slug: None
        L.channel_logo_smooth_s = lambda slug: 0
        L._live_config = lambda slug: {}
        self.fehlt = set()

        def fetch(slug, name, dest):
            if name in self.fehlt:
                dest.write_bytes(b"halb")      # abgebrochener Download
                raise OSError("timeout")
            dest.write_bytes(b"\0" * 10_000)
            return 10_000
        L.fetch_segment_http = fetch
        L.blackframe_extend_ads = lambda path, ads, channel_slug=None: ads
        L.silence_extend_ads = lambda path, ads, channel_slug=None: (
            ads, [(False, False)] * len(ads))
        self.posts = []
        L.save_live_ads = lambda state: self.posts.append(state)
        self.rc = 0
        L.subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, self.rc, CUTLIST if self.rc == 0 else "", "kaputt")

    def tearDown(self):
        for n, v in self.alt.items():
            if n == "urlopen":
                L.urllib.request.urlopen = v
            else:
                setattr(L, n, v)
        L.subprocess.run = self.alt_run

    def _alter_stand(self):
        alt = [[self.pdt + 50.0, self.pdt + 80.0]]
        return {"rtl": {"generated": 0, "ads": alt, "latest_seg": "x"}}, alt

    def test_lueckenlos(self):
        state = {}
        self.assertTrue(L.analyze("rtl", state))
        self.assertEqual(state["rtl"]["ads"], [[self.pdt + 10.0, self.pdt + 40.0]])

    def test_tvdetect_rc_behaelt_alten_stand(self):
        state, alt = self._alter_stand()
        self.rc = 1
        self.assertFalse(L.analyze("rtl", state))
        self.assertEqual(state["rtl"]["ads"], alt)
        self.assertEqual(self.posts, [])

    def test_tvdetect_timeout_behaelt_alten_stand(self):
        state, alt = self._alter_stand()

        def run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 600)
        L.subprocess.run = run
        self.assertFalse(L.analyze("rtl", state))
        self.assertEqual(state["rtl"]["ads"], alt)
        self.assertEqual(self.posts, [])

    def test_luecke_wertet_nur_den_schwanz_aus(self):
        self.fehlt = {"seg_00100.ts"}
        state = {}
        self.assertTrue(L.analyze("rtl", state))
        # Block liegt 10..40 s NACH der Luecke, also ab Segment 101
        self.assertEqual(state["rtl"]["ads"],
                         [[self.pdt + 111.0, self.pdt + 141.0]])
        # das halb geholte Segment gilt nicht als vorhanden
        self.assertFalse((self.tmp / "rtl" / "segs" / "seg_00100.ts").exists())

    def test_luecke_zu_spaet_kein_post(self):
        self.fehlt = {"seg_00250.ts"}
        state = {}
        self.assertFalse(L.analyze("rtl", state))
        self.assertEqual(self.posts, [])


if __name__ == "__main__":
    unittest.main()
