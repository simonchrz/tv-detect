#!/usr/bin/env python3
"""Daemon: Chase-Remux (Aufnahme laeuft noch) — Sweep 2026-09-25.

  (a) scheitert ein Chase, nachdem die wachsende EVENT-Playlist schon live
      ist, bleibt die uuid in chase-offen.json, bis ein hls-done sie abloest;
      nach Aufnahmeende wird sie nachgebaut (sonst Teil-VOD fuer immer);
  (b) ffmpeg bekommt -rw_timeout, und ueber Aufnahmeende + Puffer hinaus
      wird der Chase beendet;
  (c) ein Chase haelt das Detect-Gate nicht zu.
"""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tvthumbs", REPO / "daemon/tv-thumbs-daemon.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class FakeProc:
    """ffmpeg-Ersatz: legt beim Start Segmente + Playlist im Zielordner an."""
    def __init__(self, cmd, rcs, **kw):
        self.cmd = cmd
        self.rcs = list(rcs)
        self.killed = False
        out = Path(cmd[-1]).parent
        for i in range(3):
            (out / f"seg_{i:05d}.ts").write_bytes(b"s")
        (out / "index.m3u8").write_text("#EXTM3U\n")

    def poll(self):
        if self.killed:
            return -9
        self.polls = getattr(self, "polls", 0) + 1
        if self.polls > 1000:
            return 99   # Notbremse: der Test darf nie ewig haengen
        return self.rcs.pop(0) if self.rcs else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class Basis(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.alt = dict(CHASE_OFFEN_FILE=D.CHASE_OFFEN_FILE,
                        _upload_files_put=D._upload_files_put,
                        _laufende_aufnahmen=D._laufende_aufnahmen,
                        http_post_stream=D.http_post_stream,
                        CHASE_PUFFER_S=D.CHASE_PUFFER_S,
                        get_source_mit_grund=D.get_source_mit_grund,
                        process_recording=D.process_recording)
        self.alt_popen, self.alt_run = D.subprocess.Popen, D.subprocess.run
        self.alt_sleep = D.time.sleep
        D.CHASE_OFFEN_FILE = self.tmp / "chase-offen.json"
        D._upload_files_put = lambda url, files: None
        D.http_post_stream = lambda *a, **kw: b""
        D.subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, "{}", "")
        D.time.sleep = lambda s: None
        self.procs = []

    def tearDown(self):
        for k, v in self.alt.items():
            setattr(D, k, v)
        D.subprocess.Popen, D.subprocess.run = self.alt_popen, self.alt_run
        D.time.sleep = self.alt_sleep
        D._failed_until.clear()

    def _popen(self, rcs):
        def popen(cmd, **kw):
            p = FakeProc(cmd, rcs)
            self.procs.append(p)
            return p
        D.subprocess.Popen = popen


class TeilVOD(Basis):
    def test_gescheiterter_chase_bleibt_vorgemerkt(self):
        D._laufende_aufnahmen = lambda: {"u": 9e12}
        self._popen([None, 1])        # erst laufen, dann rc=1
        self.assertFalse(D.process_recording("u", True, False, chase=True))
        self.assertIn("u", D._chase_offen_laden())

    def test_hls_done_traegt_aus(self):
        D._laufende_aufnahmen = lambda: {"u": 9e12}
        self._popen([None, 0])
        self.assertTrue(D.process_recording("u", True, False, chase=True))
        self.assertNotIn("u", D._chase_offen_laden())

    def test_kandidat_erst_nach_aufnahmeende(self):
        D._chase_offen_setzen("u")
        D._laufende_aufnahmen = lambda: {"u": 9e12}
        self.assertEqual(D._chase_nachbau_kandidaten(set(), set(), set()), [])
        D._laufende_aufnahmen = lambda: {}
        self.assertEqual(D._chase_nachbau_kandidaten(set(), set(), set()), ["u"])
        # noch in hls-pending, belegt, im Cooldown, Pi stumm → nicht
        self.assertEqual(D._chase_nachbau_kandidaten({"u"}, set(), set()), [])
        self.assertEqual(D._chase_nachbau_kandidaten(set(), {"u"}, set()), [])
        self.assertEqual(D._chase_nachbau_kandidaten(set(), set(), {"u"}), [])
        D._laufende_aufnahmen = lambda: None
        self.assertEqual(D._chase_nachbau_kandidaten(set(), set(), set()), [])

    def test_nachbau_nur_aus_lokaler_quelle(self):
        aufrufe = []
        D.process_recording = lambda *a, **kw: aufrufe.append(a) or True
        D._chase_offen_setzen("u")
        D.get_source_mit_grund = lambda u: (None, D.QUELLE_FEHLER)
        self.assertFalse(D._chase_nachbau("u"))
        self.assertEqual(aufrufe, [])
        self.assertIn("u", D._chase_offen_laden(), "vorlaeufig: bleibt vorgemerkt")
        D.get_source_mit_grund = lambda u: (self.tmp / "u.ts", D.QUELLE_OK)
        self.assertTrue(D._chase_nachbau("u"))
        self.assertEqual(aufrufe, [("u", True, False)])


class Grenzen(Basis):
    def test_rw_timeout_vor_der_eingabe(self):
        D._laufende_aufnahmen = lambda: {"u": 9e12}
        self._popen([0])
        D.process_recording("u", True, False, chase=True)
        cmd = self.procs[0].cmd
        self.assertIn("-rw_timeout", cmd)
        self.assertLess(cmd.index("-rw_timeout"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-rw_timeout") + 1],
                         str(D.CHASE_RW_TIMEOUT_US))

    def test_kein_rw_timeout_ohne_chase(self):
        self._popen([0])
        D.get_source_mit_grund = lambda u: (None, D.QUELLE_FEHLER)
        D.process_recording("u", True, False)
        self.assertNotIn("-rw_timeout", self.procs[0].cmd)

    def test_wandzeit_nach_aufnahmeende(self):
        D._laufende_aufnahmen = lambda: {}   # Aufnahme ist vorbei
        D.CHASE_PUFFER_S = -1                # Puffer schon abgelaufen
        self._popen([])                      # ffmpeg endet nie von selbst
        self.assertFalse(D.process_recording("u", True, False, chase=True))
        self.assertTrue(self.procs[0].killed)


class Gate(unittest.TestCase):
    def test_chase_haelt_detect_nicht_auf(self):
        live, aktiv = D._hls_gate([{"uuid": "u", "chase": True}], set(), 0)
        self.assertEqual(live, [])
        self.assertFalse(aktiv)

    def test_normaler_remux_haelt_auf(self):
        _, aktiv = D._hls_gate([{"uuid": "u"}], set(), 0)
        self.assertTrue(aktiv)
        _, aktiv = D._hls_gate([], set(), 1)
        self.assertTrue(aktiv)


if __name__ == "__main__":
    unittest.main()
