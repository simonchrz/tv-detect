#!/usr/bin/env python3
"""zeitachsen-check: ein Abruffehler darf keine Aufnahme aus der Quarantaene holen."""
import importlib.util, io, sys, urllib.error
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "zc", Path(__file__).with_name("zeitachsen-check.py"))
zc = importlib.util.module_from_spec(spec); spec.loader.exec_module(zc)

A, B, C = "a" * 32, "b" * 32, "c" * 32


def _mit_urlopen(fn):
    alt = zc.urllib.request.urlopen
    zc.urllib.request.urlopen = fn
    return alt


def _http(code):
    def f(*a, **k):
        raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO())
    return f


def test_404_ist_kein_vod():
    alt = _mit_urlopen(_http(404))
    try:
        assert zc.voddauer(A) == 0.0
    finally:
        zc.urllib.request.urlopen = alt


def test_5xx_ist_abruffehler():
    alt = _mit_urlopen(_http(503))
    try:
        try:
            zc.voddauer(A)
        except zc.AbrufFehler:
            return
        raise AssertionError("503 wurde nicht als Abruffehler gemeldet")
    finally:
        zc.urllib.request.urlopen = alt


def test_timeout_ist_abruffehler():
    def f(*a, **k):
        raise TimeoutError("timed out")
    alt = _mit_urlopen(f)
    try:
        try:
            zc.voddauer(A)
        except zc.AbrufFehler:
            return
        raise AssertionError("Timeout wurde nicht als Abruffehler gemeldet")
    finally:
        zc.urllib.request.urlopen = alt


def test_abruffehler_behaelt_alten_eintrag():
    res = [(A, None, None, "HTTP 503"),        # vorher versetzt, heute nicht messbar
           (B, 1000.0, 1000.0, 0.0),           # vorher versetzt, jetzt deckungsgleich
           (C, 1000.0, 900.0, 100.0)]          # neu versetzt
    v = zc.versatz_liste(res, 15.0, {A: 112.0, B: 30.0})
    assert v == {A: 112.0, C: 100.0}, v


def test_abruffehler_ohne_alten_eintrag_bleibt_draussen():
    assert zc.versatz_liste([(A, None, None, "x")], 15.0, {}) == {}


def test_kein_vod_wird_quarantaeniert():
    assert zc.versatz_liste([(A, 1000.0, 0.0, 1000.0)], 15.0, {}) == {A: 1000.0}


def test_pruefe_meldet_abruffehler(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / ("_rec_" + A)
        d.mkdir()
        (d / "x.txt").write_text("FILE PROCESSING COMPLETE 2500 FRAMES AT 2500\n")
        alt = _mit_urlopen(_http(500))
        try:
            r = zc.pruefe(d)
        finally:
            zc.urllib.request.urlopen = alt
    assert r is not None and r[0] == A and r[1] is None, r


def test_main_schreibt_alten_eintrag_zurueck():
    # Ende-zu-Ende ohne Netz und ohne ~/.cache: ARCHIV auf ein Temp-Verzeichnis.
    import json, tempfile
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "hls"; root.mkdir()
        d = root / ("_rec_" + A); d.mkdir()
        (d / "ads_user.json").write_text("{}")
        (d / "x.txt").write_text("FILE PROCESSING COMPLETE 2500 FRAMES AT 2500\n")
        arch = Path(t) / "archiv"; arch.mkdir()
        (arch / "zeitachsen-versatz.json").write_text(
            json.dumps({"versetzt": {A: 112.0}}))
        alt_a, alt_argv, alt_url = zc.ARCHIV, sys.argv, _mit_urlopen(_http(502))
        zc.ARCHIV, sys.argv = arch, ["x", "--hls-root", str(root)]
        try:
            zc.main()
        finally:
            zc.ARCHIV, sys.argv = alt_a, alt_argv
            zc.urllib.request.urlopen = alt_url
        neu = json.loads((arch / "zeitachsen-versatz.json").read_text())
    assert neu["versetzt"] == {A: 112.0}, neu
    assert neu["abruf_fehler"] == [A], neu


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
