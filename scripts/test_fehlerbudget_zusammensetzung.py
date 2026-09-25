#!/usr/bin/env python3
"""fehlerbudget: nicht gemessene Aufnahmen werden benannt, und der Trend
vergleicht nur Laeufe ueber denselben Satz.

Ende-zu-Ende ohne Binary, ohne Netz und ohne ~/.cache: HOME, Messsatz,
Dumps und Replay werden auf ein Temp-Verzeichnis umgebogen.
"""
import contextlib, importlib.util, io, json, os, sys, tempfile, types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fb", Path(__file__).with_name("fehlerbudget.py"))
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)

A, B, C = "a" * 32, "b" * 32, "c" * 32
LABELS = {A: [(100.0, 200.0)], B: [(50.0, 150.0)], C: [(10.0, 90.0)]}


def lauf(t, labels, kaputt=()):
    """Ein Lauf von main(); gibt (stdout, Trendzeilen) zurueck."""
    t = Path(t)
    arch = t / ".cache/tvd-train-archive"; arch.mkdir(parents=True, exist_ok=True)
    (arch / "split-ledger.json").write_text(json.dumps({A: "test", B: "train", C: "train"}))
    ms = arch / "messsatz.json"
    ms.write_text(json.dumps({"name": "m", "hash": "h1", "uuids": [A, B, C]}))
    dumps = t / "dumps"; dumps.mkdir(exist_ok=True)
    for u in (A, B, C):
        (dumps / f"{u}.json").write_text(json.dumps({"fps": 1, "nn_confs": [0.1] * 300}))
    alt = (os.environ.get("HOME"), fb.MESSSATZ, fb.BILD, fb.replay, fb._lade, sys.argv)
    os.environ["HOME"] = str(t)
    fb.MESSSATZ, fb.BILD = ms, t / "bild"
    fb.replay = lambda p: None if Path(p).name.startswith(tuple(kaputt)) else [(100.0, 200.0)]
    fb._lade = lambda *a: types.SimpleNamespace(menschlabels=lambda: labels)
    sys.argv = ["x", "--dumps", str(dumps), "--trend", str(t / "trend.jsonl")]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            fb.main()
    finally:
        h, fb.MESSSATZ, fb.BILD, fb.replay, fb._lade, sys.argv = alt
        os.environ["HOME"] = h
    zeilen = [json.loads(x) for x in (t / "trend.jsonl").read_text().splitlines()]
    return out.getvalue(), zeilen


def test_replay_fehler_benannt_und_kein_falscher_labelalarm():
    with tempfile.TemporaryDirectory() as t:
        lauf(t, LABELS)
        out, z = lauf(t, LABELS, kaputt=(B,))
    assert z[-1]["n"] == 2 and z[-1]["n_soll"] == 3, z[-1]
    assert z[-1]["fehlend"] == {B: "Replay fehlgeschlagen"}, z[-1]
    assert B in out and "NICHT gemessen" in out, out
    assert "ZUSAMMENSETZUNG GEAENDERT" in out, out
    assert "LABELS HABEN SICH GEAENDERT" not in out, out
    # label_hash ueber den ganzen Satz: ein Replay-Fehler aendert ihn nicht
    assert z[-1]["label_hash"] == z[0]["label_hash"], z


def test_fehlendes_menschenlabel_benannt():
    with tempfile.TemporaryDirectory() as t:
        out, z = lauf(t, {A: LABELS[A], B: LABELS[B]})
    assert z[-1]["fehlend"] == {C: "kein Menschenlabel"}, z[-1]
    assert C in out, out


def test_gleicher_satz_wird_verglichen():
    with tempfile.TemporaryDirectory() as t:
        lauf(t, LABELS)
        out, _ = lauf(t, LABELS)
    assert "ZUSAMMENSETZUNG" not in out and "Labels unveraendert" in out, out


def test_label_summe_ausgegeben():
    with tempfile.TemporaryDirectory() as t:
        out, _ = lauf(t, LABELS)
    teil = out.split("=== Label-Seite")[1]
    assert "SUMME" in teil, teil


def test_alte_zeile_ohne_fehlend():
    assert fb.vergleich_mit_vorlauf({"n": 3}, {}, 3) is None
    assert fb.vergleich_mit_vorlauf({"n": 2}, {}, 3)
    assert fb.vergleich_mit_vorlauf({"n": 3}, {B: "x"}, 3)


def test_gleiche_fehlende_sind_vergleichbar():
    assert fb.vergleich_mit_vorlauf({"n": 2, "fehlend": {B: "x"}}, {B: "y"}, 3) is None


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
