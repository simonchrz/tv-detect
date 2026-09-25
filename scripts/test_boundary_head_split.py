#!/usr/bin/env python3
"""train-boundary-head: Split aus dem Ledger, nie versiegelt/quarantaeniert,
keine maschinellen Labels.

Bis 2026-09-25 hatte das Skript einen eigenen 20-%-Hash-Split; versiegelte
und Golden-Aufnahmen landeten im Training.
"""
import contextlib, importlib.util, io, json, sys, tempfile
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "tbh", Path(__file__).with_name("train-boundary-head.py"))
tbh = importlib.util.module_from_spec(spec); spec.loader.exec_module(tbh)

LEDGER = {"tr": "train", "te": "test", "vs": "versiegelt", "gp": "train",
          "qu": "train"}
GOLDEN = {"gp"}
VERSETZT = {"qu"}


def test_eimer():
    e = lambda u: tbh.eimer(u, LEDGER, GOLDEN, VERSETZT)
    assert e("tr") == "train"
    assert e("te") == "test"
    assert e("vs") is None, "versiegelt darf weder trainiert noch getestet werden"
    assert e("gp") == "test", "Golden-Pin ist immer test"
    assert e("qu") is None, "Quarantaene (zeitachsen-versatz) wird nicht verwendet"
    assert e("neu") is None, "nicht im Ledger = nicht verwenden"


def test_ausschluss_nie_test():
    u = sorted(tbh._aus.TEST_SET_EXCLUDE)[0]
    assert tbh.eimer(u, {u: "test"}, set(), set()) == "train"


def test_label_maschinell():
    with tempfile.TemporaryDirectory() as t:
        for name, d, erw in (
                ("m", {"ads": [[1, 2]]}, False),
                ("liste", [[1, 2]], False),
                ("golden", {"ads": [], "reviewed_by": "golden-audit"}, False),
                ("ac", {"ads": [], "auto_confirmed_at": 1}, True),
                ("fp", {"ads": [], "auto_confirmed_via_fingerprint": True}, True),
                ("ag", {"ads": [], "reviewed_by": "agent-review.py"}, True)):
            p = Path(t) / name; p.mkdir()
            (p / "ads_user.json").write_text(json.dumps(d))
            assert tbh.label_maschinell(p) is erw, name


def _rec(root, cache, uuid, ads, extra=None):
    d = root / f"_rec_{uuid}"; d.mkdir()
    doc = {"ads": ads}; doc.update(extra or {})
    (d / "ads_user.json").write_text(json.dumps(doc))
    rng = np.random.default_rng(sum(map(ord, uuid)))
    np.save(cache / f"{uuid}-1780000000-k.npy",
            rng.random((200, 1282)).astype(np.float32))


def test_main_haelt_versiegelt_und_quarantaene_raus():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        root, cache, arch = t / "hls", t / "cache", t / "arch"
        for d in (root, cache, arch): d.mkdir()
        ads = [[50, 120]]
        for u in ("tr1", "tr2", "vs", "qu", "neu", "te", "gp", "ac"):
            _rec(root, cache, u, ads,
                 {"auto_confirmed_at": 1} if u == "ac" else None)
        (arch / "split-ledger.json").write_text(json.dumps(
            {"tr1": "train", "tr2": "train", "vs": "versiegelt", "qu": "train",
             "te": "test", "gp": "train", "ac": "train"}))
        (arch / "golden-eval-set.json").write_text(json.dumps({"uuids": ["gp"]}))
        (arch / "zeitachsen-versatz.json").write_text(
            json.dumps({"versetzt": {"qu": 112.0}}))
        alt = sys.argv
        sys.argv = ["x", "--hls-root", str(root), "--feature-cache", str(cache),
                    "--train-archive", str(arch), "--no-write",
                    "--max-iter", "5", "--hidden-dim", "4"]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                tbh.main()
        finally:
            sys.argv = alt
    o = out.getvalue()
    assert "2 train+pos" in o and "2 test" in o, o
    assert "3 versiegelt/quarantaeniert/nicht im Ledger" in o, o
    assert "1 mit maschinellem Label" in o, o


def test_ohne_ledger_abbruch():
    with tempfile.TemporaryDirectory() as t:
        try:
            tbh.lade_split(t)
        except Exception:
            return
    raise AssertionError("fehlendes Ledger muss abbrechen, nicht auf Hash zurueckfallen")


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, str(e)[-400:])
    sys.exit(1 if fails else 0)
