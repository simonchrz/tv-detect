#!/usr/bin/env python3
"""corpus-label-audit: frischer Kopf vor dem Lesen, und alle Archivnamen.

1. head.bin kam aus dem Mac-Modell-Cache — direkt nach einem Deploy der
   Kopf von gestern (dieselbe Falle wie dumps-erneuern.modelle_frisch).
2. glob("dvr-*.npz") sah die 32-stelligen Hex-uuids nicht (93 von 912).
"""
import importlib.util, io, os, sys, tempfile, types
from contextlib import redirect_stdout
from pathlib import Path

_H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cla", _H / "corpus-label-audit.py")
cla = importlib.util.module_from_spec(spec); spec.loader.exec_module(cla)


class Halt(Exception):
    pass


def _main_bis_load_head(argv):
    """main() bis zum ersten load_head; gibt die Reihenfolge der Aufrufe."""
    reihe = []
    fake = types.SimpleNamespace(
        daemon_laden=lambda: "mod",
        modelle_frisch=lambda mod: reihe.append(("frisch", mod)) or [])
    def load_head(p):
        reihe.append(("load_head", os.path.basename(p)))
        raise Halt
    alt = (cla._dumps_erneuern, cla.load_head, sys.argv)
    cla._dumps_erneuern, cla.load_head = (lambda: fake), load_head
    sys.argv = ["x"] + argv
    try:
        with redirect_stdout(io.StringIO()):
            cla.main()
    except Halt:
        pass
    finally:
        cla._dumps_erneuern, cla.load_head, sys.argv = alt
    return reihe


def test_modelle_vor_dem_kopf_geholt():
    r = _main_bis_load_head([])
    assert r == [("frisch", "mod"), ("load_head", "head.bin")], r


def test_ohne_abruf_holt_nicht():
    assert _main_bis_load_head(["--ohne-abruf"]) == [("load_head", "head.bin")]


def test_dieselbe_funktion_wie_dumps_erneuern():
    de = cla._dumps_erneuern()
    assert callable(de.modelle_frisch)
    assert "def modelle_frisch" not in (_H / "corpus-label-audit.py").read_text()


def test_abruffehler_ist_laut():
    def kaputt():
        raise RuntimeError("Gateway weg")
    alt = cla._dumps_erneuern
    cla._dumps_erneuern = kaputt
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            assert cla.modelle_holen() is None
    finally:
        cla._dumps_erneuern = alt
    assert "nicht frisch geholt" in buf.getvalue()


def test_hex_uuids_im_archiv():
    with tempfile.TemporaryDirectory() as t:
        for n in ("dvr-rtl-1780078500.npz", "989a0bea63b249d1a6243d5f3f27e0ed.npz",
                  "notiz.npz", "zeitachsen-versatz.json", "ABCDEF.npz"):
            Path(t, n).write_bytes(b"")
        namen = [os.path.basename(f) for f in cla.archiv_dateien(t)]
    assert namen == ["989a0bea63b249d1a6243d5f3f27e0ed.npz",
                     "dvr-rtl-1780078500.npz"], namen


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
