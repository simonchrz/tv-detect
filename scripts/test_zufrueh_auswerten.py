#!/usr/bin/env python3
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("za", Path(__file__).with_name("zufrueh_auswerten.py"))
za = importlib.util.module_from_spec(spec); spec.loader.exec_module(za)
# 14 Zeichen: [-4s][0s][+4s]…[+48s]
def f(s): 
    assert len(s) == 14, len(s)
    return s

def test_kontrolle_werbung_ab_anfang_ist_ok():
    assert za.urteil(f("S" + "W"*13)) == ("ok", 0.0)

def test_eine_sendungsminute_nach_dem_anfang():
    # 3 Bilder Sendung ab dem Anfang -> 12 s hineingeschnitten
    assert za.urteil(f("S" + "SSS" + "W"*10)) == ("zu_frueh", 12.0)

def test_werbung_vor_dem_anfang_ist_unbrauchbar():
    # Block laeuft schon -> die Frage stellt sich hier nicht
    assert za.urteil(f("W" + "SSS" + "W"*10)) == ("unbrauchbar", 0.0)

def test_durchgehend_sendung_wird_gedeckelt():
    art, s = za.urteil(f("S"*14))
    assert art == "gedeckelt" and s == 52.0, (art, s)

def test_luecke_bricht_nicht_rueckwaerts():
    # nur die ERSTE ununterbrochene Kette zaehlt
    assert za.urteil(f("S" + "SS" + "W" + "SSSSSSSSSS")) == ("zu_frueh", 8.0)

def test_zu_kurze_folge():
    assert za.urteil("SS") == ("unbrauchbar", 0.0)

if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
