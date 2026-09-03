#!/usr/bin/env python3
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("va", Path(__file__).with_name("vorlauf_auswerten.py"))
va = importlib.util.module_from_spec(spec); spec.loader.exec_module(va)
# Index 12 = Blockanfang (Bilder -24..+4 in 2s -> 15 Bilder, Anfang bei Index 12)
I = 12

def test_sauberer_anfang_ist_ok():
    assert va.urteil("SSSSSSSSSSSS" + "WWW", I) == ("ok", 0)

def test_zwei_bilder_werbung_davor_ist_zu_spaet():
    assert va.urteil("SSSSSSSSSSWW" + "WWW", I) == ("zu_spaet", 2)

def test_ein_bild_davor_reicht_nicht():
    assert va.urteil("SSSSSSSSSSSW" + "WWW", I) == ("ok", 0)

def test_kette_bis_zum_fensteranfang():
    assert va.urteil("WWWWWWWWWWWW" + "WWW", I) == ("zu_spaet", 12)

def test_kontrolle_sendung_macht_unklar():
    # Agent haelt den Block selbst fuer Sendung -> kein Beleg, nicht "ok"
    assert va.urteil("WWWWWWWWWWWW" + "WSS", I) == ("unklar", 0)

def test_anfangsbild_sendung_macht_unklar():
    assert va.urteil("WWWWWWWWWWWW" + "SWW", I) == ("unklar", 0)

def test_luecke_bricht_die_kette():
    assert va.urteil("WWWWWWWWWSWW" + "WWW", I) == ("zu_spaet", 2)

if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
