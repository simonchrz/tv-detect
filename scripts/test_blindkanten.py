#!/usr/bin/env python3
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("bk", Path(__file__).with_name("blindkanten.py"))
bk = importlib.util.module_from_spec(spec); spec.loader.exec_module(bk)

def test_unten_links_wird_oben_links():
    # Kasten am unteren Bildrand -> in Pixeln UNTEN, nicht oben.
    x, y, w, h = bk.in_pixel((0.0, 0.0, 1.0, 0.1), 1000, 1000, rand=0.0)
    assert (x, w) == (0, 1000)
    assert y == 900, y

def test_oben_bleibt_oben():
    x, y, w, h = bk.in_pixel((0.0, 0.9, 1.0, 0.1), 1000, 1000, rand=0.0)
    assert y == 0, y

def test_rand_vergroessert_beidseitig():
    _, _, w, h = bk.in_pixel((0.4, 0.4, 0.2, 0.2), 1000, 1000, rand=0.05)
    assert w == 300 and h == 300, (w, h)

def test_rand_klemmt_am_bildrand():
    x, y, w, h = bk.in_pixel((0.0, 0.0, 0.2, 0.2), 1000, 1000, rand=0.05)
    assert x == 0 and x + w <= 1000 and y + h <= 1000

def test_zwei_spalten_knallen():
    class P: stdout = "/a.png\tText\n"
    import subprocess
    alt = subprocess.run
    subprocess.run = lambda *a, **k: P()
    try:
        bk.rahmen_lesen([Path("/a.png")]); raise SystemExit("kein Fehler geworfen")
    except bk.RahmenFehlt:
        pass
    finally:
        subprocess.run = alt

def test_leerer_text_ist_kein_fehler():
    class P: stdout = "/a.png\t\t\n"
    import subprocess
    alt = subprocess.run
    subprocess.run = lambda *a, **k: P()
    try:
        assert bk.rahmen_lesen([Path("/a.png")]) == {"/a.png": []}
    finally:
        subprocess.run = alt


def test_ueberlappung_erkannt():
    assert bk.ueberlappt((0.1,0.1,0.2,0.2), (0.2,0.2,0.2,0.2))

def test_beruehrung_zaehlt_nicht():
    # Kante an Kante ist keine Ueberlappung — sonst gilt jeder Nachbar als Leck.
    assert not bk.ueberlappt((0.1,0.1,0.1,0.1), (0.2,0.1,0.1,0.1))

def test_leck_nur_im_geschwaerzten_bereich():
    vorher = [(0.1,0.1,0.3,0.1)]
    rest_drin = (0.15,0.12,0.05,0.05)     # "ERBUN" von "WERBUNG"
    rest_woanders = (0.8,0.8,0.1,0.05)    # Senderlogo
    assert bk.lecks(vorher, [rest_drin, rest_woanders]) == [rest_drin]

def test_ohne_schwaerzung_kein_leck():
    assert bk.lecks([], [(0.5,0.5,0.1,0.1)]) == []

if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except Exception as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
