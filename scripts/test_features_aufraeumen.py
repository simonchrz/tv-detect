#!/usr/bin/env python3
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fa", Path(__file__).with_name("features-aufraeumen.py"))
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)
U = "a" * 32
V = "b" * 32
D = "dvr-kabel-eins-1784303804"
B = 1780000000
def f(u, t, key="fps100-l2-a1"): return "%s-%d-%s.npy" % (u, B + t if t < B else t, key)

def test_neuester_bleibt():
    assert fa.zu_loeschen([f(U, 10), f(U, 30), f(U, 20)], set()) == [f(U, 10), f(U, 20)]

def test_einzelner_stand_bleibt():
    assert fa.zu_loeschen([f(U, 10), f(V, 5)], set()) == []

def test_referenz_bleibt_auch_wenn_alt():
    assert fa.zu_loeschen([f(U, 10), f(U, 30)], {f(U, 10)}) == []

def test_verschiedene_keys_getrennt():
    assert fa.zu_loeschen([f(U, 10, "k1"), f(U, 30, "k2")], set()) == []

def test_fremde_dateien_unberuehrt():
    assert fa.zu_loeschen(["notizen.npy", f(U, 10), f(U, 20)], set()) == [f(U, 10)]

def test_stempel_numerisch_nicht_lexikalisch():
    # 999999999 (9-stellig) < 1780000000 (10-stellig), lexikalisch wäre es umgekehrt
    a, b = U + "-999999999-k.npy", U + "-1780000000-k.npy"
    assert fa.zu_loeschen([a, b], set()) == [a]

def test_dvr_form_erkannt():
    assert fa.zu_loeschen([f(D, 1784303900), f(D, 1785000000)], set()) == [f(D, 1784303900)]

def test_dvr_form_stempel_nicht_mit_start_verwechselt():
    m = fa.NAME.match(f(D, 1785000000))
    assert m and m.group(1) == D and m.group(2) == "1785000000"

if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
