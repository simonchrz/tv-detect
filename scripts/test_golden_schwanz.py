"""golden_schwanz() aus loop-status.py — die Entscheidungen, nicht die Ausgabe.

Was hier abgesichert ist: (1) auf Ablehnungs-Naechten zaehlt der Champion,
nicht der verworfene Kandidat; (2) Label-Bloecke unter min_block_s gelten
als unerreichbar und deckeln die Note; (3) which=auto wird als Echo
markiert; (4) Beharrlichkeit zaehlt Naechte im Schwanz, nicht Werte;
(5) Naechte mit zu wenig Golden-Werten fallen raus, statt den Schwanz mit
einer halben Nacht zu verfaelschen.
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "loop_status", Path(__file__).with_name("loop-status.py"))
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)

GOLDEN = {"a", "b", "c", "d"}
META = {
    "a": {"title": "A", "which": "merged", "ads": [(0, 100), (500, 530), (900, 1000)]},
    "b": {"title": "B", "which": "auto", "ads": [(0, 100)]},
    "c": {"title": "C", "which": "merged", "ads": [(0, 100)]},
    "d": {"title": "D", "which": "merged", "ads": []},
}


def nacht(ts, deploy, cand, champ):
    return {"ts": ts, "deploy": deploy, "candidate": cand, "champion": champ}


def test_ablehnungsnacht_zaehlt_champion():
    # Kandidat abgelehnt: die Produktion faehrt weiter mit dem Champion,
    # also muss der Schwanz dessen Werte zeigen.
    e = [nacht("20260901T", False,
               {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
               {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})]
    s = ls.golden_schwanz(e, GOLDEN, META.get, n=1)
    assert s["schlechteste"][0]["uuid"] == "d"
    assert s["schlechteste"][0]["iou"] == 0.6


def test_deploynacht_zaehlt_kandidat():
    e = [nacht("20260901T", True,
               {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
               {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})]
    s = ls.golden_schwanz(e, GOLDEN, META.get, n=1)
    assert s["schlechteste"][0]["uuid"] == "a"


def test_kurzer_block_ist_unerreichbar_und_deckelt():
    e = [nacht("20260901T", True, {"a": 0.5, "b": 0.9, "c": 0.9, "d": 0.9}, {})]
    s = ls.golden_schwanz(e, GOLDEN, META.get, n=1, min_block_s=60)
    r = s["schlechteste"][0]
    assert r["uuid"] == "a"
    assert r["unerreichbar"] == [(500, 530)]
    assert abs(r["decke"] - 2 / 3) < 1e-9
    # genau min_block_s lang ist NICHT zu kurz
    s2 = ls.golden_schwanz(e, GOLDEN, META.get, n=1, min_block_s=30)
    assert s2["schlechteste"][0]["unerreichbar"] == []
    assert s2["schlechteste"][0]["decke"] == 1.0


def test_auto_ist_echo():
    e = [nacht("20260901T", True, {"a": 0.9, "b": 0.5, "c": 0.9, "d": 0.9}, {})]
    s = ls.golden_schwanz(e, GOLDEN, META.get, n=2)
    nach = {r["uuid"]: r for r in s["schlechteste"]}
    assert nach["b"]["echo"] is True
    assert nach["a"]["echo"] is False


def test_beharrlich_zaehlt_naechte():
    # a ist in 3 von 4 Naechten Letzter, c nur einmal.
    naechte = [
        nacht("1", True, {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, {}),
        nacht("2", True, {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, {}),
        nacht("3", True, {"a": 0.9, "b": 0.9, "c": 0.1, "d": 0.9}, {}),
        nacht("4", True, {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, {}),
    ]
    s = ls.golden_schwanz(naechte, GOLDEN, META.get, n=1)
    assert s["n_naechte"] == 4
    assert s["beharrlich"] == [("a", 3)]
    assert s["schlechteste"][0]["im_schwanz"] == 3


def test_halbe_nacht_faellt_raus():
    # Eine Nacht mit nur einem Golden-Wert (Abbruch) darf die Serie nicht
    # bilden — sonst waere "Letzter" trivial.
    naechte = [
        nacht("1", True, {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, {}),
        nacht("2", True, {"c": 0.2}, {}),
    ]
    s = ls.golden_schwanz(naechte, GOLDEN, META.get, n=1)
    assert s["n_naechte"] == 1
    assert s["schlechteste"][0]["uuid"] == "a"


def test_leer_gibt_none():
    assert ls.golden_schwanz([], GOLDEN, META.get) is None
    assert ls.golden_schwanz([nacht("1", True, {"x": 0.1}, {})], GOLDEN, META.get) is None


def test_fehlende_meta_bricht_nicht():
    e = [nacht("1", True, {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, {})]
    s = ls.golden_schwanz(e, GOLDEN, lambda u: None, n=1)
    r = s["schlechteste"][0]
    assert r["title"] == "" and r["unerreichbar"] == [] and r["decke"] == 1.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as ex:
                fails += 1
                print(f"  FAIL {name}: {ex}")
    sys.exit(1 if fails else 0)
