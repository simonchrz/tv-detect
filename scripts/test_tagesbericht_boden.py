#!/usr/bin/env python3
"""tagesbericht: der Boden ist dieselbe Rechnung wie im Gate.

Bis 2026-09-25 fehlten der Kopie die Filter auf `missing` und
`select_rule`; eine Zeile mit fehlenden Gepinnten oder aus der alten
Auswahlregel konnte als Latte im Bericht stehen.
"""
import importlib.util, json, sys, tempfile
from pathlib import Path

_H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tb", _H / "tagesbericht.py")
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

# Der Dekoder, den das Gate gerade misst — nicht hart verdrahtet, sonst bricht
# der Test bei jedem Dekoder-Wechsel (zuletzt Inselsperre 2026-10-05).
DEC = tb._golden_bestwert()[1]


def z(ts, med, **k):
    e = {"ts": ts, "golden_median": med, "deployed": True, "set_hash": "S",
         "label_hash": "L", "decoder": DEC, "select_rule": "ensemble"}
    e.update(k)
    return e


GT = [
    z("20260901T0300", 0.990, missing=["x"]),              # nicht komposition-konstant
    z("20260902T0300", 0.980, select_rule="median-seed"),   # alte Regel
    z("20260903T0300", 0.980, select_rule=None),            # alte Regel (Default)
    z("20260904T0300", 0.950),
    z("20260905T0300", 0.940),
    z("20260906T0300", 0.930),
    z("20260907T0300", 0.945, deployed=False),              # heute
]


def _mit_datei(gt, fn):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "golden-trend.jsonl"
        p.write_text("".join(json.dumps(e) + "\n" for e in gt))
        return fn(p)


def test_missing_und_alte_regel_zaehlen_nicht():
    best, champ = _mit_datei(GT, lambda p: tb.boden_und_champion(GT, p))
    # Zweitbester der drei gueltigen Tage 0.950/0.940/0.930
    assert best["golden_median"] == 0.940, best
    assert champ["ts"] == "20260906T0300", champ


def test_gleich_dem_gate():
    bw, _ = tb._golden_bestwert()
    gate = _mit_datei(GT, lambda p: bw(p, "S", "L", ohne_ts="20260907T0300",
                                       select_rule="ensemble"))
    best, _ = _mit_datei(GT, lambda p: tb.boden_und_champion(GT, p))
    assert (best["golden_median"], best["ts"]) == gate, (best, gate)


def test_keine_eigene_kopie():
    q = (_H / "tagesbericht.py").read_text()
    assert "je_tag" not in q, "golden_bestwert gehoert importiert, nicht kopiert"


def test_ohne_passende_zeile_keine_latte():
    gt = [z("20260901T0300", 0.99, missing=["x"]), z("20260902T0300", 0.9)]
    best, champ = _mit_datei(gt, lambda p: tb.boden_und_champion(gt, p))
    assert best is None and champ is None, (best, champ)


if __name__ == "__main__":
    fails = 0
    for n, t in sorted(globals().items()):
        if n.startswith("test_"):
            try: t(); print("ok  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
