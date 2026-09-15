#!/usr/bin/env python3
"""Ein Modul-Alias darf nirgends von einer lokalen Variable beschattet werden.

Gefunden 2026-09-15, im Nightly: `main()` in train-head.py benutzt `_ta`
seit jeher lokal fuer `_augment_test_recs`. Ein am 14.09. ergaenzter
MODUL-Alias `_ta` (scripts/test_ausschluss.py) war damit in der ganzen
Funktion unerreichbar — Python haelt einen Namen fuer lokal, sobald die
Funktion ihm IRGENDWO zuweist. Der Lauf starb nach 40 Minuten Vorlauf mit

    UnboundLocalError: cannot access local variable '_ta'

`ast.parse` und `--help` sehen das nicht: die Zeile liegt tief in einem
Pfad, den nur ein echter Trainingslauf durchlaeuft. Deshalb diese Pruefung
statisch — sie kostet Millisekunden und haette die Nacht gerettet.
"""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATEIEN = ["scripts/train-head.py", "scripts/golden_v3_vorschlag.py",
           "scripts/massstab-audit.py", "scripts/label-widerspruch.py",
           "scripts/agent-review.py"]


def modul_aliase(baum):
    """Namen, die auf Modulebene zugewiesen werden (ohne __dunder__)."""
    raus = set()
    for knoten in baum.body:
        if isinstance(knoten, ast.Assign):
            for ziel in knoten.targets:
                if isinstance(ziel, ast.Name) and not ziel.id.startswith("__"):
                    raus.add(ziel.id)
    return raus


def lokale_zuweisungen(fn):
    """Namen, denen die Funktion irgendwo zuweist — genau Pythons Regel."""
    raus = set()
    for k in ast.walk(fn):
        if isinstance(k, ast.Assign):
            for z in k.targets:
                if isinstance(z, ast.Name):
                    raus.add(z.id)
        elif isinstance(k, (ast.AugAssign, ast.AnnAssign)) and isinstance(k.target, ast.Name):
            raus.add(k.target.id)
        elif isinstance(k, (ast.For, ast.comprehension)) and isinstance(getattr(k, "target", None), ast.Name):
            raus.add(k.target.id)
        elif isinstance(k, ast.withitem) and isinstance(k.optional_vars, ast.Name):
            raus.add(k.optional_vars.id)
    return raus


class KeinAliasWirdBeschattet(unittest.TestCase):
    def test_alle_dateien(self):
        for name in DATEIEN:
            pfad = REPO / name
            if not pfad.is_file():
                continue
            baum = ast.parse(pfad.read_text())
            aliase = modul_aliase(baum)
            for fn in ast.walk(baum):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # `global X` hebt die Beschattung wieder auf.
                globale = {n for k in ast.walk(fn)
                           if isinstance(k, ast.Global) for n in k.names}
                lokal = lokale_zuweisungen(fn) - globale
                benutzt = {k.id for k in ast.walk(fn)
                           if isinstance(k, ast.Name) and isinstance(k.ctx, ast.Load)}
                schatten = aliase & lokal & benutzt
                # Nur Aliase, die die Funktion auch LIEST — wer einen
                # Modulnamen bloss als lokalen Namen wiederverwendet und ihn
                # nie als Modul liest, kann nichts kaputtmachen.
                for n in sorted(schatten):
                    erste_last = min(k.lineno for k in ast.walk(fn)
                                     if isinstance(k, ast.Name) and k.id == n
                                     and isinstance(k.ctx, ast.Load))
                    erste_zuw = min(k.lineno for k in ast.walk(fn)
                                    if isinstance(k, ast.Name) and k.id == n
                                    and isinstance(k.ctx, ast.Store))
                    if erste_last < erste_zuw:
                        self.fail(
                            f"{name}:{erste_last}: '{n}' ist ein Modul-Alias "
                            f"(Zeile {min(k.lineno for k in baum.body if isinstance(k, ast.Assign) and any(isinstance(z, ast.Name) and z.id == n for z in k.targets))}), "
                            f"wird in '{fn.name}' aber ab Zeile {erste_zuw} lokal "
                            f"zugewiesen — die Last in Zeile {erste_last} laeuft "
                            f"in UnboundLocalError. Alias umbenennen.")


if __name__ == "__main__":
    unittest.main()
