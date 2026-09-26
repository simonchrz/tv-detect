#!/usr/bin/env python3
"""Keine lokale Variable darf eine Modulfunktion verschatten, die dieselbe
Funktion aufruft.

Anlass 2026-09-26: main() im tv-thumbs-daemon hatte `_hls_gate = [False]`
(Log-Flanke) — der Sweep fuehrte eine Modulfunktion `_hls_gate(...)` ein und
rief sie in main() auf. Python bindet den Namen dann in der GANZEN Funktion
lokal: der Daemon starb beim ersten Zyklus mit "'list' object is not
callable". Die Tests der herausgezogenen Funktionen sahen main() nie.
"""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATEIEN = sorted((REPO / "daemon").glob("*.py"))


def verschattungen(quelle):
    baum = ast.parse(quelle)
    modul_fn = {n.name for n in baum.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    funde = []
    for fn in ast.walk(baum):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lokal, gerufen = set(), set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                lokal.add(n.id)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                gerufen.add(n.func.id)
        for name in sorted(lokal & gerufen & modul_fn):
            funde.append(f"{fn.name}() bindet {name} lokal und ruft es auf")
    return funde


class KeineVerschattung(unittest.TestCase):
    def test_daemons(self):
        self.assertTrue(DATEIEN)
        for d in DATEIEN:
            with self.subTest(datei=d.name):
                self.assertEqual(verschattungen(d.read_text()), [])

    def test_erkennt_den_fall(self):
        q = "def g():\n  pass\ndef main():\n  g = [1]\n  g()\n"
        self.assertEqual(verschattungen(q), ["main() bindet g lokal und ruft es auf"])


if __name__ == "__main__":
    unittest.main()
