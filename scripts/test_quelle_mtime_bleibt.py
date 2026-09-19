#!/usr/bin/env python3
"""Ein Cache-Treffer darf die mtime der Quelle nicht verschieben.

Die mtime der Quell-.ts ist der Schluessel des Merkmals-Caches. Bis
2026-09-19 setzte get_source() sie bei jedem Treffer per touch() neu, und
das naechste Training extrahierte alles, was der Daemon angefasst hatte,
noch einmal (103 Aufnahmen, 89 Minuten in der Nacht zum 19.09.).
"""
import ast
import os
import tempfile
import time
import unittest
from pathlib import Path

DAEMON = Path(__file__).resolve().parent.parent / "daemon/tv-thumbs-daemon.py"
SRC = DAEMON.read_text()


def _funktion(name):
    for k in ast.parse(SRC).body:
        if isinstance(k, ast.FunctionDef) and k.name == name:
            ns = {"os": os, "time": time}
            exec(ast.get_source_segment(SRC, k), ns)
            return ns[name]
    raise AssertionError(f"{name} fehlt im Daemon")


class QuelleMtime(unittest.TestCase):
    def test_atime_neu_mtime_unveraendert(self):
        f = Path(tempfile.mkdtemp()) / "x.ts"
        f.write_bytes(b"0" * 10)
        alt = 1_780_000_000
        os.utime(f, (alt, alt))
        _funktion("_atime_auffrischen")(f)
        st = os.stat(f)
        self.assertEqual(int(st.st_mtime), alt, "mtime verschoben — Merkmals-Cache verliert den Schluessel")
        self.assertGreater(st.st_atime, alt + 1000, "atime nicht aufgefrischt — LRU raeumt falsch")

    def test_fehlende_datei_wirft_nicht(self):
        _funktion("_atime_auffrischen")(Path("/nicht/da.ts"))

    def test_kein_touch_im_daemon(self):
        # Auf dem Syntaxbaum, nicht auf dem Text: der Docstring von
        # _atime_auffrischen zitiert den alten Aufruf als Geschichte.
        treffer = [k.lineno for k in ast.walk(ast.parse(SRC))
                   if isinstance(k, ast.Call) and isinstance(k.func, ast.Attribute)
                   and k.func.attr == "touch"]
        self.assertEqual(treffer, [], f"touch() setzt die mtime mit (Zeilen {treffer})")

    def test_lru_liest_atime(self):
        i = SRC.index("def _maybe_evict_source_cache")
        self.assertIn("st_atime", SRC[i:i + 3000])


if __name__ == "__main__":
    unittest.main()
