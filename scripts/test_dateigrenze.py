#!/usr/bin/env python3
"""Der Nightly darf nicht an der launchd-Dateigrenze verhungern.

In der Nacht zum 2026-09-17 startete launchd train-head.py mit einer weichen
Grenze von 256 offenen Dateien. Jede per mmap eingespeiste Archiv-Aufnahme
haelt einen Deskriptor; nach ~250 warf np.load EMFILE, ein stilles
`except Exception: continue` schluckte das, und der Kopf lernte auf 245
statt 604 Archiv-Aufnahmen. Die Messlaeufe davor liefen im Terminal
(Grenze > 1 Mio.) und konnten es nicht sehen.

Dieser Test stellt die Nachtumgebung nach: Kindprozess mit Grenze 256.
"""
import ast
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
SRC = (HIER / "train-head.py").read_text()


def _funktion(name):
    baum = ast.parse(SRC)
    for knoten in baum.body:
        if isinstance(knoten, ast.FunctionDef) and knoten.name == name:
            return ast.get_source_segment(SRC, knoten)
    raise AssertionError(f"{name} fehlt in train-head.py")


KIND = textwrap.dedent('''
    import resource, sys, tempfile
    from pathlib import Path
    import numpy as np
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (256, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    {helfer}
    heben = sys.argv[1] == "ja"
    if heben:
        _dateigrenze_anheben()
    d = Path(tempfile.mkdtemp())
    halt = []
    for i in range(600):
        p = d / f"{{i}}.npy"
        np.save(p, np.zeros((4, 3), np.float32))
        halt.append(np.load(p, mmap_mode="c"))
    print(len(halt))
''')


class Dateigrenze(unittest.TestCase):
    def _lauf(self, heben):
        code = KIND.format(helfer=_funktion("_dateigrenze_anheben"))
        return subprocess.run([sys.executable, "-c", code,
                               "ja" if heben else "nein"],
                              capture_output=True, text=True)

    def test_ohne_anheben_scheitert_wie_in_der_nacht(self):
        # Gegenprobe: ohne sie waere der naechste Test bedeutungslos.
        r = self._lauf(False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Too many open files", r.stderr)

    def test_mit_anheben_laden_600_archive(self):
        r = self._lauf(True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(r.stdout.strip(), "600")

    def test_main_hebt_direkt_nach_dem_parsen(self):
        self.assertRegex(SRC, r"args = ap\.parse_args\(\)\n"
                              r"\s+_dg_vor, _dg_nach = _dateigrenze_anheben\(\)")

    def test_emfile_wird_nicht_verschluckt(self):
        # Beide Ladestellen der Archiv-Einspeisung muessen EMFILE durchreichen.
        block = SRC[SRC.index("injected = 0"):SRC.index("train-archive: injected")]
        self.assertEqual(len(re.findall(r"errno\.EMFILE:\s*\n\s*raise", block)), 2)


if __name__ == "__main__":
    unittest.main()
