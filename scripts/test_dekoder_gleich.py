#!/usr/bin/env python3
"""Detect (tv-thumbs-daemon) und Gate (train-head EVAL_DECODER) muessen mit
denselben Dekoder-Flags laufen — sonst misst das Gate einen anderen Dekoder
als die Produktion schneidet (2026-10-05, Inselsperre)."""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class Tests(unittest.TestCase):
    def test_gleiche_flags(self):
        th = (REPO / "scripts/train-head.py").read_text()
        ev = re.search(r"^EVAL_DECODER = \[(.*?)\]", th, re.M).group(1)
        ev = re.findall(r'"([^"]+)"', ev)
        dm = (REPO / "daemon/tv-thumbs-daemon.py").read_text()
        i = dm.index('"--decoder", "hsmm"')
        dm_flags = re.findall(r'"([^"]+)"', dm[i:dm.index("]", i)])
        self.assertEqual(ev, dm_flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
