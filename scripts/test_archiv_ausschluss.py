#!/usr/bin/env python3
"""O25: `--archiv-ausschluss` darf nur die EINSPEISUNG betreffen.

Die 90 gelisteten Archiv-Eintraege kamen nur ueber den rec_dir-Defekt ins
Archiv (2026-05-03 bis 2026-09-23). Uebersprungen statt geloescht (L6): das
Archiv bleibt, der Schalter ist umkehrbar. Live-Aufnahmen trainieren ueber
den Live-Durchgang weiter — der Ausschluss muss HINTER der Live-Pruefung
stehen und VOR dem Laden, sonst misst die Serie etwas anderes.
"""
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUELL = (REPO / "scripts/train-head.py").read_text()
LISTE = json.loads((REPO / "docs/archiv-ausschluss-o25.json").read_text())
REGEL = (REPO / "docs/o25-archiv-bereinigung-preregistration.md").read_text()


class ArchivAusschluss(unittest.TestCase):
    def test_vorgabe_ist_kein_ausschluss(self):
        self.assertIn('ap.add_argument("--archiv-ausschluss", default=None,', QUELL)

    def test_ausschluss_nach_live_und_vor_dem_laden(self):
        a = QUELL.index('for npz_path in sorted(archive_dir.glob("*.npz")):')
        b = QUELL[a:a + 1200]
        self.assertLess(b.index("if u in live_uuids:"), b.index("if u in _ausschluss:"))
        self.assertLess(b.index("if u in _ausschluss:"), b.index("np.load(npz_path"))

    def test_liste_ist_eindeutig_und_vollstaendig(self):
        u = LISTE["uuids"]
        self.assertEqual(len(u), 90)
        self.assertEqual(len(set(u)), 90)
        self.assertEqual(u, sorted(u))
        self.assertIn("rec_dir", LISTE["grund"])

    def test_arme_stehen_in_registry_und_regel(self):
        m = re.search(r"```regel\n(.*?)\n```", REGEL, re.S)
        regel = json.loads(m.group(1))
        self.assertEqual(regel["id"], "O25")
        for arm in regel["arme"].values():
            self.assertIn(f'_ts_arme["{arm}"] = (_ident, 32)', QUELL)

    def test_plist_faehrt_o25(self):
        p = (REPO / "daemon/launchd/com.user.tv-tagesserie.plist").read_text()
        self.assertIn("<string>mlp32-bereinigt,mlp32-archivalt</string>", p)
        self.assertIn("archiv-ausschluss-o25.json", p)


if __name__ == "__main__":
    unittest.main()
