#!/usr/bin/env python3
"""Tests für die Herkunftsbestimmung des Maßstabs (`massstab-audit.py`).

⚠️ Diese Funktion beantwortet die einzige Frage, die über den Wert des
Maßstabs entscheidet: **war ein Mensch an diesem Label?** Sie irrt still —
ein Fehlurteil erzeugt keine Ausnahme, sondern eine zu gute Zahl.

Der teure Fall stand am 2026-09-06 fest: `which="merged"` entsteht in
`train-head.py` aus der blossen EXISTENZ von `ads_user.json`, und
auto-confirm legt genau so eine Datei an. Gemessen an 234 lebenden
Aufnahmen waren 95 (41 %) maschinell oder agentengeschrieben — alle mit
`which="merged"`. Wer `merged` als Menschen zählt, zählt 41 % falsch.

Ausführen: python3 scripts/test_massstab_herkunft.py
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ma", Path(__file__).resolve().parent / "massstab-audit.py")
_ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ma)


class Herkunft(unittest.TestCase):
    """`herkunft(uuid, meta)` liest das Label-Backup und faellt auf das
    Archiv zurueck. Das Backup wird je Test auf ein temporaeres
    Verzeichnis umgebogen."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._alt = _ma.BACKUP
        _ma.BACKUP = Path(self._tmp.name)

    def tearDown(self):
        _ma.BACKUP = self._alt
        self._tmp.cleanup()

    def _lebend(self, uuid, inhalt):
        d = _ma.BACKUP / f"_rec_{uuid}"
        d.mkdir(parents=True)
        (d / "ads_user.json").write_text(json.dumps(inhalt))

    # --- lebende Aufnahmen: die Marker entscheiden ----------------------

    def test_sauberes_nutzerlabel_ist_mensch(self):
        self._lebend("u1", {"ads": [[10, 20]], "reviewed_at": 1})
        self.assertEqual(_ma.herkunft("u1", {})[0], "mensch")

    def test_auto_confirmed_ist_maschine(self):
        self._lebend("u2", {"ads": [[10, 20]], "auto_confirmed_at": 1})
        self.assertEqual(_ma.herkunft("u2", {})[0], "maschine")

    def test_fingerprint_bestaetigung_ist_maschine(self):
        # Der Fall aus fingerprint_bestaetigung_ist_kein_mensch: KEIN
        # auto_confirmed_at, nur ein reviewed_at — sieht menschlich aus.
        self._lebend("u3", {"ads": [[10, 20]],
                            "auto_confirmed_via_fingerprint": True,
                            "reviewed_at": 1})
        self.assertEqual(_ma.herkunft("u3", {})[0], "maschine")

    def test_agenten_schreiber_sind_maschine(self):
        for schreiber in ("agent-review.py", "claude-code", "zurueckgenommen",
                          "folgen-vergleich.py"):
            with self.subTest(schreiber=schreiber):
                uuid = f"u-{schreiber}"
                self._lebend(uuid, {"ads": [[1, 2]], "reviewed_by": schreiber})
                self.assertEqual(_ma.herkunft(uuid, {})[0], "maschine")

    def test_lebendes_label_schlaegt_das_archiv(self):
        # Auch wenn das Archiv "merged" behauptet: die Datei ist naeher dran.
        self._lebend("u4", {"ads": [[1, 2]], "auto_confirmed_at": 1})
        art, _ = _ma.herkunft("u4", {"u4": {"which": "merged"}})
        self.assertEqual(art, "maschine")

    # --- tote Aufnahmen: nur noch das Archiv ----------------------------

    def test_totes_merged_ist_UNBEKANNT(self):
        """Der Kern dieses Tests. `merged` deckt Mensch UND auto-confirm."""
        art, grund = _ma.herkunft("tot1", {"tot1": {"which": "merged"}})
        self.assertEqual(art, "unbekannt")
        self.assertIn("merged", grund)

    def test_totes_user_ist_mensch(self):
        # which="user" entsteht nur, wenn ads.json FEHLT und ads_user.json
        # da ist — das legt auto-confirm nicht an.
        self.assertEqual(_ma.herkunft("tot2", {"tot2": {"which": "user"}})[0],
                         "mensch")

    def test_totes_auto_ist_maschine(self):
        for w in ("auto", "auto-confirm"):
            with self.subTest(which=w):
                self.assertEqual(_ma.herkunft("t", {"t": {"which": w}})[0],
                                 "maschine")

    def test_ohne_alles_ist_unbekannt(self):
        self.assertEqual(_ma.herkunft("niemand", {})[0], "unbekannt")

    def test_unlesbares_label_faellt_aufs_archiv_zurueck(self):
        d = _ma.BACKUP / "_rec_kaputt"
        d.mkdir(parents=True)
        (d / "ads_user.json").write_text("{kein json")
        self.assertEqual(_ma.herkunft("kaputt", {"kaputt": {"which": "auto"}})[0],
                         "maschine")


class Listenpflege(unittest.TestCase):

    def test_nicht_mensch_deckungsgleich_mit_golden_v3(self):
        """Die Liste steht an zwei Stellen. Laufen sie auseinander, zaehlt
        die eine Auswertung Agenten als Menschen und die andere nicht."""
        spec = importlib.util.spec_from_file_location(
            "gv3", Path(__file__).resolve().parent / "golden_v3_vorschlag.py")
        gv3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gv3)
        self.assertEqual(_ma.NICHT_MENSCH, gv3.NICHT_MENSCH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
