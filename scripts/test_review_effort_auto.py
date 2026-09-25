#!/usr/bin/env python3
"""Auto-Confirm-Filter von review-effort.py: live vom Pi, Spiegel als Rueckfall.

Befund 2026-09-05: der Spiegel (04:32) kannte die Fingerprint-Bestaetigungen
von 08:00 nicht, der Tagesdurchgang (08:07) zaehlte sie als Mensch mit 0 s/h.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_p = Path(__file__).with_name("review-effort.py")
_spec = importlib.util.spec_from_file_location("review_effort", _p)
re_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(re_)


class Vereinigung(unittest.TestCase):
    def test_live_und_spiegel_werden_vereinigt(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt({"neu-0800"}, {"alt-0432"})
        self.assertEqual(uuids, {"neu-0800", "alt-0432"})
        self.assertEqual(quelle, "pi+spiegel")

    def test_leeres_live_ist_ein_ergebnis(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt(set(), {"alt"})
        self.assertEqual(uuids, {"alt"})
        self.assertEqual(quelle, "pi+spiegel")

    def test_ohne_pi_nur_spiegel_und_gesagt(self):
        uuids, quelle = re_.auto_bestaetigte_vereinigt(None, {"alt"})
        self.assertEqual(uuids, {"alt"})
        self.assertEqual(quelle, "spiegel")

    def test_live_liefert_none_wenn_ssh_fehlt(self):
        self.assertIsNone(re_.auto_bestaetigte_live(host="kein.host.invalid", timeout=5))


class AgentIstKeinMensch(unittest.TestCase):
    """reviewed_by eines Werkzeugs zaehlt nicht als menschliches Review
    (bis 2026-09-25 las review-effort das Feld nicht)."""

    FAELLE = {
        "mensch": {"ads": [[1, 2]]},
        "golden": {"ads": [[1, 2]], "reviewed_by": "golden-audit"},
        "agent": {"ads": [[1, 2]], "reviewed_by": "agent-review.py"},
        "folgen": {"ads": [[1, 2]], "reviewed_by": "folgen-vergleich.py"},
        "auto": {"ads": [[1, 2]], "auto_confirmed_at": 1781808677},
        "fp": {"ads": [[1, 2]], "auto_confirmed_via_fingerprint": True},
    }
    NICHT_MENSCH = {"agent", "folgen", "auto", "fp"}

    def test_spiegel(self):
        with tempfile.TemporaryDirectory() as t:
            for u, d in self.FAELLE.items():
                (Path(t) / f"_rec_{u}").mkdir()
                (Path(t) / f"_rec_{u}" / "ads_user.json").write_text(json.dumps(d))
            self.assertEqual(re_.auto_bestaetigte(Path(t)), self.NICHT_MENSCH)

    def test_live_grep_ausgabe(self):
        # Form von `grep -oHE` auf dem Pi, kompakt und mit Leerzeichen
        aus = "\n".join([
            '/mnt/tv/hls/_rec_golden/ads_user.json:"reviewed_by":"golden-audit"',
            '/mnt/tv/hls/_rec_agent/ads_user.json:"reviewed_by": "agent-review.py"',
            '/mnt/tv/hls/_rec_folgen/ads_user.json:"reviewed_by":"folgen-vergleich.py"',
            '/mnt/tv/hls/_rec_auto/ads_user.json:"auto_confirmed_at":1781808677',
            '/mnt/tv/hls/_rec_fp/ads_user.json:"auto_confirmed_via_fingerprint":true',
            '/mnt/tv/hls/_rec_fpnein/ads_user.json:"auto_confirmed_via_fingerprint":false',
        ])
        self.assertEqual(re_.nicht_mensch_aus_grep(aus), self.NICHT_MENSCH)

    def test_regel_ist_die_gemeinsame(self):
        # keine eigene Kopie: die Liste kommt aus label_herkunft.py
        self.assertIn("agent-review.py", re_._lh.NICHT_MENSCH)
        self.assertNotIn("NICHT_MENSCH = {", _p.read_text())


if __name__ == "__main__":
    unittest.main()
