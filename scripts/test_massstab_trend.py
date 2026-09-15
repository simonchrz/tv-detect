#!/usr/bin/env python3
"""Der Maßstab muss sich selbst berichten — und sein Urteil selbst rechnen.

Die Lehre vom 2026-09-14: `massstab-audit.py` lag seit dem 06.09. fertig im
Repo und wurde NIE aufgerufen. Die Zahl, auf die es ankommt — woraus der
Satz besteht, an dem jede registrierte Frage entschieden wird — rechnete
niemand. Ein Bericht, den niemand startet, ist kein Bericht.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKRIPT = REPO / "scripts/massstab-audit.py"
NIGHTLY = (REPO / "daemon/tv-train-head.sh").read_text()


def lauf(abstaende):
    f = Path(tempfile.mkdtemp()) / "t.jsonl"
    f.write_text("\n".join(
        json.dumps({"ts": f"n{i:02d}", "abstand": a})
        for i, a in enumerate(abstaende)) + "\n")
    return subprocess.run([sys.executable, str(SKRIPT), "--auswerten",
                           "--trend", str(f)], capture_output=True, text=True)


class DasNightlyRuftEsAuf(unittest.TestCase):
    def test_bericht_und_urteil_haengen_im_nightly(self):
        self.assertIn("massstab-audit.py", NIGHTLY,
                      "der Bericht laeuft nicht — genau der Zustand, der "
                      "acht Tage lang niemandem auffiel")
        self.assertIn("--trend", NIGHTLY)
        self.assertIn("--auswerten", NIGHTLY)

    def test_nicht_fatal(self):
        # ⚠️ Das Skript endet mit exit 1, sobald maschinelle Labels im
        # Massstab stehen (am 2026-09-14: 27). Das ist ein Bericht, keine
        # Stoerung — die Nacht darf daran nicht scheitern.
        i = NIGHTLY.index("massstab-audit.py")
        self.assertIn("|| true", NIGHTLY[i:i + 400])


class DerRichtigeKopf(unittest.TestCase):
    """⚠️ Bis 2026-09-15 las die Spur fest `champion` — den Kopf VOR dem
    Lauf. Jede Zeile trug damit den Kopf der Vornacht."""

    def setUp(self):
        import importlib.util
        s = importlib.util.spec_from_file_location("ma", SKRIPT)
        self.ma = importlib.util.module_from_spec(s); s.loader.exec_module(self.ma)

    def test_deployt_heisst_candidate(self):
        self.assertEqual(self.ma.produktionskopf({"deploy": True}), "candidate")

    def test_abgelehnt_heisst_champion(self):
        self.assertEqual(self.ma.produktionskopf({"deploy": False}), "champion")

    def test_fehlendes_flag_ist_ablehnung(self):
        # Kein Flag = nicht nachweislich deployt = der alte Kopf laeuft.
        self.assertEqual(self.ma.produktionskopf({}), "champion")

    def test_nicht_mehr_fest_verdrahtet(self):
        self.assertNotIn('pr = lauf.get("champion")', SKRIPT.read_text())


class DieRegelGreiftAnDerSchwelle(unittest.TestCase):
    def test_kein_zwischenstand_vor_zehn_naechten(self):
        r = lauf([-0.03] * 9)
        self.assertEqual(r.returncode, 0)
        self.assertIn("9/10", r.stdout)
        self.assertNotIn("REGEL", r.stdout,
                         "vor der zehnten Nacht darf kein Urteil fallen")

    def test_konstanter_versatz_ist_nicht_erfuellt(self):
        r = lauf([-0.030] * 5 + [-0.031] * 5)
        self.assertIn("NICHT ERFUELLT", r.stdout)
        self.assertEqual(r.returncode, 0)

    def test_drift_ueber_der_schwelle_ist_erfuellt(self):
        r = lauf([-0.030] * 5 + [-0.045] * 5)
        self.assertIn("REGEL ERFUELLT", r.stdout)
        self.assertEqual(r.returncode, 1)

    def test_drift_unter_der_schwelle_nicht(self):
        r = lauf([-0.030] * 5 + [-0.038] * 5)
        self.assertIn("NICHT ERFUELLT", r.stdout)

    def test_urteil_haengt_nicht_an_spaeteren_naechten(self):
        # ⚠️ Beurteilt werden die ERSTEN 10. Ein Urteil, das sich mit jeder
        # weiteren Nacht verschiebt, ist keins.
        basis = [-0.030] * 5 + [-0.031] * 5
        a = lauf(basis).stdout
        b = lauf(basis + [-0.090] * 5).stdout
        self.assertIn("NICHT ERFUELLT", a)
        self.assertIn("NICHT ERFUELLT", b)


if __name__ == "__main__":
    unittest.main()
