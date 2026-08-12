#!/usr/bin/env python3
"""Tests fuer das Registrierungs-Audit.

⚠️ Der Punkt ist NICHT, dass das Audit "ERFUELLT" sagen kann — das tut jede
kaputte Fassung, die immer ja sagt. Der Punkt ist, dass es NEIN sagt: bei
verfehlter Schwelle, bei verfehlten Vorzeichen, bei einer Nacht mit falschem
set_hash, und wenn die Regel nach Serienbeginn noch angefasst wurde. Ein
Waechter, der nie ausloest, ist von einem fehlenden Waechter nicht zu
unterscheiden — genau daran ist der Golden-Drift vier Naechte lang
vorbeigelaufen.

Ausfuehren: python3 scripts/test_audit_preregistration.py
"""
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parent / "audit-preregistration.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

HASH = "c8727e8266a8"
DEC = "--decoder hsmm --hsmm-dur-w 15"

REGEL = {
    "id": "OTEST", "frage": "Testfrage?",
    "serie_ab": "20260810", "naechte": 5,
    "arme": {"mit": "arm-mit", "ohne": "arm-ohne"},
    "gueltige_nacht": {"set_hash": HASH, "decoder": DEC, "golden_n": 38},
    "bedingungen": {"median_hoechstens": -0.010,
                    "negative_naechte_mindestens": 4},
}


def nacht(ts, mit, ohne, *, set_hash=HASH, decoder=DEC, n=38,
          quelle="nightly"):
    """Zwei jsonl-Zeilen fuer eine Nacht."""
    gem = {"ts": ts, "set_hash": set_hash, "decoder": decoder, "golden_n": n,
           "quelle": quelle}
    return [dict(gem, arch="arm-mit", golden_median=mit),
            dict(gem, arch="arm-ohne", golden_median=ohne)]


def lauf(naechte, regel=None, docs_pfad=None):
    """Audit gegen ein Archiv mit genau diesen Zeilen. Gibt (ok, text)."""
    regel = regel or REGEL
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        with open(p / "shadow-trend.jsonl", "w") as f:
            for z in naechte:
                f.write(json.dumps(z) + "\n")
        nach_ts = A.naechte_laden(p)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = A.pruefe(docs_pfad or Path("regel.md"), regel, nach_ts)
        return ok, buf.getvalue()


class SagtNein(unittest.TestCase):
    """Die Faelle, in denen das Audit ausloesen MUSS."""

    def test_median_verfehlt(self):
        # Alle 5 negativ, aber viel zu klein: Vorzeichen erfuellt, Schwelle nicht.
        z = []
        for i, d in enumerate([-0.002, -0.001, -0.003, -0.002, -0.001]):
            z += nacht(f"2026081{i}T040000", 0.900 + d, 0.900)
        ok, txt = lauf(z)
        self.assertFalse(ok)
        self.assertIn("NICHT ERFUELLT", txt)
        self.assertIn("Bedingung 1", txt)

    def test_vorzeichen_verfehlt(self):
        # Median haelt die Schwelle, aber nur 3 von 5 sind negativ —
        # ein Median, den die Vorzeichen nicht stuetzen, haengt an einer Nacht.
        z = []
        for i, d in enumerate([-0.090, -0.030, -0.012, +0.020, +0.030]):
            z += nacht(f"2026081{i}T040000", 0.900 + d, 0.900)
        ok, txt = lauf(z)
        self.assertFalse(ok)
        self.assertIn("NICHT ERFUELLT", txt)

    def test_positives_delta(self):
        z = []
        for i in range(5):
            z += nacht(f"2026081{i}T040000", 0.930, 0.900)
        ok, txt = lauf(z)
        self.assertFalse(ok)
        self.assertIn("NICHT ERFUELLT", txt)


class ZaehltRichtig(unittest.TestCase):
    """Was als gueltige Nacht durchgeht — und was nicht."""

    def test_falscher_set_hash_wird_verworfen(self):
        z = nacht("20260810T040000", 0.880, 0.900, set_hash="anders")
        for i in range(1, 5):
            z += nacht(f"2026081{i}T040000", 0.880, 0.900)
        ok, txt = lauf(z)
        self.assertIn("verworfen 20260810T040000", txt)
        self.assertIn("set_hash", txt)
        # 4 gueltige von 5 verlangten → noch offen, NICHT vorzeitig entschieden
        self.assertIn("NOCH OFFEN", txt)
        self.assertTrue(ok)

    def test_falscher_decoder_wird_verworfen(self):
        z = nacht("20260810T040000", 0.880, 0.900, decoder="--decoder form")
        ok, txt = lauf(z)
        self.assertIn("decoder", txt)
        # Alle Naechte verworfen ist NICHT dasselbe wie "noch nicht
        # begonnen" — sonst laeuft eine kaputte Serie beliebig lang als
        # "noch offen" mit.
        self.assertIn("ALLE 1 Nächte verworfen", txt)
        self.assertIn("Defekt", txt)

    def test_unvollstaendiger_golden_satz_wird_verworfen(self):
        z = nacht("20260810T040000", 0.880, 0.900, n=37)
        ok, txt = lauf(z)
        self.assertIn("golden_n", txt)

    def test_fehlender_arm_wird_verworfen(self):
        z = [dict(ts="20260810T040000", arch="arm-mit", golden_median=0.88,
                  set_hash=HASH, decoder=DEC, golden_n=38)]
        ok, txt = lauf(z)
        self.assertIn("ein Arm fehlt", txt)

    def test_naechte_vor_serienbeginn_zaehlen_nicht(self):
        # Der Handlauf vom 08-09 ist die Vorstudie, nicht die Serie.
        z = nacht("20260809T080000", 0.870, 0.900)
        ok, txt = lauf(z)
        self.assertIn("Serie hat noch nicht begonnen", txt)
        self.assertNotIn("20260809", txt.split("Stand:")[0].replace(
            "Registrierung", ""))

    def test_handlauf_zaehlt_nicht(self):
        # Ein Handlauf mitten in der Serie wuerde sie sonst still um eine
        # Nacht weiterzaehlen — unter anderen Bedingungen gemessen.
        z = nacht("20260811T093000", 0.850, 0.900, quelle="hand")
        for i in (0, 2, 3, 4):
            z += nacht(f"2026081{i}T040000", 0.890, 0.900)
        ok, txt = lauf(z)
        self.assertIn("kein Nightly", txt)
        self.assertIn("quelle=hand", txt)
        self.assertIn("NOCH OFFEN", txt)  # 4 statt 5 gueltige

    def test_fehlende_quelle_zaehlt_nicht(self):
        # Alte Zeilen ohne das Feld sind nicht nachweisbar Nightly-Zeilen.
        z = nacht("20260810T040000", 0.850, 0.900)
        for zeile in z:
            del zeile["quelle"]
        ok, txt = lauf(z)
        self.assertIn("kein Nightly", txt)

    def test_zweiter_lauf_am_selben_tag_zaehlt_nicht(self):
        z = nacht("20260810T040000", 0.880, 0.900)
        z += nacht("20260810T193000", 0.700, 0.900)   # Wiederholung
        for i in (1, 2, 3):
            z += nacht(f"2026081{i}T040000", 0.890, 0.900)
        ok, txt = lauf(z)
        self.assertIn("zweiter Lauf am 20260810", txt)
        self.assertIn("NOCH OFFEN", txt)
        # Der Ausreisser -0.200 darf den Median nicht erreichen.
        self.assertNotIn("-0.2000", txt)

    def test_zwischenstand_ist_kein_ergebnis(self):
        # 3 von 5 Naechten, alle stark negativ → trotzdem kein Urteil.
        z = []
        for i in range(3):
            z += nacht(f"2026081{i}T040000", 0.850, 0.900)
        ok, txt = lauf(z)
        self.assertIn("NOCH OFFEN", txt)
        self.assertIn("KEIN Ergebnis", txt)
        self.assertNotIn("ERFUELLT.", txt)


class SagtJa(unittest.TestCase):
    def test_klar_erfuellt(self):
        z = []
        for i, d in enumerate([-0.030, -0.025, -0.018, -0.011, +0.004]):
            z += nacht(f"2026081{i}T040000", 0.900 + d, 0.900)
        ok, txt = lauf(z)
        self.assertTrue(ok)
        self.assertIn("REGEL ERFUELLT", txt)


class Integritaet(unittest.TestCase):
    """Die Regel darf nach Serienbeginn nicht mehr angefasst werden."""

    def _repo(self, d):
        p = Path(d)
        subprocess.run(["git", "init", "-q"], cwd=p, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=p, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=p, check=True)
        return p

    def test_uncommittete_regel_schlaegt_alarm(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._repo(d)
            doc = p / "x-preregistration.md"
            doc.write_text("egal")
            # Der Anker braucht seit dem O2-Fehlalarm-Fix eine Zeile, die
            # zur REGEL gehoert (Quelle+Arm) — eine leere Nacht reicht nicht.
            nach_ts = {"20260810T040000": {
                "arm-mit": {"quelle": "nightly", "arch": "arm-mit"}}}
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = A.pruefe_integritaet(doc, nach_ts, REGEL)
            self.assertFalse(ok)
            self.assertIn("INTEGRITAET", buf.getvalue())

    def test_aenderung_nach_serienbeginn_schlaegt_alarm(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._repo(d)
            doc = p / "x-preregistration.md"
            doc.write_text("Regel v1")
            subprocess.run(["git", "add", "-A"], cwd=p, check=True)
            subprocess.run(["git", "commit", "-qm", "v1"], cwd=p, check=True)
            # Serie beginnt weit VOR dem Commit von eben (1970).
            nach_ts = {"19700102T000000": {
                "arm-mit": {"quelle": "nightly", "arch": "arm-mit"}}}
            regel = dict(REGEL, serie_ab="19700101")
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = A.pruefe_integritaet(doc, nach_ts, regel)
            self.assertFalse(ok)
            self.assertIn("NACH der ersten", buf.getvalue())

    def test_regel_vor_serienbeginn_ist_in_ordnung(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._repo(d)
            doc = p / "x-preregistration.md"
            doc.write_text("Regel v1")
            subprocess.run(["git", "add", "-A"], cwd=p, check=True)
            subprocess.run(["git", "commit", "-qm", "v1"], cwd=p, check=True)
            # Serie beginnt in ferner Zukunft, der Commit liegt davor.
            nach_ts = {"20991231T000000": {}}
            regel = dict(REGEL, serie_ab="20990101")
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = A.pruefe_integritaet(doc, nach_ts, regel)
            self.assertTrue(ok)


class EchteRegistrierung(unittest.TestCase):
    def test_o1_block_ist_lesbar_und_vollstaendig(self):
        regeln = A.regeln_laden(Path(__file__).resolve().parent.parent / "docs")
        ids = {r.get("id") for _, r in regeln}
        self.assertIn("O1", ids)
        o1 = next(r for _, r in regeln if r["id"] == "O1")
        for feld in ("serie_ab", "naechte", "arme", "gueltige_nacht",
                     "bedingungen"):
            self.assertIn(feld, o1)
        self.assertEqual(o1["arme"]["mit"], "mlp32-cwt-mp")
        self.assertEqual(o1["arme"]["ohne"], "mlp32-ct-mp")




TAGES_REGEL = dict(REGEL, serie_art="tagesserie")


def paar(ts, mit, ohne, *, seed, seed_ohne=None, quelle="tagesserie", n=38):
    """Zwei jsonl-Zeilen fuer ein Tagesserien-Paar."""
    gem = {"set_hash": HASH, "decoder": DEC, "golden_n": n, "quelle": quelle}
    return [dict(gem, ts=ts, arch="arm-mit", golden_median=mit, seed=seed),
            dict(gem, ts=ts, arch="arm-ohne", golden_median=ohne,
                 seed=seed if seed_ohne is None else seed_ohne)]


class Tagesserie(unittest.TestCase):
    """serie_art=tagesserie: Stichproben aus der Seed-Ziehung statt aus
    Naechten. Der Grund fuer den Serientyp: die baseline-Zeile der
    Nacht-Serie fittet fest mit Seed 0, eine Paarung gegen sie vermischt
    Spaltenwirkung und Seed-Differenz."""

    def test_fuenf_paare_entscheiden(self):
        z = []
        for i in range(5):
            z += paar(f"20260812T100000p{i:02d}", 0.880, 0.900, seed=100 + i)
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertTrue(ok)  # erfuellt = kein Integritaetsproblem
        self.assertIn("→ REGEL ERFUELLT", txt)
        self.assertIn("Paar", txt)

    def test_arme_muessen_seed_gepaart_sein(self):
        # DER Grund fuer den Serientyp — ungleiche Seeds sind kein Paar.
        z = paar("20260812T100000p00", 0.880, 0.900, seed=1, seed_ohne=2)
        for i in range(1, 5):
            z += paar(f"20260812T100000p{i:02d}", 0.880, 0.900, seed=100 + i)
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertIn("nicht seed-gepaart", txt)
        self.assertIn("NOCH OFFEN", txt)  # 4 statt 5

    def test_gleicher_seed_zaehlt_nur_einmal(self):
        # Derselbe Seed nochmal = Wiederholung, keine zweite Stichprobe —
        # dieselbe Logik wie der Kalendertag der Nacht-Serie.
        z = []
        for i in range(5):
            z += paar(f"20260812T100000p{i:02d}", 0.880, 0.900, seed=7)
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertIn("bereits gezählt", txt)
        self.assertIn("NOCH OFFEN: 1/5", txt)

    def test_mehrere_paare_am_selben_tag_zaehlen(self):
        # Der ganze Punkt: KEINE Kalendertag-Deduplikation.
        z = []
        for i in range(5):
            z += paar(f"20260812T100000p{i:02d}", 0.895, 0.900, seed=100 + i)
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertNotIn("zweiter Lauf am", txt)
        self.assertIn("5/5", txt.replace("NOCH OFFEN: ", ""))

    def test_nachtzeilen_stoeren_die_tagesserie_nicht(self):
        # Die parallel laufende Nacht-Serie ist planmaessig da — sie darf
        # weder zaehlen noch als verworfen erscheinen.
        z = nacht("20260813T033000", 0.880, 0.900)  # quelle=nightly
        for i in range(3):
            z += paar(f"20260812T100000p{i:02d}", 0.880, 0.900, seed=100 + i)
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertIn("NOCH OFFEN: 3/5", txt)
        self.assertNotIn("verworfen 20260813", txt)

    def test_tagesserienzeilen_stoeren_die_nachtserie_nicht(self):
        # Und umgekehrt.
        z = paar("20260812T100000p00", 0.880, 0.900, seed=100)
        for i in range(3):
            z += nacht(f"2026081{i}T040000", 0.880, 0.900)
        ok, txt = lauf(z)  # Nacht-Regel
        self.assertIn("NOCH OFFEN: 3/5", txt)
        self.assertNotIn("verworfen 20260812T100000p00", txt)

    def test_handlauf_bleibt_auch_hier_laut(self):
        # Die Stille gilt NUR der jeweils anderen Serienart.
        z = paar("20260812T100000p00", 0.880, 0.900, seed=100, quelle="hand")
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertIn("keine Tagesserie", txt)
        self.assertIn("quelle=hand", txt)




class FremdeSerien(unittest.TestCase):
    def test_fremde_tagesserie_stoert_nicht(self):
        # Eine Tagesserie zu einer ANDEREN Frage (fremde arch-Namen,
        # gleiche quelle) darf hier weder zaehlen noch als "ein Arm
        # fehlt" erscheinen — sonst waechst dieses Audit mit jeder
        # weiteren Frage um Laermzeilen.
        fremd = [{"ts": "20260813T100000p00", "arch": "ganz-anderer-arm",
                  "golden_median": 0.9, "seed": 5, "quelle": "tagesserie",
                  "set_hash": HASH, "decoder": DEC, "golden_n": 38}]
        z = fremd + [x for i in range(3)
                     for x in paar(f"20260812T100000p{i:02d}", 0.88, 0.90,
                                   seed=100 + i)]
        ok, txt = lauf(z, regel=TAGES_REGEL)
        self.assertIn("NOCH OFFEN: 3/5", txt)
        self.assertNotIn("verworfen 20260813", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
