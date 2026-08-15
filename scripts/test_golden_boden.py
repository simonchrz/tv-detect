#!/usr/bin/env python3
"""Tests fuer den Golden-Boden (die Sperrklinke gegen langsamen Drift).

⚠️ Der Punkt dieser Tests ist NICHT, dass der Boden durchlaesst — das tut
jede kaputte Fassung auch. Der Punkt ist, dass er FEUERT. Ein Gate, das nie
ausloest, ist von einem fehlenden Gate nicht zu unterscheiden, und genau so
ist der Drift vier Naechte lang unbemerkt geblieben: der Golden-Wert wurde
berechnet, protokolliert und nirgends ausgewertet.

Ausfuehren: python3 scripts/test_golden_boden.py
"""

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "th", Path(__file__).resolve().parent / "train-head.py")
_th = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_th)
except SystemExit:
    pass

HASH = "abc123"


def archiv(tmp, trend):
    """Legt golden-eval-set.json + golden-trend.jsonl an."""
    p = Path(tmp)
    uuids = [f"rec{i}" for i in range(5)]
    (p / "golden-eval-set.json").write_text(
        json.dumps({"uuids": uuids, "set_hash": HASH, "version": 2}))
    with open(p / "golden-trend.jsonl", "w") as f:
        for e in trend:
            # Seit 2026-08-06 filtert golden_bestwert zusaetzlich nach
            # `decoder` — ein Eintrag ohne das Feld gilt als "form" und
            # zaehlt nicht mehr mit, wenn EVAL_DECODER auf hsmm steht.
            # Die Testfaelle beschreiben den NORMALFALL, also den aktuell
            # gemessenen Decoder; wer den fremden Decoder testen will,
            # setzt das Feld ausdruecklich (s. test_fremder_decoder_*).
            e = dict(e)
            e.setdefault("decoder", " ".join(_th.EVAL_DECODER) or "form")
            f.write(json.dumps(e) + "\n")
    return p, uuids


def prio(uuids, wert):
    return {u: wert for u in uuids}


class GoldenBoden(unittest.TestCase):

    def lauf(self, trend, cand, champ=None, floor=0.010):
        with tempfile.TemporaryDirectory() as tmp:
            p, uuids = archiv(tmp, trend)
            zeilen = []
            deploy, grund = _th.golden_boden(
                True, "paarweise: unauffaellig",
                golden_floor=floor, train_archive=str(p),
                cand_pr=prio(uuids, cand),
                champ_pr=prio(uuids, champ) if champ is not None else {},
                melde=zeilen.append)
            return deploy, grund, "\n".join(zeilen)

    def test_drift_wird_geblockt(self):
        """Der reale Fall: 0.915 war der Bestwert, der Kandidat liegt 0.019
        darunter und verbessert den Champion nicht. Genau dieser Kandidat ist
        am 2026-08-03 durchgerutscht."""
        deploy, grund, _ = self.lauf(
            [{"ts": "n1", "golden_median": 0.915, "deployed": True, "set_hash": HASH},
             {"ts": "n2", "golden_median": 0.901, "deployed": True, "set_hash": HASH}],
            cand=0.896, champ=0.901)
        self.assertFalse(deploy, "Golden-Boden hat den Drift NICHT geblockt")
        self.assertIn("GOLDEN-BODEN", grund)
        self.assertIn("0.915", grund)

    def test_aufstieg_kommt_durch(self):
        """Kein Deadlock: unter dem Bestwert, aber besser als der Champion —
        genau so klettert der Stand nach einem Absacker wieder hoch. Wuerde
        das blocken, bliebe der Champion fuer immer stehen."""
        deploy, _, log = self.lauf(
            [{"ts": "n1", "golden_median": 0.915, "deployed": True, "set_hash": HASH}],
            cand=0.890, champ=0.880)
        self.assertTrue(deploy)
        self.assertIn("Aufstieg", log)

    def test_kleine_schwankung_passiert(self):
        """Innerhalb der Schwelle wird nicht geblockt — sonst waere jede
        Nacht ein Deadlock."""
        deploy, _, log = self.lauf(
            [{"ts": "n1", "golden_median": 0.915, "deployed": True, "set_hash": HASH}],
            cand=0.909, champ=0.910)
        self.assertTrue(deploy)
        self.assertIn("passiert", log)

    def test_fremder_satz_zaehlt_nicht(self):
        """Ein Bestwert aus einem ANDEREN Golden-Satz darf nicht gaten —
        sonst vergleicht man zwei verschiedene Messungen."""
        deploy, _, log = self.lauf(
            [{"ts": "alt", "golden_median": 0.980, "deployed": True,
              "set_hash": "anderer"}],
            cand=0.896, champ=0.901)
        self.assertTrue(deploy)
        self.assertIn("noch kein Bestwert", log)

    def test_fremder_decoder_zaehlt_nicht(self):
        """Ein Bestwert, der mit einem ANDEREN Blockbildner gemessen wurde,
        darf nicht gaten.

        Gleiche Begruendung wie beim set_hash: es waeren zwei verschiedene
        Messungen. Und die Richtung ist hier gefaehrlich — hsmm liegt auf
        diesem Satz systematisch HOEHER als form (gemessen +0.14 Mittel),
        ein geerbter form-Bestwert waere als Sperrklinke also wirkungslos
        statt zu streng. Sie haette genau das nicht mehr getan, wofuer es
        sie gibt."""
        deploy, _, log = self.lauf(
            [{"ts": "alt", "golden_median": 0.980, "deployed": True,
              "set_hash": HASH, "decoder": "form"}],
            cand=0.896, champ=0.901)
        self.assertTrue(deploy)
        self.assertIn("noch kein Bestwert", log)

    def test_eigener_decoder_gatet_weiterhin(self):
        """Gegenprobe: mit passendem Decoder muss derselbe Eintrag blocken.

        Ohne diesen Test koennte der Filter alles wegwerfen und die Tests
        blieben gruen — ein Gate, das nie ausloest, ist von einem fehlenden
        nicht zu unterscheiden."""
        deploy, grund, _ = self.lauf(
            [{"ts": "alt", "golden_median": 0.980, "deployed": True,
              "set_hash": HASH}],
            cand=0.896, champ=0.901)
        self.assertFalse(deploy)
        self.assertIn("GOLDEN-BODEN", grund.upper())

    def test_abgelehnte_kandidaten_setzen_keinen_bestwert(self):
        """Ein REJECTED-Kandidat war nie in Produktion; sein Wert darf den
        Bestwert nicht hochziehen und damit alles Folgende blocken."""
        deploy, _, _ = self.lauf(
            [{"ts": "n1", "golden_median": 0.900, "deployed": True, "set_hash": HASH},
             {"ts": "n2", "golden_median": 0.960, "deployed": False, "set_hash": HASH}],
            cand=0.898, champ=0.899)
        self.assertTrue(deploy, "abgelehnter Kandidat wurde als Bestwert benutzt")

    def test_unvollstaendiger_satz_gatet_nicht(self):
        """Fehlt eine gepinnte Aufnahme, ist der Median nicht
        komposition-konstant — dann darf er nicht gaten."""
        with tempfile.TemporaryDirectory() as tmp:
            p, uuids = archiv(tmp, [{"ts": "n1", "golden_median": 0.915,
                                     "deployed": True, "set_hash": HASH}])
            zeilen = []
            cand = prio(uuids, 0.800)
            del cand[uuids[0]]          # eine fehlt
            deploy, _, = _th.golden_boden(
                True, "x", golden_floor=0.010, train_archive=str(p),
                cand_pr=cand, champ_pr=prio(uuids, 0.900), melde=zeilen.append)
            self.assertTrue(deploy)
            self.assertIn("komposition-konstant", "\n".join(zeilen))

    def test_eintraege_mit_missing_zaehlen_nicht(self):
        """Ein frueherer Eintrag, der selbst nicht komposition-konstant war,
        darf den Bestwert nicht setzen."""
        best, _ = _th.golden_bestwert.__wrapped__ if hasattr(
            _th.golden_bestwert, "__wrapped__") else (None, None)
        with tempfile.TemporaryDirectory() as tmp:
            p, _ = archiv(tmp, [
                {"ts": "n1", "golden_median": 0.999, "deployed": True,
                 "set_hash": HASH, "missing": ["rec9"]},
                {"ts": "n2", "golden_median": 0.900, "deployed": True,
                 "set_hash": HASH, "missing": []}])
            best, ts = _th.golden_bestwert(p / "golden-trend.jsonl", HASH)
        self.assertEqual(best, 0.900)
        self.assertEqual(ts, "n2")

    def test_abschaltbar(self):
        deploy, _, log = self.lauf(
            [{"ts": "n1", "golden_median": 0.915, "deployed": True, "set_hash": HASH}],
            cand=0.500, champ=0.900, floor=0)
        self.assertTrue(deploy)
        self.assertEqual(log, "")



class GoldenStau(unittest.TestCase):
    """Ein Boden, der jede Nacht blockt, friert den Champion ein. Das ist
    richtiger als still abzurutschen — darf aber nicht selbst still
    passieren."""

    def test_stau_wird_gemeldet(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, uuids = archiv(tmp, [
                {"ts": "n0", "golden_median": 0.915, "deployed": True, "set_hash": HASH},
                {"ts": "n1", "golden_median": 0.890, "deployed": False, "set_hash": HASH},
                {"ts": "n2", "golden_median": 0.891, "deployed": False, "set_hash": HASH},
                {"ts": "n3", "golden_median": 0.892, "deployed": False, "set_hash": HASH}])
            zeilen = []
            deploy, _ = _th.golden_boden(
                True, "x", golden_floor=0.010, train_archive=str(p),
                cand_pr=prio(uuids, 0.893), champ_pr=prio(uuids, 0.900),
                melde=zeilen.append)
            self.assertFalse(deploy)
            self.assertIn("4. Nacht in Folge", "\n".join(zeilen))

    def test_kein_stau_kein_laerm(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, uuids = archiv(tmp, [
                {"ts": "n0", "golden_median": 0.915, "deployed": True, "set_hash": HASH}])
            zeilen = []
            _th.golden_boden(True, "x", golden_floor=0.010, train_archive=str(p),
                             cand_pr=prio(uuids, 0.880), champ_pr=prio(uuids, 0.900),
                             melde=zeilen.append)
            self.assertNotIn("Nacht in Folge", "\n".join(zeilen))

    def test_zaehlt_nur_den_eigenen_satz(self):
        best = _th.golden_stau(None, HASH)
        self.assertEqual(best, 0)


class BestwertIstZweitbester(unittest.TestCase):
    """Der Boden ist der hoechste ZWEIMAL erreichte Wert, nicht das Maximum.

    Warum das Tests braucht: max() sah jahrelang richtig aus. Erst der
    Seed-Sweep vom 2026-08-09 hat gezeigt, dass ein Einzelwert bis zu 0.023
    Gluecksanteil traegt — und dass die Sperrklinke deshalb verlangte, einen
    Gluecksstreffer zu wiederholen.
    """

    def _trend(self, p, eintraege):
        """eintraege: (ts, wert, deployed) — schreibt golden-trend.jsonl."""
        dec = " ".join(_th.EVAL_DECODER) or "form"
        with open(Path(p) / "golden-trend.jsonl", "w") as f:
            for ts, wert, dep in eintraege:
                f.write(json.dumps({
                    "ts": ts, "golden_median": wert, "set_hash": HASH,
                    "decoder": dec, "deployed": dep, "n": 38}) + "\n")
        return Path(p) / "golden-trend.jsonl"

    def _trend_lab(self, p, eintraege):
        """Wie _trend, aber mit label_hash je Zeile."""
        with open(Path(p) / "trend-lab.jsonl", "w") as f:
            for ts, wert, lab in eintraege:
                f.write(json.dumps({
                    "ts": ts, "golden_median": wert, "set_hash": HASH,
                    "decoder": " ".join(_th.EVAL_DECODER) or "form",
                    "deployed": True, "n": 38,
                    "label_hash": lab}) + "\n")
        return Path(p) / "trend-lab.jsonl"

    def test_eigene_zeile_ist_keine_latte(self):
        # ⚠️ Der Trend wird VOR dem Gate geschrieben. Ohne ohne_ts vergleicht
        # sich der Lauf mit sich selbst — bei einer frischen Epoche ist die
        # eigene Zeile die einzige, und die Latte waere "heutiger Wert minus
        # Toleranz". Genau das ist am 2026-08-15 passiert (Bericht meldete
        # "Latte 0.9255, +0.0100" gegen den eigenen Wert 0.9355).
        with tempfile.TemporaryDirectory() as d:
            t = self._trend_lab(d, [("20260815T033009", 0.9355, "neu")])
            self.assertEqual(
                _th.golden_bestwert(t, HASH, "neu", ohne_ts="20260815T033009"),
                (None, None))
            # ... aber eine ECHTE Vornacht traegt weiterhin
            t2 = self._trend_lab(d, [("20260814T033009", 0.9200, "neu"),
                                     ("20260815T033009", 0.9355, "neu")])
            best, _ = _th.golden_bestwert(t2, HASH, "neu", ohne_ts="20260815T033009")
            self.assertAlmostEqual(best, 0.9200, places=3)

    def test_label_epoche_schneidet_den_boden(self):
        # Der eigentliche Zweck: eine Label-Korrektur am Golden-Satz macht
        # frueherer Mediane unvergleichbar. Ohne diesen Schnitt haette die
        # Latte am 2026-08-13 (0.906 -> 0.937 durch 87 Kanten-Korrekturen)
        # eine Modellverbesserung behauptet, die nie stattgefunden hat.
        with tempfile.TemporaryDirectory() as d:
            # Je Epoche drei Tage, damit wirklich die Zweitbester-Regel
            # greift und nicht die max()-Rueckfallregel gemessen wird.
            t = self._trend_lab(d, [("20260806T040000", 0.950, "alt"),
                                    ("20260807T040000", 0.949, "alt"),
                                    ("20260808T040000", 0.940, "alt"),
                                    ("20260809T040000", 0.910, "neu"),
                                    ("20260810T040000", 0.905, "neu"),
                                    ("20260811T040000", 0.900, "neu")])
            best, ts = _th.golden_bestwert(t, HASH, "neu")
            self.assertAlmostEqual(best, 0.905, places=3)
            self.assertTrue(ts.startswith("202608"))
            # ... und die alte Epoche bleibt fuer sich messbar
            best_alt, _ = _th.golden_bestwert(t, HASH, "alt")
            self.assertAlmostEqual(best_alt, 0.949, places=3)

    def test_stern_ignoriert_die_label_epoche(self):
        # Altpfade und Tests rufen ohne Hash — dann darf nicht gefiltert
        # werden, sonst verschwindet der Boden still und das Gate faellt auf.
        with tempfile.TemporaryDirectory() as d:
            t = self._trend_lab(d, [("20260806T040000", 0.950, "alt"),
                                    ("20260807T040000", 0.949, "alt"),
                                    ("20260808T040000", 0.910, "neu")])
            best, _ = _th.golden_bestwert(t, HASH)
            self.assertAlmostEqual(best, 0.949, places=3)

    def test_nimmt_den_zweitbesten(self):
        with tempfile.TemporaryDirectory() as d:
            t = self._trend(d, [("20260806T040000", 0.909, True),
                                ("20260807T040000", 0.921, True),
                                ("20260808T040000", 0.917, True)])
            best, ts = _th.golden_bestwert(t, HASH)
            self.assertAlmostEqual(best, 0.917)
            self.assertTrue(ts.startswith("20260808"))

    def test_mehrere_laeufe_am_selben_tag_zaehlen_einmal(self):
        # Drei Laeufe am 08-06 duerfen nicht drei Ziehungen sein — sonst
        # blaeht jede Wiederholung die Zahl der Ziehungen und damit die
        # Verzerrung des Maximums auf. Gezaehlt wird der LETZTE des Tages.
        with tempfile.TemporaryDirectory() as d:
            t = self._trend(d, [("20260806T030000", 0.880, True),
                                ("20260806T120000", 0.950, True),
                                ("20260806T190000", 0.909, True),
                                ("20260807T040000", 0.921, True),
                                ("20260808T040000", 0.917, True)])
            best, ts = _th.golden_bestwert(t, HASH)
            # Tage: 08-06 -> 0.909 (letzter), 08-07 -> 0.921, 08-08 -> 0.917
            self.assertAlmostEqual(best, 0.917)

    def test_unter_drei_tagen_faellt_auf_max_zurueck(self):
        # Ein zu niedriger Boden waere schlimmer als ein leicht zu hoher.
        with tempfile.TemporaryDirectory() as d:
            t = self._trend(d, [("20260807T040000", 0.921, True),
                                ("20260808T040000", 0.900, True)])
            best, _ = _th.golden_bestwert(t, HASH)
            self.assertAlmostEqual(best, 0.921)

    def test_nicht_deployte_zaehlen_nicht(self):
        with tempfile.TemporaryDirectory() as d:
            t = self._trend(d, [("20260806T040000", 0.909, True),
                                ("20260807T040000", 0.990, False),
                                ("20260808T040000", 0.917, True),
                                ("20260809T040000", 0.921, True)])
            best, _ = _th.golden_bestwert(t, HASH)
            self.assertAlmostEqual(best, 0.917)

    def test_boden_ist_niedriger_als_das_maximum(self):
        # Die eigentliche Aussage: die Aenderung entschaerft, sie verschaerft
        # nicht. Faellt dieser Test, ist die Rechnung verdreht.
        with tempfile.TemporaryDirectory() as d:
            werte = [0.909, 0.907, 0.917, 0.921, 0.9166]
            t = self._trend(d, [(f"2026080{i+4}T040000", v, True)
                                for i, v in enumerate(werte)])
            best, _ = _th.golden_bestwert(t, HASH)
            self.assertLess(best, max(werte))
            self.assertGreater(best, statistics.median(werte))

if __name__ == "__main__":
    unittest.main(verbosity=2)
