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

if __name__ == "__main__":
    unittest.main(verbosity=2)
