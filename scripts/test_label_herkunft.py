#!/usr/bin/env python3
"""Tests für die Herkunfts-Regel (`label_herkunft.py`) und ihren Transport.

Zwei Dinge werden geprüft, und das zweite ist das wichtigere.

1. **Die Regel selbst** hat DREI Werte. `None` heisst *nicht
   entscheidbar*, nicht *kein Mensch*. Wer die beiden zusammenwirft,
   degradiert 294 tote train-Aufnahmen auf eine Vermutung.

2. **Der Transport durch `train-head.py`.** `mensch_belegt` entsteht im
   ersten Durchgang über den Korpus (Zeile ~3001) und wird im zweiten
   gebraucht (~3580). Das sind zwei getrennte Schleifen. Wer die Variable
   im zweiten Durchgang einfach liest, bekommt den Wert der LETZTEN
   Aufnahme des ersten — und der Fehler ist still: jede Aufnahme
   bekommt dieselbe fremde Antwort.

   Das ist keine Theorie. Der Kommentar an `confirmed_show` in derselben
   Schleife hält fest, dass genau das dort passiert ist: *„this loop read
   the pass-1 loop variables, which by now hold the LAST recording's
   values — 233 of 591 archive entries share the identical [22.0,
   1281.0]"*. Beim Einbau von O17 wäre es fast ein zweites Mal passiert.

   Der Test prüft deshalb strukturell (über den AST), dass
   `mensch_belegt` im zweiten Durchgang aus `rest[...]` zugewiesen wird,
   bevor es gelesen wird.

Ausführen: python3 scripts/test_label_herkunft.py
"""
import ast
import importlib.util
import unittest
from pathlib import Path

_HIER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lh", _HIER / "label_herkunft.py")
_lh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lh)


class Regel(unittest.TestCase):

    def test_sauberes_nutzerlabel(self):
        self.assertIs(_lh.mensch_aus_markern({"ads": [[1, 2]], "reviewed_at": 5}), True)

    def test_auto_confirmed(self):
        self.assertIs(_lh.mensch_aus_markern({"auto_confirmed_at": 5}), False)

    def test_fingerprint_sieht_menschlich_aus_ist_es_aber_nicht(self):
        # Kein auto_confirmed_at, nur ein reviewed_at — der Fall aus
        # fingerprint_bestaetigung_ist_kein_mensch.
        self.assertIs(_lh.mensch_aus_markern(
            {"auto_confirmed_via_fingerprint": True, "reviewed_at": 5}), False)

    def test_werkzeug_schreiber(self):
        for w in _lh.NICHT_MENSCH:
            with self.subTest(schreiber=w):
                self.assertIs(_lh.mensch_aus_markern({"reviewed_by": w}), False)

    def test_menschlicher_schreiber_bleibt_mensch(self):
        # golden-audit steht bewusst NICHT in NICHT_MENSCH.
        self.assertIs(_lh.mensch_aus_markern({"reviewed_by": "golden-audit"}), True)

    def test_keine_datei_ist_NICHT_ENTSCHEIDBAR(self):
        """Der Unterschied, der die beiden O17-Arme trennt."""
        self.assertIsNone(_lh.mensch_aus_markern(None))
        self.assertIsNone(_lh.mensch_aus_markern([]))
        self.assertIsNone(_lh.mensch_aus_markern("kaputt"))

    def test_neuer_schreiber_muss_hier_eingetragen_werden(self):
        """Wer in tv-detect Labels schreibt, gehoert in NICHT_MENSCH."""
        self.assertIn("folgen-vergleich.py", _lh.NICHT_MENSCH)
        self.assertIn("agent-review.py", _lh.NICHT_MENSCH)


class EineDefinition(unittest.TestCase):
    """Die Regel darf nicht wieder in Kopien zerfallen."""

    def _laden(self, name):
        sp = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                    _HIER / name)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    # ⚠️ Nicht auf Objekt-IDENTITAET pruefen. Jeder Leser laedt
    # label_herkunft.py ueber eine eigene importlib-Spec und bekommt damit
    # ein eigenes Modulobjekt — gleiche Menge, andere Identitaet. Die
    # Aussage, auf die es ankommt, ist strukturell: NIEMAND legt eine
    # eigene Liste an.

    def test_alle_leser_haben_denselben_inhalt(self):
        for name in ("massstab-audit.py", "golden_v3_vorschlag.py"):
            with self.subTest(datei=name):
                self.assertEqual(self._laden(name).NICHT_MENSCH,
                                 _lh.NICHT_MENSCH)

    def test_niemand_legt_eine_eigene_liste_an(self):
        for name in ("train-head.py", "massstab-audit.py",
                     "golden_v3_vorschlag.py"):
            with self.subTest(datei=name):
                baum = ast.parse((_HIER / name).read_text(encoding="utf-8"))
                literale = [n for n in ast.walk(baum)
                            if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name)
                                    and t.id == "NICHT_MENSCH"
                                    for t in n.targets)
                            and isinstance(n.value, (ast.Set, ast.List,
                                                     ast.Tuple, ast.Dict))]
                self.assertEqual(
                    [n.lineno for n in literale], [],
                    f"{name} setzt NICHT_MENSCH aus einem Literal — die Liste "
                    f"gehoert ausschliesslich in label_herkunft.py")


class TransportDurchDieSchleifen(unittest.TestCase):
    """⚠️ Der Test, der den teuren Fehler verhindert."""

    def setUp(self):
        quelle = (_HIER / "train-head.py").read_text(encoding="utf-8")
        self.baum = ast.parse(quelle)
        self.main = [n for n in ast.walk(self.baum)
                     if isinstance(n, ast.FunctionDef) and n.name == "main"][0]

    def _schleife_um(self, zeile):
        kandidaten = [n for n in ast.walk(self.main)
                      if isinstance(n, (ast.For, ast.While))
                      and n.lineno <= zeile <= n.end_lineno]
        return min(kandidaten, key=lambda n: n.end_lineno - n.lineno, default=None)

    def test_mensch_belegt_wird_im_zweiten_durchgang_neu_zugewiesen(self):
        lesen = [n for n in ast.walk(self.main)
                 if isinstance(n, ast.Name) and n.id == "mensch_belegt"
                 and isinstance(n.ctx, ast.Load)]
        self.assertTrue(lesen, "mensch_belegt wird nirgends gelesen")
        for n in lesen:
            schleife = self._schleife_um(n.lineno)
            self.assertIsNotNone(schleife,
                                 f"Lesen ausserhalb jeder Schleife (Z. {n.lineno})")
            # In DERSELBEN Schleife muss vorher eine Zuweisung stehen.
            zuweisungen = [a for a in ast.walk(schleife)
                           if isinstance(a, ast.Assign)
                           and any(isinstance(t, ast.Name)
                                   and t.id == "mensch_belegt" for t in a.targets)
                           and a.lineno < n.lineno]
            self.assertTrue(
                zuweisungen,
                f"mensch_belegt wird in Z. {n.lineno} gelesen, aber in dieser "
                f"Schleife (Z. {schleife.lineno}-{schleife.end_lineno}) nie "
                f"zugewiesen — es traegt dann den Wert der LETZTEN Aufnahme "
                f"des vorigen Durchgangs. Es muss ueber rec_info reisen.")

    def test_der_zweite_durchgang_liest_es_aus_rest(self):
        """Und zwar aus dem Tupel, nicht aus einer anderen Schleifenvariable."""
        quelle = (_HIER / "train-head.py").read_text(encoding="utf-8")
        self.assertIn("mensch_belegt = rest[", quelle,
                      "der zweite Durchgang muss mensch_belegt aus rec_info "
                      "auspacken")

    def test_beide_baustellen_von_rec_info_tragen_es(self):
        quelle = (_HIER / "train-head.py").read_text(encoding="utf-8")
        bau = quelle.count("rec_info = (uuid, title, ads, which")
        mit = quelle.count("confirmed_ad_skips, mensch_belegt)")
        self.assertGreaterEqual(bau, 1, "keine rec_info-Baustelle gefunden")
        self.assertEqual(mit, bau,
                         f"{bau} rec_info-Baustellen, aber nur {mit} tragen "
                         f"mensch_belegt — eine vergessene Baustelle liefert "
                         f"still ein zu kurzes Tupel und damit None")


if __name__ == "__main__":
    unittest.main(verbosity=2)
