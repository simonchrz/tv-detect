#!/usr/bin/env python3
"""Die Herkunftspruefung muss auch die eingespeisten Archiv-Aufnahmen treffen.

Gefunden 2026-09-13. `--herkunft-streng` meldete "0 weitere ohne lesbare
Quelle" — und das sah aus wie ein sauberes Nullergebnis. Tatsaechlich
haengt die Archiv-Einspeisung ihre Aufnahmen DIREKT an `per_rec` und lief
damit an der Pruefung im Live-Durchgang vorbei. 584 eingespeiste Aufnahmen,
645 davon korpusweit mit `has_user=True`, waren fuer beide Schalter
unsichtbar. Genau diese Gruppe IST die Frage O18.

Ein Schalter, der seine eigene Zielgruppe nie sieht, meldet 0 und ist von
einem gemessenen "kein Effekt" nicht zu unterscheiden.
"""
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
QUELL = (HIER / "train-head.py").read_text()

# Der Einspeise-Block, von der Schleife bis zum per_rec.append.
_ANF = QUELL.index('for npz_path in sorted(archive_dir.glob("*.npz")):')
_END = QUELL.index("injected += 1", _ANF)
BLOCK = QUELL[_ANF:_END]


class HerkunftImArchiv(unittest.TestCase):
    def test_einspeisung_prueft_die_herkunft(self):
        self.assertIn("args.herkunft_belegt", BLOCK,
                      "die Einspeisung ignoriert --herkunft-belegt")
        self.assertIn("args.herkunft_streng", BLOCK,
                      "die Einspeisung ignoriert --herkunft-streng")

    def test_beide_zustaende_werden_unterschieden(self):
        # `False` heisst "Maschine", `None` heisst "nicht entscheidbar".
        # Sie gleich zu behandeln waere eine Annahme, keine Messung --
        # die Unterscheidung traegt die Armtrennung von O18.
        self.assertIn("a_mensch is False", BLOCK)
        self.assertIn("a_mensch is None", BLOCK)
        self.assertIn("herkunft_entzogen.append(u)", BLOCK)
        self.assertIn("herkunft_unentscheidbar.append(u)", BLOCK)

    def test_das_gepruefte_ergebnis_landet_in_per_rec(self):
        # ⚠️ Der eigentliche Fehler: gerechnet und dann NICHT benutzt waere
        # genauso still wie gar nicht gerechnet.
        anhang = QUELL[_END - 900:_END]
        self.assertIn("a_feats, a_labels, a_has_user", anhang,
                      "per_rec bekommt weiter das ungepruefte which-Feld")

    def test_archiv_friert_die_herkunft_ein(self):
        # Ohne diesen Schluessel ist jeder Eintrag nach dem Tod der
        # Aufnahme dauerhaft "nicht entscheidbar".
        self.assertIn('"mensch_belegt": mensch_belegt,', QUELL)

    def test_die_meldung_nennt_den_archiv_anteil(self):
        # Die unsichtbare Gruppe muss in der Zahl sichtbar sein, sonst
        # laesst sich "0" wieder nicht von "nie hingesehen" trennen.
        self.assertIn("davon {_ha[0]} aus dem Archiv", QUELL)
        self.assertIn("davon {_ha[1]} aus dem Archiv", QUELL)
        self.assertIn("herkunft_aus_archiv = (0, 0)", QUELL,
                      "der Zaehler muss vorbelegt sein, sonst faellt die "
                      "Meldung aus, wenn kein Archiv eingespeist wurde")


if __name__ == "__main__":
    unittest.main()
