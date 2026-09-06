#!/usr/bin/env python3
"""Tests für den Serien-Folgen-Vergleich (`folgen-vergleich.py`).

Jeder Test hier steht für einen Fehlalarm, der am 2026-09-06 beim Bauen
wirklich aufgetreten ist. Die drei Reißleinen sind nicht vorsorglich
eingebaut, sondern nachträglich — und genau deshalb müssen sie festgehalten
werden: sie sehen wie überflüssige Vorsicht aus, bis man sie entfernt.

  1. Die Verschiebungssuche fand bei einer BEREITS ausgerichteten Serie
     noch ±35–55 s und schob damit die Folgen auseinander, die verglichen
     werden sollten.
  2. Erzählserien (Charmed, Futurama) haben kein gemeinsames Blockmuster;
     die Suche rutschte an die Bereichsgrenze und erzeugte reines
     Ausrichtungsrauschen.
  3. Ein konstanter Versatz kann keinen DAUERunterschied ausgleichen. Eine
     66-min-Folge gegen eine 72-min-Gruppe erzeugte Phantom-Kanten.

Ausführen: python3 scripts/test_folgen_vergleich.py
"""
import argparse
import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fv", Path(__file__).resolve().parent / "folgen-vergleich.py")
_fv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fv)


def std_args(**über):
    a = dict(min_folgen=5, min_andere=4, fp_schwelle=0.25, miss_schwelle=0.70,
             min_luecke=60, spanne=300, schritt=5, vs_marge=0.05,
             max_unruhig=0.34, kanten_naehe=10, dauer_toleranz=0.10)
    a.update(über)
    return argparse.Namespace(**a)


def folge(uuid, dauer, bloecke):
    return (uuid, 0, float(dauer), [[float(a), float(b)] for a, b in bloecke])


class NormTitel(unittest.TestCase):
    """Muss deckungsgleich mit tv-recorder normalizeTitle() bleiben — sonst
    gruppiert der Bericht anders als der Dienst, den er beurteilt."""

    def test_striche_und_doppelpunkt(self):
        for roh, erwartet in [
                ("Die Geissens - Eine Familie", "Die Geissens"),
                ("Die Geissens – Eine Familie!", "Die Geissens"),
                ("Die Geissens — Eine Familie", "Die Geissens"),
                ("Tatort: Der Fall", "Tatort"),
                ("Galileo", "Galileo"),
                ("Bibi-Blocksberg", "Bibi-Blocksberg")]:
            with self.subTest(roh=roh):
                self.assertEqual(_fv.norm_titel(roh), erwartet)


class Verschiebungsbremse(unittest.TestCase):
    """Reissleine 1."""

    def test_ausgerichtete_serie_bleibt_bei_null(self):
        n = 3000
        mittel = [1.0 if 1000 <= i < 1400 else 0.0 for i in range(n)]
        ind = _fv.indikator([[1000, 1400]], n)
        s, _ = _fv.beste_verschiebung(ind, mittel, 300, 5, 0.05)
        self.assertEqual(s, 0, "identische Lage darf keine Verschiebung ergeben")

    def test_echte_verschiebung_wird_gefunden(self):
        n = 3000
        mittel = [1.0 if 1000 <= i < 1400 else 0.0 for i in range(n)]
        ind = _fv.indikator([[1120, 1520]], n)      # 120 s spaeter
        s, _ = _fv.beste_verschiebung(ind, mittel, 300, 5, 0.05)
        # ⚠️ Nicht auf die Sekunde festnageln. `guete()` tastet nur jede
        # ZWEITE Sekunde ab, das Optimum ist dadurch flach, und der
        # Gleichstand-Tiebreak nimmt den kleineren Betrag — hier 115 statt
        # 120. Eine Schrittweite Abweichung ist Bauart, kein Fehler.
        self.assertLessEqual(abs(s - 120), 5, f"Verschiebung {s} zu weit von 120")

    def test_ohne_eigene_bloecke_keine_verschiebung(self):
        n = 1000
        mittel = [1.0] * n
        self.assertEqual(_fv.beste_verschiebung(bytearray(n), mittel, 300, 5, 0.05)[0], 0)


class Ausrichtungs_Reissleine(unittest.TestCase):
    """Reissleine 2: keine gemeinsame Struktur → gar kein Urteil."""

    # ⚠️ Die Reissleine feuert NUR, wenn der echte Versatz GROESSER als der
    # Suchbereich ist. Sonst gewinnt Verschiebung 0 ohnehin: die Referenz
    # aus Runde 1 ist der Mittelwert ueber ALLE Folgen einschliesslich der
    # Zielfolge, die dort also immer ihren eigenen Beitrag vorfindet. Diese
    # Selbstueberlappung plus die 5-%-Bremse halten kleine Versaetze bei 0.
    # Beim Bauen der Tests hat mich das ueberrascht: drei synthetische
    # "chaotische" Serien ergaben durchweg Verschiebung 0.

    def test_eine_unausrichtbare_folge_faellt_raus_ohne_die_serie_zu_kippen(self):
        # 11 gleiche Folgen + 1, deren Block 400 s spaeter liegt (Suchbereich
        # 300). Deren Suche rutscht an die Grenze; sie wird aus dem Vergleich
        # genommen, die Serie bleibt auswertbar.
        folgen = [folge(f"n{i}", 3000, [[1000, 1400]]) for i in range(11)]
        folgen.append(folge("weit", 3000, [[1400, 1800]]))
        funde, info = _fv.serie_pruefen(folgen, std_args())
        self.assertFalse(info["uebersprungen"])
        self.assertEqual(info["n_folgen"], 11)
        self.assertEqual([f for f in funde if f["uuid"] == "weit"], [],
                         "eine ausgeschlossene Folge darf nichts melden")

    def test_serie_wird_uebersprungen_wenn_zu_viele_nicht_ausrichtbar(self):
        # 6 von 16 = 37 % ueber der Schwelle von 34 %.
        folgen = [folge(f"n{i}", 3000, [[1000, 1400]]) for i in range(10)]
        folgen += [folge(f"v{i}", 3000, [[1400, 1800]]) for i in range(6)]
        funde, info = _fv.serie_pruefen(folgen, std_args())
        self.assertTrue(info["uebersprungen"], f"info={info}")
        self.assertIn("konvergente", info["grund"])
        self.assertEqual(funde, [])

    def test_saubere_serie_wird_nicht_uebersprungen(self):
        folgen = [folge(f"u{i}", 3000, [[1000, 1400], [2000, 2300]])
                  for i in range(6)]
        funde, info = _fv.serie_pruefen(folgen, std_args())
        self.assertFalse(info["uebersprungen"])
        self.assertEqual(funde, [], "identische Folgen duerfen nichts melden")


class Dauerfilter(unittest.TestCase):
    """Reissleine 3."""

    def test_kurze_folge_wird_nicht_gegen_lange_gehalten(self):
        # Sechs 4300-s-Folgen mit einem spaeten Block, dazu eine 3600-s-Folge,
        # deren Blockmuster zu IHRER Laenge passt. Ohne Dauerfilter meldet
        # die kurze Folge eine Luecke dort, wo die langen noch Werbung haben.
        lang = [folge(f"L{i}", 4300, [[1000, 1400], [3400, 3800]]) for i in range(6)]
        kurz = [folge("K", 3600, [[1000, 1400]])]
        funde, info = _fv.serie_pruefen(lang + kurz, std_args())
        self.assertFalse(info["uebersprungen"])
        fuer_kurz = [f for f in funde if f["uuid"] == "K"]
        self.assertEqual(fuer_kurz, [],
                         f"kurze Folge darf nicht gegen die langen gemessen "
                         f"werden, meldete aber {fuer_kurz}")


class Fundarten(unittest.TestCase):

    def test_einzelgaenger_wird_erkannt(self):
        # Eine Folge hat einen Block, den keine andere kennt.
        folgen = [folge(f"u{i}", 3000, [[1000, 1400]]) for i in range(6)]
        folgen.append(folge("X", 3000, [[1000, 1400], [2000, 2300]]))
        funde, _ = _fv.serie_pruefen(folgen, std_args())
        eg = [f for f in funde if f["uuid"] == "X" and f["art"] == "einzelgaenger"]
        self.assertEqual(len(eg), 1, f"erwartet EIN einzelgaenger, bekam {funde}")
        self.assertEqual((eg[0]["von"], eg[0]["bis"]), (2000, 2300))
        self.assertEqual(eg[0]["n_mit_block"], 0)

    def test_luecke_wird_erkannt(self):
        # Umgekehrt: allen anderen fehlt bei X der zweite Block.
        folgen = [folge(f"u{i}", 3000, [[1000, 1400], [2000, 2300]]) for i in range(6)]
        folgen.append(folge("X", 3000, [[1000, 1400]]))
        funde, _ = _fv.serie_pruefen(folgen, std_args())
        lk = [f for f in funde if f["uuid"] == "X" and f["art"] == "luecke"]
        self.assertEqual(len(lk), 1, f"erwartet EINE luecke, bekam {funde}")

    def test_angrenzende_fehlstelle_ist_kante_keine_luecke(self):
        """Der Unterschied, der über die Deutung entscheidet.

        ⚠️ Und die Deutung ist widerlegt: `kante-*` ist am 2026-09-06 mit
        0 von 8 durch die Bildprobe gefallen (eine echt verschobene
        Werbepause erzeugt dasselbe Signal). Der Test haelt nur fest, DASS
        getrennt wird — nicht, dass die Kante etwas wert ist.
        """
        folgen = [folge(f"u{i}", 3000, [[1000, 1500]]) for i in range(6)]
        folgen.append(folge("X", 3000, [[1200, 1500]]))   # Start 200 s spaeter
        funde, _ = _fv.serie_pruefen(folgen, std_args())
        arten = {f["art"] for f in funde if f["uuid"] == "X"}
        self.assertIn("kante-start", arten, f"bekam {funde}")
        self.assertNotIn("luecke", arten)


class Beweislast(unittest.TestCase):

    def test_gewicht_skaliert_mit_der_zahl_der_vergleichsfolgen(self):
        """Fuenf Folgen tragen weniger als vierzehn — sonst stehen
        Zufallsfunde aus Mini-Serien oben in der Liste."""
        def gewicht(n):
            folgen = [folge(f"u{i}", 3000, [[1000, 1400]]) for i in range(n)]
            folgen.append(folge("X", 3000, [[1000, 1400], [2000, 2300]]))
            funde, _ = _fv.serie_pruefen(folgen, std_args(min_andere=4))
            eg = [f for f in funde if f["uuid"] == "X"]
            return eg[0]["gewicht"] if eg else None
        klein, gross = gewicht(5), gewicht(12)
        self.assertIsNotNone(klein)
        self.assertIsNotNone(gross)
        self.assertLess(klein, gross)


if __name__ == "__main__":
    unittest.main(verbosity=2)
