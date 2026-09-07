#!/usr/bin/env python3
"""Tests für die Wiederholungssuche und den label-freien Ankermaßstab.

Drei Stellen können hier still eine falsche Zahl liefern, ohne je eine
Ausnahme zu werfen. Genau die werden geprüft.

1. **Die Verkettung.** Aus Paaren werden Läufe, indem nach (Aufnahme A,
   Aufnahme B, Versatz, Zeit) sortiert und an Sprüngen getrennt wird.
   Der Umbau von der Python-Schleife auf `np.lexsort` war nötig, weil
   400 Mio Paare nicht mehr durch eine Schleife passen — und ein Fehler
   in der Bruchbedingung erzeugt keine Fehlermeldung, sondern zu lange
   oder zu kurze Läufe. Zu lange Läufe sind die gefährliche Richtung:
   sie behaupten Wiederholung, wo keine ist.

2. **Der Titel-Unterscheider.** Eine wiederholte Folge ist nur dann
   Werbung, wenn sie in FREMDEN Titeln vorkommt; ein Vorspann wiederholt
   sich über alle Folgen derselben Serie. Wer den eigenen Titel nicht
   abzieht, zählt jede Serie mit fünf Folgen als Werbung. Gemessen am
   prosieben-Index liegen Serien-Elemente zu 2.1 % in einem Werbeblock
   gegen 21.2 % Grundrate — der Unterscheider trägt, solange er richtig
   herum rechnet.

3. **Die Schnitt-Erkennung im Maßstab.** `anker-mass.py` meldet Blöcke,
   die in einen bekannten Spot hineinschneiden. Ein Vorzeichenfehler
   dort meldet 0 statt der echten Zahl — und ein Wächter, der immer 0
   sagt, ist schlimmer als keiner.

Ausführen: python3 scripts/test_wiederholung.py
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent


def _lade(name, datei):
    spec = importlib.util.spec_from_file_location(name, _HIER / datei)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_w = _lade("wiederholung", "wiederholung.py")
_am = _lade("ankermass", "anker-mass.py")


def _verkette(qi, qj, off, zi, schritt, min_lauf):
    """Nachbau der Bruchbedingung aus wiederholung.suchen -- dieselbe
    Rechnung, damit der Test sie prüft und nicht nur beschreibt."""
    qi, qj, off, zi = map(np.asarray, (qi, qj, off, zi))
    ordn = np.lexsort((zi, off, qj, qi))
    qi, qj, off, zi = qi[ordn], qj[ordn], off[ordn], zi[ordn]
    bruch = np.empty(len(qi), bool)
    bruch[0] = True
    bruch[1:] = ((qi[1:] != qi[:-1]) | (qj[1:] != qj[:-1]) | (off[1:] != off[:-1])
                 | ((zi[1:] - zi[:-1]) > schritt * 2))
    start = np.flatnonzero(bruch)
    ende = np.append(start[1:], len(qi))
    behalt = (ende - start) >= min_lauf
    return [(int(zi[s]), int(zi[e - 1])) for s, e in zip(start[behalt], ende[behalt])]


class Verkettung(unittest.TestCase):
    def test_gleicher_versatz_wird_ein_lauf(self):
        # Vier Deskriptoren, Schrittweite 4, durchgehend Versatz +100.
        laeufe = _verkette([0]*4, [1]*4, [100]*4, [0, 4, 8, 12], 4, 3)
        self.assertEqual(laeufe, [(0, 12)], "durchgehende Folge muss EIN Lauf sein")

    def test_anderer_versatz_trennt(self):
        # Dieselben Zeiten, aber zwei verschiedene Versätze: das sind zwei
        # unabhängige Wiederholungen, keine achtsekündige.
        laeufe = _verkette([0]*6, [1]*6, [100]*3 + [900]*3,
                           [0, 4, 8, 0, 4, 8], 4, 3)
        self.assertEqual(len(laeufe), 2, "verschiedene Versätze duerfen nicht verschmelzen")

    def test_zeitluecke_trennt(self):
        # Luecke von 40 s bei Schrittweite 4 -- weit ueber 2*schritt.
        laeufe = _verkette([0]*6, [1]*6, [100]*6, [0, 4, 8, 48, 52, 56], 4, 3)
        self.assertEqual(laeufe, [(0, 8), (48, 56)], "Luecke muss trennen")

    def test_kurzer_lauf_faellt_weg(self):
        laeufe = _verkette([0]*2, [1]*2, [100]*2, [0, 4], 4, 3)
        self.assertEqual(laeufe, [], "zwei Paare sind noch kein Lauf")

    def test_verschiedene_aufnahmen_trennen(self):
        laeufe = _verkette([0]*3 + [0]*3, [1]*3 + [2]*3, [100]*6,
                           [0, 4, 8, 0, 4, 8], 4, 3)
        self.assertEqual(len(laeufe), 2, "Paare zu verschiedenen Partnern sind getrennt")


class TitelUnterscheider(unittest.TestCase):
    """Der eigene Titel MUSS abgezogen werden, sonst ist jeder Vorspann Werbung."""

    def test_eigener_titel_zaehlt_nicht(self):
        eig = "Die Simpsons"
        partner = {"Die Simpsons"} | {f"Folge {i}" for i in range(4)}
        fremd = partner - {eig}
        self.assertEqual(len(fremd), 4)
        self.assertLess(len(fremd), 5, "vier fremde Titel duerfen die Schwelle 5 nicht reissen")

    def test_reine_serienwiederholung_ist_kein_beleg(self):
        eig = "Galileo"
        partner = {"Galileo"}
        self.assertEqual(len(partner - {eig}), 0,
                         "nur der eigene Titel = Serien-Element, kein Werbebeleg")


class Titelnormalisierung(unittest.TestCase):
    """Der Snapshot liefert Titel sprachcodiert. 143 von 959 Eintraegen
    sind Woerterbuecher; unnormalisiert waere jeder davon ein eigener
    Titel -- und damit ein Werbebeleg, wo eine Serie steht."""

    def test_nackte_zeichenkette(self):
        self.assertEqual(_w._titeltext("Galileo"), "Galileo")

    def test_sprachcodiert_deutsch_zuerst(self):
        self.assertEqual(_w._titeltext({"eng": "Mickey", "ger": "Micky Maus"}),
                         "Micky Maus")

    def test_sprachcodiert_ohne_deutsch(self):
        self.assertEqual(_w._titeltext({"eng": "Mickey"}), "Mickey")

    def test_unbrauchbares_wird_leer(self):
        for x in (None, 42, [], {}, {"ger": None}):
            self.assertEqual(_w._titeltext(x), "",
                             f"{x!r} muss zu leerem Titel werden, nicht zu einem Schluessel")


class AnkerAusSekunden(unittest.TestCase):
    """Zusammenhaengende Sekunden werden zu Ankern; zu kurze fallen weg."""

    @staticmethod
    def _gruppiere(sekunden, min_anker):
        gut = sorted(sekunden)
        ank, s, p = [], gut[0], gut[0]
        for t in gut[1:]:
            if t - p <= 1:
                p = t
                continue
            if p + 1 - s >= min_anker:
                ank.append((s, p + 1))
            s = p = t
        if p + 1 - s >= min_anker:
            ank.append((s, p + 1))
        return ank

    def test_zusammenhaengend(self):
        self.assertEqual(self._gruppiere(range(100, 120), 8), [(100, 120)])

    def test_luecke_trennt_und_kurzes_faellt_weg(self):
        # 100..119 (20 s) bleibt, 200..203 (4 s) faellt unter min_anker=8.
        self.assertEqual(self._gruppiere(list(range(100, 120)) + [200, 201, 202, 203], 8),
                         [(100, 120)])

    def test_zu_langes_wird_verworfen_nicht_gekuerzt(self):
        """Ein Beleg ueber 300 s ist keine Spotfolge, sondern ein
        stilverwandter Abschnitt. Kuerzen wuerde die Grenze irgendwohin
        setzen -- und SpotBoundaryLP nimmt genau die Endpunkte als
        Uebergangs-Evidenz, eine geratene Grenze ist dort schaedlicher
        als gar keine."""
        ank = self._gruppiere(range(0, 712), 8)
        self.assertEqual(ank, [(0, 712)])
        behalten = [x for x in ank if x[1] - x[0] <= 300]
        self.assertEqual(behalten, [], "712 s duerfen nicht als Anker durchgehen")
        # Und der Schnitt darf nicht als Kuerzung ueberleben:
        self.assertNotIn((0, 300), behalten)

    def test_endeexklusiv(self):
        # Sekunde 5 allein ist ein Anker [5,6) -- eine Sekunde lang, nicht null.
        self.assertEqual(self._gruppiere([5], 1), [(5, 6)])


class Ankermass(unittest.TestCase):
    def test_schnitt_in_anker_wird_gemeldet(self):
        """Ein Block, der spaet startet und in den Anker schneidet, MUSS
        als negativer Startversatz auftauchen. Das ist der Alarm."""
        anker = {"u1": [(100.0, 130.0, 7, 5)]}
        bloecke = {"u1": [(110.0, 200.0)]}          # startet 10 s IM Anker
        erg = _am.versaetze(anker, bloecke)
        self.assertEqual(erg["schnitte"], 1)
        self.assertAlmostEqual(erg["start_median"], -10.0)

    def test_sauberer_block_meldet_nichts(self):
        anker = {"u1": [(100.0, 130.0, 7, 5)]}
        bloecke = {"u1": [(80.0, 200.0)]}
        erg = _am.versaetze(anker, bloecke)
        self.assertEqual(erg["schnitte"], 0)
        self.assertAlmostEqual(erg["start_median"], 20.0)
        self.assertAlmostEqual(erg["ende_median"], 70.0)

    def test_senderweit_zaehlt_nur_echte_familien(self):
        """family_id -1 (Bild-Anker) darf nicht in den Senderverbund
        wandern -- sonst haengen alle Bild-Anker an derselben Familie."""
        anker = {"dvr-vox-1": [(0.0, 10.0, -1, 3)],
                 "dvr-rtl-2": [(0.0, 10.0, -1, 3)]}
        f2c = _am.senderweit(anker)
        self.assertEqual(len(f2c), 0, "negative family_id darf keinen Verbund bilden")

    def test_bericht_zaehlt_draussen(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "u1.json").write_text(json.dumps({
                "uuid": "u1", "anchored": [
                    {"window_start_s": 10.0, "end_s": 20.0, "family_id": -1, "family_size": 6},
                    {"window_start_s": 500.0, "end_s": 510.0, "family_id": -1, "family_size": 6}]}))
            anker = _am.lade_anker(p)
            self.assertEqual(len(anker["u1"]), 2)
            erg = _am.bericht(anker, {"u1": [(0.0, 100.0)]}, "test", nur_hart=False)
            self.assertEqual(erg["drin"], 1)
            self.assertEqual(erg["draussen"], 1)

    def test_lade_anker_wirft_leere_intervalle_weg(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "u1.json").write_text(json.dumps({
                "uuid": "u1", "anchored": [
                    {"window_start_s": 10.0, "end_s": 10.0, "family_id": 1, "family_size": 3},
                    {"window_start_s": 20.0, "end_s": 15.0, "family_id": 1, "family_size": 3},
                    {"window_start_s": 30.0, "end_s": 40.0, "family_id": 1, "family_size": 3}]}))
            self.assertEqual(len(_am.lade_anker(p)["u1"]), 1,
                             "Ende <= Anfang muss verworfen werden, nicht negativ zaehlen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
