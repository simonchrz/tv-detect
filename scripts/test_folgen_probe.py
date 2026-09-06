#!/usr/bin/env python3
"""Tests für die Agentenprobe (`folgen-vergleich-pruefen.py`).

⚠️ Der gefährlichste Fehler in diesem Skript ist eine vertauschte
POLARITÄT. `kante-*` und `luecke` behaupten Werbung in der strittigen
Spanne, `einzelgaenger` behauptet Sendung. Wer das verwechselt, liest
JEDES Ergebnis spiegelverkehrt — und zwar ohne Fehlermeldung: aus
„8 von 8 widerlegt" würde „8 von 8 bestätigt", und danach löscht das
Anwenden korrekte Blöcke.

Am 2026-09-06 hingen an dieser Unterscheidung zwei gegensätzliche
Ergebnisse am selben Tag (kante 0/8, einzelgaenger 8/8).

Der zweite Prüfstein ist die Kontrolle: `unklar` darauf ist
ZURÜCKHALTUNG, kein Fehlurteil. Die erste Fassung hat deswegen zwei
gültige Läufe verworfen.

Ausführen: python3 scripts/test_folgen_probe.py
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fp", Path(__file__).resolve().parent / "folgen-vergleich-pruefen.py")
_fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fp)

S = "sendungsinhalt"
W = "produktwerbung"
U = "unklar"


def auftrag(tmp, art, streit_urteile, k_werbung=W, k_sendung=S):
    """Legt ein Probe-Verzeichnis an und gibt seinen Pfad zurück."""
    d = Path(tmp) / f"rec_{art}_{len(list(Path(tmp).iterdir()))}"
    d.mkdir(parents=True)
    punkte, urteil = {}, {}
    for i, kat in enumerate(streit_urteile, 1):
        name = f"bild_{i:02d}.jpg"
        punkte[name] = {"t": 100.0 + i, "rolle": "streit"}
        urteil[name] = kat
    for rolle, kat in (("kontrolle_werbung", k_werbung), ("kontrolle_sendung", k_sendung)):
        name = f"bild_{rolle}.jpg"
        punkte[name] = {"t": 900.0, "rolle": rolle}
        urteil[name] = kat
    (d / "_loesung.json").write_text(json.dumps({
        "uuid": "dvr-test-1", "punkte": punkte,
        "fund": {"art": art, "von": 100, "bis": 200, "eimer": "train",
                 "uuid": "dvr-test-1", "serie": "Test", "versatz_s": 0,
                 "konsens": 0.0, "n_andere": 8, "n_mit_block": 0},
    }))
    (d / "urteil.json").write_text(json.dumps(urteil))
    return d


class Polaritaet(unittest.TestCase):

    def test_tabelle_ist_vollstaendig_und_richtig_herum(self):
        self.assertEqual(_fp.POLARITAET["kante-start"], "werbung")
        self.assertEqual(_fp.POLARITAET["kante-ende"], "werbung")
        self.assertEqual(_fp.POLARITAET["luecke"], "werbung")
        self.assertEqual(_fp.POLARITAET["einzelgaenger"], "sendung",
                         "einzelgaenger ist die GEGENrichtung — dort behauptet "
                         "der Vergleich Sendung, nicht Werbung")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_einzelgaenger_mit_sendung_ist_bestaetigt(self):
        d = auftrag(self._tmp.name, "einzelgaenger", [S, S, S, S, S])
        _, anteil, falsch = _fp._urteil_fuer(d)
        self.assertFalse(falsch)
        self.assertEqual(anteil, 1.0)

    def test_einzelgaenger_mit_werbung_ist_widerlegt(self):
        d = auftrag(self._tmp.name, "einzelgaenger", [W, W, W, W, W])
        _, anteil, _ = _fp._urteil_fuer(d)
        self.assertEqual(anteil, 0.0)

    def test_kante_mit_werbung_ist_bestaetigt(self):
        d = auftrag(self._tmp.name, "kante-ende", [W, W, W, W, W])
        _, anteil, _ = _fp._urteil_fuer(d)
        self.assertEqual(anteil, 1.0)

    def test_kante_mit_sendung_ist_widerlegt(self):
        """Der reale Fall vom 2026-09-06: 8 von 8 so."""
        d = auftrag(self._tmp.name, "kante-ende", [S, S, S, S, S])
        _, anteil, _ = _fp._urteil_fuer(d)
        self.assertEqual(anteil, 0.0)

    def test_dieselben_bilder_ergeben_gegensaetzliche_urteile(self):
        """Die Probe auf den Spiegel: identische Klassifikationen müssen je
        nach Fundart das GEGENTEIL bedeuten."""
        eg = _fp._urteil_fuer(auftrag(self._tmp.name, "einzelgaenger", [S] * 5))[1]
        ka = _fp._urteil_fuer(auftrag(self._tmp.name, "kante-start", [S] * 5))[1]
        self.assertEqual(eg, 1.0)
        self.assertEqual(ka, 0.0)


class Kontrollen(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_richtige_kontrollen_gelten_als_bestanden(self):
        d = auftrag(self._tmp.name, "einzelgaenger", [S] * 5)
        self.assertFalse(_fp._urteil_fuer(d)[2])

    def test_falsche_kontrolle_disqualifiziert(self):
        # Das Kontrollbild aus einem Werbeblock als Sendung eingeordnet.
        d = auftrag(self._tmp.name, "einzelgaenger", [S] * 5, k_werbung=S)
        self.assertTrue(_fp._urteil_fuer(d)[2])

    def test_unklare_kontrolle_disqualifiziert_NICHT(self):
        """Zurückhaltung ist ein Gütezeichen, kein Fehlurteil
        (agent_review_schutzkette). Die erste Fassung hat hier zwei
        gültige Läufe verworfen."""
        d = auftrag(self._tmp.name, "einzelgaenger", [S] * 5, k_werbung=U)
        self.assertFalse(_fp._urteil_fuer(d)[2])

    def test_unklare_streitbilder_zaehlen_nicht_mit(self):
        # 3x Sendung, 2x unklar → 100 % der BESTIMMTEN, nicht 60 %.
        d = auftrag(self._tmp.name, "einzelgaenger", [S, S, S, U, U])
        self.assertEqual(_fp._urteil_fuer(d)[1], 1.0)

    def test_nur_unklar_ergibt_kein_urteil(self):
        d = auftrag(self._tmp.name, "einzelgaenger", [U] * 5)
        self.assertIsNone(_fp._urteil_fuer(d)[1])


class Kontrollpunkte(unittest.TestCase):
    """Eine Kontrolle darf nie aus der strittigen Spanne stammen — sonst
    prüft sie genau das, was sie absichern soll."""

    def test_kontrollen_meiden_die_strittige_spanne(self):
        bloecke = [[100.0, 400.0], [1000.0, 1300.0]]
        werb, send = _fp.kontrollpunkte(bloecke, 2000.0, (1000, 1300))
        self.assertIsNotNone(werb)
        self.assertIsNotNone(send)
        for t in (werb, send):
            self.assertFalse(1000 - _fp.KONTROLL_ABSTAND <= t <= 1300 + _fp.KONTROLL_ABSTAND,
                             f"Kontrolle bei {t} liegt zu nah an der Streitspanne")
        self.assertTrue(any(a <= werb <= b for a, b in bloecke))
        self.assertFalse(any(a <= send <= b for a, b in bloecke))

    def test_ohne_zweiten_block_keine_werbe_kontrolle(self):
        # Nur EIN Block, und der ist die strittige Spanne → nichts uebrig.
        werb, _ = _fp.kontrollpunkte([[1000.0, 1300.0]], 2000.0, (1000, 1300))
        self.assertIsNone(werb, "Probe ohne saubere Kontrolle darf nicht laufen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
