#!/usr/bin/env python3
"""Tests für den SCHREIBPFAD der Agentenprobe (`--anwenden`).

Dies ist die einzige Stelle in der Kette, die Labels verändert. Sie hat
sieben Schranken, und jede steht für einen Fehler, der in diesem Stapel
schon einmal bezahlt wurde. Schranken sind aber genau die Art Code, die
niemand vermisst, wenn sie stillschweigend aufhört zu greifen — ein
weggefallener Eimer-Filter meldet sich nicht, er schreibt einfach in den
Test-Satz und macht das Gate blind.

Getestet wird deshalb vor allem, was NICHT geschrieben wird.

Die Nahtstellen: `_ar_hole` (liest den aktuellen Label-Stand) und
`_urllib()` (POST) werden ersetzt, es läuft kein Server und es geht keine
Anfrage hinaus. Der Test schlägt fehl, wenn er es doch versucht.

Ausführen: python3 scripts/test_folgen_probe_anwenden.py
"""
import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fp", Path(__file__).resolve().parent / "folgen-vergleich-pruefen.py")
_fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fp)

S = "sendungsinhalt"
W = "produktwerbung"
UUID = "dvr-rtl-1781199600"


class FakeAntwort:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class FakeUrllib:
    """Nimmt POSTs entgegen, statt sie zu senden."""

    def __init__(self):
        self.gesendet = []

    def Request(self, url, data=None, headers=None):
        return {"url": url, "data": data, "headers": headers}

    def urlopen(self, req, timeout=None):
        self.gesendet.append({"url": req["url"],
                              "body": json.loads(req["data"].decode())})
        return FakeAntwort()


class Anwenden(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        wurzel = Path(self._tmp.name)
        self.arbeit = wurzel / "probe"
        self.arbeit.mkdir()
        self.quelle = wurzel / "quelle"
        self.quelle.mkdir()

        self._alt = (_fp.ARBEIT, _fp._ar.QUELLE, _fp._ar.erlaubte_uuids,
                     _fp._ar_hole, _fp._urllib)
        _fp.ARBEIT = self.arbeit
        _fp._ar.QUELLE = self.quelle
        _fp._ar.erlaubte_uuids = lambda: {UUID}

        # Aktueller Label-Stand, den das Anwenden vom Dienst holt.
        self.label = [[100.0, 400.0], [900.0, 1200.0], [1500.0, 1800.0]]
        _fp._ar_hole = lambda url: {"ads": [list(b) for b in self.label],
                                    "duration_s": 3000.0}
        self.netz = FakeUrllib()
        _fp._urllib = lambda: self.netz

    def tearDown(self):
        (_fp.ARBEIT, _fp._ar.QUELLE, _fp._ar.erlaubte_uuids,
         _fp._ar_hole, _fp._urllib) = self._alt
        self._tmp.cleanup()

    # --- Hilfsmittel ---------------------------------------------------

    def quelle_anlegen(self, groesse=4096):
        p = self.quelle / f"{UUID}.ts"
        p.write_bytes(b"x" * groesse)
        return groesse

    def auftrag(self, art="einzelgaenger", urteile=(S, S, S, S, S),
                von=900, bis=1200, eimer="train", uuid=UUID,
                quelle_bytes=4096, k_werbung=W, k_sendung=S):
        d = self.arbeit / f"{uuid}_{von}"
        d.mkdir(parents=True, exist_ok=True)
        punkte, urteil = {}, {}
        for i, kat in enumerate(urteile, 1):
            n = f"bild_{i:02d}.jpg"
            punkte[n] = {"t": float(von + i), "rolle": "streit"}
            urteil[n] = kat
        for rolle, kat in (("kontrolle_werbung", k_werbung),
                           ("kontrolle_sendung", k_sendung)):
            n = f"bild_{rolle}.jpg"
            punkte[n] = {"t": 200.0, "rolle": rolle}
            urteil[n] = kat
        (d / "_loesung.json").write_text(json.dumps({
            "uuid": uuid, "punkte": punkte, "quelle_bytes": quelle_bytes,
            "fund": {"art": art, "von": von, "bis": bis, "eimer": eimer,
                     "uuid": uuid, "serie": "Test", "versatz_s": 0,
                     "konsens": 0.0, "n_andere": 8, "n_mit_block": 0}}))
        (d / "urteil.json").write_text(json.dumps(urteil))
        return d

    def lauf(self, schreiben=True, schwelle=0.6):
        return _fp.anwenden(argparse.Namespace(
            pi="http://test", schwelle=schwelle, schreiben=schreiben))

    # --- der Normalfall ------------------------------------------------

    def test_bestaetigter_einzelgaenger_wird_geloescht(self):
        self.quelle_anlegen()
        self.auftrag()
        self.lauf()
        self.assertEqual(len(self.netz.gesendet), 1)
        body = self.netz.gesendet[0]["body"]
        self.assertEqual(body["ads"], [[100.0, 400.0], [1500.0, 1800.0]],
                         "genau der strittige Block muss fehlen, sonst keiner")
        self.assertIn("/api/recording/", self.netz.gesendet[0]["url"])

    def test_schreiber_wird_als_maschine_gekennzeichnet(self):
        """Schranke 6. Ohne die Kennzeichnung stuende ein MODELLWERT mit
        frischem Zeitstempel im Label und waere von einem menschlichen
        Urteil nicht zu unterscheiden."""
        self.quelle_anlegen()
        self.auftrag()
        self.lauf()
        self.assertEqual(self.netz.gesendet[0]["body"]["reviewed_by"],
                         "folgen-vergleich.py")
        gv3 = importlib.util.spec_from_file_location(
            "gv3", Path(__file__).resolve().parent / "golden_v3_vorschlag.py")
        m = importlib.util.module_from_spec(gv3)
        gv3.loader.exec_module(m)
        self.assertIn(_fp.SCHREIBER, m.NICHT_MENSCH,
                      "der Schreiber MUSS in NICHT_MENSCH stehen, sonst "
                      "rutscht sein Label als 'menschlich' in den Massstab")

    # --- die Schranken -------------------------------------------------

    def test_probelauf_ist_die_vorgabe(self):
        """Schranke 7."""
        self.quelle_anlegen()
        self.auftrag()
        self.lauf(schreiben=False)
        self.assertEqual(self.netz.gesendet, [])

    def test_kante_wird_NIE_geschrieben(self):
        """Schranke 1. kante-* ist am 2026-09-06 mit 0 von 8 durch die
        Bildprobe gefallen — auch ein 'bestaetigter' Kantenfund darf nicht
        ins Label."""
        self.quelle_anlegen()
        # Werbung in der Spanne = fuer eine Kante die Bestaetigung.
        self.auftrag(art="kante-ende", urteile=(W, W, W, W, W))
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_fremder_eimer_wird_abgelehnt(self):
        """Schranke 2. Test- und versiegelte Labels sind die Ground Truth
        des Gates; ein Schreiber dort laesst den Champion per Konstruktion
        gewinnen."""
        self.quelle_anlegen()
        _fp._ar.erlaubte_uuids = lambda: set()      # nichts erlaubt
        self.auftrag(eimer="versiegelt")
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_unlesbarer_ledger_schreibt_nichts(self):
        """erlaubte_uuids() gibt None zurueck, wenn es den Ledger nicht
        lesen kann — fail-closed, nicht fail-open."""
        self.quelle_anlegen()
        _fp._ar.erlaubte_uuids = lambda: None
        self.auftrag()
        self.assertEqual(self.lauf(), 1)
        self.assertEqual(self.netz.gesendet, [])

    def test_falsche_kontrolle_verhindert_das_schreiben(self):
        """Schranke 3."""
        self.quelle_anlegen()
        self.auftrag(k_werbung=S)                   # Werbeblock als Sendung
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_veraltete_quelle_verhindert_das_schreiben(self):
        """Schranke 4. Der Auftrag vom 17.08., angewandt auf eine seither
        neu geholte Quelle: die Bilder stammten aus dem alten Schnitt, ein
        Zeitstempel lag 111 s daneben."""
        self.quelle_anlegen(4096)
        self.auftrag(quelle_bytes=9999)             # Groesse passt nicht mehr
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_veraenderter_labelstand_bricht_ab(self):
        """Schranke 5. Zwischen Probe und Anwenden kann jemand anders das
        Label geaendert haben — dann ueberschreiben wir nicht."""
        self.quelle_anlegen()
        self.auftrag(von=900, bis=1200)
        self.label = [[100.0, 400.0], [950.0, 1300.0]]   # Block verschoben
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_unklare_kontrolle_verhindert_das_schreiben_NICHT(self):
        """Dieselbe Regel wie im Bericht, aus derselben Funktion.

        ⚠️ Der Test steht hier ZUSAETZLICH zum gleichnamigen im
        Bericht-Test: eine Mutationsprobe zeigte, dass die Aenderung sonst
        nur auf einem der beiden Pfade auffaellt. Seit `auswerten` und
        `anwenden` dieselbe `_urteil_fuer` benutzen, ist das eine Regel —
        und beide Tests halten sie fest, damit eine kuenftige Aufspaltung
        auffliegt.
        """
        self.quelle_anlegen()
        self.auftrag(k_werbung="unklar")
        self.lauf()
        self.assertEqual(len(self.netz.gesendet), 1,
                         "Zurueckhaltung auf einer Kontrolle ist kein Fehlurteil")

    def test_fehlende_kontrolle_verhindert_das_schreiben(self):
        """Eine Probe ohne beide Kontrollen ist keine Probe."""
        self.quelle_anlegen()
        d = self.auftrag()
        loes = json.loads((d / "_loesung.json").read_text())
        loes["punkte"] = {k: v for k, v in loes["punkte"].items()
                          if v["rolle"] != "kontrolle_werbung"}
        (d / "_loesung.json").write_text(json.dumps(loes))
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_zu_schwaches_urteil_wird_nicht_geschrieben(self):
        # 2 von 5 Sendung = 40 %, unter der Schwelle von 60 %.
        self.quelle_anlegen()
        self.auftrag(urteile=(S, S, W, W, W))
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_kein_urteil_wird_nicht_geschrieben(self):
        self.quelle_anlegen()
        d = self.auftrag()
        (d / "urteil.json").unlink()
        self.lauf()
        self.assertEqual(self.netz.gesendet, [])

    def test_zweimal_anwenden_schreibt_nur_einmal(self):
        """Der `angewandt`-Vermerk. Ohne ihn wuerde ein zweiter Lauf den
        naechsten Block loeschen, der zufaellig auf die Zeiten passt."""
        self.quelle_anlegen()
        self.auftrag()
        self.lauf()
        self.lauf()
        self.assertEqual(len(self.netz.gesendet), 1)

    def test_ruecknahme_daten_werden_festgehalten(self):
        """Ohne den Vorher-Stand ist die Aenderung nicht zurueckzunehmen —
        die API kennt nur 'setze diese Bloecke'."""
        self.quelle_anlegen()
        d = self.auftrag()
        self.lauf()
        vermerk = json.loads((d / "angewandt").read_text())
        self.assertEqual(vermerk["vorher"], self.label)
        self.assertEqual(len(vermerk["nachher"]), 2)
        self.assertEqual(vermerk["schreiber"], "folgen-vergleich.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
