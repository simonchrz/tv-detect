#!/usr/bin/env python3
"""Tests für die Audio-Dynamik-Spalte (O22) — und für ihre Fallstricke.

Diese Spalte ist die erste Änderung dieser Serie, die BEIDE Seiten
betrifft: Python trainiert den Kopf darauf, Go muss zur Inferenzzeit
dieselbe Zahl liefern. Ein Unterschied stürzt nicht ab, er macht nur
stillschweigend schlechtere Blöcke. Deshalb sind die Kanten hier
festgenagelt, damit die Go-Seite etwas hat, gegen das sie prüfen kann.

Die Definition, verbindlich für beide Seiten:

    d[i] = Standardabweichung von rms[max(0,i-15) : min(n,i+16)]

also ein ZENTRIERTES Fenster von 30 (fenster//2 vor, fenster//2 nach,
plus die Sekunde selbst), an den Rändern beschnitten statt aufgefüllt,
Populations-Standardabweichung (Nenner n, nicht n-1).

Ausführen: python3 scripts/test_audio_dynamik.py
"""
import ast
import types
import unittest
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent


def _lade_funktion(name):
    """Nur die eine Funktion aus train-head.py holen. Das Modul zu
    importieren würde den ganzen Trainer hochfahren."""
    quelle = (_HIER / "train-head.py").read_text()
    baum = ast.parse(quelle)
    fn = [n for n in baum.body
          if isinstance(n, ast.FunctionDef) and n.name == name]
    if not fn:
        raise AssertionError(f"{name} nicht in train-head.py gefunden")
    mod = types.ModuleType("ausschnitt")
    mod.np = np
    exec(compile(ast.Module([fn[0]], []), "<f>", "exec"), mod.__dict__)
    return getattr(mod, name)


dyn = _lade_funktion("audio_dynamik")


class Rechnung(unittest.TestCase):
    def test_konstante_folge_hat_null_schwankung(self):
        d = dyn(np.full(100, 0.5, np.float32), 30)
        self.assertEqual(len(d), 100)
        self.assertTrue(np.allclose(d, 0.0, atol=1e-6),
                        "konstante Lautheit muss Schwankung 0 geben")

    def test_wechselnde_folge_hat_die_erwartete_schwankung(self):
        # Abwechselnd 0 und 1 -> Populations-sd = 0.5 in der Mitte.
        a = np.array([0.0, 1.0] * 100, np.float32)
        d = dyn(a, 30)
        self.assertAlmostEqual(float(d[50]), 0.5, places=3)

    def test_zentriertes_fenster_und_rand(self):
        # Sprung genau in der Mitte: die Schwankung muss DORT am
        # groessten sein, nicht davor oder danach.
        a = np.concatenate([np.zeros(50, np.float32), np.ones(50, np.float32)])
        d = dyn(a, 30)
        self.assertEqual(int(np.argmax(d)), 49,
                         "das Maximum gehoert an die Sprungstelle")
        # Am Rand wird beschnitten, nicht aufgefuellt: dort ist alles
        # konstant, also 0.
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)
        self.assertAlmostEqual(float(d[-1]), 0.0, places=6)

    def test_populations_sd_nicht_stichproben_sd(self):
        # Bei Fenster 2 (=> lo..hi umfasst 3 Werte) muss der Nenner 3
        # sein, nicht 2. numpy std() ohne ddof ist die Referenz.
        a = np.array([0.0, 1.0, 0.0, 1.0, 0.0], np.float32)
        d = dyn(a, 2)
        self.assertAlmostEqual(float(d[2]), float(np.std(a[1:4])), places=6)

    def test_leer_und_kurz(self):
        self.assertEqual(len(dyn(np.zeros(0, np.float32), 30)), 0)
        d = dyn(np.array([0.3], np.float32), 30)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)

    def test_dtype_float32(self):
        self.assertEqual(dyn(np.full(10, 0.5, np.float32), 30).dtype,
                         np.float32)


class Schalter(unittest.TestCase):
    """Der Schalter MUSS aus bleiben, bis die Go-Seite mitzieht.

    Ein Kopf, der auf Schwankung trainiert ist, aber vom Dekoder den
    Pegel gefuettert bekommt, stuerzt nicht ab — er liefert nur
    schlechtere Bloecke. Genau die Sorte stiller Fehler, gegen die dieses
    Projekt sonst Tests hat.
    """

    def test_standard_ist_aus(self):
        quelle = (_HIER / "train-head.py").read_text()
        i = quelle.index('"--audio-dynamik"')
        block = quelle[i:i + 400]
        self.assertIn('action="store_true"', block,
                      "der Schalter muss ein Flag sein, das man setzen MUSS")
        self.assertNotIn("default=True", block)

    def test_wird_am_ladepunkt_angewandt_nicht_bei_der_extraktion(self):
        """Bei der Extraktion angewandt haette der Cache fuer neue
        Aufnahmen die Schwankung und fuer alte den Pegel — ein stiller
        Bruch mitten im Korpus."""
        quelle = (_HIER / "train-head.py").read_text()
        self.assertIn("feats[:, 1281] = audio_dynamik(", quelle,
                      "die Ersetzung gehoert an den Ladepunkt")
        i = quelle.index("def extract_audio_rms_per_second")
        j = quelle.index("def ", i + 10)
        self.assertNotIn("audio_dynamik(", quelle[i:j],
                         "in der Extraktion darf sie NICHT stehen")


class Kopplung(unittest.TestCase):
    """Die Beilage head.audio.json ist die EINZIGE Kopplung.

    Die Bauform „Spalte ersetzen" laesst die Eingabebreite bei 1282, also
    verraet der Kopf-Header nicht, ob er auf dem Pegel oder auf der
    Schwankung trainiert wurde. Ohne Beilage koennte das Gate einen
    Schwankungs-Kopf deployen, waehrend der Dekoder weiter Pegel
    liefert — kein Absturz, nur still schlechtere Bloecke.
    """

    def test_trainer_schreibt_die_beilage_immer(self):
        q = (_HIER / "train-head.py").read_text()
        self.assertIn('with_suffix(".audio.json")', q,
                      "der Trainer muss die Beilage schreiben")
        i = q.index('with_suffix(".audio.json")')
        block = q[i:i + 400]
        self.assertIn('"dynamik"', block)
        # ⚠️ IMMER schreiben, auch bei false: eine vorhandene Datei mit
        # true muss ueberschrieben werden koennen, wenn jemand den
        # Schalter wieder ausmacht.
        self.assertNotIn("if args.audio_dynamik:", q[max(0, i - 200):i],
                         "die Beilage darf nicht nur bei true entstehen")

    def test_daemon_liest_die_beilage_und_raet_nicht(self):
        d = (_HIER.parent / "daemon" / "tv-thumbs-daemon.py").read_text()
        self.assertIn("head.audio.json", d,
                      "der Daemon muss die Beilage lesen")
        i = d.index('_ad_pfad = MODEL_CACHE / "head.audio.json"')
        block = d[i:i + 900]
        self.assertIn('"--audio-dynamik"', block)
        self.assertIn('.get("dynamik")', block,
                      "die Entscheidung haengt am Feld, nicht an einer Heuristik")
        # Fehlende oder kaputte Beilage MUSS auf den Pegel zurueckfallen.
        self.assertIn("except Exception", block)

    def test_go_und_python_haben_dieselbe_fixture(self):
        fix = (_HIER.parent / "internal" / "signals" / "testdata"
               / "audio_dynamik_paritaet.json")
        self.assertTrue(fix.is_file(),
                        "die Paritaets-Fixture fehlt — ohne sie kann die "
                        "Go-Seite nicht gegen die Python-Rechnung pruefen")
        import json as _j
        faelle = _j.loads(fix.read_text())
        self.assertGreaterEqual(len(faelle), 10)
        for f in faelle[:3]:
            ist = dyn(np.array(f["ein"], np.float32), f["fenster"])
            self.assertTrue(np.allclose(ist, f["soll"], atol=1e-5),
                            f"Fixture {f['name']} passt nicht mehr zur "
                            f"Python-Rechnung — neu erzeugen und die "
                            f"Go-Seite pruefen")


def _lade_beilage_leser():
    """audio_semantik_beilage braucht json + Path, nicht np."""
    import json as _j
    quelle = (_HIER / "train-head.py").read_text()
    baum = ast.parse(quelle)
    fn = [n for n in baum.body
          if isinstance(n, ast.FunctionDef)
          and n.name == "audio_semantik_beilage"]
    if not fn:
        raise AssertionError("audio_semantik_beilage nicht in train-head.py")
    mod = types.ModuleType("ausschnitt2")
    mod.json, mod.Path = _j, Path
    exec(compile(ast.Module([fn[0]], []), "<f>", "exec"), mod.__dict__)
    return mod.audio_semantik_beilage


class Beilage(unittest.TestCase):
    """Der Champion wird im Gate auf den Spalten des KANDIDATEN gescort.
    Gleiche Breite heisst nicht gleiche Spalte — nur die Beilage sagt,
    was der deployte Kopf erwartet. Fehlt sie: Pegel, wie vor 2026-09-08."""

    def setUp(self):
        import tempfile
        self.lese = _lade_beilage_leser()
        self.tmp = Path(tempfile.mkdtemp())
        self.head = self.tmp / "head.bin"

    def test_ohne_beilage_gilt_pegel(self):
        self.assertEqual(self.lese(self.head), (False, 30))

    def test_beilage_true_wird_gelesen(self):
        (self.tmp / "head.audio.json").write_text(
            '{"dynamik": true, "fenster": 30, "ts": "x"}')
        self.assertEqual(self.lese(self.head), (True, 30))

    def test_beilage_false_ueberschreibt_nichts_stilles(self):
        (self.tmp / "head.audio.json").write_text(
            '{"dynamik": false, "fenster": 30}')
        self.assertEqual(self.lese(self.head), (False, 30))

    def test_unlesbare_beilage_faellt_auf_pegel(self):
        (self.tmp / "head.audio.json").write_text("{kaputt")
        self.assertEqual(self.lese(self.head), (False, 30))

    def test_gate_zweig_existiert(self):
        # Der Zweig, der das head-to-head bei Semantik-Unterschied
        # aussetzt, darf nicht still verschwinden.
        quelle = (_HIER / "train-head.py").read_text()
        self.assertIn("Audio-Semantik differiert", quelle)
        self.assertIn("and _audio_gleich", quelle)


if __name__ == "__main__":
    unittest.main(verbosity=2)
