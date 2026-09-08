#!/usr/bin/env python3
"""Jede Beilage, die der Trainer schreibt, muss der Gateway auch herausgeben.

WARUM ES DIESEN TEST GIBT
-------------------------
Am 2026-09-08 wurde `head.audio.json` eingefuehrt — die einzige Kopplung
zwischen einem Kopf, der auf der Lautheits-SCHWANKUNG trainiert ist, und
einem Dekoder, der ihm dieselbe Zahl liefern muss. Beide Enden waren
gebaut und getestet. Trotzdem riss die Kette in der MITTE:

  * Die Upload-Whitelist im tv-recorder (`training.go`) ist `head.*` —
    die Datei wurde also mit hochgeladen.
  * Die Download-Whitelist (`main.go`, `detectModelWhitelist`) ist
    EXPLIZIT und kannte sie nicht.

Ergebnis: der Kopf war deployt, die Beilage lieferte 404, der Daemon fiel
auf „Pegel" zurueck — und haette dem Kopf still die falsche Spalte
gefuettert. Kein Absturz, nur schlechtere Bloecke. Gefunden nur, weil
nach dem Deploy von Hand nachgesehen wurde.

Die zwei Whitelists leben in verschiedenen Dateien und driften
zwangslaeufig auseinander. Dieser Test haelt sie zusammen.

Ausfuehren: python3 scripts/test_beilagen_whitelist.py
"""
import re
import unittest
from pathlib import Path

_HIER = Path(__file__).resolve().parent
_TRAINER = _HIER / "train-head.py"
_RECORDER = Path.home() / "src" / "tv-receiver" / "cmd" / "tv-recorder" / "main.go"


def beilagen_die_der_daemon_holt():
    """Alle detect-models-Namen, die der Daemon herunterzuladen versucht.

    DAS ist die Kopplung, die reissen kann: der Daemon holt, der Gateway
    gibt heraus. Was der Trainer sonst noch neben head.bin legt (etwa
    head.per-rec-iou.json, das nur der naechtliche Lauf selbst liest),
    muss NICHT abrufbar sein — ein Test, der das verlangt, zwingt jede
    lokale Datei in die Whitelist und entwertet sie.
    """
    q = (_HIER.parent / "daemon" / "tv-thumbs-daemon.py").read_text()
    return set(re.findall(r'detect-models/([\w.-]+\.(?:json|bin|onnx))', q))


def whitelist_des_gateways():
    q = _RECORDER.read_text()
    i = q.index("detectModelWhitelist")
    block = q[i:q.index("}", q.index("{", i))]
    return set(re.findall(r'"([\w.-]+\.(?:json|bin|onnx))"', block))


class Kopplung(unittest.TestCase):
    def test_daemon_holt_ueberhaupt_etwas(self):
        b = beilagen_die_der_daemon_holt()
        self.assertGreaterEqual(len(b), 2,
                                f"nur {b} gefunden — das Muster passt "
                                f"vermutlich nicht mehr")

    def test_jede_geholte_beilage_ist_abrufbar(self):
        if not _RECORDER.is_file():
            self.skipTest("tv-receiver nicht ausgecheckt")
        fehlend = beilagen_die_der_daemon_holt() - whitelist_des_gateways()
        self.assertEqual(
            fehlend, set(),
            "Der Daemon holt Dateien, die der Gateway nicht herausgibt: "
            f"{sorted(fehlend)}. Der Upload nimmt sie mit (Whitelist "
            "head.*), der Download liefert 404, und der Daemon faellt "
            "STILL auf seinen Default zurueck. Eintragen in "
            "detectModelWhitelist in "
            "tv-receiver/cmd/tv-recorder/main.go.")

    def test_audio_beilage_ist_drin(self):
        """Der konkrete Fall vom 2026-09-08, damit er nicht zurueckkommt."""
        if not _RECORDER.is_file():
            self.skipTest("tv-receiver nicht ausgecheckt")
        self.assertIn("head.audio.json", beilagen_die_der_daemon_holt(),
                      "der Daemon muss die Audio-Beilage holen")
        self.assertIn("head.audio.json", whitelist_des_gateways(),
                      "head.audio.json ist die Kopplung fuer die "
                      "Audio-Semantik (O22). Fehlt sie, bekommt ein auf "
                      "der Schwankung trainierter Kopf still den Pegel.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
