#!/usr/bin/env python3
"""Die beiden Arme einer Tagesserie duerfen sich in GENAU einer Sache unterscheiden.

Gefunden 2026-09-13/14. Zwei O18-Serien waren ungueltig, weil sich die Arme
`--output` und `--train-archive` teilten:

* Der Label-Hygiene-Lehrer wird aus dem `--output`-PFAD geladen, und am
  Ende seines Laufs schreibt ein Arm genau dorthin seinen Kopf. Arm 2 lernte
  also mit dem Modell von Arm 1 als Lehrer — im Log sichtbar als 1282 gegen
  1301 Spalten und als unterschiedlich viele verworfene Frames.
* Dasselbe gilt fuers Archiv: Arm 1 schreibt hinein, Arm 2 liest es.

Dazu die Kommentar-Falle: ein `#` zwischen zwei fortgesetzten Zeilen eines
Aufrufs verschluckt ALLE folgenden Schalter. `bash -n` findet das nicht.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKRIPT = REPO / "daemon/tv-tagesserie.sh"
QUELL = SKRIPT.read_text()


class ArmeSindGetrennt(unittest.TestCase):
    def test_eigenes_ausgabeverzeichnis_je_arm(self):
        self.assertIn('--output "$D/out/head.bin"', QUELL,
                      "die Arme teilen sich den Ausgabepfad — Arm 2 lernt "
                      "dann mit dem Kopf von Arm 1 als Hygiene-Lehrer")

    def test_eigene_archiv_kopie_je_arm(self):
        self.assertIn('--train-archive "$D/archive"', QUELL)
        self.assertIn('cp -R "$ECHT" "$D/archive"', QUELL)

    def test_beide_arme_bekommen_denselben_lehrer(self):
        self.assertIn('cp "$LEHRER/head.bin" "$D/out/head.bin"', QUELL,
                      "ohne eingefrorenen Lehrer laeuft die Serie ohne "
                      "Label-Hygiene und damit neben der Produktion her")

    def test_zusatzschalter_je_arm(self):
        self.assertIn('ZUSATZ_MIT="${3:-}"', QUELL)
        self.assertIn('ZUSATZ_OHNE="${4:-}"', QUELL)
        # ⚠️ bash 3.2 (macOS) + set -u: ein leeres Array nackt zu
        # expandieren bricht den Lauf mit "unbound variable".
        self.assertIn('${ZUSATZ_ARR[@]+"${ZUSATZ_ARR[@]}"}', QUELL)

    def test_der_lauf_sagt_welcher_arm_was_bekam(self):
        self.assertIn("Zusatz-Schalter:", QUELL,
                      "eine stille Asymmetrie zwischen den Armen ist der "
                      "Fehler, den die Serie messen soll, nicht der, den "
                      "sie machen darf")

    def test_produktions_paritaet(self):
        # Im Nightly seit O22 (09-08) bzw. O17 scharf.
        self.assertIn("--audio-dynamik", QUELL)
        self.assertIn("--herkunft-belegt", QUELL)


class KeinKommentarInDerFortsetzung(unittest.TestCase):
    """Allgemeine Pruefung ueber alle Skripte: `#` nach `\\` frisst den Rest."""

    def test_alle_daemon_skripte(self):
        for pfad in sorted((REPO / "daemon").glob("*.sh")):
            zeilen = pfad.read_text().splitlines()
            for i, z in enumerate(zeilen[:-1]):
                if not z.rstrip().endswith("\\"):
                    continue
                # Ein fortgesetzter KOMMENTAR ist harmlos — die naechste
                # Zeile ist dann ohnehin schon Kommentar. Nur ein
                # fortgesetzter BEFEHL frisst seine Schalter.
                if z.lstrip().startswith("#"):
                    continue
                naechste = zeilen[i + 1]
                if re.match(r"\s*#", naechste):
                    self.fail(
                        f"{pfad.name}:{i + 2}: Kommentar direkt nach einer "
                        f"fortgesetzten Zeile — der Backslash klebt die "
                        f"Zeilen zusammen, das '#' verschluckt ALLE "
                        f"folgenden Schalter, und `bash -n` winkt es durch.\n"
                        f"    {z.strip()}\n    {naechste.strip()}")


if __name__ == "__main__":
    unittest.main()
