#!/usr/bin/env python3
"""Der Modell-Cache muss am INHALT erneuern, nicht an der Groesse.

Am 2026-09-09 gefunden: `http_download` erneuerte nur bei abweichender
Content-Length. Der MLP-Kopf ist 493096 Byte gross UND groessenstabil --
jede Nacht neuer Inhalt, identische Zahl. Der Mac lief zwoelf Tage auf
einem Kopf vom 28.08., waehrend der Pi den vom 09.09. auslieferte.

Besonders teuer wurde es durch die Audio-Beilage: die kam am 08.09. neu
dazu (Erstabruf laedt immer), sagte "dynamik: true" -- und der Kopf
daneben war von vor O22, also auf den Pegel trainiert. Genau die stille
Fehlpaarung, gegen die die Beilage gebaut wurde.
"""
import re
import unittest
from pathlib import Path

DAEMON = (Path(__file__).resolve().parent.parent
          / "daemon" / "tv-thumbs-daemon.py").read_text()


def rumpf():
    i = DAEMON.index("def http_download(")
    m = re.search(r"\ndef ", DAEMON[i + 10:])
    return DAEMON[i: i + 10 + (m.start() if m else len(DAEMON))]


class ModellFrische(unittest.TestCase):
    def test_prueft_last_modified(self):
        self.assertIn("Last-Modified", rumpf(),
                      "die Frische haengt am Zeitstempel, nicht an der Groesse")

    def test_groesse_ist_nur_noch_rueckfall(self):
        r = rumpf()
        i_lm = r.index("remote_lm")
        i_sz = r.index("remote_size == dest_path.stat().st_size")
        self.assertLess(i_lm, i_sz,
                        "erst der Zeitstempel, die Groesse nur ohne Header")
        self.assertIn("elif", r[i_lm:i_sz],
                      "die Groessenpruefung darf nur greifen, wenn KEIN "
                      "Last-Modified kam — sonst ist der alte Fehler zurueck")

    def test_stand_wird_gespeichert(self):
        r = rumpf()
        self.assertIn(".stand", r, "der gesehene Stand muss neben die Datei")
        self.assertIn("stand_pfad.write_text", r)

    def test_ohne_header_keine_falsche_zusage(self):
        # Verschwindet der Header spaeter, darf kein alter .stand
        # weiter "unveraendert" behaupten.
        self.assertIn("stand_pfad.unlink()", rumpf())

    def test_erstabruf_laedt_immer(self):
        # Kein dest_path -> kein HEAD, direkt laden. Sonst haette eine
        # fehlende Datei nie einen Stand.
        r = rumpf()
        self.assertIn("if dest_path.exists():", r)

    def test_die_lehre_steht_dabei(self):
        r = rumpf()
        self.assertIn("493096", r,
                      "die Zahl gehoert in den Text — sie ist der Grund, "
                      "warum die Groessenpruefung schwieg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
