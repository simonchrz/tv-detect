#!/usr/bin/env python3
"""Der Nur-Dump-Modus darf NICHTS schreiben.

Er existiert, damit das Fehlerbudget frische Dumps bekommt, ohne die
Bloecke von 98 menschlich reviewten Aufnahmen zu ersetzen. Faellt der
Schutz weg, merkt es niemand: der Lauf sieht genauso aus, nur sind
hinterher die Labels ueberschrieben. Deshalb steht er unter Test.
"""
import re
import unittest
from pathlib import Path

HIER = Path(__file__).resolve().parent
DAEMON = (HIER.parent / "daemon" / "tv-thumbs-daemon.py").read_text()
SKRIPT = (HIER / "dumps-erneuern.py").read_text()


def rumpf():
    """Der Text von process_detect bis zur naechsten Funktion."""
    i = DAEMON.index("def process_detect(")
    m = re.search(r"\ndef ", DAEMON[i + 10:])
    return DAEMON[i: i + 10 + (m.start() if m else len(DAEMON))]


class NurDump(unittest.TestCase):
    def test_modus_existiert(self):
        self.assertIn("def process_detect(uuid, nur_dump=False, dump_ziel=None):",
                      DAEMON, "die Signatur traegt den Modus")

    def test_kehrt_vor_dem_upload_um(self):
        r = rumpf()
        aus = r.index("if nur_dump:")
        hoch = r.index("cutlist-uploaded")
        self.assertLess(aus, hoch,
                        "der Nur-Dump-Lauf muss VOR dem Upload enden")

    def test_kehrt_vor_der_erfolgsmeldung_um(self):
        r = rumpf()
        self.assertLess(r.index("if nur_dump:"),
                        r.index("_record_detect_success"),
                        "ein Messlauf darf die Zaehler nicht anfassen")

    def test_kehrt_vor_der_whisper_verfeinerung_um(self):
        # Die Verfeinerung schreibt die Cutlist um. Sie kann den Dump nicht
        # mehr beeinflussen (den hat das Binary waehrend des Laufs
        # geschrieben), kostet aber ~50 s je Aufnahme.
        r = rumpf()
        self.assertLess(r.index("if nur_dump:"),
                        r.index("_maybe_whisper_refine"),
                        "die Verfeinerung gehoert hinter den Ausstieg")

    def test_fehlschlag_wird_nicht_angerechnet(self):
        r = rumpf()
        i = r.index("_record_detect_failure(uuid, force=corrupt)")
        vorher = r[max(0, i - 300):i]
        self.assertIn("if not nur_dump:", vorher,
                      "ein Messlauf darf keine Aufnahme auf aufgegeben setzen")

    def test_eigenes_zielverzeichnis(self):
        r = rumpf()
        self.assertIn("Path(dump_ziel) if dump_ziel", r,
                      "der alte Dump muss stehenbleiben koennen — alt gegen "
                      "neu IST die Messung")

    def test_kampagne_ruft_nie_ohne_den_modus(self):
        for treffer in re.finditer(r"process_detect\(([^)]*)\)", SKRIPT):
            self.assertIn("nur_dump=True", treffer.group(1),
                          "die Kampagne darf process_detect nie ohne "
                          "nur_dump=True aufrufen")

    def test_kopf_wird_zwischendurch_geprueft(self):
        # Sechs Stunden Laufzeit ueberspannen die Ausbildung um 03:30.
        self.assertIn("if jetzt != abdruck:", SKRIPT)
        self.assertIn("ABBRUCH", SKRIPT,
                      "bei Kopfwechsel abbrechen, nicht weiterlaufen — ein "
                      "gemischter Messsatz sieht sauber aus und ist es nicht")

    def test_umgebung_kommt_vom_daemon(self):
        # ⚠️ 2026-09-08 selbst hineingelaufen: derselbe Code und dieselbe
        # Kommandozeile ergaben trotzdem einen anderen Lauf, weil die
        # Konstanten beim IMPORT aus der Umgebung gelesen werden. Ohne
        # diese Uebernahme lief die Messung mit SPEAKER_ENABLE=0 und 12
        # statt 4 Stuecken.
        self.assertIn("PlistBuddy", SKRIPT,
                      "die Umgebung muss aus dem launchd-Eintrag kommen, "
                      "nicht aus einer abgeschriebenen Liste")
        # Im RUMPF von main() nachsehen, nicht im ganzen Text — sonst
        # trifft der Vergleich die Definitionen und ist immer wahr.
        m = SKRIPT[SKRIPT.index("def main():"):]
        self.assertLess(m.index("umgebung_vom_daemon()"),
                        m.index("daemon_laden()"),
                        "erst die Umgebung setzen, dann importieren")
        self.assertIn("raise SystemExit(3)", SKRIPT,
                      "unlesbare Umgebung muss abbrechen, nicht raten")

    def test_haelt_netzausfaelle_aus(self):
        # 2026-09-09, 08:26: drei Minuten ohne Internet. Der erste
        # Entwurf hakte in der Zeit 54 Aufnahmen im Fuenf-Sekunden-Takt
        # als Fehlschlag ab. Ein Lauf ueber Stunden muss Aussetzer
        # aushalten, sonst misst die Netzqualitaet mit.
        self.assertIn("gateway_wartet", SKRIPT)
        self.assertIn("healthz", SKRIPT,
                      "Erreichbarkeit am Gesundheitsendpunkt pruefen")
        m = SKRIPT[SKRIPT.index("def main():"):]
        self.assertIn("for versuch in range(1, 4):", m,
                      "dieselbe Aufnahme mehrfach versuchen")

    def test_kopfwechsel_raeumt_statt_zu_ergaenzen(self):
        # Ein Verzeichnis mit Dumps aus ZWEI Modellen sieht vollstaendig
        # aus und ist eine Mischung. Simons Entscheid 2026-09-09: der
        # Messsatz wandert mit dem Kopf mit, also muss der Wechsel
        # raeumen und nicht ergaenzen.
        self.assertIn('kopf_datei = ziel / ".kopf"', SKRIPT,
                      "der Fingerabdruck gehoert NEBEN die Dumps")
        i = SKRIPT.index("if frueher and frueher != h0:")
        blk = SKRIPT[i:i + 900]
        self.assertIn("f.unlink()", blk, "alte Dumps muessen weg")
        # ⚠️ Aber NIE im Trockenlauf. Der erste Entwurf raeumte auch dort,
        # weil der Ausstieg fuer --trocken weiter unten steht.
        self.assertIn("if a.trocken:", blk,
                      "ein Probelauf darf den Satz nicht loeschen")

    def test_nightly_fuehrt_den_messsatz_nach(self):
        sh = (HIER.parent / "daemon" / "tv-train-head.sh").read_text()
        self.assertIn("dumps-erneuern.py", sh,
                      "der naechtliche Lauf muss den Messsatz nachfuehren")
        self.assertIn("--dumps", sh,
                      "das Budget muss die frischen Dumps lesen, nicht das "
                      "Standardverzeichnis mit dem alten Kopf")
        self.assertLess(sh.index("fehlerbudget.py"), sh.index("dumps-erneuern.py"),
                        "erst messen (Stand von gestern), dann nachfuehren "
                        "(Stand fuer morgen) — sonst blockiert die Kampagne "
                        "die Ausbildung fuenf Stunden")

    def test_kampagne_holt_den_kopf_bevor_sie_ihn_liest(self):
        # ⚠️ 2026-09-10, erste Nacht: die Kampagne las den Abdruck aus
        # dem Modell-Cache, und der erneuert sich erst beim naechsten
        # Detect. Der Pi trug 04:16 den neuen Kopf, der Cache 06:17 noch
        # den von gestern — "nichts zu tun" war die Folge und falsch.
        self.assertIn("def modelle_frisch", SKRIPT)
        m = SKRIPT[SKRIPT.index("def main():"):]
        self.assertLess(m.index("modelle_frisch"), m.index("kopf_abdruck()"),
                        "erst holen, dann den Abdruck nehmen")
        self.assertIn("if a.trocken:", m[:m.index("kopf_abdruck()")],
                      "ein Trockenlauf holt nichts und muss das sagen")

    def test_budget_prueft_die_herkunft(self):
        q = (HIER / "fehlerbudget.py").read_text()
        self.assertIn('"dump_kopf"', q,
                      "die Trendzeile muss nennen, welcher Kopf die Dumps "
                      "erzeugt hat — nicht nur welcher deployt war")
        self.assertIn("ANDERER DUMP-SATZ", q,
                      "ein Wechsel der Messgrundlage darf nicht als "
                      "Modellbewegung durchgehen")
        self.assertIn("Kopf gewechselt", q)
        self.assertIn("NENNT SEINE HERKUNFT NICHT", q,
                      "alte Zeilen ohne Herkunft muessen als solche "
                      "kenntlich sein statt stillschweigend zu vergleichen")

    def test_fortsetzbar(self):
        self.assertIn('if not (ziel / f"{u}.json").is_file()', SKRIPT,
                      "ein Abbruch nach 4 h darf nicht alles wiederholen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
