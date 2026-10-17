Die Altersrampe der Produktion (nie gemessen, Teil der Abbild-Lücke) hilft im Proxy der jüngsten Ära leicht und kostet die alte Ära sichtbar; der Messsatz stammt ganz aus der alten Ära (Alter 88–158 d), und ab ~26.10. beginnt die Rampe dort Trainingsdaten auf 0 zu setzen.

Daten: Trainingsteil von K (opt_lauf.saetze/lade, ohne O25), 527 Aufn. mit ads; Alter aus meta start_ts gegen 17.10.2026.
* Alter Training q0/25/50/75/100: 0/48/92/137/171 d; 65 Aufn. ohne start_ts (tvh-Hash, Produktion: Gewicht 1).
  Mittleres Rampengewicht 0.49, 51 % der Aufn. > 90 d. Entwicklung 88–158 d (Gewicht ~0.25, 96 % > 90 d), Bestätigung 88–158 d.
* Proxy: K-MLP 96/12 Ep. auf Backbone+Logo+Audio (ohne OCR/SigLIP), jede 8. s, Hälften nach sha1(uuid), 2 Richtungen x 2 Seeds,
  Fehler je Aufnahme nach 7-Zeilen-Glättung (~56 s), Schwelle 0.5. Flach -> Rampe:
  - alte Ära (> 90 d, ~100 Eval-Aufn.): Mittel 0.0593/0.0588/0.0502/0.0552 -> 0.0624/0.0622/0.0530/0.0601 (4/4 schlechter),
    Median 0.0285/0.0299/0.0269/0.0316 -> 0.0277/0.0293/0.0278/0.0308 (3/4 besser).
  - junge Ära (<= 90 d, ~160): Mittel 0.0326/0.0316/0.0206/0.0222 -> 0.0332/0.0318/0.0196/0.0205, Median 4/4 leicht besser (-0.0001..-0.0006).
  Zeitdrift ist also real, aber klein; die alte Ära ist ohnehin doppelt so fehlerreich (0.055 gegen 0.025).
* Gestörte Quellen (Sitzungshinweis), geprüft und verworfen als Daten-Hebel: Einzelbild-Glitches (Nachbarn einig, Mitte weicht ab)
  0.20 % der Zeilen, Werbeanteil 0.24 = Grundrate; eingefrorene Bilder 0.17 %, 80 % Werbung (Packshots), Sendungs-Standbilder
  liegen zu 91 % > 120 s vom Block. Kein Label- oder Gewichtsposten.
* dvr-super-rtl-1781112000 (Phantomblock-Fall) ist ein train-Eintrag mit ads=[] (auto-confirm, Logo 0.00 die ganzen 30 min):
  K lädt leere Aufnahmen nicht (lade mit_label), sieht also nie eine logo-lose Sendung ohne Werbung. Kein Arm-Hebel
  (Satzwahl ist dem Arm verboten), aber ein Messgerät-Befund zur Abbild-Lücke und zur Seed-Instabilität auf dem Schiri-Satz.
Erwartung: Δ -0.001..-0.003 auf dem (alten) Entwicklungssatz; ein Gewinn widerlegte die Drift-Lesart.
Wert: beziffert, was die Rampe der Produktion auf der Messsatz-Ära kostet, bevor die 180-d-Kante dort Daten löscht.
Skripte: Scratchpad stoer.py, alter.py, proxy.py, sr.py, frz.py (read-only, nur Summen).

## Abgelehnt

- **ABLEHNEN (Neuheit/Begründung).** Technisch ist der Arm sauber, inhaltlich aber schon verworfen und strukturell nicht messbar.
- **Technik ok.** Der Code läuft. Ich habe ihn mit synthetischen Arrays geprüft: Alter 0 und 90 ergeben 1, 135 ergibt 0.5, 180 ergibt 0, 200 ergibt 0. start_ts fehlt, ist None oder 0, dann Gewicht 1. Ausgabe float32, Länge gleich len(rec), nie negativ. Die Rampe entspricht train-head.py 5146-5153 und 4423. Der Stichtag 1792198800 ist 17.10.2026 03:00 und fest. Es gibt keinen Zufall, kein I/O, kein Netz und keinen globalen Zustand. Labels, ads, which und uuids der Entwicklung/Bestätigung werden nicht benutzt. Es gibt nur gewichte() und kein spalten(), also keinen Train/Serve-Bruch und keine schritt-Frage. Nur der Trainingsteil bekommt gewichte(), und dort ist start_ts in meta vorhanden. Ein Fehler in dieser Richtung ist nicht der Ablehnungsgrund.
- **Rampe schon gemessen.** Die Behauptung 'Rampe nie gemessen' stimmt nicht. laeufe/026-label_bruch_aufnahmen_raus/befund.md (Z. 13-16) hat unter 'Verworfen vor dem Arm' genau diese Rampe gemessen (1 -> 0.5 bis 90 d -> 0 bei 180 d, Stichtag 17.10.). Ergebnis im Proxy: alte Aufnahmen 0.068 -> 0.076, neue 0.0251 -> 0.0255. Er sagt dort ausdrücklich 'die Rampe ist im Proxy ohne Nutzen'. Der neue Befund (4/4 Mittel schlechter in der alten Ära, junge Ära Median -0.0001..-0.0006) wiederholt das mit einem anderen Proxy-Aufbau. Er ist keine neue Information.
- **Messbarkeit schon verneint.** Die Messbarkeit ist ebenfalls schon dokumentiert und verneint. laeufe/016 befund.md (Z. 14-1).
