# O34 — Zieht der Sender-Trenner das Blockende richtig, wenn das Logo statt des NN wacht? (Vorab-Registrierung)

**Geschrieben 2026-10-13, vor dem ersten Datenpunkt auf dem Messsatz.**
Skript: `scripts/o34-trenner-ende.py`.

## Woher die Frage kommt

Bildprüfung von neun Galileo-Folgen (29.09.–12.10.): in 15 von 18 Blöcken
(Block 1 und 2) endete das Maschinenlabel 14–31 s zu früh. Nach dem
joyn-Trailer läuft ein Einzelspot im Rahmen „Jetzt geht's weiter GALILEO“ mit
Countdown, dann der Galileo-Trenner, dann das Studio. In den Signal-Dumps
derselben Folgen: `nn_confs` fällt am alten Ende von ~0.95 auf ~0.1 (das NN
liest den Einzelspot als Sendung), `logo_confs` bleibt bei 0.0–0.2,
`bumper_confs` steht mit 0.99 genau auf dem Trenner. Das NN hat den Fehler aus
Maschinenlabels gelernt, die an derselben Stelle enden (Echo).

**Diese neun Folgen sind der Fundort und dürfen die Frage nicht entscheiden.**
Keine von ihnen liegt im Messsatz (eingefroren 07.09., alle Fundorte ≥ 25.09.).

## Warum das keine Wiedervorlage ist

Trenner-Evidenz wurde am 29.07. als Sackgasse abgelegt (f3ceb21: „soft bumper
boundary evidence, refine snaps on top of hsmm15“) und der End-Snap-Guard am
14.08. als dormant (§3v). Beide Messungen liefen unter Bedingungen, die nicht
mehr gelten:

1. **Konvention:** damals „Trailer = Sendung“, seit 13.08. gehören Rand-Trailer
   zur Werbung; seit 13.10. (Simon) auch der Sender-Trenner selbst.
2. **Der Wächter war das NN** (`snapToBumperGuarded`, Veto bei NN < 0.45). Am
   Fundort liest das NN am echten Trenner 0.00–0.19: der Wächter verwirft genau
   die richtigen Treffer.
3. **Der Maßstab** stammte aus Form-Labels; heute gibt es 98 Aufnahmen mit
   menschlichem Label und Dump.

Geprüft wird eine ANDERE Regel: derselbe Trenner, aber das Logo wacht.

## Die Regel T — fixiert

Ausgangspunkt sind die Blöcke des Produktionsdekoders K (`--decoder hsmm
--hsmm-dur-w 15 --hsmm-inner-show-min 90`, `--replay-signals` auf dem Dump).
Für jedes Blockende E (Sekunden), fps aus dem Dump:

1. Treffer = Frames in (E+2 s, E+45 s] mit `bumper_confs` ≥ 0.95. Kein Treffer
   → Ende bleibt.
2. Erster Treffer f0; die Gruppe läuft weiter, solange der nächste Treffer
   höchstens 2 s nach dem vorigen liegt; p = letzter Frame der Gruppe.
3. Logo-Wächter: Mittel der gültigen (nicht NaN) `logo_confs` in [E, f0) muss
   < 0.35 sein, bei mindestens 25 gültigen Werten. Sonst bleibt das Ende.
4. Neues Ende E' = (p+1)/fps. Liegt E' nicht vor dem Beginn des nächsten
   Blocks, bleibt das Ende (kein Verschmelzen).
5. Enden in den letzten 60 s der Aufnahme bleiben.

Die Regel verlängert nur, verkürzt nie, fasst keine Anfänge an. Kein NN, kein
OCR (19 der Fundort-nahen Dumps tragen keine OCR-Spur; OCR-Werbemarker im
verlängerten Stück wird nur berichtet).

## Messung

Messsatz `messsatz-2026-09-07` (98 Aufnahmen, menschliche Labels über
`backbone-sonde.menschlabels()`), davon die mit Trenner- UND Logo-Spur im Dump
(vor der Messung gezählt: 94; comedy-central 5, disney 5, kabel-eins 9, nick 9,
prosieben 21, rtl 7, rtlzwei 18, sat 4, sixx 7, vox 9). Der Kopf ist im Dump
eingefroren; gemessen wird nur die Regel.

Kantenfehler eines Endes = Abstand zum nächstgelegenen menschlichen Blockende,
vor und nach der Regel jeweils neu bestimmt. Ein Ende zählt als „angefasst mit
Gegenkante“, wenn vor ODER nach der Regel ein menschliches Ende binnen 120 s
liegt.

## Regel der Entscheidung

Auswertbar bei **≥ 15 angefassten Enden mit Gegenkante aus ≥ 8 Aufnahmen**.
Darunter: NICHT AUSWERTBAR, die Regel geht als dritter Schatten in
`kanten-schatten.py` (prospektiv), kein zweiter Satz.

* **R1 Gewinn:** Median(Fehler_vorher − Fehler_nachher) über die angefassten
  Enden ≥ +5 s UND ≥ 70 % der angefassten Enden werden besser.
* **R2 Schaden:** höchstens 10 % der angefassten Enden werden um mehr als 5 s
  schlechter.
* **R3 Blockmaß:** mittlere `block_iou` über alle gemessenen Aufnahmen mit T
  nicht kleiner als mit K.

ERFÜLLT, wenn R1, R2 und R3 halten. Berichtet, nicht entscheidend: Aufteilung
nach Kanal, golden gegen nicht-golden, Anteil mit OCR-Werbemarker.

**Konsequenz bei ERFÜLLT:** Vorschlag an Simon, T als Option im hsmm-Pfad des
Go-Binaries zu bauen (mit Paritäts-Fixture, L5) und in Detect UND
`EVAL_DECODER` gleichzeitig zu setzen. Ohne Simons OK keine Produktionsänderung.
**Bei VERFEHLT:** Friedhof; der Fundort bleibt ein Label-Thema (Signal-Probe
als Meldung, nicht als Regel).

```regel
{"id": "O34", "frage": "Zieht der Sender-Trenner das Blockende richtig, wenn das Logo statt des NN wacht?",
 "nicht_in_serienabschluss": true,
 "name": "O34 Trenner-Ende mit Logo-Waechter", "art": "offline",
 "skript": "scripts/o34-trenner-ende.py",
 "auswertbar": ">= 15 angefasste Enden mit Gegenkante aus >= 8 Aufnahmen",
 "r1": "median(Fehler_K - Fehler_T) >= 5 s und >= 70 % der angefassten Enden besser",
 "r2": "<= 10 % der angefassten Enden um > 5 s schlechter",
 "r3": "mean(block_iou_T) >= mean(block_iou_K)",
 "parameter": {"schwelle": 0.95, "fenster_s": [2, 45], "gruppe_luecke_s": 2, "logo_max": 0.35, "logo_min_werte": 25, "rand_s": 60}}
```
