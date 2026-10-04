> **ABGESCHLOSSEN 2026-10-04 — VERFEHLT (R1 in beiden Armen).**
>
> | Arm | kipp je Seed | Σkipp | ≤ K | IoU Median | ΔIoU Median (min) | Verluste > 0.10 |
> |---|---|---|---|---|---|---|
> | K | 1 3 2 3 2 | 11 | — | 0.9158 | — | — |
> | I60 | 1 3 1 4 2 | 11 | 4/5 | 0.9173 | +0.0017 (−0.0001) | 0 |
> | I90 | 1 4 3 2 1 | 11 | 3/5 | 0.9194 | +0.0038 (+0.0016) | 0 |
>
> R2 und R3 halten in beiden Armen, R1 nicht: Die Kippzahl sinkt gar nicht.
> Konsequenz laut Regel: keine Produktionsänderung, die Option bleibt aus.
> Log `~/Library/Logs/o32-lauf.log`, Ergebnis `~/.cache/tvd-train-archive/o32-ergebnis.json`.
>
> Deutung: Mit beiden Gate-Fixes (994331a, ee713ae) kippen nur noch 1–4 von
> 147 Aufnahmen je simulierter Nacht; Sendungsinseln sind daran nicht
> beteiligt. Nebenbefund, NICHT die registrierte Frage: I90 hebt die IoU in
> allen 5 Seeds (+0.0016 bis +0.0059) ohne einen einzigen Einzelverlust. Wer
> das nutzen will, braucht eine eigene, vorab registrierte Qualitätsfrage auf
> Daten, die hier nicht ausgewertet wurden.

# O32 — Verhindert eine Mindestlänge für Sendung zwischen zwei Werbeblöcken das Teilen von Blöcken, ohne Qualität zu kosten? (Vorab-Registrierung)

**Geschrieben 2026-10-04, vor dem ersten Behandlungs-Datenpunkt.** Rein
dekoderseitig, offline, gebaut wie O31: zwei simulierte Nächte je Seed auf
denselben Zeilen, Spalten und derselben Architektur (Produktionszustand MLP7).
Skript: `scripts/o32-inselsperre.py`. Dekoder-Option:
`--hsmm-inner-show-min` (`HSMMOpts.InnerShowMinS`, 0 = aus, byte-identisch).

## Woher die Frage kommt

Nacht 04.10.: Das Gate lehnte ab (netto −5), schlechteste Aufnahme „Hot oder
Schrott“ (0.997 → 0.729). Die Untersuchung ergab drei Dinge:

1. Mitten im ersten Werbeblock liegt ein ~40 s Spot, den das Modell halb für
   Sendung hält. Ein Dekoder, der dort eine Sendungsinsel erlaubt, teilt den
   Block; `block_iou` zählt dann nur das größere Stück.
2. Diese Aufnahme hatte keine Decode-Spur — das Gate maß sie mit Schwelle +
   Laufgruppen statt mit dem HSMM. Behoben in ee713ae (`_replay_ohne_spur`).
3. Das Gate scorte den Champion mit der SigLIP-Projektion des Kandidaten
   (28/64 Vorzeichen gedreht zwischen 02. und 03.10.). Behoben in 994331a.

Offen bleibt Punkt 1 für den HSMM selbst: Er erlaubt Sendungsabschnitte ab
30 s (`hsmmShowMinS`). Gegenprobe an 736 menschlich gelabelten Aufnahmen: von
738 Sendungslücken zwischen zwei Werbeblöcken sind 2 < 30 s, 4 < 60 s, 4 < 90 s,
7 < 120 s; die vier kurzen (19–49 s) sehen selbst nach Labelfehlern aus.

Ein erster Messansatz (archivierte Nachtköpfe) war ungültig: Die sind
`--final-on-all`, also auch auf test trainiert. Deshalb jetzt O31-Bauart.

## Behandlung

Sendung zwischen zwei Werbeblöcken muss mindestens W Sekunden lang sein;
Sendung am Aufnahmeanfang/-ende behält 30 s.

| Arm | W |
|---|---|
| K | aus (Gate-Dekoder: hsmm, DurW 15) |
| I60 | 60 |
| I90 | 90 |

Je Seed s (5 Seeds): Champion trainiert OHNE die jüngsten 10 % der
train-Aufnahmen (Seed 1000+s), Kandidat auf allen (Seed s). Beide laufen auf
ALLEN test-Aufnahmen mit Archiv-Labels durch den Gate-Dekoder
(`_replay_blocks`, ohne Spur `_replay_ohne_spur`), Maß `block_iou`.

* **kipp** = Aufnahmen mit |IoU_Kandidat − IoU_Champion| > 0.1 (= Kippzahl des Gates)
* **IoU** = Mittel des Kandidaten über die test-Aufnahmen

## Rauschen, VOR der Behandlung gemessen

Arm K allein (`~/Library/Logs/o32-vormessung.log`): kipp 1 / 3 / 2 / 3 / 2, **Summe 11** (147
test-Aufnahmen, 111 mit Decode-Spur); IoU Kandidat 0.9158 / 0.9140 / 0.9151 /
0.9162 / 0.9172, **Median 0.9158, sd 0.0012**.

Das ist viel weniger als die Kippzahl des Gates in den Nächten 29.09.–04.10.
(4–18 je Nacht). Der Großteil des nächtlichen Hin und Her kam aus den beiden
Gate-Fehlern (fremde Projektion, Schwellen-Rückfall), nicht aus dem Dekoder.
Mit Σkipp_K = 11 ist R1 grob: 0.70 · 11 = 7.7, ein Arm muss also auf höchstens
7 kommen. Die Regel bleibt trotzdem unverändert.

## Regel

Je Arm I, gepaart nach Seed gegen K:
* **R1 Stabilität:** Σkipp_I ≤ 0.70 · Σkipp_K UND kipp_I ≤ kipp_K in ≥ 4 von 5 Seeds.
* **R2 Qualität:** Median(IoU_I − IoU_K) ≥ −0.002 UND kein Seed unter −0.005.
* **R3 Einzelschaden:** Über alle 5 Seeds höchstens 2 Fälle, in denen eine
  Aufnahme beim Kandidaten mehr als 0.10 gegen K verliert — ausgenommen
  Aufnahmen, deren Label selbst eine Sendungslücke < W zwischen zwei Blöcken
  enthält (einzeln aufgeführt und Simon zur Prüfung vorgelegt, nicht gezählt).

Ein Arm ist ERFÜLLT, wenn R1, R2 und R3 halten. Erfüllen beide, gilt der mit
der kleineren Σkipp; liegen sie höchstens 2 auseinander, gilt I60 (kleinerer
Eingriff).

**Konsequenz bei ERFÜLLT:** Vorschlag an Simon, die Option in Detect
(`tv-thumbs-daemon.py`) UND in `EVAL_DECODER` (`train-head.py`) gleichzeitig zu
setzen — nie nur eines von beiden. Golden-/Per-Rec-Zahlen vor und nach dem
Wechsel sind dann nicht vergleichbar (`decoder`-Feld in golden-trend.jsonl).
Ohne Simons OK keine Produktionsänderung.
**Bei VERFEHLT:** Option bleibt im Code (0 = aus), keine Änderung.

```regel
{"id": "O32", "frage": "Verhindert eine Mindestlaenge fuer Sendung zwischen zwei Werbebloecken das Teilen von Bloecken, ohne Qualitaet zu kosten?",
 "nicht_in_serienabschluss": true,
 "name": "O32 Inselsperre", "art": "offline", "skript": "scripts/o32-inselsperre.py",
 "r1": "sum(kipp_I) <= 0.70 * sum(kipp_K) und kipp_I <= kipp_K in >= 4/5 Seeds",
 "r2": "median(IoU_I - IoU_K) >= -0.002 und min >= -0.005",
 "r3": "<= 2 Verluste > 0.10 ueber alle Seeds, ausser Label mit innerer Luecke < W",
 "seeds": 5, "arme": {"K": 0, "I60": 60, "I90": 90}}
```
