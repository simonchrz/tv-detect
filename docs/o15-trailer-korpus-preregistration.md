# O15 — Lernt das Modell die Trailer-Konvention, wenn der Korpus sie ausdrückt?
(Vorab-Registrierung)

**Geschrieben 2026-08-17, bevor eine einzige Korpus-Aufnahme angefasst
wurde.** Die Bedingungen unten stehen fest, bevor die erste Zahl existiert.

## Woher die Frage kommt

Aus §3ao, und der Befund ist ungewöhnlich scharf für dieses Projekt:

```
Luecke zwischen Modellkante und Wahrheit enthaelt einen Trailer
   ENDE  Median -14.4 s   Modell frueher als die Wahrheit: 8 von 8
Luecke enthaelt KEINEN Trailer
   ENDE  Median   0.0 s                                    13 von 29
```

Das Modell beendet Werbeblöcke **vor** dem abschließenden Programmtrailer.
Die geltende Konvention (§3y, 13.08.) sagt: der Trailer gehört zur Werbung.
Simon hat das am 17.08. dreimal ungefragt am Bild bestätigt.

Die naheliegende Erklärung ist keine Modellschwäche, sondern eine Datenlücke:
**der Trainingskorpus drückt die Regel nicht aus.** §3am hat über 150
Aufnahmen gemessen, dass Modell und Korpus-Label im Median exakt
aufeinanderliegen — beide in der alten Lesart. Das Modell hat nie ein Label
gesehen, das den Trailer einschließt.

## Warum das eine Registrierung braucht

Weil die Alternative naheliegt und billiger klingt: eine Nachlauf-Regel, die
Blockenden über einen erkannten Trailer schiebt. Genau diese Bauart —
Heuristik hinter dem Decoder — ist in O13 und O14 gescheitert, und in beiden
Fällen war der Grund, dass ich sie an einem Satz gemessen habe, der die
interessanten Fälle nicht enthielt (§3ak).

Die Frage hier ist eine andere und beantwortet sich nur über das Training:
kann das Modell die Regel **lernen**, wenn sie in den Labels steht?

## Der Eingriff — fixiert

**Umfang:** die Blockenden von N Korpus-Aufnahmen auf die geltende
Konvention nachziehen, per `agent-review.py` (grob → fein), also über
Bild-Urteile und nicht über eine Trailer-Heuristik. Nur **Enden**, nur wo die
Ableitung bestimmt ist, nur wo die Abweichung über 5 s liegt — dieselben
Schranken wie bei `golden-korrigieren.py`.

**Nicht angefasst:** Blockstarts (dort zeigt §3ao kein Muster), der
Golden-Satz (der folgt der Konvention seit dem 17.08. bereits), und der
Decoder.

**N = 40**, ausgewählt reihum über die Sender (`_reihum`), aus den Aufnahmen
mit lokaler Quelle und menschlichem Label. Warum 40: unter §3am tragen etwa
zwei Drittel der Blöcke einen Rand-Trailer, das ergibt grob 50–60 geänderte
Enden — genug, damit ein Kopf sie sehen kann, und wenig genug, dass ein
Fehlschlag den Korpus nicht dominiert (40 von ~170).

## Bedingung — festgelegt, bevor eine Zahl existiert

Ausgewertet wird **ein** Nachtlauf nach der Umstellung, gegen den
korrigierten Golden-Satz (Label-Epoche ab 2026-08-17), mit dem
Produktions-Decoder `hsmm --hsmm-dur-w 15`.

**ERFÜLLT, wenn beides gilt:**

1. Der **Kantenfehler an Blockenden mit Trailer** sinkt im Median um
   **≥ 5 s** (heute −14.4 s).
2. Der **Golden-IoU** sinkt nicht (≥ heutiger Wert minus Seed-Streuung,
   also ≥ −0.02).

Zusätzlich berichtet, nicht Bedingung: Kantenfehler an Enden ohne Trailer
(darf sich nicht verschlechtern), Starts, und wie viele Enden überhaupt
geändert wurden.

⚠️ Bedingung 2 steht da, weil ein Kopf, der Trailer als Werbung liest, im
Zweifel auch **echte Sendung** als Werbung liest — und Sendung wegschneiden
ist irreversibel, ein stehengebliebener Spot nicht (dieselbe Asymmetrie wie
beim End-Snap-Guard, Memory `bumper_end_nn_guard`).

## Vorhersage

Bedingung 1 **erfüllt**, Bedingung 2 **knapp erfüllt**. Begründung: 40 von
170 Aufnahmen sind ein Viertel des Korpus, und das Signal (Trailer sieht
anders aus als Sendung) ist visuell stark. Aber der Korpus bleibt zu drei
Vierteln in der alten Lesart, also lernt der Kopf einen Widerspruch — das
könnte für Bedingung 1 zu wenig sein.

⚠️ Falls der Widerspruch das Ergebnis erklärt, ist die Antwort NICHT „mehr
Aufnahmen nachziehen, bis es klappt". Dann ist die registrierte Frage
beantwortet (Teil-Umstellung reicht nicht) und die nächste ist eine neue.

## Was jeder Ausgang bedeutet — jetzt entschieden

**ERFÜLLT** → der Rest des Korpus wird nachgezogen, und die Konvention
bekommt einen Platz, an dem sie prüfbar steht (nicht nur im Ledger — die
Lehre aus dem Gewinnspiel-Rückzieher, §3am).

**NICHT ERFÜLLT bei Bedingung 1** → Labels allein reichen nicht; die Frage
wird zu „braucht der Trailer ein eigenes Signal" (OCR liegt gebaut da,
§3ab) — und die wäre wieder eigens zu registrieren.

**NICHT ERFÜLLT bei Bedingung 2** → sofort zurück. Die 40 Aufnahmen werden
aus der Sicherung wiederhergestellt. Sendung wegschneiden ist der teuerste
Fehler, den dieses System machen kann.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1) oder den
Golden-Satz anzufassen (L2). Und die Enden, an denen es nicht klappt, werden
nicht per Frame-Review nachträglich zu Label-Fehlern erklärt.
