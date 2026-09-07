# O19 — Verbessern Spot-Anker als Übergangs-Evidenz die Blockkanten?
(Vorab-Registrierung)

> **ABGESCHLOSSEN 2026-09-07 — REGEL NICHT ERFÜLLT.** Prüfsatz mit dem
> vom Gleichstands-Tiebreak vorgeschriebenen Arm `audio w=0.5`: Median
> ΔIoU **+0.0000** gegen die Schwelle +0.005, 1 besser, 0 schlechter. Die
> vorab festgelegte Konsequenz gilt: **`spot_lp_w` bleibt 0.**
>
> **Die Regel war schlecht gebaut, und das wird hier nicht repariert.**
> Der Median über ALLE Aufnahmen kann nicht über null steigen, weil die
> weit überwiegende Mehrheit sich gar nicht ändert — im besten
> Stimmsatz-Arm ändern sich 9 von 47. Der Median war die falsche
> Statistik; das hätte vor dem Schreiben auffallen müssen. Die Regel
> nachträglich auf „Median der GEÄNDERTEN" zu drehen hieße, die Schwelle
> an die Zahl anzupassen, und dafür existiert diese Datei nicht. Ein
> Nachfolger braucht eine eigene Registrierung auf eigenen Aufnahmen.
>
> Der inhaltliche Befund steht unten unter „Ergebnis" und ist deutlicher
> als die Regel: die Bild-Anker ziehen Kanten nach INNEN.

**Geschrieben 2026-09-07, bevor ein einziger A/B-Lauf gerechnet wurde.**
Arme, Aufteilung und Schwelle stehen fest, bevor die erste Zahl existiert.

## Warum das KEINE Nachtserie ist

Die O-Serien-Maschinerie (`audit-preregistration.py`) kennt nur
`naechte` und `tagesserie`; beide lesen gepaarte **Fits** aus dem
Trainings-Archiv. Hier wird nichts trainiert. Das Modell ist eingefroren,
die NN-Ausgaben stehen fest in den Signal-Dumps, und variiert wird
ausschließlich ein **Dekoder-Parameter**. Diese Registrierung wird
deshalb bewusst **nicht** in `serien-abschluss.json` eingetragen — ein
Wächter, der jede Nacht fehlende Zeilen meldet, wird überlesen, und genau
das ist am 2026-09-07 schon einmal passiert (Ledger, O18).

Die Disziplin gilt trotzdem: Schwelle vor der Zahl.

## Woher die Frage kommt

`--spot-lp-w` ist seit 2026-09-07 gebaut (`internal/blocks/spot_lp.go`),
steht auf 0 und ist damit wirkungslos — der Dekoder ist byte-identisch.
Er wurde nie mit echten Ankern gemessen. Inzwischen gibt es zwei
Ankerquellen: die Audio-Fingerprints des Pi und die Bild-Anker aus
`scripts/wiederholung.py`.

**Die ehrliche Erwartung ist Skepsis.** Am selben Tag wurde gemessen, dass
Modellkante und Menschenkante an harten Ankern nicht zu unterscheiden
sind (p = 0.82 / 0.76 auf 206 Blöcken). Wo nichts systematisch daneben
liegt, hat ein ausdehnender Anker wenig zu holen. Diese Registrierung
existiert, damit dieses Ergebnis nicht nachträglich zurechtgelegt wird —
in keine der beiden Richtungen.

## Was gemessen wird

Für jede Aufnahme wird `--replay-signals` zweimal gefahren, einmal ohne
und einmal mit Anker-Gewicht, und `block_iou` gegen das **Menschenlabel**
gerechnet (die Definition aus `eval_production_cutlists.py`, dieselbe wie
in `train-head.py`).

## Die Aufnahmen — und warum es nicht der versiegelte Satz ist

Gefordert sind vier Dinge gleichzeitig: menschliches Label
(`label_herkunft.py`), Bild-Anker, Audio-Anker und ein Signal-Dump.

| Satz | gesamt | mit Anker + Dump | davon menschlich gelabelt |
|---|---|---|---|
| versiegelt | 38 | 18 | **0** |
| test | 142 | 47 | 23 |

**Der versiegelte Satz fällt aus** — keine einzige seiner Aufnahmen hat
ein menschliches Label neben Anker und Dump. Gegen ein maschinelles Label
zu messen hieße, den Dekoder gegen seine eigene frühere Ausgabe zu
messen; jede Änderung sähe wie ein Rückschritt aus. Das ist derselbe
Fehler, der ohne die Herkunftsprüfung schon einmal 205 von 408
Vergleichspaaren wertlos gemacht hat.

Weil das Modell hier eingefroren ist, ist die train/test-Grenze für diese
Frage nicht bindend: kein Label fließt in Gewichte. Genommen werden
deshalb **alle 98 Aufnahmen**, die die vier Bedingungen erfüllen
(76 train, 22 test).

Aufgeteilt wird nach `sha1(uuid)` in zwei Hälften:

* **Stimmsatz** (gerade) — hier darf frei gesucht werden.
* **Prüfsatz** (ungerade) — wird EINMAL angefasst, mit genau EINER
  Konfiguration.

## Arme

Stimmsatz, frei kombinierbar:

* Ankerquelle: `audio` | `bild` | `beide`
* Gewicht: 0.5 | 1 | 2 | 4 | 8

Aus dem Stimmsatz geht **genau eine** Konfiguration in den Prüfsatz: die
mit dem höchsten Median-ΔIoU. Bei Gleichstand das kleinere Gewicht, dann
`audio` vor `bild` vor `beide`.

## Bedingung

```regel
{
  "id": "O19",
  "frage": "Verbessern Spot-Anker als Uebergangs-Evidenz die Blockkanten?",
  "art": "dekoder-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "block_iou gegen Menschenlabel, Definition aus eval_production_cutlists.py",
  "aufteilung": "sha1(uuid) gerade = Stimmsatz, ungerade = Pruefsatz",
  "pruefsatz_wird_einmal_angefasst": true,
  "bedingungen": {
    "median_delta_iou_mindestens": 0.005,
    "verhaeltnis_besser_zu_schlechter_mindestens": 2.0,
    "kein_einzelverlust_groesser_als": 0.10
  },
  "konsequenz_bei_erfuellt": "spot_lp_w in der Detect-Config vorschlagen, Entscheidung beim Menschen",
  "konsequenz_bei_verfehlt": "spot_lp_w bleibt 0; das Ergebnis wird als Negativbefund verbucht"
}
```

Alle drei Bedingungen müssen gelten. Die dritte ist die wichtigste: der
Mechanismus darf einzelne Aufnahmen nicht zerreißen, auch wenn der Median
gewinnt. Ein Anker mit 92 % Präzision heißt, dass jeder zwölfte falsch
ist, und ein falscher Anker zieht eine Kante an die falsche Stelle.

## Nebenbeobachtung, ohne Entscheidungsgewicht

`spot_lp.go` behauptet im Kommentar, der Block werde durch den Bonus
„ausgedehnt, nie beschnitten". Der erste Rauchtest widerspricht dem:
bei Gewicht 8 rückte ein Blockstart von 832 auf **836**, also nach innen,
und damit näher an das Menschenlabel bei 839. Ein Übergangs-Bonus an
Sekunde t zieht eine Grenze aus BEIDEN Richtungen an. Der A/B zählt
deshalb mit, wie viele Kanten nach außen und wie viele nach innen wandern.
Das ändert nichts an der Entscheidungsregel oben.


---

## Ergebnis (2026-09-07, nach dem Lauf eingetragen)

### Stimmsatz, 47 Aufnahmen

| Quelle | Gewicht | besser | schlechter | größter Verlust | Kanten außen/innen |
|---|---|---|---|---|---|
| audio | 0.5 | 5 | 0 | 0.000 | 5 / 1 |
| audio | 1 | 6 | 0 | 0.000 | 8 / 1 |
| audio | 2 | 7 | 0 | 0.001 | 10 / 2 |
| audio | 4 | 8 | 5 | 0.066 | 11 / 8 |
| audio | 8 | 9 | 9 | 0.066 | 12 / 15 |
| bild | 0.5 | 3 | 0 | 0.000 | 1 / 4 |
| bild | 1 | 4 | 0 | 0.000 | 1 / 5 |
| bild | 2 | 6 | 2 | 0.003 | 2 / 9 |
| bild | 4 | 11 | 4 | 0.011 | 9 / 13 |
| bild | 8 | 12 | 15 | 0.040 | 15 / 26 |
| beide | 1 | 9 | 0 | 0.000 | 9 / 5 |
| beide | 2 | 12 | 2 | 0.003 | 12 / 10 |
| beide | 4 | 14 | 8 | 0.066 | 18 / 19 |
| beide | 8 | 14 | 17 | 0.066 | 23 / 32 |

Alle Mediane exakt +0.0000. Der Tiebreak der Registrierung (kleinstes
Gewicht, dann `audio`) führt auf **`audio w=0.5`**.

### Prüfsatz, 51 Aufnahmen, einmal angefasst

`audio w=0.5`: n=51, Median +0.0000, 1 besser, 0 schlechter, größter
Einzelverlust 0.000, Kanten außen 0 / innen 1.

| Bedingung | Ergebnis |
|---|---|
| Median ≥ +0.005 | **NEIN** |
| besser ≥ 2× schlechter | JA |
| kein Verlust > 0.10 | JA |

**==> O19 NICHT ERFÜLLT. `spot_lp_w` bleibt 0.**

### Was inhaltlich hängen bleibt

**Der Mechanismus ist nicht wirkungslos, er ist wirkungsARM und kippt.**
Bis Gewicht 2 verbessert er einzelne Aufnahmen und verschlechtert keine.
Ab Gewicht 4 überholt der Schaden den Nutzen, bei 8 ist er im Minus
(`beide`: 14 besser, 17 schlechter). Ein Anker-Bonus, der stark genug
ist, um eine Kante zu ziehen, ist auch stark genug, um sie an die falsche
Stelle zu ziehen.

**Die Behauptung in `spot_lp.go` ist widerlegt.** Der Kommentar sagt, der
Block werde „ausgedehnt, nie beschnitten". Die Bild-Anker tun genau das
Gegenteil: bei Gewicht 1 wandern **1 Kante nach außen und 5 nach innen**,
bei Gewicht 2 sind es 2 gegen 9. Der Grund ist keine Überraschung, wenn
man das Deckungsmaß daneben legt: 78.9 % der Bild-Anker liegen **ganz
innerhalb** eines Modellblocks, ihre Ränder sitzen also im Blockinneren,
und ein Übergangs-Bonus dort zieht die Blockgrenze nach innen. Ein
Übergangs-Bonus an Sekunde t wirkt aus BEIDEN Richtungen; die
Begründung im Kommentar gilt nur für einen Anker, dessen Rand außerhalb
liegt.

Die Audio-Anker verhalten sich umgekehrt (bei Gewicht 1: 8 außen, 1
innen), weil sie kürzer sind und näher an den Blockrändern sitzen.

**Für einen Nachfolger** wäre daraus zu lernen: nicht der Median über
alle Aufnahmen, sondern die Bilanz über die GEÄNDERTEN; und die Bild-Anker
gehören, wenn überhaupt, nur mit ihrem äußersten Rand je Block in den
Dekoder, nicht mit jedem inneren Spotrand. Beides braucht eine eigene
Registrierung.
