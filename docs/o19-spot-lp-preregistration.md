# O19 — Verbessern Spot-Anker als Übergangs-Evidenz die Blockkanten?
(Vorab-Registrierung)

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
