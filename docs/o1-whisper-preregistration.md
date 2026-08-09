# O1 — Whisper-Spalte: Vorab-Registrierung der Serie

**Geschrieben 2026-08-09, BEVOR die erste Serien-Nacht gelaufen ist.** Die
erste Zeile entsteht im Nightly vom 08-10. Der Sinn des Vorher-Schreibens ist,
dass unten alles gegen das geprüft werden kann, was tatsächlich passiert,
statt hinterher darauf zugeschnitten zu werden. Dieses Repo hat für die
Umkehrung schon zweimal bezahlt — der Boundary-Head las +0.016 auf unsauberen
Dumps und −0.015 auf sauberen, und das erste HMM-Ergebnis +0.242 gegen eine
falsch gelabelte Referenz.

Ledger-Eintrag: [`experiment-ledger.md`](experiment-ledger.md) §3 O1.
Muster: [`hsmm-holdout-preregistration.md`](hsmm-holdout-preregistration.md).

## Was getestet wird

Ob die Whisper-Spalte in der **heutigen** Architektur noch etwas beiträgt.
Gemessen als gepaarte Differenz zweier Schattenvarianten, die sich in genau
dieser Spalte unterscheiden:

```
  Δ  =  golden(mlp32-cwt-mp)  −  golden(mlp32-ct-mp)
         mit Whisper              ohne Whisper
```

Beide werden im selben Lauf, auf denselben Daten, mit denselben
Hygiene-Masken und Gewichten und gegen denselben Testsatz gefittet. Alles
außer der einen Spalte ist identisch, also ist Δ ihr Beitrag.

**Wichtige Einschränkung, jetzt festgehalten:** mit der
Wahrscheinlichkeits-Spalte fällt die Whisper-**Maske** weg (1301 → 1299). Die
Maske kodiert „Whisper-Daten vorhanden ja/nein" und hat ohne die
Wahrscheinlichkeit keinen Bezug mehr — sie *könnte* aber eigenständig etwas
tragen (welche Aufnahmen transkribiert wurden, ist nicht zufällig). O1 testet
die beiden deshalb **als Paket**. Fällt das Urteil auf „entfernen", ist damit
*nicht* gezeigt, dass die Wahrscheinlichkeit allein schadet. Die Trennung wäre
O1b und wird nur aufgemacht, wenn das Ergebnis uneindeutig ausfällt.

## Was schon gemessen ist (nicht Teil der Serie)

Drei Beobachtungen, alle vor dem Schreiben dieses Dokuments:

1. **Golden-Verlauf nach der Ausrichtungs-Korrektur:** 0.921 → 0.917 → 0.911.
   Vorher stand im Produktions-Fit eine zeitversetzte, faktisch zufällige
   Whisper-Spalte; Rauschen lernt ein Netz zu ignorieren. Seit `8aceaab`
   trägt sie ein echtes Signal bei, und der Trend zeigt nach unten.
2. **Die Sonde `channel + whisper`** liegt in zwei von drei Läufen unter
   `channel` allein (−0.027 / −0.020 / +0.008).
3. **Die isolierte Sonde, Handlauf 08-09:** Test −0.032, Golden −0.025.

*Vertrauen: mäßig.* Punkt 1 ist ein Trend über drei Nächte ohne Kontrollarm —
in diesem Zeitraum hat sich auch der Korpus verändert. Punkt 2 dreht im
dritten Lauf das Vorzeichen. Punkt 3 ist **eine** Messung, und derselbe Lauf
hat gleichzeitig die Minute-Prior-Sonde von +0.002 auf −0.023 gedreht — ein
gepaarter Vergleich derselben Bauart. Ich erwarte, dass die Serie negativ
ausfällt, aber schwächer als −0.025.

## Hypothese, mit Zahlen, vor den Daten

**H1 — die Whisper-Spalte kostet.** Über **5 gültige Serien-Nächte**
(08-10 bis 08-14):

* Median der fünf Δ **≤ −0.010**, **und**
* **≥ 4 von 5** Nächten negativ.

**Warum −0.010:** das ist die Schwelle, die dieses Repo an anderer Stelle
schon als „bedeutsam" gesetzt hat — der Vorgabewert von `--golden-floor`.
Sie wird hier nicht neu gewählt, sondern übernommen, damit sie nicht zum
Ergebnis passend gemacht werden kann.

**Warum 5 Nächte:** Präzedenz sind 3 (Minute-Prior) und 7 (temporal). Ein
Vorzeichentest über 5 Nächte ist für sich **schwach** — 4/5 einseitig ergibt
p ≈ 0.19, erst 5/5 kommt auf p ≈ 0.03. Deshalb muss *zusätzlich* die
Magnitude halten. Diese Schwäche wird hier benannt statt später als
Signifikanz verkauft.

**Vorab festgelegte Verlängerung:** ergibt der Seed-Sweep (5 Fits derselben
Architektur, nur anderer Init-Seed, läuft am 08-09) eine Standardabweichung
des Golden-Medians **> 0.010**, wird N von 5 auf **9** erhöht, bevor
entschieden wird. Begründung: liegt schon der reine Fit-Zufall über der
Entscheidungsschwelle, misst eine 5-Nächte-Serie den Seed und nicht die
Spalte. Der Sweep ist eine von O1 unabhängige Messung — sein Ergebnis darf
die Schwelle also setzen, ohne die Registrierung zu entwerten.

## Was als gültige Nacht zählt — vorher festgelegt

Eine Nacht geht in die Serie ein **genau dann**, wenn ihre Zeilen in
`shadow-trend.jsonl`:

1. `set_hash == "c8727e8266a8"` tragen (der Golden-Satz hat schon einmal
   still gewechselt),
2. `decoder == "--decoder hsmm --hsmm-dur-w 15"` tragen (der Wechsel
   `form` → `hsmm` hat jede Zahl davor entwertet),
3. `golden_n == 38` haben — ein unvollständiger Satz ist nicht
   zusammensetzungs-konstant,
4. **beide** Arme enthalten (`mlp32-cwt-mp` und `mlp32-ct-mp`).

Fällt eine Nacht durch, wird die Serie um eine Nacht verlängert, nicht die
Nacht ersetzt oder umgedeutet. Der Handlauf vom 08-09 zählt **nicht** mit: er
lief gegen einen anderen Testsatz-Zuschnitt und ist die Vorstudie, nicht die
Serie.

## Was das Ergebnis bedeutet — jetzt entschieden

| Ausgang | Konsequenz |
|---|---|
| H1 hält | Migration auf MLP6 ohne Whisper und Maske (1301 → 1299) vorbereiten. Erst Go-Seite + Paritäts-Fixture, dann Header-Bump. Nicht ohne ausdrückliches OK (Leitplanke L5). |
| Median ≤ −0.010, aber nur 3/5 negativ | **Nicht** migrieren. Serie auf 9 Nächte verlängern. Ein Median, den die Vorzeichen nicht stützen, ist von einer Nacht getragen. |
| 4/5 negativ, aber Median > −0.010 | **Nicht** migrieren. Konsistent klein ist unterhalb dessen, was das Gate ohnehin durchwinkt — die Spalte bleibt, O1 wird als „vernachlässigbar" geschlossen. |
| Uneindeutig oder gemischt | O1b aufmachen: dritter Arm mit Maske aber ohne Wahrscheinlichkeit, um die beiden zu trennen. Nicht vorher — ein dritter Arm verwässert den Haupttest. |
| Δ positiv | O1 geschlossen, Spalte bleibt, Eintrag in den Friedhof. Dann ist die Erklärung für den Abwärtstrend seit 08-07 **woanders** zu suchen, und das ist die eigentliche Erkenntnis. |

## Dieselbe Regel, maschinenlesbar

Nachtrag vom selben Tag, **vor der ersten Serien-Nacht** — Wortlaut oben
unverändert, hier nur in einer Form, die `scripts/audit-preregistration.py`
ohne mich auswerten kann. Der Sinn: am Ende der Serie urteilt nicht der, der
die Registrierung geschrieben hat. Ändert sich dieser Block nach der ersten
gezählten Nacht, schlägt das Audit Alarm — deshalb steht er hier und nicht
in meinem Kopf.

```regel
{
  "id": "O1",
  "frage": "Kostet die Whisper-Spalte mehr als sie bringt?",
  "serie_ab": "20260810",
  "naechte": 5,
  "verlaengert_auf": 9,
  "verlaengerung_wenn_seed_std_ueber": 0.010,
  "arme": {"mit": "mlp32-cwt-mp", "ohne": "mlp32-ct-mp"},
  "delta": "mit minus ohne, auf golden_median",
  "gueltige_nacht": {
    "set_hash": "c8727e8266a8",
    "decoder": "--decoder hsmm --hsmm-dur-w 15",
    "golden_n": 38
  },
  "bedingungen": {
    "median_hoechstens": -0.010,
    "negative_naechte_mindestens": 4
  }
}
```

## Was die Migration zusätzlich beachten muss

Der Dimensionswechsel 1301 → 1299 **überspringt das Head-to-Head** — das Gate
kann zwei Architekturen nicht paarweise vergleichen. Dann schützen nur noch
der historische IoU-Boden und der Golden-Boden. Deshalb jetzt festgelegt: der
erste Kopf ohne Whisper muss den Golden-Bestwert (0.921) **schlagen**, nicht
bloß im Toleranzband darunter liegen. Eine fehlende Schutzschicht wird durch
eine höhere Latte ersetzt, nicht durch Vertrauen.

## Bekannte Störgrößen — benannt, nicht wegerklärt

* **Der Korpus wächst jede Nacht.** Für Δ ist das kontrolliert (beide Arme
  sehen dieselben Daten), für das absolute Niveau nicht. Deshalb entscheidet
  Δ und nicht der Verlauf des einzelnen Arms.
* **Die Nächte sind nicht unabhängig.** Aufeinanderfolgende Korpora
  überlappen zu über 99 %. Ein Vorzeichentest tut so, als wären sie es. Das
  ist die optimistische Annahme in dieser Registrierung; sie macht 4/5 eher
  leichter als schwerer zu erreichen.
* **Die Schattenvarianten sind keine Produktionsköpfe.** Sie durchlaufen
  dieselben Gewichte und Masken (seit 2026-07-07), aber nicht den
  All-Data-Refit. Δ misst die Spalte im Fit, nicht im ausgelieferten Kopf.
