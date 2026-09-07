# O20 — Hilft es dem Kopf, wenn er zwei Sorten Werbung getrennt lernen darf?
(Vorab-Registrierung)

**Geschrieben 2026-09-07, nachdem der Kontrollarm lief und BEVOR der
zweite Arm gerechnet wurde.** Die Schwelle steht als Vielfaches des
gemessenen Rauschens fest, bevor die erste Vergleichszahl existiert.

## Woher die Frage kommt

Gemessen am selben Tag (Ledger, zwölfter Durchgang): innerhalb der EINEN
Zielklasse „Werbung" irrt der Kopf auf dem Teil, der kein nachweislich
wiederholter Spot ist, drei- bis sechsmal so oft. Der Abstand bleibt,
wenn man nach Verwechslungsrate im Einbettungsraum schichtet (also nicht
die Repräsentation) und wenn man nach Abstand zur Blockkante schichtet
(also nicht der Kontext). Es bleibt die Zielklasse.

## Die Arme

Derselbe Kopf, dieselben Merkmale, dieselben Zeilen, derselbe Seed. Der
EINZIGE Unterschied ist das Ziel.

* **Kontrolle (2 Klassen)** — 0 Sendung, 1 Werbung. Wie heute.
* **Versuch (3 Klassen)** — 0 Sendung, 1 Werbung MIT Bild-Anker,
  2 Werbung OHNE Bild-Anker.

Gelesen wird bei BEIDEN nur die binäre Ausgabe: P(Werbung) = P(1) bei
zwei Klassen, P(1) + P(2) bei drei. Die dritte Klasse ist ein
Hilfssignal, kein Ziel.

⚠️ Die Anker-Aufteilung wäre als ECHTES Trainingsziel falsch — sie sagt
„wurde fingerprintet", nicht „ist ein Produktspot", und ein Spot, der nur
einmal lief, hat keinen Anker. Für DIESE Frage stört das nicht: gefragt
ist nur, ob das Aufteilen des Ziels die binäre Leistung hebt.

## Daten

Trainings-Archiv, Aufteilung nach dem bestehenden `split-ledger.json`.
Trainiert auf `train`, gemessen auf `test`. **Der versiegelte Satz wird
nicht angefasst.** Jede vierte Sekunde im Training (545913 Sekunden,
24.1 % Werbung, davon 66 % mit Anker), Test in voller Auflösung (488002
Sekunden, 23.2 % Werbung). Beide Arme sehen exakt dieselben Zeilen.

## Metrik und Rauschen

F1 auf `test`, geglättet mit einem 10-Sekunden-Mittel je Aufnahme —
dieselbe Idee wie `smooth=10s` im Nightly.

**Rauschen gemessen, bevor die Schwelle stand:** drei Seeds des
Kontrollarms ergaben 0.8973, 0.9053, 0.9056. Median 0.9053,
Standardabweichung **0.0047**. Zum Vergleich: der Nightly meldet für den
deployten Kopf F1 0.91 — der Aufbau ist also vergleichbar.

## Bedingung

```regel
{
  "id": "O20",
  "frage": "Hebt ein dreiklassiges Ziel die BINAERE Leistung?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf test, geglaettet 10s je Aufnahme",
  "arme": {"kontrolle": "2 Klassen", "versuch": "3 Klassen, binaer gelesen"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0047,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.005,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "Heterogenitaet bestaetigt; echte Unterklassen-Labels lohnen die Beschaffung. KEIN Deploy.",
  "konsequenz_bei_verfehlt": "Idee 5 wird als gemessen abgeschlossen, wie Idee 3 und 6."
}
```

Beide Bedingungen müssen gelten. Die Schwelle 0.005 ist rund eine
Standardabweichung des ungepaarten Rauschens; die Paarung über den Seed
nimmt einen Teil davon heraus, deshalb nicht zwei. Die Vorzeichenbedingung
trägt den Rest: sie hängt nicht an der Effektgröße.

⚠️ **Offengelegt: ich habe vorher eine Vergleichszahl gesehen.** Ein
Klempner-Lauf über 25 Aufnahmen mit 4 Epochen ergab −0.1856. Er lief
gegen einen Aufbau, in dem der Kopf nachweislich kollabiert war (NaN in
der Logo-Spalte, siehe unten), und sagt über die Frage nichts. Er steht
hier, weil eine Registrierung, die verschweigt was der Autor gesehen hat,
ihren Zweck verfehlt. Bemerkenswert ist die Richtung: die Zahl war
NEGATIV, die Schwelle kann also nicht auf ein erhofftes Ergebnis
zurechtgelegt sein.

## Drei Klempner-Fehler auf dem Weg, alle vor der Schwelle gefunden

1. **Unnormierte Merkmale** — vermutet, war es aber nicht.
2. **NaN in der Logo-Spalte.** `extract_logo` scheitert still auf einem
   kaputten Stream-Stück und hinterlässt NaN
   (`logo_nan_is_contention_not_corruption`). EIN NaN vergiftet über den
   Spaltenmittelwert die ganze Spalte und damit jede Zeile: der Kopf gab
   NaN aus, sagte auf alles „Werbung" und landete bei F1 **0.3771** —
   exakt der Wert für eine Alles-ist-Werbung-Vorhersage bei 23.2 %
   Werbeanteil. Der Hinweis war, dass drei Seeds auf vier Stellen
   dieselbe Zahl lieferten. Kein Modell, ein Kollaps. Die Produktion
   setzt dort denselben Sentinel 0.5 ein.
3. **Speicher** — der volle train-Eimer wären 15 GB; jede vierte Sekunde
   hält alle Aufnahmen im Satz.

Dass die Rauschmessung zuerst lief, hat alle drei gefunden. Ohne sie wäre
der Vergleich gegen einen kollabierten Kontrollarm gelaufen.
