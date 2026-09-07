# O20 — Hilft es dem Kopf, wenn er zwei Sorten Werbung getrennt lernen darf?
(Vorab-Registrierung)

> **ABGESCHLOSSEN 2026-09-07 — REGEL NICHT ERFÜLLT.** Median-ΔF1
> **−0.0608**, 0 von 5 Seeds positiv, gegen die Schwelle +0.005 und
> 4 von 5. Die vorab festgelegte Konsequenz gilt: **Idee 5 wird als
> gemessen abgeschlossen.**
>
> ⚠️ **Aber der registrierte Arm hatte einen Konstruktionsfehler, und der
> ändert die Lesart.** Die Gewichtsregel `N/(K·n_k)` gibt bei drei Klassen
> der Werbung insgesamt das **6.3-fache** Gewicht gegenüber Sendung, bei
> zwei Klassen nur das **3.15-fache**. Der Versuchsarm bekam also nebenbei
> eine doppelt so starke Schieflagen-Korrektur — mehr Fehlalarme, weniger
> Präzision, weniger F1. Die Zielaenderung war mit einer
> Gewichtsaenderung vermengt.
>
> Nachträgliche Diagnose mit fairen Gewichten (Klasse 1 und 2 zusammen so
> schwer wie die eine Werbeklasse im Kontrollarm): Median-ΔF1
> **−0.0013**, 2 von 5 Seeds positiv. Also **null**, nicht schädlich.
>
> **Was daraus folgt.** Die registrierte Konsequenz bleibt — sie wird
> nicht nachträglich weggerechnet. Technisch heißt das Ergebnis aber
> „kein Effekt", nicht „schadet". Und getestet wurde ohnehin nur der
> billige Stellvertreter: die Anker-Aufteilung sagt „wurde
> fingerprintet", nicht „ist ein Produktspot". Dass sie nichts trägt,
> war im Voraus als Risiko benannt. **Die Hypothese aus dem zwölften
> Durchgang ist damit weder bestätigt noch widerlegt — der billige Weg,
> sie zu prüfen, ist erschöpft.** Ein echter Test braucht echte
> Unterklassen-Labels und eine eigene Registrierung.

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


---

## Ergebnis (2026-09-07, nach dem Lauf eingetragen)

| Seed | 2 Klassen | 3 Klassen (registriert) | 3 Klassen (faire Gewichte) |
|---|---|---|---|
| 0 | 0.8973 | 0.8446 | — |
| 1 | 0.9053 | 0.8440 | — |
| 2 | 0.9056 | 0.8417 | — |
| 3 | 0.9024 | 0.8416 | — |
| 4 | 0.8984 | 0.8441 | 0.8994 |
| **Median** | **0.9024** | **0.8440** | **0.8994** |

| | Median-ΔF1 | positive Seeds |
|---|---|---|
| registriert | −0.0608 | 0 von 5 |
| faire Gewichte (nachträglich) | −0.0013 | 2 von 5 |

| Bedingung | Ergebnis |
|---|---|
| Median ≥ +0.005 | **NEIN** |
| 4 von 5 Seeds positiv | **NEIN** |

**==> O20 NICHT ERFÜLLT. Idee 5 ist als gemessen abgeschlossen.**

### Was hängen bleibt

Der Kontrollarm erreicht F1 0.902 gegen 0.91 im Nightly — der
Offline-Aufbau ist also brauchbar und lässt sich für weitere
Kopf-Fragen wiederverwenden, ohne die Produktion anzufassen.

Und eine Lehre über Experimente: **beim Ändern des Ziels ändert sich die
Klassengewichtung mit, wenn man sie aus den Klassenhäufigkeiten
ableitet.** Das ist kein exotischer Fall, es passiert bei jeder
Aufteilung einer Klasse. Ohne die nachträgliche Diagnose hätte hier
gestanden, ein feineres Ziel schade dem Kopf um 0.06 F1 — eine Aussage,
die schlicht falsch gewesen wäre.
