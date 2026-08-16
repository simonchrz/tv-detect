# O14 — Auswahl statt Mischung: setzt die stärkere Flanke die bessere Kante?
(Vorab-Registrierung)

**Geschrieben 2026-08-15, bevor die Regel eine einzige Schatten-Zeile
erzeugt hat.** Läuft im selben Mechanismus wie
[O13](o13-ocr-schatten-preregistration.md), auf denselben Kanten, mit
eigener Bedingung.

## Woher die Frage kommt

Aus einem Einwand des Users: „das Senderlogo ist doch ein gutes Signal,
Werbung hat nie das Logo". Nachgemessen über 105 Aufnahmen — er hat recht,
und meine frühere Absage (§3af) war auf den Fehlerkanten gemessen und damit
zu breit:

* Logo im Werbeblock 0.104, in der Sendung 0.915. 85 % der Aufnahmen klar
  getrennt, 1 % Überlappung.
* Für die Kante ist es so gut wie das NN (Flanke 3.2 s gegen 3.3 s).
* Vor allem **komplementär**: das Logo liegt bei 37 % der Kanten näher, das
  NN bei 43 %, gleich bei 19 %.
* Und die Produktion nutzt es **gar nicht** — plain `hsmm` hat
  `emit := nnConf`.

`hsmm-blend` ist der vorhandene Weg und trägt nicht (Gesamt-Kantenfehler
4955 s gegen 4073 s), weil es **mittelt**. Zwei Signale, die jeweils in
verschiedenen Fällen recht haben, verlieren beim Mitteln beide.

## Die Regel — fixiert

Um jede hsmm-Blockgrenze im Fenster **±30 s** den nächstgelegenen
Schwellwert-Durchgang (0.5) suchen — einmal in der geglätteten
NN-Wahrscheinlichkeit, einmal im invertierten geglätteten Logo (beide 10 s
Fenster, wie in der Produktion). Von beiden die Kante mit der **größeren
lokalen Amplitude** (max − min über ±6 s um den Durchgang) nehmen. Findet
nur eines eine Flanke, gilt dieses; findet keines eine, bleibt die Kante wie
sie ist.

Keine freien Parameter zum Nachdrehen: Fenster, Glättung, Schwelle und
Amplitudenfenster stehen hier fest.

## Warum wieder prospektiv

Die Vorarbeit oben lief über **alle** 105 Aufnahmen mit frischen Signalen,
Golden-Mitglieder eingeschlossen. Ich habe daran keinen Parameter
abgestimmt — die Heuristik hat keine —, aber ich habe die Aggregate
gesehen. Nach vier Registrierungen, die genau an solchen Feinheiten
gescheitert sind (§3ad), ist das Grund genug, nicht auf denselben Zahlen zu
urteilen.

Gesammelt wird deshalb im Schattenlauf: Aufnahmen, deren Labels **nach dem
2026-08-15** entstanden sind. Die Regel wird **nicht angewandt**.

## Bedingung — festgelegt, bevor eine Zeile existiert

Ausgewertet wird erst bei **mindestens 60 Kanten** aus **mindestens 20
Aufnahmen** (mehr als bei O13, weil diese Regel fast jede Kante anfasst und
nicht nur die mit Marker). Vorher kein Zwischenstand.

**ERFÜLLT, wenn alle drei gelten:**

1. Der **Median**-Kantenfehler sinkt um **≥ 0.5 s**.
2. Der Anteil Kanten **≤ 2 s** steigt um **≥ 3 Prozentpunkte**.
3. Der **Gesamt**-Kantenfehler steigt nicht (≤ 0 % Veränderung).

Zusätzlich berichtet, nicht Bedingung: p75/p90, wie oft Logo bzw. NN
gewählt wurde, und wie oft gar keine Flanke im Fenster lag.

⚠️ Bedingung 3 steht da, weil die Vorarbeit genau dort ihre Schwäche hatte:
die Heuristik verbesserte Median und ≤2-s-Anteil, ließ den Schwanz (p75
13.5 s) aber praktisch unberührt. Eine Regel, die den Median schönt und die
Ausreißer verschlimmert, ist für den Ad-Skip wertlos.

## Vorhersage

Die Vorarbeit ergab 4.4 → 3.5 s Median und 36 % → 38 % bei ≤2 s. Damit wäre
Bedingung 1 erfüllt, Bedingung 2 **knapp verfehlt** (2 statt 3 Punkte).
Vorhersage deshalb: **NICHT ERFÜLLT, knapp**.

Das ist die erste Vorhersage heute, die nicht „besteht" lautet — nach vier
falschen in Folge (O9–O12) ist das der ehrlichere Ausgangspunkt.

## Was jeder Ausgang bedeutet — jetzt entschieden

**ERFÜLLT** → Auswahl statt Mischung wird in `tv-detect` gebaut (als
Nachlauf hinter dem Decoder, hinter einem Schalter, erst nach einer
bestätigenden Nacht scharf).

**NICHT ERFÜLLT** → Friedhof. Die Obergrenze perfekter Auswahl liegt bei
2.5 s Median gegen 4.4 s heute; wenn eine Heuristik davon nichts holt, ist
das Problem nicht die Auswahl, sondern dass man ohne Kenntnis der Wahrheit
nicht erkennt, welches Signal gerade recht hat. Dann bleibt nur, den
Signalen selbst mehr Information zu geben (§3aa).

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1) oder Labels
anzufassen (L2). Und die Kanten, an denen die Regel verliert, werden nicht
per Frame-Review als Label-Fehler umgedeutet.

---

# Ergebnis (2026-08-16)

76 Kanten aus 20 Aufnahmen, alle Labels von Review-Agenten gesetzt.

```
  [1] Median 0.0s -> 0.5s (-0.5s)       VERFEHLT (>= 0.5s besser)
  [2] <=2s 67% -> 63% (-4 Punkte)       VERFEHLT (>= 3 besser)
  [3] Gesamtfehler 482s -> 496s (+3 %)  VERFEHLT (<= 0 %)
  gewaehlt: nn 23, logo 53
```

**O14 NICHT ERFÜLLT**, in allen drei Bedingungen. Die Vorhersage lautete
„nicht erfüllt, knapp" — nicht erfüllt stimmt, knapp nicht: die Regel
verschlechtert jede der drei Größen.

## ⚠️ Was dieses Ergebnis NICHT sagt — Auswahleffekt meines eigenen Aufbaus

Die Ausgangslage ist **Median 0.0 s und 67 % der Kanten auf 2 Sekunden**.
Korpusweit sind es **4.4 s und 36 %** (§3af). Der Schattensatz ist also
dramatisch besser als der Korpus, und das hat einen Grund, den ich selbst
gebaut habe:

Ins Ledger kommt eine Aufnahme nur, wenn **jede** ihrer Kanten aus den
Bildern bestimmbar war. Genau daran sind die schlecht sitzenden Aufnahmen
gescheitert — RTL (168 Sendungsbilder gegen 22 Werbebilder selbst im
±72-s-Fenster), VOX, zweimal ProSieben. Die Ablehnung filtert also
systematisch die Fälle heraus, in denen das Modell weit danebenliegt, und
lässt die durch, bei denen es ohnehin stimmt.

**Damit prüft O14 die Flankenauswahl an genau den Kanten, an denen es
nichts zu gewinnen gibt.** Dass eine Regel dort schadet, ist wenig
überraschend: sie verschiebt eine bereits richtige Kante.

Das Urteil steht trotzdem — es war so registriert, und eine Kennzahl nach
dem Blick auf die Zahl auszutauschen ist genau das, wogegen die
Registrierung existiert. Aber die Aussage ist schwächer als „Flankenauswahl
hilft nicht": belegt ist nur, dass sie auf schon korrekten Kanten schadet.

## Was daraus folgt

Die offene Frage ist unverändert, aber sie braucht einen anderen Aufbau:
ein Satz, der die WEIT danebenliegenden Kanten enthält. Genau die kann das
Agenten-Review derzeit nicht labeln — bei RTL war im ganzen ±72-s-Fenster
kein Übergang zu sehen. Wer sie will, braucht ein Verfahren, das die Kante
erst grob findet (ganze Aufnahme grob abtasten) und dann verfeinert, statt
um die Modellkante herum zu suchen.

⚠️ Das ist kein Nachschlag zu O14 und keine dritte Runde derselben Idee,
sondern die Feststellung, dass mein Messsatz eine andere Population abbildet
als der Korpus. Wer die Flankenauswahl wirklich beurteilen will, muss das
zuerst reparieren.
