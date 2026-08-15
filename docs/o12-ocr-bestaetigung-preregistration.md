# O12 — Der eine Schuss: trägt die OCR-Regel auf dem
Bestätigungssatz? (Vorab-Registrierung)

**Geschrieben 2026-08-15, VOR der Messung.** Abschluss der Reihe O9–O11.
Es gibt **keine Abstimmung mehr** — die Regel ist fixiert, gemessen wird
einmal, und das Ergebnis gilt.

## Warum diese Registrierung anders aussieht

Drei Fehlschläge mit drei verschiedenen Gründen: O9 an der Sache (ein
globaler Regler taugt nicht), O10 an der Referenz (drei Labels waren zu
kurz), O11 an der Kennzahl (der Median über alle Aufnahmen ist blind für
einen Eingriff, der nur 11 von 56 anfasst).

⚠️ Jeder Grund war echt und ist protokolliert — aber wer oft genug neu
registriert, gewinnt irgendwann durch Zufall. Deshalb: keine vierte
Abstimmung, keine weitere Parameter-Suche, kein Frame-Review der Verlierer.
Es bleibt **eine unverbrauchte Instanz**, und die wird genau einmal
befragt.

## Die Regel — fixiert, nichts mehr verstellbar

Nach jedem hsmm-**Blockende** bis 60 s vorwärts nach dem OCR-Muster
(Wochentag / „Heute" / „MO-FR" / „Jeden Tag" **und** eine Uhrzeit) suchen,
Abtastung alle 2 s. Bei Treffern das Blockende auf `letzter Treffer + 5 s`
setzen — **außer** das Muster steht auch im Referenzfenster +180 s…+240 s,
dann ist es eine Dauer-Einblendung und wird verworfen.

Nur Enden. Kein Start, kein zweiter Parametersatz.

## Der Satz

Die 22 Golden-Aufnahmen mit Material — in O9, O10 und O11 registriert und
**nie gemessen**.

⚠️ Offengelegt wie zuvor: das OCR-**Muster** wurde an Trailer-Spannen
entworfen, von denen einige in diesen Aufnahmen liegen. Über das Muster ist
der Satz nicht naiv; über die Flüchtigkeits-Bedingung, die Parameter und
die eigentliche Frage (werden daraus bessere Kanten?) schon.

⚠️ Zweite Offenlegung: nach dieser Messung ist der Golden-Satz für **diese
Regel** verbraucht. Er bleibt der Maßstab für Modellstände, taugt aber nie
wieder als unabhängige Prüfung der OCR-Regel.

```json
[
 "dvr-comedy-central-1784109600",
 "dvr-disney-channel-1779163800",
 "dvr-kabel-eins-1778511363",
 "dvr-nick-1781550300",
 "dvr-nick-1783889700",
 "dvr-prosieben-1778926191",
 "dvr-prosieben-1779089343",
 "dvr-prosieben-1779188536",
 "dvr-prosieben-1782450900",
 "dvr-prosieben-1782484228",
 "dvr-prosieben-1783401000",
 "dvr-rtl-1782090300",
 "dvr-rtlzwei-1779128100",
 "dvr-rtlzwei-1780341300",
 "dvr-rtlzwei-1784571300",
 "dvr-sat-1-1778572809",
 "dvr-sat-1-1781979300",
 "dvr-sixx-1779031634",
 "dvr-sixx-1780378004",
 "dvr-toggo-plus-1780640400",
 "dvr-vox-1778860800",
 "dvr-vox-1779206400"
]
```

## Bedingung — diesmal der Sparsamkeit angemessen

Die Regel fasst nur Aufnahmen an, an deren Blockende ein flüchtiger
Programmhinweis steht. Gewertet wird deshalb **auf den angefassten
Aufnahmen** — welche das sind, entscheidet die Regel selbst, ohne einen
Blick auf die Labels, insofern ist die Auswahl zulässig.

**ERFÜLLT, wenn alle drei gelten:**

1. Median-IoU-Delta auf den angefassten Aufnahmen **≥ +0.010**
2. **mehr bessere als schlechtere** Aufnahmen
3. kein Einzelverlust größer **0.05**

Zusätzlich berichtet, aber nicht Bedingung: der Median über alle 22
Aufnahmen, die Zahl der angefassten, und die Sekunden „Sendung
weggeschnitten" vorher/nachher.

⚠️ Fasst die Regel **weniger als 5** Aufnahmen an, gilt die Frage als
**nicht entscheidbar** — nicht als bestanden. Bei O11 waren es 11 von 56;
auf 22 Aufnahmen können es zu wenige sein, und ein Median über drei
Aufnahmen ist keine Antwort.

## Vorhersage

Auf dem Abstimmungssatz lag der Effekt bei den angefassten Aufnahmen bei
Median +0.0201 (7 besser / 3 schlechter, Spanne −0.049…+0.190). Ich erwarte
**ERFÜLLT, aber knapp** — die Bedingung +0.010 ist bewusst tiefer als der
gemessene Effekt gesetzt, weil ein Holdout regelmäßig schwächer ausfällt
als die Abstimmung. Bei O9, O10 und O11 lag ich mit „besteht" jeweils
falsch; das sagt etwas über meine Kalibrierung, nicht über diese Regel.

## Was jeder Ausgang bedeutet — jetzt entschieden

**ERFÜLLT** → OCR kanten-lokal in den Detect-Lauf (+17 % Laufzeit), Regel
hinter `--ocr-hinweis`, standardmäßig AUS bis ein Nightly sie bestätigt.
Erst danach die teure Variante erwägen (eigene Spalte, Neu-Extraktion).

**NICHT ERFÜLLT** → Friedhof, und die kanten-lokale Spur ist abgeschlossen.
Der Befund §3aa (der Backbone kann die Einblendung nicht lesen) bleibt
davon unberührt gültig; die Konsequenz wäre dann die teure: das Merkmal
gehört in den Kopf, nicht dahinter.

**NICHT ENTSCHEIDBAR** (< 5 angefasste Aufnahmen) → die Frage bleibt offen
und braucht einen größeren unverbrauchten Satz. Kein Ausweichen auf den
Abstimmungssatz.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1) oder Labels
anzufassen (L2).
