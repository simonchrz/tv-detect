# Kalibrierung: taugt ein GEBLENDETER Agent als Kanten-Referenz?

**Geschrieben 2026-09-03, BEVOR die erste Agenten-Antwort gelesen wurde.**
Anlass: Simon hat keine Zeit für Reviews, O13 verlangt aber menschliche
Labels. Diese Kalibrierung prüft, ob ein Agent sie ersetzen kann — sie ist
NICHT O13 selbst und ändert dessen Registrierung nicht.

## Warum überhaupt geblendet

Zwei Gründe, beide gemessen, nicht vermutet:

1. **Zirkularität.** Die OCR-Regel zieht die Kante an einen Programmhinweis.
   Ein Agent, der denselben Frame sieht, liest ihn und setzt die Kante
   genau dort — die Regel bestätigt sich mit ihrer eigenen Evidenz. Deshalb
   werden die Textrahmen (Vision, `TVOCR_BOXEN=1`) geschwärzt, bevor ein
   Agent das Bild sieht, und nach dem Schwärzen wird gegengelesen: bleibt im
   geschwärzten Bereich Text stehen, ist das Fenster unbrauchbar.
   Text AUSSERHALB (Senderlogo, Produktname) bleibt und zählt nicht als
   Durchsickern — er ist nicht die Evidenz der Regel.
2. **Agenten können keine Grenzen.** Gemessen 2026-08-16: Bilder
   klassifizieren 4/4 richtig, nach der Grenze gefragt 2/3 falsch. Deshalb
   wird hier NUR pro Bild klassifiziert; die Grenze wird gerechnet, nie
   erfragt.

## Aufbau

8 Kanten aus 8 Sendern, gezogen aus Aufnahmen mit **echtem** Menschen-Label
(`ads_user.json` ohne `auto_confirmed_at`, ohne `reviewed_by`, ohne
`auto_confirmed_via_fingerprint` — siehe Ledger-Eintrag vom selben Tag).
Fenster ±6 s, 1 s Abstand, 13 Bilder. Die Fenstermitte ist um einen
zufälligen Versatz aus [-4, +4] s gegen die wahre Kante verschoben, damit
"die Mitte" nicht die Antwort ist. Dateinamen sind neutral (`bild01`…),
die Sekunden stehen nur in einer Karte, die der Agent nicht sieht.

## Ableitung der Grenze (vorab festgelegt)

Aus den 13 Urteilen wird der Trennpunkt gewählt, der die WENIGSTEN
Fehlklassifikationen erzeugt (bester einzelner Schnitt; bei Gleichstand der
früheste). Die Grenze ist die Mitte zwischen dem letzten Bild der einen und
dem ersten der anderen Klasse. Nicht-monotone Antworten sind damit zulässig
und werden nicht nachträglich geglättet.

## Bedingung (vorab festgelegt)

**ERFÜLLT**, wenn beides gilt:

1. In **mindestens 6 von 8** Kanten liegt die abgeleitete Grenze höchstens
   **2,0 s** von der Menschen-Kante entfernt. (2 s ist der Maßstab, an dem
   auch O14 gemessen wird — nicht für diese Prüfung erfunden.)
2. **Kein** Ausreißer über 5,0 s.

Nur bei ERFÜLLT wird vorgeschlagen, O13s Referenz von "Mensch" auf
"geblendeter Agent" umzuregistrieren — als eigene, neue Registrierung mit
eigenen Bedingungen, nicht als stille Ersetzung. Bei NICHT ERFÜLLT wird
O13 geparkt wie O14, und das Ergebnis steht trotzdem im Ledger.

⚠️ Die Stichprobe ist mit 8 Kanten klein. Sie kann eine grobe Untauglichkeit
zeigen, aber keine Feinheit belegen — fällt sie knapp aus, ist das ein
"nochmal mit mehr", kein Freibrief.
