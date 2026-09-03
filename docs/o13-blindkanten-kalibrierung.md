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

---

## Ergebnis 2026-09-03: NICHT ERFÜLLT

| Kante | Kanal | Art | Wahrheit | abgeleitet | Abstand | Fehlurteile |
|---|---|---|---|---|---|---|
| 1 | rtlzwei | start | 1317.9 | 1316.4 | 1.5 s | 0/13 |
| 2 | rtlzwei | ende | 2046.0 | 2046.5 | 0.5 s | 0/13 |
| 3 | prosieben | start | 2590.6 | 2590.1 | 0.5 s | 0/13 |
| 4 | sat-1 | start | 2691.1 | — | — | 0/13 |
| 5 | super-rtl | start | 844.0 | — | — | 0/13 |
| 6 | sixx | start | 1857.7 | — | — | 0/13 |
| 7 | prosieben | ende | 1290.0 | 1288.5 | 1.5 s | 0/13 |
| 8 | nick | ende | 1062.6 | 1062.1 | 0.5 s | 0/13 |

Bedingung 1: ≤2 s in **5 von 8** (verlangt 6) — VERFEHLT.
Bedingung 2: **3** Ausreißer (verlangt 0) — VERFEHLT.

**Das Urteil steht. Es wird nicht umgedeutet.**

## Beobachtung (NICHT Teil des Urteils)

Die acht Kanten zerfallen in zwei saubere Gruppen, nicht in einen Verlauf:

* **5 Treffer, alle ≤1,5 s, alle mit 0 Fehlurteilen über 13 Bilder.** Wo der
  geblendete Agent einen Übergang sieht, sitzt er sehr genau — genauer als
  der Produktions-Decoder (Median 2,0 s, Ledger O14).
* **3 Fehlschläge, alle derselben Bauart:** durchgehend WERBUNG über das
  ganze Fenster, kein Übergang, ausschließlich `start`-Kanten. Der Agent hat
  also nicht danebengegriffen, er hat 6 s VOR der Label-Kante schon Werbung
  gesehen (Joyn-Trailer „Promi taste"; Dickie-Toys-Spot; sixx-Ident +
  Autospot).

Damit stehen zwei Erklärungen offen, die sich ausschließen:

* **(A) Der Agent taugt nicht** — er hält Sendungsmaterial für Werbung.
* **(B) Das Label ist zu spät** — der Werbeblock beginnt früher, als der
  Mensch ihn gesetzt hat. Dafür spricht die Konvention aus §3y (Rand-Trailer
  gehören ZUR WERBUNG) und der Befund aus §3al (Blockstarts zu spät). Bei
  Kante 4 ist das sichtbar ein Trailer vor der Label-Kante.

## Zusatzfrage (registriert 2026-09-03, VOR dem Lauf)

Dieselben drei Kanten, Fenster ±20 s statt ±6 s, Abstand 2 s, gleiche
Blendung, gleiche Frageform.

* Findet der Agent dort einen Übergang **vor** der Label-Kante, ist (B)
  belegt — dann ist der Befund einer über LABEL, nicht über Agenten, und die
  Kalibrierung ist mit korrigierten Labels zu wiederholen.
* Bleibt es über ±20 s durchgehend WERBUNG oder liegt der Übergang **nach**
  der Label-Kante, ist (A) belegt und O13 wird geparkt wie O14.

⚠️ Auch ein Ausgang (B) macht die Kalibrierung oben NICHT nachträglich
erfüllt. Er sagt nur, woran der nächste Versuch ansetzen müsste.

## Zusatzfrage: Ergebnis — die Dichotomie war zu grob

Drei Kanten, ±20 s, 2 s Abstand. Drei verschiedene Ursachen, keine davon
sauber (A) oder (B):

| Kante | Übergang laut Agent | Label | Befund |
|---|---|---|---|
| 4 (sat-1) | 2690.1 | 2691.1 | **1,0 s daneben — das Label stimmt.** Im engen Fenster lagen nur 2 Bilder Sendung vor der Kante, im weiten 10. |
| 5 (super-rtl) | Werbung ab ≤824.0 | 844.0 | Label-Start ~20 s zu spät: Mattel-Packshot mit Copyright-Zeile, Dickie-Toys-Spots. Aber der Agent nennt 864.0 (mitten im Label-Block) SENDUNG — Paw-Patrol-Bild ohne Senderlogo, mutmaßlich ein Trailer. |
| 6 (sixx) | 1852.7 | 1857.7 | Label-Start 5,0 s zu spät: davor ein sixx-Eigenpromo, das nach §3y ZUR WERBUNG gehört. |

**Damit ist die Zusatzfrage nicht beantwortet, sondern widerlegt.** Sie
unterstellte, es gebe EINE Ursache. Tatsächlich:

1. **Ein Fehlschlag war mein Aufbau.** Fenster ±6 s mit Versatz bis ±4 s
   lässt im schlimmsten Fall 2 Bilder auf einer Seite — zu wenig, damit ein
   Klassifikator die zweite Klasse überhaupt als Klasse sieht. Kante 4
   sitzt im weiten Fenster auf 1,0 s. Der Agent war nie das Problem.
2. **Zwei Labels sind zu spät**, beide am Blockanfang, beide um ein
   Eigenpromo bzw. eine Spot-Strecke, die davor lag.
3. **Und der Agent scheitert an Trailern** (Kante 5, Bild bei 864 s). Das
   ist keine Randnotiz: §3ap sagt, der Kantenschwanz besteht aus genau
   dieser Klasse (8 von 8). Wo O13 am meisten gebraucht würde, ist der
   Agent am schwächsten — und Trailer sind Programmmaterial, das kann auch
   Blendung nicht heilen.

## Konsequenz

**Agenten ersetzen die menschliche Kanten-Referenz für O13 NICHT.** Nicht
weil sie ungenau wären — mit genug Kontext sitzen sie auf ≤1,5 s bei null
Fehlurteilen —, sondern weil sie genau die eine Klasse nicht können, aus der
der Kantenschwanz besteht. Ein Maßstab, der bei den schweren Fällen
systematisch kippt, ist als Maßstab wertlos, auch wenn er im Mittel gut
aussieht.

**Was Agenten hier sehr wohl können:** Blockanfänge auf ein davorliegendes
Eigenpromo prüfen — das ist Klassifikation, nicht Grenzziehung, und traf in
2 von 3 Fällen einen echten Label-Fehler. Das ist ein eigener Vorschlag,
keine Fortsetzung dieser Registrierung.

⚠️ Stichprobe: 8 Kanten, davon 3 nachuntersucht. Zwei zu späte Labels sind
ein Verdacht, keine Rate.
