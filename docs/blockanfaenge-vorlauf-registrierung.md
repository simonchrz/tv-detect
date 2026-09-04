# Liegt vor gelabelten Blockanfängen schon Werbung?

**Registriert 2026-09-03, VOR dem ersten Agenten-Lauf.**

## Anlass

Bei der O13-Kalibrierung (`o13-blindkanten-kalibrierung.md`) wurden drei
Menschen-Labels nachuntersucht; zwei waren am BLOCKANFANG zu spät (5 s bzw.
~20 s), beide um ein davorliegendes Eigenpromo, das nach §3y zur Werbung
gehört. Acht Kanten sind ein Verdacht, keine Rate. Diese Messung stellt die
Rate fest.

## Warum Agenten das hier dürfen

Gefragt wird NICHT nach einer Grenze, sondern pro Bild: „Werbung oder
Sendung?" Das ist Klassifikation — gemessen 4/4 richtig
(Memory `agenten_review_frage_entscheidet`), und in der Kalibrierung
2 von 3 echten Label-Fehlern gefunden.

⚠️ **Hier wird NICHT geblendet.** Die Frage hat nichts mit der OCR-Regel zu
tun, und ein Mensch läse die Einblendung an dieser Stelle auch. **Folge, die
festgehalten sein muss:** so gefundene Korrekturen dürfen NIE als
O13-Referenz dienen — sonst kommt die Zirkularität, gegen die O13 gebaut
ist, durch die Hintertür zurück.

## Aufbau

* Grundmenge: Aufnahmen mit **echtem** Menschen-Label (ohne
  `auto_confirmed_at`, `reviewed_by`, `auto_confirmed_via_fingerprint`) und
  Quelle im Cache.
* Stichprobe: **20 Blockanfänge**, zufällig, höchstens 2 je Sender.
* Je Anfang: Bilder von **−24 s bis +4 s** in 2-s-Schritten (15 Bilder). Die
  beiden Bilder NACH dem Anfang sind eine Kontrolle: hält der Agent auch die
  für Sendung, ist sein Urteil für diese Kante unbrauchbar und sie zählt als
  „unklar", nicht als Beleg.

## Auswertung (vorab festgelegt)

Ein Blockanfang gilt als **ZU SPÄT**, wenn:

1. beide Kontrollbilder (+2 s, +4 s) WERBUNG sind, UND
2. die ≥2 unmittelbar davorliegenden Bilder (−2 s, −4 s) WERBUNG sind.

„Zu spät um X s" = Länge der ununterbrochenen WERBUNG-Kette rückwärts vom
Anfang. Reicht sie bis −24 s, wird das als „≥24 s" notiert, nicht
hochgerechnet.

## Was daraus folgen darf

Nur ein **Kandidatenbericht** an Simon. Es wird kein Label geschrieben —
L2 (Labels sind Eingabe, nicht Stellschraube). Ab einer Rate von ~25 %
wäre der Kanten-Maßstab selbst betroffen und der Befund gehört vor jede
weitere Kanten-Messung.

---

## Ergebnis 2026-09-03

20 Blockanfänge, 10 Sender, 300 Bilder, 10 Agenten. Auswertung nach der
Regel oben (`scripts/vorlauf_auswerten.py`, 7 Tests).

**5 zu spät · 12 sauber · 3 unklar → Rate 5/17 = 29 %.**
Damit über der Schwelle, ab der laut Registrierung der Kanten-Maßstab
selbst betroffen ist.

| Aufnahme | Kanal | Anfang | zu spät um | was davor liegt |
|---|---|---|---|---|
| dvr-vox-1779292500 | vox | 2891.0 | 6 s | Ident + Show-Logo-Trenner |
| dvr-prosieben-1782743457 | prosieben | 690.0 | ≥24 s | Gewinnspiel (GewinnArena, Rufnummer) |
| dvr-kabel-eins-1783924200 | kabel-eins | 924.0 | 10 s | Joyn-Ident + Filmtrailer mit Sendetermin |
| dvr-comedy-central-1778617500 | comedy-central | 839.0 | ≥24 s | „Ghosts"-Trailer + Sender-Idents |
| dvr-disney-channel-1781554200 | disney-channel | 1087.1 | 12 s | Disney-Trailer „Freitag 20:15" |

**Die 29 % sind aber kein Qualitätsmaß, sondern ein Datum-Maß.** Vier der
fünf sind Trailer/Ident-Fälle, also §3y („Rand-Trailer gehören ZUR
WERBUNG", bindend seit 13.08.), einer ist Gewinnspiel (§3am, seit 17.08.
wieder Werbung). Nach dem Label-Datum:

* **3 stammen von VOR der Konvention** (23.05., 05.06., 30.06.) — sie sind
  nicht falsch gesetzt, sie folgen einer anderen Regel.
* **2 stammen von danach** (13.08., 14.08.) — das sind echte Fehlstellen
  unter einer Konvention, die zum Zeitpunkt galt.

⚠️ **Das ist wörtlich der Fall, vor dem §3am warnt:** „Wer eine Konvention
ändert, muss im selben Zug sagen, wer die Labels nachzieht — sonst misst ein
späterer Lauf die Lücke als Qualität." Der Nachzieh-Lauf vom 17.08. hat
gespeicherte BILD-Urteile neu abgeleitet und deshalb nur Aufnahmen erreicht,
für die es solche gab (5 kabel-eins). Menschlich gelabelte Aufnahmen von
Mai bis Juli hat er nie berührt.

**Folge für jede Kanten-Messung:** der Korpus wird gegen einen gemischten
Maßstab gemessen. Ein Modell, das Blockanfänge früh setzt (Trailer
mitnimmt), wird auf den alten Labels bestraft und auf den neuen belohnt.
Das erklärt einen Teil von O14 („Blockstarts zu spät", §3al) als
Label-Artefakt, ohne ihn ganz zu erklären.

**Nicht getan:** kein Label geschrieben (L2). Die fünf oben sind
Kandidaten.

**Nächster sinnvoller Schritt, falls gewünscht:** nicht mehr stichprobenhaft
messen, sondern die Frage „liegt vor diesem Blockanfang ein Trailer/Ident?"
über alle Menschen-Labels VOR dem 13.08. laufen lassen — das ist die
Menge, die der Nachzieh-Lauf ausgelassen hat. Kosten: ~1 Agentenlauf je 2
Blockanfänge.

---

## Vollständiger Lauf 2026-09-03: 12 von 85 = 14 % — Schwelle NICHT erreicht

106 Blockanfänge aus 64 Aufnahmen, 1590 Bilder, 36 Agenten.
102 ausgewertet (4 fielen einem Sitzungslimit zum Opfer, s.u.).

```
zu spät: 12   sauber: 73   unklar: 17      Rate 12/85 = 14 %
```

⚠️ **Das korrigiert die Stichprobe von heute Vormittag nach unten.** Dort
waren es 5/17 = 29 %, und ich hatte gemeldet, die vorab gesetzte Schwelle
von ~25 % sei überschritten. **Auf der vollen Menge ist sie es nicht.** 17
Kanten sind zu wenig, um eine Rate zu schätzen — genau der Fehler, vor dem
der eigene Vorbehalt („ein Verdacht, keine Rate") gewarnt hat, und ich habe
ihn trotzdem als Schwellenüberschreitung gemeldet.

### Die 12 zerfallen in zwei sehr ungleiche Gruppen

| | Kanten | Aufnahmen | Label-Datum |
|---|---|---|---|
| erheblich (≥10 s) | 5 | **3** | 4× Mai, 1× Juni |
| grenzwertig (genau 4 s) | 7 | 7 | 4× Mai, 2× Juni, 1× Juli |

Die 4-s-Fälle sind das Minimum, das die Regel überhaupt meldet (zwei
Bilder) — typischerweise ein einzelnes Ident/Bumper unmittelbar vor der
Kante. Ob das ein Fehler ist, hängt daran, wie streng man §3y liest.
**Die belastbare Menge sind die drei Aufnahmen mit ≥10 s.**

### Konventions-Drift bestätigt sich in der Richtung, nicht in der Größe

Alle 5 erheblichen Fälle tragen Labels aus Mai/Juni, keiner aus Juli — bei
einer Grundmenge von 49 Mai / 29 Juni / 28 Juli. Die Richtung passt also zur
These (ältere Labels folgen der alten Regel), aber die Menge ist klein
genug, dass sie auch Zufall sein kann. **Kein Beleg für eine flächige
Drift.** Der Nachzieh-Lauf vom 17.08. hat weniger liegen lassen, als die
Stichprobe nahelegte.

### Die 17 „unklar" sind kein Rauschen, sondern ein eigener Befund

Sieben davon stammen aus EINER Aufnahme: `dvr-rtl-1780078500` (Let's Dance,
RTL live, 3,7 h). Dort zeigt der Frame am gelabelten Blockanfang die
laufende Tanzdarbietung mit Voting-Einblendung — **der Block fängt zu FRÜH
an, es wird in die Sendung hineingeschnitten.** Das ist die Gegenrichtung
zu dieser Frage; die Regel gibt korrekt „unklar" zurück, statt eine Zahl zu
erfinden. Braucht eine eigene Messung mit Nachlauf.

### Kandidaten (nichts geschrieben, L2)

Durchgehend betroffen — jeder geprüfte Anfang zu spät, also eher eine
Aufnahme nach anderer Regel als ein Ausrutscher:

* `dvr-kabel-eins-1779119815` (23.05.) — 1526.2 → ab ~1504.2 (22 s);
  2902.2 → ab ~2882.2 (20 s)
* `dvr-prosieben-1778778286` (16.05.) — 2390.5 → ab ~2366.5 (≥24 s);
  3312.7 → ab ~3288.7 (≥24 s)

Einzeln:

* `dvr-comedy-central-1778617500` (05.06.) — 839.0 → ab ~815.0 (≥24 s)

Die sieben 4-s-Fälle stehen im Werkzeug-Bericht
(`scripts/vorlauf_bericht.py`), sind aber als Kandidaten zu schwach.

### Nachtrag: vollständig, 14 von 89 = 16 %

`e0103`–`e0106` sind nachgeholt (zwei davon 6 s zu spät). Endstand über
alle 106:

```
zu spät: 14   sauber: 75   unklar: 17      Rate 14/89 = 16 %
```

Die Aussage ändert sich nicht: **unter der registrierten Schwelle von
~25 %**, belastbar sind weiterhin die drei Aufnahmen mit ≥10 s Versatz.

⚠️ **Kostennotiz für die nächste Runde:** die 36 Agenten liefen alle auf
dem großen Modell — rund 3 Mio. Token für reine Einzelbild-Klassifikation.
Der Nachlauf über Sonnet lieferte dieselbe Aufgabe ohne erkennbaren
Qualitätsunterschied. Für Klassifikations-Sweeps gehört `model: sonnet` in
den Agenten-Aufruf; das große Modell nur dort, wo abgewogen wird.

---

## Neulauf 2026-09-04 mit korrigierter Konvention (registriert VOR dem Lauf)

Der Lauf vom 03.09. zählte eine Einblendung ÜBER dem laufenden Bild
(Voting-Balken, Gewinnspiel-Leiste, Sponsorhinweis am Rand) als WERBUNG.
Das ist falsch: darunter läuft die Sendung weiter. Aufgefallen ist es, als
die Messung vom 04.09. mit der korrigierten Regel den Kontrollen
widersprach (`blockanfaenge-zufrueh-registrierung.md`).

**Was sich ändert:** nur der Agenten-Auftrag, um genau diesen Satz:
„Eine Einblendung ÜBER dem laufenden Bild macht die Aufnahme nicht zu
Werbung, solange darunter die Sendung weiterläuft. Werbung ist es, wenn
das ganze Bild dem Spot oder der Ankündigung gehört."

**Was gleich bleibt:** dieselben 106 Kanten, dieselben Bilder (nichts wird
neu geschnitten), dieselbe Auswertungsregel (`vorlauf_auswerten.py`),
dieselben Kontrollbedingungen. Agenten auf Sonnet.

**Die alten Urteile bleiben liegen** (`sweep-urteile/`), die neuen gehen
nach `sweep-urteile-v2/`. Beides wird gegenübergestellt — die Differenz
IST der Effekt der Konvention und gehört berichtet, nicht überschrieben.

**Erwartung, vorab festgehalten:** die Rate „zu spät" sollte SINKEN. Wer
Overlays als Werbung zählt, sieht den Werbebeginn früher und hält damit
mehr Kanten für zu spät. Kommt sie stattdessen höher heraus, ist meine
Erklärung des 04.09.-Widerspruchs falsch und gehört neu untersucht.
