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
