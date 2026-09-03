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
