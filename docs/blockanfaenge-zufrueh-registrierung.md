# Wie weit schneidet ein zu früher Blockanfang in die Sendung?

**Registriert 2026-09-04, VOR dem ersten Agenten-Lauf.**

## Anlass

Der Sweep vom 03.09. (`blockanfaenge-vorlauf-registrierung.md`) hat die
Gegenrichtung nebenbei sichtbar gemacht: 17 der 106 Blockanfänge zeigten
im Fenster −24…+4 s NACH der Kante noch Sendung, 11 davon über das ganze
Fenster. Der Block beginnt dort, während die Sendung läuft.

**Warum das die wichtigere Richtung ist:** ein zu SPÄTER Anfang lässt ein
paar Sekunden Werbung stehen. Ein zu FRÜHER schneidet Sendung weg — der
Fehler, den man beim Schauen merkt. Im selben Sweep standen 16 zu frühe
gegen 14 zu späte Kanten.

## Was hier NICHT gemessen wird

**Keine Rate.** Die Häufigkeit steht schon fest (17 von 106 mit der
Signatur). Diese Messung beziffert nur, wie WEIT hineingeschnitten wird.
Wer aus den Zahlen unten eine Rate ableitet, misst eine Auswahl.

## Aufbau

* **17 Kandidaten**: alle Kanten des 03.09.-Sweeps mit Urteil „unklar".
* **5 Kontrollen**: zufällig aus den „sauber" beurteilten Kanten. Sie
  MÜSSEN 0 s ergeben. Tun sie es nicht, misst das Verfahren etwas anderes
  als gemeint, und die Kandidatenzahlen sind wertlos.
* Fenster **−4 s bis +48 s** in 4-s-Schritten (14 Bilder), ungeblendet
  (die Frage hat nichts mit der OCR-Regel zu tun — dieselbe Auflage wie
  am 03.09.: solche Urteile dürfen NIE O13-Referenz werden).
* Klassifikation je Bild, Agenten auf **Sonnet** (Einzelbild-Urteil; das
  große Modell hat am 03.09. rund 3 Mio. Token ohne Mehrwert gekostet).

## Auswertung (vorab festgelegt)

Ab dem Bild AM Blockanfang vorwärts zählen, wie viele Bilder in Folge
SENDUNG sind. `zu früh um X s` = 4 s × diese Zahl.

* Das Bild bei −4 s dient als Gegenprobe: ist es bereits WERBUNG, ist die
  Kante nicht „zu früh", sondern der Block hat dort schon begonnen — die
  Kante zählt als **unbrauchbar**, nicht als 0.
* Findet sich bis +48 s keine Werbung, wird `≥48 s` notiert, nicht
  hochgerechnet.

## Bedingung

Rein deskriptiv, keine Ja/Nein-Schwelle: das Ergebnis ist die Verteilung
der Werte plus die Kontrollen. **Interpretiert wird nur, wenn alle 5
Kontrollen 0 s ergeben.**

## Was daraus folgen darf

Ein Kandidatenbericht an Simon. Kein Label wird geschrieben (L2).
