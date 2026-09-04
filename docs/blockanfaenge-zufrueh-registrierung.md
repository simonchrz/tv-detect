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

---

## Ergebnis 2026-09-04: UNGÜLTIG — ich habe die Konvention mitten im
## Versuch geändert

Die Bedingung („alle 5 Kontrollen 0 s") ist **nicht erfüllt**. Über alle
8 Kontrollen (5 gezogen + 3 Ersatz):

```
0 s: 4    unbrauchbar: 3    FALSCH POSITIV: 1  (f023, 16 s)
```

Ein falscher Positivbefund in der Kontrolle heißt: das Verfahren erzeugt
„zu früh", wo keines ist. Damit sind die Kandidatenzahlen wertlos und
werden hier **nicht** aufgeführt — sonst stünden sie im Ledger und würden
später zitiert.

### Die Ursache, und sie liegt bei mir

Dieselbe Kante, beide Läufe, direkt gegenübergestellt:

| Kante | 03.09. bei −4 s / 0 s | 04.09. bei −4 s / 0 s |
|---|---|---|
| `dvr-rtl-1780078500` @1460.0 | S / **W** | S / **S** |
| `dvr-rtlzwei-1779128100` @2958.6 | **S** / **W** | **W** / **W** |
| `dvr-rtl-1781545200` @1439.0 | S / W | S / W |

Zwei von drei widersprechen sich auf **denselben Bildern**.

⚠️ **Grund: ich habe den Agenten-Auftrag zwischen den Läufen geändert.**
Der Sweep vom 03.09. enthielt keine Regel für Einblendungen. Heute stand
darin: „eine Einblendung ÜBER dem laufenden Bild macht die Aufnahme nicht
zu Werbung, solange darunter die Sendung weiterläuft." Das ist eine
KONVENTIONSÄNDERUNG. `dvr-rtl-1780078500` ist Let's Dance — genau der
Overlay-Fall.

Das ist wörtlich der Fehler, den §3am beschreibt und den ich am 03.09.
selbst ins Ledger geschrieben habe: *„Wer eine Konvention ändert, muss im
selben Zug sagen, wer die Labels nachzieht — sonst misst ein späterer Lauf
die Lücke als Qualität."* Einen Tag später habe ich Screening und Messung
unter zwei verschiedenen Konventionen laufen lassen.

**Was gut lief:** die Kontrollen haben es gefangen, bevor eine Zahl im
Ledger stand. Genau dafür waren sie da.

### Was ein gültiger Neulauf braucht

Nicht einfach den alten Auftrag wiederherstellen — die neue Regel ist
inhaltlich die RICHTIGE (ein Voting-Balken über der laufenden Show ist
keine Werbung; ohne die Regel hält der Agent die Sendung für Werbung).
Also: **das Screening mit der neuen Konvention wiederholen**, dann messen.
Kosten: der 106er-Sweep noch einmal, auf Sonnet.

⚠️ **Und ein Vorbehalt zum 03.09.-Ergebnis:** jener Lauf war in sich
konsistent (ein Auftrag für alle 106), seine 16 % bleiben also intern
gültig. Aber er hat Overlays als Werbung gezählt. Das verschiebt
Übergänge nach vorn und kann „zu spät" überschätzt haben. Die Zahl ist
eine Obergrenze, kein Punktwert.
