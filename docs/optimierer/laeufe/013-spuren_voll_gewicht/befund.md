Die Trainingslabels geben kaum Hebel her: lange Widersprüche zum Kopf machen 0.07–0.35 % der Zeilen aus, Trailer am Blockende sind zu 92 % einheitlich gelabelt. Größer ist eine Regime-Lücke: nur 27 % der Trainingszeilen tragen OCR- und SigLIP-Spur, bei der Bewertung sind es 100 %.

Klasse labels gezogen und geprüft (Trainingsteil, 609 Aufn., read-only):
* Confident Learning: 3 Falten nach Aufnahme. Falte C sagt Falte A voraus, markiert werden Läufe ≥ 32 s mit p > .95 gegen y=0 bzw. p < .05 gegen y=1, mehr als 120 s vom Aufnahmerand entfernt.
  Markiert sind 0.19 / 0.07 / 0.35 % der Zeilen. Der Fehler auf B ändert sich mit flip um −0.0004 / +0.0006 / +0.0026, mit drop ähnlich. Das ist Rauschen, die Labels sind sauber.
* OCR-Programmhinweise (Rohfunde, 181 Aufn. mit Spur) im Abstand ±60 s zum Blockende:
  4496 Treffer liegen im Label-Werbeblock, 128 in 0–30 s danach als Sendung. Die Konvention ist also zu 92 % einheitlich.
  Am Blockstart liegen 116 Treffer davor als Sendung und 1666 im Block.
* Werbemarker in Sendung fern einer Kante (≥ 15 s): 130 Läufe mit 3659 s, davon 973 s mit Logo < 0.3, meist prosieben. Das ist zu klein für eine Regel.
* Rand und Nachlauf sind laut Ledger 0.25 % (schon gemessen).
→ Kein Label-Arm, der die alten Messungen überträfe. Ich weiche auf die Klasse daten aus (zweithöchster Zug).

Abdeckung der Spuren (Aufnahmen mit Merkmalen; OCR gültig per spur_passt, SigLIP-Datei vorhanden):
  Training: 653 Aufn., davon 189 mit OCR und 172 mit SigLIP. Beide Spuren laut Indikator: 161 von 609 geladenen = 27.3 % der Zeilen.
  Kalibrierung: 7 von 35 mit OCR, 6 mit SigLIP. Entwicklung 76/76, Bestätigung 22/22.
  Der Werbeanteil ist bei beiden Gruppen gleich (0.227 bespurt, 0.244 Rest).

Proxy für Gewicht ×4 auf voll bespurte Aufnahmen: 2 Hälften nach Aufnahme, MLP 96 auf allen K-Spalten, 12 Epochen.
Gemessen wird der Sekundenfehler auf den voll bespurten Aufnahmen der anderen Hälfte, je Richtung 3 Seeds (bei Faktor 6 nur 2 Seeds).
  A0→B: Faktor 1 0.0859 (0.079–0.092), Faktor 4 0.0827, Faktor 6 0.0836
  A1→B: Faktor 1 0.0313, Faktor 4 0.0295, Faktor 6 0.0281
  Kante (≤ 15 s) sinkt leicht mit, der Median je Aufnahme bleibt flach (0.021–0.024).
  Nur bespurte Aufnahmen (Rest ausgeschlossen): Median je Aufnahme 0.034 statt 0.022, schadet also.
Vorbehalt: Die Streuung zwischen Seeds ist so groß wie der Effekt. Die Platt-Kalibrierung bleibt im alten Regime (nur 7 von 34 Aufnahmen bespurt).
