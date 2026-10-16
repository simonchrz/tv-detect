Maschinen-Labels (which=auto) ziehen Werbeblock-Enden rund 18 s weiter in die Sendung als Menschen-Labels; das Zurückziehen auf Logo-Rückkehr + 6 s ist im Proxy aber neutral, und fünf Nachbearbeitungen scheitern alle im Proxy.

Daten: Trainingsteil, 239 Aufnahmen mit Merkmalen (Zufallsauswahl 300, Seed 0), Proxy-Kopf = K-MLP auf Backbone+Logo+Audio,
2 Hälften nach Aufnahme, out-of-fold auf dem 1-s-Raster, global Platt; Dekoder-Ersatz = Viterbi mit 2 Zuständen, Wechselstrafe 40.
Basis: Fehler 0.0353, Aufnahme-Median 0.014; Kante 7210 / erfindet 8304 / verpasst 13483 s.

Kantenlage (erstes Kippen für 5 s, Fenster ±40 s), Median (q25/q75):
  Sendung->Werbung: merged +2 (-1/8), auto 0 (-8.5/9)
  Werbung->Sendung: merged -5 (-32/-1), auto -24 (-40/-1); Logo-Rückkehr: merged -6, auto -24 (gleiche Sender: prosieben/vox/kabel-eins/rtlzwei)
Arm-Regel trifft 0.64 % der Werbezeilen (343 Zeilen à 4 s). Proxy auf Mensch-Aufnahmen, 2 Seeds:
  Fehler 0.0250/0.0244 -> 0.0254/0.0244, Kante 4461/4453 -> 4485/4589, besser/schlechter 8/8 und 3/18 -> neutral bis leicht negativ.

Nachbearbeitung (gezogene Klasse), alle im Viterbi-Proxy, Fehler / Aufnahme-Median gegen 0.0353 / 0.014:
  Prior-Shift je Aufnahme (EM, Dämpfung 0.25-1): 0.045-0.057 (Korrelation geschätzter/wahrer Werbeanteil nur 0.21) -> tot
  kNN-Glättung in der Aufnahme (Backbone-Kosinus, ±60 s gesperrt): 0.038-0.042; mit Ähnlichkeits-Gate null
  Vorwärts-Halten der Werbe-Evidenz 4-30 s: 0.042-0.071 (erfindet x2.5)
  Logo-Logit neu kombiniert (kreuz-gefittet): Mittel 0.0316, aber Median 0.0174, 40 besser / 98 schlechter; Gewinn fast nur disney-channel, vox verliert
  Mitteln über Szenen (Backbone-Schnitte q80-q95): 0.0354-0.041
  OCR: ocr_werbung >= 0.5 nur 78 % Werbung, Proxy schon 85 % p>0.5 -> kein Hebel
Logo >= 0.8 und p >= 0.8 im Proxy: 74 % Werbung (nicht 53 %) -> Logo-Übersteuerung wäre meist falsch.
Klasse gewechselt auf labels (zweithöchster Zug), weil keine Nachbearbeitung im Proxy den Aufnahme-Median hielt.
Skripte: Scratchpad diag_pp.py, diag_s1.py, ana1-9.py (read-only, nur Summen).
