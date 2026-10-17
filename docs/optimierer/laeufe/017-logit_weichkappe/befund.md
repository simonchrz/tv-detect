Eine weiche Kappe auf die Kopf-Logits (c*tanh(z/c), c = Median |logit|) senkt im Proxy den Aufnahme-Median-Fehler um rund 10 % bei unverändertem Mittel; sie wirkt nur bei starkem Dekoder und fast ganz über weniger verpasste Werbung.

Daten: Trainingsteil 609 Aufn. (jede 4. s, 506k Zeilen), OOF-Kopf aus Scratchpad h3.py (K-MLP auf Backbone+Logo+Audio,
Hälften nach sha1(uuid), 2 Seeds), globales Platt, Dekoder-Ersatz Viterbi 2 Zustände auf 4-s-Zeilen, Wechselstrafe lam;
bewertet auf user/merged-Aufnahmen (Fehleranteil je Aufnahme). Median |logit| = 4.36; 56 % der Zeilen haben |z| > 4.
Fehler Mittel / Median (besser/schlechter Aufnahmen, Schwelle 1e-4):
  lam 5 : basis 0.0257/0.0098, harte Kappe 4: 0.0266/0.0098 (83/82)   -> bei schwachem Dekoder schädlich
  lam10 : basis 0.0250/0.0083, tanh c=med: 0.0249/0.0074 (90/50), hart c=med 0.0246/0.0075 (70/39)
  lam20 : basis 0.0244/0.0076, tanh: 0.0241/0.0069 (88/49)
  lam40 : basis 0.0245/0.0077, tanh: 0.0247/0.0071 (87/46)
Fehlerarten (lam10, tanh): fp 4050 -> 4243, fn 5915 -> 5679: weniger verpasst, etwas mehr erfunden.
Nur die Werbe-Seite kappen (min(z,4)) trägt den ganzen Effekt (0.0248/0.0073, 91/50); nur die Sendungs-Seite: neutral/schlechter.
Logo-gegatete Kappe (nur bei Logo über Median): neutral bis schlechter (logoclip2 0.0253/0.0092).
Quantil-Kappen weit oben (q80 6.4, q90 7.5): wirkungslos (17/11, 6/5) -> der Effekt kommt aus der Stauchung des mittleren Bereichs
durch das nachfolgende Platt, nicht aus dem Abschneiden einzelner Ausreißer.
Risiko: Produktions-HSMM (dur_w 15, inner_show_min 90) ist stark, das spricht für den Arm; K-Logits (3 Köpfe, OCR/SigLIP) können
anders skaliert sein, daher c relativ aus dem Kalibrierteil. Mittel neutral -> Gewinnschwelle +0.003 ist knapp erreichbar.
Skripte: Scratchpad nb1.py, nb2.py, nb3.py (read-only, nur Summen).
