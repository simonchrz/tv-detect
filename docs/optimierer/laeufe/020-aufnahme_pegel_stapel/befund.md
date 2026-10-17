Die ehrliche (out-of-fold) Bildmeinung der ganzen Aufnahme als linearer Pegel-Kontext in einem Stapel senkt im Proxy die Sekundenfehler um ~9 % gegen Stufe 1 und ~12 % gegen den 006-Stapel, ohne den Aufnahme-Median zu senken.

Daten: Trainingsteil 609 Aufn. (jede 4. s, 506k Zeilen, K-Spalten via opt_lauf.lade/baue_k). Stufe 1 = K-MLP 96/12 Ep.,
4 Falten nach Aufnahme (wie 006), OOF-Logit z. Stufe 2 haelftig nach sha1(rec), beide Richtungen, Platt auf der Trainingshaelfte,
Glaettung 9 Zeilen, Schwelle 0.5; Seeds 0/1/2. Logo-blind = AUC(Logo-Abwesenheit gegen Label) < 0.75 je Aufn. (36 Aufn., 7.5 % der Zeilen).
* Label-freie Aufnahme-Kontexte gegen logo-blind (AUC): corr(z, Logo) 0.926, NN-Anteil bei sichtbarem Logo 0.827, Median z 0.779,
  Logo-Mittel 0.30 (also 0.70 invertiert, das war 009s Information), NN-Anteil 0.56.
* Fehler / FP / FN / logloss / Aufn.-Median / logo-blind / logo-stark (Seed 0; Seeds 1,2 gleiches Bild):
  Stufe 1 + Platt      0.0295 / 0.0117 / 0.0178 / 0.160 / 0.0122 / 0.136 / 0.0208
  Stapel 006           0.0306 / 0.0124 / 0.0181 / 0.155 / 0.0135 / 0.138 / 0.0218
  Pegel linear (Arm)   0.0274 / 0.0070 / 0.0204 / 0.141 / 0.0121 / 0.113 / 0.0204   (s1 0.0265, s2 0.0270)
  Pegel im MLP (voll)  0.0262 / 0.0070 / 0.0192 / 0.136 / 0.0122 / 0.099 / 0.0203   (s1 0.0268, s2 0.0271)
* Ablation (Pegel linear): nur Logo-Mittel 0.0295 (= nichts, wie 009), nur corr 0.0339 (schlechter), nur NN-Anteil+Median
  0.0299/0.0251/0.0263 (instabil), alle 5 stabil 0.0265-0.0274. Die Information sitzt im NN-Pegel der Aufnahme, corr stabilisiert.
* Je Aufnahme (Seed 2, Schwelle 0.005): gegen Stufe 1 59 besser / 70 schlechter, schlimmste +0.075, beste -0.30;
  gegen 006 53 besser / 30 schlechter. Der Gewinn kommt wieder aus einem Schwanz (FP halbiert), FN steigt leicht.
* Linear statt MLP gewaehlt, obwohl "voll" im Proxy gleich gut ist: Stufe 2 trainiert im Lauf auf allen 609 Aufnahmen; 5 Konstanten
  je Aufnahme im MLP koennten die Aufnahme-Kennung auswendig lernen, linear nicht.
Anders als Friedhof: 009 gab Logo-Verteilung in den in-sample-Hauptkopf (kein Gradient, Proxy hier = 0); 006 hatte keinen
Aufnahme-Kontext und senkte den Aufnahme-Median; 005 zentrierte Merkmale statt den Pegel zu verschieben; 016 gewichtete nur.
Risiko: HSMM holt einen Teil; Teleshopping-/Dauerwerbe-Aufnahmen haben hohen NN-Pegel zu Recht (der Arm koennte sie abwerten);
Stufe 1 ist je Lauf fest (2 feste Seeds je Falte), die Wiederholungen streuen nur Stufe 2.
Skripte: Scratchpad p3/d1.py (OOF), d2.py/d3.py (Proxy, Ablation), read-only, nur Summen.
