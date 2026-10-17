Von sechs Trainingsverfahren trägt im Proxy nur Segment-Mixup (Mischen mit einem Nachbarn gleichen Labels ±4..32 s): der 60-s-geglättete Fehler sinkt in 4 von 4 Läufen um ~2.5 %, der Einzelsekunden-Fehler steigt leicht. Alle Verlust- und Ziel-Varianten scheitern.
Daten: Trainingsteil von K (saetze(), ohne O25), 527 Aufn. mit ads, jede 4. s, 506k Zeilen, volle K-Spalten (1350). Read-only, nur Summen.
Proxy: Hälften nach sha1(uuid). Fit auf A mit opt_lauf.trainiere (HYPER_K, 2 Köpfe), Platt auf 1/4 der Aufn. von B, Kennzahlen auf dem Rest von B.
Kennzahlen: AUC, err1 = Einzelsekunde, err15 = Logit über 15 Zeilen (60 s) geglättet, als Ersatz für den HSMM.
Spalten unten: AUC / err1 Mittel / err15 Mittel / err15 Median (Mittel über 2 Hälften × 2 Seeds):
* K                              0.9762 / 0.0530 / 0.0327 / 0.0156
* Segment-Mixup p0.5 lam0.5..1   0.9758 / 0.0539 / 0.0319 / 0.0153  (err15 Mittel 4/4 besser, Median 4/4 besser oder gleich)
* Segment-Mixup p1.0 lam0..1     0.9744 / 0.0561 / 0.0312 / 0.0155  (Fenster besser, Einzelsekunde klar schlechter)
* Fenster-Verlust (Training auf 0.5·Logit + 0.5·Nachbar-Mittel): roh-AUC −0.010, Einzel-Logits frei -> verworfen
* Focal γ2: vor Platt 7× weniger sichere Fehler, nach Platt weg (AUC −0.001) -> Platt hebt reine Skalen-Effekte auf
* GCE-artig (Gradient · p_t^q): Köpfe sättigen, AUC schlechter -> verworfen
* Weiche Kanten-Ziele (y -> 0.5, Abfall exp(−d/8 s), 5.8 % der Zeilen): Fehler ±30 s an Kanten +7 % (err1) / +10 % (err15) -> verworfen
* Mehr Optimiererschritte (batch 1024/2048 statt 8192; die Produktion macht ~200k Schritte, K ~740): AUC −0.001..−0.002 -> kein Hebel für die Abbild-Lücke
* Hygiene wie die Produktion (Lehrer in-sample, conf 0.9, user/merged ausgenommen): nur 0.06 % der Zeilen fallen weg, Kennzahlen gleich -> Hygiene erklärt die Abbild-Lücke nicht
Wichtig fürs Messgerät: ol.platt (Newton ohne Dämpfung, Start a=1) DIVERGIERT, wenn ein Kopf stark gesättigte Logits liefert (|z| ~12-14 bei vielen Zeilen).
Dann ist q durchweg 0/1 (beobachtet bei Fenster-Verlust, GCE, batch 1024). Arme, die Köpfe schärfen, können daran scheitern statt an der Idee.
K selbst hat 0.8-2.9 % der Eval-Sekunden an der Klemme ±13.8 (logit, 1e-6), konvergiert aber.
Risiko: Der Proxy hat keinen HSMM. Der Gewinn liegt nur im geglätteten Fehler, die Einzelsekunde wird leicht schlechter. Der HSMM liest eher das Geglättete.
Skripte: Scratchpad fv/proxy*.py, hyg.py, bat*.py, mix.py.
