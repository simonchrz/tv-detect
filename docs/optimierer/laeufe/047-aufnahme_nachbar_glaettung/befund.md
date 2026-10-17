Den Kopf-Logit 30 % zum Mittel der bildähnlichsten Sekunden derselben Aufnahme zu ziehen (Abstand ≥ 45 s, Kosinus ≥ 0.5), hebt im Produktions-HSMM-Proxy die Block-IoU in 2/2 Seeds (+0.0031 / +0.0019, Median +0.0027 / +0.0012) bei nur +4 s Werbezeit je Aufnahme, ist also kein Schwellen-Tausch.

Klasse nachbearbeitung (Bandit, Zug 0.76). Bisherige Nachbearbeitungen sind skalar oder zeitlich (Kappe 017, Nachhall 031, Veto 035, Ton/Bild-Fenster 036/037, Tal-Schwelle 038, Kalibrier-Versatz abgelehnt). Diese ist die erste, die über Bildgleichheit innerhalb der Aufnahme glättet. 011/018/021 hatten Selbstähnlichkeit als Spalte; hier wandert stattdessen das eigene Urteil des Kopfes über den Graphen.
Proxy: Scratchpad px.npz (out-of-fold Logit-Mittel aus 3 K-Köpfen, 134 Trainingsaufnahmen mit Signal-Dump, 132 davon heute noch mit Dump, dazu 34 Kalibrieraufnahmen). Graph aus zentriertem, normiertem Archiv-Backbone, je Sekunde top-16, Zeitabstand ≥ 45 s. Danach opt_lauf.platt auf der ebenso behandelten Kalibrierung, Produktions-HSMM, IoU gegen meta-ads. Skripte: Scratchpad nb/graph.py, nb/d1.py, nb/p1.py.
Nachbar-Reinheit (Label des Nachbarn gleich dem eigenen): Kosinus 0.3–0.5 86 %, 0.5–0.7 95.8 %, 0.7–0.85 97.9 %, > 0.85 97.8 %. Ab 0.5 haben 61 % der Sekunden mindestens einen Nachbarn.
Sekundenfehler bei Schwelle z > 1.7: 4.21 % → 4.13 % (Seed 0) bzw. 4.15 % → 4.04 % (Seed 1). Fast nur FP: −452 / −665, FN +38 / +61.
Δ IoU-Mittel S0 | S1 (Basis 0.9321 / 0.9317), dazu Aufnahmen besser/schlechter (> 0.005) und Minimum:
* tau 0.5, a 0.3 (Arm): +0.0031 | +0.0019, 17/5 | 14/7, min −0.030 | −0.034, Werbezeit +0.17 h | +0.11 h auf 132 Aufnahmen
* tau 0.5, a 0.2: +0.0030 | +0.0016. tau 0.5, a 0.4: +0.0034 | +0.0013 (mehr Streuung, 20/10)
* tau 0.6, a 0.3: +0.0026 | +0.0004. tau 0.4, a 0.3: +0.0022 | +0.0023, aber min −0.18
* a 0.5 bei tau 0.6/0.7: eine Aufnahme kippt (min −0.30 bis −0.33), Mittel ≤ 0. Nur Nachbarn mit |z| ≥ 3: +0.0011 | +0.0002
Die Fläche ist flach um tau 0.5 / a 0.2–0.4. Der Arm liegt in ihrer Mitte und ist nicht auf den besten Punkt geschnitten.
Risiko: Der Gewinn liegt mit +0.002..+0.003 an der Gewinnschwelle. Ein zu großes a kippt einzelne Aufnahmen: Ein Werbeblock voller ähnlicher Spots zieht dann Sendung mit.
Rechenzeit: Der Graph entsteht einmal je Teil in spalten() (Kalibrierung und Eval, je Sekunde, BLAS-Matmul je Aufnahme), nicht je Seed. Im Proxy brauchte er einige Minuten für 700k Zeilen.
