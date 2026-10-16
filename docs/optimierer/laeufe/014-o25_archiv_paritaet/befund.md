Maschinenlabels tragen im Proxy Information: die 86 O25-Archiv-Aufnahmen, die die Produktion seit 24.09. ausschliesst, stecken in K (12 % der Trainingszeilen), und sie herauszunehmen kostet im Proxy in 6 von 6 Laeufen AUC.

Daten: Trainingsteil 609 Aufn. (jede 4. s, 573k Zeilen), Haelften nach sha1(uuid), K-MLP 96/12 Ep. ohne OCR/SigLIP,
Bewertung auf nicht-maschinellen Aufnahmen der anderen Haelfte, Platt in-sample (fuer beide Arme gleich), Glaettung 8 Zeilen, Schwelle 0.5.
* Herkunft im Trainingsteil: merged 453, auto 147, auto-confirm 44, user 9; mensch_belegt=True nur 38.
  86 der 90 O25-uuids liegen im Training, 2 in Kalibrierung, 0 in Entwicklung. meta cluster_anchored ist bei ihnen True -> kein Meta-Kriterium, nur die Liste.
* 94 Trainingsaufnahmen ohne slug (79 ohne start_ts, uuid als Hash): andere Gruppe (merged 72), Cartoons/Sitcoms; nicht angefasst.
* O25 weg vs. basis, 2 Richtungen x 3 Seeds (Mittel):
  AUC 0.9674 -> 0.9659 (6/6 schlechter), Fehler 0.0302 -> 0.0305, Kante 0.258 -> 0.260, Median je Aufnahme 0.0176 -> 0.0181, Kinder-Sender 0.0468 -> 0.0470.
* Vorlauf (gleicher Proxy ohne Platt, 2x2 Seeds, zu verrauscht fuer kleine Effekte):
  alle 147 auto weg: Median je Aufn. 0.0178 -> 0.0230 (schadet); Gewicht 0 auf auto-Zeilen <=30/60 s an Label-Kanten: Kantenfehler 0.230 -> 0.231/0.230 (kein Hebel).
* Logo-Rueckkehr minus Label-Blockende (s, Median): Galileo korrigiert 0 (3 % <= -15), Galileo unkorrigiert 0 (5 % >= +15 = Countdown-Fall, 5 von 95 Kanten);
  kabel-eins auto -60 vs merged -5, vox auto -18 vs merged -22 -> Konventionsunterschied senderabhaengig, Logo allein kein Massstab (rtl -60 bei beiden).
Folgerung: Maschinenlabels liefern Backbone-Vielfalt, ihr Kanten-Versatz wirkt im Proxy nicht. Der Arm misst, was der O25-Ausschluss kostet.
Skripte: Scratchpad m1-m7.py, p1.py, p2.py (read-only, nur Summen).
