Die 36 Trainingsaufnahmen, in denen das Logo nicht trennt (7.5 % der Zeilen), tragen out-of-fold 39 % der Fehlerzeilen, fast alles erfundene Werbung; Gewicht 3 auf sie senkt im Proxy die Sekundenfehler in 4 von 4 Laeufen um 6.5 %.

Daten: Trainingsteil 527 Aufn. mit ads (jede 4. s, 506k Zeilen), Haelften nach sha1(uuid), K-MLP 96/12 Ep. ohne OCR/SigLIP,
OOF aus 2 Seeds je Richtung, Glaettung 8 Zeilen, Schwelle 0.5. Proxy-Bewertung auf user/merged-Aufnahmen der anderen Haelfte, 2 Richtungen x 2 Seeds.
* OOF-Fehler hat einen schweren Schwanz: die schlechtesten 20 Aufn. (4.5 % der Zeilen) tragen 41.5 % der Fehlerzeilen, ueberwiegend FP.
* Logo-Trennung je Aufnahme (AUC Logo-Abwesenheit gegen Label): <0.6: 15 Aufn., Fehler 0.264; 0.6-0.75: 21 Aufn., 0.160;
  0.75-0.9: 142, 0.049; 0.9-0.97: 214, 0.019; >=0.97: 135, 0.009.
* In den 36 logo-schwachen Aufn. ordnet das NN gut (Median-AUC 0.96), liegt aber als Ganzes auf der Werbeseite (Call Me Kat AUC 0.97 / Fehler 0.56,
  Lenssen hilft 0.98 / 0.68): kein Label-Versatz, sondern ein Pegel-Problem ohne Logo. Sender: rtl 14 (Beet-Brueder), ? 8 (Filme, Richter Hold), prosieben 5, disney 3, nick 1.
* Proxy (Mittel 4 Laeufe), Basis -> x2 -> x3 -> x5:
  Fehler 0.0352 -> 0.0337 -> 0.0329 -> 0.0327; FP 0.0248 -> 0.0234 -> 0.0226 -> 0.0226; FN 0.0104 unveraendert;
  logo-schwache Eval-Aufn. 0.209 -> 0.192 -> 0.182 -> 0.177; logo-starke mittel 0.0237 -> 0.0233 -> 0.0232 -> 0.0235, Median 0.0128 -> 0.0129 -> 0.0130 -> 0.0131;
  AUC gepoolt 0.9709 -> 0.9702 -> 0.9696 -> 0.9686. x5 kippt den Median je Aufnahme (-0.0013..+0.0015), daher Faktor 3.
* Verworfen vor dem Arm (Klasse daten):
  - Altersrampe der Produktion (age_mult) nachbilden: Entwicklung/Bestaetigung sind 88-158 Tage alt, Training Median 82 Tage -> die Rampe
    gewichtet weg von der Eval-Zeit, Messung waere verzerrt. Nebenbefund: die Produktion gibt Archiv-Aufn. ohne start_ts Alter 0 = volles
    Gewicht; das sind die 65 tvh-Hash-uuids, die aeltesten Daten (Label-Raster dort geprueft: kein Versatz, Fehler 0.045 vs 0.039).
  - Label-Hygiene der Produktion (Lehrer in-sample, conf 0.9, nur auto): streicht nur 0.7 % der auto-Zeilen, Proxy-Fehler +0.0001 -> inert.
  - K laedt keine Aufnahmen ohne ads: 44 auto-confirm (17.6k Zeilen, u. a. 9 disney, 9 rtl, toggo-plus, nick, kika), die die Produktion als
    reine Sendung trainiert; dazu 94 Trainings-uuids ohne Archiv-Meta. Ein Arm kann das nicht nachholen (Abbild-Luecke, Hinweis an Simon).
Risiko: Wirkung haengt am Anteil logo-schwacher Aufn. im Entwicklungssatz (nicht angesehen); Gewinn auf der typischen Aufnahme ist im Proxy ~0.
Skripte: Scratchpad h1-h10.py (read-only, nur Summen).
