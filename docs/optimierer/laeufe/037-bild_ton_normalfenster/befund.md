Werbeminuten liegen im 60-s-Fenstermedian der Backbone-PCA-Abweichung in einem engen Band (wie der Ton in 036); Bild- und Ton-Term zusammen heben im Produktions-HSMM-Proxy die Block-IoU in 2/2 Seeds um +0.0043 bei steigendem Aufnahme-Median, der Bild-Term allein ist instabil.

Klasse nachbearbeitung gezogen (Zug 0.63), darin geblieben.
Daten: Trainingsteil saetze() (527 Aufn., jede 4. s), volle K-Spalten (baue_k), K-MLP 3er-Logit-Mittel, out-of-fold je sha1-Haelfte, 2 Seeds.
HSMM-Proxy: 150 Trainingsaufnahmen mit altem Signal-Dump (emit-signals), K auf den uebrigen 377 trainiert (standardisiert wie K), je Sekunde, opt_lauf.platt auf dem Kalibrierteil, Produktions-HSMM ueber opt_lauf.replay, IoU gegen meta-ads.
Sackgassen in dieser Runde (Proxy, Summen):
* Logo je Aufnahme normiert (Median Logo in K-sicherer Sendung/Werbung, transduktiv): Sekundenfehler auf geglaettetem z -2..-4 %, aber NUR ueber "Logo fehlt -> Werbung"; "Logo da -> Sendung" ist null (Koeffizient 0). Aufnahme-Median sinkt, 53-66 besser / 98-116 schlechter. HSMM-Proxy -0.035 / +0.006 (Platt instabil). Bestaetigt das Muster aus auto_blockende_logo.
* OCR im Schwanz zu frueh endender Bloecke: nur 24 solche Bloecke in OCR-bespurten Aufnahmen, ocr_werbung 14 % gegen 5 % danach -> kein Hebel.
* Phantombloecke kosten die Block-IoU nichts (block_iou zaehlt je Wahrheitsblock) -> Veto-/Stoerungs-Arme sind in dieser Metrik nahe null (vgl. 035 exakt 0).
Backbone-Ausreisser je Sekunde (PCA-64 aus Trainingsteil): Fehlerrate K steigt ueber die Dezile von maha 0.045 -> 0.126, Rest rerr 0.050 -> 0.117.
Laeufe >= 60 s (K, 31-s-Mittel > 0): Median rerr echt q10/q50/q90 97.8/106.2/114.8, erfunden 79.5/104/160.4; maha echt 62.1/66.0/70.4, erfunden 53.6/78.2/93.4.
60-s-Fenstermedian in Werbezeilen: rerr 85.7/105.5/128.6 (Sendung 67/89.6/119.5), maha 55.7/65.7/77 (Sendung 46.4/59.1/74.3).
LR auf OOF-Logit (z-Koeff. ~1.03): Ar -0.25..-0.28, Am -0.21..-0.53, Am*z+ -0.06..-0.20 -> fest -0.27/-0.35/-0.12.
Sekundenfehler auf geglaettetem z (LR, kreuz): Ton allein 0.0367->0.0337 | 0.0440->0.0371, Bild+Ton 0.0324/0.0315 | 0.0374/0.0365.
HSMM-Proxy, Delta IoU-Mittel (besser/schlechter > 0.005, min), Seed 0 | Seed 1, Basis 0.9330/0.9326, Median 0.984/0.988:
  Bild allein: +0.0036 (16/10, -0.058) | +0.0004 (10/17, -0.326), Median sinkt in Seed 1
  Ton (=036): +0.0033 (23/8) | +0.0022 (16/8)  -- 036 mass auf Entwicklung +0.0013, Proxy also ~2x optimistisch
  Bild+Ton (Arm): +0.0043 (24/8, -0.086) | +0.0043 (24/12, -0.034), Median 0.984->0.989 | 0.988->0.989, Werbezeit -2 %
Abgrenzung: 015 bildwechsel_dichte = Bildaenderung (Abstand zu t+-4 s) als Merkmal; hier die LAGE des Minutenmittels im PCA-Raum, als Nachbearbeitung.
Risiko: Ton-Term ist 036 unveraendert (dort +0.0013, 3/4 Teile); der Arm steht oder faellt mit dem Bild-Zuschlag (~+0.001..+0.002 im Proxy).
Skripte: Scratchpad r3/oof.py, o1-o3.py (OOD, LR), e1.py/e3.py (HSMM-Proxy-Daten), e4.py (Varianten), a1-a4.py (Logo), t1.py (OCR-Schwanz).
