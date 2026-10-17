Klasse labels gab nichts her (Kantenlage je Sender identisch zur Entwicklung, OCR-Werbemarker in Sendung gleich haeufig); in training traegt als einziger von fuenf Mechanismen eine Backbone-Schrittweiten-Drossel (x0.35): AUC gross->klein +0.002 bis +0.004, Logo-FP -20 %.

Daten: Trainingsteil von K (saetze(), ohne O25), 527 Aufn. mit ads, jede 4. s, 506k Zeilen. Read-only, nur Summen.
Labels (gezogen, Zug 0.86):
* Blockende gegen Logo-Rueckkehr (Logo-Aufnahmen, 3-s-Glaettung, 5 s stabil): merged gesamt Median -4 s, Mensch-Entwicklung 0 s.
  Je Sender aber gleich: vox -20/-24, kabel-eins -4/-3.5, prosieben 0/+2, rtlzwei -1/-2 (train/entw) -> nur Senderzusammensetzung.
  Gepaart: 38 Trainingsaufn. mit Menschlabel, Archiv-Label == Menschlabel (51/51 Kanten exakt).
* OCR-Werbemarker (exakte time_s) mitten in Sendung (>30 s von jeder Kante): train 0.78, entw 1.40 je 1000 abgetastete s;
  Spalte P(w=1|Sendung) train 1.04 %, entw 1.14 % (Ueberschuss entw zu 41 % aus einer Aufnahme). Menschen labeln das genauso -> kein Hebel.
* Werbebloecke am Aufnahmeende: 41 Aufn., Median 200 s -> zu klein.
Domaenen-Luecke: nick/comedy-central/sixx/disney/super-rtl/toggo = 29 % der Bewertungsaufnahmen, ~4 % der Trainingszeilen (vox 28 %).
Proxy (K-MLP 96, 8 Ep., Backbone+Logo+Audio, ohne OCR/SigLIP; gross->klein = Train vox/k1/p7/rtl2/rtl/'' -> Test kleine Sender;
Haelften nach sha1(uuid)), Werte AUC / Fehler-Median je Aufn.:
* Basis: g->k 0.937/0.122, H0->1 0.966/0.048, H1->0 0.963/0.049.
* Stil-Tausch (Aufnahme-Mittel des Backbones gegen fremdes, p 0.5): alle drei schlechter (g->k 0.932/0.142) -> verworfen.
* Bagging je Aufnahme statt Seed-Ensemble (3 Koepfe, 67 %): neutral bis schlechter -> verworfen.
* Backbone L2- bzw. Zeilen-Normierung: neutral (Norm-AUC allein 0.517) -> verworfen.
* Sender-adversarial (Gradient-Umkehr, lambda 0.1/0.3): schlechter in allen Teilen -> verworfen.
* Backbone-Skala 1.0/0.5/0.35/0.2/0.1 (einzeln): g->k AUC 0.937/0.940/0.942/0.944/0.946 monoton; Haelften am besten bei 0.3-0.5,
  ab 0.2 steigt der Fehler-Mittelwert (H1->0 0.083 -> 0.086/0.089).
* 3er-Ensemble 1.0 vs 0.35, je 2 Wdh.: g->k AUC 0.943/0.940 -> 0.944/0.944, Median-Fehler 0.113/0.122 -> 0.108/0.110;
  Haelften AUC +-0.001, Median-Fehler gemischt; P(Werbung | Logo>=0.8, p>=0.8) 0.73-0.76 -> 0.80-0.82, Werbe-FP bei Logo>=0.8 0.011 -> 0.0085.
Risiko: Proxy ohne OCR/SigLIP; die Wirkung liegt bei den kleinen Sendern (nick, comedy-central, super-rtl tragen jedes Urteil). Typische vox/rtlzwei-Aufnahme koennte etwas Backbone-Schaerfe verlieren.
Skripte: Scratchpad p2/m1-m8.py (Labels), p1-p6.py (Proxy).
