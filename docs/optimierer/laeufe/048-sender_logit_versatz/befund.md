K liegt out-of-fold je Sender systematisch und haelftenstabil daneben, auch im Blockinneren (kabel-eins/rtlzwei verpassen, prosieben/disney erfinden); ein halbierter Logit-Versatz je Sender hebt im Produktions-HSMM-Proxy die Block-IoU in 2/2 Seeds (+0.004/+0.003) ohne Aufnahme unter -0.025.

Klasse nachbearbeitung (Bandit, Zug 0.45). Daten: Trainingsteil von K (saetze(), 527 Aufn. mit ads, jede 4. s, 506k Zeilen),
OOF-Kopf = K exakt (alle 1350 Spalten, MLP 96, 12 Ep., 3er-Logit-Ensemble), Haelften nach sha1(uuid), 2 Seeds. Read-only.
* Zellen (ocr_hinweis x ocr_werbung x Logo-Stufe): ueberall gleichgerichtet ueberschaetzt (Klassengewicht) -> nur global, kein Hebel.
* Kanten-Versatz OOF gegen Label: Median 0 s an Start und Ende (merged/auto/user) -> keine Zeitverschiebung.
* Logloss-optimaler Versatz je Sender nach globalem Versatz (Haelfte 0/1), Inneres = > 16 s von jeder Labelkante:
  kabel-eins (95 Aufn.) +1.12/+1.39, Inneres +1.12, FN 1964 gegen FP 290
  rtlzwei (33) +1.22/+1.02, Inneres +1.03, FN 1757 gegen FP 289
  vox (127) +0.44/+0.47 | prosieben (114) -0.79/-0.48, Inneres FP 2694 gegen FN 1288 | disney (27) -1.20/-0.81
  rtl (32) +0.63/-0.33, sat-1-gold (5) +0.36/-2.38, ohne slug (65) -2.02/+0.48 -> instabil, Versatz 0.
  Nur merged-Aufnahmen: kabel-eins +1.11, rtlzwei +0.80, prosieben -0.73 -> keine Herkunfts-Konvention.
* Produktions-HSMM-Proxy: 132 Trainingsaufn. mit altem Dump (emit-signals), OOF-p auf Sekunden gedehnt, Versatz-Tabelle
  und Platt nur aus der jeweils ANDEREN Haelfte, Block-IoU gegen Trainingslabel:
  Faktor 0.0: Mittel 0.9314/0.9314, Median 0.9817/0.9812
  Faktor 0.5: Mittel 0.9354/0.9344, Median 0.9834/0.9838; gepaart 26:10 / 21:14 Aufn., schlechteste -0.022/-0.025,
              ohne die 2 besten +0.0011/+0.0009; je Sender kabel-eins +0.010/+0.007 (40), disney +0.005/+0.007 (8)
  Faktor 1.0: Mittel 0.9316/0.9343, schlechteste -0.446 (disney) -> zu stark.
Arm: Tabelle aus OOF beider Seeds, nur Sender >= 20 Aufn. mit gleichem Vorzeichen in beiden Haelften, x0.5.
Abgrenzung O7 (Kanal als Spalte, -0.0005): dort lernt der Kopf die Lage in-sample; die Verzerrung zeigt sich erst ungesehen.
Risiko: Entwicklung hat kabel-eins 9, rtlzwei 15, prosieben 15, vox 7 -> Wirkung haengt an der Uebertragung der Tabelle;
der volle K (doppelte Daten) koennte kleinere Verzerrung haben als der Haelften-Proxy.
Skripte: Scratchpad oof.py, cell.py, edge.py, slug.py, slug2.py, hsmm.py (nur Summen).
