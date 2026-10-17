Der Logo-Wert in Sendung liegt je Sender auf ganz anderer Skala (disney 0.53, nick 0.33, comedy-central 0.29 gegen 0.8-1.0 bei vox/prosieben/kabel-eins), und K kennt den Sender nicht; eine sender-feste Logo-Skala aus dem Trainingsteil senkt im Proxy den Aufnahme-Median-Fehler in 2/2 Seeds um ~7 %.

Klasse labels gezogen (Zug 0.72, siebter Anlauf), geprueft, read-only, Trainingsteil von K (saetze(), 527 Aufn. mit ads, jede 4. s, 506k Zeilen), nur Summen:
* Programmhinweis (ocr_hinweis) in Sendungs-Labels > 20 s von jeder Kante: 1351 Zeilen = 0.27 %, Logo-Mittel 0.70 -> Einblendungen in der Sendung, kein Label-Fehler.
* Kinder-/Comedy-Sender: nick 12, disney 26, comedy 8 Aufn., je genau ein Block (Episoden ~25 min), Werbeanteil plausibel; super-rtl/toggo/kika nur leere auto-confirm (lade() nimmt sie nicht).
* Audio als Ausweich-Quelle: AUC 0.549 (je Aufnahme zentriert 0.555) -> tot.
-> Ausweichen auf merkmale (zweithoechster Zug 0.46).

Logo je Sender (Median Sendung / Werbung, Logo-AUC): vox 0.89/0.07 0.893, prosieben 0.80/0.03 0.923, kabel-eins 1.00/0.05 0.947, rtlzwei 1.00/0.08 0.953,
rtl 0.97/0.15 0.753, disney 0.53/0.08 0.933, nick 0.33/0.07 0.854, comedy-central 0.29/0.10 0.810, ohne slug 0.85/0.05 0.874.

Proxy: K-MLP 96, 6 Ep., auf ALLEN K-Spalten (baue_k), Haelften nach sha1(uuid), OOF, Glaettung 9 Zeilen, Schwelle 0.5; Sender-Statistik nur aus der Trainingshaelfte.
                  AUC            Fehler         Median je Aufn.   disney        comedy        rtl           ohne slug
  K               0.9702/0.9705  0.0443/0.0423  0.0153/0.0148     0.187/0.173   0.074/0.047   0.076/0.082   0.067/0.065
  Arm (2 Sp.)     0.9723/0.9729  0.0430/0.0413  0.0139/0.0140     0.076/0.081   0.089/0.052   0.083/0.087   0.080/0.072
  nur logo_sender 0.9718/0.9721  0.0428/0.0424  0.0143/0.0132     0.154/0.132   0.078/0.050   0.076/0.084   0.070/0.066
  Sender one-hot  0.9737/0.9741  0.0435/0.0401  0.0131/0.0130     0.200/0.173   0.034/0.028   0.113/0.112   0.058/0.053
  one-hot + Arm   0.9746/0.9750  0.0427/0.0414  0.0126/0.0129     0.170/0.142   0.032/0.029   0.106/0.112   0.063/0.061
* One-hot traegt mehr AUC, verschlechtert aber rtl (Logo-Verstecken, Beet-Brueder) um +0.03 und hebt den disney-Gewinn auf -> nicht gewaehlt.
* Die Konstante sender_show ist noetig (ohne sie halbiert sich der disney-Gewinn): der Kopf braucht die Skala, nicht nur den umgerechneten Wert.
Vorbehalte: ohne slug wird schlechter (+0.01, Filme/tvh-Hashes; im Entwicklungssatz kaum vertreten); Sender ohne Trainingsaufnahme (super-rtl, toggo) fallen auf die globale Skala = wie K.
Abgrenzung: 009 (Logo relativ zur eigenen Aufnahme, verteilungsbasiert je Aufnahme) und O29-T (Sendungs-Kennung one-hot) -- hier je SENDER, aus dem Trainingsteil, zwei dichte Spalten.
Skripte: Scratchpad r1/m1-m6.py (Labels/Statistik), r1/p1.py (Proxy).
