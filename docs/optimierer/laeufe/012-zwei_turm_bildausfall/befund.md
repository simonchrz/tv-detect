Ein additiver Klein-Turm (Logo/OCR/Audio) mit zeilenweisem Ausfall des Bild-Turms senkt im Proxy die Sekundenfehler um 3-6 % und die Fehlalarme bei sichtbarem Logo um ein Viertel, verteilt den Gewinn aber ungleich ueber die Aufnahmen.

Daten: Trainingsteil (609 Aufn., 299 je Haelfte), haelftig nach sha1(uuid), jede 8. s, 287k Zeilen, volle K-Spalten (baue_k).
Fit auf A, Vorhersage auf B, Logit-Mittel aus 2 bzw. 3 Seeds, Platt 2-fach nach Aufnahme auf B.
Zwei Seed-Saetze (0,1 | 2,3,4):
                 logloss        Fehler          erfindet     verpasst     Logo>=.8&p>=.8 (n / Werbeanteil)
  K              .1802/.1783    .0607/.0592     3458/3298    5064/5005    826 .74 / 751 .80
  Turm p=0.5     .1760/.1755    .0573/.0572     3136/3157    4907/4865    625 .85 / 551 .85
  Turm p=0.25    -    /.1762    -    /.0578     -   /3222    -   /4889    -       / 623 .84
  Turm p=0 (nur additiv) .1775/.1770  .0585/.0580  3283/3235  4930/4904  770 .81 / 719 .82
  Turm, f nur Bild (ohne kleine Spalten) p=0.5: .1910 / .0646 — schlechter, f braucht die Interaktionen.
Aufnahmefehler Median K .0440/.0455, Turm p=0.5 .0442/.0433; Mittel unveraendert (.058/.057).
Aufnahmen >0.005 besser/schlechter: p=0.5 66/71 und 56/62; p=0 42/24 und 32/18.
Lesart: p=0.5 ist seed-stabil (K streut .0592-.0607, Turm .0572-.0573) und trifft den Logo-Fehlalarm;
p=0 ist ausgewogener je Aufnahme, aber kleiner. Risiko: Aufnahme-Median der IoU (wie 006).
Skripte: Scratchpad diag_turm.py / diag_turm2.py (read-only, nur Summen).
