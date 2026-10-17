Die Reichweite der Szenen-Kette (Zusammenhangskomponente im Kosinus-Graph, Kanten <= 240 s, tau 0.42) trennt Sendung ohne Logo besser als die direkte Wiederkehr aus 018 (AUC logo-blind 0.849 gegen 0.802) und ergaenzt sie im Proxy leicht.

Daten: Trainingsteil 527 Aufn. mit ads (jede 4. s, 506k Zeilen, Werbung 23.8 %), Kosinus auf zentriertem Backbone, Vorrat jede 4. s.
Logo-blind = AUC(Logo-Abwesenheit gegen Label) < 0.75 je Aufn.: 36 Aufn., 7.5 % der Zeilen.
* tau-Wahl (nur Training), AUC der Spanne alle | logo-blind:
  0.35: 0.805 | 0.821 (Werbung mit Spanne > 600 s: 25 %), 0.42: 0.829 | 0.849 (6.4 %), 0.5: 0.787 | 0.797,
  0.6: 0.719 | 0.715, 0.7/0.8 nutzlos (Kanten zu selten; 018-max hat Median 0.55 Sendung / 0.32 Werbung).
  tau 0.42: Spanne Median Sendung 700 s, Werbung 0 s; Sendung < 60 s: 21.7 %.
  Zum Vergleich 018-Spalten: vor 0.792, nach 0.793, min 0.794, max 0.827 (logo-blind 0.791/0.790/0.804/0.802).
* Proxy wie 018 (MLP 96, 8 Ep., Backbone+Logo+Audio, OOF Haelften nach sha1(uuid), Glaettung 9 Zeilen, Schwelle 0.5), Seeds 0/1:
  basis      Fehler 0.0451/0.0704  blind 0.203/0.274  Median/Aufn 0.0217/0.0256  AUC 0.9799/0.9794
  +018       Fehler 0.0338/0.0417  blind 0.092/0.122  Median/Aufn 0.0220/0.0245  AUC 0.9834/0.9838
  +kette     Fehler 0.0287/0.0462  blind 0.071/0.107  Median/Aufn 0.0174/0.0292  AUC 0.9840/0.9831
  +018+kette Fehler 0.0329/0.0409  blind 0.086/0.105  Median/Aufn 0.0195/0.0242  AUC 0.9846/0.9842
  Kombination in beiden Seeds bester AUC und besser als 018 in blind und Median/Aufn; Abstand zu 018 klein (Proxy streut stark).
Risiko: Teleshopping/Dauerwerbung verkettet sich wie Sendung (6.4 % der Werbezeilen mit Spanne > 600 s), Trenner/Schwarzbilder
koennen Ketten bilden; Kanten nur <= 240 s, damit Wiederholungs-Spots ueber Bloecke nicht verbinden. HSMM holt einen Teil schon ab.
Skripte: Scratchpad kd.py/kd2.py/kd3.py (read-only, nur Summen).
