Decode-Stoerungen erreichen die Merkmale auch mit einem vierten Mass nicht (PCA-Residuum); der Fehler auf super-rtl ist ein Abdeckungsproblem: Kinder-Sender tragen out-of-fold doppelt so viel Fehler je Aufnahme, und Gewicht 2 auf ihre Zeilen senkt ihn in 4/4 Laeufen.

Klasse labels gezogen (Zug 0.95), geprueft, 13. Anlauf ohne Hebel; daten zweithoechster Zug (0.62).
* Stoerungsspalte (Sitzungshinweis b): Backbone z-normiert, PCA-64 aus 120 Trainingsaufn. (schritt 4), RMS-Residuum je Sekunde.
  Training q50/95/99 0.63/0.88/1.01; super-rtl-1781112000 q50/95 0.60/0.83, 0.7 % ueber Training-p99, kein 60-s-Fenster > 30 %.
  Entwicklung 1.0 % ueber p99, 1 von 76 Aufn. mit Fenster. -> Spalte wuerde nicht anschlagen (deckt sich mit 042/044: NaN, Norm,
  Flackern, SigLIP-Nachbarkosinus). Augmentation (a) haette kein Gegenstueck im Eval; Bild-Augmentationen 003/012/028/029 alle Rauschen.
* Labels: isolierte Werbe-OCR-Strecken in der Sendung (>= 29 s Werbemarker-Fenster, Logo-Mittel < 0.4, +-40 s kein Label-Block,
  meist Galileo-Countdown-Einzelspots): Training 25 als Sendung / 1 als Werbung gelabelt (611 s); Entwicklung (Summe) 8 / 0.
  -> gleiche Lesart, Umlabeln arbeitet gegen die Messlatte. Rand- und Blockfaelle wie in rand_einblendung_ziel.
* daten, verworfene Idee: 61 auto-Aufnahmen (13.9 % der Zeilen) raus -> Proxy uneinheitlich (Fehler 2 besser, 2 schlechter).
Proxy fuer diesen Arm (Scratchpad r2/p2.py, p3.py): opt_lauf.lade(training, jede 8. s), nur Backbone+Logo+Audio, MLP 96/8 Ep.,
Haelften nach sha1(uuid), Glaettung 5 Zeilen, Schwelle 0.5, Sekundenfehler je Aufnahme. Kinder = 47 Aufn. (disney 27, nick 12,
comedy-central 8; super-rtl/toggo/kika haben keine Trainingsaufnahme mit Werbung), 4.3 % der Zeilen.
  Lauf   K kinder mittel/median  -> x2             -> x3            | uebrige median K -> x2 / x3
  s0h0   .0833/.0532 -> .0734/.0315 -> .0709/.0342 | .0151 -> .0159 / .0165
  s0h1   .0668/.0333 -> .0575/.0319 -> .0517/.0265 | .0157 -> .0154 / .0158
  s1h0   .0869/.0617 -> .0765/.0433 -> .0748/.0433 | .0154 -> .0162 / .0171
  s1h1   .0759/.0418 -> .0621/.0319 -> .0536/.0316 | .0149 -> .0160 / .0160
  Uebrige Mittel unveraendert (+-0.0004). Faktor 2 gewaehlt: fast der ganze Kinder-Gewinn, halbe Kosten am uebrigen Median.
Bezug: 016 (Gewicht 3 nach Logo-Trennschaerfe) +0.0029, Gewinn fast nur comedy-central/nick/super-rtl, 3/5 positiv, gegen altes K.
Risiko: Entwicklung hat 16 Kinder-Aufn. von 76 (nick 7, comedy 4, disney 4, super-rtl 1); der uebrige Aufnahme-Median koennte leicht sinken (Proxy +0.0008 Fehler).
