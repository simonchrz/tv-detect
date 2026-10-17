Logo-Abwesenheit als Trainings-Augmentation (15 % der Zeilen Logo auf 0 gedrueckt) senkt im Proxy die erfundene Werbung in 4/4 Laeufen (Sendung ohne Logo: FP -11 bis -22 %), der Gewinn ist aber teils Pegel (AUC nur in einer Haelfte hoeher); die Klasse labels gab auch im fuenften Anlauf nichts her.

Klasse labels gezogen (Zug 0.67), geprueft (Trainingsteil 527 Aufn. mit ads, jede 4. s, 506k Zeilen, read-only, nur Summen):
* Label-Konsistenz identischen Bildmaterials ueber Aufnahmen (Backbone zentriert, Kosinus > 0.9, andere Aufnahmen):
  Zeilen mit >= 1 Duplikat 93.8k; Label widerspricht >= 80 % der Duplikat-Labels: 1827 Zeilen = 0.36 % (y0->W 1212, y1->S 615, 378 Aufn.),
  bei >= 3 Duplikaten 0.16 %, >= 10 0.09 %; je ein Drittel <= 16 s / 16-60 s / > 60 s von der Kante; merged 1.9 %, auto 2.0 % der Duplikat-Zeilen.
  -> Groessenordnung wie Confident Learning in 013 (0.07-0.35 %): kein Hebel.
* Nebenpruefung Kalibrier-Regime (Platt auf Zeilen ohne vs mit OCR/SigLIP-Spur, OOF-Haelften): a/b-Unterschied wechselt das Vorzeichen
  zwischen den Haelften (b -1.48/-1.09 gegen -0.81/-1.84) -> Zusammensetzung, keine systematische Regime-Verschiebung.
-> Ausweichen auf training (zweithoechster Zug 0.55).

Proxy: K-MLP 96 auf allen K-Spalten (baue_k), 8 Ep., Haelften nach sha1(uuid), OOF, Glaettung 9 Zeilen, Schwelle 0.5.
Logo roh: Werbung q10/50/90 0.00/0.06/0.68, Sendung 0.31/0.91/1.00. logo-blind (AUC Logo-Abwesenheit < 0.75): 36 Aufn.
Fehler / FP / Median je Aufn. / logo-blind / Sendung-mit-Logo<0.3-FP, Basis -> Anteil 0.15:
  s0 h0  0.0532/0.0424/0.0156/0.225/0.293 -> 0.0486/0.0370/0.0157/0.185/0.261  (AUC 0.9795 -> 0.9810)
  s0 h1  0.0345/0.0230/0.0134/0.169/0.164 -> 0.0343/0.0224/0.0125/0.184/0.128  (AUC 0.9855 -> 0.9847)
  s1 h0  0.0508/0.0400/0.0149/0.207/0.285 -> 0.0442/0.0322/0.0146/0.151/0.225  (AUC 0.9814 -> 0.9829)
  s1 h1  0.0386/0.0277/0.0132/0.213/0.162 -> 0.0364/0.0254/0.0124/0.204/0.126  (AUC 0.9847 -> 0.9842)
  Anteil 0.3: aehnlich, FN steigt mehr (0.0108 -> 0.0127), Median je Aufn. nicht besser -> 0.15 gewaehlt.
FN steigt leicht (+0.0001 bis +0.0011): Countdown-Spots/Trailer ohne Logo werden etwas schwaecher als Werbung gelesen -> nn_verpasst beobachten.
Vorbehalt: Platt im Messgeraet ebnet Pegel ein; h1-AUC sinkt leicht. Abgrenzung: 003 Backbone-Dropout, 012 Bild-Turm-Ausfall (Gegenrichtung),
016 Gewicht auf logo-blinde Aufnahmen (wirkt nur dort), 024 Logo-Steigung (Form, nicht Daten).
Skripte: Scratchpad k1.py (Duplikat-Labels), k2.py (Platt-Regime), k3.py (Proxy).
