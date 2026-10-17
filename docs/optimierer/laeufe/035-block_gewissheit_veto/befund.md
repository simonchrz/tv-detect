Lange Werbe-Laeufe, in denen der Kopf nirgends sicher ist (< 40 % Sekunden p > 0.9), sind im Trainingsteil fast immer erfunden; ein Veto darauf hebt im Proxy die Block-IoU in beiden Haelften um ~+0.004 ohne Aufnahme unter -0.01.

Klasse daten gezogen (Zug 0.57), aber ausgeschoepft (Befunde 026/032/033, leer_aufnahmen_ersatz, produktions_gewichte_abbild; Vorgabe: keine Alters-/Aufnahme-/Dubletten-Gewichte) -> nachbearbeitung (zweithoechster Zug 0.56).
Stoerungs-Spur geprueft: isolierte Backbone-Spitzen (roher Kosinus zu t+-1 < 0.3/0.5, cos(t-1,t+1) > 0.7/0.5) sind 0.001-0.2 % der Sekunden;
sie maskieren (p durch Median der sauberen +-10 s ersetzen) aendert die Fehler um < 0.3 % -> Einzelbild-Stoerungen sind kein Hebel. Ohne Decode-Fehler-Spur
im Archiv laesst sich "gestoerte Quelle" nicht direkt erkennen; erkennbar ist aber ihre Folge: halb ueberzeugte, lange Laeufe.

Daten: Trainingsteil saetze() (527 Aufn. mit ads), Proxy MLP96/10 Ep. auf Backbone+Logo+Audio (ohne OCR/SigLIP), schritt 4, out-of-fold nach sha1-Haelften,
Vorhersage je Sekunde. Laeufe = 31-s-Mittel p > 0.5:
  Laenge 120-300 s: echt 154 / falsch 42; Anteil p>.9 echt q10/q50 0.69/0.84, falsch q50/q90 0.31/0.49
  Laenge 300-600 s: echt 819 / falsch 15; echt 0.77/0.85, falsch 0.43/0.52
  Laenge >= 600 s:  echt 57 / falsch 3;   echt 0.82/0.86, falsch 0.50/0.61
Proxy-Dekoder (Mittel > .5, Werbung < 60 s raus, Sendungsinsel < 90 s zu), IoU je Aufnahme, Basis mittel 0.8996 / median 0.9665:
  L120 F0.4 C4 hoch .9: Haelfte 0 +0.0034 (7 besser/0 schlechter), Haelfte 1 +0.0039 (6/0, min -0.000), Median-Aufn. +0.0000
  L120 F0.5 C4: +0.0047/+0.0044, eine Aufn. -0.017; L90 F0.4: +0.005, aber eine Aufn. -0.137 (zu kurze Laeufe sind riskant)
  hoch .8 traegt nur +0.001, hoch .95 mit F0.5 kippt (min -0.99) -> F0.4 gewaehlt, weil in hoch .9 und .95 stabil.
Ganzer Trainingsteil, L60-Varianten: dFP bis -22 000 s bei dFN +1 200 s; L120 F0.4: dFP -10 100 s, dFN +175 s.
Risiko: K ist schaerfer als der Proxy (OCR, SigLIP, 3er-Ensemble); der Anteil p > 0.9 verschiebt sich -> echte Bloecke sicherer, aber weniger falsche erfasst.
Abgrenzung: O32 (Inselsperre ueber den Kopf) und Dekoder-Sweeps arbeiten auf Einzelsekunden; hier entscheidet eine Laufgroesse, die der HSMM nicht sieht.
Skripte: Scratchpad r1/d1.py (Proxy), a2.py (Laufstatistik), a3.py/a4.py (Regel-Gitter, Haelften).
