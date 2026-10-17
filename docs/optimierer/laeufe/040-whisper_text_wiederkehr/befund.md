Woertliche Text-Wiederkehr im Whisper-Transkript (Anteil der Wort-5-Gramme eines 60-s-Fensters, die in Trainingsaufnahmen FREMDER Titel vorkommen) trennt allein mit AUC 0.87-0.91 (039: bestes Text-Merkmal 0.64); Sendung ohne Logo liegt fast nie darueber.

Klasse labels gezogen (Zug 0.68), geprueft, nichts Baubares: Werbeanteil, Bloecke/h und Blocklaenge je Sender stimmen zwischen Training (merged/auto) und Entwicklung (Summen) ueberein; Ausreisser nick (Training 0.127/193 s, Entwicklung 0.175/279 s) hat nur 12 bzw. 7 Bloecke und anderen Titel-Mix (South Park nur in Entwicklung), rtl ist Beet-Brueder-lastig -> Inhalt, kein Labelfehler. Fruehere Label-Befunde (025, 028, 032, aufnahme_rand, logo_werbebloecke) bestaetigt. -> merkmale (Zug 0.45, gleichauf mit daten; daten mehrfach als ausgeschoepft befundet).
Richtung (1) Stoerung erneut geprueft (lade(), Training schritt 8 / Entwicklung schritt 4): Logo-NaN 0.79 % / 0.02 %, Ton-NaN 0.61 % / 0.09 %, Backbone-Norm |z|>6 0 / 0, Entwicklung 0 Aufnahmen mit > 60 s Logo-NaN -> ein Stoerungs-Arm greift auf Eval nicht.

Richtung (2), Messung (Skripte Scratchpad z5/z6): Whisper-JSON in Training 143/665 (117 mit ads, also geladen), Kalibrierung 4/40, Entwicklung 76/76, Bestaetigung 22/22.
* Korpus alle 316 JSONs, Titel-Ausschluss: AUC 0.910, Mittel Werbung 0.46 / Sendung 0.025; Fenster > 0.2: Werbung 73 %, Sendung 3.3 %, Sendung ohne Logo (<0.3) 4.8 %.
* Korpus NUR Trainingsteil (117, wie im Arm): AUC 0.869, Werbung 0.33 / Sendung 0.010; > 0.2: Werbung 58.5 %, Sendung 1.3 %.
* Logo luegt (Werbung mit Logo-Mittel > 0.6 oder Sendung mit < 0.3, 1681 Fenster): Werbung 0.164 gegen Sendung 0.034.
* Greift auf Eval: Entwicklung 7512 Fenster, 15.1 % > 0.2 (Mittel 0.091), 74/76 Aufnahmen mit mindestens einem Fenster > 0.2 (Summe, Labels nicht gelesen).
* Haeufigste geteilte 5-Gramme sind Spots (z. B. Hygiene-Spot in 26 Titeln); Ausschluss des eigenen Titels haelt Serien-Vorspann draussen.
Kein K-Proxy gefahren (039: Proxy -15..-32 % Fehler, Messung dann -0.0014) -- das Messgeraet entscheidet.
Risiken: Regime wie 039 (nur 22 % der Trainingsaufnahmen bespurt, Platt sieht 4/40); 60-s-Fenster grob -> Kante kann leiden; Programmhinweise auf die EIGENE Sendung in fremden Aufnahmen koennen Sendungsdialog als wiederkehrend markieren (Sendung > 0.2 nur 1.3 %).
Formtest: 40 echte Training-uuids, synthetische X: +3 Spalten, Werte in [0,1], kein NaN, ohne JSON alles 0, 0.1 s.
