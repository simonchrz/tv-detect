In Logo-Ausfall-Strecken >= 10 min ist das fehlende Logo wertlos (Trainingslabel dort 20 % Werbung), K entscheidet out-of-fold aber 41-58 % der Sendung als Werbung; ein fester Logit-Abzug 1.0 dort halbiert diese Phantomwerbung bei 0-9 % verpasster echter Werbung.

Klasse: Bandit zog labels (Zug 0.82, 0 Laeufe). Geprueft (Scratchpad r3/d1.py, 527 Trainingsaufn. mit ads, 1-s-Raster):
* 1088 Werbebloecke, Dauer-Median 469 s, 99 % 680 s, keiner >= 900 s; nur 7 Bloecke (1951 s) liegen in >= 10-min-Logo-aus-Strecken
  (2 auto, 5 merged); Bloecke mit Logo > 50 % an: 31 (= schon abgelehnter logo_werbebloecke_raus). -> Labels sind in gestoerten
  Strecken sauber, kein Label-Hebel (wie 032/040/041). Daher nachbearbeitung (zweithoechster Zug 0.47).
Vorkommen: 043 hat gemessen, dass >= 10-min-Strecken im Fit 0.48 %, Entwicklung 1.5 %, Bestaetigung 1.8 % der Sekunden sind.
Proxy (r3/p1.py): Trainingsteil (saetze(), ohne O25), jede 8. s, Backbone+Logo+Audio, K-MLP 96/8 Ep., Haelften nach sha1(uuid),
2 Seeds, Logit-Abzug d nur in der Strecke, Glaettung 9 Zeilen, Schwelle 0.5. FP = Sendung in Strecke als Werbung, FN = Werbung in Strecke verpasst.
  d     s0h0 FP/FN      s0h1 FP/FN     s1h0 FP/FN     s1h1 FP/FN    (err gesamt s0h0 / s1h0)
  0     .563/.000       .406/.000      .579/.000      .406/.000     .0434 / .0444
  0.5   .463/.000       .335/.030      .454/.000      .310/.000
  1.0   .271/.000       .200/.090      .306/.000      .219/.050     .0415 / .0427
  1.5   .120/.000       .135/.270      .166/.000      .103/.240
  2.0   .043/.028       .090/.410      .061/.000      .032/.350     .0401 / .0411
  Median je Aufnahme unveraendert (.0166->.0165, .0146, .0166->.0165, .0153). Haelfte 1 (100 Werbezeilen in Strecken)
  verliert ab d 1.5 echte Werbung -> d = 1.0 fest, nicht weiter optimiert.
Abgrenzung: 042 (Spalte + Augmentation) und 043 (Gewicht) aendern das Training; hier bleibt K unveraendert, nur der Zustand
"Logo lange aus" wird nach dem Ensemble als Prior-Korrektur gesetzt. 035 (Gewissheits-Veto) griff an Laufform, nicht an Logo-Zustand, und feuerte nie.
Erwartung: Median-Delta Entwicklung ~0 (3 betroffene Aufn.), Wirkung ueber nn_erfindet; eigentlicher Test ist der Schiedsrichter-Satz
(dvr-super-rtl-1781112000, ganz ohne Logo, 15-min-Phantomblock). Risiko: logo-lose Aufnahmen mit schwacher Bild-Evidenz fuer echte Werbung.

## Abgelehnt

- ABLEHNUNGSGRUND Regeltyp: Der Arm ist mit Regel nicht_schlechter angemeldet, die opt_urteil.py ausdruecklich nur fuer Vereinfachungen vorsieht (Entwicklung: Median Delta >= -0.001, Bestaetigung: >= -0.002). Der Arm vereinfacht nichts, er fuegt einen neuen Mechanismus hinzu (Logit-Abzug 1.0 in erkannten Strecken). Seine Hypothese beansprucht einen Gewinn (Gesamtfehler .0434 auf .0415, Phantomwerbung halbiert). Das gehoert unter 'gewinn' (Median Delta >= +0.003, 4/5 positiv, Aufnahme-Median darf nicht sinken). Die Begruendung erwartet selbst 'Median-Delta Entwicklung ~0 (3 betroffene Aufn.)'. Mit nicht_schlechter wuerde die Gewinnschwelle umgangen, an der die Vorgaenger 042 (+0.0014) und 043 (-0.0010) auf derselben Aufnahmegruppe verworfen wurden. Neu anmelden mit --regel gewinn.
- Lecks: keine gefunden. Der Arm liest weder ktx["y"] noch meta, ads oder which, und enthaelt keine uuids oder Titel. Die Maske kommt allein aus der Logo-Spalte des jeweiligen Teils. spalten() verhaelt sich fuer training, kalibrierung und eval gleich (X unveraendert, Maske unter dem Teilnamen gemerkt). ktx["schritt"] wird fuer Fenster und Mindestlaenge beruecksichtigt (Training 4 s: Fenster 9 Zeilen = 36 s statt 31 s, ein kleiner Rundungsunterschied ohne Wirkung auf die Entscheidung). Der Teilname 'eval' passt in spalten() und nachbearbeiten() zusammen. Das Dict _maske ist modulglobal, aber nur im eigenen Modul und nur im Prozess des Laeufers.
- Leitplanken: keine Dateien, kein Netz, keine Prozesse.
