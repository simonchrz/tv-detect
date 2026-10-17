Gestoerte Quellen zeigen sich in den Merkmalen nicht als Decode-Spur (Backbone-NaN 0, Norm-Ausreisser 0.03 %), sondern als minutenlanger Logo-Ausfall; eine Laufzeit-Spalte plus Strecken-Augmentation senkt im Proxy die erfundene Werbung unter simuliertem 10-30-min-Ausfall von 30-35 % auf 7-16 % ohne Fehler auf sauberen Daten.

Klasse merkmale (Bandit). Richtung laut Sitzungshinweis: Robustheit gegen gestoerte Quellen.
Vorkommen (Archiv-Features, 1-s-Raster, read-only, Skripte Scratchpad st/d1.py, st/d2.py):
* Decode-Spuren: training 200 Aufn. / entwicklung 76: Backbone-NaN 0/0, Audio-NaN 0/0, |z(Norm)| > 6 0.03 %/0.04 %,
  zusammenhaengende Ausreisser-Strecken (30 s, >30 %) 0. Sprungrate cos<0.3 (zentriert) 26 % ueberall = Schnitte, keine Stoerung.
  Logo-NaN nur in 5 Trainingsaufn. (-> 0.5 im Loader), Eval 0. -> Backbone-Rauschen-Augmentation haette kein Gegenstueck im Eval.
* Logo-lose Strecken >= 10 min (31-s-Median < 0.15): training 10 Strecken in 7/524 Aufn. (0.48 % der s), Werbeanteil Median 0.16,
  keine > 0.8; entwicklung 4 in 3/76 (1.5 %); bestaetigung 1 in 1/22 (1.8 %). Im Eval 3x haeufiger als im Fit.
Proxy (st/p1.py): opt_lauf.lade(training, jede 8. s), Backbone+Logo+Audio, MLP 96/10 Ep., Haelften nach sha1(uuid), Glaettung 9 Zeilen,
Schwelle 0.5; Stoertest = dieselbe Testhaelfte mit je einem 10-30-min-Logo-Ausfall pro Aufnahme (Ziel unveraendert).
  err sauber / Median je Aufn. / err gestoert / FP Sendung im Ausfall:
  s0h0  K .0391/.0138/.1056/.353  nur Aug .0375/.0155/.0798/.233  Spalte+Aug .0376/.0145/.0534/.100
  s0h1  K .0384/.0131/.0995/.304  nur Aug .0362/.0131/.0693/.174  Spalte+Aug .0402/.0131/.0497/.070
  s1h0  K .0403/.0151/.1009/.323  nur Aug .0390/.0144/.0798/.223  Spalte+Aug .0385/.0136/.0672/.162
  s1h1  K .0357/.0128/.0994/.314  nur Aug .0338/.0130/.0707/.189  Spalte+Aug .0333/.0130/.0449/.076
Lesart: Augmentation allein (= Strecken-Variante von 025) halbiert den Schaden nicht; erst die Laufzeit-Spalte gibt dem Kopf das Gatter.
Sauber: 3/4 gleich oder besser, s0h1 +0.0018 schlechter -> Risiko fuer den Aufnahme-Median gering, Gewinn konzentriert auf wenige Aufnahmen.
Abgrenzung: 025 zeilenweise ohne Zeitstruktur (Entwicklung +0.0034, Bestaetigung -0.0001), 004 Logo-Mittel 30 s, 009/027 Aufnahme-/Sender-Konstanten.
Zwei weitere Varianten derselben Richtung fuer spaetere Runden: (a) Ton-Ausfall-Strecken (Audio konstant) analog; (b) Spalte ohne Augmentation.
Erwartete Groesse: Entwicklung hat nur 3 betroffene Aufnahmen -> Median-Delta knapp, Wirkung eher ueber nn_erfindet und den Schiedsrichter-Satz.
