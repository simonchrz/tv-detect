Rund 35 Trainingsaufnahmen (7 %) tragen eine global verschobene Label-Zeitachse (12-60 s): zwei unabhaengige Quellen (OOF-Kopf und roher Logo-Wert) finden dieselbe Verschiebung, 3x so oft wie bei Zufall; die Produktions-Quarantaene kennt davon nur eine.

Klasse daten gezogen (Zug 0.54). Daten: Trainingsteil von K (opt_lauf.saetze/lade/baue_k, ohne O25), 527 Aufn. mit ads, jede 4. s, 506k Zeilen; OOF = K-MLP, 3 Koepfe, Haelften nach sha1(uuid).
Erst zwei Daten-Ideen geprueft, beide tot:
* Neuheit der Sendung (max. Backbone-Kosinus zu anderen Aufnahmen): neuestes Dezil FP 8.4 % gegen 5.2 % Mitte, Werbeanteil 13.8 %. Sendungszeilen im neuesten 20 % bis 3x gewichten: err 0.0685 -> 0.0684/0.0689, Phantomlaeufe (>= 60 s) 186 -> 193/192. Rauschen.
* Dubletten mit Gegenlabel (Kosinus > 0.95, Mehrheit der Treffer anderer Aufnahmen traegt das andere Label): nur 1086 Zeilen (0.2 %), 2 % der OOF-Fehler. Zu klein (wie Hygiene, 389 Zeilen).
* Labels (zweite Klasse) strukturell sauber: 8 Sendungsinseln < 90 s, 8 Bloecke < 60 s im ganzen Trainingsteil.
Zeitachsen-Versatz (r2/shift.py, r2/lshift.py):
* (a) je Aufnahme Verschiebung s in -60..+60 s (4-s-Raster), die die Log-Likelihood der Labels unter der OOF-Wahrscheinlichkeit maximiert; 239/527 nicht bei 0, |s|>=16 s mit Gewinn > 20: 46.
* (b) unabhaengig: roher Logo-Wert je Sekunde (1-s-Raster, 450 Aufn. mit Logo-Trennung >= 0.4), L1 zwischen verschobenen Labels und 1-Logo.
* Einig (gleiches Vorzeichen, |sa-sb| <= 12 s): |sa|>=12 s 35 Aufn. (Zufallspaarung ~10.9), >=16 s 27 (6.8), >=24 s 16 (2.2).
  Sender: prosieben 13, vox 6, kabel-eins 5, ohne Sender 4, nick 3, rtl 2; fast alle merged. Vorzeichen beidseitig (-40..+60 s), also kein Logo-Nachlauf.
* Gegenprobe Produktions-Quarantaene (zeitachsen-versatz.json, 3 Eintraege): dvr-rtl-1781199600 (40 s) gefunden, dvr-rtl-1779473700 (1269 s, ausserhalb des Suchfensters) nicht, der dritte ist nicht im Trainingsteil.
Proxy (OOF, die 35 aus dem Fit, bewertet nur auf den uebrigen Aufnahmen, 2 Seeds):
  err 0.0683 -> 0.0679 / 0.0683 -> 0.0672; Kantenfehler (<= 16 s an Labelkante) 3112 -> 3074 / 3133 -> 3096; FP -280/-614, FN +71/+125; e60-Mittel -0.0003/-0.0014, Median +-0.0002.
  Trotz 5 % weniger Daten in 2/2 Seeds besser, klein.
Arm: Gewicht 0 fuer die 35 plus dvr-rtl-1779473700 (Produktions-Quarantaene, K traegt sie noch), 29k Zeilen. Abgrenzung 026: dort 2 Quarantaene + 7 auto-Aufnahmen mit Logo-Bloecken, neutral; hier 34 neue, fast alle merged.
Wenn bestaetigt: Empfehlung an Simon, zeitachsen-check.py um den Logo/OOF-Detektor zu erweitern (oder die Labels um s zu verschieben, im Bild pruefen). Dann waere die Label-Version (Labels verschieben statt raus) der naechste Arm.
Skripte: Scratchpad r2/nov.py, px.py, konf.py, ins.py, shift.py, lshift.py, px2.py (read-only, nur Summen).
