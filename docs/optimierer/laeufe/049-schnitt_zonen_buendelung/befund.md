Klassen daten und nachbearbeitung geben auf dem Trainingsteil nichts mehr ueber +0.002 her; der beste Rest ist eine Einstellungs-Buendelung in unsicheren Zonen, die im Produktions-HSMM-Proxy den Aufnahme-Median in 10/10 Varianten-Seeds um +0.0006..+0.0019 hebt, das Mittel aber nur um -0.0003..+0.0011.

Klasse daten gezogen (Zug 0.54) und wie in 034/035/038/leer_aufnahmen_ersatz/produktions_gewichte_abbild ausgeschoepft; Ausweichklasse nachbearbeitung (zweithoechster Zug 0.30).
Daten: Trainingsteil von K (saetze(), 527 Aufn., jede 4. s, 506k Zeilen, K-Spalten 1350). OOF-Kopf = K (MLP 96, 12 Ep., 3er-Logit-Mittel), Haelften nach sha1(uuid), 2 Seeds.
Proxy: 132 Trainingsaufnahmen mit altem Signal-Dump (emit-signals), OOF-Logits JEDE Sekunde vorhergesagt, Produktions-HSMM, Block-IoU gegen meta-ads.
WICHTIG Proxy-Falle: opt_lauf.platt divergiert auf diesen OOF-Logits in BEIDEN Seeds (6.4 % der Zeilen an der Klemme +-13.8; Newton springt ab Iteration 3 auf |a| > 1e8).
  Mit opt_lauf.platt: K 0.8724 / 0.9395 (Seed 0 kaputt). Mit sklearn-Platt (C=1e4): K 0.9356 / 0.9352, Median 0.9849 / 0.9841. Fruehere Proxies, die opt_lauf.platt auf OOF-Daten nutzten, pruefen.
Verlust-Zerlegung der Block-IoU (K, Seed 0, 132 Aufn., Summe 0.068): Kante IoU < 0.9 0.028 (44 Bloecke; Ende zu frueh 23, Start zu spaet 17), verpasst 0.021 (11 Bloecke), Kante klein 0.016 (244), gespalten 0.003, verschmolzen 0.
  Von 11 verpassten Bloecken haben 4 (65-112 s) z > 0 auf 60-90 % der Sekunden; der Dekoder verwirft sie.
Gepruefte Hebel (Delta IoU-Mittel Seed 0 | Seed 1):
* Kurzinsel-Anhebung (15-s-Mittel > 0, 30-150 s, +1.5..3 Logit): -0.0002..-0.0025, Phantome 16 -> 29-39, Median -0.003. 132 Inseln, davon 4 echte. Verworfen.
* Blockstart per Logo-Abfall vorziehen: mittlerer Startfehler 12.4 -> 12.1 s; spaete Starts tragen schon Logo (Trailer). Kein Hebel.
* Kurzblock-Gewicht im Training (Werbezeilen in Bloecken < 240 s mit clip(240/L,1,4) bzw. < 600 s bis x12): -0.0012 | -0.0003 bzw. +0.0004 | -0.0003; weiter 11 verpasste Bloecke. Gewichte lehren die Kurzblock-Form nicht.
  (OOF: Anteil z > 0 in Label-Bloecken <= 60 s 0.54, 60-150 s 0.74, 150-300 s 0.89, laenger 0.94; Bloecke < 150 s = 4 % der Bloecke, 0.8 % der Werbezeilen.)
* Einstellungs-Mittel ueberall (A=1): -0.004 (Seed 1). Hochpass 600 s: -0.003. Medianfilter 5 s: -0.001.
* Arm, Einstellungs-Mittel nur wo |15-s-Logitmittel| < BAND (Kosinus Vorsekunde, aufnahme-zentrierter Backbone):
  TAU 0.4 BAND 5 A 0.7: +0.0002 | +0.0007, Median +0.0011 | +0.0010, besser/schlechter 6/5 | 12/7, min -0.034 | -0.022
  TAU 0.3: -0.0003 | +0.0011 (Median +0.0016 | +0.0019); TAU 0.5: -0.0000 | +0.0007; BAND 8: +0.0001 | +0.0007; A 0.5: +0.0001 | +0.0005; A 1.0: -0.0002 | +0.0005.
  Flache Flaeche, Arm in ihrer Mitte. Schnitt-Lage: Kosinus zur Vorsekunde an Label-Kanten Median 0.38 (q25 0.10), sonst 0.63.
Folgerung: Kanten- und Kurzblock-Fehler sitzen nicht in der Sekunden-Rampe und nicht im Gewicht. Was fehlt, ist Evidenz fuer die verpassten Formen (Trailer am Blockstart mit Logo, Einzelspots), nicht Glaettung.
Skripte: Scratchpad d/oof.py, oof1.py, kb.py (sklearn-Platt), kb2-kb5.py, start.py, oofw.py, cmp.py (nur Summen).
