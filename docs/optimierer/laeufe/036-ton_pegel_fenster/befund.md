Der Ton-Wert im 60-s-Fenster relativ zum Aufnahme-Median traegt Information, die K je Sekunde nicht sieht: als Nachbearbeitung des rohen Logits (vor Platt) senkt er im K-gleichen Proxy die Sekundenfehler in 4/4 Laeufen um 5-10 %, FP und FN zugleich.

Klasse daten gezogen (Zug 0.62), ehrlich geprueft und ausgeschoepft:
* Logo NaN/0.5-imputiert: 964 + 145 Zeilen von 2.0 M, 3 Aufn. ohne Logo-Spalte; Ton konstant 0.5 (fehlt) in 3 Aufn. (2861 Zeilen, Fehler 40 %), Entwicklung 0.06 % -> kein Hebel.
* Spur-Luecke (nur 32 % SigLIP-, 36 % OCR-Zeilen im Training, Entwicklung 100 %) = Lauf 013, verworfen. Neu geprueft: Platt nur auf den bespurten Kalibrierzeilen (7 Aufn.) ist instabil und schlechter (Logloss 0.59-1.52 statt 0.13); "alle" bleibt bestes Platt. K ist auf bespurten Zeilen viel schaerfer (|Logit| 6.7 gegen 4.6).
-> nachbearbeitung (zweithoechster Zug 0.60).

Daten: Trainingsteil saetze() (527 Aufn. mit ads, jede 4. s, 506k Zeilen), volle K-Spalten (baue_k), HYPER_K, 3er-Logit-Mittel, Fit je sha1-Haelfte, Platt auf der echten Kalibrierung, 2 Seeds.
Ton je Sekunde: Werbung q50 0.567, Sendung 0.561 (ueberlappt fast ganz). Lange Laeufe (32-s-Mittel p>.5, >=120 s):
  echte Bloecke Ton-Median q10-q90 0.561-0.576 (eng, R128), erfundene 0.463-0.624; Abstand zum Aufnahme-Median echt -0.003..0.016, erfunden -0.07..0.08.
  Erfundene Laeufe: Logo-Median 0.19 gegen 0.05.
Harte Lauf-Sperre (|Ton-Abstand|>0.03 und Logo>0.1): 21 erfundene gegen 0-1 echte Laeufe, aber nur ~3 Aufn. betroffen (+0.0005 Proxy) -> zu schmal, verworfen.
Weiche Form je Sekunde, A = |Median60s(Ton) - Median(Aufnahme)|, z' = z + c*A + d*max(z,0)*A:
  LR auf OOF-Logits, Koeffizienten je Richtung stabil: c -17.7/-21.7/-18.5/-23.4, d -3.4/-6.8/-3.2/-6.7 -> fest c=-20, d=-5.
  Kreuz angewandt (Koeffizienten aus der anderen Haelfte), Platt neu auf behandelter Kalibrierung:
  s0 h0 Fehler 0.0516->0.0489 (FP 5591->5458, FN 7288->6756); s0 h1 0.0591->0.0533 (FP 8349->6970)
  s1 h0 0.0519->0.0494; s1 h1 0.0599->0.0542. Block-IoU (grober Proxy-Dekoder, opt_lauf.block_iou) Median +0.016 bis +0.069.
Vorbehalt: der Proxy-Dekoder ist viel schwaecher als der HSMM (dort gibt sogar das 1-s-Logo Gewinn); bewertbar ist nur der Sekundenfehler. Fenster-Median verschmiert an Kanten (Kante = 51 % des Verlusts).
Abgrenzung: O1 (Whisper-Spalte) und Hard-Cut-Ton (Memory audio_hardcut_signals_weak) nutzen Text bzw. Pegelspruenge; hier zaehlt der Abstand des Minutenpegels vom Normalpegel. 004 (Logo-Fenster) war Bild, nicht Ton.
Skripte: Scratchpad r2/p3.py (OOF mit Roh-Logits), a2.py (Laufstatistik), a8.py (Arm-Simulation mit Platt-Neuanpassung).
