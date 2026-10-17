Die Klasse daten ist ausgeschoepft. Auch die letzte offene Abbild-Luecke, 44 leere auto-confirm-Aufnahmen (70k s reine Sendung, die die Produktion trainiert und K nicht laden kann), bewegt im Produktions-HSMM-Proxy nichts (+0.0004 / -0.0002). Nachbearbeitung als zweite Klasse ist im selben Proxy klar negativ.

Daten: saetze()/lade()/baue_k() aus opt_lauf (ohne O25), jede 4. s. Proxy wie 031: Eval = 134 Trainingsaufnahmen mit altem Signal-Dump, K (3 Koepfe, Seeds 0/1) auf den anderen 393, Platt auf dem Kalibrierteil, Dekoder ist das Produktions-HSMM (tv-detect, replay), Wahrheit = Trainingslabels. Basis 0.9330/0.9326, Median 0.984/0.988.
* Spur-Abdeckung ist schon in 013 gemessen: Kalibrierteil SigLIP 6/34 gegen Eval 76/76. Platt wird also im Regime ohne Spur angepasst. Die Produktion wendet Platt nicht auf den Dekoder an, das ist ein Messgeraet-Artefakt und kein Arm. Hinweis an Simon.
* Defekte Eingaben, gesucht in den Merkmalen des Trainingsteils (2.0M s): 0 NaN-Backbone, 0 Null-Vektoren, 964 NaN-Logo, 3425 eingefrorene Bilder (Kosinus > 0.9995 zum Vorgaenger, zu 80 % Werbung). Eine Decode-Stoerung ist aus den Merkmalen nicht erkennbar.
* OOF (Haelften nach sha1, K-MLP, 2 Seeds, Gleitmittel +-60 s): 163 Phantomlaeufe (FP >= 60 s, 47k s) gegen 1079 wahre Bloecke. Logit-Mittel Median 1.6 gegen 5.0, Anteil positiver Sekunden 0.72 gegen 0.97, Seed-Uneinigkeit |p0-p1| 0.071 gegen 0.019 (3.7x), Logo 0.29 gegen 0.18.
* Nachbearbeitung im Proxy, beide Seeds, Delta IoU-Mittel:
  * Totzone (|z| < c auf 0): c = 0.5 0.000/-0.001, 1.5 -0.001/-0.002, 3 -0.015/-0.018.
  * Konsistenz z*|2*frac+-W - 1|^g: W30 -0.012 bis -0.028, W60 -0.020 bis -0.050.
  * Grund (siehe auch 031): Phantome, die keinen wahren Block beruehren, kosten keine Block-IoU. Schwache Evidenz zu daempfen kostet nur Kanten.
* Leere Aufnahmen im Trainingsteil: 44, alle auto-confirm. rtl 9 (Unter uns, GZSZ), ohne Sender 17 (SpongeBob, Simpsons, CSI), disney 9, je 1 super-rtl (Gangnam Project), toggo-plus, kika, nick, mtv, zdf, anixe (Infomercial). Fuer super-rtl, toggo-plus und kika hat K NULL Trainingszeilen.
  * Nebenbefund: dvr-super-rtl-1781112000 (Phantomblock im Schiedsrichter) ist eine davon, also Training der Produktion.
* Arm: 50.4k Zusatz-Sendungssekunden auf 91 Aufnahmen (57.6k Zeilen, Faktor <= 4), +3.3 % Sendungsmasse.
  * Proxy: +0.0004 / -0.0002, Median -0.0005 / -0.0006, besser/schlechter 30/25 bzw. 19/27, Werbezeit +-0.2 %.
  * Menschlich gelabelte Teilmenge (18 Aufnahmen): -0.0007 / +0.0002.
  * Je Sender uneinheitlich (disney +0.019 / -0.002).
Erwartung: Delta um 0. Wert des Laufs: klaert, ob die fehlenden leeren Aufnahmen zur Kinder-Abbild-Luecke beitragen. Ohne Effekt waere es ein Beleg, dass die Luecke anderswo liegt (25 % fehlende Archiv-Meta, Gewichte).
Skripte: Scratchpad d1-d4.py, oof.py, a1.py, px1-px5.py, gew.py, leer.py (read-only, nur Summen).

## Abgelehnt

- Der Code laeuft. Ich habe ihn mit synthetischen Arrays (schritt 1 und 4, fehlende Meta, Aufnahme ohne Sendungszeilen) und mit den echten Trainingsmetas des Archivs aufgerufen, ohne opt_lauf.py zu starten. Ergebnis: float32, >= 0, Maximum 4.0, Werbezeilen bleiben bei 1.0, und das schritt-Handling stimmt. Auf den echten Daten stimmt die LEER-Liste exakt mit den 44 leeren Aufnahmen (70417 s) ueberein, und es gibt 91 Aufnahmen, 57584 Zeilen und 50.4k Zusatzsekunden, wie im Befund.
- Keine Lecks und keine Leitplanken-Verstoesse. Der Arm nutzt nur ktx["y"], rec, schritt, Titel und Sender der Trainingsaufnahmen. Es gibt kein spalten(), also keinen Train/Serve-Bruch. Er schreibt nichts, startet keine Prozesse, nutzt kein Netz und hat keinen Zufall. Der Regeltyp gewinn passt.
- Kleiner Vorbehalt: die Kindersender-Gruppe (toggo, nick) ist aus der Abbild-Luecke abgeleitet, die an den 22 Bestaetigungs-Aufnahmen gemessen wurde (toggo plus 1780640400, nick 1781550300). Das ist indirekt, aber die README veroeffentlicht diese Luecke.
- Die Paritaets-Behauptung stimmt nicht ganz. Die Hypothese sagt, die Produktion trainiere alle 44 leeren Aufnahmen als reine Sendung. Die Produktion trainiert nur auto-confirm mit nicht verdaechtiger Kohorte (train-head.py, ca. Zeile 3750). Ich habe suspect_cohorts aus dem Archiv nachgerechnet: Zwei der 44 sind verdaechtig, die SpongeBob-Aufnahmen aa73dd63... und e5f85548... in Kohorte (SpongeBob Schwammkopf, Nick). Dort hat ein menschlich geprueftes Geschwister.
