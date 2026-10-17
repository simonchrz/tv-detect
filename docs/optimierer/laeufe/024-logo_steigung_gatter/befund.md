Ein Logo-Pfad mit kontextabhaengiger, vorzeichenfester Steigung (softplus(MLP16) * logo) neben dem K-MLP senkt im Proxy die Sekundenfehler um 7-9 % und den Aufnahme-Median um 10-15 % in 3 von 3 Seeds; die Klasse labels gab nach drei Pruefungen keinen Arm her.

Klasse labels gezogen (Zug 0.82), geprueft auf dem Trainingsteil (527 Aufn. mit ads, read-only, nur Summen):
* Split-Screen-Konvention (neu 17.10.): Sendungs-Zeilen mit ocr_werbung, > 20 s von jeder Kante, nicht Rand: 104 Laeufe / 2952 s,
  Median 29 s = fast immer EINE OCR-Fundstelle (+-10 s verschmiert), Laeufe >= 45 s nur 9 (461 s), meist Galileo -> kein Hebel.
* Galileo-Regel aus §3ax verallgemeinert (Label-Kante, dann 8-45 s Logo < 0.35, dann Logo-Rueckkehr 5 s): Enden 13 von 1042 (203 s),
  Starts 18 von 1073 (372 s) -> Archiv ist dort schon sauber.
* Nachlauf: Bloecke am Aufnahmeende 41 / 8247 s, Logo>0.6 nur 17 % (Mitte 12 %) -> echte Werbung, keine Sendung als Werbung.
-> Ausweichen auf training (zweithoechster Zug 0.43).

Training-Arm, Proxy: K-MLP auf Backbone+Logo+Audio (ohne OCR/SigLIP), 6 Epochen, Haelften nach sha1(uuid), out-of-fold, Schwelle 0.5:
  K       Seeds 0/1/2: AUC 0.9644/0.9642/0.9644, Fehler 0.0826/0.0770/0.0774, Aufnahme-Median 0.0537/0.0502/0.0499
  Gatter  (Logo im Haupt-MLP genullt): AUC 0.9654/0.9653/0.9650, Fehler 0.0751/0.0741/0.0716, Median 0.0455/0.0452/0.0427
  Gatter  (Haupt-MLP sieht Logo = der Arm), Seed 2: AUC 0.9652, Fehler 0.0712, Median 0.0422
  je Aufnahme (Seed 2, |d| > 0.002): Arm besser 363, schlechter 88 von 527.
* K ist schon monoton im Logo (Logo +0.3 hebt p in 0 % der Zeilen) -> der Gewinn kommt nicht aus einer Vorzeichen-Korrektur,
  sondern aus der Form: das Logo wirkt linear mit je Zeile gelernter Staerke, statt stueckweise ueber ReLU-Knicke.
Vorsicht: AUC-Gewinn nur +0.0008; ein Teil des Fehlergewinns bei fester Schwelle kann Pegel sein, den Platt im Messgeraet ohnehin
ausgleicht. Abgrenzung zu 012 (additiver Klein-Turm + Bild-Ausfall, -0.0014): hier kein Ausfall, multiplikative Steigung nur fuers Logo.
Skripte: Scratchpad p1/m1-m6.py.
