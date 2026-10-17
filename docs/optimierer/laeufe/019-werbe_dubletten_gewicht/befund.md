Werbung ist im Trainingsteil zu 63 % Wiederholung (Sendung 4.7 %); die einmaligen Werbezeilen lehren "ungesehenes Bild = Werbung" -- sie abzuwerten senkt im Proxy die erfundene Werbung stark, aber fast nur als Pegel, die Rangordnung gewinnt wenig.

Klasse labels gezogen und geprueft (Trainingsteil, read-only), ohne Arm:
* Label gegen Logo verschoben (+-40 s je Aufnahme, 304 logo-starke): Kurve flach, Gewinn bei bester Verschiebung meist < 0.005 -> kein Versatz-Fehler.
* Werbeblock-Laengen: keine > 900 s, 83 % 420-900 s, Logo im Block 0.20 gegen 0.84 in Sendung -> keine Lang-Fehllabels.
* Kante gegen staerksten Backbone-Schnitt +-8 s: fast gleichverteilt (Versatz 0 nur 10-22 %), Schnitt-Einrasten waere Zufall.
* Mit 013 (Confident Learning 0.07-0.35 %) -> kein Label-Arm; Klasse daten (zweithoechster Zug).

Duplikate: Backbone zentriert, Kosinus > 0.9 zu Zeilen ANDERER Aufnahmen (jede 4. s, 506k Zeilen, MPS ~95 s).
  Werbung: 63 % mit Duplikat, Median 1, q90 21, q99 185. Sendung: 4.7 %, q99 4.
Proxy: Haelften nach sha1(uuid), Zaehlung nur in der eigenen Haelfte, MLP 96 / 8 Ep. auf Backbone+Logo+Audio, OOF,
Glaettung 9, Werbeklasse in Summe gleich gewichtet. Fehler / FP / FN / Median je Aufn. / logo-blind (36 Aufn.):
  Basis s0 0.0407/0.0282/0.0125/0.0152/0.212   s1 0.0399/.0267/.0132/.0142/.217   s2 0.0367/.0233/.0134/.0133/.177
  Abwerten (1+n)^-0.5: s0 0.0445, s1 0.0425 (FP steigt, 82/12 Aufn. schlechter); ^-1 noch schlechter -> Richtung falsch.
  Aufwerten (1+n)^+0.5: s0 0.0345/.0202/.0144/.0133/.169 (104 besser/34 schlechter), s1 0.0338/.0190/.0148/.0129/.165,
                        s2 0.0319/.0167/.0152/.0139/.138 (Median hier schlechter)
  Nur einmalige Werbung x0.5: s0 0.0373, s1 0.0360 (schwaecher, sauberer: 65/11).
  ABER Schwellen-Sweep s2 (Glaettung 9): bestes Fehlerniveau Basis 0.0322 (th 0.6) gegen 0.0311 (th 0.6), AUC 0.9648 -> 0.9653;
  (1+n)^+1: bestes 0.0305, AUC 0.9629, Median 0.0186 -> zu stark.
Lesart: der Grossteil des Proxy-Gewinns ist ein Pegel, den Platt in K wieder einebnet; uebrig bleibt ~3 % weniger
Sekundenfehler bei bester Schwelle und eine etwas andere Ordnung in logo-blinden Aufnahmen. Wenn der Arm verliert,
ist das der Beleg, dass die Werbe-Wiederholung als Gewicht (wie 010 als Merkmal) nichts Neues traegt.
Skripte: Scratchpad d1-d8.py (read-only, nur Summen).
