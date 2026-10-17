Lokale Bild-Wiederkehr (max. Kosinus zu den Sekunden 20..240 s vorher bzw. nachher) trennt Sendung ohne Logo und halbiert im Proxy die Fehler in logo-blinden Aufnahmen, waehrend die globale Selbstaehnlichkeit aus 011 im selben Proxy den Aufnahme-Median hebt.

Daten: Trainingsteil 527 Aufn. mit ads (jede 4. s, 506k Zeilen), Haelften nach sha1(uuid), MLP 96 / 8 Ep. auf Backbone+Logo+Audio
(ohne OCR/SigLIP), OOF beide Richtungen, Glaettung 9 Zeilen, Schwelle 0.5; 3 Seeds (0,1,2). Merkmal auf zentriertem Backbone,
Vorrat jede 4. Sekunde. Logo-blind = AUC(Logo-Abwesenheit gegen Label) < 0.75 je Aufn.: 36 Aufn., 7.5 % der Zeilen.
* Einzel-AUC (niedrige Wiederkehr -> Werbung): vor 0.792, nach 0.793, min 0.795, max 0.827. In den logo-blinden Aufn.:
  min 0.804, max 0.802 -- das Logo selbst dort nur 0.644. Neue Information genau da, wo das Logo versagt.
* Proxy Basis -> +Klammer (4 Spalten), Seeds 0/1/2:
  Fehler 0.0569/0.0452/0.0554 -> 0.0329/0.0305/0.0329; FP 0.045/0.029/0.044 -> 0.022/0.020/0.023; FN 0.012/0.017/0.012 -> 0.011/0.011/0.010
  logo-blind 0.254/0.166/0.245 -> 0.092/0.095/0.092; logo-stark 0.041/0.035/0.040 -> 0.028/0.025/0.028
  Median je Aufn. 0.0211/0.0185/0.0211 -> 0.0187/0.0171/0.0184; AUC gepoolt 0.980 -> 0.984
  Aufn. besser/schlechter um >0.005: 171/105, 186/108, 181/85; schlimmste Verschlechterung +0.07/+0.15/+0.07.
* Gegenprobe 011-Merkmal (global max/top5/Anteil>0.6, Abstand >= 90 s) im selben Proxy, Seeds 1/2:
  Fehler 0.0432/0.0391 (Basis 0.0452/0.0554), Median je Aufn. 0.0228/0.0219 -- STEIGT, wie 011 auf Entwicklung (Aufn.-Median -0.0021).
  Beide zusammen sind schlechter als Klammer allein (0.0328/0.0464). Die globale Suche findet in Werbung Treffer (Spot im naechsten
  Block) und verwischt das Signal; das lokale Fenster nicht.
Risiko: die Basis streut stark (Proxy-Basis ohne Ensemble, 8 Ep.); K hat HSMM mit Zeitkontext, der einen Teil des Proxy-Gewinns
schon abholt (011: Proxy gross, real +0.0036). Teleshopping/Dauerwerbung wirkt sendungsartig -- OCR/Logo muessen tragen.
Skripte: Scratchpad d1.py/d2.py (read-only, nur Summen).
