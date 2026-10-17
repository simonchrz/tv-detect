Verlustfunktionen verschieben im Trainingsteil vor allem den Arbeitspunkt; nur Focal (gamma=1) senkt nach kreuzgefitteter Platt-Kalibrierung die Fehler und die langen Fehlstrecken in allen Seeds, GCE (rauschfest) bricht die Kalibrierung.

Messung: Trainingsteil (609 Aufn., schritt 4, K-Merkmale inkl. OCR+SigLIP), 2 Falten nach Aufnahme, 1 Kopf, Verlust per Gradienten-Hook im modell() (CE-Gradient je Zeile mal Faktor, Batch-normiert; gegen expliziten Verlust geprueft, Abweichung 5e-7).
Roh (Logit>0, 5 Zeilen = 20 s geglaettet), Fehlerrate, je 2 Seeds:
  CE 0.0414/0.0396/0.0409/0.0397/0.0408/0.0396 | Focal g1 0.0382/0.0381/0.0376/0.0375/0.0397/0.0386
  Focal g0.5 0.0387/0.0379, g2 0.0385/0.0374 | GCE q0.3/0.5/0.7 ~CE, Aufnahme-Median schlechter (0.0159-0.0184 gegen 0.0155)
  FP sinkt (0.029 -> 0.025), FN steigt leicht (0.012 -> 0.0125): teils Verschiebung.
Bester fester Schnitt (Schwelle frei, ohne Kalibrierung): CE 0.0319/0.0315, Focal 0.0326/0.0316 -> kein Gewinn an Trennschaerfe.
Architektur (gleiches Protokoll, bester Schnitt): K 0.0322/0.0320, 2 Schichten 0.0323/0.0330, LayerNorm 0.0323/0.0332, 2 Schichten+LN 0.0327/0.0327 -> Rauschen; Breite/Tiefe/Norm geben nichts her.
Produktionsnah (Platt auf der jeweils anderen Haelfte gefittet, dann 9 Zeilen = 36 s geglaettet):
  Fehlerrate CE 0.0316/0.0316, Focal g1 0.0299/0.0301, g2 0.0302/0.0304
  falsche Strecken >= 32 s (das, was der HSMM zu Bloecken macht): Zeilen CE 10686/10640, Focal g1 9689/9980 (-8 %), Anzahl gleich (~535)
  NLL CE 0.165, Focal 0.170 (erwartet, Focal macht weichere Logits); sicher falsch (p>0.9) 1.53 % -> 1.51 %
  GCE q0.5: Platt laeuft ueber, Fehler 0.10-0.14 -> verworfen.
Deutung: Focal veraendert die FORM der Logit-Verteilung (leichte Zeilen weniger extrem), sodass Platt mit 2 Parametern einen besseren Arbeitspunkt fuer die geglaettete Entscheidung findet; die Trennschaerfe selbst bleibt (Merkmals-Decke). Erwartung daher klein.
Klasse: Bandit zog labels; nach Sitzungsvorgabe nur Verlust/Architektur, labels-Arme sind laut fuenf frueheren Befunden ausgeschoepft -> training.

## Abgelehnt

18.10. Prüfer: nicht neu. Focal (g2) wurde schon im Befund von 029-segment_mixup gemessen ("vor Platt 7x weniger sichere Fehler, nach Platt weg"); g1 ist eine Parametervariante. Außerdem eine globale Gewichtsvariante, die das README ab Durchgang 2 ausschließt (jeder Arm greift eine BENANNTE Schwäche mit neuer Information an), und die erwartete Wirkung im Proxy (~0.0017) liegt unter der Gewinnschwelle. Hinweis der Sitzung hatte "Verlustfunktion/Architektur" als offen bezeichnet -- das war falsch, beides ist durch 029 und die README-Vorgabe erledigt.
