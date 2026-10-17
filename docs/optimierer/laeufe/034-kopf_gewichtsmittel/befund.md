Klasse daten ist ausgeschoepft (drei fruehere Befunde + Vorgabe "keine Alters-/Aufnahmegewichte"), daher training (zweithoechster Zug 0.56): ein Parameter-EMA je Kopf senkt die Seed-Uneinigkeit um ~22-33 % und hebt im Proxy AUC und Logloss in 4/4 Seeds, ohne Merkmale, Gewichte, Ziel oder Kapazitaet zu aendern.

Daten: Trainingsteil saetze() (ohne O25), haelftig nach sha1(uuid) (A 256.6k / B 249.7k Zeilen, jede 4. s), nur Backbone+Logo+Audio (ohne OCR/SigLIP),
z-standardisiert auf A; Kopf = ol.trainiere mit HYPER_K (96, 12 Ep., Adam 1e-3), 4 Seeds je Variante, Vorhersage auf B, ohne Platt.
* Seed-Uneinigkeit |p_i - p_j| Einzelkopf: plain 0.0471, EMA 0.0366 (-22 %).
* Uneinigkeit zweier 2-Kopf-Ensembles (Logit-Mittel, +-8 Zeilen geglaettet): plain 0.0166, EMA 0.0111 (-33 %).
* Logloss je Seed: plain 0.2092/0.2141/0.2080/0.2118, EMA 0.1987/0.2002/0.1991/0.2045 (4/4 besser, -4.5 %).
* AUC je Seed: plain 0.9665/0.9667/0.9661/0.9657, EMA 0.9681/0.9678/0.9680/0.9674 (4/4, +0.0016).
* Fehler geglaettet >0.5: plain 0.0319/0.0335/0.0318/0.0343, EMA 0.0316/0.0326/0.0312/0.0332 (4/4, -2 %); FP-Anteil 0.0198 -> 0.0192.
Lesart: K endet nach 12 Epochen mit konstanter lr mitten im Adam-Rauschen; das Gewichtsmittel nimmt einen Teil der
Seed-Streuung heraus, die laut leer_aufnahmen_ersatz (Phantomlaeufe 3.7x Seed-Uneinigkeit) und dem Schiedsrichter-Durchgang
(K-Seeds 1/2 mit 1.3-1.8 h mehr Werbung) die Phantombloecke traegt. Erwartete Wirkung im Messgeraet: nn_erfindet sinkt, Kanten unveraendert.
Neuheit: kein EMA/SWA im Ledger, im Friedhof oder in train-head.py; nicht 003/012/028 (die aendern Eingaenge bzw. Schrittweiten), keine Ensemble-Groessen-Aenderung.
Risiko: Proxy ohne OCR/SigLIP und ohne Platt; fruehere Proxys (019, 014) sagten das Messgeraet schlecht voraus. Effekt ist klein (AUC +0.0016).
Skript: Scratchpad r3/ema.py (read-only, nur Summen).
