Trainingslabels lassen Werbebloecke hinter der Logo-Rueckkehr laenger weiterlaufen als die Menschenlabels der Entwicklung (Mittel 8.6/11.5 s gegen 5.0 s je Blockende); sonst sind die Labels auch im elften Anlauf sauber.

Klasse labels gezogen (Zug 0.67, labels hat 0 Laeufe). Daten: saetze() ohne O25, Logo-Spalte aus den Feature-npy (1-s-Raster),
nur Aufnahmen mit trennendem Logo (Median Sendung >= 0.6, Werbung <= 0.3), Bloecke >= 60 s, nicht am Aufnahmeende. Read-only, nur Summen.
Zwei neue Label-Pruefungen:
* Verpasste Bloecke (Sendung-Laeufe >= 60 s mit 30-s-Logo-Mittel < 0.15, Raender 120 s aus): Training merged 6 Laeufe/1608 s (0.14 %
  der Sendung), auto 3/671 s (0.4 %), Entwicklung 1/76 s. Zu klein -> kein Arm.
* Logo-Schwanz am Blockende (Sekunden vor Blockende mit 8-s-Logo-Mittel >= 0.6, zusammenhaengend bis zur Kante):
  Training merged 393 Aufn./773 Enden: Mittel 8.6 s, q90 24, Anteil >= 12 s 0.213; auto 48/123: 11.5 s, q90 29, 0.195;
  Entwicklung (Mensch) 52/92: 5.0 s, q90 19, 0.152.
  Gegenprobe Blockstart (Logo noch da am Anfang): merged 12.7, auto 16.6, Entwicklung 11.4 s -> Starts gleich, nur die Enden laufen nach.
  (auto_blockende_logo fand fuer auto -24 gegen merged -5 s mit OOF-Kippen; hier mit Logo allein ist der Abstand kleiner, betrifft aber
  merged genauso.)
Arm: am Blockende Zeilen ab Logo-Rueckkehr + 8 s -> Sendung. Trifft im Trainingsteil (jede 4. s) 1266 Zeilen = 1.05 % der Werbezeilen,
126 Aufnahmen (merged 1018, auto 248).
Abgrenzung Friedhof: auto_blockende_logo (abgelehnt vor Messung) war nur auto, Logo + 6 s, Proxy 2-Zustands-Viterbi neutral; hier alle
Herkunften, gemessen wird erstmals im Messgeraet mit Produktions-HSMM. 026/033 nahmen Aufnahmen heraus, hier bleiben alle Zeilen.
Risiko: echte Werbung mit Logo am Blockende (Sponsor, Trailer) wird Sendung -> dekoder/nn_verpasst an Enden kann steigen.
Wenn verworfen: Labels-Klasse hat dann einen echten Lauf; Blockende-Konvention ist als Hebel erledigt.
Skripte: Scratchpad r3/d1.py, d1b.py, d2.py, d3.py, t.py.

## Abgelehnt

ABLEHNUNG, Hauptgrund: Der Arm hat keine Obergrenze für die Kürzung und macht nicht das, was die Hypothese behauptet. Ich habe ziel() read-only auf den echten Trainingsteil angewendet (527 Aufnahmen, jede 4. s, nur Logo-Spalte und Ads-Labels aus den Feature-npy). Die 1266 geänderten Zeilen (1.05 %, 126 Aufnahmen) stimmen mit den Angaben überein. Die 193 Kürzungen sind aber nicht die behaupteten "Nachlauf-Sekunden". Median 12 s, q90 60 s, Maximum 236 s. 21 Kürzungen sind länger als 60 s, 6 länger als 120 s. Diese 21 tragen 552 von 1266 Zeilen (44 %). Die Diagnose nennt für den Logo-Schwanz q90 24 s (merged) und q90 19 s bei den Menschenlabels. Die while-Schleife in ziel() läuft zusammenhängend rückwärts, solange das Logo-Mittel >= 0.6 ist. Hat ein Block hinten ein langes Stück mit Logo (Sponsor, Senderhinweis, Trailer, Eigenwerbung), wird bis auf 8 s fast der ganze Block zu Sendung (t kann bis a laufen). Das gleicht keine Konvention an. Es schreibt Werbung in Sendung um, an einer Stelle, an der das Logo kein Sendungsbeweis ist. Das ist die Art Labelbruch, die 026 und 033 nur anders gelöst haben, und die Begründung "alle Zeilen bleiben erhalten" trifft hier nicht zu. Ein Deckel (z. B. nur Schwänze <= 20-30 s anfassen oder die Kürzung auf q90 der Menschenlabels begrenzen) fehlt.

Konvention: Ledger §3y und die Rand-Trailer-Funde (experiment-ledger.md um Zeile 1571) sagen, dass Rand-Trailer am Blockende zur Werbung gehören, und diese Trailer tragen oft das Logo.
