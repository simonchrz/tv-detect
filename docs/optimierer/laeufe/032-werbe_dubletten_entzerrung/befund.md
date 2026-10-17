Labels-Klasse gab auch im zehnten Anlauf nichts her (Decode-Stoerungen im Training sind selten und sauber gelabelt); in daten bleibt die Gegenrichtung zu 019: 019 hat im Messgeraet eine klare Steigung auf der Duplikat-Achse gemessen (Aufwerten -0.0039, 5/5 negativ), also wird hier abgewertet.

Klasse labels gezogen (Zug 0.63, labels hat 0 Laeufe), geprueft, read-only, nur Summen:
* Decode-Stoerung als Label-Frage (Hinweis K erfindet Phantombloecke bei gestoerten Quellen): isolierte Backbone-Spitzen
  (cos zu t-1 und t+1 < 0.3, cos(t-1,t+1) > 0.7) in 220 Trainingsaufn. (1-s-Raster, 727k Zeilen): 2051 Zeilen = 0.28 %,
  Werbeanteil 18.8 % gegen 23.8 % gesamt, Logo 0.68 -> Stoerungen liegen ueberwiegend in Sendung und sind als Sendung
  gelabelt; kein Label-Hebel. Spitzenrate je Aufn. q50 0.23 %, q99 1.7 % (vor allem prosieben = schnelle Schnitte, keine Stoerung).
* Fruehere neun Pruefungen (013, 019, 022-028, rand_einblendung_ziel, auto_blockende_logo): Labels sauber.
-> Ausweichen auf daten (zweithoechster Zug 0.45).

Duplikate (Trainingsteil saetze(), 527 Aufn. mit ads, jede 4. s, 506k Zeilen; Zaehlung je Haelfte nach sha1(uuid),
Backbone zentriert, Kosinus > 0.9 zu ANDEREN Aufnahmen): Werbezeilen mit 0 Treffern 47.9 %, 1-3 28.3 %, 4-20 18.8 %, > 20 4.9 %.
Proxy (MLP 96 / 8 Ep., Backbone+Logo+Audio, OOF Haelften, 2 Seeds, ohne Platt):
  basis       AUC 0.9535 / 0.9555, verpasst einmalige Werbung 14.1 / 14.8 %, haeufige 0.6 / 0.5 %
  (1+n)^-0.5  AUC 0.9521 / 0.9542, einmalig 15.0 / 10.8 %, haeufig 2.6 / 1.6 %
  (1+n)^-0.25 AUC 0.9536 / 0.9548, einmalig 13.3 / 11.6 %, haeufig 1.0 / 0.8 %
  Pegel (FP/FN) springt zwischen Seeds stark (ohne Platt), AUC-Unterschiede ~0.001 = Rauschen.
Lesart: der Proxy sieht diese Achse nicht (019: Proxy +, Messgeraet -0.0039). Einziger belastbarer Beleg ist die
Messgeraet-Steigung aus 019; ob K bei alpha=0 schon im Optimum sitzt (beide Richtungen schaden), entscheidet dieser Lauf.
Wenn er verliert: die Duplikat-Achse ist in beiden Richtungen tot.
Skripte: Scratchpad r1/g1.py (Stoerungen), r1/p1.py (Duplikate + Proxy).
