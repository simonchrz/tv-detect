Whisper-Text-Wiederkehr als fester Logit-Zuschlag NACH dem Ensemble (statt als Spalte wie 040) senkt im out-of-fold-Proxy die geglaetteten Sekundenfehler in beiden Haelften; Labels-Klasse gab erneut nichts her.

Klasse labels gezogen (Zug 0.70, 0 Laeufe). Label-Check mit Whisper (117 Trainingsaufn. mit Transkript, 2-s-Raster, 217k Zeilen):
* Sendung mit Wiederkehr > 0.2 und Logo < 0.3: 1152 s (0.26 %), 584 s davon <= 60 s an einer Kante, 26 Aufn.; > 0.5: 284 s in 5 Aufn.
  Werbung mit Wiederkehr > 0.2: 48404 s. -> keine verpassten Spots/Hinweise in den Labels, kein Label-Hebel (wie 032/040).
-> nachbearbeitung (zweithoechster Zug 0.44). Richtung (1) Stoerung greift auf Eval nicht (039/040 gemessen).

Proxy (Scratchpad w3/d1-d3.py): opt_lauf.saetze/lade/baue_k, Trainingsteil schritt 8 ohne die Test-Haelfte;
Test = Whisper-Aufnahmen einer sha1-Haelfte (schritt 2); K-MLP (HYPER_K) 3 Koepfe Logit-Mittel; Korpus = Whisper der Train-Haelfte.
1-D-Platt auf der anderen Haelfte, Fehler je Aufnahme gemittelt/Median, roh und 15-Zeilen-geglaettet (~30 s, naeher am HSMM):
  C=0  Haelfte 0: 0.0359/0.0291 glatt 0.0191/0.0102 | Haelfte 1: 0.0394/0.0151 glatt 0.0219/0.0067
  C=3            0.0326/0.0257       0.0186/0.0097 |            0.0359/0.0148       0.0200/0.0062
  C=6            0.0311/0.0227       0.0186/0.0098 |            0.0335/0.0133       0.0186/0.0066
  C=10           0.0300/0.0234       0.0191/0.0107 |            0.0311/0.0128       0.0178/0.0061
  C>=15 kippt Haelfte 0 (glatt 0.0204-0.0226). Max- statt Mittel-Wiederkehr gleichwertig, Knick bei 0.1 schlechter.
Freie Logistik (z, W, W*z+, W*z-): AUC 0.9923->0.9931 / 0.9845->0.9874, Logloss 0.097->0.080 / 0.145->0.119; Koeffizient W ~24 Logit (roh-Einheiten), zu stark fuer die Glaettung.
Wahl C=5: im Eval ist der Korpus doppelt so gross (ganzer Trainingsteil), W dort ~1.5x hoeher (040: Entwicklung Mittel 0.091 gegen Proxy 0.058) -> effektiv ~C 7.5.
Greift auf Eval: Entwicklung 76/76 Aufnahmen mit Zuschlag, Mittel +0.33 Logit, 12.6 % der Sekunden > +1 Logit (Summen, Labels nicht gelesen).
Warum nicht 040 wiederholt: dort musste K die Spalte aus 22 % bespurten Trainingszeilen lernen (78 % = 0, nicht von "keine Wiederkehr" zu trennen); hier kein Lernen, feste Staerke.
Risiken: Kanten (60-s-Fenster grob; Proxy glatt meist besser), Programmhinweise auf die eigene Sendung in fremden Titeln (Sendung > 0.2 nur 1.3 %), super-rtl-Ausreisser wie 039/040.
