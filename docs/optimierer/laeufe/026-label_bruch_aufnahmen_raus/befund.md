9 Trainingsaufnahmen in K haben einen nachweislich gebrochenen Label-Zustand (2 Zeitachsen-Versatz, die die Produktion quarantaeniert, K aber traegt; 7 weitere Maschinenlabels mit Werbebloecken unter durchgehend sichtbarem Logo); der Proxy sieht ihr Herausnehmen als neutral, das Messgeraet soll es entscheiden.

Daten: Trainingsteil von K (saetze()/lade() aus opt_lauf, jede 4. s, 527 Aufn. mit ads, 506k Zeilen), Logo-Spalte roh.
* Zeitachsen-Versatz (zeitachsen-versatz.json, Schwelle 15 s): dvr-rtl-1779473700 +1269 s, dvr-rtl-1781199600 +40 s liegen im
  Trainingsteil; Kalibrierung/Entwicklung/Bestaetigung frei. train-head prueft die Quarantaene nur im Live-Durchgang
  (Z. 3608), nicht bei archiv-eingespeisten Aufnahmen -> moegliche Produktionsluecke, nicht geprueft (kein Netz).
* Werbebloecke >= 60 s mit Logo-Mittel > 0.6: auto 16 von 168 (9.5 %, 8 Aufn.), menschlich 12 von 911 (1.3 %, meist
  Dauerwerbung/Sponsor). Schwelle 0.5/0.7 und 30 s statt 60 s aendern das kaum (15-18 Bloecke). auto: Beet-Brueder x4,
  Bella Italia x2, Let's Dance (= die versetzte), Mein Lokal. OOF-Fehler dort fn 0.13-0.30 bei fp ~0 (Label Werbung, Kopf Sendung).
* Proxy (Haelften nach sha1(uuid), K-MLP 96/12 Ep. auf Backbone+Logo+Audio, Glaettung 5 Zeilen, Eval merged/user der anderen
  Haelfte, 2 Seeds): Fehler-Mittel 0.0389 -> 0.0387, Median 0.0173 -> 0.0175, AUC 0.9716 -> 0.9715; je Lauf +0.0008/-0.0008/
  +0.0001/-0.0010. rtl+rtlzwei 0.0566 -> 0.0549. Proxy war bei Datenfragen schon daneben (014), daher kein Gegenbeleg.
Verworfen vor dem Arm (Klasse daten, alles Trainingsteil):
* Produktions-Altersrampe (1 -> 0.5 bis 90 d -> 0 bei 180 d, Stichtag 17.10.): Proxy Fehler alte Aufn. (>= 100 d) 0.068 -> 0.076,
  neue (< 100 d) 0.0251 -> 0.0255 -- kostet die alten, hilft den neuen nicht. Entwicklung/Bestaetigung sind alt (Median 139/137 d),
  der Arm waere hier ohnehin strukturell benachteiligt. Hinweis an Simon: die Rampe ist im Proxy ohne Nutzen.
* Herkunft x2 (Produktion: user/merged ausser mensch_belegt=False): Proxy neutral (0.0389 -> 0.0391).
* Fehler steigt stark mit dem Alter (Median je Aufn. < 45 d 0.008, >= 120 d 0.038): vor allem Sendermix Mai/Juni
  (disney, nick, comedy-central, Let's Dance) und logo-blinde Sendungen (21 von 151 alten mit Logo-AUC < 0.75, neu 3-5) --
  kein technischer Regimebruch (Logo-Median der Sendung je Halbmonat 0.87-0.97). Der Messsatz liegt in diesem alten Teil.
* Label-Versatz je Aufnahme (Logo-Wechsel vs. Labelkante, +-30 s): Median 0, nur 3 von 308 konsistent >= 4 s verschoben. Tot.
* Eingefrorene Backbone-Zeilen 309 von 2.0M, Logo-NaN 964; Werbeanteil plausibel (q5 0.14, q95 0.32): kein Zustandsdefekt.
Abgrenzung: auto_rest_ausschluss (09:46 entworfen, nie gemessen) nimmt alle 61 auto-Aufn. (13.6 %) nach Herkunft; dieser Arm
nur 9 nach belegtem Zustand und behaelt 79 % der auto-Zeilen (Backbone-Vielfalt vox/prosieben/kabel-eins).
Skripte: Scratchpad a1-a6.py, p1-p9.py (read-only, nur Summen).
