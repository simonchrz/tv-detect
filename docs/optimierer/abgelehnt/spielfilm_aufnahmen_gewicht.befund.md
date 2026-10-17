Spielfilme sind der schwerste Fehlerposten im Trainingsteil: 12 Film-Aufnahmen (4.6 % der Zeilen) tragen out-of-fold rund ein Viertel der Fehlerzeilen, fast alles erfundene Werbung; Gewicht 3 senkt ihren Fehler im Proxy um 30 % und laesst den Rest gleich.

Daten: Trainingsteil 527 Aufn. mit ads (jede 4. s, 506k Zeilen), Proxy-Kopf = K-MLP 96/12 Ep. auf Backbone+Logo+Audio (ohne OCR/SigLIP),
Haelften nach sha1(uuid), OOF-Glaettung 8 Zeilen, Schwelle 0.5; Proxy-Eval auf user/merged-Aufn. der anderen Haelfte, 2 Richtungen x 2 Seeds.
* Fehler je Aufnahme hat einen Film-Schwanz: Minions 0.69 (fp 0.69, Logo-AUC 0.87!), Lara Croft 0.47/0.33, Batman v Superman 0.29, Big 0.20,
  Predator 0.11. Film-Gruppe (5400-12000 s, Titelstamm <= 2x): Fehler 0.185, fp 0.174 gegen 0.035 / 0.022 sonst. Nur 3 der 12 sind
  logo-schwach (Ueberlappung mit 016 klein): das Problem ist Kinobild = Werbung/Trailer, nicht das Logo.
* Proxy, Mittel 4 Laeufe (base -> Film x0 -> Film x3):
  Nicht-Film-Fehler 0.02772 -> 0.02794 -> 0.02729 (x3 je Lauf +0.0002/-0.0002/-0.0004/-0.0014), Median 0.01405 -> 0.01385 -> 0.01386,
  AUC 0.9729 -> 0.9728 -> 0.9723; Film-Fehler 0.177 -> 0.448 -> 0.124. Filme herausnehmen zerstoert sie, dreifach gewichten traegt.
* Entwicklungssatz (nur Laengen/Titel gezaehlt): etwa 1 von 76 Aufnahmen ist filmartig -> Regel nicht_schlechter; der Nutzen liegt bei
  Filmaufnahmen der Produktion, die der Messsatz kaum enthaelt.
* Verworfen vor dem Arm (Klasse daten):
  - Randzonen-Konvention: Trainings-Aufnahmen beginnen nur zu 2.5 % mit Werbung ab Sekunde 0 (Entwicklung 6.6 %), Enden in auto-Labels
    zu 23 % (merged 6 %, Entwicklung 4 %); OOF zeigt etwa so viele unmarkierte wie markierte Rand-Werbebloecke (Start 14/13, Ende 27/42).
    Sendung-Zeilen in den ersten/letzten 120 s auf Gewicht 0: Fehler +0.0021 (4/4 schlechter), auch in der Mitte +0.0012; 300 s: +0.006. Tot.
  - tvh-Hash-uuids (65, ohne slug/start_ts): zeilengewichtet Fehler 0.071 gegen 0.035, aber Median 0.013 (besser als dvr) -> Schwanz
    aus Filmen (Minions, Lara Croft), kein Gruppeneffekt.
  - Fehler nach Sender: leerer slug 28 % der Fehler (Filme, Richter Hold), rtl 19 % (Let's Dance/Beet-Brueder, schon 016), prosieben 19 %.
Risiko: Trailer sind ebenfalls Kinobild; Film x3 kann Trailer-Erkennung schwaechen (AUC -0.0006 im Proxy). Fehlerart nn_verpasst beobachten.
Skripte: Scratchpad p_pos.py, p_dev.py, p_rand.py, p_lauf.py, p_st.py, p_arm.py, p_slug.py, p_hash.py, p_film.py, p_key.py (read-only, Summen).

## Abgelehnt

ABLEHNUNG wegen Regeltyp (Pruefpunkt 5): opt_urteil.py definiert nicht_schlechter ausdruecklich nur fuer Vereinfachungen (Median >= -0.002, hoechstens 1 Wiederholung unter -0.005). Dieser Arm ist keine Vereinfachung, er fuegt eine Gewichtung hinzu und behauptet einen Nutzen (Filmfehler 0.177 -> 0.124). Der bisher einzige nicht_schlechter-Arm war 014 (O25, Aufnahmen entfernen = Vereinfachung). Die Begruendung raeumt selbst ein, dass der Entwicklungssatz nur etwa 1 von 76 filmartige Aufnahmen enthaelt. Der Nutzen ist damit im Messsatz nicht messbar. Die gelockerte Regel liesse den Arm als 'Kandidat' durch, ohne dass irgendein Messgeraet-Ergebnis den Nutzen belegt. Der einzige Beleg waere der Trainings-Proxy (OOF, ohne OCR/SigLIP, anderer Kopf), kein Opt-Lauf. Das umgeht die feste Gewinnschwelle (+0.003).

Auf 'gewinn' umgestellt wuerde der Arm nach Konstruktion scheitern, weil der Messsatz kaum Filme hat. Der Arm braucht zuerst einen Messsatz oder eine Teilmengen-Messung mit Filmen. Beides geht nicht, ohne die fest vorgegebenen Saetze anzufassen. Als Arm in dieser Form also nicht pruefbar und nicht zulaessig.

Risiko, das der Messsatz nicht abdeckt: Trailer sind ebenfalls Kinobild und bereits die bekannte NN-Schwaeche (Kopf 0.586, §3aa). Die Befunddatei nennt AUC -0.0006 im Proxy und 'Fehlerart nn_verpasst beobachten'. Bei einem Arm mit relaxierter Schwelle faellt genau diese Verschlechterung ('Film x3 schwaecht Trailer-Erkennung') nicht auf.
