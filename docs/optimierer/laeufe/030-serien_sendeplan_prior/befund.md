Werbebloecke liegen bei derselben Sendung (Titel+Sender) Folge fuer Folge zur selben Uhrzeit: ein Sendeplan-Prior aus den anderen Trainingsfolgen trennt allein mit AUC 0.975 je Aufnahme und hebt im Proxy die Block-IoU in 2/2 Seeds um ~+0.006 (Mittel und Median).

Daten: Trainingsteil aus opt_lauf.saetze() (571 Aufn., 527 mit ads, jede 4. s, 506k Zeilen), nur Meta + Labels des Trainingsteils.
Abdeckung (Titel+Sender hat >= 1 andere Trainingsfolge): training 82 %, kalibrierung 91 %, entwicklung 88 % (nur Titel gezaehlt).
Prior allein (leave-one-out, 10-s-Faecher Ortszeit, Schrumpfung a=2 Folgen): AUC global 0.911-0.915, je Aufnahme Median 0.971 (roh) / 0.975 (+-30 s) / 0.979 (+-90 s), n=393.
Proxy: K-Merkmale via opt_lauf.lade+baue_k, K-MLP (3er-Logit-Mittel) 2 Haelften nach uuid-Hash out-of-fold, globale Platt,
  Dekoder-Ersatz = 2-Zustands-Viterbi Wechselstrafe 40 auf 4-s-Zeilen, Block-IoU wie opt_lauf.block_iou gegen Trainingslabel.
  Seed 0 Basis IoU 0.8990 / Median 0.9628:
    beta 0.25 ohne Versatz 0.9012/0.9686; beta 0.5 ohne Versatz 0.9007/0.9695 (9 Aufn. < -0.1)
    beta 0.5 Versatz +-600 s 0.9045/0.9712, besser 117 / schlechter 46 / < -0.1: 3
    beta 1.0 Versatz +-600 s 0.9043/0.9678 (89 schlechter) -> zu stark
  Seed 1 Basis 0.8994 / 0.9660: beta 0.5, Glaettung 30-120 s, Versatz 300-900 s alle 0.9044-0.9055 / 0.9720-0.9730, < -0.1: 1-3.
  Der label-freie Versatz (Korrelation Prior gegen Kopf-p) ist noetig: ohne ihn verdoppeln sich die Verlierer (verspaeteter Sendebeginn).
Teilmengen Seed 0 (beta 0.5, 60 s, 600 s): Menschlabel (merged/user, 466) 0.9063->0.9117, Median 0.9674->0.9731; auto (61) 0.8435->0.8496.
  Ohne Peers 0.7808 unveraendert (Prior = Grundrate), 1-4 Peers +0.008, >= 5 Peers +0.006.
  Sender Mittel-Delta: rtl +0.050 (32), vox +0.004, rtlzwei +0.004; disney-channel -0.009 (26), prosieben -0.001 (113).
Abgrenzung Friedhof: Minute-Prior (je Sender, Minute der Stunde, Eingabespalte) war inert, "Werbung haengt an der Sendung, nicht an der Uhr";
  hier haengt der Prior an Sendung UND Uhr und wirkt als Nachbearbeitung, nicht als Spalte. 023 serien_gedaechtnis nutzte Bildaehnlichkeit
  zu Serienfolgen, nicht die Zeitlage. Prior-Shift je Aufnahme (Anteil, abgelehnt auto_blockende_logo) ist eine Konstante, kein Zeitverlauf.
Risiko: Messsatz ist menschlich gelabelt, der Prior lernt aus gemischten Labels (Kantenkonvention auto ~18 s spaeter); Kinder-Sender leicht negativ.
Skripte (read-only): Scratchpad d1.py (Abdeckung, Prior-AUC), oof.py (OOF-Proxy), px.py/px2.py (Fusion, Teilmengen).
