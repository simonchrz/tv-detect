Der Block-IoU-Verlust sitzt an zu frueh endenden Bloecken (Schwanz Median 34 s, Kopf p 0.25, Logo schon zurueck); ein abklingender Werbe-Nachhall hinter sicheren Bloecken hebt im Produktions-HSMM-Proxy die IoU um +0.002 bis +0.004 bei nur +1.2 Pp Werbezeit.

Daten: 134 Trainingsaufnahmen mit altem Signal-Dump (~/.cache/tv-detect-daemon/emit-signals), K-Abbild trainiert auf dem Rest des
Trainingsteils (393 Aufn., jede 4. s, 3 Koepfe, Seeds 0/1), Vorhersage je Sekunde, Platt auf dem Kalibrierteil, Dekoder = Produktions-HSMM
(Emission ist reines NN, Proxy also exakt bis auf die Labels). Basis IoU 0.932/0.933, Median 0.984/0.988.
IoU-Verlust nach Art (4-s-Viterbi-Proxy, 609 Aufn.): Kante 74 %, verpasst 20 %, verschmolzen 7 %, geteilt 1 %.
Phantombloecke kosten die Block-IoU NICHTS, solange sie keinen wahren Block beruehren (block_iou zaehlt nur je Wahrheitsblock).
HSMM-Proxy: Kantenversatz Start Median 0 (q10/q90 -10/+13 s), Ende Median -1 (q10 -33, q90 +2) -> Enden schief, zu frueh.
  Zu frueh endend (>10 s): 57 Bloecke, 2640 s; Logo im Schwanz 0.75, p 0.25. Verpasste Bloecke: 12, p 0.04, Logo 0.83 (Konvention, nicht machbar).
Kontrollen (Seed 0 | 1, Δ IoU-Mittel, Median):
  Temperatur nach Platt x1.5..x6: -0.0008..+0.0008 (Dekoder-Deckel bestaetigt). Verschiebung ±2 s: -0.002..-0.006.
  Gleitmittel 3/7 s: ±0.0006. Unscharfmaske: scheinbar +0.007, aber NUR weil opt_lauf.platt divergiert (a ~ 1e11, harte
    0/1-Evidenz mit verschobener Schwelle); mit konvergiertem Platt (sklearn) -0.004..-0.008.
  Pauschal-Bias nach Platt +0.5 / +1.0 Logit: +0.0029|+0.0029 / +0.0042|+0.0053, Werbezeit +3.5 / +7.5 Pp -> die Metrik belohnt Zugaben.
Arm (W45, C3, exakt die Arm-Datei, opt_lauf.platt konvergiert a 0.98): +0.0041|+0.0023, Median +0.0021|+0.0013, besser/schlechter 29/16 |
  28/15, keine Aufnahme < -0.10, Werbezeit 1.012->1.025 | 1.006->1.017. W30/C2: +0.0042|+0.0020 (+0.5 Pp); W60/C4: +0.0046|+0.0050 (+2.5 Pp).
  -> je zusaetzlicher Werbesekunde rund dreimal wirksamer als der Pauschal-Bias, also nicht nur Pegel. Hash-Haelften: H0 -0.0006|+0.0018, H1 +0.0087|+0.0028.
WARNUNG Messgeraet: opt_lauf.platt (Newton ohne Daempfung, Start a=1,b=0) divergiert bei nicht-monotonen Vorverarbeitungen; der Arm
  konditioniert deshalb affin vor (A 0.63, B -1.10). Verdacht: die K-Instabilitaet im Schiedsrichter-Satz (Seeds 1/2) koennte ebenfalls
  eine Platt-Divergenz sein -- pruefen, indem man a,b je Seed protokolliert.
Skripte: Scratchpad e1.py (OOF je Sekunde), e2-e14.py (read-only, nur Summen).
