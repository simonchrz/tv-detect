"""Daten: Trainingsaufnahmen mit verschobener Label-Zeitachse herausnehmen.

Fehlerposten: Kante (51 % der Verlust-Sekunden). Die Produktion laesst
Aufnahmen aus, deren Label- und Merkmals-Zeitachse auseinanderlaufen
(train-head.zeitachsen_quarantaene, Liste aus zeitachsen-check.py), kennt
aber nur die wenigen Faelle, die das Gateway geprueft hat. Verschobene
Labels sehen nicht kaputt aus: die Bloecke haben die richtige Laenge und
sitzen nur an der falschen Stelle -- der Kopf lernt dort jede Kante falsch.

Befund auf dem Trainingsteil (527 Aufn. mit ads): je Aufnahme die globale
Verschiebung s (-60..+60 s), die die Labels am besten erklaert, aus ZWEI
unabhaengigen Quellen:
  (a) out-of-fold-Wahrscheinlichkeit eines K-gleichen Kopfs (Haelften nach
      sha1(uuid), 3 Koepfe), Log-Likelihood der verschobenen Labels,
  (b) roher Logo-Wert je Sekunde (nur Aufnahmen mit trennendem Logo,
      Sendung-minus-Werbung >= 0.4), L1-Abstand Labels zu 1-Logo.
Beide stimmen (gleiches Vorzeichen, |sa-sb| <= 12 s, |sa| >= 12 s) bei 35
Aufnahmen ueberein -- bei Zufallspaarung waeren ~11 zu erwarten, bei
|s| >= 24 s 16 gegen 2.2. Eine reine Konventionsfrage (Bloecke zu lang oder
zu kurz) verschiebt Anfang und Ende gegenlaeufig und erzeugt keine globale
Verschiebung; eine verschobene Zeitachse dagegen genau diese.
Dazu dvr-rtl-1779473700 (Produktions-Quarantaene, 1269 s Versatz, K traegt
sie noch); dvr-rtl-1781199600 (Quarantaene, 40 s) fand der Detektor selbst.

Regel: Gewicht 0 fuer alle Zeilen dieser Aufnahmen (5 % der Zeilen), sonst
1. Kalibrierung und Eval unberuehrt. Kein Zufall, keine Eval-Information.
"""
import numpy as np

BESCHREIBUNG = ("Daten: 36 Trainingsaufnahmen mit verschobener Label-Zeitachse "
                "(OOF-Kopf und Logo einig, 12-60 s) aus dem Fit nehmen")

VERSETZT = frozenset([
    "af2a267d7c55c5f5ac76647bc25a2ee4", "dvr-disney-channel-1781129100",
    "dvr-kabel-eins-1783609090", "dvr-kabel-eins-1783954727",
    "dvr-kabel-eins-1784217368", "dvr-kabel-eins-1784645879",
    "dvr-kabel-eins-1785337108", "dvr-nick-1778759100", "dvr-nick-1778817300",
    "dvr-nick-1778841000", "dvr-prosieben-1778679851", "dvr-prosieben-1778766521",
    "dvr-prosieben-1779198195", "dvr-prosieben-1780987800",
    "dvr-prosieben-1781013874", "dvr-prosieben-1781186670",
    "dvr-prosieben-1781261193", "dvr-prosieben-1781272923",
    "dvr-prosieben-1781342478", "dvr-prosieben-1781344194",
    "dvr-prosieben-1781533961", "dvr-prosieben-1785171917",
    "dvr-prosieben-1786986302", "dvr-rtl-1781199600", "dvr-rtl-1783863900",
    "dvr-rtlzwei-1784369700", "dvr-vox-1778688000", "dvr-vox-1780675200",
    "dvr-vox-1780840800", "dvr-vox-1781020800", "dvr-vox-1781473200",
    "dvr-vox-1783267800", "dvr-vox-1784217600", "e0c9426d85770fadc9e535f557c77f84",
    "ee05c8b4367b0601a716d7b119b0ac09",
    # Produktions-Quarantaene (zeitachsen-versatz), von K noch trainiert
    "dvr-rtl-1779473700",
])


def gewichte(ktx):
    raus = np.array([u in VERSETZT for u in ktx["uuids"]], bool)
    rec = np.asarray(ktx["rec"])
    return np.where(raus[rec], 0.0, 1.0).astype(np.float32)
