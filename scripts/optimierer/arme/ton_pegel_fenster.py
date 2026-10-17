"""Ton-Pegel-Fenster: Abweichung des Tonpegels im 60-s-Fenster vom Aufnahme-Median.

Werbung ist lautheitsreguliert (R128): ein Werbeblock sitzt im Ton-Wert
eng am Normalpegel der Aufnahme, waehrend Sendungsstrecken (Musik, Stille,
Action, gestoerte Quelle) ueber eine Minute davon abweichen. K sieht den
Ton nur je Sekunde, wo sich Werbung und Sendung fast ganz ueberlappen
(Median 0.567 gegen 0.561); im 60-s-Median relativ zur Aufnahme trennt er.

Nachbearbeitung vor der Platt-Kalibrierung, auf dem rohen Ensemble-Logit z:

    A  = | Median(audio, 60 s um t) - Median(audio, ganze Aufnahme) |
    z' = z + C_A * A + C_AZ * max(z, 0) * A

Die Konstanten stammen aus einer logistischen Regression auf
out-of-fold-Logits eines K-gleichen Kopfs im Trainingsteil (Haelften nach
sha1(uuid), 2 Seeds, je Richtung stabil: C_A -17.7 .. -23.4,
C_AZ -3.2 .. -6.8). Platt wird danach auf der ebenso behandelten
Kalibrierung neu angepasst, ein globaler Versatz faellt also heraus.
Kein Zufall, keine Labels: nur die Ton-Spalte der jeweiligen Aufnahme.
"""
import numpy as np
from scipy.ndimage import median_filter

BESCHREIBUNG = ("Ton-Pegel-Fenster: Abweichung des 60-s-Ton-Medians vom Aufnahme-Median "
                "senkt den Werbe-Logit (Werbung ist lautheitsreguliert)")

FENSTER_S = 60.0
C_A = -20.0
C_AZ = -5.0

_TON = {}


def spalten(teil, X, ktx):
    i = ktx["spalten_namen"].index("audio")
    _TON[teil] = np.asarray(X[:, i], np.float64).copy()
    return X


def _abweichung(a, rec, schritt):
    w = max(1, int(round(FENSTER_S / max(1, schritt))))
    if w % 2 == 0:
        w += 1
    A = np.zeros(len(a), np.float64)
    for r in np.unique(rec):
        j = np.flatnonzero(rec == r)
        x = a[j]
        A[j] = np.abs(median_filter(x, size=min(w, max(1, len(x) | 1)), mode="nearest")
                      - np.median(x))
    return A


def nachbearbeiten(p, rec, ktx):
    a = _TON.get(ktx["teil"])
    if a is None or len(a) != len(p):
        return p
    A = _abweichung(a, np.asarray(rec), ktx["schritt"])
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    z2 = z + C_A * A + C_AZ * np.maximum(z, 0.0) * A
    return (1.0 / (1.0 + np.exp(-z2))).astype(np.asarray(p).dtype)
