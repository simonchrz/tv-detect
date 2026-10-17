"""Weiche Kappe auf die Kopf-Logits vor der Kalibrierung.

Der HSMM summiert je Segment die Log-Evidenz der Sekunden. Sehr sichere
Sekunden (|logit| weit ueber dem Ueblichen, oft auswendig gelerntes
Bildmaterial) ueberstimmen dabei lange Strecken maessiger Gegen-Evidenz.
Die Kappe z' = c * tanh(z / c) staucht nur die Extreme; die anschliessende
Platt-Kalibrierung streckt den mittleren Bereich wieder. c = Median von
|logit| auf dem Kalibrierteil derselben Wiederholung (der Kalibrierteil
wird vor dem Eval-Teil bearbeitet), damit Kalibrierung und Eval dieselbe
Abbildung sehen. Kein Zufall, keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Weiche Logit-Kappe c*tanh(z/c), c = Median |logit| des Kalibrierteils, "
                "vor Platt: extreme Sekunden ueberstimmen den Dekoder weniger")

_zustand = {"c": None}


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def nachbearbeiten(p, rec, ktx):
    z = _logit(p)
    if ktx.get("teil") == "kalibrierung" or _zustand["c"] is None:
        c = float(np.median(np.abs(z)))
        _zustand["c"] = max(c, 0.5)
    c = _zustand["c"]
    zc = c * np.tanh(z / c)
    return 1.0 / (1.0 + np.exp(-zc))
