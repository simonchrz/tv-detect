"""Schnitt-Zonen-Buendelung: in unsicheren Zonen den Logit je Einstellung mitteln.

Werbung und Sendung sind verschiedenes Bildmaterial, ihre Grenze liegt auf
einem Bildschnitt (im Trainingsteil: Backbone-Kosinus zur Vorsekunde an
Label-Kanten Median 0.38, sonst 0.63). Der Kopf urteilt aber je Sekunde und
verschmiert den Uebergang ueber mehrere Sekunden; der HSMM legt die Kante
dann irgendwo in die Rampe. Kanten tragen 51 % der Verlust-Sekunden.

Nachbearbeitung vor Platt, je Aufnahme, nur in Kalibrierung und Eval
(beide jede Sekunde geladen):
  * Einstellungen: neue Einstellung, wo der Kosinus des aufnahme-zentrierten
    Backbones zur Vorzeile unter TAU liegt.
  * Unsichere Zone: |gleitendes 15-s-Mittel des Logits| < BAND.
  * Dort z' = (1 - A) * z + A * Mittel(z ueber die ganze Einstellung).
Ausserhalb der Zone bleibt z unveraendert. Abgrenzung: 015 nutzte die
Schnittdichte als Spalte, 047 mittelte ueber bildaehnliche FERNE Sekunden;
hier wird nur innerhalb zusammenhaengender Einstellungen und nur im
Uebergangsbereich gebuendelt. Kein Zufall, keine Labels.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

BESCHREIBUNG = ("Schnitt-Zonen-Buendelung: in unsicheren Zonen (|15-s-Logitmittel| < 5) "
                "den Logit zu 70 % durch das Mittel seiner Einstellung (Backbone-Schnitt, cos < 0.4) ersetzen")

TAU = 0.4
BAND = 5.0
A = 0.7
FENSTER_S = 15

_COS = {}


def _kosinus_vorzeile(B, rec):
    cos = np.ones(len(B), np.float64)
    for r in np.unique(rec):
        j = np.flatnonzero(rec == r)
        if len(j) < 2:
            continue
        M = np.asarray(B[j], np.float32)
        M = M - M.mean(0, keepdims=True)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-6)
        cos[j[1:]] = np.einsum("ij,ij->i", M[1:], M[:-1])
    return cos


def spalten(teil, X, ktx):
    if teil != "training":
        n = ktx["spalten_namen"]
        a, b = n.index("backbone0"), n.index("backbone1279") + 1
        _COS[teil] = _kosinus_vorzeile(X[:, a:b], np.asarray(ktx["rec"]))
    return X


def nachbearbeiten(p, rec, ktx):
    cos = _COS.get(ktx["teil"])
    rec = np.asarray(rec)
    if cos is None or len(cos) != len(p):
        return p
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    w = max(1, int(round(FENSTER_S / max(1, int(ktx["schritt"])))))
    aus = z.copy()
    for r in np.unique(rec):
        j = np.flatnonzero(rec == r)
        zi = z[j]
        neu = cos[j] < TAU
        neu[0] = True
        seg = np.cumsum(neu) - 1
        mittel = np.bincount(seg, zi) / np.bincount(seg)
        zs = uniform_filter1d(zi, w, mode="nearest")
        zone = np.abs(zs) < BAND
        aus[j] = np.where(zone, (1 - A) * zi + A * mittel[seg], zi)
    return (1.0 / (1.0 + np.exp(-aus))).astype(np.asarray(p).dtype)
