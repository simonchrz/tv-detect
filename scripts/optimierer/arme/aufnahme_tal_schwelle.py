"""Aufnahme-Tal-Schwelle: je Aufnahme die Entscheidungsschwelle ins Tal der
eigenen Logit-Verteilung legen (Otsu auf dem 32-s-geglaetteten Logit).

Der Kopf liegt je Aufnahme verschieden weit daneben (Filme, Kinder-Sender:
Sendung bei Logit ~0). Die Schwelle, die je Aufnahme am wenigsten Fehler
macht, streut im Trainingsteil out-of-fold von -0.75 bis +1.25 (IQR). Das
Tal zwischen den beiden Haeufungen der eigenen Aufnahme schaetzt sie ohne
Label (corr 0.42). Verschoben wird nur, wenn die beiden Haeufungen beidseits
von 0 liegen (sonst Aufnahme ohne klare Werbung: unveraendert), hoechstens
um 2 Logit. Platt danach wie in K (auch der Kalibrierteil wird verschoben).
Kein Zufall, keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Je Aufnahme Logit-Verschiebung auf das Otsu-Tal des 32-s-geglaetteten "
                "Logits (nur wenn beide Haeufungen beidseits 0 liegen, Kappe 2)")

FENSTER_S = 32
KAPPE = 2.0
GITTER = np.linspace(-4.0, 4.0, 81)


def _tal(l):
    """Otsu-Schwelle; Rueckgabe (gueltig, schwelle, mittel_unten, mittel_oben)."""
    v = np.sort(l)
    n = len(v)
    if n < 20:
        return False, 0.0, 0.0, 0.0
    cs = np.cumsum(v)
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in GITTER:
        k = int(np.searchsorted(v, t, side="right"))
        if k < n * 0.05 or n - k < n * 0.02:
            continue
        ma = cs[k - 1] / k
        mb = (cs[-1] - cs[k - 1]) / (n - k)
        s = k * (n - k) * (ma - mb) ** 2
        if s > best[0]:
            best = (s, float(t), float(ma), float(mb))
    return best[0] > 0, best[1], best[2], best[3]


def nachbearbeiten(p, rec, ktx):
    p = np.asarray(p, np.float64)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    k = max(1, int(round(FENSTER_S / max(1, int(ktx["schritt"])))))
    aus = z.copy()
    for i in np.unique(rec):
        m = rec == i
        zi = z[m]
        g = np.convolve(zi, np.ones(k) / k, mode="same")
        ok, t, ma, mb = _tal(g)
        if ok and ma < 0.0 < mb:
            aus[m] = zi - float(np.clip(t, -KAPPE, KAPPE))
    return 1.0 / (1.0 + np.exp(-aus))
