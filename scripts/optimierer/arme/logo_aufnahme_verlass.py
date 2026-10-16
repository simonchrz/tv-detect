"""Logo relativ zur eigenen Aufnahme: wie verlaesslich ist das Logo HIER?

Befund aus 006: der Kopf nutzt das Logo ausserhalb seiner Trainingsdaten zu
wenig. Diagnose auf dem Trainingsteil (609 Aufnahmen): wie gut das Logo
Werbung trennt, haengt stark von der Aufnahme ab und ist aus der
Logo-Verteilung DERSELBEN Aufnahme vorhersagbar. Fehler der Schwelle 0.5
je Aufnahme: Median 0.087, 90 %-Quantil 0.60; Korrelation mit dem
Logo-Median der Aufnahme -0.86, mit dem Anteil Logo > 0.8 -0.74, mit dem
Anteil im Zwischenbereich 0.2..0.8 +0.63. Die Rangordnung bleibt dabei meist
brauchbar (AUC je Aufnahme Median 0.926), nur die absolute Schwelle wandert.

Proxy (Trainingsteil, haelftig nach Aufnahme, jede 8. s, Gradient-Boosting
auf Backbone-PCA48 + logo + audio): Sekundenfehler 0.0785 -> 0.0602,
logloss 0.2185 -> 0.1815 mit diesen Spalten.

Je Aufnahme kommen 6 Spalten dazu (alle verteilungsbasiert, also bei
schritt=4 im Training und schritt=1 in Kalibrierung/Eval gleich verteilt;
nichts ueber Nachbarzeilen):

* logo_rang      Perzentil-Rang des Logo-Werts innerhalb der Aufnahme
* logo_hoch      Anteil der Sekunden mit Logo > 0.8 (Aufnahme-Konstante)
* logo_mitte     Anteil mit Logo in [0.2, 0.8] (Aufnahme-Konstante)
* logo_median    Median des Logos der Aufnahme (Aufnahme-Konstante)
* logo - Median, logo - 25 %-Quantil der Aufnahme

Benutzt nur die Logo-Spalte derselben Aufnahme, also was die Erkennung einer
fertigen Aufnahme auch hat. Kein Zufall in der Merkmalsbildung. Anders als
005 (zentrierte Backbone-/SigLIP-Spalten, 130 Stueck) nur 6 kleine Spalten
auf dem einen Merkmal, dessen Verlaesslichkeit nachweislich je Aufnahme
schwankt; anders als 004 kein Zeitfenster.
"""
import numpy as np

BESCHREIBUNG = ("Logo relativ zur Aufnahme: Rang, Anteil hoch/mitte, Median, "
                "Abstand zu Median und 25 %-Quantil (6 Spalten)")


def _rang(x):
    """Mittlerer Perzentil-Rang in (0, 1), Gleichstaende gemittelt."""
    n = len(x)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    r = np.empty(n, np.float64)
    grenzen = np.flatnonzero(np.diff(xs)) + 1
    starts = np.concatenate([[0], grenzen])
    enden = np.concatenate([grenzen, [n]])
    for a, b in zip(starts, enden):
        r[o[a:b]] = (a + b - 1) / 2.0
    return (r + 0.5) / n


def spalten(teil, X, ktx):
    namen = ktx["spalten_namen"]
    il = namen.index("logo")
    rec = np.asarray(ktx["rec"])
    zus = np.zeros((len(X), 6), np.float32)
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        lg = np.nan_to_num(np.asarray(X[m, il], np.float64), nan=0.5)
        med = float(np.median(lg))
        q25 = float(np.quantile(lg, 0.25))
        zus[m, 0] = _rang(lg)
        zus[m, 1] = float((lg > 0.8).mean())
        zus[m, 2] = float(((lg >= 0.2) & (lg <= 0.8)).mean())
        zus[m, 3] = med
        zus[m, 4] = lg - med
        zus[m, 5] = lg - q25
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
